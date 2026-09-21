#!/usr/bin/env python3
"""Review test 3 (2026-09-21 ruling): the frozen Stage-2 host (GPU).

Three checks, run ONCE on the training server:

  1. trainable params == Gamma ONLY (every requires_grad name contains
     "gamma"; the full 36-layer x {V, U} set is present);
  2. step-0 logits are IDENTICAL to the CCM-merge host — the zero-init
     Gamma (U = 0) adds an exact zero residual at every layer INCLUDING
     t = 1 (M_0 = 0), so the forward is bit-identical to the merge
     checkpoint;
  3. the Gamma scan calls one residual per compression layer: for a
     L = 3 sample, every layer's gamma receives t = 1, 2, 3 exactly
     once (L = 1 likewise gets t = 1).

Usage (on the server, qwen3 environment):
  python -u scripts/test_review_3_frozen_host.py \
      --model-name-or-path /root/autodl-tmp/qwen3-4b-instruct \
      --dialog-mirror /root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog \
      --init-from /root/autodl-tmp/outputs/ccm_qwen3/seed0_qwen3_merge_e1000/checkpoint_step550.pt \
      --gpu 0
"""
import argparse
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import train_ccm as tc


def gamma_layers(model):
    base = model
    while not hasattr(base, "layers") and hasattr(base, "model"):
        base = base.model
    return [layer.self_attn for layer in base.layers]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name-or-path", required=True)
    ap.add_argument("--dialog-mirror", required=True)
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    device = torch.device("cuda", a.gpu)
    args = types.SimpleNamespace(
        arm="ours", host="qwen3",
        model_name_or_path=a.model_name_or_path,
        dialog_mirror=a.dialog_mirror,
        relative_embedding="skip", lora_r=8, gamma_hidden=64,
        z_dim=128, rpbe_seed=0, sketch_dim=32,
        official_host=False, foundation="", official_adapter="",
        init_from=a.init_from, freeze_host=True, micro_batch=1)

    tokenizer = tc.build_tokenizer(args)
    model = tc.build_model(args, device)
    # Mirror the training build order: build -> wrap_lora ->
    # update_comp_token -> attach_gamma -> init-from -> freeze.
    model = tc.wrap_lora(model, args.lora_r)
    model.update_comp_token(
        [tokenizer.comp_token_id[k] for k in range(tc.N_TOK)],
        [tokenizer.sum_token_id[k] for k in range(tc.N_TOK)])
    model.base_model.model.model.embed_tokens \
        .comp_embeddings.weight.requires_grad_(True)
    tc.attach_gamma(model, hidden=args.gamma_hidden)
    # Mirror the training init-from block (the merge ckpt has no Gamma
    # keys; gamma must stay zero-init — load_trainable would refuse).
    _payload = torch.load(a.init_from, map_location=device,
                          weights_only=False)
    _missing, _unexpected = model.load_state_dict(_payload["model"],
                                                  strict=False)
    assert not _unexpected, sorted(_unexpected)[:5]
    _trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    _bad = [k for k in sorted(set(_missing) & _trainable)
            if "gamma" not in k]
    assert not _bad, _bad[:5]
    # mirror the training freeze-host block
    for n, p in model.named_parameters():
        if "gamma" not in n and p.requires_grad:
            p.requires_grad_(False)
    model.eval()

    # ---- check 1: trainable == Gamma only ----------------------------
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n in trainable if "gamma" not in n]
    assert not bad, "non-Gamma trainable params: {}".format(bad[:5])
    n_gamma = len(trainable)
    assert n_gamma == 36 * 3, (n_gamma, "expected 36 layers x {V.weight,"
                               " V.bias, U.weight}")
    print("check 1 PASS: {} trainable params, all Gamma "
          "(36 layers x 3 tensors)".format(n_gamma), flush=True)

    # ---- a real L=3 sample through the data protocol -----------------
    import os
    os.environ["DIALOG_MIRROR"] = a.dialog_mirror
    dialog, collator = tc.build_dataset(args, tokenizer)
    item = None
    for it in dialog.trainset:
        if len(it["dialog"]) >= 5:
            item = dict(it)
            break
    assert item is not None, "no dialogue with >= 5 turns in the train set"
    item["dialog"] = list(item["dialog"])[:5]
    item["fixed_depth"] = True
    batch = collator([item])
    batch = {k: v.to(device) for k, v in batch.items()}

    # ---- check 2: step-0 identity with the pure merge host -----------
    with torch.no_grad():
        out_ccm = tc.run_forward(model, batch, device, grad_enabled=False)
        ref_logits = out_ccm.logits.clone()
        # Detach every Gamma: the pure CCM-merge host.
        saved = [attn.gamma for attn in gamma_layers(model)]
        for attn in gamma_layers(model):
            attn.gamma = None
        out_plain = tc.run_forward(model, batch, device, grad_enabled=False)
        for attn, g in zip(gamma_layers(model), saved):
            attn.gamma = g
    max_diff = float((ref_logits - out_plain.logits).abs().max())
    assert max_diff == 0.0, ("step-0 logits differ from the merge host: "
                             "max |diff| = {}".format(max_diff))
    print("check 2 PASS: step-0 logits bit-identical to the CCM-merge "
          "host (max |diff| = {})".format(max_diff), flush=True)

    # ---- check 3: one Gamma call per compression layer ---------------
    counts = []
    saved_fwd = []
    for attn in gamma_layers(model):
        g = attn.gamma
        saved_fwd.append((g, g.forward))
        orig = g.forward
        g.forward = lambda prev, cur, t, g=g, orig=orig: (
            counts.append(int(t.reshape(-1)[0])) or orig(prev, cur, t))
    with torch.no_grad():
        tc.run_forward(model, batch, device, grad_enabled=False)
    for g, fwd in saved_fwd:
        g.forward = fwd
    # The module is SHARED across the K and V merges, so each turn calls
    # it twice (once per stream): 36 layers x 3 turns x {K, V} = 216.
    from collections import Counter
    c = Counter(counts)
    assert len(counts) == 36 * 3 * 2, (len(counts),
                                       "expected 36 x 3 turns x {K, V}")
    assert c[1] == c[2] == c[3] == 36 * 2, (
        dict(c), "each layer must see t = 1, 2, 3 (K and V streams)")
    print("check 3 PASS: {} Gamma calls = 36 layers x t=1,2,3 x "
          "{{K, V}} (L=3 sample)".format(len(counts)), flush=True)

    print("test 3 PASS: frozen host is Gamma-only, step-0 identical to "
          "CCM-merge, Gamma covers every layer")


if __name__ == "__main__":
    main()
