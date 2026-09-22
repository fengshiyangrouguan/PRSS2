#!/usr/bin/env python3
"""Review test 4 (2026-09-22 ruling): native compression actuation (GPU).

The four-verification gate for the COMP-RPBE native design (actuation
point = conditional LoRA + COMP/SUM rows; recursion node = the native
CCM merge step (M_{t-1}, u_t) -> M_t, depth D=L, Z_t = M_t).  Run ONCE
on the training server BEFORE any native training:

  Part A  construction: build -> wrap_lora -> comp rows trainable ->
          native freeze-base, NO Gamma.  The trainable set is exactly
          the native compression params.
  Part B  (verification 2): a real CE backward reaches ONLY the native
          params — lora_B and comp_embeddings grads are non-zero, every
          frozen backbone param has grad None.
  Part C  (verification 3): the z_t lift (extract_z on the pure CCM
          mean, grad-connected) pulls back to the conditional LoRA —
          lora_B / comp_embeddings VJP non-zero; lora_A is EXACTLY zero
          at theta_0 because the PEFT B=0 branch zeroes its first-step
          gradient (the native counterpart of the gamma-line U/V note).
  Part D  (verification 4, unit part): theta_0 reproducibility — the
          same seed rebuilds an identical digest AND paired_seed_hash
          (the two native arms share the construction, so identical
          hashes <=> identical sampling streams); a different seed
          changes the hash.
  Part E  RNG precondition: two forwards under the same RNG state are
          bit-identical in train() mode (pass-1/pass-2 replay identity
          with lora_dropout pinned to 0.0).

Usage (server, qwen3 environment):
  python -u scripts/test_review_4_native.py \
      --model-name-or-path /root/autodl-tmp/qwen3-4b-instruct \
      --dialog-mirror /root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog \
      --gpu 0
"""
import argparse
import re
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
from rpbe.hosts.ccm.adapter import CCMHostAdapter
from rpbe.hosts.ccm.ccm_patch import paired_seed_hash


def native_build(args, tokenizer, device):
    """Mirror the train_ccm native-actuation construction: build ->
    wrap_lora -> comp rows trainable -> freeze every non-native param.
    NO Gamma attach (the forward is the pure CCM merge)."""
    model = tc.build_model(args, device)
    model = tc.wrap_lora(model, args.lora_r)
    model.update_comp_token(
        [tokenizer.comp_token_id[k] for k in range(tc.N_TOK)],
        [tokenizer.sum_token_id[k] for k in range(tc.N_TOK)])
    model.base_model.model.model.embed_tokens \
        .comp_embeddings.weight.requires_grad_(True)
    n_frozen = 0
    for _n, p in model.named_parameters():
        if "lora_" in _n or "comp_embeddings" in _n:
            continue
        if p.requires_grad:
            p.requires_grad_(False)
            n_frozen += 1
    _base = model
    while not hasattr(_base, "layers") and hasattr(_base, "model"):
        _base = _base.model
    assert not getattr(_base, "_gamma_attached", False), \
        "native construction must not attach Gamma"
    model.train()  # lora_dropout pinned 0.0 -> replay identity
    return model, n_frozen


def get_batch(dialog, collator, device, min_turns=3):
    """A real dialogue with L >= min_turns-2 compression layers."""
    item = None
    for it in dialog.trainset:
        if len(it["dialog"]) >= min_turns + 2:
            item = dict(it)
            break
    assert item is not None, "no dialogue with enough turns"
    item["dialog"] = list(item["dialog"])[:min_turns + 2]
    item["fixed_depth"] = True
    batch = collator([item])
    return {k: v.to(device) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name-or-path", required=True)
    ap.add_argument("--dialog-mirror", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    import os
    os.environ["DIALOG_MIRROR"] = a.dialog_mirror
    device = torch.device("cuda", a.gpu)
    args = types.SimpleNamespace(
        arm="ours", host="qwen3",
        model_name_or_path=a.model_name_or_path,
        dialog_mirror=a.dialog_mirror,
        relative_embedding="skip", lora_r=8, gamma_hidden=64,
        z_dim=128, rpbe_seed=0, sketch_dim=32,
        official_host=False, foundation="", official_adapter="",
        micro_batch=1, rpbe_native_compression=True)

    global tokenizer
    tokenizer = tc.build_tokenizer(args)
    comp_ids = tokenizer.comp_token_id
    sum_ids = tokenizer.sum_token_id
    dialog, collator = tc.build_dataset(args, tokenizer)

    # ---- Part A: trainable set == native compression params ----------
    # Seed the FIRST build too: Part D compares its digest against
    # same-seed rebuilds, so the anchor construction must start from
    # the same RNG state.
    tc.seed_all(a.seed)
    model, n_frozen = native_build(args, tokenizer, device)
    trainable = [(n, p) for n, p in model.named_parameters()
                 if p.requires_grad]
    bad = [n for n, _p in trainable
           if "lora_" not in n and "comp_embeddings" not in n]
    assert not bad, "non-native trainable params: {}".format(bad[:5])
    # peft's get_peft_model already froze the base (mark_only_lora_as_
    # trainable), so the explicit freeze-base loop may freeze ZERO extra
    # params — assert the resulting STATE (every non-native param is
    # frozen), not the loop count.
    frozen_ok = all(
        not p.requires_grad
        for n, p in model.named_parameters()
        if "lora_" not in n and "comp_embeddings" not in n)
    assert frozen_ok, "some non-native param stayed trainable"
    n_layers = model.model.config.num_hidden_layers
    n_exp = n_layers * 4 * 2 + 1   # q/k/v/o x {lora_A, lora_B} + comp rows
    assert len(trainable) == n_exp, (
        len(trainable), "expected {} layers x 4 proj x {{A,B}} + 1 comp "
        "row tensor = {}".format(n_layers, n_exp))
    print("Part A PASS: trainable = conditional LoRA ({} layers x 4 proj "
          "x {{A,B}}) + COMP/SUM rows ({} tensors), every non-native "
          "param frozen ({} frozen by peft, {} by the explicit loop), "
          "no Gamma attached".format(n_layers, len(trainable),
                                     len([1 for _n, p in
                                          model.named_parameters()
                                          if not p.requires_grad]),
                                     n_frozen), flush=True)

    batch = get_batch(dialog, collator, device)

    # ---- Part B (verification 2): CE reaches ONLY native params -------
    model.zero_grad(set_to_none=True)
    fwd_out = tc.run_forward(model, batch, device, grad_enabled=True)
    task_sum, n_valid = tc.task_ce_shifted(fwd_out, batch["labels"],
                                           device)
    (task_sum / max(n_valid, 1)).backward()
    lora_b_nz = lora_a_nz = comp_nz = 0
    frozen_with_grad = []
    lora_b_zero = []
    n_layers = model.model.config.num_hidden_layers
    for n, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                frozen_with_grad.append(n)
            continue
        g = p.grad
        assert g is not None, "trainable {} has no grad".format(n)
        s = float(g.abs().sum())
        if "lora_B" in n:
            (lora_b_nz := lora_b_nz + 1) if s > 0.0 else \
                lora_b_zero.append(n)
        if "comp_embeddings" in n and s > 0.0:
            comp_nz += 1
    assert not frozen_with_grad, \
        "frozen params received gradients: {}".format(frozen_with_grad[:5])
    # Structural zeros (conditional-LoRA design, NOT a bug): the LoRA
    # branch activates ONLY at COMP/SUM positions, and the LAST layer's
    # q/o output reaches only that position's own logits — COMP/SUM
    # positions are never labels (-100 masked), so layers.35 q/o lora_B
    # grads are EXACTLY zero.  k/v stay non-zero (attended downstream).
    exp_zero = {(n_layers - 1, "q_proj"), (n_layers - 1, "o_proj")}
    got_zero = set()
    for n in lora_b_zero:
        m = re.search(r"layers\.(\d+)\.self_attn\.(\w+_proj)", n)
        got_zero.add((int(m.group(1)), m.group(2)))
    assert lora_b_nz == n_layers * 4 - 2, (
        lora_b_nz, lora_b_zero,
        "every layer/proj lora_B except the structural zeros must "
        "receive task gradient")
    assert got_zero == exp_zero, (got_zero, exp_zero)
    assert comp_nz == 1, "comp rows must receive task gradient"
    print("Part B PASS: CE grads reach all {} lora_B tensors + comp "
          "rows; structural zeros = layer {} q/o lora_B (conditional "
          "LoRA + label mask); {} frozen params have grad None".format(
              lora_b_nz, n_layers - 1,
              sum(1 for _n, _p in model.named_parameters()
                  if not _p.requires_grad)), flush=True)

    # ---- Part C (verification 3): z_t VJP to the conditional LoRA -----
    meta = tc.parse_meta(batch, comp_ids, sum_ids, 0)[0]  # per-row list
    assert meta["ok"] and meta["L"] >= 1, "sample must carry >= 1 cut"
    cfg = model.model.config
    n_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim",
                       cfg.hidden_size // cfg.num_attention_heads)
    adapter = CCMHostAdapter(model, n_layers=cfg.num_hidden_layers,
                             n_heads=n_heads, head_dim=head_dim,
                             z_dim=args.z_dim, seed=args.rpbe_seed)
    model.zero_grad(set_to_none=True)
    fwd_out = tc.run_forward(model, batch, device, grad_enabled=True)
    s0 = int(meta["blocks"][0][1])
    z = adapter.extract_z(torch.tensor(
        [[s0, s0 + 1]], dtype=torch.long, device=device))[0]
    gd = torch.randn_like(z)
    (gd * z).sum().backward()
    adapter.clear()
    lora_b_nz = lora_a_nz = comp_nz = 0
    lora_b_zero = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        if "lora_B" in n:
            # NOTE: under the z_t VJP, layer 35's q/o lora_B grads are
            # None (NO path in the graph — hidden35 feeds no downstream
            # K/V), not zero tensors as under CE; count both as zeros.
            if g is not None and float(g.detach().abs().sum()) > 0.0:
                lora_b_nz += 1
            else:
                lora_b_zero.append(n)
            continue
        if g is None:
            continue
        s = float(g.detach().abs().sum())
        if "lora_A" in n and s > 0.0:
            lora_a_nz += 1
        if "comp_embeddings" in n and s > 0.0:
            comp_nz += 1
    # Same structural zeros as Part B: z_t = lift(SUM K/V) flows back
    # through the merge chain (COMP K/V <- k_proj/v_proj <- hidden
    # chain), so every k/v lora_B plus layers 0..34 q/o (via the hidden
    # chain feeding downstream k/v) receive gradient; layer 35's q/o
    # output never reaches a label (COMP/SUM positions are masked).
    exp_zero = {(n_layers - 1, "q_proj"), (n_layers - 1, "o_proj")}
    got_zero = set()
    for n in lora_b_zero:
        m = re.search(r"layers\.(\d+)\.self_attn\.(\w+_proj)", n)
        got_zero.add((int(m.group(1)), m.group(2)))
    assert lora_b_nz == n_layers * 4 - 2 and comp_nz == 1, (
        lora_b_nz, comp_nz, "z_t must pull back to lora_B + comp rows")
    assert got_zero == exp_zero, (got_zero, exp_zero, lora_b_zero,
                                  lora_b_nz)
    assert lora_a_nz == 0, (
        "lora_A grads must be EXACTLY zero at theta_0: the PEFT B=0 "
        "zero branch kills the first-step A-gradient (the native "
        "counterpart of the gamma-line U/V note)")
    print("Part C PASS: z_t VJP reaches {} lora_B tensors + comp rows; "
          "structural zeros = layer {} q/o lora_B; lora_A exactly zero "
          "(B=0 zero branch)".format(lora_b_nz, n_layers - 1),
          flush=True)

    # ---- Part D (verification 4, unit): theta_0 reproducibility -------
    # Each rebuild re-seeds FIRST: the PEFT LoRA init draws from the
    # global RNG, so a rebuild without re-seeding would get different
    # lora_A values and fail the digest equality spuriously.
    tc.seed_all(a.seed)
    digest_1 = tc.params_digest(
        [p for p in model.parameters() if p.requires_grad])
    hash_1 = paired_seed_hash(a.seed, model)
    del model, adapter, fwd_out
    torch.cuda.empty_cache()
    tc.seed_all(a.seed)
    model2, _ = native_build(args, tokenizer, device)
    digest_2 = tc.params_digest(
        [p for p in model2.parameters() if p.requires_grad])
    hash_2 = paired_seed_hash(a.seed, model2)
    assert digest_1 == digest_2, "same-seed rebuild must be identical"
    assert hash_1 == hash_2, (
        "the two native arms share the construction: identical "
        "paired_seed_hash <=> identical sampling streams")
    del model2
    torch.cuda.empty_cache()
    tc.seed_all(a.seed)
    model3, _ = native_build(args, tokenizer, device)
    hash_3 = paired_seed_hash(a.seed + 1, model3)
    assert hash_3 != hash_1, "different seed must change the hash"
    print("Part D PASS: theta_0 digest reproducible; paired_seed_hash "
          "seed-dependent (native arms share the construction)",
          flush=True)

    # ---- Part E: RNG replay identity in train() mode ------------------
    torch.manual_seed(0)
    with torch.no_grad():
        out1 = tc.run_forward(model3, batch, device, grad_enabled=False)
    torch.manual_seed(0)
    with torch.no_grad():
        out2 = tc.run_forward(model3, batch, device, grad_enabled=False)
    max_diff = float((out1.logits - out2.logits).abs().max())
    assert max_diff == 0.0, \
        "train()-mode replay must be bit-identical (max diff {})".format(
            max_diff)
    print("Part E PASS: same-RNG forwards bit-identical in train() mode "
          "(two-pass replay precondition)", flush=True)

    print("test 4 PASS (native compression actuation): trainable = "
          "native params only, CE + z_t VJP reach the conditional LoRA, "
          "theta_0 reproducible, replay identity holds")


if __name__ == "__main__":
    main()
