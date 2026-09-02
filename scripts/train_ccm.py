#!/usr/bin/env python3
"""CCM-merge x RPBE training entry (plan v2 L5).

Three arms, identical task samples / seed / cadence:

  ccm_merge       official protocol: LoRA on merge_recur, one pass, one
                  optimizer step per ``grad_accum`` microbatch.
  gamma_task_only  Gamma attached, LoRA, the SAME two-pass window replay
                  as ours with lambda = 0 (matched control).
  ours            Gamma + RPBE: pass 1 (no grad) collects cut rows
                  incrementally into a KFMomentWindow that closes at
                  >= kf_min_cuts effective cuts; pass 2 replays the whole
                  window's microbatch dicts and trains task CE plus the
                  exact surrogate (numerically zero, gradient = window
                  J).  One optimizer step per closed window.

RNG protocol (plan L5): each microbatch's pass-1 forward runs under a
saved RNG state that is restored right after (builder counters are NOT
restored — cut ids stay monotonic), so the data-sampling stream matches
the official single-pass arm exactly.  The model carries zero dropout
(LLaMA default), so pass-2 replays the identical forward; this is
asserted at startup (L6 test 3).  Pass 2 restores the window-start state
(RNG + builder counter) so occurrence ids align with the replay plan.

Lambda follows the r_eff calibration rule (plan L5); --calibrate-lambda
measures r_eff = ||g_KF|| / ||g_task|| on the first closed window and
exits with the derived lambda.
"""

import argparse
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rpbe.hosts.ccm.adapter import CCMHostAdapter
from rpbe.hosts.ccm.ccm_patch import (N_TOK_LOCK, attach_gamma,
                                      paired_seed_hash, wrap_lora)
from rpbe.llm.dialogue_records import (DialogueCutBuilder, DialogueMeta,
                                       Llmmaps, MEM_TAU)
from rpbe.llm.utterance_embed import UtteranceEmbed
from rpbe.loss import KFMomentWindow
from rpbe.training.checkpoint import _restore_rng, _rng_state

N_TOK = N_TOK_LOCK  # comp slots; 2 comp + 2 sum = 4 added tokens


def parse_args():
    p = argparse.ArgumentParser("CCM x RPBE training (plan v2 L5)")
    p.add_argument("--arm", required=True,
                   choices=["ccm_merge", "gamma_task_only", "ours",
                            "ccm_merge_official"],
                   help="ccm_merge_official: official fixed-accumulation "
                        "cadence reproduction arm (frozen cadence is "
                        "window-matched for the three main arms; the "
                        "official arm is reported separately)")
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--dialog-mirror", required=True,
                   help="DIALOG_MIRROR: ijcnlp_dailydialog layout dir")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--grad-accum", type=int, default=128,
                   help="microbatch per update in --merge-cadence official "
                        "(official-reproduction arm only)")
    p.add_argument("--merge-cadence", default="window-matched",
                   choices=["window-matched", "official"],
                   help="window-matched: the ccm_merge arm fires an update "
                        "on the same adaptive boundary as the RPBE arms "
                        "(>= min-effective-cuts dialogues with k>=4), so "
                        "task exposure and scheduler cadence are identical "
                        "across the three arms (frozen_method.json "
                        "cadence; review P0-2).  official: fixed "
                        "grad-accum microbatch cadence for the "
                        "ccm_merge_official reproduction reference.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="max_grad_norm (official Trainer default 1.0; "
                        "frozen_method.json training.grad_clip is "
                        "authoritative and an explicit override is "
                        "refused)")
    p.add_argument("--relative-embedding", default="skip",
                   choices=["skip", "base"])
    # RPBE
    p.add_argument("--kf-lambda", type=float, default=1e-3)
    p.add_argument("--kf-min-cuts", type=int, default=128,
                   help="effective-cuts gate (window closes at >= this)")
    p.add_argument("--ridge-eps", type=float, default=1e-3)
    p.add_argument("--sketch-dim", type=int, default=64)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--rpbe-seed", type=int, default=0)
    p.add_argument("--gamma-hidden", type=int, default=64)
    p.add_argument("--calibrate-lambda", action="store_true",
                   help="measure r_eff on the first closed window and exit")
    # monitoring
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--resume-from", default="",
                   help="resume from an adapter-only checkpoint.pt")
    p.add_argument("--max-pending-mbs", type=int, default=2048,
                   help="degenerate-window guard: pending cap before abort")
    p.add_argument("--max-windows", type=int, default=0,
                   help="verification runs: stop after this many closed "
                        "windows (0 = run to max_steps).  max_steps stays "
                        "frozen at 1000, so short verification runs use "
                        "this instead of changing the step budget.")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


FROZEN_PATH = Path(__file__).resolve().parents[1] / "configs" / "ccm" \
    / "frozen_method.json"


def enforce_frozen(args):
    """The frozen method spec is authoritative (review P0-3 tail).

    The trainer entry point MUST read configs/ccm/frozen_method.json and
    refuse an inconsistent CLI override — a silent drift (e.g. a leftover
    grad_clip=5.0 default while the reported protocol is the official
    1.0) would train a different method than the one reviewed and
    frozen.  Every bound key is checked here; values the CLI does not
    carry are read back for logging only.

    Review ruling #2 extends the binds to the full method vector
    (lambda_kf, z_dim, gamma_hidden, merge_cadence, max_steps, LoRA rank)
    and pins the arm/cadence pairing: the ccm_merge_official reproduction
    arm is the ONLY arm allowed the fixed official cadence, the three main
    arms must use the machine value "window-matched", and lambda_kf is
    either null (calibration-only run permitted, anything else refused)
    or a committed number that overrides the CLI.
    """
    with open(FROZEN_PATH, encoding="utf-8") as f:
        fz = json.load(f)
    binds = [
        ("--ridge-eps", "ridge_eps", fz["rpbe"]["ridge_eps"]),
        ("--sketch-dim", "sketch_dim", fz["rpbe"]["sketch_dim_m"]),
        ("--rpbe-seed", "rpbe_seed", fz["rpbe"]["rpbe_map_seed"]),
        ("--kf-min-cuts", "kf_min_cuts",
         fz["window"]["min_effective_cuts"]),
        ("--grad-clip", "grad_clip", fz["training"]["grad_clip"]),
        ("--z-dim", "z_dim", fz["rpbe"]["z_v_dim"]),
        ("--gamma-hidden", "gamma_hidden", fz["gamma"]["hidden"]),
        ("--lora-r", "lora_r", fz["adapter"]["lora_r"]),
        ("--max-steps", "max_steps", fz["training"]["steps"]),
    ]
    for flag, name, frozen_val in binds:
        cli_val = getattr(args, name)
        if cli_val != frozen_val:
            raise SystemExit(
                "[frozen] {}={} conflicts with frozen_method.json (={}); "
                "the frozen spec is authoritative — edit the spec file "
                "to override".format(flag, cli_val, frozen_val))
    # Arm/cadence pairing: only the official reproduction arm may run
    # the fixed accumulation cadence; the three main arms share the
    # adaptive "window-matched" boundary (review P0-2 / ruling #2).
    if args.arm == "ccm_merge_official":
        if args.merge_cadence != "official":
            raise SystemExit(
                "[frozen] ccm_merge_official must run with "
                "--merge-cadence official (frozen_method.json "
                "training.merge_cadence = window-matched applies to the "
                "three main arms only)")
    elif args.merge_cadence != "window-matched":
        raise SystemExit(
            "[frozen] arm {} must run with the frozen window-matched "
            "cadence; --merge-cadence official is reserved for "
            "ccm_merge_official".format(args.arm))
    # Lambda authority: the calibration-only run writes the derived
    # lambda into the frozen spec; afterwards the number overrides any
    # CLI value and re-calibration is refused.
    lam = fz["rpbe"]["lambda_calibration"]["lambda_kf"]
    if args.arm == "ours":
        if lam is None:
            if not args.calibrate_lambda:
                raise SystemExit(
                    "[frozen] rpbe.lambda_calibration.lambda_kf is null — "
                    "the ours arm may only run the calibration-only pass "
                    "(--calibrate-lambda).  Commit its derived_lambda "
                    "into configs/ccm/frozen_method.json before the "
                    "formal runs.")
        else:
            if args.calibrate_lambda:
                raise SystemExit(
                    "[frozen] lambda_kf={} is committed in "
                    "frozen_method.json; re-running --calibrate-lambda "
                    "would retrain the calibration.  Delete the field "
                    "first if a recalibration is really intended.".format(
                        lam))
            if args.kf_lambda != lam:
                print("[frozen] CLI --kf-lambda={} overridden by the "
                      "committed lambda_kf={} "
                      "(frozen_method.json)".format(args.kf_lambda, lam),
                      flush=True)
                args.kf_lambda = lam
    elif args.calibrate_lambda:
        raise SystemExit(
            "[frozen] --calibrate-lambda measures r_eff of the surrogate "
            "(arm 'ours'); arm '{}' does not train the auxiliary "
            "term".format(args.arm))
    return fz


def build_tokenizer(args):
    from transformers import LlamaTokenizer
    tok = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None \
        else tok.eos_token_id
    tok.bos_token_id = tok.bos_token_id or 1
    tok.eos_token_id = tok.eos_token_id or 2
    tok.padding_side = "left"
    added = [f"<COMP{k}>" for k in range(N_TOK)] \
        + [f"<SUM{k}>" for k in range(N_TOK)]
    tok.add_special_tokens({"additional_special_tokens": added})
    ids = tok.additional_special_tokens_ids[-2 * N_TOK:]
    tok.comp_token_id = ids[:N_TOK]
    tok.sum_token_id = ids[N_TOK:]
    return tok


def build_model(args, device):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from src.arch.ccm_llama import LlamaForCausalLM_CCM
    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.comp_relative_embedding = args.relative_embedding
    model = LlamaForCausalLM_CCM.from_pretrained(
        args.model_name_or_path, config=config,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.resize_token_embeddings(32000 + 2 * N_TOK)
    model.update_comp_token([32000 + k for k in range(N_TOK)],
                            [32000 + N_TOK + k for k in range(N_TOK)])
    return model.to(device)


def wrap_lora(model, r):
    from rpbe.hosts.ccm.ccm_patch import wrap_lora as _wrap
    return _wrap(model, r=int(r))


def build_dataset(args, tokenizer):
    from src.arguments import CompressionArguments
    from src.data.dialogue.data import DialogueDataset
    from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA
    os.environ["DIALOG_MIRROR"] = args.dialog_mirror
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=N_TOK,
                                     add_comp_token=True,
                                     relative_embedding=args.relative_embedding)
    dialog = DialogueDataset(tokenizer, comp_token=tokenizer.comp_token_id,
                             online=True, add_comp_token=True,
                             clean_split=True)
    collator = DataCollatorForDialogue_LLAMA(
        dialog=dialog, tokenizer=tokenizer, comp_args=comp_args,
        comp_token=tokenizer.comp_token_id, sum_token=tokenizer.sum_token_id,
        padding="left", pad_token=tokenizer.pad_token_id,
        label_pad_token_id=-100)
    return dialog, collator


def parse_meta(batch, comp_ids, sum_ids, sample_id_global):
    """Deterministic per-sample metadata from the padded collator batch.

    Returns a list (one per batch row) of dicts: k, blocks (C0/S0
    positions per turn), utterance_spans, prompt_end.  Every turn block
    is [C0, C1, S0, S1] right after its utterance; the final context turn
    carries no block.  ``sample_id_global`` is the stream-global sample
    index (unique across epochs) used as the tree identity.
    """
    ids = batch["input_ids"]
    labels = batch["labels"]
    B, L = ids.shape
    metas = []
    for b in range(B):
        row = ids[b]
        c0 = (row == comp_ids[0]).nonzero(as_tuple=False).flatten().tolist()
        blocks = []
        ok = True
        for pos in c0:
            if pos + 3 >= L or row[pos + 1] != comp_ids[1] \
                    or row[pos + 2] not in sum_ids \
                    or row[pos + 3] not in sum_ids:
                ok = False
                break
            blocks.append((pos, pos + 2))  # (C0 pos, S0 pos)
        n_completion = int((labels[b] != -100).sum())
        prompt_end = L - n_completion
        k = len(blocks) + 1  # context turns = blocks + final blockless turn
        utterance_spans = []
        prev_end = -1
        for (c0_pos, _s0) in blocks:
            utterance_spans.append((prev_end + 1, c0_pos))
            prev_end = c0_pos + 3  # S1 position
        utterance_spans.append((prev_end + 1, prompt_end))
        metas.append({"sample_id": int(sample_id_global) + b, "row": b,
                      "k": k,
                      "blocks": blocks,
                      "utterance_spans": utterance_spans,
                      "prompt_end": prompt_end, "ok": ok})
    return metas


def run_forward(model, batch, device, grad_enabled):
    # labels are NOT passed here: the task CE is computed once by
    # task_ce (passing them would compute the same loss a second time).
    # attention_mask_comp IS passed: the official merge_recur collator
    # derives it from the SUM tokens and the vendored LlamaModel folds
    # it into the attention mask (comp/sum positions blocked from the
    # causal stream), so omitting it silently trains a DIFFERENT
    # attention pattern than the official protocol (caught by the L6.5
    # gate-2 parity run).  fp16 autocast matches the official Trainer:
    # the vendored conditional-LoRA layer computes its lora branch in
    # fp32, so the forward MUST run under autocast (mixed fp16/fp32
    # without autocast raises on the fp16 base path).
    ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
    amc = batch.get("attention_mask_comp")
    with ctx:
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            return model(input_ids=batch["input_ids"].to(device),
                         attention_mask=batch["attention_mask"].to(device),
                         attention_mask_comp=amc.to(device)
                         if amc is not None else None)


def task_ce_shifted(out, labels, device):
    """Official CCM task CE: SHIFTED sum + valid count.

    The vendored model's internal loss (ccm_llama.py ~line 922) shifts
    logits/labels by one ("tokens < n predict n") before the CE; the
    official CompSeq2SeqTrainer backprops that per-microbatch loss
    directly.  This helper returns (shifted_sum, n_valid) so callers can
    build the per-microbatch MEAN (= sum / valid, the official
    reduction) and normalize by the window size where the L6.5 review
    requires it.  The token-normalized CE log line uses sum / total
    valid tokens.
    """
    logits = out.logits
    labs = labels.to(device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labs[..., 1:].contiguous()
    n_valid = int((shift_labels != -100).sum())
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1), ignore_index=-100, reduction="sum")
    return loss, n_valid


def collect_replay_z(meta, adapter, device):
    """Pass-2 extraction ONLY: z_v at the cut position (gradient-
    connected).  No builder, no chi, no p — those finished their job at
    the pass-1 window close (L6.5 review structural fix)."""
    v = meta["k"] - 3
    s0_pos = meta["blocks"][v][1]
    sum_positions = torch.tensor([[s0_pos, s0_pos + 1]],
                                 dtype=torch.long, device=device)
    return adapter.extract_z(sum_positions)[0]


def collect_rows(meta, adapter, builder, utter_embed, embed_tokens, batch,
                 device):
    """Extract z_v + chi and build the two horizon rows for one sample."""
    if not meta["ok"] or meta["k"] < 4:
        return []
    v = meta["k"] - 3
    s0_pos = meta["blocks"][v][1]
    sum_positions = torch.tensor([[s0_pos, s0_pos + 1]],
                                 dtype=torch.long, device=device)
    z = adapter.extract_z(sum_positions)  # [1, z_dim]
    spans = meta["utterance_spans"]
    ids = batch["input_ids"][meta["row"]]
    chi1 = utter_embed(embed_tokens,
                       ids[spans[v + 1][0]:spans[v + 1][1]]
                       .unsqueeze(0).to(device), tag=0)
    chi2 = utter_embed(embed_tokens,
                       ids[spans[v + 1][1]:].unsqueeze(0).to(device), tag=1)
    dm = DialogueMeta(sample_id=int(meta["sample_id"]), k=int(meta["k"]),
                      sum_positions=[(p + 2, p + 3) for (p, _s)
                                     in meta["blocks"]],
                      utterance_spans=list(meta["utterance_spans"]))
    return builder.build(dm, z, chi1[0], chi2[0])


def batch_surrogate(z_rows_by_oid, batch_terms, lam, device):
    """Numerically zero surrogate; gradient = exact window J (plan L5).

    K = 1 here (one optimizer step per closed window), so the auxiliary
    is simply -lambda * sum_v (<sg(g_v), z_v> - sg(<g_v, z_v>)).

    ``batch_terms`` is ``[(occurrence_id, g), ...]`` for ONE replay batch
    only — the caller resolves which of the window's gradients belong to
    this batch through occurrence ids (review P0-1: never slice the
    window plan by batch position; the plan covers only the
    cut-producing batches while the pending list mixes in k < 4
    microbatches)."""
    terms = []
    for oid, g in batch_terms:
        z = z_rows_by_oid.get(oid)
        if z is None:
            continue
        gd = g.detach()
        terms.append((gd * z).sum() - (gd * z.detach()).sum())
    if not terms:
        return torch.zeros((), device=device), 0
    return -lam * sum(terms), len(terms)


def repr_grad_norm(params):
    total = 0.0
    n = 0
    for p in params:
        if p.grad is not None:
            total += float((p.grad.detach().double() ** 2).sum())
            n += 1
    return math.sqrt(total) if n else 0.0


def params_digest(params):
    """sha256 over the trainable weights (CPU copy).  The lambda
    calibration asserts the digest is unchanged across its measurements
    so the r_eff values really live on theta_0 (review P0-4)."""
    h = hashlib.sha256()
    for p in params:
        h.update(np.ascontiguousarray(
            p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def trainable_state_dict(model):
    """LoRA + Gamma + any other trainable params ONLY (the frozen 7B
    backbone is NOT stored; a full state_dict costs ~13GB per file)."""
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if p.requires_grad}


def save_trainable(path, model, **extra):
    torch.save({"model": trainable_state_dict(model), **extra}, path)


def load_trainable(path, model, optimizer, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"],
                                                strict=False)
    if unexpected:
        raise RuntimeError("unexpected keys in checkpoint: {}"
                           .format(sorted(unexpected)[:5]))
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def main():
    args = parse_args()
    enforce_frozen(args)  # review P0-3: frozen spec is authoritative
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"

    tokenizer = build_tokenizer(args)
    model = build_model(args, device)
    model = wrap_lora(model, args.lora_r)
    # The PEFT wrap can replace the CausalLM wrapper; re-assert the
    # comp/sum token registration on the wrapped object.
    model.update_comp_token([32000 + k for k in range(N_TOK)],
                            [32000 + N_TOK + k for k in range(N_TOK)])
    use_rpbe = args.arm in ("ours", "gamma_task_only")
    if use_rpbe:
        attach_gamma(model, hidden=args.gamma_hidden)
    cfg = model.model.config

    # RNG protocol precondition: zero dropout everywhere (pass-1/pass-2
    # replay must be identical; LLaMA ships with dropout 0).
    assert getattr(cfg, "attention_dropout", 0.0) == 0.0 \
        and getattr(cfg, "hidden_dropout", 0.0) == 0.0, \
        "two-pass replay requires zero model dropout"
    if args.lora_r:
        pass  # lora_dropout pinned to 0.0 in wrap_lora

    adapter = maps = builder = utter_embed = window = None
    if use_rpbe:
        adapter = CCMHostAdapter(model, n_layers=cfg.num_hidden_layers,
                                 n_heads=cfg.num_attention_heads,
                                 head_dim=cfg.hidden_size
                                 // cfg.num_attention_heads,
                                 z_dim=args.z_dim, seed=args.rpbe_seed)
        maps = Llmmaps(d_chi=64, d_phi=32, m=args.sketch_dim,
                       seed=args.rpbe_seed).to(device)
        builder = DialogueCutBuilder(maps, z_dim=args.z_dim,
                                     seed=args.rpbe_seed)
        utter_embed = UtteranceEmbed(hidden_dim=cfg.hidden_size, d_chi=64,
                                     seed=args.rpbe_seed).to(device)
        window = KFMomentWindow({MEM_TAU: args.z_dim}, min_ratio=2.0,
                                min_abs=args.kf_min_cuts,
                                eps=args.ridge_eps, fixed_maps=maps,
                                strict=False, autoclose=False)

    params = [p for p in model.parameters() if p.requires_grad]
    # Official CCM protocol (L6.5 review P0-2): AdamW with weight_decay=0,
    # cosine decay to zero, 3% warmup, and the official Trainer's default
    # max_grad_norm=1.0.  Same scheduler on all three arms (per-step).
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = max(1, int(args.max_steps))
    warmup_steps = max(1, int(0.03 * total_steps))

    def _lr_lambda(s):
        if s < warmup_steps:
            return float(s) / float(warmup_steps)
        progress = float(s - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    dialog, collator = build_dataset(args, tokenizer)
    comp_ids = tokenizer.comp_token_id
    sum_ids = tokenizer.sum_token_id
    pad_id = tokenizer.pad_token_id
    embed_tokens = model.get_input_embeddings()

    train_items = dialog.train_dataset
    n_items = len(train_items)
    sample_cursor = 0

    def next_batch():
        nonlocal sample_cursor
        item = train_items[int(sample_cursor % n_items)]
        sample_cursor += 1
        return collator([item]), sample_cursor - 1

    threshold = window._threshold(MEM_TAU) if window else None
    save_json(out / "config.json", {
        "arm": args.arm, "seed": args.seed, "cli": vars(args),
        "paired_seed_hash": paired_seed_hash(args.seed, model)
        if use_rpbe else "n/a",
        "threshold": threshold,
    })
    print("arm={} seed={} threshold={} n_train={}".format(
        args.arm, args.seed, threshold, n_items), flush=True)

    step = 0
    total_task_sum = 0.0
    total_tokens = 0
    total_microbatches = 0
    total_kf = 0.0
    kf_closed = 0
    aux_terms = 0
    below_threshold = 0
    lambda_kf = args.kf_lambda
    t_start = time.time()
    t_win_start = t_start
    # CCM_PROFILE=1: per-window breakdown of pass1/collect/pass2/bwd
    # wall time (diagnostic for the L6.5 step-time gate).
    profile = os.environ.get("CCM_PROFILE") == "1"
    PROF = {}

    def _pf(key, t0):
        PROF[key] = PROF.get(key, 0.0) + time.perf_counter() - t0

    pending = []
    cut_records = []
    window_start_state = None
    # Review P0-2: the ccm_merge arm counts effective cuts (dialogues
    # with k >= 4) in the same stream and fires its update on the same
    # boundary as the RPBE windows, so all three arms see the same task
    # samples at the same scheduler step.
    merge_eff_cuts = 0
    # L6.5 gate 1: every arm hashes its (sample_id, k) data stream so the
    # three arms can be compared bit-for-bit after a run; the raw stream
    # is also written for prefix comparison across unequal window sizes.
    data_flow_hash = hashlib.sha256()
    data_flow_len = 0
    # Review (cadence collision): per-window boundary records
    # w{ordinal}:{n_mb}:{n_cut_mb} hashed into boundary_hash so the three
    # arms' window boundaries are comparable window by window, not only
    # through the final data-flow hash.
    boundary_hash = hashlib.sha256()
    data_flow_path = out / "data_flow.jsonl"
    data_flow_path.unlink(missing_ok=True)
    if args.resume_from:
        payload = load_trainable(args.resume_from, model, optimizer, device)
        step = int(payload.get("step", 0))
        lambda_kf = float(payload.get("lambda_kf", args.kf_lambda))
        if "sample_cursor" in payload:
            sample_cursor = int(payload["sample_cursor"])
        if "rng" in payload:
            _restore_rng(payload["rng"])
        if "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        if "builder_oid" in payload and builder is not None:
            builder.next_oid = int(payload["builder_oid"])
        print("RESUME step={} lambda={} cursor={}".format(
            step, lambda_kf, sample_cursor), flush=True)

    def pass1_one(batch, meta_list):
        """Incremental pass 1: forward (no grad) + rows into the window.
        Returns [(meta, occurrence_id)] per cut in this batch (the pass-2
        replay contract; builder counters stay monotonic)."""
        batch_cuts = []
        with torch.no_grad():
            _t = time.perf_counter()
            run_forward(model, batch, device, grad_enabled=False)
            _pf("pass1_fwd", _t)
            for meta in meta_list:
                _t = time.perf_counter()
                rows = collect_rows(meta, adapter, builder, utter_embed,
                                    embed_tokens, batch, device)
                _pf("collect_rows", _t)
                if rows:
                    _t = time.perf_counter()
                    window.add(rows)
                    _pf("window_add", _t)
                    batch_cuts.append((meta, rows[0].occurrence_id))
            adapter.clear()
        return batch_cuts

    def pass2_one(batch, cut_meta, g_by_oid, lam):
        """Pass-2 replay: task CE (official MEAN per microbatch, divided
        by the window size below) + exact surrogate.  ``cut_meta`` is the
        pass-1 [(meta, oid)] record — NO builder/chi/p here (L6.5
        review: pass 2 only re-extracts the gradient-connected z).

        Gradients are mapped through OCCURRENCE IDs (review P0-1): this
        batch's cuts carry their pass-1 occurrence ids; a cut is replayed
        iff its id is in the window plan's ``by_oid``.  Batch positions
        are never used to slice the plan (k < 4 microbatches interleave
        and the plan covers only cut-producing batches)."""
        _t = time.perf_counter()
        out = run_forward(model, batch, device, grad_enabled=True)
        _pf("pass2_fwd", _t)
        task_sum, n_valid = task_ce_shifted(out, batch["labels"], device)
        task_mean = task_sum / max(n_valid, 1)
        aux = torch.zeros((), device=device)
        n_terms = 0
        if g_by_oid:
            z_by_oid = {}
            batch_terms = []
            _t = time.perf_counter()
            for meta, oid in cut_meta:
                g = g_by_oid.get(oid)
                if g is not None:
                    z_by_oid[oid] = collect_replay_z(meta, adapter, device)
                    batch_terms.append((oid, g))
            adapter.clear()
            _pf("collect_z", _t)
            _t = time.perf_counter()
            aux, n_terms = batch_surrogate(z_by_oid, batch_terms, lam,
                                           device)
            _pf("surrogate", _t)
        return task_mean, task_sum, n_valid, aux, n_terms

    def grad_step():
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    while step < args.max_steps:
        batch, sample_id = next_batch()
        # Data-stream hash for every arm (parse_meta is pure, no RNG).
        metas = parse_meta(batch, comp_ids, sum_ids, sample_id)
        for m in metas:
            data_flow_hash.update(struct.pack(
                ">qq", int(m["sample_id"]) % n_items, int(m["k"])))
            data_flow_len += 1
        with data_flow_path.open("a") as f:
            for m in metas:
                f.write(json.dumps({"sid": int(m["sample_id"]) % n_items,
                                    "k": int(m["k"])}) + "\n")
        if window_start_state is None:
            # Builder counters are NOT rewound here: pass 2 never touches
            # the builder, so occurrence ids stay monotonic across the
            # whole run (L6.5 review P0-1).
            window_start_state = {"rng": _rng_state()}
        pending.append((batch, sample_id))
        if use_rpbe:
            # Incremental pass 1: RNG restored after each microbatch so
            # the data-sampling stream matches the single-pass arm; the
            # cut rows and their occurrence ids accumulate monotonically.
            # Microbatches with NO possible cut (k < 4) skip the pass-1
            # forward entirely: no row can come from them, and with zero
            # dropout the forward consumes no RNG, so the replay stream
            # is unchanged (L6.5 perf: DailyDialog's effective-cut rate
            # is ~50%, halving the pass-1 cost).
            state = {"rng": _rng_state()}
            if any(m["ok"] and m["k"] >= 4 for m in metas):
                batch_cuts = pass1_one(batch, metas)
            else:
                batch_cuts = []
            cut_records.append(batch_cuts)
            _restore_rng(state["rng"])
            if window.window_ready():
                _t = time.perf_counter()
                closed, plan, diag = window.close_replay()
                _pf("close_replay", _t)
                # P0-1 fix: capture the REAL post-pass-1 data-stream RNG
                # position; pass 2 replays from the window start and the
                # stream resumes from the captured position afterwards.
                resume_rng = _rng_state()
                _restore_rng(window_start_state["rng"])
                # Review P0-1: consume the window gradients through the
                # occurrence-id index.  The legacy per-batch slice was
                # aligned to the CUT-PRODUCING batches only, so indexing
                # it with the pending position silently dropped ~99% of
                # the cuts once k < 4 microbatches interleaved
                # (aux_terms was 11 over 7 windows instead of 896).
                g_by_oid = plan.get(MEM_TAU, {}).get("by_oid", {})
                if args.calibrate_lambda and kf_closed == 0 \
                        and not args.resume_from:
                    # Review P0-4 (lambda calibration timing): the
                    # calibration must run on theta_0 — BEFORE this
                    # window's optimizer/scheduler step — replaying the
                    # theta_0 adjoint plan against theta_0 z's.  The
                    # previous order ran the real pass 2 and grad_step
                    # first, then measured on theta_1: the surrogate
                    # became J_z(theta_1)^T dJ/dz|_{theta_0} (mixed
                    # point) and the task gradient was measured at
                    # theta_1 too, so the derived lambda was not the
                    # frozen spec's r_eff calibration value.  Asserts:
                    # trainable params unchanged and no optimizer /
                    # scheduler step has fired.
                    digest0 = params_digest(params)
                    assert step == 0, \
                        "lambda calibration must fire before the first " \
                        "optimizer step (step={})".format(step)
                    # r_eff on this window: separate norm measurements.
                    # g_task uses the real training scale (per-microbatch
                    # MEAN divided by len(pending)); g_kf uses the real
                    # pass-2 scale (aux NOT divided — review P0-3).
                    optimizer.zero_grad(set_to_none=True)
                    _restore_rng(window_start_state["rng"])
                    for b, sid in pending:
                        fwd_out = run_forward(model, b, device,
                                              grad_enabled=True)
                        task_sum_m, n_valid_m = task_ce_shifted(
                            fwd_out, b["labels"], device)
                        (task_sum_m / max(n_valid_m, 1)
                         / float(len(pending))).backward()
                    g_task = repr_grad_norm(params)
                    optimizer.zero_grad(set_to_none=True)
                    _restore_rng(window_start_state["rng"])
                    for i, (b, sid) in enumerate(pending):
                        fwd_out = run_forward(model, b, device,
                                              grad_enabled=True)
                        # Same occurrence-id mapping as pass 2 (P0-1):
                        # only this batch's cuts, resolved via by_oid.
                        z_by_oid = {}
                        batch_terms = []
                        for meta, oid in cut_records[i]:
                            g = g_by_oid.get(oid)
                            if g is not None:
                                z_by_oid[oid] = collect_replay_z(
                                    meta, adapter, device)
                                batch_terms.append((oid, g))
                        adapter.clear()
                        aux, n_aux = batch_surrogate(
                            z_by_oid, batch_terms, 1.0, device)
                        # A microbatch without terms (k < 4) contributes
                        # nothing; its zero aux has no grad_fn, so skip.
                        if n_aux:
                            aux.backward()
                    _restore_rng(resume_rng)
                    g_kf = repr_grad_norm(params)
                    if params_digest(params) != digest0:
                        raise RuntimeError(
                            "lambda calibration changed the trainable "
                            "params — theta_0 violated")
                    r_eff = g_kf / max(g_task, 1e-30)
                    derived = 0.1 / max(r_eff, 1e-30)
                    save_json(out / "calibration.json", {
                        "g_task": g_task, "g_kf": g_kf, "r_eff": r_eff,
                        "derived_lambda": derived,
                        "optimizer_steps": 0, "scheduler_steps": 0,
                        "rule": "lambda = 0.1 / r_eff (plan L5), "
                                "measured on theta_0 before any step"})
                    print(json.dumps({"g_task": g_task, "g_kf": g_kf,
                                      "r_eff": r_eff,
                                      "derived_lambda": derived,
                                      "theta0_verified": True},
                                     indent=2), flush=True)
                    return
                optimizer.zero_grad(set_to_none=True)
                task_sum = 0.0
                n_tokens = 0
                for i, (b, sid) in enumerate(pending):
                    task_mean, task_raw, n_valid, aux, n_terms = pass2_one(
                        b, cut_records[i], g_by_oid,
                        0.0 if args.arm == "gamma_task_only"
                        else lambda_kf)
                    # Per-microbatch normalization: the official Trainer
                    # scales each microbatch MEAN loss by 1/accumulation,
                    # so the gradient scale is window-length independent
                    # (L6.5 review P0-2).  The KF surrogate is NOT
                    # rescaled: it is the exact window-J gradient.
                    loss = task_mean / float(len(pending)) + aux
                    _t = time.perf_counter()
                    scaler.scale(loss).backward()
                    _pf("pass2_bwd", _t)
                    task_sum += float(task_raw.detach())
                    n_tokens += n_valid
                    aux_terms += n_terms
                _t = time.perf_counter()
                grad_step()
                _pf("grad_step", _t)
                if profile:
                    n_cut = sum(1 for cr in cut_records for _ in cr)
                    parts = ["profile win={} n_mb={} n_cut_mb={}:".format(
                        kf_closed + 1, len(pending), n_cut)]
                    for key in ("pass1_fwd", "collect_rows", "window_add",
                                "close_replay", "pass2_fwd", "collect_z",
                                "surrogate", "pass2_bwd", "grad_step"):
                        if PROF.get(key, 0.0) > 0:
                            parts.append("{}={:.1f}s".format(
                                key, PROF[key]))
                    parts.append("wall={:.1f}s".format(
                        time.time() - t_win_start))
                    print(" ".join(parts), flush=True)
                    for key in PROF:
                        PROF[key] = 0.0
                    t_win_start = time.time()
                _restore_rng(resume_rng)  # data stream continues correctly
                total_task_sum += task_sum
                total_tokens += n_tokens
                total_microbatches += len(pending)
                total_kf += float(sum(closed.values()))
                kf_closed += 1
                step += 1
                # Per-window boundary record (review): every arm hashes
                # (window ordinal, n_mb, n_cut_mb) into boundary_hash so
                # cadence identity is checkable window by window, not
                # only through the final data-flow hash.
                n_cut_win = sum(1 for cr in cut_records for _ in cr)
                boundary_hash.update("w{}:{}:{}:".format(
                    kf_closed, len(pending), n_cut_win).encode())
                if args.max_windows and step >= args.max_windows:
                    break
                pending = []
                cut_records = []
                window_start_state = None
            elif len(pending) >= args.max_pending_mbs:
                raise RuntimeError(
                    "degenerate window: {} microbatch collected fewer "
                    "than {} effective cuts".format(
                        len(pending), args.kf_min_cuts))
        else:
            # Review P0-2: the official arm runs on the same dialogue
            # stream and fires on the same adaptive boundary (accumulated
            # effective cuts) as the RPBE arms.  Previously it stepped
            # every grad_accum=128 microbatches while ours/gamma stepped
            # every ~266 (one 128-cut window), so after 1000 steps the
            # first two arms had consumed ~2.07x the task data of
            # ccm_merge — a silent breach of the frozen cadence
            # requirement.  window-matched restores identical task
            # exposure; --merge-cadence official keeps the fixed cadence
            # for the ccm_merge_official reproduction reference only.
            eff = sum(1 for m in metas
                      if m["ok"] and m["k"] >= 4)
            if args.merge_cadence == "window-matched":
                merge_eff_cuts += eff
            fire = (args.merge_cadence == "window-matched"
                    and merge_eff_cuts >= args.kf_min_cuts) \
                or (args.merge_cadence == "official"
                    and len(pending) >= args.grad_accum)
            if fire:
                optimizer.zero_grad(set_to_none=True)
                task_sum = 0.0
                n_tokens = 0
                for b, sid in pending:
                    fwd_out = run_forward(model, b, device, grad_enabled=True)
                    task_raw, n_valid = task_ce_shifted(
                        fwd_out, b["labels"], device)
                    # Official Trainer protocol (verified against
                    # accelerate 1.14 Accelerator.backward + HF 4.44 in
                    # scripts/ccm_parity.py): the per-microbatch MEAN
                    # loss is divided by the window size BEFORE backward
                    # — accelerate does
                    # `loss = loss / self.gradient_accumulation_steps`
                    # inside backward().  Normalizing by len(pending)
                    # makes the task gradient scale independent of the
                    # window length, identical across all three arms.
                    # This division also keeps the fp16 backward on the
                    # safe side of overflow.
                    task_mean = task_raw / max(n_valid, 1)
                    scaler.scale(task_mean / float(len(pending))).backward()
                    task_sum += float(task_raw.detach())
                    n_tokens += n_valid
                grad_step()
                total_task_sum += task_sum
                total_tokens += n_tokens
                total_microbatches += len(pending)
                step += 1
                if args.merge_cadence == "window-matched":
                    # Same per-window boundary record as the RPBE arms
                    # (review): (ordinal, n_mb, n_cut_mb) per window.
                    print("win={} n_mb={} n_cut_mb={} (matched cadence)"
                          .format(step, len(pending), merge_eff_cuts),
                          flush=True)
                    boundary_hash.update("w{}:{}:{}:".format(
                        step, len(pending), merge_eff_cuts).encode())
                    merge_eff_cuts = 0
                if args.max_windows and step >= args.max_windows:
                    break
                pending = []
        if step and step % args.log_every == 0:
            elapsed = time.time() - t_start
            row = {"step": step,
                   "task_ce_token": total_task_sum / max(total_tokens, 1),
                   "task_microbatches": total_microbatches,
                   "task_valid_tokens": total_tokens,
                   "kf_closed": kf_closed,
                   "kf_score": total_kf / max(kf_closed, 1),
                   "aux_terms": aux_terms, "lambda": lambda_kf,
                   "elapsed": elapsed}
            print("step={} ce_token={:.4f} mbs={} kf_closed={} "
                  "kf_score={:.4f} aux={} sec={:.1f}".format(
                      step, row["task_ce_token"], total_microbatches,
                      kf_closed, row["kf_score"], aux_terms, elapsed),
                  flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        if args.checkpoint_every and step and step % args.checkpoint_every == 0:
            save_trainable(out / "checkpoint.pt", model, step=step,
                           optimizer=optimizer.state_dict(),
                           scaler=scaler.state_dict(),
                           builder_oid=builder.next_oid if builder else 0,
                           lambda_kf=lambda_kf,
                           sample_cursor=sample_cursor,
                           rng=_rng_state())

    save_trainable(out / "final.pt", model, step=step, lambda_kf=lambda_kf)
    summary = {
        "arm": args.arm, "seed": args.seed, "steps": step,
        # Token-normalized CE: the only cross-arm comparable task number
        # (L6.5 review; per-microbatch sums vary with the window length).
        "mean_task_ce_per_token": total_task_sum / max(total_tokens, 1),
        "task_microbatches": total_microbatches,
        "task_valid_tokens": total_tokens,
        "kf_closed": kf_closed,
        "mean_kf_score": total_kf / max(kf_closed, 1),
        "aux_terms": aux_terms, "lambda_kf": lambda_kf,
        "paired_seed_hash": paired_seed_hash(args.seed, model)
        if use_rpbe else "n/a",
        "data_flow_hash": data_flow_hash.hexdigest(),
        "data_flow_len": data_flow_len,
        "boundary_hash": boundary_hash.hexdigest(),
    }
    save_json(out / "summary.json", summary)
    save_json(out / "_SUCCESS.json", {"status": "complete", "steps": step})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
