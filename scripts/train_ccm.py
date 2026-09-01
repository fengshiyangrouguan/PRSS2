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
import json
import math
import os
import random
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
                   choices=["ccm_merge", "gamma_task_only", "ours"])
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--dialog-mirror", required=True,
                   help="DIALOG_MIRROR: ijcnlp_dailydialog layout dir")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--grad-accum", type=int, default=128,
                   help="microbatch per official step (ccm_merge arm)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=5.0)
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
    p.add_argument("--max-pending-mbs", type=int, default=2048,
                   help="degenerate-window guard: pending cap before abort")
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
    # fp16 autocast matches the official Trainer protocol: the vendored
    # conditional-LoRA layer computes its lora branch in fp32, so the
    # forward MUST run under autocast (mixed fp16/fp32 without autocast
    # raises on the fp16 base path).
    ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
    with ctx:
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            return model(input_ids=batch["input_ids"].to(device),
                         attention_mask=batch["attention_mask"].to(device))


def task_ce(out, labels, device):
    return torch.nn.functional.cross_entropy(
        out.logits.view(-1, out.logits.shape[-1]),
        labels.to(device).view(-1), ignore_index=-100)


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


def batch_surrogate(z_rows_by_oid, plan_by_batch, batch_offset, lam, device):
    """Numerically zero surrogate; gradient = exact window J (plan L5).

    K = 1 here (one optimizer step per closed window), so the auxiliary
    is simply -lambda * sum_v (<sg(g_v), z_v> - sg(<g_v, z_v>))."""
    terms = []
    for oid, g in plan_by_batch[batch_offset]:
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


def main():
    args = parse_args()
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
                       seed=args.rpbe_seed)
        builder = DialogueCutBuilder(maps, z_dim=args.z_dim,
                                     seed=args.rpbe_seed)
        utter_embed = UtteranceEmbed(hidden_dim=cfg.hidden_size, d_chi=64,
                                     seed=args.rpbe_seed).to(device)
        window = KFMomentWindow({MEM_TAU: args.z_dim}, min_ratio=2.0,
                                min_abs=args.kf_min_cuts,
                                eps=args.ridge_eps, fixed_maps=maps,
                                strict=False, autoclose=False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
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
    total_task = 0.0
    total_kf = 0.0
    kf_closed = 0
    aux_terms = 0
    below_threshold = 0
    lambda_kf = args.kf_lambda
    t_start = time.time()
    pending = []
    window_start_state = None

    def pass1_one(batch, meta_list):
        with torch.no_grad():
            run_forward(model, batch, device, grad_enabled=False)
            for meta in meta_list:
                rows = collect_rows(meta, adapter, builder, utter_embed,
                                    embed_tokens, batch, device)
                if rows:
                    window.add(rows)
            adapter.clear()

    def pass2_one(batch, meta_list, plan_by_batch, i, lam):
        out = run_forward(model, batch, device, grad_enabled=True)
        task = task_ce(out, batch["labels"], device)
        aux = torch.zeros((), device=device)
        n_terms = 0
        if plan_by_batch is not None:
            z_by_oid = {}
            for meta in meta_list:
                for r in collect_rows(meta, adapter, builder, utter_embed,
                                      embed_tokens, batch, device):
                    z_by_oid[r.occurrence_id] = r.z
            adapter.clear()
            aux, n_terms = batch_surrogate(z_by_oid, plan_by_batch, i, lam,
                                           device)
        return task, aux, n_terms

    def grad_step():
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    while step < args.max_steps:
        batch, sample_id = next_batch()
        if window_start_state is None:
            window_start_state = {"rng": _rng_state(),
                                  "next_oid": builder.next_oid
                                  if builder else 0}
        pending.append((batch, sample_id))
        if use_rpbe:
            # Incremental pass 1: RNG restored after each microbatch
            # (builder counters stay monotonic).
            state = {"rng": _rng_state()}
            metas = parse_meta(batch, comp_ids, sum_ids, sample_id)
            pass1_one(batch, metas)
            _restore_rng(state["rng"])
            if window.window_ready():
                closed, plan, diag = window.close_replay()
                _restore_rng(window_start_state["rng"])
                if builder:
                    builder.next_oid = window_start_state["next_oid"]
                plan_by_batch = plan.get(MEM_TAU, {}).get("by_batch", [])
                plan_by_batch = [plan_by_batch[i] if i < len(plan_by_batch)
                                 else [] for i in range(len(pending))]
                optimizer.zero_grad(set_to_none=True)
                for i, (b, sid) in enumerate(pending):
                    metas = parse_meta(b, comp_ids, sum_ids, sid)
                    task, aux, n_terms = pass2_one(
                        b, metas, plan_by_batch, i,
                        0.0 if args.arm == "gamma_task_only" else lambda_kf)
                    loss = task + aux
                    scaler.scale(loss).backward()
                    total_task += float(task.detach())
                    aux_terms += n_terms
                grad_step()
                total_kf += float(sum(closed.values()))
                kf_closed += 1
                if args.calibrate_lambda and kf_closed == 1:
                    # r_eff on this window: separate norm measurements.
                    optimizer.zero_grad(set_to_none=True)
                    _restore_rng(window_start_state["rng"])
                    builder.next_oid = window_start_state["next_oid"]
                    for b, sid in pending:
                        metas = parse_meta(b, comp_ids, sum_ids, sid)
                        out = run_forward(model, b, device,
                                          grad_enabled=True)
                        task = task_ce(out, b["labels"], device)
                        task.backward()
                    g_task = repr_grad_norm(params)
                    optimizer.zero_grad(set_to_none=True)
                    _restore_rng(window_start_state["rng"])
                    builder.next_oid = window_start_state["next_oid"]
                    for i, (b, sid) in enumerate(pending):
                        metas = parse_meta(b, comp_ids, sum_ids, sid)
                        out = run_forward(model, b, device,
                                          grad_enabled=True)
                        z_by_oid = {}
                        for meta in metas:
                            for r in collect_rows(
                                    meta, adapter, builder, utter_embed,
                                    embed_tokens, b, device):
                                z_by_oid[r.occurrence_id] = r.z
                        adapter.clear()
                        aux, _ = batch_surrogate(z_by_oid, plan_by_batch,
                                                 i, 1.0, device)
                        aux.backward()
                    g_kf = repr_grad_norm(params)
                    r_eff = g_kf / max(g_task, 1e-30)
                    derived = 0.1 / max(r_eff, 1e-30)
                    save_json(out / "calibration.json", {
                        "g_task": g_task, "g_kf": g_kf, "r_eff": r_eff,
                        "derived_lambda": derived,
                        "rule": "lambda = 0.1 / r_eff (plan L5)"})
                    print(json.dumps({"g_task": g_task, "g_kf": g_kf,
                                      "r_eff": r_eff,
                                      "derived_lambda": derived},
                                     indent=2), flush=True)
                    return
                step += 1
                pending = []
                window_start_state = None
            elif len(pending) >= args.max_pending_mbs:
                raise RuntimeError(
                    "degenerate window: {} microbatch collected fewer "
                    "than {} effective cuts".format(
                        len(pending), args.kf_min_cuts))
        else:
            if len(pending) >= args.grad_accum:
                optimizer.zero_grad(set_to_none=True)
                for b, sid in pending:
                    out = run_forward(model, b, device, grad_enabled=True)
                    task = task_ce(out, b["labels"], device)
                    scaler.scale(task).backward()
                    total_task += float(task.detach())
                grad_step()
                step += 1
                pending = []
        if step and step % args.log_every == 0:
            elapsed = time.time() - t_start
            row = {"step": step, "task": total_task / step,
                   "kf_closed": kf_closed,
                   "kf_score": total_kf / max(kf_closed, 1),
                   "aux_terms": aux_terms, "lambda": lambda_kf,
                   "elapsed": elapsed}
            print("step={} task={:.4f} kf_closed={} kf_score={:.4f} "
                  "aux={} sec={:.1f}".format(
                      step, row["task"], kf_closed, row["kf_score"],
                      aux_terms, elapsed), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        if args.checkpoint_every and step and step % args.checkpoint_every == 0:
            torch.save({"step": step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()},
                       out / "checkpoint.pt")

    torch.save({"step": step, "model": model.state_dict()},
               out / "final.pt")
    summary = {
        "arm": args.arm, "seed": args.seed, "steps": step,
        "mean_task_loss": total_task / max(step, 1),
        "kf_closed": kf_closed,
        "mean_kf_score": total_kf / max(kf_closed, 1),
        "aux_terms": aux_terms, "lambda_kf": lambda_kf,
        "paired_seed_hash": paired_seed_hash(args.seed, model)
        if use_rpbe else "n/a",
    }
    save_json(out / "summary.json", summary)
    save_json(out / "_SUCCESS.json", {"status": "complete", "steps": step})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
