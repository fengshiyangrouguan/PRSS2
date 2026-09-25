"""MSC Stage-2 (RPBE) training + canonical evaluation — standalone entry.

Frozen MSC protocol (review 2026-09-20) lives in data_msc.py; this
entry owns the two MSC-specific jobs:

  --mode train  RPBE Stage 2: from an MSC CCM-merge checkpoint
                (official Step-2 adapter trained on MSC by the vendored
                train.py), attach Gamma and train with the dialogue
                RPBE protocol (2Obs, chain-wise cuts, window-matched
                optimizer clock).  1000 optimizer updates, microbatch 1
                (two-pass replay protocol), macrobatch = window size.
  --mode eval   Canonical PPL evaluation:
                  * session-opening PPL  (first predicted response of
                    each session s = 2..5; MSC-paper Session Openings)
                  * all-response PPL     (every response in s = 2..5)
                arms: full (raw context) / merge (MSC CCM-merge ckpt) /
                ours (Stage-2 ckpt).  Reported per session + overall.

Training stream: UNIFORM sampling over the eligible-cut pool (one row
per (episode, cut) in data_msc's train_dataset — long episodes get
their natural weight; no depth buckets, no episode-row bias).  The
same RNG stream drives every arm (seed_all), so paired arms see the
identical sample sequence.

The official aggregate + rpbe_gamma_only structure (llama-line frozen
spec, R10) is preserved: task CE and the RPBE surrogate backward
SEPARATELY per closed window; non-Gamma params keep the pure task
gradient, Gamma keeps task + lambda * J.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/scripts")
sys.path.insert(0, "/root/autodl-tmp/third_party/ccm")

import torch

from scripts.train_ccm import (
    N_TOK, BranchEnsembleWindow, collect_replay_z, collect_rows,
    parse_meta, run_forward, save_trainable, seed_all, task_ce_rows,
    task_ce_shifted, batch_surrogate,
)
from rpbe.hosts.ccm.adapter import CCMHostAdapter
from rpbe.hosts.ccm.ccm_patch import attach_gamma
from rpbe.llm.dialogue_records import (
    MEM_TAU, DialogueCutBuilder, Llmmaps,
)
from rpbe.llm.utterance_embed import UtteranceEmbed
from rpbe.loss import KFMomentWindow
from rpbe.training.checkpoint import _restore_rng, _rng_state

MSC_DATA_DIR = "/root/autodl-tmp/third_party/ccm/dataset/msc"
EVAL_MAX_LENGTH = 3584   # full 5-session context + comp/sep/marker overhead


def parse_args():
    p = argparse.ArgumentParser("MSC RPBE Stage-2 + canonical eval")
    p.add_argument("--mode", required=True, choices=["train", "eval"])
    p.add_argument("--arm", default="ours",
                   choices=["ours", "gamma_task_only", "task_only"])
    p.add_argument("--model-name-or-path", default="/root/autodl-tmp/llama-7b-hf")
    p.add_argument("--msc-adapter", default="",
                   help="MSC CCM-merge adapter dir (c*_merge); required "
                        "for train / merge / ours arms")
    p.add_argument("--foundation", default="",
                   help="optional Step-1 LoRA dir to merge (MSC skips Step 1)")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--schedule-total-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rpbe-lr", type=float, default=5e-5)
    p.add_argument("--rpbe-gamma-only", action="store_true",
                   help="R10 structure: RPBE gradient on Gamma only "
                        "(llama-line frozen spec)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--kf-lambda", type=float, default=1e-3)
    p.add_argument("--calibrate-lambda", action="store_true")
    p.add_argument("--kf-min-cuts", type=int, default=128)
    p.add_argument("--ridge-eps", type=float, default=1e-3)
    p.add_argument("--sketch-dim", type=int, default=32)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--rpbe-seed", type=int, default=0)
    p.add_argument("--gamma-hidden", type=int, default=64)
    p.add_argument("--relative-embedding", default="skip")
    p.add_argument("--max-length", type=int, default=2048,
                   help="training cut-window token budget")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--resume-from", default="")
    # eval mode
    p.add_argument("--eval-split", default="test", choices=["val", "test"])
    p.add_argument("--eval-arm", default="all",
                   choices=["full", "merge", "ours", "all"])
    p.add_argument("--eval-batch", type=int, default=4)
    p.add_argument("--eval-ckpt", default="",
                   help="Stage-2 checkpoint for the ours arm")
    p.add_argument("--opening-only", action="store_true",
                   help="eval only the session-opening exchanges "
                        "(canonical primary metric; used for c*_merge "
                        "selection — all-response PPL is computed later "
                        "on the selected checkpoint only)")
    p.add_argument("--eval-sessions", default="2,3,4,5",
                   help="session ids to evaluate (c*_merge selection "
                        "uses 2,3,4 — S5 is held out)")
    p.add_argument("--max-episodes", type=int, default=0,
                   help="evaluate a FIXED random subset of episodes "
                        "(seed 1234); selection-only speedup — the "
                        "official table always runs the full split")
    return p.parse_args()


def save_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, str(path))


# ----------------------------------------------------------------------
# host construction (self-contained llama path; MSC skips Step 1 by
# default — the official adapter is trained from the raw llama-7b-hf)
# ----------------------------------------------------------------------
def build_msc_host(args, device):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from src.arch.ccm_llama import LlamaForCausalLM_CCM
    from src.model import load_lora_weight, peft_custom
    from src.utils import SeparatedEmbedding
    from peft import LoraConfig
    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.comp_relative_embedding = args.relative_embedding
    # fp32 host (official separate-embed path; see train_ccm.build_official_host)
    model = LlamaForCausalLM_CCM.from_pretrained(
        args.model_name_or_path, config=config, torch_dtype=torch.float32)
    model = model.to(device)
    if args.foundation:
        load_lora_weight(args.foundation, model, merge=True)
        print("[msc-host] merged Step-1 foundation {}".format(
            args.foundation), flush=True)
    model.update_comp_token([32000 + k for k in range(N_TOK)],
                            [32000 + N_TOK + k for k in range(N_TOK)])
    model.model.embed_tokens = SeparatedEmbedding(
        model.model.embed_tokens, 2 * N_TOK)
    adapter_dir = args.msc_adapter
    lora_cfg = LoraConfig().from_pretrained(adapter_dir)
    model = peft_custom.get_peft_model(model, lora_cfg)
    load_lora_weight(adapter_dir, model, merge=False)
    for _p in model.parameters():
        _p.requires_grad_(False)
    model.base_model.model.model.embed_tokens.comp_embeddings.weight \
        .requires_grad_(True)
    for _n, _p in model.named_parameters():
        if "lora_" in _n:
            _p.requires_grad_(True)
    model._official_host = True
    return model


def build_tokenizer(args):
    from transformers import LlamaTokenizer
    tok = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None \
        else tok.eos_token_id
    tok.bos_token_id = tok.bos_token_id or 1
    tok.eos_token_id = tok.eos_token_id or 2
    tok.padding_side = "left"
    added = ["<COMP{}>".format(k) for k in range(N_TOK)] \
        + ["<SUM{}>".format(k) for k in range(N_TOK)]
    tok.add_special_tokens({"additional_special_tokens": added})
    ids = tok.additional_special_tokens_ids[-2 * N_TOK:]
    tok.comp_token_id = ids[:N_TOK]
    tok.sum_token_id = ids[N_TOK:]
    return tok


def build_data(tok, args, max_length):
    from src.data.dialogue.data_msc import DialogueDataset
    return DialogueDataset(
        tok, comp_token=list(tok.comp_token_id), online=True,
        add_comp_token=True, eval_source="val", max_length=max_length,
        msc_dir=MSC_DATA_DIR)


def build_collator(ds, tok, device_type):
    from src.arguments import CompressionArguments
    from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=N_TOK,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    return DataCollatorForDialogue_LLAMA(
        dialog=ds, tokenizer=tok, comp_args=comp_args,
        comp_token=tok.comp_token_id, sum_token=tok.sum_token_id,
        padding="left", pad_token=tok.pad_token_id, label_pad_token_id=-100)


# ----------------------------------------------------------------------
# training (window-matched optimizer clock; cut-uniform stream)
# ----------------------------------------------------------------------
def train(args):
    seed_all(args.seed)
    assert args.arm in ("ours", "gamma_task_only"), \
        "train mode runs the RPBE Stage 2 (ours / gamma_task_only); " \
        "the CCM-merge baseline needs no Stage 2"
    assert args.msc_adapter, "--msc-adapter required in train mode"
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"

    tok = build_tokenizer(args)
    model = build_msc_host(args, device)
    use_rpbe = args.arm in ("ours", "gamma_task_only")
    if use_rpbe:
        attach_gamma(model, hidden=args.gamma_hidden)
    cfg = model.model.config
    gamma_params = []
    if use_rpbe:
        _base = model
        while not hasattr(_base, "layers") and hasattr(_base, "model"):
            _base = _base.model
        for _layer in _base.layers:
            _g = getattr(_layer.self_attn, "gamma", None)
            if _g is not None:
                gamma_params.extend(list(_g.parameters()))
    gamma_set = {id(p) for p in gamma_params}

    assert getattr(cfg, "attention_dropout", 0.0) == 0.0 \
        and getattr(cfg, "hidden_dropout", 0.0) == 0.0, \
        "two-pass replay requires zero model dropout"

    adapter = maps = builder = utter_embed = window = None
    if use_rpbe:
        n_heads = cfg.num_attention_heads
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        adapter = CCMHostAdapter(model, n_layers=cfg.num_hidden_layers,
                                 n_heads=n_heads, head_dim=head_dim,
                                 z_dim=args.z_dim, seed=args.rpbe_seed)
        maps = Llmmaps(d_chi=64, d_phi=32, m=32,
                       n_branches=Llmmaps.N_BRANCHES,
                       seed=args.rpbe_seed).to(device)
        builder = DialogueCutBuilder(maps, z_dim=args.z_dim,
                                     seed=args.rpbe_seed)
        utter_embed = UtteranceEmbed(hidden_dim=cfg.hidden_size, d_chi=64,
                                     seed=args.rpbe_seed,
                                     combine_dim=1).to(device)
        phi_embed = UtteranceEmbed(hidden_dim=cfg.hidden_size, d_chi=32,
                                   seed=args.rpbe_seed + 100).to(device)
        window = BranchEnsembleWindow([
            KFMomentWindow({MEM_TAU: args.z_dim}, min_ratio=2.0,
                           min_abs=args.kf_min_cuts, eps=args.ridge_eps,
                           fixed_maps=maps, strict=False, autoclose=False,
                           oas=True)
            for _ in range(Llmmaps.N_BRANCHES)])

    params = [p for p in model.parameters() if p.requires_grad]
    if args.rpbe_lr is not None and gamma_params:
        gids = {id(p) for p in gamma_params}
        optimizer = torch.optim.AdamW(
            [{"params": [p for p in params if id(p) in gids],
              "lr": args.rpbe_lr},
             {"params": [p for p in params if id(p) not in gids],
              "lr": args.lr}],
            lr=args.lr, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = max(1, int(args.schedule_total_steps
                             if args.schedule_total_steps is not None
                             else args.max_steps))
    warmup_steps = max(1, int(0.03 * total_steps))

    def _lr_lambda(s):
        if s < warmup_steps:
            return float(s) / float(warmup_steps)
        progress = float(s - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and not getattr(
            model, "_official_host", False)))

    ds = build_data(tok, args, max_length=args.max_length)
    collator = build_collator(ds, tok, device)
    comp_ids = tok.comp_token_id
    sum_ids = tok.sum_token_id
    embed_tokens = model.get_input_embeddings()

    # uniform eligible-cut pool: one row per (episode, cut)
    n_rows = len(ds.train_dataset)
    row_order = list(range(n_rows))
    random.shuffle(row_order)
    cursor = 0

    def sample_row():
        nonlocal cursor
        if cursor >= n_rows:                     # next epoch
            random.shuffle(row_order)
            cursor = 0
        r = row_order[cursor]
        cursor += 1
        return r

    save_json(out / "config.json", {
        "arm": args.arm, "seed": args.seed, "cli": vars(args),
        "n_cut_rows": n_rows,
        "stream": "uniform over the eligible-cut pool (one row per "
                  "(episode, cut); reshuffled per epoch",
    })

    # ---- window protocol (identical to the llama-line frozen spec) ----
    def pass1_one(batch, metas):
        batch_cuts = []
        with torch.no_grad():
            run_forward(model, batch, device, grad_enabled=False)
            for meta in metas:
                rows = collect_rows(meta, adapter, builder, utter_embed,
                                    phi_embed, embed_tokens, batch, device)
                if rows:
                    window.add(rows)
                    seen = set()
                    for r in rows:
                        if r.occurrence_id in seen:
                            continue
                        seen.add(r.occurrence_id)
                        batch_cuts.append(
                            (meta, r.occurrence_id,
                             int(r.context["cut_turn"])))
            adapter.clear()
        return batch_cuts

    step = 0
    kf_closed = 0
    aux_terms = 0
    total_task_sum = 0.0
    total_tokens = 0
    lambda_kf = args.kf_lambda
    t_start = time.time()

    while step < args.max_steps:
        pending, cut_records, pass1_rngs = [], [], []
        window_start_state = {"rng": _rng_state()}
        while not window.window_ready():
            row_idx = sample_row()
            row = ds.train_dataset[row_idx]
            batch = collator([dict(row)])
            # raw_dialogs: tokenized utterance turns of the cut window
            # (markers excluded — structural metadata, not dialogue
            # content).  Length == L + 2: L history turns + context +
            # target, matching collect_rows' assertion.  Fix 2026-09-26:
            # the MSC path never passed raw_dialogs, so meta["raw_dialog"]
            # was None and collect_rows crashed on len(None).
            raw_dialogs = [[list(t["tokens"]) for t in row["dialog"]
                            if t["kind"] == "utt"]]
            metas = parse_meta(batch, comp_ids, sum_ids,
                               len(pending), orig_ids=[int(row["orig_id"])],
                               raw_dialogs=raw_dialogs)
            pending.append((batch, metas))
            state = {"rng": _rng_state()}
            pass1_rngs.append(state["rng"])
            if any(m["ok"] and m["k"] >= 3 for m in metas):
                cut_records.append(pass1_one(batch, metas))
            else:
                cut_records.append([])
            _restore_rng(state["rng"])

        closed, plan, diag = window.close_replay()
        g_by_oid = plan.get(MEM_TAU, {}).get("by_oid", {})
        n_cut_win = sum(1 for cr in cut_records for _ in cr)
        if len(g_by_oid) != n_cut_win:
            raise RuntimeError(
                "RPBE replay incomplete: plan={} cuts={}".format(
                    len(g_by_oid), n_cut_win))
        resume_rng = _rng_state()
        _restore_rng(window_start_state["rng"])

        if args.calibrate_lambda and step == 0:
            # r_eff calibration at theta_0 (frozen spec, llama-line rule):
            # task and RPBE gradient norms on the GAMMA group, measured
            # on the FIRST window before any optimizer step;
            # lambda = 0.1 / r_eff is then committed for the real run.
            def _gamma_norm():
                tot = 0.0
                for p in gamma_params:
                    if p.grad is not None:
                        tot += float((p.grad.detach().float() ** 2).sum())
                return tot ** 0.5
            optimizer.zero_grad(set_to_none=True)
            _restore_rng(window_start_state["rng"])
            for i, (b, _metas) in enumerate(pending):
                _restore_rng(pass1_rngs[i])
                out_f = run_forward(model, b, device, grad_enabled=True)
                task_s, n_valid = task_ce_shifted(out_f, b["labels"], device)
                (task_s / max(n_valid, 1)
                 / float(len(pending))).backward()
            g_task = _gamma_norm()
            optimizer.zero_grad(set_to_none=True)
            _restore_rng(window_start_state["rng"])
            for i, (b, _metas) in enumerate(pending):
                _restore_rng(pass1_rngs[i])
                out_f = run_forward(model, b, device, grad_enabled=True)
                z_by_oid = {}
                batch_terms = []
                for meta, oid, v in cut_records[i]:
                    g = g_by_oid.get(oid)
                    if g is not None:
                        z_by_oid[oid] = collect_replay_z(meta, adapter,
                                                         device, v=v)
                        batch_terms.append((oid, g))
                adapter.clear()
                aux, n_aux = batch_surrogate(z_by_oid, batch_terms,
                                             1.0, device)
                if n_aux:
                    aux.backward()
            g_kf = _gamma_norm()
            r_eff = g_kf / max(g_task, 1e-30)
            derived = 0.1 / max(r_eff, 1e-30)
            save_json(out / "calibration.json", {
                "g_task_gamma": g_task, "g_kf_gamma": g_kf,
                "r_eff_gamma": r_eff, "derived_lambda": derived,
                "rule": "lambda = 0.1 / r_eff on the Gamma group at "
                        "theta_0 (first window)"})
            print(json.dumps({"g_task_gamma": g_task,
                              "g_kf_gamma": g_kf,
                              "r_eff_gamma": r_eff,
                              "derived_lambda": derived}, indent=2),
                  flush=True)
            return

        optimizer.zero_grad(set_to_none=True)
        aux_before = aux_terms
        task_sum = 0.0
        n_tokens = 0
        for i, (b, _metas) in enumerate(pending):
            _restore_rng(pass1_rngs[i])            # P0: mask == pass 1
            out_f = run_forward(model, b, device, grad_enabled=True)
            task_s, n_valid = task_ce_shifted(out_f, b["labels"], device)
            task_mean = task_s / max(n_valid, 1)
            z_by_oid = {}
            batch_terms = []
            for meta, oid, v in cut_records[i]:
                g = g_by_oid.get(oid)
                if g is not None:
                    z_by_oid[oid] = collect_replay_z(meta, adapter,
                                                     device, v=v)
                    batch_terms.append((oid, g))
            adapter.clear()
            aux, n_terms = batch_surrogate(
                z_by_oid, batch_terms,
                0.0 if args.arm == "gamma_task_only" else lambda_kf,
                device)
            if args.rpbe_gamma_only and n_terms:
                # R10 structure: task and RPBE backwards SPLIT; non-Gamma
                # params keep the pure task gradient.
                scaler.scale(task_mean / float(len(pending))).backward(
                    retain_graph=True)
                task_snap = {id(p): p.grad.detach().clone()
                             for p in params
                             if p.grad is not None
                             and id(p) not in gamma_set}
                scaler.scale(aux).backward()
                for p in params:
                    if id(p) not in gamma_set:
                        p.grad = task_snap.get(id(p))
            elif n_terms:
                scaler.scale(task_mean / float(len(pending))
                             + aux).backward()
            else:
                scaler.scale(task_mean / float(len(pending))).backward()
            task_sum += float(task_s.detach())
            n_tokens += n_valid
            aux_terms += n_terms
        if aux_terms - aux_before != n_cut_win:
            raise RuntimeError(
                "not every cut was replayed exactly once: "
                "replayed={} cuts={}".format(aux_terms - aux_before,
                                             n_cut_win))
        # grad step (clip + scaler + scheduler)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        _restore_rng(resume_rng)

        total_task_sum += task_sum
        total_tokens += n_tokens
        kf_closed += 1
        step += 1
        if step % args.log_every == 0:
            j_win = float(sum(closed.values()))
            rec = {
                "step": step, "kf_closed": kf_closed,
                "task_ce_token": total_task_sum / max(total_tokens, 1),
                "task_microbatches": sum(len(cr) for cr in cut_records)
                + len(pending),
                "kf_score": j_win, "aux_terms": aux_terms,
                "lambda": lambda_kf,
                "elapsed": time.time() - t_start,
            }
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print("[msc] step={} ce={:.4f} J={:.4f} "
                  "{:.0f}s".format(step, rec["task_ce_token"], j_win,
                                   rec["elapsed"]), flush=True)
        if step % args.checkpoint_every == 0:
            ckpt = {
                "model": {n: p.detach().cpu()
                          for n, p in model.named_parameters()
                          if p.requires_grad},
                "step": step, "lambda_kf": lambda_kf,
                "rng": _rng_state(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }
            torch.save(ckpt, out / "ckpt_step{}.pt".format(step))
            print("[msc] saved ckpt_step{}.pt".format(step), flush=True)
        if step >= args.max_steps:
            break
    print("[msc] TRAIN DONE step={} ce={:.4f}".format(
        step, total_task_sum / max(total_tokens, 1)), flush=True)


# ----------------------------------------------------------------------
# canonical evaluation
# ----------------------------------------------------------------------
def eval_full_collate(tok, rows, ds_full):
    """Raw-context arm: plain left-pad, no comp tokens, no comp mask."""
    inputs, labels = [], []
    for r in rows:
        inst = ds_full.sample_dialog(r)
        full = inst["input_ids"] + inst["output_ids"]
        inputs.append(full)
        labels.append([-100] * len(inst["input_ids"]) + inst["output_ids"])
    pad = tok.pad_token_id
    L = max(len(x) for x in inputs)
    batch = {
        "input_ids": torch.tensor(
            [[pad] * (L - len(x)) + x for x in inputs], dtype=torch.long),
        "labels": torch.tensor(
            [[-100] * (L - len(x)) + x for x in labels], dtype=torch.long),
        "attention_mask": torch.tensor(
            [[0] * (L - len(x)) + [1] * len(x) for x in inputs],
            dtype=torch.long),
    }
    return batch


def eval(args):
    seed_all(args.seed)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    tok = build_tokenizer(args)
    ds = build_data(tok, args, max_length=EVAL_MAX_LENGTH)
    from src.data.dialogue.data_msc import DialogueDataset as _MSC
    ds_full = _MSC(tok, comp_token=[], online=True, add_comp_token=False,
                   eval_source="val", max_length=EVAL_MAX_LENGTH,
                   msc_dir=MSC_DATA_DIR)
    collator = build_collator(ds, tok, device)
    split = "valid" if args.eval_split == "val" else "test"
    sessions = tuple(int(s) for s in args.eval_sessions.split(","))
    instances = list(ds.exchange_instances(split, sessions=sessions))
    if args.opening_only:
        instances = [r for r in instances if r["is_opening"]]
    if args.max_episodes > 0:
        rng = random.Random(1234)          # fixed subset across ckpts
        eids = sorted({r["orig_id"] for r in instances})
        keep = set(rng.sample(
            eids, min(args.max_episodes, len(eids))))
        instances = [r for r in instances if r["orig_id"] in keep]
    print("[msc-eval] {} exchanges on {} (opening_only={} "
          "sessions={} max_episodes={})".format(
              len(instances), split, args.opening_only,
              sessions, args.max_episodes), flush=True)

    arms = (["full", "merge", "ours"] if args.eval_arm == "all"
            else [args.eval_arm])

    for arm in arms:
        if arm == "full":
            model = None
            from transformers.models.llama.configuration_llama import \
                LlamaConfig
            from src.arch.ccm_llama import LlamaForCausalLM_CCM
            cfg = LlamaConfig.from_pretrained(args.model_name_or_path)
            cfg.comp_relative_embedding = args.relative_embedding
            model = LlamaForCausalLM_CCM.from_pretrained(
                args.model_name_or_path, config=cfg,
                torch_dtype=torch.float16).to(device)
            # the CCM arch's position/mask path requires comp tokens to
            # be declared even for the raw full-context arm (fix
            # 2026-09-26: comp_mask=None crashed update_position_ids);
            # the full arm's data carries no comp ids -> zero mask
            model.update_comp_token([32000, 32001], [32002, 32003])
            from src.utils import SeparatedEmbedding
            model.model.embed_tokens = SeparatedEmbedding(
                model.model.embed_tokens, 4)
            model.eval()
        elif arm == "merge":
            assert args.msc_adapter, "--msc-adapter required for the merge arm"
            model = build_msc_host(args, device)
            model.eval()
        else:  # ours
            assert args.msc_adapter, "--msc-adapter required for the ours arm"
            model = build_msc_host(args, device)
            assert args.eval_ckpt, "--eval-ckpt required for the ours arm"
            attach_gamma(model, hidden=args.gamma_hidden)
            payload = torch.load(args.eval_ckpt, map_location=device,
                                 weights_only=False)
            _m, _u = model.load_state_dict(payload["model"], strict=False)
            if _u:
                raise RuntimeError("unexpected keys: {}".format(
                    sorted(_u)[:5]))
            _bad = [k for k in sorted(set(_m)) if "gamma" not in k]
            if _bad:
                raise RuntimeError("ckpt missing trainable keys: {}".format(
                    _bad[:5]))
            model.eval()
        print("[msc-eval] arm={} ready".format(arm), flush=True)

        acc = {}
        for s in range(2, 6):
            acc[s] = {"opening_sum": 0.0, "opening_n": 0,
                      "all_sum": 0.0, "all_n": 0}
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(instances), args.eval_batch):
                rows = instances[i:i + args.eval_batch]
                metas = [(int(r["session"]), int(r["is_opening"]))
                         for r in rows]
                if arm == "full":
                    batch = eval_full_collate(tok, rows, ds_full)
                    out_f = model(input_ids=batch["input_ids"].to(device),
                                  attention_mask=batch["attention_mask"]
                                  .to(device))
                else:
                    batch = collator(rows)
                    out_f = run_forward(model, batch, device, False)
                row_sum, row_n = task_ce_rows(out_f, batch["labels"], device)
                for r in range(row_sum.shape[0]):
                    s, o = metas[r]
                    acc[s]["all_sum"] += float(row_sum[r])
                    acc[s]["all_n"] += int(row_n[r])
                    if o:
                        acc[s]["opening_sum"] += float(row_sum[r])
                        acc[s]["opening_n"] += int(row_n[r])
                if i % (args.eval_batch * 50) == 0 and i:
                    print("[msc-eval] {} {}/{} {:.0f}s".format(
                        arm, i, len(instances), time.time() - t0),
                        flush=True)
        result = {"arm": arm, "split": split, "n_instances": len(instances),
                  "sessions": {}}
        for s in range(2, 6):
            result["sessions"]["S{}".format(s)] = {
                "opening_ppl": float(torch.exp(torch.tensor(
                    acc[s]["opening_sum"] / max(acc[s]["opening_n"], 1)))),
                "opening_tokens": acc[s]["opening_n"],
                "all_ppl": float(torch.exp(torch.tensor(
                    acc[s]["all_sum"] / max(acc[s]["all_n"], 1)))),
                "all_tokens": acc[s]["all_n"],
            }
        osum = sum(acc[s]["opening_sum"] for s in range(2, 6))
        on = sum(acc[s]["opening_n"] for s in range(2, 6))
        asum = sum(acc[s]["all_sum"] for s in range(2, 6))
        an = sum(acc[s]["all_n"] for s in range(2, 6))
        result["overall_opening_ppl"] = float(torch.exp(
            torch.tensor(osum / max(on, 1))))
        result["overall_all_ppl"] = float(torch.exp(
            torch.tensor(asum / max(an, 1))))
        path = Path(args.output) / "eval_{}_{}.json".format(arm, split)
        save_json(path, result)
        print("[msc-eval] {} done {:.0f}s -> {}".format(
            arm, time.time() - t0, path), flush=True)
        print(json.dumps(result, indent=2), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    a = parse_args()
    if a.mode == "train":
        train(a)
    else:
        eval(a)
