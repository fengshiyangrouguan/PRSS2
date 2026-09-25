#!/usr/bin/env python3
"""gemma MSC R10 (RPBE Stage-2) training — ports llama train_msc.py
--mode train to the gemma host (2026-09-26).

Start: gemma MSC merge ckpt (official adapter over the gemma_MSC_no
foundation; torch ckpt with lora_/comp_embeddings keys).  Attach Gamma
and run the dialogue RPBE window protocol (2Obs, chain-wise cuts,
two-pass replay, rpbe-gamma-only split backward).

Protocol: lambda 0.005936691082765761 (gemma joint precedent — the
llama R10 value carried over, user ruling), lr 3e-5 / rpbe-lr 5e-5,
50 windows, ckpt every 10, warmup ON (schedule 1000).  Dual-card:
seed0 on GPU 0, seed1 on GPU 1.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent / "src"),
          str(HERE.parent / "third_party" / "ccm")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import train_ccm as tc
import eval_msc_gemma as emg
from rpbe.hosts.ccm.adapter import CCMHostAdapter
from rpbe.hosts.ccm.ccm_patch import attach_gamma
from rpbe.llm.dialogue_records import (
    MEM_TAU, DialogueCutBuilder, Llmmaps,
)
from rpbe.llm.utterance_embed import UtteranceEmbed
from rpbe.loss import KFMomentWindow
from rpbe.training.checkpoint import _restore_rng, _rng_state

BASE = "/root/autodl-tmp"
MSC_DATA_DIR = BASE + "/third_party/ccm/dataset/msc"


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    import os
    os.replace(tmp, str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--rpbe-lr", type=float, default=5e-5)
    ap.add_argument("--kf-lambda", type=float,
                    default=0.005936691082765761)
    ap.add_argument("--schedule-total-steps", type=int, default=1000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--kf-min-cuts", type=int, default=128)
    ap.add_argument("--ridge-eps", type=float, default=1e-3)
    ap.add_argument("--sketch-dim", type=int, default=32)
    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--rpbe-seed", type=int, default=0)
    ap.add_argument("--gamma-hidden", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--foundation",
                    default=BASE + "/result/msc/gemma_MSC_no/final.pt")
    ap.add_argument("--msc-adapter", required=True,
                    help="gemma MSC merge ckpt (torch ckpt, "
                         "checkpoint_stepN.pt)")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    seed_all(a.seed)
    device = torch.device("cuda:{}".format(a.gpu))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"

    tok = emg.build_tokenizer()
    model = emg.build_merge(device, a.foundation, a.msc_adapter)
    attach_gamma(model, hidden=a.gamma_hidden)
    cfg = model.model.config

    # Gamma params (walk down to the layer list, gemma text model layout)
    _base = model
    while not hasattr(_base, "layers") and hasattr(_base, "model"):
        _base = _base.model
    gamma_params = []
    for _layer in _base.layers:
        _g = getattr(_layer.self_attn, "gamma", None)
        if _g is not None:
            gamma_params.extend(list(_g.parameters()))
    gamma_set = {id(p) for p in gamma_params}
    print("[msc-gemma-r10] gamma params: {} / layers {}".format(
        len(gamma_params), len(_base.layers)), flush=True)

    from src.data.dialogue.data_msc import DialogueDataset
    from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA
    from src.arguments import CompressionArguments

    ds = DialogueDataset(tok, comp_token=tok.comp_token_id, online=True,
                         add_comp_token=True, eval_source="val",
                         max_length=a.max_length, msc_dir=MSC_DATA_DIR)
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=tc.N_TOK,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    collator = DataCollatorForDialogue_LLAMA(
        dialog=ds, tokenizer=tok, comp_args=comp_args,
        comp_token=tok.comp_token_id, sum_token=tok.sum_token_id,
        padding="left", pad_token=tok.pad_token_id,
        label_pad_token_id=-100)
    rows = list(ds.train_dataset)
    n_rows = len(rows)
    print("[msc-gemma-r10] train rows:", n_rows, flush=True)

    # RPBE structure (gemma heterogeneous adapter; unique KV providers)
    per_layer_dims = [
        int(cfg.per_layer_config[i].num_key_value_heads)
        * int(cfg.per_layer_config[i].head_dim)
        for i in range(cfg.num_hidden_layers)]
    _n_shared = int(getattr(cfg, "num_kv_shared_layers", 0))
    unique_layer_ids = list(range(cfg.num_hidden_layers - _n_shared))
    adapter = CCMHostAdapter(
        model, n_layers=cfg.num_hidden_layers,
        n_heads=cfg.num_key_value_heads, head_dim=0,
        z_dim=a.z_dim, seed=a.rpbe_seed,
        per_layer_dims=per_layer_dims,
        unique_layer_ids=unique_layer_ids)
    maps = Llmmaps(d_chi=64, d_phi=32, m=32,
                   n_branches=Llmmaps.N_BRANCHES,
                   seed=a.rpbe_seed).to(device)
    builder = DialogueCutBuilder(maps, z_dim=a.z_dim, seed=a.rpbe_seed)
    utter_embed = UtteranceEmbed(hidden_dim=cfg.hidden_size, d_chi=64,
                                 seed=a.rpbe_seed, combine_dim=1).to(device)
    phi_embed = UtteranceEmbed(hidden_dim=cfg.hidden_size, d_chi=32,
                               seed=a.rpbe_seed + 100,
                               combine_dim=1).to(device)
    window = tc.BranchEnsembleWindow([
        KFMomentWindow({MEM_TAU: a.z_dim}, min_ratio=2.0,
                       min_abs=a.kf_min_cuts, eps=a.ridge_eps,
                       fixed_maps=maps, strict=False, autoclose=False,
                       oas=True)
        for _ in range(Llmmaps.N_BRANCHES)])
    embed_tokens = model.get_input_embeddings()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in params if id(p) in gamma_set],
          "lr": a.rpbe_lr},
         {"params": [p for p in params if id(p) not in gamma_set],
          "lr": a.lr}],
        lr=a.lr, weight_decay=0.0)
    total_steps = max(1, int(a.schedule_total_steps
                             if a.schedule_total_steps is not None
                             else a.max_steps))
    warmup_steps = max(1, int(0.03 * total_steps))

    def _lr_lambda(s):
        if s < warmup_steps:
            return float(s) / float(warmup_steps)
        p = float(s - warmup_steps) / float(max(1, total_steps
                                                - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    row_order = list(range(n_rows))
    random.shuffle(row_order)
    cursor = 0

    def sample_row():
        nonlocal cursor
        if cursor >= n_rows:
            random.shuffle(row_order)
            cursor = 0
        r = row_order[cursor]
        cursor += 1
        return r

    save_json(out / "config.json", {
        "arm": "ours", "seed": a.seed, "cli": vars(a),
        "n_cut_rows": n_rows,
        "stream": "uniform over the eligible-cut pool",
        "protocol": "gemma MSC R10: lambda 0.00594 (joint precedent), "
                    "lr 3e-5 / rpbe-lr 5e-5, rpbe-gamma-only, 50 windows",
    })

    comp_ids = tok.comp_token_id
    sum_ids = tok.sum_token_id

    def pass1_one(batch, metas):
        batch_cuts = []
        with torch.no_grad():
            tc.run_forward(model, batch, device, grad_enabled=False)
            for meta in metas:
                rows_ = tc.collect_rows(
                    meta, adapter, builder, utter_embed, phi_embed,
                    embed_tokens, batch, device)
                if rows_:
                    window.add(rows_)
                    seen = set()
                    for r in rows_:
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
    lambda_kf = a.kf_lambda
    t_start = time.time()

    while step < a.max_steps:
        pending, cut_records, pass1_rngs = [], [], []
        window_start_state = {"rng": _rng_state()}
        while not window.window_ready():
            row_idx = sample_row()
            row = ds.train_dataset[row_idx]
            batch = collator([dict(row)])
            raw_dialogs = [[list(t["tokens"]) for t in row["dialog"]
                            if t["kind"] == "utt"]]
            metas = tc.parse_meta(batch, comp_ids, sum_ids,
                                  len(pending),
                                  orig_ids=[int(row["orig_id"])],
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

        optimizer.zero_grad(set_to_none=True)
        aux_before = aux_terms
        task_sum = 0.0
        n_tokens = 0
        for i, (b, _metas) in enumerate(pending):
            _restore_rng(pass1_rngs[i])
            out_f = tc.run_forward(model, b, device, grad_enabled=True)
            task_s, n_valid = tc.task_ce_shifted(out_f, b["labels"],
                                                 device)
            task_mean = task_s / max(n_valid, 1)
            z_by_oid = {}
            batch_terms = []
            for meta, oid, v in cut_records[i]:
                g = g_by_oid.get(oid)
                if g is not None:
                    z_by_oid[oid] = tc.collect_replay_z(
                        meta, adapter, device, v=v)
                    batch_terms.append((oid, g))
            adapter.clear()
            aux, n_terms = tc.batch_surrogate(
                z_by_oid, batch_terms, lambda_kf, device)
            if n_terms:
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
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        _restore_rng(resume_rng)

        total_task_sum += task_sum
        total_tokens += n_tokens
        kf_closed += 1
        step += 1
        if step % a.log_every == 0:
            rec = {
                "step": step, "kf_closed": kf_closed,
                "task_ce_token": total_task_sum / max(total_tokens, 1),
                "kf_score": float(sum(closed.values())),
                "aux_terms": aux_terms, "lambda": lambda_kf,
                "elapsed": time.time() - t_start,
            }
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print("[msc-gemma-r10] step={} ce={:.4f} J={:.4f} "
                  "{:.0f}s".format(step, rec["task_ce_token"],
                                   rec["kf_score"], rec["elapsed"]),
                  flush=True)
        if step % a.checkpoint_every == 0:
            torch.save({
                "model": {n: p.detach().cpu()
                          for n, p in model.named_parameters()
                          if p.requires_grad},
                "step": step, "lambda_kf": lambda_kf,
                "rng": _rng_state(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }, out / "ckpt_step{}.pt".format(step))
            print("[msc-gemma-r10] saved ckpt_step{}.pt".format(step),
                  flush=True)
    print("[msc-gemma-r10] TRAIN DONE step={} ce={:.4f}".format(
        step, total_task_sum / max(total_tokens, 1)), flush=True)


if __name__ == "__main__":
    main()
