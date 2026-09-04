#!/usr/bin/env python3
"""Pure-aux one-step experiment (dkf investigation, review ruling).

Loads an ``ours`` checkpoint, collects ONE 128-cut window, records
J_before, applies ONE pure-aux gradient update (no task CE, no clipping,
no GradScaler) with SGD step sizes eta in a small scan, and re-collects
the SAME window data to record J_after per eta.

Verdict criterion: J_after < J_before (the single-step mechanism lowers
the Ky-Fan defect on the real fp16 model path).  If J rises on every
eta, the z-extraction / Gamma-update chain has a hidden problem.
"""
import argparse
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import train_ccm as tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-name-or-path", default="/root/autodl-tmp/llama-7b-hf")
    ap.add_argument("--dialog-mirror",
                    default="/root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-mbs", type=int, default=600,
                    help="max microbatches to collect one window")
    ap.add_argument("--etas", default="3e-4,1e-3,3e-3,1e-2")
    a = ap.parse_args()

    device = torch.device("cuda", a.gpu)
    args = types.SimpleNamespace(
        arm="ours", seed=0, model_name_or_path=a.model_name_or_path,
        dialog_mirror=a.dialog_mirror, relative_embedding="skip",
        lora_r=8, z_dim=128, rpbe_seed=0, sketch_dim=64, gamma_hidden=64,
        kf_min_cuts=128, ridge_eps=1e-3, kf_lambda=1.0, max_pending_mbs=2048,
        grad_clip=1.0, lr=3e-4, max_steps=1000, grad_accum=128,
        merge_cadence="window-matched", checkpoint_every=0, log_every=10,
        calibrate_lambda=False, resume_from="", output=Path("."))
    etas = [float(x) for x in a.etas.split(",")]

    tokenizer = tc.build_tokenizer(args)
    model = tc.build_model(args, device)
    model = tc.wrap_lora(model, args.lora_r)
    model.update_comp_token([32000 + k for k in range(tc.N_TOK)],
                            [32000 + tc.N_TOK + k for k in range(tc.N_TOK)])
    tc.attach_gamma(model, hidden=args.gamma_hidden)
    cfg = model.model.config

    adapter = tc.CCMHostAdapter(model, n_layers=cfg.num_hidden_layers,
                                n_heads=cfg.num_attention_heads,
                                head_dim=cfg.hidden_size // cfg.num_attention_heads,
                                z_dim=args.z_dim, seed=args.rpbe_seed)
    maps = tc.Llmmaps(d_chi=64, d_phi=32, m=args.sketch_dim,
                      seed=args.rpbe_seed).to(device)
    builder = tc.DialogueCutBuilder(maps, z_dim=args.z_dim,
                                    seed=args.rpbe_seed)
    utter_embed = tc.UtteranceEmbed(hidden_dim=cfg.hidden_size, d_chi=64,
                                    seed=args.rpbe_seed).to(device)
    embed_tokens = model.get_input_embeddings()

    _dummy_opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    tc.load_trainable(a.checkpoint, model, _dummy_opt, device)
    print("loaded checkpoint: {}".format(a.checkpoint), flush=True)

    dialog, collator = tc.build_dataset(args, tokenizer)
    comp_ids = tokenizer.comp_token_id
    sum_ids = tokenizer.sum_token_id
    train_items = dialog.train_dataset
    n_items = len(train_items)
    cursor = 0

    def next_batch():
        nonlocal cursor
        item = train_items[int(cursor % n_items)]
        cursor += 1
        return collator([item]), cursor - 1

    def pass1(batch, meta_list, win):
        """no-grad forward + collect rows into win (train_ccm pass1_one)."""
        with torch.no_grad():
            tc.run_forward(model, batch, device, grad_enabled=False)
            for meta in meta_list:
                rows = tc.collect_rows(meta, adapter, builder, utter_embed,
                                       embed_tokens, batch, device)
                if rows:
                    win.add(rows)
            adapter.clear()

    def collect_one_window():
        win = tc.KFMomentWindow({tc.MEM_TAU: args.z_dim}, min_ratio=2.0,
                                min_abs=args.kf_min_cuts, eps=args.ridge_eps,
                                fixed_maps=maps, strict=False,
                                autoclose=False)
        batches = []
        metas_all = []
        for _ in range(a.max_mbs):
            batch, sample_id = next_batch()
            metas = tc.parse_meta(batch, comp_ids, sum_ids, sample_id)
            batches.append(batch)
            metas_all.append(metas)
            if any(m["ok"] and m["k"] >= 4 for m in metas):
                pass1(batch, metas, win)
            if win.window_ready():
                break
        if not win.window_ready():
            raise RuntimeError("window never became ready within {} mbs"
                               .format(a.max_mbs))
        closed, plan, diag = win.close_replay()
        g_by_oid = plan.get(tc.MEM_TAU, {}).get("by_oid", {})
        print("window: {} mbs, {} cuts, J={:.6f} diag_failed={}".format(
            len(batches), len(g_by_oid), closed[tc.MEM_TAU],
            diag[tc.MEM_TAU].get("failed")), flush=True)
        return win, batches, metas_all, float(closed[tc.MEM_TAU]), g_by_oid

    # ---- round 1: J_before + per-oid gradients ----
    win1, batches, metas_all, j_before, g_by_oid = collect_one_window()
    n_oids = len(g_by_oid)

    # ---- oid alignment ----
    # Round 1 emitted occurrence ids monotonically per cut in collect
    # order (builder.next_oid only advances inside collect_rows), so the
    # k-th cut in the same-order replay sequence corresponds to the k-th
    # g of the plan (by_oid preserves cut_ids_list first-appearance
    # order == collect order).
    def aux_step_and_remeasure(eta):
        # snapshot
        snapshot = {n: p.detach().clone()
                    for n, p in model.named_parameters() if p.requires_grad}
        for p in model.parameters():
            if p.requires_grad:
                p.grad = None
        # plan gradients in emission order
        g_list = list(g_by_oid.values())
        cut_idx = 0
        for batch, metas in zip(batches, metas_all):
            tc.run_forward(model, batch, device, grad_enabled=True)
            for meta in metas:
                if not (meta["ok"] and meta["k"] >= 4):
                    continue
                if cut_idx >= len(g_list):
                    continue
                z = tc.collect_replay_z(meta, adapter, device)
                gd = g_list[cut_idx].to(device)
                # numerically-zero surrogate at lam=1
                ((gd * z).sum() - (gd * z.detach()).sum()).backward()
                cut_idx += 1
            adapter.clear()
        assert cut_idx == n_oids, "replay alignment lost: {} != {}".format(
            cut_idx, n_oids)
        # manual SGD, no clip, no scaler
        gnorm = 0.0
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    gnorm += float((p.grad.detach().double() ** 2).sum())
                    p.sub_(eta * p.grad.detach())
        gnorm = gnorm ** 0.5
        # round 2: same batches, fresh window, no-grad pass 1
        win2 = tc.KFMomentWindow({tc.MEM_TAU: args.z_dim}, min_ratio=2.0,
                                 min_abs=args.kf_min_cuts, eps=args.ridge_eps,
                                 fixed_maps=maps, strict=False, autoclose=False)
        for batch, metas in zip(batches, metas_all):
            if any(m["ok"] and m["k"] >= 4 for m in metas):
                pass1(batch, metas, win2)
        closed2, _, _ = win2.close_replay()
        j_after = float(closed2[tc.MEM_TAU])
        # restore snapshot for the next eta
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(snapshot[n])
        return j_after, gnorm

    print("\neta      | J_before  | J_after   | dJ       | grad_norm",
          flush=True)
    for eta in etas:
        j_after, gnorm = aux_step_and_remeasure(eta)
        ok = "DECREASE" if j_after < j_before else "INCREASE"
        print("{:.1e} | {:.6f} | {:.6f} | {:+.6f} | {:.4e}  {}".format(
            eta, j_before, j_after, j_after - j_before, gnorm, ok),
            flush=True)

    print(json.dumps({"j_before": j_before, "n_oids": n_oids}), flush=True)


if __name__ == "__main__":
    main()
