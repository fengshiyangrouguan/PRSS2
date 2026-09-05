"""diag_rpbe_embodied.py — lambda calibration via the r_eff gradient-ratio
protocol (plan Task 8; review ruling: the calibration script was missing).

At the CURRENT parameter theta_0 (no optimizer steps), collect one RPBE
window, then for each of the two Gamma loss paths compute the Gamma-param
gradient norm ratio:

    r_eff(lambda=1) = || grad_Gamma( lambda * L_rpbe ) || / || grad_Gamma( L_task ) ||

Report p50 / p95 over windows; the frozen lambda is then
    lambda_rpbe = target_r_eff / median(r_eff(lambda=1)),
with target_r_eff in [0.05, 0.30].

Usage (server):
  HF_HUB_OFFLINE=1 LLAMA2_LOCAL_PATH=/root/autodl-tmp/Llama-2-7b-hf \
  RPBE_EMBODIED_PATH=/root/autodl-tmp \
  python scripts/diag_rpbe_embodied.py \
    --pretrained-checkpoint /root/autodl-tmp/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
    --data-root /root/autodl-tmp/libero-mem --task-filter KITCHEN_SCENE1_3 \
    --n-windows 4 --seed 42
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("RPBE_EMBODIED_PATH", "/root/autodl-tmp"))

from torch.utils.data import DataLoader  # noqa: E402

from vla import load_vla  # noqa: E402
from vla.datasets.hdf5_dataset import (  # noqa: E402
    get_hdf5_decision_stream_dataset_and_collator,
)
from rpbe_embodied import (  # noqa: E402
    EmbodiedFixedMaps, EmbodiedRPBConfig, EmbodiedRPBEWindow,
    PendingMergeQueue, gamma_replay_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--task-filter", default="KITCHEN_SCENE1_3")
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--kf-min-abs", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mem-length", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vla = load_vla(
        model_id_or_path=args.pretrained_checkpoint, hf_token=None,
        load_for_training=True, use_bf16=True, action_dim=7,
        future_action_window_size=15, action_model_type="DiT-L",
        use_ema=False, dataloader_type="stream", mem_length=args.mem_length,
        retrieval_layers=2, use_timestep_pe=True, fusion_type="gate",
        consolidate_type="tome", update_fused=False, per_token_size=256,
        use_rpbe_gamma=True, gamma_rank=64, gamma_alpha_init=1.0,
        rpbe_merge_records=True, rpbe_task_grad=True,
        rpbe_seed=args.seed)
    vla.train()

    tokenizer = vla.vlm.llm_backbone.get_tokenizer()
    image_transform = vla.vlm.vision_backbone.get_image_transform()
    ds, _, collator = get_hdf5_decision_stream_dataset_and_collator(
        data_root=Path(args.data_root), tokenizer=tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=vla.vlm.llm_backbone.prompt_builder_fn,
        future_action_window_size=15, seed=args.seed, split="train",
        pad_token_id=tokenizer.pad_token_id, task_filter=args.task_filter)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0,
                        collate_fn=collator, drop_last=True)

    cfg = EmbodiedRPBConfig(kf_min_abs=args.kf_min_abs, rpbe_seed=args.seed)
    maps = EmbodiedFixedMaps(cfg)
    queue = PendingMergeQueue()
    window = EmbodiedRPBEWindow(variant=cfg.kf_variant,
                                eps=cfg.ridge_eps, min_abs=cfg.kf_min_abs)
    merge_registry, merge_id_map = {}, {}
    task_cotangents, rpbe_cotangents = {}, {}
    last_eid = None
    r_effs = []
    step = 0

    while len(r_effs) < args.n_windows and step < 4000:
        for batch in loader:
            step += 1
            eids = [int(e) for e in batch["episode_ids"]]
            # forward (leaf snapshot before)
            leaf_snapshot = []
            for e, entries in vla.cog_mem_bank.bank.items():
                prov = vla.cog_mem_bank.prov.get(e, [])
                for (_, f), meta in zip(entries, prov):
                    if meta[4] and f.requires_grad:
                        leaf_snapshot.append((f, e, meta[0]))
            pv = batch["pixel_values"]
            if isinstance(pv, dict):
                pv = {k: v.to("cuda", dtype=torch.bfloat16)
                      for k, v in pv.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                loss, _ = vla(
                    input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                    actions=batch["actions"].to("cuda", dtype=torch.bfloat16),
                    action_masks=batch["action_masks"].to("cuda"),
                    pixel_values=pv, labels=batch["labels"].to("cuda"),
                    timesteps=batch["timesteps"],
                    episode_ids=batch["episode_ids"],
                    output_hidden_states=True, repeated_diffusion_steps=4)
            loss.backward()
            for f, e, node_id in leaf_snapshot:
                if f.grad is not None:
                    g = f.grad.detach().clone().reshape(-1)
                    task_cotangents[(e, node_id)] = (
                        task_cotangents.get(
                            (e, node_id), torch.zeros_like(g)) + g)
            vla.cog_mem_bank.refresh_leafs()
            vla.zero_grad()

            # merge plumbing (same order as the trainer)
            for rec in vla.cog_mem_bank.merge_log:
                queue.register(rec)
                merge_registry[(rec.episode_id, rec.node_id)] = rec
                merge_id_map[(rec.episode_id, rec.merge_id)] = rec.node_id
            vla.cog_mem_bank.merge_log = []
            with torch.no_grad():
                vf = vla.vlm.vision_feats.detach()
            instrs = batch["instruction"]
            for i in range(len(eids)):
                ctx = {"vision_feat": vf[i].mean(0).detach().cpu(),
                       "instruction": instrs[i], "delta_s": 1.0,
                       "horizon": 1}
                y = batch["actions"][i].reshape(-1)
                queue.offer(eids[i], int(batch["timesteps"][i]), ctx,
                            y.detach().cpu())
            for e in eids:
                if last_eid is not None and e != last_eid:
                    rows = queue.drain_episode(last_eid)
                    if rows:
                        for r in rows:
                            r.outcome = maps.pv(r.context, r.outcome)
                        window.add(rows)
                last_eid = e

            if window.ready():
                j, g_by_cut, diag = window.close()
                for k, g in g_by_cut.items():
                    e, mid, _ = k
                    nid = merge_id_map.get((e, mid))
                    if nid is not None:
                        rpbe_cotangents[(e, nid)] = g
                # --- r_eff measurement ---
                tkeys = [k for k in task_cotangents if k in merge_registry]
                rkeys = [k for k in rpbe_cotangents if k in merge_registry]
                vla.zero_grad()
                if tkeys:
                    m_a = torch.stack([merge_registry[k].left_state
                                       for k in tkeys]
                                      ).to("cuda", dtype=torch.bfloat16)
                    m_b = torch.stack([merge_registry[k].right_state
                                       for k in tkeys]
                                      ).to("cuda", dtype=torch.bfloat16)
                    gamma_replay_loss(vla.gamma, m_a, m_b, task_cotangents,
                                      tkeys).backward()
                    g_norms = torch.stack([
                        p.grad.norm() for p in vla.gamma.parameters()
                        if p.grad is not None])
                    g_task_norm = g_norms.norm().item()
                else:
                    g_task_norm = float("inf")
                vla.zero_grad()
                if rkeys:
                    m_a_r = torch.stack([merge_registry[k].left_state
                                         for k in rkeys]
                                        ).to("cuda", dtype=torch.bfloat16)
                    m_b_r = torch.stack([merge_registry[k].right_state
                                         for k in rkeys]
                                        ).to("cuda", dtype=torch.bfloat16)
                    gamma_replay_loss(vla.gamma, m_a_r, m_b_r,
                                      rpbe_cotangents, rkeys).backward()
                    r_norms = torch.stack([
                        p.grad.norm() for p in vla.gamma.parameters()
                        if p.grad is not None])
                    g_rpbe_norm = r_norms.norm().item()
                else:
                    g_rpbe_norm = 0.0
                vla.zero_grad()
                if g_task_norm > 0 and math.isfinite(g_task_norm):
                    r_eff = g_rpbe_norm / g_task_norm
                    r_effs.append(r_eff)
                    print(f"[window {len(r_effs)}] J={j:.4f} "
                          f"|grad_task|={g_task_norm:.3e} "
                          f"|grad_rpbe(lam=1)|={g_rpbe_norm:.3e} "
                          f"r_eff={r_eff:.4f}", flush=True)
                task_cotangents, rpbe_cotangents = {}, {}
                window = EmbodiedRPBEWindow(variant=cfg.kf_variant,
                                            eps=cfg.ridge_eps,
                                            min_abs=cfg.kf_min_abs)
                if len(r_effs) >= args.n_windows:
                    break

    if r_effs:
        arr = np.asarray(r_effs)
        med = np.median(arr)
        p95 = np.percentile(arr, 95)
        print(f"\nr_eff over {len(arr)} windows: median={med:.4f} "
              f"p50={np.percentile(arr, 50):.4f} p95={p95:.4f}")
        for target in (0.05, 0.15, 0.30):
            lam = target / max(med, 1e-12)
            print(f"  target_r_eff={target} -> lambda_rpbe={lam:.4e} "
                  f"(scaled p95={p95 * lam / med:.4f})")
    else:
        print("no windows closed; check min_abs / data stream")


if __name__ == "__main__":
    main()
