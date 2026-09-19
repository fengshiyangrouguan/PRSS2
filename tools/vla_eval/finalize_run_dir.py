"""Finalize the T1 run directory so the LIBERO rollout can load it.

`load_vla` asserts that the checkpoint's run dir contains BOTH `config.json`
and `dataset_statistics.json`. The trainer only ever writes
`dataset_statistics.json` (at the very end, in a flat `{q01,q99,mask}` shape
that `MemoryVLA.get_action_stats` cannot read), and never writes `config.json`.

This script fixes that, using the ACTUAL T1 training config recovered from the
checkpoint itself (every `_ckpt_dict` payload carries `arm`, `task_filter`,
`seed`, `mem_length`, `lambda_rpbe` and a `config` block), and the ACTUAL
BOUNDS_Q99 statistics recomputed with the trainer's own `compute_action_stats`
on the T1 train split.

Nothing here borrows an OXE key or an OXE statistic.

Run with the TRAINING env (env_memvla): it needs torch/h5py/numpy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RPBE_VLA_ROOT", "/root/autodl-tmp"))
RUN_DIR = Path(os.environ.get("RPBE_RUN_DIR", str(ROOT / "runs/t1_avg_seed42")))
DATA_ROOT = Path(os.environ.get("RPBE_DATA_ROOT", str(ROOT / "datasets/LIBERO-Mem")))
OPENVLA_CFG = Path(os.environ.get("RPBE_OPENVLA_CFG",
                                  str(ROOT / "openvla-7b-prismatic/config.json")))
MEMVLA = Path(os.environ.get(
    "MEMVLA_DIR", str(ROOT / "PRSS2/third_party/memoryvla")))
TASK_FILTER = os.environ.get("RPBE_TASK_FILTER", "KITCHEN_SCENE1_1")
# eval_libero.sh builds its key as f"{task_suite_name}_no_noops"; the suite is
# `libero_mem`, so the canonical T1 key is exactly this.
UNNORM_KEY = "libero_mem_no_noops"

# deploy.py only exposes 11 CLI flags; every other `load_vla` kwarg has to come
# from `<run_dir>/config.yaml`. These mirror `build_vla()` in
# train_libero_mem_rpbe.py exactly for the `avg` arm (no Gamma, no RPBE).
LOAD_VLA_KWARGS = {
    "use_bf16": True,
    "action_dim": 7,
    "future_action_window_size": 15,
    "action_model_type": "DiT-L",
    "use_ema": False,
    "dataloader_type": "stream",
    "mem_length": 16,
    "retrieval_layers": 2,
    "use_timestep_pe": True,
    "fusion_type": "gate",
    "consolidate_type": "tome",
    "update_fused": False,
    "per_token_size": 256,
    "use_rpbe_gamma": False,      # arm == avg
    "gamma_rank": 64,
    "gamma_alpha_init": 1.0,
    "rpbe_merge_records": False,  # arm == avg
    "rpbe_task_grad": False,      # arm == avg
    "rpbe_seed": 42,
}



def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(RUN_DIR))
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--task-filter", default=TASK_FILTER)
    ap.add_argument("--unnorm-key", default=UNNORM_KEY)
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written, touch nothing")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_root = Path(args.data_root)

    print("=" * 70)
    print("T1 run-dir finalize")
    print("=" * 70)
    print(f"run_dir     : {run_dir}")
    print(f"data_root   : {data_root}")
    print(f"task_filter : {args.task_filter}")
    print(f"unnorm_key  : {args.unnorm_key}")
    print()

    # ---------------------------------------------------------------- 1) ckpt
    ckpts = sorted(run_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        print("!! no .pt in run dir yet -- training has not written one")
        return 2
    print("checkpoints present:")
    for p in ckpts:
        print(f"   {p.name:22s} {p.stat().st_size/1e9:6.2f} GB  "
              f"mtime={p.stat().st_mtime:.0f}")
    # prefer best.pt (the one we intend to roll out), else the newest
    ref = next((p for p in ckpts if p.name == "best.pt"), ckpts[-1])
    print(f"\nrecovering training config from: {ref.name}")

    import torch
    payload = torch.load(ref, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        print("!! unexpected checkpoint payload type")
        return 2

    train_cfg = {}
    for k in ("arm", "step", "micro_step", "mem_length", "lambda_rpbe",
              "seed", "task_filter", "best_val", "episodes_seen",
              "param_version", "lora_config"):
        if k in payload:
            v = payload[k]
            train_cfg[k] = v
    inner = payload.get("config", {})
    print("\nrecovered T1 training config (from the checkpoint itself):")
    print(json.dumps({**train_cfg, "config": inner}, indent=2, default=str))

    tf_in_ckpt = train_cfg.get("task_filter")
    if tf_in_ckpt and tf_in_ckpt != args.task_filter:
        print(f"\n!! WARNING: checkpoint task_filter={tf_in_ckpt!r} does not "
              f"match --task-filter={args.task_filter!r}")
    print(f"\nckpt sha256: {sha256_file(ref)[:16]}...  ({ref.name})")

    # ------------------------------------------------------- 2) config.json
    # The architecture block must be the one training actually loaded: the
    # base OpenVLA prismatic config. We keep `vla` verbatim and ADD explicit
    # T1 provenance so this file is never mistaken for the OXE run.
    with open(OPENVLA_CFG) as f:
        base_cfg = json.load(f)

    cfg_out = dict(base_cfg)
    cfg_out["run_id"] = "libero_mem_t1_avg"
    cfg_out["run_id_note"] = (
        f"{args.task_filter}|arm={train_cfg.get('arm')}|"
        f"seed={train_cfg.get('seed')}"
    )
    vla_block = dict(base_cfg.get("vla", {}))
    vla_block["data_mix"] = f"{args.task_filter}_no_noops"
    cfg_out["vla"] = vla_block
    cfg_out["t1_provenance"] = {
        "note": ("config.json reconstructed for the T1 LIBERO-Mem rollout; the "
                 "`vla` block is the base architecture training loaded, "
                 "everything under `t1_training` is recovered from the "
                 "checkpoint, NOT from the OXE release."),
        "dataset": "libero-mem/LIBERO-Mem",
        "task_filter": args.task_filter,
        "unnorm_key": args.unnorm_key,
        "reference_checkpoint": ref.name,
        "reference_checkpoint_sha256": sha256_file(ref),
        "t1_training": {**train_cfg, "config": inner},
    }

    # -------------------------------------------- 3) dataset_statistics.json
    # Recompute BOUNDS_Q99 with the trainer's OWN function on the T1 TRAIN
    # split (demo_1..demo_80), then wrap it in the nested layout that
    # MemoryVLA.get_action_stats expects: norm_stats[key]["action"]["q01"].
    sys.path.insert(0, str(MEMVLA))
    from vla.datasets.hdf5_dataset import scan_episodes, compute_action_stats

    episodes, ranges = scan_episodes(data_root, "train",
                                     task_filter=args.task_filter)
    print(f"\nT1 train episodes for stats: {len(episodes)} "
          f"(demo range {ranges})")
    if not episodes:
        print("!! no train episodes found -- check --data-root/--task-filter")
        return 2

    stats = compute_action_stats(episodes)
    n_frames = sum(ep["T"] for ep in episodes)
    print(f"frames contributing to stats: {n_frames}")
    print(f"q01  = {[round(float(x), 5) for x in stats['q01']]}")
    print(f"q99  = {[round(float(x), 5) for x in stats['q99']]}")
    print(f"mask = {[bool(x) for x in stats['mask']]}")

    stats_out = {
        args.unnorm_key: {
            "action": {
                "q01": [float(x) for x in stats["q01"]],
                "q99": [float(x) for x in stats["q99"]],
                "mask": [bool(x) for x in stats["mask"]],
            },
            "num_transitions": int(n_frames),
            "num_trajectories": int(len(episodes)),
            "source": {
                "dataset": "libero-mem/LIBERO-Mem",
                "task_filter": args.task_filter,
                "split": "train (demo_1..demo_80)",
                "estimator": "compute_action_stats (BOUNDS_Q99, "
                             "gripper relabeled to {0,1}, mask[6]=False)",
            },
        }
    }

    # ---------------------------------------------------------------- write
    cfg_path = run_dir / "config.json"
    stats_path = run_dir / "dataset_statistics.json"

    if args.dry_run:
        print("\n--- DRY RUN, nothing written ---")
        print(f"would write {cfg_path}:")
        print(json.dumps(cfg_out, indent=2)[:1200] + "\n   ...(truncated)")
        print(f"\nwould write {stats_path}:")
        print(json.dumps(stats_out, indent=2))
        return 0

    for p in (cfg_path, stats_path):
        if p.exists():
            bak = p.with_suffix(p.suffix + ".bak")
            shutil.copy2(p, bak)
            print(f"backed up existing {p.name} -> {bak.name}")

    with open(cfg_path, "w") as f:
        json.dump(cfg_out, f, indent=2)
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\nwrote {cfg_path}")
    print(f"wrote {stats_path}")

    # ------------------------------------------- 4) deploy.py's expected layout
    # deploy.py does dirname(dirname(saved_model_path)) to find config.yaml, and
    # load_vla asserts checkpoint_pt.parent.name == "checkpoints". So the server
    # must be pointed at <run_dir>/checkpoints/<ckpt>.pt, with <run_dir>
    # holding config.yaml.
    import yaml

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    deploy_ckpt = ckpt_dir / ref.name
    if deploy_ckpt.exists() or deploy_ckpt.is_symlink():
        deploy_ckpt.unlink()
    os.symlink(ref.resolve(), deploy_ckpt)
    print(f"linked {deploy_ckpt} -> {ref.resolve()}")

    # The yaml supplies the load_vla kwargs deploy.py's argparse does not expose.
    # CLI args win over yaml in deploy.py's deep_update, so keep it to model
    # construction only.
    yaml_path = run_dir / "config.yaml"
    yaml_payload = dict(LOAD_VLA_KWARGS)
    yaml_payload["future_action_window_size"] = LOAD_VLA_KWARGS[
        "future_action_window_size"]
    yaml_payload["rpbe_seed"] = int(train_cfg.get("seed", 42))
    yaml_payload["mem_length"] = int(train_cfg.get("mem_length", 16))
    arm = train_cfg.get("arm")
    if arm != "avg":
        print(f"!! WARNING: checkpoint arm={arm!r}; the RPBE kwargs below "
              f"assume 'avg'. Adjust LOAD_VLA_KWARGS for gamma arms.")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_payload, f, default_flow_style=False, sort_keys=False)
    print(f"wrote {yaml_path}")
    print(json.dumps(yaml_payload, indent=2))

    # ------------------------------------------------------------ validation
    print("\n=== validation ===")
    with open(run_dir / "config.json") as f:
        c = json.load(f)
    print("config.json has vla.base_vlm:", c["vla"]["base_vlm"])
    with open(run_dir / "dataset_statistics.json") as f:
        s = json.load(f)
    assert args.unnorm_key in s, "unnorm key missing!"
    print("dataset_statistics key present:", list(s))
    a = s[args.unnorm_key]["action"]
    print("action keys:", list(a))
    assert len(a["q01"]) == len(a["q99"]) == len(a["mask"]) == 7
    print("action dims OK (7)")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
