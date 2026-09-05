"""Task 0 probe: inspect LIBERO-Mem HDF5 schema / splits / dtypes (READ-ONLY).

Usage:
  python scripts/probe_libero_mem_hdf5.py --data-root /path/to/libero-mem [--out probe_libero_mem.json]

Answers (plan Task 0 checklist):
  1. file naming & per-file demo groups
  2. train/val split source (metainfo.json or filename/group-index convention;
     verify NO trajectory overlap)
  3. dtypes & encodings (agentview_rgb dtype, dones semantics, gripper vs action dim 7)
  4. language instruction: per-demo or task-level
  5. total sizes / frame counts
  6. oracle keys (success/subgoal/boxes) present but must NOT enter training
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="probe_libero_mem.json")
    ap.add_argument("--max-demos", type=int, default=3,
                    help="demos to open per file (schema probing only)")
    args = ap.parse_args()

    h5_files = sorted(glob.glob(os.path.join(args.data_root, "*.hdf5")))
    meta_path = os.path.join(args.data_root, "metainfo.json")
    report = {"files": [], "metainfo": None, "checks": {}}

    # --- metainfo.json (language instructions + oracle fields) ---
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        report["metainfo"] = {
            "type": type(meta).__name__,
            "top_keys": list(meta.keys())[:20] if isinstance(meta, dict) else None,
            "n_entries": len(meta) if hasattr(meta, "__len__") else None,
            "sample": None,
        }
        if isinstance(meta, dict):
            k0 = list(meta.keys())[0]
            sample = meta[k0]
            if isinstance(sample, dict):
                report["metainfo"]["sample"] = {kk: type(vv).__name__
                                                for kk, vv in list(sample.items())[:12]}
        elif isinstance(meta, list) and meta:
            report["metainfo"]["sample"] = {kk: type(vv).__name__
                                            for kk, vv in list(meta[0].items())[:12]}
    else:
        report["checks"]["metainfo_exists"] = False

    # --- per h5 file: groups, shapes, dtypes ---
    total_demos = 0
    total_bytes = 0
    for hf in h5_files:
        size = os.path.getsize(hf)
        total_bytes += size
        info = {"file": os.path.basename(hf), "size_mb": round(size / 1e6, 1)}
        try:
            with h5py.File(hf, "r") as f:
                demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
                n_demos = len(demo_keys)
                total_demos += n_demos
                info["n_demos"] = n_demos
                info["demo_key_example"] = demo_keys[0] if demo_keys else None
                # open a few demos
                for dk in demo_keys[: args.max_demos]:
                    g = f["data"][dk]
                    dinfo = {}
                    for k in g.keys():
                        v = g[k]
                        if isinstance(v, h5py.Dataset):
                            dinfo[k] = {"shape": v.shape, "dtype": str(v.dtype)}
                        else:
                            dinfo[k] = {"group_keys": list(v.keys())[:10]}
                    info.setdefault("demo_schema", dinfo)
                # also inspect one demo deeper: obs sub-datasets
                g = f["data"][demo_keys[0]]
                obs = g.get("obs")
                if obs is not None and isinstance(obs, h5py.Group):
                    info["obs_keys"] = {k: (obs[k].shape, str(obs[k].dtype))
                                        for k in obs.keys()}
        except Exception as e:  # noqa: BLE001
            info["error"] = repr(e)
        report["files"].append(info)

    report["totals"] = {"n_files": len(h5_files),
                        "n_demos_total": total_demos,
                        "total_gb": round(total_bytes / 1e9, 1)}

    # --- split convention: check metainfo for train/val markers ---
    report["checks"]["split_hint"] = "inspect metainfo.json sample entry for train/val fields"
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("probe written ->", args.out)
    print("files:", len(h5_files), "| demos:", total_demos,
          "| total GB:", round(total_bytes / 1e9, 1))


if __name__ == "__main__":
    main()
