"""Stage 1 of the T1 checkpoint-restore audit: STRUCTURE ONLY, read-only.

Compares the key layout of
  base  : /root/autodl-tmp/openvla-7b-prismatic/checkpoints/step-295000-....pt
  delta : /root/autodl-tmp/runs/t1_avg_seed42/best.pt
without instantiating any model and without writing anything.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import torch

BASE = Path("/root/autodl-tmp/openvla-7b-prismatic/checkpoints/"
            "step-295000-epoch-40-loss=0.2200.pt")
DELTA = Path("/root/autodl-tmp/runs/t1_avg_seed42/best.pt")


def top_prefixes(keys, depth=1):
    c = collections.Counter()
    for k in keys:
        parts = k.split(".")
        c[".".join(parts[:depth])] += 1
    return c


print("=" * 74)
print("STAGE 1 — structure audit (read-only, no model instantiated)")
print("=" * 74)

# ---------------------------------------------------------------- delta
d = torch.load(DELTA, map_location="cpu", weights_only=False)
print(f"\n[delta] top-level payload keys: {list(d.keys())}")
delta_model = d["model"]
print(f"[delta] payload['model'] is a {type(delta_model).__name__} "
      f"with {len(delta_model)} entries")
dk = list(delta_model.keys())
print("[delta] sample keys:")
for k in dk[:6]:
    print(f"        {k}   {tuple(delta_model[k].shape)}")
print()
print("[delta] first-segment histogram (depth=1):")
for p, n in top_prefixes(dk, 1).most_common():
    print(f"        {p:24s} {n}")
print()
print("[delta] depth=2 histogram:")
for p, n in top_prefixes(dk, 2).most_common(20):
    print(f"        {p:44s} {n}")

print()
print(f"[delta] lora_config in payload: {d.get('lora_config')}")
print(f"[delta] arm={d.get('arm')!r} step={d.get('step')} seed={d.get('seed')}")

lora_keys = [k for k in dk if "lora" in k.lower()]
print(f"[delta] keys containing 'lora': {len(lora_keys)}")
for k in lora_keys[:6]:
    print(f"        {k}   {tuple(delta_model[k].shape)}")
lo = sorted({k.split(".lora_")[1].split(".")[0] for k in lora_keys
             if ".lora_" in k})
print(f"[delta] lora sub-kinds: {lo}")

# ---------------------------------------------------------------- base
print()
print("-" * 74)
b_raw = torch.load(BASE, map_location="cpu", weights_only=False, mmap=True)
print(f"[base] top-level payload keys: {list(b_raw.keys())}")
bm = b_raw["model"]
print(f"[base] payload['model'] is a {type(bm).__name__} "
      f"with {len(bm)} entries")
print("[base] module names inside payload['model']:")
for mod in sorted(bm.keys()):
    sub = bm[mod]
    try:
        n = len(sub)
        example = next(iter(sub)) if n else "<empty>"
    except TypeError:
        n, example = "?", "?"
    print(f"        {mod:24s} n={n:<7} e.g. {example}")

# ------------------------------------------- how do the two line up?
print()
print("-" * 74)
print("MAPPING ANALYSIS")
base_mods = set(bm.keys())
print(f"base module names: {sorted(base_mods)}")

# delta keys are named_parameters() paths on the MemoryVLA instance.
# vlm.* belongs to the PrismaticVLM; everything else is a MemoryVLA attr.
by_seg = collections.Counter()
for k in dk:
    head = k.split(".")[0]
    by_seg[head] += 1
print(f"\ndelta first-segments: {dict(by_seg)}")

vlm_keys = [k for k in dk if k.startswith("vlm.")]
print(f"\nvlm.* delta keys: {len(vlm_keys)}")
print("  vlm sub-module histogram (depth=2 from 'vlm'):")
for p, n in top_prefixes([k[4:] for k in vlm_keys], 1).most_common():
    print(f"        vlm.{p:28s} {n}")

non_vlm = [k for k in dk if not k.startswith("vlm.")]
print(f"\nnon-vlm delta keys: {len(non_vlm)}")
for p, n in top_prefixes(non_vlm, 1).most_common():
    print(f"        {p:28s} {n}")

print()
print("VERDICT (structure):")
inter = sorted(set(p.split(".")[0] for p in
                   [k[4:] for k in vlm_keys]) & base_mods)
print(f"  base modules reachable from delta via 'vlm.<mod>.': {inter}")
print(f"  delta non-vlm modules (MemoryVLA attrs): "
      f"{sorted(set(k.split('.')[0] for k in non_vlm))}")
print("\nNo model constructed, nothing written.")
