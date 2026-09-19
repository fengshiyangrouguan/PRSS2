"""Settle the image-preprocessing question empirically.

Renders the sim at a demo's initial_state and compares that frame against the
SAME frame stored in the released hdf5, under identity / hflip / vflip /
rot180. Whichever matches is the transform the training data carries, and
therefore the transform eval must reproduce.

CPU/sim only (no model, no big GPU memory).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(os.environ.get("RPBE_VLA_ROOT", "/root/autodl-tmp"))
LIBERO_ROOT = Path(os.environ.get("LIBERO_MEM_REPO", "/root/autodl-tmp/libero-mem"))
DATA_DIR = Path(os.environ.get("RPBE_DATA_ROOT", "/root/autodl-tmp/datasets/LIBERO-Mem"))
sys.path.insert(0, str(LIBERO_ROOT))

import h5py                                            # noqa: E402
import numpy as np                                     # noqa: E402
from libero.libero.envs import OffScreenRenderEnv      # noqa: E402

# Override with RPBE_TASK to check a different task's hdf5 (e.g. the T3 one).
TASK = os.environ.get(
    "RPBE_TASK",
    "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate")
BDDL = LIBERO_ROOT / "libero/libero/bddl_files/libero_mem" / f"{TASK}.bddl"
H5 = DATA_DIR / f"{TASK}_demo.hdf5"
MI = DATA_DIR / "metainfo.json"

mi = json.load(open(MI))
meta = mi[TASK]["demo_1"]
init_state = np.asarray(meta["initial_state"], dtype=np.float64)

env = OffScreenRenderEnv(bddl_file_name=str(BDDL),
                         camera_heights=256, camera_widths=256)
env.reset()
base = env.env if hasattr(env, "env") else env
try:
    base.reset_subgoal_progress()
    base._overshot = False
except Exception:
    pass
obs = env.set_init_state(init_state)
sim = np.asarray(obs["agentview_image"])
print("sim agentview_image:", sim.shape, sim.dtype)

with h5py.File(H5, "r") as f:
    n_demos = len(f["data"])
    keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
    print(f"hdf5 demos: {n_demos}, first={keys[0]}")
    store = np.asarray(f["data"][keys[0]]["obs"]["agentview_rgb"][0])
print("hdf5 agentview_rgb[0]:", store.shape, store.dtype)

env.close()


def diff(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.abs(a - b).mean())


print()
print("transform     mean|sim - hdf5|   (lower = closer)")
cands = {
    "identity": sim,
    "hflip   ": sim[:, ::-1],
    "vflip   ": sim[::-1, :],
    "rot180  ": sim[::-1, ::-1],
}
res = {k: diff(v, store) for k, v in cands.items()}
for k, v in res.items():
    print(f"  {k}   {v:10.3f}")
best = min(res, key=res.get)
print()
print(f"=> best match: {best.strip()}  (mean abs diff {res[best]:.3f})")
print("   NOTE: exact equality is not expected (JPEG in the demo pipeline,")
print("   sim noise); what matters is which candidate is clearly closest.")
