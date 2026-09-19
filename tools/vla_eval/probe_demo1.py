"""Why does demo_1 score tier=0 even though it looks like it succeeds?

Sim-only probe (no model): set the sim to demo_1's initial_state and print the
task predicate ("bowl on plate") plus the subgoal counters at t=0 and over a
few dummy steps. If the predicate is already TRUE at t=0, then "do nothing"
keeps it TRUE forever, `_sub_goal_nonlive_time` never reaches the >15 needed by
the sequence branch, and the subgoal can never be registered -- i.e. tier=0 is
the protocol's intended answer for a trajectory that never *changes* the goal
state, not a bug in the evaluator.

Compare with demo_2 (which DID register) to see the difference.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

LIBERO_ROOT = Path("/root/autodl-tmp/libero-mem")
sys.path.insert(0, str(LIBERO_ROOT))

import numpy as np                                     # noqa: E402
from libero.libero.envs import OffScreenRenderEnv      # noqa: E402

TASK = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
BDDL = LIBERO_ROOT / "libero/libero/bddl_files/libero_mem" / f"{TASK}.bddl"
MI = Path("/root/autodl-tmp/datasets/LIBERO-Mem/metainfo.json")
mi = json.load(open(MI))

env = OffScreenRenderEnv(bddl_file_name=str(BDDL),
                         camera_heights=256, camera_widths=256)
base = env.env if hasattr(env, "env") else env

for demo in ("demo_1", "demo_2"):
    meta = mi[TASK][demo]
    env.reset()
    base.reset_subgoal_progress()
    base._overshot = False
    env.set_init_state(np.asarray(meta["initial_state"], dtype=np.float64))
    print(f"\n=== {demo} ===")
    print(f"  t=0  predicate(inc=False) = {base._check_success(inc=False)}")
    print(f"  t=0  n_satisfied={len(base._satisfied_subgoals)} "
          f"live={base._sub_goal_live_time} nonlive={base._sub_goal_nonlive_time}")
    # 20 dummy no-op steps, advancing the machine exactly as the evaluator does
    for k in range(20):
        base._check_success(inc=True)
        env.step([0, 0, 0, 0, 0, 0, -1.0])
    print(f"  after 20 no-op steps: "
          f"n_satisfied={len(base._satisfied_subgoals)} "
          f"live={base._sub_goal_live_time} nonlive={base._sub_goal_nonlive_time} "
          f"overshot={base._overshot}")

env.close()
print("\nINTERPRETATION: if demo_1's predicate is True at t=0 and nonlive stays "
      "0, the sequence branch's `live>5 AND nonlive>15` test can never fire, "
      "so tier=0 is structural for a trajectory that never removes the bowl "
      "from the plate.")
