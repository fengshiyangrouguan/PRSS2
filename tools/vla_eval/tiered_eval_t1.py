"""tiered_eval_t1.py — tiered_eval.py generalised to an arbitrary LIBERO-Mem task.

`scripts/tiered_eval.py` is the reviewer-fixed evaluator, but it hardcodes
  TASK = "KITCHEN_SCENE1_3_..."      (3 subgoals, hence "tier/3")
and paths from the old VLA box. This keeps its PROTOCOL verbatim and only
parameterises the task and derives the subgoal count from the bddl goal, so it
can run on KITCHEN_SCENE1_1 (T1, 1 subgoal).

Protocol preserved from tiered_eval.py:
  1. NEVER break on env.step()'s `done` — LIBERO-Mem returns True when an
     INTERMEDIATE subgoal is satisfied, which truncates a trajectory that is
     still on track (audit item A2).
  2. `base._check_success(inc=True)` advances the subgoal machine BEFORE each
     env.step (audit item A1: the stock harness never calls it, so
     `_satisfied_subgoals` stays empty and every demo is a structural 0).
  3. `reset_subgoal_progress()` AND `base._overshot = False` per demo — the
     official reset does not clear `_overshot`, so it pollutes across episodes.
  4. Per-demo paired seed (`seed + demo_index`).
  5. Image is `obs["agentview_image"][::-1, :]` handed straight to
     `vla.predict_action` — no get_libero_image, no extra resize.
  6. CHUNK actions predicted, only `--exec` executed; gripper thresholded to
     sim space {-1,+1}.
  7. Reports tiered progress separately from strict success.

Runs in ONE process on the GPU (no Flask): needs an env with both the VLA stack
and the sim stack (env_memvla, once robosuite/mujoco/bddl/gym were added).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("LLAMA2_LOCAL_PATH", "/root/autodl-tmp/Llama-2-7b-hf")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PRSS2_SRC = Path(os.environ.get("PRSS2_SRC", "/root/autodl-tmp/PRSS2/src"))
LIBERO_ROOT = Path(os.environ.get("LIBERO_MEM_REPO", "/root/autodl-tmp/libero-mem"))
MEMVLA = Path(os.environ.get("MEMVLA_DIR",
                            "/root/autodl-tmp/PRSS2/third_party/memoryvla"))
# ORDER MATTERS: libero-mem vendors its own (older) `prismatic` package. Only
# libero-mem has `libero`, only memoryvla has the VLA we trained, so put
# libero-mem first and then memoryvla ON TOP of it so `import prismatic`
# resolves to memoryvla's copy (libero-mem's needs dlimp and is unusable here).
sys.path.insert(0, str(LIBERO_ROOT))
sys.path.insert(0, str(PRSS2_SRC))
sys.path.insert(0, str(MEMVLA))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
from PIL import Image                                 # noqa: E402
from peft import LoraConfig, get_peft_model           # noqa: E402
from libero.libero.envs import OffScreenRenderEnv      # noqa: E402
import libero.libero.envs.bddl_utils as BDDLUtils      # noqa: E402
from vla import load_vla                               # noqa: E402

BASE_DEFAULT = ("/root/autodl-tmp/openvla-7b-prismatic/checkpoints/"
                "step-295000-epoch-40-loss=0.2200.pt")
METAINFO_DEFAULT = "/root/autodl-tmp/datasets/LIBERO-Mem/metainfo.json"
BDDL_DIR = LIBERO_ROOT / "libero/libero/bddl_files/libero_mem"
WAIT = 10
CHUNK = 16


def goal_subgoal_count(bddl_path: Path) -> int:
    """Mirror `_check_success`'s leaf/sequence discrimination."""
    p = BDDLUtils.robosuite_parse_problem(str(bddl_path))
    gs = p["goal_state"]
    head = gs[0][0]
    if head not in ("sequence", "or"):
        return 1                       # leaf goal behaves as a single check
    n = len(gs[0]) - 1
    if n <= 0:
        raise ValueError(f"could not derive subgoal count from {bddl_path}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="KITCHEN_SCENE1_1_pick_up_the_bowl_"
                                      "and_place_it_back_on_the_plate")
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--metainfo", default=METAINFO_DEFAULT)
    ap.add_argument("--demo-start", type=int, default=1)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--exec", type=int, default=8)
    ap.add_argument("--maxsteps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--trace", type=int, default=0,
                    help="1 = log the per-step subgoal state machine")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bddl = BDDL_DIR / f"{args.task}.bddl"
    if not bddl.exists():
        print(f"!! bddl not found: {bddl}")
        return 2
    n_sub = goal_subgoal_count(bddl)
    print("=" * 70)
    print(f"T1 tiered eval | task={args.task}")
    print(f"  bddl        : {bddl}")
    print(f"  subgoals    : {n_sub}   (tier is reported out of this)")
    print(f"  ckpt        : {args.ckpt}")
    print(f"  n={args.n} exec={args.exec}/{CHUNK} maxsteps={args.maxsteps} "
          f"seed={args.seed}")
    print("=" * 70, flush=True)

    # ------------------------------------------------------------- restore
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arm, mem = ck["arm"], ck.get("mem_length", 16)
    print(f"ckpt: arm={arm} step={ck.get('step')} mem_length={mem}", flush=True)
    vla = load_vla(
        model_id_or_path=args.base, hf_token=None, load_for_training=True,
        use_bf16=True, action_dim=7, future_action_window_size=15,
        action_model_type="DiT-L", use_ema=False, dataloader_type="stream",
        mem_length=mem, retrieval_layers=2, use_timestep_pe=True,
        fusion_type="gate", consolidate_type="tome", update_fused=False,
        per_token_size=256, use_rpbe_gamma=(arm in ("gamma-task", "gamma-rpbe")),
        gamma_rank=64, gamma_alpha_init=1.0,
        rpbe_merge_records=(arm != "avg"),
        rpbe_task_grad=(arm in ("gamma-task", "gamma-rpbe")), rpbe_seed=42)
    vla.vlm.requires_grad_(False)
    lc = ck["lora_config"]
    vla.vlm.llm_backbone.llm = get_peft_model(
        vla.vlm.llm_backbone.llm,
        LoraConfig(r=lc["r"], lora_alpha=lc["lora_alpha"],
                   lora_dropout=lc["lora_dropout"], target_modules="all-linear",
                   task_type="CAUSAL_LM"))
    named = dict(vla.named_parameters())
    n_applied = 0
    for n_, t in ck["model"].items():
        if n_ in named and named[n_].requires_grad:
            named[n_].data.copy_(t.to(named[n_].dtype))
            n_applied += 1
    print(f"delta applied: {n_applied}/{len(ck['model'])}", flush=True)

    stats_path = Path(args.ckpt).parent / "dataset_statistics.json"
    with open(stats_path) as f:
        vla.norm_stats = json.load(f)
    print(f"norm_stats keys: {list(vla.norm_stats)}", flush=True)
    vla.eval()

    mi = json.load(open(args.metainfo))
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                             camera_heights=256, camera_widths=256)
    base_env = env.env if hasattr(env, "env") else env

    rows = []
    jsonl = out_dir / f"tiered_{args.task[:30]}_{Path(args.ckpt).stem}.jsonl"
    if jsonl.exists():
        jsonl.unlink()
    for i in range(args.n):
        demo = "demo_%d" % (args.demo_start + i)
        if args.task not in mi or demo not in mi[args.task]:
            print(f"[{demo}] not in metainfo, skipping", flush=True)
            continue
        ep_seed = args.seed + int(demo.split("_")[1])
        torch.manual_seed(ep_seed)
        torch.cuda.manual_seed_all(ep_seed)
        np.random.seed(ep_seed)
        meta = mi[args.task][demo]
        instr = meta["task_description"]
        # Log BEFORE any env work: a hang during reset/init used to be
        # indistinguishable from a hang in the previous demo.
        print(f">>> {demo} starting (seed={ep_seed})", flush=True)
        try:
            env.reset()
            base_env.reset_subgoal_progress()
            base_env._overshot = False
            obs = env.set_init_state(np.asarray(meta["initial_state"],
                                                dtype=np.float64))
            for _ in range(WAIT):
                obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1.0])
        except Exception as e:                     # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            print(f"[{demo}] SETUP FAILED -> {msg}", flush=True)
            rows.append(dict(demo=demo, tier=0, n_sub=n_sub, peak=0,
                             strict=False, complete=False, overshot=False,
                             q=0, steps=0, ep_seed=ep_seed, error=msg))
            with open(jsonl, "a") as f:
                f.write(json.dumps(rows[-1]) + "\n")
            continue

        t = WAIT
        q = 0
        max_sat = 0
        ep_first = "True"
        overshot_ever = False
        trace = []
        while t < args.maxsteps + WAIT:
            try:
                acts, _ = vla.predict_action(
                    image=Image.fromarray(
                        np.ascontiguousarray(obs["agentview_image"][::-1, :])),
                    instruction=instr, unnorm_key=None, use_ddim=True,
                    num_ddim_steps=10, episode_first_frame=ep_first)
            except Exception as e:                 # noqa: BLE001
                print(f"[{demo}] predict_action failed: "
                      f"{type(e).__name__}: {e}", flush=True)
                break
            ep_first = "False"
            q += 1
            acts = np.asarray(acts)
            if acts.shape != (CHUNK, 7):
                print(f"[{demo}] unexpected action shape {acts.shape}; stop",
                      flush=True)
                break
            for j in range(min(args.exec, args.maxsteps + WAIT - t)):
                ok_raw = base_env._check_success(inc=True)   # advance machine (A1)
                sat = len(base_env._satisfied_subgoals)
                if args.trace:
                    trace.append(dict(
                        t=t, pred=bool(ok_raw), n_sat=sat,
                        live=base_env._sub_goal_live_time,
                        nonlive=base_env._sub_goal_nonlive_time,
                        overshot=bool(base_env._overshot)))
                if sat > max_sat:
                    max_sat = sat
                if base_env._overshot:
                    overshot_ever = True
                a = acts[j].copy()
                a[6] = -1.0 if a[6] >= 0.5 else +1.0
                try:
                    # NEVER break on step_done (A2)
                    obs, rew, step_done, info = env.step(a.tolist())
                except Exception as e:             # noqa: BLE001
                    print(f"[{demo}] env.step failed at t={t}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    done_flag = True
                    break
                t += 1

        final_tier = len(base_env._satisfied_subgoals)
        strict = bool(final_tier >= n_sub and not base_env._overshot
                      and base_env._check_success(inc=False))
        row = dict(demo=demo, tier=final_tier, n_sub=n_sub, peak=max_sat,
                   strict=strict, complete=bool(final_tier >= n_sub),
                   overshot=overshot_ever, q=q, steps=t - WAIT,
                   ep_seed=ep_seed)
        rows.append(row)
        with open(jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        if args.trace:
            tp = out_dir / f"trace_{demo}.jsonl"
            with open(tp, "w") as f:
                for r in trace:
                    f.write(json.dumps(r) + "\n")
            # compact summary of the predicate's true/false runs
            runs = []
            for r in trace:
                if not runs or runs[-1][0] != r["pred"]:
                    runs.append([r["pred"], 1])
                else:
                    runs[-1][1] += 1
            print(f"[{demo}] predicate runs (value,len): {runs}", flush=True)
            print(f"[{demo}] trace -> {tp}", flush=True)
        print(f"[{demo}] tier={final_tier}/{n_sub} peak={max_sat} "
              f"complete={row['complete']} strict={strict} "
              f"overshot={overshot_ever} q={q}", flush=True)

    env.close()

    n = len(rows)
    dist = {c: sum(1 for r in rows if r["tier"] == c)
            for c in range(n_sub + 1)}
    n_strict = sum(1 for r in rows if r["strict"])
    n_complete = sum(1 for r in rows if r["complete"])
    n_ov = sum(1 for r in rows if r["overshot"])
    wsum = sum(r["tier"] for r in rows)
    weighted = wsum / (float(n_sub) * max(1, n))

    print("\n==== TIERED EVAL (T1) ====", flush=True)
    print(f"task={args.task} ckpt={Path(args.ckpt).name} arm={arm} "
          f"exec={args.exec}/{CHUNK} n={n} maxsteps={args.maxsteps}", flush=True)
    print(f"tier_dist={dist}", flush=True)
    print(f"weighted_success={weighted * 100:.1f}%  "
          f"complete(=SR)={n_complete}/{n}  strict_success={n_strict}/{n}  "
          f"overshot_demos={n_ov}", flush=True)
    summary = dict(task=args.task, ckpt=args.ckpt, arm=arm, n=n,
                   n_subgoals=n_sub, exec_per_chunk=args.exec, chunk=CHUNK,
                   maxsteps=args.maxsteps, seed=args.seed,
                   tier_dist={str(k): v for k, v in dist.items()},
                   weighted_success=weighted, n_complete=n_complete,
                   n_strict=n_strict, n_overshot=n_ov, rows=rows,
                   ts=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    sp = out_dir / f"tiered_{args.task[:30]}_{Path(args.ckpt).stem}.summary.json"
    sp.write_text(json.dumps(summary, indent=2))
    print(f"summary -> {sp}")
    print("TIERED_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
