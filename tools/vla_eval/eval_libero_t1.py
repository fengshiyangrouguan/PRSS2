"""T1-only LIBERO-Mem rollout evaluator.

Why this exists instead of using `script/eval/libero/eval_libero.sh`:

  * the stock evaluator has NO `libero_mem` branch in its hardcoded `max_steps`
    dict, so it NameErrors on our suite;
  * it loops over every task in the suite and dies on the first task whose
    `.pruned_init` is absent (only 4 of the 10 are shipped);
  * it waits a blanket `sleep 1800` for the policy server instead of polling.

This one evaluates exactly one task, takes its init states from an explicit
(configurable) path, polls the policy server for readiness with a hard timeout,
and writes one JSONL record per episode plus a summary with SR / SGC.

CPU-side: importing this module and running with --dry-run touches no
simulator and no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# osmesa is the stock choice and works here; allow an override.
os.environ.setdefault("MUJOCO_GL", "osmesa")

DEFAULT_TASK = "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate"
DEFAULT_INIT_DIR = Path("/root/autodl-tmp/libero-mem/libero/libero/init_files/libero_mem")
INIT_FALLBACK_DIR = Path("/root/autodl-tmp/libero-mem/scripts/__init_data")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------- readiness
def tcp_ready(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_server(host: str, port: int, timeout_s: float,
                    poll_s: float = 3.0, proc=None) -> float:
    """Poll until the policy server accepts connections.

    `deploy.py` constructs the model BEFORE calling `app.run()`, so the port
    only starts listening once the weights are loaded -- a successful TCP
    connect is therefore a valid readiness signal, not just "process alive".
    """
    t0 = time.time()
    last_note = 0.0
    while True:
        elapsed = time.time() - t0
        if tcp_ready(host, port):
            return elapsed
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"policy server exited (rc={proc.returncode}) after "
                f"{elapsed:.0f}s before becoming ready")
        if elapsed >= timeout_s:
            raise TimeoutError(
                f"policy server on {host}:{port} not ready after "
                f"{timeout_s:.0f}s")
        if elapsed - last_note >= 30:
            last_note = elapsed
            print(f"   ...waiting for policy server ({elapsed:.0f}s "
                  f"elapsed)", flush=True)
        time.sleep(poll_s)


# ------------------------------------------------------------------- tasks
def resolve_task(task_suite, task_name: str | None, task_id: int | None):
    n = task_suite.n_tasks
    if task_id is not None:
        if not (0 <= task_id < n):
            raise ValueError(f"--task-id {task_id} out of range [0,{n})")
        return task_id, task_suite.get_task(task_id)
    if task_name is None:
        raise ValueError("need --task-name or --task-id")
    names = [task_suite.get_task(i).name for i in range(n)]
    if task_name in names:
        i = names.index(task_name)
        return i, task_suite.get_task(i)
    hits = [i for i, x in enumerate(names) if task_name in x]
    if len(hits) == 1:
        return hits[0], task_suite.get_task(hits[0])
    raise ValueError(
        f"task {task_name!r} matched {len(hits)} tasks: "
        f"{[names[i] for i in hits]}"
    )


def load_init_states(explicit: str | None, task, task_suite, task_id: int):
    """Resolution order: --init-path > canonical init_files dir > repo fallback."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--init-path {p} does not exist")
        import torch
        return torch.load(p, weights_only=False), str(p)

    canonical = DEFAULT_INIT_DIR / f"{task.name}.pruned_init"
    if canonical.exists():
        import torch
        return torch.load(canonical, weights_only=False), str(canonical)

    fallback = INIT_FALLBACK_DIR / f"{task.name}.pruned_init"
    if fallback.exists():
        import torch
        return torch.load(fallback, weights_only=False), str(fallback)

    # last resort: whatever the benchmark itself resolves (may raise)
    return task_suite.get_task_init_states(task_id), "<benchmark default>"


def problem_env(env):
    """Return the object exposing the LIBERO-Mem subgoal API.

    `OffScreenRenderEnv` (ControlEnv) wraps the real BDDL problem in `.env`, and
    the wrapper has no `__getattr__` forwarding. The subgoal machinery
    (`_check_success(inc=...)`, `reset_subgoal_progress`) lives on the wrapper's
    inner problem, so we have to dig for it.
    """
    for cand in (env, getattr(env, "env", None),
                 getattr(getattr(env, "env", None), "env", None)):
        if cand is not None and hasattr(cand, "_check_success"):
            return cand
    return None


def subgoal_state(problem):
    """Best-effort subgoal readout; the kitchen problem tracks
    `_satisfied_subgoals` and only counts a subgoal once it has been stable for
    a while (see `_sub_goal_live_time` / `_sub_goal_nonlive_time`)."""
    if problem is None:
        return None
    for fn in ("get_satisfied_subgoals",):
        f = getattr(problem, fn, None)
        if callable(f):
            try:
                try:
                    s = f()
                except TypeError:
                    s = f(None)
                return list(s) if s is not None else []
            except Exception:
                pass
    s = getattr(problem, "_satisfied_subgoals", None)
    if s is not None:
        return list(s)
    return None


def goal_subgoal_count(problem):
    """How many subgoals the parsed goal has, mirroring `_check_success`'s own
    leaf/sequence discrimination. Returns None for a leaf-level goal (in which
    case `_check_success`'s return value IS the success signal)."""
    try:
        gs = problem.parsed_problem["goal_state"]
        head = gs[0][0]
    except Exception:
        return None
    if head not in ("sequence", "or"):
        return None                     # leaf level
    # NB: `_check_success` does `goal_state = goal_state[0]` for non-leaf
    # goals, so the sequence list is gs[0], NOT gs. Using gs here yields
    # len-1 == 0 whose `>= 0` test is trivially true and marks EVERY episode
    # a success -- that mistake faked 4 successes before it was caught.
    seq = gs[0]
    n = len(seq) - 1
    if n <= 0:
        return None                     # malformed / no usable subgoals
    return n


def subgoal_progress(problem):
    """Debug counters behind the sequenced success check."""
    if problem is None:
        return None
    return {
        "n_satisfied": len(getattr(problem, "_satisfied_subgoals", []) or []),
        "live_time": getattr(problem, "_sub_goal_live_time", None),
        "nonlive_time": getattr(problem, "_sub_goal_nonlive_time", None),
        "overshot": getattr(problem, "_overshot", None),
    }


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-suite", default="libero_mem")
    ap.add_argument("--task-name", default=DEFAULT_TASK)
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--init-path", default=None,
                    help="explicit .pruned_init; overrides the canonical path")
    ap.add_argument("--num-trials", type=int, default=50,
                    help="episodes to run (capped by available init states)")
    ap.add_argument("--max-steps", type=int, default=600,
                    help="per-episode step budget AFTER num-steps-wait")
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6800)
    ap.add_argument("--server-timeout", type=float, default=900.0,
                    help="seconds to wait for the policy server (default 15 min)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", default="",
                    help="checkpoint path, recorded in every record")
    ap.add_argument("--unnorm-key", default="libero_mem_no_noops")
    ap.add_argument("--config-path", default="",
                    help="config.json used by the policy server, recorded")
    ap.add_argument("--save-video", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve task/init-states/paths and stop; no sim, no GPU")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"t1_{args.task_name[:40]}"

    print("=" * 70)
    print("T1 LIBERO-Mem rollout evaluator")
    print("=" * 70)
    print(f"suite        : {args.task_suite}")
    print(f"task-name    : {args.task_name}")
    print(f"out-dir      : {out_dir}")
    print(f"ckpt         : {args.ckpt or '(unset)'}")
    print(f"unnorm-key   : {args.unnorm_key}")
    print(f"server       : {args.host}:{args.port} (timeout {args.server_timeout:.0f}s)")
    print(f"num-trials   : {args.num_trials}")
    print(f"max-steps    : {args.max_steps}  (+{args.num_steps_wait} wait)")
    print()

    # ---------------------------------------------------- 1) task + init
    from libero.libero import benchmark
    suites = benchmark.get_benchmark_dict()
    if args.task_suite not in suites:
        print(f"!! suite {args.task_suite!r} not in {sorted(suites)}")
        return 2
    task_suite = suites[args.task_suite]()
    task_id, task = resolve_task(task_suite, args.task_name, args.task_id)
    print(f"resolved task_id : {task_id}")
    print(f"task.name        : {task.name}")
    print(f"task.language    : {task.language}")
    print(f"problem_folder   : {task.problem_folder}")
    print(f"bddl_file        : {task.bddl_file}")

    init_states, init_src = load_init_states(
        args.init_path, task, task_suite, task_id)
    import numpy as np
    init_arr = np.asarray(init_states)
    print(f"init states      : {init_arr.shape} dtype={init_arr.dtype}")
    print(f"init states from : {init_src}")
    print(f"n init states    : {len(init_arr)}")

    n_trials = min(args.num_trials, len(init_arr))
    if n_trials < args.num_trials:
        print(f"!! --num-trials {args.num_trials} > available "
              f"{len(init_arr)}; capping to {n_trials}")
    print(f"episodes to run  : {n_trials}")

    if args.dry_run:
        print("\n--- DRY RUN: paths/task/init resolved OK, no simulator "
              "started, no GPU touched ---")
        meta = {
            "dry_run": True,
            "task_suite": args.task_suite,
            "task_id": task_id,
            "task_name": task.name,
            "task_language": task.language,
            "init_source": init_src,
            "init_shape": list(init_arr.shape),
            "n_init_states": int(len(init_arr)),
            "n_trials": n_trials,
            "max_steps": args.max_steps,
            "unnorm_key": args.unnorm_key,
            "ckpt": args.ckpt,
            "config_path": args.config_path,
        }
        (out_dir / f"{tag}_dryrun.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote {out_dir / (tag + '_dryrun.json')}")
        return 0

    # ---------------------------------------------------- 2) sim + policy
    from libero_utils import get_libero_env, get_libero_image, quat2axisangle
    from robot_utils import set_seed_everywhere
    from vla_policy import LLaVAClient

    set_seed_everywhere(args.seed)

    run_id = (f"{tag}-{n_trials}trials-seed{args.seed}-"
              f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    jsonl_path = out_dir / f"{run_id}.episodes.jsonl"
    log_path = out_dir / f"{run_id}.log"
    log_f = open(log_path, "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"run_id={run_id} started={utcnow()}")
    log(f"task_id={task_id} task_name={task.name}")

    env, task_description = get_libero_env(task, resolution=args.resolution)
    log(f"task_description={task_description!r}")

    problem = problem_env(env)
    n_subgoals = goal_subgoal_count(problem) if problem is not None else None
    log(f"subgoal API: "
        + (f"found on {type(problem).__name__}" if problem is not None
           else "NOT FOUND -- success will be the raw env 'done' only")
        + (f" | goal is a sequence of {n_subgoals} subgoal(s)"
           if n_subgoals is not None else " | goal is leaf-level"))

    log(f"waiting for policy server at {args.host}:{args.port} ...")
    try:
        waited = wait_for_server(args.host, args.port, args.server_timeout)
    except Exception as e:
        log(f"!! policy server never became ready: {type(e).__name__}: {e}")
        log_f.close()
        return 3
    log(f"policy server ready after {waited:.0f}s")
    policy = LLaVAClient(base_url=f"http://{args.host}:{args.port}")

    # ---------------------------------------------------- 3) rollouts
    records = []
    n_success = 0
    t_start = time.time()
    for ep in range(n_trials):
        env.reset()
        policy.reset()
        episode_first_frame = "True"
        obs = env.set_init_state(init_arr[ep])

        # LIBERO-Mem's sequenced goals need their progress state cleared per
        # episode, otherwise the subgoal counters carry over from the last one.
        if problem is not None:
            try:
                problem.reset_subgoal_progress()
            except Exception as e:                 # noqa: BLE001
                log(f"   !! reset_subgoal_progress failed: "
                    f"{type(e).__name__}: {e}")

        t = 0
        success = False
        steps_taken = 0
        subgoals = None
        err = ""
        replay = []
        # --- per-episode performance accounting ---------------------------
        n_requests = 0
        n_actions_returned = 0
        t_policy = 0.0     # policy.process_frame (7B + diffusion + HTTP + JPEG)
        t_env = 0.0        # the env.step() calls that consume one chunk
        chunk_sizes: list[int] = []
        while t < args.max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, _r, _d, _i = env.step([0, 0, 0, 0, 0, 0, -1])
                    t += 1
                    continue

                img = get_libero_image(obs, args.resolution)
                if args.save_video:
                    replay.append(img)
                observation = {
                    "base_cam": img,
                    "states": np.concatenate((
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )),
                }
                _t0 = time.perf_counter()
                action = policy.process_frame(
                    text=task_description,
                    episode_first_frame=episode_first_frame,
                    **observation)
                t_policy += time.perf_counter() - _t0
                n_requests += 1
                if ";" in action:
                    action = action.replace(";", " ")
                action = np.array([float(x) for x in action.split(" ")], dtype=float)
                episode_first_frame = "False"

                action_dim = 7
                for i in range(len(action)):
                    if i % action_dim == action_dim - 1:
                        if action[i] == 1.0:
                            action[i] = -1.0
                        elif action[i] == 0.0:
                            action[i] = 1.0

                done_flag = False
                chunk = len(action) // action_dim
                chunk_sizes.append(chunk)
                n_actions_returned += chunk
                _t1 = time.perf_counter()
                for i in range(chunk):
                    obs, _r, done, _i = env.step(
                        action[i * action_dim:(i + 1) * action_dim])
                    steps_taken += 1
                    # LIBERO-Mem's sequenced goals: the env's own `done` calls
                    # _check_success() with inc=False, so the subgoal timers
                    # (_sub_goal_live_time / _sub_goal_nonlive_time) never
                    # advance and `done` can NEVER become True no matter how
                    # well the policy does. Drive the check with inc=True, the
                    # way the official mem_2/3/5/6 scripts do.
                    if problem is not None:
                        try:
                            _ok = problem._check_success(inc=True)
                        except TypeError:
                            _ok = False
                        # A2: `_check_success` returns True the moment ONE
                        # intermediate subgoal is confirmed (the sequence
                        # branch appends to _satisfied_subgoals and returns
                        # True). Breaking on that truncates a trajectory that
                        # is right on track, so only the FINAL subgoal counts.
                        if n_subgoals is None:
                            if _ok:
                                done = True
                        elif len(getattr(problem, "_satisfied_subgoals", [])
                                 or []) >= n_subgoals:
                            done = True
                    if done:
                        success = True
                        done_flag = True
                        break
                    t += 1
                t_env += time.perf_counter() - _t1
                if done_flag:
                    break
            except Exception as e:                     # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                break

        subgoals = subgoal_state(problem)
        n_sg = len(subgoals) if subgoals is not None else None
        sg_prog = subgoal_progress(problem)
        if success:
            n_success += 1

        video_path = ""
        if args.save_video and replay:
            try:
                from libero_utils import save_rollout_video
                video_path = save_rollout_video(
                    replay, ep, success, task_description, log_file=None,
                    rollout_dir=str(out_dir / "videos"),
                )
                log(f"   video -> {video_path}")
            except Exception as e:                 # noqa: BLE001
                log(f"   !! video save failed: {type(e).__name__}: {e}")

        rec = {
            "run_id": run_id,
            "task_id": task_id,
            "task_name": task.name,
            "task_language": task.language,
            "episode": ep,
            "init_id": ep,
            "init_source": init_src,
            "success": bool(success),
            "n_subgoals_satisfied": n_sg,
            "subgoals": subgoals,
            "subgoal_progress": sg_prog,
            "episode_len": int(steps_taken),
            "n_requests": n_requests,
            "n_actions_returned": n_actions_returned,
            "chunk_size_min": (min(chunk_sizes) if chunk_sizes else None),
            "chunk_size_max": (max(chunk_sizes) if chunk_sizes else None),
            "steps_per_request": (round(steps_taken / n_requests, 3)
                                  if n_requests else None),
            "t_policy_s": round(t_policy, 3),
            "t_env_s": round(t_env, 3),
            "per_request_policy_ms": (round(1000 * t_policy / n_requests, 1)
                                      if n_requests else None),
            "per_request_env_ms": (round(1000 * t_env / n_requests, 1)
                                   if n_requests else None),
            "max_steps": args.max_steps,
            "error": err,
            "video": video_path,
            "seed": args.seed,
            "unnorm_key": args.unnorm_key,
            "ckpt": args.ckpt,
            "config_path": args.config_path,
            "ts": utcnow(),
        }
        records.append(rec)
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(rec) + "\n")

        perf = ""
        if n_requests:
            perf = (f" | req={n_requests} chunk={min(chunk_sizes)}-{max(chunk_sizes)}"
                    f" steps/req={steps_taken / n_requests:.2f}"
                    f" policy={t_policy:.1f}s ({1000 * t_policy / n_requests:.0f}ms/req)"
                    f" env={t_env:.1f}s ({1000 * t_env / n_requests:.0f}ms/req)")
        log(f"[ep {ep + 1}/{n_trials}] success={success} "
            f"len={steps_taken} subgoals={n_sg} "
            f"SR={n_success}/{ep + 1} ({100 * n_success / (ep + 1):.1f}%)"
            + perf + (f" ERR={err}" if err else ""))

    # ---------------------------------------------------- 4) summary
    elapsed = time.time() - t_start
    sg_vals = [r["n_subgoals_satisfied"] for r in records
               if r["n_subgoals_satisfied"] is not None]
    summary = {
        "run_id": run_id,
        "task_suite": args.task_suite,
        "task_id": task_id,
        "task_name": task.name,
        "task_language": task.language,
        "ckpt": args.ckpt,
        "config_path": args.config_path,
        "unnorm_key": args.unnorm_key,
        "seed": args.seed,
        "init_source": init_src,
        "n_init_states_total": int(len(init_arr)),
        "n_episodes": len(records),
        "n_success": n_success,
        "SR": (n_success / len(records)) if records else None,
        "SGC_mean": (float(np.mean(sg_vals)) if sg_vals else None),
        "SGC_sum": (int(np.sum(sg_vals)) if sg_vals else None),
        "SGC_available": bool(sg_vals),
        "mean_episode_len": (float(np.mean([r["episode_len"] for r in records]))
                             if records else None),
        "n_errors": sum(1 for r in records if r["error"]),
        "perf": ({
            "total_policy_s": round(sum(r["t_policy_s"] for r in records), 2),
            "total_env_s": round(sum(r["t_env_s"] for r in records), 2),
            "total_requests": sum(r["n_requests"] for r in records),
            "chunk_size_min": min(
                (r["chunk_size_min"] for r in records
                 if r["chunk_size_min"] is not None), default=None),
            "chunk_size_max": max(
                (r["chunk_size_max"] for r in records
                 if r["chunk_size_max"] is not None), default=None),
            "ms_per_request_policy": round(
                1000 * sum(r["t_policy_s"] for r in records)
                / max(1, sum(r["n_requests"] for r in records)), 1),
            "ms_per_request_env": round(
                1000 * sum(r["t_env_s"] for r in records)
                / max(1, sum(r["n_requests"] for r in records)), 1),
        } if records else None),
        "max_steps": args.max_steps,
        "num_steps_wait": args.num_steps_wait,
        "server_wait_s": waited,
        "wall_clock_s": elapsed,
        "started": t_start,
        "ts": utcnow(),
    }
    summary_path = out_dir / f"{run_id}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    log("")
    log("=" * 70)
    log(f"RESULT  task={task.name}")
    log(f"  episodes   : {len(records)}")
    log(f"  SR         : {n_success}/{len(records)} "
        f"= {(100 * n_success / len(records)) if records else 0:.1f}%")
    if sg_vals:
        log(f"  SGC (mean) : {np.mean(sg_vals):.2f}  (sum {int(np.sum(sg_vals))})")
    else:
        log("  SGC        : unavailable (env exposed no subgoal state)")
    log(f"  wall clock : {elapsed:.0f}s")
    log(f"  episodes   : {jsonl_path}")
    log(f"  summary    : {summary_path}")
    log("=" * 70)

    log_f.close()
    try:
        env.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
