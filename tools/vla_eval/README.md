# RPBE-VLA evaluation & run tooling

Tooling written while bringing **MemoryVLA × LIBERO-Mem** up on a single A100,
for the RPBE arm (`--arm gamma-rpbe`). Kept in-tree so it is not lost with an
instance.

Full step-by-step handoff (hardware, weights, dataset, env, train, eval, and
the eight gotchas we actually hit) is in `HANDOFF_rpbe_vla_t3.md` at the repo
root of the handoff bundle.

## Scripts

| File | Purpose |
|---|---|
| `tiered_eval_t1.py` | LIBERO rollout evaluator. Name says t1 but it is generic — `--task <KITCHEN_SCENE1_N_...>` works for any LIBERO-Mem task; the subgoal count is derived from the `.bddl` goal. |
| `run_tiered.sh` | Launcher. **Raises `ulimit -n` to 65536** — without it the 20th demo dies on fd exhaustion. |
| `finalize_run_dir.py` | Writes `config.json` + `dataset_statistics.json` into a run dir. Needed whenever training was stopped early, because the trainer only writes them after the whole loop finishes. |
| `compare_images.py` | **Run this first on any new task.** Renders the sim at a demo's `initial_state` and compares against the same frame stored in the hdf5 under identity / hflip / vflip / rot180, so the correct preprocessing is measured rather than assumed. |
| `watch_steps.py` | Copies `best.pt`/`latest.pt` aside as `snapshot_<step>.pt` for a requested step list. |
| `patch_reset_cap.py` | Caps LIBERO's unbounded `ControlEnv.reset` retry loop (a pathological randomisation otherwise hangs the eval forever with no output). |
| `patch_libero_torchload.py` | `weights_only=False` for the `.pruned_init` numpy pickles under torch ≥ 2.6. |
| `audit_stage1.py`, `audit_stage2.py` | Verify that a training checkpoint's flat trainable-only state dict maps back onto a freshly built model (stage 2 builds the real architecture and checks every tensor). |
| `probe_demo1.py` | Sim-only probe of the task predicate / subgoal counters at an `initial_state`. |
| `count_chunks.py` | Counts how many actions the policy server actually returned per request, from its own log. |
| `eval_libero_t1.py`, `deploy_t1.py`, `run_t1_rollout.sh` | The earlier two-process (Flask) design. Superseded by the in-process `tiered_eval_t1.py`, kept for reference. |
| `watch_best.py`, `stop_at_step.py` | Checkpoint watcher / stop-training-at-step helper. |

## Paths

All hardcoded paths are overridable by environment variable — no edits needed:

```
RPBE_VLA_ROOT      root for the default paths
LIBERO_MEM_REPO    the libero-mem clone (sim + bddl + assets)
MEMVLA_DIR         <PRSS2>/third_party/memoryvla
PRSS2_SRC          <PRSS2>/src
RPBE_PYTHON        the python interpreter
RPBE_DATA_ROOT     where the LIBERO-Mem hdf5 + metainfo.json live
RPBE_TASK          task name for compare_images.py
```
