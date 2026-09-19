# Reproducing the Stage8 run (the one that produced the 25.0% peak)

Short answer: **the code clones and runs; the run cannot be resumed, and a bit-exact
exact rerun needs work you cannot skip.** This file says exactly which parts
you get from the clone, which you must supply yourself, and which are gone.

## 1. What you get from the clone

| path | what it is |
|---|---|
| `third_party/memoryvla/train_libero_mem_rpbe.py` | the trainer, with `--train-scope`, `--rpbe-*`, `--proj-*` |
| `third_party/memoryvla/vla/memory_vla.py` | the host + GammaMerger |
| `third_party/memoryvla/vla/datasets/hdf5_dataset.py` | dense sliding window, gripper relabel, augmentation |
| `third_party/memoryvla/action_model/` | the diffusion action model |
| `src/rpbe_embodied/` | the RPBE plugin: feasibility projection, boundary protocol, resume contract, **60 unit tests** |
| `scripts/tiered_eval.py` | the rollout / tiered-success evaluator |
| `reproduce/*.sh` | the two training stages and the eval, flags recovered verbatim |
| `results/` | every number in the write-up |

The code state that actually ran is commit **`c05e4fb`** (`[rpbe] freeze the
protocol`). Nothing outside `results/` and `_recovered/` changed after it.

## 2. What is missing, and how bad each gap is

| missing | recoverable? | consequence |
|---|---|---|
| **host** `avg_s42_stage5_official/best.pt` (1.85 GB) | **No** | it is the init of *both* stages; without it this run cannot be re-run or continued |
| **stage-1 checkpoint** `gamma-only_ours_s8g/checkpoint.pt` | **No** | the continuation that hit 25.0% cannot be resumed |
| the Stage1→Stage4 chain that produced the host | partly | only the Stage5 command and some intermediate flags were recorded |
| `libero-mem-no-noops` build script | No | the dataset is a derived build; the filter is not in the repo |
| per-demo rollout lines | **No** | the eval was piped through a summary `grep`; see §5 |

Everything listed lived only in `outputs/` on one AutoDL box, which is gone.
There is no second copy.

**Therefore the honest statement is: the 25.0% run is *documented*, not
*resumable*.** Anyone wanting to verify it must retrain the host first.

## 3. What you must supply

| artifact | where it comes from |
|---|---|
| base VLA checkpoint `step-295000-epoch-40-loss=0.2200.pt` | the MemoryVLA / OpenVLA-7B prismatic release. **We did not record the exact URL** —— get it from the MemoryVLA release page |
| `Llama-2-7b-hf` | Meta's Llama-2 release (the trainer reads `LLAMA2_LOCAL_PATH`) |
| LIBERO-Mem task suite + `libero-mem-code` (the env) | the LIBERO-Mem release. Not vendored here |
| `libero-mem-no-noops` | **derived**: LIBERO-Mem with no-op transitions removed (~9% of frames). The filter script was never committed. `scripts/probe_libero_mem_hdf5.py` is the starting point for re-deriving it; the criterion was "drop transitions where the robot neither moves nor toggles the gripper" |
| a **host** | either retrain Stage5-avg (§4) or supply your own |

## 4. Running it

```bash
# one-time
apt/dnf ...               # nothing exotic; see requirements.txt
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export RPBE_EMBODIED_PATH=/root/autodl-tmp     # dir containing rpbe_embodied/

# units (CPU, no host needed) -- proves the projection/contract layer
python -m pytest src/rpbe_embodied/test_projection.py src/rpbe_embodied/test_s5r_window.py

# stage 1: RPBE, 0 -> 15000
HOST=/path/to/your/host.pt  bash reproduce/run_ours_s8g.sh

# stage 2: weight-initialised continuation (this is what reached 25.0%)
S8G=outputs/gamma-only_ours_s8g/checkpoint.pt  bash reproduce/run_ours_s8h.sh

# rollout at any checkpoint
CKPT=... STATS=.../dataset_statistics.json  bash reproduce/eval_rollout.sh
```

**The host** was Stage5 of a multi-stage chain (avg arm, diffusion head,
`--max-steps 20000 --batch-size 4 --grad-accum 4 --lr 2e-5 --sched const
--warmup-steps 0 --mem-length 16 --repeated-diffusion-steps 4
--future-action-window-size 15 --eval-every 1000 --checkpoint-every 1000
--image-aug 1 --dim-weight 0 --seed 42`, init from the Stage4
`avg_s42_eqw_restart/best.pt`). The earlier stages' configs are only
partially recorded, so retraining the host is not a copy-paste job.

## 5. The eval protocol —— and the one thing not to repeat

```
weighted_success = sum(tier) / (3 * n)      tier = completed 3-cycle repetitions
strict_success   = tier3 AND not overshot AND check_success(inc=False)
```

`reproduce/eval_rollout.sh` writes the **full** stdout. Please keep it that
way. The Stage8 evals were piped through

```bash
grep -E "tier_dist|weighted_success|TIERED_DONE"
```

which discarded every `[demo_N] tier=X/3 ...` line before it reached a file.
As a result **no Stage8 rollout can be traced to individual demos** ——
including the 25.0% peak —— and the "are the demos that complete three cycles
at 17000 the same ones that do at 19000?" check cannot be run. See
`results/PER_DEMO_RECOVERY.md`.

## 6. Environment that ran it

AutoDL container, 1× A800 80 GB, Python at `/root/env_eval/bin/python`
(torch 2.8.0+cu128, `torch.func` available), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`HF_HUB_OFFLINE=1` with a local Llama-2 path. Training used ~20-22 GB of the
80 GB; a concurrent rollout eval added ~17 GB.

## 7. Code identity —— what to check before a multi-seed sweep

If you are running several seeds and the only requirement is that they all run
the **same code**, this is the check:

```bash
bash reproduce/check_code_identity.sh          # default: HEAD
bash reproduce/check_code_identity.sh <ref>    # any commit/tag/branch
```

It hashes the git tree entries of the three code roots ——
`third_party/memoryvla`, `src/rpbe_embodied`, `scripts/tiered_eval.py`
(138 files) —— and compares against the frozen value:

```
634c18f53b59099f5b0c09af2b29c639cc34f94ea5c6031f352beebb98b03926
```

frozen at **`c05e4fb8b6fd89f0bccd99ed4c3b9ef3448aed36`**. Because it hashes
tree entries rather than files on disk, it is stable across machines, checkout
order, line endings and mtimes —— only a real content change moves it, and on
a mismatch it prints the per-file diff.

Every commit after the freeze touched only `results/` and `_recovered/`, so
the branch tip carries identical code; pin `c05e4fb` if you want it explicit
in the record.

### What a fingerprint does not pin

The code can be byte-identical while the run still differs. For a multi-seed
sweep these are the things to hold fixed deliberately:

| knob | value used here | why it matters |
|---|---|---|
| host checkpoint | `avg_s42_stage5_official/best.pt` | a different host is a different experiment, not another seed |
| `--seed` | 42 | only this was ever run |
| budget / structure | 15000 steps, then a **weight warm start** for the next 10000 | see §8 —— the second stage is a fresh optimiser, not a resume |
| data | `libero-mem-no-noops` | a different dataset build changes every number |
| eval protocol | 20 demos, `exec=8`, `maxsteps=370`, seed 42 | `exec` in particular moves the success rate a lot |

## 8. The two stages are NOT one trajectory

Stage 2 (`run_ours_s8h.sh`) was launched with `--init-from-weights`, i.e. it
**loads stage-1 weights and starts a fresh optimiser, scheduler and RNG**.
Stage 1 ran with `--no-fullstate 1`, so there was no optimiser state to
continue from in the first place.

Consequences, which any multi-seed design has to decide about:

- The 15000 → 21000 "curve" is two separate runs sampled at successive weight
  points, not one training trajectory.
- The overfitting decline (25.0 → 23.3 → 18.3) is therefore measured across a
  run that had its optimiser reset at 15000. A single uninterrupted 25000-step
  run might not show the same shape.
- If the sweep is meant to answer "does RPBE hold up", each seed needs the
  **same structure** as the recorded run —— two stages, one warm start —— or
  the comparison is between different training procedures.

## 9. The multi-seed sweep we settled on (single segment, 18000 steps)

The two-stage mess in §8 is avoided entirely by running **one uninterrupted
segment to 18000 steps** and evaluating the rollout at **12000 / 15000 /
18000**.

```bash
# 1. check the code is the frozen one
bash reproduce/check_code_identity.sh

# 2. train: one process per GPU, two GPUs by default
HOST=/path/to/host.pt SEEDS="42 7 123" ARMS="gamma-rpbe" GPUS="0 1" \
  bash reproduce/sweep_launch.sh

# 3. AFTER all training finishes, evaluate each run
RUN=gamma-rpbe_seed42_18k  bash reproduce/eval_sweep_18000.sh
RUN=gamma-rpbe_seed7_18k   bash reproduce/eval_sweep_18000.sh
...
```

What each script pins:

| script | pins |
|---|---|
| `train_sweep_18000.sh` | `--max-steps 18000 --snapshot-steps 12000,15000 --no-fullstate 1`, `--train-scope gamma-only`, `kappa=0.02`, `tau=3e-4`, `--seed $SEED`, host via `$HOST` |
| `eval_sweep_18000.sh` | the three checkpoints, 20 demos, `exec=8`, `maxsteps=370`, seed 42, **full unfiltered stdout** |
| `sweep_launch.sh` | arm × seed product, at most `len($GPUS)` concurrent |

### Two traps this design sidesteps, and one it must respect

1. **`dataset_statistics.json` only exists after training finishes.** The
   trainer writes it at completion, and `tiered_eval.py` reads it from the
   checkpoint's own directory —— so evaluating a mid-run snapshot fails with
   `FileNotFoundError`. This bit us in Stage8 and the workaround was to
   hand-copy the file from another run. Hence: **train first, evaluate after**,
   or pre-place the file.
2. **Do not pipe the eval through a summary `grep`.** Keep
   `[demo_N] tier=X/3 ...`. Losing it cost us every per-demo identity in
   Stage8.
3. **Budget.** ~1.56 s/step → 18000 steps ≈ **7.8 h per run**; **~9.3 GB
   per run** (2 snapshots + best + latest + checkpoint, no fullstate). Three
   seeds × 3 checkpoints of eval ≈ 18 min of GPU per run on top.

### What to record per seed, so this doesn't repeat

Per run, keep at least: `train.log`, the val curve, `rollout_full.log` (with
per-demo lines), the `tier_dist` per checkpoint, and
`check_code_identity.sh`'s output. The last one is what proves the seeds are
comparable at all.



## 10. The four things asked for, and where each one is answered

### (1) Is this the same code that produced the 3-cycle successes?

Yes —— and it is verified at two levels, not argued:

| level | check | value |
|---|---|---|
| whole code tree | `git ls-tree` over `third_party/memoryvla` + `src/rpbe_embodied` + `scripts/tiered_eval.py`, 138 files | `634c18f53b59099f5b0c09af2b29c639cc34f94ea5c6031f352beebb98b03926` |
| **the 5 files that were on the box** | md5, recorded during the run and compared server-vs-local at `c05e4fb` | see below |

```
9a8cdde57a7f0d7d1353bada37683e8c  src/rpbe_embodied/loss.py
5f50ad9cf0d5a030641360d3ee76c011  src/rpbe_embodied/boundary.py
f826bcf1566d66b9f2b4c13cdcb9c8c5  src/rpbe_embodied/resume.py
45760bbd5f167da28c0b608782cf3134  src/rpbe_embodied/test_projection.py
c05e25f20f4dddded7aac1acec0539b7  third_party/memoryvla/train_libero_mem_rpbe.py
```

`bash reproduce/check_code_identity.sh` checks both and fails loudly otherwise.

**Footnote, because it looks like a discrepancy and is not one.** The md5s
recorded *in the run log* are the CRLF form (`38751ce7…`, `17fed5d0…`) because
the box received the Windows working-tree copy over sftp; the values above are
the same content with `\r` stripped. Feeding the `c05e4fb` blob through
`sed 's/$/\r/'` reproduces the recorded numbers exactly, and
`git diff c05e4fb HEAD` over these files is empty. The check script strips CR
so it gives the same answer on either platform.

### (2) Training loss and val, every 1000 steps —— the file

`train_sweep_18000.sh` now runs `--eval-every 1000`, and after evaluation
`extract_curves.py` writes **`eval_<run>/curves.csv`**:

```
step,train_loss,val_loss,n_train_points
1000,0.0xxxxx,0.0xxxxx,20
...
```

`train_loss` is the mean of the training losses logged inside that 1000-step
block (20 samples at `--log-every 50`); `val_loss` is the `[eval @ opt N] val
action loss` at that step, blank when a step has no eval. Standalone use:
`python reproduce/extract_curves.py <train.log> curves.csv`.

### (3) Per-demo cycle count at each of the three checkpoints

`eval_sweep_18000.sh` runs `tier_table.py` over the **unfiltered** eval log and
writes, per run:

- **`tier_by_demo.csv`** —— `demo, snapshot_12000, snapshot_15000, checkpoint`,
  each cell = how many times that demo completed the 3-cycle task (0-3).
- **`tier_summary.csv`** —— per checkpoint: `n_0/n_1/n_2/n_3`, `>=1`, `>=2`,
  `strict`, `sum_tier`, `weighted_success_pct`.

`tier_table.py` exits with an error if it finds no per-demo lines, i.e. if the
log was filtered —— the failure mode that destroyed the Stage8 evidence.

### (4) Where the data is —— and how to get it

**None of it is in git** (fetch it with `bash reproduce/fetch_data.sh`):

| artifact | source | size |
|---|---|---|
| **T3 task data** = `KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times_demo.hdf5` | HF dataset **`libero-mem/LIBERO-Mem`**, file at `/datasets/libero-mem/LIBERO-Mem/resolve/main/…` | 18.4 GB |
| `metainfo.json` | same repo | 0.7 GB |
| base model `step-295000-epoch-40-loss=0.2200.pt` | HF **`openvla/openvla-7b-prismatic`** | 30 GB |
| Llama-2-7b-hf | HF **`NousResearch/Llama-2-7b-hf`** | 13.5 GB |
| the simulator env | `github.com/libero-mem/libero-mem` | small |

So "T3" is the **3-times task** (`KITCHEN_SCENE1_3`), which is the task every
number in `results/` was measured on.

**One caveat you cannot download away.** The trainer's `--data-root` points at
`libero-mem-no-noops`, a build of the file above with no-op transitions
removed (~9% of frames). That filter script was never committed. Two options:

- re-derive it (start from `scripts/probe_libero_mem_hdf5.py`); **for a
  multi-seed comparison it only has to be fixed across your seeds —— it does
  not have to match ours**, because every seed eats the same dataset; or
- skip it and point `--data-root` at the raw directory —— then say so
  explicitly, because that is a different dataset build from ours.

Disk budget: 30 + 13.5 + 18.4 + 0.7 ≈ 63 GB of inputs, plus ~9.3 GB per run.
Budget **200 GB+**.
