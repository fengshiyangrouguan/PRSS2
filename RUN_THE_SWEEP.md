# TASK SPEC — run the RPBE multi-seed sweep on LIBERO-Mem T3

**Audience: an AI agent working on a fresh GPU box.** Follow this file top to
bottom. Every phase has a GATE: an expected result you must observe before
continuing. If a gate fails, **stop and report** —— do not improvise a fix and
do not continue; a run that silently deviates is worthless.

Branch to use: **`fix/avg-lora-clock`**. Everything below is on that branch.

---

## Phase 0 — Orientation (read these, in this order)

| file | why |
|---|---|
| `REPRODUCE.md` | the full reproduction picture, including what is **not** recoverable |
| `results/STAGE8_RESULTS.md` | what the previous run measured, and what it does and does not establish |
| `results/OFFICIAL_CODE_FIXES.md` | the implementation fixes already applied —— tells you which behaviours are deliberate |
| `results/PER_DEMO_RECOVERY.md` | the mistake this task must not repeat |

Three facts you need before you touch anything:

1. **The task is `KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times`** ——
   "pick the bowl up and put it back on the plate, **3 times**". This is what
   "T3" means everywhere. One rollout episode gets **up to 3 completed
   repetitions**; that count is its **tier** (0, 1, 2 or 3).
2. **The metric is `weighted_success = sum(tier) / (3 * n)`** over `n=20`
   demos. `strict_success` is a separate, stricter count (tier 3, no overshoot,
   final task check).
3. **Ours = RPBE = `--arm gamma-rpbe`**, a cheap adaptation that trains only the
   `GammaMerger` (0.57M params) on top of a frozen host, with a feasibility
   projection on the update. It **starts from a host checkpoint**.


### Terminology —— "base" and "host" are two different things

The trainer takes two independent checkpoints. Conflating them is the single
most common confusion, so read this twice:

| flag | what it is | has task ability? | required? |
|---|---|---|---|
| `--pretrained-checkpoint` | **the BASE** —— OpenVLA-7B prismatic (`step-295000-epoch-40-loss=0.2200.pt`). Loaded through `load_vla()`; it *builds the architecture* (vision encoder + LLM + DiT head) and initialises its weights. | **no** —— a general robot policy, never trained on LIBERO-Mem | **yes, always** |
| `--init-from-weights` | **the HOST** —— a MemoryVLA checkpoint that was already trained on LIBERO-Mem. Copies `ck["model"]` on top of the base at opt 0, with a fresh optimizer/scheduler/RNG. | **yes** —— ours scored 23.3% | no (default `""`) |

So:

- **The base is not the host.** The base is the architecture init; the host is
  a *product of training* that the base was the input to.
- **The thing Γ fine-tunes on top of is the HOST, not the base.**
- **Both of our Stage8 stages used the same base** and differed only in
  `--init-from-weights`: stage 1 used the host, stage 2 used stage 1's
  checkpoint.

**Why you cannot just use the base.** `--train-scope gamma-only` freezes the
action head and the memory modules and trains only Γ (0.57M params, a memory
*merge* operator). If the frozen parts never learned the task, there is no task
ability to preserve and Γ cannot create any —— every rollout is ~0. The base
alone therefore answers a different question, and a much harder one.

**The good historical result was itself produced by training from the base** ——
as a chain: `base → Stage1 → … → Stage5` → the 23.3% checkpoint *is* the host.
"Our run from scratch scored well" and "you need a host" are the same statement
seen from two ends.

**What counts as a usable host:** same architecture as `load_vla()` builds
(prismatic dinosiglip + llama2-7b-pure + DiT-L, matching `mem_length` /
`retrieval_layers` / `fusion_type` / `per_token_size`), trained on LIBERO-Mem
T3, with **rollout > 0**, and shipped with the `dataset_statistics.json` that
matches its `--data-root`/`--task-filter`. Record its hash and its baseline
`weighted_success` —— every number you report is relative to it.

---

## Phase 1 — GATE: confirm you are running the frozen code

```bash
cd <repo>
git fetch origin && git checkout fix/avg-lora-clock
bash reproduce/check_code_identity.sh
```

**Expected output — all of it:**

```
RESULT: MATCH -- this is the frozen Stage8 code.
        (frozen at c05e4fb8b6fd89f0bccd99ed4c3b9ef3448aed36)
deployed-file check (CR-stripped, so line endings do not matter):
  ok   src/rpbe_embodied/loss.py
  ... (5 files)
RESULT: all 5 deployed files verified against the frozen record.
```

If you see `MISMATCH`, **stop and report**. Do not train on a mismatched tree.
Do not "fix" the file to make the hash pass —— the mismatch means you have the
wrong code, and the fix is to check out `c05e4fb`, not to edit anything.

Also run the unit tests; they need no GPU and no data:

```bash
python -m pytest src/rpbe_embodied/test_projection.py src/rpbe_embodied/test_s5r_window.py
```

**Gate: expect `40 passed`** (projection) plus the s5r window suite. A failure
here means the projection/contract layer is broken —— stop and report.

---

## Phase 2 — Environment

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export RPBE_EMBODIED_PATH=<repo>/src        # so `import rpbe_embodied` resolves
```

Headless rendering needs system libs (`libegl1 libgl1 libosmesa6 libglew2.2
libglib2.0-0 libsm6 libxext6 libxrender1 libxi6`) and the simulator stack
(`mujoco==3.1.6 bddl==1.0.1 gym==0.25.2 cloudpickle==2.1.0 easydict thop`).

**Gate:** `python -c "import torch, torch.func; print(torch.__version__)"`
prints a 2.8.x version. `torch.func` is required by the projection's batched
VJP path; without it the code falls back to a slow path and says so in the log.

---

## Phase 3 — Data (nothing large is in the repo)

```bash
ROOT=<scratch-dir-with-200GB-free>  bash reproduce/fetch_data.sh
```

That fetches, from HuggingFace (mirror by default):

| artifact | source |
|---|---|
| `KITCHEN_SCENE1_3_…_3_times_demo.hdf5` (18.4 GB) | dataset `libero-mem/LIBERO-Mem` |
| `metainfo.json` (0.7 GB) | same |
| `openvla-7b-prismatic` incl. `checkpoints/step-295000-epoch-40-loss=0.2200.pt` (30 GB) | `openvla/openvla-7b-prismatic` |
| `Llama-2-7b-hf` (13.5 GB) | `NousResearch/Llama-2-7b-hf` |
| the simulator env | `github.com/libero-mem/libero-mem` |

### Decision you must make and record

The trainer's `--data-root` historically pointed at a **no-op-filtered** build
of that hdf5 (`libero-mem-no-noops`; no-op transitions, ~9% of frames,
removed). **That filter script is not in the repo.** Choose one:

- **(a) re-derive it.** Start from `scripts/probe_libero_mem_hdf5.py`. For a
  multi-seed sweep the filter only has to be **fixed across your seeds** —— it
  does **not** have to match ours, because every seed eats the same dataset.
- **(b) skip it** and point `--data-root` at the raw directory.

Either is acceptable **if you state which you did**. Do not silently use a
different dataset build than the one you report.

**Gate:** the hdf5 is present and non-empty (18 GB scale), `metainfo.json`
parses as JSON, and one demo can be opened with `h5py`. `metainfo.json` is not
optional —— the loader reads `task_description` from it and raises
`ValueError: task ... missing from metainfo.json` without it.

---

## Phase 3.5 — You need a HOST checkpoint

Ours is **not** trained from the base model; it is trained from a **host**
(`--init-from-weights $HOST`). The host we used is **lost** (see
`REPRODUCE.md` §2), so:

- if you were handed a host checkpoint, use it and record its hash;
- otherwise train one. The recorded host recipe was: `--arm avg`, diffusion
  head (`--reg-head 0`), `--max-steps 20000 --batch-size 4 --grad-accum 4
  --lr 2e-5 --sched const --warmup-steps 0 --mem-length 16
  --repeated-diffusion-steps 4 --future-action-window-size 15 --eval-every 1000
  --checkpoint-every 1000 --image-aug 1 --dim-weight 0 --seed 42`, initialised
  from a Stage4 average-merge checkpoint.

**Gate: the host must roll out.** Before sweeping, evaluate the host with
`reproduce/eval_rollout.sh` and confirm `weighted_success > 0`. A host that
scores 0 makes the whole sweep uninterpretable —— ours only means something if
there is task ability to preserve. Our host scored **23.3%** (14/60 tier
units, `tier_dist {0:10, 1:6, 2:4, 3:0}`).

Record the host's hash and its rollout number. Every result you report is
relative to it.


### Host candidates —— what exists publicly (surveyed)

**There is no released LIBERO-Mem checkpoint.** The LIBERO-Mem authors
(`github.com/libero-mem/libero-mem`) publish **data only** —— their README links
`datasets/libero-mem/LIBERO-Mem` and `LIBERO-Mem-Raw`, and the repo has **zero
releases**.

What does exist:

| source | trained on | usable? |
|---|---|---|
| `shihao1895/memvla-libero-{spatial,object,goal,100}` (+ `-plus-` variants, `memvla-plus-libero-mix`) | standard LIBERO suites | **best candidate** —— same codebase: ships `checkpoints/<name>.pt` in the MemoryVLA/OpenVLA layout **plus `dataset_statistics.json`**. But *not* trained on the 3-times task |
| `shihao1895/memvla-{bridge,mikasa,fractal}`, `memvla-plus-maniskill2` | other benchmarks | no —— wrong embodiment/suite |
| `Dexmal/libero-db-memvla` | standard LIBERO | **no** —— Dexbotic codebase, backbone `Qwen2.5-7B`; architecture-incompatible with this repo (Llama-2-7b-pure prismatic) |
| `aleksantari/memvla-libero-ckpts` | unknown | one bare `step_0030000.pt`, no README, suite unstated |
| `tarmus/memvla-libero-100k` | — | empty repo |

**Recommended path.** Try `shihao1895/memvla-libero-100` (LIBERO-100 is the
closest suite to our kitchen/long-horizon task) as the host:

1. **Check the config matches** what `load_vla()` builds here: `mem_length`,
   `retrieval_layers`, `fusion_type`, `per_token_size`, `action_model_type`
   (`DiT-L`), `action_dim=7`. A mismatch means the weights will not mean the
   same thing.
2. **Watch the load line.** `--init-from-weights` reports unexpected / missing
   keys. A small number is normal; a large number means you are silently
   loading a partially-matching model —— that invalidates the comparison.
3. **Measure its rollout on T3.** This is the gate that decides everything.
   It has never seen the 3-times task, so expect a low number —— possibly 0.
   **If it is 0, this host does not work** and you are back to training one.

Because this host is not the one the historical numbers came from, its
*measured* rollout becomes the baseline for every seed. Do not compare against
23.3% —— that belonged to a different, lost checkpoint.

---

## Phase 4 — Train the sweep

One uninterrupted run per seed, **18000 steps**, rollouts at **12000 / 15000 /
18000**.

```bash
HOST=$HOST SEEDS="42 7 123" ARMS="gamma-rpbe" GPUS="0 1" \
  bash reproduce/sweep_launch.sh
```

What this pins, and what you must not change without saying so:

| knob | value |
|---|---|
| `--max-steps` | 18000 |
| `--snapshot-steps` | 12000,15000 (18000 = `checkpoint.pt`, written at the end) |
| `--train-scope` | `gamma-only` (host **and** LoRA frozen; only Γ trains) |
| `--rpbe-mode project --kappa 0.02 --proj-tau 3e-4 --proj-iters 400 --proj-iters-max 1600` | the projection |
| `--eval-every` | 1000 (val action loss) |
| `--seed` | per run |
| `--no-fullstate 1` | halves disk; means the run is **not** resumable |

**Budget: ~1.56 s/step → ~7.8 h and ~9.3 GB per run.** Two runs per 2-GPU box.

**Gate, during the run —— check these in `train.log` and report them:**

- `abort` count stays **0**. A non-zero count means the projection failed to
  certify an update, and the boundary was skipped.
- `vmax_proj` ≲ `tau = 3e-4` (ours sat ~1e-8, four orders of margin).
- `proj_n_candidates > 0` in the large majority of constrained boundaries.
  Ours bound in **98.3%** of them; if yours rarely binds, the constraint is
  doing nothing and the run does not test what it claims to.

---

## Phase 5 — Evaluate

**Train first, evaluate after** —— not concurrently:

```bash
RUN=gamma-rpbe_seed42_18k  bash reproduce/eval_sweep_18000.sh
```

Why the ordering matters: `tiered_eval.py` reads `dataset_statistics.json`
from the checkpoint's own directory, and **the trainer only writes that file
when the run completes**. Evaluating a mid-run snapshot fails with
`FileNotFoundError`. (That is what happened to us; the workaround then was to
hand-copy the file from another run.) The script refuses to run early.

Protocol, fixed: **20 demos, `exec=8`, `maxsteps=370`, `seed=42`** —— the same
as the numbers in `results/`.

---

## Phase 6 — Deliverables

Per run, in `eval_<run>/`:

| file | content |
|---|---|
| `rollout_full.log` | the **unfiltered** eval output |
| `tier_by_demo.csv` | `demo, snapshot_12000, snapshot_15000, checkpoint` —— **how many times each demo completed the 3-cycle task**, per checkpoint |
| `tier_summary.csv` | per checkpoint: `n_0/n_1/n_2/n_3`, `>=1`, `>=2`, `strict`, `sum_tier`, `weighted_success_pct` |
| `curves.csv` | `step, train_loss, val_loss, n_train_points` —— one row per 1000 steps |

Report back, in this shape:

1. `check_code_identity.sh` output (both gates).
2. The host: path/hash, its rollout `weighted_success` and `tier_dist`.
3. Which data decision you took in Phase 3 (a or b), and the `--data-root`.
4. Per seed × checkpoint: `weighted_success`, `tier_dist`, `strict_success`.
5. The per-demo tables (`tier_by_demo.csv` contents).
6. Constraint audit: `abort`, `vmax_proj` min/median/max, fraction of
   constrained boundaries with candidates > 0.
7. Anything that deviated from this spec, stated explicitly.

---

## Hard rules —— violating any of these invalidates the run

1. **Never pipe the evaluation through a summary `grep`.** Keep
   `[demo_N] tier=X/3 …`. The Stage8 evals were piped through
   `grep -E "tier_dist|weighted_success|TIERED_DONE"`, which discarded every
   per-demo line *before* it reached a file —— as a result **no Stage8 rollout
   can be traced to individual demos, including the one that produced the best
   number**, and a stability check ("are the demos that complete 3 cycles at
   one checkpoint the same ones at the next?") became impossible.
   `tier_table.py` errors out if it sees no per-demo lines; if you hit that,
   you filtered the log.
2. **Never select a checkpoint using the rollout numbers.** The three
   checkpoints are a fixed, pre-declared grid. Picking the best is
   test-selection leakage, and with 20 demos the noise band is ±9.3
   percentage points —— roughly 2 tier units —— so the top of a table is
   mostly noise.
3. **Never change `--seed`, `--kappa`, `--proj-tau`, the eval protocol, or the
   step grid mid-sweep.** If you must, stop and say so.
4. **Keep `tier3` (3 completed cycles) as its own column.** It is not
   interchangeable with `weighted_success`: a demo that completes 2 cycles
   scores 2, and the interesting effect is in the tail.
5. **Do not resume or extend a run** without checking with the requester. Runs
   here use `--no-fullstate 1`, so a "continuation" is a weight warm start with
   a fresh optimiser —— a different training procedure, not a longer run.

## What is NOT in this repo (do not waste time looking)

- the host checkpoint, and any Stage8 checkpoint —— lost, no second copy;
- the `libero-mem-no-noops` filter script;
- the base models and the dataset —— download them (Phase 3);
- anything else large. If it is not in `git ls-files`, it is not here.

## What "success" means for this task

Not "beat the host". The deliverable is a **clean, comparable, honest
measurement**: same code (Phase 1 gate), same data, same grid, per-demo
evidence kept, constraint audit reported. Whether RPBE helps is the thing the
numbers are supposed to decide —— and the previous single-seed run **did not**
establish it, which is why this sweep exists.
