# Reproducing the Stage8 run (the one that produced the 25.0% peak)

Short answer: **the code clones and runs; the run cannot be resumed, and a
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
