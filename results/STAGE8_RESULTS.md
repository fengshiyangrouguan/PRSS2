# Stage8 — proposal-space projection: results record

Frozen code: `c05e4fb` (branch `fix/avg-lora-clock`).
Step-by-step metrics recovered from the working session transcript after the
AutoDL box (`nma1:21744`) became unreachable. Raw `train.log` files and
checkpoints lived only on that box and are NOT recoverable; everything below is
reconstructed from log lines actually read during the runs and is therefore
partial (see "Provenance" at the end).

## Method (frozen)

RPBE is a CONSTRAINT on the Gamma/compressor update, not an objective.

```
1 save Gamma_t
2 forward / backward
3 task gradient g_task  + per-interface gradients g_i(theta_t)
4 ONE global clip  ->  g~_task
5 STATELESS AdamW preview on clones (real m/v/step untouched) -> Delta_adamw
6 proposal-space QP:
      Delta* = argmin_D 1/2 ||D - Delta_adamw||^2
               s.t.  g_i . D >= -kappa ||g_i|| ||Delta_adamw||   for all i
7 HARD GATE: full-interface certificate v_max(Delta*) <= tau, else abort
8 advance AdamW moments exactly once (same clipped gradient)
9 Gamma <- Gamma_t + Delta*   (FULL overwrite)
10 scheduler exactly once
11 fp32 writeback AUDIT (reported, never gated)
```

- `kappa = 0.02` (frozen geometric calibration; never tuned on results)
- `tau = 3e-4` (certificate tolerance only; it does not change the feasible set)
- `--train-scope gamma-only`: Stage5 host frozen + Stage5 LoRA frozen
  (requires_grad=False AND out of every optimizer param group); only the
  0.57M-parameter Gamma merger trains
- diffusion action head (`--reg-head 0`), seed 42, host init
  `avg_s42_stage5_official/best.pt`

## What is measured, and how often

Two different things, at two different cadences —— do not confuse them:

1. **`run_eval`, every `--eval-every` optimizer steps** (500 for the s8g/s8h
   runs, 250 for the earlier r1 runs). It scores the **3 fixed validation
   demos** and reports **`val action loss`** —— the training objective itself
   (diffusion MSE under the diffusion head; SmoothL1 + gripper BCE under the
   regression head). Cheap, fully automatic, and logged as
   `[eval @ opt N] val action loss X (3 demos)`.
2. **`tiered_eval.py`, only at checkpoints I chose by hand** (8000 / 10000 /
   13000 / 15000, plus the auto-daemon at 17k / 19k / 21k). This is the
   **rollout** metric: `weighted_success`, run in a separate process against
   the official LIBERO-Mem subgoal machine. It is NOT run every eval.

So "every epoch" is not what happened: the automatic per-500-step number is
**val action loss**, and the rollout numbers are sparse manual measurements.

## Validation (action loss) curves

Recovered from `results/val_curves.csv` (90 points across 9 runs).

| step | avg host | gamma-rpbe (stage5) | **ours** (s8g) | gamma FROZEN (s8f) |
|---|---|---|---|---|
| 500 | — | — | **0.0694** | 0.0694 |
| 1000 | 0.0959 | 0.0964 | **0.0695** | 0.0695 |
| 1500 | — | — | **0.0694** | 0.0693 |
| 2000 | 0.0871 | 0.0866 | **0.0690** | 0.0695 |
| 2500 | — | — | **0.0694** | 0.0694 |
| 3000 | 0.0849 | 0.0847 | **0.0690** | 0.0691 |
| 3500 | — | — | **0.0690** | 0.0691 |
| 4000 | 0.0780 | 0.0789 | **0.0688** | 0.0690 |
| 5000 | 0.0824 | 0.0806 | — | 0.0688 |
| 6000 | 0.0754 | 0.0760 | — | 0.0687 |
| 7000 | 0.0742 | 0.0746 | — | 0.0690 |
| 8000 | 0.0757 | 0.0749 | **0.0688** | 0.0690 |
| 8500 | — | — | **0.0686** | 0.0687 |
| 9000 | 0.0738 | — | — | 0.0685 |
| 10000 | — | — | **0.0687** | 0.0690 |
| 12000 | — | — | — | 0.0688 |
| 14000 | 0.0723 | 0.0710 | — | 0.0688 |
| 15000 | 0.0703 | 0.0701 | — | 0.0684 |
| 16000 | 0.0708 | 0.0710 | — | — |
| 17000 | 0.0701 | 0.0706 | — | — |
| 18000 | 0.0694 | — | — | — |

plus `ours_s8h` (weight-initialised continuation of the 15000 checkpoint):
500 -> 0.0689.

Reading:
- **avg host converges cleanly**: 0.0959 -> 0.0694 over 1000 -> 18000 steps,
  monotone apart from one ~0.004 wiggle at 5000.
- **ours is flat and stable at 0.0686-0.0695 for all 11 recovered points, from
  step 500 to step 10000.** It starts below the host and never rises, i.e. the
  gamma-only adaptation neither diverges nor improves the action loss.
- **The gamma FROZEN run is indistinguishable from ours** (0.0684-0.0695 over
  500-15000; see the 30-point curve in the CSV). **Val action loss therefore
  cannot tell "Gamma moves" from "Gamma frozen"** —— which is exactly why the
  rollout had to be measured by hand.

## Rollout (the metric that matters)

`tiered_eval.py`, official LIBERO-Mem subgoal machine, 20 demos
(`demo_1..demo_20`), `exec=8`, `maxsteps=370`, `seed=42`.
`weighted_success = sum(tier)/(3*n)`; `tier` = completed 3-cycle repetitions.

| absolute step | weighted_success | 3 cycles | 2 cycles | 1 cycle | 0 | tier3 | strict |
|---|---|---|---|---|---|---|---|
| **host (untrained)** | **23.3%** | 0 | 4 | 6 | 10 | 0 | 0 |
| host (re-run, same ckpt) | 23.3% | 0 | 4 | 6 | 10 | 0 | 0 |
| 8000 | 15.0% | 1 | 1 | 4 | 14 | 1 | 0 |
| 10000 | 20.0% | 0 | 5 | 2 | 13 | 0 | 0 |
| 13000 | 20.0% | 0 | 3 | 6 | 11 | 0 | 0 |
| 15000 | 21.7% | 0 | 5 | 3 | 12 | 0 | 0 |
| **17000 (peak)** | **25.0%** | **2** | 3 | 3 | 12 | **2** | **1** |
| 19000 | 23.3% | 1 | 4 | 3 | 12 | 1 | 0 |
| 21000 | 18.3% | 0 | 4 | 3 | 13 | 0 | 0 |

- **15000 -> 21000 is a WEIGHT-INITIALISED continuation**
  (`--init-from-weights` on the 15000 `checkpoint.pt`), not a bit-exact resume:
  the first segment ran with `--no-fullstate`.
- Peak at 17000: first time any arm exceeded the host, and the first
  `strict_success` (3 cycles with no overshoot) anywhere in this line. The host
  never produced a single 3-cycle demo in either measurement.
- Then 25.0 -> 23.3 -> 18.3: two consecutive declines with the 3-cycle column
  falling to 0 —— the overfitting signature that stopped the run.
- **Statistical caveat:** at n=20 the success standard deviation is ~9.3%
  (~1.9 tier units). The 4-unit drop is ~2 sigma: a signal, not proof. The
  25.0 vs 23.3 gap is 1 unit and is NOT significant on its own.
- **No matched control was run** (the box lost its second GPU), so the peak
  cannot be attributed to the projection rather than to Gamma training itself.

## Constraint-audit evidence (the algorithm doing what it claims)

From the 15000-step run (`gamma-only_ours_s8g`):

- `abort = 0` for the entire run.
- `vmax_proj` ~1e-8 against `tau = 3e-4` (4 orders of margin).
- 229 constrained boundaries; `proj_n_candidates > 0` in **98.3%** of them —— the
  constraint is genuinely binding, not cosmetic.
- `proj_shift_ratio` (how far the projection moves the AdamW proposal):
  mean 4.0%, max 23.7%.
- `proj_max_viol_before` (how much the UNPROJECTED AdamW step would have
  violated, cosine scale, kappa=0.02): mean 0.023, max 0.143, >1e-4 in 95.5% of
  boundaries.

Raw-vs-proposal evidence (kept in `outputs/AUDIT_raw_vs_realized/` on the box):

| projection space | realized violation |
|---|---|
| raw gradient (`p.grad = g_task - corr`) | **5.4 x tau** (fails; cos(d*, d_theta_real) 0.56-0.63) |
| AdamW proposal (this method) | **<= tau** (certified; cos 1.0) |

## Provenance / what is lost

- Code: safe. Local `D:\RPBE_vla_fix` == GitHub `fix/avg-lora-clock` == the
  five file hashes deployed on the box.
- Results: recovered from the session transcript
  (`~/.claude/projects/D--RPBE/63764db6-*.jsonl`), because the box's
  `outputs/` was the only other copy.
- **Lost**: raw `train.log` files, all checkpoints (`best.pt`, `latest.pt`,
  `checkpoint.pt`, snapshots), the rollout chain logs as files. The numbers
  above survive; the artifacts do not.
- Missing for a conclusive result: 100+ demos on the endpoint (n=20 gives
  1.67% resolution), a per-demo paired test against the host, and the matched
  Gamma-TASK control (kappa=1.0; same code path with the projection provably
  inactive).
