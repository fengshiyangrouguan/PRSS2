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
  rollout had to be measured by hand, and why val is not the checkpoint
  selector (see "Checkpoint selection" below).

## The host: which avg checkpoint, and why that one

Every arm in this document starts from **`avg_s42_stage5_official/best.pt`**:
the avg arm (merge = plain average) of the Stage5 official-budget run,
diffusion action head, ~18k optimizer steps, checkpoint dated 2026-09-10
17:46.  It is the host because it is the only checkpoint in this line that
could do the task at all.  All three rows below were measured with the same
tiered protocol used everywhere in this document (demo_1..demo_20, exec=8,
maxsteps=370, seed 42):

| arm | head | steps | weighted_success | >=1 cycle |
|---|---|---|---|---|
| **avg host** `avg_s42_stage5_official/best.pt` | diffusion | 18k | **23.3%** | 10/20 |
| `gamma-task_s42_s8` | reg-head | 8k | 0.0% | 0/20 |
| `gamma-rpbe_s42_s8` | reg-head | 8k | 3.3% | 2/20 |

The two fine-tuned arms had already destroyed the host (reg-head + every
module in AdamW + long budget), so the host for Stage8 could not be any
checkpoint lying around: it had to be one that still rolls out, or every
comparison below would be 0% vs 0%.  The diffusion-head avg Stage5 checkpoint
was that checkpoint, and the search for it is what produced the 23.3% number.

### The host's own test result, in detail

`23.3%` = 14/60 tier units with `tier_dist = {0:10, 1:6, 2:4, 3:0}`: 10 demos
never finished a cycle, 6 finished one, 4 finished two, **0 ever finished
three**.  Measured twice, with identical output:

- during the search for a strong diffusion avg checkpoint;
- re-run under the identical protocol on 2026-09-13 (`eval_s8f/host_rerun.log`)
  —— **bit-identical `tier_dist`**.

That reproduction is why 23.3% is used as a fixed reference line rather than a
single-sample baseline.

### The host is noisier than one number suggests

A 5-seed sweep of the SAME checkpoint also exists (demo_81..demo_100, exec=4,
maxsteps=370).  It is the only multi-seed evidence we have about the host's
test behaviour:

| seed | 42 | 7 | 123 | 99 | 11 | mean (4 known) |
|---|---|---|---|---|---|---|
| weighted_success | 10.5% | 12.3% | **24.6%** | 15.8% | not recovered | 15.8% |

(seed 11's log was written on the box but its summary line never reached the
transcript; n=19 demos in that sweep, so the denominator is 57, not 60.)

So the host's rollout rate is somewhere in 10-25% depending on seed and demo
subset, and 23.3% is the seed-42 / demo_1..20 draw.  This is the same ~9-point
noise band that makes the 25.0-vs-23.3 gap insignificant, and it is why the
host is drawn as a line with a +/-1 sigma band in `rollout_curve.png` and as
the scatter in `avg_host_rollout.png`.

**Older, NOT comparable** (different data generation, protocol and
checkpoint): avg stage2 `20.0%` (6/30, 10 demos); avg stage3-wloss `15.0%`
(9/60) and `16.7%` (5/30).  Do not put these on the same axis as the table
above.

### Which step is `best.pt`, and what the seed-42 run looks like

`best.pt` is the **minimum-val checkpoint of that run's own val curve**, and
that run is seed 42.  The within-seed curve (14 recovered points, 1000 ->
18000) is in `val_curves.csv` and drawn in `avg_host_curve.png`:

| step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 |
|---|---|---|---|---|---|---|---|---|---|
| val | .0959 | .0871 | .0849 | .0780 | .0824 | .0754 | .0742 | .0757 | .0738 |

| step | 14000 | 15000 | 16000 | 17000 | **18000** |
|---|---|---|---|---|---|
| val | .0723 | .0703 | .0708 | .0701 | **.0694** |

The running minimum moved 9000 (.0738) -> 15000 (.0703) -> 17000 (.0701) ->
**18000 (.0694)**, so `best.pt` is the **18000-step** weights -- the last eval
of the run, because training was stopped at ~18k of its 20k budget.  The host
is therefore "the best this seed reached over that many steps", which is
exactly how it was selected.

**There is no within-seed rollout curve.**  Only `best.pt` was ever rolled
out; the planned snapshot eval (15k / 17.5k / 20k) never ran, and the
snapshots were deleted during disk pruning.  So for the host we have one
rollout number (measured twice, bit-identical) and a per-step val curve --
not a per-step rollout curve.  `avg_host_rollout.png` deliberately shows the
two rollout protocol variants and NOT a step axis, because none exists.

### Does the host's number depend on the selection rule?

`best.pt` is written by the trainer's best-val rule, i.e. the conventional
"report the best validation checkpoint" practice.  For this run **the two
rules agree**, and that is the useful part: training was stopped at ~18k of
its 20k budget when the eval was called, and the last val eval (18000,
0.0694) was also the running minimum.  So 23.3% is simultaneously

- the **endpoint** (last checkpoint of the run), and
- the **val-best** checkpoint of that run.

That removes the obvious objection "you compared our fixed endpoint against
the host's best-found checkpoint": both arms are reported at their endpoint,
and the host's endpoint happens to coincide with its val-best.  The rule
choice does not decide the comparison, so nothing needs to be switched after
the fact.

For completeness, applying val-selection to OURS (min val among the recovered
points) would pick **step 8500 (0.0686)**, where no rollout was ever measured;
its nearest measured neighbour is 8000 = **15.0%**.  So val selection would
not have improved our number —— which is the second reason not to switch: the
rule was fixed before the runs, and the rule that looks "more conventional"
would have produced a worse number for us, not a better one.

## Checkpoint selection: why the 15000 endpoint, not the 17000 peak

Every arm is reported at its **fixed-budget endpoint**, and the budget was
written down before the runs (`PRE_REGISTRATION.md`: `kappa=0.02`, `tau=3e-4`,
15000 steps, seed 42, host `avg_s42_stage5_official/best.pt`). Two things are
deliberately NOT used to choose a checkpoint:

1. **Not the rollout.** The 20 evaluation demos are the only rollout set that
   exists, so picking the best value on it is test-selection leakage. At n=20
   the success standard deviation is ~9.3% (~1.9 tier units), which makes the
   top of the table mostly noise: 25.0 vs 23.3 is one unit.
2. **Not val action loss either** —— and this is the less obvious one, because
   val is the metric that is *supposed* to be safe (it is held out and it is
   not the test set). It fails here for two independent reasons:

   - **No discriminative power.** The val curve for ours is FLAT:
     0.0686-0.0695 over all 11 recovered points, and the gamma-FROZEN run is
     point-for-point identical (0.0684-0.0695 over 30 points). A metric that
     cannot separate "Gamma moved" from "Gamma frozen" cannot select a Gamma
     checkpoint; ranking by it ranks by noise.
   - **No stopping signal.** Val never rises. On ours it is flat, and on the
     host it falls monotonically (0.0959 -> 0.0694). Read alone, val always
     says "keep training" —— which is exactly what it said at 15000, when the
     rollout had already turned.

So "just look at val" is not an option: it would have selected an arbitrary
point inside a flat line and would never have stopped.

The run was therefore extended past the budget (15000 -> 21000,
weight-initialised continuation) with one purpose: to test whether more steps
help. They do not —— 25.0 -> 23.3 -> 18.3, with the 3-cycle column falling to
0. **That decline is what makes the 15000 endpoint defensible**: it shows the
endpoint is a pre-registered truncation, not a hiding place, and that the
17000 peak cannot be promoted to "the result" without fitting the test set.
The peak is reported as a diagnostic and nothing more.

The host row sits in the same table, under the same protocol, for the same
reason: 23.3% is the host checkpoint that **all arms start from**
(`avg_s42_stage5_official/best.pt`), measured with the identical 20-demo
rollout. So every row is an adaptation measured against a common origin ——
no arm's number was chosen by sweeping rollouts, and no arm is "its best".

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
