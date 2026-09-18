# Per-demo recovery: which rollouts can be traced back to individual demos

Status after an exhaustive pass over every transcript in
`~/.claude/projects/D--RPBE/` (1324 occurrences of `[demo_N]` examined,
grouped, and attributed).

## Bottom line

**No rollout of the Stage8 measurement chain — including the 17000 peak — can
be traced back to individual demo identities.** What survives for those is the
distribution only (`{0: n0, 1: n1, 2: n2, 3: n3}` plus the strict / overshoot
flags). The per-demo lines were never persisted.

## Why: three independent mechanical reasons

1. **The auto-daemon filtered at the source.** `_evdaemon.sh` ran
   `tiered_eval.py ... 2>&1 | grep -E "tier_dist|weighted_success|TIERED_DONE"
   >> rollout_chain.log`. The `grep` sits *inside the pipe*, so
   `[demo_7] tier=2/3 ...` was discarded before any file was written. It only
   ever existed in a stdout stream that was not kept. This covers the
   continuation points **17000 / 19000 / 21000**.
2. **Every read of the other eval logs was itself a `grep`.** For
   `e8000_full.log`, `e10000_full.log`, `e13000_full.log`, `e15000_full.log`
   the command was always
   `grep -E "tier_dist|weighted_success|TIERED_DONE|EXIT=|Error" ... | tail -N`.
   The "`_full`" logs did contain per-demo lines — a health check even counted
   them with `grep -c "tier="` — but the lines were never printed into the
   transcript.
3. **The one raw `tail` I took of an eval log landed on a crash.** The
   `tail -20 e8000_full.log` captured the model-loading banner and a
   `FileNotFoundError` (missing `dataset_statistics.json`), not demos.

All three logs lived only on the box, which is gone.

## What survives, measurement by measurement

| measurement | protocol | summary survived | per-demo identities |
|---|---|---|---|
| host `avg_s42_stage5_official/best.pt` | 20 demos, exec 8, seed 42 | yes, twice, identical | **no** |
| host re-run (`host_rerun.log`) | same | yes | **no** |
| ours 8000 / 10000 / 13000 / 15000 | same | yes | **no** |
| **ours 17000 (peak)** | same | yes | **no** |
| ours 19000 / 21000 | same | yes | **no** |
| host 5-seed sweep (exec 4, demo 81-100) | 5 seeds | 4 of 5 seeds | **no** |
| r1 chain, `gamma-lora` / `gamma-only` @250/500/1000/1500 | 20 demos, exec 8 | yes (all 8 points) | **partial** (see below) |
| early smoke / overfit runs | 3-5 demos, and val demos 82-100 | yes | fragments |

## The fragments that do survive

These are the only places where a demo index can be attached to a tier. All
are **partial** — a tail of an eval that was still running, or a mid-run
snapshot.

### r1 chain — `gamma-lora_s42_r1 :: snapshot_500`, weighted_success 13.3% (8/60)

Demos 1-11 were not printed; demos 12-20 were:

| demo | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| tier | 0 | 0 | 1 | 0 | 1 | 2 | 1 | 0 | 0 |

Subtotal from these 9: 5 tier units, ≥1cycle 4, ≥2cycle 1. The full-eval
summary is 8 units / ≥1cycle 6 / ≥2cycle 2, so demos 1-11 contributed the
remaining 3 units, 2 more ≥1cycle and 1 more ≥2cycle — **but which ones is not
recorded.**

### Early smoke eval (3 demos)

`demo_1 tier=3/3 peak=2 done=True fin=True q=46`; `demo_2 tier=0/3`;
`demo_3 tier=0/3`. This is the first 3/3 any of our arms produced.

### Early mid-run snapshots (partial, 5 demos)

Two snapshots exist with demos 1-5 only, e.g.
`demo_1 tier=1/3, demo_2 tier=1/3, demo_3 tier=0/3, demo_4 tier=0/3,
demo_5 tier=0/3`. Useful only as a sanity check that early rollouts were
already non-zero.

### Val-split demos 82-100

A few per-demo lines for demos 82-100 appear in the 5-seed sweep region, but
they are interleaved across seeds and checkpoints and cannot be attributed to
a single run with confidence.

## What the distribution alone still tells us about the peak

`ours @ absolute 17000`, 20 demos:

| cycles completed | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| demos | 12 | 3 | 3 | 2 |

`tier3_reached = 2`, `strict_success = 1` (one of the two 3-cycle demos
overshot, the other did not), `overshot_demos = 1`, `>=1cycle = 8`,
`>=2cycle = 5`.

## A check that is now impossible

With demo identities we could have asked whether the demos that complete three
cycles at 17000 are the *same* demos that do at 19000 — which would separate
"a stable capability on a few demos" from "noise". That check cannot be run
from what survives. It needs a re-run that keeps per-demo lines (i.e. does not
pipe the eval through a summary `grep`).
