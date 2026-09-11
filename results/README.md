# UCI pair-link experiment results

Small (non-checkpoint) result files from the UCI temporal-link-prediction runs,
copied out of `outputs/` (which is gitignored, and whose `.pt` checkpoints are
12 MB each) so the curves and final numbers are version-controlled.

Path layout mirrors `outputs/`:

```
results/
  uci_formal_v2/<seed>_TGN_<hop>/<arm>/     # the formal arm sweep
  uci_gates/<gate>/                         # GATE-2/GATE-3 controls
  uci_probe/<probe>/                        # tree-wise kappa probe + solve smokes
```

## Files per run

| file | content |
| --- | --- |
| `val_history.jsonl` | one row per epoch: full-val `ap_all` (checkpoint selection metric), `auc_all`; `nll_*`/`paired_acc`/`*_hist` are NaN and `n_hist=0` because the `2obs_aligned` arm has no time-paired eval samples |
| `metrics.jsonl` | one row per epoch: train loss, gauges, and the `rpbe_constrain` block (mode/scope/kappa, per-epoch `epoch_summary`, per-group `last_group`) |
| `summary.json` | final `best_epoch`, `best_ap_all`, and the one-shot `test` score from `best.pt` |
| `config.json` | resolved run config (arm axes, lambda, optimizer split, CLI) |
| `window_diag.jsonl` | per macro-group per tau: `M_unique_trees`, threshold, records |
| `comparison_audit.json` | audit sidecar (commit, config hash, fixed features) |

## Arms

- `taskonly` — official TGN host, task head only, **no** RPBE / auxiliary.
- `ours` — TGN + RPBE Ky-Fan auxiliary, unconstrained.
- `ours_cstr` — **legacy** aggregate constrained RPBE (rejected: the aggregate
  half-space projection cancels tree-level conflicts, and its task accumulator
  added the running `p.grad` total instead of the per-batch delta).
- `ours_cstr_treewise` — **current**: per-(tree, interface) directions kept
  separate, group close solves the multi-half-space QP
  `min_d 1/2||d-t||^2 s.t. g_j^T d >= -kappa||g_j||||t||` (dual, FISTA),
  correction restricted to Gamma/compressor.  kappa = 0.05.
- `vanilla` / `nomem` / `nocomp` / `off` / `memcomp` — gate controls.

## Headline (best val AP)

| arm | best AP | note |
| --- | --- | --- |
| `taskonly` seed0 3hop (50 ep) | **0.9156** | strongest so far |
| `ours` seed0 3hop (50 ep) | 0.9129 | RPBE aux unconstrained |
| `ours` seed1 3hop (50 ep) | 0.9126 | |
| `ours_cstr` seed0 (warm-start to ep70) | 0.9126 | legacy algorithm |

Seed spread for the *same* arm is ~0.004 (3hop `taskonly`: 0.9156 / 0.9113 /
0.9114), i.e. comparable to the between-arm differences — arm comparisons need
matched budgets and more seeds before they are decidable.
