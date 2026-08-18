# PRSS monitoring contract

Every run has `metrics.jsonl`, `summary.json`, and a `monitor/` directory.

```text
monitor/
  step_metrics.jsonl
  epoch_metrics.jsonl
  alerts.jsonl
  monitor_summary.json
  projection_snapshots/epoch_XXXX.pt
```

## Step-level monitoring

At `--monitor-every` batches the trainer records:

- task, structured-response, normalized spectral-tail, and unrestricted-monitor loss;
- positive count in the temporal batch;
- gradient L2 norms for the official decoder, candidate builders, structured readers, outside
  encoder, and (when enabled) the original TGN host modules;
- candidate norm, finite fraction, and per-coordinate batch standard-deviation statistics;
- future matrix `||B(C)||_F` mean/std/min/max and finite fraction by recursive layer;
- per-layer Gram/SVD counters and current quotient/spectrum diagnostics.

The unrestricted reader is optimized separately on detached `(C,h)` and cannot alter the main
representation. It exists only to tell us whether the linear-in-history structured reader is too
weak.

## Spectral monitoring

Per recursive interface:

- number of future-reader Gram updates;
- number of actual SVD/eigendecomposition solves;
- effective predictive rank;
- leading eigenvalues of `G=E[B^TB]`;
- energy at `k/4`, `k/2`, host width `k`, and full candidate width;
- tail at `k`;
- current `R` coverage of the live Gram;
- Gram trace and relative symmetry residual;
- relative row-orthogonality error `||RR^T-I||_F/sqrt(k)`;
- projector distance and principal-angle statistics between successive spectral solves;
- predictive operator energy before/after the solve and its gain;
- mean future-reader matrix Frobenius norm.

Every real solve prints immediately:

```text
SVD_UPDATE step=... layer=... total=... rank=... energy@k=... tail@k=... proj_dist=... gain=...
```

Periodic console lines print cumulative `svd_total`; they cannot repeat the earlier misleading
`svd=0` display.

## Hard implementation gates

The run fails by default if a monitored loss/candidate/reader matrix becomes non-finite, if a
quotient loses row orthogonality, if the Gram loses symmetry, or if validation/test changes Gram,
SVD counts, or `R`.

At the end of held-out test, `embedding_dims_observed` records the actual source/destination/
negative TGN widths. PRSS must still expose exactly the original host width.

## Reading the experiment

The exact upstream run is only a provenance/task anchor. The method comparison is matched
`vanilla_matched` versus `prss_matched` under the same host-training switch and clean split.

The most important mechanism evidence is not task AP/AUC alone. Also inspect whether:

1. structured response approaches the unrestricted monitor;
2. `B(C)` remains finite/non-collapsed;
3. predictive rank/spectrum is nontrivial;
4. SVD updates actually occur and stabilize;
5. `R` captures more reader-operator energy after each solve;
6. validation/test leave the learned quotient frozen.
