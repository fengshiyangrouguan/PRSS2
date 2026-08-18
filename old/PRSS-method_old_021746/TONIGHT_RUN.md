# Tonight: PRSS first-look experiment

## Main task

Use the standard TGN **dynamic node classification** task on JODIE Wikipedia.
The target is the event-time `state_label` (user ban state), not random-negative link existence.
Both Vanilla TGN and PRSS-TGN are trained **end-to-end from the same random seed** on this same target.

Why this first:
- Wikipedia random-negative link prediction is near ceiling and is not a useful first-look stress test.
- Wikipedia dynamic node classification is an official TGN task and is materially harder.
- The already-preprocessed Wikipedia data can be reused tonight; no download is required.
- The PRSS continuation reader is trained on the same node-label target as the host task.

## Two-GPU schedule

- physical GPU 0: matched Vanilla TGN
- physical GPU 1: full PRSS-TGN

No DDP is used because TGN memory is chronological. The two independent matched runs are parallelized instead.

The default first-look run intentionally skips `response_only` to minimize wall-clock time. If full PRSS is promising,
run the deep-supervision control afterward with `RUN_RESPONSE_ONLY=1`.

## Run

```bash
cd /root/autodl-tmp/PRSS2/PRSS-method   # adjust only if you unpacked elsewhere
unset CUDA_VISIBLE_DEVICES
bash run_autodl.sh
```

Defaults for the first look:
- dataset: wikipedia
- TGN recursion layers: 2
- neighbors per layer: 10
- epochs: 15, patience: 3
- seed: 0
- candidate dim: 256; host interface width is read from TGN (172 on Wikipedia)
- auxiliary traced roots: 4 per task batch, positives retained first
- GPU0 vanilla / GPU1 full PRSS

Follow progress:

```bash
OUT=/root/autodl-tmp/PRSS2/PRSS-method/outputs/node_classification/wikipedia/seed_0

tail -f "$OUT/logs/gpu0_vanilla.log"
# second terminal:
tail -f "$OUT/logs/gpu1_prss_full.log"
```

Final comparison:

```bash
cat "$OUT/comparison.json"
```

Primary first-look metrics: test ROC-AUC, AP, NLL. Because labels are sparse, do not use accuracy.

## If the first-look result is promising

Run the response-only control, which separates auxiliary continuation supervision from spectral quotient selection:

```bash
RUN_RESPONSE_ONLY=1 bash run_autodl.sh
```

Completed vanilla/full runs are reused.

## Not tonight's default

`run_link_prediction_legacy.sh` retains the old future-link experiment only as a compatibility/ablation entry point.
Do not use it for the first-look result.


## Spectral implementation sanity
Full PRSS forms a training-only bank of continuation-conditioned future-reading matrices `B(C)`. The implementation streams `G=mean(B^T B)` and eigendecomposes it; this is algebraically identical to SVD of the vertically stacked future-operator matrix. PCA is never used by the `full` variant.

`pytest` is optional and is disabled in the default experiment runner; it is a development check, not a runtime dependency.
