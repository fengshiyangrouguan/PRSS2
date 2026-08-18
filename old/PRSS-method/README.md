# PRSS on the official TGN node-classification mother

This package is the clean experimental implementation of the current PRSS method. It keeps an
unaltered copy of the official TGN dynamic-node-classification code and adds PRSS through a
single mother-derived `vanilla|prss` switch.

## What to run tonight

Recommended recursive experiment:

```bash
cd /root/autodl-tmp/PRSS2/PRSS-method
unset CUDA_VISIBLE_DEVICES
N_LAYER=2 FINETUNE_HOST=0 bash run_autodl.sh
```

- GPU0 starts the matched vanilla run.
- GPU1 starts the matched PRSS run.
- when GPU0 finishes, it also runs the byte-for-byte upstream TGN node-classification script as a
  provenance/task anchor while GPU1 may still be finishing PRSS.
- no pytest installation is required; development tests run only if `RUN_TESTS=1` and pytest is
  already available.

For the exact upstream paper-default depth instead:

```bash
N_LAYER=1 FINETUNE_HOST=0 bash run_autodl.sh
```

For a separate co-adaptation experiment where the pretrained TGN host is fine-tuned identically in
both matched modes:

```bash
N_LAYER=2 FINETUNE_HOST=1 bash run_autodl.sh
```

Do not mix frozen-host and fine-tuned-host runs in one comparison.

## The method in one line

For every occurrence in the real recursive computation tree, PRSS learns how its legal upper
continuation reads the current rich history state. Same-layer future-reading matrices form a
conceptual operator bank. The deployed fixed-width quotient is the top right-singular subspace of
that bank, computed online as

\[
G_l=E[B(C)^TB(C)] \xrightarrow{\mathrm{eigh}} R_l.
\]

Every parent sees only `z_l=R_l h_l` at the original TGN width. There is no PCA or alternate
compression path.

Read `METHOD.md`, `OFFICIAL_PARITY.md`, and `MONITORING.md` before interpreting results.

## Important outputs

```text
outputs/node_classification/wikipedia/seed_0_l2/
  vanilla_matched/
    metrics.jsonl
    summary.json
    monitor/...
  prss_matched/
    metrics.jsonl
    summary.json
    monitor/...
  official_reference/
    summary.json
  logs/
    gpu0_vanilla.log
    gpu1_prss.log
    gpu0_official.log
  comparison.txt
```

Watch live:

```bash
OUT=/root/autodl-tmp/PRSS2/PRSS-method/outputs/node_classification/wikipedia/seed_0_l2

tail -f "$OUT/logs/gpu0_vanilla.log"
```

and in another terminal:

```bash
tail -f "$OUT/logs/gpu1_prss.log"
```

Real spectral solves appear as `SVD_UPDATE ...`; periodic logs report cumulative `svd_total`.
