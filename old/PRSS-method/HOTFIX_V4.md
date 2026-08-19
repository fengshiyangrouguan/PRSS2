# PRSS stability hotfix v4

This patch addresses the regression exposed by the official-mother L2 run.

- The SVD/eigh solve still computes the exact predictive target subspace of the accumulated future-reader Gram.
- Deployment of that target into the recursive TGN interface now uses a row-orthonormal trust-region step (default 0.25), accepted only when current-Gram captured predictive energy does not decrease. This restores the stable update policy used by the earlier PRSS spectral implementation and avoids a single SVD event abruptly replacing the live recursive coordinate system.
- The first spectral solve remains after warmup; spectral loss remains disabled before it.
- Non-finite diagnostics now distinguish outside-context failure, reader-output failure before backward, and parameter corruption after optimizer step.
- A rolling exact mid-epoch checkpoint is saved every 50 batches by default. Future failures can resume from `prss_matched/rolling_step.pt` instead of replaying the whole epoch.

The quotient target remains SVD-only; no PCA path is introduced.
