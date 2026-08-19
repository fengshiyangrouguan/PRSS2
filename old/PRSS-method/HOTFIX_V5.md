# PRSS hotfix v5

This patch addresses two issues exposed by the v4 run:

1. `B.detach()` was stored for the operator bank/monitor. `detach()` shares storage with the live autograd output. The pre-backward finiteness checks showed B finite and all reader/outside/builder parameters remained finite after the optimizer step, yet the later monitor reported the stored B as non-finite. The operator-bank snapshot is now `B.detach().clone()` (and outside contexts likewise), making the pre-update statistic immutable.
2. v4 counted a spectral solve as an update even when the trust-region deployment accepted step 0. That incorrectly activated `L_spec` against the untouched identity-compatible `R=[I,0]`. `L_spec` now activates only after the deployed R has actually moved away from that initialization.
3. The damped live spectral deployment now uses a bounded, backtracked ascent step on the same predictive-energy objective, with fallback toward the exact SVD target. `spectral_step_size=1` still deploys the exact analytic SVD solution and the explicit-stack SVD equivalence tests remain exact.

No PCA or alternative reduction is introduced. The predictive Gram and exact spectral target remain `G=E[B^T B]` and its top-k eigenspace.

The v4 rolling checkpoint schema is unchanged, so a crashed run can resume from its last rolling checkpoint.
