# PRSS strict implementation contract (TGN carrier)

This file records the executable interpretation of `PRSS_method_spec_v1_2.md` used by the TGN experiment.

## Core mathematical block
For each legal training continuation `C_i`, the training-only reader outputs

`B_i = B_eta(C_i) in R^{p x d}`.

Conceptually stack all future-reading matrices

`B_stack = [B_1; B_2; ...]`.

The code does **not** materialize this large matrix. It streams the exactly equivalent Gram

`G = mean_i B_i^T B_i = (1/N) B_stack^T B_stack`.

Therefore the top-k eigenvectors of `G` are exactly the top-k right singular vectors of `B_stack`.
The deployed quotient is `R = V_top^T`, with host-prescribed output width k.

## TGN recursive interface
For layer l, lower child states are already quotients. The current candidate is built from the actual
pre-aggregation tuple `(source lower quotient, neighbor lower quotients, edge-time, edge feature, mask)`.
To preserve an exact vanilla initialization, the candidate is

`h_l = [F_TGN(preagg tuple), E_history(preagg tuple)]`.

`R_l=[I,0]` initially, so `z_l=F_TGN(...)`. After spectral updates, `R_l` may trade vanilla coordinates
for continuation-predictive history coordinates. The parent layer sees **only** `z_l`; it never sees h_l.

The history encoder uses a small learned multi-query pooling over every sampled joint neighbor token;
it is not a PCA and it does not use hand-written PCA dimensions.

## What PCA means
PCA exists only under the named `pca` ablation. `full` always uses future-reader operators B(C), their
predictive Gram, and analytic eig/SVD quotient selection.

## Spectral loss
The main method follows the method spec exactly:

`L_spec = ||B(C)(I-P_R)||_F^2 / (||B(C)||_F^2 + eps)`.

R is detached/non-gradient. `L_resp` prevents B collapse and trains both B and the candidate representation;
periodic eig/SVD updates R. A short response-only warmup is used before the first R update so the
identity-compatible initialization does not suppress new history coordinates before B has learned them.

## Task
Tonight's default runner uses dynamic node classification, not the near-ceiling Wikipedia random-negative
link prediction task. GPU0 runs vanilla then the response-only control; GPU1 runs full PRSS.
