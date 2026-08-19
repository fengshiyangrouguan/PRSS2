# PRSS strict implementation specification

## 0. Non-negotiable runtime contract

For every recursive interface type/layer \(\tau\), the host model fixes an external width
\(k_\tau\). PRSS may construct a wider candidate \(d_\tau\ge k_\tau\), but the parent receives
only

\[
z_v=R_\tau h_v,\qquad R_\tau\in\mathbb R^{k_\tau\times d_\tau},\qquad
R_\tau R_\tau^\top=I.
\]

There is no post-hoc compression stage. Once a child occurrence returns `z`, no parent-side path
may recover or access that child's discarded coordinates.

The only quotient solver in the current runtime is the predictive future-operator spectral solve:

\[
B(C)\;\longrightarrow\;G_\tau=E[B(C)^TB(C)]\;\longrightarrow\;
\operatorname{eigh}(G_\tau)\;\longrightarrow\;R_\tau.
\]

## 1. Contextual predictive equivalence

For history/subtree \(H\) and legal upper continuation \(C[\square]\), the ideal equivalence is

\[
H\sim H'\iff P(Y\mid C[H])=P(Y\mid C[H'])\quad\forall C.
\]

Because any parent+sibling+higher-path composition induces another legal continuation for the
child, this all-context equivalence is a congruence of the recursive constructors.

Natural data cannot enumerate all counterfactual contexts, so PRSS learns a supported-context,
fixed-rank approximation.

## 2. Candidate state at a TGN recursive occurrence

The original TGN recursive aggregate is retained. All lower states it receives are already
host-width quotient states from previous recursive calls.

For layer \(l\), let the official aggregate inputs be

\[
x_l^{pre}=
[z_l^{src},\phi(0),\{z_l^{nbr},\phi(\Delta t),e,mask\}].
\]

The official aggregate first produces its normal host-width state \(h_l^{TGN}\). PRSS enriches the
same constructor result with a lightweight function of **all exact aggregate inputs**:

\[
h_l=[h_l^{TGN};\phi_l(x_l^{pre})]\in\mathbb R^{d_l}.
\]

Then immediately

\[
z_l=R_lh_l
\]

is returned to the parent. The extra coordinates never propagate directly.

Initialization is

\[
R_l=[I\;0],
\]

so before the first spectral solve the main PRSS forward is numerically the official TGN forward.

## 3. Training-only outside continuation

Each traced occurrence \(v\) receives an outside state \(c_v=O(C_v)\). The current candidate
\(h_v\) itself is forbidden from `c_v`.

For a child of parent \(p\), the encoder may use:

- the parent's outside state;
- exact parent-local edge/time/mask constructor metadata;
- child relation/role and time gap;
- detached pre-quotient candidates of **other** siblings;
- higher continuation inherited from the parent.

The target label is never a context input. It appears only in losses.

## 4. Conditional future-reading matrices

For binary node classification, each occurrence has

\[
B_v=B_\tau(c_v)\in\mathbb R^{1\times d_\tau},\qquad b_v=b_\tau(c_v),
\]

with structured response

\[
\hat y_v=b_v+B_vh_v.
\]

The response loss uses the same node label as the main task.

A stronger unrestricted `MLP(c,h)` exists only as a diagnostic. Its inputs are detached, it has a
separate optimizer, it never affects the main representation, and it never contributes to the
spectral statistic.

## 5. One operator bank per shared recursive interface

PRSS does **not** solve one quotient per message occurrence. Each occurrence supplies one
future-reading operator, but all occurrences of the same interface share one quotient.

Conceptually,

\[
\mathcal B_\tau=
\begin{bmatrix}
B(C_1)\\B(C_2)\\\vdots
\end{bmatrix}.
\]

The predictive Gram is

\[
G_\tau=E[B(C)^TB(C)]=\mathcal B_\tau^T\mathcal B_\tau/N.
\]

At host budget \(k_\tau\), the analytic optimum of

\[
\min_{RR^T=I}\;E\|B(C)(I-R^TR)\|_F^2
\]

is the top-\(k_\tau\) right singular subspace of \(\mathcal B_\tau\), equivalently the top
eigenvectors of \(G_\tau\).

The implementation streams an EMA of batch `B^T B` and never materializes the giant bank.

## 6. Neural/spectral alternating objective

\[
L=L_{task}+\lambda_{resp}L_{resp}+\lambda_{spec}L_{spec},
\]

where

\[
L_{spec}=E\frac{\|B(C)(I-R^TR)\|_F^2}{\|B(C)\|_F^2+\epsilon}.
\]

`R` is a buffer, never a gradient parameter. Neural modules are updated by ordinary backprop.
After the Gram warm-up, every spectral interval solves the rank-constrained subproblem, then uses
orthogonal Procrustes to align the new basis to the old coordinates.

When the predictive Gram rank is below host width, all identified predictive directions are kept;
the remaining null-space directions are completed from the previous quotient rather than chosen by
an arbitrary eigensolver rotation.

Default strict hyperparameters follow the earlier specification:

```text
lambda_response = 1.0
lambda_spectral = 0.1
gram_ema         = 0.05
spectral_warmup  = 200 steps
spectral_interval= 200 steps
outside_dim      = 64
```

## 7. Inference boundary

At validation/test/deployment:

- no outside trace is constructed;
- no future reader is needed;
- no Gram statistic is updated;
- no eig/SVD is run;
- learned \(R_\tau\) is frozen;
- every recursive state remains exactly the host-required width.

The trainer audits these invariants before/after held-out evaluation.

## 8. Official TGN benchmark adaptation

The byte-for-byte upstream dynamic-node-classification script is retained and run as an exact
reference. The matched `vanilla|prss` derivative preserves the full official
`compute_temporal_embeddings(source,destination,destination,...)` path and official MLP decoder.

The initial official-style comparison keeps the pretrained TGN host frozen in both matched modes.
This intentionally follows the official node-classification protocol. A separate
`FINETUNE_HOST=1` experiment may co-adapt the host in both modes; it is reported separately.

Using a self-supervised TGN checkpoint is therefore a benchmark-specific initialization inherited
from the official task protocol. PRSS itself is active from the first node-classification step;
there is no node-task vanilla warm-up followed by offline compression.

## 9. Required monitors

Implementation must expose, per layer:

- `B(C)` norm/finite statistics;
- Gram trace/symmetry and update count;
- actual SVD/eigh update count;
- predictive rank/eigenvalues;
- energy at host-width fractions and spectral tail;
- current-quotient live Gram coverage;
- row-orthogonality error;
- projector distance/principal angles;
- captured predictive energy before/after spectral update;
- candidate norm/coordinate variance/finite fraction;
- main/reader/outside/candidate/host gradient norms;
- structured-vs-unrestricted response quality;
- validation/test spectral-isolation audit.
