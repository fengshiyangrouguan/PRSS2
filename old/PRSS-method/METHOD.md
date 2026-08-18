# PRSS — Predictive Recursive State Sheaf / predictive spectral quotient runtime

This file is the runtime contract for the current TGN experiment.

## 1. Recursive object being compressed

For each TGN recursive layer `l`, the host requires width `k_l` (Wikipedia TGN: 172). The original
TGN recursive aggregate receives only lower-layer quotient states and its legal temporal
edge/time inputs. From the exact inputs consumed by that aggregate, PRSS forms

\[
x_l^{pre}=[z_l^{src},\phi(0),\{z_l^{nbr},\phi(\Delta t),e,mask\}],
\]

and a richer pre-quotient candidate

\[
h_l=[h_l^{TGN};\phi_l(x_l^{pre})]\in\mathbb R^{d_l},\quad d_l\ge k_l.
\]

The deployed recursive state is

\[
\boxed{z_l=R_lh_l\in\mathbb R^{k_l}},\qquad R_lR_l^\top=I.
\]

The parent sees only `z_l`. There is no post-hoc compression and no path around an earlier
quotient. Initialization uses `R_l=[I,0]`, so step zero reproduces the upstream TGN state.

Layer 0 has `d_0=k_0` and therefore no dimensional compression. It is retained only as the
upstream leaf/base state needed by the recursion; it does **not** train a continuation reader,
does not contribute to the predictive operator bank, and does not run a spectral solve.
Nontrivial quotient learning begins at recursive aggregate layers `l>=1`.

## 2. What is SVD'd

**Not one message at a time, and not one giant tensor for one tree.**

Every traced occurrence `v` in a real source computation tree has its own training-only upper
continuation `C_v`: parent local/event information, siblings, higher outside path, root query
metadata, and the common final node-label task. The current `h_v` itself is excluded.

A conditional matrix reader produces

\[
B_v=B(C_v)\in\mathbb R^{1\times d_l}
\]

for binary node classification and predicts

\[
\operatorname{logit}P(Y\mid h_v,C_v)\approx b(C_v)+B(C_v)h_v.
\]

All occurrences of the same recursive interface/layer across all sampled training trees
conceptually form one operator bank

\[
\mathcal B_l=\begin{bmatrix}B(C_1)\\B(C_2)\\\vdots\end{bmatrix}.
\]

One quotient `R_l` is shared by that interface. Its optimal fixed-width predictive subspace is the
top-`k_l` **right singular subspace** of `mathcal B_l`.

The implementation streams the algebraically equivalent statistic

\[
\boxed{G_l=\mathbb E[B(C)^\top B(C)]
      =\mathcal B_l^\top\mathcal B_l/N}
\]

and periodically applies `torch.linalg.eigh(G_l)`. Thus the eigensolve is exactly the right-singular
subspace calculation without materializing the huge stacked operator bank.

There is no PCA, variance compression, learned `d->k` projection, or alternate reduction path in
this runtime.

## 3. Alternating neural/spectral training

The main task is the official TGN dynamic-node-classification label. PRSS uses

\[
L=L_{task}+\lambda_{resp}L_{resp}+\lambda_{spec}L_{spec},
\]

with

\[
L_{spec}=\frac{\|B(C)(I-R^\top R)\|_F^2}{\|B(C)\|_F^2+\epsilon}.
\]

`R` is a detached buffer. Neural backprop trains the candidate builder, structured future reader,
outside encoder, decoder, and optionally the matched TGN host. The stronger unrestricted reader
is a **monitor only**: detached inputs, a separate optimizer, no contribution to the main loss, and
never contributes to `G`.

After Gram warm-up, the spectral subproblem is solved exactly. If the identified predictive rank
is below host width, PRSS preserves those predictive directions and fills the unidentifiable null
space from the previous quotient, avoiding arbitrary null-space rotations. Orthogonal Procrustes
then aligns coordinates before installing the new `R`.

## 4. Causality

Main inference uses only event-time available history. Outside contexts and `B(C)` are training
only. The target label enters losses only, never the context encoder. Sibling candidates used by
the outside teacher are detached. Validation/test freeze all Gram and quotient state.

## 5. Official task comparison

The package contains the byte-for-byte upstream `train_supervised.py` as `official_reference`.
The matched comparison uses one mother-derived trainer with a single `vanilla|prss` switch. The
full upstream source/destination/destination TGN embedding call is retained; there is no shortcut.

Default tonight protocol is official-style frozen host in both matched modes. Set
`FINETUNE_HOST=1` only for a separate matched co-adaptation experiment.

## 6. Auxiliary weighting across a branching tree

The future-reader objective is defined only on compressive interfaces `l>=1`.  For each traced
root, occurrence losses are averaged *within layer first* and then averaged across represented
layers.  Thus a lower layer cannot dominate the neural auxiliary objective merely because an
`n_degree=10` computation tree contains roughly ten times as many occurrences there.  In contrast,
the spectral statistic remains per-interface: every valid occurrence of layer `l` contributes its
`B(C)^T B(C)` sample to that layer's own operator bank / Gram.
