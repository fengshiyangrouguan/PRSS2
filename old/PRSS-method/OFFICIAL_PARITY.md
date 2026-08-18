# Official TGN parity contract

## 1. Byte-for-byte upstream anchor

`official_tgn/train_supervised.py` is byte-for-byte the upstream
`twitter-research/tgn/train_supervised.py` at commit
`d55bbe678acabb9fc3879c408fd1f2e15919667c`.

The TGN core files used by the derivative trainer are also archived from that same commit. Their
SHA256 hashes are frozen in `official_tgn/UPSTREAM_CORE_SHA256.json` and checked by tests. We do
**not** modify the upstream TGN source to implement PRSS.

The exact upstream script is executed as `official_reference`. It keeps the original frozen
self-supervised TGN encoder and trains only the upstream MLP node classifier. Its purpose is to
prove that the Wikipedia dynamic-node-classification task and pretrained checkpoint behave like
the official code. Its historical quirks are preserved rather than silently repaired.

## 2. One mother-derived comparison trainer

`experiments/train_supervised_prss_switch.py` is the only matched comparison implementation. It
keeps these upstream semantics in both `--mode vanilla` and `--mode prss`:

1. `get_data_node_classification` and timestamp ordering;
2. `get_neighbor_finder` over the training graph;
3. the upstream `TGN` constructor and same self-supervised checkpoint;
4. the full upstream `compute_temporal_embeddings(source,destination,destination,...)` call;
5. no source-only shortcut;
6. the upstream `utils.utils.MLP` decoder;
7. the original attention/sum aggregator, memory updater, message function, and message queue;
8. memory reset at each training epoch;
9. the same optimizer/lr and host-training switch in the matched vanilla/PRSS pair.

The derivative adds the same clean 70/15/15 validation/test protocol to both matched modes, AP/NLL
monitoring, best-checkpoint replay, and the PRSS switch. These additions are not method-specific.

## 3. Host training switch

For the first official-style comparison, use the default:

```text
FINETUNE_HOST=0
```

Both matched runs start from the same self-supervised checkpoint and keep the original TGN host
frozen. Vanilla trains only the node decoder. PRSS trains the node decoder plus its candidate,
outside, and future-reader modules; the upstream TGN host remains frozen.

A second matched experiment may set:

```text
FINETUNE_HOST=1
```

Then the **same** pretrained host is fine-tuned in both vanilla and PRSS. This is a separate
co-adaptation experiment and is never mixed with the frozen-host result.

## 4. Exact recursive insertion point

The upstream recursive constructor is not replaced. At recursive layer `l` the official aggregate
still computes its normal state from lower host-width states and the exact edge/time/mask tensors.
PRSS then forms a richer candidate from **all inputs of that official aggregate plus its output**:

\[
h_l=[h_l^{TGN};\phi_l(x_l^{pre})].
\]

The shared quotient is applied immediately:

\[
z_l=R_l h_l,
\]

and the parent recursion receives **only** `z_l` at the original TGN width. Because every lower
recursive call already returned `z`, no rich child state can bypass an earlier quotient.

At initialization `R_l=[I,0]`, so before the first spectral update the PRSS main forward is
numerically the upstream TGN forward.

## 5. Evaluation isolation

Validation and test clear PRSS tracing. They cannot accumulate `B^T B`, run an eigendecomposition,
or change `R`. The trainer audits Gram/SVD counts and the exact quotient matrices before and after
validation/test and fails if they change.
