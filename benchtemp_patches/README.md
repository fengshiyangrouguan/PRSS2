# BenchTemp pin + local patch (UCI link-gate runs)

The UCI runner (`scripts/train_uci_link.py`) imports the **external** BenchTemp
repo's `model.tgn.TGN` / `utils.utils.get_neighbor_finder` from
`benchtemp/experimental_codes/tgn-jodie-dyrep` (PYTHONPATH, or the fallback
absolute path `/root/autodl-tmp/benchtemp/...`). Reproducibility therefore
requires the exact BenchTemp tree.

- Pinned BenchTemp commit: **`ca017fa`** (github.com/qianghuangwhu/benchtemp).
- Required local patch (this dir): `uci_benchtemp_local.patch`, which applies
  two hunks on top of `ca017fa`:
  1. `experimental_codes/tgn-jodie-dyrep/model/tgn.py` — monotonic-time clamp in
     `update_memory` (homogeneous graphs can interleave a node's src/dst events
     within one batch; the official "time in the past" assert then fires). The
     clamp lives INSIDE the `len(unique_nodes) > 0` guard: with start-mode
     memory, `update_memory` is also called on empty pending messages, where
     `torch.maximum(list, tensor)` would otherwise raise.
  2. `lp/dataloader.py` — `random.sample` needs a sequence under py3.11+
     (list() the set of ints).
- Apply with, from the benchtemp repo root:
  `git apply <path>/uci_benchtemp_local.patch`

Nothing in this patch changes the memory learning path or the protocol
semantics; it only guards the official time-ordering assert and the loader.
