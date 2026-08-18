# Vendored TGB baseline modules

These files are copied **unmodified** from the TGB repository (`py-tgb` 2.3.0,
`modules/` directory — the modules are shipped with the repo but not inside the
pip-installable `tgb` package).  They provide the PyG-style TGN host used under the
TGB protocol:

- `memory_module.py` — `TGNMemory` (+ `DyRepMemory`)
- `emb_module.py` — `GraphAttentionEmbedding`, `TimeEmbedding`
- `neighbor_loader.py` — `LastNeighborLoader`
- `decoder.py` — `LinkPredictor`, `NodePredictor`
- `time_enc.py` — `TimeEncoder`
- `msg_func.py` — `IdentityMessage`
- `msg_agg.py` — `LastAggregator`, `MeanAggregator`
- `early_stopping.py` — `EarlyStopMonitor`

Do not edit these files.  If a checkpointing need arises that TGNMemory does not
expose (e.g. its raw message stores), implement it in `prss.training.checkpoint`,
not here.

One documented exception: `msg_agg.py` replaces the upstream `torch_scatter`
dependency (no wheels for newer torch) with a torch-native `scatter_reduce`
re-implementation of identical semantics; the patch is annotated in the file.
`__init__.py` registers a `modules` alias package so the upstream repo-root
import layout keeps working without `sys.path` surgery.
