# Vendored upstream twitter-research/tgn

## Provenance

- **Upstream**: https://github.com/twitter-research/tgn
- **Commit**: `d55bbe678acabb9fc3879c408fd1f2e15919667c`
- **Copied from**: `E:\project\PRSS2\old\PRSS-method\official_tgn\source\`
  (which is itself a byte-for-byte copy of the upstream commit, verified
  against `old/PRSS-method/official_tgn/UPSTREAM_CORE_SHA256.json`)
- **Purpose**: JODIE-protocol node classification (TGN paper protocol).
  The JODIE data loader and the PRSS adapter live outside this package;
  this folder is kept as close to upstream as possible so that
  provenance-based comparison against official numbers stays honest.

## File classification

| File | Change vs upstream |
|---|---|
| `model/tgn.py` | imports only (6 lines) + **local patch**: memory-update consistency assert `atol=1e-5` → `1e-4` (upstream tolerance fails intermittently under multi-process GPU sharing; see comment at the assert) |
| `model/time_encoding.py` | none (byte-for-byte) |
| `model/temporal_attention.py` | imports only (1 line) |
| `modules/memory.py` | none |
| `modules/memory_updater.py` | none |
| `modules/message_aggregator.py` | none |
| `modules/message_function.py` | none |
| `modules/embedding_module.py` | imports only (1 line) |
| `utils/utils.py` | none |

## Import rewrites (the only diff except the tgn.py patch above)

All `from utils.utils import ...` / `from modules.xxx import ...` /
`from model.xxx import ...` were prefixed with `rpbe.hosts.official_tgn.`.

## Not vendored

- `utils/data_processing.py` — hardcodes `./data/`; reimplemented cleanly in
  `src/prss/data/jodie.py` (identical split logic, `--data-dir` driven).
- `utils/preprocess_data.py` — superseded by `scripts/preprocess_jodie.py`.
- `train_self_supervised.py` / `train_supervised.py` / `evaluation/` — the
  official anchor numbers are produced by the old v1 pipeline; the new line
  consumes its *weights*, not its scripts.

## sha256 (current files, run after any change)

```
# from repo root:
# find src/prss/hosts/official_tgn -name "*.py" -not -name "__init__.py" | sort | xargs sha256sum
```
