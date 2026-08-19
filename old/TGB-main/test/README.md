# TGB test suite

Offline-by-default `pytest` suite guarding dataset metadata, download logic, and
core pure functions.

## Running

```bash
pip install -r requirements-dev.txt      # pytest, pytest-mock
pip install -e .                         # tgb + runtime deps

pytest                                   # offline suite (network tests skipped)
pytest --run-network                     # also hit the real dataset URLs
TGB_RUN_NETWORK=1 pytest                 # same, via env var
```

## Layout

| File | Scope | Network |
| --- | --- | --- |
| `test_info.py` | `tgb/utils/info.py` dict/URL invariants — the primary guard for URL migrations (e.g. PR #128) | no |
| `test_url_reachability.py` | each dataset URL actually serves data | **opt-in** (`network` marker) |
| `test_download.py` | `download()` / unzip logic (mocked HTTP) for link & node datasets | no |
| `test_evaluate.py` | `Evaluator` mrr/hits and error handling | no |
| `test_utils.py` | helpers in `tgb/utils/utils.py` | no |
| `test_pre_process.py` | CSV loaders in `tgb/utils/pre_process.py` | no |
| `test_negative_sampler.py` | `NegativeEdgeSampler` load/query behavior | no |
| `test_no_torch.py` | simulates torch/torch_geometric being uninstalled; guards issue #127 | no |

All dataset lists are derived from the dictionaries in `info.py`, so new datasets
are covered automatically. Shared fixtures and the `--run-network` gate live in
`conftest.py`.
