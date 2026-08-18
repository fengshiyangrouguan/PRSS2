import os

# 208-core containers deadlock small BLAS/torch ops across hundreds of OpenMP
# threads.  Pin every thread layer to 1 BEFORE importing torch; env can override.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

"""Shared test configuration.

Inserts ``src/`` on sys.path so tests run both under an editable install and under a
plain ``python -m pytest`` / ``python -m unittest`` invocation.  All tests in this
repository use ``unittest.TestCase`` classes so both runners discover the same suite.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import os

import torch

# 208-core machines thrash tiny ops across 200+ OpenMP threads; keep a sane default.
# Env var OMP_NUM_THREADS overrides.
torch.backends.mkldnn.enabled = False  # oneDNN deadlocks on 208-core containers
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
