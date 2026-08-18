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
