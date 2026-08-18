"""
Shared pytest fixtures and configuration for the TGB test suite.

Network gating
--------------
Tests that hit the real dataset object store are marked ``@pytest.mark.network``.
They are skipped by default so the suite stays deterministic and offline. Opt in
with either::

    pytest --run-network
    TGB_RUN_NETWORK=1 pytest
"""
import io
import os
import zipfile

import pytest


# --------------------------------------------------------------------------- #
# network marker gating
# --------------------------------------------------------------------------- #
def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked 'network' that access the real dataset URLs",
    )


def _network_enabled(config) -> bool:
    if config.getoption("--run-network"):
        return True
    return os.getenv("TGB_RUN_NETWORK", "").lower() in ("1", "true", "yes")


def pytest_collection_modifyitems(config, items):
    if _network_enabled(config):
        return
    skip_network = pytest.mark.skip(
        reason="network test skipped; enable with --run-network or TGB_RUN_NETWORK=1"
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_zip_bytes():
    """
    Factory returning the raw bytes of an in-memory ``.zip`` archive.

    Call as ``make_zip_bytes({"tgbl-mock_edgelist.csv": "u,i,ts\\n0,1,0\\n"})``.
    Used by the download tests to mock a real dataset archive without touching
    the network.
    """
    def _make(files: dict) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for arcname, content in files.items():
                zf.writestr(arcname, content)
        return buf.getvalue()

    return _make


@pytest.fixture
def write_csv(tmp_path):
    """
    Factory that writes ``text`` to a file under ``tmp_path`` and returns its path.
    """
    def _write(filename: str, text: str) -> str:
        path = tmp_path / filename
        path.write_text(text)
        return str(path)

    return _write
