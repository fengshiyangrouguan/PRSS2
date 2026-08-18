"""
Live reachability checks for the dataset download URLs.

These are marked ``network`` and are **skipped by default**. Enable with::

    pytest --run-network
    TGB_RUN_NETWORK=1 pytest

They prove that the URLs currently in ``tgb/utils/info.py`` actually serve data,
which is exactly what a URL migration (e.g. PR #128) needs to be checked against.
A single byte is requested per dataset (``Range: bytes=0-0``) so no dataset is
downloaded.
"""
import pytest
import requests

from tgb.utils.info import DATA_URL_DICT

pytestmark = pytest.mark.network

_TIMEOUT = 30  # seconds


@pytest.mark.parametrize("name", sorted(DATA_URL_DICT))
def test_dataset_url_is_reachable(name):
    url = DATA_URL_DICT[name]
    try:
        resp = requests.get(
            url,
            stream=True,
            timeout=_TIMEOUT,
            headers={"Range": "bytes=0-0"},
        )
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        pytest.fail(f"{name}: request to {url} failed: {exc}")

    try:
        # 200 (full body) or 206 (partial content, honoring the Range header)
        assert resp.status_code in (200, 206), (
            f"{name}: {url} returned HTTP {resp.status_code}"
        )
        # object stores should advertise a size / type for a real archive
        assert (
            "content-length" in resp.headers or "content-type" in resp.headers
        ), f"{name}: {url} response is missing content-length/content-type headers"
    finally:
        resp.close()
