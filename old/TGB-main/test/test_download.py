"""
Tests for the ``download()`` logic in the dataset classes.

These exercise the download / unzip path **offline** by mocking ``requests.get``
to return a real (small) in-memory zip archive. The heavy ``__init__`` (which
requires the actual dataset on disk) is bypassed via ``object.__new__`` so we can
drive ``download()`` in isolation.

Covered for both ``LinkPropPredDataset`` and ``NodePropPredDataset``:
  * happy path  -> archive is written and extracted into ``root``
  * skip path   -> nothing is downloaded when the data already exists
  * no-URL path -> ``ValueError`` when the dataset has no download URL
"""
import os
import os.path as osp

import pytest

from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.nodeproppred.dataset import NodePropPredDataset


class _FakeResponse:
    """Minimal stand-in for a streamed ``requests`` response."""

    def __init__(self, data: bytes):
        self._data = data
        self.headers = {"content-length": str(len(data))}

    def iter_content(self, chunk_size=1024):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]


# --------------------------------------------------------------------------- #
# helpers to build bare dataset instances (no real data required)
# --------------------------------------------------------------------------- #
def _bare_link_dataset(name, root, url, extra_meta=None):
    ds = object.__new__(LinkPropPredDataset)
    ds.name = name
    ds.url = url
    ds.root = str(root)
    ds.version_passed = False
    meta = {"fname": str(root) + "/" + name + "_edgelist.csv"}
    if extra_meta:
        meta.update(extra_meta)
    ds.meta_dict = meta
    return ds


def _bare_node_dataset(name, root, url, extra_meta=None):
    ds = object.__new__(NodePropPredDataset)
    ds.name = name
    ds.url = url
    ds.root = str(root)
    meta = {
        "fname": str(root) + "/" + name + "_edgelist.csv",
        "nodefile": str(root) + "/" + name + "_node.csv",
    }
    if extra_meta:
        meta.update(extra_meta)
    ds.meta_dict = meta
    return ds


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_link_download_writes_and_extracts(tmp_path, make_zip_bytes, mocker):
    name = "tgbl-mock"
    root = tmp_path / "tgbl_mock"  # intentionally does not exist yet
    zip_bytes = make_zip_bytes({f"{name}_edgelist.csv": "u,i,ts\n0,1,0\n"})
    get = mocker.patch(
        "tgb.linkproppred.dataset.requests.get",
        return_value=_FakeResponse(zip_bytes),
    )

    ds = _bare_link_dataset(name, root, url="https://example.com/tgbl-mock.zip")
    ds.download()

    get.assert_called_once()
    assert osp.exists(str(root) + f"/{name}.zip"), "archive was not written"
    assert osp.exists(ds.meta_dict["fname"]), "edgelist was not extracted"
    assert ds.version_passed is True


def test_node_download_writes_and_extracts(tmp_path, make_zip_bytes, mocker):
    name = "tgbn-mock"
    root = tmp_path / "tgbn_mock"
    zip_bytes = make_zip_bytes(
        {
            f"{name}_edgelist.csv": "u,i,ts\n0,1,0\n",
            f"{name}_node.csv": "node,label\n0,0\n",
        }
    )
    get = mocker.patch(
        "tgb.nodeproppred.dataset.requests.get",
        return_value=_FakeResponse(zip_bytes),
    )

    ds = _bare_node_dataset(name, root, url="https://example.com/tgbn-mock.zip")
    ds.download()

    get.assert_called_once()
    assert osp.exists(str(root) + f"/{name}.zip")
    assert osp.exists(ds.meta_dict["fname"])
    assert osp.exists(ds.meta_dict["nodefile"])


# --------------------------------------------------------------------------- #
# skip path: data already present -> no network call
# --------------------------------------------------------------------------- #
def test_link_download_skips_when_file_exists(tmp_path, mocker):
    name = "tgbl-mock"
    root = tmp_path / "tgbl_mock"
    os.makedirs(root)
    ds = _bare_link_dataset(name, root, url="https://example.com/tgbl-mock.zip")
    # pre-create the edgelist so download() should short-circuit
    with open(ds.meta_dict["fname"], "w") as f:
        f.write("u,i,ts\n")

    get = mocker.patch("tgb.linkproppred.dataset.requests.get")
    ds.download()
    get.assert_not_called()


def test_node_download_skips_when_files_exist(tmp_path, mocker):
    name = "tgbn-mock"
    root = tmp_path / "tgbn_mock"
    os.makedirs(root)
    ds = _bare_node_dataset(name, root, url="https://example.com/tgbn-mock.zip")
    for key in ("fname", "nodefile"):
        with open(ds.meta_dict[key], "w") as f:
            f.write("x\n")

    get = mocker.patch("tgb.nodeproppred.dataset.requests.get")
    ds.download()
    get.assert_not_called()


# --------------------------------------------------------------------------- #
# no-URL path: unsupported dataset
# --------------------------------------------------------------------------- #
def test_link_download_raises_without_url(tmp_path, mocker):
    root = tmp_path / "tgbl_nourl"
    ds = _bare_link_dataset("tgbl-nourl", root, url=None)
    get = mocker.patch("tgb.linkproppred.dataset.requests.get")

    with pytest.raises(ValueError):
        ds.download()
    get.assert_not_called()


def test_node_download_raises_without_url(tmp_path, mocker):
    root = tmp_path / "tgbn_nourl"
    ds = _bare_node_dataset("tgbn-nourl", root, url=None)
    get = mocker.patch("tgb.nodeproppred.dataset.requests.get")

    with pytest.raises(ValueError):
        ds.download()
    get.assert_not_called()
