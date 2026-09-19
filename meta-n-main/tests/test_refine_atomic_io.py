"""Regression tests for ``atomic_json_dump`` (cluster C10b, spec B4, F193).

Canonical home: ``meta_n/utils/atomic_io.py`` (the orchestrator re-exports it
for its 9 in-module call sites).

Contract: write JSON to ``<path>.tmp`` then rename into place; on ANY failure
remove the tmp file and re-raise (callers keep their own warn/raise policy);
payload bytes identical to ``json.dump(obj, f, indent=2)``.
"""

import json

import pytest

from meta_n.utils.atomic_io import atomic_json_dump
from meta_n.core.evolutionary_orchestrator import atomic_json_dump as _eo_reexport


def test_orchestrator_reexport_is_same_object():
    assert _eo_reexport is atomic_json_dump


def test_happy_path_bytes_identical_to_json_dump(tmp_path):
    obj = {"a": 1, "b": [1, 2, 3], "c": "x", "nested": {"d": None}}
    path = tmp_path / "out.json"
    atomic_json_dump(path, obj)
    assert path.read_text() == json.dumps(obj, indent=2)
    assert not (tmp_path / "out.json.tmp").exists()


def test_accepts_str_path(tmp_path):
    path = tmp_path / "out.json"
    atomic_json_dump(str(path), [1.0, 0.5])
    assert json.loads(path.read_text()) == [1.0, 0.5]


def test_failure_removes_tmp_and_preserves_original(tmp_path):
    path = tmp_path / "out.json"
    path.write_text('{"original": true}')
    with pytest.raises(TypeError):
        atomic_json_dump(path, {"bad": object()})  # unserializable
    # The destination never saw the partial write; the tmp file is gone.
    assert path.read_text() == '{"original": true}'
    assert not (tmp_path / "out.json.tmp").exists()


def test_overwrites_existing_destination(tmp_path):
    path = tmp_path / "out.json"
    atomic_json_dump(path, {"v": 1})
    atomic_json_dump(path, {"v": 2})
    assert json.loads(path.read_text()) == {"v": 2}


def test_custom_indent_matches_json_dump(tmp_path):
    obj = {"k": [1, 2]}
    path = tmp_path / "out.json"
    atomic_json_dump(path, obj, indent=4)
    assert path.read_text() == json.dumps(obj, indent=4)
