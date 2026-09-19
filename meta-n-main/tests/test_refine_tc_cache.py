"""Regression tests for the symptom2disease cache writer (spec B4, F193).

The loader's cache write goes through ``meta_n.utils.atomic_io.atomic_json_dump``
with ``indent=None``: the on-disk bytes must stay identical to the historical
hand-rolled ``json.dump(data, f)`` (compact, single line), and an interrupted
write must never leave a partial cache or a stray ``.tmp`` file behind.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

import meta_n.integrations.text_classification as tc


def _install_fake_datasets(monkeypatch):
    fake = types.ModuleType("datasets")

    def load_dataset(name):
        return {
            "train": [
                {"input_text": f"t{i}", "output_text": f"d{i % 3}"} for i in range(10)
            ],
            "test": [
                {"input_text": f"x{i}", "output_text": f"d{i % 3}"} for i in range(5)
            ],
        }

    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_cache_bytes_are_compact_json(tmp_path, monkeypatch):
    """Cache bytes == json.dumps(data) — the pre-migration json.dump(data, f)
    format (no indent). An indented write would silently change the bytes the
    exists()/json.load read-back path caches forever."""
    _install_fake_datasets(monkeypatch)
    data = tc._load_symptom2disease(tmp_path, max_val=3, max_test=None)
    cache = tmp_path / "symptom2disease" / "cached_s42_v3_tall.json"
    assert cache.read_bytes() == json.dumps(data).encode()


def test_cache_write_routes_through_atomic_json_dump(tmp_path, monkeypatch):
    """The writer delegates to the shared atomic helper with indent=None."""
    _install_fake_datasets(monkeypatch)
    calls: list[tuple] = []

    def spy(path, obj, *, indent=2):
        calls.append((path, indent))

    monkeypatch.setattr(tc, "atomic_json_dump", spy)
    tc._load_symptom2disease(tmp_path, max_val=3, max_test=None)
    assert len(calls) == 1
    path, indent = calls[0]
    assert path == tmp_path / "symptom2disease" / "cached_s42_v3_tall.json"
    assert indent is None


def test_interrupted_cache_write_leaves_no_tmp(tmp_path, monkeypatch):
    """A failing serialization leaves neither the cache nor a .tmp behind,
    and the exception propagates (caller warn/raise policy preserved)."""
    _install_fake_datasets(monkeypatch)

    real_dumps_dir = tmp_path / "symptom2disease"

    def boom(*args, **kwargs):
        raise TypeError("boom")

    # json is one shared module object: patching dump here also patches the
    # copy atomic_io calls, simulating a mid-write serialization failure.
    monkeypatch.setattr(tc.json, "dump", boom)
    with pytest.raises(TypeError):
        tc._load_symptom2disease(tmp_path, max_val=3, max_test=None)
    assert not (real_dumps_dir / "cached_s42_v3_tall.json").exists()
    assert not (real_dumps_dir / "cached_s42_v3_tall.json.tmp").exists()


def test_cache_read_back_roundtrip(tmp_path, monkeypatch):
    """Second load returns the cached split byte-for-byte semantics."""
    _install_fake_datasets(monkeypatch)
    first = tc._load_symptom2disease(tmp_path, max_val=3, max_test=None)
    second = tc._load_symptom2disease(tmp_path, max_val=3, max_test=None)
    assert second == first
