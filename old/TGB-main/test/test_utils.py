"""
Tests for pure helper functions in ``tgb/utils/utils.py``.
"""
import numpy as np
import pandas as pd
import pytest

from tgb.utils.utils import (
    add_inverse_quadruples,
    find_nearest,
    load_pkl,
    save_pkl,
    split_by_time,
)


def test_find_nearest():
    assert find_nearest([1, 5, 10], 6) == 5
    assert find_nearest([1, 5, 10], 1) == 1
    assert find_nearest(np.array([0.0, 2.5, 9.9]), 3.0) == 2.5


def test_save_and_load_pkl_roundtrip(tmp_path):
    obj = {"a": [1, 2, 3], "b": np.arange(4)}
    fname = str(tmp_path / "obj.pkl")
    save_pkl(obj, fname)
    loaded = load_pkl(fname)
    assert loaded["a"] == [1, 2, 3]
    assert np.array_equal(loaded["b"], np.arange(4))


def test_add_inverse_quadruples_doubles_and_offsets():
    df = pd.DataFrame(
        {
            "u": [0, 1],
            "i": [2, 3],
            "ts": [0, 1],
            "idx": [0, 1],
            "w": [1.0, 1.0],
            "edge_type": [0, 1],
        }
    )
    out = add_inverse_quadruples(df)

    # two originals + two inverses
    assert len(out) == 4
    num_rels = df["edge_type"].nunique()  # == 2

    orig, inv = out.iloc[:2], out.iloc[2:]
    # inverse relation ids are offset by num_rels
    assert list(inv["edge_type"]) == [t + num_rels for t in df["edge_type"]]
    # inverse swaps source and destination
    assert list(inv["u"]) == list(df["i"])
    assert list(inv["i"]) == list(df["u"])
    # labels column added, all ones
    assert set(out["label"]) == {1.0}


def test_add_inverse_quadruples_requires_edge_type():
    df = pd.DataFrame({"u": [0], "i": [1], "ts": [0], "idx": [0], "w": [1.0]})
    with pytest.raises(ValueError):
        add_inverse_quadruples(df)


def test_split_by_time_groups_snapshots():
    # columns: [src, rel, dst, timestamp]
    data = np.array(
        [
            [0, 1, 2, 10],
            [3, 4, 5, 10],
            [6, 7, 8, 20],
        ]
    )
    snapshots = split_by_time(data)
    assert len(snapshots) == 2  # two distinct timestamps
    # each snapshot keeps only the first three columns (src, rel, dst)
    assert snapshots[0].shape == (2, 3)
    assert snapshots[1].shape == (1, 3)
    assert np.array_equal(snapshots[1], np.array([[6, 7, 8]]))
