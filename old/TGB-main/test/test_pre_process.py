"""
Tests for CSV loaders in ``tgb/utils/pre_process.py``.

Each loader is exercised on a tiny synthetic CSV so the expected output shape and
id/offset logic can be checked exactly. No real dataset assets are needed.
"""
import numpy as np
import pytest

from tgb.utils.pre_process import load_edgelist_trade, load_edgelist_wiki


def test_load_edgelist_wiki(write_csv):
    # JODIE format: first row is a header (skipped by the loader), then
    # user_id, item_id, timestamp, state_label, feat0, feat1
    csv_text = (
        "user_id,item_id,timestamp,state_label,features\n"
        "0,0,0.0,0,0.1,0.2\n"
        "1,1,1.0,0,0.3,0.4\n"
    )
    fname = write_csv("wiki.csv", csv_text)

    df, msg, third = load_edgelist_wiki(fname)

    assert list(df.columns) == ["u", "i", "ts", "idx", "w"]
    assert list(df["u"]) == [0, 1]
    # destinations are offset by max(src) + 1  (== 2 here)
    assert list(df["i"]) == [2, 3]
    assert list(df["ts"]) == [0.0, 1.0]
    assert list(df["idx"]) == [0, 1]
    assert np.array_equal(df["w"].values, np.ones(2))
    assert msg.shape == (2, 2)  # two message/feature columns
    assert third is None


def test_load_edgelist_trade(write_csv):
    # header, then rows of: timestamp, source, destination, weight
    csv_text = "ts,u,v,w\n" "0,A,B,1.5\n" "1,B,C,2.0\n"
    fname = write_csv("trade.csv", csv_text)

    df, feat, node_ids = load_edgelist_trade(fname)

    # nodes are re-indexed to contiguous ids in first-seen order
    assert node_ids == {"A": 0, "B": 1, "C": 2}
    assert list(df["u"]) == [0.0, 1.0]
    assert list(df["i"]) == [1.0, 2.0]
    assert list(df["ts"]) == [0.0, 1.0]
    assert feat.shape == (2, 1)
    assert np.array_equal(feat.ravel(), np.array([1.5, 2.0]))
