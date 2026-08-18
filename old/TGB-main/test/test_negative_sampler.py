"""
Tests for ``NegativeEdgeSampler`` (tgb/linkproppred/negative_sampler.py).

The evaluation set is normally loaded from a pickle produced during dataset
processing. Here we either write a small pickle with ``save_pkl`` or seed the
in-memory ``eval_set`` directly, so no real dataset is required.
"""
import numpy as np
import pytest

from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.utils.utils import save_pkl


def test_constructor_rejects_unknown_strategy():
    with pytest.raises(AssertionError):
        NegativeEdgeSampler(dataset_name="tgbl-mock", strategy="bogus")


def test_load_eval_set_rejects_bad_split_mode():
    sampler = NegativeEdgeSampler(dataset_name="tgbl-mock")
    with pytest.raises(AssertionError):
        sampler.load_eval_set("whatever.pkl", split_mode="train")


def test_load_eval_set_missing_file_raises():
    sampler = NegativeEdgeSampler(dataset_name="tgbl-mock")
    with pytest.raises(FileNotFoundError):
        sampler.load_eval_set("/nonexistent/path/ns.pkl", split_mode="val")


def test_load_eval_set_roundtrip_and_query(tmp_path):
    eval_set = {(0, 1, 0): [5, 6, 7]}
    fname = str(tmp_path / "ns.pkl")
    save_pkl(eval_set, fname)

    sampler = NegativeEdgeSampler(dataset_name="tgbl-mock")
    sampler.load_eval_set(fname, split_mode="test")

    neg = sampler.query_batch(
        np.array([0]), np.array([1]), np.array([0]), split_mode="test"
    )
    assert neg == [[5, 6, 7]]


def test_query_batch_raises_when_eval_set_not_loaded():
    sampler = NegativeEdgeSampler(dataset_name="tgbl-mock")
    sampler.reset_eval_set(split_mode="test")  # sets eval_set["test"] = None
    with pytest.raises(ValueError):
        sampler.query_batch(
            np.array([0]), np.array([1]), np.array([0]), split_mode="test"
        )


def test_query_batch_raises_on_unknown_edge():
    sampler = NegativeEdgeSampler(dataset_name="tgbl-mock")
    sampler.eval_set["test"] = {(0, 1, 0): [5, 6, 7]}
    with pytest.raises(ValueError):
        sampler.query_batch(
            np.array([9]), np.array([9]), np.array([9]), split_mode="test"
        )
