"""
Tests for the link-prediction ``Evaluator`` (tgb/linkproppred/evaluate.py).

Uses small, hand-computed inputs so the metric values are exact.
"""
import numpy as np
import pytest

from tgb.linkproppred.evaluate import Evaluator

# For pos score 1.0 against neg scores [0.5, 0.9, 2.0]:
#   optimistic  rank = #(neg  > pos) = 1
#   pessimistic rank = #(neg >= pos) = 1
#   rank = 0.5 * (1 + 1) + 1 = 2  ->  mrr = 0.5, hits@10 = 1.0
POS = np.array([1.0])
NEG = np.array([[0.5, 0.9, 2.0]])


def test_eval_numpy_mrr_and_hits():
    evaluator = Evaluator(name="tgbl-wiki", k_value=10)
    result = evaluator.eval(
        {"y_pred_pos": POS, "y_pred_neg": NEG, "eval_metric": ["mrr"]}
    )
    assert result["mrr"] == pytest.approx(0.5)
    assert result["hits@10"] == pytest.approx(1.0)


def test_eval_perfect_prediction_gives_mrr_one():
    evaluator = Evaluator(name="tgbl-wiki", k_value=10)
    result = evaluator.eval(
        {
            "y_pred_pos": np.array([5.0]),
            "y_pred_neg": np.array([[0.1, 0.2, 0.3]]),
            "eval_metric": ["mrr"],
        }
    )
    assert result["mrr"] == pytest.approx(1.0)


def test_eval_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    evaluator = Evaluator(name="tgbl-wiki", k_value=10)
    result = evaluator._eval_hits_and_mrr(
        torch.tensor([1.0]),
        torch.tensor([[0.5, 0.9, 2.0]]),
        type_info="torch",
        k_value=10,
    )
    assert float(result["mrr"]) == pytest.approx(0.5)
    assert float(result["hits@10"]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #
def test_unknown_dataset_raises():
    with pytest.raises(NotImplementedError):
        Evaluator(name="not-a-real-dataset")


def test_missing_eval_metric_key_raises():
    evaluator = Evaluator(name="tgbl-wiki")
    with pytest.raises(RuntimeError):
        evaluator.eval({"y_pred_pos": POS, "y_pred_neg": NEG})


def test_unsupported_metric_raises():
    evaluator = Evaluator(name="tgbl-wiki")
    with pytest.raises(ValueError):
        evaluator.eval(
            {"y_pred_pos": POS, "y_pred_neg": NEG, "eval_metric": ["bogus"]}
        )
