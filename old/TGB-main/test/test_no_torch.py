"""
Regression test for https://github.com/shenyangHuang/TGB/issues/127.

Simulates torch/torch_geometric being uninstalled and verifies that the
numpy-only import surface (dataset classes, ``Evaluator``, negative samplers,
``tgb.utils.utils``) still imports and works. Also locks in the intentional
boundary: the PyG-only wrapper modules (``dataset_pyg``) still require torch
and should fail with a clear ``ImportError`` rather than something more
confusing, since guarding them would strip their only purpose.
"""
import builtins
import importlib
import sys

import numpy as np
import pytest

TGB_MODULES = [
    "tgb.utils.utils",
    "tgb.utils.pre_process",
    "tgb.linkproppred.dataset",
    "tgb.linkproppred.evaluate",
    "tgb.linkproppred.negative_sampler",
    "tgb.linkproppred.thg_negative_sampler",
    "tgb.linkproppred.tkg_negative_sampler",
    "tgb.nodeproppred.dataset",
    "tgb.nodeproppred.evaluate",
]


@pytest.fixture
def no_torch(monkeypatch):
    """Simulate torch/torch_geometric being uninstalled and force TGB
    modules to re-import under that condition."""
    saved_modules = sys.modules.copy()
    for name in list(sys.modules):
        if name == "torch" or name.startswith(("torch.", "torch_geometric")) \
           or name.startswith("tgb."):
            sys.modules.pop(name, None)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith(("torch.", "torch_geometric")):
            raise ImportError(f"simulated: {name} is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    yield
    sys.modules.clear()
    sys.modules.update(saved_modules)


def test_core_modules_import_without_torch(no_torch):
    for name in TGB_MODULES:
        mod = importlib.import_module(name)
        assert getattr(mod, "torch", None) is None


def test_negative_sampler_works_without_torch(no_torch):
    negative_sampler = importlib.import_module("tgb.linkproppred.negative_sampler")
    sampler = negative_sampler.NegativeEdgeSampler(dataset_name="tgbl-mock")
    sampler.eval_set["test"] = {(0, 1, 0): [5, 6, 7]}
    neg = sampler.query_batch(
        np.array([0]), np.array([1]), np.array([0]), split_mode="test"
    )
    assert neg == [[5, 6, 7]]


def test_evaluator_works_without_torch(no_torch):
    evaluate = importlib.import_module("tgb.linkproppred.evaluate")
    evaluator = evaluate.Evaluator(name="tgbl-wiki", k_value=10)
    result = evaluator.eval(
        {
            "y_pred_pos": np.array([1.0]),
            "y_pred_neg": np.array([[0.5, 0.9, 2.0]]),
            "eval_metric": ["mrr"],
        }
    )
    assert result["mrr"] == pytest.approx(0.5)


def test_set_random_seed_works_without_torch(no_torch):
    utils = importlib.import_module("tgb.utils.utils")
    utils.set_random_seed(0)  # should not raise even though torch is None


def test_pyg_dataset_still_requires_torch(no_torch):
    with pytest.raises(ImportError):
        importlib.import_module("tgb.linkproppred.dataset_pyg")
