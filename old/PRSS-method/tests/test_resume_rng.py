import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "train_supervised_prss_switch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_supervised_prss_switch_rng_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rng_state_normalizer_forces_cpu_uint8_contiguous():
    mod = _load_module()
    x = torch.arange(16, dtype=torch.int64)[::2]
    y = mod._cpu_byte_rng_state(x)
    assert y.device.type == "cpu"
    assert y.dtype == torch.uint8
    assert y.is_contiguous()


def test_cpu_torch_rng_roundtrip_works_after_normalization():
    mod = _load_module()
    saved = torch.get_rng_state().clone()
    try:
        torch.manual_seed(12345)
        expected_state = torch.get_rng_state().clone()
        _ = torch.rand(8)
        torch.set_rng_state(mod._cpu_byte_rng_state(expected_state))
        a = torch.rand(8)
        torch.set_rng_state(mod._cpu_byte_rng_state(expected_state))
        b = torch.rand(8)
        assert torch.equal(a, b)
    finally:
        torch.set_rng_state(saved)
