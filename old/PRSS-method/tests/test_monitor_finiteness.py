import torch

from prss.monitoring import matrix_stats


def test_matrix_stats_all_finite_is_exactly_one():
    B = torch.randn(97, 1, 256)
    st = matrix_stats({1: B})["1"]
    assert st["all_finite"] is True
    assert st["nonfinite_count"] == 0
    assert st["finite_count"] == B.numel()
    assert st["finite_fraction"] == 1.0


def test_matrix_stats_reports_exact_nonfinite_count():
    B = torch.randn(11, 1, 256)
    B[2, 0, 7] = float("nan")
    B[8, 0, 9] = float("inf")
    st = matrix_stats({1: B})["1"]
    assert st["all_finite"] is False
    assert st["nonfinite_count"] == 2
    assert st["finite_count"] == B.numel() - 2
    assert st["finite_fraction"] == (B.numel() - 2) / B.numel()
