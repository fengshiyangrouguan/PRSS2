"""Proxy-overfit tests (Step 8): sign-agnostic dev_test_gap (6.4a) +
task_id-literal branch detection (6.4b)."""

from meta_n.analysis.metrics import dev_test_direction
from meta_n.utils.safety import task_id_literal_branch, validate_code


# --------------------------------------------------------------------------- #
# 6.4a — sign-agnostic dev_test_gap (sign is property-dependent)
# --------------------------------------------------------------------------- #

def test_dev_above_test_is_overfit():
    assert dev_test_direction(0.9 - 0.5) == "overfit"      # dev > test


def test_test_above_dev_generalizes_not_overfit():
    # CO-Bench-gpt5.2: test > dev → GENERALIZES, must NOT be labeled overfit
    assert dev_test_direction(0.5 - 0.9) == "generalizes"


def test_matched_within_eps():
    assert dev_test_direction(0.0) == "matched"
    assert dev_test_direction(0.01) == "matched"           # within eps=0.02


# --------------------------------------------------------------------------- #
# 6.4b — task_id-literal branch detection (eval-set memorization)
# --------------------------------------------------------------------------- #

def test_literal_task_id_branch_flagged():
    assert task_id_literal_branch("if task.task_id == 'foo':\n    x = 1") == "foo"


def test_literal_task_id_branch_reversed_operands():
    assert task_id_literal_branch("if 'bar' == task.task_id:\n    x = 1") == "bar"


def test_feature_branch_not_flagged():
    assert task_id_literal_branch("if 'fever' in task.description:\n    x = 1") is None


def test_plain_task_id_use_not_flagged():
    assert task_id_literal_branch("tid = task.task_id\n") is None


def test_validate_code_rejects_task_id_literal():
    ok, reason = validate_code("if task.task_id == 'foo':\n    pass")
    assert ok is False
    assert "task_id" in reason


def test_validate_code_allows_feature_branch():
    ok, _ = validate_code("if 'fever' in task.description:\n    pass")
    assert ok is True
