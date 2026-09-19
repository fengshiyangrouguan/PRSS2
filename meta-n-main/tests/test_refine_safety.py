"""Refinement-wave regression tests for meta_n/utils/safety.py.

Covers: F151 (single dunder-escape set for both block paths), F153
(task_id-literal guard sibling holes: startswith/endswith, match/case, dict
membership), F152/F234 (smoke-test wall-clock bound). No LLM / Docker / network.
"""

import time

import pytest

from meta_n.utils.safety import (
    _DUNDER_ESCAPE_NAMES,
    smoke_test_function,
    task_id_literal_branch,
    validate_code,
)


# --------------------------------------------------------------------------- #
# F151 — the ast.Attribute walk and the getattr string-literal check share one
# blocked set: every name in it must be blocked on BOTH paths.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(_DUNDER_ESCAPE_NAMES))
def test_attribute_and_getattr_dunder_blocks_share_one_set(name):
    assert validate_code(f"x.{name}")[0] is False
    assert validate_code(f"getattr(x, {name!r})")[0] is False


# --------------------------------------------------------------------------- #
# F153 — sibling spellings of the task_id-literal memorization branch.
# --------------------------------------------------------------------------- #


class TestTaskIdLiteralSiblings:
    def test_startswith_literal_flagged(self):
        assert task_id_literal_branch(
            "if task.task_id.startswith('task_5'):\n    pass"
        ) == "task_5"

    def test_endswith_literal_flagged(self):
        assert task_id_literal_branch(
            "if task.task_id.endswith('_v2'):\n    pass"
        ) == "_v2"

    def test_startswith_tuple_arg_flagged(self):
        assert task_id_literal_branch(
            "if task.task_id.startswith(('task_5', 'task_9')):\n    pass"
        ) == "task_5"

    def test_match_statement_flagged(self):
        code = (
            "match task.task_id:\n"
            "    case 'task_7':\n"
            "        x = 1\n"
            "    case _:\n"
            "        x = 0\n"
        )
        assert task_id_literal_branch(code) == "task_7"

    def test_dict_membership_flagged(self):
        assert task_id_literal_branch(
            "if task.task_id in {'task_3': 1}:\n    pass"
        ) == "task_3"

    def test_validate_code_rejects_startswith_spelling(self):
        ok, reason = validate_code("if task.task_id.startswith('task_5'):\n    pass")
        assert ok is False
        assert "task_id" in reason

    # --- negative controls: feature branching / literal-free forms stay allowed

    def test_description_startswith_not_flagged(self):
        assert task_id_literal_branch(
            "if task.description.startswith('x'):\n    pass"
        ) is None

    def test_literal_free_startswith_not_flagged(self):
        assert task_id_literal_branch(
            "if s.startswith(task.task_id):\n    pass"
        ) is None

    def test_match_on_other_attribute_not_flagged(self):
        code = "match task.category:\n    case 'x':\n        pass\n"
        assert task_id_literal_branch(code) is None

    def test_feature_neutral_method_not_flagged(self):
        assert task_id_literal_branch("tid = task.task_id.lower()\n") is None


# --------------------------------------------------------------------------- #
# F152/F234 — smoke_test_function is wall-clock bounded (#48 mirror).
# --------------------------------------------------------------------------- #


class TestSmokeTestWallClock:
    def test_hanging_module_level_code_is_bounded(self):
        # The hang payload SLEEPS instead of busy-spinning: the abandoned
        # daemon worker cannot be killed (documented leak), and a spinning
        # thread burns GIL time for the remainder of the pytest process,
        # slowing every later test. A bounded sleep exercises the same
        # timeout path and self-terminates.
        t0 = time.monotonic()
        passed, msg = smoke_test_function(
            "f", "import time\ntime.sleep(1.5)\ndef f(): pass", timeout=0.2
        )
        elapsed = time.monotonic() - t0
        assert passed is False
        assert "wall-clock" in msg
        assert elapsed < 2.0  # returned promptly, did not hang

    def test_healthy_helper_default_timeout(self):
        assert smoke_test_function("f", "def f(): return 1") == (True, "")

    def test_exception_message_shape_through_worker(self):
        passed, msg = smoke_test_function(
            "f", "raise ValueError('boom')\ndef f(): pass"
        )
        assert passed is False
        assert msg.startswith("exec failed: ValueError: boom")
