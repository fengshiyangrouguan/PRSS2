"""Offline regression tests for audit fixes in meta_n/utils/safety.py.

Each test targets one confirmed audit finding and is written so it FAILS on the
original (pre-fix) code and PASSES after the fix. No LLM / Docker / network.
"""

from meta_n.utils.safety import (
    BLOCKED_ATTRIBUTES,
    BLOCKED_IMPORTS,
    task_id_literal_branch,
    validate_code,
)


# ---------------------------------------------------------------------------
# Finding #28: BLOCKED_ATTRIBUTES startswith-match wrongly rejected identifiers
# that merely START WITH eval/exec/__import__ (executor.run, evaluation.append,
# eval_scores.sort). The bare names must not be undotted prefix entries.
# ---------------------------------------------------------------------------
def test_finding_28_identifiers_starting_with_eval_exec_not_blocked():
    valid_snippets = [
        "executor.run('cmd')",
        "evaluation.append(1)",
        "eval_scores.sort()",
        "execute_plan.run()",
        "execution_trace.append(x)",
        "evaluator.score(candidate)",
    ]
    for src in valid_snippets:
        ok, msg = validate_code(src)
        assert ok, f"{src!r} should be VALID, got blocked: {msg}"

    # The bare undotted prefix entries must be gone from BLOCKED_ATTRIBUTES.
    assert "eval" not in BLOCKED_ATTRIBUTES
    assert "exec" not in BLOCKED_ATTRIBUTES
    assert "__import__" not in BLOCKED_ATTRIBUTES


def test_finding_28_direct_calls_still_blocked():
    # Regression guard: the genuinely dangerous direct-call forms must stay
    # blocked via the ast.Name branch.
    for src in ("eval('1+1')", "exec('x=1')", "__import__('os')"):
        ok, _ = validate_code(src)
        assert not ok, f"{src!r} should still be blocked"
    # And dotted os.exec* / os.spawn* family matching must remain intact.
    for src in ("os.execv('/bin/sh', [])", "os.spawnl(0, 'sh')"):
        ok, _ = validate_code(src)
        assert not ok, f"{src!r} should still be blocked"


# ---------------------------------------------------------------------------
# Finding #64: BLOCKED_IMPORTS omitted posix/nt (C-level os twins exposing
# system/execv/popen/fork/spawn), letting `import posix; posix.system(...)`
# bypass the entire blocklist on host-exec benches.
# ---------------------------------------------------------------------------
def test_finding_64_posix_import_and_call_blocked():
    ok, msg = validate_code("import posix as p\np.system('rm -rf /')")
    assert not ok, f"posix import + call must be blocked, got: {ok}/{msg}"

    ok, _ = validate_code("import posix\nposix.system('id')")
    assert not ok, "import posix must be blocked"

    ok, _ = validate_code("import nt\nnt.system('dir')")
    assert not ok, "import nt must be blocked"

    # The blocklist sets must contain the new entries.
    assert "posix" in BLOCKED_IMPORTS
    assert "nt" in BLOCKED_IMPORTS
    assert "posix.system" in BLOCKED_ATTRIBUTES


# ---------------------------------------------------------------------------
# Finding #73: task_id container-membership test bypassed the eval-set
# memorization guard — `task.task_id in ('task_5','task_12')` slipped through.
# ---------------------------------------------------------------------------
def test_finding_73_container_membership_task_id_branch_blocked():
    # Tuple membership
    assert task_id_literal_branch(
        "if task.task_id in ('task_5', 'task_12'): pass"
    ) == "task_5"
    # Set membership
    assert task_id_literal_branch(
        "if task.task_id in {'task_5'}: pass"
    ) == "task_5"
    # List membership
    assert task_id_literal_branch(
        "if task.task_id in ['task_9']: pass"
    ) == "task_9"

    ok, msg = validate_code("if task.task_id in ('task_5', 'task_12'):\n    score = 1.0")
    assert not ok, f"container-membership task_id branch must be blocked, got: {ok}/{msg}"


def test_finding_73_existing_equality_branch_still_blocked():
    # Regression guard: the scalar == form the guard already handled stays blocked.
    assert task_id_literal_branch("if task.task_id == 'task_5': pass") == "task_5"
    # And a feature branch (no task_id) is still allowed.
    assert task_id_literal_branch("if 'fever' in task.description: pass") is None
    ok, _ = validate_code("if 'fever' in task.description:\n    score = 1.0")
    assert ok
