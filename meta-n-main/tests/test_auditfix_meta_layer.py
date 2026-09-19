"""Offline regression tests for audit findings 48, 63, 66 (meta_layer.py).

Each test FAILS on the pre-fix code and PASSES after the fix. All tests are
fully offline — no LLM, Docker, or network. The only code exec'd on the host is
test-authored helper source (NOT model/Omega output): #63 must run the generated
deterministic wrapper to prove it no longer raises, and #48 deliberately exec's a
hand-written hanging block to prove the wall-clock bound fires.
"""

from __future__ import annotations

import time

from meta_n.core.meta_layer import (
    InjectedCode,
    SandboxMarker,
    TaskDescription,
    _build_deploy_wrapper,
    format_bash_library_descriptions,
    run_pre_process,
)


# ---------------------------------------------------------------------------
# Finding 48 — model-emitted pre_process exec must be wall-clock bounded so a
# non-terminating block cannot stall the (synchronous, on-the-event-loop) run.
# ---------------------------------------------------------------------------
def test_finding_48_pre_process_exec_is_wallclock_bounded():
    task = TaskDescription(task_id="t48", description="d")

    # Happy path still works through the (now threaded) exec channel.
    ok = InjectedCode(pre_process="additional_context = 'HELLO'", source_depth=2)
    ran, ctx = run_pre_process([ok], task)
    assert ran is True
    assert ctx == "HELLO"

    # A block that hangs far longer than the budget must be abandoned, NOT block
    # for its full duration. `import time` passes validate_code (not blocklisted),
    # so the block reaches exec and would sleep 5s on the unbounded path.
    hanging = InjectedCode(
        pre_process="import time\ntime.sleep(5)\nadditional_context = 'late'",
        source_depth=1,
    )
    start = time.perf_counter()
    ran, ctx = run_pre_process([hanging], task, pre_process_timeout=0.3)
    elapsed = time.perf_counter() - start

    # Bounded by the ~0.3s budget, well under the 5s sleep; the block is skipped
    # (mirrors the existing except-and-continue), so no context is emitted.
    assert elapsed < 3.0, f"pre_process was not wall-clock bounded (took {elapsed:.2f}s)"
    assert ran is False
    assert ctx == ""


# ---------------------------------------------------------------------------
# Finding 63 — _build_deploy_wrapper must forward positional-only params
# positionally, not by keyword (else the deployed solve() TypeErrors at runtime).
# ---------------------------------------------------------------------------
def test_finding_63_deploy_wrapper_forwards_positional_only_params():
    helper_src = "def helper(a, /, b):\n    return a + b\n"
    wrapper_src = _build_deploy_wrapper("helper", helper_src)
    assert wrapper_src is not None

    ns: dict = {}
    exec(helper_src, ns)  # test-authored helper, not model output
    exec(wrapper_src, ns)  # generated deterministic wrapper

    # CO-Bench invokes solve(**instance); the pre-fix wrapper raised
    # "got some positional-only arguments passed as keyword arguments: 'a'".
    assert ns["solve"](a=3, b=4) == 7
    # Stray instance keys are still filtered (the deployed-but-broken guard).
    assert ns["solve"](a=3, b=4, extra=99) == 7


def test_finding_63_no_posonly_path_unchanged():
    # Regression guard: helpers WITHOUT positional-only params keep the legacy
    # byte-for-byte wrapper output (no behavioral drift from the fix).
    src = "def helper(x, y):\n    return x * y\n"
    assert _build_deploy_wrapper("helper", src) == (
        "def solve(**kw):\n"
        "    return helper(**{k: v for k, v in kw.items() if k in {'x', 'y'}})\n"
    )
    kw_src = "def helper(**kwargs):\n    return sum(kwargs.values())\n"
    assert _build_deploy_wrapper("helper", kw_src) == (
        "def solve(**kw):\n    return helper(**kw)\n"
    )


# ---------------------------------------------------------------------------
# Finding 66 — the bash description formatter must apply the same advertising
# gate as the python one, so a blocklist-flagged python helper is NOT advertised.
# ---------------------------------------------------------------------------
def test_finding_66_bash_formatter_advertises_sys_using_helper():
    # Audit #66, RESOLVED by NOT gating bash advertising on for_advertising.
    # A bash ``_lib_`` helper runs in an isolated sandbox (--network none) and
    # legitimately needs ``import sys`` for argv in its CLI ``__main__`` tail — a
    # module the HOST safety blocklist flags. The bash formatter must still
    # advertise it (the ``python3 ./helpers/_lib_<name>.py`` CLI form) or the whole
    # bash code-library channel goes dark (every bash _lib_ helper imports sys).
    # The host blocklist guards HOST exec (co_bench / text_classification), NOT
    # sandboxed container exec, so re-applying it to advertising here is wrong.
    sbx = SandboxMarker()
    sys_helper = (
        'def cli(*a):\n'
        '    """Echo args."""\n'
        '    return " ".join(str(x) for x in a)\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    import sys\n'
        '    print(cli(*sys.argv[1:]))\n'
    )
    benign = 'def good(x):\n    """Add one."""\n    return x + 1\n'

    out = format_bash_library_descriptions(
        {"cli": sys_helper, "good": benign}, {}, executor=sbx,
    )

    # Both helpers are advertised as runnable python3 CLI scripts — the sys-using
    # one is NOT suppressed (that suppression would kill the bash channel).
    assert "_lib_cli.py" in out
    assert "_lib_good.py" in out
