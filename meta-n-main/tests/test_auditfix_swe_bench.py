"""Offline regression tests for audit finding #22.

Finding #22: SWEBenchVerifiedAdapter subclasses TerminalBenchAdapter, which
overrides ``advertises_spine_builtin()`` -> True. SWE-bench did NOT override it
back, so ``--base-solver builtin --benchmark swe_bench_verified`` misrouted onto
the external-agent spine + BuiltinTBBackend (wired to TerminalBench tasks, not
SWE-bench), silently scoring the wrong harness against the wrong tasks. The fix
overrides ``advertises_spine_builtin()`` -> False so builtin stays on the native
TerminalBenchExecutor path.

These tests are fully offline: they construct the adapter (its __init__ only
creates a local tempdir, no network/Docker) and inspect the predicate.
"""

from meta_n.integrations.swe_bench import SWEBenchVerifiedAdapter
from meta_n.integrations.terminal_bench import TerminalBenchAdapter


def test_swe_bench_does_not_advertise_spine_builtin():
    """builtin must stay on the native executor for SWE-bench (FAILS pre-fix)."""
    adapter = SWEBenchVerifiedAdapter()
    try:
        assert adapter.advertises_spine_builtin() is False
    finally:
        adapter.cleanup()


def test_swe_bench_overrides_parent_true():
    """SWE-bench must override the parent's spine-builtin=True, not inherit it."""
    # The TerminalBench parent advertises True; the SWE-bench subclass must not.
    parent = TerminalBenchAdapter()
    try:
        assert parent.advertises_spine_builtin() is True
    finally:
        parent.cleanup()

    # The override must be defined on the subclass itself (not merely inherited).
    assert (
        "advertises_spine_builtin" in SWEBenchVerifiedAdapter.__dict__
    ), "SWEBenchVerifiedAdapter must define its own advertises_spine_builtin override"
