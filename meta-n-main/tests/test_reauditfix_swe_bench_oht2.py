"""Regression test for re-audit finding #3.

SWE-bench Verified's prior fix closed only the BUILTIN spine route
(``advertises_spine_builtin() -> False``). But ``--base-solver
openhands/terminus2`` ALWAYS routes through the external spine regardless of
adapter, and ``SWEBenchVerifiedAdapter`` inherited
``make_agent_backend``/``make_env_provider``/``make_scorer`` from
``TerminalBenchAdapter`` unchanged — those build TerminalBench-runner-wired
backends pointed at the SWE-bench task dir (harbor ``task.toml`` +
``instruction.md``, no ``task.yaml``). The result: an unsupported combo would
silently score the WRONG harness/tasks instead of failing loudly.

These tests assert the SWE-bench adapter now rejects OH/T2 backend
construction (all three factories) while keeping the ``builtin`` native path
working. They are fully offline: constructing the adapter only makes a
tempdir; no download / Docker / LLM.
"""

import pytest

from meta_n.integrations.swe_bench import SWEBenchVerifiedAdapter


@pytest.fixture
def adapter():
    a = SWEBenchVerifiedAdapter(task_cache_dir=None, task_names=None)
    try:
        yield a
    finally:
        a.cleanup()


@pytest.mark.parametrize("kind", ["openhands", "terminus2"])
def test_make_agent_backend_rejects_external_kinds(adapter, kind):
    """OH/T2 must fail LOUDLY at agent-backend construction (pre-fix: builds
    a TerminalBench-runner backend against SWE-bench tasks)."""
    with pytest.raises(ValueError) as ei:
        adapter.make_agent_backend(kind)
    msg = str(ei.value)
    assert kind in msg
    assert "swe_bench_verified" in msg


@pytest.mark.parametrize("kind", ["openhands", "terminus2"])
def test_make_env_provider_rejects_external_kinds(adapter, kind):
    """Sibling factory: env provider must reject OH/T2 too."""
    with pytest.raises(ValueError):
        adapter.make_env_provider(kind)


@pytest.mark.parametrize("kind", ["openhands", "terminus2"])
def test_make_scorer_rejects_external_kinds(adapter, kind):
    """Sibling factory: scorer must reject OH/T2 too."""
    with pytest.raises(ValueError):
        adapter.make_scorer(kind)


def test_builtin_native_path_preserved(adapter):
    """The BUILTIN route must stay on the native SWE-bench executor.

    ``advertises_spine_builtin`` stays False (builtin never enters the spine),
    and the make_* factories must NOT reject ``builtin`` with the SWE-bench
    unsupported-combo error. env/scorer factories build offline for builtin.
    """
    assert adapter.advertises_spine_builtin() is False
    # _reject_external_kind is a no-op for builtin (does not raise).
    assert adapter._reject_external_kind("builtin") is None
    # env + scorer factories delegate to the parent and construct fine.
    assert adapter.make_env_provider("builtin") is not None
    assert adapter.make_scorer("builtin") is not None
    # make_agent_backend('builtin') delegates to the parent, which then raises
    # the PARENT's error about the missing authoring 'solver' — crucially NOT
    # the SWE-bench unsupported-combo error. This proves builtin is not
    # rejected by our override and still reaches the native builtin path.
    with pytest.raises(ValueError) as ei:
        adapter.make_agent_backend("builtin")
    assert "not supported for benchmark" not in str(ei.value)
