"""R3-C_verify_gate-1: the crew verify gate rejects any helper the name-based
deploy wrapper (``adoption._build_deploy_wrapper``) could not bind.

Host-side, LLM-free, no Docker: these exercise the pure ``deploy_name_forwardable``
predicate + the OFF-path directly. The predicate encodes the deploy binding
contract, so a verify PASS implies deployed score == verified score.
"""
from meta_n.integrations.co_bench import (
    _CREW_DEPLOY_KEYS,
    COBenchAdapter,
    deploy_name_forwardable,
)

NAME = "solve_crew_scheduling"


def _fwd(src: str, name: str = NAME) -> bool:
    return deploy_name_forwardable(src, name, _CREW_DEPLOY_KEYS)


def test_deploy_keys_are_the_runner_positional_keys():
    # SINGLE SOURCE OF TRUTH: the deploy keys are exactly the crew positional keys.
    assert _CREW_DEPLOY_KEYS == frozenset({"N", "K", "time_limit", "tasks", "arcs"})


def test_canonical_signature_forwardable():
    assert _fwd("def solve_crew_scheduling(N, K, time_limit, tasks, arcs):\n    return {}\n")


def test_kwargs_plus_extra_default_forwardable():
    # Edge (a): **kwargs + canonical names + extra defaulted param -> kept.
    src = (
        "def solve_crew_scheduling(N, K, time_limit, tasks, arcs, extra=5, **kw):\n"
        "    return {}\n"
    )
    assert _fwd(src)


def test_required_unknown_positional_not_forwardable():
    # Edge (b): a REQUIRED param name absent from keys -> dropped (fail-closed;
    # deploy could not bind it, and verify's 5-positional call also raises).
    src = "def solve_crew_scheduling(N, K, time_limit, tasks, arcs, extra):\n    return {}\n"
    assert not _fwd(src)


def test_defaulted_unknown_positional_forwardable():
    # A param whose name is absent but HAS a default is safe (it defaults
    # identically under positional-verify and name-deploy).
    src = "def solve_crew_scheduling(N, K, time_limit, tasks, arcs, extra=7):\n    return {}\n"
    assert _fwd(src)


def test_all_renamed_not_forwardable():
    assert not deploy_name_forwardable(
        "def solve_crew(n, k, tl, t, a):\n    return {}\n", "solve_crew", _CREW_DEPLOY_KEYS
    )


def test_posonly_canonical_forwardable():
    # Edge (d): posonly canonical names bind by name in the deploy posonly branch.
    src = "def solve_crew_scheduling(N, K, time_limit, tasks, arcs, /):\n    return {}\n"
    assert _fwd(src)


def test_posonly_unknown_name_not_forwardable():
    # A posonly name absent from keys breaks the positional present-prefix.
    src = "def solve_crew_scheduling(N, K, time_limit, tasks, extra, /):\n    return {}\n"
    assert not _fwd(src)


def test_kwonly_required_unknown_not_forwardable():
    src = "def solve_crew_scheduling(N, K, time_limit, tasks, arcs, *, extra):\n    return {}\n"
    assert not _fwd(src)


def test_kwonly_defaulted_unknown_forwardable():
    src = "def solve_crew_scheduling(N, K, time_limit, tasks, arcs, *, extra=1):\n    return {}\n"
    assert _fwd(src)


def test_unparseable_source_fail_closed():
    assert not _fwd("def solve_crew_scheduling(:\n")


def test_missing_def_fail_closed():
    assert not _fwd("def other(N, K):\n    return {}\n")


def test_off_path_non_crew_returns_none_unchanged():
    # OFF-path byte-identity: a non-crew selection still returns None (the guarded
    # verifier is never even constructed).
    adapter = COBenchAdapter(data_dir="/tmp/co_bench", task_names=["Assignment problem"])
    assert adapter.make_heldout_verifier() is None
