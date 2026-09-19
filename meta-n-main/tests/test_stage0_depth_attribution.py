"""S0.4 / S0.5 offline-analysis gate — depth attribution + per-task delta.

Read-only helpers over the on-disk archive (`meta_n.analysis.depth_attribution`).
The corpus-gated tests reproduce the audit hand-computed depth split; the
synthetic tests run on a fresh checkout where `experiments/` is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meta_n.analysis import depth_attribution as DA
from meta_n.analysis.depth_attribution import (
    ancestry_chain,
    archive_depth_distribution,
    classify_injection_layer,
    deepest_leaf_id,
    load_archive,
    main as depth_probe_main,
    per_task_delta,
    per_task_delta_from_archive,
    run_depth_probe,
    split_candidate_mean,
    trace_depth_distribution,
    within_task_depth_profile,
    within_task_verdict,
)
from meta_n.core.archive import Archive, Candidate
from meta_n.core.meta_layer import InjectedCode, Trace

REPO = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = REPO / "experiments" / "metan_e2_g9_v3_s42" / "treatment" / "run" / "archive"
AUDIT_SPLIT = {1: 55, 2: 14, 3: 13, 4: 13, 5: 7}

corpus = pytest.mark.skipif(
    not ARCHIVE_DIR.exists(), reason="on-disk archive corpus absent"
)


# --------------------------------------------------------------------------- #
# S0.4 — synthetic (always runs)
# --------------------------------------------------------------------------- #
def test_split_fresh_vs_inherited_synthetic():
    cand = Candidate(
        candidate_id="c", parent_id="p", depth=3,
        per_task_scores={"a": 1.0, "b": 0.0, "c": 0.5},
        traces=[
            Trace(task_id="a", depth=3, score=1.0, success=True),   # fresh
            Trace(task_id="b", depth=1, score=0.0, success=False),  # inherited
            Trace(task_id="c", depth=2, score=0.5, success=True),   # inherited
        ],
    )
    s = split_candidate_mean(cand)
    assert s.fresh_tasks == ["a"]
    assert sorted(s.inherited_tasks) == ["b", "c"]
    assert s.n_fresh == 1 and s.n_inherited == 2
    assert s.fresh_mean == 1.0
    assert s.inherited_mean == 0.25
    assert s.overall_mean == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert s.fresh_fraction == pytest.approx(1 / 3)


def test_split_ignores_tasks_without_a_trace():
    cand = Candidate(
        candidate_id="c", depth=2,
        per_task_scores={"a": 1.0, "ghost": 0.9},
        traces=[Trace(task_id="a", depth=2, score=1.0, success=True)],
    )
    s = split_candidate_mean(cand)
    assert s.fresh_tasks == ["a"]
    assert s.n_inherited == 0
    assert s.overall_mean == 1.0  # 'ghost' has no trace -> not attributed


def test_trace_depth_distribution_synthetic():
    traces = [Trace(task_id=str(i), depth=d) for d, n in {1: 3, 2: 1}.items() for i in range(n)]
    assert trace_depth_distribution(traces) == {1: 3, 2: 1}


# --------------------------------------------------------------------------- #
# S0.5 — synthetic (always runs)
# --------------------------------------------------------------------------- #
def test_per_task_delta_synthetic():
    parent = Candidate(candidate_id="p", per_task_scores={"a": 0.2, "b": 0.5, "x": 0.1})
    child = Candidate(candidate_id="c", parent_id="p", per_task_scores={"a": 0.6, "b": 0.5, "y": 0.9})
    d = per_task_delta(child, parent)
    assert d == {"a": pytest.approx(0.4), "b": pytest.approx(0.0)}  # only shared tasks


def test_per_task_delta_from_archive_seed_is_none():
    arc = Archive()
    arc.add(Candidate(candidate_id="seed", parent_id=None, per_task_scores={"a": 0.5}))
    assert per_task_delta_from_archive(arc, "seed") is None


def test_per_task_delta_from_archive_join():
    arc = Archive()
    arc.add(Candidate(candidate_id="p", parent_id=None, per_task_scores={"a": 0.2}))
    arc.add(Candidate(candidate_id="c", parent_id="p", per_task_scores={"a": 0.7}))
    assert per_task_delta_from_archive(arc, "c") == {"a": pytest.approx(0.5)}


# --------------------------------------------------------------------------- #
# S0.4 / S0.5 — on-disk corpus (reproduces the audit)
# --------------------------------------------------------------------------- #
@corpus
def test_reproduce_audit_depth_split():
    arc = load_archive(ARCHIVE_DIR)
    assert archive_depth_distribution(arc) == AUDIT_SPLIT
    assert sum(AUDIT_SPLIT.values()) == 102


@corpus
def test_split_consistency_over_corpus():
    arc = load_archive(ARCHIVE_DIR)
    for c in arc.candidates:
        s = split_candidate_mean(c)
        # Highest-depth trace wins per task (matches split_candidate_mean's collapse).
        depth_by_task: dict[str, int] = {}
        for t in c.traces:
            if t.task_id not in depth_by_task or t.depth > depth_by_task[t.task_id]:
                depth_by_task[t.task_id] = t.depth
        # Fresh tasks have a trace at EXACTLY this candidate's own depth; inherited
        # tasks (shallower carried-up OR deeper per-task-best frozen) do not.
        for tid in s.fresh_tasks:
            assert depth_by_task[tid] == c.depth
        for tid in s.inherited_tasks:
            assert depth_by_task[tid] != c.depth
        assert s.n_fresh + s.n_inherited <= len(c.per_task_scores)


@corpus
def test_per_task_delta_join_over_corpus():
    arc = load_archive(ARCHIVE_DIR)
    non_seed = [c for c in arc.candidates if c.parent_id is not None]
    assert non_seed, "expected at least one non-seed candidate"
    for c in non_seed:
        d = per_task_delta_from_archive(arc, c.candidate_id)
        assert d is not None
        parent = arc.get(c.parent_id)
        for tid, delta in d.items():
            assert tid in c.per_task_scores and tid in parent.per_task_scores
            assert delta == pytest.approx(
                c.per_task_scores[tid] - parent.per_task_scores[tid]
            )


# --------------------------------------------------------------------------- #
# DEPTH-PROBE — within-task recursion vs router-stacking (always runs)
# --------------------------------------------------------------------------- #
def _ic(d, *, pre=None, lib=None, tsm=None) -> InjectedCode:
    return InjectedCode(
        pre_process=pre, code_library=lib or {}, task_solution_map=tsm or {},
        source_depth=d,
    )


def _genuine_chain() -> Archive:
    """A 2-task cell re-solved FRESH at every depth, helper re-authored each layer
    (non-consolidate-shaped): every task acted on by every stacked layer."""
    arc = Archive()
    arc.add(Candidate(
        candidate_id="c1", depth=1, per_task_scores={"donor": 1.0, "recip": 0.3},
        traces=[Trace(task_id="donor", depth=1, score=1.0),
                Trace(task_id="recip", depth=1, score=0.3)],
    ))
    arc.add(Candidate(
        candidate_id="c2", parent_id="c1", depth=2,
        injected_codes=[_ic(2, lib={"explore_maze": "v1"})],
        per_task_scores={"donor": 1.0, "recip": 0.5},
        traces=[Trace(task_id="donor", depth=2, score=1.0),
                Trace(task_id="recip", depth=2, score=0.5)],
    ))
    arc.add(Candidate(
        candidate_id="c3", parent_id="c2", depth=3,
        injected_codes=[_ic(2, lib={"explore_maze": "v1"}),
                        _ic(3, lib={"explore_maze": "v2"})],
        per_task_scores={"donor": 1.0, "recip": 0.8},
        traces=[Trace(task_id="donor", depth=3, score=1.0),
                Trace(task_id="recip", depth=3, score=0.8)],
    ))
    return arc


def _router_chain() -> Archive:
    """A deep chain that ROUTES each task to its own frozen winner (the CO-Bench
    artifact): task_solution_map + a task-identity-gated pre_process dominate."""
    arc = Archive()
    arc.add(Candidate(
        candidate_id="r1", depth=1,
        per_task_scores={"t1": 1.0, "t2": 0.0, "t3": 0.0, "t4": 0.0},
        traces=[Trace(task_id=t, depth=1, score=(1.0 if t == "t1" else 0.0))
                for t in ("t1", "t2", "t3", "t4")],
    ))
    arc.add(Candidate(
        candidate_id="r2", parent_id="r1", depth=2,
        injected_codes=[_ic(2, tsm={"t1": "frozen"})],
        per_task_scores={"t1": 1.0, "t2": 1.0, "t3": 0.0, "t4": 0.0},
        traces=[Trace(task_id="t1", depth=1, score=1.0),
                Trace(task_id="t2", depth=2, score=1.0),
                Trace(task_id="t3", depth=1, score=0.0),
                Trace(task_id="t4", depth=1, score=0.0)],
    ))
    arc.add(Candidate(
        candidate_id="r3", parent_id="r2", depth=3,
        injected_codes=[
            _ic(2, tsm={"t1": "frozen"}),
            _ic(3, pre='if "alpha" in task.task_id:\n    additional_context = "x"\n'
                       'else:\n    additional_context = ""'),
        ],
        per_task_scores={"t1": 1.0, "t2": 1.0, "t3": 1.0, "t4": 0.0},
        traces=[Trace(task_id="t1", depth=1, score=1.0),
                Trace(task_id="t2", depth=2, score=1.0),
                Trace(task_id="t3", depth=3, score=1.0),
                Trace(task_id="t4", depth=1, score=0.0)],
    ))
    return arc


def test_ancestry_chain_orders_seed_to_leaf():
    arc = _genuine_chain()
    assert [c.candidate_id for c in ancestry_chain(arc, "c3")] == ["c1", "c2", "c3"]
    assert deepest_leaf_id(arc) == "c3"


def test_classify_injection_layer_router_vs_within_task():
    assert classify_injection_layer(_ic(2, tsm={"t": "x"})) == "router_frozen"
    assert classify_injection_layer(_ic(2, pre="x = task.description")) == "router_taskgate"
    assert classify_injection_layer(_ic(2, pre="x = task.task_id")) == "router_taskgate"
    # Conditioning on task STRUCTURE (metadata) is within-task, NOT a router.
    assert classify_injection_layer(_ic(2, pre='n = task.metadata["n"]')) == "generic_preprocess"
    assert classify_injection_layer(_ic(2, lib={"h": "def h(): ..."})) == "code_helper"
    assert classify_injection_layer(_ic(2)) == "empty"


def test_within_task_profile_genuine_depth_passes():
    arc = _genuine_chain()
    p = within_task_depth_profile(arc, "c3")
    assert p.max_chain_depth == 3
    assert p.max_multiplicity == 3              # both tasks re-solved at depths 1,2,3
    assert sorted(p.genuine_depth_tasks) == ["donor", "recip"]
    assert p.router_fraction == 0.0
    assert p.helper_reuse_depth["explore_maze"] == 2  # authored at source_depths 2 and 3
    assert p.mean_adjacent_overlap == pytest.approx(1.0)
    v = within_task_verdict(p)
    assert v["verdict"] == "PASS"


def test_within_task_profile_router_stacking_fails():
    arc = _router_chain()
    p = within_task_depth_profile(arc, deepest_leaf_id(arc))
    assert p.max_chain_depth == 3
    assert p.router_fraction == pytest.approx(1.0)  # both layers are routers
    assert p.layer_classes == ["router_frozen", "router_taskgate"]
    v = within_task_verdict(p)
    assert v["verdict"] == "FAIL"
    assert "ROUTER-STACKING" in v["reason"]


def test_within_task_verdict_shallow_is_inconclusive():
    arc = Archive()
    arc.add(Candidate(candidate_id="s", depth=1, per_task_scores={"a": 1.0},
                      traces=[Trace(task_id="a", depth=1, score=1.0)]))
    arc.add(Candidate(candidate_id="d2", parent_id="s", depth=2,
                      injected_codes=[_ic(2, lib={"h": "def h(): ..."})],
                      per_task_scores={"a": 1.0},
                      traces=[Trace(task_id="a", depth=2, score=1.0)]))
    p = within_task_depth_profile(arc, "d2")
    v = within_task_verdict(p)  # depth 2 < min_chain_depth=3
    assert v["verdict"] == "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# DEPTH-PROBE CLI — run_depth_probe + main (synthetic, monkeypatched loader)
# --------------------------------------------------------------------------- #
def test_run_depth_probe_genuine_chain(monkeypatch):
    """run_depth_probe auto-targets the deepest leaf and returns a PASS payload."""
    monkeypatch.setattr(DA, "load_archive", lambda _d: _genuine_chain())
    v = run_depth_probe("ignored/path")
    assert v["verdict"] == "PASS"
    assert v["leaf_id"] == "c3"           # deepest_leaf_id picked it
    assert v["n_tasks"] == 2
    assert v["archive_dir"] == "ignored/path"
    assert v["multiplicity"] == {"donor": 3, "recip": 3}


def test_run_depth_probe_unknown_leaf_raises(monkeypatch):
    monkeypatch.setattr(DA, "load_archive", lambda _d: _genuine_chain())
    with pytest.raises(ValueError, match="not found"):
        run_depth_probe("ignored", leaf_id="nope")


def test_run_depth_probe_empty_archive_raises(monkeypatch):
    monkeypatch.setattr(DA, "load_archive", lambda _d: Archive())
    with pytest.raises(ValueError, match="empty"):
        run_depth_probe("ignored")


def test_cli_main_exit_code_pass(monkeypatch, capsys):
    monkeypatch.setattr(DA, "load_archive", lambda _d: _genuine_chain())
    rc = depth_probe_main(["somedir"])
    assert rc == 0                         # PASS -> exit 0
    out = capsys.readouterr().out
    assert "DEPTH-PROBE" in out and "VERDICT: PASS" in out


def test_cli_main_exit_code_fail_router(monkeypatch, capsys):
    monkeypatch.setattr(DA, "load_archive", lambda _d: _router_chain())
    rc = depth_probe_main(["somedir"])
    assert rc == 1                         # FAIL -> exit 1
    assert "ROUTER-STACKING" in capsys.readouterr().out


def test_cli_main_json_payload(monkeypatch, capsys):
    monkeypatch.setattr(DA, "load_archive", lambda _d: _genuine_chain())
    rc = depth_probe_main(["somedir", "--json"])
    assert rc == 0
    import json as _json
    payload = _json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "PASS"
    assert payload["router_fraction"] == 0.0


def test_cli_main_missing_dir_is_error_exit(monkeypatch, capsys):
    # Real loader on a missing dir rebuilds an EMPTY archive -> ValueError -> rc 3.
    rc = depth_probe_main(["/no/such/archive/dir/xyz"])
    assert rc == 3
    assert "error" in capsys.readouterr().err


@corpus
def test_cli_main_on_real_cobench_corpus_is_fail():
    """The real CO-Bench depth-5 archive is router-stacking -> CLI exits 1 (FAIL)."""
    rc = depth_probe_main([str(ARCHIVE_DIR)])
    assert rc == 1


# --------------------------------------------------------------------------- #
# T3.8 — helper SUPPLY DUPLICATION vs ACTUAL adoption (utilities_called)
# --------------------------------------------------------------------------- #
def _adoption_chain() -> Archive:
    """Genuine within-task chain whose traces RECORD helper calls (T3.8 adoption).

    ``explore_maze`` is SHIPPED at source_depths 2 and 3 (supply duplication) AND
    actually CALLED at trace depths 2 and 3 (utilities_called) — the two signals
    coincide here so the test can tell them apart from the unmeasured case below.
    """
    arc = Archive()
    arc.add(Candidate(
        candidate_id="a1", depth=1, per_task_scores={"m": 0.3},
        traces=[Trace(task_id="m", depth=1, score=0.3)],
    ))
    arc.add(Candidate(
        candidate_id="a2", parent_id="a1", depth=2,
        injected_codes=[_ic(2, lib={"explore_maze": "v1"})],
        per_task_scores={"m": 0.6},
        traces=[Trace(task_id="m", depth=2, score=0.6,
                      utilities_called=["explore_maze"],
                      utilities_call_counts={"explore_maze": 2})],
    ))
    arc.add(Candidate(
        candidate_id="a3", parent_id="a2", depth=3,
        injected_codes=[_ic(2, lib={"explore_maze": "v1"}),
                        _ic(3, lib={"explore_maze": "v2"})],
        per_task_scores={"m": 0.9},
        traces=[Trace(task_id="m", depth=3, score=0.9,
                      utilities_called=["explore_maze"],
                      utilities_call_counts={"explore_maze": 1})],
    ))
    return arc


def test_helper_adoption_from_utilities_called():
    p = within_task_depth_profile(_adoption_chain(), "a3")
    # SUPPLY DUPLICATION: name shipped at source_depths 2 and 3 (static).
    assert p.helper_reuse_depth["explore_maze"] == 2
    assert p.max_helper_reuse == 2
    # ACTUAL adoption (T3.8): CALLED at trace depths 2 and 3, read from utilities_called.
    assert p.helper_adoption_depth == {"explore_maze": 2}
    assert p.max_helper_adoption == 2


def test_helper_adoption_unmeasurable_is_none():
    # _genuine_chain traces never set utilities_called -> adoption unmeasurable.
    p = within_task_depth_profile(_genuine_chain(), "c3")
    assert p.helper_adoption_depth is None
    assert p.max_helper_adoption is None
    # supply duplication is still measured (it is a static property of the code).
    assert p.max_helper_reuse == 2


def test_pass_reason_relabels_supply_vs_adoption():
    p = within_task_depth_profile(_adoption_chain(), "a3")
    v = within_task_verdict(p)
    assert v["verdict"] == "PASS"
    # The misleading "live code-channel reuse" framing is dropped.
    assert "supply duplication" in v["reason"]
    assert "code channel is live" not in v["reason"]
    assert "max helper reuse" not in v["reason"]
    # Both signals are surfaced on the verdict dict.
    assert v["max_helper_reuse"] == 2
    assert v["max_helper_adoption"] == 2


def test_pass_reason_adoption_na_when_unmeasured():
    p = within_task_depth_profile(_genuine_chain(), "c3")
    v = within_task_verdict(p)
    assert v["verdict"] == "PASS"
    assert "adoption n/a (unmeasured)" in v["reason"]
    assert v["max_helper_adoption"] is None
