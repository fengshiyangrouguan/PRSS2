"""SRI formal protocol: the 18 required offline tests (design §14).

Runs with pytest if available, and as a plain script otherwise:

    python -m pytest tests/test_sri_formal_protocol.py -q
    python tests/test_sri_formal_protocol.py

Every test is OFFLINE: no network, no torch, no benchmark. Each one is a
closed-loop check on the protocol layer, not a smoke test. Item numbers follow
the design document so a reviewer can map failures back to requirements.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meta_n.sri import ledger as L                      # noqa: E402
from meta_n.sri import metrics as M                     # noqa: E402
from meta_n.sri import pairing as P                     # noqa: E402
from meta_n.sri import protocol as PR                   # noqa: E402
from meta_n.sri import transition_audit as TA           # noqa: E402
from meta_n.sri.hooks import SRIHooks                   # noqa: E402

COHORT = list(PR.PRIMARY6)


def _profile(**over):
    """The FROZEN profile as the YAML declares it.

    Built by loading `configs/sri_primary6.yaml` rather than by hand: a test that
    hand-builds a profile silently stops covering the shipped one the moment a
    required field is added, which is exactly what happened to tests 13-15 when
    the profile gained the pinned-parameter fields.
    """
    import dataclasses
    prof = PR.load_profile(PR.default_profile_path("sri_primary6"))
    return dataclasses.replace(prof, **over) if over else prof


def _ledger(tmp: Path, arm="official", seed=0, gamma=None):
    return L.SlotLedger(tmp / "proposal_slots.jsonl", run_id="r", arm=arm,
                        backbone="gpt-5.5", cohort_id="sri_primary6:6",
                        search_seed=seed, root_bundle_sha256="R" * 64,
                        gamma_checkpoint_sha256=gamma)


def _mat(cid, depth, **scripts):
    return TA.CandidateMaterial(candidate_id=cid, structural_depth=depth,
                                task_scripts=scripts)


class _Ev:
    def __init__(self, table, *, inherited=()):
        self.table, self.inherited = table, set(inherited)

    def evaluate(self, task_id, source, *, seed):
        if (source, task_id) in self.inherited:
            return TA.RawEval(0.5, True, False, "reused")
        return TA.RawEval(float(self.table.get((source, task_id), 0.0)), True, True)


# -- 1 ---------------------------------------------------------------------
def test_01_gate_rejected_child_in_ledger_and_edges(tmp_path=None):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        led = _ledger(td)
        hk = SRIHooks(ledger=led, run_dir=td, gate_configured=3,
                      gate_reason="consolidation_focus", cohort=["T1"])
        h = hk.open(iteration=0, parent_slot=0, child_slot=0, parent_id="p2",
                    parent_structural_depth=2, gate_effective=False,
                    proposed_child_id="c3", proposed_child_depth=3)
        mp = hk.capture_material(slot_id=h.slot_id, candidate_id="c3",
                                 structural_depth=3,
                                 task_scripts={"T1": "def solve(): pass"})
        hk.close(h, "gate_rejected", archive_admitted=False, material_path=mp)

        row = led.rows()[0]
        assert row["terminal_status"] == "gate_rejected"
        assert row["archive_admitted"] is False

        parent = _mat("p2", 2, T1="p")
        child = _mat("c3", 3, T1="c")
        edges, drops = TA.build_edges(led.rows(), lambda r, side: parent
                                      if side == "parent" else child)
        assert len(edges) == 1, (edges, drops)
        assert edges[0].child_terminal_status == "gate_rejected"
        assert edges[0].child_was_admitted is False


# -- 2 ---------------------------------------------------------------------
def test_02_empty_injection_consumes_exactly_one_slot(tmp_path=None):
    with tempfile.TemporaryDirectory() as td:
        led = _ledger(Path(td))
        hk = SRIHooks(ledger=led, run_dir=Path(td), cohort=["T1"])
        h = hk.open(iteration=0, parent_slot=0, child_slot=0, parent_id="p",
                    parent_structural_depth=1, gate_effective=True)
        hk.close(h, "empty_injection", failure_class="empty_omega")
        assert len(led.rows()) == 1
        assert led.rows()[0]["terminal_status"] == "empty_injection"


# -- 3 ---------------------------------------------------------------------
def test_03_failures_are_not_backfilled(tmp_path=None):
    with tempfile.TemporaryDirectory() as td:
        led = _ledger(Path(td))
        hk = SRIHooks(ledger=led, run_dir=Path(td), cohort=["T1"])
        for c in range(3):
            h = hk.open(iteration=0, parent_slot=0, child_slot=c, parent_id="p",
                        parent_structural_depth=1, gate_effective=True)
            hk.close(h, "generation_error", failure_class="boom")
        assert len(led.rows()) == 3
        assert len(led.unfinalized()) == 0
        assert led.counts_by_status() == {"generation_error": 3}


# -- 4 ---------------------------------------------------------------------
def test_04_accepted_and_rejected_children_both_enter_the_audit(tmp_path=None):
    parent = _mat("p2", 2, T1="p")
    good, bad = _mat("g3", 3, T1="g"), _mat("b3", 3, T1="b")
    rows = [
        {"slot_id": "s1", "iteration": 0, "parent_structural_depth": 2,
         "proposed_child_depth": 3, "terminal_status": "evaluated_admitted",
         "archive_admitted": True},
        {"slot_id": "s2", "iteration": 0, "parent_structural_depth": 2,
         "proposed_child_depth": 3, "terminal_status": "evaluated_rejected",
         "archive_admitted": False},
    ]
    mats = {"s1": good, "s2": bad}
    edges, drops = TA.build_edges(
        rows, lambda r, side: parent if side == "parent" else mats[r["slot_id"]])
    assert len(edges) == 2 and not drops
    res = TA.run_canonical_audit(edges, cohort=["T1"],
                                 evaluator=_Ev({("p", "T1"): 0.5, ("g", "T1"): 0.6,
                                                ("b", "T1"): 0.4}),
                                 search_seed=0, repeats=1)
    ids = {r["candidate_id"] for r in res.raw_rows}
    assert ids == {"p2", "g3", "b3"}, ids


# -- 5 ---------------------------------------------------------------------
def test_05_inherited_scores_are_rejected_as_audit_input(tmp_path=None):
    edges = [TA.Edge(slot_id="s", iteration=0, parent=_mat("p", 2, T1="p"),
                     child=_mat("c", 3, T1="c"), child_was_admitted=True,
                     child_terminal_status="evaluated_admitted")]
    try:
        TA.run_canonical_audit(edges, cohort=["T1"],
                               evaluator=_Ev({("c", "T1"): 0.9},
                                             inherited={("c", "T1")}),
                               search_seed=0, repeats=1)
        raise AssertionError("an inherited score was aggregated")
    except TA.InheritedScore:
        pass


# -- 6 ---------------------------------------------------------------------
def test_06_canonical_audit_freshly_evaluates_all_cohort_tasks(tmp_path=None):
    edges = [TA.Edge(slot_id="s", iteration=0,
                     parent=_mat("p", 2, **{t: "p" + t for t in COHORT}),
                     child=_mat("c", 3, **{t: "c" + t for t in COHORT}),
                     child_was_admitted=True,
                     child_terminal_status="evaluated_admitted")]
    table = {}
    for t in COHORT:
        table[("p" + t, t)] = 0.5
        table[("c" + t, t)] = 0.6
    res = TA.run_canonical_audit(edges, cohort=COHORT, evaluator=_Ev(table),
                                 search_seed=0, repeats=1)
    seen = {r["task_id"] for r in res.raw_rows}
    assert seen == set(COHORT), seen
    assert all(r["freshly_executed"] is True for r in res.raw_rows)


# -- 7 ---------------------------------------------------------------------
def test_07_matched_task_repeat_seeds(tmp_path=None):
    assert TA.audit_seed(1, "T1", 0) == TA.audit_seed(1, "T1", 0)
    assert TA.audit_seed(1, "T1", 0) != TA.audit_seed(1, "T1", 1)
    assert TA.audit_seed(1, "T1", 0) != TA.audit_seed(1, "T2", 0)
    # independent of candidate and depth, because it takes neither
    import inspect
    params = list(inspect.signature(TA.audit_seed).parameters)
    assert params == ["search_seed", "task_id", "repeat_index"], params


# -- 8 ---------------------------------------------------------------------
def test_08_repeats_do_not_inflate_the_denominator(tmp_path=None):
    edges = [TA.Edge(slot_id="s", iteration=0, parent=_mat("p", 2, T1="p"),
                     child=_mat("c", 3, T1="c"), child_was_admitted=True,
                     child_terminal_status="evaluated_admitted")]
    r1 = TA.run_canonical_audit(edges, cohort=["T1"],
                                evaluator=_Ev({("p", "T1"): 0.5, ("c", "T1"): 0.6}),
                                search_seed=0, repeats=1)
    r5 = TA.run_canonical_audit(edges, cohort=["T1"],
                                evaluator=_Ev({("p", "T1"): 0.5, ("c", "T1"): 0.6}),
                                search_seed=0, repeats=5)
    assert r1.metrics["transitions"]["R_2to3"]["valid_edge_task_pairs"] == 1
    assert r5.metrics["transitions"]["R_2to3"]["valid_edge_task_pairs"] == 1
    assert r5.metrics["raw_observations"] > r1.metrics["raw_observations"]


# -- 9 ---------------------------------------------------------------------
def test_09_archive_best_by_depth_is_cumulative_and_not_iteration(tmp_path=None):
    sel = M.select_deployable([{"candidate_id": "c3", "depth": 3, "creation_index": 2}],
                              lambda c: 0.8)
    pr = M.per_run_metrics(
        selected=sel, test_result={"test_mean_score": 0.7},
        audit_metrics={"transitions": {}, "failure_accounting": {}},
        ledger_counts={}, resource_use={},
        archive_candidates=[{"candidate_id": "c1", "depth": 1, "creation_index": 0},
                            {"candidate_id": "c2", "depth": 2, "creation_index": 1},
                            {"candidate_id": "c3", "depth": 3, "creation_index": 2}],
        dev_score_of=lambda c: {"c1": 0.7, "c2": 0.3,
                                "c3": 0.8}[c["candidate_id"]],
        iteration_archive_best=[{"iteration": 0, "best_dev": 0.7},
                                {"iteration": 9, "best_dev": 0.8}])
    assert pr["exact_depth_dev_best"]["3"] == 0.8
    assert pr["archive_best_dev_by_recursive_depth"] == {
        "1": 0.7, "2": 0.7, "3": 0.8}
    assert pr["archive_best_depth_table_name"] ==         "Archive-best development score by recursive depth"
    assert pr["iteration_archive_best"][-1]["iteration"] == 9
    assert "structural" in pr["depth_vs_iteration_note"]


# -- 10 --------------------------------------------------------------------
def test_10_one_deployable_with_deterministic_ties(tmp_path=None):
    cands = [{"candidate_id": "b", "depth": 2, "creation_index": 1},
             {"candidate_id": "a", "depth": 2, "creation_index": 1},
             {"candidate_id": "z", "depth": 3, "creation_index": 0}]
    sel = M.select_deployable(cands, lambda c: 0.5)
    assert sel.candidate_id == "z"          # creation index wins first
    sel2 = M.select_deployable(cands[::1], lambda c: 1.0 if c["candidate_id"] != "z"
                               else 0.0)
    assert sel2.candidate_id == "a"         # then lexicographic


# -- 11 --------------------------------------------------------------------
def test_11_test_data_cannot_affect_selection(tmp_path=None):
    """Selection sees dev scores and nothing else.

    Passing a `test_score` field must not change the outcome -- the selector is
    never handed one, and a test-only winner stays unselected.
    """
    cands = [{"candidate_id": "dev_win", "depth": 2, "creation_index": 0,
              "test_score": 0.10},
             {"candidate_id": "test_win", "depth": 2, "creation_index": 1,
              "test_score": 0.99}]
    sel = M.select_deployable(cands, lambda c: 0.80 if c["candidate_id"] == "dev_win"
                              else 0.10)
    assert sel.candidate_id == "dev_win", sel
    # and the dev_score_of callable is the only channel consulted
    calls = []

    def dev_only(c):
        calls.append(c["candidate_id"])
        return 0.80 if c["candidate_id"] == "dev_win" else 0.10

    M.select_deployable(cands, dev_only)
    assert set(calls) == {"dev_win", "test_win"}


# -- 12 --------------------------------------------------------------------
def test_12_virtual_oracle_cannot_become_final_score(tmp_path=None):
    try:
        M.assert_no_oracle_as_final({"oracle_test_mean_score": 0.99})
        raise AssertionError("an oracle field passed")
    except PR.ProtocolError:
        pass
    M.assert_no_oracle_as_final({"oracle_upper_bound_dev": 0.99})


# -- 13 --------------------------------------------------------------------
def test_13_effective_gate_reflects_consolidation_bypass(tmp_path=None):
    # The frozen profile itself: consolidate=True, so `gate_tasks` MUST be 0 and
    # the table must say the gate will not run rather than echo a number.
    prof = _profile()
    on = PR.resolve_effective_config(prof, reduction_mode="official")
    assert on["gate_configured"] == 0 and on["gate_effective"] is False
    assert on["gate_reason"] == "consolidation_focus"
    assert "will NOT run" in PR.format_effective_config_table(on)
    assert on["within_task_recursion"] is True
    # A gated variant (consolidate off) reports the gate as ACTIVE.
    off = PR.resolve_effective_config(_profile(consolidate=False, gate_tasks=3),
                                      reduction_mode="official")
    assert off["gate_effective"] is True and off["gate_reason"] == "active"


# -- 14 --------------------------------------------------------------------
def test_14_formal_resume_rejects_result_affecting_drift(tmp_path=None):
    prof = _profile()
    base = dict(profile=prof.identity(), profile_sha256=prof.sha256(),
                root_bundle_sha256="R" * 64,
                shared={"T": 6, "model": "gpt-5.5", "cohort": list(PR.PRIMARY6)},
                operational={"output_dir": "/tmp/x"})
    a = PR.RunManifest(arm="official",
                       treatment={"reduction_mode": "official"}, **base)
    for key, val in (("T", 7), ("model", "other")):
        bad = PR.RunManifest(**a.to_dict())
        bad.shared[key] = val
        try:
            PR.assert_no_protocol_drift(a, bad)
            raise AssertionError("drift in {} accepted".format(key))
        except PR.ProtocolError as e:
            assert "resume drift" in str(e)
    # operational fields outside the allowlist are refused
    bad2 = PR.RunManifest(**a.to_dict())
    bad2.operational["sneaky"] = 1
    try:
        PR.assert_no_protocol_drift(a, bad2)
        raise AssertionError("an unlisted operational field was accepted")
    except PR.ProtocolError as e:
        assert "allowlist" in str(e)


# -- 15 --------------------------------------------------------------------
def test_15_root_hashes_and_shared_manifests_match_across_arms(tmp_path=None):
    prof = _profile()
    common = dict(profile=prof.identity(), profile_sha256=prof.sha256(),
                  root_bundle_sha256="R" * 64,
                  shared={"T": 6, "model": "gpt-5.5"})
    off = PR.RunManifest(arm="official",
                         treatment={"reduction_mode": "official"}, **common)
    pre = PR.RunManifest(arm="predictive",
                         treatment={"reduction_mode": "predictive",
                                    "gamma_checkpoint_sha256": "G" * 64},
                         **common)
    PR.assert_treatment_parity(off, pre)
    # a difference in a shared field is fatal
    bad = PR.RunManifest(arm="predictive",
                         treatment={"reduction_mode": "predictive"},
                         **dict(common, shared={"T": 7}))
    try:
        PR.assert_treatment_parity(off, bad)
        raise AssertionError("a shared-field difference was accepted")
    except PR.ProtocolError:
        pass
    # a different root is fatal
    bad2 = PR.RunManifest(arm="predictive",
                          treatment={"reduction_mode": "predictive"},
                          **dict(common, root_bundle_sha256="S" * 64))
    try:
        PR.assert_treatment_parity(off, bad2)
        raise AssertionError("a root mismatch was accepted")
    except PR.ProtocolError as e:
        assert "root bundle mismatch" in str(e)


# -- 16 --------------------------------------------------------------------
def test_16_pairing_effectiveness_is_truthful_at_depth_gt_one(tmp_path=None):
    rec = P.build_record(requested=True, model="m", backend="relay",
                         outer_seeded=True, inner_seeded=False,
                         executor_honours=True, test_seeded=True)
    assert rec.pairing_outer_effective is True
    assert rec.pairing_inner_effective is False
    assert any("inner llm()" in x for x in rec.pairing_limitations)
    try:
        P.assert_formal_pairing(rec, formal=True)
        raise AssertionError("an inert inner channel passed formal startup")
    except PR.ProtocolError as e:
        assert "INERT" in str(e)
    P.assert_formal_pairing(rec, formal=False)
    mf = rec.manifest_fields()
    assert "pairing_inner_effective" in mf and "pairing_limitations" in mf
    assert mf["crn_claimed"] is False


# -- 17 --------------------------------------------------------------------
def test_17_client_construction_works_against_the_supported_sdk(tmp_path=None):
    """Both real client contracts hold.

    (a) With no paid backend named, LLMClient resolves an OFFLINE backend and
        builds NO transport at all -- this is what keeps unit tests and smokes
        free and credential-free.
    (b) With a paid backend named AND the explicit opt-in, the OpenAI-compatible
        transport IS built and carries the extra-headers workaround our relay
        runs depend on (openai 3.14 + httpx2 zstd).
    """
    import os
    try:
        from meta_n.core.llm_client import LLMClient, LLMConfig
    except ImportError as e:
        # A missing optional SDK is not a protocol defect. SKIP with a reason
        # rather than reporting a red test on a box that simply lacks the dep.
        print("     (skipped: {}; run this test on the server box)".format(e))
        return

    def cfg():
        return LLMConfig(base_url="http://stub.invalid/v1", api_key="k",
                         model="deepseek-flash", daily_budget_usd=0)

    saved = {k: os.environ.get(k) for k in
             ("LLM_BACKEND", "ALLOW_PAID_API", "META_N_EXTRA_HEADERS_JSON")}
    try:
        # (a) offline default: no transport
        for k in saved:
            os.environ.pop(k, None)
        c = LLMClient(cfg())
        assert c._client is None, "an offline backend must build no transport"
        assert c._rpbe_backend is not None

        # (b) paid backend + opt-in: transport built, header forwarded.
        # An SDK-less box cannot run this half; SKIP rather than fail, because a
        # missing optional dependency is not a protocol defect -- and part (a),
        # the half that guarantees no transport exists without the opt-in, has
        # already run above.
        try:
            import openai  # noqa: F401
        except ImportError:
            print("     (skipped part (b): the openai SDK is not installed "
                  "here; run this on the server box)")
            return
        os.environ["META_N_EXTRA_HEADERS_JSON"] = '{"Accept-Encoding": "identity"}'
        os.environ["LLM_BACKEND"] = "relay"
        os.environ["ALLOW_PAID_API"] = "YES_I_ACCEPT_REAL_COST"
        c2 = LLMClient(cfg())
        assert c2._client is not None, "a paid backend must build the transport"
        hdrs = dict(getattr(c2._client, "default_headers", None) or {})
        assert hdrs.get("accept-encoding") == "identity" or \
            hdrs.get("Accept-Encoding") == "identity", hdrs
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# -- 18 --------------------------------------------------------------------
def test_18_aggregator_rejects_mixed_cohorts_and_reproduces_counts(tmp_path=None):
    def run(seed, arm, cohort, prof, final):
        return M.SeedRun(search_seed=seed, arm=arm, profile_sha256=prof,
                         cohort_id=cohort, root_bundle_sha256="R" * 64,
                         final_macro=final, per_transition={"R_2to3": 0.1})
    runs = [run(s, a, "sri_primary6:6", "P" * 64, 0.7 + 0.01 * s)
            for s in (0, 1) for a in ("official", "predictive")]
    agg = M.aggregate(runs)
    assert agg["n_runs"] == 4
    assert agg["per_arm"]["official"]["n_search_seeds"] == 2
    assert agg["paired"]["n_pairs"] == 2
    assert abs(agg["paired"]["final_delta_mean"]) < 1e-9
    for bad_cohort in ("sri_extended10:10",):
        try:
            M.aggregate(runs + [run(2, "official", bad_cohort, "P" * 64, 0.7)])
            raise AssertionError("a mixed cohort aggregated")
        except PR.ProtocolError:
            pass
    try:
        M.aggregate(runs + [run(2, "official", "sri_primary6:6", "Q" * 64, 0.7)])
        raise AssertionError("a mixed profile aggregated")
    except PR.ProtocolError:
        pass


# --------------------------------------------------------------------------

# -- 19 --------------------------------------------------------------------
def _orchestrator_source() -> str:
    p = (Path(__file__).resolve().parent.parent / "meta_n" / "core"
         / "evolutionary_orchestrator.py")
    return p.read_text(encoding="utf-8")


def test_19_nominal_grid_matches_the_orchestrators_iteration_base(tmp_path=None):
    """The nominal slot grid must equal the iterations the orchestrator ACTUALLY
    opens slots for.

    This test exists because the grid was 0-based while the breeding loop is
    1-based, so `assert_completeness` rejected every real run with
    `missing=[it0-*] extra=[it6-*]` -- and the offline fixture, also 0-based,
    passed green. A fixture cannot pin a contract it shares the assumption with,
    so this one reads the orchestrator's own source instead.
    """
    src = _orchestrator_source()

    # (a) the loop increments at the TOP, so the first bred generation is 1
    m = re.search(r"while\s+iteration\s*<\s*self\.config\.max_iterations[^\n]*:"
                  r"\s*\n\s*iteration\s*\+=\s*1", src)
    assert m is not None, (
        "the breeding loop no longer matches `while iteration < "
        "self.config.max_iterations: iteration += 1`; if its base changed, "
        "nominal_slot_ids() must change with it")
    # (b) the seed sets iteration to 0 BEFORE that loop, so 0 owns no slot
    assert re.search(r"\n\s+iteration\s*=\s*0\s*\n", src) is not None, (
        "the seed's `iteration = 0` assignment moved; re-derive the grid base")
    # (c) the slot is opened with that same 1-based variable
    assert "iteration=iteration," in src, (
        "self.sri.open() no longer receives `iteration=iteration`; the slot ids "
        "would no longer describe the orchestrator's generations")

    # (d) and the grid follows: 1..T, with 0 explicitly absent
    ids = L.nominal_slot_ids(2, 2, 6)
    assert ids[0] == "it1-p0-c0", ids[0]
    assert ids[-1] == "it6-p1-c1", ids[-1]
    assert not [s for s in ids if s.startswith("it0-")], (
        "iteration 0 must own no slot: it is the seed, which breeds nothing")
    # the size is still exactly B*K*T
    assert len(ids) == 2 * 2 * 6 == len(set(ids))


# -- 20 --------------------------------------------------------------------
def test_20_task_slug_mirrors_the_integration(tmp_path=None):
    """The SRI layer's slug must equal the id CO-Bench actually names things by.

    Every run ARTIFACT is keyed by the slug: the trace FILENAME, every
    `per_task_scores` / `per_task_best` key in the archive index, and the id the
    ledger records. The SRI layer keeps the display name only for the DATA
    directory and `--bench-tasks`. A real run proved the cost of getting this
    wrong: material loaders asking for `traces/<display name>.py` find nothing,
    so every edge becomes a missing-material drop.
    """
    # the shapes observed in a real run on the server
    assert PR.task_slug("Aircraft landing") == "aircraft_landing"
    assert PR.task_slug("Bin packing - one-dimensional") == \
        "bin_packing___one_dimensional"
    assert PR.task_slug("Common due date scheduling") == \
        "common_due_date_scheduling"
    # the profile keeps display names; the canonical cohort is the slugs
    prof = _profile()
    assert list(prof.cohort) == list(PR.PRIMARY6)
    assert prof.canonical_cohort == tuple(
        PR.task_slug(t) for t in prof.cohort)
    assert prof.canonical_cohort != prof.cohort
    assert PR.cohort_id_map(prof.cohort)[
        "bin_packing___one_dimensional"] == "Bin packing - one-dimensional"

    # ...and where the integration IS importable, check the mirror directly.
    # It needs a POSIX-only import chain, so on a box that cannot import it the
    # assertion above (pinned to the run's real keys) is what stands.
    try:
        from meta_n.integrations.co_bench import _task_id_from_name
    except ImportError as e:
        print("     (skipped the direct mirror check: {}; the pinned shapes "
              "above are what the server's artifacts show)".format(e))
        return
    for name in PR.EXTENDED10:
        assert PR.task_slug(name) == _task_id_from_name(name), name


# -- 21 --------------------------------------------------------------------
#: The KEY SET of a REAL run's config.json (runs/phaseH/phaseH_off, 72 keys).
#: Key names only -- no values -- so this is safe to keep in the repo, and it is
#: the cheapest possible guard against the most expensive kind of bug: a pinned
#: key that meta-n does not actually WRITE, which makes the arm stage raise only
#: after that arm's paid run has already finished.
REAL_CONFIG_JSON_KEYS = [
    "agentic_error_hints",
    "agentic_max_budget_usd",
    "agentic_max_turns",
    "agentic_preamble",
    "agentic_spend_budget",
    "agentic_temperature",
    "agentic_time_limit_s",
    "agentic_token_budget",
    "base_solver",
    "base_url",
    "beam_candidates",
    "beam_width",
    "bench_tasks",
    "benchmark",
    "benchmark_config",
    "benchmark_config_applied",
    "classify_balanced_json_fallback",
    "consolidate",
    "deploy_verified_code",
    "elite_rotation",
    "empty_retry_max_tokens",
    "epsilon",
    "eval_repeats",
    "eval_repeats_gate_topup",
    "exclude_providers",
    "executor",
    "focus_current_headroom",
    "force_code_library_live",
    "foster_adoption",
    "gate_margin",
    "gate_repeats",
    "gate_tasks",
    "instance_workers",
    # Added with the transport-retry knob (2026-09-21). This fixture is a
    # snapshot of a real config.json, so a newly RECORDED key belongs here --
    # main.py writes llm_max_retries next to max_retries. If that write is ever
    # dropped this test goes red again, which is the point: a key pinned for
    # verification must exist in a real run's config.
    "llm_max_retries",
    "max_depth",
    "max_docker",
    "max_iterations",
    "max_retries",
    "max_test",
    "max_tokens",
    "max_val",
    "model",
    "n_few_shot",
    "no_code_library",
    "no_early_stop",
    "no_outer_context",
    "novelty_alpha",
    "omega_context_budget",
    "orchestrator",
    "paired_eval",
    "parallel",
    "patience",
    "protect_floor",
    "reasoning_effort",
    "regression_guard",
    "regression_guard_repeats",
    "repropagation",
    "request_timeout",
    "resume_config_drift",
    "retry_threshold",
    "scratch_root",
    "seed",
    "seed_code_library",
    "solver_language",
    "symmetric_trace_sampling",
    "tasks_file",
    "temperatures",
    "timestamp",
    "use_agentic",
    "use_inspiration",
    "verified_code",
    "within_layer_refine",
    "within_task_recursion"
]


def test_21_every_verified_key_exists_in_a_real_config_json(tmp_path=None):
    """`verify_run_config` may only require keys a real run actually records.

    This caught a live one: `use_archive` was pinned for verification, but
    meta-n records the archive-orchestrator mode as `orchestrator:
    "evolutionary"` and never writes `use_archive` at all. The arm stage would
    have spent its whole budget and THEN refused the config.
    """
    prof = _profile()
    pinned = PR.pinned_run_config(prof, backbone="gpt-5.5", search_seed=0)
    real = set(REAL_CONFIG_JSON_KEYS)

    verified = [k for k in pinned if k not in PR.RENDER_ONLY_CONFIG_KEYS]
    missing = sorted(k for k in verified if k not in real)
    assert not missing, (
        "pinned for VERIFICATION but absent from a real config.json: {}. "
        "Either the key is not recorded (move it to RENDER_ONLY_CONFIG_KEYS and "
        "verify the provenance key that IS written), or the recording side "
        "changed.".format(missing))

    # the archive mode is verified through its real provenance key
    assert "orchestrator" in verified
    assert "use_archive" in PR.RENDER_ONLY_CONFIG_KEYS
    assert PR.CONFIG_KEY_TO_CLI.get("use_archive") == "--use-archive"

    # ...and the skip is not silent: without the provenance key it refuses
    cfg = {k: pinned[k] for k in verified if k != "orchestrator"}
    cfg.pop("orchestrator", None)          # the provenance key is ABSENT
    try:
        PR.verify_run_config(cfg, pinned, context="t")
        raise AssertionError("an unverifiable archive mode was accepted")
    except PR.ProtocolError as e:
        assert "UNVERIFIABLE" in str(e)


def test_22_slot_accounting_is_actual_and_reasoning_unavailability_is_explicit(
        tmp_path=None):
    with tempfile.TemporaryDirectory() as td:
        led = _ledger(Path(td))
        sid = led.open_slot(
            iteration=1, parent_slot=0, child_slot=0,
            parent_id="p", parent_structural_depth=1,
            gate_configured=0, gate_effective=False,
            gate_reason="consolidation_focus", reduction_mode="official")
        led.finalize(
            sid, terminal_status="evaluated_admitted",
            outer_calls=2, outer_successful_calls=1, inner_calls=3,
            evaluator_calls=3, prompt_tokens=11, completion_tokens=7,
            total_tokens=18, inner_prompt_tokens=3,
            inner_completion_tokens=2, inner_total_tokens=5,
            failed_calls=1, retry_count=1, empty_responses=1,
            reasoning_tokens=None, reasoning_tokens_available=False)
        row = led.rows()[0]
        assert row["token_usage"] == {
            "input_tokens": 14,
            "output_tokens": 9,
            "total_tokens": 23,
            "reasoning_tokens": None,
            "reasoning_tokens_available": False,
        }
        assert row["call_accounting"] == {
            "api_calls": 5,
            "outer_requests": 2,
            "outer_successful_calls": 1,
            "inner_calls": 3,
            "evaluator_calls": 3,
            "failed_calls": 1,
            "retry_count": 1,
            "empty_responses": 1,
        }
        assert row["total_wall_seconds"] == 0.0


TESTS = [(n, o) for n, o in sorted(globals().items())
         if n.startswith("test_") and callable(o)]


def main() -> int:
    print("SRI formal protocol -- 18 required offline tests (design §14)")
    print("=" * 72)
    failed = []
    for name, fn in TESTS:
        try:
            fn()
            print("  OK   {}".format(name))
        except Exception as e:                                   # noqa: BLE001
            failed.append((name, e))
            print("  FAIL {}  {}: {}".format(name, type(e).__name__, e))
    print("-" * 72)
    print("{} passed, {} failed".format(len(TESTS) - len(failed), len(failed)))
    print("VERDICT:", "ALL OK" if not failed else "FAILURES")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------- #
# §4b — the arm's Gamma must be the same FILE the plan bound
# --------------------------------------------------------------------------- #

def _gamma_file(tmp_path, payload=b"clean-traceonly-gamma"):
    p = tmp_path / "gamma.pt"
    p.write_bytes(payload)
    return p


def test_22_gamma_identity_accepts_a_match_and_returns_the_sha(tmp_path=None):
    """The happy path returns the sha it VERIFIED, so the caller can record it."""
    import hashlib
    import tempfile

    from scripts.run_sri_formal import assert_gamma_identity

    with tempfile.TemporaryDirectory() as td:
        p = _gamma_file(Path(td))
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        assert assert_gamma_identity(sha, sha, str(p)) == sha


def test_23_gamma_identity_refuses_a_stale_or_swapped_checkpoint(tmp_path=None):
    """The three failures mean different things, so all three must raise.

    This is the case `load_gamma_checkpoint` cannot catch: the retired Gamma
    (trained on the superseded traces+codes records) shares its geometry with
    the clean trace-only one, so it loads without complaint. Only the FILE hash
    distinguishes them.
    """
    import hashlib
    import tempfile

    import pytest

    from scripts.run_sri_formal import StageError, assert_gamma_identity

    with tempfile.TemporaryDirectory() as td:
        p = _gamma_file(Path(td))
        good = hashlib.sha256(p.read_bytes()).hexdigest()

        # unpinned plan: identity was never bound
        with pytest.raises(StageError, match="recorded no gamma_checkpoint_sha256"):
            assert_gamma_identity(None, good, str(p))

        # plan vs arm: the recorded treatment is not the executed one
        with pytest.raises(StageError, match="identity drift"):
            assert_gamma_identity("A" * 64, good, str(p))

        # file vs arm: plan and arm AGREE, but the file on disk is not that
        # Gamma -- the checkpoint was swapped between planning and execution
        stale = _gamma_file(Path(td), b"retired-traces-plus-codes-gamma")
        stale_sha = hashlib.sha256(stale.read_bytes()).hexdigest()
        assert stale_sha != good
        with pytest.raises(StageError, match="replaced between planning"):
            assert_gamma_identity(good, good, str(stale))

        # missing file: identity cannot be checked at all
        with pytest.raises(StageError, match="is missing"):
            assert_gamma_identity(good, good, str(Path(td) / "gone.pt"))

