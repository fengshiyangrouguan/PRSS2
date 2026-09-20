"""End-to-end acceptance for the SRI formal runner (design §12, audit item #8).

The unit tests in `test_sri_formal_protocol.py` check each protocol module in
isolation. This one drives the RUNNER's post-search stages for real -- freeze ->
audit -> final -> aggregate -- against a fabricated but structurally complete
experiment: a shared root, two forked arms, 24 nominal slots each (one
gate-rejected edge with out-of-archive material, one deliberately dropped edge),
a synthesized `merge_oracle` that must never win, and two search seeds so the
aggregator has something to pair.

The SEARCH itself is played by the fixture: it is the only way to exercise these
stages offline, and it is also the honest split -- the runner consumes
artifacts and never generates them.

Four gates get their own negative tests, because a gate that has never fired is
a gate nobody has tested:

  * a frozen artifact edited after the freeze -> the audit refuses;
  * a ledger missing a terminal row            -> the freeze refuses;
  * a knob differing OUTSIDE the treatment     -> the freeze refuses;
  * an arm that REGENERATED its root instead of inheriting the fork (§4) ->
    the inheritance check refuses. Nothing else can see this one: a failed
    `--resume` starts a fresh run whose ledger still fills every nominal slot.

One further test pins a property rather than a gate:
`test_cached_repeats_are_marked_and_the_test_split_never_caches` covers the
repeat cache -- that it is verified per task before it is used, that a cached
repeat SAYS it was cached in the raw row, and that the held-out test split is
never served from it.

Runs with pytest or as a plain script:

    python -m pytest tests/test_sri_formal_e2e.py -q
    python tests/test_sri_formal_e2e.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from meta_n.sri import protocol as PR                    # noqa: E402
from meta_n.sri import ledger as L                      # noqa: E402
from meta_n.sri.protocol import (RunManifest, load_profile,  # noqa: E402
                                 sha256_of)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "sri_runner", REPO / "scripts" / "run_sri_formal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_runner()
PROFILE = load_profile(PR.default_profile_path("sri_primary6"))
# The CANONICAL id space (slugs). The fixture deliberately uses the same ids the
# real artifacts do -- trace filenames and `per_task_scores` keys are slugs, not
# display names. Using display names here is how the earlier fixture passed
# green while every real run would have found no material at all.
COHORT = list(PROFILE.canonical_cohort)


def _args(out: Path, **over):
    base = dict(profile="sri_primary6", backbone="gpt-5.5", search_seed=0,
                out=str(out), stage="all", gamma_checkpoint=None,
                execute=True, allow_inert_pairing=False, evaluator="mock",
                data_dir="./data/co_bench", timeout=10)
    base.update(over)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------
# the fixture: a fabricated but structurally complete experiment
# --------------------------------------------------------------------------

def _write_traces(cand_dir: Path, cid: str) -> None:
    tdir = cand_dir / "traces"
    tdir.mkdir(parents=True, exist_ok=True)
    for t in COHORT:
        (tdir / (t + ".py")).write_text(
            "# {}\n# {}\ndef solve():\n    return 0\n".format(cid, t),
            encoding="utf-8")


def _add_candidate(archive: Path, cid: str, depth: int, score: float) -> dict:
    d = archive / cid
    _write_traces(d, cid)
    return {"candidate_id": cid, "parent_id": None, "iteration": 0,
            "depth": depth, "mean_score": score, "pass_at_1": score,
            "num_children": 0, "temperature_used": 0.7, "total_tokens": 0,
            "per_task_scores": {t: score for t in COHORT}}


def _slot_plan():
    """24 slots with a depth pattern that produces all four audited transitions.

    Iterations are **1..T** (the orchestrator's breeding loop increments at the
    top, so iteration 0 is the seed and owns no slot). `pd`/`cd` are capped at
    the profile's max_depth, so the last two iterations sit at 5->6 and fall
    outside the audited transition set -- which is itself worth having in the
    fixture: out-of-scope rows must not enter any denominator.
    """
    out = []
    for it in range(1, PROFILE.max_iterations + 1):
        for ps in range(PROFILE.beam_width):
            for cs in range(PROFILE.beam_candidates):
                pd = min(it, PROFILE.max_depth - 1)
                cd = min(it + 1, PROFILE.max_depth)
                out.append({"slot_id": L.slot_id_for(it, ps, cs), "iteration": it,
                            "parent_slot": ps, "child_slot": cs,
                            "pd": pd, "cd": cd})
    return out


def _fabricate_seed(out: Path, seed: int) -> dict:
    """Write the root, both arms, and the stage records a real search would."""
    out.mkdir(parents=True, exist_ok=True)
    root_exp = R.root_exp_dir(out)
    archive = root_exp / "archive"
    archive.mkdir(parents=True, exist_ok=True)

    plan = _slot_plan()
    # parents at every depth a slot can start from
    parents = sorted({(s["pd"], s["parent_slot"]) for s in plan})
    cands = [_add_candidate(archive, "gen0_seed", 1, 0.50)]
    for depth, ps in parents:
        cands.append(_add_candidate(archive, "p{}_{}".format(depth, ps), depth,
                                    0.50 + 0.01 * depth))
    for s in plan:
        # the ONE deliberately dropped child (no material anywhere) proves the
        # failure accounting is not silently zero
        if s["slot_id"] == "it2-p1-c1":
            continue
        cands.append(_add_candidate(archive, "c_" + s["slot_id"], s["cd"],
                                    0.50 + 0.01 * s["cd"]))
    # the synthesized oracle: highest score, fabricated depth -- it must lose
    cands.append(_add_candidate(archive, "merge_oracle", 2, 0.99))
    (archive / "index.json").write_text(json.dumps(
        {"size": len(cands), "best_mean_score": 0.99,
         "best_candidate_id": "merge_oracle",
         "per_task_best": {t: {"score": 0.99, "candidate_id": "merge_oracle"}
                           for t in COHORT},
         "candidates": cands}, indent=2, sort_keys=True), encoding="utf-8")
    (root_exp / "summary.json").write_text(json.dumps({"mock": True}),
                                           encoding="utf-8")
    # the checkpoint the arms resume from: `try_resume` reads
    # `<out_dir>/checkpoint.json` and rebuilds the archive from `<out_dir>/archive`
    (root_exp / "checkpoint.json").write_text(json.dumps(
        {"iteration": 0, "patience_counter": 0, "prev_best": 0.5,
         "total_tokens": 0, "convergence_history": [0.5],
         "rng_state": [1, [1], None]}), encoding="utf-8")

    pinned = PR.pinned_run_config(PROFILE, backbone="gpt-5.5", search_seed=seed)
    cfg = dict(pinned)
    cfg["sri"] = {"run_id": R.run_id_for(PROFILE, "gpt-5.5", seed),
                  "arm": "official", "search_seed": seed,
                  "reduction_mode": "official"}
    (root_exp / "config.json").write_text(json.dumps(cfg, indent=2,
                                                     sort_keys=True),
                                          encoding="utf-8")

    bundle = PR.collect_root_bundle(
        archive_dir=archive, cohort=COHORT, backbone="gpt-5.5",
        search_seed=seed, temperatures=PROFILE.temperatures,
        max_tokens=PROFILE.max_tokens,
        reasoning_effort=PROFILE.reasoning_effort,
        worker_config={"parallel": 1, "instance_workers": 2})
    rb = PR.root_bundle_sha256(bundle)
    inherited = sha256_of(PR.root_candidate_digest(archive, COHORT))
    (out / "root_bundle.json").write_text(json.dumps(bundle, indent=2,
                                                    sort_keys=True),
                                          encoding="utf-8")

    R.record_stage(out, "preflight",
                   inputs={"profile_sha256": PROFILE.sha256(),
                           "backbone": "gpt-5.5", "search_seed": seed,
                           "evaluator": "mock"},
                   outputs={"effective_config": PR.resolve_effective_config(
                                PROFILE, reduction_mode="official"),
                            "pairing": {"pairing_requested": False,
                                        "pairing_limitations": []},
                            "pinned": pinned, "evaluator_mode": "mock"})
    R.record_stage(out, "root",
                   inputs={"profile_sha256": PROFILE.sha256(),
                           "backbone": "gpt-5.5", "search_seed": seed},
                   outputs={"root_dir": str(root_exp), "root_bundle_sha256": rb,
                            "inherited_root_sha256": inherited,
                            "environment_sha256": sha256_of(
                                bundle["environment_fingerprint"]),
                            "root_config_sha256": PR.sha256_file(
                                root_exp / "config.json")},
                   extra={"root_bundle": bundle})
    R.record_stage(out, "fork", inputs={"root_bundle_sha256": rb},
                   outputs={"arms": {a: str(R.arm_dir(out, a)) for a in R.ARMS},
                            "gamma_checkpoint_sha256":
                                ("G" * 64 if seed >= 0 else None),
                            "rule5_byte_identical": True})

    for arm in R.ARMS:
        ad = R.arm_dir(out, arm)
        ad.mkdir(parents=True, exist_ok=True)
        for child in archive.iterdir():
            if child.is_dir():
                import shutil
                shutil.copytree(child, ad / "archive" / child.name,
                                dirs_exist_ok=True)
            else:
                (ad / "archive").mkdir(parents=True, exist_ok=True)
                (ad / "archive" / child.name).write_bytes(child.read_bytes())
        (ad / "summary.json").write_text(json.dumps({"mock": True}),
                                         encoding="utf-8")
        (ad / "checkpoint.json").write_text(
            (root_exp / "checkpoint.json").read_text(encoding="utf-8"),
            encoding="utf-8")
        acfg = dict(pinned)
        # `timestamp` differs between arms in a REAL run (they start at
        # different moments) and is on the §10 operational allowlist, so the
        # freeze's config-diff gate must tolerate it. Without this the fixture
        # would pass while every real run was rejected.
        acfg["timestamp"] = "2026-09-20T00:00:0{}.000000".format(
            1 if arm == "official" else 9)
        acfg["sri"] = {"run_id": R.run_id_for(PROFILE, "gpt-5.5", seed),
                       "arm": arm, "search_seed": seed,
                       "reduction_mode": arm,
                       "gamma_checkpoint_sha256":
                           ("G" * 64 if arm == "predictive" else None)}
        (ad / "config.json").write_text(json.dumps(acfg, indent=2,
                                                   sort_keys=True),
                                        encoding="utf-8")

        man = RunManifest(
            arm=arm, profile=PROFILE.identity(),
            profile_sha256=PROFILE.sha256(), root_bundle_sha256=rb,
            shared={"profile_sha256": PROFILE.sha256(),
                    "cohort": list(PROFILE.cohort),
                    "root_bundle_sha256": rb, "backbone": "gpt-5.5",
                    "search_seed": seed, "evaluator_mode": "mock",
                    "inherited_root_sha256": inherited},
            treatment=({"reduction_mode": arm} if arm == "official" else
                       {"reduction_mode": arm,
                        "gamma_checkpoint_sha256": "G" * 64}))
        man.write(ad / "manifest.json")

        led = L.SlotLedger(ad / R.LEDGER_NAME,
                           run_id=R.run_id_for(PROFILE, "gpt-5.5", seed),
                           arm=arm, backbone="gpt-5.5",
                           cohort_id=PROFILE.cohort_id, search_seed=seed,
                           root_bundle_sha256=rb,
                           gamma_checkpoint_sha256=(
                               "G" * 64 if arm == "predictive" else None))
        rejected_mp = None
        for s in plan:
            sid = s["slot_id"]
            led.open_slot(iteration=s["iteration"],
                          parent_slot=s["parent_slot"],
                          child_slot=s["child_slot"],
                          parent_id="p{}_{}".format(s["pd"], s["parent_slot"]),
                          parent_structural_depth=s["pd"],
                          gate_configured=0, gate_effective=False,
                          gate_reason="consolidation_focus",
                          reduction_mode=arm,
                          proposed_child_id="c_" + sid,
                          proposed_child_depth=s["cd"])
            if sid == "it2-p0-c1":
                # a gate-rejected but EXECUTABLE child, material kept OUT of the
                # archive: §2.2 requires it to be audited anyway
                rd = ad / R.REJECTED_DIRNAME / sid
                _write_traces(rd, "c_" + sid)
                rejected_mp = str(rd)
                led.finalize(sid, terminal_status="gate_rejected",
                             gate_status="rejected", archive_admitted=False,
                             archive_rejection_reason="quality_gate",
                             material_path=rejected_mp)
            elif sid == "it5-p0-c1":
                led.finalize(sid, terminal_status="empty_injection",
                             failure_class="empty_omega")
            else:
                led.finalize(sid, terminal_status="evaluated_admitted",
                             archive_admitted=True,
                             fresh_task_ids=list(COHORT),
                             outer_calls=1, prompt_tokens=10,
                             completion_tokens=5)
        R.record_stage(out, arm,
                       inputs={"manifest_sha256":
                                   RunManifest.read(ad / "manifest.json").sha256(),
                               "root_bundle_sha256": rb},
                       outputs={"arm_dir": str(ad), "ledger":
                               str(ad / R.LEDGER_NAME),
                               "config_sha256": PR.sha256_file(
                                   ad / "config.json")})
    return {"root_bundle_sha256": rb, "bundle": bundle}


def _seed_dir(parent: Path, seed: int) -> Path:
    d = parent / "s{}".format(seed)
    _fabricate_seed(d, seed)
    return d


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_e2e_freeze_audit_final_aggregate():
    with tempfile.TemporaryDirectory() as td:
        parent = Path(td)
        d0 = _seed_dir(parent, 0)
        d1 = _seed_dir(parent, 1)
        a = _args(d0)

        frz = R.stage_freeze(a, PROFILE, d0)
        counts = frz["outputs"]["ledger_counts"]
        assert counts["official"]["evaluated_admitted"] == 22
        assert counts["official"]["gate_rejected"] == 1
        assert counts["official"]["empty_injection"] == 1
        assert sum(counts["official"].values()) == PROFILE.nominal_slots == 24
        assert frz["outputs"]["treatment_parity"] is True

        aud = R.stage_audit(a, PROFILE, d0)
        off = aud["outputs"]["arms"]["official"]
        assert off["planned"] is False
        # the 2->3 transition is iteration 2 (1-based grid): 4 slots, 3 edges,
        # one deliberately unmaterialized
        # unmaterialized child (a drop) -- so the rejected-but-executable child
        # IS audited and the missing one is DISCLOSED rather than ignored
        assert off["R_2to3"] is not None
        edges = [json.loads(l) for l in
                 (d0 / "audit" / "official" / "edges.jsonl").read_text(
                     encoding="utf-8").splitlines() if l.strip()]
        r23 = [e for e in edges if e["transition"] == "23"]
        assert len(r23) == 3, [e["slot_id"] for e in r23]
        assert {e["slot_id"] for e in r23} == {"it2-p0-c0", "it2-p0-c1",
                                              "it2-p1-c0"}
        assert any(not e["child_was_admitted"] and
                   e["child_terminal_status"] == "gate_rejected" for e in r23)
        metrics = json.loads((d0 / "audit" / "official" / "metrics.json")
                             .read_text(encoding="utf-8"))
        invalid = metrics["invalid_edges"]
        assert [d["slot_id"] for d in invalid] == ["it2-p1-c1"], invalid
        assert invalid[0]["transition"] == "23"
        # ...so the failure counts are non-zero rather than silently 0
        assert off["drops"] >= 1
        assert off["F_overall"] > 0.0
        assert off["complete"] is False
        # §2.3 symbols are the OVERALL values; the 2->3 view is the diagnostic
        fa_all = metrics["failure_accounting"]
        assert fa_all["scope"] == "overall"
        assert fa_all["expected_edge_task_pairs"] == fa_all[
            "valid_edge_task_pairs"] + fa_all["missing_edge_task_pairs"]
        assert metrics["completeness"]["ok"] is False
        # 2->3: expected = (3 edges + 1 drop) * 6 tasks = 24, missing = 6
        fa = fa_all["by_transition"]["R_2to3"]
        assert fa["attempted_edge_task_slots"] == 24, fa
        assert fa["failed_edge_task_slots"] == 6, fa
        assert abs(fa["F"] - 0.25) < 1e-12, fa
        assert metrics["evaluator_mode"] == "mock"
        # repeats never enlarge the denominator, and they are never reported as
        # independent observations
        assert metrics["transitions"]["R_2to3"]["valid_edge_task_pairs"] == \
            3 * len(COHORT)
        assert metrics["repeats"]["requested"] == PROFILE.audit_repeats
        assert metrics["repeats"]["executed"] <= metrics["repeats"]["requested"] \
            * metrics["deduplicated_executions"]
        assert "NEVER an independent observation" in metrics["repeats"]["note"]

        fin = R.stage_final(a, PROFILE, d0)
        off_fin = fin["outputs"]["arms"]["official"]
        assert off_fin["selected"] != "merge_oracle", off_fin
        assert off_fin["selected"].startswith("c_it"), off_fin
        assert isinstance(off_fin["test_macro"], float)
        pr = json.loads((d0 / "final" / "official.json").read_text(
            encoding="utf-8"))
        assert "merge_oracle" in pr["synthesized_excluded_from_depth"]
        assert pr["final"]["selected_candidate_id"] == off_fin["selected"]
        # the depth series must NOT report the oracle's 0.99 as depth-2's best
        assert pr["exact_depth_dev_best"].get("2") != 0.99, \
            pr["exact_depth_dev_best"]
        assert "merge_oracle" in [e.get("candidate_id") for e in
                                 pr["selected_pool"]["excluded"]], \
            pr["selected_pool"]
        # Final Score is a single deployable candidate, never the oracle
        assert pr["final"]["final_test_macro_score"] == off_fin["test_macro"]

        # the second seed goes through the identical pipeline so the aggregate
        # has two independent runs per arm to pool
        R.stage_freeze(_args(d1), PROFILE, d1)
        R.stage_audit(_args(d1), PROFILE, d1)
        R.stage_final(_args(d1), PROFILE, d1)

        agg = R.stage_aggregate(_args(parent), PROFILE, parent)
        payload = json.loads((parent / "aggregate.json").read_text(
            encoding="utf-8"))
        assert payload["n_runs"] == 4, payload["n_runs"]
        assert payload["per_arm"]["official"]["n_search_seeds"] == 2
        assert payload["paired"]["n_pairs"] == 2
        assert payload["paired"]["final_delta_mean"] is not None
        assert payload["evaluator_mode"] == "mock"
        assert payload["seed_dirs"] == [str(d0), str(d1)]
        assert agg["outputs"]["n_runs"] == 4

        del agg, d1  # assertions above are the point


def test_audit_refuses_a_frozen_artifact_that_moved():
    with tempfile.TemporaryDirectory() as td:
        d = _seed_dir(Path(td), 0)
        a = _args(d)
        R.stage_freeze(a, PROFILE, d)
        # a post-freeze edit: exactly what the freeze exists to catch
        (R.arm_dir(d, "official") / "archive" / "index.json").write_text(
            json.dumps({"size": 0, "candidates": []}), encoding="utf-8")
        try:
            R.stage_audit(a, PROFILE, d)
            raise AssertionError("the audit ran on a moved frozen artifact")
        except R.StageError as e:
            assert "CHANGED after the freeze" in str(e)


def test_freeze_refuses_an_incomplete_ledger():
    with tempfile.TemporaryDirectory() as td:
        d = _seed_dir(Path(td), 0)
        p = R.arm_dir(d, "official") / R.LEDGER_NAME
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        # drop the LAST finalize record: that slot is now open with no terminal
        # row, which must fail completion instead of shrinking the denominator
        p.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        try:
            R.stage_freeze(_args(d), PROFILE, d)
            raise AssertionError("an incomplete ledger was frozen")
        except PR.ProtocolError as e:
            assert "not complete" in str(e)


def test_freeze_refuses_a_knob_that_differs_outside_the_treatment():
    with tempfile.TemporaryDirectory() as td:
        d = _seed_dir(Path(td), 0)
        p = R.arm_dir(d, "predictive") / "config.json"
        cfg = json.loads(p.read_text(encoding="utf-8"))
        cfg["eval_repeats"] = 1            # a shared knob silently drifted
        p.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
        try:
            R.stage_freeze(_args(d), PROFILE, d)
            raise AssertionError("arms differing outside the treatment froze")
        except R.StageError as e:
            assert "OUTSIDE the treatment" in str(e)


def test_aggregate_refuses_to_pool_a_mock_with_a_real_run():
    with tempfile.TemporaryDirectory() as td:
        parent = Path(td)
        d0 = _seed_dir(parent, 0)
        _seed_dir(parent, 1)
        a = _args(d0)
        for d in sorted(parent.iterdir()):
            R.stage_freeze(_args(d), PROFILE, d)
            R.stage_audit(_args(d), PROFILE, d)
            R.stage_final(_args(d), PROFILE, d)
        # stamp one arm as if it had run against the real evaluator
        f = d0 / "final" / "official.json"
        obj = json.loads(f.read_text(encoding="utf-8"))
        obj["evaluator_mode"] = "real"
        f.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
        try:
            R.stage_aggregate(_args(parent), PROFILE, parent)
            raise AssertionError("a mock run was pooled with a real one")
        except R.StageError as e:
            assert "different evaluators" in str(e)
        del a


def test_arm_must_inherit_the_shared_root():
    with tempfile.TemporaryDirectory() as td:
        d = _seed_dir(Path(td), 0)
        want = R.read_stage_manifest(d)["root"]["outputs"][
            "inherited_root_sha256"]
        # the clean case
        assert R.assert_root_inherited(R.arm_dir(d, "official"), want,
                                       COHORT,
                                       label="arm official") == want
        # a REGENERATED root: same filenames, different program source -- which
        # is exactly what a failed `--resume` produces. Every other artifact
        # looks normal, so only this check can see it.
        bad = R.arm_dir(d, "official") / "archive" / "gen0_seed" / "traces"
        (bad / (COHORT[0] + ".py")).write_text(
            "# regenerated\ndef solve():\n    return 1\n", encoding="utf-8")
        try:
            R.assert_root_inherited(R.arm_dir(d, "official"), want,
                                    COHORT, label="arm official")
            raise AssertionError("a regenerated root was accepted")
        except R.StageError as e:
            assert "did NOT inherit" in str(e) and "arm official" in str(e)



def _bare_evaluator(*, deterministic_cache=True, dev_repeats=3, scores=None):
    """A COBenchEvaluator whose execution is stubbed, so `_run`'s caching policy
    can be tested without the CO-Bench SDK (which needs a POSIX-only import).
    """
    ev = R.COBenchEvaluator.__new__(R.COBenchEvaluator)
    ev.mode = "real"
    ev.cohort = list(COHORT)
    ev.data_dir = Path(".")
    ev._evals = {}
    ev.deterministic_cache = deterministic_cache
    ev.dev_repeats = dev_repeats
    ev._cache, ev._probe, ev._probe_src = {}, {}, {}
    ev._verified, ev._violated = set(), set()
    ev.calls = ev.fresh = ev.cache_hits = 0
    seq = list(scores or [])
    seen: Dict[str, int] = {}

    def fake_execute(task_id, source, split):
        ev.calls += 1
        ev.fresh += 1
        if seq:
            i = seen.get(task_id, 0)
            seen[task_id] = i + 1
            return R.RawEval(score=float(seq[i % len(seq)]), success=True,
                             freshly_executed=True,
                             detail="stub:{}".format(split))
        return R.RawEval(score=0.5, success=True, freshly_executed=True,
                         detail="stub:{}".format(split))

    ev._execute = fake_execute
    return ev, seen


def test_cached_repeats_are_marked_and_the_test_split_never_caches():
    """The repeat cache must be VERIFIED and VISIBLE, and must never serve test.

    The defect this pins: a repeat served from the cache was byte-identical to a
    fresh execution in the raw artifact -- same `freshly_executed: True`, same
    detail -- so a reader could not tell three observations from one.
    """
    ev, _ = _bare_evaluator(deterministic_cache=True, dev_repeats=3)
    # candidate A on the dev split: all three repeats EXECUTE (they are the probe)
    a = [ev.evaluate("T1", "src_A", seed=r) for r in range(3)]
    assert ev.fresh == 3 and ev.cache_hits == 0, (ev.fresh, ev.cache_hits)
    assert all("cached" not in x.detail for x in a)
    # the repeats agreed, so the task is now verified deterministic
    assert "dev|T1" in ev.determinism_report()["verified_tasks"]

    # candidate B on the same task: one fresh + two CACHED, and each says so
    b = [ev.evaluate("T1", "src_B", seed=r) for r in range(3)]
    assert ev.fresh == 4, ev.fresh                 # only the first B call ran
    assert ev.cache_hits == 2, ev.cache_hits
    assert "cached" not in b[0].detail
    assert all("cached_deterministic_repeat" in x.detail for x in b[1:])
    assert b[1].score == b[0].score

    # the TEST split is never cached, however deterministic the evaluator is
    t = [ev.evaluate_test("T1", "src_A", seed=r) for r in range(3)]
    assert ev.cache_hits == 2, ev.cache_hits       # unchanged by the test calls
    assert all("cached" not in x.detail for x in t)
    assert ev.determinism_report()["note"].count("test split is never cached")

    # a task whose repeats DISAGREE is never cached, and is reported
    ev2, _ = _bare_evaluator(deterministic_cache=True, dev_repeats=3,
                             scores=[0.1, 0.2, 0.3, 0.4])
    for r in range(3):
        ev2.evaluate("T1", "src_A", seed=r)
    assert "dev|T1" in ev2.determinism_report()["violated_tasks"]
    ev2.evaluate("T1", "src_B", seed=0)
    assert ev2.cache_hits == 0, "a non-deterministic task must never be cached"

    # with the declaration off, nothing is cached at all
    ev3, _ = _bare_evaluator(deterministic_cache=False, dev_repeats=3)
    for r in range(3):
        ev3.evaluate("T1", "src_A", seed=r)
    for r in range(3):
        ev3.evaluate("T1", "src_A", seed=r)
    assert ev3.cache_hits == 0 and ev3.fresh == 6


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("OK  {}".format(fn.__name__))
    print("\nVERDICT: ALL OK ({} e2e test(s))".format(len(fns)))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
