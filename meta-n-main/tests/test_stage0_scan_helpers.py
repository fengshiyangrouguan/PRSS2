"""S0.2 unit gate — `scan_helper_calls` + the 4 adoption Trace fields.

Covers:
  * `scan_helper_calls` reproduces the omega.py:453-476 regex semantics
    (def-shadow exclusion + helpers/-anchored FILE form) for a single script.
  * The omega `## Helper Usage` section is byte-identical to a hand-rolled
    reference using the OLD inline logic (single-source-of-truth equivalence).
  * `MetaLayer.execute` populates the adoption fields at the fresh-trace
    finalize site: live helpers -> measured list; demoted (no live helpers) ->
    `utilities_called is None`; a RETRIED task reports `command_count > 1`.
  * Additive-field round-trip: `model_dump -> model_validate` and
    `Archive.rebuild_from_disk` preserve the defaults.
"""

from __future__ import annotations

import json
import re

import pytest

from meta_n.core.archive import Archive, Candidate
from meta_n.core.meta_layer import (
    InjectedCode,
    MetaLayer,
    TaskDescription,
    Trace,
    scan_helper_calls,
)


# --------------------------------------------------------------------------- #
# scan_helper_calls — regex semantics
# --------------------------------------------------------------------------- #
def _old_called(script: str, name: str) -> bool:
    """The exact pre-S0.2 inline predicate from omega.py:453-476."""
    pat = re.compile(rf"\b{re.escape(name)}\b")
    def_pat = re.compile(rf"\bdef\s+{re.escape(name)}\b")
    file_pat = re.compile(rf"helpers/(?:_lib_)?{re.escape(name)}\.(?:py|sh)")
    return bool((pat.search(script) and not def_pat.search(script)) or file_pat.search(script))


def test_scan_bare_word_hit():
    called, counts = scan_helper_calls("x = range(10)\nrange(3)", ["range"])
    assert called == ["range"]
    assert counts["range"] == 2


def test_scan_def_shadow_excluded():
    """A local `def <name>` shadows the helper -> NOT counted as a call."""
    script = "def get_swap_delta(a, b):\n    return get_swap_delta(a-1, b)"
    called, counts = scan_helper_calls(script, ["get_swap_delta"])
    assert called == []
    assert counts == {}


def test_scan_file_form_counted_unconditionally():
    """The helpers/(_lib_)<name>.{py,sh} path form counts even when def-shadowed."""
    script = "def mcmf(): ...\nsubprocess.run(['python3', './helpers/_lib_mcmf.py'])"
    called, _ = scan_helper_calls(script, ["mcmf"])
    assert called == ["mcmf"]  # file form wins despite the def shadow


def test_scan_absent_name():
    called, counts = scan_helper_calls("print('hi')", ["never_referenced"])
    assert called == []
    assert counts == {}


def test_scan_matches_old_predicate_across_cases():
    scripts = [
        "range(10)",
        "def range(): pass\nrange()",
        "source ./helpers/log_summary.sh",
        "python3 helpers/_lib_solve.py",
        "no helper here",
    ]
    for s in scripts:
        for name in ("range", "log_summary", "solve"):
            called, _ = scan_helper_calls(s, [name])
            assert (name in called) == _old_called(s, name), (s, name)


def test_scan_native_inline_lib_form_counted():
    # Native bash path advertises+stages `python3 /tmp/_lib_<name>.py` (code_library.py
    # default lib_path_fmt) — the solver invokes the helper via that path, so it must count.
    called, counts = scan_helper_calls("python3 /tmp/_lib_foo.py 3 5", ["foo"])
    assert called == ["foo"] and counts["foo"] == 1
    called_sh, _ = scan_helper_calls("bash /tmp/_lib_router.sh", ["router"])
    assert called_sh == ["router"]
    # cross-detector parity: agrees with the post-hoc telemetry detector on this form.
    from meta_n.analysis.injection_telemetry import helper_called_in
    assert helper_called_in("foo", "python3 /tmp/_lib_foo.py 3 5") is True


# --------------------------------------------------------------------------- #
# MetaLayer.execute population
# --------------------------------------------------------------------------- #
class _FakeExecutor:
    """Returns a scripted score per call; records the executed script."""

    def __init__(self, scores):
        self._scores = list(scores)
        self.scripts: list[str] = []

    async def execute(self, script: str, task: TaskDescription) -> Trace:
        self.scripts.append(script)
        score = self._scores.pop(0) if self._scores else 0.0
        return Trace(
            task_id=task.task_id,
            script=script,
            score=score,
            success=score >= 0.999,
        )


class _FakeSolver:
    def __init__(self, script: str):
        self.script = script

    async def solve(self, task, additional_context: str = "", **kw):
        return self.script, "reasoning", 7


TASK = TaskDescription(task_id="t1", description="d")


async def test_execute_live_helpers_measured_list():
    """Live helper present -> utilities_available set, utilities_called measured."""
    solver = _FakeSolver("result = mcmf_solve(graph)\nappend(out, result)")
    layer = MetaLayer(
        depth=2,
        injected_code=InjectedCode(code_library={"mcmf_solve": "def mcmf_solve(): ..."}),
        inner_solver=solver,
        executor=_FakeExecutor([1.0]),
        merged_code_library={"mcmf_solve": "def mcmf_solve(): ...", "unused_helper": "def unused_helper(): ..."},
    )
    trace, _ = await layer.execute(TASK)
    assert trace.utilities_available == ["mcmf_solve", "unused_helper"]  # sorted
    assert trace.utilities_called == ["mcmf_solve"]      # only the called one
    assert trace.utilities_call_counts == {"mcmf_solve": 1}
    assert trace.command_count == 1


async def test_execute_demoted_path_keeps_none():
    """No live helpers (CO-Bench demoted) -> utilities_called stays None."""
    solver = _FakeSolver("def solve():\n    return 42")
    layer = MetaLayer(
        depth=2,
        injected_code=InjectedCode(),
        inner_solver=solver,
        executor=_FakeExecutor([1.0]),
        merged_code_library={},          # demoted: python library zeroed upstream
        merged_code_library_bash={},
    )
    trace, _ = await layer.execute(TASK)
    assert trace.utilities_called is None        # None, NOT a misleading []
    assert trace.utilities_available == []
    assert trace.utilities_call_counts == {}
    assert trace.command_count == 1              # but command_count is still measured


async def test_execute_retried_task_command_count_gt_one():
    """The meta_layer.py:716 retry path reports command_count > 1 (not 0)."""
    # First solve scores 0.2 (< threshold 0.5) -> 2 retries; second improves to 0.9.
    executor = _FakeExecutor([0.2, 0.9, 0.9])
    layer = MetaLayer(
        depth=2,
        injected_code=InjectedCode(),
        inner_solver=_FakeSolver("def solve(): return 1"),
        executor=executor,
        merged_code_library={},
        max_retries=2,
        retry_threshold=0.5,
    )
    trace, _ = await layer.execute(TASK)
    assert trace.score == 0.9
    assert trace.command_count > 1               # gate: retried -> not 0
    # initial execute (0.2) + 1 retry (0.9, >= threshold so the loop breaks) = 2.
    assert trace.command_count == 2
    assert len(executor.scripts) == trace.command_count


# --------------------------------------------------------------------------- #
# Additive-field round-trip + rebuild
# --------------------------------------------------------------------------- #
def test_model_dump_roundtrip_preserves_new_fields():
    t = Trace(
        task_id="t",
        utilities_available=["a", "b"],
        utilities_called=["a"],
        utilities_call_counts={"a": 3},
        command_count=4,
        parse_failure_turns=2,
    )
    back = Trace.model_validate(json.loads(json.dumps(t.model_dump())))
    assert back.utilities_available == ["a", "b"]
    assert back.utilities_called == ["a"]
    assert back.utilities_call_counts == {"a": 3}
    assert back.command_count == 4
    assert back.parse_failure_turns == 2


def test_legacy_trace_json_defaults_the_new_fields():
    """A HEAD-era trace JSON (no new keys) validates with the documented defaults."""
    legacy = {"task_id": "t", "depth": 1, "score": 0.5, "success": True}
    t = Trace.model_validate(legacy)
    assert t.utilities_available == []
    assert t.utilities_called is None
    assert t.utilities_call_counts == {}
    assert t.command_count == 0
    assert t.parse_failure_turns == 0


def test_rebuild_from_disk_preserves_new_fields(tmp_path):
    archive_dir = tmp_path / "archive"
    cand_dir = archive_dir / "cand0"
    (cand_dir / "traces").mkdir(parents=True)
    (cand_dir / "summary.json").write_text(json.dumps({
        "candidate_id": "cand0", "parent_id": None, "iteration": 0, "depth": 1,
        "mean_score": 0.5, "pass_at_1": 1.0, "per_task_scores": {"t1": 0.5},
        "num_children": 0, "temperature_used": 0.7, "total_tokens": 1,
        "created_at": "2026-01-01T00:00:00",
    }))
    tr = Trace(
        task_id="t1", depth=1, score=0.5, success=True,
        utilities_available=["h"], utilities_called=["h"],
        utilities_call_counts={"h": 2}, command_count=1, parse_failure_turns=0,
    )
    (cand_dir / "traces" / "t1.json").write_text(json.dumps(tr.model_dump()))

    archive = Archive.rebuild_from_disk(archive_dir)
    rebuilt = archive.get("cand0").traces[0]
    assert rebuilt.utilities_available == ["h"]
    assert rebuilt.utilities_called == ["h"]
    assert rebuilt.utilities_call_counts == {"h": 2}
    assert rebuilt.command_count == 1
