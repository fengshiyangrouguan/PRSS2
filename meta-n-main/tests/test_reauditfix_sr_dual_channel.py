"""Reaudit #9 dual-channel resolution: per-task-best ranks (success, score).

The openevolve fix floors hard failures to score 0.0 (reporting stays clean, no
-1e9 sentinel poisoning any averaged metric). This test pins the RANKING half of
the dual channel: the archive per-task-best must prefer a trace that RAN
(success=True) over a crashed one (success=False) regardless of score, so a valid
negative SR fit is not inverted by a 0.0-floored crash — while staying byte-
identical to plain score-max on non-negative [0,1] scales.
"""

from __future__ import annotations

from meta_n.core.archive import Archive, Candidate
from meta_n.core.meta_layer import Trace


def _cand(cid: str, task: str, score: float, success: bool) -> Candidate:
    return Candidate(
        candidate_id=cid,
        depth=1,
        mean_score=score,
        per_task_scores={task: score},
        traces=[Trace(task_id=task, depth=1, score=score, success=success)],
    )


def test_sr_valid_negative_fit_beats_crash_in_per_task_best():
    # Negative-capable scale: a valid fit that RAN (success=True, -0.70) must be
    # the per-task-best over a 0.0-floored crash (success=False), in EITHER
    # insertion order (a plain score-max would wrongly pick the 0.0 crash).
    for order in (["crash", "valid"], ["valid", "crash"]):
        arc = Archive()
        cands = {
            "crash": _cand("crash", "t", 0.0, False),
            "valid": _cand("valid", "t", -0.70, True),
        }
        for k in order:
            arc.add(cands[k])
        best = arc.per_task_best_scores()
        assert best["t"] == -0.70, f"order={order}: {best}"
        assert arc.per_task_best_traces()["t"].success is True


def test_unit_scale_per_task_best_is_score_max():
    # [0,1] scale: failures floor to 0.0 (success=False), successes score >= 0
    # (success=True), so per-task-best == max score (byte-identical to score-max).
    arc = Archive()
    arc.add(_cand("a", "t", 0.3, True))
    arc.add(_cand("b", "t", 0.0, False))  # a floored failure never wins
    arc.add(_cand("c", "t", 0.7, True))
    assert arc.per_task_best_scores()["t"] == 0.7


def test_per_task_best_never_holds_sentinel():
    # No -1e9 sentinel in the reported per-task-best: a crash contributes a
    # bounded 0.0 (openevolve floors it), so oracle_mean can never be poisoned.
    arc = Archive()
    arc.add(_cand("crash", "t", 0.0, False))
    assert arc.per_task_best_scores()["t"] == 0.0
