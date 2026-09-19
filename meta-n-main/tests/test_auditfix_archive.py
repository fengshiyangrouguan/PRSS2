"""Regression tests for audit fixes in meta_n/core/archive.py.

Offline / LLM-free. Covers:
  - Finding 4: archive-best must track a strictly-NEGATIVE best on a
    negative-capable continuous score scale (symbolic_regression: hi=None,
    fitness goes negative) instead of being floored at the 0.0 init.

Finding 5: ``_fresh_headroom_signal`` previously hardcoded a literal 1.0 as the
absolute "score-scale max" ceiling, but ``score``/``per_task_scores`` are RAW
benchmark values (not normalized to [0,1]). On a ``continuous`` unbounded scale
(``score_scale()['hi'] is None`` — symbolic_regression / AlgoTune / AlphaEvolve)
that mis-fires: every per-task-best holder whose score happens to sit below 1.0
is reported as having headroom, so the intended "saturated -> 0.0" pruning never
engages. The fix plumbs the benchmark ceiling in via ``Archive(score_ceiling=)``
(default 1.0 ⇒ [0,1]-scale behavior byte-identical); ``score_ceiling=None`` drops
the absolute-ceiling clause so headroom relies solely on the per-task-best lag.
This signal is consulted only when ``within_task_depth_bonus > 0`` (default 0.0),
so default runs are unaffected.
"""

from meta_n.core.archive import Archive, Candidate
from meta_n.core.meta_layer import Trace


def _cand(cid: str, mean: float, task: str) -> Candidate:
    return Candidate(
        candidate_id=cid,
        mean_score=mean,
        per_task_scores={task: mean},
        traces=[Trace(task_id=task, score=mean)],
    )


def test_negative_best_is_tracked_not_floored_at_zero():
    """Finding 4: a strictly-negative best candidate must be recorded.

    On the original code (``mean_score > self._best_mean_score`` with a 0.0
    init), every candidate here has a negative mean, so the guard never fires:
    ``best_candidate`` stays None and ``best_mean_score`` reports a phantom 0.0.
    """
    arc = Archive()
    arc.add(_cand("c0", -5.0, "t0"))   # weak fit, MSE>1 -> negative fitness
    arc.add(_cand("c1", -2.0, "t0"))   # better, but still negative
    arc.add(_cand("c2", -9.0, "t0"))   # worse

    assert arc.best_candidate is not None
    assert arc.best_candidate.candidate_id == "c1"   # the largest (least-negative)
    assert arc.best_mean_score == -2.0


def test_first_finite_candidate_becomes_best_regardless_of_sign():
    """The very first finite-mean candidate is the unconditional archive-best."""
    arc = Archive()
    arc.add(_cand("only", -3.5, "t0"))
    assert arc.best_candidate is not None
    assert arc.best_candidate.candidate_id == "only"
    assert arc.best_mean_score == -3.5


def test_empty_archive_serializes_zero_best_unchanged():
    """Parity: an EMPTY archive still reports best=0.0 / id=None (to_dict
    byte-identical to before the fix)."""
    arc = Archive()
    assert arc.best_candidate is None
    assert arc.best_mean_score == 0.0
    d = arc.to_dict()
    assert d["best_mean_score"] == 0.0
    assert d["best_candidate_id"] is None


def test_unit_scale_best_tracking_unchanged():
    """Default [0,1]-scale behavior preserved: max positive mean still wins."""
    arc = Archive()
    arc.add(_cand("a", 0.2, "t0"))
    arc.add(_cand("b", 0.8, "t0"))
    arc.add(_cand("c", 0.5, "t0"))
    assert arc.best_candidate.candidate_id == "b"
    assert arc.best_mean_score == 0.8


# --------------------------------------------------------------------------- #
# Finding 5: _fresh_headroom_signal must consult the benchmark score ceiling   #
# (score_scale()['hi']), not a hardcoded 1.0. On a continuous unbounded scale  #
# (hi=None -> score_ceiling=None) the absolute-ceiling clause is dropped.       #
# --------------------------------------------------------------------------- #

def _fresh_cand(cid: str, score: float, task: str) -> Candidate:
    """A depth-1 candidate whose single trace is FRESH (trace.depth >= depth),
    so it is eligible for the headroom signal."""
    return Candidate(
        candidate_id=cid,
        depth=1,
        mean_score=score,
        per_task_scores={task: score},
        traces=[Trace(task_id=task, score=score, depth=1)],
    )


def test_headroom_default_ceiling_is_byte_identical_to_1p0():
    """Default ``score_ceiling=1.0`` reproduces the original
    ``or score < 1.0 - 1e-9`` clause exactly: a per-task-best holder sitting
    below 1.0 still reports headroom (1.0)."""
    arc = Archive(within_task_depth_bonus=0.5)  # bonus live so signal is consulted
    c = _fresh_cand("c0", 0.5, "t0")
    arc.add(c)  # c holds the per-task best at 0.5
    # lag is False (it IS the best) but 0.5 < 1.0 -> headroom under the [0,1] ceiling
    assert arc._fresh_headroom_signal(c) == 1.0


def test_headroom_default_ceiling_saturates_at_1p0():
    """Default ceiling: a per-task-best holder AT the 1.0 ceiling is saturated."""
    arc = Archive(within_task_depth_bonus=0.5)
    c = _fresh_cand("c0", 1.0, "t0")
    arc.add(c)
    assert arc._fresh_headroom_signal(c) == 0.0


def test_headroom_continuous_unbounded_drops_absolute_ceiling():
    """Finding 5: with ``score_ceiling=None`` (continuous scale, hi=None) a
    per-task-best holder at a raw score below 1.0 is SATURATED (0.0), not a
    phantom-headroom 1.0. Under the old hardcoded-1.0 code this returned 1.0
    for every below-1.0 best holder, so the saturation pruning never engaged."""
    arc = Archive(within_task_depth_bonus=0.5, score_ceiling=None)
    c = _fresh_cand("c0", 0.5, "t0")
    arc.add(c)  # c holds the per-task best at 0.5
    assert arc._fresh_headroom_signal(c) == 0.0


def test_headroom_continuous_per_task_lag_still_fires():
    """With ``score_ceiling=None`` the per-task-best lag comparison is retained:
    a candidate whose fresh task lags a deeper sibling's archived best has
    headroom (a deeper chain could still catch up), even far above 1.0."""
    arc = Archive(within_task_depth_bonus=0.5, score_ceiling=None)
    # A sibling already posted 5.0 as the per-task best for t0...
    leader = _fresh_cand("leader", 5.0, "t0")
    arc.add(leader)
    # ...so a fresh candidate at 3.0 lags it -> headroom 1.0.
    laggard = _fresh_cand("laggard", 3.0, "t0")
    arc.add(laggard)
    assert arc._fresh_headroom_signal(laggard) == 1.0


def test_headroom_default_ceiling_unaffected_above_1p0():
    """Parity guard: on the default ceiling a per-task-best holder at a raw
    score ABOVE 1.0 is saturated (0.0) under both old and new code (the literal
    1.0 clause is False and lag is False) — the fix preserves this."""
    arc = Archive(within_task_depth_bonus=0.5)  # default ceiling 1.0
    c = _fresh_cand("c0", 3.0, "t0")
    arc.add(c)
    assert arc._fresh_headroom_signal(c) == 0.0
