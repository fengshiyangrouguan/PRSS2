"""G6 — archive-integrity fixes from the silent-void audit.

T3.9: ``rebuild_from_disk`` cross-checks ``len(injected_codes) == depth - 1`` and
      skips a candidate whose injected_code sidecars were partially written
      (crash between summary.json and the sidecars).
T3.10: ``set_base_floor`` only RAISES the per-task-best index off a floor entry
      whose trace is routable (non-None with a script); a None/empty-script floor
      trace still seeds ``_base_floor`` (regression clamp) but is never indexed as
      a per-task winner.

Each test pins the NEW (fixed) behavior AND that the default/clean path is
unchanged.
"""

import json
import logging

from meta_n.core.archive import Archive
from meta_n.core.meta_layer import Trace


# --------------------------------------------------------------------------- #
# T3.10 — set_base_floor routability guard                                    #
# --------------------------------------------------------------------------- #
def _trace(task_id, score, script="def solve(**kwargs): pass"):
    return Trace(task_id=task_id, score=score, success=True, script=script)


def test_floor_with_real_script_raises_per_task_best():
    """DEFAULT regression_guard path is unchanged: a floor trace carrying a
    script raises the per-task-best index exactly as before."""
    arch = Archive(regression_guard=True)
    arch.set_base_floor({"t": 0.5}, {"t": _trace("t", 0.5)})
    assert arch.best_score_for_task("t") == 0.5
    assert "t" in arch._base_floor


def test_floor_with_none_trace_seeds_floor_but_not_per_task_best():
    """T3.10 — a None floor trace is unroutable: it seeds _base_floor (so the
    regression clamp still fires) but must NOT raise per-task-best on score
    alone."""
    arch = Archive(regression_guard=True)
    arch.set_base_floor({"t": 0.9}, {"t": None})
    # NOT indexed as a per-task winner (no script to deploy)...
    assert arch.best_score_for_task("t") is None
    # ...but still recorded as the base floor for the regression clamp.
    assert arch._base_floor["t"][0] == 0.9
    assert arch._base_floor["t"][2] is None


def test_floor_with_empty_script_trace_not_indexed():
    """T3.10 — an empty-script floor trace is also unroutable."""
    arch = Archive(regression_guard=True)
    arch.set_base_floor({"t": 0.9}, {"t": _trace("t", 0.9, script="")})
    assert arch.best_score_for_task("t") is None
    assert arch._base_floor["t"][0] == 0.9


def test_floor_regression_clamp_still_uses_none_floor():
    """The None/empty-script floor must still clamp a later sub-floor Ω trace
    out of per-task-best (the _base_floor entry is consulted in add())."""
    arch = Archive(regression_guard=True)
    arch.set_base_floor({"t": 0.9}, {"t": None})
    # An Ω candidate below the floor must be barred from per-task-best.
    below = Trace(task_id="t", score=0.4, success=True, script="x=1")
    from meta_n.core.archive import Candidate

    cand = Candidate(candidate_id="gen1_b0_k0", depth=2, traces=[below],
                     mean_score=0.4)
    arch.add(cand)
    assert arch.best_score_for_task("t") is None  # clamped out, never shipped


# --------------------------------------------------------------------------- #
# T3.9 — rebuild_from_disk partial-write cross-check                          #
# --------------------------------------------------------------------------- #
def test_rebuild_skips_truncated_injected_code_chain(
    tmp_path, make_candidate, write_candidate_dir, caplog
):
    """T3.9 — a depth=2 candidate whose only injected_code sidecar is missing
    (partial write) must be skipped, not loaded with a full-chain mean_score
    over a truncated solver."""
    from meta_n.core.meta_layer import InjectedCode

    archive_dir = tmp_path / "archive"
    seed = make_candidate("gen0_seed", mean_score=0.5, iteration=0)
    write_candidate_dir(archive_dir, seed)

    child = make_candidate(
        "gen1_b0_k0", mean_score=0.99, parent_id="gen0_seed", iteration=1,
        depth=2, injected_codes=[InjectedCode(pre_process="x = 1")],
    )
    write_candidate_dir(archive_dir, child)
    # Simulate a crash between summary.json (written first) and the sidecars
    # (written last): drop the depth-2 sidecar while summary.json claims depth=2.
    (archive_dir / "gen1_b0_k0" / "injected_code_d2.json").unlink()

    with caplog.at_level(logging.WARNING):
        rebuilt = Archive.rebuild_from_disk(archive_dir)

    assert "gen1_b0_k0" not in rebuilt._by_id  # truncated candidate skipped
    assert "gen0_seed" in rebuilt._by_id  # clean seed loads
    assert any("injected_code integrity" in r.message for r in caplog.records)
    # The corrupted full-chain score never seeds archive-best.
    assert rebuilt.best_mean_score == 0.5


def test_rebuild_loads_clean_chain_unchanged(
    tmp_path, make_candidate, write_candidate_dir
):
    """DEFAULT path: a complete depth=2 candidate (all sidecars present) loads
    exactly as before — the cross-check is a no-op on a clean write."""
    from meta_n.core.meta_layer import InjectedCode

    archive_dir = tmp_path / "archive"
    seed = make_candidate("gen0_seed", mean_score=0.5, iteration=0)
    write_candidate_dir(archive_dir, seed)
    child = make_candidate(
        "gen1_b0_k0", mean_score=0.99, parent_id="gen0_seed", iteration=1,
        depth=2, injected_codes=[InjectedCode(pre_process="x = 1")],
    )
    write_candidate_dir(archive_dir, child)

    rebuilt = Archive.rebuild_from_disk(archive_dir)
    assert "gen1_b0_k0" in rebuilt._by_id
    assert rebuilt.get("gen1_b0_k0").depth == 2
    assert len(rebuilt.get("gen1_b0_k0").injected_codes) == 1
    assert rebuilt.best_mean_score == 0.99
