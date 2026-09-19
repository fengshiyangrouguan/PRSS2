"""Refinement-wave regression tests for meta_n/core/archive.py public accessors.

Covers:
  F050 — public accessors replacing orchestrator private reach-ins:
         ``find`` (None-safe lookup), ``per_task_best_sources``,
         ``frozen_score_range`` property round-trip, ``base_floor_snapshot``,
         and byte-identity of the oracle_summary per-task map built from the
         accessors vs the old direct index walk.
  F055 — ``Archive.ucb_exploration_bonus`` is the single UCB source for
         ``_selection_weights`` AND the orchestrator's parent-selection
         diagnostic; frozen-vector test pins the pre-extraction weights.

(Do not confuse with tests/test_refine_archive_verified.py, which covers the
archive/self_repair/verified cluster.) No LLM / Docker / network.
"""

import json
import math

import pytest

from meta_n.core.archive import Archive, Candidate
from meta_n.core.meta_layer import Trace


def _cand(cid, mean, tasks=None, parent_id=None, children=0):
    """Candidate with one successful trace per {task_id: score} entry."""
    tasks = tasks or {}
    return Candidate(
        candidate_id=cid,
        parent_id=parent_id,
        mean_score=mean,
        num_children=children,
        per_task_scores=dict(tasks),
        traces=[
            Trace(task_id=tid, script=f"echo {tid}", success=True, score=s)
            for tid, s in tasks.items()
        ],
    )


def _populated_archive() -> Archive:
    arch = Archive(novelty_alpha=0.3)
    arch.add(_cand("gen0_seed", 0.35, {"t1": 0.5, "t2": 0.2}))
    arch.add(_cand("gen1_b0_k0", 0.55, {"t1": 0.9, "t2": 0.2}, parent_id="gen0_seed"))
    return arch


# --------------------------------------------------------------------------- #
# F050 — find() (None-safe lookup)
# --------------------------------------------------------------------------- #


class TestFind:
    def test_returns_candidate_for_known_id(self):
        arch = _populated_archive()
        assert arch.find("gen0_seed") is arch.get("gen0_seed")

    def test_returns_none_for_missing_id(self):
        arch = _populated_archive()
        assert arch.find("no_such_candidate") is None

    def test_returns_none_for_none_id(self):
        # Seed parent_id is None — lineage walks need no pre-check.
        arch = _populated_archive()
        assert arch.find(None) is None

    def test_get_still_raises_for_missing_id(self):
        # find() is the None-returning sibling; get() keeps its KeyError contract.
        arch = _populated_archive()
        with pytest.raises(KeyError):
            arch.get("no_such_candidate")


# --------------------------------------------------------------------------- #
# F050 — per_task_best_sources()
# --------------------------------------------------------------------------- #


class TestPerTaskBestSources:
    def test_matches_index_and_scores_siblings(self):
        arch = _populated_archive()
        srcs = arch.per_task_best_sources()
        assert srcs == {"t1": "gen1_b0_k0", "t2": "gen0_seed"}
        # Same keys AND iteration order as the score/trace siblings.
        assert list(srcs) == list(arch.per_task_best_scores())
        assert list(srcs) == list(arch.per_task_best_traces())

    def test_empty_archive_yields_empty_dict(self):
        assert Archive().per_task_best_sources() == {}

    def test_oracle_summary_map_byte_identical_to_index_walk(self):
        # F050 byte-identity gate: the oracle_summary per-task map built from
        # per_task_best_scores() + per_task_best_sources() must serialize to
        # the same bytes as the old direct walk over the private index.
        arch = _populated_archive()
        old = {
            tid: {"score": entry[0], "candidate_id": entry[1]}
            for tid, entry in arch._best_per_task.items()
        }
        scores = arch.per_task_best_scores()
        srcs = arch.per_task_best_sources()
        new = {
            tid: {"score": scores[tid], "candidate_id": srcs[tid]}
            for tid in srcs
        }
        assert json.dumps(new, indent=2) == json.dumps(old, indent=2)


# --------------------------------------------------------------------------- #
# F050 — frozen_score_range property
# --------------------------------------------------------------------------- #


class TestFrozenScoreRange:
    def test_none_until_frozen(self):
        assert Archive().frozen_score_range is None

    def test_setter_round_trips_verbatim(self):
        # Checkpoint restore writes the persisted value back with no coercion.
        arch = Archive()
        arch.frozen_score_range = 2.5
        assert arch.frozen_score_range == 2.5
        assert arch.score_range() == 2.5  # consumed by the normalizer
        arch.frozen_score_range = None
        assert arch.frozen_score_range is None

    def test_freeze_score_range_visible_through_property(self):
        arch = _populated_archive()
        frozen = arch.freeze_score_range()
        assert arch.frozen_score_range == frozen


# --------------------------------------------------------------------------- #
# F050 — base_floor_snapshot()
# --------------------------------------------------------------------------- #


class TestBaseFloorSnapshot:
    def test_empty_without_regression_guard(self):
        assert _populated_archive().base_floor_snapshot() == {}

    def test_snapshot_matches_floor_and_is_a_copy(self):
        arch = Archive(regression_guard=True)
        arch.add(_cand("gen0_seed", 0.3, {"t1": 0.3}))
        floor_trace = Trace(task_id="t1", script="echo floor", success=True, score=0.4)
        arch.set_base_floor({"t1": 0.4}, {"t1": floor_trace})

        snap = arch.base_floor_snapshot()
        assert snap == {"t1": (0.4, "gen0_seed", floor_trace)}

        # Mutating the snapshot must not leak into the archive.
        snap["t1"] = (0.0, "hacked", None)
        snap["t2"] = (9.9, "hacked", None)
        assert arch.base_floor_snapshot() == {"t1": (0.4, "gen0_seed", floor_trace)}


# --------------------------------------------------------------------------- #
# F055 — ucb_exploration_bonus (single UCB source, no drift)
# --------------------------------------------------------------------------- #


class TestUcbExplorationBonus:
    @pytest.mark.parametrize("alpha", [0.0, 0.3, 1.7])
    @pytest.mark.parametrize("pool_size", [2, 3, 5, 40])
    @pytest.mark.parametrize("children", [0, 1, 7])
    def test_matches_inline_formula(self, alpha, pool_size, children):
        # Exact equality: identical float ops/order as the retired inline
        # formulas in _selection_weights and the orchestrator diagnostic.
        expected = alpha * math.sqrt(math.log(pool_size) / (1 + children))
        assert Archive.ucb_exploration_bonus(alpha, pool_size, children) == expected

    @pytest.mark.parametrize("pool_size", [1, 0, -3])
    def test_zero_for_degenerate_pool(self, pool_size):
        # n <= 1 ⇒ 0.0 (the old `ln_n > 0.0` / `pool_n > 1` guards).
        assert Archive.ucb_exploration_bonus(0.3, pool_size, 0) == 0.0


class TestSelectionWeightsFrozen:
    def test_selection_weights_unchanged_after_extraction(self):
        # Frozen vectors captured from _selection_weights BEFORE the F055
        # extraction (same candidates, novelty_alpha=0.3). Exact equality:
        # selection weights are research-critical.
        arch = Archive(novelty_alpha=0.3)
        cands = [
            Candidate(candidate_id="a", mean_score=0.20, num_children=0),
            Candidate(candidate_id="b", mean_score=0.50, num_children=1),
            Candidate(candidate_id="c", mean_score=0.50, num_children=2),
            Candidate(candidate_id="d", mean_score=0.90, num_children=0),
            Candidate(candidate_id="e", mean_score=float("nan"), num_children=3),
        ]
        assert arch._selection_weights(cands) == [
            0.6305908723538558,
            0.7691183866991151,
            0.7197342426046132,
            1.3805908723538558,
            0.19029543617692793,
        ]
        # Singleton pool: rank_norm 0.0 + ucb 0.0 → floored at 1e-06.
        assert arch._selection_weights([cands[0]]) == [1e-06]
        assert arch._selection_weights(cands[:2]) == [
            0.2497663833473093,
            1.1766115033773212,
        ]
