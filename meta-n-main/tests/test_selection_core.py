"""Selection-core tests (Step 1): anti-regression pool (6.1), elitism +
scale-invariant UCB selection (2.1), temperature decouple (2.4), sentinel
clamp (N4c)."""

import random

from types import SimpleNamespace

from meta_n.core.archive import Archive, Candidate
from meta_n.core.evolutionary_orchestrator import EvolutionaryOrchestrator
from meta_n.integrations.openevolve import _coerce_eval_score


# --------------------------------------------------------------------------- #
# 6.1 — breedable pool (anti-regression, archive stays unconditional)
# --------------------------------------------------------------------------- #

def test_breedable_pool_excludes_regressor(make_candidate):
    arch = Archive()
    arch.add(make_candidate("gen0_seed", mean_score=0.8, depth=1))
    arch.add(make_candidate("regr", mean_score=0.3, parent_id="gen0_seed", depth=2,
                            per_task_scores={"task_a": 0.3, "task_b": 0.3}))
    arch.add(make_candidate("impr", mean_score=0.85, parent_id="gen0_seed", depth=2,
                            per_task_scores={"task_a": 0.85, "task_b": 0.85}))
    ids = {c.candidate_id for c in arch.breedable_pool(max_depth=3)}
    assert "regr" not in ids        # pure regressor excluded from PARENT pool
    assert "gen0_seed" in ids       # seed always breedable
    assert "impr" in ids            # improver (also the elite) breedable
    assert len(arch) == 3           # archive.add stays UNCONDITIONAL


def test_breedable_pool_keeps_disjoint_winner(make_candidate, make_trace):
    arch = Archive()
    arch.add(make_candidate("gen0_seed", mean_score=0.6, tasks=["a", "b"],
                            per_task_scores={"a": 0.6, "b": 0.6}, depth=1))
    # Regresses on MEAN (0.45 < 0.6) but WINS task b (0.9) → oracle-thesis: keep.
    arch.add(make_candidate(
        "disj", mean_score=0.45, parent_id="gen0_seed", depth=2,
        per_task_scores={"a": 0.0, "b": 0.9},
        traces=[make_trace("a", success=False, score=0.0), make_trace("b", score=0.9)],
    ))
    assert "disj" in {c.candidate_id for c in arch.breedable_pool(max_depth=3)}


def test_breedable_pool_keeps_equal_mean_tie(make_candidate):
    arch = Archive()
    arch.add(make_candidate("gen0_seed", mean_score=0.5, depth=1))
    arch.add(make_candidate("tie", mean_score=0.5, parent_id="gen0_seed", depth=2))
    assert "tie" in {c.candidate_id for c in arch.breedable_pool(max_depth=3)}


def test_breedable_pool_fallback_never_empty(make_candidate):
    arch = Archive()
    arch.add(make_candidate("gen0_seed", mean_score=0.8, depth=1))
    for i in range(3):
        arch.add(make_candidate(f"r{i}", mean_score=0.1, parent_id="gen0_seed", depth=2,
                                per_task_scores={"task_a": 0.1, "task_b": 0.1}))
    pool = arch.breedable_pool(max_depth=3)
    assert pool                                   # never empty
    assert "gen0_seed" in {c.candidate_id for c in pool}


def test_breedable_pool_empty_when_all_at_max_depth(make_candidate):
    arch = Archive()
    arch.add(make_candidate("gen0_seed", mean_score=0.8, depth=3))
    assert arch.breedable_pool(max_depth=3) == []  # nothing extendable


# --------------------------------------------------------------------------- #
# 2.1 — elitism + scale-invariant UCB
# --------------------------------------------------------------------------- #

def _ratio_hi_over_lo(arch, n_draws=1000):
    rng = random.Random(42)
    counts = {"lo": 0, "hi": 0}
    for _ in range(n_draws):
        counts[arch.select_parents(1, rng=rng)[0].candidate_id] += 1
    return counts["hi"] / max(1, counts["lo"])


def test_selection_is_scale_invariant():
    unit = Archive()
    unit.add(Candidate(candidate_id="lo", mean_score=0.5, iteration=0))
    unit.add(Candidate(candidate_id="hi", mean_score=0.9, iteration=1))
    cont = Archive()
    cont.add(Candidate(candidate_id="lo", mean_score=29.0, iteration=0))
    cont.add(Candidate(candidate_id="hi", mean_score=38.0, iteration=1))
    r_unit, r_cont = _ratio_hi_over_lo(unit), _ratio_hi_over_lo(cont)
    # rank-normalization makes the selection ratio identical across scales;
    # the retired additive bonus made the continuous case nearly deterministic.
    assert 0.5 < r_unit / r_cont < 2.0


def test_elitism_reserves_best_even_when_explored(make_candidate):
    arch = Archive()
    best = make_candidate("best", mean_score=0.9)
    other = make_candidate("other", mean_score=0.4)
    arch.add(best)
    arch.add(other)
    rng = random.Random(0)
    for _ in range(20):
        best.num_children = 100  # heavily explored → tiny UCB, but reserved anyway
        parents = arch.select_parents(2, rng=rng)
        assert "best" in {p.candidate_id for p in parents}


def test_selection_degenerate_no_divide_by_zero(make_candidate):
    one = Archive()
    one.add(make_candidate("only", mean_score=0.5))
    assert one.select_parents(1)[0].candidate_id == "only"
    assert len(one.select_parents(3)) == 3
    equal = Archive()
    for i in range(3):
        equal.add(make_candidate(f"c{i}", mean_score=0.5))
    assert len(equal.select_parents(2)) == 2


def test_selection_is_deterministic(make_candidate):
    def build():
        a = Archive()
        for i, m in enumerate([0.2, 0.5, 0.8, 0.5]):
            a.add(make_candidate(f"c{i}", mean_score=m))
        return a
    s1 = [p.candidate_id for p in build().select_parents(3, rng=random.Random(7))]
    s2 = [p.candidate_id for p in build().select_parents(3, rng=random.Random(7))]
    assert s1 == s2


# --------------------------------------------------------------------------- #
# 2.4 — temperature decoupled from k
# --------------------------------------------------------------------------- #

def test_temperature_decoupled_from_k():
    stub = SimpleNamespace(config=SimpleNamespace(temperatures=[0.5, 0.7, 0.9]))
    seen = set()
    for iteration in range(1, 4):
        for k in range(2):  # K=2 beam
            seen.add(EvolutionaryOrchestrator._select_temperature(stub, k, iteration))
    assert seen == {0.5, 0.7, 0.9}  # 0.9 now reachable at K=2 (was pinned to temps[0])


# --------------------------------------------------------------------------- #
# N4c — failure-sentinel clamp
# --------------------------------------------------------------------------- #

def test_coerce_eval_score_clamps_sentinel():
    assert _coerce_eval_score(0.9) == (0.9, None)
    assert _coerce_eval_score(-1e9) == (0.0, -1e9)   # sentinel neutralized, raw kept
    assert _coerce_eval_score(None) == (0.0, None)
    assert _coerce_eval_score(float("nan")) == (0.0, None)
    assert _coerce_eval_score("bad") == (0.0, None)


def test_sentinel_does_not_poison_mean():
    scores = [0.9, 0.9, _coerce_eval_score(-1e9)[0]]
    assert abs(sum(scores) / len(scores) - 0.6) < 1e-9  # 0.6, not -3.3e8
