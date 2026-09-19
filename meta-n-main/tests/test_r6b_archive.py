"""Refine §6b archive-selection regressions (F003, F006).

F003 — ``Archive.get_inspiration_traces`` no longer floors a MISSING parent
per-task score at a phantom 0.0 for the inclusion test: a task absent from
``parent.per_task_scores`` (the parent never produced a finite score there —
crashed / non-finite) now admits a NEGATIVE archive best, while keeping the
0.0 crash-floor baseline for gap ordering (byte-identical on non-negative
scales).

F006 — ``Archive.select_parents`` grows an ``elite_rotation: int = 0``
offset that, when nonzero AND the reserved window truncates the elite list
(``len(elite) > n - 1``), pins the archive-best at slot 0 and rotates the
per-task-winner tail so every elite gets a reserved slot within ``len(tail)``
consecutive generations. Default 0 = OFF = byte-identical reservation
(including the rng stream).
"""

import math
import random

from meta_n.core.archive import Archive


# --------------------------------------------------------------------------- #
# F003 — inspiration crash-floor on negative scales
# --------------------------------------------------------------------------- #

class TestF003InspirationCrashFloor:

    def test_negative_best_inspires_crashed_parent(
        self, make_candidate, make_trace, make_tasks
    ):
        """The repro: parent crashed on "t" (no per_task_scores key); another
        candidate holds a NEGATIVE per-task best (success=True, -0.5) — the
        archive's knowledge must be offered, not suppressed by a 0.0 floor."""
        arch = Archive()
        arch.add(make_candidate(
            "winner", mean_score=-0.5, tasks=["t"],
            per_task_scores={"t": -0.5},
            traces=[make_trace("t", success=True, score=-0.5)],
        ))
        parent = make_candidate("parent", per_task_scores={}, traces=[])

        traces = arch.get_inspiration_traces(parent, make_tasks(["t"]))
        assert len(traces) == 1
        assert traces[0].task_id == "t"
        assert traces[0].score == -0.5

    def test_missing_parent_nonnegative_scale_byte_identical(
        self, make_candidate, make_trace, make_tasks
    ):
        """Missing key on a non-negative scale keeps HEAD behavior: a positive
        best is included with gap == best (0.0 baseline), a 0.0 best is
        excluded (``best != 0.0`` <=> ``best > 0.0`` when scores >= 0)."""
        arch = Archive()
        # t1: parent missing, best 0.7 → gap 0.7. t2: parent 0.5, best 0.9 →
        # gap 0.4. Ordering [t1, t2] pins gap == best for the missing key.
        arch.add(make_candidate(
            "winner", mean_score=0.8, tasks=["t1", "t2"],
            per_task_scores={"t1": 0.7, "t2": 0.9},
            traces=[make_trace("t1", score=0.7), make_trace("t2", score=0.9)],
        ))
        # z: best exactly 0.0 → excluded (no signal above the crash floor).
        arch.add(make_candidate(
            "zero", mean_score=0.0, tasks=["z"],
            per_task_scores={"z": 0.0},
            traces=[make_trace("z", success=True, score=0.0)],
        ))
        parent = make_candidate(
            "parent", per_task_scores={"t2": 0.5}, traces=[]
        )

        traces = arch.get_inspiration_traces(parent, make_tasks(["t1", "t2", "z"]))
        assert [t.task_id for t in traces] == ["t1", "t2"]  # z excluded

    def test_parent_finite_score_branch_unchanged(
        self, make_candidate, make_trace, make_tasks
    ):
        """A present finite parent score keeps the strict ``best > parent``
        inclusion test bit-for-bit."""
        arch = Archive()
        arch.add(make_candidate(
            "winner", mean_score=0.6, tasks=["t"],
            per_task_scores={"t": 0.6},
            traces=[make_trace("t", score=0.6)],
        ))
        tasks = make_tasks(["t"])

        behind = make_candidate("behind", per_task_scores={"t": 0.4}, traces=[])
        assert len(arch.get_inspiration_traces(behind, tasks)) == 1

        tied = make_candidate("tied", per_task_scores={"t": 0.6}, traces=[])
        assert arch.get_inspiration_traces(tied, tasks) == []

    def test_negative_gap_ranks_after_positive_gaps(
        self, make_candidate, make_trace, make_tasks
    ):
        """A crashed task with a negative best (gap == best < 0) sorts AFTER
        every positive gap, and is the first dropped under max_traces."""
        arch = Archive()
        arch.add(make_candidate(
            "w_neg", mean_score=-0.5, tasks=["neg"],
            per_task_scores={"neg": -0.5},
            traces=[make_trace("neg", success=True, score=-0.5)],
        ))
        arch.add(make_candidate(
            "w_pos", mean_score=0.9, tasks=["pos"],
            per_task_scores={"pos": 0.9},
            traces=[make_trace("pos", score=0.9)],
        ))
        parent = make_candidate(
            "parent", per_task_scores={"pos": 0.1}, traces=[]
        )  # "neg" missing → crashed; "pos" gap = 0.8

        tasks = make_tasks(["neg", "pos"])
        assert [t.task_id for t in arch.get_inspiration_traces(parent, tasks)] \
            == ["pos", "neg"]
        # max_traces truncation drops the negative-gap entry first.
        assert [t.task_id for t in
                arch.get_inspiration_traces(parent, tasks, max_traces=1)] == ["pos"]

    def test_nonfinite_parent_score_treated_as_missing(
        self, make_candidate, make_trace, make_tasks
    ):
        """Deliberate corner pin: a hand-built NaN parent score is the same
        never-scored state as a missing key (was silently excluded at HEAD)."""
        arch = Archive()
        arch.add(make_candidate(
            "winner", mean_score=-0.5, tasks=["t"],
            per_task_scores={"t": -0.5},
            traces=[make_trace("t", success=True, score=-0.5)],
        ))
        parent = make_candidate(
            "parent", per_task_scores={"t": float("nan")}, traces=[]
        )
        assert not math.isfinite(parent.per_task_scores["t"])

        traces = arch.get_inspiration_traces(parent, make_tasks(["t"]))
        assert [t.task_id for t in traces] == ["t"]

    def test_self_best_still_excluded(
        self, make_candidate, make_trace, make_tasks
    ):
        """The parent holding the per-task best on a key MISSING from its own
        per_task_scores is still excluded by the best_cand_id guard."""
        arch = Archive()
        parent = make_candidate(
            "parent", mean_score=-0.5, per_task_scores={},
            traces=[make_trace("t", success=True, score=-0.5)],
        )
        arch.add(parent)  # add() indexes per-task-best from TRACES → parent wins "t"
        assert arch.per_task_best_sources() == {"t": "parent"}

        assert arch.get_inspiration_traces(parent, make_tasks(["t"])) == []


# --------------------------------------------------------------------------- #
# F006 — elite-window rotation (module side; default OFF byte-identical)
# --------------------------------------------------------------------------- #

def _starved_archive(make_candidate, make_trace):
    """The F006 repro fixture: archive-best + 5 DISTINCT per-task winners.

    ``arch_best`` has the highest mean (0.95) but never wins a task; ``w_<x>``
    wins ``t_<x>`` at 0.9. ``_elite_ids()`` is therefore
    ``["arch_best", "w_a", ..., "w_e"]`` — with n=3 the HEAD window reserves
    only ``[arch_best, w_a]`` every generation.
    """
    arch = Archive()
    tasks = [f"t_{x}" for x in "abcde"]
    arch.add(make_candidate(
        "arch_best", mean_score=0.95, tasks=tasks,
        per_task_scores={t: 0.5 for t in tasks},
        traces=[make_trace(t, score=0.5) for t in tasks],
    ))
    for x in "abcde":
        winner = f"t_{x}"
        scores = {t: (0.9 if t == winner else 0.1) for t in tasks}
        arch.add(make_candidate(
            f"w_{x}", mean_score=sum(scores.values()) / len(scores),
            tasks=tasks, per_task_scores=scores,
            traces=[make_trace(t, score=s) for t, s in scores.items()],
        ))
    assert arch._elite_ids() == ["arch_best", "w_a", "w_b", "w_c", "w_d", "w_e"]
    return arch


class TestF006EliteRotation:

    def test_rotation_off_byte_identical_including_rng_stream(
        self, make_candidate, make_trace
    ):
        """The kwarg default (0) is inert: identical id lists AND identical
        post-call rng state (the rotation consumes no rng draws) — plus a
        frozen-literal pin of the OFF-path selection itself."""
        arch = _starved_archive(make_candidate, make_trace)
        # Frozen pre-6b behavior pin (seed 0): the OFF-path select_parents id
        # sequence, verified to match real HEAD before the F006 change landed.
        # Kwarg-equivalence alone would not catch OFF-path drift (both sides
        # run the same implementation).
        assert [
            c.candidate_id for c in arch.select_parents(3, rng=random.Random(0))
        ] == ["arch_best", "w_a", "w_d"]
        for seed in range(20):
            rng_head = random.Random(seed)
            rng_new = random.Random(seed)
            head = [c.candidate_id for c in arch.select_parents(3, rng=rng_head)]
            new = [c.candidate_id
                   for c in arch.select_parents(3, rng=rng_new, elite_rotation=0)]
            assert head == new
            assert rng_head.random() == rng_new.random()  # same stream position

    def test_default_reservation_is_alphabetical_first(
        self, make_candidate, make_trace
    ):
        """Honest pin of the OFF behavior (the F006 repro): over 50 generations
        the reserved slots are ALWAYS {arch_best, w_a}; w_b..w_e never get one."""
        arch = _starved_archive(make_candidate, make_trace)
        rng = random.Random(0)
        reserved_union = set()
        for _ in range(50):
            parents = arch.select_parents(3, rng=rng)
            reserved_union.update(p.candidate_id for p in parents[:2])
        assert reserved_union == {"arch_best", "w_a"}

    def test_rotation_covers_all_elites_within_tail_len_generations(
        self, make_candidate, make_trace
    ):
        """Flag-ON semantics: archive-best pinned at slot 0 every generation,
        and all 5 tail winners reserved within len(tail)=5 consecutive
        generations (coverage bound with window w=1)."""
        arch = _starved_archive(make_candidate, make_trace)
        rng = random.Random(0)
        rotated_slot = []
        for gen in range(1, 6):  # 1-based generation index, as plumbed
            parents = arch.select_parents(3, rng=rng, elite_rotation=gen)
            assert parents[0].candidate_id == "arch_best"
            rotated_slot.append(parents[1].candidate_id)
        assert set(rotated_slot) == {"w_a", "w_b", "w_c", "w_d", "w_e"}

    def test_rotation_noop_when_no_truncation(self, make_candidate):
        """When all elites fit in the window (len(elite) <= n-1) a nonzero
        offset is identical to OFF — the branch is gated on truncation biting."""
        arch = Archive()
        arch.add(make_candidate("best", mean_score=0.9))   # archive-best AND
        arch.add(make_candidate("other", mean_score=0.4))  # all-task winner
        assert arch._elite_ids() == ["best"]
        for seed in range(10):
            off = [c.candidate_id for c in
                   arch.select_parents(3, rng=random.Random(seed))]
            on = [c.candidate_id for c in
                  arch.select_parents(3, rng=random.Random(seed), elite_rotation=7)]
            assert off == on

    def test_rotation_with_best_absent_from_pool(self, make_candidate, make_trace):
        """Pool excludes the archive-best: the intersected elite list rotates
        without a pin (no KeyError); all pool elites reserved over len(list)
        generations."""
        arch = _starved_archive(make_candidate, make_trace)
        pool = [arch.get(f"w_{x}") for x in "abcde"]  # no arch_best
        rng = random.Random(0)
        reserved_union = set()
        for gen in range(1, 6):
            parents = arch.select_parents(3, rng=rng, pool=pool, elite_rotation=gen)
            ids = [p.candidate_id for p in parents]
            assert "arch_best" not in ids
            reserved_union.update(ids[:2])
        assert reserved_union == {"w_a", "w_b", "w_c", "w_d", "w_e"}

    def test_rotation_at_beam_width_2_pins_best_every_gen(
        self, make_candidate, make_trace
    ):
        """At n == 2 the sole reserved slot (reserved[:1]) holds the pinned
        archive-best every generation — the per-task-winner tail is never
        rotated into a reserved slot, so coverage does NOT hold here (the
        corrected documented scope). ON is byte-identical to OFF at n == 2."""
        arch = _starved_archive(make_candidate, make_trace)
        reserved_union = set()
        for gen in range(1, 7):
            parents = arch.select_parents(2, rng=random.Random(0), elite_rotation=gen)
            assert parents[0].candidate_id == "arch_best"
            reserved_union.add(parents[0].candidate_id)
        # Only the pinned best is ever reserved; the tail never gets a slot.
        assert reserved_union == {"arch_best"}
        # ON == OFF byte-identity at n == 2 (id sequence AND rng stream position).
        for seed in range(6):
            r_off, r_on = random.Random(seed), random.Random(seed)
            assert (
                [c.candidate_id
                 for c in arch.select_parents(2, rng=r_off, elite_rotation=0)]
                == [c.candidate_id
                    for c in arch.select_parents(2, rng=r_on, elite_rotation=seed + 1)]
            )
            assert r_off.random() == r_on.random()
