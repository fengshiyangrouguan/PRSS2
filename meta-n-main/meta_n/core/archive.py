"""Archive and Candidate data structures for evolutionary search."""

from __future__ import annotations

import json
import logging
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace
from meta_n.core.self_repair import SelfRepairEvent

logger = logging.getLogger(__name__)


class Candidate(BaseModel):
    """A complete solver chain in the evolutionary archive.

    The chain is defined by `injected_codes`: to reconstruct the solver,
    start with Layer1Solver, then wrap MetaLayer(depth=2, code=injected_codes[0]),
    MetaLayer(depth=3, code=injected_codes[1]), etc.
    """

    candidate_id: str
    parent_id: Optional[str] = None
    iteration: int = 0
    depth: int = 1  # 1 + len(injected_codes)

    # Chain definition
    injected_codes: list[InjectedCode] = Field(default_factory=list)

    # Evaluation results
    traces: list[Trace] = Field(default_factory=list)
    pass_at_1: float = 0.0
    mean_score: float = 0.0
    per_task_scores: dict[str, float] = Field(default_factory=dict)

    # Metadata
    num_children: int = 0
    total_tokens: int = 0  # outer-LLM tokens (Omega + solver code generation)
    inner_tokens: int = 0  # inner-LLM tokens (evolved solve() calling llm())
    inner_prompt_tokens: int = 0
    inner_completion_tokens: int = 0
    inner_calls: int = 0
    temperature_used: float = 0.7
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # --- Self-repair provenance (Stage 2 refine / Stage 3 re-propagation) ---
    # NON-behavioral substrate. EMPTY for every candidate the current
    # orchestration produces (gen0 + ordinary breeding), so the candidate
    # sidecar writer emits NOTHING and ``_save_candidate_incremental`` stays
    # byte-for-byte identical (the no-event guard mirrors the existing
    # ``if rollup is not None`` rollup guard). Only an opted-in within-layer
    # refine / downward re-propagation that actually repaired a buggy injection
    # appends a :class:`SelfRepairEvent` here. Monotonic: a repair builds a NEW
    # candidate (never mutates the parent), and the event is its provenance.
    self_repair_events: list[SelfRepairEvent] = Field(default_factory=list)


class Archive:
    """Monotonically growing collection of evaluated candidates.

    Candidates are never removed. The archive tracks per-task best scores
    and supports weighted parent selection (fitness + exploration bonus).
    """

    def __init__(
        self,
        novelty_alpha: float = 0.3,
        regression_guard: bool = False,
        within_task_depth_bonus: float = 0.0,
        score_ceiling: Optional[float] = 1.0,
    ):
        self.candidates: list[Candidate] = []
        self._by_id: dict[str, Candidate] = {}
        self._best_per_task: dict[str, tuple[float, str, Trace]] = {}
        # task_id -> (score, candidate_id, trace)
        self._best_mean_score: float = 0.0
        self._best_candidate_id: Optional[str] = None
        # Default matches EvolutionaryConfig.novelty_alpha (0.3); production
        # always overrides via Archive(novelty_alpha=config.novelty_alpha).
        self.novelty_alpha = novelty_alpha
        # Forensic improvement #3 — WITHIN-TASK RECURSION depth term. When > 0,
        # _selection_weights adds a depth*headroom bonus that favors EXTENDING a
        # deep chain on a high-headroom task (so breeding builds within-task depth
        # the Stage-3 re-propagation can then revise). Default 0.0 ⇒ the depth
        # branch in _selection_weights is NEVER taken (strictly behind
        # ``if self.within_task_depth_bonus:``), so the weights are bit-for-bit
        # identical to HEAD. Production sets it to a modest BETA only when
        # ``within_task_recursion`` is ON.
        self.within_task_depth_bonus = within_task_depth_bonus
        # Audit #5 — absolute score ceiling consulted by _fresh_headroom_signal.
        # ``score``/``per_task_scores`` are RAW benchmark values, not normalized to
        # [0,1]; the headroom signal must therefore compare against the benchmark's
        # own ``score_scale()['hi']`` rather than a hardcoded 1.0. ``hi`` is None on
        # a ``continuous`` (unbounded) scale (symbolic_regression / AlgoTune /
        # AlphaEvolve), in which case the absolute-ceiling clause MUST be dropped and
        # headroom relies solely on the per-task-best lag comparison. Default 1.0
        # preserves the original [0,1]-scale behavior bit-for-bit. Only consulted
        # when ``within_task_depth_bonus > 0`` (non-default), so default runs are
        # unaffected either way.
        self._score_ceiling = score_ceiling
        # Magnitude of the score scale for scale-invariant selection / stopping
        # (roadmap v2 N4). None until frozen after warmup; see score_range().
        self._frozen_score_range: Optional[float] = None
        # Forensic improvement #1 — REGRESSION GUARD (default OFF = byte-identical).
        # When ON, the deployable per-task-best for a task may NEVER drop below the
        # BASE (seed-only, no-Ω) resample best for that task: per_task_best :=
        # max(all archived candidates, base floor). The floor is supplied once at
        # gen0 via set_base_floor() and pre-seeds _best_per_task so the existing
        # monotone-max in add() naturally refuses any Ω trace below the floor.
        # Empty/False here ⇒ set_base_floor is never called and the add() clamp
        # short-circuits, so the archive is bit-identical to HEAD.
        self._regression_guard = regression_guard
        self._base_floor: dict[str, tuple[float, str, Optional[Trace]]] = {}
        # Forensic improvement #2 — fired verified-code non-adoption bars,
        # {candidate_id: set of barred task_ids}. Recorded by add() when a
        # bar_from_best entry actually skips a per-task-best update, persisted
        # in checkpoint.json and re-applied by rebuild_from_disk so a barred
        # trace cannot become per-task-best after a --resume. Empty on every
        # default (bar-free) run.
        self._barred_from_best: dict[str, set[str]] = {}

    def add(
        self,
        candidate: Candidate,
        *,
        bar_from_best: Optional[set] = None,
    ) -> None:
        """Add candidate to archive. Updates per-task best tracking.

        Idempotent: if a candidate with the same candidate_id is already in
        the archive, this is a no-op (with a warning log). This makes resume-
        after-crash safe — if a crash lands between the disk write and the
        checkpoint update, the rebuilt archive may already contain a candidate
        whose id the loop is about to regenerate.

        ``bar_from_best`` (Forensic improvement #2 — default ``None`` ⇒ byte-
        identical): a set of task_ids whose trace must NOT update per-task-best
        even if it scored higher (the verified-code non-adoption penalty — the
        solver re-derived a verified helper inline instead of calling it). The
        candidate is still added in full (monotonic; mean_score intact); only the
        per-task-best index skips those tasks. ``None``/empty ⇒ the loop is
        unchanged.
        """
        if candidate.candidate_id in self._by_id:
            logger.warning(
                "Archive.add: candidate_id %s already present; "
                "ignoring duplicate (likely resume-after-crash)",
                candidate.candidate_id,
            )
            return

        self.candidates.append(candidate)
        self._by_id[candidate.candidate_id] = candidate

        # Update archive-best. Guard with isfinite as defense in depth — the
        # evolutionary orchestrator already filters non-finite scores out of
        # candidate.mean_score, but if anything else builds a Candidate by
        # hand (analysis scripts, future code paths) we don't want NaN/inf
        # to break parent-selection weights downstream.
        # Audit fix #4 — the first finite-mean candidate is the UNCONDITIONAL
        # archive-best (``_best_candidate_id is None``), mirroring the per-task
        # ``current is None`` seed below. The old ``> self._best_mean_score`` guard
        # floored the archive-best at the 0.0 init, so on a negative-capable
        # continuous scale (symbolic_regression: score_scale hi=None, fitness goes
        # negative) a strictly-negative best left ``_best_candidate_id`` None and
        # ``best_mean_score`` reported a phantom 0.0. ``_best_mean_score`` keeps its
        # 0.0 init so an EMPTY archive still serializes 0.0 (to_dict unchanged).
        if math.isfinite(candidate.mean_score) and (
            self._best_candidate_id is None
            or candidate.mean_score > self._best_mean_score
        ):
            self._best_mean_score = candidate.mean_score
            self._best_candidate_id = candidate.candidate_id

        # Update per-task best.
        # Skip NaN/inf scores — they're never "better" than a finite score,
        # and storing them first would corrupt the per-task best tracking
        # because (finite > NaN) is False (NaN comparisons are non-orderable).
        for trace in candidate.traces:
            if not math.isfinite(trace.score):
                continue
            # Forensic improvement #2 — non-adoption penalty: a trace that
            # re-derived a verified helper inline is barred from per-task-best.
            # Default bar_from_best is None ⇒ short-circuit ⇒ byte-identical.
            if bar_from_best and trace.task_id in bar_from_best:
                self._barred_from_best.setdefault(
                    candidate.candidate_id, set()
                ).add(trace.task_id)
                continue
            # REGRESSION-GUARD defensive clamp (dual-channel). When the guard is ON
            # and a base floor exists for this task, an Ω trace that is a genuine
            # (success, score) regression BELOW the floor can never become
            # per-task-best (never ship a per-task regression). The comparison is
            # dual-channel — it mirrors the (success, score) per-task-best ranking
            # below and the floor pick — so a None-trace floor entry (fl[2] is None)
            # counts as a successful seed win, and a valid success that out-ranks a
            # crash floor is NOT clamped. The clamp is defensive: set_base_floor
            # normally pre-seeds _best_per_task, but a crash floor with no script
            # does not, leaving `current` None so the monotone-max alone would
            # otherwise promote any trace. OFF / no floor ⇒ get(...) is None ⇒
            # short-circuit ⇒ byte-identical.
            if self._regression_guard:
                fl = self._base_floor.get(trace.task_id)
                if fl is not None:
                    floor_key = (bool(getattr(fl[2], "success", True)), fl[0])
                    if (bool(trace.success), trace.score) < floor_key:
                        continue
            current = self._best_per_task.get(trace.task_id)
            # Per-task-best is (success, score) lexicographic, NOT score alone: a
            # trace that RAN (success=True) out-ranks a crashed one (success=False)
            # regardless of score — `success` is the ran-vs-crashed RANKING channel,
            # `score` the reporting channel (Reaudit #9 dual-channel resolution). On a
            # negative-capable scale (symbolic_regression) a valid-but-poor fit
            # (success=True, score -0.70) must beat a 0.0-floored crash (success=False),
            # which a plain score-max inverts. BYTE-IDENTICAL on non-negative scales: a
            # failure floors to score 0.0 while any success scores >= 0, so the
            # success term never flips the max there (it could only differ on an exact
            # score tie with differing success, which those scales do not produce). A
            # base-floor entry (current[2] may be None) counts as a successful seed win.
            cur_key = (
                (bool(getattr(current[2], "success", True)), current[0])
                if current is not None
                else None
            )
            if cur_key is None or (bool(trace.success), trace.score) > cur_key:
                self._best_per_task[trace.task_id] = (
                    trace.score,
                    candidate.candidate_id,
                    trace,
                )

    def get(self, candidate_id: str) -> Candidate:
        """Get candidate by ID."""
        return self._by_id[candidate_id]

    def find(self, candidate_id: "str | None") -> "Candidate | None":
        """None-safe lookup (:meth:`get` raises KeyError).

        Returns None for a missing id and for ``candidate_id=None`` (e.g. a
        seed's ``parent_id``), so lineage walks need no pre-check.
        """
        if candidate_id is None:
            return None
        return self._by_id.get(candidate_id)

    @property
    def best_mean_score(self) -> float:
        return self._best_mean_score

    @property
    def best_candidate(self) -> Optional[Candidate]:
        if self._best_candidate_id is None:
            return None
        return self._by_id[self._best_candidate_id]

    def best_score_for_task(self, task_id: str) -> Optional[float]:
        """Return best finite score for a task, or None if no candidate has
        a finite score for it.

        Returns None (rather than 0.0) so callers can distinguish:
          - "no candidate ever produced a finite score for this task" (None)
          - "a candidate scored exactly 0.0" (0.0)

        NaN/inf scores are filtered at insertion (see add()), so a task with
        only non-finite traces will return None here.
        """
        entry = self._best_per_task.get(task_id)
        return entry[0] if entry else None

    def per_task_best_scores(self) -> dict[str, float]:
        """Return {task_id: best_score} across all archive candidates.

        Tasks where no candidate has a finite score are omitted from the dict
        (rather than mapped to 0.0). Callers iterating over expected task ids
        must check membership; do not assume keys cover every evaluated task.
        """
        return {tid: entry[0] for tid, entry in self._best_per_task.items()}

    def per_task_best_sources(self) -> dict[str, str]:
        """Return {task_id: winning candidate_id} — sibling of
        :meth:`per_task_best_scores` / :meth:`per_task_best_traces`.

        Same keys and iteration order as the other two (all three are derived
        from the same per-task-best index), so zipping them per task is safe.
        """
        return {tid: entry[1] for tid, entry in self._best_per_task.items()}

    def per_task_best_traces(self) -> dict[str, Trace]:
        """Return {task_id: best_trace} across all archive candidates."""
        return {tid: entry[2] for tid, entry in self._best_per_task.items()}

    def set_base_floor(
        self,
        floor: dict[str, float],
        traces: dict[str, Trace],
    ) -> None:
        """Forensic improvement #1 — install the BASE (seed-only) per-task floor.

        Records the no-Ω resample best per task and RAISES the per-task-best index
        to at least that floor, so the deployable best can never regress below what
        the base solver alone achieves. Called once at gen0 (and re-applied on
        resume). Only meaningful when the archive was built with
        ``regression_guard=True``; a no-op (other than recording the floor) when the
        floor does not beat an existing real win.

        MONOTONIC: this updates an INDEX entry (``_best_per_task[tid]``) with a NEW
        resampled Trace under the synthetic id ``gen0_seed``; it never mutates the
        stored ``gen0_seed`` Candidate or its traces. The base-floor trace is a
        fresh draw, so the archived seed candidate's ``per_task_scores`` are
        untouched.
        """
        for tid, score in floor.items():
            if not math.isfinite(score):
                continue
            tr = traces.get(tid)
            self._base_floor[tid] = (score, "gen0_seed", tr)
            # T3.10 — only RAISE the per-task-best index off a floor entry when
            # its trace is ROUTABLE (non-None with a script). A None/empty-script
            # floor trace still seeds _base_floor (for the regression clamp in
            # add()), but raising per-task-best on score alone would seed
            # archive-best/oracle with an unroutable winner — there is no script
            # to deploy and the task is absent from task_solution_map. In the
            # normal regression_guard run the seed traces carry their scripts, so
            # this guard is a no-op and the index is raised exactly as before.
            if tr is None or not tr.script:
                continue
            current = self._best_per_task.get(tid)
            if current is None or score > current[0]:
                self._best_per_task[tid] = (score, "gen0_seed", tr)

    def base_floor_snapshot(self) -> dict[str, tuple[float, str, Optional[Trace]]]:
        """Shallow copy of the regression-guard base floor
        ``{task_id: (score, candidate_id, trace|None)}``.

        Empty unless :meth:`set_base_floor` was called (regression_guard runs).
        The copy is shallow: mutating the returned dict never touches the
        archive, but the tuples/traces are the stored objects.
        """
        return dict(self._base_floor)

    def barred_from_best_snapshot(self) -> dict[str, list[str]]:
        """Fired verified-code bars ``{candidate_id: sorted barred task_ids}``.

        Empty unless :meth:`add` skipped a per-task-best update because of
        ``bar_from_best`` (verified_code runs). Persisted in checkpoint.json by
        ``RunPersistence.save_checkpoint`` and re-applied via
        :meth:`rebuild_from_disk` so the non-adoption penalty survives --resume.
        """
        return {cid: sorted(tids) for cid, tids in self._barred_from_best.items()}

    def _elite_ids(self) -> list[str]:
        """Ordered, de-duplicated elite candidate ids: the archive-best first,
        then every per-task winner (sorted by task id for determinism).

        These are the richest stepping-stones — reserved in selection and always
        breedable. Reserved coverage in :meth:`select_parents` extends only to
        the first ``n - 1`` ids per generation; with more distinct winners than
        slots the task-id sort makes the reservation deterministically
        alphabetical-first unless the caller supplies an ``elite_rotation``
        offset. All elites remain breedable and weighted-samplable regardless.
        """
        ids: list[str] = []
        if self._best_candidate_id is not None:
            ids.append(self._best_candidate_id)
        for tid in sorted(self._best_per_task):
            cid = self._best_per_task[tid][1]
            if cid not in ids:
                ids.append(cid)
        return ids

    @staticmethod
    def ucb_exploration_bonus(alpha: float, pool_size: int, num_children: int) -> float:
        """The UCB exploration term of :meth:`_selection_weights`:
        ``alpha * sqrt(ln(pool_size) / (1 + num_children))``, 0.0 for a pool
        of <= 1. Single source for the selection weights AND the orchestrator's
        parent-selection run-log diagnostic, so the logged bonus can never
        drift from the one actually used.
        """
        if pool_size <= 1:
            return 0.0
        return alpha * math.sqrt(math.log(pool_size) / (1 + num_children))

    def _selection_weights(self, candidates: list[Candidate]) -> list[float]:
        """Scale-invariant parent-selection weights (roadmap v2 2.1 / N4b).

        ``weight = rank_norm(mean_score) + alpha * sqrt(ln N / (1 + num_children))``

        ``rank_norm`` is the fraction of candidates with a strictly smaller
        FINITE mean_score (in [0,1]) — invariant to any positive affine rescale
        of the score, so the exploration term never vanishes against a large
        score magnitude the way the old additive ``alpha/(1+children)`` bonus did
        on continuous scales. Non-finite means rank at the bottom. The UCB term
        favors under-explored candidates. Deterministic given the inputs.
        """
        n = len(candidates)
        finite_means = [
            c.mean_score if math.isfinite(c.mean_score) else float("-inf")
            for c in candidates
        ]
        # Forensic improvement #3 — within-task depth term (gated). max_depth is
        # the pool ceiling for depth_norm; only computed when the bonus is live.
        max_depth = (
            max((c.depth for c in candidates), default=1)
            if self.within_task_depth_bonus else 1
        )
        weights: list[float] = []
        for i, c in enumerate(candidates):
            strictly_below = sum(1 for m in finite_means if m < finite_means[i])
            rank_norm = strictly_below / (n - 1) if n > 1 else 0.0
            ucb = self.ucb_exploration_bonus(self.novelty_alpha, n, c.num_children)
            w = rank_norm + ucb
            # Forensic improvement #3 — WITHIN-TASK RECURSION depth term. STRICTLY
            # behind ``if self.within_task_depth_bonus:`` (never ``+ 0.0``), so when
            # the bonus is 0.0 (default / flag OFF) this branch is NOT taken and
            # ``w`` is bit-for-bit identical to HEAD. When live (flag ON), favor
            # EXTENDING a deep chain whose fresh task still has headroom, so
            # breeding builds within-task depth on the same high-headroom task.
            if self.within_task_depth_bonus:
                depth_norm = c.depth / max_depth if max_depth > 0 else 0.0
                headroom = self._fresh_headroom_signal(c)
                w += self.within_task_depth_bonus * depth_norm * headroom
            weights.append(w if math.isfinite(w) and w > 1e-6 else 1e-6)
        return weights

    def _fresh_headroom_signal(self, candidate: Candidate) -> float:
        """1.0 if ``candidate``'s FRESH task(s) still have room to improve, else 0.0.

        Forensic improvement #3 helper (only consulted when
        ``within_task_depth_bonus > 0``). A task is FRESH when its trace was
        produced at this candidate's own depth (``trace.depth >= candidate.depth``
        — mirrors :func:`analysis.depth_attribution.split_candidate_mean`). Headroom
        exists when a fresh task's score lags the archive's per-task best for that
        SAME task (room to catch up / a deeper sibling could still beat it) OR is
        below the benchmark's absolute score ceiling (``self._score_ceiling`` —
        the ``score_scale()['hi']``; room to the score-scale max). A chain whose
        fresh task already holds the per-task best AND sits at the ceiling is
        saturated → 0.0 (no point deepening it). Audit #5: ``score``/
        ``per_task_scores`` are RAW benchmark values, not normalized to [0,1], so
        the ceiling must NOT be a hardcoded 1.0 — on a ``continuous`` unbounded
        scale (``hi=None`` ⇒ ``self._score_ceiling is None``) the absolute-ceiling
        clause is dropped entirely and only the per-task-best lag is consulted.
        Read-only over ``candidate.traces`` / ``per_task_scores`` and
        ``self._best_per_task``.
        """
        fresh = {
            t.task_id for t in candidate.traces
            if math.isfinite(t.score) and t.depth >= candidate.depth
        }
        if not fresh:
            return 0.0
        for tid in fresh:
            score = candidate.per_task_scores.get(tid)
            if score is None or not math.isfinite(score):
                continue
            best = self._best_per_task.get(tid)
            best_score = best[0] if best is not None else score
            # Per-task-best lag: a deeper sibling could still catch up / beat it.
            if score < best_score - 1e-9:
                return 1.0
            # Absolute score ceiling. Audit #5: consult the benchmark's own
            # ``score_scale()['hi']`` (``self._score_ceiling``), NOT a hardcoded
            # 1.0. ``None`` ⇒ continuous/unbounded scale ⇒ no absolute ceiling, so
            # this clause is skipped. With the default 1.0 this is byte-identical
            # to the original ``or score < 1.0 - 1e-9``.
            if self._score_ceiling is not None and score < self._score_ceiling - 1e-9:
                return 1.0
        return 0.0

    def select_parents(
        self,
        n: int,
        rng: random.Random | None = None,
        pool: list[Candidate] | None = None,
        elite_rotation: int = 0,
    ) -> list[Candidate]:
        """Select n parents: reserved elites (when n>=2) + scale-invariant
        weighted sampling (rank-normalized fitness + UCB exploration).

        The reserved-elite window keeps only the first ``n - 1`` elites of the
        task-id-sorted list, so with more distinct per-task winners than slots
        the reservation is deterministically alphabetical-first. A nonzero
        ``elite_rotation`` offset rotates the window to cure this ONLY when a
        rotating slot exists beyond the pinned archive-best (n >= 3); at n == 2
        the sole reserved slot holds the pinned best every generation (same as
        OFF).

        Args:
            pool: If provided, select only from this subset of candidates (e.g.
                  the breedable pool of extendable, non-regressing candidates).
            elite_rotation: Deterministic rotation offset for the reserved-elite
                  window, typically the 1-based generation index. ``0`` = OFF =
                  byte-identical HEAD reservation.
        """
        candidates = pool if pool is not None else self.candidates
        if not candidates:
            return []
        rng = rng or random.Random()

        selected: list[Candidate] = []
        # Elitism: when breeding multiple parents, RESERVE slots for the elite
        # set (archive-best + per-task winners) intersected with the pool, so the
        # best stepping-stones are never dropped. Gated to n>=2 so a single-beam
        # run is not fully determined by the elite (weighted sampling already
        # favors it); leave >=1 slot for exploration.
        if n >= 2:
            by_id = {c.candidate_id: c for c in candidates}
            elite = [i for i in self._elite_ids() if i in by_id]
            # Refine §6b F006 — flag-gated elite-window rotation. The reserved window
            # keeps only the first n-1 elites of the task-id-sorted list, so with more
            # distinct per-task winners than slots the alphabetically-first winners
            # were reserved EVERY generation and later ones never (partial starvation;
            # they still compete in weighted sampling). When a nonzero rotation offset
            # is supplied (--elite-rotation plumbs the generation index) AND the
            # truncation actually bites (len(elite) > n - 1), pin the archive-best at
            # slot 0 (strongest elitism guarantee, and absent-from-pool safe) and
            # rotate the per-task-winner tail by offset % len(tail). Full coverage
            # (every elite reserved within len(tail) consecutive generations)
            # requires at least one ROTATING reserved slot beyond the pin, i.e.
            # n - 1 > len(pinned); with the single archive-best pin that means
            # n >= 3. At n == 2 with the best pinned there is exactly one reserved
            # slot (reserved[:1]); it holds the pinned best every generation, so the
            # tail is NOT rotated into a reserved slot (byte-identical to OFF) — the
            # tail still competes in weighted sampling for the remaining slot.
            # offset == 0 (default) never enters this branch — byte-identical, same
            # rng stream.
            if elite_rotation and len(elite) > n - 1:
                pinned = elite[:1] if elite[0] == self._best_candidate_id else []
                tail = elite[len(pinned):]
                r = elite_rotation % len(tail)
                elite = pinned + tail[r:] + tail[:r]
            reserved = [by_id[i] for i in elite]
            selected.extend(reserved[: max(0, n - 1)])

        remaining = n - len(selected)
        if remaining > 0:
            weights = self._selection_weights(candidates)
            selected.extend(rng.choices(candidates, weights=weights, k=remaining))
        return selected

    def breedable_pool(self, max_depth: int, tol_frac: float = 0.05) -> list[Candidate]:
        """Extendable candidates eligible to be PARENTS (roadmap v2 6.1).

        ``archive.add`` stays unconditional (full history + oracle harvest are
        preserved); this filters the PARENT POOL only, excluding regressors so
        the search stops breeding from the weak-chain cloud. A candidate at
        ``depth < max_depth`` is breedable iff:
          - it is an ELITE (archive-best or a per-task winner), OR
          - it is a seed (``parent_id is None``), OR
          - it did NOT regress vs its parent: ``mean_score >= parent.mean - tol``
            AND no task dropped more than ``tol`` below the parent's score on it.

        ``tol`` is RANGE-RELATIVE (``tol_frac * score_range``), never absolute, so
        the guard behaves consistently on [0,1], continuous, and binary scales.
        Equal means count as non-regressing (``>=``) so binary ties stay
        breedable. Never returns empty when any candidate is extendable: falls
        back to seeds ∪ elites, then to all extendable candidates.
        """
        # A synthesized Ω_merge candidate (carries a per-task routing map) is a
        # deployable oracle, not a breedable lineage — never select it as a parent.
        extendable = [
            c for c in self.candidates
            if c.depth < max_depth
            and not any(ic.task_solution_map for ic in c.injected_codes)
        ]
        if not extendable:
            return []
        tol = tol_frac * self.score_range()
        elite_ids = set(self._elite_ids())
        pool = [c for c in extendable if self._is_breedable(c, elite_ids, tol)]
        if pool:
            return pool
        fb = [c for c in extendable if c.parent_id is None or c.candidate_id in elite_ids]
        return fb if fb else extendable

    def _is_breedable(self, c: Candidate, elite_ids: set[str], tol: float) -> bool:
        if c.candidate_id in elite_ids or c.parent_id is None:
            return True
        parent = self._by_id.get(c.parent_id)
        if parent is None:
            return True  # orphaned parent (shouldn't happen) — don't exclude
        if not math.isfinite(c.mean_score):
            return False
        if c.mean_score < parent.mean_score - tol:
            return False  # regressed on the mean
        for tid, score in c.per_task_scores.items():
            p_score = parent.per_task_scores.get(tid)
            if p_score is not None and math.isfinite(score) and score < p_score - tol:
                return False  # regressed on a task vs the parent
        return True

    def score_range(self) -> float:
        """Deterministic magnitude of the score scale (roadmap v2 N4).

        Returns ``max - min`` over the FINITE candidate ``mean_score``s, floored
        at ``1e-9`` so it is always a safe denominator for scale-invariant
        selection / stopping (never zero, even with a single candidate). Computed
        over a SORTED copy of the finite means, so the result is independent of
        insertion order (reproducibility).

        If a range has been frozen (:meth:`freeze_score_range` — called after the
        seed+gen1 warmup and persisted in ``checkpoint.json``), that frozen value
        is returned instead, so the normalizer does not drift as the archive
        grows. Not wired into selection/stopping until the steps that consume it.
        """
        if self._frozen_score_range is not None:
            return self._frozen_score_range
        finite = sorted(
            c.mean_score for c in self.candidates if math.isfinite(c.mean_score)
        )
        if len(finite) < 2:
            return 1e-9
        return max(1e-9, finite[-1] - finite[0])

    @property
    def frozen_score_range(self) -> Optional[float]:
        """The frozen score-range normalizer, or None while still unfrozen.

        Pure pass-through (no coercion): the value is exactly what
        :meth:`freeze_score_range` stored. The setter exists for checkpoint
        restore, which writes the persisted value back verbatim.
        """
        return self._frozen_score_range

    @frozen_score_range.setter
    def frozen_score_range(self, value: Optional[float]) -> None:
        self._frozen_score_range = value

    def freeze_score_range(self, value: Optional[float] = None) -> float:
        """Freeze the score range (after warmup) so it stops drifting.

        With ``value=None`` the current :meth:`score_range` is captured. The
        orchestrator persists the frozen value in ``checkpoint.json`` and
        restores it on resume, so selection weights are reproducible across a
        pause/resume boundary.
        """
        self._frozen_score_range = (
            value if value is not None else self.score_range()
        )
        return self._frozen_score_range

    def get_inspiration_traces(
        self,
        parent: Candidate,
        tasks: list[TaskDescription],
        max_traces: int = 5,
    ) -> list[Trace]:
        """Get best solutions from other candidates for tasks the parent failed or scored low.

        Returns traces from OTHER candidates that outperform the parent on specific tasks.
        Prioritizes tasks with the largest score gap (biggest improvement opportunity).
        A task missing from ``parent.per_task_scores`` (parent never produced a
        finite score) counts as failed at the 0.0 crash-floor baseline; on
        negative-capable scales the negative archive best is still offered.
        """
        # Collect all candidates for inspiration with their score gaps
        candidates_for_inspiration: list[tuple[float, Trace]] = []

        for task in tasks:
            # Refine §6b F003 — a task ABSENT from parent.per_task_scores means the
            # parent produced NO finite score there (crashed / non-finite): the
            # orchestrator builds per_task_scores from finite traces only. The old
            # ``.get(task.task_id, 0.0)`` floored that state at a phantom 0.0, so on
            # a negative-capable scale (symbolic_regression) a negative archive best
            # (e.g. -0.5, success=True) failed ``best_score > 0.0`` and the parent got
            # NO inspiration for exactly the task it crashed on — the same 0.0-floor
            # class fixed by audit #4 (archive-best) and reaudit #9 (per-task-best
            # dual channel). Missing keeps the 0.0 BASELINE for gap ordering
            # (BYTE-IDENTICAL on non-negative scales, where ``best != 0.0`` <=>
            # ``best > 0.0`` and ``gap = best - 0.0`` is unchanged); only the
            # inclusion test generalizes to ``best_score != 0.0`` so a negative best
            # is no longer suppressed. A 0.0 best stays excluded on every scale (no
            # signal above the crash floor). Negative gaps therefore rank AFTER all
            # positive gaps — deliberate: re-ranking crashed tasks first ("missing =
            # worst possible") would perturb [0,1] default-path prompt bytes and is
            # left as a possible future flag-gated strategy.
            parent_score = parent.per_task_scores.get(task.task_id)
            if parent_score is not None and not math.isfinite(parent_score):
                # Defense in depth: a hand-built candidate (analysis/tests) may carry
                # NaN/inf; "no finite score" is the same never-scored state as a
                # missing key. Unreachable from the orchestrator (finite-only build).
                parent_score = None

            best_entry = self._best_per_task.get(task.task_id)
            if best_entry is None:
                continue

            best_score, best_cand_id, best_trace = best_entry
            # Only include if ANOTHER candidate did better.
            if best_cand_id == parent.candidate_id:
                continue
            if parent_score is None:
                include = best_score != 0.0
                gap = best_score  # crash-floor 0.0 baseline (ordering unchanged on [0,1])
            else:
                include = best_score > parent_score
                gap = best_score - parent_score
            if include:
                candidates_for_inspiration.append((gap, best_trace))

        # Sort by largest gap first (biggest improvement opportunity)
        candidates_for_inspiration.sort(key=lambda x: -x[0])
        return [trace for _, trace in candidates_for_inspiration[:max_traces]]

    def to_dict(self) -> dict:
        """Serialize archive for saving."""
        return {
            "size": len(self.candidates),
            "best_mean_score": self._best_mean_score,
            "best_candidate_id": self._best_candidate_id,
            "per_task_best": {
                tid: {"score": entry[0], "candidate_id": entry[1]}
                for tid, entry in self._best_per_task.items()
            },
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "parent_id": c.parent_id,
                    "iteration": c.iteration,
                    "depth": c.depth,
                    "mean_score": c.mean_score,
                    "pass_at_1": c.pass_at_1,
                    "num_children": c.num_children,
                    "temperature_used": c.temperature_used,
                    "total_tokens": c.total_tokens,
                    "per_task_scores": c.per_task_scores,
                }
                for c in self.candidates
            ],
        }

    @classmethod
    def rebuild_from_disk(
        cls,
        archive_dir: Path,
        novelty_alpha: float = 0.3,
        regression_guard: bool = False,
        within_task_depth_bonus: float = 0.0,
        score_ceiling: Optional[float] = 1.0,
        barred_from_best: "Optional[dict[str, list[str]]]" = None,
    ) -> Archive:
        """Reconstruct archive from saved candidate directories.

        Reads each candidate's summary.json, injected_code_d*.json,
        traces/*.json, and repropagation_d*.json self-repair sidecars to
        rebuild the full Archive state.

        ``novelty_alpha`` defaults to 0.3 to match ``Archive.__init__`` /
        ``EvolutionaryConfig.novelty_alpha`` (production always passes it
        explicitly).

        ``regression_guard`` is propagated to the rebuilt archive so a resumed run
        re-honors the base floor (the floor itself is re-applied via
        ``set_base_floor`` from the checkpoint by the orchestrator). Default False
        ⇒ legacy byte-identical rebuild. ``within_task_depth_bonus`` (forensic #3,
        default 0.0) is likewise propagated so a resumed within-task-recursion run
        keeps its depth-aware selection term live; 0.0 ⇒ byte-identical rebuild.
        ``score_ceiling`` (audit #5 / reaudit #1) is the bound benchmark's
        ``score_scale()['hi']`` consulted by ``_fresh_headroom_signal``; it MUST be
        propagated so a resumed within-task-recursion run restores the same
        absolute-ceiling calibration (``None`` on a continuous scale drops the
        clause). Default 1.0 preserves the legacy [0,1]-scale behavior; only
        consulted when ``within_task_depth_bonus > 0``, so default rebuilds are
        byte-identical either way.

        ``barred_from_best`` (forensic #2) is the checkpoint's fired-bar record
        ``{candidate_id: [task_ids]}``; each candidate's bars are re-passed to
        ``add()`` during replay so a barred trace cannot become per-task-best
        after a --resume. Default ``None`` (legacy checkpoints lack the key) ⇒
        the bare legacy replay.
        """
        archive = cls(
            novelty_alpha=novelty_alpha,
            regression_guard=regression_guard,
            within_task_depth_bonus=within_task_depth_bonus,
            score_ceiling=score_ceiling,
        )

        if not archive_dir.exists():
            return archive

        # Find candidate directories and pre-read summaries (one read per candidate)
        candidate_entries: list[tuple[Path, dict]] = []
        for d in archive_dir.iterdir():
            summary_path = d / "summary.json"
            if d.is_dir() and summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    candidate_entries.append((d, summary))
                except Exception as e:
                    logger.warning("Failed to read summary for %s: %s — skipping", d.name, e)

        # Sort by (iteration, name) to replay add() in original order
        candidate_entries.sort(key=lambda x: (x[1].get("iteration", 0), x[0].name))

        for cand_dir, summary in candidate_entries:
            try:

                # Rebuild traces
                traces = []
                traces_dir = cand_dir / "traces"
                if traces_dir.exists():
                    for trace_file in sorted(traces_dir.glob("*.json")):
                        trace_data = json.loads(trace_file.read_text())
                        traces.append(Trace.model_validate(trace_data))

                # Rebuild injected codes
                injected_codes = []
                depth = summary.get("depth", 1)
                for d in range(2, depth + 1):
                    ic_path = cand_dir / f"injected_code_d{d}.json"
                    if ic_path.exists():
                        ic_data = json.loads(ic_path.read_text())
                        injected_codes.append(InjectedCode.model_validate(ic_data))

                # T3.9 — partial-write integrity cross-check. The writer emits
                # summary.json FIRST and the injected_code_d*.json sidecars LAST,
                # so a crash between them leaves a depth=N summary with a full
                # mean_score over a TRUNCATED solver chain (missing sidecars). A
                # clean write always yields len(injected_codes) == depth - 1.
                # On mismatch, skip the candidate (mirrors the num_children
                # integrity warning below) rather than seed archive-best/oracle
                # off a full-chain score backed by a truncated solver.
                expected_ics = depth - 1
                if len(injected_codes) != expected_ics:
                    logger.warning(
                        "injected_code integrity: candidate %s declares depth=%d "
                        "but only %d/%d injected_code sidecar(s) present "
                        "(partial write?) — skipping",
                        summary.get("candidate_id", cand_dir.name),
                        depth, len(injected_codes), expected_ics,
                    )
                    continue

                # Self-repair provenance: read back the repropagation_d{t}.json
                # sidecars the writer emits. raw_omega_prompt/raw_omega_response
                # restore as "" (parity with InjectedCode: raw Ω text is excluded
                # from the JSON sidecar and not restored). A corrupt sidecar is
                # skipped — diagnostic, not load-bearing — never sinks the
                # candidate. Glob is lexicographic (deterministic; each event
                # carries its own target_depth, so order is not load-bearing).
                self_repair_events = []
                for ev_path in sorted(cand_dir.glob("repropagation_d*.json")):
                    try:
                        self_repair_events.append(
                            SelfRepairEvent.model_validate(json.loads(ev_path.read_text()))
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to read self-repair sidecar %s for %s: %s — skipping",
                            ev_path.name, cand_dir.name, e,
                        )

                candidate = Candidate(
                    candidate_id=summary["candidate_id"],
                    parent_id=summary.get("parent_id"),
                    iteration=summary.get("iteration", 0),
                    depth=summary.get("depth", 1),
                    injected_codes=injected_codes,
                    traces=traces,
                    pass_at_1=summary.get("pass_at_1", 0.0),
                    mean_score=summary.get("mean_score", 0.0),
                    per_task_scores=summary.get("per_task_scores", {}),
                    num_children=summary.get("num_children", 0),
                    total_tokens=summary.get("total_tokens", 0),
                    # Cost integrity: per-candidate INNER-LLM accounting survives
                    # resume when the writer emitted the (gated, non-zero-only)
                    # keys; legacy summaries lack them -> 0, identical to before.
                    inner_tokens=summary.get("inner_tokens", 0),
                    inner_prompt_tokens=summary.get("inner_prompt_tokens", 0),
                    inner_completion_tokens=summary.get("inner_completion_tokens", 0),
                    inner_calls=summary.get("inner_calls", 0),
                    temperature_used=summary.get("temperature_used", 0.7),
                    created_at=summary.get("created_at") or datetime.now().isoformat(),
                    self_repair_events=self_repair_events,
                )
                # Union the checkpoint-supplied bar with the candidate's own
                # co-located bar (barred_from_best_task_ids in summary.json).
                # The per-candidate copy is authoritative — always in sync with
                # the candidate dir — so a stale checkpoint (crash before its
                # write) cannot lose the bar. Bars are monotonic, so union never
                # over-bars. Both absent -> None -> byte-identical legacy replay.
                ck_bar = (barred_from_best or {}).get(candidate.candidate_id)
                disk_bar = summary.get("barred_from_best_task_ids")
                merged = set(ck_bar or []) | set(disk_bar or [])
                archive.add(candidate, bar_from_best=merged or None)
            except Exception as e:
                logger.warning("Failed to load candidate from %s: %s — skipping", cand_dir.name, e)

        # 2.2: recompute num_children from parent_id edges (self-healing on
        # resume). The per-candidate summary.json captures num_children at
        # creation (0) and is never re-flushed when the parent later gains
        # children, so reading it back resets exploration weighting to 0. The
        # live count is reconstructed deterministically from the loaded edges.
        child_counts: dict[str, int] = {}
        for c in archive.candidates:
            if c.parent_id is not None:
                child_counts[c.parent_id] = child_counts.get(c.parent_id, 0) + 1
        for c in archive.candidates:
            c.num_children = child_counts.get(c.candidate_id, 0)
        assigned = sum(c.num_children for c in archive.candidates)
        n_non_seed = sum(1 for c in archive.candidates if c.parent_id is not None)
        if assigned != n_non_seed:
            logger.warning(
                "num_children integrity: sum(num_children)=%d != non-seed count=%d "
                "(dangling parent_id?)", assigned, n_non_seed,
            )

        logger.info(
            "Rebuilt archive from disk: %d candidates, best=%.3f",
            len(archive), archive.best_mean_score,
        )
        return archive

    def __len__(self) -> int:
        return len(self.candidates)
