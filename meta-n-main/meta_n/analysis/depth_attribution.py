"""Offline depth-attribution + per-task-delta analysis (S0.4 / S0.5).

Read-only post-hoc helpers over an on-disk evolutionary archive. These add NO
framework change and NO new persisted field — every input is data ALREADY on
disk: each ``Trace.depth`` (written by every solve) and each candidate's
``per_task_scores`` / ``parent_id`` (written into ``summary.json``).

S0.4 — *depth attribution*: split a candidate's mean score into the portion
earned by FRESHLY-solved tasks (``trace.depth == candidate.depth``) vs the
portion INHERITED from a shallower layer (``trace.depth < candidate.depth``).
This replaces the two dropped persisted fields (``solved_at_depth`` is byte-equal
to ``Trace.depth``; ``inherited_from`` is recoverable) with an offline computation.

S0.5 — *per-task delta*: ``child.per_task_scores - parent.per_task_scores`` over
the tasks present in both, an offline join over the archive's parent edges.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from meta_n.core.archive import Archive, Candidate
from meta_n.core.meta_layer import InjectedCode, Trace


# --------------------------------------------------------------------------- #
# S0.4 — depth attribution
# --------------------------------------------------------------------------- #
def trace_depth_distribution(traces: Iterable[Trace]) -> dict[int, int]:
    """Histogram ``trace.depth`` over an iterable of traces (sorted by depth).

    Over the full S0 golden corpus this reproduces the audit hand-computed
    split ``{1: 55, 2: 14, 3: 13, 4: 13, 5: 7}``.
    """
    dist: dict[int, int] = {}
    for t in traces:
        dist[t.depth] = dist.get(t.depth, 0) + 1
    return dict(sorted(dist.items()))


def archive_depth_distribution(archive: Archive) -> dict[int, int]:
    """Depth histogram over every trace of every candidate in ``archive``."""
    return trace_depth_distribution(
        t for c in archive.candidates for t in c.traces
    )


@dataclass
class DepthSplit:
    """A candidate's mean decomposed into fresh-vs-inherited contributions."""

    candidate_id: str
    candidate_depth: int
    fresh_tasks: list[str]      # trace.depth == candidate.depth
    inherited_tasks: list[str]  # trace.depth <  candidate.depth
    n_fresh: int
    n_inherited: int
    fresh_mean: float           # mean per_task_score over fresh tasks (0.0 if none)
    inherited_mean: float       # mean per_task_score over inherited tasks (0.0 if none)
    overall_mean: float         # mean per_task_score over all attributed tasks

    @property
    def fresh_fraction(self) -> float:
        """Share of attributed tasks that were freshly solved at this depth."""
        total = self.n_fresh + self.n_inherited
        return self.n_fresh / total if total else 0.0


def split_candidate_mean(candidate: Candidate) -> DepthSplit:
    """Split ``candidate``'s per-task scores into fresh vs inherited (S0.4).

    A task is FRESH when its trace was produced at this candidate's own depth
    (``trace.depth == candidate.depth``) and INHERITED when carried up from a
    shallower layer (``trace.depth < candidate.depth``). Scores come from
    ``candidate.per_task_scores`` (the value that feeds ``mean_score``), so the
    fresh/inherited means are a faithful decomposition of the candidate mean.
    """
    depth_by_task: dict[str, int] = {}
    for t in candidate.traces:
        # Highest-depth trace wins if a task has several (matches the solver's
        # current-score collapse); ties keep the first seen.
        if t.task_id not in depth_by_task or t.depth > depth_by_task[t.task_id]:
            depth_by_task[t.task_id] = t.depth

    fresh_tasks: list[str] = []
    inherited_tasks: list[str] = []
    fresh_scores: list[float] = []
    inherited_scores: list[float] = []
    all_scores: list[float] = []
    for tid, score in candidate.per_task_scores.items():
        d = depth_by_task.get(tid)
        if d is None:
            continue  # no trace on disk for this task — cannot attribute
        all_scores.append(score)
        if d == candidate.depth:
            # Fresh: this candidate's own solve produced the trace at its own depth.
            fresh_tasks.append(tid)
            fresh_scores.append(score)
        else:
            # Inherited: a trace at d < candidate.depth was carried up frozen from a
            # shallower layer, and a trace at d > candidate.depth was NOT produced by
            # this candidate's own solve either — it is a deeper per-task-best
            # inherited frozen (plain --consolidate keeps the producing depth with no
            # within-task clamp). The d > depth score is counted here (in all_scores
            # and inherited_scores), not dropped.
            inherited_tasks.append(tid)
            inherited_scores.append(score)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return DepthSplit(
        candidate_id=candidate.candidate_id,
        candidate_depth=candidate.depth,
        fresh_tasks=fresh_tasks,
        inherited_tasks=inherited_tasks,
        n_fresh=len(fresh_tasks),
        n_inherited=len(inherited_tasks),
        fresh_mean=_mean(fresh_scores),
        inherited_mean=_mean(inherited_scores),
        overall_mean=_mean(all_scores),
    )


# --------------------------------------------------------------------------- #
# S0.5 — per-task delta (child vs parent)
# --------------------------------------------------------------------------- #
def per_task_delta(child: Candidate, parent: Candidate) -> dict[str, float]:
    """``child.per_task_scores - parent.per_task_scores`` over shared tasks."""
    return {
        tid: child.per_task_scores[tid] - parent.per_task_scores[tid]
        for tid in child.per_task_scores
        if tid in parent.per_task_scores
    }


def per_task_delta_from_archive(
    archive: Archive, child_id: str
) -> Optional[dict[str, float]]:
    """Offline join: per-task delta of ``child_id`` vs its archived parent.

    Returns ``None`` for a seed (``parent_id is None``) or when the parent is
    absent from the archive (dangling edge).
    """
    child = archive.get(child_id)
    if child.parent_id is None:
        return None
    try:
        parent = archive.get(child.parent_id)
    except KeyError:
        return None
    return per_task_delta(child, parent)


# --------------------------------------------------------------------------- #
# DEPTH-PROBE — genuine WITHIN-TASK recursion vs ROUTER-STACKING
# --------------------------------------------------------------------------- #
# The screen tested BASE-SOLVER headroom; this answers the prerequisite the
# CO-Bench depth-5 result exposed: when the chain grows to depth D, is that D a
# stack of layers that act on the SAME task (genuine within-task recursion — a
# layer reasoning about the layer below on the same problem, which is what Stage 2
# within-layer-refine and Stage 3 downward-re-propagation need to have anything
# to act on), or is it a stack of DISJOINT per-task routers + frozen inheritance
# (the CO-Bench artifact: --consolidate routed each task to its own frozen winner,
# so ``trace.depth`` collapsed 55/102 to depth-1 even though the chain was depth-5)?
#
# Everything below is read-only over data ALREADY on disk: each candidate's
# ``parent_id`` (the chain), each ``Trace.depth`` (which layer FRESH-solved a task
# vs inherited it frozen), and each ``InjectedCode`` (whether a layer is a generic
# within-task pre_process / helper, or a per-task ROUTER — ``task_solution_map`` is
# the Ω_merge frozen-route channel; a ``pre_process`` that branches on
# ``task.task_id`` / ``task.description`` is the disjoint-description gate the Ω
# prompt itself flags as non-transferable, prompts.py:196).
# --------------------------------------------------------------------------- #
def ancestry_chain(archive: Archive, leaf_id: str) -> list[Candidate]:
    """The seed→…→``leaf_id`` candidate chain via ``parent_id`` (read-only).

    Walks parent edges up from ``leaf_id`` to the seed (``parent_id is None``),
    then reverses so index 0 is the seed and the last element is the leaf. Stops
    defensively on a dangling/absent parent or a cycle (a candidate seen twice).
    """
    seen: set[str] = set()
    rev: list[Candidate] = []
    cur: Optional[str] = leaf_id
    while cur is not None and cur not in seen:
        seen.add(cur)
        try:
            c = archive.get(cur)
        except KeyError:
            break
        rev.append(c)
        cur = c.parent_id
    rev.reverse()
    return rev


def _preprocess_gates_on_task_identity(src: str) -> bool:
    """True iff a ``pre_process`` body branches on ``task.task_id`` / ``task.description``.

    That is the *disjoint-description gate* signature: the layer routes by WHICH
    task it is (CO-Bench's task-disjoint router), not by task STRUCTURE
    (``task.metadata`` size/shape, the within-task-transferable conditioning the Ω
    prompt prescribes, prompts.py:196). Reading ``task.metadata`` is therefore NOT
    flagged. AST-based with a substring fallback for un-parseable code.
    """
    if not src:
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ("task.task_id" in src) or ("task.description" in src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in ("task_id", "description")
            and isinstance(node.value, ast.Name)
            and node.value.id == "task"
        ):
            return True
    return False


def classify_injection_layer(ic: InjectedCode) -> str:
    """Coarse role of ONE injected layer (the within-task-vs-router discriminator).

    Returns one of:
      * ``"router_frozen"``     — non-empty ``task_solution_map`` (the Ω_merge
        per-task frozen-route channel; pure disjoint routing).
      * ``"router_taskgate"``   — ``pre_process`` that gates on task identity
        (``task.task_id`` / ``task.description``); a disjoint-description gate.
      * ``"code_helper"``       — ships a Python/bash ``code_library`` helper (the
        constructive within-task code channel; reusable across re-solves).
      * ``"generic_preprocess"``— a ``pre_process`` that does NOT gate on identity
        (structure-conditioned guidance; within-task).
      * ``"empty"``             — contributes nothing (a resample-only layer).

    A layer that ships BOTH a helper and a task-identity gate is classified as the
    router (the gate is the load-bearing routing behaviour).
    """
    if ic.task_solution_map:
        return "router_frozen"
    if ic.pre_process and _preprocess_gates_on_task_identity(ic.pre_process):
        return "router_taskgate"
    if ic.code_library or ic.code_library_bash:
        return "code_helper"
    if ic.pre_process:
        return "generic_preprocess"
    return "empty"


_ROUTER_CLASSES = frozenset({"router_frozen", "router_taskgate"})


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard overlap of two task sets (1.0 when both empty — vacuously equal)."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


@dataclass
class WithinTaskDepthProfile:
    """Whether a candidate's chain is genuine within-task depth or router-stacking.

    ``multiplicity`` / ``max_multiplicity`` / ``multiplicity_hist`` are the raw
    seed-INCLUSIVE counts (a task's seed depth-1 fresh baseline counts toward its
    multiplicity). The genuine/verdict determination instead gates on the DEEP
    (seed-excluded) ``deep_multiplicity``: genuine within-task depth requires a task
    re-solved by >= 2 STACKED DEEP layers (persistence), not seed + one focus.
    """

    leaf_id: str
    chain_ids: list[str]
    max_chain_depth: int
    n_tasks: int
    # task -> sorted distinct depths at which it was FRESH-solved along the chain
    fresh_depths_by_task: dict[str, list[int]]
    multiplicity: dict[str, int]            # task -> # distinct layers that re-solved it (seed-INCLUSIVE raw)
    genuine_depth_tasks: list[str]          # deep_multiplicity >= 2 (>= 2 STACKED DEEP layers)
    genuine_fraction: float                 # |genuine| / n_tasks
    max_multiplicity: int                   # seed-INCLUSIVE raw max
    deep_multiplicity: dict[str, int]       # task -> # DEEP (non-seed) layers that re-solved it
    max_deep_multiplicity: int              # DEEP max; genuine/verdict gate on this
    multiplicity_hist: dict[int, int]       # multiplicity -> # tasks (seed-INCLUSIVE raw)
    mean_adjacent_overlap: float            # mean Jaccard of fresh-task sets, adjacent steps
    n_injected_layers: int
    layer_classes: list[str]                # classify_injection_layer per layer, shallow→deep
    router_fraction: float                  # routers / non-empty injected layers
    # T3.8: SUPPLY DUPLICATION, NOT adoption. helper name -> # distinct source_depths
    # the name was SHIPPED at. This is a static property of the injected code; it does
    # NOT read Trace.utilities_called, so it says nothing about whether the helper was
    # ever CALLED at runtime (the live code channel). See helper_adoption_depth below.
    helper_reuse_depth: dict[str, int]      # helper name -> # distinct source_depths SHIPPED
    max_helper_reuse: int                   # max supply duplication (names shipped at N depths)
    # T3.8: ACTUAL adoption — helper name -> # distinct trace depths where it was CALLED
    # (read from Trace.utilities_called across the chain). ``None`` when no trace
    # measured helper calls (utilities_called is None everywhere → unmeasurable).
    helper_adoption_depth: Optional[dict[str, int]] = None
    max_helper_adoption: Optional[int] = None
    fresh_split: list[DepthSplit] = field(default_factory=list)  # per-chain-candidate S0.4 split


def within_task_depth_profile(archive: Archive, leaf_id: str) -> WithinTaskDepthProfile:
    """Profile the ``leaf_id`` chain for genuine within-task depth (read-only).

    For every candidate on the seed→leaf chain, the set of FRESH-solved tasks
    (``trace.depth == candidate.depth``) is collected. A task's *multiplicity* is
    the number of DISTINCT chain depths at which it was fresh-solved — i.e. how
    many stacked layers actually acted on that SAME task. Multiplicity ``>= 2``
    means a deeper layer re-worked a task the layer below had already solved
    (genuine within-task recursion); multiplicity ``== 1`` means the task was
    solved once and then carried up frozen (router-stacking / inheritance).

    The injected layers along the chain are independently classified
    (router vs within-task). Two distinct helper signals are reported:
    ``helper_reuse_depth`` is SUPPLY DUPLICATION (how many depths each helper NAME
    was *shipped* at — a static property of the injected code), while
    ``helper_adoption_depth`` is ACTUAL adoption read from ``Trace.utilities_called``
    (how many depths each helper was *called* at runtime — the live code channel),
    or ``None`` when no trace measured helper calls.
    """
    chain = ancestry_chain(archive, leaf_id)
    leaf = chain[-1] if chain else archive.get(leaf_id)

    fresh_split = [split_candidate_mean(c) for c in chain]
    fresh_sets = [set(s.fresh_tasks) for s in fresh_split]

    fresh_depths: dict[str, set[int]] = {}
    for c, fresh in zip(chain, fresh_sets):
        for tid in fresh:
            fresh_depths.setdefault(tid, set()).add(c.depth)

    # Union of every task the chain ever touched (fresh OR carried in scores).
    all_tasks: set[str] = set(fresh_depths)
    all_tasks.update(leaf.per_task_scores)
    for tid in all_tasks:
        fresh_depths.setdefault(tid, set())

    multiplicity = {tid: len(ds) for tid, ds in fresh_depths.items()}
    # DEEP (seed-excluded) multiplicity: how many STACKED DEEP layers — those below
    # the seed's universal depth-1 fresh baseline — re-solved a task. Genuine
    # within-task depth (persistence) requires >= 2 of these, so a task carried by
    # the seed plus one deeper focus (deep multiplicity 1) does NOT count.
    seed_depth = chain[0].depth if chain else 1
    deep_multiplicity = {
        tid: sum(1 for d in ds if d > seed_depth) for tid, ds in fresh_depths.items()
    }
    max_deep_multiplicity = max(deep_multiplicity.values(), default=0)
    genuine = sorted(tid for tid, m in deep_multiplicity.items() if m >= 2)
    n_tasks = len(all_tasks)
    hist: dict[int, int] = {}
    for m in multiplicity.values():
        hist[m] = hist.get(m, 0) + 1

    overlaps = [
        _jaccard(fresh_sets[i], fresh_sets[i + 1]) for i in range(len(fresh_sets) - 1)
    ]
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 1.0

    # Injected-layer roles + helper reuse depth over the leaf's full chain.
    layer_classes = [classify_injection_layer(ic) for ic in leaf.injected_codes]
    non_empty = [c for c in layer_classes if c != "empty"]
    n_routers = sum(1 for c in layer_classes if c in _ROUTER_CLASSES)
    router_fraction = (n_routers / len(non_empty)) if non_empty else 0.0

    helper_depths: dict[str, set[int]] = {}
    for ic in leaf.injected_codes:
        for name in list(ic.code_library) + list(ic.code_library_bash):
            helper_depths.setdefault(name, set()).add(int(ic.source_depth))
    helper_reuse = {name: len(ds) for name, ds in helper_depths.items()}

    # T3.8: ACTUAL helper adoption from Trace.utilities_called across the chain —
    # how many distinct trace depths each helper was CALLED at (the live code
    # channel), as opposed to merely shipped. ``utilities_called is None`` marks an
    # unmeasured trace; if NO trace measured calls the adoption signal is None.
    adoption_depths: dict[str, set[int]] = {}
    measured = False
    for c in chain:
        for tr in c.traces:
            if tr.utilities_called is None:
                continue  # unmeasurable (no live helpers staged for this trace)
            measured = True
            for name in tr.utilities_called:
                adoption_depths.setdefault(name, set()).add(int(tr.depth))
    helper_adoption = (
        {name: len(ds) for name, ds in adoption_depths.items()} if measured else None
    )
    max_adoption = (
        max(helper_adoption.values(), default=0) if helper_adoption is not None else None
    )

    return WithinTaskDepthProfile(
        leaf_id=leaf.candidate_id,
        chain_ids=[c.candidate_id for c in chain],
        max_chain_depth=leaf.depth,
        n_tasks=n_tasks,
        fresh_depths_by_task={tid: sorted(ds) for tid, ds in fresh_depths.items()},
        multiplicity=multiplicity,
        genuine_depth_tasks=genuine,
        genuine_fraction=(len(genuine) / n_tasks) if n_tasks else 0.0,
        max_multiplicity=max(multiplicity.values(), default=0),
        deep_multiplicity=deep_multiplicity,
        max_deep_multiplicity=max_deep_multiplicity,
        multiplicity_hist=dict(sorted(hist.items())),
        mean_adjacent_overlap=mean_overlap,
        n_injected_layers=len(leaf.injected_codes),
        layer_classes=layer_classes,
        router_fraction=router_fraction,
        helper_reuse_depth=helper_reuse,
        max_helper_reuse=max(helper_reuse.values(), default=0),
        helper_adoption_depth=helper_adoption,
        max_helper_adoption=max_adoption,
        fresh_split=fresh_split,
    )


def within_task_verdict(
    profile: WithinTaskDepthProfile,
    *,
    min_chain_depth: int = 3,
    min_genuine_multiplicity: int = 2,
    max_router_fraction: float = 0.5,
) -> dict[str, object]:
    """PASS / FAIL / INCONCLUSIVE on the within-task-depth probe.

    * ``INCONCLUSIVE`` — the chain did not grow (``max_chain_depth <
      min_chain_depth``); the run has no deep candidate yet, so depth cannot be
      judged. Run more iterations / a viable cell.
    * ``FAIL (router-stacking)`` — routers dominate the chain
      (``router_fraction >= max_router_fraction``) OR no task was re-solved by
      ``>= min_genuine_multiplicity`` STACKED DEEP (non-seed) layers (every deep
      task touched once then frozen — depth is chain-length only). This is the
      CO-Bench artifact: Stage 2/3 have nothing genuine to refine / re-propagate.
    * ``PASS (genuine within-task depth)`` — the chain grew AND at least one task
      was re-solved by ``>= min_genuine_multiplicity`` STACKED DEEP (non-seed)
      layers AND routers do not dominate. A deep layer reasons about the layer
      below on the SAME task, so Stage 2 within-layer-refine and Stage 3 downward-
      re-propagation have real multi-layer within-task chains to operate on.
    """
    p = profile
    if p.max_chain_depth < min_chain_depth:
        verdict = "INCONCLUSIVE"
        reason = (
            f"chain depth {p.max_chain_depth} < {min_chain_depth}: no deep "
            f"candidate yet — run more iterations on a viable cell before judging depth."
        )
    elif p.router_fraction >= max_router_fraction:
        verdict = "FAIL"
        reason = (
            f"ROUTER-STACKING: router_fraction={p.router_fraction:.2f} >= "
            f"{max_router_fraction} ({p.layer_classes}) — the chain is disjoint "
            f"per-task routers/frozen inheritance (CO-Bench artifact), not within-task "
            f"recursion. Stage 2/3 have nothing genuine to refine/re-propagate."
        )
    elif p.max_deep_multiplicity < min_genuine_multiplicity:
        verdict = "FAIL"
        reason = (
            f"NO WITHIN-TASK DEPTH: max DEEP task multiplicity {p.max_deep_multiplicity} "
            f"< {min_genuine_multiplicity} — no task re-solved by >= {min_genuine_multiplicity} "
            f"STACKED DEEP layers; each deep task touched once = disjoint per-task focus / "
            f"router breadth, not within-task depth (depth {p.max_chain_depth} is "
            f"chain-length only). Stage 2/3 moot."
        )
    else:
        verdict = "PASS"
        # T3.8: report helper SUPPLY DUPLICATION (names shipped at N depths) and,
        # separately, ACTUAL adoption (helpers CALLED at runtime) — NOT framed as
        # "the code channel is live". Adoption is "n/a" when unmeasured.
        adoption_str = (
            "n/a (unmeasured)" if p.max_helper_adoption is None
            else str(p.max_helper_adoption)
        )
        reason = (
            f"GENUINE WITHIN-TASK DEPTH: depth {p.max_chain_depth}, "
            f"{len(p.genuine_depth_tasks)} task(s) re-solved by >= "
            f"{min_genuine_multiplicity} STACKED DEEP layers ({p.genuine_depth_tasks}), "
            f"router_fraction={p.router_fraction:.2f}, mean adjacent overlap "
            f"{p.mean_adjacent_overlap:.2f}, max helper supply duplication "
            f"{p.max_helper_reuse} (names shipped at N depths), max helper "
            f"adoption {adoption_str}. "
            f"Stage 2/3 have real multi-layer within-task chains to act on."
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "leaf_id": p.leaf_id,
        "max_chain_depth": p.max_chain_depth,
        "genuine_fraction": p.genuine_fraction,
        "max_multiplicity": p.max_multiplicity,          # seed-INCLUSIVE raw (compat)
        "max_deep_multiplicity": p.max_deep_multiplicity,  # DEEP; genuine/verdict gate
        "multiplicity_hist": p.multiplicity_hist,
        "router_fraction": p.router_fraction,
        "layer_classes": p.layer_classes,
        "mean_adjacent_overlap": p.mean_adjacent_overlap,
        "max_helper_reuse": p.max_helper_reuse,        # supply duplication (T3.8)
        "max_helper_adoption": p.max_helper_adoption,  # actual adoption / None (T3.8)
        "genuine_depth_tasks": p.genuine_depth_tasks,
    }


def deepest_leaf_id(archive: Archive) -> Optional[str]:
    """The candidate id with the greatest depth (ties → most children, then id).

    The natural probe target: the deepest chain the run produced. Returns ``None``
    for an empty archive.
    """
    best: Optional[Candidate] = None
    for c in archive.candidates:
        if best is None or (c.depth, c.num_children, c.candidate_id) > (
            best.depth, best.num_children, best.candidate_id
        ):
            best = c
    return best.candidate_id if best is not None else None


# --------------------------------------------------------------------------- #
# Disk entry point
# --------------------------------------------------------------------------- #
def load_archive(archive_dir: str | Path) -> Archive:
    """Rebuild an :class:`Archive` from an on-disk archive directory (read-only).

    R5: replays the checkpoint's fired verified-code bars (``barred_from_best``,
    checkpoint.json in the run dir or the archive dir itself) so the analysis
    per-task-best matches the run's — without it a barred trace can surface as
    best. The regression-guard floor is still not re-applied here (it needs
    the run's config; analysis stays config-free).
    """
    path = Path(archive_dir)
    barred = None
    for ck in (path.parent / "checkpoint.json", path / "checkpoint.json"):
        if ck.is_file():
            try:
                barred = json.loads(ck.read_text()).get("barred_from_best")
            except (OSError, ValueError):
                barred = None
            break
    return Archive.rebuild_from_disk(path, barred_from_best=barred)


# --------------------------------------------------------------------------- #
# CLI — depth-probe verdict over an on-disk archive
# --------------------------------------------------------------------------- #
# Exit codes (so a harness can branch on the verdict without parsing stdout):
#   0  PASS         — genuine within-task depth; Stage 2/3 have real chains.
#   1  FAIL         — router-stacking / no within-task recursion (CO-Bench-like).
#   2  INCONCLUSIVE — chain did not grow; run more iterations on a viable cell.
#   3  usage/IO error (empty archive, missing dir, bad leaf id).
_VERDICT_EXIT = {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}


def run_depth_probe(
    archive_dir: str | Path,
    *,
    leaf_id: Optional[str] = None,
    min_chain_depth: int = 3,
    min_genuine_multiplicity: int = 2,
    max_router_fraction: float = 0.5,
) -> dict[str, object]:
    """Load ``archive_dir``, profile a chain, return the verdict dict (read-only).

    ``leaf_id`` defaults to :func:`deepest_leaf_id` (the deepest chain the run
    produced — the natural probe target). The returned dict is the
    :func:`within_task_verdict` payload, augmented with ``archive_dir``,
    ``multiplicity`` (per-task), ``fresh_depths_by_task``, and ``n_tasks`` so the
    full profile is JSON-serialisable for downstream tooling.

    Raises ``ValueError`` for an empty archive or an unknown ``leaf_id``.
    """
    archive = load_archive(archive_dir)
    if not archive.candidates:
        raise ValueError(f"archive at {archive_dir} is empty (no candidates)")
    target = leaf_id or deepest_leaf_id(archive)
    if target is None:
        raise ValueError(f"archive at {archive_dir} has no probe-able leaf")
    # Surface an explicit error for an unknown id rather than a KeyError trace.
    try:
        archive.get(target)
    except KeyError as exc:
        raise ValueError(f"leaf id {target!r} not found in {archive_dir}") from exc

    profile = within_task_depth_profile(archive, target)
    verdict = within_task_verdict(
        profile,
        min_chain_depth=min_chain_depth,
        min_genuine_multiplicity=min_genuine_multiplicity,
        max_router_fraction=max_router_fraction,
    )
    verdict["archive_dir"] = str(archive_dir)
    verdict["n_tasks"] = profile.n_tasks
    verdict["multiplicity"] = profile.multiplicity            # seed-INCLUSIVE raw (compat)
    verdict["deep_multiplicity"] = profile.deep_multiplicity  # DEEP; genuine/verdict gate
    verdict["fresh_depths_by_task"] = profile.fresh_depths_by_task
    verdict["helper_reuse_depth"] = profile.helper_reuse_depth
    verdict["helper_adoption_depth"] = profile.helper_adoption_depth
    return verdict


def _format_probe_report(v: dict[str, object]) -> str:
    """Human-readable depth-probe report from a :func:`run_depth_probe` dict."""
    lines = [
        "═" * 70,
        f"DEPTH-PROBE  —  {v['archive_dir']}",
        "─" * 70,
        f"  leaf id              : {v['leaf_id']}",
        f"  max chain depth      : {v['max_chain_depth']}",
        f"  tasks in chain       : {v['n_tasks']}",
        f"  max multiplicity     : {v['max_multiplicity']}  "
        f"(# stacked layers acting on the SAME task, seed-INCLUSIVE raw)",
        f"  max deep multiplicity: {v.get('max_deep_multiplicity')}  "
        f"(# STACKED DEEP (non-seed) layers; genuine/verdict gate)",
        f"  multiplicity hist    : {v['multiplicity_hist']}",
        f"  genuine-depth tasks  : {v['genuine_depth_tasks']}",
        f"  router fraction      : {float(v['router_fraction']):.2f}",
        f"  layer roles          : {v['layer_classes']}",
        f"  mean adjacent overlap: {float(v['mean_adjacent_overlap']):.2f}",
        f"  helper supply dup    : {v['max_helper_reuse']}  "
        f"(max # depths a helper NAME was shipped at — NOT adoption)",
        f"  helper adoption      : {v.get('max_helper_adoption')}  "
        f"(max # depths a helper was CALLED; None=unmeasured)",
        "─" * 70,
        f"  VERDICT: {v['verdict']}",
        f"  {v['reason']}",
        "═" * 70,
    ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m meta_n.analysis.depth_attribution <archive_dir>`` (read-only).

    Reports whether the deepest chain in an on-disk archive is GENUINE within-task
    recursion (Stage 2/3 have real multi-layer chains to refine / re-propagate) or
    ROUTER-STACKING (the CO-Bench depth-5 artifact). Exit code encodes the verdict
    (see ``_VERDICT_EXIT``); ``--json`` prints the full machine-readable payload.
    """
    ap = argparse.ArgumentParser(
        prog="python -m meta_n.analysis.depth_attribution",
        description=(
            "Depth-probe: genuine within-task recursion vs router-stacking over an "
            "on-disk evolutionary archive (read-only). Exit 0=PASS, 1=FAIL, "
            "2=INCONCLUSIVE, 3=error."
        ),
    )
    ap.add_argument(
        "archive_dir",
        help="run archive directory (e.g. <output-dir>/run/archive)",
    )
    ap.add_argument(
        "--leaf-id",
        default=None,
        help="probe this candidate's chain (default: the deepest leaf)",
    )
    ap.add_argument("--min-chain-depth", type=int, default=3,
                    help="min chain depth to judge depth (default 3)")
    ap.add_argument("--min-genuine-multiplicity", type=int, default=2,
                    help="min same-task layer multiplicity for PASS (default 2)")
    ap.add_argument("--max-router-fraction", type=float, default=0.5,
                    help="router_fraction >= this is FAIL (default 0.5)")
    ap.add_argument("--json", action="store_true",
                    help="emit the full verdict payload as JSON to stdout")
    args = ap.parse_args(argv)

    try:
        verdict = run_depth_probe(
            args.archive_dir,
            leaf_id=args.leaf_id,
            min_chain_depth=args.min_chain_depth,
            min_genuine_multiplicity=args.min_genuine_multiplicity,
            max_router_fraction=args.max_router_fraction,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[depth-probe] error: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(verdict, indent=2, default=list))
    else:
        print(_format_probe_report(verdict))
    return _VERDICT_EXIT.get(str(verdict["verdict"]), 3)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
