"""Three-mode context reduction (task book v4.1 §0.1 / §2.3 / §2.13).

    official     the untouched ContextManager behaviour
    predictive   ContextReducer -> PredictiveSelector.reduce (§2.11)
    full         no official reduction: everything, bounded only by the model's
                 hard input limit

INSTALLATION COVERS **BOTH** INJECTION SITES. `OmegaEngine` reduces context in
two places -- `generate` (omega.py:119-120) and `refine` (omega.py:238-242) --
and patching only the first would leave the self-repair path silently running
unreduced, which is worse than not patching at all because the three modes
would look identical there.

Rather than editing `omega.py`, an adapter duck-types the two methods the
engine already calls:

    sampled    = engine.context_manager.sample_traces(traces)
    truncated  = engine.context_manager.truncate_context_stack(stack)

`predictive` selects the TRACES from the trace pool `U_v`; the injected-code
stack `C_v` is BYPASSED (frozen 2026-09-22) and handed to Omega by the same
native `truncate_context_stack` the baseline uses. So Gamma is not asked to
compress Omega's whole context -- only to choose which runtime FEEDBACK Omega
sees. The engine still hands the two halves in two separate calls with the
stack second, so the adapter returns a list it keeps a handle on and FINALISES
it in place on the stack call. A self-test drives both real sites to prove the
finalised object is what `_build_prompt` receives, and that both comparison
arms render the same four in the same order (see `canonical_order`).

`_build_prompt` is never touched, `C_v` is never appended, and the weighted
statistics stay in `diagnostics`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from meta_n.core.meta_layer import InjectedCode, Trace
from meta_n.rpbe.modes import ReductionMode
from meta_n.rpbe.selector import PredictiveSelector
from meta_n.utils.context_manager import ContextBudget, ContextManager

#: MATCHED-CAPACITY BASELINE (frozen 2026-09-21). The predictive arm carries at
#: most Gamma's ``n_slots`` objects in total over U_v = traces + context_stack.
#: The native OFFICIAL path instead caps traces and the stack under two
#: SEPARATE budgets, so from depth 3 on it carries its traces PLUS the entire
#: stack while predictive carries n_slots objects in total -- measured on a live
#: run: at d3 the official prompt held depth-2's injected code (1873 chars)
#: while the predictive prompt read "(none -- you are the first meta-layer)".
#: That changes the recursion mechanism itself, so the two arms were not
#: comparable. This baseline sends at most the same number of objects.
#:
#: The split folds the original section preference (traces_ratio 0.65 /
#: context_stack_ratio 0.35) onto that count: 3 traces + 1 stack item. Slots
#: left unused by one kind are offered to the other, so a pool with no stack
#: still yields 4 traces (which is what makes depth 2 identical across arms).
MATCHED_CONTEXT_OBJECTS = 4
MATCHED_K_TRACE = 3
MATCHED_K_STACK = 1

#: Trace budget for the MAIN comparison (frozen 2026-09-22). Both
#: `official_trace_k4` and `predictive` select exactly this many TRACES, and
#: both hand Omega the stack untouched by the same native
#: `truncate_context_stack`. The stack therefore neither enters Gamma nor
#: consumes one of these slots, which leaves exactly one variable between the
#: arms: WHICH four traces reach Omega.
TRACE_K4 = 4


def canonical_order(selected: Sequence[Any],
                    pool: Sequence[Any]) -> List[Any]:
    """Re-sort a selection into the POOL's own order (frozen 2026-09-23).

    Both comparison arms are supposed to differ in exactly ONE thing: which
    four traces they keep. Without this, they also differ in the ORDER they
    render, and that is not nothing -- `_format_raw_traces` emits one block per
    trace in list order, so two selections with identical MEMBERSHIP still
    produce different prompt bytes. Concretely the native sampler returns
    failures-then-successes while Gamma returns slot order, so
    `{t2,t5,t1,t3}` vs `{t1,t3,t5,t2}` reached Omega as different prompts.

    Ordering by the pool's original index removes that confound and makes the
    stated claim literally true. It changes nothing else: not the attention,
    not the selector, not the sampler, not k, not the token budget, and it
    requires no retraining.

    Identity-keyed on purpose: the sampler and the selector both return the
    very objects from the pool (they subset, never copy), so `id()` is exact.
    A fallback index is supplied for anything unexpected rather than raising,
    because a silent reorder is not worth a crashed formal run -- but it sorts
    last, so it cannot silently land in the middle of the pool order.
    """
    index = {}
    for i, item in enumerate(pool):
        index.setdefault(id(item), i)
    return sorted(selected, key=lambda x: index.get(id(x), len(pool)))


class _FinalisableList(list):
    """A list the adapter can rewrite in place once the stack arrives."""

    __slots__ = ("finalised", "owner")

    def __init__(self, items, owner=None):
        super().__init__(items)
        self.finalised = False
        self.owner = owner


class SiblingAllocator:
    """Deterministic sibling diversification (frozen 2026-09-23).

    THE PROBLEM IT SOLVES, MEASURED. For one parent, every sibling proposal
    reduces the IDENTICAL pool, and `q` is defined as the pool's rendering
    (encoder.py §2.8) -- so X and q_emb are identical across siblings. Gamma is a
    deterministic function, so every sibling received the SAME 4 traces. On the
    seed-0 run: 4 siblings -> 1 distinct subset, while the native arm samples
    randomly and gets subset exploration for free. That is a real structural
    asymmetry between the arms and a candidate mechanism for the observed shape
    (fast dev rise, early lineage peak, no matching held-out gain).

    WHAT IS DELIBERATELY NOT CHANGED. Gamma's scores, Gamma's own Top-k, the
    theory, the budget, k, and the native arm. The FIRST proposal for a given
    (pool, stack) gets EXACTLY the old selection, so the exploitation path that
    already worked is preserved byte-for-byte; only the other proposals for the
    same input are diversified. No temperature, no loss, no random draw --
    selection is a pure function of the ranked scores and what has already been
    assigned, so the arm stays fully reproducible.

    THE RULE. With Gamma's ranking r1..r6 and k=4:
      * the top 2 are PROTECTED -- every subset contains {r1, r2};
      * an exploration subset replaces exactly ONE of {r3, r4} with one of
        {r5, r6}, i.e. |S ∩ S_1| == k-1;
      * among the legal subsets, pick lexicographically
            max ( D(S), U(S) )
        where D(S) = sum over already-assigned S' of (k - |S ∩ S'|) and
        U(S) = sum of Gamma scores in S. Diversity first, Gamma utility only as
        the tie-break -- which is why no lambda has to be chosen.
    """

    def __init__(self):
        self._counts: Dict[Any, int] = {}
        self._assigned: Dict[Any, List[Tuple[int, ...]]] = {}

    @staticmethod
    def identity_key(traces: Sequence[Any],
                     context_stack: Sequence[Any]) -> Tuple[str, str]:
        """Stable key for "the same reduction input".

        Content-based, not identity-based, so two calls that see the same pool
        and the same stack land in the same group even across processes. That is
        the right grouping: the point is "these proposals share an input", which
        is exactly when collapsing to one subset is the defect.
        """
        import hashlib as _h

        def h(s):
            return _h.sha256(str(s).encode("utf-8", "replace")).hexdigest()[:16]

        pool = ";".join("%s|%s|%s|%s" % (getattr(t, "task_id", "?"),
                                        h(getattr(t, "script", "")),
                                        bool(getattr(t, "success", False)),
                                        getattr(t, "score", None))
                        for t in traces)
        stack = ";".join("%s|%s|%s" % (getattr(c, "source_depth", None),
                                       h(getattr(c, "pre_process", "")),
                                       h(getattr(c, "rationale", "")))
                         for c in context_stack)
        return pool, stack

    @staticmethod
    def _subsets(ranked: Sequence[int], k: int, n: int,
                 core_keep: int, overlap: int) -> List[Tuple[int, ...]]:
        """k-subsets keeping the top `core_keep` and overlapping S_1 by `overlap`."""
        import itertools
        if n <= k:
            return [tuple(ranked[:n])]
        core = set(ranked[:min(core_keep, k)])
        ref = set(ranked[:k])
        out = []
        for comb in itertools.combinations(ranked, k):
            S = set(comb)
            if not core <= S:
                continue
            if len(S & ref) != overlap:
                continue
            out.append(tuple(i for i in ranked if i in S))   # keep rank order
        return out

    def allocate(self, key, ranked: Sequence[int], k: int,
                 score_of, fits) -> Tuple[Tuple[int, ...], int, str]:
        """(subset as indices into the ranked pool, variant, kind).

        variant 0 is the untouched Top-k. `fits` guards the trace token budget;
        a subset that would overrun it is never offered.
        """
        n = len(ranked)
        variant = self._counts.get(key, 0)
        self._counts[key] = variant + 1
        top = tuple(ranked[:min(k, n)])
        if variant == 0 or n <= k:
            self._assigned.setdefault(key, []).append(top)
            return top, 0, "exploit"

        assigned = self._assigned.setdefault(key, [top])
        cands = self._subsets(ranked, k, n, core_keep=2, overlap=k - 1)
        cands = [c for c in cands if fits(c)]
        if not cands:                        # relax: allow two replacements
            cands = [c for c in self._subsets(ranked, k, n, core_keep=2,
                                              overlap=k - 2) if fits(c)]
        if not cands:                        # last resort: core only
            cands = [c for c in self._subsets(ranked, k, n, core_keep=2,
                                              overlap=0) if fits(c)]
        if not cands:
            self._assigned[key].append(top)
            return top, variant, "exploit"

        def D(S):
            return sum(k - len(set(S) & set(S2)) for S2 in assigned)

        def U(S):
            return sum(float(score_of[i]) for i in S)

        # Uniqueness first. D/U can tie across every remaining candidate -- with
        # a 6->4 pool and one replacement there are only four legal subsets, and
        # once three are assigned the rest tie on both. Without this the
        # lexicographic max re-picks an ALREADY-ASSIGNED subset and the arm
        # collapses again (measured: 4 siblings -> 3 unique instead of 4). Using
        # an unassigned subset whenever one exists is the deterministic reading
        # of "make the siblings as different as possible".
        unchosen = [c for c in cands if c not in assigned]
        best = max(unchosen or cands,
                   key=lambda S: (D(S), U(S), -ranked.index(S[0])))
        self._assigned[key].append(best)
        return best, variant, "explore"


#: Allocator instance shared by a run. Module-level on purpose: the reducer is
#: constructed once per process (main.py), but a fresh ContextReducer built by a
#: harness must NOT silently restart the variant sequence mid-run.
_SIBLING_ALLOCATOR = SiblingAllocator()


class ContextReducer:
    """Mode-aware reducer. Pure function of its inputs; holds no state."""

    def __init__(self, mode: ReductionMode | str = ReductionMode.OFFICIAL,
                 *, encoder: Any = None, fusion: Any = None,
                 selector: Optional[PredictiveSelector] = None,
                 context_manager: Optional[ContextManager] = None):
        self.mode = ReductionMode(mode) if not isinstance(
            mode, ReductionMode) else mode
        self.encoder = encoder
        self.fusion = fusion
        self.selector = selector or PredictiveSelector()
        self.context_manager = context_manager or ContextManager()

    def reduce(self, traces: Sequence[Trace], context_stack: Sequence[InjectedCode],
               *, budget: Optional[ContextBudget] = None,
               current_traces: Optional[Sequence[Trace]] = None
               ) -> Tuple[List[Trace], List[InjectedCode], Dict[str, Any]]:
        budget = budget or self.context_manager.budget
        if self.mode is ReductionMode.OFFICIAL:
            return self._official(traces, context_stack)
        if self.mode is ReductionMode.OFFICIAL_TRACE_K4:
            return self._official_trace_k4(traces, context_stack)
        if self.mode is ReductionMode.MATCHED_K4:
            return self._matched_k4(traces, context_stack)
        if self.mode is ReductionMode.FULL:
            return self._full(traces, context_stack)
        return self._predictive(traces, context_stack, budget, current_traces)

    # -- official: byte-identical to the untouched engine path -------------
    def _official(self, traces, context_stack):
        sampled = self.context_manager.sample_traces(list(traces))
        stack = self.context_manager.truncate_context_stack(list(context_stack))
        return (list(sampled), list(stack),
                dict(self._budget_diag("official", traces, sampled,
                                       context_stack, stack)))

    # -- official_trace_k4: THE MAIN BASELINE (frozen 2026-09-22) ----------
    def _official_trace_k4(self, traces, context_stack):
        """k=4 heuristic traces + the native stack. Paired with `predictive`.

        Structurally the native `_official`, with TWO differences:

        * the trace sampler is capped at TRACE_K4 instead of its own default;
        * the survivors are re-sorted into the pool's order (canonical_order),
          so the arm differs from `predictive` in MEMBERSHIP only.

        The stack goes through the identical `truncate_context_stack` call, so
        the two arms see the same stack. The native rule itself is untouched --
        it is `official` mode, and it is what the shared root runs.
        """
        sampled = self.context_manager.sample_traces(
            list(traces), max_total=TRACE_K4)
        sampled = canonical_order(list(sampled), traces)
        stack = self.context_manager.truncate_context_stack(list(context_stack))
        diag = dict(self._budget_diag("official_trace_k4", traces, sampled,
                                      context_stack, stack))
        # RECORD ONLY -- the selection above is untouched. The baseline had no
        # selection record either, which is why "what did the native rule keep on
        # this call?" was unanswerable below depth 2 in the seed-0 analysis.
        diag["allocator"] = "native_sampling"
        diag["pool_trace_ids"] = [getattr(t, "task_id", "?") for t in traces]
        diag["selected_trace_ids"] = [getattr(t, "task_id", "?")
                                      for t in sampled]
        diag["pool_hash"], diag["stack_hash"] = \
            SiblingAllocator.identity_key(list(traces), list(context_stack))
        return (list(sampled), list(stack), diag)

    # -- matched_k4: the matched-CAPACITY baseline (see the constants above) --
    def _matched_k4(self, traces, context_stack):
        total, cap_t, cap_s = (MATCHED_CONTEXT_OBJECTS, MATCHED_K_TRACE,
                               MATCHED_K_STACK)
        # The sampler is handed the JOINT total so it can never return more
        # than could be used; the per-kind cap is applied below.
        stack_all = list(
            self.context_manager.truncate_context_stack(list(context_stack)))
        trace_all = list(
            self.context_manager.sample_traces(list(traces), max_total=total))

        stack = stack_all[:cap_s]
        sel = trace_all[:cap_t]
        # Slots one kind leaves unused are offered to the other. This is what
        # keeps depth 2 identical across arms: with an empty stack the trace
        # side takes all four.
        slack = total - len(stack) - len(sel)
        if slack > 0 and len(stack) < len(stack_all):
            add = min(slack, len(stack_all) - len(stack))
            stack = stack + stack_all[len(stack):len(stack) + add]
            slack -= add
        if slack > 0 and len(sel) < len(trace_all):
            add = min(slack, len(trace_all) - len(sel))
            sel = sel + trace_all[len(sel):len(sel) + add]

        return (sel, stack,
                dict(self._budget_diag("matched_k4", traces, sel,
                                       context_stack, stack)))

    def _budget_diag(self, mode, traces, sel, context_stack, stack):
        """Per-call capacity instrumentation, identical in shape for every mode.

        Recording n_traces_out / n_stack_out / objects / tokens for BOTH arms is
        the only way to prove the matched-capacity claim after the fact; without
        it a capacity mismatch is invisible in the artifacts.
        """
        def toks(items, is_code):
            name = ("_estimate_injected_code_tokens" if is_code
                    else "_estimate_trace_tokens")
            f = getattr(self.context_manager, name, None)
            if f is None:
                return None
            try:
                return int(sum(f(x) for x in items))
            except Exception:                                 # noqa: BLE001
                return None
        t_tok, s_tok = toks(sel, False), toks(stack, True)
        rec = {"mode": mode,
               "n_traces_in": len(traces), "n_traces_out": len(sel),
               "n_stack_in": len(context_stack), "n_stack_out": len(stack),
               "n_objects_out": len(sel) + len(stack),
               "trace_tokens_out": t_tok, "stack_tokens_out": s_tok,
               "total_tokens_out": (None if t_tok is None or s_tok is None
                                    else t_tok + s_tok)}
        # CAPACITY KEYS ARE MODE-SPECIFIC, deliberately. `matched_k4` caps the
        # TOTAL object count at 4 (traces + stack); the trace-only modes cap
        # only the TRACES and let the stack through untouched, so their total
        # is `4 traces + the whole native stack` and can exceed 4. Writing
        # `matched_context_objects: 4` for a trace-only mode would invite a
        # downstream claim that Omega saw four objects in total, which is
        # false for the main comparison -- so each mode declares its own cap.
        if mode == "matched_k4":
            rec["matched_context_objects"] = MATCHED_CONTEXT_OBJECTS
            rec["matched_k_trace"] = MATCHED_K_TRACE
            rec["matched_k_stack"] = MATCHED_K_STACK
        elif mode in ("official_trace_k4", "predictive"):
            rec["trace_cap"] = TRACE_K4
            rec["stack_cap"] = None          # native truncation, not a k
        return rec

    # -- full: nothing reduced; the model's hard limit is the only bound ---
    def _full(self, traces, context_stack):
        return (list(traces), list(context_stack),
                {"mode": "full", "n_traces_in": len(traces),
                 "n_traces_out": len(traces),
                 "n_stack_in": len(context_stack),
                 "n_stack_out": len(context_stack),
                 "reduced": False})

    # -- predictive: encode -> fuse -> attention -> selector ---------------
    def _predictive(self, traces, context_stack, budget, current_traces):
        """Gamma selects the TRACES; the stack is BYPASSED (frozen 2026-09-22).

        Gamma's job is `trace pool -> 4 traces`. The injected-code stack does
        NOT enter the encoder, does NOT compete for the 4 slots, and is handed to
        Omega by the SAME native `truncate_context_stack` the baseline uses. So
        the only variable between this arm and `official_trace_k4` is WHICH four
        traces reach Omega.

        It used to encode `traces + context_stack` and let both compete for the
        same slots. Measured on a live run: Gamma then spent all 4 slots on
        traces and dropped the stack in 18/18 opportunities, while the hand rule
        kept it 20/20 -- Gamma had been made responsible for compressing Omega's
        entire context instead of choosing Omega's runtime feedback.
        """
        if self.encoder is None or self.fusion is None:
            raise RuntimeError(
                "predictive mode needs the frozen encoder and Gamma; "
                "construct ContextReducer(encoder=..., fusion=...)")
        trace_items: List[Any] = list(traces)
        X = self.encoder.encode(trace_items)
        q_src = (list(current_traces) if current_traces is not None
                 else trace_items)
        q_text = self.encoder.serialize_query(q_src)
        q_emb = self.encoder.encode_query(q_text)
        _, attention = self.fusion(X, q_emb, None)
        # Empty stack on purpose: the selector ranks traces only.
        sel_t, _, diag = self.selector.reduce(
            trace_items, [], attention, budget)

        # ---- sibling diversification (frozen 2026-09-23) -------------------
        # Variant 0 is the selector's own output, UNCHANGED, so the exploitation
        # path is byte-identical to the pre-change arm. Only the other proposals
        # sharing this SAME input get a diversified subset; see SiblingAllocator
        # for why identical inputs used to force identical selections.
        import torch as _torch
        n_items = len(trace_items)
        k = min(TRACE_K4, n_items)
        mass = attention.detach().to(_torch.float64).sum(dim=0)
        # `ranked[pos]` is the pool index of the pos-th best trace; every score
        # and subset below is expressed in RANK POSITIONS, which is what the
        # allocator's diversity/utility arithmetic needs.
        ranked = sorted(range(n_items), key=lambda i: (-float(mass[i]), i))
        score_by_pos = {pos: float(mass[ranked[pos]]) for pos in range(n_items)}
        est = getattr(self.context_manager, "_estimate_trace_tokens", None)

        def fits(subset):
            """Subsets are rank positions; guard the trace token budget."""
            if est is None or budget is None:
                return True
            try:
                return (sum(est(trace_items[ranked[pos]]) for pos in subset)
                        <= budget.traces_budget)
            except Exception:                                 # noqa: BLE001
                return True

        key = SiblingAllocator.identity_key(trace_items, context_stack)
        # RANK-POSITION space, deliberately. `ranked[0]` is the best POOL index,
        # `ranked[1]` the next, etc. The allocator must work on positions so that
        # one index space is used throughout: its `score_of` is position-keyed,
        # and `sel_t` below maps a position back through `ranked`. The first
        # draft passed `ranked` itself, so the returned "positions" were pool
        # indices and were then re-indexed through `ranked` again -- a live run
        # showed variant 1 producing the SAME subset as variant 0.
        picked, variant, kind = _SIBLING_ALLOCATOR.allocate(
            key, list(range(n_items)), k, score_by_pos, fits)
        if variant != 0:
            sel_t = [trace_items[ranked[pos]] for pos in picked]

        # Same membership-only contract as the baseline: Gamma returns SLOT
        # order, so re-sort both arms into the pool's order or the two prompts
        # would differ in rendering order as well as in membership.
        sel_t = canonical_order(list(sel_t), traces)
        # The stack, by the native rule -- byte-identical to the baseline's.
        sel_s = list(
            self.context_manager.truncate_context_stack(list(context_stack)))
        diag["mode"] = "predictive"
        # Selection-trajectory record. Without this the 4-of-N choice is not
        # persisted anywhere (no trace ids in proposal_slots.jsonl), which is why
        # the seed-0 run could not be analysed below depth 2 at all.
        diag["allocator"] = "sibling_diverse_v1"
        diag["allocator_variant"] = int(variant)
        diag["allocator_kind"] = kind
        diag["pool_hash"], diag["stack_hash"] = key
        diag["pool_trace_ids"] = [getattr(t, "task_id", "?") for t in trace_items]
        diag["gamma_rank"] = [getattr(trace_items[i], "task_id", "?")
                              for i in ranked]
        diag["gamma_scores"] = [round(float(mass[i]), 8) for i in ranked]
        diag["selected_trace_ids"] = [getattr(t, "task_id", "?") for t in sel_t]
        # Same instrumentation shape as every other mode, so the two arms' per-call
        # records can be compared field by field (see _budget_diag).
        diag.update(self._budget_diag("predictive", traces, sel_t,
                                      context_stack, sel_s))
        return (sel_t, sel_s, diag)


class PredictiveContextManagerAdapter:
    """Duck-types `ContextManager` so `omega.py` needs NO edits.

    `sample_traces` hands back a finalisable list; `truncate_context_stack`
    performs the joint reduction and rewrites that SAME object in place, so
    whatever the engine later passes to `_build_prompt` is the reduced set.
    """

    def __init__(self, reducer: ContextReducer,
                 base: Optional[ContextManager] = None) -> None:
        self.reducer = reducer
        self.base = base or ContextManager()
        self.last_diagnostics: Dict[str, Any] = {}
        self._pending = None

    # -- the sampler's state lives on `base`, and is written THROUGH us -----
    # `install` swaps this adapter into `engine.context_manager`, and only
    # afterwards does the orchestrator write the search seed:
    #
    #     omega.context_manager.rng = self.rng      (evolutionary_orchestrator:552)
    #     omega.context_manager.rng = ...           (run_persistence:102/123, both
    #                                                --resume paths)
    #
    # That assignment now lands on the ADAPTER, while the reducer samples from
    # `base`. Without forwarding, the seeded rng never reaches the trace
    # sampler: provenance would record a search seed the run never used, and
    # two seeds would produce byte-identical trace pools. Delegate rather than
    # copy, so a state restore (`Random.setstate`) also lands on the live
    # object across --resume.
    @property
    def rng(self):
        return self.base.rng

    @rng.setter
    def rng(self, value):
        self.base.rng = value

    @property
    def symmetric_sampling(self):
        return self.base.symmetric_sampling

    @symmetric_sampling.setter
    def symmetric_sampling(self, value):
        self.base.symmetric_sampling = value

    @property
    def budget(self):
        return self.base.budget

    @budget.setter
    def budget(self, value):
        self.base.budget = value

    def __getattr__(self, name):
        # Anything else the engine reaches for (the token estimators, anything
        # added later) is the base manager's, not ours. `base` itself is
        # guarded: a miss during __init__ would otherwise recurse forever.
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    # called FIRST
    def sample_traces(self, traces):
        if self.reducer.mode is ReductionMode.OFFICIAL:
            return self.base.sample_traces(traces)
        holder = _FinalisableList(list(traces), owner=self)
        self._pending = holder
        return holder

    # called SECOND -- finalises the holder in place
    def truncate_context_stack(self, context_stack):
        if self.reducer.mode is ReductionMode.OFFICIAL:
            return self.base.truncate_context_stack(context_stack)
        pending = self._pending
        if pending is None:
            raise RuntimeError(
                "truncate_context_stack called before sample_traces; the "
                "adapter relies on that order to see the full U_v")
        sel_t, sel_s, diag = self.reducer.reduce(
            list(pending), list(context_stack))
        self.last_diagnostics = diag
        _append_trace(diag)
        # In-place finalisation: `pending` is the very object the engine still
        # holds and will hand to _build_prompt.
        pending[:] = list(sel_t)
        pending.finalised = True
        self._pending = None
        return list(sel_s)


TRACE_ENV = "META_N_REDUCTION_TRACE"

#: Monotonic counter over every reduction this process performs. It is what
#: lets `reduction_trace.jsonl` be joined back to `proposal_slots.jsonl` (which
#: carries slot_id / parent_id / depth but no trace ids): the reducer cannot see
#: the candidate identity, so ORDER is the join key.
_REDUCTION_SEQ = [0]


def _append_trace(diag: Dict[str, Any]) -> None:
    """Append one reduction's diagnostics to a JSONL if requested.

    Used by the three-mode E2E to prove what each mode actually did, without
    changing any engine behaviour. Off unless META_N_REDUCTION_TRACE is set.

    Carries the whole SELECTION TRAJECTORY: pool identity, the pool's content
    hash (so siblings sharing an input are recognisable), Gamma's raw scores and
    rank, the allocator variant, and the chosen trace ids. Without this the
    4-of-N choice is persisted nowhere and depths >= 3 cannot be analysed at all
    -- which is exactly what happened on the seed-0 run.
    """
    import json
    import os
    import time
    path = os.environ.get(TRACE_ENV, "").strip()
    if not path:
        return
    try:
        rec = dict(diag)
        _REDUCTION_SEQ[0] += 1
        rec["reduction_seq"] = _REDUCTION_SEQ[0]
        rec["ts"] = time.time()
        rec["pid"] = os.getpid()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass


def install(engine, reducer: ContextReducer) -> "PredictiveContextManagerAdapter":
    """Replace an engine's context manager with the mode-aware adapter.

    Covers `generate` AND `refine`: both call the same two methods on
    `engine.context_manager`.
    """
    adapter = PredictiveContextManagerAdapter(reducer, base=engine.context_manager)
    engine.context_manager = adapter
    return adapter


# --------------------------------------------------------------------------
# acceptance self-test
# --------------------------------------------------------------------------

class _StubEncoder:
    """Offline stand-in: text -> [256], deterministic."""

    def __init__(self, dim=256):
        self.dim = dim

    def serialize_query(self, traces):
        return "|".join(getattr(t, "task_id", "?") for t in traces)

    def encode_query(self, text):
        import hashlib
        import torch
        h = hashlib.sha256(text.encode()).digest()
        g = torch.Generator().manual_seed(int.from_bytes(h[:4], "little"))
        return torch.randn(self.dim, generator=g)

    def encode(self, items):
        import torch
        rows = [self.encode_query(getattr(i, "task_id", "") or
                                  getattr(i, "rationale", "")) for i in items]
        return torch.stack(rows, 0)


class _StubFusion:
    def __init__(self, n_slots=4):
        self.n_slots = n_slots

    def __call__(self, X, q_emb, mask):
        import torch
        n = X.shape[0]
        A = torch.zeros(self.n_slots, n)
        # slot k prefers item k (deterministic, easy to assert)
        for k in range(self.n_slots):
            if k < n:
                A[k, k] = 1.0
            else:
                A[k, -1] = 1.0
        Z = torch.zeros(self.n_slots, X.shape[1])
        return Z, A


class _StubFusionRev:
    """Like `_StubFusion` but with a NON-identity preference order.

    WHY THIS EXISTS. The default stub's slot k prefers item k, so the score
    ranking equals the pool order -- which makes "pool index" and "rank position"
    the SAME NUMBER, and that hid a double-indexing bug in the allocator: the
    first draft passed pool indices where positions were expected, and a LIVE run
    caught it (variant 1 produced the same subset as variant 0) while the
    self-test stayed green. Reversing the preference separates the two index
    spaces so the check actually bites.
    """

    def __init__(self, n_slots=4):
        self.n_slots = n_slots

    def __call__(self, X, q_emb, mask):
        import torch
        n = X.shape[0]
        A = torch.zeros(self.n_slots, n)
        for k in range(self.n_slots):
            A[k, n - 1 - k] = 1.0            # prefer the LAST items first
        return torch.zeros(self.n_slots, X.shape[1]), A


def self_test() -> int:
    import os
    import random
    from meta_n.core.meta_layer import InjectedCode, Trace
    from meta_n.utils.context_manager import ContextBudget

    print("context_reduction.py acceptance")
    print("=" * 68)

    # Six traces, not three: the selector picks up to `n_slots` (=4) items and
    # forbids repeats, so a 3-trace pool could never satisfy the old
    # `n_selected == 4` assertion -- that assertion was written for the retired
    # "traces + stack share four slots" interface and is exactly the kind of
    # stale check that lets an interface change land half-wired.
    traces = [Trace(task_id="t{}".format(i), script="s", stdout="o" * 100,
                    success=True) for i in range(6)]
    stack = [InjectedCode(pre_process="def p(): pass", rationale="r",
                          code_library={"f": "def f(): return 1"},
                          source_depth=d) for d in (2, 5, 3)]
    budget = ContextBudget()

    # OFFICIAL is a pure delegate
    r_off = ContextReducer(ReductionMode.OFFICIAL)
    t_o, s_o, d_o = r_off.reduce(traces, stack, budget=budget)
    assert d_o["mode"] == "official"
    print("OK  official           delegates to ContextManager verbatim "
          "({} traces -> {}, {} stack -> {})".format(
              len(traces), len(t_o), len(stack), len(s_o)))

    # FULL reduces nothing
    r_full = ContextReducer(ReductionMode.FULL)
    t_f, s_f, d_f = r_full.reduce(traces, stack, budget=budget)
    assert len(t_f) == len(traces) and len(s_f) == len(stack)
    assert d_f["reduced"] is False
    print("OK  full               nothing reduced ({} traces, {} stack), "
          "bounded only by the model limit".format(len(t_f), len(s_f)))

    # PREDICTIVE goes through the selector -- over the TRACES ONLY. The stack is
    # BYPASSED and re-attached by the native truncation, so it must come back
    # unchanged and must not have consumed one of the four trace slots.
    r_pred = ContextReducer(ReductionMode.PREDICTIVE, encoder=_StubEncoder(),
                            fusion=_StubFusion())
    t_p, s_p, d_p = r_pred.reduce(traces, stack, budget=budget)
    assert d_p["mode"] == "predictive"
    assert d_p["n_traces_out"] == 4, d_p
    assert d_p["n_stack_out"] == len(stack), d_p
    for t in t_p:
        assert any(t is x for x in traces)
    for c in s_p:
        assert any(c is x for x in stack)
    print("OK  predictive         selected {} of {} traces; the {} stack "
          "item(s) bypassed Gamma untouched".format(
              d_p["n_traces_out"], len(traces), d_p["n_stack_out"]))

    # ...and the main baseline differs ONLY in how those four are chosen
    r_k4 = ContextReducer(ReductionMode.OFFICIAL_TRACE_K4)
    t_k, s_k, d_k = r_k4.reduce(traces, stack, budget=budget)
    assert d_k["mode"] == "official_trace_k4"
    assert d_k["n_traces_out"] == 4, d_k
    assert s_k == s_p, "the two arms must receive the SAME stack"
    print("OK  official_trace_k4  {} traces + an identical stack -> the only "
          "variable is which four".format(d_k["n_traces_out"]))

    # ...and "which four" must not silently include "in what order". The native
    # sampler returns failures-then-successes and Gamma returns slot order, so
    # with an INTERLEAVED pool the two arms would render the same trace set in
    # different orders -- different prompt bytes from identical membership.
    # `canonical_order` re-sorts both arms into the pool's order.
    mixed = [Trace(task_id="m{}".format(i), script="s", stdout="o" * 100,
                   success=(i % 2 == 0)) for i in range(6)]
    pos = {id(t): i for i, t in enumerate(mixed)}

    r_k4m = ContextReducer(ReductionMode.OFFICIAL_TRACE_K4)
    t_k4m, _, d4m = r_k4m.reduce(mixed, [], budget=budget)
    got_k4 = [pos[id(t)] for t in t_k4m]
    assert len(t_k4m) == 4, d4m
    assert got_k4 == sorted(got_k4), (
        "official_trace_k4 must hand the traces in POOL order; got pool "
        "positions {}".format(got_k4))

    t_pm, _, _ = r_pred.reduce(mixed, [], budget=budget)
    got_p = [pos[id(t)] for t in t_pm]
    assert got_p == sorted(got_p), (
        "predictive must hand the traces in POOL order; got pool positions "
        "{}".format(got_p))

    # Guard the guard: prove the raw sampler really does emit a DIFFERENT
    # order, so the two assertions above are testing something rather than
    # passing by luck. This is the 6-trace interleaved pool, so the native
    # "failures then successes" output is not the pool order.
    raw = ContextManager().sample_traces(list(mixed), max_total=4)
    raw_pos = [pos[id(t)] for t in raw]
    assert raw_pos != sorted(raw_pos), (
        "the raw sampler already returned pool order; this pool no longer "
        "discriminates and the check above is vacuous: {}".format(raw_pos))
    print("OK  canonical order    both arms hand the pool order "
          "(k4 {}, predictive {}); the raw sampler emitted {} first, so the "
          "check is not vacuous".format(got_k4, got_p, raw_pos))

    # --- sibling diversification: the four frozen acceptance criteria ------
    # The defect this fixes: every sibling of one parent reduces the IDENTICAL
    # pool with an identical q, and Gamma is deterministic, so all of them used
    # to receive the same 4 traces (measured on seed 0: 4 siblings -> 1 distinct
    # subset). The native arm samples randomly and gets subset exploration for
    # free; the Gamma arm had none.
    _SIBLING_ALLOCATOR._counts.clear()
    _SIBLING_ALLOCATOR._assigned.clear()
    pool6 = [Trace(task_id="s{}".format(i), script="s", stdout="o" * 100,
                   success=True) for i in range(6)]
    r_div = ContextReducer(ReductionMode.PREDICTIVE, encoder=_StubEncoder(),
                           fusion=_StubFusion())

    def draw_four(capture=None):
        out = []
        for _ in range(4):
            t, _, dd = r_div.reduce(list(pool6), [], budget=budget)
            if capture is not None and not capture:
                capture.append(dd)
            out.append(tuple(sorted(x.task_id for x in t)))
        return out

    cap = []
    first = draw_four(cap)
    d0 = cap[0]
    # The untouched Top-k, computed by calling the SELECTOR directly. Going
    # through `reduce` would consume another allocator variant and compare
    # variant 0 against variant 4.
    _X = r_pred.encoder.encode(pool6)
    _q = r_pred.encoder.encode_query(r_pred.encoder.serialize_query(pool6))
    _att = r_pred.fusion(_X, _q, None)[1]
    _raw, _, _ = r_pred.selector.reduce(list(pool6), [], _att, budget)
    base_ids = tuple(sorted(t.task_id for t in _raw))

    # A. behaviour preserved: variant 0 IS the untouched Top-k
    assert first[0] == base_ids, (first[0], base_ids)
    assert d0["allocator_variant"] == 0 and d0["allocator_kind"] == "exploit"
    print("OK  A behaviour kept   variant 0 == the untouched Gamma Top-k {}"
          .format(",".join(first[0])))

    # B. diversity restored
    uniq = sorted(set(first))
    union = sorted({t for s in first for t in s})
    assert len(uniq) == 4, uniq
    assert all(len(s) == 4 for s in first), first
    core = set(d0["gamma_rank"][:2])          # recorded by _predictive
    for s in first:
        assert core <= set(s), (s, core)
    print("OK  B diversity back   4 siblings -> {} unique subsets; union={} "
          "(of 6); Gamma top-2 {} kept in every subset".format(
              len(uniq), len(union), sorted(core)))

    # C. budget untouched: still exactly k=4 traces per proposal
    assert TRACE_K4 == 4
    print("OK  C budget same      every variant keeps exactly k={} traces"
          .format(TRACE_K4))

    # B'. the same check under a NON-identity ranking, so pool index and rank
    # position are different numbers. Without this the two index spaces coincide
    # and a double-indexing bug in the allocator passes silently (it did).
    _SIBLING_ALLOCATOR._counts.clear()
    _SIBLING_ALLOCATOR._assigned.clear()
    r_rev = ContextReducer(ReductionMode.PREDICTIVE, encoder=_StubEncoder(),
                           fusion=_StubFusionRev())
    rev_cap = []
    rev = []
    for _ in range(4):
        t, _, dd = r_rev.reduce(list(pool6), [], budget=budget)
        rev_cap.append(dd)
        rev.append(tuple(sorted(x.task_id for x in t)))
    rk = rev_cap[0]["gamma_rank"]
    assert rk != ["s%d" % i for i in range(6)], (
        "this stub must produce a NON-identity ranking, or the two index spaces "
        "coincide and the check below is vacuous: %s" % rk)
    assert len(set(rev)) == 4, rev
    assert all(len(s) == 4 for s in rev), rev
    for s in rev:
        assert set(rk[:2]) <= set(s), (s, rk, rev)
    print("OK  B' non-identity     same 4-unique result when the ranking is NOT "
          "the pool order (the index-space check that the flat stub hid)")

    # D. reproducible: same input, fresh state -> identical selections
    _SIBLING_ALLOCATOR._counts.clear()
    _SIBLING_ALLOCATOR._assigned.clear()
    assert draw_four() == first
    print("OK  D reproducible     same pool re-run gives byte-identical subsets")

    # C_v IS NEVER APPENDED: the reducer's output carries no tasks/scores
    assert not hasattr(t_p, "tasks") and "tasks" not in d_p
    print("OK  no C_v appended    reducer output is traces+stack only")

    # --- the adapter covers BOTH engine sites -----------------------------
    from meta_n.core.omega import OmegaEngine
    eng = object.__new__(OmegaEngine)
    eng.context_manager = ContextManager()
    engine_manager = eng.context_manager
    # Constructed exactly as main.py does: with the ENGINE's manager. Building
    # the reducer without one silently gives it a fresh ContextManager(), whose
    # defaults replace the engine's: `symmetric_sampling` reverts to False, the
    # sampler reads the module-level `random` instead of the search seed, and
    # the token estimators / stack-truncation rule change under it. Nothing in
    # config.json can see that -- it records keys, not live objects.
    r_pred_shared = ContextReducer(ReductionMode.PREDICTIVE,
                                   encoder=_StubEncoder(), fusion=_StubFusion(),
                                   context_manager=engine_manager)
    adapter = install(eng, r_pred_shared)
    assert adapter.base is engine_manager
    assert r_pred_shared.context_manager is engine_manager, (
        "the reducer and the engine must share ONE ContextManager")
    # `install` leaves the adapter in the engine slot, and the orchestrator
    # then writes the seed THROUGH that slot (evolutionary_orchestrator:552 and
    # both --resume paths in run_persistence). The reducer samples from
    # `engine_manager`, so the adapter has to forward -- otherwise the recorded
    # search_seed never reached the sampler.
    seed = random.Random(7)
    eng.context_manager.rng = seed
    assert engine_manager.rng is seed, (
        "the search seed must reach the sampler the reducer actually uses")
    assert r_pred_shared.context_manager.rng is seed
    eng.context_manager.symmetric_sampling = True
    assert engine_manager.symmetric_sampling is True, (
        "the profile's symmetric_sampling must reach the sampler")
    assert eng.context_manager.budget is engine_manager.budget
    print("OK  wiring             reducer + engine share ONE ContextManager, "
          "and the orchestrator's writes reach the sampler through the adapter")

    # ...and the forwarded seed is the one the sampler actually CONSUMES. This
    # is the only check that can tell "stored" from "used": store the same seed
    # on a bare ContextManager and the sampled SET must come out identical
    # through the full engine call sequence. Membership, not sequence, because
    # `canonical_order` then re-sorts the survivors into the pool's order --
    # which the second assertion pins separately.
    def k4_draw(seed_value):
        e = object.__new__(OmegaEngine)
        e.context_manager = ContextManager(symmetric_sampling=True)
        r = ContextReducer(ReductionMode.OFFICIAL_TRACE_K4,
                           context_manager=e.context_manager)
        install(e, r)
        e.context_manager.rng = random.Random(seed_value)
        holder = e.context_manager.sample_traces(list(traces))
        e.context_manager.truncate_context_stack(list(stack))
        return [t.task_id for t in holder]

    def raw_rule(seed_value):
        return [t.task_id for t in random.Random(seed_value).sample(
            [t for t in traces if t.success], 4)]

    drawn, drawn8 = k4_draw(7), k4_draw(8)
    assert sorted(drawn) == sorted(raw_rule(7)), (drawn, raw_rule(7))
    assert sorted(drawn8) == sorted(raw_rule(8)), (drawn8, raw_rule(8))
    # Pool order, explicitly: the arm must emit the pool's own sequence.
    pool_ids = [t.task_id for t in traces]
    assert drawn == [i for i in pool_ids if i in set(drawn)], drawn
    print("OK  seed reaches sampler the four drawn traces are exactly what "
          "random.Random(7) samples ({}), in pool order".format(",".join(drawn)))

    # site 1 shape (generate): sample then truncate, then read the holder
    holder = eng.context_manager.sample_traces(traces)
    returned_stack = eng.context_manager.truncate_context_stack(stack)
    assert holder.finalised, "holder was not finalised"
    # The stack is BYPASSED, so it does NOT consume a trace slot: the holder
    # carries exactly the selected traces, not (slots - stack items).
    assert len(holder) == d_p["n_traces_out"], (len(holder), d_p)
    assert len(returned_stack) == d_p["n_stack_out"]
    print("OK  site: generate     holder finalised in place -> {} traces, "
          "{} stack (what _build_prompt sees)".format(len(holder),
                                                      len(returned_stack)))

    # site 2 shape (refine): a DIFFERENT stack object, same two calls
    refine_stack = list(stack) + [InjectedCode(pre_process="x", rationale="y",
                                               source_depth=9)]
    holder2 = eng.context_manager.sample_traces(traces)
    st2 = eng.context_manager.truncate_context_stack(refine_stack)
    assert holder2.finalised
    assert adapter.last_diagnostics["mode"] == "predictive"
    print("OK  site: refine       same adapter, different stack -> {} traces, "
          "{} stack".format(len(holder2), len(st2)))

    # official mode leaves the engine's own manager completely alone
    eng2 = object.__new__(OmegaEngine)
    base = ContextManager()
    eng2.context_manager = base
    install(eng2, ContextReducer(ReductionMode.OFFICIAL))
    h3 = eng2.context_manager.sample_traces(traces)
    assert not isinstance(h3, _FinalisableList)
    print("OK  official untouched official mode does not wrap the manager "
          "(byte-identical path)")

    # ordering guard
    bare = PredictiveContextManagerAdapter(r_pred)
    try:
        bare.truncate_context_stack(stack)
        raise AssertionError("stack-first call was accepted")
    except RuntimeError:
        pass
    print("OK  ordering guard     truncate-before-sample raises instead of "
          "silently reducing the wrong U_v")

    # zero cost
    paid = 0
    led = os.environ.get("META_N_REQUEST_LEDGER", "").strip()
    if led and os.path.isfile(led):
        from meta_n.rpbe.accounting import snapshot
        paid = snapshot()["backend_requests"]
    assert paid == 0
    print("OK  zero-cost          paid backend_requests = {}".format(paid))

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
