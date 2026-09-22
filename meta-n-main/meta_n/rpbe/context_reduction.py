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

`predictive` needs the JOINT decision (attention spans the concatenated
item list `U_v = traces + context_stack`), but the engine hands the two halves
in two separate calls with the stack second. The adapter therefore returns a
list it keeps a handle on and FINALISES it in place on the stack call. A
self-test drives both real sites to prove the finalised object is what
`_build_prompt` receives.

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


class _FinalisableList(list):
    """A list the adapter can rewrite in place once the stack arrives."""

    __slots__ = ("finalised", "owner")

    def __init__(self, items, owner=None):
        super().__init__(items)
        self.finalised = False
        self.owner = owner


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

        Structurally the native `_official`, with ONE difference: the trace
        sampler is capped at TRACE_K4 instead of its own default. The stack goes
        through the identical `truncate_context_stack` call, so the two arms see
        the same stack and the only remaining variable is which four traces were
        chosen. That is the comparison the whole experiment is for.
        """
        sampled = self.context_manager.sample_traces(
            list(traces), max_total=TRACE_K4)
        stack = self.context_manager.truncate_context_stack(list(context_stack))
        return (list(sampled), list(stack),
                dict(self._budget_diag("official_trace_k4", traces, sampled,
                                       context_stack, stack)))

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
        return {"mode": mode,
                "n_traces_in": len(traces), "n_traces_out": len(sel),
                "n_stack_in": len(context_stack), "n_stack_out": len(stack),
                "n_objects_out": len(sel) + len(stack),
                "matched_context_objects": MATCHED_CONTEXT_OBJECTS,
                "trace_tokens_out": t_tok, "stack_tokens_out": s_tok,
                "total_tokens_out": (None if t_tok is None or s_tok is None
                                     else t_tok + s_tok)}

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
        # The stack, by the native rule -- byte-identical to the baseline's.
        sel_s = list(
            self.context_manager.truncate_context_stack(list(context_stack)))
        diag["mode"] = "predictive"
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
        self.budget = self.base.budget
        self.last_diagnostics: Dict[str, Any] = {}
        self._pending = None

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


def _append_trace(diag: Dict[str, Any]) -> None:
    """Append one reduction's diagnostics to a JSONL if requested.

    Used by the three-mode E2E to prove what each mode actually did, without
    changing any engine behaviour. Off unless META_N_REDUCTION_TRACE is set.
    """
    import json
    import os
    import time
    path = os.environ.get(TRACE_ENV, "").strip()
    if not path:
        return
    try:
        rec = dict(diag)
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


def self_test() -> int:
    import os
    from meta_n.core.meta_layer import InjectedCode, Trace
    from meta_n.utils.context_manager import ContextBudget

    print("context_reduction.py acceptance")
    print("=" * 68)

    traces = [Trace(task_id="t{}".format(i), script="s", stdout="o" * 100,
                    success=True) for i in range(3)]
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

    # PREDICTIVE goes through the selector
    r_pred = ContextReducer(ReductionMode.PREDICTIVE, encoder=_StubEncoder(),
                            fusion=_StubFusion())
    t_p, s_p, d_p = r_pred.reduce(traces, stack, budget=budget)
    assert d_p["mode"] == "predictive"
    assert d_p["n_selected"] == 4, d_p["n_selected"]
    for t in t_p:
        assert any(t is x for x in traces)
    for c in s_p:
        assert any(c is x for x in stack)
    print("OK  predictive         selector picked {} items; every returned "
          "object is ORIGINAL (identity preserved)".format(d_p["n_selected"]))

    # C_v IS NEVER APPENDED: the reducer's output carries no tasks/scores
    assert not hasattr(t_p, "tasks") and "tasks" not in d_p
    print("OK  no C_v appended    reducer output is traces+stack only")

    # --- the adapter covers BOTH engine sites -----------------------------
    from meta_n.core.omega import OmegaEngine
    eng = object.__new__(OmegaEngine)
    eng.context_manager = ContextManager()
    adapter = install(eng, r_pred)

    # site 1 shape (generate): sample then truncate, then read the holder
    holder = eng.context_manager.sample_traces(traces)
    returned_stack = eng.context_manager.truncate_context_stack(stack)
    assert holder.finalised, "holder was not finalised"
    assert len(holder) == d_p["n_selected"] - len(returned_stack)
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
