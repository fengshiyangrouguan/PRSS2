"""PredictiveSelector -- the deployment-time renderer (task book v4.1 §2.3/§2.11).

This module maps the four-slot attention `A_v` back onto the ORIGINAL prompt
objects. It is a renderer, not a model: it never re-summarises, never generates
text, and never invents an item that was not already there.

It does NOT do batch-level record selection. Which CutRecords are eligible for
the training window is decided by `lineage.py` (§2.5) and `window.py` (§4.2);
keeping those apart is deliberate.

FROZEN CONTRACT (§2.11, eight steps, implemented literally):

    1. slots are processed k = 0,1,2,3 in order
    2. used = {}                                       (no item selected twice)
    3. for slot k walk items by descending attention, skipping used ones
    4. if item j would push its TYPE past that type's Official budget, skip it
       and try the slot's next unused item
    5. on selection: used.add(j), record (k, j)
    6. at most ONE item per slot; at most four original objects in total;
       fewer is fine -- never keep adding just to fill the budget
    7. traces keep slot-selection order; InjectedCode is sorted by source_depth
    8. no text is generated

Items are the concatenation `U_v = traces + context_stack` (§2.8), so item j is
a trace when `j < len(traces)` and injected code otherwise. The item order MUST
match the order the encoder saw, or the attention indices point at the wrong
objects -- the width check below enforces that.

Budget arithmetic reuses the OFFICIAL estimators
(`ContextManager._estimate_trace_tokens` / `_estimate_injected_code_tokens` /
`estimate_tokens`) rather than a hand-rolled counter, so "the budget is
enforced" means the same thing here as it does in `official` mode.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from meta_n.core.meta_layer import InjectedCode, Trace
from meta_n.utils.context_manager import ContextBudget, ContextManager

TRACE = "trace"
CODE = "code"

# §2.8: U_v = (traces, context_stack). One place, so every caller agrees.
KIND_ORDER = (TRACE, CODE)


class SelectorError(RuntimeError):
    """The attention tensor and the item list disagree."""


class PredictiveSelector:
    """Attention -> original Trace/InjectedCode. Deterministic, no LLM."""

    def __init__(self, counter: Optional[ContextManager] = None) -> None:
        # Reuse the official estimator instance; it is stateless for our use.
        self._cm = counter or ContextManager()

    # -- public API (v4.1 §2.3) -------------------------------------------
    def reduce(
        self,
        traces: Sequence[Trace],
        context_stack: Sequence[InjectedCode],
        attention: torch.Tensor,
        budget: ContextBudget,
    ) -> Tuple[List[Trace], List[InjectedCode], Dict[str, Any]]:
        """-> (selected_traces, selected_stack, diagnostics).

        `diagnostics` carries the weighted statistics. They are DIAGNOSTICS
        ONLY -- the frozen contract forbids appending them to the prompt, and
        `_build_prompt` is not touched by this module.
        """
        items: List[Any] = list(traces) + list(context_stack)
        kinds = [TRACE] * len(traces) + [CODE] * len(context_stack)
        n_items = len(items)
        diag: Dict[str, Any] = {"n_items": n_items,
                                "n_traces_in": len(traces),
                                "n_stack_in": len(context_stack)}

        if attention.dim() != 2:
            raise SelectorError("attention must be [n_slots, n_v], got shape {}"
                                .format(tuple(attention.shape)))
        n_slots, n_v = int(attention.shape[0]), int(attention.shape[1])
        if n_v != n_items:
            raise SelectorError(
                "attention width {} != item count {} (traces {} + stack {}). "
                "The item order is U_v = traces + context_stack; a mismatch "
                "means the attention was built from a different item list."
                .format(n_v, n_items, len(traces), len(context_stack)))
        if n_slots < 1:
            raise SelectorError("attention must have at least one slot")

        caps = {TRACE: int(budget.traces_budget),
                CODE: int(budget.context_stack_budget)}
        used: set = set()
        used_tokens = {TRACE: 0, CODE: 0}
        picks: List[Tuple[int, int]] = []          # (slot, item index)
        slot_diag: List[Dict[str, Any]] = []

        # --- steps 1-6 ----------------------------------------------------
        for k in range(n_slots):
            row = attention[k].detach().to(torch.float64)
            if not bool(torch.isfinite(row).all()) or float(row.sum()) <= 0.0:
                # An invalid slot renders NOTHING. Never let a degenerate
                # attention row pull an arbitrary item into the prompt.
                slot_diag.append({"slot": k, "item": None, "kind": None,
                                  "attention": None, "tokens": 0,
                                  "reason": "invalid_attention"})
                continue
            order = sorted(range(n_items),
                           key=lambda j: (-float(row[j]), j))
            chosen = None
            for j in order:
                if j in used:
                    continue
                kind = kinds[j]
                toks = self._item_tokens(items[j], kind)
                if used_tokens[kind] + toks > caps[kind]:
                    continue
                chosen, chosen_toks = j, toks
                break
            if chosen is None:
                slot_diag.append({"slot": k, "item": None, "kind": None,
                                  "attention": None, "tokens": 0,
                                  "reason": "no_item_within_budget"})
                continue
            used.add(chosen)
            used_tokens[kinds[chosen]] += chosen_toks
            picks.append((k, chosen))
            slot_diag.append({"slot": k, "item": chosen,
                              "kind": kinds[chosen],
                              "attention": float(row[chosen]),
                              "tokens": chosen_toks, "reason": "selected"})

        # --- step 7: output order ----------------------------------------
        sel_traces = [items[j] for (k, j) in picks if kinds[j] == TRACE]
        sel_codes = [items[j] for (k, j) in picks if kinds[j] == CODE]
        sel_codes.sort(key=lambda c: int(getattr(c, "source_depth", 0)))

        # --- diagnostics (weighted statistics live here, never in a prompt)
        diag.update({
            "slots": slot_diag,
            "n_selected": len(picks),
            "n_selected_traces": len(sel_traces),
            "n_selected_stack": len(sel_codes),
            "trace_budget": caps[TRACE],
            "stack_budget": caps[CODE],
            "trace_tokens_used": used_tokens[TRACE],
            "stack_tokens_used": used_tokens[CODE],
        })
        diag.update(self._weighted_stats(picks, items, kinds, attention))
        return sel_traces, sel_codes, diag

    # -- internals ---------------------------------------------------------
    def _item_tokens(self, item: Any, kind: str) -> int:
        """Official estimator, so 'budget enforced' means the same as in
        `official` mode. Delegated, never re-implemented."""
        if kind == TRACE:
            return int(self._cm._estimate_trace_tokens(item))
        return int(self._cm._estimate_injected_code_tokens(item))

    @staticmethod
    def _weighted_stats(picks, items, kinds, attention) -> Dict[str, Any]:
        """Weighted mean score / failure mass / class + depth distributions.

        Weighted by the slot attention that selected each item.
        """
        num = den = 0.0
        failure_mass = 0.0
        fail_classes: Dict[str, int] = {}
        depths: Dict[int, int] = {}
        for (k, j) in picks:
            w = float(attention[k][j])
            kind = kinds[j]
            if kind == TRACE:
                num += w * float(getattr(items[j], "score", 0.0) or 0.0)
                den += w
                if not bool(getattr(items[j], "success", False)):
                    failure_mass += w
                    cls = getattr(items[j], "failure_class", "") or "unclassified"
                    fail_classes[cls] = fail_classes.get(cls, 0) + 1
            else:
                d = int(getattr(items[j], "source_depth", 0))
                depths[d] = depths.get(d, 0) + 1
        return {
            "weighted_mean_score": (num / den) if den > 0 else None,
            "failure_mass": failure_mass,
            "failure_class_distribution": fail_classes,
            "source_depth_distribution": depths,
        }


# --------------------------------------------------------------------------
# acceptance self-test -- 7 checks, zero network
# --------------------------------------------------------------------------

def _trace(i: int, score: float = 0.5, success: bool = True,
           failure_class: str = "") -> Trace:
    return Trace(task_id="t{}".format(i), script="print({})".format(i),
                 stdout="out" * 20, stderr="", exit_code=0 if success else 1,
                 success=success, score=score, error_summary="",
                 eval_feedback="", failure_class=failure_class)


def _code(depth: int, body: str = "def f(): return 1") -> InjectedCode:
    return InjectedCode(pre_process="def pre(): pass", rationale="r",
                        code_library={"f": body}, source_depth=depth)


def _attn(n_slots: int, n_v: int, rows: List[List[float]]) -> torch.Tensor:
    a = torch.zeros(n_slots, n_v, dtype=torch.float32)
    for k, r in enumerate(rows):
        a[k, :len(r)] = torch.tensor(r, dtype=torch.float32)
    return a


def self_test() -> int:
    import os
    sel = PredictiveSelector()
    budget = ContextBudget()
    print("selector.py acceptance")
    print("=" * 68)

    traces = [_trace(0), _trace(1), _trace(2)]
    stack = [_code(2), _code(5), _code(3)]
    n = len(traces) + len(stack)                      # 6 items
    # slot 0 wants item 0 (trace0), slot 1 item 3 (code@2), etc.
    A = _attn(4, n, [[0.9, 0.05, 0.05, 0, 0, 0],
                     [0, 0, 0, 0.7, 0.2, 0.1],
                     [0.1, 0.8, 0.1, 0, 0, 0],
                     [0, 0, 0, 0.1, 0.2, 0.7]])

    # 1 -- DETERMINISTIC
    r1 = sel.reduce(traces, stack, A, budget)
    r2 = sel.reduce(traces, stack, A.clone(), budget)
    assert [id(t) for t in r1[0]] == [id(t) for t in r2[0]]
    assert [id(c) for c in r1[1]] == [id(c) for c in r2[1]]
    assert r1[2]["slots"] == r2[2]["slots"]
    print("OK  deterministic      identical inputs -> identical objects+diag")

    # 2 -- SLOT FIDELITY: 4 distinct slots, no crossing / missing / duplicating
    tsel, csel, d = r1
    assert d["n_selected"] == 4, d["n_selected"]
    idx = [s["item"] for s in d["slots"]]
    assert len(set(idx)) == 4, idx                    # no item used twice
    assert [s["slot"] for s in d["slots"]] == [0, 1, 2, 3]
    # slot 0 -> item 0 ; slot 1 -> item 3 ; slot 2 -> item 1 ; slot 3 -> item 5
    assert idx == [0, 3, 1, 5], idx
    print("OK  slot fidelity      slots {} -> items {} (no cross/reuse)"
          .format([s["slot"] for s in d["slots"]], idx))

    # 3 -- ORIGINAL-OBJECT FIDELITY: the very same objects, not copies
    assert tsel[0] is traces[0] and tsel[1] is traces[1]
    assert all(any(c is orig for orig in stack) for c in csel)
    print("OK  object fidelity    returned the ORIGINAL Trace/InjectedCode "
          "objects (identity, not equality)")

    # 4 -- OUTPUT ORDER (§2.11 step 7)
    assert [t.task_id for t in tsel] == ["t0", "t1"]   # slot order
    assert [c.source_depth for c in csel] == [2, 3]    # ascending depth
    print("OK  output order       traces by slot order; stack by source_depth "
          "{}".format([c.source_depth for c in csel]))

    # 5 -- INVALID SLOT: a degenerate row renders NOTHING
    A_bad = A.clone()
    A_bad[2, :] = 0.0                                  # all-zero slot
    t3, c3, d3 = sel.reduce(traces, stack, A_bad, budget)
    assert d3["slots"][2]["reason"] == "invalid_attention", d3["slots"][2]
    assert d3["slots"][2]["item"] is None
    assert d3["n_selected"] == 3
    A_nan = A.clone()
    A_nan[3, :] = float("nan")
    _, _, d4 = sel.reduce(traces, stack, A_nan, budget)
    assert d4["slots"][3]["reason"] == "invalid_attention"
    print("OK  invalid slots      all-zero and NaN rows render nothing")

    # 6 -- BUDGET CORRECTNESS (official estimators; skip-and-try-next)
    tiny = ContextBudget(max_tokens=2000, prompt_overhead=0,
                         traces_ratio=0.0001, context_stack_ratio=0.0001)
    _, _, d5 = sel.reduce(traces, stack, A, tiny)
    assert d5["n_selected"] == 0, d5["n_selected"]
    assert all(s["reason"] == "no_item_within_budget" for s in d5["slots"])
    # a budget that fits exactly one trace: only the top trace enters
    one = ContextBudget(max_tokens=4000, prompt_overhead=0,
                        traces_ratio=0.02, context_stack_ratio=0.0001)
    t6, _, d6 = sel.reduce(traces, stack, A, one)
    assert len(t6) == 1 and t6[0] is traces[0], [x.task_id for x in t6]
    assert d6["trace_tokens_used"] <= d6["trace_budget"]
    print("OK  budget             starved budget selects nothing; a 1-trace "
          "budget takes only the top trace (used {} <= {})".format(
              d6["trace_tokens_used"], d6["trace_budget"]))

    # 7 -- NO FUTURE LEAKAGE + ZERO COST
    import inspect
    src = inspect.getsource(PredictiveSelector.reduce)
    for banned in ("lineage", "future", "descendant", "llm", "complete("):
        assert banned not in src, banned
    params = set(inspect.signature(sel.reduce).parameters)
    assert params == {"traces", "context_stack", "attention", "budget"}, params
    paid = 0
    led = os.environ.get("META_N_REQUEST_LEDGER", "").strip()
    if led and os.path.isfile(led):
        from meta_n.rpbe.accounting import snapshot
        paid = snapshot()["backend_requests"]
    assert paid == 0, paid
    print("OK  no-leakage/cost    signature is exactly "
          "(traces, context_stack, attention, budget); paid requests = {}"
          .format(paid))

    # width mismatch must be loud, not silently mis-indexed
    try:
        sel.reduce(traces, stack, A[:, :-1], budget)
        raise AssertionError("accepted an attention width mismatch")
    except SelectorError:
        pass
    print("OK  width guard        attention/item mismatch raises")

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
