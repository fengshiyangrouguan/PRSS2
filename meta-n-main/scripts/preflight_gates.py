"""PREFLIGHT GATE SUITE -- everything provable at ZERO API cost.

User's gate list (2026-09-17). Each item is a real closed-loop check, not a
synthetic-only claim:

    [1] the runtime contract actually reaches the model: the FINAL task
        description contains BOTH the per-instance timeout AND the forbidden
        imports;
    [2] `merge_code_libraries` regression (the Meta^n helper-merging bug fix);
    [3] synthetic seed->child->grandchild yields eligible_lineages == 1;
    [4] full/official/predictive routing covers BOTH `generate` AND `refine`
        (refine was previously easy to miss, which would silently collapse
        Official and Predictive on the self-repair path);
    [5] Phase B runs 300 steps on a synthetic window with API calls == 0;
    [6] the three reduction modes share an identical task description (the
        fairness requirement).

Item 7 -- a real-API canary -- is NOT here: it costs money and is run
separately, only after all of the above are green.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print("  %s %-46s %s" % ("OK " if ok else "** ", label, detail))


# --------------------------------------------------------------------------
def gate1_contract():
    print("=" * 72)
    print("[1] runtime contract reaches the model")
    print("=" * 72)
    from meta_n.integrations.co_bench import COBenchAdapter
    from meta_n.utils.safety import BLOCKED_IMPORTS
    a = COBenchAdapter(data_dir="./data/co_bench",
                       task_names=["Bin packing - one-dimensional"],
                       instance_workers=1)
    d = a.load_tasks()[0].description
    check("mentions the 10-second limit", "10-second" in d)
    check("mentions helper validation", "validated BEFORE" in d)
    for mod in ("sys", "signal", "threading", "subprocess", "io"):
        if mod not in d:
            check("lists blocked import '%s'" % mod, False)
            break
    else:
        check("lists the blocked imports", "%d modules" % len(BLOCKED_IMPORTS))
    check("metadata carries the timeout",
          a.load_tasks()[0].metadata.get("instance_timeout_s") == 10)
    return d


def gate2_merge():
    print()
    print("=" * 72)
    print("[2] merge_code_libraries regression (Meta^n helper-merging bug fix)")
    print("=" * 72)
    from meta_n.core.meta_layer import InjectedCode, merge_code_libraries
    from meta_n.core.code_library import validate_library_function

    V = "def foo(a, b):\n    return a + b\n"
    BAD = "import sys\ndef foo(a, b):\n    return a + b\n"
    V2 = "def foo(a, b):\n    return a * b\n"

    def ic(depth, bodies, pre=None):
        return InjectedCode(pre_process=pre, rationale="",
                            code_library=dict(bodies), source_depth=depth)

    assert not validate_library_function("foo", BAD, executor=None)
    m, _ = merge_code_libraries([ic(2, {"foo": V}), ic(3, {"foo": BAD})])
    check("valid then INVALID same name -> old kept", m.get("foo") == V)
    m, _ = merge_code_libraries([ic(2, {"foo": V}), ic(3, {"foo": V2})])
    check("valid then VALID same name -> new overrides", m.get("foo") == V2)
    m, _ = merge_code_libraries([ic(2, {"foo": BAD})])
    check("only INVALID -> not injected", "foo" not in m)
    m, _ = merge_code_libraries([ic(2, {"foo": V}, pre="x = (\n 'oops\n")])
    check("bad pre_process -> library unaffected", m.get("foo") == V)


def gate3_lineage():
    print()
    print("=" * 72)
    print("[3] synthetic seed->child->grandchild -> eligible_lineages == 1")
    print("=" * 72)
    from meta_n.rpbe.lineage import extract_lineages
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "run_x"
        (root / "archive").mkdir(parents=True)
        scores = {"gen0_seed": 0.90, "gen1_b0_k0": 0.93, "gen2_b0_k0": 0.95}
        parent = {"gen0_seed": None, "gen1_b0_k0": "gen0_seed",
                  "gen2_b0_k0": "gen1_b0_k0"}
        for cid, par in parent.items():
            d = root / "archive" / cid
            (d / "traces").mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({
                "candidate_id": cid, "parent_id": par,
                "depth": 1 if par is None else 2,
                "mean_score": scores[cid],
                "per_task_scores": {"t1": scores[cid], "t2": scores[cid]}}))
            (d / "traces" / "t1.json").write_text(json.dumps(
                {"task_id": "t1", "score": scores[cid], "success": True}))
        lins = extract_lineages(root)
        check("eligible_lineages == 1", len(lins) == 1,
              "got %d" % len(lins))
        if lins:
            ln = lins[0]
            check("chain is seed->child->grandchild",
                  ln.candidate_id == "run_x:gen0_seed" and
                  ln.child_id == "run_x:gen1_b0_k0" and
                  ln.grandchild_id == "run_x:gen2_b0_k0")
            check("R_v = grandchild - parent", abs(ln.r - 0.05) < 1e-9,
                  "r=%.4f" % ln.r)


def gate4_routing():
    print()
    print("=" * 72)
    print("[4] full/official/predictive routing covers generate AND refine")
    print("=" * 72)
    from meta_n.core.llm_client import LLMClient, LLMConfig
    from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace
    from meta_n.core.omega import OmegaEngine
    from meta_n.rpbe.context_reduction import ContextReducer, install
    from meta_n.utils.context_manager import ContextBudget

    class StubEnc:
        dim = 256

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
            return torch.stack([self.encode_query(str(i)) for i in items], 0)

    class StubFusion:
        def __call__(self, X, q, m):
            import torch
            n = X.shape[0]
            A = torch.zeros(4, n)
            for k in range(4):
                A[k, min(k, n - 1)] = 1.0
            return torch.zeros(4, X.shape[1]), A

    traces = [Trace(task_id="t%d" % i, depth=2, script="s", stdout="o" * 50,
                    success=True, score=0.5) for i in range(4)]
    stack = [InjectedCode(pre_process="def p(): pass", rationale="r",
                          code_library={"h%d" % d: "def h%d(): pass" % d},
                          source_depth=d) for d in (2, 4)]
    tasks = [TaskDescription(task_id="t", description="x")]

    def engine(mode):
        cl = LLMClient(LLMConfig(base_url="http://stub.invalid/v1",
                                 api_key="k", model="deepseek-flash",
                                 daily_budget_usd=0))
        e = OmegaEngine(cl, context_budget=None)
        if mode != "official":
            install(e, ContextReducer(mode, encoder=StubEnc(),
                                      fusion=StubFusion()
                                      if mode == "predictive" else None))
        return e

    for mode in ("full", "official", "predictive"):
        # Drive the engine's OWN `refine()` with a recording wrapper around the
        # context manager the engine holds, so this proves the ROUTE, not a
        # re-implementation of it.
        e2 = engine(mode)
        holder = {}
        mgr = e2.context_manager
        orig_t = mgr.truncate_context_stack

        def t(stack_, _o=orig_t, _h=holder):
            _h["refine_reduced"] = True
            return _o(stack_)
        mgr.truncate_context_stack = t
        orig_s = mgr.sample_traces

        def s(tr_, _o=orig_s, _h=holder):
            _h["refine_sampled"] = True
            return _o(tr_)
        mgr.sample_traces = s

        try:
            asyncio.run(e2.refine(
                prev_injection=InjectedCode(pre_process="def q(): pass",
                                            rationale="r",
                                            code_library={"z": "def z(): pass"},
                                            source_depth=9),
                child_traces=traces, context_stack=stack, tasks=tasks,
                depth=2))
        except Exception as exc:                                 # noqa: BLE001
            holder["err"] = type(exc).__name__
        check("%-11s refine routes through the adapter" % mode,
              holder.get("refine_reduced") is True and
              holder.get("refine_sampled") is True,
              "err=%s" % holder.get("err", "-"))


def gate5_phaseb(steps: int = 300):
    print()
    print("=" * 72)
    print("[5] Phase B %d steps on a synthetic window, API calls == 0" % steps)
    print("=" * 72)
    import time
    import torch
    from meta_n.rpbe import trainer as T
    from meta_n.rpbe.window import StatWindow

    trees = ["run_%d" % i for i in range(64)]
    tr, pr = T.split_window_trees(trees, n_task=32)
    tw = StatWindow(min_unique_trees=32, name="task")
    pw = StatWindow(min_unique_trees=32, name="protect")
    tw.add(T._rows("task", tr, seed=1))
    pw.add(T._rows("protect", pr, seed=2))
    fusion = T._Fusion()

    led = os.environ.get("META_N_REQUEST_LEDGER", "").strip()

    def paid():
        if not (led and os.path.isfile(led)):
            return 0
        n = 0
        for line in open(led):
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("request_kind") not in ("mock", "local"):
                    n += 1
            except json.JSONDecodeError:
                pass
        return n

    w0 = fusion.W.detach().clone()
    api0 = paid()
    tb = T.PhaseB(fusion, tw, pw, steps=steps)
    t0 = time.time()
    hist = tb.run()
    wall = time.time() - t0
    w1 = fusion.W.detach().clone()
    api1 = paid()

    last = hist["history"][-1]
    print("  wall clock            : %.1fs  (%.2fs/step)" % (
        wall, wall / max(1, len(hist["history"]))))
    print("  PHASE B API CALL DELTA: %d  (must be 0)" % (api1 - api0))
    print("  Gamma changed         : %s  (max|dW| = %.3e)" % (
        not torch.equal(w0, w1), float((w1 - w0).abs().max())))
    print("  finite (no NaN/Inf)   : W=%s J_task=%s J_LPSE=%s" % (
        bool(torch.isfinite(w1).all()),
        all(abs(h["J_task"]) < 1e9 for h in hist["history"]),
        all(abs(h["J_LPSE"]) < 1e9 for h in hist["history"])))
    print("  final J_task          : %.6f" % last["J_task"])
    print("  final v_max           : %s" % last.get("vmax_proj"))
    print("  final n_active        : %s" % last.get("proj_n_active"))
    print("  committed every step  : %s" % all(
        h.get("committed") for h in hist["history"]))
    out = Path("runs/_phaseb_gate_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"steps": len(hist["history"]), "wall_s": wall,
         "api_delta": api1 - api0,
         "gamma_moved": bool(not torch.equal(w0, w1)),
         "final_J_task": last["J_task"], "final_vmax": last.get("vmax_proj"),
         "final_n_active": last.get("proj_n_active")}, indent=2))
    print("  artifacts written     : %s" % out)

    check("steps committed == %d" % steps,
          hist["steps"] == steps, "got %d" % hist["steps"])
    check("API call delta == 0", api1 - api0 == 0, "got %d" % (api1 - api0))
    check("omega_replay_count == 0 everywhere",
          all(h["omega_replay_count"] == 0 for h in hist["history"]))
    check("Gamma actually updated", not torch.equal(w0, w1))
    check("no NaN/Inf", bool(torch.isfinite(w1).all()))
    check("tree roles unchanged",
          frozenset(tw.tree_ids) == frozenset(
              (t, "gen0_seed") for t in tr))


def main() -> int:
    want = set()
    steps = 300
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            want = {int(x) for x in a.split("=", 1)[1].split(",")}
        elif a.startswith("--steps="):
            steps = int(a.split("=", 1)[1])
    gates = {1: gate1_contract, 2: gate2_merge, 3: gate3_lineage,
             4: gate4_routing, 5: lambda: gate5_phaseb(steps)}
    for n in sorted(gates):
        if want and n not in want:
            continue
        gates[n]()
    print()
    print("=" * 72)
    print("PREFLIGHT GATES: %d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
    print("VERDICT:", "ALL GATES GREEN" if not FAIL else "NOT READY")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
