"""archive -> CutRecord builder (approved 2026-09-18).

FIRST-VERSION SCOPE, deliberately narrow. It does ONE thing: for a CONFIRMED
Official run's archive, enumerate the REAL ``c_v -> c_{v+1} -> c_{v+2}``
occurrences and materialise each as a v4.2 :class:`CutRecord` -- the real
``X_v / q_emb / mask / p / r / weight / IDs``.

It does NOT touch: the task/protect split protocol, ``trainer.py``, the Meta^n
main loop, or any v4.2 mathematical definition. It only reads archives and
builds records. Changing the split is a separate, later decision.

CONSUMER CONTRACT -- Official data only. A run directory carrying a
``CONTROL_NOTICE.txt`` is REFUSED: those runs passed ``--benchmark-config none``,
the documented control / bare-pre-metacognition path, so their archives are not
Official data and must never enter the training set.

WHAT A CUT IS HERE
------------------
A cut is taken at ``c_v`` -- the candidate whose observation Omega consumed.
Following the v4.2 §2.8 ``U_v`` contract and the run's own artifacts:

    U_v items = [traces of c_v] + [injected-code layers c_v carries]
    q_emb     = E_0(OmegaEngine._format_raw_traces(c_v traces))
    mask      = zeros([N_SLOTS, n_v])   # only padding is masked; there is none
    Y_{v+1}   = traces of c_{v+1}       # the REAL child observation
    Y_{v+2}   = traces of c_{v+2}       # the REAL grandchild observation
    p         = phi_r(Y_{v+1}, Y_{v+2})        (FutureSketcher, frozen)
    r         = mean_score(c_{v+2}) - mean_score(c_v)   (the real two-step return)

OCCURRENCE CONVENTION. ``lineage.extract_lineages`` already weights each path by
``1/len(paths)`` (v4.2 §2.5: a parent with many children must not be amplified).
So a candidate ``c_v`` with k real paths yields **k** records that share one
``X_v`` (same observation) and carry k different futures, with ``occurrence_seq``
0..k-1 and ``weight = 1/k`` -- their weights sum to 1 per parent.

Usage:
    python -m meta_n.rpbe.build_records --out runs/_records.jsonl <run_dir> ...
    python -m meta_n.rpbe.build_records --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from meta_n.core.meta_layer import InjectedCode
from meta_n.rpbe import config as C
from meta_n.rpbe.lineage import Lineage, extract_lineages, load_traces
from meta_n.rpbe.records import CutMeta, CutRecord, make_candidate_id, \
    parse_candidate_id, save_jsonl

CONTROL_NOTICE = "CONTROL_NOTICE.txt"


class NotOfficialData(RuntimeError):
    """The run is a control run and must never enter the training set."""


# --------------------------------------------------------------------------
# archive reads
# --------------------------------------------------------------------------

def assert_official_run(run_dir: Path) -> None:
    """Refuse a control run outright (it carries a CONTROL_NOTICE.txt)."""
    notice = Path(run_dir) / CONTROL_NOTICE
    if notice.is_file():
        raise NotOfficialData(
            "{} is a CONTROL run ({} present). It used "
            "--benchmark-config none, the bare-pre-metacognition path, so its "
            "archive is not Official data and must not be used for training. "
            "First line: {}".format(
                run_dir, CONTROL_NOTICE,
                notice.read_text(encoding="utf-8").splitlines()[0]))


def load_injected_codes(run_dir: Path, bare_candidate_id: str
                        ) -> List[InjectedCode]:
    """The candidate's injection chain, in depth order.

    The orchestrator persists one ``injected_code_d{i}.json`` per layer
    (``run_persistence.save_candidate_incremental``), which is exactly the
    context stack this cut's Omega call saw.
    """
    import re
    cdir = Path(run_dir) / "archive" / bare_candidate_id
    if not cdir.is_dir():
        return []
    found = []
    for p in cdir.glob("injected_code_d*.json"):
        m = re.search(r"d(\d+)", p.name)
        found.append((int(m.group(1)) if m else 0, p))
    out: List[InjectedCode] = []
    for _, p in sorted(found, key=lambda t: t[0]):
        try:
            out.append(InjectedCode(**json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, ValueError, TypeError):
            continue
    return [ic for ic in out if not ic.is_empty]


def cut_items(run_dir: Path, bare_candidate_id: str) -> List[Any]:
    """``U_v`` items for a cut at this candidate: traces first, then code."""
    traces = list(load_traces(run_dir, bare_candidate_id))
    codes = load_injected_codes(run_dir, bare_candidate_id)
    return traces + codes


# --------------------------------------------------------------------------
# record construction
# --------------------------------------------------------------------------

def build_run_records(run_dir: os.PathLike | str, encoder: Any,
                      sketcher: Any = None,
                      *, max_items: Optional[int] = None) -> List[CutRecord]:
    """Every real lineage in one run -> one CutRecord per (c_v, path).

    ``encoder`` must expose ``encode(items) -> [n, d_e]``,
    ``encode_query(text) -> [d_e]`` and ``serialize_query(traces) -> str``
    (``FrozenItemEncoder`` satisfies all three). ``sketcher`` defaults to a
    ``FutureSketcher`` over the same encoder; its projections are seed-fixed, so
    passing one in only saves rebuilding them.
    """
    run_dir = Path(run_dir)
    assert_official_run(run_dir)
    if sketcher is None:
        from meta_n.rpbe.future import FutureSketcher
        sketcher = FutureSketcher(encoder)

    lineages = extract_lineages(run_dir)
    if not lineages:
        return []

    # Group paths by the parent candidate: one X_v per parent, k futures.
    by_parent: Dict[str, List[Lineage]] = {}
    for ln in lineages:
        by_parent.setdefault(ln.candidate_id, []).append(ln)

    records: List[CutRecord] = []
    for candidate_id in sorted(by_parent):
        paths = by_parent[candidate_id]
        head = paths[0]
        _, bare = parse_candidate_id(candidate_id)
        items = cut_items(run_dir, bare)
        if not items:
            # No observation to encode -> the cut carries no X_v. Skipping is the
            # only correct move; a [0, d_e] X_v would silently break the fusion.
            continue
        if max_items is not None and len(items) > max_items:
            items = items[:max_items]

        trace_part = [i for i in items
                      if type(i).__name__ == "Trace"]

        # X_v encodes the TRACE POOL ONLY. It used to encode `items` =
        # traces + injected codes, which made Gamma responsible for compressing
        # Omega's ENTIRE context; that is a different (and much larger) job than
        # choosing which runtime FEEDBACK Omega should see. Measured consequence
        # on a live run: deployed against a stack it had been trained to
        # compress, Gamma spent all 4 slots on traces and dropped the stack in
        # 18/18 opportunities, while the hand rule kept it 20/20. The stack is
        # now BYPASSED -- both arms hand the same stack to Omega untouched --
        # so the only variable left is which 4 traces are selected.
        X_v = torch.as_tensor(encoder.encode(trace_part), dtype=torch.float32)
        q_emb = torch.as_tensor(
            encoder.encode_query(encoder.serialize_query(trace_part)),
            dtype=torch.float32).reshape(-1)
        n_v = int(X_v.shape[0])
        mask = torch.zeros(C.N_SLOTS, n_v, dtype=torch.float32)

        meta = CutMeta(candidate_id=candidate_id, depth=head.depth,
                       run_id=head.run_id, root_candidate_id=head.root_candidate_id)
        for occ, ln in enumerate(paths):
            p = torch.as_tensor(
                sketcher.sketch(ln.child_traces, ln.grandchild_traces),
                dtype=torch.float32)
            records.append(CutRecord(
                meta=meta,
                cut_id=(candidate_id, occ),
                tree_id=ln.tree_id,
                occurrence_seq=occ,
                X_v=X_v.clone(),
                q_emb=q_emb.clone(),
                mask=mask.clone(),
                p=p,
                r=float(ln.r),
                weight=float(ln.weight),
            ))
    return records


def build_records(run_dirs: Sequence[os.PathLike | str], encoder: Any,
                  sketcher: Any = None,
                  **kw) -> List[CutRecord]:
    """Concatenate every run's records. Control runs raise, never skip silently."""
    from meta_n.rpbe.future import FutureSketcher
    if sketcher is None:
        sketcher = FutureSketcher(encoder)
    out: List[CutRecord] = []
    for rd in run_dirs:
        out.extend(build_run_records(rd, encoder, sketcher, **kw))
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def summarize(records: Sequence[CutRecord]) -> Dict[str, Any]:
    """The counts the pilot asked for, computed from the REAL records."""
    trees: Dict[Tuple[str, str], int] = {}
    parents = set()
    runs = set()
    for r in records:
        trees[r.tree_id] = trees.get(r.tree_id, 0) + 1
        parents.add(r.meta.candidate_id)
        runs.add(r.meta.run_id)
    return {
        "cuts": len(records),
        "unique_parents": len(parents),
        "unique_trees": len(trees),
        "runs": len(runs),
        "cuts_per_tree": {"{}|{}".format(*k): v for k, v in sorted(trees.items())},
        "weights_sum": float(sum(r.weight for r in records)),
        "r_values": [float(r.r) for r in records],
    }


def phase_b_runnable(records: Sequence[CutRecord], *, split: str = "v421"
                     ) -> Dict[str, Any]:
    """v4.2.1 §5: can Phase B be CONSTRUCTED? (Not: will it learn anything.)

        phase_b_runnable =
            task_records_nonempty
            AND protect_records_nonempty
            AND structural_contracts_pass
            AND task/protect closure succeeds
            AND required diagnostics are finite

    It deliberately does NOT require ``alpha < 1`` or ``J > 0``: ``alpha = 1,
    J = 0`` means "runnable but no training signal", which is a statement about
    the DATA, not a structural error. Conflating the two is what this split
    exists to prevent.

    ``formal_32x32_ready`` rides alongside as a confidence LABEL only; it never
    decides whether Phase B is constructed.
    """
    from meta_n.rpbe.census import (SPLIT_PROTOCOL, fixed_hash_split,
                                    partition_tree_roles_v421)
    from meta_n.rpbe.window import StatWindow, WindowError

    tree_ids = sorted({r.tree_id for r in records})
    if split == "v421":
        task_roots, protect_roots = partition_tree_roles_v421(tree_ids)
        rule = SPLIT_PROTOCOL
    elif split == "legacy":
        # kept ONLY to reproduce historical results / explicit control use
        task_roots, protect_roots = fixed_hash_split(tree_ids,
                                                     n_task=C.N_TREES_TASK)
        rule = "legacy_truncation"
    else:
        raise ValueError("unknown split {!r}".format(split))

    out: Dict[str, Any] = {
        "split_protocol": rule,
        "trees_total": len(tree_ids),
        "task_trees": len(task_roots),
        "protect_trees": len(protect_roots),
        "formal_32x32_ready": bool(len(task_roots) >= C.N_TREES_TASK
                                   and len(protect_roots) >= C.N_TREES_PROTECT),
    }

    fam: Dict[str, float] = {}
    for name, roots in (("task", set(task_roots)),
                        ("protect", set(protect_roots))):
        rows = [r for r in records if r.tree_id in roots]
        out["{}_rows".format(name)] = len(rows)
        if not rows:
            out["{}_note".format(name)] = (
                "EMPTY window -> closure / D / OAS / J are NOT computable this "
                "side, and PhaseB refuses an empty window.")
            continue
        w = StatWindow(name=name)
        try:
            w.add(rows)                       # structural contracts
            fusion = _init_fusion()
            if name == "task":
                S = w.close_task(fusion)
                out["task_J"] = float(S["J_task"])
                out["task_alpha_z"] = S.get("oas_alpha_z_task")
                if S.get("oas_alpha_z_task") is not None:
                    fam["task_alpha_z"] = float(S["oas_alpha_z_task"])
                fam["task_J"] = float(S["J_task"])
            else:
                S = w.close_lpse(fusion)
                alphas = [a for a in (S.get("oas_alpha_z_protect") or [])
                          if a is not None]
                out["protect_J"] = float(S["J_LPSE"])
                out["protect_alpha_z_per_branch"] = [float(a) for a in alphas]
                if alphas:
                    fam["protect_alpha_z"] = sum(alphas) / len(alphas)
                fam["protect_J"] = float(S["J_LPSE"])
            out["{}_D".format(name)] = float(S["D"])
            out["{}_n_trees".format(name)] = int(w.n_trees)
            out["{}_closure".format(name)] = "ok"
            fam["{}_D".format(name)] = float(S["D"])
        except (WindowError, RuntimeError, ValueError, KeyError) as e:  # noqa: BLE001
            out["{}_closure".format(name)] = "{}: {}".format(
                type(e).__name__, e)

    nonempty = bool(out["task_rows"] > 0 and out["protect_rows"] > 0)
    closed = (out.get("task_closure") == "ok"
              and out.get("protect_closure") == "ok")
    finite = bool(fam) and all(math.isfinite(v) for v in fam.values())
    out["required_diagnostics"] = {k: round(v, 6) for k, v in fam.items()}
    out["diagnostics_all_finite"] = finite
    out["phase_b_runnable"] = bool(nonempty and closed and finite)
    return out


def _init_fusion():
    """A SlottedFusion at its FROZEN init.

    Its parameters must keep ``requires_grad=True``: ``close_task`` /
    ``close_lpse`` backprop a cotangent THROUGH Gamma, so switching the flag off
    would break the graph. Nothing here trains it -- no optimizer is ever built
    and no ``.backward()`` is called on the returned fusion.
    """
    from meta_n.rpbe.fusion import SlottedFusion
    return SlottedFusion()


def _encoder():
    path = os.environ.get("CODEBERT_PATH", "").strip()
    if not path or not os.path.isdir(path):
        raise SystemExit(
            "set CODEBERT_PATH to the frozen encoder directory (the builder "
            "must use the real E_0; a stub would make the records meaningless)")
    from meta_n.rpbe.encoder import FrozenItemEncoder
    return FrozenItemEncoder(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--out", default=None, help="write records as JSONL here")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.run_dirs:
        ap.error("need at least one run_dir")

    enc = _encoder()
    from meta_n.rpbe.future import FutureSketcher
    sk = FutureSketcher(enc)

    per_run = {}
    all_records: List[CutRecord] = []
    for rd in args.run_dirs:
        try:
            recs = build_run_records(rd, enc, sk, max_items=args.max_items)
        except NotOfficialData as e:
            print("REFUSED {}: {}".format(rd, e))
            return 2
        per_run[Path(rd).name] = summarize(recs)
        all_records.extend(recs)

    print("=" * 78)
    print("archive -> CutRecord builder (Official runs only; split UNCHANGED)")
    print("=" * 78)
    for name, s in per_run.items():
        print("  {:<46} cuts={:<4} parents={:<3} trees={:<3} W={:.4f}".format(
            name[:46], s["cuts"], s["unique_parents"], s["unique_trees"],
            s["weights_sum"]))
    total = summarize(all_records)
    print("-" * 78)
    print(json.dumps(total, indent=2, ensure_ascii=False))
    print()
    status = phase_b_runnable(all_records)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    print()
    print("split_protocol      : {}".format(status["split_protocol"]))
    print("phase_b_runnable    : {}   (structure only -- NOT a claim that "
          "Gamma has signal)".format(status["phase_b_runnable"]))
    print("formal_32x32_ready  : {}   (confidence label only)".format(
        status["formal_32x32_ready"]))

    if args.out:
        save_jsonl(args.out, all_records)
        print("\nwrote {} records -> {}".format(len(all_records), args.out))
    return 0


# --------------------------------------------------------------------------
# acceptance self-test (offline stub encoder; no model, no network)
# --------------------------------------------------------------------------

class _StubEncoder:
    """Deterministic stand-in so the builder's WIRING is testable offline."""

    dim = C.D_E

    def encode(self, items):
        return torch.stack([self._vec(str(i)) for i in items], 0) if items \
            else torch.zeros(0, self.dim)

    def encode_query(self, text):
        return self._vec(text)

    @staticmethod
    def serialize_query(traces):
        return "|".join(getattr(t, "task_id", "?") for t in traces)

    def _vec(self, text):
        import hashlib
        h = hashlib.sha256(str(text).encode("utf-8")).digest()
        g = torch.Generator().manual_seed(int.from_bytes(h[:4], "little"))
        return torch.randn(self.dim, generator=g)


def self_test() -> int:
    import tempfile
    print("build_records.py acceptance")
    print("=" * 68)

    def mk(run: Path, spec, control=False):
        (run / "archive").mkdir(parents=True, exist_ok=True)
        if control:
            (run / CONTROL_NOTICE).write_text(
                "CONTROL / BARE-PRE-METACOGNITION\nmore\n", encoding="utf-8")
        for cid, par, sc in spec:
            d = run / "archive" / cid
            (d / "traces").mkdir(parents=True, exist_ok=True)
            (d / "summary.json").write_text(json.dumps({
                "candidate_id": cid, "parent_id": par, "depth": 2,
                "mean_score": sc, "per_task_scores": {"t1": sc}}),
                encoding="utf-8")
            (d / "traces" / "t1.json").write_text(json.dumps({
                "task_id": "t1", "depth": 2, "script": "print(1)",
                "success": True, "score": sc}), encoding="utf-8")

    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "run_ok"
        mk(run, [("gen0_seed", None, 0.1), ("gen1_b0_k0", "gen0_seed", 0.5),
                 ("gen2_b0_k0", "gen1_b0_k0", 0.9)])
        recs = build_run_records(run, _StubEncoder())
        assert len(recs) == 1, len(recs)
        r0 = recs[0]
        assert r0.tree_id == ("run_ok", "gen0_seed")
        assert r0.meta.candidate_id == "run_ok:gen0_seed"
        assert r0.cut_id == ("run_ok:gen0_seed", 0)
        assert abs(r0.r - 0.8) < 1e-9, r0.r
        assert abs(r0.weight - 1.0) < 1e-9, r0.weight
        assert r0.p.shape == (C.N_BRANCHES, C.M_SKETCH), r0.p.shape
        assert r0.X_v.shape[0] == 1, r0.X_v.shape     # one trace, no code layers
        assert r0.mask.shape == (C.N_SLOTS, 1)
        print("OK  one lineage       -> 1 CutRecord, r=+0.800, weight=1.0, "
              "p=[4,8], mask=[4,1]")

        # two paths from one parent => 2 records, weights sum to 1
        run2 = Path(td) / "run_two"
        mk(run2, [("gen0_seed", None, 0.1), ("gen1_b0_k0", "gen0_seed", 0.5),
                  ("gen2_b0_k0", "gen1_b0_k0", 0.9),
                  ("gen2_b0_k1", "gen1_b0_k0", 0.7)])
        recs2 = build_run_records(run2, _StubEncoder())
        assert len(recs2) == 2, len(recs2)
        assert {r.occurrence_seq for r in recs2} == {0, 1}
        assert {r.cut_id for r in recs2} == {("run_two:gen0_seed", 0),
                                             ("run_two:gen0_seed", 1)}
        assert abs(sum(r.weight for r in recs2) - 1.0) < 1e-9
        assert torch.equal(recs2[0].X_v, recs2[1].X_v)      # same observation
        assert not torch.equal(recs2[0].p, recs2[1].p)      # different futures
        print("OK  2 real paths      -> 2 records, one X_v, two p, "
              "weights sum to 1.0 (parent-normalised)")

        # control run is REFUSED
        ctl = Path(td) / "run_ctl"
        mk(ctl, [("gen0_seed", None, 0.1), ("gen1_b0_k0", "gen0_seed", 0.5),
                 ("gen2_b0_k0", "gen1_b0_k0", 0.9)], control=True)
        try:
            build_run_records(ctl, _StubEncoder())
            raise AssertionError("control run was accepted")
        except NotOfficialData:
            pass
        print("OK  control run       REFUSED (CONTROL_NOTICE.txt present)")

    # The fusion must KEEP requires_grad: close_task/close_lpse backprop a
    # cotangent THROUGH Gamma. It is never trained here (no optimizer is built,
    # no backward is called on it), but switching the flag off breaks the graph.
    f = _init_fusion()
    assert any(p.requires_grad for p in f.parameters())
    assert sum(p.numel() for p in f.parameters()) == 108_288
    print("OK  fusion init       SlottedFusion = 108,288 params, flag stays ON "
          "(cotangent path needs the graph; nothing trains it here)")

    led = os.environ.get("META_N_REQUEST_LEDGER", "").strip()
    paid = 0
    if led and os.path.isfile(led):
        from meta_n.rpbe.accounting import snapshot
        paid = snapshot()["backend_requests"]
    assert paid == 0
    print("OK  zero-cost          paid backend_requests = {}".format(paid))

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
