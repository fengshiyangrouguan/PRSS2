"""CO-Bench archive-productivity preflight (user protocol 2026-09-17).

Answers ONE question before any paid Phase A run is authorised:

    can a strong-enough candidate model produce an archive with real
    child / grandchild structure on CO-Bench, without the evolved solver
    systematically calling llm()?

A weak model (Qwen2.5-Coder-1.5B) produced archive_size=1 with every child
rejected, so there was no lineage at all -- "the model is too weak" is a
legitimate outcome, not evidence against the method. This harness measures the
same six numbers on a model that can actually solve CO-Bench tasks.

Six metrics per run (the user's list):
    1. archive_size
    2. accepted children
    3. eligible parent->child->grandchild lineages
    4. unique trees
    5. inner llm() calls
    6. share of child solvers containing llm()/llm_batch()

Acceptance (user-set):
    * at least some runs reach archive_size >= 3;
    * a real eligible grandchild lineage appears;
    * inner LLM calls are not a systematic explosion.

Everything is local: `paid_backend_requests` must stay 0 throughout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")

REPO = Path(__file__).resolve().parent.parent
TASKS_DEFAULT = [
    "Bin packing - one-dimensional",
    "Common due date scheduling",
    "Constrained guillotine cutting",
]


def _run_one(task: str, run_idx: int, out_root: Path, *, iterations: int,
             base_url: str, model: str, timeout: int) -> dict:
    out = out_root / "run_{}_{}".format(re.sub(r"[^A-Za-z0-9]+", "_", task),
                                        run_idx)
    if out.exists():
        shutil.rmtree(out)
    env = dict(os.environ)
    env.update({
        "LLM_BACKEND": "local",
        "LOCAL_LLM_BASE_URL": base_url,
        "LOCAL_LLM_MODEL": model,
        "META_N_REQUEST_LEDGER": str(out_root / "requests.jsonl"),
        "HF_ENDPOINT": "https://hf-mirror.com",
    })
    cmd = [sys.executable, "-m", "meta_n.main",
           "--benchmark", "co_bench", "--bench-tasks", task,
           "--max-iterations", str(iterations), "--no-early-stop",
           "--beam-width", "1", "--beam-candidates", "1",
           "--max-tokens", "1024", "--model", model,
           "--reduction-mode", "official",
           "--use-archive", "--benchmark-config", "none",
           "--output-dir", str(out)]
    print("  $ {}".format(" ".join(cmd[:8] + ["...", task])), flush=True)
    try:
        subprocess.run(cmd, cwd=str(REPO), env=env, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        return {"task": task, "run": run_idx, "error": "timeout"}

    run_dirs = sorted(p for p in out.glob("*") if p.is_dir())
    if not run_dirs:
        return {"task": task, "run": run_idx, "error": "no run dir"}
    rd = run_dirs[-1]
    return _collect(rd, task, run_idx)


def _collect(run_dir: Path, task: str, run_idx: int) -> dict:
    from meta_n.rpbe.lineage import extract_lineages

    rec = {"task": task, "run": run_idx, "run_dir": str(run_dir)}
    idx = run_dir / "archive" / "index.json"
    cands = []
    if idx.is_file():
        cands = json.loads(idx.read_text()).get("candidates", [])
    rec["archive_size"] = len(cands)
    rec["accepted_children"] = sum(1 for c in cands if c.get("parent_id"))
    rec["unique_trees"] = 1 if cands else 0

    # eligible two-step lineages (real grandchild required)
    try:
        lins = extract_lineages(run_dir)
        rec["eligible_lineages"] = len(lins)
        rec["lineage_trees"] = len({ln.tree_id for ln in lins})
    except Exception as e:                                      # noqa: BLE001
        rec["eligible_lineages"] = 0
        rec["lineage_error"] = "{}: {}".format(type(e).__name__, e)

    # inner LLM calls: the per-call inner log is authoritative
    inner = run_dir / "llm_io" / "inner.jsonl"
    rec["inner_calls"] = (sum(1 for _ in inner.open()) if inner.is_file() else 0)

    # how many child solvers call llm()/llm_batch()?
    kids = [c for c in cands if c.get("parent_id")]
    with_llm = 0
    for c in kids:
        tdir = run_dir / "archive" / str(c["candidate_id"]) / "traces"
        for py in sorted(tdir.glob("*.py")) if tdir.is_dir() else []:
            src = py.read_text(encoding="utf-8", errors="replace")
            if re.search(r"\bllm_batch\s*\(|\bllm\s*\(", src):
                with_llm += 1
                break
    rec["children_with_llm"] = with_llm
    rec["children_with_llm_ratio"] = (with_llm / len(kids)) if kids else None
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=TASKS_DEFAULT)
    ap.add_argument("--runs-per-task", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="qwen3-coder-30b-a3b")
    ap.add_argument("--out", default="/root/autodl-tmp/preflight")
    ap.add_argument("--timeout", type=int, default=5400)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    req_ledger = out_root / "requests.jsonl"
    if req_ledger.exists():
        req_ledger.unlink()

    rows = []
    for task in args.tasks:
        for r in range(args.runs_per_task):
            print("[{}] run {}/{}".format(task, r + 1, args.runs_per_task),
                  flush=True)
            rows.append(_run_one(task, r, out_root, iterations=args.iterations,
                                 base_url=args.base_url, model=args.model,
                                 timeout=args.timeout))

    print()
    print("=" * 96)
    print("CO-BENCH ARCHIVE-PRODUCTIVITY PREFLIGHT")
    print("=" * 96)
    hdr = ("{:<34} {:>3} {:>6} {:>7} {:>9} {:>7} {:>7} {:>8}")
    print(hdr.format("task", "run", "arch", "child", "lineage", "trees",
                     "inner", "kid_llm"))
    for r in rows:
        if r.get("error"):
            print("{:<34} {:>3}  ERROR {}".format(r["task"][:34], r["run"],
                                                  r["error"]))
            continue
        print(hdr.format(
            r["task"][:34], r["run"], r["archive_size"], r["accepted_children"],
            r["eligible_lineages"], r["unique_trees"], r["inner_calls"],
            r["children_with_llm"]))
    print("-" * 96)
    ok = [r for r in rows if not r.get("error")]
    n_arch3 = sum(1 for r in ok if r["archive_size"] >= 3)
    n_lin = sum(1 for r in ok if r["eligible_lineages"] > 0)
    inner_total = sum(r["inner_calls"] for r in ok)
    paid = 0
    if req_ledger.is_file():
        for line in req_ledger.open():
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("request_kind") not in ("mock", "local"):
                    paid += 1
            except json.JSONDecodeError:
                pass
    print("runs                        : {}".format(len(ok)))
    print("runs with archive_size >= 3 : {}".format(n_arch3))
    print("runs with a real lineage    : {}".format(n_lin))
    print("inner llm() calls (total)   : {}".format(inner_total))
    print("paid backend requests       : {}".format(paid))
    verdict = ("PASS" if (n_arch3 > 0 and n_lin > 0 and paid == 0)
               else "NOT YET")
    print("VERDICT                     : {}".format(verdict))
    (out_root / "preflight_summary.json").write_text(
        json.dumps({"rows": rows, "n_arch3": n_arch3, "n_lineage": n_lin,
                    "inner_total": inner_total, "paid": paid,
                    "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
