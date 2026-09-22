"""Phase-A GENERATION-contract audit (0 API): was Omega handed the FULL pool?

WHY THIS IS A SEPARATE QUESTION FROM THE RECORDS AUDIT
`_audit2`/`_audit4` proved which builder produced a RECORDS file. They say
nothing about the regime the ARCHIVES were generated under. A commit
(`3b1a6b2`) had temporarily changed the global ContextManager default
`sample_traces(max_total=20)` to 4 and was reverted in `9625fa4`. Had the six
`mtgem_*` archives been generated while that edit was live, every observation
Omega consumed in Phase A would already have been restricted to 4 of 6 traces
-- by a rule that is neither the native Official rule nor a Gamma rule -- and
the clean Gamma would have been trained on a trajectory produced under a third
rule. That is a Gamma-validity question.

HOW IT IS DECIDED, WITHOUT TRUSTING GIT OR MTIMES
Each archive keeps the exact prompt Omega was given:
`archive/<candidate>/omega_prompt_d<depth>.txt`. Both prompt templates
(prompts.py:146 and :282) head the trace section with the COUNT THAT WAS HANDED
IN, and omega.py:431-432 computes those counts from the very list it renders:

    depth 2    "## Execution Traces (sampled: F failures, S successes)"
               -> handed_in = F + S, and every handed-in trace is rendered
    depth >=3  "## Representative Traces (K of N)"
               -> handed_in = N; the native path then shows K representatives

So the prompt reports its own input width at EVERY depth. Every candidate in
these archives holds exactly 6 traces and the traces token budget is ~63.7k
against a ~9.6k prompt, so the native rule (cap 20, everything fits) must
report 6 everywhere. The temporary global cap of 4 would report 4.

    every prompt reports 6 -> NATIVE Official trajectories
    any prompt reports 4   -> the global-4 edit was live; STOP and re-plan

A first draft counted `--- Task:` blocks instead. That over-counts: at depth 2
other sections reuse the marker, and at depth >=3 the section holds only the K
representatives. The header is the sampler's own report and needs no such
guessing, so it is the primary discriminator here and the block count is kept
only as a within-section cross-check.
"""
import glob
import os
import re
import sys

os.environ.setdefault("CODEBERT_PATH", "/root/autodl-tmp/models/codebert-base")
TREE = "/root/autodl-tmp/rpbe-sri/meta-n-main"
sys.path.insert(0, TREE)

from pathlib import Path                                            # noqa: E402
from meta_n.rpbe.lineage import load_traces                         # noqa: E402
from meta_n.utils.context_manager import ContextManager              # noqa: E402

ARCH = sorted(glob.glob(
    "/root/autodl-tmp/meta-n-main/runs/mtgem_run*/*co_bench_gemini-3.1-pro"))

H_SAMPLED = re.compile(
    r"^## Execution Traces \(sampled: (\d+) failures?, (\d+) successes?\)",
    re.M)
H_REPR = re.compile(r"^## Representative Traces \((\d+) of (\d+)\)", re.M)
BLOCK = re.compile(r"^--- Task: (\S+) \[", re.M)

cm = ContextManager()
print("=" * 78)
print("Phase-A generation contract: how many traces Omega was HANDED")
print("=" * 78)
print("  header counts come from omega.py:431-432, i.e. the rendered list")
print("  traces_budget=%d tokens vs a ~9.6k prompt, so the native cap of 20"
      % cm.budget.traces_budget)
print("  cannot bite -- a full pool MUST be reported.")
print()
hdr = "  %-30s %-13s %-4s %6s %8s %9s %-9s %s"
print(hdr % ("run", "candidate", "d", "pool", "handed_in", "rendered",
             "header", "verdict"))

rows = []
for a in ARCH:
    rid = os.path.basename(a)
    for cd in sorted(glob.glob(os.path.join(a, "archive", "*"))):
        if not os.path.isdir(cd):
            continue
        cid = os.path.basename(cd)
        try:
            pool_n = len(list(load_traces(Path(a), cid)))
        except Exception as e:                                      # noqa: BLE001
            print("  %-30s %-13s pool load failed: %s" % (rid[:30], cid[:13], e))
            continue
        for p in sorted(glob.glob(os.path.join(cd, "omega_prompt_d*.txt"))):
            depth = int(re.search(r"d(\d+)", os.path.basename(p)).group(1))
            txt = open(p, encoding="utf-8", errors="replace").read()
            ms, mr = H_SAMPLED.search(txt), H_REPR.search(txt)
            if ms:
                handed = int(ms.group(1)) + int(ms.group(2))
                kind, rendered_expect = "sampled", handed
            elif mr:
                handed = int(mr.group(2))
                kind, rendered_expect = "repr", int(mr.group(1))
            else:
                print("  %-30s %-13s %-4d %6d %8s %9s %-9s %s"
                      % (rid[:30], cid[:13], depth, pool_n, "-", "-",
                         "none", "other"))
                rows.append((rid, cid, depth, pool_n, None, None, "none", "other"))
                continue
            nblk = len(BLOCK.findall(txt))
            verdict = ("native" if handed == pool_n
                       else "capped" if handed == 4 and pool_n > 4
                       else "other")
            rows.append((rid, cid, depth, pool_n, handed, nblk, kind, verdict))
            print("  %-30s %-13s %-4d %6d %8d %9d %-9s %s"
                  % (rid[:30], cid[:13], depth, pool_n, handed, nblk, kind,
                     verdict))

tot = {}
for r in rows:
    tot[r[7]] = tot.get(r[7], 0) + 1
print()
print("  prompts examined : %d" % len(rows))
print("  verdicts         : %s" % tot)
by_depth = {}
for r in rows:
    if r[4] is not None:
        by_depth.setdefault(r[2], []).append((r[3], r[4]))
print("  by depth (pool, handed_in) distinct pairs:")
for d in sorted(by_depth):
    pairs = sorted(set(by_depth[d]))
    print("    depth %-3d %s" % (d, pairs))
print()
if tot.get("capped") or tot.get("other"):
    print("  RESULT: NOT every prompt reports the full pool. The archives were")
    print("          not produced under the native rule -- investigate before")
    print("          trusting the clean Gamma.")
else:
    print("  RESULT: EVERY prompt reports the FULL pool at EVERY depth, so the")
    print("          six Phase-A archives are native-Official trajectories. The")
    print("          temporary global k=4 default was NOT in force for them.")
