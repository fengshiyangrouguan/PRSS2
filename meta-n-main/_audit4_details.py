"""Follow-up: the details behind the trace-only verdict.

  0. WHAT are the 6 archives? (mtime, backbone, depth, seed, CONTROL_NOTICE) --
     the identity of the data, before anything is claimed about it.
  1. Why did cut_items() raise TypeError for every candidate? (a str/Path bug
     in the FIRST draft of this audit, kept visible so the corrected numbers can
     be trusted)
  2. Per-candidate composition: traces vs injected-code layers. Every candidate
     has 6 traces and 0..4 code layers, so the retired contract would give
     X_v widths 6..10 while the trace-only contract gives 6 always -- which is
     what makes the width in a records file a CONTRACT discriminator.
  3. The one record in _gem6_traceonly.jsonl whose X_v values are not identical
     to a fresh rebuild, and by how much.
"""
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CODEBERT_PATH", "/root/autodl-tmp/models/codebert-base")
sys.path.insert(0, "/root/autodl-tmp/rpbe-sri/meta-n-main")

import torch                                                     # noqa: E402
from meta_n.rpbe.build_records import build_run_records, cut_items  # noqa: E402
from meta_n.rpbe.encoder import FrozenItemEncoder                # noqa: E402
from meta_n.rpbe.future import FutureSketcher                    # noqa: E402

ARCH = sorted(glob.glob(
    "/root/autodl-tmp/meta-n-main/runs/mtgem_run*/*co_bench_gemini-3.1-pro"))

print("=" * 78)
print("0. the 6 archives -- identity of the data")
print("=" * 78)
print("  %-30s %-19s %-6s %-5s %-6s %-6s %s"
      % ("run dir", "mtime", "ctnox", "cands", "depth", "seed", "backbone"))
for a in ARCH:
    st = os.stat(a)
    ctl = os.path.isfile(os.path.join(a, "CONTROL_NOTICE.txt"))
    cands = [d for d in glob.glob(os.path.join(a, "archive", "*"))
             if os.path.isdir(d)]
    depth = seed = None
    bb = "?"
    cfgp = os.path.join(a, "config.json")
    if os.path.isfile(cfgp):
        try:
            c = json.load(open(cfgp, encoding="utf-8"))
            depth = c.get("max_depth")
            seed = c.get("seed", c.get("search_seed"))
            bb = c.get("model") or c.get("backbone") or "?"
        except Exception:                                     # noqa: BLE001
            pass
    print("  %-30s %-19s %-6s %-5d %-6s %-6s %s"
          % (os.path.basename(a)[:30],
             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
             ctl, len(cands), depth, seed, bb))

print()
print("=" * 78)
print("0b. HOST POLICY of each archive vs the formal profile")
print("=" * 78)
# The formal profile pins all five of these to true. They do NOT affect the
# Phase-C A/B FAIRNESS -- both arms run the identical host -- but they decide
# how to STATE what Gamma learned: if Phase-A used this same policy, the claim
# is "offline predictive retention under the host policy"; if it did not, the
# honest phrasing is "frozen predictive retention deployed across
# search-policies". Neither needs a retrain; the difference is in the write-up.
POLICY = ["consolidate", "regression_guard", "within_task_recursion",
          "focus_current_headroom", "symmetric_trace_sampling"]
PROFILE_POLICY = {k: True for k in POLICY}     # meta_n/configs/sri_primary6.yaml
print("  formal sri_primary6 pins: %s"
      % ", ".join("%s=%s" % (k, PROFILE_POLICY[k]) for k in POLICY))
print()
print("  %-30s %s" % ("run", "  ".join(k[:12] for k in POLICY)))
mismatch = []
for a in ARCH:
    cfgp = os.path.join(a, "config.json")
    vals = {}
    if os.path.isfile(cfgp):
        try:
            c = json.load(open(cfgp, encoding="utf-8"))
            vals = {k: c.get(k, "(absent)") for k in POLICY}
        except Exception:                                     # noqa: BLE001
            pass
    for k in POLICY:
        if vals.get(k, "(absent)") != PROFILE_POLICY[k]:
            mismatch.append((os.path.basename(a)[:30], k, vals.get(k)))
    print("  %-30s %s"
          % (os.path.basename(a)[:30],
             "  ".join("%-12s" % str(vals.get(k, "(absent)"))[:12]
                       for k in POLICY)))
print()
if mismatch:
    print("  MISMATCH vs the formal profile (%d):" % len(mismatch))
    for rid, k, v in mismatch:
        print("    %-30s %-26s archive=%r profile=%r"
              % (rid, k, v, PROFILE_POLICY[k]))
    print("  -> note it in the write-up; it does NOT invalidate the pair and")
    print("     does NOT by itself justify a retrain.")
else:
    print("  RESULT: all five match the formal profile in every archive, so the")
    print("          Gamma was trained under the same host policy it is deployed")
    print("          into. Report as 'offline predictive retention under the")
    print("          host policy'.")

print()
print("=" * 78)
print("1. cut_items() failure in the FIRST draft (kept visible)")
print("=" * 78)
a0 = ARCH[0]
cd0 = sorted(d for d in glob.glob(os.path.join(a0, "archive", "*"))
             if os.path.isdir(d))[0]
try:
    cut_items(Path(a0), os.path.basename(cd0))
    print("  cut_items(Path(...)) works -- the draft passed a str, and "
          "lineage.load_traces does `run_dir / 'archive'`, which is a TypeError "
          "for str. Nothing in the pipeline was wrong; the audit was.")
except Exception:                                             # noqa: BLE001
    traceback.print_exc()

print()
print("=" * 78)
print("2. per-candidate composition (traces vs injected-code layers)")
print("=" * 78)
print("  %-30s %-14s %6s %6s %7s %7s" % ("run", "candidate", "traces", "codes",
                                         "n_items", "X_v_new"))
rows = []
for a in ARCH:
    for cd in sorted(glob.glob(os.path.join(a, "archive", "*"))):
        if not os.path.isdir(cd):
            continue
        cid = os.path.basename(cd)
        try:
            items = cut_items(Path(a), cid)
        except Exception as e:                                # noqa: BLE001
            print("  %-30s %-14s  FAILED %s" % (os.path.basename(a)[:30],
                                                cid[:14], e))
            continue
        ntr = sum(1 for i in items if type(i).__name__ == "Trace")
        ncd = len(items) - ntr
        print("  %-30s %-14s %6d %6d %7d %7d"
              % (os.path.basename(a)[:30], cid[:14], ntr, ncd, len(items), ntr))
        rows.append((os.path.basename(a), cid, ntr, ncd))

print()
print("=" * 78)
print("3. the single value mismatch in _gem6_traceonly.jsonl")
print("=" * 78)
enc = FrozenItemEncoder(os.environ["CODEBERT_PATH"])
sk = FutureSketcher(enc)
fresh = []
for a in ARCH:
    fresh.extend(build_run_records(a, enc, sk))
fidx = {(r.meta.run_id, r.meta.candidate_id.split(":", 1)[-1],
         r.occurrence_seq): r for r in fresh}

for path in ("/root/autodl-tmp/_gem6_traceonly.jsonl",
             "/root/autodl-tmp/_gem5_records.jsonl"):
    print()
    print("  %s" % path)
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        m = r["meta"]
        bare = m["candidate_id"].split(":", 1)[-1]
        key = (m["run_id"], bare, r["occurrence_seq"])
        f = fidx.get(key)
        if f is None:
            print("    %-30s %-14s MISSING in fresh" % (m["run_id"][-14:], bare))
            continue
        stored = torch.tensor(r["X_v"])
        if stored.shape != f.X_v.shape:
            print("    %-30s %-14s occ=%d  width stored=%d fresh=%d  RETIRED"
                  % (m["run_id"][-14:], bare, r["occurrence_seq"],
                     stored.shape[0], f.X_v.shape[0]))
            continue
        d = float((stored - f.X_v).abs().max())
        tag = "identical" if d <= 1e-5 else "DIFFERS max|dx|=%.3e" % d
        if d > 1e-5:
            print("    %-30s %-14s occ=%d  width=%d  %s"
                  % (m["run_id"][-14:], bare, r["occurrence_seq"],
                     stored.shape[0], tag))
            print("        per-row max|dx|: %s"
                  % [round(float(v), 3) for v in
                     (stored - f.X_v).abs().max(dim=1).values])
            print("        q_emb max|dx|   : %.3e"
                  % float((torch.tensor(r["q_emb"])
                           - f.q_emb).abs().max()))
