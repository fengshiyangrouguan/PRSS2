"""seed-0 metrics, in the FROZEN final structure (user-frozen 2026-09-23).

MAIN (mechanism)
    Shared-root Selection Gain  SG_root^(m) = mean over C_1 of
                                                [s_dev(c) - s_dev(x_root)]
    i.e. ONLY the d1 -> d2 transition, because at that point the two arms share
    the identical root, the identical initial trace pool, the same k=4, the same
    candidate count and the same model/generation budget. The ONLY treatment is
    which traces the strategy keeps.

    The full-tree Selection Yield was REJECTED as a headline: the arms' parents
    have already diverged, the per-depth edge counts differ, one arm has a
    d5->d6 edge the other lacks, and deep parents are near saturation so their
    gains are structurally smaller. Averaging all edges answers "how did two
    different search trees move", not "did Gamma pick more useful traces".

APPENDIX (4 items, fixed)
    1 archive-best dev by iteration
    2 parent->child dev gain by depth -- full table, NO cross-depth average
    3 per-task held-out final score            (needs the final stage)
    4 search budget and cost

0 API, no CodeBERT, no re-evaluation -- safe to run while `final` is evaluating
(a CO-Bench score counts instances finishing inside a 10s timeout, so a heavy
job here would move the number being measured).
"""
import json

OUT = "/root/autodl-tmp/sri_r3_seed0"
ARMS = ("official", "predictive")


def index(arm):
    with open("{}/arms/{}/archive/index.json".format(OUT, arm),
              encoding="utf-8") as f:
        return json.load(f)


def cands_of(arm):
    return {c["candidate_id"]: c
            for c in (index(arm).get("candidates") or [])}


def root_of(cands):
    """The shared root: the depth-1 seed candidate."""
    roots = [c for c in cands.values() if int(c.get("depth") or 0) == 1]
    if len(roots) != 1:
        raise SystemExit("expected exactly 1 depth-1 candidate, got %d"
                         % len(roots))
    return roots[0]


print("=" * 78)
print("MAIN -- Shared-root Selection Gain  (d1 -> d2 ONLY)")
print("=" * 78)
sg = {}
for arm in ARMS:
    cands = cands_of(arm)
    r = root_of(cands)
    s_root = float(r["mean_score"])
    c1 = [c for c in cands.values()
          if int(c.get("depth") or 0) == 2 and c.get("parent_id") == r["candidate_id"]]
    gains = [float(c["mean_score"]) - s_root for c in c1]
    sg[arm] = sum(gains) / len(gains)
    print("  %-11s root=%s s_root=%.4f  |C_1|=%d  SG_root = %+.4f"
          % (arm, r["candidate_id"], s_root, len(c1), sg[arm]))
    print("               children: %s"
          % ", ".join("%s=%.4f" % (c["candidate_id"], float(c["mean_score"]))
                      for c in sorted(c1, key=lambda x: x["candidate_id"])))
print()
print("  paired gap  dSG_root = SG_root(Gamma) - SG_root(Official) = %+.4f"
      % (sg["predictive"] - sg["official"]))
print("  (single seed -- a preliminary observation, not the paper's result;")
print("   the frozen protocol reports a paired mean+-std over the 5 seeds.)")

print()
print("=" * 78)
print("APPENDIX 1 -- archive-best dev by iteration")
print("=" * 78)
for arm in ARMS:
    conv = json.load(open("{}/arms/{}/convergence.json".format(OUT, arm),
                          encoding="utf-8"))
    print("  %-11s %s" % (arm, "  ".join("%.4f" % v for v in conv)))

print()
print("=" * 78)
print("APPENDIX 2 -- parent->child dev gain by structural depth")
print("=" * 78)
print("  (full table; deliberately NO cross-depth average -- see the module doc)")
print("  %-12s %-18s %-18s" % ("transition", "official", "predictive"))
for d in (2, 3, 4, 5, 6):
    cells = []
    for arm in ARMS:
        cands = cands_of(arm)
        g = [float(c["mean_score"]) - float(cands[c["parent_id"]]["mean_score"])
             for c in cands.values()
             if int(c.get("depth") or 0) == d and c.get("parent_id") in cands]
        cells.append("%+.4f (n=%d)" % (sum(g) / len(g), len(g)) if g else "--")
    print("  %-12s %-18s %-18s" % ("d%d->d%d" % (d - 1, d), cells[0], cells[1]))

print()
print("=" * 78)
print("APPENDIX 4 -- search budget and cost")
print("=" * 78)
for arm in ARMS:
    c = json.load(open("{}/arms/{}/cost_summary.json".format(OUT, arm),
                       encoding="utf-8"))
    rows = [json.loads(l) for l in
            open("{}/arms/{}/proposal_slots.jsonl".format(OUT, arm),
                 encoding="utf-8") if l.strip()]
    phys = sum((r.get("call_accounting") or {}).get("api_calls", 0) for r in rows)
    logi = sum((r.get("call_accounting") or {}).get("evaluator_calls", 0)
               for r in rows)
    ret = sum((r.get("call_accounting") or {}).get("retry_count", 0) for r in rows)
    print("  %-11s slots=%-3d logical=%-4d physical=%-4d retries=%-3d "
          "prompt=%-8d completion=%d"
          % (arm, len(rows), logi, phys, ret,
             int(c.get("outer_prompt_tokens") or 0),
             int(c.get("outer_completion_tokens") or 0)))
print()
print("  NOTE ON FAIRNESS (corrected): the arms did NOT make the same number of")
print("  PHYSICAL calls, and that is not what makes them comparable. Fairness")
print("  rests on the IDENTICAL proposal slots, retry/call policy and output")
print("  cap; a physical-call difference needs its source stated, not waved at.")
print("  Likewise the completion-token difference is 'more was generated' --")
print("  it is NOT evidence that Gamma cost that much extra.")
