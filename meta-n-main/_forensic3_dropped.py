"""FORENSIC #3, restricted to the ONLY window where it is clean: depth 2.

WHY THE RESTRICTION
The prompt template reports the section differently by depth:

    depth 2      "## Execution Traces (sampled: F failures, S successes)"
                 -> EVERY trace handed in is rendered. So shown == selected.
    depth >= 3   "## Representative Traces (K of N)"
                 -> only K representatives are rendered, chosen by the NATIVE
                    `_build_prompt` FROM the 4 we handed it.

A first draft of this forensic diffed the parent's pool against the tasks
appearing in the prompt, at every depth. That conflates OUR 4-of-6 choice with
the native K-of-N subsampling that happens afterwards, and it produced absurd
numbers (one task "dropped 60/60"). It was measuring two different operators at
once. `reduction_trace.jsonl` would have separated them; it was not enabled for
this run, so depths >= 3 are NOT analysable and are excluded rather than
guessed at.

Depth 2 is also the ONLY window where the comparison is fair: both arms then
have the identical root, the identical 6-trace pool, the same k=4 and the same
candidate count. So this is the same window as the Shared-root Selection Gain.

0 API, no evaluation.
"""
import hashlib
import collections
import glob
import json
import os
import re

OUT = "/root/autodl-tmp/sri_r3_seed0"
SAMPLED_HDR = re.compile(
    r"^## Execution Traces \(sampled: \d+ failures?, \d+ successes?\)", re.M)
BLOCK = re.compile(r"^--- Task: (\S+) \[(SUCCESS|FAILURE)([^\]]*)\] ---", re.M)


def depth2_calls(arm):
    """UNIQUE d1->d2 decisions, keyed by the prompt's own content.

    A candidate directory INHERITS its ancestors' prompt files, so the same
    d1->d2 prompt appears in every descendant: 24 files per arm collapse to 4
    unique ones for official and 3 for predictive (verified by md5). Counting
    the files as independent observations would multiply one decision by its
    number of descendants -- the first draft of this script did exactly that
    and reported "24 calls", which is why the dedup is keyed on the prompt
    text and not on the path.
    """
    base = os.path.join(OUT, "arms", arm, "archive")
    seen = {}
    for cdir in sorted(glob.glob(os.path.join(base, "*"))):
        sj = os.path.join(cdir, "summary.json")
        if not os.path.isfile(sj):
            continue
        s = json.load(open(sj, encoding="utf-8"))
        pid = s.get("parent_id")
        if not pid:
            continue
        pdir = os.path.join(base, pid, "traces")
        tasks = sorted(os.path.basename(p)[:-5]
                       for p in glob.glob(os.path.join(pdir, "*.json")))
        if not tasks:
            continue
        for pf in sorted(glob.glob(os.path.join(cdir, "omega_prompt_d*.txt"))):
            txt = open(pf, encoding="utf-8", errors="replace").read()
            if not SAMPLED_HDR.search(txt):
                continue                      # depth >= 3 -> not analysable
            shown = {m[0] for m in BLOCK.findall(txt)}
            if not shown:
                continue
            digest = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:10]
            seen.setdefault(digest, (os.path.basename(cdir), tasks, shown,
                                     [t for t in tasks if t not in shown]))
    return [v for _, v in sorted(seen.items())]


print("=" * 78)
print("FORENSIC #3 (clean window) -- which 4 of the 6 tasks each arm keeps")
print("=" * 78)

per_arm = {}
all_tasks = set()
for arm in ("official", "predictive"):
    rows = depth2_calls(arm)
    per_arm[arm] = rows
    print()
    print("  %s -- %d depth-2 call(s)" % (arm, len(rows)))
    for cid, tasks, shown, drop in rows:
        print("    %-14s keep %d: %s" % (cid, len(shown), sorted(shown)))
        if drop:
            print("    %-14s drop  %d: %s" % ("", len(drop), sorted(drop)))
    all_tasks |= {t for _, ts, _, _ in rows for t in ts}

print()
print("=" * 78)
print("SIDE BY SIDE -- share of depth-2 calls that KEEP each task")
print("=" * 78)
print("  %-38s %12s %12s" % ("task", "official", "ours"))
for t in sorted(all_tasks):
    cells = []
    for arm in ("official", "predictive"):
        rows = per_arm[arm]
        if not rows:
            cells.append("--")
            continue
        k = sum(1 for _, _, shown, _ in rows if t in shown)
        cells.append("%d/%d" % (k, len(rows)))
    print("  %-38s %12s %12s" % (t, cells[0], cells[1]))

print()
print("  SAMPLE SIZE WARNING: after dedup this is a HANDFUL of unique d1->d2")
print("  decisions (official 4, ours 3 -> 2 distinct selections). That is the")
print("  entire clean window, so read it as \"what each arm did at d1->d2\",")
print("  and NOT as a rate.")
print()
print("  To make depths >= 3 analysable at all, a future seed must export")
print("  META_N_REDUCTION_TRACE so the 4-of-6 choice is recorded directly.")
