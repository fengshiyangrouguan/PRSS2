"""Per-depth diversity + search-shape table (zero cost, no evaluation).

Requested metrics, per structural depth d:
    N_candidates(d)      how many candidates sit at depth d
    unique_parents(d)    how many DISTINCT parents produced them
    unique_subsets(d)    how many DISTINCT kept-trace sets were used
    max_s_dev(d)         the best dev macro among them

WHAT IS EXACT AND WHAT IS A PROXY -- this distinction is the point.

  depth 2   EXACT. The prompt renders every handed-in trace under
            "## Execution Traces (sampled: F failures, S successes)", so the
            kept set is readable straight off the prompt.

  depth >=3 PROXY. There the template switches to "## Representative Traces
            (K of 4)" and renders only K of the 4, so the kept set is NOT
            recoverable. The K-representative SECTION is a function of the kept
            4 (the native chooser picks from them), so hashing that section
            approximates subset identity. It can over-count (two different
            4-sets yielding the same rendered representatives) and is labelled
            as a proxy everywhere it is used.

WHY depth >= 3 has no exact record: nothing persists the reduction's decision.
`proposal_slots.jsonl` carries slot/parent/depth/temperature but no trace ids,
and `reduction_trace.jsonl` was not exported for this run. A future seed must
export META_N_REDUCTION_TRACE to make this exact.

DEDUPLICATION: candidate directories INHERIT their ancestors' prompt files, so
raw file counts are inflated by the number of descendants. Every count here is
deduplicated by content (md5 of the prompt / of the section), not by path.
"""
import collections
import glob
import hashlib
import json
import os
import re

OUT = "/root/autodl-tmp/sri_r3_seed0"
SAMPLED = re.compile(
    r"^## Execution Traces \(sampled: \d+ failures?, \d+ successes?\)", re.M)
REPR = re.compile(r"^## Representative Traces \(\d+ of (\d+)\)\n(.*?)(?=\n## |\Z)",
                  re.M | re.S)
BLOCK = re.compile(r"^--- Task: (\S+) \[(SUCCESS|FAILURE)([^\]]*)\] ---", re.M)


def own_prompt(cdir, depth):
    """The prompt that generated THIS candidate: d<its own depth>."""
    return os.path.join(cdir, "omega_prompt_d%d.txt" % depth)


def kept_exact(txt):
    if not SAMPLED.search(txt):
        return None
    return tuple(sorted({m[0] for m in BLOCK.findall(txt)}))


def repr_proxy(txt):
    """Hash of the K-representative section (proxy for the kept 4)."""
    m = REPR.search(txt)
    if not m:
        return None
    return hashlib.md5(m.group(2).encode("utf-8", "replace")).hexdigest()[:10]


print("=" * 100)
print("PER-DEPTH DIVERSITY AND SEARCH SHAPE")
print("=" * 100)
for arm in ("official", "predictive"):
    idx = json.load(open(os.path.join(OUT, "arms", arm, "archive", "index.json"),
                         encoding="utf-8"))
    cands = idx.get("candidates") or []
    buckets = collections.defaultdict(list)
    n_oracle = 0
    for c in cands:
        # `merge_oracle` is a SYNTHESIZED candidate the protocol forbids from
        # being deployed, and its dev score is set to the archive MAX. Leaving
        # it in a depth bucket silently inflates that depth's max_s_dev (and the
        # original draft of this table did exactly that).
        if str(c.get("candidate_id")) == "merge_oracle" or c.get("synthesized"):
            n_oracle += 1
            continue
        buckets[int(c.get("depth") or 0)].append(c)

    print()
    print("  %s   (excluded %d synthesized oracle candidate(s))" % (arm, n_oracle))
    print("    %-6s %6s %9s %9s %11s %11s %9s"
          % ("depth", "N_cand", "uniq_par", "uniq_sub", "exact?", "max_s_dev",
             "dev@d"))
    prev_best = None
    for d in sorted(buckets):
        cs = buckets[d]
        parents = {c.get("parent_id") for c in cs if c.get("parent_id")}
        devs = [float(c["mean_score"]) for c in cs if c.get("mean_score")
                is not None]
        # subset identity from each candidate's OWN depth-d prompt
        exact_seen, proxy_seen, any_exact = set(), set(), False
        for c in cs:
            cd = os.path.join(OUT, "arms", arm, "archive", str(c["candidate_id"]))
            pf = own_prompt(cd, d)
            if not os.path.isfile(pf):
                continue
            txt = open(pf, encoding="utf-8", errors="replace").read()
            k = kept_exact(txt)
            if k is not None:
                exact_seen.add(k)
                any_exact = True
            else:
                pr = repr_proxy(txt)
                if pr:
                    proxy_seen.add(pr)
        # The depth>=3 "proxy" does NOT isolate the subset: it hashes the
        # rendered K representatives, whose CONTENT differs between parents even
        # when the kept 4 tasks are identical. Reporting it as subset diversity
        # would be wrong, so it is suppressed rather than shown misleadingly.
        nsub = len(exact_seen) if any_exact else None
        print("    %-6d %6d %9d %9s %11s %11.4f %9s"
              % (d, len(cs), len(parents),
                 (nsub if nsub is not None else "n/m"),
                 "EXACT" if any_exact else "not meas.",
                 max(devs) if devs else float("nan"),
                 "" if prev_best is None else
                 ("%.4f" % (float(max(devs)) - prev_best))))
        if devs:
            prev_best = max(devs)

print()
print("=" * 100)
print("  READING: `uniq_sub` is the trace-subset diversity at that depth.")
print("  `uniq_par` shows whether the depth was reached through many parents")
print("  (broad exploration) or few (repeat exploitation). `dev@d` is the gain")
print("  in best-dev over the previous depth -- negative means the deeper layer")
print("  did NOT beat what dev-max already had.")
print()
print("  EXACT only at depth 2. At depth >= 3 the subset column is a PROXY built")
print("  from the K-representative section, because the kept 4 is not persisted")
print("  anywhere (no reduction_trace.jsonl, no trace ids in proposal_slots).")
