"""FORENSIC #4: is the held-out failure a DEPTH problem or a SELECTION problem?

Runs entirely on the existing archive. No search re-run, no API, no new Gamma.
Every candidate below is scored with the SAME evaluator `stage_final` used
(reused via the runner module), the same 6 held-out tasks, the same 3 repeats,
and the same per-(seed,task,repeat) test-seed derivation.

THE THREE GROUPS (user-specified)

  1. cross-arm equal-depth
       both arms' best depth-4 candidate by dev macro, chosen through the SAME
       `select_deployable` the protocol uses. The COUNT of depth-4 candidates is
       reported for each arm: picking a max over 2 vs over 10 is a different
       amount of selection, and that difference must not hide inside the score.

  2. the official winner's ancestry
       the official winner is depth 6. Re-run its d4 -> d5 -> d6 chain on test.
       Answers: did official's edge on a task accrue with the extra two layers?

  3. the ours depth-4 winner's descendants
       ours' d4 winner has dev(d4)=0.9547 and a d5 descendant at dev 0.9227.
       Run those on test. If  s_dev(d5) < s_dev(d4)  but  s_test(d5) > s_test(d4),
       then continuing to deepen IMPROVED generalisation while dev-max REJECTED
       it -- i.e. the dev/test gap is selection overfitting (winner's curse), not
       a Gamma defect. If ours' d5/d6 test also degrades, depth 4 is that
       lineage's real peak and the search for the cause moves to #1/#2.

Output per candidate: depth, s_dev, s_test, s_dev - s_test, and all six held-out
per-task scores (macro alone hides which task moved).
"""
import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

OUT = Path("/root/autodl-tmp/sri_r3_seed0")
REPO = Path("/root/autodl-tmp/rpbe-sri/meta-n-main")
sys.path.insert(0, str(REPO))

from meta_n.sri.protocol import (default_profile_path, load_profile,  # noqa: E402
                                 sha256_of)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "sri_runner", REPO / "scripts" / "run_sri_formal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = load_runner()
PROFILE = load_profile(default_profile_path("sri_primary6"))
SLUGS, NAME_OF = R.cohort_ids(PROFILE)
ARGS = argparse.Namespace(evaluator="real",
                          data_dir="/root/autodl-tmp/meta-n-main/data/co_bench",
                          timeout=10, search_seed=0)
EV = R.make_evaluator(ARGS, PROFILE, split="test")

_cache = {}


def test_eval(arm, cid, depth):
    """(macro, per_task) on the held-out split -- same protocol as stage_final."""
    key = (arm, cid)
    if key in _cache:
        return _cache[key]
    mat = R.load_material_from_dir(OUT / "arms" / arm / "archive" / cid,
                                   cid, depth, SLUGS)
    per, n = {}, 0
    for t in SLUGS:
        src = (mat.task_scripts or {}).get(t) if mat is not None else None
        if not src:
            per[t] = None
            continue
        vals = []
        for rep in range(int(PROFILE.test_repeats)):
            seed = int(sha256_of({"s": ARGS.search_seed, "t": t,
                                  "r": rep})[:8], 16) & 0x7FFFFFFF
            ev = EV.evaluate_test(t, src, seed=seed)
            n += 1
            if ev.score is None or not math.isfinite(float(ev.score)):
                continue
            vals.append(float(ev.score))
        per[t] = statistics.median(vals) if vals else None
    got = [v for v in per.values() if v is not None]
    macro = sum(got) / len(got) if got else float("nan")
    _cache[key] = (macro, per)
    return _cache[key]


def archive(arm):
    idx = json.load(open(OUT / "arms" / arm / "archive" / "index.json",
                         encoding="utf-8"))
    for i, c in enumerate(idx.get("candidates") or []):
        c["creation_index"] = i
    return list(idx.get("candidates") or [])


def executable_of(arm):
    def f(c):
        m = R.load_material_from_dir(
            OUT / "arms" / arm / "archive" / str(c.get("candidate_id")),
            str(c.get("candidate_id")), int(c.get("depth") or 1), SLUGS)
        return m is not None and m.is_executable_on(SLUGS)
    return f


def by_depth(cands, d):
    return [c for c in cands if int(c.get("depth") or 0) == d]


def best_at(cands, d, arm):
    pool = by_depth(cands, d)
    if not pool:
        return None, 0
    sel = R.select_deployable(pool, lambda c: c.get("mean_score"),
                              creation_index_of=lambda c: c["creation_index"],
                              is_executable_of=executable_of(arm))
    return sel, len(pool)


def chain_to_root(cands, cid):
    """[cid, parent, grandparent, ...] following parent_id."""
    m = {c["candidate_id"]: c for c in cands}
    out, cur, seen = [], cid, set()
    while cur and cur in m and cur not in seen:
        seen.add(cur)
        out.append(cur)
        cur = m[cur].get("parent_id")
    return out


def children_of(cands, cid):
    return [c for c in cands if c.get("parent_id") == cid]


print("=" * 96)
print("FORENSIC #4 -- depth vs selection, on the existing archive (same evaluator as final)")
print("=" * 96)

plan = {}
for arm in ("official", "predictive"):
    cands = archive(arm)
    counts = {}
    for c in cands:
        counts[int(c.get("depth") or 0)] = counts.get(int(c.get("depth") or 0), 0) + 1
    print()
    print("  %s: %d candidates, depth histogram %s"
          % (arm, len(cands), dict(sorted(counts.items()))))

    d4sel, d4n = best_at(cands, 4, arm)
    print("    depth-4 pool size = %d  -> best-dev = %s (dev %.4f, depth %s)"
          % (d4n, getattr(d4sel, "candidate_id", None),
             getattr(d4sel, "dev_score", float("nan")),
             getattr(d4sel, "structural_depth", None)))
    plan[arm] = {"cands": cands, "d4": d4sel, "d4n": d4n}

# the winner each arm actually shipped
for arm in ("official", "predictive"):
    fin = json.load(open(OUT / "final" / ("%s.json" % arm), encoding="utf-8"))
    f = fin.get("final") or {}
    print("    shipped winner: %s (depth %s, dev %.4f)"
          % (f.get("selected_candidate_id"), f.get("selected_candidate_depth"),
             f.get("selected_candidate_dev_score") or float("nan")))

# ---- assemble the candidate list -------------------------------------------
todo = []
for arm in ("official", "predictive"):
    cands = plan[arm]["cands"]
    d4 = plan[arm]["d4"]
    if d4 is not None:
        todo.append((arm, d4.candidate_id, d4.structural_depth, d4.dev_score,
                     "cross-arm equal-depth (d4 winner)"))
        # descendants of ours' d4 winner
        for ch in children_of(cands, d4.candidate_id):
            todo.append((arm, ch["candidate_id"], int(ch["depth"]),
                         float(ch["mean_score"]), "d4-winner descendant"))
            for g in children_of(cands, ch["candidate_id"]):
                todo.append((arm, g["candidate_id"], int(g["depth"]),
                             float(g["mean_score"]), "d4-winner grandchild"))
    # official winner's ancestry
    win = json.load(open(OUT / "final" / ("%s.json" % arm), encoding="utf-8"))
    wid = (win.get("final") or {}).get("selected_candidate_id")
    if wid:
        m = {c["candidate_id"]: c for c in cands}
        for cid in chain_to_root(cands, wid):
            c = m[cid]
            if int(c.get("depth") or 0) in (4, 5, 6) and cid != (getattr(d4, "candidate_id", None)):
                todo.append((arm, cid, int(c["depth"]), float(c["mean_score"]),
                             "shipped-winner ancestry"))

seen, ordered = set(), []
for row in todo:
    k = (row[0], row[1])
    if k in seen:
        continue
    seen.add(k)
    ordered.append(row)

print()
print("=" * 96)
print("  evaluating %d candidates x %d tasks x %d repeats on the HELD-OUT split"
      % (len(ordered), len(SLUGS), PROFILE.test_repeats))
print("=" * 96)

print()
hdr = ("  %-11s %-14s %-5s %8s %8s %9s  %s"
       % ("arm", "candidate", "depth", "s_dev", "s_test", "dev-test", "why"))
print(hdr)
print("  " + "-" * 92)
rows = []
for arm, cid, depth, sdev, why in ordered:
    stest, per = test_eval(arm, cid, depth)
    rows.append((arm, cid, depth, sdev, stest, per, why))
    print("  %-11s %-14s %-5d %8.4f %8.4f %+9.4f  %s"
          % (arm, cid, depth, sdev, stest, sdev - stest, why))

print()
print("=" * 96)
print("  PER-TASK held-out scores (macro alone hides which task moved)")
print("=" * 96)
short = {t: t[:20] for t in SLUGS}
print("  %-11s %-14s %-5s %s"
      % ("arm", "candidate", "dep", " ".join("%20s" % short[t] for t in SLUGS)))
for arm, cid, depth, sdev, stest, per, why in rows:
    cells = " ".join("%20s" % ("%.4f" % per[t] if per.get(t) is not None else "-")
                     for t in SLUGS)
    print("  %-11s %-14s %-5d %s" % (arm, cid, depth, cells))

json.dump([{"arm": a, "candidate": c, "depth": d, "s_dev": sd, "s_test": st,
            "per_task": p, "why": w} for a, c, d, sd, st, p, w in rows],
          open(OUT / "forensic4_depth_vs_selection.json", "w"), indent=1)
print()
print("  wrote %s" % (OUT / "forensic4_depth_vs_selection.json"))
