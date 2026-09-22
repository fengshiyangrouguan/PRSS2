"""Negative control for the RNG-stream split (the blocking audit finding).

THE CLAIM: the official and predictive arms must draw the SAME parents from
`Archive.select_parents` for the same `search_seed`. They did not, because ONE
`random.Random` object served both roles:

    orchestrator:  self.rng = random.Random(seed)
                   self.omega.context_manager.rng = self.rng     # trace sampling
                   self.archive.select_parents(..., rng=self.rng) # parent choice

The official arm's sampler CONSUMES that stream (`sample_traces -> rng.sample`);
the predictive arm's Gamma selector consumes NOTHING. So the shared stream is
advanced by one arm and not the other, and from iteration 2 the second parent
diverges -- mixing "which four traces" with "different outer random draws".

This probe runs the REAL `Archive.select_parents` (not a mock) in the exact
order the orchestrator calls things, once per arm, for several iterations, and
reports the chosen parents. `beam_width=2` is the profile's value and matters:
with n=2 exactly one parent is a reserved elite (deterministic) and the other
comes from `rng.choices`, which is where the arms can diverge.

Zero API, zero model, no Gamma.
"""
import os
import random
import sys

sys.path.insert(0, "/root/autodl-tmp/rpbe-sri/meta-n-main")

from meta_n.core.archive import Archive, Candidate            # noqa: E402
from meta_n.core.evolutionary_orchestrator import (           # noqa: E402
    _context_rng_seed)

SEED = 20260923
BEAM_WIDTH = 2
ITERS = 5
POOL_TASKS = ["aircraft_landing", "assignment_problem", "assortment_problem",
              "bin_packing", "capacitated_warehouse_location",
              "common_due_date_scheduling"]


def build_archive():
    """A realistic small archive: several distinct, extendable candidates."""
    a = Archive()
    specs = [("c0", 0.91), ("c1", 0.77), ("c2", 0.55), ("c3", 0.34),
             ("c4", 0.12)]
    for cid, sc in specs:
        a.add(Candidate(candidate_id=cid, depth=2, mean_score=sc,
                        per_task_scores={POOL_TASKS[0]: sc,
                                         POOL_TASKS[1]: sc},
                        num_children=0))
    return a


def run_arm(arm, shared: bool):
    """Return the parent ids chosen at each iteration for one arm.

    `shared=True` reproduces the PRE-FIX wiring (one object for both roles);
    `shared=False` is the fixed wiring (two streams, both seeded from SEED).
    """
    if shared:
        outer = context = random.Random(SEED)
    else:
        outer = random.Random(SEED)
        context = random.Random(_context_rng_seed(SEED))
    archive = build_archive()
    pool = archive.breedable_pool(max_depth=6)
    picks = []
    for _ in range(ITERS):
        # (1) the arm's trace step, in the order the engine does it
        if arm == "official":
            # ContextManager.sample_traces: the only consumer of `context`
            # (6-trace pool, max_total=4 -> rng.sample, i.e. a real draw)
            context.sample(list(range(6)), 4)
        # predictive: the Gamma selector is deterministic; it draws NOTHING
        # (2) the outer search's parent selection, same call as the orchestrator
        parents = archive.select_parents(BEAM_WIDTH, rng=outer, pool=pool)
        picks.append(tuple(p.candidate_id for p in parents))
    return picks


def report(title, before, after):
    print("  %s" % title)
    print("    %-10s %s" % ("iter", "  ".join("arm=%-10s" % a for a in
                                             ("official", "predictive"))))
    for i in range(ITERS):
        print("    %-10d %-17s %-17s %s"
              % (i + 1, before[0][i], before[1][i],
                 "MATCH" if before[0][i] == before[1][i] else "DIFFER"))
    print()
    print("  %s" % after)


print("=" * 78)
print("RNG-STREAM SPLIT: do the two arms select the SAME parents?")
print("=" * 78)
print("  seed=%d  beam_width=%d  iterations=%d" % (SEED, BEAM_WIDTH, ITERS))
print()

pre = [run_arm("official", shared=True), run_arm("predictive", shared=True)]
report("PRE-FIX (one shared stream)", pre,
       "arms match at every iteration: %s"
       % all(pre[0][i] == pre[1][i] for i in range(ITERS)))

post = [run_arm("official", shared=False), run_arm("predictive", shared=False)]
report("POST-FIX (outer + context streams)", post,
       "arms match at every iteration: %s"
       % all(post[0][i] == post[1][i] for i in range(ITERS)))

print()
first_div_pre = next((i + 1 for i in range(ITERS) if pre[0][i] != pre[1][i]),
                     None)
first_div_post = next((i + 1 for i in range(ITERS) if post[0][i] != post[1][i]),
                      None)
print("  PRE-FIX  first divergent iteration : %s" % first_div_pre)
print("  POST-FIX first divergent iteration : %s" % first_div_post)
print()
print("  NOTE on the iteration index: this probe runs the trace step BEFORE")
print("  parent selection, so the shared stream is already offset by the first")
print("  parent draw and divergence shows at iteration 1. In the real loop the")
print("  order is select_parents -> generate, so the first iteration's parents")
print("  agree and divergence starts at iteration 2. The index is an artifact")
print("  of the ordering; the finding does not depend on it.")
print()
ok = first_div_pre is not None and first_div_post is None
print("  RESULT: %s"
      % ("the confound was REAL (the arms chose different parents) and is "
         "FIXED (they now match at every iteration)" if ok
         else "unexpected -- inspect the table above"))

# Reproducibility must survive the split: same seed, same run.
r1 = run_arm("official", shared=False)
r2 = run_arm("official", shared=False)
print("  same seed reproduces the same run  : %s" % (r1 == r2))
