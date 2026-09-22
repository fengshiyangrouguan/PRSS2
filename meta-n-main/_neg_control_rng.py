"""Negative control for the wiring fix (issue #1).

Reproduces the PRE-FIX construction -- reducer built without `context_manager`
-- and shows the orchestrator's seed write does not reach the sampler the
reducer uses. Then shows the post-fix construction does. No API, no Gamma.
"""
import random
import sys

sys.path.insert(0, "/root/autodl-tmp/rpbe-sri/meta-n-main")

from meta_n.core.meta_layer import InjectedCode, Trace   # noqa: E402
from meta_n.core.omega import OmegaEngine                # noqa: E402
from meta_n.rpbe.context_reduction import (ContextReducer, install)  # noqa: E402
from meta_n.rpbe.modes import ReductionMode              # noqa: E402
from meta_n.utils.context_manager import ContextManager  # noqa: E402


def traces():
    return [Trace(task_id="t%d" % i, script="s", stdout="o" * 100,
                  success=True) for i in range(6)]


def probe(build):
    eng = object.__new__(OmegaEngine)
    eng.context_manager = ContextManager()
    engine_manager = eng.context_manager
    reducer = build(eng)
    install(eng, reducer)
    # exactly what the orchestrator does, AFTER install swapped in the adapter
    eng.context_manager.rng = random.Random(7)
    holder = eng.context_manager.sample_traces(traces())
    eng.context_manager.truncate_context_stack(
        [InjectedCode(pre_process="def p(): pass", source_depth=2)])
    drew = [t.task_id for t in holder]
    expect = [t.task_id for t in
              random.Random(7).sample(traces(), 4)]
    # The DETERMINISTIC criterion (no 1-in-15 coincidence): which rng object
    # does the sampler the reducer actually calls hold?
    return dict(
        shared=(reducer.context_manager is engine_manager),
        reducer_seed_is_seeded=(reducer.context_manager.rng is not random),
        drew=drew, seeded_rule=expect, follows_seed=(drew == expect))


print("=" * 74)
print("NEGATIVE CONTROL: does the orchestrator's rng write reach the sampler?")
print("=" * 74)

before = probe(lambda e: ContextReducer(ReductionMode.OFFICIAL_TRACE_K4))
after = probe(lambda e: ContextReducer(ReductionMode.OFFICIAL_TRACE_K4,
                                       context_manager=e.context_manager))

for name, r in (("PRE-FIX (no context_manager=)", before),
                ("POST-FIX (context_manager=engine's)", after)):
    print()
    print("  %s" % name)
    print("    reducer shares the engine's manager : %s" % r["shared"])
    print("    the REDUCER's sampler rng is seeded  : %s"
          % r["reducer_seed_is_seeded"])
    print("    traces drawn                         : %s" % ",".join(r["drew"]))
    print("    what seed 7 should draw              : %s"
          % ",".join(r["seeded_rule"]))
    print("    draw follows the search seed         : %s" % r["follows_seed"])

print()
ok = (not before["shared"] and not before["reducer_seed_is_seeded"]
      and after["shared"] and after["reducer_seed_is_seeded"]
      and after["follows_seed"])
print("  RESULT: %s" % ("issue #1 was REAL and is now FIXED" if ok
                        else "unexpected -- inspect above"))
