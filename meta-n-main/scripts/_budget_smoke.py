"""0-API smoke for the Structural6 Phase-A launcher's BUDGET ARMING.

The first Phase-A crashed for a budget reason and no other: the launcher set
`META_N_DAILY_BUDGET_USD`, which arms `meta_n.utils.cost_tracker`, which needs a
per-model price table and died on `gemini-3.1-pro`. So the thing worth an offline
test is not "does the fuse fire" but "is the DOLLAR mechanism absent and the
REQUEST mechanism present". Three assertions, no network, no LLM:

  1. `arm_relay_env` sets the request cap and the relay backend;
  2. it does NOT leave `META_N_DAILY_BUDGET_USD` set, even if the inherited
     environment had one (the crash condition, reproduced then cleared);
  3. `requests_used()` reads the same counter file `accounting.reserve()` bumps,
     so the outer backstop is not watching a file nobody writes.

Run from the repo root:  python -m scripts._budget_smoke
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meta_n.rpbe.accounting import (CAP_ENV, REQUEST_LEDGER_ENV,  # noqa: E402
                                    backend_request_cap, reserve)
from scripts.run_phase_a_structural6 import (arm_relay_env,      # noqa: E402
                                            requests_used)

print("=" * 74)
print("SMOKE 1: arm_relay_env sets the request cap, not a dollar cap")
print("=" * 74)
# Reproduce the exact crash condition: a dollar cap inherited from the shell.
os.environ["META_N_DAILY_BUDGET_USD"] = "36.0"
os.environ.pop(CAP_ENV, None)
arm_relay_env(200, "https://api-key.xyz/api/v1")
print("  LLM_BACKEND                = %r" % os.environ.get("LLM_BACKEND"))
print("  ALLOW_PAID_API             = %r" % os.environ.get("ALLOW_PAID_API"))
print("  %s = %r" % (CAP_ENV, os.environ.get(CAP_ENV)))
print("  backend_request_cap()      = %d" % backend_request_cap())
print("  RELAY_BASE_URL             = %r" % os.environ.get("RELAY_BASE_URL"))
assert backend_request_cap() == 200, backend_request_cap()
assert os.environ["LLM_BACKEND"] == "relay"

print()
print("=" * 74)
print("SMOKE 2: the dollar tracker cannot be armed by this launcher")
print("=" * 74)
left = os.environ.get("META_N_DAILY_BUDGET_USD")
print("  META_N_DAILY_BUDGET_USD after arming = %r" % left)
assert left is None, (
    "a dollar cap survived arming -- that is what crashed the first run with "
    "KeyError: no pricing for gemini-3.1-pro")

print()
print("=" * 74)
print("SMOKE 3: requests_used() reads the counters accounting.reserve() bumps")
print("=" * 74)
tmp = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp")
ledger = tmp / "_s6_smoke_requests.jsonl"
for stale in (ledger, Path(str(ledger) + ".count")):
    try:
        stale.unlink()
    except OSError:
        pass
os.environ[REQUEST_LEDGER_ENV] = str(ledger)
assert requests_used(ledger) == 0, requests_used(ledger)
for _ in range(3):
    reserve()
print("  after 3 reserve() calls, requests_used() = %d" % requests_used(ledger))
assert requests_used(ledger) == 3, requests_used(ledger)

# And the cap really does stop the NEXT send rather than the one after it.
os.environ[CAP_ENV] = "3"
from meta_n.rpbe.accounting import LLMCallBudgetExceeded  # noqa: E402
try:
    reserve()
except LLMCallBudgetExceeded as e:
    print("  4th reserve() -> %s: %s" % (type(e).__name__, str(e)[:70]))
else:
    raise AssertionError("the cap did not stop request 4")
for stale in (ledger, Path(str(ledger) + ".count")):
    try:
        stale.unlink()
    except OSError:
        pass

print()
print("  ALL THREE SMOKES PASS")
