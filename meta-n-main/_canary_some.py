"""Canary a HAND-PICKED set of models, twice each, to catch minute-level flapping.

The full-catalog scan (`_canary_all.py`) is a snapshot. Provisioning on this
relay rotates -- a model that answers `model_not_provisioned` now can serve
minutes later, which is why "I remember gpt-5.5 working" is a real signal and
not a mistake. So this re-probes a short list TWICE and reports both rounds, so
a single unlucky sample is not read as a verdict.

Usage: python _canary_some.py model1 model2 ...
"""
import asyncio
import importlib.util
import sys

REPO = "/root/autodl-tmp/rpbe-sri/meta-n-main"

# `_probe_backbones` parses `sys.argv[1]` as an int AT MODULE LEVEL, so import
# it with the argv blanked or a model name crashes the int() -- then keep our
# own list aside, since the import must not see it.
MODELS = sys.argv[1:] or ["grok-4.6", "grok-4.5", "gpt-5.5", "gemini-3.1-pro"]
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location(
    "pb", REPO + "/_probe_backbones.py")
PB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PB)
sys.argv = _saved_argv

ROUNDS = 2


async def main() -> int:
    print("=" * 88)
    print("TARGETED CANARY  (%d models x %d rounds)" % (len(MODELS), ROUNDS))
    print("=" * 88)
    verdicts = {}
    for r in range(ROUNDS):
        print("--- round %d ---" % (r + 1), flush=True)
        for m in MODELS:
            try:
                v, d, meta = await PB.canary(m)
            except BaseException as e:                          # noqa: BLE001
                v, d, meta = "PROBE-CRASH", str(e)[:70], {}
            verdicts.setdefault(m, []).append(v)
            print("  %-18s %-17s %-40s rtok=%s"
                  % (m, v, str(d)[:40], meta.get("reasoning_tokens")), flush=True)
        if r + 1 < ROUNDS:
            await asyncio.sleep(20)
    print()
    print("-" * 88)
    print("%-18s %s" % ("model", "rounds"))
    for m in MODELS:
        vs = verdicts[m]
        mark = "  <== USABLE" if "REAL" in vs else ""
        print("%-18s %s%s" % (m, " | ".join(vs), mark))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
