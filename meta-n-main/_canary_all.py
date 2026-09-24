"""Canary the WHOLE relay catalog: which models actually answer right now?

WHY THIS EXISTS SEPARATELY FROM `_probe_backbones.py`. That script has a
hardcoded 10-model list from 2026-09-21 and NO stage-1-only mode -- it always
falls through to stage 2, which fires the real 64k aircraft solver prompt at
every survivor. The question here is narrower and much cheaper: *which models
are live*, before anything is chosen.

It reuses `_probe_backbones.canary` by IMPORT rather than copying it, so the
call path is literally the same object the real run uses -- the per-model
parameter branches (max_completion_tokens for gpt-5.x, temperature pinned for
kimi-k2) cannot drift between the probe and the thing being probed.

The canary demands the exact product `178451`, because this relay has been
observed to hand back canned 200s and its `model_not_provisioned` /
`capacity_unavailable` states come and go minute to minute -- "HTTP 200" is not
evidence a model works.

Cost: one ~30-token prompt per model with output capped at 512. This is a
liveness check, not a benchmark -- nothing here measures quality.

Usage:  python _canary_all.py [max_tokens]      (default 512)
"""
import asyncio
import importlib.util
import json
import subprocess
import sys

REPO = "/root/autodl-tmp/rpbe-sri/meta-n-main"
RELAY = "https://api-key.xyz/api/v1"


def load_probe():
    spec = importlib.util.spec_from_file_location(
        "pb", REPO + "/_probe_backbones.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PB = load_probe()


def catalog():
    """Fetch the catalog with curl, not urllib.

    urllib gets a hard 403 from Cloudflare on this endpoint while curl succeeds:
    the block is on the request's fingerprint (urllib's default
    `Python-urllib/3.x` User-Agent), not on the key -- the same bearer token
    works from curl. Shelling out keeps the probe on the path already known to
    work instead of adding headers until it happens to look like a browser.
    """
    out = subprocess.run(
        ["curl", "-s", "--max-time", "30",
         "-H", "Authorization: Bearer " + PB.env["RELAY_API_KEY"],
         RELAY + "/models"],
        capture_output=True, check=True).stdout
    return sorted(m["id"] for m in json.loads(out).get("data", []))


async def main() -> int:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    models = catalog()
    print("=" * 92)
    print("RELAY CATALOG CANARY  (%d models, max_tokens=%d)" % (len(models), cap))
    print("a model is REAL only if it returns the exact product 178451")
    print("=" * 92)

    real, by_verdict = [], {}
    for m in models:
        try:
            verdict, detail, meta = await PB.canary(m)
        except BaseException as e:                              # noqa: BLE001
            verdict, detail, meta = "PROBE-CRASH", str(e)[:70], {}
        rtok = meta.get("reasoning_tokens")
        print("  %-30s %-17s %-42s rtok=%s"
              % (m, verdict, str(detail)[:42], rtok), flush=True)
        if verdict == "REAL":
            real.append(m)
        else:
            by_verdict.setdefault(verdict, []).append(m)

    print()
    print("-" * 92)
    print("REAL  (%d): %s" % (len(real), ", ".join(real) or "(none)"))
    for v, ms in sorted(by_verdict.items()):
        print("%-14s(%d): %s" % (v, len(ms), ", ".join(ms)))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
