"""Read the provenance `run_phase_b.py` STAMPS INTO each Gamma checkpoint.

`scripts/run_phase_b.py` calls
    tb.save_gamma(args.out, extra={"records": args.records,
                                   "split_protocol": SPLIT_PROTOCOL, ...})
so every checkpoint names the records file that trained it. That is the one
piece of provenance in this pipeline that is not reconstructed from mtimes.
"""
import glob
import os

import torch

PATHS = sorted(
    set(glob.glob("/root/autodl-tmp/rpbe-sri/meta-n-main/runs/*.pt") +
        glob.glob("/root/autodl-tmp/meta-n-main/runs/*.pt")))

print("=" * 78)
print("Gamma checkpoint provenance (from the file itself)")
print("=" * 78)
for p in PATHS:
    print()
    print("  %s" % p)
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:                                    # noqa: BLE001
        print("    UNREADABLE: %s" % e)
        continue
    if not isinstance(d, dict):
        print("    unexpected payload type %s" % type(d).__name__)
        continue
    meta = d.get("meta") or {}
    extra = d.get("extra") or meta.get("extra") or {}
    print("    steps        : %s" % meta.get("steps"))
    print("    status       : %s" % meta.get("status"))
    print("    saved_at     : %s" % meta.get("saved_at"))
    print("    checksum     : %s" % meta.get("checksum"))
    print("    extra        : %s" % (extra if extra else "(none stamped)"))
    if not extra:
        print("    -> trained by a path that did NOT stamp the records file")
