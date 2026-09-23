"""Rebuild the `official` stage record that the rerun setup deleted (0 API).

WHY THIS IS NEEDED. `_setup_ours_rerun.sh` pruned the stage manifest to
preflight/root/fork so that `predictive` would be recomputed. That also removed
the `official` record -- and `stage_freeze` refuses to run when an arm stage is
absent:

    missing = [arm for arm in ARMS if arm not in man]  ->  StageError

so the rerun would spend the predictive arm and then die at freeze. The official
ARCHIVE on disk is still valid (the allocator never touched the official path),
so the record is REBUILT from those artifacts rather than by re-running the arm.

NOTHING IS TRUSTED. Before writing the record this VALIDATES the artifacts it is
derived from:
  * the arm manifest's root_bundle_sha256 equals the root stage's;
  * the arm's root candidate digest equals the frozen inherited_root_sha256
    (i.e. the arm really did inherit the shared root, not regenerate one);
  * config.json's sri.reduction_mode is the mode ARM_REDUCTION_MODE names;
  * the ledger exists and every slot is terminal.
If any check fails it refuses to write anything.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

OUT = Path("/root/autodl-tmp/sri_r3_seed0")
REPO = Path("/root/autodl-tmp/rpbe-sri/meta-n-main")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "sri_runner", REPO / "scripts" / "run_sri_formal.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

from meta_n.sri.protocol import (RunManifest, default_profile_path,  # noqa: E402
                                 load_profile, sha256_file)

PROFILE = load_profile(default_profile_path("sri_primary6"))
SLUGS, _ = R.cohort_ids(PROFILE)
ARM = "official"
a = R.arm_dir(OUT, ARM)

print("=" * 84)
print("VALIDATE the official arm's artifacts before rebuilding its stage record")
print("=" * 84)
man = RunManifest.read(a / "manifest.json")
root = R.read_stage_manifest(OUT)["root"]["outputs"]
problems = []

if man.root_bundle_sha256 != root["root_bundle_sha256"]:
    problems.append("manifest.root_bundle_sha256 %r != root stage %r"
                    % (man.root_bundle_sha256, root["root_bundle_sha256"]))

got = R.sha256_of(R.root_candidate_digest(a / "archive", SLUGS))
want = root.get("inherited_root_sha256")
if want and got != want:
    problems.append("root candidate digest %s != inherited_root_sha256 %s"
                    % (got, want))

cfg = json.load(open(a / "config.json", encoding="utf-8"))
mode = (cfg.get("sri") or {}).get("reduction_mode")
if mode != R.ARM_REDUCTION_MODE[ARM]:
    problems.append("config.json sri.reduction_mode %r != %r"
                    % (mode, R.ARM_REDUCTION_MODE[ARM]))

ledger = a / R.LEDGER_NAME
rows = []
admits = 0
if ledger.is_file():
    rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    # The ledger is WRITE-AHEAD: one `kind=open` row when a slot starts and one
    # `kind=finalize` row when it ends, joined by slot_id. Counting rows and
    # calling the opens "unfinalized" is wrong -- 24 slots are 48 rows.
    opens = {r["slot_id"] for r in rows if r.get("kind") == "open"}
    fins = {r["slot_id"] for r in rows if r.get("kind") == "finalize"}
    missing = sorted(opens - fins)
    if missing:
        problems.append("%d slot(s) opened but never finalized: %s"
                        % (len(missing), missing[:4]))
    admits = sum(1 for r in rows
                 if r.get("terminal_status") == "evaluated_admitted")
    if not opens:
        problems.append("the ledger records no `open` rows at all")
else:
    problems.append("no %s" % R.LEDGER_NAME)

print("  arm manifest root_bundle_sha256 : %s" % man.root_bundle_sha256)
print("  root stage  root_bundle_sha256 : %s" % root["root_bundle_sha256"])
print("  root candidate digest          : %s" % got)
print("  frozen inherited_root_sha256   : %s" % want)
print("  config.json reduction_mode     : %s" % mode)
print("  ledger rows                    : %d  (open+finalize pairs)" % len(rows))
print("  slots evaluated_admitted       : %d" % admits)
print()
if problems:
    print("  REFUSING to rebuild:")
    for p in problems:
        print("    - %s" % p)
    raise SystemExit(1)
print("  all checks passed")

print()
print("=" * 84)
print("REBUILD the `official` stage record (same shape stage_arm writes)")
print("=" * 84)
before = R.read_stage_manifest(OUT)
print("  stages before: %s" % sorted(before))
R.record_stage(
    OUT, ARM,
    inputs={"manifest_sha256": man.sha256(),
            "root_bundle_sha256": man.root_bundle_sha256},
    outputs={"arm_dir": str(a),
             "log": str(OUT / ("%s.log" % ARM)),
             "context": str(R.context_path(OUT, ARM)),
             "ledger": str(ledger),
             "config_sha256": sha256_file(a / "config.json")})
after = R.read_stage_manifest(OUT)
print("  stages after : %s" % sorted(after))
print("  official inputs_sha256 = %s" % after[ARM]["inputs_sha256"])
print()
print("  NOTE: `outputs` is rebuilt from the artifacts on disk; the record is a")
print("  faithful re-derivation, NOT a byte-copy of the pre-prune entry (the")
print("  original is gone). Everything downstream reads the ARTIFACTS, and those")
print("  are the untouched ones from the original run.")
