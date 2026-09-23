#!/bin/bash
# Re-run ONLY the predictive/Ours arm from the SAME frozen seed-0 root.
#
# WHY THIS IS LEGITIMATE AND CHEAP
#   * the official reducer was NOT changed (only a log line was added to it), and
#     its proposal/search budget is identical -- so its archive is still valid
#     and does NOT need a paid re-run;
#   * the sibling allocator changed ONLY the predictive path;
#   * the root is the shared generation, so it is reused byte-for-byte, which is
#     also what makes old-Ours vs new-Ours a clean A/B on ONE thing:
#       1 unique subset per parent  ->  4 unique subsets.
#   * the final selector is back to the original dev-max, applied to BOTH arms,
#     so no protocol asymmetry is introduced.
#
# Steps, all reversible: the old predictive arm is MOVED ASIDE, never deleted.
set -uo pipefail

OUT=/root/autodl-tmp/sri_r3_seed0
cd "$OUT" || exit 1
say() { echo "[$(date '+%F %T')] $*"; }

say "=== 0. before: root identity ==="
sha256sum root_bundle.json | awk '{print "  root_bundle.json sha256 = "$1}'
/root/miniconda3/bin/python - <<'PY'
import json
m = json.load(open('/root/autodl-tmp/sri_r3_seed0/sri_stage_manifest.json'))
print("  root_bundle_sha256   =", m['root']['outputs']['root_bundle_sha256'])
print("  inherited_root_sha256=", m['root']['outputs']['inherited_root_sha256'])
print("  fork inputs          =", m['fork']['inputs'])
PY

say "=== 1. move the OLD predictive arm aside (never deleted) ==="
if [ -d arms/predictive ]; then
  rm -rf arms/_predictive_OLD_selector
  mv arms/predictive arms/_predictive_OLD_selector
  say "  arms/predictive -> arms/_predictive_OLD_selector"
fi

say "=== 2. seed arms/predictive with the FROZEN ROOT (exactly what fork does) ==="
mkdir -p arms/predictive
cp -r root/phaseH_root/. arms/predictive/
ls arms/predictive | head -6

say "=== 3. verify the seeded arm is byte-identical to the root ==="
if diff -r root/phaseH_root arms/predictive >/dev/null; then
  say "  OK: arms/predictive == root/phaseH_root (byte-identical)"
else
  say "  ERROR: the seeded arm differs from the root -- aborting"; exit 1
fi

say "=== 4. drop post-fork stage records so they are recomputed ==="
/root/miniconda3/bin/python - <<'PY'
import json
p = '/root/autodl-tmp/sri_r3_seed0/sri_stage_manifest.json'
m = json.load(open(p))
keep = {k: v for k, v in m.items() if k in ('preflight', 'root', 'fork')}
# repoint the fork record at the arm dir that exists now
if 'fork' in keep:
    keep['fork'].setdefault('outputs', {})['arms'] = {
        'official': '/root/autodl-tmp/sri_r3_seed0/arms/official',
        'predictive': '/root/autodl-tmp/sri_r3_seed0/arms/predictive'}
json.dump(keep, open(p, 'w'), indent=1, sort_keys=True)
print("  kept stages:", sorted(keep))
PY

say "=== 5. clear post-fork LOCAL artifacts (freeze/final/audit are recomputed) ==="
rm -rf frozen final audit
rm -rf forensic4_narrow.jsonl dryrun_selection
say "  removed frozen/ final/ audit/"

say "SETUP_OK -- ready to re-run the predictive arm"
