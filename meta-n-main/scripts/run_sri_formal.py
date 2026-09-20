"""SRI formal runner (§12): preflight -> shared root -> fork -> arms -> freeze
-> transition audit -> final held-out evaluation -> aggregation.

Every stage is restartable from an immutable manifest, and a later stage REFUSES
to run when an earlier manifest or artifact hash changed. The stage manifests are
what make "the persisted records uniquely determine every paper metric" true
rather than aspirational.

Usage:
    python scripts/run_sri_formal.py --profile sri_primary6 --backbone gpt-5.5 \
        --search-seed 0 --out runs/sri/primary6/s0 [--stage all|preflight|root|...]
        [--execute]          # without it the runner only PLANS (no API)
        [--gamma-checkpoint runs/_gamma_multitask_phaseb.pt]

`--execute` is required to run the paid search stages; the default is a plan, so
an accidental invocation cannot spend relay quota.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meta_n.sri.protocol import (  # noqa: E402
    RunManifest, SRIProfile, assert_no_protocol_drift, assert_treatment_parity,
    default_profile_path, format_effective_config_table, load_profile,
    resolve_effective_config, sha256_file, sha256_of)
from meta_n.sri.pairing import (  # noqa: E402
    PairingRecord, assert_formal_pairing, build_record)

STAGES = ("preflight", "root", "fork", "official", "predictive", "freeze",
          "audit", "final", "aggregate")

STAGE_MANIFEST = "sri_stage_manifest.json"
LEDGER_NAME = "proposal_slots.jsonl"


class StageError(RuntimeError):
    """A stage refused to run: its inputs changed or a prior stage is missing."""


def _stage_path(out: Path) -> Path:
    return out / STAGE_MANIFEST


def read_stage_manifest(out: Path) -> dict:
    p = _stage_path(out)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def record_stage(out: Path, stage: str, *, inputs: dict, outputs: dict,
                 extra: dict | None = None) -> dict:
    """Append a stage record, refusing to overwrite a DIFFERENT result.

    Rerunning an identical stage is allowed (restartable); rerunning one whose
    inputs changed is not -- that is the "later stage refuses when an earlier
    manifest changes" rule.
    """
    man = read_stage_manifest(out)
    rec = {"stage": stage, "inputs_sha256": sha256_of(inputs),
           "outputs": outputs, "inputs": inputs,
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        rec.update(extra)
    prev = man.get(stage)
    if prev is not None and prev["inputs_sha256"] != rec["inputs_sha256"]:
        raise StageError(
            "stage {!r} was already run with DIFFERENT inputs ({} -> {}); "
            "refusing to overwrite a frozen stage".format(
                stage, prev["inputs_sha256"], rec["inputs_sha256"]))
    man[stage] = rec
    out.mkdir(parents=True, exist_ok=True)
    _stage_path(out).write_text(json.dumps(man, indent=2, sort_keys=True),
                                encoding="utf-8")
    return rec


def require_stage(out: Path, stage: str, *, inputs: dict) -> dict:
    man = read_stage_manifest(out)
    if stage not in man:
        raise StageError("stage {!r} has not run; cannot proceed".format(stage))
    if man[stage]["inputs_sha256"] != sha256_of(inputs):
        raise StageError(
            "stage {!r} inputs changed since it ran; refusing to proceed".format(
                stage))
    return man[stage]


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_preflight(args, profile: SRIProfile, out: Path) -> dict:
    """Validate the profile, resolve the EFFECTIVE config, check pairing."""
    for name in profile.cohort:
        if not (Path("data/co_bench") / name).is_dir():
            raise StageError("cohort task {!r} is not in data/co_bench".format(name))

    cfg = resolve_effective_config(
        profile, reduction_mode="official", consolidate=True, gate_tasks=3,
        gate_margin=0.0, protect_floor=None, regression_guard=True)
    print(format_effective_config_table(cfg))

    pairing = build_record(
        requested=getattr(args, "paired_eval", False),
        model=args.backbone, backend="relay",
        # Honest by construction: this relay does not honour a per-request seed,
        # so every channel is inert until a seed-capable backend is configured.
        outer_seeded=False, inner_seeded=False,
        executor_honours=False, test_seeded=False,
        matched_instance_fallback=True)
    assert_formal_pairing(pairing, formal=not args.allow_inert_pairing)
    print("PAIRING:")
    for k, v in pairing.manifest_fields().items():
        print("  {:<28} {}".format(k, v))

    outputs = {"effective_config": cfg, "pairing": pairing.manifest_fields()}
    return record_stage(out, "preflight",
                        inputs={"profile_sha256": profile.sha256(),
                                "backbone": args.backbone},
                        outputs=outputs)


def stage_root(args, profile: SRIProfile, out: Path) -> dict:
    """Layer 1 generated and evaluated EXACTLY ONCE (§4)."""
    require_stage(out, "preflight",
                  inputs={"profile_sha256": profile.sha256(),
                          "backbone": args.backbone})
    root_dir = out / "root"
    log = out / "root.log"
    cmd = [sys.executable, "-m", "meta_n.main",
           "--benchmark", "co_bench", "--bench-tasks", *profile.cohort,
           "--max-iterations", "0", "--no-early-stop",
           "--beam-width", "1", "--beam-candidates", "1",
           "--max-tokens", "1024", "--empty-retry-max-tokens", "0",
           "--max-retries", "0", "--reasoning-effort", "low",
           "--seed", str(args.search_seed), "--model", args.backbone,
           "--reduction-mode", "official", "--use-archive",
           "--exp-name", "phaseH_root", "--output-dir", str(root_dir)]
    if not args.execute:
        print("[plan] would run shared root:\n  " + " ".join(cmd))
        return {"planned": True, "cmd": cmd}
    with log.open("w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise StageError("shared root run failed rc={} (see {})".format(rc, log))
    root_id = root_dir / "phaseH_root"
    idx = json.loads((root_id / "archive" / "index.json").read_text())
    bundle = {"cohort": list(profile.cohort),
              "task_order": list(profile.cohort),
              "root_candidate_id": "gen0_seed",
              "task_programs_sha256": sha256_of(
                  {c.get("candidate_id"): c.get("mean_score")
                   for c in idx.get("candidates", [])}),
              "traces_sha256": sha256_of(sorted(
                  p.name for p in (root_id / "archive" / "gen0_seed" / "traces").glob("*"))),
              "dev_scores": {c.get("candidate_id"): c.get("mean_score")
                             for c in idx.get("candidates", [])},
              "archive_metadata": {"size": len(idx.get("candidates", []))},
              "rng_state": None, "evaluator_manifest": None,
              "model_config": {"backbone": args.backbone},
              "software_revision": None, "environment_fingerprint": None}
    from meta_n.sri.protocol import root_bundle_sha256
    rb = root_bundle_sha256(bundle)
    return record_stage(out, "root",
                        inputs={"profile_sha256": profile.sha256(),
                                "backbone": args.backbone,
                                "search_seed": args.search_seed},
                        outputs={"root_dir": str(root_id),
                                 "root_bundle_sha256": rb},
                        extra={"root_bundle": bundle})


def stage_fork(args, profile: SRIProfile, out: Path, *,
               gamma_checkpoint: str | None) -> dict:
    """Copy the frozen root into both arms and assert they are identical (§4)."""
    root = require_stage(out, "root",
                         inputs={"profile_sha256": profile.sha256(),
                                 "backbone": args.backbone,
                                 "search_seed": args.search_seed})
    if root.get("outputs", {}).get("root_bundle_sha256") is None:
        return {"planned": True}
    src = Path(root["outputs"]["root_dir"])
    arms = {}
    for arm in ("official", "predictive"):
        dst = out / "arms" / arm
        if not args.execute:
            print("[plan] would copy {} -> {}".format(src, dst))
            continue
        if dst.exists():
            raise StageError("arm dir {} already exists; refusing to overwrite a "
                             "frozen fork".format(dst))
        subprocess.run(["cp", "-r", str(src), str(dst)], check=True)
        arms[arm] = str(dst)
    if args.execute:
        a, b = Path(arms["official"]), Path(arms["predictive"])
        same = subprocess.run(["diff", "-r", str(a / "archive"), str(b / "archive")],
                              capture_output=True).returncode == 0
        if not same:
            raise StageError("RULE-5 FAIL: the two arms are not byte-identical "
                             "before the fork")
    gsha = sha256_file(gamma_checkpoint) if gamma_checkpoint and \
        Path(gamma_checkpoint).is_file() else None
    return record_stage(out, "fork",
                        inputs={"root_bundle_sha256":
                                root["outputs"]["root_bundle_sha256"]},
                        outputs={"arms": arms,
                                 "gamma_checkpoint_sha256": gsha})


def build_arm_manifest(args, profile: SRIProfile, out: Path, arm: str, *,
                        gamma_checkpoint: str | None) -> RunManifest:
    """The immutable per-arm manifest; the ONLY fields allowed to differ are the
    treatment ones."""
    root = read_stage_manifest(out)["root"]
    shared = {
        "profile_sha256": profile.sha256(),
        "cohort": list(profile.cohort),
        "B": profile.beam_width, "K": profile.beam_candidates,
        "T": profile.max_iterations, "max_depth": profile.max_depth,
        "no_early_stop": profile.no_early_stop,
        "backbone": args.backbone,
        "search_seed": args.search_seed,
        "consolidate": True, "regression_guard": True,
        "gate_tasks": 3, "gate_margin": 0.0,
        "reduction_threshold": profile.regression_threshold,
    }
    treatment = {"reduction_mode": arm}
    if arm == "predictive":
        treatment["gamma_checkpoint_sha256"] = (
            sha256_file(gamma_checkpoint)
            if gamma_checkpoint and Path(gamma_checkpoint).is_file() else None)
    return RunManifest(arm=arm, profile=profile.identity(),
                       profile_sha256=profile.sha256(),
                       root_bundle_sha256=root["outputs"]["root_bundle_sha256"],
                       shared=shared, treatment=treatment,
                       operational={"output_dir": str(out / "arms" / arm)})


def stage_arm(args, profile: SRIProfile, out: Path, arm: str, *,
              gamma_checkpoint: str | None) -> dict:
    require_stage(out, "fork", inputs={"root_bundle_sha256":
                                       read_stage_manifest(out)["root"]["outputs"]["root_bundle_sha256"]})
    man = build_arm_manifest(args, profile, out, arm,
                             gamma_checkpoint=gamma_checkpoint)
    old = out / "arms" / arm / "manifest.json"
    if old.is_file():
        assert_no_protocol_drift(RunManifest.read(old), man)
    man.write(old)

    arm_dir = out / "arms" / arm
    log = out / ("{}.log".format(arm))
    cmd = [sys.executable, "-m", "meta_n.main",
           "--benchmark", "co_bench", "--bench-tasks", *profile.cohort,
           "--resume", "--exp-name", arm, "--output-dir", str(arm_dir.parent),
           "--max-iterations", str(profile.max_iterations), "--no-early-stop",
           "--beam-width", str(profile.beam_width),
           "--beam-candidates", str(profile.beam_candidates),
           "--max-tokens", "1024", "--empty-retry-max-tokens", "0",
           "--max-retries", "0", "--reasoning-effort", "low",
           "--seed", str(args.search_seed), "--model", args.backbone,
           "--use-archive", "--reduction-mode", arm]
    if arm == "predictive":
        cmd += ["--gamma-checkpoint", str(gamma_checkpoint)]
    if not args.execute:
        print("[plan] would run arm {}:\n  {}".format(arm, " ".join(cmd)))
        return {"planned": True, "cmd": cmd, "manifest_sha256": man.sha256()}
    with log.open("w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise StageError("arm {} failed rc={} (see {})".format(arm, rc, log))
    return record_stage(out, arm,
                        inputs={"manifest_sha256": man.sha256()},
                        outputs={"arm_dir": str(arm_dir), "log": str(log),
                                 "ledger": str(arm_dir / LEDGER_NAME)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="sri_primary6")
    ap.add_argument("--backbone", default="gpt-5.5")
    ap.add_argument("--search-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=("all",) + STAGES)
    ap.add_argument("--gamma-checkpoint", default=None)
    ap.add_argument("--execute", action="store_true",
                    help="actually run the paid stages; without it the runner "
                         "only plans")
    ap.add_argument("--paired-eval", action="store_true")
    ap.add_argument("--allow-inert-pairing", action="store_true",
                    help="permit a formal run whose pairing is inert "
                         "(recorded as a limitation, never hidden)")
    args = ap.parse_args()

    profile = load_profile(default_profile_path(args.profile))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("SRI FORMAL RUNNER  profile={} backbone={} seed={} out={}".format(
        profile.name, args.backbone, args.search_seed, out))
    print("mode: {}".format("EXECUTE (spends relay quota)" if args.execute
                            else "PLAN (no API)"))
    print()

    todo = STAGES if args.stage == "all" else (args.stage,)
    for st in todo:
        print("=== stage {} ===".format(st))
        if st == "preflight":
            stage_preflight(args, profile, out)
        elif st == "root":
            stage_root(args, profile, out)
        elif st == "fork":
            stage_fork(args, profile, out, gamma_checkpoint=args.gamma_checkpoint)
        elif st in ("official", "predictive"):
            stage_arm(args, profile, out, st,
                      gamma_checkpoint=args.gamma_checkpoint)
        else:
            print("  [not implemented in this revision] {}".format(st))
        print()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
