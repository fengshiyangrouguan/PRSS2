"""SRI formal runner (§12): preflight -> shared root -> fork -> arms -> freeze
-> transition audit -> final held-out evaluation -> aggregation.

Every stage is restartable from an immutable manifest, and a later stage REFUSES
to run when an earlier manifest or artifact hash changed. The stage manifests are
what make "the persisted records uniquely determine every paper metric" true
rather than aspirational.

WHAT THE GATES ARE FOR. Three things can make a formal run unusable while every
number in it still looks fine:

  * an INCOMPLETE ledger -- a slot that never finalized shrinks the regression
    denominator instead of failing;
  * ARMS THAT DID NOT ACTUALLY DIFFER ONLY IN THE TREATMENT -- a shared knob
    drifting between arms turns the contrast into noise;
  * ARTIFACTS THAT MOVED AFTER THE FREEZE -- a post-freeze edit re-introduces
    exactly the search-time feedback the audit exists to exclude.

Each is checked at the stage that can first see it, and each fails closed.

MOCK MODE. `--evaluator mock` replaces the real CO-Bench evaluator with a
deterministic offline score derived from the script's hash, so the whole
pipeline (including a deliberately rejected edge) can be exercised end to end
without spending relay quota. Every artifact a mock run writes is stamped
`evaluator_mode: "mock"`, and the aggregator refuses to pool modes -- a mock
number can never be mistaken for a real one.

Usage:
    python scripts/run_sri_formal.py --profile sri_primary6 --backbone gpt-5.5 \
        --search-seed 0 --out runs/sri/primary6/s0 [--stage all|preflight|...]
        [--execute]          # without it the runner only PLANS (no API)
        [--gamma-checkpoint runs/_gamma_multitask_phaseb.pt]
        [--evaluator real|mock]

`--execute` is required to run the paid search stages; the default is a plan, so
an accidental invocation cannot spend relay quota. The post-freeze stages (audit
/ final) are LOCAL evaluation and need no LLM, but they still require `--execute`
so that no benchmark work happens without an explicit go.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("meta_n.sri.runner")
from meta_n.sri.protocol import (  # noqa: E402
    OPERATIONAL_ALLOWLIST, ProtocolError, RunManifest, SRIProfile,
    assert_no_protocol_drift, assert_treatment_parity, collect_root_bundle,
    default_profile_path, format_effective_config_table, load_profile,
    pinned_run_config, render_pinned_flags, resolve_effective_config,
    root_bundle_sha256, root_candidate_digest, sha256_file, sha256_of,
    verify_run_config)
from meta_n.sri.pairing import (  # noqa: E402
    assert_formal_pairing, build_record)
from meta_n.sri.ledger import SlotLedger, assert_completeness  # noqa: E402
from meta_n.sri.metrics import (  # noqa: E402
    SeedRun, aggregate, per_run_metrics, select_deployable)
from meta_n.sri.transition_audit import (  # noqa: E402
    CandidateMaterial, RawEval, assert_placeable, build_edges,
    finalize_exact_depth, load_material_from_dir, run_canonical_audit)

STAGES = ("preflight", "root", "fork", "official", "predictive", "freeze",
          "audit", "final", "aggregate")
ARMS = ("official", "predictive")

STAGE_MANIFEST = "sri_stage_manifest.json"
LEDGER_NAME = "proposal_slots.jsonl"
REJECTED_DIRNAME = "rejected"
DEFAULT_DATA_DIR = "./data/co_bench"


def run_id_for(profile: SRIProfile, backbone: str, search_seed: int) -> str:
    """The run identity shared by the context file and the ledger reader.

    One definition, two callers: if the reader computed it any other way the
    ledger's header check would fire on a correct run.
    """
    return "{}:{}:{}".format(profile.name, backbone, int(search_seed))


class StageError(RuntimeError):
    """A stage refused to run: its inputs changed or a prior stage is missing."""


# --------------------------------------------------------------------------
# stage manifest
# --------------------------------------------------------------------------

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
           "planned": bool(outputs.get("planned", False)),
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        rec.update(extra)
    prev = man.get(stage)
    if prev is not None:
        if prev.get("planned") and not rec["planned"]:
            pass                       # a plan is replaced by the real thing
        elif not prev.get("planned") and rec["planned"]:
            raise StageError(
                "stage {!r} already ran for real; refusing to overwrite it with "
                "a plan".format(stage))
        elif prev["inputs_sha256"] != rec["inputs_sha256"]:
            raise StageError(
                "stage {!r} was already run with DIFFERENT inputs ({} -> {}); "
                "refusing to overwrite a frozen stage".format(
                    stage, prev["inputs_sha256"], rec["inputs_sha256"]))
    man[stage] = rec
    out.mkdir(parents=True, exist_ok=True)
    _stage_path(out).write_text(json.dumps(man, indent=2, sort_keys=True),
                               encoding="utf-8")
    return rec


def require_stage(out: Path, stage: str, *, inputs: dict,
                  allow_planned: bool = False) -> dict:
    man = read_stage_manifest(out)
    if stage not in man:
        raise StageError("stage {!r} has not run; cannot proceed".format(stage))
    rec = man[stage]
    if rec.get("planned") and not allow_planned:
        raise StageError(
            "stage {!r} was only PLANNED (no --execute); refusing to build on a "
            "run that never happened".format(stage))
    if rec["inputs_sha256"] != sha256_of(inputs):
        raise StageError(
            "stage {!r} inputs changed since it ran; refusing to proceed".format(
                stage))
    return rec


# --------------------------------------------------------------------------
# paths and digests
# --------------------------------------------------------------------------

def arm_dir(out: Path, arm: str) -> Path:
    return out / "arms" / arm


def root_exp_dir(out: Path) -> Path:
    return out / "root" / "phaseH_root"


def context_dir(out: Path) -> Path:
    return out / "context"


def context_path(out: Path, arm: str) -> Path:
    return context_dir(out) / (arm + ".json")


def _digest_of_tree(root: Path) -> Dict[str, str]:
    """sha256 of every file under `root`, keyed by relative posix path."""
    out: Dict[str, str] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_dir() or "__pycache__" in p.parts:
            continue
        out[p.relative_to(root).as_posix()] = sha256_file(p)
    return out


def artifacts_sha256(arm_path: Path) -> Dict[str, Any]:
    """The freeze gate's subject: config.json + archive/ + summary.json + ledger.

    Content only (no paths, no mtimes), so two identical copies agree.
    """
    parts: Dict[str, Any] = {}
    for name in ("config.json", "summary.json", "lineage.json",
                 "convergence.json", "checkpoint.json", LEDGER_NAME):
        p = arm_path / name
        parts[name] = sha256_file(p) if p.is_file() else None
    parts["archive"] = _digest_of_tree(arm_path / "archive")
    return parts


def artifacts_digest(arm_path: Path) -> str:
    return sha256_of(artifacts_sha256(arm_path))


def read_json(p: Path) -> Optional[dict]:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise StageError("{} is not valid JSON: {}".format(p, e))


def read_run_config(arm_path: Path) -> dict:
    cfg = read_json(arm_path / "config.json")
    if cfg is None:
        raise StageError(
            "no config.json under {}; the arm did not run (or ran without "
            "--use-archive)".format(arm_path))
    return cfg


# --------------------------------------------------------------------------
# arm command assembly (§8 / audit #3)
# --------------------------------------------------------------------------

def pinned_for(args, profile: SRIProfile) -> Dict[str, Any]:
    """The profile's pinned parameters, plus the run-level search seed."""
    return pinned_run_config(profile, backbone=args.backbone,
                             search_seed=args.search_seed)


def build_root_cmd(args, profile: SRIProfile, out: Path) -> List[str]:
    """The shared root: the profile's config with iterations pinned to 0.

    Every OTHER parameter comes from the profile, exactly as for the arms. The
    first revision hand-wrote a handful of them, so the root was generated under
    a different configuration than the arms inherited -- max_depth 10 vs the
    profile's 6 -- and nothing recorded the difference.
    """
    pinned = dict(pinned_for(args, profile))
    pinned["max_iterations"] = 0            # ROOT ONLY: generate, do not breed
    return ([sys.executable, "-m", "meta_n.main"] + render_pinned_flags(pinned)
            + ["--no-test-eval",
               "--output-dir", str(out / "root"),
               "--exp-name", "phaseH_root"])


def build_arm_cmd(args, profile: SRIProfile, out: Path, arm: str,
                  context_file: Path) -> List[str]:
    """One arm's command: the profile's pinned parameters, plus the SRI context.

    Both arms get the IDENTICAL argument list except for the context file, which
    is where the treatment (reduction mode + Gamma identity) lives.

    `--resume` is NOT a profile parameter -- it is a stage-level requirement, and
    omitting it was a real defect: without it the orchestrator does not restore
    the archive from the copied root, so each arm would regenerate its OWN root
    and the shared-root design would quietly collapse.
    """
    pinned = pinned_for(args, profile)
    return ([sys.executable, "-m", "meta_n.main"]
            + render_pinned_flags(pinned)
            + ["--no-test-eval", "--resume", "--output-dir", str(out / "arms"),
               "--exp-name", arm, "--sri-context", str(context_file)])


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_preflight(args, profile: SRIProfile, out: Path) -> dict:
    """Validate the profile, resolve the EFFECTIVE config, check pairing."""
    if args.evaluator == "real":
        for name in profile.cohort:
            if not (Path(args.data_dir) / name).is_dir():
                raise StageError("cohort task {!r} is not in {}".format(
                    name, args.data_dir))
    else:
        print("NOTE: --evaluator mock reads NO task data, so the cohort-data "
              "check is skipped. Every artifact this run writes is stamped "
              "evaluator_mode=mock.")

    cfg = resolve_effective_config(profile, reduction_mode="official")
    print(format_effective_config_table(cfg))

    pairing = build_record(
        requested=bool(profile.paired_eval),
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

    pinned = pinned_for(args, profile)
    print("PINNED PARAMETERS (§8):")
    for k in sorted(pinned):
        print("  {:<28} {}".format(k, pinned[k]))
    print("  {:<28} {}".format("evaluator_mode", args.evaluator))
    print("  {:<28} {}".format("audit_repeats", profile.audit_repeats))
    print("  {:<28} {}".format("test_repeats", profile.test_repeats))

    outputs = {"effective_config": cfg, "pairing": pairing.manifest_fields(),
               "pinned": pinned, "evaluator_mode": args.evaluator}
    return record_stage(out, "preflight",
                        inputs={"profile_sha256": profile.sha256(),
                                "backbone": args.backbone,
                                "search_seed": args.search_seed,
                                "evaluator": args.evaluator},
                        outputs=outputs)


def stage_root(args, profile: SRIProfile, out: Path) -> dict:
    """Layer 1 generated and evaluated EXACTLY ONCE (§4)."""
    require_stage(out, "preflight",
                  inputs={"profile_sha256": profile.sha256(),
                          "backbone": args.backbone,
                          "search_seed": args.search_seed,
                          "evaluator": args.evaluator},
                  allow_planned=not args.execute)
    cmd = build_root_cmd(args, profile, out)
    exp = root_exp_dir(out)
    log = out / "root.log"
    if not args.execute:
        print("[plan] shared root:\n  " + " ".join(cmd))
        return record_stage(out, "root",
                            inputs={"profile_sha256": profile.sha256(),
                                    "backbone": args.backbone,
                                    "search_seed": args.search_seed},
                            outputs={"planned": True, "cmd": cmd})

    if (exp / "archive" / "index.json").is_file():
        raise StageError(
            "the shared root already exists at {}; the root is generated "
            "EXACTLY ONCE (§4) -- delete it deliberately if you mean to redo "
            "the whole experiment".format(exp))
    out.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise StageError("shared root run failed rc={} (see {})".format(rc, log))

    # the root's REAL config.json must say what the profile pinned
    cfg = read_run_config(exp)
    pinned_root = dict(pinned_for(args, profile))
    pinned_root["max_iterations"] = 0
    verify_run_config(cfg, pinned_root, context="root {}".format(exp))

    bundle = collect_root_bundle(
        archive_dir=exp / "archive", cohort=profile.cohort,
        backbone=args.backbone, search_seed=args.search_seed,
        data_dir=Path(args.data_dir), temperatures=profile.temperatures,
        max_tokens=profile.max_tokens,
        reasoning_effort=profile.reasoning_effort,
        worker_config={"parallel": cfg.get("parallel"),
                       "instance_workers": cfg.get("instance_workers")},
        repo_dir=Path(__file__).resolve().parent.parent)
    rb = root_bundle_sha256(bundle)
    (out / "root_bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    missing = [t for t, h in bundle["task_programs_sha256"].items() if h is None]
    if missing:
        raise StageError(
            "the shared root has no program for cohort task(s) {}; a root "
            "missing cohort material cannot seed a matched comparison".format(
                missing))
    inherited = sha256_of(root_candidate_digest(exp / "archive",
                                                profile.cohort))
    print("root bundle sha256 = {}".format(rb))
    print("inherited root     = {}".format(inherited))
    return record_stage(out, "root",
                        inputs={"profile_sha256": profile.sha256(),
                                "backbone": args.backbone,
                                "search_seed": args.search_seed},
                        outputs={"root_dir": str(exp),
                                 "root_bundle_sha256": rb,
                                 "inherited_root_sha256": inherited,
                                 "environment_sha256": sha256_of(
                                     bundle["environment_fingerprint"]),
                                 "root_config_sha256": sha256_file(
                                     exp / "config.json")},
                        extra={"root_bundle": bundle})


def stage_fork(args, profile: SRIProfile, out: Path, *,
               gamma_checkpoint: str | None) -> dict:
    """Copy the frozen root into both arms and assert they are identical (§4)."""
    root = require_stage(out, "root",
                         inputs={"profile_sha256": profile.sha256(),
                                 "backbone": args.backbone,
                                 "search_seed": args.search_seed},
                         allow_planned=not args.execute)
    src = Path(root["outputs"].get("root_dir") or root_exp_dir(out))
    rb_sha = root["outputs"].get("root_bundle_sha256")
    gsha = (sha256_file(gamma_checkpoint)
            if gamma_checkpoint and Path(gamma_checkpoint).is_file() else None)
    if not args.execute:
        for arm in ARMS:
            print("[plan] copy {} -> {}".format(src, arm_dir(out, arm)))
        if gamma_checkpoint:
            print("[plan] gamma checkpoint sha256 = {}".format(gsha))
        return record_stage(out, "fork",
                            inputs={"root_bundle_sha256": rb_sha},
                            outputs={"planned": True, "gamma_checkpoint_sha256":
                                     gsha})
    if gamma_checkpoint and gsha is None:
        raise StageError(
            "--gamma-checkpoint {!r} does not exist; the predictive arm must "
            "record its Gamma identity (§4)".format(gamma_checkpoint))
    if not src.is_dir():
        raise StageError(
            "the shared root is not at {}; run the root stage first (§4)".format(
                src))
    # The arms can only continue the shared root through `--resume`, and
    # `try_resume` reads `<out_dir>/checkpoint.json`. Forking a root with no
    # checkpoint would leave both arms starting fresh -- so refuse to fork.
    if not (src / "checkpoint.json").is_file():
        raise StageError(
            "the shared root at {} has no checkpoint.json, so the fork could "
            "never be resumed and BOTH arms would regenerate their own root. "
            "The root stage must have written one (it persists right after the "
            "seed is evaluated)".format(src))

    arms_out: Dict[str, str] = {}
    for arm in ARMS:
        dst = arm_dir(out, arm)
        if dst.exists():
            raise StageError("arm dir {} already exists; refusing to overwrite a "
                             "frozen fork".format(dst))
        dst.mkdir(parents=True)
        for child in src.iterdir():
            if child.is_dir():
                subprocess.run(["cp", "-r", str(child), str(dst / child.name)],
                               check=True)
            else:
                subprocess.run(["cp", str(child), str(dst / child.name)],
                               check=True)
        arms_out[arm] = str(dst)
    a, b = Path(arms_out["official"]), Path(arms_out["predictive"])
    same = subprocess.run(["diff", "-r", str(a / "archive"), str(b / "archive")],
                          capture_output=True).returncode == 0
    if not same:
        raise StageError("RULE-5 FAIL: the two arms are not byte-identical "
                         "before the fork")
    return record_stage(out, "fork",
                        inputs={"root_bundle_sha256":
                                root["outputs"]["root_bundle_sha256"]},
                        outputs={"arms": arms_out,
                                 "gamma_checkpoint_sha256": gsha,
                                 "rule5_byte_identical": True})


def build_arm_manifest(args, profile: SRIProfile, out: Path, arm: str, *,
                       gamma_checkpoint: str | None) -> RunManifest:
    """The immutable per-arm manifest; the ONLY fields allowed to differ are the
    treatment ones."""
    root = read_stage_manifest(out)["root"]
    pinned = pinned_for(args, profile)
    shared = {
        "profile_sha256": profile.sha256(),
        "cohort": list(profile.cohort),
        "root_bundle_sha256": root["outputs"]["root_bundle_sha256"],
        # The root candidate's own content digest. `--resume` SHOULD inherit it
        # from the copied fork; this is what proves the resume actually happened
        # rather than the arm quietly regenerating its own root.
        "inherited_root_sha256": root["outputs"].get("inherited_root_sha256"),
        "backbone": args.backbone,
        "search_seed": args.search_seed,
        "evaluator_mode": args.evaluator,
        "pinned": pinned,
        "pairing": read_stage_manifest(out).get(
            "preflight", {}).get("outputs", {}).get("pairing"),
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
                       operational={"output_dir": str(arm_dir(out, arm))})


def write_sri_context(args, profile: SRIProfile, out: Path, arm: str,
                      manifest: RunManifest) -> Path:
    """The handoff file main.py reads to build the ledger (§5)."""
    man = read_stage_manifest(out)
    root_bundle = man["root"].get("root_bundle") or {}
    ctx = {
        "run_id": run_id_for(profile, args.backbone, args.search_seed),
        "arm": arm,
        "backbone": args.backbone,
        "cohort_id": profile.cohort_id,
        "search_seed": int(args.search_seed),
        "cohort": list(profile.cohort),
        "root_bundle_sha256": manifest.root_bundle_sha256,
        "ledger_path": str(arm_dir(out, arm) / LEDGER_NAME),
        "gamma_checkpoint_sha256": manifest.treatment.get(
            "gamma_checkpoint_sha256"),
        "reduction_mode": arm,
        "profile_sha256": profile.sha256(),
        "manifest_sha256": manifest.sha256(),
        "gate_reason": resolve_effective_config(
            profile, reduction_mode=arm)["gate_reason"],
        "evaluator_mode": args.evaluator,
        # §6/§7: the search must never touch held-out test data; the final
        # stage is the only place a test split is read, and it runs afterwards.
        "no_test_eval": True,
        "data_dir": str(args.data_dir),
        "environment_sha256": man.get("root", {}).get("outputs", {}).get(
            "environment_sha256"),
    }
    ctx["sri_context_sha256"] = sha256_of(
        {k: v for k, v in ctx.items() if k not in ("ledger_path",)})
    p = context_path(out, arm)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ctx, indent=2, sort_keys=True), encoding="utf-8")
    return p


def assert_root_inherited(arm_path: Path, want: Optional[str],
                          cohort: Sequence[str], *, label: str = "") -> str:
    """§4: the arm must have INHERITED the shared root, not regenerated one.

    A failed `--resume` silently starts a FRESH run (the orchestrator treats an
    unrecoverable checkpoint as a new run by design). Nothing else in the
    pipeline would notice: the ledger still fills every nominal slot and
    config.json still names the shared root, so the two arms would look clean
    while being generated from different roots -- which is the single thing the
    shared-root design exists to prevent.

    Returns the digest it measured, so the caller can record it.
    """
    got = sha256_of(root_candidate_digest(arm_path / "archive", cohort))
    if want and got != want:
        raise StageError(
            "{} did NOT inherit the shared root: its root candidate hashes to "
            "{} but the frozen root hashes to {}. The resume did not restore the "
            "copied root (look for 'fresh run' in the arm log), so this arm "
            "generated its own root and the two arms are not comparable (§4)"
            .format(label or str(arm_path), got, want))
    return got


def stage_arm(args, profile: SRIProfile, out: Path, arm: str, *,
              gamma_checkpoint: str | None) -> dict:
    root_sha = read_stage_manifest(out)["root"]["outputs"].get(
        "root_bundle_sha256")
    if not args.execute:
        # PLAN: print the exact command and record it, but write NO manifest.
        # A placeholder manifest on disk would make a later real run's
        # `assert_no_protocol_drift` fire against a hash that was never real.
        cmd = build_arm_cmd(args, profile, out, arm, context_path(out, arm))
        print("[plan] arm {}:\n  {}".format(arm, " ".join(cmd)))
        return record_stage(out, arm,
                            inputs={"root_bundle_sha256": root_sha,
                                    "arm": arm, "planned": True},
                            outputs={"planned": True, "cmd": cmd})

    fork = require_stage(out, "fork",
                         inputs={"root_bundle_sha256": root_sha})
    man = build_arm_manifest(args, profile, out, arm,
                             gamma_checkpoint=gamma_checkpoint)
    old = arm_dir(out, arm) / "manifest.json"
    if old.is_file():
        assert_no_protocol_drift(RunManifest.read(old), man)
    man.write(old)
    ctx = write_sri_context(args, profile, out, arm, man)

    cmd = build_arm_cmd(args, profile, out, arm, ctx)
    log = out / ("{}.log".format(arm))
    if fork.get("outputs", {}).get("gamma_checkpoint_sha256") is None and \
            arm == "predictive":
        raise StageError(
            "the predictive arm has no Gamma checkpoint hash; the fork stage "
            "did not receive --gamma-checkpoint (§4)")

    # BEFORE spending: prove the copied fork can actually be resumed. A missing
    # `checkpoint.json` (or an empty rebuilt archive) makes the arm start a
    # FRESH run -- which `assert_root_inherited` would catch, but only after the
    # arm had spent its whole budget on a root nobody asked for.
    ck = arm_dir(out, arm) / "checkpoint.json"
    if not ck.is_file():
        raise StageError(
            "{} has no checkpoint.json, so `--resume` would silently start a "
            "fresh run instead of continuing the copied root. The shared-root "
            "stage must have written one; check that the fork copied it.".format(
                arm_dir(out, arm)))
    idx_path = arm_dir(out, arm) / "archive" / "index.json"
    idx0 = read_json(idx_path)
    if not idx0 or not (idx0.get("candidates") or []):
        raise StageError(
            "{}'s copied archive is empty; `--resume` would start a fresh run"
            .format(arm_dir(out, arm)))
    assert_root_inherited(arm_dir(out, arm),
                          man.shared.get("inherited_root_sha256"),
                          profile.cohort,
                          label="arm {} (pre-flight)".format(arm))

    with log.open("w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise StageError("arm {} failed rc={} (see {})".format(arm, rc, log))

    # the arm's REAL config.json must equal what the profile pinned
    cfg = read_run_config(arm_dir(out, arm))
    verify_run_config(cfg, pinned_for(args, profile), context=arm)
    sri = cfg.get("sri") or {}
    if sri.get("reduction_mode") != arm:
        raise StageError(
            "arm {} ran with reduction_mode={!r}; the treatment was not "
            "applied".format(arm, sri.get("reduction_mode")))
    if arm == "predictive" and not sri.get("gamma_checkpoint_sha256"):
        raise StageError(
            "arm predictive recorded no gamma_checkpoint_sha256 in config.json")

    # §4: the arm must have INHERITED the shared root, not regenerated one.
    assert_root_inherited(arm_dir(out, arm),
                          man.shared.get("inherited_root_sha256"),
                          profile.cohort, label="arm {}".format(arm))
    return record_stage(out, arm,
                        inputs={"manifest_sha256": man.sha256(),
                                "root_bundle_sha256": man.root_bundle_sha256},
                        outputs={"arm_dir": str(arm_dir(out, arm)),
                                 "log": str(log),
                                 "context": str(ctx),
                                 "ledger": str(arm_dir(out, arm) / LEDGER_NAME),
                                 "config_sha256": sha256_file(
                                     arm_dir(out, arm) / "config.json")})


# --------------------------------------------------------------------------
# freeze
# --------------------------------------------------------------------------

def _load_frozen_ledger(profile: SRIProfile, out: Path, arm: str
                        ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Open one arm's ledger and prove it is complete and placeable."""
    root = read_stage_manifest(out)["root"]["outputs"]
    man = RunManifest.read(arm_dir(out, arm) / "manifest.json")
    backbone = str(man.shared.get("backbone") or "?")
    led = SlotLedger(arm_dir(out, arm) / LEDGER_NAME,
                     run_id=run_id_for(profile, backbone,
                                       int(man.shared.get("search_seed"))),
                     arm=arm, backbone=backbone,
                     cohort_id=profile.cohort_id,
                     search_seed=int(man.shared.get("search_seed")),
                     root_bundle_sha256=man.root_bundle_sha256,
                     gamma_checkpoint_sha256=man.treatment.get(
                         "gamma_checkpoint_sha256"))
    if led.header().get("root_bundle_sha256") != root["root_bundle_sha256"]:
        raise StageError(
            "arm {}'s ledger records root {} but the frozen root is {}; the "
            "arms did not share one root (§4)".format(
                arm, led.header().get("root_bundle_sha256"),
                root["root_bundle_sha256"]))
    assert_completeness(led, profile.beam_width, profile.beam_candidates,
                        profile.max_iterations)
    rows = led.rows()
    assert_placeable(rows)
    return led.header(), rows


def stage_freeze(args, profile: SRIProfile, out: Path) -> dict:
    """Both arms ran: prove completeness, parity and immutability, then stop.

    After this stage no arm artifact may change again -- the audit re-checks the
    digests recorded here, which is what makes "post-freeze evaluation" a
    property of the files rather than a promise in a docstring.
    """
    man = read_stage_manifest(out)
    root_sha = man.get("root", {}).get("outputs", {}).get("root_bundle_sha256")
    missing = [arm for arm in ARMS if arm not in man]
    if missing:
        raise StageError(
            "cannot freeze: arm stage(s) {} have not run".format(missing))
    if not args.execute:
        print("[plan] would freeze: {} nominal slot(s) per arm, treatment "
              "parity, the config.json diff outside the treatment block, and "
              "the artifact digests the audit later re-checks".format(
                  profile.nominal_slots))
        return record_stage(out, "freeze",
                            inputs={"root_bundle_sha256": root_sha},
                            outputs={"planned": True,
                                     "nominal_slots": profile.nominal_slots})

    _planned = not args.execute
    off = require_stage(out, "official", allow_planned=_planned,
                        inputs={"manifest_sha256": RunManifest.read(
                            arm_dir(out, "official") / "manifest.json").sha256(),
                            "root_bundle_sha256": read_stage_manifest(
                                out)["root"]["outputs"]["root_bundle_sha256"]})
    pre = require_stage(out, "predictive", allow_planned=_planned,
                        inputs={"manifest_sha256": RunManifest.read(
                            arm_dir(out, "predictive") / "manifest.json").sha256(),
                            "root_bundle_sha256": read_stage_manifest(
                                out)["root"]["outputs"]["root_bundle_sha256"]})
    del off, pre

    # 1. treatment parity: the two arms differ ONLY in the treatment block
    m_off = RunManifest.read(arm_dir(out, "official") / "manifest.json")
    m_pre = RunManifest.read(arm_dir(out, "predictive") / "manifest.json")
    assert_treatment_parity(m_off, m_pre)

    # 2. the REAL config.json files must agree outside the treatment block.
    # `timestamp` (and the other §10 operational keys) legitimately differ: the
    # two arms ran at different moments. Comparing them naively rejected EVERY
    # real run -- the gate has to exclude the allowlist it already defines for
    # exactly this reason.
    c_off = read_run_config(arm_dir(out, "official"))
    c_pre = read_run_config(arm_dir(out, "predictive"))
    s_off = dict(c_off.get("sri") or {})
    s_pre = dict(c_pre.get("sri") or {})

    def comparable(cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in cfg.items()
                if k not in OPERATIONAL_ALLOWLIST and k != "sri"
                and k != "resume_config_drift"}

    a, b = comparable(c_off), comparable(c_pre)
    diff = {k: {"official": a.get(k), "predictive": b.get(k)}
            for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    if diff:
        raise StageError(
            "the two arms' config.json differ OUTSIDE the treatment block: {} "
            "-- the contrast is not clean".format(diff))
    if (s_off.get("reduction_mode"), s_pre.get("reduction_mode")) != \
            ("official", "predictive"):
        raise StageError(
            "unexpected treatment in config.json: official.sri={!r} "
            "predictive.sri={!r}".format(s_off.get("reduction_mode"),
                                         s_pre.get("reduction_mode")))

    # 3. ledger completeness + placeability, per arm
    counts: Dict[str, Dict[str, int]] = {}
    for arm in ARMS:
        led_header, rows = _load_frozen_ledger(profile, out, arm)
        c: Dict[str, int] = {}
        for r in rows:
            k = r.get("terminal_status") or "OPEN"
            c[k] = c.get(k, 0) + 1
        counts[arm] = c
        print("  {:<11} {} slots: {}".format(arm, len(rows),
                                             dict(sorted(c.items()))))

    # 4. the freeze: hash every arm artifact and record it
    digests = {arm: artifacts_sha256(arm_dir(out, arm)) for arm in ARMS}
    digest_sha = {arm: sha256_of(d) for arm, d in digests.items()}
    (out / "frozen").mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        (out / "frozen" / "{}.slots.jsonl".format(arm)).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n"
                    for r in _load_frozen_ledger(profile, out, arm)[1]),
            encoding="utf-8")
        (out / "frozen" / "{}_counts.json".format(arm)).write_text(
            json.dumps(counts[arm], indent=2, sort_keys=True), encoding="utf-8")
    (out / "frozen" / "digests.json").write_text(
        json.dumps({"parts": digests, "sha256": digest_sha}, indent=2,
                   sort_keys=True), encoding="utf-8")

    return record_stage(out, "freeze",
                        inputs={"root_bundle_sha256":
                                m_off.root_bundle_sha256},
                        outputs={"artifacts": digest_sha,
                                 "ledger_counts": counts,
                                 "nominal_slots": profile.nominal_slots,
                                 "treatment_parity": True,
                                 "config_diff_outside_treatment": {}})


def _assert_frozen(out: Path, *, allow_planned: bool = False) -> dict:
    """Refuse to proceed if any frozen arm artifact moved (§6)."""
    rec = read_stage_manifest(out).get("freeze")
    if rec is None:
        raise StageError("stage 'freeze' has not run; the audit is post-freeze "
                         "by construction")
    if rec.get("planned"):
        if allow_planned:
            return rec
        raise StageError(
            "stage 'freeze' was only PLANNED (no --execute); refusing to build "
            "on a run that never happened")
    want = rec["outputs"]["artifacts"]
    now = {arm: artifacts_digest(arm_dir(out, arm)) for arm in ARMS}
    moved = {arm: {"frozen": want[arm], "now": now[arm]}
             for arm in ARMS if now[arm] != want[arm]}
    if moved:
        raise StageError(
            "a frozen artifact CHANGED after the freeze: {} -- the canonical "
            "audit must run on the frozen state".format(moved))
    return rec


# --------------------------------------------------------------------------
# evaluators
# --------------------------------------------------------------------------

class MockEvaluator:
    """Deterministic offline stand-in for the CO-Bench evaluator.

    Scores a script by hashing (task_id, source), so the value is a pure
    function of the material: re-running the same audit reproduces it exactly.
    It exists ONLY to exercise the pipeline without spending relay quota, and
    every artifact it produces is stamped `evaluator_mode: "mock"`.
    """

    mode = "mock"

    def __init__(self, cohort: Sequence[str], **_) -> None:
        self.cohort = list(cohort)
        self.calls = 0

    def _score(self, task_id: str, source: str) -> float:
        h = sha256_of({"t": task_id, "s": source})
        return round(0.30 + 0.65 * (int(h[:6], 16) / float(0xFFFFFF)), 6)

    def evaluate(self, task_id, source, *, seed) -> RawEval:
        self.calls += 1
        s = self._score(task_id, source)
        ok = s >= 0.35
        return RawEval(score=s, success=ok, freshly_executed=True,
                       detail="mock:evaluator_mode=mock")

    def evaluate_test(self, task_id, source, *, seed=0) -> RawEval:
        return self.evaluate(task_id, source, seed=seed)


class COBenchEvaluator:
    """The real canonical evaluator: fresh dev/test runs of a task script (§6).

    Dev evaluation reads only the task's `dev_map` instances, i.e. the same
    canonical development instances the search used and the test split never
    touches.

    REPEATS AND THE CACHE. The profile may declare
    `deterministic_evaluator_caches_repeats`, which §3.3 allows ("declaring it
    there keeps the optimisation visible instead of implicit"). That declaration
    is now VERIFIED rather than trusted, and the cache is only ever consulted
    where the verification was free:

      * **dev**: the first candidate seen on a task executes all of its declared
        repeats fresh. If they agree, the task is marked verified and later
        candidates' repeats are served from the cache -- and every such row says
        `cached_deterministic_repeat` in its detail, so the count is auditable
        instead of invisible. If they disagree, the task is NEVER cached and is
        reported as a determinism violation.
      * **test**: never cached. `stage_final` evaluates ONE candidate, so the
        declared repeats are the entire point, and serving them from a cache
        would report three observations from one execution.

    Before this, a repeat served from the cache was byte-indistinguishable from a
    fresh execution in the raw artifact: same `freshly_executed: True`, same
    detail. The value was right (a deterministic evaluator returns the same
    number), but the artifact claimed something it had not done.
    """

    mode = "real"

    def __init__(self, cohort: Sequence[str], *, data_dir=DEFAULT_DATA_DIR,
                 timeout: int = 10, instance_workers: int = 2,
                 deterministic_cache: bool = False,
                 dev_repeats: int = 1) -> None:
        from meta_n.integrations.co_bench import _TaskEvaluator
        self.cohort = list(cohort)
        self.data_dir = Path(data_dir)
        self._make = lambda t: _TaskEvaluator(
            t, self.data_dir, timeout=int(timeout),
            instance_workers=int(instance_workers))
        self._evals: Dict[str, Any] = {}
        self.deterministic_cache = bool(deterministic_cache)
        self.dev_repeats = max(1, int(dev_repeats))
        self._cache: Dict[Tuple[str, str, str], RawEval] = {}
        # (split, task) -> the scores the FIRST candidate's repeats produced
        self._probe: Dict[Tuple[str, str], List[float]] = {}
        self._probe_src: Dict[Tuple[str, str], str] = {}
        self._verified: set = set()
        self._violated: set = set()
        self.calls = 0
        self.fresh = 0
        self.cache_hits = 0

    def _evaluator(self, task_id: str):
        if task_id not in self._evals:
            self._evals[task_id] = self._make(task_id)
        return self._evals[task_id]

    def _execute(self, task_id: str, source: str, split: str) -> RawEval:
        self.calls += 1
        ev = self._evaluator(task_id)
        try:
            out = ev.evaluate(source) if split == "dev" else ev.evaluate_test(
                source)
        except Exception as e:                                # noqa: BLE001
            return RawEval(score=float("nan"), success=False,
                           freshly_executed=True,
                           detail="{}:{}".format(type(e).__name__, e))
        score_key = "dev_score" if split == "dev" else "test_score"
        s = out.get(score_key)
        if s is None:
            fb = out.get(split + "_feedback") or out.get("error") or ""
            return RawEval(score=float("nan"), success=False,
                           freshly_executed=True, detail=str(fb)[:300])
        self.fresh += 1
        return RawEval(score=float(s), success=bool(float(s) > 0),
                       freshly_executed=True,
                       detail="cobench:{}".format(split))

    def _run(self, task_id: str, source: str, split: str) -> RawEval:
        # Only DEV is cacheable, and only once its repeats have been OBSERVED to
        # agree. The test split is never cached (see the class docstring).
        cacheable = self.deterministic_cache and split == "dev"
        skey = (split, task_id)
        src_sha = sha256_of(source)
        key = (split, task_id, src_sha)
        if cacheable and skey in self._verified and key in self._cache:
            self.cache_hits += 1
            prev = self._cache[key]
            return RawEval(score=prev.score, success=prev.success,
                           freshly_executed=True,
                           detail="{}:cached_deterministic_repeat".format(split))
        ev = self._execute(task_id, source, split)
        if not cacheable:
            return ev
        base = self._probe_src.get(skey)
        if base is None:
            # the first candidate on this task: its repeats are the probe
            self._probe_src[skey] = src_sha
            self._probe[skey] = [ev.score]
        elif src_sha == base:
            self._probe.setdefault(skey, []).append(ev.score)
            if len(self._probe[skey]) >= self.dev_repeats and \
                    skey not in self._verified:
                vals = self._probe[skey]
                if all(v == vals[0] for v in vals):
                    self._verified.add(skey)
                else:
                    self._violated.add(skey)
                    logger.warning(
                        "task %s: repeats DISAGREE (%s); the deterministic-"
                        "repeat cache is disabled for it", task_id, vals)
        self._cache[key] = ev
        return ev

    def determinism_report(self) -> Dict[str, Any]:
        return {
            "declared_deterministic": bool(self.deterministic_cache),
            "verified_tasks": sorted("{}|{}".format(*k) for k in self._verified),
            "violated_tasks": sorted("{}|{}".format(*k) for k in self._violated),
            "cache_hits": int(self.cache_hits),
            "note": "a repeat served from the cache is marked in the raw row's "
                    "detail as cached_deterministic_repeat; the test split is "
                    "never cached",
        }

    def evaluate(self, task_id, source, *, seed) -> RawEval:
        return self._run(task_id, source, "dev")

    def evaluate_test(self, task_id, source, *, seed=0) -> RawEval:
        return self._run(task_id, source, "test")


def make_evaluator(args, profile: SRIProfile, *, split: str = "dev"):
    if args.evaluator == "mock":
        return MockEvaluator(profile.cohort)
    return COBenchEvaluator(
        profile.cohort, data_dir=args.data_dir, timeout=args.timeout,
        instance_workers=args.instance_workers,
        deterministic_cache=(profile.deterministic_evaluator_caches_repeats
                            and split == "dev"),
        dev_repeats=profile.audit_repeats)


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------

def material_lookup(arm_path: Path, cohort: Sequence[str]):
    """Resolve a ledger row's parent/child material (§5).

    Admitted candidates come from the archive; a rejected/failed child comes
    from the out-of-archive `rejected/<slot_id>/` directory the hooks wrote, so
    the audit covers children the archive is not allowed to hold. A candidate
    whose material is missing makes the edge INVALID, which `build_edges`
    records rather than hides.
    """
    archive = arm_path / "archive"
    cache: Dict[Tuple[str, str], Optional[CandidateMaterial]] = {}

    def get(directory: Path, cid: str, depth: Any,
            side: str) -> Optional[CandidateMaterial]:
        key = (str(directory), str(cid))
        if key not in cache:
            cache[key] = load_material_from_dir(directory, str(cid),
                                                int(depth or 1), cohort)
        return cache[key]

    def of(row, side):
        if side == "parent":
            pid = row.get("parent_id")
            if not pid:
                return None
            return get(archive / str(pid), str(pid),
                       row.get("parent_structural_depth"), side)
        cid = row.get("proposed_child_id")
        depth = row.get("proposed_child_depth")
        if row.get("material_path"):
            mp = Path(row["material_path"])
            if not mp.is_absolute():
                mp = (Path.cwd() / mp).resolve()
            m = load_material_from_dir(mp, str(cid or row.get("slot_id")),
                                       int(depth or 1), cohort)
            if m is not None:
                return m
        if cid:
            return get(archive / str(cid), str(cid), depth, side)
        return None

    return of


def stage_audit(args, profile: SRIProfile, out: Path) -> dict:
    """§6: canonical post-freeze depth-2 -> 3 audit, per arm, fresh evaluations."""
    frz = _assert_frozen(out, allow_planned=not args.execute)
    if not args.execute:
        print("[plan] would audit each arm's frozen slots: build the "
              "depth-2->3 edge set from the ledger (gate-rejected children "
              "included), then re-evaluate every parent/child on all {} cohort "
              "task(s) x {} repeat(s) with the '{}' evaluator".format(
                  len(profile.cohort), profile.audit_repeats, args.evaluator))
        return record_stage(out, "audit",
                            inputs={"freeze": frz["inputs_sha256"]},
                            outputs={"planned": True})
    audits: Dict[str, Any] = {}
    for arm in ARMS:
        rows = [json.loads(l) for l in
                (out / "frozen" / "{}.slots.jsonl".format(arm)).read_text(
                    encoding="utf-8").splitlines() if l.strip()]
        edges, drops = build_edges(rows, material_lookup(arm_dir(out, arm),
                                                         profile.cohort))
        evaluator = make_evaluator(args, profile, split="dev")
        res = run_canonical_audit(
            edges, cohort=profile.cohort, evaluator=evaluator,
            search_seed=int(args.search_seed),
            repeats=int(profile.audit_repeats),
            threshold=float(profile.regression_threshold), drops=drops,
            pairing_limitations=list(
                (read_stage_manifest(out).get("preflight", {}).get(
                    "outputs", {}).get("pairing") or {}).get(
                        "pairing_limitations") or []))
        # §2.4: the structural depth series over the FROZEN archive
        idx = read_json(arm_dir(out, arm) / "archive" / "index.json") or {}
        cands = list(idx.get("candidates") or [])
        finalize_exact_depth(res, cands,
                             dev_score_of=lambda c: c.get("mean_score"))
        paths = res.write(out / "audit" / arm)
        res.metrics["evaluator_mode"] = evaluator.mode
        res.metrics["evaluator_calls"] = int(getattr(evaluator, "calls", 0))
        res.metrics["evaluator_fresh_executions"] = int(
            getattr(evaluator, "fresh", 0))
        res.metrics["evaluator_cache_hits"] = int(
            getattr(evaluator, "cache_hits", 0))
        if hasattr(evaluator, "determinism_report"):
            res.metrics["determinism"] = evaluator.determinism_report()
        Path(paths["metrics.json"]).write_text(
            json.dumps(res.metrics, indent=2, sort_keys=True),
            encoding="utf-8")
        Path(paths["exact_depth.json"]).write_text(
            json.dumps(res.exact_depth, indent=2, sort_keys=True),
            encoding="utf-8")
        r23 = res.metrics["transitions"]["R_2to3"]
        fa = res.metrics["failure_accounting"]
        print("  {:<11} edges={} drops={} pairs={} regressions={} R_2to3={} "
              "F={}".format(arm, len(edges), len(drops),
                            r23["valid_edge_task_pairs"], r23["regressions"],
                            r23["R_valid"], fa["F"]))
        audits[arm] = {"planned": False, "edges": len(edges),
                       "drops": len(drops),
                       "R_2to3": r23["R_valid"], "F": fa["F"],
                       "R_worst": fa["R_worst"],
                       "raw_observations": res.metrics["raw_observations"],
                       "deduplicated_executions":
                           res.metrics["deduplicated_executions"],
                       "metrics_sha256": sha256_file(
                           Path(paths["metrics.json"])),
                       "evaluator_mode": evaluator.mode}
    return record_stage(out, "audit",
                        inputs={"freeze": frz["inputs_sha256"]},
                        outputs={"arms": audits})


# --------------------------------------------------------------------------
# final
# --------------------------------------------------------------------------

def _selected_material(arm_path: Path, selected, cohort: Sequence[str]):
    m = load_material_from_dir(arm_path / "archive" / selected.candidate_id,
                               selected.candidate_id,
                               selected.structural_depth, cohort)
    return m


def stage_final(args, profile: SRIProfile, out: Path) -> dict:
    """§7: ONE dev-selected deployable candidate, evaluated on held-out test."""
    frz = _assert_frozen(out, allow_planned=not args.execute)
    if not args.execute:
        print("[plan] would select ONE deployable candidate per arm on DEV "
              "only (synthesized oracles excluded), then evaluate it on the "
              "held-out test split x {} repeat(s)".format(profile.test_repeats))
        return record_stage(out, "final",
                            inputs={"freeze": frz["inputs_sha256"]},
                            outputs={"planned": True})
    fin: Dict[str, Any] = {}
    for arm in ARMS:
        a = arm_dir(out, arm)
        idx = read_json(a / "archive" / "index.json")
        if idx is None:
            raise StageError("arm {} has no archive index".format(arm))
        cands = list(idx.get("candidates") or [])
        for i, c in enumerate(cands):
            c["creation_index"] = i

        def executable(c, _a=a):
            return load_material_from_dir(
                _a / "archive" / str(c.get("candidate_id")),
                str(c.get("candidate_id")), int(c.get("depth") or 1),
                profile.cohort) is not None

        sel = select_deployable(cands, lambda c: c.get("mean_score"),
                               is_executable_of=executable)
        mat = _selected_material(a, sel, profile.cohort)
        if mat is None:
            raise StageError(
                "arm {}: the selected candidate {} has no material on disk".format(
                    arm, sel.candidate_id))

        evaluator = make_evaluator(args, profile, split="test")
        raw: List[Dict[str, Any]] = []
        per_task: Dict[str, Optional[float]] = {}
        for t in profile.cohort:
            src = mat.task_scripts.get(t)
            if not src:
                per_task[t] = None
                continue
            vals = []
            for rep in range(int(profile.test_repeats)):
                seed = int(sha256_of({"s": args.search_seed, "t": t,
                                      "r": rep})[:8], 16) & 0x7FFFFFFF
                ev = evaluator.evaluate_test(t, src, seed=seed)
                if ev.score is None or not math.isfinite(float(ev.score)):
                    raw.append({"task_id": t, "repeat_index": rep, "seed": seed,
                                "score": None, "success": False,
                                "freshly_executed": ev.freshly_executed,
                                "cached_deterministic_repeat": False,
                                "detail": ev.detail})
                    continue
                vals.append(float(ev.score))
                raw.append({"task_id": t, "repeat_index": rep, "seed": seed,
                            "score": float(ev.score),
                            "success": bool(ev.success),
                            "freshly_executed": ev.freshly_executed,
                            # derived from what the evaluator ACTUALLY did, not
                            # from `rep > 0`: the test split is never cached, so
                            # a hardcoded rep>0 flag would claim a cache hit that
                            # never happened.
                            "cached_deterministic_repeat": bool(
                                "cached_deterministic_repeat" in str(ev.detail)),
                            "detail": ev.detail})
            per_task[t] = statistics.median(vals) if vals else None

        scored = [v for v in per_task.values() if v is not None]
        macro = statistics.fmean(scored) if len(scored) == len(profile.cohort) \
            else None
        test_result = {"test_scores": per_task, "test_mean_score": macro,
                       "missing_tasks": [t for t, v in per_task.items()
                                         if v is None]}
        pr = per_run_metrics(
            selected=sel, test_result=test_result,
            audit_metrics=read_json(out / "audit" / arm / "metrics.json") or {},
            ledger_counts=read_stage_manifest(out)["freeze"]["outputs"][
                "ledger_counts"][arm],
            resource_use={"evaluator_calls": int(getattr(evaluator, "calls", 0)),
                          "evaluator_fresh_executions": int(
                              getattr(evaluator, "fresh", 0)),
                          "test_repeats_declared": int(profile.test_repeats)},
            archive_candidates=cands,
            dev_score_of=lambda c: c.get("mean_score"))
        pr["evaluator_mode"] = evaluator.mode
        pr["evaluator_cache_hits"] = int(getattr(evaluator, "cache_hits", 0))
        pr["selected_material_sha256"] = mat.sha256()
        pr["oracle_upper_bound_dev"] = idx.get("best_mean_score")
        Path(out / "final").mkdir(parents=True, exist_ok=True)
        (out / "final" / (arm + ".raw.jsonl")).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in raw),
            encoding="utf-8")
        (out / "final" / (arm + ".json")).write_text(
            json.dumps(pr, indent=2, sort_keys=True), encoding="utf-8")
        print("  {:<11} selected {} (dev {:.5f}) -> test macro {}".format(
            arm, sel.candidate_id, sel.dev_score, macro))
        fin[arm] = {"planned": False, "selected": sel.candidate_id,
                    "dev_score": sel.dev_score,
                    "depth": sel.structural_depth,
                    "test_macro": macro,
                    "test_scores": per_task,
                    "evaluator_mode": evaluator.mode,
                    "final_sha256": sha256_file(out / "final" / (arm + ".json"))}
    return record_stage(out, "final",
                        inputs={"freeze": frz["inputs_sha256"]},
                        outputs={"arms": fin})


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------

def discover_run_dirs(parent: Path) -> List[Path]:
    """Every immediate subdirectory of `parent` that holds a final result."""
    if (parent / "final").is_dir():
        return [parent]
    return sorted(p for p in parent.iterdir()
                  if p.is_dir() and (p / "final").is_dir())


def stage_aggregate(args, profile: SRIProfile, out: Path) -> dict:
    """§11: pool the frozen per-seed records. Consumes artifacts only."""
    if not args.execute:
        print("[plan] would pool every seed directory under {} that holds a "
              "final/ (one per --out of a search seed), pairing each seed's "
              "Official and Predictive arms under the shared root".format(out))
        return record_stage(out, "aggregate", inputs={"out": str(out)},
                            outputs={"planned": True})
    dirs = discover_run_dirs(out)
    if not dirs:
        raise StageError(
            "no seed directories with a final/ under {}; aggregate scans the "
            "directory that CONTAINS the per-seed run dirs".format(out))
    runs: List[SeedRun] = []
    for d in dirs:
        man = read_stage_manifest(d)
        root = man.get("root", {}).get("outputs", {})
        for arm in ARMS:
            fj = read_json(d / "final" / (arm + ".json"))
            if fj is None:
                raise StageError("{} has no final result for arm {}".format(d, arm))
            m = RunManifest.read(arm_dir(d, arm) / "manifest.json")
            runs.append(SeedRun(
                search_seed=int(m.shared["search_seed"]), arm=arm,
                profile_sha256=m.profile_sha256,
                cohort_id=profile.cohort_id,
                root_bundle_sha256=m.root_bundle_sha256,
                backbone=str(m.shared.get("backbone") or ""),
                environment_sha256=str(root.get("environment_sha256") or ""),
                final_macro=fj["final"].get("final_test_macro_score"),
                per_transition=dict(fj.get("R_valid") or {}),
                payload={"evaluator_mode": fj.get("evaluator_mode"),
                         "selected": fj["final"].get("selected_candidate_id")}))
    if not runs:
        raise StageError("no arms produced a final result")
    modes = sorted({(r.payload or {}).get("evaluator_mode") for r in runs})
    if len(modes) > 1:
        raise StageError(
            "refusing to pool runs produced by different evaluators: {} -- a "
            "mock number is not a real one".format(modes))
    agg = aggregate(runs)
    agg["evaluator_mode"] = modes[0]
    agg["seed_dirs"] = [str(d) for d in dirs]
    agg["search_seed_values"] = sorted({r.search_seed for r in runs})
    p = out / "aggregate.json"
    p.write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
    print("aggregate over {} seed dir(s), {} run(s), evaluator={}".format(
        len(dirs), len(runs), modes[0]))
    if agg.get("paired", {}).get("final_delta_mean") is not None:
        print("  paired final delta (predictive - official) = {:+.5f} over {} "
              "pair(s)".format(agg["paired"]["final_delta_mean"],
                               agg["paired"]["n_pairs"]))
    return record_stage(out, "aggregate",
                        inputs={"seed_dirs": [str(d) for d in dirs]},
                        outputs={"aggregate": str(p),
                                 "n_runs": len(runs),
                                 "evaluator_mode": modes[0],
                                 "paired_final_delta_mean":
                                     agg.get("paired", {}).get(
                                         "final_delta_mean")})


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

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
    ap.add_argument("--allow-inert-pairing", action="store_true",
                    help="permit a formal run whose pairing is inert "
                         "(recorded as a limitation, never hidden)")
    ap.add_argument("--evaluator", default="real", choices=("real", "mock"),
                    help="'mock' is a deterministic offline stand-in used to "
                         "exercise the pipeline without spending relay quota; "
                         "its artifacts are stamped and never pooled with real "
                         "ones")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--instance-workers", type=int, default=2)
    args = ap.parse_args()

    profile = load_profile(default_profile_path(args.profile))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("SRI FORMAL RUNNER  profile={} backbone={} seed={} out={}".format(
        profile.name, args.backbone, args.search_seed, out))
    print("mode: {}   evaluator: {}".format(
        "EXECUTE" if args.execute else "PLAN (no API)", args.evaluator))
    print()

    todo = STAGES if args.stage == "all" else (args.stage,)
    try:
        for st in todo:
            print("=== stage {} ===".format(st))
            if st == "preflight":
                stage_preflight(args, profile, out)
            elif st == "root":
                stage_root(args, profile, out)
            elif st == "fork":
                stage_fork(args, profile, out,
                           gamma_checkpoint=args.gamma_checkpoint)
            elif st in ARMS:
                stage_arm(args, profile, out, st,
                          gamma_checkpoint=args.gamma_checkpoint)
            elif st == "freeze":
                stage_freeze(args, profile, out)
            elif st == "audit":
                stage_audit(args, profile, out)
            elif st == "final":
                stage_final(args, profile, out)
            elif st == "aggregate":
                stage_aggregate(args, profile, out)
            print()
    except (StageError, ProtocolError) as e:
        print("\nREFUSED: {}".format(e), file=sys.stderr)
        return 2
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
