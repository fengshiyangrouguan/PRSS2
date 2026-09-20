"""SRI formal protocol: frozen profiles, run manifests, drift validation.

Design doc: `2026-09-20-sri-formal-protocol-design.md` (repo branch `metan-rpbe`).

This module is deliberately dependency-light (no torch, no network) so the whole
protocol layer can be unit-tested in milliseconds. It owns three things:

  1. **Profiles** (§3, §12) -- the versioned, human-readable search profiles
     (`meta_n/configs/sri_primary6.yaml`, `sri_extended10.yaml`). A profile fixes
     the cohort, B/K/T, depth, the seed families and the repeat families. The
     nominal recursive slot count is DERIVED (`B * K * T`), never a free field.

  2. **Manifests** (§4, §10) -- the immutable per-arm record of every
     result-affecting field. `assert_treatment_parity` fails CLOSED when the two
     arms differ in anything but the treatment fields.

  3. **Resume drift** (§10) -- `assert_no_protocol_drift` compares a saved
     manifest against the live one and aborts on any result-affecting change.
     This is stricter than the historical warn-only drift record, which stays
     available outside formal mode.

The module never invents a number: `nominal_slots` is computed, `macros` are
checked for the frozen cohort, and hashes are taken over canonical JSON so two
processes agree byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

PROFILE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1

# Frozen cohorts (§3.1, §3.2). The extended cohort is the primary six plus four;
# it is reported separately and may never replace the primary cohort.
PRIMARY6: Tuple[str, ...] = (
    "Aircraft landing",
    "Assignment problem",
    "Assortment problem",
    "Bin packing - one-dimensional",
    "Capacitated warehouse location",
    "Common due date scheduling",
)
EXTENDED10: Tuple[str, ...] = PRIMARY6 + (
    "Crew scheduling",
    "Flow shop scheduling",
    "Hybrid Reentrant Shop Scheduling",
    "Job shop scheduling",
)

# Profile name -> the cohort it must carry. A profile whose cohort does not match
# its name is refused, so the two formal cohorts can never be silently mixed.
PROFILE_COHORTS: Dict[str, Tuple[str, ...]] = {
    "sri_primary6": PRIMARY6,
    "sri_extended10": EXTENDED10,
}

# §4: the ONLY fields the two arms may differ in. Anything else differing means
# the comparison is not a clean treatment contrast and formal mode fails closed.
TREATMENT_FIELDS: Tuple[str, ...] = (
    "reduction_mode",
    "gamma_checkpoint_sha256",
    "selected_context_sha256",
    "rendered_prompt_sha256",
)

# §10: operational fields that MAY differ without invalidating the contrast.
OPERATIONAL_ALLOWLIST: Tuple[str, ...] = (
    "output_dir",
    "timestamp",
    "started_at",
    "finished_at",
    "wall_seconds",
    "pid",
    "hostname",
)


class ProtocolError(RuntimeError):
    """A formal-protocol invariant was violated. Always fail closed."""


# --------------------------------------------------------------------------
# canonical hashing
# --------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace drift, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    """Streaming file hash -- used for the Gamma checkpoint identity (§4)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SRIProfile:
    """A frozen formal search profile (§3.3)."""

    name: str
    cohort: Tuple[str, ...]
    beam_width: int
    beam_candidates: int
    max_iterations: int
    max_depth: int
    no_early_stop: bool
    search_seeds: Tuple[int, ...]
    search_eval_repeats: int
    audit_repeats: int
    test_repeats: int
    # Frozen estimator constants (§2.2). Kept on the profile so a run cannot
    # quietly move the bar after seeing a result.
    regression_threshold: float = 0.02
    schema_version: int = PROFILE_SCHEMA_VERSION
    # Declared evaluator optimisations (§3.3: "the manifest must declare that
    # optimization"). A deterministic evaluator may cache identical repeats.
    deterministic_evaluator_caches_repeats: bool = True

    def __post_init__(self) -> None:
        if self.name not in PROFILE_COHORTS:
            raise ProtocolError(
                "unknown profile {!r}; known: {}".format(
                    self.name, sorted(PROFILE_COHORTS)))
        want = PROFILE_COHORTS[self.name]
        if tuple(self.cohort) != want:
            raise ProtocolError(
                "profile {!r} must carry its frozen cohort; expected {} got {}"
                .format(self.name, list(want), list(self.cohort)))
        for k in ("beam_width", "beam_candidates", "max_iterations",
                  "max_depth", "search_eval_repeats", "audit_repeats",
                  "test_repeats"):
            if int(getattr(self, k)) < 1:
                raise ProtocolError("{} must be >= 1".format(k))
        if not self.search_seeds:
            raise ProtocolError("search_seeds must be non-empty")
        if len(set(self.search_seeds)) != len(self.search_seeds):
            raise ProtocolError("search_seeds must be distinct")
        if self.regression_threshold <= 0:
            raise ProtocolError("regression_threshold must be > 0")

    # -- derived, never stored ------------------------------------------------
    @property
    def nominal_slots(self) -> int:
        """§3.3: nominal recursive slots per arm and search seed = B * K * T."""
        return self.beam_width * self.beam_candidates * self.max_iterations

    @property
    def cohort_id(self) -> str:
        return "{}:{}".format(self.name, len(self.cohort))

    def identity(self) -> Dict[str, Any]:
        """The hashable identity of the profile (excluding operational bits)."""
        d = asdict(self)
        return d

    def sha256(self) -> str:
        return sha256_of(self.identity())


def load_profile(path) -> SRIProfile:
    """Read a YAML profile and validate it against the frozen cohorts."""
    import yaml
    p = Path(path)
    if not p.is_file():
        raise ProtocolError("profile not found: {}".format(str(p)))
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ProtocolError("profile {!r} must be a mapping".format(str(p)))
    raw = dict(raw)
    for k in ("search_seeds", "cohort"):
        if k in raw and isinstance(raw[k], (list, tuple)):
            raw[k] = tuple(raw[k])
    return SRIProfile(**raw)


def default_profile_path(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / (name + ".yaml")


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

@dataclass
class RunManifest:
    """Every result-affecting field of ONE arm, plus the provenance hashes.

    `treatment` holds the four fields the arms are allowed to differ in;
    `shared` holds everything that must match; `operational` holds the
    allowlisted non-result-affecting fields (§10).
    """

    schema_version: int = MANIFEST_SCHEMA_VERSION
    profile: Dict[str, Any] = field(default_factory=dict)
    profile_sha256: str = ""
    arm: str = ""                     # "official" | "predictive"
    root_bundle_sha256: str = ""
    shared: Dict[str, Any] = field(default_factory=dict)
    treatment: Dict[str, Any] = field(default_factory=dict)
    operational: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arm:
            raise ProtocolError("manifest.arm must be set")
        if not self.root_bundle_sha256:
            raise ProtocolError(
                "manifest.root_bundle_sha256 must be set -- both arms record "
                "the SAME root bundle hash (§4)")
        # Fail fast on a misplaced key: splatting a dict into `treatment` is an
        # easy way to swallow a field that belongs at the top level (or in
        # `shared`), and parity would then compare an empty value on both sides.
        illegal = sorted(set(self.treatment) - set(TREATMENT_FIELDS))
        if illegal:
            raise ProtocolError(
                "non-allowlisted key(s) in the treatment block: {} -- only {} "
                "may differ between arms".format(illegal, list(TREATMENT_FIELDS)))
        if not self.profile_sha256:
            self.profile_sha256 = sha256_of(self.profile)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        return sha256_of(self.to_dict())

    def write(self, path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True),
                     encoding="utf-8")
        return self.sha256()

    @classmethod
    def read(cls, path) -> "RunManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)


def assert_treatment_parity(a: RunManifest, b: RunManifest) -> None:
    """§4: the two arms may differ ONLY in the treatment fields.

    Fails closed. Also requires the same root bundle hash and the same profile.
    """
    if a.profile_sha256 != b.profile_sha256:
        raise ProtocolError(
            "profile mismatch across arms: {} != {}".format(
                a.profile_sha256, b.profile_sha256))
    if a.root_bundle_sha256 != b.root_bundle_sha256:
        raise ProtocolError(
            "root bundle mismatch across arms: {} != {} (the arms did not share "
            "one frozen root)".format(a.root_bundle_sha256, b.root_bundle_sha256))

    # every shared key must match exactly
    keys = set(a.shared) | set(b.shared)
    bad = {k: (a.shared.get(k), b.shared.get(k)) for k in sorted(keys)
           if a.shared.get(k) != b.shared.get(k)}
    if bad:
        raise ProtocolError(
            "treatment-external fields differ across arms: {}".format(bad))

    # treatment keys may differ, but only the allowlisted ones may even appear
    tkeys = set(a.treatment) | set(b.treatment)
    illegal = sorted(tkeys - set(TREATMENT_FIELDS))
    if illegal:
        raise ProtocolError(
            "non-allowlisted keys placed in the treatment block: {}".format(
                illegal))

    # operational keys are unrestricted, but must be disjoint from shared
    overlap = set(a.operational) & set(a.shared)
    if overlap:
        raise ProtocolError(
            "field(s) both shared and operational: {}".format(sorted(overlap)))


def assert_no_protocol_drift(saved: RunManifest, live: RunManifest) -> None:
    """§10: resume fails closed on ANY result-affecting drift.

    Compares profile, root hash, every shared field and every treatment field.
    Operational fields are compared against the allowlist and otherwise ignored.
    """
    if saved.profile_sha256 != live.profile_sha256:
        raise ProtocolError(
            "resume drift: profile changed ({} -> {})".format(
                saved.profile_sha256, live.profile_sha256))
    if saved.root_bundle_sha256 != live.root_bundle_sha256:
        raise ProtocolError(
            "resume drift: root bundle changed ({} -> {})".format(
                saved.root_bundle_sha256, live.root_bundle_sha256))
    if saved.arm != live.arm:
        raise ProtocolError(
            "resume drift: arm changed ({!r} -> {!r})".format(
                saved.arm, live.arm))

    for block in ("shared", "treatment"):
        s, l = getattr(saved, block), getattr(live, block)
        keys = set(s) | set(l)
        bad = {k: (s.get(k), l.get(k)) for k in sorted(keys)
               if s.get(k) != l.get(k)}
        if bad:
            raise ProtocolError(
                "resume drift in {!r}: {}".format(block, bad))

    unlisted = sorted(set(live.operational) - set(OPERATIONAL_ALLOWLIST))
    if unlisted:
        raise ProtocolError(
            "operational field(s) outside the explicit allowlist: {} -- a field "
            "that can affect a result must live in `shared`, not here"
            .format(unlisted))


# --------------------------------------------------------------------------
# root bundle (§4)
# --------------------------------------------------------------------------

ROOT_BUNDLE_FIELDS: Tuple[str, ...] = (
    "cohort", "task_order", "root_candidate_id", "task_programs_sha256",
    "traces_sha256", "dev_scores", "archive_metadata",
    "rng_state", "evaluator_manifest", "model_config",
    "software_revision", "environment_fingerprint",
)


def root_bundle_sha256(bundle: Mapping[str, Any]) -> str:
    """Hash a root bundle, refusing one that is missing required material.

    A partial bundle would let the two arms start from subtly different states
    while reporting the same hash, which is exactly the failure §4 guards.
    """
    missing = [k for k in ROOT_BUNDLE_FIELDS if k not in bundle]
    if missing:
        raise ProtocolError(
            "root bundle missing required field(s): {}".format(missing))
    return sha256_of({k: bundle[k] for k in ROOT_BUNDLE_FIELDS})


# --------------------------------------------------------------------------
# resolved-effective configuration table (§8)
# --------------------------------------------------------------------------

def resolve_effective_config(profile: SRIProfile, *,
                             reduction_mode: str,
                             consolidate: bool,
                             gate_tasks: int,
                             gate_margin: Optional[float],
                             protect_floor: Optional[float],
                             regression_guard: bool) -> Dict[str, Any]:
    """What the run will ACTUALLY do, so an inert `gate_tasks=3` cannot be
    mistaken for an active three-task gate (§8).

    With `consolidate=true` the quality gate is skipped entirely; the table says
    so explicitly rather than echoing the configured number.
    """
    gate_effective = bool(gate_tasks > 0 and not consolidate)
    if consolidate:
        gate_reason = "consolidation_focus"
    elif gate_tasks <= 0:
        gate_reason = "gate_disabled"
    else:
        gate_reason = "active"
    return {
        "profile": profile.name,
        "cohort_id": profile.cohort_id,
        "cohort_size": len(profile.cohort),
        "reduction_mode": reduction_mode,
        "B": profile.beam_width,
        "K": profile.beam_candidates,
        "T": profile.max_iterations,
        "max_depth": profile.max_depth,
        "no_early_stop": profile.no_early_stop,
        "nominal_slots": profile.nominal_slots,
        "search_seeds": list(profile.search_seeds),
        "regression_threshold": profile.regression_threshold,
        "consolidate": bool(consolidate),
        "regression_guard": bool(regression_guard),
        "gate_configured": int(gate_tasks),
        "gate_effective": gate_effective,
        "gate_reason": gate_reason,
        "gate_margin": gate_margin,
        "protect_floor": protect_floor,
    }


def format_effective_config_table(cfg: Mapping[str, Any]) -> str:
    """Render the startup table. Printed so the operator cannot misread it."""
    order = ["profile", "cohort_id", "cohort_size", "reduction_mode", "B", "K",
             "T", "max_depth", "no_early_stop", "nominal_slots", "search_seeds",
             "consolidate", "regression_guard", "gate_configured",
             "gate_effective", "gate_reason", "gate_margin", "protect_floor",
             "regression_threshold"]
    w = max(len(k) for k in order)
    lines = ["SRI RESOLVED-EFFECTIVE CONFIGURATION", "=" * (w + 30)]
    for k in order:
        if k in cfg:
            lines.append("  {:<{w}} : {}".format(k, cfg[k], w=w))
    if not cfg.get("gate_effective"):
        lines.append("  NOTE: gate_effective=false ({}); the configured "
                     "gate_tasks={} will NOT run.".format(
                         cfg.get("gate_reason"), cfg.get("gate_configured")))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# acceptance self-test (offline; no torch, no network)
# --------------------------------------------------------------------------

def self_test() -> int:
    import tempfile
    print("protocol.py acceptance")
    print("=" * 68)

    p = SRIProfile(name="sri_primary6", cohort=PRIMARY6, beam_width=2,
                   beam_candidates=2, max_iterations=6, max_depth=6,
                   no_early_stop=True, search_seeds=(0, 1, 2, 3, 4),
                   search_eval_repeats=3, audit_repeats=3, test_repeats=3)
    assert p.nominal_slots == 24, p.nominal_slots
    print("OK  derived slots      B*K*T = 2*2*6 = {} (never a stored field)"
          .format(p.nominal_slots))

    # cohort/name mismatch is refused
    try:
        SRIProfile(name="sri_primary6", cohort=EXTENDED10, beam_width=2,
                   beam_candidates=2, max_iterations=6, max_depth=6,
                   no_early_stop=True, search_seeds=(0,), search_eval_repeats=3,
                   audit_repeats=3, test_repeats=3)
        raise AssertionError("a profile carrying the wrong cohort was accepted")
    except ProtocolError:
        pass
    print("OK  cohort lock        primary6 cannot carry the extended cohort")

    # hashing is stable and order-insensitive
    assert sha256_of({"b": 1, "a": 2}) == sha256_of({"a": 2, "b": 1})
    print("OK  canonical hash     key order does not change the digest")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        common = dict(profile=p.identity(), profile_sha256=p.sha256(),
                      root_bundle_sha256="R" * 64,
                      shared={"cohort": list(PRIMARY6), "model": "gpt-5.5",
                              "temperature": 1.0, "B": 2, "K": 2, "T": 6},
                      operational={"output_dir": str(td)})
        off = RunManifest(arm="official",
                          treatment={"reduction_mode": "official"}, **common)
        pre = RunManifest(arm="predictive",
                          treatment={"reduction_mode": "predictive",
                                     "gamma_checkpoint_sha256": "G" * 64},
                          **common)
        assert_treatment_parity(off, pre)
        print("OK  parity             two arms differing only in treatment pass")

        # a misplaced treatment key is refused AT CONSTRUCTION
        try:
            RunManifest(arm="official", treatment={"reduction_mode": "official",
                                                   "model": "gpt-5.5"}, **common)
            raise AssertionError("a non-allowlisted treatment key was accepted")
        except ProtocolError as e:
            assert "non-allowlisted key" in str(e)
        print("OK  treatment keys     a misplaced key is refused at construction")

        # a shared field differing is fatal
        bad = RunManifest(arm="predictive",
                          treatment={"reduction_mode": "predictive"},
                          **dict(common, shared=dict(common["shared"], T=7)))
        try:
            assert_treatment_parity(off, bad)
            raise AssertionError("a differing shared field was accepted")
        except ProtocolError as e:
            assert "treatment-external" in str(e)
        print("OK  parity fails closed T changed across arms -> refused")

        # a different root bundle hash is fatal
        bad2 = RunManifest(arm="predictive",
                           treatment={"reduction_mode": "predictive"},
                           **dict(common, root_bundle_sha256="S" * 64))
        try:
            assert_treatment_parity(off, bad2)
            raise AssertionError("differing root bundles were accepted")
        except ProtocolError as e:
            assert "root bundle mismatch" in str(e)
        print("OK  shared root        differing root hashes refused")

        # resume drift
        assert_no_protocol_drift(off, RunManifest(**off.to_dict()))
        drifted = RunManifest(**off.to_dict())
        drifted.shared["model"] = "other"
        try:
            assert_no_protocol_drift(off, drifted)
            raise AssertionError("model drift on resume was accepted")
        except ProtocolError as e:
            assert "resume drift" in str(e)
        print("OK  resume drift       model change on resume refused")

        # operational fields must be allowlisted
        unlisted = RunManifest(**off.to_dict())
        unlisted.operational["something_sneaky"] = 1
        try:
            assert_no_protocol_drift(off, unlisted)
            raise AssertionError("an unlisted operational field was accepted")
        except ProtocolError as e:
            assert "allowlist" in str(e)
        print("OK  allowlist          unlisted operational field refused")

        # write / read round-trip
        f = td / "manifest.json"
        h = off.write(f)
        assert RunManifest.read(f).sha256() == h
        print("OK  manifest round-trip byte-identical digest after write/read")

    # root bundle completeness
    full = {k: k for k in ROOT_BUNDLE_FIELDS}
    assert len(root_bundle_sha256(full)) == 64
    try:
        root_bundle_sha256({k: k for k in ROOT_BUNDLE_FIELDS[:-1]})
        raise AssertionError("an incomplete root bundle was accepted")
    except ProtocolError as e:
        assert "missing required field" in str(e)
    print("OK  root bundle        incomplete bundle refused")

    # effective-config table must not echo an inert gate as active
    cfg = resolve_effective_config(p, reduction_mode="official",
                                   consolidate=True, gate_tasks=3,
                                   gate_margin=0.0, protect_floor=None,
                                   regression_guard=True)
    assert cfg["gate_configured"] == 3 and cfg["gate_effective"] is False
    assert cfg["gate_reason"] == "consolidation_focus"
    txt = format_effective_config_table(cfg)
    assert "will NOT run" in txt
    print("OK  effective table    consolidate -> gate_effective=false, "
          "gate_reason=consolidation_focus, warned in the printed table")

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
