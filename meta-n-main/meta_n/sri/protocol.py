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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    # Not a paper profile: the same protocol at minimum scale (B=K=T=1), used to
    # run the whole real chain once before any formal spend. It is registered here
    # so it still cannot carry a cohort that does not match its name.
    "sri_smoke": PRIMARY6,
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


def task_slug(name: str) -> str:
    """The canonical CO-Bench task id: what the RUN's artifacts are keyed by.

    CO-Bench has TWO id spaces and they are not interchangeable:

      * the **display name** (`"Bin packing - one-dimensional"`) is the data
        directory under `data/co_bench/` and the value `--bench-tasks` takes;
      * the **slug** (`"bin_packing___one_dimensional"`) is `TaskDescription
        .task_id`, and therefore the trace FILENAME, every `per_task_scores` /
        `per_task_best` key in the archive index, and the id the ledger records.

    Mirrors `meta_n.integrations.co_bench._task_id_from_name` exactly. A real run
    proved why this matters: material loaders looking for
    `traces/<display name>.py` find NOTHING, so every edge becomes a
    missing-material drop and `task_programs_sha256` comes back all-None.
    `tests/test_sri_formal_protocol.py::test_20` checks this mirror against the
    integration on a box that can import it.
    """
    return str(name).lower().replace(" ", "_").replace("-", "_")


def cohort_id_map(cohort: Sequence[str]) -> Dict[str, str]:
    """slug -> display name, for the few places that must read a DATA directory."""
    return {task_slug(t): str(t) for t in cohort}


# §2.1: candidates meta-n SYNTHESIZES by assembly rather than breeding. The
# archive holds one -- `merge_oracle`, the task-wise router over per-task
# solutions, written with a fabricated `depth` (it is a merge of many depths)
# and `mean_score = oracle_mean_score` (the highest achievable, since it is the
# virtual oracle by construction).
#
# Such a candidate may not compete for Final Score: it is a different object from
# the single deployable state §2.1 defines, and its fabricated depth would
# contaminate `exact_depth_best` with the oracle's own score. It survives ONLY
# as the explicitly named diagnostic `oracle_upper_bound_dev`.
SYNTHESIZED_CANDIDATE_IDS: frozenset = frozenset({"merge_oracle"})


def is_synthesized_candidate(c: Mapping[str, Any]) -> bool:
    """Whether a candidate is a synthesized oracle/merge, not a bred candidate."""
    cid = str(c.get("candidate_id") or "")
    if cid in SYNTHESIZED_CANDIDATE_IDS:
        return True
    return str(c.get("kind") or c.get("origin") or "") in ("synthesized",
                                                           "oracle", "merge")


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
    """A frozen formal search profile (§3.3).

    EVERY result-affecting command parameter lives here and nowhere else. The
    runner renders the CLI and the config.json expectation from this object, and
    then verifies the run's real ``config.json`` against it field by field.

    The alternative was tried and failed: the first revision passed only a few
    flags and let the rest come from the *bundled* benchmark-features YAML (which
    turns on `consolidate` / `regression_guard` / `within_task_recursion` /
    `focus_current_headroom` / `symmetric_trace_sampling` for every benchmark).
    A profile then claimed `max_depth=6, eval_repeats=3` while the run actually
    used the argparse defaults (`10`, `1`) -- a silent 6-vs-10 depth mismatch
    that no artifact recorded. Pinning the flags explicitly makes the profile the
    single source of truth and makes the shipping YAML unable to move a number.
    """

    name: str
    cohort: Tuple[str, ...]
    # -- search shape --------------------------------------------------------
    beam_width: int
    beam_candidates: int
    max_iterations: int
    max_depth: int
    no_early_stop: bool
    # -- seed / repeat families (§3.3) ---------------------------------------
    search_seeds: Tuple[int, ...]
    search_eval_repeats: int
    audit_repeats: int
    test_repeats: int
    # -- generation knobs consumed by the evolutionary path ------------------
    max_tokens: int
    empty_retry_max_tokens: int
    max_retries: int
    reasoning_effort: str
    # -- execution concurrency, pinned so the SEARCH and the AUDIT agree -----
    # These are not cosmetic. Leaving `instance_workers` unset made the search
    # resolve it to min(cpu_count, 8) while the audit defaulted to 2 -- and for
    # CO-Bench the dev score of a slow task is literally
    # "#instances finishing inside the 10s timeout / #instances", so the two
    # sides were scoring under different timeouts-per-instance. Measured on the
    # real data: `assignment_problem` scored 0.5000 under the search's
    # configuration and 0.2500 under a contended one, from the SAME
    # deterministic script (`assign400/700/p3000` -> "Timeout (10s)").
    #
    # `0` is refused on purpose: 0 means "resolve to cpu_count", which is a
    # machine-dependent number and therefore not a reproducible parameter.
    parallel: int
    instance_workers: int
    # A hard ceiling on BILLED backend requests PER ARM. Not a cost-optimisation:
    # it bounds what a runaway generation loop can spend on someone else's relay
    # quota. The value is generous relative to the work (the phaseH launch used
    # 120 for a 4-iteration, B=1 run; this profile asks for 24 slots per arm), so
    # it should never fire on a healthy run -- and if it ever does, the slot ends
    # as `budget_halt` in the ledger, which the audit counts rather than ignores.
    max_backend_requests: int
    temperatures: Tuple[float, ...]
    novelty_alpha: float
    # -- gate / consolidation knobs (§8) ------------------------------------
    consolidate: bool
    gate_tasks: int
    gate_repeats: int
    gate_margin: Optional[float]
    protect_floor: Optional[float]
    regression_guard: bool
    regression_guard_repeats: int
    within_task_recursion: bool
    focus_current_headroom: bool
    symmetric_trace_sampling: bool
    paired_eval: bool
    elite_rotation: bool
    # -- provenance of the shipped feature file ------------------------------
    # Formal runs pin this to "none" so the bundled YAML cannot contribute a
    # single result-affecting key behind the profile's back.
    benchmark_config: str
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
                  "test_repeats", "max_tokens", "gate_repeats",
                  "regression_guard_repeats"):
            if int(getattr(self, k)) < 1:
                raise ProtocolError("{} must be >= 1".format(k))
        if int(self.max_tokens) < 1:
            raise ProtocolError("max_tokens must be >= 1")
        if int(self.empty_retry_max_tokens) < 0:
            raise ProtocolError("empty_retry_max_tokens must be >= 0")
        if int(self.max_retries) < 0:
            raise ProtocolError("max_retries must be >= 0")
        for k in ("parallel", "instance_workers", "max_backend_requests"):
            if int(getattr(self, k)) < 1:
                raise ProtocolError(
                    "{} must be >= 1: 0 means 'no cap' (or 'resolve to "
                    "cpu_count'), which would let the search and the audit score "
                    "under different configurations, or leave a runaway loop "
                    "unbounded on someone else's relay quota".format(k))
        if int(self.gate_tasks) < 0:
            raise ProtocolError("gate_tasks must be >= 0 (0 disables the gate)")
        if self.gate_margin is not None and not isinstance(
                self.gate_margin, float):
            raise ProtocolError(
                "gate_margin must be a float or None (None = no thresholding)")
        if self.protect_floor is not None and not isinstance(
                self.protect_floor, float):
            raise ProtocolError("protect_floor must be a float or None")
        if not self.reasoning_effort:
            raise ProtocolError("reasoning_effort must be set")
        if not self.temperatures:
            raise ProtocolError("temperatures must be non-empty")
        if any(not (0.0 <= float(t) <= 2.0) for t in self.temperatures):
            raise ProtocolError("every temperature must lie in [0, 2]")
        if float(self.novelty_alpha) < 0:
            raise ProtocolError("novelty_alpha must be >= 0")
        if self.benchmark_config != "none":
            raise ProtocolError(
                "formal profiles must pin benchmark_config='none'; the bundled "
                "benchmark-features YAML would otherwise set result-affecting "
                "keys (consolidate / regression_guard / within_task_recursion / "
                "focus_current_headroom / symmetric_trace_sampling) that the "
                "profile does not name")
        if not self.search_seeds:
            raise ProtocolError("search_seeds must be non-empty")
        if len(set(self.search_seeds)) != len(self.search_seeds):
            raise ProtocolError("search_seeds must be distinct")
        if self.regression_threshold <= 0:
            raise ProtocolError("regression_threshold must be > 0")
        # The gate is skipped under consolidation, so a profile that claims both
        # an active gate and consolidation would be self-contradictory -- and an
        # inert `gate_tasks` is exactly what §8 forbids reporting as active.
        if self.consolidate and self.gate_tasks > 0:
            raise ProtocolError(
                "consolidate=True skips the quality gate entirely, so "
                "gate_tasks={} would be inert; set gate_tasks=0 and let the "
                "effective-config table report gate_effective=false".format(
                    self.gate_tasks))

    # -- derived, never stored ------------------------------------------------
    @property
    def nominal_slots(self) -> int:
        """§3.3: nominal recursive slots per arm and search seed = B * K * T."""
        return self.beam_width * self.beam_candidates * self.max_iterations

    @property
    def canonical_cohort(self) -> Tuple[str, ...]:
        """The cohort in the CANONICAL id space (slugs) -- see `task_slug`.

        The profile keeps the display names because that is the frozen cohort
        identity and what `--bench-tasks` accepts; everything that touches a
        run ARTIFACT (trace files, `per_task_scores`, the ledger) uses this.
        """
        return tuple(task_slug(t) for t in self.cohort)

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

ROOT_CANDIDATE_ID = "gen0_seed"


def root_candidate_digest(archive_dir, cohort: Sequence[str],
                          root_candidate_id: str = ROOT_CANDIDATE_ID
                          ) -> Dict[str, Any]:
    """The parts of a root that EVERY arm must inherit UNCHANGED (§4).

    Contents only: the root candidate's program SOURCE per cohort task, every
    file under its `traces/`, and its development scores. Deliberately NOT the
    whole archive: an arm breeds, so its `index.json` grows and a whole-archive
    digest could never match after the fork.

    This is the check that catches a silently FRESH arm. `--resume` restores the
    archive from the copied root, but a failed resume quietly starts a new run
    (the orchestrator treats it as a fresh run by design), and nothing else in
    the pipeline would notice: the ledger would still fill 24 slots and the
    config.json would still name the shared root. Re-hashing the root candidate
    from the ARM's own directory is what makes "both arms inherited one root" a
    measurement rather than an assumption.
    """
    archive_dir = Path(archive_dir)
    root_dir = archive_dir / root_candidate_id
    if not root_dir.is_dir():
        raise ProtocolError(
            "root candidate {!r} is not at {}; the shared root has not been "
            "generated (§4)".format(root_candidate_id, str(root_dir)))
    traces_dir = root_dir / "traces"

    programs: Dict[str, Optional[str]] = {}
    for t in [str(x) for x in cohort]:
        src = None
        for ext in (".py", ".json", ".md", ""):
            c = traces_dir / (t + ext)
            if c.is_file():
                src = c.read_text(encoding="utf-8", errors="replace")
                break
        programs[t] = sha256_of(src) if src is not None else None

    dev_scores: Dict[str, Any] = {}
    idx_path = archive_dir / "index.json"
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        for c in idx.get("candidates", []):
            if str(c.get("candidate_id")) == root_candidate_id:
                dev_scores = dict(c.get("per_task_scores") or {})
                dev_scores["__macro__"] = c.get("mean_score")
    return {"root_candidate_id": root_candidate_id,
            "task_programs_sha256": programs,
            "traces_sha256": hash_tree(traces_dir),
            "dev_scores": dev_scores}


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_tree(root) -> Dict[str, Any]:
    """Content-addressed manifest of a directory tree: name -> sha256.

    Hashes CONTENT, never paths or mtimes, so the digest is the same for two
    copies of the same material living in different directories. That is what
    lets the two arms' root bundles be compared at all.
    """
    root = Path(root)
    out: Dict[str, str] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_dir() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        out[rel] = _sha256_bytes(p.read_bytes())
    return out


def git_revision(repo_dir=None) -> Dict[str, Any]:
    """The software revision the run was produced from (§4).

    A DIRTY tree is recorded as dirty rather than silently hashing to the last
    commit: an uncommitted edit is exactly the kind of unreproducibility the
    root bundle exists to expose.
    """
    import subprocess
    d = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent.parent.parent
    try:
        sha = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        st = subprocess.run(["git", "-C", str(d), "status", "--porcelain"],
                            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as e:      # noqa: BLE001
        return {"git_sha": None, "dirty": None, "error": str(e)}
    if sha.returncode != 0:
        return {"git_sha": None, "dirty": None,
                "error": (sha.stderr or "").strip()[:200]}
    return {"git_sha": sha.stdout.strip(),
            "dirty": bool((st.stdout or "").strip())}


def environment_fingerprint(*, worker_config: Optional[Mapping[str, Any]] = None
                            ) -> Dict[str, Any]:
    """What machine/process the run happened on (§4, §9).

    Deliberately part of the root hash: two machines that do not share a
    fingerprint do not share a root, so their scores can never be paired (§9).
    """
    import os
    import platform
    fp: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "worker_config": dict(worker_config or {}),
    }
    for mod in ("numpy", "torch", "transformers"):
        try:
            m = __import__(mod)
            fp[mod] = getattr(m, "__version__", "unknown")
        except Exception:                                    # noqa: BLE001
            fp[mod] = None
    return fp


def collect_root_bundle(*, archive_dir, cohort: Sequence[str], backbone: str,
                        search_seed: int, data_dir=None, base_url=None,
                        temperatures: Sequence[float] = (),
                        max_tokens: Optional[int] = None,
                        reasoning_effort: Optional[str] = None,
                        worker_config: Optional[Mapping[str, Any]] = None,
                        repo_dir=None,
                        data_dir_name_of: Optional[Mapping[str, str]] = None,
                        root_candidate_id: str = "gen0_seed") -> Dict[str, Any]:
    """Build the §4 root bundle from REAL artifacts.

    The first revision hashed candidate ids and scores (`task_programs_sha256`),
    a list of trace FILENAMES (`traces_sha256`), and left four fields None. None
    of those pin the material: two roots whose children were generated from
    different programs hash identically. This version hashes program SOURCE,
    trace CONTENT, the evaluator's data tree, the model/worker configuration and
    the software revision -- the things a reader would need to re-execute.

    `cohort` is the CANONICAL id space (slugs) because that is what the trace
    files are named; `data_dir_name_of` maps a slug to its DATA-DIRECTORY name,
    because CO-Bench names those with the display name. See `task_slug`.

    Treatment-free by construction: nothing here depends on the reduction mode,
    so both arms produce the SAME hash from the same root.
    """
    archive_dir = Path(archive_dir)
    inherited = root_candidate_digest(archive_dir, cohort, root_candidate_id)
    traces_dir = archive_dir / root_candidate_id / "traces"

    # -- the programs: SOURCE of every cohort task, not ids ---------------
    cohort = [str(t) for t in cohort]
    dev_scores = dict(inherited["dev_scores"])
    index_sha = None
    candidates_meta: Dict[str, Any] = {}
    idx_path = archive_dir / "index.json"
    if idx_path.is_file():
        index_sha = _sha256_bytes(idx_path.read_bytes())
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        candidates_meta = {"size": len(idx.get("candidates", [])),
                           "candidate_ids": sorted(
                               str(c.get("candidate_id"))
                               for c in idx.get("candidates", []))}

    # -- the evaluator: the canonical dev instances it reads ---------------
    evaluator: Dict[str, Any] = {"data_dir": None, "tasks": {}}
    if data_dir is not None:
        dd = Path(data_dir)
        evaluator["data_dir"] = "relative:" + str(
            dd.name) if dd.is_absolute() else str(dd)
        # the DATA directory is named with the display name; the cohort key is
        # the canonical slug (see `task_slug`)
        name_of = dict(data_dir_name_of or {})
        for t in cohort:
            td = dd / name_of.get(t, t)
            if td.is_dir():
                evaluator["tasks"][t] = hash_tree(td)
            else:
                evaluator["tasks"][t] = None

    return {
        "cohort": cohort,
        "task_order": cohort,
        "root_candidate_id": root_candidate_id,
        "task_programs_sha256": inherited["task_programs_sha256"],
        "traces_sha256": inherited["traces_sha256"],
        "dev_scores": dev_scores,
        "archive_metadata": dict(candidates_meta, index_sha256=index_sha),
        # The process RNG is not persisted by meta-n, so the bundle records the
        # full deterministic seed material instead of claiming a state it does
        # not have.
        "rng_state": {"search_seed": int(search_seed),
                      "search_seeds_persisted": False,
                      "temperatures": [float(x) for x in temperatures]},
        "evaluator_manifest": evaluator,
        "model_config": {"backbone": backbone, "base_url": base_url,
                         "temperatures": [float(x) for x in temperatures],
                         "max_tokens": max_tokens,
                         "reasoning_effort": reasoning_effort},
        "software_revision": git_revision(repo_dir),
        "environment_fingerprint": environment_fingerprint(
            worker_config=worker_config),
    }


# --------------------------------------------------------------------------
# pinned command parameters (§8, audit #3)
# --------------------------------------------------------------------------

# config.json key -> the CLI flag that sets it. Used in BOTH directions: to
# render the command from the profile, and to verify the run's real config.json
# afterwards. The two directions sharing one table is what makes "the profile is
# the only source of truth" checkable instead of aspirational.
CONFIG_KEY_TO_CLI: Dict[str, str] = {
    "model": "--model",
    "max_tokens": "--max-tokens",
    "empty_retry_max_tokens": "--empty-retry-max-tokens",
    "max_depth": "--max-depth",
    "max_retries": "--max-retries",
    "reasoning_effort": "--reasoning-effort",
    "seed": "--seed",
    "benchmark": "--benchmark",
    "bench_tasks": "--bench-tasks",
    "benchmark_config": "--benchmark-config",
    "use_archive": "--use-archive",
    "beam_width": "--beam-width",
    "beam_candidates": "--beam-candidates",
    "max_iterations": "--max-iterations",
    "no_early_stop": "--no-early-stop",
    "gate_tasks": "--gate-tasks",
    "gate_repeats": "--gate-repeats",
    "gate_margin": "--gate-margin",
    "eval_repeats": "--eval-repeats",
    "paired_eval": "--paired-eval",
    "consolidate": "--consolidate",
    "protect_floor": "--protect-floor",
    "regression_guard": "--regression-guard",
    "regression_guard_repeats": "--regression-guard-repeats",
    "within_task_recursion": "--within-task-recursion",
    "focus_current_headroom": "--focus-current-headroom",
    "symmetric_trace_sampling": "--symmetric-trace-sampling",
    "novelty_alpha": "--novelty-alpha",
    "elite_rotation": "--elite-rotation",
    "temperatures": "--temperatures",
    "parallel": "--parallel",
    "instance_workers": "--instance-workers",
    "max_backend_requests": "--max-backend-requests",
}

# Flags that accept an explicit `--no-` form (argparse.BooleanOptionalAction).
# Rendering BOTH directions explicitly is what stops the bundled YAML from
# re-enabling a knob the profile turned off.
NEGATABLE_BOOL_CONFIG_KEYS: Tuple[str, ...] = (
    "use_archive", "consolidate", "regression_guard", "within_task_recursion",
    "focus_current_headroom", "symmetric_trace_sampling", "elite_rotation")

# `store_true` flags: the CLI cannot express False, so False renders as an
# OMITTED flag and the config.json verification then proves the default really
# was False.
POSITIVE_ONLY_BOOL_CONFIG_KEYS: Tuple[str, ...] = ("no_early_stop",
                                                   "paired_eval")

# Keys whose flag is a plain `type=float` with default None: there is no
# `none` token, so None must render as an omitted flag.
OMIT_WHEN_NONE_CONFIG_KEYS: Tuple[str, ...] = ("protect_floor",)

# Pinned for RENDERING but not verifiable against config.json, because meta-n
# records the same fact under a different provenance key. `--use-archive` is the
# flag; `config.json` carries `orchestrator: "evolutionary"` instead, which is
# what the verification checks. Verified on a REAL config.json (72 keys, from
# runs/phaseH): `use_archive` is absent, so requiring it would have failed the
# arm stage AFTER its paid run had finished.
# `max_backend_requests` is in the same category: main.py turns the flag into
# $META_N_MAX_BACKEND_REQUESTS (main.py:1711) and never writes it into
# config.json (verified against the real 72-key config.json).
RENDER_ONLY_CONFIG_KEYS: Tuple[str, ...] = ("use_archive",
                                           "max_backend_requests")


def pinned_run_config(profile: SRIProfile, *, backbone: str, search_seed: int,
                      benchmark: str = "co_bench") -> Dict[str, Any]:
    """The exact result-affecting config.json keys this profile requires (§8).

    Everything NOT listed here is left to the code default and is verified equal
    across arms by the runner's config.json diff, so it can never carry the
    treatment.
    """
    return {
        "model": backbone,
        "max_tokens": int(profile.max_tokens),
        "empty_retry_max_tokens": int(profile.empty_retry_max_tokens),
        "max_depth": int(profile.max_depth),
        "max_retries": int(profile.max_retries),
        "reasoning_effort": str(profile.reasoning_effort),
        "seed": int(search_seed),
        "benchmark": benchmark,
        "bench_tasks": list(profile.cohort),
        "benchmark_config": str(profile.benchmark_config),
        "benchmark_config_applied": [],
        "use_archive": True,
        # the provenance key meta-n actually writes for the archive orchestrator
        "orchestrator": "evolutionary",
        "beam_width": int(profile.beam_width),
        "beam_candidates": int(profile.beam_candidates),
        "max_iterations": int(profile.max_iterations),
        "no_early_stop": bool(profile.no_early_stop),
        "gate_tasks": int(profile.gate_tasks),
        "gate_repeats": int(profile.gate_repeats),
        "gate_margin": (None if profile.gate_margin is None
                        else float(profile.gate_margin)),
        "eval_repeats": int(profile.search_eval_repeats),
        "paired_eval": bool(profile.paired_eval),
        "consolidate": bool(profile.consolidate),
        "protect_floor": (None if profile.protect_floor is None
                          else float(profile.protect_floor)),
        "regression_guard": bool(profile.regression_guard),
        "regression_guard_repeats": int(profile.regression_guard_repeats),
        "within_task_recursion": bool(profile.within_task_recursion),
        "focus_current_headroom": bool(profile.focus_current_headroom),
        "symmetric_trace_sampling": bool(profile.symmetric_trace_sampling),
        "novelty_alpha": float(profile.novelty_alpha),
        "elite_rotation": bool(profile.elite_rotation),
        "temperatures": [float(t) for t in profile.temperatures],
        "parallel": int(profile.parallel),
        "instance_workers": int(profile.instance_workers),
        "max_backend_requests": int(profile.max_backend_requests),
    }


def render_pinned_flags(pinned: Mapping[str, Any]) -> List[str]:
    """Render the pinned parameters as argv, or refuse one that cannot be said.

    A `store_true` flag needs a `--no-` form to express False; where none exists
    the parameter would silently fall back to whatever the code default is, so
    this raises instead of emitting a command that does not say what it means.
    """
    argv: List[str] = []
    for key, value in sorted(pinned.items()):
        flag = CONFIG_KEY_TO_CLI.get(key)
        if flag is None:
            continue                     # not a CLI-backed key (e.g. applied[])
        if key in NEGATABLE_BOOL_CONFIG_KEYS:
            argv.append(flag if value else "--no-" + flag[2:])
        elif key in POSITIVE_ONLY_BOOL_CONFIG_KEYS:
            if value:
                argv.append(flag)
            elif flag == "--paired-eval":
                pass                     # default False: omitting says it
            else:
                raise ProtocolError(
                    "{} is a store_true flag with no --no- form, so the profile "
                    "cannot pin it to False via the CLI".format(flag))
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(x) for x in value)
        elif value is None and key in OMIT_WHEN_NONE_CONFIG_KEYS:
            continue                     # flag default IS None: omit it
        elif value is None:
            argv += [flag, "none"]
        else:
            argv += [flag, str(value)]
    return argv


def verify_run_config(actual: Mapping[str, Any], pinned: Mapping[str, Any],
                      *, context: str = "") -> None:
    """Field-by-field check of the run's REAL config.json against the profile.

    The failure this catches is not hypothetical: the first revision let
    `max_depth` (profile 6) and `eval_repeats` (profile 3) come from argparse
    defaults, so the run silently used 10 and 1 while every manifest claimed
    otherwise.
    """
    bad: Dict[str, Any] = {}
    skipped: List[str] = []
    for key, want in sorted(pinned.items()):
        if key in RENDER_ONLY_CONFIG_KEYS:
            # pinned as a FLAG; the recorded provenance is another key (see
            # RENDER_ONLY_CONFIG_KEYS). Skipping it silently would be the same
            # mistake in the other direction, so it is reported.
            skipped.append(key)
            continue
        got = actual.get(key, "<absent>")
        if got != want:
            bad[key] = {"profile": want, "config_json": got}
    if skipped and not any(k == "orchestrator" for k in actual):
        raise ProtocolError(
            "config.json has neither {} nor the provenance key that replaces "
            "them; the archive-orchestrator mode is UNVERIFIABLE".format(
                list(skipped)))
    if bad:
        raise ProtocolError(
            "the run's config.json does not match the profile{}: {}".format(
                (" ({})".format(context) if context else ""), bad))


# --------------------------------------------------------------------------
# resolved-effective configuration table (§8)
# --------------------------------------------------------------------------

def resolve_effective_config(profile: SRIProfile, *,
                             reduction_mode: str) -> Dict[str, Any]:
    """What the run will ACTUALLY do, so an inert `gate_tasks=3` cannot be
    mistaken for an active three-task gate (§8).

    Every gate/consolidation value is read FROM THE PROFILE (§3.3): the first
    revision took them as loose keyword arguments, which let the printed table
    describe one configuration while the command line ran another. With
    `consolidate=true` the quality gate is skipped entirely; the table says so
    explicitly rather than echoing the configured number.
    """
    consolidate = bool(profile.consolidate)
    gate_tasks = int(profile.gate_tasks)
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
        "consolidate": consolidate,
        "regression_guard": bool(profile.regression_guard),
        "gate_configured": gate_tasks,
        "gate_effective": gate_effective,
        "gate_reason": gate_reason,
        "gate_margin": profile.gate_margin,
        "protect_floor": profile.protect_floor,
        "within_task_recursion": bool(profile.within_task_recursion),
        "focus_current_headroom": bool(profile.focus_current_headroom),
        "symmetric_trace_sampling": bool(profile.symmetric_trace_sampling),
        "paired_eval": bool(profile.paired_eval),
        "temperatures": list(profile.temperatures),
        "benchmark_config": profile.benchmark_config,
    }


def format_effective_config_table(cfg: Mapping[str, Any]) -> str:
    """Render the startup table. Printed so the operator cannot misread it."""
    order = ["profile", "cohort_id", "cohort_size", "reduction_mode", "B", "K",
             "T", "max_depth", "no_early_stop", "nominal_slots", "search_seeds",
             "consolidate", "regression_guard", "gate_configured",
             "gate_effective", "gate_reason", "gate_margin", "protect_floor",
             "within_task_recursion", "focus_current_headroom",
             "symmetric_trace_sampling", "paired_eval", "temperatures",
             "benchmark_config", "regression_threshold"]
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

def _demo_profile(**over) -> SRIProfile:
    """The frozen primary6 profile as the YAML declares it, for self-tests."""
    base = dict(
        name="sri_primary6", cohort=PRIMARY6, beam_width=2, beam_candidates=2,
        max_iterations=6, max_depth=6, no_early_stop=True,
        search_seeds=(0, 1, 2, 3, 4), search_eval_repeats=3, audit_repeats=3,
        test_repeats=3, max_tokens=1024, empty_retry_max_tokens=0,
        max_retries=0, reasoning_effort="low",
        parallel=1, instance_workers=8, max_backend_requests=400,
        temperatures=(0.5, 0.7, 0.9), novelty_alpha=0.3, consolidate=True,
        gate_tasks=0, gate_repeats=1, gate_margin=0.0, protect_floor=None,
        regression_guard=True, regression_guard_repeats=3,
        within_task_recursion=True, focus_current_headroom=True,
        symmetric_trace_sampling=True, paired_eval=False, elite_rotation=False,
        benchmark_config="none")
    base.update(over)
    return SRIProfile(**base)


def self_test() -> int:
    import tempfile
    print("protocol.py acceptance")
    print("=" * 68)

    p = _demo_profile()
    assert p.nominal_slots == 24, p.nominal_slots
    print("OK  derived slots      B*K*T = 2*2*6 = {} (never a stored field)"
          .format(p.nominal_slots))

    # cohort/name mismatch is refused
    try:
        _demo_profile(cohort=EXTENDED10)
        raise AssertionError("a profile carrying the wrong cohort was accepted")
    except ProtocolError:
        pass
    print("OK  cohort lock        primary6 cannot carry the extended cohort")

    # consolidate + a claimed-active gate is refused: the gate would not run
    try:
        _demo_profile(gate_tasks=3)
        raise AssertionError("an inert gate under consolidate was accepted")
    except ProtocolError as e:
        assert "inert" in str(e)
    print("OK  inert gate refused consolidate=True with gate_tasks=3 is refused "
          "at construction (the gate never runs)")

    # only the shipped feature file can be the config source
    try:
        _demo_profile(benchmark_config="auto")
        raise AssertionError("a profile pointing at the bundled YAML was accepted")
    except ProtocolError as e:
        assert "benchmark_config" in str(e)
    print("OK  feature file pinned a profile may not defer to the bundled "
          "benchmark-features YAML")

    # hashing is stable and order-insensitive
    assert sha256_of({"b": 1, "a": 2}) == sha256_of({"a": 2, "b": 1})
    print("OK  canonical hash     key order does not change the digest")

    # ---- §8/audit #3: the pinned parameters are the profile's, exactly -----
    pinned = pinned_run_config(p, backbone="gpt-5.5", search_seed=0)
    assert pinned["max_depth"] == 6 and pinned["eval_repeats"] == 3
    assert pinned["consolidate"] is True and pinned["gate_tasks"] == 0
    argv = render_pinned_flags(pinned)
    joined = " ".join(argv)
    # the negatable booleans are rendered in BOTH directions, so the bundled
    # YAML cannot re-enable one behind the profile's back
    assert "--no-elite-rotation" in joined
    assert "--within-task-recursion" in joined
    assert "--no-consolidate" not in joined
    assert "--benchmark-config none" in joined
    assert "--gate-margin 0.0" in joined
    assert "--protect-floor" not in joined          # None -> omitted, not "none"
    print("OK  pinned flags       rendered from the profile: depth/eval_repeats "
          "pinned, booleans in both directions, None omitted")

    # a store_true flag pinned False has no CLI form -> refused, never silently
    # left to the code default
    try:
        render_pinned_flags({"no_early_stop": False})
        raise AssertionError("a False store_true flag was rendered")
    except ProtocolError as e:
        assert "--no-early-stop" in str(e)
    print("OK  unrepresentable   a store_true flag cannot be pinned False")

    # verification is field-by-field and names the offender
    good = dict(pinned)
    verify_run_config(good, pinned)
    bad_cfg = dict(pinned, max_depth=10)            # the real historical bug
    try:
        verify_run_config(bad_cfg, pinned, context="root")
        raise AssertionError("a max_depth mismatch passed verification")
    except ProtocolError as e:
        assert "max_depth" in str(e) and "root" in str(e)
    print("OK  config verify      the real max_depth=10-vs-6 mismatch is caught "
          "and names the field")

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

    # ---- §4/audit #6: the bundle hashes MATERIAL, not ids or filenames -----
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "archive" / "gen0_seed"
        (root / "traces").mkdir(parents=True)
        (root / "traces" / "T1.py").write_text("def solve():\n    return 1\n")
        (root / "traces" / "T2.py").write_text("def solve():\n    return 2\n")
        data = td / "data"
        (data / "T1").mkdir(parents=True)
        (data / "T1" / "config.py").write_text("x = 1\n")
        (data / "T1" / "case1").write_text("instance\n")
        b1 = collect_root_bundle(archive_dir=td / "archive", cohort=["T1", "T2"],
                                 backbone="gpt-5.5", search_seed=0,
                                 data_dir=data)
        h1 = root_bundle_sha256(b1)

        # a DIFFERENT PROGRAM with the same filename and the same scores must
        # change the hash -- hashing candidate ids did not catch this
        (root / "traces" / "T1.py").write_text("def solve():\n    return 99\n")
        b2 = collect_root_bundle(archive_dir=td / "archive", cohort=["T1", "T2"],
                                 backbone="gpt-5.5", search_seed=0,
                                 data_dir=data)
        assert root_bundle_sha256(b2) != h1
        assert b1["task_programs_sha256"] != b2["task_programs_sha256"]
        print("OK  programs hashed    an edited program changes the root hash "
              "(ids/scores would not)")

        # a DIFFERENT INSTANCE DATA TREE changes the hash too
        b3 = collect_root_bundle(archive_dir=td / "archive", cohort=["T1", "T2"],
                                 backbone="gpt-5.5", search_seed=0, data_dir=data)
        (data / "T1" / "case1").write_text("different instance\n")
        b4 = collect_root_bundle(archive_dir=td / "archive", cohort=["T1", "T2"],
                                 backbone="gpt-5.5", search_seed=0, data_dir=data)
        assert root_bundle_sha256(b4) != root_bundle_sha256(b3)
        print("OK  evaluator hashed   a changed dev instance changes the root hash")

        # the SAME material in a DIFFERENT directory hashes the same: content,
        # never paths
        td2 = Path(td.parent) / (td.name + "_copy")
        import shutil
        shutil.copytree(td, td2)
        b5 = collect_root_bundle(archive_dir=td2 / "archive", cohort=["T1", "T2"],
                                 backbone="gpt-5.5", search_seed=0,
                                 data_dir=td2 / "data")
        b6 = collect_root_bundle(archive_dir=td / "archive", cohort=["T1", "T2"],
                                 backbone="gpt-5.5", search_seed=0, data_dir=data)
        assert root_bundle_sha256(b5) == root_bundle_sha256(b6)
        shutil.rmtree(td2, ignore_errors=True)
        print("OK  path independent   the same material under a different path "
              "hashes identically")

        # an absent root candidate is refused rather than hashed as empty
        try:
            collect_root_bundle(archive_dir=td / "nope", cohort=["T1"],
                                backbone="b", search_seed=0)
            raise AssertionError("a missing root candidate was accepted")
        except ProtocolError as e:
            assert "has not been generated" in str(e)
        print("OK  missing root       a bundle with no root candidate is refused")

    # effective-config table must not echo an inert gate as active
    cfg = resolve_effective_config(p, reduction_mode="official")
    assert cfg["gate_configured"] == 0 and cfg["gate_effective"] is False
    assert cfg["gate_reason"] == "consolidation_focus"
    assert cfg["within_task_recursion"] is True      # the YAML-only knob, pinned
    txt = format_effective_config_table(cfg)
    assert "will NOT run" in txt
    print("OK  effective table    consolidate -> gate_effective=false, "
          "gate_reason=consolidation_focus, every pinned knob shown")

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
