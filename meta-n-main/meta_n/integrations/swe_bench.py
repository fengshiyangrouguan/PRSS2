"""SWE-bench Verified adapter — thin subclass of TerminalBenchAdapter.

SWE-bench Verified tasks are converted into Harbor-compatible task
directories by harbor's swebench adapter (see
``harbor-ttu/adapters/swebench/``). The directory layout is identical
to TerminalBench 2's, so we inherit ~95% of the loading and execution
logic from TerminalBenchAdapter. The only meaningful divergences:

  (1) task.toml uses ``memory = '4G'`` (string) instead of TB2's
      ``memory_mb = 4096`` (int).
  (2) tests/config.json carries SWE-bench-specific fields
      (instance_id, repo, base_commit, FAIL_TO_PASS, PASS_TO_PASS)
      that we surface into TaskDescription.metadata for richer Ω
      trace feedback.

The Docker runtime (TerminalBenchExecutor) is reused unchanged — the
swebench task verifier (tests/test.sh) writes the same
/logs/verifier/reward.txt 1/0 protocol that TB2 uses.

Data flow:
  harbor SDK pulls task dirs from harbor-datasets ──▶ task_cache_dir
    ─▶ load_tasks() augments metadata with config.json fields
      ─▶ TerminalBenchExecutor.execute() runs solve.sh + test.sh
        ─▶ Trace.score in {0.0, 1.0} (binary: resolved vs not)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from meta_n.integrations.terminal_bench import TerminalBenchAdapter
from meta_n.integrations.terminal_bench.adapter import _BASE_SOLVER_UNDECLARED

logger = logging.getLogger(__name__)


_MEMORY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMG]?)i?\s*$", re.IGNORECASE)


# Environment constraints surfaced into ``TaskDescription.metadata[
# "omega_env_notes"]`` so the Ω engine can prepend them to its prompt
# without hardcoding any benchmark-specific knowledge in core. Empirically
# observed across gpt-5.2 + Qwen3-Coder runs: Ω synthesizes helpers that
# create venvs, call `rg`, or `pip install` — all of which break SWE-bench's
# prebuilt conda env. Anchoring these constraints up-front keeps Ω's output
# inside the envelope the verifier accepts.
_OMEGA_ENV_NOTES = (
    "## Benchmark Environment Notes (SWE-bench Verified)\n"
    "The solver runs inside a prebuilt Docker container per task. Helper "
    "code and pre_process hints must respect these constraints:\n"
    "- Python environment: conda env `testbed` at `/opt/miniconda3` is the "
    "ONLY environment the verifier imports against. Do NOT create new "
    "venvs (`python -m venv`), do NOT install packages (`pip install`), "
    "and do NOT modify `PYTHONPATH`. The testbed env already has every "
    "dependency the repo's tests need; touching it breaks the verifier.\n"
    "- Tools available by default: bash, python (via conda), git, sed, "
    "awk, grep, patch, find, head/tail. DO NOT assume ripgrep (`rg`), "
    "`fd`, `bat`, or other newer CLIs are installed — they often aren't. "
    "Use `grep -rn` or `find ... | xargs grep` instead.\n"
    "- Repository: at `/testbed`, which is also the script's cwd by "
    "default (image WORKDIR). The repo is a git checkout at the task's "
    "base_commit. Modifications outside `/testbed` are not graded.\n"
    "- Tests: the verifier resets test files in test_patch before "
    "grading, so any helper that edits `tests/*` wastes turns."
)


def _parse_memory_string(value, default_mb: int = 2048) -> int:
    """Convert SWE-bench-style memory specs to MB ints.

    Examples:
        '4G'   -> 4096
        '4Gi'  -> 4096   (kubernetes-style suffixes accepted)
        '512M' -> 512
        '2g'   -> 2048   (case-insensitive)
        '1024' -> 1024   (bare number = MB)
        1024   -> 1024   (passthrough for int/float)
        None   -> default_mb
        'garbage' -> default_mb
    """
    if value is None:
        return default_mb
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    m = _MEMORY_RE.match(s)
    if not m:
        return default_mb
    n, unit = float(m.group(1)), m.group(2).upper()
    if unit == "G":
        return int(n * 1024)
    if unit == "K":
        return max(1, int(n / 1024))
    return int(n)


class SWEBenchVerifiedAdapter(TerminalBenchAdapter):
    """Adapter for SWE-bench Verified via harbor's task generator.

    Inherits download/build/run logic from TerminalBenchAdapter; overrides
    only what's needed to (a) parse SWE-bench's memory-string syntax in
    task.toml and (b) surface the per-instance fields from
    tests/config.json into TaskDescription.metadata.
    """

    def __init__(
        self,
        task_cache_dir: str | None = None,
        task_names: list[str] | None = None,
        base_solver: "str | None | object" = _BASE_SOLVER_UNDECLARED,
    ):
        super().__init__(
            task_cache_dir=task_cache_dir,
            task_names=task_names,
            dataset_name="swebench-verified",
            base_solver=base_solver,
        )

    @property
    def name(self) -> str:
        return "swe_bench_verified"

    def code_library_is_live(self) -> bool:
        # Measured helper call-rate ~0: the agent regenerates patch code inline
        # rather than importing prepended helpers (roadmap v2 4.3) — demote them.
        return False

    def advertises_spine_builtin(self) -> bool:
        """Keep ``base_solver="builtin"`` on the NATIVE executor for SWE-bench.

        TerminalBenchAdapter overrides this to ``True`` because only the legacy
        ``original-tasks`` layout cannot run under the native
        ``TerminalBenchExecutor``. That rationale does NOT apply to SWE-bench:
        its harbor ``task.toml`` layout IS runnable by the native executor (the
        documented path — see this module's docstring). Inheriting the parent's
        ``True`` would misroute ``--base-solver builtin --benchmark
        swe_bench_verified`` onto the external-agent spine + ``BuiltinTBBackend``,
        which is wired to the TerminalBench tasks dir / runner and has zero
        SWE-bench wiring — silently scoring the wrong harness against the wrong
        tasks. Restore the base ``False`` so builtin stays native.
        """
        return False

    # --- External-agent spine: OH/T2 are NOT wired for SWE-bench ---------
    #
    # ``--base-solver openhands/terminus2`` ALWAYS routes through the external
    # spine (``EvolutionaryOrchestrator._uses_external_spine`` returns True for
    # any base not in {None, "builtin"}, regardless of adapter — the
    # ``advertises_spine_builtin`` override above only gates the BUILTIN route).
    # TerminalBenchAdapter's inherited make_* factories would happily build an
    # ``OpenHandsTBBackend`` / ``Terminus2Backend`` pointed at
    # ``_resolve_t2_paths(self._task_cache_dir)`` and drive the *TerminalBench*
    # runner (scripts/{oh,t2}_tb_runner.py). That runner's TrialHandler expects
    # a TB2 ``<tasks_dir>/<task_id>/task.yaml`` layout; SWE-bench's harbor dirs
    # carry ``task.toml`` + ``instruction.md`` (no ``task.yaml``), so the run is
    # silently misrouted onto the wrong harness/tasks. Until a real SWE-bench
    # external-agent runner is wired, fail LOUDLY at backend construction so the
    # unsupported combo can never score the wrong harness.
    _UNSUPPORTED_EXTERNAL_KINDS = ("openhands", "terminus2")

    def _reject_external_kind(self, kind: str) -> None:
        if kind in self._UNSUPPORTED_EXTERNAL_KINDS:
            raise ValueError(
                f"--base-solver {kind!r} is not supported for benchmark "
                f"{self.name!r}: the {kind!r} external-agent backend drives the "
                "TerminalBench runner against a TB2 task.yaml layout, but "
                "SWE-bench Verified uses harbor task.toml + instruction.md dirs "
                "(no task.yaml) — so the run would silently score the wrong "
                "harness/tasks. Use --base-solver builtin (native SWE-bench "
                "executor) until a real SWE-bench external-agent runner is wired."
            )

    def make_agent_backend(self, kind: str, **kw):  # -> AgentBackend | None
        self._reject_external_kind(kind)
        return super().make_agent_backend(kind, **kw)

    def make_env_provider(self, kind: str):  # -> AgentEnvProvider | None
        self._reject_external_kind(kind)
        return super().make_env_provider(kind)

    def make_scorer(self, kind: str):  # -> Scorer | None
        self._reject_external_kind(kind)
        return super().make_scorer(kind)

    # ``download()`` itself is inherited from TerminalBenchAdapter (the harbor
    # preamble is shared); SWE-bench customizes it through the parent's two
    # template hooks below plus the cache-log label.
    _cache_log_label = "SWE-bench task directory"

    def _filter_task_configs(self, task_configs: list) -> list:
        """Apply the ``task_names`` filter at the harbor SDK call so we only
        materialize requested tasks instead of all 500."""
        if self._task_names_filter:
            wanted = set(self._task_names_filter)
            # Exact match on the last path segment (= the task_name), not
            # substring. Substring would over-match — e.g., "-13741" would
            # also match "-137410" or "-13741-extra" if such IDs existed.
            # GitTaskId.path is a PosixPath (confirmed against the live
            # harbor SDK), so .name gives the bare task name directly.
            total_count = len(task_configs)
            filtered = [
                tc for tc in task_configs
                if tc.get_task_id().path.name in wanted
            ]
            if not filtered:
                # Hard-fail rather than silently downloading all 500. The
                # user passed a specific filter — if it matched nothing,
                # something is wrong (typo in task id, dataset changed)
                # and we want to surface it immediately, not start a huge
                # background download.
                available_sample = sorted(
                    tc.get_task_id().path.name for tc in task_configs[:5]
                )
                raise RuntimeError(
                    f"task_names filter {sorted(wanted)} matched 0 of "
                    f"{total_count} configs in dataset "
                    f"{self._dataset_name!r}. First 5 available: "
                    f"{available_sample}"
                )
            task_configs = filtered
            logger.info(
                "task_names filter narrowed download to %d/%d tasks",
                len(filtered), total_count,
            )
        return task_configs

    def _stage_downloaded(self, result) -> None:
        """Symlink-stage downloaded SWE-bench task dirs into one cache dir.

        Overrides the parent hook for two reasons:

        (1) **Layout difference.** The two benchmarks use mirror-image
            harbor layouts:

              TB2:        ``<root>/<task_name>/<hash>/task.toml``
              SWE-bench:  ``<root>/<hash>/<task_name>/task.toml``

        (2) **Content-addressed dispersal.** Harbor's content-addressed
            cache puts each SWE-bench task in its own hash dir. So a
            multi-task download lands paths in DIFFERENT parents:

              ~/.cache/harbor/tasks/AAAA/django__django-11265/
              ~/.cache/harbor/tasks/BBBB/sympy__sympy-13798/

            The parent's "first.parent.parent" trick doesn't help here
            because the tasks share no useful common ancestor (the actual
            common ancestor, ``~/.cache/harbor/tasks/``, would also expose
            unrelated cache dirs from other datasets / runs).

        Fix: after harbor downloads, stage every returned task dir into a
        single managed staging dir via symlinks. That stable, dataset-
        specific staging dir becomes our ``_task_cache_dir`` — load_tasks
        sees a flat list of task_name subdirs, exactly the layout it
        expects.
        """
        # Stage all downloaded task dirs into a single dataset-specific
        # location via symlinks. Each result.paths[i] may live under its
        # own content-addressed harbor hash dir; symlinking gives
        # load_tasks the flat layout it expects.
        staging = Path.home() / ".cache" / "meta_n" / "swebench_verified"
        staging.mkdir(parents=True, exist_ok=True)
        staged = 0
        for path in result.paths:
            link = staging / path.name
            # Refresh: remove a stale symlink (possibly dangling) but
            # never rm a real directory. If something other than a
            # symlink occupies the name, leave it and warn — the user
            # may have manually placed task dirs there.
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                logger.warning(
                    "staging path %s exists and is not a symlink; "
                    "leaving in place (load_tasks will use it as-is)",
                    link,
                )
                staged += 1
                continue
            link.symlink_to(path, target_is_directory=True)
            staged += 1
        self._task_cache_dir = staging
        logger.info(
            "Staged %d SWE-bench task(s) into %s (via symlinks to %d harbor cache dir(s))",
            staged, staging,
            len({p.parent for p in result.paths}),
        )

    def _extract_task_config(self, data: dict) -> dict:
        """Reuse parent extraction, then handle SWE-bench's memory string.

        SWE-bench task.toml uses ``memory = '4G'`` (string) in [environment].
        TB2 uses ``memory_mb = 4096`` (int). Either is accepted; explicit
        ``memory_mb`` always wins if both are present. Works on the dict the
        parent's ``_parse_task_toml`` already parsed — the file is read once.
        """
        cfg = super()._extract_task_config(data)
        env = data.get("environment", {})
        if "memory" in env and "memory_mb" not in env:
            cfg["memory_mb"] = _parse_memory_string(env["memory"], cfg["memory_mb"])
        return cfg

    def load_tasks(self, limit: int | None = None, seed_shuffle: int | None = None):
        """Load tasks via parent, then augment metadata from tests/config.json."""
        tasks = super().load_tasks(limit=limit, seed_shuffle=seed_shuffle)
        for task in tasks:
            # Tell Ω about this benchmark's runtime constraints. Read by
            # the generic ``omega_env_notes`` lookup in OmegaEngine._build_prompt;
            # no core changes needed when other adapters add their own.
            task.metadata["omega_env_notes"] = _OMEGA_ENV_NOTES

            task_dir = Path(task.metadata["task_dir"])
            cfg_path = task_dir / "tests" / "config.json"
            if not cfg_path.exists():
                continue
            try:
                data = json.loads(cfg_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "Failed to read SWE-bench config.json for %s: %s",
                    task.task_id, e,
                )
                continue
            task.metadata.update({
                "instance_id": data.get("instance_id"),
                "repo": data.get("repo"),
                "base_commit": data.get("base_commit"),
                "version": data.get("version"),
                "fail_to_pass_count": len(data.get("FAIL_TO_PASS") or []),
                "pass_to_pass_count": len(data.get("PASS_TO_PASS") or []),
            })
        return tasks
