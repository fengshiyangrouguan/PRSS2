"""TerminalBenchAdapter — task download/loading, image cache, spine hooks.

Also home to the Terminus 2 subprocess-bridge path resolution and the
litellm provider-prefix routing helpers the adapter's backend factories
use. Split verbatim out of the old single-file
``meta_n/integrations/terminal_bench.py`` module.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
import weakref
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from meta_n.core.meta_layer import TaskDescription
from meta_n.integrations.benchmark import BenchmarkAdapter, EvalResult
from meta_n.integrations.terminal_bench.compose import (
    _COMPOSE_BASE,
    _COMPOSE_BUILD,
    _COMPOSE_NO_NETWORK,
    _COMPOSE_PREBUILT,
    _sanitize_compose_name,
    _sanitize_image_name,
)
from meta_n.integrations.terminal_bench.spine import (
    TBExternalEnvProvider,
    TBExternalScorer,
)

# Patch-target indirection: the compose helpers were module globals of the old
# single-file ``meta_n.integrations.terminal_bench`` module, and existing tests
# patch them ON THE PACKAGE (e.g. ``mock.patch("meta_n.integrations.
# terminal_bench._run_compose_command")`` / ``monkeypatch.setattr(tb, ...)``).
# Cross-function calls therefore resolve through the package namespace at call
# time (exactly like the old module-global lookup), never through a local
# binding a package-level patch could not see.
from meta_n.integrations import terminal_bench as _tb_pkg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# External-agent (Terminus 2) subprocess-bridge path resolution.
#
# meta-n must NEVER import ``terminal_bench``; the Terminus 2 agent runs entirely
# in a child process under the dedicated external-agents interpreter, driven by
# ``scripts/t2_runner.py``. These helpers resolve the three paths that child
# needs (interpreter, runner dir, tasks dir). All are env-overridable so a run
# can point at a non-default checkout without code changes.
# ---------------------------------------------------------------------------

#: Repo root (``.../meta-n``): this file lives at
#: ``meta_n/integrations/terminal_bench/``.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: External-agents interpreter (the only env with ``terminal_bench`` installed).
_DEFAULT_T2_VENV_PYTHON = str(_REPO_ROOT / ".venv_external_agents" / "bin" / "python")

#: Directory containing ``t2_runner.py`` (added to the child ``PYTHONPATH``).
_DEFAULT_T2_RUNNER_DIR = str(_REPO_ROOT / "scripts")

#: Directory holding the terminal-bench task folders the runner's ``TrialHandler``
#: reads (``<tasks_dir>/<task_id>/task.yaml``).
_DEFAULT_T2_TASKS_DIR = str(
    _REPO_ROOT / "baselines" / "terminal-bench" / "original-tasks"
)


#: Provider prefixes litellm already understands as routing tokens. When a model
#: id begins with one of these (``<prefix>/...``) it is left untouched; anything
#: else is treated as a bare model name and routed through ``openai/`` so a custom
#: OpenAI-compatible ``api_base`` (LM Studio, vLLM, …) is actually used. Kept small
#: and explicit on purpose — meta-n's own pricing keys whose leading segment is NOT
#: a litellm provider (e.g. ``google/gemma-...``) would be wrongly skipped by a
#: naive "has a slash → already routed" test. CAVEAT: a pricing key whose head IS
#: a litellm provider (e.g. ``anthropic/claude-...``) is deliberately left
#: unrouted here and would go to litellm's NATIVE provider — which the TB bridge
#: child cannot authenticate (only ``OPENAI_API_KEY`` reaches the runner env) —
#: so at this seam such ids must arrive pre-prefixed (``openai/anthropic/...`` or
#: ``openrouter/...``). Dropping ``anthropic`` from the set requires a lockstep
#: change in ``_runner_common.KNOWN_LITELLM_PROVIDERS`` (F118 parity gate).
#: Kept as the UNION of this set and oh_runner's ``_KNOWN_LITELLM_PROVIDERS`` so
#: OH and T2 route a given model id identically (audit reuse-simplify):
#: ``text-completion-openai`` is included here (it was OH-only) and ``cohere``
#: stays (it was T2-only).
_LITELLM_PROVIDER_PREFIXES: frozenset[str] = frozenset({
    "openai", "azure", "anthropic", "bedrock", "vertex_ai", "gemini",
    "openrouter", "ollama", "together_ai", "groq", "mistral", "cohere",
    "deepseek", "fireworks_ai", "xai", "hosted_vllm", "lm_studio",
    "text-completion-openai",
})


def _litellm_route_model(model: str) -> str:
    """Return ``model`` with an explicit litellm provider prefix for routing.

    Terminus 2's inner LLM goes through litellm, which only honours a custom
    ``api_base`` when the model carries an explicit provider prefix. meta-n's
    outer ``--model`` is the un-prefixed cost-ledger / pricing key (e.g.
    ``google/gemma-4-31b-qat``), so without this the request would be misrouted
    (litellm reads ``google/`` as Vertex, not the local endpoint). We prepend
    ``openai/`` (the OpenAI-compatible provider) unless the model already begins
    with a recognised litellm provider prefix — including a pricing key whose
    head collides with one (``anthropic/claude-...``), which is passed through
    unrouted; see the ``_LITELLM_PROVIDER_PREFIXES`` caveat. Idempotent and
    prefix-safe.

    Args:
        model: The model id as meta-n's config / cost ledger knows it.

    Returns:
        A litellm-routable model id. ``""`` stays ``""`` (the backend defaults).
    """
    if not model:
        return model
    head = model.split("/", 1)[0]
    if head in _LITELLM_PROVIDER_PREFIXES:
        return model
    return f"openai/{model}"


def _resolve_t2_paths(tasks_dir_hint: str | None = None) -> dict[str, str]:
    """Resolve the (venv_python, runner_dir, tasks_dir) for the T2 bridge.

    Each is taken from an env var if set, else the repo-relative default. The
    ``tasks_dir`` additionally honors an adapter-supplied hint (the adapter's task
    cache dir) when no env override is present and the hint exists.

    Args:
        tasks_dir_hint: Optional adapter task-cache dir to prefer for tasks_dir.

    Returns:
        ``{"venv_python", "runner_dir", "tasks_dir"}`` of absolute path strings.
    """
    tasks_dir = os.environ.get("T2_TASKS_DIR")
    if not tasks_dir:
        if tasks_dir_hint and Path(tasks_dir_hint).exists():
            tasks_dir = str(Path(tasks_dir_hint).resolve())
        else:
            tasks_dir = _DEFAULT_T2_TASKS_DIR
    return {
        "venv_python": os.environ.get("T2_VENV_PYTHON", _DEFAULT_T2_VENV_PYTHON),
        "runner_dir": os.environ.get("T2_RUNNER_DIR", _DEFAULT_T2_RUNNER_DIR),
        "tasks_dir": tasks_dir,
    }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

#: Sentinel: the caller did not declare the run's base solver. The load-time
#: legacy-layout guard stays off then — spine runs load the same legacy tasks
#: through this adapter, so only an EXPLICIT native declaration may fail fast.
_BASE_SOLVER_UNDECLARED = object()


class TerminalBenchAdapter(BenchmarkAdapter):
    """Adapter for TerminalBench 2.0 benchmark.

    Tasks are loaded from a local directory (pre-downloaded or via Harbor).
    Docker image builds and container lifecycle are managed by the paired
    TerminalBenchExecutor.
    """

    def __init__(
        self,
        task_cache_dir: str | None = None,
        task_names: list[str] | None = None,
        dataset_name: str = "terminal-bench/terminal-bench-2",
        base_solver: "str | None | object" = _BASE_SOLVER_UNDECLARED,
    ):
        self._task_cache_dir = Path(task_cache_dir) if task_cache_dir else None
        self._task_names_filter = task_names
        self._dataset_name = dataset_name
        # The run's ``--base-solver`` when the caller declares it (``None`` IS a
        # declaration: the native Layer1Solver default). Lets ``load_tasks`` fail
        # PRE-SPEND — before any solver LLM call — when a native run is pointed
        # at the legacy ``task.yaml`` layout the native executor cannot run.
        self._base_solver = base_solver

        # Populated by load_tasks()
        self._task_dirs: dict[str, Path] = {}  # task_id -> task directory
        self._task_configs: dict[str, dict] = {}  # task_id -> parsed task.toml

        # Compose template files (written once). Force mode 0o700 explicitly
        # so umask can't widen permissions on shared/CI hosts.
        self._compose_dir = Path(tempfile.mkdtemp(prefix="tb2_compose_"))
        os.chmod(self._compose_dir, 0o700)
        self._write_compose_templates()

        # cleanup() is only called explicitly by tests, so on every real run the
        # tb2_compose_* tempdir would otherwise leak. Register a finalizer that
        # removes it when this adapter is garbage-collected or at interpreter
        # exit. The callback must NOT capture ``self`` (it would keep the adapter
        # alive forever and never fire); it closes over the path only.
        self._finalizer = weakref.finalize(
            self, shutil.rmtree, self._compose_dir, ignore_errors=True
        )

        # Docker image cache
        self._image_built: set[str] = set()
        self._build_locks: dict[str, asyncio.Lock] = {}

    @property
    def name(self) -> str:
        return "terminal_bench"

    # ------------------------------------------------------------------ #
    # External-agent factory hooks (plan §2.3, §5).
    #
    # BOTH ``terminus2`` AND ``openhands`` are wired as SUBPROCESS BRIDGES: the
    # external agent runs entirely in a child process under the external-agents
    # interpreter (driven by ``scripts/t2_runner.py`` for Terminus 2 /
    # ``scripts/oh_tb_runner.py`` for OpenHands), which imports the
    # ``terminal_bench`` SDK and owns the FULL env + score lifecycle (container
    # build, tmux session, the verifier, and the binary ``is_resolved`` reward).
    # meta-n NEVER imports ``terminal_bench`` — so the env provider / scorer here
    # are thin shims SHARED across both backends: the provider only carries the
    # ``task_id`` + ``staged_files`` + ``run_label`` the backend forwards into the
    # request JSON, and the scorer reads the already-computed binary reward back
    # off the ``AgentRunResult`` (it reads ``native_resolved`` / ``native_score``,
    # which both backends populate identically from ``is_resolved``).
    #
    # ``builtin`` IS supported here too — but as a SAME-CONTAINER CONTROL, not an
    # external agent. On terminal_bench the native ``TerminalBenchExecutor`` cannot
    # run the legacy ``original-tasks`` layout (it needs the harbor-derived
    # ``environment/`` + ``task.toml`` + ``test.sh`` shape), so a real builtin
    # control MUST run through the SAME terminal-bench harness the OH / T2 runners
    # use. ``make_agent_backend("builtin")`` therefore returns a
    # :class:`~meta_n.core.external_agents.backends.builtin_tb.BuiltinTBBackend`
    # (a subprocess bridge to ``scripts/builtin_tb_runner.py``): meta-n's native
    # ``Layer1Solver`` authors a one-shot bash script (ONE outer LLM call,
    # meta-n-side), and that script runs in the harness-provisioned TB container +
    # verifier. The env provider + scorer are the SAME backend-agnostic shims OH/T2
    # use (the scorer reads the verifier's ``native_resolved`` / ``native_score``).
    #
    # This adapter advertising ``builtin`` on the spine (``advertises_spine_builtin``
    # → True) is the signal the orchestrator uses to route ``base_solver="builtin"``
    # through the spine ONLY on terminal_bench (every OTHER benchmark inherits the
    # base ``False``, so ``builtin`` keeps the legacy native dispatch there —
    # CO-Bench MUST NOT regress).
    # ------------------------------------------------------------------ #

    #: External-AGENT base_solver kinds (subprocess bridges driving an agent with
    #: its OWN inner LLM inside the runner). ``builtin`` is handled separately — it
    #: is a same-container CONTROL (the script is authored meta-n-side, NO inner LLM
    #: in the runner), so it is NOT in this set even though the same factories build
    #: its backend / env provider / scorer.
    _EXTERNAL_BACKEND_KINDS: frozenset[str] = frozenset({"terminus2", "openhands"})

    #: Every base_solver kind these factories build — the external agents PLUS the
    #: same-container ``builtin`` control. The env provider + scorer are shared
    #: across all three (backend-agnostic); only ``make_agent_backend`` branches.
    _SUPPORTED_BACKEND_KINDS: frozenset[str] = frozenset(
        {"terminus2", "openhands", "builtin"}
    )

    def advertises_spine_builtin(self) -> bool:
        """Route ``base_solver="builtin"`` through the spine as a same-container control.

        Overrides the base ``False``: only terminal_bench runs ``builtin`` on the
        spine, because only here does the native ``TerminalBenchExecutor`` fail to
        run the legacy ``original-tasks`` layout. Every other adapter (e.g.
        CO-Bench) inherits ``False`` so ``builtin`` stays on the legacy native
        ``Layer1Solver`` dispatch there and is NOT regressed. The orchestrator
        consults this predicate (not ``make_agent_backend``, which would need a
        constructed backend) to pick the route; this adapter therefore also makes
        ``make_agent_backend('builtin')`` / ``make_env_provider('builtin')`` /
        ``make_scorer('builtin')`` non-``None`` (the spine-builtin contract).
        """
        return True

    def _require_supported_kind(self, kind: str) -> None:
        """Raise ValueError if ``kind`` is not a supported base_solver kind."""
        if kind not in self._SUPPORTED_BACKEND_KINDS:
            raise ValueError(
                "TerminalBench base_solver supports only "
                f"{sorted(self._SUPPORTED_BACKEND_KINDS)} (got {kind!r})."
            )

    def make_agent_backend(self, kind: str, **kw):  # -> AgentBackend | None
        """Build the requested subprocess-bridge backend.

        Dispatches ``terminus2`` →
        :class:`~meta_n.core.external_agents.backends.terminus2.Terminus2Backend`,
        ``openhands`` →
        :class:`~meta_n.core.external_agents.backends.openhands_tb.OpenHandsTBBackend`,
        and ``builtin`` →
        :class:`~meta_n.core.external_agents.backends.builtin_tb.BuiltinTBBackend`
        (the same-container control — the script is authored meta-n-side via the
        passed ``solver`` + ``llm_client``, NO inner LLM in the runner). All three
        shell out to a runner under ``.venv_external_agents`` (the only env with the
        SDKs); meta-n never imports ``terminal_bench`` / ``openhands``.

        Args:
            kind: Backend identifier; ``"terminus2"`` / ``"openhands"`` /
                ``"builtin"``.
            **kw: The orchestrator's inner-backend kwargs
                (``model`` / ``api_base`` / ``api_key`` / ``provider_env_var``)
                forwarded as the inner-LLM routing for the AGENT kinds, PLUS the
                builtin-only authoring collaborators (``solver`` / ``llm_client`` /
                ``solver_language``) the orchestrator passes for ``builtin``.

        Returns:
            A configured backend for the requested kind.

        Raises:
            ValueError: if ``kind`` is not a supported kind, or if ``builtin`` is
                requested without the authoring ``solver``.
        """
        self._require_supported_kind(kind)

        paths = _resolve_t2_paths(
            str(self._task_cache_dir) if self._task_cache_dir else None
        )
        api_base = kw.get("api_base") or "http://127.0.0.1:1234/v1"
        api_key = kw.get("api_key")
        provider_env_var = kw.get("provider_env_var")
        # litellm routing reconciliation: the inner agent routes the model
        # through litellm, which needs an EXPLICIT provider prefix to send a
        # request to a custom OpenAI-compatible ``api_base`` (a bare
        # ``google/gemma-4-31b-qat`` would be treated as a Vertex/Gemini model
        # and never reach the local endpoint). meta-n's outer ``--model`` /
        # cost-ledger key is intentionally the UN-prefixed pricing key, so the
        # orchestrator hands us that form; we prepend ``openai/`` HERE (only for
        # the litellm-routed request) when no provider prefix is present. The
        # CostGuard keeps pricing on the un-prefixed key — the two never collide.
        model = _litellm_route_model(kw.get("model", "") or "")

        bridge_kwargs = dict(
            model=model,
            api_base=api_base,
            api_key=api_key,
            provider_env_var=provider_env_var,
            venv_python=paths["venv_python"],
            runner_dir=paths["runner_dir"],
            tasks_dir=paths["tasks_dir"],
        )

        if kind == "builtin":
            # Same-container CONTROL: the bash script is authored meta-n-side by the
            # passed native ``Layer1Solver`` (one outer LLM call through the outer
            # ``llm_client``); the runner runs NO LLM, only executes the script in
            # the harness-provisioned container + verifier. The bridge ``model`` /
            # ``api_base`` are inert routing metadata here (no inner LLM), kept for
            # the shared :class:`_ExternalTBBackend` construction signature.
            from meta_n.core.external_agents.backends.builtin_tb import (
                BuiltinTBBackend,
            )

            solver = kw.get("solver")
            if solver is None:
                raise ValueError(
                    "TerminalBench builtin base_solver requires an authoring "
                    "'solver' (the native Layer1Solver) in make_agent_backend "
                    "kwargs; the orchestrator passes it for the builtin kind."
                )
            return BuiltinTBBackend(
                solver=solver,
                llm_client=kw.get("llm_client"),
                solver_language=kw.get("solver_language", "bash"),
                **bridge_kwargs,
            )

        if kind == "openhands":
            # Lazy import: keeps the external_agents package import-free of
            # ``openhands`` (the backend itself shells out, never imports it).
            from meta_n.core.external_agents.backends.openhands_tb import (
                OpenHandsTBBackend,
            )

            return OpenHandsTBBackend(**bridge_kwargs)

        # Lazy import: keeps the external_agents package import-free of
        # ``terminal_bench`` (the backend itself shells out, never imports it).
        from meta_n.core.external_agents.backends.terminus2 import Terminus2Backend

        return Terminus2Backend(**bridge_kwargs)

    def make_env_provider(self, kind: str):  # -> AgentEnvProvider | None
        """Build the shared thin env provider (the runner owns the real env).

        The env provider is backend-agnostic: it carries only the
        ``task_id`` / ``staged_files`` / ``run_label`` (and the full ``task``, read
        only by the builtin backend to author its script) the backends fold into
        their request JSON; the real Docker environment, verifier and reward live
        inside the runner. ALL three kinds — ``terminus2`` / ``openhands`` /
        ``builtin`` — therefore get the SAME :class:`TBExternalEnvProvider`.

        Args:
            kind: Backend identifier; ``"terminus2"`` / ``"openhands"`` /
                ``"builtin"``.

        Returns:
            A :class:`TBExternalEnvProvider`.

        Raises:
            ValueError: if ``kind`` is not a supported kind.
        """
        self._require_supported_kind(kind)
        return TBExternalEnvProvider(self)

    def make_scorer(self, kind: str):  # -> Scorer | None
        """Build the shared scorer (reads the runner's binary verifier reward).

        The scorer is backend-agnostic: it derives success/score directly from
        the ``native_resolved`` / ``native_score`` the run carries, which ALL three
        backends populate identically from the verifier's binary ``is_resolved``
        (the builtin control runs the IDENTICAL verifier, so it scores the same
        way). ALL three kinds therefore get the SAME :class:`TBExternalScorer`.

        Args:
            kind: Backend identifier; ``"terminus2"`` / ``"openhands"`` /
                ``"builtin"``.

        Returns:
            A :class:`TBExternalScorer`.

        Raises:
            ValueError: if ``kind`` is not a supported kind.
        """
        self._require_supported_kind(kind)
        return TBExternalScorer(self)

    # --- Compose template management ---

    def _write_compose_templates(self):
        """Write compose YAML templates to the compose dir."""
        (self._compose_dir / "compose-base.yaml").write_text(_COMPOSE_BASE)
        (self._compose_dir / "compose-build.yaml").write_text(_COMPOSE_BUILD)
        (self._compose_dir / "compose-prebuilt.yaml").write_text(_COMPOSE_PREBUILT)
        (self._compose_dir / "compose-no-network.yaml").write_text(_COMPOSE_NO_NETWORK)

    def _get_compose_files(
        self, task_id: str, prebuilt: bool = True
    ) -> list[Path]:
        """Get ordered compose file list for a task.

        Order: base, build_or_prebuilt, [task-compose], [no-network]
        Matches Harbor's compose merging order.
        """
        build_or_prebuilt = (
            self._compose_dir / "compose-prebuilt.yaml"
            if prebuilt
            else self._compose_dir / "compose-build.yaml"
        )
        files = [
            self._compose_dir / "compose-base.yaml",
            build_or_prebuilt,
        ]

        # Include task's own docker-compose.yaml if present
        task_dir = self._task_dirs.get(task_id)
        if task_dir:
            task_compose = task_dir / "environment" / "docker-compose.yaml"
            if task_compose.exists():
                files.append(task_compose)

        # Disable network if task requires it
        config = self._task_configs.get(task_id, {})
        if not config.get("allow_internet", True):
            files.append(self._compose_dir / "compose-no-network.yaml")

        return files

    # --- Task download ---

    #: Human label for the cache short-circuit log line. SWE-bench overrides it
    #: (``"SWE-bench task directory"``) so both adapters' "Using existing ..."
    #: log lines keep their exact rendered bytes through the shared template.
    _cache_log_label: str = "task directory"

    async def download(self):
        """Download tasks from Harbor registry. No-op if task_cache_dir exists.

        Template method: the harbor preamble (cache short-circuit, lazy SDK
        import, fetch sequence) is shared; subclasses customize via
        :meth:`_filter_task_configs` (narrow the download set) and
        :meth:`_stage_downloaded` (map downloaded paths to
        ``_task_cache_dir``). The harbor import stays INSIDE this method so
        harbor remains optional and tests can monkeypatch ``sys.modules``.
        """
        if self._task_cache_dir and self._task_cache_dir.exists():
            n_dirs = sum(1 for d in self._task_cache_dir.iterdir() if d.is_dir())
            if n_dirs > 0:
                logger.info(
                    "Using existing %s: %s (%d tasks)",
                    self._cache_log_label, self._task_cache_dir, n_dirs,
                )
                return

        # Try Harbor SDK for download
        try:
            from harbor.models.job.config import DatasetConfig
            from harbor.tasks.client import TaskClient
        except ImportError as e:
            raise RuntimeError(
                "Harbor is not installed. Either install it (pip install harbor) "
                "or provide --bench-data-dir pointing to pre-downloaded tasks."
            ) from e

        logger.info("Downloading %s via Harbor...", self._dataset_name)
        dataset = DatasetConfig(name=self._dataset_name)
        task_configs = await dataset.get_task_configs()
        task_configs = self._filter_task_configs(task_configs)

        client = TaskClient()
        task_ids = [tc.get_task_id() for tc in task_configs]
        result = await client.download_tasks(task_ids)

        if not result.paths:
            raise RuntimeError("No tasks downloaded from Harbor registry")

        self._stage_downloaded(result)

    def _filter_task_configs(self, task_configs: list) -> list:
        """Hook: narrow the harbor task-config list before downloading.

        Identity by default (TerminalBench downloads the full dataset);
        SWE-bench overrides with an exact-match ``task_names`` filter.
        """
        return task_configs

    def _stage_downloaded(self, result) -> None:
        """Hook: derive ``_task_cache_dir`` from harbor's downloaded paths."""
        # Harbor stores tasks as packages/<org>/<task_name>/<hash>/
        # Set cache dir to the org level so load_tasks can find them
        first = result.paths[0]
        # Walk up from .../org/task_name/hash/ to .../org/
        self._task_cache_dir = first.parent.parent
        logger.info(
            "Downloaded %d tasks to %s",
            len(result.paths), self._task_cache_dir,
        )

    # --- Task loading ---

    def load_tasks(
        self,
        limit: int | None = None,
        seed_shuffle: int | None = None,
    ) -> list[TaskDescription]:
        """Load TerminalBench tasks from local cache directory.

        Args:
            limit: max number of tasks to return (after optional shuffle).
            seed_shuffle: if not None, deterministically shuffles the
                alphabetically-sorted task list with ``random.Random(seed)``
                BEFORE applying ``limit``. This matches DGM's
                ``task_terminal_bench.load_subsets`` exactly, so when both
                baselines run with the same ``--seed``, they evaluate on
                the SAME task subset — apples-to-apples per-task
                comparison. Without seed_shuffle, tasks are returned in
                alphabetical order (legacy behavior).
        """
        if not self._task_cache_dir or not self._task_cache_dir.exists():
            raise RuntimeError(
                "Task cache directory not set. Call download() first "
                "or provide --bench-data-dir."
            )

        # Resolve task directories. Supports two layouts:
        # 1. Flat: cache_dir/<task_name>/task.toml  (--bench-data-dir)
        # 2. Harbor package: cache_dir/<task_name>/<hash>/task.toml
        raw_dirs = sorted(d for d in self._task_cache_dir.iterdir() if d.is_dir())
        task_dirs: list[tuple[str, Path]] = []  # (task_name, resolved_dir)
        for d in raw_dirs:
            if (d / "task.toml").exists():
                task_dirs.append((d.name, d))
            else:
                # Harbor hash layout: look for a single subdirectory with task.toml
                subdirs = [s for s in d.iterdir() if s.is_dir() and (s / "task.toml").exists()]
                if subdirs:
                    task_dirs.append((d.name, subdirs[0]))

        # Layout 3 (LEGACY terminal-bench ``task.yaml``): the ``original-tasks``
        # baseline that the Terminus 2 subprocess runner loads is the pre-2.0
        # terminal-bench layout (``<task>/task.yaml`` + ``tests/`` + Dockerfile),
        # not Harbor's ``task.toml``/``instruction.md``. The harbor scan above
        # finds nothing there, so when it comes up empty fall back to the
        # legacy loader. This is the task set the ``--base-solver terminus2``
        # CLI route runs against (the runner owns the real env + verifier; this
        # only needs the folder name + instruction for the Trace/Ω surface).
        if not task_dirs:
            return self._load_legacy_yaml_tasks(
                raw_dirs, limit=limit, seed_shuffle=seed_shuffle,
            )

        # Seed-stable shuffle BEFORE filter/parse so the resulting subset
        # matches DGM's ``random.Random(seed).shuffle(sorted_names)[:N]``
        # algorithm (see baselines/dgm/src/task_terminal_bench.py).
        # task_dirs is already sorted alphabetically by the line above,
        # which is the same starting state DGM uses (sorted task names).
        if seed_shuffle is not None:
            import random as _random
            rng = _random.Random(seed_shuffle)
            rng.shuffle(task_dirs)

        tasks: list[TaskDescription] = []
        for task_name, task_dir in task_dirs:

            # Apply name filter
            if self._task_names_filter and task_name not in self._task_names_filter:
                continue

            # Validate minimum structure
            instruction_path = task_dir / "instruction.md"
            env_dir = task_dir / "environment"
            tests_dir = task_dir / "tests"
            if not instruction_path.exists() or not env_dir.exists():
                logger.warning("Skipping %s: missing instruction.md or environment/", task_name)
                continue
            if not tests_dir.exists():
                logger.warning("Skipping %s: missing tests/", task_name)
                continue

            # Parse task.toml — skip (don't abort the entire benchmark load) on
            # a single malformed/unreadable file. A corrupt task.toml among the
            # 113 (TB2) / 500 (SWE-bench) tasks otherwise raises TOMLDecodeError
            # straight out of load_tasks and crashes the whole run before any
            # task executes; defensively skipping that one task is the right
            # robustness posture (it also matches the structure-validation
            # ``continue``s just above).
            try:
                config = self._parse_task_toml(task_dir / "task.toml")
            except (tomllib.TOMLDecodeError, OSError) as exc:  # noqa: PERF203
                logger.warning("Skipping %s: bad task.toml (%s)", task_name, exc)
                continue
            instruction = instruction_path.read_text().strip()

            task_id = _sanitize_compose_name(task_name)

            self._task_dirs[task_id] = task_dir
            self._task_configs[task_id] = config

            task = TaskDescription(
                task_id=task_id,
                description=instruction,
                metadata={
                    "benchmark": self.name,
                    "task_name": task_name,
                    "task_dir": str(task_dir.resolve()),
                    "solution_language": "bash",
                    "cpus": config.get("cpus", 1),
                    "memory_mb": config.get("memory_mb", 2048),
                    "timeout_sec": config.get("timeout_sec", 1800),
                    "verifier_timeout_sec": config.get("verifier_timeout_sec", 900),
                    "build_timeout_sec": config.get("build_timeout_sec", 600),
                    "allow_internet": config.get("allow_internet", True),
                    "has_compose": (env_dir / "docker-compose.yaml").exists(),
                    "category": config.get("category", ""),
                    "difficulty": config.get("difficulty", ""),
                },
            )
            tasks.append(task)

            if limit is not None and len(tasks) >= limit:
                break

        logger.info(
            "Loaded %d TerminalBench tasks (limit=%s, seed_shuffle=%s)",
            len(tasks), limit, seed_shuffle,
        )
        return tasks

    def _load_legacy_yaml_tasks(
        self,
        raw_dirs: list[Path],
        limit: int | None = None,
        seed_shuffle: int | None = None,
    ) -> list[TaskDescription]:
        """Load pre-2.0 terminal-bench ``task.yaml`` tasks (the runner baseline).

        The ``baselines/terminal-bench/original-tasks`` set the Terminus 2
        subprocess runner reads uses the legacy layout::

            <task_name>/
              task.yaml         # instruction + difficulty/category/timeouts
              tests/            # verifier
              Dockerfile / docker-compose.yaml

        meta-n must NEVER drive that env itself (the runner's ``TrialHandler``
        owns container build, the tmux session and the verifier), so this loader
        is deliberately thin: it carries only the on-disk folder name
        (``task_name`` — the field the Terminus 2 env provider forwards to the
        runner) plus the instruction text and a few advisory timeouts for the
        Trace / Ω surface. It does not validate or parse the Docker env.

        Args:
            raw_dirs: The already-sorted immediate subdirectories of the cache
                dir (from :meth:`load_tasks`).
            limit: Max number of tasks to return (after the optional shuffle).
            seed_shuffle: If set, deterministically shuffle (matching the harbor
                path) before applying ``limit``.

        Returns:
            The loaded :class:`TaskDescription` list (possibly empty).
        """
        import yaml

        # (task_name, dir) for every legacy-layout task: a task.yaml + tests/.
        legacy: list[tuple[str, Path]] = []
        for d in raw_dirs:
            if (d / "task.yaml").exists() and (d / "tests").is_dir():
                legacy.append((d.name, d))

        if legacy and self._native_route_declared():
            raise RuntimeError(
                f"{self._task_cache_dir} holds legacy terminal-bench task.yaml "
                "tasks, which the native TerminalBenchExecutor cannot run (they "
                "have no environment/ + task.toml — every evaluation would fail "
                "at image build after paying the solver's LLM authoring call). "
                "Run this layout through the external-agent spine instead: "
                "--base-solver builtin, openhands, or terminus2."
            )

        if seed_shuffle is not None:
            import random as _random
            _random.Random(seed_shuffle).shuffle(legacy)

        tasks: list[TaskDescription] = []
        for task_name, task_dir in legacy:
            if self._task_names_filter and task_name not in self._task_names_filter:
                continue
            try:
                spec = yaml.safe_load((task_dir / "task.yaml").read_text()) or {}
            except (OSError, yaml.YAMLError) as exc:  # noqa: PERF203
                logger.warning("Skipping legacy task %s: bad task.yaml (%s)", task_name, exc)
                continue
            if not isinstance(spec, dict):
                logger.warning("Skipping legacy task %s: task.yaml is not a mapping", task_name)
                continue

            instruction = str(spec.get("instruction", "") or "").strip()
            task_id = _sanitize_compose_name(task_name)
            self._task_dirs[task_id] = task_dir
            # Map the legacy timeout keys onto the same config shape the harbor
            # path produces, so anything reading _task_configs stays uniform.
            config = {
                "timeout_sec": spec.get("max_agent_timeout_sec", 1800),
                "verifier_timeout_sec": spec.get("max_test_timeout_sec", 900),
                "category": spec.get("category", ""),
                "difficulty": spec.get("difficulty", ""),
                "allow_internet": True,
            }
            self._task_configs[task_id] = config

            tasks.append(TaskDescription(
                task_id=task_id,
                description=instruction or "(loaded from disk by the Terminus 2 runner)",
                metadata={
                    "benchmark": self.name,
                    # ``task_name`` is the on-disk folder name the env provider
                    # forwards to the runner as the runner_task_id — load-bearing.
                    "task_name": task_name,
                    "task_dir": str(task_dir.resolve()),
                    "solution_language": "bash",
                    "layout": "legacy_yaml",
                    "timeout_sec": config["timeout_sec"],
                    "verifier_timeout_sec": config["verifier_timeout_sec"],
                    # Declared verifier timeout, ``None`` when task.yaml declares
                    # none — the spine backends thread it into the runner's
                    # ``global_test_timeout_sec`` so a >bridge-default test suite
                    # is not clamped (the 900 fallback above is native-path only).
                    "max_test_timeout_sec": spec.get("max_test_timeout_sec"),
                    "category": config["category"],
                    "difficulty": config["difficulty"],
                    "has_compose": (task_dir / "docker-compose.yaml").exists(),
                },
            ))
            if limit is not None and len(tasks) >= limit:
                break

        logger.info(
            "Loaded %d legacy-yaml TerminalBench tasks (limit=%s, seed_shuffle=%s)",
            len(tasks), limit, seed_shuffle,
        )
        return tasks

    def _native_route_declared(self) -> bool:
        """Whether the caller declared a base solver that stays on the native path.

        ``False`` when ``base_solver`` was left undeclared (the route is unknown;
        spine callers load the same legacy tasks through this adapter, so the
        legacy-layout guard must not fire for them). Shares the spine-routing
        predicate with main.py / the orchestrator so the two never disagree.
        """
        if self._base_solver is _BASE_SOLVER_UNDECLARED:
            return False
        from meta_n.core.spine_routing import uses_external_spine

        return not uses_external_spine(self._base_solver, self)

    def _parse_task_toml(self, path: Path) -> dict:
        """Parse task.toml and extract the fields we need.

        Split into parse + :meth:`_extract_task_config` so subclasses that
        need extra fields (e.g. SWE-bench's ``memory`` string) extend the
        extraction WITHOUT re-reading/re-parsing the file.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return self._extract_task_config(data)

    def _extract_task_config(self, data: dict) -> dict:
        """Extract the config fields from an already-parsed task.toml dict."""
        env = data.get("environment", {})
        agent = data.get("agent", {})
        verifier = data.get("verifier", {})
        meta = data.get("metadata", {})

        return {
            "cpus": env.get("cpus", 1),
            "memory_mb": env.get("memory_mb", 2048),
            "allow_internet": env.get("allow_internet", True),
            "timeout_sec": agent.get("timeout_sec", 1800),
            "verifier_timeout_sec": verifier.get("timeout_sec", 900),
            "build_timeout_sec": env.get("build_timeout_sec", 600),
            "category": meta.get("category", ""),
            "difficulty": meta.get("difficulty", ""),
        }

    async def evaluate(self, task: TaskDescription, solution: str) -> EvalResult:
        """Evaluate a solution via the paired executor.

        Not used during the main loop (the executor is called directly),
        but provided for interface completeness and test-time evaluation.
        """
        if not hasattr(self, "_executor"):
            return EvalResult(
                success=False, score=0.0,
                feedback="No executor attached to adapter",
            )
        trace = await self._executor.execute(solution, task)
        return EvalResult(
            success=trace.success,
            score=trace.score,
            raw_score=trace.score,
            feedback=trace.eval_feedback,
        )

    # --- Docker image management ---

    @staticmethod
    def _image_tag(task_id: str) -> str:
        """Deterministic image tag for a task."""
        return _sanitize_image_name(f"tb2-{task_id}")

    async def _image_exists(self, image_tag: str) -> bool:
        """Check if a Docker image exists locally."""
        proc = await asyncio.create_subprocess_exec(
            "docker", "image", "inspect", image_tag,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def _ensure_image(self, task_id: str) -> None:
        """Build Docker image if not already cached. Per-task lock prevents duplicate builds."""
        if task_id in self._image_built:
            return

        lock = self._build_locks.setdefault(task_id, asyncio.Lock())
        async with lock:
            if task_id in self._image_built:
                return

            image_tag = self._image_tag(task_id)

            # Check Docker's image store (survives process restarts)
            if await self._image_exists(image_tag):
                logger.info("Image %s already cached, skipping build", image_tag)
                self._image_built.add(task_id)
                return

            task_dir = self._task_dirs[task_id]
            config = self._task_configs[task_id]

            logger.info("Building Docker image %s for %s...", image_tag, task_id)
            build_start = time.time()

            compose_files = self._get_compose_files(task_id, prebuilt=False)
            project_name = _sanitize_compose_name(f"tb2-build-{task_id}")

            env = self._compose_env(task_id, host_logs_path="/tmp/tb2_build_dummy")
            await _tb_pkg._run_compose_command(
                compose_files=compose_files,
                project_name=project_name,
                project_dir=task_dir / "environment",
                command=["build"],
                env_overrides=env,
                timeout_sec=config.get("build_timeout_sec", 600),
            )

            # Clean up the build project (no containers to keep)
            try:
                await _tb_pkg._run_compose_command(
                    compose_files=compose_files,
                    project_name=project_name,
                    project_dir=task_dir / "environment",
                    command=["down"],
                    env_overrides=env,
                )
            except Exception:
                pass  # Best-effort cleanup

            self._image_built.add(task_id)
            logger.info(
                "Built image %s in %.1fs", image_tag, time.time() - build_start
            )

    def _compose_env(self, task_id: str, host_logs_path: str) -> dict[str, str]:
        """Build environment variables for compose commands."""
        task_dir = self._task_dirs[task_id]
        config = self._task_configs[task_id]
        return {
            "CONTEXT_DIR": str((task_dir / "environment").resolve()),
            "IMAGE_NAME": self._image_tag(task_id),
            "HOST_LOGS_PATH": str(host_logs_path),
            "CPUS": str(config.get("cpus", 1)),
            "MEMORY": f"{config.get('memory_mb', 2048)}M",
        }

    def cleanup(self):
        """Clean up compose template directory.

        Idempotent: runs the registered finalizer (which ``shutil.rmtree``s the
        compose dir) once and detaches it, so a later GC/atexit pass is a no-op.
        """
        self._finalizer()
