"""TerminalBench 2.0 adapter — Docker-based terminal agent benchmark.

Manages Docker containers directly (Option B): builds images once per task,
spins up fresh containers per evaluation, and tears them down without
removing the cached image.

Data download:
    # Option A: pre-download via harbor CLI
    harbor run -d terminal-bench/terminal-bench-2 --agent oracle --n-concurrent 1

    # Option B: point at a local directory
    python -m meta_n.main --benchmark terminal_bench --bench-data-dir ./data/terminal_bench

Usage:
    adapter = TerminalBenchAdapter(task_cache_dir="./data/terminal_bench")
    await adapter.download()   # no-op if task_cache_dir exists
    tasks = adapter.load_tasks(limit=5)
    executor = TerminalBenchExecutor(adapter)
"""

from __future__ import annotations

# ``tempfile`` / ``tomllib`` are deliberate module attributes of this
# package: tests patch ``meta_n.integrations.terminal_bench.tempfile.
# mkdtemp`` and read/patch ``...terminal_bench.tomllib.load`` — both
# resolve THROUGH this package to the shared stdlib module object, so the
# submodules (which import the same objects) see the patch too.
import tempfile  # noqa: F401  (re-export, see comment above)

try:
    import tomllib  # noqa: F401  (re-export, see comment above)
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]  # noqa: F401

from meta_n.integrations.terminal_bench.compose import (
    _COMPOSE_BASE,
    _COMPOSE_BUILD,
    _COMPOSE_NO_NETWORK,
    _COMPOSE_PREBUILT,
    _compose_exec,
    _ExecResult,
    _purge_verifier_dir,
    _read_reward,
    _run_compose_command,
    _sanitize_compose_name,
    _sanitize_image_name,
)
from meta_n.integrations.terminal_bench.adapter import (
    _DEFAULT_T2_RUNNER_DIR,
    _DEFAULT_T2_TASKS_DIR,
    _DEFAULT_T2_VENV_PYTHON,
    _LITELLM_PROVIDER_PREFIXES,
    _litellm_route_model,
    _REPO_ROOT,
    _resolve_t2_paths,
    TerminalBenchAdapter,
)
from meta_n.integrations.terminal_bench.executor import TerminalBenchExecutor
from meta_n.integrations.terminal_bench.spine import (
    _TBTerminus2Env,
    TBExternalEnvProvider,
    TBExternalScorer,
    TBTerminus2EnvProvider,
    TBTerminus2Scorer,
)

__all__ = [
    "TBExternalEnvProvider",
    "TBExternalScorer",
    "TBTerminus2EnvProvider",
    "TBTerminus2Scorer",
    "TerminalBenchAdapter",
    "TerminalBenchExecutor",
    "_COMPOSE_BASE",
    "_COMPOSE_BUILD",
    "_COMPOSE_NO_NETWORK",
    "_COMPOSE_PREBUILT",
    "_DEFAULT_T2_RUNNER_DIR",
    "_DEFAULT_T2_TASKS_DIR",
    "_DEFAULT_T2_VENV_PYTHON",
    "_ExecResult",
    "_LITELLM_PROVIDER_PREFIXES",
    "_REPO_ROOT",
    "_TBTerminus2Env",
    "_compose_exec",
    "_litellm_route_model",
    "_purge_verifier_dir",
    "_read_reward",
    "_resolve_t2_paths",
    "_run_compose_command",
    "_sanitize_compose_name",
    "_sanitize_image_name",
]
