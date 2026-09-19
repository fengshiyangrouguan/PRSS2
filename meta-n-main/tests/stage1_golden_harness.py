"""Stage 1 golden-capture harness (HEAD-state baseline for the OH floor-raisers).

Stage 1 adds two ORTHOGONAL, flag-gated, default-OFF base-agent affordances to
the *builtin* AgenticSolver (`docs/openhands_orthogonal_adoption_plan.md`):

  * R1 — an error-hint taxonomy in the rendered observation, behind
    ``--agentic-error-hints`` (AgenticSolver._build_observation /
    OBSERVATION_TEMPLATE).
  * R2 — a behavioral preamble in AGENTIC_SYSTEM_PROMPT, behind
    ``--agentic-preamble`` (AgenticSolver._build_system_message).

The KEY safety property: with BOTH flags OFF the rendered observation and the
system prompt must be **byte-identical** to current HEAD. This harness captures
that flag-OFF baseline NOW (before any feature edit) so the post-edit gate can
prove byte-identity.

Unlike the Stage 0 harness, this one is fully SELF-CONTAINED: it renders a small
fixed set of synthetic ``Trace`` fixtures through the REAL AgenticSolver methods
(``_build_observation`` / ``_build_system_message``) plus the literal prompt
constants. No experiments corpus, no LLM, no Docker, no network — the solver is
built with ``llm_client=None`` / ``executor=None`` because the two render methods
never touch either (they only read ``self.max_turns`` / ``self.solver_language``
and call pure formatting helpers).

Forward use: each failure fixture is keyed and its ``classify_error`` result is
recorded in the manifest, so the SAME fixtures (a) prove the flag-OFF byte
identity and (b) become the basis for the R1 hint test once the flag exists.

Usage:
    python -m tests.stage1_golden_harness capture   # write HEAD goldens
    python -m tests.stage1_golden_harness verify     # recompute + diff
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from meta_n.core.agentic_prompts import AGENTIC_SYSTEM_PROMPT, OBSERVATION_TEMPLATE
from meta_n.core.agentic_solver import AgenticSolver
from meta_n.core.meta_layer import TaskDescription, Trace, classify_error

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "stage1_golden"


# --------------------------------------------------------------------------- #
# Solver factory (flag-OFF / HEAD-equivalent)
# --------------------------------------------------------------------------- #
def _solver(*, language: str = "python", max_turns: int = 5) -> AgenticSolver:
    """A no-LLM/no-Docker AgenticSolver whose render methods are pure.

    Built with ``llm_client=None`` / ``executor=None`` and no injected codes —
    ``_build_observation`` and ``_build_system_message`` reference neither. When
    the R1/R2 flags exist they MUST default OFF, so constructing the solver
    without passing them reproduces today's bytes exactly.
    """
    return AgenticSolver(
        llm_client=None,
        executor=None,
        injected_codes=None,
        solver_language=language,
        max_turns=max_turns,
    )


# --------------------------------------------------------------------------- #
# Fixtures — a small fixed corpus exercising every render branch + every
# classify_error class. (key, Trace, turn).
# --------------------------------------------------------------------------- #
def _trace_fixtures() -> list[tuple[str, Trace, int]]:
    return [
        # --- happy / structural branches of _build_observation -------------
        (
            "success_full",
            Trace(
                task_id="t_success",
                score=1.0,
                exit_code=0,
                success=True,
                stdout="All tests passed\nDone",
                stderr="",
                duration_s=2.5,
            ),
            1,
        ),
        (
            "empty_io",  # -> "Unknown error" (empty text, non-actionable)
            Trace(
                task_id="t_empty",
                score=0.0,
                exit_code=1,
                success=False,
                stdout="",
                stderr="",
                duration_s=0.3,
            ),
            2,
        ),
        (
            "with_eval_feedback",
            Trace(
                task_id="t_eval",
                score=0.4,
                exit_code=0,
                success=False,
                stdout="partial output",
                stderr="",
                eval_feedback="case_3 mismatch: expected A got B",
                duration_s=1.1,
            ),
            2,
        ),
        (
            "long_output",  # exercise _head_tail truncation deterministically
            Trace(
                task_id="t_long",
                score=0.2,
                exit_code=1,
                success=False,
                stdout="\n".join(f"line {i:04d}" for i in range(400)),
                stderr="\n".join(f"err {i:04d}" for i in range(200)),
                duration_s=9.9,
            ),
            3,
        ),
        # --- one fixture per classify_error class --------------------------
        (
            "numeric_instability",
            Trace(
                task_id="t_numeric",
                score=0.0,
                exit_code=1,
                success=False,
                stdout="",
                stderr="ZeroDivisionError: division by zero",
                duration_s=0.5,
            ),
            2,
        ),
        (
            "dependency_error",
            Trace(
                task_id="t_dep",
                score=0.0,
                exit_code=1,
                success=False,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'numpy'",
                duration_s=0.4,
            ),
            2,
        ),
        (
            "timeout",
            Trace(
                task_id="t_timeout",
                score=0.0,
                exit_code=124,
                success=False,
                stdout="",
                stderr="Process timed out after 60s",
                duration_s=60.0,
            ),
            2,
        ),
        (
            "constraint_violation",
            Trace(
                task_id="t_constraint",
                score=0.0,
                exit_code=1,
                success=False,
                stdout="",
                stderr="constraint violated: capacity exceeded on bin 4",
                duration_s=0.7,
            ),
            2,
        ),
        (
            "indexing_error",
            Trace(
                task_id="t_index",
                score=0.0,
                exit_code=1,
                success=False,
                stdout="",
                stderr="IndexError: list index out of range",
                duration_s=0.6,
            ),
            2,
        ),
        (
            "syntax_error",
            Trace(
                task_id="t_syntax",
                score=0.0,
                exit_code=1,
                success=False,
                stdout="",
                stderr="SyntaxError: unexpected EOF while parsing",
                duration_s=0.2,
            ),
            2,
        ),
        (
            "format_parse_error",
            Trace(
                task_id="t_format",
                score=0.0,
                exit_code=-1,
                success=False,
                stdout="",
                stderr="",
                error_summary="produced no executable code (could not parse)",
                duration_s=0.1,
            ),
            2,
        ),
        (
            "turn_starvation",  # non-actionable
            Trace(
                task_id="t_turns",
                score=0.3,
                exit_code=1,
                success=False,
                stdout="",
                stderr="",
                error_summary="max_turns reached without confirmation",
                terminated_by="max_turns",
                duration_s=12.0,
            ),
            5,
        ),
        (
            "environment_fault",  # non-actionable
            Trace(
                task_id="t_env",
                score=0.0,
                exit_code=1,
                success=False,
                stdout="",
                stderr="docker compose failed to start the container",
                terminated_by="env_error",
                duration_s=3.0,
            ),
            1,
        ),
        (
            "runtime_error",  # non-actionable (generic)
            Trace(
                task_id="t_runtime",
                score=0.0,
                exit_code=1,
                success=False,
                stdout="",
                stderr="RuntimeError: solver returned an invalid result",
                duration_s=0.8,
            ),
            2,
        ),
    ]


# Fixed (key, language, context, metadata) combos for _build_system_message.
def _system_message_fixtures() -> list[tuple[str, str, str, dict]]:
    desc = (
        "Implement solve(cases, labels, few_shot) that classifies each symptom "
        "description into exactly one disease label."
    )
    return [
        ("sysmsg_python_nocontext", "python", "", {}),
        (
            "sysmsg_python_context",
            "python",
            "Prior attempts failed on degenerate (empty) inputs.",
            {},
        ),
        ("sysmsg_bash_nocontext", "bash", "", {}),
        ("sysmsg_classify_nocontext", "classify", "", {}),
        (
            "sysmsg_swebench_bash",
            "bash",
            "",
            {"benchmark": "swe_bench_verified"},
        ),
    ]


_SYS_DESC = (
    "Implement solve(cases, labels, few_shot) that classifies each symptom "
    "description into exactly one disease label."
)


# --------------------------------------------------------------------------- #
# Golden computation (single source of truth for capture AND verify)
# --------------------------------------------------------------------------- #
# Each golden is ("json"|"text", value). "json" -> sort_keys serialization;
# "text" -> stored verbatim.
def compute_goldens() -> dict[str, tuple[str, object]]:
    goldens: dict[str, tuple[str, object]] = {}

    # (1) Literal prompt constants — the rawest byte-identity anchor.
    goldens["AGENTIC_SYSTEM_PROMPT.txt"] = ("text", AGENTIC_SYSTEM_PROMPT)
    goldens["OBSERVATION_TEMPLATE.txt"] = ("text", OBSERVATION_TEMPLATE)

    # (2) Rendered observations over the fixed trace corpus (R1 surface).
    obs_solver = _solver(language="python", max_turns=5)
    for key, trace, turn in _trace_fixtures():
        goldens[f"observation__{key}.txt"] = (
            "text",
            obs_solver._build_observation(trace, turn),
        )

    # (3) Rendered system messages over the fixed config corpus (R2 surface).
    for key, language, context, metadata in _system_message_fixtures():
        solver = _solver(language=language, max_turns=5)
        task = TaskDescription(
            task_id=f"task_{key}",
            description=_SYS_DESC,
            metadata=metadata,
        )
        goldens[f"{key}.txt"] = (
            "text",
            solver._build_system_message(task, context),
        )

    # (4) classify_error map over the fixtures (validates ERROR_HINTS keys).
    goldens["classify_error_map.json"] = (
        "json",
        {key: classify_error(trace) for key, trace, _ in _trace_fixtures()},
    )

    return goldens


def _serialize(kind: str, value: object) -> str:
    if kind == "json":
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return value  # text golden stored verbatim


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "(unavailable)"


def capture() -> dict:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    goldens = compute_goldens()

    sha = {}
    for name, (kind, value) in goldens.items():
        blob = _serialize(kind, value)
        (GOLDEN_DIR / name).write_text(blob)
        sha[name] = hashlib.sha256(blob.encode()).hexdigest()

    classify_map = {key: classify_error(t) for key, t, _ in _trace_fixtures()}
    manifest = {
        "harness": "tests/stage1_golden_harness.py",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "trace_fixtures": [k for k, _, _ in _trace_fixtures()],
        "system_message_fixtures": [k for k, _, _, _ in _system_message_fixtures()],
        "classify_error_map": classify_map,
        "classify_error_classes": sorted(set(classify_map.values())),
        "golden_sha256": sha,
    }
    (GOLDEN_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


# --------------------------------------------------------------------------- #
# Verify (used by the gate tests AND the CLI)
# --------------------------------------------------------------------------- #
def verify() -> list[str]:
    """Recompute goldens and return human-readable byte-mismatch strings."""
    mismatches: list[str] = []
    goldens = compute_goldens()
    for name, (kind, value) in goldens.items():
        path = GOLDEN_DIR / name
        if not path.exists():
            mismatches.append(f"{name}: golden missing (run capture)")
            continue
        if _serialize(kind, value) != path.read_text():
            mismatches.append(f"{name}: BYTE MISMATCH vs HEAD golden")
    return mismatches


def _main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "verify"
    if cmd == "capture":
        m = capture()
        print(json.dumps(m, indent=2, sort_keys=True))
        return 0
    if cmd == "verify":
        bad = verify()
        if bad:
            print("MISMATCHES:")
            for b in bad:
                print(" -", b)
            return 1
        print("OK: all Stage 1 goldens byte-identical to HEAD baseline")
        return 0
    print(f"unknown command {cmd!r} (use capture|verify)")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
