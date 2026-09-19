"""Refinement-wave regression tests for meta_n/core/meta_layer.py + prompts.py.

Covers: F001 (classify_error structured-timeout precedence), F004 (deploy scan
gated on stageable helpers), F007 (bash library heading de-duplication), F014
(single strip-library-prefix source), F017 (execute/solve context-byte pins),
F022 (async helpers advertised), F026 (pre_process task isolation), F195
(detect_script_language — shared Ω-fence / trace-extension contract).
No LLM / Docker / network.
"""

import logging
import time

import pytest

from meta_n.core.base_executor import BaseExecutor
from meta_n.core.meta_layer import (
    InjectedCode,
    MetaLayer,
    SandboxMarker,
    TaskDescription,
    Trace,
    _strip_library_prefix_for_scan,
    classify_error,
    detect_script_language,
    format_bash_library_descriptions,
    format_python_library_descriptions,
    run_pre_process,
)

FROZEN_REASONING = "frozen Ω_merge winner (no re-solve)"
LIB = {"greet": 'def greet(name):\n    """Say hi."""\n    return name'}
PRE_EMIT_PP = "additional_context = 'PP'"
PRE_EMIT_OUTER = "additional_context = 'saw:' + outer_context"


class CapturingSolver:
    """Inner solver that records the exact additional_context it receives."""

    def __init__(self, script: str = "print('hi')"):
        self.script = script
        self.contexts: list[str] = []

    async def solve(self, task, additional_context: str = ""):
        self.contexts.append(additional_context)
        return self.script, "captured", 5


class CapturingExecutor(BaseExecutor):
    """Sandboxed executor that records the exec_script passed to execute()."""

    def __init__(self):
        self.exec_scripts: list[str] = []

    @property
    def is_sandboxed(self) -> bool:
        return True

    async def execute(self, script, task, timeout: int = 30) -> Trace:
        self.exec_scripts.append(script)
        return Trace(task_id=task.task_id, script=script, success=True, score=1.0)


class UnsandboxedCapturingExecutor(CapturingExecutor):
    """Capturing executor on the full-validation (non-sandboxed) branch."""

    @property
    def is_sandboxed(self) -> bool:
        return False


@pytest.fixture
def task():
    return TaskDescription(task_id="t", description="do the thing")


def _layer(*, pre: str | None = None, lib: bool = False, **kw):
    return MetaLayer(
        depth=2,
        injected_code=InjectedCode(pre_process=pre, source_depth=2),
        inner_solver=CapturingSolver(),
        executor=CapturingExecutor(),
        merged_code_library=dict(LIB) if lib else None,
        **kw,
    )


# --------------------------------------------------------------------------- #
# F017 — pins: exact inner-solver context bytes on BOTH paths (2x2 matrix).
# These pass on the pre-refactor code and MUST keep passing afterwards.
# --------------------------------------------------------------------------- #


MATRIX = [(False, False), (False, True), (True, False), (True, True)]


class TestPrepareContextPins:
    @pytest.mark.parametrize("pre,lib", MATRIX)
    async def test_execute_path_context(self, task, pre, lib):
        layer = _layer(pre=PRE_EMIT_PP if pre else None, lib=lib)
        lib_desc = layer._format_library_descriptions()
        await layer.execute(task)
        expected = "PP" if pre else ""
        if lib_desc:
            # execute() join: no leading newline when context is empty.
            expected = f"{expected}\n{lib_desc}" if expected else lib_desc
        assert layer.inner_solver.contexts == [expected]

    @pytest.mark.parametrize("pre,lib", MATRIX)
    async def test_solve_path_context_empty_incoming(self, task, pre, lib):
        layer = _layer(pre=PRE_EMIT_PP if pre else None, lib=lib)
        lib_desc = layer._format_library_descriptions()
        await layer.solve(task)
        expected = "PP" if pre else ""
        if lib_desc:
            # solve() join quirk: leading "\n" even when context is empty.
            expected = f"{expected}\n{lib_desc}"
        assert layer.inner_solver.contexts == [expected]

    @pytest.mark.parametrize("pre,lib", MATRIX)
    async def test_solve_path_context_with_incoming(self, task, pre, lib):
        layer = _layer(pre=PRE_EMIT_PP if pre else None, lib=lib)
        lib_desc = layer._format_library_descriptions()
        await layer.solve(task, "OUTER")
        expected = "OUTER\nPP" if pre else "OUTER"
        if lib_desc:
            expected = f"{expected}\n{lib_desc}"
        assert layer.inner_solver.contexts == [expected]

    async def test_solve_threads_outer_context_execute_does_not(self, task):
        solve_layer = _layer(pre=PRE_EMIT_OUTER)
        await solve_layer.solve(task, "OUTER")
        assert solve_layer.inner_solver.contexts == ["OUTER\nsaw:OUTER"]

        exec_layer = _layer(pre=PRE_EMIT_OUTER)
        await exec_layer.execute(task)
        # execute() never threads an outer context into pre_process.
        assert exec_layer.inner_solver.contexts == ["saw:"]

    async def test_no_outer_context_ablation_pin(self, task):
        layer = _layer(pre=PRE_EMIT_OUTER, no_outer_context=True)
        await layer.solve(task, "OUTER")
        # E3 ablation: pre_process sees outer_context="", the accumulated
        # additional_context still flows to the inner solver.
        assert layer.inner_solver.contexts == ["OUTER\nsaw:"]

    async def test_frozen_routing_solve_path(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(
                task_solution_map={"t": "FROZEN"}, source_depth=2
            ),
            inner_solver=CapturingSolver(),
            executor=CapturingExecutor(),
            merged_code_library=dict(LIB),
        )
        script, reasoning, tokens = await layer.solve(task)
        # Frozen winner returned VERBATIM: no inner solve, no prepend.
        assert (script, reasoning, tokens) == ("FROZEN", FROZEN_REASONING, 0)
        assert layer.inner_solver.contexts == []

    async def test_frozen_routing_execute_path(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(
                task_solution_map={"t": "FROZEN"}, source_depth=2
            ),
            inner_solver=CapturingSolver(),
            executor=CapturingExecutor(),
            merged_code_library=dict(LIB),
        )
        trace, tokens = await layer.execute(task)
        assert tokens == 0
        assert layer.executor.exec_scripts == ["FROZEN"]
        assert trace.reasoning == FROZEN_REASONING
        assert layer.inner_solver.contexts == []

    async def test_tail_is_shared_between_paths(self, task):
        solve_layer = _layer(lib=True)
        script, _, _ = await solve_layer.solve(task)
        assert script == solve_layer._prepend_library(solve_layer.inner_solver.script)

        exec_layer = _layer(lib=True)
        await exec_layer.execute(task)
        # execute() runs the same prepared (deploy + prepend) script.
        assert exec_layer.executor.exec_scripts == [script]


# --------------------------------------------------------------------------- #
# F001 — classify_error: structured terminated_by="timeout" wins even with
# empty error text (docstring contract).
# --------------------------------------------------------------------------- #


class TestClassifyErrorStructuredPrecedence:
    def test_tb_timeout_with_empty_text_is_timeout(self):
        t = Trace(task_id="t", terminated_by="timeout", error_summary="", stderr="")
        assert classify_error(t) == "Timeout"

    def test_empty_everything_stays_unknown(self):
        assert classify_error(Trace(task_id="t")) == "Unknown error"

    def test_tb_max_turns_with_empty_text_is_turn_starvation(self):
        t = Trace(task_id="t", terminated_by="max_turns")
        assert classify_error(t) == "Turn starvation"


# --------------------------------------------------------------------------- #
# F004 — deploy fallback: adoption scan runs over STAGEABLE helpers only, so a
# solve() calling only a validation-failing helper still gets the wrapper.
# --------------------------------------------------------------------------- #


BAD_HELPER = "import os\ndef bad_helper(**kw):\n    return {}\n"
GOOD_HELPER = "def good_helper(**kw):\n    return {}\n"


def _deploy_layer(authored: str, executor) -> MetaLayer:
    return MetaLayer(
        depth=2,
        injected_code=InjectedCode(),
        inner_solver=CapturingSolver(authored),
        executor=executor,
        merged_code_library={"bad_helper": BAD_HELPER, "good_helper": GOOD_HELPER},
        deploy_verified_code=True,
        solver_language="python",
    )


class TestDeployScansStageableOnly:
    async def test_call_to_unstageable_helper_still_deploys_wrapper(self, task):
        ex = UnsandboxedCapturingExecutor()
        authored = "def solve(**kw):\n    return bad_helper(**kw)\n"
        layer = _deploy_layer(authored, ex)
        await layer.execute(task)
        # bad_helper is never prepended (import os fails validation) — the
        # authored call would NameError, so the stageable helper is deployed.
        body = _strip_library_prefix_for_scan(ex.exec_scripts[0])
        assert body == "def solve(**kw):\n    return good_helper(**kw)\n"

    async def test_authored_call_to_stageable_helper_kept(self, task):
        ex = UnsandboxedCapturingExecutor()
        authored = "def solve(**kw):\n    return good_helper(**kw)\n"
        layer = _deploy_layer(authored, ex)
        await layer.execute(task)
        assert ex.exec_scripts[0] == layer._prepend_library(authored)


# --------------------------------------------------------------------------- #
# F007 — bash library rules get their own heading (no duplicated H2).
# --------------------------------------------------------------------------- #


BASH_FMT_LIB = {"recover": 'def recover(path):\n    """Recover data."""\n    return []'}
BASH_FMT_BASH = {"setup": "setup() { echo ok; }"}


def test_bash_library_description_has_single_section_heading():
    out = format_bash_library_descriptions(
        BASH_FMT_LIB, BASH_FMT_BASH, SandboxMarker()
    )
    assert out.count("## Available Helper Code") == 1
    assert "## Helper Code Rules" in out


def test_bash_library_description_foster_has_single_required_heading():
    out = format_bash_library_descriptions(
        BASH_FMT_LIB, BASH_FMT_BASH, SandboxMarker(), foster_adoption=True
    )
    assert out.count("## Available Helper Code") == 1
    assert "## Helper Code Rules (REQUIRED — call the helpers, do NOT re-implement them)" in out


# --------------------------------------------------------------------------- #
# F014 — one strip-library-prefix implementation feeds every surface.
# --------------------------------------------------------------------------- #


class TestStripLibraryPrefixSingleSource:
    MARKED = (
        "# --- injected code library ---\n\n# lib:f\ndef f(): pass\n\n"
        "# --- end injected code library ---\n\nprint('own')"
    )
    UNMARKED = "print('own')"

    def test_matches_omega_strip(self):
        from meta_n.core.omega import OmegaEngine

        for script in (self.MARKED, self.UNMARKED):
            assert OmegaEngine._strip_library_prefix(script) == (
                _strip_library_prefix_for_scan(script)
            )

    def test_helper_behavior(self):
        assert _strip_library_prefix_for_scan(self.MARKED) == "print('own')"
        assert _strip_library_prefix_for_scan(self.UNMARKED) == "print('own')"

    def test_debug_context_uses_helper(self, task):
        layer = _layer()
        failed = Trace(task_id="t", script=self.MARKED, score=0.0, stderr="boom")
        ctx = layer._build_debug_context("", failed, 1)
        assert "print('own')" in ctx
        assert "injected code library" not in ctx


# --------------------------------------------------------------------------- #
# F022 — async helpers are advertised on both formatter paths.
# --------------------------------------------------------------------------- #


ASYNC_SRC = 'async def fetch_all(urls):\n    """Fetch every URL."""\n    return []'


class TestAsyncHelpersAdvertised:
    def test_python_formatter_shows_async_signature(self):
        out = format_python_library_descriptions({"fetch_all": ASYNC_SRC}, SandboxMarker())
        assert "- async def fetch_all(urls):" in out
        assert "- fetch_all()" not in out

    def test_bash_formatter_shows_async_cli_form(self):
        out = format_bash_library_descriptions({"fetch_all": ASYNC_SRC}, {}, SandboxMarker())
        assert "python3 /tmp/_lib_fetch_all.py <args>" in out
        assert "Fetch every URL." in out

    def test_sync_helper_output_pinned(self):
        from meta_n.core.prompts import LIBRARY_INSTRUCTIONS

        src = 'def greet(name):\n    """Say hi."""\n    return name'
        out = format_python_library_descriptions({"greet": src}, SandboxMarker())
        expected = (
            "\n## Available Helper Functions "
            "(already implemented and available at runtime — just call them)\n"
            '- def greet(name):\n    """Say hi.\n    """'
            "\n" + LIBRARY_INSTRUCTIONS
        )
        assert out == expected


# --------------------------------------------------------------------------- #
# F026 — pre_process blocks exec against a defensive COPY of the task.
# --------------------------------------------------------------------------- #


class TestPreProcessTaskIsolation:
    def test_mutating_block_cannot_touch_live_task(self):
        live = TaskDescription(task_id="t", description="orig")
        ic = InjectedCode(
            pre_process="task.description = 'HACKED'\nadditional_context = 'x'",
            source_depth=2,
        )
        ran, ctx = run_pre_process([ic], live)
        assert ran is True and ctx == "x"
        assert live.description == "orig"

    def test_abandoned_block_cannot_touch_live_task(self):
        # The abandoned worker signals through a logger record AFTER mutating
        # (open()/pathlib are safety-blocked; logging is not), so the final
        # assertion is deterministic — we only check the live task once the
        # mutation provably happened, with no wall-clock coupling.
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        sentinel_logger = logging.getLogger("test_refine_sentinel")
        handler = _Capture()
        sentinel_logger.addHandler(handler)
        try:
            live = TaskDescription(
                task_id="t", description="d", metadata={"k": "v"}
            )
            ic = InjectedCode(
                pre_process=(
                    "import time\nimport logging\ntime.sleep(0.3)\n"
                    "task.metadata['k'] = 'HACKED'\n"
                    "logging.getLogger('test_refine_sentinel')"
                    ".warning('mutated')\n"
                    "additional_context = 'late'"
                ),
                source_depth=2,
            )
            ran, ctx = run_pre_process([ic], live, pre_process_timeout=0.05)
            assert ran is False and ctx == ""
            deadline = time.monotonic() + 10.0
            while not records and time.monotonic() < deadline:
                time.sleep(0.02)
            assert records, "abandoned worker never reached its mutation"
            assert live.metadata == {"k": "v"}
        finally:
            sentinel_logger.removeHandler(handler)


# --------------------------------------------------------------------------- #
# F195 — detect_script_language: single source for the Ω prompt fence language
# AND the persisted .py/.sh trace extension (frozen tuple contract).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "script, lang",
    [
        ("def f():\n    pass", "python"),
        ("import os", "python"),
        ("  from x import y", "python"),
        ("#!/bin/bash\necho hi", "bash"),
        # KNOWN misfile, frozen on purpose: the identical tuple drives the Ω
        # prompt fence language (golden-gated) — do not "fix" one side alone.
        ("# comment\nimport os", "bash"),
        ("", "bash"),
    ],
)
def test_detect_script_language(script, lang):
    assert detect_script_language(script) == lang


def test_detect_script_language_single_source():
    """Both consumers bind the meta_layer helper — no per-module copies."""
    from meta_n.core import evolutionary_orchestrator, omega

    assert evolutionary_orchestrator.detect_script_language is detect_script_language
    assert omega.detect_script_language is detect_script_language
