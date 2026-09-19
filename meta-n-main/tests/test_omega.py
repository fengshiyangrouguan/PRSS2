"""Tests for the Omega engine."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.llm_client import LLMClient, LLMConfig
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace
from meta_n.core.omega import OmegaEngine


SAMPLE_OMEGA_RESPONSE = """Here's my analysis of the failure patterns:

```rationale
Most failures are due to missing 'set -e' in scripts, causing silent errors.
Tasks involving file creation fail when parent directories don't exist.
```

```pre_process
# Add hints based on common failure patterns
if "file" in task.description.lower() and "directory" in task.description.lower():
    additional_context = "IMPORTANT: Create parent directories first using mkdir -p before creating files."
elif "compute" in task.description.lower():
    additional_context = "Write ONLY the numeric result, no extra text or newlines."
else:
    additional_context = ""
```

```post_process
# Ensure all scripts start with set -e
if not script.startswith("set -e"):
    script = "set -e\\n" + script
```

```utility:classify_task
def classify_task(description):
    desc = description.lower()
    if "file" in desc and "create" in desc:
        return "file_creation"
    elif "compute" in desc or "calculate" in desc:
        return "computation"
    elif "directory" in desc:
        return "directory_ops"
    return "general"
```

```utility:add_error_check
def add_error_check(script, check_cmd):
    return script + f"\\n# Verify\\n{check_cmd}"
```
"""


MINIMAL_RESPONSE = """
```rationale
Adding basic error handling.
```

```post_process
script = "set -e\\n" + script
```
"""


MALFORMED_RESPONSE = """
I think we should improve the scripts but I'm not sure how.
Let me just suggest being more careful.
"""


@pytest.fixture
def mock_llm_client():
    config = LLMConfig(api_key="test")
    client = LLMClient(config)
    return client


@pytest.fixture
def sample_traces():
    return [
        Trace(task_id="t1", depth=1, script="echo hello", success=True, stdout="hello\n"),
        Trace(task_id="t2", depth=1, script="cat /nonexistent", success=False, stderr="No such file"),
        Trace(task_id="t3", depth=1, script="exit 1", success=False, error_summary="Non-zero exit"),
        Trace(task_id="t4", depth=1, script="rm /tmp/x", success=False, stderr="Permission denied"),
    ]


@pytest.fixture
def sample_tasks():
    return [
        TaskDescription(task_id="t1", description="Create a file"),
        TaskDescription(task_id="t2", description="Compute something"),
    ]


class TestOmegaEngine:
    @pytest.mark.asyncio
    async def test_generate_full_response(self, mock_llm_client, sample_traces, sample_tasks):
        mock_llm_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(SAMPLE_OMEGA_RESPONSE, 500)
        )

        engine = OmegaEngine(mock_llm_client)
        injected, tokens = await engine.generate(
            traces=sample_traces,
            context_stack=[],
            tasks=sample_tasks,
            depth=2,
        )

        assert tokens == 500
        assert injected.source_depth == 2
        assert injected.pre_process is not None
        assert "additional_context" in injected.pre_process
        assert "failures" in injected.rationale
        assert not injected.is_empty

    @pytest.mark.asyncio
    async def test_generate_minimal_response(self, mock_llm_client, sample_traces, sample_tasks):
        mock_llm_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(MINIMAL_RESPONSE, 100)
        )

        engine = OmegaEngine(mock_llm_client)
        injected, tokens = await engine.generate(
            traces=sample_traces,
            context_stack=[],
            tasks=sample_tasks,
            depth=2,
        )

        assert injected.pre_process is None
        # MINIMAL_RESPONSE only has post_process (removed) and rationale,
        # so parsed result is effectively empty
        assert injected.is_empty

    @pytest.mark.asyncio
    async def test_generate_malformed_response(self, mock_llm_client, sample_traces, sample_tasks):
        mock_llm_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(MALFORMED_RESPONSE, 50)
        )

        engine = OmegaEngine(mock_llm_client)
        injected, tokens = await engine.generate(
            traces=sample_traces,
            context_stack=[],
            tasks=sample_tasks,
            depth=2,
        )

        # Should return empty InjectedCode gracefully
        assert injected.is_empty
        assert tokens == 50


class TestContextManagerIntegration:
    """Verify OmegaEngine uses ContextManager for sampling and truncation."""

    def test_engine_has_context_manager(self):
        config = LLMConfig(api_key="test")
        client = LLMClient(config)
        engine = OmegaEngine(client)
        assert engine.context_manager is not None

    @pytest.mark.asyncio
    async def test_generate_uses_context_manager(self, mock_llm_client, sample_traces, sample_tasks):
        """Omega should sample traces and truncate context via ContextManager."""
        mock_llm_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(MINIMAL_RESPONSE, 100)
        )

        engine = OmegaEngine(mock_llm_client)

        # Create a large context stack to verify truncation is wired
        big_stack = [
            InjectedCode(
                pre_process="x" * 500,
                rationale="layer",
                source_depth=d,
            )
            for d in range(1, 6)
        ]

        injected, tokens = await engine.generate(
            traces=sample_traces,
            context_stack=big_stack,
            tasks=sample_tasks,
            depth=6,
        )

        # Should complete without error (context manager handles truncation)
        assert tokens == 100


class TestResponseParsing:
    def test_parse_full(self):
        config = LLMConfig(api_key="test")
        client = LLMClient(config)
        engine = OmegaEngine(client)

        injected = engine._parse_response(SAMPLE_OMEGA_RESPONSE, depth=3)
        assert injected.pre_process is not None
        assert injected.source_depth == 3

    def test_parse_minimal(self):
        config = LLMConfig(api_key="test")
        client = LLMClient(config)
        engine = OmegaEngine(client)

        injected = engine._parse_response(MINIMAL_RESPONSE, depth=2)
        assert injected.pre_process is None

    def test_parse_empty(self):
        config = LLMConfig(api_key="test")
        client = LLMClient(config)
        engine = OmegaEngine(client)

        injected = engine._parse_response("no code blocks here", depth=1)
        assert injected.is_empty


class TestPromptBuilding:
    def test_build_prompt_with_context(self, mock_llm_client, sample_traces, sample_tasks):
        engine = OmegaEngine(mock_llm_client)
        prev_code = InjectedCode(
            pre_process="additional_context = 'hint'",
            rationale="Added hints",
            source_depth=2,
        )

        prompt = engine._build_prompt(
            traces=sample_traces,
            context_stack=[prev_code],
            tasks=sample_tasks,
            depth=3,
        )

        assert "Current depth: 3" in prompt
        assert "Tasks attempted: 2" in prompt
        assert "Depth 2" in prompt
        assert "hint" in prompt

    def test_build_prompt_empty_context(self, mock_llm_client, sample_traces, sample_tasks):
        engine = OmegaEngine(mock_llm_client)

        prompt = engine._build_prompt(
            traces=sample_traces,
            context_stack=[],
            tasks=sample_tasks,
            depth=2,
        )

        assert "you are the first meta-layer" in prompt

    def test_build_prompt_with_inspiration(self, mock_llm_client, sample_traces, sample_tasks):
        engine = OmegaEngine(mock_llm_client)
        inspiration = [
            Trace(task_id="task_a", success=True, score=0.95, script="def solve(): return 42"),
        ]

        prompt = engine._build_prompt(
            traces=sample_traces,
            context_stack=[],
            tasks=sample_tasks,
            depth=2,
            inspiration_traces=inspiration,
        )

        assert "Inspiration from Archive" in prompt
        assert "from another candidate" in prompt
        assert "task_a" in prompt
        assert "def solve(): return 42" in prompt

    def test_build_prompt_no_inspiration(self, mock_llm_client, sample_traces, sample_tasks):
        engine = OmegaEngine(mock_llm_client)

        prompt = engine._build_prompt(
            traces=sample_traces,
            context_stack=[],
            tasks=sample_tasks,
            depth=2,
            inspiration_traces=None,
        )

        assert "Inspiration from Archive" not in prompt

    @pytest.mark.parametrize(
        "depth,extra_args,traces_header",
        [
            (2, {}, "## Execution Traces"),
            (3, {"previous_scores": {"t1": 0.0}}, "## Performance Summary & Analysis"),
        ],
        ids=["depth2_OMEGA_PROMPT", "depth3_OMEGA_PROMPT_META"],
    )
    def test_build_prompt_injects_env_notes_from_metadata(
        self, mock_llm_client, sample_traces, depth, extra_args, traces_header,
    ):
        """If any task in the batch carries ``metadata['omega_env_notes']``,
        Ω surfaces its content as a top-level section between System Context
        and the per-task traces. The Ω engine itself is benchmark-agnostic —
        adapters own their own env contracts.

        Structural check: env notes are a SYSTEM-LEVEL concern (apply to all
        tasks), so they must appear OUTSIDE the per-task trace section, not
        nested inside it. Regression test for the "nested ## headers" bug
        the post-refactor audit caught. Parametrized to cover both the
        depth-2 template (OMEGA_PROMPT, ``## Execution Traces``) and the
        depth-3+ template (OMEGA_PROMPT_META, ``## Performance Summary &
        Analysis``) — a regression in either's ``{env_notes_section}``
        placement would re-nest the env block under the traces header."""
        engine = OmegaEngine(mock_llm_client)
        custom_notes = "## Custom Env Notes\nDo not summon the void."
        tasks = [
            TaskDescription(
                task_id="t1", description="x",
                metadata={"omega_env_notes": custom_notes},
            ),
        ]
        prompt = engine._build_prompt(
            traces=sample_traces, context_stack=[], tasks=tasks, depth=depth,
            **extra_args,
        )
        assert custom_notes in prompt
        notes_idx = prompt.find(custom_notes)
        traces_header_idx = prompt.find(traces_header)
        assert traces_header_idx != -1, f"template missing '{traces_header}' section"
        assert notes_idx < traces_header_idx, (
            f"env notes must appear BEFORE the '{traces_header}' header, "
            f"otherwise they're visually nested under it"
        )

    def test_build_prompt_dedups_env_notes_across_tasks(
        self, mock_llm_client, sample_traces
    ):
        """All tasks in a batch share the same env contract → block is
        included once, not N times."""
        engine = OmegaEngine(mock_llm_client)
        note = "## Env Notes\nA single block."
        tasks = [
            TaskDescription(task_id=f"t{i}", description="x", metadata={"omega_env_notes": note})
            for i in range(5)
        ]
        prompt = engine._build_prompt(
            traces=sample_traces, context_stack=[], tasks=tasks, depth=2,
        )
        assert prompt.count(note) == 1

    def test_build_prompt_no_env_notes_when_metadata_absent(
        self, mock_llm_client, sample_traces, sample_tasks
    ):
        """Tasks without ``omega_env_notes`` keep Ω's prompt unchanged —
        backward compat for CO-Bench, TB2, classification etc."""
        engine = OmegaEngine(mock_llm_client)
        prompt = engine._build_prompt(
            traces=sample_traces, context_stack=[], tasks=sample_tasks, depth=2,
        )
        assert "Benchmark Environment Notes" not in prompt
        assert "Custom Env Notes" not in prompt

    def test_universal_small_helpers_advice_present_in_both_templates(
        self, mock_llm_client, sample_traces, sample_tasks
    ):
        """The "favor small helpers" guidance is universal — it must appear
        in BOTH the depth-2 base template AND the depth-3+ meta template,
        since both depths emit helpers. Regression test for the inconsistency
        the post-refactor audit caught (advice was only in the base template)."""
        engine = OmegaEngine(mock_llm_client)
        marker = "Favor small"
        prompt_d2 = engine._build_prompt(
            traces=sample_traces, context_stack=[], tasks=sample_tasks, depth=2,
        )
        # Depth 3 with previous_scores triggers OMEGA_PROMPT_META
        prompt_d3 = engine._build_prompt(
            traces=sample_traces, context_stack=[], tasks=sample_tasks, depth=3,
            previous_scores={"t1": 0.0, "t2": 0.0},
        )
        assert marker in prompt_d2, "OMEGA_PROMPT (depth 2) missing helpers advice"
        assert marker in prompt_d3, "OMEGA_PROMPT_META (depth 3+) missing helpers advice"


class TestOmegaTemperatureAndInspiration:
    @pytest.mark.asyncio
    async def test_generate_with_temperature(self, mock_llm_client, sample_traces, sample_tasks):
        mock_llm_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(MINIMAL_RESPONSE, 100)
        )

        engine = OmegaEngine(mock_llm_client)
        await engine.generate(
            traces=sample_traces,
            context_stack=[],
            tasks=sample_tasks,
            depth=2,
            temperature=0.3,
        )

        # Verify temperature was passed to LLM
        call_kwargs = mock_llm_client._client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_generate_default_temperature(self, mock_llm_client, sample_traces, sample_tasks):
        mock_llm_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(MINIMAL_RESPONSE, 100)
        )

        engine = OmegaEngine(mock_llm_client)
        await engine.generate(
            traces=sample_traces,
            context_stack=[],
            tasks=sample_tasks,
            depth=2,
        )

        # Should default to 0.7
        call_kwargs = mock_llm_client._client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_generate_with_inspiration(self, mock_llm_client, sample_traces, sample_tasks):
        mock_llm_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(MINIMAL_RESPONSE, 100)
        )

        engine = OmegaEngine(mock_llm_client)
        inspiration = [
            Trace(task_id="task_x", success=True, score=0.95, script="best solution"),
        ]

        await engine.generate(
            traces=sample_traces,
            context_stack=[],
            tasks=sample_tasks,
            depth=2,
            inspiration_traces=inspiration,
        )

        # Verify prompt includes inspiration
        call_args = mock_llm_client._client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "Inspiration from Archive" in prompt
        assert "best solution" in prompt


class TestTaskCategorization:
    def test_scheduling(self):
        assert OmegaEngine._categorize_task("job_shop_scheduling") == "scheduling"

    def test_scheduling_landing(self):
        assert OmegaEngine._categorize_task("aircraft_landing") == "scheduling"

    def test_packing(self):
        assert OmegaEngine._categorize_task("bin_packing___one_dimensional") == "packing/cutting"

    def test_cutting(self):
        assert OmegaEngine._categorize_task("constrained_guillotine_cutting") == "packing/cutting"

    def test_assignment(self):
        assert OmegaEngine._categorize_task("assignment_problem") == "assignment/location"

    def test_routing(self):
        assert OmegaEngine._categorize_task("vehicle_routing:_period_routing") == "routing/path"

    def test_tsp(self):
        assert OmegaEngine._categorize_task("travelling_salesman_problem") == "routing/path"

    def test_knapsack(self):
        assert OmegaEngine._categorize_task("multidimensional_knapsack_problem") == "knapsack"

    def test_graph(self):
        assert OmegaEngine._categorize_task("graph_colouring") == "graph"

    def test_unknown(self):
        assert OmegaEngine._categorize_task("mystery_problem_xyz") == "other"

    def test_case_insensitive(self):
        assert OmegaEngine._categorize_task("FLOW_SHOP_SCHEDULING") == "scheduling"


class TestErrorClassification:
    def test_timeout(self):
        t = Trace(task_id="t1", success=False, error_summary="execution timed out after 10s")
        assert OmegaEngine._classify_error(t) == "Timeout"

    def test_dependency(self):
        t = Trace(task_id="t1", success=False, stderr="ModuleNotFoundError: No module named 'scipy'")
        assert OmegaEngine._classify_error(t) == "Dependency error"

    def test_constraint(self):
        t = Trace(task_id="t1", success=False, error_summary="constraint violated: capacity exceeded")
        assert OmegaEngine._classify_error(t) == "Constraint violation"

    def test_overlap(self):
        t = Trace(task_id="t1", success=False, error_summary="Overlap detected in stock")
        assert OmegaEngine._classify_error(t) == "Constraint violation"

    def test_indexing(self):
        t = Trace(task_id="t1", success=False, error_summary="IndexError: list index out of range")
        assert OmegaEngine._classify_error(t) == "Indexing error"

    def test_syntax(self):
        t = Trace(task_id="t1", success=False, error_summary="SyntaxError: unexpected indent")
        assert OmegaEngine._classify_error(t) == "Syntax error"

    def test_fallback(self):
        t = Trace(task_id="t1", success=False, error_summary="something went wrong")
        assert OmegaEngine._classify_error(t) == "Runtime error"


class TestSummarizeTraces:
    def _make_traces(self):
        return [
            Trace(task_id="job_scheduling_1", success=True, score=0.8, depth=2, script="def solve(): pass"),
            Trace(task_id="bin_packing_1", success=False, score=0.0, depth=2, script="def solve(): pass",
                  error_summary="constraint violated"),
            Trace(task_id="assignment_1", success=True, score=1.0, depth=2, script="def solve(): pass"),
            Trace(task_id="tsp_1", success=False, score=0.0, depth=2, script="def solve(): pass",
                  error_summary="execution timed out"),
        ]

    def test_summary_has_category_section(self):
        from meta_n.core.llm_client import LLMConfig, LLMClient
        engine = OmegaEngine(LLMClient(LLMConfig(api_key="test")))
        traces = self._make_traces()
        prev_scores = {"job_scheduling_1": 0.5, "bin_packing_1": 0.3, "assignment_1": 0.8, "tsp_1": 0.2}

        summary = engine._summarize_traces(traces, prev_scores, [])
        assert "Task Category Performance" in summary

    def test_summary_has_failure_patterns(self):
        from meta_n.core.llm_client import LLMConfig, LLMClient
        engine = OmegaEngine(LLMClient(LLMConfig(api_key="test")))
        traces = self._make_traces()
        prev_scores = {"job_scheduling_1": 0.5, "bin_packing_1": 0.3, "assignment_1": 0.8, "tsp_1": 0.2}

        summary = engine._summarize_traces(traces, prev_scores, [])
        assert "Failure Pattern Distribution" in summary
        assert "Timeout" in summary
        assert "Constraint violation" in summary

    def test_summary_has_effectiveness(self):
        from meta_n.core.llm_client import LLMConfig, LLMClient
        engine = OmegaEngine(LLMClient(LLMConfig(api_key="test")))
        traces = self._make_traces()
        prev_scores = {"job_scheduling_1": 0.5, "bin_packing_1": 0.3, "assignment_1": 0.8, "tsp_1": 0.2}

        summary = engine._summarize_traces(traces, prev_scores, [])
        assert "Previous Layer Effectiveness" in summary
        assert "IMPROVED" in summary

    def test_summary_has_representative_traces(self):
        from meta_n.core.llm_client import LLMConfig, LLMClient
        engine = OmegaEngine(LLMClient(LLMConfig(api_key="test")))
        traces = self._make_traces()
        prev_scores = {"job_scheduling_1": 0.5, "bin_packing_1": 0.3, "assignment_1": 0.8, "tsp_1": 0.2}

        summary = engine._summarize_traces(traces, prev_scores, [])
        assert "Representative Traces" in summary


class TestFocusTracePin:
    """P2 — in consolidate/focus mode the focus task's OWN trace must be
    force-included so Ω does not diagnose it from its name alone."""

    def _engine(self):
        from meta_n.core.llm_client import LLMConfig, LLMClient
        return OmegaEngine(LLMClient(LLMConfig(api_key="test")))

    def _fixtures(self):
        # 'assortment' is a passing, non-best, non-regressed task → the default
        # failure/regression/success picks OMIT it. Give it a unique script
        # marker so we can detect whether its trace reached the summary.
        assortment = Trace(
            task_id="assortment", success=True, score=0.575, depth=2,
            script="def solve(data):\n    return CUTTING_PACKING_MARKER",
        )
        fail_a = Trace(
            task_id="fail_a", success=False, score=0.0, depth=2,
            script="def solve(data): pass", error_summary="execution timed out",
        )
        best_b = Trace(
            task_id="best_b", success=True, score=1.0, depth=2,
            script="def solve(data): return 1",
        )
        current = {t.task_id: t for t in (assortment, fail_a, best_b)}
        prev = {"assortment": 0.575, "fail_a": 0.0, "best_b": 1.0}
        return [assortment, fail_a, best_b], current, prev

    def test_default_selection_omits_focus_task(self):
        """Without focus_task, the passing non-best 'assortment' is NOT picked
        (this is the bug P2 addresses) — keeps backward behavior unchanged."""
        engine = self._engine()
        _traces, current, prev = self._fixtures()
        selected = engine._select_representative_traces(current, prev)
        assert all(t.task_id != "assortment" for t in selected)

    def test_focus_task_pinned_as_item_zero(self):
        """With focus_task set, the focus trace is force-included as item 0,
        BEFORE the existing failure/regression/success picks (still present)."""
        engine = self._engine()
        _traces, current, prev = self._fixtures()
        selected = engine._select_representative_traces(
            current, prev, focus_task="assortment"
        )
        assert selected[0].task_id == "assortment"
        # Existing diverse picks are PREPENDED, not removed.
        assert any(t.task_id == "fail_a" for t in selected)

    def test_focus_trace_script_reaches_summary(self):
        """End-to-end: the focus task's script source reaches the rendered
        summary only when focus_task is threaded through _summarize_traces."""
        engine = self._engine()
        traces, _current, prev = self._fixtures()
        without = engine._summarize_traces(traces, prev, [])
        withf = engine._summarize_traces(traces, prev, [], focus_task="assortment")
        assert "CUTTING_PACKING_MARKER" not in without
        assert "CUTTING_PACKING_MARKER" in withf

    def test_unknown_focus_task_is_safe_noop(self):
        """A focus_task not present in the traces must not raise and must leave
        the default selection unchanged."""
        engine = self._engine()
        _traces, current, prev = self._fixtures()
        baseline = engine._select_representative_traces(current, prev)
        selected = engine._select_representative_traces(
            current, prev, focus_task="does_not_exist"
        )
        assert [t.task_id for t in selected] == [t.task_id for t in baseline]


class TestDepthAwarePrompt:
    def test_depth_2_uses_raw_traces(self, mock_llm_client, sample_traces, sample_tasks):
        engine = OmegaEngine(mock_llm_client)
        prompt = engine._build_prompt(sample_traces, [], sample_tasks, depth=2)
        assert "Execution Traces" in prompt
        assert "Performance Summary" not in prompt
        assert "HIGHER-ORDER" not in prompt

    def test_depth_3_with_scores_uses_summary(self, mock_llm_client):
        engine = OmegaEngine(mock_llm_client)
        traces = [
            Trace(task_id="t1", success=True, score=0.8, depth=2, script="def solve(): pass"),
            Trace(task_id="t2", success=False, score=0.0, depth=2, script="def solve(): pass",
                  error_summary="timed out"),
        ]
        tasks = [TaskDescription(task_id="t1", description="A"), TaskDescription(task_id="t2", description="B")]
        prev_scores = {"t1": 0.5, "t2": 0.3}
        ctx = [InjectedCode(pre_process="x=1", rationale="fixed stuff", source_depth=2)]

        prompt = engine._build_prompt(traces, ctx, tasks, depth=3, previous_scores=prev_scores)
        assert "Performance Summary" in prompt
        assert "higher-order" in prompt.lower()
        assert "Task Category Performance" in prompt
        assert "Previous Layer Effectiveness" in prompt

    def test_depth_3_without_scores_falls_back(self, mock_llm_client, sample_traces, sample_tasks):
        engine = OmegaEngine(mock_llm_client)
        prompt = engine._build_prompt(sample_traces, [], sample_tasks, depth=3, previous_scores=None)
        # Should fall back to raw traces / OMEGA_PROMPT
        assert "Execution Traces" in prompt
        assert "Performance Summary" not in prompt

    def test_depth_2_with_previous_scores_adds_comparison(self, mock_llm_client):
        """At depth 2, previous_scores triggers Baseline Comparison section."""
        engine = OmegaEngine(mock_llm_client)
        traces = [
            Trace(task_id="t1", success=True, score=0.6, depth=1, script="pass"),
            Trace(task_id="t2", success=False, score=0.3, depth=1, script="pass"),
        ]
        tasks = [TaskDescription(task_id="t1", description="A"),
                 TaskDescription(task_id="t2", description="B")]
        prev_scores = {"t1": 0.5, "t2": 0.7}

        prompt = engine._build_prompt(traces, [], tasks, depth=2,
                                      previous_scores=prev_scores)
        assert "Baseline Comparison" in prompt
        assert "IMPROVED" in prompt
        assert "REGRESSED" in prompt
        # Template should still be OMEGA_PROMPT (raw traces), not META
        assert "Execution Traces" in prompt
        assert "Performance Summary" not in prompt

    def test_depth_2_archive_best_shown_for_regressions(self, mock_llm_client):
        """Archive best shown alongside regressed tasks."""
        engine = OmegaEngine(mock_llm_client)
        traces = [
            Trace(task_id="t1", success=True, score=0.4, depth=1, script="pass"),
        ]
        tasks = [TaskDescription(task_id="t1", description="A")]
        prev_scores = {"t1": 0.7}
        archive_best = {"t1": 0.8}

        prompt = engine._build_prompt(traces, [], tasks, depth=2,
                                      previous_scores=prev_scores,
                                      archive_best_scores=archive_best)
        assert "[archive best: 0.800]" in prompt

    def test_depth_2_no_comparison_without_previous_scores(self, mock_llm_client, sample_traces, sample_tasks):
        """Without previous_scores, depth 2 has no Baseline Comparison."""
        engine = OmegaEngine(mock_llm_client)
        prompt = engine._build_prompt(sample_traces, [], sample_tasks, depth=2)
        assert "Baseline Comparison" not in prompt

    @pytest.mark.asyncio
    async def test_generate_backward_compat(self, mock_llm_client, sample_traces, sample_tasks):
        """Calling generate() without previous_scores works at any depth."""
        mock_llm_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(MINIMAL_RESPONSE, 100)
        )
        engine = OmegaEngine(mock_llm_client)

        # Depth 3 without previous_scores — should not crash
        injected, tokens = await engine.generate(
            traces=sample_traces, context_stack=[], tasks=sample_tasks, depth=3,
        )
        assert tokens == 100


def _mock_response(content: str, tokens: int):
    """Create a mock OpenAI response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage = MagicMock()
    resp.usage.total_tokens = tokens
    return resp


# --- C2.2: helper-usage scan must match the rendered (truncated) stack ---

class TestHelperUsageMatchesTruncatedStack:
    """C2.2: _helper_usage_section ran over the FULL stack while _build_prompt
    rendered the TRUNCATED stack, so on truncation Omega was told to prune
    helpers it could no longer see. The two views must be consistent. When no
    truncation fires the rendered output is byte-identical.
    """

    def _engine_small_budget(self, mock_llm_client):
        from meta_n.utils.context_manager import ContextBudget
        eng = OmegaEngine(mock_llm_client, context_budget=ContextBudget(max_tokens=10_000))
        return eng

    def test_dropped_helper_absent_from_usage_section(self, mock_llm_client):
        eng = self._engine_small_budget(mock_llm_client)
        old_layer = InjectedCode(
            pre_process="x" * 20_000,
            code_library={"old_helper": "def old_helper(v):\n    return v"},
            source_depth=1,
        )
        new_layer = InjectedCode(
            pre_process="y" * 20_000,
            code_library={"new_helper": "def new_helper(v):\n    return v"},
            source_depth=2,
        )
        stack = [old_layer, new_layer]
        truncated = eng.context_manager.truncate_context_stack(stack)
        # The oldest layer (and its helper) is dropped from the rendered view.
        assert [c.source_depth for c in truncated] == [2]

        traces = [Trace(task_id="t1", depth=1, script="result = 1", success=False)]
        # Full-stack scan would surface old_helper; truncated scan (the fix) must not.
        full_section = eng._helper_usage_section(traces, stack)
        trunc_section = eng._helper_usage_section(traces, truncated)
        assert "old_helper" in full_section
        assert "old_helper" not in trunc_section
        assert "new_helper" in trunc_section

    def test_no_truncation_sections_identical(self, mock_llm_client):
        # Small stack within budget -> truncated == full -> byte-identical scan.
        eng = OmegaEngine(mock_llm_client)  # default 100k budget
        layer = InjectedCode(
            pre_process="z" * 50,
            code_library={"h1": "def h1(v):\n    return v"},
            source_depth=1,
        )
        stack = [layer]
        truncated = eng.context_manager.truncate_context_stack(stack)
        assert [c.source_depth for c in truncated] == [1]
        traces = [Trace(task_id="t1", depth=1, script="h1(2)", success=True)]
        assert eng._helper_usage_section(traces, stack) == eng._helper_usage_section(
            traces, truncated
        )
