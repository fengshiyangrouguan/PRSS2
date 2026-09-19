"""Tests for code library injection."""

import json

import pytest

from meta_n.core.base_executor import LocalExecutor
from meta_n.core.meta_layer import InjectedCode, MetaLayer, TaskDescription, merge_code_libraries


class MockSolver:
    """Mock solver that returns a fixed script."""

    def __init__(self, script: str = "echo hello"):
        self.script = script
        self.last_additional_context = ""

    async def solve(self, task: TaskDescription, additional_context: str = "") -> tuple[str, str, int]:
        self.last_additional_context = additional_context
        return self.script, "mock reasoning", 50


@pytest.fixture
def task():
    return TaskDescription(task_id="test_lib", description="Test task")


# --- InjectedCode with code_library ---


class TestInjectedCodeLibrary:
    def test_not_empty_with_code_library(self):
        ic = InjectedCode(code_library={"f": "def f(): pass"})
        assert not ic.is_empty

    def test_empty_without_anything(self):
        ic = InjectedCode()
        assert ic.is_empty

    def test_serialization_roundtrip(self):
        ic = InjectedCode(
            pre_process='additional_context = "hello"',
            code_library={"greet": "def greet(name):\n    return f'Hello {name}'"},
            rationale="test",
            source_depth=2,
        )
        data = json.loads(ic.model_dump_json())
        assert "code_library" in data
        assert "greet" in data["code_library"]

        restored = InjectedCode.model_validate(data)
        assert restored.code_library == ic.code_library

    def test_backward_compat_no_code_library_field(self):
        """Old JSON without code_library should load with empty dict."""
        old_json = {
            "pre_process": "additional_context = ''",
            "rationale": "old",
            "source_depth": 2,
        }
        ic = InjectedCode.model_validate(old_json)
        assert ic.code_library == {}
        assert not ic.is_empty  # has pre_process


# --- merge_code_libraries ---


class TestMergeCodeLibraries:
    def test_empty(self):
        py, bash = merge_code_libraries([])
        assert py == {}
        assert bash == {}

    def test_single(self):
        ic = InjectedCode(code_library={"f": "def f(): pass"})
        py, bash = merge_code_libraries([ic])
        assert py == {"f": "def f(): pass"}
        assert bash == {}

    def test_additive(self):
        ic1 = InjectedCode(code_library={"f": "def f(): pass"})
        ic2 = InjectedCode(code_library={"g": "def g(): pass"})
        py, bash = merge_code_libraries([ic1, ic2])
        assert py == {"f": "def f(): pass", "g": "def g(): pass"}

    def test_override_by_name(self):
        ic1 = InjectedCode(code_library={"f": "def f(): return 1"})
        ic2 = InjectedCode(code_library={"f": "def f(): return 2"})
        py, bash = merge_code_libraries([ic1, ic2])
        assert py == {"f": "def f(): return 2"}  # later wins

    def test_mixed_override_and_additive(self):
        ic1 = InjectedCode(code_library={"f": "v1", "g": "v1"})
        ic2 = InjectedCode(code_library={"f": "v2", "h": "v1"})
        py, bash = merge_code_libraries([ic1, ic2])
        assert py == {"f": "v2", "g": "v1", "h": "v1"}

    def test_bash_libraries(self):
        ic1 = InjectedCode(code_library_bash={"setup": "setup() { echo ok; }"})
        ic2 = InjectedCode(code_library_bash={"cleanup": "cleanup() { rm -f /tmp/*.tmp; }"})
        py, bash = merge_code_libraries([ic1, ic2])
        assert py == {}
        assert bash == {"setup": "setup() { echo ok; }", "cleanup": "cleanup() { rm -f /tmp/*.tmp; }"}

    def test_mixed_python_and_bash(self):
        ic = InjectedCode(
            code_library={"helper": "def helper(): pass"},
            code_library_bash={"setup": "setup() { echo ok; }"},
        )
        py, bash = merge_code_libraries([ic])
        assert py == {"helper": "def helper(): pass"}
        assert bash == {"setup": "setup() { echo ok; }"}


# --- _format_library_descriptions ---


class TestFormatLibraryDescriptions:
    def test_empty(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
        )
        assert layer._format_library_descriptions() == ""

    def test_basic_signature(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={"greet": "def greet(name):\n    return f'Hello {name}'"},
        )
        desc = layer._format_library_descriptions()
        assert "Available Helper Functions" in desc
        assert "def greet(name):" in desc

    def test_with_docstring(self, task):
        code = 'def search(items, key):\n    """Find key in items."""\n    pass'
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={"search": code},
        )
        desc = layer._format_library_descriptions()
        assert "Find key in items." in desc

    def test_full_docstring_shown(self, task):
        code = (
            'def pack(items: list, cap: int) -> tuple:\n'
            '    """Pack items into bins.\n'
            '\n'
            '    Args:\n'
            '        items: list[tuple[int, int]] — each is (w, h).\n'
            '        cap: int — max capacity.\n'
            '\n'
            '    Returns:\n'
            '        tuple[int, list] — (cost, placements).\n'
            '    """\n'
            '    pass'
        )
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={"pack": code},
        )
        desc = layer._format_library_descriptions()
        assert "Args:" in desc
        assert "Returns:" in desc
        assert "list[tuple[int, int]]" in desc

    def test_library_instructions_appended(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={"f": "def f(x):\n    return x"},
        )
        desc = layer._format_library_descriptions()
        assert "Do NOT import them" in desc
        assert "solver_lib" in desc

    def test_no_instructions_when_empty(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
        )
        desc = layer._format_library_descriptions()
        assert desc == ""
        assert "Do NOT import" not in desc

    def test_syntax_error_excluded(self, task):
        """Functions with syntax errors should not appear in descriptions
        (they won't be prepended either, so the solver shouldn't know about them)."""
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={"broken": "def broken(: pass"},
        )
        desc = layer._format_library_descriptions()
        assert desc == ""  # broken function excluded entirely

    def test_mixed_valid_and_invalid(self, task):
        """Only valid functions should appear in descriptions."""
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={
                "good": "def good(x):\n    return x + 1",
                "broken": "def broken(: pass",
                "dangerous": "import os\ndef dangerous():\n    os.system('rm -rf /')",
            },
        )
        desc = layer._format_library_descriptions()
        assert "def good(x):" in desc
        assert "broken" not in desc
        assert "dangerous" not in desc

    def test_smoke_test_failure_excluded(self, task):
        """Functions that pass static validation but fail smoke test are excluded."""
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={
                "good": "def good(x):\n    return x + 1",
                # Passes validate_code (nonexistent_xyz not in BLOCKED_IMPORTS)
                # but fails smoke test (ModuleNotFoundError on exec)
                "bad_import": "import nonexistent_xyz_999\ndef bad_import(): pass",
            },
        )
        desc = layer._format_library_descriptions()
        assert "def good(x):" in desc
        assert "bad_import" not in desc

    def test_name_mismatch_excluded(self, task):
        """Library key 'helper' but source defines 'wrong_name' → excluded."""
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={
                "helper": "def wrong_name(): return 1",
            },
        )
        desc = layer._format_library_descriptions()
        assert desc == ""  # name mismatch → excluded


# --- _prepend_library ---


class TestPrependLibrary:
    def test_empty_library(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
        )
        assert layer._prepend_library("print('hi')") == "print('hi')"

    def test_basic_prepend(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={"helper": "def helper():\n    return 42"},
        )
        result = layer._prepend_library("print(helper())")
        assert "injected code library" in result
        assert "def helper():" in result
        assert result.index("def helper()") < result.index("print(helper())")

    def test_python_prepends(self, task):
        """Python solver mode prepends library (classification now uses Python)."""
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={"helper": "def helper(): pass"},
            solver_language="python",
        )
        script = "def solve(cases, labels, few_shot):\n    return {}"
        result = layer._prepend_library(script)
        assert "def helper():" in result
        assert result.index("def helper()") < result.index("def solve(")

    def test_validation_failure_skipped(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={
                "good": "def good():\n    return 1",
                "bad": "import os\ndef bad():\n    os.system('rm -rf /')",
            },
        )
        result = layer._prepend_library("pass")
        assert "def good():" in result
        assert "def bad():" not in result

    def test_smoke_test_failure_skipped(self, task):
        """Functions that pass static validation but fail exec are skipped."""
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={
                "good": "def good():\n    return 1",
                "bad_import": "import nonexistent_xyz_999\ndef bad_import(): pass",
            },
        )
        result = layer._prepend_library("pass")
        assert "def good():" in result
        assert "bad_import" not in result

    def test_name_mismatch_skipped(self, task):
        """Library key doesn't match defined function → skipped."""
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={
                "helper": "def wrong_name(): return 1",
            },
        )
        result = layer._prepend_library("pass")
        assert "def wrong_name" not in result  # mismatched function not prepended
        assert "def helper" not in result


# --- End-to-end: MetaLayer with library ---


class TestExecuteWithLibrary:
    @pytest.mark.asyncio
    async def test_library_prepended_to_executed_script(self, task):
        solver = MockSolver(script="echo 'solver output'")
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=solver,
            executor=LocalExecutor(),
            merged_code_library={"helper": "def helper():\n    return 42"},
        )
        trace, tokens = await layer.execute(task)
        # trace.script has the full executed script (library + solver code)
        # so it's reproducible for test evaluation and debugging
        assert "def helper()" in trace.script
        assert "echo 'solver output'" in trace.script
        assert trace.script.index("def helper()") < trace.script.index("echo 'solver output'")

    @pytest.mark.asyncio
    async def test_library_descriptions_in_solver_context(self, task):
        solver = MockSolver(script="echo test")
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=solver,
            executor=LocalExecutor(),
            merged_code_library={"helper": "def helper(x):\n    \"\"\"Do something.\"\"\"\n    pass"},
        )
        await layer.execute(task)
        assert "Available Helper Functions" in solver.last_additional_context
        assert "def helper(x):" in solver.last_additional_context
