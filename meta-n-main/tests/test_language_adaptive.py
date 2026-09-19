"""Tests for language-adaptive code injection (bash solver support)."""

import pytest

from meta_n.core.base_executor import BaseExecutor, LocalExecutor
from meta_n.core.meta_layer import (
    InjectedCode,
    MetaLayer,
    TaskDescription,
    merge_code_libraries,
    validate_library_function,
    wrap_python_as_file,
)
from meta_n.core.omega import OmegaEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockSolver:
    """Mock solver that returns a fixed script."""

    def __init__(self, script: str = "echo hello"):
        self.script = script
        self.last_additional_context = ""

    async def solve(self, task, additional_context=""):
        self.last_additional_context = additional_context
        return self.script, "mock reasoning", 50


class SandboxedExecutor(BaseExecutor):
    """Mock sandboxed executor (Docker-like)."""

    @property
    def is_sandboxed(self) -> bool:
        return True

    async def execute(self, script, task, timeout=30):
        from meta_n.core.meta_layer import Trace
        return Trace(task_id=task.task_id, script=script, success=True, score=1.0)


@pytest.fixture
def task():
    return TaskDescription(task_id="test_bash", description="Test bash task")


# ---------------------------------------------------------------------------
# InjectedCode — bash fields
# ---------------------------------------------------------------------------


class TestInjectedCodeBash:
    def test_not_empty_with_bash_library(self):
        ic = InjectedCode(code_library_bash={"setup": "setup() { echo ok; }"})
        assert not ic.is_empty

    def test_empty_without_anything(self):
        ic = InjectedCode()
        assert ic.is_empty

    def test_not_empty_with_only_python_library(self):
        ic = InjectedCode(code_library={"f": "def f(): pass"})
        assert not ic.is_empty

    def test_backward_compat_no_bash_field(self):
        """Old JSON without code_library_bash should load with empty dict."""
        old_json = {
            "pre_process": "additional_context = ''",
            "code_library": {"f": "def f(): pass"},
            "rationale": "old",
            "source_depth": 2,
        }
        ic = InjectedCode.model_validate(old_json)
        assert ic.code_library_bash == {}
        assert not ic.is_empty

    def test_serialization_roundtrip_with_bash(self):
        ic = InjectedCode(
            code_library={"helper": "def helper(): pass"},
            code_library_bash={"setup": "setup() { echo ok; }"},
            rationale="test",
            source_depth=2,
        )
        import json
        data = json.loads(ic.model_dump_json())
        assert "code_library_bash" in data
        assert "setup" in data["code_library_bash"]

        restored = InjectedCode.model_validate(data)
        assert restored.code_library_bash == ic.code_library_bash
        assert restored.code_library == ic.code_library


# ---------------------------------------------------------------------------
# _prepend_library — bash mode
# ---------------------------------------------------------------------------


class TestPrependBashLibrary:
    def test_bash_functions_prepended_directly(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=SandboxedExecutor(),
            merged_code_library_bash={"setup": "setup() {\n    echo ok\n}"},
            solver_language="bash",
        )
        result = layer._prepend_library("echo hello")
        assert "setup() {" in result
        assert result.index("setup()") < result.index("echo hello")
        assert "injected code library" in result

    def test_python_as_heredoc_file(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=SandboxedExecutor(),
            merged_code_library={"recover": "def recover(path: str) -> list:\n    return []"},
            solver_language="bash",
        )
        result = layer._prepend_library("echo hello")
        assert "cat <<" in result
        assert "/tmp/_lib_recover.py" in result
        assert "PYTHON_LIB_RECOVER_EOF" in result
        assert 'if __name__ == "__main__":' in result

    def test_mixed_bash_and_python(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=SandboxedExecutor(),
            merged_code_library={"helper": "def helper(x: str) -> str:\n    return x"},
            merged_code_library_bash={"setup": "setup() { apt update; }"},
            solver_language="bash",
        )
        result = layer._prepend_library("echo hello")
        assert "lib_bash:setup" in result
        assert "lib_py:helper" in result
        assert "/tmp/_lib_helper.py" in result

    def test_empty_libraries_no_prepend(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=SandboxedExecutor(),
            solver_language="bash",
        )
        assert layer._prepend_library("echo hello") == "echo hello"

    def test_python_solver_unchanged(self, task):
        """Python solver mode uses original prepend logic."""
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={"helper": "def helper():\n    return 42"},
            solver_language="python",
        )
        result = layer._prepend_library("print(helper())")
        # Should NOT have heredoc
        assert "cat <<" not in result
        # Should have direct Python prepend
        assert "def helper():" in result
        assert "lib:" in result  # original marker


# ---------------------------------------------------------------------------
# _format_library_descriptions — bash mode
# ---------------------------------------------------------------------------


class TestFormatBashLibraryDescriptions:
    def test_bash_functions_described(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=SandboxedExecutor(),
            merged_code_library_bash={"setup": "setup() {\n    apt update\n}"},
            solver_language="bash",
        )
        desc = layer._format_library_descriptions()
        assert "setup()" in desc
        assert "bash function" in desc
        assert "Available Helper Code" in desc

    def test_python_scripts_described(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=SandboxedExecutor(),
            merged_code_library={"recover": 'def recover(path: str) -> list:\n    """Recover data from file."""\n    return []'},
            solver_language="bash",
        )
        desc = layer._format_library_descriptions()
        assert "python3 /tmp/_lib_recover.py" in desc
        assert "Recover data from file." in desc

    def test_mixed_descriptions(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=SandboxedExecutor(),
            merged_code_library={"recover": 'def recover(path: str) -> list:\n    """Recover data."""\n    return []'},
            merged_code_library_bash={"setup": "setup() { echo ok; }"},
            solver_language="bash",
        )
        desc = layer._format_library_descriptions()
        assert "bash function" in desc
        assert "python3 /tmp/_lib_recover.py" in desc

    def test_empty_returns_empty(self, task):
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=SandboxedExecutor(),
            solver_language="bash",
        )
        assert layer._format_library_descriptions() == ""

    def test_python_solver_uses_original_format(self, task):
        """Python solver descriptions should use the original format."""
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=MockSolver(),
            executor=LocalExecutor(),
            merged_code_library={"greet": "def greet(name):\n    return f'Hello {name}'"},
            solver_language="python",
        )
        desc = layer._format_library_descriptions()
        assert "Available Helper Functions" in desc
        assert "def greet(name):" in desc
        # Should NOT have bash-style descriptions
        assert "python3 /tmp/" not in desc


# ---------------------------------------------------------------------------
# validate_library_function — sandboxed vs unsandboxed (module-level function,
# the production path; the MetaLayer._validate_library_function shim is gone)
# ---------------------------------------------------------------------------


class TestValidationSandboxed:
    def test_sandboxed_skips_safety_blocklist(self):
        """Sandboxed executor should skip safety validation (Docker IS the sandbox)."""
        # This would fail safety validation (open() is blocked) but should pass
        # in sandboxed mode (only syntax check)
        source = 'def read_file(path: str) -> str:\n    with open(path) as f:\n        return f.read()'
        assert validate_library_function(
            "read_file", source, executor=SandboxedExecutor(),
        ) is True

    def test_sandboxed_still_checks_syntax(self):
        """Even sandboxed, syntax errors should be caught."""
        assert validate_library_function(
            "broken", "def broken(: pass", executor=SandboxedExecutor(),
        ) is False

    def test_unsandboxed_uses_full_validation(self):
        """LocalExecutor (unsandboxed) should use full safety validation."""
        # open() is blocked by safety validator
        source = 'def read_file(path: str) -> str:\n    with open(path) as f:\n        return f.read()'
        assert validate_library_function(
            "read_file", source, executor=LocalExecutor(),
        ) is False

    def test_bash_functions_always_valid(self):
        """Bash functions bypass all Python validation."""
        assert validate_library_function(
            "setup", "this is not python at all!", is_bash=True,
            executor=LocalExecutor(),
        ) is True


# ---------------------------------------------------------------------------
# Omega parser — bash block extraction
# ---------------------------------------------------------------------------


class TestOmegaParserBash:
    def test_extract_solver_libs_bash(self):
        engine = OmegaEngine.__new__(OmegaEngine)
        response = (
            "Here is my improvement:\n\n"
            "```solver_lib_bash:install_deps\n"
            "install_deps() {\n"
            "    apt-get update -qq\n"
            "}\n"
            "```\n"
        )
        libs = engine._extract_solver_libs_bash(response)
        assert "install_deps" in libs
        assert "apt-get update" in libs["install_deps"]

    def test_extract_both_block_types(self):
        engine = OmegaEngine.__new__(OmegaEngine)
        response = (
            "```solver_lib:recover\n"
            "def recover(path: str) -> list:\n"
            "    return []\n"
            "```\n\n"
            "```solver_lib_bash:setup\n"
            "setup() { echo ok; }\n"
            "```\n"
        )
        py_libs = engine._extract_solver_libs(response)
        bash_libs = engine._extract_solver_libs_bash(response)
        assert "recover" in py_libs
        assert "setup" in bash_libs

    def test_parse_response_includes_both(self):
        engine = OmegaEngine.__new__(OmegaEngine)
        response = (
            "```rationale\nImprove setup\n```\n\n"
            "```pre_process\nadditional_context = 'test'\n```\n\n"
            "```solver_lib:helper\n"
            "def helper(x: str) -> str:\n"
            '    """Help."""\n'
            "    return x\n"
            "```\n\n"
            "```solver_lib_bash:install\n"
            "install() { apt install -y python3; }\n"
            "```\n"
        )
        injected = engine._parse_response(response, depth=2)
        assert injected.pre_process is not None
        assert "helper" in injected.code_library
        assert "install" in injected.code_library_bash
        assert injected.rationale == "Improve setup"
        assert not injected.is_empty

    def test_format_context_stack_includes_bash(self):
        engine = OmegaEngine.__new__(OmegaEngine)
        stack = [
            InjectedCode(
                code_library={"helper": "def helper(): pass"},
                code_library_bash={"setup": "setup() { echo ok; }"},
                rationale="test",
                source_depth=2,
            )
        ]
        text = engine._format_context_stack(stack)
        assert "solver_lib:helper" in text
        assert "solver_lib_bash:setup" in text
        assert "```python" in text
        assert "```bash" in text


# ---------------------------------------------------------------------------
# BaseExecutor.is_sandboxed
# ---------------------------------------------------------------------------


class TestIsSandboxed:
    def test_local_executor_not_sandboxed(self):
        assert LocalExecutor().is_sandboxed is False

    def test_sandboxed_executor(self):
        assert SandboxedExecutor().is_sandboxed is True


# ---------------------------------------------------------------------------
# wrap_python_as_file
# ---------------------------------------------------------------------------


class TestWrapPythonAsFile:
    def test_basic_wrap(self):
        result = wrap_python_as_file("recover", "def recover(path):\n    return []")
        assert result.startswith("cat << 'PYTHON_LIB_RECOVER_EOF'")
        assert "/tmp/_lib_recover.py" in result
        assert "def recover(path):" in result
        assert '__name__ == "__main__"' in result
        assert result.endswith("PYTHON_LIB_RECOVER_EOF")
