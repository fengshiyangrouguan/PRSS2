"""InjectionMapper.build — gen0 parity, staged paths, re-export, routing (§12.1)."""

from __future__ import annotations

from meta_n.core.external_agents.injection import InjectionMapper
from meta_n.core.meta_layer import SandboxMarker

from .conftest import make_injected, make_task


def _mapper(injected, language="python"):
    return InjectionMapper(injected, language, SandboxMarker())


# --- gen0 parity -----------------------------------------------------------


def test_gen0_empty_injected_codes_is_vanilla_baseline():
    plan = _mapper([]).build(make_task())
    assert plan.prompt.system_suffix == ""
    assert plan.prompt.prefix == ""
    assert plan.staged_files == {}
    assert plan.utilities_available == []
    assert plan.pre_process_ran is False


def test_gen0_with_additional_context_only_sets_prefix():
    plan = _mapper([]).build(make_task(), additional_context="from a higher layer")
    assert plan.prompt.prefix == "from a higher layer"
    # No libraries, no pre_process emission -> empty suffix, no staged files.
    assert plan.prompt.system_suffix == ""
    assert plan.staged_files == {}
    assert plan.pre_process_ran is False


# --- merged python libraries ----------------------------------------------


def test_python_lib_stages_helper_and_reexport(py_helper_source):
    injected = [make_injected(code_library={"greet": py_helper_source})]
    plan = _mapper(injected, "python").build(make_task())

    assert "helpers/greet.py" in plan.staged_files
    assert plan.staged_files["helpers/greet.py"] == py_helper_source

    # The package __init__ re-exports every helper for `from helpers import x`.
    init = plan.staged_files["helpers/__init__.py"]
    assert "from .greet import greet" in init
    assert '__all__ = ["greet"]' in init

    # python language does NOT emit the bash /tmp wrapper.
    assert "helpers/_lib_greet.py" not in plan.staged_files

    assert plan.utilities_available == ["greet"]
    # The descriptions are advisory and carry the helpers-note.
    assert "helpers/" in plan.prompt.system_suffix


def test_utilities_available_sorted_and_deepest_overrides():
    # Two layers define different helpers; deepest (later) layer wins on clash.
    # Reaudit #8: utilities_available is now the actually-STAGED set (the adoption
    # denominator == "what the agent can invoke"), so each helper's source must
    # bind its keyed name — a name-mismatched helper is dropped (it would
    # ImportError on `from helpers import <key>`), not silently counted.
    shallow = make_injected(
        code_library={"zeta": "def zeta(x):\n    return x\n"}, source_depth=1
    )
    deep = make_injected(
        code_library={"alpha": "def alpha(x):\n    return x\n"}, source_depth=2
    )
    plan = _mapper([shallow, deep], "python").build(make_task())
    # utilities_available is sorted regardless of insertion order.
    assert plan.utilities_available == ["alpha", "zeta"]


def test_python_lib_name_clash_deepest_wins():
    shallow = make_injected(
        code_library={"f": "def f():\n    return 1\n"}, source_depth=1
    )
    deep = make_injected(
        code_library={"f": "def f():\n    return 2\n"}, source_depth=2
    )
    plan = _mapper([shallow, deep], "python").build(make_task())
    assert plan.staged_files["helpers/f.py"] == "def f():\n    return 2\n"


# --- bash routing ----------------------------------------------------------


def test_bash_lib_stages_sh_file(bash_helper_source):
    injected = [make_injected(code_library_bash={"count_lines": bash_helper_source})]
    plan = _mapper(injected, "bash").build(make_task())
    assert plan.staged_files["helpers/count_lines.sh"] == bash_helper_source
    assert plan.utilities_available == ["count_lines"]


def test_bash_language_stages_python_runnable_wrapper(py_helper_source):
    # With bash routing AND a python helper present, a *real, runnable Python*
    # _lib file is staged (NOT a bash heredoc — that was the bug: literal bash
    # inside a .py that python3 could never execute).
    import ast

    injected = [make_injected(code_library={"greet": py_helper_source})]
    plan = _mapper(injected, "bash").build(make_task())
    assert "helpers/greet.py" in plan.staged_files
    assert "helpers/_lib_greet.py" in plan.staged_files

    wrapper = plan.staged_files["helpers/_lib_greet.py"]
    # The staged content must be valid, importable/executable Python — never a
    # bash heredoc (no `cat <<` / `_EOF` markers, no /tmp write redirect).
    ast.parse(wrapper)  # raises SyntaxError if it were bash
    assert "cat <<" not in wrapper
    assert "_EOF" not in wrapper
    assert "> /tmp/_lib_greet.py" not in wrapper
    # It carries the helper definition plus the __main__ CLI-dispatch tail so
    # `python3 ./helpers/_lib_greet.py <args>` actually runs the helper.
    assert "def greet(" in wrapper
    assert 'if __name__ == "__main__":' in wrapper
    assert "greet(*sys.argv[1:])" in wrapper

    # Executable check: compiling + exec'ing the module body must define `greet`
    # (i.e. it really is runnable Python, not just parseable).
    ns: dict = {}
    exec(compile(wrapper, "helpers/_lib_greet.py", "exec"), ns)  # noqa: S102 - test fixture
    assert callable(ns["greet"])
    assert ns["greet"]("bob") == "hi bob"

    # The bash description points at the staged workspace-relative path, not /tmp.
    suffix = plan.prompt.system_suffix
    assert "python3 ./helpers/_lib_greet.py" in suffix
    assert "/tmp/_lib_greet.py" not in suffix


def test_python_language_no_runnable_wrapper(py_helper_source):
    injected = [make_injected(code_library={"greet": py_helper_source})]
    plan = _mapper(injected, "python").build(make_task())
    assert "helpers/_lib_greet.py" not in plan.staged_files


# --- pre_process channel ---------------------------------------------------


def test_pre_process_ran_flag_and_suffix():
    # A pre_process block that emits context sets pre_process_ran and the suffix.
    pp = "additional_context = 'PREPROCESSED: ' + task.task_id"
    injected = [make_injected(pre_process=pp)]
    plan = _mapper(injected, "python").build(make_task(task_id="t7"))
    assert plan.pre_process_ran is True
    assert "PREPROCESSED: t7" in plan.prompt.system_suffix


def test_pre_process_empty_emission_does_not_set_ran():
    # A block that leaves additional_context empty contributes no context.
    pp = "additional_context = ''"
    injected = [make_injected(pre_process=pp)]
    plan = _mapper(injected, "python").build(make_task())
    assert plan.pre_process_ran is False
    assert plan.prompt.system_suffix == ""


def test_pre_process_runs_with_empty_outer_context():
    # The execute() entry point runs pre_process with outer_context="" — verify
    # the block sees an empty outer_context (the prefix carries inter-layer ctx).
    pp = "additional_context = 'outer=[' + outer_context + ']'"
    injected = [make_injected(pre_process=pp)]
    plan = _mapper(injected, "python").build(
        make_task(), additional_context="prefix-only"
    )
    assert "outer=[]" in plan.prompt.system_suffix
    assert plan.prompt.prefix == "prefix-only"


# --- G8: concrete, directive helper affordance -----------------------------


def test_helpers_note_gives_concrete_call_syntax_per_language(py_helper_source):
    """G8: forensic ab4 analysis found agents NEVER called staged helpers under
    the old vague advisory note. The affordance is now concrete (exact call
    syntax) and directive (prefer-these), language-aware for the staged-file
    model."""
    injected = [make_injected(code_library={"greet": py_helper_source})]
    py = _mapper(injected, "python").build(make_task()).prompt.system_suffix
    assert "PREFER calling them" in py
    assert "from helpers import" in py

    bash = _mapper(injected, "bash").build(make_task()).prompt.system_suffix
    assert "PREFER calling them" in bash
    assert "python3 ./helpers/_lib_<name>.py" in bash
