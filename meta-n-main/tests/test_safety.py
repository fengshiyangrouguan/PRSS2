"""Tests for code safety validation."""

import pytest

from meta_n.utils.safety import (
    MAX_CODE_LENGTH,
    smoke_test_function,
    validate_code,
)


# --- validate_code tests ---

class TestValidateCode:
    # --- Valid code ---

    def test_empty_code(self):
        ok, err = validate_code("")
        assert ok is True
        assert err == ""

    def test_simple_assignment(self):
        ok, err = validate_code("x = 42")
        assert ok is True

    def test_string_manipulation(self):
        ok, err = validate_code("additional_context = 'hint: ' + task.description[:50]")
        assert ok is True

    def test_conditionals(self):
        code = """
if 'file' in task.description.lower():
    additional_context = 'use touch or echo >'
else:
    additional_context = ''
"""
        ok, err = validate_code(code)
        assert ok is True

    def test_function_definition(self):
        code = """
def classify(desc):
    if 'compute' in desc:
        return 'math'
    return 'general'
"""
        ok, err = validate_code(code)
        assert ok is True

    def test_list_comprehension(self):
        ok, err = validate_code("results = [x * 2 for x in range(10)]")
        assert ok is True

    def test_string_methods(self):
        ok, err = validate_code("script = script.replace('echo', 'printf')")
        assert ok is True

    def test_re_import_allowed(self):
        """Standard library modules not in blocklist should pass."""
        ok, err = validate_code("import re\nresult = re.sub('a', 'b', 'cat')")
        assert ok is True

    def test_json_import_allowed(self):
        ok, err = validate_code("import json")
        assert ok is True

    # --- Blocked imports ---

    def test_blocked_os(self):
        ok, err = validate_code("import os")
        assert ok is False
        assert "os" in err

    def test_blocked_subprocess(self):
        ok, err = validate_code("import subprocess")
        assert ok is False
        assert "subprocess" in err

    def test_blocked_from_os(self):
        ok, err = validate_code("from os import system")
        assert ok is False
        assert "os" in err

    def test_blocked_os_path(self):
        ok, err = validate_code("import os.path")
        assert ok is False
        assert "os" in err

    def test_blocked_shutil(self):
        ok, err = validate_code("import shutil")
        assert ok is False

    def test_blocked_socket(self):
        ok, err = validate_code("import socket")
        assert ok is False

    def test_blocked_http(self):
        ok, err = validate_code("from http.client import HTTPConnection")
        assert ok is False

    def test_blocked_requests(self):
        ok, err = validate_code("import requests")
        assert ok is False

    def test_blocked_ctypes(self):
        ok, err = validate_code("import ctypes")
        assert ok is False

    def test_blocked_sys(self):
        ok, err = validate_code("import sys")
        assert ok is False

    # --- Blocked builtins ---

    def test_blocked_open(self):
        ok, err = validate_code("f = open('/etc/passwd')")
        assert ok is False
        assert "open" in err

    def test_blocked_eval(self):
        ok, err = validate_code("eval('1+1')")
        assert ok is False
        assert "eval" in err

    def test_blocked_exec(self):
        ok, err = validate_code("exec('print(1)')")
        assert ok is False
        assert "exec" in err

    def test_blocked_dunder_import(self):
        ok, err = validate_code("__import__('os')")
        assert ok is False
        assert "__import__" in err

    def test_blocked_compile(self):
        ok, err = validate_code("compile('x=1', '<string>', 'exec')")
        assert ok is False

    # --- Syntax errors ---

    def test_syntax_error(self):
        ok, err = validate_code("def f(:\n  pass")
        assert ok is False
        assert "Syntax error" in err

    # --- Length check ---

    def test_code_too_long(self):
        code = "x = 1\n" * (MAX_CODE_LENGTH + 1)
        ok, err = validate_code(code)
        assert ok is False
        assert "maximum length" in err

    # --- Edge cases ---

    def test_whitespace_only(self):
        ok, err = validate_code("   \n\n  ")
        assert ok is True

    def test_comments_only(self):
        ok, err = validate_code("# just a comment\n# another")
        assert ok is True

    def test_nested_function_ok(self):
        code = """
def helper():
    def inner():
        return 42
    return inner()
"""
        ok, err = validate_code(code)
        assert ok is True

    # --- getattr policy: read-only access allowed, dunder-escape blocked ---

    def test_getattr_with_default_allowed(self):
        ok, err = validate_code('score = getattr(result, "score", 0.0)')
        assert ok is True, err

    def test_getattr_two_arg_allowed(self):
        ok, err = validate_code('value = getattr(obj, "name")')
        assert ok is True, err

    def test_getattr_dynamic_name_allowed(self):
        # Dynamic name slips past static analysis; that's accepted —
        # Docker sandbox is the real boundary.
        ok, err = validate_code('attr_name = "score"\nv = getattr(obj, attr_name)')
        assert ok is True, err

    def test_getattr_class_blocked(self):
        ok, err = validate_code('cls = getattr(obj, "__class__")')
        assert ok is False
        assert "__class__" in err

    def test_getattr_globals_blocked(self):
        ok, err = validate_code('g = getattr(f, "__globals__")')
        assert ok is False
        assert "__globals__" in err

    def test_setattr_still_blocked(self):
        ok, err = validate_code('setattr(obj, "x", 1)')
        assert ok is False
        assert "setattr" in err


class TestSmokeTestFunction:
    def test_valid_function(self):
        passed, err = smoke_test_function("foo", "def foo(): return 42")
        assert passed
        assert err == ""

    def test_syntax_error(self):
        passed, err = smoke_test_function("foo", "def foo(: pass")
        assert not passed
        assert "SyntaxError" in err

    def test_name_mismatch(self):
        passed, err = smoke_test_function("foo", "def bar(): return 1")
        assert not passed
        assert "'foo' not defined" in err
        assert "bar" in err

    def test_import_error(self):
        passed, err = smoke_test_function("f", "import nonexistent_xyz_999\ndef f(): pass")
        assert not passed
        assert "ModuleNotFoundError" in err

    def test_undefined_name_at_module_level(self):
        passed, err = smoke_test_function("f", "x = undefined_var_999\ndef f(): return x")
        assert not passed
        assert "NameError" in err

    def test_non_callable(self):
        passed, err = smoke_test_function("foo", "foo = 42")
        assert not passed
        assert "not callable" in err

    def test_empty_source(self):
        passed, err = smoke_test_function("foo", "")
        assert not passed
        assert "Empty" in err

    def test_function_with_stdlib_import(self):
        passed, err = smoke_test_function("f", "import math\ndef f(): return math.pi")
        assert passed

    def test_multiple_functions(self):
        code = "def helper(): return 1\ndef main_func(): return helper()"
        passed, err = smoke_test_function("main_func", code)
        assert passed

    def test_class_is_callable(self):
        passed, err = smoke_test_function("Foo", "class Foo: pass")
        assert passed
