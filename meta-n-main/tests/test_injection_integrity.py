"""Injection-integrity tests (Step 3): funcname re-key (b), solver_lib bare-name
rewrite (a), phantom-signature drop under sandbox (c)."""

from meta_n.core.meta_layer import (
    format_python_library_descriptions,
    prepend_python_library,
)
from meta_n.core.omega import OmegaEngine
from meta_n.utils.safety import smoke_test_function


class _Sandbox:
    is_sandboxed = True


# --------------------------------------------------------------------------- #
# (b) register by discovered funcname  (the gemma 17-37% load-death)
# --------------------------------------------------------------------------- #

def test_extract_solver_libs_rekeys_by_funcname():
    libs = OmegaEngine._extract_solver_libs(None, "```solver_lib:foo\ndef bar(x):\n    return x + 1\n```")
    assert "bar" in libs and "foo" not in libs  # re-keyed to the real funcname


def test_extract_solver_libs_matching_label_unchanged():
    libs = OmegaEngine._extract_solver_libs(None, "```solver_lib:rz\ndef rz(v):\n    return v\n```")
    assert "rz" in libs


def test_rekeyed_helper_now_passes_smoke_test():
    libs = OmegaEngine._extract_solver_libs(None, "```solver_lib:foo\ndef bar(x):\n    return x\n```")
    name, source = next(iter(libs.items()))
    passed, _ = smoke_test_function(name, source)
    assert passed is True  # keyed as 'foo' this FAILED (the load-death) — now 'bar'


def test_extract_solver_libs_unparseable_falls_back_to_label():
    libs = OmegaEngine._extract_solver_libs(None, "```solver_lib:foo\ndef bar(:\n  broken\n```")
    assert "foo" in libs  # syntax error → keep the label


# --------------------------------------------------------------------------- #
# (a) solver_lib bare-name rewrite + namespace shim
# --------------------------------------------------------------------------- #

def test_prepend_rewrites_solver_lib_dotted_calls():
    lib = {"robust_zscore": "def robust_zscore(d):\n    return d * 2"}
    out = prepend_python_library("def solve(**kw):\n    return solver_lib.robust_zscore(5)", lib)
    assert "solver_lib.robust_zscore(5)" not in out  # dotted form rewritten to bare
    ns = {}
    exec(out, ns)  # no NameError
    assert ns["solve"]() == 10


def test_prepend_shim_exposes_solver_lib_namespace():
    out = prepend_python_library("x = 1", {"helper": "def helper():\n    return 42"})
    ns = {}
    exec(out, ns)
    assert ns["solver_lib"].helper() == 42  # belt-and-suspenders shim resolves


def test_prepend_no_library_is_noop():
    assert prepend_python_library("x = 1", {}) == "x = 1"


# --------------------------------------------------------------------------- #
# (c) phantom-signature drop under sandbox (advertise-set ⊆ prepend-set)
# --------------------------------------------------------------------------- #

def test_sandboxed_flagged_helper_not_advertised_but_prepended():
    unsafe = {"sneaky": "def sneaky():\n    return globals()"}  # parseable but flagged
    desc = format_python_library_descriptions(unsafe, executor=_Sandbox())
    assert "sneaky" not in desc                # NOT advertised (blocklist applies for ads)
    prepended = prepend_python_library("x = 1", unsafe, executor=_Sandbox())
    assert "def sneaky" in prepended           # ...but still prepended (sandbox contains it)


def test_sandboxed_safe_helper_advertised_and_prepended():
    safe = {"clean": "def clean(v):\n    return v + 1"}
    assert "clean" in format_python_library_descriptions(safe, executor=_Sandbox())
    assert "def clean" in prepend_python_library("x = 1", safe, executor=_Sandbox())
