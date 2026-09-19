"""Code safety validation for Omega-generated code.

Defense-in-depth: Docker sandbox is the primary safety boundary.
This module provides static analysis as a second layer.

Host-exec reality: ``smoke_test_function`` (and the pre_process pipeline in
meta_layer.py) exec model-emitted code in the meta-n HOST process, gated only
by this static validation plus a wall-clock bound — not by the Docker sandbox.
"""

from __future__ import annotations

import ast
import threading
from typing import Optional


# Imports/attributes that should not appear in injected code
BLOCKED_IMPORTS = {
    "os",
    "subprocess",
    "shutil",
    "socket",
    "http",
    "urllib",
    "urllib3",
    "requests",
    "pathlib",
    "ctypes",
    "importlib",
    "sys",
    "multiprocessing",
    "threading",
    "signal",
    "tempfile",
    "io",
    # C-level os twins + escape/serialization vectors (#64). `posix`/`nt`
    # directly expose system/execv/popen/fork/spawn* without going through `os`.
    "posix",
    "nt",
    "gc",
    "inspect",
    "pickle",
    "marshal",
    "fcntl",
    "mmap",
    "resource",
    "builtins",
}

# NOTE: dotted entries are matched by PREFIX (``attr_chain.startswith(blocked)``)
# below, so ``os.exec``/``os.spawn``/``posix.exec``/``posix.spawn`` intentionally
# cover the whole ``execl/execv/...`` and ``spawnl/spawnv/...`` families.
# The bare names ``eval``/``exec``/``__import__`` are deliberately NOT listed
# here: their direct-call forms are already caught by the ``ast.Name`` branch in
# ``validate_code``, and as undotted prefix entries they fired as false positives
# on any receiver whose root identifier merely STARTS WITH them
# (``executor.run``, ``evaluation.append``, ``eval_scores.sort``) — see #28.
BLOCKED_ATTRIBUTES = {
    "os.system",
    "os.popen",
    "os.exec",
    "os.spawn",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "posix.system",
    "posix.exec",
    "posix.spawn",
    "posix.popen",
    "posix.fork",
    "nt.system",
    "nt.popen",
    "nt.spawn",
    "nt.startfile",
}

MAX_CODE_LENGTH = 30_000  # characters

# Dunder names that are sandbox-escape vectors. Single source for BOTH the
# direct ``.{attr}`` ast.Attribute walk and the getattr string-literal check
# in ``validate_code``.
_DUNDER_ESCAPE_NAMES = frozenset({
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__builtins__", "__globals__", "__code__", "__func__", "__dict__",
})


def validate_code(code: str) -> tuple[bool, str]:
    """
    Validate Python code for safety before execution.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    if not code or not code.strip():
        return True, ""

    # Length check
    if len(code) > MAX_CODE_LENGTH:
        return False, f"Code exceeds maximum length ({len(code)} > {MAX_CODE_LENGTH})"

    # Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # AST walk for dangerous patterns
    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".")[0]
                if module_root in BLOCKED_IMPORTS:
                    return False, f"Blocked import: {alias.name}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_root = node.module.split(".")[0]
                if module_root in BLOCKED_IMPORTS:
                    return False, f"Blocked import: from {node.module}"

        # Check for dangerous builtin calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in (
                "open", "__import__", "eval", "exec", "compile",
                "setattr", "delattr", "globals", "locals", "vars",
                "breakpoint", "input",
            ):
                return False, f"Blocked builtin call: {node.func.id}()"

            # ``getattr(obj, "x", default)`` is a read-only access pattern
            # the LLM often reaches for (e.g., ``getattr(result, "score", 0.0)``)
            # — blocking it wholesale dropped 5 cells' Ω pre_process in the
            # v2 smoke. Allow it, but block the sandbox-escape spelling
            # ``getattr(obj, "__class__")`` etc. when the attr name is a
            # string literal we can statically inspect. (Dynamic attrs like
            # ``getattr(obj, name)`` slip through static analysis either way;
            # the Docker sandbox is the real safety boundary.)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and node.args[1].value in _DUNDER_ESCAPE_NAMES
            ):
                return False, f"Blocked dunder access via getattr: {node.args[1].value!r}"

            # Check for dangerous attribute calls like os.system()
            if isinstance(node.func, ast.Attribute):
                attr_chain = _get_attribute_chain(node.func)
                if attr_chain:
                    for blocked in BLOCKED_ATTRIBUTES:
                        if attr_chain.startswith(blocked):
                            return False, f"Blocked call: {attr_chain}()"

        # Check for dunder attribute access (sandbox escape vectors)
        elif isinstance(node, ast.Attribute):
            if node.attr in _DUNDER_ESCAPE_NAMES:
                return False, f"Blocked dunder access: .{node.attr}"

    # 6.4b: forbid branching on a LITERAL task_id (eval-set memorization /
    # feature-routing reward-hack). Branching on task FEATURES is allowed.
    literal = task_id_literal_branch(code)
    if literal is not None:
        return False, f"Blocked task_id-literal branch: task.task_id == {literal!r} (memorizes the eval set)"

    return True, ""


def task_id_literal_branch(code: str) -> "str | None":
    """Return the literal task id a comparison branches on, or ``None`` (6.4b).

    Flags ``task.task_id == 'foo'`` style branching on a LITERAL task id — a
    feature-routing reward-hack that memorizes the eval set. A branch on task
    FEATURES (e.g. ``'fever' in task.description``) is NOT flagged.

    Also flags the container-membership spelling
    ``task.task_id in ('task_5', 'task_12')`` / ``in {...}`` / ``in [...]`` /
    ``in {...: ...}``, whose string literals live inside a container operand
    rather than as a bare ``ast.Constant`` (#73) — the natural multi-id
    memorization form — plus the sibling spellings
    ``task.task_id.startswith('task_5')`` / ``.endswith(...)`` and
    ``match task.task_id: case 'task_5':``. Feature-neutral methods
    (``.lower()``, ``.split()``) and aliasing (``tid = task.task_id``) are NOT
    flagged (static-analysis limitation, pinned in tests/test_overfit.py).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            has_task_id = any(
                isinstance(o, ast.Attribute) and o.attr == "task_id" for o in operands
            )
            literal = _first_str_literal(operands)
            if has_task_id and literal is not None:
                return literal
        elif isinstance(node, ast.Call):
            # task.task_id.startswith('lit') / .endswith('lit' | ('a', 'b')).
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in ("startswith", "endswith")
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "task_id"
            ):
                literal = _first_str_literal(list(node.args))
                if literal is not None:
                    return literal
        elif isinstance(node, ast.Match):
            # match task.task_id: case 'lit': (subject may be a Tuple).
            subject = node.subject
            subjects = subject.elts if isinstance(subject, ast.Tuple) else [subject]
            if any(
                isinstance(s, ast.Attribute) and s.attr == "task_id" for s in subjects
            ):
                for case in node.cases:
                    for sub in ast.walk(case.pattern):
                        if (
                            isinstance(sub, ast.MatchValue)
                            and isinstance(sub.value, ast.Constant)
                            and isinstance(sub.value.value, str)
                        ):
                            return sub.value.value
    return None


def _first_str_literal(operands: list) -> "str | None":
    """First string literal among ``operands``, looking inside container literals.

    Returns the value of the first bare ``ast.Constant`` string operand, or the
    first string element inside a top-level ``ast.Tuple``/``ast.Set``/``ast.List``
    operand (the ``x in ('a', 'b')`` membership form), or the first string KEY of
    an ``ast.Dict`` operand (``x in {'a': 1}`` membership). Returns ``None`` if none.
    """
    for o in operands:
        if isinstance(o, ast.Constant) and isinstance(o.value, str):
            return o.value
        if isinstance(o, (ast.Tuple, ast.Set, ast.List)):
            for elt in o.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    return elt.value
        if isinstance(o, ast.Dict):
            for key in o.keys:
                if key is None:  # ** splat — no literal key
                    continue
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    return key.value
    return None


# Wall-clock bound (seconds) for a single helper smoke-test exec (mirrors the
# #48 pre_process bound): a healthy helper defines in milliseconds, so this
# generous default only fires on pathological module-level code (``while
# True`` / hanging import) that would otherwise wedge the event loop and every
# concurrent task.
_SMOKE_TEST_TIMEOUT_DEFAULT = 30.0


def smoke_test_function(
    name: str, source: str, *, timeout: float = _SMOKE_TEST_TIMEOUT_DEFAULT
) -> tuple[bool, str]:
    """Smoke test a library function by exec-ing it in an isolated namespace.

    Verifies:
    1. Source can be exec'd without errors (catches bad imports, undefined names)
    2. exec produces a callable named ``name``

    Does NOT call the function — just checks definition succeeds. The exec runs
    HOST-side in a daemon worker thread bounded by ``timeout`` wall-clock
    seconds; on overrun the helper is skipped like any other smoke failure
    (the abandoned thread cannot be killed — accepted #48-style residual).

    Returns:
        Tuple of (passed, error_message). error_message is empty if passed.
    """
    if not source or not source.strip():
        return False, "Empty source code"

    ns: dict = {}
    exc: dict[str, BaseException] = {}

    def _exec_source() -> None:
        try:
            exec(source, ns)  # noqa: S102
        except BaseException as e:  # noqa: BLE001 - SystemExit in model code
            # must surface as a diagnostic, not vanish with the worker thread.
            exc["e"] = e

    worker = threading.Thread(target=_exec_source, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return False, f"smoke test exceeded {timeout:.1f}s wall-clock"
    if "e" in exc:
        e = exc["e"]
        return False, f"exec failed: {type(e).__name__}: {e}"

    if name not in ns:
        defined = [k for k in ns if not k.startswith("_") and callable(ns.get(k))]
        if defined:
            return False, f"'{name}' not defined (found: {defined})"
        return False, f"'{name}' not defined by source"

    if not callable(ns[name]):
        return False, f"'{name}' is not callable ({type(ns[name]).__name__})"

    return True, ""


def _get_attribute_chain(node: ast.Attribute) -> Optional[str]:
    """Reconstruct dotted attribute chain from AST node, e.g. 'os.path.join'."""
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None
