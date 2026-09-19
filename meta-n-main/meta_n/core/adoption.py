"""Failure classification and helper-adoption attribution (split from meta_layer.py).

F257: relocated verbatim from ``meta_n.core.meta_layer`` — ``classify_error``
feeds the Ω prompt sections (omega.py) and ``scan_helper_calls`` /
``populate_adoption_fields`` are the S0.2 adoption populators shared by the
native and spine paths. ``meta_layer.py`` remains the import hub and
re-exports every name defined here.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import TYPE_CHECKING, Optional

from meta_n.core.code_library import validate_library_function

if TYPE_CHECKING:
    from meta_n.core.base_executor import BaseExecutor
    from meta_n.core.meta_layer import Trace

logger = logging.getLogger(__name__)


def classify_error(trace: "Trace") -> str:
    """Classify a failure trace into a unified failure class (6.3).

    Precedence: the STRUCTURED ``terminated_by`` signal (spine / native
    agentic) wins over keyword sniffing — e.g. a MAX_TURNS run whose error
    text mentions 'timeout' is still turn-starvation. Used to make Ω's
    channel guidance failure-aware (turn-starvation → raise turns; numeric →
    guarded math; format → output coercion; etc.).

    S0.1: extracted verbatim from ``OmegaEngine._classify_error`` so the Ω
    prompt sections that render it (omega.py:558/703/716/722/832/834) have a
    single source of truth. ``OmegaEngine._classify_error`` is kept as a
    ``staticmethod(classify_error)`` alias so all 8 callers resolve unchanged.
    """
    tb = (getattr(trace, "terminated_by", "") or "").lower()
    text = ((trace.error_summary or "") + " " + (trace.stderr or "")).lower()
    # Structured signal first.
    # Native suffixed cap labels deliberately CONTAIN their base substring
    # ("max_turns_unconfirmed_complete" matches here); the
    # "token_budget" / "spend_budget" labels (plain or "_unconfirmed_complete"
    # suffixed) have no structured branch and fall through to the text scan.
    if "max_turns" in tb:
        return "Turn starvation"
    if "env_error" in tb:
        return "Environment fault"
    if "timeout" in tb:  # structured signal wins even with empty error text
        return "Timeout"
    # Text fallback, reached only when terminated_by carried no structured signal.
    if "max_turns" in text or "max turns" in text:
        return "Turn starvation"
    if not text.strip():
        return "Unknown error"
    if "timeout" in text or "timed out" in text:
        return "Timeout"
    # Numeric instability — precise tokens (avoid 'inf' matching 'infeasible').
    if any(w in text for w in ("nan", "zerodivision", "divide by zero",
                                "overflowerror", "-1e9", "-1e+09", "not finite")):
        return "Numeric instability"
    if "import" in text or "modulenotfounderror" in text or "no module" in text:
        return "Dependency error"
    if any(w in text for w in ("constraint", "infeasible", "violat", "capacity",
                                "overlap", "boundary", "out of bound", "exceed")):
        return "Constraint violation"
    if any(w in text for w in ("index", "indexerror", "keyerror", "key error")):
        return "Indexing error"
    if any(w in text for w in ("syntax", "indent", "unexpected")):
        return "Syntax error"
    if any(w in text for w in ("no executable code", "produced no executable",
                                "could not parse", "malformed", "without producing code")):
        return "Format/parse error"
    return "Runtime error"


def scan_helper_calls(
    script: str, names: list[str]
) -> tuple[list[str], dict[str, int]]:
    """Detect which injected helpers a single solver ``script`` actually CALLS.

    S0.2: the single source of truth for the regex previously inlined at
    omega.py:453-476 — shared by ``OmegaEngine._helper_usage_section`` (which
    renders the byte-identity-gated ``## Helper Usage`` prompt section) and the
    fresh-trace adoption populator. For each helper ``name`` the script counts
    as a CALL when:

      * the bare word-boundary form ``\\bname\\b`` appears AND is NOT shadowed by
        a local ``def <name>`` (Python scoping rebinds every reference in that
        script to the solver's OWN definition — counting it would be a false
        positive, observed on equitable's ``get_swap_delta``), OR
      * a staged-helper FILE path appears (path-anchored, counted
        UNCONDITIONALLY — a path is not a python rebind, so the def-shadow
        exclusion must not drop it), in EITHER call-site form:
        the spine ``helpers/(_lib_)?<name>.{py,sh}`` form OR the native inline
        ``_lib_<name>.{py,sh}`` form (``/tmp/_lib_<name>.py`` — the
        code_library ``lib_path_fmt`` default the native bash path stages).

    Mirrors ``telemetry.attribute_utilities`` so the single-shot DEAD detector
    and the spine attribution agree.

    Args:
        script: the SOLVER script with any injected library prefix ALREADY
            stripped (see :func:`_strip_library_prefix_for_scan`).
        names: candidate helper names to look for.

    Returns:
        ``(called, counts)`` — ``called`` is the list of names judged called
        (in ``names`` order); ``counts`` maps each called name to its match
        count. ``counts`` is informational; the byte-identity-gated prompt
        section consumes only the ``called`` membership.
    """
    called: list[str] = []
    counts: dict[str, int] = {}
    for name in names:
        if not name:
            continue
        pat = re.compile(rf"\b{re.escape(name)}\b")
        def_pat = re.compile(rf"\bdef\s+{re.escape(name)}\b")
        file_pat = re.compile(rf"(?:helpers/(?:_lib_)?|_lib_){re.escape(name)}\.(?:py|sh)")
        word_hit = bool(pat.search(script)) and not def_pat.search(script)
        file_hits = file_pat.findall(script)
        if word_hit or file_hits:
            called.append(name)
            counts[name] = (
                len(pat.findall(script)) if word_hit else 0
            ) + len(file_hits)
    return called, counts


def _build_deploy_wrapper(helper_name: str, helper_source: str) -> Optional[str]:
    """Build a deterministic ``solve`` that calls ``helper_name`` (P1a fallback).

    Parses ``helper_source`` to recover the helper's parameter names so the
    generated wrapper passes ONLY matching kwargs — CO-Bench calls
    ``solve(**instance)`` where ``instance`` may carry keys the helper does not
    accept (a stray key would raise ``TypeError`` ⇒ a deployed-but-broken solve
    scoring 0). If the helper's signature has a ``**kwargs`` catch-all, all
    kwargs are forwarded.

    Returns ``None`` when the helper ``def`` cannot be located/parsed, so the
    caller can fall back to the authored script (never ship a broken wrapper).
    """
    try:
        tree = ast.parse(helper_source)
    except SyntaxError:
        return None
    func = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == helper_name:
            func = node
            break
    if func is None:
        return None
    a = func.args
    posonly = [arg.arg for arg in a.posonlyargs]
    # Params forwardable BY KEYWORD: positional-or-keyword (``args``) + kwonly.
    # Positional-only params (``/``-marked) are NEVER keyword-forwardable —
    # passing them by keyword raises ``TypeError`` (#63), or, with a ``**kwargs``
    # catch-all, silently pollutes it — so they are forwarded positionally below.
    named = [arg.arg for arg in (list(a.args) + list(a.kwonlyargs))]
    has_var_keyword = a.kwarg is not None

    if has_var_keyword:
        # Forward everything except the positional-only names (those go
        # positionally); ``**kw`` verbatim when there are no posonly params
        # (byte-identical to the legacy wrapper).
        if posonly:
            kw_expr = (
                "{k: v for k, v in kw.items() if k not in "
                + repr(set(posonly)) + "}"
            )
        else:
            kw_expr = "kw"
    else:
        allowed = "{" + ", ".join(repr(p) for p in named) + "}"
        kw_expr = f"{{k: v for k, v in kw.items() if k in {allowed}}}"

    if not posonly:
        # Legacy fast path — preserved byte-for-byte.
        return f"def solve(**kw):\n    return {helper_name}(**{kw_expr})\n"

    # Forward positional-only params POSITIONALLY, in declaration order, for the
    # contiguous prefix that ``instance`` actually provides (a missing one stops
    # forwarding so a later value is never mis-bound to an earlier slot; any
    # required-but-absent param then surfaces the helper's own TypeError, exactly
    # as a missing keyword param already does).
    posonly_repr = "[" + ", ".join(repr(p) for p in posonly) + "]"
    return (
        "def solve(**kw):\n"
        f"    _po_names = {posonly_repr}\n"
        "    _po = []\n"
        "    for _k in _po_names:\n"
        "        if _k in kw:\n"
        "            _po.append(kw[_k])\n"
        "        else:\n"
        "            break\n"
        f"    return {helper_name}(*_po, **{kw_expr})\n"
    )


def _strip_library_prefix_for_scan(script: str) -> str:
    """Strip the injected code-library prefix so downstream consumers (helper
    scans, debug context, Ω prompt rendering) see only the solver's OWN code.
    Single source of truth for the marker-strip logic."""
    marker = "# --- end injected code library ---"
    if marker in script:
        return script[script.index(marker) + len(marker):].lstrip()
    return script


def populate_adoption_fields(
    trace: "Trace",
    *,
    command_count: int,
    merged_code_library: Optional[dict[str, str]] = None,
    merged_code_library_bash: Optional[dict[str, str]] = None,
    executor: "BaseExecutor | None" = None,
    solver_language: str = "python",
) -> None:
    """Populate the S0.2 helper-adoption fields on a FRESH trace, in place.

    Non-behavioral: only the additive attribution fields are touched. The helper
    scan runs ONLY when live helpers are staged (``utilities_available`` ends up
    non-empty); otherwise ``utilities_called`` stays ``None`` (unmeasurable), NOT
    a misleading measured-zero ``[]``. The CO-Bench demoted path (the merged
    Python library is zeroed upstream and there are no bash helpers) therefore
    keeps ``None``.

    ``command_count`` is the number of ``executor.execute`` invocations behind
    this trace (1 for a single-shot native solve, the executed-turn count for an
    agentic solve, ``1 + retries`` for a self-debug retry loop).

    ``solver_language`` selects the gate mirroring the advertise+stage path that
    was ACTUALLY used, so ``utilities_available`` equals what the solver was
    shown+staged (see below).
    """
    trace.command_count = command_count
    # R2-CS-2: ``utilities_available`` must equal what the solver was ACTUALLY
    # shown. advertise and stage both skip Python helpers failing
    # ``validate_library_function``, so a value-verified-but-validate-skipped
    # helper is never offered to the solver yet would sit in
    # ``utilities_available`` — over-barring its task under ``verified_code`` and
    # deflating S0.2 "available-but-not-called" telemetry. Filter the Python names
    # through the SAME gate the solver's language actually applied:
    #   * python solver → ``for_advertising=True`` (parity with
    #     format_python_library_descriptions:639 / prepend_python_library, which
    #     under a sandboxed executor bar a blocklist-flagged helper).
    #   * bash solver   → NO ``for_advertising`` (parity with the audit-#66
    #     by-design exception: format_bash_library_descriptions:732 /
    #     prepend_bash_library:868 advertise+stage the sys-using CLI form
    #     ``python3 ./helpers/_lib_<name>.py`` WITHOUT the blocklist, so a
    #     sys/os-using helper IS shown+staged and must count as available —
    #     otherwise a LIVE-helper bash trace deflates or collapses to
    #     ``utilities_called=None``, conflating it with the CO-Bench demoted path).
    # bash helpers (``merged_code_library_bash``) count ONLY for a bash solver
    # (format_bash_library_descriptions shows them unconditionally; the python
    # advertise/stage path never shows or stages them). On the demoted/CO-Bench
    # default path ``merged_code_library`` is empty, so this is byte-identical
    # (no helpers) for either language.
    for_advertising = solver_language != "bash"
    py_names = [
        name
        for name, source in (merged_code_library or {}).items()
        if validate_library_function(
            name, source, executor=executor, for_advertising=for_advertising
        )
    ]
    bash_names = (
        set(merged_code_library_bash or {}) if solver_language == "bash" else set()
    )
    names = sorted(set(py_names) | bash_names)
    if not names:
        # No live helpers staged → leave utilities_called == None (unmeasurable),
        # NOT a measured-zero []. Demoted/CO-Bench path lands here.
        return
    trace.utilities_available = names
    solver_script = _strip_library_prefix_for_scan(trace.script or "")
    called, counts = scan_helper_calls(solver_script, names)
    trace.utilities_called = called
    trace.utilities_call_counts = counts
