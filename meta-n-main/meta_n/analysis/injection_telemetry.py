"""Helper call-rate & injection-diversity telemetry (roadmap v2 §3.5).

Post-hoc, **read-only** analysis over an evolutionary run directory. It produces
the measurement the telemetry-gated injection tuning (4.3 / 4.4) keys on: are the
Ω-generated ``code_library`` helpers actually CALLED by the solver scripts, or
dead weight prepended to every script? Plus injection-diversity signals
(helper-name Jaccard and rationale n-gram cosine between consecutive layers, and
a run-level ``redundant_generation_rate``) for 3.5 / 6.5.

Zero runtime blast radius: this module imports only the standard library and is
never imported by the orchestrator. Run it after a run completes, e.g.::

    from meta_n.analysis.injection_telemetry import run_injection_report
    report = run_injection_report("experiments/azure_gpt52/meta_n/cobench_s42")
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: The prefix marker the solver scripts carry between the injected code-library
#: and the solver-authored body (see meta_layer.py / omega._strip_library_prefix
#: / agentic_solver.py). The solver BODY is everything after the LAST occurrence.
LIB_END_MARKER = "# --- end injected code library ---"

#: Staged bash/python helper file convention. Two call-site forms, both matched
#: (mirrors the canonical ``scan_helper_calls`` file pattern in meta_layer.py):
#:   * inline:  ``/tmp/_lib_<name>.py`` / ``/tmp/_lib_<name>.sh``
#:   * spine:   ``./helpers/<name>.sh`` / ``./helpers/_lib_<name>.py`` — the
#:              ``_lib_`` prefix is OPTIONAL under ``helpers/`` and BOTH ``.py``
#:              and ``.sh`` are staged (injection.py:210,223).
#: The literal ``.`` is escaped so it does not over-match.
_LIB_FILE_RE = r"(?:helpers/(?:_lib_)?|_lib_){name}\.(?:py|sh)"


# --------------------------------------------------------------------------- #
# Solver-body extraction + call detection
# --------------------------------------------------------------------------- #

def solver_body(script: str) -> str:
    """Return the solver-authored region of ``script``: everything AFTER the
    injected code-library prefix marker. With no marker (no library prepended),
    the whole script is the solver body."""
    idx = script.rfind(LIB_END_MARKER)
    if idx == -1:
        return script
    return script[idx + len(LIB_END_MARKER):]


def _python_calls(body: str) -> Counter:
    """Count called names (``ast.Call`` over ``Name``/``Attribute`` funcs) in a
    python solver body. Empty Counter if the body does not parse (bash scripts,
    partial snippets) — the bash regime is handled separately by a file regex."""
    counts: Counter = Counter()
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return counts
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                counts[fn.id] += 1
            elif isinstance(fn, ast.Attribute):
                counts[fn.attr] += 1
    return counts


def _parses_as_python(body: str) -> bool:
    """Whether ``body`` is syntactically valid Python (a bash script is not)."""
    try:
        ast.parse(body)
        return True
    except SyntaxError:
        return False


def helper_called_in(
    name: str, body: str, py_calls: Optional[Counter] = None,
    *, body_is_python: Optional[bool] = None,
) -> bool:
    """Whether helper ``name`` is invoked in this solver body, in either regime:

    - **python**: an ``ast.Call`` to ``name(...)`` somewhere in the body — AST,
      so the helper's own ``def name`` / comments / docstrings never count.
      A name shadowed by a local ``def name`` is EXCLUDED: Python scoping
      rebinds every call in the script to the solver's OWN definition, so
      counting it is a false positive (observed on equitable's
      ``get_swap_delta``; mirrors ``scan_helper_calls``' def-shadow guard).
    - **staged-file**: the call-site path ``(_lib_)?<name>.{py,sh}`` (inline
      ``/tmp/_lib_name.py`` or spine ``./helpers/name.sh``); path-anchored, so
      counted unconditionally — a path is not a Python rebind.
    - **bash**: a bash body never parses as Python (empty ``py_calls`` above),
      so the AST channel is blind to a helper sourced/prepended and invoked by
      bare name. For such bodies fall back to the canonical ``\\bname\\b`` form.

    ``body_is_python`` lets a caller that checks many helpers against one body
    (``candidate_helper_stats``) pass the parse verdict once instead of
    re-running ``ast.parse`` per helper; ``None`` (the default) computes it
    on demand with identical results.
    """
    if py_calls is None:
        py_calls = _python_calls(body)
    if py_calls.get(name, 0) > 0 and not re.search(
        rf"\bdef\s+{re.escape(name)}\b", body
    ):
        return True
    if re.search(_LIB_FILE_RE.format(name=re.escape(name)) + r"\b", body) is not None:
        return True
    if not py_calls:
        if body_is_python is None:
            body_is_python = _parses_as_python(body)
        if not body_is_python and re.search(rf"\b{re.escape(name)}\b", body) is not None:
            return True
    return False


# --------------------------------------------------------------------------- #
# Diversity primitives
# --------------------------------------------------------------------------- #

def jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two sets; two empty sets are identical (1.0)."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _ngrams(text: str, n: int = 3) -> Counter:
    toks = re.findall(r"\w+", (text or "").lower())
    if not toks:
        return Counter()
    if len(toks) < n:
        return Counter([" ".join(toks)])
    return Counter(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))


def cosine(a: Counter, b: Counter) -> float:
    """Cosine similarity of two n-gram count vectors."""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# Per-candidate + run-level walkers
# --------------------------------------------------------------------------- #

def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("injection_telemetry: failed to read %s: %s", p, e)
        return None


def _injected_layers(candidate_dir: Path) -> list[dict]:
    """The candidate's injected_code_d{N}.json blobs, ordered by depth."""
    layers = []
    for icp in candidate_dir.glob("injected_code_d*.json"):
        m = re.search(r"d(\d+)", icp.name)
        ic = _load_json(icp)
        if m and ic is not None:
            layers.append((int(m.group(1)), ic))
    layers.sort(key=lambda x: x[0])
    return [ic for _, ic in layers]


def candidate_helper_stats(candidate_dir: Path) -> dict:
    """Per-helper call stats for one ``archive/<id>/`` candidate dir.

    Returns ``{helper_name: {kind, call_rate, tasks_called, n_tasks}}`` over the
    helpers advertised across ALL of the candidate's injected layers, checked
    against every ``traces/<task>.json`` solver body.
    """
    advertised: dict[str, str] = {}
    for ic in _injected_layers(candidate_dir):
        for n in (ic.get("code_library") or {}):
            advertised[n] = "python"
        for n in (ic.get("code_library_bash") or {}):
            advertised.setdefault(n, "bash")

    bodies = []
    traces_dir = candidate_dir / "traces"
    if traces_dir.exists():
        for tf in sorted(traces_dir.glob("*.json")):
            td = _load_json(tf) or {}
            body = solver_body(td.get("script", "") or "")
            pc = _python_calls(body)
            # A non-empty call Counter proves the body parsed as Python; only
            # run the parse check once per body when it is empty, so the bash
            # fallback below never re-parses per advertised helper.
            bodies.append((tf.stem, body, pc, bool(pc) or _parses_as_python(body)))

    n_tasks = len(bodies)
    stats = {}
    for name, kind in advertised.items():
        called = [
            tid for tid, body, pc, is_py in bodies
            if helper_called_in(name, body, pc, body_is_python=is_py)
        ]
        stats[name] = {
            "kind": kind,
            "tasks_called": called,
            "n_tasks": n_tasks,
            "call_rate": (len(called) / n_tasks) if n_tasks else 0.0,
        }
    return stats


def candidate_layer_diversity(candidate_dir: Path) -> dict:
    """Helper-name Jaccard + rationale n-gram cosine between consecutive layers."""
    layers = _injected_layers(candidate_dir)
    keysets = [
        set((ic.get("code_library") or {}).keys())
        | set((ic.get("code_library_bash") or {}).keys())
        for ic in layers
    ]
    rationales = [ic.get("rationale", "") or "" for ic in layers]
    # Only score transitions where at least one layer ships a helper. A
    # both-empty pair has jaccard(∅,∅)=1.0, which would mis-report a run that
    # generated ZERO helpers (the local-gemma 0/102 regime) as MAXIMAL redundant
    # re-emission — the exact opposite of the docstring's "every layer re-emits
    # the same helper set" (finding 24). Excluding them keeps "no helpers
    # generated" provably distinct from "same helpers re-emitted".
    jaccards = [
        jaccard(k1, k2)
        for k1, k2 in zip(keysets, keysets[1:])
        if (k1 or k2)
    ]
    cosines = [cosine(_ngrams(r1), _ngrams(r2)) for r1, r2 in zip(rationales, rationales[1:])]
    return {
        "n_layers": len(layers),
        "helper_name_jaccards": jaccards,
        "rationale_cosines": cosines,
    }


def run_injection_report(run_dir) -> dict:
    """Walk ``<run_dir>/archive/`` and aggregate helper call-rate + diversity.

    Returns a run-level summary plus the per-candidate breakdown. ``overall_call_rate``
    is the fraction of (advertised-helper, candidate) pairs that were called on at
    least one task — the headline number 4.3/4.4 gate on. ``redundant_generation_rate``
    is the mean consecutive-layer helper-name Jaccard over helper-bearing
    transitions only (both-empty pairs are excluded, so 1.0 = every layer
    re-emits the same helper set, and ``None`` = no helpers generated at all).
    """
    archive_dir = Path(run_dir) / "archive"
    candidates: dict[str, dict] = {}
    diversity: dict[str, dict] = {}
    if archive_dir.exists():
        for cdir in sorted(archive_dir.iterdir()):
            if not cdir.is_dir() or not (cdir / "summary.json").exists():
                continue
            candidates[cdir.name] = candidate_helper_stats(cdir)
            diversity[cdir.name] = candidate_layer_diversity(cdir)

    advertised_total = sum(len(s) for s in candidates.values())
    called_total = sum(
        1 for s in candidates.values() for st in s.values() if st["tasks_called"]
    )
    all_jaccards = [j for d in diversity.values() for j in d["helper_name_jaccards"]]
    all_cosines = [c for d in diversity.values() for c in d["rationale_cosines"]]

    return {
        "advertised_helpers": advertised_total,
        "called_helpers": called_total,
        "overall_call_rate": (called_total / advertised_total) if advertised_total else None,
        "redundant_generation_rate": (sum(all_jaccards) / len(all_jaccards)) if all_jaccards else None,
        "mean_rationale_cosine": (sum(all_cosines) / len(all_cosines)) if all_cosines else None,
        "candidates": candidates,
        "diversity": diversity,
    }


# --------------------------------------------------------------------------- #
# CLI entry point (stdlib-only, preserving the module's import contract)
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m meta_n.analysis.injection_telemetry",
        description="Helper call-rate & injection-diversity report over an "
                    "evolutionary run dir (read-only; roadmap v2 §3.5).",
    )
    ap.add_argument("run_dir", help="run directory containing archive/")
    ap.add_argument("--output", default=None,
                    help="also write the JSON report to this path")
    args = ap.parse_args(argv)
    report = run_injection_report(args.run_dir)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
