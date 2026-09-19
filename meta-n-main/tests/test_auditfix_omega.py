"""Regression tests for audit findings 19 and 31 (meta_n/core/omega.py).

LLM-free / offline: these exercise pure string-formatting and module-import
behavior only — no LM Studio, no Docker, no network.
"""

from __future__ import annotations

import meta_n.core.omega as omega_mod
from meta_n.core.omega import OmegaEngine


def _engine() -> OmegaEngine:
    # OmegaEngine only needs an llm_client object stored as an attribute; none of
    # the methods under test ever call it, so None is fine for these unit tests.
    return OmegaEngine.__new__(OmegaEngine)


# ---------------------------------------------------------------------------
# Finding 31 — dead module-global ``console = Console()`` and its dead import.
# ---------------------------------------------------------------------------

def test_finding31_no_dead_console_global():
    # The unused Rich console global is removed.
    assert not hasattr(omega_mod, "console"), (
        "omega module still defines a dead `console` global"
    )


def test_finding31_no_dead_console_import():
    # The import existed only to feed the dead global; it should be gone too.
    assert not hasattr(omega_mod, "Console"), (
        "omega module still imports the now-unused `Console`"
    )


# ---------------------------------------------------------------------------
# Finding 19 — absolute-room fallback must not assume a 1.0 ceiling on a
# continuous / negative-capable score_scale.
# ---------------------------------------------------------------------------

def test_finding19_negative_scale_no_phantom_1_0_ceiling():
    """On a continuous/negative scale at best-known, the fallback table must not
    invent room toward a 'perfect 1.0' that does not exist."""
    eng = _engine()
    # All tasks at best-known (gap == 0 → triggers the absolute-room fallback),
    # with a negative score that no [0,1] scale could ever produce.
    cur = {"sr_task": -5.0}
    best = {"sr_task": -5.0}
    _summary, headroom = eng._format_objective_and_headroom(
        traces=[], current_scores=cur, archive_best_scores=best, pass_at_1=0.0,
    )
    # The fallback table still fires (lowest-scoring task surfaced)...
    assert "sr_task" in headroom
    assert "-5.000" in headroom
    # ...but NOT with the phantom 1.0 ceiling narrative / column.
    assert "perfect 1.0" not in headroom
    assert "room->1.0" not in headroom
    # And it must not render the bogus magnitude 1.0 - (-5) = 6.0.
    assert "6.000" not in headroom


def test_finding19_unit_scale_byte_identical():
    """On the default [0,1] scale the fallback table is unchanged byte-for-byte."""
    eng = _engine()
    cur = {"task_a": 0.2, "task_b": 0.9}
    best = {"task_a": 0.2, "task_b": 0.9}  # all at best-known → fallback branch
    _summary, headroom = eng._format_objective_and_headroom(
        traces=[], current_scores=cur, archive_best_scores=best, pass_at_1=0.5,
    )
    expected = (
        "## Opportunity — lowest-scoring tasks (most absolute room)\n"
        "All tasks are at the best-known score; these have the most "
        "room toward a perfect 1.0:\n"
        "\n"
        "| task | now | room->1.0 |\n"
        "|------|-----|----------|\n"
        "| task_a | 0.200 | 0.800 |\n"
        "| task_b | 0.900 | 0.100 |\n"
    )
    assert headroom == expected
