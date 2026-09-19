"""Stage 1 byte-identity GATE (R1 error-hints / R2 preamble, both default OFF).

These tests recompute the HEAD-captured goldens from
``tests/stage1_golden_harness.py`` and assert byte-identity. At HEAD they pass
trivially (the render code is unchanged). After the R1/R2 edits land they are the
merge blocker that proves: with both flags OFF, the rendered observation and the
system prompt are byte-identical to current HEAD.

Self-contained — no experiments corpus, no LLM, no Docker — so this gate runs on
every checkout (unlike the Stage 0 golden gate, which skips when the corpus is
absent).
"""

from __future__ import annotations

import json

import pytest

from tests import stage1_golden_harness as H


def test_goldens_were_captured():
    assert (H.GOLDEN_DIR / "MANIFEST.json").exists(), (
        "run `python -m tests.stage1_golden_harness capture` first"
    )


@pytest.mark.parametrize("name", sorted(H.compute_goldens().keys()))
def test_golden_byte_identical(name):
    """Every flag-OFF render artifact matches the HEAD baseline byte-for-byte."""
    mismatches = [m for m in H.verify() if m.startswith(name)]
    assert not mismatches, "\n".join(mismatches)


def test_no_mismatches_overall():
    mismatches = H.verify()
    assert not mismatches, "\n".join(mismatches)


def test_classify_error_keys_cover_actionable_hint_set():
    """The ERROR_HINTS taxonomy (R1) must key off the REAL classify_error strings.

    Captured baseline of which class each fixture maps to; this asserts the
    actionable subset the R1 hint map will cover is exactly the classes whose
    fixtures are present, and that the four non-actionable classes resolve to
    their documented strings (so R1 can emit "" for them).
    """
    manifest = json.loads((H.GOLDEN_DIR / "MANIFEST.json").read_text())
    classify_map = manifest["classify_error_map"]
    # Actionable classes that WILL get a hint (R1).
    actionable = {
        "Numeric instability",
        "Dependency error",
        "Timeout",
        "Constraint violation",
        "Indexing error",
        "Syntax error",
        "Format/parse error",
    }
    # Non-actionable classes that MUST get no hint (R1 emits "").
    non_actionable = {
        "Unknown error",
        "Runtime error",
        "Turn starvation",
        "Environment fault",
    }
    observed = set(classify_map.values())
    # Every observed class is one of the known 11; nothing unexpected leaked in.
    assert observed <= (actionable | non_actionable), observed
    # The fixtures collectively exercise the full actionable + non-actionable set.
    assert actionable <= observed
    assert non_actionable <= observed
