"""Byte-identity gate: all new flags OFF == HEAD orchestration + Ω prompt.

The Stage-0b/2/3 substrate (SelfRepairEvent sidecars, S0.6 de-reap) and the
behavioral flags it precedes are required to leave the default path BYTE-IDENTICAL
to HEAD. This re-runs the deterministic capture (the seeded fake-solver micro-run
+ the single depth-3 Ω prompt render) and asserts it equals the goldens captured
on pristine HEAD (``orchestration_golden.json`` / ``omega_prompt_golden.txt``).

A failure here means a substrate change leaked into the all-flags-OFF orchestration
or the rendered Ω prompt — a hard merge blocker.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent


def _load_capture():
    spec = importlib.util.spec_from_file_location(
        "capture_stage23_golden", GOLDEN_DIR / "capture_stage23_golden.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_orchestration_golden_byte_identical(tmp_path):
    cap = _load_capture()
    captured = asyncio.run(cap._run_orchestration_golden(tmp_path / "run"))
    new = json.dumps(captured, indent=2, sort_keys=True)
    stored = (GOLDEN_DIR / "orchestration_golden.json").read_text()
    assert new == stored, (
        "orchestration golden drifted — a flags-OFF change leaked into the "
        "archive/summary.json structure"
    )


def test_omega_prompt_golden_byte_identical():
    cap = _load_capture()
    new = cap._omega_prompt_golden()
    stored = (GOLDEN_DIR / "omega_prompt_golden.txt").read_text()
    assert new == stored, (
        "omega-prompt golden drifted — the downstream-feedback param (absent) "
        "changed the rendered prompt"
    )
