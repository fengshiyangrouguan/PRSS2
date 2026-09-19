"""Regression tests for audit fixes in meta_n/core/llm_helpers.py.

Finding 33 (dead-code): the module defined an unused
``logger = logging.getLogger(__name__)`` binding, supported by a sole
``import logging`` that was otherwise unreferenced. Both are dead and were
removed. These tests are offline / LLM-free — they only inspect module
attributes and the module source text.
"""

from __future__ import annotations

import meta_n.core.llm_helpers as llm_helpers


def test_no_dead_logger_binding():
    """Finding 33: the unused module-level ``logger`` binding is gone."""
    assert not hasattr(llm_helpers, "logger"), (
        "llm_helpers should not define an unused module-level `logger`"
    )


def test_logging_module_not_imported():
    """Finding 33: ``import logging`` (only there to support the dead logger)
    is removed, so the module namespace no longer exposes ``logging``."""
    assert not hasattr(llm_helpers, "logging"), (
        "llm_helpers should no longer import the unused `logging` module"
    )


def test_helpers_still_importable():
    """Smoke: the public factory functions remain importable and callable."""
    assert callable(llm_helpers.make_llm_func)
    assert callable(llm_helpers.make_llm_func_from_config)
    # LLMUsageTracker still works (offline, no LLM).
    tracker = llm_helpers.LLMUsageTracker()
    tracker.record(10, prompt_tokens=4, completion_tokens=6)
    assert tracker.total_tokens == 10
    assert tracker.call_count == 1
