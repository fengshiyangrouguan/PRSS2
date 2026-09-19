"""Regression tests for audit fixes in meta_n/utils/list_azure_deployments.py.

Offline / LLM-free. Covers:

  Finding 16 (dead-code): `_is_chat_capable` was defined but never called;
  the live chat filter is `_looks_chat`. The dead helper is removed.
"""

import meta_n.utils.list_azure_deployments as laz


def test_dead_helper_is_chat_capable_removed():
    """Finding 16: the unused `_is_chat_capable` helper must be gone.

    FAILS on the original code (the attribute exists), PASSES after the
    dead-code removal.
    """
    assert not hasattr(laz, "_is_chat_capable")


def test_live_chat_filter_still_works():
    """Smoke: the module still imports and `_looks_chat` (the actually-used
    filter) behaves as before."""
    assert laz._looks_chat({"id": "gpt-4o", "capabilities": {"chat_completion": True}})
    assert laz._looks_chat({"id": "gpt-5-preview"})  # id-prefix fallback
    assert not laz._looks_chat({"id": "text-embedding-3-small"})
    assert not laz._looks_chat({"id": "gpt-4o-audio"})
