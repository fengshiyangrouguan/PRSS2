"""_outer_token_attribution — the shared builtin outer-ledger helpers (§7.11).

Both ``BuiltinBackend`` and ``BuiltinTBBackend`` recover the agent-token split
from the outer ``LLMClient.cumulative_usage`` delta via these single-sourced free
functions. Locks the snapshot / clamped-delta arithmetic and that BOTH backends'
``_usage_delta`` methods (which now delegate here) return the SAME split for the
same ledger delta. Install-free: stdlib + meta_n only.
"""

from __future__ import annotations

from meta_n.core.external_agents._outer_token_attribution import (
    snapshot_usage,
    usage_delta,
)
from meta_n.core.external_agents.backends.builtin import BuiltinBackend
from meta_n.core.external_agents.backends.builtin_tb import BuiltinTBBackend


class _Client:
    def __init__(self, usage):
        self.cumulative_usage = usage


def test_snapshot_returns_shallow_copy():
    usage = {"prompt": 10, "completion": 5, "total": 15, "cached": 2, "calls": 1}
    snap = snapshot_usage(_Client(usage))
    assert snap == usage
    # Mutating the live ledger afterward does not change the snapshot.
    usage["prompt"] = 999
    assert snap["prompt"] == 10


def test_snapshot_missing_ledger_is_zeroed():
    assert snapshot_usage(object()) == {
        "prompt": 0, "completion": 0, "total": 0, "calls": 0, "cached": 0,
    }
    assert snapshot_usage(_Client("not-a-dict"))["total"] == 0


def test_usage_delta_clamped_at_zero():
    before = {"prompt": 100, "completion": 50, "total": 150, "cached": 10, "calls": 3}
    after = {"prompt": 130, "completion": 70, "total": 200, "cached": 12, "calls": 5}
    assert usage_delta(before, after) == (30, 20, 50, 2, 2)
    # A racy negative delta (after < before) clamps to 0, never leaks negative.
    assert usage_delta(after, before) == (0, 0, 0, 0, 0)


def test_both_builtin_backends_delegate_to_same_split():
    """``BuiltinBackend._usage_delta`` and ``BuiltinTBBackend._usage_delta`` must
    produce the IDENTICAL split for the same ledger delta (the duplication this
    shared helper removed)."""
    before = {"prompt": 1, "completion": 1, "total": 2, "cached": 0, "calls": 1}
    after = {"prompt": 40, "completion": 25, "total": 65, "cached": 3, "calls": 4}

    b = BuiltinBackend.__new__(BuiltinBackend)
    b.llm_client = _Client(after)
    b_split = b._usage_delta(before)

    tb = BuiltinTBBackend.__new__(BuiltinTBBackend)
    tb._llm_client = _Client(after)
    tb_split = tb._usage_delta(before)

    assert b_split == tb_split == (39, 24, 63, 3, 3)
