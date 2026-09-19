"""Shared outer-token usage-ledger helpers for the two builtin backends.

``BuiltinBackend`` (host workspace) and ``BuiltinTBBackend`` (terminal-bench
container) both run meta-n's native ``Layer1Solver`` to AUTHOR a solution, which
spends tokens on the OUTER ``LLMClient``. Neither agent runs an inner LLM, so the
"agent" token attribution is recovered from the outer client's cumulative-usage
ledger delta across the authoring call. The two backends extend different bases
(``AgentBackend`` vs ``_ExternalTBBackend``), so there is no natural shared
*class* home; these free functions (no inheritance) single-source the identical
snapshot + clamped-delta arithmetic instead.

Pure-import: stdlib only, no ``openhands`` / ``terminal_bench`` / ``docker``.
"""

from __future__ import annotations

__all__ = ["snapshot_usage", "usage_delta"]

#: The ledger keys both backends read off ``LLMClient.cumulative_usage``.
_USAGE_KEYS = ("prompt", "completion", "total", "cached", "calls")


def snapshot_usage(llm_client: object) -> dict[str, int | float]:
    """Snapshot the outer ``LLMClient.cumulative_usage`` ledger, or zeros.

    Returns a shallow copy so a post-run delta is computed against a stable
    baseline even if another concurrent task mutates the live dict in between
    (the ledger is shared under ``--parallel`` > 1).

    Args:
        llm_client: The outer client exposing a ``cumulative_usage`` dict (any
            object; a missing / non-dict ledger yields the all-zero baseline).

    Returns:
        A shallow copy of the usage dict, or a zeroed ``{prompt, completion,
        total, cached, calls}`` baseline when the ledger is unavailable.
    """
    usage = getattr(llm_client, "cumulative_usage", None)
    if not isinstance(usage, dict):
        return {"prompt": 0, "completion": 0, "total": 0, "calls": 0, "cached": 0}
    return dict(usage)


def usage_delta(
    before: dict[str, int | float],
    after: dict[str, int | float],
) -> tuple[int, int, int, int, int]:
    """Compute the prompt/completion/total/cached/calls delta ``after - before``.

    The native solvers return only a total token count, but the outer client
    accumulates the full prompt/completion/cached/calls breakdown on every
    ``complete()``. The delta across the authoring call recovers that split for
    the telemetry row. Each component is clamped at ``0`` so a racy negative delta
    (shared ledger under ``--parallel``) never leaks a negative count.

    Args:
        before: The :func:`snapshot_usage` taken before the authoring call.
        after: The :func:`snapshot_usage` taken after the authoring call.

    Returns:
        ``(prompt_tokens, completion_tokens, total_tokens, cached_tokens,
        calls)`` — each ``max(0, after[k] - before[k])``.
    """

    def _d(key: str) -> int:
        return max(0, int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0))

    return _d("prompt"), _d("completion"), _d("total"), _d("cached"), _d("calls")
