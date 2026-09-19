"""Shared helpers for plugging Azure OpenAI + cost tracking into baselines.

Both ``baselines/godel_agent`` and ``baselines/openevolve`` (and the
upstream ``openevolve`` library) talk to the LLM via raw ``openai`` SDK
clients (``openai.OpenAI`` / ``AsyncOpenAI``). To run them on the
csci8980-group16 Azure deployment with the same daily-cap ledger as the
Meta^n driver, we need two things at every client construction site:

  1. **Build the right client class.** Azure inference uses
     ``AzureOpenAI`` (different URL routing, ``api-key`` header, mandatory
     ``api-version`` query param) — pointing ``OpenAI(base_url=...)`` at
     an Azure endpoint sends ``Authorization: Bearer ...`` and 401s.
  2. **Wrap ``chat.completions.create``** so each call (a) refuses to
     fire when today's spend has reached the daily cap, (b) translates
     ``max_tokens``→``max_completion_tokens`` and drops
     ``temperature``/``top_p`` for gpt-5.x / o-series reasoning
     deployments that reject the legacy params, and (c) appends a
     cost-bearing line to ``~/.meta_n_costs/<YYYY-MM-DD>.jsonl``.

Both baselines call into this module from a single chokepoint each — no
copy-paste of Azure detection or kwarg translation.

Idempotent: ``install_cost_tracking`` is safe to call once per client; a
second call detects the existing wrapper and is a no-op.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from meta_n.core.llm_client import (
    _supports_custom_temperature,
    _uses_max_completion_tokens,
)
from meta_n.utils.cost_tracker import (
    BudgetExceededError,
    CostTracker,
)

logger = logging.getLogger(__name__)

# Sentinel attribute name placed on a wrapped client to prevent double-
# installation when a runner calls ``install_cost_tracking`` twice. The
# value stored is the literal ``True`` and is checked with ``is True``
# (identity, not truthiness — MagicMock auto-creates truthy attributes
# that would false-positive a plain ``getattr(...)`` check). WHICH tracker
# is attached is exposed separately via the ``_TRACKER_ATTR`` attribute,
# which the idempotent re-install path returns.
_INSTALLED_FLAG = "_meta_n_cost_tracking_installed"
_TRACKER_ATTR = "_meta_n_cost_tracker"


def looks_like_azure_endpoint(url: str | None) -> bool:
    """True for Azure resource URLs like
    ``https://<name>.openai.azure.com/`` (with or without a trailing
    path). Used to auto-route the openevolve upstream library to
    ``AzureOpenAI`` without changing its config schema."""
    if not url:
        return False
    return "openai.azure.com" in url.lower()


def normalize_azure_endpoint(url: str) -> str:
    """Strip trailing ``/openai/v1`` / ``/openai`` / ``/v1`` so callers
    can pass any of the URL shapes Azure docs use and ``AzureOpenAI``
    builds the deployment URL the same way every time."""
    endpoint = url.rstrip("/")
    for suffix in ("/openai/v1", "/openai", "/v1"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
    return endpoint


def build_chat_client(
    *,
    backend: str,
    api_key: str,
    base_url: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str = "2024-12-01-preview",
    timeout: Any = None,
    max_retries: int = 0,
    sync: bool = True,
):
    """Construct an ``openai`` client of the right kind for the backend.

    Args:
        backend: ``"azure"`` → ``AzureOpenAI``; anything else →
            ``OpenAI`` (covers OpenRouter, OpenAI direct, vLLM, LMStudio).
        api_key: API key (Azure key for ``backend=azure``, OpenRouter
            key otherwise).
        base_url: OpenAI-compatible endpoint URL — required when
            backend is not azure.
        azure_endpoint: Azure resource URL. Defaults to
            ``$AZURE_OPENAI_ENDPOINT``.
        azure_api_version: Azure API version.
        timeout: Forwarded to the SDK client (httpx.Timeout instance or
            float seconds).
        max_retries: Forwarded to the SDK; default 0 because callers
            generally have their own retry layer that we don't want to
            multiply with.
        sync: True → blocking client (godel_agent, openevolve library);
            False → ``Async*`` client (meta_n driver).

    Returns:
        An ``openai.{Async,}OpenAI`` or ``openai.{Async,}AzureOpenAI``
        instance, ready to call ``chat.completions.create``.
    """
    from openai import (
        AsyncAzureOpenAI,
        AsyncOpenAI,
        AzureOpenAI,
        OpenAI,
    )

    if backend == "azure":
        endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if not endpoint:
            raise ValueError(
                "Azure backend selected but azure_endpoint is empty and "
                "AZURE_OPENAI_ENDPOINT env var is unset."
            )
        endpoint = normalize_azure_endpoint(endpoint)
        cls = AzureOpenAI if sync else AsyncAzureOpenAI
        kw: dict[str, Any] = {
            "api_key": api_key,
            "azure_endpoint": endpoint,
            "api_version": azure_api_version,
            "max_retries": max_retries,
        }
        if timeout is not None:
            kw["timeout"] = timeout
        return cls(**kw)

    # Non-Azure path — unchanged for legacy callers.
    cls = OpenAI if sync else AsyncOpenAI
    kw = {
        "api_key": api_key,
        "max_retries": max_retries,
    }
    if base_url:
        kw["base_url"] = base_url
    if timeout is not None:
        kw["timeout"] = timeout
    return cls(**kw)


def install_cost_tracking(
    client,
    *,
    model: str,
    daily_budget_usd: float,
    cost_ledger_dir: str = "~/.meta_n_costs",
    cost_reservation_usd: float = 1.0,
    cost_utc_day: bool = False,
    extra: dict | None = None,
) -> CostTracker | None:
    """Wrap ``client.chat.completions.create`` with the cost tracker.

    The returned wrapper, on every call:
      * raises ``BudgetExceededError`` when the daily ledger is at/over
        the cap (re-checked per call, so a multi-process run where one
        worker pushes us over halts the next worker on its next call);
      * for reasoning deployments (``gpt-5.x``, ``o*``) translates
        ``max_tokens`` → ``max_completion_tokens`` and drops
        ``temperature``/``top_p`` so the call doesn't 400;
      * after a successful response, appends a JSONL record with the
        usage breakdown and computed USD cost.

    A no-op when ``daily_budget_usd <= 0`` (cost tracking disabled).

    Idempotent: a second call on the same client does nothing — useful
    for runners that re-init their client after each iteration.

    Returns the ``CostTracker`` (or ``None`` if disabled) so callers can
    inspect ``today_total_usd()`` for status logging.
    """
    if not daily_budget_usd or daily_budget_usd <= 0:
        return None
    # Identity check (``is True``), not truthiness — MagicMock auto-creates
    # attributes that test truthy but aren't our sentinel. The ``is True``
    # form distinguishes a real install from an auto-generated mock attr.
    if getattr(client, _INSTALLED_FLAG, None) is True:
        # Already wrapped on a previous call — refuse to double-wrap.
        return getattr(client, _TRACKER_ATTR, None)

    # Fail-fast validation: a typo'd model name (e.g. "gpt-5-2" vs
    # "gpt-5.2") would otherwise silently skip cost recording at every
    # call (KeyError caught by ``_record_from_response``) and the cap
    # would never fire. Raise here so the user sees the misconfiguration
    # at startup, not after a $295 surprise.
    from meta_n.utils.cost_tracker import get_pricing
    try:
        get_pricing(model)
    except KeyError as e:
        raise KeyError(
            f"Cannot install cost tracking: {e}. Add the deployment to "
            f"meta_n/utils/cost_tracker.PRICING or set "
            f"META_N_PRICING_OVERRIDE_JSON before launching."
        ) from e

    tracker = CostTracker(
        ledger_dir=cost_ledger_dir,
        daily_cap_usd=daily_budget_usd,
        reservation_usd=cost_reservation_usd,
        utc=cost_utc_day,
    )

    is_reasoning = _uses_max_completion_tokens(model)
    skip_temperature = not _supports_custom_temperature(model)
    orig_create = client.chat.completions.create
    is_async = _looks_async(orig_create)

    if is_async:
        async def wrapped(*args, **kwargs):
            tracker.assert_under_cap()
            _translate_kwargs_for_reasoning(
                kwargs, is_reasoning=is_reasoning, skip_temperature=skip_temperature,
            )
            resp = await orig_create(*args, **kwargs)
            _record_from_response(tracker, model, resp, extra=extra)
            return resp
    else:
        def wrapped(*args, **kwargs):
            tracker.assert_under_cap()
            _translate_kwargs_for_reasoning(
                kwargs, is_reasoning=is_reasoning, skip_temperature=skip_temperature,
            )
            resp = orig_create(*args, **kwargs)
            _record_from_response(tracker, model, resp, extra=extra)
            return resp

    client.chat.completions.create = wrapped
    setattr(client, _INSTALLED_FLAG, True)
    setattr(client, _TRACKER_ATTR, tracker)
    logger.info(
        "[azure_compat] cost tracking installed for model=%s cap=$%.2f "
        "ledger=%s",
        model, daily_budget_usd, cost_ledger_dir,
    )
    return tracker


def _looks_async(fn) -> bool:
    """True iff ``fn`` is a coroutine function. We avoid the qualname
    heuristic — MagicMock and other duck types have non-string
    ``__qualname__`` attributes that return mock objects whose ``in``
    operator returns truthy mocks, falsely reporting async. Sticking
    to ``asyncio.iscoroutinefunction`` is correct for the OpenAI SDK
    (the AsyncOpenAI client's ``chat.completions.create`` is registered
    as a coroutine function)."""
    import asyncio
    return asyncio.iscoroutinefunction(fn)


def _translate_kwargs_for_reasoning(
    kwargs: dict, *, is_reasoning: bool, skip_temperature: bool,
) -> None:
    """Mutate ``kwargs`` in place: rename ``max_tokens`` and drop
    custom-temperature args when the deployment is a reasoning model.

    No-op for non-reasoning deployments.
    """
    if not is_reasoning:
        return
    if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    if skip_temperature:
        for key in ("temperature", "top_p"):
            kwargs.pop(key, None)


def _record_from_response(
    tracker: CostTracker, model: str, resp, *, extra: dict | None = None,
) -> None:
    """Pull usage off the response and call ``tracker.record``. Defensive
    against providers that omit fields — never raises from the
    accounting path."""
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        cached = 0
        ptd = getattr(usage, "prompt_tokens_details", None)
        if ptd is not None:
            cached = int(getattr(ptd, "cached_tokens", 0) or 0)
        tracker.record(
            model=model,
            prompt_tokens=pt,
            completion_tokens=ct,
            cached_tokens=cached,
            extra=extra,
        )
    # BudgetExceededError derives from BaseException, so it would propagate
    # past this ``except Exception`` by construction (it also cannot arise
    # here: assert_under_cap runs in the wrapper, before orig_create).
    except Exception as e:  # noqa: BLE001
        logger.warning("[azure_compat] failed to record cost: %r", e)


__all__ = [
    "BudgetExceededError",
    "build_chat_client",
    "install_cost_tracking",
    "looks_like_azure_endpoint",
    "normalize_azure_endpoint",
]
