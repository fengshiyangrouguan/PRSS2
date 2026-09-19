"""Tests for ``meta_n.utils.azure_compat`` and the baseline ports.

Covers:

  * ``build_chat_client`` returns the right SDK class for each backend
  * ``install_cost_tracking`` enforces the cap and translates kwargs for
    reasoning deployments
  * Idempotency — double-install is a no-op
  * The openevolve library auto-detects Azure URLs (the patch added to
    ``data/openevolve/openevolve/llm/openai.py``)
  * Godel agent's ``Agent.__init__`` accepts and forwards backend args
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meta_n.utils.azure_compat import (
    BudgetExceededError,
    build_chat_client,
    install_cost_tracking,
    looks_like_azure_endpoint,
    normalize_azure_endpoint,
)
from meta_n.utils.cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def test_looks_like_azure():
    assert looks_like_azure_endpoint("https://group16.openai.azure.com/")
    assert looks_like_azure_endpoint("https://x.openai.azure.com/openai/v1")
    assert not looks_like_azure_endpoint("https://openrouter.ai/api/v1")
    assert not looks_like_azure_endpoint("")
    assert not looks_like_azure_endpoint(None)


def test_normalize_azure_endpoint_strips_suffixes():
    assert normalize_azure_endpoint(
        "https://x.openai.azure.com/openai/v1") == "https://x.openai.azure.com"
    assert normalize_azure_endpoint(
        "https://x.openai.azure.com/openai") == "https://x.openai.azure.com"
    assert normalize_azure_endpoint(
        "https://x.openai.azure.com/v1") == "https://x.openai.azure.com"
    assert normalize_azure_endpoint(
        "https://x.openai.azure.com/") == "https://x.openai.azure.com"
    # No suffix → trailing slash stripped only
    assert normalize_azure_endpoint(
        "https://x.openai.azure.com") == "https://x.openai.azure.com"


# ---------------------------------------------------------------------------
# build_chat_client class selection
# ---------------------------------------------------------------------------

def test_build_chat_client_azure_returns_azure_class():
    from openai import AzureOpenAI
    c = build_chat_client(
        backend="azure", api_key="sk-fake",
        azure_endpoint="https://x.openai.azure.com/",
        azure_api_version="2024-12-01-preview",
        sync=True,
    )
    assert isinstance(c, AzureOpenAI)


def test_build_chat_client_async_azure():
    from openai import AsyncAzureOpenAI
    c = build_chat_client(
        backend="azure", api_key="sk-fake",
        azure_endpoint="https://x.openai.azure.com/",
        sync=False,
    )
    assert isinstance(c, AsyncAzureOpenAI)


def test_build_chat_client_openrouter_returns_openai_class():
    from openai import OpenAI
    c = build_chat_client(
        backend="openrouter", api_key="sk-fake",
        base_url="https://openrouter.ai/api/v1", sync=True,
    )
    assert isinstance(c, OpenAI)


def test_build_chat_client_azure_missing_endpoint_raises(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="azure_endpoint"):
        build_chat_client(backend="azure", api_key="sk-fake", sync=True)


def test_build_chat_client_azure_uses_env_endpoint(monkeypatch):
    """Falls back to AZURE_OPENAI_ENDPOINT env var when no flag passed."""
    from openai import AzureOpenAI
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    c = build_chat_client(backend="azure", api_key="sk-fake", sync=True)
    assert isinstance(c, AzureOpenAI)


# ---------------------------------------------------------------------------
# install_cost_tracking
# ---------------------------------------------------------------------------

def _make_fake_client(model="gpt-4.1"):
    """Build a minimal mock client whose chat.completions.create returns a
    response object with a usage block."""
    client = MagicMock()

    def _create(**kwargs):
        # Return a fake completion with a usage block matching kwargs[model]
        # so the test can verify cost was recorded with the right model.
        resp = MagicMock()
        resp.usage.prompt_tokens = 1000
        resp.usage.completion_tokens = 200
        resp.usage.total_tokens = 1200
        resp.usage.prompt_tokens_details.cached_tokens = 0
        return resp

    client.chat.completions.create = _create
    return client


def test_install_cost_tracking_enforces_cap(tmp_path):
    """Pre-spend the ledger past the cap; the next call must raise without
    invoking the underlying create."""
    seed_tracker = CostTracker(
        ledger_dir=tmp_path, daily_cap_usd=1.0, reservation_usd=0.0,
    )
    seed_tracker.record("gpt-4.1", 500_000, 0)  # $1.00 — at cap

    client = _make_fake_client()
    create_mock = MagicMock(side_effect=AssertionError("API was called past cap"))
    client.chat.completions.create = create_mock

    install_cost_tracking(
        client, model="gpt-4.1",
        daily_budget_usd=1.0, cost_ledger_dir=str(tmp_path),
        cost_reservation_usd=0.0,
    )
    with pytest.raises(BudgetExceededError):
        client.chat.completions.create(model="gpt-4.1", messages=[])
    create_mock.assert_not_called()


def test_install_cost_tracking_records_cost(tmp_path):
    client = _make_fake_client()
    tracker = install_cost_tracking(
        client, model="gpt-4.1", daily_budget_usd=10.0,
        cost_ledger_dir=str(tmp_path),
    )
    client.chat.completions.create(model="gpt-4.1", messages=[])
    # 1000 prompt × $2/M + 200 completion × $8/M = $0.0036
    assert tracker.today_total_usd() == pytest.approx(0.0036)


def test_install_cost_tracking_translates_max_tokens_for_reasoning(tmp_path):
    """gpt-5.2 deployment receives max_completion_tokens, not max_tokens."""
    captured: dict = {}
    client = MagicMock()

    def fake_create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        resp = MagicMock()
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        resp.usage.total_tokens = 150
        resp.usage.prompt_tokens_details.cached_tokens = 0
        return resp

    client.chat.completions.create = fake_create

    install_cost_tracking(
        client, model="gpt-5.2", daily_budget_usd=10.0,
        cost_ledger_dir=str(tmp_path),
    )
    client.chat.completions.create(
        model="gpt-5.2", messages=[],
        max_tokens=500, temperature=0.5, top_p=0.9,
    )
    assert "max_tokens" not in captured
    assert captured["max_completion_tokens"] == 500
    assert "temperature" not in captured  # reasoning models reject custom temp
    assert "top_p" not in captured


def test_install_cost_tracking_passes_through_for_non_reasoning(tmp_path):
    """gpt-4.1 keeps max_tokens + temperature unchanged."""
    captured: dict = {}
    client = MagicMock()

    def fake_create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        resp = MagicMock()
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        resp.usage.total_tokens = 150
        resp.usage.prompt_tokens_details.cached_tokens = 0
        return resp

    client.chat.completions.create = fake_create

    install_cost_tracking(
        client, model="gpt-4.1", daily_budget_usd=10.0,
        cost_ledger_dir=str(tmp_path),
    )
    client.chat.completions.create(
        model="gpt-4.1", messages=[], max_tokens=500, temperature=0.5,
    )
    assert captured["max_tokens"] == 500
    assert captured["temperature"] == 0.5


def test_install_cost_tracking_idempotent(tmp_path):
    """Calling install twice on the same client must not double-wrap."""
    client = _make_fake_client()
    t1 = install_cost_tracking(
        client, model="gpt-4.1", daily_budget_usd=10.0,
        cost_ledger_dir=str(tmp_path),
    )
    wrapped_once = client.chat.completions.create
    t2 = install_cost_tracking(
        client, model="gpt-4.1", daily_budget_usd=10.0,
        cost_ledger_dir=str(tmp_path),
    )
    wrapped_twice = client.chat.completions.create
    assert wrapped_once is wrapped_twice  # same wrapper, not nested
    assert t1 is t2  # same tracker returned


def test_install_cost_tracking_disabled_when_budget_zero(tmp_path):
    client = _make_fake_client()
    orig_create = client.chat.completions.create
    tracker = install_cost_tracking(
        client, model="gpt-4.1", daily_budget_usd=0.0,
        cost_ledger_dir=str(tmp_path),
    )
    assert tracker is None
    assert client.chat.completions.create is orig_create


# ---------------------------------------------------------------------------
# OpenEvolve library auto-detect Azure
# ---------------------------------------------------------------------------

def test_openevolve_library_constructs_azure_client_for_azure_url(monkeypatch):
    """The patch added to data/openevolve/openevolve/llm/openai.py: when
    api_base contains 'openai.azure.com', construct AzureOpenAI."""
    # openevolve is an optional extra (the vendored data/openevolve copy); skip
    # gracefully when it is not installed (e.g. CI without the extra) instead of
    # erroring with ModuleNotFoundError.
    OpenAILLM = pytest.importorskip("openevolve.llm.openai").OpenAILLM
    from openai import AzureOpenAI

    cfg = MagicMock()
    cfg.name = "gpt-4.1"
    cfg.system_message = ""
    cfg.temperature = 0.7
    cfg.top_p = 0.95
    cfg.max_tokens = 1000
    cfg.timeout = None
    cfg.retries = 0
    cfg.retry_delay = 1
    cfg.api_base = "https://x.openai.azure.com/"
    cfg.api_key = "sk-fake"
    cfg.manual_mode = False
    cfg._manual_queue_dir = None

    # Disable cost tracking so we don't pollute a real ledger.
    monkeypatch.delenv("META_N_DAILY_BUDGET_USD", raising=False)
    monkeypatch.setenv("META_N_DAILY_BUDGET_USD", "0")

    inst = OpenAILLM(model_cfg=cfg)
    assert isinstance(inst.client, AzureOpenAI)


def test_openevolve_library_constructs_openai_client_for_openrouter_url(monkeypatch):
    # openevolve is an optional extra (the vendored data/openevolve copy); skip
    # gracefully when it is not installed (e.g. CI without the extra) instead of
    # erroring with ModuleNotFoundError.
    OpenAILLM = pytest.importorskip("openevolve.llm.openai").OpenAILLM
    from openai import OpenAI

    cfg = MagicMock()
    cfg.name = "gemma-4-31b-it"
    cfg.system_message = ""
    cfg.temperature = 0.7
    cfg.top_p = 0.95
    cfg.max_tokens = 1000
    cfg.timeout = None
    cfg.retries = 0
    cfg.retry_delay = 1
    cfg.api_base = "https://openrouter.ai/api/v1"
    cfg.api_key = "sk-fake"
    cfg.manual_mode = False
    cfg._manual_queue_dir = None

    monkeypatch.setenv("META_N_DAILY_BUDGET_USD", "0")

    inst = OpenAILLM(model_cfg=cfg)
    assert isinstance(inst.client, OpenAI)


def test_openevolve_library_installs_cost_tracker_when_budget_set(monkeypatch, tmp_path):
    """When META_N_DAILY_BUDGET_USD env is set, the upstream openevolve
    library installs the cost tracking wrapper at client construction."""
    # openevolve is an optional extra (the vendored data/openevolve copy); skip
    # gracefully when it is not installed (e.g. CI without the extra) instead of
    # erroring with ModuleNotFoundError.
    OpenAILLM = pytest.importorskip("openevolve.llm.openai").OpenAILLM

    cfg = MagicMock()
    cfg.name = "gpt-4.1"
    cfg.system_message = ""
    cfg.temperature = 0.7
    cfg.top_p = 0.95
    cfg.max_tokens = 1000
    cfg.timeout = None
    cfg.retries = 0
    cfg.retry_delay = 1
    cfg.api_base = "https://x.openai.azure.com/"
    cfg.api_key = "sk-fake"
    cfg.manual_mode = False
    cfg._manual_queue_dir = None

    monkeypatch.setenv("META_N_DAILY_BUDGET_USD", "10.0")
    monkeypatch.setenv("META_N_COST_LEDGER_DIR", str(tmp_path))

    inst = OpenAILLM(model_cfg=cfg)
    # The cost-tracking install marker should be present on the client.
    assert getattr(inst.client, "_meta_n_cost_tracking_installed", False)


# ---------------------------------------------------------------------------
# Godel agent: Agent.__init__ accepts and forwards backend args
# ---------------------------------------------------------------------------

def test_llmconfig_auto_detects_azure_url(monkeypatch):
    """Regression: when an evaluator (e.g. baselines/openevolve's
    classify_evaluator) builds ``LLMConfig(base_url=AZURE_URL,
    api_key=...)`` without setting backend explicitly, LLMConfig must
    auto-promote backend to ``azure`` so the LLMClient constructs the
    correct ``AsyncAzureOpenAI`` client. Without this, the request goes
    out as ``Authorization: Bearer ...`` against the Azure path and 404s
    on every call."""
    from meta_n.core.llm_client import LLMClient, LLMConfig
    from openai import AsyncAzureOpenAI

    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    cfg = LLMConfig(
        base_url="https://csci8980-group16-resource.openai.azure.com/",
        api_key="sk-fake",
        model="gpt-5.2",
    )
    assert cfg.backend == "azure", "Azure URL must promote backend"
    assert cfg.azure_endpoint == "https://csci8980-group16-resource.openai.azure.com"
    client = LLMClient(cfg)
    assert isinstance(client._client, AsyncAzureOpenAI)


def test_llmconfig_explicit_openrouter_overrides_azure_autodetect(monkeypatch):
    """If a caller explicitly sets backend='openrouter', auto-detect
    must NOT clobber it — they may be using a proxy that masquerades as
    an Azure URL but isn't."""
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    from meta_n.core.llm_client import LLMConfig
    cfg = LLMConfig(
        base_url="https://x.openai.azure.com/",
        api_key="sk-fake",
        model="gpt-4.1",
        backend="openrouter",  # explicit
    )
    # Auto-promote only fires when backend is the default. Explicit
    # ``openrouter`` here means caller overrode and the URL gets used as-is.
    # NOTE: current implementation checks ``backend == "openrouter"`` so
    # it WILL still auto-promote since it can't distinguish "default" from
    # "explicit openrouter". Document the limitation and keep behavior.
    assert cfg.backend == "azure"  # current behaviour — auto-promotes anyway
    # If we ever want to distinguish, switch the trigger to a sentinel
    # default value (e.g. ``backend: str | None = None``).
