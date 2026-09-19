"""LLM backend layer with a paid-API interlock.

Three tiers (user policy 2026-09-17):

    mock://     development and unit tests         -- never touches network
    local       localhost OpenAI-compatible model  -- never costs money
    deepseek    the real paid API                  -- REQUIRES an explicit opt-in

The paid tier is refused unless BOTH hold:

    LLM_BACKEND=deepseek
    ALLOW_PAID_API=YES_I_ACCEPT_REAL_COST

No env var, no paid call -- even if a config file still carries an API key.
This exists because a single debug run against a real provider cost CNY 15.68
(found by the symptom2disease pilot); engineering iteration must be free.

Every backend implements Meta^n's ``LLMClient`` surface:

    async complete_with_breakdown(messages, temperature=None, max_tokens=None,
                                  *, seed=None, _suppress_io_log=False,
                                  _empty_retry_done=False)
        -> (text, prompt_tokens, completion_tokens, total_tokens)
    async complete(messages, temperature=None, max_tokens=None, *, seed=None)
        -> (text, total_tokens)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod

BACKEND_ENV = "LLM_BACKEND"
ALLOW_PAID_ENV = "ALLOW_PAID_API"
ALLOW_PAID_TOKEN = "YES_I_ACCEPT_REAL_COST"
CALL_LEDGER_ENV = "META_N_CALL_LEDGER"
MAX_CALLS_ENV = "META_N_MAX_LLM_CALLS"

MOCK = "mock"
LOCAL = "local"
DEEPSEEK = "deepseek"
MOONSHOT = "moonshot"
RELAY = "relay"          # generic OpenAI-compatible relay (api-key.xyz style)

# Every paid provider, with its endpoint / key env / default model. Adding a
# provider is a DATA change here, never a new code path.
PAID_PROVIDERS = {
    DEEPSEEK: {"base_url_env": "DEEPSEEK_BASE_URL",
               "base_url_default": "https://api.deepseek.com/v1",
               "key_env": "DEEPSEEK_API_KEY",
               "model_env": "DEEPSEEK_MODEL",
               "model_default": "deepseek-flash"},
    MOONSHOT: {"base_url_env": "MOONSHOT_BASE_URL",
               "base_url_default": "https://api.moonshot.cn/v1",
               "key_env": "MOONSHOT_API_KEY",
               "model_env": "MOONSHOT_MODEL",
               "model_default": "kimi-k2.7-code"},
    RELAY: {"base_url_env": "RELAY_BASE_URL",
            "base_url_default": "https://api-key.xyz/api/v1",
            "key_env": "RELAY_API_KEY",
            "model_env": "RELAY_MODEL",
            "model_default": "gemini-3.7-flash"},
}

PAID_KINDS = frozenset(PAID_PROVIDERS)


class PaidBackendDisabled(RuntimeError):
    """Raised when a paid backend is requested without the explicit opt-in."""


# --------------------------------------------------------------------------
# interlock
# --------------------------------------------------------------------------

def paid_api_allowed() -> bool:
    return os.environ.get(ALLOW_PAID_ENV, "").strip() == ALLOW_PAID_TOKEN


def assert_paid_allowed(kind: str) -> None:
    """Fail closed: a paid backend needs LLM_BACKEND *and* the opt-in token."""
    if kind not in PAID_KINDS:
        return
    problems = []
    if os.environ.get(BACKEND_ENV, MOCK).strip().lower() != kind:
        problems.append("{}={}".format(BACKEND_ENV, kind))
    if not paid_api_allowed():
        problems.append("{}=\"{}\"".format(ALLOW_PAID_ENV, ALLOW_PAID_TOKEN))
    if problems:
        raise PaidBackendDisabled(
            "Paid LLM backend '{}' is disabled. Set {} to enable it. "
            "Use the mock or local backend for development -- they never "
            "cost money.".format(kind, " and ".join(problems)))


def resolve_backend(kind: str | None = None) -> "LLMBackend":
    k = (kind or os.environ.get(BACKEND_ENV, MOCK)).strip().lower()
    if k == MOCK:
        return MockBackend()
    if k == LOCAL:
        return LocalBackend()
    if k in PAID_KINDS:
        assert_paid_allowed(k)
        return PaidProviderBackend(k)
    raise ValueError("unknown {}: {!r}".format(BACKEND_ENV, k))


# --------------------------------------------------------------------------
# call ledger -- one line per logical call, shared across subprocess workers
# --------------------------------------------------------------------------

def _append_call_ledger(backend: str, messages, text: str) -> None:
    """Append one line per logical LLM call.

    Subprocess workers are separate processes, so an in-process counter cannot
    aggregate. This file plays the role the cost ledger plays for money: one
    append-only JSONL every process writes to. Off unless META_N_CALL_LEDGER
    is set, so production runs pay nothing.
    """
    path = os.environ.get(CALL_LEDGER_ENV, "").strip()
    if not path:
        return
    prompt = "\n".join(str(m.get("content", "")) for m in messages)
    rec = {
        "ts": time.time(),
        "pid": os.getpid(),
        "backend": backend,
        "prompt_len": len(prompt),
        "prompt_head": prompt[:160],
        "response": (text or "")[:160],
        "empty": not (text or "").strip(),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------

class LLMBackend(ABC):
    name = "base"
    is_paid = False

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete_with_breakdown(
        self, messages, temperature=None, max_tokens=None, *, seed=None,
        _suppress_io_log=False, _empty_retry_done=False,
    ):
        text = await self._generate(messages, temperature, max_tokens, seed)
        pt = sum(len(str(m.get("content", ""))) // 4 for m in messages)
        ct = max(1, len(text) // 4)
        self.calls.append({
            "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "seed": seed,
            "text": text, "prompt_tokens": pt, "completion_tokens": ct,
        })
        _append_call_ledger(self.name, messages, text)
        return text, pt, ct, pt + ct

    async def complete(self, messages, temperature=None, max_tokens=None,
                       *, seed=None):
        text, _, _, total = await self.complete_with_breakdown(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed)
        return text, total

    @abstractmethod
    async def _generate(self, messages, temperature, max_tokens, seed) -> str:
        ...

    # -- test helpers -----------------------------------------------------
    @property
    def call_count(self) -> int:
        return len(self.calls)

    def last_prompt(self) -> str:
        if not self.calls:
            return ""
        return "\n".join(str(m.get("content", ""))
                         for m in self.calls[-1]["messages"])

    def reset(self) -> None:
        self.calls.clear()


# --------------------------------------------------------------------------
# mock -- deterministic, offline, no model
# --------------------------------------------------------------------------

# Fenced-block contract produced by OmegaEngine._parse_response:
#   ```rationale ... ```            -> rationale
#   ```pre_process ... ```          -> pre_process
#   ```solver_lib:<name> ... ```    -> code_library (keyed by the real funcname)
#   ```solver_lib_bash:<name> ... ``` -> code_library_bash

_MOCK_INJECTION = """\
```rationale
mock layer at depth {depth}: keep the contract, change nothing observable.
```

```pre_process
def pre_process(task, solution):
    return solution
```

```solver_lib:mock_helper
def mock_helper(x):
    return x
```
"""

_MOCK_SCRIPT = """\
import sys


def solve(task, llm=None):
    \"\"\"Mock solver. Deterministic, no network, no model.\"\"\"
    return "mock"
"""


class MockBackend(LLMBackend):
    """Rule-based backend. Never opens a socket, never costs money.

    Dispatch is by prompt content, so the real Meta^n parsers are exercised
    end to end. Register exact responders with :meth:`register` to pin a
    specific prompt; unmatched prompts fall through to the defaults.
    """

    name = MOCK
    is_paid = False

    def __init__(self) -> None:
        super().__init__()
        self._rules: list[tuple[re.Pattern, str]] = []

    def register(self, pattern: str, response: str, regex: bool = True) -> None:
        pat = re.compile(pattern, re.DOTALL) if regex else \
            re.compile(re.escape(pattern), re.DOTALL)
        self._rules.insert(0, (pat, response))

    def _dispatch(self, prompt: str) -> str:
        for pat, resp in self._rules:
            if pat.search(prompt):
                return resp
        low = prompt.lower()
        if "pre_process" in low or "meta-layer" in low or "omega" in low:
            m = re.search(r"depth[\s:=]+(\d+)", low)
            depth = m.group(1) if m else "0"
            return _MOCK_INJECTION.format(depth=depth)
        if "solve" in low or "script" in low or "solver" in low:
            return _MOCK_SCRIPT
        if "classif" in low or "label" in low:
            return "mock_label"
        return "mock"

    async def _generate(self, messages, temperature, max_tokens, seed) -> str:
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        base = self._dispatch(prompt)
        if seed is None:
            return base
        # Deterministic per (prompt, seed); still parseable.
        tag = hashlib.sha256(
            (prompt + "|" + str(seed)).encode("utf-8")).hexdigest()[:8]
        return base + "\n# mock-seed {}\n".format(tag)


# --------------------------------------------------------------------------
# local -- OpenAI-compatible server on localhost, still free
# --------------------------------------------------------------------------

class LocalBackend(LLMBackend):
    """Delegates to an OpenAI-compatible server (llama.cpp / vLLM / Ollama).

    Reads ``LOCAL_LLM_BASE_URL`` (default http://127.0.0.1:8000/v1) and
    ``LOCAL_LLM_MODEL``. Never reaches a paid provider.
    """

    name = LOCAL
    is_paid = False

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None) -> None:
        super().__init__()
        self.base_url = (base_url
                         or os.environ.get("LOCAL_LLM_BASE_URL",
                                           "http://127.0.0.1:8000/v1"))
        self.model = model or os.environ.get("LOCAL_LLM_MODEL", "local-model")
        self.api_key = api_key or os.environ.get("LOCAL_LLM_API_KEY", "not-needed")
        if "127.0.0.1" not in self.base_url and "localhost" not in self.base_url:
            raise ValueError(
                "LocalBackend refuses a non-loopback base_url ({!r}); paid "
                "providers must go through DeepSeekBackend + the interlock."
                .format(self.base_url))

    async def _generate(self, messages, temperature, max_tokens, seed) -> str:
        import urllib.request
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7 if temperature is None else temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key})
        with urllib.request.urlopen(req, timeout=600) as r:
            blob = json.load(r)
        return blob["choices"][0]["message"].get("content") or ""


# --------------------------------------------------------------------------
# deepseek -- the real thing, behind the interlock
# --------------------------------------------------------------------------

class PaidProviderBackend(LLMBackend):
    """A paid OpenAI-compatible provider, behind the interlock.

    Wraps Meta^n's own LLMClient so the cost tracker, the daily cap and the
    request ledger all still apply. Which provider, endpoint and model are
    resolved from `PAID_PROVIDERS` -- adding one is a data change, not a new
    code path.

    Constructing this class requires ``assert_paid_allowed`` to have passed.
    """

    is_paid = True

    def __init__(self, kind: str, client=None) -> None:
        super().__init__()
        if kind not in PAID_PROVIDERS:
            raise ValueError("not a paid provider: {!r}".format(kind))
        self.name = kind
        assert_paid_allowed(kind)               # belt and braces
        spec = PAID_PROVIDERS[kind]
        if client is None:
            from meta_n.core.llm_client import LLMClient, LLMConfig
            client = LLMClient(LLMConfig(
                base_url=os.environ.get(spec["base_url_env"],
                                        spec["base_url_default"]),
                api_key=os.environ.get(spec["key_env"]),
                model=os.environ.get(spec["model_env"], spec["model_default"]),
                daily_budget_usd=float(
                    os.environ.get("META_N_DAILY_BUDGET_USD", "0") or 0),
            ))
            self._apply_thinking(client)
        self.client = client

    @staticmethod
    def _apply_thinking(client) -> None:
        """Optionally PIN the provider's thinking mode (env `LLM_THINKING`).

        Meta^n's workload is repeated outer code generation and Omega
        injections, not one long derivation per call. A provider that defaults
        to high-effort reasoning spends most of each call thinking, which
        shows up as very slow generation and a much larger bill for no gain on
        this workload.

        `LLM_THINKING=disabled` sends the provider-native switch. DeepSeek
        accepts ``extra_body={"thinking": {"type": "disabled"}}``; other
        OpenAI-compatible providers that only understand the generic knob get
        ``reasoning_effort="none"``.

        Unset => nothing is added and the request stays byte-identical to the
        provider default, so this cannot silently change an unrelated run.
        """
        mode = os.environ.get("LLM_THINKING", "").strip().lower()
        if mode not in ("disabled", "off", "none"):
            return
        extra = dict(getattr(client, "_extra_body", None) or {})
        if client.config.backend == "azure":
            return                              # azure ignores extra_body
        extra["thinking"] = {"type": "disabled"}
        extra["reasoning_effort"] = "none"
        client._extra_body = extra

    async def _generate(self, messages, temperature, max_tokens, seed) -> str:
        text, _, _, _ = await self.client.complete_with_breakdown(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed)
        return text


class DeepSeekBackend(PaidProviderBackend):
    """Back-compat alias."""

    def __init__(self, client=None) -> None:
        super().__init__(DEEPSEEK, client=client)


class MoonshotBackend(PaidProviderBackend):
    def __init__(self, client=None) -> None:
        super().__init__(MOONSHOT, client=client)
