"""Unified LLM client supporting OpenRouter and Azure OpenAI.

Backend is selected via ``LLMConfig.backend``:

  * ``openrouter`` (default) — talks to any OpenAI-compatible endpoint
    (OpenRouter, vLLM, LMStudio) via ``AsyncOpenAI`` with a custom
    ``base_url``. The legacy path; nothing changes for existing callers.
  * ``azure`` — uses ``AsyncAzureOpenAI``. ``model`` is treated as the
    deployment name. Pricing is keyed on the deployment string, so name
    deployments after the model family (``gpt-4.1``, ``gpt-5.2``, …) or
    add an alias to ``meta_n/utils/cost_tracker.PRICING``.

When ``daily_budget_usd > 0``, every call routes through ``CostTracker``:
``assert_under_cap`` before the request, ``record`` after. A cap hit
raises :class:`BudgetExceededError` (a ``BaseException`` subclass) which
bypasses ``except Exception`` blocks in benchmark integrations and
propagates up to the orchestrator's checkpoint-save path.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import json
import os
import random
from dataclasses import dataclass

from json import JSONDecodeError

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from meta_n.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


def _uses_max_completion_tokens(model: str) -> bool:
    """gpt-5.x and o-series reasoning deployments reject the legacy
    ``max_tokens`` field. The OpenAI Chat Completions API renamed it to
    ``max_completion_tokens`` for reasoning models so the SDK can carve
    out room for hidden chain-of-thought tokens separately from visible
    output. Detect by model-family prefix.

    Notes:
      * ``gpt-5-chat-*`` and ``gpt-5.x-chat-*`` are the *non-reasoning*
        chat variants; they accept the old ``max_tokens``. We exclude
        those by checking for the ``-chat`` segment.
      * Adding a new reasoning family later? Extend this match — the
        symptom is HTTP 400 "Unsupported parameter: 'max_tokens'".
    """
    # Normalize a provider-prefixed id (OpenRouter / LiteLLM style, e.g.
    # ``openai/gpt-5.2`` or ``openai/o3-mini``) down to the bare model id
    # before family detection. Without this, the leading ``"<provider>/"``
    # segment defeats every ``startswith`` test below and the reasoning-model
    # payload shaping silently regresses (HTTP 400 max_tokens / temperature).
    # Azure bare-deployment names have no ``/`` and are unaffected.
    m = model.lower().rsplit("/", 1)[-1]
    if "-chat" in m:
        return False
    if m.startswith(("o1", "o3", "o4")):
        return True
    if m == "gpt-5" or m.startswith("gpt-5-") or m.startswith("gpt-5."):
        return True
    return False


def _fixes_temperature(model: str) -> bool:
    """Families that PIN temperature at 1 but still accept the legacy
    ``max_tokens`` field.

    Deliberately SEPARATE from ``_uses_max_completion_tokens``: folding these
    into the same predicate would also switch the token field, which these
    models do not want. Symptom of a missing entry here is HTTP 400
    ``invalid temperature: only 1 is allowed for this model``.
    """
    m = model.lower().rsplit("/", 1)[-1]
    # Moonshot / Kimi reasoning family (kimi-k2.6, kimi-k2.7-code, ...).
    if m.startswith("kimi-k2"):
        return True
    return False


def _supports_custom_temperature(model: str) -> bool:
    """Reasoning deployments fix temperature at 1; passing a custom value
    returns 400. Same family detection as above, with the same
    ``-chat`` exception -- plus the families that ONLY fix temperature."""
    return not (_uses_max_completion_tokens(model) or _fixes_temperature(model))


def _supports_request_seed(model: str, backend: str) -> bool:
    """Whether the backend is KNOWN to honour a per-request ``seed`` kwarg on
    ``chat.completions.create()`` — the capability gate for CRN / paired eval.

    Default-DENY: returns True only for backends we have positively confirmed
    accept a per-request integer seed (Azure / OpenAI-proper). Everything else
    (OpenRouter, LM Studio, vLLM, …) returns False so the seed is dropped
    inside ``complete_with_breakdown`` and the request payload stays
    byte-identical to today.

    CRITICAL: this is INTENTIONALLY independent of ``_supports_custom_temperature``.
    ``seed`` is orthogonal to the temperature=1 lock — reasoning families such
    as gpt-5.2 (``_uses_max_completion_tokens(model) == True``) reject a custom
    temperature but DO accept ``seed``. Gating seed behind the temperature
    predicate would silently make CRN a no-op on the exact backbone it must run
    on. ``model`` is accepted so a future per-deployment omit-list can be added
    here (e.g. a specific Azure deployment that 400s on seed) WITHOUT reusing
    the temperature predicate.
    """
    return backend == "azure"


# F068: relative component of the empty-content escalation cap. The absolute
# config cap (default 32768) was calibrated against the 16384 OUTER default;
# inner llm() calls default to max_tokens=512 and must not re-issue at 64x
# their scale. 4x preserves the calibrated outer behaviour exactly:
# min(32768, 4*16384) == 32768.
_EMPTY_ESCALATION_SCALE = 4


def stable_crn_seed(run_seed: int, task_id: str, repeat_index: int) -> int:
    """Common-Random-Numbers (CRN) seed for one ``(task, repeat)``, identical
    across ALL candidates in a run.

    The whole mechanism rests on the ABSENCE of a ``candidate_id`` parameter:
    parent and child evaluating the same ``(task_id, repeat_index)`` get the
    SAME seed, so on a seed-honouring backend the LLM-sampler luck is correlated
    and the paired child-vs-parent comparison cancels the shared noise.

    Implementation notes (each a unit-tested invariant):
      * ``hashlib.blake2b`` — NOT builtin ``hash()``, which is salted per-process
        via ``PYTHONHASHSEED`` and would break ``--resume`` and the CO-Bench
        subprocess workers (different processes would derive different seeds for
        the same key). blake2b is deterministic across calls / processes /
        machines.
      * NUL (``\\x00``) separators prevent boundary collisions, e.g.
        ``(task="a", r=11)`` vs ``(task="a1", r=1)``.
      * masked to a non-negative 31-bit int so every backend's integer seed
        field accepts it.

    Args:
        run_seed: the run-level base (``EvolutionaryConfig.seed`` / ``--seed``).
            Different ``run_seed`` ⇒ different seed (the lever that keeps the
            best-of-N control's runs independent).
        task_id: the task identifier.
        repeat_index: 0-based repeated-eval index. Different index ⇒ different
            seed, so ``eval_repeats`` re-solves stay diverse and CRN composes
            WITH repeated eval instead of collapsing it.

    Returns:
        A deterministic int in ``[0, 2**31 - 1]``.
    """
    basis = f"{int(run_seed)}\x00{task_id}\x00{int(repeat_index)}".encode("utf-8")
    digest = hashlib.blake2b(basis, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF


@dataclass
class LLMConfig:
    """Configuration for the LLM backend."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str | None = None
    model: str = "anthropic/claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 16384
    # EMPTY-CONTENT ESCALATION CAP. A reasoning / QAT model can spend its ENTIRE
    # ``max_tokens`` budget on hidden reasoning and return ``finish_reason=='length'``
    # with EMPTY ``message.content`` — observed on the builtin TB spine:
    # gemma-4-31b-qat burned completion_tokens==16384, content_len==0, which made
    # ``Layer1Solver`` author an EMPTY bash script => a SPURIOUS recipient failure
    # in the terminal_bench headroom screen (a config artifact, not task
    # difficulty). When this is > 0 AND an empty+length response comes back,
    # ``complete_with_breakdown`` RE-ISSUES the same request ONCE at
    # ``max(this, the cap just hit)`` so the model has room to finish reasoning AND
    # emit content. Scoped to the degenerate EMPTY+length case (a non-empty but
    # truncated completion keeps its partial content) and bounded to a SINGLE
    # escalation. Default 32768 = one free escalation above the 16384 default; the
    # screen/feal spine raises both caps (see build_solver). Set to 0 to disable.
    # The effective re-issue cap is ``min(this, 4*the call's requested
    # max_tokens)`` (``_EMPTY_ESCALATION_SCALE``), so small-scale inner ``llm()``
    # calls (default 512) escalate to 2048, not 32768; the 16384-outer-default
    # calibration (min(32768, 65536) == 32768) is unchanged.
    empty_content_retry_max_tokens: int = 32768
    max_retries: int = 3
    retry_base_delay: float = 2.0
    retry_max_delay: float = 60.0
    # Hard cap on a single chat.completions.create call. The OpenAI SDK
    # default is 600 s, which is far too long when a request hangs mid-
    # stream. Empirically gpt-5.2 reasoning on a ~1500-token prompt with
    # max_completion_tokens=16384 took 204 s (the model uses the budget
    # for hidden chain-of-thought before emitting the response) — so 180 s
    # was too tight and triggered false-positive timeouts. 360 s gives
    # ~1.7× headroom over the observed worst-case while still recovering
    # from a true wedge in minutes rather than tens of minutes.
    #
    # NONE-SENTINEL: defaults to ``None`` so ``__post_init__`` can tell an
    # explicit constructor ``request_timeout=360.0`` apart from "unset" — the
    # env fallback (META_N_LLM_REQUEST_TIMEOUT) only fires when this is None,
    # then None is coalesced to 360.0. An explicit value is therefore never
    # clobbered by the env var. Always a float after __post_init__.
    request_timeout: float | None = None
    exclude_providers: list[str] | None = None  # OpenRouter: exclude these providers

    # Reasoning-model control. When set (e.g. ``"none"``/``"low"``/``"medium"``/
    # ``"high"``), sent as ``reasoning_effort`` on every chat completion so a
    # reasoning/QAT model (e.g. LM Studio Gemma-QAT) can be told not to spend
    # its token budget on hidden reasoning. Lives on the CONFIG (not the client
    # instance) so it survives ``asdict(config)`` into the subprocess-isolated
    # evaluators. Default ``None`` = omit the field entirely → byte-identical
    # requests for non-reasoning callers.
    reasoning_effort: str | None = None

    # Azure-specific. Used only when ``backend == "azure"``.
    backend: str = "openrouter"  # "openrouter" | "azure"
    azure_endpoint: str | None = None
    azure_api_version: str = "2024-12-01-preview"

    # Cost tracking (off when daily_budget_usd <= 0).
    daily_budget_usd: float = 0.0
    cost_ledger_dir: str = "~/.meta_n_costs"
    # Headroom held back below the daily cap. Default 0.0 (the cap IS the cap):
    # a non-zero reservation silently shrinks the usable budget, which makes a
    # small ``--daily-budget-usd`` (e.g. 1) unusable without also knowing to
    # zero this out — the BUDGET-PRECHECK footgun. Operators who want a safety
    # margin still set it explicitly via the constructor or
    # ``META_N_COST_RESERVATION_USD``. See ``CostGuard.precheck`` for the
    # headroom-clamp that complements this default.
    #
    # NONE-SENTINEL: defaults to ``None`` (NOT 0.0) so ``__post_init__`` can
    # distinguish an explicit constructor ``cost_reservation_usd=0.0`` (which
    # must be honoured) from "unset". The env fallback
    # (META_N_COST_RESERVATION_USD) only fires when this is None, then None is
    # coalesced to 0.0 — so an explicit 0.0 is never clobbered by the env var.
    # Always a float after __post_init__.
    cost_reservation_usd: float | None = None
    # When True, the tracker rolls at UTC midnight; otherwise at local
    # midnight. Local feels more natural for personal budgets.
    cost_utc_day: bool = False

    def __post_init__(self):
        # Auto-detect Azure backend when ``base_url`` is an Azure resource
        # URL but the caller forgot (or doesn't know how) to set
        # ``backend=azure``. This makes evaluator code paths that build an
        # LLMConfig from raw env vars — e.g. baselines/openevolve's
        # classify_evaluator — work on Azure without needing to know
        # about the backend dispatch. Without this, OpenAI(base_url=
        # https://x.openai.azure.com/) sends `Authorization: Bearer ...`
        # against the wrong path and Azure 404s every call.
        # Only auto-promote when the caller left backend at the default
        # ("openrouter") AND base_url unambiguously points at Azure;
        # explicit ``backend="openrouter"`` is respected.
        if (
            self.backend == "openrouter"
            and self.base_url
            and "openai.azure.com" in str(self.base_url).lower()
        ):
            # Hoist base_url → azure_endpoint, strip URL suffixes that
            # AzureOpenAI would append again (the SDK builds the
            # deployment URL itself).
            endpoint = str(self.base_url).rstrip("/")
            for suffix in ("/openai/v1", "/openai", "/v1"):
                if endpoint.endswith(suffix):
                    endpoint = endpoint[: -len(suffix)]
            self.azure_endpoint = endpoint
            self.backend = "azure"

        if self.backend == "azure":
            if self.api_key is None:
                self.api_key = (
                    os.environ.get("AZURE_OPENAI_API_KEY")
                    or os.environ.get("OPENROUTER_API_KEY", "")
                )
            if self.azure_endpoint is None:
                self.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            if not self.api_key:
                logger.warning(
                    "Azure backend selected but AZURE_OPENAI_API_KEY is unset."
                )
            if not self.azure_endpoint:
                logger.warning(
                    "Azure backend selected but AZURE_OPENAI_ENDPOINT is unset."
                )
        else:
            if self.api_key is None:
                self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not self.api_key:
                logger.warning(
                    "No API key configured (set OPENROUTER_API_KEY or pass --api-key). "
                    "Requests will fail unless using a local server that requires no auth."
                )

        # Cost-tracking env-var fallback. The baselines' subprocess
        # evaluators (classify_evaluator, co_bench_evaluator, …) build
        # ``LLMConfig`` from a fixed set of fields without knowing about
        # the daily cap. When the orchestrator-level launcher exports
        # ``META_N_DAILY_BUDGET_USD`` and ``META_N_COST_LEDGER_DIR``,
        # auto-pick them up here so per-task ``llm()`` calls inside the
        # evaluated solve() also count against the cap. Without this,
        # only outer Ω/program-evolution calls were tracked and the cap
        # could be missed by a wide margin on solve()-heavy benchmarks
        # (S2D, LawBench, ARC).
        if not self.daily_budget_usd or self.daily_budget_usd <= 0:
            try:
                env_cap = float(os.environ.get("META_N_DAILY_BUDGET_USD", "0") or 0)
            except ValueError:
                env_cap = 0.0
            if env_cap > 0:
                self.daily_budget_usd = env_cap
        if self.cost_ledger_dir == "~/.meta_n_costs":
            env_dir = os.environ.get("META_N_COST_LEDGER_DIR", "").strip()
            if env_dir:
                self.cost_ledger_dir = env_dir
        # Same env-var fallback for the reservation so the openevolve
        # library and the meta-n driver stay in sync — both paths now
        # honour ``META_N_COST_RESERVATION_USD`` when set, with a parse-
        # safe fallback on a malformed value. The gate is against the None
        # sentinel (unset) so an EXPLICIT constructor value — including an
        # explicit 0.0 — is never clobbered by the env var. After the fallback
        # the still-unset case is coalesced to the documented 0.0 default.
        if self.cost_reservation_usd is None:
            try:
                env_res = os.environ.get("META_N_COST_RESERVATION_USD", "").strip()
                if env_res:
                    self.cost_reservation_usd = float(env_res)
            except ValueError:
                pass
        if self.cost_reservation_usd is None:
            self.cost_reservation_usd = 0.0
        # Same env-var fallback for the per-call request timeout. Slow local
        # reasoning models (e.g. LM Studio Gemma-QAT) can exceed the 360 s
        # default on heavy solve()-style prompts; META_N_LLM_REQUEST_TIMEOUT
        # lets a launcher raise it without a code change. Gated on the None
        # sentinel (unset) so an explicit constructor value — including an
        # explicit 360.0 — is never clobbered. After the fallback the still-
        # unset case is coalesced to the documented 360.0 default.
        if self.request_timeout is None:
            try:
                env_to = float(
                    os.environ.get("META_N_LLM_REQUEST_TIMEOUT", "").strip() or 0
                )
                if env_to > 0:
                    self.request_timeout = env_to
            except ValueError:
                pass
        if self.request_timeout is None:
            self.request_timeout = 360.0


class LLMClient:
    """Unified LLM client supporting OpenRouter, Azure, LMStudio, and vLLM."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

        # --- RPBE backend interlock (see meta_n/rpbe/backends.py) -----------
        # Resolved FIRST: the mock/local backends never build a transport, so
        # neither credentials nor a socket are needed. A paid provider is
        # refused unless LLM_BACKEND names it AND ALLOW_PAID_API carries the
        # opt-in token -- belt-and-braces, so a config that still carries a
        # real key cannot spend during unit tests or pipeline smokes.
        from meta_n.rpbe import backends as _rpbe_bk
        _kind = os.environ.get(_rpbe_bk.BACKEND_ENV, _rpbe_bk.MOCK).strip().lower()
        if _kind in _rpbe_bk.PAID_KINDS:
            _rpbe_bk.assert_paid_allowed(_kind)
            self._rpbe_backend = None      # DeepSeek IS this class
        else:
            self._rpbe_backend = _rpbe_bk.resolve_backend(_kind)
        # Hard call-count fuse. 0 disables it. Counts LOGICAL calls at the
        # entry of complete_with_breakdown; retries inside one call are not
        # separate logical calls (the transport layer owns those).
        try:
            self._rpbe_call_cap = int(
                os.environ.get(_rpbe_bk.MAX_CALLS_ENV, "0") or 0)
        except ValueError:
            self._rpbe_call_cap = 0
        self._rpbe_calls = 0

        # max_retries=0 on the SDK so OUR wrapper is the single source of
        # truth for retry policy. Otherwise the SDK retries N times AND
        # our wrapper retries N more times → up to (N+1)² attempts on
        # transient errors, with no log indication of the doubling.
        # ``timeout`` bounds a single call. httpx.Timeout breaks down the
        # phases (connect / read / write / pool) so a slow-stream wedge
        # past the read-idle threshold doesn't sit on a 10-minute SDK
        # default. See LLMConfig.request_timeout for the why.
        import httpx
        timeout = httpx.Timeout(
            self.config.request_timeout, connect=10.0,
        )
        # Optional extra request headers, JSON, from the environment. Needed for
        # relays that answer with a transfer-encoding the installed httpx
        # decoder cannot handle (observed: openai 3.14.1 pulls httpx2 2.13.0,
        # whose zstd path raises `TypeError: process() takes no keyword
        # arguments`). `{"Accept-Encoding": "identity"}` side-steps the decoder
        # entirely. Unset => None, so the request stays byte-identical.
        _extra_headers = None
        _raw_hdrs = os.environ.get("META_N_EXTRA_HEADERS_JSON", "").strip()
        if _raw_hdrs:
            try:
                _extra_headers = json.loads(_raw_hdrs)
            except ValueError:
                _extra_headers = None
        if self._rpbe_backend is not None:
            self._client = None            # offline backend: no transport
        elif self.config.backend == "azure":
            self._client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                api_version=self.config.azure_api_version,
                azure_endpoint=self.config.azure_endpoint or "",
                max_retries=0,
                timeout=timeout,
                default_headers=_extra_headers,
            )
        else:
            self._client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                max_retries=0,
                timeout=timeout,
                default_headers=_extra_headers,
            )
        # OpenRouter provider routing (extra_body passed to every request).
        # Azure ignores ``extra_body``; leaving it None there is correct.
        self._extra_body: dict | None = None
        if self.config.backend != "azure" and self.config.exclude_providers:
            self._extra_body = {"provider": {"ignore": self.config.exclude_providers}}
        # Reasoning-effort control rides the same extra_body passthrough. Merged
        # here (not only when exclude_providers is set) so it reaches every
        # backend; None leaves the request byte-identical.
        if self.config.reasoning_effort is not None:
            self._extra_body = {
                **(self._extra_body or {}),
                "reasoning_effort": self.config.reasoning_effort,
            }

        # Optional USD cost tracker. Enabled when a positive cap is set.
        self.cost_tracker: CostTracker | None = None
        if self.config.daily_budget_usd and self.config.daily_budget_usd > 0:
            # Fail-fast on unknown model: with a cap configured, we don't
            # want recording to silently skip per call and let the cap
            # never fire. Validate now so a typo surfaces at startup.
            from meta_n.utils.cost_tracker import get_pricing
            try:
                get_pricing(self.config.model)
            except KeyError as e:
                raise KeyError(
                    f"Cannot enable cost tracking: {e}. Add the model to "
                    f"meta_n/utils/cost_tracker.PRICING or set "
                    f"META_N_PRICING_OVERRIDE_JSON='{{...}}' before "
                    f"launching."
                ) from e
            self.cost_tracker = CostTracker(
                ledger_dir=self.config.cost_ledger_dir,
                daily_cap_usd=self.config.daily_budget_usd,
                reservation_usd=self.config.cost_reservation_usd,
                utc=self.config.cost_utc_day,
            )

        # Cumulative token accounting across all complete() calls. Read at
        # end-of-run to surface input/output split in summary.json without
        # changing the existing (text, total_tokens) return contract that
        # legacy callers (omega/solver/llm_helpers/agentic_solver) depend on.
        # `complete()` already returns total — this tracks the prompt /
        # completion breakdown the response carries.
        # Updated under an asyncio.Lock so concurrent calls don't race.
        self.cumulative_usage = {
            "prompt": 0, "completion": 0, "total": 0, "calls": 0,
            "cached": 0, "cost_usd": 0.0,
            "requests": 0, "failed_calls": 0, "retries": 0,
            "empty_responses": 0, "empty_escalations": 0,
            "reasoning": 0, "reasoning_reported_calls": 0,
        }
        self._usage_lock = asyncio.Lock()
        # Optional raw I/O logger. When set (typically by an orchestrator
        # before run() starts), every call records messages + response to
        # a JSONL file. Off by default — null logger is a no-op so unit
        # tests and library callers don't pay any I/O cost.
        self.io_logger = None  # type: ignore[assignment]

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        seed: int | None = None,
    ) -> tuple[str, int]:
        """
        Send a chat completion request.

        Retries transient errors (404 on OpenRouter, connection/timeout,
        JSON-decode, 429, 5xx) in this wrapper's own loop — throttle classes
        (429/5xx) back off from a >=15s base delay. SDK-internal retries are
        disabled (both clients are constructed with max_retries=0), so this
        wrapper is the single source of truth for retry policy.

        Side effect: each successful call increments ``self.cumulative_usage``
        with the prompt/completion/total token counts the API returned.

        Args:
            seed: optional per-request CRN seed (paired eval). Forwarded to
                ``chat.completions.create()`` ONLY when the backend is known to
                honour it (see ``_supports_request_seed``); otherwise dropped so
                the payload is byte-identical. ``None`` (default) ⇒ no seed on
                the wire (the default-off path).

        Returns:
            Tuple of (response_text, total_tokens_used)
        """
        text, _, _, total = await self.complete_with_breakdown(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed,
        )
        return text, total

    async def complete_with_breakdown(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        seed: int | None = None,
        _suppress_io_log: bool = False,
        _empty_retry_done: bool = False,
        _logical_call_id: str | None = None,
        _parent_request_id: int | None = None,
    ) -> tuple[str, int, int, int]:
        """
        Same as :meth:`complete` but returns the input/output token split.

        Args:
            _suppress_io_log: When True, skip ``self.io_logger.log()`` for
                this call. Used by ``llm_helpers.make_llm_func`` to prevent
                inner-LLM calls from double-logging — the inner path has
                its own dedicated logger writing to a separate JSONL.

        Returns:
            Tuple of (response_text, prompt_tokens, completion_tokens, total_tokens)

        Raises:
            BudgetExceededError: when a CostTracker is attached and
                today's spend has reached the cap. Inherits from
                BaseException so ``except Exception`` won't absorb it.
        """
        # --- RPBE backend override + backend-request fuse -------------------
        # The fuse counts REAL HTTP requests at the send site below, NOT
        # semantic calls: this method recurses for the empty-content escalation
        # under _suppress_io_log=True, which made 983 billed requests invisible
        # to the inner I/O log in the 2026-09-17 pilot. Here we only mint the
        # logical id that ties one call's requests together.
        from meta_n.rpbe import accounting as _acct
        if _logical_call_id is None:
            _logical_call_id = _acct.new_logical_call_id()
        _primary_request_id: int | None = None
        # Mock/Local short-circuit the transport entirely: no socket, no money.
        if getattr(self, "_rpbe_backend", None) is not None:
            _acct.record_offline(self._rpbe_backend.name, _logical_call_id)
            return await self._rpbe_backend.complete_with_breakdown(
                messages, temperature=temperature, max_tokens=max_tokens,
                seed=seed)
        # Old checkpoints and third-party test doubles may carry the original
        # six-key usage dictionary. Backfill the additive counters before any
        # increment so resuming one cannot fail with KeyError.
        async with self._usage_lock:
            for key in (
                "requests", "failed_calls", "retries", "empty_responses",
                "empty_escalations", "reasoning", "reasoning_reported_calls",
            ):
                self.cumulative_usage.setdefault(key, 0)
        last_error: Exception | None = None
        # Build kwargs once per call. Reasoning families (gpt-5.x, o*) want
        # ``max_completion_tokens`` and reject custom ``temperature``;
        # legacy chat models (gpt-4.1, gpt-4o, claude-on-azure, …) want
        # the old ``max_tokens`` and honour temperature.
        eff_max_tokens = max_tokens or self.config.max_tokens
        token_kwargs: dict
        if _uses_max_completion_tokens(self.config.model):
            token_kwargs = {"max_completion_tokens": eff_max_tokens}
        else:
            token_kwargs = {"max_tokens": eff_max_tokens}
        eff_temperature = temperature if temperature is not None else self.config.temperature
        temperature_kwargs: dict
        if _supports_custom_temperature(self.config.model):
            temperature_kwargs = {"temperature": eff_temperature}
        else:
            # Reasoning models only support temperature=1 (default). Omit
            # rather than send temperature=1 explicitly — keeps payloads
            # symmetric with the SDK's own default and avoids a future
            # validator change rejecting the explicit value.
            temperature_kwargs = {}

        # CRN / paired-eval seed (default-OFF). THREE gates collapse this to a
        # no-op: (1) caller passes seed=None unless paired_eval is on;
        # (2) capability gate — drop the seed unless the backend honours a
        # per-request seed; (3) transport gate — conditional splat omits the
        # kwarg ENTIRELY rather than sending ``seed=None`` (the SDK serializes an
        # explicit None as ``"seed": null``, which would NOT be byte-identical to
        # today). Mirrors the temperature_kwargs conditional-omit above.
        seed_kwargs: dict
        if seed is not None and _supports_request_seed(
            self.config.model, self.config.backend
        ):
            seed_kwargs = {"seed": seed}
        else:
            seed_kwargs = {}
        # ``extra_body`` is passed by IDENTITY when not seeding (never mutate the
        # shared ``self._extra_body`` dict built once at construction). The
        # OpenRouter seed path (if it is ever whitelisted in
        # ``_supports_request_seed``) would copy per-call; today the only
        # seed-honouring backend is Azure, whose ``create()`` takes a top-level
        # ``seed=`` kwarg and ignores ``extra_body`` — so no copy is needed.
        eff_extra_body = self._extra_body

        # Empty-content escalation is captured here and performed AFTER the
        # retry loop (audit #51): the recursive re-issue must NOT run inside the
        # parent ``try`` below, or a transient escalation error would be caught
        # by the parent ``except`` and re-arm the retry loop — re-issuing and
        # re-billing the already-successful base request. Tuple of
        # (pt, ct, tt, escalate_cap) for the booked first (empty) call.
        empty_escalation: tuple[int, int, int, int] | None = None

        for attempt in range(self.config.max_retries + 1):
            # Re-check the cap on every attempt — the retry backoff can
            # be tens of seconds, during which other processes (subprocess
            # workers, parallel orchestrator threads) may push the
            # cumulative ledger past the cap. Without re-checking, a
            # mid-retry budget hit would let attempt N+1 spend after the
            # cap was already breached.
            if self.cost_tracker is not None:
                self.cost_tracker.assert_under_cap()
            try:
                # --- backend-request fuse: COUNT AND CHECK, THEN SEND --------
                # Atomic across processes (flock) so subprocess workers share
                # one budget. Raising here means no request was issued.
                _kind = (_acct.EMPTY_ESCALATION if _empty_retry_done else
                         (_acct.TRANSPORT_RETRY if attempt > 0
                          else _acct.PRIMARY))
                _req_id = _acct.reserve(
                    _kind, logical_call_id=_logical_call_id,
                    parent_request_id=_parent_request_id)
                if _kind == _acct.PRIMARY:
                    _primary_request_id = _req_id
                # Count actual transport attempts, not just successful
                # responses. reserve() has returned, so the money fuse permits
                # this request and it is now about to be issued.
                async with self._usage_lock:
                    self.cumulative_usage["requests"] = int(
                        self.cumulative_usage.get("requests", 0) or 0) + 1
                    if attempt > 0:
                        self.cumulative_usage["retries"] = int(
                            self.cumulative_usage.get("retries", 0) or 0) + 1
                    if _empty_retry_done:
                        self.cumulative_usage["empty_escalations"] = int(
                            self.cumulative_usage.get(
                                "empty_escalations", 0) or 0) + 1
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    extra_body=eff_extra_body,
                    **temperature_kwargs,
                    **token_kwargs,
                    **seed_kwargs,
                )
                pt = ct = tt = 0
                cached = 0
                reasoning = 0
                reasoning_reported = False
                if response.usage is not None:
                    pt = int(getattr(response.usage, "prompt_tokens", 0) or 0)
                    ct = int(getattr(response.usage, "completion_tokens", 0) or 0)
                    tt = int(getattr(response.usage, "total_tokens", 0) or (pt + ct))
                    pt_details = getattr(response.usage, "prompt_tokens_details", None)
                    if pt_details is not None:
                        cached = int(getattr(pt_details, "cached_tokens", 0) or 0)
                    ct_details = getattr(
                        response.usage, "completion_tokens_details", None)
                    raw_reasoning = getattr(
                        ct_details, "reasoning_tokens", None)
                    if raw_reasoning is not None:
                        reasoning = int(raw_reasoning or 0)
                        reasoning_reported = True

                # Record cost in the daily ledger before bumping in-mem
                # counters, so a record() failure surfaces immediately.
                cost = 0.0
                if self.cost_tracker is not None and tt > 0:
                    cost = self.cost_tracker.record(
                        model=self.config.model,
                        prompt_tokens=pt,
                        completion_tokens=ct,
                        cached_tokens=cached,
                    )

                async with self._usage_lock:
                    self.cumulative_usage["prompt"] += pt
                    self.cumulative_usage["completion"] += ct
                    self.cumulative_usage["total"] += tt
                    self.cumulative_usage["calls"] += 1
                    self.cumulative_usage["cached"] += cached
                    self.cumulative_usage["cost_usd"] += cost
                    self.cumulative_usage["reasoning"] += reasoning
                    if reasoning_reported:
                        self.cumulative_usage["reasoning_reported_calls"] += 1

                if not response.choices:
                    logger.warning("LLM returned empty choices (tokens=%d)", tt)
                    async with self._usage_lock:
                        self.cumulative_usage["empty_responses"] += 1
                    if self.io_logger is not None and not _suppress_io_log:
                        self.io_logger.log(
                            messages=messages, response="",
                            model=self.config.model,
                            prompt_tokens=pt, completion_tokens=ct,
                            total_tokens=tt,
                            extra={"empty_choices": True, "cached_tokens": cached, "cost_usd": cost},
                        )
                    return "", pt, ct, tt
                text = response.choices[0].message.content or ""
                if not text.strip():
                    async with self._usage_lock:
                        self.cumulative_usage["empty_responses"] += 1
                # Surface a length-truncated completion so a reasoning model that
                # spent its whole budget on hidden reasoning_tokens (leaving
                # message.content empty) is LOGGED rather than silently becoming an
                # empty response downstream. finish_reason=='length' means the
                # cap (max_tokens / max_completion_tokens) was hit — raise the cap
                # to reserve room for the reasoning budget if content is empty.
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                if finish_reason == "length":
                    logger.warning(
                        "LLM response truncated (finish_reason='length', "
                        "completion_tokens=%d, cap=%d, content_len=%d) — the model "
                        "hit its token cap; for reasoning models raise max_tokens "
                        "above the reasoning budget to avoid empty content.",
                        ct, eff_max_tokens, len(text),
                    )
                # EMPTY-CONTENT ESCALATION: a reasoning / QAT model that burned its
                # whole budget on hidden reasoning returns finish_reason=='length'
                # with EMPTY content. Re-issue the SAME request ONCE at a higher cap
                # so it has room to finish reasoning AND emit content, rather than
                # silently returning "" (which authored an empty script => spurious
                # task failure on the builtin TB spine). Scoped to the EMPTY+length
                # case and bounded to a single escalation via _empty_retry_done. The
                # first (empty) call's tokens were already recorded above — both
                # calls are correctly accounted in the ledger / cumulative usage.
                escalate_cap = min(
                    self.config.empty_content_retry_max_tokens,
                    _EMPTY_ESCALATION_SCALE * eff_max_tokens,
                )
                if (
                    finish_reason == "length"
                    and not text.strip()
                    and not _empty_retry_done
                    and escalate_cap > eff_max_tokens
                ):
                    logger.warning(
                        "Empty content at cap=%d (finish_reason='length') — "
                        "re-issuing once at higher cap=%d.",
                        eff_max_tokens, escalate_cap,
                    )
                    # Audit #65 — log the first (empty, token-burning) round-trip
                    # to the io_logger JSONL before escalating. Without this the
                    # only io_logger.log on this method (fall-through path below)
                    # is skipped by the escalation return, so the wasted first
                    # call would silently never appear in the I/O audit trail —
                    # unlike the sibling empty_choices path, which DOES log it.
                    if self.io_logger is not None and not _suppress_io_log:
                        self.io_logger.log(
                            messages=messages, response=text,
                            model=self.config.model,
                            prompt_tokens=pt, completion_tokens=ct,
                            total_tokens=tt,
                            extra={
                                "empty_content_escalated": True,
                                "cached_tokens": cached,
                                "cost_usd": cost,
                            },
                        )
                    # Audit #51 — capture the booked first-call tokens and BREAK
                    # out of the parent retry loop. The escalation re-issue is
                    # performed AFTER the loop (outside this try) so a transient
                    # escalation error cannot be caught by the parent ``except``
                    # and re-bill the already-successful base request.
                    empty_escalation = (pt, ct, tt, escalate_cap)
                    break
                if self.io_logger is not None and not _suppress_io_log:
                    self.io_logger.log(
                        messages=messages, response=text,
                        model=self.config.model,
                        prompt_tokens=pt, completion_tokens=ct,
                        total_tokens=tt,
                        extra={"cached_tokens": cached, "cost_usd": cost} if (cached or cost) else None,
                    )
                return text, pt, ct, tt
            except (NotFoundError, APIConnectionError, APITimeoutError,
                    JSONDecodeError, RateLimitError, InternalServerError) as e:
                async with self._usage_lock:
                    self.cumulative_usage["failed_calls"] += 1
                last_error = e
                if attempt < self.config.max_retries:
                    # Use a longer base delay for rate-limit/server errors
                    # since the upstream provider needs time to recover.
                    is_throttle = isinstance(e, (RateLimitError, InternalServerError))
                    base = max(self.config.retry_base_delay, 15.0) if is_throttle else self.config.retry_base_delay
                    delay = min(base * (2 ** attempt), self.config.retry_max_delay)
                    sleep_time = delay * random.uniform(0.5, 1.5)
                    logger.warning(
                        "LLM request failed (attempt %d/%d): %s %s — retrying in %.1fs",
                        attempt + 1,
                        self.config.max_retries + 1,
                        type(e).__name__,
                        str(e)[:200],
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(
                        "LLM request failed after %d attempts: %s %s",
                        self.config.max_retries + 1,
                        type(e).__name__,
                        str(e)[:200],
                    )
        if empty_escalation is not None:
            # Audit #51 — single escalation re-issue, performed OUTSIDE the
            # parent try. The first (empty) call's tokens were already booked
            # into the ledger / cumulative usage above; folding (pt,ct,tt) into
            # the recursive return keeps the audit token tuple from
            # undercounting the wasted escalation call. If THIS call raises a
            # retryable error after exhausting its own internal retries, it
            # propagates to the caller (it does NOT re-arm the parent loop /
            # re-bill the base request).
            pt, ct, tt, escalate_cap = empty_escalation
            text2, pt2, ct2, tt2 = await self.complete_with_breakdown(
                messages,
                temperature=temperature,
                max_tokens=escalate_cap,
                seed=seed,
                _suppress_io_log=_suppress_io_log,
                _empty_retry_done=True,
                # Same logical call, chained to the request it is escalating.
                _logical_call_id=_logical_call_id,
                _parent_request_id=_primary_request_id,
            )
            return text2, pt + pt2, ct + ct2, tt + tt2
        raise last_error  # type: ignore[misc]
