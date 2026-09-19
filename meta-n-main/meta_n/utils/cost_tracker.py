"""USD cost tracking with a per-day spend cap.

Wraps every LLM call site to:
  1. Refuse to call when today's accumulated spend is at/over the cap
     (``assert_under_cap`` — raises ``BudgetExceededError``).
  2. Record actual cost in a daily JSONL ledger after each successful call
     (``record`` — appends one line under ``flock`` so subprocess workers
     and the parent share one file safely).

``BudgetExceededError`` inherits from ``BaseException`` (not ``Exception``)
on purpose: integration code in ``co_bench.py`` / ``openevolve.py`` /
``text_classification.py`` has many ``except Exception`` blocks that
otherwise absorb the budget signal and let the run continue, defeating
the hard stop. Following the standard library's precedent for
``KeyboardInterrupt`` / ``SystemExit``, a budget hit is treated as a
process-control signal that bypasses normal error handling and propagates
to the orchestrator's checkpoint-save path.

The ledger is human-readable JSONL — one line per call — and rotated daily
so it stays grep-friendly:

    ~/.meta_n_costs/2026-05-08.jsonl
    ~/.meta_n_costs/2026-05-09.jsonl
    ...

Use ``CostTracker.today_total_usd()`` or the bundled ``cost_report`` CLI
to inspect spend mid-run.
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from meta_n.utils.flock_append import flock_append_bytes

logger = logging.getLogger(__name__)


class BudgetExceededError(BaseException):
    """Hard stop: today's spend has reached the daily budget cap.

    Inherits from BaseException so broad ``except Exception`` blocks in
    benchmark integrations (subprocess wrappers, eval harnesses) cannot
    absorb the signal. A budget hit must propagate to the orchestrator's
    checkpoint-save path, the same way KeyboardInterrupt does.
    """


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens. ``cached_per_M`` is the discounted rate when
    the provider reports ``prompt_tokens_details.cached_tokens``; equals
    ``input_per_M`` when no cache discount applies."""
    input_per_M: float
    output_per_M: float
    cached_per_M: float | None = None


# Per-model pricing as of 2026-05-08. Sources tracked in PR description.
# Keys are deployment / model-family names. Azure deployments typically
# match the model family ("gpt-4.1"), but if the deployment was given a
# custom name, add an alias here.
PRICING: dict[str, ModelPricing] = {
    # Azure / OpenAI Chat-completions models
    "gpt-4.1":          ModelPricing(input_per_M=2.00, output_per_M=8.00),
    "gpt-4.1-mini":     ModelPricing(input_per_M=0.40, output_per_M=1.60),
    "gpt-4.1-nano":     ModelPricing(input_per_M=0.10, output_per_M=0.40),
    "gpt-5":            ModelPricing(input_per_M=1.25, output_per_M=10.00),
    "gpt-5.1":          ModelPricing(input_per_M=1.25, output_per_M=10.00),
    "gpt-5.1-codex":    ModelPricing(input_per_M=1.25, output_per_M=10.00),
    "gpt-5.2":          ModelPricing(input_per_M=1.75, output_per_M=14.00, cached_per_M=0.175),
    "gpt-5.2-codex":    ModelPricing(input_per_M=1.75, output_per_M=14.00, cached_per_M=0.175),

    # OpenRouter — Gemma 4 31B Instruct. Pricing as advertised on OpenRouter
    # at the time of writing (override via META_N_PRICING_OVERRIDE_JSON if
    # the rate changes).
    "google/gemma-4-31b-it": ModelPricing(input_per_M=0.20, output_per_M=0.50),

    # Local-served Gemma 4 31B (QAT) via an OpenAI-compatible endpoint
    # (e.g. LM Studio / llama.cpp at http://127.0.0.1:1234/v1). Self-hosted,
    # so real spend is $0 — but get_pricing()/CostGuard fail-fast on any model
    # absent from PRICING (cost_tracker.py:128-137, budget.py:131-138), which
    # would block external-agent spine runs that require --daily-budget-usd > 0.
    # A $0 entry lets the budget>0 gate pass while spend stays $0. (Equivalent
    # to the no-edit escape hatch META_N_PRICING_OVERRIDE_JSON.)
    "google/gemma-4-31b-qat": ModelPricing(input_per_M=0.0, output_per_M=0.0),

    # Local-served Qwen3.6 35B A3B (MoE; ~3B active) via the same OpenAI-compatible
    # endpoint. Self-hosted -> $0; the entry only exists so CostGuard's fail-fast
    # does not block the spine on --daily-budget-usd > 0. Same rationale as the
    # gemma-qat line above. Used as the same-model-at-all-layers agentic backbone
    # (concise bounded reasoning + MoE speed, unlike gemma-qat's over-reasoning).
    "qwen/qwen3.6-35b-a3b": ModelPricing(input_per_M=0.0, output_per_M=0.0),

    # OpenRouter — Qwen3 Coder 30B A3B Instruct (MoE coding model).
    # Pricing per OpenRouter listing (no prompt-cache discount).
    "qwen/qwen3-coder-30b-a3b-instruct": ModelPricing(
        input_per_M=0.07, output_per_M=0.27,
    ),
}


# Parse cache keyed on the RAW env string. get_pricing() runs on every
# recorded LLM call; the override JSON only needs re-parsing when the env
# var's value actually changes (changed raw -> cache miss -> re-parse, so
# a mid-process env change is still honored). Also latches the empty and
# malformed cases, so the malformed-JSON warning logs once per distinct
# value instead of once per call. Per-process, like the env read itself.
_PRICING_OVERRIDE_CACHE: tuple[str, dict[str, ModelPricing]] | None = None


def _load_pricing_overrides() -> dict[str, ModelPricing]:
    """Optional ``META_N_PRICING_OVERRIDE_JSON`` env var lets callers add
    or override a model's pricing without editing this file. Format:

        {"my-deployment": {"input": 1.5, "output": 6.0, "cached": 0.15}}
    """
    global _PRICING_OVERRIDE_CACHE
    raw = os.environ.get("META_N_PRICING_OVERRIDE_JSON", "").strip()
    if _PRICING_OVERRIDE_CACHE is not None and _PRICING_OVERRIDE_CACHE[0] == raw:
        # Shared by reference: the only call site (get_pricing) does
        # membership tests and lookups of frozen dataclasses, so the dict is
        # effectively read-only.
        return _PRICING_OVERRIDE_CACHE[1]
    if not raw:
        _PRICING_OVERRIDE_CACHE = (raw, {})
        return _PRICING_OVERRIDE_CACHE[1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Ignoring malformed META_N_PRICING_OVERRIDE_JSON: %s", e)
        _PRICING_OVERRIDE_CACHE = (raw, {})
        return _PRICING_OVERRIDE_CACHE[1]
    out: dict[str, ModelPricing] = {}
    for name, p in (data or {}).items():
        try:
            out[name] = ModelPricing(
                input_per_M=float(p["input"]),
                output_per_M=float(p["output"]),
                cached_per_M=float(p["cached"]) if "cached" in p else None,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Ignoring malformed pricing for %s: %s", name, e)
    _PRICING_OVERRIDE_CACHE = (raw, out)
    return out


def get_pricing(model: str) -> ModelPricing:
    """Return pricing for ``model``, with override env var taking precedence.

    Raises KeyError with a clear hint if the model is unknown.
    """
    overrides = _load_pricing_overrides()
    if model in overrides:
        return overrides[model]
    if model in PRICING:
        return PRICING[model]
    known = sorted(set(PRICING) | set(overrides))
    raise KeyError(
        f"No pricing for model {model!r}. Either add it to "
        f"meta_n/utils/cost_tracker.py PRICING, or set "
        f"META_N_PRICING_OVERRIDE_JSON='{{\"{model}\": "
        f"{{\"input\": <usd_per_M>, \"output\": <usd_per_M>}}}}'. "
        f"Known: {known}"
    )


def compute_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """USD cost for one call. ``cached_tokens`` is subtracted from
    ``prompt_tokens`` and billed at the cached rate when available."""
    p = get_pricing(model)
    cached = max(0, min(int(cached_tokens or 0), int(prompt_tokens or 0)))
    uncached_input = max(0, int(prompt_tokens or 0) - cached)
    cached_rate = p.cached_per_M if p.cached_per_M is not None else p.input_per_M
    cost = (
        (uncached_input / 1_000_000.0) * p.input_per_M
        + (cached / 1_000_000.0) * cached_rate
        + (int(completion_tokens or 0) / 1_000_000.0) * p.output_per_M
    )
    return float(cost)


class CostTracker:
    """File-backed daily spend tracker, safe across processes.

    The ledger lives at ``<ledger_dir>/<YYYY-MM-DD>.jsonl`` (local date by
    default; set ``utc=True`` to roll at UTC midnight instead). Every call
    appends one JSONL record under ``fcntl.flock`` so multiple subprocess
    workers can share the same file. ``today_total_usd()`` re-scans the
    file on each call — slow O(n) per check but n is hours-of-history
    bounded; correctness over speed.
    """

    def __init__(
        self,
        ledger_dir: str | Path,
        daily_cap_usd: float,
        reservation_usd: float = 0.0,
        utc: bool = False,
    ):
        self.ledger_dir = Path(ledger_dir).expanduser()
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.daily_cap_usd = float(daily_cap_usd)
        self.reservation_usd = float(reservation_usd)
        # Fail fast on a misconfigured reservation: the effective stop
        # threshold is (cap - reservation), so reservation >= cap means the
        # FIRST call would raise BudgetExceededError with $0 spent — a
        # misconfiguration masquerading as budget exhaustion. Single
        # chokepoint: covers llm_client, azure_compat, and CostGuard paths.
        if self.reservation_usd < 0.0:
            raise ValueError(
                f"reservation_usd must be >= 0 (got ${self.reservation_usd:.4f}): a negative "
                f"reservation would raise the effective stop threshold ABOVE the daily cap."
            )
        if self.daily_cap_usd > 0 and self.reservation_usd >= self.daily_cap_usd:
            raise ValueError(
                f"reservation_usd (${self.reservation_usd:.2f}) must be smaller than "
                f"daily_cap_usd (${self.daily_cap_usd:.2f}): the effective threshold "
                f"(cap - reservation) would be <= 0 and the very first call would raise "
                f"BudgetExceededError with $0 spent. Lower the reservation (azure_compat's "
                f"install_cost_tracking defaults it to $1.00) or raise --daily-budget-usd."
            )
        self.utc = bool(utc)
        # Within-process lock prevents two asyncio tasks from racing on
        # the same fd. Cross-process safety is handled by flock below.
        self._lock = threading.Lock()
        # Cache the soft-warn flag so we only print the 80% banner once
        # per process. Subprocess workers each get their own copy, which
        # is the right behaviour — the banner should fire in any process
        # that observes the threshold.
        self._warned_80pct = False
        # Day the latch was last set, so the banner fires once PER DAY (not
        # once per process): the daily spend basis re-baselines at midnight
        # via _today_path(), so the latch must be cleared on day rollover.
        self._warned_80pct_day: str | None = None

    def _today_key(self) -> str:
        if self.utc:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return datetime.now().strftime("%Y-%m-%d")

    def _today_path(self) -> Path:
        return self.ledger_dir / f"{self._today_key()}.jsonl"

    def today_total_usd(self) -> float:
        """Sum ``cost_usd`` across today's ledger. Returns 0.0 if no
        ledger file exists yet (start-of-day or first run)."""
        path = self._today_path()
        total = 0.0
        # Shared lock is enough — concurrent reads are fine; the flock
        # protects against a torn read mid-append. The no-file case (start
        # of day / first run) is handled here rather than by a pre-check,
        # which would be redundant and racy (TOCTOU).
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except FileNotFoundError:
            return 0.0
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            try:
                with os.fdopen(fd, "r", closefd=False) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            val = float(rec.get("cost_usd", 0.0))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            # A truncated tail line shouldn't sink the
                            # whole read — skip and continue.
                            continue
                        # Defense-in-depth: json.loads parses a literal NaN /
                        # Infinity WITHOUT raising (allow_nan default), so the
                        # except above does not catch it. Skip a non-finite
                        # cost so an already-persisted poisoned line cannot
                        # turn the whole total into NaN and disable the day cap.
                        if not math.isfinite(val):
                            continue
                        total += val
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        return total

    def assert_under_cap(self) -> None:
        """Raise BudgetExceededError if today's spend is at or over the
        cap (minus reservation). Call before every LLM request."""
        today = self.today_total_usd()
        threshold = self.daily_cap_usd - self.reservation_usd
        # Reset the soft-warn latch on a day rollover so the banner fires
        # once per day (the daily spend basis re-baselines at midnight).
        today_key = self._today_key()
        if self._warned_80pct_day != today_key:
            self._warned_80pct = False
        # Soft warn at 80% of the EFFECTIVE threshold (not the raw cap), so
        # the early-warning still precedes the hard stop when a non-trivial
        # reservation pulls the threshold below 0.8*cap.
        warn_at = 0.80 * threshold
        if today >= warn_at and not self._warned_80pct:
            logger.warning(
                "[CostTracker] Daily spend at $%.2f / $%.2f cap (>=80%%). "
                "Run will halt at $%.2f.",
                today, self.daily_cap_usd, threshold,
            )
            self._warned_80pct = True
            self._warned_80pct_day = today_key
        if today >= threshold:
            raise BudgetExceededError(
                f"Daily budget reached: ${today:.4f} >= ${threshold:.4f} "
                f"(${self.daily_cap_usd:.2f} cap - ${self.reservation_usd:.2f} "
                f"reservation). Ledger: {self._today_path()}. "
                f"Resume tomorrow or pass --daily-budget-usd <higher>."
            )

    def _append_ledger_record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        cost: float,
        extra: dict | None,
    ) -> None:
        """Build one ledger record and append it under lock + flock.

        Single source for the persisted line shape shared by :meth:`record`
        and :meth:`record_usd` — key order and serialization flags are a
        stable, grep/jq-consumed on-disk contract.
        """
        record = {
            "ts": time.time(),
            "model": model,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "cached_tokens": int(cached_tokens or 0),
            "cost_usd": round(cost, 6),
            "pid": os.getpid(),
        }
        if extra:
            record["extra"] = extra
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            flock_append_bytes(self._today_path(), line)

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        extra: dict | None = None,
    ) -> float:
        """Append one ledger line for a successful call. Returns the
        cost (so callers can log/sum)."""
        cost = compute_cost_usd(model, prompt_tokens, completion_tokens, cached_tokens)
        self._append_ledger_record(
            model, prompt_tokens, completion_tokens, cached_tokens, cost, extra,
        )
        return cost

    def record_usd(
        self,
        model: str,
        cost_usd: float,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        extra: dict | None = None,
    ) -> float:
        """Append a ledger line with a PRECOMPUTED cost, bypassing ``PRICING``.

        Mirrors :meth:`record` exactly — same daily JSONL file, same
        ``flock``-guarded atomic append, same record shape — except the
        ``cost_usd`` field is taken verbatim from the caller instead of being
        derived from the ``PRICING`` table via :func:`compute_cost_usd`. This
        lets externally-priced spend (e.g. OpenHands' ``accumulated_cost``, or
        a Terminus-2 cost we priced ourselves from token counts) land in
        :meth:`today_total_usd` on its true USD basis, with no requirement that
        ``model`` exist in ``PRICING``.

        The token fields are recorded for auditability only; they do **not**
        influence ``cost_usd``. Callers should stamp ``extra`` (e.g.
        ``{"source": "openhands", "basis": "native_usd"}``) so ledger lines are
        attributable. Returns the (unchanged) ``cost_usd`` so callers can
        log/sum it.
        """
        cost = float(cost_usd or 0.0)
        # Clamp a non-finite/negative precomputed cost to $0 before it reaches
        # the ledger: ``x or 0.0`` does NOT coerce NaN (NaN is truthy) nor a
        # negative, and a persisted ``"cost_usd": NaN`` line makes
        # today_total_usd() NaN, which silently defeats the daily cap for the
        # rest of the day and across restart.
        if not (math.isfinite(cost) and cost >= 0.0):
            logger.warning(
                "[CostTracker] Dropping non-finite/negative cost_usd %r for "
                "model %s (recording $0.0 so a poisoned value cannot defeat "
                "the daily cap).", cost_usd, model,
            )
            cost = 0.0
        self._append_ledger_record(
            model, prompt_tokens, completion_tokens, cached_tokens, cost, extra,
        )
        return cost
