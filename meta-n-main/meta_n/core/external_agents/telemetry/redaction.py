"""Secret masking + priced-model resolution for the external-agents telemetry.

Split out of the former single-module ``telemetry.py`` (mechanical move; see
the package ``__init__`` for the full design contract).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("meta_n.external_agents.telemetry")

__all__ = [
    "redact",
]

# ---------------------------------------------------------------------------
# Redaction (plan §7.9)
# ---------------------------------------------------------------------------

_REDACTED = "***REDACTED***"

#: Ordered (broadest-first inside each family) secret patterns. Matching is
#: case-insensitive on the assignment keys; the token shapes are matched as-is.
#: Covers OPENAI/OPENROUTER/ANTHROPIC/AZURE keys, ``GH_TOKEN``, ``sk-…`` keys,
#: ``Bearer …`` headers and ``ghp_…`` GitHub PATs (plan §7.9).
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # KEY=value / KEY: value / KEY "value" assignments for the named providers,
    # INCLUDING the JSON / dict-repr quoted-key form ``"KEY": "value"`` /
    # ``{'KEY': 'value'}``. The separator group accepts an OPTIONAL closing quote
    # before the ``:``/``=`` so a key immediately followed by ``"`` (the common
    # serialized form) still matches and its value is masked — without it
    # ``"HF_TOKEN": "hf_…"`` and ``"AZURE_OPENAI_API_KEY": "…"`` escaped entirely.
    re.compile(
        r"(?i)\b("
        r"OPENAI_API_KEY|OPENROUTER_API_KEY|ANTHROPIC_API_KEY|"
        r"AZURE_OPENAI_API_KEY|AZURE_OPENAI_KEY|GH_TOKEN|GITHUB_TOKEN|"
        r"HF_TOKEN|HUGGINGFACE_TOKEN"
        r")([\"']?\s*[:=]\s*|\s+)(\"|')?[^\s\"']+(\"|')?"
    ),
    # OpenAI/Anthropic-style ``sk-…`` and project ``sk-proj-…`` keys.
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    # OpenRouter ``or-…`` keys in bare (non-assignment) form. The assignment-key
    # pattern [0] already masks ``OPENROUTER_API_KEY=…``; this covers a raw value
    # echoed without its key prefix.
    re.compile(r"\bor-[A-Za-z0-9]{20,}"),
    # HuggingFace ``hf_…`` tokens in bare (non-assignment) form. Pattern [0] masks
    # the ``HF_TOKEN``/``HUGGINGFACE_TOKEN`` assignment forms; this covers an
    # ``hf_`` value echoed without its key prefix (no other bare-value pattern
    # catches it, so an HF token would otherwise escape entirely).
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    # ``Bearer <token>`` Authorization headers.
    re.compile(r"(?i)\bBearer\s+\S+"),
    # GitHub personal-access / OAuth / app tokens.
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
)


def redact(text: str) -> str:
    """Mask provider secrets in free text before it is written (plan §7.9).

    Replaces OPENAI/OPENROUTER/ANTHROPIC/AZURE API keys (in ``KEY=value`` form),
    ``GH_TOKEN``/``GITHUB_TOKEN``, bare ``sk-…`` and ``or-…`` keys, ``Bearer …``
    headers and ``ghp_…`` GitHub tokens with ``***REDACTED***``. Non-secret text is
    returned unchanged. Idempotent and never raises (returns the input on error).

    Args:
        text: Arbitrary free text (a transcript line, error summary, feedback…).

    Returns:
        The redacted text. ``""`` for falsy input.
    """
    if not text:
        return text or ""
    try:
        out = str(text)
        # Named assignments keep the key but mask the value.
        out = _SECRET_PATTERNS[0].sub(
            lambda m: f"{m.group(1)}{m.group(2) or '='}{_REDACTED}", out
        )
        for pat in _SECRET_PATTERNS[1:]:
            out = pat.sub(_REDACTED, out)
        return out
    except Exception:  # noqa: BLE001 - redaction must never sink a write
        logger.warning("redact() failed; emitting empty string for safety")
        return ""


def _resolve_priced_model(solver: object, backend: object) -> str:
    """Best-effort: recover the RAW pricing-table model id off the solver chain.

    Returns the same string the outer ``LLMClient`` prices with — i.e.
    ``backend.llm_client.config.model`` *unnormalized* (the ``PRICING`` table keys
    are mixed bare/prefixed, e.g. ``gpt-5.2`` vs ``google/gemma-4-31b-it``, and the
    client looks them up with the full ``config.model``). Used ONLY to derive a
    display-only USD cost for an outer-basis builtin row in
    :meth:`AgentTelemetry.finish_record`. Never raises — returns ``""`` when the
    chain is absent (e.g. an inner-basis backend, or a test double), in which case
    no display cost is derived and ``cost_usd`` stays ``0.0``.
    """
    try:
        # builtin: ``backend.llm_client``; builtin_tb: ``backend._llm_client``.
        client = getattr(backend, "llm_client", None) or getattr(
            backend, "_llm_client", None
        )
        if client is None:
            client = getattr(solver, "llm_client", None)
        config = getattr(client, "config", None)
        model = getattr(config, "model", None)
        return str(model) if model else ""
    except Exception:  # noqa: BLE001 - resolution is best-effort, must not raise
        return ""
