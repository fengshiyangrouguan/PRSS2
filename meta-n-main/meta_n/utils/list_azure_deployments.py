"""Discover Azure OpenAI models and deployments visible to the current key.

Student / inference-only keys cannot hit the control-plane
``/openai/deployments`` listing endpoint (403/404). To work around that,
this tool combines two data-plane requests that DO work with an
inference key:

  1. ``GET /openai/models?api-version=...`` — every model the resource
     can serve, with capability flags. Filtered to chat-completion-capable
     and non-deprecated entries. This is the catalog of *what could be*
     deployed on this resource.

  2. A tiny ``chat.completions.create`` probe against each chat-capable
     model id and a small set of common deployment name aliases. A 200
     response confirms a deployment exists at that name; a 404
     ("DeploymentNotFound") confirms it doesn't. This is the catalog of
     *what is actually* deployed.

Set ``--no-probe`` to skip the second step (saves a handful of cheap
calls but only tells you what's deployable, not what's deployed).

Usage:

    python -m meta_n.utils.list_azure_deployments
    python -m meta_n.utils.list_azure_deployments --no-probe
    python -m meta_n.utils.list_azure_deployments --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from urllib.parse import urljoin

import httpx

from meta_n.utils.cost_tracker import PRICING


# Common deployment name aliases that students get on top of the model
# id (Azure deployments are usually named after the model family but
# sometimes carry suffixes). We probe these in addition to whatever
# turns up in /openai/models.
EXTRA_PROBE_NAMES = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-5",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.2",
    "gpt-5.2-codex",
]


async def _list_models(
    endpoint: str, api_key: str, api_version: str
) -> list[dict]:
    """Hit the data-plane ``/openai/models`` endpoint, which returns the
    catalog of models the resource can serve."""
    if not endpoint.endswith("/"):
        endpoint = endpoint + "/"
    url = urljoin(endpoint, f"openai/models?api-version={api_version}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, headers={"api-key": api_key})
        r.raise_for_status()
        body = r.json() or {}
    if isinstance(body, dict) and "data" in body:
        return list(body["data"])
    return []


async def _probe_deployment(
    endpoint: str, api_key: str, api_version: str, deployment: str
) -> tuple[bool, str]:
    """Send a 1-token chat probe to confirm whether ``deployment`` exists.

    Returns ``(deployed, info)`` where ``info`` is either the model id
    Azure echoed back on success, or a short error description on 4xx.
    Network/timeout errors are reported as not-deployed with the error.

    gpt-5.x and o-series reasoning deployments reject ``max_tokens`` and
    custom ``temperature`` — when we see Azure's "unsupported parameter"
    400, we retry once with ``max_completion_tokens`` and no temperature.
    A 200 on the retry still confirms the deployment exists.
    """
    if not endpoint.endswith("/"):
        endpoint = endpoint + "/"
    url = urljoin(
        endpoint,
        f"openai/deployments/{deployment}/chat/completions?api-version={api_version}",
    )

    async def _send(payload: dict) -> httpx.Response:
        async with httpx.AsyncClient(timeout=20.0) as client:
            return await client.post(
                url, headers={"api-key": api_key, "content-type": "application/json"},
                json=payload,
            )

    # First attempt: legacy ``max_tokens`` + temperature=0 (works for
    # gpt-4.1, gpt-4o, claude-on-azure, llama, etc.).
    payload_v1 = {
        "messages": [{"role": "user", "content": "."}],
        "max_tokens": 1,
        "temperature": 0,
    }
    try:
        r = await _send(payload_v1)
    except httpx.HTTPError as e:
        return False, f"net_error: {e!r}"

    # If the deployment is a reasoning model, retry with the new name.
    if r.status_code == 400 and (
        "max_tokens" in r.text or "max_completion_tokens" in r.text
        or "temperature" in r.text
    ):
        # Reasoning models burn hidden chain-of-thought tokens against the
        # same budget; ``max_completion_tokens=1`` returns a 400 about
        # "max_tokens or model output limit was reached" before any
        # visible output. Use 16 — small enough to keep probe cost
        # negligible, large enough to produce a real response.
        payload_v2 = {
            "messages": [{"role": "user", "content": "."}],
            "max_completion_tokens": 16,
        }
        try:
            r = await _send(payload_v2)
        except httpx.HTTPError as e:
            return False, f"net_error_v2: {e!r}"

    if r.status_code == 200:
        try:
            body = r.json()
            return True, body.get("model") or "ok"
        except Exception:  # noqa: BLE001
            return True, "ok"
    if r.status_code == 404:
        return False, "404 (no such deployment)"
    if r.status_code == 401:
        return False, "401 (auth)"
    if r.status_code == 429:
        # Rate limit is not a "no deployment" answer — treat as deployed.
        return True, "429 (rate limited; deployment exists)"
    if r.status_code == 400 and "unsupported" in r.text.lower():
        # Deployment exists but doesn't speak chat-completions (e.g.
        # codex variants only respond on the Responses API). Mark as
        # not-deployed-for-our-use-case but note the situation.
        return False, "exists but no chat-completions (try Responses API)"
    if r.status_code == 400 and (
        "max_tokens" in r.text or "model output limit" in r.text
    ):
        # Reasoning model returned 400 about the token budget — that
        # only happens when the deployment IS live and processed our
        # request. Treat as deployed; the user just needs to send a
        # bigger ``max_completion_tokens``.
        return True, "deployed (reasoning model — needs larger max_completion_tokens)"
    return False, f"http {r.status_code}: {r.text[:120]}"


def _is_deprecated(m: dict) -> bool:
    """Treat deprecation.inference < now as out-of-life."""
    import time
    dep = m.get("deprecation") or {}
    end = dep.get("inference")
    if isinstance(end, (int, float)):
        return float(end) < time.time()
    return False


def _looks_chat(m: dict) -> bool:
    """Best-effort filter for chat-completion models: capability says so
    OR the id starts with a known chat family prefix. Image / embeddings
    / audio models are excluded."""
    if _is_deprecated(m):
        return False
    cap = m.get("capabilities") or {}
    if cap.get("chat_completion"):
        return True
    # Some preview entries lack the chat_completion flag but are chat
    # under the hood. Match by id prefix as a fallback.
    mid = str(m.get("id") or "").lower()
    return any(mid.startswith(p) for p in (
        "gpt-3.5", "gpt-4", "gpt-5", "o1", "o3", "o4",
    )) and not any(mid.startswith(b) for b in (
        "gpt-4o-realtime", "gpt-4o-audio", "gpt-4o-transcribe",
    ))


def _format_table(rows: list[tuple[str, ...]]) -> str:
    """Column-aligned table for CLI output. ``rows[0]`` is the header; a
    dashed separator row is emitted after it. Trailing ljust padding on the
    last column is intentional (kept byte-stable for diffing probe logs)."""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for i, r in enumerate(rows):
        out.append("  ".join(c.ljust(widths[j]) for j, c in enumerate(r)))
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def _format_models(models: list[dict]) -> str:
    if not models:
        return "(no chat-capable models returned)"
    rows = [("Model id", "Status", "$ in / out / cached / 1M")]
    for m in models:
        mid = str(m.get("id") or "?")
        status = m.get("status") or "?"
        pricing_str = "(unknown)"
        if mid in PRICING:
            p = PRICING[mid]
            pricing_str = (
                f"${p.input_per_M:.2f} / ${p.output_per_M:.2f}"
                + (f" / ${p.cached_per_M:.3f}" if p.cached_per_M is not None else "")
            )
        rows.append((mid, str(status), pricing_str))
    return _format_table(rows)


def _format_probes(probes: list[dict]) -> str:
    if not probes:
        return "(no deployment probes ran)"
    rows = [("Deployment", "Deployed?", "Notes / model echoed")]
    for p in probes:
        rows.append((
            p["deployment"],
            "yes" if p["deployed"] else "no",
            p["info"],
        ))
    return _format_table(rows)


async def _run(args):
    models = await _list_models(args.endpoint, args.api_key, args.api_version)
    chat_models = [m for m in models if _looks_chat(m)]
    chat_models.sort(key=lambda m: str(m.get("id", "")))

    probes: list[dict] = []
    if not args.no_probe:
        # Probe every chat-capable model id, plus a curated alias list.
        candidate_names = sorted({
            *(str(m.get("id") or "") for m in chat_models),
            *EXTRA_PROBE_NAMES,
        })
        candidate_names = [n for n in candidate_names if n]
        # Skip anything obviously not chat (image/audio/etc.) to keep the
        # probe budget down — these were already filtered by _looks_chat.
        sem = asyncio.Semaphore(4)

        async def one(name):
            async with sem:
                deployed, info = await _probe_deployment(
                    args.endpoint, args.api_key, args.api_version, name,
                )
                return {"deployment": name, "deployed": deployed, "info": info}

        probes = await asyncio.gather(*[one(n) for n in candidate_names])

    return chat_models, probes


def main():
    parser = argparse.ArgumentParser(
        description="List Azure OpenAI models and probe deployments visible to the current API key."
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        help="Azure OpenAI endpoint (default: $AZURE_OPENAI_ENDPOINT)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        help="Azure OpenAI API key (default: $AZURE_OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--api-version",
        default="2024-12-01-preview",
        help="Azure OpenAI API version (default: 2024-12-01-preview)",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the per-deployment probe step (only list deployable models).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of formatted tables.",
    )
    args = parser.parse_args()

    if not args.endpoint or not args.api_key:
        sys.exit(
            "ERROR: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set "
            "(or passed via --endpoint / --api-key)."
        )

    try:
        chat_models, probes = asyncio.run(_run(args))
    except httpx.HTTPStatusError as e:
        sys.exit(
            f"ERROR: HTTP {e.response.status_code} from Azure: "
            f"{e.response.text[:500]}"
        )
    except httpx.HTTPError as e:
        sys.exit(f"ERROR: request failed: {e!r}")

    if args.json:
        print(json.dumps({
            "chat_models": chat_models,
            "deployment_probes": probes,
        }, indent=2))
        return

    print(f"Azure endpoint: {args.endpoint}")
    print(f"API version:    {args.api_version}")
    print()
    print(f"Chat-capable models on this resource ({len(chat_models)}):")
    print(_format_models(chat_models))
    print()
    if probes:
        confirmed = [p for p in probes if p["deployed"]]
        print(f"Deployment probes ({len(confirmed)}/{len(probes)} deployed):")
        print(_format_probes(probes))


if __name__ == "__main__":
    main()
