"""Capture the TRUE wire bytes at the HTTP boundary.

Why this exists: the harness only ever looked at ``message.content``. When a
response came back with ``finish_reason='tool_calls'`` and a null content, that
collapsed into "empty response" -- the tool-call payload, and with it any chance
of telling what the provider actually did, was discarded. Reconstructing the
request from SDK kwargs is not proof either: serialisation, header handling and
the httpx layer all sit between the call site and the socket.

So this wraps the transport itself. It records, per physical request:

  * the EXACT serialised request body as sent, plus its sha256
  * URL, method and headers (credential headers redacted)
  * the raw response body, HTTP status, response headers and sha256
  * the dependency versions that produced all of the above

Installed on an ``AsyncOpenAI`` via :func:`install`, or automatically by
``LLMClient`` when ``META_N_WIRE_CAPTURE`` names an output path.

Reads the response body inside the transport and hands the same response back.
That is deliberate -- httpx caches the bytes in ``_content`` on ``aread()``, so
the SDK still sees a complete body -- but it is also the one thing here that
could, in principle, interfere with the caller, so it is off by default and the
SDK path is exercised in tests.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import threading
import time
from pathlib import Path

#: Header names whose VALUES never get written. Names are still recorded so the
#: diff between two requests stays readable.
_REDACTED = frozenset({
    "authorization", "api-key", "x-api-key", "cookie", "set-cookie",
    "proxy-authorization", "openai-api-key",
})

ENV_PATH = "META_N_WIRE_CAPTURE"


def _redact(headers) -> dict:
    out = {}
    for k, v in dict(headers).items():
        out[k] = "<redacted>" if k.lower() in _REDACTED else v
    return out


def environment_fingerprint() -> dict:
    """Versions that decide serialisation and response parsing.

    Pinned here rather than assumed: the same kwargs on two machines are not the
    same request if the SDK or httpx differs.
    """
    fp = {}
    for pkg in ("openai", "httpx", "httpx2", "pydantic", "httpcore"):
        try:
            fp[pkg] = md.version(pkg)
        except Exception:                                     # noqa: BLE001
            fp[pkg] = None
    return fp


class WireCapture:
    """A transport wrapper. NOT an httpx2 subclass.

    Duck-typed instead of subclassing ``httpx2.AsyncBaseTransport`` on purpose:
    the class is resolved at install time from whatever module the client was
    actually built from (openai 3.14.1 ships against httpx2 here, but that is a
    dependency detail, not a contract). Only ``handle_async_request`` is used.
    """

    def __init__(self, inner, out_path):
        self._inner = inner
        self._path = Path(out_path)
        self._lock = threading.Lock()
        self.records: list[dict] = []

    # -- transport lifecycle -------------------------------------------------
    # httpx calls these on the transport when the client is closed. Omitting
    # them made every shutdown raise AttributeError('WireCapture' object has no
    # attribute 'aclose') -- noise in the log, and a real defect in a wrapper
    # that is supposed to be transparent. Delegate everything.
    async def aclose(self):
        return await self._inner.aclose()

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)

    def close(self):
        return self._inner.close() if hasattr(self._inner, "close") else None

    async def handle_async_request(self, request):
        try:
            req_body = request.content or b""
        except Exception:                                     # noqa: BLE001
            req_body = b""

        rec = {
            "ts": time.time(),
            "method": getattr(request, "method", None),
            "url": str(getattr(request, "url", "")),
            "request_headers": _redact(getattr(request, "headers", {})),
            "request_bytes": len(req_body),
            "request_sha256": hashlib.sha256(req_body).hexdigest(),
            "request_body": req_body.decode("utf-8", "replace"),
            "environment": environment_fingerprint(),
        }

        resp = await self._inner.handle_async_request(request)

        try:
            # Caches into httpx's _content; the SDK reads a complete body after.
            raw = await resp.aread()
        except Exception as e:                                # noqa: BLE001
            raw = b""
            rec["response_read_error"] = repr(e)

        rec.update({
            "status": getattr(resp, "status_code", None),
            "response_headers": _redact(getattr(resp, "headers", {})),
            "response_bytes": len(raw),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "response_body": raw.decode("utf-8", "replace"),
        })

        with self._lock:
            self.records.append(rec)
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:                                 # noqa: BLE001
                pass          # capture must never break the call it observes
        return resp


def install(openai_client, out_path: str | os.PathLike | None) -> "WireCapture | None":
    """Wrap an AsyncOpenAI's transport. Returns the capture, or None if disabled."""
    if not out_path:
        return None
    httpx_client = openai_client._client
    inner = getattr(httpx_client, "_transport", None)
    if inner is None:
        return None
    cap = WireCapture(inner, out_path)
    httpx_client._transport = cap
    return cap


def install_from_env(openai_client) -> "WireCapture | None":
    return install(openai_client, os.environ.get(ENV_PATH, "").strip() or None)
