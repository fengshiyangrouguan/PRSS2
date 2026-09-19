"""Synchronous llm() helper for use inside solve() exec namespaces.

Provides two factory functions:
- make_llm_func: for main-process usage (text classification)
- make_llm_func_from_config: for subprocess usage (CO-Bench)

Both return a sync ``llm(prompt, *, temperature, max_tokens) -> str`` closure
that internally submits to a per-factory background event loop to call the
async LLMClient.

When a ``log_path`` is supplied, every call appends one JSONL record
(messages + response + usage) so post-hoc analysis can audit exactly what
the evolved solve() sent to the model.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from meta_n.utils.llm_io_logger import LLMIOLogger


@dataclass
class LLMUsageTracker:
    """Thread-safe token counter for inner-LLM calls within a single solve() run.

    Records the OpenAI-style breakdown — prompt (input) tokens, completion
    (output) tokens, and the total — plus a call count. The breakdown is
    surfaced in summary.json so cost analysis can separate "LLM input" from
    "LLM output" for both the outer driver and the evaluated solve()'s own
    LLM calls.
    """

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, tokens: int, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Accumulate one call's usage. ``tokens`` is total (legacy positional
        compat); prompt/completion default to 0 if unknown."""
        with self._lock:
            self.total_tokens += int(tokens or 0)
            self.prompt_tokens += int(prompt_tokens or 0)
            self.completion_tokens += int(completion_tokens or 0)
            self.call_count += 1


class _LoopRunner:
    """One background event-loop thread per factory; all inner-LLM I/O for
    the factory's closures runs on it. F073: httpx pooled connections are
    loop-affine — the previous per-call ``asyncio.run`` closed the loop each
    call, leaving keep-alive connections bound to a dead loop; the next call
    reusing one faulted (RuntimeError: Event loop is closed), which the
    retry loop masked at ~2s + one wasted attempt per call. A single
    persistent loop keeps the pool valid (and keep-alive actually working).
    The thread is a daemon started lazily on first call; it dies with the
    process (subprocess workers exit after their chunk — same lifetime the
    un-aclosed client already had).

    Accepted edges: a KeyboardInterrupt in the caller leaves the in-flight
    coroutine running on the daemon loop (per-call ``asyncio.run`` would have
    cancelled it) — irrelevant for subprocess workers, which exit anyway; one
    daemon thread exists per factory instance, lazily started, bounded by the
    adapter/subprocess count (the same order as the client count).
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._thread is None or not self._thread.is_alive():
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever, daemon=True, name="metan-inner-llm-loop",
                )
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    def run(self, coro):
        # concurrent.futures.Future.result() re-raises BaseException
        # subclasses too, so BudgetExceededError still propagates.
        return asyncio.run_coroutine_threadsafe(coro, self._ensure_loop()).result()


def _make_llm_funcs(
    client_provider,
    model_provider,
    tracker: LLMUsageTracker | None,
    io_logger,
):
    """Shared closure factory behind ``make_llm_func`` / ``make_llm_func_from_config``.

    Contract: the closures' keyword-only signatures, docstrings, and the JSONL
    record fields (messages/response/model/prompt_tokens/completion_tokens/
    total_tokens) are the runtime API advertised to evolved solve() code
    (LANG_INSTRUCTIONS_PYTHON) and must stay identical across both factories.
    Every record is additionally stamped with ``extra.pid`` (the writing
    process's pid, resolved at call time so it is correct inside mp.Process
    workers) — ``_usage_from_log_since(pid=...)`` uses it to exclude sibling
    subprocesses' records when several workers share one log file.

    Args:
        client_provider: Zero-arg callable returning the LLMClient (eager
            capture or lazy subprocess construction — caller's choice).
        model_provider: Zero-arg callable returning the model name for logs.
        tracker: Optional usage tracker to accumulate token counts.
        io_logger: Optional already-constructed LLMIOLogger (or None).
    """
    runner = _LoopRunner()

    def _record_and_log(prompt, text, pt, ct, tt):
        if tracker is not None:
            tracker.record(tt, prompt_tokens=pt, completion_tokens=ct)
        if io_logger is not None:
            io_logger.log(
                messages=[{"role": "user", "content": prompt}],
                response=text, model=model_provider(),
                prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
                extra={"pid": os.getpid()},
            )

    def llm(prompt: str, *, temperature: float = 0.3, max_tokens: int = 512) -> str:
        """Call the LLM and return the response text.

        Args:
            prompt: The user prompt string.
            temperature: Sampling temperature (default 0.3).
            max_tokens: Maximum response tokens (default 512).

        Returns:
            The LLM response as a string.
        """
        client = client_provider()
        messages = [{"role": "user", "content": prompt}]
        # ``_suppress_io_log=True`` prevents the outer LLMClient.io_logger
        # (if one is attached for orchestrator-level outer-LLM calls) from
        # also logging this inner call. Without it, when an in-process
        # adapter shares the same LLMClient, every inner llm() would land
        # in BOTH outer.jsonl and inner.jsonl. The inner path below logs
        # via ``io_logger`` directly, so the outer logger's record would
        # be a duplicate. (Subprocess path is unaffected — the subprocess
        # builds a fresh LLMClient with no io_logger.)
        text, pt, ct, tt = runner.run(
            client.complete_with_breakdown(
                messages, temperature=temperature, max_tokens=max_tokens,
                _suppress_io_log=True,
            )
        )
        _record_and_log(prompt, text, pt, ct, tt)
        return text

    def llm_batch(
        prompts: list[str], *, temperature: float = 0.3, max_tokens: int = 512,
        max_concurrent: int = 10,
    ) -> list[str]:
        """Call the LLM concurrently for multiple prompts.

        Args:
            prompts: List of prompt strings.
            temperature: Sampling temperature (default 0.3).
            max_tokens: Maximum response tokens per call (default 512).
            max_concurrent: Maximum concurrent requests (default 10).

        Returns:
            List of response strings, in the same order as prompts.
        """
        if not prompts:
            return []

        client = client_provider()
        sem = asyncio.Semaphore(max_concurrent)

        async def _one(prompt: str) -> tuple[str, str, int, int, int]:
            async with sem:
                text, pt, ct, tt = await client.complete_with_breakdown(
                    [{"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens,
                    _suppress_io_log=True,
                )
                # Bind the prompt back into the result so we can log the
                # (request, response) pair in order — gather() preserves
                # input order but each task's prompt is otherwise lost.
                return prompt, text, pt, ct, tt

        async def _batch():
            # On any sibling failure: cancel the still-pending tasks (so no
            # in-flight request keeps spending on the daemon loop after the
            # caller sees the exception), await the settle, and hand the
            # settled entries back so completed siblings' usage is still
            # recorded below. Success path returns the ordered results as
            # before.
            tasks = [asyncio.ensure_future(_one(p)) for p in prompts]
            try:
                return await asyncio.gather(*tasks), None
            except BaseException as exc:
                for t in tasks:
                    t.cancel()
                settled = await asyncio.gather(*tasks, return_exceptions=True)
                return settled, exc

        results, failure = runner.run(_batch())
        texts = []
        for entry in results:
            if isinstance(entry, BaseException):
                continue
            prompt, text, pt, ct, tt = entry
            _record_and_log(prompt, text, pt, ct, tt)
            texts.append(text)
        if failure is not None:
            raise failure
        return texts

    return llm, llm_batch


def make_llm_func(
    llm_client,
    tracker: LLMUsageTracker | None = None,
    log_path: str | Path | None = None,
):
    """Create sync llm() and llm_batch() closures for the main-process exec namespace.

    Caller must not invoke the returned closures from a thread with a running
    asyncio event loop (e.g., run _run_solve_code via asyncio.to_thread): the
    closures block on a cross-thread ``Future.result()``, which would stall
    that loop — the same operational constraint the previous per-call
    ``asyncio.run`` imposed, with a different failure shape.

    Args:
        llm_client: An LLMClient instance.
        tracker: Optional usage tracker to accumulate token counts.
        log_path: Optional path to a JSONL file. When set, each llm()/
            llm_batch() call appends one record (messages, response, usage,
            timestamp). The file is opened append-only with cross-process
            flock locking, so multiple subprocess workers may share it.

    Returns:
        Tuple of (llm, llm_batch) functions.
    """
    io_logger = LLMIOLogger(log_path, source="inner_llm") if log_path else None
    # Model name resolved eagerly at factory time (contract: matches the
    # captured client, even if client.config mutates later).
    model = getattr(getattr(llm_client, "config", None), "model", "")
    return _make_llm_funcs(lambda: llm_client, lambda: model, tracker, io_logger)


def make_llm_func_from_config(
    config_dict: dict,
    tracker: LLMUsageTracker | None = None,
    log_path: str | Path | None = None,
):
    """Create sync llm() and llm_batch() closures for subprocess exec namespaces.

    The LLMClient is created lazily on first call because subprocesses
    cannot receive the parent's LLMClient (AsyncOpenAI is not picklable).

    Args:
        config_dict: A plain dict from ``dataclasses.asdict(LLMConfig)``.
        tracker: Optional usage tracker to accumulate token counts.
        log_path: Optional path to a JSONL file. Same semantics as
            ``make_llm_func`` — the path is a plain string so it survives
            pickling into the subprocess; flock makes concurrent writers
            safe across mp.Process boundaries.

    Returns:
        Tuple of (llm, llm_batch) functions.
    """
    _client_holder: dict = {}
    io_logger = LLMIOLogger(log_path, source="inner_llm") if log_path else None

    def _get_client():
        # Lazy import stays INSIDE the provider: the subprocess must not pay
        # (or fail) the LLMClient import until the first actual call.
        if "client" not in _client_holder:
            from meta_n.core.llm_client import LLMClient, LLMConfig

            config = LLMConfig(**config_dict)
            _client_holder["client"] = LLMClient(config)
        return _client_holder["client"]

    def _model_name() -> str:
        return config_dict.get("model", "") or ""

    return _make_llm_funcs(_get_client, _model_name, tracker, io_logger)
