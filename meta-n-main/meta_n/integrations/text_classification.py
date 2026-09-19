"""Text classification benchmark adapter.

Supports two datasets:
  - Symptom2Disease: 22-class medical diagnosis (English)
  - LawBench charge prediction (task 3-3): multi-label criminal charges (Chinese)

Usage:
    adapter = TextClassificationAdapter("symptom2disease")
    tasks   = adapter.load_tasks()
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import random
import urllib.request
from pathlib import Path
from typing import Any

from meta_n.core.meta_layer import TaskDescription

# Compat re-exports: tests and baselines import/monkeypatch these via this
# module's namespace; keep them bound here.
from meta_n.integrations._subprocess_utils import (  # noqa: F401
    _add_usage,
    _coerce_usage,
    _empty_usage,
    _kill_process_tree,
    _snapshot_log_offset,
    _tracker_usage,
    _usage_from_log_since,
    detach_process_group,
    run_process_with_timeout,
)
from meta_n.integrations.benchmark import AdapterExecutor, BenchmarkAdapter, EvalResult
from meta_n.utils.atomic_io import atomic_json_dump

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

LAWBENCH_URL = (
    "https://raw.githubusercontent.com/open-compass/LawBench/main/data/one_shot/3-3.json"
)


def _load_symptom2disease(
    data_dir: Path,
    max_val: int = 50,
    max_test: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Load Symptom2Disease from HuggingFace ``gretelai/symptom_to_diagnosis``.

    Returns dict with keys: train, val, test, labels, metric, language.
    """
    # Cache keyed by (seed, max_val, max_test) so parameter changes invalidate it.
    # Earlier versions omitted max_test from the key, which silently returned a
    # mismatched test slice when callers varied max_test across runs.
    test_key = "all" if max_test is None else str(max_test)
    cache_path = (
        data_dir
        / "symptom2disease"
        / f"cached_s{seed}_v{max_val}_t{test_key}.json"
    )
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    from datasets import load_dataset

    ds = load_dataset("gretelai/symptom_to_diagnosis")

    all_train = [
        {"text": row["input_text"], "label": row["output_text"]}
        for row in ds["train"]
    ]
    all_test = [
        {"text": row["input_text"], "label": row["output_text"]}
        for row in ds["test"]
    ]

    # Shuffle before splitting to ensure val covers all labels.
    # Without this, the first max_val examples may only cover a
    # fraction of the 22 diseases (observed: 12/22 with max_val=20).
    rng = random.Random(seed)
    rng.shuffle(all_train)

    val = all_train[:max_val]
    train = all_train[max_val:]
    # None-sentinel (not truthiness): max_test=0 must yield an EMPTY test
    # split, matching the "_t0" cache key above. Default (None) is unchanged.
    test = all_test[:max_test] if max_test is not None else all_test

    labels = sorted({row["label"] for row in all_train + all_test})

    data = {
        "train": train,
        "val": val,
        "test": test,
        "labels": labels,
        "metric": "accuracy",
        "language": "en",
    }

    # Cache locally — atomic tmp-then-rename write, so an interrupted write
    # never leaves a half-serialized cache that the exists()/json.load
    # read-back path (above) would choke on. indent=None keeps the compact
    # single-line bytes the read-back path has always cached.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(cache_path, data, indent=None)
    logger.info(
        "Symptom2Disease: loaded %d train, %d val, %d test, %d labels",
        len(train), len(val), len(test), len(labels),
    )
    return data


def _load_lawbench_charge(
    data_dir: Path,
    max_val: int = 50,
    max_test: int | None = None,
) -> dict[str, Any]:
    """Load LawBench task 3-3 (charge prediction) from GitHub.

    Returns dict with keys: train, val, test, labels, metric, language.
    """
    cache_dir = data_dir / "lawbench_charge"
    raw_path = cache_dir / "3-3.json"

    # Download if missing. Fetch into a sibling temp file and atomically
    # promote it into place only after the bytes fully land AND parse, so a
    # dropped/partial transfer (or a kill mid-write) can never leave a corrupt
    # 3-3.json that the exists() guard treats as cached and json.load then
    # chokes on forever. On any failure the partial temp file is removed.
    if not raw_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading LawBench 3-3 from %s …", LAWBENCH_URL)
        tmp_path = raw_path.with_name(raw_path.name + ".tmp")
        try:
            urllib.request.urlretrieve(LAWBENCH_URL, tmp_path)  # noqa: S310
            with open(tmp_path) as f:
                json.load(f)  # validate the download before promoting it
            os.replace(tmp_path, raw_path)
        except BaseException:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

    with open(raw_path) as f:
        raw = json.load(f)

    # Parse entries — answer format is "罪名:charge1;charge2" or "罪名:charge"
    all_data: list[dict[str, str]] = []
    for item in raw:
        answer_raw = item.get("answer", "")
        # Strip "罪名:" prefix if present
        if answer_raw.startswith("罪名:"):
            answer_raw = answer_raw[len("罪名:"):]
        all_data.append({
            "text": item["question"],
            "label": answer_raw.strip(),
            "instruction": item.get("instruction", ""),
        })

    # Split: first 200 train, next max_val val, rest test
    train = all_data[:200]
    val = all_data[200 : 200 + max_val]
    test_start = 200 + max_val
    # None-sentinel (not truthiness): max_test=0 means an empty test split.
    test = (
        all_data[test_start : test_start + max_test]
        if max_test is not None
        else all_data[test_start:]
    )

    # Collect all unique charges
    all_labels: set[str] = set()
    for item in all_data:
        for charge in item["label"].split(";"):
            charge = charge.strip()
            if charge:
                all_labels.add(charge)

    data = {
        "train": train,
        "val": val,
        "test": test,
        "labels": sorted(all_labels),
        "metric": "f1",
        "language": "zh",
    }
    logger.info(
        "LawBench charge: loaded %d train, %d val, %d test, %d labels",
        len(train), len(val), len(test), len(all_labels),
    )
    return data


# ---------------------------------------------------------------------------
# Subprocess helpers — isolate solver exec() from the orchestrator process
# ---------------------------------------------------------------------------
#
# The design rule is "Ω-generated code executes only inside Docker containers,
# never on host." Pure-Docker isn't viable for text classification (the inner
# llm() helper needs the parent's API key plumbing), but we can at least
# isolate solver code in a separate Python process with a hard timeout — the
# same pattern co_bench.py uses. See co_bench._run_with_timeout.


def _coerce_predictions(predictions: Any, balanced_json_fallback: bool) -> Any:
    """F075 live-site recovery: when the flag is ON and solve() returned a
    string, run the classify extraction chain (fenced -> flat regex ->
    balanced scan, see ``extract_case_json``) on it and return the parsed
    dict when one is recovered. Flag OFF, dict returns, and unrecoverable
    strings pass through unchanged, so the downstream ``expected dict``
    error path stays byte-identical to the flag-less behaviour.
    """
    if not balanced_json_fallback or not isinstance(predictions, str):
        return predictions
    from meta_n.core.solver import extract_case_json

    recovered = extract_case_json(predictions, balanced_fallback=True)
    if recovered is None:
        return predictions
    try:
        obj = json.loads(recovered)
    except json.JSONDecodeError:
        return predictions
    return obj if isinstance(obj, dict) else predictions


def _run_solve_in_process(
    solution: str,
    cases: dict[str, str],
    labels: list[str],
    few_shot: list[dict],
    llm_config_dict: dict | None,
    queue: mp.Queue,
    inner_log_path: str | None = None,
    balanced_json_fallback: bool = False,
) -> None:
    """Subprocess target: exec solve(), call it, put result on the queue.

    Result is a 3-tuple ``(status, payload, usage_dict)`` where usage_dict
    has ``{"total": int, "prompt": int, "completion": int, "calls": int}``.
    The dict shape (rather than a wider tuple) keeps the queue protocol
    extensible without churning every consumer when we add fields.
      ("ok", predictions_dict, usage_dict)
      ("error", error_str,    usage_dict)

    When ``inner_log_path`` is provided, every llm()/llm_batch() call inside
    solve() appends one JSONL record with its messages + response. The
    file is opened with flock locking so concurrent chunk subprocesses
    can share a single log file without corruption.
    """
    detach_process_group()
    try:
        from meta_n.core.llm_helpers import (
            LLMUsageTracker,
            make_llm_func_from_config,
        )

        tracker = LLMUsageTracker()
        ns: dict = {}
        if llm_config_dict is not None:
            ns["llm"], ns["llm_batch"] = make_llm_func_from_config(
                llm_config_dict, tracker, log_path=inner_log_path,
            )
        exec(solution, ns)  # noqa: S102 — inside isolated subprocess
        if "solve" not in ns:
            queue.put(("error", "No solve() function defined", _tracker_usage(tracker)))
            return
        predictions = _coerce_predictions(
            ns["solve"](cases, labels, few_shot), balanced_json_fallback,
        )
        if not isinstance(predictions, dict):
            queue.put((
                "error",
                f"solve() returned {type(predictions).__name__}, expected dict",
                _tracker_usage(tracker),
            ))
            return
        queue.put(("ok", predictions, _tracker_usage(tracker)))
    except Exception as e:
        # Preserve any inner-LLM usage solve() already incurred before it
        # crashed: ``tracker`` (created above) holds the real prompt/completion
        # counts for calls that did happen, and those calls cost real
        # tokens/money. Only fall back to _empty_usage() when the failure
        # preceded tracker creation (e.g. the import block above raised).
        try:
            usage = _tracker_usage(tracker)
        except NameError:
            usage = _empty_usage()
        queue.put(("error", f"{type(e).__name__}: {e}", usage))


def _run_solve_with_timeout(
    solution: str,
    cases: dict[str, str],
    labels: list[str],
    few_shot: list[dict],
    llm_config_dict: dict | None,
    timeout: int,
    inner_log_path: str | None = None,
    balanced_json_fallback: bool = False,
) -> tuple[str, Any, dict[str, int]]:
    """Run solve() in a subprocess with hard timeout.

    Returns (status, payload, usage):
      status="ok"    → payload is the predictions dict
      status="error" → payload is a human-readable error string
      usage          → {"total", "prompt", "completion", "calls"}
                       On a clean finish these come from the subprocess's
                       in-process tracker via the queue. On timeout/kill the
                       child's tracker dies with it, so partial usage is
                       reconstructed from the inner-log JSONL (the records this
                       subprocess appended before being killed). When no
                       inner-log is configured there is nothing to recover and
                       timeout usage is reported as zero.

    ``inner_log_path``: forwarded to the subprocess so each inner llm()
    call appends a JSONL record. None disables logging.
    """
    # Snapshot the inner-log size before the child starts so a timeout/kill
    # can reconstruct just this subprocess's partial inner-LLM usage from the
    # records it appends, without counting pre-existing history (see
    # _usage_from_log_since; pid attribution excludes concurrent siblings).
    log_start_offset = _snapshot_log_offset(inner_log_path)
    queue: mp.Queue = mp.Queue()
    return run_process_with_timeout(
        _run_solve_in_process,
        (solution, cases, labels, few_shot, llm_config_dict, queue,
         inner_log_path, balanced_json_fallback),
        timeout,
        queue=queue,
        # The timeout message deliberately carries the elapsed clock (unlike
        # the sibling integrations) — keep the wording byte-identical.
        on_timeout=lambda elapsed, pid: (
            "error",
            f"Timeout (configured={timeout}s, elapsed={elapsed:.1f}s)",
            _usage_from_log_since(inner_log_path, log_start_offset, pid=pid),
        ),
        # Distinguish "queue genuinely empty" from "queue corrupted /
        # subprocess died before put". Helps debugging.
        on_no_result=lambda e, pid: (
            "error",
            f"No result from subprocess ({type(e).__name__}: {e})",
            _usage_from_log_since(inner_log_path, log_start_offset, pid=pid),
        ),
    )


# ---------------------------------------------------------------------------
# Parallel chunk worker
# ---------------------------------------------------------------------------


class _ChunkError(RuntimeError):
    """Raised by ``_run_chunk`` on a failed chunk, carrying any partial
    inner-LLM ``usage`` reconstructed by ``_run_solve_with_timeout``.

    Subclasses ``RuntimeError`` so existing callers that only care about the
    error still behave identically; the extra ``usage`` attribute lets the
    parallel caller fold a timed-out chunk's partial inner-LLM tokens into the
    aggregate (mirroring the sequential path, which keeps ``usage`` on
    ``status != "ok"``). Without this, ``eval_workers>1`` timeouts would
    under-count inner_tokens/inner_calls — the #71 accounting loss relocated to
    the parallel caller.
    """

    def __init__(self, message: str, usage: dict[str, int] | None = None):
        super().__init__(message)
        self.usage = _coerce_usage(usage)


def _run_chunk(
    solution: str,
    chunk_cases: dict[str, str],
    labels: list[str],
    few_shot: list[dict],
    llm_config_dict: dict | None,
    timeout: int = 300,
    inner_log_path: str | None = None,
    balanced_json_fallback: bool = False,
) -> tuple[dict, dict[str, int]]:
    """Run solve() on a subset of cases in an isolated subprocess.

    Returns (predictions, usage). Raises ``_ChunkError`` (a RuntimeError
    subclass carrying the reconstructed partial ``usage``) on failure so the
    caller can return a structured EvalResult with the error feedback while
    still accounting for inner-LLM tokens spent before the failure.
    """
    status, payload, usage = _run_solve_with_timeout(
        solution, chunk_cases, labels, few_shot, llm_config_dict, timeout,
        inner_log_path=inner_log_path,
        balanced_json_fallback=balanced_json_fallback,
    )
    if status == "ok":
        return payload, usage
    raise _ChunkError(f"chunk subprocess failed: {payload}", usage)


def _run_chunk_inprocess(
    solution: str,
    chunk_cases: dict[str, str],
    labels: list[str],
    few_shot: list[dict],
    llm_config_dict: dict | None,
    inner_log_path: str | None = None,
    balanced_json_fallback: bool = False,
) -> tuple[dict, dict[str, int]]:
    """In-process variant of _run_chunk used when ``subprocess_isolate=False``.

    Production runs default to the subprocess-isolated ``_run_chunk`` for
    safety. This in-process variant exists for tests that mock ``LLMClient``
    via ``MagicMock`` (mocks aren't picklable across ``mp.Process``).
    Returns (predictions, usage) where usage has total/prompt/completion/calls.
    """
    from meta_n.core.llm_helpers import LLMUsageTracker, make_llm_func_from_config

    tracker = LLMUsageTracker()
    ns: dict = {}
    if llm_config_dict is not None:
        ns["llm"], ns["llm_batch"] = make_llm_func_from_config(
            llm_config_dict, tracker, log_path=inner_log_path,
        )
    exec(solution, ns)  # noqa: S102
    if "solve" not in ns:
        raise RuntimeError("No solve() function defined")
    predictions = _coerce_predictions(
        ns["solve"](chunk_cases, labels, few_shot), balanced_json_fallback,
    )
    if not isinstance(predictions, dict):
        raise TypeError(
            f"solve() returned {type(predictions).__name__}, expected dict"
        )
    return predictions, _tracker_usage(tracker)


def _fold_inner_usage(result: EvalResult, u: dict[str, int]) -> EvalResult:
    """Fold a canonical usage dict into ``result`` (mutating it).

    Sets the four ``inner_*`` fields and, when any inner-LLM call happened,
    appends the prompt-visible ``[inner-llm]`` feedback suffix. The suffix
    wording is a byte-stable contract shared by the sequential, in-process,
    and parallel eval paths.
    """
    result.inner_tokens = u["total"]
    result.inner_prompt_tokens = u["prompt"]
    result.inner_completion_tokens = u["completion"]
    result.inner_calls = u["calls"]
    if u["calls"] > 0:
        result.feedback += (
            f"\n[inner-llm] {u['calls']} calls, "
            f"{u['total']} tokens (in={u['prompt']}, out={u['completion']})"
        )
    return result


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class TextClassificationAdapter(BenchmarkAdapter):
    """Adapter for text classification benchmarks (Symptom2Disease, LawBench)."""

    DATASETS = {"symptom2disease", "lawbench_charge"}

    # Class-level default (not just an ``__init__`` assignment) so instances
    # built via ``TextClassificationAdapter.__new__`` — a supported
    # construction path in tests — still resolve the F075 flag to OFF.
    _balanced_json_fallback: bool = False

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "./data/text_classification",
        n_few_shot: int = 5,
        max_val: int = 50,
        max_test: int | None = None,
        llm_client=None,
        eval_workers: int = 1,
        solve_timeout: int | None = None,
        subprocess_isolate: bool = True,
        inner_log_path: str | Path | None = None,
        balanced_json_fallback: bool = False,
    ):
        if dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Choose from {self.DATASETS}"
            )
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.n_few_shot = n_few_shot
        self.max_val = max_val
        self.max_test = max_test
        self._data: dict[str, Any] | None = None
        self._llm_client = llm_client
        self._eval_workers = eval_workers
        # Per-chunk hard timeout for solver execution (production: subprocess).
        self._solve_timeout = (
            solve_timeout
            if solve_timeout is not None
            else int(os.environ.get("META_N_CLASSIFY_TIMEOUT", "300"))
        )
        # Production runs in subprocess for safety. Tests that pass MagicMock
        # LLM clients must opt out (mocks aren't picklable across mp.Process).
        self._subprocess_isolate = subprocess_isolate
        # Path stored as a plain string (or None). Plain string survives
        # pickling into mp.Process without dragging the Path object's
        # internals; subprocess workers reconstitute their own LLMIOLogger.
        self._inner_log_path = str(inner_log_path) if inner_log_path else None
        # F075 (--classify-balanced-json-fallback): when ON, a solve() that
        # returns a string instead of a dict gets one recovery pass through
        # the classify extraction chain (see _coerce_predictions). Default
        # OFF keeps every parse path byte-identical.
        self._balanced_json_fallback = balanced_json_fallback

    # --- BenchmarkAdapter interface ---

    @property
    def name(self) -> str:
        return f"classify_{self.dataset_name}"

    def split_type(self) -> str:
        # Genuine held-out test split (train dev vs held-out test labels).
        return "held_out"

    def load_tasks(self, limit: int | None = None) -> list[TaskDescription]:
        data = self._get_data()

        description = self._build_description(data)

        task = TaskDescription(
            task_id=self.dataset_name,
            description=description,
            # INVARIANT (selection integrity, F127): val-split payloads —
            # per-case texts or gold labels — must NEVER enter this metadata
            # dict. Ω-injected pre_process has runtime read access to
            # task.metadata and its additional_context output is appended to
            # the solver prompt, so any val payload here would let injected
            # code leak gold DEV labels into the solver while dev scores drive
            # archive selection. Evaluation is independent: _run_solve_code
            # re-derives cases/labels/few_shot from _get_data(). few-shot
            # TRAIN examples are deliberately solver-visible (also in the
            # description); label_set is vocabulary, not a per-case mapping.
            metadata={
                "benchmark": "text_classification",
                "dataset_name": self.dataset_name,
                "solution_language": "python",
                "label_set": data["labels"],
                "metric": data["metric"],
                "language": data["language"],
                "few_shot_examples": data["train"][: self.n_few_shot],
            },
        )
        tasks = [task]
        return tasks[:limit] if limit is not None else tasks

    def _run_solve_code(
        self, solution: str, examples: list[dict], metric: str
    ) -> EvalResult:
        """Execute a solve() function and score its predictions."""
        if self._eval_workers > 1 and len(examples) > self._eval_workers:
            return self._run_solve_code_parallel(solution, examples, metric)

        cases = {f"case_{i}": ex["text"] for i, ex in enumerate(examples)}
        data = self._get_data()
        labels = data["labels"]
        few_shot = data["train"][: self.n_few_shot]

        if not self._subprocess_isolate:
            return self._run_solve_code_inprocess(
                solution, cases, labels, few_shot, examples, metric,
            )

        from dataclasses import asdict

        llm_config_dict = (
            asdict(self._llm_client.config)
            if self._llm_client is not None
            else None
        )
        status, payload, usage = _run_solve_with_timeout(
            solution, cases, labels, few_shot,
            llm_config_dict, self._solve_timeout,
            inner_log_path=self._inner_log_path,
            balanced_json_fallback=self._balanced_json_fallback,
        )
        u = _coerce_usage(usage)
        if status != "ok":
            return EvalResult(
                success=False, score=0.0,
                feedback=f"solve() error: {payload}",
                inner_tokens=u["total"],
                inner_prompt_tokens=u["prompt"],
                inner_completion_tokens=u["completion"],
                inner_calls=u["calls"],
            )
        predictions_json = json.dumps(payload)
        result = self._score_predictions(predictions_json, examples, metric)
        return _fold_inner_usage(result, u)

    def _run_solve_code_inprocess(
        self,
        solution: str,
        cases: dict[str, str],
        labels: list[str],
        few_shot: list[dict],
        examples: list[dict],
        metric: str,
    ) -> EvalResult:
        """In-process solver execution. Test-only path: bypassed in production
        by the subprocess wrapper. Required because MagicMock LLM clients
        aren't picklable across mp.Process boundaries.
        """
        from meta_n.core.llm_helpers import LLMUsageTracker, make_llm_func

        tracker = LLMUsageTracker()

        def _result_from_tracker_error(feedback: str) -> EvalResult:
            return EvalResult(
                success=False, score=0.0, feedback=feedback,
                inner_tokens=tracker.total_tokens,
                inner_prompt_tokens=tracker.prompt_tokens,
                inner_completion_tokens=tracker.completion_tokens,
                inner_calls=tracker.call_count,
            )

        try:
            ns: dict = {}
            if self._llm_client is not None:
                ns["llm"], ns["llm_batch"] = make_llm_func(
                    self._llm_client, tracker,
                    log_path=self._inner_log_path,
                )
            exec(solution, ns)  # noqa: S102
            if "solve" not in ns:
                return _result_from_tracker_error("No solve() function defined")
            predictions = _coerce_predictions(
                ns["solve"](cases, labels, few_shot),
                self._balanced_json_fallback,
            )
            if not isinstance(predictions, dict):
                return _result_from_tracker_error(
                    f"solve() returned {type(predictions).__name__}, expected dict"
                )
            predictions_json = json.dumps(predictions)
        except Exception as e:
            return _result_from_tracker_error(
                f"solve() error: {type(e).__name__}: {e}"
            )

        result = self._score_predictions(predictions_json, examples, metric)
        return _fold_inner_usage(result, _tracker_usage(tracker))

    def _run_solve_code_parallel(
        self, solution: str, examples: list[dict], metric: str
    ) -> EvalResult:
        """Run solve() on chunks of cases in parallel threads, then merge.

        With ``subprocess_isolate=True`` (default), each thread spawns its own
        subprocess via ``_run_chunk``. With ``subprocess_isolate=False``, falls
        back to in-process exec via ``_run_chunk_inprocess`` — required for
        tests that mock the LLM client.
        """
        from concurrent.futures import ThreadPoolExecutor
        from dataclasses import asdict

        data = self._get_data()
        labels = data["labels"]
        few_shot = data["train"][: self.n_few_shot]
        llm_config_dict = None
        if self._llm_client is not None and self._subprocess_isolate:
            llm_config_dict = asdict(self._llm_client.config)

        # Split cases into chunks with global indices
        all_cases = [(f"case_{i}", ex["text"]) for i, ex in enumerate(examples)]
        chunk_size = (len(all_cases) + self._eval_workers - 1) // self._eval_workers
        chunks = [
            dict(all_cases[i : i + chunk_size])
            for i in range(0, len(all_cases), chunk_size)
        ]

        # In-process fallback also needs llm_config_dict if llm client present;
        # but it accepts the live client by reconstructing per-chunk via
        # make_llm_func_from_config (same code path) so we pass the same dict.
        # When subprocess_isolate=False AND _llm_client is mocked, we route
        # through the in-process variant which still uses make_llm_func_from_config
        # — but mocks get the in-process namespace via the parent's _llm_client.
        # In that case we keep llm_config_dict=None and inject a fresh make_llm_func
        # binding via the in-process path. The chunk worker for tests does the same.
        if not self._subprocess_isolate and self._llm_client is not None:
            # Tests with mocked LLM use _run_chunk_inprocess; we can't pass the
            # mock through llm_config_dict. _run_chunk_inprocess takes the
            # config dict but tests usually leave _llm_client=None for parallel.
            # Set llm_config_dict to a marker so _run_chunk_inprocess raises
            # if a test actually tries to mix mocked LLM + parallel chunks.
            try:
                llm_config_dict = asdict(self._llm_client.config)
            except Exception:
                # Mock object — parallel + mock not supported; fall back to
                # sequential in-process path.
                return self._run_solve_code_inprocess(
                    solution,
                    {f"case_{i}": ex["text"] for i, ex in enumerate(examples)},
                    labels, few_shot, examples, metric,
                )

        chunk_fn = _run_chunk if self._subprocess_isolate else _run_chunk_inprocess

        merged: dict = {}
        total_usage = _empty_usage()
        first_error: Exception | None = None
        try:
            with ThreadPoolExecutor(max_workers=self._eval_workers) as pool:
                if self._subprocess_isolate:
                    futures = [
                        pool.submit(
                            chunk_fn, solution, chunk, labels, few_shot,
                            llm_config_dict, self._solve_timeout,
                            self._inner_log_path,
                            balanced_json_fallback=self._balanced_json_fallback,
                        )
                        for chunk in chunks
                    ]
                else:
                    futures = [
                        pool.submit(
                            chunk_fn, solution, chunk, labels, few_shot,
                            llm_config_dict, self._inner_log_path,
                            balanced_json_fallback=self._balanced_json_fallback,
                        )
                        for chunk in chunks
                    ]
                # Consume EVERY future (mirroring the sequential path, which
                # keeps `usage` on status != "ok"): each failed chunk folds its
                # reconstructed partial inner-LLM usage into the aggregate, so
                # an eval_workers>1 failure reports the tokens ALL chunks spent
                # — not just the chunks consumed before the first failure. The
                # pool exit waits for every future anyway, so this costs no
                # extra wall clock. The reported error stays the submission-
                # order-first failure (the same exception that propagated
                # before), keeping the feedback string unchanged.
                for f in futures:
                    try:
                        preds, chunk_usage = f.result()
                    except Exception as e:  # noqa: BLE001 — folded, re-reported below
                        if isinstance(e, _ChunkError):
                            total_usage = _add_usage(total_usage, e.usage)
                        if first_error is None:
                            first_error = e
                        continue
                    merged.update(preds)
                    total_usage = _add_usage(total_usage, _coerce_usage(chunk_usage))
            if first_error is not None:
                return EvalResult(
                    success=False, score=0.0,
                    feedback=(
                        f"solve() error: {type(first_error).__name__}: {first_error}"
                    ),
                    inner_tokens=total_usage["total"],
                    inner_prompt_tokens=total_usage["prompt"],
                    inner_completion_tokens=total_usage["completion"],
                    inner_calls=total_usage["calls"],
                )
            predictions_json = json.dumps(merged)
        except Exception as e:
            # Non-future failures (pool construction, json.dumps) keep the
            # same EvalResult shape as a chunk failure.
            if isinstance(e, _ChunkError):
                total_usage = _add_usage(total_usage, e.usage)
            return EvalResult(
                success=False, score=0.0,
                feedback=f"solve() error: {type(e).__name__}: {e}",
                inner_tokens=total_usage["total"],
                inner_prompt_tokens=total_usage["prompt"],
                inner_completion_tokens=total_usage["completion"],
                inner_calls=total_usage["calls"],
            )

        result = self._score_predictions(predictions_json, examples, metric)
        return _fold_inner_usage(result, total_usage)

    async def evaluate(self, task: TaskDescription, solution: str) -> EvalResult:
        """Score predictions against the val split."""
        data = self._get_data()
        return await asyncio.to_thread(
            self._run_solve_code,
            solution, data["val"], task.metadata.get("metric", "accuracy"),
        )

    async def evaluate_test(self, task: TaskDescription, solution: str) -> EvalResult:
        """Score predictions against the test split.

        The solution is reusable Python code (a solve() function) — we exec it
        with test examples to get predictions.
        """
        data = self._get_data()
        return await asyncio.to_thread(
            self._run_solve_code,
            solution, data["test"], task.metadata.get("metric", "accuracy"),
        )

    # --- Internal helpers ---

    def _get_data(self) -> dict[str, Any]:
        if self._data is None:
            if self.dataset_name == "symptom2disease":
                self._data = _load_symptom2disease(
                    self.data_dir, self.max_val, self.max_test
                )
            else:
                self._data = _load_lawbench_charge(
                    self.data_dir, self.max_val, self.max_test
                )
        return self._data

    def _build_description(self, data: dict[str, Any]) -> str:
        """Build the task description seen by the solver and Omega."""
        labels_str = ", ".join(data["labels"])
        few_shot = self._format_few_shot(data["train"][: self.n_few_shot])

        if self.dataset_name == "symptom2disease":
            desc = (
                "Medical Diagnosis Classification\n\n"
                "Given a patient's symptom description, predict the most likely "
                f"disease from the following {len(data['labels'])} diseases:\n"
                f"{labels_str}\n\n"
                f"## Few-Shot Examples\n{few_shot}\n\n"
                "## Evaluation\n"
                "Exact-match accuracy (case-insensitive, whitespace-normalized)."
            )
        else:  # lawbench_charge
            desc = (
                "刑事案件罪名预测 (Criminal Charge Prediction)\n\n"
                "根据案件事实描述，预测适用的罪名。一个案件可能涉及多个罪名，"
                "请用分号(;)分隔。\n\n"
                f"## 可选罪名列表 ({len(data['labels'])} 个)\n"
                f"{labels_str}\n\n"
                f"## 示例\n{few_shot}\n\n"
                "## 评估指标\n"
                "Per-example F1, averaged (multi-label classification)."
            )

        desc += (
            '\n\n## Implement in Solve Function\n\n'
            '```python\n'
            'def solve(cases: dict[str, str], labels: list[str], '
            'few_shot: list[dict]) -> dict[str, str]:\n'
            '    """Classify each case.\n'
            '\n'
            '    Args:\n'
            '        cases: dict mapping case_id (e.g. "case_0") to input text.\n'
            '        labels: list of valid label strings.\n'
            '        few_shot: list of dicts, each with \'text\' and \'label\' keys.\n'
            '\n'
            '    Returns:\n'
            '        dict mapping each case_id to a predicted label from the labels list.\n'
            '        For multi-label tasks, separate labels with semicolons '
            '(e.g. "Label A;Label B").\n'
            '    """\n'
            '    pass\n'
            '```'
        )
        return desc

    @staticmethod
    def _format_few_shot(examples: list[dict[str, str]]) -> str:
        lines = []
        for ex in examples:
            lines.append(f'Input: "{ex["text"]}"')
            lines.append(f'Label: {ex["label"]}')
            lines.append("")
        return "\n".join(lines)

    def _score_predictions(
        self,
        predictions_json: str,
        examples: list[dict[str, str]],
        metric: str,
    ) -> EvalResult:
        """Parse the predictions JSON string and compute accuracy or F1."""
        try:
            predictions = json.loads(predictions_json)
        except json.JSONDecodeError as e:
            return EvalResult(
                success=False,
                score=0.0,
                feedback=f"Invalid JSON: {e}",
            )

        if not isinstance(predictions, dict):
            return EvalResult(
                success=False,
                score=0.0,
                feedback=f"Expected JSON object, got {type(predictions).__name__}",
            )

        if metric in ("micro_f1", "f1"):
            return self._score_f1(predictions, examples)
        return self._score_accuracy(predictions, examples)

    def _score_accuracy(
        self, predictions: dict[str, str], examples: list[dict[str, str]]
    ) -> EvalResult:
        correct = 0
        total = len(examples)
        mismatches: list[str] = []

        for i, ex in enumerate(examples):
            key = f"case_{i}"
            pred = _normalize(str(predictions.get(key, "")))
            gold = _normalize(ex["label"])
            if pred == gold:
                correct += 1
            elif len(mismatches) < 5:
                mismatches.append(
                    f"  {key}: predicted '{pred}', expected '{gold}'"
                )

        score = correct / total if total > 0 else 0.0
        feedback = f"Accuracy: {correct}/{total} = {score:.4f}"
        if mismatches:
            feedback += "\nSample mismatches:\n" + "\n".join(mismatches)
        return EvalResult(
            success=score > 0,
            score=score,
            # TC-2: report the fraction (0..1) to match _score_f1's raw_score
            # convention; the prior count (0..N) was a latent unit mismatch.
            raw_score=score,
            feedback=feedback,
        )

    def _score_f1(
        self, predictions: dict[str, str], examples: list[dict[str, str]]
    ) -> EvalResult:
        """Per-example F1 averaged across examples.

        Matches the original LawBench evaluation (``compute_ljp_accusation``):
        compute precision/recall/F1 for each example's predicted vs. gold
        label sets, then average the per-example F1 scores.
        """
        f1_scores: list[float] = []
        abstentions = 0

        for i, ex in enumerate(examples):
            key = f"case_{i}"
            pred_labels = _normalize_multi(str(predictions.get(key, "")))
            gold_labels = _normalize_multi(ex["label"])

            if not pred_labels:
                abstentions += 1

            precision = (
                len(pred_labels & gold_labels) / len(pred_labels)
                if pred_labels
                else 0.0
            )
            recall = (
                len(pred_labels & gold_labels) / len(gold_labels)
                if gold_labels
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            f1_scores.append(f1)

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        return EvalResult(
            success=avg_f1 > 0,
            score=avg_f1,
            raw_score=avg_f1,
            feedback=(
                f"F1: {avg_f1:.4f} (avg over {len(f1_scores)} examples, "
                f"{abstentions} abstentions)"
            ),
        )


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Case-insensitive, whitespace-normalized string for comparison."""
    return " ".join(text.lower().split())


def _normalize_multi(text: str) -> set[str]:
    """Split multi-label answer by semicolons, normalize each."""
    return {
        " ".join(part.lower().split())
        for part in text.split(";")
        if part.strip()
    }


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class TextClassificationExecutor(AdapterExecutor):
    """Executor that scores classification predictions via the adapter.

    Pure ``AdapterExecutor`` defaults (``.4f`` score format,
    feedback-truncation error summary), same as ``COBenchExecutor`` — byte
    behavior pinned by tests/test_refine_benchmark.py goldens.
    """
