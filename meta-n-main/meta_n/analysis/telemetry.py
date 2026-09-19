"""Read-only analysis surface for the external-agents telemetry (plan §7.12).

This module is the *read* side of the external-agents telemetry written by
:mod:`meta_n.core.external_agents.telemetry`. It is deliberately
**pure pandas + stdlib**: it never imports the core telemetry module (or any
external-agent SDK / ``openhands`` / ``terminal_bench`` / ``docker``), so it can
be used in a bare analysis environment to load and compare runs that were
produced under any backend.

The three public helpers mirror the plan's §7.12 surface:

* :func:`load_runs` — read ``telemetry/agent_runs.jsonl`` into a ``DataFrame``
  (one row per ``execute()``), de-duped on the deterministic ``run_id`` (§7.3,
  §7.8) and optionally globbed across many run dirs (§7.12).
* :func:`pointer_resolve` — turn a column of relpath pointers
  (``transcript_ptr`` / ``agent_logs_ptr``, relpaths from the run's
  ``output_dir`` root) into absolute paths.
* :func:`fair_comparison` — the apples-to-apples table (§7.11): group by
  ``agent`` (splitting per ``benchmark`` on a cross-benchmark frame) but keep
  the *outer* and *inner* ``token_basis`` totals in **separate** columns (never
  summed under one "tokens" header), break out the ``cost_basis`` split, and
  annotate OpenHands ``agent_calls`` as ``unmeasured`` (excluded from per-call
  ratios).

On-disk envelope contract
-------------------------
Telemetry rows are appended through :class:`~meta_n.utils.llm_io_logger.LLMIOLogger`,
which wraps every record in a JSON envelope and nests the real
``AgentRunRecord`` under ``extra.record``. The loaders un-nest ``extra.record``
(with a top-level ``run_id``/field fallback for forward compatibility), exactly
as the writer's own de-dup pre-load does.
"""

from __future__ import annotations

import glob as _glob
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pandas import DataFrame, Series

logger = logging.getLogger("meta_n.analysis.telemetry")

__all__ = [
    "load_runs",
    "pointer_resolve",
    "fair_comparison",
    "unwrap_record",
]

# Telemetry filenames (must mirror ``AgentTelemetry.RUNS_FILE`` without
# importing the core module).
_RUNS_FILE = "agent_runs.jsonl"
_TELEMETRY_DIR = "telemetry"

# Pointer columns whose values are relpaths from the run's ``output_dir`` root.
_POINTER_COLUMNS = ("transcript_ptr", "agent_logs_ptr")

# The OpenHands ``agent_calls`` column is reported as ``0`` until a verified
# per-call accessor exists, so it is *unmeasured* and must never be put in the
# same "calls" column as T2's real ``budget_llm.call_count`` (§7.11).
#
# NOTE (post-fairness-fix): both OH and T2 now emit a REAL ``inner_calls`` count
# (OH via ``len(metrics.token_usages)``; T2 via the budgeted-LiteLLM call tally),
# so :func:`fair_comparison` additionally reports a measured ``inner_calls`` column
# for every agent. The legacy ``agent_calls`` column (with the OH ``"unmeasured"``
# sentinel) is retained unchanged for backward compatibility with existing callers.
_UNMEASURED = "unmeasured"
_OPENHANDS_AGENT = "openhands"
_BUILTIN_AGENT = "builtin"

# ``terminated_by`` values that mean a resource CAP bound the run (it stopped
# because it ran out of its envelope, not because it finished or genuinely
# failed): the inner token budget, the turn/episode ceiling, the wall-clock
# timeout, and either USD-budget stop (soft per-run exhaustion or the up-front
# pre-check denial). Used to derive the per-run ``cap_bound`` indicator so the
# fair-comparison table surfaces WHICH envelope actually bound each agent.
#
# DELIBERATE EXCLUSION (not an omission): ``context_len`` is NOT counted as a
# cap-bound stop even though it is also an "envelope" limit. Running out of the
# model context window is treated as a genuine FAILURE here — a well-behaved
# agent should summarize / compact before overflowing, so a context overflow is
# an agent-quality signal, unlike ``token_budget`` (an externally-imposed spend
# ceiling the agent cannot avoid). Keep this in agreement with the
# ``fair_comparison`` docstring, which lists ``context_len`` among the genuinely-
# failing states. Revisit if a future backbone makes context overflow unavoidable.
#
# SPINE-only TerminatedBy taxonomy (AgentRunRecord rows). The native
# AgenticSolver label set (spend_budget, *_unconfirmed_complete, ...) never
# reaches telemetry/agent_runs.jsonl — deliberately disjoint; do not extend
# this frozenset for native labels.
_CAP_BOUND_TERMINATED_BY = frozenset(
    {
        "token_budget",
        "max_turns",
        "timeout",
        "budget_usd",
        "budget_denied",
    }
)


# ---------------------------------------------------------------------------
# Envelope un-nesting
# ---------------------------------------------------------------------------
#
# Writer-side contract: rows are appended by LLMIOLogger with the real
# ``AgentRunRecord`` nested under ``extra.record``; the
# canonical line-scan generator on the writer side is
# ``meta_n.core.external_agents.telemetry.iter_jsonl_objects`` (which yields
# raw envelope objects and documents that envelope unwrapping is deliberately
# caller-specific). The unwrap here is STRICTER than the writer-side readers:
# a bare object without ``extra.record`` AND without a top-level ``run_id`` is
# dropped (returns ``None``), because a row that cannot be keyed would corrupt
# the de-dup / fair-comparison frames. This read-side strictness is deliberate
# and separately tested — do NOT merge these semantics with the writer-side
# ``(obj.get("extra") or {}).get("record") or obj`` fallback.


def _unwrap_record(obj: dict) -> Optional[dict]:
    """Extract the telemetry record from an :class:`LLMIOLogger` envelope.

    The writer nests the real ``AgentRunRecord`` under ``extra.record``;
    older / forward-compatible rows may carry the fields at the top level.
    Mirrors the writer's own de-dup pre-load (which reads
    ``extra.record.run_id`` with a top-level ``run_id`` fallback).

    Args:
        obj: One parsed JSONL object (the full envelope or a bare record).

    Returns:
        The record dict, or ``None`` if ``obj`` is not a usable mapping.
    """
    if not isinstance(obj, dict):
        return None
    record = (obj.get("extra") or {}).get("record")
    if isinstance(record, dict):
        return record
    # Forward/back-compat: a bare top-level record (no LLMIOLogger envelope).
    # Distinguish it from an empty envelope by the presence of a record-ish key.
    if "run_id" in obj:
        return obj
    return None


# Public alias for external tooling (scripts/build_ab_summary.py); the
# underscore name is retained for the internal callers (_read_jsonl_records).
unwrap_record = _unwrap_record


def _read_jsonl_records(path: Path) -> list[dict]:
    """Read one JSONL file into a list of un-nested telemetry records.

    Skips blank lines and lines that fail to parse (a single bad line must not
    sink the whole load). Returns ``[]`` for a missing/empty file. The line-scan
    mirrors the writer-side ``iter_jsonl_objects`` (core external_agents
    telemetry) but is kept separate: this reader additionally applies the strict
    :func:`_unwrap_record` filter and degrades OSError to an empty result, where
    the writer-side readers propagate it (see the section comment above).

    Args:
        path: Path to an ``agent_runs.jsonl`` file.

    Returns:
        A list of record dicts (envelope already un-nested).
    """
    records: list[dict] = []
    if not path.exists():
        return records
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("skipping unparseable telemetry line in %s", path)
                    continue
                record = _unwrap_record(obj)
                if record is not None:
                    records.append(record)
    except OSError as e:  # noqa: BLE001 - a read error degrades to empty, never raises
        logger.warning("could not read telemetry file %s: %r", path, e)
    return records


# ---------------------------------------------------------------------------
# Path / glob discovery
# ---------------------------------------------------------------------------


def _runs_paths(output_dir: str | Path) -> list[tuple[Path, Path]]:
    """Resolve ``(run_dir, agent_runs.jsonl)`` pairs under ``output_dir``.

    Accepts three shapes:

    * a single run dir (``<output_dir>/telemetry/agent_runs.jsonl``),
    * a parent dir holding many run dirs (``<output_dir>/*/telemetry/
      agent_runs.jsonl``) — the cross-run case (§7.12),
    * a direct path to an ``agent_runs.jsonl`` file.

    The ``run_dir`` is the run's ``output_dir`` root (the parent of
    ``telemetry/``), used both to namespace ``run_id`` on a cross-run concat and
    to resolve pointer relpaths.

    Args:
        output_dir: A run dir, a parent of run dirs, or a direct JSONL path.

    Returns:
        ``[(run_dir, jsonl_path), ...]`` — possibly empty.
    """
    return _discover_paths(output_dir, (_RUNS_FILE,))


def _pick_in_dir(telemetry_dir: Path, accepted: tuple[str, ...]) -> Optional[Path]:
    """Return the single telemetry file in ``telemetry_dir``, canonical first.

    ``accepted`` is ordered canonical-first (the canonical filename before any
    legacy aliases); the first one present wins so a dir holding both files is
    not double-loaded.
    """
    for name in accepted:
        candidate = telemetry_dir / name
        if candidate.is_file():
            return candidate
    return None


def _discover_paths(
    output_dir: str | Path, accepted: tuple[str, ...]
) -> list[tuple[Path, Path]]:
    """Shared single/cross-run telemetry-file discovery.

    At most one file is returned per ``telemetry/`` dir (the canonical name wins
    over a legacy alias, see :func:`_pick_in_dir`).

    Args:
        output_dir: A run dir, a parent of run dirs, or a direct JSONL path.
        accepted: Accepted filenames, ordered canonical-first (canonical + any
            legacy aliases).

    Returns:
        ``[(run_dir, jsonl_path), ...]``, sorted for deterministic concat order.
    """
    p = Path(output_dir)

    # Direct path to a JSONL telemetry file.
    if p.is_file() and p.name in accepted:
        run_dir = p.parent.parent if p.parent.name == _TELEMETRY_DIR else p.parent
        return [(run_dir, p)]

    # Single run dir: <output_dir>/telemetry/<file> (one file, canonical first).
    single = _pick_in_dir(p / _TELEMETRY_DIR, accepted)
    if single is not None:
        return [(p, single)]

    # Cross-run: <output_dir>/*/telemetry/ (§7.12). Glob the telemetry dirs and
    # pick one file each; sorted so concat order — and thus the namespaced
    # run_id — is deterministic across invocations.
    pairs: list[tuple[Path, Path]] = []
    for tel_dir in sorted(_glob.glob(str(p / "*" / _TELEMETRY_DIR))):
        tel_path = Path(tel_dir)
        chosen = _pick_in_dir(tel_path, accepted)
        if chosen is not None:
            pairs.append((tel_path.parent, chosen))
    return pairs


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_runs(output_dir: str | Path) -> "DataFrame":
    """Load ``telemetry/agent_runs.jsonl`` into a ``DataFrame`` (§7.12).

    Reads one :class:`AgentRunRecord` per row (un-nesting the
    :class:`LLMIOLogger` ``extra.record`` envelope), de-dups on the deterministic
    ``run_id`` (§7.3, §7.8 — a re-executed task on resume yields the same id, so
    the duplicate is dropped on read), and concatenates across run dirs when
    ``output_dir`` is a parent of many runs (``*/telemetry/agent_runs.jsonl``,
    §7.12).

    On a cross-run concat the per-run ``run_id`` is deterministic but only unique
    *within* a run; a ``run_dir`` column is added and a ``global_run_id`` column
    (``"<run_dir_name>/<run_id>"``) gives a globally unique key, used as the
    de-dup basis across runs (within a single run, ``run_id`` is the basis).

    Args:
        output_dir: A run dir (``<dir>/telemetry/agent_runs.jsonl``), a parent of
            many run dirs, or a direct path to an ``agent_runs.jsonl`` file.

    Returns:
        A ``DataFrame`` (one row per unique run). Empty (no columns) when no
        telemetry is found.
    """
    pairs = _runs_paths(output_dir)
    if not pairs:
        logger.info("no agent_runs.jsonl found under %s", output_dir)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for run_dir, path in pairs:
        records = _read_jsonl_records(path)
        if not records:
            continue
        frame = pd.DataFrame(records)
        frame["run_dir"] = str(run_dir)
        # Namespace the deterministic run_id by run dir so a cross-run concat
        # stays globally unique (§7.12); within a single run, run_id alone is
        # unique and global_run_id collapses to "<name>/<run_id>".
        if "run_id" in frame.columns:
            frame["global_run_id"] = run_dir.name + "/" + frame["run_id"].astype(str)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)

    # De-dup: globally on global_run_id across runs (resume idempotency, §7.8).
    # keep="last" (H12): a degraded row may be SUPERSEDED on resume by a later
    # NON-degraded re-execution that the writer appends as a second physical line
    # with the same run_id (AgentTelemetry._append_run supersede). Keeping the
    # LAST row surfaces the clean outcome. This is safe for non-supersede data:
    # within a single run the writer emits a unique run_id (the duplicate is
    # dropped on write), and a cross-run concat de-dups on the unique
    # global_run_id — so keep="last" only ever changes the supersede case.
    dedup_key = "global_run_id" if "global_run_id" in df.columns else None
    if dedup_key is None and "run_id" in df.columns:
        dedup_key = "run_id"
    if dedup_key is not None:
        before = len(df)
        df = df.drop_duplicates(subset=[dedup_key], keep="last").reset_index(
            drop=True
        )
        dropped = before - len(df)
        if dropped:
            logger.info("de-dup dropped %d duplicate run row(s) on %s", dropped, dedup_key)
    return df


# ---------------------------------------------------------------------------
# Pointer resolution
# ---------------------------------------------------------------------------


def pointer_resolve(df: "DataFrame", col: str) -> "Series":
    """Resolve a column of relpath pointers to absolute paths (§7.7).

    Pointer columns (``transcript_ptr`` / ``agent_logs_ptr``)
    hold relpaths from the run's ``output_dir`` root. This joins each non-null
    pointer onto its row's ``run_dir`` (added by :func:`load_runs`); rows missing
    a ``run_dir`` fall back to the relpath unchanged. ``None``/NaN pointers stay
    ``None`` (the run has no such artifact).

    Args:
        df: A frame from :func:`load_runs` (carries ``run_dir`` + the pointer
            column).
        col: The pointer column to resolve (e.g. ``"transcript_ptr"``).

    Returns:
        A ``Series`` of resolved absolute paths as strings (or ``None`` where the
        pointer was absent), index-aligned to ``df``.

    Raises:
        KeyError: If ``col`` is not a column of ``df``.
    """
    if col not in df.columns:
        raise KeyError(
            f"{col!r} is not a column of the runs frame; "
            f"available pointer columns: "
            f"{[c for c in _POINTER_COLUMNS if c in df.columns]}"
        )
    if df.empty:
        return pd.Series([], dtype=object, index=df.index)

    has_run_dir = "run_dir" in df.columns
    ptr_vals = df[col].tolist()
    run_dirs = df["run_dir"].tolist() if has_run_dir else [None] * len(df)

    def _resolve(ptr: Any, run_dir: Any) -> Optional[str]:
        # ``pd.isna`` safely covers Python ``None`` *and* numpy float NaN (which
        # is what a None pointer becomes once a mixed column is loaded — note
        # ``isinstance(np.float64('nan'), float)`` is False, so a bare isinstance
        # check would miss it).
        if pd.isna(ptr) or ptr == "":
            return None
        ptr_str = str(ptr)
        if run_dir is not None and not pd.isna(run_dir):
            return str(Path(str(run_dir)) / ptr_str)
        return ptr_str

    # Build the result with an explicit ``object`` dtype and preserve ``None``.
    # A row-wise ``df.apply(axis=1)`` would infer a ``str`` dtype from the
    # non-null cells and coerce the ``None`` results back to ``NaN`` — losing the
    # "no such artifact" signal — so we materialize the values directly instead.
    resolved = [_resolve(ptr, rd) for ptr, rd in zip(ptr_vals, run_dirs)]
    return pd.Series(resolved, index=df.index, dtype=object)


# ---------------------------------------------------------------------------
# Fair comparison (the apples-to-apples contract, §7.11)
# ---------------------------------------------------------------------------


def _coerce_numeric(df: "DataFrame", col: str) -> "Series":
    """Return ``df[col]`` coerced to numeric (missing column → all-zero).

    Telemetry rows are JSON, so ints/floats arrive typed, but a partial/legacy
    frame may be missing a column entirely — in which case the metric is treated
    as zero rather than raising.
    """
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _terminated_by_series(df: "DataFrame") -> "Series":
    """Return ``df['terminated_by']`` normalized to lowercase strings.

    Missing column → an all-``"unknown"`` series, so the cap-bound / breakdown
    aggregates degrade gracefully on a legacy frame that predates the column.
    """
    if "terminated_by" not in df.columns:
        return pd.Series("unknown", index=df.index)
    return df["terminated_by"].astype(str).str.strip().str.lower()


def _summarize_terminated_by(values: "Series") -> str:
    """Compact, deterministic ``terminated_by`` breakdown for a group of runs.

    Renders the per-state counts as ``"state:count"`` pairs ordered by descending
    count then state name (e.g. ``"completed:2,token_budget:1"``), so a single
    table cell shows the mix of stop reasons for the agent.
    """
    counts = values.value_counts()
    items = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return ",".join(f"{state}:{int(n)}" for state, n in items)


def fair_comparison(
    df: "DataFrame", phases: tuple[str, ...] = ("eval",)
) -> "DataFrame":
    """Build the apples-to-apples per-agent comparison table (§7.11).

    The single ``df.groupby('agent')`` table is only fair if no column silently
    mixes bases or units. This helper enforces the §7.11 contract structurally:

    * **Token basis split.** ``total_tokens`` is *never* summed across rows of
      different ``token_basis``. The table reports ``inner_total_tokens`` and
      ``outer_total_tokens`` as **separate** columns (a builtin row's *outer*
      tokens are never added to an OH/T2 row's *inner* tokens under one "tokens"
      header). ``inner_tokens`` (the per-row inner field) is summed only over
      ``token_basis == "inner"`` rows.
    * **Cost basis breakout.** Cost *sums* are valid (all land in the same
      ``today_total_usd()`` USD), but the table surfaces ``cost_native_usd`` and
      ``cost_priced_from_tokens`` so the OH (``native_usd``) vs T2/builtin
      (``priced_from_tokens``) difference stays visible.
    * **``agent_calls`` annotated.** T2 reports a real ``inner_calls`` count; OH
      reports ``0`` until a verified per-call accessor exists, so OH's
      ``agent_calls`` is annotated ``"unmeasured"`` and excluded from per-call
      ratios (never put in one "calls" column as if comparable).
    * **``steps`` labelled approximate.** OH event-derived steps and T2
      episode-count are not equated; the column is summed but documented as
      ``steps (turns, approx)`` and never used as an LLM-call proxy.

    Consumed-quantity + cap-bound surface (fairness upgrade)
    -------------------------------------------------------
    To make an OH-vs-T2 A/B readable at a glance, the table additionally reports
    the CONSUMED quantities and WHICH envelope bound each agent:

    * **``inner_calls``** — the real summed inner LLM-call count for *every* agent.
      Both OH (``len(metrics.token_usages)``) and T2 (the budgeted-LiteLLM tally)
      now measure this, so unlike the legacy ``agent_calls`` column (kept as-is,
      OH → ``"unmeasured"``) this is a true measured number for both.
    * **``wall_s``** — total consumed wall-clock seconds (summed) and ``wall_s_mean``.
    * **``terminated_by``** — a compact per-agent breakdown of stop reasons
      (``"completed:2,token_budget:1"``).
    * **``n_cap_bound`` / ``cap_bound_rate``** — how many of the agent's runs were
      bound by a resource CAP (``token_budget`` / ``max_turns`` / ``timeout`` /
      ``budget_usd`` / ``budget_denied``) rather than finishing (``completed``) or
      genuinely failing (``agent_error`` / ``env_error`` / ``parse_error`` /
      ``context_len`` / ``unknown``). This surfaces the §-fairness question
      directly: did OH and T2 stop for the SAME reason under a matched envelope?

    IMPORTANT — the inner-token axis is comparable only ACROSS inner-basis agents
    (OH, T2). The ``builtin`` backend is ``token_basis="outer"``: its spend lands
    in ``outer_total_tokens`` and is **NOT** comparable to the OH/T2 inner-token
    columns (``inner_total_tokens`` / ``inner_tokens`` / ``inner_calls``). The
    ``notes`` column on the builtin row states this explicitly.

    Benchmark discriminator — ``agent`` is NOT unique across benchmarks. The
    CO-Bench ``OpenHandsBackend`` and the TB ``OpenHandsTBBackend`` both report
    ``name="openhands"``; likewise ``BuiltinBackend`` (host) and ``BuiltinTBBackend``
    both report ``name="builtin"``. New-format rows (schema v2) therefore carry a
    ``benchmark`` column (the adapter's ``name``), and when the frame holds MORE
    THAN ONE distinct non-empty ``benchmark`` (a cross-benchmark concat of
    new-format rows) the table groups by ``[agent, benchmark]`` so the two
    genuinely-different same-named backends never blend their
    ``inner_total_tokens`` / ``inner_calls`` / ``mean_score`` / ``steps`` /
    ``wall_s`` / ``terminated_by``. Legacy rows predating the column are read as
    ``""`` (unknown) and are NEVER split on: a frame with no ``benchmark``
    column, an all-``""`` column, or a single non-empty benchmark keeps the
    plain ``agent`` index (byte-identical to the historical table) — so a
    cross-benchmark concat of LEGACY rows still blends, exactly as before
    (split such data per run dir before aggregating).

    Phase gate (H6)
    ---------------
    Under ``--gate-repeats`` / quality-gating a spine candidate runs a subset of
    tasks TWICE — once in the gate-check pre-screen (``phase="gate"``) and again
    in the full evaluation (``phase="eval"``). Both write telemetry rows (now at
    DISTINCT run_ids), so including the gate subset would DOUBLE-COUNT it in the
    per-agent A/B aggregates. This helper therefore defaults to the EVAL-phase
    rows only (``phases=("eval",)``). Rows predating the ``phase`` field (no
    column, or NaN) are treated as ``"eval"`` for back-compat, so historical
    telemetry is fully counted unchanged. Pass ``phases=("eval", "gate")`` (or a
    superset) to opt the gate rows back in.

    Args:
        df: A runs frame from :func:`load_runs`.
        phases: The execution phases to include (default eval-only). A row whose
            ``phase`` is absent / NaN is treated as ``"eval"``.

    Returns:
        A ``DataFrame`` indexed by ``agent`` — or by ``(agent, benchmark)`` when
        the frame carries more than one distinct non-empty ``benchmark`` (a
        cross-benchmark concat of new-format rows) — with the basis-split token
        columns, the cost-basis breakout, the success/score aggregates, the legacy
        ``agent_calls`` column (``"unmeasured"`` for OpenHands), and the
        consumed-quantity + cap-bound columns described above. Empty when ``df`` is
        empty / has no ``agent`` column / no row survives the phase gate.
    """
    if df is None or df.empty or "agent" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    # H6 phase gate: keep only the requested phases (default eval-only) so a gate
    # subset is not double-counted. Back-compat: a missing/NaN phase is 'eval'.
    if "phase" in work.columns:
        keep = work["phase"].fillna("eval").astype(str).isin(phases)
        work = work[keep]
        if work.empty:
            return pd.DataFrame()
    # F157: group by [agent, benchmark] ONLY when the frame carries >1 distinct
    # non-empty benchmark (a cross-benchmark concat of new-format rows). Legacy
    # frames (no column, or all-'' / single benchmark) keep the plain 'agent'
    # index so every existing single-benchmark table is byte-identical.
    group_keys: list[str] = ["agent"]
    if "benchmark" in work.columns:
        bench = work["benchmark"].fillna("").astype(str)
        work["benchmark"] = bench
        if bench[bench != ""].nunique() > 1:
            group_keys = ["agent", "benchmark"]
    work["agent"] = work["agent"].astype(str)

    # Pre-coerce the numeric columns we aggregate (missing → zero).
    work["_total_tokens"] = _coerce_numeric(work, "total_tokens")
    work["_inner_tokens"] = _coerce_numeric(work, "inner_tokens")
    work["_inner_calls"] = _coerce_numeric(work, "inner_calls")
    work["_cost_usd"] = _coerce_numeric(work, "cost_usd")
    work["_steps"] = _coerce_numeric(work, "steps")
    work["_score"] = _coerce_numeric(work, "score")
    work["_wall_s"] = _coerce_numeric(work, "wall_s")

    # terminated_by → per-run cap_bound indicator (which envelope bound the run).
    terminated_by = _terminated_by_series(work)
    work["_terminated_by"] = terminated_by
    work["_cap_bound"] = terminated_by.isin(_CAP_BOUND_TERMINATED_BY).astype(int)

    token_basis = (
        work["token_basis"].astype(str)
        if "token_basis" in work.columns
        else pd.Series("inner", index=work.index)
    )
    cost_basis = (
        work["cost_basis"].astype(str)
        if "cost_basis" in work.columns
        else pd.Series("priced_from_tokens", index=work.index)
    )
    success = (
        work["success"].fillna(False).astype(bool)
        if "success" in work.columns
        else pd.Series(False, index=work.index)
    )

    # Basis-masked token columns — never blend outer + inner under one header.
    is_inner = token_basis == "inner"
    is_outer = token_basis == "outer"
    work["_inner_total_tokens"] = work["_total_tokens"].where(is_inner, 0.0)
    work["_outer_total_tokens"] = work["_total_tokens"].where(is_outer, 0.0)
    # The per-row inner_tokens / inner_calls fields are only meaningful on
    # inner-basis rows. We mask BOTH to inner-basis rows here so the aggregator is
    # robust even against legacy rows that mislabeled outer authoring spend onto
    # the inner axis (the record now zeroes these on outer-basis rows, but masking
    # keeps the surface correct symmetrically and for historical telemetry).
    work["_inner_tokens_masked"] = work["_inner_tokens"].where(is_inner, 0.0)
    work["_inner_calls_masked"] = work["_inner_calls"].where(is_inner, 0.0)

    # Cost broken out by basis (sums are valid; the breakout keeps OH vs T2/builtin
    # visible).
    work["_cost_native_usd"] = work["_cost_usd"].where(
        cost_basis == "native_usd", 0.0
    )
    work["_cost_priced_from_tokens"] = work["_cost_usd"].where(
        cost_basis == "priced_from_tokens", 0.0
    )

    work["_success"] = success.astype(int)

    # Single-key case: pass the bare string so the resulting index object is
    # bit-for-bit what pandas produced before the [agent, benchmark] split.
    grouped = work.groupby(
        group_keys[0] if len(group_keys) == 1 else group_keys, sort=True
    )
    out = pd.DataFrame(index=grouped.size().index)
    if len(group_keys) == 1:
        out.index.name = "agent"

    out["n_runs"] = grouped.size()
    out["n_success"] = grouped["_success"].sum().astype(int)
    out["success_rate"] = grouped["_success"].mean()
    out["mean_score"] = grouped["_score"].mean()

    # Token columns — separate basis-stamped headers (§7.11).
    out["inner_total_tokens"] = grouped["_inner_total_tokens"].sum().astype("int64")
    out["outer_total_tokens"] = grouped["_outer_total_tokens"].sum().astype("int64")
    out["inner_tokens"] = grouped["_inner_tokens_masked"].sum().astype("int64")

    # Cost — total is sum-valid, with the cost_basis breakout (§7.11).
    out["cost_usd"] = grouped["_cost_usd"].sum()
    out["cost_native_usd"] = grouped["_cost_native_usd"].sum()
    out["cost_priced_from_tokens"] = grouped["_cost_priced_from_tokens"].sum()

    # steps (turns, approx) — summed but NOT an LLM-call proxy (§7.11).
    out["steps_turns_approx"] = grouped["_steps"].sum().astype("int64")

    # agent_calls — OH is unmeasured; everyone else is the real summed inner_calls
    # (§7.11). Object dtype so the "unmeasured" sentinel and ints coexist.
    # RETAINED unchanged for backward compatibility (existing callers/tests).
    # Per-row comparisons read the AGENT level (the index may be a
    # (agent, benchmark) MultiIndex on a cross-benchmark frame); ``real_calls``
    # is groupwise-aligned to ``out.index``, so the positional read is exact.
    agent_level = (
        out.index.get_level_values("agent")
        if isinstance(out.index, pd.MultiIndex)
        else out.index
    )
    real_calls = grouped["_inner_calls"].sum().astype("int64")
    agent_calls: list[Any] = []
    for i, agent in enumerate(agent_level):
        if agent == _OPENHANDS_AGENT:
            agent_calls.append(_UNMEASURED)
        else:
            agent_calls.append(int(real_calls.iloc[i]))
    out["agent_calls"] = pd.Series(agent_calls, index=out.index, dtype=object)

    # --- consumed-quantity + cap-bound surface (fairness upgrade) -------------
    # inner_calls — the REAL measured INNER-basis call count for every inner-basis
    # agent (OH + T2 both emit it now). It is basis-masked: outer-basis (builtin)
    # rows contribute 0 here, mirroring ``inner_tokens`` above, so the inner-call
    # axis no longer mixes in builtin's OUTER authoring calls. (The builtin row's
    # outer authoring spend is still visible via ``outer_total_tokens``.)
    out["inner_calls"] = grouped["_inner_calls_masked"].sum().astype("int64")

    # wall_s — consumed wall-clock seconds (total + mean).
    out["wall_s"] = grouped["_wall_s"].sum()
    out["wall_s_mean"] = grouped["_wall_s"].mean()

    # terminated_by — compact per-agent breakdown of stop reasons.
    out["terminated_by"] = grouped["_terminated_by"].agg(_summarize_terminated_by)

    # cap_bound — how often (and what fraction) a resource cap bound the agent's
    # runs (token_budget / max_turns / timeout / budget_*), vs finishing or
    # genuinely failing. Surfaces WHICH cap actually bound each agent's runs.
    out["n_cap_bound"] = grouped["_cap_bound"].sum().astype("int64")
    out["cap_bound_rate"] = grouped["_cap_bound"].mean()

    # notes — make the cross-basis caveat explicit on the table itself: builtin's
    # outer tokens are NOT comparable to OH/T2 inner tokens / inner_calls, and its
    # cost_usd is a DISPLAY-ONLY price derived from its own outer token counts
    # (the authoritative ledgered authoring spend is the outer LLMClient line).
    notes: list[str] = []
    for agent in agent_level:
        if agent == _BUILTIN_AGENT:
            notes.append(
                "outer-basis: inner_total_tokens/inner_tokens/inner_calls NOT "
                "comparable to OH/T2 inner-token axis; cost_usd is display-only "
                "(priced from outer tokens, not the authoritative LLMClient ledger)"
            )
        else:
            notes.append("")
    out["notes"] = pd.Series(notes, index=out.index, dtype=object)

    return out
