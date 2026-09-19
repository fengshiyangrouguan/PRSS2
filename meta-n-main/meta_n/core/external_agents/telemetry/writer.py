"""The :class:`AgentTelemetry` JSONL writer + run-id index machinery.

Split out of the former single-module ``telemetry.py`` (mechanical move; see
the package ``__init__`` for the full design contract).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from meta_n.core.meta_layer import Trace
from meta_n.utils.llm_io_logger import LLMIOLogger

from ..backend import AgentRunResult
from ..terminated import TerminatedBy
from .attribution import attribute_utilities
from .records import SCHEMA_VERSION, AgentRunRecord, compute_run_id
from .redaction import _resolve_priced_model, redact

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime import cycles
    from meta_n.integrations.benchmark import EvalResult

logger = logging.getLogger("meta_n.external_agents.telemetry")

__all__ = [
    "iter_jsonl_objects",
    "AgentTelemetry",
]

def iter_jsonl_objects(path: "Path | str"):
    """Yield the parsed object for every well-formed line of a JSONL file.

    The canonical line-scan for ``agent_runs.jsonl`` readers (the writer-side
    ``_load_run_id_index`` here and the orchestrator's ``_read_agent_run_rows``):
    blank lines, ``json.JSONDecodeError`` lines and valid-JSON lines whose top
    level is not an object are SKIPPED (every consumer immediately calls
    ``.get`` on the yield, so a list/scalar/null top level is as malformed as an
    undecodable line); ``OSError`` (missing / unreadable file) PROPAGATES so
    each caller keeps its own error policy (partial-index vs empty-result).
    Envelope unwrapping (``extra.record`` vs top-level) is deliberately
    caller-specific — the strictness of the unwrap is part of each reader's
    contract.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            yield obj


# ---------------------------------------------------------------------------
# Telemetry writer
# ---------------------------------------------------------------------------

#: The genuinely-DEGRADED ``terminated_by`` states — they describe HOW a run
#: ended (a resource cap, an environment/parse fault, or an up-front budget
#: denial), NOT the grade. :meth:`AgentTelemetry.finish_record` leaves these
#: untouched during scorer reconciliation; every OTHER tag (COMPLETED /
#: AGENT_ERROR / UNKNOWN / MAX_TURNS / TOKEN_BUDGET) is coerced to agree with the
#: authoritative scorer outcome. Reconciling on this NOT-degraded axis (rather
#: than a {COMPLETED, AGENT_ERROR} allow-list) keeps the reconciliation decoupled
#: from any backend's clean-completion marker vocabulary.
_DEGRADED_TERMINATED_BY: frozenset[str] = frozenset(
    {
        TerminatedBy.TIMEOUT.value,
        TerminatedBy.BUDGET_DENIED.value,
        TerminatedBy.BUDGET_USD.value,
        TerminatedBy.ENV_ERROR.value,
        TerminatedBy.PARSE_ERROR.value,
        TerminatedBy.CONTEXT_LEN.value,
    }
)


class AgentTelemetry:
    """Append-only, flock-safe telemetry writer for external-agent runs.

    Owns the ``telemetry/`` subtree under ``output_dir``:

    * ``telemetry/agent_runs.jsonl`` — one :class:`AgentRunRecord` per
      ``execute()`` (flock-append; parallel task workers append without a
      barrier, reusing :class:`LLMIOLogger`'s write path, §7.7).
    * ``telemetry/schema.json`` — written once: schema version, field docs, basis
      enums.

    Rows are de-duplicated on write by the deterministic ``run_id``
    (:func:`compute_run_id`) so a resume that re-executes a task does not append a
    second line (§7.3, §7.8). The de-dup index is rebuilt from the on-disk JSONL
    at construction so it survives a process restart.

    The spine owns timing/tokens, so the typical flow is:
    :meth:`start_record` → backend ``run`` → :meth:`build_trace` → either
    :meth:`finish_record` (success/normal) or one of the degraded finishers
    (:meth:`finish_budget_denied` / :meth:`finish_timeout` / :meth:`finish_error`).
    """

    RUNS_FILE = "agent_runs.jsonl"
    SCHEMA_FILE = "schema.json"

    def __init__(self, output_dir: str | Path, *, generation: int = 0):
        """Create the ``telemetry/`` tree and write ``schema.json`` once.

        Args:
            output_dir: Run root; ``telemetry/`` is created beneath it.
            generation: Default generation coordinate for records started by a
                solver that does not expose ``generation`` (test doubles /
                harness scripts). ``ExternalAgentSolver`` always stamps its own,
                so production reads never fall back to this.
        """
        self.output_dir = Path(output_dir)
        self.telemetry_dir = self.output_dir / "telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.generation = int(generation)

        # Reuse the existing flock-append write path (§7.7).
        self._runs_log = LLMIOLogger(
            self.telemetry_dir / self.RUNS_FILE, source="agent_runs"
        )

        # Resume idempotency: rebuild the seen-run_id index from disk so a
        # re-executed task de-dups against rows written before the restart.
        #
        # H12 supersede: run_ids whose LAST on-disk row is DEGRADED (a timeout /
        # env / parse / budget fault). A later NON-degraded re-execution of the
        # same run_id is allowed to append (supersede) instead of de-dupping, so
        # a resume that re-ran a transiently-failed task surfaces the clean
        # outcome (the reader keeps the LAST row). Rebuilt from disk (like
        # ``_seen_run_ids``) so a CROSS-PROCESS resume — a fresh writer after a
        # restart — recognizes a PRIOR process's degraded row as a supersede
        # candidate and appends the clean re-execution, rather than dropping it
        # (which would keep the scored-zero degraded row and bias scores down).
        #
        # Both indexes are built in ONE pass over the (possibly large) resume
        # JSONL rather than two full scans.
        self._seen_run_ids, self._degraded_run_ids = self._load_run_id_index()

        # T-R2.1 gate-reuse re-stamp cache: the serialized row of each GATE-phase
        # run we wrote, keyed by run_id. Under the 1.6 gate-reuse optimization a
        # gate-PASSING task is reused at eval WITHOUT re-solving, so its only
        # physical telemetry row is the gate row — which fair_comparison drops
        # (eval-only default), undercounting the positively-selected remainder.
        # :meth:`restamp_reused_gate` re-emits that cached row as an EVAL-phase
        # clone keyed on the SAME run_id, so both read-side de-dups (load_runs
        # keep='last', _read_agent_run_rows newest-wins) supersede the gate row and
        # the single physical run is counted once as eval. Process-local: a row
        # written by a prior process is not in this cache (only on disk), so a
        # cross-process resume simply cannot re-stamp — which is correct, the
        # in-process reuse optimization is exactly the regime this addresses.
        self._gate_rows: dict[str, dict] = {}
        # Self-bounding coordinate for ``_gate_rows`` (spinefix). A gate-REJECTED
        # candidate is NEVER evaluated, so :meth:`restamp_reused_gate` (the only
        # eviction) never pops its cached gate rows and they would live for the
        # whole process — one dead entry per rejected task per generation,
        # unbounded growth. The child loop is SEQUENTIAL (gate → reject/continue
        # OR pass → eval+restamp) and fully resolves a candidate before the next
        # candidate is gated, so the arrival of a gate row for a DIFFERENT
        # candidate coordinate proves every prior candidate is done (passed →
        # rows already restamped-popped, or rejected → rows now dead). We track
        # the last-cached ``(generation, candidate_id, depth)`` and clear stale
        # rows when it changes, bounding the cache to the current candidate.
        self._gate_rows_coord: "tuple[int, str, int] | None" = None

        self._write_schema_once()

    # -- construction helpers ------------------------------------------------

    def _load_run_id_index(self) -> "tuple[set[str], set[str]]":
        """One pass over ``agent_runs.jsonl`` → ``(seen_run_ids, degraded_run_ids)``.

        ``seen`` is the resume de-dup index (every run_id on disk). ``degraded``
        is the H12 supersede set: for each run_id the LAST on-disk row wins (the
        keep='last' read contract), so the scan records, per run_id, whether its
        most recent row is DEGRADED (``terminated_by`` in
        :data:`_DEGRADED_TERMINATED_BY`) — a CROSS-PROCESS clean re-execution of
        such a run_id is appended (superseding) instead of being de-dupped away.

        The two id reads deliberately differ: ``seen`` falls back to a top-level
        ``run_id`` even when the ``extra.record`` envelope exists but lacks one
        (forward compatibility), while the degraded marker reads ``run_id`` and
        ``terminated_by`` off the SAME record dict (the envelope when truthy,
        else the top level) so the pair stays coherent. Never raises — on a
        read failure the partial index is returned (per-task de-dup still
        applies), and malformed lines (an envelope whose ``record`` is
        null/non-dict; non-object top levels are already dropped by
        :func:`iter_jsonl_objects`) are skipped and counted like undecodable
        lines.
        """
        seen: set[str] = set()
        last_degraded: dict[str, bool] = {}
        malformed = 0
        path = self.telemetry_dir / self.RUNS_FILE
        if not path.exists():
            return seen, set()
        try:
            for obj in iter_jsonl_objects(path):
                # The flock-append path nests the record under
                # ``extra.record`` (LLMIOLogger envelope); fall back to a
                # top-level ``run_id`` for forward compatibility. A PRESENT but
                # null/non-dict ``record`` is a malformed envelope, not a
                # top-level record — skip it (the never-raise contract).
                extra = obj.get("extra")
                env_rec = extra.get("record", {}) if isinstance(extra, dict) else {}
                if not isinstance(env_rec, dict):
                    malformed += 1
                    continue
                rid = env_rec.get("run_id") or obj.get("run_id")
                if rid:
                    seen.add(rid)
                # H12 marker: run_id + terminated_by off the SAME record
                # (no cross-envelope fallback — the pair must be coherent).
                rec = env_rec or obj
                deg_rid = rec.get("run_id")
                if deg_rid:
                    last_degraded[deg_rid] = (
                        rec.get("terminated_by") in _DEGRADED_TERMINATED_BY
                    )
        except OSError as e:  # noqa: BLE001
            logger.warning("could not pre-load run_ids from %s: %r", path, e)
        if malformed:
            logger.warning(
                "skipped %d malformed envelope line(s) while pre-loading "
                "run_ids from %s", malformed, path,
            )
        return seen, {rid for rid, is_deg in last_degraded.items() if is_deg}

    def _write_schema_once(self) -> None:
        """Write ``telemetry/schema.json`` (version + field docs + basis enums).

        Write-once PER VERSION: an existing file is kept unless its persisted
        ``schema_version`` differs from :data:`SCHEMA_VERSION` — a resumed
        pre-v2 dir gets its header refreshed so it describes the rows the
        current writer appends (per-row ``schema_version`` stamps keep mixed
        files readable either way). An unreadable/invalid existing file is
        left as-is (the no-crash contract).
        """
        path = self.telemetry_dir / self.SCHEMA_FILE
        if path.exists():
            try:
                persisted = json.loads(path.read_text(encoding="utf-8")).get(
                    "schema_version"
                )
            except (OSError, ValueError, AttributeError):
                return  # silent-on-corrupt: leave the file alone
            if persisted == SCHEMA_VERSION:
                return
        schema = {
            "schema_version": SCHEMA_VERSION,
            "run_id": "sha1(f'{generation}:{candidate_id}:{task_id}:{depth}')[:16] "
            "— deterministic, session/container-uuid independent (§7.3)",
            "token_basis": {
                "outer": "builtin backend — native outer LLMClient tokens",
                "inner": "openhands / terminus2 — agent inner spend (Trace.inner_*)",
            },
            "cost_basis": {
                "native_usd": "OpenHands accumulated_cost (its own litellm ledger)",
                "priced_from_tokens": "Terminus 2 / builtin — priced from token counts",
            },
            "attribution": {
                "utilities_called=null": "unmeasurable on this backend (no command stream)",
                "utilities_called=[]": "measured: none of the injected utilities used",
            },
            "terminated_by": [m.value for m in TerminatedBy],
            "benchmark": (
                "adapter name the row was produced under; '' on legacy rows"
            ),
            "records": {
                "agent_runs.jsonl": "one AgentRunRecord per execute()",
            },
        }
        try:
            path.write_text(
                json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as e:  # noqa: BLE001
            logger.warning("could not write telemetry schema.json: %r", e)

    # -- record lifecycle ----------------------------------------------------

    def _run_coordinates(
        self, task: object, solver: object
    ) -> "tuple[str, int, int, str]":
        """``(task_id, depth, generation, candidate_id)`` — the run_id coordinates.

        Read with the SAME ``getattr`` fallbacks everywhere (:meth:`start_record`
        and :meth:`restamp_reused_gate`), so the restamp's coordinate match can
        never drift from the coordinates the gate row was stamped with.
        """
        task_id = str(getattr(task, "task_id", "") or "")
        depth = int(getattr(solver, "depth", 1) or 1)
        generation = int(getattr(solver, "generation", self.generation) or 0)
        candidate_id = str(getattr(solver, "candidate_id", "") or "")
        return task_id, depth, generation, candidate_id

    def start_record(
        self,
        task: object,
        solver: object,
        parent_run_id: Optional[str] = None,
    ) -> AgentRunRecord:
        """Open a fresh :class:`AgentRunRecord` for one ``execute()``.

        Reads the four run_id coordinates (generation, candidate_id, task_id,
        depth) and the agent name off the ``solver`` / ``task`` via ``getattr``
        with safe fallbacks (the solver attributes are populated by
        ``ExternalAgentSolver`` / the orchestrator). The run_id is computed
        deterministically here so the de-dup contract holds end-to-end.

        Args:
            task: The :class:`~meta_n.core.meta_layer.TaskDescription` (or any
                object exposing ``task_id``).
            solver: The :class:`ExternalAgentSolver` (exposes ``depth``,
                ``backend``, and the ``generation``/``candidate_id`` coordinates).
            parent_run_id: The originating run_id when this run is a re-solve /
                child, else ``None``.

        Returns:
            A populated-but-unfinished :class:`AgentRunRecord`.
        """
        task_id, depth, generation, candidate_id = self._run_coordinates(
            task, solver
        )
        backend = getattr(solver, "backend", None)
        agent = str(getattr(backend, "name", None) or "builtin")
        # Benchmark coordinate (F157): stamped onto the solver from the adapter's
        # ``name`` so same-named agents (the two "openhands" / two "builtin"
        # backends) stay separable across benchmarks. Test doubles without the
        # attribute get "" — the same getattr-fallback convention as every other
        # coordinate.
        benchmark = str(getattr(solver, "benchmark", "") or "")
        token_basis = "outer" if getattr(backend, "outer_token_mode", False) else "inner"
        max_turns = int(getattr(solver, "max_turns", 0) or 0)

        # Execution-phase discriminator for the run_id (de-dup correctness): the
        # orchestrator stamps ``solver.execution_phase = "gate"`` around the
        # gate-check ``execute()`` so a gate row does not de-dup against the
        # later full-eval row for the SAME (generation, candidate_id, task_id,
        # depth). Absent the stamp this reads ``"eval"`` → the run_id basis is
        # byte-identical to the historical formula (the common full-eval path).
        phase = str(getattr(solver, "execution_phase", "eval") or "eval")
        # Repeated-eval / gate-repeat sample index (de-dup correctness): the
        # orchestrator stamps ``solver.repeat_index = r`` around each extra
        # median-of-R / gate sample so the R executions at IDENTICAL coordinates
        # write R distinct rows instead of de-dupping to one. Absent the stamp
        # this reads 0 → the run_id basis is byte-identical to the historical
        # formula (the common single-sample path).
        repeat_index = getattr(solver, "repeat_index", 0) or 0

        run_id = compute_run_id(
            generation, candidate_id, task_id, depth, phase, repeat_index
        )
        rec = AgentRunRecord(
            run_id=run_id,
            generation=generation,
            candidate_id=candidate_id,
            task_id=task_id,
            agent=agent,
            depth=depth,
            parent_run_id=parent_run_id,
            phase=phase,
            benchmark=benchmark,
            token_basis=token_basis,
            max_turns=max_turns,
        )
        # Stamp the priced (bare) model id as a NON-FIELD attribute so it never
        # enters the serialized schema (``to_row``/``asdict`` ignore non-fields)
        # but ``finish_record`` can use it to derive a DISPLAY-only USD cost for an
        # outer-basis (builtin) row, whose ``run.cost_usd`` is always 0.0 (the
        # authoritative spend is the outer LLMClient ledger; the daily cap is
        # billed there, never here). Resolved off the solver/backend's LLMClient.
        rec._priced_model = _resolve_priced_model(solver, backend)  # type: ignore[attr-defined]
        return rec

    def build_trace(
        self,
        task: object,
        run: AgentRunResult,
        evalr: "EvalResult",
        depth: int,
        outer_token_mode: bool = False,
    ) -> Trace:
        """Build the meta-n :class:`Trace` from a finished run + eval (§5).

        Agent spend is **inner** — ``run.agent_*`` populate ``Trace.inner_*`` —
        for the OpenHands / Terminus 2 backends, whose tokens are reported as
        inner spend and whose outer-return int is ``0``.

        The builtin backend (``outer_token_mode=True``) is the exception: its
        spend is returned as the *outer* int by ``execute()``, so writing the
        same counts into ``Trace.inner_*`` would let any consumer that sums both
        channels (e.g. ``_evaluate_candidate`` adding the outer int to
        ``candidate.total_tokens`` AND summing ``trace.inner_tokens``)
        double-count. We therefore zero the Trace's inner channel when
        ``outer_token_mode`` is set, keeping ``token_basis='outer'`` authoritative
        on the row while leaving the Trace's inner channel honest (plan §7.1).

        The script/solution text, std streams, score and feedback are sourced per
        the manifest §5 mapping. ``terminated_by`` is **not** a Trace field (it
        lives on the record / JSONL only).

        Args:
            task: The task (exposes ``task_id``).
            run: The backend's :class:`AgentRunResult`.
            evalr: The scorer's ``EvalResult`` (carries ``score``/``success``/
                ``feedback``/``valid``/``feasible``).
            depth: The solver depth for this run.
            outer_token_mode: ``True`` when the backend reports its spend as the
                outer int (builtin) — the Trace's inner channel is then zeroed to
                avoid double-counting against the outer return.

        Returns:
            A populated :class:`Trace`.
        """
        success = bool(getattr(evalr, "success", False))
        # For outer-token backends the spend is reported as the outer int, so the
        # Trace's inner channel is zeroed to keep token accounting single-source.
        inner_tokens = 0 if outer_token_mode else int(run.agent_tokens or 0)
        inner_prompt = 0 if outer_token_mode else int(run.agent_prompt_tokens or 0)
        inner_completion = (
            0 if outer_token_mode else int(run.agent_completion_tokens or 0)
        )
        inner_calls = 0 if outer_token_mode else int(run.agent_calls or 0)
        return Trace(
            task_id=str(getattr(task, "task_id", "") or ""),
            depth=int(depth),
            script=self._solution_text(run),
            stdout=run.stdout_tail or "",
            stderr=run.stderr_tail or "",
            exit_code=0 if success else 1,
            success=success,
            score=float(getattr(evalr, "score", 0.0) or 0.0),
            reasoning=run.reasoning_summary or "",
            duration_s=float(run.wall_s or 0.0),
            error_summary=(run.failure_mode or "")[:200],
            eval_feedback=str(getattr(evalr, "feedback", "") or ""),
            inner_tokens=inner_tokens,
            inner_prompt_tokens=inner_prompt,
            inner_completion_tokens=inner_completion,
            inner_calls=inner_calls,
        )

    @staticmethod
    def _solution_text(run: AgentRunResult) -> str:
        """Best-effort solution text for ``Trace.script`` (diff / solve.py / transcript).

        The env provider's ``extract_solution`` result (the T2 diff / CO-Bench
        ``solve.py``) is the authoritative ``script`` and the spine stamps it onto
        the run as a ``solution`` attribute when available; absent that, fall back
        to the transcript so the Trace still carries something inspectable.
        """
        return str(getattr(run, "solution", "") or run.transcript or "")

    def finish_record(
        self,
        rec: AgentRunRecord,
        run: AgentRunResult,
        evalr: "EvalResult",
        lease: object = None,
    ) -> None:
        """Populate the run/eval-derived fields on ``rec`` and flock-append it.

        Folds in tokens/cost (with their bases), control + termination, scoring
        (including the additive ``valid``/``feasible`` ``EvalResult`` fields),
        non-invasive utility attribution, timing and diagnostics; then writes the
        row to ``agent_runs.jsonl`` (de-duped by ``run_id``). Never raises — a
        lost telemetry line must not sink a run.

        Args:
            rec: The record opened by :meth:`start_record`.
            run: The backend's :class:`AgentRunResult`.
            evalr: The scorer's ``EvalResult`` (or ``None`` on a degraded path).
            lease: The :class:`~meta_n.core.external_agents.env.EnvLease` (unused
                here beyond keeping the call signature stable; provisioning timing
                is set by the spine).
        """
        self._finish_tokens_cost(rec, run)
        self._finish_control_scoring(rec, run, evalr)
        self._finish_attribution(rec, run)
        self._finish_pointers_timing(rec, run)
        self._append_run(rec)

    def _finish_tokens_cost(
        self, rec: AgentRunRecord, run: AgentRunResult
    ) -> None:
        """:meth:`finish_record` sub-phase: tokens / cost (with their bases)."""
        # token / cost
        #
        # ``total_tokens``/``prompt``/``completion`` are the row's own token
        # quantity regardless of basis (for an outer-basis builtin row these are
        # the legitimate OUTER authoring totals). The INNER axis, however, is only
        # meaningful on an inner-basis row: on an outer-basis row the spend was
        # reported as the outer int, so populating ``inner_tokens``/``inner_calls``
        # with the same counts would mislabel outer authoring spend as inner agent
        # spend (and force downstream aggregators to mask it). We therefore zero
        # the inner axis on outer-basis rows — matching what ``build_trace``
        # already does for ``Trace.inner_*`` — so the record is self-consistent
        # without the aggregator needing to mask the mislabel.
        is_outer_basis = rec.token_basis == "outer"
        rec.prompt_tokens = int(run.agent_prompt_tokens or 0)
        rec.completion_tokens = int(run.agent_completion_tokens or 0)
        rec.total_tokens = int(run.agent_tokens or 0)
        rec.cached_tokens = int(run.agent_cached_tokens or 0)
        rec.inner_tokens = 0 if is_outer_basis else int(run.agent_tokens or 0)
        rec.inner_calls = 0 if is_outer_basis else int(run.agent_calls or 0)
        rec.cost_usd = float(run.cost_usd or 0.0)
        rec.cost_basis = run.cost_basis or rec.cost_basis

        # DISPLAY-ONLY cost for an outer-basis (builtin) row. The builtin backends
        # report ``cost_usd=0.0`` because their authoring spend is billed on the
        # OUTER LLMClient ledger (the daily cap is enforced there; re-billing it
        # here would double-count). But a fair_comparison on a PRICED backbone then
        # reads the builtin control at $0 while OH/T2 read their true spend. We
        # price the row from ITS OWN outer token counts (basis is already
        # ``priced_from_tokens``) so the headline cost column is self-consistent —
        # WITHOUT touching any ledger. Defensive: an unpriced / $0 / unknown model
        # (e.g. a self-hosted $0 backbone, or a model absent from PRICING) leaves
        # cost_usd at 0.0 (compute_cost_usd raises KeyError for an unknown model).
        if is_outer_basis and rec.cost_usd == 0.0:
            model = getattr(rec, "_priced_model", "") or ""
            if model:
                try:
                    from meta_n.utils.cost_tracker import compute_cost_usd

                    rec.cost_usd = float(
                        compute_cost_usd(
                            model,
                            rec.prompt_tokens,
                            rec.completion_tokens,
                            rec.cached_tokens,
                        )
                    )
                except Exception:  # noqa: BLE001 - pricing must never sink a write
                    pass

    def _finish_control_scoring(
        self, rec: AgentRunRecord, run: AgentRunResult, evalr: "EvalResult"
    ) -> None:
        """:meth:`finish_record` sub-phase: control / termination + scoring."""
        # control / termination
        rec.steps = int(run.steps or 0)
        rec.terminated_by = self._term_value(run.terminated_by)
        rec.failure_mode = run.failure_mode

        # scoring (additive valid/feasible default-True when absent)
        if evalr is not None:
            rec.score = float(getattr(evalr, "score", 0.0) or 0.0)
            rec.raw_score = float(getattr(evalr, "raw_score", 0.0) or 0.0)
            rec.success = bool(getattr(evalr, "success", False))
            rec.feasibility = bool(getattr(evalr, "feasible", True))
            rec.validity = bool(getattr(evalr, "valid", True))
            rec.eval_feedback = str(getattr(evalr, "feedback", "") or "")

            # Reconcile termination against the AUTHORITATIVE scorer outcome.
            # Some backends derive ``terminated_by`` from a coarse, divergent
            # signal — e.g. the builtin depth-1 path stamps COMPLETED/AGENT_ERROR
            # from the native bash-executor trace.success, which only reflects
            # whether bash accepted the text, not whether the task was solved. The
            # env provider's Scorer is the real grader, so when it ran we keep
            # ``success``/``terminated_by`` coherent: a scorer PASS that the
            # backend tagged anything non-degraded (AGENT_ERROR / UNKNOWN /
            # MAX_TURNS / TOKEN_BUDGET) becomes COMPLETED, and a scorer FAIL that
            # the backend tagged COMPLETED becomes AGENT_ERROR with a diagnostic.
            #
            # We reconcile on the GRADE axis (independent of any backend's
            # terminal-state vocabulary): only the genuinely-DEGRADED "how it
            # ended" states are left untouched, because they describe a resource
            # cap / environment fault, not the grade. This is intentionally NOT
            # keyed off a {COMPLETED, AGENT_ERROR} allow-list — a clean ran-but-
            # wrong TB run is tagged UNKNOWN (``_external_tb`` ``_CLEAN_FAILURE_TAGS``),
            # so a future partial-credit scorer returning a PASS on an UNKNOWN row
            # must still coerce to COMPLETED rather than ship an internally-
            # contradictory ``(terminated_by=unknown, success=True)`` row.
            if rec.terminated_by not in _DEGRADED_TERMINATED_BY:
                if rec.success and rec.terminated_by != TerminatedBy.COMPLETED.value:
                    rec.terminated_by = TerminatedBy.COMPLETED.value
                    rec.failure_mode = None
                elif (
                    not rec.success
                    and rec.terminated_by == TerminatedBy.COMPLETED.value
                ):
                    rec.terminated_by = TerminatedBy.AGENT_ERROR.value
                    if not rec.failure_mode:
                        rec.failure_mode = "scored_fail"

    def _finish_attribution(
        self, rec: AgentRunRecord, run: AgentRunResult
    ) -> None:
        """:meth:`finish_record` sub-phase: non-invasive utility attribution."""
        # attribution (non-invasive; None-vs-[] preserved)
        called, call_counts = attribute_utilities(
            run.command_history,
            rec.utilities_available,
            bool(run.attribution_available),
        )
        rec.utilities_called = called
        rec.utilities_call_counts = call_counts
        rec.attribution_available = bool(run.attribution_available)
        # command_count is the load-bearing discriminator between a MEASURED-zero
        # (``command_count > 0`` AND ``utilities_called == []`` — a real command
        # stream was captured, none of the staged utilities were called) and a
        # LOST/ambiguous stream (``command_count == 0`` — no stream was captured
        # though attribution_available may be True, so a [] / None
        # utilities_called is non-behavioral, NOT a measured zero). A serialized
        # count makes the two provably distinct on read (§7.6).
        command_history = run.command_history or []
        rec.command_count = len(command_history)

        # Lost-stream alarm: if utilities were staged AND the backend advertises
        # that capture was POSSIBLE (attribution_available) yet no command stream
        # came back, the attribution was EXPECTED but is UNMEASURED — a [] here is
        # not an honest measured-none. Surface it so a silently-lost stream does
        # not masquerade as a zero call-rate.
        if (
            rec.utilities_available
            and not command_history
            and bool(run.attribution_available)
        ):
            logger.warning(
                "attribution expected but unmeasured: %d utilities staged but "
                "command_history empty (task=%s agent=%s)",
                len(rec.utilities_available),
                rec.task_id,
                rec.agent,
            )

    def _finish_pointers_timing(
        self, rec: AgentRunRecord, run: AgentRunResult
    ) -> None:
        """:meth:`finish_record` sub-phase: pointers + timing + diagnostics."""
        # pointers (relpaths from output_dir; measured-zero [] is provably distinct
        # from a lost stream once these resolve). Best-effort: never raises.
        #
        # Do NOT clobber an already-set pointer: the spine runs ``dereap_agent_logs``
        # BEFORE ``finish_record`` so the DURABLE archive-relative pointer (the copy
        # of the scratch-lease ``agent_logs``) is already stamped on ``rec``. For
        # OH/T2/TB backends ``run.artifacts_path`` is the scratch lease dir OUTSIDE
        # ``output_dir``, so ``_rel_to_output`` returns ``None``; overwriting the
        # de-reaped pointer with that ``None`` is exactly the bug that left
        # ``agent_runs.jsonl`` carrying ``transcript_ptr=null`` even though the logs
        # were copied to a resolvable archive location. Honor the de-reaped pointer.
        rec.agent_logs_ptr = rec.agent_logs_ptr or self._rel_to_output(
            getattr(run, "artifacts_path", None)
        )
        if rec.transcript_ptr is None and rec.agent_logs_ptr:
            rec.transcript_ptr = rec.agent_logs_ptr

        # timing (provision/score split is set by the spine before finish)
        rec.wall_s = float(run.wall_s or 0.0)
        rec.agent_s = rec.agent_s or float(run.wall_s or 0.0)

        # diagnostics — use the (possibly reconciled) rec.failure_mode, not the
        # raw run.failure_mode, so error_summary stays coherent with terminated_by
        # after the scorer reconciliation above.
        rec.error_summary = (rec.failure_mode or rec.error_summary or "")[:200]

    # -- degraded finishers (build a scored-zero Trace + write the row) ------

    def finish_budget_denied(self, task: object, rec: AgentRunRecord, depth: int) -> Trace:
        """Record a pre-check budget denial and return a scored-zero Trace.

        The spine calls this when :meth:`CostGuard.precheck` denied the run up
        front (no agent ran). Sets ``terminated_by=BUDGET_DENIED`` (§2.3) and
        writes the row.
        """
        return self._finish_degraded(
            task, rec, depth,
            terminated_by=TerminatedBy.BUDGET_DENIED,
            failure_mode="budget_denied",
            error_summary="run denied by budget pre-check",
        )

    def finish_timeout(self, task: object, rec: AgentRunRecord, depth: int) -> Trace:
        """Record a hard-timeout termination and return a scored-zero Trace (§2.3)."""
        return self._finish_degraded(
            task, rec, depth,
            terminated_by=TerminatedBy.TIMEOUT,
            failure_mode="agent_timeout",
            error_summary="run exceeded the hard wall-clock timeout",
        )

    def finish_error(
        self, task: object, rec: AgentRunRecord, exc: BaseException, depth: int
    ) -> Trace:
        """Record an unexpected agent/env error and return a scored-zero Trace.

        Used for the ``execute()`` ``BaseException`` wrap (``CancelledError`` is
        re-raised by the caller and never reaches here). The exception repr is
        redacted and truncated before it is written.
        """
        summary = redact(f"{type(exc).__name__}: {exc}")[:200]
        return self._finish_degraded(
            task, rec, depth,
            terminated_by=TerminatedBy.AGENT_ERROR,
            failure_mode="agent_error",
            error_summary=summary,
        )

    def _finish_degraded(
        self,
        task: object,
        rec: AgentRunRecord,
        depth: int,
        *,
        terminated_by: TerminatedBy,
        failure_mode: str,
        error_summary: str,
    ) -> Trace:
        """Shared degraded-path finisher: zero-score Trace + a written row."""
        rec.terminated_by = terminated_by.value
        rec.failure_mode = failure_mode
        rec.error_summary = error_summary[:200]
        rec.success = False
        rec.score = 0.0
        rec.raw_score = 0.0
        # No attribution measured on a degraded path.
        rec.utilities_called = None
        rec.attribution_available = False
        # H12: mark this run_id degraded so a later NON-degraded re-execution can
        # supersede it (recorded BEFORE the append so the first degraded write
        # itself is not mistaken for a supersede candidate).
        self._degraded_run_ids.add(rec.run_id)
        self._append_run(rec)
        return Trace(
            task_id=str(getattr(task, "task_id", "") or ""),
            depth=int(depth),
            success=False,
            score=0.0,
            exit_code=1,
            error_summary=error_summary[:200],
            duration_s=float(rec.wall_s or 0.0),
        )

    # -- gate-reuse re-stamp (T-R2.1) ---------------------------------------

    def restamp_reused_gate(self, task: object, solver: object) -> bool:
        """Re-emit a REUSED gate-phase run as an eval-phase row (T-R2.1).

        Under the 1.6 gate-reuse optimization a gate-PASSING task is reused at
        eval *without* re-solving (``_evaluate_candidate``'s ``precomputed``
        branch), so its ONLY physical telemetry row is the ``phase="gate"`` row
        written during the gate check. ``fair_comparison`` defaults to the
        eval-phase rows and drops ``phase="gate"`` (so the legacy double-solve is
        not double-counted), which silently undercounts ``n_runs`` over a
        positively-selected remainder (gate-pass requires a success).

        This reads the SAME four coordinates :meth:`start_record` uses (via the
        shared :meth:`_run_coordinates` helper) and matches every cached gate
        row on them (covering ``--gate-repeats R>1``), then re-emits each with
        ``phase`` flipped to ``"eval"`` — keyed on its own ``run_id``. Both
        read-side de-dups
        (``load_runs`` keep='last'; ``_read_agent_run_rows`` newest-wins) then
        supersede the gate row with its eval clone, so each single physical run is
        counted exactly ONCE as eval. No new agent ran; no token/cost is
        duplicated (the rollups de-dup on ``run_id`` too). A gate-FAILED task is
        never reused, so its ``phase="gate"`` row is correctly never re-stamped and
        stays excluded.

        Returns ``True`` when a cached gate row was found and re-emitted, else
        ``False`` (e.g. a non-gate ``precomputed`` source such as the consolidation
        reuse, or a cross-process resume where the gate row is only on disk). Never
        raises — a lost telemetry line must not sink a run.
        """
        task_id, depth, generation, candidate_id = self._run_coordinates(
            task, solver
        )
        # Re-stamp EVERY cached gate row for these four coordinates, not just the
        # r0 sample: under ``--gate-repeats R>1`` the gate phase wrote R distinct
        # rows (run_ids ``…:gate``, ``…:gate:r1``, … ``…:gate:r{R-1}``), all cached
        # by ``_append_run``. Re-stamping only the r0 row would leave the other
        # R-1 PHYSICAL gate solves (and their real token/cost spend) tagged
        # ``phase="gate"`` and so dropped by the eval-default fair_comparison view,
        # undercounting the reused task's runs/cost by a factor of R. We match on
        # the four coordinates carried in the cached row dict (which avoids having
        # to know R here) and re-emit each as an eval-phase clone keyed on its OWN
        # run_id, so every physical gate run is counted once as eval.
        matches = [
            (rid, row)
            for rid, row in self._gate_rows.items()
            if row.get("phase") == "gate"
            and int(row.get("generation", 0) or 0) == generation
            and str(row.get("candidate_id", "") or "") == candidate_id
            and str(row.get("task_id", "") or "") == task_id
            and int(row.get("depth", 1) or 1) == depth
        ]
        if not matches:
            return False
        restamped = False
        for gate_run_id, row in matches:
            self._gate_rows.pop(gate_run_id, None)
            eval_row = dict(row)
            eval_row["phase"] = "eval"
            try:
                self._runs_log.log(extra={"record": eval_row})
                logger.debug(
                    "restamp gate->eval run_id=%s task=%s: reused gate run now "
                    "counts once as eval (supersedes the gate row on read)",
                    gate_run_id, task_id,
                )
                restamped = True
            except Exception as e:  # noqa: BLE001 - a lost line must not sink a run
                logger.warning("failed to re-stamp reused gate run row: %r", e)
        return restamped

    # -- low-level append + de-dup ------------------------------------------

    def _append_run(self, rec: AgentRunRecord) -> None:
        """Flock-append ``rec`` to ``agent_runs.jsonl``, de-duped by ``run_id``.

        A re-executed task on resume yields the same deterministic ``run_id``
        (§7.3); if that id is already on disk (or written earlier this process),
        the row is dropped rather than appended. Never raises.
        """
        try:
            if rec.run_id in self._seen_run_ids:
                # H12: a CLEAN (resolved) re-execution supersedes a prior DEGRADED
                # row for the same run_id (a resume that re-ran a transiently-failed
                # task). The superseding row is appended (the keep='last' reader
                # surfaces it); every other duplicate is dropped. "Clean" requires
                # BOTH a non-degraded terminal state AND a resolved outcome
                # (success, or a clean COMPLETED) — so a re-execution that merely
                # AGENT_ERRORed / scored-fail does NOT overwrite a degraded row.
                superseding = (
                    rec.run_id in self._degraded_run_ids
                    and rec.terminated_by not in _DEGRADED_TERMINATED_BY
                    and (
                        rec.success
                        or rec.terminated_by == TerminatedBy.COMPLETED.value
                    )
                )
                if not superseding:
                    logger.debug(
                        "de-dup: run_id %s already written; dropping duplicate row",
                        rec.run_id,
                    )
                    return
                # The clean re-execution wins: drop the degraded marker so a
                # later duplicate of the now-clean row de-dups normally.
                self._degraded_run_ids.discard(rec.run_id)
                logger.debug(
                    "supersede: run_id %s re-executed non-degraded; appending to "
                    "override the prior degraded row",
                    rec.run_id,
                )
            self._seen_run_ids.add(rec.run_id)
            row = rec.to_row()
            self._runs_log.log(extra={"record": row})
            # T-R2.1: cache GATE-phase rows so a later gate-reuse (the candidate
            # passed and is reused at eval without re-solving) can re-emit this
            # exact row as an eval-phase clone (see :meth:`restamp_reused_gate`).
            if rec.phase == "gate":
                # spinefix: bound the cache to the candidate currently being
                # gated. A gate row for a NEW candidate coordinate proves every
                # prior candidate is fully resolved (sequential child loop:
                # gate→reject OR gate→pass→eval+restamp completes before the next
                # candidate gates), so any rows still cached under a different
                # coordinate are dead — a gate-REJECTED candidate's rows that
                # restamp_reused_gate never pops. Drop them here instead of
                # leaking one entry per rejected task for the whole process. Same
                # coordinate (more tasks / --gate-repeats rows for THIS candidate)
                # accumulates as before, so the restamp path is unchanged.
                coord = (
                    int(row.get("generation", 0) or 0),
                    str(row.get("candidate_id", "") or ""),
                    int(row.get("depth", 1) or 1),
                )
                if coord != self._gate_rows_coord:
                    self._gate_rows.clear()
                    self._gate_rows_coord = coord
                self._gate_rows[rec.run_id] = row
            logger.info(
                "agent run %s task=%s agent=%s score=%.4f terminated_by=%s "
                "cost_usd=%.4f cost_basis=%s wall_s=%.1f",
                rec.run_id, rec.task_id, rec.agent, rec.score,
                rec.terminated_by, rec.cost_usd, rec.cost_basis, rec.wall_s,
            )
        except Exception as e:  # noqa: BLE001 - a lost line must not sink a run
            logger.warning("failed to append agent run record: %r", e)

    def _rel_to_output(self, p: object) -> Optional[str]:
        """Best-effort relpath of ``p`` from ``output_dir`` for a pointer field.

        The JSONL row carries POINTERS, not blobs (§7.7): native logs land under
        ``archive/.../agent_logs/`` and the row stores only a relpath from the run
        root. Returns ``None`` for a falsy / non-relativizable path (e.g. an
        absolute artifacts dir outside ``output_dir``, or a missing one), so the
        pointer is honestly absent rather than a misleading absolute path. Never
        raises — a pointer is diagnostic, not load-bearing for the run.
        """
        if not p:
            return None
        try:
            return str(Path(str(p)).relative_to(self.output_dir))
        except (ValueError, OSError, TypeError):
            return None

    def dereap_agent_logs(
        self,
        rec: AgentRunRecord,
        source_logs_dir: object,
        candidate_id: object,
        run_id: object,
        output_dir: object = None,
    ) -> Optional[str]:
        """Snapshot a lease's ``agent_logs`` into the archive before rmtree (S0.6).

        The external-agent spine runs each task inside a lease whose ``workdir``
        is a scratch ``mkdtemp`` OUTSIDE ``output_dir`` (concurrency.py), and that
        whole dir is ``shutil.rmtree``'d the instant the lease releases. So the
        backend's native ``agent_logs`` (the OpenHands event stream, the T2
        transcript) never survive the run, and ``transcript_ptr`` —
        ``_rel_to_output`` of a path outside ``output_dir`` — comes back ``None``
        (the FEAL null-transcript finding). This copies the logs into the durable
        ``<output_dir>/archive/<candidate_id>/agent_logs/<run_id>/`` and repoints
        ``rec.transcript_ptr``/``agent_logs_ptr`` at the copy, so the pointer
        RESOLVES after the lease is gone.

        Honest ``None``: a backend that wrote no logs (the source dir is missing
        or empty) leaves the pointers UNCHANGED — a backend without a captured
        stream keeps an honest ``None`` pointer rather than a copied-empty-dir
        relpath. **Best-effort / never-raises** (stdlib ``shutil`` only): a failed
        copy logs a warning and leaves the pointers as ``finish_record`` set them.

        Args:
            rec: The (already-finished) :class:`AgentRunRecord` whose pointers are
                repointed at the durable copy.
            source_logs_dir: The lease's ``agent_logs`` dir
                (``lease.workdir / "agent_logs"``) to snapshot.
            candidate_id: The archived candidate id (the ``archive/<id>/`` subtree).
            run_id: The run id (the per-run leaf dir under ``agent_logs/``).
            output_dir: Run-root override; defaults to ``self.output_dir`` (the
                single source of truth for the relpath pointers).

        Returns:
            The new archive-relative pointer (relpath from ``output_dir``) on a
            successful snapshot, else ``None`` (no logs / failure) — and in that
            ``None`` case ``rec``'s pointers are left untouched.
        """
        try:
            root = Path(str(output_dir)) if output_dir else self.output_dir
            src = Path(str(source_logs_dir))
            # No-logs backend (missing or empty dir) -> honest None: do not copy an
            # empty tree, do not rewrite the pointer.
            if not src.is_dir() or not any(src.iterdir()):
                return None

            dest = (
                root
                / "archive"
                / str(candidate_id or "")
                / "agent_logs"
                / str(run_id or "")
            )
            shutil.copytree(src, dest, dirs_exist_ok=True)

            # Repoint at the durable copy with an output_dir-relative pointer
            # (parity ``_rel_to_output``). If the dest is somehow not under the
            # root (custom override), keep the pointers as-is rather than write a
            # misleading absolute path.
            try:
                rel = str(dest.relative_to(root))
            except (ValueError, OSError, TypeError):
                return None
            rec.agent_logs_ptr = rel
            rec.transcript_ptr = rel
            logger.debug(
                "de-reaped agent_logs run_id=%s -> %s", rec.run_id, rel
            )
            return rel
        except Exception as e:  # noqa: BLE001 - de-reap must never sink a run
            logger.warning(
                "agent_logs de-reap failed (run_id=%s candidate=%s): %r",
                getattr(rec, "run_id", "?"),
                candidate_id,
                e,
            )
            return None

    @staticmethod
    def _term_value(terminated_by: object) -> str:
        """Coerce a :class:`TerminatedBy` (or raw string) to its serialized value."""
        if isinstance(terminated_by, TerminatedBy):
            return terminated_by.value
        if terminated_by is None:
            return TerminatedBy.UNKNOWN.value
        return str(terminated_by)
