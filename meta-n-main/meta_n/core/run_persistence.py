"""Run persistence collaborator for the evolutionary orchestrator.

Extracted verbatim from ``EvolutionaryOrchestrator`` (F253): checkpoint /
resume, the incremental summary + candidate writers, the run-config dump,
the final ``save_results`` writer, and the external-agent telemetry rollups.
The on-disk artifact shapes (checkpoint.json / summary.json / per-candidate
summary.json / oracle_summary.json / lineage) are byte-for-byte contracts
asserted by ``test_persistence_compat.py`` and ``tests/golden/`` — do not
reorder keys or rename fields here.

Seam design: :class:`RunPersistence` holds a back-reference to the
orchestrator (mechanical fidelity over purity — ``try_resume`` must REPLACE
``orchestrator.archive`` / ``orchestrator.rng`` in place, and the rollup /
writer cross-calls must route through the orchestrator's delegating methods
so tests that patch those methods on the instance keep intercepting). All
run()-loop state (iteration, patience, convergence/oracle histories, result)
flows in explicitly as method arguments, exactly as before the extraction.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from meta_n.core.archive import Archive, Candidate
from meta_n.core.meta_layer import detect_script_language
from meta_n.core.self_repair import write_self_repair_sidecars
from meta_n.utils.atomic_io import atomic_json_dump

if TYPE_CHECKING:  # pragma: no cover — import cycle guard (orchestrator imports us)
    from meta_n.core.evolutionary_orchestrator import (
        EvolutionaryOrchestrator,
        EvolutionaryResult,
    )

logger = logging.getLogger(__name__)
console = Console()


class RunPersistence:
    """Checkpoint / summary / telemetry-rollup writers for one orchestrator run.

    Constructed with the owning :class:`EvolutionaryOrchestrator`; reads its
    ``config`` / ``archive`` / ``rng`` / ``_tasks`` through the back-reference
    (and mutates ``archive`` / ``rng`` on resume). Cross-calls between the
    extracted methods go through the orchestrator's one-line delegates
    (``self._orch._read_agent_run_rows()`` etc.) so instance-level patches on
    the orchestrator keep working.
    """

    def __init__(self, orchestrator: "EvolutionaryOrchestrator"):
        self._orch = orchestrator

    def try_resume(self, out_dir: Path) -> dict | None:
        """Attempt to load checkpoint from a previous run.

        Returns checkpoint dict if found and valid, None otherwise.
        """
        checkpoint_path = out_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)

            # Rebuild archive from disk (same kwarg bundle as construction so
            # resume restores an identically-configured archive). The fired
            # verified-code bars ride along from the checkpoint so a barred
            # trace cannot become per-task-best in the rebuilt index (forensic
            # #2 sibling of the base-floor re-application in run()).
            archive_dir = out_dir / "archive"
            self._orch.archive = Archive.rebuild_from_disk(
                archive_dir,
                barred_from_best=checkpoint.get("barred_from_best"),
                **self._orch._archive_kwargs(),
            )
            if len(self._orch.archive) == 0:
                logger.warning("Archive is empty after rebuild — cannot resume")
                console.print("[yellow]Archive is empty after rebuild — starting fresh[/yellow]")
                self._orch.archive = Archive(**self._orch._archive_kwargs())
                return None

            # Restore RNG state: convert lists back to tuples. TWO streams as of
            # 2026-09-23 (outer search + context sampling); see the orchestrator's
            # two-stream note. A checkpoint written before the split carries no
            # `context_rng_state`, in which case the context stream restarts from
            # its derived seed -- the run is still deterministic, it just does not
            # continue the pre-split trace sequence (which no longer exists).
            rng_state = checkpoint.get("rng_state")
            ctx_state = checkpoint.get("context_rng_state")
            try:
                if rng_state is not None and len(rng_state) >= 3:
                    self._orch.rng.setstate((
                        rng_state[0],
                        tuple(rng_state[1]),
                        rng_state[2],
                    ))
            except (ValueError, TypeError) as e:
                logger.warning("Failed to restore RNG state: %s — using fresh RNG", e)
                self._reset_rng_streams()
            try:
                if ctx_state is not None and len(ctx_state) >= 3:
                    self._orch.context_rng.setstate((
                        ctx_state[0],
                        tuple(ctx_state[1]),
                        ctx_state[2],
                    ))
                elif rng_state is not None:
                    logger.warning(
                        "Checkpoint has no context_rng_state (written before the "
                        "two-stream split) — restarting the trace-sampling stream "
                        "from its derived seed. The run stays deterministic; it "
                        "just does not continue the pre-split trace sequence.")
            except (ValueError, TypeError) as e:
                logger.warning("Failed to restore context RNG state: %s — "
                               "using fresh context RNG", e)
                self._reset_rng_streams()

            logger.info(
                "Resumed from checkpoint: iteration=%d, archive=%d candidates, "
                "best=%.3f, tokens=%d",
                checkpoint["iteration"], len(self._orch.archive),
                self._orch.archive.best_mean_score, checkpoint["total_tokens"],
            )
            console.print(
                f"[bold green]Resumed from checkpoint: "
                f"iteration {checkpoint['iteration']}, "
                f"{len(self._orch.archive)} candidates in archive, "
                f"best={self._orch.archive.best_mean_score:.3f}[/bold green]"
            )
            return checkpoint
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s — starting fresh", e)
            console.print(f"[yellow]Failed to load checkpoint: {e} — starting fresh[/yellow]")
            # Reset archive in case it was partially rebuilt before the error
            self._orch.archive = Archive(**self._orch._archive_kwargs())
            self._reset_rng_streams()
            return None

    def _reset_rng_streams(self) -> None:
        """Rebuild BOTH rng streams from the configured seed, re-thread them.

        Used by the resume fallbacks. The context stream must be re-assigned to
        the engine's manager too: after `install`, `omega.context_manager` is the
        adapter, whose `rng` property forwards to its base -- so this single
        assignment reaches the sampler the arm actually uses. A local import
        keeps `run_persistence` free of a module-level cycle with the
        orchestrator, which imports this module.
        """
        from meta_n.core.evolutionary_orchestrator import _context_rng_seed
        seed = self._orch.config.seed
        self._orch.rng = random.Random(seed)
        self._orch.context_rng = random.Random(_context_rng_seed(seed))
        self._orch.omega.context_manager.rng = self._orch.context_rng

    def save_checkpoint(
        self,
        out_dir: Path,
        iteration: int,
        patience_counter: int,
        prev_best: float,
        total_tokens: int,
        convergence_history: list[float],
        prev_oracle: float = 0.0,
        oracle_history: list[float] | None = None,
        result: "EvolutionaryResult | None" = None,
    ):
        """Save orchestrator state for resume capability."""
        # Snapshot cumulative outer-LLM usage so resume can restore the
        # prompt/completion split. Without this, ``LLMClient.cumulative_usage``
        # restarts at zero on resume and the final summary's
        # ``token_usage.outer_prompt`` / ``outer_completion`` fields only
        # cover the post-resume slice — even though ``total_tokens`` itself
        # is correctly restored from this same checkpoint.
        outer_usage = {
            "prompt": 0, "completion": 0, "total": 0, "calls": 0,
            "cached": 0, "cost_usd": 0.0,
        }
        if getattr(self._orch, "llm_client", None) is not None:
            cu = getattr(self._orch.llm_client, "cumulative_usage", None) or {}
            outer_usage = {
                "prompt": int(cu.get("prompt", 0) or 0),
                "completion": int(cu.get("completion", 0) or 0),
                "total": int(cu.get("total", 0) or 0),
                "calls": int(cu.get("calls", 0) or 0),
                # The resume branch reads these back with .get defaults —
                # without persisting them, post-resume cost_summary.json
                # restarts outer_cached_tokens / outer_cost_usd at zero.
                "cached": int(cu.get("cached", 0) or 0),
                "cost_usd": float(cu.get("cost_usd", 0.0) or 0.0),
            }
            for key in (
                "requests", "failed_calls", "retries", "empty_responses",
                "empty_escalations", "reasoning", "reasoning_reported_calls",
            ):
                # Conditional preserves old/custom test ledgers exactly while
                # current LLMClient instances persist the extended counters.
                if key in cu:
                    outer_usage[key] = int(cu.get(key, 0) or 0)
        checkpoint = {
            "iteration": iteration,
            "patience_counter": patience_counter,
            "prev_best": prev_best,
            "prev_oracle": prev_oracle,
            "oracle_history": oracle_history or [],
            "frozen_score_range": self._orch.archive.frozen_score_range,
            "total_tokens": total_tokens,
            "convergence_history": convergence_history,
            "rng_state": list(self._orch.rng.getstate()),
            # The context (trace-sampling) stream is a SEPARATE object as of
            # 2026-09-23 -- see the two-stream note in the orchestrator __init__.
            # Both must survive --resume, or a resumed arm would sample traces
            # from a stream the uninterrupted arm never had.
            "context_rng_state": list(self._orch.context_rng.getstate()),
            # Audit #1: persist the INNER-LLM token accounting so --resume can
            # restore it (read back with .get defaults in the resume branch).
            # Without this, token_usage.inner_* in summary.json covers only the
            # post-resume slice. ``result`` is passed by every live call site;
            # default 0 keeps a no-result caller safe.
            "inner_tokens": int(result.inner_tokens) if result is not None else 0,
            "inner_prompt_tokens": (
                int(result.inner_prompt_tokens) if result is not None else 0
            ),
            "inner_completion_tokens": (
                int(result.inner_completion_tokens) if result is not None else 0
            ),
            "inner_calls": int(result.inner_calls) if result is not None else 0,
            "use_agentic": self._orch.config.use_agentic,
            "outer_cumulative_usage": outer_usage,
        }
        # Resume provenance: stamp the CURRENT process's run_config (stashed by
        # run() as ``_run_config_snapshot``) so a later --resume can detect
        # config drift — e.g. a switched/edited --benchmark-config YAML.
        # "timestamp" is per-process volatile and excluded so a drift-free
        # resume compares clean; everything else in run_config is JSON-safe
        # (it is json.dump'ed to config.json today). Additive + .get-read:
        # old checkpoints simply lack the key, and direct callers that never
        # pass a run_config (unit tests) write no key at all.
        snap = getattr(self._orch, "_run_config_snapshot", None)
        if snap:
            checkpoint["run_config_snapshot"] = {
                k: v for k, v in snap.items() if k != "timestamp"
            }
        # Forensic improvement #1 — persist the base floor + denoised focus scores
        # ONLY when the guard is ON, so checkpoint.json is byte-identical to HEAD
        # when OFF (the key is simply absent). Restored in the resume branch above.
        if self._orch.config.regression_guard:
            checkpoint["base_floor"] = {
                tid: {
                    "score": e[0],
                    "candidate_id": e[1],
                    "trace": (e[2].model_dump(mode="json") if e[2] is not None else None),
                }
                for tid, e in self._orch.archive.base_floor_snapshot().items()
            }
            checkpoint["base_focus_scores"] = dict(self._orch._base_focus_scores)
        # Forensic improvement #2 — persist the fired verified-code bars so
        # try_resume's rebuild re-applies them (rebuild_from_disk replays add()
        # with bar_from_best). Absent when no bar ever fired, so fresh and
        # bar-free checkpoints stay byte-identical.
        barred = self._orch.archive.barred_from_best_snapshot()
        if barred:
            checkpoint["barred_from_best"] = barred
        # RNG state is (version, internalstate_tuple, gauss_next) — convert tuples to lists
        checkpoint["rng_state"][1] = list(checkpoint["rng_state"][1])
        checkpoint["context_rng_state"][1] = list(
            checkpoint["context_rng_state"][1])
        try:
            atomic_json_dump(out_dir / "checkpoint.json", checkpoint)
        except Exception:
            # warning, not debug: a silent checkpoint failure (e.g. disk
            # full) means a later --resume reads stale state and re-runs
            # candidates the user thought were saved.
            logger.warning(
                "Failed to write checkpoint at iteration %d (non-fatal, but "
                "--resume will read stale state)",
                iteration,
                exc_info=True,
            )

    def save_running_summary(
        self,
        out_dir: Path,
        iteration: int,
        result: "EvolutionaryResult",
        run_start: float,
    ):
        """Write convergence/oracle files + the MID-RUN summary.json shape atomically.

        DUAL-SCHEMA CONTRACT (F038 — load-bearing, do not unify casually):
        summary.json has TWO deliberate shapes.
          * MID-RUN (this writer): {iteration, archive_size, best_mean_score,
            oracle_mean_score, best_candidate_id, per_task_best_scores,
            total_tokens, elapsed_s}. Left behind by a crash or a hard
            BudgetExceededError kill (main.py deliberately skips save_results,
            so a budget-killed run keeps this persistent mid-run shape).
          * FINAL (save_results → EvolutionaryResult.to_dict): adds
            total_iterations, token_usage, run_status, ... and overwrites this
            file.
        The DISCRIMINATOR is key presence: consumers treat a summary.json WITHOUT
        'total_iterations' as "run not completed" (scripts/azure_full_*.sh,
        run_tb2_single_shot_all.sh). Therefore this mid-run shape must NEVER gain
        'total_iterations' or 'run_status' (and the final shape must never lose
        them) without auditing every presence-based consumer. Pinned by
        tests/test_r6b_run_persistence.py; the FINAL shape is frozen by
        tests/test_persistence_compat.py::FROZEN_SUMMARY_KEYS.
        """
        try:
            atomic_json_dump(out_dir / "convergence.json", result.convergence_history)
            atomic_json_dump(out_dir / "oracle_convergence.json", result.oracle_history)

            ptb = self._orch.archive.per_task_best_scores()
            tasks = getattr(self._orch, "_tasks", None) or []
            ptb_mean = (
                sum(ptb.get(t.task_id, 0.0) for t in tasks) / len(tasks)
                if tasks else (sum(ptb.values()) / len(ptb) if ptb else 0.0)
            )
            running_summary = {
                "iteration": iteration,
                "archive_size": len(self._orch.archive),
                "best_mean_score": self._orch.archive.best_mean_score,
                "oracle_mean_score": ptb_mean,
                "best_candidate_id": (
                    self._orch.archive.best_candidate.candidate_id
                    if self._orch.archive.best_candidate else ""
                ),
                "per_task_best_scores": ptb,
                "total_tokens": result.total_tokens,
                "elapsed_s": time.time() - run_start,
            }
            atomic_json_dump(out_dir / "summary.json", running_summary)
        except Exception:
            logger.warning(
                "Failed to write incremental summary at iteration %d "
                "(non-fatal, but the on-disk summary.json is now stale)",
                iteration,
                exc_info=True,
            )

    def save_candidate_incremental(self, candidate: Candidate, out_dir: Path):
        """Save a single candidate's data immediately after evaluation."""
        archive_dir = out_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        cand_dir = archive_dir / candidate.candidate_id
        cand_dir.mkdir(exist_ok=True)

        # Summary
        cand_summary = {
            "candidate_id": candidate.candidate_id,
            "parent_id": candidate.parent_id,
            "iteration": candidate.iteration,
            "depth": candidate.depth,
            "mean_score": candidate.mean_score,
            "pass_at_1": candidate.pass_at_1,
            "per_task_scores": candidate.per_task_scores,
            "num_children": candidate.num_children,
            "temperature_used": candidate.temperature_used,
            "total_tokens": candidate.total_tokens,
            "created_at": candidate.created_at,
        }
        # External-agent telemetry rollup (plan §7.8): re-read agent_runs.jsonl
        # filtered by candidate_id (source of truth is the flock-appended JSONL,
        # not an in-process buffer). Absent for legacy / builtin runs so the
        # existing summary.json stays byte-for-byte the same.
        rollup = self._orch._candidate_agent_telemetry_rollup(candidate.candidate_id)
        if rollup is not None:
            cand_summary["agent_telemetry"] = rollup
        # S0.3: failure-class / parse-failure rollups over this candidate's
        # traces. GATED behind non-empty checks (mirroring the `if rollup is not
        # None` guard above) so the EMPTY path — a clean run with no classified
        # failures and no parse-failure turns — keeps a byte-identical
        # summary.json (the new keys are absent, not empty).
        failure_classes = [t.failure_class for t in candidate.traces if t.failure_class]
        if failure_classes:
            dist: dict[str, int] = {}
            for fc in failure_classes:
                dist[fc] = dist.get(fc, 0) + 1
            cand_summary["failure_class_distribution"] = dist
        parse_failure_total = sum(
            int(getattr(t, "parse_failure_turns", 0) or 0) for t in candidate.traces
        )
        if parse_failure_total > 0:
            n_traces = len(candidate.traces) or 1
            # Fraction of solves that wasted >=1 turn on a non-actionable parse
            # failure (phantom-completion / parse error / no-code).
            cand_summary["parse_failure_rate"] = (
                sum(1 for t in candidate.traces
                    if int(getattr(t, "parse_failure_turns", 0) or 0) > 0)
                / n_traces
            )
        # Cost integrity: per-candidate INNER-LLM accounting must survive resume
        # (rebuild_from_disk reads summary.json). Gated on non-zero so a run with
        # no inner-LLM usage keeps a byte-identical summary.json (keys absent,
        # not zero) — mirrors the agent_telemetry / failure-class gating above.
        if (candidate.inner_tokens or candidate.inner_prompt_tokens
                or candidate.inner_completion_tokens or candidate.inner_calls):
            cand_summary["inner_tokens"] = candidate.inner_tokens
            cand_summary["inner_prompt_tokens"] = candidate.inner_prompt_tokens
            cand_summary["inner_completion_tokens"] = candidate.inner_completion_tokens
            cand_summary["inner_calls"] = candidate.inner_calls
        # verified_code bars are recorded in the live archive at add() time and
        # persisted in checkpoint.json, but a crash between THIS candidate write
        # and the next checkpoint would leave the candidate dir on disk while the
        # atomic prior checkpoint lacks its bar — so rebuild_from_disk would
        # resurrect the barred trace as per-task-best (and completed_set skips
        # re-breeding it, making the loss permanent). Co-locate the fired bar
        # with the candidate so rebuild reconstructs it from an authoritative,
        # always-in-sync source. Gated on non-empty -> verified_code-OFF /
        # bar-free runs keep a byte-identical summary.json.
        try:
            cand_bar = self._orch.archive.barred_from_best_snapshot().get(
                candidate.candidate_id
            )
        except (AttributeError, TypeError):
            cand_bar = None
        if cand_bar and isinstance(cand_bar, list):
            cand_summary["barred_from_best_task_ids"] = cand_bar
        with open(cand_dir / "summary.json", "w") as f:
            json.dump(cand_summary, f, indent=2)

        # Traces
        traces_dir = cand_dir / "traces"
        traces_dir.mkdir(exist_ok=True)
        for trace in candidate.traces:
            with open(traces_dir / f"{trace.task_id}.json", "w") as f:
                json.dump(trace.model_dump(), f, indent=2)
            ext = ".py" if detect_script_language(trace.script) == "python" else ".sh"
            with open(traces_dir / f"{trace.task_id}{ext}", "w") as f:
                f.write(trace.script)

        # Injected codes (full chain — always save JSON for consistent rebuild).
        # raw_omega_prompt + raw_omega_response are excluded from JSON to keep
        # injected_code_d{N}.json compact; they're persisted as paired sidecar
        # .txt files instead so a reviewer can diff prompt ↔ response easily.
        for i, ic in enumerate(candidate.injected_codes):
            with open(cand_dir / f"injected_code_d{i+2}.json", "w") as f:
                json.dump(
                    ic.model_dump(exclude={"raw_omega_prompt", "raw_omega_response"}),
                    f, indent=2,
                )
            if not ic.is_empty:
                if ic.raw_omega_prompt:
                    with open(cand_dir / f"omega_prompt_d{i+2}.txt", "w") as f:
                        f.write(ic.raw_omega_prompt)
                if ic.raw_omega_response:
                    with open(cand_dir / f"omega_response_d{i+2}.txt", "w") as f:
                        f.write(ic.raw_omega_response)
                if ic.pre_process:
                    with open(cand_dir / f"pre_process_d{i+2}.py", "w") as f:
                        f.write(ic.pre_process)
                for name, src in ic.code_library.items():
                    with open(cand_dir / f"solver_lib_{name}_d{i+2}.py", "w") as f:
                        f.write(src)
                for name, src in ic.code_library_bash.items():
                    with open(cand_dir / f"solver_lib_bash_{name}_d{i+2}.sh", "w") as f:
                        f.write(src)

        # Self-repair provenance sidecars (Stage 2 within-layer refine / Stage 3
        # downward re-propagation). GATED behind a non-empty ``events`` check
        # INSIDE ``write_self_repair_sidecars`` (mirroring the ``if rollup is not
        # None`` guard above): every current candidate carries an EMPTY
        # ``self_repair_events`` list, so this writes NOTHING and the on-disk
        # candidate dir is byte-for-byte identical to HEAD — the
        # ``repropagation_d{t}.json`` + paired ``.txt`` appear ONLY for a
        # candidate an opted-in refine/re-propagation actually repaired.
        write_self_repair_sidecars(
            cand_dir, getattr(candidate, "self_repair_events", None)
        )

        # Update running archive index (atomic write to prevent corruption on
        # crash; the helper removes the tmp file and re-raises on failure).
        atomic_json_dump(archive_dir / "index.json", self._orch.archive.to_dict())

    def read_agent_run_rows(self) -> list[dict]:
        """Read every :class:`AgentRunRecord` row from ``telemetry/agent_runs.jsonl``.

        The rows are written through the :class:`LLMIOLogger` flock-append path,
        which nests the record under ``extra.record``; older / forward-compatible
        rows may carry a top-level ``run_id`` instead, so both shapes are
        unwrapped. De-dups on the deterministic ``run_id`` (§7.3) so a resumed
        run's re-executed task contributes a single row. Returns ``[]`` when no
        telemetry tree exists (legacy / builtin runs).

        Returns:
            The de-duplicated list of record dicts (the unwrapped ``extra.record``
            payload), newest-wins on a ``run_id`` collision.
        """
        out_dir = self._orch.config.output_dir
        if not out_dir:
            return []
        path = Path(out_dir) / "telemetry" / "agent_runs.jsonl"
        if not path.exists():
            return []
        # Audit #47: cache the parsed + de-duped rows keyed by (size, mtime_ns).
        # _save_candidate_incremental calls this once per candidate, and the
        # append-only ledger grows ~one row per (candidate, task, repeat); without
        # a cache, re-parsing the whole file each save is O(M^2) over a run. The
        # file only ever grows, so a (size, mtime) match means the parse is still
        # valid and we can reuse it. Callers only read the returned rows.
        try:
            st = path.stat()
            cache_key = (st.st_size, st.st_mtime_ns)
        except OSError:
            cache_key = None
        cached = getattr(self._orch, "_agent_rows_cache", None)
        if cache_key is not None and cached is not None and cached[0] == cache_key:
            return cached[1]
        # Lazy import: the spine's telemetry module is dependency-free (no SDK)
        # but the legacy path should not pay its import cost at module load.
        from meta_n.core.external_agents.telemetry import iter_jsonl_objects

        by_run_id: dict[str, dict] = {}
        unkeyed: list[dict] = []
        malformed = 0
        try:
            for obj in iter_jsonl_objects(path):
                # Mirror writer._load_run_id_index's envelope unwrap: a
                # non-dict ``extra`` falls back to the top level; a PRESENT but
                # null/non-dict envelope ``record`` is a malformed line — skip
                # it (never-raise contract; the except below is OSError-only,
                # so an AttributeError here would kill the candidate save).
                extra = obj.get("extra")
                env_rec = extra.get("record", {}) if isinstance(extra, dict) else {}
                if not isinstance(env_rec, dict):
                    malformed += 1
                    continue
                rec = env_rec or obj
                rid = rec.get("run_id")
                if rid:
                    by_run_id[rid] = rec  # newest-wins de-dup
                else:
                    unkeyed.append(rec)
        except OSError:
            logger.warning(
                "Failed reading agent telemetry rows from %s", path, exc_info=True
            )
            return []
        if malformed:
            logger.warning(
                "Skipped %d malformed envelope line(s) reading agent telemetry "
                "rows from %s", malformed, path,
            )
        rows = list(by_run_id.values()) + unkeyed
        if cache_key is not None:
            self._orch._agent_rows_cache = (cache_key, rows)
        return rows

    @staticmethod
    def aggregate_agent_rows(rows: list[dict]) -> dict:
        """Shared core of the candidate / run-level telemetry rollups.

        Token sums are split by ``token_basis`` so a builtin "outer" row is
        never blended with an OH/T2 "inner" row under one header (§7.11).
        Callers assemble their OWN final dict in their frozen key order (the
        rollups are persisted into summary.json, where insertion order is part
        of the byte contract) — never return this dict directly.
        """
        terminated_by: dict[str, int] = {}
        for r in rows:
            key = str(r.get("terminated_by", "") or "unknown")
            terminated_by[key] = terminated_by.get(key, 0) + 1
        return {
            "runs": len(rows),
            "successes": sum(1 for r in rows if r.get("success")),
            "inner_total_tokens": sum(
                int(r.get("total_tokens", 0) or 0)
                for r in rows if r.get("token_basis") == "inner"
            ),
            "outer_total_tokens": sum(
                int(r.get("total_tokens", 0) or 0)
                for r in rows if r.get("token_basis") == "outer"
            ),
            "cost_usd": sum(float(r.get("cost_usd", 0.0) or 0.0) for r in rows),
            "terminated_by": terminated_by,
        }

    def candidate_agent_telemetry_rollup(self, candidate_id: str) -> dict | None:
        """Roll up the external-agent telemetry rows for one candidate (§7.8).

        Re-reads ``telemetry/agent_runs.jsonl`` (the source of truth — not an
        in-process buffer that must survive across async tasks) and aggregates
        the rows whose ``candidate_id`` matches. Returns ``None`` when no external
        agent ran for this candidate (legacy / builtin), so the per-candidate
        ``summary.json`` stays byte-for-byte the same on the legacy path.

        Args:
            candidate_id: The candidate chain id to filter on.

        Returns:
            A flat rollup dict (run/token/cost/score/terminated_by counts), or
            ``None`` if there are no matching rows.
        """
        rows = [
            r for r in self._orch._read_agent_run_rows()
            if str(r.get("candidate_id", "")) == str(candidate_id)
        ]
        if not rows:
            return None

        core = self._orch._aggregate_agent_rows(rows)
        n = core["runs"]
        scores = [float(r.get("score", 0.0) or 0.0) for r in rows]
        agents = sorted({str(r.get("agent", "") or "") for r in rows if r.get("agent")})
        cost_bases = sorted({str(r.get("cost_basis", "") or "") for r in rows if r.get("cost_basis")})

        # Key order is frozen (persisted into per-candidate summary.json).
        return {
            "runs": core["runs"],
            "agents": agents,
            "successes": core["successes"],
            "mean_score": (sum(scores) / n) if n else 0.0,
            "inner_total_tokens": core["inner_total_tokens"],
            "outer_total_tokens": core["outer_total_tokens"],
            "cost_usd": core["cost_usd"],
            "cost_basis": cost_bases,
            "terminated_by": core["terminated_by"],
        }

    def run_level_agent_telemetry_rollup(self) -> dict | None:
        """Aggregate the external-agent telemetry across the whole run (§7.8).

        Re-reads the de-duplicated ``agent_runs.jsonl`` rows and produces the
        run-level ``agent_telemetry_rollup`` block surfaced in the top-level
        ``summary.json`` / :meth:`EvolutionaryResult.to_dict`. Returns ``None``
        when no external agent ran (legacy / builtin), so the existing summary
        stays unchanged.

        Returns:
            The run-level rollup dict, or ``None`` if there are no agent rows.
        """
        rows = self._orch._read_agent_run_rows()
        if not rows:
            return None
        core = self._orch._aggregate_agent_rows(rows)
        by_agent: dict[str, int] = {}
        for r in rows:
            agent = str(r.get("agent", "") or "")
            by_agent[agent] = by_agent.get(agent, 0) + 1
        # Key order is frozen (persisted into the run-level summary.json).
        return {
            "runs": core["runs"],
            "runs_by_agent": by_agent,
            "successes": core["successes"],
            "inner_total_tokens": core["inner_total_tokens"],
            "outer_total_tokens": core["outer_total_tokens"],
            "cost_usd": core["cost_usd"],
            "terminated_by": core["terminated_by"],
        }

    def write_run_config(
        self,
        out_dir: Path | str,
        run_config: dict | None,
        *,
        stage: str = "end",
    ) -> None:
        """Dump ``run_config`` to ``<out_dir>/config.json``.

        Shared by the START dump in ``run()`` and the END dump in
        ``save_results()`` so both write identical content. No-op when
        ``run_config`` is falsy.

        ``stage="start"`` (the early dump) MERGE-PRESERVES any keys an existing
        config.json already carries that the START payload lacks, so a START
        re-write (e.g. on resume) can never clobber the richer fields a prior
        END dump recorded. ``stage="end"`` overwrites with the full payload
        (the end-of-run config is authoritative)."""
        if not run_config:
            return
        cfg_path = Path(out_dir) / "config.json"
        payload = dict(run_config)
        if stage == "start" and cfg_path.exists():
            try:
                existing = json.loads(cfg_path.read_text())
                if isinstance(existing, dict):
                    # Existing-only (END-only) keys survive; START values refresh
                    # the keys they share.
                    merged = dict(existing)
                    merged.update(payload)
                    payload = merged
            except (OSError, ValueError):  # ValueError covers JSONDecodeError
                pass
        # Resume provenance: when a resume detected config drift (stashed by
        # the resume branch as ``_resume_config_drift``), record BOTH sides —
        # {key: {"checkpoint": old, "resumed": new}} — under one additive key.
        # The top-level keys keep the RESUMED values (they are what the
        # continued generations actually ran under); the drift record is the
        # honest trail. Absent on fresh runs and drift-free resumes, so those
        # config.json files stay byte-identical. A second drifted resume
        # overwrites with its latest drift (last-drift-wins: the checkpoint
        # snapshot is refreshed every iteration to the current process's
        # config, which is the correct comparison base). Appended AFTER the
        # run_config keys so the END dump's frozen key order is preserved.
        drift = getattr(self._orch, "_resume_config_drift", None)
        if drift:
            payload["resume_config_drift"] = drift
        with open(cfg_path, "w") as f:
            json.dump(payload, f, indent=2)

    def save_results(
        self,
        result: "EvolutionaryResult",
        output_dir: str | None = None,
        run_config: dict | None = None,
    ):
        """Save final results. Candidates are already saved incrementally to
        self.config.output_dir during run(). This method saves summary, convergence,
        per-task best, and lineage to the same directory.

        Overwrites the mid-run summary.json shape with the FINAL shape — see the
        dual-schema contract on :meth:`save_running_summary` (F038).
        """
        out_dir = Path(output_dir or self._orch.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Config (END dump — authoritative, full overwrite). Mirrors the START
        # dump in run(); both go through _write_run_config so the content is
        # byte-identical when run_config is unchanged.
        self._orch._write_run_config(out_dir, run_config, stage="end")

        # Summary
        summary_payload = result.to_dict()
        # T2.1 (audit): record whether --paired-eval actually took effect. Only
        # surfaced when the flag was requested (default OFF ⇒ key absent ⇒ legacy
        # summary.json byte-for-byte unchanged).
        if self._orch.config.paired_eval:
            summary_payload["paired_eval_effective"] = self._orch._paired_eval_effective
        # Atomic like the mid-run writers of the SAME filenames, so a crash
        # during the final save can't leave a truncated summary/convergence.
        atomic_json_dump(out_dir / "summary.json", summary_payload)

        # Convergence history
        atomic_json_dump(out_dir / "convergence.json", result.convergence_history)
        # N7: oracle (per-task-best mean) trajectory — sibling of convergence.json
        atomic_json_dump(out_dir / "oracle_convergence.json", result.oracle_history)

        # Per-task best
        best_dir = out_dir / "per_task_best"
        best_dir.mkdir(exist_ok=True)
        best_traces = self._orch.archive.per_task_best_traces()
        for task_id, trace in best_traces.items():
            with open(best_dir / f"{task_id}.json", "w") as f:
                json.dump(trace.model_dump(), f, indent=2)
            ext = ".py" if detect_script_language(trace.script) == "python" else ".sh"
            with open(best_dir / f"{task_id}{ext}", "w") as f:
                f.write(trace.script)

        # Final archive index (with updated num_children counts)
        archive_dir = out_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        atomic_json_dump(archive_dir / "index.json", result.archive_data)

        # Oracle summary
        oracle_scores = self._orch.archive.per_task_best_scores()
        # Audit #17: average over the FULL task set with missing tasks as 0.0 —
        # identical to the authoritative run() computation (result.oracle_mean_score)
        # and _save_running_summary, so oracle_summary.json agrees with summary.json
        # on the shared ``oracle_mean_score`` key. per_task_best_scores() only
        # contains tasks with a finite-scored candidate, so the bare-subset
        # denominator inflated the oracle whenever any task lacked one.
        _oracle_tasks = getattr(self._orch, "_tasks", None)
        if _oracle_tasks:
            oracle_mean = (
                sum(oracle_scores.get(t.task_id, 0.0) for t in _oracle_tasks)
                / len(_oracle_tasks)
            )
        else:
            oracle_mean = (
                sum(oracle_scores.values()) / len(oracle_scores)
                if oracle_scores else 0.0
            )
        # oracle_scores / oracle_srcs share the per-task-best index's keys and
        # iteration order, so the per-task map below is byte-identical to the
        # old direct walk over the index.
        oracle_srcs = self._orch.archive.per_task_best_sources()
        sources = set(oracle_srcs.values())
        oracle_summary = {
            "oracle_mean_score": oracle_mean,
            "num_contributing_chains": len(sources),
            "per_task_best_sources": {
                tid: {"score": oracle_scores[tid], "candidate_id": oracle_srcs[tid]}
                for tid in oracle_srcs
            },
        }
        with open(out_dir / "oracle_summary.json", "w") as f:
            json.dump(oracle_summary, f, indent=2)

        # Test results (oracle per-task best)
        if result.test_scores:
            with open(out_dir / "test_results.json", "w") as f:
                json.dump({
                    "test_mean_score": result.test_mean_score,
                    "test_scores": result.test_scores,
                }, f, indent=2)

        # Test results (single best chain)
        if result.chain_test_scores:
            with open(out_dir / "chain_test_results.json", "w") as f:
                json.dump({
                    "chain_test_mean_score": result.chain_test_mean_score,
                    "chain_test_scores": result.chain_test_scores,
                    "candidate_id": result.best_candidate_id,
                }, f, indent=2)

        # Best candidate lineage
        lineage_dir = out_dir / "lineage"
        lineage_dir.mkdir(exist_ok=True)
        best = self._orch.archive.best_candidate
        if best:
            lineage = {
                "best_candidate_id": best.candidate_id,
                "depth": best.depth,
                "mean_score": best.mean_score,
                "injected_codes": [
                    ic.model_dump(exclude={"raw_omega_prompt", "raw_omega_response"})
                    for ic in best.injected_codes
                ],
            }
            chain = []
            cand = best
            while cand:
                chain.append(cand.candidate_id)
                cand = self._orch.archive.find(cand.parent_id)
            lineage["ancestry"] = list(reversed(chain))
            with open(lineage_dir / "best_chain.json", "w") as f:
                json.dump(lineage, f, indent=2)

        console.print(f"\n[green]Results saved to {out_dir}/[/green]")
        console.print(f"  summary.json           — scores, iterations, convergence")
        console.print(f"  oracle_summary.json    — per-task best with source chains")
        if result.test_scores:
            console.print(f"  test_results.json      — oracle test set scores")
        if result.chain_test_scores:
            console.print(f"  chain_test_results.json — single-chain test set scores")
        console.print(f"  per_task_best/         — best solution per task across archive")
        console.print(f"  archive/               — all candidates with traces")
        console.print(f"  lineage/best_chain.json — ancestry of the best candidate")
        console.print(f"  run.log                — detailed debug log")
        logger.info("Results saved to %s", out_dir)
