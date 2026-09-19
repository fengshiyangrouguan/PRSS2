"""Cost guard for external-agent runs — one thin adapter onto one ``CostTracker``.

This is WAVE 2 of the external-agents integration (plan §4.7). It owns a single
class, :class:`CostGuard`, that bridges the heterogeneous cost reporting of the
external agents onto meta-n's *one* spend ledger
(:class:`meta_n.utils.cost_tracker.CostTracker`):

* **OpenHands** reports a *native* USD figure (litellm's ``accumulated_cost``)
  because its LLM calls go through its own litellm client, never meta-n's
  ``LLMClient``. That USD is appended to the ledger verbatim via
  :meth:`CostTracker.record_usd` (``cost_basis="native_usd"``).
* **Terminus 2** has no native cost; it reports only token counts. Those are
  priced with the *same* :func:`compute_cost_usd` the ledger itself uses, then
  appended via :meth:`CostTracker.record_usd` so OH and T2 share one append path
  and one ``today_total_usd()`` basis (``cost_basis="priced_from_tokens"``).
* **builtin** spend already flows through ``LLMClient``/``CostTracker.record``, so
  it does not pass through this guard at all.

Two policies make the daily cap a true backstop instead of a silent no-op:

#. **Fail-fast model coverage (plan §4.7.2).** For the token-priced (T2) path a
   model absent from ``PRICING``/``META_N_PRICING_OVERRIDE_JSON`` would otherwise
   be *silently zero-priced*, defeating the cap. :class:`CostGuard` resolves
   :func:`get_pricing` **at construction** and raises a clear error naming the
   model and the override escape hatch — never at the first (already-spent) call.
#. **Never-raise pre-check (plan §4.7.3).** :meth:`CostGuard.precheck` reads
   ``today_total_usd()`` directly and returns a *bool* — it never calls
   :meth:`CostTracker.assert_under_cap` (which raises
   :class:`BudgetExceededError`, a ``BaseException``) so a stray budget signal
   cannot escape the spine's ``execute()`` and cancel sibling tasks in the
   evaluation ``gather`` (plan §4.7.4).

How the daily cap is actually enforced (no per-run SDK stop)
-----------------------------------------------------------
The external backends do **not** enforce a per-run USD ceiling: OpenHands stamps
``llm.metrics.max_budget_per_task`` but meta-n does not rely on an in-run SDK stop
(the SDK records-without-enforcing it — verified against 1.28.0), and Terminus 2
never forwards a USD budget at all. So the daily cap
is enforced entirely on the meta-n side by two real gates:

* **Per-task admission re-check** — :meth:`CostGuard.precheck` re-reads
  ``today_total_usd()`` fresh at the start of *every* run (inside the per-task
  semaphore) and denies once the day's headroom is gone.
* **Generation-boundary halt** — the orchestrator calls
  :meth:`CostGuard.headroom_exhausted` at each candidate/iteration boundary and
  stops dispatching further candidates once the day is spent.

Because tasks run concurrently under ``Semaphore(parallel)`` and spend is only
recorded at run end, up to ``parallel`` in-flight runs admitted against the same
pre-record total can still overshoot the cap by at most that many runs; there is
no in-run USD kill to clip an individual over-spending run. This is the honest,
documented bound — not a hard per-run stop.

Ledger undercount bound (the symmetric caveat)
----------------------------------------------
Spend is folded into the ledger at exactly one point (the spine's
``cost_guard.record`` call right after ``backend.run``/``collect_metrics``).
A degraded run whose agent actually STARTED but returned no parseable result
under-counts ``today_total_usd()`` by that run's true spend:
(a) a hard-killed runner (SIGKILL is uncatchable; the OH/T2 runners write
their result JSON only at completion) reaches the fold with $0/0 tokens;
(b) the outer last-resort envelope cancels the run before the fold
(``finish_timeout``); (c) a contract-violating raise from
``backend.run``/``collect_metrics`` loses the result object
(``finish_error``). In all three the true figure is UNKNOWABLE host-side,
so the ledger deliberately records what it can prove ($0) rather than an
estimate — the daily cap therefore bounds PROVEN spend; worst-case true
spend additionally includes up to one un-metered hard-envelope run per
degraded row on these paths (per-run wall bounded by
``hard_timeout(time_limit_s) + 300s``; NOT bounded by ``max_budget_usd``,
which no backend enforces). Paths where no agent ran (budget denial,
injection-build failure, provisioning faults) spend $0 and are correctly
exempt. The principled future fix is SIGTERM-then-SIGKILL escalation with
a runner-side partial-metrics flush; adopting it changes live
subprocess-bridge kill semantics and needs its own sign-off.

This module imports only :mod:`meta_n.utils.cost_tracker` and the standard
library, so the ``external_agents`` package stays importable without
``openhands``, ``terminal_bench`` or ``docker`` installed (no external SDK is
referenced here at all).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

# Reused (never reimplemented) from the single cost ledger — see plan §4.7 and
# the interface manifest §4. The canonical ``BudgetExceededError`` type lives in
# ``meta_n.utils.cost_tracker``; this module never raises or catches it
# (precheck returns a bool by design) and does not re-export it.
from meta_n.utils.cost_tracker import compute_cost_usd, get_pricing

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime import cost
    from meta_n.utils.cost_tracker import CostTracker

    from .backend import AgentRunResult

logger = logging.getLogger(__name__)

__all__ = ["CostGuard"]


class _NonFiniteLedger(ValueError):
    """Internal: ``today_total_usd()`` came back non-finite (corrupted ledger).

    Raised only by :meth:`CostGuard._headroom_parts` so both gate readers can
    branch to their fail-closed decision with the offending value in hand.
    """

    def __init__(self, today: float) -> None:
        super().__init__(f"non-finite today_total_usd(): {today!r}")
        self.today = today


class CostGuard:
    """Thin adapter onto the one :class:`CostTracker` ledger for external agents.

    A :class:`CostGuard` is constructed once per external-agent backend (with the
    backend's resolved inner model id) and is shared across the runs that backend
    drives. It does two jobs:

    * :meth:`precheck` — a soft, never-raising per-candidate aggregate budget
      gate read off ``today_total_usd()`` (plan §4.7.3).
    * :meth:`record` — fold a finished run's real spend into the ledger so it
      lands in ``today_total_usd()`` on its true basis (native USD for OpenHands,
      priced-from-tokens for Terminus 2; plan §4.7.1-2).

    The cap itself (``cap - reservation``) is enforced by the orchestrator's
    generation-boundary :meth:`headroom_exhausted` check plus this guard's own
    per-task :meth:`precheck`; neither external backend enforces a per-run USD
    stop, so an admitted run is not clipped mid-flight. This guard deliberately
    never invokes the raising :meth:`CostTracker.assert_under_cap` (plan §4.7.4).

    Attributes:
        cost_tracker: The single shared :class:`CostTracker` ledger.
        model: The backend's resolved inner model id used for token pricing.
    """

    def _headroom_parts(self) -> "tuple[float, float, float, float]":
        """Read the ledger once → ``(today, cap, reservation, headroom)``.

        The single shared read + arithmetic behind :meth:`headroom_exhausted`
        and :meth:`precheck` (``headroom = (cap - reservation) - today``).
        Raises :class:`_NonFiniteLedger` when ``today_total_usd()`` is
        non-finite (the corrupted-ledger fail-closed case) and propagates any
        other tracker read error; each caller translates both into its own
        logged decision.
        """
        today = float(self.cost_tracker.today_total_usd())
        # Fail CLOSED on a corrupted ledger: a non-finite total (e.g. a NaN
        # persisted by an upstream cost bug) makes every ``>=``/``<=``
        # comparison False, which would silently disable the day cap.
        if not math.isfinite(today):
            raise _NonFiniteLedger(today)
        cap = float(self.cost_tracker.daily_cap_usd)
        reservation = float(self.cost_tracker.reservation_usd)
        return today, cap, reservation, (cap - reservation) - today

    def headroom_exhausted(self) -> bool:
        """``True`` when today's spend has consumed the day's headroom.

        A never-raising read off ``today_total_usd()`` used by the orchestrator
        at each candidate/generation boundary to halt further dispatch once the
        day is spent (``today >= cap - reservation``). A corrupted (non-finite)
        ledger fails CLOSED (``True`` — dispatch halts); a malformed/unreadable
        ledger is treated as "not exhausted" (``False``) so a transient read
        failure never wedges the run; the per-task :meth:`precheck` re-checks
        anyway. Never raises.
        """
        try:
            _today, _cap, _reservation, headroom = self._headroom_parts()
            return headroom <= 0.0
        except _NonFiniteLedger as exc:
            logger.warning(
                "[CostGuard] headroom_exhausted: today_total_usd() is "
                "non-finite (%r); treating the day as exhausted (fail "
                "closed) to keep the daily cap enforced.", exc.today,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - never raise out of the gate
            logger.warning(
                "[CostGuard] headroom_exhausted check failed (treating as not "
                "exhausted; per-task precheck still applies): %s", exc,
            )
            return False

    def __init__(self, cost_tracker: "CostTracker", model: str) -> None:
        """Bind to the shared ledger and fail fast on an unpriceable model.

        For the token-priced (Terminus 2 / builtin-style) path, a model absent
        from ``PRICING``/``META_N_PRICING_OVERRIDE_JSON`` would be silently
        zero-priced at every call, defeating the daily cap. To avoid that, the
        constructor resolves :func:`get_pricing` *now* and lets its ``KeyError``
        (which already names the model and the override escape hatch) surface —
        translated to a clear, attributed ``RuntimeError`` — so the failure
        happens at startup, never after spend has occurred (plan §4.7.2).

        OpenHands records *native* USD and so does not need ``model`` in
        ``PRICING``; the construction-time check is still performed because a
        single :class:`CostGuard` may price T2 token usage, and a missing model
        is a configuration error regardless of which backend ends up calling
        :meth:`record`.

        Args:
            cost_tracker: The shared :class:`CostTracker` ledger. Must not be
                ``None`` — external-agent runs require ``--daily-budget-usd > 0``
                (enforced upstream in the orchestrator, plan §4.7), which is the
                only way a tracker is constructed.
            model: The resolved inner model id (e.g. ``"gpt-5.2"``) used to price
                token-only spend.

        Raises:
            RuntimeError: If ``cost_tracker`` is ``None`` or ``model`` has no
                pricing entry. The message names the model and the
                ``META_N_PRICING_OVERRIDE_JSON`` escape hatch.
        """
        if cost_tracker is None:
            raise RuntimeError(
                "CostGuard requires a CostTracker ledger, got None. "
                "External-agent runs must be launched with --daily-budget-usd > 0 "
                "so agent spend is capped."
            )
        self.cost_tracker = cost_tracker
        self.model = model
        # Fail-fast model-coverage guard (plan §4.7.2): resolve pricing now so an
        # unpriceable model raises at construction, never after the first (already
        # billed) token-priced call. get_pricing()'s KeyError already names the
        # model and the override env var; we re-raise as RuntimeError for a clean
        # startup failure and to keep the never-raise (Base?) surface uniform.
        try:
            get_pricing(model)
        except KeyError as exc:
            raise RuntimeError(
                f"CostGuard cannot price model {model!r}: external-agent token "
                f"usage would be silently zero-priced, defeating the daily cap. "
                f"{exc}"
            ) from exc

    def precheck(self, max_budget_usd: float) -> bool:
        """Soft per-candidate budget gate — ``True`` means *deny this run*.

        Denies only when today's spend has already consumed the day's headroom
        (``today_total_usd() >= daily_cap_usd - reservation_usd``). While
        headroom remains, the run is **admitted** even if its declared per-run
        worst-case ``max_budget_usd`` is larger than the remaining headroom. Note
        that ``max_budget_usd`` is *not* actually clamped or forwarded as a
        reduced ceiling — neither external backend enforces a per-run USD stop —
        so a large-budget run is admitted as-is and the only enforcement is this
        day-headroom gate (re-checked at every run's start) plus the
        orchestrator's generation-boundary halt. This removes the UX footgun
        where a small ``--daily-budget-usd`` plus the default
        ``--agent-max-budget`` silently denied *every* run before it started,
        while keeping the daily hard cap meaningful (once headroom is gone, every
        run is denied).

        This **reads** ``today_total_usd()`` directly and **never** calls
        :meth:`CostTracker.assert_under_cap`, so it cannot raise
        :class:`BudgetExceededError` (a ``BaseException``) out of the spine and
        cancel sibling tasks (plan §4.7.3-4).

        The guard never raises for *any* reason: a malformed tracker / ledger is
        logged and treated as "do not deny" (``False``), leaving the
        generation-boundary :meth:`headroom_exhausted` halt to enforce.

        Args:
            max_budget_usd: The per-run worst-case USD budget for the run about
                to start.

        Returns:
            ``True`` to deny (the spine should abort this run early with a logged
            warning and record a degraded ``BUDGET_DENIED`` row); ``False`` to
            admit. Specifically returns ``True`` exactly when
            ``today_total_usd() >= daily_cap_usd - reservation_usd`` (the day's
            headroom is exhausted). A requested ``max_budget_usd`` larger than the
            remaining headroom is *admitted with a WARNING* (not clamped, not
            denied) — neither external backend enforces a per-run USD stop.
        """
        try:
            today, cap, reservation, headroom = self._headroom_parts()
            threshold = cap - reservation
            requested = float(max_budget_usd or 0.0)

            # Hard stop: no usable headroom left today. This is the real daily
            # cap — denying here keeps it meaningful regardless of per-run budget.
            if headroom <= 0.0:
                logger.warning(
                    "[CostGuard] Denying run: today $%.4f has reached the day's "
                    "headroom $%.4f (cap $%.2f - reservation $%.2f). No budget "
                    "left; resume tomorrow or raise --daily-budget-usd.",
                    today, threshold, cap, reservation,
                )
                return True

            # Headroom remains. If the declared per-run worst-case exceeds it,
            # admit anyway (log the arithmetic) instead of denying every run. The
            # per-run budget is NOT clamped and no backend enforces a per-run USD
            # stop, so this run can overshoot the remaining headroom; the day is
            # still bounded by the next-run admission denial and the
            # generation-boundary halt (overshoot <= ``parallel`` in-flight runs).
            if requested > headroom:
                logger.warning(
                    "[CostGuard] Admitting run whose per-run budget exceeds the "
                    "day's remaining headroom: requested $%.4f > headroom $%.4f "
                    "(today $%.4f, cap $%.2f - reservation $%.2f). The budget is "
                    "NOT clamped and no backend enforces a per-run USD stop, so "
                    "this run may overshoot; later runs are denied once headroom "
                    "hits $0. Pass --agent-max-budget <= $%.4f to silence this.",
                    requested, headroom, today, cap, reservation, headroom,
                )
            return False
        except _NonFiniteLedger as exc:
            # Fail CLOSED on a corrupted ledger: with a non-finite ``today`` the
            # ``headroom <= 0.0`` guard above would be False (NaN comparisons
            # are all False), admitting every run and silently defeating the
            # day cap. Deny instead so the cap stays meaningful.
            logger.warning(
                "[CostGuard] Denying run: today_total_usd() is non-finite "
                "(%r); the ledger is corrupted, denying to keep the daily "
                "cap enforced (fail closed).", exc.today,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - never raise out of the gate
            # A pre-check failure must not abort the candidate batch; defer to the
            # generation-boundary headroom_exhausted halt.
            logger.warning(
                "[CostGuard] precheck failed (admitting run; daily headroom halt "
                "still applies): %s", exc,
            )
            return False

    def record(
        self,
        run: "AgentRunResult",
        task: object | None = None,
        solver: object | None = None,
    ) -> None:
        """Fold a finished run's real spend into the single ledger.

        Routes by the run's ``cost_basis`` so OpenHands and Terminus 2 share one
        append path (:meth:`CostTracker.record_usd`) and therefore one
        ``today_total_usd()`` basis (plan §4.7.1-2):

        * ``cost_basis == "native_usd"`` (OpenHands): append ``run.cost_usd``
          verbatim — no model needs to exist in ``PRICING`` (plan §4.7.1).
        * otherwise (Terminus 2 / priced-from-tokens): price the token usage with
          :func:`compute_cost_usd` against :data:`self.model` and append the
          result (plan §4.7.2). The model is guaranteed priceable because
          :meth:`__init__` resolved it at construction.

        The ledger line is stamped with an auditable ``extra`` carrying the
        source backend, the basis, the task id and the depth so
        ``agent_runs.jsonl`` reconciles with ``cost_summary.json`` (plan §4.7.5).

        Like the rest of the external path, this never raises out: a recording
        failure is logged (the run is already scored) rather than propagated.

        Only runs that produced an :class:`AgentRunResult` reach this fold; see
        the module docstring's *Ledger undercount bound* for the degraded paths
        that cannot be ledgered.

        Args:
            run: The finished :class:`AgentRunResult` carrying ``cost_usd`` /
                ``cost_basis`` and the ``agent_*`` token counts.
            task: Optional task object; its ``task_id`` (if present) is stamped
                into the ledger line's ``extra`` for auditing.
            solver: Optional solver object; its ``depth`` (if present) is stamped
                into the ledger line's ``extra`` for auditing.
        """
        try:
            source = getattr(run, "cost_basis", "priced_from_tokens")
            prompt_tokens = int(getattr(run, "agent_prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(run, "agent_completion_tokens", 0) or 0)
            cached_tokens = int(getattr(run, "agent_cached_tokens", 0) or 0)

            extra = {
                "source": "external_agent",
                "basis": source,
                "task_id": getattr(task, "task_id", None),
                "depth": getattr(solver, "depth", None),
            }

            # The branch derives ONLY ``cost_usd``; the ledger append itself is
            # the single shared call below (byte-identical line either way).
            if source == "native_usd":
                # OpenHands: append the agent's own litellm USD verbatim. No
                # PRICING entry required for this model (plan §4.7.1). Clamp a
                # non-finite/negative value to $0 first: ``x or 0.0`` does NOT
                # coerce NaN (NaN is truthy) or a negative, and a poisoned line
                # would make today_total_usd() NaN and silently disable the day
                # cap for the rest of the day (and across restart).
                cost_usd = float(getattr(run, "cost_usd", 0.0) or 0.0)
                if not (math.isfinite(cost_usd) and cost_usd >= 0.0):
                    logger.warning(
                        "[CostGuard] Dropping non-finite/negative native cost "
                        "%r for model %s (recording $0.0 to protect the daily "
                        "cap).", cost_usd, self.model,
                    )
                    cost_usd = 0.0
            else:
                # Terminus 2 / builtin-style: price the token usage with the same
                # function the ledger uses, then append via the shared USD path so
                # every basis lands in today_total_usd() identically (plan §4.7.2).
                cost_usd = compute_cost_usd(
                    self.model,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                )
            self.cost_tracker.record_usd(
                self.model,
                cost_usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                extra=extra,
            )
        except Exception as exc:  # noqa: BLE001 - recording must not abort scoring
            logger.warning(
                "[CostGuard] record failed (run already scored; spend not "
                "ledgered): %s", exc,
            )
