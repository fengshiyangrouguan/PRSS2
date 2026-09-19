"""Phase B: offline Gamma training (task book v4.1 §5.1 / §2.12 / §7).

The data windows are FIXED and the roles are locked: 64 roots establish one
mutually-exclusive task/protect pair, and that pair is replayed for
`TRAIN_STEPS` steps. A window is a dataset, not a consumable -- the gate
answers "is there enough data", not "how many times may I use it".

Per step, in order:

    B_task, B_protect      recomputed under the CURRENT theta (never cached)
    a_task, a_protect      re-derived by close_task / close_lpse (OAS inside)
    g_task                 -grad_theta J_task            (§2.12: NEGATIVE)
    q_v                    +grad_theta <sg(a_v), b_v>    (§2.12: POSITIVE)
    delta_adam             counterfactual AdamW on clones
    d_star                 proposal-space QP
    commit                 once, or ABORT the whole Phase B

The signs are not cosmetic: swapping them trains Gamma to LOWER the true
return. `self_test` asserts the direction empirically.

CERTIFICATE FAILURE EARLY-STOPS PHASE B (v4.2 §5.1) -- it is NOT a failure.
Under a fixed window and a fixed theta whose last update was rejected, the next
step recomputes bit-for-bit identical inputs and fails identically, so spinning
out the remaining steps would only repeat one deterministic outcome. The blocked
step committed nothing, so parameters, Adam moments, the step counter and the
scheduler are untouched and the in-memory theta IS the last valid Gamma; `run()`
reports `status="certificate_blocked"`, a `stop_reason` of
`"deterministic_certificate_failure"`, and how many steps actually committed.
Structural problems (role lock moved, window contents changed, protect-row count
mismatch) still raise `PhaseBFailed`.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# Thread pinning. Phase B does many small (<= a few hundred) matmuls and
# Cholesky/eigen solves per step; letting torch fan them out across every core
# of the host is pure contention. Measured on an AutoDL container reporting
# `nproc == 128`: a 30-step synthetic gate had not finished after 10 minutes at
# ~1600% CPU. The vendored PRSS2 trainer pins threads for the same reason.
# Override with OMP_NUM_THREADS / META_N_TORCH_THREADS.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, os.environ.get("META_N_TORCH_THREADS", "1"))
try:
    torch.set_num_threads(int(os.environ.get("META_N_TORCH_THREADS", "1")))
except (TypeError, ValueError):
    torch.set_num_threads(1)

from meta_n.rpbe import config as C
from meta_n.rpbe.census import fixed_hash_split
from meta_n.rpbe.qp import proposal_space_update

TreeId = Tuple[str, str]


class PhaseBFailed(RuntimeError):
    """A STRUCTURAL failure: the role lock moved, window contents changed, or
    the protect-row count disagreed. Nothing was written."""


class CertificateBlocked(Exception):
    """V4.2 §5.1: the proposal-space certificate rejected ONE update.

    This is NOT a training failure. The blocked step committed nothing, so the
    parameters, Adam moments, step counter and scheduler are all exactly as they
    stood after the last certified step -- the in-memory theta IS the last valid
    Gamma. Phase B therefore EARLY-STOPS here and reports it, rather than
    raising a failure, spinning the remaining steps, or switching windows.

    Spinning would be pointless on purpose: with a fixed window and a fixed
    theta whose last update was rejected, the next step recomputes the same
    B / cotangent / delta_adam / QP inputs bit-for-bit and fails identically.
    """

    def __init__(self, step: int, aborted: Any = None) -> None:
        super().__init__(
            "current proposal was not certified; Phase B terminated at the "
            "last valid Gamma because repeated evaluation would be "
            "deterministic under the frozen window/state (blocked at step {}"
            "{}). This is NOT a training failure and NOT a claim that the "
            "window carries no signal.".format(
                step, "" if aborted is None else ", {}".format(aborted)))
        self.step = step
        self.aborted = aborted


class PhaseB:
    """One offline Gamma run over one fixed task/protect window pair."""

    def __init__(self, fusion, task_window, protect_window,
                 *, steps: int = C.TRAIN_STEPS, lr: float = C.LR,
                 betas: Tuple[float, float] = C.BETAS, eps: float = C.EPS,
                 weight_decay: float = C.WEIGHT_DECAY,
                 grad_clip: float = C.GRAD_CLIP,
                 warmup_steps: int = C.WARMUP_STEPS,
                 total_steps: int = C.TOTAL_STEPS,
                 kappa: float = C.KAPPA, tau: float = C.TAU,
                 foreach: bool = C.FOREACH):
        self.fusion = fusion
        self.task_window = task_window
        self.protect_window = protect_window
        self.steps = int(steps)
        self.grad_clip = float(grad_clip)
        self.kappa, self.tau = float(kappa), float(tau)
        self.params = [p for p in fusion.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("fusion has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            self.params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, foreach=foreach)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=_warmup_cosine(
                int(warmup_steps), int(total_steps)))
        # Role lock: captured once, asserted never to move.
        self._task_roles = frozenset(task_window.tree_ids)
        self._protect_roles = frozenset(protect_window.tree_ids)
        self._assert_disjoint_and_locked()

    def _assert_disjoint_and_locked(self) -> None:
        if not self._task_roles or not self._protect_roles:
            raise ValueError("both windows must be non-empty")
        if self._task_roles & self._protect_roles:
            raise ValueError(
                "task and protect trees overlap: {}"
                .format(sorted(self._task_roles & self._protect_roles)))
        for w, roles in ((self.task_window, self._task_roles),
                         (self.protect_window, self._protect_roles)):
            if frozenset(w.tree_ids) != roles:
                raise ValueError("window tree roles changed after construction")

    # -- per-cut cotangent replay (§5.1 pass 2) ---------------------------
    def _per_cut_grads(self, window,
                       cotangent_by_cut: Dict[Any, torch.Tensor],
                       *, sign: float) -> List[torch.Tensor]:
        """`q_v = grad_theta <sg(a_v), b_v(theta)>` for every cut.

        The value is identically zero by construction:
            <sg(a_v), b_v> - <sg(a_v), sg(b_v)>  ==  0
        only its gradient is meaningful. `sign` is +1 for protect and -1 for
        the task direction, which maximises J_task.
        """
        B = window.replay_B(self.fusion, grad=True)          # [M, b_v_dim]
        if not B.requires_grad:
            raise PhaseBFailed("replay_B produced no graph")
        rows = list(window._rows)
        grads: List[torch.Tensor] = []
        for i, r in enumerate(rows):
            a = torch.as_tensor(cotangent_by_cut[r.cut_id],
                                dtype=B.dtype).detach()
            b_i = B[i]
            surrogate = (a * b_i).sum() - (a * b_i.detach()).sum()
            g = torch.autograd.grad(surrogate, self.params,
                                    retain_graph=(i + 1 < len(rows)),
                                    allow_unused=False)
            grads.append(torch.cat([x.reshape(-1) for x in g]))
        return grads

    def _aggregate_grad(self, window, cotangent_by_cut, *, sign: float):
        """`grad_theta (sign * sum_v <sg(a_v), b_v(theta)>)` as param .grad."""
        B = window.replay_B(self.fusion, grad=True)
        loss = 0
        for i, r in enumerate(list(window._rows)):
            a = torch.as_tensor(cotangent_by_cut[r.cut_id],
                                dtype=B.dtype).detach()
            loss = loss + (a * B[i]).sum()
        (sign * loss).backward()
        grad = [p.grad.detach().clone() if p.grad is not None
                else torch.zeros_like(p) for p in self.params]
        self.optimizer.zero_grad(set_to_none=True)
        return grad

    # -- one boundary ------------------------------------------------------
    def step(self, n: int) -> Dict[str, Any]:
        win_roles_before = (frozenset(self.task_window.tree_ids),
                            frozenset(self.protect_window.tree_ids))
        n_rows_before = (self.task_window.n_rows, self.protect_window.n_rows)

        # pass 1: statistics under the CURRENT theta (never cached)
        task_out = self.task_window.close_task(self.fusion)
        protect_out = self.protect_window.close_lpse(self.fusion)

        # task direction: MAXIMISE J_task -> the surrogate is NEGATIVE (§2.12)
        g_task = self._aggregate_grad(
            self.task_window, task_out["cotangent_by_cut"], sign=-1.0)

        # protect rows: one direction PER cut, never pre-summed (§2.7)
        q_rows = self._per_cut_grads(
            self.protect_window, protect_out["cotangent_by_cut"], sign=+1.0)
        if len(q_rows) != self.protect_window.n_rows:
            raise PhaseBFailed("q_rows count != protect rows")

        # Install g_task LAST: `_per_cut_grads` uses torch.autograd.grad, which
        # writes into `.grad` and would otherwise clobber it.
        for p, g in zip(self.params, g_task):
            p.grad = g.clone()

        out = proposal_space_update(
            self.optimizer, self.params, q_rows, g_task,
            scheduler=self.scheduler, kappa=self.kappa, tau_feas=self.tau,
            grad_clip=self.grad_clip)
        if not out.get("committed", False):
            # V4.2 §5.1: reject THIS update and early-stop Phase B at the last
            # valid Gamma. Not a failure, not a spin, no window switch.
            raise CertificateBlocked(n, out.get("aborted"))

        # Contents and roles must be untouched by an optimiser step.
        if (frozenset(self.task_window.tree_ids),
                frozenset(self.protect_window.tree_ids)) != win_roles_before:
            raise PhaseBFailed("tree roles moved during a step")
        if (self.task_window.n_rows, self.protect_window.n_rows) != n_rows_before:
            raise PhaseBFailed("window contents changed during a step")

        out["step"] = n
        out["J_task"] = task_out["J_task"]
        out["J_LPSE"] = protect_out["J_LPSE"]
        out["N_unique_trees_task"] = self.task_window.n_trees
        out["N_unique_trees_protect"] = self.protect_window.n_trees
        out["M_cuts_task"] = self.task_window.n_rows
        out["M_cuts_protect"] = self.protect_window.n_rows
        out["D_task"] = task_out["D"]
        out["D_protect"] = protect_out["D"]
        out["oas_alpha_z_task"] = task_out.get("oas_alpha_z_task")
        out["oas_alpha_p_task"] = task_out.get("oas_alpha_p_task")
        out["oas_alpha_z_protect"] = protect_out.get("oas_alpha_z_protect")
        out["oas_alpha_p_protect"] = protect_out.get("oas_alpha_p_protect")
        out["gamma_param_count"] = sum(p.numel() for p in self.params)
        out["omega_replay_count"] = 0          # Phase B never calls Omega
        return out

    def run(self) -> Dict[str, Any]:
        """Run up to ``self.steps`` boundaries.

        ``status`` is ``"completed"`` or ``"certificate_blocked"``. The early
        stop is a NORMAL outcome (V4.2 §5.1), not a failure: the blocked step
        wrote nothing, so the fusion module still holds the last valid Gamma and
        ``history`` holds every certified step. Only a STRUCTURAL problem (role
        lock moved, window contents changed, protect-row count mismatch) raises
        :class:`PhaseBFailed`.
        """
        history: List[Dict[str, Any]] = []
        status, stop_reason, blocked = "completed", None, None
        for n in range(self.steps):
            try:
                history.append(self.step(n))
            except CertificateBlocked as stop:
                status = "certificate_blocked"
                stop_reason = "deterministic_certificate_failure"
                blocked = {"step": stop.step, "aborted": stop.aborted}
                break
        out = {
            "steps": len(history),
            "planned_steps": self.steps,
            "history": history,
            "ended_early": status != "completed",
            "status": status,
            "stop_reason": stop_reason,
            "blocked_at": blocked,
            # No separate "save_last_valid_gamma()" is needed: the blocked step
            # committed nothing, so the parameters and Adam moments in memory
            # already ARE the state after the last certified step.
            "gamma_is_last_valid": True,
        }
        self.last_result = out
        return out

    def save_gamma(self, path, *, extra=None, steps=None, status=None,
                   stop_reason=None):
        """Persist the CURRENT parameters as the frozen Phase-C Gamma.

        Whatever the module holds at the end of :meth:`run` IS the last valid
        Gamma -- a certificate-blocked step commits nothing (v4.2 §5.1). The
        recorded ``steps`` / ``status`` / ``stop_reason`` default to the most
        recent :meth:`run`, so the checkpoint carries how training ended.
        """
        from meta_n.rpbe.checkpoint import save_gamma_checkpoint
        last = getattr(self, "last_result", None) or {}
        return save_gamma_checkpoint(
            path, self.fusion,
            steps=int(last.get("steps", 0) if steps is None else steps),
            status=str(last.get("status", "not_run") if status is None
                       else status),
            stop_reason=(last.get("stop_reason") if stop_reason is None
                         else stop_reason),
            extra=extra)


def _warmup_cosine(warmup: int, total: int):
    """LR multiplier: linear warmup then cosine decay to 0 at `total`."""
    warmup = max(1, int(warmup))
    total = max(warmup + 1, int(total))

    def fn(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        prog = min(1.0, float(step - warmup) / float(total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return fn


def split_window_trees(tree_ids: Sequence[TreeId],
                       n_task: int = C.N_TREES_TASK) -> Tuple[List, List]:
    """Frozen, reproducible task/protect split (§4.2). Never re-cut on results."""
    return fixed_hash_split(tree_ids, n_task=n_task)


# --------------------------------------------------------------------------
# acceptance self-test
# --------------------------------------------------------------------------

class _Fusion(torch.nn.Module):
    """Trainable stand-in with the SlottedFusion interface."""

    def __init__(self, d_e: int = C.D_E):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.W = torch.nn.Parameter(torch.randn(d_e, d_e, generator=g) * 0.05)

    def forward(self, X_v, q_emb, mask):
        h = X_v @ self.W
        Z = h[:C.N_SLOTS]
        if Z.shape[0] < C.N_SLOTS:
            pad = torch.zeros(C.N_SLOTS - Z.shape[0], h.shape[1])
            Z = torch.cat([Z, pad], 0)
        A = torch.softmax(torch.zeros(C.N_SLOTS, X_v.shape[0]), dim=-1)
        return Z, A


class _Row:
    def __init__(self, cut_id, tree_id, n_v, seed, r, weight=1.0):
        self.cut_id, self.tree_id = cut_id, tree_id
        self.X_v = torch.randn(n_v, C.D_E)
        self.q_emb = torch.randn(C.D_E)
        self.mask = torch.zeros(C.N_SLOTS, n_v)
        self.p = torch.randn(C.N_BRANCHES, C.M_SKETCH)
        self.r, self.weight = r, weight


def _rows(prefix, trees, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for t in trees:
        for occ in range(2):
            out.append(_Row((t + ":" + prefix, occ), (t, "gen0_seed"),
                            5, 0, 0.1 * occ))
            out[-1].p = torch.randn(C.N_BRANCHES, C.M_SKETCH, generator=g)
            out[-1].X_v = torch.randn(5, C.D_E, generator=g)
    return out


def self_test() -> int:
    import os
    from meta_n.rpbe.window import StatWindow

    print("trainer.py acceptance")
    print("=" * 68)

    # REALISTIC sizes. With only a handful of trees the effective dof D falls
    # to ~4 against B_V_DIM=32, OAS pushes alpha to 1, C_ZP is annihilated and
    # J is exactly 0 -- a true small-sample property, not a bug. The frozen
    # protocol's 32 trees/window is what makes the score non-degenerate.
    trees = ["run_{}".format(i) for i in range(64)]
    task_roots, protect_roots = split_window_trees(trees, n_task=32)
    assert not (set(task_roots) & set(protect_roots))
    assert len(task_roots) == 32 and len(protect_roots) == 32
    print("OK  frozen split       {} task / {} protect trees, disjoint, "
          "reproducible".format(len(task_roots), len(protect_roots)))

    tw = StatWindow(min_unique_trees=32, name="task")
    pw = StatWindow(min_unique_trees=32, name="protect")
    tw.add(_rows("task", task_roots, seed=1))
    pw.add(_rows("protect", protect_roots, seed=2))

    fusion = _Fusion()
    trainer = PhaseB(fusion, tw, pw, steps=3)
    w0 = fusion.W.detach().clone()
    hist = trainer.run()
    assert hist["steps"] == 3
    w1 = fusion.W.detach().clone()
    # Bitwise inequality, not allclose: the effective lr is ~3e-5, so a real
    # (correct) update can still sit inside allclose's default tolerance.
    dmax = float((w1 - w0).abs().max())
    assert not torch.equal(w0, w1), "Gamma did not move at all"
    print("OK  runs               {} steps committed; max |dW| = {:.3e}".format(
        hist["steps"], dmax))

    # Omega is never called in Phase B
    assert all(h["omega_replay_count"] == 0 for h in hist["history"])
    print("OK  omega_replay_count 0 for every step")

    # window contents/roles unchanged across all steps
    assert frozenset(tw.tree_ids) == frozenset(
        (t, "gen0_seed") for t in task_roots)
    assert frozenset(pw.tree_ids) == frozenset(
        (t, "gen0_seed") for t in protect_roots)
    assert tw.n_rows == 64 and pw.n_rows == 64
    print("OK  role lock          300-step-style replay leaves window contents "
          "and tree roles identical")

    # DIAGNOSTIC DIRECTION: J_task must not collapse when we MAXIMISE it.
    j0 = hist["history"][0]["J_task"]
    j2 = hist["history"][-1]["J_task"]
    assert j2 >= j0 - 1e-6, (j0, j2)
    print("OK  sign direction     J_task {} -> {} (maximised, not minimised)"
          .format(round(j0, 4), round(j2, 4)))

    # a scalar R gives oas_alpha_p = 0 exactly
    assert all(h["oas_alpha_p_task"] == 0.0 for h in hist["history"])
    print("OK  scalar R OAS       oas_alpha_p_task == 0.0 at every step")

    # CERT FAILURE EARLY-STOPS PHASE B (v4.2 §5.1) -- and is NOT a failure.
    # The QP's own abort semantics live in qp.self_test; here we pin the
    # TRAINER's handling of "not committed": it must (a) commit nothing, (b)
    # stop at the first blocked step instead of spinning the remaining 4,
    # (c) report status="certificate_blocked" rather than raising, and (d) leave
    # the parameters bit-for-bit untouched, so the in-memory Gamma is the last
    # valid one. (Patching is used because provoking a real certificate failure
    # also depends on the interface geometry, which would make this test flaky.)
    # NOTE: patch the module that is actually executing. Under `-m` that is
    # `__main__`, NOT `meta_n.rpbe.trainer` (which would be a second, unused
    # module object).
    import sys
    mod = sys.modules[__name__]
    ran = {"n": 0}
    real_psu = mod.proposal_space_update

    def never_commits(*a, **k):
        ran["n"] += 1
        return {"committed": False, "aborted": "test_forced"}

    mod.proposal_space_update = never_commits
    try:
        f2 = _Fusion()
        w_before = f2.W.detach().clone()
        t2 = PhaseB(f2, tw, pw, steps=5)
        res2 = t2.run()
        assert ran["n"] == 1, ("Phase B kept stepping after the block: "
                               "{} calls".format(ran["n"]))
        assert res2["status"] == "certificate_blocked", res2["status"]
        assert res2["stop_reason"] == "deterministic_certificate_failure"
        assert res2["steps"] == 0 and res2["planned_steps"] == 5, res2
        assert res2["ended_early"] is True
        assert res2["blocked_at"]["step"] == 0, res2["blocked_at"]
        assert res2["gamma_is_last_valid"] is True
        assert torch.equal(w_before, f2.W.detach()), (
            "a certificate-blocked step wrote parameters")
    finally:
        mod.proposal_space_update = real_psu
    print("OK  cert blocked       early-stop at step 0 (proposal_space_update "
          "called {}x, not 5): status={!r} stop_reason={!r} steps={}/{}; "
          "parameters bit-for-bit untouched".format(
              ran["n"], res2["status"], res2["stop_reason"], res2["steps"],
              res2["planned_steps"]))

    # role mismatch is refused at construction
    bad = StatWindow(min_unique_trees=32)
    bad.add(_rows("task", task_roots, seed=3))     # SAME trees as task
    try:
        PhaseB(_Fusion(), tw, bad, steps=1)
        raise AssertionError("overlapping windows accepted")
    except ValueError:
        pass
    print("OK  disjointness       overlapping task/protect trees refused")

    paid = 0
    led = os.environ.get("META_N_REQUEST_LEDGER", "").strip()
    if led and os.path.isfile(led):
        from meta_n.rpbe.accounting import snapshot
        paid = snapshot()["backend_requests"]
    assert paid == 0
    print("OK  zero-cost          paid backend_requests = {}".format(paid))

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
