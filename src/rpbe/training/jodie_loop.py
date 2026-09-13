"""JODIE node-classification loop with a one-pass RPBE training path.

The task protocol remains the upstream chronological BCE loop.  When RPBE is
enabled, the same TGN query also emits bounded internal cut states.  Their
Y1/Y2 rows update a detached running Ky Fan reference and a lagged VJP is
added to the *same* loss before the batch's single backward.  There is no
shadow memory, macro-window graph, or second complete query.
"""

import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from rpbe.loss import KFLaggedWindow, KFMomentWindow
from rpbe.training import checkpoint as ckpt
from rpbe.training.isolation import assert_clean, rpbe_fingerprint

EPS = 1e-7


def metric_bundle(labels, probs):
    """AUC/AP/NLL plus the diagnostics the v1 protocol reported."""
    labels = np.asarray(labels).astype(np.float64)
    probs = np.clip(np.asarray(probs).astype(np.float64), EPS, 1 - EPS)
    auc = float(roc_auc_score(labels, probs)) \
        if len(np.unique(labels)) > 1 else float("nan")
    ap = float(average_precision_score(labels, probs)) \
        if labels.sum() > 0 else 0.0
    nll = float(-(labels * np.log(probs)
                  + (1 - labels) * np.log(1 - probs)).mean())
    pos = labels > 0.5
    neg = ~pos
    return {
        "auc": auc,
        "ap": ap,
        "nll": nll,
        "positive_nll": float(-np.log(probs[pos]).mean())
        if pos.any() else float("nan"),
        "negative_nll": float(-np.log(1 - probs[neg]).mean())
        if neg.any() else float("nan"),
        "positives": int(pos.sum()),
        "pairs": int(len(labels)),
        "positive_rate": float(pos.mean()),
        "mean_prob_positive": float(probs[pos].mean())
        if pos.any() else float("nan"),
        "mean_prob_negative": float(probs[neg].mean())
        if neg.any() else float("nan"),
    }


def select_trace_rows(labels, max_roots: int, seed: int, batch_index: int,
                      mode: str = "positive_first") -> list:
    """Select a bounded set of top-level query rows."""
    if mode == "off" or max_roots <= 0:
        return []
    labels = np.asarray(labels)
    if mode == "evenly_spaced":
        if len(labels) == 0:
            return []
        rows = np.linspace(0, len(labels) - 1,
                           min(max_roots, len(labels))).astype(np.int64)
        return [int(row) for row in rows]
    if mode != "positive_first":
        raise ValueError("unknown trace mode {}".format(mode))
    pos = np.flatnonzero(labels > 0.5).tolist()
    neg = np.flatnonzero(labels <= 0.5).tolist()
    chosen = pos[:max_roots]
    remain = max_roots - len(chosen)
    if remain > 0 and neg:
        rng = np.random.RandomState(seed + 104729 * (batch_index + 1))
        chosen.extend(neg if len(neg) <= remain
                      else rng.choice(neg, size=remain,
                                      replace=False).tolist())
    return sorted(chosen)


class JodieNodeClassificationLoop:
    """Own the official-TGN stream; RPBE hooks are optional."""

    def __init__(self, *, tgn, decoder, repr_optimizer, head_optimizer,
                 device, batch_size, n_neighbors, grad_clip, monitor, seed,
                 finetune_host=False, selection_metric="auc", adapter=None,
                 cut_builder=None, fixed_maps=None, rpbe_cfg=None,
                 trace_roots=8, trace_mode="evenly_spaced",
                 train_eval_auc=False, grad_diag=None,
                 kf_estimator="exact_replay",
                 kf_fail_below_threshold=False,
                 rpbe_constrain=False, rpbe_kappa=0.05):
        self.tgn = tgn
        self.decoder = decoder
        # Representation parameters (host + compressor, everything that can
        # change a cut's z) update once per macro-group; the head updates
        # every batch.  Both groups receive gradients from every backward.
        # Three SEPARATE switches (review):
        #   repr_train_on — the representation optimizer is nonempty (pure
        #                   frozen-host vanilla leaves it None)
        #   component_on  — the compressor (Gamma) is attached
        #   kf_on         — the Ky Fan term is computed
        self.repr_optimizer = repr_optimizer
        self.head_optimizer = head_optimizer
        self.repr_train_on = bool(
            repr_optimizer is not None
            and repr_optimizer.param_groups
            and repr_optimizer.param_groups[0]["params"])
        self.repr_params = (list(repr_optimizer.param_groups[0]["params"])
                            if self.repr_train_on else [])
        self.head_params = list(head_optimizer.param_groups[0]["params"])
        # Constrained RPBE (reviewer formulation, 2026-09-13): per-tree
        # RPBE directions stay SEPARATE and the group-end projection solves
        #   min_d 1/2||d - t||^2  s.t.  h_j^T d >= -kappa ||t||
        # with ROW-NORMALIZED rows h_j = g_j/||g_j|| (cosine Gram), a
        # FISTA budget ladder (certificate escalation), and a hard gate:
        # when the certificate still fails the representation step is
        # SKIPPED for that group (no constrained update is executed).
        # This is a PRE-OPTIMIZER gradient-space projection (the paper
        # wording must say so; Adam still reshapes the actual displacement).
        # Scope: compressor (Gamma) parameters only — every other repr
        # param keeps the plain task gradient.
        self.rpbe_constrain = bool(rpbe_constrain)
        self.rpbe_kappa = float(rpbe_kappa)
        self._cstr_scope_params = []   # compressor params (the scope)
        self._cstr_scope_task = None   # aggregate task grad accumulators
        self._cstr_scope_snap = None   # per-batch task-grad snapshots
        self._cstr_tree_aux = {}       # (tree_id, tau) -> [grad per scope p]
        self._cstr_groups = 0
        self._cstr_active = 0
        self._cstr_skips = 0
        self._cstr_corr_ratios = []
        self.device = device
        self.batch_size = int(batch_size)
        self.n_neighbors = int(n_neighbors)
        self.grad_clip = float(grad_clip)
        self.monitor = monitor
        self.seed = int(seed)
        self.finetune_host = bool(finetune_host)
        self.selection_metric = selection_metric
        self.adapter = adapter
        self.cut_builder = cut_builder
        self.fixed_maps = fixed_maps
        self.rpbe_cfg = rpbe_cfg
        self.lambda_kf = float(rpbe_cfg.lambda_kf) \
            if rpbe_cfg is not None else 0.0
        self.trace_roots = int(trace_roots)
        self.trace_mode = trace_mode
        self.train_eval_auc = bool(train_eval_auc)
        # Optional per-batch gradient diagnostics (the sprint script).
        # A dict in; its "fn" hook (implemented OUTSIDE the training
        # module, so the source contract stays replay-free) is called on
        # every KF batch before the single backward, and its return value
        # is appended to grad_diag["rows"].
        self.grad_diag = grad_diag
        self._grad_diag_fn = None
        if grad_diag is not None:
            grad_diag.setdefault("rows", [])
            self._grad_diag_fn = grad_diag.get("fn")
        # Eighth review D: lambda=0 is a TRUE task-only fast path — the
        # compressor still shapes the forward (TGN + Gamma, task only), but
        # no tracing, no cuts, no P projection, no window updates.
        self.component_on = bool(adapter is not None
                                 and fixed_maps is not None
                                 and rpbe_cfg is not None)
        # Review: the one-window-lag estimator is retired from production
        # (parameter-space diagnosis: lagged-exact cosine 0.002-0.007,
        # held-out virtual step -0.0009 vs exact +0.0418).  The default
        # is the same-window two-pass exact replay; "lagged" remains for
        # diagnostics and reproduction only.
        if kf_estimator not in ("exact_replay", "lagged", "off"):
            raise ValueError("unknown kf_estimator {}".format(kf_estimator))
        self.kf_estimator = kf_estimator
        self.kf_fail_below_threshold = bool(kf_fail_below_threshold)
        self.kf_on = bool(self.component_on and self.lambda_kf > 0.0
                          and cut_builder is not None
                          and kf_estimator != "off")
        self.rpbe_on = self.component_on
        if self.rpbe_on and self.trace_mode == "positive_first":
            raise ValueError(
                "RPBE cut selection cannot depend on the current root label; "
                "use trace_mode='evenly_spaced' or 'off'")

        kf_dims = {}
        if self.rpbe_on:
            allowed = set(getattr(adapter, "compression_taus",
                                  rpbe_cfg.state_dims.keys()))
            if rpbe_cfg.kf_taus is not None:
                allowed &= set(rpbe_cfg.kf_taus)
            kf_dims = {tau: dim for tau, dim in rpbe_cfg.state_dims.items()
                       if tau in allowed}
            if not kf_dims:
                raise ValueError(
                    "RPBE needs at least one internal interface; leaf/root "
                    "interfaces are not compressible")
        # Eighth review A: per-interface rank normalization coefficients.
        # J_norm = sum_tau alpha_tau J_tau / rank_tau / sum alpha_tau lives
        # in ~[0, 1]; the denominator is FIXED by the configured
        # interfaces, never by which taus happen to activate in a batch.
        # rank_tau is variant-dependent: min(d_tau, m) for full
        # balancing, d_tau * m for diagonal whitening (its natural
        # upper bound is dm, not the Ky Fan rank), 1 for reconstruction
        # (the review form already normalizes J_rec into ~[0, 1]).
        self._tau_coeff = {}
        if self.kf_on:
            alpha_sum = float(sum(
                rpbe_cfg.alpha(tau) for tau in kf_dims))
            for tau in kf_dims:
                self._tau_coeff[tau] = rpbe_cfg.alpha(tau) / (
                    self._tau_rank(tau) * alpha_sum)
        if self.rpbe_on:
            if kf_estimator == "exact_replay":
                # First release: only full_balancing has an exact adjoint
                # path (close_replay -> latent_z_adjoint).
                if rpbe_cfg.kf_variant != "full_balancing":
                    raise ValueError(
                        "kf_estimator=exact_replay supports only "
                        "kf_variant=full_balancing for now")
                self.kf_window = KFMomentWindow(
                    kf_dims, min_ratio=rpbe_cfg.kf_min_ratio,
                    min_abs=rpbe_cfg.kf_min_abs, eps=rpbe_cfg.ridge_eps,
                    fixed_maps=fixed_maps, variant=rpbe_cfg.kf_variant,
                    autoclose=False)
            else:
                self.kf_window = KFLaggedWindow(
                    kf_dims, min_ratio=rpbe_cfg.kf_min_ratio,
                    min_abs=rpbe_cfg.kf_min_abs, eps=rpbe_cfg.ridge_eps,
                    fixed_maps=fixed_maps, variant=rpbe_cfg.kf_variant)
        else:
            self.kf_window = None

    def _clip(self, params):
        if self.grad_clip <= 0:
            return
        live = [p for p in params if p.grad is not None]
        if live:
            torch.nn.utils.clip_grad_norm_(
                live, max_norm=self.grad_clip, error_if_nonfinite=True)

    # ------------------------------------------------------------- stream state
    def reset_memory(self):
        if self.tgn.use_memory:
            self.tgn.memory.__init_memory__()

    def _full_official_embedding_call(self, sources, destinations, timestamps,
                                      edge_idxs, grad_enabled):
        """Exactly one upstream src/dst/dst temporal-embedding query."""
        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            return self.tgn.compute_temporal_embeddings(
                sources, destinations, destinations, timestamps, edge_idxs,
                self.n_neighbors)

    def _consume_kf(self, cuts, step):
        scores, surrogates, cold = self.kf_window.consume(cuts)
        raw_score = float(sum(
            self.rpbe_cfg.alpha(tau) * score
            for tau, score in scores.items()))
        norm_score = float(sum(
            self._tau_coeff[tau] * score
            for tau, score in scores.items()))
        if surrogates:
            auxiliary = -self.lambda_kf * sum(
                self._tau_coeff[tau] * value
                for tau, value in surrogates.items())
        else:
            auxiliary = torch.zeros((), device=self.device)

        if scores:
            bounds = {tau: self._tau_rank(tau) for tau in scores}
            self.monitor.validate_kf(scores, bounds, step)
        return raw_score, norm_score, scores, auxiliary, set(cold)

    def _tau_rank(self, tau) -> float:
        """Normalization bound for one interface, variant-dependent."""
        d = min(int(self.rpbe_cfg.state_dims[tau]), int(self.fixed_maps.m))
        if self.rpbe_cfg.kf_variant == "diagonal":
            return float(int(self.rpbe_cfg.state_dims[tau])
                         * int(self.fixed_maps.m))
        if self.rpbe_cfg.kf_variant == "reconstruction":
            return 1.0
        return float(d)

    def _cstr_accum_scope_task(self):
        """Accumulate the per-batch task gradient on the correction scope.

        repr grads are zeroed once per group, so ``p.grad`` is the running
        total — accumulate the DELTA against the previous snapshot (the
        running total would triangularly reweight earlier batches).
        """
        if self._cstr_scope_task is None:
            self._cstr_scope_task = [
                torch.zeros_like(p, device="cpu", dtype=torch.float32)
                for p in self._cstr_scope_params]
            self._cstr_scope_snap = [None] * len(self._cstr_scope_params)
        for k, p in enumerate(self._cstr_scope_params):
            g = p.grad
            if g is None:
                self._cstr_scope_snap[k] = None
                continue
            prev = self._cstr_scope_snap[k]
            delta = g if prev is None else (g - prev)
            self._cstr_scope_task[k].add_(delta.detach().float().cpu())
            self._cstr_scope_snap[k] = g.detach().clone()

    def _cstr_accum_tree_dirs(self, term_keys, term_pieces):
        """Per-(tree, tau) RPBE directions, kept SEPARATE.

        Aggregates the raw surrogate pieces per (tree, tau) into one scalar
        per key (negated to match the aggregate aux sign convention), then
        extracts the parameter-space rows with CHUNKED batched VJPs
        (``is_grads_batched=True``): each chunk of 8 scalars is one
        autograd.grad call, the last chunk releases the batch graph
        (``retain_graph=False``).  Rows are detached, cast to float32 and
        parked in a CPU row bank — the N x P bank NEVER lives on the GPU;
        only the small active set is materialized there at group close.
        """
        by_tree: Dict[tuple, list] = {}
        for k, key in enumerate(term_keys):
            by_tree.setdefault(key, []).append(k)
        keys = sorted(by_tree.keys())
        n_keys = len(keys)
        for idx, key in enumerate(keys):
            idxs = by_tree[key]
            sub = term_pieces[idxs[0]]
            for _j in idxs[1:]:
                sub = sub + term_pieces[_j]
            # sign convention (see UCI counterpart); cast to float32 —
            # the surrogate VALUE is exactly zero, only its graph matters.
            sub = (-sub).float()
            # the LAST VJP releases the batch graph
            retain = (idx < n_keys - 1)
            gs = torch.autograd.grad(
                sub, self._cstr_scope_params,
                retain_graph=retain, allow_unused=True)
            acc = self._cstr_tree_aux.get(key)
            if acc is None:
                acc = [torch.zeros_like(
                    p, device="cpu", dtype=torch.float32)
                    for p in self._cstr_scope_params]
                self._cstr_tree_aux[key] = acc
            for _k2, gg in enumerate(gs):
                if gg is not None:
                    acc[_k2].add_(gg.detach().float().cpu())

    def _cstr_group_close(self, group_k, group_start):
        """Reviewer-formulation group-end projection, memory-layered:

            min_d 1/2||d - t||^2  s.t.  h_j^T d >= -kappa ||t||,
            h_j = g_j / ||g_j||          (cosine Gram — row normalization)

        * rows live in a CPU bank (N x P never sits on the GPU);
        * cosine scan on the CPU picks the ACTIVE set (violated rows);
        * only the small active matrix is materialized on the GPU and
          solved (row-normalized QP, FISTA budget ladder);
        * a full CPU re-scan certifies EVERY row; newly violated rows join
          the active set for another solve round (max 3 rounds);
        * hard gate: certificate failure after the ladder SKIPS the
          representation step (returns False).

        t is the aggregate task gradient on the SCOPE (compressor).
        """
        dev = self.device
        scope = self._cstr_scope_params
        self._cstr_groups += 1
        if self._cstr_scope_task is None:
            self._cstr_reset_group_state()
            return True
        t_cpu = torch.cat([a.reshape(-1)
                           for a in self._cstr_scope_task])  # [P] CPU
        nt = float(t_cpu.norm())
        keys = sorted(self._cstr_tree_aux.keys())
        if not keys or nt <= 1e-12:
            self._cstr_reset_group_state()
            return True

        def _row(k):
            return torch.cat([a.reshape(-1)
                              for a in self._cstr_tree_aux[k]])

        # ---- CPU cosine scan (chunked) ----
        cos_all = torch.zeros(len(keys), dtype=torch.float64)
        gn_all = torch.zeros(len(keys), dtype=torch.float64)
        for cs in range(0, len(keys), 200):
            ce = min(len(keys), cs + 200)
            rows = torch.stack([_row(keys[j]) for j in range(cs, ce)])
            rows = rows.double()
            gn = rows.norm(dim=1)
            cos = (rows @ t_cpu.double()) / (gn * nt + 1e-30)
            cos_all[cs:ce] = cos
            gn_all[cs:ce] = gn.float()
        c_np = cos_all.numpy()
        frac_below = float(np.mean(c_np < -self.rpbe_kappa))
        active_keys = [keys[j] for j in range(len(keys))
                       if c_np[j] < -self.rpbe_kappa]
        if not active_keys:
            self._cstr_reset_group_state()
            return True
        # ---- active-set solve on the GPU ----
        A = torch.stack([_row(k) for k in active_keys]).to(dev)  # [Na, P]
        gn = A.norm(dim=1)
        H = A / gn[:, None]
        b = -self.rpbe_kappa * nt * torch.ones(H.shape[0], device=dev)
        t = t_cpu.float().to(dev)
        K = H @ H.t()
        cvec = b - (H @ t)
        Na = int(H.shape[0])
        lam = torch.zeros(Na, device=dev)
        with torch.no_grad():
            v = torch.randn(Na, device=dev)
            v = v / (v.norm() + 1e-30)
            lam_max = 1.0
            for _ in range(30):
                v = K @ v
                nv = float(v.norm())
                if nv <= 1e-30:
                    break
                v = v / nv
                lam_max = nv
        eta = 1.0 / (lam_max + 1e-12)
        d = None
        viol_max = float("inf")
        for _round in range(3):
            # ---- FISTA budget ladder on the active set ----
            for budget in (2000, 8000, 32000):
                y = lam
                tk = 1.0
                for _ in range(budget):
                    lam_new = torch.clamp(y + eta * (cvec - K @ y),
                                          min=0.0)
                    tk_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
                    y = lam_new + ((tk - 1.0) / tk_new) * (lam_new - lam)
                    lam = lam_new
                    tk = tk_new
                d = t + H.t() @ lam
                viol = b - (H @ d)
                viol_max = float(
                    (viol / (nt + 1e-30)).max()) if viol.numel() else 0.0
                if viol_max <= 1e-6:
                    break
            if viol_max > 1e-6:
                break
            # ---- full CPU re-scan: certify EVERY row ----
            d_cpu = d.detach().cpu().double()
            worst = -1.0
            worst_j = -1
            for cs in range(0, len(keys), 200):
                ce = min(len(keys), cs + 200)
                rows = torch.stack([_row(keys[j])
                                    for j in range(cs, ce)]).double()
                viol = (-self.rpbe_kappa * nt
                        - (rows @ d_cpu)
                        / (rows.norm(dim=1) * nt + 1e-30))
                if viol.numel() and float(viol.max()) > worst:
                    worst = float(viol.max())
                    worst_j = int(cs + torch.argmax(viol).item())
            if worst <= 1e-6:
                viol_max = 0.0
                break
            # newly violated row joins the active set
            new_key = keys[worst_j]
            if new_key in active_keys:
                break
            active_keys.append(new_key)
            A = torch.stack([_row(k) for k in active_keys]).to(dev)
            gn = A.norm(dim=1)
            H = A / gn[:, None]
            b = -self.rpbe_kappa * nt * torch.ones(H.shape[0], device=dev)
            K = H @ H.t()
            cvec = b - (H @ t)
            lam = torch.zeros(H.shape[0], device=dev)
            with torch.no_grad():
                v = torch.randn(H.shape[0], device=dev)
                v = v / (v.norm() + 1e-30)
                lam_max = 1.0
                for _ in range(30):
                    v = K @ v
                    nv = float(v.norm())
                    if nv <= 1e-30:
                        break
                    v = v / nv
                    lam_max = nv
            eta = 1.0 / (lam_max + 1e-12)
            viol_max = 1.0  # force the ladder again next round
        active = int((lam > 1e-9).sum().item())
        if viol_max > 1e-6:
            self._cstr_skips += 1
            print("[wiki-rpbe-cstr] group=%d CERT_FAIL max_viol=%.3e "
                  "skipping repr step" % (group_start // group_k,
                                          viol_max), flush=True)
            self._cstr_reset_group_state()
            return False
        # ---- write back: scope gets d, other repr params keep task grad --
        off = 0
        for p in scope:
            n = int(p.numel())
            if p.grad is None:
                p.grad = torch.zeros_like(p, device=dev)
            p.grad.copy_(d[off:off + n].reshape(p.shape).to(p.dtype))
            off += n
        if active > 0:
            self._cstr_active += 1
            self._cstr_corr_ratios.append(
                float((d - t).norm()) / (nt + 1e-30))
        print("[wiki-rpbe-cstr] group=%d N=%d cos_mean=%.3f "
              "frac_below=%.3f active=%d/%d max_viol=%.2e corr/task=%.3f"
              % (group_start // group_k, len(keys),
                 float(np.mean(c_np)), frac_below,
                 active, len(active_keys), viol_max,
                 float((d - t).norm()) / (nt + 1e-30)), flush=True)
        self._cstr_reset_group_state()
        return True

    def _cstr_reset_group_state(self):
        """Clear the per-group accumulators (called at every group close)."""
        self._cstr_scope_task = None
        self._cstr_scope_snap = None
        self._cstr_tree_aux = {}

    def _close_repr_group(self, group_batch_count: int, global_step: int,
                          new_ref_j: dict, param_version: int, stats: dict):
        """Close one macro group: build the next KF reference (KF path),
        step the representation optimizer, advance the version, commit.

        The commit happens strictly AFTER the optimizer step, so the new
        reference always lags the updated parameters by exactly one
        version.  Returns ``(new_param_version, refreshes)``; collects
        below-threshold and pending-tree bookkeeping into ``stats``.
        """
        refreshes = 0
        if self.kf_on:
            diagnostics, refreshed = self.kf_window.close_group()
            refreshes = len(refreshed)
            for tau, diag in diagnostics.items():
                if diag.get("failed"):
                    self.monitor.alert(
                        "warning", "kf_reference_failed",
                        "{} {} scale_z={:.3e} scale_p={:.3e} trees={}".format(
                            tau, diag["failed"],
                            diag.get("scale_z", float("nan")),
                            diag.get("scale_p", float("nan")),
                            diag["M_unique_trees"]),
                        step=global_step, interface=tau)
                if diag.get("below_threshold"):
                    # Never silently discard: a window below threshold
                    # means the group length is too short for the real
                    # valid-root rate.
                    stats["below_threshold_groups"] += 1
                    stats["pending_trees"][tau] = diag["M_unique_trees"]
                    stats["threshold"][tau] = diag["threshold"]
                    self.monitor.alert(
                        "warning", "kf_below_threshold",
                        "{} window {} trees < threshold {}; discarded. "
                        "Consider raising kf_group_batches.".format(
                            tau, diag["M_unique_trees"], diag["threshold"]),
                        step=global_step, interface=tau)
                if diag.get("score") is not None:
                    new_ref_j[tau] = diag["score"]
        for p in self.repr_params:
            if p.grad is not None:
                p.grad.div_(float(max(1, group_batch_count)))
        self._clip(self.repr_params)
        self.repr_optimizer.step()
        param_version += 1
        if self.kf_on:
            stale = self.kf_window.commit_reference(
                current_version=param_version)
            for tau in stale:
                self.monitor.alert(
                    "warning", "kf_reference_stale",
                    "{} candidate spans several parameter versions; "
                    "discarded".format(tau),
                    step=global_step, interface=tau)
        return param_version, refreshes

    def _save_group_state(self) -> dict:
        """Everything the two-pass replay must restore (review P2): host
        memory + last_update + messages, all RNGs, the adapter occurrence
        counter and any model buffers (running stats).  Parameters are
        NOT saved — pass 1 performs no optimizer step."""
        modules = {"tgn": self.tgn, "adapter": self.adapter,
                   "compressor": getattr(self.adapter, "compressor", None)}
        buffers = {}
        for mname, module in modules.items():
            if module is None:
                continue
            for name, buf in module.named_buffers():
                buffers["{}.{}".format(mname, name)] = \
                    buf.detach().clone()
        return {
            "memory": (self.tgn.memory.backup_memory()
                       if self.tgn.use_memory else None),
            "rng": ckpt._rng_state(),
            "next_oid": getattr(self.adapter, "_next_oid", 0),
            "buffers": buffers,
        }

    def _restore_group_state(self, state: dict) -> None:
        if self.tgn.use_memory and state["memory"] is not None:
            self.tgn.memory.restore_memory(state["memory"])
        ckpt._restore_rng(state["rng"])
        if hasattr(self.adapter, "_next_oid"):
            self.adapter._next_oid = state["next_oid"]
        modules = {"tgn": self.tgn, "adapter": self.adapter,
                   "compressor": getattr(self.adapter, "compressor", None)}
        for mname, module in modules.items():
            if module is None:
                continue
            for name, buf in module.named_buffers():
                buf.copy_(state["buffers"]["{}.{}".format(mname, name)])

    def _batch_surrogate_exact(self, cuts, replay_plan, group_k,
                               batch_offset):
        """Pass-2 surrogate: numerically zero, gradient = exact J.

        Matches THIS batch's traced z to its replay gradients by (tau,
        occurrence_id); ``by_batch[batch_offset]`` is the plan slice for
        the current batch of the group (pass-1 and pass-2 share RNG and
        the occurrence counter, so ids agree exactly).
        ``S = sum_tau c_tau sum_v <sg(g), z> - sg(<g,z>)`` and the
        auxiliary is ``-lambda * K * S`` (K cancels the common repr
        grad /= K at group close).  Returns (auxiliary, n_terms,
        (n_planned, n_matched)).
        """
        z_by_oid = {}
        tree_by_oid = {}
        for cut in cuts:
            key = (cut.tau, cut.occurrence_id)
            z_by_oid[key] = cut.z
            tree_by_oid[key] = int(cut.tree_id)
        terms = []
        term_keys = []
        n_terms = 0
        n_planned = 0
        for tau, plan in replay_plan.items():
            by_batch = plan.get("by_batch") or []
            if batch_offset >= len(by_batch):
                continue
            for oid, g in by_batch[batch_offset]:
                n_planned += 1
                z = z_by_oid.get((tau, oid))
                if z is None:
                    continue
                gd = g.detach()
                terms.append(self._tau_coeff[tau] * (
                    (gd * z).sum() - (gd * z.detach()).sum()))
                term_keys.append((tree_by_oid[(tau, oid)], tau))
                n_terms += 1
        if not terms:
            return (torch.zeros((), device=self.device), 0,
                    (n_planned, 0), [], [])
        auxiliary = -self.lambda_kf * float(group_k) * sum(terms)
        return auxiliary, n_terms, (n_planned, n_terms), term_keys, terms

    def _train_epoch_exact_replay(self, epoch: int, global_step: int,
                                  train: object, max_batches: int = None
                                  ) -> Dict:
        """Same-window two-pass exact replay (review P2).

        Pass 1 (no grad): the macro group's batches replay
        chronologically (memory advancing, decoder forward for RNG
        parity), cuts accumulate into KFMomentWindow; close_replay()
        contracts the exact window J onto cut-level z-adjoints.

        Restore: memory/messages, RNG, occurrence counter, buffers.

        Pass 2 (grad): the same batches replay; each batch's traced z
        matches its replay gradient by (tau, occurrence_id) and the
        numerically-zero surrogate carries the EXACT J gradient.
        loss = task - lambda * K * S_b with K cancelling the group-close
        ``repr_grad /= K``.  The head steps every batch; the
        representation group steps once at the group end (same cadence
        as the lambda=0 grouped baseline).
        """
        self.reset_memory()
        self.kf_window.reset()
        self.tgn.train(self.finetune_host)
        self.decoder.train()

        if self.rpbe_cfg.kf_group_batches is not None:
            fixed_group = max(1, int(self.rpbe_cfg.kf_group_batches))
        else:
            fixed_group = max(1, math.ceil(
                self.rpbe_cfg.kf_min_abs / max(1, self.trace_roots)))
        num_batch = math.ceil(len(train.sources) / self.batch_size)
        run_batches = num_batch if max_batches is None else min(
            num_batch, max(0, int(max_batches)))

        total_task = 0.0
        total_raw = 0.0
        kf_sum = {}
        kf_count = {}
        refreshes = 0
        aux_batches = 0
        below_threshold_groups = 0
        pending_trees = {}
        threshold = {}
        param_version = 0
        total_planned = 0
        total_matched = 0
        train_probs, train_labels = [], []

        def _run_one_pass(batch_index, grad_enabled):
            start = batch_index * self.batch_size
            stop = min(len(train.sources),
                       (batch_index + 1) * self.batch_size)
            sources = train.sources[start:stop]
            destinations = train.destinations[start:stop]
            timestamps = train.timestamps[start:stop]
            edge_idxs = train.edge_idxs[start:stop]
            labels_np = train.labels[start:stop]
            trace_rows = select_trace_rows(
                labels_np, self.trace_roots, self.seed, global_step + (
                    batch_index - group_start), self.trace_mode)
            self.adapter.set_trace_source_rows(trace_rows)
            context = torch.enable_grad() if grad_enabled else torch.no_grad()
            with context:
                src_emb, _, _ = self.tgn.compute_temporal_embeddings(
                    sources, destinations, destinations, timestamps,
                    edge_idxs, self.n_neighbors)
                prediction = self.decoder(src_emb)
            cuts = None
            if self.adapter.trace is not None:
                cuts = self.cut_builder.build(
                    self.adapter.trace, batch_seed=global_step)
            self.adapter.clear_trace()
            return prediction, cuts, labels_np

        group_start = 0
        while group_start < run_batches:
            group_end = min(group_start + fixed_group, run_batches)
            group_k = group_end - group_start
            # ---------------- pass 1: collect, no grad ----------------
            state = self._save_group_state()
            with torch.no_grad():
                for b in range(group_start, group_end):
                    _, cuts, _ = _run_one_pass(b, grad_enabled=False)
                    if cuts:
                        self.kf_window.add(cuts)
            closed, replay_plan, group_diag = \
                self.kf_window.close_replay()
            refreshes += len(closed)
            for tau, score in closed.items():
                kf_sum[tau] = kf_sum.get(tau, 0.0) + score
                kf_count[tau] = kf_count.get(tau, 0) + 1
                total_raw += score
            for tau, diag in group_diag.items():
                if diag.get("below_threshold"):
                    below_threshold_groups += 1
                    pending_trees[tau] = diag["M_unique_trees"]
                    threshold[tau] = diag["threshold"]
                    msg = ("{} window {} trees < threshold {}; discarded "
                           "(exact_replay)".format(
                               tau, diag["M_unique_trees"],
                               diag["threshold"]))
                    if self.kf_fail_below_threshold:
                        raise RuntimeError(
                            "kf_below_threshold fail-fast: " + msg)
                    self.monitor.alert(
                        "warning", "kf_below_threshold", msg,
                        step=global_step, interface=tau)
            # -------------------- restore to group start --------------
            self._restore_group_state(state)
            # ---------------- pass 2: train, exact surrogate ----------
            self.repr_optimizer.zero_grad(set_to_none=True)
            if self.rpbe_constrain and not self._cstr_scope_params:
                comp = getattr(self.adapter, "compressor", None)
                if comp is not None:
                    self._cstr_scope_params = [
                        p for p in comp.parameters() if p.requires_grad]
            for b in range(group_start, group_end):
                self.head_optimizer.zero_grad(set_to_none=True)
                prediction, cuts, labels_np = _run_one_pass(
                    b, grad_enabled=True)
                prediction = prediction.sigmoid()
                labels_t = torch.from_numpy(labels_np).float().to(
                    self.device)
                task_loss = F.binary_cross_entropy(prediction, labels_t)
                auxiliary = torch.zeros((), device=self.device)
                term_keys = []
                aligned_planned = 0
                aligned_matched = 0
                if cuts and closed:
                    (auxiliary, n_terms,
                     (aligned_planned, aligned_matched),
                     term_keys, term_pieces) = \
                        self._batch_surrogate_exact(
                            cuts, replay_plan, group_k, b - group_start)
                    if n_terms:
                        aux_batches += 1
                else:
                    term_pieces = []
                if self.rpbe_constrain and self._cstr_scope_params:
                    # Constrained RPBE: separate the two components.
                    # task backward first (retain the graph for the
                    # per-tree VJPs below), accumulate the SCOPE task
                    # gradient by batch delta, then one VJP per
                    # (tree, tau) on the raw surrogate pieces (the
                    # surrogate is 0-valued; only its graph matters).
                    # The auxiliary's aggregate backward is NOT needed —
                    # per-tree directions come from autograd.grad.
                    task_loss.backward(retain_graph=True)
                    self._cstr_accum_scope_task()
                    if term_keys:
                        self._cstr_accum_tree_dirs(term_keys, term_pieces)
                else:
                    # Per-batch gradient diagnostics (the sprint script),
                    # same hook as the single-pass path: r_eff =
                    # |grad(aux)| / |grad(task)| carries lambda and the
                    # rank coefficients already.
                    if self._grad_diag_fn is not None and self.kf_on \
                            and auxiliary.requires_grad and self.repr_params:
                        row = self._grad_diag_fn(
                            self, task_loss, auxiliary, global_step)
                        self.grad_diag["rows"].append(row)
                    loss = task_loss + auxiliary
                    self.monitor.validate_losses({
                        "task": float(task_loss.detach()),
                        "kf_score": total_raw,
                        "main_total": float(loss.detach()),
                    }, global_step)
                    loss.backward()
                self._clip(self.head_params)
                self.head_optimizer.step()
                if self.tgn.use_memory and \
                        (self.finetune_host or self.component_on):
                    self.tgn.memory.detach_memory()
                total_task += float(task_loss.detach())
                total_planned += aligned_planned
                total_matched += aligned_matched
                train_probs.append(prediction.detach().cpu().numpy())
                train_labels.append(labels_np)
                global_step += 1
            # ---------------- group close: one repr step ---------------
            if self.rpbe_constrain and self._cstr_scope_params:
                if not self._cstr_group_close(group_k, group_start):
                    # certificate failed after escalation: SKIP the
                    # representation step for this group (no constrained
                    # update is executed; the next group starts fresh).
                    group_start = group_end
                    continue
            for p in self.repr_params:
                if p.grad is not None:
                    p.grad.div_(float(max(1, group_k)))
            self._clip(self.repr_params)
            self.repr_optimizer.step()
            param_version += 1
            group_start = group_end

        train_metrics = metric_bundle(np.concatenate(train_labels),
                                      np.concatenate(train_probs))
        train_metrics["online_auc"] = train_metrics.pop("auc")
        train_metrics["online_ap"] = train_metrics.pop("ap")
        kf_out = None
        if kf_count and self.rpbe_cfg is not None:
            means = {tau: value / kf_count[tau]
                     for tau, value in kf_sum.items()}
            kf_out = {
                "estimator": "same_window_two_pass_exact_replay",
                "J": means,
                "J_norm": {tau: value / self._tau_rank(tau)
                           for tau, value in means.items()},
                "skipped_types": [],
                "kf_score": total_raw / max(run_batches, 1),
                "kf_loss": -self.lambda_kf * total_raw
                           / max(run_batches, 1),
                "reference_refreshes": refreshes,
                "below_threshold_groups": below_threshold_groups,
                "pending_trees": dict(pending_trees),
                "threshold": dict(threshold),
                "group_batches": fixed_group,
                "aux_batches": aux_batches,
                "repr_steps": param_version,
                # Gate 1: pass-1/pass-2 occurrence alignment (must be
                # 100%: duplicate = missing = 0).
                "replay_align_planned": total_planned,
                "replay_align_matched": total_matched,
                "replay_align_missing": total_planned - total_matched,
            }
        return {
            "train_task_loss": total_task / max(run_batches, 1),
            "train_kf_score": total_raw / max(run_batches, 1),
            "kf": kf_out,
            "train": train_metrics,
            "n_batches": run_batches,
            "global_step": global_step,
        }

    # ----------------------------------------------------------------- training
    def train_epoch(self, epoch: int, global_step: int, train: object,
                    max_batches: int = None) -> Dict:
        """One chronological epoch.

        Macro-group timing (eighth review C): representation parameters
        (host + compressor) are frozen within a macro-group and update once
        per FIXED group of ceil(kf_min_abs / trace_roots) batches — the
        same cadence for the KF path and the lambda=0 task-only fast path
        (fair ablation, review D).  The head updates every batch.  One
        query, one backward, one head step per batch — no replay, no
        cross-batch graphs.

        ``max_batches`` (the sprint diagnostic) stops the epoch after N
        batches and drains that final, possibly partial macro-group.
        """
        if self.kf_on and self.kf_estimator == "exact_replay":
            return self._train_epoch_exact_replay(
                epoch, global_step, train, max_batches)
        self.reset_memory()
        if self.kf_window is not None:
            self.kf_window.reset(clear_reference=True)   # eighth review C
        self.tgn.train(self.finetune_host)
        self.decoder.train()

        # Fixed group length shared by both component protocols.  The
        # default ceil(kf_min_abs / trace_roots) assumes every traced
        # root contributes a cut; the real valid-root rate is lower
        # (strict-future masking), so kf_group_batches overrides it.  A
        # pure host finetune (no compressor) updates every batch.
        if self.component_on:
            if self.rpbe_cfg.kf_group_batches is not None:
                fixed_group = max(1, int(self.rpbe_cfg.kf_group_batches))
            else:
                fixed_group = max(1, math.ceil(
                    self.rpbe_cfg.kf_min_abs / max(1, self.trace_roots)))
        else:
            fixed_group = 1

        total_task = 0.0
        total_raw = 0.0
        total_norm = 0.0
        kf_sum = {}
        kf_count_tau = {}
        new_ref_j = {}
        cold_types = set()
        refreshes = 0
        aux_batches = 0
        no_cuts_alerted = False
        param_version = 0
        repr_group_active = False
        group_batch_count = 0
        group_stats = {"below_threshold_groups": 0,
                       "pending_trees": {}, "threshold": {}}
        train_probs, train_labels = [], []
        num_batch = math.ceil(len(train.sources) / self.batch_size)
        run_batches = num_batch if max_batches is None else min(
            num_batch, max(0, int(max_batches)))
        group_target_count = 1

        for batch_index in range(num_batch):
            if max_batches is not None and batch_index >= max_batches:
                break
            start = batch_index * self.batch_size
            stop = min(len(train.sources), (batch_index + 1) * self.batch_size)
            sources = train.sources[start:stop]
            destinations = train.destinations[start:stop]
            timestamps = train.timestamps[start:stop]
            edge_idxs = train.edge_idxs[start:stop]
            labels_np = train.labels[start:stop]
            labels_t = torch.from_numpy(labels_np).float().to(self.device)

            self.head_optimizer.zero_grad(set_to_none=True)
            # The macro-group machinery runs whenever representation
            # parameters exist (compressor attached OR host finetune);
            # only a fully frozen vanilla run keeps it silent.
            if not repr_group_active and self.repr_optimizer is not None \
                    and (self.component_on or self.repr_train_on):
                self.repr_optimizer.zero_grad(set_to_none=True)
                if self.kf_on:
                    self.kf_window.begin_group(param_version, epoch)
                repr_group_active = True
                group_batch_count = 0
                group_target_count = min(
                    fixed_group, max(1, run_batches - batch_index))

            trace_rows = []
            if self.kf_on:
                trace_rows = select_trace_rows(
                    labels_np, self.trace_roots, self.seed, global_step,
                    self.trace_mode)
                self.adapter.set_trace_source_rows(trace_rows)
            elif self.adapter is not None:
                self.adapter.clear_trace()

            src_emb, _, _ = self._full_official_embedding_call(
                sources, destinations, timestamps, edge_idxs,
                grad_enabled=(self.finetune_host or self.component_on))
            prediction = self.decoder(src_emb).sigmoid()
            task_loss = F.binary_cross_entropy(prediction, labels_t)
            auxiliary = torch.zeros((), device=self.device)
            raw_score = 0.0
            norm_score = 0.0

            if self.kf_on and trace_rows and self.adapter.trace is not None:
                cuts = self.cut_builder.build(
                    self.adapter.trace, batch_seed=global_step)
                if cuts:
                    (raw_score, norm_score, scores, auxiliary,
                     cold) = self._consume_kf(cuts, global_step)
                    cold_types.update(cold)
                    if auxiliary.requires_grad:
                        # Raw per-batch moment VJPs SUM to the macro-window
                        # linearization.  _close_repr_group divides every
                        # accumulated representation gradient by K to average
                        # the task loss, so multiply this zero-valued
                        # auxiliary by the actual group K to leave the KF
                        # window gradient unshrunk.  This is exact even when
                        # batches have unequal valid-cut weights.
                        auxiliary = auxiliary * float(group_target_count)
                        aux_batches += 1
                    for tau, score in scores.items():
                        kf_sum[tau] = kf_sum.get(tau, 0.0) + score
                        kf_count_tau[tau] = kf_count_tau.get(tau, 0) + 1
                elif not no_cuts_alerted:
                    self.monitor.alert(
                        "warning", "kf_no_future_rows",
                        "traced cuts have no strictly-future train outcomes",
                        step=global_step)
                    no_cuts_alerted = True

            # Per-batch gradient diagnostics (the sprint script): task
            # vs KF gradients on the representation parameters, before
            # the single backward.  Implemented outside this module.
            if self._grad_diag_fn is not None and self.kf_on \
                    and auxiliary.requires_grad and self.repr_params:
                row = self._grad_diag_fn(
                    self, task_loss, auxiliary, global_step)
                self.grad_diag["rows"].append(row)

            loss = task_loss + auxiliary
            self.monitor.validate_losses({
                "task": float(task_loss.detach()),
                "kf_score": raw_score,
                "kf_normalized": norm_score,
                "main_total": float(loss.detach()),
            }, global_step)
            loss.backward()
            self._clip(self.head_params)
            self.head_optimizer.step()
            if self.tgn.use_memory and \
                    (self.finetune_host or self.component_on):
                self.tgn.memory.detach_memory()

            total_task += float(task_loss.detach())
            total_raw += raw_score
            total_norm += norm_score
            train_probs.append(prediction.detach().cpu().numpy())
            train_labels.append(labels_np)
            group_batch_count += 1
            global_step += 1

            # Macro-group close (eighth review C): FIXED cadence shared by
            # the KF path and the lambda=0 task-only path —
            # ceil(kf_min_abs / trace_roots) batches — so both protocols
            # share the same representation-update timing (fair ablation,
            # review D).  The window's threshold gate lives in
            # close_group, which discards below-threshold windows.
            if repr_group_active and group_batch_count >= fixed_group:
                param_version, refs = self._close_repr_group(
                    group_batch_count, global_step, new_ref_j, param_version,
                    group_stats)
                refreshes += refs
                repr_group_active = False

        # Epoch drain: the same close sequence for the unfinished group
        # (close + commit + one representation step), so the reference
        # lifecycle never crosses an epoch boundary.
        if repr_group_active:
            param_version, refs = self._close_repr_group(
                group_batch_count, global_step, new_ref_j, param_version,
                group_stats)
            refreshes += refs
            self.repr_optimizer.zero_grad(set_to_none=True)

        train_metrics = metric_bundle(np.concatenate(train_labels),
                                      np.concatenate(train_probs))
        train_metrics["online_auc"] = train_metrics.pop("auc")
        train_metrics["online_ap"] = train_metrics.pop("ap")
        if self.train_eval_auc:
            backup = self.tgn.memory.backup_memory() \
                if self.tgn.use_memory else None
            eval_row = self.evaluate_split(train, reset=True)
            if backup is not None:
                self.tgn.memory.restore_memory(backup)
            train_metrics["eval_auc"] = eval_row["auc"]
            train_metrics["eval_ap"] = eval_row["ap"]

        kf_out = None
        if kf_count_tau and self.rpbe_cfg is not None:
            means = {tau: value / kf_count_tau[tau]
                     for tau, value in kf_sum.items()}
            kf_out = {
                "estimator": "one_pass_lagged_moment_adjoint",
                # J/J_norm are the ACTIVE reference values (honest
                # monitoring, never the current model's score).
                "J": means,
                "J_norm": {tau: value / self._tau_rank(tau)
                           for tau, value in means.items()},
                # The NEXT reference built at this epoch's group closes
                # (None where the group was below threshold).
                "J_new": dict(new_ref_j),
                "reference_age": {
                    tau: self.kf_window.reference_age(tau)
                    for tau in means},
                "skipped_types": sorted(cold_types),
                "kf_score": total_raw / max(num_batch, 1),
                "kf_normalized": total_norm / max(num_batch, 1),
                "kf_loss": -self.lambda_kf * total_norm / max(num_batch, 1),
                "p_cache_hits": getattr(self.fixed_maps, "_p_cache_hits", 0),
                "p_cache_misses": getattr(
                    self.fixed_maps, "_p_cache_misses", 0),
                "reference_refreshes": refreshes,
                "stale_drops": getattr(self.kf_window, "stale_drops", 0),
                "below_threshold_groups": group_stats[
                    "below_threshold_groups"],
                "pending_trees": dict(group_stats["pending_trees"]),
                "threshold": dict(group_stats["threshold"]),
                "group_batches": fixed_group,
                "aux_batches": aux_batches,
                "repr_steps": param_version,
            }
        return {
            "train_task_loss": total_task / max(num_batch, 1),
            "train_kf_score": total_raw / max(num_batch, 1),
            "train_kf_normalized": total_norm / max(num_batch, 1),
            "kf": kf_out,
            "train": train_metrics,
            "n_batches": num_batch,
            "global_step": global_step,
        }

    # --------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate_split(self, split: object, *, reset: bool = False) -> Dict:
        """AUC/AP/NLL over one split; advances memory with ground truth."""
        if reset:
            self.reset_memory()
        before = rpbe_fingerprint(self.fixed_maps)
        self.tgn.eval()
        self.decoder.eval()
        if self.adapter is not None:
            self.adapter.clear_trace()
        probs, labels = [], []
        observed_dims = None
        for batch_index in range(math.ceil(len(split.sources)
                                           / self.batch_size)):
            start = batch_index * self.batch_size
            stop = min(len(split.sources),
                       (batch_index + 1) * self.batch_size)
            src_emb, _, _ = self._full_official_embedding_call(
                split.sources[start:stop], split.destinations[start:stop],
                split.timestamps[start:stop], split.edge_idxs[start:stop],
                grad_enabled=False)
            if observed_dims is None:
                observed_dims = {"source": int(src_emb.shape[-1])}
            probs.append(self.decoder(src_emb).sigmoid().cpu().numpy())
            labels.append(split.labels[start:stop])
        trace_created = bool(self.adapter is not None
                             and self.adapter.trace is not None)
        assert_clean(before, self.fixed_maps, trace_created, "evaluate_split")
        out = metric_bundle(np.concatenate(labels), np.concatenate(probs))
        out["embedding_dims_observed"] = observed_dims or {}
        return out

    @torch.no_grad()
    def replay_split(self, split: object) -> None:
        """Rebuild memory from zero over a split, with no scores."""
        before = rpbe_fingerprint(self.fixed_maps)
        self.tgn.eval()
        if self.adapter is not None:
            self.adapter.clear_trace()
        for batch_index in range(math.ceil(len(split.sources)
                                           / self.batch_size)):
            start = batch_index * self.batch_size
            stop = min(len(split.sources),
                       (batch_index + 1) * self.batch_size)
            self._full_official_embedding_call(
                split.sources[start:stop], split.destinations[start:stop],
                split.timestamps[start:stop], split.edge_idxs[start:stop],
                grad_enabled=False)
        trace_created = bool(self.adapter is not None
                             and self.adapter.trace is not None)
        assert_clean(before, self.fixed_maps, trace_created, "replay_split")
