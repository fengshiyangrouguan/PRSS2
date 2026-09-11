"""TGB tgbl-wiki link training with pair-window exact replay (4 arms).

Joint training from step 0::

    L = L_link + lambda * L_PRBE

* L_link  : official positive/negative BCE over real train edges with
            uniform-over-destination train negatives (never in Y / PRBE).
* L_PRBE  : same-window two-pass exact-replay Ky Fan over child-parent
            boundary records; one (z, p) row per pair; per-tree weight 1.

Arms:
    gamma_task_only : lambda=0, host+Gamma trained on L_link only
    1obs            : pairs with m_p=0 (child marginal only)
    2obs_aligned    : pairs with m_p=1 (true parent future)
    2obs_mispaired  : pairs with m_p=1 and a deranged parent future

Pass structure per macro group (fixed ``group_batches``):
    pass 1 (no grad): advance memory, collect adapter pairs -> boundary
        records -> per-tau PairKFWindow; close_replay -> per-position adjoints
    restore memory / RNG / oid
    pass 2 (grad): same batches, L_link + linear surrogate over live pairs;
        head steps every batch; the representation group (host+Gamma) steps
        once per group.

Evaluation uses the official TGB val/test negative sampler and MRR (separate
runner), not this loop.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from rpbe.loss import _score_from_covs
from rpbe.pair_window import PairKFWindow, tree_equal_weights
from rpbe.training import checkpoint as ckpt
from rpbe.training.isolation import assert_clean, rpbe_fingerprint
from rpbe.training.jodie_loop import select_trace_rows
from rpbe.link_records import build_boundary_records
from rpbe.pair_arms import (feasible_positions,
                            build_mispaired_parent_map)
from rpbe.audit import AuditAccumulator

ARMS = ("gamma_task_only", "1obs", "2obs_aligned", "2obs_mispaired")


def arm_use_parent(arm: str) -> int:
    return 1 if arm in ("2obs_aligned", "2obs_mispaired") else 0


def arm_kf_on(arm: str, lambda_kf: float) -> bool:
    return arm != "gamma_task_only" and lambda_kf > 0.0


def _flat_grad(gs, params):
    """Flatten an ``autograd.grad`` list into one vector aligned with
    ``params``; a None entry becomes zeros so the task and aux components
    (which may touch different parameter subsets) stay dimension-matched."""
    parts = []
    for g, p in zip(gs, params):
        if g is None:
            parts.append(torch.zeros_like(p, device="cpu",
                                          dtype=torch.float32).reshape(-1))
            continue
        parts.append(g.detach().reshape(-1).float().cpu())
    if not parts:
        return None
    return torch.cat(parts)


def _grad_cosine(params, g1, g2):
    """Cosine between two flat gradient lists aligned on ``params``;
    returns (cos, has1, has2)."""
    a = _flat_grad(g1, params)
    b = _flat_grad(g2, params)
    if a is None or b is None:
        return 0.0, False, False
    an = float(a.norm())
    bn = float(b.norm())
    if an < 1e-12 or bn < 1e-12:
        return 0.0, False, False
    return float((a * b).sum() / (an * bn)), True, True


class TGBPairLinkLoop:
    """One recursive TGN host + pair exact-replay trainer for one arm."""

    def __init__(self, *, tgn, device, batch_size, n_neighbors, grad_clip,
                 monitor, seed, adapter, link_future_index, boundary_maps,
                 edge_table, arm, rpbe_cfg, repr_optimizer, head_optimizer,
                 trace_roots=32, trace_pairs_per_parent=2,
                 kf_group_batches=56, kf_min_trees=896,
                 n_observations=2, trace_mode="evenly_spaced",
                 fail_below=False, train_neg_sampler=None,
                 audit_trace=False, aux_prefix_groups=None,
                 group_plan_sha=None, use_parent=None, mispaired=None,
                 context_mode="full", variant="full_balancing",
                 aux_kind="kyfan", aux_heads=None, aux_optimizer=None,
                 aux_lambda=None, calibrate_groups=0,
                 memory_grad_probe=False,
                 grad_align_diag=False,
                 rpbe_constrain=False, rpbe_kappa=0.05):
        self.tgn = tgn
        self.device = device
        self.batch_size = int(batch_size)
        self.n_neighbors = int(n_neighbors)
        self.grad_clip = float(grad_clip)
        self.monitor = monitor
        self.seed = int(seed)
        self.adapter = adapter
        self.link_future_index = link_future_index
        self.boundary_maps = boundary_maps
        self.edge_table = edge_table
        self.arm = arm
        self.rpbe_cfg = rpbe_cfg
        self.lambda_kf = float(rpbe_cfg.lambda_kf) if rpbe_cfg else 0.0
        self.repr_optimizer = repr_optimizer
        self.head_optimizer = head_optimizer
        self.repr_params = (list(repr_optimizer.param_groups[0]["params"])
                            if repr_optimizer is not None else [])
        self.head_params = list(head_optimizer.param_groups[0]["params"])
        self.trace_roots = int(trace_roots)
        self.trace_pairs_per_parent = int(trace_pairs_per_parent)
        self.kf_group_batches = int(kf_group_batches)
        self.kf_min_trees = int(kf_min_trees)
        self.n_observations = int(n_observations)
        self.trace_mode = trace_mode
        # config-resolved axes (9-config registry): parent usage, mispaired
        # pairing, auxiliary context mode, score variant, aux supervision kind.
        self.use_parent = (int(use_parent) if use_parent is not None
                           else arm_use_parent(arm))
        self.mispaired = (bool(mispaired) if mispaired is not None
                          else arm == "2obs_mispaired")
        self.context_mode = str(context_mode)
        self.variant = str(variant)
        self.aux_kind = str(aux_kind)   # kyfan / rec / pred / none
        self.aux_heads = aux_heads
        self.aux_optimizer = aux_optimizer
        self.aux_lambda = (float(aux_lambda) if aux_lambda is not None
                           else self.lambda_kf)
        # lambda calibration mode: run up to N eligible macro-groups at a FIXED
        # initialization without stepping any optimizer, capturing compressor
        # task/aux gradient gauges (spec §final item 7).
        self.calibrate_groups = int(calibrate_groups)
        self.calibrate = self.calibrate_groups > 0
        # memory-GRU grad probe: records the first nonzero gradient norm over
        # the memory updater's parameters so runners can hard-assert the
        # cross-batch memory path is actually learning (start-mode only).
        self.memory_grad_probe = bool(memory_grad_probe)
        self._mem_probe_params = (
            self._memory_updater_params() if self.memory_grad_probe else [])
        self._mem_probe_done = False
        self._mem_probe_batches = 0
        self._mem_grad_norm = 0.0
        # Gradient-alignment DIAGNOSTIC (reviewer item 5, phase 3 — record
        # only, never gate).  Measures, per macro group, the cosine between
        # the TASK update direction and the RPBE update direction on the
        # same repr params, same batch, same theta:
        #
        #   d_task  = -g_t            (L_link is minimized)
        #   d_rpbe  = -g_a ∝ +grad J  (the surrogate minimizes -lambda*J;
        #                               g_a = -lambda*group_k * grad_theta J)
        #
        # Both objectives are MINIMIZED, so cos(d_task, d_rpbe) equals
        # cos(g_t, g_a) — the two sign flips cancel — and we record
        # cos(g_t, g_a) directly.  Do NOT read this cosine as
        # cos(grad L_task, grad J): that pair would flip the sign.
        self.grad_align_diag = bool(grad_align_diag)
        self._ga_cos = []      # per-group update-direction cosines
        self._ga_ratios = []   # per-group ||d_rpbe|| / ||d_task||
        # Plan B (task-primary constrained RPBE, reviewer-approved): at each
        # macro-group boundary the task update direction t = -g_t is
        # projected into the half space {(grad J)^T d >= -kappa||g_a||||g_t||}
        # where g_a = grad L_aux = -lambda*group_k * grad J, so the trigger
        #   g_a^T g_t < -kappa ||g_a|| ||g_t||
        # and the write-back
        #   g_write = g_t + mu * g_a,  mu = (-kappa||ga||||gt|| - ga^T gt)/||ga||^2
        # are lambda-INVARIANT (the two lambda factors cancel).  When the
        # trigger does not fire, the aux component is DISCARDED and the pure
        # task gradient is used (d = t) — the orthogonal RPBE updates no
        # longer pollute the step.  Head params never see the projection.
        self.rpbe_constrain = bool(rpbe_constrain)
        self.rpbe_kappa = float(rpbe_kappa)
        self._cstr_task_acc = None   # per-repr-param task grad accumulators
        self._cstr_aux_acc = None    # per-repr-param aux grad accumulators
        self._cstr_snap = None       # per-batch task snapshot (aux = delta)
        self._cstr_active = 0
        self._cstr_groups = 0
        self._cstr_corr_ratios = []
        self.kf_on = self.aux_kind in ("kyfan", "rec", "pred") \
            and self.lambda_kf > 0
        self.fail_below = bool(fail_below)
        # audit side-channel: when True the trace runs even without KF so
        # gamma_task_only records the SAME underlying population as the aux
        # arms (spec §24.5 acceptance).  Pure record-keeping: trace row
        # selection is deterministic (evenly_spaced, no RNG) and the audit
        # touches no tensors and no optimizer state.
        self.audit_trace = bool(audit_trace)
        self.audit = AuditAccumulator()
        self._mispaired_map: Dict[int, object] = {}
        # prefix-causal train negative sampler (root task only); when None a
        # uniform-over-destination-universe fallback is used.
        self.train_neg_sampler = train_neg_sampler
        self._dst_univ = None
        # shared group plan (spec §1.1-final): aux_prefix_groups = number of
        # leading macro groups eligible for the auxiliary windows; groups after
        # it are a task-only censored tail (never counted as n_below).  When
        # None the legacy per-group below/fail behaviour is unchanged.
        self.aux_prefix_groups = (None if aux_prefix_groups is None
                                  else int(aux_prefix_groups))
        self.group_plan_sha = group_plan_sha
        self.n_censored_tail_groups = 0
        # canonical pair child taus == the adapter's compressible internal
        # layers (layer 1..n_layers-1); every eligible group must produce them.
        self._canon_taus = list(getattr(
            self.adapter, "compression_taus", []) or [])
        # gradient gauges (fairness gates): compressor grad norms / param delta
        comp = getattr(self.adapter, "compressor", None)
        self._comp_params = list(comp.parameters()) if comp is not None else []
        self._gauge_group_aux = []
        self._gauge_group_task = []
        self._gauge_param_delta = None
        self._gauge_pd_snapshot = None

        # per-interface PairKFWindow (child tau keys)
        self.pair_windows: Dict[str, PairKFWindow] = {}
        self._window_eps = float(rpbe_cfg.ridge_eps) if rpbe_cfg else 1e-3
        # window diagnostics: per group per tau M_unique_trees / threshold /
        # actual group batch count (used to calibrate the macro-group length
        # from real child-parent pair yield on tgbl-wiki v2).
        self.window_diag = []

        if self.kf_on:
            from rpbe.pair_rows import PairRowProjector
            self.projector = PairRowProjector(
                boundary_maps, edge_table, use_parent=self.use_parent,
                d_ctx=int(getattr(rpbe_cfg, "d_c", 32)))
        else:
            self.projector = None
            # gamma_task_only: still needs the projector? no KF -> not used.

    # ------------------------------------------------------------- stream state
    def reset_memory(self):
        if self.tgn.use_memory:
            self.tgn.memory.__init_memory__()

    def _message_for(self, edge_id):
        row = self.edge_table[int(edge_id)]
        return torch.as_tensor(row, dtype=torch.float32, device=self.device)

    # ------------------------------------------------ record -> p row helpers
    def _records_from_trace(self, trace):
        """Build valid BoundaryRecords from the adapter's consumed pairs."""
        if trace is None or not trace.pairs:
            return []
        recs = build_boundary_records(trace.pairs, self.link_future_index)
        return recs

    def _pv(self, rec):
        """One p_v row for a record under this arm's use_parent.

        The mispaired arm consumes a donor parent future (position-level
        perfect derangement built in pass 1); every other arm consumes the
        true parent future.  ``parent_future_used`` is attached to the
        pass-1 record object only — the record's own fields are untouched.
        """
        ctx = _ctx_vec(rec, d_ctx=self._ctx_dim(), device=self.device)
        if self.context_mode == "constant":
            # C1 no-C: the aux witness drops chi(C) (zero context block) while
            # keeping the intercept / p dim / fixed sketch unchanged.
            ctx = torch.zeros_like(ctx)
        child = self._event(rec.child_future, rec.child_time)
        parent = self._event(
            getattr(rec, "parent_future_used", rec.parent_future),
            rec.parent_time)
        return self.boundary_maps.pv(ctx, child, parent,
                                     use_parent=self.use_parent)

    def _event(self, future, cut_time):
        return {
            "counterpart": future.counterpart,
            "role": future.role,
            "delta_t": float(future.time) - float(cut_time),
            "message": self._message_for(future.message_idx),
        }

    def _ctx_dim(self):
        return int(getattr(self.rpbe_cfg, "d_c", 32)) \
            if self.rpbe_cfg else 32

    # --------------------------------------------------------------- windowing
    def _win(self, tau):
        if tau not in self.pair_windows:
            self.pair_windows[tau] = PairKFWindow(
                tau=tau, eps=self._window_eps,
                min_unique_trees=self.kf_min_trees,
                variant=self.variant)
        return self.pair_windows[tau]

    def _save_group_state(self):
        """Memory / RNG / occurrence counters (mirrors jodie_loop)."""
        modules = {"tgn": self.tgn, "adapter": self.adapter,
                   "compressor": getattr(self.adapter, "compressor", None)}
        buffers = {}
        for mname, module in modules.items():
            if module is None:
                continue
            for name, buf in module.named_buffers():
                buffers["{}.{}".format(mname, name)] = buf.detach().clone()
        return {
            "memory": (self.tgn.memory.backup_memory()
                       if self.tgn.use_memory else None),
            "rng": ckpt._rng_state(),
            "next_oid": getattr(self.adapter, "_next_oid", 0),
            "next_pair_id": getattr(self.adapter, "_next_pair_id", 0),
            "buffers": buffers,
        }

    def _restore_group_state(self, state):
        if self.tgn.use_memory and state["memory"] is not None:
            self.tgn.memory.restore_memory(state["memory"])
        ckpt._restore_rng(state["rng"])
        if hasattr(self.adapter, "_next_oid"):
            self.adapter._next_oid = state["next_oid"]
        if hasattr(self.adapter, "_next_pair_id"):
            self.adapter._next_pair_id = state["next_pair_id"]
        modules = {"tgn": self.tgn, "adapter": self.adapter,
                   "compressor": getattr(self.adapter, "compressor", None)}
        for mname, module in modules.items():
            if module is None:
                continue
            for name, buf in module.named_buffers():
                buf.copy_(state["buffers"]["{}.{}".format(mname, name)])

    # ------------------------------------------------------------- negative
    def _sample_negatives(self, sources, dst_pos, timestamps, global_rows,
                          epoch=None):
        """HistRand (prefix-causal) negatives for the CURRENT batch rows."""
        if self.train_neg_sampler is not None:
            return self.train_neg_sampler.sample(
                epoch, 0, sources, dst_pos, timestamps, global_rows)
        size = len(sources)
        univ = self._dst_univ if self._dst_univ is not None \
            else np.asarray(dst_pos)
        return np.random.choice(univ, size=size, replace=True)

    # ---------------------------------------------------------------- forward
    def _run_batch(self, train, batch_index, global_step, grad_enabled,
                   epoch=None):
        start = batch_index * self.batch_size
        stop = min(len(train.sources), (batch_index + 1) * self.batch_size)
        if stop <= start:
            return None
        sources = train.sources[start:stop]
        destinations = train.destinations[start:stop]
        timestamps = train.timestamps[start:stop]
        edge_idxs = train.edge_idxs[start:stop]
        size = len(sources)
        # global event row = internal edge id - 1 (stream rows are contiguous
        # in the FULL dataset; train.eidx are 1-based full-stream rows)
        negatives = self._sample_negatives(
            sources, destinations, timestamps, edge_idxs - 1, epoch)

        trace_rows = []
        if self.kf_on or self.audit_trace:
            trace_rows = select_trace_rows(
                np.zeros(size), self.trace_roots, self.seed, global_step,
                mode=self.trace_mode)
            self.adapter.set_trace_source_rows(trace_rows)
            self.adapter.set_trace_batch(global_step)
        elif self.adapter is not None:
            self.adapter.clear_trace()

        ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with ctx:
            positive, negative = self.tgn.compute_edge_probabilities(
                sources, destinations, negatives, timestamps, edge_idxs,
                self.n_neighbors)
        link_loss = (F.binary_cross_entropy(
            positive.squeeze(), torch.ones(size, device=self.device))
            + F.binary_cross_entropy(
                negative.squeeze(), torch.zeros(size, device=self.device)))
        records = []
        if (self.kf_on or self.audit_trace) and trace_rows \
                and self.adapter.trace is not None:
            records = self._records_from_trace(self.adapter.trace)
        self.adapter.clear_trace()
        return link_loss, records

    # ------------------------------------------------------------ train epoch
    def train_epoch(self, epoch, global_step, train, max_batches=None):
        self.reset_memory()
        self.tgn.train(True)
        if self.train_neg_sampler is None and self._dst_univ is None:
            self._dst_univ = np.unique(np.asarray(train.destinations))
        num_batch = math.ceil(len(train.sources) / self.batch_size)
        run_batches = num_batch if max_batches is None else min(
            num_batch, max(0, int(max_batches)))
        if run_batches <= 0:
            return {}
        total_link = 0.0
        total_aux = 0.0
        n_aux_batches = 0
        n_closed = 0
        below = 0
        n_censored_tail_groups = 0
        aux_terms_total = 0
        self.reset_gauges()
        self._reset_memory_grad_probe()
        self.window_diag = []

        group_start = 0
        repr_step = 0
        gi = 0
        while group_start < run_batches:
            if self.calibrate and gi >= self.calibrate_groups:
                break
            group_end = min(group_start + self.kf_group_batches, run_batches)
            group_k = group_end - group_start
            # group-anchored step counter: pass 2 increments `global_step`
            # per batch, so pass 1 and pass 2 MUST both derive the per-batch
            # step from the group-start value — otherwise the trace_batch
            # seed (and hence the neighbor-slot sampling of the official
            # neighbor finder, which draws np.random per batch) diverges
            # after the first batch and exact replay silently breaks.
            group_gs = global_step
            # ---- task-only censored tail (shared group plan): past the
            # auxiliary-eligible prefix these batches run pure task training —
            # no pass 1 / no windows / no aux; never counted as n_below.
            if self.aux_prefix_groups is not None and \
                    gi >= self.aux_prefix_groups:
                self.repr_optimizer.zero_grad(set_to_none=True)
                for b in range(group_start, group_end):
                    self.head_optimizer.zero_grad(set_to_none=True)
                    out = self._run_batch(
                        train, b, group_gs + (b - group_start),
                        grad_enabled=True, epoch=epoch)
                    if out is None:
                        continue
                    link_loss, _records = out
                    link_loss.backward()
                    self._probe_memory_grad()
                    self._clip(self.head_params)
                    self.head_optimizer.step()
                    if self.tgn.use_memory:
                        self.tgn.memory.detach_memory()
                    total_link += float(link_loss.detach())
                    global_step += 1
                if self.repr_optimizer is not None and self.repr_params:
                    for p in self.repr_params:
                        if p.grad is not None:
                            p.grad.div_(float(max(1, group_k)))
                    self._clip(self.repr_params)
                    self.repr_optimizer.step()
                    repr_step += 1
                n_censored_tail_groups += 1
                group_start = group_end
                gi += 1
                continue
            if self.aux_kind in ("rec", "pred"):
                r_ = self._train_group_supervised(
                    train, group_start, group_end, group_gs, group_k, epoch,
                    global_step)
                total_link += r_["link_sum"]
                total_aux += r_["aux_sum"]
                n_aux_batches += r_["aux_batches"]
                aux_terms_total += r_["aux_terms"]
                n_closed += r_["closed"]
                repr_step += r_["repr_step"]
                global_step = r_["gs_end"]
                group_start = group_end
                gi += 1
                continue
            # ------------ pass 1: collect records per batch, no grad -------
            state = self._save_group_state()
            pass1_records = []
            with torch.no_grad():
                for b in range(group_start, group_end):
                    out = self._run_batch(
                        train, b, group_gs + (b - group_start),
                        grad_enabled=False, epoch=epoch)
                    if out is None:
                        continue
                    _, records = out
                    # root_row is a batch-local index (0..batch_size) reused
                    # every batch; remap to a globally-unique tree id so the
                    # per-tree window count / tree weight are correct across
                    # the macro group (matching node-class _tree_counter).
                    for rec in records:
                        rec.root_row = int(b) * self.batch_size \
                            + int(rec.root_row)
                    if records:
                        pass1_records.extend(records)
            # window + close (only for the enabled arm)
            g_by_pos_all = {}
            closed_tau = {}
            # shared surviving set across ALL arms: infeasible (mispaired)
            # positions are dropped identically everywhere, and the mispaired
            # arm additionally deranges the consumed parent futures.  The
            # audit population (valid_cut / occurrence / y multisets) is
            # recorded here so gamma_task_only sees the same data population.
            if pass1_records:
                fea = feasible_positions(pass1_records, seed=self.seed,
                                         batch_seed=global_step)
                if self.mispaired:
                    self._mispaired_map = build_mispaired_parent_map(
                        pass1_records, seed=self.seed,
                        batch_seed=global_step)
                    # map keys must agree with the feasibility set (same
                    # bucket + derangement internals); intersect for safety
                    fea = [i for i in fea if i in self._mispaired_map]
                else:
                    self._mispaired_map = {}
                for i in fea:
                    r = pass1_records[i]
                    if self._mispaired_map:
                        r.parent_future_used = self._mispaired_map[i]
                    self.audit.add_population(r)
            if self.kf_on and self.aux_prefix_groups is not None \
                    and not pass1_records:
                raise RuntimeError(
                    "eligible group {}..{} produced no pair records".format(
                        group_start, group_end))
            if self.kf_on and pass1_records:
                weights = tree_equal_weights(pass1_records)
                for r in pass1_records:
                    r.weight = weights[int(r.root_row)]
                by_tau: Dict[str, List] = {}
                for i in fea:
                    by_tau.setdefault(pass1_records[i].tau, []).append(i)
                if self.aux_prefix_groups is not None:
                    # eligible (non-tail) group must produce EVERY canonical
                    # tau; a missing one is an immediate failure.
                    missing = [t for t in self._canon_taus
                               if t not in by_tau]
                    if missing:
                        raise RuntimeError(
                            "eligible group {}..{} missing canonical tau(s) "
                            "{}".format(group_start, group_end, missing))
                for tau, idxs in by_tau.items():
                    win = self._win(tau)
                    tau_recs = [pass1_records[i] for i in idxs]
                    # cap: reset per group (single window per tau per group)
                    win.reset()
                    for r in tau_recs:
                        win.add(r)
                    m_trees = win.n_unique_trees()
                    self.window_diag.append({
                        "group_batches": group_k,
                        "tau": tau,
                        "M_unique_trees": m_trees,
                        "threshold": self.kf_min_trees,
                        "n_records": len(tau_recs)})
                    if not win.ready():
                        below += 1
                        if self.fail_below or \
                                self.aux_prefix_groups is not None:
                            raise RuntimeError(
                                "kf window below threshold: tau={} "
                                "M_unique_trees={} < {} in {} batches".format(
                                    tau, m_trees, self.kf_min_trees,
                                    group_k))
                        continue
                    j, g_pos, diag = win.close_replay(
                        _MapsAdapter(self, tau))
                    if j is None:
                        below += 1
                        continue
                    n_closed += 1
                    closed_tau[tau] = j
                    for pos_in_tau, g in g_pos.items():
                        # key by pair_id: pass 2 rebuilds fresh record objects
                        # whose pair_ids match (RNG/oid restored), but whose
                        # python id() differs.
                        g_by_pos_all[tau_recs[pos_in_tau].pair_id] = g
                        # audit: consumed pairing (donor future for mispaired)
                        r = tau_recs[pos_in_tau]
                        pf_used = getattr(r, "parent_future_used",
                                          r.parent_future)
                        eid = int(pf_used.event_id) if pf_used is not None \
                            else -1
                        self.audit.add_pairing(r, eid)
                    self.audit.add_window_close(
                        tau, [r.pair_id for r in win.records])
            # ------------ restore and pass 2: train -----------------------
            self._restore_group_state(state)
            self.repr_optimizer.zero_grad(set_to_none=True)
            self.snapshot_comp_params()
            _g_task_done = False
            _g_aux_done = False
            _grp_hits = 0
            _grp_recs = 0
            for b in range(group_start, group_end):
                self.head_optimizer.zero_grad(set_to_none=True)
                out = self._run_batch(
                    train, b, group_gs + (b - group_start),
                    grad_enabled=True, epoch=epoch)
                if out is None:
                    continue
                link_loss, records = out
                auxiliary = torch.zeros((), device=self.device)
                terms = []  # reset per batch (stale-terms bug fix)
                if self.kf_on and records and g_by_pos_all:
                    for r in records:
                        _grp_recs += 1
                        g = g_by_pos_all.get(r.pair_id)
                        if g is None:
                            continue
                        _grp_hits += 1
                        self.audit.add_replay(r)
                        z = r.z
                        gd = g.detach()
                        terms.append((gd * z).sum() - (gd * z.detach()).sum())
                    if terms:
                        coeff = -self.lambda_kf * float(group_k)
                        auxiliary = coeff * sum(terms)
                        n_aux_batches += 1
                        aux_terms_total += len(terms)
                        total_aux += float(auxiliary.detach())
                # ---- grad-align probe: last batch of the group, BEFORE the
                # backward, while both graphs are still live.  The
                # straight-line surrogate is 0-valued but carries the
                # adjoint direction, so the two components separate through
                # two autograd.grad calls (no .grad mutation; retain_graph
                # keeps the graphs for the real backward below).  Record
                # only — the training update is never modified.
                if (self.grad_align_diag and self.kf_on and terms
                        and b == group_end - 1 and self.repr_params):
                    g_t = torch.autograd.grad(
                        link_loss, self.repr_params,
                        retain_graph=True, allow_unused=True)
                    g_a = torch.autograd.grad(
                        auxiliary, self.repr_params,
                        retain_graph=True, allow_unused=True)
                    cos, _h1, _h2 = _grad_cosine(self.repr_params, g_t, g_a)
                    ft = _flat_grad(g_t, self.repr_params)
                    fa = _flat_grad(g_a, self.repr_params)
                    nt = float(ft.norm()) if ft is not None else 0.0
                    na = float(fa.norm()) if fa is not None else 0.0
                    ratio = na / nt if nt > 1e-12 else float("nan")
                    self._ga_cos.append(cos)
                    self._ga_ratios.append(ratio)
                    print("[grad-align] group=%d cos=%.4f ratio=%.4f"
                          % (group_start // self.kf_group_batches, cos,
                             ratio), flush=True)
                loss = link_loss + auxiliary
                if self._comp_params:
                    if not _g_task_done and b == group_start:
                        _g_task_done = True
                        tn, _an = self.gauge_comp(link_loss, None)
                        self._gauge_group_task.append(tn)
                    if not _g_aux_done and terms:
                        _g_aux_done = True
                        print("[gauge-probe] group=%d terms=%d t0_rg=%s"
                              % (group_start // self.kf_group_batches,
                                 len(terms),
                                 getattr(terms[0], "requires_grad", "?"))
                              if terms else "", flush=True)
                        _tn, an = self.gauge_comp(link_loss, auxiliary)
                        self._gauge_group_aux.append(an)
                if self.rpbe_constrain:
                    # Plan B: separate the two components so the group-end
                    # projection can write back g_t + mu*g_a.  Task backward
                    # first, snapshot repr grads, then the aux backward whose
                    # repr delta is the aux component.  The straight-line
                    # surrogate is 0-valued, so the loss VALUE is unchanged;
                    # lambda cancels in the projection (scale-invariant).
                    if self._cstr_task_acc is None:
                        self._cstr_reset()
                    link_loss.backward()
                    for i, p in enumerate(self.repr_params):
                        g = p.grad
                        if g is None:
                            self._cstr_snap[i] = None
                            continue
                        self._cstr_task_acc[i].add_(g)
                        self._cstr_snap[i] = g.detach().clone()
                    if terms:
                        auxiliary.backward()
                        for i, p in enumerate(self.repr_params):
                            g = p.grad
                            if g is None or self._cstr_snap[i] is None:
                                continue
                            self._cstr_aux_acc[i].add_(
                                g - self._cstr_snap[i])
                else:
                    loss.backward()
                self._probe_memory_grad()
                self._clip(self.head_params)
                if not self.calibrate:
                    self.head_optimizer.step()
                if self.tgn.use_memory:
                    self.tgn.memory.detach_memory()
                total_link += float(link_loss.detach())
                global_step += 1
            # group close: repr step once
            if self.kf_on:
                print("[audit-debug] group=%d keys=%d records=%d hits=%d"
                      % (group_start // self.kf_group_batches,
                         len(g_by_pos_all), _grp_recs, _grp_hits), flush=True)
            if self.repr_optimizer is not None and self.repr_params:
                if self.rpbe_constrain and self._cstr_task_acc is not None:
                    # Plan-B group-end projection on the accumulated
                    # components (lambda-invariant; see __init__ note):
                    #   trigger iff g_a^T g_t < -kappa ||g_a|| ||g_t||
                    #   g_write = g_t + mu*g_a with
                    #   mu = (-kappa||ga||||gt|| - ga^T gt) / ||ga||^2
                    # otherwise g_write = g_t (aux discarded — d = t).
                    gt = torch.cat([a.reshape(-1).float()
                                    for a in self._cstr_task_acc])
                    ga = torch.cat([a.reshape(-1).float()
                                    for a in self._cstr_aux_acc])
                    nt = float(gt.norm())
                    na = float(ga.norm())
                    s = float((ga * gt).sum())
                    self._cstr_groups += 1
                    active = False
                    mu = 0.0
                    if (na > 1e-12 and nt > 1e-12
                            and s < -self.rpbe_kappa * na * nt):
                        mu = (-self.rpbe_kappa * na * nt - s) / (na * na)
                        active = True
                        self._cstr_active += 1
                        self._cstr_corr_ratios.append(
                            float(mu * na / nt))
                        print("[rpbe-constrain] group=%d kappa=%.2f "
                              "cos=%.4f mu=%.3e corr/task=%.4f"
                              % (group_start // self.kf_group_batches,
                                 self.rpbe_kappa, s / (na * nt),
                                 mu, mu * na / nt), flush=True)
                    for a_t, a_a, p in zip(self._cstr_task_acc,
                                           self._cstr_aux_acc,
                                           self.repr_params):
                        if p.grad is not None:
                            p.grad.copy_(a_t + mu * a_a)
                    self._cstr_reset()
                for p in self.repr_params:
                    if p.grad is not None:
                        p.grad.div_(float(max(1, group_k)))
                self._clip(self.repr_params)
                if not self.calibrate:
                    self.repr_optimizer.step()
                    self.record_param_delta()
                repr_step += 1
            group_start = group_end
            gi += 1
        gsum = self.gauge_summary()
        return {
            "train_link_loss": total_link / max(run_batches, 1),
            "train_aux": total_aux / max(n_aux_batches, 1)
            if n_aux_batches else 0.0,
            "n_aux_batches": n_aux_batches,
            "n_closed": n_closed,
            "n_below": below,
            "n_censored_tail_groups": n_censored_tail_groups,
            "n_batches": run_batches,
            "global_step": global_step,
            "task_step": global_step,
            "repr_step": repr_step,
            "closed_window_step": n_closed,   # real closed windows (>= kf arms)
            "aux_terms": aux_terms_total,
            "aux_comp_grad_norm": gsum["aux_comp_grad_norm"],
            "task_comp_grad_norm": gsum["task_comp_grad_norm"],
            "comp_param_delta": gsum["comp_param_delta"],
            "mem_grad_norm": self._mem_grad_norm,
            "window_diag": list(self.window_diag),
            "grad_align": self._grad_align_summary(),
            "rpbe_constrain": {
                "enabled": bool(self.rpbe_constrain),
                "kappa": self.rpbe_kappa,
                "groups": self._cstr_groups,
                "active": self._cstr_active,
                "active_rate": (self._cstr_active
                                / max(1, self._cstr_groups)),
                "mean_corr_ratio": (float(np.mean(self._cstr_corr_ratios))
                                    if self._cstr_corr_ratios else None),
            },
        }

    def _cstr_reset(self):
        """Zero the per-group task/aux gradient accumulators (plan B)."""
        dev = self.device
        self._cstr_task_acc = [torch.zeros_like(p, device=dev)
                               for p in self.repr_params]
        self._cstr_aux_acc = [torch.zeros_like(p, device=dev)
                              for p in self.repr_params]
        self._cstr_snap = [None] * len(self.repr_params)

    def _grad_align_summary(self):
        """Distribution summary of update-direction cosines (reviewer item
        5: mean / median / P(>0) / P(<0) / three bins / norm ratios)."""
        if not self._ga_cos:
            return {"enabled": bool(self.grad_align_diag), "checks": 0}
        cs = np.asarray(self._ga_cos, dtype=np.float64)
        rs = np.asarray(self._ga_ratios, dtype=np.float64)
        rs_f = rs[~np.isnan(rs)]
        return {
            "enabled": True,
            "checks": int(len(cs)),
            "mean_cos": round(float(cs.mean()), 4),
            "median_cos": round(float(np.median(cs)), 4),
            "p_pos": round(float((cs > 0).mean()), 4),
            "p_neg": round(float((cs < 0).mean()), 4),
            "n_lt_m01": int((cs < -0.1).sum()),
            "n_mid_01": int((np.abs(cs) <= 0.1).sum()),
            "n_gt_p01": int((cs > 0.1).sum()),
            "norm_ratio_mean": (round(float(rs_f.mean()), 4)
                                if rs_f.size else None),
            "norm_ratio_median": (round(float(np.median(rs_f)), 4)
                                  if rs_f.size else None),
            "group_cos": [round(float(c), 4) for c in cs],
            "group_ratios": [round(float(r), 4) for r in rs],
        }

    def _clip(self, params):
        if self.grad_clip <= 0:
            return
        live = [p for p in params if p.grad is not None]
        if live:
            torch.nn.utils.clip_grad_norm_(
                live, max_norm=self.grad_clip, error_if_nonfinite=True)

    # ------------------------------------------------------ memory-grad probe
    def _memory_updater_params(self):
        """Parameters of the cross-batch memory GRU (trainable in repr)."""
        mu = getattr(self.tgn, "memory_updater", None)
        return list(mu.parameters()) if mu is not None else []

    def _probe_memory_grad(self):
        """After a grad-enabled backward, latch the first nonzero memory-GRU
        grad norm.  Two grad batches are required: in start-mode the GRU only
        enters the loss graph once the previous batch has stored messages."""
        if self._mem_probe_done or not self.memory_grad_probe \
                or self.calibrate or not self.tgn.use_memory \
                or not self._mem_probe_params:
            return
        self._mem_probe_batches += 1
        if self._mem_probe_batches < 2:
            return
        grads = [p.grad for p in self._mem_probe_params
                 if p.grad is not None]
        if not grads:
            return
        norm = float(sum(float((g.detach() ** 2).sum())
                         for g in grads) ** 0.5)
        if norm > 0.0:
            self._mem_grad_norm = float(norm)
            self._mem_probe_done = True

    def _reset_memory_grad_probe(self):
        self._mem_probe_done = False
        self._mem_probe_batches = 0
        self._mem_grad_norm = 0.0

    # --------------------------------------------------------- topology scan
    def _collect_pass1_records(self, train, g0, g1, gstep0, epoch):
        recs = []
        with torch.no_grad():
            for b in range(g0, g1):
                out = self._run_batch(train, b, gstep0 + (b - g0),
                                      grad_enabled=False, epoch=epoch)
                if out is None:
                    continue
                _, records = out
                for rec in records:
                    rec.root_row = b * self.batch_size + int(rec.root_row)
                recs.extend(records)
        return recs

    # ---------------------------------------------------- gradient gauges
    @staticmethod
    def _g_norm(grads):
        s = 0.0
        for g in grads:
            if g is not None:
                s += float((g.detach() ** 2).sum())
        return float(s ** 0.5)

    def gauge_comp(self, link_loss, aux_scalar):
        """Compressor grad norms (task-only and aux-only) via autograd.grad.

        Does not mutate ``.grad`` (retain_graph=True), so the caller still does
        its own backward.  ``aux_scalar=None`` for task-only arms.
        """
        if not self._comp_params:
            return 0.0, 0.0
        params = self._comp_params
        g_t = torch.autograd.grad(link_loss, params, retain_graph=True,
                                  allow_unused=True)
        task_n = self._g_norm(g_t)
        aux_n = 0.0
        if aux_scalar is not None:
            if not getattr(aux_scalar, "requires_grad", False):
                print("[gauge-warn] aux_scalar requires_grad=False; "
                      "skip aux gauge", flush=True)
                return float(task_n), 0.0
            g_a = torch.autograd.grad(aux_scalar, params, retain_graph=True,
                                      allow_unused=True)
            aux_n = self._g_norm(g_a)
        return float(task_n), float(aux_n)

    def snapshot_comp_params(self):
        if self._comp_params and self._gauge_pd_snapshot is None:
            self._gauge_pd_snapshot = [
                p.detach().clone().to("cpu") for p in self._comp_params]

    def record_param_delta(self):
        """First-repr-step compressor parameter change (one-shot per epoch)."""
        if self._gauge_param_delta is not None or not self._comp_params:
            return
        snap = self._gauge_pd_snapshot
        if snap is None:
            return
        delta = 0.0
        for s, p in zip(snap, self._comp_params):
            delta += float(((p.detach().cpu() - s) ** 2).sum())
        self._gauge_param_delta = float(delta ** 0.5)
        self._gauge_pd_snapshot = None

    def reset_gauges(self):
        self._gauge_group_aux = []
        self._gauge_group_task = []
        self._gauge_param_delta = None
        self._gauge_pd_snapshot = None

    def gauge_summary(self):
        return {
            "aux_comp_grad_norm": float(np.mean(self._gauge_group_aux))
            if self._gauge_group_aux else 0.0,
            "task_comp_grad_norm": float(np.mean(self._gauge_group_task))
            if self._gauge_group_task else 0.0,
            "comp_param_delta": (float(self._gauge_param_delta)
                                 if self._gauge_param_delta is not None
                                 else 0.0),
        }

    def _super_target(self, rec):
        """Detached supervision target for one surviving record (P1/P2)."""
        if self.aux_kind == "rec":
            u = getattr(rec, "u", None)
            if u is None:
                raise RuntimeError("rec target missing u on record")
            return u.detach().reshape(-1)
        # pred: bitwise R0 aligned-2Obs p for the same cut / C / Y
        return self._pv(rec).detach().reshape(-1)

    def _super_ctx(self, rec):
        from rpbe.pair_rows import build_ctx_vector
        d_ctx = int(getattr(self.rpbe_cfg, "d_c", 32))
        return build_ctx_vector(rec, d_ctx, device=self.device)

    def _train_group_supervised(self, train, g0, g1, group_gs, group_k,
                                epoch, gs0):
        """Rec/Pred macro group: single-population, per-tree-weighted MSE.

        Population and tree weights mirror the R0 kyfan surviving set
        (feasible_positions + mispaired derangement-map intersection), so P1/P2
        supervise the SAME cut intersection as R0.  Pass 2 needs no surrogate:
        the head decodes live ``[z, chi(C)]`` against the frozen-normalized
        detached target.
        """
        state = self._save_group_state()
        recs = self._collect_pass1_records(train, g0, g1, group_gs, epoch)
        fea = feasible_positions(recs, seed=self.seed,
                                 batch_seed=group_gs) if recs else []
        if self.mispaired:
            mmap = build_mispaired_parent_map(recs, seed=self.seed,
                                              batch_seed=group_gs)
            fea = [i for i in fea if i in mmap]
        weights = tree_equal_weights(recs)
        meta = {}
        by_tau: Dict[str, List] = {}
        w_by_tau: Dict[str, float] = {}
        for i in fea:
            r = recs[i]
            r.weight = weights[int(r.root_row)]
            meta[r.pair_id] = float(r.weight)
            by_tau.setdefault(r.tau, []).append(r)
            w_by_tau[r.tau] = w_by_tau.get(r.tau, 0.0) + float(r.weight)
        if not fea:
            raise RuntimeError(
                "eligible supervised group {}..{} produced no records".format(
                    g0, g1))
        missing = [t for t in self._canon_taus if t not in by_tau]
        if missing:
            raise RuntimeError(
                "supervised group {}..{} missing canonical tau(s) {}".format(
                    g0, g1, missing))
        ntaus = len(by_tau)
        # readiness + shared window audit (same as the kyfan arm would close)
        for tau, rl in by_tau.items():
            mt = len({int(r.root_row) for r in rl})
            self.window_diag.append({
                "group_batches": group_k, "tau": tau,
                "M_unique_trees": mt, "threshold": self.kf_min_trees,
                "n_records": len(rl)})
            if mt < self.kf_min_trees:
                raise RuntimeError(
                    "supervised group below trees: tau={} {} < {}".format(
                        tau, mt, self.kf_min_trees))
            self.audit.add_window_close(tau, [r.pair_id for r in rl])
        for i in fea:
            self.audit.add_population(recs[i])
        # FROZEN per-tau target stats, PER-TREE-WEIGHTED over the whole window
        if self.aux_heads is not None:
            for tau, rl in by_tau.items():
                tgt = torch.stack([self._super_target(r) for r in rl])
                wts = torch.tensor([float(r.weight) for r in rl],
                                   dtype=tgt.dtype, device=tgt.device)
                self.aux_heads.fit_weighted(tau, tgt, wts)
        self._restore_group_state(state)

        self.repr_optimizer.zero_grad(set_to_none=True)
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad(set_to_none=True)
        link_sum = 0.0
        aux_sum = 0.0
        aux_batches = 0
        aux_terms = 0
        closed = len(by_tau)     # accepted (eligible) tau windows this group
        gauge_done = False
        gs = gs0
        self.snapshot_comp_params()
        for b in range(g0, g1):
            self.head_optimizer.zero_grad(set_to_none=True)
            out = self._run_batch(train, b, group_gs + (b - g0),
                                  grad_enabled=True, epoch=epoch)
            gs += 1
            if out is None:
                continue
            link_loss, records = out
            contrib = torch.zeros((), device=self.device)
            if self.aux_heads is not None and records and meta:
                parts = []
                for r in records:
                    w = meta.get(r.pair_id)
                    if w is None:
                        continue
                    Wtau = w_by_tau.get(r.tau, 0.0)
                    if Wtau <= 0.0:
                        continue
                    tgt = self._super_target(r)
                    Lr = self.aux_heads.regression_loss(
                        r.tau, r.z.reshape(1, -1),
                        self._super_ctx(r).reshape(1, -1),
                        tgt.reshape(1, -1))
                    # per-batch partial sums, across the group, to
                    # (lambda / n_layers) * per-tree-weighted window mean
                    parts.append((w / Wtau) * (self.aux_lambda / ntaus) * Lr)
                    aux_terms += 1
                if parts:
                    contrib = sum(parts)
                    aux_batches += 1
                    aux_sum += float(contrib.detach())
                    for r in records:
                        if r.pair_id in meta:
                            self.audit.add_replay(r)
            loss = link_loss + contrib
            if not gauge_done and self._comp_params \
                    and float(contrib.detach()) != 0.0:
                tn, an = self.gauge_comp(link_loss, contrib)
                self._gauge_group_task.append(tn)
                self._gauge_group_aux.append(an)
                gauge_done = True
            loss.backward()
            self._probe_memory_grad()
            self._clip(self.head_params)
            if not self.calibrate:
                self.head_optimizer.step()
            if self.tgn.use_memory:
                self.tgn.memory.detach_memory()
            link_sum += float(link_loss.detach())
        # group close: repr and aux heads step once (per macro group)
        if not self.calibrate:
            for opt in (self.repr_optimizer, self.aux_optimizer):
                if opt is None:
                    continue
                ps = opt.param_groups[0]["params"]
                for p in ps:
                    if p.grad is not None:
                        p.grad.div_(float(max(1, group_k)))
                live = [p for p in ps if p.grad is not None]
                if live:
                    torch.nn.utils.clip_grad_norm_(
                        live, max_norm=self.grad_clip)
                opt.step()
            self.record_param_delta()
        return {"link_sum": link_sum, "aux_sum": aux_sum,
                "aux_batches": aux_batches, "aux_terms": aux_terms,
                "closed": closed, "repr_step": 1, "gs_end": gs}

    def scan_topology(self, epoch, train, group_batches=None,
                      n_batches=None):
        """No-grad full-train scan mirroring the TRAINING window path.

        Each macro group runs the exact pass-1 record collection (per-batch
        global-step schedule, root_row remap), then the SAME surviving-set
        filtering the training loop applies — ``feasible_positions`` and, for
        the mispaired arm, the derangement-map intersection — before counting
        per-tau unique trees.  Every canonical tau is reported (0 trees when a
        group produced none).  Used to fix the auxiliary-eligible prefix and
        mark the future-censored tail (§1.1-final).  No close_replay / no
        backward / no optimizer; does not mutate training state.
        """
        gb = int(group_batches or self.kf_group_batches)
        self.reset_memory()
        self.tgn.train(False)
        num_batch = math.ceil(len(train.sources) / self.batch_size)
        if n_batches is not None:
            num_batch = min(num_batch, max(0, int(n_batches)))
        if num_batch <= 0:
            return []
        groups = []
        g0 = 0
        gstep = 0
        while g0 < num_batch:
            g1 = min(g0 + gb, num_batch)
            group_gs = gstep
            recs_all = []
            with torch.no_grad():
                for b in range(g0, g1):
                    out = self._run_batch(
                        train, b, group_gs + (b - g0),
                        grad_enabled=False, epoch=epoch)
                    gstep += 1
                    if out is None:
                        continue
                    _, recs = out
                    for rec in recs:
                        rec.root_row = b * self.batch_size + int(rec.root_row)
                    recs_all.extend(recs)
            # surviving set = what the training window would actually see
            if recs_all:
                fea = feasible_positions(recs_all, seed=self.seed,
                                         batch_seed=group_gs)
                if self.mispaired:
                    mmap = build_mispaired_parent_map(
                        recs_all, seed=self.seed, batch_seed=group_gs)
                    fea = [i for i in fea if i in mmap]
            else:
                fea = []
            by_tau: Dict[str, List] = {}
            for i in fea:
                by_tau.setdefault(recs_all[i].tau, []).append(recs_all[i])
            row = {"group_start": g0, "group_end": g1, "n_batches": g1 - g0,
                   "taus": {}}
            for tau in sorted(set(self._canon_taus) | set(by_tau)):
                recs = by_tau.get(tau, [])
                win = PairKFWindow(tau=tau, eps=self._window_eps,
                                   min_unique_trees=self.kf_min_trees)
                for r in recs:
                    win.add(r)
                row["taus"][tau] = {
                    "unique_trees": int(win.n_unique_trees()),
                    "n_records": len(recs),
                    "ready": bool(win.ready())}
            groups.append(row)
            g0 = g1
        return groups


class _MapsAdapter:
    """Adapts the PairKFWindow close to build one p row per record."""

    def __init__(self, loop, tau):
        self.loop = loop
        self.tau = tau

    def pv_row(self, rec):
        return self.loop._pv(rec)


def _ctx_vec(rec, d_ctx, device=None):
    """Structural context C_v (cut-time-known only); see pair_rows."""
    from rpbe.pair_rows import build_ctx_vector
    return build_ctx_vector(rec, d_ctx, device=device)
