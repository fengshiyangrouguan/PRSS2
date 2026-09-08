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
                 aux_lambda=None):
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
        self.window_diag = []

        group_start = 0
        repr_step = 0
        gi = 0
        while group_start < run_batches:
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
                if self.kf_on and records and g_by_pos_all:
                    terms = []
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
                        total_aux += float(auxiliary.detach())
                loss = link_loss + auxiliary
                loss.backward()
                self._clip(self.head_params)
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
                for p in self.repr_params:
                    if p.grad is not None:
                        p.grad.div_(float(max(1, group_k)))
                self._clip(self.repr_params)
                self.repr_optimizer.step()
                repr_step += 1
            group_start = group_end
            gi += 1
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
            "window_diag": list(self.window_diag),
        }

    def _clip(self, params):
        if self.grad_clip <= 0:
            return
        live = [p for p in params if p.grad is not None]
        if live:
            torch.nn.utils.clip_grad_norm_(
                live, max_norm=self.grad_clip, error_if_nonfinite=True)

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
        for i in fea:
            r = recs[i]
            r.weight = weights[int(r.root_row)]
            meta[r.pair_id] = float(r.weight)
            by_tau.setdefault(r.tau, []).append(r)
        if not fea:
            raise RuntimeError(
                "eligible supervised group {}..{} produced no records".format(
                    g0, g1))
        missing = [t for t in self._canon_taus if t not in by_tau]
        if missing:
            raise RuntimeError(
                "supervised group {}..{} missing canonical tau(s) {}".format(
                    g0, g1, missing))
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
        # fit FROZEN per-tau target statistics once (calibration), then pass 2
        if self.aux_heads is not None:
            for tau, rl in by_tau.items():
                tgt = torch.stack([self._super_target(r) for r in rl])
                self.aux_heads.fit(tau, tgt)
        self._restore_group_state(state)

        self.repr_optimizer.zero_grad(set_to_none=True)
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad(set_to_none=True)
        link_sum = 0.0
        aux_sum = 0.0
        aux_batches = 0
        closed = 0
        gs = gs0
        for b in range(g0, g1):
            self.head_optimizer.zero_grad(set_to_none=True)
            out = self._run_batch(train, b, group_gs + (b - g0),
                                  grad_enabled=True, epoch=epoch)
            gs += 1
            if out is None:
                continue
            link_loss, records = out
            aux_term = torch.zeros((), device=self.device)
            if self.aux_heads is not None and records and meta:
                terms = []
                wsum = 0.0
                for r in records:
                    w = meta.get(r.pair_id)
                    if w is None:
                        continue
                    tgt = self._super_target(r)
                    L = self.aux_heads.regression_loss(
                        r.tau, r.z.reshape(1, -1),
                        self._super_ctx(r).reshape(1, -1),
                        tgt.reshape(1, -1))
                    terms.append(w * L)
                    wsum += w
                if wsum > 0.0 and terms:
                    aux_term = sum(terms) / wsum
                    aux_batches += 1
                    aux_sum += float(aux_term.detach())
            loss = link_loss + self.aux_lambda * aux_term
            loss.backward()
            self._clip(self.head_params)
            self.head_optimizer.step()
            if self.tgn.use_memory:
                self.tgn.memory.detach_memory()
            link_sum += float(link_loss.detach())
        # group close: repr and aux heads step once (per macro group)
        for opt in (self.repr_optimizer, self.aux_optimizer):
            if opt is None:
                continue
            ps = opt.param_groups[0]["params"]
            for p in ps:
                if p.grad is not None:
                    p.grad.div_(float(max(1, group_k)))
            live = [p for p in ps if p.grad is not None]
            if live:
                torch.nn.utils.clip_grad_norm_(live, max_norm=self.grad_clip)
            opt.step()
        return {"link_sum": link_sum, "aux_sum": aux_sum,
                "aux_batches": aux_batches, "closed": closed,
                "repr_step": 1, "gs_end": gs}

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
