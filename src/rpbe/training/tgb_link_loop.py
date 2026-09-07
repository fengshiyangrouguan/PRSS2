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
                 n_observations=2, trace_mode="evenly_spaced"):
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
        self.use_parent = arm_use_parent(arm)
        self.kf_on = arm_kf_on(arm, self.lambda_kf)

        # per-interface PairKFWindow (child tau keys)
        self.pair_windows: Dict[str, PairKFWindow] = {}
        self._window_eps = float(rpbe_cfg.ridge_eps) if rpbe_cfg else 1e-3

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
        """One p_v row for a record under this arm's use_parent."""
        ctx = _ctx_vec(rec, d_ctx=self._ctx_dim())
        child = self._event(rec.child_future, rec.child_time)
        parent = self._event(rec.parent_future, rec.parent_time)
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
                min_unique_trees=self.kf_min_trees)
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
    def _sample_negatives(self, train, size):
        return np.random.choice(np.asarray(train.destinations),
                                size=size, replace=True)

    # ---------------------------------------------------------------- forward
    def _run_batch(self, train, batch_index, global_step, grad_enabled):
        start = batch_index * self.batch_size
        stop = min(len(train.sources), (batch_index + 1) * self.batch_size)
        if stop <= start:
            return None
        sources = train.sources[start:stop]
        destinations = train.destinations[start:stop]
        timestamps = train.timestamps[start:stop]
        edge_idxs = train.edge_idxs[start:stop]
        size = len(sources)
        negatives = self._sample_negatives(train, size)

        trace_rows = []
        if self.kf_on:
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
        if self.kf_on and trace_rows and self.adapter.trace is not None:
            records = self._records_from_trace(self.adapter.trace)
        self.adapter.clear_trace()
        return link_loss, records

    # ------------------------------------------------------------ train epoch
    def train_epoch(self, epoch, global_step, train, max_batches=None):
        self.reset_memory()
        self.tgn.train(True)
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

        group_start = 0
        while group_start < run_batches:
            group_end = min(group_start + self.kf_group_batches, run_batches)
            group_k = group_end - group_start
            # ------------ pass 1: collect records per batch, no grad -------
            state = self._save_group_state()
            pass1_records = []
            with torch.no_grad():
                for b in range(group_start, group_end):
                    out = self._run_batch(
                        train, b, global_step + (b - group_start),
                        grad_enabled=False)
                    if out is None:
                        continue
                    _, records = out
                    if records:
                        pass1_records.extend(records)
            # window + close (only for the enabled arm)
            g_by_pos_all = {}
            closed_tau = {}
            if self.kf_on and pass1_records:
                weights = tree_equal_weights(pass1_records)
                for r in pass1_records:
                    r.weight = weights[int(r.root_row)]
                # drop infeasible (mispaired) positions identically for the
                # pair arm so aligned/mispaired share the surviving set
                if self.arm == "2obs_mispaired":
                    fea = feasible_positions(pass1_records, seed=self.seed,
                                             batch_seed=global_step)
                else:
                    fea = list(range(len(pass1_records)))
                by_tau: Dict[str, List] = {}
                for i in fea:
                    by_tau.setdefault(pass1_records[i].tau, []).append(i)
                for tau, idxs in by_tau.items():
                    win = self._win(tau)
                    tau_recs = [pass1_records[i] for i in idxs]
                    # cap: reset per group (single window per tau per group)
                    win.reset()
                    for r in tau_recs:
                        win.add(r)
                    if not win.ready():
                        below += 1
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
            # ------------ restore and pass 2: train -----------------------
            self._restore_group_state(state)
            self.repr_optimizer.zero_grad(set_to_none=True)
            for b in range(group_start, group_end):
                self.head_optimizer.zero_grad(set_to_none=True)
                out = self._run_batch(
                    train, b, global_step + (b - group_start),
                    grad_enabled=True)
                if out is None:
                    continue
                link_loss, records = out
                auxiliary = torch.zeros((), device=self.device)
                if self.kf_on and records and g_by_pos_all:
                    terms = []
                    for r in records:
                        g = g_by_pos_all.get(r.pair_id)
                        if g is None:
                            continue
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
            if self.repr_optimizer is not None and self.repr_params:
                for p in self.repr_params:
                    if p.grad is not None:
                        p.grad.div_(float(max(1, group_k)))
                self._clip(self.repr_params)
                self.repr_optimizer.step()
            group_start = group_end
        return {
            "train_link_loss": total_link / max(run_batches, 1),
            "train_aux": total_aux / max(n_aux_batches, 1)
            if n_aux_batches else 0.0,
            "n_aux_batches": n_aux_batches,
            "n_closed": n_closed,
            "n_below": below,
            "n_batches": run_batches,
            "global_step": global_step,
        }

    def _clip(self, params):
        if self.grad_clip <= 0:
            return
        live = [p for p in params if p.grad is not None]
        if live:
            torch.nn.utils.clip_grad_norm_(
                live, max_norm=self.grad_clip, error_if_nonfinite=True)


class _MapsAdapter:
    """Adapts the PairKFWindow close to build one p row per record."""

    def __init__(self, loop, tau):
        self.loop = loop
        self.tau = tau

    def pv_row(self, rec):
        return self.loop._pv(rec)


def _ctx_vec(rec, d_ctx):
    """Structural context C_v (cut-time-known only); see pair_rows."""
    from rpbe.pair_rows import build_ctx_vector
    return build_ctx_vector(rec, d_ctx)
