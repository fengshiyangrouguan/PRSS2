"""Stage-1 TGN link pretraining with a one-pass RPBE hook.

The official positive/negative BCE and ``backprop_every`` accumulation are
unchanged.  RPBE cut collection happens inside each ordinary edge query; a
lagged running-moment VJP is added to that same microbatch loss.  No batch is
queried twice and fabricated link negatives never enter future supervision.
"""

import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from rpbe.loss import KFLaggedWindow
from rpbe.training.isolation import assert_clean, rpbe_fingerprint
from rpbe.training.jodie_loop import select_trace_rows


class TGNPretrainLoop:
    def __init__(self, *, tgn, optimizer, device, batch_size, n_neighbors,
                 backprop_every, grad_clip, monitor, seed, adapter=None,
                 cut_builder=None, fixed_maps=None, rpbe_cfg=None,
                 trace_roots=8):
        self.tgn = tgn
        self.optimizer = optimizer
        self.device = device
        self.batch_size = int(batch_size)
        self.n_neighbors = int(n_neighbors)
        self.backprop_every = int(backprop_every)
        self.grad_clip = float(grad_clip)
        self.monitor = monitor
        self.seed = int(seed)
        self.adapter = adapter
        self.cut_builder = cut_builder
        self.fixed_maps = fixed_maps
        self.rpbe_cfg = rpbe_cfg
        self.lambda_kf = float(rpbe_cfg.lambda_kf) \
            if rpbe_cfg is not None else 0.0
        self.trace_roots = int(trace_roots)
        self.rpbe_on = bool(adapter is not None and cut_builder is not None
                            and fixed_maps is not None
                            and rpbe_cfg is not None)

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
        self.kf_window = (KFLaggedWindow(
            kf_dims, min_ratio=rpbe_cfg.kf_min_ratio,
            min_abs=rpbe_cfg.kf_min_abs, eps=rpbe_cfg.ridge_eps,
            fixed_maps=fixed_maps, variant=rpbe_cfg.kf_variant)
            if self.rpbe_on else None)

    # ------------------------------------------------------------- stream state
    def reset_memory(self):
        if self.tgn.use_memory:
            self.tgn.memory.__init_memory__()

    def _sample_negatives(self, data, size: int) -> np.ndarray:
        """Official uniform-over-destination negative sampler."""
        return np.random.choice(np.asarray(data.destinations),
                                size=size, replace=True)

    def _clip_all_groups(self):
        if self.grad_clip <= 0:
            return
        params = [param for group in self.optimizer.param_groups
                  for param in group["params"] if param.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(
                params, max_norm=self.grad_clip, error_if_nonfinite=True)

    def _consume_kf(self, cuts, step):
        scores, surrogates, diagnostics, cold, refreshed = \
            self.kf_window.step(cuts)
        weighted_score = float(sum(
            self.rpbe_cfg.alpha(tau) * score
            for tau, score in scores.items()))
        if surrogates:
            auxiliary = -self.lambda_kf * sum(
                self.rpbe_cfg.alpha(tau) * value
                for tau, value in surrogates.items())
        else:
            auxiliary = torch.zeros((), device=self.device)
        if scores:
            bounds = {tau: min(int(self.rpbe_cfg.state_dims[tau]),
                               int(self.fixed_maps.m))
                      for tau in scores}
            self.monitor.validate_kf(scores, bounds, step)
        for tau, diag in diagnostics.items():
            if diag.get("failed"):
                self.monitor.alert(
                    "warning", "kf_reference_failed",
                    "{} {} scale_z={:.3e} scale_p={:.3e} trees={}".format(
                        tau, diag["failed"],
                        diag.get("scale_z", float("nan")),
                        diag.get("scale_p", float("nan")),
                        diag["M_unique_trees"]),
                    step=step, interface=tau)
        return (weighted_score, scores, auxiliary, set(cold),
                len(refreshed))

    # ----------------------------------------------------------------- training
    def train_epoch(self, epoch: int, global_step: int, train) -> Dict:
        """One epoch; every microbatch performs exactly one TGN query."""
        del epoch
        self.reset_memory()
        self.tgn.train(True)
        num_batch = math.ceil(len(train.sources) / self.batch_size)
        total_link = 0.0
        total_kf = 0.0
        kf_sum = {}
        kf_count_tau = {}
        skipped_types = set()
        refreshes = 0
        aux_batches = 0
        n_steps = 0
        no_cuts_alerted = False

        for group_start in range(0, num_batch, self.backprop_every):
            self.optimizer.zero_grad(set_to_none=True)
            group_loss = None
            group_batches = 0
            for offset in range(self.backprop_every):
                batch_index = group_start + offset
                if batch_index >= num_batch:
                    continue
                start = batch_index * self.batch_size
                stop = min(len(train.sources),
                           (batch_index + 1) * self.batch_size)
                sources = train.sources[start:stop]
                destinations = train.destinations[start:stop]
                timestamps = train.timestamps[start:stop]
                edge_idxs = train.edge_idxs[start:stop]
                size = len(sources)
                negatives = self._sample_negatives(train, size)

                trace_rows = []
                if self.rpbe_on:
                    trace_rows = select_trace_rows(
                        np.zeros(size), self.trace_roots, self.seed,
                        global_step, mode="evenly_spaced")
                    self.adapter.set_trace_source_rows(trace_rows)
                elif self.adapter is not None:
                    self.adapter.clear_trace()

                positive, negative = self.tgn.compute_edge_probabilities(
                    sources, destinations, negatives, timestamps, edge_idxs,
                    self.n_neighbors)
                link_loss = (
                    F.binary_cross_entropy(
                        positive.squeeze(),
                        torch.ones(size, device=self.device))
                    + F.binary_cross_entropy(
                        negative.squeeze(),
                        torch.zeros(size, device=self.device)))
                auxiliary = torch.zeros((), device=self.device)
                kf_value = 0.0

                if self.rpbe_on and trace_rows \
                        and self.adapter.trace is not None:
                    cuts = self.cut_builder.build(
                        self.adapter.trace, batch_seed=global_step)
                    if cuts:
                        (kf_value, scores, auxiliary, cold,
                         new_refreshes) = self._consume_kf(
                            cuts, global_step)
                        skipped_types.update(cold)
                        refreshes += new_refreshes
                        if auxiliary.requires_grad:
                            aux_batches += 1
                        for tau, score in scores.items():
                            kf_sum[tau] = kf_sum.get(tau, 0.0) + score
                            kf_count_tau[tau] = \
                                kf_count_tau.get(tau, 0) + 1
                    elif not no_cuts_alerted:
                        self.monitor.alert(
                            "warning", "kf_no_future_rows",
                            "traced cuts have no strictly-future train outcomes",
                            step=global_step)
                        no_cuts_alerted = True

                micro_loss = link_loss + auxiliary
                group_loss = micro_loss if group_loss is None \
                    else group_loss + micro_loss
                group_batches += 1
                total_link += float(link_loss.detach())
                total_kf += kf_value
                self.monitor.validate_losses({
                    "link": float(link_loss.detach()),
                    "kf_score": kf_value,
                    "main_total": float(micro_loss.detach()),
                }, global_step)
                global_step += 1

            if group_batches:
                # Preserve upstream scaling, including its final partial group.
                (group_loss / self.backprop_every).backward()
                self._clip_all_groups()
                self.optimizer.step()
                n_steps += 1
            if self.tgn.use_memory:
                self.tgn.memory.detach_memory()

        kf_out = None
        if kf_count_tau and self.rpbe_cfg is not None:
            means = {tau: value / kf_count_tau[tau]
                     for tau, value in kf_sum.items()}
            kf_out = {
                "estimator": "one_pass_lagged_moment_adjoint",
                "J": means,
                "J_frac": {
                    tau: value / min(int(self.rpbe_cfg.state_dims[tau]),
                                     int(self.fixed_maps.m))
                    for tau, value in means.items()},
                "skipped_types": sorted(skipped_types),
                "kf_score": total_kf / max(num_batch, 1),
                "kf_loss": -self.lambda_kf * total_kf / max(num_batch, 1),
                "reference_refreshes": refreshes,
                "aux_batches": aux_batches,
                "p_cache_hits": getattr(self.fixed_maps, "_p_cache_hits", 0),
                "p_cache_misses": getattr(
                    self.fixed_maps, "_p_cache_misses", 0),
            }
        return {
            "train_link_loss": total_link / max(num_batch, 1),
            "train_kf_score": total_kf / max(num_batch, 1),
            "train_kf_loss": -self.lambda_kf * total_kf / max(num_batch, 1),
            "kf": kf_out,
            "n_steps": n_steps,
            "n_batches": num_batch,
            "global_step": global_step,
        }

    # --------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate_edge_prediction(self, split, neg_seed: int) -> Dict:
        """AP/AUC over split edges versus fixed-seed negative destinations."""
        before = rpbe_fingerprint(self.fixed_maps)
        self.tgn.eval()
        if self.adapter is not None:
            self.adapter.clear_trace()
        sampler = np.random.RandomState(neg_seed)
        destination_pool = np.asarray(split.destinations)
        probs, labels = [], []
        for batch_index in range(math.ceil(len(split.sources)
                                           / self.batch_size)):
            start = batch_index * self.batch_size
            stop = min(len(split.sources),
                       (batch_index + 1) * self.batch_size)
            size = stop - start
            negatives = sampler.choice(
                destination_pool, size=size, replace=True)
            positive, negative = self.tgn.compute_edge_probabilities(
                split.sources[start:stop], split.destinations[start:stop],
                negatives, split.timestamps[start:stop],
                split.edge_idxs[start:stop], self.n_neighbors)
            probs.append(positive.squeeze().cpu().numpy())
            probs.append(negative.squeeze().cpu().numpy())
            labels.append(np.ones(size))
            labels.append(np.zeros(size))
        probs_np = np.concatenate(probs)
        labels_np = np.concatenate(labels)
        trace_created = bool(self.adapter is not None
                             and self.adapter.trace is not None)
        assert_clean(before, self.fixed_maps, trace_created,
                     "evaluate_edge_prediction")
        return {
            "val_ap": float(average_precision_score(labels_np, probs_np)),
            "val_auc": float(roc_auc_score(labels_np, probs_np)),
        }
