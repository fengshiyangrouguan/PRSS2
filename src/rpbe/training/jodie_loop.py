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

from rpbe.loss import KFLaggedWindow
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

    def __init__(self, *, tgn, decoder, optimizer, device, batch_size,
                 n_neighbors, grad_clip, monitor, seed, finetune_host=False,
                 selection_metric="auc", adapter=None, cut_builder=None,
                 fixed_maps=None, rpbe_cfg=None, trace_roots=8,
                 trace_mode="evenly_spaced", train_eval_auc=False):
        self.tgn = tgn
        self.decoder = decoder
        self.optimizer = optimizer
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
        self.rpbe_on = bool(adapter is not None and cut_builder is not None
                            and fixed_maps is not None
                            and rpbe_cfg is not None)
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
        self.kf_window = (KFLaggedWindow(
            kf_dims, min_ratio=rpbe_cfg.kf_min_ratio,
            min_abs=rpbe_cfg.kf_min_abs, eps=rpbe_cfg.ridge_eps,
            fixed_maps=fixed_maps, variant=rpbe_cfg.kf_variant)
            if self.rpbe_on else None)

    def _clip_all_groups(self):
        if self.grad_clip <= 0:
            return
        params = [param for group in self.optimizer.param_groups
                  for param in group["params"] if param.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(
                params, max_norm=self.grad_clip, error_if_nonfinite=True)

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
        return (weighted_score, scores, auxiliary, diagnostics,
                set(cold), len(refreshed))

    # ----------------------------------------------------------------- training
    def train_epoch(self, epoch: int, global_step: int, train: object) -> Dict:
        """One chronological epoch, one query/backward/step per batch."""
        del epoch
        self.reset_memory()
        self.tgn.train(self.finetune_host)
        self.decoder.train()

        total_task = 0.0
        total_kf = 0.0
        kf_sum = {}
        kf_count_tau = {}
        skipped_types = set()
        refreshes = 0
        aux_batches = 0
        train_probs, train_labels = [], []
        no_cuts_alerted = False
        num_batch = math.ceil(len(train.sources) / self.batch_size)

        for batch_index in range(num_batch):
            start = batch_index * self.batch_size
            stop = min(len(train.sources), (batch_index + 1) * self.batch_size)
            sources = train.sources[start:stop]
            destinations = train.destinations[start:stop]
            timestamps = train.timestamps[start:stop]
            edge_idxs = train.edge_idxs[start:stop]
            labels_np = train.labels[start:stop]
            labels_t = torch.from_numpy(labels_np).float().to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            trace_rows = []
            if self.rpbe_on:
                trace_rows = select_trace_rows(
                    labels_np, self.trace_roots, self.seed, global_step,
                    self.trace_mode)
                self.adapter.set_trace_source_rows(trace_rows)
            elif self.adapter is not None:
                self.adapter.clear_trace()

            src_emb, _, _ = self._full_official_embedding_call(
                sources, destinations, timestamps, edge_idxs,
                grad_enabled=(self.finetune_host or self.rpbe_on))
            prediction = self.decoder(src_emb).sigmoid()
            task_loss = F.binary_cross_entropy(prediction, labels_t)
            auxiliary = torch.zeros((), device=self.device)
            kf_value = 0.0

            if self.rpbe_on and trace_rows and self.adapter.trace is not None:
                cuts = self.cut_builder.build(
                    self.adapter.trace, batch_seed=global_step)
                if cuts:
                    (kf_value, scores, auxiliary, _diagnostics, cold,
                     new_refreshes) = self._consume_kf(cuts, global_step)
                    skipped_types.update(cold)
                    refreshes += new_refreshes
                    if auxiliary.requires_grad:
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

            loss = task_loss + auxiliary
            self.monitor.validate_losses({
                "task": float(task_loss.detach()),
                "kf_score": kf_value,
                "main_total": float(loss.detach()),
            }, global_step)
            loss.backward()
            self._clip_all_groups()
            self.optimizer.step()
            if self.tgn.use_memory and (self.finetune_host or self.rpbe_on):
                self.tgn.memory.detach_memory()

            total_task += float(task_loss.detach())
            total_kf += kf_value
            train_probs.append(prediction.detach().cpu().numpy())
            train_labels.append(labels_np)
            global_step += 1

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
                "J": means,
                "J_frac": {
                    tau: value / min(int(self.rpbe_cfg.state_dims[tau]),
                                     int(self.fixed_maps.m))
                    for tau, value in means.items()},
                "skipped_types": sorted(skipped_types),
                "kf_score": total_kf / max(num_batch, 1),
                "kf_loss": -self.lambda_kf * total_kf / max(num_batch, 1),
                "p_cache_hits": getattr(self.fixed_maps, "_p_cache_hits", 0),
                "p_cache_misses": getattr(
                    self.fixed_maps, "_p_cache_misses", 0),
                "reference_refreshes": refreshes,
                "aux_batches": aux_batches,
            }
        return {
            "train_task_loss": total_task / max(num_batch, 1),
            "train_kf_score": total_kf / max(num_batch, 1),
            "train_kf_loss": -self.lambda_kf * total_kf / max(num_batch, 1),
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
