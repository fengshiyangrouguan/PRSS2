"""JODIE node-classification training loop on the official TGN host (stage 2).

Protocol clone of upstream ``official_tgn/train_supervised.py`` plus the v1
matched corrections: supervised epochs over natural labels (no negative
sampling), BCE, memory reset at epoch start, chronological train -> validation
replay (no reset between train and val, upstream semantics), best held-out
selection, then zero-memory train+val replay before the held-out test.

RPBE hooks: when an adapter + cut builder are attached, each training batch
traces a few roots, builds CutRecord rows over the computation tree, computes
the per-interface Ky Fan score and adds ``lambda_kf * (-sum alpha J)`` to the
task loss (one backward, one optimizer over host + compressor + decoder).
Evaluation never builds cuts and never touches the fixed maps.
"""

import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from rpbe.loss import KyFanTracker, kf_loss
from rpbe.training.isolation import assert_clean, rpbe_fingerprint

EPS = 1e-7


def metric_bundle(labels, probs):
    """AUC/AP/NLL plus the diagnostics the v1 protocol reported."""
    labels = np.asarray(labels).astype(np.float64)
    probs = np.clip(np.asarray(probs).astype(np.float64), EPS, 1 - EPS)
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    ap = float(average_precision_score(labels, probs)) if labels.sum() > 0 else 0.0
    nll = float(-(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean())
    pos = labels > 0.5
    neg = ~pos
    return {
        "auc": auc,
        "ap": ap,
        "nll": nll,
        "positive_nll": float(-np.log(probs[pos]).mean()) if pos.any() else float("nan"),
        "negative_nll": float(-np.log(1 - probs[neg]).mean()) if neg.any() else float("nan"),
        "positives": int(pos.sum()),
        "pairs": int(len(labels)),
        "positive_rate": float(pos.mean()),
        "mean_prob_positive": float(probs[pos].mean()) if pos.any() else float("nan"),
        "mean_prob_negative": float(probs[neg].mean()) if neg.any() else float("nan"),
    }


def select_trace_rows(labels, max_roots: int, seed: int, batch_index: int,
                      mode: str = "positive_first") -> list:
    """How traced roots are picked within a batch (positive-first, v1 semantics)."""
    if mode == "off" or max_roots <= 0:
        return []
    labels = np.asarray(labels)
    if mode == "evenly_spaced":
        if len(labels) == 0:
            return []
        rng = np.random.RandomState(seed + 104729 * (batch_index + 1))
        rows = np.linspace(0, len(labels) - 1,
                           min(max_roots, len(labels))).astype(np.int64)
        return [int(r) for r in rows]
    if mode != "positive_first":
        raise ValueError("unknown trace mode {}".format(mode))
    pos = np.flatnonzero(labels > 0.5).tolist()
    neg = np.flatnonzero(labels <= 0.5).tolist()
    chosen = pos[:max_roots]
    remain = max_roots - len(chosen)
    if remain > 0 and neg:
        rng = np.random.RandomState(seed + 104729 * (batch_index + 1))
        chosen.extend(neg if len(neg) <= remain
                      else rng.choice(neg, size=remain, replace=False).tolist())
    return sorted(chosen)


class JodieNodeClassificationLoop:
    """Owns the official-TGN stream; RPBE hooks are all optional."""

    def __init__(self, *, tgn, decoder, optimizer, device, batch_size, n_neighbors,
                 grad_clip, monitor, seed, finetune_host=False,
                 selection_metric="auc", adapter=None, cut_builder=None,
                 fixed_maps=None, rpbe_cfg=None,
                 trace_roots=8, trace_mode="positive_first"):
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
        # Single source of truth: the component weight lives in RPBConfig
        # (a CLI/cfg duplication would silently diverge).
        self.lambda_kf = float(rpbe_cfg.lambda_kf) if rpbe_cfg is not None else 0.0
        self.trace_roots = int(trace_roots)
        self.trace_mode = trace_mode
        self.rpbe_on = bool(adapter is not None and cut_builder is not None
                            and fixed_maps is not None and rpbe_cfg is not None)
        self.kf_tracker = (KyFanTracker(
            rpbe_cfg.interfaces, ema_rho=rpbe_cfg.kf_ema_rho,
            min_ratio=rpbe_cfg.kf_min_ratio, min_abs=rpbe_cfg.kf_min_abs,
            eps=rpbe_cfg.ridge_eps, fixed_maps=fixed_maps)
            if self.rpbe_on else None)

    # ------------------------------------------------------------- stream state
    def reset_memory(self):
        if self.tgn.use_memory:
            self.tgn.memory.__init_memory__()

    def _full_official_embedding_call(self, sources, destinations, timestamps,
                                      edge_idxs, grad_enabled):
        """Exactly the src/dst/dst call of upstream train_supervised.py."""
        ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with ctx:
            return self.tgn.compute_temporal_embeddings(
                sources, destinations, destinations, timestamps, edge_idxs,
                self.n_neighbors)

    # ----------------------------------------------------------------- training
    def train_epoch(self, epoch: int, global_step: int, train: object) -> Dict:
        """One supervised epoch over the chronological train stream."""
        self.reset_memory()
        self.tgn.train(self.finetune_host)
        self.decoder.train()

        total_task = total_kf = 0.0
        n_batches = 0
        kf_sum = {}
        kf_count_tau = {}
        kf_count = 0
        skipped_types = set()
        train_probs, train_labels = [], []
        num_batch = math.ceil(len(train.sources) / self.batch_size)
        for k in range(num_batch):
            s, e = k * self.batch_size, min(len(train.sources),
                                            (k + 1) * self.batch_size)
            sources = train.sources[s:e]
            dests = train.destinations[s:e]
            times = train.timestamps[s:e]
            edge_idxs = train.edge_idxs[s:e]
            labels_np = train.labels[s:e]
            labels_t = torch.from_numpy(labels_np).float().to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            trace_rows = []
            if self.adapter is not None:
                trace_rows = select_trace_rows(
                    labels_np, self.trace_roots, self.seed, global_step,
                    self.trace_mode)
                self.adapter.set_trace_source_rows(trace_rows)

            src_emb, _, _ = self._full_official_embedding_call(
                sources, dests, times, edge_idxs,
                grad_enabled=(self.finetune_host or self.rpbe_on))
            logits = self.decoder(src_emb)
            pred = logits.sigmoid()
            # Upstream node-classification objective exactly: sigmoid + BCE.
            # (BCEWithLogits would be numerically stabler, but the protocol
            # stays byte-compatible with the upstream baseline.)
            task_loss = F.binary_cross_entropy(pred, labels_t)

            kf_v = 0.0
            kf_detail = {}
            if self.rpbe_on and trace_rows and self.adapter.trace is not None:
                cuts = self.cut_builder.build(self.adapter.trace,
                                              batch_seed=global_step)
                if not cuts:
                    # Monitoring hole fix: a batch with no valid cuts must be
                    # visible, not silent.
                    self.monitor.alert("warning", "kf_no_cuts",
                                       "batch produced no valid cuts",
                                       step=global_step)
                    main_loss = task_loss
                else:
                    scores, skipped = self.kf_tracker.update(cuts)
                    skipped_types.update(skipped)
                    if skipped:
                        self.monitor.alert(
                            "warning", "kf_gated_tau",
                            "gated below min unique cuts: {}"
                            .format(sorted(skipped)), step=global_step)
                    if scores:
                        kf_term = kf_loss(scores, self.rpbe_cfg.alphas)
                        kf_v = float(kf_term.detach())
                        kf_detail = {tau: float(j.detach())
                                     for tau, j in scores.items()}
                        for tau, jv in kf_detail.items():
                            kf_sum[tau] = kf_sum.get(tau, 0.0) + jv
                            kf_count_tau[tau] = kf_count_tau.get(tau, 0) + 1
                        kf_count += 1
                        # J bound is min(r, n-1) with the EFFECTIVE sample
                        # size, not the P sketch dim.
                        dims = {}
                        for tau in kf_detail:
                            n_eff = self.kf_tracker.effective_n(tau)
                            dims[tau] = int(min(self.rpbe_cfg.interfaces[tau],
                                                max(1, n_eff - 1)))
                        self.monitor.validate_kf(kf_detail, dims, global_step)
                        main_loss = task_loss + self.lambda_kf * kf_term
                    else:
                        main_loss = task_loss
            else:
                main_loss = task_loss

            self.monitor.validate_losses({
                "task": float(task_loss.detach()),
                "kf": kf_v,
                "main_total": float(main_loss.detach()),
            }, global_step)

            main_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.optimizer.param_groups[0]["params"]
                     if p.grad is not None], max_norm=self.grad_clip,
                    error_if_nonfinite=True)
            self.optimizer.step()

            # Upstream truncation invariant: detach the memory graph when the
            # host can carry gradients.
            if self.tgn.use_memory and (self.finetune_host or self.rpbe_on):
                self.tgn.memory.detach_memory()

            train_probs.append(pred.detach().cpu().numpy())
            train_labels.append(labels_np)
            total_task += float(task_loss.detach())
            total_kf += kf_v
            n_batches += 1
            global_step += 1

        train_metrics = metric_bundle(np.concatenate(train_labels),
                                      np.concatenate(train_probs))
        kf_out = None
        if kf_count and self.rpbe_cfg is not None:
            # Per-tau denominators: each tau averages over its own scored
            # batches, and J_frac uses the saturation-aware bound
            # min(r_tau, n_eff-1), not the sketch dim m.
            j_frac = {}
            for tau, v in kf_sum.items():
                n_eff = self.kf_tracker.effective_n(tau)
                bound = min(self.rpbe_cfg.interfaces[tau], max(1, n_eff - 1))
                j_frac[tau] = (v / kf_count_tau.get(tau, 1)) / bound
            kf_out = {
                "J": {tau: v / kf_count_tau.get(tau, 1)
                      for tau, v in kf_sum.items()},
                "J_frac": j_frac,
                "skipped_types": sorted(skipped_types),
                "kf_loss": total_kf / max(n_batches, 1),
            }
        return {
            "train_task_loss": total_task / max(n_batches, 1),
            "train_kf_loss": total_kf / max(n_batches, 1),
            "kf": kf_out,
            "train": train_metrics,
            "n_batches": n_batches,
            "global_step": global_step,
        }

    # --------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate_split(self, split: object, *, reset: bool = False) -> Dict:
        """AUC/AP/NLL over one split; advances the stream with ground truth.

        ``reset=False`` continues from the memory left by the previous split
        (upstream semantics: validation follows the train stream).
        """
        if reset:
            self.reset_memory()
        before = rpbe_fingerprint(self.fixed_maps)
        self.tgn.eval()
        self.decoder.eval()
        if self.adapter is not None:
            self.adapter.clear_trace()
        probs, labels = [], []
        observed_dims = None
        for k in range(math.ceil(len(split.sources) / self.batch_size)):
            s, e = k * self.batch_size, min(len(split.sources),
                                            (k + 1) * self.batch_size)
            src_emb, _, _ = self._full_official_embedding_call(
                split.sources[s:e], split.destinations[s:e],
                split.timestamps[s:e], split.edge_idxs[s:e],
                grad_enabled=False)
            if observed_dims is None:
                observed_dims = {"source": int(src_emb.shape[-1])}
            probs.append(self.decoder(src_emb).sigmoid().cpu().numpy())
            labels.append(split.labels[s:e])
        trace_created = bool(self.adapter is not None
                             and self.adapter.trace is not None)
        assert_clean(before, self.fixed_maps, trace_created, "evaluate_split")
        out = metric_bundle(np.concatenate(labels), np.concatenate(probs))
        out["embedding_dims_observed"] = observed_dims or {}
        return out

    @torch.no_grad()
    def replay_split(self, split: object) -> None:
        """Rebuild the stream from zero over a split, no scores (test setup)."""
        before = rpbe_fingerprint(self.fixed_maps)
        self.tgn.eval()
        if self.adapter is not None:
            self.adapter.clear_trace()
        for k in range(math.ceil(len(split.sources) / self.batch_size)):
            s, e = k * self.batch_size, min(len(split.sources),
                                            (k + 1) * self.batch_size)
            self._full_official_embedding_call(
                split.sources[s:e], split.destinations[s:e],
                split.timestamps[s:e], split.edge_idxs[s:e],
                grad_enabled=False)
        trace_created = bool(self.adapter is not None
                             and self.adapter.trace is not None)
        assert_clean(before, self.fixed_maps, trace_created, "replay_split")
