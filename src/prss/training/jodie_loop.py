"""JODIE node-classification training loop on the official TGN host.

Protocol clone of upstream ``official_tgn/train_supervised.py`` plus the v1
matched corrections: 10 supervised epochs, BCE over natural labels (no
negative sampling), memory reset at epoch start, chronological train ->
validation replay (no reset between train and val, upstream semantics), best
held-out selection, then zero-memory train+val replay before the held-out
test.  The PRSS block is the same block-coordinate design as the TGB line:
outside continuation contexts per traced root, response+spectral losses on the
main path, unrestricted reader monitored through its own optimizer, Gram/SVD
statistics hard-gated and audited around every evaluation.
"""

import math
import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from prss.compressors import InterfaceData
from prss.training.isolation import assert_clean, counts_of_spectral, r_copies

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
    """B1 hook: how traced roots are picked within a batch.

    ``positive_first`` replicates v1: all positives up to ``max_roots``, then
    deterministic negatives.  ``evenly_spaced`` is the TGB-line selector.
    ``off`` disables tracing entirely (vanilla-equivalent training dynamics).
    """
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
        raise ValueError("unknown --trace-mode {}".format(mode))
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
    """Owns the official-TGN stream and the PRSS auxiliary block."""

    def __init__(self, *, tgn, decoder, adapter, bridge, prss_core, optimizer,
                 unrestricted_optimizer, device, batch_size, n_neighbors,
                 grad_clip, lambda_resp, lambda_spec, trace_roots, trace_mode,
                 spectral_warmup, spectral_interval, monitor, seed,
                 finetune_host=False, selection_metric="auc"):
        self.tgn = tgn
        self.decoder = decoder
        self.adapter = adapter
        self.bridge = bridge
        self.prss_core = prss_core
        self.optimizer = optimizer
        self.unrestricted_optimizer = unrestricted_optimizer
        self.device = device
        self.batch_size = int(batch_size)
        self.n_neighbors = int(n_neighbors)
        self.grad_clip = float(grad_clip)
        self.lambda_resp = float(lambda_resp)
        self.lambda_spec = float(lambda_spec)
        self.trace_roots = int(trace_roots)
        self.trace_mode = trace_mode
        self.spectral_warmup = int(spectral_warmup)
        self.spectral_interval = int(spectral_interval)
        self.monitor = monitor
        self.seed = int(seed)
        self.finetune_host = bool(finetune_host)
        self.selection_metric = selection_metric
        if self.bridge is not None:
            self.aux_use_resp, self.aux_use_spec = prss_core.aux_contract()
        else:
            self.aux_use_resp = self.aux_use_spec = False
        self.compressive_taus = list(prss_core.compressive_interfaces) \
            if prss_core is not None else []

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
        """One supervised epoch over the chronological train stream.

        ``train`` is a JodieData-like object (sources/destinations/timestamps/
        edge_idxs/labels arrays).
        """
        self.reset_memory()
        # Official train_supervised.py passes whether the host is trainable;
        # a frozen host stays in eval mode (no dropout) exactly as upstream
        # does for the pretrained checkpoint.
        self.tgn.train(self.finetune_host)
        self.decoder.train()
        if self.prss_core is not None:
            self.prss_core.train()

        total_task = total_resp = total_spec = total_unres = 0.0
        n_batches = 0
        batch_index = 0
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
            if self.unrestricted_optimizer is not None:
                self.unrestricted_optimizer.zero_grad(set_to_none=True)

            trace_rows = []
            if self.adapter is not None:
                trace_rows = select_trace_rows(
                    labels_np, self.trace_roots, self.seed, global_step,
                    self.trace_mode)
                self.adapter.set_trace_source_rows(trace_rows)

            src_emb, _, _ = self._full_official_embedding_call(
                sources, dests, times, edge_idxs,
                grad_enabled=(self.finetune_host or self.prss_core is not None))
            logits = self.decoder(src_emb)
            pred = logits.sigmoid()
            # Upstream node-classification objective exactly: sigmoid + BCE.
            task_loss = F.binary_cross_entropy(pred, labels_t)

            aux = None
            resp_v = spec_v = unres_v = 0.0
            if self.bridge is not None and trace_rows:
                root_rows = list(self.adapter.trace.root_rows)
                aux = self.bridge.build(times[root_rows],
                                        labels_t[root_rows].detach())
                resp_v = float(aux.response_loss.detach())
                spec_v = float(aux.spectral_loss.detach())
                unres_v = float(aux.unrestricted_loss.detach())
                main_loss = task_loss
                if self.aux_use_resp:
                    main_loss = main_loss + self.lambda_resp * aux.response_loss
                if self.aux_use_spec:
                    main_loss = main_loss + self.lambda_spec * aux.spectral_loss
            else:
                main_loss = task_loss

            self.monitor.validate_losses({
                "task": float(task_loss.detach()),
                "response": resp_v,
                "spectral": spec_v,
                "unrestricted_monitor": unres_v,
                "main_total": float(main_loss.detach()),
            }, global_step)

            main_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.optimizer.param_groups[0]["params"]
                     if p.grad is not None], max_norm=self.grad_clip,
                    error_if_nonfinite=True)
            self.optimizer.step()

            if self.unrestricted_optimizer is not None and aux is not None:
                aux.unrestricted_loss.backward()
                self.unrestricted_optimizer.step()

            # Upstream truncation invariant: detach the memory graph when the
            # host can carry gradients (PRSS core trains even if the host is
            # frozen, so this always applies in PRSS mode).
            if self.tgn.use_memory and (self.finetune_host
                                        or self.prss_core is not None):
                self.tgn.memory.detach_memory()

            # Block-coordinate spectral statistic / update. R is never a
            # gradient parameter.
            if self.prss_core is not None and aux is not None:
                with torch.no_grad():
                    stats = {}
                    for tau in self.compressive_taus:
                        stats[tau] = InterfaceData(
                            candidates=(self.adapter.traced_candidates(tau)
                                        if self.adapter is not None else None),
                            reader_matrices=aux.matrices_by_tau.get(tau))
                    self.prss_core.update_statistics(global_step, stats)
                    completed = global_step + 1
                    if (completed >= self.spectral_warmup
                            and completed % self.spectral_interval == 0):
                        for tau, updated in self.prss_core.maybe_update(
                                completed).items():
                            if updated:
                                snap = self.prss_core.quotients[tau].snapshot()
                                print(
                                    f"SVD_UPDATE step={completed} tau={tau} "
                                    f"total={snap.get('spectral_updates')} "
                                    f"rank={snap.get('effective_predictive_rank')} "
                                    f"energy@k={snap.get('energy_at_k', 0):.6f} "
                                    f"tail@k={snap.get('tail_at_k', 0):.6f} "
                                    f"gain={snap.get('captured_energy_gain', 0):.6f}",
                                    flush=True)

            train_probs.append(pred.detach().cpu().numpy())
            train_labels.append(labels_np)
            total_task += float(task_loss.detach())
            total_resp += resp_v
            total_spec += spec_v
            total_unres += unres_v
            n_batches += 1
            batch_index += 1
            global_step += 1

        train_metrics = metric_bundle(np.concatenate(train_labels),
                                      np.concatenate(train_probs))
        return {
            "train_task_loss": total_task / max(n_batches, 1),
            "train_response_loss": total_resp / max(n_batches, 1),
            "train_spectral_loss": total_spec / max(n_batches, 1),
            "train_unrestricted_loss": total_unres / max(n_batches, 1),
            "train": train_metrics,
            "n_batches": n_batches,
            "global_step": global_step,
        }

    # --------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate_split(self, split: object, *, reset: bool = False) -> Dict:
        """AUC/AP/NLL over one split; advances the stream with ground truth.

        ``reset=False`` continues from the memory left by the previous split
        (upstream semantics: validation follows the train stream).  The trace
        is cleared and spectral updates are hard-gated; callers audit the
        state around this.
        """
        if reset:
            self.reset_memory()
        self.tgn.eval()
        self.decoder.eval()
        if self.adapter is not None:
            self.adapter.clear_trace()
        if self.prss_core is not None:
            self.prss_core.set_spectral_updates_allowed(False)
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
                observed_dims = {
                    "source": int(src_emb.shape[-1]),
                }
            probs.append(self.decoder(src_emb).sigmoid().cpu().numpy())
            labels.append(split.labels[s:e])
        out = metric_bundle(np.concatenate(labels), np.concatenate(probs))
        out["embedding_dims_observed"] = observed_dims or {}
        return out

    @torch.no_grad()
    def replay_split(self, split: object) -> None:
        """Rebuild the stream from zero over a split, no scores (test setup)."""
        self.tgn.eval()
        if self.adapter is not None:
            self.adapter.clear_trace()
        if self.prss_core is not None:
            self.prss_core.set_spectral_updates_allowed(False)
        for k in range(math.ceil(len(split.sources) / self.batch_size)):
            s, e = k * self.batch_size, min(len(split.sources),
                                            (k + 1) * self.batch_size)
            self._full_official_embedding_call(
                split.sources[s:e], split.destinations[s:e],
                split.timestamps[s:e], split.edge_idxs[s:e],
                grad_enabled=False)

    # ----------------------------------------------------------------- auditing
    def audit_before(self):
        return counts_of_spectral(self.prss_core), r_copies(self.prss_core)

    def audit_after(self, before_counts, before_r, label):
        trace_created = bool(self.adapter is not None
                             and self.adapter.trace is not None)
        assert_clean(before_counts, before_r, self.prss_core, trace_created,
                     label)

    def reenable_spectral(self):
        if self.prss_core is not None:
            self.prss_core.set_spectral_updates_allowed(True)
