"""Stage-1 self-supervised pretraining loop on the official TGN host.

Protocol clone of upstream ``old/tgn/train_self_supervised.py``: BCE over the
real interaction (y=1) and one randomly sampled negative destination (y=0),
gradient accumulation over ``backprop_every`` batches, memory reset at epoch
start, ``detach_memory`` after every backward, val AP/AUC early stopping, and
memory backup/restore around evaluation.

With ``--stage1-rpbe`` the adapter + compressor + cut builder (LINK scenario)
are attached: the Ky Fan term ``-lambda_kf * sum alpha J`` accumulates into
the same loss, so the host and Gamma_theta are jointly pretrained from step 0.
"""

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from rpbe.loss import kf_loss, kf_scores_from_rows
from rpbe.training.jodie_loop import select_trace_rows

EPS = 1e-7


class TGNPretrainLoop:
    def __init__(self, *, tgn, optimizer, device, batch_size, n_neighbors,
                 backprop_every, grad_clip, monitor, seed,
                 adapter=None, cut_builder=None, fixed_maps=None,
                 rpbe_cfg=None, lambda_kf=1.0, trace_roots=8):
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
        self.lambda_kf = float(lambda_kf)
        self.trace_roots = int(trace_roots)
        self.rpbe_on = bool(adapter is not None and cut_builder is not None
                            and fixed_maps is not None and rpbe_cfg is not None)
        self._neg_rng = np.random.RandomState(int(seed))

    # ------------------------------------------------------------- stream state
    def reset_memory(self):
        if self.tgn.use_memory:
            self.tgn.memory.__init_memory__()

    def _sample_negatives(self, data, size: int) -> np.ndarray:
        # RandEdgeSampler semantics: negatives drawn from the destination
        # column (the official uniform-over-destinations protocol).
        dst = np.asarray(data.destinations)
        return self._neg_rng.choice(dst, size=size, replace=True)

    # ----------------------------------------------------------------- training
    def train_epoch(self, epoch: int, global_step: int, train) -> Dict:
        self.reset_memory()
        self.tgn.train(True)
        num_batch = math.ceil(len(train.sources) / self.batch_size)
        total_link = total_kf = 0.0
        n_steps = 0
        for k in range(0, num_batch, self.backprop_every):
            self.optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            for j in range(self.backprop_every):
                batch_idx = k + j
                if batch_idx >= num_batch:
                    continue
                s = batch_idx * self.batch_size
                e = min(num_batch, batch_idx + 1) * self.batch_size
                sources = train.sources[s:e]
                dests = train.destinations[s:e]
                times = train.timestamps[s:e]
                edge_idxs = train.edge_idxs[s:e]
                size = len(sources)
                negatives = self._sample_negatives(train, size)

                trace_rows = []
                if self.adapter is not None:
                    # Link pretraining has no labels: trace by deterministic
                    # evenly-spaced rows of this batch.
                    trace_rows = select_trace_rows(
                        np.zeros(size), self.trace_roots, self.seed,
                        global_step, mode="evenly_spaced")
                    self.adapter.set_trace_source_rows(trace_rows)

                pos_prob, neg_prob = self.tgn.compute_edge_probabilities(
                    sources, dests, negatives, times, edge_idxs,
                    self.n_neighbors)
                link_loss = (F.binary_cross_entropy(
                    pos_prob.squeeze(), torch.ones(size, device=self.device))
                    + F.binary_cross_entropy(
                        neg_prob.squeeze(), torch.zeros(size, device=self.device)))
                loss = loss + link_loss

                if self.rpbe_on and trace_rows and self.adapter.trace is not None:
                    cuts = self.cut_builder.build(self.adapter.trace,
                                                  batch_seed=global_step)
                    if cuts:
                        scores, skipped = kf_scores_from_rows(
                            cuts, self.rpbe_cfg.interfaces, self.fixed_maps,
                            min_cuts_per_type=self.rpbe_cfg.min_cuts_per_type,
                            eps=self.rpbe_cfg.ridge_eps)
                        if scores:
                            kf_term = kf_loss(scores, self.rpbe_cfg.alphas)
                            total_kf += float(kf_term.detach())
                            loss = loss + self.lambda_kf * kf_term

                total_link += float(link_loss.detach())
                global_step += 1
            loss = loss / self.backprop_every
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.optimizer.param_groups[0]["params"]
                     if p.grad is not None], max_norm=self.grad_clip,
                    error_if_nonfinite=True)
            self.optimizer.step()
            self.monitor.validate_losses({
                "link": float(total_link) / max(global_step, 1),
                "kf": total_kf / max(global_step, 1),
                "main_total": float(loss.detach()),
            }, global_step)
            if self.tgn.use_memory:
                self.tgn.memory.detach_memory()
            n_steps += 1
        return {"train_link_loss": total_link / max(n_steps, 1),
                "train_kf_loss": total_kf / max(n_steps, 1),
                "n_steps": n_steps, "global_step": global_step}

    # --------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate_edge_prediction(self, split, neg_seed: int) -> Dict:
        """AP/AUC over split edges vs fixed-seed negative destinations."""
        self.tgn.eval()
        if self.adapter is not None:
            self.adapter.clear_trace()
        sampler = np.random.RandomState(neg_seed)
        dst_pool = np.asarray(split.destinations)
        probs, labels = [], []
        for k in range(math.ceil(len(split.sources) / self.batch_size)):
            s = k * self.batch_size
            e = min(len(split.sources), (k + 1) * self.batch_size)
            size = e - s
            negatives = sampler.choice(dst_pool, size=size, replace=True)
            pos_prob, neg_prob = self.tgn.compute_edge_probabilities(
                split.sources[s:e], split.destinations[s:e], negatives,
                split.timestamps[s:e], split.edge_idxs[s:e], self.n_neighbors)
            probs.append(pos_prob.squeeze().cpu().numpy())
            probs.append(neg_prob.squeeze().cpu().numpy())
            labels.append(np.ones(size))
            labels.append(np.zeros(size))
        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
        auc = float(roc_auc_score(labels, probs))
        ap = float(average_precision_score(labels, probs))
        return {"val_ap": ap, "val_auc": auc}
