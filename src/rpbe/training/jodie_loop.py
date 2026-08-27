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

from rpbe.loss import (KFMomentWindow, dedup_cut_rows, kf_vjp_batch,
                       kf_loss)
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
        # KF windows cover the kf_taus whitelist only: the root interface
        # has no upward walk and is excluded EXPLICITLY (no "still
        # accumulating" warnings from a tau that can never close).
        kf_dims = rpbe_cfg.state_dims
        if self.rpbe_on and rpbe_cfg.kf_taus is not None:
            kf_dims = {t: d for t, d in rpbe_cfg.state_dims.items()
                       if t in rpbe_cfg.kf_taus}
        self.kf_window = (KFMomentWindow(
            kf_dims, min_ratio=rpbe_cfg.kf_min_ratio,
            min_abs=rpbe_cfg.kf_min_abs, eps=rpbe_cfg.ridge_eps,
            fixed_maps=fixed_maps)
            if self.rpbe_on else None)

    def _clip_all_groups(self):
        """Gradient clipping across EVERY optimizer parameter group."""
        if self.grad_clip <= 0:
            return
        params = [p for g in self.optimizer.param_groups
                  for p in g["params"] if p.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip,
                                           error_if_nonfinite=True)

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
    def _backup_replay_state(self):
        """Shadow state at macro-window start: memory triple + all RNGs."""
        mem = self.tgn.memory.backup_memory() if self.tgn.use_memory else None
        return (mem, {"torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all()
                      if torch.cuda.is_available() else None,
                      "numpy": np.random.get_state()})

    def _restore_replay_state(self, shadow):
        """Restore the shadow state before the pass-2 replay."""
        mem, rngs = shadow
        if self.tgn.use_memory and mem is not None:
            self.tgn.memory.restore_memory(mem)
        torch.set_rng_state(rngs["torch"])
        if rngs["cuda"] is not None:
            torch.cuda.set_rng_state_all(rngs["cuda"])
        np.random.set_state(rngs["numpy"])

    @staticmethod
    def _root_events(trace_rows, dests, labels_np, times, edge_idxs):
        """Root task records: the tree-level supervisor at the end of the
        upward walk (dst + natural label of the traced event; event_idx =
        the 1-based graph_df.idx)."""
        return {int(row): {"dst": int(dests[row]),
                           "label": float(labels_np[row]),
                           "time": float(times[row]),
                           "event_idx": int(edge_idxs[row])}
                for row in trace_rows}

    def _close_and_replay(self, window_batches, window_shadow, task_only):
        """Pass-1 -> pass-2 bridge.

        Closes the moment windows into small-matrix adjoints (``task_only``
        skips the KF part — the epoch-end drain), restores the shadow
        state, replays every batch WITH gradients: task loss / K plus the
        per-tau KF VJP surrogate, one immediate backward per batch (graph
        freed), ``detach_memory`` after each, ONE optimizer step at the
        end.  Returns ``(kf_v, kf_detail, diag, task_sum, K)``.
        """
        kf_v = 0.0
        kf_detail = {}
        diag = {}
        adjoints = {}
        if not task_only:
            closed, adjoints, diag = self.kf_window.close_replay()
            if closed:
                kf_v = float(sum(self.rpbe_cfg.alpha(tau) * j
                                 for tau, j in closed.items()))
                kf_detail = dict(closed)
        self._restore_replay_state(window_shadow)
        K = float(len(window_batches))
        self.optimizer.zero_grad(set_to_none=True)
        task_sum = 0.0
        for (sources, dests, times, edge_idxs, labels_np, trace_rows,
             step) in window_batches:
            labels_t = torch.from_numpy(labels_np).float().to(self.device)
            if trace_rows:
                self.adapter.set_trace_source_rows(trace_rows)
            src_emb, _, _ = self._full_official_embedding_call(
                sources, dests, times, edge_idxs, grad_enabled=True)
            logits = self.decoder(src_emb)
            pred = logits.sigmoid()
            task_loss = F.binary_cross_entropy(pred, labels_t)
            task_sum += float(task_loss.detach())
            loss = task_loss / K
            if not task_only and adjoints and trace_rows \
                    and self.adapter.trace is not None:
                root_events = self._root_events(
                    trace_rows, dests, labels_np, times, edge_idxs)
                cuts = self.cut_builder.build(
                    self.adapter.trace, root_events=root_events,
                    batch_seed=step)
                for tau in set(c.tau for c in cuts):
                    entry = adjoints.get(tau)
                    if entry is None:
                        continue
                    adj, r = entry
                    tau_rows = [c for c in cuts if c.tau == tau]
                    _, _, _, zs, ps, weights = dedup_cut_rows(
                        tau_rows, self.fixed_maps)
                    loss = loss + self.lambda_kf * kf_vjp_batch(
                        zs, ps, weights, r["mu_z"], r["mu_p"], adj)
            loss.backward()
            # Upstream truncation invariant: detach the memory graph when
            # the host can carry gradients.
            if self.tgn.use_memory and (self.finetune_host or self.rpbe_on):
                self.tgn.memory.detach_memory()
        self._clip_all_groups()
        self.optimizer.step()
        return kf_v, kf_detail, diag, task_sum, K

    def train_epoch(self, epoch: int, global_step: int, train: object) -> Dict:
        """One supervised epoch over the chronological train stream.

        RPBE on: Moment-Adjoint Replay.  Pass 1 (no_grad) runs every batch
        of the macro-window, feeding detached rows into the KF moment
        windows (the REAL score is logged when they close).  The window's
        shadow state is restored and pass 2 replays the same batches with
        gradients — task loss / K plus the KF VJP surrogate — one
        immediate backward per batch, one optimizer.step per macro-window.
        Vanilla keeps the upstream per-batch step.
        """
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
        window_batches = []          # macro-window batch snapshots
        window_shadow = None         # (memory backup, rng states)
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
            if not self.rpbe_on:
                self.optimizer.zero_grad(set_to_none=True)

            trace_rows = []
            if self.adapter is not None:
                trace_rows = select_trace_rows(
                    labels_np, self.trace_roots, self.seed, global_step,
                    self.trace_mode)
                self.adapter.set_trace_source_rows(trace_rows)

            kf_v = 0.0
            if self.rpbe_on:
                # ---------------- pass 1: statistics only (no graph kept)
                if not window_batches:
                    window_shadow = self._backup_replay_state()
                with torch.no_grad():
                    src_emb, _, _ = self._full_official_embedding_call(
                        sources, dests, times, edge_idxs, grad_enabled=False)
                    logits = self.decoder(src_emb)
                    pred = logits.sigmoid()
                    if trace_rows and self.adapter.trace is not None:
                        root_events = self._root_events(
                            trace_rows, dests, labels_np, times, edge_idxs)
                        cuts = self.cut_builder.build(
                            self.adapter.trace, root_events=root_events,
                            batch_seed=global_step)
                        if not cuts:
                            self.monitor.alert(
                                "warning", "kf_no_cuts",
                                "batch produced no valid cuts",
                                step=global_step)
                        else:
                            _, _, gated = self.kf_window.add(cuts)
                            skipped_types.update(gated)
                            if gated:
                                self.monitor.alert(
                                    "warning", "kf_gated_tau",
                                    "still accumulating: {}"
                                    .format(sorted(gated)),
                                    step=global_step)
                window_batches.append(
                    (sources, dests, times, edge_idxs, labels_np,
                     trace_rows, global_step))
            else:
                src_emb, _, _ = self._full_official_embedding_call(
                    sources, dests, times, edge_idxs,
                    grad_enabled=self.finetune_host)
                logits = self.decoder(src_emb)
                pred = logits.sigmoid()
                task_loss = F.binary_cross_entropy(pred, labels_t)
                self.monitor.validate_losses({
                    "task": float(task_loss.detach()),
                    "kf": 0.0,
                    "main_total": float(task_loss.detach()),
                }, global_step)
                task_loss.backward()
                self._clip_all_groups()
                self.optimizer.step()
                if self.tgn.use_memory and self.finetune_host:
                    self.tgn.memory.detach_memory()
                total_task += float(task_loss.detach())

            train_probs.append(pred.detach().cpu().numpy())
            train_labels.append(labels_np)

            if self.rpbe_on and self.kf_window.window_ready():
                kf_v, kf_detail, diag, task_sum, K = self._close_and_replay(
                    window_batches, window_shadow, task_only=False)
                window_batches = []
                window_shadow = None
                total_task += task_sum
                total_kf += kf_v
                for tau, jv in kf_detail.items():
                    kf_sum[tau] = kf_sum.get(tau, 0.0) + jv
                    kf_count_tau[tau] = kf_count_tau.get(tau, 0) + 1
                kf_count += 1
                dims = {}
                for tau in kf_detail:
                    m_u = int(diag[tau]["M_unique"])
                    dims[tau] = int(min(self.rpbe_cfg.state_dims[tau],
                                        max(1, m_u - 1)))
                self.monitor.validate_kf(kf_detail, dims, global_step)
                for tau, jv in kf_detail.items():
                    if diag[tau].get("failed"):
                        self.monitor.alert(
                            "warning", "kf_window_failed",
                            f"{tau} {diag[tau]['failed']} "
                            f"scale_z={diag[tau].get('scale_z', float('nan')):.3e} "
                            f"scale_p={diag[tau].get('scale_p', float('nan')):.3e} "
                            f"M={diag[tau]['M_unique_trees']}",
                            step=global_step, interface=tau)
                self.monitor.validate_losses({
                    "task": task_sum / K,
                    "kf": kf_v,
                    "main_total": task_sum / K + kf_v,
                }, global_step)

            n_batches += 1
            global_step += 1

        # Epoch-end drain: the unfinished macro-window replays task-only
        # (no KF — the moments never closed).
        if self.rpbe_on and window_batches:
            self.monitor.alert("warning", "kf_window_unclosed",
                               "epoch ended with an unclosed KF window; "
                               "task-only replay step", step=global_step)
            _, _, _, task_sum, _ = self._close_and_replay(
                window_batches, window_shadow, task_only=True)
            total_task += task_sum
            self.kf_window.reset()
            window_batches = []

        train_metrics = metric_bundle(np.concatenate(train_labels),
                                      np.concatenate(train_probs))
        kf_out = None
        if kf_count and self.rpbe_cfg is not None:
            # Per-tau denominators: each tau averages over its own closed
            # windows; J_frac uses the saturation-aware bound
            # min(d_tau, M_window-1) of the LAST closed window.
            j_frac = {}
            for tau, v in kf_sum.items():
                m_u = self.kf_window.window_m(tau)
                bound = min(self.rpbe_cfg.state_dims[tau], max(1, m_u - 1))
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
