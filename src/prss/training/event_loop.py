"""Host-orchestration training loop for the TGB link-prediction protocol.

Follows the official TGB tgn.py baseline order-of-operations exactly (reset state
per epoch, forward before memory update, update_state + insert + detach per batch),
with the PRSS auxiliary/spectral block inserted between the forward and the
memory update.  Validation/test run with tracing off and spectral updates
hard-gated, audited before and after.
"""

import time
from typing import Dict, Optional

import numpy as np
import torch

from prss.compressors import InterfaceData
from prss.hosts.tgn_pyg import TAU
from prss.training.isolation import assert_clean, counts_of_spectral, r_copies

EPS = 1e-7


def _metric_bundle(y_pred_pos, y_pred_neg, evaluator, metric):
    input_dict = {
        "y_pred_pos": np.array([y_pred_pos.cpu()]),
        "y_pred_neg": np.array([y_pred_neg.cpu()]),
        "eval_metric": [metric],
    }
    return float(evaluator.eval(input_dict)[metric])


def select_trace_rows(batch_size: int, max_roots: int, seed: int, batch_index: int) -> list:
    """Deterministic evenly-spaced root selection (all events are positives)."""
    if max_roots <= 0 or batch_size == 0:
        return []
    rng = np.random.RandomState(seed + 104729 * (batch_index + 1))
    rows = np.linspace(0, batch_size - 1, min(max_roots, batch_size)).astype(np.int64)
    return [int(r) for r in rows]


class TGBLinkPredictionLoop:
    """Owns the memory/loader state machines and all forward/eval orchestration."""

    def __init__(self, *, dataset, memory, gnn, link_pred, neighbor_loader,
                 adapter, bridge, prss_core, optimizer, unrestricted_optimizer,
                 criterion, device, batch_size, n_neighbors, grad_clip,
                 lambda_resp, lambda_spec, trace_roots, spectral_warmup,
                 spectral_interval, monitor, seed):
        self.dataset = dataset
        self.memory = memory
        self.gnn = gnn
        self.link_pred = link_pred
        self.neighbor_loader = neighbor_loader
        self.adapter = adapter
        self.bridge = bridge
        self.prss_core = prss_core
        self.optimizer = optimizer
        self.unrestricted_optimizer = unrestricted_optimizer
        self.criterion = criterion
        self.device = device
        self.batch_size = int(batch_size)
        self.n_neighbors = int(n_neighbors)
        self.grad_clip = float(grad_clip)
        self.lambda_resp = float(lambda_resp)
        self.lambda_spec = float(lambda_spec)
        self.trace_roots = int(trace_roots)
        self.spectral_warmup = int(spectral_warmup)
        self.spectral_interval = int(spectral_interval)
        self.monitor = monitor
        self.seed = int(seed)
        self.aux_use_resp, self.aux_use_spec = (
            prss_core.aux_contract() if prss_core is not None else (False, False))
        self.evaluator = dataset.evaluator
        self.neg_sampler = dataset._ds.negative_sampler
        self.metric = dataset.eval_metric
        self.assoc = torch.empty(dataset.num_nodes, dtype=torch.long, device=device)

    # ------------------------------------------------------------ state machines
    def reset_stream(self):
        self.memory.reset_state()
        self.neighbor_loader.reset_state()

    def _forward_nodes(self, n_id, edge_index, e_id, root_local_ids=None, root_times=None):
        data = self.dataset._data
        if self.adapter is not None:
            return self.adapter.embed(n_id, edge_index, data.t[e_id].to(self.device),
                                      data.msg[e_id].to(self.device),
                                      root_local_ids, root_times)
        z_mem, last_update = self.memory(n_id)
        return self.gnn(z_mem, last_update, edge_index, data.t[e_id].to(self.device),
                        data.msg[e_id].to(self.device))

    def _advance_stream(self, src, pos_dst, t, msg):
        self.memory.update_state(src, pos_dst, t, msg)
        self.neighbor_loader.insert(src, pos_dst)

    # ------------------------------------------------------------------ training
    def train_epoch(self, epoch: int, global_step: int) -> Dict:
        loader = self.dataset.build_loader("train", self.batch_size)
        self.reset_stream()
        total_task = total_resp = total_spec = total_unres = 0.0
        n_batches = 0
        batch_index = 0
        for batch in loader:
            batch = batch.to(self.device)
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            neg_dst = torch.randint(self.dataset.min_dst_idx,
                                    self.dataset.max_dst_idx + 1, (src.size(0),),
                                    dtype=torch.long, device=self.device)
            n_id = torch.cat([src, pos_dst, neg_dst]).unique()
            n_id, edge_index, e_id = self.neighbor_loader(n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

            self.optimizer.zero_grad(set_to_none=True)
            if self.unrestricted_optimizer is not None:
                self.unrestricted_optimizer.zero_grad(set_to_none=True)

            trace_rows = []
            root_local_ids = root_times = None
            if self.adapter is not None:
                trace_rows = select_trace_rows(src.size(0), self.trace_roots,
                                               self.seed, batch_index)
                if trace_rows:
                    local_ids = []
                    for r in trace_rows:
                        local_ids += [int(self.assoc[src[r]]), int(self.assoc[pos_dst[r]]),
                                      int(self.assoc[neg_dst[r]])]
                    root_local_ids = torch.tensor(local_ids, device=self.device)
                    root_times = torch.tensor([float(t[r])] * len(trace_rows), device=self.device)

            z = self._forward_nodes(n_id, edge_index, e_id, root_local_ids, root_times)
            pos_out = self.link_pred(z[self.assoc[src]], z[self.assoc[pos_dst]])
            neg_out = self.link_pred(z[self.assoc[src]], z[self.assoc[neg_dst]])
            task_loss = self.criterion(pos_out, torch.ones_like(pos_out))
            task_loss += self.criterion(neg_out, torch.zeros_like(neg_out))

            aux = None
            resp_v = spec_v = unres_v = 0.0
            if self.bridge is not None and trace_rows:
                aux = self.bridge.build(z, t, self.assoc[src], self.assoc[pos_dst],
                                        self.assoc[neg_dst], trace_rows)
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

            # Ground-truth stream advance, then detach (official baseline order).
            self._advance_stream(src, pos_dst, t, msg)
            self.memory.detach()

            # Block-coordinate spectral statistic / update. R is never a parameter.
            if self.prss_core is not None and aux is not None:
                with torch.no_grad():
                    self.prss_core.update_statistics(global_step, {
                        TAU: InterfaceData(
                            candidates=(self.adapter.traced_candidates(TAU)
                                        if self.adapter is not None else None),
                            reader_matrices=aux.matrices_by_tau.get(TAU)),
                    })
                    completed = global_step + 1
                    if completed >= self.spectral_warmup and completed % self.spectral_interval == 0:
                        for tau, updated in self.prss_core.maybe_update(completed).items():
                            if updated:
                                snap = self.prss_core.quotients[tau].snapshot()
                                print(f"SVD_UPDATE step={completed} tau={tau} "
                                      f"total={snap.get('spectral_updates')} "
                                      f"rank={snap.get('effective_predictive_rank')} "
                                      f"energy@k={snap.get('energy_at_k', 0):.6f} "
                                      f"tail@k={snap.get('tail_at_k', 0):.6f} "
                                      f"gain={snap.get('captured_energy_gain', 0):.6f}", flush=True)

            total_task += float(task_loss.detach()) * batch.num_events
            total_resp += resp_v * batch.num_events
            total_spec += spec_v * batch.num_events
            total_unres += unres_v * batch.num_events
            n_batches += 1
            batch_index += 1
            global_step += 1

        n_events = max(int(self.dataset.train_data.num_events), 1)
        return {
            "train_task_loss": total_task / n_events,
            "train_response_loss": total_resp / n_events,
            "train_spectral_loss": total_spec / n_events,
            "train_unrestricted_loss": total_unres / n_events,
            "n_batches": n_batches,
            "global_step": global_step,
        }

    # --------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate_split(self, split: str) -> Dict:
        """One-vs-many MRR over a split; advances the stream with ground truth."""
        if split == "val":
            self.dataset.load_val_ns()
            split_mode = "val"
        else:
            self.dataset.load_test_ns()
            split_mode = "test"
        loader = self.dataset.build_loader(split, self.batch_size)
        if self.adapter is not None:
            self.adapter.clear_trace()
        if self.prss_core is not None:
            self.prss_core.set_spectral_updates_allowed(False)

        perf_list = []
        for pos_batch in loader:
            pos_batch = pos_batch.to(self.device)
            pos_src, pos_dst, pos_t, pos_msg = (
                pos_batch.src, pos_batch.dst, pos_batch.t, pos_batch.msg)
            neg_batch_list = self.neg_sampler.query_batch(pos_src, pos_dst, pos_t,
                                                          split_mode=split_mode)
            for idx, neg_batch in enumerate(neg_batch_list):
                src = torch.full((1 + len(neg_batch),), pos_src[idx], device=self.device)
                dst = torch.tensor(np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0), device=self.device)
                n_id = torch.cat([src, dst]).unique()
                n_id, edge_index, e_id = self.neighbor_loader(n_id)
                self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
                z = self._forward_nodes(n_id, edge_index, e_id)
                y_pred = self.link_pred(z[self.assoc[src]], z[self.assoc[dst]])
                perf_list.append(_metric_bundle(
                    y_pred[0, :].squeeze(dim=-1), y_pred[1:, :].squeeze(dim=-1),
                    self.evaluator, self.metric))
            self._advance_stream(pos_src, pos_dst, pos_t, pos_msg)

        return {f"test_{self.metric}" if split == "test" else f"val_{self.metric}":
                float(torch.tensor(perf_list).mean())}

    # ------------------------------------------------------------------- auditing
    @torch.no_grad()
    def replay_split(self, split: str) -> None:
        """Advance memory/loader through a split with ground-truth events, no scores.

        Used to rebuild the exact stream state from zero before the held-out test
        (the official state_dict does not persist the Python message queues).
        """
        if self.adapter is not None:
            self.adapter.clear_trace()
        loader = self.dataset.build_loader(split, self.batch_size)
        for batch in loader:
            batch = batch.to(self.device)
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            n_id = torch.cat([src, pos_dst]).unique()
            n_id, edge_index, e_id = self.neighbor_loader(n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
            self._forward_nodes(n_id, edge_index, e_id)
            self._advance_stream(src, pos_dst, t, msg)
            self.memory.detach()

    def audit_before(self):
        return counts_of_spectral(self.prss_core), r_copies(self.prss_core)

    def audit_after(self, before_counts, before_r, label):
        trace_created = bool(self.adapter is not None and self.adapter.trace is not None)
        assert_clean(before_counts, before_r, self.prss_core, trace_created, label)

    def reenable_spectral(self):
        if self.prss_core is not None:
            self.prss_core.set_spectral_updates_allowed(True)
