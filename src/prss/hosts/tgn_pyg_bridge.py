"""Training-only continuation contexts for TGB link prediction (PyG TGN host).

Four scenarios per traced event, mirroring the archived TGNLinkOutsideBridge:
  (source tree,  counterpart = positive destination, role 0, y=1)
  (dest tree,    counterpart = positive source,      role 1, y=1)
  (source tree,  counterpart = negative destination, role 0, y=0)
  (neg-dest tree,counterpart = positive source,      role 1, y=0)

Root metadata = [counterpart quotient (detached); normalized log time; role].
The label enters losses only and never the context encoder; negative samples are
random draws, unrelated to any future event.
"""

from collections import defaultdict
from typing import List

import torch

from prss.auxiliary import AuxiliaryBatch
from prss.hosts.base import OutsideBridge
from prss.losses import response_loss


class TGBLinkOutsideBridge(OutsideBridge):
    def __init__(self, adapter, prss_core, time_mean: float = 0.0,
                 time_std: float = 1.0, max_nodes_per_scenario: int = 64,
                 response_task: str = "binary"):
        self.adapter = adapter
        self.prss = prss_core
        self.time_mean = float(time_mean)
        self.time_std = max(float(time_std), 1e-12)
        self.max_nodes_per_scenario = int(max_nodes_per_scenario)
        self.response_task = response_task
        expected_root = adapter.emb_dim + 2
        if prss_core.config.root_metadata_dim != expected_root:
            raise ValueError("link root_metadata_dim must be emb_dim + 2 ({})".format(expected_root))

    def _root_metadata(self, counterpart, timestamp, role):
        normalized = (torch.log1p(timestamp.clamp_min(0)) - self.time_mean) / self.time_std
        normalized = normalized.reshape(-1, 1)
        role_vec = torch.full_like(normalized, float(role))
        return torch.cat([counterpart.detach().reshape(1, -1), normalized, role_vec], dim=-1)

    def build(self, z, timestamps, src_local, pos_local, neg_local, trace_rows) -> AuxiliaryBatch:
        """z: quotient states [N, emb] of the forward; *_local: local indices into n_id
        aligned with the batch; trace_rows: which rows to supervise."""
        if self.adapter.trace is None:
            raise RuntimeError("Enable PRSS trace before the host forward pass")
        row_to_time = {int(r): timestamps[i] for i, r in enumerate(trace_rows)}

        structured_logits, unrestricted_logits, targets = [], [], []
        readers_by_tau = defaultdict(list)
        spectral_terms = []
        for row in trace_rows:
            t = row_to_time[int(row)]
            s_l, p_l, n_l = int(src_local[row]), int(pos_local[row]), int(neg_local[row])
            scenarios = [
                (s_l, z[p_l], t, 0, 1.0),
                (p_l, z[s_l], t, 1, 1.0),
                (s_l, z[n_l], t, 0, 0.0),
                (n_l, z[s_l], t, 1, 0.0),
            ]
            for root_local, counterpart, time, role, target in scenarios:
                oid = self.adapter.occurrence_for_local(root_local)
                if oid is None:
                    continue
                metadata = self._root_metadata(counterpart, time, role)
                for tau, struct, unres, matrix in self._outside_for_root(oid, metadata):
                    structured_logits.append(struct)
                    unrestricted_logits.append(unres)
                    targets.append(target)
                    readers_by_tau[tau].append(matrix)
                    spectral_terms.append(self.prss.spectral_loss(tau, matrix))

        if not structured_logits:
            zero = z.sum() * 0.0
            return AuxiliaryBatch(zero, zero, zero, z[:0], z[:0], z[:0], {}, {}, {})

        struct_t = torch.stack(structured_logits).squeeze(-1)
        unres_t = torch.stack(unrestricted_logits).squeeze(-1)
        target_t = torch.as_tensor(targets, device=struct_t.device, dtype=struct_t.dtype)
        response = response_loss(struct_t.unsqueeze(-1), target_t, task=self.response_task)
        unres_loss = response_loss(unres_t.unsqueeze(-1), target_t, task=self.response_task)
        spectral = torch.stack(spectral_terms).mean()
        matrices = {tau: torch.stack(v, dim=0).detach().clone()
                    for tau, v in readers_by_tau.items()}
        return AuxiliaryBatch(
            response_loss=response,
            spectral_loss=spectral,
            unrestricted_loss=unres_loss,
            structured_logits=struct_t.detach(),
            unrestricted_logits=unres_t.detach(),
            targets=target_t,
            matrices_by_tau=matrices,
            occurrence_counts={tau: int(v.shape[0]) for tau, v in matrices.items()},
            contexts_by_tau={},
        )

    def _outside_for_root(self, root_id: int, root_metadata: torch.Tensor):
        """One top-down outside pass; returns (tau, structured_logits, unrestricted_logits, B)."""
        trace = self.adapter.trace
        root = trace.occurrences[int(root_id)]
        contexts = {int(root_id): self.prss.outside.root_context(root_metadata, root.tau).squeeze(0)}
        queue: List[int] = [int(root_id)]
        order: List[int] = []
        visited = set()
        while queue:
            pid = queue.pop(0)
            if pid in visited:
                continue
            visited.add(pid)
            order.append(pid)
            parent = trace.occurrences[pid]
            pc = contexts[pid]
            for pos, cid in enumerate(parent.children):
                child = trace.occurrences[cid]
                siblings_by_tau = defaultdict(list)
                for sid in parent.children:
                    if sid == cid:
                        continue
                    sibling = trace.occurrences[sid]
                    siblings_by_tau[sibling.tau].append(sibling.state.candidate)
                sibling_tensors = {tau: torch.stack(v, dim=0) for tau, v in siblings_by_tau.items()}
                summary = self.prss.outside.summarize_siblings(sibling_tensors, pc)
                contexts[cid] = self.prss.outside.child_context(
                    parent_outside=pc,
                    parent_local=parent.local_features,
                    relation_ids=parent.child_relations[pos],
                    delta_t=parent.child_delta_t[pos],
                    sibling_summary=summary,
                    child_type=child.tau,
                )
                queue.append(cid)
        if self.max_nodes_per_scenario > 0 and len(order) > self.max_nodes_per_scenario:
            import numpy as np
            indexes = np.linspace(0, len(order) - 1, self.max_nodes_per_scenario, dtype=np.int64)
            order = [order[int(i)] for i in indexes]
        outputs = []
        for oid in order:
            occ = trace.occurrences[oid]
            tau = occ.tau
            if tau not in self.prss.readers:
                continue
            context = contexts[oid]
            cand = occ.state.candidate.unsqueeze(0)
            logits, B, _ = self.prss.structured_read(tau, context.unsqueeze(0), cand)
            ulogit = self.prss.unrestricted_read(tau, context.detach().unsqueeze(0),
                                                 cand.detach())
            outputs.append((tau, logits, ulogit, B))
        return outputs
