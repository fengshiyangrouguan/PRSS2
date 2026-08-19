from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .outside import Occurrence, Trace


def _masked_mean(values, valid):
    w = valid.to(values.dtype).unsqueeze(-1)
    return (values * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


class PRSSTGNEmbeddingAdapter(nn.Module):
    """Minimal wrapper around the official recursive GraphEmbedding.

    Host aggregation is untouched. The only changed interface is each recursive return:
      exact pre-aggregation tensors -> rich candidate h -> shared spectral quotient R_l -> z
    and the parent receives z only.

    Trace instrumentation is training-only and can be limited to chosen top-level source rows.
    """
    def __init__(self, host_embedding, prss_core):
        super().__init__()
        if not hasattr(host_embedding, "aggregate"):
            raise ValueError("PRSS requires TGN graph_attention/graph_sum recursive embedding")
        self.host = host_embedding
        self.prss = prss_core
        self.embedding_dimension = host_embedding.embedding_dimension
        self.device = host_embedding.device
        self.n_layers = host_embedding.n_layers
        self.n_edge_features = host_embedding.n_edge_features
        self.n_time_features = host_embedding.n_time_features
        self.use_memory = host_embedding.use_memory
        self._trace_top_rows = set()
        self.trace = None
        self._next_oid = 0

    @property
    def neighbor_finder(self):
        return self.host.neighbor_finder

    @neighbor_finder.setter
    def neighbor_finder(self, value):
        self.host.neighbor_finder = value

    def set_trace_source_rows(self, rows: Sequence[int]):
        self._trace_top_rows = set(int(x) for x in rows)

    def clear_trace(self):
        self._trace_top_rows = set()
        self.trace = None

    def _new_occurrence(self, layer, candidate, local, children, relations, deltas):
        oid = self._next_oid
        self._next_oid += 1
        self.trace.occurrences[oid] = Occurrence(
            oid=oid, layer=int(layer), candidate=candidate,
            local=local, children=list(children), relations=list(relations), deltas=list(deltas))
        return oid

    def _parent_local(self, source_time, edge_time, edge_features, mask):
        b = edge_features.shape[0]
        return torch.cat([
            source_time.reshape(b, -1),
            edge_time.reshape(b, -1),
            edge_features.reshape(b, -1),
            mask.to(edge_features.dtype).reshape(b, -1),
        ], dim=-1)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20,
                          time_diffs=None, use_time_proj=True):
        source_nodes = np.asarray(source_nodes)
        timestamps = np.asarray(timestamps)
        if int(n_neighbors) <= 0:
            raise ValueError("PRSS requires n_neighbors > 0 so the host interface width is fixed")
        active = np.zeros(len(source_nodes), dtype=bool)
        for r in self._trace_top_rows:
            if 0 <= r < len(active):
                active[r] = True
        if active.any():
            self.trace = Trace()
            self._next_oid = 0
        else:
            self.trace = None
        z, ids = self._compute(memory, source_nodes, timestamps, int(n_layers), int(n_neighbors), active)
        if self.trace is not None:
            roots, rows = [], []
            for row in np.flatnonzero(active):
                oid = int(ids[row])
                if oid >= 0:
                    roots.append(oid)
                    rows.append(int(row))
            self.trace.roots = roots
            self.trace.root_rows = rows
        return z

    def _compute(self, memory, source_nodes, timestamps, layer, n_neighbors, active):
        device = self.device
        source_nodes_t = torch.from_numpy(source_nodes).long().to(device)
        timestamps_t = torch.from_numpy(timestamps).float().to(device).unsqueeze(1)
        source_time = self.host.time_encoder(torch.zeros_like(timestamps_t))
        raw_source = self.host.node_features[source_nodes_t]
        if self.use_memory:
            raw_source = memory[source_nodes] + raw_source

        ids = np.full(len(source_nodes), -1, dtype=np.int64)
        if layer == 0:
            candidate = self.prss.make_candidate(0, raw_source)
            z = self.prss.project(0, candidate)
            if self.trace is not None and active.any():
                local = torch.zeros(len(source_nodes), self.prss.parent_local_dim,
                                    device=device, dtype=raw_source.dtype)
                for row in np.flatnonzero(active):
                    ids[row] = self._new_occurrence(0, candidate[row], local[row], [], [], [])
            return z, ids

        source_lower, source_ids = self._compute(
            memory, source_nodes, timestamps, layer - 1, n_neighbors, active)
        neighbors, edge_idxs_np, edge_times = self.neighbor_finder.get_temporal_neighbor(
            source_nodes, timestamps, n_neighbors=n_neighbors)
        neighbors_t = torch.from_numpy(neighbors).long().to(device)
        edge_idxs = torch.from_numpy(edge_idxs_np).long().to(device)
        edge_deltas_np = timestamps[:, None] - edge_times
        edge_deltas = torch.from_numpy(edge_deltas_np).float().to(device)
        flat_neighbors = neighbors.reshape(-1)
        repeated_times = np.repeat(timestamps, n_neighbors)
        neighbor_active = np.repeat(active, n_neighbors)
        neighbor_lower, neighbor_ids = self._compute(
            memory, flat_neighbors, repeated_times, layer - 1, n_neighbors, neighbor_active)
        neighbor_lower = neighbor_lower.view(len(source_nodes), n_neighbors, -1)
        neighbor_ids = neighbor_ids.reshape(len(source_nodes), n_neighbors)
        edge_time = self.host.time_encoder(edge_deltas)
        edge_features = self.host.edge_features[edge_idxs]
        mask = neighbors_t == 0
        vanilla = self.host.aggregate(
            layer, source_lower, source_time, neighbor_lower, edge_time, edge_features, mask)
        candidate = self.prss.make_candidate(
            layer, vanilla, source_lower, source_time, neighbor_lower, edge_time, edge_features, mask)
        z = self.prss.project(layer, candidate)

        if self.trace is not None and active.any():
            local = self._parent_local(source_time, edge_time, edge_features, mask)
            for row in np.flatnonzero(active):
                children, relations, deltas = [], [], []
                sid = int(source_ids[row])
                if sid >= 0:
                    children.append(sid); relations.append(0); deltas.append(0.0)
                for j in range(n_neighbors):
                    nid = int(neighbor_ids[row, j])
                    if nid >= 0 and not bool(mask[row, j]):
                        children.append(nid); relations.append(1); deltas.append(float(edge_deltas_np[row, j]))
                ids[row] = self._new_occurrence(
                    layer, candidate[row], local[row], children, relations, deltas)
        return z, ids
