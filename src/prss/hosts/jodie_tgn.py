"""PRSS adapter for the official twitter-research TGN host (JODIE protocol).

The official ``GraphEmbedding`` computes a recursive L-layer temporal-attention
tree per query node: layer l's occurrence has children = its own layer-(l-1)
embedding plus the ``n_degree`` layer-(l-1) embeddings of its temporal
neighbors.  Each recursive layer is one PRSS interface (``tjo:layer{l}``), so
the host computation tree maps 1:1 onto the outside continuation tree.

The host aggregate is never replaced: the adapter computes the vanilla
aggregate, widens it into the candidate, projects it, and passes only the
quotient up.  Without tracing the forward is bit-identical to the official
host.  The memory lives entirely inside TGN; the adapter only wraps
``embedding_module``.
"""

from typing import Optional, Sequence

import numpy as np
import torch

from prss.hosts.base import HostAdapter
from prss.state import QuotientState, RecursiveOccurrence, RecursiveTrace

TAU_TEMPLATE = "tjo:layer{}"


def jodie_preagg_dim(host_dim: int, time_dim: int, edge_dim: int,
                     n_neighbors: int) -> int:
    """Width of the exact aggregate-input packing:
    [source_lower, source_time, n x (neighbor_lower, edge_time, edge_features),
     mask] (order identical to the v1 exact_preagg)."""
    return host_dim + time_dim + n_neighbors * (host_dim + time_dim + edge_dim) \
        + n_neighbors


class JodieTGNAdapter(HostAdapter):
    """Wraps the official ``GraphAttentionEmbedding``; ``TGN.compute_temporal_embeddings``
    keeps its exact call surface (``compute_embedding`` with numpy arrays)."""

    def __init__(self, host_embedding, prss_core, n_neighbors: int):
        super().__init__()
        if not hasattr(host_embedding, "aggregate"):
            raise ValueError(
                "PRSS requires TGN graph_attention/graph_sum recursive embedding")
        if not hasattr(host_embedding, "time_encoder"):
            raise ValueError("PRSS requires a host time_encoder")
        self.host = host_embedding
        self.prss = prss_core
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors <= 0:
            raise ValueError(
                "PRSS requires n_neighbors > 0 so the host interface width is fixed")

        self.embedding_dimension = host_embedding.embedding_dimension
        self.device = host_embedding.device
        self.n_layers = host_embedding.n_layers
        self.n_edge_features = host_embedding.n_edge_features
        self.n_time_features = host_embedding.n_time_features
        self.use_memory = host_embedding.use_memory

        self.taus = [TAU_TEMPLATE.format(layer)
                     for layer in range(self.n_layers + 1)]
        if sorted(prss_core.config.interfaces) != sorted(self.taus):
            raise ValueError("PRSS interface keys must be {}".format(self.taus))
        host_dim = self.embedding_dimension
        time_dim = self.n_time_features
        edge_dim = self.n_edge_features
        for tau in self.taus:
            spec = prss_core.config.interface(tau)
            if spec.host_dim != host_dim:
                raise ValueError("host_dim must equal embedding_dimension for {}".format(tau))
        self.preagg_dim = jodie_preagg_dim(host_dim, time_dim, edge_dim,
                                           self.n_neighbors)
        if prss_core.config.parent_local_dim != self.preagg_dim:
            raise ValueError("config.parent_local_dim must equal jodie_preagg_dim "
                             "({})".format(self.preagg_dim))
        # C1: outside must not see the subtree's own states; the child-state
        # block (source_lower + neighbor_lower) is zeroed in local_features.
        self._local_zero_dim = host_dim + self.n_neighbors * host_dim

        self._trace_top_rows = set()
        self.trace = None
        self._next_oid = 0

    # ------------------------------------------------------------ tracing hooks
    def set_trace_source_rows(self, rows: Sequence[int]) -> None:
        self._trace_top_rows = set(int(x) for x in rows)

    def clear_trace(self) -> None:
        self._trace_top_rows = set()
        self.trace = None

    @property
    def trace(self) -> Optional[RecursiveTrace]:
        return self._trace

    def traced_candidates(self, tau: Optional[str] = None) -> Optional[torch.Tensor]:
        """Detached candidate rows of the current trace (PCA statistic source)."""
        if self.trace is None:
            return None
        values = [occ.state.candidate.detach()
                  for occ in self.trace.occurrences.values()
                  if tau is None or occ.tau == tau]
        if not values:
            return None
        return torch.stack(values, dim=0)

    @property
    def neighbor_finder(self):
        return self.host.neighbor_finder

    @neighbor_finder.setter
    def neighbor_finder(self, value):
        self.host.neighbor_finder = value

    # ------------------------------------------------------------- main forward
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers,
                          n_neighbors=20, time_diffs=None, use_time_proj=True):
        """Exact official call surface; tracing is active-row-limited and
        training-only (bit-identical forward otherwise)."""
        source_nodes = np.asarray(source_nodes)
        timestamps = np.asarray(timestamps)
        if int(n_neighbors) != self.n_neighbors:
            raise ValueError("adapter fixed n_neighbors {} but got {}".format(
                self.n_neighbors, n_neighbors))
        active = np.zeros(len(source_nodes), dtype=bool)
        for row in self._trace_top_rows:
            if 0 <= row < len(active):
                active[row] = True
        if active.any():
            self.trace = RecursiveTrace()
            self._next_oid = 0
        else:
            self.trace = None
        z, ids = self._compute(memory, source_nodes, timestamps,
                               int(n_layers), int(n_neighbors), active)
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

    def _compute(self, memory, source_nodes, timestamps, layer, n_neighbors,
                 active):
        device = self.device
        source_nodes_t = torch.from_numpy(source_nodes).long().to(device)
        timestamps_t = torch.from_numpy(timestamps).float().to(device).unsqueeze(1)
        source_time = self.host.time_encoder(torch.zeros_like(timestamps_t))
        raw_source = self.host.node_features[source_nodes_t]
        if self.use_memory:
            raw_source = memory[source_nodes] + raw_source

        ids = np.full(len(source_nodes), -1, dtype=np.int64)
        if layer == 0:
            tau = TAU_TEMPLATE.format(0)
            cand = self.prss.make_candidate(tau, raw_source)
            z = self.prss.project(tau, cand)
            if self.trace is not None and active.any():
                local = torch.zeros(len(source_nodes), self.preagg_dim,
                                    device=device, dtype=raw_source.dtype)
                for row in np.flatnonzero(active):
                    ids[row] = self._new_occurrence(
                        tau, raw_source[row].detach(), cand[row], z[row],
                        local[row], [], [], [])
            return z, ids

        tau = TAU_TEMPLATE.format(layer)
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
            memory, flat_neighbors, repeated_times, layer - 1, n_neighbors,
            neighbor_active)
        neighbor_lower = neighbor_lower.view(len(source_nodes), n_neighbors, -1)
        neighbor_ids = neighbor_ids.reshape(len(source_nodes), n_neighbors)
        edge_time = self.host.time_encoder(edge_deltas)
        edge_features = self.host.edge_features[edge_idxs]
        mask = neighbors_t == 0
        vanilla = self.host.aggregate(
            layer, source_lower, source_time, neighbor_lower, edge_time,
            edge_features, mask)

        b = len(source_nodes)
        flat_preagg = torch.cat([
            source_lower,
            source_time.reshape(b, -1),
            neighbor_lower.reshape(b, -1),
            edge_time.reshape(b, -1),
            edge_features.reshape(b, -1),
            mask.to(source_lower.dtype).reshape(b, -1),
        ], dim=-1)
        if flat_preagg.shape[-1] != self.preagg_dim:
            raise ValueError("preagg width mismatch")
        cand = self.prss.make_candidate(tau, vanilla, flat_preagg)
        z = self.prss.project(tau, cand)

        if self.trace is not None and active.any():
            local = flat_preagg.detach().clone()
            local[:, :self._local_zero_dim] = 0.0
            for row in np.flatnonzero(active):
                children, relations, deltas = [], [], []
                sid = int(source_ids[row])
                if sid >= 0:
                    children.append(sid)
                    relations.append(0)
                    deltas.append(0.0)
                for j in range(n_neighbors):
                    nid = int(neighbor_ids[row, j])
                    if nid >= 0 and not bool(mask[row, j]):
                        children.append(nid)
                        relations.append(1)
                        deltas.append(float(edge_deltas_np[row, j]))
                ids[row] = self._new_occurrence(
                    tau, vanilla[row].detach(), cand[row], z[row], local[row],
                    children, relations, deltas)
        return z, ids

    def _new_occurrence(self, tau, raw, cand, z, local, children, relations,
                        deltas):
        oid = self._next_oid
        self._next_oid += 1
        self.trace.add(RecursiveOccurrence(
            occurrence_id=oid, tau=tau,
            state=QuotientState(tau=tau, raw=raw, candidate=cand, quotient=z),
            local_features=local,
            children=list(children),
            child_relations=list(relations),
            child_delta_t=list(deltas),
            metadata={"layer": int(tau.split(":")[1][len("layer"):])},
        ))
        return oid
