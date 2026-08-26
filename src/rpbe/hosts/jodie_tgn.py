"""RPBE adapter for the official twitter-research TGN host (JODIE protocol).

The official ``GraphEmbedding`` computes a recursive L-layer temporal-attention
tree per query node: layer l's occurrence has children = its own layer-(l-1)
embedding plus the ``n_degree`` layer-(l-1) embeddings of its temporal
neighbors.  Each recursive layer is one interface (``tjo:layer{l}``), so the
host computation tree maps 1:1 onto the cut tree.

The host aggregate is never replaced: the adapter computes the vanilla
aggregate (over the children's compressed states) and passes it to the
recursive compressor as the child-aggregation token, which stacks A/G/Q on
top.  With no compressor attached (``compressor=None``) the forward is
bit-identical to the official host.  The memory lives entirely inside TGN;
the adapter only wraps ``embedding_module``.
"""

from typing import Optional, Sequence

import numpy as np
import torch

from rpbe.hosts.base import HostAdapter
from rpbe.state import OccurrenceState, RecursiveOccurrence, RecursiveTrace

TAU_TEMPLATE = "tjo:layer{}"


def jodie_preagg_dim(host_dim: int, time_dim: int, edge_dim: int,
                     n_neighbors: int) -> int:
    """Width of the exact aggregate-input packing:
    [source_lower, source_time, n x (neighbor_lower, edge_time, edge_features),
     mask]."""
    return host_dim + time_dim + n_neighbors * (host_dim + time_dim + edge_dim) \
        + n_neighbors


class JodieTGNAdapter(HostAdapter):
    """Wraps the official ``GraphAttentionEmbedding``; ``TGN.compute_temporal_embeddings``
    keeps its exact call surface (``compute_embedding`` with numpy arrays)."""

    def __init__(self, host_embedding, compressor=None, n_neighbors: int = 10):
        super().__init__()
        if not hasattr(host_embedding, "aggregate"):
            raise ValueError(
                "RPBE requires TGN graph_attention/graph_sum recursive embedding")
        if not hasattr(host_embedding, "time_encoder"):
            raise ValueError("RPBE requires a host time_encoder")
        self.host = host_embedding
        self.compressor = compressor
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors <= 0:
            raise ValueError(
                "RPBE requires n_neighbors > 0 so the host interface width is fixed")

        self.embedding_dimension = host_embedding.embedding_dimension
        self.device = host_embedding.device
        self.n_layers = host_embedding.n_layers
        self.n_edge_features = host_embedding.n_edge_features
        self.n_time_features = host_embedding.n_time_features
        self.use_memory = host_embedding.use_memory

        self.taus = [TAU_TEMPLATE.format(layer)
                     for layer in range(self.n_layers + 1)]
        self.preagg_dim = jodie_preagg_dim(self.embedding_dimension,
                                           self.n_time_features,
                                           self.n_edge_features,
                                           self.n_neighbors)
        # C1 (legacy): the child-state block of local_features is zeroed.
        # Nothing consumes local_features today; the field is kept for
        # diagnostics.
        self._local_zero_dim = self.embedding_dimension \
            + self.n_neighbors * self.embedding_dimension

        self._trace_top_rows = set()
        self._trace = None
        self._next_oid = 0

    # ------------------------------------------------------------ tracing hooks
    def set_trace_source_rows(self, rows: Sequence[int]) -> None:
        self._trace_top_rows = set(int(x) for x in rows)

    def clear_trace(self) -> None:
        self._trace_top_rows = set()
        self._trace = None

    @property
    def trace(self) -> Optional[RecursiveTrace]:
        return self._trace

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
            self._trace = RecursiveTrace()
            self._next_oid = 0
        else:
            self._trace = None
        z, ids = self._compute(memory, source_nodes, timestamps,
                               int(n_layers), int(n_neighbors), active)
        if self._trace is not None:
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
            # No children: identity passes the raw state through untouched;
            # the compressor receives it in both slots (own input == aggregate).
            z = raw_source if self.compressor is None else self.compressor.compress(
                tau=tau, own_input=raw_source, aggregate_output=raw_source)
            if self.trace is not None and active.any():
                local = torch.zeros(len(source_nodes), self.preagg_dim,
                                    device=device, dtype=raw_source.dtype)
                for row in np.flatnonzero(active):
                    ids[row] = self._new_occurrence(
                        tau, z[row], local[row], [], [], [],
                        node=int(source_nodes[row]),
                        time=float(timestamps[row]),
                        own_raw=raw_source[row].detach())
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

        # The host aggregate (with children's compressed states as input) is
        # the child-aggregation token of Gamma; the compressor stacks A/G/Q on
        # top without touching the aggregation itself.
        z = vanilla if self.compressor is None else self.compressor.compress(
            tau=tau, own_input=raw_source, aggregate_output=vanilla)

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
                    tau, z[row], local[row], children, relations, deltas,
                    node=int(source_nodes[row]),
                    time=float(timestamps[row]),
                    own_raw=raw_source[row].detach())
        return z, ids

    def _new_occurrence(self, tau, z, local, children, relations, deltas, *,
                        node: int, time: float, own_raw):
        oid = self._next_oid
        self._next_oid += 1
        self.trace.add(RecursiveOccurrence(
            occurrence_id=oid, tau=tau,
            state=OccurrenceState(tau=tau, z=z),
            local_features=local,
            children=list(children),
            child_relations=list(relations),
            child_delta_t=list(deltas),
            metadata={"layer": int(tau.split(":")[1][len("layer"):]),
                      "node": int(node), "time": float(time),
                      "own_raw": own_raw},
        ))
        return oid
