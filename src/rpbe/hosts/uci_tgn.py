"""UCI (BenchTemp-protocol) host adapter: BenchTemp GraphEmbedding-shaped
recursion with Gamma insertion + consumed child-parent pair tracing.

Mirrors ``JodieTGNAdapter`` (tgbl path) one-to-one in protocol, adapted to
the BenchTemp TGN recursion shape::

    source_lower is the node/memory feature of THIS layer (not a lower
    aggregation — BenchTemp's aggregate() fuses [node/memory features] with
    [neighbor lower recursion] at every layer), and the layer counter counts
    DOWN from n_layers.

Gamma is inserted only at internal layers (0 < layer < n_layers), same
interface contract as the tgbl host::

    z = compress(tau, own_input=raw_source, aggregate_output=vanilla)

``compute_embedding`` keeps the official signature (memory, source_nodes,
timestamps, n_layers, n_neighbors, time_diffs, use_time_proj) so the adapter
drops into BenchTemp's ``TGN.compute_temporal_embeddings`` unchanged.
``time_diffs`` is accepted and ignored exactly like the official
GraphEmbedding does.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from rpbe.state import CompactCutTrace, ConsumedPairCandidate, CutCandidate

TAU_TEMPLATE = "tjo:layer{}"
NEIGHBOR_REL = 1


class UciTGNAdapter(nn.Module):
    """Recursive BenchTemp-shaped host query + trace + Gamma compression."""

    def __init__(self, host_embedding, compressor=None, n_neighbors: int = 10,
                 trace_pairs_per_parent: int = 0, retain_child_u: bool = False):
        super().__init__()
        self.host = host_embedding
        self.compressor = compressor
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be > 0")
        self.trace_pairs_per_parent = int(trace_pairs_per_parent)
        self.retain_child_u = bool(retain_child_u)
        self.device = host_embedding.device

        if compressor is not None:
            for tau, d_tau in compressor.cfg.state_dims.items():
                if int(d_tau) != int(host_embedding.embedding_dimension):
                    raise ValueError(
                        "state_dims[{}]={} must equal host width {}".format(
                            tau, d_tau, host_embedding.embedding_dimension))
        self.embedding_dimension = host_embedding.embedding_dimension
        # internal compressible interfaces (0 < layer < n_layers)
        _nl = int(getattr(host_embedding, "n_layers", 1))
        self.compression_taus = [TAU_TEMPLATE.format(l)
                                 for l in range(1, _nl)]

        # trace state (same contract as jodie_tgn)
        self._trace_top_rows: set = set()
        self._trace = None
        self._trace_batch = 0
        self._next_oid = 0
        self._next_pair_id = 0

    @property
    def neighbor_finder(self):
        """Proxy to the host finder (official eval swaps it to the full
        graph finder via set_neighbor_finder / attribute assignment)."""
        return self.host.neighbor_finder

    @neighbor_finder.setter
    def neighbor_finder(self, finder) -> None:
        self.host.neighbor_finder = finder

    # ------------------------------------------------------------- trace API
    def set_trace_source_rows(self, rows) -> None:
        self._trace_top_rows = set(int(x) for x in rows)

    def set_trace_batch(self, batch_index: int) -> None:
        self._trace_batch = int(batch_index)

    def clear_trace(self) -> None:
        self._trace_top_rows = set()
        self._trace = None

    @property
    def trace(self) -> Optional[CompactCutTrace]:
        return self._trace

    # ---------------------------------------------------------------- entry
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers,
                          n_neighbors=20, time_diffs=None, use_time_proj=True):
        """Official BenchTemp signature; trace selection lives on the wrapper
        state (set_trace_source_rows / set_trace_batch)."""
        source_nodes = np.asarray(source_nodes)
        timestamps = np.asarray(timestamps)
        if int(n_neighbors) != self.n_neighbors:
            raise ValueError("adapter fixed n_neighbors {} but got {}".format(
                self.n_neighbors, n_neighbors))
        root_rows = sorted(r for r in self._trace_top_rows
                           if 0 <= r < len(source_nodes))
        if root_rows:
            self._trace = CompactCutTrace(root_rows=list(root_rows))
            paths: Dict[int, List[Tuple[int, float]]] = {
                int(row): [] for row in root_rows}
        else:
            self._trace = None
            paths = {}
        z, _ = self._compute(memory, source_nodes, timestamps,
                             int(n_layers), int(n_neighbors), paths)
        return z

    # ------------------------------------------------------------- recursion
    def _compute(self, memory, source_nodes, timestamps, layer, n_neighbors,
                 trace_paths):
        """BenchTemp GraphEmbedding.compute_embedding recursion + Gamma.

        Returns ``(z, u_pre)`` where z is the compressed (or vanilla) state
        passed upward and u_pre is the pre-compression vanilla aggregate
        (A1 reconstruction target).
        """
        source_nodes_t = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_t = torch.from_numpy(timestamps).float().to(self.device)
        timestamps_t = timestamps_t.unsqueeze(1)
        source_time = self.host.time_encoder(
            torch.zeros_like(timestamps_t))
        raw_source = self.host.node_features[source_nodes_t]
        if self.host.use_memory:
            raw_source = memory[source_nodes, :] + raw_source

        if layer == 0:
            return raw_source, raw_source

        neighbors, edge_idxs_np, edge_times = \
            self.host.neighbor_finder.get_temporal_neighbor(
                source_nodes, timestamps, n_neighbors=n_neighbors)
        neighbors_t = torch.from_numpy(neighbors).long().to(self.device)
        edge_idxs = torch.from_numpy(edge_idxs_np).long().to(self.device)
        edge_deltas_np = timestamps[:, None] - edge_times
        edge_deltas = torch.from_numpy(edge_deltas_np).float().to(self.device)
        flat_neighbors = neighbors.reshape(-1)
        repeated_times = np.repeat(timestamps, n_neighbors)

        neighbor_lower, neighbor_u = self._compute(
            memory, flat_neighbors, repeated_times, layer - 1, n_neighbors,
            {})
        neighbor_lower = neighbor_lower.view(
            len(source_nodes), n_neighbors, -1)
        neighbor_u = (neighbor_u.view(len(source_nodes), n_neighbors, -1)
                      if neighbor_u is not None else None)

        edge_time = self.host.time_encoder(edge_deltas)
        edge_features = self.host.edge_features[edge_idxs]
        mask = neighbors_t == 0

        # -------- consumed child-parent pair tracing (recursive closure) ----
        if self._trace is not None and self.trace_pairs_per_parent > 0 \
                and trace_paths and 0 < layer - 1 < int(self._trace_n_layers()):
            parent_layer = layer
            child_layer = layer - 1
            child_tau = TAU_TEMPLATE.format(int(child_layer))
            for prow, path in trace_paths.items():
                pnode = int(source_nodes[prow])
                if pnode == 0:
                    continue
                ptime = float(timestamps[prow])
                nonpad = [s for s in range(n_neighbors)
                          if int(neighbors[prow, s]) != 0]
                if not nonpad:
                    continue
                rng = np.random.RandomState(
                    ((self._trace_batch * 1000003) ^ (int(prow) * 104729)
                     ^ (int(parent_layer) * 7919)) & 0xFFFFFFFF)
                k = min(self.trace_pairs_per_parent, len(nonpad))
                slots = [nonpad[i] for i in
                         rng.choice(len(nonpad), size=k, replace=False)]
                for slot in slots:
                    child_node = int(neighbors[prow, slot])
                    rel_edge_id = int(edge_idxs_np[prow, slot])
                    rel_time = float(edge_times[prow, slot])
                    child_z = neighbor_lower[prow, slot]
                    child_u = (neighbor_u[prow, slot]
                               if (neighbor_u is not None
                                   and self.retain_child_u) else None)
                    child_path = tuple(list(path) + [(NEIGHBOR_REL,
                                                      ptime - rel_time)])
                    pair_id = (self._next_pair_id, int(prow),
                               int(parent_layer), int(slot))
                    self._trace.add_pair(ConsumedPairCandidate(
                        pair_id=pair_id,
                        root_row=int(prow),
                        tau=child_tau,
                        child_layer=int(child_layer),
                        parent_layer=int(parent_layer),
                        child_node=child_node,
                        child_time=ptime,
                        parent_node=pnode,
                        parent_time=ptime,
                        relation_time=rel_time,
                        relation_edge_id=rel_edge_id,
                        relation_lag=ptime - rel_time,
                        relation_slot=int(slot),
                        path=child_path,
                        z=child_z,
                        u=child_u))
                    self._next_pair_id += 1

        vanilla = self.host.aggregate(
            layer, raw_source, source_time, neighbor_lower, edge_time,
            edge_features, mask)

        if self.compressor is not None and 0 < layer < self._trace_n_layers():
            tau = TAU_TEMPLATE.format(int(layer))
            z = self.compressor.compress(
                tau=tau, own_input=raw_source, aggregate_output=vanilla)
        else:
            z = vanilla

        if self._trace is not None and 0 < layer < self._trace_n_layers():
            tau = TAU_TEMPLATE.format(int(layer))
            for row, path in trace_paths.items():
                node = int(source_nodes[row])
                if node == 0:
                    continue
                self._trace.add(CutCandidate(
                    occurrence_id=self._next_oid,
                    root_row=int(row),
                    tau=tau,
                    node=node,
                    time=float(timestamps[row]),
                    z=z[row],
                    u=vanilla[row],
                    path=list(path)))
                self._next_oid += 1
        return z, vanilla

    def _trace_n_layers(self) -> int:
        """Total host layer count (read once from the host if cached)."""
        if not hasattr(self, "_n_layers_cached"):
            self._n_layers_cached = int(self.host.n_layers) \
                if hasattr(self.host, "n_layers") else 1
        return self._n_layers_cached
