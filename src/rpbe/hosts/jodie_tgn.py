"""RPBE adapter for the official twitter-research TGN host.

The host's recursive aggregation remains intact.  Gamma is inserted only at
an *internal aggregated state* that is passed to another aggregation:

* layer 0 (leaf): return the raw host state;
* 0 < layer < L: aggregate normally, then apply Gamma;
* layer L (task root): return the vanilla host aggregate.

This is both the semantic boundary and the deployment fast path.  Leaves have
nothing to aggregate, while the root is consumed by the task head rather than
another tree node.  With ``compressor=None`` the wrapper is bit-identical to
the official host.

Training trace collection is fused into this same query.  It records only the
internal states on selected roots' query-node (SELF) spines; temporal-neighbor
states are still fully computed and aggregated, but no recursive tree object
is allocated for them.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from rpbe.hosts.base import HostAdapter
from rpbe.state import CompactCutTrace, CutCandidate

TAU_TEMPLATE = "tjo:layer{}"


def jodie_preagg_dim(host_dim: int, time_dim: int, edge_dim: int,
                     n_neighbors: int) -> int:
    """Width of the documented aggregate-input packing."""
    return host_dim + time_dim + n_neighbors * (host_dim + time_dim + edge_dim) \
        + n_neighbors


class JodieTGNAdapter(HostAdapter):
    """Wrap the official ``GraphEmbedding.compute_embedding`` surface."""

    def __init__(self, host_embedding, compressor=None, n_neighbors: int = 10,
                 edge_tables=None):
        super().__init__()
        if not hasattr(host_embedding, "aggregate"):
            raise ValueError(
                "RPBE requires TGN graph_attention/graph_sum recursive embedding")
        if not hasattr(host_embedding, "time_encoder"):
            raise ValueError("RPBE requires a host time_encoder")
        self.host = host_embedding
        self.compressor = compressor
        # Kept only as a source-compatible constructor argument.  Historical
        # edge tables must never supervise a cut; JodieFutureIndex owns the
        # strictly-future, train-only outcome lookup in records.py.
        del edge_tables
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors <= 0:
            raise ValueError(
                "RPBE requires n_neighbors > 0 so the host interface width is fixed")

        if compressor is not None:
            for tau, d_tau in compressor.cfg.state_dims.items():
                if int(d_tau) != int(host_embedding.embedding_dimension):
                    raise ValueError(
                        "state_dims[{}]={} must equal host width {}".format(
                            tau, d_tau, host_embedding.embedding_dimension))
        self.embedding_dimension = host_embedding.embedding_dimension
        self.device = host_embedding.device
        self.n_layers = host_embedding.n_layers
        self.n_edge_features = host_embedding.n_edge_features
        self.n_time_features = host_embedding.n_time_features
        self.use_memory = host_embedding.use_memory

        self.taus = [TAU_TEMPLATE.format(layer)
                     for layer in range(self.n_layers + 1)]
        # The only interfaces at which Gamma and the KF objective are valid.
        self.compression_taus = [TAU_TEMPLATE.format(layer)
                                 for layer in range(1, self.n_layers)]

        self._trace_top_rows = set()
        self._trace = None
        # Global across queries, so CutRecord identities cannot collide when
        # a lagged moment window spans batches or epochs.
        self._next_oid = 0

    # ------------------------------------------------------------ tracing hooks
    def set_trace_source_rows(self, rows: Sequence[int]) -> None:
        self._trace_top_rows = set(int(x) for x in rows)

    def clear_trace(self) -> None:
        self._trace_top_rows = set()
        self._trace = None

    @property
    def trace(self) -> Optional[CompactCutTrace]:
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
        """Official call surface with optional, bounded in-query tracing."""
        del time_diffs, use_time_proj
        source_nodes = np.asarray(source_nodes)
        timestamps = np.asarray(timestamps)
        if int(n_neighbors) != self.n_neighbors:
            raise ValueError("adapter fixed n_neighbors {} but got {}".format(
                self.n_neighbors, n_neighbors))
        if int(n_layers) != int(self.n_layers):
            raise ValueError("adapter root layer {} but got {}".format(
                self.n_layers, n_layers))

        # Official TGN queries [src, positive-dst, negative-dst] together.
        # Only task-source and optional positive-destination rows may be
        # traced; negative placeholders are not observations.
        if self._trace_top_rows:
            if len(source_nodes) % 3 != 0:
                raise ValueError(
                    "concatenated [src, dst, neg] roots expected, got {}"
                    .format(len(source_nodes)))
            batch_size = len(source_nodes) // 3
            if min(self._trace_top_rows) < 0 or \
                    max(self._trace_top_rows) >= 2 * batch_size:
                raise ValueError(
                    "trace rows must select src/positive-dst roots in [0, {})"
                    .format(2 * batch_size))

        root_rows = sorted(r for r in self._trace_top_rows
                           if 0 <= r < len(source_nodes))
        if root_rows:
            self._trace = CompactCutTrace(root_rows=list(root_rows))
            paths: Dict[int, List[Tuple[int, float]]] = {
                int(row): [] for row in root_rows}
        else:
            self._trace = None
            paths = {}
        return self._compute(memory, source_nodes, timestamps,
                             int(n_layers), int(n_neighbors), paths)

    def _compute(self, memory, source_nodes, timestamps, layer, n_neighbors,
                 trace_paths):
        """Recursive host query plus a SELF-spine trace token dictionary.

        ``trace_paths`` maps rows in this recursion call to their top-level
        structural path.  Tokens go only through ``source_lower``.  The
        neighbor recursion receives no tokens, which removes full-tree trace
        allocation without changing any embedding computation.
        """
        device = self.device
        source_nodes_t = torch.from_numpy(source_nodes).long().to(device)
        timestamps_t = torch.from_numpy(timestamps).float().to(device).unsqueeze(1)
        source_time = self.host.time_encoder(torch.zeros_like(timestamps_t))
        raw_source = self.host.node_features[source_nodes_t]
        if self.use_memory:
            raw_source = memory[source_nodes] + raw_source

        # A leaf has no child aggregation, so Gamma is undefined here.
        if layer == 0:
            return raw_source

        source_paths = {
            int(row): list(path) + [(0, 0.0)]
            for row, path in trace_paths.items()}
        source_lower = self._compute(
            memory, source_nodes, timestamps, layer - 1, n_neighbors,
            source_paths)

        neighbors, edge_idxs_np, edge_times = \
            self.neighbor_finder.get_temporal_neighbor(
                source_nodes, timestamps, n_neighbors=n_neighbors)
        neighbors_t = torch.from_numpy(neighbors).long().to(device)
        edge_idxs = torch.from_numpy(edge_idxs_np).long().to(device)
        edge_deltas_np = timestamps[:, None] - edge_times
        edge_deltas = torch.from_numpy(edge_deltas_np).float().to(device)
        flat_neighbors = neighbors.reshape(-1)
        repeated_times = np.repeat(timestamps, n_neighbors)
        # Neighbor states are computed exactly as before; only trace tokens
        # are absent, so no neighbor occurrence objects are retained.
        neighbor_lower = self._compute(
            memory, flat_neighbors, repeated_times, layer - 1, n_neighbors,
            {})
        neighbor_lower = neighbor_lower.view(len(source_nodes), n_neighbors, -1)
        edge_time = self.host.time_encoder(edge_deltas)
        edge_features = self.host.edge_features[edge_idxs]
        mask = neighbors_t == 0
        vanilla = self.host.aggregate(
            layer, source_lower, source_time, neighbor_lower, edge_time,
            edge_features, mask)

        # Only an internal aggregate is passed upward to another tree node.
        if self.compressor is not None and 0 < layer < self.n_layers:
            tau = TAU_TEMPLATE.format(layer)
            z = self.compressor.compress(
                tau=tau, own_input=raw_source, aggregate_output=vanilla)
        else:
            z = vanilla

        # Root and leaf are deliberately absent.  Each selected query emits
        # at most one graph-connected state for this internal interface.
        if self._trace is not None and 0 < layer < self.n_layers:
            tau = TAU_TEMPLATE.format(layer)
            for row, path in trace_paths.items():
                node = int(source_nodes[row])
                if node == 0:  # official padding sentinel is not an event node
                    continue
                self._trace.add(CutCandidate(
                    occurrence_id=self._next_oid,
                    root_row=int(row),
                    tau=tau,
                    node=node,
                    time=float(timestamps[row]),
                    z=z[row],
                    # Pre-compression rich state for the reconstruction
                    # ablation (Table 2, row 2).
                    u=vanilla[row],
                    path=list(path)))
                self._next_oid += 1
        return z
