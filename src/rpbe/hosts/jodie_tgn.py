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
        # ``edge_tables`` = (idx -> (src, dst), idx -> label, user set,
        # page set) from ``records.build_edge_tables``; used ONLY to stamp
        # each neighbor occurrence's consumption record.  None (tests /
        # no-data paths) just skips the stamping.
        if edge_tables is not None:
            self._endpoints = dict(edge_tables[0])
            self._user_nodes = set(edge_tables[2])
        else:
            self._endpoints = None
            self._user_nodes = None
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors <= 0:
            raise ValueError(
                "RPBE requires n_neighbors > 0 so the host interface width is fixed")

        if compressor is not None:
            # Host-width contract: d_tau must equal the width the host
            # attention actually consumes; a mismatch would only surface as
            # a shape error inside the vendored aggregate.
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
        # ``jodie_preagg_dim`` remains exported for interface-width docs but
        # the adapter no longer materializes the preagg packing (the only
        # consumer, local_features, was removed as wasted GiB-scale storage).

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
        # Trace scope (fifth review): the official TGN concatenates
        # [source, destination, negative] roots and calls
        # compute_embedding ONCE (model/tgn.py:148).  The node-
        # classification protocol trains on task-source roots only:
        # traced rows must live in the first third.  POS_DST shadow audit
        # and NEG exclusion are future work; this assert makes the current
        # scope explicit instead of an accidental artifact of row indexing.
        if self._trace_top_rows:
            if len(source_nodes) % 3 != 0:
                raise ValueError(
                    "concatenated [src, dst, neg] roots expected, got {}"
                    .format(len(source_nodes)))
            B = len(source_nodes) // 3
            assert max(self._trace_top_rows) < B, \
                "trace scope: only task-source roots (rows < {}) may be " \
                "traced".format(B)
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
                for row in np.flatnonzero(active):
                    ids[row] = self._new_occurrence(
                        tau, z[row], [], [], [],
                        node=int(source_nodes[row]),
                        time=float(timestamps[row]))
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

        # The host aggregate (with children's compressed states as input) is
        # the child-aggregation token of Gamma; the compressor stacks A/G/Q on
        # top without touching the aggregation itself.
        z = vanilla if self.compressor is None else self.compressor.compress(
            tau=tau, own_input=raw_source, aggregate_output=vanilla)

        if self.trace is not None and active.any():
            # Perf: read the numpy neighbors array, never CUDA scalar bools.
            np_neighbors = np.asarray(neighbors)
            for row in np.flatnonzero(active):
                children, relations, deltas = [], [], []
                sid = int(source_ids[row])
                if sid >= 0:
                    children.append(sid)
                    relations.append(0)
                    deltas.append(0.0)
                    # SELF recursion step: no interaction of its own; the
                    # cut walker skips it upward (path keeps the step).
                    self.trace.occurrences[sid].metadata.setdefault(
                        "consumption", {"kind": "self"})
                for j in range(n_neighbors):
                    nid = int(neighbor_ids[row, j])
                    if nid >= 0 and int(np_neighbors[row, j]) != 0:
                        children.append(nid)
                        relations.append(1)
                        deltas.append(float(edge_deltas_np[row, j]))
                        # Consumption record of the neighbor occurrence: the
                        # real historical interaction through which the
                        # parent consumed its state.  ``edge_idx`` is the
                        # 1-based graph_df.idx carried by the neighbor
                        # finder; endpoints/label owner come from the
                        # explicit tables (no indexing convention assumed).
                        child_occ = self.trace.occurrences[nid]
                        cons = {"kind": "edge",
                                "edge_idx": int(edge_idxs_np[row, j]),
                                "edge_time": float(edge_times[row, j]),
                                "counterpart": int(source_nodes[row])}
                        if self._endpoints is not None:
                            src, dst = self._endpoints.get(
                                int(edge_idxs_np[row, j]), (-1, -1))
                            carrier = int(np_neighbors[row, j])
                            # JODIE semantics (fifth review): the label is
                            # the SOURCE user's state change.  The owner is
                            # ALWAYS src — never inferred from whatever the
                            # carrier happens to be.  endpoint_role only
                            # describes the carrier's side (0=source,
                            # 1=destination); the probe's validity comes
                            # from the owner lying on the continuation
                            # path, which a historical edge always does.
                            if self._user_nodes is not None:
                                assert src in self._user_nodes, \
                                    "label owner (source) must be a user"
                            if src == carrier:
                                cons.update({"endpoint_role": 0,
                                             "label_owner": int(src)})
                            elif dst == carrier:
                                cons.update({"endpoint_role": 1,
                                             "label_owner": int(src)})
                            else:
                                cons.update({"endpoint_role": -1,
                                             "label_owner": -1})
                        child_occ.metadata["consumption"] = cons
                ids[row] = self._new_occurrence(
                    tau, z[row], children, relations, deltas,
                    node=int(source_nodes[row]),
                    time=float(timestamps[row]))
        return z, ids

    def _new_occurrence(self, tau, z, children, relations, deltas, *,
                        node: int, time: float):
        oid = self._next_oid
        self._next_oid += 1
        self.trace.add(RecursiveOccurrence(
            occurrence_id=oid, tau=tau,
            state=OccurrenceState(tau=tau, z=z),
            children=list(children),
            child_relations=list(relations),
            child_delta_t=list(deltas),
            metadata={"layer": int(tau.split(":")[1][len("layer"):]),
                      "node": int(node), "time": float(time)},
        ))
        return oid
