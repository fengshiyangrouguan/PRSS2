"""PRSS adapter for the PyG-style TGN host (TGB protocol baseline).

The PyG TGN host aggregates over a *sampled one-hop subgraph* with a single
TransformerConv layer — there is no recursive L-layer computation tree as in the
twitter-research host.  The PRSS interface therefore sits at exactly one point:
conv output -> LinkPredictor input, with ``tau = "tgp:node_conv"`` and
``host_dim = emb_dim``.  The training-only computation tree has depth one: an
occurrence's children are its incoming neighbors in the sampled subgraph.

The memory and neighbor-loader state machines are untouched and remain owned by
the training loop; the adapter only intercepts the embedding computation.
"""

from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from prss.config import PRSSConfig
from prss.core import PRSSCore
from prss.hosts.base import HostAdapter
from prss.state import QuotientState, RecursiveOccurrence, RecursiveTrace

TAU = "tgp:node_conv"


def pyg_preagg_dim(mem_dim: int, time_dim: int, msg_dim: int, size: int) -> int:
    """Width of the exact-aggregate-input packing: own memory + per-slot
    (relative-time encoding, edge msg, neighbor memory) + validity mask."""
    return mem_dim + size * (time_dim + msg_dim + mem_dim) + size


class PyGTGNAdapter(HostAdapter):
    """Wraps ``TGNMemory`` + ``GraphAttentionEmbedding``; ``LinkPredictor`` is not wrapped.

    ``embed`` is called with the local subgraph produced by ``LastNeighborLoader``
    (the same call surface as the official TGB tgn.py baseline) and returns the
    host-width quotient states for every node in ``n_id``.
    """

    def __init__(self, memory, gnn, prss_core: PRSSCore, n_neighbors: int,
                 mem_dim: int, time_dim: int, msg_dim: int, emb_dim: int,
                 time_mean: float = 0.0, time_std: float = 1.0):
        super().__init__()
        self.memory = memory
        self.gnn = gnn
        self.prss = prss_core
        self.n_neighbors = int(n_neighbors)
        self.mem_dim = int(mem_dim)
        self.time_dim = int(time_dim)
        self.msg_dim = int(msg_dim)
        self.emb_dim = int(emb_dim)
        self.time_mean = float(time_mean)
        self.time_std = max(float(time_std), 1e-12)
        spec = prss_core.config.interface(TAU)
        if spec.host_dim != self.emb_dim:
            raise ValueError("host_dim must equal gnn output dim")
        self.preagg_dim = pyg_preagg_dim(self.mem_dim, self.time_dim, self.msg_dim,
                                         self.n_neighbors)
        if spec.candidate_dim > spec.raw_dim:
            if prss_core.config.parent_local_dim != self.preagg_dim:
                raise ValueError("config.parent_local_dim must equal pyg_preagg_dim")
        self._trace: Optional[RecursiveTrace] = None
        self._oid_by_local: Dict[int, int] = {}
        self._next_oid = 0
        self._trace_requested = False
        self._last_candidates: Optional[torch.Tensor] = None
        self._last_preagg: Optional[torch.Tensor] = None

    # ------------------------------------------------------------ tracing hooks
    def set_trace_source_rows(self, rows: Sequence[int]) -> None:
        # Rows are converted into local root ids by the caller via ``embed``;
        # this hook only exists for the HostAdapter contract.
        self._trace_requested = True

    def clear_trace(self) -> None:
        self._trace_requested = False
        self._trace = None

    @property
    def trace(self) -> Optional[RecursiveTrace]:
        return self._trace

    def occurrence_for_local(self, local_id: int) -> Optional[int]:
        return self._oid_by_local.get(int(local_id))

    def traced_candidates(self, tau: Optional[str] = None) -> Optional[torch.Tensor]:
        """Detached candidate rows of the current trace (PCA statistic source)."""
        if self._trace is None:
            return None
        values = [occ.state.candidate.detach()
                  for occ in self._trace.occurrences.values()
                  if tau is None or occ.tau == tau]
        if not values:
            return None
        return torch.stack(values, dim=0)

    # ------------------------------------------------------------- main forward
    def embed(self, n_id: torch.Tensor, edge_index: torch.Tensor, t_e: torch.Tensor,
              msg_e: torch.Tensor, root_local_ids: Optional[torch.Tensor] = None,
              root_times: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_mem, last_update = self.memory(n_id)
        vanilla = self.gnn(z_mem, last_update, edge_index, t_e, msg_e)

        preagg, valid, src_sorted, dest_sorted, edge_times_sorted = self._pack_preagg(
            n_id, edge_index, t_e, msg_e, last_update, z_mem)
        cand = self.prss.make_candidate(TAU, vanilla, preagg)
        z = self.prss.project(TAU, cand)

        if root_local_ids is not None:
            self._build_trace(n_id, cand, preagg, src_sorted, dest_sorted,
                              edge_times_sorted, root_local_ids, root_times)
        return z

    def _pack_preagg(self, n_id, edge_index, t_e, msg_e, last_update, z_mem):
        """Vectorized per-node gather of the exact conv aggregate inputs.

        Rows of ``edge_index`` are (neighbor, node) from LastNeighborLoader; every
        node has in-degree <= n_neighbors by the loader invariant.
        """
        device = n_id.device
        N = int(n_id.numel())
        dest = edge_index[1]
        src = edge_index[0]
        counts = torch.bincount(dest, minlength=N)
        if int(counts.max()) > self.n_neighbors:
            raise ValueError("in-degree exceeds n_neighbors; loader invariant broken")
        order = torch.argsort(dest, stable=True)
        dest_sorted = dest[order]
        src_sorted = src[order]
        edge_times_sorted = t_e[order]
        starts = torch.cumsum(counts, dim=0) - counts
        pos = torch.arange(dest_sorted.numel(), device=device) - starts[dest_sorted]

        rel_t = last_update[src_sorted] - edge_times_sorted
        rel_enc = self.gnn.time_enc(rel_t.to(z_mem.dtype))
        feats = torch.cat([rel_enc, msg_e[order], z_mem[src_sorted]], dim=-1)
        feat_dim = feats.shape[-1]
        buffer = torch.zeros(N * self.n_neighbors, feat_dim, device=device,
                             dtype=feats.dtype)
        flat_idx = dest_sorted * self.n_neighbors + pos
        buffer[flat_idx] = feats
        valid = torch.zeros(N, self.n_neighbors, dtype=torch.bool, device=device)
        valid.view(-1)[flat_idx] = True
        preagg = torch.cat([
            z_mem,
            buffer.reshape(N, self.n_neighbors * feat_dim),
            valid.to(feats.dtype).reshape(N, self.n_neighbors),
        ], dim=-1)
        if preagg.shape[-1] != self.preagg_dim:
            raise ValueError("preagg width mismatch")
        return preagg, valid, src_sorted, dest_sorted, edge_times_sorted

    def _build_trace(self, n_id, cand, preagg, src_sorted, dest_sorted,
                     edge_times_sorted, root_local_ids, root_times):
        """Depth-one computation-tree trace for the requested root nodes."""
        self._trace = RecursiveTrace()
        self._oid_by_local = {}
        self._next_oid = 0
        N = int(n_id.numel())
        root_local_ids = root_local_ids.long()
        root_time_by_local = {}
        if root_times is not None:
            for local, t in zip(root_local_ids.tolist(), root_times.tolist()):
                root_time_by_local[int(local)] = float(t)

        stack: List[tuple] = []
        for local in root_local_ids.tolist():
            stack.append((int(local), None, None))
        while stack:
            local, parent_oid, t_root = stack.pop()
            if local in self._oid_by_local:
                if parent_oid is not None:
                    occ = self._trace.occurrences[parent_oid]
                    if self._oid_by_local[local] not in occ.children:
                        occ.children.append(self._oid_by_local[local])
                        occ.child_relations.append(1)
                        occ.child_delta_t.append(float(t_root))
                continue
            if t_root is None:
                t_root = root_time_by_local.get(local)
            # Gather this node's incoming edges in the subgraph.
            mask = dest_sorted == local
            child_locals = src_sorted[mask].tolist()
            edge_ts = edge_times_sorted[mask].tolist()
            oid = self._next_oid
            self._next_oid += 1
            self._oid_by_local[local] = oid
            cand_v = cand[local]
            self._trace.add(RecursiveOccurrence(
                occurrence_id=oid, tau=TAU,
                state=QuotientState(TAU, raw=cand_v.detach(), candidate=cand_v,
                                    quotient=self.prss.project(TAU, cand_v)),
                local_features=preagg[local].detach(),
                children=[], child_relations=[], child_delta_t=[],
            ))
            if parent_oid is not None:
                parent = self._trace.occurrences[parent_oid]
                parent.children.append(oid)
                parent.child_relations.append(1)
                parent.child_delta_t.append(float(t_root))
            for child_local, e_t in zip(child_locals, edge_ts):
                stack.append((int(child_local), oid, float(t_root) - float(e_t)))
        # Register roots in the caller's order.
        roots, rows = [], []
        for local in root_local_ids.tolist():
            oid = self._oid_by_local.get(int(local))
            if oid is not None:
                roots.append(oid)
                rows.append(int(local))
        self._trace.roots = roots
        self._trace.root_rows = rows
