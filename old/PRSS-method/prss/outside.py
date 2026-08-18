from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import nn


@dataclass
class Occurrence:
    oid: int
    layer: int
    candidate: torch.Tensor
    local: torch.Tensor
    children: List[int] = field(default_factory=list)
    relations: List[int] = field(default_factory=list)
    deltas: List[float] = field(default_factory=list)


@dataclass
class Trace:
    occurrences: Dict[int, Occurrence] = field(default_factory=dict)
    roots: List[int] = field(default_factory=list)
    root_rows: List[int] = field(default_factory=list)


class OutsideContextEncoder(nn.Module):
    """Training-only inside/outside continuation encoder.

    The current child's candidate is never an input to its context. Parent-local metadata is the
    exact edge/time/mask tensor used by the parent constructor, passed through a small training-only
    encoder. Sibling candidates are detached, as permitted by the method specification.
    """
    def __init__(self, candidate_dims: Dict[int, int], parent_local_dim: int,
                 context_dim: int = 64, relation_dim: int = 16):
        super().__init__()
        self.layers = sorted(candidate_dims)
        self.layer_to_idx = {l: i for i, l in enumerate(self.layers)}
        self.context_dim = int(context_dim)
        self.parent_local_dim = int(parent_local_dim)
        self.relation = nn.Embedding(2, relation_dim)  # self-lower vs temporal neighbor
        self.layer_emb = nn.Embedding(len(self.layers), relation_dim)
        self.sibling_proj = nn.ModuleDict({
            str(l): nn.Linear(candidate_dims[l], context_dim, bias=False) for l in self.layers
        })
        self.parent_local_proj = nn.Sequential(
            nn.Linear(self.parent_local_dim, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )
        # Root continuation for node classification is the common classifier; time is legal query metadata.
        self.root = nn.Sequential(
            nn.Linear(1 + relation_dim, context_dim), nn.GELU(), nn.LayerNorm(context_dim)
        )
        child_in = context_dim + context_dim + relation_dim + relation_dim + 1 + context_dim
        self.child = nn.Sequential(
            nn.Linear(child_in, context_dim), nn.GELU(), nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim), nn.GELU(), nn.LayerNorm(context_dim)
        )

    def root_context(self, normalized_log_time: torch.Tensor, layer: int):
        idx = torch.full(normalized_log_time.shape, self.layer_to_idx[layer],
                         device=normalized_log_time.device, dtype=torch.long)
        le = self.layer_emb(idx)
        return self.root(torch.cat([normalized_log_time.unsqueeze(-1), le], dim=-1))

    def sibling_summary(self, trace: Trace, parent: Occurrence, excluded_child: int, reference):
        projected = []
        for sid in parent.children:
            if sid == excluded_child:
                continue
            s = trace.occurrences[sid]
            projected.append(self.sibling_proj[str(s.layer)](s.candidate.detach()))
        if not projected:
            return torch.zeros(self.context_dim, device=reference.device, dtype=reference.dtype)
        return torch.stack(projected, dim=0).mean(dim=0)

    def child_context(self, parent_context, parent_local, relation_id: int, delta_t: float,
                      sibling_summary, child_layer: int):
        device, dtype = parent_context.device, parent_context.dtype
        rel = self.relation(torch.tensor(relation_id, device=device, dtype=torch.long))
        lay = self.layer_emb(torch.tensor(self.layer_to_idx[child_layer], device=device, dtype=torch.long))
        dt = torch.log1p(torch.tensor(max(float(delta_t), 0.0), device=device, dtype=dtype)).view(1)
        local = self.parent_local_proj(parent_local.unsqueeze(0)).squeeze(0)
        x = torch.cat([parent_context, local, rel, lay, dt, sibling_summary], dim=-1)
        return self.child(x)
