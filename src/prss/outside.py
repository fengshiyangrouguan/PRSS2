"""Training-only inside/outside continuation encoder with detached sibling teachers."""

from typing import Mapping

import torch
from torch import nn


class OutsideContextEncoder(nn.Module):
    """Encodes the upper continuation of a subtree WITHOUT seeing the subtree itself.

    The current child's candidate is never an input to its context.  Sibling candidates
    are detached (per the method specification) so the current quotient cannot blind the
    training-only teacher.
    """

    def __init__(self, interface_specs, root_metadata_dim, parent_local_dim,
                 context_dim=64, relation_count=4, relation_dim=16,
                 layers=2, detach_siblings=True):
        super().__init__()
        self.interface_names = list(interface_specs)
        self._tau_to_index = {tau: index for index, tau in enumerate(self.interface_names)}
        self.context_dim = context_dim
        self.root_metadata_dim = root_metadata_dim
        self.parent_local_dim = parent_local_dim
        self.detach_siblings = detach_siblings
        self.relation_embedding = nn.Embedding(relation_count, relation_dim)
        self.type_embedding = nn.Embedding(len(self.interface_names), relation_dim)
        self.sibling_projectors = nn.ModuleDict({
            "type_{:04d}".format(index): nn.Linear(spec.candidate_dim, context_dim, bias=False)
            for index, spec in enumerate(interface_specs.values())
        })
        self.root_encoder = self._mlp(root_metadata_dim + relation_dim, context_dim,
                                      context_dim, layers)
        child_input_dim = (context_dim + parent_local_dim + relation_dim + relation_dim +
                           1 + context_dim)
        self.child_encoder = self._mlp(child_input_dim, context_dim, context_dim, layers)

    @staticmethod
    def _mlp(input_dim, hidden_dim, output_dim, layers):
        modules = []
        current = input_dim
        for _ in range(max(layers - 1, 0)):
            modules.extend([nn.Linear(current, hidden_dim), nn.GELU()])
            current = hidden_dim
        modules.extend([nn.Linear(current, output_dim), nn.LayerNorm(output_dim)])
        return nn.Sequential(*modules)

    def _type_index(self, tau, device, batch_shape=()):
        if tau not in self._tau_to_index:
            raise KeyError("Unknown outside-context interface type: {}".format(tau))
        return torch.full(batch_shape, self._tau_to_index[tau], device=device, dtype=torch.long)

    def root_context(self, root_metadata, root_type):
        if root_metadata.shape[-1] != self.root_metadata_dim:
            raise ValueError("Root metadata width mismatch")
        type_ids = self._type_index(root_type, root_metadata.device, root_metadata.shape[:-1])
        type_features = self.type_embedding(type_ids)
        return self.root_encoder(torch.cat([root_metadata, type_features], dim=-1))

    def summarize_siblings(self, candidates_by_type, reference):
        """Mean projected sibling candidate; the current child must be excluded by the caller."""
        projected_sums = []
        counts = []
        for tau, candidates in candidates_by_type.items():
            if candidates.numel() == 0:
                continue
            values = candidates.detach() if self.detach_siblings else candidates
            key = "type_{:04d}".format(self._tau_to_index[tau])
            projected = self.sibling_projectors[key](values)
            if projected.ndim == reference.ndim:
                projected = projected.unsqueeze(-2)
            projected_sums.append(projected.sum(dim=-2))
            counts.append(projected.shape[-2])
        if not projected_sums:
            return torch.zeros(*reference.shape[:-1], self.context_dim,
                               device=reference.device, dtype=reference.dtype)
        return sum(projected_sums) / float(sum(counts))

    def child_context(self, parent_outside, parent_local, relation_ids, delta_t,
                      sibling_summary, child_type):
        if parent_outside.shape[-1] != self.context_dim:
            raise ValueError("Parent outside width mismatch")
        if parent_local.shape[-1] != self.parent_local_dim:
            raise ValueError("Parent local width mismatch")
        if sibling_summary.shape[-1] != self.context_dim:
            raise ValueError("Sibling summary width mismatch")
        batch_shape = parent_outside.shape[:-1]
        relation_ids = torch.as_tensor(relation_ids, device=parent_outside.device, dtype=torch.long)
        relation_ids = relation_ids.expand(batch_shape)
        relation = self.relation_embedding(relation_ids)
        type_ids = self._type_index(child_type, parent_outside.device, batch_shape)
        type_features = self.type_embedding(type_ids)
        delta_t = torch.as_tensor(delta_t, device=parent_outside.device,
                                  dtype=parent_outside.dtype).expand(batch_shape)
        encoded_time = torch.log1p(delta_t.clamp_min(0)).unsqueeze(-1)
        values = torch.cat([parent_outside, parent_local, relation, type_features,
                            encoded_time, sibling_summary], dim=-1)
        return self.child_encoder(values)
