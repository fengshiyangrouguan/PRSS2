"""A0 fixed probes: context features a(C), outcome one-hots φ_Y(Y), root-label
propagation, and per-tau stacking of the conditional-moment rows.

All probes are FIXED features (theory doc 5.2): no trainable encoder.  The
context probe is a seed-fixed random projection of the occurrence's parent-side
``local_features`` block (child-state block already zeroed by C1); the outcome
probe is the one-hot of the label; the history lift x(H) is the occurrence's
candidate (with a vanilla PRSS core this equals the host embedding bitwise).
"""

from typing import Dict, List, Sequence

import torch


def onehot_y(y):
    """Binary one-hot of a label: [1-y, y] (phi_Y in the theory doc)."""
    y = float(y)
    return torch.tensor([1.0 - y, y])


class A0Probes:
    """Fixed random context projection P_c and the feature extractors around it."""

    def __init__(self, *, preagg_dim: int, d_context: int, seed: int = 0,
                 device=None):
        if d_context <= 0:
            raise ValueError("d_context must be positive")
        self.preagg_dim = int(preagg_dim)
        self.d_context = int(d_context)
        generator = torch.Generator()
        generator.manual_seed(int(seed) + 7717)
        p_c = torch.randn(self.d_context, self.preagg_dim,
                          device=device, dtype=torch.float32,
                          generator=generator)
        # Row-normalized so every context row lives on a comparable scale.
        self.p_c = p_c / (p_c.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        self.device = device

    def probe_a(self, local_features: torch.Tensor) -> torch.Tensor:
        """a(C) = P_c @ local_features in R^{d_context} (fixed random probe)."""
        if local_features.shape[-1] != self.preagg_dim:
            raise ValueError("local_features width mismatch: {} vs {}".format(
                local_features.shape[-1], self.preagg_dim))
        return local_features.to(self.p_c.dtype) @ self.p_c.transpose(0, 1)

    def to(self, device):
        self.p_c = self.p_c.to(device)
        self.device = device
        return self


def propagate_root_labels(trace, root_labels: Sequence) -> Dict[int, float]:
    """Every cut of a tree shares its root label (theory doc 4.4).

    ``root_labels`` aligns with ``trace.root_rows``.  Returns {oid: y} for
    every occurrence in the trace.
    """
    out: Dict[int, float] = {}
    for root_oid, y in zip(trace.roots, root_labels):
        y = float(y)
        out[root_oid] = y
        stack = [root_oid]
        while stack:
            oid = stack.pop()
            for child in trace.occurrences[oid].children:
                if child not in out:
                    out[child] = y
                    stack.append(child)
    return out


def stack_by_tau(trace, oid_labels: Dict[int, float], probes: A0Probes,
                 device=None) -> Dict[str, Dict[str, torch.Tensor]]:
    """Per-tau stacks of the conditional-moment rows over one trace.

    Returns {tau: {"X": (n_tau, p), "A": (n_tau, d_c), "U": (n_tau, 2 d_c),
    "Y": (n_tau,)}} where X is the history lift (candidate), A the context
    probe, U = vec(A ⊗ φ_Y(Y)), Y the propagated root label.
    """
    by_tau: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    for occ in trace.occurrences.values():
        tau = occ.tau
        x = occ.state.candidate.detach()
        a = probes.probe_a(occ.local_features.detach())
        y = oid_labels[occ.occurrence_id]
        phi = onehot_y(y).to(device=a.device, dtype=a.dtype)
        # vec(A ⊗ φ): for binary phi this is concat(A * (1-y), A * y).
        u = torch.cat([a * phi[0], a * phi[1]], dim=-1)
        bucket = by_tau.setdefault(tau, {"X": [], "A": [], "U": [], "Y": []})
        bucket["X"].append(x)
        bucket["A"].append(a)
        bucket["U"].append(u)
        bucket["Y"].append(torch.as_tensor(y, device=a.device, dtype=a.dtype))
    out = {}
    for tau, bucket in by_tau.items():
        out[tau] = {
            "X": torch.stack(bucket["X"], dim=0),
            "A": torch.stack(bucket["A"], dim=0),
            "U": torch.stack(bucket["U"], dim=0),
            "Y": torch.stack(bucket["Y"], dim=0),
        }
    return out
