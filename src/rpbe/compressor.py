"""Recursive compressor Gamma_theta.

For the TGN degeneracy (homogeneous neighbors, one recursive operator) the
host aggregate stays exactly where it is: the adapter computes the vanilla
aggregate with the host's own ``aggregate`` and hands it to ``compress`` as
the child-aggregation token.  Gamma adds the thin stack on top::

    u_v = A_tau(o_v)                       # input adapter   -> width D
    a_v = A_agg_tau(aggregate_output)      # aggregate adapter -> width D
    h_v = G(u_v, a_v)                      # shared nonlinear core
    z_v = Q_tau(h_v)                       # output head     -> budget r_tau

``compress`` is the deployment path: it reads only the node's own input and
the aggregate of the children's compressed states — never any future.
"""

import torch
from torch import nn


class RecursiveCompressor(nn.Module):
    def __init__(self, cfg, *, agg_dims=None, activation=nn.GELU):
        super().__init__()
        self.cfg = cfg
        D = int(cfg.width_D)
        agg_dims = dict(agg_dims) if agg_dims is not None else dict(cfg.own_dims)
        self.adapters = nn.ModuleDict({
            tau: nn.Linear(int(cfg.own_dims[tau]), D)
            for tau in cfg.interfaces})
        self.agg_adapters = nn.ModuleDict({
            tau: nn.Linear(int(agg_dims[tau]), D)
            for tau in cfg.interfaces})
        self.heads = nn.ModuleDict({
            tau: nn.Linear(D, int(r_tau))
            for tau, r_tau in cfg.interfaces.items()})
        # Shared core: [A(o); A_agg(aggregate)] -> D, with residual.
        self.core = nn.Sequential(
            nn.Linear(2 * D, D), activation(), nn.Linear(D, D))

    def compress(self, *, tau: str, own_input: torch.Tensor,
                 aggregate_output: torch.Tensor) -> torch.Tensor:
        """One interface call: [N, d_o] x [N, d_agg] -> [N, r_tau].

        ``aggregate_output`` is the host aggregate result (layer 0 has no
        children: the host passes its raw state in both slots).
        """
        u = self.adapters[tau](own_input)
        a = self.agg_adapters[tau](aggregate_output.to(u.dtype))
        h = self.core(torch.cat([u, a], dim=-1)) + u
        return self.heads[tau](h)

    def interface_dims(self) -> dict:
        return dict(self.cfg.interfaces)
