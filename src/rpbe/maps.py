"""Fixed measurement maps: psi(c, y) = sketch([1; phi_C(c)] (x) phi_Y(y)).

Everything here is frozen: all random matrices are generated once from
``rpbe_seed`` and registered as buffers; ``psi`` never creates gradient.  The
tensor-product structure is mandatory — a plain concatenation would only keep
marginal information and could not represent context-outcome pairings.

Small helper note: the continuous context feature (delta_t) is scaled by a
fixed ``delta_t_scale`` before the RFF pass (no learned or running statistics
here; batch whitening happens later, inside the Ky Fan score).
"""

import hashlib

import torch
from torch import nn


def _fixed_binary(shape, seed, device=None, dtype=None):
    """Deterministic +-1 table from a scalar seed (no RNG state leakage)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randint(0, 2, shape, generator=g,
                         dtype=torch.int64).to(device=device) * 2 - 1


class FixedMaps(nn.Module):
    """phi_C, phi_Y and the sparse CountSketch of their tensor product."""

    def __init__(self, cfg, *, num_counter_bins: int = 4096):
        super().__init__()
        self.cfg = cfg
        self.d_c = int(cfg.d_c)
        self.d_f = int(cfg.d_f)
        self.m = int(cfg.m)
        self.num_counter_bins = int(num_counter_bins)
        seed = int(cfg.rpbe_seed)

        # Rows: [0, bins) counterpart hash; bins+0/1 -> role; bins+2/3 -> query.
        self.register_buffer("categorical_c", _fixed_binary(
            (self.num_counter_bins + 4, self.d_c), seed + 1), persistent=True)
        # Future y in {0, 1}: two fixed d_f-dim signatures.
        self.register_buffer("future_table", _fixed_binary(
            (2, self.d_f), seed + 2), persistent=True)
        # RFF for the scaled delta_t: cos(W x + b).
        self.register_buffer("rff_w", torch.randn(
            (1, self.d_c), generator=torch.Generator().manual_seed(seed + 3)),
            persistent=True)
        self.register_buffer("rff_b", torch.rand(
            (self.d_c,), generator=torch.Generator().manual_seed(seed + 4)) * 6.2832,
            persistent=True)

        # CountSketch: one signed hash per full tensor-product row
        # [1; phi_C] (x) phi_Y has (d_c+1) * d_f rows, k nonzeros per column.
        k = 10
        full_dim = (self.d_c + 1) * self.d_f
        g = torch.Generator().manual_seed(seed + 5)
        self._sketch_nnz = k * self.m
        rows = torch.randint(0, full_dim, (self._sketch_nnz,), generator=g)
        cols = torch.randint(0, self.m, (self._sketch_nnz,), generator=g)
        signs = torch.randint(0, 2, (self._sketch_nnz,), generator=g) * 2 - 1
        sketch_indices = torch.stack([rows, cols], dim=0)  # [2, nnz]
        self.register_buffer("sketch_indices", sketch_indices, persistent=True)
        self.register_buffer("sketch_signs", signs, persistent=True)
        self._sketch_full_dim = full_dim

    # ------------------------------------------------------------- primitives
    def context_vector(self, context) -> torch.Tensor:
        """phi_C(c) for one row: {delta_t, counterpart, role, query_type}."""
        delta = float(context["delta_t"]) / float(self.cfg.delta_t_scale)
        delta_t = torch.tensor(delta, dtype=self.rff_w.dtype,
                               device=self.rff_w.device).reshape(1, 1)
        rff = torch.cos(delta_t @ self.rff_w + self.rff_b)  # [1, d_c]
        partner = int(context["counterpart"]) % self.num_counter_bins
        role = int(context["role"]) % 2
        query = int(context["query_type"]) % 2
        cat = (self.categorical_c[partner]
               + self.categorical_c[self.num_counter_bins + role]
               + self.categorical_c[self.num_counter_bins + 2 + query])
        return rff + cat.to(rff.dtype)  # [1, d_c]

    def future_vector(self, outcome: float) -> torch.Tensor:
        """phi_Y(y) for one row; y in {0, 1}."""
        y = 1 if float(outcome) > 0.5 else 0
        return self.future_table[y].to(self.rff_w.dtype)  # [d_f]

    def psi(self, context, outcome: float) -> torch.Tensor:
        """sketch([1; phi_C] (x) phi_Y) -> [m]; no gradient."""
        with torch.no_grad():
            c = self.context_vector(context)               # [1, d_c]
            f = self.future_vector(outcome)                # [d_f]
            body = torch.cat([torch.ones(1, dtype=c.dtype, device=c.device),
                              c[0]])                        # [1+d_c]
            prod = torch.outer(body, f).reshape(-1)         # [(1+d_c)(1+d_f)]
            out = torch.zeros(self.m, dtype=prod.dtype, device=prod.device)
            out.index_add_(0, self.sketch_indices[1],
                           prod[self.sketch_indices[0]] * self.sketch_signs)
            return out

    def pv(self, context, outcome: float) -> torch.Tensor:
        """One joint-test row for one cut."""
        return self.psi(context, outcome)

    # ------------------------------------------------------------------ audit
    def isolation_fingerprint(self) -> dict:
        """Seed/version + content hash of every frozen buffer."""
        h = hashlib.sha256()
        h.update(str(self.cfg.rpbe_seed).encode())
        h.update(str(self._sketch_full_dim).encode())
        for name in ("categorical_c", "future_table", "rff_w", "rff_b",
                     "sketch_indices", "sketch_signs"):
            buf = getattr(self, name)
            h.update(buf.reshape(-1)[: min(buf.numel(), 4096)]
                     .detach().cpu().to(torch.int64).numpy().tobytes())
        return {"seed": int(self.cfg.rpbe_seed), "sha256": h.hexdigest()}
