"""A0 phase B: per-constructor recursive operators by convex ridge.

Each constructor sigma maps one source child plus a variable-length neighbor
sibling set to the parent interface (the TGN recursion).  Two fixed
interaction feature classes (theory doc 5.6):

- ``meanpool``: chi = [1; z_s; z_bar_n; z_s ⊙ z_bar_n; a] (exact multiaffine
  on the mean-pooled sibling summary);
- ``sketch``: TensorSketch features over the augmented tensor product
  b(o) ⊗ [1; z_s] ⊗ ⊗_i [1; z_i] — per-factor CountSketch maps, sibling
  symmetry via power sums, width 3s (Pham-Pagh style, no explicit
  Kronecker).

The operator is a one-shot multi-output ridge solve on an independent data
window; B̂ is frozen afterwards (never a gradient parameter).
"""

import torch


def chi_sigma(z_source: torch.Tensor, z_neigh_mean: torch.Tensor,
              a_parent: torch.Tensor) -> torch.Tensor:
    """Fixed mean-pool interaction features for one constructor step.

    ``z_source`` (r,), ``z_neigh_mean`` (r,) (zero vector when there are no
    neighbor children), ``a_parent`` (d_c,).  Returns (s,) with s = 1+3r+d_c.
    """
    return torch.cat([
        torch.ones(1, device=z_source.device, dtype=z_source.dtype),
        z_source,
        z_neigh_mean,
        z_source * z_neigh_mean,
        a_parent.to(dtype=z_source.dtype, device=z_source.device),
    ], dim=-1)


def chi_width(r: int, d_context: int) -> int:
    return 1 + 3 * r + d_context


class TensorSketchFeatures:
    """Fixed CountSketch maps for the augmented tensor-product features.

    Each factor ([1; z_s], [1; z_i], a) gets its own seed-fixed CountSketch
    (s buckets, random signs); the sketch of the tensor product is the
    elementwise product of the factor sketches.  Unordered neighbors are
    symmetrized by power sums (first and second order), so the width is 3s
    regardless of the neighbor count (doc 5.6 eq. 9).
    """

    def __init__(self, r: int, d_context: int, s: int = 64, seed: int = 0):
        self.r = int(r)
        self.d_context = int(d_context)
        self.s = int(s)
        gen = torch.Generator().manual_seed(int(seed) + 9173)
        self.h_src = torch.randint(0, self.s, (self.r + 1,), generator=gen)
        self.sg_src = (torch.randint(0, 2, (self.r + 1,), generator=gen)
                       * 2 - 1).float()
        self.h_nb = torch.randint(0, self.s, (self.r + 1,), generator=gen)
        self.sg_nb = (torch.randint(0, 2, (self.r + 1,), generator=gen)
                      * 2 - 1).float()
        self.h_obs = torch.randint(0, self.s, (self.d_context,), generator=gen)
        self.sg_obs = (torch.randint(0, 2, (self.d_context,), generator=gen)
                       * 2 - 1).float()

    def width(self) -> int:
        return 3 * self.s

    def _sketch(self, x, h, sg):
        """x (..., d) -> (..., s) CountSketch (bucket sums with signs)."""
        flat = x.reshape(-1, x.shape[-1]).double()
        out = torch.zeros(flat.shape[0], self.s, dtype=torch.float64,
                          device=x.device)
        idx = h.to(device=x.device).expand(flat.shape[0], -1)
        vals = flat * sg.to(dtype=x.dtype, device=x.device)
        out.scatter_add_(1, idx, vals)
        return out.reshape(*x.shape[:-1], self.s)

    def chi(self, z_source: torch.Tensor, z_neighbors: torch.Tensor,
            a_parent: torch.Tensor) -> torch.Tensor:
        """Sketch chi for one constructor step.

        ``z_source`` (r,), ``z_neighbors`` (k, r) (may be empty), ``a_parent``
        (d_c,).  Returns (3s,).
        """
        one = torch.ones(1, device=z_source.device, dtype=z_source.dtype)
        s_src = self._sketch(torch.cat([one, z_source.double()]),
                             self.h_src, self.sg_src)
        s_obs = self._sketch(a_parent.double().unsqueeze(0),
                             self.h_obs, self.sg_obs).squeeze(0)
        nb_sum = torch.zeros(self.s, dtype=torch.float64, device=z_source.device)
        nb_sq = torch.zeros_like(nb_sum)
        if z_neighbors.shape[0] > 0:
            aug = torch.cat([torch.ones(z_neighbors.shape[0], 1,
                                        device=z_source.device,
                                        dtype=torch.float64),
                             z_neighbors.double()], dim=-1)
            sk = self._sketch(aug, self.h_nb, self.sg_nb)
            nb_sum = sk.sum(dim=0)
            nb_sq = (sk * sk).sum(dim=0)
        return torch.cat([s_src * s_obs, nb_sum * s_obs, nb_sq * s_obs],
                         dim=-1)


class OperatorRidge:
    """Streaming ΦᵀΦ / ΦᵀZ plus one-shot frozen ridge B̂ for one constructor."""

    def __init__(self, child_tau: str, parent_tau: str, s: int, r: int):
        self.child_tau = child_tau
        self.parent_tau = parent_tau
        self.s = int(s)
        self.r = int(r)
        self.n = 0
        self.phi_phi = torch.zeros(self.s, self.s, dtype=torch.float64)
        self.phi_z = torch.zeros(self.s, self.r, dtype=torch.float64)
        self.b_matrix = None        # (r, s) frozen
        self.condition_number = None
        self.effective_rank = None
        self._chol_l = None         # ridge design Cholesky (leverage scores)

    @torch.no_grad()
    def accumulate(self, phi: torch.Tensor, z_rich: torch.Tensor) -> None:
        """phi (n, s) interaction rows, z_rich (n, r) supervision targets."""
        if self.b_matrix is not None:
            raise RuntimeError("operator {}->{} already frozen".format(
                self.child_tau, self.parent_tau))
        phi = phi.detach().to(dtype=torch.float64)
        z_rich = z_rich.detach().to(dtype=torch.float64)
        if phi.shape[-1] != self.s or z_rich.shape[-1] != self.r:
            raise ValueError("width mismatch for {}->{}: phi {}, z {}".format(
                self.child_tau, self.parent_tau, phi.shape, z_rich.shape))
        if self.phi_phi.device != phi.device:
            self.phi_phi = self.phi_phi.to(phi.device)
            self.phi_z = self.phi_z.to(phi.device)
        self.phi_phi.add_(phi.transpose(0, 1) @ phi)
        self.phi_z.add_(phi.transpose(0, 1) @ z_rich)
        self.n += int(phi.shape[0])

    @torch.no_grad()
    def solve(self, lambda_gamma: float = 1e-3) -> None:
        """One-shot convex ridge: B̂ = (ΦᵀΦ + λI)^{-1} ΦᵀZ, frozen."""
        if self.n == 0:
            raise RuntimeError("operator {}->{} has no rows".format(
                self.child_tau, self.parent_tau))
        eye = torch.eye(self.s, dtype=torch.float64, device=self.phi_phi.device)
        a = self.phi_phi + lambda_gamma * eye
        chol = torch.linalg.cholesky(a)
        self.b_matrix = torch.cholesky_solve(self.phi_z, chol).transpose(0, 1)
        self._chol_l = chol  # kept for leverage/OOD scores
        design = self.phi_phi / max(self.n, 1)
        eig = torch.linalg.eigvalsh(design).clamp_min(0.0)
        if float(eig[-1].item()) > 0:
            self.condition_number = float((eig[-1] / (eig[0] + 1e-12)).item())
            self.effective_rank = int((eig > 1e-8 * eig[-1]).sum().item())
        else:
            self.condition_number = float("inf")
            self.effective_rank = 0

    @torch.no_grad()
    def predict(self, phi: torch.Tensor) -> torch.Tensor:
        """z_rec = B̂ chi for any leading batch dims."""
        if self.b_matrix is None:
            raise RuntimeError("operator {}->{} not solved".format(
                self.child_tau, self.parent_tau))
        b_t = self.b_matrix.transpose(0, 1).to(device=phi.device)
        return phi.double() @ b_t

    @torch.no_grad()
    def leverage(self, phi: torch.Tensor) -> torch.Tensor:
        """Hat-diagonal scores h_i = ‖L⁻¹ φ_i‖² under the ridge design.

        For in-support rows h_i ∈ [0, 1] with mean s/n; rows with h_i ≫ 1
        live outside the training support (doc 5.7.6: train-to-deploy OOD
        score — collinear brothers concentrate leverage and OOD rows stick
        out).
        """
        if self._chol_l is None:
            raise RuntimeError("operator {}->{} not solved".format(
                self.child_tau, self.parent_tau))
        phi = phi.detach().to(dtype=torch.float64)
        l = self._chol_l.to(device=phi.device)
        solved = torch.linalg.solve_triangular(
            l, phi.transpose(0, 1), upper=False).transpose(0, 1)
        return (solved * solved).sum(dim=-1)

    def gain(self) -> float:
        """Estimated Lipschitz gain on the source-child block of chi.

        The rows of B̂ multiply the stacked [1; z_s; z_bar; z_s⊙z_bar; a];
        the pure z_s block (columns 1:1+r) is the source-child Jacobian, its
        spectral norm bounds the child-error amplification (theory doc 7).
        """
        if self.b_matrix is None:
            return float("nan")
        block = self.b_matrix[:, 1:1 + self.r]
        return float(torch.linalg.norm(block, ord=2).item())

    def snapshot(self):
        return {
            "child_tau": self.child_tau,
            "parent_tau": self.parent_tau,
            "n": self.n,
            "s": self.s,
            "r": self.r,
            "condition_number": self.condition_number,
            "effective_rank": self.effective_rank,
            "gain": self.gain(),
            "frozen": self.b_matrix is not None,
        }
