"""A0 phase B: per-constructor recursive operators by convex ridge.

Each constructor sigma maps one source child plus a variable-length neighbor
sibling set to the parent interface (the TGN recursion).  Fixed multiaffine
interaction features (theory doc 5.6, mean-pooled siblings in this first
version, no TensorSketch):

    chi_sigma(z_s, z_bar_n, a) = [1; z_s; z_bar_n; z_s ⊙ z_bar_n; a]

The operator is a one-shot multi-output ridge solve on an independent data
window; B̂ is frozen afterwards (never a gradient parameter).
"""

import torch


def chi_sigma(z_source: torch.Tensor, z_neigh_mean: torch.Tensor,
              a_parent: torch.Tensor) -> torch.Tensor:
    """Fixed interaction features for one constructor step.

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
