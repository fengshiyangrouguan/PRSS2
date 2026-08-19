import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
from torch import nn


def _symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def row_orthonormalize(rows: torch.Tensor) -> torch.Tensor:
    # rows: k x d. QR on transpose gives orthonormal columns => orthonormal rows after transpose.
    q, _ = torch.linalg.qr(rows.transpose(0, 1), mode="reduced")
    return q.transpose(0, 1)


def projector(rows: torch.Tensor) -> torch.Tensor:
    return rows.transpose(0, 1) @ rows


def procrustes_align_rows(new_rows: torch.Tensor, old_rows: torch.Tensor) -> torch.Tensor:
    """Align row basis of the *same subspace* to the previous coordinates.

    Solve min_Q ||Q new_rows - old_rows||_F, Q orthogonal.
    """
    m = old_rows @ new_rows.transpose(0, 1)
    u, _, vh = torch.linalg.svd(m, full_matrices=False)
    q = u @ vh
    return q @ new_rows


def principal_angles(old_rows: torch.Tensor, new_rows: torch.Tensor) -> torch.Tensor:
    s = torch.linalg.svdvals(old_rows @ new_rows.transpose(0, 1)).clamp(0.0, 1.0)
    return torch.arccos(s)


def _complete_with_old_nullspace(pred_rows: torch.Tensor, old_rows: torch.Tensor, k: int) -> torch.Tensor:
    """Complete predictive rows to k rows without arbitrary rotations in Gram nullspace.

    If Gram rank r < k, every k-dimensional subspace that contains the r predictive rows is
    spectrally optimal. We choose remaining directions from the old quotient projected onto the
    orthogonal complement. This is an exact optimizer, not an EMA/interpolation of R.
    """
    device, dtype = old_rows.device, old_rows.dtype
    if pred_rows.numel() == 0:
        return old_rows.clone()
    pred_rows = row_orthonormalize(pred_rows)
    need = k - pred_rows.shape[0]
    if need <= 0:
        return pred_rows[:k]
    p = projector(pred_rows)
    residual = old_rows @ (torch.eye(old_rows.shape[1], device=device, dtype=dtype) - p)
    # SVD gives stable orthonormal row basis spanning the old quotient's residual component.
    _, s, vh = torch.linalg.svd(residual, full_matrices=False)
    keep = int(min(need, (s > 1e-8).sum().item()))
    rows = [pred_rows]
    if keep > 0:
        rows.append(vh[:keep])
        need -= keep
    if need > 0:
        # Deterministic fallback: standard basis projected onto complement of what we already have.
        current = torch.cat(rows, dim=0)
        pcur = projector(row_orthonormalize(current))
        eye = torch.eye(old_rows.shape[1], device=device, dtype=dtype)
        cand = eye @ (eye - pcur)
        _, s2, vh2 = torch.linalg.svd(cand, full_matrices=False)
        rows.append(vh2[:need])
    return row_orthonormalize(torch.cat(rows, dim=0)[:k])


@dataclass
class SpectralSnapshot:
    interface: str
    host_dim: int
    candidate_dim: int
    dimensional_compression: bool
    spectral_updates: int
    reader_gram_updates: int
    gram_updates_since_spectral: int
    effective_predictive_rank: int
    projector_distance: float
    max_principal_angle: float
    mean_principal_angle: float
    captured_energy_before: float
    captured_energy_after: float
    captured_energy_gain: float
    accepted_spectral_step: float
    energy_at_quarter: float
    energy_at_half: float
    energy_at_k: float
    energy_at_full: float
    tail_at_k: float
    b_fro_mean: float
    last_update_step: int
    gram_trace: float
    gram_symmetry_relative: float
    row_orthogonality_relative: float
    live_current_R_energy: float
    eigenvalues_descending: list


class SpectralQuotient(nn.Module):
    """Per-interface predictive Gram + exact rank-k spectral quotient.

    The matrix R is a buffer, not a trainable parameter. Updates are obtained from the right
    singular subspace of the conceptual stacked future-reader operator bank B(C_1); B(C_2); ...
    via eigendecomposition of mean B^T B.
    """
    def __init__(self, name: str, host_dim: int, candidate_dim: int, gram_ema: float = 0.05,
                 eps: float = 1e-8, rank_rtol: float = 1e-5, spectral_step_size: float = 1.0):
        super().__init__()
        if candidate_dim < host_dim:
            raise ValueError("candidate_dim must be >= host_dim")
        self.name = name
        self.host_dim = int(host_dim)
        self.candidate_dim = int(candidate_dim)
        self.gram_ema = float(gram_ema)
        self.eps = float(eps)
        self.rank_rtol = float(rank_rtol)
        if not 0 < spectral_step_size <= 1:
            raise ValueError("spectral_step_size must be in (0,1]")
        self.spectral_step_size = float(spectral_step_size)
        r = torch.zeros(self.host_dim, self.candidate_dim)
        r[:, :self.host_dim] = torch.eye(self.host_dim)
        self.register_buffer("R", r)
        self.register_buffer("G", torch.zeros(self.candidate_dim, self.candidate_dim))
        self.register_buffer("spectral_updates_t", torch.zeros((), dtype=torch.long))
        self.register_buffer("reader_gram_updates_t", torch.zeros((), dtype=torch.long))
        self.register_buffer("gram_updates_since_t", torch.zeros((), dtype=torch.long))
        self.register_buffer("b_fro_sum", torch.zeros(()))
        self.register_buffer("b_fro_count", torch.zeros(()))
        self.register_buffer("last_update_step_t", torch.full((), -1, dtype=torch.long))
        self._last_eigvals = torch.zeros(self.candidate_dim)
        self._last_projector_distance = 0.0
        self._last_angles = torch.zeros(self.host_dim)
        self._last_energy_before = 0.0
        self._last_energy_after = 0.0
        self._last_effective_rank = 0
        self._last_accepted_step = 0.0

    @property
    def dimensional_compression(self):
        return self.candidate_dim > self.host_dim

    def project(self, candidate: torch.Tensor) -> torch.Tensor:
        return candidate @ self.R.transpose(0, 1)

    def _has_deployed_data_driven_quotient(self) -> bool:
        # A spectral solve attempt is not enough: the damped deployment may reject the move.
        # Only turn on L_spec after R has actually left the identity-compatible initialization.
        # This is state-dict stable and lets v4 rolling checkpoints resume without a schema change.
        init = torch.zeros_like(self.R)
        init[:, :self.host_dim] = torch.eye(self.host_dim, device=self.R.device, dtype=self.R.dtype)
        return bool((self.R.detach() - init).abs().max().item() > 1e-7)

    def spectral_loss(self, B: torch.Tensor) -> torch.Tensor:
        # B [..., p, d]. Do not regularize against R=[I,0] merely because an SVD was *solved*.
        # If a trust-region deployment was rejected, the live quotient is still the arbitrary
        # initialization and is not a valid consistency target.
        if not self._has_deployed_data_driven_quotient():
            return B.sum() * 0.0

        p = projector(self.R).detach()
        eye = torch.eye(self.candidate_dim, device=B.device, dtype=B.dtype)
        residual = B @ (eye - p)
        num = residual.square().sum(dim=(-1, -2))

        # Same relative-tail objective in the forward pass, but stop the denominator gradient.
        # d(||BQ||^2 / ||B||^2)/dB scales like 1/||B|| near B=0; on Wikipedia's extremely
        # sparse node labels the reader can temporarily drive ||B|| close to zero, producing an
        # avoidable gradient singularity.  A detached floor changes only the optimization
        # conditioning around that degenerate zero-operator point; it does not enter the Gram or
        # the SVD/eigh solution.
        den = B.detach().square().sum(dim=(-1, -2)).clamp_min(1e-4)
        return (num / den).mean()

    @torch.no_grad()
    def accumulate(self, B: torch.Tensor):
        # Average all leading occurrences and response rows into one batch Gram.
        flat = B.reshape(-1, self.candidate_dim)
        if flat.numel() == 0:
            return
        gram = (flat.transpose(0, 1) @ flat) / float(flat.shape[0])
        gram = _symmetrize(gram)
        if int(self.reader_gram_updates_t.item()) == 0:
            self.G.copy_(gram)
        else:
            self.G.mul_(1.0 - self.gram_ema).add_(gram, alpha=self.gram_ema)
        self.reader_gram_updates_t.add_(1)
        self.gram_updates_since_t.add_(1)
        self.b_fro_sum.add_(B.detach().norm(dim=(-1, -2)).mean())
        self.b_fro_count.add_(1)

    @torch.no_grad()
    def update(self, global_step: int) -> bool:
        if not self.dimensional_compression:
            return False
        if int(self.gram_updates_since_t.item()) <= 0:
            return False
        g = _symmetrize(self.G.double())
        g = g + self.eps * torch.eye(self.candidate_dim, device=g.device, dtype=g.dtype)
        eigvals, eigvecs = torch.linalg.eigh(g)
        eigvals = eigvals.clamp_min(0.0)
        order = torch.argsort(eigvals, descending=True)
        vals = eigvals[order]
        vecs = eigvecs[:, order]
        maxv = float(vals[0].item()) if vals.numel() else 0.0
        effective_rank = int((vals > max(self.eps, self.rank_rtol * maxv)).sum().item()) if maxv > 0 else 0
        r_pred = min(effective_rank, self.host_dim)
        pred_rows = vecs[:, :r_pred].transpose(0, 1).to(self.R.dtype)
        old = self.R.clone()
        if r_pred < self.host_dim:
            new = _complete_with_old_nullspace(pred_rows, old, self.host_dim)
        else:
            new = row_orthonormalize(pred_rows[:self.host_dim])
        # The eigenspace above is the exact SVD target.  For spectral_step_size==1 we deploy
        # that analytic optimum exactly (used by the equivalence tests).  For the recursive
        # online runtime we use a bounded Grassmann-ascent step on the *same* objective
        # tr(R G R^T), with backtracking.  Unlike Euclidean basis interpolation, this has a
        # genuine local ascent direction and therefore does not get stuck reporting step=0
        # simply because two equivalent row bases are badly aligned.
        target = procrustes_align_rows(new, old)
        target = row_orthonormalize(target)
        old_p = projector(old.double())
        denom = float(vals.sum().item()) + self.eps
        before = float(torch.trace(old_p @ g).item() / denom)
        accepted = 0.0
        proposal = old
        if self.spectral_step_size >= 1.0 - 1e-12:
            cand = target
            cand_p = projector(cand.double())
            score = float(torch.trace(cand_p @ g).item() / denom)
            if score + 1e-12 >= before:
                proposal = cand
                accepted = 1.0
        else:
            eye_d = torch.eye(self.candidate_dim, device=g.device, dtype=g.dtype)
            old64 = old.double()
            # Grassmann gradient for f(R)=tr(R G R^T): remove the component already inside row(R).
            direction = old64 @ g @ (eye_d - old_p)
            dnorm = float(torch.linalg.norm(direction, ord='fro').item())
            if math.isfinite(dnorm) and dnorm > 1e-14:
                direction = direction / dnorm
                alpha = self.spectral_step_size
                while alpha >= 1e-4:
                    cand = row_orthonormalize((old64 + alpha * direction).to(old.dtype))
                    cand_p = projector(cand.double())
                    score = float(torch.trace(cand_p @ g).item() / denom)
                    if score > before + 1e-12:
                        proposal = cand
                        accepted = float(alpha)
                        break
                    alpha *= 0.5
            # A non-top invariant eigenspace has zero Grassmann gradient.  In that saddle case,
            # fall back to a bounded move toward the exact SVD target and still require measured
            # predictive-energy improvement.
            if accepted == 0.0:
                alpha = self.spectral_step_size
                while alpha >= 1e-4:
                    cand = row_orthonormalize(((1.0 - alpha) * old + alpha * target).to(old.dtype))
                    cand_p = projector(cand.double())
                    score = float(torch.trace(cand_p @ g).item() / denom)
                    if score > before + 1e-12:
                        proposal = cand
                        accepted = float(alpha)
                        break
                    alpha *= 0.5
        new = proposal
        new_p = projector(new.double())
        after = float(torch.trace(new_p @ g).item() / denom)
        angles = principal_angles(old.double(), new.double()).float().cpu()
        self.R.copy_(new)
        self.spectral_updates_t.add_(1)
        self.gram_updates_since_t.zero_()
        self.last_update_step_t.fill_(int(global_step))
        self._last_eigvals = vals.float().cpu()
        self._last_projector_distance = float(torch.linalg.norm(new_p - old_p, ord="fro").item())
        self._last_angles = angles
        self._last_energy_before = before
        self._last_energy_after = after
        self._last_effective_rank = effective_rank
        self._last_accepted_step = accepted
        return True

    def _energy(self, q: int) -> float:
        vals = self._last_eigvals
        if vals.numel() == 0 or float(vals.sum()) <= 0:
            return 0.0
        q = max(0, min(int(q), vals.numel()))
        return float(vals[:q].sum().item() / vals.sum().item())

    def snapshot(self) -> Dict:
        k = self.host_dim
        d = self.candidate_dim
        quarter = max(1, k // 4)
        half = max(1, k // 2)
        bmean = float((self.b_fro_sum / self.b_fro_count.clamp_min(1)).item())
        with torch.no_grad():
            gd = self.G.detach().double()
            gram_trace = float(torch.trace(gd).item())
            gram_sym = float(torch.linalg.norm(gd - gd.T, ord="fro").item() /
                             max(torch.linalg.norm(gd, ord="fro").item(), self.eps))
            rr = self.R.detach().double() @ self.R.detach().double().T
            eye_k = torch.eye(k, device=rr.device, dtype=rr.dtype)
            orth = float(torch.linalg.norm(rr - eye_k, ord="fro").item() / max(math.sqrt(k), 1.0))
            if gram_trace > self.eps:
                live_energy = float(torch.trace(projector(self.R.detach().double()) @ gd).item() /
                                    (gram_trace + self.eps))
                evals = torch.linalg.eigvalsh(_symmetrize(gd)).clamp_min(0.0)
                live_vals = torch.sort(evals, descending=True).values.float().cpu()
                maxv = float(live_vals[0].item()) if live_vals.numel() else 0.0
                live_rank = int((live_vals > max(self.eps, self.rank_rtol * maxv)).sum().item()) if maxv > 0 else 0
            else:
                live_energy = 0.0
                live_vals = torch.zeros(d)
                live_rank = 0

        def energy(vals, q):
            if vals.numel() == 0 or float(vals.sum()) <= 0:
                return 0.0
            q = max(0, min(int(q), vals.numel()))
            return float(vals[:q].sum().item() / vals.sum().item())

        if not self.dimensional_compression and int(self.spectral_updates_t.item()) == 0:
            self._last_energy_before = 1.0
            self._last_energy_after = 1.0

        snap = SpectralSnapshot(
            interface=self.name,
            host_dim=k,
            candidate_dim=d,
            dimensional_compression=bool(self.dimensional_compression),
            spectral_updates=int(self.spectral_updates_t.item()),
            reader_gram_updates=int(self.reader_gram_updates_t.item()),
            gram_updates_since_spectral=int(self.gram_updates_since_t.item()),
            effective_predictive_rank=live_rank,
            projector_distance=float(self._last_projector_distance),
            max_principal_angle=float(self._last_angles.max().item()) if self._last_angles.numel() else 0.0,
            mean_principal_angle=float(self._last_angles.mean().item()) if self._last_angles.numel() else 0.0,
            captured_energy_before=float(self._last_energy_before),
            captured_energy_after=float(self._last_energy_after),
            captured_energy_gain=float(self._last_energy_after - self._last_energy_before),
            accepted_spectral_step=float(self._last_accepted_step),
            energy_at_quarter=(1.0 if not self.dimensional_compression else energy(live_vals, quarter)),
            energy_at_half=(1.0 if not self.dimensional_compression else energy(live_vals, half)),
            energy_at_k=(1.0 if not self.dimensional_compression else energy(live_vals, k)),
            energy_at_full=(1.0 if not self.dimensional_compression else energy(live_vals, d)),
            tail_at_k=(0.0 if not self.dimensional_compression else 1.0 - energy(live_vals, k)),
            b_fro_mean=bmean,
            last_update_step=int(self.last_update_step_t.item()),
            gram_trace=gram_trace,
            gram_symmetry_relative=gram_sym,
            row_orthogonality_relative=orth,
            live_current_R_energy=live_energy,
            eigenvalues_descending=[float(x) for x in live_vals[:min(32, d)].tolist()],
        )
        return asdict(snap)
