"""A0 phase C: audit metrics and the G0-G4 failure gates (theory doc 5.7/10).

Every gate failure stops the algorithm hypothesis instead of looping back to
a different solver.  In ``report`` mode the gates only annotate the audit
table; in ``stop`` mode a failed gate aborts the run with a stop reason.
"""

from typing import Dict, Optional

import torch


def ridge_coefficients(features: torch.Tensor, targets: torch.Tensor,
                       lambda_reg: float):
    """(FᵀF + λI)^{-1} FᵀT in float64; features (n, d), targets (n, m)."""
    features = features.detach().to(dtype=torch.float64)
    targets = targets.detach().to(dtype=torch.float64)
    d = features.shape[-1]
    eye = torch.eye(d, dtype=torch.float64, device=features.device)
    gram = features.transpose(0, 1) @ features + lambda_reg * eye
    chol = torch.linalg.cholesky(gram)
    rhs = features.transpose(0, 1) @ targets
    return torch.cholesky_solve(rhs, chol)


class ResidualAccumulator:
    """Streaming normalized ridge residual over an audit window.

    Keeps only FᵀF / FᵀT / ‖T‖² (window rows never materialize): the fitted
    sum is trace(Wᵀ FᵀT) with W = (FᵀF + λI)^{-1} FᵀT, so the normalized
    residual is (‖T‖² − fitted)/‖T‖².
    """

    def __init__(self, d_features: int, d_targets: int):
        self.ftf = torch.zeros(int(d_features), int(d_features),
                               dtype=torch.float64)
        self.ftt = torch.zeros(int(d_features), int(d_targets),
                               dtype=torch.float64)
        self.tt_sum = 0.0
        self.n = 0

    @torch.no_grad()
    def accumulate(self, features, targets) -> None:
        f = features.detach().to(dtype=torch.float64)
        t = targets.detach().to(dtype=torch.float64)
        if f.shape[-1] != self.ftf.shape[0] or t.shape[-1] != self.ftt.shape[1]:
            raise ValueError("width mismatch: {} vs {}".format(
                (f.shape, t.shape), (self.ftf.shape, self.ftt.shape)))
        if self.ftf.device != f.device:
            self.ftf = self.ftf.to(f.device)
            self.ftt = self.ftt.to(f.device)
        self.ftf.add_(f.transpose(0, 1) @ f)
        self.ftt.add_(f.transpose(0, 1) @ t)
        self.tt_sum += float((t * t).sum().item())
        self.n += int(f.shape[0])

    def relative_residual(self, lambda_reg: float) -> float:
        if self.n == 0 or self.tt_sum <= 1e-12:
            return 0.0
        eye = torch.eye(self.ftf.shape[0], dtype=torch.float64,
                        device=self.ftf.device)
        a = self.ftf + lambda_reg * eye
        chol = torch.linalg.cholesky(a)
        w = torch.cholesky_solve(self.ftt, chol)
        fitted = float((w.transpose(0, 1) @ self.ftt).diag().sum().item())
        return max(0.0, (self.tt_sum - fitted) / self.tt_sum)


def relative_residual(features, targets, lambda_reg: float) -> float:
    """‖T − F(FᵀF+λI)^{-1}FᵀT‖² / ‖T‖² (normalized ridge residual)."""
    w = ridge_coefficients(features, targets, lambda_reg)
    residual = targets.detach().to(dtype=torch.float64) - \
        features.detach().to(dtype=torch.float64) @ w
    return float((residual ** 2).sum().item() /
                 max(float((targets.detach().to(dtype=torch.float64) ** 2).sum().item()),
                     1e-12))


# ------------------------------------------------------------ proper scores

def _logistic_ridge(features, targets, lambda_reg=1e-3, n_iter=20):
    """IRLS fit of a logistic readout p = sigmoid(F w); returns w (float64).

    A strict proper scoring rule (log/Brier) evaluates this same small
    readout class on compressed vs rich histories (theory doc 5.7.2).
    """
    features = features.detach().to(dtype=torch.float64)
    targets = targets.detach().to(dtype=torch.float64)
    d = features.shape[-1]
    w = torch.zeros(d, dtype=torch.float64, device=features.device)
    eye = torch.eye(d, dtype=torch.float64, device=features.device)
    for _ in range(int(n_iter)):
        eta = features @ w
        p = torch.sigmoid(eta).clamp_min(1e-6)
        wt = (p * (1.0 - p)).clamp_min(1e-6)
        working = eta + (targets - p) / wt
        fw = features * wt.sqrt().unsqueeze(-1)
        gram = fw.transpose(0, 1) @ fw + lambda_reg * eye
        rhs = (fw.transpose(0, 1) @ (working * wt.sqrt())).unsqueeze(-1)
        chol = torch.linalg.cholesky(gram)
        w = torch.cholesky_solve(rhs, chol).squeeze(-1)
    return w


def _score(p, y):
    p = p.clamp(1e-9, 1.0 - 1e-9)
    log_score = float(-(y * p.log() + (1.0 - y) * (1.0 - p).log()).mean())
    brier = float(((p - y) ** 2).mean())
    return log_score, brier


def proper_score_regret(z_rows, x_rows, y_rows, lambda_reg=1e-3,
                        fit_frac=0.6, n_iter=20) -> Dict:
    """Log/Brier regret of the compressed readout vs the rich-history readout.

    ``z_rows`` (n, r) compressed coordinates, ``x_rows`` (n, p) rich history,
    ``y_rows`` (n,) labels.  The first fit_frac rows fit the logistic
    readouts; the rest evaluate both scores (lower is better, so a positive
    regret means the compression costs prediction quality).
    """
    n = int(z_rows.shape[0])
    n_fit = max(1, int(n * fit_frac))
    if n < 2 * max(2, z_rows.shape[-1]):
        return {"n": n, "n_fit": n_fit, "n_eval": 0,
                "log_regret": float("nan"), "brier_regret": float("nan")}
    y = y_rows.detach().to(dtype=torch.float64)
    w_z = _logistic_ridge(z_rows[:n_fit], y[:n_fit], lambda_reg, n_iter)
    w_x = _logistic_ridge(x_rows[:n_fit], y[:n_fit], lambda_reg, n_iter)
    p_z = torch.sigmoid(z_rows[n_fit:].to(dtype=torch.float64) @ w_z)
    p_x = torch.sigmoid(x_rows[n_fit:].to(dtype=torch.float64) @ w_x)
    log_z, brier_z = _score(p_z, y[n_fit:])
    log_x, brier_x = _score(p_x, y[n_fit:])
    return {
        "n": n, "n_fit": n_fit, "n_eval": n - n_fit,
        "log_score_compressed": log_z,
        "log_score_rich": log_x,
        "brier_compressed": brier_z,
        "brier_rich": brier_x,
        "log_regret": log_z - log_x,
        "brier_regret": brier_z - brier_x,
    }


def prediction_residuals(u_rows, x_rows, z_rows, lambda_audit: float) -> Dict:
    """G1 support: eps_pred on the r-dimensional coordinate vs the unrestricted
    (full-x) ridge baseline on the same rows."""
    base = relative_residual(x_rows, u_rows, lambda_audit)
    compressed = relative_residual(z_rows, u_rows, lambda_audit)
    return {
        "unrestricted_ridge_residual": base,
        "rank_r_ridge_residual": compressed,
        "prediction_gap": max(0.0, compressed - base),
    }


def closure_residual(z_rich_rows, z_rec_rows, sigma) -> Dict:
    """G2 support: eps_cl in the Sigma-weighted predictive metric (theory
    doc 5.5 eq. 8)."""
    z_rich = z_rich_rows.detach().to(dtype=torch.float64)
    z_rec = z_rec_rows.detach().to(dtype=torch.float64)
    sig = sigma.detach().to(dtype=torch.float64)
    diff = z_rich - z_rec
    weighted = diff * sig
    return {
        "closure_residual": float((weighted ** 2).mean().item()),
        "closure_residual_unweighted": float((diff ** 2).mean().item()),
        "n_closure_rows": int(z_rich.shape[0]),
    }


def path_gain_report(operators) -> Dict:
    """G3 support: per-constructor source-child Lipschitz gains and the
    worst-case depth-D product (first version reports numbers only; no
    contraction/JSR certificate is claimed)."""
    layers = {}
    for op in operators:
        try:
            layer = int(op.parent_tau.split("layer")[1])
        except (IndexError, ValueError):
            layer = 0
        layers.setdefault(layer, []).append(op.gain())
    per_layer = {int(l): max(gains) for l, gains in layers.items()}
    product = 1.0
    for l in sorted(per_layer):
        product *= max(per_layer[l], 1e-12)
    return {
        "gain_by_parent_layer": per_layer,
        "path_gain_product": float(product),
    }


GATE_DEFS = {
    # gate: (audit_key, comparator, doc)
    "G0": ("ess_frac_min", "min", "context-overlap identifiability"),
    "G1": ("rank_tail_max", "max", "fixed-budget compressibility"),
    "G2": ("closure_residual_max", "max", "recursive closure"),
    "G3": ("path_gain_product", "max", "deep-tree stability"),
    "G4": ("auc_delta", "min", "real task gain over the baseline readout"),
}


def evaluate_gates(audit: Dict, gates: Optional[Dict], mode: str = "report",
                   phase: str = "") -> Dict:
    """Judge the gates present in ``audit`` against thresholds.

    ``gates`` maps gate name to threshold (None = not enforced).  Returns
    {"gate_results": {name: {"value", "threshold", "passed"}},
     "gates_passed": bool, "failed_gates": [names]}.
    """
    results = {}
    failed = []
    for name, (key, comp, _doc) in GATE_DEFS.items():
        if key not in audit or gates is None or name not in gates:
            continue
        threshold = gates.get(name)
        if threshold is None:
            continue
        value = float(audit[key])
        if value != value:  # NaN (e.g. AUC delta without positives): no verdict
            continue
        passed = value >= threshold if comp == "min" else value <= threshold
        results[name] = {"value": value, "threshold": float(threshold),
                         "passed": bool(passed)}
        if not passed:
            failed.append(name)
    return {
        "gate_results": results,
        "gates_passed": not failed,
        "failed_gates": failed,
    }
