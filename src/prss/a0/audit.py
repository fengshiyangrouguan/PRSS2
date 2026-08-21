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
    "G0": ("ess", "min", "context-overlap identifiability"),
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
