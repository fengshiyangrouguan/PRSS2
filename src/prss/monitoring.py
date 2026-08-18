"""Monitoring utilities: JSONL writers, exact finiteness accounting, gradient stats."""

import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch


def _finite_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def grad_l2(module: Optional[torch.nn.Module]) -> float:
    if module is None:
        return 0.0
    total = 0.0
    for p in module.parameters():
        if not p.requires_grad or not p.is_leaf:
            continue
        if p.grad is not None:
            total += float(p.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def tensor_summary(x: Optional[torch.Tensor]) -> Dict:
    if x is None or x.numel() == 0:
        return {"count": 0, "finite_fraction": 1.0}
    y = x.detach().float().reshape(-1)
    finite = torch.isfinite(y)
    ff = float(finite.float().mean().item())
    if not bool(finite.any()):
        return {"count": int(y.numel()), "finite_fraction": ff}
    y = y[finite]
    return {
        "count": int(y.numel()),
        "finite_fraction": ff,
        "mean": float(y.mean().item()),
        "std": float(y.std(unbiased=False).item()),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "l2_mean": float(y.square().mean().sqrt().item()),
    }


def candidate_stats(trace) -> Dict:
    if trace is None:
        return {}
    by_tau = {}
    for occ in trace.occurrences.values():
        by_tau.setdefault(occ.tau, []).append(occ.state.candidate.detach())
    out = {}
    for tau, vals in by_tau.items():
        x = torch.stack(vals, dim=0)
        finite = torch.isfinite(x)
        coord_std = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).std(dim=0, unbiased=False)
        out[tau] = {
            **tensor_summary(x),
            "rows": int(x.shape[0]),
            "dim": int(x.shape[-1]),
            "coord_std_mean": float(coord_std.mean().item()),
            "coord_std_min": float(coord_std.min().item()),
            "coord_std_max": float(coord_std.max().item()),
            "finite_fraction": float(finite.float().mean().item()),
        }
    return out


def matrix_stats(matrices_by_tau: Dict[str, torch.Tensor]) -> Dict:
    """Exact finiteness accounting plus descriptive reader-matrix statistics.

    Finiteness is a hard invariant: never derive it from a floating-point mean of a
    boolean mask (CUDA reductions can make a value that is mathematically 1.0 appear as
    the adjacent float below 1.0).  Count finite/nonfinite entries as integers.
    """
    out = {}
    for tau, B in matrices_by_tau.items():
        detached = B.detach()
        finite_mask = torch.isfinite(detached)
        total = int(detached.numel())
        nonfinite_count = int((~finite_mask).sum(dtype=torch.int64).item()) if total else 0
        finite_count = total - nonfinite_count
        exact_all_finite = (nonfinite_count == 0)

        safe = torch.nan_to_num(detached.float(), nan=0.0, posinf=0.0, neginf=0.0)
        norm = safe.norm(dim=(-1, -2))
        out[tau] = {
            "occurrences": int(B.shape[0]),
            "response_rows": int(B.shape[-2]),
            "candidate_dim": int(B.shape[-1]),
            "elements": total,
            "finite_count": finite_count,
            "nonfinite_count": nonfinite_count,
            "all_finite": bool(exact_all_finite),
            "fro_mean": float(norm.mean().item()) if norm.numel() else 0.0,
            "fro_std": float(norm.std(unbiased=False).item()) if norm.numel() else 0.0,
            "fro_min": float(norm.min().item()) if norm.numel() else 0.0,
            "fro_max": float(norm.max().item()) if norm.numel() else 0.0,
            "finite_fraction": (float(finite_count) / float(total) if total else 1.0),
        }
    return out


def module_finiteness(module: Optional[torch.nn.Module]) -> Dict:
    if module is None:
        return {"parameters": 0, "finite_fraction": 1.0}
    total = 0
    finite = 0
    max_abs = 0.0
    for p in module.parameters():
        x = p.detach().float().reshape(-1)
        total += int(x.numel())
        if x.numel():
            f = torch.isfinite(x)
            finite += int(f.sum().item())
            if bool(f.any()):
                max_abs = max(max_abs, float(x[f].abs().max().item()))
    return {
        "parameters": total,
        "finite_fraction": (float(finite) / float(total) if total else 1.0),
        "max_abs": max_abs,
    }


class MonitorWriter:
    def __init__(self, run_dir: Path, fail_on_error: bool = True,
                 orth_tol: float = 5e-4, gram_sym_tol: float = 1e-6,
                 response_gap_warn: float = 0.25, reset_files: bool = True):
        self.root = Path(run_dir) / "monitor"
        self.root.mkdir(parents=True, exist_ok=True)
        self.step_path = self.root / "step_metrics.jsonl"
        self.epoch_path = self.root / "epoch_metrics.jsonl"
        self.alert_path = self.root / "alerts.jsonl"
        self.snap_dir = self.root / "projection_snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.fail_on_error = bool(fail_on_error)
        self.orth_tol = float(orth_tol)
        self.gram_sym_tol = float(gram_sym_tol)
        self.response_gap_warn = float(response_gap_warn)
        if reset_files:
            for p in (self.step_path, self.epoch_path, self.alert_path):
                if p.exists():
                    p.unlink()

    @staticmethod
    def _append(path: Path, obj: Dict):
        with path.open("a") as f:
            f.write(json.dumps(obj, allow_nan=True) + "\n")

    def alert(self, severity: str, code: str, message: str, **meta):
        row = {"severity": severity, "code": code, "message": message, **meta}
        self._append(self.alert_path, row)
        print(f"MONITOR_{severity.upper()} code={code} {message}", flush=True)
        if severity == "error" and self.fail_on_error:
            raise RuntimeError(f"monitor invariant failed [{code}]: {message}")

    def validate_losses(self, losses: Dict[str, float], step: int):
        for name, value in losses.items():
            v = _finite_float(value)
            if v is None:
                self.alert("error", "nonfinite_loss", f"{name}={value}", step=step)

    def validate_spectral(self, spectral: Dict, step: int):
        for name, snap in spectral.items():
            orth = float(snap.get("row_orthogonality_relative", 0.0))
            sym = float(snap.get("gram_symmetry_relative", 0.0))
            # The direct ablation learns R end-to-end without an orthogonality
            # constraint; its snapshot flags that the invariant does not apply.
            if snap.get("projection_expected_orthogonal", True):
                if not math.isfinite(orth) or orth > self.orth_tol:
                    self.alert("error", "quotient_not_row_orthonormal",
                               f"{name} orthogonality={orth:.3e}", step=step, interface=name)
            if not math.isfinite(sym) or sym > self.gram_sym_tol:
                self.alert("error", "gram_not_symmetric",
                           f"{name} symmetry={sym:.3e}", step=step, interface=name)

    def write_step(self, row: Dict):
        self._append(self.step_path, row)

    def write_epoch(self, row: Dict):
        self._append(self.epoch_path, row)

    def save_projection_snapshot(self, epoch: int, prss):
        if prss is None:
            return
        payload = {}
        for tau, compressor in prss.quotients.items():
            payload[tau] = {
                "R": compressor.projection().detach().cpu(),
                "snapshot": compressor.snapshot(),
            }
        torch.save(payload, self.snap_dir / f"epoch_{int(epoch):04d}.pt")

    def finalize(self, summary: Dict):
        alerts = []
        if self.alert_path.exists():
            alerts = [json.loads(x) for x in self.alert_path.read_text().splitlines() if x.strip()]
        out = {
            "status": "complete",
            "error_alerts": sum(a.get("severity") == "error" for a in alerts),
            "warning_alerts": sum(a.get("severity") == "warning" for a in alerts),
            "summary": summary,
        }
        with (self.root / "monitor_summary.json").open("w") as f:
            json.dump(out, f, indent=2, allow_nan=True)
