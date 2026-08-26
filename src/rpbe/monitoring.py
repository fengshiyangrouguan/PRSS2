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
                 reset_files: bool = True):
        self.root = Path(run_dir) / "monitor"
        self.root.mkdir(parents=True, exist_ok=True)
        self.step_path = self.root / "step_metrics.jsonl"
        self.epoch_path = self.root / "epoch_metrics.jsonl"
        self.alert_path = self.root / "alerts.jsonl"
        self.fingerprint_path = self.root / "rpbe_fingerprints.jsonl"
        self.fail_on_error = bool(fail_on_error)
        if reset_files:
            for p in (self.step_path, self.epoch_path, self.alert_path,
                      self.fingerprint_path):
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

    def validate_kf(self, kf_by_tau: Dict[str, float], dims: Dict[str, int],
                    step: int):
        """Per-interface Ky Fan score bounds: 0 <= J_tau <= d_tau."""
        for tau, j in kf_by_tau.items():
            v = _finite_float(j)
            d = int(dims.get(tau, 0))
            if v is None:
                self.alert("error", "nonfinite_kf_score",
                           f"{tau} J={j}", step=step, interface=tau)
                continue
            if v < 0 or v > d + 1e-4:
                self.alert("error", "kf_score_out_of_bounds",
                           f"{tau} J={v:.6f} dim={d}", step=step, interface=tau)

    def write_step(self, row: Dict):
        self._append(self.step_path, row)

    def write_epoch(self, row: Dict):
        self._append(self.epoch_path, row)

    def save_fingerprint(self, epoch: int, fingerprint: Dict):
        self._append(self.fingerprint_path, {"epoch": int(epoch),
                                             "fingerprint": fingerprint})

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
