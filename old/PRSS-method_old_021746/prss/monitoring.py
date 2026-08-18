"""Evidence-oriented monitoring for PRSS training and mechanism validation."""

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from prss.spectral import reader_gram


@dataclass
class MonitorThresholds:
  max_orthogonality_error: float = 1e-3
  max_gram_symmetry_error: float = 1e-5
  min_reader_frobenius_norm: float = 1e-6
  min_candidate_coordinate_std: float = 1e-6
  max_structured_to_unrestricted_loss_ratio: float = 2.0
  max_projector_distance: float = 10.0


def _finite_float(value):
  value = float(value)
  return value if math.isfinite(value) else None


def _binary_accuracy(logits, targets):
  predictions = (torch.sigmoid(logits.detach().reshape(-1)) >= 0.5)
  truth = targets.detach().reshape(-1) >= 0.5
  return float((predictions == truth).float().mean())


def gradient_group_norms(module):
  """L2 gradient norms split by the mechanism that receives the gradient."""
  sums = Counter()
  counts = Counter()
  for name, parameter in module.named_parameters():
    if parameter.grad is None:
      continue
    if (".lifts." in name or name.startswith("lifts.") or
        ".preagg_lift." in name or name.startswith("preagg_lift.") or
        ".preagg_lifts." in name or name.startswith("preagg_lifts.") or
        ".leaf_lift." in name or name.startswith("leaf_lift.")):
      group = "lift"
    elif ".readers." in name or name.startswith("readers."):
      group = "structured_reader"
    elif ".outside." in name or name.startswith("outside."):
      group = "outside_encoder"
    elif "unrestricted_readers" in name:
      group = "unrestricted_reader"
    elif name.endswith(".R"):
      group = "direct_projection"
    else:
      group = "host"
    sums[group] += float(parameter.grad.detach().double().square().sum())
    counts[group] += parameter.numel()
  result = {name: math.sqrt(value) for name, value in sums.items()}
  result["parameter_counts_with_gradient"] = dict(counts)
  return result


@torch.no_grad()
def predictive_energy_coverage(rows, reader_matrix):
  """Fraction of held-out reader-Gram energy captured by the span of rows."""
  rows = rows.detach().double()
  span_rows = torch.linalg.qr(rows.T, mode="reduced").Q.T
  gram = reader_gram(reader_matrix.detach()).double()
  trace = gram.diagonal().sum().clamp_min(torch.finfo(torch.double).eps)
  return float((torch.trace(span_rows @ gram @ span_rows.T) / trace).clamp(0, 1))


@torch.no_grad()
def interface_evidence(system, readers_by_tau, candidates_by_tau):
  result = {}
  for tau, spec in system.config.interfaces.items():
    state = system.quotients.state_for(tau)
    rows = state.R.detach().double()
    identity = torch.eye(spec.host_dim, device=rows.device, dtype=rows.dtype)
    orthogonality = torch.linalg.matrix_norm(rows @ rows.T - identity, ord="fro")
    orthogonality_relative = orthogonality / math.sqrt(spec.host_dim)
    gram = state.G_ema.detach().double()
    gram_symmetric = 0.5 * (gram + gram.T)
    gram_trace = gram_symmetric.diagonal().sum().clamp_min(torch.finfo(torch.double).eps)
    gram_symmetry = torch.linalg.matrix_norm(gram - gram.T, ord="fro")
    eigenvalues = torch.linalg.eigvalsh(gram_symmetric).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(torch.finfo(torch.double).eps)
    positive = probabilities[probabilities > 0]
    effective_rank = torch.exp(-(positive * positive.log()).sum()) if len(positive) else torch.zeros(())
    # Direct projection may be non-orthogonal; evaluate the span through an orthonormal basis.
    span_rows = torch.linalg.qr(rows.T, mode="reduced").Q.T
    captured = torch.trace(span_rows @ gram_symmetric @ span_rows.T) / gram_trace
    entry = {
      "host_dim": spec.host_dim,
      "candidate_dim": spec.candidate_dim,
      "projection_shape": list(rows.shape),
      "projection_orthogonality_error": float(orthogonality),
      "projection_orthogonality_error_relative": float(orthogonality_relative),
      "gram_symmetry_error": float(gram_symmetry),
      "gram_trace": float(gram_trace),
      "gram_effective_rank": float(effective_rank),
      "predictive_energy_captured_by_current_R": float(captured.clamp(0, 1)),
      "reader_gram_updates": int(state.gram_updates),
      "spectral_updates": int(state.spectral_updates),
      "projector_distance_last_update": float(state.last_projector_distance),
      "principal_angle_max_last_update": float(state.last_principal_angles.max()),
    }
    readers = readers_by_tau.get(tau)
    if readers is not None and readers.numel():
      norms = readers.detach().double().square().sum(dim=(-2, -1)).sqrt()
      entry["reader_frobenius_norm"] = {
        "mean": float(norms.mean()), "std": float(norms.std(unbiased=False)),
        "min": float(norms.min()), "max": float(norms.max()),
      }
      entry["batch_predictive_energy_captured_by_current_R"] = predictive_energy_coverage(
        rows, readers)
    candidates = candidates_by_tau.get(tau)
    if candidates is not None and candidates.numel():
      values = candidates.detach().double().reshape(-1, spec.candidate_dim)
      coordinate_std = values.std(dim=0, unbiased=False)
      vector_norm = values.norm(dim=-1)
      entry["candidate_state"] = {
        "samples": len(values),
        "vector_norm_mean": float(vector_norm.mean()),
        "vector_norm_std": float(vector_norm.std(unbiased=False)),
        "coordinate_std_mean": float(coordinate_std.mean()),
        "coordinate_std_min": float(coordinate_std.min()),
        "finite_fraction": float(torch.isfinite(values).double().mean()),
      }
    result[tau] = entry
  return result


class PRSSEvidenceMonitor:
  def __init__(self, output_dir, variant, monitor_every=50, warmup_steps=200,
               thresholds=None, tensorboard=True, fail_on_error=True):
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.snapshot_dir = self.output_dir / "projection_snapshots"
    self.snapshot_dir.mkdir(exist_ok=True)
    self.variant = variant
    self.monitor_every = max(1, int(monitor_every))
    self.warmup_steps = int(warmup_steps)
    self.thresholds = thresholds or MonitorThresholds()
    self.fail_on_error = bool(fail_on_error)
    self.started_at = time.time()
    self.alert_counts = Counter()
    self.last_payload = None
    self.writer = None
    if tensorboard:
      try:
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(self.output_dir / "tensorboard"))
      except ImportError:
        self._write_alert(-1, "warning", "tensorboard_unavailable",
                          "Install tensorboard to enable the event dashboard")
    with open(self.output_dir / "monitor_config.json", "w", encoding="utf-8") as handle:
      json.dump({
        "variant": variant,
        "monitor_every": self.monitor_every,
        "warmup_steps": self.warmup_steps,
        "fail_on_error": self.fail_on_error,
        "thresholds": asdict(self.thresholds),
      }, handle, indent=2)

  def _append(self, filename, payload):
    with open(self.output_dir / filename, "a", encoding="utf-8") as handle:
      handle.write(json.dumps(payload, allow_nan=False) + "\n")

  def _write_alert(self, step, severity, code, message, interface=None, value=None):
    alert = {
      "time": time.time(), "step": int(step), "severity": severity,
      "code": code, "message": message,
    }
    if interface is not None:
      alert["interface"] = interface
    if value is not None:
      alert["value"] = _finite_float(value)
    self.alert_counts[severity] += 1
    self._append("alerts.jsonl", alert)
    if severity == "error" and self.fail_on_error:
      raise RuntimeError("PRSS monitor error [{}]: {}".format(code, message))

  def should_record(self, step):
    return step == 0 or (step + 1) % self.monitor_every == 0

  def _check(self, step, payload, projection_expected_orthogonal,
             expected_gradient_groups):
    losses = payload["losses"]
    if any(value is None for value in losses.values()):
      self._write_alert(step, "error", "nonfinite_loss", "A monitored loss is NaN or Inf")
    if step < self.warmup_steps:
      return
    for group in expected_gradient_groups:
      if payload["gradients"].get(group, 0.0) <= 0:
        self._write_alert(step, "warning", "missing_gradient_{}".format(group),
                          "Expected mechanism received no gradient: {}".format(group))
    unrestricted = max(losses["unrestricted_response"], 1e-12)
    ratio = losses["response"] / unrestricted
    if ratio > self.thresholds.max_structured_to_unrestricted_loss_ratio:
      self._write_alert(step, "warning", "structured_reader_gap",
                        "Structured reader is much worse than unrestricted reader", value=ratio)
    for tau, evidence in payload["interfaces"].items():
      if projection_expected_orthogonal and (
          evidence["projection_orthogonality_error_relative"] >
          self.thresholds.max_orthogonality_error):
        self._write_alert(step, "error", "projection_not_orthogonal",
                          "Relative Frobenius error of R R^T exceeds threshold", tau,
                          evidence["projection_orthogonality_error_relative"])
      if evidence["gram_symmetry_error"] > self.thresholds.max_gram_symmetry_error:
        self._write_alert(step, "error", "gram_not_symmetric",
                          "EMA Gram is not symmetric", tau, evidence["gram_symmetry_error"])
      if evidence["projector_distance_last_update"] > self.thresholds.max_projector_distance:
        self._write_alert(step, "warning", "unstable_projection",
                          "Consecutive predictive subspaces changed sharply", tau,
                          evidence["projector_distance_last_update"])
      reader = evidence.get("reader_frobenius_norm")
      if reader and reader["mean"] < self.thresholds.min_reader_frobenius_norm:
        self._write_alert(step, "warning", "reader_collapse",
                          "Conditional reader norm is near zero", tau, reader["mean"])
      candidate = evidence.get("candidate_state")
      if candidate and candidate["finite_fraction"] < 1.0:
        self._write_alert(step, "error", "nonfinite_candidate",
                          "Candidate state contains NaN/Inf", tau, candidate["finite_fraction"])
      if candidate and candidate["coordinate_std_mean"] < self.thresholds.min_candidate_coordinate_std:
        self._write_alert(step, "warning", "candidate_collapse",
                          "Candidate coordinates have near-zero batch variation", tau,
                          candidate["coordinate_std_mean"])

  def record_step(self, step, epoch, losses, auxiliary, model, system,
                  spectral_updated=False, projection_expected_orthogonal=True,
                  expected_gradient_groups=()):
    if not self.should_record(step):
      return None
    interfaces = interface_evidence(
      system, auxiliary.readers_by_tau, auxiliary.candidates_by_tau)
    payload = {
      "time": time.time(), "elapsed_seconds": time.time() - self.started_at,
      "step": int(step), "epoch": int(epoch), "variant": self.variant,
      "losses": {name: _finite_float(value.detach() if torch.is_tensor(value) else value)
                 for name, value in losses.items()},
      "response": {
        "structured_accuracy": _binary_accuracy(auxiliary.structured_logits, auxiliary.targets),
        "unrestricted_accuracy": _binary_accuracy(auxiliary.unrestricted_logits, auxiliary.targets),
        "structured_minus_unrestricted_loss": float(
          auxiliary.response.detach() - auxiliary.unrestricted_response.detach()),
      },
      "gradients": gradient_group_norms(model),
      "interfaces": interfaces,
      "spectral_updated_this_step": bool(spectral_updated),
      "occurrence_count": int(auxiliary.occurrence_count),
    }
    if torch.cuda.is_available():
      payload["cuda"] = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
      }
    self._check(step, payload, projection_expected_orthogonal,
                expected_gradient_groups)
    self._append("step_metrics.jsonl", payload)
    self.last_payload = payload
    if self.writer is not None:
      self._tensorboard_step(payload)
    return payload

  def _tensorboard_step(self, payload):
    step = payload["step"]
    for name, value in payload["losses"].items():
      if value is not None:
        self.writer.add_scalar("loss/{}".format(name), value, step)
    for name, value in payload["response"].items():
      self.writer.add_scalar("response/{}".format(name), value, step)
    for group, value in payload["gradients"].items():
      if group != "parameter_counts_with_gradient":
        self.writer.add_scalar("gradient_norm/{}".format(group), value, step)
    for tau, evidence in payload["interfaces"].items():
      for name in ("projection_orthogonality_error", "projection_orthogonality_error_relative",
                   "gram_symmetry_error", "gram_trace",
                   "gram_effective_rank", "predictive_energy_captured_by_current_R",
                   "projector_distance_last_update", "principal_angle_max_last_update"):
        self.writer.add_scalar("interface/{}/{}".format(tau, name), evidence[name], step)
      if "reader_frobenius_norm" in evidence:
        self.writer.add_scalar("interface/{}/reader_norm".format(tau),
                               evidence["reader_frobenius_norm"]["mean"], step)
      if "batch_predictive_energy_captured_by_current_R" in evidence:
        self.writer.add_scalar("interface/{}/batch_predictive_energy".format(tau),
                               evidence["batch_predictive_energy_captured_by_current_R"], step)
      if "candidate_state" in evidence:
        self.writer.add_scalar("interface/{}/candidate_coordinate_std".format(tau),
                               evidence["candidate_state"]["coordinate_std_mean"], step)

  @torch.no_grad()
  def record_epoch(self, epoch, global_step, train_metrics, validation, system):
    payload = {
      "time": time.time(), "epoch": int(epoch), "global_step": int(global_step),
      "train": train_metrics, "validation": validation,
      "spectral": system.spectral_diagnostics(),
    }
    self._append("epoch_metrics.jsonl", payload)
    snapshot = {
      "epoch": int(epoch), "global_step": int(global_step),
      "interfaces": {
        tau: {
          "R": system.quotients.state_for(tau).R.detach().cpu(),
          "G_ema": system.quotients.state_for(tau).G_ema.detach().cpu(),
          "eigenvalues": system.quotients.state_for(tau).eigenvalues.detach().cpu(),
        }
        for tau in system.config.interfaces
      },
    }
    torch.save(snapshot, self.snapshot_dir / "epoch_{:04d}.pt".format(epoch))
    if self.writer is not None:
      for name, value in validation.items():
        if value is not None:
          self.writer.add_scalar("validation/{}".format(name), value, epoch)

  def close(self, inference_contract=None):
    summary = {
      "variant": self.variant,
      "elapsed_seconds": time.time() - self.started_at,
      "alert_counts": dict(self.alert_counts),
      "inference_contract": inference_contract,
      "final_interfaces": self.last_payload["interfaces"] if self.last_payload else {},
      "health_status": "failed" if self.alert_counts["error"] else "passed_with_warnings"
                       if self.alert_counts["warning"] else "passed",
      "interpretation": (
        "Health status validates execution and mechanism instrumentation only; "
        "method efficacy requires the paired multi-seed ablation report."),
    }
    with open(self.output_dir / "monitor_summary.json", "w", encoding="utf-8") as handle:
      json.dump(summary, handle, indent=2)
    if self.writer is not None:
      self.writer.flush()
      self.writer.close()
    return summary
