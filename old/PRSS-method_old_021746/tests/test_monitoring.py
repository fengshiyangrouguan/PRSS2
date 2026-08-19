import json
from types import SimpleNamespace

import torch

from experiments.compare_prss_runs import REQUIRED_VARIANTS, summarize
from prss.config import InterfaceSpec, PRSSConfig
from prss.monitoring import PRSSEvidenceMonitor, interface_evidence
from prss.system import PRSSSystem


def make_system():
  return PRSSSystem(PRSSConfig(
    interfaces={"tau": InterfaceSpec("tau", 6, 6, 2)},
    context_dim=4, gram_ema_rho=1.0,
    spectral_update_interval=1, spectral_warmup_steps=0))


def test_monitor_records_predictive_energy_geometry_and_gradients(tmp_path):
  torch.manual_seed(5)
  system = make_system()
  candidates = torch.randn(12, 6)
  readers = torch.randn(12, 1, 6)
  system.update_spectral_statistics({"tau": readers})
  system.maybe_spectral_update(0)
  state = system.make_state("tau", candidates)
  loss = state.quotient.square().mean()
  loss.backward()
  auxiliary = SimpleNamespace(
    readers_by_tau={"tau": readers}, candidates_by_tau={"tau": candidates},
    structured_logits=torch.randn(12, 1), unrestricted_logits=torch.randn(12, 1),
    targets=torch.randint(0, 2, (12,), dtype=torch.float32),
    response=torch.tensor(0.8), unrestricted_response=torch.tensor(0.7),
    occurrence_count=12,
  )
  monitor = PRSSEvidenceMonitor(
    tmp_path, "full", monitor_every=1, warmup_steps=100,
    tensorboard=False, fail_on_error=True)
  payload = monitor.record_step(
    0, 0,
    {"task": loss, "response": auxiliary.response,
     "spectral": torch.tensor(0.1),
     "unrestricted_response": auxiliary.unrestricted_response,
     "total": loss + 0.9},
    auxiliary, system, system, spectral_updated=True)
  evidence = payload["interfaces"]["tau"]
  assert evidence["projection_shape"] == [2, 6]
  assert evidence["projection_orthogonality_error"] < 1e-5
  assert 0 <= evidence["predictive_energy_captured_by_current_R"] <= 1
  assert 0 <= evidence["batch_predictive_energy_captured_by_current_R"] <= 1
  assert evidence["reader_gram_updates"] == 1
  summary = monitor.close({"gram_updated": False, "svd_updated": False})
  assert summary["health_status"] == "passed"
  with open(tmp_path / "step_metrics.jsonl", encoding="utf-8") as handle:
    assert json.loads(handle.readline())["spectral_updated_this_step"] is True


def test_interface_evidence_detects_nonorthogonal_direct_span_without_crashing():
  system = make_system()
  state = system.quotients.state_for("tau")
  state.R.mul_(3.0)
  evidence = interface_evidence(system, {}, {})["tau"]
  assert evidence["projection_orthogonality_error"] > 1
  assert 0 <= evidence["predictive_energy_captured_by_current_R"] <= 1


def complete_record(variant, seed, ap):
  return {
    "path": "unused", "variant": variant, "seed": seed,
    "host_dim": 4, "candidate_dim": 8,
    "test_ap": ap, "test_auc": ap,
    "structured_response_nll": 0.6,
    "unrestricted_response_nll": 0.5,
    "structured_unrestricted_ratio": 1.2,
    "predictive_energy_coverage": ap,
    "operator_statistic_energy_at_host_k": ap,
    "tail_at_host_k": 1 - ap,
    "gram_updates": 5 if variant == "full" else 0,
    "spectral_updates": 2 if variant == "full" else 0,
    "inference_contract": {
      "validation_test_gram_updated": False,
      "validation_test_svd_updated": False,
      "standard_inference_outside_reader_used": False,
      "standard_inference_gram_updated": False,
      "standard_inference_svd_updated": False,
    },
    "monitor_health": "passed",
  }


def test_comparison_requires_complete_paired_evidence_and_reports_positive_ci():
  records = []
  for variant in REQUIRED_VARIANTS:
    for seed in (0, 1, 2):
      ap = 0.80 if variant == "full" else 0.60
      records.append(complete_record(variant, seed, ap))
  report = summarize(records, bootstrap_samples=200, seed=1)
  assert report["missing_variants"] == []
  assert report["evidence_status"] == "supported_by_completed_paired_ablations"
  assert report["paired_effects"]["full_minus_direct"]["test_ap"]["ci95"][0] > 0


def test_comparison_refuses_to_claim_success_from_single_full_run():
  report = summarize([complete_record("full", 0, 0.8)], bootstrap_samples=10)
  assert report["evidence_status"] == "incomplete_evidence_missing_runs"
  assert "direct" in report["missing_variants"]
