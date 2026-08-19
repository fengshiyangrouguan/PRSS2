"""Known-subspace D,E -> A -> Y sanity test required before TGN integration.

This is a method-identification test, not evidence about Wikipedia.  D and E are marginally
uninformative about Y; their interaction through a known k-dimensional subspace determines Y.
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prss.config import InterfaceSpec, PRSSConfig
from prss.spectral import principal_angles, random_semi_orthogonal
from prss.system import PRSSSystem


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=1200)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--candidate-dim", type=int, default=12)
  parser.add_argument("--host-dim", type=int, default=3)
  parser.add_argument("--learning-rate", type=float, default=2e-3)
  parser.add_argument("--lambda-spec", type=float, default=0.05)
  parser.add_argument("--update-interval", type=int, default=20)
  parser.add_argument("--warmup", type=int, default=40)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--device", default="auto")
  parser.add_argument("--output", default="outputs/synthetic_tree.json")
  parser.add_argument("--assert-max-angle", type=float, default=0.20)
  return parser.parse_args()


def make_true_rows(dimension, rank, device):
  return torch.linalg.qr(torch.randn(dimension, rank, device=device), mode="reduced").Q.T


def sample_batch(batch_size, true_rows):
  device = true_rows.device
  d = true_rows.shape[1]
  left = torch.randn(batch_size, d, device=device)
  right_signal = torch.randn(batch_size, true_rows.shape[0], device=device)
  right_noise = torch.randn(batch_size, d, device=device)
  projector = true_rows.T @ true_rows
  right_noise = right_noise @ (torch.eye(d, device=device) - projector)
  right = right_signal @ true_rows + right_noise
  target = ((left @ true_rows.T) * right_signal).sum(dim=-1) / np.sqrt(true_rows.shape[0])
  return left, right, target


class SyntheticPRSS(nn.Module):
  def __init__(self, d, k, lambda_spec, update_interval, warmup):
    super().__init__()
    config = PRSSConfig(
      interfaces={
        "branch": InterfaceSpec("branch", raw_dim=d, candidate_dim=d, host_dim=k,
                                response_dim=1),
        # The root interface is compatibility mode d_tau == k_tau, not dimensional compression.
        "parent": InterfaceSpec("parent", raw_dim=4, candidate_dim=4, host_dim=4,
                                response_dim=1),
      },
      context_dim=32,
      root_metadata_dim=1,
      parent_local_dim=2,
      relation_count=3,
      relation_dim=8,
      reader_hidden_dim=64,
      lambda_resp=1.0,
      lambda_spec=lambda_spec,
      gram_ema_rho=0.1,
      spectral_update_interval=update_interval,
      spectral_warmup_steps=warmup,
      ridge_eps=1e-6,
      initialization="random",
    )
    self.prss = PRSSSystem(config, no_lift_types={"branch", "parent"})
    self.aggregator = nn.Sequential(
      nn.Linear(2 * k + 2, 32), nn.GELU(), nn.Linear(32, 4))
    self.readout = nn.Linear(4, 1)

  def forward_components(self, left, right):
    batch_size = len(left)
    left_state = self.prss.make_state("branch", left)
    right_state = self.prss.make_state("branch", right)
    parent_local = torch.zeros(batch_size, 2, device=left.device)
    parent_raw = self.aggregator(torch.cat(
      [left_state.quotient, right_state.quotient, parent_local], dim=-1))
    parent_state = self.prss.make_state("parent", parent_raw)
    task_prediction = self.readout(parent_state.quotient).squeeze(-1)

    root_metadata = torch.zeros(batch_size, 1, device=left.device)
    parent_context = self.prss.outside.root_context(root_metadata, "parent")
    right_summary = self.prss.outside.summarize_siblings(
      {"branch": right_state.candidate.unsqueeze(-2)}, parent_context)
    left_summary = self.prss.outside.summarize_siblings(
      {"branch": left_state.candidate.unsqueeze(-2)}, parent_context)
    left_context = self.prss.outside.child_context(
      parent_context, parent_local, 1, 1.0, right_summary, "branch")
    right_context = self.prss.outside.child_context(
      parent_context, parent_local, 2, 1.0, left_summary, "branch")

    left_logits, left_matrix, _ = self.prss.structured_read(
      "branch", left_context, left_state.candidate)
    right_logits, right_matrix, _ = self.prss.structured_read(
      "branch", right_context, right_state.candidate)
    parent_logits, parent_matrix, _ = self.prss.structured_read(
      "parent", parent_context, parent_state.candidate)
    unrestricted = torch.cat([
      self.prss.unrestricted_read("branch", left_context.detach(),
                                 left_state.candidate.detach()),
      self.prss.unrestricted_read("branch", right_context.detach(),
                                 right_state.candidate.detach()),
      self.prss.unrestricted_read("parent", parent_context.detach(),
                                 parent_state.candidate.detach()),
    ], dim=0).squeeze(-1)
    return {
      "task": task_prediction,
      "structured": torch.cat([left_logits, right_logits, parent_logits], dim=0).squeeze(-1),
      "unrestricted": unrestricted,
      "branch_readers": torch.cat([left_matrix, right_matrix], dim=0),
      "parent_readers": parent_matrix,
      "branch_candidates": torch.cat([left_state.candidate, right_state.candidate], dim=0),
      "parent_candidates": parent_state.candidate,
    }


def subspace_coverage(rows, true_rows):
  return float((rows @ true_rows.T).square().sum() / true_rows.shape[0])


def train_task_baseline(kind, true_rows, steps, batch_size, learning_rate):
  d = true_rows.shape[1]
  k = true_rows.shape[0]
  device = true_rows.device
  if kind == "direct":
    projection = nn.Linear(d, k, bias=False).to(device)
    parameters = list(projection.parameters())
    project = projection
  else:
    if kind == "random":
      rows = random_semi_orthogonal(k, d, device=device)
    elif kind == "pca":
      left, right, _ = sample_batch(8192, true_rows)
      data = torch.cat([left, right], dim=0)
      _, _, vh = torch.linalg.svd(data - data.mean(dim=0), full_matrices=False)
      rows = vh[:k]
    else:
      raise ValueError(kind)
    project = lambda values: F.linear(values, rows)
    parameters = []
  predictor = nn.Sequential(nn.Linear(2 * k, 32), nn.GELU(), nn.Linear(32, 1)).to(device)
  optimizer = torch.optim.Adam(parameters + list(predictor.parameters()), lr=learning_rate)
  for _ in range(steps):
    left, right, target = sample_batch(batch_size, true_rows)
    prediction = predictor(torch.cat([project(left), project(right)], dim=-1)).squeeze(-1)
    loss = F.mse_loss(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  left, right, target = sample_batch(4096, true_rows)
  with torch.no_grad():
    prediction = predictor(torch.cat([project(left), project(right)], dim=-1)).squeeze(-1)
    mse = F.mse_loss(prediction, target)
    learned_rows = projection.weight if kind == "direct" else rows
    # Direct rows need orthonormalization before geometric coverage is meaningful.
    learned_rows = torch.linalg.qr(learned_rows.T, mode="reduced").Q.T
  return {"test_mse": float(mse), "true_subspace_coverage": subspace_coverage(
    learned_rows, true_rows)}


def run(args):
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  if args.device == "auto":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device(args.device)
  true_rows = make_true_rows(args.candidate_dim, args.host_dim, device)
  model = SyntheticPRSS(args.candidate_dim, args.host_dim, args.lambda_spec,
                       args.update_interval, args.warmup).to(device)
  main_parameters = [parameter for name, parameter in model.named_parameters()
                     if "unrestricted_readers" not in name]
  diagnostic_parameters = list(model.prss.unrestricted_readers.parameters())
  optimizer = torch.optim.Adam(main_parameters, lr=args.learning_rate, weight_decay=1e-5)
  diagnostic_optimizer = torch.optim.Adam(diagnostic_parameters, lr=args.learning_rate)
  history = []

  for step in range(args.steps):
    model.train()
    model.prss.set_spectral_updates_allowed(True)
    left, right, target = sample_batch(args.batch_size, true_rows)
    output = model.forward_components(left, right)
    repeated_target = target.repeat(3)
    task_loss = F.mse_loss(output["task"], target)
    response = F.mse_loss(output["structured"], repeated_target)
    spectral = 0.5 * (
      model.prss.state_spectral_loss(
        "branch", output["branch_readers"], output["branch_candidates"]) +
      model.prss.state_spectral_loss(
        "parent", output["parent_readers"], output["parent_candidates"]))
    spectral_ready = any(
      int(state.spectral_updates.item()) > 0
      for state in model.prss.quotients.states.values())
    total = task_loss + response + (args.lambda_spec * spectral if spectral_ready else 0.0)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    diagnostic = F.mse_loss(output["unrestricted"], repeated_target)
    diagnostic_optimizer.zero_grad()
    diagnostic.backward()
    diagnostic_optimizer.step()

    model.prss.update_spectral_statistics({
      "branch": output["branch_readers"], "parent": output["parent_readers"]})
    updated = model.prss.maybe_spectral_update(step)
    if updated or step in (0, args.steps - 1):
      rows = model.prss.quotients.state_for("branch").R
      angles = principal_angles(rows, true_rows)
      history.append({
        "step": step + 1,
        "task_mse": float(task_loss.detach()),
        "response_mse": float(response.detach()),
        "spectral_tail": float(spectral.detach()),
        "max_principal_angle": float(angles.max()),
        "subspace_coverage": subspace_coverage(rows, true_rows),
      })

  model.eval()
  model.prss.set_spectral_updates_allowed(False)
  left, right, target = sample_batch(8192, true_rows)
  with torch.no_grad():
    output = model.forward_components(left, right)
    repeated_target = target.repeat(3)
    final_rows = model.prss.quotients.state_for("branch").R
    angles = principal_angles(final_rows, true_rows)
    structured_mse = F.mse_loss(output["structured"], repeated_target)
    unrestricted_mse = F.mse_loss(output["unrestricted"], repeated_target)
    task_mse = F.mse_loss(output["task"], target)

  baseline_steps = max(300, args.steps // 2)
  baselines = {
    kind: train_task_baseline(kind, true_rows, baseline_steps, args.batch_size,
                              args.learning_rate)
    for kind in ("random", "pca", "direct")
  }
  result = {
    "seed": args.seed,
    "candidate_dim": args.candidate_dim,
    "host_dim": args.host_dim,
    "steps": args.steps,
    "final": {
      "task_mse": float(task_mse),
      "structured_response_mse": float(structured_mse),
      "unrestricted_response_mse": float(unrestricted_mse),
      "response_gap": float(structured_mse - unrestricted_mse),
      "principal_angles_radians": angles.cpu().tolist(),
      "max_principal_angle": float(angles.max()),
      "true_subspace_coverage": subspace_coverage(final_rows, true_rows),
    },
    "baselines": baselines,
    "spectral": model.prss.spectral_diagnostics(),
    "history": history,
    "note": "Synthetic identification test only; not natural-data evidence.",
  }
  if result["final"]["max_principal_angle"] > args.assert_max_angle:
    raise AssertionError("PRSS failed known-subspace recovery: max angle {:.4f} > {:.4f}".format(
      result["final"]["max_principal_angle"], args.assert_max_angle))
  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w") as handle:
    json.dump(result, handle, indent=2)
  return result


def main():
  args = parse_args()
  result = run(args)
  print(json.dumps(result["final"], indent=2))
  print("Wrote {}".format(Path(args.output).resolve()))


if __name__ == "__main__":
  main()
