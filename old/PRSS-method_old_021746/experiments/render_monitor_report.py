"""Render a compact static PRSS mechanism dashboard from monitor JSONL artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-dir", required=True)
  parser.add_argument("--output", default=None)
  return parser.parse_args()


def read_jsonl(path):
  if not path.exists():
    return []
  with open(path, encoding="utf-8") as handle:
    return [json.loads(line) for line in handle if line.strip()]


def plot_series(axis, x, series, title, log=False):
  for label, values in series.items():
    if values and any(value is not None for value in values):
      axis.plot(x, values, label=label, linewidth=1.4)
  axis.set_title(title)
  axis.grid(alpha=0.25)
  if log:
    axis.set_yscale("log")
  if axis.lines:
    axis.legend(fontsize=7)


def render(run_dir, output=None):
  run_dir = Path(run_dir)
  monitor = run_dir / "monitor"
  steps = read_jsonl(monitor / "step_metrics.jsonl")
  epochs = read_jsonl(monitor / "epoch_metrics.jsonl")
  alerts = read_jsonl(monitor / "alerts.jsonl")
  if not steps:
    raise ValueError("No monitor step metrics in {}".format(monitor))
  output = Path(output) if output else monitor / "mechanism_dashboard.png"
  x = [item["step"] for item in steps]
  interfaces = sorted({tau for item in steps for tau in item["interfaces"]})
  figure, axes = plt.subplots(3, 3, figsize=(18, 13), constrained_layout=True)

  plot_series(axes[0, 0], x, {
    name: [item["losses"].get(name) for item in steps]
    for name in ("task", "response", "unrestricted_response", "spectral")
  }, "Training losses", log=True)
  plot_series(axes[0, 1], x, {
    name: [item["response"].get(name) for item in steps]
    for name in ("structured_accuracy", "unrestricted_accuracy")
  }, "Future-response accuracy")
  plot_series(axes[0, 2], x, {
    "tail/{}".format(tau): [
      max(1.0 - item["interfaces"][tau].get(
        "batch_predictive_energy_captured_by_current_R", 0.0), 1e-12)
      for item in steps]
    for tau in interfaces
  }, "Batch predictive tail outside deployed R", log=True)

  geometry = {}
  for tau in interfaces:
    geometry["orth/{}".format(tau)] = [
      item["interfaces"][tau]["projection_orthogonality_error_relative"] for item in steps]
    geometry["gram_sym/{}".format(tau)] = [
      item["interfaces"][tau]["gram_symmetry_error"] for item in steps]
  plot_series(axes[1, 0], x, geometry, "Invariant residuals", log=True)
  stability = {}
  for tau in interfaces:
    stability["projector/{}".format(tau)] = [
      item["interfaces"][tau]["projector_distance_last_update"] for item in steps]
    stability["max_angle/{}".format(tau)] = [
      item["interfaces"][tau]["principal_angle_max_last_update"] for item in steps]
  plot_series(axes[1, 1], x, stability, "Subspace stability")
  representation = {}
  for tau in interfaces:
    representation["reader_norm/{}".format(tau)] = [
      item["interfaces"][tau].get("reader_frobenius_norm", {}).get("mean")
      for item in steps]
    representation["candidate_std/{}".format(tau)] = [
      item["interfaces"][tau].get("candidate_state", {}).get("coordinate_std_mean")
      for item in steps]
  plot_series(axes[1, 2], x, representation, "Reader/state collapse sentinels", log=True)

  gradient_names = sorted({name for item in steps for name in item["gradients"]
                           if name != "parameter_counts_with_gradient"})
  plot_series(axes[2, 0], x, {
    name: [item["gradients"].get(name) for item in steps] for name in gradient_names
  }, "Gradient flow by mechanism", log=True)
  epoch_x = [item["epoch"] for item in epochs]
  plot_series(axes[2, 1], epoch_x, {
    name: [item["validation"].get(name) for item in epochs] for name in ("ap", "auc")
  }, "Validation task performance")
  severities = ("warning", "error")
  counts = [sum(alert["severity"] == severity for alert in alerts) for severity in severities]
  axes[2, 2].bar(severities, counts, color=["#f2b134", "#d64545"])
  axes[2, 2].set_title("Monitor alerts")
  for index, value in enumerate(counts):
    axes[2, 2].text(index, value, str(value), ha="center", va="bottom")

  figure.suptitle("PRSS mechanism dashboard: {}".format(run_dir.name), fontsize=16)
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=160)
  plt.close(figure)
  return output


def main():
  args = parse_args()
  print(render(args.run_dir, args.output).resolve())


if __name__ == "__main__":
  main()
