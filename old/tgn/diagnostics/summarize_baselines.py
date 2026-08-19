"""Collect official AP/AUC supporting evidence across TGN depth and memory settings."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--artifacts-root", required=True)
  parser.add_argument("--output", required=True)
  return parser.parse_args()


def main():
  args = parse_args()
  root = Path(args.artifacts_root)
  output = Path(args.output)
  output.mkdir(parents=True, exist_ok=True)
  rows = []
  for config_path in root.glob("**/saved_models/*.json"):
    with open(config_path) as handle:
      config = json.load(handle)
    result_path = config_path.parents[1] / "results" / (config["prefix"] + ".pkl")
    if not result_path.exists():
      continue
    with open(result_path, "rb") as handle:
      result = pickle.load(handle)
    rows.append({
      "memory": "memory" if config.get("use_memory") else "no_memory",
      "layers": int(config["n_layer"]),
      "seed": int(config.get("seed", 0)),
      "test_ap": float(result["test_ap"]),
      "test_auc": float(result.get("test_auc", np.nan)),
      "new_node_test_ap": float(result["new_node_test_ap"]),
      "new_node_test_auc": float(result.get("new_node_test_auc", np.nan)),
      "config": str(config_path.resolve()),
    })
  if not rows:
    raise RuntimeError("No completed baseline results found under {}".format(root))
  rows.sort(key=lambda row: (row["memory"], row["layers"], row["seed"]))
  with open(output / "tgn_depth_runs.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

  summary = []
  for memory in ("memory", "no_memory"):
    for layer in sorted(set(row["layers"] for row in rows)):
      selected = [row for row in rows if row["memory"] == memory and row["layers"] == layer]
      if not selected:
        continue
      entry = {"memory": memory, "layers": layer, "runs": len(selected)}
      for metric in ("test_ap", "test_auc", "new_node_test_ap", "new_node_test_auc"):
        values = np.asarray([row[metric] for row in selected], dtype=np.float64)
        entry[metric + "_mean"] = float(np.nanmean(values))
        entry[metric + "_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
      summary.append(entry)
  with open(output / "tgn_depth_summary.json", "w") as handle:
    json.dump(summary, handle, indent=2)

  fig, axes = plt.subplots(1, 2, figsize=(10, 4))
  for memory, marker in (("memory", "o"), ("no_memory", "s")):
    selected = [row for row in summary if row["memory"] == memory]
    if not selected:
      continue
    layers = [row["layers"] for row in selected]
    axes[0].errorbar(layers, [row["test_ap_mean"] for row in selected],
                     yerr=[row["test_ap_std"] for row in selected], marker=marker, label=memory)
    axes[1].errorbar(layers, [row["test_auc_mean"] for row in selected],
                     yerr=[row["test_auc_std"] for row in selected], marker=marker, label=memory)
  for axis, metric in zip(axes, ("AP", "AUC")):
    axis.set_xlabel("official TGN graph-attention layers")
    axis.set_ylabel(metric)
    axis.set_xticks(sorted(set(row["layers"] for row in rows)))
    axis.legend()
  fig.tight_layout()
  fig.savefig(output / "tgn_depth.png", dpi=180)
  plt.close(fig)


if __name__ == "__main__":
  main()
