"""Find near-collisions in frozen (h, C) with divergent rich-probe future responses."""

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from diagnostics.common import (CutDataset, Standardizer, load_probe, paired_rows,
                                predict_logits, resolve_device, save_json, sigmoid)


def parse_args():
  parser = argparse.ArgumentParser(description="Natural-data predictive collision analysis")
  parser.add_argument("--cuts", required=True)
  parser.add_argument("--probe", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--variant", default="all")
  parser.add_argument("--hop", type=int, default=3)
  parser.add_argument("--neighbors", type=int, default=20)
  parser.add_argument("--jobs", type=int, default=1,
                      help="Nearest-neighbor CPU workers (1 is portable; use -1 on AutoDL)")
  parser.add_argument("--max-rows", type=int, default=30000,
                      help="Deterministic test-row sample for NN cost; 0 uses every row")
  parser.add_argument("--tau", default="0.02,0.05,0.1,0.2")
  parser.add_argument("--seed", type=int, default=2027)
  parser.add_argument("--gpu", type=int, default=0)
  return parser.parse_args()


def main():
  args = parse_args()
  output = Path(args.output)
  output.mkdir(parents=True, exist_ok=True)
  device = resolve_device(args.gpu)
  dataset = CutDataset(args.cuts)
  model, scaler, metadata = load_probe(args.probe, device)
  if metadata.get("variant") != args.variant or int(metadata.get("hop", -1)) != args.hop:
    raise ValueError("Probe metadata does not match requested variant/hop")

  train_base, _, _ = dataset.base_rows("train")
  test_base, _, test_pair_ids = dataset.base_rows("test")
  test_history = dataset.history("test", args.hop, args.variant)
  test_x = paired_rows(test_base, test_history)
  rich_probabilities = sigmoid(predict_logits(model, test_x, scaler, device))
  embedding_dim = int(dataset.manifest["embedding_dim"])
  h_train = train_base[:, :embedding_dim]
  c_train = train_base[:, embedding_dim:]
  h_scaler = Standardizer.fit(h_train)
  c_scaler = Standardizer.fit(c_train)
  h = h_scaler.transform(test_base[:, :embedding_dim])
  c = c_scaler.transform(test_base[:, embedding_dim:])

  n_rows = len(test_base)
  rng = np.random.RandomState(args.seed)
  if args.max_rows > 0 and args.max_rows < n_rows:
    selected = np.sort(rng.choice(n_rows, size=args.max_rows, replace=False))
  else:
    selected = np.arange(n_rows)
  h = h[selected]
  c = c[selected]
  pair_ids = test_pair_ids[selected]
  probabilities = rich_probabilities[selected]
  joint = np.concatenate([h / np.sqrt(h.shape[1]), c / np.sqrt(c.shape[1])], axis=1)

  index = NearestNeighbors(n_neighbors=min(args.neighbors + 1, len(joint)),
                           metric="euclidean", n_jobs=args.jobs)
  index.fit(joint)
  _, neighbor_indexes = index.kneighbors(joint, return_distance=True)
  matches = np.full(len(joint), -1, dtype=np.int64)
  for row in range(len(joint)):
    candidates = neighbor_indexes[row]
    candidates = candidates[pair_ids[candidates] != pair_ids[row]]
    if len(candidates):
      matches[row] = candidates[0]
  valid = matches >= 0
  if not valid.any():
    raise RuntimeError("No different-history neighbor was found; increase --neighbors or rows")
  left = np.flatnonzero(valid)
  right = matches[valid]
  d_h = np.linalg.norm(h[left] - h[right], axis=1) / np.sqrt(h.shape[1])
  d_c = np.linalg.norm(c[left] - c[right], axis=1) / np.sqrt(c.shape[1])
  d_future = np.abs(probabilities[left] - probabilities[right])

  np.savez_compressed(output / "collision_pairs.npz", left_rows=selected[left],
                      right_rows=selected[right], d_h=d_h, d_c=d_c,
                      d_future=d_future, p_left=probabilities[left], p_right=probabilities[right])
  with open(output / "collision_pairs.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["left_test_row", "right_test_row", "d_h", "d_c", "d_future",
                     "p_left", "p_right"])
    for values in zip(selected[left], selected[right], d_h, d_c, d_future,
                      probabilities[left], probabilities[right]):
      writer.writerow(values)

  h_quantiles = [0.01, 0.02, 0.05, 0.10, 0.20]
  c_quantiles = [0.10, 0.25, 0.50]
  taus = [float(value) for value in args.tau.split(",") if value]
  curve = []
  for h_q in h_quantiles:
    h_epsilon = float(np.quantile(d_h, h_q))
    for c_q in c_quantiles:
      c_epsilon = float(np.quantile(d_c, c_q))
      mask = (d_h <= h_epsilon) & (d_c <= c_epsilon)
      for tau in taus:
        curve.append({
          "h_quantile": h_q,
          "c_quantile": c_q,
          "epsilon_h": h_epsilon,
          "epsilon_c": c_epsilon,
          "tau": tau,
          "matched_pairs": int(mask.sum()),
          "collision_rate": float(np.mean(d_future[mask] > tau)) if mask.any() else None,
          "mean_future_divergence": float(d_future[mask].mean()) if mask.any() else None,
        })
  summary = {
    "variant": args.variant,
    "hop": args.hop,
    "rows_available": n_rows,
    "rows_analyzed": int(len(selected)),
    "valid_matches": int(valid.sum()),
    "curve": curve,
  }
  save_json(summary, output / "collision_summary.json")

  fig, axis = plt.subplots(figsize=(7, 5))
  plot = axis.hexbin(d_h, d_future, gridsize=60, bins="log", mincnt=1, cmap="viridis")
  axis.set_xlabel("standardized frozen-state distance $d_h$")
  axis.set_ylabel("rich-probe future divergence")
  axis.set_title("Wikipedia predictive near-collisions")
  fig.colorbar(plot, ax=axis, label="log pair count")
  fig.tight_layout()
  fig.savefig(output / "collision_scatter.png", dpi=180)
  plt.close(fig)
  save_json({"status": "complete", "valid_matches": int(valid.sum())},
            output / "_SUCCESS.json")
  print("Collision results: {}".format((output / "collision_summary.json").resolve()))


if __name__ == "__main__":
  main()
