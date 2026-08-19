"""Fail-fast verification on the real Wikipedia files (never a synthetic experiment)."""

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from diagnostics.common import (build_frozen_tgn, load_tgn_data, make_neighbor_finders,
                                save_json, set_seed)
from diagnostics.history import describe_sampled_history


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw-csv", required=True)
  parser.add_argument("--data-dir", required=True)
  parser.add_argument("--output", required=True)
  return parser.parse_args()


def main():
  args = parse_args()
  with open(args.raw_csv, newline="") as handle:
    reader = csv.reader(handle)
    header = next(reader)
    first_row = next(reader)
  if header[:4] != ["user_id", "item_id", "timestamp", "state_label"]:
    raise ValueError("Unexpected raw Wikipedia header")
  if len(first_row) - 4 != 172:
    raise ValueError("Expected 172 edge features, found {}".format(len(first_row) - 4))

  node_features, edge_features, full_data, train_data, _, _, _, _ = load_tgn_data(args.data_dir)
  checks = {
    "raw_feature_dim": len(first_row) - 4,
    "interactions": int(full_data.n_interactions),
    "nodes_including_padding": int(node_features.shape[0]),
    "edge_feature_shape": list(edge_features.shape),
    "node_feature_shape": list(node_features.shape),
    "timestamps_monotone": bool(np.all(np.diff(full_data.timestamps) >= 0)),
    "padding_edge_zero": bool(np.all(edge_features[0] == 0)),
    "padding_node_zero": bool(np.all(node_features[0] == 0)),
  }
  expected = {"interactions": 157474, "nodes_including_padding": 9228}
  for key, value in expected.items():
    if checks[key] != value:
      raise ValueError("{}: expected {}, got {}".format(key, value, checks[key]))
  if not checks["timestamps_monotone"]:
    raise ValueError("Wikipedia interactions are not chronological")

  set_seed(123)
  train_finder, _ = make_neighbor_finders(train_data, full_data)
  config = {
    "n_layer": 2, "n_head": 2, "drop_out": 0.1, "use_memory": False,
    "mean_time_shift_src": 0, "std_time_shift_src": 1,
    "mean_time_shift_dst": 0, "std_time_shift_dst": 1, "n_degree": 3,
  }
  model = build_frozen_tgn(config, node_features, edge_features, train_finder,
                           torch.device("cpu")).eval()
  indexes = np.arange(1000, 1004)
  sources = train_data.sources[indexes]
  destinations = train_data.destinations[indexes]
  negatives = np.roll(destinations, 1)
  timestamps = train_data.timestamps[indexes]
  edge_idxs = train_data.edge_idxs[indexes]
  with torch.no_grad():
    reference = model.compute_temporal_embeddings(
      sources, destinations, negatives, timestamps, edge_idxs, 3)[0]
    traces = []
    model.set_diagnostic_observer(traces.append)
    observed = model.compute_temporal_embeddings(
      sources, destinations, negatives, timestamps, edge_idxs, 3)[0]
  checks["observer_bitwise_equal"] = bool(torch.equal(reference, observed))
  checks["observer_max_abs_difference"] = float((reference - observed).abs().max())
  checks["observer_trace_calls"] = len(traces)
  if not checks["observer_bitwise_equal"]:
    raise RuntimeError("Diagnostic observer changed the official TGN forward output")

  structure, edge_desc, state_desc = describe_sampled_history(
    sources, timestamps, train_finder, edge_features, node_features, max_hops=3, n_neighbors=3)
  checks["descriptor_shapes"] = [list(structure.shape), list(edge_desc.shape),
                                  list(state_desc.shape)]
  checks["descriptors_finite"] = bool(
    np.isfinite(structure).all() and np.isfinite(edge_desc).all() and
    np.isfinite(state_desc).all())
  if not checks["descriptors_finite"]:
    raise RuntimeError("Non-finite history descriptor")
  save_json(checks, args.output)
  print("Real Wikipedia verification passed: {}".format(Path(args.output).resolve()))


if __name__ == "__main__":
  main()
