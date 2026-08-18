"""Strict, memory-bounded JODIE CSV -> official TGN files conversion."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument("--data", default="wikipedia")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--bipartite", action="store_true")
  parser.add_argument("--allow-unsorted", action="store_true",
                      help="Not recommended: retain rows even when timestamps decrease")
  return parser.parse_args()


def _parse_prefix(row, line_number):
  if len(row) < 5:
    raise ValueError("line {} has {} columns; expected 4 + edge features".format(
      line_number, len(row)))
  try:
    user = int(row[0])
    item = int(row[1])
    timestamp = float(row[2])
    label = float(row[3])
  except ValueError as error:
    raise ValueError("invalid interaction prefix at line {}".format(line_number)) from error
  if not math.isfinite(timestamp) or not math.isfinite(label):
    raise ValueError("non-finite timestamp/label at line {}".format(line_number))
  return user, item, timestamp, label


def inspect_csv(input_path, allow_unsorted=False):
  users, items = set(), set()
  rows = 0
  feature_dim = None
  previous_timestamp = -float("inf")
  with open(input_path, "r", encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle)
    try:
      header = next(reader)
    except StopIteration as error:
      raise ValueError("empty CSV") from error
    if len(header) < 5:
      raise ValueError("header must contain 4 interaction columns and edge features")
    for line_number, row in enumerate(reader, start=2):
      user, item, timestamp, _ = _parse_prefix(row, line_number)
      width = len(row) - 4
      if feature_dim is None:
        feature_dim = width
      if width != feature_dim:
        raise ValueError("line {} feature width {}; expected {}".format(
          line_number, width, feature_dim))
      try:
        features = np.asarray(row[4:], dtype=np.float32)
      except ValueError as error:
        raise ValueError("invalid edge feature at line {}".format(line_number)) from error
      if not np.isfinite(features).all():
        raise ValueError("non-finite edge feature at line {}".format(line_number))
      if not allow_unsorted and timestamp < previous_timestamp:
        raise ValueError(
          "timestamps decrease at line {}; causal order must be explicit".format(line_number))
      previous_timestamp = timestamp
      users.add(user)
      items.add(item)
      rows += 1
  if rows == 0:
    raise ValueError("CSV has a header but no interactions")
  return {
    "header": header,
    "interactions": rows,
    "feature_dim": feature_dim,
    "users": users,
    "items": items,
    "timestamps_sorted": not allow_unsorted,
  }


def preprocess(input_path, data_name, output_dir, bipartite=True, allow_unsorted=False):
  input_path = Path(input_path).resolve()
  output_dir = Path(output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  inspection = inspect_csv(input_path, allow_unsorted=allow_unsorted)

  if bipartite:
    user_map = {value: index + 1 for index, value in enumerate(sorted(inspection["users"]))}
    item_offset = len(user_map) + 1
    item_map = {value: item_offset + index
                for index, value in enumerate(sorted(inspection["items"]))}
    maximum_node = len(user_map) + len(item_map)
  else:
    nodes = sorted(inspection["users"] | inspection["items"])
    node_map = {value: index + 1 for index, value in enumerate(nodes)}
    user_map = item_map = node_map
    maximum_node = len(node_map)

  edge_path = output_dir / "ml_{}.npy".format(data_name)
  node_path = output_dir / "ml_{}_node.npy".format(data_name)
  frame_path = output_dir / "ml_{}.csv".format(data_name)
  edge_features = np.lib.format.open_memmap(
    edge_path, mode="w+", dtype=np.float32,
    shape=(inspection["interactions"] + 1, inspection["feature_dim"]))
  edge_features[0] = 0
  node_features = np.lib.format.open_memmap(
    node_path, mode="w+", dtype=np.float32,
    shape=(maximum_node + 1, inspection["feature_dim"]))
  node_features[:] = 0

  with open(input_path, "r", encoding="utf-8", newline="") as source, \
       open(frame_path, "w", encoding="utf-8", newline="") as destination:
    reader = csv.reader(source)
    next(reader)
    writer = csv.writer(destination)
    # Match pandas.DataFrame.to_csv consumed by the official TGN loader.
    writer.writerow(["", "u", "i", "ts", "label", "idx"])
    for zero_index, row in enumerate(reader):
      line_number = zero_index + 2
      user, item, timestamp, label = _parse_prefix(row, line_number)
      edge_index = zero_index + 1
      writer.writerow([
        zero_index, user_map[user], item_map[item], timestamp, label, edge_index])
      edge_features[edge_index] = np.asarray(row[4:], dtype=np.float32)
  edge_features.flush()
  node_features.flush()
  del edge_features, node_features

  manifest = {
    "dataset": data_name,
    "input": str(input_path),
    "interactions": inspection["interactions"],
    "users": len(inspection["users"]),
    "items": len(inspection["items"]),
    "nodes_after_reindex": maximum_node,
    "edge_feature_dim": inspection["feature_dim"],
    "bipartite": bool(bipartite),
    "timestamps_sorted": inspection["timestamps_sorted"],
    "schema": "user_id,item_id,timestamp,state_label,edge_features...",
  }
  with open(output_dir / "ml_{}_manifest.json".format(data_name), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
  return manifest


def main():
  args = parse_args()
  result = preprocess(args.input, args.data, args.output_dir, args.bipartite,
                      args.allow_unsorted)
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()

