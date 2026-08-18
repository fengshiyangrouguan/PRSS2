import csv

import numpy as np
import pytest

from experiments.preprocess_jodie import preprocess


def write_csv(path, rows):
  with open(path, "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["user_id", "item_id", "timestamp", "state_label", "features"])
    writer.writerows(rows)


def test_strict_preprocessor_preserves_per_interaction_features(tmp_path):
  source = tmp_path / "wikipedia.csv"
  write_csv(source, [
    [10, 7, 1.0, 0, 0.1, 0.2, 0.3],
    [20, 7, 2.0, 1, 1.1, 1.2, 1.3],
    [10, 9, 3.0, 0, 2.1, 2.2, 2.3],
  ])
  result = preprocess(source, "wikipedia", tmp_path / "processed", bipartite=True)
  edges = np.load(tmp_path / "processed" / "ml_wikipedia.npy")
  nodes = np.load(tmp_path / "processed" / "ml_wikipedia_node.npy")
  assert result["edge_feature_dim"] == 3
  assert edges.shape == (4, 3)
  assert np.allclose(edges[0], 0)
  assert np.allclose(edges[2], [1.1, 1.2, 1.3])
  assert nodes.shape == (5, 3)
  assert np.allclose(nodes, 0)


def test_preprocessor_rejects_ragged_features_and_time_reversal(tmp_path):
  ragged = tmp_path / "ragged.csv"
  write_csv(ragged, [[0, 0, 1, 0, 1, 2], [0, 1, 2, 0, 3]])
  with pytest.raises(ValueError, match="feature width"):
    preprocess(ragged, "bad", tmp_path / "bad", bipartite=True)
  reversed_time = tmp_path / "reversed.csv"
  write_csv(reversed_time, [[0, 0, 2, 0, 1], [0, 1, 1, 0, 2]])
  with pytest.raises(ValueError, match="timestamps decrease"):
    preprocess(reversed_time, "bad", tmp_path / "bad2", bipartite=True)

