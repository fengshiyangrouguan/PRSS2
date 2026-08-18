"""Shared, leakage-aware utilities for the Wikipedia/TGN diagnostics.

The diagnostic dataset is pair-centric: each interaction stores one frozen source state/history
and two continuations (the observed destination and one sampled destination).  This prevents the
large history descriptor from being duplicated for the positive and negative rows and makes the
paired bootstrap unit explicit.
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from model.tgn import TGN
from utils.data_processing import compute_time_statistics, get_data
from utils.utils import get_neighbor_finder


SPLITS = ("train", "val", "test")


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def load_json(path):
  with open(path, "r") as handle:
    return json.load(handle)


def save_json(obj, path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as handle:
    json.dump(obj, handle, indent=2, sort_keys=True)


def load_tgn_data(data_dir, dataset="wikipedia", different_new_nodes=False):
  return get_data(dataset,
                  different_new_nodes_between_val_and_test=different_new_nodes,
                  randomize_features=False,
                  data_dir=str(data_dir))


def build_frozen_tgn(config, node_features, edge_features, neighbor_finder, device):
  """Build a TGN from the training manifest and load only learned parameters.

  The caller loads the checkpoint and then invokes :func:`reset_memory_for_replay`.  A plain TGN
  state_dict does not contain the Python ``memory.messages`` queues, so using its serialized
  memory as if it were a complete temporal snapshot would be incorrect.
  """
  model = TGN(
    neighbor_finder=neighbor_finder,
    node_features=node_features,
    edge_features=edge_features,
    device=device,
    n_layers=int(config["n_layer"]),
    n_heads=int(config.get("n_head", 2)),
    dropout=float(config.get("drop_out", 0.1)),
    use_memory=bool(config.get("use_memory", False)),
    memory_update_at_start=not bool(config.get("memory_update_at_end", False)),
    message_dimension=int(config.get("message_dim", 100)),
    memory_dimension=int(config.get("memory_dim", node_features.shape[1])),
    embedding_module_type=config.get("embedding_module", "graph_attention"),
    message_function=config.get("message_function", "identity"),
    mean_time_shift_src=float(config["mean_time_shift_src"]),
    std_time_shift_src=max(float(config["std_time_shift_src"]), 1e-12),
    mean_time_shift_dst=float(config["mean_time_shift_dst"]),
    std_time_shift_dst=max(float(config["std_time_shift_dst"]), 1e-12),
    n_neighbors=int(config.get("n_degree", 10)),
    aggregator_type=config.get("aggregator", "last"),
    memory_updater_type=config.get("memory_updater", "gru"),
    use_destination_embedding_in_message=bool(
      config.get("use_destination_embedding_in_message", False)),
    use_source_embedding_in_message=bool(config.get("use_source_embedding_in_message", False)),
    dyrep=bool(config.get("dyrep", False)),
  ).to(device)
  return model


def load_checkpoint(model, checkpoint, device):
  payload = torch.load(checkpoint, map_location=device)
  state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
  model.load_state_dict(state)
  model.eval()
  for parameter in model.parameters():
    parameter.requires_grad_(False)
  return model


def reset_memory_for_replay(model):
  if model.use_memory:
    model.memory.__init_memory__()


def effective_memory(model):
  """Return the exact memory tensor seen by GraphEmbedding for the next batch."""
  if not model.use_memory:
    return None
  if model.memory_update_at_start:
    memory, _ = model.get_updated_memory(list(range(model.n_nodes)), model.memory.messages)
    return memory
  return model.memory.get_memory(list(range(model.n_nodes)))


def deterministic_negative_samples(destination_universe, positives, rng):
  """Match official random-destination sampling while excluding the observed destination."""
  destination_universe = np.asarray(destination_universe)
  draw = destination_universe[rng.randint(0, len(destination_universe), size=len(positives))]
  collision = draw == positives
  while collision.any():
    draw[collision] = destination_universe[
      rng.randint(0, len(destination_universe), size=int(collision.sum()))]
    collision = draw == positives
  return draw.astype(np.int64, copy=False)


def evenly_spaced_selection(n, maximum):
  """Deterministic chronological coverage used only when an explicit cap is requested."""
  if maximum is None or maximum <= 0 or maximum >= n:
    return np.arange(n, dtype=np.int64)
  return np.unique(np.linspace(0, n - 1, num=maximum, dtype=np.int64))


def add_split_masks(full_data, train_data, val_data, test_data):
  """Return interaction-index to split maps for audit metadata."""
  result = {}
  for name, data in zip(SPLITS, (train_data, val_data, test_data)):
    for edge_idx in data.edge_idxs:
      if int(edge_idx) in result:
        raise ValueError("An interaction was assigned to multiple temporal splits")
      result[int(edge_idx)] = name
  return result


ARRAY_SPECS = {
  "source_h": (np.float32, "embedding_dim"),
  "positive_c": (np.float32, "embedding_dim"),
  "negative_c": (np.float32, "embedding_dim"),
  "base_logits": (np.float32, 2),
  "x_structure": (np.float32, ("max_hops", "structure_dim")),
  "x_edge": (np.float16, ("max_hops", "edge_descriptor_dim")),
  "x_state": (np.float16, ("max_hops", "state_descriptor_dim")),
  "timestamps": (np.float64, None),
  "source_ids": (np.int64, None),
  "positive_ids": (np.int64, None),
  "negative_ids": (np.int64, None),
  "edge_idxs": (np.int64, None),
  "source_degree": (np.int32, None),
}


def _resolve_shape(n, shape_spec, manifest):
  if shape_spec is None:
    return (n,)
  if isinstance(shape_spec, int):
    return (n, shape_spec)
  if isinstance(shape_spec, str):
    return (n, int(manifest[shape_spec]))
  return (n,) + tuple(int(manifest[item]) if isinstance(item, str) else int(item)
                      for item in shape_spec)


class SplitWriter:
  def __init__(self, root, split, n_rows, manifest):
    self.root = Path(root) / split
    self.root.mkdir(parents=True, exist_ok=True)
    self.n_rows = int(n_rows)
    self.position = 0
    self.arrays = {}
    for name, (dtype, shape_spec) in ARRAY_SPECS.items():
      shape = _resolve_shape(self.n_rows, shape_spec, manifest)
      self.arrays[name] = np.lib.format.open_memmap(
        self.root / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)

  def append(self, values):
    n = len(values["timestamps"])
    end = self.position + n
    if end > self.n_rows:
      raise RuntimeError("Attempted to write beyond declared split length")
    for name, array in self.arrays.items():
      array[self.position:end] = values[name]
    self.position = end

  def close(self):
    if self.position != self.n_rows:
      raise RuntimeError("Split {} wrote {} rows, expected {}".format(
        self.root.name, self.position, self.n_rows))
    for array in self.arrays.values():
      array.flush()
    self.arrays.clear()


class CutDataset:
  def __init__(self, root, mmap_mode="r"):
    self.root = Path(root)
    self.manifest = load_json(self.root / "manifest.json")
    self.splits = {}
    for split in SPLITS:
      split_dir = self.root / split
      self.splits[split] = {
        name: np.load(split_dir / (name + ".npy"), mmap_mode=mmap_mode)
        for name in ARRAY_SPECS
      }

  def pair_count(self, split):
    return len(self.splits[split]["timestamps"])

  def history(self, split, hop, variant="all"):
    if hop < 1 or hop > int(self.manifest["max_hops"]):
      raise ValueError("hop must be in [1, max_hops]")
    arrays = self.splits[split]
    pieces = [np.asarray(arrays["x_structure"][:, :hop]).reshape(len(arrays["timestamps"]), -1)]
    if variant in ("structure_edge", "all"):
      pieces.append(np.asarray(arrays["x_edge"][:, :hop], dtype=np.float32).reshape(
        len(arrays["timestamps"]), -1))
    if variant == "all":
      pieces.append(np.asarray(arrays["x_state"][:, :hop], dtype=np.float32).reshape(
        len(arrays["timestamps"]), -1))
    if variant not in ("structure", "structure_edge", "all"):
      raise ValueError("Unknown history variant: {}".format(variant))
    return np.concatenate(pieces, axis=1)

  def base_rows(self, split):
    """Expand pair-centric data into positive/negative supervised rows."""
    arrays = self.splits[split]
    h = np.asarray(arrays["source_h"], dtype=np.float32)
    timestamp = np.asarray(arrays["timestamps"], dtype=np.float64)
    time_feature = normalized_log_time(timestamp, self.manifest)
    positive = np.concatenate([h, np.asarray(arrays["positive_c"], dtype=np.float32),
                               time_feature[:, None]], axis=1)
    negative = np.concatenate([h, np.asarray(arrays["negative_c"], dtype=np.float32),
                               time_feature[:, None]], axis=1)
    base = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate([np.ones(len(h), dtype=np.float32),
                             np.zeros(len(h), dtype=np.float32)])
    pair_ids = np.concatenate([np.arange(len(h)), np.arange(len(h))])
    return base, labels, pair_ids


def normalized_log_time(timestamps, manifest):
  values = np.log1p(np.maximum(np.asarray(timestamps) - float(manifest["time_min"]), 0.0))
  return ((values - float(manifest["log_time_mean"])) /
          max(float(manifest["log_time_std"]), 1e-12)).astype(np.float32)


def paired_rows(base, history):
  return np.concatenate([base, np.concatenate([history, history], axis=0)], axis=1)


def iter_minibatches(n, batch_size, rng=None, shuffle=True):
  indexes = np.arange(n)
  if shuffle:
    if rng is None:
      np.random.shuffle(indexes)
    else:
      rng.shuffle(indexes)
  for start in range(0, n, batch_size):
    yield indexes[start:start + batch_size]


class Standardizer:
  def __init__(self, mean, scale):
    self.mean = np.asarray(mean, dtype=np.float32)
    self.scale = np.asarray(scale, dtype=np.float32)

  @classmethod
  def fit(cls, array):
    # ``dtype`` controls accumulation without materializing a second full float64 matrix.
    array = np.asarray(array)
    mean = np.mean(array, axis=0, dtype=np.float64)
    scale = np.std(array, axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    return cls(mean, scale)

  def transform(self, array):
    return (np.asarray(array, dtype=np.float32) - self.mean) / self.scale

  def state_dict(self):
    return {"mean": self.mean, "scale": self.scale}

  @classmethod
  def from_state_dict(cls, state):
    return cls(state["mean"], state["scale"])


class ShallowProbe(torch.nn.Module):
  """The same one-hidden-layer architecture is used for every probe/control."""
  def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
    super().__init__()
    self.network = torch.nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Dropout(dropout),
      torch.nn.Linear(hidden_dim, 1),
    )

  def forward(self, values):
    return self.network(values).squeeze(-1)


def predict_logits(model, array, standardizer, device, batch_size=4096):
  model.eval()
  output = np.empty(len(array), dtype=np.float32)
  with torch.no_grad():
    for start in range(0, len(array), batch_size):
      end = min(start + batch_size, len(array))
      values = standardizer.transform(array[start:end])
      tensor = torch.from_numpy(values).to(device)
      output[start:end] = model(tensor).cpu().numpy()
  return output


def binary_nll_from_logits(logits, labels):
  logits = np.asarray(logits, dtype=np.float64)
  labels = np.asarray(labels, dtype=np.float64)
  return np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))


def sigmoid(values):
  values = np.asarray(values, dtype=np.float64)
  output = np.empty_like(values)
  positive = values >= 0
  output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
  exp_values = np.exp(values[~positive])
  output[~positive] = exp_values / (1.0 + exp_values)
  return output


def train_probe(train_x, train_y, val_x, val_y, device, hidden_dim=128, dropout=0.1,
                weight_decay=1e-4, learning_rate=1e-3, batch_size=1024, max_epochs=100,
                patience=10, seed=0):
  set_seed(seed)
  standardizer = Standardizer.fit(train_x)
  model = ShallowProbe(train_x.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)
  criterion = torch.nn.BCEWithLogitsLoss()
  rng = np.random.RandomState(seed)
  best_state = None
  best_val = math.inf
  stale = 0
  for epoch in range(max_epochs):
    model.train()
    for indexes in iter_minibatches(len(train_x), batch_size, rng=rng, shuffle=True):
      values = torch.from_numpy(standardizer.transform(train_x[indexes])).to(device)
      labels = torch.from_numpy(np.asarray(train_y[indexes], dtype=np.float32)).to(device)
      optimizer.zero_grad()
      loss = criterion(model(values), labels)
      loss.backward()
      optimizer.step()
    val_logits = predict_logits(model, val_x, standardizer, device, batch_size=batch_size * 2)
    val_loss = float(binary_nll_from_logits(val_logits, val_y).mean())
    if val_loss < best_val - 1e-6:
      best_val = val_loss
      best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
      stale = 0
    else:
      stale += 1
      if stale >= patience:
        break
  model.load_state_dict(best_state)
  return model, standardizer, {"val_nll": best_val, "epochs": epoch + 1}


def save_probe(path, model, standardizer, metadata):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save({
    "model_state": model.state_dict(),
    "input_dim": int(model.network[0].in_features),
    "hidden_dim": int(model.network[0].out_features),
    "standardizer": standardizer.state_dict(),
    "metadata": metadata,
  }, path)


def load_probe(path, device):
  payload = torch.load(path, map_location=device)
  model = ShallowProbe(payload["input_dim"], payload["hidden_dim"],
                       dropout=float(payload.get("metadata", {}).get("dropout", 0.0))).to(device)
  model.load_state_dict(payload["model_state"])
  model.eval()
  return model, Standardizer.from_state_dict(payload["standardizer"]), payload.get("metadata", {})


def resolve_device(gpu=0):
  return torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")


def infer_training_config(config_path, data_dir, dataset="wikipedia"):
  """Load a manifest; fill time statistics for legacy official checkpoints if absent."""
  config = load_json(config_path)
  required = ("mean_time_shift_src", "std_time_shift_src",
              "mean_time_shift_dst", "std_time_shift_dst")
  if not all(key in config for key in required):
    _, _, full_data, _, _, _, _, _ = load_tgn_data(data_dir, dataset)
    values = compute_time_statistics(full_data.sources, full_data.destinations,
                                     full_data.timestamps)
    for key, value in zip(required, values):
      config[key] = float(value)
  return config


def make_neighbor_finders(train_data, full_data, uniform=False):
  if uniform:
    raise ValueError("Diagnostics require deterministic recent-neighbor sampling; do not use --uniform")
  return get_neighbor_finder(train_data, False), get_neighbor_finder(full_data, False)
