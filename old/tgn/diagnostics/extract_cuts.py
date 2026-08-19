"""Replay a frozen official TGN and export real Wikipedia cut states and histories."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from diagnostics.common import (SPLITS, SplitWriter, build_frozen_tgn,
                                deterministic_negative_samples, effective_memory,
                                evenly_spaced_selection, infer_training_config,
                                load_checkpoint, load_tgn_data, make_neighbor_finders,
                                reset_memory_for_replay, resolve_device, save_json, set_seed)
from diagnostics.history import (STRUCTURE_NAMES, describe_sampled_history,
                                 exact_source_degree)


def parse_args():
  parser = argparse.ArgumentParser(
    description="Export frozen h, continuation C, and sampled pre-compression history X")
  parser.add_argument("--data-dir", required=True)
  parser.add_argument("--dataset", default="wikipedia")
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--config", required=True,
                      help="JSON manifest written by the patched official training script")
  parser.add_argument("--output", required=True)
  parser.add_argument("--max-hops", type=int, default=3)
  parser.add_argument("--history-degree", type=int, default=None,
                      help="Defaults to the TGN n_degree; changing it is recorded in the manifest")
  parser.add_argument("--batch-size", type=int, default=200,
                      help="Must match training/evaluation batching for exact memory semantics")
  parser.add_argument("--max-train", type=int, default=0,
                      help="Optional explicit diagnostic cap; 0 exports the full split")
  parser.add_argument("--max-val", type=int, default=0)
  parser.add_argument("--max-test", type=int, default=0)
  parser.add_argument("--seed", type=int, default=2027)
  parser.add_argument("--gpu", type=int, default=0)
  return parser.parse_args()


def sha256(path):
  digest = hashlib.sha256()
  with open(path, "rb") as handle:
    while True:
      block = handle.read(1024 * 1024)
      if not block:
        break
      digest.update(block)
  return digest.hexdigest()


def git_commit():
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
      text=True).strip()
  except Exception:
    return "unknown"


def main():
  args = parse_args()
  if args.max_hops < 1:
    raise ValueError("--max-hops must be positive")
  set_seed(args.seed)
  device = resolve_device(args.gpu)
  config = infer_training_config(args.config, args.data_dir, args.dataset)
  if bool(config.get("uniform", False)):
    raise ValueError("Uniform sampling is stochastic and unsupported for exact diagnostics")
  if args.batch_size != int(config.get("bs", args.batch_size)):
    raise ValueError("--batch-size must equal the training batch size ({}) to preserve memory "
                     "update semantics".format(config.get("bs")))
  history_degree = args.history_degree or int(config.get("n_degree", 10))

  node_features, edge_features, full_data, train_data, val_data, test_data, _, _ = \
    load_tgn_data(args.data_dir, args.dataset,
                  different_new_nodes=bool(config.get("different_new_nodes", False)))
  train_finder, full_finder = make_neighbor_finders(
    train_data, full_data, uniform=bool(config.get("uniform", False)))
  model = build_frozen_tgn(config, node_features, edge_features, train_finder, device)
  load_checkpoint(model, args.checkpoint, device)
  reset_memory_for_replay(model)

  split_data = {"train": train_data, "val": val_data, "test": test_data}
  caps = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
  selections = {name: evenly_spaced_selection(len(data.sources), caps[name])
                for name, data in split_data.items()}
  all_log_time = np.log1p(np.maximum(full_data.timestamps - full_data.timestamps.min(), 0.0))
  manifest = {
    "format_version": 1,
    "dataset": args.dataset,
    "checkpoint": str(Path(args.checkpoint).resolve()),
    "checkpoint_sha256": sha256(args.checkpoint),
    "official_tgn_commit": git_commit(),
    "use_memory": bool(config.get("use_memory", False)),
    "tgn_layers": int(config["n_layer"]),
    "tgn_degree": int(config.get("n_degree", 10)),
    "history_degree": int(history_degree),
    "max_hops": int(args.max_hops),
    "batch_size": int(args.batch_size),
    "seed": int(args.seed),
    "embedding_dim": int(node_features.shape[1]),
    "edge_feature_dim": int(edge_features.shape[1]),
    "structure_dim": len(STRUCTURE_NAMES),
    "edge_descriptor_dim": int(3 * edge_features.shape[1]),
    "state_descriptor_dim": int(3 * node_features.shape[1]),
    "structure_names": list(STRUCTURE_NAMES),
    "time_min": float(full_data.timestamps.min()),
    "log_time_mean": float(all_log_time.mean()),
    "log_time_std": float(max(all_log_time.std(), 1e-12)),
    "split_pairs": {name: int(len(selection)) for name, selection in selections.items()},
    "split_total_interactions": {name: int(len(split_data[name].sources)) for name in SPLITS},
    "selection": "full" if not any(caps.values()) else "deterministic_evenly_spaced_cap",
    "descriptor": {
      "structure_time": "12 fixed per-hop topology/log-time statistics",
      "edge": "per-hop mean, standard deviation, and most-recent raw edge feature",
      "lower_state": "per-hop mean frontier state, mean neighbor state, neighbor-state std",
      "recursive_cut_time": "original query timestamp, exactly as official GraphEmbedding",
    },
  }
  output = Path(args.output)
  output.mkdir(parents=True, exist_ok=True)
  save_json(manifest, output / "manifest.json")
  writers = {name: SplitWriter(output, name, len(selections[name]), manifest) for name in SPLITS}
  with torch.no_grad():
    for split_number, split in enumerate(SPLITS):
      data = split_data[split]
      finder = train_finder if split == "train" else full_finder
      destination_universe = np.unique(
        train_data.destinations if split == "train" else full_data.destinations)
      model.set_neighbor_finder(finder)
      model.eval()
      selection_mask = np.zeros(len(data.sources), dtype=bool)
      selection_mask[selections[split]] = True
      rng = np.random.RandomState(args.seed + 1009 * (split_number + 1))

      for start in range(0, len(data.sources), args.batch_size):
        end = min(start + args.batch_size, len(data.sources))
        sources = data.sources[start:end]
        destinations = data.destinations[start:end]
        timestamps = data.timestamps[start:end]
        edge_idxs = data.edge_idxs[start:end]
        negatives = deterministic_negative_samples(destination_universe, destinations, rng)
        chosen = np.flatnonzero(selection_mask[start:end])

        visible_memory = effective_memory(model)
        if chosen.size:
          if visible_memory is None:
            lower_states = model.node_raw_features.detach().cpu().numpy()
          else:
            lower_states = (model.node_raw_features + visible_memory).detach().cpu().numpy()
          structure, edge_desc, state_desc = describe_sampled_history(
            sources[chosen], timestamps[chosen], finder, edge_features, lower_states,
            max_hops=args.max_hops, n_neighbors=history_degree)

        source_h, positive_c, negative_c = model.compute_temporal_embeddings(
          sources, destinations, negatives, timestamps, edge_idxs,
          n_neighbors=int(config.get("n_degree", 10)))

        if chosen.size:
          selected_h = source_h[chosen]
          selected_pos = positive_c[chosen]
          selected_neg = negative_c[chosen]
          positive_logits = model.affinity_score(selected_h, selected_pos).view(-1)
          negative_logits = model.affinity_score(selected_h, selected_neg).view(-1)
          writers[split].append({
            "source_h": selected_h.cpu().numpy(),
            "positive_c": selected_pos.cpu().numpy(),
            "negative_c": selected_neg.cpu().numpy(),
            "base_logits": torch.stack([positive_logits, negative_logits], dim=1).cpu().numpy(),
            "x_structure": structure,
            "x_edge": edge_desc.astype(np.float16),
            "x_state": state_desc.astype(np.float16),
            "timestamps": timestamps[chosen],
            "source_ids": sources[chosen],
            "positive_ids": destinations[chosen],
            "negative_ids": negatives[chosen],
            "edge_idxs": edge_idxs[chosen],
            "source_degree": exact_source_degree(sources[chosen], timestamps[chosen], finder),
          })
      writers[split].close()

  save_json({"status": "complete", "pairs": manifest["split_pairs"]}, output / "_SUCCESS.json")
  print("Wrote frozen TGN diagnostics to {}".format(output.resolve()))


if __name__ == "__main__":
  main()
