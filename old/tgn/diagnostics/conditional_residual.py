"""Conditional residual tests for Y independent of H given frozen TGN h and context C."""

import argparse
import csv
import gc
import sys
from pathlib import Path

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from diagnostics.common import (CutDataset, binary_nll_from_logits, load_probe,
                                paired_rows, predict_logits, resolve_device, save_json,
                                save_probe, sigmoid, train_probe)


def parse_args():
  parser = argparse.ArgumentParser(description="Frozen-TGN conditional residual probes")
  parser.add_argument("--cuts", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--hops", default="1,2,3")
  parser.add_argument("--variants", default="structure,structure_edge,all",
                      help="Comma list from structure, structure_edge, all")
  parser.add_argument("--hidden-dim", type=int, default=128)
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument("--weight-decays", default="0,1e-5,1e-4,1e-3")
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--max-epochs", type=int, default=100)
  parser.add_argument("--patience", type=int, default=10)
  parser.add_argument("--bootstrap", type=int, default=5000)
  parser.add_argument("--time-bins", type=int, default=10)
  parser.add_argument("--degree-bins", type=int, default=5)
  parser.add_argument("--seed", type=int, default=2027)
  parser.add_argument("--gpu", type=int, default=0)
  return parser.parse_args()


def quantile_bins(values, n_bins):
  if n_bins <= 1 or len(values) == 0:
    return np.zeros(len(values), dtype=np.int64)
  edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
  if len(edges) <= 2:
    return np.zeros(len(values), dtype=np.int64)
  return np.searchsorted(edges[1:-1], values, side="right").astype(np.int64)


def local_history_permutation(arrays, time_bins, degree_bins, seed):
  """Shuffle pair-level history within nearby time/activity strata.

  Wikipedia is bipartite, so every diagnostic root has the same node type (user).  Source
  pre-query degree is used as the within-type activity stratum.  Groups are cyclically shifted,
  which avoids fixed points whenever a stratum has at least two members.
  """
  rng = np.random.RandomState(seed)
  time_group = quantile_bins(np.asarray(arrays["timestamps"]), time_bins)
  degree_group = quantile_bins(np.log1p(np.asarray(arrays["source_degree"])), degree_bins)
  groups = time_group * max(degree_bins, 1) + degree_group
  permutation = np.arange(len(groups))
  for group in np.unique(groups):
    indexes = np.flatnonzero(groups == group)
    if len(indexes) > 1:
      shuffled = indexes.copy()
      rng.shuffle(shuffled)
      shift = int(rng.randint(1, len(indexes)))
      # Assign on the shuffled ordering itself so a non-zero cyclic shift is a derangement.
      permutation[shuffled] = np.roll(shuffled, shift)
  return permutation


def select_regularization(train_x, train_y, val_x, val_y, weight_decays, args, device,
                          seed_offset=0):
  candidates = []
  for index, weight_decay in enumerate(weight_decays):
    model, standardizer, info = train_probe(
      train_x, train_y, val_x, val_y, device,
      hidden_dim=args.hidden_dim, dropout=args.dropout,
      weight_decay=weight_decay, learning_rate=args.learning_rate,
      batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience,
      # Hold initialization/minibatch order fixed so validation compares regularization, not luck.
      seed=args.seed + seed_offset)
    candidates.append((info["val_nll"], weight_decay, model, standardizer, info))
  candidates.sort(key=lambda item: item[0])
  return candidates[0][1:]


def paired_bootstrap(loss_a, loss_b, n_pairs, n_bootstrap, seed):
  """Bootstrap interaction pairs, never individual positive/negative rows."""
  pair_delta = 0.5 * ((loss_a[:n_pairs] - loss_b[:n_pairs]) +
                      (loss_a[n_pairs:] - loss_b[n_pairs:]))
  rng = np.random.RandomState(seed)
  estimates = np.empty(n_bootstrap, dtype=np.float64)
  for start in range(0, n_bootstrap, 256):
    count = min(256, n_bootstrap - start)
    indexes = rng.randint(0, n_pairs, size=(count, n_pairs))
    estimates[start:start + count] = pair_delta[indexes].mean(axis=1)
  return {
    "mean": float(pair_delta.mean()),
    "ci_low": float(np.quantile(estimates, 0.025)),
    "ci_high": float(np.quantile(estimates, 0.975)),
    "bootstrap_probability_positive": float(np.mean(estimates > 0)),
  }


def metrics(logits, labels):
  probabilities = sigmoid(logits)
  losses = binary_nll_from_logits(logits, labels)
  return {
    "nll": float(losses.mean()),
    "ap": float(average_precision_score(labels, probabilities)),
    "auc": float(roc_auc_score(labels, probabilities)),
  }, losses


def main():
  args = parse_args()
  output = Path(args.output)
  output.mkdir(parents=True, exist_ok=True)
  device = resolve_device(args.gpu)
  dataset = CutDataset(args.cuts)
  hops = [int(value) for value in args.hops.split(",") if value]
  variants = [value.strip() for value in args.variants.split(",") if value.strip()]
  weight_decays = [float(value) for value in args.weight_decays.split(",") if value]
  for hop in hops:
    if hop > int(dataset.manifest["max_hops"]):
      raise ValueError("Requested hop {} exceeds exported max_hops".format(hop))

  train_base, train_y, _ = dataset.base_rows("train")
  val_base, val_y, _ = dataset.base_rows("val")
  test_base, test_y, test_pair_ids = dataset.base_rows("test")
  n_test_pairs = dataset.pair_count("test")

  base_wd, base_model, base_scaler, base_info = select_regularization(
    train_base, train_y, val_base, val_y, weight_decays, args, device)
  base_test_logits = predict_logits(base_model, test_base, base_scaler, device,
                                    batch_size=args.batch_size * 2)
  base_metrics, base_losses = metrics(base_test_logits, test_y)
  save_probe(output / "probe_base.pt", base_model, base_scaler, {
    "kind": "base", "weight_decay": base_wd, "dropout": args.dropout,
    "selection": base_info,
  })

  exported_tgn_logits = np.concatenate([
    np.asarray(dataset.splits["test"]["base_logits"][:, 0]),
    np.asarray(dataset.splits["test"]["base_logits"][:, 1]),
  ])
  tgn_metrics, _ = metrics(exported_tgn_logits, test_y)
  results = {
    "cuts_manifest": dataset.manifest,
    "base_probe": {**base_metrics, "weight_decay": base_wd, **base_info},
    "frozen_tgn": tgn_metrics,
    "tests": [],
  }

  permutations = {
    split: local_history_permutation(dataset.splits[split], args.time_bins,
                                     args.degree_bins, args.seed + 7919 * index)
    for index, split in enumerate(("train", "val", "test"))
  }
  results["permutation_audit"] = {
    split: {
      "pairs": int(len(permutation)),
      "moved_pairs": int(np.sum(permutation != np.arange(len(permutation)))),
      "moved_fraction": float(np.mean(permutation != np.arange(len(permutation)))),
      "time_bins": args.time_bins,
      "degree_bins_within_user_type": args.degree_bins,
    }
    for split, permutation in permutations.items()
  }

  for variant_number, variant in enumerate(variants):
    for hop in hops:
      histories = {split: dataset.history(split, hop, variant) for split in ("train", "val", "test")}
      train_x = paired_rows(train_base, histories["train"])
      val_x = paired_rows(val_base, histories["val"])
      test_x = paired_rows(test_base, histories["test"])
      rich_wd, rich_model, rich_scaler, rich_info = select_regularization(
        train_x, train_y, val_x, val_y, weight_decays, args, device,
        seed_offset=1000 + 100 * variant_number + hop * 10)
      rich_test_logits = predict_logits(rich_model, test_x, rich_scaler, device,
                                        batch_size=args.batch_size * 2)
      rich_metrics, rich_losses = metrics(rich_test_logits, test_y)

      shuffled_history = {
        split: histories[split][permutations[split]] for split in ("train", "val", "test")
      }
      shuffle_train_x = paired_rows(train_base, shuffled_history["train"])
      shuffle_val_x = paired_rows(val_base, shuffled_history["val"])
      shuffle_test_x = paired_rows(test_base, shuffled_history["test"])
      # Same input width, architecture, and chosen regularization as the corresponding rich probe.
      shuffle_model, shuffle_scaler, shuffle_info = train_probe(
        shuffle_train_x, train_y, shuffle_val_x, val_y, device,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        weight_decay=rich_wd, learning_rate=args.learning_rate,
        batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience,
        seed=args.seed + 5000 + 100 * variant_number + hop)
      shuffle_test_logits = predict_logits(shuffle_model, shuffle_test_x, shuffle_scaler, device,
                                           batch_size=args.batch_size * 2)
      shuffle_metrics, shuffle_losses = metrics(shuffle_test_logits, test_y)

      base_vs_rich = paired_bootstrap(base_losses, rich_losses, n_test_pairs,
                                      args.bootstrap, args.seed + hop)
      shuffle_vs_rich = paired_bootstrap(shuffle_losses, rich_losses, n_test_pairs,
                                         args.bootstrap, args.seed + 100 + hop)
      record = {
        "variant": variant,
        "hop": hop,
        "history_dim": int(histories["train"].shape[1]),
        "weight_decay": rich_wd,
        "rich": {**rich_metrics, **rich_info},
        "shuffle": {**shuffle_metrics, **shuffle_info},
        "delta_nll_base_minus_rich": base_vs_rich,
        "delta_nll_shuffle_minus_rich": shuffle_vs_rich,
      }
      results["tests"].append(record)
      stem = "{}_k{}".format(variant, hop)
      save_probe(output / ("probe_" + stem + ".pt"), rich_model, rich_scaler, {
        "kind": "rich", "variant": variant, "hop": hop, "weight_decay": rich_wd,
        "dropout": args.dropout, "selection": rich_info,
      })
      save_probe(output / ("probe_shuffle_" + stem + ".pt"), shuffle_model, shuffle_scaler, {
        "kind": "shuffle", "variant": variant, "hop": hop, "weight_decay": rich_wd,
        "dropout": args.dropout, "selection": shuffle_info,
      })
      np.savez_compressed(output / ("test_predictions_" + stem + ".npz"),
                          labels=test_y, pair_ids=test_pair_ids,
                          base_logits=base_test_logits, rich_logits=rich_test_logits,
                          shuffle_logits=shuffle_test_logits)
      save_json(results, output / "conditional_residual.json")
      del histories, train_x, val_x, test_x, shuffled_history
      del shuffle_train_x, shuffle_val_x, shuffle_test_x
      gc.collect()

  with open(output / "conditional_residual.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["variant", "hop", "history_dim", "base_nll", "rich_nll", "shuffle_nll",
                     "delta_nll", "ci_low", "ci_high", "shuffle_minus_rich"])
    for record in results["tests"]:
      delta = record["delta_nll_base_minus_rich"]
      writer.writerow([record["variant"], record["hop"], record["history_dim"],
                       base_metrics["nll"], record["rich"]["nll"], record["shuffle"]["nll"],
                       delta["mean"], delta["ci_low"], delta["ci_high"],
                       record["delta_nll_shuffle_minus_rich"]["mean"]])
  save_json({"status": "complete", "tests": len(results["tests"])}, output / "_SUCCESS.json")
  print("Conditional residual results: {}".format((output / "conditional_residual.json").resolve()))


if __name__ == "__main__":
  main()
