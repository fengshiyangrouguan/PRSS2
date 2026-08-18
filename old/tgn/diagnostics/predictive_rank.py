"""Cross-fitted predictive signatures, predictive SVD, and held-out reduced-rank regression."""

import argparse
import gc
import sys
from pathlib import Path

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from diagnostics.common import (CutDataset, evenly_spaced_selection, load_probe,
                                normalized_log_time, paired_rows, predict_logits,
                                resolve_device, save_json, sigmoid, train_probe)


def parse_args():
  parser = argparse.ArgumentParser(description="Predictive response rank diagnostics")
  parser.add_argument("--cuts", required=True)
  parser.add_argument("--probe", required=True,
                      help="Selected rich probe from conditional_residual.py")
  parser.add_argument("--output", required=True)
  parser.add_argument("--variant", default="all")
  parser.add_argument("--hop", type=int, default=3)
  parser.add_argument("--contexts", type=int, default=256)
  parser.add_argument("--train-histories", type=int, default=4000)
  parser.add_argument("--val-histories", type=int, default=2000)
  parser.add_argument("--test-histories", type=int, default=2000)
  parser.add_argument("--cross-fit-folds", type=int, default=5)
  parser.add_argument("--ranks", default="1,2,4,8,16,32,64,128")
  parser.add_argument("--whiten-rank", type=int, default=256,
                      help="PCA dimension used only to whiten X; 0 requests full numerical rank")
  parser.add_argument("--ridge", default="0,1e-4,1e-2,1")
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--max-epochs", type=int, default=100)
  parser.add_argument("--patience", type=int, default=10)
  parser.add_argument("--seed", type=int, default=2027)
  parser.add_argument("--gpu", type=int, default=0)
  return parser.parse_args()


def choose_contexts(dataset, count):
  base, _, _ = dataset.base_rows("train")
  embedding_dim = int(dataset.manifest["embedding_dim"])
  contexts = base[:, embedding_dim:]
  n_pairs = dataset.pair_count("train")
  positive_count = count // 2
  negative_count = count - positive_count
  pos_idx = evenly_spaced_selection(n_pairs, positive_count)
  neg_idx = n_pairs + evenly_spaced_selection(n_pairs, negative_count)
  return np.asarray(contexts[np.concatenate([pos_idx, neg_idx])], dtype=np.float32)


def signature_inputs(dataset, split, indexes, contexts, history):
  arrays = dataset.splits[split]
  h = np.asarray(arrays["source_h"][indexes], dtype=np.float32)
  x = np.asarray(history[indexes], dtype=np.float32)
  return h, x


def predict_signature(model, scaler, h, history, contexts, device, batch_histories=32):
  n, m = len(h), len(contexts)
  output = np.empty((n, m), dtype=np.float32)
  embedding_dim = h.shape[1]
  model.eval()
  with torch.no_grad():
    for start in range(0, n, batch_histories):
      end = min(start + batch_histories, n)
      count = end - start
      base = np.concatenate([
        np.repeat(h[start:end], m, axis=0),
        np.tile(contexts, (count, 1)),
      ], axis=1)
      repeated_history = np.repeat(history[start:end], m, axis=0)
      values = np.concatenate([base, repeated_history], axis=1)
      logits = predict_logits(model, values, scaler, device, batch_size=8192)
      output[start:end] = sigmoid(logits).reshape(count, m)
  return output


def fit_cross_fitted_signatures(dataset, args, contexts, selections, full_probe_metadata, device):
  """Generate OOF train signatures; validation/test signatures use the train-fitted full probe."""
  train_base, train_y, train_pair_ids = dataset.base_rows("train")
  val_base, val_y, _ = dataset.base_rows("val")
  train_history = dataset.history("train", args.hop, args.variant)
  val_history = dataset.history("val", args.hop, args.variant)
  train_x = paired_rows(train_base, train_history)
  val_x = paired_rows(val_base, val_history)
  n_pairs = dataset.pair_count("train")
  if args.cross_fit_folds < 2 or args.cross_fit_folds > n_pairs:
    raise ValueError("--cross-fit-folds must be between 2 and the number of train pairs")
  rng = np.random.RandomState(args.seed)
  pair_order = rng.permutation(n_pairs)
  pair_fold = np.empty(n_pairs, dtype=np.int64)
  pair_fold[pair_order] = np.arange(n_pairs) % args.cross_fit_folds
  selected_train = selections["train"]
  h_selected, x_selected = signature_inputs(
    dataset, "train", selected_train, contexts, train_history)
  signatures = np.empty((len(selected_train), len(contexts)), dtype=np.float32)
  fold_records = []

  for fold in range(args.cross_fit_folds):
    row_train_mask = pair_fold[train_pair_ids] != fold
    model, scaler, info = train_probe(
      train_x[row_train_mask], train_y[row_train_mask], val_x, val_y, device,
      hidden_dim=int(full_probe_metadata["hidden_dim"]),
      dropout=float(full_probe_metadata.get("dropout", 0.1)),
      weight_decay=float(full_probe_metadata["weight_decay"]),
      learning_rate=float(full_probe_metadata.get("learning_rate", 1e-3)),
      batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience,
      seed=args.seed + fold)
    selected_mask = pair_fold[selected_train] == fold
    signatures[selected_mask] = predict_signature(
      model, scaler, h_selected[selected_mask], x_selected[selected_mask], contexts, device)
    fold_records.append({"fold": fold, "train_pairs": int(np.sum(pair_fold != fold)),
                         "signature_pairs": int(selected_mask.sum()), **info})
    del model, scaler
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  del train_x, val_x, train_history, val_history
  gc.collect()
  return signatures, fold_records


def svd_rank_curve(g_train, g_val, g_test, ranks):
  column_mean = g_train.mean(axis=0, keepdims=True)
  centered = g_train - column_mean
  _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
  total_energy = float(np.sum(singular_values ** 2))
  curves = []
  for rank in ranks:
    rank = min(rank, len(singular_values))
    basis = vt[:rank].T
    record = {"rank": int(rank),
              "train_energy": float(np.sum(singular_values[:rank] ** 2) / max(total_energy, 1e-12))}
    for name, matrix in (("train", g_train), ("val", g_val), ("test", g_test)):
      centered_matrix = matrix - column_mean
      reconstruction = centered_matrix @ basis @ basis.T
      mse = np.mean((centered_matrix - reconstruction) ** 2)
      baseline = np.mean(centered_matrix ** 2)
      record[name + "_mse"] = float(mse)
      record[name + "_relative_mse"] = float(mse / max(baseline, 1e-12))
    curves.append(record)
  return singular_values, column_mean, curves


def whiten_history(x_train, x_val, x_test, whiten_rank, seed):
  mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
  scale = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
  scale[scale < 1e-6] = 1.0
  train = (x_train - mean) / scale
  val = (x_val - mean) / scale
  test = (x_test - mean) / scale
  maximum = min(train.shape[0] - 1, train.shape[1])
  n_components = maximum if whiten_rank <= 0 else min(whiten_rank, maximum)
  solver = "full" if n_components == maximum else "randomized"
  pca = PCA(n_components=n_components, whiten=True, svd_solver=solver, random_state=seed)
  z_train = pca.fit_transform(train).astype(np.float32)
  return (z_train, pca.transform(val).astype(np.float32), pca.transform(test).astype(np.float32),
          pca, mean, scale)


def rrr_curve(x_train, x_val, x_test, g_train, g_val, g_test, ranks, ridges,
              whiten_rank, seed):
  z_train, z_val, z_test, pca, _, _ = whiten_history(
    x_train, x_val, x_test, whiten_rank, seed)
  y_mean = g_train.mean(axis=0, keepdims=True)
  y_train = g_train - y_mean
  y_val = g_val - y_mean
  y_test = g_test - y_mean
  ztz = z_train.T @ z_train
  zty = z_train.T @ y_train
  max_rank = min(max(ranks), z_train.shape[1], g_train.shape[1])
  candidates = []
  for ridge in ridges:
    # Tiny numerical jitter makes the requested ridge=0 case stable without acting as model
    # selection regularization.
    coefficient = np.linalg.solve(
      ztz + (float(ridge) + 1e-8) * np.eye(ztz.shape[0], dtype=np.float32), zty)
    u, singular_values, vt = np.linalg.svd(coefficient, full_matrices=False)
    prediction = (((z_val @ u[:, :max_rank]) * singular_values[:max_rank]) @
                  vt[:max_rank])
    val_mse = float(np.mean((y_val - prediction) ** 2))
    candidates.append((val_mse, float(ridge), u, singular_values, vt))
  candidates.sort(key=lambda item: item[0])
  _, selected_ridge, u, singular_values, vt = candidates[0]
  curves = []
  for rank in ranks:
    rank = min(rank, len(singular_values))
    record = {"rank": int(rank)}
    for name, z, y in (("train", z_train, y_train), ("val", z_val, y_val),
                       ("test", z_test, y_test)):
      prediction = ((z @ u[:, :rank]) * singular_values[:rank]) @ vt[:rank]
      mse = np.mean((y - prediction) ** 2)
      baseline = np.mean(y ** 2)
      record[name + "_mse"] = float(mse)
      record[name + "_relative_mse"] = float(mse / max(baseline, 1e-12))
    curves.append(record)
  metadata = {
    "selected_ridge": selected_ridge,
    "whiten_components": int(pca.n_components_),
    "whiten_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
  }
  return singular_values, curves, metadata


def main():
  args = parse_args()
  output = Path(args.output)
  output.mkdir(parents=True, exist_ok=True)
  device = resolve_device(args.gpu)
  dataset = CutDataset(args.cuts)
  full_model, full_scaler, probe_metadata = load_probe(args.probe, device)
  if probe_metadata.get("variant") != args.variant or int(probe_metadata.get("hop", -1)) != args.hop:
    raise ValueError("Probe metadata does not match requested variant/hop")
  # Make architectural facts explicit for the fold trainers.
  payload = torch.load(args.probe, map_location="cpu")
  probe_metadata = dict(probe_metadata)
  probe_metadata["hidden_dim"] = int(payload["hidden_dim"])

  requested = {"train": args.train_histories, "val": args.val_histories,
               "test": args.test_histories}
  selections = {split: evenly_spaced_selection(dataset.pair_count(split), requested[split])
                for split in ("train", "val", "test")}
  contexts = choose_contexts(dataset, args.contexts)
  g_train, fold_records = fit_cross_fitted_signatures(
    dataset, args, contexts, selections, probe_metadata, device)

  histories = {split: dataset.history(split, args.hop, args.variant)
               for split in ("val", "test")}
  signatures = {"train": g_train}
  for split in ("val", "test"):
    h, x = signature_inputs(dataset, split, selections[split], contexts, histories[split])
    signatures[split] = predict_signature(full_model, full_scaler, h, x, contexts, device)
  g_val, g_test = signatures["val"], signatures["test"]
  del histories

  np.save(output / "contexts.npy", contexts)
  for split in ("train", "val", "test"):
    np.save(output / ("G_" + split + ".npy"), signatures[split])
    np.save(output / ("history_indexes_" + split + ".npy"), selections[split])

  ranks = sorted(set(int(value) for value in args.ranks.split(",") if value))
  ranks = [rank for rank in ranks if rank <= min(len(contexts), len(g_train))]
  singular_values, _, svd_curve = svd_rank_curve(g_train, g_val, g_test, ranks)

  x = {split: dataset.history(split, args.hop, args.variant)[selections[split]]
       for split in ("train", "val", "test")}
  rrr_singular_values, rrr_results, rrr_metadata = rrr_curve(
    x["train"], x["val"], x["test"], g_train, g_val, g_test, ranks,
    [float(value) for value in args.ridge.split(",") if value],
    args.whiten_rank, args.seed)
  result = {
    "variant": args.variant,
    "hop": args.hop,
    "contexts": int(len(contexts)),
    "history_counts": {split: int(len(selections[split])) for split in selections},
    "cross_fit_folds": fold_records,
    "predictive_svd": {
      "singular_values": singular_values.tolist(),
      "curve": svd_curve,
      "centering": "training-signature column mean",
    },
    "reduced_rank_regression": {
      "coefficient_singular_values": rrr_singular_values.tolist(),
      "curve": rrr_results,
      **rrr_metadata,
    },
  }
  save_json(result, output / "predictive_rank.json")

  fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
  axes[0].semilogy(np.arange(1, len(singular_values) + 1), singular_values, marker=".")
  axes[0].set_xlabel("predictive signature singular index")
  axes[0].set_ylabel("singular value")
  axes[0].set_title("Centered future-response spectrum")
  axes[1].plot([row["rank"] for row in svd_curve],
               [row["test_relative_mse"] for row in svd_curve], marker="o")
  axes[1].set_xscale("log", base=2)
  axes[1].set_xlabel("rank")
  axes[1].set_ylabel("held-out relative MSE")
  axes[1].set_title("Predictive SVD")
  axes[2].plot([row["rank"] for row in rrr_results],
               [row["test_relative_mse"] for row in rrr_results], marker="o")
  axes[2].set_xscale("log", base=2)
  axes[2].set_xlabel("rank")
  axes[2].set_ylabel("held-out relative MSE")
  axes[2].set_title("History-to-signature RRR")
  fig.tight_layout()
  fig.savefig(output / "predictive_rank.png", dpi=180)
  plt.close(fig)
  save_json({"status": "complete", "variant": args.variant, "hop": args.hop},
            output / "_SUCCESS.json")
  print("Predictive rank results: {}".format((output / "predictive_rank.json").resolve()))


if __name__ == "__main__":
  main()
