"""End-to-end dynamic node classification with matched vanilla TGN and TGN+PRSS.

This is intentionally *not* the official two-stage "link-pretrain then freeze encoder" script.
For the PRSS scientific question the predictive target Y must be the node label itself, so both
vanilla and PRSS baselines are trained from random initialization on the same chronological node
classification task.  This keeps the comparison matched and avoids learning a link-specific quotient
then evaluating a different target.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
TGN_DIR = Path(os.environ.get("TGN_DIR", WORKSPACE / "tgn")).resolve()
if not (TGN_DIR / "model" / "tgn.py").is_file():
  raise FileNotFoundError("TGN source not found at {}".format(TGN_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(TGN_DIR))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from model.tgn import TGN
from prss.ablations import VARIANTS, configure_ablation
from prss.integrations.tgn import compute_source_event_embeddings, install_tgn_prss
from utils.data_processing import compute_time_statistics, get_data_node_classification
from utils.utils import EarlyStopMonitor, MLP, get_neighbor_finder


def parse_args():
  p = argparse.ArgumentParser("Matched end-to-end TGN node classification")
  p.add_argument("--data", default="wikipedia")
  p.add_argument("--data-dir", required=True)
  p.add_argument("--output", required=True)
  p.add_argument("--model", choices=["vanilla", "prss"], required=True)
  p.add_argument("--prss-variant", choices=VARIANTS, default="full")
  p.add_argument("--candidate-dim", type=int, default=256)
  p.add_argument("--context-dim", type=int, default=64)
  p.add_argument("--reader-hidden-dim", type=int, default=128)
  p.add_argument("--preagg-summary-dim", type=int, default=32)
  p.add_argument("--lambda-resp", type=float, default=0.5)
  p.add_argument("--lambda-spec", type=float, default=0.05)
  p.add_argument("--gram-ema-rho", type=float, default=0.1)
  p.add_argument("--spectral-update-interval", type=int, default=100)
  p.add_argument("--spectral-warmup-steps", type=int, default=100)
  p.add_argument("--spectral-step-size", type=float, default=0.25)
  p.add_argument("--spectral-eigen-floor-ratio", type=float, default=1e-4)
  p.add_argument("--max-aux-roots", type=int, default=8,
                 help="Trace only this many source roots per task batch; 0 traces all")
  p.add_argument("--max-aux-nodes", type=int, default=16,
                 help="Maximum internal occurrences supervised per traced source root")
  p.add_argument("--batch-size", type=int, default=200)
  p.add_argument("--n-degree", type=int, default=10)
  p.add_argument("--n-layer", type=int, default=2)
  p.add_argument("--n-head", type=int, default=2)
  p.add_argument("--dropout", type=float, default=0.1)
  p.add_argument("--memory-dim", type=int, default=172)
  p.add_argument("--message-dim", type=int, default=100)
  p.add_argument("--epochs", type=int, default=25)
  p.add_argument("--learning-rate", type=float, default=1e-4)
  p.add_argument("--weight-decay", type=float, default=1e-5)
  p.add_argument("--patience", type=int, default=5)
  p.add_argument("--selection-metric", choices=["ap", "auc", "neg_nll"], default="ap",
                 help="Validation metric used for checkpointing; AP is preferred for sparse labels")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--gpu", type=int, default=0)
  p.add_argument("--pos-weight-mode", choices=["none", "sqrt", "full"], default="sqrt")
  p.add_argument("--pos-weight-cap", type=float, default=100.0)
  p.add_argument("--max-train-interactions", type=int, default=0,
                 help="Engineering smoke only; 0 means full chronological train split")
  p.add_argument("--max-val-interactions", type=int, default=0)
  p.add_argument("--max-test-interactions", type=int, default=0)
  return p.parse_args()


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def capture_torch_rng():
  return (torch.get_rng_state(),
          torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_torch_rng(state):
  cpu_state, cuda_states = state
  torch.set_rng_state(cpu_state)
  if cuda_states is not None:
    torch.cuda.set_rng_state_all(cuda_states)


def cap_data(data, maximum):
  if maximum <= 0 or maximum >= len(data.sources):
    return data
  # Prefix, not evenly-spaced subsampling: chronological memory semantics must stay valid.
  end = int(maximum)
  from utils.data_processing import Data
  return Data(data.sources[:end], data.destinations[:end], data.timestamps[:end],
              data.edge_idxs[:end], data.labels[:end])


def safe_metrics(labels, scores, logits):
  labels = np.asarray(labels)
  scores = np.asarray(scores)
  logits = np.asarray(logits)
  if len(np.unique(labels)) < 2:
    auc = float("nan")
    ap = float(labels.mean())
  else:
    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
  # Stable binary NLL from logits.
  nll = float(np.mean(np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))))
  return {"auc": auc, "ap": ap, "nll": nll, "positives": int(labels.sum()),
          "pairs": int(len(labels))}


def pos_weight_from_labels(labels, mode, cap):
  positives = float(np.asarray(labels).sum())
  negatives = float(len(labels) - positives)
  if mode == "none" or positives <= 0:
    return 1.0
  ratio = negatives / positives
  if mode == "sqrt":
    ratio = math.sqrt(ratio)
  return float(min(ratio, cap))



def choose_aux_trace_rows(labels, maximum):
  """Label-balanced auxiliary sampling without putting the label into model inputs.

  Dynamic node labels in Wikipedia/Reddit are very sparse.  Uniformly tracing a handful of roots
  can give the continuation reader almost no positive examples.  Always retain positive training
  rows (up to the cap) and fill the remaining budget with evenly spaced negatives.
  """
  labels = np.asarray(labels).reshape(-1)
  n = len(labels)
  if maximum <= 0 or maximum >= n:
    return np.arange(n, dtype=np.int64)
  positive = np.flatnonzero(labels > 0.5)
  negative = np.flatnonzero(labels <= 0.5)
  if len(positive) >= maximum:
    pick = np.linspace(0, len(positive) - 1, maximum, dtype=np.int64)
    return np.sort(positive[np.unique(pick)])
  remaining = maximum - len(positive)
  if remaining > 0 and len(negative) > 0:
    pick = np.linspace(0, len(negative) - 1, min(remaining, len(negative)), dtype=np.int64)
    negative_pick = negative[np.unique(pick)]
  else:
    negative_pick = np.empty(0, dtype=np.int64)
  return np.sort(np.concatenate([positive, negative_pick])).astype(np.int64)


def assert_module_on_device(module, device, name):
  wrong = []
  for parameter_name, parameter in module.named_parameters():
    if parameter.device != device:
      wrong.append((parameter_name, str(parameter.device)))
  if wrong:
    raise RuntimeError("{} has parameters off {}: {}".format(name, device, wrong[:5]))

def save_checkpoint(tgn, decoder, path, epoch, validation_score, model_kind):
  payload = {
    "state_dict": tgn.state_dict(),
    "decoder_state_dict": decoder.state_dict(),
    "epoch": int(epoch),
    "validation_score": float(validation_score),
    "model": model_kind,
  }
  if tgn.use_memory:
    payload["memory_backup"] = tgn.memory.backup_memory()
  torch.save(payload, path)


def load_checkpoint(tgn, decoder, path, device):
  try:
    # This is a checkpoint produced by this training script and includes TGN memory/message
    # Python objects, not just tensors. PyTorch >=2.6 defaults weights_only=True.
    payload = torch.load(path, map_location=device, weights_only=False)
  except TypeError:
    # PyTorch versions predating the weights_only keyword.
    payload = torch.load(path, map_location=device)
  tgn.load_state_dict(payload["state_dict"])
  decoder.load_state_dict(payload["decoder_state_dict"])
  if tgn.use_memory:
    tgn.memory.restore_memory(payload["memory_backup"])
  return payload


@torch.no_grad()
def evaluate(tgn, decoder, data, neighbor_finder, n_degree, batch_size, adapter=None):
  tgn.eval()
  decoder.eval()
  tgn.set_neighbor_finder(neighbor_finder)
  if adapter is not None:
    adapter.prss.eval()
    adapter.prss.set_spectral_updates_allowed(False)
    adapter.enable_trace(False)
  all_labels, all_scores, all_logits = [], [], []
  batches = math.ceil(len(data.sources) / batch_size)
  for batch_index in range(batches):
    start = batch_index * batch_size
    end = min(start + batch_size, len(data.sources))
    source_h = compute_source_event_embeddings(
      tgn, data.sources[start:end], data.destinations[start:end], data.timestamps[start:end],
      data.edge_idxs[start:end], n_neighbors=n_degree)
    logits = decoder(source_h)
    scores = torch.sigmoid(logits)
    all_labels.append(np.asarray(data.labels[start:end], dtype=np.float64))
    all_scores.append(scores.detach().cpu().numpy().astype(np.float64))
    all_logits.append(logits.detach().cpu().numpy().astype(np.float64))
    if tgn.use_memory:
      tgn.memory.detach_memory()
  return safe_metrics(np.concatenate(all_labels), np.concatenate(all_scores),
                      np.concatenate(all_logits))


def main():
  args = parse_args()
  set_seed(args.seed)
  output = Path(args.output)
  output.mkdir(parents=True, exist_ok=True)
  device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

  full_data, node_features, edge_features, train_data, val_data, test_data = \
    get_data_node_classification(args.data, use_validation=True, data_dir=args.data_dir)
  train_data = cap_data(train_data, args.max_train_interactions)
  val_data = cap_data(val_data, args.max_val_interactions)
  test_data = cap_data(test_data, args.max_test_interactions)
  max_node_idx = int(max(full_data.sources.max(), full_data.destinations.max()))
  train_finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_node_idx)
  full_finder = get_neighbor_finder(full_data, uniform=False, max_node_idx=max_node_idx)
  # Train-only normalization avoids even weak future-statistic leakage.
  time_stats = compute_time_statistics(train_data.sources, train_data.destinations,
                                       train_data.timestamps)

  tgn = TGN(
    neighbor_finder=train_finder,
    node_features=node_features,
    edge_features=edge_features,
    device=device,
    n_layers=args.n_layer,
    n_heads=args.n_head,
    dropout=args.dropout,
    use_memory=True,
    memory_dimension=args.memory_dim,
    message_dimension=args.message_dim,
    embedding_module_type="graph_attention",
    message_function="identity",
    mean_time_shift_src=time_stats[0],
    std_time_shift_src=max(time_stats[1], 1e-12),
    mean_time_shift_dst=time_stats[2],
    std_time_shift_dst=max(time_stats[3], 1e-12),
    n_neighbors=args.n_degree,
    aggregator_type="last",
    memory_updater_type="gru",
  ).to(device)
  # Construct the task decoder *before* PRSS modules consume any RNG.  With the same seed this makes
  # vanilla, response-only and full PRSS start from bitwise-matched host+decoder parameters.
  decoder = MLP(node_features.shape[1], drop=args.dropout).to(device)

  adapter = bridge = policy = None
  if args.model == "prss":
    # PRSS has extra randomly initialized training-only modules.  They must not shift the global RNG
    # stream used by host/decoder dropout, otherwise same-seed vanilla vs PRSS is not a matched run.
    task_rng_state = capture_torch_rng()
    adapter, bridge = install_tgn_prss(
      tgn,
      candidate_dim=args.candidate_dim,
      context_dim=args.context_dim,
      reader_hidden_dim=args.reader_hidden_dim,
      lambda_resp=args.lambda_resp,
      lambda_spec=args.lambda_spec,
      gram_ema_rho=args.gram_ema_rho,
      spectral_update_interval=args.spectral_update_interval,
      spectral_warmup_steps=args.spectral_warmup_steps,
      spectral_step_size=args.spectral_step_size,
      spectral_eigen_floor_ratio=args.spectral_eigen_floor_ratio,
      max_nodes_per_scenario=args.max_aux_nodes,
      preagg_summary_dim=args.preagg_summary_dim,
      trace_max_roots=args.max_aux_roots,
      task="node",
    )
    policy = configure_ablation(adapter.prss, args.prss_variant, integration_adapter=adapter)
    restore_torch_rng(task_rng_state)
    assert_module_on_device(adapter, device, "PRSS TGN adapter")
    log_time = np.log1p(np.maximum(train_data.timestamps, 0))
    bridge.time_mean = float(log_time.mean())
    bridge.time_std = float(max(log_time.std(), 1e-12))

  diagnostic_parameters = []
  if adapter is not None:
    diagnostic_parameters = list(adapter.prss.unrestricted_readers.parameters())
    diagnostic_ids = {id(x) for x in diagnostic_parameters}
    main_parameters = [p for p in list(tgn.parameters()) + list(decoder.parameters())
                       if p.requires_grad and id(p) not in diagnostic_ids]
  else:
    main_parameters = list(tgn.parameters()) + list(decoder.parameters())
  optimizer = torch.optim.Adam(main_parameters, lr=args.learning_rate,
                               weight_decay=args.weight_decay)
  diagnostic_optimizer = (torch.optim.Adam(diagnostic_parameters, lr=args.learning_rate,
                                           weight_decay=args.weight_decay)
                          if diagnostic_parameters else None)

  pos_weight = pos_weight_from_labels(train_data.labels, args.pos_weight_mode,
                                      args.pos_weight_cap)
  task_criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))
  early_stopper = EarlyStopMonitor(max_round=args.patience)
  checkpoint = output / "best_model.pt"
  metrics_path = output / "metrics.jsonl"
  global_step = 0

  config = vars(args).copy()
  config.update({
    "device": str(device),
    "train_positives": int(np.asarray(train_data.labels).sum()),
    "val_positives": int(np.asarray(val_data.labels).sum()),
    "test_positives": int(np.asarray(test_data.labels).sum()),
    "effective_pos_weight": pos_weight,
    "training_protocol": "end_to_end_dynamic_node_classification",
    "vanilla_pretraining": False,
    "chronological_batches": True,
    "candidate_before_bottleneck": args.model == "prss",
    "identity_compatible_initial_quotient": args.model == "prss",
    "spectral_loss_starts_after_first_spectral_update": args.model == "prss",
    "matched_initialization_contract": "host_and_task_decoder_built_before_prss_modules",
    "selection_metric": args.selection_metric,
  })
  if adapter is not None:
    config.update({
      "host_dim": adapter.prss.config.interface(adapter.root_interface_type).host_dim,
      "candidate_dim_actual": adapter.prss.config.interface(adapter.root_interface_type).candidate_dim,
      "prss_interface_types": list(adapter.interface_types),
      "aux_trace_sampling": "all_positive_then_even_negative_fill",
      "prss_config": adapter.prss.config.as_dict(),
    })
  with open(output / "config.json", "w") as handle:
    json.dump(config, handle, indent=2)

  for epoch in range(args.epochs):
    start_epoch = time.time()
    tgn.train()
    decoder.train()
    tgn.set_neighbor_finder(train_finder)
    tgn.memory.__init_memory__()
    if adapter is not None:
      adapter.prss.train()
      adapter.prss.set_spectral_updates_allowed(True)
      adapter.enable_trace(True)
    epoch_values = defaultdict(list)
    batches = math.ceil(len(train_data.sources) / args.batch_size)

    for batch_index in range(batches):
      start = batch_index * args.batch_size
      end = min(start + args.batch_size, len(train_data.sources))
      sources = train_data.sources[start:end]
      destinations = train_data.destinations[start:end]
      timestamps = train_data.timestamps[start:end]
      edge_idxs = train_data.edge_idxs[start:end]
      labels = np.asarray(train_data.labels[start:end], dtype=np.float32)

      if adapter is not None:
        adapter.set_trace_rows(choose_aux_trace_rows(labels, args.max_aux_roots))
      source_h = compute_source_event_embeddings(
        tgn, sources, destinations, timestamps, edge_idxs, n_neighbors=args.n_degree)
      logits = decoder(source_h)
      target = torch.from_numpy(labels).to(device)
      task_loss = task_criterion(logits, target)
      total = task_loss
      auxiliary = None
      spectral_updated = False
      if adapter is not None:
        auxiliary = bridge.build(timestamps, labels, pos_weight=pos_weight)
        if policy.use_response_loss:
          total = total + args.lambda_resp * auxiliary.response
        # Before the first predictive spectral update R is an identity-compatible host projection.
        # Applying the state-tail penalty during this warm-up would merely suppress the extra
        # pre-aggregation coordinates before B(C) has estimated which of them are useful.
        spectral_ready = any(
          int(state.spectral_updates.item()) > 0
          for state in adapter.prss.quotients.states.values())
        if policy.use_spectral_loss and spectral_ready:
          total = total + args.lambda_spec * auxiliary.spectral

      optimizer.zero_grad(set_to_none=True)
      total.backward()
      torch.nn.utils.clip_grad_norm_(main_parameters, 5.0)
      optimizer.step()

      if auxiliary is not None and diagnostic_optimizer is not None:
        diagnostic_optimizer.zero_grad(set_to_none=True)
        auxiliary.unrestricted_response.backward()
        diagnostic_optimizer.step()
        policy.update_statistics(adapter.prss, auxiliary.readers_by_tau,
                                 auxiliary.candidates_by_tau)
        spectral_updated = policy.maybe_update(adapter.prss, global_step)

      tgn.memory.detach_memory()
      epoch_values["task"].append(float(task_loss.detach()))
      epoch_values["total"].append(float(total.detach()))
      if auxiliary is not None:
        epoch_values["response"].append(float(auxiliary.response.detach()))
        epoch_values["state_spectral"].append(float(auxiliary.spectral.detach()))
        epoch_values["unrestricted_response"].append(
          float(auxiliary.unrestricted_response.detach()))
      global_step += 1

      if batch_index % 100 == 0:
        extra = ""
        if auxiliary is not None:
          extra = " resp={:.4f} state_spec={:.4f} svd={}".format(
            float(auxiliary.response.detach()), float(auxiliary.spectral.detach()),
            int(spectral_updated))
        print("epoch={} batch={}/{} task={:.4f}{}".format(
          epoch, batch_index, batches, float(task_loss.detach()), extra), flush=True)

    # Validation continues chronologically from the end-of-train memory.
    validation = evaluate(tgn, decoder, val_data, full_finder, args.n_degree,
                          args.batch_size, adapter=adapter)
    record = {
      "epoch": epoch,
      "global_step": global_step,
      "seconds": time.time() - start_epoch,
      "train": {k: float(np.mean(v)) for k, v in epoch_values.items()},
      "validation": validation,
    }
    if adapter is not None:
      record["spectral"] = adapter.prss.spectral_diagnostics()
    with open(metrics_path, "a") as handle:
      handle.write(json.dumps(record) + "\n")
    print("epoch={} model={} train_task={:.5f} val_auc={:.5f} val_ap={:.5f} val_nll={:.5f}".format(
      epoch, args.model, record["train"]["task"], validation["auc"], validation["ap"],
      validation["nll"]), flush=True)

    selection_value = ({
      "ap": validation["ap"],
      "auc": validation["auc"],
      "neg_nll": -validation["nll"],
    })[args.selection_metric]
    if not math.isfinite(selection_value):
      raise RuntimeError("Validation selection metric {} is non-finite; positives={}".format(
        args.selection_metric, validation["positives"]))
    should_stop = early_stopper.early_stop_check(selection_value)
    if early_stopper.best_epoch == epoch:
      save_checkpoint(tgn, decoder, checkpoint, epoch, selection_value, args.model)
    if should_stop:
      break

  best = load_checkpoint(tgn, decoder, checkpoint, device)
  # Checkpoint memory is the end of validation for the best epoch, so test is causal continuation.
  test = evaluate(tgn, decoder, test_data, full_finder, args.n_degree, args.batch_size,
                  adapter=adapter)
  result = {
    "model": args.model,
    "prss_variant": args.prss_variant if args.model == "prss" else None,
    "best_epoch": int(best["epoch"]),
    "selection_metric": args.selection_metric,
    "best_validation_score": float(best["validation_score"]),
    "test": test,
    "effective_pos_weight": pos_weight,
  }
  if adapter is not None:
    result["spectral"] = adapter.prss.spectral_diagnostics()
  with open(output / "results.json", "w") as handle:
    json.dump(result, handle, indent=2)
  with open(output / "_SUCCESS.json", "w") as handle:
    json.dump({"status": "complete", "best_epoch": result["best_epoch"]}, handle, indent=2)
  print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
  main()
