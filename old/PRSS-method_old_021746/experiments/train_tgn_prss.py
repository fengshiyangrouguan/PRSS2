"""Train official TGN with PRSS active at every recursive embedding interface from step zero."""

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
  raise FileNotFoundError(
    "Official TGN source not found at {}. Set TGN_DIR to its directory.".format(TGN_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(TGN_DIR))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from model.tgn import TGN
from prss.ablations import VARIANTS, configure_ablation
from prss.integrations.tgn import install_tgn_prss
from prss.monitoring import (MonitorThresholds, PRSSEvidenceMonitor,
                            predictive_energy_coverage)
from utils.data_processing import Data, compute_time_statistics, get_data
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder


def parse_args():
  parser = argparse.ArgumentParser(description="Compressed-first TGN + PRSS training")
  parser.add_argument("--data", default="wikipedia")
  parser.add_argument("--data-dir", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--candidate-dim", type=int, default=256,
                      help="d_tau; k_tau is always derived from TGN")
  parser.add_argument("--variant", choices=VARIANTS, default="full",
                      help="Named method/ablation; each changes the actual training rule")
  parser.add_argument("--context-dim", type=int, default=64)
  parser.add_argument("--reader-hidden-dim", type=int, default=128)
  parser.add_argument("--lambda-resp", type=float, default=1.0)
  parser.add_argument("--lambda-spec", type=float, default=0.1)
  parser.add_argument("--gram-ema-rho", type=float, default=0.05)
  parser.add_argument("--spectral-update-interval", type=int, default=200)
  parser.add_argument("--spectral-warmup-steps", type=int, default=200)
  parser.add_argument("--spectral-step-size", type=float, default=0.25)
  parser.add_argument("--spectral-eigen-floor-ratio", type=float, default=1e-4)
  parser.add_argument("--ridge-eps", type=float, default=1e-5)
  parser.add_argument("--max-aux-nodes", type=int, default=64)
  parser.add_argument("--response-eval-batches", type=int, default=20)
  parser.add_argument("--monitor-every", type=int, default=50)
  parser.add_argument("--monitor-warmup-steps", type=int, default=200)
  parser.add_argument("--max-response-gap-ratio", type=float, default=2.0)
  parser.add_argument("--no-tensorboard", action="store_true")
  parser.add_argument("--no-fail-on-monitor-error", action="store_true",
                      help="Log invariant violations instead of stopping; not recommended")
  parser.add_argument("--batch-size", type=int, default=200)
  parser.add_argument("--n-degree", type=int, default=10)
  parser.add_argument("--n-layer", type=int, default=1)
  parser.add_argument("--n-head", type=int, default=2)
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument("--use-memory", action="store_true")
  parser.add_argument("--memory-dim", type=int, default=172)
  parser.add_argument("--message-dim", type=int, default=100)
  parser.add_argument("--message-function", choices=["identity", "mlp"], default="identity")
  parser.add_argument("--memory-updater", choices=["gru", "rnn"], default="gru")
  parser.add_argument("--aggregator", choices=["last", "mean"], default="last")
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--reader-weight-decay", type=float, default=1e-5)
  parser.add_argument("--patience", type=int, default=5)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--max-train-interactions", type=int, default=0,
                      help="Engineering smoke only; 0 uses the full temporal train split")
  parser.add_argument("--max-val-interactions", type=int, default=0)
  parser.add_argument("--max-test-interactions", type=int, default=0)
  return parser.parse_args()


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


def save_checkpoint(model, path, epoch, validation_ap):
  payload = {
    "state_dict": model.state_dict(),
    "epoch": int(epoch),
    "validation_ap": float(validation_ap),
  }
  if model.use_memory:
    payload["memory_backup"] = model.memory.backup_memory()
  torch.save(payload, path)


def load_checkpoint(model, path, device):
  try:
    # This is a checkpoint produced by this training script and includes TGN memory/message
    # Python objects, not just tensors. PyTorch >=2.6 defaults weights_only=True.
    payload = torch.load(path, map_location=device, weights_only=False)
  except TypeError:
    # PyTorch versions predating the weights_only keyword.
    payload = torch.load(path, map_location=device)
  model.load_state_dict(payload["state_dict"])
  if model.use_memory:
    if "memory_backup" not in payload:
      raise RuntimeError("PRSS TGN checkpoint lacks pending raw messages")
    model.memory.restore_memory(payload["memory_backup"])
  return payload


def capped_data(data, maximum):
  if maximum <= 0 or maximum >= len(data.sources):
    return data
  # A temporal-memory smoke must remain a causal prefix. Evenly-spaced thinning changes the memory
  # trajectory and silently turns the engineering cap into a different temporal process.
  end = int(maximum)
  return Data(data.sources[:end], data.destinations[:end], data.timestamps[:end],
              data.edge_idxs[:end], data.labels[:end])


@torch.no_grad()
def evaluate(model, adapter, bridge, sampler, data, n_neighbors, batch_size,
             response_eval_batches=0):
  model.eval()
  adapter.prss.eval()
  adapter.prss.set_spectral_updates_allowed(False)
  sampler.reset_random_state()
  aps, aucs = [], []
  structured_losses, unrestricted_losses = [], []
  predictive_coverages = defaultdict(list)
  num_batches = math.ceil(len(data.sources) / batch_size)
  for batch_index in range(num_batches):
    start = batch_index * batch_size
    end = min(start + batch_size, len(data.sources))
    sources = data.sources[start:end]
    destinations = data.destinations[start:end]
    timestamps = data.timestamps[start:end]
    edge_idxs = data.edge_idxs[start:end]
    _, negatives = sampler.sample(len(sources))
    collect_response = batch_index < response_eval_batches
    adapter.enable_trace(collect_response)
    positive, negative = model.compute_edge_probabilities(
      sources, destinations, negatives, timestamps, edge_idxs, n_neighbors)
    labels = np.concatenate([np.ones(len(sources)), np.zeros(len(sources))])
    scores = np.concatenate([positive.cpu().numpy().reshape(-1),
                             negative.cpu().numpy().reshape(-1)])
    aps.append(average_precision_score(labels, scores))
    aucs.append(roc_auc_score(labels, scores))
    if collect_response:
      auxiliary = bridge.build(timestamps)
      structured_losses.append(float(auxiliary.response))
      unrestricted_losses.append(float(auxiliary.unrestricted_response))
      for tau, readers in auxiliary.readers_by_tau.items():
        predictive_coverages[tau].append(predictive_energy_coverage(
          adapter.prss.quotients.state_for(tau).R, readers))
  adapter.enable_trace(False)
  return {
    "ap": float(np.mean(aps)),
    "auc": float(np.mean(aucs)),
    "structured_response_nll": (float(np.mean(structured_losses))
                                if structured_losses else None),
    "unrestricted_response_nll": (float(np.mean(unrestricted_losses))
                                  if unrestricted_losses else None),
    "response_gap": (float(np.mean(structured_losses) - np.mean(unrestricted_losses))
                     if structured_losses else None),
    "predictive_energy_coverage": {
      tau: float(np.mean(values)) for tau, values in predictive_coverages.items()
    },
  }


def main():
  args = parse_args()
  set_seed(args.seed)
  output = Path(args.output)
  output.mkdir(parents=True, exist_ok=True)
  device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
  node_features, edge_features, full_data, train_data, val_data, test_data, _, _ = get_data(
    args.data, data_dir=args.data_dir)
  train_data = capped_data(train_data, args.max_train_interactions)
  val_data = capped_data(val_data, args.max_val_interactions)
  test_data = capped_data(test_data, args.max_test_interactions)
  max_node_idx = int(max(full_data.sources.max(), full_data.destinations.max()))
  train_finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_node_idx)
  full_finder = get_neighbor_finder(full_data, uniform=False, max_node_idx=max_node_idx)
  train_sampler = RandEdgeSampler(train_data.sources, train_data.destinations, seed=args.seed + 101)
  val_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=args.seed + 202)
  test_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=args.seed + 303)
  time_statistics = compute_time_statistics(
    train_data.sources, train_data.destinations, train_data.timestamps)

  tgn = TGN(
    neighbor_finder=train_finder,
    node_features=node_features,
    edge_features=edge_features,
    device=device,
    n_layers=args.n_layer,
    n_heads=args.n_head,
    dropout=args.dropout,
    use_memory=args.use_memory,
    memory_dimension=args.memory_dim,
    message_dimension=args.message_dim,
    embedding_module_type="graph_attention",
    message_function=args.message_function,
    mean_time_shift_src=time_statistics[0],
    std_time_shift_src=max(time_statistics[1], 1e-12),
    mean_time_shift_dst=time_statistics[2],
    std_time_shift_dst=max(time_statistics[3], 1e-12),
    n_neighbors=args.n_degree,
    aggregator_type=args.aggregator,
    memory_updater_type=args.memory_updater,
  ).to(device)
  # Installation occurs before the optimizer and before the first forward: no vanilla pretraining.
  # Preserve the host RNG stream across extra PRSS-module initialization so named ablations share
  # the same TGN dropout sequence under a matched seed.
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
    ridge_eps=args.ridge_eps,
    max_nodes_per_scenario=args.max_aux_nodes,
    task="link",
  )
  policy = configure_ablation(adapter.prss, args.variant, integration_adapter=adapter)
  restore_torch_rng(task_rng_state)
  all_log_time = np.log1p(np.maximum(train_data.timestamps, 0))
  bridge.time_mean = float(all_log_time.mean())
  bridge.time_std = float(max(all_log_time.std(), 1e-12))

  main_parameters = [parameter for name, parameter in tgn.named_parameters()
                     if parameter.requires_grad and "unrestricted_readers" not in name]
  diagnostic_parameters = list(adapter.prss.unrestricted_readers.parameters())
  optimizer = torch.optim.Adam(main_parameters, lr=args.learning_rate,
                               weight_decay=args.reader_weight_decay)
  diagnostic_optimizer = torch.optim.Adam(
    diagnostic_parameters, lr=args.learning_rate, weight_decay=args.reader_weight_decay)
  criterion = torch.nn.BCELoss()
  early_stopper = EarlyStopMonitor(max_round=args.patience)
  checkpoint_path = output / "best_model.pt"
  metrics_path = output / "metrics.jsonl"
  global_step = 0

  manifest = vars(args).copy()
  manifest.update({
    "device": str(device),
    "host_interface_dim": int(adapter.prss.config.interface(adapter.root_interface_type).host_dim),
    "candidate_dim": int(adapter.prss.config.interface(adapter.root_interface_type).candidate_dim),
    "compression_ratio": float(adapter.prss.config.interface(adapter.root_interface_type).compression_ratio),
    "prss_interface_types": list(adapter.interface_types),
    "no_vanilla_pretrain": True,
    "parent_receives_only_quotient": True,
    "identity_compatible_initial_quotient": True,
    "spectral_loss_starts_after_first_spectral_update": True,
    "outside_reader_in_inference": False,
    "validation_test_spectral_updates": False,
    "ablation_policy": {
      "name": policy.name,
      "statistic": policy.statistic,
      "use_response_loss": policy.use_response_loss,
      "use_spectral_loss": policy.use_spectral_loss,
      "update_projection": policy.update_projection,
      "direct_projection_is_trainable": args.variant == "direct",
    },
    "prss_config": adapter.prss.config.as_dict(),
  })
  with open(output / "config.json", "w") as handle:
    json.dump(manifest, handle, indent=2)
  monitor = PRSSEvidenceMonitor(
    output / "monitor",
    variant=args.variant,
    monitor_every=args.monitor_every,
    warmup_steps=args.monitor_warmup_steps,
    thresholds=MonitorThresholds(
      max_structured_to_unrestricted_loss_ratio=args.max_response_gap_ratio),
    tensorboard=not args.no_tensorboard,
    fail_on_error=not args.no_fail_on_monitor_error,
  )
  if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

  for epoch in range(args.epochs):
    start_time = time.time()
    tgn.train()
    adapter.prss.train()
    adapter.prss.set_spectral_updates_allowed(True)
    adapter.enable_trace(True)
    tgn.set_neighbor_finder(train_finder)
    if args.use_memory:
      tgn.memory.__init_memory__()
    train_sampler.reset_random_state()
    order = np.arange(len(train_data.sources))
    epoch_losses = defaultdict(list)
    num_batches = math.ceil(len(order) / args.batch_size)

    for batch_index in range(num_batches):
      start = batch_index * args.batch_size
      end = min(start + args.batch_size, len(order))
      sources = train_data.sources[start:end]
      destinations = train_data.destinations[start:end]
      timestamps = train_data.timestamps[start:end]
      edge_idxs = train_data.edge_idxs[start:end]
      _, negatives = train_sampler.sample(len(sources))
      positive, negative = tgn.compute_edge_probabilities(
        sources, destinations, negatives, timestamps, edge_idxs, args.n_degree)
      auxiliary = bridge.build(timestamps)
      positive_target = torch.ones_like(positive)
      negative_target = torch.zeros_like(negative)
      task_loss = criterion(positive, positive_target) + criterion(negative, negative_target)
      total = task_loss
      if policy.use_response_loss:
        total = total + args.lambda_resp * auxiliary.response
      spectral_ready = any(
        int(state.spectral_updates.item()) > 0
        for state in adapter.prss.quotients.states.values())
      if policy.use_spectral_loss and spectral_ready:
        total = total + args.lambda_spec * auxiliary.spectral
      optimizer.zero_grad()
      total.backward()
      torch.nn.utils.clip_grad_norm_(main_parameters, max_norm=5.0)
      optimizer.step()

      diagnostic_optimizer.zero_grad()
      auxiliary.unrestricted_response.backward()
      diagnostic_optimizer.step()

      policy.update_statistics(
        adapter.prss, auxiliary.readers_by_tau, auxiliary.candidates_by_tau)
      spectral_updated = policy.maybe_update(adapter.prss, global_step)
      expected_gradient_groups = (
        ("lift", "structured_reader", "outside_encoder")
        if policy.use_response_loss else ("lift",))
      if args.variant == "direct":
        expected_gradient_groups = expected_gradient_groups + ("direct_projection",)
      monitor.record_step(
        global_step, epoch,
        losses={
          "task": task_loss,
          "response": auxiliary.response,
          "spectral": auxiliary.spectral,
          "unrestricted_response": auxiliary.unrestricted_response,
          "total": total,
        },
        auxiliary=auxiliary,
        model=tgn,
        system=adapter.prss,
        spectral_updated=spectral_updated,
        projection_expected_orthogonal=args.variant != "direct",
        expected_gradient_groups=expected_gradient_groups,
      )
      if args.use_memory:
        tgn.memory.detach_memory()
      epoch_losses["task"].append(float(task_loss.detach()))
      epoch_losses["response"].append(float(auxiliary.response.detach()))
      epoch_losses["spectral"].append(float(auxiliary.spectral.detach()))
      epoch_losses["unrestricted_response"].append(
        float(auxiliary.unrestricted_response.detach()))
      for tau, norm in adapter.prss.reader_norms(auxiliary.readers_by_tau).items():
        epoch_losses["reader_frobenius_norm/{}".format(tau)].append(norm)
      global_step += 1

    tgn.set_neighbor_finder(full_finder)
    validation = evaluate(
      tgn, adapter, bridge, val_sampler, val_data, args.n_degree, args.batch_size,
      response_eval_batches=args.response_eval_batches)
    record = {
      "epoch": epoch,
      "global_step": global_step,
      "seconds": time.time() - start_time,
      "train": {name: float(np.mean(values)) for name, values in epoch_losses.items()},
      "validation": validation,
      "spectral": adapter.prss.spectral_diagnostics(),
    }
    with open(metrics_path, "a") as handle:
      handle.write(json.dumps(record) + "\n")
    monitor.record_epoch(epoch, global_step, record["train"], validation, adapter.prss)

    should_stop = early_stopper.early_stop_check(validation["ap"])
    if early_stopper.best_epoch == epoch:
      save_checkpoint(tgn, checkpoint_path, epoch, validation["ap"])
    print("epoch={} task={:.5f} resp={:.5f} spec={:.5f} val_ap={:.5f} val_auc={:.5f}".format(
      epoch, record["train"]["task"], record["train"]["response"],
      record["train"]["spectral"], validation["ap"], validation["auc"]))
    if should_stop:
      break

  best = load_checkpoint(tgn, checkpoint_path, device)
  tgn.set_neighbor_finder(full_finder)
  inference_counters_before = {
    tau: (int(adapter.prss.quotients.state_for(tau).gram_updates),
          int(adapter.prss.quotients.state_for(tau).spectral_updates))
    for tau in adapter.prss.config.interfaces
  }
  test = evaluate(tgn, adapter, bridge, test_sampler, test_data, args.n_degree,
                  args.batch_size, response_eval_batches=args.response_eval_batches)
  inference_counters_after = {
    tau: (int(adapter.prss.quotients.state_for(tau).gram_updates),
          int(adapter.prss.quotients.state_for(tau).spectral_updates))
    for tau in adapter.prss.config.interfaces
  }
  # Run a separate standard-inference probe with trace/outside disabled. Response diagnostics
  # during evaluation are explicitly allowed to use the training-only branch without updates.
  probe_memory = tgn.memory.backup_memory() if args.use_memory else None
  adapter.enable_trace(False)
  probe_source = test_data.sources[:1]
  probe_destination = test_data.destinations[:1]
  _, probe_negative = test_sampler.sample(1)
  probe_timestamp = test_data.timestamps[:1]
  probe_edge = test_data.edge_idxs[:1]
  probe_before = {
    tau: (int(adapter.prss.quotients.state_for(tau).gram_updates),
          int(adapter.prss.quotients.state_for(tau).spectral_updates))
    for tau in adapter.prss.config.interfaces
  }
  with torch.no_grad():
    probe_embeddings = tgn.compute_temporal_embeddings(
      probe_source, probe_destination, probe_negative, probe_timestamp, probe_edge,
      args.n_degree)
  probe_after = {
    tau: (int(adapter.prss.quotients.state_for(tau).gram_updates),
          int(adapter.prss.quotients.state_for(tau).spectral_updates))
    for tau in adapter.prss.config.interfaces
  }
  if probe_memory is not None:
    tgn.memory.restore_memory(probe_memory)
  inference_contract = {
    "standard_inference_outside_reader_used": adapter.last_trace is not None,
    "validation_test_gram_updated": any(
      inference_counters_before[tau][0] != inference_counters_after[tau][0]
      for tau in inference_counters_before),
    "validation_test_svd_updated": any(
      inference_counters_before[tau][1] != inference_counters_after[tau][1]
      for tau in inference_counters_before),
    "standard_inference_gram_updated": any(
      probe_before[tau][0] != probe_after[tau][0] for tau in probe_before),
    "standard_inference_svd_updated": any(
      probe_before[tau][1] != probe_after[tau][1] for tau in probe_before),
    "actual_output_widths": [int(values.shape[-1]) for values in probe_embeddings],
    "host_output_width": int(adapter.prss.config.interface(adapter.root_interface_type).host_dim),
    "response_diagnostics_used_outside_reader_during_eval": args.response_eval_batches > 0,
  }
  monitor_summary = monitor.close(inference_contract)
  final = {
    "variant": args.variant,
    "best_epoch": int(best["epoch"]),
    "best_validation_ap": float(best["validation_ap"]),
    "test": test,
    "spectral": adapter.prss.spectral_diagnostics(),
    "inference_contract": inference_contract,
    "monitor_summary": monitor_summary,
  }
  with open(output / "results.json", "w") as handle:
    json.dump(final, handle, indent=2)
  with open(output / "_SUCCESS.json", "w") as handle:
    json.dump({"status": "complete", "best_epoch": final["best_epoch"]}, handle, indent=2)
  print(json.dumps(final, indent=2))


if __name__ == "__main__":
  main()
