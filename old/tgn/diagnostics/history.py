"""Deterministic descriptors of the exact temporal tree sampled below a TGN state.

The official GraphEmbedding recursion keeps the original query timestamp at every recursive
call.  This module deliberately mirrors that behavior.  It does not replace the TGN state and
does not modify the model forward pass; it only records fixed statistics of information that was
available below the compression interface.
"""

import numpy as np


STRUCTURE_NAMES = (
  "log_frontier_nodes",
  "log_valid_edges",
  "valid_slot_fraction",
  "log_unique_neighbors",
  "unique_neighbor_fraction",
  "log_dt_mean",
  "log_dt_std",
  "log_dt_min",
  "log_dt_max",
  "log_dt_most_recent",
  "branching_mean",
  "branching_std",
)


def _group_count(owners, n_roots):
  return np.bincount(owners, minlength=n_roots).astype(np.float64)


def _group_mean_std(values, owners, n_roots):
  values = np.asarray(values, dtype=np.float64)
  if values.ndim == 1:
    values = values[:, None]
  sums = np.zeros((n_roots, values.shape[1]), dtype=np.float64)
  squares = np.zeros_like(sums)
  np.add.at(sums, owners, values)
  np.add.at(squares, owners, values * values)
  counts = np.maximum(_group_count(owners, n_roots), 1.0)[:, None]
  mean = sums / counts
  variance = np.maximum(squares / counts - mean * mean, 0.0)
  return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _group_min_max(values, owners, n_roots):
  minimum = np.full(n_roots, np.inf, dtype=np.float64)
  maximum = np.full(n_roots, -np.inf, dtype=np.float64)
  np.minimum.at(minimum, owners, values)
  np.maximum.at(maximum, owners, values)
  minimum[~np.isfinite(minimum)] = 0.0
  maximum[~np.isfinite(maximum)] = 0.0
  return minimum.astype(np.float32), maximum.astype(np.float32)


def _latest_rows(values, owners, times, n_roots):
  output = np.zeros((n_roots, values.shape[1]), dtype=np.float32)
  if len(values) == 0:
    return output
  order = np.lexsort((times, owners))
  ordered_owners = owners[order]
  last = np.r_[ordered_owners[1:] != ordered_owners[:-1], True]
  indexes = order[last]
  output[owners[indexes]] = values[indexes]
  return output


def _unique_counts(values, owners, n_roots):
  counts = np.zeros(n_roots, dtype=np.float64)
  if len(values) == 0:
    return counts
  pairs = np.stack([owners, values.astype(np.int64)], axis=1)
  unique_pairs = np.unique(pairs, axis=0)
  np.add.at(counts, unique_pairs[:, 0], 1)
  return counts


def describe_sampled_history(source_nodes, timestamps, neighbor_finder, edge_features,
                             lower_states, max_hops=3, n_neighbors=10):
  """Return per-hop structure/time, edge-feature, and lower-state descriptors.

  ``lower_states`` is exactly the layer-zero node tensor visible to GraphEmbedding for the current
  batch: raw node features for no-memory TGN and raw features plus effective memory for full TGN.
  Every statistic is fixed (no learned projection), so a positive residual result remains a
  one-way refutation of conditional sufficiency.
  """
  source_nodes = np.asarray(source_nodes, dtype=np.int64)
  timestamps = np.asarray(timestamps, dtype=np.float64)
  edge_features = np.asarray(edge_features, dtype=np.float32)
  lower_states = np.asarray(lower_states, dtype=np.float32)
  n_roots = len(source_nodes)
  edge_dim = edge_features.shape[1]
  state_dim = lower_states.shape[1]
  structure = np.zeros((n_roots, max_hops, len(STRUCTURE_NAMES)), dtype=np.float32)
  edge_descriptor = np.zeros((n_roots, max_hops, 3 * edge_dim), dtype=np.float32)
  state_descriptor = np.zeros((n_roots, max_hops, 3 * state_dim), dtype=np.float32)

  frontier_nodes = source_nodes.copy()
  frontier_times = timestamps.copy()
  frontier_owners = np.arange(n_roots, dtype=np.int64)

  for hop in range(max_hops):
    if len(frontier_nodes) == 0:
      break
    neighbors, edge_idxs, edge_times = neighbor_finder.get_temporal_neighbor(
      frontier_nodes, frontier_times, n_neighbors=n_neighbors)
    valid = neighbors != 0
    repeated_owners = np.repeat(frontier_owners, neighbors.shape[1])
    edge_owners = repeated_owners[valid.reshape(-1)]
    valid_neighbors = neighbors.reshape(-1)[valid.reshape(-1)].astype(np.int64)
    valid_edge_idxs = edge_idxs.reshape(-1)[valid.reshape(-1)].astype(np.int64)
    valid_edge_times = edge_times.reshape(-1)[valid.reshape(-1)].astype(np.float64)
    valid_cut_times = np.repeat(frontier_times, neighbors.shape[1])[valid.reshape(-1)]

    frontier_count = _group_count(frontier_owners, n_roots)
    edge_count = _group_count(edge_owners, n_roots)
    unique_count = _unique_counts(valid_neighbors, edge_owners, n_roots)
    branching_per_frontier = valid.sum(axis=1).astype(np.float64)
    branching_mean, branching_std = _group_mean_std(
      branching_per_frontier, frontier_owners, n_roots)

    log_dt = np.log1p(np.maximum(valid_cut_times - valid_edge_times, 0.0))
    dt_mean, dt_std = _group_mean_std(log_dt, edge_owners, n_roots)
    dt_min, dt_max = _group_min_max(log_dt, edge_owners, n_roots)
    recent_dt = np.zeros(n_roots, dtype=np.float32)
    if len(log_dt):
      order = np.lexsort((valid_edge_times, edge_owners))
      ordered_owners = edge_owners[order]
      last = np.r_[ordered_owners[1:] != ordered_owners[:-1], True]
      indexes = order[last]
      recent_dt[edge_owners[indexes]] = log_dt[indexes]

    structure[:, hop, :] = np.stack([
      np.log1p(frontier_count),
      np.log1p(edge_count),
      edge_count / np.maximum(frontier_count * n_neighbors, 1.0),
      np.log1p(unique_count),
      unique_count / np.maximum(edge_count, 1.0),
      dt_mean[:, 0],
      dt_std[:, 0],
      dt_min,
      dt_max,
      recent_dt,
      branching_mean[:, 0],
      branching_std[:, 0],
    ], axis=1).astype(np.float32)

    if len(valid_edge_idxs):
      current_edge_features = edge_features[valid_edge_idxs]
      edge_mean, edge_std = _group_mean_std(current_edge_features, edge_owners, n_roots)
      edge_recent = _latest_rows(current_edge_features, edge_owners, valid_edge_times, n_roots)
      edge_descriptor[:, hop, :] = np.concatenate(
        [edge_mean, edge_std, edge_recent], axis=1)

    # At each recursive level record both sides of the actual pre-aggregation lower-state input:
    # mean frontier/source state, mean neighbor state, and neighbor-state standard deviation.
    frontier_mean, _ = _group_mean_std(lower_states[frontier_nodes], frontier_owners, n_roots)
    if len(valid_neighbors):
      neighbor_mean, neighbor_std = _group_mean_std(
        lower_states[valid_neighbors], edge_owners, n_roots)
    else:
      neighbor_mean = np.zeros((n_roots, state_dim), dtype=np.float32)
      neighbor_std = np.zeros_like(neighbor_mean)
    state_descriptor[:, hop, :] = np.concatenate(
      [frontier_mean, neighbor_mean, neighbor_std], axis=1)

    # Critical fidelity point: official GraphEmbedding repeats the original query cut time, not
    # the connecting edge time, for the next recursive call.
    frontier_nodes = valid_neighbors
    frontier_times = valid_cut_times
    frontier_owners = edge_owners

  return structure, edge_descriptor, state_descriptor


def exact_source_degree(source_nodes, timestamps, neighbor_finder):
  return np.asarray([
    len(neighbor_finder.find_before(int(node), float(timestamp))[0])
    for node, timestamp in zip(source_nodes, timestamps)
  ], dtype=np.int32)
