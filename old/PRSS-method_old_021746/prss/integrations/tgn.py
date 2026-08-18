"""TGN integration for Predictive Relation-State Sheaf (PRSS).

The critical contract is *candidate-before-bottleneck*: an internal recursive state is built from
TGN's actual pre-aggregation tensors, then projected to the host width before the parent sees it.
This avoids the invalid ``k -> d -> k`` pattern that merely expands an already-compressed TGN
embedding and therefore cannot recover history discarded by the host aggregator.

The original TGN aggregation result is retained as one input to the candidate lift.  PRSS adds a
small factorized summary of source/neighbor/edge/time pre-aggregation tensors, constructs a rich
candidate ``h in R^d``, and exposes only ``z = R h in R^k`` to the next recursive level.
Outside/context readers are training-only and are never required at inference.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prss.config import InterfaceSpec, PRSSConfig
from prss.losses import response_loss
from prss.state import RecursiveOccurrence, RecursiveTrace
from prss.system import PRSSSystem


def _masked_mean(values, valid):
  weights = valid.to(values.dtype).unsqueeze(-1)
  return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_std(values, valid, eps=1e-6):
  mean = _masked_mean(values, valid)
  weights = valid.to(values.dtype).unsqueeze(-1)
  count = weights.sum(dim=1).clamp_min(1.0)
  variance = ((values - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
  # sqrt'(0) is infinite and produced NaN gradients for empty/constant temporal neighborhoods.
  # Stabilize before sqrt, then restore an exact zero summary when the row has no real neighbor.
  std = torch.sqrt(variance.clamp_min(0) + eps)
  has_valid = valid.any(dim=1, keepdim=True).to(values.dtype)
  return std * has_valid


def _last_valid(values, valid):
  # Official TGN's non-uniform sampler right-aligns the most recent interactions, so the last
  # column is the newest real neighbor whenever one exists. Uniform sampling is re-sorted by time.
  any_valid = valid.any(dim=1, keepdim=True).to(values.dtype)
  return values[:, -1, :] * any_valid


class TGNZeroPaddedLeafLift(nn.Module):
  """No-extra-information leaf candidate.

  Layer-0 has no recursive neighborhood to rescue.  Learned extra leaf coordinates would turn the
  method into a generic feature-expansion MLP and confound a gain from history preservation with a
  gain from additional current-state capacity.  We therefore append exact zeros at leaves.
  """

  def __init__(self, host_dim, candidate_dim):
    super().__init__()
    self.host_dim = int(host_dim)
    self.candidate_dim = int(candidate_dim)
    self.extra_dim = self.candidate_dim - self.host_dim
    if self.extra_dim < 0:
      raise ValueError("candidate_dim must be >= host_dim")

  def forward(self, raw):
    if self.extra_dim == 0:
      return raw
    zeros = raw.new_zeros(*raw.shape[:-1], self.extra_dim)
    return torch.cat([raw, zeros], dim=-1)


class TGNLinearPreAggregationCandidateLift(nn.Module):
  """Linear-fusion ablation over joint pre-aggregation history summaries."""

  def __init__(self, host_dim, time_dim, edge_dim, candidate_dim):
    super().__init__()
    self.host_dim = int(host_dim)
    self.time_dim = int(time_dim)
    self.edge_dim = int(edge_dim)
    self.candidate_dim = int(candidate_dim)
    self.extra_dim = self.candidate_dim - self.host_dim
    if self.extra_dim < 0:
      raise ValueError("candidate_dim must be >= host_dim")
    token_dim = self.host_dim + self.time_dim + self.edge_dim
    # source state + mean/std/last of *joint* (neighbor, time, edge) tokens + valid fraction.
    summary_width = self.host_dim + 3 * token_dim + 1
    self.extra = nn.Linear(summary_width, self.extra_dim) if self.extra_dim > 0 else None

  def forward(self, raw_output, source_lower, source_time, neighbor_lower,
              edge_time_embeddings, edge_features, mask, layer_fraction):
    del source_time, layer_fraction  # source time is always the zero-lag TGN encoding; layer has its own lift.
    if self.extra is None:
      return raw_output
    valid = ~mask
    tokens = torch.cat([neighbor_lower, edge_time_embeddings, edge_features], dim=-1)
    valid_fraction = valid.to(raw_output.dtype).mean(dim=1, keepdim=True)
    summary = torch.cat([
      source_lower,
      _masked_mean(tokens, valid),
      _masked_std(tokens, valid),
      _last_valid(tokens, valid),
      valid_fraction,
    ], dim=-1)
    return torch.cat([raw_output, self.extra(summary)], dim=-1)


class TGNPreAggregationCandidateLift(nn.Module):
  """Rich candidate from the *true* pre-aggregation tuple before the host-width interface.

  The first ``k`` coordinates are the exact vanilla TGN aggregate so ``R=[I,0]`` reproduces
  vanilla TGN at initialization.  The extra coordinates are learned directly from the joint
  neighbor tuples ``(lower quotient state, edge-time embedding, edge feature)`` and the source
  lower quotient.  A small bank of learned queries performs masked attentive pooling over all
  sampled historical neighbors.  This is deliberately richer than hand-written mean/std/last
  statistics while remaining O(n_neighbors * summary_dim * num_queries).

  Importantly, the parent recursive layer never sees this candidate.  It only sees ``R h``.
  """

  def __init__(self, host_dim, time_dim, edge_dim, candidate_dim, summary_dim=32,
               num_history_queries=4):
    super().__init__()
    self.host_dim = int(host_dim)
    self.time_dim = int(time_dim)
    self.edge_dim = int(edge_dim)
    self.candidate_dim = int(candidate_dim)
    self.extra_dim = self.candidate_dim - self.host_dim
    if self.extra_dim < 0:
      raise ValueError("candidate_dim must be >= host_dim")
    self.summary_dim = int(max(8, summary_dim))
    self.num_history_queries = int(max(1, num_history_queries))

    if self.extra_dim == 0:
      self.source_projector = None
      self.token_projector = None
      self.query_vectors = None
      self.scalar_projector = None
      self.fusion = None
      return

    self.source_projector = nn.Sequential(
      nn.Linear(self.host_dim, self.summary_dim), nn.GELU())
    token_dim = self.host_dim + self.time_dim + self.edge_dim
    self.token_projector = nn.Sequential(
      nn.Linear(token_dim, self.summary_dim), nn.GELU())
    self.query_vectors = nn.Parameter(torch.randn(self.num_history_queries, self.summary_dim) * 0.02)
    self.scalar_projector = nn.Sequential(nn.Linear(1, self.summary_dim), nn.GELU())
    # source + Q attentive summaries + global mean + newest + valid fraction
    fusion_width = (self.num_history_queries + 4) * self.summary_dim
    hidden = max(self.summary_dim * 2, self.extra_dim)
    self.fusion = nn.Sequential(
      nn.Linear(fusion_width, hidden),
      nn.GELU(),
      nn.Linear(hidden, self.extra_dim),
      # Fix the scale of the new coordinates; the first k vanilla coordinates remain untouched.
      nn.LayerNorm(self.extra_dim, elementwise_affine=False),
    )

  def _attentive_pool(self, tokens, valid):
    # scores: [batch, neighbors, queries]
    scale = float(self.summary_dim) ** -0.5
    scores = torch.einsum("bnd,qd->bnq", tokens, self.query_vectors) * scale
    valid_f = valid.to(tokens.dtype)
    scores = scores.masked_fill(~valid.unsqueeze(-1), -1e4)
    weights = torch.softmax(scores, dim=1) * valid_f.unsqueeze(-1)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    pooled = torch.einsum("bnq,bnd->bqd", weights, tokens)
    return pooled.reshape(tokens.shape[0], self.num_history_queries * self.summary_dim)

  def forward(self, raw_output, source_lower, source_time, neighbor_lower,
              edge_time_embeddings, edge_features, mask, layer_fraction):
    del source_time, layer_fraction
    if self.extra_dim == 0:
      return raw_output
    valid = ~mask
    joint_tokens = torch.cat([neighbor_lower, edge_time_embeddings, edge_features], dim=-1)
    token_features = self.token_projector(joint_tokens)
    valid_fraction = valid.to(raw_output.dtype).mean(dim=1, keepdim=True)
    attentive = self._attentive_pool(token_features, valid)
    blocks = [
      self.source_projector(source_lower),
      attentive,
      _masked_mean(token_features, valid),
      _last_valid(token_features, valid),
      self.scalar_projector(valid_fraction),
    ]
    extra = self.fusion(torch.cat(blocks, dim=-1))
    return torch.cat([raw_output, extra], dim=-1)


@dataclass
class TGNLinkAuxiliaryBatch:
  structured_logits: torch.Tensor
  unrestricted_logits: torch.Tensor
  targets: torch.Tensor
  readers_by_tau: dict
  candidates_by_tau: dict
  response: torch.Tensor
  spectral: torch.Tensor
  unrestricted_response: torch.Tensor
  occurrence_count: int


@dataclass
class TGNNodeAuxiliaryBatch(TGNLinkAuxiliaryBatch):
  pass


class TGNRecursiveEmbeddingAdapter(nn.Module):
  """Layer-typed recursive TGN adapter.

  Each recursive depth is a distinct host interface type.  TGN's attention blocks are layer-specific,
  so sharing one candidate lift/reader/quotient across L0/L1/L2 mixes incompatible coordinate systems
  and makes a single Gram average meaningless.  PRSS therefore learns one tau and one R per depth.
  """

  def __init__(self, host_embedding, prss_system, interface_prefix="tgn",
               preagg_summary_dim=32, trace_max_roots=0):
    super().__init__()
    if not hasattr(host_embedding, "aggregate"):
      raise ValueError("TGN PRSS requires a recursive GraphEmbedding host")
    self.host = host_embedding
    self.prss = prss_system
    self.interface_prefix = str(interface_prefix)
    self.interface_types = [self._name_for_layer(i) for i in range(int(host_embedding.n_layers) + 1)]
    self.root_interface_type = self.interface_types[-1]
    # Backward-compatible attribute: callers that only care about the root interface see the root tau.
    self.interface_type = self.root_interface_type
    host_width = int(host_embedding.embedding_dimension)
    for tau in self.interface_types:
      spec = prss_system.config.interface(tau)
      if spec.raw_dim != host_width or spec.host_dim != host_width:
        raise ValueError("TGN recursive host contract requires raw_dim == host_dim == {}".format(
          host_width))
    expected_local = int(host_embedding.n_edge_features) + 4
    if prss_system.config.parent_local_dim != expected_local:
      raise ValueError("TGN parent_local_dim must equal edge_dim + 4 ({})".format(expected_local))
    self._host_width = host_width
    self._time_width = int(host_embedding.n_time_features)
    self._edge_width = int(host_embedding.n_edge_features)
    self._candidate_width = int(prss_system.config.interface(self.root_interface_type).candidate_dim)
    self._preagg_summary_dim = int(preagg_summary_dim)
    self.preagg_lifts = nn.ModuleDict({
      str(layer): TGNPreAggregationCandidateLift(
        self._host_width, self._time_width, self._edge_width,
        int(prss_system.config.interface(self._name_for_layer(layer)).candidate_dim),
        summary_dim=self._preagg_summary_dim)
      for layer in range(1, int(host_embedding.n_layers) + 1)
    })
    self.leaf_lift = TGNZeroPaddedLeafLift(
      self._host_width,
      int(prss_system.config.interface(self._name_for_layer(0)).candidate_dim))
    self.trace_enabled = True
    self.trace_max_roots = int(trace_max_roots)
    self._forced_trace_rows = None
    self.last_trace = None
    self.last_root_ids = None
    self.last_root_rows = None
    self.last_root_quotients = None
    self._next_occurrence_id = 0

  def _name_for_layer(self, layer):
    return "{}_layer_{}".format(self.interface_prefix, int(layer))

  def tau_for_layer(self, layer):
    layer = int(layer)
    if layer < 0 or layer > self.host.n_layers:
      raise ValueError("Invalid TGN recursion layer {}".format(layer))
    return self._name_for_layer(layer)

  @property
  def neighbor_finder(self):
    return self.host.neighbor_finder

  @neighbor_finder.setter
  def neighbor_finder(self, value):
    self.host.neighbor_finder = value

  def enable_trace(self, enabled=True):
    self.trace_enabled = bool(enabled)

  def set_trace_max_roots(self, maximum):
    self.trace_max_roots = int(maximum)

  def set_trace_rows(self, rows):
    """Force exact top-level rows to trace on the next forward.

    Node labels are extremely sparse in JODIE Wikipedia.  Uniformly tracing 4/200 roots can miss
    almost every positive example, making the continuation reader/Gram a negative-only estimator.
    Training may therefore choose a label-balanced set of *which training examples receive auxiliary
    supervision*.  The label is never passed into the state or outside context.
    """
    if rows is None:
      self._forced_trace_rows = None
      return
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    self._forced_trace_rows = np.unique(rows)

  def use_linear_candidate_lift(self):
    """Swap only the TGN pre-aggregation lift for the no-nonlinear-lift ablation."""
    device = self.host.node_features.device
    self.preagg_lifts = nn.ModuleDict({
      str(layer): TGNLinearPreAggregationCandidateLift(
        self._host_width, self._time_width, self._edge_width,
        int(self.prss.config.interface(self.tau_for_layer(layer)).candidate_dim)).to(device)
      for layer in range(1, int(self.host.n_layers) + 1)
    })
    # Leaves still append zeros: the ablation changes history lift nonlinearity, not current-state capacity.
    self.leaf_lift = TGNZeroPaddedLeafLift(
      self._host_width,
      int(self.prss.config.interface(self.tau_for_layer(0)).candidate_dim)).to(device)

  def _root_record_mask(self, count):
    mask = np.zeros(count, dtype=bool)
    if not self.trace_enabled:
      self._forced_trace_rows = None
      return mask
    forced = self._forced_trace_rows
    self._forced_trace_rows = None  # one-shot; never leak a previous batch's row selection
    if forced is not None:
      if np.any(forced < 0) or np.any(forced >= count):
        raise ValueError("Forced trace row outside current batch")
      mask[forced] = True
      return mask
    if self.trace_max_roots <= 0 or self.trace_max_roots >= count:
      mask[:] = True
    else:
      indexes = np.linspace(0, count - 1, self.trace_max_roots, dtype=np.int64)
      mask[np.unique(indexes)] = True
    return mask

  def _new_occurrence(self, state, local, children, relations, deltas, metadata):
    identifier = self._next_occurrence_id
    self._next_occurrence_id += 1
    occurrence = RecursiveOccurrence(
      occurrence_id=identifier,
      tau=state.tau,
      state=state,
      local_features=local,
      children=children,
      child_relations=relations,
      child_delta_t=deltas,
      metadata=metadata,
    )
    self.last_trace.add(occurrence)
    return identifier

  def _leaf_local(self, batch_size, device, dtype):
    return torch.zeros(batch_size, self.prss.config.parent_local_dim,
                       device=device, dtype=dtype)

  def _parent_local(self, edge_features, edge_deltas, original_mask, n_layers):
    valid = ~original_mask
    edge_mean = _masked_mean(edge_features, valid)
    delta = torch.log1p(edge_deltas.clamp_min(0))
    count = valid.sum(dim=1).clamp_min(1).to(delta.dtype)
    delta_mean = (delta * valid).sum(dim=1) / count
    centered = (delta - delta_mean.unsqueeze(1)) * valid
    delta_std = torch.sqrt((centered.square().sum(dim=1) / count).clamp_min(0))
    valid_fraction = valid.to(delta.dtype).mean(dim=1)
    layer_value = torch.full_like(valid_fraction, float(n_layers) / max(self.host.n_layers, 1))
    return torch.cat([edge_mean, delta_mean[:, None], delta_std[:, None],
                      valid_fraction[:, None], layer_value[:, None]], dim=1)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20,
                        time_diffs=None, use_time_proj=True):
    del time_diffs, use_time_proj  # GraphAttentionEmbedding ignores these too.
    source_nodes = np.asarray(source_nodes)
    timestamps = np.asarray(timestamps)
    if self.trace_enabled:
      self.last_trace = RecursiveTrace()
      self._next_occurrence_id = 0
      record_mask = self._root_record_mask(len(source_nodes))
    else:
      self.last_trace = None
      self._forced_trace_rows = None
      record_mask = np.zeros(len(source_nodes), dtype=bool)
    quotient, occurrence_ids = self._compute(
      memory, source_nodes, timestamps, n_layers, n_neighbors, record_mask)
    self.last_root_quotients = quotient
    if self.trace_enabled:
      self.last_root_rows = np.flatnonzero(record_mask)
      self.last_root_ids = occurrence_ids[record_mask]
      self.last_trace.roots = self.last_root_ids.tolist()
    else:
      self.last_root_rows = None
      self.last_root_ids = None
    return quotient

  def _compute(self, memory, source_nodes, timestamps, n_layers, n_neighbors, record_mask):
    if n_layers < 0:
      raise ValueError("n_layers cannot be negative")
    tau = self.tau_for_layer(n_layers)
    device = self.host.device
    source_nodes_torch = torch.from_numpy(source_nodes).long().to(device)
    timestamps_torch = torch.from_numpy(timestamps).float().to(device).unsqueeze(1)
    source_time = self.host.time_encoder(torch.zeros_like(timestamps_torch))
    raw_source = self.host.node_features[source_nodes_torch]
    if self.host.use_memory:
      raw_source = memory[source_nodes] + raw_source

    if n_layers == 0:
      candidate = self.leaf_lift(raw_source)
      state = self.prss.make_state_from_candidate(tau, raw_source, candidate)
      if not self.trace_enabled:
        return state.quotient, None
      local = self._leaf_local(len(source_nodes), device, raw_source.dtype)
      ids = np.full(len(source_nodes), -1, dtype=np.int64)
      for row in np.flatnonzero(record_mask):
        row_state = type(state)(state.tau, state.raw[row], state.candidate[row], state.quotient[row])
        ids[row] = self._new_occurrence(
          row_state, local[row], [], [], [],
          {"source_node": int(source_nodes[row]), "timestamp": float(timestamps[row]),
           "layer": 0})
      return state.quotient, ids

    source_lower, source_ids = self._compute(
      memory, source_nodes, timestamps, n_layers - 1, n_neighbors, record_mask)
    neighbors, edge_idxs_numpy, edge_times = self.neighbor_finder.get_temporal_neighbor(
      source_nodes, timestamps, n_neighbors=n_neighbors)
    effective_neighbors = int(neighbors.shape[1])
    neighbors_torch = torch.from_numpy(neighbors).long().to(device)
    edge_idxs = torch.from_numpy(edge_idxs_numpy).long().to(device)
    edge_deltas_numpy = timestamps[:, np.newaxis] - edge_times
    edge_deltas = torch.from_numpy(edge_deltas_numpy).float().to(device)
    flat_neighbors = neighbors.reshape(-1)
    repeated_timestamps = np.repeat(timestamps, effective_neighbors)
    neighbor_record_mask = np.repeat(record_mask, effective_neighbors)
    neighbor_lower, neighbor_ids = self._compute(
      memory, flat_neighbors, repeated_timestamps, n_layers - 1, n_neighbors,
      neighbor_record_mask)
    neighbor_lower = neighbor_lower.view(len(source_nodes), effective_neighbors, -1)
    if self.trace_enabled:
      neighbor_ids = neighbor_ids.reshape(len(source_nodes), effective_neighbors)
    edge_time_embeddings = self.host.time_encoder(edge_deltas)
    edge_features = self.host.edge_features[edge_idxs]
    mask = neighbors_torch == 0
    original_mask = mask.clone()

    raw_output = self.host.aggregate(
      n_layers, source_lower, source_time, neighbor_lower,
      edge_time_embeddings, edge_features, mask)
    candidate = self.preagg_lifts[str(int(n_layers))](
      raw_output, source_lower, source_time, neighbor_lower,
      edge_time_embeddings, edge_features, original_mask,
      float(n_layers) / max(self.host.n_layers, 1))
    state = self.prss.make_state_from_candidate(tau, raw_output, candidate)
    if not self.trace_enabled:
      return state.quotient, None

    local = self._parent_local(edge_features, edge_deltas, original_mask, n_layers)
    ids = np.full(len(source_nodes), -1, dtype=np.int64)
    for row in np.flatnonzero(record_mask):
      children = [int(source_ids[row])]
      relations = [0]
      deltas = [0.0]
      for column in range(effective_neighbors):
        if not bool(original_mask[row, column]):
          child_id = int(neighbor_ids[row, column])
          if child_id < 0:
            raise RuntimeError("Trace sampling lost a required child occurrence")
          children.append(child_id)
          relations.append(1)
          deltas.append(float(edge_deltas_numpy[row, column]))
      row_state = type(state)(state.tau, state.raw[row], state.candidate[row], state.quotient[row])
      ids[row] = self._new_occurrence(
        row_state, local[row], children, relations, deltas,
        {"source_node": int(source_nodes[row]), "timestamp": float(timestamps[row]),
         "layer": int(n_layers)})
    return state.quotient, ids


class _TGNOutsideMixin:
  def _outside_for_root(self, root_id, root_metadata):
    trace = self.adapter.last_trace
    root = trace.occurrences[int(root_id)]
    contexts = {int(root_id): self.prss.outside.root_context(root_metadata, root.tau)}
    queue = [int(root_id)]
    order = []
    while queue:
      parent_id = queue.pop(0)
      order.append(parent_id)
      parent = trace.occurrences[parent_id]
      parent_context = contexts[parent_id]
      for position, child_id in enumerate(parent.children):
        child = trace.occurrences[child_id]
        siblings = defaultdict(list)
        for sibling_id in parent.children:
          if sibling_id != child_id:
            sibling = trace.occurrences[sibling_id]
            siblings[sibling.tau].append(sibling.state.candidate)
        sibling_tensors = {tau: torch.stack(values, dim=0)
                           for tau, values in siblings.items()}
        summary = self.prss.outside.summarize_siblings(sibling_tensors, parent_context)
        contexts[child_id] = self.prss.outside.child_context(
          parent_context, parent.local_features, parent.child_relations[position],
          parent.child_delta_t[position], summary, child.tau)
        queue.append(child_id)
    if self.max_nodes_per_scenario > 0 and len(order) > self.max_nodes_per_scenario:
      indexes = np.linspace(0, len(order) - 1, self.max_nodes_per_scenario, dtype=np.int64)
      order = [order[index] for index in indexes]
    outputs = []
    for occurrence_id in order:
      occurrence = trace.occurrences[occurrence_id]
      context = contexts[occurrence_id]
      structured, matrix, _ = self.prss.structured_read(
        occurrence.tau, context, occurrence.state.candidate)
      unrestricted = self.prss.unrestricted_read(
        occurrence.tau, context.detach(), occurrence.state.candidate.detach())
      outputs.append((occurrence.tau, structured, unrestricted, matrix, occurrence.state.candidate))
    return outputs

  def _pack(self, structured_logits, unrestricted_logits, targets, readers, spectral_terms,
            batch_cls, pos_weight=None):
    if not structured_logits:
      raise RuntimeError("No traced PRSS occurrences were available for response supervision")
    structured_logits = torch.stack(structured_logits)
    unrestricted_logits = torch.stack(unrestricted_logits)
    targets = torch.as_tensor(targets, device=structured_logits.device,
                              dtype=structured_logits.dtype)
    readers_by_tau = {tau: torch.stack(values) for tau, values in readers.items()}
    if pos_weight is None:
      response = response_loss(structured_logits, targets, task="binary")
      unrestricted_response = response_loss(unrestricted_logits, targets, task="binary")
    else:
      weight = torch.as_tensor(float(pos_weight), device=structured_logits.device,
                               dtype=structured_logits.dtype)
      response = F.binary_cross_entropy_with_logits(
        structured_logits.squeeze(-1), targets, pos_weight=weight)
      unrestricted_response = F.binary_cross_entropy_with_logits(
        unrestricted_logits.squeeze(-1), targets, pos_weight=weight)
    spectral = torch.stack(spectral_terms).mean()
    candidates_by_tau = defaultdict(list)
    for occurrence in self.adapter.last_trace.occurrences.values():
      candidates_by_tau[occurrence.tau].append(occurrence.state.candidate)
    candidates_by_tau = {tau: torch.stack(values) for tau, values in candidates_by_tau.items()}
    return batch_cls(
      structured_logits, unrestricted_logits, targets, readers_by_tau, candidates_by_tau,
      response, spectral, unrestricted_response, len(targets))


class TGNLinkOutsideBridge(_TGNOutsideMixin):
  """Training-only continuation contexts for TGN link prediction."""

  def __init__(self, adapter, time_mean=0.0, time_std=1.0, max_nodes_per_scenario=64):
    self.adapter = adapter
    self.prss = adapter.prss
    self.time_mean = float(time_mean)
    self.time_std = max(float(time_std), 1e-12)
    self.max_nodes_per_scenario = int(max_nodes_per_scenario)
    expected_root = self.prss.config.interface(adapter.interface_type).host_dim + 2
    if self.prss.config.root_metadata_dim != expected_root:
      raise ValueError("TGN link root_metadata_dim must be host_dim + 2 ({})".format(expected_root))

  def _root_metadata(self, counterpart, timestamp, candidate_role):
    normalized_time = ((torch.log1p(timestamp.clamp_min(0)) - self.time_mean) /
                       self.time_std).unsqueeze(-1)
    role = torch.full_like(normalized_time, float(candidate_role))
    return torch.cat([counterpart.detach(), normalized_time, role], dim=-1)

  def build(self, timestamps):
    if self.adapter.last_trace is None or self.adapter.last_root_ids is None:
      raise RuntimeError("Enable PRSS trace before the TGN forward pass")
    if self.adapter.last_root_rows is None or len(self.adapter.last_root_rows) != len(
        self.adapter.last_root_quotients):
      raise RuntimeError("Link bridge requires tracing all source/destination/negative roots")
    roots = self.adapter.last_root_ids
    quotients = self.adapter.last_root_quotients
    if len(roots) % 3 != 0:
      raise RuntimeError("Expected TGN roots in source/destination/negative thirds")
    batch_size = len(roots) // 3
    timestamps = torch.as_tensor(timestamps, device=quotients.device, dtype=quotients.dtype)
    if len(timestamps) != batch_size:
      raise ValueError("Timestamp batch length mismatch")
    source_q = quotients[:batch_size]
    positive_q = quotients[batch_size:2 * batch_size]
    negative_q = quotients[2 * batch_size:]
    scenarios = []
    for row in range(batch_size):
      time = timestamps[row]
      scenarios.extend([
        (roots[row], self._root_metadata(positive_q[row], time, 0), 1.0),
        (roots[batch_size + row], self._root_metadata(source_q[row], time, 1), 1.0),
        (roots[row], self._root_metadata(negative_q[row], time, 0), 0.0),
        (roots[2 * batch_size + row], self._root_metadata(source_q[row], time, 1), 0.0),
      ])
    structured_logits, unrestricted_logits, targets = [], [], []
    readers = defaultdict(list)
    spectral_terms = []
    for root_id, metadata, target in scenarios:
      for tau, structured, unrestricted, matrix, candidate in self._outside_for_root(root_id, metadata):
        structured_logits.append(structured)
        unrestricted_logits.append(unrestricted)
        targets.append(target)
        readers[tau].append(matrix)
        spectral_terms.append(self.prss.state_spectral_loss(tau, matrix, candidate))
    return self._pack(structured_logits, unrestricted_logits, targets, readers, spectral_terms,
                      TGNLinkAuxiliaryBatch)


class TGNNodeOutsideBridge(_TGNOutsideMixin):
  """Training-only continuation contexts for dynamic source-node classification.

  The task target is the event-time source label.  Root metadata contains only normalized time and a
  constant role coordinate; it never contains the label or the current destination identity.  Thus
  the reader asks whether an internal historical state remains predictive of the eventual node label
  under the legal upper computation, without a target leak.
  """

  def __init__(self, adapter, time_mean=0.0, time_std=1.0, max_nodes_per_scenario=32):
    self.adapter = adapter
    self.prss = adapter.prss
    self.time_mean = float(time_mean)
    self.time_std = max(float(time_std), 1e-12)
    self.max_nodes_per_scenario = int(max_nodes_per_scenario)
    if self.prss.config.root_metadata_dim != 2:
      raise ValueError("TGN node-classification root_metadata_dim must be 2")

  def _root_metadata(self, timestamp):
    normalized_time = ((torch.log1p(timestamp.clamp_min(0)) - self.time_mean) /
                       self.time_std).unsqueeze(-1)
    constant_role = torch.zeros_like(normalized_time)
    return torch.cat([normalized_time, constant_role], dim=-1)

  def build(self, timestamps, labels, pos_weight=None):
    if self.adapter.last_trace is None or self.adapter.last_root_ids is None:
      raise RuntimeError("Enable PRSS trace before the source-node forward pass")
    rows = self.adapter.last_root_rows
    timestamps = torch.as_tensor(timestamps, device=self.adapter.last_root_quotients.device,
                                 dtype=self.adapter.last_root_quotients.dtype)
    labels = torch.as_tensor(labels, device=timestamps.device, dtype=timestamps.dtype)
    structured_logits, unrestricted_logits, targets = [], [], []
    readers = defaultdict(list)
    spectral_terms = []
    for root_id, row in zip(self.adapter.last_root_ids, rows):
      metadata = self._root_metadata(timestamps[int(row)])
      target = float(labels[int(row)].item())
      for tau, structured, unrestricted, matrix, candidate in self._outside_for_root(root_id, metadata):
        structured_logits.append(structured)
        unrestricted_logits.append(unrestricted)
        targets.append(target)
        readers[tau].append(matrix)
        spectral_terms.append(self.prss.state_spectral_loss(tau, matrix, candidate))
    return self._pack(structured_logits, unrestricted_logits, targets, readers, spectral_terms,
                      TGNNodeAuxiliaryBatch, pos_weight=pos_weight)


def install_tgn_prss(tgn_model, candidate_dim, context_dim=64, interface_prefix="tgn",
                     reader_hidden_dim=128, lambda_resp=1.0, lambda_spec=0.0,
                     gram_ema_rho=0.05, spectral_update_interval=200,
                     spectral_warmup_steps=200, ridge_eps=1e-5,
                     spectral_step_size=0.25, spectral_eigen_floor_ratio=1e-4,
                     max_nodes_per_scenario=32, preagg_summary_dim=32,
                     trace_max_roots=0, task="link"):
  """Install PRSS before training with one quotient interface per TGN recursion depth.

  ``k_tau`` is derived from the live TGN host.  L0, L1, ..., L_L are distinct interface types because
  TGN uses distinct aggregation blocks and the semantic coordinate systems differ by depth.  Sharing
  one B/R across those depths is invalid even when their widths happen to be equal.
  """
  host = tgn_model.embedding_module
  host_dim = int(host.embedding_dimension)
  edge_dim = int(host.n_edge_features)
  if candidate_dim < host_dim:
    raise ValueError("candidate_dim must be >= the host-derived width {}".format(host_dim))
  if task not in ("link", "node"):
    raise ValueError("task must be 'link' or 'node'")
  root_metadata_dim = host_dim + 2 if task == "link" else 2
  interface_specs = {
    "{}_layer_{}".format(interface_prefix, layer): InterfaceSpec(
      "{}_layer_{}".format(interface_prefix, layer),
      raw_dim=host_dim,
      # L0 has no recursive neighborhood to preserve, hence no learned expansion/compression.
      # Keeping d_0=k_0 also makes its quotient exactly identity-compatible for the whole run.
      candidate_dim=(host_dim if layer == 0 else int(candidate_dim)),
      host_dim=host_dim, response_dim=1)
    for layer in range(int(host.n_layers) + 1)
  }
  config = PRSSConfig(
    interfaces=interface_specs,
    context_dim=context_dim,
    root_metadata_dim=root_metadata_dim,
    parent_local_dim=edge_dim + 4,
    relation_count=2,
    relation_dim=16,
    reader_hidden_dim=reader_hidden_dim,
    lambda_resp=lambda_resp,
    lambda_spec=lambda_spec,
    gram_ema_rho=gram_ema_rho,
    spectral_update_interval=spectral_update_interval,
    spectral_warmup_steps=spectral_warmup_steps,
    ridge_eps=ridge_eps,
    spectral_step_size=spectral_step_size,
    spectral_eigen_floor_ratio=spectral_eigen_floor_ratio,
    initialization="identity_like",
  )
  device = host.node_features.device
  system = PRSSSystem(config).to(device)
  # TGN constructs candidates from its true pre-aggregation interface; generic raw->candidate lifts
  # are intentionally unused here and frozen to avoid dead optimizer state.
  for parameter in system.lifts.parameters():
    parameter.requires_grad_(False)
  adapter = TGNRecursiveEmbeddingAdapter(
    host, system, interface_prefix=interface_prefix, preagg_summary_dim=preagg_summary_dim,
    trace_max_roots=trace_max_roots).to(device)
  # Important: tgn_model was usually moved to CUDA *before* installation.  Assigning a fresh adapter
  # afterwards does not recursively move newly-created pre-aggregation lifts unless we do it here.
  tgn_model.embedding_module = adapter
  if task == "link":
    bridge = TGNLinkOutsideBridge(adapter, max_nodes_per_scenario=max_nodes_per_scenario)
  else:
    bridge = TGNNodeOutsideBridge(adapter, max_nodes_per_scenario=max_nodes_per_scenario)
  return adapter, bridge




def compute_source_event_embeddings(tgn_model, source_nodes, destination_nodes, edge_times,
                                    edge_idxs, n_neighbors=20):
  """Compute source embeddings and advance TGN event memory without 3x link-prediction expansion.

  Official ``compute_temporal_embeddings`` computes source, destination and negative embeddings in
  one call because link prediction needs all three.  Dynamic node classification only consumes the
  source embedding.  This helper preserves the same chronological memory/message semantics while
  avoiding unnecessary destination/negative recursive trees.  The default TGN message path uses
  memory states rather than current embeddings; embedding-in-message modes are rejected explicitly.
  """
  if tgn_model.dyrep:
    raise ValueError("Source-only node classification helper does not support dyrep mode")
  if tgn_model.use_source_embedding_in_message or tgn_model.use_destination_embedding_in_message:
    raise ValueError("Source-only helper requires memory-based TGN messages")
  source_nodes = np.asarray(source_nodes)
  destination_nodes = np.asarray(destination_nodes)
  edge_times = np.asarray(edge_times)
  edge_idxs = np.asarray(edge_idxs)

  memory = None
  time_diffs = None
  if tgn_model.use_memory:
    if tgn_model.memory_update_at_start:
      memory, last_update = tgn_model.get_updated_memory(
        list(range(tgn_model.n_nodes)), tgn_model.memory.messages)
    else:
      memory = tgn_model.memory.get_memory(list(range(tgn_model.n_nodes)))
      last_update = tgn_model.memory.last_update
    source_time_diffs = torch.as_tensor(edge_times, device=tgn_model.device).long() - \
      last_update[source_nodes].long()
    source_time_diffs = ((source_time_diffs - tgn_model.mean_time_shift_src) /
                         max(tgn_model.std_time_shift_src, 1e-12))
    time_diffs = source_time_diffs

  source_embedding = tgn_model.embedding_module.compute_embedding(
    memory=memory,
    source_nodes=source_nodes,
    timestamps=edge_times,
    n_layers=tgn_model.n_layers,
    n_neighbors=n_neighbors,
    time_diffs=time_diffs)

  if tgn_model.use_memory:
    positives = np.concatenate([source_nodes, destination_nodes])
    if tgn_model.memory_update_at_start:
      tgn_model.update_memory(positives, tgn_model.memory.messages)
      if not torch.allclose(memory[positives], tgn_model.memory.get_memory(positives), atol=1e-5):
        raise RuntimeError("TGN memory replay mismatch in source-only node task")
      tgn_model.memory.clear_messages(positives)

    # Current embeddings are not used by the default memory-based message configuration, but the
    # original API requires placeholders.
    dummy_destination = torch.zeros_like(source_embedding)
    unique_sources, source_messages = tgn_model.get_raw_messages(
      source_nodes, source_embedding, destination_nodes, dummy_destination,
      edge_times, edge_idxs)
    unique_destinations, destination_messages = tgn_model.get_raw_messages(
      destination_nodes, dummy_destination, source_nodes, source_embedding,
      edge_times, edge_idxs)
    if tgn_model.memory_update_at_start:
      tgn_model.memory.store_raw_messages(unique_sources, source_messages)
      tgn_model.memory.store_raw_messages(unique_destinations, destination_messages)
    else:
      tgn_model.update_memory(unique_sources, source_messages)
      tgn_model.update_memory(unique_destinations, destination_messages)

  return source_embedding
