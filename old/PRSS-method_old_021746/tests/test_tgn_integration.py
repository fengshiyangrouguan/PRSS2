import sys
from pathlib import Path

import numpy as np
import pytest
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
TGN_DIR = WORKSPACE / "tgn"
DATA_DIR = WORKSPACE / "processed_tgn_data"
sys.path.insert(0, str(TGN_DIR))

from model.tgn import TGN  # noqa: E402
from utils.data_processing import get_data  # noqa: E402
from utils.utils import get_neighbor_finder  # noqa: E402

from prss.integrations.tgn import install_tgn_prss  # noqa: E402


@pytest.mark.skipif(not (DATA_DIR / "ml_wikipedia.csv").exists(),
                    reason="real preprocessed Wikipedia data is unavailable")
def test_real_wikipedia_tgn_parent_only_sees_prss_width_and_auxiliary_is_train_only():
  node_features, edge_features, full_data, train_data, _, _, _, _ = get_data(
    "wikipedia", data_dir=str(DATA_DIR))
  finder = get_neighbor_finder(train_data, uniform=False)
  device = torch.device("cpu")
  tgn = TGN(
    neighbor_finder=finder,
    node_features=node_features,
    edge_features=edge_features,
    device=device,
    n_layers=1,
    n_heads=2,
    dropout=0.0,
    use_memory=False,
    n_neighbors=3,
  ).to(device)
  adapter, bridge = install_tgn_prss(
    tgn, candidate_dim=192, context_dim=32, reader_hidden_dim=48,
    spectral_update_interval=1, spectral_warmup_steps=0,
    max_nodes_per_scenario=16)
  adapter.enable_trace(True)
  indexes = np.arange(1000, 1004)
  sources = train_data.sources[indexes]
  destinations = train_data.destinations[indexes]
  negatives = np.roll(destinations, 1)
  timestamps = train_data.timestamps[indexes]
  edge_idxs = train_data.edge_idxs[indexes]

  source_h, destination_h, negative_h = tgn.compute_temporal_embeddings(
    sources, destinations, negatives, timestamps, edge_idxs, n_neighbors=3)
  assert source_h.shape == (4, 172)
  assert destination_h.shape == (4, 172)
  assert negative_h.shape == (4, 172)
  assert adapter.interface_types == ["tgn_layer_0", "tgn_layer_1"]
  assert adapter.prss.config.interface("tgn_layer_0").host_dim == 172
  assert adapter.prss.config.interface("tgn_layer_0").candidate_dim == 172
  assert adapter.prss.config.interface("tgn_layer_1").candidate_dim == 192
  # Recursive layer 1 starts from R=[I,0]; L0 is exact identity compatibility mode.
  leaf_R = adapter.prss.quotients.state_for("tgn_layer_0").R
  initial_R = adapter.prss.quotients.state_for("tgn_layer_1").R
  assert torch.allclose(leaf_R, torch.eye(172), atol=1e-7)
  assert torch.allclose(initial_R[:, :172], torch.eye(172), atol=1e-7)
  assert torch.count_nonzero(initial_R[:, 172:]) == 0
  assert adapter.last_trace is not None
  assert all(occurrence.state.quotient.shape[-1] == 172
             for occurrence in adapter.last_trace.occurrences.values())
  for occurrence in adapter.last_trace.occurrences.values():
    expected = 172 if occurrence.tau == "tgn_layer_0" else 192
    assert occurrence.state.candidate.shape[-1] == expected

  auxiliary = bridge.build(timestamps)
  assert auxiliary.occurrence_count > 0
  assert auxiliary.candidates_by_tau["tgn_layer_0"].shape[-1] == 172
  assert auxiliary.candidates_by_tau["tgn_layer_1"].shape[-1] == 192
  assert torch.isfinite(auxiliary.response)
  assert torch.isfinite(auxiliary.spectral)
  task_logits = tgn.affinity_score(source_h, destination_h).view(-1)
  task_loss = torch.nn.functional.binary_cross_entropy_with_logits(
    task_logits, torch.ones_like(task_logits))
  loss = task_loss + auxiliary.response + 0.1 * auxiliary.spectral
  loss.backward()
  assert adapter.prss.quotients.state_for("tgn_layer_1").R.grad is None
  integration_lift_parameters = list(adapter.preagg_lifts.parameters()) + list(adapter.leaf_lift.parameters())
  assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
             for parameter in integration_lift_parameters)
  assert any(parameter.grad is not None for parameter in adapter.prss.readers.parameters())

  # The final metadata coordinate is only source/candidate-tree role. Changing the
  # counterpart never creates a positive/negative-label channel.
  time = torch.tensor(float(timestamps[0]))
  positive_role = bridge._root_metadata(source_h[0], time, candidate_role=1)
  negative_role = bridge._root_metadata(negative_h[0], time, candidate_role=1)
  assert positive_role[-1].item() == negative_role[-1].item() == 1.0

  adapter.prss.train()
  adapter.prss.update_spectral_statistics(auxiliary.readers_by_tau)
  before_leaf = int(adapter.prss.quotients.state_for("tgn_layer_0").spectral_updates)
  before_updates = int(adapter.prss.quotients.state_for("tgn_layer_1").spectral_updates)
  assert adapter.prss.maybe_spectral_update(0)
  assert int(adapter.prss.quotients.state_for("tgn_layer_0").spectral_updates) == before_leaf
  assert int(adapter.prss.quotients.state_for("tgn_layer_1").spectral_updates) == before_updates + 1

  adapter.enable_trace(False)
  adapter.prss.eval()
  adapter.prss.set_spectral_updates_allowed(False)
  with torch.no_grad():
    inference = tgn.compute_temporal_embeddings(
      sources, destinations, negatives, timestamps, edge_idxs, n_neighbors=3)[0]
  assert inference.shape == (4, 172)
  assert adapter.last_trace is None
  with pytest.raises(RuntimeError):
    adapter.prss.update_spectral_statistics(auxiliary.readers_by_tau)


def test_preaggregation_candidate_uses_information_not_present_in_fixed_raw_output():
  from prss.integrations.tgn import TGNPreAggregationCandidateLift

  torch.manual_seed(17)
  lift = TGNPreAggregationCandidateLift(
    host_dim=6, time_dim=6, edge_dim=5, candidate_dim=11, summary_dim=8).eval()
  batch, neighbors = 3, 4
  raw = torch.randn(batch, 6)
  source = torch.randn(batch, 6)
  source_time = torch.randn(batch, 1, 6)
  neighbor = torch.randn(batch, neighbors, 6)
  edge_time = torch.randn(batch, neighbors, 6)
  edge = torch.randn(batch, neighbors, 5)
  mask = torch.zeros(batch, neighbors, dtype=torch.bool)

  candidate_a = lift(raw, source, source_time, neighbor, edge_time, edge, mask, 0.5)
  changed_neighbor = neighbor.clone()
  changed_neighbor[:, 0, :] += 3.0
  candidate_b = lift(raw, source, source_time, changed_neighbor, edge_time, edge, mask, 0.5)

  # raw is deliberately identical.  If Phi were the invalid post-bottleneck k->d expansion,
  # these candidates would be equal; a true pre-aggregation candidate must react to the history.
  assert torch.equal(raw, raw.clone())
  assert candidate_a.shape == candidate_b.shape == (batch, 11)
  assert torch.allclose(candidate_a[:, :6], raw)
  assert torch.allclose(candidate_b[:, :6], raw)
  assert not torch.allclose(candidate_a[:, 6:], candidate_b[:, 6:])


@pytest.mark.skipif(not (DATA_DIR / "ml_wikipedia.csv").exists(),
                    reason="real preprocessed Wikipedia data is unavailable")
def test_source_only_node_forward_matches_official_tgn_memory_semantics():
  from prss.integrations.tgn import compute_source_event_embeddings

  node_features, edge_features, full_data, train_data, _, _, _, _ = get_data(
    "wikipedia", data_dir=str(DATA_DIR))
  max_node = int(max(full_data.sources.max(), full_data.destinations.max()))
  finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_node)
  device = torch.device("cpu")

  def make_tgn():
    return TGN(
      neighbor_finder=finder, node_features=node_features, edge_features=edge_features,
      device=device, n_layers=1, n_heads=2, dropout=0.0, use_memory=True,
      memory_dimension=172, message_dimension=100, embedding_module_type="graph_attention",
      message_function="identity", aggregator_type="last", n_neighbors=2,
    ).to(device).eval()

  torch.manual_seed(123)
  official = make_tgn()
  torch.manual_seed(123)
  source_only = make_tgn()
  source_only.load_state_dict(official.state_dict())
  official.memory.__init_memory__()
  source_only.memory.__init_memory__()

  for indexes in (np.arange(1000, 1004), np.arange(1004, 1008)):
    sources = train_data.sources[indexes]
    destinations = train_data.destinations[indexes]
    timestamps = train_data.timestamps[indexes]
    edge_idxs = train_data.edge_idxs[indexes]
    with torch.no_grad():
      expected, _, _ = official.compute_temporal_embeddings(
        sources, destinations, destinations, timestamps, edge_idxs, n_neighbors=2)
      actual = compute_source_event_embeddings(
        source_only, sources, destinations, timestamps, edge_idxs, n_neighbors=2)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(source_only.memory.memory, official.memory.memory, atol=1e-6, rtol=1e-5)
    assert torch.allclose(source_only.memory.last_update, official.memory.last_update,
                          atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not (DATA_DIR / "ml_wikipedia.csv").exists(),
                    reason="real preprocessed Wikipedia data is unavailable")
def test_node_classification_bridge_has_no_label_in_root_metadata_and_is_finite():
  from prss.integrations.tgn import compute_source_event_embeddings

  node_features, edge_features, full_data, train_data, _, _, _, _ = get_data(
    "wikipedia", data_dir=str(DATA_DIR))
  max_node = int(max(full_data.sources.max(), full_data.destinations.max()))
  finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_node)
  device = torch.device("cpu")
  tgn = TGN(
    neighbor_finder=finder, node_features=node_features, edge_features=edge_features,
    device=device, n_layers=1, n_heads=2, dropout=0.0, use_memory=False, n_neighbors=2,
  ).to(device)
  adapter, bridge = install_tgn_prss(
    tgn, candidate_dim=192, context_dim=24, reader_hidden_dim=32,
    spectral_update_interval=2, spectral_warmup_steps=0,
    max_nodes_per_scenario=8, trace_max_roots=2, task="node")
  adapter.enable_trace(True)
  indexes = np.arange(1000, 1004)
  sources = train_data.sources[indexes]
  destinations = train_data.destinations[indexes]
  timestamps = train_data.timestamps[indexes]
  edge_idxs = train_data.edge_idxs[indexes]
  _ = compute_source_event_embeddings(
    tgn, sources, destinations, timestamps, edge_idxs, n_neighbors=2)

  labels_a = np.zeros(4, dtype=np.float32)
  labels_b = np.ones(4, dtype=np.float32)
  # Root metadata depends only on timestamp/constant role, never on either label choice.
  ts = torch.tensor(float(timestamps[0]))
  assert torch.equal(bridge._root_metadata(ts), bridge._root_metadata(ts))
  auxiliary = bridge.build(timestamps, labels_a, pos_weight=1.0)
  assert auxiliary.occurrence_count > 0
  assert torch.isfinite(auxiliary.response)
  assert torch.isfinite(auxiliary.spectral)
  # Changing labels changes only the supervised target, not the outside context constructor.
  auxiliary_b = bridge.build(timestamps, labels_b, pos_weight=1.0)
  assert not torch.equal(auxiliary.targets, auxiliary_b.targets)


def test_preaggregation_lift_has_finite_gradients_with_empty_and_constant_neighborhoods():
  from prss.integrations.tgn import TGNPreAggregationCandidateLift

  torch.manual_seed(29)
  lift = TGNPreAggregationCandidateLift(6, 6, 5, 11, summary_dim=8)
  raw = torch.randn(2, 6, requires_grad=True)
  source = torch.randn(2, 6, requires_grad=True)
  source_time = torch.randn(2, 1, 6, requires_grad=True)
  neighbor = torch.zeros(2, 4, 6, requires_grad=True)
  edge_time = torch.zeros(2, 4, 6, requires_grad=True)
  edge = torch.zeros(2, 4, 5, requires_grad=True)
  # Row 0 is completely padded; row 1 is constant-valued, another zero-variance case.
  mask = torch.tensor([[True, True, True, True], [False, False, False, False]])
  out = lift(raw, source, source_time, neighbor, edge_time, edge, mask, 0.5)
  out.square().mean().backward()
  tensors = [raw, source, neighbor, edge_time, edge]
  assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in tensors)
  # source_time is deliberately excluded: in TGN it is the constant zero-lag encoding.
  assert source_time.grad is None
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in lift.parameters())


def test_joint_history_lift_keeps_neighbor_edge_pairing_information():
  from prss.integrations.tgn import TGNPreAggregationCandidateLift

  torch.manual_seed(314)
  lift = TGNPreAggregationCandidateLift(4, 4, 3, 8, summary_dim=8).eval()
  raw = torch.randn(1, 4)
  source = torch.randn(1, 4)
  source_time = torch.zeros(1, 1, 4)
  neighbor = torch.tensor([[[1., 0., 0., 0.], [0., 2., 0., 0.]]])
  edge_time = torch.tensor([[[0., 0., 1., 0.], [0., 0., 0., 2.]]])
  edge = torch.tensor([[[1., 3., 0.], [2., 0., 4.]]])
  mask = torch.zeros(1, 2, dtype=torch.bool)
  a = lift(raw, source, source_time, neighbor, edge_time, edge, mask, 0.5)
  # Preserve every modality's separate multiset but break which edge/time belongs to which neighbor.
  b = lift(raw, source, source_time, neighbor, edge_time.flip(1), edge.flip(1), mask, 0.5)
  assert torch.allclose(a[:, :4], b[:, :4])
  assert not torch.allclose(a[:, 4:], b[:, 4:])


@pytest.mark.skipif(not (DATA_DIR / "ml_wikipedia.csv").exists(),
                    reason="real preprocessed Wikipedia data is unavailable")
def test_each_recursive_depth_has_its_own_candidate_lift_reader_and_projection():
  node_features, edge_features, full_data, train_data, _, _, _, _ = get_data(
    "wikipedia", data_dir=str(DATA_DIR))
  max_node = int(max(full_data.sources.max(), full_data.destinations.max()))
  finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_node)
  tgn = TGN(
    neighbor_finder=finder, node_features=node_features, edge_features=edge_features,
    device=torch.device("cpu"), n_layers=2, n_heads=2, dropout=0.0,
    use_memory=False, n_neighbors=1).cpu()
  adapter, _ = install_tgn_prss(tgn, candidate_dim=192, task="node")
  assert adapter.interface_types == ["tgn_layer_0", "tgn_layer_1", "tgn_layer_2"]
  assert adapter.preagg_lifts["1"] is not adapter.preagg_lifts["2"]
  s1 = adapter.prss.quotients.state_for("tgn_layer_1")
  s2 = adapter.prss.quotients.state_for("tgn_layer_2")
  assert s1 is not s2 and s1.R.data_ptr() != s2.R.data_ptr()
  r1 = adapter.prss.readers[adapter.prss._key("tgn_layer_1")]
  r2 = adapter.prss.readers[adapter.prss._key("tgn_layer_2")]
  assert r1 is not r2
  assert all(parameter.device.type == "cpu" for parameter in adapter.parameters())


def test_forced_trace_rows_are_exact_and_one_shot():
  # Unit-test the root selection contract without constructing a full TGN.
  from prss.integrations.tgn import TGNRecursiveEmbeddingAdapter
  adapter = object.__new__(TGNRecursiveEmbeddingAdapter)
  torch.nn.Module.__init__(adapter)
  adapter.trace_enabled = True
  adapter.trace_max_roots = 2
  adapter._forced_trace_rows = None
  adapter.set_trace_rows([1, 4])
  first = adapter._root_record_mask(6)
  assert np.flatnonzero(first).tolist() == [1, 4]
  # The next batch must fall back to normal sampling, never reuse prior labels/rows.
  second = adapter._root_record_mask(6)
  assert np.flatnonzero(second).tolist() == [0, 5]


@pytest.mark.skipif(not (DATA_DIR / "ml_wikipedia.csv").exists(),
                    reason="real preprocessed Wikipedia data is unavailable")
def test_multilayer_prss_is_exactly_vanilla_compatible_at_initialization():
  node_features, edge_features, full_data, train_data, _, _, _, _ = get_data(
    "wikipedia", data_dir=str(DATA_DIR))
  max_node = int(max(full_data.sources.max(), full_data.destinations.max()))
  finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_node)
  device = torch.device("cpu")

  def make_model():
    return TGN(
      neighbor_finder=finder, node_features=node_features, edge_features=edge_features,
      device=device, n_layers=2, n_heads=2, dropout=0.0, use_memory=False,
      n_neighbors=2).to(device).eval()

  torch.manual_seed(777)
  vanilla = make_model()
  prss_model = make_model()
  prss_model.load_state_dict(vanilla.state_dict())
  adapter, _ = install_tgn_prss(prss_model, candidate_dim=192, task="node")
  adapter.enable_trace(False)
  indexes = np.arange(1200, 1204)
  sources = train_data.sources[indexes]
  destinations = train_data.destinations[indexes]
  negatives = np.roll(destinations, 1)
  timestamps = train_data.timestamps[indexes]
  edge_idxs = train_data.edge_idxs[indexes]
  with torch.no_grad():
    expected = vanilla.compute_temporal_embeddings(
      sources, destinations, negatives, timestamps, edge_idxs, n_neighbors=2)
    actual = prss_model.compute_temporal_embeddings(
      sources, destinations, negatives, timestamps, edge_idxs, n_neighbors=2)
  for x, y in zip(expected, actual):
    assert torch.allclose(x, y, atol=1e-7, rtol=1e-6)
