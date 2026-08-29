"""Diagnose the KF-vs-task gradient direction and the radial push.

Two questions behind the lambda scan result (task-only 0.867 > every
lambda>0 run):

1. Does the KF surrogate gradient point WITH or AGAINST the task-loss
   gradient on the same parameters?
2. Does a pure KF step rescale z without changing J (the radial degree
   of freedom of the lagged linearization), i.e. does it silently hurt
   the decoder input?

Usage: run on the cloud with the same config as the lambda scan.
"""
import sys
sys.path.insert(0, '/root/autodl-tmp/PRSS2/src')
import numpy as np, torch, random
random.seed(0); np.random.seed(0); torch.manual_seed(0)
from rpbe.data.jodie import JodieDataset
from rpbe.hosts.official_tgn import TGN, get_neighbor_finder, MLP
from rpbe.hosts.jodie_tgn import JodieTGNAdapter, TAU_TEMPLATE
from rpbe.config import RPBConfig
from rpbe.records import build_edge_tables, JodieCutBuilder, JodieFutureIndex, NODE_CLASS
from rpbe.maps import FixedMaps
from rpbe.compressor import RecursiveCompressor
from rpbe.loss import KFLaggedWindow, kf_score, kf_vjp_batch
from rpbe.training.jodie_loop import select_trace_rows

ds = JodieDataset('wikipedia',
                  data_dir='/root/autodl-tmp/PRSS2/old/processed_tgn_data',
                  use_validation=True)
full, train, _, _ = ds.splits()
finder = get_neighbor_finder(train, uniform=False,
                             max_node_idx=max(full.unique_nodes))
ms, ss, md, sd = ds.time_stats()
device = torch.device('cuda:0')
tgn = TGN(neighbor_finder=finder, node_features=ds.node_features,
          edge_features=ds.edge_features, device=device, n_layers=3,
          n_heads=2, dropout=0.1, use_memory=True, message_dimension=100,
          memory_dimension=172, memory_update_at_start=True,
          embedding_module_type='graph_attention', message_function='identity',
          aggregator_type='last', n_neighbors=5, mean_time_shift_src=ms,
          std_time_shift_src=ss, mean_time_shift_dst=md,
          std_time_shift_dst=sd).to(device)
taus = [TAU_TEMPLATE.format(l) for l in range(4)]
cfg = RPBConfig(state_dims={t: 172 for t in taus},
                own_dims={t: 172 for t in taus}, m=64, rpbe_seed=0,
                delta_t_scale=1e6, cuts_per_tau=1024, kf_min_abs=1024,
                kf_taus=taus[:3])
comp = RecursiveCompressor(cfg).to(device)
adapter = JodieTGNAdapter(tgn.embedding_module, comp, n_neighbors=5)
tgn.embedding_module = adapter
maps = FixedMaps(cfg).to(device)
index = JodieFutureIndex(train)
builder = JodieCutBuilder(index, stage=NODE_CLASS, seed=0,
                          cuts_per_tau=1024)
window = KFLaggedWindow({t: 172 for t in taus[:3]}, min_ratio=2.0,
                        min_abs=1024, eps=1e-4, fixed_maps=maps)
decoder = MLP(dim=172, drop=0.1).to(device)
opt = torch.optim.Adam(list(tgn.parameters()) + list(comp.parameters())
                       + list(decoder.parameters()), lr=3e-4)

# Warm the lagged reference window.
k = 0
while window.reference_score("tjo:layer1") is None and k < 40:
    s, e = k * 200, min(len(train.sources), (k + 1) * 200)
    srcs = train.sources[s:e]
    dsts = train.destinations[s:e]
    tms = train.timestamps[s:e]
    eis = train.edge_idxs[s:e]
    size = len(srcs)
    trace_rows = select_trace_rows(np.zeros(size), 32, 0, k, 'evenly_spaced')
    adapter.set_trace_source_rows(trace_rows)
    with torch.no_grad():
        tgn.compute_temporal_embeddings(srcs, dsts, dsts, tms, eis, 5)
    if adapter.trace is not None:
        cuts = builder.build(adapter.trace, batch_seed=k)
        window.step(cuts)
    k += 1
print('reference warmed after', k, 'batches; score:',
      window.reference_score("tjo:layer1"))

# Diagnosis batches: compare task vs KF gradients, and the radial push.
comp_params = list(comp.parameters())
cos_list, ratio_list = [], []
z_norm_before, z_norm_after, j_before, j_after = [], [], [], []
for diag_batch in range(4):
    s = (k + diag_batch) * 200
    e = min(len(train.sources), (k + diag_batch + 1) * 200)
    srcs = train.sources[s:e]
    dsts = train.destinations[s:e]
    tms = train.timestamps[s:e]
    eis = train.edge_idxs[s:e]
    labels_np = train.labels[s:e]
    size = len(srcs)
    trace_rows = select_trace_rows(np.zeros(size), 32, 0, k + diag_batch,
                                   'evenly_spaced')
    adapter.set_trace_source_rows(trace_rows)
    labels_t = torch.from_numpy(labels_np).float().to(device)
    src_emb, _, _ = tgn.compute_temporal_embeddings(
        srcs, dsts, dsts, tms, eis, 5)
    pred = decoder(src_emb).sigmoid()
    task_loss = torch.nn.functional.binary_cross_entropy(pred, labels_t)
    g_task = torch.autograd.grad(task_loss, comp_params, retain_graph=True,
                                 allow_unused=True)
    cuts = builder.build(adapter.trace, batch_seed=k + diag_batch)
    _, surrogates, _, _, _ = window.step(cuts)
    aux = sum(surrogates.values()) if surrogates else None
    if aux is not None:
        g_kf = torch.autograd.grad(-aux, comp_params, retain_graph=True,
                                   allow_unused=True)
        tn = float(sum(t.norm() for t in g_task if t is not None))
        kn = float(sum(t.norm() for t in g_kf if t is not None))
        dot = float(sum((a * b).sum() for a, b in zip(g_task, g_kf)
                        if a is not None and b is not None))
        cos = dot / (tn * kn) if tn > 0 and kn > 0 else float('nan')
        cos_list.append(cos)
        ratio_list.append(kn / max(tn, 1e-9))
        print('batch', diag_batch, 'cos(task, kf) = {:.4f}, '
              'kf/task norm ratio = {:.3f}'.format(cos, kn / max(tn, 1e-9)))

    # Radial push: one pure KF step on a copy of the batch z.
    if aux is not None and cuts:
        zs = [c.z for c in cuts[:64]]
        z_stack = torch.stack(zs)
        ps = torch.stack([maps.pv(c.context, c.outcome) for c in cuts[:64]])
        z_leaf = z_stack.detach().clone().requires_grad_(True)
        w = torch.tensor([c.weight for c in cuts[:64]],
                         dtype=torch.float64, device=device)
        j0 = float(kf_score(z_leaf, ps, eps=1e-4))
        # Recompute the batch VJP against the reference for the same rows.
        ref = window._reference["tjo:layer1"]
        raw = kf_vjp_batch(z_leaf, ps, w, ref["mu_z"], ref["mu_p"],
                           ref["adjoints"])
        opt_z = torch.optim.Adam([z_leaf], lr=0.05)
        opt_z.zero_grad()
        (-raw).backward()
        opt_z.step()
        j1 = float(kf_score(z_leaf.detach(), ps, eps=1e-4))
        j_before.append(j0)
        j_after.append(j1)
        z_norm_before.append(float(z_stack.norm()))
        z_norm_after.append(float(z_leaf.detach().norm()))
        print('batch', diag_batch, 'pure-KF step: J {:.3f} -> {:.3f}, '
              '||z|| {:.3f} -> {:.3f}'.format(
                  j0, j1, z_norm_before[-1], z_norm_after[-1]))

print()
print('mean cos(task, kf) = {:.4f}'.format(float(np.mean(cos_list))))
print('mean kf/task norm ratio = {:.3f}'.format(float(np.mean(ratio_list))))
print('mean J change (pure KF step) = {:.4f}'.format(
    float(np.mean(j_after)) - float(np.mean(j_before))))
print('mean ||z|| change (pure KF step) = {:.4f}'.format(
    float(np.mean(z_norm_after)) - float(np.mean(z_norm_before))))
