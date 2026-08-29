"""One KF window gradient diagnosis (seventh review): uncut task vs KF
gradient norms and cosine on the compressor parameters."""
import sys
sys.path.insert(0, '/root/autodl-tmp/PRSS2/src')
import numpy as np, torch, random
random.seed(0); np.random.seed(0); torch.manual_seed(0)
from rpbe.data.jodie import JodieDataset
from rpbe.hosts.official_tgn import TGN, get_neighbor_finder
from rpbe.hosts.jodie_tgn import JodieTGNAdapter, TAU_TEMPLATE
from rpbe.config import RPBConfig
from rpbe.records import build_edge_tables, JodieCutBuilder, NODE_CLASS
from rpbe.maps import FixedMaps
from rpbe.compressor import RecursiveCompressor
from rpbe.loss import KFMomentWindow
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
ep, lb, us, pg, _ = build_edge_tables(ds)
tables = (ep, lb, us, pg)
taus = [TAU_TEMPLATE.format(l) for l in range(4)]
cfg = RPBConfig(state_dims={t: 172 for t in taus},
                own_dims={t: 172 for t in taus}, m=64, rpbe_seed=0,
                delta_t_scale=1e6, cuts_per_tau=1024, kf_min_abs=1024,
                kf_taus=taus[:3])
comp = RecursiveCompressor(cfg).to(device)
adapter = JodieTGNAdapter(tgn.embedding_module, comp, n_neighbors=5,
                          edge_tables=tables)
tgn.embedding_module = adapter
maps = FixedMaps(cfg).to(device)
builder = JodieCutBuilder(tables, stage=NODE_CLASS, seed=0,
                          cuts_per_tau=1024)
window = KFMomentWindow({t: 172 for t in taus[:3]}, min_ratio=2.0,
                        min_abs=1024, eps=1e-4, fixed_maps=maps,
                        autoclose=False)

shadow = None
window_batches = []
k = 0
while not window.window_ready() and k < 30:
    s, e = k * 200, min(len(train.sources), (k + 1) * 200)
    srcs = train.sources[s:e]
    dsts = train.destinations[s:e]
    tms = train.timestamps[s:e]
    eis = train.edge_idxs[s:e]
    size = len(srcs)
    trace_rows = select_trace_rows(np.zeros(size), 32, 0, k, 'evenly_spaced')
    adapter.set_trace_source_rows(trace_rows)
    with torch.no_grad():
        negs = np.random.choice(train.destinations, size=size, replace=True)
        tgn.compute_edge_probabilities(srcs, dsts, negs, tms, eis, 5)
    revs = {int(r): {'counterpart': int(dsts[r]), 'label': 1.0,
                     'time': float(tms[r]), 'event_idx': int(eis[r]),
                     'role': 0} for r in trace_rows}
    cuts = builder.build(adapter.trace, root_events=revs, batch_seed=k)
    window.add(cuts)
    window_batches.append((srcs, dsts, tms, eis, negs, trace_rows, k))
    if shadow is None:
        shadow = tgn.memory.backup_memory()
    k += 1
print('window ready after', k, 'batches')
closed, replay_plan, diag = window.close_replay()
print('J:', {t: round(v, 3) for t, v in closed.items()})
tgn.memory.restore_memory(shadow)
kf_params = list(comp.parameters())
def add_grads(ga, gb):
    return [None if a is None and b is None else
            (a if b is None else (b if a is None else a + b))
            for a, b in zip(ga, gb)]
task_norms, kf_norms, cosines = [], [], []
for bi, (src2, dst2, t2, e2, neg2, tr2, step2) in enumerate(window_batches):
    adapter.set_trace_source_rows(tr2)
    pp, np_ = tgn.compute_edge_probabilities(src2, dst2, neg2, t2, e2, 5)
    link_loss = (torch.nn.functional.binary_cross_entropy(
        pp.squeeze(), torch.ones(len(src2), device=device))
        + torch.nn.functional.binary_cross_entropy(
            np_.squeeze(), torch.zeros(len(src2), device=device)))
    g_task = torch.autograd.grad(link_loss, kf_params, retain_graph=True,
                                 allow_unused=True)
    g_kf = None
    for tau, plan in replay_plan.items():
        if bi >= len(plan["by_batch"]):
            continue
        for (occ_id, g) in plan["by_batch"][bi]:
            z = adapter.trace.occurrences[occ_id].state.z
            term = -(g * z.float()).sum()      # MINUS: maximize J
            g_i = torch.autograd.grad(term, kf_params, retain_graph=True,
                                      allow_unused=True)
            g_kf = g_i if g_kf is None else add_grads(g_kf, g_i)
    tn = float(sum(t.norm() for t in g_task if t is not None))
    kn = float(sum(t.norm() for t in g_kf if t is not None))
    dot = float(sum((a * b).sum() for a, b in zip(g_task, g_kf)
                    if a is not None and b is not None))
    cos = dot / (tn * kn) if tn > 0 and kn > 0 else float('nan')
    task_norms.append(tn)
    kf_norms.append(kn)
    cosines.append(cos)
print('per batch: task_norm kf_norm cos')
for bi in range(len(window_batches)):
    print('  batch', bi, 'task={:.3f} kf={:.3f} cos={:.3f}'.format(
        task_norms[bi], kf_norms[bi], cosines[bi]))
print('mean task_norm {:.3f}, mean kf_norm {:.3f}, mean cos {:.3f}, '
      'ratio kf/task {:.3f}'.format(
          float(np.mean(task_norms)), float(np.mean(kf_norms)),
          float(np.mean(cosines)),
          float(np.mean(kf_norms)) / max(float(np.mean(task_norms)), 1e-9)))
