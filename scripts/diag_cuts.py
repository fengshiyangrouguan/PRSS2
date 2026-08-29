"""Diagnose per-tau cut yields for the t1 (pretrain NL=3) configuration."""
import sys
sys.path.insert(0, '/root/autodl-tmp/PRSS2/src')
import numpy as np, torch, random
from collections import Counter
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
import time
t_start = time.perf_counter()
last = t_start
for k in range(30):
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
    now = time.perf_counter()
    if k % 5 == 4:
        print('batch', k, 'cuts_by_tau', dict(Counter(c.tau for c in cuts)),
              'cum_s', round(now - t_start, 1),
              'per_batch_ms', round((now - t_start) / (k + 1) * 1000, 1),
              'cache_entries', len(maps._p_cache))
    last = now
