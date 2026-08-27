#!/usr/bin/env python3
"""Per-stage timing probe for the latent-adjoint replay loop.

Every stage boundary calls ``torch.cuda.synchronize()`` so CUDA async
kernels are attributed to the stage that LAUNCHED them (sixth review:
exclusive phase sum must close against wall time).  The pass-2 loop
mirrors the real latent-adjoint structure: no pv, no moments, no re-walk.
"""
import sys, time
sys.path.insert(0, '/root/autodl-tmp/PRSS2/src')
import numpy as np, torch, random
random.seed(0); np.random.seed(0); torch.manual_seed(0)

BS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
NL = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ND = int(sys.argv[3]) if len(sys.argv) > 3 else 10
VANILLA = (len(sys.argv) > 4 and sys.argv[4] == "vanilla")

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
          edge_features=ds.edge_features, device=device, n_layers=NL,
          n_heads=2, dropout=0.1, use_memory=True, message_dimension=100,
          memory_dimension=172, memory_update_at_start=True,
          embedding_module_type='graph_attention', message_function='identity',
          aggregator_type='last', n_neighbors=ND, mean_time_shift_src=ms,
          std_time_shift_src=ss, mean_time_shift_dst=md,
          std_time_shift_dst=sd).to(device)
endpoints, labels_tbl, users, pages, _ = build_edge_tables(ds)
tables = (endpoints, labels_tbl, users, pages)
taus = [TAU_TEMPLATE.format(l) for l in range(NL + 1)]
cfg = RPBConfig(state_dims={t: 172 for t in taus},
                own_dims={t: 172 for t in taus}, m=64, rpbe_seed=0,
                delta_t_scale=1e6, cuts_per_tau=1024, kf_min_abs=1024,
                kf_taus=taus[:NL])
compressor = RecursiveCompressor(cfg).to(device)
adapter = JodieTGNAdapter(tgn.embedding_module, compressor, n_neighbors=ND,
                          edge_tables=tables)
tgn.embedding_module = adapter
maps = FixedMaps(cfg).to(device)
builder = JodieCutBuilder(tables, stage=NODE_CLASS, seed=0,
                          cuts_per_tau=1024)
window = KFMomentWindow({t: 172 for t in taus[:NL]}, min_ratio=2.0,
                        min_abs=1024, eps=1e-4, fixed_maps=maps,
                        autoclose=False)
opt = torch.optim.Adam(list(tgn.parameters())
                       + list(compressor.parameters()), lr=3e-4)

times = {}
_last = None

def acc(name):
    global _last
    torch.cuda.synchronize()
    now = time.perf_counter()
    if _last is not None:
        times[name] = times.get(name, 0.0) + (now - _last)
    _last = now

N = 12
wins = 0
window_batches = []
shadow = None
from rpbe.hosts.official_tgn import MLP
decoder = MLP(dim=172, drop=0.1).to(device)
acc('start')
if VANILLA:
    # Official rhythm: ONE forward (no trace, no adapter hook) + task BCE
    # + ONE backward + step per batch.  The adapter is still installed but
    # traces nothing (empty trace set), so the host path is untouched.
    for k in range(N):
        s, e = k * BS, min(len(train.sources), (k + 1) * BS)
        srcs = train.sources[s:e]
        dsts = train.destinations[s:e]
        tms = train.timestamps[s:e]
        eis = train.edge_idxs[s:e]
        lbs = train.labels[s:e]
        opt.zero_grad(set_to_none=True)
        emb, _, _ = tgn.compute_temporal_embeddings(
            srcs, dsts, dsts, tms, eis, ND)
        logits = decoder(emb)
        acc('vanilla_fwd')
        loss = torch.nn.functional.binary_cross_entropy(
            logits.sigmoid(),
            torch.from_numpy(lbs).float().to(device))
        loss.backward()
        acc('vanilla_backward')
        tgn.memory.detach_memory()
        opt.step()
        acc('vanilla_step')
    acc('end')
else:
    # RPBE rounds: round 1 warms the fixed-measurement p cache (epoch 1
    # cost); round 2 measures the epoch-2+ speed with cache hits.
    def run_rpbe_round(measure):
        global _last, wins, shadow, window_batches
        _last = None
        wins = 0
        window_batches = []
        shadow = None
        window.reset()
        tgn.memory.__init_memory__()   # epoch-start reset (real loop does this)
        if measure:
            times.clear()
            acc('start')
        for k in range(N):
            s, e = k * BS, min(len(train.sources), (k + 1) * BS)
            srcs = train.sources[s:e]
            dsts = train.destinations[s:e]
            tms = train.timestamps[s:e]
            eis = train.edge_idxs[s:e]
            lbs = train.labels[s:e]
            trace_rows = select_trace_rows(lbs, 32, 0, k, 'positive_first')
            adapter.set_trace_source_rows(trace_rows)
            with torch.no_grad():
                tgn.compute_temporal_embeddings(
                    srcs, dsts, dsts, tms, eis, ND)
            acc('pass1_fwd')
            revs = {int(r): {'dst': int(dsts[r]), 'label': float(lbs[r]),
                             'time': float(tms[r]),
                             'event_idx': int(eis[r])} for r in trace_rows}
            cuts = builder.build(adapter.trace, root_events=revs,
                                 batch_seed=k)
            acc('build_cuts')
            window.add(cuts)
            acc('window_add')
            window_batches.append(
                (srcs, dsts, tms, eis, lbs, trace_rows, k))
            if shadow is None:
                shadow = tgn.memory.backup_memory()
            if window.window_ready():
                closed, replay_plan, diag = window.close_replay()
                acc('close_replay')
                tgn.memory.restore_memory(shadow)
                opt.zero_grad(set_to_none=True)
                for bi, (src2, dst2, t2, e2, l2, tr2, step2) in \
                        enumerate(window_batches):
                    adapter.set_trace_source_rows(tr2)
                    emb, _, _ = tgn.compute_temporal_embeddings(
                        src2, dst2, dst2, t2, e2, ND)
                    acc('pass2_fwd')
                    loss = emb.sum() * 0.0
                    for tau, plan in replay_plan.items():
                        if bi >= len(plan["by_batch"]):
                            continue
                        for (occ_id, g) in plan["by_batch"][bi]:
                            z = adapter.trace.occurrences[occ_id].state.z
                            loss = loss + (g * z.float()).sum()
                    acc('pass2_surrogate')
                    loss.backward()
                    acc('backward')
                    tgn.memory.detach_memory()
                opt.step()
                acc('optimizer_step')
                window_batches = []
                shadow = None
                wins += 1
        if measure:
            acc('end')
            print('closed windows:', wins, 'of', N, 'batches')
        return wins

    run_rpbe_round(measure=False)   # warm the p cache
    run_rpbe_round(measure=True)    # epoch-2+ speed
total = sum(times.values())
for kk, v in sorted(times.items(), key=lambda x: -x[1]):
    print('{:<16} {:7.2f}s  {:5.1f}%  ({:6.1f} ms/batch)'.format(
        kk, v, 100 * v / total, v / N * 1000))
print('TOTAL per batch: {:.1f} ms'.format(total / N * 1000))
