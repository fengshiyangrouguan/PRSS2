#!/usr/bin/env python3
"""TGB tgbl-wiki recursive-closure link training runner (one arm/seed).

Builds the official multi-layer recursive Twitter-TGN on tgbl-wiki, attaches
the pair exact-replay trainer for one of the four arms, and writes rolling
checkpoints + per-epoch metrics.  Selection and final test MRR use the TGB
official evaluator in a separate evaluation step (this entry trains only).

Usage:
    python -m scripts.train_tgb_link --arm 2obs_aligned --seed 0 \
        --data-dir datasets --output outputs/tgbl_wiki_recursive_abl/seed0/2obs_aligned \
        --gpu 0
"""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rpbe.data.tgb_link import TGBLinkDataset
from rpbe.hosts.official_tgn import TGN, MLP, get_neighbor_finder
from rpbe.hosts.jodie_tgn import JodieTGNAdapter, TAU_TEMPLATE
from rpbe.config import RPBConfig
from rpbe.compressor import RecursiveCompressor
from rpbe.pair_maps import BoundaryMaps
from rpbe.link_records import LinkFutureIndex
from rpbe.training.tgb_link_loop import TGBPairLinkLoop, ARMS
from rpbe.audit import AuditAccumulator

# 9 unique configurations (spec §5).  ``arm`` stays one of the historical
# ARMS so feasibility/audit code paths are reused; the axes override arm_derived
# defaults in the loop.  ``lambda`` (None -> keep --lambda-kf) sets aux weight.
CONFIG_MAP = {
    "R0": dict(arm="2obs_aligned", aux_kind="kyfan", use_parent=1,
               mispaired=False, context_mode="full", variant="full_balancing"),
    "P0": dict(arm="gamma_task_only", aux_kind="none", use_parent=1,
               mispaired=False, context_mode="full", variant="full_balancing"),
    "P1": dict(arm="2obs_aligned", aux_kind="rec", use_parent=1,
               mispaired=False, context_mode="full", variant="full_balancing"),
    "P2": dict(arm="2obs_aligned", aux_kind="pred", use_parent=1,
               mispaired=False, context_mode="full", variant="full_balancing"),
    "S1": dict(arm="1obs", aux_kind="kyfan", use_parent=0,
               mispaired=False, context_mode="full", variant="full_balancing"),
    "S2": dict(arm="2obs_mispaired", aux_kind="kyfan", use_parent=1,
               mispaired=True, context_mode="full", variant="full_balancing"),
    "C1": dict(arm="2obs_aligned", aux_kind="kyfan", use_parent=1,
               mispaired=False, context_mode="constant",
               variant="full_balancing"),
    "B1": dict(arm="2obs_aligned", aux_kind="kyfan", use_parent=1,
               mispaired=False, context_mode="full", variant="unbalanced"),
    "E1": dict(arm="2obs_aligned", aux_kind="kyfan", use_parent=1,
               mispaired=False, context_mode="full", variant="diagonal"),
}


def resolve_config(args):
    """Return (arm, axes-dict or None) after applying ``--config``."""
    if args.config:
        if args.config not in CONFIG_MAP:
            raise ValueError("unknown --config {}".format(args.config))
        ax = dict(CONFIG_MAP[args.config])
        return ax.pop("arm"), ax
    return args.arm, None


def parse_args():
    p = argparse.ArgumentParser("tgbl-wiki pair link train")
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--arm", choices=list(ARMS), default="2obs_aligned")
    p.add_argument("--config", default="",
                   help="9-config id (R0/P0/P1/P2/S1/S2/C1/B1/E1); overrides arm "
                        "and sets aux_kind/use_parent/mispaired/context/variant")
    p.add_argument("--aux-lambda", type=float, default=0.0,
                   help="rec/pred supervision weight (0 -> reuse --lambda-kf)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-3,
                   help="val selection must beat best by >= this AP to reset")
    p.add_argument("--budget-cap", type=int, default=60,
                   help="hard cap when a run is budget-censored (spec §6)")
    p.add_argument("--bs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-neighbors", type=int, default=10)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--trace-roots", type=int, default=32)
    p.add_argument("--trace-pairs-per-parent", type=int, default=2)
    p.add_argument("--kf-group-batches", type=int, default=56)
    p.add_argument("--kf-min-trees", type=int, default=896)
    p.add_argument("--lambda-kf", type=float, default=0.088)
    p.add_argument("--ridge-eps", type=float, default=1e-3)
    p.add_argument("--sketch-dim", type=int, default=64)
    p.add_argument("--width-D", type=int, default=128)
    p.add_argument("--rpbe-seed", type=int, default=0)
    p.add_argument("--no-fail-on-monitor-error", action="store_true")
    p.add_argument("--max-batches", type=int, default=0,
                   help="cap each train epoch at N batches (0=full; smoke)")
    p.add_argument("--eval-every", type=int, default=0,
                   help="[legacy] score fixed val query set every N epochs "
                        "(0=off) and select best.pt by sampled_query_val_mrr")
    p.add_argument("--query-sets", default="",
                   help="[legacy] fixed query-id json (shared across arms)")
    p.add_argument("--negatives", default="",
                   help="Wiki-LR-Binary fixed one-neg-per-positive manifest; "
                        "when set, selection is per-epoch FULL-val ap_all "
                        "(spec §2.4) and train negatives are HistRand")
    p.add_argument("--group-plan", default="",
                   help="shared macro-group plan json (gate output). When set "
                        "the aux-eligible prefix and the task-only censored "
                        "tail are enforced; plan fields are validated strictly")
    p.add_argument("--val-every", type=int, default=1,
                   help="epochs between full-val ap_all evaluations (1=every)")
    p.add_argument("--kf-fail-below", action="store_true",
                   help="abort at the FIRST below-threshold KF window "
                        "(fail-fast; calibration runs leave it off so "
                        "window_diag.jsonl records real yields)")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def build_model(args, device):
    ds = TGBLinkDataset(root=args.data_dir)
    full, train, val, test = ds.splits()
    finder = get_neighbor_finder(
        train, uniform=False, max_node_idx=int(ds.n_internal_nodes - 1))
    # time stats from train
    ts = train.timestamps.astype(np.float64)
    ms, ss = float(ts.mean()), float(ts.std()) + 1e-8

    tgn = TGN(
        neighbor_finder=finder,
        node_features=ds.node_features.astype(np.float32),
        edge_features=ds.edge_features.astype(np.float32),
        device=device,
        n_layers=args.n_layers,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
        message_dimension=100,
        memory_dimension=172,
        memory_update_at_start=True,
        embedding_module_type="graph_attention",
        message_function="identity",
        aggregator_type="last",
        n_neighbors=args.n_neighbors,
        mean_time_shift_src=ms, std_time_shift_src=ss,
        mean_time_shift_dst=ms, std_time_shift_dst=ss,
    ).to(device)

    host_dim = int(tgn.embedding_dimension)
    taus = [TAU_TEMPLATE.format(l) for l in range(args.n_layers + 1)]
    rpbe_cfg = RPBConfig(
        state_dims={tau: host_dim for tau in taus},
        own_dims={tau: host_dim for tau in taus},
        width_D=args.width_D, m=args.sketch_dim,
        lambda_kf=args.lambda_kf, ridge_eps=args.ridge_eps,
        delta_t_scale=1.0,
        cuts_per_tau=1024, kf_min_ratio=2.0, kf_min_abs=args.kf_min_trees,
        kf_group_batches=args.kf_group_batches,
        kf_variant="full_balancing", n_observations=2,
        supervision_mode="production",
        kf_taus=list(taus[:-1]),
        rpbe_seed=args.rpbe_seed,
        dense_future=False)
    compressor = RecursiveCompressor(rpbe_cfg).to(device)
    adapter = JodieTGNAdapter(
        tgn.embedding_module, compressor=compressor,
        n_neighbors=args.n_neighbors,
        trace_pairs_per_parent=args.trace_pairs_per_parent)
    tgn.embedding_module = adapter
    # Optimizer ownership must be disjoint (Wiki-LR-Binary spec §4.1): head =
    # the affinity decoder ONLY; repr = every OTHER trainable parameter (host
    # encoder + Gamma + compressor + memory).  A leaf is visited twice under
    # tgn.parameters() (e.g. time_encoder is shared with the embedding module),
    # so id-dedup AFTER excluding head; a param in BOTH optimizers would be
    # double-stepped (head each batch, repr each macro group).
    head_params = [p for p in tgn.affinity_score.parameters()
                   if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    repr_params = []
    _seen_ids = set()
    for p in tgn.parameters():
        if not p.requires_grad or id(p) in head_ids or id(p) in _seen_ids:
            continue
        _seen_ids.add(id(p))
        repr_params.append(p)
    assert head_ids.isdisjoint({id(p) for p in repr_params}), \
        "head and repr param sets must be disjoint"
    head_optimizer = torch.optim.Adam(head_params, lr=args.lr)
    repr_optimizer = torch.optim.Adam(repr_params, lr=args.lr)
    opt_names = {id(p): n for n, p in tgn.named_parameters()}
    head_names = sorted(opt_names[id(p)] for p in head_params)
    repr_names = sorted(opt_names[id(p)] for p in repr_params)
    # boundary maps
    d_msg = int(ds.msg_dim)
    boundary_maps = BoundaryMaps(
        d_ctx=int(rpbe_cfg.d_c), d_event=int(rpbe_cfg.d_f), m=int(rpbe_cfg.m),
        d_msg=d_msg, delta_t_scale=1.0, msg_scale=1e-3,
        num_counter_bins=4096, seed=int(args.rpbe_seed)).to(device)
    boundary_maps.message_for = lambda eid: torch.as_tensor(
        ds.edge_features[int(eid)], dtype=torch.float32, device=device)
    link_future_index = LinkFutureIndex(
        train.sources, train.destinations, train.timestamps,
        train.edge_idxs)
    return dict(ds=ds, tgn=tgn, rpbe_cfg=rpbe_cfg, adapter=adapter,
                compressor=compressor, head_optimizer=head_optimizer,
                repr_optimizer=repr_optimizer, boundary_maps=boundary_maps,
                link_future_index=link_future_index,
                edge_table=ds.edge_features,
                opt_split={"n_head": len(head_params),
                           "n_repr": len(repr_params),
                           "head_names": head_names,
                           "repr_names": repr_names})


def _group_plan_canonical(plan):
    """Canonical subset of a group plan that runner and gate must agree on."""
    sm = plan.get("scan", {}).get("summary", {})
    chosen = plan.get("chosen", {})
    return {
        "data": plan.get("data"),
        "n_neighbors": plan.get("n_neighbors"),
        "n_layers": plan.get("n_layers"),
        "bs": plan.get("bs"),
        "kf_min_trees": plan.get("kf_min_trees"),
        "group_batches": chosen.get("group_batches"),
        "trace_roots": chosen.get("trace_roots"),
        "n_train_events": plan.get("n_train_events"),
        "n_train_batches": plan.get("n_train_batches"),
        "aux_prefix_groups": sm.get("aux_prefix_groups"),
        "censored_tail_groups": sm.get("censored_tail_groups"),
        "prefix_min": sm.get("prefix_min"),
        "prefix_p5": sm.get("prefix_p5"),
    }


def _validate_group_plan(plan, args, ds, train):
    """Strict gate: plan must match the run config + data + its own SHA."""
    import math as _math
    from rpbe.data.wiki_binary_negatives import content_sha256
    chk = {
        "data": plan.get("data"),
        "bs": plan.get("bs"),
        "n_neighbors": plan.get("n_neighbors"),
        "n_layers": plan.get("n_layers"),
        "kf_min_trees": plan.get("kf_min_trees"),
        "group_batches": plan.get("chosen", {}).get("group_batches"),
        "trace_roots": plan.get("chosen", {}).get("trace_roots"),
        "n_train_events": plan.get("n_train_events"),
        "n_train_batches": plan.get("n_train_batches"),
    }
    exp = {
        "data": ds.name,
        "bs": args.bs,
        "n_neighbors": args.n_neighbors,
        "n_layers": args.n_layers,
        "kf_min_trees": args.kf_min_trees,
        "group_batches": args.kf_group_batches,
        "trace_roots": args.trace_roots,
        "n_train_events": int(len(train.sources)),
        "n_train_batches": int(_math.ceil(
            len(train.sources) / float(max(1, args.bs)))),
    }
    for k, v in exp.items():
        if chk.get(k) != v:
            raise ValueError(
                "group plan field {} = {} but run requires {}".format(
                    k, chk.get(k), v))
    sha = content_sha256(_group_plan_canonical(plan))
    if plan.get("plan_sha") != sha:
        raise ValueError("group plan sha {} != recomputed {}".format(
            plan.get("plan_sha"), sha))
    sm = plan.get("scan", {}).get("summary", {})
    return sha, int(sm.get("aux_prefix_groups")), int(
        sm.get("censored_tail_groups"))


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    c = build_model(args, device)
    ds = c["ds"]
    train = ds.train
    val = ds.val
    monitor = None  # minimal; metrics go to jsonl
    # ---- 9-config registry axes (config overrides arm-derived defaults)
    eff_arm, axes = resolve_config(args)
    aux_kind = (axes or {}).get("aux_kind", "kyfan")
    use_parent = (axes or {}).get("use_parent")
    mispaired = (axes or {}).get("mispaired")
    context_mode = (axes or {}).get("context_mode", "full")
    variant = (axes or {}).get("variant", "full_balancing")
    if axes is None and args.arm == "gamma_task_only":
        aux_kind = "none"
    aux_lambda = args.aux_lambda if args.aux_lambda > 0.0 else args.lambda_kf
    aux_heads = None
    aux_optimizer = None
    if aux_kind in ("rec", "pred"):
        import random as _random
        from rpbe.training.aux_heads import AuxHeads
        # RNG isolation (fairness): building the extra predictors must not
        # perturb the global RNG stream the main model / task dropout uses, so
        # a P1/P2 run and a P0/R0 run draw identical task forward/update RNGs.
        _rng_saved = (torch.get_rng_state(), np.random.get_state(),
                      _random.getstate(),
                      torch.cuda.get_rng_state(device)
                      if torch.cuda.is_available() else None)
        host_dim = int(c["tgn"].embedding_dimension)
        taus = list(getattr(c["adapter"], "compression_taus", []) or [])
        d_out = host_dim if aux_kind == "rec" else int(c["boundary_maps"].m)
        aux_heads = AuxHeads(
            aux_kind, taus=taus, z_dim=host_dim,
            d_ctx=int(getattr(c["rpbe_cfg"], "d_c", 32)),
            d_out=d_out).to(device)
        torch.set_rng_state(_rng_saved[0])
        np.random.set_state(_rng_saved[1])
        _random.setstate(_rng_saved[2])
        if _rng_saved[3] is not None:
            torch.cuda.set_rng_state(_rng_saved[3], device)
        aux_optimizer = torch.optim.Adam(aux_heads.parameters(), lr=args.lr)
        tgn_ids = {id(p) for p in c["tgn"].parameters()}
        assert tgn_ids.isdisjoint(
            {id(p) for p in aux_heads.parameters()}), \
            "aux head params must be disjoint from the host"
        print("aux_heads({}) taus={} z={} d_out={}".format(
            aux_kind, taus, host_dim, d_out), flush=True)
    # ---- shared group plan: aux-eligible prefix + task-only censored tail
    plan_sha = None
    aux_prefix = None
    censored_tail = None
    if args.group_plan:
        plan = json.load(open(args.group_plan))
        plan_sha, aux_prefix, censored_tail = _validate_group_plan(
            plan, args, ds, train)
        print("group-plan ok: prefix={} censored_tail={} sha={}".format(
            aux_prefix, censored_tail, plan_sha), flush=True)
    loop = TGBPairLinkLoop(
        tgn=c["tgn"], device=device, batch_size=args.bs,
        n_neighbors=args.n_neighbors, grad_clip=5.0, monitor=monitor,
        seed=args.seed, adapter=c["adapter"],
        link_future_index=c["link_future_index"],
        boundary_maps=c["boundary_maps"], edge_table=c["edge_table"],
        arm=eff_arm, rpbe_cfg=c["rpbe_cfg"],
        repr_optimizer=c["repr_optimizer"], head_optimizer=c["head_optimizer"],
        trace_roots=args.trace_roots,
        trace_pairs_per_parent=args.trace_pairs_per_parent,
        kf_group_batches=args.kf_group_batches,
        kf_min_trees=args.kf_min_trees,
        fail_below=args.kf_fail_below,
        audit_trace=True,
        aux_prefix_groups=aux_prefix, group_plan_sha=plan_sha,
        use_parent=use_parent, mispaired=mispaired,
        context_mode=context_mode, variant=variant,
        aux_kind=aux_kind, aux_heads=aux_heads,
        aux_optimizer=aux_optimizer, aux_lambda=aux_lambda)

    save_json(out / "config.json", {
        "data": "tgbl-wiki", "seed": args.seed, "arm": eff_arm,
        "config": args.config, "aux_kind": aux_kind,
        "use_parent": int(use_parent if use_parent is not None
                          else (1 if eff_arm in ("2obs_aligned",
                                                 "2obs_mispaired") else 0)),
        "mispaired": bool(mispaired if mispaired is not None
                          else (eff_arm == "2obs_mispaired")),
        "context_mode": context_mode, "variant": variant,
        "device": str(device), "epochs": args.epochs, "bs": args.bs,
        "n_neighbors": args.n_neighbors, "n_layers": args.n_layers,
        "lambda_kf": args.lambda_kf, "aux_lambda": aux_lambda,
        "kf_group_batches": args.kf_group_batches,
        "kf_min_trees": args.kf_min_trees,
        "group_plan_sha": plan_sha,
        "maps_sha": c["boundary_maps"].isolation_fingerprint()["sha256"],
        "opt_split": c["opt_split"], "cli": vars(args)})

    # ---- Wiki-LR-Binary negatives: fixed one-neg-per-positive manifest for
    # val/test selection + prefix-causal HistRand negatives for train.
    selection = "sampled_query_val_mrr"   # legacy fallback
    negs = None
    if args.negatives:
        from rpbe.data.wiki_binary_negatives import CompactNegatives
        from rpbe.training.histrand_sampler import HistRandTrainSampler
        negs = CompactNegatives(args.negatives)
        loop.train_neg_sampler = HistRandTrainSampler(
            train.sources, train.destinations, train.timestamps,
            model_seed=args.seed)
        selection = "ap_all"
    else:
        loop.train_neg_sampler = None
    if negs is not None:
        negs.verify(ds)   # fail-closed load gate (§1.1 #4)

    metrics_path = out / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    val_history_path = out / "val_history.jsonl"
    val_history_path.unlink(missing_ok=True)
    best_val = -1e9
    bad = 0
    best_epoch = -1
    gs = 0
    repr_cum = 0
    closed_cum = 0
    win_diag_path = out / "window_diag.jsonl"

    start_epoch = 0
    epoch_aps = []
    budget = args.epochs if args.epochs > 0 else args.budget_cap
    extended = False
    stop_reason = "budget"

    epoch = start_epoch
    while epoch < budget:
        t0 = time.time()
        row = loop.train_epoch(
            epoch, gs, train,
            max_batches=args.max_batches if args.max_batches > 0 else None)
        gs = row.get("global_step", gs)
        row["epoch"] = epoch
        row["epoch_seconds"] = time.time() - t0
        # cumulative repr/closed step counters across epochs (§1.1 #6); curves
        # use cumulative repr_step as their x-axis.
        repr_cum += int(row.get("repr_step", 0))
        closed_cum += int(row.get("closed_window_step", 0))
        row["repr_step"] = repr_cum
        row["closed_window_step"] = closed_cum
        for wd in row.get("window_diag", []):
            wd["epoch"] = epoch
            with win_diag_path.open("a") as f:
                f.write(json.dumps(wd, allow_nan=True) + "\n")
        row.pop("window_diag", None)
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, allow_nan=True) + "\n")
        print(json.dumps({"epoch": epoch, "repr_step": row.get("repr_step"),
              **{k: row[k] for k in
                 ("train_link_loss", "n_closed", "n_below",
                  "n_aux_batches")}}, allow_nan=True), flush=True)

        # ---- checkpoint selection --------------------------------
        val_row = None
        if negs is not None and (epoch + 1) % args.val_every == 0:
            vm = _val_ap_inprocess(args, device, c, ds, negs)
            val_row = {"epoch": epoch,
                       "repr_step": row.get("repr_step", gs),
                       "selection": "ap_all",
                       **{k: vm.get(k, float("nan")) for k in
                          ("ap_all", "auc_all", "nll_all", "paired_acc",
                           "ap_hist", "auc_hist", "nll_hist",
                           "n_hist", "n_random")}}
            epoch_aps.append(float(vm.get("ap_all", float("nan"))))
            improved = vm.get("ap_all", float("nan")) > best_val + args.min_delta
        elif negs is None and args.eval_every > 0 and args.query_sets \
                and (epoch + 1) % args.eval_every == 0:
            vm = {"sampled_query_val_mrr":
                  _val_mrr_inprocess(args, device, c, ds, out)}
            val_row = {"epoch": epoch, "selection": "sampled_query_val_mrr",
                       "sampled_query_val_mrr":
                       vm["sampled_query_val_mrr"]}
            improved = vm["sampled_query_val_mrr"] > best_val
        else:
            improved = False
            vm = {}

        if val_row is not None:
            with val_history_path.open("a") as f:
                f.write(json.dumps(val_row, allow_nan=True) + "\n")
            print("epoch {} {}={:.5f}".format(
                epoch, selection, val_row.get(selection)), flush=True)
            if improved:
                best_val = float(val_row.get(selection))
                best_epoch = epoch
                bad = 0
                _save_ckpt(out / "best.pt", c, epoch, best_val, args, selection, aux_heads)
            else:
                bad += 1
                if negs is not None and bad >= args.patience:
                    stop_reason = "early_stop_ap"
                    break
                if negs is None and bad >= args.patience:
                    stop_reason = "early_stop_mrr"
                    break
        elif epoch == 0:
            # always keep a first best.pt so downstream eval has one
            _save_ckpt(out / "best.pt", c, epoch, 0.0, args, selection, aux_heads)
        _save_ckpt(out / "last.pt", c, epoch, best_val, args, selection, aux_heads)
        epoch += 1
        # ---- budget-censored extension (spec §6): only when the epoch cap
        # is hit, training is still visibly improving, and the best lies in
        # the last few epochs.  One extension of +20 up to the hard cap.
        if epoch >= budget and stop_reason != "early_stop_ap" \
                and negs is not None and not extended \
                and budget < args.budget_cap:
            recent = epoch_aps[-5:] if epoch_aps else []
            if len(recent) == 5 and recent[-1] - recent[0] > 0.002 \
                    and best_epoch >= epoch - 4:
                extended = True
                budget = min(args.budget_cap, budget + 20)
                print("budget-censored: extend to epoch {}".format(budget),
                      flush=True)
    summary = {"data": "tgbl-wiki", "seed": args.seed, "arm": args.arm,
               "best_epoch": int(best_epoch),
               "best_{}".format(selection): float(best_val),
               "selection": selection, "stop_reason": stop_reason,
               "extended": bool(extended)}
    save_json(out / "summary.json", summary)
    # ---- read-only comparison sidecar (spec §26 item 13) ----
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10, cwd=str(Path(__file__).resolve().parents[1])
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    config_hash = hashlib.blake2b(
        json.dumps(vars(args), sort_keys=True, default=str).encode("utf-8"),
        digest_size=16).hexdigest()
    fixed_feature = {
        "rpbe_seed": int(args.rpbe_seed),
        "sketch_dim": int(args.sketch_dim),
        "width_D": int(args.width_D),
        "d_ctx": int(c["rpbe_cfg"].d_c) if hasattr(c["rpbe_cfg"], "d_c") else None,
        "d_event": int(c["rpbe_cfg"].d_f) if hasattr(c["rpbe_cfg"], "d_f") else None,
    }
    audit = loop.audit
    save_json(out / "comparison_audit.json",
              audit.dump(commit=commit, config_hash=config_hash,
                         fixed_feature=fixed_feature))
    save_json(out / "_SUCCESS.json", {"status": "complete",
                                      "best_epoch": int(best_epoch),
                                      "selection": selection,
                                      "stop_reason": stop_reason})
    print(json.dumps(summary), flush=True)


def _save_ckpt(path, c, epoch, score, args, selection, aux_heads=None):
    ck = {"model": {"tgn": c["tgn"].state_dict(),
                    "compressor": c["compressor"].state_dict()},
          "epoch": int(epoch), "score": float(score),
          "arm": args.arm, "seed": args.seed,
          "selection": selection}
    if aux_heads is not None:
        ck["aux_heads"] = aux_heads.state_dict()
    torch.save(ck, path)


def _val_ap_inprocess(args, device, c, ds, negs):
    """Full-split val ap_all at current model state (memory restored after)."""
    from rpbe.training.wiki_binary_eval import evaluate_val
    with torch.no_grad():
        return evaluate_val(c["tgn"], ds, negs,
                            n_neighbors=args.n_neighbors, bs=args.bs)


def _val_mrr_inprocess(args, device, c, ds, out):
    """Score the fixed val query set at the current model state."""
    from rpbe.link_eval import score_split
    ds.load_val_ns()
    qs = json.load(open(args.query_sets))
    qids = qs["val_query_ids"]
    tgn = c["tgn"]

    def _adv(src, dst, t, eidx, bs=1):
        with torch.no_grad():
            tgn.compute_edge_probabilities(
                np.asarray(src, dtype=np.int64),
                np.asarray(dst, dtype=np.int64),
                np.asarray(dst, dtype=np.int64),
                np.asarray(t, dtype=np.float64),
                np.asarray(eidx, dtype=np.int64), args.n_neighbors)

    # train_epoch ends with memory at train-end state; replay the fixed
    # val stream? score_split advances per scored event. To keep it a clean
    # per-event online eval we pass advance_stream that steps one event.
    def step(src, dst, t, eidx):
        _adv(src, dst, t, eidx, bs=1)

    with torch.no_grad():
        res = score_split(tgn, ds, "val", qids,
                          n_neighbors=args.n_neighbors,
                          chunk=64, advance_stream=step)
    return float(res.get("sampled_query_val_mrr", float("nan")))


if __name__ == "__main__":
    main()
