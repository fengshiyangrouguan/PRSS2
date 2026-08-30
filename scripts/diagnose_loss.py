#!/usr/bin/env python3
"""Falsification-first audit for the RPBE measurement and lagged gradient.

This script does not train.  It loads one frozen checkpoint, replays the
chronological prefix, and collects the exact cuts used by the current main
branch.  It answers four questions before another lambda or seed sweep:

1. Are Y1/Y2 actually parent-pullback records, or just later stream events?
2. Is the Ky Fan score carried by the observed outcome, or by context alone?
3. How much independent support remains after outcome/node reuse and censoring?
4. Does the production one-group-lagged VJP point in the direction of the
   exact current-group Ky Fan gradient at the latent cut states?

Recommended first run (use a matched ``--rpbe --kf-lambda 0`` checkpoint):

    python -m scripts.diagnose_loss \
      --run-dir outputs/s0_lambda_scan/lam0 \
      --data-dir old/processed_tgn_data --audit-split val \
      --group-batches 40 --groups 3 --permutations 20

Run it again on the RPBE checkpoint.  Do not use test for iteration; the
validation split is sufficient for this mechanism audit.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rpbe.data.jodie import JodieDataset
from rpbe.loss import (WeightedWelford, _matrix_diag, _score_from_covs,
                       kf_adjoint, kf_vjp_batch)
from rpbe.maps import FixedMaps
from rpbe.records import JodieCutBuilder, JodieFutureIndex, NODE_CLASS
from rpbe.training.jodie_loop import select_trace_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="run containing config.json and best.pt")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--audit-split", choices=["train", "val"],
                        default="val")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--group-batches", type=int, default=0,
                        help="0 = value from run config/default formula")
    parser.add_argument("--groups", type=int, default=3,
                        help="consecutive groups to collect; >=2 for lag audit")
    parser.add_argument("--trace-roots", type=int, default=0,
                        help="0 = value from run config")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--window-sweep", default="20,40,80",
                        help="comma list of group sizes for the "
                             "exact-exact gradient stability sweep")
    # Records.py seeds np.random.RandomState with seed * 1000003, so the
    # builder seed must stay below 2**32 / 1000003 ~ 4294.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def _load_frozen_run(args, device):
    run_dir = Path(args.run_dir)
    config = json.load(open(run_dir / "config.json"))
    cli = dict(config["cli"])
    if not cli.get("rpbe", False):
        raise ValueError(
            "diagnose_loss needs a run built with --rpbe (lambda=0 is fine)")

    import scripts.train_jodie as train_jodie

    # best.pt already contains the complete stage-2 state.  Avoid requiring
    # the original stage-1 checkpoint path merely to reconstruct modules.
    cli["pretrained_checkpoint"] = ""
    cli["data_dir"] = args.data_dir
    namespace = argparse.Namespace(**cli)
    dataset = JodieDataset(cli["data"], data_dir=args.data_dir,
                           use_validation=True)
    components = train_jodie.build_components(namespace, device, dataset)
    # Reconstruct the checkpoint's fixed measurement exactly.  Runs created
    # before the train-only delta_t_scale fix stored the old full-stream value
    # in config.json; using today's constructor value would silently audit a
    # different P than the one that trained the checkpoint.
    saved_rpbe = config.get("rpbe") or {}
    saved_delta_scale = saved_rpbe.get("delta_t_scale")
    if saved_delta_scale is not None:
        components["rpbe_cfg"].delta_t_scale = float(saved_delta_scale)
        components["fixed_maps"] = FixedMaps(
            components["rpbe_cfg"]).to(device)
    best = torch.load(run_dir / "best.pt", map_location=device,
                      weights_only=False)
    components["decoder"].load_state_dict(best["model"]["decoder"])
    components["tgn"].load_state_dict(best["model"]["tgn"])
    if components["compressor"] is not None:
        components["compressor"].load_state_dict(
            best["model"]["compressor"])

    for module_name in ("tgn", "decoder", "compressor"):
        module = components.get(module_name)
        if module is not None:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return config, cli, dataset, components


def _forward_stream(tgn, adapter, stream, batch_size, n_neighbors,
                    trace_roots, future_index=None, builder=None,
                    max_batches=None, seed=0):
    """Chronological no-grad pass; optionally collect current cut rows."""
    batches = []
    candidate_rows = []
    stats = {}
    total_batches = math.ceil(len(stream.sources) / batch_size)
    if max_batches is not None:
        total_batches = min(total_batches, int(max_batches))
    with torch.no_grad():
        for batch_index in range(total_batches):
            start = batch_index * batch_size
            stop = min(len(stream.sources), (batch_index + 1) * batch_size)
            sources = stream.sources[start:stop]
            destinations = stream.destinations[start:stop]
            timestamps = stream.timestamps[start:stop]
            edge_idxs = stream.edge_idxs[start:stop]
            size = stop - start
            if builder is None:
                if adapter is not None:
                    adapter.clear_trace()
            else:
                roots = select_trace_rows(
                    np.zeros(size), trace_roots, seed, batch_index,
                    "evenly_spaced")
                adapter.set_trace_source_rows(roots)

            tgn.compute_temporal_embeddings(
                sources, destinations, destinations, timestamps, edge_idxs,
                n_neighbors)

            rows = []
            if builder is not None and adapter.trace is not None:
                trace = adapter.trace
                for cut in trace.cuts:
                    future = future_index.query(cut.node, cut.time, limit=2)
                    candidate_rows.append({
                        "tau": str(cut.tau), "node": int(cut.node),
                        "time": float(cut.time), "n_future": len(future),
                        "path": list(cut.path),
                    })
                rows = builder.build(trace, batch_seed=batch_index,
                                     stats=stats)
            batches.append(rows)
            if adapter is not None:
                adapter.clear_trace()
    return batches, candidate_rows, stats


def _row_tensors(rows, fixed_maps, future_mode="joint", outcomes=None):
    contexts = [row.context for row in rows]
    observed = [row.outcome for row in rows] if outcomes is None else outcomes
    z = torch.stack([row.z.detach() for row in rows])
    p = fixed_maps.pv_batch(
        contexts, observed, future_mode=future_mode, use_cache=False).detach()
    w = torch.tensor([float(row.weight) for row in rows],
                     dtype=torch.float64, device=z.device)
    cut_ids = [tuple(row.cut_id) for row in rows]
    return z, p, w, cut_ids


def _weighted_score_tensor(z, p, w, cut_ids, eps):
    z = z.double()
    p = p.detach().double()
    w = w.detach().double().reshape(-1)
    total = w.sum()
    if len(z) < 2 or float(total) <= 0.0:
        return None, {"failed": "too_few_rows"}
    mu_z = (z * w[:, None]).sum(0, keepdim=True) / total
    mu_p = (p * w[:, None]).sum(0, keepdim=True) / total
    zc = z - mu_z
    pc = p - mu_p
    sw = w.sqrt()[:, None]
    mzz = (zc * sw).t() @ (zc * sw)
    mpp = (pc * sw).t() @ (pc * sw)
    mzp = (zc * sw).t() @ (pc * sw)
    by_cut = defaultdict(float)
    for cut_id, weight in zip(cut_ids, w.tolist()):
        by_cut[cut_id] += float(weight)
    w2_cut = sum(value * value for value in by_cut.values())
    den = float(total) - w2_cut / float(total)
    if den <= 0.0:
        return None, {"failed": "nonpositive_dof"}
    score, diag = _score_from_covs(
        mzz / den, mzp / den, mpp / den, eps, "full_balancing")
    if score is not None:
        diag = dict(diag)
        diag.update(_matrix_diag("zz", mzz / den))
        diag.update(_matrix_diag("pp", mpp / den))
        diag.update({"W": float(total), "D": den,
                     "n_rows": int(z.shape[0]),
                     "n_cuts": len(by_cut)})
    return score, diag


def _score(rows, fixed_maps, eps, future_mode="joint", outcomes=None,
           use_u=False):
    z, p, w, cut_ids = _row_tensors(
        rows, fixed_maps, future_mode=future_mode, outcomes=outcomes)
    if use_u:
        if any(row.u is None for row in rows):
            return None, {"failed": "missing_u"}
        z = torch.stack([row.u.detach() for row in rows])
    value, diag = _weighted_score_tensor(z, p, w, cut_ids, eps)
    return (None if value is None else float(value.detach())), diag


def _stratified_outcome_permutation(rows, rng):
    outcomes = np.asarray([row.outcome for row in rows], dtype=np.float64)
    shuffled = outcomes.copy()
    strata = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(int(row.horizon), int(row.context.get("role", 0)))].append(index)
    for indexes in strata.values():
        indexes = np.asarray(indexes, dtype=np.int64)
        shuffled[indexes] = outcomes[rng.permutation(indexes)]
    return shuffled.tolist()


def _stratified_row_permutation(rows, rng):
    permutation = np.arange(len(rows))
    strata = defaultdict(list)
    for index, row in enumerate(rows):
        strata[int(row.horizon)].append(index)
    for indexes in strata.values():
        indexes = np.asarray(indexes, dtype=np.int64)
        permutation[indexes] = rng.permutation(indexes)
    return permutation


def _score_audit(all_rows, fixed_maps, eps, permutations, seed,
                 pi_train=0.5):
    report = {}
    taus = sorted({row.tau for row in all_rows})
    for tau_index, tau in enumerate(taus):
        rows = [row for row in all_rows if row.tau == tau]
        entry = {
            "n_rows": len(rows),
            "n_cuts": len({row.cut_id for row in rows}),
            "n_trees": len({row.tree_id for row in rows}),
            "positives": int(sum(row.outcome > 0.5 for row in rows)),
            "positive_rate": float(np.mean(
                [row.outcome > 0.5 for row in rows])),
            "rows_by_horizon": dict(Counter(
                str(row.horizon) for row in rows)),
        }
        for mode in ("joint", "context_common", "outcome_contrast"):
            value_z, diag_z = _score(
                rows, fixed_maps, eps, future_mode=mode)
            value_u, diag_u = _score(
                rows, fixed_maps, eps, future_mode=mode, use_u=True)
            entry[mode] = {"J_Z": value_z, "J_U": value_u,
                           "diag_Z": diag_z, "diag_U": diag_u}

        rng = np.random.RandomState(seed + 1009 * (tau_index + 1))
        contrast_shuffle = []
        joint_pair_shuffle = []
        z, p_joint, w, cut_ids = _row_tensors(
            rows, fixed_maps, future_mode="joint")
        for _ in range(permutations):
            shuffled_y = _stratified_outcome_permutation(rows, rng)
            value, _ = _score(
                rows, fixed_maps, eps, future_mode="outcome_contrast",
                outcomes=shuffled_y)
            if value is not None:
                contrast_shuffle.append(value)
            permutation = _stratified_row_permutation(rows, rng)
            permutation_t = torch.as_tensor(
                permutation, dtype=torch.long, device=p_joint.device)
            value_t, _ = _weighted_score_tensor(
                z, p_joint[permutation_t], w, cut_ids, eps)
            if value_t is not None:
                joint_pair_shuffle.append(float(value_t.detach()))

        def summarize(values):
            if not values:
                return {"n": 0}
            array = np.asarray(values, dtype=np.float64)
            return {"n": len(values), "mean": float(array.mean()),
                    "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                    "p95": float(np.percentile(array, 95)),
                    "values": [float(x) for x in array]}

        entry["nulls"] = {
            "outcome_contrast_label_shuffle": summarize(contrast_shuffle),
            "joint_row_shuffle_within_horizon": summarize(joint_pair_shuffle),
        }
        contrast_real = entry["outcome_contrast"]["J_Z"]
        contrast_p95 = entry["nulls"][
            "outcome_contrast_label_shuffle"].get("p95")
        joint_real = entry["joint"]["J_Z"]
        joint_null = entry["nulls"][
            "joint_row_shuffle_within_horizon"].get("mean")
        entry["excess"] = {
            "joint_over_pair_shuffle": None if joint_null is None
            else joint_real - joint_null,
            "contrast_over_label_shuffle_mean": None
            if not contrast_shuffle else contrast_real - float(
                np.mean(contrast_shuffle)),
        }
        entry["gates"] = {
            "outcome_contrast_above_shuffle_p95": bool(
                contrast_p95 is not None and contrast_real > contrast_p95),
        }
        # Review follow-up: pure label-residual tests for Z and U.  A
        # residual test removes the context-predictable channel BEFORE
        # scoring; passing with U but failing with Z means the
        # compression lost information, failing with U too means the
        # future state-change label itself carries nothing beyond
        # context here.
        for res_name, res_values in (
                ("residual_balanced",
                 _label_residual_values(rows, float(pi_train))),
                ("residual_context",
                 _context_residual_values(rows, fixed_maps))):
            if res_values is None:
                entry[res_name] = None
                continue
            value_z, _ = _score(rows, fixed_maps, eps,
                                future_mode="residual",
                                outcomes=res_values)
            value_u, _ = _score(rows, fixed_maps, eps,
                                future_mode="residual",
                                outcomes=res_values, use_u=True)
            shuffled_scores = []
            rng_local = np.random.RandomState(seed + 7331)
            base = np.asarray(res_values, dtype=np.float64)
            for _ in range(permutations):
                perm = base[rng_local.permutation(len(base))]
                v, _ = _score(rows, fixed_maps, eps,
                              future_mode="residual",
                              outcomes=perm.tolist())
                if v is not None:
                    shuffled_scores.append(v)
            p95 = float(np.percentile(shuffled_scores, 95)) \
                if shuffled_scores else None
            entry[res_name] = {
                "J_Z": value_z, "J_U": value_u,
                "shuffle_p95": p95,
                "Z_above_shuffle_p95": bool(
                    value_z is not None and p95 is not None
                    and value_z > p95),
                "U_above_shuffle_p95": bool(
                    value_u is not None and p95 is not None
                    and value_u > p95),
            }
        report[tau] = entry
    return report


def _merge_cut_gradients(gradient, cut_ids):
    merged = {}
    for cut_id, value in zip(cut_ids, gradient):
        merged[cut_id] = merged.get(cut_id, torch.zeros_like(value)) + value
    return merged


def _flatten_aligned(left, right):
    keys = sorted(set(left) & set(right))
    if not keys:
        return None, None
    return (torch.cat([left[key].reshape(-1) for key in keys]).double(),
            torch.cat([right[key].reshape(-1) for key in keys]).double())


def _reference_from_batches(batches, tau, fixed_maps, dim_z, eps):
    accumulator = WeightedWelford(dim_z, fixed_maps.m)
    for rows_all in batches:
        rows = [row for row in rows_all if row.tau == tau]
        if not rows:
            continue
        z, p, w, cut_ids = _row_tensors(rows, fixed_maps, "joint")
        accumulator.add(z, p, w, cut_ids)
    result = accumulator.result()
    score, adjoints, diag = kf_adjoint(result, eps=eps,
                                       variant="full_balancing")
    return result, score, adjoints, diag


def _production_latent_gradient(batches, tau, fixed_maps, reference,
                                adjoints, group_multiplier=True):
    merged = {}
    group_batches = len(batches)
    for rows_all in batches:
        rows = [row for row in rows_all if row.tau == tau]
        if not rows:
            continue
        z, p, w, cut_ids = _row_tensors(rows, fixed_maps, "joint")
        z_var = z.detach().double().requires_grad_(True)
        value = kf_vjp_batch(
            z_var, p, w, reference["mu_z"], reference["mu_p"],
            adjoints)
        if group_multiplier:
            value = value * group_batches
        gradient = torch.autograd.grad(value, z_var)[0].detach()
        for cut_id, cut_gradient in _merge_cut_gradients(
                gradient, cut_ids).items():
            merged[cut_id] = merged.get(
                cut_id, torch.zeros_like(cut_gradient)) + cut_gradient
    # This is the exact division performed at _close_repr_group, including
    # batches with no valid rows for this tau.
    return {key: value / max(group_batches, 1)
            for key, value in merged.items()}


def _exact_latent_gradient(batches, tau, fixed_maps, eps):
    rows = [row for batch in batches for row in batch if row.tau == tau]
    if not rows:
        return None, {}, {"failed": "no_rows"}
    z, p, w, cut_ids = _row_tensors(rows, fixed_maps, "joint")
    z_var = z.detach().double().requires_grad_(True)
    score, diag = _weighted_score_tensor(z_var, p, w, cut_ids, eps)
    if score is None:
        return None, {}, diag
    gradient = torch.autograd.grad(score, z_var)[0].detach()
    return float(score.detach()), _merge_cut_gradients(gradient, cut_ids), diag


def _gradient_comparison(exact, surrogate):
    exact_flat, surrogate_flat = _flatten_aligned(exact, surrogate)
    if exact_flat is None:
        return {"failed": "no_aligned_cuts"}
    exact_norm = exact_flat.norm().clamp(min=1e-30)
    surrogate_norm = surrogate_flat.norm()
    cosine = float((exact_flat @ surrogate_flat)
                   / (exact_norm * surrogate_norm.clamp(min=1e-30)))
    return {
        "failed": None,
        "cosine": cosine,
        "norm_ratio": float(surrogate_norm / exact_norm),
        "relative_error": float(
            (surrogate_flat - exact_flat).norm() / exact_norm),
        "n_coordinates": int(exact_flat.numel()),
    }


def _lag_audit(groups, fixed_maps, eps):
    report = []
    taus = sorted({row.tau for group in groups for batch in group
                   for row in batch})
    for pair_index in range(len(groups) - 1):
        previous = groups[pair_index]
        current = groups[pair_index + 1]
        pair = {"reference_group": pair_index,
                "current_group": pair_index + 1, "taus": {}}
        for tau in taus:
            current_rows = [row for batch in current for row in batch
                            if row.tau == tau]
            if not current_rows:
                continue
            dim_z = int(current_rows[0].z.numel())
            reference, j_ref, adjoints, ref_diag = _reference_from_batches(
                previous, tau, fixed_maps, dim_z, eps)
            j_current, exact, current_diag = _exact_latent_gradient(
                current, tau, fixed_maps, eps)
            # Review follow-up: the decisive control — do the EXACT
            # gradients of adjacent groups point the same way?  If they
            # are near-orthogonal too, the window CCA gradient itself is
            # noisy and a two-pass replay would only chase random
            # directions more precisely.
            j_prev, exact_prev, prev_diag = _exact_latent_gradient(
                previous, tau, fixed_maps, eps)
            if adjoints is None or not exact:
                pair["taus"][tau] = {
                    "failed": "reference_or_current_score_failed",
                    "reference_diag": ref_diag,
                    "current_diag": current_diag,
                }
                continue
            lagged = _production_latent_gradient(
                current, tau, fixed_maps, reference, adjoints,
                group_multiplier=True)
            lagged_raw = _production_latent_gradient(
                current, tau, fixed_maps, reference, adjoints,
                group_multiplier=False)

            # Same-point control: replace only the lagged adjoint/means with
            # current-group values.  Failure here is implementation/scaling;
            # failure only in lagged is reference/data drift.
            current_ref, _, current_adjoints, same_diag = \
                _reference_from_batches(current, tau, fixed_maps, dim_z, eps)
            if current_adjoints is None:
                pair["taus"][tau] = {
                    "failed": "same_point_reference_failed",
                    "reference_diag": ref_diag,
                    "current_diag": current_diag,
                    "same_point_diag": same_diag,
                }
                continue
            same_point = _production_latent_gradient(
                current, tau, fixed_maps, current_ref, current_adjoints,
                group_multiplier=True)
            lag_metrics = _gradient_comparison(exact, lagged)
            raw_metrics = _gradient_comparison(exact, lagged_raw)
            same_metrics = _gradient_comparison(exact, same_point)
            pair["taus"][tau] = {
                "failed": None,
                "J_reference": j_ref,
                "J_current": j_current,
                "reference_W": float(reference["W"]),
                "current_W": float(current_ref["W"]),
                "production_lagged": lag_metrics,
                "unscaled_regression_control": raw_metrics,
                "same_point_control": same_metrics,
                "exact_exact": (_gradient_comparison(exact_prev, exact)
                                if exact_prev and exact else None),
                "gates": {
                    "same_point_cosine_ge_0_99": bool(
                        same_metrics.get("cosine", -1.0) >= 0.99),
                    "same_point_norm_ratio_0_99_to_1_01": bool(
                        0.99 <= same_metrics.get("norm_ratio", -1.0) <= 1.01),
                    "same_point_relative_error_le_0_01": bool(
                        same_metrics.get("relative_error", float("inf"))
                        <= 0.01),
                    "lagged_cosine_ge_0_90": bool(
                        lag_metrics.get("cosine", -1.0) >= 0.90),
                    "lagged_norm_ratio_0_5_to_2": bool(
                        0.5 <= lag_metrics.get("norm_ratio", -1.0) <= 2.0),
                },
                "reference_diag": ref_diag,
                "current_diag": current_diag,
                "same_point_diag": same_diag,
            }
        report.append(pair)
    return report


def _param_grad_vec(params, divide_by=1.0):
    parts = []
    for p_ in params:
        g = p_.grad
        parts.append((torch.zeros_like(p_) if g is None else g)
                     .detach().double().reshape(-1) / divide_by)
    return torch.cat(parts)


def _parameter_gradient_pass(tgn, decoder, adapter, stream, batch_size,
                             n_neighbors, trace_roots, group_batches,
                             max_groups, seed, builder, fixed_maps, eps,
                             params, prefix=None, heldout_extra=True):
    """Chronological pass WITH grad (review P1).

    Per-group records, all in the compressor PARAMETER space (the only
    shared coordinate system across windows):

    * ``exact_vec``  — grad of this group's exact J(M_w);
    * ``lagged_vec`` — the production one-group-lag surrogate (previous
      group's adjoints on this group's z, group-K cancelled);
    * ``task_vec``   — mean BCE gradient over the group's batches (the
      ordinary-SGD noise baseline);
    * ``reference``  — the group's adjoints/means (close_group semantics);
    * the LAST statistical group keeps its z/p/w/cut_pool (graph
      retained) for the common-probe and moment decomposition.

    One extra group (``heldout_extra``) runs at the end with NO backward:
    its z is detached and its memory state snapshotted so the held-out
    virtual step can replay it after parameter perturbations.
    """
    if tgn.use_memory:
        tgn.memory.__init_memory__()
    if prefix is not None:
        _forward_stream(tgn, adapter, prefix, batch_size, n_neighbors,
                        trace_roots=0)
    for p in params:
        p.requires_grad_(True)
    total_groups = max_groups + (1 if heldout_extra else 0)
    total_batches = min(math.ceil(len(stream.sources) / batch_size),
                        group_batches * total_groups)
    groups = []
    heldout = None
    z_pool, p_pool, w_pool, cut_pool = [], [], [], []
    task_grads_b = []
    previous_reference = None
    group_index = 0
    device = next(iter(params)).device
    with torch.enable_grad():
        for batch_index in range(total_batches):
            start = batch_index * batch_size
            stop = min(len(stream.sources),
                       (batch_index + 1) * batch_size)
            sources = stream.sources[start:stop]
            destinations = stream.destinations[start:stop]
            timestamps = stream.timestamps[start:stop]
            edge_idxs = stream.edge_idxs[start:stop]
            is_heldout = group_index >= max_groups
            if is_heldout and heldout is None:
                heldout = {"memory": (
                    tgn.memory.backup_memory() if tgn.use_memory
                    else None),
                    "batch_start": batch_index}
            if batch_index % group_batches == 0 and not is_heldout:
                # Each statistical group snapshots its memory so the
                # multi-window virtual steps can replay it.  Stored in
                # current_record and merged into the group record at the
                # group close (NOT appended separately — the pairwise
                # report would see a record without gradient vectors).
                current_record = {"memory": (
                    tgn.memory.backup_memory() if tgn.use_memory
                    else None),
                    "batch_start": batch_index}
            roots = select_trace_rows(
                np.zeros(stop - start), trace_roots, seed, batch_index,
                "evenly_spaced")
            adapter.set_trace_source_rows(roots)
            src_emb, _, _ = tgn.compute_temporal_embeddings(
                sources, destinations, destinations, timestamps,
                edge_idxs, n_neighbors)
            if adapter.trace is not None:
                rows = builder.build(adapter.trace,
                                     batch_seed=batch_index)
                for row in rows:
                    if is_heldout:
                        z_pool.append(row.z.detach())
                    else:
                        z_pool.append(row.z)      # graph-connected
                    p_pool.append(fixed_maps.pv(row.context, row.outcome))
                    w_pool.append(float(row.weight))
                    cut_pool.append(tuple(row.cut_id))
            adapter.clear_trace()
            # Task gradient: mean BCE over the group, per-batch gradient
            # collected with the graph retained for the later VJPs.
            if not is_heldout and decoder is not None:
                labels_t = torch.from_numpy(
                    stream.labels[start:stop]).float().to(device)
                pred = decoder(src_emb).sigmoid()
                task_loss = torch.nn.functional.binary_cross_entropy(
                    pred, labels_t)
                for p_ in params:
                    p_.grad = None
                g_task = torch.autograd.grad(
                    task_loss, params, retain_graph=True,
                    allow_unused=True)
                task_grads_b.append([
                    torch.zeros_like(p_).double()
                    if g is None else g.detach().double()
                    for g, p_ in zip(g_task, params)])
            if tgn.use_memory:
                tgn.memory.detach_memory()
            at_boundary = ((batch_index + 1) % group_batches == 0
                           or batch_index == total_batches - 1)
            if at_boundary:
                if is_heldout:
                    heldout["n_batches"] = batch_index \
                        - heldout["batch_start"] + 1
                    heldout["z"] = torch.stack(z_pool) if z_pool else None
                    heldout["p"] = torch.stack(p_pool) if p_pool else None
                    heldout["w"] = torch.tensor(
                        w_pool, dtype=torch.float64,
                        device=device) if w_pool else None
                    heldout["cut_ids"] = list(cut_pool)
                    break
                record = {"index": group_index,
                          "exact_vec": None, "lagged_vec": None,
                          "task_vec": None, "reference": None,
                          "memory": current_record["memory"],
                          "batch_start": current_record["batch_start"]}
                if z_pool:
                    z = torch.stack(z_pool)
                    p = torch.stack(p_pool)
                    w = torch.tensor(w_pool, dtype=torch.float64,
                                     device=z.device)
                    score, diag = _weighted_score_tensor(
                        z, p, w, cut_pool, eps)
                    record["score"] = score.detach() \
                        if score is not None else None
                    record["diag"] = diag
                    prev_ref = previous_reference
                    if score is not None and diag.get("failed") is None:
                        for p_ in params:
                            p_.grad = None
                        score.backward(retain_graph=True)
                        record["exact_vec"] = _param_grad_vec(params)
                        if prev_ref is not None:
                            for p_ in params:
                                p_.grad = None
                            lag_value = kf_vjp_batch(
                                z, p, w,
                                prev_ref["mu_z"],
                                prev_ref["mu_p"],
                                prev_ref["adjoints"]) \
                                * float(group_batches)
                            lag_value.backward(retain_graph=True)
                            record["lagged_vec"] = _param_grad_vec(params)
                        accumulator = WeightedWelford(
                            z.shape[1], p.shape[1])
                        accumulator.add(z.detach(), p.detach(),
                                        w.detach(), cut_pool)
                        res = accumulator.result()
                        _, adjoints, adj_diag = kf_adjoint(
                            res, eps=eps, variant="full_balancing")
                        if adjoints is not None:
                            previous_reference = {
                                "mu_z": res["mu_z"], "mu_p": res["mu_p"],
                                "adjoints": adjoints,
                                "score": score.detach()}
                        record["reference"] = previous_reference
                    if task_grads_b:
                        record["task_vec"] = sum(
                            torch.cat([gg.reshape(-1) for gg in g])
                            for g in task_grads_b) \
                            / float(len(task_grads_b))
                    # The last statistical group keeps its pools for the
                    # common-probe / moment decomposition (graph still
                    # alive thanks to retain_graph).
                    if group_index == max_groups - 1:
                        record["z_pool"] = z
                        record["p_pool"] = p
                        record["w_pool"] = w
                        record["cut_pool"] = list(cut_pool)
                        record["prev_reference"] = prev_ref
                groups.append(record)
                z_pool, p_pool, w_pool, cut_pool = [], [], [], []
                task_grads_b = []
                group_index += 1
    for p in params:
        p.requires_grad_(False)
    return {"groups": groups, "heldout": heldout,
            "n_batches": total_batches}


def _cos(u, v):
    denom = u.norm() * v.norm()
    return float((u * v).sum() / denom.clamp(min=1e-30))


def _adjacent_cosines(pairs):
    """pairs: [(exact_vec, lagged_vec), ...] -> two cosine series.

    ``exact_exact`` compares a window's exact gradient with the NEXT
    window's exact gradient (stability of the window objective itself);
    ``lagged_exact`` compares the production lagged gradient of window t
    (previous window's adjoints on window t) with window t's exact
    gradient (the direction error of the lag mechanism).
    """
    exacts = [p[0] for p in pairs]
    lagged = [p[1] for p in pairs]
    out_exact, out_lagged = [], []
    for t in range(len(pairs) - 1):
        a = exacts[t]
        b = exacts[t + 1]
        if a is not None and b is not None:
            out_exact.append(_cos(a, b))
    for t in range(len(pairs)):
        a = exacts[t]
        b = lagged[t]
        if a is not None and b is not None:
            out_lagged.append(_cos(a, b))
    return out_exact, out_lagged


def _common_probe_report(last_record, params):
    """A_{t-1} and A_t applied to the SAME current-window cuts.

    Isolates the covariance-adjoint drift from sample-Jacobian drift:
    a high cosine means the adjoints agree when facing identical data,
    so window-to-window rotation comes from the samples; a low cosine
    means the whitening adjoint itself drifts.
    """
    out = {"failed": None}
    z = last_record.get("z_pool")
    prev_ref = last_record.get("prev_reference")
    cur_ref = last_record.get("reference")
    if z is None or cur_ref is None:
        out["failed"] = "no_current_reference"
        return out
    p = last_record["p_pool"]
    w = last_record["w_pool"]

    def vjp_grad(ref):
        for p_ in params:
            p_.grad = None
        value = kf_vjp_batch(z, p, w, ref["mu_z"], ref["mu_p"],
                             ref["adjoints"])
        value.backward(retain_graph=True)
        return _param_grad_vec(params)

    combos = {}
    labels = []
    for mu_name, mu_ref in (("mu_prev", prev_ref), ("mu_cur", cur_ref)):
        for a_name, a_ref in (("A_prev", prev_ref), ("A_cur", cur_ref)):
            if mu_ref is None or a_ref is None:
                continue
            combos[mu_name + "_" + a_name] = vjp_grad(
                {"mu_z": mu_ref["mu_z"], "mu_p": mu_ref["mu_p"],
                 "adjoints": a_ref["adjoints"]})
            labels.append((mu_name, a_name))
    cosines = {}
    keys = sorted(combos)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            cosines[ka + "_vs_" + kb] = _cos(combos[ka], combos[kb])
    out["pairwise_cosines"] = cosines
    out["exact_vec_norm"] = float(
        last_record["exact_vec"].norm()) if last_record.get(
            "exact_vec") is not None else None
    # Review: the all-zero cosines were not believable until the probe
    # gradients are shown to be nonzero and repeatable.
    out["grad_norms"] = {k: float(v.norm()) for k, v in combos.items()}
    out["nonzero_counts"] = {
        k: int((v.abs() > 1e-12).sum()) for k, v in combos.items()}
    intersections = {}
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            nz_a = combos[ka].abs() > 1e-12
            nz_b = combos[kb].abs() > 1e-12
            intersections[ka + "_vs_" + kb] = int(
                (nz_a & nz_b).sum())
    out["nonzero_intersections"] = intersections
    # Repeatability: the same (mu, A) recomputed twice must have
    # cosine ~ 1 AND matching norms — a low value here means the 0.0
    # cross-cosines are a numerical artifact, not a signal.
    if cur_ref is not None:
        repeat = vjp_grad({"mu_z": cur_ref["mu_z"],
                           "mu_p": cur_ref["mu_p"],
                           "adjoints": cur_ref["adjoints"]})
        first = combos["mu_cur_A_cur"]
        out["repeat_consistency"] = {
            "cosine": _cos(first, repeat),
            "norm_ratio": float(repeat.norm()
                                / first.norm().clamp(min=1e-30)),
            "relative_error": float(
                (repeat - first).norm()
                / first.norm().clamp(min=1e-30)),
        }
    return out


def _forward_heldout_metrics(tgn, decoder, adapter, stream, batch_start,
                             group_batches, batch_size, n_neighbors,
                             trace_roots, seed, builder, fixed_maps, eps):
    """No-grad replay of a window: (J_joint, J_outcome_contrast, task NLL)."""
    z_pool, p_pool, w_pool, cut_pool = [], [], [], []
    rows_meta = []
    labels_all = []
    probs_all = []
    total = math.ceil(len(stream.sources) / batch_size)
    with torch.no_grad():
        for batch_index in range(batch_start,
                                 min(total, batch_start + group_batches)):
            start = batch_index * batch_size
            stop = min(len(stream.sources), (batch_index + 1) * batch_size)
            roots = select_trace_rows(
                np.zeros(stop - start), trace_roots, seed, batch_index,
                "evenly_spaced")
            adapter.set_trace_source_rows(roots)
            src_emb, _, _ = tgn.compute_temporal_embeddings(
                stream.sources[start:stop], stream.destinations[start:stop],
                stream.destinations[start:stop], stream.timestamps[start:stop],
                stream.edge_idxs[start:stop], n_neighbors)
            if adapter.trace is not None:
                rows = builder.build(adapter.trace,
                                     batch_seed=batch_index)
                for row in rows:
                    z_pool.append(row.z.detach())
                    p_pool.append(fixed_maps.pv(row.context, row.outcome))
                    w_pool.append(float(row.weight))
                    cut_pool.append(tuple(row.cut_id))
                    rows_meta.append((row.context, row.outcome))
            adapter.clear_trace()
            if decoder is not None:
                probs_all.append(decoder(src_emb).sigmoid()
                                 .detach().cpu().numpy().ravel())
                labels_all.append(stream.labels[start:stop])
    if not z_pool:
        return {"failed": "no_heldout_rows"}
    z = torch.stack(z_pool)
    p = torch.stack(p_pool)
    w = torch.tensor(w_pool, dtype=torch.float64, device=z.device)
    score, _ = _weighted_score_tensor(z, p, w, cut_pool, eps)
    out = {
        "J": float(score.detach()) if score is not None else None,
        "J_outcome_contrast": None,
        "task_nll": None,
    }
    # outcome-contrast J: the SAME z rows, P restricted to the
    # outcome-dependent part b_y of the future signature.
    p_contrast = fixed_maps.pv_batch(
        [meta[0] for meta in rows_meta],
        [meta[1] for meta in rows_meta],
        future_mode="outcome_contrast", use_cache=False)
    score_contrast, _ = _weighted_score_tensor(
        z, p_contrast, w, cut_pool, eps)
    out["J_outcome_contrast"] = float(score_contrast.detach()) \
        if score_contrast is not None else None
    if probs_all and labels_all:
        probs = np.clip(np.concatenate(probs_all), 1e-7, 1 - 1e-7)
        labels = np.concatenate(labels_all)
        out["task_nll"] = float(-(labels * np.log(probs)
                                  + (1 - labels) * np.log(1 - probs)).mean())
    return out


def _multi_window_virtual_steps(result, tgn, decoder, adapter, stream,
                                batch_size, n_neighbors, trace_roots,
                                group_batches, seed, builder, fixed_maps,
                                eps, params, ratios=(0.05, 0.1, 0.2)):
    """Diagnostics 1-2 (review-corrected): repeated virtual steps.

    For every adjacent pair (t, t+1): directions from window t, each
    applied at +-eps (1% of the parameter norm), and window t+1's
    (J, J_outcome_contrast, task_nll) measured by replay — window t+1
    never contributed to window t's gradients.

    Direction convention (review): g_J and g_T are unit gradients.
      * ``exact``          = +g_J   (expect D_J > 0);
      * ``task_grad``      = +g_T   (expect D_NLL > 0 — verifies the
                            metric is gradient-sensitive);
      * ``task_update``    = -g_T   (the REAL task optimizer step);
      * ``joint_update_rX``= normalize(-g_T + r g_J)  (the REAL joint
                            optimizer step for relative strength r);
      * ``lagged``         = +g_lagged.

    The +-eps deltas are NEVER averaged together: their average measures
    curvature, not direction.  Per pair and direction the report keeps
    the central directional effect A_i = (delta^+ - delta^-)/2, the
    curvature C_i = (delta^+ + delta^-)/2, and a per-pair median /
    sign-rate / clustered bootstrap CI over pairs.
    """
    groups = result["groups"]
    total_norm = float(sum(p_.norm() ** 2 for p_ in params)) ** 0.5
    # Linearity check (review): three step sizes; the directional
    # derivative must be similar across them and the symmetric term must
    # scale ~ eps^2 — then the largest still-linear step is used for
    # the Task-51 comparisons.
    step_fracs = (0.0025, 0.005, 0.01)
    records = []
    zero_checks = []
    n_pairs = 0
    for t in range(len(groups) - 1):
        g = groups[t]
        nxt = groups[t + 1]
        if g.get("exact_vec") is None or nxt.get("memory") is None:
            continue
        task_vec = g.get("task_vec")
        exact = g["exact_vec"]
        lagged = g.get("lagged_vec")
        g_unit = exact / exact.norm().clamp(min=1e-30)
        directions = {"exact": g_unit}
        if task_vec is not None:
            t_unit = task_vec / task_vec.norm().clamp(min=1e-30)
            directions["task_grad"] = t_unit
            # Review: -g/|g| is the RAW gradient-descent direction, NOT
            # the actual Adam update — name it accordingly.  A true
            # optimizer-fidelity check would need Adam state.
            directions["task_raw_descent"] = -t_unit
            for r in ratios:
                joint = -t_unit + r * g_unit
                directions["joint_raw_descent_r{}".format(r)] = joint \
                    / joint.norm().clamp(min=1e-30)
        if lagged is not None:
            directions["lagged"] = lagged \
                / lagged.norm().clamp(min=1e-30)
        # ---- FULL state snapshot BEFORE any baseline replay (review):
        # memory + RNG + occurrence counter + builder counters.  The
        # parameter base is cloned here too (untouched by pass 1). ----
        state = {
            "memory": nxt["memory"],
            "rng": ckpt._rng_state(),
            "next_oid": getattr(adapter, "_next_oid", 0),
            "tree_counter": getattr(builder, "_tree_counter", 0),
            "build_calls": getattr(builder, "_build_calls", 0),
        }
        theta_base = [p.detach().clone() for p in params]

        def _restore_full():
            if tgn.use_memory and state["memory"] is not None:
                tgn.memory.restore_memory(state["memory"])
            ckpt._restore_rng(state["rng"])
            if hasattr(adapter, "_next_oid"):
                adapter._next_oid = state["next_oid"]
            if hasattr(builder, "_tree_counter"):
                builder._tree_counter = state["tree_counter"]
            if hasattr(builder, "_build_calls"):
                builder._build_calls = state["build_calls"]

        def _set_params(direction, signed_step):
            with torch.no_grad():
                offset = 0
                for p_, base in zip(params, theta_base):
                    n = p_.numel()
                    p_.copy_(base + (
                        direction[offset:offset + n].reshape(
                            p_.shape).float() * signed_step))
                    offset += n

        def _reset_params():
            with torch.no_grad():
                for p_, base in zip(params, theta_base):
                    p_.copy_(base)

        # ---- TWO unperturbed baselines from the SAME state (review):
        # the zero-replay consistency must be a real zero perturbation. ----
        _restore_full()
        before = _forward_heldout_metrics(
            tgn, decoder, adapter, stream, nxt["batch_start"],
            group_batches, batch_size, n_neighbors, trace_roots, seed,
            builder, fixed_maps, eps)
        _restore_full()
        zero_baseline = _forward_heldout_metrics(
            tgn, decoder, adapter, stream, nxt["batch_start"],
            group_batches, batch_size, n_neighbors, trace_roots, seed,
            builder, fixed_maps, eps)
        zero_consistency = {}
        for key in ("J", "J_outcome_contrast", "task_nll"):
            b = before.get(key)
            z = zero_baseline.get(key)
            zero_consistency[key] = (b, z, b == z)
        if before.get("J") is None:
            continue
        n_pairs += 1
        for name, d in directions.items():
            for step_frac in step_fracs:
                step_size = step_frac * total_norm
                for sign in (+1.0, -1.0):
                    try:
                        _restore_full()          # same state as baseline
                        _set_params(d, sign * step_size)
                        after = _forward_heldout_metrics(
                            tgn, decoder, adapter, stream,
                            nxt["batch_start"], group_batches, batch_size,
                            n_neighbors, trace_roots, seed, builder,
                            fixed_maps, eps)
                    finally:
                        _reset_params()          # parameters first
                        _restore_full()          # then memory/RNG/counters
                    for key in ("J", "J_outcome_contrast", "task_nll"):
                        b = before.get(key)
                        a = after.get(key)
                        if b is None or a is None:
                            continue
                        records.append({
                            "pair": t, "direction": name,
                            "sign": int(sign), "metric": key,
                            "before": b, "after": a, "delta": a - b,
                            "step_frac": step_frac,
                            "step_size": step_size,
                        })
        zero_checks.append({"pair": t, "per_metric": zero_consistency})
    out = {"n_pairs": n_pairs,
           "step_fracs": list(step_fracs),
           "records": records,
           "summary": _summarize_virtual_records(records),
           # Disjoint (non-overlapping) window pairs (0,1),(2,3),(4,5),
           # (6,7): too few for significance, but checks whether the
           # conclusions depend on overlapping windows.  Still only a
           # DESCRIPTIVE bootstrap (temporal windows may correlate even
           # when they do not share a group).
           "disjoint_summary": _summarize_disjoint_records(records),
           # Zero-perturbation replay consistency, PER PAIR and per
           # metric: two unperturbed baselines replayed from the SAME
           # saved state must match exactly.
           "zero_replay_consistency": zero_checks}
    return out


def _summarize_disjoint_records(records):
    global _CI_KIND
    saved = _CI_KIND
    _CI_KIND = "descriptive_bootstrap_disjoint_pairs"
    try:
        return _summarize_virtual_records(
            [r for r in records if r["pair"] % 2 == 0])
    finally:
        _CI_KIND = saved


_CI_KIND = "descriptive_bootstrap_overlapping_pairs"


def _summarize_virtual_records(records):
    """Per direction/metric/step_frac: the CENTRAL directional effect
    and its step-normalized derivative.

    Review corrections baked in:
    * the +1/-1 deltas are paired by (pair, direction, metric,
      step_frac) — never by list order; duplicates raise;
    * the denominators are the REAL parameter step
      ``eps_abs = step_frac * |theta|``: D = (d+ - d-)/(2 eps_abs),
      H = (d+ + d-)/eps_abs^2;
    * missing sides are counted per (direction, metric, step_frac);
    * the 95% clustered bootstrap uses the 2.5/97.5 percentiles and is
      labelled DESCRIPTIVE (adjacent pairs share windows — the CIs are
      not independent-sample intervals);
    * the raw per-pair effects are kept (7 pairs is too few to trust a
      CI alone).
    """
    paired = {}
    for r in records:
        key = (r["pair"], r["direction"], r["metric"],
               r.get("step_frac", 0.01))
        sides = paired.setdefault(key, {})
        if int(r["sign"]) in sides:
            raise AssertionError(
                "duplicate virtual-step side: {}".format(key))
        sides[int(r["sign"])] = r
    summary = {}

    def _bootstrap_mean_ci(values, n_boot=2000, seed_boot=0):
        values = np.asarray(values, dtype=np.float64)
        if len(values) == 0:
            return None
        rng = np.random.RandomState(seed_boot)
        means = np.array([
            values[rng.randint(0, len(values), len(values))].mean()
            for _ in range(n_boot)])
        return {"mean": float(values.mean()),
                "median": float(np.median(values)),
                "ci_lo": float(np.percentile(means, 2.5)),
                "ci_hi": float(np.percentile(means, 97.5)),
                "sign_rate": float(np.mean(values > 0.0)),
                "n": int(len(values)),
                "values": [float(v) for v in values]}

    for (pair, direction, metric, step_frac), sides in sorted(
            paired.items()):
        key = direction + "_" + metric + "_step{}".format(step_frac)
        entry = summary.get(key)
        if entry is None:
            entry = {"effects": [], "sym_sums": [],
                     "missing_plus_pairs": 0, "missing_minus_pairs": 0,
                     "step_frac": step_frac, "metric": metric,
                     "direction": direction}
            summary[key] = entry
        if 1 not in sides:
            entry["missing_plus_pairs"] += 1
        if -1 not in sides:
            entry["missing_minus_pairs"] += 1
        if 1 not in sides or -1 not in sides:
            continue
        eps_plus = float(sides[1]["step_size"])
        eps_minus = float(sides[-1]["step_size"])
        if eps_plus <= 0 or abs(eps_plus - eps_minus) > 1e-12:
            raise AssertionError(
                "mismatched step sizes: {} vs {}".format(
                    eps_plus, eps_minus))
        d_plus = sides[1]["delta"]
        d_minus = sides[-1]["delta"]
        entry["step_size"] = eps_plus
        entry["effects"].append((d_plus - d_minus) / 2.0)
        entry["sym_sums"].append(d_plus + d_minus)
    for key, entry in summary.items():
        eps_abs = float(entry.get("step_size", 1.0))
        effects = np.asarray(entry["effects"], dtype=np.float64)
        sym_sums = np.asarray(entry["sym_sums"], dtype=np.float64)
        entry["central_directional_effect"] = _bootstrap_mean_ci(effects)
        entry["directional_derivative"] = _bootstrap_mean_ci(
            effects / eps_abs)
        # Review: H_i = (d+ + d-)/eps_abs^2 (NO extra /2).
        entry["second_directional_derivative_mean"] = float(
            (sym_sums / (eps_abs * eps_abs)).mean())
        entry["n_complete_pairs"] = int(len(effects))
        entry.pop("effects")
        entry.pop("sym_sums")
        entry["ci_kind"] = _CI_KIND
    return summary


def _pairwise_report(pass_result):
    """Five cosine series (review P1): exact-exact, task-task,
    task-exact, task-lagged, lagged-exact — with the task-task series as
    the ordinary-SGD noise baseline instead of an arbitrary 0.9 gate."""
    groups = pass_result["groups"]
    exacts = [g.get("exact_vec") for g in groups]
    lagged = [g.get("lagged_vec") for g in groups]
    tasks = [g.get("task_vec") for g in groups]

    def series(vecs, shift, base=None):
        out = []
        for t in range(len(groups) - shift):
            a = vecs[t]
            b = (base[t + shift] if base is not None
                 else vecs[t + shift])
            if a is not None and b is not None:
                out.append(_cos(a, b))
        return out

    return {
        "exact_exact": series(exacts, 1),
        "task_task": series(tasks, 1),
        "task_exact": [ _cos(a, b) for a, b in zip(tasks, exacts)
                        if a is not None and b is not None],
        "task_lagged": [_cos(a, b) for a, b in zip(tasks, lagged)
                        if a is not None and b is not None],
        "lagged_exact": [_cos(a, b) for a, b in zip(lagged, exacts)
                         if a is not None and b is not None],
    }


def _parameter_space_report(tgn, decoder, adapter, stream, batch_size,
                            n_neighbors, trace_roots, group_batches,
                            max_groups, seed, builder, fixed_maps, eps,
                            params, prefix=None):
    """Review P1 full report on one frozen checkpoint: five cosine
    series, common-probe, moment decomposition and the held-out virtual
    step (the decisive check — does a small step along exact/lagged
    improve the exact J on an INDEPENDENT future window?)."""
    result = _parameter_gradient_pass(
        tgn, decoder, adapter, stream, batch_size, n_neighbors,
        trace_roots, group_batches, max_groups, seed, builder, fixed_maps,
        eps, params, prefix=prefix, heldout_extra=True)
    last = result["groups"][-1] if result["groups"] else {}
    # Order matters: the common-probe backprops through the retained
    # graph of the last group, so it must run BEFORE the held-out
    # virtual step (whose inplace add_/sub_ bumps the autograd version
    # counters of the parameters).
    report = {
        "n_groups": len(result["groups"]),
        "n_batches": result["n_batches"],
        "pairwise": _pairwise_report(result),
        "group_scores": [float(g["score"]) if g.get("score") is not None
                         else None for g in result["groups"]],
        "common_probe": _common_probe_report(last, params),
        "multi_window_virtual_steps": _multi_window_virtual_steps(
            result, tgn, decoder, adapter, stream, batch_size, n_neighbors,
            trace_roots, group_batches, seed, builder, fixed_maps, eps,
            params),
    }
    # Ridge-after-condition diagnostics from the last group's score diag.
    diag = last.get("diag") or {}
    report["condition"] = {
        "zz_min_eig": diag.get("zz_min_eig"),
        "zz_max_eig": diag.get("zz_max_eig"),
        "zz_cond": diag.get("zz_cond"),
        "zz_r_eff": diag.get("zz_r_eff"),
        "pp_r_eff": diag.get("pp_r_eff"),
        "ridge_eps": eps,
        "zz_cond_after_ridge": (
            (diag["zz_max_eig"] + eps) / (diag["zz_min_eig"] + eps)
            if diag.get("zz_max_eig") is not None
            and diag.get("zz_min_eig") is not None else None),
    }
    return report


def _exact_stability_sweep(tgn, decoder, adapter, stream, batch_size,
                           n_neighbors, trace_roots, window_sizes,
                           max_groups, seed, builder, fixed_maps, eps,
                           params, prefix=None):
    """Parameter-space gradient comparisons for several window sizes.

    ``exact_exact`` — adjacent windows' EXACT gradients: if these are
    also near-orthogonal, the window CCA objective itself is noisy and a
    two-pass replay would only chase random directions (then fix window
    size / measurement dims first).  ``lagged_exact`` — the production
    lagged gradient against the current window's exact gradient.  The
    task-task series is the ordinary-SGD noise baseline.
    """
    out = {}
    for window in window_sizes:
        result = _parameter_gradient_pass(
            tgn, decoder, adapter, stream, batch_size, n_neighbors,
            trace_roots, window, max_groups, seed, builder, fixed_maps,
            eps, params, prefix=prefix, heldout_extra=False)
        series = _pairwise_report(result)
        entry = {"n_groups": len(result["groups"])}
        for name in ("exact_exact", "task_task", "task_exact",
                     "task_lagged", "lagged_exact"):
            vals = series[name]
            entry[name + "_cosine_mean"] = float(np.mean(vals)) \
                if vals else None
            entry[name + "_cosine_min"] = float(np.min(vals)) \
                if vals else None
            entry[name + "_cosine_list"] = [float(v) for v in vals]
        out[str(window)] = entry
    return out


def _label_residual_values(rows, pi):
    """Class-balanced residual y_tilde = (y - pi) / sqrt(pi (1 - pi))."""
    scale = math.sqrt(pi * (1.0 - pi)) if 0.0 < pi < 1.0 else 1.0
    return [((1.0 if row.outcome > 0.5 else 0.0) - pi) / scale
            for row in rows]


def _context_residual_values(rows, fixed_maps):
    """Context-conditional residual y_tilde = y - p_hat(y=1 | C).

    p_hat comes from an in-sample logistic fit on phi_C — an UPPER-bound
    control: in-sample residuals remove everything context can predict,
    so passing this test is stronger than passing balanced residuals,
    while a failure here cannot be blamed on context predictability.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None
    feats = np.stack([fixed_maps.context_vector(
        row.context).detach().cpu().numpy().ravel() for row in rows])
    ys = np.asarray([1.0 if row.outcome > 0.5 else 0.0 for row in rows],
                    dtype=np.float64)
    if len(np.unique(ys)) < 2:
        return None
    clf = LogisticRegression(max_iter=500, C=1.0).fit(feats, ys)
    p_hat = clf.predict_proba(feats)[:, 1]
    return (ys - p_hat).tolist()


def _population_audit(rows, candidates, stats, stream):
    outcome_weight = defaultdict(float)
    outcome_count = Counter()
    node_weight = defaultdict(float)
    for row in rows:
        outcome_weight[tuple(row.outcome_id)] += float(row.weight)
        outcome_count[tuple(row.outcome_id)] += 1
        node_weight[int(row.node)] += float(row.weight)

    def kish(weight_by_cluster):
        values = np.asarray(list(weight_by_cluster.values()), dtype=np.float64)
        return float(values.sum() ** 2 / np.square(values).sum()) \
            if len(values) and np.square(values).sum() > 0 else 0.0

    outcomes_by_tree_h = defaultdict(set)
    node_time_by_tree = defaultdict(set)
    path_relations = set()
    for row in rows:
        outcomes_by_tree_h[(int(row.tree_id), int(row.horizon))].add(
            tuple(row.outcome_id))
        node_time_by_tree[int(row.tree_id)].add(
            (int(row.node), float(row.time)))
        path_relations.update(int(rel) for rel, _ in row.context.get("path", []))

    cross_tau_same_outcome = [len(values) == 1
                              for values in outcomes_by_tree_h.values()]
    cross_tau_same_node_time = [len(values) == 1
                                for values in node_time_by_tree.values()]
    time_quantiles = np.quantile(stream.timestamps, [0.25, 0.5, 0.75])
    future_by_quartile = defaultdict(Counter)
    for candidate in candidates:
        quartile = int(np.searchsorted(
            time_quantiles, candidate["time"], side="right")) + 1
        future_by_quartile[str(quartile)][str(candidate["n_future"])] += 1

    raw = stats.get("raw_candidates", {})
    valid = stats.get("valid_rows", {})
    missing = stats.get("missing_horizons", {})
    return {
        "target_definition": "first two strictly later incident events",
        "parent_pullback_records_present": False,
        "path_relation_values": sorted(path_relations),
        "paths_are_self_only": bool(path_relations <= {0}),
        "cross_tau_same_outcome_fraction": float(np.mean(
            cross_tau_same_outcome)) if cross_tau_same_outcome else None,
        "cross_tau_same_node_time_fraction": float(np.mean(
            cross_tau_same_node_time)) if cross_tau_same_node_time else None,
        "n_rows": len(rows),
        "n_cuts": len({row.cut_id for row in rows}),
        "n_trees": len({row.tree_id for row in rows}),
        "n_nodes": len(node_weight),
        "n_unique_outcomes": len(outcome_weight),
        "outcome_cluster_ess": kish(outcome_weight),
        "node_cluster_ess": kish(node_weight),
        "outcome_max_row_multiplicity": max(
            outcome_count.values()) if outcome_count else 0,
        "outcome_max_total_weight": max(
            outcome_weight.values()) if outcome_weight else 0.0,
        "candidate_future_count_by_time_quartile": {
            quartile: dict(counts)
            for quartile, counts in future_by_quartile.items()},
        "builder_stats": {
            "raw_candidates": {str(key): value for key, value in raw.items()},
            "valid_rows": {str(key): value for key, value in valid.items()},
            "missing_horizons": {
                str(key): value for key, value in missing.items()},
        },
    }


def main():
    args = parse_args()
    if args.groups < 2:
        raise ValueError("--groups must be at least 2")
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    config, cli, dataset, components = _load_frozen_run(args, device)
    full, train, val, _ = dataset.splits()
    stream = train if args.audit_split == "train" else val
    prefix = None if args.audit_split == "train" else train

    tgn = components["tgn"]
    adapter = components["adapter"]
    fixed_maps = components["fixed_maps"]
    rpbe_cfg = components["rpbe_cfg"]
    batch_size = int(cli.get("bs", 200))
    n_neighbors = int(cli.get("n_degree", 5))
    trace_roots = int(args.trace_roots or cli.get("trace_roots", 32))
    group_batches = int(args.group_batches or
                        cli.get("kf_group_batches") or math.ceil(
                            int(cli.get("kf_min_abs", 1024))
                            / max(1, trace_roots)))

    if tgn.use_memory:
        tgn.memory.__init_memory__()
    if prefix is not None:
        _forward_stream(tgn, adapter, prefix, batch_size, n_neighbors,
                        trace_roots=0)

    future_index = JodieFutureIndex(stream)
    builder = JodieCutBuilder(
        future_index, stage=NODE_CLASS, cuts_per_tau=10 ** 9,
        seed=args.seed, n_observations=2)
    sweep_sizes = [int(part) for part in args.window_sweep.split(",")
                   if part.strip()]
    max_needed = max([group_batches] + sweep_sizes)
    max_batches = min(math.ceil(len(stream.sources) / batch_size),
                      max_needed * args.groups)
    batch_rows, candidates, stats = _forward_stream(
        tgn, adapter, stream, batch_size, n_neighbors,
        trace_roots=trace_roots, future_index=future_index, builder=builder,
        max_batches=max_batches, seed=args.seed)
    groups = [batch_rows[start:start + group_batches]
              for start in range(0, len(batch_rows), group_batches)]
    groups = groups[:args.groups]
    all_rows = [row for group in groups for batch in group for row in batch]
    if not all_rows:
        raise RuntimeError("no valid future rows were collected")

    report = {
        "run_dir": str(Path(args.run_dir).resolve()),
        "checkpoint_epoch": int(torch.load(
            Path(args.run_dir) / "best.pt", map_location="cpu",
            weights_only=False).get("epoch", -1)),
        "checkpoint_loss_units": config.get(
            "rpbe_loss_units", "legacy_or_unknown_pre_2026-08-30"),
        "audit_split": args.audit_split,
        "batch_size": batch_size,
        "trace_roots": trace_roots,
        "group_batches": group_batches,
        "groups_collected": len(groups),
        "fixed_measurement": fixed_maps.isolation_fingerprint(),
        "gradient_estimator_audited":
            "raw_batch_vjp_sum_with_group_K_cancellation",
        "score_variant_audited": "full_balancing",
        "measurement_future_decomposition":
            fixed_maps.future_decomposition(),
        "population": _population_audit(all_rows, candidates, stats, stream),
        "scores": _score_audit(
            all_rows, fixed_maps, rpbe_cfg.ridge_eps,
            args.permutations, args.seed,
            pi_train=float(np.mean(
                dataset.train.labels > 0.5))),
        "lagged_gradient": _lag_audit(
            groups, fixed_maps, rpbe_cfg.ridge_eps),
        "parameter_space": _parameter_space_report(
            tgn, components["decoder"], adapter, stream, batch_size,
            n_neighbors, trace_roots, group_batches, args.groups,
            args.seed, builder, fixed_maps, rpbe_cfg.ridge_eps,
            list(components["compressor"].parameters()), prefix=prefix),
        "exact_stability_sweep": _exact_stability_sweep(
            tgn, components["decoder"], adapter, stream, batch_size,
            n_neighbors, trace_roots, sweep_sizes, args.groups,
            args.seed, builder, fixed_maps, rpbe_cfg.ridge_eps,
            list(components["compressor"].parameters()),
            prefix=prefix),
        "interpretation_contract": {
            "balancing_claim":
                "finite-feature joint predictive CCA energy",
            "not_established_without_extra_assumptions": [
                "all-context conditional sufficiency",
                "parent-pullback closure",
                "arbitrary-depth predictive quotient equivalence",
            ],
        },
    }
    out_json = Path(args.out_json) if args.out_json else (
        Path(args.run_dir) / "diagnostics"
        / "loss_diagnosis_{}.json".format(args.audit_split))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))
    print("wrote {}".format(out_json), flush=True)


if __name__ == "__main__":
    main()
