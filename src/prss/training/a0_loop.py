"""A0 four-phase training loop on the official TGN host (theory doc 2026-08-20).

Phase A calibrates per-interface rank-r coordinate maps R_tau from the
conditional future moment on the first train window; phase B fits per-
constructor recursive operators B_sigma by one-shot convex ridge on the next
window; phase C audits prediction/closure/sibling-support/path-gain on the
third window and judges the G0-G3 gates; phase D trains two readout heads on
the SAME frozen host forward — the A0 readout on z_root = R x_root and the
baseline decoder on x_root (the train_jodie vanilla head, the run's built-in
self-check) — each with its own val early stopping, then a zero-memory
train+val replay before the held-out test.

The host stays frozen throughout; the vanilla PRSS core only produces the
trace.  R and B̂ are calibration results (never gradient parameters); the
evaluation path needs no trace (z_root = R · src_emb is a matrix product).
"""

import copy
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from prss.a0.audit import (ResidualAccumulator, evaluate_gates,
                           path_gain_report, proper_score_regret)
from prss.a0.operators import (OperatorRidge, TensorSketchFeatures, chi_sigma,
                               chi_width)
from prss.a0.probes import A0Probes, propagate_root_labels, stack_by_tau
from prss.a0.quotient import A0Quotient
from prss.a0.weights import DensityRatioWeights
from prss.hosts.jodie_tgn import TAU_TEMPLATE
from prss.hosts.official_tgn import MLP
from prss.training.jodie_loop import metric_bundle, select_trace_rows


class A0NodeClassificationLoop:
    """Owns the frozen TGN stream, the A0 calibration/audit stages, and the
    phase-D dual readout training."""

    def __init__(self, *, tgn, adapter, prss_core, probes, device, batch_size,
                 n_neighbors, trace_roots, trace_mode, rank_r, lambda_x,
                 lambda_gamma, lambda_audit, frac_a, frac_b, frac_c,
                 d_slice_only, gates, gate_mode, monitor, seed, out_dir,
                 lr, n_epoch, patience, drop_out, selection_metric="auc",
                 use_weights=False, weight_calib_frac=0.2, w_min=0.1,
                 w_max=10.0, chi_mode="meanpool", sketch_s=64,
                 deploy_events=0):
        self.tgn = tgn
        self.adapter = adapter
        self.prss_core = prss_core
        self.probes = probes
        self.device = device
        self.batch_size = int(batch_size)
        self.n_neighbors = int(n_neighbors)
        self.trace_roots = int(trace_roots)
        self.trace_mode = trace_mode
        self.rank_r = int(rank_r)
        self.lambda_x = float(lambda_x)
        self.lambda_gamma = float(lambda_gamma)
        self.lambda_audit = float(lambda_audit)
        self.frac_a = float(frac_a)
        self.frac_b = float(frac_b)
        self.frac_c = float(frac_c)
        self.d_slice_only = bool(d_slice_only)
        self.gates = gates or {}
        self.gate_mode = gate_mode
        self.use_weights = bool(use_weights)
        self.weight_calib_frac = float(weight_calib_frac)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.chi_mode = chi_mode
        if self.chi_mode not in ("meanpool", "sketch"):
            raise ValueError("unknown chi_mode {}".format(self.chi_mode))
        self.sketch_feats = (TensorSketchFeatures(self.rank_r,
                                                  probes.d_context,
                                                  s=sketch_s, seed=seed)
                             if self.chi_mode == "sketch" else None)
        self.deploy_events = int(deploy_events)
        self.weight_report = {}
        self.monitor = monitor
        self.seed = int(seed)
        self.out_dir = out_dir
        self.lr = float(lr)
        self.n_epoch = int(n_epoch)
        self.patience = int(patience)
        self.drop_out = float(drop_out)
        self.selection_metric = selection_metric
        if self.frac_a + self.frac_b + self.frac_c >= 1.0:
            raise ValueError("frac_a+frac_b+frac_c must leave data for phase D")

        self.host_dim = int(tgn.embedding_dimension)
        self.n_layers = int(tgn.embedding_module.n_layers)
        self.taus = [TAU_TEMPLATE.format(l) for l in range(self.n_layers + 1)]
        d_context = probes.d_context
        self.quotients: Dict[str, A0Quotient] = {
            tau: A0Quotient(tau, p=self.host_dim, m=2 * d_context)
            for tau in self.taus}
        self.operators: Dict[tuple, OperatorRidge] = {}
        s = (self.sketch_feats.width() if self.sketch_feats is not None
             else chi_width(self.rank_r, d_context))
        for l in range(1, self.n_layers + 1):
            key = (TAU_TEMPLATE.format(l - 1), TAU_TEMPLATE.format(l))
            self.operators[key] = OperatorRidge(
                child_tau=key[0], parent_tau=key[1], s=s, r=self.rank_r)
        self.top_tau = TAU_TEMPLATE.format(self.n_layers)
        self._epoch_rows: List[Dict] = []

    def _chi(self, z_source, z_neigh_list, a_parent):
        """Interaction features for one parent occurrence (mode-dependent)."""
        if self.sketch_feats is not None:
            z_neigh = (torch.stack(z_neigh_list, dim=0) if z_neigh_list
                       else torch.zeros(0, self.rank_r,
                                        dtype=z_source.dtype,
                                        device=z_source.device))
            return self.sketch_feats.chi(z_source, z_neigh, a_parent)
        z_neigh_mean = (torch.stack(z_neigh_list).mean(dim=0) if z_neigh_list
                        else torch.zeros_like(z_source))
        return chi_sigma(z_source, z_neigh_mean, a_parent)

    # ------------------------------------------------------------- stream prims
    def reset_memory(self):
        if self.tgn.use_memory:
            self.tgn.memory.__init_memory__()

    def _num_batches(self, data):
        return math.ceil(len(data.sources) / self.batch_size)

    def _forward(self, sources, dests, times, edge_idxs):
        with torch.no_grad():
            return self.tgn.compute_temporal_embeddings(
                sources, dests, dests, times, edge_idxs, self.n_neighbors)[0]

    def _collect_trace(self, data, k, labels_np):
        """Trace one batch's cut-trees; returns stacked per-tau rows + oid labels."""
        trace_rows = select_trace_rows(labels_np, self.trace_roots, self.seed,
                                       k, self.trace_mode)
        if not trace_rows:
            return None
        self.adapter.set_trace_source_rows(trace_rows)
        s, e = k * self.batch_size, min(len(data.sources),
                                        (k + 1) * self.batch_size)
        self._forward(data.sources[s:e], data.destinations[s:e],
                      data.timestamps[s:e], data.edge_idxs[s:e])
        trace = self.adapter.trace
        if trace is None or not trace.roots:
            self.adapter.clear_trace()
            return None
        root_labels = np.asarray(labels_np)[trace.root_rows]
        oid_labels = propagate_root_labels(trace, root_labels)
        stacks = stack_by_tau(trace, oid_labels, self.probes)
        return trace, stacks, oid_labels

    def _z_by_oid(self, trace):
        z_by_oid = {}
        for occ in trace.occurrences.values():
            z_by_oid[occ.occurrence_id] = self.quotients[occ.tau].project(
                occ.state.candidate)
        return z_by_oid

    # ------------------------------------------------------------------ phase A
    @torch.no_grad()
    def calibrate_phase_A(self, train, steps: int) -> Dict:
        self.reset_memory()
        self.tgn.eval()
        self.adapter.clear_trace()
        # Context-overlap weights (doc 5.4): the first part of the A window
        # fits per-tau density ratios; the rest accumulates the importance-
        # corrected moments.  Unweighted mode keeps the previous behavior.
        fit_weights = None
        cal_rows = {tau: [] for tau in self.taus}  # (X, A) pairs for the fit
        if self.use_weights:
            n_cal = max(1, int(steps * self.weight_calib_frac))
            quota = max(1, math.ceil(2000 / max(n_cal, 1)))
            for k in range(n_cal):
                s, e = k * self.batch_size, min(len(train.sources),
                                                (k + 1) * self.batch_size)
                collected = self._collect_trace(train, k, train.labels[s:e])
                if collected is None:
                    continue
                _, stacks, _ = collected
                for tau, st in stacks.items():
                    if len(cal_rows[tau]) < quota * n_cal:
                        n_take = min(quota, int(st["X"].shape[0]))
                        cal_rows[tau].append((st["X"][:n_take].detach(),
                                              st["A"][:n_take].detach()))
                self.adapter.clear_trace()
            fit_weights = {}
            self.weight_report = {}
            for tau in self.taus:
                if not cal_rows[tau]:
                    continue
                x_rows = torch.cat([r[0] for r in cal_rows[tau]], dim=0)
                a_rows = torch.cat([r[1] for r in cal_rows[tau]], dim=0)
                model = DensityRatioWeights(self.w_min, self.w_max)
                model.fit(x_rows, a_rows)
                w = model.weights(x_rows, a_rows)
                fit_weights[tau] = model
                self.weight_report[tau] = {
                    "n_cal": int(x_rows.shape[0]),
                    "ess": DensityRatioWeights.ess(w),
                    "ess_frac": DensityRatioWeights.ess(w) / x_rows.shape[0],
                    "w_mean": float(w.mean().item()),
                    "w_max": float(w.max().item()),
                }
        acc_start = max(1, int(steps * self.weight_calib_frac)) \
            if self.use_weights else 0
        for k in range(acc_start, steps):
            s, e = k * self.batch_size, min(len(train.sources),
                                            (k + 1) * self.batch_size)
            collected = self._collect_trace(train, k, train.labels[s:e])
            if collected is None:
                continue
            _, stacks, _ = collected
            for tau, st in stacks.items():
                w = None
                if fit_weights and tau in fit_weights:
                    w = fit_weights[tau].weights(st["X"], st["A"])
                self.quotients[tau].accumulate(st["X"], st["U"], w=w)
            self.adapter.clear_trace()
        solve_info = {}
        for tau, q in self.quotients.items():
            tail = q.solve(self.rank_r, self.lambda_x)
            solve_info[tau] = {"rank_tail": tail, "n": q.n}
            if self.monitor is not None:
                self.monitor.write_step({"phase": "A", "event": "solve",
                                         "tau": tau, "rank_tail": tail,
                                         "n": q.n})
        return solve_info

    # ------------------------------------------------------------------ phase B
    @torch.no_grad()
    def fit_phase_B(self, train, steps: int, offset: int = 0) -> Dict:
        self.tgn.eval()
        acc: Dict[tuple, Dict[str, list]] = {
            key: {"phi": [], "z": []} for key in self.operators}
        for k in range(steps):
            s, e = k * self.batch_size, min(len(train.sources),
                                            (k + 1) * self.batch_size)
            collected = self._collect_trace(train, offset + k,
                                            train.labels[s:e])
            if collected is None:
                continue
            trace, _, oid_labels = collected
            z_by_oid = self._z_by_oid(trace)
            for occ in trace.occurrences.values():
                if occ.metadata.get("layer", 0) < 1:
                    continue
                if occ.occurrence_id not in oid_labels:
                    continue  # padded-neighbor orphan: no root label (probes.py)
                z_source = None
                z_neigh = []
                for cid, rel in zip(occ.children, occ.child_relations):
                    zc = z_by_oid.get(cid)
                    if zc is None:
                        continue
                    if rel == 0:
                        z_source = zc
                    else:
                        z_neigh.append(zc)
                if z_source is None:
                    continue
                a_parent = self.probes.probe_a(occ.local_features.detach())
                phi = self._chi(z_source, z_neigh, a_parent)
                key = self._operator_key(occ)
                acc[key]["phi"].append(phi)
                acc[key]["z"].append(z_by_oid[occ.occurrence_id])
            self.adapter.clear_trace()
        fit_info = {}
        for key, lists in acc.items():
            op = self.operators[key]
            if lists["phi"]:
                op.accumulate(torch.stack(lists["phi"], dim=0),
                              torch.stack(lists["z"], dim=0))
            op.solve(self.lambda_gamma)
            fit_info["{}->{}".format(*key)] = op.snapshot()
            if self.monitor is not None:
                self.monitor.write_step({"phase": "B", "event": "solve",
                                         "sigma": "{}->{}".format(*key),
                                         **op.snapshot()})
        return fit_info

    def _operator_key(self, occ):
        layer = int(occ.metadata.get("layer", 0))
        child_tau = TAU_TEMPLATE.format(layer - 1)
        key = (child_tau, occ.tau)
        if key not in self.operators:
            raise KeyError("no operator for {}->{}".format(*key))
        return key

    # ------------------------------------------------------------------ phase C
    @torch.no_grad()
    def audit_phase_C(self, train, steps: int, offset: int = 0) -> Dict:
        self.tgn.eval()
        m = 2 * self.probes.d_context
        pred_z = {tau: ResidualAccumulator(self.rank_r, m)
                  for tau in self.taus}
        pred_x = {tau: ResidualAccumulator(self.host_dim, m)
                  for tau in self.taus}
        closure_sums = {key: 0.0 for key in self.operators}
        closure_n = {key: 0 for key in self.operators}
        # Proper-score rows (uniform per-batch quota keeps the window bounded).
        proper_quota = max(1, math.ceil(20000 / max(steps, 1)))
        proper_z = {tau: [] for tau in self.taus}
        proper_x = {tau: [] for tau in self.taus}
        proper_y = {tau: [] for tau in self.taus}
        lev_rows = {key: [] for key in self.operators}
        total_rows = 0
        for k in range(steps):
            s, e = k * self.batch_size, min(len(train.sources),
                                            (k + 1) * self.batch_size)
            collected = self._collect_trace(train, offset + k,
                                            train.labels[s:e])
            if collected is None:
                continue
            trace, stacks, oid_labels = collected
            z_by_oid = self._z_by_oid(trace)
            for tau, st in stacks.items():
                z_rows = self.quotients[tau].project(st["X"])
                pred_z[tau].accumulate(z_rows, st["U"])
                pred_x[tau].accumulate(st["X"], st["U"])
                total_rows += int(st["X"].shape[0])
                n_take = min(proper_quota, int(st["X"].shape[0]))
                proper_z[tau].append(z_rows[:n_take].detach())
                proper_x[tau].append(st["X"][:n_take].detach().to(
                    dtype=torch.float64))
                proper_y[tau].append(st["Y"][:n_take].detach())
            for occ in trace.occurrences.values():
                if occ.metadata.get("layer", 0) < 1:
                    continue
                if occ.occurrence_id not in oid_labels:
                    continue  # padded-neighbor orphan: no root label (probes.py)
                z_source = None
                z_neigh = []
                for cid, rel in zip(occ.children, occ.child_relations):
                    zc = z_by_oid.get(cid)
                    if zc is None:
                        continue
                    if rel == 0:
                        z_source = zc
                    else:
                        z_neigh.append(zc)
                if z_source is None:
                    continue
                a_parent = self.probes.probe_a(occ.local_features.detach())
                phi = self._chi(z_source, z_neigh, a_parent)
                key = self._operator_key(occ)
                op = self.operators[key]
                z_rec = op.predict(phi.unsqueeze(0))[0]
                z_rich = z_by_oid[occ.occurrence_id]
                sigma = self.quotients[occ.tau].sigma
                diff = sigma * (z_rich - z_rec)
                closure_sums[key] += float((diff * diff).sum().item())
                closure_n[key] += 1
                if len(lev_rows[key]) < proper_quota:
                    lev_rows[key].append(phi.detach())
            self.adapter.clear_trace()

        audit = {"ess": total_rows}
        if self.weight_report:
            audit["context_weights"] = self.weight_report
            audit["ess_frac_min"] = min(
                v["ess_frac"] for v in self.weight_report.values())
        else:
            audit["ess_frac_min"] = 1.0
        per_tau_pred = {}
        rank_tails = []
        for tau in self.taus:
            rank_tails.append(self.quotients[tau].rank_tail)
            per_tau_pred[tau] = {
                "unrestricted_ridge_residual":
                    pred_x[tau].relative_residual(self.lambda_audit),
                "rank_r_ridge_residual":
                    pred_z[tau].relative_residual(self.lambda_audit),
            }
            per_tau_pred[tau]["prediction_gap"] = max(
                0.0,
                per_tau_pred[tau]["rank_r_ridge_residual"]
                - per_tau_pred[tau]["unrestricted_ridge_residual"])
        audit["rank_tail_max"] = max(rank_tails) if rank_tails else 0.0
        audit["prediction_by_tau"] = per_tau_pred
        # Proper-score regret per interface (doc 5.7.2): the compressed vs
        # rich readout gap under the log/Brier scoring rules, depth-stratified.
        proper_regret = {}
        for tau in self.taus:
            if proper_z[tau]:
                proper_regret[tau] = proper_score_regret(
                    torch.cat(proper_z[tau], dim=0),
                    torch.cat(proper_x[tau], dim=0),
                    torch.cat(proper_y[tau], dim=0),
                    lambda_reg=self.lambda_audit)
        audit["proper_score_regret_by_tau"] = proper_regret
        closure_by_sigma = {}
        for key, op in self.operators.items():
            name = "{}->{}".format(*key)
            closure_by_sigma[name] = {
                "closure_residual": (closure_sums[key] / max(closure_n[key], 1)),
                "n_closure_rows": closure_n[key],
                "condition_number": op.condition_number,
                "effective_rank": op.effective_rank,
            }
        audit["closure_by_sigma"] = closure_by_sigma
        audit["closure_residual_max"] = max(
            (v["closure_residual"] for v in closure_by_sigma.values()),
            default=0.0)
        audit["sibling_support"] = {
            "{}->{}".format(*key): {
                "condition_number": op.condition_number,
                "effective_rank": op.effective_rank,
            }
            for key, op in self.operators.items()}
        # Train-to-audit leverage/OOD scores (doc 5.7.6): h_i in [0,1] for
        # in-support rows with mean s/n; rows with h_i >> 1 are OOD.
        for key, op in self.operators.items():
            name = "{}->{}".format(*key)
            if lev_rows[key]:
                h = op.leverage(torch.stack(lev_rows[key], dim=0))
                audit["sibling_support"][name].update({
                    "leverage_mean": float(h.mean().item()),
                    "leverage_max": float(h.max().item()),
                    "leverage_frac_gt_1": float((h > 1.0).float().mean().item()),
                })
        audit.update(path_gain_report(list(self.operators.values())))
        gate_out = evaluate_gates(audit, self.gates, mode=self.gate_mode,
                                  phase="C")
        audit["gates"] = gate_out
        if self.monitor is not None:
            self.monitor.write_step({"phase": "C", "event": "audit",
                                     **{k: v for k, v in audit.items()
                                        if k != "gates"}})
        return audit

    # ------------------------------------------------------------------ phase D
    def _make_heads(self):
        readout = MLP(self.rank_r, drop=self.drop_out).to(self.device)
        decoder = MLP(self.host_dim, drop=self.drop_out).to(self.device)
        return readout, decoder

    @torch.no_grad()
    def _evaluate_split(self, split, readout, decoder, *, reset=False):
        if reset:
            self.reset_memory()
        self.tgn.eval()
        readout.eval()
        decoder.eval()
        self.adapter.clear_trace()
        probs_a, probs_b, labels = [], [], []
        for k in range(self._num_batches(split)):
            s, e = k * self.batch_size, min(len(split.sources),
                                            (k + 1) * self.batch_size)
            src_emb = self._forward(split.sources[s:e], split.destinations[s:e],
                                    split.timestamps[s:e], split.edge_idxs[s:e])
            z_root = self.quotients[self.top_tau].project(src_emb)
            probs_a.append(readout(z_root.float()).sigmoid().cpu().numpy())
            probs_b.append(decoder(src_emb).sigmoid().cpu().numpy())
            labels.append(split.labels[s:e])
        labels_np = np.concatenate(labels)
        return (metric_bundle(labels_np, np.concatenate(probs_a)),
                metric_bundle(labels_np, np.concatenate(probs_b)))

    @torch.no_grad()
    def _replay_split(self, split):
        self.tgn.eval()
        self.adapter.clear_trace()
        for k in range(self._num_batches(split)):
            s, e = k * self.batch_size, min(len(split.sources),
                                            (k + 1) * self.batch_size)
            self._forward(split.sources[s:e], split.destinations[s:e],
                          split.timestamps[s:e], split.edge_idxs[s:e])

    def train_phase_D(self, train, val, test) -> Dict:
        readout, decoder = self._make_heads()
        opt_a = torch.optim.Adam(readout.parameters(), lr=self.lr)
        opt_b = torch.optim.Adam(decoder.parameters(), lr=self.lr)

        if self.d_slice_only:
            start_batch = (math.ceil((self.frac_a + self.frac_b + self.frac_c)
                                     * len(train.sources) / self.batch_size))
        else:
            start_batch = 0
        d_batches = list(range(start_batch, self._num_batches(train)))

        best_a = {"score": -1.0, "epoch": -1, "state": None}
        best_b = {"score": -1.0, "epoch": -1, "state": None}
        bad_a = bad_b = 0
        for epoch in range(self.n_epoch):
            self.reset_memory()
            self.tgn.eval()
            readout.train()
            decoder.train()
            total_a = total_b = 0.0
            n_steps = 0
            for k in d_batches:
                s, e = k * self.batch_size, min(len(train.sources),
                                                (k + 1) * self.batch_size)
                labels_t = torch.from_numpy(train.labels[s:e]).float().to(
                    self.device)
                opt_a.zero_grad(set_to_none=True)
                opt_b.zero_grad(set_to_none=True)
                src_emb = self._forward(
                    train.sources[s:e], train.destinations[s:e],
                    train.timestamps[s:e], train.edge_idxs[s:e])
                z_root = self.quotients[self.top_tau].project(src_emb)
                loss_a = F.binary_cross_entropy(
                    readout(z_root.float()).sigmoid(), labels_t)
                loss_b = F.binary_cross_entropy(decoder(src_emb).sigmoid(),
                                                labels_t)
                loss_a.backward()
                loss_b.backward()
                opt_a.step()
                opt_b.step()
                total_a += float(loss_a.detach())
                total_b += float(loss_b.detach())
                n_steps += 1
                if self.monitor is not None:
                    self.monitor.validate_losses(
                        {"task_a0": float(loss_a.detach()),
                         "task_baseline": float(loss_b.detach())},
                        epoch * len(d_batches) + k)
            val_a, val_b = self._evaluate_split(val, readout, decoder,
                                                reset=False)
            row = {
                "phase": "D",
                "epoch": epoch,
                "a0_train_loss": total_a / max(n_steps, 1),
                "baseline_train_loss": total_b / max(n_steps, 1),
                "a0_val": val_a,
                "baseline_val": val_b,
            }
            if self.monitor is not None:
                self.monitor.write_epoch(row)
            self._epoch_rows.append(row)
            improved_a = val_a.get(self.selection_metric, -1.0) > best_a["score"]
            improved_b = val_b.get(self.selection_metric, -1.0) > best_b["score"]
            for best, metrics, head, improved in (
                    (best_a, val_a, readout, improved_a),
                    (best_b, val_b, decoder, improved_b)):
                if improved:
                    best["score"] = float(metrics[self.selection_metric])
                    best["epoch"] = epoch
                    best["state"] = copy.deepcopy(head.state_dict())
            bad_a = 0 if improved_a else bad_a + 1
            bad_b = 0 if improved_b else bad_b + 1
            if min(bad_a, bad_b) >= self.patience:
                break

        readout.load_state_dict(best_a["state"])
        decoder.load_state_dict(best_b["state"])
        self.reset_memory()
        self._replay_split(train)
        self._replay_split(val)
        self.adapter.clear_trace()
        test_a, test_b = self._evaluate_split(test, readout, decoder,
                                              reset=False)
        self.adapter.clear_trace()
        out = {
            "a0_readout": {"best_epoch": best_a["epoch"],
                           "best_val_score": best_a["score"],
                           "test": test_a},
            "baseline_decoder": {"best_epoch": best_b["epoch"],
                                 "best_val_score": best_b["score"],
                                 "test": test_b},
        }
        if test_a["auc"] == test_a["auc"] and test_b["auc"] == test_b["auc"]:
            out["delta_auc"] = test_a["auc"] - test_b["auc"]
        else:
            out["delta_auc"] = float("nan")
        out["delta_ap"] = test_a["ap"] - test_b["ap"]
        return out

    # -------------------------------------------------------------- deployment
    def _lift_map(self, q):
        """Min-norm lift of an r-dim state back into the host width:
        P_lift = (R Rᵀ + εI)^{-1} R, so deploy-time preagg blocks can feed
        the same fixed context probe (z @ P_lift is host-width)."""
        r_mat = q.r_matrix
        eye = torch.eye(int(r_mat.shape[0]), dtype=torch.float64,
                        device=r_mat.device)
        gram_inv = torch.linalg.inv(
            r_mat @ r_mat.transpose(0, 1) + 1e-6 * eye)
        return (gram_inv @ r_mat).clone()

    @torch.no_grad()
    def deploy_recursive(self, data, max_events=None, verbose=False):
        """Doc 5.8 deployment: leaf-to-root recursion of r-dimensional states
        through the frozen B̂, no host memory, no host aggregate.

        Reports deploy-vs-rich root-state deviation, wall time per event, and
        the state footprint (r vs the host embedding width).
        """
        host = self.adapter.host
        n_events = (int(max_events) if max_events
                    else min(len(data.sources), 500))
        lifts = {tau: self._lift_map(q) for tau, q in self.quotients.items()}
        device = self.device
        zero_time = torch.zeros(1, 1, device=device)
        # Upstream TimeEncode emits (1, 1, time_dim); take the single row.
        source_time = host.time_encoder(zero_time)[0, 0]

        def deploy_state(v, ts):
            """Recursive r-state of node v at time ts (full tree, B̂ at every
            internal node)."""
            def recurse(node, t, layer):
                if layer == 0:
                    x0 = host.node_features[int(node)].to(device).float()
                    return self.quotients[TAU_TEMPLATE.format(0)].project(
                        x0.unsqueeze(0))[0]
                child_tau = TAU_TEMPLATE.format(layer - 1)
                tau = TAU_TEMPLATE.format(layer)
                z_src = recurse(node, t, layer - 1)
                neighbors, eidxs, etimes = \
                    host.neighbor_finder.get_temporal_neighbor(
                        np.asarray([node]), np.asarray([t]),
                        n_neighbors=self.n_neighbors)
                z_neigh, blocks = [], []
                for j in range(self.n_neighbors):
                    nb = int(neighbors[0, j])
                    if nb == 0:
                        blocks.append(None)
                        continue
                    z_j = recurse(nb, float(etimes[0, j]), layer - 1)
                    z_neigh.append(z_j)
                    edge_t = host.time_encoder(torch.tensor(
                        [[float(t) - float(etimes[0, j])]],
                        device=device))[0, 0]
                    edge_f = host.edge_features[
                        int(eidxs[0, j])].to(device)
                    lift = lifts[child_tau]
                    blocks.append((z_j, edge_t, edge_f))
                # Deploy preagg: source/neighbor blocks lifted into host width,
                # edge time/features and mask as in the training packing.
                lift_child = lifts[child_tau]
                src_block = (z_src @ lift_child).float()
                rows = [src_block, source_time]
                for j in range(self.n_neighbors):
                    b = blocks[j]
                    if b is None:
                        rows.extend([torch.zeros_like(src_block),
                                     torch.zeros_like(source_time),
                                     torch.zeros(int(host.n_edge_features),
                                                 device=device)])
                    else:
                        z_j, edge_t, edge_f = b
                        rows.extend([(z_j @ lift_child).float(), edge_t,
                                     edge_f])
                mask = torch.zeros(self.n_neighbors, device=device)
                for j in range(self.n_neighbors):
                    if blocks[j] is None:
                        mask[j] = 1.0
                rows.append(mask)
                flat = torch.cat(rows, dim=-1)
                if flat.shape[-1] != self.probes.preagg_dim:
                    raise ValueError("deploy preagg width mismatch: {} vs {}"
                                     .format(flat.shape[-1],
                                             self.probes.preagg_dim))
                a_deploy = self.probes.probe_a(flat)
                chi = self._chi(z_src, z_neigh, a_deploy)
                op = self.operators[(child_tau, tau)]
                return op.predict(chi.unsqueeze(0))[0]
            return recurse(v, ts, self.n_layers)

        import time
        start = time.time()
        rich_devs = []
        n_done = 0
        for k in range(0, n_events, self.batch_size):
            s, e = k, min(n_events, k + self.batch_size)
            src_emb = self._forward(data.sources[s:e], data.destinations[s:e],
                                    data.timestamps[s:e], data.edge_idxs[s:e])
            z_rich = self.quotients[self.top_tau].project(src_emb)
            for i, row in enumerate(range(s, e)):
                z_dep = deploy_state(int(data.sources[row]),
                                     float(data.timestamps[row]))
                dev = float((self.quotients[self.top_tau].sigma
                             * (z_rich[i] - z_dep)).norm().item())
                rich_devs.append(dev)
                n_done += 1
        elapsed = max(time.time() - start, 1e-9)
        return {
            "events": n_done,
            "deploy_vs_rich_deviation_mean": float(
                sum(rich_devs) / max(len(rich_devs), 1)),
            "deploy_vs_rich_deviation_max": max(rich_devs, default=0.0),
            "wall_seconds": elapsed,
            "events_per_second": n_done / elapsed,
            "state_dim_deployed": self.rank_r,
            "state_dim_host": self.host_dim,
            "state_bytes_per_node_deployed": self.rank_r * 4,
            "state_bytes_per_node_host": self.host_dim * 4,
        }

    # --------------------------------------------------------------------- run
    def run(self, train, val, test) -> Dict:
        n_total = self._num_batches(train)
        steps_a = max(1, int(self.frac_a * n_total))
        steps_b = max(1, int(self.frac_b * n_total))
        steps_c = max(1, int(self.frac_c * n_total))
        a_info = self.calibrate_phase_A(train, steps_a)
        b_info = self.fit_phase_B(train, steps_b, offset=steps_a)
        audit = self.audit_phase_C(train, steps_c, offset=steps_a + steps_b)
        summary = {
            "protocol": "a0",
            "phases": {"a_steps": steps_a, "b_steps": steps_b,
                       "c_steps": steps_c, "d_epochs": self.n_epoch},
            "audit": audit,
            "quotients": {tau: q.snapshot() for tau, q in self.quotients.items()},
            "operators": {"{}->{}".format(*k): op.snapshot()
                          for k, op in self.operators.items()},
        }
        if self.gate_mode == "stop" and audit["gates"].get("failed_gates"):
            summary["status"] = "stopped"
            summary["stop_reason"] = ",".join(audit["gates"]["failed_gates"])
            summary["stop_phase"] = "C"
            return summary
        d_info = self.train_phase_D(train, val, test)
        summary.update(d_info)
        # G4's gate key is auc_delta (the A0-minus-baseline test AUC gap).
        summary["auc_delta"] = d_info.get("delta_auc")
        if self.deploy_events > 0:
            summary["deployment"] = self.deploy_recursive(
                train, max_events=self.deploy_events)
        summary["gates_d"] = evaluate_gates(
            summary, self.gates, mode=self.gate_mode, phase="D")
        if self.gate_mode == "stop" and summary["gates_d"].get("failed_gates"):
            summary["status"] = "stopped"
            summary["stop_reason"] = ",".join(summary["gates_d"]["failed_gates"])
            summary["stop_phase"] = "D"
        else:
            summary["status"] = "complete"
        return summary

    def finalize(self, summary: Dict) -> None:
        """metrics.jsonl / summary.json / _SUCCESS.json (config.json is the
        entry script's job, matching the train_jodie convention)."""
        import json
        from pathlib import Path
        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "metrics.jsonl").open("w") as f:
            for row in self._epoch_rows:
                f.write(json.dumps(row, allow_nan=True) + "\n")
            f.write(json.dumps({"phase": "final", "status": summary["status"],
                                **({"stop_reason": summary["stop_reason"]}
                                   if summary.get("stop_reason") else {})},
                               allow_nan=True) + "\n")
        if self.monitor is not None:
            self.monitor.write_epoch({"phase": "final", "summary": summary})
        with (out / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, allow_nan=True)
        success = {"status": summary["status"]}
        if summary.get("stop_reason"):
            success["stop_reason"] = summary["stop_reason"]
        if summary.get("a0_readout"):
            success["best_epoch_a0"] = summary["a0_readout"]["best_epoch"]
            success["best_epoch_baseline"] = \
                summary["baseline_decoder"]["best_epoch"]
            success["test"] = {"a0_readout": summary["a0_readout"]["test"],
                               "baseline_decoder":
                                   summary["baseline_decoder"]["test"]}
        with (out / "_SUCCESS.json").open("w") as f:
            json.dump(success, f, indent=2, allow_nan=True)
        if self.monitor is not None:
            self.monitor.finalize(summary)
