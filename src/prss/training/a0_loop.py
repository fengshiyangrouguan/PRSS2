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
                           path_gain_report)
from prss.a0.operators import OperatorRidge, chi_sigma, chi_width
from prss.a0.probes import A0Probes, propagate_root_labels, stack_by_tau
from prss.a0.quotient import A0Quotient
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
                 lr, n_epoch, patience, drop_out, selection_metric="auc"):
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
        s = chi_width(self.rank_r, d_context)
        for l in range(1, self.n_layers + 1):
            key = (TAU_TEMPLATE.format(l - 1), TAU_TEMPLATE.format(l))
            self.operators[key] = OperatorRidge(
                child_tau=key[0], parent_tau=key[1], s=s, r=self.rank_r)
        self.top_tau = TAU_TEMPLATE.format(self.n_layers)
        self._epoch_rows: List[Dict] = []

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
        for k in range(steps):
            s, e = k * self.batch_size, min(len(train.sources),
                                            (k + 1) * self.batch_size)
            collected = self._collect_trace(train, k, train.labels[s:e])
            if collected is None:
                continue
            _, stacks, _ = collected
            for tau, st in stacks.items():
                self.quotients[tau].accumulate(st["X"], st["U"])
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
                z_neigh_mean = (torch.stack(z_neigh).mean(dim=0)
                                if z_neigh else torch.zeros_like(z_source))
                a_parent = self.probes.probe_a(occ.local_features.detach())
                phi = chi_sigma(z_source, z_neigh_mean, a_parent)
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
                z_neigh_mean = (torch.stack(z_neigh).mean(dim=0)
                                if z_neigh else torch.zeros_like(z_source))
                a_parent = self.probes.probe_a(occ.local_features.detach())
                phi = chi_sigma(z_source, z_neigh_mean, a_parent)
                key = self._operator_key(occ)
                op = self.operators[key]
                z_rec = op.predict(phi.unsqueeze(0))[0]
                z_rich = z_by_oid[occ.occurrence_id]
                sigma = self.quotients[occ.tau].sigma
                diff = sigma * (z_rich - z_rec)
                closure_sums[key] += float((diff * diff).sum().item())
                closure_n[key] += 1
            self.adapter.clear_trace()

        audit = {"ess": total_rows}
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
            "{}->{}".format(*key): op.condition_number
            for key, op in self.operators.items()}
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
