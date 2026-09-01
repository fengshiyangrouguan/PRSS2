"""Eighth-review acceptance tests (sections A-F of the review).

Contract surface locked here:
  A. rank normalization of the KF loss: J_norm = sum_tau alpha_tau J_tau /
     min(d_tau, m) / sum alpha_tau, denominator FIXED by the configured
     interfaces (never by which taus happen to activate in a batch).
  B. lagged-reference lifecycle: reset clears reference+pending at epoch
     start; close_group builds the NEXT reference without activating it;
     commit_reference enforces an exact one-update lag.
  C. macro-group timing: the representation group (host + compressor)
     updates once per group (gradient divided by the group's batch count,
     epoch-drain for the unfinished group); the head updates every batch.
  D. lambda=0 is a TRUE task-only fast path: no tracing, no cuts, no
     window calls — the compressor still shapes the forward.
  E. F1: the P projection freezes delta_t_scale at construction.
  F. F2: LINK-stage RPBE supervision is refused.
"""

import unittest

import numpy as np
import torch

from rpbe.compressor import RecursiveCompressor
from rpbe.config import RPBConfig
from rpbe.data.jodie import JodieData
from rpbe.loss import (KFLaggedWindow, WeightedWelford, kf_adjoint,
                      kf_vjp_batch, latent_z_adjoint)
from rpbe.maps import FixedMaps
from rpbe.records import JodieCutBuilder, JodieFutureIndex, LINK, NODE_CLASS
from rpbe.state import CompactCutTrace
from rpbe.training.jodie_loop import JodieNodeClassificationLoop

from test_jodie_adapter import install_adapter, make_tiny_tgn
from test_jodie_vendor import REQUIRES_NUMPY_BRIDGE
from test_rpbe_loss import _FakeMaps, _feed_window, make_cut_rows
from test_rpbe_records import candidate, stream


class _FakeMonitor:
    def validate_losses(self, losses, step):
        for k, v in losses.items():
            assert np.isfinite(v), (k, v)

    def validate_kf(self, kf_by_tau, dims, step):
        for tau, j in kf_by_tau.items():
            assert np.isfinite(j), (tau, j)
            assert 0.0 <= j <= dims[tau] + 1e-4, (tau, j, dims[tau])

    def alert(self, severity, code, message, **meta):
        pass


def _tiny_loop(rpbe_cfg, adapter, cut_builder, fixed_maps,
               repr_optimizer, head_optimizer, tgn, decoder, device,
               train_eval_auc=False):
    return JodieNodeClassificationLoop(
        tgn=tgn, decoder=decoder, repr_optimizer=repr_optimizer,
        head_optimizer=head_optimizer, device=device, batch_size=8,
        n_neighbors=4, grad_clip=5.0, monitor=_FakeMonitor(), seed=0,
        finetune_host=True, adapter=adapter, cut_builder=cut_builder,
        fixed_maps=fixed_maps, rpbe_cfg=rpbe_cfg, trace_roots=4,
        trace_mode="evenly_spaced", train_eval_auc=train_eval_auc)


@REQUIRES_NUMPY_BRIDGE
class TestRankNormalization(unittest.TestCase):
    """Section A: per-interface rank normalization of the KF loss."""

    def _cfg(self, lambda_kf=0.01):
        # kf_dims is intersected with the adapter's compression_taus
        # (internal layers 0 < layer < L), so the config must name exactly
        # those interfaces; a 3-layer tree compresses layers 1 and 2.
        return RPBConfig(
            state_dims={"tjo:layer1": 128, "tjo:layer2": 256},
            own_dims={"tjo:layer1": 128, "tjo:layer2": 256},
            m=32, kf_min_abs=4, rpbe_seed=0, delta_t_scale=1.0,
            lambda_kf=lambda_kf, kf_taus=["tjo:layer1", "tjo:layer2"])

    def _loop(self, lambda_kf=0.01):
        tgn, device, stream_data = make_tiny_tgn(n_layers=3)
        sources, destinations, timestamps, edge_idxs, labels = stream_data
        train = JodieData(sources[:24], destinations[:24], timestamps[:24],
                          edge_idxs[:24], labels[:24])
        cfg = self._cfg(lambda_kf)
        from rpbe.hosts.official_tgn import MLP
        decoder = MLP(dim=8, drop=0.1).to(device)
        compressor = RecursiveCompressor(cfg).to(device)
        adapter = install_adapter(tgn)
        adapter.compressor = compressor
        fixed_maps = FixedMaps(cfg).to(device)
        cut_builder = JodieCutBuilder(
            JodieFutureIndex(train), stage=NODE_CLASS, seed=0)
        head_opt = _SpyOpt(decoder.parameters())
        repr_opt = _SpyOpt(
            list(tgn.parameters()) + list(compressor.parameters()))
        loop = _tiny_loop(cfg, adapter, cut_builder, fixed_maps,
                          repr_opt, head_opt, tgn, decoder, device)
        return loop, cfg, train

    def test_tau_coeff_denominator_is_fixed_by_configuration(self):
        # Denominator is the FULL configured interface set, never the set
        # of taus that happen to activate in a batch.  With default alphas
        # (all 1.0): coeff_tau = 1 / (min(d_tau, m) * n_taus).
        loop, cfg, _ = self._loop()
        coeffs = loop._tau_coeff
        self.assertEqual(set(coeffs), {"tjo:layer1", "tjo:layer2"})
        # Normalization identity: sum_tau coeff_tau * min(d_tau, m) == 1.
        total = sum(coeffs[t] * min(cfg.state_dims[t], cfg.m)
                    for t in coeffs)
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertAlmostEqual(coeffs["tjo:layer1"],
                               1.0 / (min(128, 32) * 2), places=6)
        self.assertAlmostEqual(coeffs["tjo:layer2"],
                               1.0 / (min(256, 32) * 2), places=6)

    def test_consume_kf_raw_versus_normalized_arithmetic(self):
        # raw uses the plain alpha weighting (J_hat in "score units"),
        # norm uses the rank-normalized coefficients (J_norm in ~[0, 1]).
        loop, cfg, _ = self._loop()

        class _FakeWindow:
            def consume(self, rows):
                return ({"tjo:layer1": 0.5}, {}, set())

        loop.kf_window = _FakeWindow()
        raw, norm, scores, auxiliary, cold = loop._consume_kf([], 0)
        self.assertAlmostEqual(raw, 0.5 * cfg.alpha("tjo:layer1"))
        coeff = loop._tau_coeff["tjo:layer1"]
        self.assertAlmostEqual(norm, 0.5 * coeff)
        self.assertEqual(cold, set())
        self.assertEqual(float(auxiliary.detach()), 0.0)
        # J_norm stays in [0, 1] for any score in [0, min(d, m)].
        self.assertLessEqual(norm, 1.0)


class _SpyOpt:
    """Counts optimizer steps; records the grad snapshot at each step."""

    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]
        self.steps = 0
        self.grad_snaps = []
        self.param_snaps_at_zero = []
        self.param_snaps_at_step = []

    def _snap(self):
        return [p.detach().clone()
                for p in self.param_groups[0]["params"]]

    def zero_grad(self, set_to_none=True):
        self.param_snaps_at_zero.append(self._snap())
        for p in self.param_groups[0]["params"]:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self):
        self.steps += 1
        # Real SGD update so parameter fingerprints actually move (the
        # spy must behave like an optimizer for the timing tests).
        for p in self.param_groups[0]["params"]:
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-1e-3)
        self.param_snaps_at_step.append(self._snap())
        self.grad_snaps.append([
            None if p.grad is None else float(p.grad.detach().abs().sum())
            for p in self.param_groups[0]["params"]])


class TestVJPEquivalence(unittest.TestCase):
    """Review acceptance 1-4: the moment-adjoint VJP is the exact first
    derivative of the window score on the SAME data, additive across
    batches, and the /W_ref scaling does NOT satisfy the equivalence.

    Pure torch: runs on the local box too.
    """

    def _window(self, n_trees=8, rows_per_cut=4, r=4, m=8):
        # Cluster weights (rows of one cut share z) make D strictly
        # smaller than W: D/W = (n-1)/n, which the /W_ref negative test
        # relies on.
        rows = make_cut_rows(n_trees, r, m, rows_per_cut=rows_per_cut)
        zs = torch.stack([row.z for row in rows])
        ps = torch.stack([_FakeMaps(m).pv(row.context, row.outcome)
                          for row in rows])
        w = torch.tensor([row.weight for row in rows], dtype=torch.float64)
        cut_ids = [(row.tree_id, row.occurrence_id, row.tau)
                   for row in rows]
        wf = WeightedWelford(r, m)
        wf.add(zs, ps, w, cut_ids)
        return wf.result(), rows, cut_ids

    def _vjp_row_grads(self, rows, adjoints, result):
        """Per-row gradients of the raw batch VJP <A, M2_batch>."""
        zs = torch.stack([row.z.clone() for row in rows])
        ps = torch.stack([_FakeMaps(8).pv(row.context, row.outcome)
                          for row in rows])
        w = torch.tensor([row.weight for row in rows], dtype=torch.float64)
        zl = zs.requires_grad_(True)
        raw = kf_vjp_batch(zl, ps, w, result["mu_z"], result["mu_p"],
                           adjoints)
        raw.backward()
        return zl.grad.detach()

    def _merge_by_cut(self, rows, row_grads):
        merged = {}
        for row, g in zip(rows, row_grads):
            cid = (row.tree_id, row.occurrence_id, row.tau)
            merged[cid] = merged.get(cid, 0.0) + g
        return [merged[cid] for cid in sorted(merged)]

    def test_raw_surrogate_gradient_cosine_matches_direct(self):
        # Acceptance 1/2: on the SAME window data the raw VJP gradient
        # and the direct score gradient are collinear (cosine > 0.99999),
        # so the training direction -grad(-surrogate) points the same way
        # as grad(J).
        result, rows, cut_ids = self._window()
        _, adjoints, _ = kf_adjoint(result, eps=1e-4)
        g_vjp = self._merge_by_cut(rows, self._vjp_row_grads(
            rows, adjoints, result))
        z_rows = torch.stack([row.z for row in rows])
        p_rows = torch.stack([_FakeMaps(8).pv(row.context, row.outcome)
                              for row in rows])
        w = torch.tensor([row.weight for row in rows], dtype=torch.float64)
        _, g_by_cut, diag = latent_z_adjoint(
            z_rows, p_rows, w, cut_ids, result["mu_z"], result["mu_p"],
            result["D"], 1e-4)
        self.assertIsNone(diag["failed"])
        g_direct = [g_by_cut[cid] for cid in sorted(g_by_cut)]
        gv = torch.stack(g_vjp).flatten().double()
        gd = torch.stack(g_direct).flatten().double()
        cos = float((gv * gd).sum()
                    / (gv.norm() * gd.norm()).clamp(min=1e-30))
        self.assertGreater(cos, 0.99999)

    def test_batch_split_vjp_sums_to_direct_gradient(self):
        # Acceptance 3: the raw VJP of the whole window equals the sum of
        # per-batch VJPs, and that sum IS the exact direct gradient of
        # the window score.  (The score is scale-invariant —
        # _score_from_covs normalizes by mean(diag) — so the 1/D of the
        # covariance convention cancels exactly and NO rescaling is the
        # correct recovery.)
        result, rows, cut_ids = self._window()
        _, adjoints, _ = kf_adjoint(result, eps=1e-4)
        total = None
        for start in range(0, len(rows), 5):
            chunk = rows[start:start + 5]
            g = self._vjp_row_grads(chunk, adjoints, result)
            total = g if total is None else torch.cat([total, g])
        g_raw_sum = total  # row order preserved by the loop slicing
        z_rows = torch.stack([row.z for row in rows])
        p_rows = torch.stack([_FakeMaps(8).pv(row.context, row.outcome)
                              for row in rows])
        w = torch.tensor([row.weight for row in rows], dtype=torch.float64)
        _, g_by_cut, diag = latent_z_adjoint(
            z_rows, p_rows, w, cut_ids, result["mu_z"], result["mu_p"],
            result["D"], 1e-4)
        self.assertIsNone(diag["failed"])
        g_direct = self._merge_by_cut(rows, g_raw_sum)
        ref = [g_by_cut[cid] for cid in sorted(g_by_cut)]
        for a, b in zip(g_direct, ref):
            err = float((a.double() - b.double()).norm()
                        / b.double().norm().clamp(min=1e-30))
            self.assertLess(err, 1e-5)

    def test_kfold_detached_reference_adjoint_scales_inversely(self):
        """The review's decisive test: J(cM) = J(M) (degree-0 homogeneous)
        implies grad_M J(cM) = (1/c) grad_M J(M).

        Build a batch B, then a reference R of k DETACHED copies of B
        with fresh tree/cut ids and NO weight renormalization (so
        M_R = k M_B, W_R = k W_B).  Then:

            A_R = A_B / k
            g_raw = grad_z <A_R, M_B(z)> = g_direct / k
            (W_R/W_B) g_raw = k g_raw = g_direct

        The copies MUST be detached (production references are); using
        one repeated graph-connected tensor would let autograd sum the k
        paths and falsely show raw == direct.  The weights must NOT be
        renormalized to total 1, or the scale information is lost.
        """
        k = 3
        rows_b = make_cut_rows(8, 4, 8, rows_per_cut=4)
        z_b = torch.stack([row.z.detach().clone() for row in rows_b])
        p_b = torch.stack([_FakeMaps(8).pv(row.context, row.outcome)
                           for row in rows_b])
        w_b = torch.tensor([row.weight for row in rows_b],
                           dtype=torch.float64)
        cut_b = [(row.tree_id, row.occurrence_id, row.tau)
                 for row in rows_b]

        wf_b = WeightedWelford(4, 8)
        wf_b.add(z_b, p_b, w_b, cut_b)
        res_b = wf_b.result()
        _, a_batch, _ = kf_adjoint(res_b, eps=1e-4)

        # Reference: k detached copies, fresh ids, weights unchanged.
        z_r = z_b.repeat(k, 1)
        p_r = p_b.repeat(k, 1)
        w_r = w_b.repeat(k)
        cut_r = [("c", j * 1000 + i) for j in range(k)
                 for i in range(len(cut_b))]
        wf_r = WeightedWelford(4, 8)
        wf_r.add(z_r, p_r, w_r, cut_r)
        res_r = wf_r.result()
        self.assertAlmostEqual(res_r["W"], k * res_b["W"], places=9)
        _, a_ref, _ = kf_adjoint(res_r, eps=1e-4)

        # A_R = A_B / k (zero-degree homogeneity of the gradient).
        for key in ("M2_zz", "M2_zp", "M2_pp"):
            err = float((a_ref[key] - a_batch[key] / k).norm()
                        / a_batch[key].norm().clamp(min=1e-30))
            self.assertLess(err, 1e-6, key)

        # g_raw = grad_z <A_R, M_B(z)> = g_direct / k.
        zl = z_b.clone().requires_grad_(True)
        raw = kf_vjp_batch(zl, p_b, w_b, res_r["mu_z"], res_r["mu_p"],
                           a_ref)
        raw.backward()
        g_raw = zl.grad.detach()
        zl2 = z_b.clone().requires_grad_(True)
        direct = kf_vjp_batch(zl2, p_b, w_b, res_b["mu_z"], res_b["mu_p"],
                              a_batch)
        direct.backward()
        g_direct = zl2.grad.detach()
        err_raw = float((g_raw - g_direct / k).norm()
                        / g_direct.norm().clamp(min=1e-30))
        self.assertLess(err_raw, 1e-6)

        # (W_R / W_B) * g_raw = k * g_raw = g_direct.
        err_scaled = float((k * g_raw - g_direct).norm()
                           / g_direct.norm().clamp(min=1e-30))
        self.assertLess(err_scaled, 1e-6)

    def test_group_K_cancellation_preserves_unequal_batch_weights(self):
        # The LOSS_DIAGNOSIS P0 regression: production multiplies each
        # zero-valued auxiliary by the actual group batch count K and
        # divides ALL accumulated representation gradients by K at group
        # close.  Task gradients get the mean; the KF raw VJP sum must
        # come through UNSHRUNK — exact even when batches carry unequal
        # valid-cut weights (W_ref/W_batch would reweight them).
        result, rows, cut_ids = self._window()
        _, adjoints, _ = kf_adjoint(result, eps=1e-4)
        # Split the window into 3 batches with UNEQUAL weight sums.
        chunks = []
        total_rows = len(rows)
        ends = [9, 23, total_rows]  # 9, 14, 9 rows -> unequal weights
        start = 0
        for end in ends:
            chunks.append(rows[start:end])
            start = end
        K = len(chunks)
        production_sum = None
        for chunk in chunks:
            g = self._vjp_row_grads(chunk, adjoints, result) * float(K)
            production_sum = g if production_sum is None \
                else torch.cat([production_sum, g])
        production_sum = production_sum / float(K)  # common grad /= K
        z_rows = torch.stack([row.z for row in rows])
        p_rows = torch.stack([_FakeMaps(8).pv(row.context, row.outcome)
                              for row in rows])
        w = torch.tensor([row.weight for row in rows], dtype=torch.float64)
        _, g_by_cut, diag = latent_z_adjoint(
            z_rows, p_rows, w, cut_ids, result["mu_z"], result["mu_p"],
            result["D"], 1e-4)
        self.assertIsNone(diag["failed"])
        merged = self._merge_by_cut(rows, production_sum)
        ref = [g_by_cut[cid] for cid in sorted(g_by_cut)]
        for a, b in zip(merged, ref):
            err = float((a.double() - b.double()).norm()
                        / b.double().norm().clamp(min=1e-30))
            self.assertLess(err, 1e-5)
        # The same sum WITHOUT the K cancellation is shrunk by exactly K.
        raw_sum = torch.cat(
            [self._vjp_row_grads(c, adjoints, result)
             for c in chunks]) / float(K)
        shrunk = self._merge_by_cut(rows, raw_sum)
        for a, b in zip(shrunk, ref):
            err = float((a.double() - b.double()).norm()
                        / b.double().norm().clamp(min=1e-30))
            self.assertGreater(err, (1.0 - 1.0 / K) * 0.5)

    def test_divide_by_wref_breaks_equivalence(self):
        # Acceptance 4 (updated to scheme B): the raw VJP is the exact
        # direct gradient on the same window data — that is the
        # production surrogate.  BOTH rescaling conventions break the
        # equivalence: /W_ref shrinks it by 1/W_ref; W_ref/W_batch blows
        # it up by W_ref/W_batch (here = 1 on the same window, so use the
        # k-fold perspective: the group sum of raw VJPs already carries
        # the window multiplicity, any extra factor double-counts it).
        result, rows, _ = self._window()
        _, adjoints, _ = kf_adjoint(result, eps=1e-4)
        g_raw = self._vjp_row_grads(rows, adjoints, result)
        W_ref = result["W"]
        g_direct_flat = torch.stack(
            self._merge_by_cut(rows, g_raw)).flatten().double()
        g_divw = torch.stack(
            self._merge_by_cut(rows, g_raw / W_ref)).flatten().double()
        norm = g_direct_flat.norm().clamp(min=1e-30)
        # /W_ref convention: off by the 1/W_ref constant (7/8 here).
        err_divw = float((g_divw - g_direct_flat).norm() / norm)
        self.assertGreater(err_divw, (1.0 - 1.0 / W_ref) * 0.5)


class TestReferenceLifecycle(unittest.TestCase):
    """Section B: the lagged reference window lifecycle.

    Pure torch (no numpy bridge needed): runs on the local box too.
    """

    def _window(self):
        return KFLaggedWindow(
            {"t": 4}, min_ratio=1.0, min_abs=4, fixed_maps=_FakeMaps(8))

    def test_commit_reference_enforces_exact_one_update_lag(self):
        # Review B.8: a candidate reference that does NOT lag exactly one
        # representation update is discarded AND the old active reference
        # is dropped — an age>=2 reference must never keep being used, so
        # the next group goes cold for that tau.
        window = self._window()
        _feed_window(window, make_cut_rows(4, 4, 8, offset=0),
                     param_version=0)
        self.assertEqual(window._reference["t"]["param_version"], 0)
        # Next group claims version 2 — a refresh was skipped in between.
        window.begin_group(2, 0)
        window.consume(make_cut_rows(4, 4, 8, offset=50))
        window.close_group()
        stale = window.commit_reference(current_version=3)
        self.assertEqual(stale, ["t"])
        self.assertEqual(window.stale_drops, 1)
        # The old reference is GONE (cold restart); the candidate is not
        # queued either.
        self.assertIsNone(window.reference_score("t"))
        self.assertEqual(window._next_reference, {})
        # The next group that reaches threshold builds a fresh reference
        # (first activation skips the version check).
        window.begin_group(4, 0)
        window.consume(make_cut_rows(4, 4, 8, offset=100))
        window.close_group()
        self.assertEqual(window.commit_reference(current_version=5), [])
        self.assertIsNotNone(window.reference_score("t"))
        self.assertEqual(window._reference["t"]["param_version"], 4)
        self.assertEqual(window.reference_age("t"), 1)

    def test_epoch_reset_clears_reference_and_pending(self):
        window = self._window()
        _feed_window(window, make_cut_rows(4, 4, 8, offset=0),
                     param_version=0)
        self.assertIsNotNone(window.reference_score("t"))
        window.reset(clear_reference=True)
        self.assertIsNone(window.reference_score("t"))
        self.assertEqual(window.pending_tree_count("t"), 0)
        # After the epoch reset, version numbering restarts at 0.
        _feed_window(window, make_cut_rows(4, 4, 8, offset=200),
                     param_version=0)
        self.assertIsNotNone(window.reference_score("t"))

    def test_window_score_bounded_by_rank(self):
        # Acceptance 10: the ridge-regularized Ky Fan score stays within
        # [0, min(d_tau, m)], so the rank-normalized objective J_norm
        # stays within [0, 1].
        window = self._window()
        _feed_window(window, make_cut_rows(24, 4, 8, offset=0),
                     param_version=0)
        j = window.reference_score("t")
        self.assertIsNotNone(j)
        self.assertGreaterEqual(j, 0.0)
        self.assertLessEqual(j, 4.0 * (1.0 + 1e-6))
        self.assertLessEqual(j / 4.0, 1.0)

    def test_close_group_builds_next_without_activating(self):
        window = self._window()
        window.begin_group(0, 0)
        window.consume(make_cut_rows(4, 4, 8, offset=0))
        self.assertIsNone(window.reference_score("t"))
        diagnostics, refreshed = window.close_group()
        self.assertEqual(refreshed, ["t"])
        # close_group only builds the NEXT reference; the active one is
        # untouched until commit_reference.
        self.assertIsNone(window.reference_score("t"))
        window.commit_reference()
        self.assertIsNotNone(window.reference_score("t"))
        self.assertEqual(window._reference["t"]["param_version"], 0)

    def test_below_threshold_group_close_discards_pending(self):
        # Eighth review C: a reference window must live inside ONE macro
        # group (the representation parameters are frozen within a group),
        # so a below-threshold group discards its partial window instead
        # of carrying rows across parameter versions.
        window = self._window()
        window.begin_group(0, 0)
        window.consume(make_cut_rows(2, 4, 8, rows_per_cut=2))
        self.assertEqual(window.pending_tree_count("t"), 2)
        diagnostics, refreshed = window.close_group()
        self.assertEqual(refreshed, [])
        self.assertTrue(diagnostics["t"]["below_threshold"])
        self.assertEqual(diagnostics["t"]["dropped_trees"], 2)
        self.assertIsNone(window.reference_score("t"))
        # The discarded rows do NOT survive into the next group.
        window.begin_group(1, 0)
        window.consume(make_cut_rows(2, 4, 8, rows_per_cut=2, offset=50))
        self.assertEqual(window.pending_tree_count("t"), 2)
        # A group that reaches the threshold refreshes and activates.
        window.consume(make_cut_rows(2, 4, 8, rows_per_cut=2, offset=100))
        self.assertEqual(window.pending_tree_count("t"), 4)
        diagnostics, refreshed = window.close_group()
        self.assertEqual(refreshed, ["t"])
        self.assertIsNone(window.reference_score("t"))  # not active yet
        window.commit_reference(current_version=2)
        self.assertIsNotNone(window.reference_score("t"))
        self.assertEqual(window.reference_age("t"), 1)
        # A refresh resets the pending for the next group.
        self.assertEqual(window.pending_tree_count("t"), 0)


@REQUIRES_NUMPY_BRIDGE
class TestMacroGroupTiming(unittest.TestCase):
    """Section C: representation once per macro-group, head every batch."""

    def setUp(self):
        tgn, device, stream_data = make_tiny_tgn()
        sources, destinations, timestamps, edge_idxs, labels = stream_data
        self.tgn = tgn
        self.device = device
        self.train = JodieData(sources[:24], destinations[:24],
                               timestamps[:24], edge_idxs[:24],
                               labels[:24])  # 3 batches at bs=8

    def _task_only_loop(self):
        # component_on=True, kf_on=False (lambda=0 fast path): the fixed
        # cadence is ceil(kf_min_abs / trace_roots) = ceil(8 / 4) = 2
        # batches per group.
        cfg = RPBConfig(
            state_dims={"tjo:layer0": 8, "tjo:layer1": 8, "tjo:layer2": 8},
            own_dims={"tjo:layer0": 8, "tjo:layer1": 8, "tjo:layer2": 8},
            m=64, kf_min_abs=8, rpbe_seed=0, delta_t_scale=1.0,
            lambda_kf=0.0)
        from rpbe.hosts.official_tgn import MLP
        decoder = MLP(dim=8, drop=0.1).to(self.device)
        compressor = RecursiveCompressor(cfg).to(self.device)
        adapter = install_adapter(self.tgn)
        adapter.compressor = compressor
        fixed_maps = FixedMaps(cfg).to(self.device)
        repr_opt = _SpyOpt(
            list(self.tgn.parameters()) + list(compressor.parameters()))
        head_opt = _SpyOpt(decoder.parameters())
        loop = _tiny_loop(cfg, adapter, None, fixed_maps,
                          repr_opt, head_opt, self.tgn, decoder, self.device)
        return loop, repr_opt, head_opt

    def test_head_steps_every_batch_repr_once_per_group(self):
        loop, repr_opt, head_opt = self._task_only_loop()
        loop.train_epoch(0, 0, self.train)  # 3 batches, group length 2
        # Head: one step per batch.  Representation: group close at batch 2
        # plus the epoch drain of the unfinished group.
        self.assertEqual(head_opt.steps, 3)
        self.assertEqual(repr_opt.steps, 2)

    def test_repr_grad_divided_by_group_batch_count(self):
        loop, repr_opt, head_opt = self._task_only_loop()
        divs = []
        orig = torch.Tensor.div_
        def spy_div(self, other, *args, **kwargs):
            if isinstance(other, (float, int)):
                divs.append(float(other))
            return orig(self, other, *args, **kwargs)
        torch.Tensor.div_ = spy_div
        try:
            loop.train_epoch(0, 0, self.train)
        finally:
            torch.Tensor.div_ = orig
        # Group of 2 batches divides by 2; the epoch drain divides by 1.
        self.assertIn(2.0, divs)
        self.assertIn(1.0, divs)
        # Both representation steps actually carried gradients.
        self.assertTrue(any(snap is not None for snap in repr_opt.grad_snaps))

    def test_repr_fingerprint_unchanged_within_group(self):
        # Acceptance 5: the representation parameters are untouched
        # between the group's zero_grad and its single step — group 2's
        # start state equals the state right after step 1, and the step
        # really moved the parameters.
        loop, repr_opt, head_opt = self._task_only_loop()
        loop.train_epoch(0, 0, self.train)  # 3 batches, group length 2
        snaps_zero = repr_opt.param_snaps_at_zero
        snaps_step = repr_opt.param_snaps_at_step
        # group1 start, group2 start, and the post-drain cleanup zero_grad
        self.assertEqual(len(snaps_zero), 3)
        self.assertEqual(len(snaps_step), 2)   # group close + epoch drain
        s0, s1 = snaps_zero[:2]
        step0 = snaps_step[0]
        # Nothing touched the representation parameters between the two
        # groups (the batch-3 gradient accumulation does not move params).
        for a, b in zip(s1, step0):
            self.assertTrue(torch.equal(a, b))
        # And the representation step really moved at least one of the
        # parameters that carried a gradient (params without a gradient
        # legitimately stay put).
        moved = sum(not torch.equal(a, b) for a, b in zip(s0, step0))
        self.assertGreater(moved, 0)


@REQUIRES_NUMPY_BRIDGE
class TestLambdaZeroFastPath(unittest.TestCase):
    """Section D: lambda=0 must be a true task-only fast path."""

    def test_lambda_zero_skips_trace_build_and_window(self):
        tgn, device, stream_data = make_tiny_tgn()
        sources, destinations, timestamps, edge_idxs, labels = stream_data
        train = JodieData(sources[:24], destinations[:24], timestamps[:24],
                          edge_idxs[:24], labels[:24])
        cfg = RPBConfig(
            state_dims={"tjo:layer0": 8, "tjo:layer1": 8, "tjo:layer2": 8},
            own_dims={"tjo:layer0": 8, "tjo:layer1": 8, "tjo:layer2": 8},
            m=64, kf_min_abs=8, rpbe_seed=0, delta_t_scale=1.0,
            lambda_kf=0.0)
        from rpbe.hosts.official_tgn import MLP
        decoder = MLP(dim=8, drop=0.1).to(device)

        class _Calls:
            trace = 0
            clear = 0
            build = 0
            begin = 0
            consume = 0
            close = 0
            commit = 0

        calls = _Calls()
        real_adapter = install_adapter(tgn)
        real_adapter.compressor = RecursiveCompressor(cfg).to(device)

        class _SpyAdapter:
            def set_trace_source_rows(self, rows):
                calls.trace += 1

            def clear_trace(self):
                calls.clear += 1

        class _SpyBuilder:
            def build(self, trace, batch_seed=0, stats=None):
                calls.build += 1
                return []

        class _SpyWindow:
            def begin_group(self, param_version, epoch):
                calls.begin += 1

            def consume(self, rows):
                calls.consume += 1
                return {}, {}, set()

            def close_group(self):
                calls.close += 1
                return {}, []

            def commit_reference(self):
                calls.commit += 1

            def all_pending_ready(self):
                return False

            def reset(self, clear_reference=True):
                pass

        fixed_maps = FixedMaps(cfg).to(device)
        repr_opt = _SpyOpt(list(tgn.parameters()))
        head_opt = _SpyOpt(decoder.parameters())
        loop = _tiny_loop(cfg, _SpyAdapter(), _SpyBuilder(), fixed_maps,
                          repr_opt, head_opt, tgn, decoder, device)
        loop.kf_window = _SpyWindow()
        row = loop.train_epoch(0, 0, train)
        # No tracing, no cut building, no window activity at all; the
        # adapter trace is cleared every batch instead.
        self.assertEqual(calls.trace, 0)
        self.assertEqual(calls.build, 0)
        self.assertEqual(calls.begin, 0)
        self.assertEqual(calls.consume, 0)
        self.assertEqual(calls.close, 0)
        self.assertEqual(calls.commit, 0)
        self.assertEqual(calls.clear, 3)  # one clear_trace per batch
        self.assertIsNone(row["kf"])
        self.assertEqual(row["train_kf_score"], 0.0)
        self.assertEqual(row["train_kf_normalized"], 0.0)

    def test_vanilla_no_component_runs_clean(self):
        # rpbe_cfg=None (pure vanilla): component off, empty representation
        # group, no window — the epoch must still run cleanly.
        tgn, device, stream_data = make_tiny_tgn()
        sources, destinations, timestamps, edge_idxs, labels = stream_data
        train = JodieData(sources[:24], destinations[:24], timestamps[:24],
                          edge_idxs[:24], labels[:24])
        from rpbe.hosts.official_tgn import MLP
        decoder = MLP(dim=8, drop=0.1).to(device)
        repr_opt = _SpyOpt([])  # vanilla + frozen host: empty repr group
        head_opt = _SpyOpt(decoder.parameters())
        loop = JodieNodeClassificationLoop(
            tgn=tgn, decoder=decoder, repr_optimizer=repr_opt,
            head_optimizer=head_opt, device=device, batch_size=8,
            n_neighbors=4, grad_clip=5.0, monitor=_FakeMonitor(), seed=0,
            finetune_host=False, trace_mode="evenly_spaced")
        row = loop.train_epoch(0, 0, train)
        self.assertIsNone(row["kf"])
        self.assertTrue(np.isfinite(row["train_task_loss"]))
        # Vanilla: the representation group never steps.
        self.assertEqual(repr_opt.steps, 0)
        self.assertEqual(head_opt.steps, 3)


class TestFrozenScaleAndLinkRefusal(unittest.TestCase):
    """Sections E/F: F1 frozen delta_t_scale, F2 LINK refusal.

    Pure torch / python fixtures: runs on the local box too.
    """

    def test_pv_batch_frozen_scale_consistent_with_pv(self):
        # Acceptance 13 (batch path): pv_batch uses the frozen
        # construction-time delta_t_scale, and stays consistent with the
        # per-row pv after external config mutation.
        cfg = RPBConfig(state_dims={"t": 4}, own_dims={"t": 4}, m=8,
                        delta_t_scale=10.0)
        maps = FixedMaps(cfg)
        contexts = [{"horizon": 1, "delta_t": 500.0 * (k + 1),
                     "counterpart": k, "role": 0, "query_type": 1,
                     "endpoint_role": 1, "path": []} for k in range(5)]
        outcomes = [float(k) for k in range(5)]
        out1 = maps.pv_batch(contexts, outcomes)
        for k, ctx in enumerate(contexts):
            self.assertTrue(torch.equal(out1[k], maps.pv(ctx, outcomes[k])))
        cfg.delta_t_scale = 999.0  # post-construction mutation
        out2 = maps.pv_batch(contexts, outcomes)
        self.assertTrue(torch.equal(out1, out2))
        for k, ctx in enumerate(contexts):
            self.assertTrue(torch.equal(out2[k], maps.pv(ctx, outcomes[k])))

    def test_pv_batch_frozen_delta_t_scale(self):
        # F1: the P projection freezes delta_t_scale at construction;
        # later config mutations must not change its output.
        cfg = RPBConfig(state_dims={"t": 4}, own_dims={"t": 4}, m=8,
                        delta_t_scale=10.0)
        maps = FixedMaps(cfg)
        context = {"horizon": 1, "delta_t": 500.0, "counterpart": 2,
                   "role": 0, "query_type": 1, "endpoint_role": 1,
                   "path": []}
        out1 = maps.pv(context, 1.0)
        cfg.delta_t_scale = 999.0  # post-construction mutation
        out2 = maps.pv(context, 1.0)
        self.assertEqual(float(out1.sum()), float(out2.sum()))
        self.assertTrue(torch.equal(out1, out2))

    def test_link_stage_refused_not_implemented(self):
        # F2: node-classification labels must never masquerade as link
        # outcomes, so the LINK stage refuses RPBE supervision.
        trace = CompactCutTrace(root_rows=[0], cuts=[candidate()])
        builder = JodieCutBuilder(JodieFutureIndex(stream()), stage=LINK,
                                  seed=0)
        with self.assertRaises(NotImplementedError) as cm:
            builder.build(trace)
        self.assertIn("must not masquerade", str(cm.exception))
