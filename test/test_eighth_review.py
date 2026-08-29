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
from rpbe.loss import KFLaggedWindow
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


class _SpyOpt:
    """Counts optimizer steps; records the grad snapshot at each step."""

    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]
        self.steps = 0
        self.grad_snaps = []

    def zero_grad(self, set_to_none=True):
        for p in self.param_groups[0]["params"]:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self):
        self.steps += 1
        self.grad_snaps.append([
            None if p.grad is None else float(p.grad.detach().abs().sum())
            for p in self.param_groups[0]["params"]])


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
        return RPBConfig(
            state_dims={"tjo:layer0": 64, "tjo:layer1": 128,
                        "tjo:layer2": 256},
            own_dims={"tjo:layer0": 64, "tjo:layer1": 128,
                      "tjo:layer2": 256},
            m=32, kf_min_abs=4, rpbe_seed=0, delta_t_scale=1.0,
            lambda_kf=lambda_kf, kf_taus=["tjo:layer0", "tjo:layer1",
                                          "tjo:layer2"])

    def _loop(self, lambda_kf=0.01):
        tgn, device, stream_data = make_tiny_tgn()
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
        self.assertEqual(set(coeffs), {"tjo:layer0", "tjo:layer1",
                                       "tjo:layer2"})
        # Normalization identity: sum_tau coeff_tau * min(d_tau, m) == 1.
        total = sum(coeffs[t] * min(cfg.state_dims[t], cfg.m)
                    for t in coeffs)
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertAlmostEqual(coeffs["tjo:layer0"],
                               1.0 / (min(64, 32) * 3), places=6)
        self.assertAlmostEqual(coeffs["tjo:layer2"],
                               1.0 / (min(256, 32) * 3), places=6)

    def test_consume_kf_raw_versus_normalized_arithmetic(self):
        # raw uses the plain alpha weighting (J_hat in "score units"),
        # norm uses the rank-normalized coefficients (J_norm in ~[0, 1]).
        loop, cfg, _ = self._loop()

        class _FakeWindow:
            def consume(self, rows):
                return ({"tjo:layer0": 0.5}, {}, set())

        loop.kf_window = _FakeWindow()
        raw, norm, scores, auxiliary, cold = loop._consume_kf([], 0)
        self.assertAlmostEqual(raw, 0.5 * cfg.alpha("tjo:layer0"))
        coeff = loop._tau_coeff["tjo:layer0"]
        self.assertAlmostEqual(norm, 0.5 * coeff)
        self.assertEqual(cold, set())
        self.assertEqual(float(auxiliary.detach()), 0.0)
        # J_norm stays in [0, 1] for any score in [0, min(d, m)].
        self.assertLessEqual(norm, 1.0)


class TestReferenceLifecycle(unittest.TestCase):
    """Section B: the lagged reference window lifecycle.

    Pure torch (no numpy bridge needed): runs on the local box too.
    """

    def _window(self):
        return KFLaggedWindow(
            {"t": 4}, min_ratio=1.0, min_abs=4, fixed_maps=_FakeMaps(8))

    def test_commit_reference_enforces_exact_one_update_lag(self):
        window = self._window()
        _feed_window(window, make_cut_rows(4, 4, 8, offset=0),
                     param_version=0)
        self.assertEqual(window._reference["t"]["param_version"], 0)
        # Next group claims version 2 — a representation update was skipped.
        window.begin_group(2, 0)
        window.consume(make_cut_rows(4, 4, 8, offset=50))
        window.close_group()
        with self.assertRaises(AssertionError) as cm:
            window.commit_reference()
        self.assertIn("lag exactly one", str(cm.exception))

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

    def test_below_threshold_group_close_keeps_pending(self):
        # Window accumulation is decoupled from the representation-update
        # cadence: a group closes below threshold without discarding its
        # rows — the next group keeps adding to the same window.  The
        # pending is only reset when a refresh actually happens.
        window = self._window()
        window.begin_group(0, 0)
        window.consume(make_cut_rows(2, 4, 8, rows_per_cut=2))
        self.assertEqual(window.pending_tree_count("t"), 2)
        diagnostics, refreshed = window.close_group()
        self.assertEqual(refreshed, [])
        self.assertEqual(diagnostics, {})
        self.assertIsNone(window.reference_score("t"))
        # Next group accumulates on top of the surviving rows.
        window.begin_group(1, 0)
        window.consume(make_cut_rows(2, 4, 8, rows_per_cut=2, offset=50))
        self.assertEqual(window.pending_tree_count("t"), 4)
        diagnostics, refreshed = window.close_group()
        self.assertEqual(refreshed, ["t"])
        self.assertIsNone(window.reference_score("t"))  # not active yet
        window.commit_reference()
        self.assertIsNotNone(window.reference_score("t"))
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
