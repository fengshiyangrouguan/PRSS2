"""Unit tests for the diagnose_loss.py pure logic (P1 audit).

The full audit needs a GPU checkpoint; these lock the arithmetic and the
group-bookkeeping pieces that previously crashed on the cloud
(None-grad params, mixed-dim tensor concatenation, no-aligned-cuts
comparisons).  Pure torch/numpy: runs on the local box.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

import scripts.diagnose_loss as dl


def _fake_rows(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        rows.append(SimpleNamespace(
            horizon=1 + (i % 2),
            outcome=float(i % 5 == 0),
            context={"role": i % 2, "delta_t": float(i), "counterpart": i,
                     "query_type": 0, "path": []},
            weight=0.5 if i % 3 else 1.0,
            cut_id=("t", i, "layer1"),
            tree_id=i // 2,
            tau="layer1"))
    return rows


class TestGradientVec(unittest.TestCase):
    def test_param_grad_vec_handles_none_grads(self):
        a = torch.nn.Parameter(torch.randn(3, 4))
        b = torch.nn.Parameter(torch.randn(2))
        a.grad = torch.randn(3, 4)
        b.grad = None
        vec = dl._param_grad_vec([a, b])
        self.assertEqual(vec.numel(), 3 * 4 + 2)
        # a-part equals its gradient, b-part is zeros.
        self.assertTrue(torch.allclose(
            vec[:12].reshape(3, 4), a.grad.detach().double()))
        self.assertTrue(torch.all(vec[12:] == 0.0))

    def test_param_grad_vec_divides(self):
        a = torch.nn.Parameter(torch.ones(4))
        a.grad = torch.full((4,), 2.0)
        vec = dl._param_grad_vec([a], divide_by=2.0)
        self.assertTrue(torch.all(vec == 1.0))


class TestCosAndComparisons(unittest.TestCase):
    def test_cos_basic(self):
        u = torch.tensor([1.0, 0.0])
        v = torch.tensor([0.0, 1.0])
        self.assertAlmostEqual(dl._cos(u, v), 0.0, places=6)
        self.assertAlmostEqual(dl._cos(u, u), 1.0, places=6)
        self.assertAlmostEqual(dl._cos(u, -u), -1.0, places=6)

    def test_gradient_comparison_identical(self):
        g = {"a": torch.randn(5)}
        cmp = dl._gradient_comparison(g, g)
        self.assertAlmostEqual(cmp["cosine"], 1.0, places=6)
        self.assertAlmostEqual(cmp["norm_ratio"], 1.0, places=6)
        self.assertAlmostEqual(cmp["relative_error"], 0.0, places=6)

    def test_gradient_comparison_no_aligned_cuts(self):
        cmp = dl._gradient_comparison({"a": torch.randn(3)},
                                      {"b": torch.randn(3)})
        self.assertEqual(cmp, {"failed": "no_aligned_cuts"})

    def test_flatten_aligned_intersection(self):
        left = {"a": torch.randn(3), "b": torch.randn(2)}
        right = {"b": torch.randn(2), "c": torch.randn(4)}
        lf, rf = dl._flatten_aligned(left, right)
        self.assertEqual(lf.numel(), 2)
        self.assertTrue(torch.allclose(lf, left["b"].double()))

    def test_merge_cut_gradients_sums(self):
        merged = dl._merge_cut_gradients(
            [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
            ["a", "a"])
        self.assertTrue(torch.allclose(merged["a"],
                                       torch.tensor([4.0, 6.0])))


class TestPairwiseSeries(unittest.TestCase):
    def _pairs(self):
        u = torch.tensor([1.0, 0.0])
        v = torch.tensor([0.0, 1.0])
        return [(u, v), (u, u), (v, v)]

    def test_adjacent_cosines(self):
        exact, lagged = dl._adjacent_cosines(self._pairs())
        # exacts = u, u, v -> adjacent cos = 1, 0
        self.assertEqual([round(x, 2) for x in exact], [1.0, 0.0])
        # lagged vs exact at same index: (v,u)=0, (u,u)=1, (v,v)=1
        self.assertEqual([round(x, 2) for x in lagged], [0.0, 1.0, 1.0])

    def test_pairwise_report_structure(self):
        groups = [
            {"exact_vec": torch.tensor([1.0, 0.0]),
             "lagged_vec": torch.tensor([0.0, 1.0]),
             "task_vec": torch.tensor([1.0, 0.0])},
            {"exact_vec": torch.tensor([1.0, 0.0]),
             "lagged_vec": torch.tensor([1.0, 0.0]),
             "task_vec": torch.tensor([0.0, 1.0])},
        ]
        rep = dl._pairwise_report({"groups": groups})
        for key in ("exact_exact", "task_task", "task_exact",
                    "task_lagged", "lagged_exact"):
            self.assertIn(key, rep)
        self.assertAlmostEqual(rep["exact_exact"][0], 1.0)
        self.assertAlmostEqual(rep["task_task"][0], 0.0)
        self.assertAlmostEqual(rep["lagged_exact"][0], 0.0)
        self.assertAlmostEqual(rep["lagged_exact"][1], 1.0)


class TestResiduals(unittest.TestCase):
    def test_label_residual_balanced_pi_half(self):
        rows = [SimpleNamespace(outcome=1.0), SimpleNamespace(outcome=0.0)]
        vals = dl._label_residual_values(rows, pi=0.5)
        self.assertAlmostEqual(vals[0], 1.0, places=6)
        self.assertAlmostEqual(vals[1], -1.0, places=6)

    def test_label_residual_balanced_pi_quarter(self):
        rows = [SimpleNamespace(outcome=1.0), SimpleNamespace(outcome=0.0)]
        vals = dl._label_residual_values(rows, pi=0.25)
        # y=1 -> (1-0.25)/sqrt(0.25*0.75) = 0.75/sqrt(0.1875) = sqrt(3)
        self.assertAlmostEqual(vals[0], 3.0 ** 0.5, places=6)
        # y=0 -> -0.25/sqrt(0.1875) = -1/sqrt(3)
        self.assertAlmostEqual(vals[1], -(3.0 ** -0.5), places=6)

    def test_stratified_outcome_permutation_stays_in_strata(self):
        rows = _fake_rows(24, seed=7)
        rng = np.random.RandomState(0)
        for _ in range(5):
            shuffled = dl._stratified_outcome_permutation(rows, rng)
            for r, y in zip(rows, shuffled):
                self.assertEqual(int(r.horizon), int(1 + (rows.index(r) % 2)))
            # Each (horizon, role) stratum is a permutation of itself.
            by_stratum = {}
            for r, y in zip(rows, shuffled):
                key = (int(r.horizon), int(r.context["role"]))
                by_stratum.setdefault(key, ([], []))
                by_stratum[key][0].append(r.outcome)
                by_stratum[key][1].append(y)
            for (orig, perm) in by_stratum.values():
                self.assertEqual(sorted(orig), sorted(perm))


class TestVirtualStepSummary(unittest.TestCase):
    def _linear_records(self, effect=0.01, curvature=0.0, n_pairs=7,
                        step_frac=0.01, step_size=None):
        """Pure first-order fake: f(+e d) = f0 + effect, f(-e d) = f0 -
        effect.  The +-eps average is 0; the central effect is the
        signal.  ``step_size`` is the REAL parameter step (eps_abs)."""
        if step_size is None:
            step_size = step_frac  # |theta| = 1 default
        records = []
        for t in range(n_pairs):
            for sign in (1, -1):
                records.append({
                    "pair": t, "direction": "exact", "sign": sign,
                    "metric": "J", "before": 0.0,
                    "after": float(sign) * effect
                    + float(sign ** 2) * curvature,
                    "delta": float(sign) * effect
                    + float(sign ** 2) * curvature,
                    "step_frac": step_frac,
                    "step_size": step_size,
                })
        return records

    def test_central_effect_is_direction_not_average(self):
        summary = dl._summarize_virtual_records(
            self._linear_records(effect=0.01))
        entry = summary["exact_J_step0.01"]
        eff = entry["central_directional_effect"]
        self.assertAlmostEqual(eff["mean"], 0.01, places=6)
        self.assertEqual(eff["sign_rate"], 1.0)
        self.assertEqual(eff["median"], 0.01)
        self.assertEqual(entry["n_complete_pairs"], 7)
        # Directional derivative = effect / eps_abs = 0.01 / 0.01 = 1.
        der = entry["directional_derivative"]
        self.assertAlmostEqual(der["mean"], 1.0, places=6)
        self.assertAlmostEqual(
            entry["second_directional_derivative_mean"], 0.0, places=9)
        self.assertEqual(entry["missing_plus_pairs"], 0)
        self.assertEqual(entry["missing_minus_pairs"], 0)
        self.assertEqual(entry["ci_kind"],
                         "descriptive_bootstrap_overlapping_pairs")

    def test_derivative_uses_real_step_size_not_frac(self):
        # |theta| = 5, eps_abs = 0.01 * 5 = 0.05.  A per-step effect of
        # 0.05 must give D = 0.05 / 0.05 = 1 (the old code would divide
        # by 0.01 and report 5).
        summary = dl._summarize_virtual_records(
            self._linear_records(effect=0.05, step_frac=0.01,
                                 step_size=0.05))
        entry = summary["exact_J_step0.01"]
        self.assertAlmostEqual(
            entry["directional_derivative"]["mean"], 1.0, places=6)
        # H = (d+ + d-)/eps_abs^2 with pure curvature 0.005.
        summary2 = dl._summarize_virtual_records(
            self._linear_records(effect=0.0, curvature=0.005,
                                 step_frac=0.01, step_size=0.05))
        e2 = summary2["exact_J_step0.01"]
        # sym sum = 0.01 -> / 0.05^2 = 4.0
        self.assertAlmostEqual(
            e2["second_directional_derivative_mean"], 4.0, places=6)

    def test_curvature_goes_to_second_derivative_slot(self):
        summary = dl._summarize_virtual_records(
            self._linear_records(effect=0.0, curvature=0.005))
        entry = summary["exact_J_step0.01"]
        eff = entry["central_directional_effect"]
        self.assertAlmostEqual(eff["mean"], 0.0, places=9)
        # H = (d+ + d-)/eps^2 = 2*0.005 / 1e-4 = 100.
        self.assertAlmostEqual(
            entry["second_directional_derivative_mean"], 100.0, places=6)

    def test_directional_derivative_scales_with_step(self):
        # Same linear slope: derivative must be step-independent.
        s1 = dl._summarize_virtual_records(
            self._linear_records(effect=0.0025, step_frac=0.0025))
        s2 = dl._summarize_virtual_records(
            self._linear_records(effect=0.01, step_frac=0.01))
        d1 = s1["exact_J_step0.0025"]["directional_derivative"]["mean"]
        d2 = s2["exact_J_step0.01"]["directional_derivative"]["mean"]
        self.assertAlmostEqual(d1, d2, places=6)
        self.assertAlmostEqual(d1, 1.0, places=6)

    def test_pair_alignment_insensitive_to_record_order(self):
        records = self._linear_records(effect=0.01, n_pairs=7)
        rng = np.random.RandomState(3)
        shuffled = list(records)
        rng.shuffle(shuffled)
        s_orig = dl._summarize_virtual_records(records)
        s_shuf = dl._summarize_virtual_records(shuffled)
        a = s_orig["exact_J_step0.01"]
        b = s_shuf["exact_J_step0.01"]
        self.assertAlmostEqual(
            a["central_directional_effect"]["mean"],
            b["central_directional_effect"]["mean"], places=9)
        self.assertEqual(a["n_complete_pairs"], b["n_complete_pairs"])

    def test_incomplete_pair_excluded_and_counted(self):
        records = self._linear_records(effect=0.01, n_pairs=3)
        # Drop the -1 side of pair 1 entirely.
        records = [r for r in records
                   if not (r["pair"] == 1 and r["sign"] == -1)]
        summary = dl._summarize_virtual_records(records)
        entry = summary["exact_J_step0.01"]
        self.assertEqual(entry["n_complete_pairs"], 2)
        self.assertEqual(entry["missing_minus_pairs"], 1)
        self.assertEqual(entry["missing_plus_pairs"], 0)
        eff = entry["central_directional_effect"]
        self.assertEqual(eff["n"], 2)

    def test_duplicate_side_raises(self):
        records = self._linear_records(effect=0.01, n_pairs=1)
        duplicate = dict(records[0])
        records.append(duplicate)
        with self.assertRaises(AssertionError):
            dl._summarize_virtual_records(records)

    def test_bootstrap_ci_95(self):
        # With a strictly positive effect the 95% CI must not cross 0.
        summary = dl._summarize_virtual_records(
            self._linear_records(effect=0.01, n_pairs=20))
        eff = summary["exact_J_step0.01"]["central_directional_effect"]
        self.assertGreater(eff["ci_lo"], 0.0)
        self.assertAlmostEqual(eff["ci_hi"], 0.01, places=6)


class TestZeroReplayCheck(unittest.TestCase):
    """Locks the zero-replay consistency contract (the regression that
    once reported a +eps result as the 'zero' replay)."""

    def _metrics(self, j, contrast, nll):
        return {"J": j, "J_outcome_contrast": contrast,
                "task_nll": nll}

    def test_identical_passes(self):
        m = self._metrics(4.5, 2.0, 0.1)
        per, ok = dl._zero_replay_check(m, m)
        self.assertTrue(ok)
        for key in ("J", "J_outcome_contrast", "task_nll"):
            self.assertTrue(per[key]["available"])
            self.assertTrue(per[key]["identical"])
            self.assertLessEqual(per[key]["abs_diff"], 1e-6)

    def test_mismatch_fails_with_diff(self):
        a = self._metrics(4.5, 2.0, 0.1)
        b = self._metrics(4.5, 2.0, 0.2)  # NLL differs
        per, ok = dl._zero_replay_check(a, b)
        self.assertFalse(ok)
        self.assertFalse(per["task_nll"]["identical"])
        self.assertGreater(per["task_nll"]["abs_diff"], 1e-6)

    def test_none_is_not_consistency(self):
        a = self._metrics(4.5, None, 0.1)
        b = self._metrics(4.5, None, 0.1)
        per, ok = dl._zero_replay_check(a, b)
        self.assertFalse(ok)
        self.assertFalse(per["J_outcome_contrast"]["available"])
        self.assertFalse(per["J_outcome_contrast"]["identical"])

    def test_tolerance_boundary(self):
        # float32-scale tolerance: last-bit CUDA noise (~1e-8) passes,
        # a genuine state-restore failure (~1e-3) fails.
        a = self._metrics(4.5, 2.0, 0.1)
        b = self._metrics(4.5 + 1e-8, 2.0, 0.1)
        per, ok = dl._zero_replay_check(a, b)
        self.assertTrue(ok)  # within float32 tol
        b2 = self._metrics(4.5 + 1e-3, 2.0, 0.1)
        per2, ok2 = dl._zero_replay_check(a, b2)
        self.assertFalse(ok2)


class TestCommonProbe(unittest.TestCase):
    def test_common_probe_fails_cleanly_without_reference(self):
        out = dl._common_probe_report({}, [])
        self.assertEqual(out, {"failed": "no_current_reference"})


if __name__ == "__main__":
    unittest.main()
