"""A0 loop: end-to-end four-phase smoke on tiny synthetic data.

Drives the vendored TGN host through a vanilla PRSS core (identity trace
producer), so it needs the torch<->numpy bridge and runs on the GPU box
(skipped locally, same as test_jodie_vendor/adapter/loop).
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from prss.a0.probes import A0Probes
from prss.data.jodie import JodieData
from prss.hosts.jodie_tgn import jodie_preagg_dim
from prss.monitoring import MonitorWriter
from prss.training.a0_loop import A0NodeClassificationLoop

from test_jodie_adapter import make_tiny_prss, make_tiny_tgn, install_adapter
from test_jodie_vendor import REQUIRES_NUMPY_BRIDGE


@REQUIRES_NUMPY_BRIDGE
class TestA0LoopSmoke(unittest.TestCase):
    """Four phases run on the tiny host: R frozen, operators fit, audit table
    filled, dual heads trained, the four output files written."""

    def setUp(self):
        tgn, device, stream = make_tiny_tgn()
        sources, destinations, timestamps, edge_idxs, labels = stream
        self.tgn = tgn
        self.device = device
        # Chronological splits that continue the train stream (same contract
        # as TestLoopSmoke): val/test never rewind into the past.
        self.train = JodieData(sources[:60], destinations[:60],
                               timestamps[:60], edge_idxs[:60], labels[:60])
        self.val = JodieData(sources[60:80], destinations[60:80],
                             timestamps[60:80], edge_idxs[60:80], labels[60:80])
        self.test = JodieData(sources[80:], destinations[80:],
                              timestamps[80:], edge_idxs[80:], labels[80:])
        config, prss = make_tiny_prss(variant="vanilla", candidate_dim=8)
        adapter = install_adapter(tgn, prss)
        preagg_dim = jodie_preagg_dim(8, 8, 8, 4)
        probes = A0Probes(preagg_dim=preagg_dim, d_context=8, seed=0,
                          device=device)
        self.out_dir = Path(tempfile.mkdtemp(prefix="a0_smoke_"))
        monitor = MonitorWriter(self.out_dir, fail_on_error=True,
                                reset_files=True)
        self.loop = A0NodeClassificationLoop(
            tgn=tgn, adapter=adapter, prss_core=prss, probes=probes,
            device=device, batch_size=8, n_neighbors=4, trace_roots=4,
            trace_mode="evenly_spaced", rank_r=3, lambda_x=1e-4,
            lambda_gamma=1e-3, lambda_audit=1e-3, frac_a=0.2, frac_b=0.2,
            frac_c=0.2, d_slice_only=False, gates=None, gate_mode="report",
            monitor=monitor, seed=0, out_dir=self.out_dir, lr=3e-4,
            n_epoch=2, patience=3, drop_out=0.1, selection_metric="auc")

    def test_four_phases_run_and_outputs(self):
        summary = self.loop.run(self.train, self.val, self.test)
        self.loop.finalize(summary)

        self.assertEqual(summary["status"], "complete")
        # Phase A: every quotient solved and frozen.
        for tau, q in self.loop.quotients.items():
            self.assertTrue(q.solved, tau)
            self.assertEqual(tuple(q.r_matrix.shape), (3, 8))
            self.assertTrue(torch.isfinite(q.r_matrix).all())
        # Phase B: every operator frozen with a finite B̂.
        for op in self.loop.operators.values():
            self.assertIsNotNone(op.b_matrix)
            self.assertTrue(torch.isfinite(op.b_matrix).all())
        # Phase C: audit table with the gate-relevant keys.
        audit = summary["audit"]
        for key in ("ess", "rank_tail_max", "closure_residual_max",
                    "path_gain_product", "prediction_by_tau",
                    "closure_by_sigma", "gates"):
            self.assertIn(key, audit)
        self.assertGreater(audit["ess"], 0)
        # Phase D: dual heads trained with finite held-out metrics.
        self.assertIn("a0_readout", summary)
        self.assertIn("baseline_decoder", summary)
        for head in ("a0_readout", "baseline_decoder"):
            self.assertTrue(np.isfinite(summary[head]["test"]["nll"]))
            self.assertIn("auc", summary[head]["test"])
        self.assertIn("delta_auc", summary)
        # Four output files.
        for name in ("metrics.jsonl", "summary.json", "_SUCCESS.json"):
            self.assertTrue((self.out_dir / name).exists(), name)
        success = json.loads((self.out_dir / "_SUCCESS.json").read_text())
        self.assertEqual(success["status"], "complete")
        self.assertEqual(success["best_epoch_a0"],
                         summary["a0_readout"]["best_epoch"])
        # Isolation: no trace survives evaluation.
        self.assertIsNone(self.loop.adapter.trace)

    def test_gate_stop_writes_summary_with_reason(self):
        self.loop.gate_mode = "stop"
        self.loop.gates = {"G1": 0.0}  # any rank tail fails compressibility
        summary = self.loop.run(self.train, self.val, self.test)
        self.loop.finalize(summary)
        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(summary["stop_phase"], "C")
        self.assertIn("G1", summary["stop_reason"])
        success = json.loads((self.out_dir / "_SUCCESS.json").read_text())
        self.assertEqual(success["status"], "stopped")
        self.assertIn("stop_reason", success)
        # Phase D never ran: no readout metrics.
        self.assertNotIn("a0_readout", summary)


if __name__ == "__main__":
    unittest.main()
