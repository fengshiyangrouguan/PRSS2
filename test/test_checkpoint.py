"""Rolling-checkpoint save/restore contracts (pure torch, no PyG).

A full CheckpointManager save/load must restore model, optimizer and RNG; the
JODIE memory backup rides along through ``extra_payload`` verbatim.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from rpbe.training.checkpoint import CheckpointManager


class TestCheckpointManager(unittest.TestCase):
    """Full save/load: model + optimizer + RNG."""

    def test_full_roundtrip(self):
        device = torch.device("cpu")
        model = torch.nn.Linear(8, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = CheckpointManager(str(Path(tmp) / "rolling.pt"))
            ckpt.save(
                model_components={"model": model},
                optimizer=opt,
                epoch=7, next_batch=0, global_step=3900,
                best_score=0.42, best_epoch=6, bad_rounds=1,
                train_state={"phase": "b"})
            # RNG reference: what the save-time generator produces next.
            after_save = torch.rand(3, device=device)

            # Corrupt everything, then restore.
            for p in model.parameters():
                p.data.mul_(0.0)
            ckpt.load(
                model_components={"model": model},
                optimizer=opt, device=device)

            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, model_snapshot[k]))
            # RNG restored: next random draw equals the save-time continuation.
            self.assertTrue(torch.equal(torch.rand(3, device=device), after_save))


class TestExtraPayload(unittest.TestCase):
    """JODIE memory_backup rides along without touching the rest."""

    def test_extra_payload_roundtrip_verbatim(self):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        backup = {
            "memory": torch.randn(5, 8),
            "last_update": torch.zeros(5),
            "messages": {1: [(torch.randn(3, 4), torch.tensor(1.5))]},
        }
        extra = {"memory_backup": backup, "epoch_score": 0.71}
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = CheckpointManager(str(Path(tmp) / "extra.pt"))
            ckpt.save(
                model_components={"m": model}, optimizer=opt,
                epoch=1, next_batch=0, global_step=10, best_score=0.5,
                best_epoch=1, bad_rounds=0, train_state={}, extra_payload=extra)
            payload = ckpt.load(
                model_components={"m": model}, optimizer=opt,
                device=torch.device("cpu"))
        got = payload["extra"]
        self.assertEqual(got["epoch_score"], 0.71)
        self.assertTrue(torch.equal(got["memory_backup"]["memory"], backup["memory"]))
        self.assertTrue(torch.equal(got["memory_backup"]["last_update"],
                                    backup["last_update"]))
        msg = got["memory_backup"]["messages"][1][0]
        self.assertTrue(torch.equal(msg[0], backup["messages"][1][0][0]))
        self.assertEqual(float(msg[1]), 1.5)

    def test_without_extra_key_present_as_none(self):
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = CheckpointManager(str(Path(tmp) / "plain.pt"))
            ckpt.save(
                model_components={"m": model}, optimizer=opt,
                epoch=0, next_batch=0, global_step=0, best_score=0.0,
                best_epoch=0, bad_rounds=0, train_state={})
            payload = ckpt.load(
                model_components={"m": model}, optimizer=opt,
                device=torch.device("cpu"))
            self.assertIsNone(payload["extra"])


if __name__ == "__main__":
    unittest.main()
