"""Rolling-checkpoint save/restore contracts.

Regression for the bug that killed the spectral/direct runs: checkpoint.py
assumed PyG's TGNMemory message stores were ``[(msg, t), ...]`` lists, but
PyG >= 2.6 keeps them as a two-level dict ``{tag: {node: (src, dst, t, msg)}}``
whose values are 4-tuples of tensors (empty tensors after reset, batched once
messages arrive).  The helpers must round-trip exactly that structure, and a
full CheckpointManager save/load must restore model, optimizer, RNG and stores.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from prss.training.checkpoint import (
    CheckpointManager, _msg_store_to_cpu, _msg_store_from_cpu,
)


def _fake_store(num_nodes=8, raw_msg_dim=8, touched=None):
    """Build the exact PyG 2.8 reset/update shapes on CPU."""
    i = torch.empty(0, dtype=torch.long)
    msg = torch.empty(0, raw_msg_dim)
    store = {j: (i, i, i, msg) for j in range(num_nodes)}
    for node, (s, d, t, m) in (touched or {}).items():
        store[node] = (s, d, t, m)
    return store


def _tgb_module():
    """Return (TGNMemory, IdentityMessage, LastAggregator) or skip."""
    try:
        from torch_geometric.nn.models.tgn import (  # noqa: PLC0415
            IdentityMessage, LastAggregator, TGNMemory)
        return TGNMemory, IdentityMessage, LastAggregator
    except ImportError:
        raise unittest.SkipTest("torch_geometric not available")


class TestMsgStoreHelpers(unittest.TestCase):
    """Two-level {tag: {node: (src, dst, t, msg)}} survives CPU round-trip."""

    def test_roundtrip_with_updates_and_empties(self):
        store = {
            "s": _fake_store(touched={
                2: (torch.tensor([10, 11]), torch.tensor([20, 21]),
                    torch.tensor([1.5, 2.5]), torch.randn(2, 8)),
                5: (torch.tensor([30]), torch.tensor([40]),
                    torch.tensor([3.5]), torch.randn(1, 8)),
            }),
            "d": _fake_store(touched={
                3: (torch.tensor([9]), torch.tensor([8]),
                    torch.tensor([7.5]), torch.randn(1, 8)),
            }),
        }
        payload = _msg_store_to_cpu(store)
        restored = {"s": {}, "d": {}}  # live-store wrapper, as train.py passes
        _msg_store_from_cpu(restored, payload, torch.device("cpu"))
        self.assertEqual(set(restored), {"s", "d"})
        for tag in store:
            self.assertEqual(set(restored[tag]), set(store[tag]))
            for node, entries in store[tag].items():
                self.assertEqual(len(entries), 4)
                for a, b in zip(entries, restored[tag][node]):
                    self.assertEqual(a.shape, b.shape)
                    self.assertTrue(torch.equal(a, b))

    def test_reset_state_empty_store_roundtrips(self):
        store = {"s": _fake_store(), "d": _fake_store()}
        payload = _msg_store_to_cpu(store)
        restored = {"s": {}, "d": {}}
        _msg_store_from_cpu(restored, payload, torch.device("cpu"))
        self.assertEqual(len(restored["s"]), 8)
        self.assertEqual(tuple(restored["s"][0][3].shape), (0, 8))

    def test_restore_does_not_mutate_payload(self):
        store = {"s": _fake_store(touched={1: (torch.tensor([7]), torch.tensor([8]),
                                               torch.tensor([9.0]), torch.randn(1, 8))}),
                 "d": _fake_store()}
        payload = _msg_store_to_cpu(store)
        snap = payload["s"][1]
        restored = {"s": {}, "d": {}}
        _msg_store_from_cpu(restored, payload, torch.device("cpu"))
        self.assertTrue(all(x.device.type == "cpu" for x in snap))
        self.assertTrue(torch.equal(snap[0], torch.tensor([7])))


class TestTGNMemoryMsgStore(unittest.TestCase):
    """End-to-end against the real PyG module on whatever device is present."""

    def setUp(self):
        self.TGNMemory, self.IdentityMessage, self.LastAggregator = _tgb_module()

    @staticmethod
    def _device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _make_memory(self):
        return self.TGNMemory(
            num_nodes=50, raw_msg_dim=8, memory_dim=16, time_dim=16,
            message_module=self.IdentityMessage(raw_msg_dim=8, memory_dim=16,
                                                time_dim=16),
            aggregator_module=self.LastAggregator())

    def _make_trained(self):
        """Construct on device then reset: PyG's ``.to()`` does NOT move the
        message-store entries created at ``__init__``, so a real training loop
        resets after moving (our event loop does exactly that)."""
        device = self._device()
        mem = self._make_memory().to(device)
        mem.reset_state()
        mem.train()
        return mem, device

    def test_store_roundtrip_preserves_compute_msg(self):
        mem, device = self._make_trained()
        src = torch.tensor([1, 2, 3, 1], device=device)
        dst = torch.tensor([4, 5, 6, 7], device=device)
        # TGB timestamps are int64 (seconds); float here would be promoted
        # against the long empty store entries and break pyg's long
        # ``last_update`` buffer on the train->eval flush.
        t = torch.tensor([10, 20, 30, 40], device=device)
        raw = torch.randn(4, 8, device=device)
        mem.update_state(src, dst, t, raw)
        n_id = torch.tensor([1, 4, 2, 5, 9], device=device)
        mem(n_id)  # exercise _compute_msg over mixed empty/updated entries

        # Roundtrip must not change what _compute_msg would produce; capture
        # the pre-roundtrip outputs per tag first (_compute_msg is pure).
        before = {tag: mem._compute_msg(n_id, getattr(mem, f"msg_{tag}_store"),
                                        mem.msg_s_module)
                  for tag in ("s", "d")}

        payload = _msg_store_to_cpu({"s": mem.msg_s_store, "d": mem.msg_d_store})
        restored = {"s": mem.msg_s_store, "d": mem.msg_d_store}
        _msg_store_from_cpu(restored, payload, device)

        for tag in ("s", "d"):
            after = mem._compute_msg(n_id, getattr(mem, f"msg_{tag}_store"),
                                     mem.msg_s_module)
            for a, b in zip(before[tag], after):
                self.assertTrue(torch.equal(a, b))

    def test_reset_state_store_roundtrips(self):
        mem, device = self._make_trained()
        n_id = torch.tensor([3], device=device)
        before = mem._compute_msg(n_id, mem.msg_s_store, mem.msg_s_module)

        payload = _msg_store_to_cpu({"s": mem.msg_s_store, "d": mem.msg_d_store})
        restored = {"s": mem.msg_s_store, "d": mem.msg_d_store}
        _msg_store_from_cpu(restored, payload, device)

        after = mem._compute_msg(n_id, mem.msg_s_store, mem.msg_s_module)
        for a, b in zip(before, after):
            self.assertTrue(torch.equal(a, b))


class TestCheckpointManager(unittest.TestCase):
    """Full save/load: model + optimizer + RNG + message stores."""

    def test_full_roundtrip(self):
        TGNMemory, IdentityMessage, LastAggregator = _tgb_module()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mem = TGNMemory(num_nodes=20, raw_msg_dim=4, memory_dim=8, time_dim=8,
                        message_module=IdentityMessage(raw_msg_dim=4, memory_dim=8,
                                                       time_dim=8),
                        aggregator_module=LastAggregator()).to(device)
        mem.reset_state()
        mem.train()
        opt = torch.optim.Adam(mem.parameters(), lr=1e-3)

        src = torch.tensor([1, 2, 3, 1], device=device)
        dst = torch.tensor([4, 5, 6, 7], device=device)
        # int64 like TGB timestamps (see test_store_roundtrip_preserves_compute_msg).
        t = torch.tensor([10, 20, 30, 40], device=device)
        raw = torch.randn(4, 4, device=device)
        mem.update_state(src, dst, t, raw)
        mem_snapshot = mem.memory.detach().clone()

        # Built before save so its RNG consumption cannot disturb the
        # save-time stream reference.  Enter eval mode immediately (separate
        # calls: pyg 2.8's train() override returns None, breaking the
        # ``.to().eval()`` chain).  A train->eval transition would flush the
        # empty store into ``last_update`` and zero the values we later copy
        # over via load_state_dict.
        ref = TGNMemory(num_nodes=20, raw_msg_dim=4, memory_dim=8, time_dim=8,
                        message_module=IdentityMessage(raw_msg_dim=4, memory_dim=8,
                                                       time_dim=8),
                        aggregator_module=LastAggregator()).to(device)
        ref.eval()

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = CheckpointManager(str(Path(tmp) / "rolling.pt"))
            ckpt.save(
                model_components={"memory": mem},
                optimizer=opt,
                unrestricted_optimizer=None,
                epoch=7, next_batch=0, global_step=3900,
                best_score=0.42, best_epoch=6, bad_rounds=1,
                train_state={"phase": "b"},
                memory_msg_stores={"s": mem.msg_s_store, "d": mem.msg_d_store})
            # RNG reference: what the save-time generator produces next.
            after_save = torch.rand(3, device=device)

            # Corrupt everything, then restore.
            mem.memory.data.mul_(0.0)
            mem.msg_s_store.clear()
            mem.msg_d_store.clear()
            ckpt.load(
                model_components={"memory": mem},
                optimizer=opt,
                unrestricted_optimizer=None,
                memory_msg_stores={"s": mem.msg_s_store, "d": mem.msg_d_store},
                device=device)

            # Model memory exactly restored.
            self.assertTrue(torch.equal(mem.memory, mem_snapshot))
            # Message stores restored (node 1 received a message above).
            self.assertEqual(len(mem.msg_s_store), 20)
            self.assertGreater(mem.msg_s_store[1][0].numel(), 0)
            self.assertEqual(mem.msg_s_store[1][0].device.type, device.type)
            # State_dict matches a clean copy (eval forward reads memory only).
            # Enter eval FIRST: pyg's train->eval transition flushes pending
            # store messages into memory, so the clean copy must be taken from
            # the flushed state to compare like with like.
            mem.eval()
            ref.load_state_dict(mem.state_dict())
            out_a, last_a = mem(torch.arange(20, device=device))
            out_b, last_b = ref(torch.arange(20, device=device))
            self.assertTrue(torch.equal(out_a, out_b))
            self.assertTrue(torch.equal(last_a, last_b))
            # Optimizer and RNG restored: next random draw equals the
            # save-time continuation.
            self.assertTrue(torch.equal(torch.rand(3, device=device), after_save))


class TestExtraPayload(unittest.TestCase):
    """JODIE memory_backup rides along without touching the TGB path."""

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
                unrestricted_optimizer=None, epoch=1, next_batch=0,
                global_step=10, best_score=0.5, best_epoch=1, bad_rounds=0,
                train_state={}, memory_msg_stores=None, extra_payload=extra)
            payload = ckpt.load(
                model_components={"m": model}, optimizer=opt,
                unrestricted_optimizer=None, memory_msg_stores=None,
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
        """TGB line: no extra_payload -> payload["extra"] is None, no crash."""
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = CheckpointManager(str(Path(tmp) / "plain.pt"))
            ckpt.save(
                model_components={"m": model}, optimizer=opt,
                unrestricted_optimizer=None, epoch=0, next_batch=0,
                global_step=0, best_score=0.0, best_epoch=0, bad_rounds=0,
                train_state={}, memory_msg_stores=None)
            payload = ckpt.load(
                model_components={"m": model}, optimizer=opt,
                unrestricted_optimizer=None, memory_msg_stores=None,
                device=torch.device("cpu"))
            self.assertIsNone(payload["extra"])


if __name__ == "__main__":
    unittest.main()
