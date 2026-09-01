"""L3 acceptance tests: CCMHostAdapter + J_mem (plan v2 L3).

Test C (strengthened form): replacing ALL future inputs u_{v+1..k}
(equal-length swaps, v = k-3) must leave the cut memory M_v and
z_v = J_mem(M_v) exactly unchanged — the extraction is causal.
Plus: z_v carries gradient back into Gamma, the cache protocol, and the
fixed CountSketch lift against a manual reference.
"""

import sys
import unittest
from pathlib import Path

_CCM_DIR = Path(__file__).resolve().parents[1] / "third_party" / "ccm"
if str(_CCM_DIR) not in sys.path:
    sys.path.insert(0, str(_CCM_DIR))

import torch

from transformers.models.llama.configuration_llama import LlamaConfig

from src.arch.ccm_llama import LlamaForCausalLM_CCM
from rpbe.hosts.ccm.ccm_patch import attach_gamma
from rpbe.hosts.ccm.adapter import CCMHostAdapter
from rpbe.llm.mem_lift import JMemLift

C0, C1, S0, S1 = 32000, 32001, 32002, 32003
COMP_IDS = [C0, C1]
SUM_IDS = [S0, S1]


def make_config():
    cfg = LlamaConfig(vocab_size=32004, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2,
                      max_position_embeddings=512)
    cfg.comp_relative_embedding = "base"
    return cfg


def make_model():
    model = LlamaForCausalLM_CCM(make_config())
    model.update_comp_token(COMP_IDS, SUM_IDS)
    return model


def make_merge_input(n_turns, u_len=3, u_base=100):
    seq = [1]
    for t in range(n_turns):
        seq += [u_base + t] * u_len
        seq += [C0, C1, S0, S1]
    return torch.tensor([seq])


def sum_rows(n_turns, u_len=3):
    """Positions of (S0, S1) per turn for make_merge_input."""
    out = []
    pos = 1
    for _ in range(n_turns):
        pos += u_len + 2
        out.append((pos, pos + 1))
        pos += 2
    return out


class TestJMemLift(unittest.TestCase):
    def test_matches_manual_sketch(self):
        torch.manual_seed(3)
        lift = JMemLift(n_layers=2, n_heads=2, n_slots=2, kv_pairs=2,
                        head_dim=4, z_dim=8, seed=11)
        mem = torch.randn(3, lift.full_dim)
        z = lift(mem)
        self.assertEqual(tuple(z.shape), (3, 8))
        # Manual reference: same index_add contract.
        ref = torch.zeros(3, 8)
        ref.index_add_(1, lift.sketch_cols,
                       mem[:, lift.sketch_rows] * lift.sketch_signs)
        ref = ref * lift.scale
        self.assertTrue(torch.allclose(z, ref, rtol=0, atol=0))

    def test_layout_roundtrip(self):
        torch.manual_seed(4)
        B, H, R, D = 2, 2, 4, 4
        k_rows = [torch.randn(B, H, R, D) for _ in range(2)]
        v_rows = [torch.randn(B, H, R, D) for _ in range(2)]
        mem = JMemLift.pack_sum_mem(k_rows, v_rows)
        n_layers, kv, hd = 2, 2, D
        # Layout: (layer, head, row, kv, dim) with R SUM rows per layer.
        self.assertEqual(mem.shape[1], n_layers * H * R * kv * hd)
        # Unpack spot checks: layer 1, head 0, row 1, K (kv 0).
        base = ((1 * H + 0) * R + 1) * kv * hd + 0 * hd
        self.assertTrue(torch.equal(mem[:, base:base + D],
                                    k_rows[1][:, 0, 1, :]))


class TestHostAdapter(unittest.TestCase):
    """Test C: causal extraction of z_v = J_mem(M_v)."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = make_model()
        attach_gamma(self.model, hidden=16)
        self.adapter = CCMHostAdapter(
            self.model, n_layers=2, n_heads=2, head_dim=16, z_dim=16,
            seed=5)
        self.ids = make_merge_input(5)
        self.rows = sum_rows(5)

    def _z_at(self, ids, v):
        with torch.no_grad():
            self.model(input_ids=ids)
            pos = torch.tensor([self.rows[v]], dtype=torch.long)
            return self.adapter.extract_z(pos)

    def test_z_shape_and_finite(self):
        z = self._z_at(self.ids, v=2)
        self.assertEqual(tuple(z.shape), (1, 16))
        self.assertTrue(torch.isfinite(z).all())
        self.assertGreater(float(z.abs().sum()), 0.0)

    def test_causal_isolation_future_swap(self):
        """Swapping ALL future utterances (equal length) leaves z_v exact."""
        z_ref = self._z_at(self.ids, v=2)
        swapped = self.ids.clone()
        # Turns 3..5 (0-indexed future of the v=2 cut) get different
        # tokens of the SAME length.
        for t in range(3, 5):
            span = slice(self.rows[t][0] - 5, self.rows[t][0] - 2)
            swapped[0, span] = 300 + t
        z_new = self._z_at(swapped, v=2)
        self.assertTrue(torch.allclose(z_ref, z_new, rtol=0, atol=0))

    def test_z_carries_gamma_gradient(self):
        for layer in self.model.model.layers:
            layer.self_attn.gamma.s.data.fill_(0.1)
        self.model.train()
        out = self.model(input_ids=self.ids)
        pos = torch.tensor([self.rows[2]], dtype=torch.long)
        z = self.adapter.extract_z(pos)
        (z.sum() + out.logits.sum() * 0.0).backward()
        for layer in self.model.model.layers:
            g = layer.self_attn.gamma
            self.assertIsNotNone(g.s.grad)
            self.assertNotEqual(float(g.s.grad), 0.0)
            self.assertIsNotNone(g.U.weight.grad)
        self.adapter.clear()

    def test_cache_requires_forward(self):
        self.adapter.clear()
        with self.assertRaises(RuntimeError):
            self.adapter.extract_z(torch.tensor([[0, 1]]))

    def test_clear_after_batch(self):
        self._z_at(self.ids, v=2)
        self.adapter.clear()
        self.assertTrue(all(e is None for e in self.adapter._cache))


if __name__ == "__main__":
    unittest.main()
