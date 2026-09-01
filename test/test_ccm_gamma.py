"""L2 acceptance tests for the Gamma residual on the vendored CCM llama.

Covers (plan v2 L2):

- Test A: with the gate s == 0 the Gamma path reproduces the official
  arithmetic-mean merge exactly (bit-identical logits).
- Mask construction: per-row turn counts, previous-turn COMP means and the
  previous same-slot SUM selector.
- The recurrence scan equals a manual per-turn gold loop (the vectorized
  scan implements M_t = mean + R_theta(M_{t-1}, h_t, t) exactly).
- Arbitrary recurrence length k > 14 (20-turn input) stays finite and the
  residual changes the output once s != 0.
- RoPE: same-slot COMP positions share one basis across turns.
- Engineering closure: PEFT wrapping keeps Gamma trainable (1), optimizer
  param groups contain all Gamma params (2), checkpoint roundtrip is exact
  (3/4), paired-seed hash covers LoRA+Gamma+COMP/SUM embeddings (5), and
  the first optimizer step moves s while U/V move from the second (6).
"""

import sys
import unittest
from pathlib import Path

# The vendored CCM package is named ``src``; put its parent first so
# ``src.arch.ccm_llama`` resolves there (the repo's own ``rpbe`` imports
# are unaffected — they do not use the ``src.`` prefix).
_CCM_DIR = Path(__file__).resolve().parents[1] / "third_party" / "ccm"
if str(_CCM_DIR) not in sys.path:
    sys.path.insert(0, str(_CCM_DIR))

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from src.arch.ccm_llama import LlamaForCausalLM_CCM, update_position_ids
from rpbe.hosts.ccm.ccm_patch import attach_gamma, paired_seed_hash
from rpbe.hosts.ccm.gamma_residual import GammaResidual
from rpbe.training.checkpoint import CheckpointManager

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


def make_merge_input(n_turns, u_len=3):
    """[bos, u_t, C0, C1, S0, S1] per turn (every turn carries a block)."""
    seq = [1]
    for t in range(n_turns):
        seq += [100 + t] * u_len
        seq += [C0, C1, S0, S1]
    return torch.tensor([seq])


def sum_rows(n_turns, u_len=3):
    """Positions of S0/S1 per turn for make_merge_input."""
    out = []
    pos = 1
    for _ in range(n_turns):
        pos += u_len + 2
        out.append((pos, pos + 1))  # (S0, S1)
        pos += 2
    return out


class TestGammaZeroInit(unittest.TestCase):
    """Test A: s == 0 reproduces the official merge exactly."""

    def test_identical_logits_with_zero_gate(self):
        torch.manual_seed(0)
        official = make_model()
        torch.manual_seed(0)
        ours = make_model()
        attach_gamma(ours, hidden=16)
        ids = make_merge_input(3)
        with torch.no_grad():
            logits_ref = official(input_ids=ids).logits
            logits_ours = ours(input_ids=ids).logits
        # The residual is s * U tanh(V x) with s == 0, i.e. exactly 0.0,
        # so the merged K/V and therefore the logits must be identical.
        self.assertTrue(torch.allclose(logits_ref, logits_ours, rtol=0,
                                       atol=0))

    def test_gamma_params_zero_at_attach(self):
        torch.manual_seed(0)
        model = make_model()
        attach_gamma(model, hidden=16)
        for layer in model.model.layers:
            g = layer.self_attn.gamma
            self.assertEqual(g.s.detach().item(), 0.0)
            self.assertNotEqual(g.V.weight.detach().abs().sum().item(), 0.0)
            self.assertNotEqual(g.U.weight.detach().abs().sum().item(), 0.0)

    def test_attach_validations(self):
        model = make_model()
        model.update_comp_token([C0], [S0])  # n_tok = 1 is rejected
        with self.assertRaises(ValueError):
            attach_gamma(model)
        model.update_comp_token(COMP_IDS, SUM_IDS)
        attach_gamma(model)
        with self.assertRaises(ValueError):
            attach_gamma(model)  # double attach


class TestGammaMasks(unittest.TestCase):
    """Mask-level construction checks (previous-turn mean + SUM selector)."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = make_model()
        attach_gamma(self.model, hidden=16)
        self.ids = make_merge_input(3)
        (self.comp_mask, self.sum_mask, self.sum_attn, self.sum_prev,
         self.sum_prev_sum, self.sum_count) = \
            self.model.model.get_comp_sum_mask(self.ids)

    def test_turn_counts(self):
        rows = sum_rows(3)
        for t, (s0, s1) in enumerate(rows, start=1):
            self.assertEqual(float(self.sum_count[0, s0]), float(t))
            self.assertEqual(float(self.sum_count[0, s1]), float(t))
        # Non-SUM rows carry zero counts.
        self.assertEqual(float(self.sum_count[0, 0]), 0.0)

    def test_prev_mean_columns(self):
        rows = sum_rows(3)
        # Turn 3 S0: same-slot COMPs of turns 1-2 only (own-turn C0 at
        # s0-2 is excluded).
        s0, s1 = rows[2]
        self.assertEqual(self.sum_prev[0, s0].nonzero().flatten().tolist(),
                         [rows[0][0] - 2, rows[1][0] - 2])
        self.assertEqual(self.sum_prev[0, s1].nonzero().flatten().tolist(),
                         [rows[0][1] - 2, rows[1][1] - 2])
        # Turn 1 has no previous mean.
        self.assertEqual(float(self.sum_prev[0, rows[0][0]].sum()), 0.0)
        # Normalized rows: every valid row sums to 1.
        for s0, s1 in rows[1:]:
            self.assertAlmostEqual(float(self.sum_prev[0, s0].sum()), 1.0,
                                   places=6)
            self.assertAlmostEqual(float(self.sum_prev[0, s1].sum()), 1.0,
                                   places=6)

    def test_prev_sum_selector(self):
        rows = sum_rows(3)
        # Each row >= turn 2 selects exactly the previous same-slot SUM.
        self.assertEqual(self.sum_prev_sum[0, rows[1][0]].nonzero()
                         .flatten().tolist(), [rows[0][0]])
        self.assertEqual(self.sum_prev_sum[0, rows[1][1]].nonzero()
                         .flatten().tolist(), [rows[0][1]])
        self.assertEqual(self.sum_prev_sum[0, rows[2][0]].nonzero()
                         .flatten().tolist(), [rows[1][0]])
        # Turn-1 rows select nothing.
        self.assertEqual(float(self.sum_prev_sum[0, rows[0][0]].sum()), 0.0)
        self.assertEqual(float(self.sum_prev_sum[0, rows[0][1]].sum()), 0.0)

    def test_official_mask_unchanged(self):
        # The official (normalized tril) mask rows sum to 1 exactly where
        # a SUM row has at least one same-slot COMP.
        row_sums = self.sum_attn.sum(-1)
        self.assertTrue(torch.allclose(
            row_sums, (row_sums > 0).float(), rtol=0, atol=1e-6))


class TestGammaRecurrence(unittest.TestCase):
    """The vectorized scan equals a manual per-turn gold recurrence."""

    def test_scan_matches_gold_loop(self):
        torch.manual_seed(7)
        model = make_model()
        attach_gamma(model, hidden=16)
        ids = make_merge_input(3)
        masks = model.model.get_comp_sum_mask(ids)
        comp_mask, sum_mask, sum_attn, sum_prev, sum_prev_sum, sum_count = masks
        gamma = model.model.layers[0].self_attn.gamma
        with torch.no_grad():
            gamma.s.copy_(torch.tensor(0.37))
            B, L = ids.shape
            H, D = 2, model.model.layers[0].self_attn.head_dim
            K = torch.randn(B, H, L, D)
            # Vectorized scan (mirrors the vendored merge block).
            base = torch.matmul(sum_attn.unsqueeze(1), K)
            prev = torch.matmul(sum_prev.unsqueeze(1), K)
            t = sum_count.unsqueeze(1).unsqueeze(-1)
            cur = t * base - (t - 1) * prev
            res = torch.zeros_like(K)
            for t_i in range(2, int(sum_count.max()) + 1):
                res_prev = torch.matmul(sum_prev_sum.unsqueeze(1), res)
                rows_t = (sum_count == t_i).float().unsqueeze(0).unsqueeze(-1)
                res = res + gamma(prev + res_prev, cur, sum_count) * rows_t
            merged = K + res
            # Manual gold: per SUM row in turn order, M_{t-1} carries the
            # previous same-slot SUM's own residual.
            def apply_row(m_prev, cur_row, t_i):
                # [H, D] -> gamma([1, H, 1, D], [1, H, 1, D], [[t_i]])
                x = m_prev.unsqueeze(0).unsqueeze(2)
                y = cur_row.unsqueeze(0).unsqueeze(2)
                return gamma(x, y, torch.tensor([[t_i]]))[0, :, 0]

            gold = torch.zeros_like(K)
            rows = sum_rows(3)
            order = [p for pair in rows for p in pair]  # S0,S1 per turn
            for i in order:
                t_i = int(sum_count[0, i])
                if t_i < 2:
                    continue
                prev_sum_row = sum_prev_sum[0, i].nonzero()
                m_prev = prev[0, :, i] if not len(prev_sum_row) else \
                    prev[0, :, i] + gold[0, :, int(prev_sum_row[0])]
                gold[0, :, i] = apply_row(m_prev, cur[0, :, i], t_i)
            self.assertTrue(torch.allclose(merged, K + gold, rtol=1e-5,
                                           atol=1e-6))


class TestGammaLongRecurrence(unittest.TestCase):
    """Arbitrary k: a 20-turn input (>14, plan L2) stays finite and the
    residual path changes the output once s != 0."""

    def test_twenty_turns(self):
        torch.manual_seed(0)
        official = make_model()
        torch.manual_seed(0)
        ours = make_model()
        attach_gamma(ours, hidden=16)
        ids = make_merge_input(20, u_len=2)
        with torch.no_grad():
            for layer in ours.model.layers:
                layer.self_attn.gamma.s.copy_(torch.tensor(0.05))
            ref = official(input_ids=ids).logits
            out = ours(input_ids=ids).logits
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(torch.allclose(ref, out, rtol=0, atol=1e-9))
        # All 20 turns are counted.
        _, _, _, _, _, sum_count = ours.model.get_comp_sum_mask(ids)
        self.assertEqual(int(sum_count.max()), 20)
        # The pure-mean (s=0) path still matches the official model.
        with torch.no_grad():
            for layer in ours.model.layers:
                layer.self_attn.gamma.s.copy_(torch.tensor(0.0))
            self.assertTrue(torch.allclose(official(input_ids=ids).logits,
                                           ours(input_ids=ids).logits,
                                           rtol=0, atol=0))


class TestRoPESlotPositions(unittest.TestCase):
    """Same-slot COMP tokens share one RoPE basis across turns."""

    def test_slot_positions_consistent(self):
        ids = make_merge_input(4)
        # update_position_ids expects the float mask get_comp_mask builds.
        comp_mask = ((ids == C0) | (ids == C1)).float()
        pos_ids = update_position_ids(comp_mask, COMP_IDS,
                                      torch.ones_like(ids),
                                      type_="skip")
        # The per-slot position basis (the comp-relative offset) is the
        # same on every turn: slot 0 -> +1, slot 1 -> +2 (n_tok=2).
        rel = (comp_mask.long().cumsum(-1) - 1) % len(COMP_IDS) + 1
        c0_rel = rel[0][ids[0] == C0].tolist()
        c1_rel = rel[0][ids[0] == C1].tolist()
        self.assertEqual(len(c0_rel), 4)
        self.assertEqual(len(set(c0_rel)), 1)
        self.assertEqual(len(set(c1_rel)), 1)
        self.assertNotEqual(c0_rel[0], c1_rel[0])  # distinct slots
        # The same offsets appear inside the final position ids.
        c0_pos = pos_ids[0][ids[0] == C0]
        base = (torch.ones_like(ids).float()
                * (1.0 - comp_mask)).long().cumsum(-1) - 1
        self.assertTrue(torch.equal(
            c0_pos, (base[0][ids[0] == C0] + c0_rel[0]).long()))
        self.assertTrue(torch.equal(
            pos_ids[0][ids[0] == C1],
            (base[0][ids[0] == C1] + c1_rel[0]).long()))


class TestGammaClosure(unittest.TestCase):
    """Engineering closure 1-6 (plan L2)."""

    def _tiny_model(self):
        torch.manual_seed(0)
        model = make_model()
        attach_gamma(model, hidden=16)
        return model

    def test_peft_keeps_gamma_trainable(self):  # closure 1
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("peft not installed")
        torch.manual_seed(0)
        model = make_model()
        peft_model = get_peft_model(
            model, LoraConfig(r=4, lora_alpha=8,
                              target_modules=["q_proj", "k_proj",
                                              "v_proj", "o_proj"]))
        attach_gamma(peft_model, hidden=16)  # attach AFTER PEFT wrapping
        # Every Gamma parameter must be trainable: 2 layers x
        # (V.weight, V.bias, U.weight, s).
        all_gamma = [p for n, p in peft_model.named_parameters()
                     if "gamma" in n]
        self.assertEqual(len(all_gamma), 2 * 4)
        self.assertTrue(all(p.requires_grad for p in all_gamma))
        # And the frozen LoRA convention still holds for the backbone.
        frozen = [n for n, p in peft_model.named_parameters()
                  if not p.requires_grad and "lora" not in n
                  and "gamma" not in n]
        self.assertTrue(any("q_proj" in n for n in frozen))

    def test_optimizer_contains_gamma(self):  # closure 2
        model = self._tiny_model()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=1e-3)
        gamma_ids = {id(p) for n, p in model.named_parameters()
                     if "gamma" in n}
        self.assertTrue(gamma_ids)
        opt_ids = {id(p) for group in opt.param_groups
                   for p in group["params"]}
        self.assertTrue(gamma_ids <= opt_ids)

    def test_checkpoint_roundtrip(self):  # closures 3 + 4
        model = self._tiny_model()
        ckpt = CheckpointManager("_tmp_ccm_gamma_ckpt.pt")
        ckpt.save(model_components={"ccm": model},
                  optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
                  epoch=0, next_batch=0, global_step=0, best_score=0.0,
                  best_epoch=0, bad_rounds=0, train_state={})
        torch.manual_seed(0)
        fresh = make_model()
        attach_gamma(fresh, hidden=16)
        ckpt.load(model_components={"ccm": fresh},
                  optimizer=torch.optim.Adam(fresh.parameters(), lr=1e-3),
                  device="cpu")
        for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                      fresh.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2), n1)
        import os
        os.remove("_tmp_ccm_gamma_ckpt.pt")

    def test_paired_seed_hash(self):  # closure 5
        model = self._tiny_model()
        h0 = paired_seed_hash(7, model)
        self.assertEqual(h0, paired_seed_hash(7, model))
        self.assertNotEqual(h0, paired_seed_hash(8, model))  # base seed
        with torch.no_grad():
            g = model.model.layers[0].self_attn.gamma
            g.s.copy_(torch.tensor(0.5))
        self.assertNotEqual(h0, paired_seed_hash(7, model))  # Gamma weight
        with torch.no_grad():
            g.s.copy_(torch.tensor(0.0))
        self.assertEqual(h0, paired_seed_hash(7, model))
        with torch.no_grad():
            model.model.embed_tokens.weight[C0] += 0.1  # COMP embed row
        self.assertNotEqual(h0, paired_seed_hash(7, model))

    def test_first_steps_move_gamma(self):  # closure 6
        model = self._tiny_model()
        ids = make_merge_input(3)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=0.1)
        g = model.model.layers[0].self_attn.gamma
        u0 = g.U.weight.detach().clone()
        v0 = g.V.weight.detach().clone()
        for step in range(2):
            opt.zero_grad()
            loss = model(input_ids=ids).logits.sum()
            loss.backward()
            opt.step()
            if step == 0:
                # Step 1: only the gate s can move (U/V grads are ~ s).
                self.assertNotEqual(g.s.detach().item(), 0.0)
                self.assertTrue(torch.equal(g.U.weight.detach(), u0))
                self.assertTrue(torch.equal(g.V.weight.detach(), v0))
            else:
                # Step 2: s != 0 opens the U/V gradient path.
                self.assertFalse(torch.equal(g.U.weight.detach(), u0))
                self.assertFalse(torch.equal(g.V.weight.detach(), v0))

    def test_parameter_budget(self):
        model = self._tiny_model()
        n = sum(g.n_params() for l in model.model.layers
                for g in [l.self_attn.gamma])
        # Tiny model: hidden 32 / 2 heads -> head_dim 16.
        self.assertEqual(model.model.layers[0].self_attn.gamma.head_dim, 16)
        self.assertLess(n, 1_000_000)
        # 7B head_dim=128, hidden=64: 32 x ~25.7k = ~821k < 1M (plan L2).
        full = GammaResidual(128)
        self.assertLess(32 * full.n_params(), 1_000_000)


if __name__ == "__main__":
    unittest.main()
