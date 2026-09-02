"""L6 acceptance tests for the two-pass replay protocol (plan v2 L6).

New five:
  1. RPBE / Gamma / LoRA gradients all nonzero on one replay step.
  2. Checkpoint roundtrip (covered by test_ccm_gamma closure 3/4; the
     window/builder state is pure Python counters replayed by protocol).
  3. Pass-1 / pass-2 state and RNG replay: with zero dropout the two
     passes are bit-identical; the trainer asserts zero dropout.
  4. Window non-degeneracy: an all-zero C_ZZ close is reported, never
     silently trained on.
  5. eager/compile consistency: the three arms run with torch.compile
     disabled (frozen_method.json), so the check is N/A and recorded.
"""

import sys
import unittest
from pathlib import Path

_CCM_DIR = Path(__file__).resolve().parents[1] / "third_party" / "ccm"
if str(_CCM_DIR) not in sys.path:
    sys.path.insert(0, str(_CCM_DIR))

import numpy as np
import torch

from transformers.models.llama.configuration_llama import LlamaConfig

from src.arch.ccm_llama import LlamaForCausalLM_CCM
from rpbe.hosts.ccm.adapter import CCMHostAdapter
from rpbe.hosts.ccm.ccm_patch import attach_gamma, wrap_lora
from rpbe.llm.dialogue_records import DialogueCutBuilder, Llmmaps, MEM_TAU
from rpbe.llm.utterance_embed import UtteranceEmbed
from rpbe.loss import KFMomentWindow
from rpbe.training.checkpoint import _restore_rng, _rng_state

C0, C1, S0, S1 = 32000, 32001, 32002, 32003
COMP_IDS = [C0, C1]
SUM_IDS = [S0, S1]


def make_config(dropout=0.0):
    cfg = LlamaConfig(vocab_size=32004, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2,
                      max_position_embeddings=512,
                      attention_dropout=dropout, hidden_dropout=dropout)
    cfg.comp_relative_embedding = "base"
    return cfg


def make_model(dropout=0.0):
    model = LlamaForCausalLM_CCM(make_config(dropout))
    model.update_comp_token(COMP_IDS, SUM_IDS)
    return model


def make_batch(n_turns=5, u_len=3, u_base=100):
    """A collator-shaped batch: input ids with blocks + labels
    (completion = the last u_len tokens, prompt = everything before)."""
    seq = [1]
    for t in range(n_turns):
        seq += [u_base + t] * u_len
        seq += [C0, C1, S0, S1]
    ids = torch.tensor([seq])
    labels = torch.full_like(ids, -100)
    labels[0, -u_len:] = ids[0, -u_len:]
    return {"input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": labels}


def build_rpbe_kit(model, z_dim=16, seed=5, min_abs=4):
    attach_gamma(model, hidden=16)
    cfg = model.model.config
    adapter = CCMHostAdapter(model, n_layers=cfg.num_hidden_layers,
                             n_heads=cfg.num_attention_heads,
                             head_dim=cfg.hidden_size
                             // cfg.num_attention_heads,
                             z_dim=z_dim, seed=seed)
    maps = Llmmaps(d_chi=8, d_phi=8, m=4, seed=seed)
    builder = DialogueCutBuilder(maps, z_dim=z_dim, seed=seed)
    utter = UtteranceEmbed(hidden_dim=cfg.hidden_size, d_chi=8, seed=seed)
    window = KFMomentWindow({MEM_TAU: z_dim}, min_ratio=2.0, min_abs=min_abs,
                            eps=1e-3, fixed_maps=maps, strict=True,
                            autoclose=False)
    return adapter, maps, builder, utter, window


class TestPassReplay(unittest.TestCase):
    """New item 3: pass-1 / pass-2 RNG and state replay."""

    def test_two_passes_bit_identical(self):
        torch.manual_seed(0)
        model = make_model(dropout=0.0)
        batch = make_batch()
        state = _rng_state()
        with torch.no_grad():
            out1 = model(input_ids=batch["input_ids"],
                         labels=batch["labels"]).logits
        _restore_rng(state)
        with torch.no_grad():
            out2 = model(input_ids=batch["input_ids"],
                         labels=batch["labels"]).logits
        self.assertTrue(torch.allclose(out1, out2, rtol=0, atol=0))

    def test_forward_consumes_no_rng(self):
        # The vendored forward has no dropout path (the official Llama
        # dropout is not wired into the custom attention), so pass-1 and
        # pass-2 replays are identical by construction.  Verify the RNG
        # state is untouched by a forward — the replay protocol's
        # precondition (train_ccm still asserts zero dropout as a guard
        # against future edits).
        torch.manual_seed(0)
        model = make_model(dropout=0.0)
        model.train()
        batch = make_batch()
        state = _rng_state()
        model(input_ids=batch["input_ids"], labels=batch["labels"])
        after = _rng_state()

        def rng_equal(a, b):
            if a["python"] != b["python"]:
                return False
            for x, y in zip(a["numpy"], b["numpy"]):
                if isinstance(x, np.ndarray):
                    if not np.array_equal(x, y):
                        return False
                elif x != y:
                    return False
            return torch.equal(a["torch"], b["torch"])

        self.assertTrue(rng_equal(state, after))

    def test_zero_dropout_asserted_in_trainer(self):
        import ast
        src = Path(__file__).resolve().parents[1] / "scripts" \
            / "train_ccm.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        has_assert = any(
            isinstance(n, ast.Assert) and
            "dropout" in ast.dump(n).lower()
            for n in ast.walk(tree))
        self.assertTrue(has_assert)


class TestWindowNonDegenerate(unittest.TestCase):
    """New item 4: an all-zero C_ZZ close is reported, never trained."""

    def test_zero_z_close_fails_visibly(self):
        torch.manual_seed(1)
        maps = Llmmaps(d_chi=8, d_phi=8, m=4, seed=3)
        builder = DialogueCutBuilder(maps, z_dim=8)
        win = KFMomentWindow({"mem": 8}, min_ratio=2.0, min_abs=4,
                             eps=1e-3, fixed_maps=maps, strict=False,
                             autoclose=False)
        from rpbe.llm.dialogue_records import DialogueMeta
        z = torch.zeros(8)
        for s in range(10):  # threshold = max(2*4, 4) = 8
            meta = DialogueMeta(sample_id=s, k=5,
                                sum_positions=[(0, 1)] * 5,
                                utterance_spans=[(0, 1)] * 5)
            win.add(builder.build(meta, z, torch.randn(8), torch.randn(8)))
        closed, plan, diag = win.close_replay()
        self.assertEqual(closed.get("mem"), 0.0)  # differentiable zero
        self.assertIsNotNone(diag["mem"]["failed"])  # visible, not silent
        self.assertEqual(plan["mem"]["by_batch"], [[]])


class TestReplayGradients(unittest.TestCase):
    """New item 1: RPBE / Gamma / LoRA gradients all nonzero on one step."""

    def test_all_components_receive_gradient(self):
        try:
            import peft  # noqa: F401
        except ImportError:
            self.skipTest("peft not installed")
        torch.manual_seed(2)
        model = make_model()
        model = wrap_lora(model, r=4)
        adapter, maps, builder, utter, window = build_rpbe_kit(
            model, z_dim=16, seed=5, min_abs=2)
        # Pass 1: collect 10 cuts (k=6 -> v=3 per sample); threshold =
        # max(2*min(16,4), 2) = 8.  The pass-1 record keeps
        # [(meta, occurrence_id)] per batch; pass 2 only re-extracts z
        # (L6.5 review: no builder/chi/p in pass 2).
        from scripts.train_ccm import (batch_surrogate, collect_replay_z,
                                       collect_rows, parse_meta,
                                       task_ce_shifted)
        batches = [make_batch(u_base=100 + 10 * s) for s in range(10)]
        window_start = {"rng": _rng_state()}
        cut_records = []
        for s, batch in enumerate(batches):
            state = _rng_state()
            with torch.no_grad():
                model(input_ids=batch["input_ids"],
                      labels=batch["labels"])
                metas = parse_meta(batch, COMP_IDS, SUM_IDS,
                                   sample_id_global=s)
                bc = []
                for meta in metas:
                    rows = collect_rows(meta, adapter, builder, utter,
                                        model.get_input_embeddings(),
                                        batch, torch.device("cpu"))
                    if rows:
                        window.add(rows)
                        bc.append((meta, rows[0].occurrence_id))
                adapter.clear()
            cut_records.append(bc)
            _restore_rng(state)
        closed, plan, diag = window.close_replay()
        self.assertIn("mem", closed)
        resume_rng = _rng_state()  # P0-1: real post-pass-1 stream state
        _restore_rng(window_start["rng"])
        # Pass 2: task + surrogate backward, one optimizer step.  The
        # window plan is consumed through OCCURRENCE IDS (review P0-1:
        # by_batch is aligned to the cut-producing batches only); the
        # surrogate follows the real training scale — task divided by
        # the window size, aux NOT divided (review P0-3).
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=1e-3)
        g_by_oid = plan["mem"]["by_oid"]
        opt.zero_grad(set_to_none=True)
        n_aux_terms = 0
        for s, batch in enumerate(batches):
            model.train()
            out = model(input_ids=batch["input_ids"], labels=batch["labels"])
            z_by_oid = {oid: collect_replay_z(meta, adapter,
                                              torch.device("cpu"))
                        for meta, oid in cut_records[s]}
            adapter.clear()
            task, _ = task_ce_shifted(out, batch["labels"],
                                      torch.device("cpu"))
            terms = [(oid, g_by_oid[oid]) for _, oid in cut_records[s]
                     if oid in g_by_oid]
            n_aux_terms += len(terms)
            aux, _ = batch_surrogate(z_by_oid, terms, 1.0,
                                     torch.device("cpu"))
            (task / len(batches) + aux).backward()
        # Every effective cut in the window carried a gradient and was
        # replayed exactly once (aux_terms == effective cuts).
        self.assertEqual(n_aux_terms, len(batches))
        _restore_rng(resume_rng)
        # Every component type carries a nonzero gradient.
        lora_grads = [p.grad for n, p in model.named_parameters()
                      if "lora" in n and p.requires_grad]
        gamma_grads = [p.grad for n, p in model.named_parameters()
                       if "gamma" in n]
        self.assertTrue(lora_grads and all(g is not None for g in lora_grads))
        self.assertTrue(gamma_grads and all(g is not None for g in gamma_grads))
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in lora_grads))
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in gamma_grads))

    def test_compile_consistency_na(self):
        # New item 5: the three arms run with torch.compile disabled
        # (configs/ccm/frozen_method.json), so eager/compile equivalence
        # is recorded N/A for this experiment.
        import json
        frozen = json.load(open(Path(__file__).resolve().parents[1]
                                / "configs" / "ccm"
                                / "frozen_method.json", encoding="utf-8"))
        self.assertEqual(frozen["backbone"]["torch_compile"],
                         "disabled on all three arms")


class TestInterleavedK3Alignment(unittest.TestCase):
    """Review P0-1 regression: k < 4 microbatches interleaved into the
    window must not shift the replay plan.

    The legacy consumption aligned ``plan[tau]["by_batch"]`` — which is
    indexed by CUT-PRODUCING batches only — with the full pending list,
    so the first interleaved k=3 microbatch shifted every later cut off
    its gradient (measured on the gate runs: aux_terms=11 over 7 windows
    instead of 896; ours silently decaying toward gamma_task_only).
    """

    # make_batch(n_turns) -> k = n_turns + 1 (one COMP/SUM block per
    # turn): n_turns=2 -> k=3 (no cut), n_turns=5 -> k=6 (cut at v=3).
    # Threshold with z_dim=16/m=4/min_abs=4 = max(2*4, 4) = 8 trees.

    def _grad_snapshot(self, turns_seq, seed):
        """One full two-pass window over ``turns_seq`` = [(n_turns,
        u_base, sample_id), ...]; returns (grads_by_name, n_aux_terms)
        for an AUX-ONLY backward at lambda=1.  sample_id is caller-
        controlled so two streams can share identical cut identities
        while interleaving k=3 batches differently."""
        from scripts.train_ccm import (batch_surrogate, collect_replay_z,
                                       collect_rows, parse_meta)
        torch.manual_seed(seed)
        model = make_model()
        adapter, maps, builder, utter, window = build_rpbe_kit(
            model, z_dim=16, seed=5, min_abs=4)
        batches = [make_batch(n_turns=t, u_base=base)
                   for t, base, _sid in turns_seq]
        cut_records = []
        n_cuts = 0
        for idx, batch in enumerate(batches):
            state = _rng_state()
            with torch.no_grad():
                model(input_ids=batch["input_ids"],
                      labels=batch["labels"])
                metas = parse_meta(batch, COMP_IDS, SUM_IDS,
                                   sample_id_global=turns_seq[idx][2])
                bc = []
                for meta in metas:
                    if not (meta["ok"] and meta["k"] >= 4):
                        continue
                    rows = collect_rows(meta, adapter, builder, utter,
                                        model.get_input_embeddings(),
                                        batch, torch.device("cpu"))
                    if rows:
                        window.add(rows)
                        bc.append((meta, rows[0].occurrence_id))
                        n_cuts += 1
                adapter.clear()
            cut_records.append(bc)
            _restore_rng(state)
        closed, plan, diag = window.close_replay()
        g_by_oid = plan["mem"]["by_oid"]
        self.assertTrue(g_by_oid, "window did not close "
                        "(closed={} trees={})".format(
                            closed, diag["mem"].get("M_unique_trees")))
        params = [p for p in model.parameters() if p.requires_grad]
        for p in params:
            p.grad = None
        n_terms = 0
        for i, batch in enumerate(batches):
            model.train()
            out = model(input_ids=batch["input_ids"],
                        labels=batch["labels"])
            z_by_oid = {oid: collect_replay_z(meta, adapter,
                                              torch.device("cpu"))
                        for meta, oid in cut_records[i]}
            adapter.clear()
            terms = [(oid, g_by_oid[oid]) for _, oid in cut_records[i]
                     if oid in g_by_oid]
            n_terms += len(terms)
            aux, n_aux = batch_surrogate(z_by_oid, terms, 1.0,
                                         torch.device("cpu"))
            if n_aux:  # no-terms batches (k < 4) contribute a bare zero
                aux.backward()
        grads = {n: p.grad.detach().clone()
                 for n, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None}
        return grads, n_terms, n_cuts

    def test_aux_terms_equals_effective_cuts_interleaved(self):
        # k = [6, 6, 3, 6, 6, 3, ...]: 8 cuts + 4 k=3 interleaves.
        seq = []
        cut_sid = 0
        for s in range(8):
            seq.append((5, 100 + 10 * s, cut_sid))
            cut_sid += 1
            if s % 2 == 1:
                seq.append((2, 900 + s, 100 + s))
        grads, n_terms, n_cuts = self._grad_snapshot(seq, seed=11)
        self.assertEqual(n_cuts, 8)
        self.assertEqual(n_terms, n_cuts,   # aux_terms == effective cuts
                         "interleaved k=3 batches must not drop terms")
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads.values()),
                        "aux gradient must be nonzero at lambda=1")

    def test_k3_insertion_does_not_change_aux_grads(self):
        # The SAME 8 cuts, once alone and once with k=3 batches inserted
        # between them.  Cut identities (sample_id) are identical, so the
        # window rows are identical and the RPBE gradient must be
        # bit-comparable — inserting or deleting k=3 batches changes
        # nothing about the auxiliary loss.
        alone = [(5, 100 + 10 * s, s) for s in range(8)]
        interleaved = []
        cut_sid = 0
        for s in range(8):
            interleaved.append((5, 100 + 10 * s, cut_sid))
            cut_sid += 1
            if s % 2 == 1:
                interleaved.append((2, 900 + s, 100 + s))
        ga, n_a, _c = self._grad_snapshot(alone, seed=11)
        gb, n_b, _c = self._grad_snapshot(interleaved, seed=11)
        self.assertEqual(n_a, 8)
        self.assertEqual(n_b, 8)
        self.assertEqual(set(ga), set(gb))
        for name in ga:
            self.assertTrue(torch.allclose(ga[name], gb[name],
                                           rtol=1e-4, atol=1e-7),
                            "aux gradient of {} changed when k=3 batches "
                            "were interleaved".format(name))


if __name__ == "__main__":
    unittest.main()
