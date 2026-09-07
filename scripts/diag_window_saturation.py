#!/usr/bin/env python3
"""CCA saturation audit (review round 5/6): dialogue-grouped shuffle on a
fixed-L=13 VALIDATION cohort window, evaluated with the conditional
checkpoint @250.

The training-run window_diag.jsonl does not exist for the running
processes (they predate the persistence patch), so this script rebuilds
a window at EVALUATION time: 51 dialogues x 2 horizon rows, conditional
P construction, z_v extracted from the SUM rows exactly as training
does. Then:

    J_real  = Ky-Fan score on the true pairing
    J_shuff = dialogue-grouped permutation ((p1,p2) move together),
              200 draws

If the real-vs-shuffle gap is ~0, the weak PPL advantage cannot be
explained as a conditional future signal.
"""
import json
import os
import sys
import types
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/src")
sys.path.insert(0, "/root/autodl-tmp/third_party/ccm")
os.environ["DIALOG_MIRROR"] = "/root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import train_ccm as tc  # noqa: E402
from rpbe.loss import _score_from_covs, _covs  # noqa: E402
from rpbe.llm.utterance_embed import UtteranceEmbed  # noqa: E402
from rpbe.llm.dialogue_records import Llmmaps, DialogueCutBuilder, MEM_TAU  # noqa: E402
from rpbe.hosts.ccm.adapter import CCMHostAdapter  # noqa: E402

BASE = "/root/autodl-tmp"
DEVICE = torch.device("cuda", 0)
CKPT = BASE + "/outputs/ccm_pilot/seed0_CCM_Diag_ours_cond_e50/checkpoint_step250.pt"
L = 13
N_SHUFFLES = 200


def build_stuff():
    tok = tc.build_tokenizer(types.SimpleNamespace(
        model_name_or_path=BASE + "/llama-7b-hf"))
    from src.arguments import CompressionArguments
    from src.data.dialogue.data import DialogueDataset
    from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=tc.N_TOK,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    ds = DialogueDataset(tok, comp_token=tok.comp_token_id,
                         online=True, add_comp_token=True, clean_split=True)
    collator = DataCollatorForDialogue_LLAMA(
        dialog=ds, tokenizer=tok, comp_args=comp_args,
        comp_token=tok.comp_token_id, sum_token=tok.sum_token_id,
        padding="left", pad_token=tok.pad_token_id, label_pad_token_id=-100)
    dialogs = [(i, d) for i, d in enumerate(ds.valset["dialog"])
               if len(d) >= 15]
    return tok, collator, dialogs


def build_model_and_rpbe(tok):
    from src.model import load_lora_weight
    args = types.SimpleNamespace(arm="ours",
                                 model_name_or_path=BASE + "/llama-7b-hf",
                                 relative_embedding="skip",
                                 lora_r=8, gamma_hidden=64)
    model = tc.build_model(args, DEVICE)
    load_lora_weight(BASE + "/result/dialog/llama-7b-no", model, merge=True)
    model = tc.wrap_lora(model, args.lora_r)
    model.update_comp_token([32000 + k for k in range(tc.N_TOK)],
                            [32000 + tc.N_TOK + k for k in range(tc.N_TOK)])
    gammas = tc.attach_gamma(model, hidden=args.gamma_hidden)
    for g in gammas:
        g.half()
    dummy = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    payload = tc.load_trainable(CKPT, model, dummy, DEVICE)
    n = 2 * tc.N_TOK
    model.get_input_embeddings().weight[-n:] = \
        payload["new_token_rows"]["input_embed_rows"].to(DEVICE)
    model.lm_head.weight[-n:] = \
        payload["new_token_rows"]["lm_head_rows"].to(DEVICE)
    model.eval()
    cfg = model.config
    adapter = CCMHostAdapter(model, n_layers=cfg.num_hidden_layers,
                             n_heads=cfg.num_attention_heads,
                             head_dim=cfg.hidden_size // cfg.num_attention_heads,
                             z_dim=128, seed=0)
    maps = Llmmaps(d_chi=64, d_phi=32, m=64, seed=0).to(DEVICE)
    utter_embed = UtteranceEmbed(cfg.hidden_size, d_chi=64, seed=0,
                                 combine_dim=1).to(DEVICE)
    phi_embed = UtteranceEmbed(cfg.hidden_size, d_chi=32,
                               seed=100).to(DEVICE)
    builder = DialogueCutBuilder(maps, seed=0, z_dim=128)
    return model, adapter, maps, utter_embed, phi_embed, builder


def main():
    tok, collator, dialogs = build_stuff()
    model, adapter, maps, utter_embed, phi_embed, builder = \
        build_model_and_rpbe(tok)
    embed_tokens = model.get_input_embeddings()
    comp_ids = [32000, 32001]
    sum_ids = [32002, 32003]
    zs, ps, cut_ids = [], [], []
    with torch.no_grad():
        for di, (dname, dialog) in enumerate(dialogs):
            item = {"dialog": dialog[13 - L:13]
                    + [dialog[13], dialog[14]],
                    "is_train": False, "act": []}
            batch = collator([item])
            # forward to populate the memory/SUM states
            model(input_ids=batch["input_ids"].to(DEVICE),
                  attention_mask=batch["attention_mask"].to(DEVICE),
                  attention_mask_comp=batch["attention_mask_comp"].to(DEVICE))
            metas = tc.parse_meta(batch, comp_ids, sum_ids, di)
            m = metas[0]
            if not m.get("ok") or m["k"] < 4:
                continue
            v = m["k"] - 3
            s0 = m["blocks"][v][1]
            z = adapter.extract_z(torch.tensor(
                [[s0, s0 + 1]], dtype=torch.long, device=DEVICE))[0]
            ids = batch["input_ids"][0]
            labs = batch["labels"][0]
            spans = m["utterance_spans"]
            u1 = ids[spans[v + 1][0]:spans[v + 1][1]].unsqueeze(0).to(DEVICE)
            u2 = ids[spans[v + 2][0]:spans[v + 2][1]].unsqueeze(0).to(DEVICE)
            valid_pos = (labs != -100).nonzero(as_tuple=False).flatten()
            uk_pos = valid_pos[:-1] if len(valid_pos) > 1 else valid_pos
            chi1 = utter_embed(embed_tokens, u1, tag=0)
            phi1 = phi_embed(embed_tokens, u2, tag=0)
            chi2 = utter_embed.combine(embed_tokens, u1, u2, tag=1)
            phi2 = phi_embed(embed_tokens, ids[uk_pos].unsqueeze(0).to(DEVICE),
                             tag=1)
            p1 = maps.pv(chi1[0], phi1[0])
            p2 = maps.pv(chi2[0], phi2[0])
            for h, p in ((1, p1), (2, p2)):
                zs.append(z.detach().float())
                ps.append(p.detach().float())
                cut_ids.append((di, h))
    z_all = torch.stack(zs)
    p_all = torch.stack(ps)
    N = z_all.shape[0]
    print("window rows: {} ({} cuts)".format(N, N // 2), flush=True)
    # J_real
    w = (torch.ones(N) / N).to(z_all.device)
    mu_z = z_all.mean(0, keepdim=True)
    mu_p = p_all.mean(0, keepdim=True)
    zc = z_all.double() - mu_z.double()
    pc = p_all.double() - mu_p.double()
    czz, cpp, czp, _ = _covs(zc, pc, N - 1, w=w)
    j_real, _ = _score_from_covs(czz, czp, cpp, 1e-4)
    # dialogue-grouped shuffle
    groups = defaultdict(list)
    for i, cid in enumerate(cut_ids):
        groups[cid[0]].append(i)
    gkeys = list(groups.keys())
    rng = np.random.default_rng(0)
    j_shuff = []
    for _ in range(N_SHUFFLES):
        perm = rng.permutation(len(gkeys))
        new_p = p_all.clone()
        for gi, gk in enumerate(gkeys):
            target = groups[gkeys[perm[gi]]]
            for a, b in zip(groups[gk], target):
                new_p[a] = p_all[b]
        pc_s = new_p.double() - mu_p.double()
        _, _, czp_s, _ = _covs(zc, pc_s, N - 1, w=w)
        js, _ = _score_from_covs(czz, czp_s, cpp, 1e-4)
        if js is not None:
            j_shuff.append(float(js))
    j_shuff = np.array(j_shuff)
    gap = float(j_real) - float(j_shuff.mean())
    print(json.dumps({
        "N_rows": N, "N_cuts": N // 2, "L": L,
        "J_real": float(j_real),
        "J_shuff_mean": float(j_shuff.mean()),
        "J_shuff_std": float(j_shuff.std()),
        "gap": gap,
        "gap_over_m": gap / 64.0,
        "shuff_frac_below_real": float((j_shuff < float(j_real)).mean()),
        "ci95_shuff": [float(np.percentile(j_shuff, 2.5)),
                       float(np.percentile(j_shuff, 97.5))],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
