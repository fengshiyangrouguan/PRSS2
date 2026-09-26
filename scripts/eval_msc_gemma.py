#!/usr/bin/env python3
"""gemma MSC canonical evaluation — mirrors llama train_msc.py --mode eval.

Arms: full (raw gemma-4-E4B text base, no compression) + merge (official
msc merge adapter ckpt over the gemma_MSC_no foundation).  Protocol:
exchange_instances (off-by-one FIXED 2026-09-26), opening-only,
sessions 2..5, max-episodes 250 (Random(1234) fixed subset), per-row
shifted CE -> token-weighted PPL, EOS included in labels (output_ids
carry eos), batch 4.  Output json identical in shape to the llama
msc_evals json (S2..S5 + overall).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent / "src"),
          str(HERE.parent / "third_party" / "ccm")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import train_ccm as tc
import train_msc_merge_gemma as tmg

BASE = "/root/autodl-tmp"
MSC_DATA_DIR = BASE + "/third_party/ccm/dataset/msc"


def build_tokenizer():
    import types as _t
    args = _t.SimpleNamespace(host="gemma4",
                              model_name_or_path=BASE + "/gemma-4-E4B",
                              relative_embedding="skip", foundation="")
    return tc.build_tokenizer(args)


def build_full(device):
    import types as _t
    args = _t.SimpleNamespace(host="gemma4",
                              model_name_or_path=BASE + "/gemma-4-E4B",
                              relative_embedding="skip", foundation="",
                              official_host=False)
    model = tc.build_model(args, device)
    model.eval()
    return model


def build_merge(device, foundation, adapter_ckpt):
    import types as _t
    a = _t.SimpleNamespace(host="gemma4",
                           model_name_or_path=BASE + "/gemma-4-E4B",
                           relative_embedding="skip",
                           foundation=foundation,
                           lora_r=8, lora_dropout=0.05)
    _, model = tmg.build_model(a, device)
    ck = torch.load(adapter_ckpt, map_location=device, weights_only=False)
    _m, _u = model.load_state_dict(ck["model"], strict=False)
    if _u:
        raise RuntimeError("unexpected keys: {}".format(sorted(_u)[:5]))
    print("[msc-gemma-eval] merge adapter step={} loaded: {} missing "
          "keys".format(ck.get("step"), len(_m)), flush=True)
    model.eval()
    return model


def build_ours(device, foundation, adapter_ckpt, rpbe_ckpt,
               gamma_hidden):
    """Ours arm: merge host + Gamma + the R10 Stage-2 trainable state."""
    from rpbe.hosts.ccm.ccm_patch import attach_gamma
    model = build_merge(device, foundation, adapter_ckpt)
    attach_gamma(model, hidden=gamma_hidden)
    ck = torch.load(rpbe_ckpt, map_location=device, weights_only=False)
    _m, _u = model.load_state_dict(ck["model"], strict=False)
    if _u:
        raise RuntimeError("unexpected keys: {}".format(sorted(_u)[:5]))
    print("[msc-gemma-eval] ours ckpt step={} loaded: {} missing keys"
          .format(ck.get("step"), len(_m)), flush=True)
    model.eval()
    return model


def eval_full_collate(tok, rows, ds_full):
    inputs, labels = [], []
    for r in rows:
        inst = ds_full.sample_dialog(r)
        full = inst["input_ids"] + inst["output_ids"]
        inputs.append(full)
        labels.append([-100] * len(inst["input_ids"]) + inst["output_ids"])
    pad = tok.pad_token_id
    L = max(len(x) for x in inputs)
    return {
        "input_ids": torch.tensor(
            [[pad] * (L - len(x)) + x for x in inputs], dtype=torch.long),
        "labels": torch.tensor(
            [[-100] * (L - len(x)) + x for x in labels], dtype=torch.long),
        "attention_mask": torch.tensor(
            [[0] * (L - len(x)) + [1] * len(x) for x in inputs],
            dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--eval-sessions", default="2,3,4,5")
    ap.add_argument("--opening-only", action="store_true", default=True)
    ap.add_argument("--max-episodes", type=int, default=250)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--foundation",
                    default=BASE + "/result/msc/gemma_MSC_no/final.pt")
    ap.add_argument("--msc-adapter", default="")
    ap.add_argument("--arm", default="", choices=["full", "merge", "ours"],
                    help="empty = full if no msc-adapter else merge")
    ap.add_argument("--rpbe-ckpt", default="",
                    help="R10 Stage-2 ckpt (ckpt_stepN.pt) for the "
                         "ours arm — attach Gamma and load the "
                         "trainable state over the merge host")
    ap.add_argument("--gamma-hidden", type=int, default=64)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    device = torch.device("cuda:{}".format(a.gpu))
    Path(a.output).mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer()
    from src.data.dialogue.data_msc import DialogueDataset
    from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA
    from src.arguments import CompressionArguments

    ds_full = DialogueDataset(tok, comp_token=[], online=True,
                              add_comp_token=False, eval_source="val",
                              max_length=2048, msc_dir=MSC_DATA_DIR)
    # merge-arm collator needs the comp-token-carrying dataset view
    # (the llama host uses its add_comp_token=True ds for the collator;
    # with the full-arm ds the batch has no comp ids and the gemma
    # model asserts comp_mask is None -> sum_attn_mask is not None).
    ds_ccm = DialogueDataset(tok, comp_token=tok.comp_token_id,
                             online=True, add_comp_token=True,
                             eval_source="val", max_length=2048,
                             msc_dir=MSC_DATA_DIR)
    split = "valid" if a.eval_split == "val" else "test"
    sessions = tuple(int(s) for s in a.eval_sessions.split(","))
    instances = list(ds_full.exchange_instances(split, sessions=sessions))
    if a.opening_only:
        instances = [r for r in instances if r["is_opening"]]
    if a.max_episodes > 0:
        rng = random.Random(1234)
        eids = sorted({r["orig_id"] for r in instances})
        keep = set(rng.sample(eids, min(a.max_episodes, len(eids))))
        instances = [r for r in instances if r["orig_id"] in keep]
    print("[msc-gemma-eval] {} exchanges on {} (opening_only={} "
          "sessions={} max_episodes={})".format(
              len(instances), split, a.opening_only, sessions,
              a.max_episodes), flush=True)

    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=tc.N_TOK,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    collator = DataCollatorForDialogue_LLAMA(
        dialog=ds_ccm, tokenizer=tok, comp_args=comp_args,
        comp_token=tok.comp_token_id, sum_token=tok.sum_token_id,
        padding="left", pad_token=tok.pad_token_id,
        label_pad_token_id=-100)

    if a.arm:
        arms = [a.arm]
    else:
        arms = ["full"] if not a.msc_adapter else ["merge"]
    for arm in arms:
        if arm == "full":
            model = build_full(device)
        elif arm == "ours":
            assert a.msc_adapter and a.rpbe_ckpt, (
                "--msc-adapter and --rpbe-ckpt required for the ours arm")
            model = build_ours(device, a.foundation, a.msc_adapter,
                               a.rpbe_ckpt, a.gamma_hidden)
        else:
            model = build_merge(device, a.foundation, a.msc_adapter)
        print("[msc-gemma-eval] arm={} ready".format(arm), flush=True)

        acc = {}
        for s in range(2, 6):
            acc[s] = {"opening_sum": 0.0, "opening_dlg": 0,
                      "all_sum": 0.0, "all_dlg": 0}
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(instances), a.eval_batch):
                rows = instances[i:i + a.eval_batch]
                metas = [(int(r["session"]), int(r["is_opening"]))
                         for r in rows]
                if arm == "full":
                    batch = eval_full_collate(tok, rows, ds_full)
                    out_f = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
                else:
                    batch = collator(rows)
                    out_f = tc.run_forward(model, batch, device, False)
                # Protocol A: EOS excluded — mask the final EOS label
                # of every row (labels are left-padded; the last valid
                # position of each row is the EOS token).
                labs = batch["labels"]
                last_valid = (labs != -100).sum(-1) - 1
                labs = labs.clone()
                labs[torch.arange(labs.shape[0]), last_valid] = -100
                row_sum, row_n = tc.task_ce_rows(out_f, labs, device)
                # dialogue-mean: per-dialogue mean NLL, then the mean
                # across dialogues (NOT token-pooled).
                row_mean = row_sum / row_n.clamp(min=1)
                for r in range(row_mean.shape[0]):
                    s, o = metas[r]
                    acc[s]["all_sum"] += float(row_mean[r])
                    acc[s]["all_dlg"] += 1
                    if o:
                        acc[s]["opening_sum"] += float(row_mean[r])
                        acc[s]["opening_dlg"] += 1
                if i % (a.eval_batch * 50) == 0 and i:
                    print("[msc-gemma-eval] {} {}/{} {:.0f}s".format(
                        arm, i, len(instances), time.time() - t0),
                        flush=True)
        result = {"arm": arm, "split": split,
                  "n_instances": len(instances), "sessions": {}}
        for s in range(2, 6):
            result["sessions"]["S{}".format(s)] = {
                "opening_ppl": float(torch.exp(torch.tensor(
                    acc[s]["opening_sum"]
                    / max(acc[s]["opening_dlg"], 1)))),
                "opening_dialogues": acc[s]["opening_dlg"],
                "all_ppl": float(torch.exp(torch.tensor(
                    acc[s]["all_sum"] / max(acc[s]["all_dlg"], 1)))),
                "all_dialogues": acc[s]["all_dlg"],
            }
        result["overall_opening_ppl"] = float(torch.exp(torch.tensor(
            sum(acc[s]["opening_sum"] for s in range(2, 6))
            / max(sum(acc[s]["opening_dlg"] for s in range(2, 6)), 1))))
        result["overall_all_ppl"] = float(torch.exp(torch.tensor(
            sum(acc[s]["all_sum"] for s in range(2, 6))
            / max(sum(acc[s]["all_dlg"] for s in range(2, 6)), 1))))
        json.dump(result,
                  open(Path(a.output) / "eval_{}_{}.json".format(arm,
                                                                 split),
                       "w"), indent=2)
        print("[msc-gemma-eval] arm={} overall_opening_ppl={:.4f} "
              "({:.0f}s)".format(arm, result["overall_opening_ppl"],
                                 time.time() - t0), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
