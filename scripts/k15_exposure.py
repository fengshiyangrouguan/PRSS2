#!/usr/bin/env python3
"""Review item 5: actual training exposure of deep-chain prefixes (k>=13,
k==15) from the real data_flow.jsonl of the e50 run."""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/src")
sys.path.insert(0, "/root/autodl-tmp/third_party/ccm")
os.environ["DIALOG_MIRROR"] = "/root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import train_ccm as tc  # noqa: E402

BASE = "/root/autodl-tmp"
DF_PATH = BASE + "/outputs/ccm_pilot/seed0_CCM_Diag_ours_fnd_l03_e50/data_flow.jsonl"

tok = tc.build_tokenizer(tc.parse_args().parse_args([]) if False else
                         __import__("types").SimpleNamespace(
                             model_name_or_path=BASE + "/llama-7b-hf"))
from src.data.dialogue.data import DialogueDataset
ds = DialogueDataset(tok, comp_token=tok.comp_token_id,
                     online=True, add_comp_token=True, clean_split=True)
lens = [len(d) for d in ds.trainset["dialog"]]
n_items = len(lens)
print("train set: {} dialogues; length distribution:".format(n_items))
for lo, hi in [(3, 5), (6, 8), (9, 12), (13, 14), (15, 15), (16, 100)]:
    cnt = sum(1 for L in lens if lo <= L <= hi)
    print("  len {}-{}: {} dialogues".format(lo, hi if hi < 100 else "+", cnt))

# sample counts from data flow
sid_count = Counter()
k_count = Counter()
sid_k = defaultdict(Counter)
for line in open(DF_PATH):
    row = json.loads(line)
    sid = int(row["sid"]) % n_items
    k = int(row["k"])
    sid_count[sid] += 1
    k_count[k] += 1
    sid_k[sid][k] += 1

total = sum(sid_count.values())
print("total samples: {}".format(total))

# exactly-15 dialogues
s15 = [i for i, L in enumerate(lens) if L == 15]
n_s15 = len(s15)
samp15 = sum(sid_count[i] for i in s15)
k15_hits = sum(sid_k[i].get(15, 0) for i in s15)
print("\n== exactly-15-turn dialogues ==")
print("  dialogues: {}".format(n_s15))
print("  times sampled: {} ({:.2f}% of all samples)".format(
    samp15, 100.0 * samp15 / total))
print("  times sampled with k==15: {} -> {:.2f}% of their samples "
      "(theoretical 1/13 = {:.2f}%)".format(
          k15_hits, 100.0 * k15_hits / max(samp15, 1), 100.0 / 13))

# deep-chain exposure: k >= 13 on dialogues long enough to allow it
deep = sum(k_count.get(k, 0) for k in k_count if k >= 13)
print("\n== deep-chain exposure (k>=13) ==")
print("  samples with k>=13: {} ({:.2f}% of total)".format(
    deep, 100.0 * deep / total))
print("  k distribution tail:")
for k in sorted(k_count)[-8:]:
    print("    k={}: {} ({:.2f}%)".format(k, k_count[k],
                                         100.0 * k_count[k] / total))

# per-dialogue worst coverage: 15-turn dialogues never sampled with k>=14
never_deep = [i for i in s15 if max(sid_k[i], default=0) < 14]
print("  15-turn dialogues never sampled at k>=14: {}/{}".format(
    len(never_deep), n_s15))
