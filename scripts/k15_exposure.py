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

# REVIEW ROUND 4 FIX: data_flow logs parse_meta k = random_prefix - 1.
# A full-length 15-turn prefix appears as logged k=14; dialogues longer
# than 15 turns ALSO provide 15-turn-prefix samples (logged k=14) and
# must be counted.
s15plus = [i for i, L in enumerate(lens) if L >= 15]
n_s15plus = len(s15plus)
samp15plus = sum(sid_count[i] for i in s15plus)
full15_hits = sum(sid_k[i].get(14, 0) for i in s15plus)
print("\n== >=15-turn dialogues (log k=14 == full 15-turn prefix) ==")
print("  dialogues: {}".format(n_s15plus))
print("  times sampled: {} ({:.2f}% of all samples)".format(
    samp15plus, 100.0 * samp15plus / total))
print("  full 15-turn-prefix samples (log k=14): {} -> {:.2f}% of their "
      "samples".format(full15_hits,
                       100.0 * full15_hits / max(samp15plus, 1)))
# exactly-15 dialogues subset (review's original 1/13 framing)
s15 = [i for i, L in enumerate(lens) if L == 15]
samp15 = sum(sid_count[i] for i in s15)
k14_exact = sum(sid_k[i].get(14, 0) for i in s15)
print("  exactly-15 subset: {} dialogues, {} samples, {} full-prefix hits "
      "({:.2f}%, theory 1/13 = {:.2f}%)".format(
          len(s15), samp15, k14_exact,
          100.0 * k14_exact / max(samp15, 1), 100.0 / 13))

# deep-chain exposure (log k >= 13 == prefix >= 14)
deep = sum(k_count.get(k, 0) for k in k_count if k >= 13)
print("\n== deep-chain exposure (log k>=13, i.e. prefix>=14) ==")
print("  samples: {} ({:.2f}% of total)".format(deep,
                                                100.0 * deep / total))
print("  log-k distribution tail:")
for k in sorted(k_count)[-8:]:
    print("    logk={}: {} ({:.2f}%)".format(k, k_count[k],
                                             100.0 * k_count[k] / total))

# coverage: >=15-turn dialogues never sampled with a >=14-turn prefix
never_deep = [i for i in s15plus if max(sid_k[i], default=0) < 13]
print("  >=15-turn dialogues never sampled at prefix>=14 (logk>=13): "
      "{}/{}".format(len(never_deep), n_s15plus))
