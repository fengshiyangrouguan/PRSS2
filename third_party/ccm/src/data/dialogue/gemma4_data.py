"""Gemma4 dialogue protocol for the CCM port (2026-09-20).

Follows the OFFICIAL Llama-host protocol (src/data/dialogue/data.py
sample_dialog) - gemma-4-E4B is a BASE model, so there is no chat
template: the frozen layout is the official plain-text one.

Serialization contract (official sample_dialog semantics):
  - every history turn (dialog[:-2]) carries [C0 C1 (+S0 S1 in
    merge_recur)] followed by the sep token ('a\\nA:');
  - the context turn (dialog[-2]) is appended bare, NO comp block;
  - the target is the last turn + eos; the input starts with bos;
  - random-k truncation (k >= 3) happens per batch, train split only.

Attention-visibility construction (pad_inputs + the official
get_comp_attn_mask_recur on the SUM tokens) is shared with the Llama
host; only the tokenizer differs.
"""

import os
import numpy as np
from collections import defaultdict

import torch

from ..collator import pad_inputs, prepare_comp_attn_mask_llama


def _read_mirror(mirror):
    """Read the official DailyDialog mirror (ijcnlp layout).

    Each line is one dialogue, turns separated by ``__eou__``; acts are
    space-separated and must align per row (same checks as the official
    DialogueDataset L0 mirror loader).
    """
    def _read_dialog(split):
        out = []
        with open(os.path.join(mirror, split,
                               "dialogues_{}.txt".format(split)),
                  encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                out.append([t.strip() for t in
                            line.strip().split("__eou__")
                            if t.strip()])
        return out

    def _read_act(split):
        out = []
        with open(os.path.join(mirror, split,
                               "dialogues_act_{}.txt".format(split)),
                  encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                out.append([a.strip() for a in
                            line.strip().split() if a.strip()])
        return out

    dataset = {}
    for split in ("train", "validation", "test"):
        dialogs = _read_dialog(split)
        acts = _read_act(split)
        assert len(dialogs) == len(acts), (split, len(dialogs), len(acts))
        for i, (d, a) in enumerate(zip(dialogs, acts)):
            assert len(d) == len(a), (split, i, len(d), len(a))
        dataset[split] = {"dialog": dialogs, "act": acts}
    return dataset


def _preprocess(text):
    """Official text normalization (same as DialogueDataset._preprocess)."""
    text = text.replace("。", ".").replace("’", "'")
    text = text.replace(" ,", ",").replace(" .", ".").replace(" '", "'")
    text = text.replace(" ?", "?").replace(" !", "!").replace(" ;", ";")
    text = text.replace("' ", "'")
    text = [t.strip() for t in text.split(".")]
    return ". ".join(text).strip()


class Gemma4DialogueDataset:
    """DailyDialog under the official plain-text protocol with a Gemma4
    tokenizer.

    Tokenized turns are stored per split; the collator renders the
    frozen block layout at batch time (random-k truncation happens
    there, same as the official online loader).
    """

    def __init__(self, tokenizer, mirror, max_len_turn=128,
                 max_len_total=None, pooled=False):
        self.tokenizer = tokenizer
        dataset = _read_mirror(mirror)
        # official DialogueDataset thresholds: train 400, val/test 600
        if max_len_total is None:
            max_len_total = {"train": 400, "validation": 600,
                             "test": 600}
        # sep = 'a\\nA:' ids (official); bos/eos from the tokenizer.
        self.sep_token = tokenizer.encode("a\nA:",
                                          add_special_tokens=False)
        assert len(self.sep_token) >= 1, \
            "sep_token should have length >= 1"
        self.bos_token = ([tokenizer.bos_token_id]
                         if tokenizer.bos_token_id is not None else [])
        self.eos_token = ([tokenizer.eos_token_id]
                         if tokenizer.eos_token_id is not None else [])

        self._splits = {}
        for split in ("train", "validation", "test"):
            thr = (max_len_total[split] if isinstance(max_len_total, dict)
                   else max_len_total)
            items = []
            for d, a in zip(dataset[split]["dialog"], dataset[split]["act"]):
                dialog = [_preprocess(t) for t in d]
                tok = [tokenizer(t, add_special_tokens=False)["input_ids"]
                       for t in dialog]
                if len(dialog) <= 2:
                    continue
                if any(len(t) > max_len_turn for t in tok):
                    continue
                if sum(len(t) for t in tok) > thr:
                    continue
                items.append({"dialog": tok, "act": a, "orig": dialog,
                              "split": split,
                              "is_train": split == "train"})
            self._splits[split] = items
        self.trainset = self._splits["train"]
        self.valset = self._splits["validation"]
        self.testset = self._splits["test"]
        # train_ccm compatibility alias (next_batch consumes train_dataset)
        self.train_dataset = self.trainset
        if pooled:
            # official clean_split=False protocol: val + test merged
            # (protocol B, same as the Llama-line pooled evaluation)
            self.valset = self.valset + self.testset
        self.eval_dataset = {
            "turn_{}".format(k):
                self._subsample(self.valset, n_turn=k)
            for k in (3, 4, 6, 10, 15)
        }
        print("[gemma4-dialog] train {} / val {} / test {}{}".format(
            len(self.trainset), len(self.valset), len(self.testset),
            " (pooled)" if pooled else ""))

    def _subsample(self, items, n_turn):
        """Official bucket semantics: keep dialogues with >= n_turn turns
        truncated to the first n_turn (turn_14 bucket uses 15, same as
        the official _subsample)."""
        out = []
        for item in items:
            if len(item["dialog"]) >= n_turn:
                it = dict(item)
                it["dialog"] = list(item["dialog"])[:n_turn]
                out.append(it)
        return out

    def sample(self, item, random_k=False, comp_ids=(), sum_ids=(),
               sum_recur=False, neg_control=False, online=True):
        """Render one instance under the official plain-text layout.

        online=True (ccm arms): history turns carry [C0 C1 (+S0 S1)].
        online=False (full_ctx ref): pure concatenation, one trailing
        comp block appended after the context turn (mirrors the Llama
        host _concat_dialog).  neg_control (no_ctx ref): no history.
        Returns dict(input_ids, output_ids) where output_ids is the
        fully supervised target (last turn + eos).
        """
        dialog = item["dialog"]
        if random_k and item["is_train"]:
            k = int(np.random.randint(3, len(dialog) + 1))
            dialog = dialog[:k]

        prompt = []
        if neg_control:
            prompt += list(dialog[-2]) + list(self.sep_token)
        else:
            for i in range(len(dialog) - 2):
                # LOCAL FIX lineage (gate_r7): build fresh lists, never
                # mutate the stored rows in place.
                prompt += list(dialog[i])
                if online:
                    prompt += list(comp_ids)
                    if sum_recur:
                        prompt += list(sum_ids)
                prompt += list(self.sep_token)
            # context turn (second-to-last): no comp block under online
            prompt += list(dialog[-2]) + list(self.sep_token)

        target = list(dialog[-1]) + list(self.eos_token)
        prompt = list(self.bos_token) + prompt
        return {"input_ids": prompt, "output_ids": target,
                "orig": item["orig"][:len(dialog)], "split": item["split"],
                "act": item["act"][:len(dialog)]
                if "act" in item else None}


class Gemma4DialogueCollator:
    """Left-padded batch collator; comp mask pipeline shared with the
    official Llama host (pad_inputs + prepare_comp_attn_mask_llama)."""

    def __init__(self, dataset, tokenizer, comp_args, comp_token,
                 sum_token, pad_token, label_pad_token_id=-100,
                 online=True, neg_control=False):
        self.dialog = dataset
        self.tokenizer = tokenizer
        self.comp_args = comp_args
        if type(comp_token) == int:
            comp_token = [comp_token]
        self.comp_token = comp_token
        self.sum_token = sum_token
        self.pad_token = pad_token
        self.label_pad_token_id = label_pad_token_id
        self.online = online
        self.neg_control = neg_control

    def __call__(self, batch):
        model_inputs = defaultdict(list)
        for instance in batch:
            inst = self.dialog.sample(
                instance,
                random_k=not instance.get("fixed_depth", False),
                comp_ids=self.comp_token,
                sum_ids=self.sum_token,
                sum_recur=self.comp_args.attn_type == "merge_recur",
                neg_control=self.neg_control,
                online=self.online,
            )
            input_token = inst["input_ids"]
            output_token = inst["output_ids"]
            full_token = input_token + output_token
            labels = [self.label_pad_token_id] * len(input_token) \
                + output_token
            model_inputs["input_ids"].append(full_token)
            model_inputs["labels"].append(labels)
            model_inputs["attention_mask"].append([1 for _ in full_token])
            model_inputs["prompt_input_ids"].append(input_token)
            model_inputs["prompt_attention_mask"].append(
                [1 for _ in input_token])
            model_inputs["completion_input_ids"].append(output_token)
            model_inputs["completion_attention_mask"].append(
                [1 for _ in output_token])

        sink_token = self.tokenizer.bos_token_id if self.comp_args.sink \
            else None
        model_inputs = pad_inputs("left", model_inputs,
                                  self.label_pad_token_id, self.pad_token)
        model_inputs = prepare_comp_attn_mask_llama(
            model_inputs, self.comp_args, self.comp_token, self.sum_token,
            self.pad_token, sink_token=sink_token)
        return dict(model_inputs)
