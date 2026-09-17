"""Qwen3 dialogue protocol for the CCM port (2026-09-17).

Replaces the Llama host protocol (src/data/dialogue/data.py) while
keeping the same cut/merge semantics: every history turn carries one
[C0 C1 S0 S1] block, the final context turn carries none, and the
target is a single assistant turn.

Serialization contract (CCM_Qwen3_Work_Implementation_Spec 5.1, the
"frozen fixture" reading):
  - Qwen native chat template, empty system prompt, text user/assistant
    roles only;
  - the LAST turn is the assistant target, the second-to-last the user
    query, alternating backwards from the end;
  - one history block = <|im_start|>{role}\\n{body}<|im_end|>\\n, then
    the [C0 C1 S0 S1] block (merge_recur layout);
  - the context turn keeps header+body+im_end+newline and NO comp block;
  - the target is body+<|im_end|>; the assistant header before it is
    prompt (unsupervised), mirroring the Llama sep-token placement.

Attention-visibility construction (pad_inputs + the official
get_comp_attn_mask_recur on the SUM tokens) is shared with the Llama
host; only the token layout differs.
"""

import os
import torch
import numpy as np
from collections import defaultdict

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


class Qwen3DialogueDataset:
    """DailyDialog under the Qwen3 chat-template protocol.

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
        print("[qwen3-dialog] train {} / val {} / test {}{}".format(
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

    # -- frozen block rendering -------------------------------------------

    def _header_ids(self, role):
        """<|im_start|>{role}\\n as token ids (no chat-template call)."""
        return self.tokenizer("<|im_start|>{}\n".format(role),
                              add_special_tokens=False)["input_ids"]

    def _end_ids(self):
        """<|im_end|>\\n (block footer + template newline)."""
        return self.tokenizer("<|im_end|>\n",
                              add_special_tokens=False)["input_ids"]

    def _roles(self, k):
        """Roles for k turns ending at an assistant target.

        The last turn is the assistant target, the second-to-last is the
        user query, alternating backwards.
        """
        roles = []
        for i in range(k):
            roles.append("assistant" if (k - 1 - i) % 2 == 0 else "user")
        return roles

    def sample(self, item, random_k=False, comp_ids=(), sum_ids=(),
               sum_recur=False, neg_control=False, online=True):
        """Render one instance.

        online=True (ccm arms): history turns carry [C0 C1 (+S0 S1)].
        online=False (full_ctx ref): pure concatenation, one trailing
        comp block appended after the context turn (mirrors the Llama
        host _concat_dialog).  neg_control (no_ctx ref): no history.
        Returns dict(input_ids, output_ids) where output_ids is the
        fully supervised target (body + <|im_end|>).
        """
        dialog = item["dialog"]
        k = len(dialog)
        if random_k and item["is_train"]:
            k = int(np.random.randint(3, k + 1))
            dialog = dialog[:k]
        roles = self._roles(k)

        prompt = []
        hist = range(k - 2) if not neg_control else range(0)
        for i in hist:
            prompt += self._header_ids(roles[i])
            prompt += list(dialog[i])
            prompt += self._end_ids()
            if online:
                if sum_recur:
                    prompt += list(comp_ids) + list(sum_ids)
                else:
                    prompt += list(comp_ids)
        # context turn (second-to-last): no comp block under online
        prompt += self._header_ids(roles[k - 2])
        prompt += list(dialog[k - 2])
        prompt += self._end_ids()
        if not online:
            prompt += list(comp_ids)
            if len(sum_ids) and not sum_recur:
                prompt += list(sum_ids)
        # target header (assistant) is prompt; body + im_end supervised
        prompt += self._header_ids("assistant")
        target = list(dialog[k - 1]) + \
            self.tokenizer("<|im_end|>",
                           add_special_tokens=False)["input_ids"]
        return {"input_ids": prompt, "output_ids": target,
                "orig": item["orig"][:k], "split": item["split"],
                "act": item["act"][:k] if "act" in item else None}


class Qwen3DialogueCollator:
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
