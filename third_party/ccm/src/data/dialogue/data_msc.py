"""MSC (Multi-Session Chat) dataset for the dialogue host. (frozen 2026-09-20)

Frozen protocol (review 2026-09-20):
- Source: ``dataset/msc/{train,valid,test}.jsonl`` — the HF mirror
  ``gonced8/multi-session_chat`` (a faithful re-format of the official
  ParlAI MSC data; the official parl.ai zip is no longer served).
  Verified against official numbers: train 236,987 utterances,
  valid 1,000 rows, test 501 rows (all 5-session).
- One row = one episode prefix: ``{id, init_personas, sessions[]}``;
  a session = ``{session_id, personas, dialogue: [{speaker, text}],
  time_elapsed}``.  Speakers alternate strictly (Speaker 1 opens).
- Session boundary marker: plain text ``[Session resumes after {te}]``
  inserted between sessions with ``time_elapsed.strip() != '0'``.
  Shared verbatim by ALL arms; NO special token / vocab change.  The
  marker carries NO comp token (it is structural metadata, not
  dialogue content) and never serves as a cut position or a future.
- The official ``_threshold(max_len_total=512)`` does NOT apply (MSC
  sessions are far longer than DailyDialog; test rows reach ~2.8k
  utterance tokens).  A per-utterance cap of 256 keeps degenerate
  rows out; window length is budgeted at cut-sampling time.
- Trainset: ONE ROW PER ELIGIBLE CUT — every (episode, v_utt) with
  v_utt + 2 < n_utt (the cut's two future utterances exist; markers
  occupy no utt index, so a cut may naturally span a session
  boundary).  Each row's ``dialog`` is the window ending at the cut's
  second future utterance, budgeted to ``max_length`` tokens (oldest
  turns dropped first).  ``fixed_depth=True`` keeps the collator's
  random-k truncation off; train_msc samples rows UNIFORMLY (eligible
  cut pool), which gives long episodes their natural weight.
- Eval buckets: ``eval_dataset['sess_{s}']`` for s in 2..5 — rows end
  at the LAST utterance of session s (history S1..S(s-1) + full
  session s), so ``sample_dialog`` evaluates that session's final
  exchange and the official train.py eval loop stays functional.
  The CANONICAL session-opening / all-response PPL is computed by
  ``train_msc --eval`` through ``exchange_instances`` (teacher-forced
  per-exchange windows, opening = first predicted response of each
  session s >= 2, matching the MSC paper's Session Openings).
"""

import json
import os

from datasets import Dataset, DatasetDict

from path_config import DATAPATH

MARKER_FMT = "[Session resumes after {}]"
MAX_UTT_TOK = 256           # per-utterance cap (official 512 TOTAL cap dropped)
DEFAULT_MAX_LENGTH = 2048   # training window budget; eval uses 3072


class DialogueDataset:
    """MSC dialogue dataset, interface-compatible with the official
    DialogueDataset (sample_dialog contract for the shared collator)."""

    def __init__(self, tokenizer, comp_token=[], online=True,
                 add_comp_token=True, clean_split=False,
                 eval_source="val", max_length=DEFAULT_MAX_LENGTH,
                 msc_dir=None):
        assert eval_source in ("val", "test")
        self.tokenizer = tokenizer
        self.online = online
        self.max_length = int(max_length)
        self.bos_token = ([tokenizer.bos_token_id]
                          if tokenizer.bos_token_id is not None else [])
        self.eos_token = ([tokenizer.eos_token_id]
                          if tokenizer.eos_token_id is not None else [])
        self.comp_token = list(comp_token) if add_comp_token else []
        self.sep_token = tokenizer.encode("a\nA:",
                                          add_special_tokens=False)[1:]
        assert len(self.sep_token) >= 1

        base = msc_dir or os.path.join(DATAPATH, "msc")
        self.episodes = {}
        for split in ("train", "valid", "test"):
            path = os.path.join(base, split + ".jsonl")
            rows = [json.loads(l) for l in
                    open(path, encoding="utf-8").read().strip().split("\n")]
            eps, dropped = [], 0
            for r in rows:
                ep = self._build_episode(r)
                if ep is None:
                    dropped += 1
                    continue
                eps.append(ep)
            self.episodes[split] = eps
            print("[msc] {}: {} episodes kept ({} dropped)".format(
                split, len(eps), dropped), flush=True)

        # Train rows: one row per eligible cut (the uniform-cut pool).
        train_rows = []
        for ei, ep in enumerate(self.episodes["train"]):
            for v in range(ep["n_utt"] - 2):
                win = self._cut_window(ep, v)
                train_rows.append({
                    "dialog": win,
                    "is_train": True,
                    "fixed_depth": True,
                    "orig_id": ei,
                    "cut_v": v,
                })
        self.train_dataset = Dataset.from_list(train_rows, split="train")
        # dict view for non-datasets consumers (train_msc cut sampler)
        self.trainset = {"dialog": [r["dialog"] for r in train_rows],
                         "orig_id": [r["orig_id"] for r in train_rows],
                         "cut_v": [r["cut_v"] for r in train_rows],
                         "is_train": [True] * len(train_rows)}

        # Eval buckets: one row per (episode, session s in 2..5), ending
        # at that session's last utterance.
        src = self.episodes["valid" if eval_source == "val" else "test"]
        self.eval_dataset = DatasetDict()
        for s in range(2, 6):
            rows = []
            for ei, ep in enumerate(src):
                if s > ep["n_sessions"]:
                    continue
                last = ep["utt_turn_idx"][ep["sess_utt_end"][s - 1] - 1]
                rows.append({
                    "dialog": [dict(t) for t in ep["turns"][:last + 1]],
                    "is_train": False,
                    "fixed_depth": True,
                    "orig_id": ei,
                    "session": s,
                })
            self.eval_dataset["sess_{}".format(s)] = Dataset.from_list(
                rows, split="{}_s{}".format(eval_source, s))
        print(self.train_dataset)
        print(self.eval_dataset)

    # ------------------------------------------------------------------
    # episode construction
    # ------------------------------------------------------------------
    def _build_episode(self, row):
        """One episode row -> {turns, utt index maps, session ranges}.

        ``turns``: [{"kind": "utt"|"marker", "tokens": [...]}, ...]
        ``utt_turn_idx``: turn index of each utterance (utt-space -> turns)
        ``utt_sessions``: session id of each utterance
        ``sess_utt_end``: [s] = exclusive utt-space end of session s
        """
        turns, utt_sessions, utt_turn_idx = [], [], []
        for si, sess in enumerate(row["sessions"]):
            te = (sess.get("time_elapsed") or "0").strip()
            if si > 0 and te and te != "0":
                mt = self.tokenizer(MARKER_FMT.format(te),
                                    add_special_tokens=False)["input_ids"]
                turns.append({"kind": "marker", "tokens": mt})
            for u in sess["dialogue"]:
                toks = self.tokenizer((u.get("text") or "").strip(),
                                      add_special_tokens=False)["input_ids"]
                if not toks or len(toks) > MAX_UTT_TOK:
                    return None          # degenerate utterance: drop row
                turns.append({"kind": "utt", "tokens": toks})
                utt_sessions.append(si)
                utt_turn_idx.append(len(turns) - 1)
        if len(utt_turn_idx) < 3:
            return None                  # no legal cut (v + 2 futures)
        n_sess = len(row["sessions"])
        sess_utt_end = [0] * n_sess
        for s in utt_sessions:
            sess_utt_end[s] += 1
        for s in range(1, n_sess):
            sess_utt_end[s] += sess_utt_end[s - 1]
        return {"turns": turns, "n_utt": len(utt_turn_idx),
                "n_sessions": n_sess, "utt_sessions": utt_sessions,
                "utt_turn_idx": utt_turn_idx, "sess_utt_end": sess_utt_end}

    def _turn_cost(self, t):
        c = len(t["tokens"]) + len(self.sep_token)
        if t["kind"] == "utt" and self.online:
            c += len(self.comp_token)
        return c

    def _cut_window(self, ep, v, budget=None):
        """Window ending at the second future utterance of cut v,
        budgeted to max_length tokens (oldest WHOLE turns dropped)."""
        budget = budget or self.max_length
        end = ep["utt_turn_idx"][v + 2]              # inclusive turn index
        cost = sum(self._turn_cost(t) for t in ep["turns"][:end + 1])
        start = 0
        while cost > budget and start < end:
            cost -= self._turn_cost(ep["turns"][start])
            start += 1
        return [dict(t) for t in ep["turns"][start:end + 1]]

    # ------------------------------------------------------------------
    # shared collator contract
    # ------------------------------------------------------------------
    def _concat_turns(self, hist, sum_token=None, sum_recur=False):
        """Official _concat_dialog semantics with marker awareness:
        utterance turns get comp (+sum if recur), markers get neither."""
        context = []
        for t in hist:
            tk = list(t["tokens"])
            if t["kind"] == "utt" and self.online:
                tk += self.comp_token
            if sum_recur:
                tk += sum_token
            context += tk + self.sep_token
        if hist:
            context = context[:-len(self.sep_token)]
        if not self.online:
            context += self.comp_token
        if sum_token is not None and not sum_recur:
            context += sum_token
        context += self.sep_token
        return context

    def sample_dialog(self, instance, random_k=False, sum_token=None,
                      sum_recur=False, neg_control=False):
        """Official contract: history (compressed) + immediate context
        utterance + sep -> target utterance + eos.  The final two
        UTTERANCES are the context/target pair; markers in between are
        appended to the context (a session-opening target therefore sees
        its boundary marker, exactly as in training).

        ``random_k`` is intentionally unused: MSC training rows are
        pre-built cut windows (fixed_depth=True) and the official
        random-prefix protocol has no MSC counterpart.
        """
        turns = instance["dialog"]
        utt_pos = [i for i, t in enumerate(turns) if t["kind"] == "utt"]
        out_pos = utt_pos[-1]
        in_pos = utt_pos[-2]
        if neg_control:
            context = list(turns[in_pos]["tokens"]) + self.sep_token
        else:
            context = self._concat_turns(turns[:in_pos], sum_token,
                                         sum_recur)
            context += list(turns[in_pos]["tokens"]) + self.sep_token
            for t in turns[in_pos + 1:out_pos]:     # markers only
                context += list(t["tokens"]) + self.sep_token
        output = list(turns[out_pos]["tokens"])
        return {"input_ids": self.bos_token + context,
                "output_ids": output + self.eos_token}

    # ------------------------------------------------------------------
    # canonical evaluation (train_msc --eval)
    # ------------------------------------------------------------------
    def exchange_instances(self, split, sessions=(2, 3, 4, 5)):
        """One row per (episode, session s, exchange e): the window
        ending at that exchange's RESPONSE utterance, so sample_dialog
        evaluates exactly that exchange.  ``is_opening`` marks e == 0,
        the MSC-paper Session Opening (first predicted response of
        each session s >= 2)."""
        for ei, ep in enumerate(self.episodes[split]):
            for s in sessions:
                if s > ep["n_sessions"]:
                    continue
                i0 = ep["sess_utt_end"][s - 1]
                i1 = (ep["sess_utt_end"][s]
                      if s < ep["n_sessions"] else ep["n_utt"])
                n_ex = (i1 - i0) // 2        # exchanges in session s
                for e in range(n_ex):
                    last = ep["utt_turn_idx"][i0 + 2 * e + 1]
                    yield {
                        "dialog": [dict(t) for t in
                                   ep["turns"][:last + 1]],
                        "is_train": False,
                        "fixed_depth": True,
                        "orig_id": ei,
                        "session": s,
                        "is_opening": int(e == 0),
                    }

    def _stat(self):
        for split, eps in self.episodes.items():
            n_utt = sum(ep["n_utt"] for ep in eps)
            n_sess = sum(ep["n_sessions"] for ep in eps)
            print("[msc] {}: {} episodes, {} sessions, {} utts, "
                  "{} train cuts".format(
                      split, len(eps), n_sess, n_utt,
                      len(self.train_dataset) if split == "train" else 0))


if __name__ == "__main__":
    from transformers import LlamaTokenizer
    tok = LlamaTokenizer.from_pretrained(
        os.environ.get("MODEL_PATH", "llama-7b-hf"))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    ds = DialogueDataset(tok, comp_token=[32000, 32001], online=True,
                         eval_source="val", msc_dir=os.environ.get("MSC_DIR"))
    ds._stat()
    # smoke: one cut window + one opening exchange
    inst = ds.train_dataset[0]
    print("train row keys:", sorted(inst.keys()))
    tokout = ds.sample_dialog(inst, sum_token=[32002, 32003],
                              sum_recur=True)
    print("cut input :", tok.decode(tokout["input_ids"])[:200])
    print("cut output:", tok.decode(tokout["output_ids"])[:120])
    ex = next(ds.exchange_instances("valid"))
    tokout = ds.sample_dialog(ex)
    print("opening input :", tok.decode(tokout["input_ids"])[-200:])
    print("opening output:", tok.decode(tokout["output_ids"])[:120])
