#!/usr/bin/env python
"""LaMP-2 two-arm training (official merge protocol).

L2 (LaMP line).  Arm structure follows the dialog line R8 protocol with
LaMP-specific simplifications:
  - NO depth stratification: every official sample is naturally a legal
    cut (16 profiles -> 4 COMP -> SUM -> answer), 5914 users is ample.
  - Cut = the SUM memory state on the official training trajectory;
    future = the answer logprob (taken from the training forward's
    logits, zero extra compute).
  - 4x32 sketch ensemble + OAS shrinkage + lambda-by-r_eff carry over
    unchanged (L2b adds them; this L2a stage is task_only-only).

Training protocol = official (Table 13): fp16 AMP, batch 2 x accum 64
(=128), 300 steps, lr 3e-4 cosine w/ 3% warmup, wd 0, grad clip 1.0.
Trainable = comp_embeddings (8 GIST) + Step-2 conditional LoRA only.
Evaluation runs in a SEPARATE process (eval_lamp_ckpt.py), dialog-line
style, to keep the AMP master weights untouched.

Run from /root/autodl-tmp/third_party/ccm:
  python scripts/train_lamp.py --arm task_only --seed 0 \
      --output outputs/lamp_task_s0
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

CCM_ROOT = "/root/autodl-tmp/third_party/ccm"
RPBE_ROOT = "/root/autodl-tmp/src"
sys.path.insert(0, CCM_ROOT)
sys.path.insert(0, RPBE_ROOT)
os.chdir(CCM_ROOT)

N_TOK = 4  # COMP tokens; merge doubles it (4 COMP + 4 SUM)
MEM_TAU = "mem"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["task_only", "ours"], default="task_only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_name_or_path", default="/root/autodl-tmp/llama-7b-hf")
    p.add_argument("--foundation", default="result/lamp/llama-7b-no")
    p.add_argument("--adapter",
                   default="result/lamp/finetune/llama-7b-no-online-merge-ntok4")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--accum_steps", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--kf_lambda", type=float, default=0.0,
                   help="RPBE lambda (ours arm, additive mode); calibrated by "
                        "r_eff rule")
    p.add_argument("--rpbe_mode", choices=["additive", "project"],
                   default="additive",
                   help="additive = loss += -lambda*J (current); project = "
                        "task-primary half-space guardrail: Gamma task grad is "
                        "projected so gJ.d >= b (b=-kappa|g||t|, or the "
                        "J_floor boundary).  Only Gamma (repr) is projected.")
    p.add_argument("--kappa", type=float, default=0.05,
                   help="project mode: dimensionless guardrail margin; the "
                        "correction fires only when cos(gJ, -g_task) < -kappa. "
                        "kappa=0 is the hard constraint.")
    p.add_argument("--j_floor", type=float, default=None,
                   help="project mode: optional preservation floor J>=j_floor; "
                        "boundary b=(j_floor-J_t)/lr.  Default None = pure "
                        "local guardrail (no active raise).")
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--gamma_hidden", type=int, default=64)
    p.add_argument("--rpbe_seed", type=int, default=0)
    p.add_argument("--kf_min_cuts", type=int, default=128,
                   help="min cuts per RPBE window (2 cuts per microbatch, "
                        "so 128 cuts = 64 microbatches = 1 step)")
    p.add_argument("--ridge_eps", type=float, default=1e-4)
    p.add_argument("--calibrate_lambda", action="store_true",
                   help="lambda calibration mode: run with lambda=1, log "
                        "r_eff per closed window (Gamma grad norm ratio)")
    p.add_argument("--output", default="outputs/lamp_task_s0")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--resume_from", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16_weights", action="store_true",
                   help="load the base in fp16 (official protocol; "
                        "comp_embeddings stay fp32 like the official "
                        "separate-embed path).  Needed for T~1024 training "
                        "on 80GB.")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_tokenizer(model_name_or_path):
    from transformers import LlamaTokenizer
    tok = LlamaTokenizer.from_pretrained(model_name_or_path)
    tok.add_special_tokens(
        {"additional_special_tokens": [f"<COMP{k}>" for k in range(2 * N_TOK)]})
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.comp_token_id = tok.additional_special_tokens_ids[-2 * N_TOK:-N_TOK]
    tok.sum_token_id = tok.additional_special_tokens_ids[-N_TOK:]
    return tok


def build_official_host(args, device):
    """Official merge host (L1-verified build; 83.69% dev parity)."""
    from transformers.models.llama.configuration_llama import LlamaConfig
    from src.arch.ccm_llama import LlamaForCausalLM_CCM
    from src.model import load_lora_weight, peft_custom
    from src.utils import SeparatedEmbedding
    from peft import LoraConfig

    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.comp_relative_embedding = "skip"
    config.attn_type = "merge"
    base_dtype = (torch.float16 if getattr(args, "fp16_weights", False)
                  else torch.float32)
    model = LlamaForCausalLM_CCM.from_pretrained(
        args.model_name_or_path, config=config, torch_dtype=base_dtype)
    model = model.to(device)
    load_lora_weight(args.foundation, model, merge=True)
    model.update_comp_token([32000 + k for k in range(N_TOK)],
                            [32000 + N_TOK + k for k in range(N_TOK)])
    model.model.embed_tokens = SeparatedEmbedding(model.model.embed_tokens,
                                                  2 * N_TOK)
    lora_cfg = LoraConfig().from_pretrained(args.adapter)
    model = peft_custom.get_peft_model(model, lora_cfg)
    load_lora_weight(args.adapter, model, merge=False)
    for _p in model.parameters():
        _p.requires_grad_(False)
    # peft wraps one level deeper: PeftModel -> base Llama -> LlamaModel
    model.base_model.model.model.embed_tokens.comp_embeddings.weight \
        .requires_grad_(True)
    for _n, _p in model.named_parameters():
        if "lora_" in _n:
            _p.requires_grad_(True)
    model._official_host = True
    print("[lamp-host] official merge host, trainable = comp_embeddings + "
          "Step-2 cond LoRA", flush=True)
    return model


def build_lamp_user_ids(n_expected, split="train"):
    """Re-derive the LaMP user ids in dataset order (the generator yields
    ids as example KEYS, which load_dataset drops).  Mirrors the lamp2
    cutoff: max_tok_len=100, cutoff_ratio=0.9, lamp_index != 3 branch."""
    import json
    import numpy as np
    base = Path("dataset/lamp/lamp2")
    q_path = base / ("train_questions.json" if split == "train"
                     else "dev_questions.json")
    with open(q_path) as f:
        questions = json.load(f)
    tok = np.load(base / ("token_len_train.npy" if split == "train"
                          else "token_len_dev.npy"))
    valid = tok > 0
    cutoff = (tok < 100) * valid
    remains = cutoff.sum(axis=1)
    total = valid.sum(axis=1)
    ratio = remains / total
    ids = []
    for i, q in enumerate(questions):
        if ratio[i] < 0.9:
            continue
        ids.append(int(q["id"]))
    if n_expected is not None:
        assert len(ids) == n_expected, \
            f"user id count {len(ids)} != dataset rows {n_expected}"
    return ids


def build_dataset_and_collator(tokenizer, model, args):
    from datasets import load_dataset
    from src.arguments import CompressionArguments
    from src.data.lamp import retriever, collator as lamp_collator

    ds = load_dataset("src/data/lamp/lamp.py", name="lamp2",
                      cache_dir=None, split="train")
    ds = ds.add_column("user_id",
                       build_lamp_user_ids(len(ds), split="train"))
    comp_args = CompressionArguments(attn_type="merge",
                                     num_comp_tokens=N_TOK,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    col = lamp_collator.LLaMACollatorForLaMP(
        tokenizer=tokenizer,
        retriever=retriever.BaseRetriever(k=args.k),
        comp_args=comp_args,
        model=model,
        padding="longest",
        comp_token=tokenizer.comp_token_id,
        sum_token=tokenizer.sum_token_id,
        pad_token=tokenizer.pad_token_id,
        k=args.k,
    )
    return ds, col


def trainable_state_dict(model):
    state = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            state[n] = p.detach().cpu().clone()
    return state


def save_trainable(path, model, **extra):
    payload = {"model": trainable_state_dict(model)}
    payload.update(extra)
    torch.save(payload, path)


def load_trainable(path, model, optimizer, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"]
    for n, p in model.named_parameters():
        if p.requires_grad:
            if n not in state:
                raise RuntimeError(f"checkpoint missing param {n}")
            p.data.copy_(state[n].to(device))
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def build_rpbe_components(model, args, device):
    """ours-arm RPBE stack: 4-slot z adapter, 4x32 sketch ensemble,
    frozen chi/phi embeds, LampCutBuilder, OAS branch windows."""
    from rpbe.hosts.ccm.adapter import CCMHostAdapter
    from rpbe.llm.dialogue_records import Llmmaps
    from rpbe.llm.lamp_records import LampCutBuilder
    from rpbe.llm.utterance_embed import UtteranceEmbed
    from rpbe.loss import KFMomentWindow

    # GammaOnetime must already be attached (both arms carry it; review
    # fix for structure parity).
    gammas = [attn.gamma for attn in _gamma_attns(model)]
    cfg = model.model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    adapter = CCMHostAdapter(model, n_layers=n_layers, n_heads=n_heads,
                             n_slots=N_TOK, kv_pairs=2, head_dim=head_dim,
                             z_dim=args.z_dim, seed=args.rpbe_seed)
    maps = Llmmaps(d_chi=64, d_phi=32, m=32, n_branches=4,
                   seed=args.rpbe_seed).to(device)
    builder = LampCutBuilder(maps, z_dim=args.z_dim, seed=args.rpbe_seed)
    utter_embed = UtteranceEmbed(hidden_dim=cfg.hidden_size, d_chi=64,
                                 seed=args.rpbe_seed).to(device)
    phi_embed = UtteranceEmbed(hidden_dim=cfg.hidden_size, d_chi=32,
                               seed=args.rpbe_seed + 100).to(device)
    branches = [KFMomentWindow({MEM_TAU: args.z_dim}, min_ratio=2.0,
                               min_abs=args.kf_min_cuts,
                               eps=args.ridge_eps, fixed_maps=maps,
                               strict=False, autoclose=False, oas=True)
                for _ in range(4)]
    window = BranchEnsembleWindow(branches)
    return {"adapter": adapter, "maps": maps, "builder": builder,
            "utter_embed": utter_embed, "phi_embed": phi_embed,
            "window": window, "gammas": gammas,
            "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim}


class BranchEnsembleWindow:
    """Four independent 32-dim sketch branches (copied from train_ccm.py
    review round 8): J_ens = mean_r J_r at close; the replay gradient is
    the branch-mean of the per-cut adjoints."""

    def __init__(self, branches):
        self.ws = list(branches)
        self.nb = len(self.ws)

    def add(self, rows):
        import dataclasses
        for br, w in enumerate(self.ws):
            sub = [dataclasses.replace(r, p_override=r.p_override[br])
                   for r in rows]
            w.add(sub)
        return {}, {}, []

    def window_ready(self) -> bool:
        return all(w.window_ready() for w in self.ws)

    def _threshold(self, tau: str) -> float:
        return self.ws[0]._threshold(tau)

    def close_replay(self):
        closed: dict = {}
        plans = []
        diags = []
        j_branches = []
        for br, w in enumerate(self.ws):
            c, p, d = w.close_replay()
            for tau, j in c.items():
                closed[tau] = closed.get(tau, 0.0) + float(j) / self.nb
            plans.append(p)
            diags.append(d)
            j_branches.append({tau: float(jj) for tau, jj in c.items()})
        plan: dict = {}
        diag: dict = {}
        for tau in closed:
            oid_all = set()
            for p in plans:
                oid_all |= set(p.get(tau, {}).get("by_oid", {}).keys())
            by_oid = {}
            for oid in oid_all:
                gs = [p.get(tau, {}).get("by_oid", {}).get(oid)
                      for p in plans]
                gs = [g for g in gs if g is not None]
                if gs:
                    by_oid[oid] = (sum(gs) / self.nb).float()
            plan[tau] = {"by_batch": [[]], "by_oid": by_oid}
            d0 = None
            for p_d in diags:
                if tau in p_d:
                    d0 = dict(p_d[tau])
                    break
            if d0 is None:
                diag[tau] = {"failed": None, "below_threshold": True}
            else:
                d0["J_branches"] = [jb.get(tau, float("nan"))
                                    for jb in j_branches]
                d0["J_ens"] = closed.get(tau, 0.0)
                if d0.get("J_shuffled") is not None:
                    d0["J_real_minus_shuffled"] = (
                        float(closed.get(tau, 0.0))
                        - float(d0["J_shuffled"]))
                diag[tau] = d0
        return closed, plan, diag


def answer_logprob(fwd_out, labels, eos_id):
    """Sum of answer-token logprobs (shift semantics, eos excluded) per
    batch row: [B].  Zero extra forward."""
    log_probs = torch.nn.functional.log_softmax(
        fwd_out.logits.to(dtype=torch.float), dim=-1)
    n = labels.shape[0]
    out = torch.zeros(n, dtype=torch.float, device=labels.device)
    for i in range(n):
        y = labels[i]
        valid = (y != -100) & (y != eos_id)
        pos = valid.nonzero(as_tuple=False).flatten()
        if len(pos) == 0:
            continue
        out[i] = log_probs[i, pos - 1, y[pos]].sum()
    return out


def collect_lamp_cuts(batch, labels, sum_ids, comp_ids, pad_id, eos_id,
                      fwd_out, rpbe, embed_tokens, user_ids, model):
    """Per-row cut collection for one microbatch: z (detached), chi
    (whole prompt minus special tokens), phi (answer tokens), outcome
    (answer logprob), and the per-layer Gamma replay caches."""
    adapter = rpbe["adapter"]
    builder = rpbe["builder"]
    B = batch["input_ids"].shape[0]
    ids = batch["input_ids"]
    rows = []
    sum_positions = torch.zeros(B, N_TOK, dtype=torch.long,
                                device=ids.device)
    for b in range(B):
        for s, sid in enumerate(sum_ids):
            pos = (ids[b] == sid).nonzero(as_tuple=False).flatten()
            if len(pos):
                sum_positions[b, s] = pos[0]
    z_all = adapter.extract_z(sum_positions).detach()  # [B, z_dim]
    ans_lp = answer_logprob(fwd_out, labels, eos_id)
    special = set(sum_ids) | set(comp_ids)
    oids = []
    for b in range(B):
        pids = batch["prompt_input_ids"][b]
        pmask = torch.tensor(
            [(int(t) not in special and int(t) != pad_id) for t in pids],
            dtype=torch.bool, device=ids.device)
        chi_ids = pids[pmask].unsqueeze(0)
        chi = rpbe["utter_embed"](embed_tokens, chi_ids, tag=0) \
            if chi_ids.numel() else torch.zeros(1, 64, device=ids.device)
        labs = labels[b]
        apos = ((labs != -100) & (labs != eos_id)).nonzero(
            as_tuple=False).flatten()
        phi_ids = labs[apos].unsqueeze(0)
        phi = rpbe["phi_embed"](embed_tokens, phi_ids, tag=0) \
            if phi_ids.numel() else torch.zeros(1, 32, device=ids.device)
        meta = LampMetaLite(sample_id=0, sum_positions=sum_positions[b],
                            user_id=int(user_ids[b]))
        recs = builder.build(meta, z_all[b:b + 1], chi, phi,
                             outcome=float(ans_lp[b]))
        rows.extend(recs)
        oids.append(recs[0].occurrence_id)
    # Per-layer replay caches for the local Gamma replay.
    cache = [tuple(t.clone() for t in attn._lamp_cache)
             for attn in _gamma_attns(model)]
    return rows, oids, cache


class LampMetaLite:
    """Minimal LampMeta shim for builder.build."""

    def __init__(self, sample_id, sum_positions, user_id):
        self.sample_id = int(sample_id)
        self.sum_positions = sum_positions
        self.user_id = int(user_id)


def _gamma_attns(model):
    from rpbe.hosts.ccm.ccm_patch import _base_model
    base = _base_model(model)
    return [layer.self_attn for layer in base.layers]


def gamma_replay_surrogate(rpbe, by_oid, mb_rows_oids, mb_caches, lam,
                           device):
    """Local Gamma replay (L2, explicit design): g_z -> J^T -> per-layer
    SUM-row pseudo grads -> <g, Gamma(x, pool)> surrogate (numerically
    zero; first-order gradient EXACT for the Gamma parameters because
    J_mem is a fixed linear sketch and the cached inputs are detached
    leaves).  The RPBE gradient scope is DELIBERATELY Gamma-only: comp
    embeddings and LoRA receive the task gradient alone (a cleaner
    attribution than the dialog line's whole-model replay; the dialog
    line keeps its own replay protocol).  Gamma's own forward runs in
    fp32 on both paths (GammaOnetime casts its inputs), so the numerical
    precision path matches the main forward's Gamma call.  Never re-runs
    the 7B forward."""
    from rpbe.llm.mem_lift import JMemLift
    aux = torch.zeros((), device=device)
    n_terms = 0
    j_mem = rpbe["adapter"].j_mem
    n_layers = rpbe["n_layers"]
    n_heads = rpbe["n_heads"]
    head_dim = rpbe["head_dim"]
    gammas = rpbe["gammas"]
    for mb_idx, oids in enumerate(mb_rows_oids):
        for b, oid in enumerate(oids):
            g = by_oid.get(oid)
            if g is None:
                continue
            g_s = j_mem.transpose(g.unsqueeze(0))  # [1, full_dim]
            k_g, v_g = JMemLift.unpack_sum_mem(
                g_s, n_layers=n_layers, n_heads=n_heads, n_slots=N_TOK,
                kv_pairs=2, head_dim=head_dim)
            cache = mb_caches[mb_idx]
            for li in range(n_layers):
                x_k, x_v, _pos = cache[li]
                res_k = gammas[li](x_k[b:b + 1], x_k[b:b + 1])
                res_v = gammas[li](x_v[b:b + 1], x_v[b:b + 1])
                gk = k_g[li]   # [1, H, 4, D] (detached constant)
                gv = v_g[li]
                aux = aux + (gk * res_k).sum() \
                    - (gk * res_k.detach()).sum()
                aux = aux + (gv * res_v).sum() \
                    - (gv * res_v.detach()).sum()
            n_terms += 1
    if n_terms == 0:
        return torch.zeros((), device=device), 0
    return -lam * aux, n_terms


def project_fp_guardrail(g_task_gamma, gamma_params, g_rpbe, boundary,
                         min_norm2=1e-24):
    """Plan B: task-primary half-space guardrail on the Gamma (repr) scope.

    t = -g_task is the task descent direction; g = g_rpbe is the RPBE ascent
    gradient (dJ/dGamma, obtained from the surrogate with lam=1).  Choose

        d = t + mu*g,   mu = (boundary - g.t) / ||g||^2   if g.t < boundary
        d = t                                            otherwise,

    written back as ``p.grad = g_task - mu*g`` so the optimizer's
    ``theta -= lr*grad`` implements ``theta += lr*d``.  boundary =
    -kappa*||g||*||t|| (local guardrail) or (j_floor - J_t)/lr (floor).
    No +eps in the denominator (that would leave a systematic residual);
    ||g|| too small -> no-op.  Returns diagnostics for the log."""
    t = torch.cat([-x.flatten().float() for x in g_task_gamma])
    g = torch.cat([x.flatten().float() for x in g_rpbe])
    normg2 = float((g * g).sum())
    ng = float(g.norm())
    nt = float(t.norm())
    s = float((g * t).sum())                     # g.t  (gJ_dot_d_before)
    diag = {
        "proj_s": s,
        "proj_boundary": float(boundary),
        "proj_norm_t": nt,
        "proj_norm_g": ng,
        "proj_cos": (s / (ng * nt)) if (ng > 0 and nt > 0) else float("nan"),
        "proj_active": False,
        "proj_mu": 0.0,
        "proj_corr_ratio": 0.0,
        "proj_gJ_dot_d_before": s,
        "proj_gJ_dot_d_after": s,                # == g.t when inactive
        "proj_task_dot_d_after": nt * nt,
    }
    if normg2 <= min_norm2 or ng == 0.0:
        return diag
    if s < boundary:
        mu = (boundary - s) / normg2             # > 0
        with torch.no_grad():
            for p, gt, gj in zip(gamma_params, g_task_gamma, g_rpbe):
                p.grad = gt - mu * gj
        diag["proj_active"] = True
        diag["proj_mu"] = float(mu)
        diag["proj_corr_ratio"] = float(mu * ng / max(nt, 1e-12))
        diag["proj_gJ_dot_d_after"] = float(s + mu * normg2)   # == boundary
        diag["proj_task_dot_d_after"] = float(nt * nt + mu * s)
    return diag


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"

    tokenizer = build_tokenizer(args.model_name_or_path)
    model = build_official_host(args, device)
    model.train()

    ds, collator = build_dataset_and_collator(tokenizer, model, args)
    n_items = len(ds)
    print(f"[lamp-train] arm={args.arm} seed={args.seed} "
          f"n_train={n_items}", flush=True)

    use_rpbe = args.arm == "ours"
    rpbe = None
    if args.arm in ("ours", "task_only"):
        # Review fix: BOTH arms carry GammaOnetime (identical merge
        # structure); only `ours` adds the RPBE window/loss.  task_only
        # is the dialog line's gamma_task_only arm.
        from rpbe.hosts.ccm.ccm_patch import attach_gamma_onetime
        gammas = attach_gamma_onetime(model, hidden=args.gamma_hidden)
        print(f"[lamp-train] GammaOnetime attached to BOTH arms' host "
              f"({sum(g.n_params() for g in gammas)} params)", flush=True)
    if use_rpbe:
        rpbe = build_rpbe_components(model, args, device)

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"[lamp-train] trainable params: {n_params}", flush=True)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = max(1, args.max_steps)
    warmup_steps = max(1, int(0.03 * total_steps))

    def _lr_lambda(s):
        if s < warmup_steps:
            return float(s) / float(warmup_steps)
        progress = float(s - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    save_json(out / "config.json", {
        "arm": args.arm, "seed": args.seed, "cli": vars(args),
        "n_items": n_items,
        "protocol": "official LaMP Table 13: fp16 AMP, eff batch 128, "
                    "300 steps, lr 3e-4 cosine 3% warmup, wd 0, clip 1.0",
    })

    # Official stream: shuffled train set cycled forever; same seed =>
    # identical stream on both arms.
    step = 0
    optimizer_steps_executed = 0
    amp_skipped_steps = 0
    total_loss_sum = 0.0
    total_tokens = 0
    epoch = 0
    t_start = time.time()
    last_saved = -1

    if args.resume_from:
        payload = load_trainable(args.resume_from, model, optimizer, device)
        step = int(payload.get("step", 0))
        optimizer_steps_executed = int(payload.get("optimizer_steps_executed", 0))
        amp_skipped_steps = int(payload.get("amp_skipped_steps", 0))
        total_loss_sum = float(payload.get("total_loss_sum", 0.0))
        total_tokens = int(payload.get("total_tokens", 0))
        epoch = int(payload.get("epoch", 0))
        if "rng" in payload:
            random.setstate(payload["rng"]["py"])
            np.random.set_state(payload["rng"]["np"])
            torch.set_rng_state(payload["rng"]["torch"])
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        if "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        if "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        print(f"[lamp-train] RESUME step={step} epoch={epoch} "
              f"executed={optimizer_steps_executed} skips={amp_skipped_steps}",
              flush=True)

    class CollatorWithIds:
        """Official collator + per-sample LaMP user ids (tree identity)."""

        def __init__(self, base):
            self.base = base

        def __call__(self, batch):
            out = self.base(batch)
            out["user_ids"] = [int(b_["user_id"]) for b_ in batch]
            return out

    if use_rpbe:
        collator = CollatorWithIds(collator)

    def _make_dataloader():
        g = torch.Generator().manual_seed(args.seed + epoch)
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, generator=g,
            collate_fn=collator, drop_last=False)

    dataloader = _make_dataloader()

    def rng_state():
        return {
            "py": random.getstate(), "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        }

    comp_ids = tokenizer.comp_token_id
    sum_ids = tokenizer.sum_token_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    # Review 2 diagnostics: per-window Gamma grad ratio/angle and the
    # memory magnitude (z norm) so scale-growth or gradient-conflict
    # hypotheses can be checked from the logs.
    gamma_params = ([p for g in rpbe["gammas"] for p in g.parameters()]
                    if use_rpbe else [])
    z_norm_acc = 0.0
    z_norm_n = 0
    embed_tokens = model.get_input_embeddings()
    mb_rows_oids = []   # per mb: [oid per batch row]
    mb_caches = []      # per mb: per-layer gamma cache clones
    win_stats = {"closed": 0, "below": 0, "failed": 0}
    last_j = float("nan")

    # Review fix: the data iterator spans steps (one continuous stream per
    # epoch; 64 microbatches per optimizer step ADVANCE the stream instead
    # of re-reading the same shuffle prefix every step).
    dataloader_iter = iter(dataloader)
    while step < total_steps:
        step += 1  # global step (HF Trainer semantics: advances even on
                   # AMP skip)
        opt_took = False
        skip_this = False
        step_loss = 0.0
        step_tokens = 0
        micros = 0
        t0 = time.perf_counter()
        # Review fix: zero the grads before accumulating this step's 64
        # microbatches.  Without it, the previous step's unscaled+clipped
        # gradients were silently carried into the next backward pass.
        optimizer.zero_grad(set_to_none=True)
        for mb in range(args.accum_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                epoch += 1
                dataloader = _make_dataloader()
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            user_ids = batch.pop("user_ids", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                fwd_out = model(**batch)
                loss = fwd_out.loss / args.accum_steps
            step_loss += float(fwd_out.loss.item() or 0.0)
            step_tokens += int((labels != -100).sum().item())
            if use_rpbe:
                rows, oids, cache = collect_lamp_cuts(
                    batch, labels, sum_ids, comp_ids, pad_id, eos_id,
                    fwd_out, rpbe, embed_tokens, user_ids, model)
                rpbe["window"].add(rows)
                mb_rows_oids.append(oids)
                mb_caches.append(cache)
                z_norm_acc += float(
                    sum(r.z.detach().float().norm() for r in rows))
                z_norm_n += len(rows)
            scaler.scale(loss).backward()
            micros += 1
        # Window close (ours arm): adjoint + local Gamma replay.
        aux_terms = 0
        j_closed = float("nan")
        if use_rpbe and rpbe["window"].window_ready():
            closed, plan, diag = rpbe["window"].close_replay()
            by_oid = plan.get(MEM_TAU, {}).get("by_oid", {})
            j_closed = closed.get(MEM_TAU, 0.0)
            d0 = diag.get(MEM_TAU, {})
            if d0.get("below_threshold"):
                win_stats["below"] += 1
            elif d0.get("failed") is not None:
                win_stats["failed"] += 1
            else:
                win_stats["closed"] += 1
                last_j = j_closed
                # Gradient snapshot BEFORE the RPBE surrogate backward
                # (review 2 diagnostics): per-window Gamma task/aux grad
                # ratio and angle; calibrate mode additionally reports
                # the all-params r_eff at theta_0 (no param update).
                g_task_all = None
                g_task_gamma = None
                if args.calibrate_lambda or gamma_params \
                        or args.rpbe_mode == "project":
                    g_task_all = [p.grad.detach().clone()
                                  if p.grad is not None
                                  else torch.zeros_like(p)
                                  for p in params]
                    g_task_gamma = [p.grad.detach().clone()
                                    if p.grad is not None
                                    else torch.zeros_like(p)
                                    for p in gamma_params]
                # additive: loss += -lam*J.  project: use lam=1 only to get
                # g = dJ/dGamma, then overwrite the Gamma grad below.
                lam_aux = 1.0 if args.rpbe_mode == "project" \
                    else args.kf_lambda
                aux, aux_terms = gamma_replay_surrogate(
                    rpbe, by_oid, mb_rows_oids, mb_caches, lam_aux, device)
                if aux_terms:
                    scaler.scale(aux).backward()
                calib = {}
                gdiag = {}
                proj = {}
                if g_task_gamma:
                    n_task_g = torch.cat(
                        [g.flatten().float() for g in g_task_gamma]).norm()
                    if args.rpbe_mode == "project":
                        # g_rpbe = dJ/dGamma  (p.grad = g_task - lam_aux*g)
                        g_rpbe = [g0 - p.grad.detach().clone()
                                  for g0, p in zip(g_task_gamma,
                                                   gamma_params)]
                        gG = torch.cat([x.flatten().float() for x in g_rpbe])
                        n_aux_g = gG.norm()
                        cos_g = float((torch.cat(
                            [(-g).flatten().float() for g in g_task_gamma])
                            * gG).sum() / max(n_task_g * n_aux_g, 1e-12)) \
                            if n_task_g > 0 and n_aux_g > 0 \
                            else float("nan")
                        lr_now = optimizer.param_groups[0]["lr"]
                        boundary = (-args.kappa * float(n_task_g)
                                    * float(n_aux_g)) \
                            if args.j_floor is None else \
                            (args.j_floor - j_closed) / max(lr_now, 1e-12)
                        proj = project_fp_guardrail(
                            g_task_gamma, gamma_params, g_rpbe, boundary)
                    else:
                        g_aux = torch.cat([(p.grad.detach().clone()
                                            - g0).flatten().float()
                                           for g0, p in zip(g_task_gamma,
                                                            gamma_params)])
                        n_aux_g = g_aux.norm()
                        cos_g = float((torch.cat(
                            [g.flatten().float() for g in g_task_gamma])
                            * g_aux).sum() / max(n_task_g * n_aux_g, 1e-12)) \
                            if n_task_g > 0 and n_aux_g > 0 \
                            else float("nan")
                    gdiag = {
                        "g_gamma_task": float(n_task_g),
                        "g_gamma_aux": float(n_aux_g),
                        "g_gamma_ratio": float(
                            n_aux_g / max(n_task_g, 1e-12)),
                        "g_gamma_cos": cos_g,
                    }
                if args.calibrate_lambda and g_task_all is not None:
                    g_total = [p.grad.detach().clone() for p in params]
                    n_task = torch.cat(
                        [g.flatten().float() for g in g_task_all]).norm()
                    n_rpbe = torch.cat(
                        [(t - g0).flatten().float()
                         for t, g0 in zip(g_total, g_task_all)]).norm()
                    # Dual-scope r_eff (review 2 follow-up): the lambda
                    # decision uses the GAMMA scope (the aux gradient
                    # acts on Gamma only), while the all-params ratio is
                    # kept for reference.
                    calib = {
                        "r_eff": float(n_rpbe / max(n_task, 1e-12)),
                        "r_eff_gamma": float(gdiag.get("g_gamma_ratio",
                                                      float("nan"))),
                        "n_task": float(n_task),
                        "n_rpbe": float(n_rpbe),
                    }
                with log_path.open("a") as f:
                    rec = {
                        "step": step, "event": "window_close",
                        "J_ens": j_closed, "aux_terms": aux_terms,
                        "rpbe_mode": args.rpbe_mode,
                        "M_unique_trees": d0.get("M_unique_trees"),
                        "alpha_z": d0.get("alpha_z"),
                        "alpha_p": d0.get("alpha_p"),
                        "J_branches": d0.get("J_branches"),
                        "J_shuffled": d0.get("J_shuffled"),
                        "J_real_minus_shuffled":
                            d0.get("J_real_minus_shuffled"),
                        "z_norm_mean": float(
                            z_norm_acc / max(z_norm_n, 1)),
                    }
                    rec.update(gdiag)
                    if proj:
                        rec.update(proj)
                    if calib:
                        rec.update(calib)
                    f.write(json.dumps(rec) + "\n")
            mb_rows_oids.clear()
            mb_caches.clear()
            z_norm_acc = 0.0
            z_norm_n = 0
        scaler.unscale_(optimizer)
        grad_ok = all(p.grad is None or torch.isfinite(p.grad).all()
                      for p in params)
        if not grad_ok:
            amp_skipped_steps += 1
            skip_this = True
        elif args.calibrate_lambda:
            # Review fix: calibrate mode measures r_eff at theta_0 with
            # NO parameter update (the R8 protocol).  scaler.update()
            # still runs to restore AMP state.
            skip_this = True
        else:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            opt_took = True
        if skip_this:
            # Review fix: AMP protocol requires scaler.update() after
            # every unscale, even on a skip (overflow needs the scale to
            # step DOWN; calibrate mode consumed scaled grads without a
            # step).
            scaler.update()
        if opt_took:
            optimizer_steps_executed += 1
            total_loss_sum += step_loss
            total_tokens += step_tokens

        dt = time.perf_counter() - t0
        if step == 1 or step % 10 == 0 or step == total_steps or skip_this:
            lr_now = scheduler.get_last_lr()[0]
            j_str = f" J={last_j:.4f}" if use_rpbe else ""
            print(f"[lamp-train] step {step}/{total_steps} "
                  f"loss={step_loss:.4f} lr={lr_now:.2e} "
                  f"t={dt:.1f}s opt={opt_took} skip={skip_this} "
                  f"epoch={epoch}{j_str}", flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps({
                    "step": step, "loss": step_loss, "lr": lr_now,
                    "sec": dt, "opt": opt_took, "skip": skip_this,
                    "epoch": epoch, "tokens": step_tokens,
                }) + "\n")

        if step % args.save_every == 0 and step != last_saved:
            save_trainable(out / f"ckpt_step{step}.pt", model,
                           step=step,
                           optimizer=optimizer.state_dict(),
                           scheduler=scheduler.state_dict(),
                           scaler=scaler.state_dict(),
                           rng=rng_state(),
                           epoch=epoch,
                           optimizer_steps_executed=optimizer_steps_executed,
                           amp_skipped_steps=amp_skipped_steps,
                           total_loss_sum=total_loss_sum,
                           total_tokens=total_tokens)
            last_saved = step
            print(f"[lamp-train] saved ckpt_step{step}.pt", flush=True)

    # Final save (always, even if not on the save_every grid)
    save_trainable(out / "ckpt_final.pt", model,
                   step=step,
                   optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(),
                   scaler=scaler.state_dict(),
                   rng=rng_state(),
                   epoch=epoch,
                   optimizer_steps_executed=optimizer_steps_executed,
                   amp_skipped_steps=amp_skipped_steps,
                   total_loss_sum=total_loss_sum,
                   total_tokens=total_tokens)
    avg_loss = total_loss_sum / max(optimizer_steps_executed, 1)
    print(f"[lamp-train] DONE steps={step} executed={optimizer_steps_executed} "
          f"skips={amp_skipped_steps} avg_step_loss={avg_loss:.4f} "
          f"total_tokens={total_tokens} "
          f"wall={time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
