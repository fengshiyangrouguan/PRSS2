#!/usr/bin/env python3
"""Same-Interface Oracle Ceiling Test (R9 2026-09-16).

Freeze the OFFICIAL CCM-M weights completely and allow ONE thing to
change: the per-dialogue SUM merge weights alpha in Delta^t (prefix
softmax of per-dialogue logits, shared across layers and SUM rows;
row j uses alpha[:j] renormalized).  Zero logits reproduce the
official uniform 1/j mask bit-for-bit (sanity gate: the uniform
restart's step-0 NLL must equal the official merge NLL).

Every dialogue gets its own alpha, optimized DIRECTLY against the
target CE (shifted + EOS-excluded, the official protocol) — the
oracle sees the answer, so this is a ceiling diagnostic, never a
method and never a baseline:

    H_L = NLL_merge,L - NLL_oracle,L   (merge-rule headroom)
    C_L = NLL_merge,L - NLL_full,L     (compression gap)
    R_L = H_L / C_L                    (oracle-recoverable fraction)

Depth protocol C: L in {2, 4} (plus L=1 as the no-freedom sanity),
sampled 100 dialogues per depth, 5 restarts x 150 Adam steps,
trajectory minimum.  Batch 8: the alpha tensor is [B, t] and the
objective is the batch-average target NLL (reported as such).
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import train_ccm as tc
import eval_ccm_depth as ed
from rpbe.hosts.ccm.ccm_patch import attach_gamma  # noqa: F401 (unused
# here, but keeps the import surface identical to eval_ccm_depth)
from src.arch.ccm_llama import (set_oracle_logits,  # noqa: F401
                                clear_oracle_logits)

DEPTHS = [1, 2, 4]
N_SAMPLE = 100
N_RESTARTS = 5
N_STEPS = 150
LR = 0.5


def batch_size_for(L):
    """fp32 7B: L=4 builds 6-turn sequences (OOM at 16 on 79GB)."""
    return 16 if L <= 2 else 8


def build_merge_model(args, device):
    """Official CCM-M: build_official_host WITHOUT any ours checkpoint."""
    tokenizer = tc.build_tokenizer(args)
    model = tc.build_official_host(args, device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return tokenizer, model


def make_inits(t, seed):
    """5 restarts: uniform / recent-heavy / early-heavy / random x2."""
    torch.manual_seed(seed)
    inits = [torch.zeros(t)]                                   # uniform
    inits.append(torch.arange(t, dtype=torch.float32) ** 2)    # recent-heavy
    inits.append(-(torch.arange(t, dtype=torch.float32) ** 2))  # early-heavy
    inits.append(torch.randn(t))
    inits.append(torch.randn(t))
    return inits


def optimize_batch(model, batch, device, t, seed):
    """Jointly optimize [B, t] logits for one batch; return
    (best_batch_nll, per_dialogue_nlls_at_best, restart_summary).

    L=1 fast path: one history token => the simplex has ONE point, the
    softmax is constant 1 and NO optimization can change the forward —
    run a short 1-restart pass purely to certify the identity."""
    B = batch["input_ids"].shape[0]
    labels = batch["labels"].to(device)
    n_valid = int((labels[:, 1:] != -100).sum())
    bests = []
    restarts = []
    if t <= 1:
        inits = [torch.zeros(t)]
        n_steps = 10
    else:
        inits = make_inits(t, seed)
        n_steps = N_STEPS
    for r_i, init in enumerate(inits):
        logits = torch.nn.Parameter(
            init[None, :].repeat(B, 1).to(device))
        opt = torch.optim.Adam([logits], lr=LR)
        traj = []
        best_nll = float("inf")
        best_logits = None
        for s in range(n_steps):
            set_oracle_logits(logits)
            out = tc.run_forward(model, batch, device, grad_enabled=True)
            loss, n = ed.eval_ce_shifted_noeos(out, labels, device)
            nll = float(loss.detach()) / max(n, 1)
            traj.append(nll)
            if nll < best_nll:
                best_nll = nll
                best_logits = logits.detach().clone()
            opt.zero_grad()
            (loss / max(n, 1)).backward()
            opt.step()
        clear_oracle_logits()
        # per-dialogue NLL at the best point
        set_oracle_logits(best_logits)
        with torch.no_grad():
            out = tc.run_forward(model, batch, device, grad_enabled=False)
        clear_oracle_logits()
        per_d = []
        for b_i in range(B):
            lab_b = labels[b_i, 1:]
            n_b = int((lab_b != -100).sum())
            if n_b == 0:
                per_d.append(float("nan"))
                continue
            import types
            out_b = types.SimpleNamespace(logits=out.logits[b_i:b_i + 1])
            s_b, _ = ed.eval_ce_shifted_noeos(
                out_b, labels[b_i:b_i + 1], device)
            per_d.append(float(s_b.detach()) / n_b)
        bests.append((best_nll, per_d))
        restarts.append({"restart": r_i,
                         "min_batch_nll": float(min(traj)),
                         "final_batch_nll": float(traj[-1]),
                         "best_step": int(traj.index(min(traj)))})
    best_nll, per_d = min(bests, key=lambda x: x[0])
    return best_nll, per_d, restarts


def run_depth(model, tokenizer, comp_args, collator, collator_nc,
              dialogs, device, L, out):
    """One depth: sample, merge baseline, oracle optimization, full ctx."""
    turns = L + 2
    pool = [d for d in dialogs if len(d) >= turns]
    rng = random.Random(1234 + L)
    sampled = rng.sample(pool, min(N_SAMPLE, len(pool)))
    items = [{"dialog": d[:turns], "act": [], "is_train": False}
             for d in sampled]

    results = {"L": L, "n_dialogues": len(items),
               "per_dialogue": [], "restarts": []}
    merge_nlls = []
    oracle_nlls = []
    full_nlls = []
    for i in range(0, len(items), batch_size_for(L)):
        chunk = items[i:i + batch_size_for(L)]
        batch = collator(chunk)
        # official merge baseline (uniform logits = zeros)
        set_oracle_logits(torch.zeros(
            batch["input_ids"].shape[0], L, device=device))
        with torch.no_grad():
            fwd_merge = tc.run_forward(model, batch, device,
                                       grad_enabled=False)
        clear_oracle_logits()
        loss, _n = ed.eval_ce_shifted_noeos(
            fwd_merge, batch["labels"].to(device), device)
        merge_batch_nll = float(loss.detach()) / max(_n, 1)
        # full-context reference on the SAME dialogues
        batch_nc = collator_nc(chunk)
        with torch.no_grad():
            out_nc = model(
                input_ids=batch_nc["input_ids"].to(device),
                attention_mask=batch_nc["attention_mask"].to(device))
        loss_nc, n_nc = ed.eval_ce_shifted_noeos(
            out_nc, batch_nc["labels"].to(device), device)
        full_batch_nll = float(loss_nc.detach()) / max(n_nc, 1)
        # oracle optimization
        best_nll, per_d, restarts = optimize_batch(
            model, batch, device, L, seed=5678 + L * 31 + i)
        merge_nlls.append(merge_batch_nll)
        oracle_nlls.append(best_nll)
        full_nlls.append(full_batch_nll)
        results["restarts"].append({"batch": i // batch_size_for(L),
                                    "merge_nll": merge_batch_nll,
                                    "full_nll": full_batch_nll,
                                    "oracle_nll": best_nll,
                                    "restart_diag": restarts})
        for j in range(len(chunk)):
            results["per_dialogue"].append({
                "idx": i + j,
                "merge_nll": merge_batch_nll,
                "oracle_nll": per_d[j],
            })
        print("L={} batch {} merge={:.4f} oracle={:.4f} full={:.4f} "
              "H={:+.4f}".format(
                  L, i // batch_size_for(L), merge_batch_nll, best_nll,
                  full_batch_nll, merge_batch_nll - best_nll), flush=True)
    m = sum(merge_nlls) / len(merge_nlls)
    o = sum(oracle_nlls) / len(oracle_nlls)
    f = sum(full_nlls) / len(full_nlls)
    results["summary"] = {
        "merge_nll": m, "oracle_nll": o, "full_nll": f,
        "H": m - o, "C": m - f,
        "R": (m - o) / max(m - f, 1e-12)}
    with open(out, "w") as _f:
        json.dump(results, _f, indent=2)
    print("L={} SUMMARY merge={:.4f} oracle={:.4f} full={:.4f} "
          "H={:+.4f} C={:+.4f} R={:.3f} -> {}".format(
              L, m, o, f, m - o, m - f, results["summary"]["R"], out),
          flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name-or-path",
                    default="/root/autodl-tmp/llama-7b-hf")
    ap.add_argument("--dialog-mirror",
                    default="/root/autodl-tmp/dailydialog_mirror/"
                            "ijcnlp_dailydialog")
    ap.add_argument("--foundation",
                    default="/root/autodl-tmp/result/dialog/llama-7b-no")
    ap.add_argument("--official-adapter",
                    default="/root/autodl-tmp/result/dialog/"
                            "llama-7b-no-online-merge_recur-ntok2")
    ap.add_argument("--out-dir",
                    default="/root/autodl-tmp/oracle_ceiling")
    ap.add_argument("--depths", default="1,2,4")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    device = torch.device("cuda", a.gpu)
    import os
    os.environ["DIALOG_MIRROR"] = a.dialog_mirror
    import types
    args = types.SimpleNamespace(
        arm="ours", model_name_or_path=a.model_name_or_path,
        dialog_mirror=a.dialog_mirror, relative_embedding="skip",
        lora_r=8, z_dim=128, rpbe_seed=0, sketch_dim=64, gamma_hidden=64,
        foundation=a.foundation, official_adapter=a.official_adapter,
        official_host=True)

    tokenizer, model = build_merge_model(args, device)
    dialog, comp_args = ed.build_eval_dataset(
        args, tokenizer, False, online=True, comp_type="online")
    collator = ed.build_collator(dialog, tokenizer, comp_args,
                                 comp_type="online", sum_recur=True)
    dialog_nc, comp_args_nc = ed.build_eval_dataset(
        args, tokenizer, False, online=False, comp_type="online")
    collator_nc = ed.build_collator(dialog_nc, tokenizer, comp_args_nc,
                                    comp_type="online", sum_recur=False)
    eval_dialogs = dialog.testset["dialog"]

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for L in [int(x) for x in a.depths.split(",")]:
        res = run_depth(model, tokenizer, comp_args, collator,
                        collator_nc, eval_dialogs, device, L,
                        out_dir / "oracle_L{}.json".format(L))
        all_summaries[str(L)] = res["summary"]
        if L == 1:
            # sanity: no weighting freedom -> oracle == merge exactly
            m1 = res["summary"]["merge_nll"]
            o1 = res["summary"]["oracle_nll"]
            assert abs(m1 - o1) < 1e-6, (
                "L=1 sanity failed: merge {:.6f} != oracle {:.6f}"
                .format(m1, o1))
            print("L=1 SANITY PASS: oracle == merge bit-for-bit",
                  flush=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(json.dumps(all_summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
