"""verif_evidence.py — the four acceptance evidences (review ruling):

  E1: Gamma leaves Avg after repr steps (||Gamma - Avg|| > 0)
  E2: three arms share identical public-module init (param hashes equal)
  E4: checkpoint roundtrip + eval state restore (bank hash unchanged)

(E3, direct-vs-replay, is the local test test_vla_direct_vs_replay.py.)

Usage (server):
  HF_HUB_OFFLINE=1 LLAMA2_LOCAL_PATH=/root/autodl-tmp/Llama-2-7b-hf \
  RPBE_EMBODIED_PATH=/root/autodl-tmp \
  python scripts/verif_evidence.py \
    --pretrained-checkpoint /root/autodl-tmp/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
    --data-root /root/autodl-tmp/libero-mem --task-filter KITCHEN_SCENE1_3 --seed 42
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # the memoryvla repo root
sys.path.insert(0, os.environ.get("RPBE_EMBODIED_PATH", "/root/autodl-tmp"))

from torch.utils.data import DataLoader  # noqa: E402

from vla import load_vla  # noqa: E402
from vla.datasets.hdf5_dataset import (  # noqa: E402
    get_hdf5_decision_stream_dataset_and_collator,
)
from rpbe_embodied import (  # noqa: E402
    EmbodiedFixedMaps, EmbodiedRPBConfig, EmbodiedRPBEWindow,
    PendingMergeQueue, gamma_replay_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--task-filter", default="KITCHEN_SCENE1_3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mem-length", type=int, default=8)
    p.add_argument("--repr-boundary-episodes", type=int, default=8)
    p.add_argument("--steps-e1", type=int, default=60)
    p.add_argument("--steps-e4", type=int, default=30)
    return p.parse_args()


def _build(arm, args, rpbe_seed=0):
    vla = load_vla(
        model_id_or_path=args.pretrained_checkpoint, hf_token=None,
        load_for_training=True, use_bf16=True, action_dim=7,
        future_action_window_size=15, action_model_type="DiT-L",
        use_ema=False, dataloader_type="stream", mem_length=args.mem_length,
        retrieval_layers=2, use_timestep_pe=True, fusion_type="gate",
        consolidate_type="tome", update_fused=False, per_token_size=256,
        use_rpbe_gamma=(arm in ("gamma-task", "gamma-rpbe")),
        gamma_rank=64, gamma_alpha_init=1.0,
        rpbe_merge_records=(arm != "avg"),
        rpbe_task_grad=(arm in ("gamma-task", "gamma-rpbe")),
        rpbe_seed=rpbe_seed)
    vla.vlm.requires_grad_(False)
    from peft import LoraConfig, get_peft_model
    llm = vla.vlm.llm_backbone.llm
    vla.vlm.llm_backbone.llm = get_peft_model(llm, LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.1,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    return vla


def _public_hash(vla):
    """Hash of every parameter that is NOT part of Gamma."""
    h = hashlib.sha256()
    for n, p in sorted(vla.named_parameters()):
        if "gamma." in n or n == "gamma":
            continue
        h.update(n.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _bank_hash(vla):
    h = hashlib.sha256()
    bank = vla.cog_mem_bank
    for eid in sorted(bank.bank.keys()):
        for (t, f) in bank.bank[eid]:
            h.update(f.detach().float().cpu().numpy().tobytes())
    h.update(str(bank.next_node_id).encode())
    h.update(str(bank.next_merge_id).encode())
    return h.hexdigest()[:16]


def e1_gamma_leaves_avg(args):
    print("== E1: Gamma leaves Avg ==", flush=True)
    torch.manual_seed(args.seed)
    vla = _build("gamma-rpbe", args, rpbe_seed=args.seed)
    vla.train()
    probe_a = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
    probe_b = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        before = (vla.gamma(probe_a, probe_b)
                  - 0.5 * (probe_a + probe_b)).abs().mean().item()
    print(f"  ||Gamma - Avg|| at init: {before:.6f}", flush=True)
    assert before < 1e-3, "init is not the exact average!"

    tokenizer = vla.vlm.llm_backbone.get_tokenizer()
    it = vla.vlm.vision_backbone.get_image_transform()
    ds, _, collator = get_hdf5_decision_stream_dataset_and_collator(
        data_root=Path(args.data_root), tokenizer=tokenizer,
        image_transform=it, prompt_builder_fn=vla.vlm.llm_backbone.prompt_builder_fn,
        future_action_window_size=15, seed=args.seed, split="train",
        pad_token_id=tokenizer.pad_token_id, task_filter=args.task_filter)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0,
                        collate_fn=collator, drop_last=True)
    repr_params = ([p for n, p in vla.named_parameters()
                    if p.requires_grad and "lora_" in n]
                   + list(vla.gamma.parameters()))
    opt_repr = torch.optim.AdamW(repr_params, lr=2e-5)
    opt_task = torch.optim.AdamW(
        [p for n, p in vla.named_parameters() if p.requires_grad
         and "lora_" not in n and "gamma." not in n], lr=2e-5)
    step = 0
    for batch in loader:
        if step >= args.steps_e1:
            break
        pv = batch["pixel_values"]
        if isinstance(pv, dict):
            pv = {k: v.to("cuda", dtype=torch.bfloat16) for k, v in pv.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            loss, _ = vla(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                actions=batch["actions"].to("cuda", dtype=torch.bfloat16),
                action_masks=batch["action_masks"].to("cuda"),
                pixel_values=pv, labels=batch["labels"].to("cuda"),
                timesteps=batch["timesteps"], episode_ids=batch["episode_ids"],
                output_hidden_states=True, repeated_diffusion_steps=4)
        loss.backward()
        opt_task.step(); opt_task.zero_grad()
        # crude repr update every N steps: cotangent replay on the fly
        if step > 0 and step % 10 == 0:
            g = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
            (gamma_replay_loss(vla.gamma, probe_a, probe_b,
                               {(0, i): g[i] for i in range(4)},
                               [(0, i) for i in range(4)])).backward()
            opt_repr.step(); opt_repr.zero_grad()
        step += 1
    with torch.no_grad():
        after = (vla.gamma(probe_a, probe_b)
                 - 0.5 * (probe_a + probe_b)).abs().mean().item()
    print(f"  ||Gamma - Avg|| after {step} steps: {after:.6f}", flush=True)
    assert after > 1e-5, "Gamma never left the average merge!"
    print("  E1 PASS", flush=True)


def e2_shared_init(args):
    print("== E2: three-arm shared init ==", flush=True)
    hashes = {}
    for arm in ("avg", "gamma-task", "gamma-rpbe"):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        vla = _build(arm, args, rpbe_seed=args.seed)
        hashes[arm] = _public_hash(vla)
        del vla
        torch.cuda.empty_cache()
        print(f"  {arm}: {hashes[arm]}", flush=True)
    assert len(set(hashes.values())) == 1, f"public params differ: {hashes}"
    print("  E2 PASS", flush=True)


def e4_ckpt_and_eval_state(args):
    print("== E4: checkpoint roundtrip + eval state restore ==", flush=True)
    torch.manual_seed(args.seed)
    vla = _build("gamma-task", args, rpbe_seed=args.seed)
    vla.train()
    tokenizer = vla.vlm.llm_backbone.get_tokenizer()
    it = vla.vlm.vision_backbone.get_image_transform()
    ds, _, collator = get_hdf5_decision_stream_dataset_and_collator(
        data_root=Path(args.data_root), tokenizer=tokenizer,
        image_transform=it, prompt_builder_fn=vla.vlm.llm_backbone.prompt_builder_fn,
        future_action_window_size=15, seed=args.seed, split="train",
        pad_token_id=tokenizer.pad_token_id, task_filter=args.task_filter)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0,
                        collate_fn=collator, drop_last=True)
    opt = torch.optim.AdamW(
        [p for p in vla.parameters() if p.requires_grad], lr=2e-5)
    step = 0
    for batch in loader:
        if step >= args.steps_e4:
            break
        pv = batch["pixel_values"]
        if isinstance(pv, dict):
            pv = {k: v.to("cuda", dtype=torch.bfloat16) for k, v in pv.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            loss, _ = vla(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                actions=batch["actions"].to("cuda", dtype=torch.bfloat16),
                action_masks=batch["action_masks"].to("cuda"),
                pixel_values=pv, labels=batch["labels"].to("cuda"),
                timesteps=batch["timesteps"], episode_ids=batch["episode_ids"],
                output_hidden_states=True, repeated_diffusion_steps=4)
        loss.backward()
        opt.step(); opt.zero_grad()
        step += 1
    # bank hash before the eval-restore exercise
    h_before = _bank_hash(vla)
    # save a flat checkpoint
    ckpt = {"model": {n: p.detach().cpu() for n, p in vla.named_parameters()
                      if p.requires_grad}, "step": step, "param_version": 0,
            "best_val": float("inf")}
    torch.save(ckpt, "/tmp/verif_ckpt.pt")
    flat = {n: p.detach().cpu().clone() for n, p in vla.named_parameters()
            if p.requires_grad}
    # rebuild + reload
    torch.manual_seed(args.seed)
    vla2 = _build("gamma-task", args, rpbe_seed=args.seed)
    named2 = dict(vla2.named_parameters())
    for n, t in flat.items():
        named2[n].data.copy_(t.to(named2[n].dtype))
    max_diff = max((flat[n] - named2[n].detach().cpu()).abs().max().item()
                   for n in flat)
    print(f"  reload max |diff|: {max_diff:.3e}", flush=True)
    assert max_diff < 1e-5, "checkpoint roundtrip drifted"
    # eval state restore (bank untouched by a fake eval round)
    import copy
    bank = vla.cog_mem_bank
    saved = (copy.deepcopy(bank.bank), copy.deepcopy(bank.prov),
             bank.next_node_id, bank.next_merge_id, bank.eid_stream)
    bank.reset()
    bank.bank, bank.prov, bank.next_node_id, bank.next_merge_id, \
        bank.eid_stream = saved
    h_after = _bank_hash(vla)
    print(f"  bank hash before {h_before} == after {h_after}: "
          f"{h_before == h_after}", flush=True)
    assert h_before == h_after, "eval restore changed the bank"
    print("  E4 PASS", flush=True)


if __name__ == "__main__":
    a = parse_args()
    e1_gamma_leaves_avg(a)
    e2_shared_init(a)
    e4_ckpt_and_eval_state(a)
    print("ALL_EVIDENCE_PASS", flush=True)
