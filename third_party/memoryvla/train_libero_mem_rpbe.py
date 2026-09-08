"""
train_libero_mem_rpbe.py — single-GPU MemoryVLA training on LIBERO-Mem
with the RPBE plugin, following the OFFICIAL dense training protocol.

Protocol (2026-09-08 protocol-fix ruling):
  * dense HDF5 sliding window (hdf5_dataset.py emits one row per physical
    frame; a 276-frame demo yields 276 samples).
  * official gripper labels {0,1} (data/env -1=open,+1=close -> model
    1=open,0=close); gripper dim is NOT BOUNDS-normalized (mask[6]=False).
  * --max-steps counts OPTIMIZER steps (not microbatch steps).
  * LoRA joins the normal task optimizer and updates every grad-accum block
    in ALL three arms (avg host control = dense LoRA).
  * Gamma is updated ONLY at the shared repr macro boundary by the trainer's
    local replay (it gets no graph gradient: bank merge is @torch.no_grad).
  * mem_length=16, repr_boundary_episodes=1 under dense (an episode has many
    cuts), validation saves/restores cog+per banks and CPU/CUDA RNG with a
    fixed eval noise stream so eval never perturbs the training RNG.

Arms:
  avg         : official average merge host control (dense LoRA)
  gamma-task  : Gamma merge + task-gradient replay at macro boundaries
  gamma-rpbe  : Gamma merge + task + RPBE dual-adjoint replay at macro bounds
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("RPBE_EMBODIED_PATH", "/root/autodl-tmp"))

from peft import LoraConfig, get_peft_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from vla import load_vla  # noqa: E402
from vla.datasets.hdf5_dataset import (  # noqa: E402
    get_hdf5_decision_stream_dataset_and_collator,
)

from rpbe_embodied import (  # noqa: E402
    EmbodiedFixedMaps, EmbodiedRPBConfig, EmbodiedRPBEWindow,
    PendingMergeQueue, gamma_replay_loss,
)


def arm_flags(arm):
    """Semantic booleans for every arm string.  Single source of truth."""
    return dict(
        is_gamma=arm in ("gamma-task", "gamma-rpbe"),
        is_rpbe=arm == "gamma-rpbe",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", choices=["avg", "gamma-task", "gamma-rpbe"],
                   default="avg")
    p.add_argument("--task-filter", default="KITCHEN_SCENE1_3")
    # NOTE: --max-steps counts OPTIMIZER steps (each = grad_accum micro-batches).
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=500,
                   help="run val-split action-loss evaluation every N "
                        "OPTIMIZER steps (0 = never)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=100,
                   help="warmup in OPTIMIZER steps")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--future-action-window-size", type=int, default=15)
    p.add_argument("--mem-length", type=int, default=16,
                   help="CogMemBank capacity (dense protocol: 16 frames)")
    p.add_argument("--repeated-diffusion-steps", type=int, default=4)
    p.add_argument("--resume-from", default="")
    # RPBE
    p.add_argument("--kf-variant", choices=["full_dual", "diag"],
                   default="full_dual")
    p.add_argument("--kf-min-abs", type=int, default=64,
                   help="min unique merges per RPBE window")
    p.add_argument("--lambda-rpbe", type=float, default=0.0,
                   help="frozen after calibration")
    # macro boundary: with dense protocol one episode already yields many cuts,
    # so every finished episode is one boundary.
    p.add_argument("--repr-boundary-episodes", type=int, default=1)
    # gamma LR scheduler in units of ACTUAL gamma steps (param_version).
    p.add_argument("--repr-warmup-steps", type=int, default=30,
                   help="warmup length for the gamma scheduler in gamma steps")
    p.add_argument("--repr-total-steps", type=int, default=1200,
                   help="cosine denominator for the gamma scheduler in gamma "
                        "steps (dense: ~1 gamma step per episode)")
    return p.parse_args()


def build_vla(args: argparse.Namespace):
    print("== loading MemoryVLA ==", flush=True)
    flags = arm_flags(args.arm)
    vla = load_vla(
        model_id_or_path=args.pretrained_checkpoint,
        hf_token=None,
        load_for_training=True,
        use_bf16=True,
        action_dim=7,
        future_action_window_size=args.future_action_window_size,
        action_model_type="DiT-L",
        use_ema=False,
        dataloader_type="stream",
        mem_length=args.mem_length,
        retrieval_layers=2,
        use_timestep_pe=True,
        fusion_type="gate",
        consolidate_type="tome",
        update_fused=False,
        per_token_size=256,
        use_rpbe_gamma=flags["is_gamma"],
        gamma_rank=64,
        # alpha=1 + U=0 (review ruling B1): training start == official
        # AvgMerge but dL/dU = alpha*h != 0 opens the learning path.  alpha=0
        # would freeze every Gamma parameter at zero gradient forever.
        gamma_alpha_init=1.0,
        rpbe_merge_records=flags["is_gamma"],
        rpbe_task_grad=flags["is_gamma"],
    )
    vla.vlm.requires_grad_(False)
    llm = vla.vlm.llm_backbone.llm
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules="all-linear",
        task_type="CAUSAL_LM")
    vla.vlm.llm_backbone.llm = get_peft_model(llm, lora_config)

    for name, param in vla.named_parameters():
        if "lora_" in name:
            assert param.requires_grad, name
        elif name.startswith("vlm."):
            assert not param.requires_grad, name
    return vla, lora_config


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    flags = arm_flags(args.arm)
    IS_GAMMA, IS_RPBE = flags["is_gamma"], flags["is_rpbe"]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    vla, lora_config = build_vla(args)
    vla.train()

    tokenizer = vla.vlm.llm_backbone.get_tokenizer()
    image_transform = vla.vlm.vision_backbone.get_image_transform()
    train_dataset, dataset_statistics, collator = (
        get_hdf5_decision_stream_dataset_and_collator(
            data_root=Path(args.data_root), tokenizer=tokenizer,
            image_transform=image_transform,
            prompt_builder_fn=vla.vlm.llm_backbone.prompt_builder_fn,
            future_action_window_size=args.future_action_window_size,
            seed=args.seed, split="train",
            pad_token_id=tokenizer.pad_token_id,
            task_filter=args.task_filter))
    print("train episodes:", len(train_dataset), flush=True)
    n_train_rows = sum(max(0, ep["T"]) for ep in train_dataset.episodes)
    print(f"train rows/epoch (dense): {n_train_rows}", flush=True)

    # num_workers MUST be 0: worker processes would each shuffle the episode
    # stream independently and interleave rows across episodes.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=0,
        collate_fn=collator, drop_last=True)

    val_loader = None
    if args.eval_every > 0:
        val_dataset, _, _ = get_hdf5_decision_stream_dataset_and_collator(
            data_root=Path(args.data_root), tokenizer=tokenizer,
            image_transform=image_transform,
            prompt_builder_fn=vla.vlm.llm_backbone.prompt_builder_fn,
            future_action_window_size=args.future_action_window_size,
            seed=args.seed, split="val",
            pad_token_id=tokenizer.pad_token_id,
            task_filter=args.task_filter)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, num_workers=0,
            collate_fn=collator, drop_last=True)
        print("val episodes:", len(val_dataset), flush=True)

    def _snapshot_bank(bank):
        return {
            "bank": copy.deepcopy(bank.bank),
            "prov": copy.deepcopy(bank.prov),
            "merge_log": copy.deepcopy(bank.merge_log),
            "next_node_id": bank.next_node_id,
            "next_merge_id": bank.next_merge_id,
            "eid_stream": bank.eid_stream,
        }

    def _restore_bank(bank, snap):
        bank.bank = snap["bank"]
        bank.prov = snap["prov"]
        bank.merge_log = snap["merge_log"]
        bank.next_node_id = snap["next_node_id"]
        bank.next_merge_id = snap["next_merge_id"]
        bank.eid_stream = snap["eid_stream"]

    def run_eval(optimizer_step: int) -> float:
        """Val-split action loss over min(40, len) batches, with full
        isolation: cog+per bank state and CPU/CUDA/python RNG are snapshotted
        before and restored after, and eval draws its own FIXED seed so it
        can never perturb the training random stream."""
        cog = vla.cog_mem_bank
        per = vla.per_mem_bank
        cog_snap = _snapshot_bank(cog)
        per_snap = _snapshot_bank(per)
        rng_snapshot = (copy.deepcopy(torch.get_rng_state()),
                        copy.deepcopy(torch.cuda.get_rng_state()),
                        np.random.get_state())
        losses = []
        try:
            vla.eval()
            cog.reset()
            per.reset()
            torch.manual_seed(1234 + optimizer_step % 1000)
            torch.cuda.manual_seed(1234 + optimizer_step % 1000)
            with torch.no_grad():
                for batch in val_loader:
                    if len(losses) >= 40:
                        break
                    pixel_values = batch["pixel_values"]
                    if isinstance(pixel_values, dict):
                        pixel_values = {k: v.to("cuda", dtype=torch.bfloat16)
                                        for k, v in pixel_values.items()}
                    else:
                        pixel_values = pixel_values.to(
                            "cuda", dtype=torch.bfloat16)
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=True):
                        loss, _ = vla(
                            input_ids=batch["input_ids"].to("cuda"),
                            attention_mask=batch["attention_mask"].to("cuda"),
                            actions=batch["actions"].to(
                                "cuda", dtype=torch.bfloat16),
                            action_masks=batch["action_masks"].to("cuda"),
                            pixel_values=pixel_values,
                            labels=batch["labels"].to("cuda"),
                            timesteps=batch["timesteps"],
                            episode_ids=batch["episode_ids"],
                            output_hidden_states=True,
                            repeated_diffusion_steps=args.repeated_diffusion_steps,
                        )
                    losses.append(loss.item())
        finally:
            _restore_bank(cog, cog_snap)
            _restore_bank(per, per_snap)
            torch.set_rng_state(rng_snapshot[0])
            torch.cuda.set_rng_state(rng_snapshot[1])
            np.random.set_state(rng_snapshot[2])
            vla.train()
        if losses:
            mean = sum(losses) / len(losses)
            print(f"[eval @ opt {optimizer_step}] val action loss "
                  f"{mean:.4f} ({len(losses)} batches)", flush=True)
            return mean
        return float("inf")

    # ---- parameter split ----
    # opt_task: cog/per banks + per_compr + DiT + LoRA -> dense (every block).
    # opt_gamma: gamma merge operator only -> macro boundary replay.
    lora_params = [p for n, p in vla.named_parameters()
                   if p.requires_grad and "lora_" in n]
    lora_ids = {id(p) for p in lora_params}
    task_modules = [vla.cog_mem_bank, vla.per_mem_bank, vla.per_compr,
                    vla.action_model]
    task_params = [p for m in task_modules for p in m.parameters()
                   if p.requires_grad and id(p) not in lora_ids]
    task_params += lora_params
    gamma_params = (list(vla.gamma.parameters())
                    if vla.gamma is not None else [])
    opt_task = torch.optim.AdamW(task_params, lr=args.lr)
    if gamma_params:
        opt_gamma = torch.optim.AdamW(gamma_params, lr=args.lr)
    else:
        opt_gamma = None

    # cosine decay w/ warmup in OPTIMIZER-step units for the dense (task+LoRA)
    # optimizer.
    def lr_lambda_task(t):
        if t < args.warmup_steps:
            return t / max(1, args.warmup_steps)
        progress = (t - args.warmup_steps) / max(
            1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched_task = torch.optim.lr_scheduler.LambdaLR(opt_task, lr_lambda_task)
    # gamma scheduler indexed by param_version (gamma step count).
    def lr_lambda_gamma(rs):
        if rs < args.repr_warmup_steps:
            return rs / max(1, args.repr_warmup_steps)
        progress = (rs - args.repr_warmup_steps) / max(
            1, args.repr_total_steps - args.repr_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched_gamma = None
    if opt_gamma is not None:
        sched_gamma = torch.optim.lr_scheduler.LambdaLR(opt_gamma,
                                                        lr_lambda_gamma)
    print(f"task params: {sum(p.numel() for p in task_params)/1e6:.2f}M | "
          f"gamma params: {sum(p.numel() for p in gamma_params)/1e6:.2f}M",
          flush=True)

    # ---- RPBE machinery (gamma arms only) ----
    rpbe_cfg = EmbodiedRPBConfig(
        kf_variant=args.kf_variant, kf_min_abs=args.kf_min_abs,
        lambda_rpbe=args.lambda_rpbe, rpbe_seed=args.seed)
    maps = EmbodiedFixedMaps(rpbe_cfg) if IS_RPBE else None
    queue = PendingMergeQueue() if IS_GAMMA else None
    window = EmbodiedRPBEWindow(
        variant=rpbe_cfg.kf_variant, eps=rpbe_cfg.ridge_eps,
        min_abs=rpbe_cfg.kf_min_abs) if IS_RPBE else None
    task_cotangents: dict = {}
    rpbe_cotangents: dict = {}
    rpbe_pending_loss: list = []
    merge_registry: dict = {}
    merge_id_map: dict = {}

    optimizer_step = 0
    micro_step = 0
    episodes_since_boundary = 0
    last_eid = None
    t0 = time.time()
    print("== training loop start (arm={}) ==".format(args.arm), flush=True)

    def feed_merges_and_futures(batch):
        """Consume merge_log -> queue.register; offer current-decision
        futures; drain finished episodes.  Causal order: register this
        batch's merges first, then offer, then drain completed episodes."""
        nonlocal last_eid, episodes_since_boundary
        bank = vla.cog_mem_bank
        eids = [int(e) for e in batch["episode_ids"]]
        for rec in bank.merge_log:
            queue.register(rec)
            merge_registry[(rec.episode_id, rec.node_id)] = rec
            merge_id_map[(rec.episode_id, rec.merge_id)] = rec.node_id
        bank.merge_log = []
        with torch.no_grad():
            vf = vla.vlm.vision_feats.detach() \
                if hasattr(vla.vlm, "vision_feats") else None
        instrs = (batch["instruction"] if isinstance(batch["instruction"], list)
                  else [batch["instruction"]] * len(eids))
        for i in range(len(eids)):
            ctx = {
                "vision_feat": vf[i].mean(0).detach().cpu()
                if vf is not None else torch.zeros(2176),
                "instruction": instrs[i],
                "delta_s": 1.0,
                "horizon": 1,
            }
            y = batch["actions"][i].reshape(-1)
            queue.offer(eids[i], int(batch["timesteps"][i]), ctx,
                        y.detach().cpu())
        n_done = 0
        for eid in eids:
            if last_eid is not None and eid != last_eid:
                rows = queue.drain_episode(last_eid)
                if window is not None and rows:
                    for r in rows:
                        r.outcome = maps.pv(r.context, r.outcome)
                    window.add(rows)
                n_done += 1
            last_eid = eid
        episodes_since_boundary += n_done

    def _gamma_boundary():
        """One gamma macro-boundary update (dense protocol: repr_boundary
        episodes per boundary).  Gamma gets no graph grads; its ONLY grads
        are the local task / RPBE replay losses computed here."""
        nonlocal window, task_cotangents, rpbe_cotangents, rpbe_pending_loss
        if not IS_GAMMA or vla.gamma is None:
            return
        # close RPBE statistics window -> cache cotangents
        if window is not None and window.n_unique_cuts > 0:
            if window.ready():
                j, g_by_cut, diag = window.close()
                if g_by_cut:
                    for k, g in g_by_cut.items():
                        eid, mid, _ = k
                        nid = merge_id_map.get((eid, mid))
                        if nid is not None:
                            rpbe_cotangents[(eid, nid)] = g
                    rpbe_pending_loss.append(j)
                print(f"[window close] J={j:.4f} "
                      f"cuts={diag.get('n_unique_cuts')} "
                      f"episodes={diag.get('n_unique_episodes')} "
                      f"censored={queue.n_censored}", flush=True)
            else:
                n_discard = window.discard()
                print(f"[window discard] underfull "
                      f"({n_discard} cuts < {rpbe_cfg.kf_min_abs})",
                      flush=True)
            window = EmbodiedRPBEWindow(
                variant=rpbe_cfg.kf_variant, eps=rpbe_cfg.ridge_eps,
                min_abs=rpbe_cfg.kf_min_abs)
        # task replay (independent key set from RPBE)
        keys = [k for k in task_cotangents if k in merge_registry]
        if keys:
            m_a = torch.stack(
                [merge_registry[k].left_state for k in keys]
            ).to("cuda", dtype=torch.bfloat16)
            m_b = torch.stack(
                [merge_registry[k].right_state for k in keys]
            ).to("cuda", dtype=torch.bfloat16)
            l_task = gamma_replay_loss(vla.gamma, m_a, m_b, task_cotangents,
                                       keys)
            l_task.backward()
            print(f"[gamma step] task replay {l_task.item():.4f}",
                  flush=True)
        # rpbe replay (independent key set)
        rkeys = [k for k in rpbe_cotangents if k in merge_registry]
        if rkeys and args.lambda_rpbe > 0:
            m_a_r = torch.stack(
                [merge_registry[k].left_state for k in rkeys]
            ).to("cuda", dtype=torch.bfloat16)
            m_b_r = torch.stack(
                [merge_registry[k].right_state for k in rkeys]
            ).to("cuda", dtype=torch.bfloat16)
            l_rpbe = gamma_replay_loss(vla.gamma, m_a_r, m_b_r,
                                       rpbe_cotangents, rkeys)
            (args.lambda_rpbe * l_rpbe).backward()
            print(f"[gamma step] rpbe replay "
                  f"{args.lambda_rpbe * l_rpbe.item():.4f}", flush=True)
        if opt_gamma is not None:
            torch.nn.utils.clip_grad_norm_(gamma_params, args.grad_clip)
            opt_gamma.step()
            opt_gamma.zero_grad()
            sched_gamma.step()
        vla.cog_mem_bank.param_version += 1
        task_cotangents = {}
        rpbe_cotangents = {}
        rpbe_pending_loss = []

    def _log_lora_norm():
        b_norms = []
        n_b = 0
        for n, p in vla.named_parameters():
            if "lora_B" in n:
                n_b += 1
                b_norms.append(float(p.detach().float().abs().mean()))
        if b_norms:
            mean_b = sum(b_norms) / len(b_norms)
            nz = sum(1 for v in b_norms if v > 0.0)
            lr_now = opt_task.param_groups[-1]["lr"]
            print(f"[lora] pv={vla.cog_mem_bank.param_version} "
                  f"lr={lr_now:.2e} mean|B|={mean_b:.3e} "
                  f"nzB={nz}/{n_b}", flush=True)

    def _opt_state(opt):
        """Optimizer state moved to CPU for a disk-friendly checkpoint."""
        st = opt.state_dict()
        for pstate in st["state"].values():
            for k, v in pstate.items():
                if torch.is_tensor(v):
                    pstate[k] = v.detach().cpu()
        return st

    def _ckpt_dict():
        payload = {
            "model": {n: p.detach().cpu()
                      for n, p in vla.named_parameters()
                      if p.requires_grad},
            "lora_config": {"r": lora_config.r,
                            "lora_alpha": lora_config.lora_alpha,
                            "lora_dropout": lora_config.lora_dropout},
            "arm": args.arm,
            "step": optimizer_step,
            "micro_step": micro_step,
            "param_version": vla.cog_mem_bank.param_version,
            "mem_length": args.mem_length,
            "lambda_rpbe": args.lambda_rpbe,
            "seed": args.seed,
            "task_filter": args.task_filter,
            "best_val": best_val,
            "opt_task": _opt_state(opt_task),
        }
        if opt_gamma is not None:
            payload["opt_gamma"] = _opt_state(opt_gamma)
            payload["sched_gamma"] = sched_gamma.state_dict()
        payload["sched_task"] = sched_task.state_dict()
        return payload

    best_val = float("inf")
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location="cpu",
                        weights_only=False)
        named = dict(vla.named_parameters())
        unexpected = 0
        for n, t in ck["model"].items():
            if n in named and named[n].requires_grad:
                named[n].data.copy_(t.to(named[n].dtype))
            else:
                unexpected += 1
        optimizer_step = ck.get("step", 0)
        micro_step = ck.get("micro_step", optimizer_step * args.grad_accum)
        best_val = ck.get("best_val", float("inf"))
        vla.cog_mem_bank.param_version = ck.get("param_version", 0)
        # restore optimizer + scheduler state so a mid-run resume does not
        # reset Adam momentum / LR phase
        if "opt_task" in ck:
            opt_task.load_state_dict(ck["opt_task"])
            sched_task.load_state_dict(ck.get("sched_task", sched_task.state_dict()))
            if opt_gamma is not None and "opt_gamma" in ck:
                opt_gamma.load_state_dict(ck["opt_gamma"])
                sched_gamma.load_state_dict(ck.get("sched_gamma",
                                                   sched_gamma.state_dict()))
        print(f"resumed from {args.resume_from} @ opt {optimizer_step} "
              f"(micro {micro_step}, pv {vla.cog_mem_bank.param_version}, "
              f"unexpected keys: {unexpected})", flush=True)
        # fast-forward the data stream so the batch sequence matches the
        # checkpoint exactly (dense HDF5 stream is deterministic per seed)
        n_rows = micro_step * args.batch_size
        it = iter(train_loader)
        consumed = 0
        while consumed < n_rows:
            try:
                b = next(it)
                consumed += len(b["input_ids"])
            except StopIteration:
                raise RuntimeError(
                    "resume step beyond dataset epoch boundary")
        train_loader = it

    # ---- main loop ----
    it = iter(train_loader)
    while optimizer_step < args.max_steps:
        batch = next(it)
        micro_step += 1

        pixel_values = batch["pixel_values"]
        if isinstance(pixel_values, dict):
            pixel_values = {k: v.to("cuda", dtype=torch.bfloat16)
                            for k, v in pixel_values.items()}
        else:
            pixel_values = pixel_values.to("cuda", dtype=torch.bfloat16)

        # snapshot merged-leaf features for gamma task cotangents (plan §25)
        leaf_snapshot = []
        if IS_GAMMA:
            for eid, entries in vla.cog_mem_bank.bank.items():
                prov = vla.cog_mem_bank.prov.get(eid, [])
                for (_, f), meta in zip(entries, prov):
                    if meta[4] and f.requires_grad:
                        leaf_snapshot.append((f, eid, meta[0]))

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            loss, _ = vla(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                actions=batch["actions"].to("cuda", dtype=torch.bfloat16),
                action_masks=batch["action_masks"].to("cuda"),
                pixel_values=pixel_values,
                labels=batch["labels"].to("cuda"),
                timesteps=batch["timesteps"],
                episode_ids=batch["episode_ids"],
                output_hidden_states=True,
                repeated_diffusion_steps=args.repeated_diffusion_steps,
            )
        (loss / args.grad_accum).backward()

        if IS_GAMMA:
            for f, eid, node_id in leaf_snapshot:
                if f.grad is not None:
                    key = (eid, node_id)
                    g = f.grad.detach().clone().reshape(-1)
                    task_cotangents[key] = (task_cotangents.get(
                        key, torch.zeros_like(g)) + g)
            vla.cog_mem_bank.refresh_leafs()

        # gamma arms feed merges/futures every microbatch (tracks episodes)
        if IS_GAMMA:
            feed_merges_and_futures(batch)

        # dense optimizer step every grad_accum micro-batches
        if micro_step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(task_params, args.grad_clip)
            opt_task.step()
            opt_task.zero_grad()
            sched_task.step()
            optimizer_step += 1

            if args.log_every > 0 and optimizer_step % args.log_every == 0:
                dt = time.time() - t0
                diag = ""
                if IS_GAMMA and queue is not None:
                    diag = (f" | ep_since {episodes_since_boundary}"
                            f" censored {queue.n_censored}"
                            f" window_cuts {window.n_unique_cuts if window else '-'}")
                print(f"opt {optimizer_step}/{args.max_steps} "
                      f"(micro {micro_step}) | loss {loss.item():.4f}"
                      f"{diag} | {dt/60:.1f} min", flush=True)

            # gamma macro boundary (dense: every repr_boundary_episodes)
            if IS_GAMMA and episodes_since_boundary >= args.repr_boundary_episodes:
                _gamma_boundary()
                episodes_since_boundary = 0

            if args.eval_every > 0 and optimizer_step % args.eval_every == 0:
                val_loss = run_eval(optimizer_step)
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(_ckpt_dict(), run_dir / "best.pt")
                    print(f"[best @ opt {optimizer_step}] val "
                          f"{val_loss:.4f}", flush=True)

            if args.checkpoint_every > 0 and \
                    optimizer_step % args.checkpoint_every == 0:
                torch.save(_ckpt_dict(), run_dir / "latest.pt")
                print(f"latest saved @ opt {optimizer_step}", flush=True)

    torch.save(_ckpt_dict(), run_dir / "checkpoint.pt")
    with open(run_dir / "dataset_statistics.json", "w") as f:
        json.dump(dataset_statistics, f, indent=2)
    print("== training done ==", flush=True)
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
