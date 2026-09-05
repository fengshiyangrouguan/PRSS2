"""
train_libero_mem_rpbe.py — single-GPU MemoryVLA training on LIBERO-Mem
with RPBE plugin (three arms):

  avg        : official average merge, no Gamma, no RPBE (host control)
  gamma-task : Gamma merge, task-gradient replay only
  gamma-rpbe : Gamma merge, task-gradient + RPBE dual-adjoint replay

Plan §4/§25/§26/§27: two optimizers.
  opt_task (retrieval/gate/per_compr/DiT) steps every grad-accum block.
  opt_repr (LoRA + Gamma) steps only at the shared repr macro boundary
  (every --repr-boundary-episodes episodes, IDENTICAL across arms).
  Gamma-Task has NO RPBE window but HAS the same repr update clock.
"""
import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", choices=["avg", "gamma-task", "gamma-rpbe"],
                   default="avg")
    p.add_argument("--task-filter", default="KITCHEN_SCENE1_3")
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=1000,
                   help="run val-split action-loss evaluation every N steps "
                        "(0 = never)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--future-action-window-size", type=int, default=15)
    p.add_argument("--mem-length", type=int, default=8,
                   help="T3 has ~17.5 decisions/episode; 16 yields ~0.2 "
                        "valid cuts/ep (review ruling 4) -> 8")
    p.add_argument("--repeated-diffusion-steps", type=int, default=4)
    p.add_argument("--resume-from", default="")
    # RPBE
    p.add_argument("--kf-variant", choices=["full_dual", "diag"], default="full_dual")
    p.add_argument("--kf-min-abs", type=int, default=64,
                   help="min unique merges per RPBE window (128 was TGN-scale)")
    p.add_argument("--lambda-rpbe", type=float, default=0.0,
                   help="frozen after calibration (Task 8)")
    p.add_argument("--repr-boundary-episodes", type=int, default=8,
                   help="shared repr macro boundary (identical across arms)")
    return p.parse_args()


def build_vla(args: argparse.Namespace):
    print("== loading MemoryVLA ==", flush=True)
    arm = args.arm
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
        use_rpbe_gamma=(arm in ("gamma-task", "gamma-rpbe")),
        gamma_rank=64,
        gamma_alpha_init=0.0,
        rpbe_merge_records=(arm != "avg"),
        rpbe_task_grad=(arm in ("gamma-task", "gamma-rpbe")),
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


def _params_of(module_list, prefix=""):
    out = []
    for m in module_list:
        out += [p for p in m.parameters() if p.requires_grad]
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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

    # num_workers MUST be 0: worker processes would each shuffle the episode
    # stream independently and interleave rows across episodes, breaking the
    # CogMemBank 'stream' accumulation semantics (official code uses 0 too).
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=0,
        collate_fn=collator, drop_last=True)

    # periodic val evaluation (action diffusion loss on the val split)
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

    def run_eval(step: int) -> float:
        """Val-split action loss over min(40, len) batches.

        Review ruling B6: full save/restore of the training memory state
        (banks, provenance, id counters, stream cursor) so validation can
        never truncate a training episode or desync the id numbering.
        Returns the mean val loss (inf on failure)."""
        import copy
        bank = vla.cog_mem_bank
        saved = (copy.deepcopy(bank.bank), copy.deepcopy(bank.prov),
                 bank.next_node_id, bank.next_merge_id, bank.eid_stream)
        vla.eval()
        bank.reset()
        vla.per_mem_bank.reset()
        losses = []
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if n_batches >= 40:
                    break
                pixel_values = batch["pixel_values"]
                if isinstance(pixel_values, dict):
                    pixel_values = {k: v.to("cuda", dtype=torch.bfloat16)
                                    for k, v in pixel_values.items()}
                else:
                    pixel_values = pixel_values.to("cuda", dtype=torch.bfloat16)
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
                losses.append(loss.item())
                n_batches += 1
        # restore the full training memory state (B6)
        bank.bank, bank.prov, bank.next_node_id, bank.next_merge_id, \
            bank.eid_stream = saved
        vla.per_mem_bank.reset()
        vla.train()
        if losses:
            mean = sum(losses) / len(losses)
            print(f"[eval @ {step}] val action loss {mean:.4f} "
                  f"({len(losses)} batches)", flush=True)
            return mean
        return float("inf")

    # --- optimizers: opt_task every block; opt_repr at repr boundaries ---
    task_modules = [vla.cog_mem_bank, vla.per_mem_bank, vla.per_compr,
                    vla.action_model]
    repr_params = ([p for n, p in vla.named_parameters()
                    if p.requires_grad and "lora_" in n]
                   + (list(vla.gamma.parameters()) if vla.gamma is not None else []))
    repr_ids = {id(p) for p in repr_params}
    task_params = [p for m in task_modules for p in m.parameters()
                   if p.requires_grad and id(p) not in repr_ids]
    opt_task = torch.optim.AdamW(task_params, lr=args.lr)
    opt_repr = torch.optim.AdamW(repr_params, lr=args.lr)
    # cosine decay with warmup (review ruling: warmup/cosine were declared
    # but never implemented)
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(
            1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched_task = torch.optim.lr_scheduler.LambdaLR(opt_task, lr_lambda)
    sched_repr = torch.optim.lr_scheduler.LambdaLR(opt_repr, lr_lambda)
    print(f"task params: {sum(p.numel() for p in task_params)/1e6:.2f}M | "
          f"repr params: {sum(p.numel() for p in repr_params)/1e6:.2f}M",
          flush=True)

    # --- RPBE machinery (arm-gated) ---
    rpbe_cfg = EmbodiedRPBConfig(
        kf_variant=args.kf_variant, kf_min_abs=args.kf_min_abs,
        lambda_rpbe=args.lambda_rpbe, rpbe_seed=args.seed)
    maps = EmbodiedFixedMaps(rpbe_cfg) if args.arm == "gamma-rpbe" else None
    queue = PendingMergeQueue() if args.arm != "avg" else None
    window = EmbodiedRPBEWindow(
        variant=rpbe_cfg.kf_variant, eps=rpbe_cfg.ridge_eps,
        min_abs=rpbe_cfg.kf_min_abs) if args.arm == "gamma-rpbe" else None
    task_cotangents: dict = {}       # (eid, node_id) -> accumulated g_task
    rpbe_cotangents: dict = {}       # (eid, node_id) -> g_rpbe (last window)
    rpbe_pending_loss: list = []     # deferred window adjoint (applied at
                                     # the next repr boundary, plan §4)
    merge_registry: dict = {}        # (eid, node_id) -> MergeRecord (all
                                     # merges; survives queue drain for replay)
    merge_id_map: dict = {}          # (eid, merge_id) -> node_id

    K = args.future_action_window_size + 1
    step = 0
    episodes_since_boundary = 0
    last_eid = None
    t0 = time.time()
    print("== training loop start (arm={}) ==".format(args.arm), flush=True)

    def feed_merges_and_futures(batch):
        """Consume merge_log -> queue.register; offer current-decision
        futures (context = frozen vision features, plan §14).

        Order matters (causal protocol): offer FIRST (feeds merges from
        previous batches), THEN drain finished episodes, THEN register
        this batch's new merges (whose futures start next batch)."""
        nonlocal last_eid, episodes_since_boundary
        bank = vla.cog_mem_bank
        eids = [int(e) for e in batch["episode_ids"]]
        # 1) register this batch's merges FIRST: their futures may be in
        # THIS batch too (episode tail rows d+1/d+2 after the merge row)
        for rec in bank.merge_log:
            queue.register(rec)
            merge_registry[(rec.episode_id, rec.node_id)] = rec
            merge_id_map[(rec.episode_id, rec.merge_id)] = rec.node_id
        bank.merge_log = []
        # 2) offer: this batch's rows feed merges from this AND previous
        # batches (rows equal to tau are skipped by the queue)
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
            y = batch["actions"][i].reshape(-1)      # [112] normalized
            queue.offer(eids[i], int(batch["timesteps"][i]), ctx, y.detach().cpu())
        # 3) drain episodes that finished (tail merges have their Y1/Y2)
        nonlocal_ep_counter = 0
        for eid in eids:
            if last_eid is not None and eid != last_eid:
                # per-episode merge count (review ruling: was passing the
                # cumulative next_merge_id, which distorted tree weights)
                rows = queue.drain_episode(last_eid)
                if window is not None and rows:
                    # B3: the fixed map replaces the raw 112D outcome; the
                    # window statistics run on P = psi(C, Y) in R^64
                    for r in rows:
                        r.outcome = maps.pv(r.context, r.outcome)
                    window.add(rows)
                nonlocal_ep_counter += 1
            last_eid = eid
        episodes_since_boundary += nonlocal_ep_counter

    def _ckpt_dict():
        """Checkpoint payload (B7): flat trainable increments + full
        metadata so a resume can rebuild the exact arm configuration."""
        return {
            "model": {n: p.detach().cpu()
                      for n, p in vla.named_parameters()
                      if p.requires_grad},
            "lora_config": {"r": lora_config.r,
                            "lora_alpha": lora_config.lora_alpha,
                            "lora_dropout": lora_config.lora_dropout},
            "arm": args.arm,
            "step": step,
            "param_version": vla.cog_mem_bank.param_version,
            "mem_length": args.mem_length,
            "lambda_rpbe": args.lambda_rpbe,
            "seed": args.seed,
            "task_filter": args.task_filter,
            "best_val": best_val,
        }

    best_val = float("inf")
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location="cpu",
                        weights_only=False)
        named = dict(vla.named_parameters())
        missing, unexpected = [], []
        for n, t in ck["model"].items():
            if n in named and named[n].requires_grad:
                named[n].data.copy_(t.to(named[n].dtype))
            else:
                unexpected.append(n)
        step = ck.get("step", 0)
        best_val = ck.get("best_val", float("inf"))
        vla.cog_mem_bank.param_version = ck.get("param_version", 0)
        print(f"resumed from {args.resume_from} @ step {step} "
              f"(unexpected keys: {len(unexpected)})", flush=True)
        # fast-forward the data stream so the batch sequence matches the
        # checkpoint exactly (HDF5 stream is deterministic per seed)
        n_rows = step * args.batch_size
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

    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            pixel_values = batch["pixel_values"]
            if isinstance(pixel_values, dict):
                pixel_values = {k: v.to("cuda", dtype=torch.bfloat16)
                                for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values.to("cuda", dtype=torch.bfloat16)

            # B4a: snapshot ALL current merged-entry leaves (with their
            # (eid, node_id)) BEFORE backward; leaves that get re-merged or
            # cleared this batch are still read from the snapshot after
            # backward (refresh_leafs rebuilds clones every step, so any
            # leaf not in this batch's graph has grad None).
            leaf_snapshot = []
            if vla.gamma is not None:
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
            step += 1

            # task cotangents from the leaf snapshot (plan §25, B4a)
            if vla.gamma is not None:
                for f, eid, node_id in leaf_snapshot:
                    if f.grad is not None:
                        key = (eid, node_id)
                        g = f.grad.detach().clone().reshape(-1)
                        task_cotangents[key] = (task_cotangents.get(
                            key, torch.zeros_like(g)) + g)
                vla.cog_mem_bank.refresh_leafs()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in task_params], args.grad_clip)
                opt_task.step()
                opt_task.zero_grad()
                sched_task.step()

            if vla.gamma is not None and args.arm != "avg":
                feed_merges_and_futures(batch)

            if step % args.log_every == 0:
                dt = time.time() - t0
                aux = (f"| kf_pending {len(rpbe_pending_loss)}"
                       if window is not None else "")
                diag = ""
                if vla.gamma is not None and queue is not None:
                    diag = (f" | merges_log {len(vla.cog_mem_bank.merge_log)}"
                            f" queue {len(queue.pending)}"
                            f" ep_since {episodes_since_boundary}"
                            f" censored {queue.n_censored}"
                            f" window_cuts {window.n_unique_cuts if window else '-'}")
                print(f"step {step}/{args.max_steps} | loss {loss.item():.4f} "
                      f"{aux}{diag} | {dt/60:.1f} min", flush=True)

            # repr macro boundary (SHARED clock across ALL arms, B2:
            # the gamma check was removed -- avg's LoRA steps here too)
            if episodes_since_boundary >= args.repr_boundary_episodes:
                # B5: close the statistics window at every repr boundary so
                # it never mixes parameter versions; underfull windows are
                # discarded with a counter (fail-fast via window.add assert)
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
                # B4b: task and RPBE replay INDEPENDENTLY (each uses its
                # own key set; no intersection with the other's keys)
                if vla.gamma is not None:
                    # task replay
                    keys = [k for k in task_cotangents if k in merge_registry]
                    if keys:
                        m_a = torch.stack(
                            [merge_registry[k].left_state for k in keys]
                        ).to("cuda", dtype=torch.bfloat16)
                        m_b = torch.stack(
                            [merge_registry[k].right_state for k in keys]
                        ).to("cuda", dtype=torch.bfloat16)
                        l_task = gamma_replay_loss(
                            vla.gamma, m_a, m_b, task_cotangents, keys)
                        l_task.backward()
                        print(f"[repr step] task replay {l_task.item():.4f}",
                              flush=True)
                    # rpbe replay (independent key set)
                    rkeys = [k for k in rpbe_cotangents
                             if k in merge_registry]
                    if rkeys and args.lambda_rpbe > 0:
                        m_a_r = torch.stack(
                            [merge_registry[k].left_state for k in rkeys]
                        ).to("cuda", dtype=torch.bfloat16)
                        m_b_r = torch.stack(
                            [merge_registry[k].right_state for k in rkeys]
                        ).to("cuda", dtype=torch.bfloat16)
                        l_rpbe = gamma_replay_loss(
                            vla.gamma, m_a_r, m_b_r, rpbe_cotangents, rkeys)
                        (args.lambda_rpbe * l_rpbe).backward()
                        print(f"[repr step] rpbe replay "
                              f"{args.lambda_rpbe * l_rpbe.item():.4f}",
                              flush=True)
                torch.nn.utils.clip_grad_norm_(repr_params, args.grad_clip)
                opt_repr.step()
                opt_repr.zero_grad()
                sched_repr.step()
                vla.cog_mem_bank.param_version += 1
                task_cotangents = {}
                rpbe_cotangents = {}
                rpbe_pending_loss = []
                # NOTE: merge_registry is NOT cleared -- merges still pending
                # in the queue need their records for the next replay.
                episodes_since_boundary = 0

            if step % args.eval_every == 0 and args.eval_every > 0:
                val_loss = run_eval(step)
                # B7: best-val selection with a dedicated checkpoint
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(_ckpt_dict(), run_dir / "best.pt")
                    print(f"[best @ {step}] val {val_loss:.4f}", flush=True)

            if step % args.checkpoint_every == 0:
                torch.save(_ckpt_dict(), run_dir / "checkpoint.pt")
                print(f"checkpoint saved @ {step}", flush=True)

    torch.save({
        "model": {n: p.detach().cpu() for n, p in vla.named_parameters()
                  if p.requires_grad},
        "lora_config": {"r": lora_config.r, "lora_alpha": lora_config.lora_alpha,
                        "lora_dropout": lora_config.lora_dropout},
        "step": step, "param_version": vla.cog_mem_bank.param_version,
    }, run_dir / "checkpoint.pt")
    with open(run_dir / "dataset_statistics.json", "w") as f:
        json.dump(dataset_statistics, f, indent=2)
    print("== training done ==", flush=True)
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
