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
  gamma-rpbe  : Gamma merge + task replay + the TGN-isomorphic multi-halfspace
                feasibility projection (--rpbe-mode project, DEFAULT).  One
                accumulated task direction t = -g_task and one half-space
                g_i.d >= -kappa||g_i||||t|| per memory interface i; a
                cutting-plane QP CHECKS EVERY interface and replaces ONLY the
                Gamma gradient (p.grad = g_task - sum_i mu_i g_i), then ONE
                clip + ONE AdamW step.  If the dimensionless max violation
                cannot be brought under --proj-tau the boundary is ABORTED (no
                Gamma step) rather than applied infeasibly.
                --rpbe-mode additive is the RETIRED L_task - lambda*J
                formulation, kept for old runs.

Ruling (2026-09-12, B3): RPBE is a CONSTRAINT, not a second objective.  The
compaudit showed J's predictive modes are first-order orthogonal AND
second-order flat to the task loss (cos ~ 0, D2 ~ 0), so lambda has no task
meaning and no lambda/weight/gate tuning can help.  Task loss stays the only
direction; feasibility projection supplies the preservation requirement.
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
    PendingMergeQueue, gamma_replay_loss, dual_latent_z_adjoint_modes,
    apply_gamma_boundary_update, boundary_config, common_config,
    realign_lambda_scheduler, verify_resume_config,
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
    p.add_argument("--eval-every", type=int, default=1000,
                   help="run whole-episode val evaluation every N OPTIMIZER "
                        "steps (0 = never)")
    p.add_argument("--val-episodes", type=int, default=3,
                   help="number of complete val demos scored per eval "
                        "(equal-weight averaged)")
    p.add_argument("--eval-seed", type=int, default=1234,
                   help="FIXED eval seed (independent of training step) so "
                        "re-eval of a checkpoint is bitwise reproducible")
    p.add_argument("--image-aug", type=int, default=1,
                   help="apply official RandomResizedCrop+ColorJitter image "
                        "augmentation on the TRAIN split only (0=off)")
    p.add_argument("--aug-seed", type=int, default=12345,
                   help="base seed for the independent per-row augment RNG "
                        "(derived as aug_seed+epoch+eid+t; never touches the "
                        "global diffusion RNG stream)")
    p.add_argument("--dim-weight", type=int, default=0,
                   help="0 (DEFAULT, review 2026-09-10: weighted-loss line "
                        "abandoned) = plain equal-weight diffusion MSE; "
                        "1 = per-dim weighted loss (x/y=1.5,z=2.0,rot=0.5,"
                        "grip=1.0) -- deprecated, invalidated control")
    p.add_argument("--limit-episodes", type=int, default=0,
                   help="train on only the first N train demos (0 = all); "
                        "single-demo overfit / coverage diagnostics")
    p.add_argument("--opt-audit", type=int, default=0,
                   help="verbose optimizer-closure audit (coverage + per-group "
                        "grad/update norms) for the first N DENSE steps, then "
                        "stop training (0 = off)")
    p.add_argument("--train-scope",
                   choices=["full", "lora-gamma", "gamma-only"],
                   default="full",
                   help="how much of the HOST may move.  full = the legacy "
                        "recipe (memory banks + per_compr + DiT + LoRA).  "
                        "lora-gamma = ONLY LoRA (+ reg_head) in the task "
                        "optimizer, so the diffusion action model and the "
                        "memory modules stay frozen.  gamma-only = nothing in "
                        "the task optimizer; only Gamma trains (pure merger "
                        "adaptation from a host that already rolls out).  "
                        "Starting from a working checkpoint, 'full' is a "
                        "second round of training that destroys the host.")
    p.add_argument("--reg-head", type=int, default=0,
                   help="Stage7: 1 = use the shared deterministic regression "
                        "action head (6DoF equal-weight SmoothL1 + gripper "
                        "BCE) instead of the diffusion loss")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=100,
                   help="warmup in OPTIMIZER steps")
    p.add_argument("--sched", choices=["cosine", "const"], default="const",
                   help="LR schedule for the task optimizer: const (official "
                        "MemoryVLA default: constant 2e-5, warmup ignored) or "
                        "cosine (legacy)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--future-action-window-size", type=int, default=15)
    p.add_argument("--mem-length", type=int, default=16,
                   help="CogMemBank capacity (dense protocol: 16 frames)")
    p.add_argument("--repeated-diffusion-steps", type=int, default=4)
    p.add_argument("--init-from-weights", default="",
                   help="WEIGHTS-SNAPSHOT ONLY: load trainable weights from a "
                        "checkpoint and start training at opt 0 with fresh "
                        "optimizer/scheduler/RNG/memory.  NOT a resumption -- "
                        "never splice its output onto the prior run.")
    p.add_argument("--resume-full", default="",
                   help="TRUE continuation: load a full-state checkpoint "
                        "(weights + optimizer + schedulers + step counters + "
                        "CPU/CUDA/numpy RNG) and continue.  Training must be "
                        "resumed with the SAME max_steps/batch/grad_accum "
                        "that produced it.  Data stream restarts from a fresh "
                        "epoch (deterministic seed) -- not bitwise cursor "
                        "continuation.")
    p.add_argument("--no-fullstate", type=int, default=0,
                   help="1 = do NOT write the rolling fullstate.pt (Adam fp32 "
                        "state makes it ~2x the weights).  For short pilots "
                        "where resuming is not needed this halves the disk "
                        "cost per run; latest.pt/checkpoint.pt still hold the "
                        "weights.")
    p.add_argument("--snapshot-steps", default="30000,35000,40000",
                   help="optimizer steps (of THIS run) at which to also write "
                        "a weights-only snapshot_<step>.pt for offline eval")
    # RPBE
    p.add_argument("--kf-variant", choices=["full_dual", "diag"],
                   default="full_dual")
    p.add_argument("--kf-min-abs", type=int, default=64,
                   help="min unique merges per RPBE window")
    p.add_argument("--lambda-rpbe", type=float, default=0.0,
                   help="LEGACY additive mode only (--rpbe-mode additive): "
                        "weight on the -lambda*J term.  Unused by the default "
                        "project mode (RPBE is a constraint, not an objective).")
    p.add_argument("--rpbe-mode", choices=["additive", "project"],
                   default="project",
                   help="Gamma update rule for the gamma-rpbe arm.  project "
                        "(DEFAULT, TGN-isomorphic): per-interface influence "
                        "gradients -> multi-halfspace feasibility QP -> replace "
                        "ONLY the Gamma gradient -> one clip + one AdamW step. "
                        "additive (RETIRED legacy): L_task - lambda*J.")
    p.add_argument("--kappa", type=float, default=0.05,
                   help="project mode: an interface with cos(g_i, -g_task) < "
                        "-kappa is protected.  FIXED constant on the formal "
                        "method (no annealing).  kappa=0 = hard per-interface "
                        "constraint (STRICTEST); kappa>=1 = never binds "
                        "(== pure task).")
    p.add_argument("--proj-iters", type=int, default=400,
                   help="project mode: FISTA iterations at the FIRST rung of "
                        "the solver-budget ladder.")
    p.add_argument("--proj-iters-max", type=int, default=1600,
                   help="project mode: top of the FISTA ladder (400 -> 800 -> "
                        "1600).  A boundary that cannot be certified at a rung "
                        "escalates instead of aborting; the certificate "
                        "tolerance --proj-tau is never relaxed.")
    p.add_argument("--proj-tau", type=float, default=1e-3,
                   help="project mode: feasibility tolerance on the "
                        "DIMENSIONLESS violation v_i = max(0, b_i - g_i.d)/"
                        "(||g_i||||t||).  v_max must be <= tau or the boundary "
                        "is aborted (no Gamma step).")
    p.add_argument("--proj-max-rounds", type=int, default=8,
                   help="project mode: cutting-plane rounds per boundary.")
    p.add_argument("--proj-max-active", type=int, default=2048,
                   help="project mode: cap on the ACTIVE set (worst violators "
                        "enter first; ALL interfaces are always checked).")
    p.add_argument("--proj-add-per-round", type=int, default=512,
                   help="project mode: new violators added to the active set "
                        "per cutting-plane round.")
    p.add_argument("--migrate-legacy-gamma-state", type=int, default=0,
                   help="allow --resume-full from a Stage7 checkpoint (no "
                        "rpbe_mode in its config): the Gamma optimizer state is "
                        "RESET (old additive -lambda*J Adam moments must never "
                        "seed a feasibility-projection run).  0 = refuse (the "
                        "default).")
    # macro boundary: with dense protocol one episode already yields many cuts,
    # so every finished episode is one boundary.
    p.add_argument("--repr-boundary-episodes", type=int, default=1)
    # gamma LR scheduler in units of ACTUAL gamma steps (param_version).
    p.add_argument("--repr-warmup-steps", type=int, default=30,
                   help="warmup length for the gamma scheduler in gamma steps")
    p.add_argument("--repr-total-steps", type=int, default=1200,
                   help="cosine denominator for the gamma scheduler in gamma "
                        "steps (dense: ~1 gamma step per episode)")
    # --- Stage5-R structural fix (reviewer 2026-09-10) ---
    p.add_argument("--gamma-lr", type=float, default=-1.0,
                   help="AdamW lr for the Gamma merge operator (opt_gamma). "
                        "<0 = use --lr.  Calibrated, not guessed.")
    p.add_argument("--gamma-replay-batch-size", type=int, default=64,
                   help="episode-end Gamma replay: split the episode's replay "
                        "keys into minibatches of this many cuts and take one "
                        "opt_gamma.step() per minibatch (Gamma stays fixed "
                        "WITHIN an episode; more updates per episode).")
    p.add_argument("--gamma-task-boundary-episodes", type=int, default=1,
                   help="number of finished episodes that trigger one Gamma "
                        "task-replay boundary (Gamma task clock).")
    p.add_argument("--rpbe-stats-episodes", type=int, default=1,
                   help="number of finished episodes accumulated into ONE RPBE "
                        "statistics window before closing it (RPBE clock, "
                        "decoupled from the Gamma task clock).")
    p.add_argument("--perm-null", type=int, default=0,
                   help="cut-block permutation null repetitions for the J gap "
                        "(calibration: >=128; formal training: 0 = off, avoids "
                        "the CPU Cholesky cost).")
    p.add_argument("--rpbe-comp-audit", type=int, default=0,
                   help="component-wise RPBE audit: for the first K RPBE "
                        "window closes, decompose J into predictive modes "
                        "(squared canonical correlations) and log "
                        "cos(g_{J_k}, g_task) per mode.")
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
        rpbe_seed=args.seed,
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
    # kappa is a cosine margin: it must be a finite, non-negative number.  A
    # negative kappa would flip the sense of g_i.d >= -kappa||g_i||||t||.
    if not math.isfinite(args.kappa) or args.kappa < 0:
        raise SystemExit(f"--kappa must be finite and >= 0, got {args.kappa}")
    if args.proj_tau <= 0 or not math.isfinite(args.proj_tau):
        raise SystemExit(f"--proj-tau must be finite and > 0, got "
                         f"{args.proj_tau}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    flags = arm_flags(args.arm)
    IS_GAMMA, IS_RPBE = flags["is_gamma"], flags["is_rpbe"]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    vla, lora_config = build_vla(args)
    # per-dim weighted loss toggle (reviewer: default on; off = equal-weight
    # restart control experiment)
    vla.use_dim_weight = bool(args.dim_weight)
    vla.use_reg_head = bool(args.reg_head)
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
            task_filter=args.task_filter,
            image_aug=bool(args.image_aug), aug_seed=args.aug_seed))
    print("train episodes:", len(train_dataset), flush=True)
    if args.limit_episodes:
        train_dataset.episodes = train_dataset.episodes[:args.limit_episodes]
        print(f"[data] LIMIT to first {len(train_dataset.episodes)} train "
              f"episodes: {[e['demo_key'] for e in train_dataset.episodes]}",
              flush=True)
    n_train_rows = sum(max(0, ep["T"]) for ep in train_dataset.episodes)
    print(f"train rows/epoch (dense): {n_train_rows}", flush=True)

    # num_workers MUST be 0: worker processes would each shuffle the episode
    # stream independently and interleave rows across episodes.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=0,
        collate_fn=collator, drop_last=True)

    val_dataset = None
    if args.eval_every > 0:
        val_dataset, _, _ = get_hdf5_decision_stream_dataset_and_collator(
            data_root=Path(args.data_root), tokenizer=tokenizer,
            image_transform=image_transform,
            prompt_builder_fn=vla.vlm.llm_backbone.prompt_builder_fn,
            future_action_window_size=args.future_action_window_size,
            seed=args.seed, split="val",
            pad_token_id=tokenizer.pad_token_id,
            task_filter=args.task_filter)
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
        """Whole-episode val action loss over --val-episodes complete demos.

        Each demo is run from its first frame to its last IN ORDER with
        continuous CogMemBank state, the per-demo loss is its frame mean, and
        the returned value is the EQUAL-WEIGHT mean over demos.  A fixed seed
        (--eval-seed, NOT a function of optimizer_step) gives bitwise
        reproducible re-evals of one checkpoint.  Full isolation: cog+per bank
        state and CPU/CUDA/python RNG are snapshotted before and restored
        after so eval never perturbs the training stream."""
        cog = vla.cog_mem_bank
        per = vla.per_mem_bank
        cog_snap = _snapshot_bank(cog)
        per_snap = _snapshot_bank(per)
        rng_snapshot = (copy.deepcopy(torch.get_rng_state()),
                        copy.deepcopy(torch.cuda.get_rng_state()),
                        np.random.get_state())
        demo_losses = []
        try:
            vla.eval()
            torch.manual_seed(args.eval_seed)
            torch.cuda.manual_seed(args.eval_seed)
            n_demos = min(args.val_episodes, len(val_dataset.episodes))
            for di in range(n_demos):
                cog.reset()
                per.reset()
                ep_losses = []
                rows_buf = []
                for row in val_dataset.iter_episode(di):
                    rows_buf.append(row)
                    if len(rows_buf) >= args.batch_size:
                        batch = collator(rows_buf)
                        rows_buf = []
                        with torch.no_grad():
                            pv = batch["pixel_values"]
                            if isinstance(pv, dict):
                                pv = {k: v.to("cuda", dtype=torch.bfloat16)
                                      for k, v in pv.items()}
                            else:
                                pv = pv.to("cuda", dtype=torch.bfloat16)
                            with torch.autocast("cuda", dtype=torch.bfloat16,
                                                enabled=True):
                                loss, _ = vla(
                                    input_ids=batch["input_ids"].to("cuda"),
                                    attention_mask=batch["attention_mask"].to("cuda"),
                                    actions=batch["actions"].to(
                                        "cuda", dtype=torch.bfloat16),
                                    action_masks=batch["action_masks"].to("cuda"),
                                    pixel_values=pv,
                                    labels=batch["labels"].to("cuda"),
                                    timesteps=batch["timesteps"],
                                    episode_ids=batch["episode_ids"],
                                    output_hidden_states=True,
                                    repeated_diffusion_steps=args.repeated_diffusion_steps,
                                )
                        ep_losses.append(loss.item())
                # flush any partial final batch (keep whole-demo coverage)
                if rows_buf:
                    batch = collator(rows_buf)
                    rows_buf = []
                    with torch.no_grad():
                        pv = batch["pixel_values"]
                        if isinstance(pv, dict):
                            pv = {k: v.to("cuda", dtype=torch.bfloat16)
                                  for k, v in pv.items()}
                        else:
                            pv = pv.to("cuda", dtype=torch.bfloat16)
                        with torch.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=True):
                            loss, _ = vla(
                                input_ids=batch["input_ids"].to("cuda"),
                                attention_mask=batch["attention_mask"].to("cuda"),
                                actions=batch["actions"].to(
                                    "cuda", dtype=torch.bfloat16),
                                action_masks=batch["action_masks"].to("cuda"),
                                pixel_values=pv,
                                labels=batch["labels"].to("cuda"),
                                timesteps=batch["timesteps"],
                                episode_ids=batch["episode_ids"],
                                output_hidden_states=True,
                                repeated_diffusion_steps=args.repeated_diffusion_steps,
                            )
                    ep_losses.append(loss.item())
                if ep_losses:
                    demo_mean = sum(ep_losses) / len(ep_losses)
                    demo_losses.append(demo_mean)
                    print(f"[eval demo {di}] mean {demo_mean:.4f} "
                          f"({len(ep_losses)} batches)", flush=True)
        finally:
            _restore_bank(cog, cog_snap)
            _restore_bank(per, per_snap)
            torch.set_rng_state(rng_snapshot[0])
            torch.cuda.set_rng_state(rng_snapshot[1])
            np.random.set_state(rng_snapshot[2])
            vla.train()
        if demo_losses:
            mean = sum(demo_losses) / len(demo_losses)
            print(f"[eval @ opt {optimizer_step}] val action loss "
                  f"{mean:.4f} ({len(demo_losses)} demos)", flush=True)
            return mean
        return float("inf")

    # ---- parameter split ----
    # opt_task: cog/per banks + per_compr + DiT + LoRA -> dense (every block).
    # opt_gamma: gamma merge operator only -> macro boundary replay.
    # NOTE: gamma is registered both on vla.gamma AND as cog_mem_bank.gamma
    # (memory_vla.py:594), so cog_mem_bank.parameters() already includes it;
    # it must be EXCLUDED from the task optimizer and live only in opt_gamma.
    lora_params = [p for n, p in vla.named_parameters()
                   if p.requires_grad and "lora_" in n]
    lora_ids = {id(p) for p in lora_params}
    gamma_params = (list(vla.gamma.parameters())
                    if vla.gamma is not None else [])
    gamma_ids = {id(p) for p in gamma_params}
    if args.train_scope == "full":
        task_modules = [vla.cog_mem_bank, vla.per_mem_bank, vla.per_compr,
                        vla.action_model]
        task_params = [p for m in task_modules for p in m.parameters()
                       if p.requires_grad and id(p) not in lora_ids
                       and id(p) not in gamma_ids]
        task_params += lora_params
        if getattr(vla, "reg_head", None) is not None:
            task_params += [p for p in vla.reg_head.parameters()
                            if p.requires_grad and id(p) not in gamma_ids]
    elif args.train_scope == "lora-gamma":
        # freeze the host: the DiT action model and the memory modules keep the
        # rollout ability the init checkpoint already has.
        task_params = list(lora_params)
        if getattr(vla, "reg_head", None) is not None:
            task_params += [p for p in vla.reg_head.parameters()
                            if p.requires_grad and id(p) not in gamma_ids]
    else:  # gamma-only: nothing but the merger trains
        task_params = []
    # disjointness guarantee (reviewer): no trainable param in both optimizers
    task_ids = {id(p) for p in task_params}
    overlap = task_ids & gamma_ids
    assert not overlap, f"gamma/task param overlap: {len(overlap)}"
    gamma_lr = args.gamma_lr if args.gamma_lr > 0 else args.lr
    opt_task = (torch.optim.AdamW(task_params, lr=args.lr, weight_decay=0.0,
                                  foreach=False) if task_params else None)
    if gamma_params:
        opt_gamma = torch.optim.AdamW(gamma_params, lr=gamma_lr,
                               weight_decay=0.0, foreach=False)
    else:
        opt_gamma = None

    # cosine decay w/ warmup in OPTIMIZER-step units for the dense (task+LoRA)
    # optimizer.
    # LR schedule: official MemoryVLA = CONSTANT (warmup ignored); legacy runs
    # used cosine-with-warmup.  Under --sched const BOTH optimizers hold base
    # lr for the whole run (gamma cosine would otherwise anneal gamma to LR 0
    # after repr_total_steps, freezing it mid-40k).
    def lr_lambda_task(t):
        if args.sched == "const":
            return min(1.0, t / max(1, args.warmup_steps))  # warmup then const
        if t < args.warmup_steps:
            return t / max(1, args.warmup_steps)
        progress = (t - args.warmup_steps) / max(
            1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched_task = (torch.optim.lr_scheduler.LambdaLR(opt_task, lr_lambda_task)
                  if opt_task is not None else None)
    # gamma scheduler indexed by param_version (gamma step count).
    def lr_lambda_gamma(rs):
        if args.sched == "const":
            return min(1.0, rs / max(1, args.repr_warmup_steps))  # warmup->const
        if rs < args.repr_warmup_steps:
            return rs / max(1, args.repr_warmup_steps)
        progress = (rs - args.repr_warmup_steps) / max(
            1, args.repr_total_steps - args.repr_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched_gamma = None
    if opt_gamma is not None:
        sched_gamma = torch.optim.lr_scheduler.LambdaLR(opt_gamma,
                                                        lr_lambda_gamma)
    print(f"scope={args.train_scope} | "
          f"task params: {sum(p.numel() for p in task_params)/1e6:.2f}M | "
          f"gamma params: {sum(p.numel() for p in gamma_params)/1e6:.2f}M",
          flush=True)

    # ---- optimizer-closure audit (--opt-audit N) ----
    audit_groups = {
        "other_task": [p for p in task_params if id(p) not in lora_ids],
        "lora": lora_params,
        "gamma": gamma_params,
    }

    def _audit_once():
        """Print coverage/membership invariants once before the first step."""
        all_tr = [p for p in vla.parameters() if p.requires_grad]
        cov = {id(p) for p in task_params} | gamma_ids
        missing = [(n, id(p)) for n, p in vla.named_parameters()
                   if p.requires_grad and id(p) not in cov]
        lora_leak = lora_ids - task_ids
        print(f"[audit] trainable={len(all_tr)} covered={len(cov)} "
              f"missing={len(missing)} lora_not_in_task={len(lora_leak)} "
              f"gamma_overlap={len(task_ids & gamma_ids)}", flush=True)
        for n, pid in missing[:20]:
            print(f"[audit]   MISSING {n}", flush=True)
        return missing

    def _audit_snapshot():
        snap = {}
        for gname, ps in audit_groups.items():
            g2 = 0.0
            for p in ps:
                if p.grad is not None:
                    g2 += float(p.grad.detach().float().pow(2).sum())
            snap[gname] = (math.sqrt(g2), [(p, p.detach().clone())
                                           for p in ps])
        return snap

    def _audit_report(step, snap):
        cov = {id(p) for p in task_params} | gamma_ids
        miss = sum(1 for p in vla.parameters()
                   if p.requires_grad and id(p) not in cov)
        for gname, (gnorm, pairs) in snap.items():
            upd2 = 0.0
            dev = set()
            for p, p0 in pairs:
                upd2 += float((p.detach() - p0).float().pow(2).sum())
                dev.add(str(p.device))
            if gname == "gamma" and opt_gamma is not None:
                lr = opt_gamma.param_groups[-1]["lr"]
            elif opt_task is not None:
                lr = opt_task.param_groups[-1]["lr"]
            else:
                lr = float("nan")
            print(f"[audit {step}] {gname:10s} n={len(pairs)} "
                  f"grad={gnorm:.3e} upd={math.sqrt(upd2):.3e} "
                  f"lr={lr:.2e} dev={sorted(dev)}", flush=True)
        # optimizer state device sample (lazy AdamW state after step).
        # Flag ANY param whose state lives on a different device than itself
        # (the CPU-poisoning hazard the closure audit is meant to catch).
        id2name = {id(p): n for n, p in vla.named_parameters()}
        off = []
        devs = set()
        for g in (opt_task.param_groups if opt_task is not None else []):
            for p in g["params"]:
                if p in opt_task.state:
                    for st in opt_task.state[p].values():
                        if torch.is_tensor(st):
                            devs.add(str(st.device))
                            if str(st.device) != str(p.device):
                                off.append((id2name.get(id(p), "?"),
                                            str(p.device), str(st.device)))
        print(f"[audit {step}] coverage: missing={miss} "
              f"opt_task_state_dev={sorted(devs)}", flush=True)
        for nm, pd_, sd in off[:30]:
            print(f"[audit {step}]   OFFDEVICE {nm} param={pd_} state={sd}",
                  flush=True)

    if args.opt_audit > 0:
        _audit_once()

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
    rpbe_input_map: dict = {}   # cut_id -> (child_left, child_right) rebuilt
    rpbe_pending_loss: list = []
    merge_registry: dict = {}
    merge_id_map: dict = {}
    rpbe_window_episodes = 0   # finished episodes added to the RPBE window
                               # since its last close (rpbe_stats_episodes)

    optimizer_step = 0
    micro_step = 0
    episodes_since_boundary = 0
    episodes_seen = 0         # total finished episodes (budget lock)
    comp_audit_done = 0       # RPBE component-audit windows completed
    window_micro = 0          # micro-backwards since last gamma boundary
    boundary_pending = False  # repr threshold crossed; fire at fresh-episode top
    last_eid = None
    t0 = time.time()
    print("== training loop start (arm={}) ==".format(args.arm), flush=True)

    def feed_merges_and_futures(batch):
        """Consume merge_log -> queue.register; offer current-decision
        futures; drain finished episodes.  Causal order: register this
        batch's merges first, then offer, then drain completed episodes."""
        nonlocal last_eid, episodes_since_boundary, rpbe_window_episodes
        nonlocal episodes_seen
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
                if window is not None:
                    window.add_records(
                        [merge_registry[k] for k in list(merge_registry)
                         if k[0] == last_eid])
                rows = queue.drain_episode(last_eid)
                if window is not None and rows:
                    for r in rows:
                        r.outcome = maps.pv(r.context, r.outcome)
                    window.add(rows)
                    rpbe_window_episodes += 1
                n_done += 1
            last_eid = eid
        episodes_since_boundary += n_done
        episodes_seen += n_done

    def _gamma_boundary(scale):
        """One gamma macro-boundary update.  scale = grad_accum/window_micro
        normalizes the task cotangents (summed over window_micro micro-back-
        wards each scaled by 1/grad_accum) to a per-micro mean BEFORE any
        replay adds its contribution, keeping the RPBE-vs-task relative
        weight independent of episode length under the dense protocol."""
        nonlocal window, task_cotangents, rpbe_cotangents, rpbe_pending_loss
        nonlocal rpbe_input_map, rpbe_window_episodes
        nonlocal comp_audit_done
        if not IS_GAMMA or vla.gamma is None:
            return
        comp_zpw = None

        def _merge_fn(l, r):
            # single-node current-Gamma merge for fixed-trace rebuild
            with torch.no_grad():
                md = next(vla.gamma.parameters()).dtype
                return vla.gamma(l.to("cuda", dtype=md), r.to("cuda", dtype=md))

        # close the RPBE statistics window ONLY once enough finished episodes
        # have accumulated (RPBE clock, decoupled from the Gamma task clock)
        if (window is not None and window.n_unique_cuts > 0
                and rpbe_window_episodes >= args.rpbe_stats_episodes):
            if window.ready():
                j, g_by_cut, rpbe_inputs, diag = window.close(
                    merge_fn=_merge_fn, n_perm=args.perm_null)
                comp_zpw = getattr(window, "last_zpw", None)
                if g_by_cut:
                    # key by cut_id directly; replay inputs are rebuilt with
                    # the CURRENT Gamma (no shared registry needed)
                    rpbe_cotangents.update(g_by_cut)
                    for cid, (cl, cr) in rpbe_inputs.items():
                        rpbe_input_map[cid] = (cl, cr)
                    rpbe_pending_loss.append(j)
                print(f"[window close] J={j:.4f} J_gap={diag.get('J_gap')} "
                      f"J_p95={diag.get('J_perm_p95')} "
                      f"cuts={diag.get('n_unique_cuts')} "
                      f"episodes={diag.get('n_unique_episodes')} "
                      f"rebuild_inputs={len(rpbe_inputs)} "
                      f"censored={queue.n_censored}", flush=True)
            else:
                n_discard = window.discard()
                print(f"[window discard] underfull "
                      f"({n_discard} cuts < {rpbe_cfg.kf_min_abs})",
                      flush=True)
            window = EmbodiedRPBEWindow(
                variant=rpbe_cfg.kf_variant, eps=rpbe_cfg.ridge_eps,
                min_abs=rpbe_cfg.kf_min_abs)
            rpbe_window_episodes = 0
        # --- episode-end minibatch replay (Stage5-R) ---
        # Scale the summed leaf cotangents to a per-micro mean BEFORE any
        # replay adds its own contribution (normalization protocol).
        if task_cotangents and scale != 1.0:
            task_cotangents = {k: v * scale for k, v in task_cotangents.items()}
        task_keys = [k for k in task_cotangents if k in merge_registry]
        # project mode needs the rpbe rows regardless of lambda (RPBE is a
        # constraint there); additive mode only uses them when lambda > 0.
        rpbe_keys = (list(rpbe_cotangents.keys())
                     if (IS_RPBE and (args.rpbe_mode == "project"
                                      or args.lambda_rpbe > 0)) else [])
        # Deterministic per-episode shuffle -> the TASK replay minibatch
        # partition is identical for gamma-task and gamma-rpbe; the rpbe arm
        # only ADDS rpbe gradients (clean attribution).
        rng = np.random.default_rng(
            args.seed * 131 + int(vla.cog_mem_bank.param_version))
        task_keys = ([task_keys[i] for i in rng.permutation(len(task_keys))]
                     if task_keys else [])
        rpbe_keys = ([rpbe_keys[i] for i in rng.permutation(len(rpbe_keys))]
                     if rpbe_keys else [])
        B = max(1, args.gamma_replay_batch_size)
        # Blocker A: a boundary with no task keys must be skipped by BOTH arms
        # (never let the rpbe arm take a pure-RPBE step).  In formal training
        # ntask>0 always holds; a skip here means a degenerate episode.
        if not task_keys:
            print("[gamma] no task keys at this boundary -> skip both arms",
                  flush=True)
            task_cotangents = {}
            rpbe_cotangents = {}
            rpbe_input_map = {}
            rpbe_pending_loss = []
            return
        # audit: alpha / ||U|| / residual (Gamma vs Avg) on a key sample
        if opt_gamma is not None:
            with torch.no_grad():
                gm = vla.gamma
                ga = float(gm.alpha.detach())
                umag = sum(float(p.detach().float().pow(2).sum())
                           for p in gm.parameters() if p is not gm.alpha) ** 0.5
                kk = task_keys[:min(32, len(task_keys))]
                a_ = torch.stack([merge_registry[k].left_state
                                  for k in kk]).to("cuda", dtype=torch.bfloat16)
                b_ = torch.stack([merge_registry[k].right_state
                                  for k in kk]).to("cuda", dtype=torch.bfloat16)
                mavg = (a_.float() + b_.float()) / 2
                zg = gm(a_, b_).float()
                res = float(((zg - mavg).pow(2).sum() /
                             (mavg.pow(2).sum() + 1e-8)) ** 0.5)
            print(f"[gamma audit] alpha={ga:.4f} |U|={umag:.4f} "
                  f"residual={res:.4f}", flush=True)
        # Blocker 2: opt_gamma step count is set by the TASK key count ALONE,
        # so gamma-task and gamma-rpbe step in lockstep; the rpbe keys are
        # distributed round-robin into those same n_mb buckets.
        # ---- component-wise RPBE gradient-alignment audit ----
        if (args.rpbe_comp_audit > 0 and comp_audit_done < args.rpbe_comp_audit
                and comp_zpw is not None and rpbe_keys and task_keys):
            z_, p_, w_, cids_ = comp_zpw
            modes, Jsum, _md = dual_latent_z_adjoint_modes(
                z_, p_, w_, cids_, eps=rpbe_cfg.ridge_eps, n_modes=8)
            opt_gamma.zero_grad()
            m_a = torch.stack([merge_registry[k].left_state
                               for k in task_keys]).to("cuda", dtype=torch.bfloat16)
            m_b = torch.stack([merge_registry[k].right_state
                               for k in task_keys]).to("cuda", dtype=torch.bfloat16)
            gamma_replay_loss(vla.gamma, m_a, m_b,
                              task_cotangents, task_keys).backward()
            gt = [p.grad.detach().clone() if p.grad is not None else None
                  for p in gamma_params]
            opt_gamma.zero_grad()
            gt_n = sum(float(g.float().pow(2).sum())
                       for g in gt if g is not None) ** 0.5
            for mi, (Jk, gbc) in modes.items():
                rk = [k for k in rpbe_keys if k in gbc]
                if not rk:
                    continue
                m_ar = torch.stack([rpbe_input_map[k][0].to("cuda", dtype=torch.bfloat16)
                                    for k in rk])
                m_br = torch.stack([rpbe_input_map[k][1].to("cuda", dtype=torch.bfloat16)
                                    for k in rk])
                opt_gamma.zero_grad()
                gamma_replay_loss(vla.gamma, m_ar, m_br, gbc, rk).backward()
                gk = [p.grad.detach().clone() if p.grad is not None else None
                      for p in gamma_params]
                dot = sum(float((a.float() * b.float()).sum())
                          for a, b in zip(gk, gt)
                          if a is not None and b is not None)
                gn = sum(float(a.float().pow(2).sum())
                         for a in gk if a is not None) ** 0.5
                cos = dot / (gn * gt_n + 1e-12)
                print(f"[comp-audit] pv={vla.cog_mem_bank.param_version} "
                      f"mode={mi} Jk={Jk:.5f} |gJk|={gn:.3e} "
                      f"rel={gn/(gt_n+1e-12):.3f} cos_task={cos:+.3f}",
                      flush=True)
                # second-order intervention: theta += ±eps*ghat_k, measure task loss
                if gn > 0 and fixed_batch is not None:
                    eps = 0.05
                    theta0 = [p.detach().clone() for p in gamma_params]
                    L0 = _task_loss_on_batch(fixed_batch)
                    for p, a in zip(gamma_params, gk):
                        if a is not None:
                            p.data.copy_(p.data + eps * a.float() / gn)
                    Lp = _task_loss_on_batch(fixed_batch)
                    for i, (p, a) in enumerate(zip(gamma_params, gk)):
                        if a is not None:
                            p.data.copy_(theta0[i] - eps * a.float() / gn)
                    Lm = _task_loss_on_batch(fixed_batch)
                    for i, p in enumerate(gamma_params):
                        p.data.copy_(theta0[i])
                    D2 = Lp + Lm - 2.0 * L0
                    print(f"[interv] mode={mi} L0={L0:.6f} L+={Lp:.6f} "
                          f"L-={Lm:.6f} D2={D2:+.6f}", flush=True)
            opt_gamma.zero_grad()
            comp_audit_done += 1
            print(f"[comp-audit] window {comp_audit_done}/"
                  f"{args.rpbe_comp_audit} J={Jsum:.3f} nmode={len(modes)}",
                  flush=True)

        n_mb = max(1, (len(task_keys) + B - 1) // B)
        n_opt = 0

        def _gm():
            return [p.grad.detach().clone() if p.grad is not None else None
                    for p in gamma_params]

        if args.rpbe_mode == "additive":
            # ---- LEGACY additive protocol (L_task - lambda*J).  Retired: J is
            # first-order orthogonal and second-order flat to the task loss, so
            # lambda carries no task signal.  Kept only to reproduce old runs.
            rpbe_buckets = [[] for _ in range(n_mb)]
            for i, k in enumerate(rpbe_keys):
                rpbe_buckets[i % n_mb].append(k)
            for i in range(n_mb):
                t_sl = task_keys[i * B:(i + 1) * B]
                r_sl = rpbe_buckets[i]
                if opt_gamma is not None:
                    opt_gamma.zero_grad()
                if t_sl:
                    m_a = torch.stack([merge_registry[k].left_state
                                       for k in t_sl]).to("cuda", dtype=torch.bfloat16)
                    m_b = torch.stack([merge_registry[k].right_state
                                       for k in t_sl]).to("cuda", dtype=torch.bfloat16)
                    l_task = gamma_replay_loss(vla.gamma, m_a, m_b,
                                               task_cotangents, t_sl)
                    l_task.backward()
                g_t = _gm() if opt_gamma is not None else []
                if opt_gamma is not None:
                    opt_gamma.zero_grad()
                if r_sl:
                    m_ar = torch.stack([rpbe_input_map[k][0].to("cuda", dtype=torch.bfloat16)
                                        for k in r_sl])
                    m_br = torch.stack([rpbe_input_map[k][1].to("cuda", dtype=torch.bfloat16)
                                        for k in r_sl])
                    l_rpbe = gamma_replay_loss(vla.gamma, m_ar, m_br,
                                               rpbe_cotangents, r_sl)
                    # RPBE maximizes J -> negative sign (gradient ascent on J).
                    (-args.lambda_rpbe * l_rpbe).backward()
                g_r = _gm() if opt_gamma is not None else []
                if opt_gamma is not None:
                    pre = [p.detach().clone() for p in gamma_params]
                    nt2 = nr2 = dot = 0.0
                    for p, a, b in zip(gamma_params, g_t, g_r):
                        av = a if a is not None else torch.zeros_like(p)
                        bv = b if b is not None else torch.zeros_like(p)
                        p.grad = av + bv
                        fa = av.float(); fb = bv.float()
                        nt2 += float(fa.pow(2).sum()); nr2 += float(fb.pow(2).sum())
                        dot += float((fa * fb).sum())
                    nt = nt2 ** 0.5; nr = nr2 ** 0.5
                    cos = dot / (nt * nr + 1e-12)
                    clip = torch.nn.utils.clip_grad_norm_(gamma_params,
                                                          args.grad_clip)
                    opt_gamma.step()
                    sched_gamma.step()
                    upd = sum(float((p.detach() - p0).float().pow(2).sum())
                              for p, p0 in zip(gamma_params, pre)) ** 0.5
                    print(f"[gamma audit] pv={vla.cog_mem_bank.param_version} "
                          f"mb={i} ntask={len(t_sl)} nrpbe={len(r_sl)} "
                          f"|g_task|={nt:.3e} |g_rpbe|={nr:.3e} cos={cos:.3f} "
                          f"r_eff={nr/(nt+1e-12):.3f} clip={float(clip):.3f} "
                          f"|dgamma|={upd:.3e}", flush=True)
                    n_opt += 1
            assert n_opt == n_mb, (
                f"gamma step count {n_opt} != expected task-replay steps {n_mb}")
            print(f"[gamma] episode replay: {len(task_keys)} task / "
                  f"{len(rpbe_keys)} rpbe keys -> {n_opt} opt_gamma steps "
                  f"(mb<= {B})", flush=True)
        else:
            # ---- TGN-isomorphic single update (default) ----
            # RPBE is a CONSTRAINT, not an objective: ONE accumulated task
            # direction, one half-space per memory interface, a cutting-plane
            # projection that CHECKS EVERY interface, then ONE clip + ONE
            # opt_gamma.step().  gamma-task runs the SAME protocol with no rpbe
            # rows, so the two arms differ ONLY by the projection.  An
            # infeasible projection aborts the boundary (no Gamma step).
            task_pairs = [(merge_registry[k].left_state,
                           merge_registry[k].right_state) for k in task_keys]
            task_cots = [task_cotangents[k] for k in task_keys]
            r_pairs = [rpbe_input_map[k] for k in rpbe_keys] if rpbe_keys else []
            r_cots = [rpbe_cotangents[k] for k in rpbe_keys] if rpbe_keys else []
            # kappa is a FIXED constant on the formal method (no annealing)
            kap = float(args.kappa)
            pre = [p.detach().clone() for p in gamma_params]
            diag = apply_gamma_boundary_update(
                gamma=vla.gamma, gamma_params=gamma_params,
                optimizer=opt_gamma, scheduler=sched_gamma,
                task_pairs=task_pairs, task_cotangents=task_cots,
                rpbe_pairs=r_pairs, rpbe_cotangents=r_cots,
                kappa=kap, proj_iters=args.proj_iters,
                proj_iters_max=args.proj_iters_max,
                tau_feas=args.proj_tau,
                max_rounds=args.proj_max_rounds,
                max_active=args.proj_max_active,
                add_per_round=args.proj_add_per_round,
                grad_clip=args.grad_clip, minibatch=B, device="cuda")
            diag["proj_kappa"] = float(kap)
            n_opt += diag["gamma_steps"]
            upd = sum(float((p.detach() - p0).float().pow(2).sum())
                      for p, p0 in zip(gamma_params, pre)) ** 0.5
            shown = " ".join(f"{k}={v}" for k, v in diag.items()
                             if not k.startswith("_"))
            print(f"[gamma proj] pv={vla.cog_mem_bank.param_version} "
                  f"steps={diag['gamma_steps']} "
                  f"aborted={diag['gamma_aborted']} |dgamma|={upd:.3e} "
                  f"{shown}", flush=True)
        # param_version is the Gamma PARAMETER version: it advances only when
        # Gamma really changed (gamma_steps == 1).  It is what MergeRecord
        # stamps onto every merge and what the Gamma scheduler steps on, so an
        # aborted boundary never fabricates a version for untouched merges.
        if n_opt > 0:
            vla.cog_mem_bank.param_version += 1
        else:
            print("[gamma proj] aborted boundary: param_version stays "
                  f"{vla.cog_mem_bank.param_version}", flush=True)
        task_cotangents = {}
        rpbe_cotangents = {}
        rpbe_input_map = {}
        rpbe_pending_loss = []
        # free replay bookkeeping for episodes whose merges have all drained
        # (no longer pending futures).  Their records are not needed again:
        # cotangents were consumed above and re-accumulate per new episode.
        # Kept keys -> queue.pending episodes (still maturing futures).
        live = queue.pending_episodes()
        dead = [k for k in list(merge_registry) if k[0] not in live]
        for k in dead:
            merge_registry.pop(k, None)
        dead_m = [k for k in list(merge_id_map) if k[0] not in live]
        for k in dead_m:
            merge_id_map.pop(k, None)
        if dead or dead_m:
            print(f"[gamma step] freed registry {len(dead)} maps {len(dead_m)} "
                  f"(live eps {len(live)})", flush=True)

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
            lr_now = (opt_task.param_groups[-1]["lr"]
                      if opt_task is not None
                      else opt_gamma.param_groups[-1]["lr"])
            print(f"[lora] pv={vla.cog_mem_bank.param_version} "
                  f"lr={lr_now:.2e} mean|B|={mean_b:.3e} "
                  f"nzB={nz}/{n_b}", flush=True)

    def _ckpt_dict():
        # WEIGHTS-ONLY snapshot (reviewer ruling): never serialize optimizer
        # or scheduler state.  Avoids the Adam-state CPU-poisoning hazard and
        # the extra save-time memory peak entirely.
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
            "episodes_seen": episodes_seen,
            "config": {
                "sched": args.sched, "lr": args.lr,
                "gamma_lr": (args.gamma_lr if args.gamma_lr > 0 else args.lr),
                "weight_decay": 0.0, "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "gamma_replay_batch_size": args.gamma_replay_batch_size,
                "gamma_task_boundary_episodes": args.gamma_task_boundary_episodes,
                "rpbe_stats_episodes": args.rpbe_stats_episodes,
                "mem_length": args.mem_length, "kf_min_abs": args.kf_min_abs,
                "lambda_rpbe": args.lambda_rpbe,
                **boundary_config(args),
                "image_aug": args.image_aug, "dim_weight": args.dim_weight,
                "reg_head": args.reg_head,
            },
            "weights_snapshot_only": True,
        }
        return payload

    def _full_dict():
        """FULL-STATE checkpoint: weights + Adam + schedulers + step counters
        + CPU/CUDA/numpy RNG, for TRUE continuation (--resume-full).  A single
        rolling fullstate.pt per run (Adam fp32 state makes it ~4-5x weights)."""
        payload = _ckpt_dict()
        payload["weights_snapshot_only"] = False
        payload["full_state"] = True
        payload["opt_task"] = (opt_task.state_dict()
                               if opt_task is not None else None)
        payload["opt_gamma"] = (opt_gamma.state_dict()
                                if opt_gamma is not None else None)
        payload["sched_task"] = (sched_task.state_dict()
                                 if sched_task is not None else None)
        payload["sched_gamma"] = (sched_gamma.state_dict()
                                  if sched_gamma is not None else None)
        payload["rng_cuda"] = torch.cuda.get_rng_state()
        payload["rng_cpu"] = torch.get_rng_state()
        payload["rng_np"] = np.random.get_state()
        return payload

    best_val = float("inf")
    snapshot_set = {int(x) for x in args.snapshot_steps.split(",") if x.strip()}
    snapshot_saved = set()
    if args.init_from_weights:
        ck = torch.load(args.init_from_weights, map_location="cpu",
                        weights_only=False)
        named = dict(vla.named_parameters())
        unexpected = 0
        for n, t in ck["model"].items():
            if n in named and named[n].requires_grad:
                named[n].data.copy_(t.to(named[n].dtype))
            else:
                unexpected += 1
        # WEIGHTS-SNAPSHOT ONLY: do NOT restore optimizer/scheduler/RNG/
        # memory/queue/cursor state.  optimizer_step stays 0 and every
        # training structure is freshly initialized; this is an initialisation
        # of weights from a prior run, NOT a resumption, and its output must
        # never be spliced onto the prior run's trajectory.
        print(f"[weights-snapshot] loaded trainable weights from "
              f"{args.init_from_weights} (unexpected keys: {unexpected}); "
              f"training starts at opt 0 with fresh optimizer/scheduler/RNG",
              flush=True)

    if args.resume_full:
        assert not args.init_from_weights, \
            "use EITHER --init-from-weights OR --resume-full, not both"
        ck = torch.load(args.resume_full, map_location="cpu",
                        weights_only=False)
        named = dict(vla.named_parameters())
        for n, t in ck["model"].items():
            if n in named and named[n].requires_grad:
                named[n].data.copy_(t.to(named[n].dtype))
        if opt_task is not None:
            assert ck["opt_task"] is not None, \
                "checkpoint has no task-optimizer state"
            opt_task.load_state_dict(ck["opt_task"])
        # verify the run config the checkpoint was produced under BEFORE any
        # optimizer state is restored (reviewer: resume must not silently
        # change the boundary recipe, and Stage7->Stage8 must be explicit)
        reset_gamma_state = False
        legacy_gamma_ckpt = False
        if "config" in ck:
            want = {**common_config(args), **boundary_config(args)}
            bad, legacy_gamma_ckpt = verify_resume_config(
                ck["config"], want,
                allow_legacy_gamma=bool(args.migrate_legacy_gamma_state))
            if bad:
                raise SystemExit(f"[resume-full] CONFIG MISMATCH {bad}")
            if legacy_gamma_ckpt:
                reset_gamma_state = True
                print("[resume-full] LEGACY Stage7 checkpoint (no rpbe_mode): "
                      "allowing migration -- Gamma optimizer state (Adam "
                      "moments from the retired additive -lambda*J objective) "
                      "is RESET and the Gamma scheduler is realigned to "
                      "param_version.", flush=True)
            else:
                print("[resume-full] config verified against checkpoint",
                      flush=True)
        else:
            print("[resume-full] WARNING: checkpoint has no config block",
                  flush=True)
        if opt_gamma is not None:
            assert ck["opt_gamma"] is not None, "full ckpt has no gamma state"
            if not reset_gamma_state:
                opt_gamma.load_state_dict(ck["opt_gamma"])
        if sched_task is not None and ck["sched_task"] is not None:
            sched_task.load_state_dict(ck["sched_task"])
        if sched_gamma is not None and not reset_gamma_state:
            sched_gamma.load_state_dict(ck["sched_gamma"])
        optimizer_step = int(ck.get("optimizer_step", ck["step"]))
        micro_step = int(ck["micro_step"])
        episodes_seen = int(ck.get("episodes_seen", 0))
        vla.cog_mem_bank.param_version = int(ck["param_version"])
        if reset_gamma_state and sched_gamma is not None:
            # migration: opt_gamma got a FRESH Adam and the scheduler was not
            # loaded, so realign it to the boundary clock -- otherwise it would
            # sit at step 0 and hand out the warmup LR regardless of how far
            # the run had already progressed.
            lr0 = realign_lambda_scheduler(sched_gamma,
                                           vla.cog_mem_bank.param_version)
            print(f"[resume-full] sched_gamma realigned to param_version "
                  f"{vla.cog_mem_bank.param_version} -> lr={lr0}", flush=True)
        torch.set_rng_state(ck["rng_cpu"])
        torch.cuda.set_rng_state(ck["rng_cuda"])
        np.random.set_state(ck["rng_np"])
        print(f"[resume-full] TRUE continuation from {args.resume_full} at "
              f"opt {optimizer_step}/{args.max_steps} micro {micro_step} "
              f"pv {vla.cog_mem_bank.param_version}.  Data stream restarts a "
              f"fresh epoch (deterministic seed); bank/queue reset.", flush=True)
        assert optimizer_step < args.max_steps, \
            "resume point already at/past --max-steps; raise --max-steps"

    # ---- fixed independent batch + task-loss evaluator (RPBE mode intervention) ----
    fixed_batch = None

    def _task_loss_on_batch(batch):
        """Reg-head task loss on a FIXED batch under the CURRENT gamma params
        (memory banks reset; no grad).  Used for the second-order intervention
        along RPBE mode directions."""
        vla.cog_mem_bank.reset(); vla.per_mem_bank.reset()
        pv = batch["pixel_values"]
        if isinstance(pv, dict):
            pv = {k: v.to("cuda", dtype=torch.bfloat16) for k, v in pv.items()}
        else:
            pv = pv.to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                loss, _ = vla(input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                    actions=batch["actions"].to("cuda", dtype=torch.bfloat16),
                    action_masks=batch["action_masks"].to("cuda"),
                    pixel_values=pv, labels=batch["labels"].to("cuda"),
                    timesteps=batch["timesteps"], episode_ids=batch["episode_ids"],
                    output_hidden_states=True, repeated_diffusion_steps=1)
        return float(loss.item())

    if args.rpbe_comp_audit > 0:
        _ei = min(3, len(train_dataset.episodes) - 1)
        _rows = [r for i, r in zip(range(args.batch_size),
                                    train_dataset.iter_episode(_ei))]
        fixed_batch = collator(_rows)
        print(f"[interv] fixed eval batch: {len(_rows)} rows from episode {_ei}",
              flush=True)

    # ---- main loop ----
    it = iter(train_loader)
    while optimizer_step < args.max_steps:
        batch = next(it)
        micro_step += 1

        # gamma macro boundary fires at the TOP of a batch, once a completed
        # episode has crossed the repr threshold (set by feed_merges_and_-
        # futures below).  Firing here -- before any frame of the next episode
        # is processed -- guarantees an episode never mixes two gamma
        # versions: merges only form after mem_length>=16 banked frames, and
        # the fire happens strictly before that can occur.
        if IS_GAMMA and boundary_pending:
            scale = args.grad_accum / max(1, window_micro)
            _gamma_boundary(scale)
            episodes_since_boundary = 0
            window_micro = 0
            boundary_pending = False

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
        window_micro += 1

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
            # threshold crossed -> fire at the TOP of the next batch (fresh
            # episode edge), never version-splitting an in-progress episode.
            if episodes_since_boundary >= args.gamma_task_boundary_episodes:
                boundary_pending = True

        # dense optimizer step every grad_accum micro-batches
        if micro_step % args.grad_accum == 0:
            audit_snap = (_audit_snapshot()
                          if args.opt_audit and optimizer_step < args.opt_audit
                          else None)
            if opt_task is not None:
                torch.nn.utils.clip_grad_norm_(task_params, args.grad_clip)
                opt_task.step()
                sched_task.step()
            # Clear EVERY parameter's grad, not just the optimizer's params.
            # Under lora-gamma the frozen banks/DiT still receive grads from
            # the backward, and the gamma task cotangents are read straight off
            # bank-leaf .grad -- letting them accumulate would both grow them
            # unboundedly and corrupt those cotangents.
            vla.zero_grad(set_to_none=True)
            optimizer_step += 1
            if audit_snap is not None:
                _audit_report(optimizer_step, audit_snap)
            if args.opt_audit and optimizer_step >= args.opt_audit:
                print(f"[opt-audit] DONE after {optimizer_step} dense steps",
                      flush=True)
                break

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
                if not args.no_fullstate:
                    torch.save(_full_dict(), run_dir / "fullstate.pt")
                    print(f"latest+fullstate saved @ opt {optimizer_step}",
                          flush=True)
                else:
                    print(f"latest saved @ opt {optimizer_step} "
                          f"(fullstate disabled)", flush=True)

            if optimizer_step in snapshot_set and \
                    optimizer_step not in snapshot_saved:
                snapshot_saved.add(optimizer_step)
                torch.save(_ckpt_dict(),
                           run_dir / f"snapshot_{optimizer_step}.pt")
                print(f"weights snapshot saved @ opt {optimizer_step}",
                      flush=True)

    if args.opt_audit:
        print("[opt-audit] audit complete; skipping final checkpoint save",
              flush=True)
        sys.exit(0)

    torch.save(_ckpt_dict(), run_dir / "checkpoint.pt")
    with open(run_dir / "dataset_statistics.json", "w") as f:
        json.dump(dataset_statistics, f, indent=2)
    print("== training done ==", flush=True)
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
