"""
train_libero_mem_rpbe.py — single-GPU MemoryVLA training on LIBERO-Mem
with RPBE plugin.

Arms:
  avg / avg-boundary : official average merge host control.  LoRA steps on
                       the shared repr macro clock (every
                       --repr-boundary-episodes finished demos).
  avg-dense          : DIAGNOSTIC ONLY (not in the 3-arm table).  LoRA steps
                       every grad-accum block to probe whether a dense-LoRA
                       official host can learn the task.
  gamma-task         : Gamma merge, task-gradient replay only
  gamma-rpbe         : Gamma merge, task-gradient + RPBE dual-adjoint replay

Plan §4/§25/§26/§27: two optimizers.
  opt_task (retrieval/gate/per_compr/DiT) steps every grad-accum block.
  opt_repr (LoRA + Gamma) steps at the shared repr macro boundary (every
  --repr-boundary-episodes episodes, IDENTICAL across all boundary arms),
  with the repr step DEFERRED to a clean episode edge so a demo is never
  version-split.
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
    """Semantic booleans for every arm string.  Single source of truth so a
    new arm can never be silently mis-gated by a scattered `arm == "avg"`."""
    return dict(
        is_gamma=arm in ("gamma-task", "gamma-rpbe"),
        is_rpbe=arm == "gamma-rpbe",
        is_dense=arm == "avg-dense",
        is_avg_boundary=arm in ("avg", "avg-boundary"),
    )


def count_finished_episodes(eids, last_eid):
    """Number of episode boundaries crossed in a row list, mirroring the
    episode-drain detection inside feed_merges_and_futures.  `episode_ids`
    are per-episode global ints, monotonic within an episode (hdf5_dataset
    L122-123), so a change of value == one finished episode."""
    n = 0
    for eid in eids:
        if last_eid is not None and eid != last_eid:
            n += 1
        last_eid = eid
    return n, last_eid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", choices=["avg", "avg-boundary", "avg-dense",
                                     "gamma-task", "gamma-rpbe"],
                   default="avg",
                   help="avg / avg-boundary: host control, LoRA steps on the "
                        "shared repr macro clock.  avg-dense: diagnostic only, "
                        "LoRA steps every grad-accum block (not in the 3-arm "
                        "table).  gamma-task/gamma-rpbe: Gamma merge arms.")
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
    # repr (LoRA) scheduler in units of ACTUAL repr steps (param_version),
    # shared by every boundary arm (blocker 4).
    p.add_argument("--repr-warmup-steps", type=int, default=10,
                   help="warmup length for the repr (LoRA) cosine in units of "
                        "repr steps")
    p.add_argument("--repr-total-steps", type=int, default=600,
                   help="cosine denominator for the repr (LoRA) scheduler in "
                        "units of repr steps (calibrate from a short-run firing "
                        "rate, then override)")
    return p.parse_args()


def build_vla(args: argparse.Namespace):
    print("== loading MemoryVLA ==", flush=True)
    arm = args.arm
    flags = arm_flags(arm)
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
        gamma_alpha_init=0.0,
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


def _params_of(module_list, prefix=""):
    out = []
    for m in module_list:
        out += [p for p in m.parameters() if p.requires_grad]
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    flags = arm_flags(args.arm)
    IS_GAMMA, IS_RPBE, IS_DENSE = (flags["is_gamma"], flags["is_rpbe"],
                                   flags["is_dense"])
    IS_AVG_BOUNDARY = flags["is_avg_boundary"]
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

    def _snapshot_bank(bank):
        """Deepcopy every piece of CogMemBank/PerMemBank runtime state that
        training mutates, so an intervening eval can be rolled back exactly."""
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

    def run_eval(step: int) -> float:
        """Val-split action loss over min(40, len) batches.

        RNG isolation: action_model.loss samples diffusion noise + timesteps
        with torch.randn_like/torch.randint (global CUDA RNG).  Eval must
        therefore run on its OWN fixed seed and restore the Python / NumPy /
        CPU-Torch / CUDA RNG states afterwards, or it perturbs the training
        noise stream.  try/finally guarantees bank state + RNG are restored
        even if eval raises.  Returns the mean val loss (inf on failure)."""
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
            # fixed eval seed: same number of val batches drawn every call
            # regardless of how many train steps happened since the last eval
            torch.manual_seed(1234 + step % 1000)
            torch.cuda.manual_seed(1234 + step % 1000)
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
            # restore bank state + RNG unconditionally (blocker 6)
            _restore_bank(cog, cog_snap)
            _restore_bank(per, per_snap)
            torch.set_rng_state(rng_snapshot[0])
            torch.cuda.set_rng_state(rng_snapshot[1])
            np.random.set_state(rng_snapshot[2])
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

    # cosine decay with warmup.  Denominator is the optimizer's OWN step
    # count, not raw data steps:
    #   opt_task steps once per grad-accum block -> ceil(max_steps/grad_accum)
    #   opt_repr (boundary arms) steps once per repr macro boundary; its LR is
    #     indexed by the ACTUAL repr step count (param_version), shared across
    #     all three boundary arms so the table stays fair (blocker 4).
    task_total_updates = int(math.ceil(args.max_steps / args.grad_accum))

    def lr_lambda_task(t):
        if t < args.warmup_steps:
            return t / max(1, args.warmup_steps)
        progress = (t - args.warmup_steps) / max(
            1, task_total_updates - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def lr_lambda_repr(rs):
        # rs == param_version == true repr-step counter (advanced in the same
        # place sched_repr.step() is called); warmup/denominator are given in
        # repr-step units.
        if rs < args.repr_warmup_steps:
            return rs / max(1, args.repr_warmup_steps)
        progress = (rs - args.repr_warmup_steps) / max(
            1, args.repr_total_steps - args.repr_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched_task = torch.optim.lr_scheduler.LambdaLR(opt_task, lr_lambda_task)
    if IS_DENSE:
        # diagnostic arm: LoRA steps every grad-accum block like opt_task, so
        # reuse the task schedule (blocker 4 keeps this OUT of the 3-arm table)
        sched_repr = torch.optim.lr_scheduler.LambdaLR(opt_repr, lr_lambda_task)
    else:
        # all boundary arms (avg / avg-boundary / gamma-task / gamma-rpbe)
        # share ONE repr-step-indexed schedule -> identical LR at identical
        # param_version (blocker 4 fairness)
        sched_repr = torch.optim.lr_scheduler.LambdaLR(opt_repr, lr_lambda_repr)
    print(f"task params: {sum(p.numel() for p in task_params)/1e6:.2f}M | "
          f"repr params: {sum(p.numel() for p in repr_params)/1e6:.2f}M",
          flush=True)

    # --- RPBE machinery (arm-gated) ---
    rpbe_cfg = EmbodiedRPBConfig(
        kf_variant=args.kf_variant, kf_min_abs=args.kf_min_abs,
        lambda_rpbe=args.lambda_rpbe, rpbe_seed=args.seed)
    maps = EmbodiedFixedMaps(rpbe_cfg) if IS_RPBE else None
    queue = PendingMergeQueue() if IS_GAMMA else None
    window = EmbodiedRPBEWindow(
        variant=rpbe_cfg.kf_variant, eps=rpbe_cfg.ridge_eps,
        min_abs=rpbe_cfg.kf_min_abs) if IS_RPBE else None
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
    window_micro = 0       # micro-backwards accumulated since last repr step
    boundary_pending = False  # threshold crossed; step deferred to clean edge
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

    def _do_repr_boundary(scale):
        """One repr macro-boundary update.  Called ONLY on a clean episode
        edge (blocker 5), so no demo is ever version-split.

        Gradient normalization (blockers 2 + 3): accumulated LoRA/task grads
        were summed over `window_micro` micro-backwards each scaled by
        1/grad_accum, i.e. total = (1/G) * sum.  Dividing by N gives the
        per-micro mean -> multiply repr grads by scale = grad_accum/window_micro
        BEFORE any replay adds its own contribution, so the relative weight of
        task-vs-RPBE replay is not distorted."""
        nonlocal window, task_cotangents, rpbe_cotangents, rpbe_pending_loss
        if vla.gamma is not None:
            # RPBE-only state: close statistics window, cache cotangents.
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
            # Scale the accumulated window grads to per-micro mean BEFORE any
            # replay backward adds to the same repr_params (blocker 3).
            if scale != 1.0:
                for p in repr_params:
                    if p.grad is not None:
                        p.grad.mul_(scale)
            # task replay (independent key set from RPBE)
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
                # same per-micro normalization applies to the replay grads
                (scale * l_task).backward()
                print(f"[repr step] task replay {l_task.item():.4f}",
                      flush=True)
            # rpbe replay (independent key set, NOT scaled by the task clock;
            # its own window already aggregates over the boundary)
            rkeys = [k for k in rpbe_cotangents if k in merge_registry]
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
        else:
            # avg arms: repr_params == pure LoRA; scale the summed window
            # grads to the per-micro mean (blocker 2)
            if scale != 1.0:
                for p in repr_params:
                    if p.grad is not None:
                        p.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(repr_params, args.grad_clip)
        opt_repr.step()
        opt_repr.zero_grad()
        sched_repr.step()
        vla.cog_mem_bank.param_version += 1
        _log_lora_norm()
        task_cotangents = {}
        rpbe_cotangents = {}
        rpbe_pending_loss = []
        # NOTE: merge_registry is NOT cleared -- merges still pending in the
        # queue need their records for the next replay.

    def _log_lora_norm():
        """Mean lora_B abs-norm at repr-step frequency.  Frozen baseline has
        all-zero B; a non-zero mean proves the repr clock steps the adapter."""
        b_norms = []
        n_b = 0
        for n, p in vla.named_parameters():
            if "lora_B" in n:
                n_b += 1
                b_norms.append(float(p.detach().float().abs().mean()))
        if b_norms:
            mean_b = sum(b_norms) / len(b_norms)
            nz = sum(1 for v in b_norms if v > 0.0)
            lr_now = opt_repr.param_groups[0]["lr"]
            print(f"[repr step] pv={vla.cog_mem_bank.param_version} "
                  f"lr={lr_now:.2e} mean|B|={mean_b:.3e} "
                  f"nzB={nz}/{n_b}", flush=True)

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
            # deferred repr macro boundary: fire at the top of the first batch
            # that begins a NEW episode (blocker 5: never version-split a demo
            # already in progress).  scale normalizes the summed window grads
            # to the per-micro mean (blockers 2/3).
            if boundary_pending and not IS_DENSE:
                first_eid = int(batch["episode_ids"][0])
                if last_eid is None or first_eid != last_eid:
                    scale = args.grad_accum / max(1, window_micro)
                    _do_repr_boundary(scale)
                    episodes_since_boundary = 0
                    window_micro = 0
                    boundary_pending = False
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
            window_micro += 1

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

            # count finished episodes for the shared repr macro clock.
            # gamma: inside feed_merges_and_futures (it also drains merges).
            # avg-boundary: pure count, NO queue/window/maps state (blocker 1).
            # avg-dense: no macro clock at all -> not counted here.
            if IS_GAMMA:
                feed_merges_and_futures(batch)
            elif IS_AVG_BOUNDARY:
                eids = [int(e) for e in batch["episode_ids"]]
                n_done, last_eid = count_finished_episodes(eids, last_eid)
                episodes_since_boundary += n_done
            # flag that the boundary threshold has been crossed; the actual
            # opt_repr step is DEFERRED to the next batch that starts a fresh
            # episode (blocker 5: never version-split an in-progress demo).
            if not IS_DENSE and episodes_since_boundary >= args.repr_boundary_episodes:
                boundary_pending = True

            # dense diagnostic arm: LoRA steps every grad-accum block, at the
            # same cadence as opt_task (blocker 1 / plan avg-dense).
            if IS_DENSE and step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(repr_params, args.grad_clip)
                opt_repr.step()
                opt_repr.zero_grad()
                sched_repr.step()
                vla.cog_mem_bank.param_version += 1
                if step % (args.grad_accum * 50) == 0:
                    _log_lora_norm()

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

            if step % args.eval_every == 0 and args.eval_every > 0:
                val_loss = run_eval(step)
                # B7: best-val selection with a dedicated checkpoint
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(_ckpt_dict(), run_dir / "best.pt")
                    print(f"[best @ {step}] val {val_loss:.4f}", flush=True)

            if step % args.checkpoint_every == 0:
                # rolling "latest" overwrite (long-run retention: only
                # best.pt + latest.pt accumulate; checkpoint.pt is written
                # once at the very end)
                torch.save(_ckpt_dict(), run_dir / "latest.pt")
                print(f"latest saved @ {step}", flush=True)

    # final checkpoint keeps FULL metadata via _ckpt_dict() (arm/best_val/
    # task_filter/seed), not a bare weights-only dict (approval small fix 2)
    torch.save(_ckpt_dict(), run_dir / "checkpoint.pt")
    with open(run_dir / "dataset_statistics.json", "w") as f:
        json.dump(dataset_statistics, f, indent=2)
    print("== training done ==", flush=True)
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
