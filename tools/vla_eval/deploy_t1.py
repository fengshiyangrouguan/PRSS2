"""Two-stage deploy for RPBE-VLA checkpoints (base + trainable delta).

The RPBE trainer's `_ckpt_dict()` writes a FLAT `named_parameters()` dump of the
trainable subset only (880 tensors / 1.84 GB for the T1 `avg` arm). `deploy.py`
goes through `load_vla` -> `MemoryVLA.from_pretrained`, which consumes a
MODULE-NESTED dict and asserts `projector` + `llm_backbone` are present. The two
formats cannot be reconciled by a plain dict update.

Rather than physically merging a ~30 GB checkpoint, this patches
`vla.load_vla` to restore in memory:

    1. build from the official BASE checkpoint (= frozen 7B backbone)
    2. re-attach LoRA with the config recorded inside the delta
    3. load the delta back module-by-module, strict

...then hands control to the UNMODIFIED `deploy.py`, so the Flask contract is
byte-for-byte the one the evaluator already speaks.

Env:
    T1_BASE_CKPT   official openvla-7b-prismatic checkpoints/*.pt
    T1_DELTA_CKPT  the RPBE training checkpoint (best.pt / snapshot_*.pt)
    T1_VERIFY_ONLY 1 = restore + verify equality, print a report, exit (no Flask)
"""
from __future__ import annotations

import collections
import json
import os
import runpy
import sys
from pathlib import Path

MEMVLA = Path(os.environ.get(
    "MEMVLA_DIR", "/root/autodl-tmp/PRSS2/third_party/memoryvla"))
sys.path.insert(0, str(MEMVLA))
sys.path.insert(0, os.environ.get("PRSS2SRC", "/root/autodl-tmp/PRSS2/src"))

import torch                                    # noqa: E402

import vla as vla_mod                           # noqa: E402
from peft import LoraConfig, get_peft_model     # noqa: E402

BASE_CKPT = os.environ.get("T1_BASE_CKPT", "")
DELTA_CKPT = os.environ.get("T1_DELTA_CKPT", "")
VERIFY_ONLY = os.environ.get("T1_VERIFY_ONLY", "0") == "1"

_orig_load_vla = vla_mod.load_vla


def _group_delta(delta: dict) -> dict[str, dict]:
    """Flat `named_parameters()` paths -> {module_path: {relative_key: tensor}}.

    `vlm.llm_backbone.<rest>` targets the submodule `vlm.llm_backbone`;
    everything else targets `getattr(model, first_segment)`.
    """
    groups: dict[str, dict] = collections.defaultdict(dict)
    for k, t in delta.items():
        parts = k.split(".")
        if k.startswith("vlm."):
            mod, sub = "vlm." + parts[1], ".".join(parts[2:])
        else:
            mod, sub = parts[0], ".".join(parts[1:])
        groups[mod][sub] = t
    return dict(groups)


def _resolve(model, mod_path: str):
    target = model
    for seg in mod_path.split("."):
        target = getattr(target, seg, None)
        if target is None:
            return None
    return target


def restore(base_ckpt: str, delta_ckpt: str, verbose: bool = True, **load_kwargs):
    """base (frozen backbone) + delta (trainable) -> a ready MemoryVLA."""
    if not base_ckpt or not Path(base_ckpt).exists():
        raise FileNotFoundError(f"base checkpoint not found: {base_ckpt!r}")
    if not delta_ckpt or not Path(delta_ckpt).exists():
        raise FileNotFoundError(f"delta checkpoint not found: {delta_ckpt!r}")

    payload = torch.load(delta_ckpt, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise KeyError(f"{delta_ckpt} has no 'model' key (keys: "
                       f"{list(payload.keys())})")
    delta = payload["model"]
    lora_cfg = payload.get("lora_config")
    if verbose:
        print(f"[restore] delta={Path(delta_ckpt).name} "
              f"tensors={len(delta)} arm={payload.get('arm')!r} "
              f"step={payload.get('step')}", flush=True)
        print(f"[restore] lora_config={lora_cfg}", flush=True)

    # ---- 1) frozen backbone straight from the official base -------------
    if verbose:
        print(f"[restore] building from base {Path(base_ckpt).name}", flush=True)
    model = _orig_load_vla(model_id_or_path=base_ckpt,
                           load_for_training=False, **load_kwargs)

    # ---- 2) re-attach LoRA exactly as training did ----------------------
    if lora_cfg is not None:
        peft_cfg = LoraConfig(
            r=lora_cfg["r"], lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules="all-linear",
        )
        model.vlm.llm_backbone.llm = get_peft_model(
            model.vlm.llm_backbone.llm, peft_cfg)
        if verbose:
            print("[restore] LoRA re-attached", flush=True)
    else:
        raise KeyError("delta payload has no 'lora_config'; cannot rebuild "
                       "the adapter structure")

    # ---- 2b) the T1 normalisation statistics ---------------------------
    # Building from the BASE means MemoryVLA picked up the BASE's
    # dataset_statistics.json, which is the OXE one. Our policy was trained on
    # T1, so the stats must come from the delta's own run dir -- otherwise
    # `--unnorm_key libero_mem_no_noops` fails _check_unnorm_key.
    stats_path = Path(delta_ckpt).parent / "dataset_statistics.json"
    if stats_path.exists():
        with open(stats_path) as fh:
            model.norm_stats = json.load(fh)
        if verbose:
            print(f"[restore] norm_stats <- {stats_path.name} "
                  f"(keys: {list(model.norm_stats)})", flush=True)
    else:
        print(f"[restore] WARNING: {stats_path} absent; unnorm_key will "
              f"resolve against the BASE's stats and likely fail", flush=True)

    # ---- 3) load the delta back, module by module, strict ---------------
    groups = _group_delta(delta)
    report = []
    for mod_path, sub in sorted(groups.items()):
        target = _resolve(model, mod_path)
        if target is None:
            raise RuntimeError(f"[restore] module {mod_path!r} not on the model")
        res = target.load_state_dict(sub, strict=False)
        if res.unexpected_keys:
            raise RuntimeError(
                f"[restore] {mod_path}: unexpected keys "
                f"{list(res.unexpected_keys)[:5]} -- refusing to continue")
        report.append((mod_path, len(sub), len(res.missing_keys)))

    if verbose:
        print("[restore] delta applied:")
        for mod_path, supplied, frozen in report:
            print(f"           {mod_path:26s} supplied={supplied:<4d} "
                  f"frozen_untouched={frozen}")
    return model, delta, report


def _patched_load_vla(model_id_or_path=None, load_for_training=False, **kwargs):
    """Replaces vla.load_vla for this process: whatever path deploy.py was
    pointed at, we restore from BASE + DELTA instead."""
    model, _delta, _rep = restore(BASE_CKPT, DELTA_CKPT, verbose=True, **kwargs)
    return model


if VERIFY_ONLY:
    base_kw = {
        "use_bf16": True, "action_dim": 7, "future_action_window_size": 15,
        "action_model_type": "DiT-L", "use_ema": False,
        "dataloader_type": "stream", "mem_length": 16, "retrieval_layers": 2,
        "use_timestep_pe": True, "fusion_type": "gate",
        "consolidate_type": "tome", "update_fused": False,
        "per_token_size": 256, "use_rpbe_gamma": False, "gamma_rank": 64,
        "gamma_alpha_init": 1.0, "rpbe_merge_records": False,
        "rpbe_task_grad": False, "rpbe_seed": 42,
    }
    print("=" * 74)
    print("T1 RESTORE VERIFY-ONLY (no Flask, no rollout)")
    print("=" * 74, flush=True)
    model, delta, report = restore(BASE_CKPT, DELTA_CKPT, verbose=True,
                                   **base_kw)

    print("\n--- equality check: does every delta tensor actually sit in the model? ---")
    live = dict(model.named_parameters())
    n_ok = n_bad = n_missing = 0
    for k, t in delta.items():
        p = live.get(k)
        if p is None:
            n_missing += 1
            continue
        if tuple(p.shape) != tuple(t.shape):
            n_bad += 1
            continue
        if torch.equal(p.detach().to(t.dtype).cpu(), t.cpu()):
            n_ok += 1
        else:
            n_bad += 1
    print(f"  identical : {n_ok}")
    print(f"  differing : {n_bad}")
    print(f"  missing   : {n_missing}")
    print(f"  total     : {len(delta)}")
    verdict = (n_bad == 0 and n_missing == 0 and n_ok == len(delta))
    print("\nVERIFY:", "PASS" if verdict else "FAIL")
    print("no Flask started, nothing written")
    raise SystemExit(0 if verdict else 1)

vla_mod.load_vla = _patched_load_vla
print(f"[deploy_t1] load_vla patched: base={Path(BASE_CKPT).name} "
      f"delta={Path(DELTA_CKPT).name}", flush=True)
runpy.run_path(str(MEMVLA / "deploy.py"), run_name="__main__")
