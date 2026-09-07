"""
hdf5_dataset.py

LIBERO-Mem HDF5 dense-stream dataset (RPBE-VLA). Reproduces the OFFICIAL
MemoryVLA / OpenVLA dense sliding-window training protocol on top of the
LIBERO-Mem HDF5 format:

  * prediction length K = future_action_window_size + 1 = 16
  * training stride = 1: for EVERY physical frame t we emit
        image[t] -> actions[t : t+16] ,  timestep = t
    so a 276-frame demo yields 276 samples (official dense sliding window,
    NOT the old n_decisions = T//16 / k = d*16 sparse adapter).
  * rows within an episode are strictly in-frame order; episodes are
    re-shuffled per epoch deterministically (num_workers=0).
  * tail <16 frames: pad with official-style NEUTRAL actions (6-dim zero
    action, gripper = last seen state) and a correct action_mask that flags
    which of the 16 target steps are real vs padding.

Gripper protocol (official libero_dataset_transform):
  env/data gripper  -1 = open ,  +1 = close
  model gripper       1 = open ,   0 = close
  conversion:  g_model = 1.0 - clip(g_env, 0, 1)
The gripper channel is NOT BOUNDS_Q99-normalized: it stays {0,1} in model
space (mask[6] == False), matching memory_vla.predict_action's <0.5
threshold which is defined on the official {0,1} labels.

Split convention: train = demo_1..80, val = demo_81..100 (no official val
release).  Action statistics computed on the TRAIN split only.

This module deliberately does NOT import vla.datasets.datasets (the TF /
RLDS chain), so it works without tensorflow.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

IGNORE_INDEX = -100

# official LIBERO-Mem convention: 100 released demos per task; split 80/20
TRAIN_DEMO_RANGE = (1, 80)   # inclusive
VAL_DEMO_RANGE = (81, 100)   # inclusive

# BOUNDS_Q99 epsilon used by the official RLDS pipeline
NORM_EPS = 1e-8


def env_to_model_gripper(g_env):
    """Official gripper relabel: data/env -1=open,+1=close -> model 1=open,
    0=close (g_model = 1 - clip(g_env, 0, 1))."""
    return 1.0 - np.clip(np.asarray(g_env, dtype=np.float32), 0.0, 1.0)


class HDF5Collator(PaddedCollatorForActionPrediction):
    """Official action-prediction collator + pass-through of the per-row
    `instruction` string (needed for the RPBE fixed context map)."""

    def __call__(self, instances):
        out = super().__call__(instances)
        out["instruction"] = [inst["instruction"] for inst in instances]
        return out


@dataclass
class HDF5BatchTransform:
    """Row transform replicating RLDSBatchTransform prompt/label logic with
    official BOUNDS_Q99 normalization on the 6 continuous dims.  The gripper
    (dim 6) and any q01==q99 dim (e.g. rotation==0 here) are NOT normalized:
    mask False keeps their raw model-space value."""
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Any
    action_q01: np.ndarray      # [7]
    action_q99: np.ndarray      # [7]
    action_mask: np.ndarray     # [7] bool: True where q01 != q99
    predict_stop_token: bool = True
    action_window: int = 16

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        img = Image.fromarray(row["agentview_rgb"])            # uint8 HWC
        lang = row["instruction"].lower()

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": ""},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # actions already in MODEL space (gripper {0,1}); shape [K, 7]
        a = np.asarray(row["actions"], dtype=np.float32)
        # BOUNDS_Q99 normalization on masked dims only; unmasked (gripper /
        # zero-range) dims keep their raw value (NOT zeroed -- official).
        norm = np.where(
            self.action_mask[None, :],
            2.0 * (a - self.action_q01[None, :])
            / (self.action_q99[None, :] - self.action_q01[None, :] + NORM_EPS) - 1.0,
            a,
        )
        norm = np.clip(norm, -1.0, 1.0)
        actions = torch.tensor(norm, dtype=torch.float32)

        # Mask prompt tokens before the first <EOS-ish> token id 2
        eos_positions = torch.where(input_ids == 2)[0]
        if len(eos_positions) > 0:
            labels[: int(eos_positions[0])] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=row["task_name"],
            instruction=row["instruction"],
            actions=actions,
            action_masks=torch.from_numpy(np.asarray(
                row["valid_mask"], dtype=bool)),
            timesteps=np.array([row["t"]], dtype=np.int64),
            episode_ids=np.array([row["episode_idx"]], dtype=np.int64),
        )


def scan_episodes(
    data_root: Path, split: str, task_filter: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Scan h5 files + metainfo.json -> episode manifest.

    Returns (episodes, demo_range_by_task).  Each episode:
      {task_name, h5_path, demo_key, instruction, T}
    Split rule: train = demo_1..demo_80, val = demo_81..demo_100.
    """
    lo, hi = TRAIN_DEMO_RANGE if split == "train" else VAL_DEMO_RANGE
    meta_path = data_root / "metainfo.json"
    with open(meta_path) as f:
        meta = json.load(f)

    episodes: List[Dict[str, Any]] = []
    demo_range_by_task: Dict[str, List[int]] = {}
    for h5_path in sorted(data_root.glob("*.hdf5")):
        task_name = h5_path.stem.replace("_demo", "")
        if task_filter and task_filter not in task_name:
            continue
        if task_name not in meta:
            raise ValueError(f"task {task_name} missing from metainfo.json")
        task_meta = meta[task_name]
        demo_keys = [f"demo_{i}" for i in range(lo, hi + 1)]
        missing = [k for k in demo_keys if k not in task_meta]
        if missing:
            print(f"[scan] task {task_name}: {len(missing)} demos missing from "
                  f"metainfo, skipping: {missing}", flush=True)
            demo_keys = [k for k in demo_keys if k not in missing]
        with h5py.File(h5_path, "r") as f:
            available = set(f["data"].keys())
        missing_h5 = [k for k in demo_keys if k not in available]
        if missing_h5:
            print(f"[scan] task {task_name}: {len(missing_h5)} demos missing from "
                  f"hdf5, skipping: {missing_h5}", flush=True)
            demo_keys = [k for k in demo_keys if k not in missing_h5]
        with h5py.File(h5_path, "r") as f:
            for dk in demo_keys:
                T = f["data"][dk]["actions"].shape[0]
                instruction = task_meta[dk]["task_description"]
                episodes.append(dict(
                    task_name=task_name,
                    h5_path=str(h5_path),
                    demo_key=dk,
                    instruction=instruction,
                    T=int(T),
                ))
        demo_range_by_task[task_name] = [lo, hi]
    return episodes, demo_range_by_task


def compute_action_stats(episodes: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """BOUNDS_Q99 statistics over ALL frame-level actions of the given
    episodes (no truncation to multiples of 16).  Gripper is relabeled to
    model space {0,1} BEFORE stats.  Dims that are constant (e.g. rotation
    ==0 in this task) yield q01==q99 and are auto-masked out.  The gripper
    dim (6) is FORCED mask False: it stays raw {0,1} in model space and is
    never BOUNDS-normalized (official {0,1} protocol, predict_action <0.5
    threshold is defined on it)."""
    all_actions: List[np.ndarray] = []
    for ep in episodes:
        with h5py.File(ep["h5_path"], "r") as f:
            a = f["data"][ep["demo_key"]]["actions"][:].astype(np.float32)
        a = a.copy()
        a[:, 6] = env_to_model_gripper(a[:, 6])
        all_actions.append(a)
    cat = np.concatenate(all_actions, axis=0)            # [N, 7] frame level
    q01 = np.percentile(cat, 1, axis=0).astype(np.float32)
    q99 = np.percentile(cat, 99, axis=0).astype(np.float32)
    mask = (q01 != q99)
    mask[6] = False          # gripper stays {0,1}, never normalized
    return {"q01": q01, "q99": q99, "mask": mask}


class HDF5DenseDataset(IterableDataset):
    """One row per PHYSICAL frame (dense sliding window).  Episodes are
    re-shuffled per epoch; within an episode rows stay in strict frame order.
    num_workers MUST be 0 so CogMemBank 'stream' semantics hold."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        batch_transform: HDF5BatchTransform,
        q01: np.ndarray,
        q99: np.ndarray,
        mask: np.ndarray,
        action_window: int = 16,
        seed: int = 0,
        repeat: bool = True,
        task_filter: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "val")
        self.data_root = Path(data_root)
        self.split = split
        self.batch_transform = batch_transform
        self.action_window = action_window
        self.q01 = np.asarray(q01, dtype=np.float32)
        self.q99 = np.asarray(q99, dtype=np.float32)
        self.mask = np.asarray(mask, dtype=bool)
        self.seed = seed
        self.repeat = repeat

        self.episodes, _ = scan_episodes(self.data_root, split,
                                         task_filter=task_filter)
        # preload each episode's raw actions (model space) once per epoch lazily
        self._cache = {}

    def __len__(self) -> int:
        return len(self.episodes)

    def _load(self, idx):
        if idx not in self._cache:
            ep = self.episodes[idx]
            with h5py.File(ep["h5_path"], "r") as f:
                a = f["data"][ep["demo_key"]]["actions"][:].astype(np.float32)
                rgb = np.asarray(f["data"][ep["demo_key"]]["obs"]["agentview_rgb"])
            a = a.copy()
            a[:, 6] = env_to_model_gripper(a[:, 6])     # {0,1}
            self._cache[idx] = (ep, a, rgb)
        return self._cache[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        order = list(range(len(self.episodes)))
        if self.split == "train":
            rng = np.random.default_rng(self.seed + (worker_info.id if worker_info else 0))
            rng.shuffle(order)

        while True:
            self._cache = {}    # fresh cache each epoch
            for idx in order:
                ep, a, rgb = self._load(idx)
                T = a.shape[0]
                K = self.action_window
                # neutral tail action (model space): continuous dims = their
                # norm-zero value = q01-midpoint only if masked; unmasked
                # dims already q01==q99 so their 'neutral' is meaningless; we
                # fill continuous masked dims with 0 in NORMALIZED space by
                # letting transform see a flag?  Instead we build the neutral
                # as the raw action equal to the BOUNDS midpoint (maps to 0).
                tail = T - 0  # frames available beyond t
                # For each physical frame t emit a K-step window.
                for t in range(T):
                    n_avail = T - t
                    if n_avail >= K:
                        chunk = a[t:t + K]
                        valid = np.ones(K, dtype=bool)
                    else:
                        # neutral padding for the missing tail
                        chunk = np.zeros((K, 7), dtype=np.float32)
                        chunk[:n_avail] = a[t:T]
                        # neutral continuous value = midpoint of q01/q99 for
                        # masked dims (normalized -> 0); for gripper keep last
                        # known state (model space).
                        last_g = a[T - 1, 6]
                        for j in range(n_avail, K):
                            chunk[j, 6] = last_g
                            for d in range(6):
                                if self.mask[d]:
                                    chunk[j, d] = 0.5 * (self.q01[d] + self.q99[d])
                                else:
                                    chunk[j, d] = 0.0
                        valid = np.zeros(K, dtype=bool)
                        valid[:n_avail] = True
                    row = dict(
                        agentview_rgb=rgb[t],
                        actions=chunk,
                        valid_mask=valid,
                        instruction=ep["instruction"],
                        task_name=ep["task_name"],
                        t=t,
                        episode_idx=idx,
                    )
                    yield self.batch_transform(row)
            if not self.repeat:
                return


def get_hdf5_decision_stream_dataset_and_collator(
    data_root: Path,
    tokenizer: PreTrainedTokenizerBase,
    image_transform: ImageTransform,
    prompt_builder_fn: Any,
    future_action_window_size: int = 15,
    seed: int = 0,
    split: str = "train",
    model_max_length: int = 2048,
    pad_token_id: int = 0,
    task_filter: Optional[str] = None,
):
    """Build dataset + action stats + collator for a single split.

    Action statistics are computed on the TRAIN split only (both splits
    share the same normalization, as in the official pipeline)."""
    action_window = future_action_window_size + 1
    train_episodes, _ = scan_episodes(Path(data_root), "train",
                                      task_filter=task_filter)
    stats = compute_action_stats(train_episodes)

    transform = HDF5BatchTransform(
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=prompt_builder_fn,
        action_q01=stats["q01"],
        action_q99=stats["q99"],
        action_mask=stats["mask"],
        action_window=action_window,
    )

    dataset = HDF5DenseDataset(
        data_root=Path(data_root),
        split=split,
        batch_transform=transform,
        q01=stats["q01"],
        q99=stats["q99"],
        mask=stats["mask"],
        action_window=action_window,
        seed=seed,
        task_filter=task_filter,
    )

    collator = HDF5Collator(
        model_max_length=model_max_length,
        pad_token_id=pad_token_id,
        pixel_values_dtype=torch.float32,
    )

    dataset_statistics = {
        "libero_mem": {
            "action": {
                "q01": stats["q01"].tolist(),
                "q99": stats["q99"].tolist(),
                "mask": stats["mask"].tolist(),
            },
            "num_trajectories": len(train_episodes),
        }
    }
    return dataset, dataset_statistics, collator
