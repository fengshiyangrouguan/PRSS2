"""
hdf5_dataset.py

LIBERO-Mem HDF5 DecisionStream dataset (RPBE-VLA). Implements the plan's
Task 2 semantics on top of the official LIBERO-Mem HDF5 format:

  * decision stride K = future_action_window_size + 1 = 16
  * timesteps are DECISION indices (0, 1, 2, ...) -- aligned with
    MemoryVLA.predict_action's cur_timestep semantics
  * supervision per decision d = actions[d*K : (d+1)*K] ([K, 7])
  * tail decisions whose chunk would run past the episode end are dropped
    (with a counter), mirroring truncation at inference
  * episode_ids filled per row so CogMemBank 'stream' semantics hold

Split convention (2026-09-05 review ruling A): official val was never
released; we use demo_1..demo_80 as train and demo_81..demo_100 as val,
a deterministic convention shared by all seeds.

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


class HDF5Collator(PaddedCollatorForActionPrediction):
    """Official action-prediction collator + pass-through of the per-row
    `instruction` string (needed for the RPBE fixed context map)."""

    def __call__(self, instances):
        out = super().__call__(instances)
        out["instruction"] = [inst["instruction"] for inst in instances]
        return out

# official LIBERO-Mem convention: 100 released demos per task; split 80/20
TRAIN_DEMO_RANGE = (1, 80)   # inclusive
VAL_DEMO_RANGE = (81, 100)   # inclusive

# BOUNDS_Q99 epsilon used by the official RLDS pipeline
NORM_EPS = 1e-8


@dataclass
class HDF5BatchTransform:
    """Row transform replicating RLDSBatchTransform prompt/label logic,
    with the action BOUNDS_Q99 normalization done in numpy (official
    recipe from vla/datasets/rlds/utils/data_utils.py)."""
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Any
    action_q01: np.ndarray      # [7]
    action_q99: np.ndarray      # [7]
    action_mask: np.ndarray     # [7] bool: True where q01 != q99
    predict_stop_token: bool = True
    decision_stride: int = 16

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

        # official BOUNDS_Q99 normalization (numpy replica, WITH the
        # [-1, 1] clip -- review ruling: the clip changes the training
        # target and the downstream RFF inputs)
        a = row["actions"].astype(np.float32)                 # [K, 7]
        a = np.where(
            self.action_mask,
            2.0 * (a - self.action_q01) / (self.action_q99 - self.action_q01 + NORM_EPS) - 1.0,
            np.zeros_like(a),
        )
        a = np.clip(a, -1.0, 1.0)
        actions = torch.tensor(a, dtype=torch.float32)

        # Mask prompt tokens before the first <EOS-ish> token id 2
        eos_positions = torch.where(input_ids == 2)[0]
        if len(eos_positions) > 0:
            labels[: int(eos_positions[0])] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        action_masks = torch.ones(self.decision_stride, dtype=torch.bool)

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=row["task_name"],
            instruction=row["instruction"],
            actions=actions,
            action_masks=action_masks,
            timesteps=np.array([row["decision_idx"]], dtype=np.int64),
            episode_ids=np.array([row["episode_idx"]], dtype=np.int64),
        )


def scan_episodes(
    data_root: Path, split: str, task_filter: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Scan h5 files + metainfo.json -> episode manifest.

    Returns (episodes, demo_range_by_task).  Each episode:
      {task_name, h5_path, demo_key, instruction, T, frame_range}
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
            # official release is missing a few demos per task (e.g. T3 lacks
            # demo_48 and demo_81 in BOTH hdf5 and metainfo) -- skip, do not raise
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


def compute_action_stats(
    episodes: List[Dict[str, Any]], decision_stride: int,
) -> Dict[str, np.ndarray]:
    """BOUNDS_Q99 statistics over full-action windows of the given episodes
    (frame-level actions, matching the official statistics convention)."""
    all_actions: List[np.ndarray] = []
    for ep in episodes:
        with h5py.File(ep["h5_path"], "r") as f:
            a = f["data"][ep["demo_key"]]["actions"][:].astype(np.float32)
        n_win = a.shape[0] // decision_stride
        if n_win == 0:
            continue
        all_actions.append(a[: n_win * decision_stride].reshape(-1, decision_stride, 7))
    cat = np.concatenate(all_actions, axis=0)            # [N, K, 7]
    cat = cat.reshape(-1, 7)                              # frame level
    q01 = np.percentile(cat, 1, axis=0).astype(np.float32)
    q99 = np.percentile(cat, 99, axis=0).astype(np.float32)
    mask = (q01 != q99)
    return {"q01": q01, "q99": q99, "mask": mask}


class HDF5DecisionStreamDataset(IterableDataset):
    """One decision per row; episodes streamed in shuffled order (train)."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        batch_transform: HDF5BatchTransform,
        decision_stride: int = 16,
        seed: int = 0,
        repeat: bool = True,
        episode_shuffle: bool = True,
        task_filter: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "val")
        self.data_root = Path(data_root)
        self.split = split
        self.batch_transform = batch_transform
        self.decision_stride = decision_stride
        self.seed = seed
        self.repeat = repeat
        self.episode_shuffle = episode_shuffle

        self.episodes, _ = scan_episodes(self.data_root, split,
                                         task_filter=task_filter)
        self.n_dropped_tail = 0

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        order = list(range(len(self.episodes)))
        if self.episode_shuffle and self.split == "train":
            rng = np.random.default_rng(self.seed + (worker_info.id if worker_info else 0))
            rng.shuffle(order)

        while True:
            for idx in order:
                ep = self.episodes[idx]
                with h5py.File(ep["h5_path"], "r") as f:
                    actions = f["data"][ep["demo_key"]]["actions"]
                    rgb = f["data"][ep["demo_key"]]["obs"]["agentview_rgb"]
                    T = actions.shape[0]
                    n_decisions = T // self.decision_stride
                    for d in range(n_decisions):
                        k = d * self.decision_stride
                        row = dict(
                            agentview_rgb=np.asarray(rgb[k]),
                            actions=np.asarray(actions[k: k + self.decision_stride]),
                            instruction=ep["instruction"],
                            task_name=ep["task_name"],
                            decision_idx=d,
                            episode_idx=idx,
                        )
                        yield self.batch_transform(row)
                # tail frames beyond the last full decision are dropped,
                # mirroring inference truncation (count them once per pass)
                tail = T - n_decisions * self.decision_stride
                if tail > 0:
                    self.n_dropped_tail += 1
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
    share the same normalization, as in the official pipeline).
    """
    decision_stride = future_action_window_size + 1
    train_episodes, _ = scan_episodes(Path(data_root), "train",
                                      task_filter=task_filter)
    stats = compute_action_stats(train_episodes, decision_stride)

    transform = HDF5BatchTransform(
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=prompt_builder_fn,
        action_q01=stats["q01"],
        action_q99=stats["q99"],
        action_mask=stats["mask"],
        decision_stride=decision_stride,
    )

    dataset = HDF5DecisionStreamDataset(
        data_root=Path(data_root),
        split=split,
        batch_transform=transform,
        decision_stride=decision_stride,
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
