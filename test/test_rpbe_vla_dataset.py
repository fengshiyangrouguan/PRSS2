"""Task 2 acceptance tests: HDF5DecisionStreamDataset semantics.

Verified on synthetic h5 files (no LIBERO-Mem download needed):
  1. timesteps are DECISION indices 0,1,2,... (predict_action alignment)
  2. supervision chunk = actions[d*K : (d+1)*K]
  3. tail partial decision dropped (with counter)
  4. episode_ids per row; train/val split convention (demo_1..80 / 81..100)
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "memoryvla"))

from vla.datasets.hdf5_dataset import (  # noqa: E402
    TRAIN_DEMO_RANGE, VAL_DEMO_RANGE,
    HDF5DecisionStreamDataset, HDF5BatchTransform,
    scan_episodes, compute_action_stats,
)

K = 16  # decision stride = future_action_window_size + 1


class _FakeImageTransform:
    def __call__(self, img):
        return {"dino": torch.zeros(1, 3, 224, 224), "siglip": torch.zeros(1, 3, 224, 224)}


class _FakePromptBuilder:
    def __call__(self, _):  # returns object with add_turn + get_prompt
        return _FakeBuilder()


class _FakeBuilder:
    def add_turn(self, *args, **kw):
        pass

    def get_prompt(self):
        return "<prompt>"


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=True):
        return type("R", (), {"input_ids": [1, 2, 3, 4, 5, 6, 7]})()


def _make_synthetic_data(root: Path, task="KITCHEN_SCENE1_3_x", n_demos=100,
                         T_by_demo=None, seed=0) -> Path:
    rng = np.random.default_rng(seed)
    h5_path = root / f"{task}_demo.hdf5"
    with h5py.File(h5_path, "w") as f:
        g = f.create_group("data")
        for i in range(1, n_demos + 1):
            T = (T_by_demo(i) if T_by_demo else 40 + (i % 3) * 17)
            d = g.create_group(f"demo_{i}")
            d.create_dataset("actions", data=rng.normal(size=(T, 7)))
            d.create_dataset("dones", data=np.zeros(T, dtype=np.uint8))
            obs = d.create_group("obs")
            obs.create_dataset("agentview_rgb", data=np.zeros((T, 256, 256, 3), dtype=np.uint8))
            # extra keys the loader must ignore
            obs.create_dataset("eye_in_hand_rgb", data=np.zeros((T, 256, 256, 3), dtype=np.uint8))
    meta = {}
    for i in range(1, n_demos + 1):
        meta.setdefault(task, {})[f"demo_{i}"] = {
            "success": True, "task_description": "lift the bowl 3 times",
            "task_nouns": ["bowl"], "initial_state": [], "exo_boxes": [], "ego_boxes": [],
        }
    with open(root / "metainfo.json", "w") as f:
        json.dump(meta, f)
    return h5_path


class TestDecisionStreamSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.root = Path(cls.tmp)
        _make_synthetic_data(cls.root)

    def test_scan_splits(self):
        train, _ = scan_episodes(self.root, "train")
        val, _ = scan_episodes(self.root, "val")
        self.assertEqual(len(train), TRAIN_DEMO_RANGE[1] - TRAIN_DEMO_RANGE[0] + 1)  # 80
        self.assertEqual(len(val), VAL_DEMO_RANGE[1] - VAL_DEMO_RANGE[0] + 1)        # 20

    def test_scan_task_filter(self):
        train, _ = scan_episodes(self.root, "train", task_filter="KITCHEN_SCENE1_3_x")
        self.assertEqual(len(train), 80)
        other, _ = scan_episodes(self.root, "train", task_filter="NO_SUCH_TASK")
        self.assertEqual(len(other), 0)

    def test_decision_stream_rows(self):
        transform = HDF5BatchTransform(
            base_tokenizer=_FakeTokenizer(),
            image_transform=_FakeImageTransform(),
            prompt_builder_fn=_FakePromptBuilder(),
            action_q01=np.zeros(7, dtype=np.float32),
            action_q99=np.ones(7, dtype=np.float32),
            action_mask=np.ones(7, dtype=bool),
            decision_stride=K,
        )
        ds = HDF5DecisionStreamDataset(
            self.root, "train", transform, decision_stride=K,
            seed=0, repeat=False, episode_shuffle=False)
        # collect first episode rows (demo_1 has T = 40 + 1*17 = 57 -> 3 decisions)
        rows = []
        ep_ids, timesteps = set(), []
        for row in ds:
            if row["episode_ids"][0] != 0:
                break
            rows.append(row)
            ep_ids.add(int(row["episode_ids"][0]))
            timesteps.append(int(row["timesteps"][0]))
        self.assertEqual(timesteps, [0, 1, 2])          # decision indices
        self.assertEqual(ep_ids, {0})                    # one episode
        for i, row in enumerate(rows):
            self.assertEqual(tuple(row["actions"].shape), (K, 7))

    def test_tail_drop_counter(self):
        # T=57 -> 3 full decisions (48 frames) + 9 tail frames dropped
        transform = HDF5BatchTransform(
            base_tokenizer=_FakeTokenizer(), image_transform=_FakeImageTransform(),
            prompt_builder_fn=_FakePromptBuilder(),
            action_q01=np.zeros(7, dtype=np.float32),
            action_q99=np.ones(7, dtype=np.float32),
            action_mask=np.ones(7, dtype=bool), decision_stride=K)
        ds = HDF5DecisionStreamDataset(
            self.root, "train", transform, decision_stride=K,
            seed=0, repeat=False, episode_shuffle=False)
        n_rows = 0
        for _ in ds:
            n_rows += 1
        # total decisions across 80 demos; every demo has some tail (T%16 != 0)
        self.assertGreater(ds.n_dropped_tail, 0)
        self.assertEqual(n_rows, sum(
            (40 + (i % 3) * 17) // K for i in range(1, 81)))

    def test_action_stats(self):
        train, _ = scan_episodes(self.root, "train")
        stats = compute_action_stats(train, K)
        self.assertEqual(stats["q01"].shape, (7,))
        self.assertTrue((stats["mask"] == (stats["q01"] != stats["q99"])).all())


if __name__ == "__main__":
    unittest.main()
