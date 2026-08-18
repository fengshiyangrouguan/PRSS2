"""Rolling mid-epoch checkpoints with exact resume: RNG, memory, optimizer, loop state.

The PyG TGNMemory keeps its pending raw-message stores (``msg_s_store`` /
``msg_d_store``) in plain dicts that ``state_dict`` does not serialize; they are
saved/restored explicitly here.
"""

import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def _rng_state() -> Dict:
    out = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def _cpu_byte_rng_state(x):
    """Normalize an RNG state for PyTorch generator APIs (CPU uint8 contiguous).

    Rolling checkpoints are loaded with ``map_location=device``, which remaps the
    saved CPU RNG ByteTensor to CUDA along with model tensors; ``torch.set_rng_state``
    requires a CPU ByteTensor instead.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.uint8)
    return x.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _restore_rng(state: Optional[Dict]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte_rng_state(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([_cpu_byte_rng_state(x) for x in state["cuda"]])


def _msg_store_to_cpu(store: Dict) -> list:
    return [[(m.detach().cpu(), t.detach().cpu()) for m, t in entries]
            for entries in store.values()]


def _msg_store_from_cpu(store: Dict, payload: list, device) -> None:
    for node_key, entries in zip(store.keys(), payload):
        store[node_key] = [(m.to(device), t.to(device)) for m, t in entries]


class CheckpointManager:
    """Atomic rolling checkpoints with exact resume support."""

    def __init__(self, path: str):
        self.path = Path(path)

    def save(self, *, model_components: Dict[str, torch.nn.Module],
             optimizer, unrestricted_optimizer, epoch: int, next_batch: int,
             global_step: int, best_score: float, best_epoch: int, bad_rounds: int,
             train_state: Dict, memory_msg_stores: Optional[Dict] = None) -> None:
        payload = {
            "model": {k: m.state_dict() for k, m in model_components.items()},
            "optimizer": optimizer.state_dict(),
            "unrestricted_optimizer":
                unrestricted_optimizer.state_dict() if unrestricted_optimizer is not None else None,
            "epoch": int(epoch),
            "next_batch": int(next_batch),
            "global_step": int(global_step),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "bad_rounds": int(bad_rounds),
            "train_state": train_state,
            "rng": _rng_state(),
            "memory_msg_stores": (_msg_store_to_cpu(memory_msg_stores)
                                  if memory_msg_stores is not None else None),
        }
        tmp = Path(str(self.path) + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, self.path)

    def load(self, *, model_components: Dict[str, torch.nn.Module],
             optimizer, unrestricted_optimizer, memory_msg_stores: Optional[Dict],
             device) -> Dict:
        payload = torch.load(self.path, map_location=device, weights_only=False)
        for k, m in model_components.items():
            m.load_state_dict(payload["model"][k])
        optimizer.load_state_dict(payload["optimizer"])
        if unrestricted_optimizer is not None and payload.get("unrestricted_optimizer") is not None:
            unrestricted_optimizer.load_state_dict(payload["unrestricted_optimizer"])
        if memory_msg_stores is not None and payload.get("memory_msg_stores") is not None:
            _msg_store_from_cpu(memory_msg_stores, payload["memory_msg_stores"], device)
        _restore_rng(payload.get("rng"))
        return payload

    def exists(self) -> bool:
        return self.path.exists()
