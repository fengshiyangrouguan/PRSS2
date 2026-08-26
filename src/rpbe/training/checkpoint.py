"""Rolling mid-epoch checkpoints with exact resume: RNG, model, optimizer, loop state.

The JODIE line carries the host memory backup through ``extra_payload``
(``tgn.memory.backup_memory()``), so an interrupted run resumes with the exact
memory state.  There is no second optimizer and no PyG message store anymore
(the TGB line was removed from this branch).
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
    """Normalize an RNG state for PyTorch generator APIs (CPU uint8 contiguous)."""
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


class CheckpointManager:
    """Atomic rolling checkpoints with exact resume support."""

    def __init__(self, path: str):
        self.path = Path(path)

    def save(self, *, model_components: Dict[str, torch.nn.Module],
             optimizer, epoch: int, next_batch: int, global_step: int,
             best_score: float, best_epoch: int, bad_rounds: int,
             train_state: Dict, extra_payload: Optional[Dict] = None) -> None:
        """``extra_payload`` is stored verbatim under payload["extra"]."""
        payload = {
            "model": {k: m.state_dict() for k, m in model_components.items()},
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "next_batch": int(next_batch),
            "global_step": int(global_step),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "bad_rounds": int(bad_rounds),
            "train_state": train_state,
            "rng": _rng_state(),
            "extra": extra_payload,
        }
        tmp = Path(str(self.path) + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, self.path)

    def load(self, *, model_components: Dict[str, torch.nn.Module],
             optimizer, device) -> Dict:
        payload = torch.load(self.path, map_location=device, weights_only=False)
        for k, m in model_components.items():
            m.load_state_dict(payload["model"][k])
        optimizer.load_state_dict(payload["optimizer"])
        _restore_rng(payload.get("rng"))
        return payload

    def exists(self) -> bool:
        return self.path.exists()
