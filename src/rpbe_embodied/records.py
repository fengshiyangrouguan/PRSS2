"""rpbe_embodied.records — MergeRecord / PendingMergeQueue / EmbodiedCutRow.

Replaces the TGN tree-query machinery (CompactCutTrace / JodieFutureIndex /
JodieCutBuilder): the merge tree GROWS NATURALLY out of the chronological
consolidation stream.  One real consolidation = one cut.

Supervision protocol (2026-09-05 review ruling 1, causal cut):
  merge at post-decision memory-update time tau_v = d;
  Y1 = A_{d+1}, Y2 = A_{d+2} are the ONLY admissible futures.
  Valid RPBE cut <=> BOTH futures observed.  Cuts with tau+2 >= T_episode
  are CENSORED: they never enter the RPBE window (they still participate
  in task loss normally).  No "partial-horizon" rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

HORIZON_OMEGA = (0.5, 0.5)


@dataclass
class MergeRecord:
    """One real consolidation event in CogMemBank."""
    episode_id: int
    merge_id: int                    # global monotonic counter (cut_id key)
    merge_decision_time: int         # tau_v (post-decision update index)
    left_state: torch.Tensor         # m_a.detach()  [4096]
    right_state: torch.Tensor        # m_b.detach()  [4096]
    merged_state: torch.Tensor       # z_v.detach()  [4096]
    left_id: int
    right_id: int
    node_id: int
    depth: int
    start_step: int
    end_step: int
    param_version: int               # LoRA+Gamma version at write time
    weight_meta: dict = field(default_factory=dict)

    def cut_id(self) -> Tuple[int, int, str]:
        return (self.episode_id, self.merge_id, "cog")

    def row_id(self, h: int) -> Tuple[int, int, str, int]:
        return self.cut_id() + (h,)


@dataclass
class PendingMerge:
    rec: MergeRecord
    y1: Optional[torch.Tensor] = None   # A_{tau+1} [112] normalized
    y2: Optional[torch.Tensor] = None   # A_{tau+2} [112]
    c1: Optional[dict] = None           # future context dict at tau+1
    c2: Optional[dict] = None


@dataclass
class EmbodiedCutRow:
    """Thin row: the only thing that enters the RPBE window (plan §21)."""
    cut_id: tuple
    horizon: int
    z: torch.Tensor                # [4096] detached
    context: dict                  # {horizon, delta_s, instruction_hash, vision_feat}
    outcome: torch.Tensor          # [112] normalized action chunk
    weight: float


class PendingMergeQueue:
    """Two-level future maturation, strict future_decision > merge_decision.

    No tree traversal, no memory querying (plan §6 / §31.3).
    """

    def __init__(self, horizon_weights: tuple = HORIZON_OMEGA):
        self.pending: Dict[Tuple[int, int], PendingMerge] = {}
        self.omega_sum = sum(horizon_weights)
        self.horizon_weights = horizon_weights
        self.n_censored = 0          # merges whose Y2 never matured
        self.n_missing_y1 = 0

    def register(self, rec: MergeRecord) -> None:
        key = (rec.episode_id, rec.merge_id)
        assert key not in self.pending, f"duplicate merge {key}"
        self.pending[key] = PendingMerge(rec=rec)

    def offer(self, episode_id: int, decision_idx: int,
              context: dict, outcome: torch.Tensor) -> List[EmbodiedCutRow]:
        """Feed one post-decision future sample; records futures only.

        Maturation happens at drain_episode, where the tree_weight
        (1/n_merges) is known.  Strict: decision_idx > tau_v is asserted
        for every merge in this episode (causal cut protocol).
        """
        for key, pm in list(self.pending.items()):
            if pm.rec.episode_id != episode_id:
                continue
            tau = pm.rec.merge_decision_time
            # Only STRICTLY-later decisions are futures.  Rows earlier than
            # tau are legal co-batch rows (the data stream is monotone, so
            # they cannot leak future information) and are skipped.  Rows
            # equal to tau are the merge's own batch and skipped too.
            if decision_idx <= tau:
                continue
            if decision_idx == tau + 1 and pm.y1 is None:
                pm.y1 = outcome
                pm.c1 = context
            elif decision_idx == tau + 2 and pm.y2 is None:
                pm.y2 = outcome
                pm.c2 = context
        return []

    def drain_episode(self, episode_id: int, n_merges: int) -> List[EmbodiedCutRow]:
        """Episode end: only merges with BOTH futures become rows.

        tree_weight = 1/n_merges (per-episode merge count), split by
        HORIZON_OMEGA.  Merges with any missing future are censored
        (counted, dropped)."""
        rows: List[EmbodiedCutRow] = []
        for key, pm in list(self.pending.items()):
            if pm.rec.episode_id != episode_id:
                continue
            del self.pending[key]
            if pm.y1 is None:
                self.n_missing_y1 += 1
                self.n_censored += 1
                continue
            if pm.y2 is None:
                self.n_censored += 1
                continue
            rows.extend(self._mature(pm, tree_weight=1.0 / max(n_merges, 1)))
        return rows

    def _mature(self, pm: PendingMerge,
                tree_weight: Optional[float] = None) -> List[EmbodiedCutRow]:
        """Both futures present: one row per horizon sharing the cut (2Obs)."""
        rec = pm.rec
        if tree_weight is None:
            tree_weight = rec.weight_meta.get("tree_weight", 1.0)
        base = tree_weight / self.omega_sum
        rows = []
        for h, (y, c) in enumerate(((pm.y1, pm.c1), (pm.y2, pm.c2)), start=1):
            rows.append(EmbodiedCutRow(
                cut_id=rec.cut_id(),
                horizon=h,
                z=rec.merged_state,
                context=c or {"horizon": h, "delta_s": h},
                outcome=y,
                weight=base * self.horizon_weights[h - 1],
            ))
        return rows

    def reset(self) -> None:
        self.pending.clear()
        self.n_censored = 0
        self.n_missing_y1 = 0
