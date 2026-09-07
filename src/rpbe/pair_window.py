"""Same-window two-pass KF window over child-parent boundary records.

Each ``BoundaryRecord`` is ONE supervised row::

    z  = child compressed state (gradient connected)
    p  = BoundaryMaps joint pair test (fixed)     [one row per pair]

mirroring the exact-replay contract of the node-classification
``KFMomentWindow``: pass 1 (no grad) accumulates detached rows; ``close_replay``
contracts the whole-window Ky Fan score onto CUT-LEVEL z-adjoints; pass 2 then
indexes the replayed traced z and applies the linear surrogate.

Per-tree weight: every root tree's total pair weight is 1 (pairs of one tree
share tree_weight = 1/n_pairs_in_tree), so deep/high-degree trees do not
dominate.

Only ``full_balancing`` is supported here (the pair path does not add
diagonal/reconstruction variants).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from rpbe.loss import (WeightedWelford, _score_from_covs,
                       latent_z_adjoint)


def tree_equal_weights(records: List) -> Dict[int, float]:
    """Per-tree pair weight: each tree's total pair weight = 1."""
    counts: Dict[int, int] = {}
    for r in records:
        counts[int(r.root_row)] = counts.get(int(r.root_row), 0) + 1
    return {t: 1.0 / float(c) for t, c in counts.items()}


class PairKFWindow:
    """Accumulate BoundaryRecords and close them into cut-level adjoints."""

    def __init__(self, *, tau: str, eps: float = 1e-4,
                 min_unique_trees: int = 64, strict: bool = False):
        self.tau = str(tau)
        self.eps = float(eps)
        self.min_unique_trees = int(min_unique_trees)
        self.strict = bool(strict)
        self.reset()

    def reset(self):
        self._records: List = []
        self._tree_seen = set()
        self._pair_seen = set()
        self._closed_this_batch = False

    # ------------------------------------------------------------ accumulate
    def add(self, record) -> None:
        """Accumulate one BoundaryRecord (detached handling happens at close)."""
        self._records.append(record)
        self._tree_seen.add(int(record.root_row))
        self._pair_seen.add(record.boundary_key)

    def n_unique_trees(self) -> int:
        return len(self._tree_seen)

    def ready(self) -> bool:
        return len(self._tree_seen) >= self.min_unique_trees

    # --------------------------------------------------------------- scoring
    def _score(self, records: List, maps):
        """Stack (z, p, w) for records and return the Ky Fan score tensor."""
        if not records:
            return None
        zs = [r.z for r in records]
        z = torch.stack(zs)                      # [N, d]
        ps = [maps.pv_row(r) for r in records]   # one p per record
        p = torch.stack(ps)                      # [N, m]
        w = torch.tensor([float(r.weight) for r in records],
                         dtype=torch.float64, device=z.device)
        # weighted centering via direct stack (records already tree-weighted)
        zc = z.double() - (z.double() * w[:, None]).sum(0, keepdim=True) \
            / w.sum()
        pc = p.double() - (p.double() * w[:, None]).sum(0, keepdim=True) \
            / w.sum()
        sw = w.sqrt().reshape(-1, 1)
        mzz = (zc * sw).t() @ (zc * sw)
        mzp = (zc * sw).t() @ (pc * sw)
        mpp = (pc * sw).t() @ (pc * sw)
        W = float(w.sum())
        W2 = float((w * w).sum())
        D = W - W2 / W if W > 0 else 0.0
        if not (D > 0.0):
            return None
        j, diag = _score_from_covs(mzz / D, mzp / D, mpp / D, self.eps,
                                   "full_balancing")
        if diag["failed"] is not None:
            if self.strict:
                raise RuntimeError("pair window close failed: {}".format(diag))
            return None
        return j

    # --------------------------------------------------------------- replay
    def close_replay(self, maps):
        """Contract onto per-record z adjoints for the exact-replay pass 2.

        Returns ``(closed_score, g_by_position, diag)`` where
        ``g_by_position`` maps a record's position in ``_records`` to its
        merged z gradient tensor.
        """
        records = self._records
        if len(records) == 0:
            return None, {}, {"below_threshold": False, "M": 0}
        if not self.ready():
            return None, {}, {
                "below_threshold": True,
                "M_unique_trees": len(self._tree_seen),
                "threshold": self.min_unique_trees}
        z = torch.stack([r.z for r in records]).double()
        p = torch.stack([maps.pv_row(r) for r in records]).double()
        w = torch.tensor([float(r.weight) for r in records],
                         dtype=torch.float64, device=z.device)
        W = float(w.sum())
        W2 = float((w * w).sum())
        D = W - W2 / W if W > 0 else 0.0
        if not (D > 0.0):
            return None, {}, {"failed": "nonpositive_weight", "W": W}
        mu_z = (z * w[:, None]).sum(0, keepdim=True) / W
        mu_p = (p * w[:, None]).sum(0, keepdim=True) / W
        j, g_by_cut, score_diag = latent_z_adjoint(
            z.detach().float(), p.float(), w,
            [r.boundary_key for r in records],
            mu_z.float(), mu_p.float(), D, self.eps, self.strict,
            variant="full_balancing")
        if score_diag["failed"] is not None:
            return None, {}, score_diag
        # latent_z_adjoint keys by cut_id; map boundary_key -> position
        pos_of = {r.boundary_key: i for i, r in enumerate(records)}
        g_by_position = {pos_of[k]: g for k, g in g_by_cut.items()}
        return float(j), g_by_position, {
            "M_unique_trees": len(self._tree_seen),
            "below_threshold": False}

    def surrogate(self, position_grads: Dict[int, torch.Tensor],
                  live_records: List, coeff: float) -> torch.Tensor:
        """Linear surrogate for pass 2: sum_v <sg(g_v), z_v(theta)>."""
        terms = []
        for pos, g in position_grads.items():
            if pos >= len(live_records):
                continue
            z = live_records[pos].z
            gd = g.detach()
            terms.append((gd * z).sum() - (gd * z.detach()).sum())
        if not terms:
            return torch.zeros((), device=self._device_of(live_records))
        return coeff * sum(terms)

    @staticmethod
    def _device_of(records):
        return records[0].z.device if records else torch.device("cpu")
