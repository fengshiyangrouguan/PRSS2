"""LaMP-2 cut records: one sample -> one RPBE row (1Obs, L2).

The official LaMP training sample is a single legal cut: the SUM memory
state (4 slots, one-shot merge of the 16 profile COMP blocks) is
supervised by the strictly future answer:

    row: (z_v, p_v, w=1.0, cut_id)   p = Sketch([1; chi] (x) phi)

where chi = chi(query) (the condition of the answer, a frozen
UtteranceEmbed sketch) and phi = phi(answer) (the future answer text,
frozen sketch).  The answer logprob (from the training forward's
logits) is recorded in ``outcome`` for the statistical report; it does
not enter the covariance (dialog-line convention: the P side carries
the future content, the outcome is a record field).

``tree_id`` = the LaMP user id (one user = one independent history);
rows from the SAME user's re-samplings share the tree gate.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

from rpbe.records import CutRecord

MEM_TAU = "mem"
HORIZON_WEIGHTS = (1.0,)  # one horizon: the answer


@dataclass
class LampMeta:
    """Deterministic per-sample metadata for the LaMP cut.

    ``sum_positions``: padded positions of the 4 SUM tokens.
    ``query_span``: (start, stop) token slice of the query part.
    ``user_id``: stable LaMP user id (tree identity).
    ``answer_positions``: label (non -100) token positions.
    """

    sample_id: int
    sum_positions: List[int]     # 4 SUM token positions (padded)
    query_span: tuple            # (start, stop) of the query text
    user_id: int                 # stable user id (tree_id)
    answer_positions: List[int]  # label token positions (non -100)


class LampCutBuilder:
    """One LaMP sample -> one CutRecord row (1Obs)."""

    def __init__(self, maps, *, seed: int = 0, z_dim: int = 128):
        self.maps = maps
        self.seed = int(seed)
        self.z_dim = int(z_dim)
        self._next_oid = 0

    @property
    def next_oid(self) -> int:
        return self._next_oid

    @next_oid.setter
    def next_oid(self, value: int) -> None:
        self._next_oid = int(value)

    def build(self, meta: LampMeta, z_v: torch.Tensor, chi: torch.Tensor,
              phi: torch.Tensor, outcome: float = 1.0) -> List[CutRecord]:
        """One cut -> one row; z_v keeps its graph (pass 2 replays the
        exact gradient), chi/phi are frozen constants."""
        cut_occurrence = self._next_oid
        self._next_oid += 1
        if chi.dim() == 2:
            chi = chi[0]
        if phi.dim() == 2:
            phi = phi[0]
        # No depth buckets on LaMP (no depth stratification; every
        # official sample is a full 16-profile compression).
        p = self.maps.pv(chi, phi, L=None)
        return [CutRecord(
            tree_id=int(meta.user_id),
            occurrence_id=cut_occurrence,
            tau=MEM_TAU,
            horizon=1,
            node=int(meta.user_id),
            time=16.0,  # fixed profile count (official k=16)
            z=z_v[0] if z_v.dim() == 2 else z_v,
            context={"horizon": 1, "n_profile": 16},
            outcome=float(outcome),
            outcome_id=(int(meta.sample_id), cut_occurrence, 1),
            weight=HORIZON_WEIGHTS[0],
            p_override=p,
        )]
