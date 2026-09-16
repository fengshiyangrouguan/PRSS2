"""LaMP-2 cut records: chain-wise RPBE under the unified recursive protocol.

Chain definition (frozen 2026-09-17): the memory chain is the profile
blocks in the order CCM actually consumes them,

    P1 -> P2 -> ... -> P_N -> Q          (Q = the query)

with

    M_1 = Compress(P_1)
    M_j = Compress(M_{j-1}, P_j)         j = 2..N

`merge_recur` produces exactly these states: the j-th SUM row is the
running aggregate over everything before it, so M_1..M_N are all
recoverable from ONE forward.

Observations are the profile-side events plus the query:

    O_1..O_N   = the N profile blocks
    O_{N+1}    = the query
    A          = the query ANSWER (the task target)

and the RPBE rows mirror the dialogue line's R10 freeze EXACTLY — every
cut carries one LOCAL future and one FINAL TASK future:

    cut j (1 <= j < N-1):
        row 1 (local):  C = O_{j+1},                      Y = O_{j+2}
        row 2 (task):   C = (O_{j+1}, O_{j+2}) combined,  Y = A

    cut j = N-1 (deepest):  row 2 only, weight 1.0
        C = (O_N, O_{N+1}) combined, Y = A

The deepest cut's local row is dropped because its next observation is
the query, which the decoder already sees — the same endpoint rule the
dialogue line applies to its decoder-visible context turn.  A is shared
by every cut, exactly as the dialogue line shares its final u_k; it is
the TASK target, not a sliding profile future.

The final query therefore enters as the local observation of the last
cuts, and the answer enters as the shared task future — never copied as
a per-position target.

Row count for the official N = 16: 2*(N-2) + 1 = 29.

``L`` is the ACTUAL recursion position j (not None), which is what makes
``J_real_minus_shuffled_within_L`` stratify by recursion depth instead of
collapsing to one bucket.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from rpbe.records import CutRecord
from rpbe.llm.dialogue_records import Llmmaps as _DialogueLlmmaps

MEM_TAU = "mem"

# Aligned 2Obs closure: two horizon rows per cut, equal weight.
HORIZON_WEIGHTS = (0.5, 0.5)


class LampLlmmaps(_DialogueLlmmaps):
    """Llmmaps with a LaMP-specific depth ladder.

    The base ``_bucket`` does ``L % len(DEPTH_LEVELS)``, so reusing the
    dialogue ladder (1,2,4,8,13) would alias every j >= 14 back onto an
    early bucket (e.g. 15 % 5 == 0 -> level 1).  LaMP recursion positions
    run j = 1..16, so we give each position its own slot: 16 entries,
    index j maps to itself.  Slot 0 is never produced by a real cut.
    """

    DEPTH_LEVELS = tuple(range(16))


@dataclass
class LampMeta:
    """Deterministic per-sample metadata for a LaMP chain.

    ``n_profile``: number of profile blocks actually present in the
    forward (the official collator pops trailing profiles when the input
    exceeds max_length, so this can be < 16).
    ``user_id``: stable LaMP user id (RPBE tree identity).
    ``sample_id``: dataset row index (audit only).
    """

    sample_id: int
    user_id: int
    n_profile: int


class LampCutBuilder:
    """One LaMP sample -> every legal recursive cut (2 rows each)."""

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

    def build(self, meta: LampMeta, z_list: torch.Tensor,
              chi1: List[torch.Tensor], chi2: List[torch.Tensor],
              phi1: List[torch.Tensor], phi2: torch.Tensor,
              ) -> List[CutRecord]:
        """One sample -> 2*(N-2) + 1 rows (29 for the official N = 16).

        Every cut carries one LOCAL future (horizon 1) and one FINAL TASK
        future (horizon 2, the shared answer sketch ``phi2``) — the same
        "one local + one task" structure the dialogue line's R10 freeze
        uses.  ``z_list`` is [N-1, z_dim] with row j-1 == M_j (the memory
        after j profile blocks); the sketches are frozen constants.

        The deepest cut j = N-1 has NO legal local future: its next
        observation is the query, which the decoder already sees, so the
        local row is dropped and the surviving task row takes the full
        cut weight (weight 1.0) — the dialogue line's endpoint rule.

        Returns [] for N < 2 (no cut exists).
        """
        n = int(meta.n_profile)
        if n < 2:
            return []
        n_cuts = n - 1                      # j = 1 .. N-1
        rows: List[CutRecord] = []
        for j in range(1, n_cuts + 1):
            z = z_list[j - 1]
            cut_occurrence = self._next_oid
            self._next_oid += 1
            horizons = (1, 2) if j < n_cuts else (2,)
            for horizon in horizons:
                if horizon == 1:
                    chi, phi = chi1[j], phi1[j]
                else:
                    chi, phi = chi2[j], phi2
                if chi.dim() == 2:
                    chi = chi[0]
                if phi.dim() == 2:
                    phi = phi[0]
                # L = the ACTUAL recursion position (1..N-1).
                p = self.maps.pv(chi, phi, L=int(j))
                rows.append(CutRecord(
                    tree_id=int(meta.user_id),
                    occurrence_id=cut_occurrence,
                    tau=MEM_TAU,
                    horizon=horizon,
                    node=int(meta.user_id),
                    time=float(j),
                    z=z if z.dim() == 1 else z[0],
                    context={"horizon": horizon, "cut_pos": j,
                             "n_profile": n, "k": j},
                    outcome=1.0,   # p is the content sketch, no label
                    outcome_id=(int(meta.sample_id), cut_occurrence, horizon),
                    weight=(1.0 if len(horizons) == 1
                            else HORIZON_WEIGHTS[horizon - 1]),
                    p_override=p,
                ))
        return rows
