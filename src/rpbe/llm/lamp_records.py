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

and the RPBE rows realise the ALIGNED 2Obs CLOSURE frozen in the method
section: cut j is supervised by the adjacent pair

    cut j   : (O_{j+1}, O_{j+2})
    cut j+1 : (O_{j+2}, O_{j+3})

i.e. the second observation of one interface IS the first observation of
the next one.  Two rows per cut (horizon 1 and 2), exactly mirroring
``DialogueCutBuilder``:

    row 1: (z_j, p_{j,1})   chi = O_{j+1},                     phi = O_{j+2}
    row 2: (z_j, p_{j,2})   chi = combine(O_{j+1}, O_{j+2}),   phi = O_{j+3}

The final query therefore enters ONLY as the local observation of the
last cuts (O_{N+1} = Q), never as a target copied onto every cut: the
query answer is not repeated across the chain.

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
              phi1: List[torch.Tensor], phi2: List[torch.Tensor],
              ) -> List[CutRecord]:
        """One sample -> up to 2*(N-1) rows.

        ``z_list`` is [N, z_dim]: row j is the memory state after j
        profile blocks (z_list[j-1] == M_j).  The per-cut sketch lists
        are indexed by cut position j = 1..N-1 and hold the observations
        O_{j+1}, O_{j+2}, O_{j+3} as described in the module docstring.
        All z rows keep their graph; the sketches are frozen constants.

        Returns [] when the chain is too short to carry a 2-step closure
        (N < 3), mirroring the dialogue line's ``k < MIN_K`` guard: the
        task CE keeps its own gradient and the sample simply carries no
        RPBE row.
        """
        n = int(meta.n_profile)
        if n < 3:
            return []
        n_cuts = n - 1                      # j = 1 .. N-1
        rows: List[CutRecord] = []
        for j in range(1, n_cuts + 1):
            z = z_list[j - 1]
            cut_occurrence = self._next_oid
            self._next_oid += 1
            # Tail handling: cut j's second horizon needs O_{j+3}; the
            # last cuts that would reach past O_{N+1} keep only their
            # horizon-1 row and take the full weight (the dialogue line's
            # R10 endpoint principle).
            horizons = (1, 2) if (j + 2) <= n else (1,)
            for horizon in horizons:
                if horizon == 1:
                    chi, phi = chi1[j], phi1[j]
                else:
                    chi, phi = chi2[j], phi2[j]
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
