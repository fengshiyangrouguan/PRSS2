"""Small, training-only trace records emitted during the host query.

The adapter deliberately does *not* materialize the recursive TGN tree.  A
traced root keeps only the internal states on its query-node (SELF) spine:
one candidate per compressible interface.  That is all the future-outcome
builder consumes, and it bounds trace storage by
``traced_roots * (n_layers - 1)`` instead of the tree's branching factor.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class CutCandidate:
    """One graph-connected internal state eligible for future supervision.

    ``root_row`` identifies the top-level query that produced the state.
    ``path`` is the structural route from that query root to the cut.  For
    the compact JODIE trace it contains only SELF steps; the state itself
    still aggregates the host's full temporal-neighbor computation.
    """

    occurrence_id: int
    root_row: int
    tau: str
    node: int
    time: float
    z: torch.Tensor
    # Pre-compression rich state (the vanilla host aggregate).  Used only
    # by the profiled-reconstruction ablation (Table 2, row 2): J_rec
    # measures how much of U's variance Z can linearly reconstruct.
    # None for low-level synthetic tests that do not model U.
    u: Optional[torch.Tensor] = None
    path: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class CompactCutTrace:
    """Only selected query roots and their bounded cut candidates."""

    root_rows: List[int] = field(default_factory=list)
    cuts: List[CutCandidate] = field(default_factory=list)

    def add(self, candidate: CutCandidate) -> None:
        self.cuts.append(candidate)
