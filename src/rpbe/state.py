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
class ConsumedPairCandidate:
    """One real child-parent consumption pair recorded at the parent
    aggregation site (paper recursive-closure supervision).

    ``child`` is a NEIGHBOR state actually consumed by ``parent``'s
    aggregation::

        child  = neighbor_lower[parent_row, slot]   (layer = parent_layer - 1)
        parent = source state at ``parent_layer`` that aggregated it

    Both child and parent query times are the PARENT's query time ``t`` — the
    time the host actually recursed them at (repeated_times).  We NEVER change
    a child's query time to its historical edge_time: that would make ``z``
    include information between edge_time and ``t`` (leakage).  The historical
    relation time is kept separately as ``relation_time`` (<= query time).

    Only INTERNAL compressed child layers (0 < child_layer < L) are recorded,
    and only the actual neighbor_lower tensor used by aggregate() is kept as
    ``z`` (full gradient connection).
    """

    pair_id: tuple               # globally-unique, exact-replay-stable
    root_row: int
    tau: str                     # child compressible interface ("tjo:layer<c>")
    child_layer: int
    parent_layer: int
    child_node: int
    child_time: float            # = parent query time (repeated_times)
    parent_node: int
    parent_time: float
    relation_time: float         # historical edge time, <= query time
    relation_edge_id: int
    relation_lag: float          # parent_query_time - relation_time
    relation_slot: int
    path: Tuple[int, float]      # tuple of (relation_code, delta_t) steps
    z: torch.Tensor              # neighbor_lower[row, slot] (gradient-connected)
    # Pre-Gamma vanilla aggregate of the SAME child occurrence (the tensor the
    # child returned before Gamma compression), for the Reconstruction aux
    # (P1): D_rec([z, chi(C)]) -> sg(u).  None when not traced.
    u: Optional[torch.Tensor] = None


@dataclass
class CompactCutTrace:
    """Only selected query roots and their bounded cut candidates."""

    root_rows: List[int] = field(default_factory=list)
    cuts: List[CutCandidate] = field(default_factory=list)
    pairs: List[ConsumedPairCandidate] = field(default_factory=list)

    def add(self, candidate: CutCandidate) -> None:
        self.cuts.append(candidate)

    def add_pair(self, candidate: ConsumedPairCandidate) -> None:
        self.pairs.append(candidate)
