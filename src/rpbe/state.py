"""Typed states and computation-tree traces for recursive hosts."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch


@dataclass
class OccurrenceState:
    """The compressed state actually propagated to the parent: z only.

    ``tau`` is the interface type of this occurrence.  The old quotient-era
    raw/candidate fields are gone with the spectral architecture.
    """

    tau: str
    z: torch.Tensor


@dataclass
class RecursiveOccurrence:
    occurrence_id: int
    tau: str
    state: OccurrenceState
    children: List[int] = field(default_factory=list)
    child_relations: List[int] = field(default_factory=list)
    child_delta_t: List[float] = field(default_factory=list)
    # Contract keys: ``node`` (global node id) and ``time`` (as-of time =
    # the query timestamp of the tree, for every occurrence).  Filled by
    # the host adapter.
    #
    # ``local_features`` was removed: it cloned the full per-batch preagg
    # matrix (~GiB of wasted storage per epoch) while nothing consumed it.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecursiveTrace:
    occurrences: Dict[int, RecursiveOccurrence] = field(default_factory=dict)
    roots: List[int] = field(default_factory=list)
    root_rows: List[int] = field(default_factory=list)

    def add(self, occurrence):
        if occurrence.occurrence_id in self.occurrences:
            raise ValueError("Duplicate occurrence id")
        self.occurrences[occurrence.occurrence_id] = occurrence

    def postorder(self):
        visited = set()
        result = []

        def visit(identifier):
            if identifier in visited:
                return
            for child in self.occurrences[identifier].children:
                visit(child)
            visited.add(identifier)
            result.append(identifier)

        for root in self.roots:
            visit(root)
        return result
