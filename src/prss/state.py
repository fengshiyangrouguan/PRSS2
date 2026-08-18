"""Typed inside/outside states and traces for recursive hosts."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch


@dataclass
class QuotientState:
    tau: str
    raw: torch.Tensor
    candidate: torch.Tensor
    quotient: torch.Tensor


@dataclass
class RecursiveOccurrence:
    occurrence_id: int
    tau: str
    state: QuotientState
    local_features: torch.Tensor
    children: List[int] = field(default_factory=list)
    child_relations: List[int] = field(default_factory=list)
    child_delta_t: List[float] = field(default_factory=list)
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
