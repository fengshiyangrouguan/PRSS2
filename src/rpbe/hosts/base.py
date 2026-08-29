"""Host adapter and cut-builder abstractions.

A host adapter keeps the host's original call surface and changes only the
internal aggregated states passed to another recursive node.  The cut builder
turns the adapter's bounded training-only trace into CutRecord rows.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence

import torch
from torch import nn

from rpbe.state import CompactCutTrace


class HostAdapter(nn.Module, ABC):
    """Wraps one host model's recursive aggregation surface.

    The host aggregate itself is never replaced.  Tracing is fused into the
    training query, bounded, and must never change the main forward.
    """

    @abstractmethod
    def set_trace_source_rows(self, rows: Sequence[int]) -> None:
        """Select which batch rows are traced as roots for the auxiliary pass."""

    @abstractmethod
    def clear_trace(self) -> None:
        """Disable tracing; the main forward must be bit-identical afterwards."""

    @property
    @abstractmethod
    def trace(self) -> Optional[CompactCutTrace]:
        """The bounded cut trace of the last forward (None if tracing off)."""


class CutBuilder(ABC):
    """Training-only: turns a trace plus batch metadata into CutRecord rows.

    The outcome y enters the fixed measurement only, never the compressor.
    """

    @abstractmethod
    def build(self, *batch_context) -> List[Any]:
        ...
