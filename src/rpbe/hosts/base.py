"""Host adapter and cut-builder abstractions.

A host adapter keeps the host's original call surface and only changes what the
parent sees after each recursive aggregation: the compressed state
``z_v = Gamma(o_v, {z_children}, xi)``.  The cut builder turns the adapter's
training-only trace plus batch metadata into CutRecord rows for the Ky Fan loss.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence

import torch
from torch import nn

from rpbe.state import RecursiveTrace


class HostAdapter(nn.Module, ABC):
    """Wraps one host model's recursive aggregation surface.

    The host aggregate itself is never replaced: the adapter computes the vanilla
    aggregate and passes it (with the children states) through the recursive
    compressor.  Tracing is training-only and must never change the main forward.
    """

    @abstractmethod
    def set_trace_source_rows(self, rows: Sequence[int]) -> None:
        """Select which batch rows are traced as roots for the auxiliary pass."""

    @abstractmethod
    def clear_trace(self) -> None:
        """Disable tracing; the main forward must be bit-identical afterwards."""

    @property
    @abstractmethod
    def trace(self) -> Optional[RecursiveTrace]:
        """The computation-tree trace of the last forward (None if tracing off)."""


class CutBuilder(ABC):
    """Training-only: turns a trace plus batch metadata into CutRecord rows.

    The outcome y enters the fixed measurement only, never the compressor.
    """

    @abstractmethod
    def build(self, *batch_context) -> List[Any]:
        ...
