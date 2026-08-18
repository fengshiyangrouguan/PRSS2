"""Host adapter and outside-bridge abstractions.

A host adapter keeps the host's original call surface and only changes what the
parent sees after each recursive aggregation: the host-width quotient ``z = R h``.
The outside bridge builds training-only continuation contexts and auxiliary losses
from the adapter's trace.
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
from torch import nn

from prss.auxiliary import AuxiliaryBatch
from prss.state import RecursiveTrace


class HostAdapter(nn.Module, ABC):
    """Wraps one host model's recursive aggregation surface.

    The host aggregate itself is never replaced: the adapter computes the vanilla
    aggregate, builds the candidate, projects it, and returns only the quotient to
    the parent.  Tracing is training-only and must never change the main forward.
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


class OutsideBridge(ABC):
    """Training-only: turns a trace plus batch metadata into an AuxiliaryBatch.

    The target label enters losses only, never the context encoder.
    """

    @abstractmethod
    def build(self, *batch_context) -> AuxiliaryBatch:
        ...
