"""Context-reduction modes for the Meta^n x RPBE port (task book v4.1 §0.2)."""

from enum import Enum


class ReductionMode(str, Enum):
    """The three context-reduction modes.

    OFFICIAL and PREDICTIVE share an identical ContextBudget (task book v4.1
    §2.13); FULL applies no official reduction and is bounded only by the
    model's hard input limit.
    """

    FULL = "full"
    OFFICIAL = "official"
    PREDICTIVE = "predictive"

    @property
    def uses_official_budget(self) -> bool:
        return self in (ReductionMode.OFFICIAL, ReductionMode.PREDICTIVE)
