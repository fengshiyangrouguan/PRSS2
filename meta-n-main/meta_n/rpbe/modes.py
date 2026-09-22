"""Context-reduction modes for the Meta^n x RPBE port (task book v4.1 §0.2)."""

from enum import Enum


class ReductionMode(str, Enum):
    """The context-reduction modes.

    OFFICIAL and PREDICTIVE share an identical ContextBudget (task book v4.1
    §2.13); FULL applies no official reduction and is bounded only by the
    model's hard input limit.

    MATCHED_K4 is the matched-CAPACITY baseline added 2026-09-21, kept separate
    from OFFICIAL on purpose. OFFICIAL answers "how do we compare to the
    original host"; MATCHED_K4 answers "at the SAME compression capacity, is
    learned selection better than the hand-written rule". Two different
    questions -- they must not be collapsed into one number.
    """

    FULL = "full"
    OFFICIAL = "official"
    PREDICTIVE = "predictive"
    MATCHED_K4 = "matched_k4"

    @property
    def uses_official_budget(self) -> bool:
        return self in (ReductionMode.OFFICIAL, ReductionMode.PREDICTIVE,
                        ReductionMode.MATCHED_K4)
