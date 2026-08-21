"""A0: pullback-closed conditional-moment tree compression (theory doc 2026-08-20).

Phase A learns per-interface rank-r coordinate maps R_tau from the conditional
future moment E[a(C) ⊗ φ_Y(Y) | H]; phase B fits per-constructor recursive
operators B_sigma by convex ridge; phase C audits prediction/closure/support/
gain with G0-G4 failure certificates; phase D trains a final readout on the
frozen r-dimensional coordinates.
"""

from prss.a0.audit import evaluate_gates
from prss.a0.operators import OperatorRidge, chi_sigma
from prss.a0.probes import A0Probes, propagate_root_labels, stack_by_tau
from prss.a0.quotient import A0Quotient, randomized_svd

__all__ = [
    "A0Probes",
    "A0Quotient",
    "OperatorRidge",
    "chi_sigma",
    "evaluate_gates",
    "propagate_root_labels",
    "stack_by_tau",
    "randomized_svd",
]
