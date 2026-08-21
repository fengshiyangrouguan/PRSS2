"""A0: pullback-closed conditional-moment tree compression (theory doc 2026-08-20).

Phase A learns per-interface rank-r coordinate maps R_tau from the conditional
future moment E[a(C) ⊗ φ_Y(Y) | H]; phase B fits per-constructor recursive
operators B_sigma by convex ridge; phase C audits prediction/closure/support/
gain with G0-G4 failure certificates; phase D trains a final readout on the
frozen r-dimensional coordinates.
"""

from prss.a0.audit import (ResidualAccumulator, evaluate_gates,
                           proper_score_regret)
from prss.a0.operators import (OperatorRidge, TensorSketchFeatures, chi_sigma,
                               chi_width)
from prss.a0.probes import A0Probes, propagate_root_labels, stack_by_tau
from prss.a0.quotient import A0Quotient, randomized_svd
from prss.a0.weights import DensityRatioWeights

__all__ = [
    "A0Probes",
    "A0Quotient",
    "DensityRatioWeights",
    "OperatorRidge",
    "ResidualAccumulator",
    "TensorSketchFeatures",
    "chi_sigma",
    "chi_width",
    "evaluate_gates",
    "propagate_root_labels",
    "proper_score_regret",
    "stack_by_tau",
    "randomized_svd",
]
