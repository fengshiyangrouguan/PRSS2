"""Validation/test spectral-isolation audit.

Hard invariant: held-out evaluation must never touch Gram statistics, run a
spectral solve, or change any quotient R.
"""

from typing import Dict

import torch


def counts_of_spectral(prss) -> Dict:
    if prss is None:
        return {"gram": 0, "svd": 0}
    gram = sum(int(q.snapshot().get("reader_gram_updates", 0))
               for q in prss.quotients.values())
    svd = sum(int(q.snapshot().get("spectral_updates", 0))
              for q in prss.quotients.values())
    return {"gram": gram, "svd": svd}


def r_copies(prss) -> Dict:
    if prss is None:
        return {}
    return {tau: q.projection().detach().cpu().clone()
            for tau, q in prss.quotients.items()}


def r_max_change(before: Dict, prss) -> float:
    if prss is None:
        return 0.0
    values = []
    for tau, old in before.items():
        values.append(float((old - prss.quotients[tau].projection().detach().cpu()).abs().max()))
    return max(values) if values else 0.0


def assert_clean(before_counts: Dict, before_r: Dict, prss, trace_created: bool,
                 label: str) -> None:
    """Raise if counts, R, or trace changed across a held-out evaluation."""
    after_counts = counts_of_spectral(prss)
    r_change = r_max_change(before_r, prss)
    if before_counts != after_counts or r_change != 0.0 or trace_created:
        raise RuntimeError(
            f"{label} mutated spectral state: counts {before_counts} -> {after_counts}, "
            f"R_change={r_change}, trace_created={trace_created}")
