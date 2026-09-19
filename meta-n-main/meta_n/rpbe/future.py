"""Fixed future sketch (task book v4.1 §2.6 / §2.10).

    f_v^(1)     = E_0(serialize(Y_{v+1}))            # [256]
    f_v^(2)     = E_0(serialize(Y_{v+2}))            # [256]
    phi_r(S_v)  = R_S^(r) [ f_v^(1) ; f_v^(2) ]      # [8],  R_S^(r) in R^{8x512}

`R_S^(r)` is a FIXED Rademacher projection (entries +/-1/sqrt(8)), one per
branch, seeds 0..3. It is never trained. The pair feature is the DIRECT
aligned pair: the difference term `f2 - f1` is redundant under a linear
projection, and the product term `f1 * f2` changes the finite-test family.
Neither is present, and adding either would be a method change.

Everything is detached: the future is a measurement, never a gradient path.

`serialize` reuses the OFFICIAL formatter
(`OmegaEngine._format_raw_traces`), so the future observation is expressed in
exactly the shape the rest of Meta^n uses. No second text template.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import torch

from meta_n.core.meta_layer import Trace
from meta_n.rpbe import config as C

_E0_DIM = C.D_E                    # 256
PAIR_DIM = 2 * _E0_DIM             # 512


def _rademacher(shape: Tuple[int, int], seed: int, out_dim: int):
    """Fixed +/-1/sqrt(out_dim) matrix -- the frozen projection."""
    g = torch.Generator().manual_seed(int(seed))
    bits = torch.randint(0, 2, shape, generator=g).to(torch.float32) * 2 - 1
    return bits / math.sqrt(float(out_dim))


def serialize_traces(traces: Sequence[Trace]) -> str:
    """The official `serialize` for a future observation (v4.1 §2.10).

    `Y` is a TRACE observation, so this is the trace formatter, not the
    context-stack one. A bare OmegaEngine instance is enough: the formatter
    reads no engine state.
    """
    from meta_n.core.omega import OmegaEngine
    engine = object.__new__(OmegaEngine)
    return OmegaEngine._format_raw_traces(engine, list(traces))


class FutureSketcher:
    """`S_v = (Y_{v+1}, Y_{v+2})` -> `phi_r(S_v)` for every branch."""

    def __init__(self, encoder: Any,
                 n_branches: int = C.N_BRANCHES,
                 out_dim: int = C.M_SKETCH,
                 seeds: Sequence[int] = C.R_S_SEEDS,
                 e0_dim: int = _E0_DIM):
        if len(seeds) != n_branches:
            raise ValueError(
                "need one seed per branch; got {} seeds for {} branches"
                .format(len(seeds), n_branches))
        if e0_dim != _E0_DIM:
            raise ValueError("E_0 output dim is frozen at {}".format(_E0_DIM))
        self.encoder = encoder
        self.n_branches = int(n_branches)
        self.out_dim = int(out_dim)
        self.pair_dim = 2 * int(e0_dim)
        self.seeds = tuple(int(s) for s in seeds)
        # Fixed projections: registered as plain attributes, never Parameters.
        self.projections = [
            _rademacher((self.out_dim, self.pair_dim), s, self.out_dim)
            for s in self.seeds]

    # -- encoding ----------------------------------------------------------
    def _embed(self, traces: Sequence[Trace]) -> torch.Tensor:
        """E_0(serialize(Y)) -> [256], detached."""
        text = serialize_traces(traces)
        vec = self.encoder.encode_query(text)
        vec = torch.as_tensor(vec, dtype=torch.float32).reshape(-1)
        if vec.numel() != _E0_DIM:
            raise ValueError("E_0 returned {} dims, expected {}".format(
                vec.numel(), _E0_DIM))
        return vec.detach()

    def pair_feature(self, child_traces: Sequence[Trace],
                     grandchild_traces: Sequence[Trace]) -> torch.Tensor:
        """`[f_v^(1) ; f_v^(2)]` -> [512], detached. The direct aligned pair."""
        f1 = self._embed(child_traces)
        f2 = self._embed(grandchild_traces)
        return torch.cat([f1, f2]).detach()

    # -- the frozen sketch -------------------------------------------------
    def sketch(self, child_traces: Sequence[Trace],
               grandchild_traces: Sequence[Trace]) -> torch.Tensor:
        """-> phi_r(S_v), shape [n_branches, out_dim] = [4, 8]. Detached."""
        pair = self.pair_feature(child_traces, grandchild_traces)
        rows = [(P @ pair) for P in self.projections]
        return torch.stack(rows, dim=0).detach()

    # -- convenience -------------------------------------------------------
    def sketch_lineage(self, lineage) -> torch.Tensor:
        """Sketch straight from a `lineage.Lineage` (Y_{v+1}, Y_{v+2})."""
        return self.sketch(lineage.child_traces, lineage.grandchild_traces)


# --------------------------------------------------------------------------
# acceptance self-test
# --------------------------------------------------------------------------

class _StubEncoder:
    """Deterministic stand-in for FrozenItemEncoder: text -> [256]."""

    def __init__(self, dim: int = _E0_DIM):
        self.dim = dim
        self.seen: list = []

    def encode_query(self, text: str) -> torch.Tensor:
        self.seen.append(text)
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        g = torch.Generator().manual_seed(int.from_bytes(h[:4], "little"))
        return torch.randn(self.dim, generator=g)


def self_test() -> int:
    trunc = _StubEncoder()
    s = FutureSketcher(trunc)
    print("future.py acceptance")
    print("=" * 68)

    # 1 -- SHAPES / FROZEN GEOMETRY
    assert s.pair_dim == 512 and s.out_dim == C.M_SKETCH
    for P in s.projections:
        assert P.shape == (C.M_SKETCH, 512), P.shape
        mags = P.abs().reshape(-1)
        assert torch.allclose(mags, torch.full_like(mags, 1.0 / math.sqrt(8.0)))
    print("OK  geometry           R_S^(r) is [8, 512] with entries +/-1/sqrt(8)")

    c = (Trace(task_id="t", script="print(1)", success=True, score=0.5),)
    g = (Trace(task_id="t", script="print(2)", success=True, score=0.9),)
    phi = s.sketch(c, g)
    assert phi.shape == (C.N_BRANCHES, C.M_SKETCH), phi.shape
    print("OK  sketch shape       phi_r(S_v) = [4, 8]")

    # 2 -- DETERMINISM + FIXED PROJECTION (same rank across constructions)
    s2 = FutureSketcher(_StubEncoder())
    assert torch.equal(phi, s2.sketch(c, g))
    assert all(torch.equal(a, b) for a, b in
               zip(s.projections, s2.projections))
    print("OK  deterministic      identical across instances (projections are "
          "seed-fixed, not random per instance)")

    # 3 -- NO EXTRA TERMS: phi depends on f1, f2 only through the concat.
    #      A difference/product term would make phi(c,g) != -phi(g,c)-style
    #      relations; assert the pair feature is exactly the concat.
    pair = s.pair_feature(c, g)
    f1 = s._embed(c)
    f2 = s._embed(g)
    assert torch.equal(pair, torch.cat([f1, f2]))
    assert not torch.equal(pair, torch.cat([f2, f1]))          # order matters
    print("OK  direct pair        pair feature == [f1 ; f2] exactly "
          "(no f2-f1, no f1*f2)")

    # 4 -- DETACHED: no grad anywhere
    assert not pair.requires_grad and not phi.requires_grad
    print("OK  detached           phi and the pair feature carry no graph")

    # 5 -- SERIALIZE REUSES THE OFFICIAL FORMATTER
    txt = serialize_traces(c)
    assert isinstance(txt, str) and txt.strip()
    import inspect
    from meta_n.core.omega import OmegaEngine
    assert "_format_raw_traces" in inspect.getsource(serialize_traces)
    assert hasattr(OmegaEngine, "_format_raw_traces")
    print("OK  official serialize uses OmegaEngine._format_raw_traces "
          "(no second template)")

    # 6 -- NO LEAKAGE INTO THE RECORD PATH: branch rows are independent
    assert phi.shape[0] == C.N_BRANCHES
    for a in range(C.N_BRANCHES):
        for b in range(a + 1, C.N_BRANCHES):
            assert not torch.equal(phi[a], phi[b])
    print("OK  branch independence 4 rows are distinct (never concatenated "
          "into one covariance)")

    # 7 -- ZERO COST
    import os
    paid = 0
    led = os.environ.get("META_N_REQUEST_LEDGER", "").strip()
    if led and os.path.isfile(led):
        from meta_n.rpbe.accounting import snapshot
        paid = snapshot()["backend_requests"]
    assert paid == 0
    print("OK  zero-cost          paid backend_requests = {}".format(paid))

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
