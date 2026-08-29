"""Fixed measurement maps: psi(c, y) = sketch([1; phi_C(c)] (x) phi_Y(y)).

Everything here is frozen: all random matrices are generated once from
``rpbe_seed`` and registered as buffers; ``psi`` never creates gradient.  The
tensor-product structure is mandatory — a plain concatenation would only keep
marginal information and could not represent context-outcome pairings.

Small helper note: the continuous context feature (delta_t) is scaled by a
fixed ``delta_t_scale`` before the RFF pass (no learned or running statistics
here; batch whitening happens later, inside the Ky Fan score).
"""

import hashlib

import torch
from torch import nn


def _fixed_binary(shape, seed, device=None, dtype=None):
    """Deterministic +-1 table from a scalar seed (no RNG state leakage)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randint(0, 2, shape, generator=g,
                         dtype=torch.int64).to(device=device) * 2 - 1


class FixedMaps(nn.Module):
    """phi_C, phi_Y and the sparse CountSketch of their tensor product."""

    def __init__(self, cfg, *, num_counter_bins: int = 4096,
                 p_cache_max_entries: int = 4096):
        super().__init__()
        self.cfg = cfg
        self.d_c = int(cfg.d_c)
        self.d_f = int(cfg.d_f)
        self.m = int(cfg.m)
        self.num_counter_bins = int(num_counter_bins)
        # Fixed-measurement batch cache (see pv_batch docstring).  4096
        # batches x ~2300 rows x 64 floats x 4B ~ 2.4GB worst case; set
        # 0 to disable.
        self.p_cache_max_entries = int(p_cache_max_entries)
        self._p_cache = {}
        # Projection counters for performance/audit reporting.
        self._pv_calls = 0
        self._pv_batch_calls = 0
        # Cache effectiveness counters (seventh review): the batch-level
        # key depends on the sampled cut set, which changes with the
        # growing global_step seed — hit/miss reporting tells us whether
        # the cache actually pays off across epochs.
        self._p_cache_hits = 0
        self._p_cache_misses = 0
        # Frozen at construction: later cfg mutations must not change the
        # measurement (the fingerprint covers this value).
        self._delta_t_scale = float(cfg.delta_t_scale)
        seed = int(cfg.rpbe_seed)

        # Rows: [0, bins) counterpart hash; bins+0/1 -> role; bins+2/3 -> query.
        self.register_buffer("categorical_c", _fixed_binary(
            (self.num_counter_bins + 4, self.d_c), seed + 1), persistent=True)
        # Future y in {0, 1}: two fixed d_f-dim signatures.  Both rows are
        # nonzero +-1 signatures, so phi_Y(0) != 0 (a zero signature would
        # kill every negative-label row's tensor product).
        self.register_buffer("future_table", _fixed_binary(
            (2, self.d_f), seed + 2), persistent=True)
        # Exactly two real future observations Y1/Y2.  There is no synthetic
        # root/star horizon and missing futures are masked by row omission.
        self.register_buffer("horizon_table", _fixed_binary(
            (2, self.d_c), seed + 6), persistent=True)
        # PathSketch: fixed per-step signatures for the upward-walk
        # structure.  rel 0 = SELF recursion step, rel 1 = neighbor edge.
        self.register_buffer("path_rel_table", _fixed_binary(
            (2, self.d_c), seed + 7), persistent=True)
        # RFF for the scaled delta_t: cos(W x + b).
        self.register_buffer("rff_w", torch.randn(
            (1, self.d_c), generator=torch.Generator().manual_seed(seed + 3)),
            persistent=True)
        self.register_buffer("rff_b", torch.rand(
            (self.d_c,), generator=torch.Generator().manual_seed(seed + 4)) * 6.2832,
            persistent=True)

        # CountSketch: every input coordinate of the full tensor product
        # [1; phi_C] (x) phi_Y ((d_c+1) * d_f rows) is mapped at least once
        # (standard sketch guarantee — the old random-only construction could
        # drop coordinates entirely), plus k-1 random repetitions.
        k = 10
        full_dim = (self.d_c + 1) * self.d_f
        g = torch.Generator().manual_seed(seed + 5)
        base_rows = torch.arange(full_dim)
        base_cols = torch.arange(full_dim) % self.m
        base_signs = torch.ones(full_dim, dtype=torch.int64)
        extra_n = (k - 1) * full_dim
        extra_rows = torch.randint(0, full_dim, (extra_n,), generator=g)
        extra_cols = torch.randint(0, self.m, (extra_n,), generator=g)
        extra_signs = torch.randint(0, 2, (extra_n,), generator=g) * 2 - 1
        self._sketch_nnz = full_dim + extra_n
        rows = torch.cat([base_rows, extra_rows])
        cols = torch.cat([base_cols, extra_cols])
        signs = torch.cat([base_signs, extra_signs])
        sketch_indices = torch.stack([rows, cols], dim=0)  # [2, nnz]
        self.register_buffer("sketch_indices", sketch_indices, persistent=True)
        self.register_buffer("sketch_signs", signs, persistent=True)
        self._sketch_full_dim = full_dim

    # ------------------------------------------------------------- primitives
    def _path_vector(self, path, dtype, device) -> torch.Tensor:
        """Fixed PathSketch of the upward walk: sum over steps of
        ``cos(dt/scale W + b) + rel_signature``.  Empty path -> zeros."""
        out = torch.zeros((1, self.d_c), dtype=dtype, device=device)
        for rel, dt in path:
            if int(rel) not in (0, 1):
                raise ValueError("path relation must be 0 or 1, got {}"
                                 .format(rel))
            d = torch.tensor(float(dt) / self._delta_t_scale,
                             dtype=dtype, device=device).reshape(1, 1)
            out = out + torch.cos(d @ self.rff_w + self.rff_b) \
                + self.path_rel_table[int(rel)].to(dtype)
        return out

    def context_vector(self, context) -> torch.Tensor:
        """phi_C(c) for one row: {delta_t, counterpart, role, query_type,
        horizon, path}."""
        delta = float(context["delta_t"]) / self._delta_t_scale
        delta_t = torch.tensor(delta, dtype=self.rff_w.dtype,
                               device=self.rff_w.device).reshape(1, 1)
        rff = torch.cos(delta_t @ self.rff_w + self.rff_b)  # [1, d_c]
        partner = int(context["counterpart"]) % self.num_counter_bins
        role = int(context["role"]) % 2
        query = int(context["query_type"]) % 2
        h = int(context["horizon"])
        if h not in (1, 2):
            raise ValueError("horizon must be 1 or 2, got {}".format(h))
        cat = (self.categorical_c[partner]
               + self.categorical_c[self.num_counter_bins + role]
               + self.categorical_c[self.num_counter_bins + 2 + query]
               + self.horizon_table[h - 1])
        path_vec = self._path_vector(context.get("path", []),
                                     self.rff_w.dtype, self.rff_w.device)
        return rff + cat.to(rff.dtype) + path_vec  # [1, d_c]

    def future_vector(self, outcome: float) -> torch.Tensor:
        """phi_Y(y) for one row; y in {0, 1}."""
        y = 1 if float(outcome) > 0.5 else 0
        return self.future_table[y].to(self.rff_w.dtype)  # [d_f]

    def psi(self, context, outcome: float) -> torch.Tensor:
        """sketch([1; phi_C] (x) phi_Y) -> [m]; no gradient."""
        self._pv_calls += 1
        with torch.no_grad():
            c = self.context_vector(context)               # [1, d_c]
            f = self.future_vector(outcome)                # [d_f]
            body = torch.cat([torch.ones(1, dtype=c.dtype, device=c.device),
                              c[0]])                        # [1+d_c]
            prod = torch.outer(body, f).reshape(-1)         # [(1+d_c)(1+d_f)]
            out = torch.zeros(self.m, dtype=prod.dtype, device=prod.device)
            out.index_add_(0, self.sketch_indices[1],
                           prod[self.sketch_indices[0]] * self.sketch_signs)
            return out

    def pv(self, context, outcome: float) -> torch.Tensor:
        """One joint-test row for one cut."""
        return self.psi(context, outcome)

    # ----------------------------------------------------------- batch path
    def pv_batch(self, contexts, outcomes) -> torch.Tensor:
        """Vectorized joint tests: [N, m], no gradient, one pass.

        ``contexts`` is a list of context dicts, ``outcomes`` a float list;
        the per-row small-operator storm of ``pv`` is avoided by batching the
        RFF pass, the categorical gathers and a single index_add.

        Fixed-measurement cache (sixth review, step 4): psi is a FIXED map
        and the sampling/tree structure is deterministic, so the same
        batch of (context, outcome) rows recurs every epoch.  The batch is
        keyed by its full row-semantic tuple; hits skip the CountSketch
        scatter entirely (the measured ~210 ms/batch hotspot).  The cache
        holds CPU float32 copies and is bounded (oldest-eviction by
        dict order); a map/seed change produces different keys by
        construction, so no version stamping is needed.
        """
        n = len(contexts)
        if n == 0:
            return torch.zeros((0, self.m), dtype=self.rff_w.dtype,
                               device=self.rff_w.device)
        dev = self.rff_w.device
        key = tuple((int(c["horizon"]), float(c["delta_t"]),
                     int(c["counterpart"]), int(c["role"]),
                     int(c["query_type"]),
                     tuple((int(rel), float(dt))
                           for rel, dt in c.get("path", [])),
                     float(y))
                    for c, y in zip(contexts, outcomes))
        self._pv_batch_calls += 1
        hit = self._p_cache.get(key)
        if hit is not None:
            self._p_cache_hits += 1
            return hit.to(dev, dtype=self.rff_w.dtype)
        self._p_cache_misses += 1
        with torch.no_grad():
            deltas = torch.tensor(
                [float(c["delta_t"]) for c in contexts],
                dtype=self.rff_w.dtype, device=dev).reshape(n, 1) \
                / float(self._delta_t_scale)
            rff = torch.cos(deltas @ self.rff_w + self.rff_b)        # [N, d_c]
            partners = torch.tensor(
                [int(c["counterpart"]) % self.num_counter_bins
                 for c in contexts], dtype=torch.long, device=dev)
            roles = torch.tensor([int(c["role"]) % 2 for c in contexts],
                                 dtype=torch.long, device=dev)
            queries = torch.tensor([int(c["query_type"]) % 2
                                    for c in contexts],
                                   dtype=torch.long, device=dev)
            horizons = torch.tensor(
                [self._check_horizon(int(c["horizon"])) for c in contexts],
                dtype=torch.long, device=dev)
            cat = (self.categorical_c[partners]
                   + self.categorical_c[self.num_counter_bins + roles]
                   + self.categorical_c[self.num_counter_bins + 2 + queries]
                   + self.horizon_table[horizons - 1])
            path_vec = torch.stack(
                [self._path_vector(c.get("path", []), self.rff_w.dtype, dev)[0]
                 for c in contexts], dim=0)                          # [N, d_c]
            c_vec = rff + cat.to(rff.dtype) + path_vec               # [N, d_c]
            y_idx = torch.tensor([1 if float(y) > 0.5 else 0
                                  for y in outcomes],
                                 dtype=torch.long, device=dev)
            f_vec = self.future_table[y_idx].to(rff.dtype)           # [N, d_f]
            body = torch.cat([
                torch.ones(n, 1, dtype=c_vec.dtype, device=dev), c_vec],
                dim=1)                                               # [N, 1+d_c]
            prod = torch.einsum("ni,nj->nij", body, f_vec).reshape(n, -1)
            flat = prod.reshape(-1)
            row_off = (torch.arange(n, device=dev) * self._sketch_full_dim
                       ).unsqueeze(1)
            col_off = (torch.arange(n, device=dev) * self.m).unsqueeze(1)
            row_idx = (self.sketch_indices[0].unsqueeze(0) + row_off).reshape(-1)
            col_idx = (self.sketch_indices[1].unsqueeze(0) + col_off).reshape(-1)
            out = torch.zeros((n, self.m), dtype=flat.dtype, device=dev)
            out.view(-1).index_add_(
                0, col_idx,
                flat[row_idx] * self.sketch_signs.repeat(n))
            if self.p_cache_max_entries > 0:
                while len(self._p_cache) >= self.p_cache_max_entries:
                    self._p_cache.pop(next(iter(self._p_cache)))
                self._p_cache[key] = out.detach().float().cpu()
            return out

    @staticmethod
    def _check_horizon(h: int) -> int:
        if h not in (1, 2):
            raise ValueError("horizon must be 1 or 2, got {}".format(h))
        return h

    # ------------------------------------------------------------------ audit
    def isolation_fingerprint(self) -> dict:
        """Seed/version + FULL byte hash of every frozen buffer and of the
        measurement-affecting config (delta_t_scale)."""
        h = hashlib.sha256()
        h.update(str(self.cfg.rpbe_seed).encode())
        h.update(str(self._sketch_full_dim).encode())
        h.update(str(self._delta_t_scale).encode())
        for name in ("categorical_c", "future_table", "rff_w", "rff_b",
                     "sketch_indices", "sketch_signs",
                     "horizon_table", "path_rel_table"):
            buf = getattr(self, name)
            h.update(buf.detach().cpu().reshape(-1).contiguous()
                     .numpy().tobytes())
        return {"seed": int(self.cfg.rpbe_seed),
                "delta_t_scale": self._delta_t_scale,
                "sha256": h.hexdigest()}
