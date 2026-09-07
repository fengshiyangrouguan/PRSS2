"""Dialogue cut records: one cut -> two horizon rows (plan v2 L4).

A dialogue sampled with a random prefix of k turns yields ONE cut at
turn v = k - 3 (the last turn whose memory block is fully in the past of
the query).  Its memory M_v (the SUM-token K/V pair, lifted to z_v by
J_mem) is supervised by the two strictly future utterances:

    row 1: (z_v, p_{v,1}, w=0.5, cut_id)   p_1 = Sketch([1; chi_1] (x) phi_1)
    row 2: (z_v, p_{v,2}, w=0.5, cut_id)   p_2 = Sketch([1; chi_2] (x) phi_2)

where chi_1 = chi(u_{v+1}), chi_2 = chi(u_{v+2}) with the one-update
marker (u_{v+2} is one memory update further into the future), and
phi_1/phi_2 are fixed horizon signatures.  Both rows share the SAME
cut_id and enter the same Ky Fan window; the clustered correction runs
through the existing WeightedWelford path.  Dialogues with k = 3 keep
their task CE and carry NO RPBE row (w_RPBE = 0).

There are no node ids or timestamps here: ``tree_id`` carries the sample
identity (one dialogue = one independent history, which is what the
window's unique-tree gate counts) and ``node`` is a placeholder for the
shared CutRecord schema.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from rpbe.records import CutRecord

MEM_TAU = "mem"
HORIZON_WEIGHTS = (0.5, 0.5)
# Review round 8 (depth-stratified legal cuts): the candidate pool spans
# depths L in {1, 2, 4, 8, 13} where L = compressed-history turn count and
# k = L + 2.  L = 1 gives k = 3 and v = k - 3 = 0: the memory after the
# FIRST turn's compression, supervised by u_1 and u_2 — a legal cut (both
# futures exist inside the same fixed-k prefix).  MIN_K = 3 admits it.
MIN_K = 3  # k >= 3 gives v = k - 3 >= 0 (one compressed turn suffices)


def _fixed_binary(shape, seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.randint(0, 2, shape, generator=g) * 2 - 1


class Llmmaps(nn.Module):
    """Fixed LLM measurement, 4-branch ensemble (review round 8).

    Review round 8 replaces the single 64-dim sketch (variance-prone
    under turn-14's small samples) with FOUR independent low-dim
    sketches: each branch keeps d_chi = 64 and outputs m = 32, so one
    branch is more stable under small samples while the ensemble covers
    ~128 independent directions.  The four branch scores J_1..J_4 are
    averaged at the WINDOW level (J_ens = mean_r J_r) — the branches are
    never concatenated into one 128-dim covariance (that would bring the
    small-sample problem back).

    Depth encoding (review round 8): the cut depth L (compressed-history
    turn count, known AT the cut — no future leakage) modulates the
    future signature: phi_r(h, b(L)) = phi_h + b_L^(r) with a fixed
    per-branch per-depth +/-1 bucket signature.  Shallow and turn-14
    rows therefore live in distinct measurement subspaces and the loss
    can identify "this is a deep compression interface".

    Same construction guarantee as the TGN FixedMaps: every coordinate of
    the (1 + d_chi) x d_phi tensor product is mapped at least once, plus
    fixed repetitions.  Fully frozen; ``pv`` never creates gradient.
    """

    N_BRANCHES = 4          # review round 8 ensemble size
    DEPTH_LEVELS = (1, 2, 4, 8, 13)  # L = compressed-history turn count

    def __init__(self, d_chi: int = 64, d_phi: int = 32, m: int = 32,
                 seed: int = 0, repeats: int = 3, n_branches: int = 4):
        super().__init__()
        self.d_chi = int(d_chi)
        self.d_phi = int(d_phi)
        self.m = int(m)              # per-branch output dim (m = 32)
        self.n_branches = int(n_branches)
        full_dim = (1 + d_chi) * d_phi
        # Per-branch sketch buffers (independent fixed seeds).
        for r in range(self.n_branches):
            bseed = int(seed) + 1000 * r
            rows = torch.arange(full_dim)
            cols = rows % m
            signs = torch.ones(full_dim)
            g = torch.Generator()
            g.manual_seed(bseed + 21)
            extra_n = int(repeats) * full_dim
            extra_rows = torch.randint(0, full_dim, (extra_n,), generator=g)
            extra_cols = torch.randint(0, m, (extra_n,), generator=g)
            extra_signs = torch.randint(0, 2, (extra_n,), generator=g) * 2 - 1
            self.register_buffer(
                "sketch_rows_{}".format(r),
                torch.cat([rows, extra_rows]), persistent=True)
            self.register_buffer(
                "sketch_cols_{}".format(r),
                torch.cat([cols, extra_cols]), persistent=True)
            self.register_buffer(
                "sketch_signs_{}".format(r),
                torch.cat([signs, extra_signs]), persistent=True)
            self.register_buffer(
                "scale_{}".format(r),
                torch.tensor((m / full_dim) ** 0.5), persistent=True)
        # Fixed per-branch per-depth bucket signatures (d_phi-dim, +/-1).
        self.depth_bucket_seed = int(seed) + 7000
        for r in range(self.n_branches):
            for L in self.DEPTH_LEVELS:
                self.register_buffer(
                    "bucket_L{}_{}".format(L, r),
                    _fixed_binary((d_phi,), self.depth_bucket_seed
                                  + 131 * L + 17 * r), persistent=True)

    def _bucket(self, r: int, L) -> torch.Tensor:
        if L not in self.DEPTH_LEVELS:
            L = self.DEPTH_LEVELS[int(L) % len(self.DEPTH_LEVELS)]
        return getattr(self, "bucket_L{}_{}".format(int(L), r))

    def pv(self, chi: torch.Tensor, phi: torch.Tensor,
           L: Optional[int] = None) -> torch.Tensor:
        """CONDITIONAL measurement with depth bucket (review round 8):
        p_r = Sketch_r([1; chi] (x) (phi + b_L^r)) for r = 0..3, where
        chi is the CONTEXT measurement, phi the FUTURE measurement and
        b_L^r the fixed depth signature.  chi [d_chi] or [B, d_chi],
        phi [d_phi] (input); output [n_branches, m] for single inputs or
        [B, n_branches, m]."""
        single = chi.dim() == 1
        if single:
            chi = chi.unsqueeze(0)
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        with torch.no_grad():
            body = torch.cat([torch.ones(chi.shape[0], 1, dtype=chi.dtype,
                                         device=chi.device), chi], dim=1)
            outs = []
            for r in range(self.n_branches):
                phi_r = phi + self._bucket(r, L).to(
                    dtype=phi.dtype, device=phi.device)
                prod = torch.einsum("bd,bf->bdf", body, phi_r).reshape(
                    chi.shape[0], -1)
                cols = getattr(self, "sketch_cols_{}".format(r)).to(
                    chi.device)
                rows = getattr(self, "sketch_rows_{}".format(r)).to(
                    chi.device)
                signs = getattr(self, "sketch_signs_{}".format(r)).to(
                    dtype=chi.dtype, device=chi.device)
                out = torch.zeros(chi.shape[0], self.m, dtype=chi.dtype,
                                  device=chi.device)
                out.index_add_(1, cols, prod[:, rows] * signs)
                out = out * getattr(self, "scale_{}".format(r))
                outs.append(out)
            p = torch.stack(outs, dim=1).detach()  # [B, nb, m]
            return p[0].detach() if single else p


@dataclass
class DialogueMeta:
    """Deterministic per-sample metadata (L3): k, turn spans, SUM rows.

    ``sum_positions``: list (per turn) of the padded (S0, S1) positions.
    ``utterance_spans``: list (per turn) of (start, stop) token slices in
    the FULL padded input.  Both are computed by the collator and are
    consumed BEFORE the model (never part of the model input).
    """

    sample_id: int
    k: int                      # fixed depth prefix length (turn count)
    sum_positions: List[tuple]  # per turn (0-indexed block turns)
    utterance_spans: List[tuple]  # per turn, aligned with sum blocks
    orig_id: int = -1           # stable ORIGINAL dialogue id (dataset row
                                # index); review round 8 tree identity.
                                # -1 = legacy stream-cursor path.


class DialogueCutBuilder:
    """One dialogue -> zero or two CutRecord rows (2Obs, plan L4)."""

    def __init__(self, maps: Llmmaps, *, seed: int = 0, z_dim: int = 128):
        self.maps = maps
        self.seed = int(seed)
        self.z_dim = int(z_dim)
        self._next_oid = 0

    @property
    def next_oid(self) -> int:
        return self._next_oid

    @next_oid.setter
    def next_oid(self, value: int) -> None:
        self._next_oid = int(value)

    def build(self, meta: DialogueMeta, z_v: torch.Tensor,
              chi_1: torch.Tensor, chi_2: torch.Tensor,
              phi_1: torch.Tensor, phi_2: torch.Tensor,
              stats: Optional[dict] = None) -> List[CutRecord]:
        """One cut (v = k - 3) -> two horizon rows sharing the cut_id.

        ``z_v`` keeps its graph (pass 2 replays the exact gradient);
        ``chi_1/2`` are constants (the UtteranceEmbed path is no_grad).
        Returns [] for k < 4 (task CE keeps its own gradient, w_RPBE=0).
        """
        k = int(meta.k)
        if k < MIN_K:
            if stats is not None:
                stats.setdefault("skipped_k_lt_3", 0)
                stats["skipped_k_lt_3"] += 1
            return []
        v = k - 3
        cut_occurrence = self._next_oid
        self._next_oid += 1
        rows: List[CutRecord] = []
        for horizon in (1, 2):
            chi = chi_1 if horizon == 1 else chi_2
            phi = phi_1 if horizon == 1 else phi_2
            if chi.dim() == 2:
                chi = chi[0]
            if phi.dim() == 2:
                phi = phi[0]
            # Review round 8: depth L = meta.k - 1 (compressed-history
            # turn count) enters the measurement via the fixed depth
            # bucket — known AT the cut, no future leakage.
            p = self.maps.pv(chi, phi, L=int(meta.k) - 1)
            # Review round 8: tree identity = the STABLE original dialogue
            # id, not the per-step stream cursor (which would count every
            # re-sampling of a dialogue as an independent history).
            did = int(meta.orig_id) if meta.orig_id is not None \
                and int(meta.orig_id) >= 0 else int(meta.sample_id)
            rows.append(CutRecord(
                tree_id=did,
                occurrence_id=cut_occurrence,
                tau=MEM_TAU,
                horizon=horizon,
                node=did,  # stable dialogue id (schema placeholder filled)
                time=float(v),
                z=z_v[0] if z_v.dim() == 2 else z_v,
                context={"horizon": horizon, "cut_turn": v, "k": k,
                         "chi_tag": 0 if horizon == 1 else 1},
                outcome=1.0,  # no binary label; p is the content sketch
                outcome_id=(int(meta.sample_id), cut_occurrence, horizon),
                weight=HORIZON_WEIGHTS[horizon - 1],
                p_override=p,
            ))
        return rows
