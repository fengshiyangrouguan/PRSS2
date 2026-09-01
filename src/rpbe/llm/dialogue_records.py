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
MIN_K = 4  # k >= 4 gives v = k - 3 >= 1 (memory must exist)


def _fixed_binary(shape, seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.randint(0, 2, shape, generator=g) * 2 - 1


class Llmmaps(nn.Module):
    """Fixed LLM measurement: p_h = CountSketch([1; chi] (x) phi_h) -> [m].

    Same construction guarantee as the TGN FixedMaps: every coordinate of
    the (1 + d_chi) x d_phi tensor product is mapped at least once, plus
    fixed repetitions.  Fully frozen; ``pv`` never creates gradient.
    """

    def __init__(self, d_chi: int = 64, d_phi: int = 32, m: int = 64,
                 seed: int = 0, repeats: int = 3):
        super().__init__()
        self.d_chi = int(d_chi)
        self.d_phi = int(d_phi)
        self.m = int(m)
        self.register_buffer("phi_table", _fixed_binary((2, d_phi),
                                                         seed + 20),
                             persistent=True)
        full_dim = (1 + d_chi) * d_phi
        rows = torch.arange(full_dim)
        cols = rows % m
        signs = torch.ones(full_dim)
        g = torch.Generator()
        g.manual_seed(int(seed) + 21)
        extra_n = int(repeats) * full_dim
        extra_rows = torch.randint(0, full_dim, (extra_n,), generator=g)
        extra_cols = torch.randint(0, m, (extra_n,), generator=g)
        extra_signs = torch.randint(0, 2, (extra_n,), generator=g) * 2 - 1
        self.register_buffer("sketch_rows", torch.cat([rows, extra_rows]),
                             persistent=True)
        self.register_buffer("sketch_cols", torch.cat([cols, extra_cols]),
                             persistent=True)
        self.register_buffer("sketch_signs", torch.cat([signs, extra_signs]),
                             persistent=True)
        self.register_buffer(
            "scale", torch.tensor((m / full_dim) ** 0.5), persistent=True)

    def pv(self, chi: torch.Tensor, horizon: int) -> torch.Tensor:
        """p_h = Sketch([1; chi] (x) phi_h); chi [d_chi] or [B, d_chi]
        (output shape follows the input rank)."""
        if int(horizon) not in (1, 2):
            raise ValueError("horizon must be 1 or 2, got {}".format(horizon))
        single = chi.dim() == 1
        if single:
            chi = chi.unsqueeze(0)
        with torch.no_grad():
            phi = self.phi_table[int(horizon) - 1].to(dtype=chi.dtype,
                                                      device=chi.device)
            body = torch.cat([torch.ones(chi.shape[0], 1, dtype=chi.dtype,
                                         device=chi.device), chi], dim=1)
            prod = torch.einsum("bd,f->bdf", body, phi).reshape(
                chi.shape[0], -1)
            cols = self.sketch_cols.to(chi.device)
            rows = self.sketch_rows.to(chi.device)
            signs = self.sketch_signs.to(dtype=chi.dtype, device=chi.device)
            out = torch.zeros(chi.shape[0], self.m, dtype=chi.dtype,
                              device=chi.device)
            out.index_add_(1, cols, prod[:, rows] * signs)
            out = out * self.scale
            return out[0].detach() if single else out.detach()


@dataclass
class DialogueMeta:
    """Deterministic per-sample metadata (L3): k, turn spans, SUM rows.

    ``sum_positions``: list (per turn) of the padded (S0, S1) positions.
    ``utterance_spans``: list (per turn) of (start, stop) token slices in
    the FULL padded input.  Both are computed by the collator and are
    consumed BEFORE the model (never part of the model input).
    """

    sample_id: int
    k: int                      # random prefix length (turn count)
    sum_positions: List[tuple]  # per turn (0-indexed block turns)
    utterance_spans: List[tuple]  # per turn, aligned with sum blocks


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
              stats: Optional[dict] = None) -> List[CutRecord]:
        """One cut (v = k - 3) -> two horizon rows sharing the cut_id.

        ``z_v`` keeps its graph (pass 2 replays the exact gradient);
        ``chi_1/2`` are constants (the UtteranceEmbed path is no_grad).
        Returns [] for k < 4 (task CE keeps its own gradient, w_RPBE=0).
        """
        k = int(meta.k)
        if k < MIN_K:
            if stats is not None:
                stats.setdefault("skipped_k_lt_4", 0)
                stats["skipped_k_lt_4"] += 1
            return []
        v = k - 3
        cut_occurrence = self._next_oid
        self._next_oid += 1
        rows: List[CutRecord] = []
        for horizon in (1, 2):
            chi = chi_1 if horizon == 1 else chi_2
            if chi.dim() == 2:
                chi = chi[0]
            p = self.maps.pv(chi, horizon)
            rows.append(CutRecord(
                tree_id=int(meta.sample_id),
                occurrence_id=cut_occurrence,
                tau=MEM_TAU,
                horizon=horizon,
                node=int(meta.sample_id),  # schema placeholder
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
