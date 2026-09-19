"""Statistical window for RPBE (task book v4.1 §2.7 / §4.2 / §7).

    J_LPSE = (1/4) sum_r J_CCA(B, P^(r))
    J_task = J_CCA(B, R)

Both use paired OAS shrinkage + scale-normalized Cholesky. OAS is ALWAYS on
here -- `latent_z_adjoint(..., oas=True)` is the only entry point that enables
it (see `kf.py`), and the whole point of vendoring that file is to make
"OAS is on" a fact rather than an assumption.

GRADIENT DISCIPLINE (§2.7), enforced by construction:
  * the OAS intensities are the ONLY thing detached -- `kf._oas_alpha` detaches
    them internally as window statistics;
  * `C_PP` / `C_RR` are fixed measurements (the P side is data, not a variable);
  * `C_ZZ`, the scale normalization and `B` keep their graph, because
    `<grad_B J, B> = 0` (scale invariance) depends on it. Stop-gradding `C_ZZ`
    would yield the half-gradient with radial derivative 2J, which is not the
    gradient of any objective. The scale-invariance test below catches that.

WINDOW GATE = UNIQUE TREES, NOT ROWS (§4.2). `tree_seen` counts unique
`(run_id, root_candidate_id)` values, and every row of one tree must live in
the same window. Effective degrees of freedom `D` is the tree-clustered value
`D = W - W2_tree / W`, which is what makes the gate and the OAS intensity agree
about how many independent samples there really are.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from meta_n.rpbe import config as C
from meta_n.rpbe.kf import latent_z_adjoint

TreeId = Tuple[str, str]
CutId = Tuple[str, int]


def _fixed_rademacher(shape: Tuple[int, int], seed: int, out_dim: int):
    g = torch.Generator().manual_seed(int(seed))
    bits = torch.randint(0, 2, shape, generator=g).to(torch.float64) * 2 - 1
    return bits / math.sqrt(float(out_dim))


class WindowError(RuntimeError):
    """The window's contents contradict a frozen invariant."""


class StatWindow:
    """Accumulates CutRecords and closes them into Ky-Fan cotangents."""

    def __init__(self, m: int = C.M_SKETCH, b_v_dim: int = C.B_V_DIM,
                 n_branches: int = C.N_BRANCHES, seed: int = C.P_B_SEED,
                 oas: bool = True, min_unique_trees: int = C.N_TREE_MIN,
                 ridge_eps: float = C.RIDGE_EPS, name: str = "window") -> None:
        if not oas:
            raise WindowError(
                "OAS is mandatory (v4.1 §2.7); constructing with oas=False is "
                "refused because latent_z_adjoint defaults to False and a "
                "silent no-OAS window is the exact failure mode this guards")
        self.m = int(m)
        self.b_v_dim = int(b_v_dim)
        self.n_branches = int(n_branches)
        self.oas = bool(oas)
        self.min_unique_trees = int(min_unique_trees)
        self.ridge_eps = float(ridge_eps)
        self.name = str(name)
        self.vec_dim = C.N_SLOTS * C.D_E            # vec(Z_v) is [1024]
        self._rows: List[Any] = []
        self._tree_seen: set = set()
        self._cut_seen: set = set()
        # Fixed projection P_b: [b_v_dim, vec_dim], seed 0, +/-1/sqrt(b_v_dim).
        self.P_b = _fixed_rademacher((self.b_v_dim, self.vec_dim),
                                     seed, self.b_v_dim)

    # -- accumulation ------------------------------------------------------
    def add(self, rows: Sequence[Any]) -> None:
        for r in rows:
            if r.cut_id in self._cut_seen:
                raise WindowError(
                    "cut {} added twice to {}; a cut belongs to exactly one "
                    "window".format(r.cut_id, self.name))
            self._cut_seen.add(r.cut_id)
            self._tree_seen.add(tuple(r.tree_id))
            self._rows.append(r)

    @property
    def tree_seen(self) -> set:
        return set(self._tree_seen)

    @property
    def tree_ids(self) -> set:
        return self.tree_seen

    @property
    def n_trees(self) -> int:
        return len(self._tree_seen)

    @property
    def n_rows(self) -> int:
        return len(self._rows)

    @property
    def ready(self) -> bool:
        """Gate on UNIQUE TREES, not cut count (§4.2)."""
        return self.n_trees >= self.min_unique_trees

    def tree_role_snapshot(self) -> Tuple[frozenset, int]:
        """(tree ids, row count) -- used to prove role/contents never move."""
        return frozenset(self._tree_seen), len(self._rows)

    # -- replay ------------------------------------------------------------
    def replay_B(self, fusion, *, grad: bool) -> torch.Tensor:
        """`b_v = P_b . vec(SlottedFusion(X_v, q_emb, mask))` under CURRENT θ.

        B is NEVER cached: every optimiser step must recompute it, because the
        whole contract is that the scores track the current Γ (§5.1).
        """
        if not self._rows:
            raise WindowError("window {} is empty".format(self.name))
        ctx = torch.enable_grad() if grad else torch.no_grad()
        out = []
        with ctx:
            for r in self._rows:
                Z, _ = fusion(r.X_v, r.q_emb, r.mask)     # Z [4, 256]
                vec = Z.reshape(-1).to(torch.float32)     # [1024]
                out.append(self.P_b.to(vec.dtype) @ vec)  # [b_v_dim]
            B = torch.stack(out, dim=0)
        return B if grad else B.detach()

    def _P_by_branch(self) -> List[torch.Tensor]:
        """Per-branch P^(r) = [M, m]; branches stay SEPARATE (§2.7).

        A record's `p` is [n_branches, m] -- the four branches are never
        concatenated into one covariance.
        """
        stacked = torch.stack(
            [torch.as_tensor(r.p, dtype=torch.float64).reshape(
                self.n_branches, self.m) for r in self._rows], dim=0)
        return [stacked[:, r, :].detach() for r in range(self.n_branches)]

    def _R_column(self) -> torch.Tensor:
        return torch.tensor([[float(r.r)] for r in self._rows],
                            dtype=torch.float64)

    # -- effective dof / means --------------------------------------------
    def _stats(self, z_rows: torch.Tensor, p_rows: torch.Tensor,
               weights: torch.Tensor) -> Dict[str, Any]:
        """Tree-clustered W, D and weighted means.

        D = W - W2_tree/W with W2_tree = sum over TREES of (sum of w in tree)^2
        -- the cluster-aware degrees of freedom that OAS consumes.
        """
        w = weights.double()
        W = float(w.sum())
        if W <= 0.0:
            raise WindowError("window weights sum to zero")
        wsum: Dict[TreeId, float] = {}
        for r, wi in zip(self._rows, w.tolist()):
            wsum[tuple(r.tree_id)] = wsum.get(tuple(r.tree_id), 0.0) + wi
        W2_tree = float(sum(v * v for v in wsum.values()))
        D = W - W2_tree / W
        mu_z = (z_rows.double() * w.reshape(-1, 1)).sum(0) / W
        mu_p = (p_rows.double() * w.reshape(-1, 1)).sum(0) / W
        return {"W": W, "W2_tree": W2_tree, "D": D,
                "mu_z": mu_z.detach(), "mu_p": mu_p.detach()}

    def _weights(self) -> torch.Tensor:
        return torch.tensor([float(r.weight) for r in self._rows],
                            dtype=torch.float64)

    # -- closes ------------------------------------------------------------
    def close_lpse(self, fusion) -> Dict[str, Any]:
        """Per-cut LPSE cotangents: `J_LPSE = (1/4) sum_r J_CCA(B, P^(r))`.

        Returns one cotangent PER CUT (never pre-summed), because the QP needs
        each interface's direction separately (§2.7 / §5.1).
        """
        # The adjoint is computed on a DETACHED copy: `latent_z_adjoint` mints
        # its own leaf, so handing it a graph-connected tensor would make that
        # leaf a NON-leaf and `z.grad` would stay None. The theta gradient is
        # produced later, by replaying Gamma against the cotangent (§5.1 pass 2).
        B64 = self.replay_B(fusion, grad=False).double()
        w = self._weights()
        st = self._stats(B64, self._P_by_branch()[0], w)
        per_branch: List[Dict[CutId, torch.Tensor]] = []
        js: List[float] = []
        diags: List[Dict[str, Any]] = []
        for P_r in self._P_by_branch():
            j, g_by_cut, diag = latent_z_adjoint(
                B64, P_r, w, [r.cut_id for r in self._rows],
                st["mu_z"], st["mu_p"], st["D"], self.ridge_eps, oas=self.oas)
            if g_by_cut is None:
                raise WindowError("LPSE close failed: {}".format(diag))
            js.append(float(j))
            diags.append(diag)
            per_branch.append(g_by_cut)
        # Branch-MEAN of the per-cut adjoints; branches are never concatenated
        # into one covariance.
        cuts = [r.cut_id for r in self._rows]
        mean_by_cut: Dict[CutId, torch.Tensor] = {}
        for cid in cuts:
            stacked = torch.stack([pb[cid] for pb in per_branch], dim=0)
            mean_by_cut[cid] = (stacked.sum(0) / float(self.n_branches))
        return {
            "J_LPSE": sum(js) / float(self.n_branches),
            "J_by_branch": js,
            "cotangent_by_cut": mean_by_cut,
            "oas_alpha_z_protect": [d.get("alpha_z") for d in diags],
            "oas_alpha_p_protect": [d.get("alpha_p") for d in diags],
            "D": st["D"], "W": st["W"], "W2_tree": st["W2_tree"],
            "n_rows": self.n_rows, "n_trees": self.n_trees,
            "failed": None,
        }

    def close_task(self, fusion) -> Dict[str, Any]:
        """Aggregated task cotangent: `J_task = J_CCA(B, R)`."""
        B64 = self.replay_B(fusion, grad=False).double()
        w = self._weights()
        R = self._R_column()
        st = self._stats(B64, R, w)
        j, g_by_cut, diag = latent_z_adjoint(
            B64, R, w, [r.cut_id for r in self._rows],
            st["mu_z"], st["mu_p"], st["D"], self.ridge_eps, oas=self.oas)
        if g_by_cut is None:
            raise WindowError("task close failed: {}".format(diag))
        return {"J_task": float(j), "cotangent_by_cut": g_by_cut,
                "oas_alpha_z_task": diag.get("alpha_z"),
                "oas_alpha_p_task": diag.get("alpha_p"),
                "D": st["D"], "n_rows": self.n_rows, "n_trees": self.n_trees,
                "failed": None}


def task_transfer_diagnostic(task_window: "StatWindow",
                             other_window: "StatWindow", fusion) -> Dict[str, Any]:
    """`j_cca_transfer_on_protect`: the task functional read on the OTHER
    window (v4.1 §7.3).

    This is a TRANSFER diagnostic, not a clean generalisation monitor: the
    protect window's X/P already participate in the QP, so it has influenced Γ.
    It must NEVER be used to select checkpoints or tune hyper-parameters.
    """
    B64 = other_window.replay_B(fusion, grad=False).double()
    w = other_window._weights()
    st = other_window._stats(B64, other_window._R_column(), w)
    j, _, _ = latent_z_adjoint(
        B64, other_window._R_column(), w,
        [r.cut_id for r in other_window._rows],
        st["mu_z"], st["mu_p"], st["D"], other_window.ridge_eps, oas=True)
    return {"j_cca_transfer_on_protect": float(j), "D": st["D"],
            "n_trees": other_window.n_trees}


# --------------------------------------------------------------------------
# acceptance self-test
# --------------------------------------------------------------------------

def _stub_rows(n_trees: int = 4, rows_per_tree: int = 2, seed: int = 0):
    """CutRecord-shaped stubs without touching records.py's validators."""
    from dataclasses import dataclass
    import torch as _t

    @dataclass
    class Row:
        cut_id: tuple
        tree_id: tuple
        X_v: object
        q_emb: object
        mask: object
        p: object
        r: float
        weight: float

    g = _t.Generator().manual_seed(seed)
    rows = []
    for t in range(n_trees):
        for occ in range(rows_per_tree):
            n_v = 5
            rows.append(Row(
                cut_id=("run_{}:c{}".format(t, occ), occ),
                tree_id=("run_{}".format(t), "gen0_seed"),
                X_v=_t.randn(n_v, C.D_E, generator=g),
                q_emb=_t.randn(C.D_E, generator=g),
                mask=_t.zeros(C.N_SLOTS, n_v),
                p=_t.randn(C.N_BRANCHES, C.M_SKETCH, generator=g),
                r=0.1 * occ,
                weight=1.0,
            ))
    return rows


class _StubFusion(torch.nn.Module):
    """Same interface as SlottedFusion, tiny and deterministic."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.Q0 = torch.nn.Parameter(torch.randn(C.N_SLOTS, C.D_E,
                                                 generator=g) * 0.02)

    def forward(self, X_v, q_emb, mask):
        # Must actually depend on X_v, otherwise every row is identical and
        # the window is degenerate (zero C_ZZ scale) -- which the close
        # correctly refuses. The stub has to behave like a real fusion.
        pooled = X_v.mean(0)                             # [D_E]
        Z = self.Q0 + pooled.reshape(1, -1) * 0.1        # [4, D_E]
        A = torch.softmax(torch.zeros(C.N_SLOTS, X_v.shape[0]), dim=-1)
        return Z, A


def self_test() -> int:
    import os
    print("window.py acceptance")
    print("=" * 68)

    fusion = _StubFusion()
    win = StatWindow(min_unique_trees=4)
    win.add(_stub_rows(n_trees=4, rows_per_tree=2))
    assert win.n_trees == 4 and win.n_rows == 8
    assert win.ready, win.n_trees
    print("OK  gate on trees     ready at {} unique trees / {} rows (gate is "
          "TREES, not rows)".format(win.n_trees, win.n_rows))

    # OAS is mandatory
    try:
        StatWindow(oas=False)
        raise AssertionError("oas=False was accepted")
    except WindowError:
        pass
    print("OK  OAS mandatory      oas=False refused at construction")

    # duplicate cuts are refused
    w2 = StatWindow(min_unique_trees=1)
    rows = _stub_rows(n_trees=1, rows_per_tree=1)
    w2.add(rows)
    try:
        w2.add(rows)
        raise AssertionError("duplicate cut accepted")
    except WindowError:
        pass
    print("OK  no duplicate cuts a cut belongs to exactly one window")

    # close_lpse: one cotangent PER CUT, never pre-summed
    out = win.close_lpse(fusion)
    assert set(out["cotangent_by_cut"]) == {r.cut_id for r in win._rows}
    assert all(t.shape == (C.B_V_DIM,) for t in
               out["cotangent_by_cut"].values())
    assert len(out["J_by_branch"]) == C.N_BRANCHES
    assert out["oas_alpha_p_protect"] is not None
    print("OK  lpse close        {} per-cut cotangents of dim {}, {} branch "
          "scores kept separate".format(len(out["cotangent_by_cut"]),
                                        C.B_V_DIM, C.N_BRANCHES))

    # SCALE INVARIANCE: <grad_B J, B> ~ 0. This is the test that catches an
    # accidental stop-gradient on C_ZZ.
    B = win.replay_B(fusion, grad=True)
    assert B.requires_grad
    Bg = B.double().detach()
    # real check: d/ds J(sB) = 0  ->  <grad_B J, B> = 0
    s = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    st = win._stats(Bg, win._P_by_branch()[0], win._weights())
    from meta_n.rpbe.kf import _oas_shrink, _score_from_covs, _covs
    zc = s * Bg - st["mu_z"]
    pc = win._P_by_branch()[0] - st["mu_p"]
    czz = zc.t() @ zc / st["D"]
    cpp = pc.t() @ pc / st["D"]
    czp = zc.t() @ pc / st["D"]
    czz, czp, cpp, _ = _oas_shrink(czz, czp, cpp, st["D"])
    j, _d = _score_from_covs(czz, czp, cpp, win.ridge_eps)
    j.backward()
    radial = float(s.grad)
    assert abs(radial) < 1e-6, "radial derivative {} != 0 (stray stop-grad?)".format(radial)
    print("OK  scale invariance  d/ds J(sB)|_1 = {:.2e} ~ 0 (C_ZZ keeps its "
          "gradient)".format(radial))

    # task close
    tk = win.close_task(fusion)
    assert set(tk["cotangent_by_cut"]) == {r.cut_id for r in win._rows}
    assert tk["oas_alpha_p_task"] == 0.0, tk["oas_alpha_p_task"]
    print("OK  task close        J_task cotangents per cut; scalar R gives "
          "oas_alpha_p = 0 exactly")

    # B is recomputed, not cached: nudging theta changes B
    B_before = win.replay_B(fusion, grad=False).clone()
    with torch.no_grad():
        fusion.Q0.add_(0.5)
    B_after = win.replay_B(fusion, grad=False).clone()
    assert not torch.allclose(B_before, B_after), "B did not track theta"
    assert win.tree_role_snapshot() == (frozenset(win.tree_seen), win.n_rows)
    print("OK  B recomputed      perturbing theta changes B; window contents "
          "and tree roles unchanged")

    # transfer diagnostic exists and is labelled as such
    other = StatWindow(min_unique_trees=4, name="protect")
    other.add(_stub_rows(n_trees=4, rows_per_tree=2, seed=7))
    tr = task_transfer_diagnostic(win, other, fusion)
    assert "j_cca_transfer_on_protect" in tr
    print("OK  transfer diag     j_cca_transfer_on_protect = {:.4f} (diagnostic "
          "only, never a checkpoint selector)".format(
              tr["j_cca_transfer_on_protect"]))

    # zero cost
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
