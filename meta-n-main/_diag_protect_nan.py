"""Locate the FIRST NaN in the protect-window LPSE close.

WHY THIS EXISTS. `build_records` reported, for the 3-row Structural6 protect
window:

    protect_closure = "LPSE close failed: {'failed': 'cholesky',
                       'info_z': 1, 'info_p': 1, 'scale_z': nan}"

"three rows is rank-deficient" does NOT explain that. OAS with alpha=1 gives
Sigma_hat = mu*I, which is the easiest possible Cholesky, and a rank-deficient
sample covariance should give `not positive definite`, not `NaN`. A NaN means
some quantity was already NaN BEFORE the factorization.

So this walks the exact stage order of `latent_z_adjoint` and prints, per stage,
where finiteness is first lost:

    z -> center -> weighted scatter -> /D  -> OAS shrink -> normalize -> chol

The hypothesis it is built to test is in `_stats`:

    D = W - W2_tree / W

For a SINGLE-TREE window W2_tree == W**2, so D == 0 exactly, and `czz = mzz / D`
is a division by zero -- inf where the scatter is non-zero, NaN where it is 0.
That would make the failure about the TREE COUNT, not the row count.

Run:  python -u _diag_protect_nan.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch_threads = os.environ.get("META_N_TORCH_THREADS", "1")

import torch                                                       # noqa: E402
torch.set_num_threads(int(torch_threads))
sys.path.insert(0, ".")

from meta_n.rpbe import kf as KF                                   # noqa: E402
from meta_n.rpbe.census import partition_tree_roles_v421            # noqa: E402
from meta_n.rpbe.fusion import SlottedFusion                       # noqa: E402
from meta_n.rpbe.records import load_jsonl                         # noqa: E402
from meta_n.rpbe.window import StatWindow                          # noqa: E402

RECORDS = sys.argv[1] if len(sys.argv) > 1 \
    else "/root/autodl-tmp/sri_s6_phasea/records.jsonl"


def fin(t):
    return "nan=%d inf=%d" % (int(torch.isnan(t).sum()), int(torch.isinf(t).sum()))


def f(x):
    try:
        return "%+.6e" % float(x)
    except Exception:                                              # noqa: BLE001
        return str(x)


def walk(label, rows, fusion):
    w = StatWindow(name=label)
    w.add(rows)
    print("=" * 74)
    print("%s   rows=%d  trees=%d" % (label, w.n_rows, w.n_trees))
    print("   trees: %s" % [t[0] for t in w.tree_ids])

    # ---- stage 0: the raw z rows -------------------------------------
    B64 = w.replay_B(fusion, grad=False).double()
    print("  [0] z_rows (replay_B)          : %s  shape=%s %s"
          % (fin(B64), tuple(B64.shape), B64.dtype))

    ww = w._weights()
    P = w._P_by_branch()[0]
    st = w._stats(B64, P, ww)
    W, W2, D = st["W"], st["W2_tree"], st["D"]
    print("  [1] weights / dof              : W=%.6f  W2_tree=%.6f"
          % (W, W2))
    print("      D = W - W2_tree/W          : %s   %s"
          % (f(D), "*** D == 0 ***" if abs(float(D)) < 1e-12 else ""))

    # ---- stage 2: per-dimension spread -------------------------------
    sd = B64.std(0)
    print("  [2] per-dim std                : min=%s median=%s max=%s"
          % (f(sd.min()), f(sd.median()), f(sd.max())))

    # ---- stage 3: weighted scatter (before /D) -----------------------
    zc = B64.clone() - st["mu_z"]
    pc = P.double() - st["mu_p"]
    sw = ww.double().sqrt().reshape(-1, 1)
    mzz = (zc * sw).t() @ (zc * sw)
    mzp = (zc * sw).t() @ (pc * sw)
    mpp = (pc * sw).t() @ (pc * sw)
    print("  [3] scatter mzz (before /D)    : %s  trace=%s"
          % (fin(mzz), f(torch.trace(mzz))))

    # ---- stage 4: THE SUSPECT -- czz = mzz / D -----------------------
    czz = mzz / D
    czp = mzp / D
    cpp = mpp / D
    print("  [4] empirical czz = mzz / D    : %s   <-- FIRST NaN HERE?"
          % fin(czz))
    print("      trace(czz)                 : %s" % f(torch.trace(czz)))
    dd = czz.diagonal()
    print("      diag(czz)                  : min=%s max=%s"
          % (f(dd.min()), f(dd.max())))

    # ---- stage 5: OAS ------------------------------------------------
    az = KF._oas_alpha(czz, D)
    mu_z = torch.trace(czz) / float(czz.shape[0])
    print("  [5] OAS alpha_z                : %s" % f(az))
    print("      OAS mu_z = tr/d             : %s" % f(mu_z))
    trc2 = (czz * czz).sum()
    tr2 = torch.trace(czz) ** 2
    p = float(czz.shape[0])
    beta = (1.0 - 2.0 / p) * trc2 + tr2
    delta = (D + 1.0 - 2.0 / p) * (trc2 - tr2 / p)
    print("      beta=%s  delta=%s  beta/delta=%s  D+1=%s"
          % (f(beta), f(delta), f(beta / torch.clamp(delta, min=1e-12)),
             f(D + 1.0)))

    # ---- stage 6: shrunk covariance ----------------------------------
    shr = (1.0 - az) * czz + az * mu_z * torch.eye(
        czz.shape[0], dtype=czz.dtype, device=czz.device)
    shr = 0.5 * (shr + shr.t())
    print("  [6] shrunk czz (OAS)           : %s" % fin(shr))
    try:
        ev = torch.linalg.eigvalsh(shr.double())
        print("      eigenvalues                : min=%s max=%s"
              % (f(ev.min()), f(ev.max())))
    except Exception as e:                                         # noqa: BLE001
        print("      eigenvalues                : FAILED %s" % e)
    print("      diagonal                   : min=%s max=%s"
          % (f(shr.diagonal().min()), f(shr.diagonal().max())))

    # ---- stage 7: what _score_from_covs actually reads ---------------
    sz = shr.diagonal().mean()
    print("  [7] scale_z = mean(diag)       : %s   <-- the reported value"
          % f(sz))
    eps = w.ridge_eps
    a = shr / sz
    a = 0.5 * (a + a.t()) + eps * torch.eye(
        a.shape[0], dtype=a.dtype, device=a.device)
    lz, info = torch.linalg.cholesky_ex(a)
    print("      ridge_eps                  : %s" % f(eps))
    print("      cholesky info_z            : %d   (0 == success)"
          % int(info.max().item()))
    return D


def main():
    recs = load_jsonl(RECORDS)
    trees = sorted({r.tree_id for r in recs})
    task_roots, protect_roots = partition_tree_roles_v421(trees)
    fusion = SlottedFusion()
    print("records=%d  trees=%d  task=%d protect=%d"
          % (len(recs), len(trees), len(task_roots), len(protect_roots)))
    walk("PROTECT", [r for r in recs if r.tree_id in set(protect_roots)],
         fusion)
    walk("TASK", [r for r in recs if r.tree_id in set(task_roots)], fusion)
    print("=" * 74)
    print("READING THE TRACE. If [4] is where the NaN first appears and")
    print("step [1] shows D == 0, the cause is the TREE COUNT (a one-tree")
    print("window has W2_tree == W**2 so D == 0), not the row count -- and")
    print("OAS never gets a chance because czz is already NaN before it.")


if __name__ == "__main__":
    main()
