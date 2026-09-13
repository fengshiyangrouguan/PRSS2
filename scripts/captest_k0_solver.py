"""κ=0 solver capability test (offline, synthetic dense-violation regime).

Replicates the production active-set QP solver VERBATIM (same dual objective,
same row-normalised cosine Gram, same 1e-6 certificate tolerance) and asks a
single question:

    when half of the directions violate the κ=0 boundary, is the certificate
    failure a SOLVER-CAPACITY limit (more budget/rounds/batch fixes it) or
    evidence that the constraint system cannot be certified at all?

Legacy config          : 3 rounds, +1 row/round, FISTA ladder 2000..32000
Strengthened (κ=0)     : 12 rounds, +64 rows/round, ladder 2000..512000

Anchors taken from the real k0 s0 run on UCI (N≈3528 directions, 50.4 %
violating, cos_min ≈ -0.22, cos_p5 ≈ -0.085) — the synthetic problem matches
that density and cosine scale, and additionally probes a harsher regime.

Pure torch (no numpy): the host Anaconda numpy install is broken by a 2.x
residue, and torch is what production actually uses.

Run:  python scripts/captest_k0_solver.py [--n 1800] [--p 1200] [--seed 0]
"""
import argparse
import math
import time

import torch


def make_problem(n, p, seed, scale=1.0):
    """Random directions + task vector → (H_bar, t) with the real cos scale."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(p, generator=g)
    t = t / t.norm()
    A = torch.randn(n, p, generator=g) * scale
    gn = A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return A / gn, t


def fista_solve(H, t, *, kappa, ladder, rounds, batch, tol=1e-6,
                record_every=None):
    """Production solver logic (verbatim), returning the certificate trace."""
    dev = H.device
    n = int(H.shape[0])
    nt = float(t.norm())
    K = H @ H.t()
    cvec = (-kappa * nt * torch.ones(n, device=dev)) - (H @ t)

    # spectral step size: 30 power iterations, exactly as production
    with torch.no_grad():
        v = torch.randn(n, device=dev)
        v = v / (v.norm() + 1e-30)
        lam_max = 1.0
        for _ in range(30):
            v = K @ v
            nv = float(v.norm())
            if nv <= 1e-30:
                break
            v = v / nv
            lam_max = nv
    eta = 1.0 / (lam_max + 1e-12)

    cos = H @ t
    active_keys = [int(i) for i in
                   torch.nonzero(cos < -kappa).reshape(-1).tolist()]
    trace = [{"round": -1, "viol_max": None, "active_keys": len(active_keys),
              "fista_iter": 0, "phase": "init"}]

    def solve_active(idx):
        Hi = H[idx]
        Ki = Hi @ Hi.t()
        ci = cvec[idx]
        lam = torch.zeros(len(idx), device=dev)
        d = None
        viol_max = float("inf")
        n_iter = 0
        for budget in ladder:
            y = lam
            tk = 1.0
            done = 0
            while done < budget:
                step = budget - done if record_every is None else \
                    min(record_every, budget - done)
                for _ in range(step):
                    lam_new = torch.clamp(y + eta * (ci - Ki @ y), min=0.0)
                    tk_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
                    y = lam_new + ((tk - 1.0) / tk_new) * (lam_new - lam)
                    lam = lam_new
                    tk = tk_new
                done += step
                n_iter += step
                d = t + Hi.t() @ lam
                viol = (-kappa * nt) - (Hi @ d)
                rel = viol / (nt + 1e-30)
                viol_max = float(rel.max()) if rel.numel() else 0.0
                if record_every is not None:
                    trace.append({"round": len(trace), "viol_max": viol_max,
                                  "active_keys": len(idx), "fista_iter": n_iter,
                                  "phase": "fista"})
                if viol_max <= tol:
                    break
            if viol_max <= tol:
                break
        return d, viol_max, n_iter

    total_iter = 0
    viol_max = float("inf")
    d = None
    t_start = time.time()
    for r in range(rounds):
        d, viol_max, n_it = solve_active(active_keys)
        total_iter += n_it
        trace.append({"round": r, "viol_max": viol_max,
                      "active_keys": len(active_keys), "fista_iter": total_iter,
                      "phase": "solve"})
        if viol_max > tol:
            break
        # full re-scan over EVERY row (production: CPU chunks of 200, fp64)
        d_cpu = d.detach().cpu().double()
        Hd = H.double()
        viol = (-kappa * nt) - (Hd @ d_cpu) / (Hd.norm(dim=1) * nt + 1e-30)
        bad = torch.nonzero(viol > tol).reshape(-1).tolist()
        hits = [(float(viol[i]), int(i)) for i in bad]
        if not hits:
            viol_max = 0.0
            break
        hits.sort(key=lambda x: -x[0])
        added = 0
        for _v, j in hits[:batch]:
            if j not in active_keys:
                active_keys.append(j)
                added += 1
        if added == 0:
            break
        viol_max = 1.0
    return {"passed": viol_max <= tol, "viol_max": viol_max,
            "rounds": len([x for x in trace if x["phase"] == "solve"]),
            "final_active": len(active_keys), "fista_iter": total_iter,
            "seconds": time.time() - t_start, "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1800)
    ap.add_argument("--p", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--record-every", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dt = torch.float32 if args.dtype == "float32" else torch.float64
    H, t = make_problem(args.n, args.p, args.seed, args.scale)
    H, t = H.to(dt), t.to(dt)
    cos = H @ t
    print("problem: N=%d P=%d dtype=%s | cos mean=%+.4f min=%+.4f "
          "frac<0=%.3f frac<-0.05=%.3f"
          % (args.n, args.p, args.dtype, float(cos.mean()), float(cos.min()),
             float((cos < 0).float().mean()),
             float((cos < -0.05).float().mean())), flush=True)

    rec = args.record_every if args.record_every > 0 else None
    for name, cfg in (
        ("LEGACY   ", dict(ladder=(2000, 8000, 32000), rounds=3, batch=1)),
        ("STRONG-k0", dict(ladder=(2000, 8000, 32000, 128000, 512000),
                           rounds=12, batch=64)),
    ):
        if args.only and args.only not in name.lower():
            continue
        r = fista_solve(H, t, kappa=0.0, tol=1e-6, record_every=rec, **cfg)
        print("[%s] passed=%-5s viol_max=%.3e rounds=%d active=%d "
              "fista=%d  %.1fs"
              % (name, r["passed"], r["viol_max"], r["rounds"],
                 r["final_active"], r["fista_iter"], r["seconds"]), flush=True)
        if rec:
            step = max(1, len(r["trace"]) // 12)
            for x in r["trace"][::step]:
                vm = "    n/a " if x["viol_max"] is None else \
                    "%.3e" % x["viol_max"]
                print("     r=%3d %-5s viol=%s keys=%4d it=%7d"
                      % (x["round"], x["phase"], vm,
                         x["active_keys"], x["fista_iter"]), flush=True)


if __name__ == "__main__":
    main()
