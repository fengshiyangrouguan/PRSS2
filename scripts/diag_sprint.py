"""Sprint diagnostics (eighth-review section 5): 100-200 batches through
the REAL training entry point, reporting the acceptance checklist:

  reference_refreshes > 0, aux_batches > 0, reference_age == 1,
  stale_drops == 0, below_threshold_groups == 0,
  effective_grad_ratio p50 in [0.05, 0.30], p95 < 1,
  plus cos(task, kf), rho_radial, W_batch/W_ref and adjoint norms.

Runs one train_epoch capped at --max-batches (no epoch drain), so the
run costs a few minutes.  Cloud usage:

  /root/miniconda3/bin/python -m scripts.diag_sprint \
    -d wikipedia --data-dir old/processed_tgn_data \
    --bs 200 --n-layer 3 --n-degree 5 --kf-lambda 1e-3 \
    --kf-group-batches 40 --max-batches 160 --seed 0
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, 'src')
import scripts.train_jodie as tj
from rpbe.monitoring import MonitorWriter
from rpbe.training.jodie_loop import JodieNodeClassificationLoop


def grad_diag_fn(loop, task_loss, auxiliary, step):
    """Per-batch gradient statistics on the representation parameters.

    Injected into the loop so the training module itself stays free of
    explicit autograd.grad calls (source contract).  ``auxiliary``
    already carries lambda and the rank coefficients.
    """
    g_task = torch.autograd.grad(
        task_loss, loop.repr_params, retain_graph=True, allow_unused=True)
    g_kf = torch.autograd.grad(
        auxiliary, loop.repr_params, retain_graph=True, allow_unused=True)
    tn = float(sum(t.norm() for t in g_task if t is not None))
    kn = float(sum(t.norm() for t in g_kf if t is not None))
    cos = float("nan")
    if tn > 0 and kn > 0:
        dot = sum(float((a * b).sum())
                  for a, b in zip(g_task, g_kf)
                  if a is not None and b is not None)
        cos = dot / (tn * kn)
    rho = dict(getattr(loop.kf_window, "_last_rho", {}))
    wb_wr = {}
    age = {}
    adj_norm = {}
    for tau in loop._tau_coeff:
        ref = loop.kf_window._reference.get(tau)
        if ref is not None:
            wb = loop.kf_window._last_batch_weight.get(tau, 0.0)
            wb_wr[tau] = wb / ref["W"] if ref["W"] > 0 else float("nan")
            adj_norm[tau] = float(sum(
                a.norm() for a in ref["adjoints"].values()))
        age[tau] = loop.kf_window.reference_age(tau)
    return {
        "step": int(step),
        "g_task_norm": tn,
        "g_kf_norm": kn,
        "ratio_raw": kn / max(tn, 1e-12),
        "ratio_effective": kn / max(tn, 1e-12),
        "cos_task_kf": cos,
        "rho_radial": rho,
        "reference_age": age,
        "W_batch_over_W_ref": wb_wr,
        "adjoint_norm": adj_norm,
    }


def main():
    # The full train_jodie CLI (data/checkpoint/model/KF switches) plus
    # --max-batches; defaults match the sprint call in the docstring.
    args = tj.parse_args()
    tj.seed_all(args.seed)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available()
        else "cpu")
    dataset = tj.JodieDataset(args.data, data_dir=args.data_dir,
                              use_validation=True)
    full, train, val, test = dataset.splits()
    components = tj.build_components(args, device, dataset)
    tgn = components["tgn"]

    grad_diag = {"rows": [], "fn": grad_diag_fn}
    loop = JodieNodeClassificationLoop(
        tgn=tgn, decoder=components["decoder"],
        repr_optimizer=components["repr_optimizer"],
        head_optimizer=components["head_optimizer"],
        device=device, batch_size=args.bs, n_neighbors=args.n_degree,
        grad_clip=args.grad_clip, monitor=MonitorWriter(
            tj.Path("outputs/diag_sprint")),
        seed=args.seed, finetune_host=args.finetune_host,
        adapter=components["adapter"], cut_builder=components["cut_builder"],
        fixed_maps=components["fixed_maps"], rpbe_cfg=components["rpbe_cfg"],
        trace_roots=args.trace_roots, trace_mode=args.trace_mode,
        grad_diag=grad_diag)

    row = loop.train_epoch(0, 0, train, max_batches=args.max_batches)
    kf = row.get("kf") or {}
    rows = grad_diag["rows"]
    # Diagnostics need the KF component; refuse to silently report a
    # vanilla run as a failed sprint.
    if not loop.kf_on:
        print("SPRINT ABORTED: --rpbe was not passed (kf_on=False); "
              "the run was plain vanilla.")
        return
    print("\n===== SPRINT ACCEPTANCE =====")
    print("batches run:", row["global_step"])
    print("aux_batches:", kf.get("aux_batches", 0))
    print("reference_refreshes:", kf.get("reference_refreshes", 0))
    print("stale_drops:", kf.get("stale_drops", 0))
    print("below_threshold_groups:", kf.get("below_threshold_groups", 0))
    print("pending_trees:", kf.get("pending_trees", {}),
          "threshold:", kf.get("threshold", {}))
    print("group_batches:", kf.get("group_batches", None))

    ratio = np.array([r["ratio_effective"] for r in rows]) if rows \
        else np.array([])
    cos = np.array([r["cos_task_kf"] for r in rows]) if rows \
        else np.array([])
    rho_all = [v for r in rows for v in r["rho_radial"].values()]
    age_all = [v for r in rows for v in r["reference_age"].values()
               if v is not None]
    if rows:
        print("effective_grad_ratio p50: {:.5f}  p95: {:.5f}  max: {:.5f}"
              .format(float(np.percentile(ratio, 50)),
                      float(np.percentile(ratio, 95)),
                      float(ratio.max())))
        # Lambda-free ratio (rank-normalized, scheme-B window scale): the
        # quantity the lambda sweep must calibrate into [0.05, 0.30].
        raw_ratio = np.array([r["g_kf_norm"] / max(r["g_task_norm"], 1e-12)
                              / max(args.kf_lambda, 1e-12) for r in rows])
        print("lambda-free rank-normalized ratio p50: {:.5f}"
              .format(float(np.percentile(raw_ratio, 50))))
        print("cos(task, kf): mean {:.4f}  min {:.4f}".format(
            float(np.mean(cos)), float(cos.min())))
        print("rho_radial: mean |rho| {:.4f}  max |rho| {:.4f}".format(
            float(np.mean(np.abs(rho_all))),
            float(np.max(np.abs(rho_all))) if rho_all else float("nan")))
        print("reference_age values:", sorted(set(age_all)))
        wb = [v for r in rows for v in r["W_batch_over_W_ref"].values()]
        if wb:
            print("W_batch/W_ref p50: {:.4f}".format(
                float(np.percentile(wb, 50))))
        an = [v for r in rows for v in r["adjoint_norm"].values()]
        if an:
            print("adjoint_norm p50: {:.3e}".format(
                float(np.percentile(an, 50))))
    else:
        print("NO diagnostic rows: the KF reference was never active "
              "(all groups below threshold?)")

    print("\n===== CHECKLIST =====")
    ok = True
    checks = [
        ("reference_refreshes > 0",
         kf.get("reference_refreshes", 0) > 0),
        ("aux_batches > 0", kf.get("aux_batches", 0) > 0),
        ("reference_age always 1",
         bool(age_all) and set(age_all) == {1}),
        ("stale_drops == 0", kf.get("stale_drops", 0) == 0),
        ("below_threshold_groups == 0",
         kf.get("below_threshold_groups", 0) == 0),
    ]
    if len(ratio):
        p50 = float(np.percentile(ratio, 50))
        p95 = float(np.percentile(ratio, 95))
        checks.append(("effective_grad_ratio p50 in [0.05, 0.30]",
                       0.05 <= p50 <= 0.30))
        checks.append(("effective_grad_ratio p95 < 1", p95 < 1.0))
    else:
        p50 = p95 = float("nan")
        checks.append(("effective_grad_ratio p50 in [0.05, 0.30]", False))
        checks.append(("effective_grad_ratio p95 < 1", False))
    for name, passed in checks:
        ok = ok and passed
        print(("PASS " if passed else "FAIL ") + name)
    print("SPRINT " + ("PASSED" if ok else "FAILED"))


if __name__ == "__main__":
    main()
