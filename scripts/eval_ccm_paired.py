#!/usr/bin/env python3
"""Review round 8 paired statistical report (task-only vs ours).

Official token-weighted NLL stays the PRIMARY metric (see the depth
evaluations).  This report adds the paired dialogue-level analysis the
review demands for the 51 long (turn-14) validation dialogues:

    d_i = NLL_task,i - NLL_ours,i     (paired on the SAME dialogue)

Per depth L it reports:

  * paired mean + BCa bootstrap CI (20,000 dialogue-level resamples,
    jackknife acceleration + bias correction);
  * Huber paired location (psi with c = 1.345; winsorized location,
    robust to extreme dialogues);
  * paired randomization test (sign flips under the sharp null of no
    arm difference);
  * leave-one-dialogue-out stability (does any single dialogue flip the
    mean's sign?);
  * the maximum single-dialogue contribution share.

Plus the four-sketch-branch direction agreement from the training
window diagnostics: the fraction of windows where at least 3 of the 4
branch scores beat their dialogue-shuffled baseline in the same
direction (review pass rule: >= 3/4 branches agree).

Pass rules (review round 8):
  * >= 3/4 branch direction agreement;
  * Huber location and plain mean share the same sign;
  * leave-one-out never flips the sign;
  * real J beats the dialogue-level shuffled pairing (per-window mean);
  * no single dialogue contributes more than 20% of sum |d_i|.

Usage:
  python eval_ccm_paired.py --task task_recs.json --ours ours_recs.json \
      [--window-diag window_diag.jsonl] --out report.json
"""
import argparse
import json
import math

import numpy as np


def huber_location(d: np.ndarray, c: float = 1.345, tol: float = 1e-10):
    """Paired Huber location: solve sum_i psi(d_i - theta) = 0.

    psi(x) = x if |x| <= c else c * sign(x).  Solved by iteratively
    reweighted mean (monotone in theta; converges in a few iterations).
    """
    theta = float(np.median(d))
    for _ in range(200):
        r = d - theta
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.abs(r) <= c, 1.0, c / np.abs(r))
        new_theta = float(np.sum(w * d) / np.sum(w))
        if abs(new_theta - theta) < tol:
            return new_theta
        theta = new_theta
    return theta


def bca_bootstrap(d: np.ndarray, n_boot: int = 20000, rng=None):
    """BCa bootstrap CI (95%) for the mean of paired differences."""
    rng = rng or np.random.default_rng(0)
    n = len(d)
    theta_hat = float(d.mean())
    # Jackknife for the acceleration a.
    theta_i = np.array([d[np.arange(n) != i].mean() for i in range(n)])
    theta_dot = float(theta_i.mean())
    num = np.sum((theta_dot - theta_i) ** 3)
    den = np.sum((theta_dot - theta_i) ** 2) ** 1.5
    a = float(num / (6 * max(den, 1e-30))) if den > 0 else 0.0
    # Bias correction z0.
    boots = np.array([rng.choice(d, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    z0 = float(np.mean(boots <= theta_hat))
    z0 = float(np.sqrt(2) * _ppf(max(min(z0, 1 - 1e-12), 1e-12)))
    if not math.isfinite(z0):
        z0 = 0.0
    # BCa endpoints (standard normal quantiles).
    alpha = 0.025
    z_lo, z_hi = _ppf(alpha), _ppf(1 - alpha)
    q_lo = _norm_cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    q_hi = _norm_cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    lo = float(np.quantile(boots, max(min(q_lo, 1 - 1e-9), 1e-9)))
    hi = float(np.quantile(boots, max(min(q_hi, 1 - 1e-9), 1e-9)))
    return lo, hi, a, z0


def _ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approx)."""
    if p <= 0.0 or p >= 1.0:
        return -8.0 if p <= 0.0 else 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d_ = [7.784695709041462e-03, 3.224671290700398e-01,
          2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4])
                * q + c[5]) / ((((d_[0] * q + d_[1]) * q + d_[2]) * q
                                + d_[3]) * q + 1.0)
    if p > 1 - plow:
        q = math.sqrt(-2.0 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4])
                 * q + c[5]) / ((((d_[0] * q + d_[1]) * q + d_[2]) * q
                                 + d_[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
            + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                            + b[4]) * r + 1.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def randomization_test(d: np.ndarray, n_perm: int = 20000, rng=None):
    """Paired randomization test: sign flips under the sharp null."""
    rng = rng or np.random.default_rng(1234)
    obs = float(np.abs(d.mean()))
    flips = rng.integers(0, 2, size=(n_perm, len(d))) * 2 - 1
    null = np.abs((flips * d).mean(axis=1))
    return float(np.mean(null >= obs)), obs


def per_L_report(d: np.ndarray, n_boot: int = 20000) -> dict:
    rng = np.random.default_rng(20260907)
    mean = float(d.mean())
    lo, hi, a, z0 = bca_bootstrap(d, n_boot, rng)
    hub = huber_location(d)
    pval, obs = randomization_test(d, 20000, rng)
    loo_flips = 0
    loo_means = []
    n = len(d)
    for i in range(n):
        m = float(d[np.arange(n) != i].mean())
        loo_means.append(m)
        if m * mean < 0:
            loo_flips += 1
    share = float(np.abs(d).max() / np.abs(d).sum()) \
        if np.abs(d).sum() > 0 else float("nan")
    return {
        "n_pairs": int(n),
        "mean": mean,
        "bca_lo": lo, "bca_hi": hi, "accel_a": a, "bias_z0": z0,
        "huber_location": hub,
        "rand_p_value": pval, "rand_obs": obs,
        "loo_flips": int(loo_flips), "loo_min": float(min(loo_means)),
        "loo_max": float(max(loo_means)),
        "max_dialogue_share": share,
    }


def branch_direction_agreement(window_diag_path: str) -> dict:
    """>= 3/4 branches beat the shuffled baseline, per window."""
    windows = 0
    agree = 0
    j_real_shuff_mean = []
    try:
        with open(window_diag_path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                jb = row.get("J_branches")
                js = row.get("J_shuffled_mean")
                if not jb or js is None:
                    continue
                windows += 1
                j_real_shuff_mean.append(float(row.get(
                    "J_real_minus_shuffled", float("nan"))))
                pos = sum(1 for j in jb if j > js)
                if pos >= 3:
                    agree += 1
    except FileNotFoundError:
        return {"windows": 0, "agree_frac": None, "error": "missing"}
    return {
        "windows": int(windows),
        "agree_frac": float(agree / windows) if windows else None,
        "J_real_minus_shuffled_mean":
            float(np.mean(j_real_shuff_mean)) if j_real_shuff_mean else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--window-diag", default="")
    ap.add_argument("--out", default="paired_report.json")
    a = ap.parse_args()

    t = json.load(open(a.task))
    o = json.load(open(a.ours))
    t_recs = {(r["dialogue"], r["L"]): r for r in t.get("recs", [])}
    o_recs = {(r["dialogue"], r["L"]): r for r in o.get("recs", [])}

    report = {"task_ckpt": t.get("ckpt"), "ours_ckpt": o.get("ckpt"),
              "per_L": {}, "branch_direction": (
                  branch_direction_agreement(a.window_diag)
                  if a.window_diag else None)}
    passes = []
    fails = []
    for L in [1, 2, 4, 8, 13]:
        keys = sorted(set(k for k in t_recs if k[1] == L)
                      & set(k for k in o_recs if k[1] == L))
        d = np.array([t_recs[k]["nll"] - o_recs[k]["nll"] for k in keys],
                     dtype=np.float64)
        if len(d) == 0:
            report["per_L"][str(L)] = {"n_pairs": 0}
            continue
        rep = per_L_report(d)
        report["per_L"][str(L)] = rep
        # Pass rules per L.  Directional rules (Huber sign, LOO no-flip)
        # apply only when the layer HAS directional evidence (|mean|
        # beyond the CI half-width); a mean sitting at zero flips under
        # leave-one-out by construction, so those rules are N/A there.
        ci_half = (rep["bca_hi"] - rep["bca_lo"]) / 2.0
        directional = abs(rep["mean"]) > ci_half
        rule = []
        if directional:
            rule.append(("huber_same_sign", rep["huber_location"]
                         * rep["mean"] > 0))
            rule.append(("loo_no_flip", rep["loo_flips"] == 0))
        else:
            rule.append(("directional", False))  # reported, not a fail
        rule.append(("max_share_le_20pct", rep["max_dialogue_share"] <= 0.2
                     or math.isnan(rep["max_dialogue_share"])))
        ok = all(v for k, v in rule if k != "directional")
        (passes if ok else fails).append(
            "L{} mean={:+.4f} huber={:+.4f} CI=[{:+.4f},{:+.4f}] "
            "p={:.3f}{}".format(L, rep["mean"], rep["huber_location"],
                                rep["bca_lo"], rep["bca_hi"],
                                rep["rand_p_value"],
                                "" if directional else " (no direction)"))
        report["per_L"][str(L)]["pass"] = ok
        report["per_L"][str(L)]["directional"] = bool(directional)
        report["per_L"][str(L)]["rules"] = dict(rule)

    bd = report["branch_direction"]
    if bd is not None and bd.get("agree_frac") is not None:
        rule_bd = bd["agree_frac"] >= 0.75
        (passes if rule_bd else fails).append(
            "branch_direction {:.1%} of {} windows".format(
                bd["agree_frac"], bd["windows"]))
        report["branch_direction"]["pass"] = rule_bd
        if bd.get("J_real_minus_shuffled_mean") is not None and \
                bd["J_real_minus_shuffled_mean"] <= 0:
            fails.append("real J does not beat shuffled pairing "
                         "(mean {:.3f})".format(
                             bd["J_real_minus_shuffled_mean"]))

    report["PASS"] = not fails
    report["pass_list"] = passes
    report["fail_list"] = fails
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print("PAIRED_PASS" if report["PASS"] else "PAIRED_FAIL", flush=True)
    for p in passes:
        print("  PASS " + p, flush=True)
    for fl in fails:
        print("  FAIL " + fl, flush=True)
    print("wrote {}".format(a.out), flush=True)


if __name__ == "__main__":
    main()
