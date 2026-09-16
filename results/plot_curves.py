"""Plot the recovered Stage8 curves.

Reads val_curves.csv (val action loss, one row per [run, step]) and the
rollout table transcribed from STAGE8_RESULTS.md, and writes three figures:

    val_curves.png       -- val action loss, all arms
    avg_host_curve.png   -- the host on its own (the arm we start from)
    rollout_curve.png    -- tiered rollout, ours vs the host it starts from

The val curves are RECOVERED points, not a complete log: a missing step is a
gap in the recovery, never a flat segment.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# (csv run name, label, colour, linewidth, alpha, zorder)
MAIN = [
    ("avg_s42_stage5_official",
     "avg host (Stage5, official)", "#111111", 2.4, 1.0, 5),
    ("gamma-rpbe_s42_stage5_official",
     "gamma-rpbe (Stage5)", "#1f77b4", 1.8, 0.9, 4),
    ("ours_gamma-only_ours_s8g(proposal-space)",
     "ours: gamma-only + proposal-space", "#d62728", 2.4, 1.0, 6),
    ("gamma-only_s42_s8f(GAMMA_FROZEN)",
     "gamma FROZEN (control that is not a run)",
     "#7f7f7f", 1.8, 0.9, 3),
]
FAINT = [
    "ours_gamma-only_ours_s8h(continuation)",
    "gamma-lora_s42_s8f",
    "gamma-rpbe_s42_r1(gamma-only,raw_proj)",
    "gamma-lora_s42_r1",
    "s5r(gamma-only,diffusion,earlier_stage)",
]

# absolute step, weighted_success, 3 cycles
ROLLOUT = [
    (8000, 15.0, 1), (10000, 20.0, 0), (13000, 20.0, 0), (15000, 21.7, 0),
    (17000, 25.0, 2), (19000, 23.3, 1), (21000, 18.3, 0),
]
HOST = 23.3
HOST_SIGMA = 9.3          # binom std at n=20, in percentage points
BUDGET = 15000


def load():
    """Run names may themselves contain commas, so split from the right."""
    runs = {}
    with open(os.path.join(HERE, "val_curves.csv"), newline="") as f:
        rows = [ln.rstrip("\r\n") for ln in f if ln.strip()]
    for ln in rows[1:]:
        name, step, val = ln.rsplit(",", 2)
        runs.setdefault(name.strip(), []).append((int(step), float(val)))
    for v in runs.values():
        v.sort()
    return runs


def plot_val(runs, path):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for name, lab, c, lw, al, z in MAIN:
        if name not in runs:
            continue
        xs = [p[0] for p in runs[name]]
        ys = [p[1] for p in runs[name]]
        ax.plot(xs, ys, "-o", color=c, lw=lw, ms=4.5, alpha=al, label=lab,
                zorder=z)
    for name in FAINT:
        if name not in runs:
            continue
        xs = [p[0] for p in runs[name]]
        ys = [p[1] for p in runs[name]]
        ax.plot(xs, ys, "--", color="#999999", lw=1.0, alpha=0.7, zorder=1,
                dashes=(4, 3))
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(4, -8),
                    textcoords="offset points", fontsize=6.5, color="#888888")

    ax.set_xlabel("optimizer step")
    ax.set_ylabel("val action loss  (3 fixed demos)")
    ax.set_title("RPBE Stage8 -- validation action loss, recovered points\n"
                 "a gap is a missing recovery point, not a flat segment",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.set_ylim(0.062, 0.10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print("wrote", path)


def plot_avg_only(runs, path):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    name = "avg_s42_stage5_official"
    xs = [p[0] for p in runs[name]]
    ys = [p[1] for p in runs[name]]
    ax.plot(xs, ys, "-o", color="#111111", lw=2.4, ms=5)
    ax.annotate(f"{ys[0]:.4f}", (xs[0], ys[0]), xytext=(6, 6),
                textcoords="offset points", fontsize=9)
    ax.annotate(f"{ys[-1]:.4f}  (step {xs[-1]})", (xs[-1], ys[-1]),
                xytext=(-10, 10), textcoords="offset points", fontsize=9)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("val action loss  (3 fixed demos)")
    ax.set_title("avg host -- this is the arm our run starts from\n"
                 "host checkpoint rollout 23.3% (same 20-demo protocol)",
                 fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_ylim(0.062, 0.10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print("wrote", path)


def plot_rollout(path):
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    xs = [r[0] for r in ROLLOUT]
    ys = [r[1] for r in ROLLOUT]
    ax.axhspan(HOST - HOST_SIGMA, HOST + HOST_SIGMA, color="#111111",
               alpha=0.07, zorder=0)
    ax.axhline(HOST, color="#111111", lw=2.0, zorder=2)
    ax.annotate("host, untrained-by-Gamma  23.3%", (5600, HOST),
                xytext=(0, 6), textcoords="offset points", fontsize=9)
    ax.annotate("+/-1 sigma (n=20 -> sigma ~ 9.3%)", (5600, HOST + HOST_SIGMA),
                xytext=(0, 4), textcoords="offset points", fontsize=7.5,
                color="#666666")
    ax.axvline(BUDGET, color="#d62728", lw=1.2, ls=":", zorder=1)
    ax.annotate("pre-registered\nbudget (15000)", (BUDGET, 11.4),
                xytext=(-4, 0), textcoords="offset points", fontsize=8,
                color="#d62728", ha="right")
    ax.axvspan(BUDGET, 21400, color="#d62728", alpha=0.05, zorder=0)
    ax.annotate("weight-initialised continuation", (18100, 10.6), fontsize=8,
                color="#d62728", ha="center")

    ax.plot(xs, ys, "-o", color="#d62728", lw=2.2, ms=6, zorder=4)
    for x, y, t3 in ROLLOUT:
        if t3:
            ax.annotate(f"{t3}x3-cycle", (x, y), xytext=(0, 9),
                        textcoords="offset points", fontsize=8, ha="center")
    ax.annotate("25.0% (peak, diagnostic only)", (17000, 25.0),
                xytext=(-6, 12), textcoords="offset points", fontsize=8.5,
                ha="right")
    ax.annotate("21.7%", (15000, 21.7), xytext=(-6, -12),
                textcoords="offset points", fontsize=8.5, ha="right")
    ax.annotate("18.3%", (21000, 18.3), xytext=(-8, -4),
                textcoords="offset points", fontsize=8.5, ha="right")

    ax.set_xlabel("optimizer step")
    ax.set_ylabel("weighted_success  (20 demos, exec=8, maxsteps=370)")
    ax.set_title("RPBE Stage8 -- rollout, and the overfitting that stopped it\n"
                 "every point is a separate tiered_eval process; the endpoint "
                 "is reported, the peak is not", fontsize=11)
    ax.set_xlim(4000, 23000)
    ax.set_ylim(8, 34)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print("wrote", path)


def plot_host_rollout(path):
    """avg host rollout: the reported protocol, plus the seed sweep."""
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    labels, vals, cols, hatches = [], [], [], []

    # protocol used for every number in STAGE8_RESULTS.md
    for i, lab in enumerate(["run 1", "run 2\n(bit-identical)"]):
        labels.append(lab)
        vals.append(23.3)
        cols.append("#111111")
        hatches.append("")
    # the pre-existing 5-seed sweep: demo 81-100, exec=4
    for seed, v in [(42, 10.5), (7, 12.3), (123, 24.6), (99, 15.8), (11, None)]:
        labels.append(f"seed {seed}")
        vals.append(v)
        cols.append("#1f77b4")
        hatches.append("" if v is not None else "//")

    xs = list(range(len(labels)))
    for i, (x, v, c, h) in enumerate(zip(xs, vals, cols, hatches)):
        if v is None:
            ax.bar(x, 30.0, color="white", edgecolor="#1f77b4", hatch=h,
                   linewidth=1.2, zorder=3)
            ax.annotate("not recovered", (x, 30.0), xytext=(0, -26),
                        textcoords="offset points", fontsize=8, ha="center",
                        color="#1f77b4", rotation=90)
        else:
            ax.bar(x, v, color=c, zorder=3)
            ax.annotate(f"{v:.1f}%", (x, v), xytext=(0, 3),
                        textcoords="offset points", fontsize=9, ha="center")

    ax.axhline(HOST, color="#d62728", lw=1.4, ls="--", zorder=2)
    ax.axhspan(HOST - HOST_SIGMA, HOST + HOST_SIGMA, color="#d62728",
               alpha=0.07, zorder=0)
    ax.annotate("reported host baseline 23.3%\n(+/-1 sigma, n=20 -> 9.3%)",
                (0.4, HOST), xytext=(0, 5), textcoords="offset points",
                fontsize=8.5, color="#d62728")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("weighted_success  (20 demos, maxsteps=370)")
    ax.set_title("avg host `avg_s42_stage5_official/best.pt` -- its test results\n"
                 "black: demo_1..20, exec=8, seed 42 (the protocol used "
                 "throughout)   blue: demo_81..100, exec=4",
                 fontsize=10.5)
    ax.set_ylim(0, 36)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print("wrote", path)


def plot_avg_vs_ours(runs, path):
    """avg vs ours only -- the pair, with the recovery boundary drawn."""
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    a = runs["avg_s42_stage5_official"]
    o = runs["ours_gamma-only_ours_s8g(proposal-space)"]

    ax.plot([p[0] for p in a], [p[1] for p in a], "-o", color="#111111",
            lw=2.4, ms=6, label="avg host (Stage5, official)  -- 18k endpoint")
    ax.plot([p[0] for p in o], [p[1] for p in o], "-o", color="#d62728",
            lw=2.4, ms=6,
            label="ours (gamma-only + proposal-space)  -- 15k endpoint")

    # the frozen-control band: it is why val cannot select a checkpoint
    ax.axhspan(0.0684, 0.0695, color="#7f7f7f", alpha=0.13, zorder=0)
    ax.annotate("gamma-FROZEN control band 0.0684-0.0695  (ours sits inside it)",
                (600, 0.0687), xytext=(0, 3), textcoords="offset points",
                fontsize=8, color="#555555")

    # the recovery boundary: nothing of ours survives past 10000
    ax.axvline(10000, color="#d62728", lw=1.2, ls=":", zorder=1)
    ax.axvspan(10000, 18800, color="#d62728", alpha=0.05, zorder=0)
    ax.annotate("no `ours` val line survived past 10000:\n"
                "the 10500-15000 evals were only ever written to the box",
                (10450, 0.0955), fontsize=8, color="#d62728", va="top")

    # nearest surviving measurement to ours' endpoint: s8h step 500 (absolute
    # 15500), i.e. ours@15000 weights after 500 more continuation steps
    ax.plot([15500], [0.0689], "o", mfc="white", mec="#d62728", mew=2.0,
            ms=8, zorder=6)
    ax.annotate("s8h @ 500 (absolute 15500) = 0.0689\n"
                "ours@15000 weights + 500 continuation steps",
                (15500, 0.0689), xytext=(-8, -34), textcoords="offset points",
                fontsize=8, color="#d62728", ha="right")

    for x, y, lab in [(18000, 0.0694, "0.0694"), (10000, 0.0687, "0.0687"),
                      (500, 0.0694, "0.0694")]:
        ax.annotate(lab, (x, y), xytext=(4, 7), textcoords="offset points",
                    fontsize=8.5)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("val action loss  (3 fixed demos)")
    ax.set_title("avg vs ours -- the only two curves the result rests on\n"
                 "ours starts where the host ends: same level, no separation",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.set_ylim(0.062, 0.10)
    ax.set_xlim(0, 19000)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print("wrote", path)


def plot_alignment(path):
    """Where ours' best val and best rollout sit, on one absolute-step axis.

    Two axes because the two metrics have different units.  Left = val action
    loss (blue), right = weighted_success (grey).  s8g points are filled, s8h
    continuation points hollow: the two segments are weight-initialised, not a
    bit-exact resume, so they must stay distinguishable.
    """
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    ax2 = ax.twinx()

    g = [(500, .0694), (1000, .0695), (1500, .0694), (2000, .0690),
         (2500, .0694), (3000, .0690), (3500, .0690), (4000, .0688),
         (8000, .0688), (8500, .0686), (10000, .0687)]
    h = [(15500, .0689), (18500, .0685)]

    ax.plot([p[0] for p in g], [p[1] for p in g], "-o", color="#1f77b4",
            lw=2.2, ms=5, label="ours val -- s8g (500-10000)")
    ax.plot([p[0] for p in h], [p[1] for p in h], "--o", color="#1f77b4",
            lw=2.0, ms=7, mfc="white", mew=2.0,
            label="ours val -- s8h continuation (abs 15500 / 18500)")
    ax.axvspan(17000, 18500, color="#d62728", alpha=0.08, zorder=0)

    ax.annotate("ours val minimum recovered anywhere\n"
                "0.0685  @ absolute 18500", (18500, 0.0685),
                xytext=(-12, 26), textcoords="offset points", fontsize=8.5,
                color="#1f77b4", ha="right")
    ax.plot([18500], [0.0685], "*", color="#1f77b4", ms=17, zorder=6)

    roll = [(8000, 15.0), (10000, 20.0), (13000, 20.0), (15000, 21.7),
            (17000, 25.0), (19000, 23.3), (21000, 18.3)]
    ax2.plot([p[0] for p in roll], [p[1] for p in roll], "-s",
             color="#444444", lw=2.0, ms=6, label="ours rollout (20 demos)")
    ax2.axhline(23.3, color="#111111", lw=1.6, ls="--")
    ax2.annotate("host baseline 23.3%", (4600, 23.3), xytext=(0, 4),
                 textcoords="offset points", fontsize=8.5)
    ax2.plot([17000], [25.0], "*", color="#444444", ms=17, zorder=6)
    ax2.annotate("ours rollout maximum\n25.0%  @ absolute 17000",
                 (17000, 25.0), xytext=(-6, 10), textcoords="offset points",
                 fontsize=8.5, ha="right")
    ax2.annotate("both optima\nin this window\n(1500 steps apart)",
                 (17750, 12.2), fontsize=8.5, color="#d62728", ha="center")

    ax.set_xlabel("optimizer step (absolute; s8g 0-15000, s8h 15000-22000)")
    ax.set_ylabel("val action loss  (3 fixed demos)", color="#1f77b4")
    ax2.set_ylabel("weighted_success  (20 demos)", color="#444444")
    ax.tick_params(axis="y", colors="#1f77b4")
    ax2.tick_params(axis="y", colors="#444444")
    ax.set_ylim(0.0680, 0.0700)
    ax2.set_ylim(8, 32)
    ax.set_xlim(0, 22000)
    ax.set_title("ours: where the best val and the best rollout sit\n"
                 "diagnostic alignment, NOT the reported result -- the "
                 "pre-registered endpoint is 15000",
                 fontsize=11)
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labs, loc="lower left", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print("wrote", path)


def plot_memoryvla_vs_ours(runs, path):
    """MemoryVLA (avg host) vs ours only, each marked at its val minimum.

    Two lines, no control arms and no continuation: black = MemoryVLA (the
    official avg host), blue = ours (the RPBE run itself, `s8g`).  The x axis
    runs to the host's 20000 budget; the DATA stops earlier on both sides and
    the line is simply not drawn past the last recovered point.
    """
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    a = runs["avg_s42_stage5_official"]
    o = runs["ours_gamma-only_ours_s8g(proposal-space)"]

    ax.plot([p[0] for p in a], [p[1] for p in a], "-o", color="#111111",
            lw=2.4, ms=6, zorder=4,
            label="MemoryVLA -- avg host (Stage5, official)")
    ax.plot([p[0] for p in o], [p[1] for p in o], "-o", color="#1f77b4",
            lw=2.4, ms=6, zorder=4, label="ours -- RPBE (gamma-only, s8g)")

    # each curve's own val minimum, among the RECOVERED points
    ax.plot([18000], [0.0694], "*", color="#111111", ms=20, zorder=6)
    ax.annotate("MemoryVLA val minimum\n0.0694  @ 18000",
                (18000, 0.0694), xytext=(-10, 10), textcoords="offset points",
                fontsize=9, ha="right")
    ax.plot([8500], [0.0686], "*", color="#1f77b4", ms=20, zorder=6)
    ax.annotate("ours val minimum  0.0686  @ 8500",
                (8500, 0.0686), xytext=(-10, 44), textcoords="offset points",
                fontsize=9, color="#1f77b4", ha="right")

    # where the data actually stops on each side
    ax.axvline(20000, color="#bbbbbb", lw=1.0, ls=":")
    ax.annotate("host budget 20000\n(run stopped ~18k: no eval past it)",
                (20000, 0.0722), xytext=(-6, 0), textcoords="offset points",
                fontsize=8, color="#888888", ha="right")
    ax.axvspan(10000, 18000, color="#1f77b4", alpha=0.05, zorder=0)
    ax.annotate("ours' log stops at 10000:\n"
                "the later evals were only ever written to the box",
                (12400, 0.0960), fontsize=8.5, color="#1f77b4", ha="center")

    ax.set_xlabel("optimizer step")
    ax.set_ylabel("val action loss  (3 fixed demos)")
    ax.set_title("MemoryVLA vs ours -- each marked at its own best val\n"
                 "ours is flat from its first eval; the host only arrives "
                 "there at the end", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.set_ylim(0.0675, 0.100)
    ax.set_xlim(0, 20300)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print("wrote", path)


if __name__ == "__main__":
    runs = load()
    plot_val(runs, os.path.join(HERE, "val_curves.png"))
    plot_avg_only(runs, os.path.join(HERE, "avg_host_curve.png"))
    plot_rollout(os.path.join(HERE, "rollout_curve.png"))
    plot_host_rollout(os.path.join(HERE, "avg_host_rollout.png"))
    plot_avg_vs_ours(runs, os.path.join(HERE, "val_avg_vs_ours.png"))
    plot_alignment(os.path.join(HERE, "ours_best_val_best_rollout.png"))
    plot_memoryvla_vs_ours(runs, os.path.join(HERE, "val_memoryvla_vs_ours.png"))
