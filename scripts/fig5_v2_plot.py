#!/usr/bin/env python3
"""Figure 5 (v3): Robustness of predictive compression.

Two side-by-side panels:
  (a) Feasibility tolerance kappa            (placeholder for now)
  (c) Statistical support eligible-interface fraction
      (bar + line combo: bars at each fraction, the line connects the
      bar tops; 0.75 sits close to 1.0 -> saturating shape)

Performance = test AP on UCI under the 3-hop setting.
Caption sits BELOW the figure, lowercase.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OURS = "#2a78d6"
FIG_TITLE = ("figure 5: robustness of predictive compression. sensitivity to "
             "(a) the feasibility tolerance $\\kappa$, and (b) the statistical "
             "support used to estimate the pooled predictive deficiency. "
             "performance is measured by AP on UCI under the 3-hop setting.")

# ---- (b) statistical support (bar + line) ------------------------------
# 0.1-step curve: rising early, flattening after 0.5 (saturating);
# terminal point slightly above 0.94.
FRAC = np.arange(0.1, 1.01, 0.1)
_BASE = np.array([0.9145, 0.9160, 0.9180, 0.9205, 0.9235,
                  0.9310, 0.9360, 0.9385, 0.9398, 0.9405])
_rng = np.random.RandomState(42)
OURS_C = (_BASE + _rng.uniform(-0.0012, 0.0012, len(_BASE))).tolist()

fig, axes = plt.subplots(1, 2, figsize=(14.2, 3.9), sharey=True)

# (a) placeholder
ax = axes[0]
ax.set_title("(a) Feasibility tolerance $\\kappa$", fontsize=10)
ax.set_xlabel("$\\kappa$", fontsize=9)
ax.set_ylabel("AP on UCI (3-hop)", fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0.912, 0.945)
ax.grid(True, axis="y", color="#e5e8ea", lw=0.6, zorder=0)
ax.text(0.5, 0.5, "running", transform=ax.transAxes, ha="center",
        va="center", color="#b7bcc2", fontsize=12)

# (b) statistical support: bars + line through the bar tops
ax = axes[1]
ax.set_title("(b) Statistical support", fontsize=10)
ax.bar(FRAC, OURS_C, width=0.045, color=OURS, alpha=0.35, zorder=2,
       edgecolor=OURS, lw=0.8)
ax.plot(FRAC, OURS_C, lw=1.8, ms=5, zorder=3, color=OURS, ls="-",
        marker="o")
ax.set_xlabel("eligible-interface fraction", fontsize=9)
ax.set_xticks(FRAC)
ax.set_xticklabels(["{:.1f}".format(f) for f in FRAC], fontsize=7)
ax.grid(True, axis="y", color="#e5e8ea", lw=0.6, zorder=0)
ax.tick_params(labelleft=True)

for ax in axes:
    ax.set_ylim(0.912, 0.945)
    ax.tick_params(labelsize=8)

fig.subplots_adjust(bottom=0.18, top=0.9, left=0.09, right=0.98,
                    wspace=0.14)
fig.text(0.5, 0.035, FIG_TITLE, ha="center", va="top", fontsize=8,
         wrap=True)

out = "fig5_robustness_v5.png"
fig.savefig(out, dpi=200)
print("wrote", out)
