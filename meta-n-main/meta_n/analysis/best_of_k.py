"""best_of_k — reduce an ARM (a set of runs) to a best-of-K headline, the way
evolutionary baselines report a single best-of-N score, and compare arms fairly.

An ARM is a set of independent meta-n run directories (each with a summary.json).
Per the reporting protocol in docs/metan_new_instruments_experiment_plan.md:

- **best-of-K (pick the best run)** — the run-level max over K runs, matching how
  the literature reports best-of-N. This is the headline reducer.
- **best-of-search** (`oracle_mean_score`) is the correct metric for search-vs-search
  (meta-n vs OpenEvolve/Gödel — all report their best candidate); **deployable**
  (`chain_test_mean_score`, the single shipped chain on the held-out split) is the
  honest "what you'd ship" number and the right metric vs a single-shot baseline.
- **compute parity** (total LLM calls) is the real fairness lever for search-vs-search
  and is reported alongside every arm so parity can be verified post-hoc.
- **thesis ablation:** a paired per-task comparison of meta-n's best-of-search against
  a base-solver best-of-N control at matched compute (max reducer, both over ~N samples).

Usage:
    python -m meta_n.analysis.best_of_k \\
        --arm meta-n exp/treat_s42 exp/treat_s43 exp/treat_s44 \\
        --arm control exp/control_seed1 ... exp/control_seed8 \\
        --pair meta-n control
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "RunMetrics",
    "load_run",
    "Arm",
    "paired_per_task",
    "render_arm_table",
]

# summary.json field -> our label. best-of-search = oracle (max over candidates);
# deployable = single shipped chain on the held-out split.
_BEST_OF_SEARCH_DEV = "oracle_mean_score"
_DEPLOYABLE_DEV = "best_mean_score"
_BEST_OF_SEARCH_TEST = "test_mean_score"
_DEPLOYABLE_TEST = "chain_test_mean_score"


@dataclass
class RunMetrics:
    name: str
    best_of_search_dev: float | None
    deployable_dev: float | None
    best_of_search_test: float | None
    deployable_test: float | None
    per_task: dict[str, float]  # per-task best-of-search (dev)
    calls: int  # total LLM calls (outer + inner) = compute for parity
    run_status: str


def _summary_path(run_dir: Path) -> Path:
    """Locate summary.json. Accept either the run dir itself or a parent that
    contains exactly one run (e.g. output_dir when --exp-name was used)."""
    direct = run_dir / "summary.json"
    if direct.exists():
        return direct
    hits = sorted(run_dir.glob("*/summary.json")) + sorted(run_dir.glob("*/*/summary.json"))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(f"no summary.json under {run_dir}")
    raise ValueError(f"{run_dir} contains {len(hits)} runs; point at a single run dir")


def load_run(run_dir: str | Path) -> RunMetrics:
    run_dir = Path(run_dir)
    d = json.loads(_summary_path(run_dir).read_text())
    tu = d.get("token_usage") or {}
    calls = int(tu.get("outer_calls", 0) or 0) + int(tu.get("inner_calls", 0) or 0)

    def _fin(v):
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None

    per_task = {
        k: float(v)
        for k, v in (d.get("per_task_best_scores") or {}).items()
        if isinstance(v, (int, float)) and math.isfinite(v)
    }
    return RunMetrics(
        name=run_dir.name,
        best_of_search_dev=_fin(d.get(_BEST_OF_SEARCH_DEV)),
        deployable_dev=_fin(d.get(_DEPLOYABLE_DEV)),
        best_of_search_test=_fin(d.get(_BEST_OF_SEARCH_TEST)),
        deployable_test=_fin(d.get(_DEPLOYABLE_TEST)),
        per_task=per_task,
        calls=calls,
        run_status=str(d.get("run_status", "")),
    )


_METRICS = [
    ("best_of_search_dev", "best-of-search (dev)"),
    ("deployable_dev", "deployable (dev)"),
    ("best_of_search_test", "best-of-search (test)"),
    ("deployable_test", "deployable (test)"),
]


@dataclass
class Arm:
    name: str
    runs: list[RunMetrics] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.runs)

    def best_of_k(self, metric: str) -> tuple[float, str] | None:
        """Pick the best RUN (run-level max) for `metric` — the literature best-of-N."""
        scored = [(getattr(r, metric), r.name) for r in self.runs if getattr(r, metric) is not None]
        return max(scored) if scored else None

    def per_task_best_of_k(self) -> dict[str, float]:
        """Per-task max across the arm's runs (oracle over the K runs) — used for
        the paired thesis-ablation vector; more generous than pick-the-best-run."""
        out: dict[str, float] = {}
        for r in self.runs:
            for t, v in r.per_task.items():
                if t not in out or v > out[t]:
                    out[t] = v
        return out

    def mean_calls(self) -> float:
        c = [r.calls for r in self.runs]
        return sum(c) / len(c) if c else 0.0


def render_arm_table(arms: list[Arm]) -> str:
    lines = [
        "| arm | K | " + " | ".join(lbl for _, lbl in _METRICS) + " | mean calls/run |",
        "|---|---|" + "|".join("---" for _ in _METRICS) + "|---|",
    ]
    for a in arms:
        cells = []
        for m, _ in _METRICS:
            bok = a.best_of_k(m)
            cells.append(f"{bok[0]:.3f} ({bok[1]})" if bok else "—")
        lines.append(f"| {a.name} | {a.k} | " + " | ".join(cells) + f" | {a.mean_calls():.0f} |")
    return "\n".join(lines)


def _sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign test (ties excluded), p under H0=0.5."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    # P(X<=k) two-sided
    cum = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * cum)


def paired_per_task(treat: dict[str, float], control: dict[str, float], *, eps: float = 1e-9) -> dict:
    """Paired per-task comparison (thesis ablation): treat=meta-n best-of-search,
    control=base best-of-N. Returns per-task deltas, W/L/T, mean delta, sign-test p."""
    tasks = sorted(set(treat) & set(control))
    deltas = {t: treat[t] - control[t] for t in tasks}
    wins = sum(1 for d in deltas.values() if d > eps)
    losses = sum(1 for d in deltas.values() if d < -eps)
    ties = len(tasks) - wins - losses
    mean_delta = sum(deltas.values()) / len(tasks) if tasks else 0.0
    return {
        "n_tasks": len(tasks),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_delta": mean_delta,
        "sign_test_p": _sign_test_p(wins, losses),
        "per_task_delta": deltas,
    }


def _build_arms(arm_specs: list[list[str]]) -> list[Arm]:
    arms = []
    for spec in arm_specs:
        name, dirs = spec[0], spec[1:]
        arms.append(Arm(name=name, runs=[load_run(d) for d in dirs]))
    return arms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="best-of-K arm comparison")
    ap.add_argument("--arm", nargs="+", action="append", required=True,
                    metavar=("NAME", "RUN_DIR"),
                    help="arm name followed by its run dirs (repeatable)")
    ap.add_argument("--pair", nargs=2, metavar=("TREAT", "CONTROL"),
                    help="paired per-task thesis-ablation between two arm names")
    ap.add_argument("--output", default=None, help="write the report markdown here")
    args = ap.parse_args(argv)

    arms = _build_arms(args.arm)
    by_name = {a.name: a for a in arms}

    out = ["# best-of-K arm comparison\n",
           "Best-of-K = pick the best run (run-level max), matching a literature "
           "best-of-N headline. Compute (mean calls/run) is shown for parity — "
           "search-vs-search is only fair at matched compute.\n",
           render_arm_table(arms), ""]

    if args.pair:
        t, c = args.pair
        if t not in by_name or c not in by_name:
            ap.error(f"--pair names must be arms; got {t},{c}")
        res = paired_per_task(by_name[t].per_task_best_of_k(), by_name[c].per_task_best_of_k())
        out += [
            f"\n## Thesis ablation (paired per-task): {t} best-of-search vs {c} best-of-N",
            f"- tasks: {res['n_tasks']} | wins: {res['wins']} losses: {res['losses']} ties: {res['ties']}",
            f"- mean per-task delta: {res['mean_delta']:+.4f}",
            f"- sign-test p (two-sided): {res['sign_test_p']:.4f}",
            f"- compute: {t} {by_name[t].mean_calls():.0f} calls/run × {by_name[t].k} vs "
            f"{c} {by_name[c].mean_calls():.0f} calls/run × {by_name[c].k}",
            "\n> Reject H0 (Ω beats best-of-N) only if mean_delta>0 AND sign-test p<0.05 "
            "AND compute is matched. Otherwise report as a TIE.",
        ]

    report = "\n".join(out)
    if args.output:
        Path(args.output).write_text(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
