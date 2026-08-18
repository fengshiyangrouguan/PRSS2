"""Aggregate experiment summaries into a mean±std markdown table."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _collect(root: Path):
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        job = summary_path.parent.name
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as error:
            rows.append({"job": job, "error": str(error)})
            continue
        test = summary.get("test", {})
        rows.append({
            "job": job,
            "dataset": summary.get("dataset", "?"),
            "variant": summary.get("variant", "?"),
            "seed": summary.get("seed", "?"),
            "best_epoch": summary.get("best_epoch"),
            "metric": list(test.keys())[0] if test else "?",
            "score": float(next(iter(test.values()))) if test else None,
            "spectral": summary.get("spectral", {}),
        })
    return rows


def summarize(root: Path) -> str:
    rows = _collect(root)
    by_key = defaultdict(list)
    for row in rows:
        if row.get("score") is None:
            continue
        by_key[(row["dataset"], row["variant"])].append(row)

    lines = []
    lines.append("| dataset | variant | n | score (mean ± std) | best_epoch | mechanism |")
    lines.append("|---|---|---|---|---|---|")
    for (dataset, variant), group in sorted(by_key.items()):
        scores = [r["score"] for r in group]
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / max(len(scores) - 1, 1)
        std = var ** 0.5
        mechanism = ""
        if variant == "spectral" and group[0]["spectral"]:
            snaps = group[0]["spectral"]
            entries = []
            for tau, snap in snaps.items():
                if not snap.get("dimensional_compression"):
                    continue
                entries.append(
                    f"{tau.split(':')[-1]}:rank={snap.get('effective_predictive_rank')},"
                    f"energy@k={snap.get('energy_at_k', 0):.3f},"
                    f"updates={snap.get('spectral_updates')}")
            mechanism = "; ".join(entries)
        lines.append(
            f"| {dataset} | {variant} | {len(group)} | "
            f"{mean:.6f} ± {std:.6f} | {min(r['best_epoch'] for r in group)} | {mechanism} |")
    errors = [r for r in rows if r.get("error")]
    if errors:
        lines.append("")
        lines.append("Failed/unreadable jobs:")
        for row in errors:
            lines.append(f"- {row['job']}: {row['error']}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser("PRSS2 experiment summary table")
    p.add_argument("root", help="experiment output root")
    args = p.parse_args()
    print(summarize(Path(args.root)))


if __name__ == "__main__":
    main()
