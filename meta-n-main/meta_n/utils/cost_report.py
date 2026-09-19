"""Print today's (or any day's) cost ledger as a per-model summary.

Usage:

    python -m meta_n.utils.cost_report                  # today
    python -m meta_n.utils.cost_report --date 2026-05-08
    python -m meta_n.utils.cost_report --since 2026-05-01  # date range from --since to today
    python -m meta_n.utils.cost_report --ledger-dir /custom/path

Reads the JSONL ledgers written by ``CostTracker.record``. Each line is a
plain dict with model / prompt_tokens / completion_tokens / cached_tokens /
cost_usd / pid / ts — easy to grep, awk, or pipe into jq if you want to
slice differently.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


def _read_ledger(path: Path) -> list[dict]:
    """Read one daily ledger file. Returns [] if the file is missing."""
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _summarise(rows: list[dict]) -> dict:
    """Roll up rows into per-model totals + global totals."""
    by_model: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "prompt": 0, "completion": 0, "cached": 0, "cost_usd": 0.0}
    )
    total = {"calls": 0, "prompt": 0, "completion": 0, "cached": 0, "cost_usd": 0.0}
    for r in rows:
        # json.loads parses a literal NaN/Infinity WITHOUT raising, and
        # ``nan or 0.0`` keeps the NaN (NaN is truthy) — one poisoned line
        # would turn every total into NaN. Treat a non-finite/non-numeric
        # cost as $0 so the report agrees with the enforcement path
        # (CostTracker.today_total_usd skips non-finite values). The call
        # still counts: a call happened; only its cost is unusable.
        try:
            val = float(r.get("cost_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            val = 0.0
        if not math.isfinite(val):
            val = 0.0
        m = r.get("model", "?")
        bm = by_model[m]
        bm["calls"] += 1
        bm["prompt"] += int(r.get("prompt_tokens", 0) or 0)
        bm["completion"] += int(r.get("completion_tokens", 0) or 0)
        bm["cached"] += int(r.get("cached_tokens", 0) or 0)
        bm["cost_usd"] += val
        total["calls"] += 1
        total["prompt"] += int(r.get("prompt_tokens", 0) or 0)
        total["completion"] += int(r.get("completion_tokens", 0) or 0)
        total["cached"] += int(r.get("cached_tokens", 0) or 0)
        total["cost_usd"] += val
    return {"by_model": dict(by_model), "total": total}


def _print_table(label: str, summary: dict) -> None:
    print(f"\n=== {label} ===")
    if not summary["by_model"]:
        print("  (no calls)")
        return
    print(f"  {'Model':<24} {'Calls':>8} {'Prompt':>14} {'Compl':>12} "
          f"{'Cached':>10} {'Cost USD':>12}")
    for model, s in sorted(summary["by_model"].items()):
        print(
            f"  {model:<24} {s['calls']:>8} {s['prompt']:>14,} "
            f"{s['completion']:>12,} {s['cached']:>10,} {s['cost_usd']:>12.4f}"
        )
    t = summary["total"]
    print(
        f"  {'TOTAL':<24} {t['calls']:>8} {t['prompt']:>14,} "
        f"{t['completion']:>12,} {t['cached']:>10,} {t['cost_usd']:>12.4f}"
    )


def _daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=1)


def _today(utc: bool) -> date:
    """'Today' for ledger-file selection. Matches CostTracker's day keys:
    local date by default, UTC date when the tracker rolls at UTC midnight."""
    return datetime.now(timezone.utc).date() if utc else date.today()


def main():
    parser = argparse.ArgumentParser(
        description="Summarise the meta_n daily cost ledger."
    )
    parser.add_argument(
        "--ledger-dir",
        default=os.environ.get("META_N_COST_LEDGER_DIR", "~/.meta_n_costs"),
        help="Directory holding daily JSONL ledgers (default: $META_N_COST_LEDGER_DIR or ~/.meta_n_costs)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Specific day (YYYY-MM-DD) to summarise. Defaults to today.",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Range mode: summarise every day from --since (YYYY-MM-DD) up to today, "
             "with a per-day breakdown plus a grand total.",
    )
    parser.add_argument(
        "--utc",
        action="store_true",
        help="Interpret 'today' at UTC midnight (matches CostTracker(utc=True) "
             "day keys). Default: local date, matching the tracker default.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a table (for piping into other tools).",
    )
    args = parser.parse_args()

    ledger_dir = Path(args.ledger_dir).expanduser()

    if args.since:
        start = datetime.strptime(args.since, "%Y-%m-%d").date()
        end = _today(args.utc)
        per_day = {}
        grand_rows: list[dict] = []
        for d in _daterange(start, end):
            path = ledger_dir / f"{d.isoformat()}.jsonl"
            rows = _read_ledger(path)
            per_day[d.isoformat()] = _summarise(rows)
            grand_rows.extend(rows)
        grand = _summarise(grand_rows)
        if args.json:
            print(json.dumps({"per_day": per_day, "grand_total": grand}, indent=2))
            return
        for d, s in per_day.items():
            _print_table(d, s)
        _print_table(f"GRAND TOTAL ({start} → {end})", grand)
        return

    target = args.date or _today(args.utc).isoformat()
    path = ledger_dir / f"{target}.jsonl"
    rows = _read_ledger(path)
    summary = _summarise(rows)
    if args.json:
        print(json.dumps(summary, indent=2))
        return
    _print_table(f"{target} ({path})", summary)


if __name__ == "__main__":
    main()
