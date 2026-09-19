"""Hard budget fuse for paid Phase A runs.

User policy (2026-09-17), after the symptom2disease pilot cost ~$2 for three
iterations on ONE task -- the inner solver turned 853 training rows into
per-row LLM calls:

  * no batch Phase A run may start without an explicit cap;
  * a per-run cap (default $0.25) kills the run the moment it is exceeded;
  * a whole-Phase-A cap (default $0.50) kills every run once reached;
  * the fuse refuses to start at all if the provider balance cannot cover the
    requested total cap plus a buffer.

Meta^n has its own ``META_N_DAILY_BUDGET_USD`` guard, but it counts the whole
DAY (shared across runs) and only fires between calls. This module is the
belt-and-braces version: it attributes cost to a PID via the ledger's ``pid``
field and kills the process externally.

Ledger record shape (``~/.meta_n_costs/YYYY-MM-DD.jsonl``)::

    {"ts": float, "model": str, "prompt_tokens": int, "completion_tokens": int,
     "cached_tokens": int, "cost_usd": float, "pid": int}
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

LEDGER_DIR = Path(os.environ.get("META_N_COST_LEDGER_DIR",
                                 str(Path.home() / ".meta_n_costs")))

# Defaults chosen for the CURRENT balance (see --balance-check). Both are
# deliberately far below the remaining credit.
DEFAULT_PER_RUN_CAP = 0.25
DEFAULT_TOTAL_CAP = 0.50
BALANCE_BUFFER = 0.50          # never plan to spend the last half dollar


class BudgetExceeded(RuntimeError):
    pass


# --------------------------------------------------------------------------
# ledger reading
# --------------------------------------------------------------------------

def ledger_path(when: Optional[float] = None) -> Path:
    import datetime as _dt
    d = _dt.datetime.utcfromtimestamp(when if when is not None else time.time())
    return LEDGER_DIR / "{}.jsonl".format(d.strftime("%Y-%m-%d"))


def iter_records(path: Path) -> Iterator[dict]:
    if not path.is_file():
        return
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _cost(rec: dict) -> float:
    return float(rec.get("cost_usd") or rec.get("cost") or 0.0)


def total_cost(path: Optional[Path] = None) -> float:
    return sum(_cost(r) for r in iter_records(path or ledger_path()))


def cost_for_pid(pid: int, path: Optional[Path] = None) -> float:
    return sum(_cost(r) for r in iter_records(path or ledger_path())
               if int(r.get("pid") or -1) == int(pid))


# --------------------------------------------------------------------------
# provider balance
# --------------------------------------------------------------------------

def deepseek_balance_usd(api_key: Optional[str] = None,
                         base_url: str = "https://api.deepseek.com",
                         retries: int = 3) -> Optional[float]:
    """Remaining DeepSeek credit in USD, or None if it cannot be read.

    The balance endpoint is occasionally flaky; retry a few times so a
    transient blip does not make the fuse refuse a legitimate run. Callers
    must still treat None as fail-CLOSED (never as "enough credit").
    """
    import urllib.request
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return None
    last = None
    for attempt in range(max(1, int(retries))):
        try:
            req = urllib.request.Request(
                base_url + "/user/balance",
                headers={"Authorization": "Bearer " + key})
            with urllib.request.urlopen(req, timeout=20) as r:
                blob = json.load(r)
            for info in blob.get("balance_infos", []):
                cur = str(info.get("currency", "")).upper()
                if cur == "CNY":
                    return float(info["total_balance"]) / 7.2   # rough CNY->USD
                if cur == "USD":
                    return float(info["total_balance"])
            return None
        except Exception as e:                                  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    return None


def assert_balance_covers(total_cap: float, api_key: Optional[str] = None,
                          buffer: float = BALANCE_BUFFER) -> float:
    """Refuse to start unless the balance covers the cap plus a buffer."""
    bal = deepseek_balance_usd(api_key)
    if bal is None:
        raise BudgetExceeded(
            "cannot read provider balance; refusing to start a paid run "
            "without knowing the remaining credit")
    if total_cap + buffer > bal:
        raise BudgetExceeded(
            "total cap ${:.2f} + buffer ${:.2f} exceeds remaining balance "
            "${:.2f}; lower the cap or top up".format(total_cap, buffer, bal))
    return bal


# --------------------------------------------------------------------------
# fuse
# --------------------------------------------------------------------------

@dataclass
class Caps:
    per_run: float = DEFAULT_PER_RUN_CAP
    total: float = DEFAULT_TOTAL_CAP


class Fuse:
    """Tracks one run's spend against the per-run and whole-phase caps."""

    def __init__(self, pid: int, caps: Caps, *, phase_baseline: float = 0.0,
                 path: Optional[Path] = None, poll: float = 2.0):
        self.pid = int(pid)
        self.caps = caps
        self.path = path or ledger_path()
        self.phase_baseline = float(phase_baseline)
        self.poll = float(poll)
        self.run_spend = 0.0
        self.phase_spend = 0.0

    def refresh(self) -> None:
        self.run_spend = cost_for_pid(self.pid, self.path)
        self.phase_spend = total_cost(self.path) - self.phase_baseline

    def check(self) -> Optional[str]:
        """Return a reason string if a cap is breached, else None."""
        self.refresh()
        if self.run_spend > self.caps.per_run:
            return ("per-run cap: ${:.4f} > ${:.2f}".format(
                self.run_spend, self.caps.per_run))
        if self.phase_spend > self.caps.total:
            return ("phase-A cap: ${:.4f} > ${:.2f}".format(
                self.phase_spend, self.caps.total))
        return None

    def watch(self, on_breach=None) -> str:
        """Block until the process exits or a cap is breached."""
        while True:
            reason = self.check()
            if reason is not None:
                if on_breach is not None:
                    on_breach(reason)
                return reason
            try:
                os.kill(self.pid, 0)
            except OSError:
                return "process exited"
            time.sleep(self.poll)


def kill_tree(pid: int, grace: float = 3.0) -> None:
    """SIGTERM the process group, then SIGKILL if it lingers."""
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


# --------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase A budget fuse")
    ap.add_argument("--watch", type=int, metavar="PID",
                    help="watch a running Meta^n PID and kill it on breach")
    ap.add_argument("--per-run", type=float, default=DEFAULT_PER_RUN_CAP)
    ap.add_argument("--total", type=float, default=DEFAULT_TOTAL_CAP)
    ap.add_argument("--baseline", type=float, default=0.0,
                    help="ledger total at the start of Phase A")
    ap.add_argument("--check-balance", action="store_true",
                    help="verify the provider balance covers --total + buffer")
    ap.add_argument("--report", action="store_true",
                    help="print today's ledger total and provider balance")
    args = ap.parse_args(argv)

    if args.report:
        print("ledger total today : ${:.4f}".format(total_cost()))
        bal = deepseek_balance_usd()
        print("provider balance   : {}".format(
            "unreadable" if bal is None else "${:.2f}".format(bal)))
        return 0

    if args.check_balance:
        bal = assert_balance_covers(args.total)
        print("balance ${:.2f} covers cap ${:.2f} + buffer ${:.2f}  OK".format(
            bal, args.total, BALANCE_BUFFER))
        return 0

    if not args.watch:
        ap.error("give --watch PID, --report, or --check-balance")

    caps = Caps(per_run=args.per_run, total=args.total)
    fuse = Fuse(args.watch, caps, phase_baseline=args.baseline)
    if fuse.check() is not None:
        print("already over cap before watching; killing now", file=sys.stderr)
        reason = fuse.check()
        kill_tree(args.watch)
        print("KILLED: " + reason)
        return 2
    reason = fuse.watch(on_breach=lambda r: kill_tree(args.watch))
    if reason == "process exited":
        fuse.refresh()
        print("process exited; run spend ${:.4f}, phase spend ${:.4f}".format(
            fuse.run_spend, fuse.phase_spend))
        return 0
    print("KILLED: " + reason)
    print("run spend ${:.4f}, phase spend ${:.4f}".format(
        fuse.run_spend, fuse.phase_spend))
    return 2


if __name__ == "__main__":
    sys.exit(main())
