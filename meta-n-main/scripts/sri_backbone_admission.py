"""Backbone admission test: is one endpoint STABLE enough to start the SRI run?

This is an engineering admission test, not a statistical claim. PRIMARY6 x R
independent calls through the REAL code path -- the pinned SRI profile's
LLMConfig, LLMClient (with its transport retries) and Layer1Solver (with its
extraction and the new solver_output validator). Nothing here re-implements the
pipeline, so a pass means the pipeline itself is sound on this endpoint.

Criterion (user-set): every call yields a complete, parseable program and NO
call reports finish_reason='length'. 18/18 does not prove P(valid) >= 0.95 --
it is a go/no-go gate for spending a real run, nothing more.

Reported per call: status / finish_reason / prompt,completion,reasoning and
VISIBLE tokens / parse-pass / latency / retry count. Retries are reported, never
hidden: `attempts == 1` is a raw first-attempt success, and the summary reports
that rate separately from the post-retry one.

Credential/env and the CO-Bench data dir are NOT in this tree (the new harness
lives in rpbe-sri/meta-n-main, which carries neither), so both are taken from
the old tree by default.

Usage:
  python scripts/sri_backbone_admission.py --model gemini-3.8-flash
  python scripts/sri_backbone_admission.py --model gemini-3.6-flash --repeats 1
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from meta_n.core.llm_client import LLMClient, LLMConfig          # noqa: E402
from meta_n.core.solver import Layer1Solver                      # noqa: E402
from meta_n.integrations.co_bench import COBenchAdapter          # noqa: E402
from meta_n.sri.protocol import (default_profile_path,           # noqa: E402
                                 load_profile, PRIMARY6)

OLD_TREE = Path("/root/autodl-tmp/meta-n-main")


def read_env(path: Path) -> dict:
    env = {}
    if not path.is_file():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--profile", default="sri_primary6")
    ap.add_argument("--data-dir", default=str(OLD_TREE / "data/co_bench"))
    ap.add_argument("--env-file", default=str(OLD_TREE / ".env"))
    ap.add_argument("--base-url", default="https://api-key.xyz/api/v1")
    ap.add_argument("--out", default="")
    # Profile supplies the budget/retry config; these are escape hatches only,
    # so a run that deviates is visibly a deviation.
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    profile = load_profile(default_profile_path(args.profile))
    max_tokens = (profile.max_tokens if args.max_tokens is None
                  else args.max_tokens)

    env = read_env(Path(args.env_file))
    api_key = (env.get("RELAY_API_KEY") or env.get("OPENROUTER_API_KEY")
               or env.get("OPENAI_API_KEY"))
    if not api_key:
        print("FATAL: no relay key found in %s" % args.env_file, file=sys.stderr)
        return 2
    base_url = (env.get("RELAY_BASE_URL") or args.base_url)

    config = LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=args.model,
        max_tokens=max_tokens,
        # From the PROFILE, so this measures the configuration the run would use.
        empty_content_retry_max_tokens=profile.empty_retry_max_tokens,
        max_retries=profile.llm_max_retries,
        reasoning_effort=profile.reasoning_effort,
    )
    client = LLMClient(config)
    solver = Layer1Solver(client, language="python")

    adapter = COBenchAdapter(data_dir=args.data_dir, task_names=list(PRIMARY6))
    tasks = adapter.load_tasks()
    if not tasks:
        print("FATAL: no CO-Bench tasks under %s" % args.data_dir,
              file=sys.stderr)
        return 2

    print("=" * 96)
    print("BACKBONE ADMISSION  model=%s  profile=%s" % (args.model, args.profile))
    print("  max_tokens=%d  empty_retry=%d  llm_max_retries=%d  "
          "reasoning_effort=%s  temperature=%.2f"
          % (max_tokens, profile.empty_retry_max_tokens, profile.llm_max_retries,
             profile.reasoning_effort, args.temperature))
    print("  data_dir=%s" % args.data_dir)
    print("  %d tasks x %d repeats = %d calls"
          % (len(tasks), args.repeats, len(tasks) * args.repeats))
    print("=" * 96)
    hdr = ("%-32s %4s %9s %6s %6s %6s %7s %7s %6s %5s %6s"
           % ("task", "rep", "finish", "ptok", "ctok", "rtok", "vis_tok",
              "resp_ch", "parse", "retry", "sec"))
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    rows = []
    for rep in range(1, args.repeats + 1):
        for task in tasks:
            verdict: dict = {}
            t0 = time.perf_counter()
            error = ""
            try:
                await solver.solve(
                    task, temperature=args.temperature, _verdict=verdict)
            except BaseException as e:                        # noqa: BLE001
                error = "%s: %s" % (type(e).__name__, str(e)[:120])
            secs = time.perf_counter() - t0

            ct = verdict.get("completion_tokens")
            rt = verdict.get("reasoning_tokens")
            vis = (ct - rt) if isinstance(ct, int) and isinstance(rt, int) else ct
            row = {
                "task": task.task_id, "repeat": rep, "model": args.model,
                "ok": bool(verdict.get("ok")),
                "reason": verdict.get("reason", "" if not error else "error"),
                "finish_reason": verdict.get("finish_reason"),
                "attempts": verdict.get("attempts"),
                "retry_count": ((verdict.get("attempts") or 1) - 1
                                if verdict.get("attempts") else None),
                "escalated": verdict.get("escalated"),
                "prompt_tokens": verdict.get("prompt_tokens"),
                "completion_tokens": ct,
                "reasoning_tokens": rt,
                "visible_tokens": vis,
                "response_len": verdict.get("response_len"),
                "script_len": verdict.get("script_len"),
                "seconds": round(secs, 1),
                "error": error,
            }
            rows.append(row)
            print("%-32s %4d %9s %6s %6s %6s %7s %7s %6s %5s %6.1f%s" % (
                task.task_id, rep, error[:9] or row["finish_reason"],
                row["prompt_tokens"], row["completion_tokens"],
                row["reasoning_tokens"], row["visible_tokens"],
                row["response_len"], "PASS" if row["ok"] else "FAIL",
                row["retry_count"], secs,
                ("   <-- " + row["reason"]) if not row["ok"] else ""),
                flush=True)

    usable = [r for r in rows if r["ok"]]
    truncated = [r for r in rows if (r["finish_reason"] == "length")]
    first_try = [r for r in usable if r["retry_count"] == 0]
    retried = [r for r in usable if (r["retry_count"] or 0) > 0]
    vis = [r["visible_tokens"] for r in rows
           if isinstance(r["visible_tokens"], int)]

    def pct(vals, q):
        if not vals:
            return None
        s = sorted(vals)
        i = min(len(s) - 1, int(round(q * (len(s) - 1))))
        return s[i]

    print("-" * len(hdr), flush=True)
    print("calls=%d  usable=%d  finish=length=%d"
          % (len(rows), len(usable), len(truncated)), flush=True)
    print("first-attempt success=%d/%d   rescued-by-retry=%d   "
          "retry-then-failed=%d"
          % (len(first_try), len(rows), len(retried),
             len([r for r in rows if (r["retry_count"] or 0) > 0
                  and not r["ok"]])), flush=True)
    print("visible tokens: p50=%s p90=%s max=%s"
          % (pct(vis, 0.50), pct(vis, 0.90), max(vis) if vis else None),
          flush=True)
    admitted = (len(usable) == len(rows) and not truncated)
    print("=" * 96, flush=True)
    print("ADMISSION: %s  (%d/%d usable, %d with finish_reason='length')"
          % ("PASS" if admitted else "FAIL", len(usable), len(rows),
             len(truncated)), flush=True)
    print("=" * 96, flush=True)

    out = Path(args.out) if args.out else (
        REPO / "runs" / ("admission_%s.jsonl"
                         % args.model.replace("/", "_")))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("rows -> %s" % out, flush=True)
    print("RESULT_JSON " + json.dumps({
        "model": args.model, "profile": args.profile, "calls": len(rows),
        "usable": len(usable), "length_finish": len(truncated),
        "first_attempt_success": len(first_try),
        "rescued_by_retry": len(retried),
        "visible_tokens_p50": pct(vis, 0.50),
        "visible_tokens_p90": pct(vis, 0.90),
        "visible_tokens_max": max(vis) if vis else None,
        "admitted": admitted,
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
