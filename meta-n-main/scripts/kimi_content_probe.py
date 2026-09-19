"""Does kimi-k2.7-code emit CONTENT given a large enough budget?

The model is a reasoning model: at max_tokens=16 it spent 15 tokens on hidden
reasoning and returned empty content. Omega needs a long structured response
(rationale + pre_process + solver_lib blocks), so the question is how much
budget the reasoning actually needs before content appears.

Also confirms the account concurrency limit: calls are spaced out to avoid
colliding with the single allowed in-flight request.
"""

import json
import time
import urllib.error
import urllib.request

KEY = "sk-5MvLRsVnkTAiEMPhWeCjJjVyCpbYEjbH6MnRtfQYy3BI4j9n"
URL = "https://api.moonshot.cn/v1/chat/completions"

PROMPT = (
    "You are Omega, a meta-layer that writes improvement code.\n"
    "Output EXACTLY this structure, nothing else:\n\n"
    "```rationale\nkeep the contract\n```\n\n"
    "```pre_process\ndef pre_process(task, solution):\n    return solution\n```\n"
)

FENCE = "`" * 3


def call(model, mt):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": mt,
    }).encode()
    req = urllib.request.Request(
        URL, data=body,
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        ch = d["choices"][0]
        u = d.get("usage", {})
        det = u.get("completion_tokens_details", {}) or {}
        c = ch["message"].get("content") or ""
        print("  %-16s mt=%-6d ct=%-6s reasoning=%-6s finish=%-7s "
              "content_len=%-5d (%.0fs)" % (
                  model, mt, u.get("completion_tokens"),
                  det.get("reasoning_tokens"), ch.get("finish_reason"),
                  len(c), time.time() - t), flush=True)
        print("     head: %r" % c[:90], flush=True)
        return len(c)
    except urllib.error.HTTPError as e:
        print("  %-16s mt=%-6d HTTP %s %s" % (
            model, mt, e.code, e.read().decode()[:90]), flush=True)
    except Exception as e:                                      # noqa: BLE001
        print("  %-16s mt=%-6d %s" % (model, mt, type(e).__name__), flush=True)
    return -1


def main():
    for model in ("kimi-k2.7-code", "kimi-k2.6"):
        for mt in (4096, 32768):
            call(model, mt)
            time.sleep(20)          # concurrency is 1: space the calls out
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
