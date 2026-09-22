"""Matched pair: is the harness's WIRE REQUEST identical to a direct one?

Prints, for one task, the request the harness actually put on the socket and the
request a bare client puts on the socket, then compares their sha256. Only if
those match does "the model is just nondeterministic" become a meaningful
statement -- otherwise the two calls are not the same experiment.

Also dumps the RAW provider response (pre-SDK), which is where a tool call
actually lives. Earlier we only ever saw ``message.content``.

Usage: python scripts/sri_wire_matched_pair.py [task]
"""
import asyncio
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openai import AsyncOpenAI                                     # noqa: E402

from meta_n.core.llm_client import LLMClient, LLMConfig            # noqa: E402
from meta_n.core.solver import Layer1Solver                        # noqa: E402
from meta_n.integrations.co_bench import COBenchAdapter            # noqa: E402
from meta_n.utils.wire_capture import environment_fingerprint, install  # noqa: E402

TASK = sys.argv[1] if len(sys.argv) > 1 else "Aircraft landing"
OLD_TREE = Path("/root/autodl-tmp/meta-n-main")
OUT_DIR = Path("/tmp/wire_matched")

#: Parameters the OpenAI SDK accepts by name. Everything else in the serialized
#: body arrived via extra_body and must be replayed the same way.
SDK_PARAMS = {
    "model", "messages", "temperature", "max_tokens", "max_completion_tokens",
    "top_p", "n", "stop", "stream", "seed", "user", "logprobs", "top_logprobs",
    "presence_penalty", "frequency_penalty", "logit_bias", "response_format",
    "tools", "tool_choice", "parallel_tool_calls", "stream_options",
    "service_tier", "metadata", "store", "modalities", "reasoning_effort",
}


def read_env(path: Path) -> dict:
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def brief(rec, label):
    print("--- %s ---" % label)
    print("  url            : %s" % rec["url"])
    print("  status         : %s" % rec["status"])
    print("  request_bytes  : %s" % rec["request_bytes"])
    print("  request_sha256 : %s" % rec["request_sha256"])
    print("  response_bytes : %s" % rec["response_bytes"])
    print("  response_sha256: %s" % rec["response_sha256"])
    try:
        body = json.loads(rec["response_body"])
        ch = (body.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        print("  raw id         : %s" % body.get("id"))
        print("  raw model      : %s" % body.get("model"))
        print("  raw finish     : %s" % ch.get("finish_reason"))
        print("  raw usage      : %s" % json.dumps(body.get("usage")))
        print("  raw content    : %s chars" % len(msg.get("content") or ""))
        print("  raw tool_calls : %s" % json.dumps(msg.get("tool_calls")))
        print("  raw refusal    : %s" % json.dumps(msg.get("refusal")))
        extra = set(msg) - {"content", "role", "tool_calls", "refusal",
                            "annotations", "audio", "function_call"}
        if extra:
            print("  raw msg extra  : %s" % sorted(extra))
    except Exception as e:                                    # noqa: BLE001
        print("  (response not JSON: %r)" % (e,))
        print("  raw head       : %r" % rec["response_body"][:300])


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = read_env(OLD_TREE / ".env")
    key = env["RELAY_API_KEY"]
    base_url = "https://api-key.xyz/api/v1"
    model = "gemini-3.8-flash"

    adapter = COBenchAdapter(data_dir=str(OLD_TREE / "data/co_bench"),
                             task_names=[TASK])
    task = adapter.load_tasks()[0]

    print("=" * 92)
    print("environment: %s" % json.dumps(environment_fingerprint()))
    print("task=%s  model=%s  base_url=%s" % (TASK, model, base_url))
    print("=" * 92)

    # ---------------- PART A: the harness path -------------------------------
    cap_a_path = OUT_DIR / "harness.jsonl"
    cap_a_path.unlink(missing_ok=True)
    client = LLMClient(LLMConfig(
        base_url=base_url, api_key=key, model=model, max_tokens=32768,
        empty_content_retry_max_tokens=0, max_retries=2,
        reasoning_effort="low"))
    cap_a = install(client._client, cap_a_path)
    assert cap_a is not None, "wire capture did not install on the harness client"
    solver = Layer1Solver(client, language="python")

    verdict: dict = {}
    script, reasoning, tokens = await solver.solve(
        task, temperature=0.3, _verdict=verdict)
    print("PART A physical requests: %d" % len(cap_a.records))
    rec_a = cap_a.records[-1]
    brief(rec_a, "PART A  harness wire record")
    print("  solver verdict : %s" % json.dumps(verdict)[:400])

    # ---------------- PART B: direct replay of the SAME body -----------------
    cap_b_path = OUT_DIR / "direct.jsonl"
    cap_b_path.unlink(missing_ok=True)
    bare = AsyncOpenAI(base_url=base_url, api_key=key, max_retries=0,
                       timeout=900.0,
                       default_headers={"Accept-Encoding": "identity"})
    cap_b = install(bare, cap_b_path)
    assert cap_b is not None, "wire capture did not install on the bare client"

    sent = json.loads(rec_a["request_body"])
    call_kwargs = {k: v for k, v in sent.items() if k in SDK_PARAMS}
    extra = {k: v for k, v in sent.items() if k not in SDK_PARAMS}
    if extra:
        call_kwargs["extra_body"] = extra
    print()
    print("PART B replays: params=%s extra_body=%s"
          % (sorted(call_kwargs), json.dumps(extra)))
    await bare.chat.completions.create(**call_kwargs)
    rec_b = cap_b.records[-1]
    brief(rec_b, "PART B  direct wire record")

    # ---------------- comparison --------------------------------------------
    print()
    print("=" * 92)
    print("COMPARISON")
    print("=" * 92)
    same_body = rec_a["request_sha256"] == rec_b["request_sha256"]
    print("  request_body_sha256  %s" % ("IDENTICAL" if same_body else "DIFFERENT"))
    print("    A: %s" % rec_a["request_sha256"])
    print("    B: %s" % rec_b["request_sha256"])
    if not same_body:
        a, b = rec_a["request_body"], rec_b["request_body"]
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print("    first divergence at byte %d:" % i)
                print("      A: %r" % a[max(0, i - 60):i + 90])
                print("      B: %r" % b[max(0, i - 60):i + 90])
                break
        else:
            print("    (one is a prefix of the other; lenA=%d lenB=%d)"
                  % (len(a), len(b)))

    for label, rec in (("A harness", rec_a), ("B direct", rec_b)):
        try:
            body = json.loads(rec["response_body"])
            ch = (body.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            print("  %-10s finish=%-12s content=%-6d tool_calls=%s"
                  % (label, ch.get("finish_reason"),
                     len(msg.get("content") or ""),
                     json.dumps(msg.get("tool_calls"))))
        except Exception:                                     # noqa: BLE001
            print("  %-10s <unparseable response>" % label)
    print("  raw wire records written under %s" % OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
