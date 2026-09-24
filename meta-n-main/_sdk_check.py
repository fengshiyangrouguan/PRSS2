"""Verify the relay against the vendor's OWN documented snippet.

The question being settled: is `model_not_provisioned` for gpt-5.5 a
CONNECTION problem (wrong base_url / wrong auth header) or an AUTHORIZATION
decision by the relay about this key?

So this is deliberately the documented form and nothing else:

    OpenAI(api_key="dfk_...", base_url="https://api-key.xyz/api/v1")

Same base_url and same key as the real harness uses. If gpt-5.5 fails here too,
while grok-4.6 and gemini-3.1-pro succeed on the identical connection, then the
channel is provably fine and the difference is per-model provisioning.
"""
import os
import sys

from openai import OpenAI

ENV = "/root/autodl-tmp/meta-n-main/.env"
BASE = "https://api-key.xyz/api/v1"

key = None
for line in open(ENV):
    line = line.strip()
    if line.startswith("RELAY_API_KEY="):
        key = line.split("=", 1)[1].strip().strip('"').strip("'")
if not key:
    sys.exit("RELAY_API_KEY not found in " + ENV)

print("base_url :", BASE)
print("key      :", key[:12] + "..." + key[-6:])
print()

client = OpenAI(
    api_key=key, base_url=BASE,
    # The relay gzips responses and the installed httpx2 decoder then dies with
    # `process() takes no keyword arguments`. The harness passes this exact
    # header via META_N_EXTRA_HEADERS_JSON; a bare SDK does not read that env
    # var, so without it here every row would fail for a reason that has nothing
    # to do with the model.
    default_headers={"Accept-Encoding": "identity"})
ASK = "What is 4823*37? Reply with only the number, nothing else."

for model in ["gpt-5.5", "gpt-5.4", "grok-4.6", "gemini-3.1-pro"]:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": ASK}],
            max_tokens=64)
        txt = (r.choices[0].message.content or "").strip()
        print("  %-16s OK    model_echo=%-16s -> %r"
              % (model, r.model, txt[:24]))
    except Exception as e:                                     # noqa: BLE001
        print("  %-16s FAIL  %s: %s"
              % (model, type(e).__name__, str(e)[:170]))
print()
print("READING: identical connection for every row. If the failing rows name")
print("the model in the error body, the channel is fine and the key is simply")
print("not authorized for that model.")
