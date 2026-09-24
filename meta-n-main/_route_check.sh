#!/bin/bash
# Does the ANTHROPIC-native route provision differently from the OpenAI one?
#
# The vendor doc gives the base URL as `https://api-key.xyz/api` and says the
# same key works either as `x-api-key: dfk_...` or `Authorization: Bearer
# dfk_...` -- while the OpenAI snippet uses `/api/v1`. Everything this project
# has run so far went through the OpenAI-compatible `/api/v1` route, where
# every claude-* model came back CAPACITY or NOT_PROVISIONED. If the two routes
# carry SEPARATE entitlements, the Anthropic-native one could open a family
# that is currently unreachable -- which is worth knowing before concluding
# that only gemini + grok are available.
#
# So this walks route x header x model and prints the raw status + body head.
set -uo pipefail

set -a; . /root/autodl-tmp/meta-n-main/.env; set +a
K="$RELAY_API_KEY"
ASK='What is 4823*37? Reply with only the number, nothing else.'
BODY='{"model":"MODEL","max_tokens":64,"messages":[{"role":"user","content":"'"$ASK"'"}]}'

probe() {   # probe <label> <url> <authheader>
  local label="$1" url="$2" auth="$3"
  local out code
  out=$(curl -s -w '\n__CODE__%{http_code}' --max-time 30 \
        -H "$auth" -H 'content-type: application/json' \
        -H 'anthropic-version: 2023-06-01' \
        -H 'Accept-Encoding: identity' \
        -d "$(echo "$BODY" | sed "s/MODEL/$MODEL/")" "$url" 2>&1)
  code=$(printf '%s' "$out" | sed -n 's/.*__CODE__//p')
  printf '  %-46s http=%-4s %s\n' "$label" "$code" \
     "$(printf '%s' "$out" | sed 's/__CODE__.*//' | tr -d '\n' | cut -c1-150)"
}

for MODEL in claude-opus-5-5 claude-opus-4-8 claude-sonnet-5 gpt-5.5 grok-4.6; do
  echo "=== $MODEL"
  probe "POST /api/v1/messages  x-api-key"     "https://api-key.xyz/api/v1/messages" "x-api-key: $K"
  probe "POST /api/v1/messages  Bearer"        "https://api-key.xyz/api/v1/messages" "Authorization: Bearer $K"
  probe "POST /api/messages     x-api-key"     "https://api-key.xyz/api/messages"    "x-api-key: $K"
  probe "POST /api/v1/chat/completions Bearer" "https://api-key.xyz/api/v1/chat/completions" "Authorization: Bearer $K"
  echo
done
