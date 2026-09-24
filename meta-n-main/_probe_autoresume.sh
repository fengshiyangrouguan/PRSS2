#!/bin/bash
# Auto-resume the Structural6 formal-shape search probe until it finishes.
#
# WHY. The relay (api-key.xyz) is FLAPPING: two 502 `origin_bad_gateway`
# responses ~8 minutes apart (22:52Z and 23:00Z) killed two attempts mid
# iteration-2, each after only ~3 requests. The client already retries
# InternalServerError (`llm-max-retries 2` -> 3 tries); the outage simply
# outlasted that window, so a longer client-side retry would not have helped --
# what is needed is a retry at the RUN level, which is what this is.
#
# THE ONE DESIGN POINT THAT MATTERS: the endpoint is tested FOR FREE before
# every attempt, and an attempt only starts when the origin actually answers.
# A naive retry loop would spend ~3 requests per spin against a dead origin.
#
# Each attempt resumes from the arm's OWN checkpoint, so an interrupted attempt
# never re-buys the iterations it already paid for -- measured: the second
# attempt resumed at iteration 1 and re-ran nothing, and the third skipped the
# two already-terminal gen2 slots.
set -uo pipefail

PROBE=/root/autodl-tmp/sri_s6_probe
LOG=$PROBE/autoresume.log
URL=https://api-key.xyz/api/v1/models
MAX_ATTEMPTS=20

echo "[$(date +%F\ %T)] autoresume starting (max $MAX_ATTEMPTS attempts)" >> "$LOG"

for attempt in $(seq 1 $MAX_ATTEMPTS); do
  code=000
  for wait in $(seq 1 60); do
    # 401 == upstream alive and answering; we are simply not authenticated on
    # this probe. 000/timeout/5xx mean the origin is still down.
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$URL" 2>/dev/null || echo 000)
    case "$code" in
      200|401|403) break ;;
    esac
    echo "[$(date +%F\ %T)] endpoint http=$code -- waiting 60s" >> "$LOG"
    sleep 60
  done
  echo "[$(date +%F\ %T)] attempt $attempt starting (endpoint http=$code)" >> "$LOG"
  bash "$PROBE/launch.sh" >> "$LOG" 2>&1
  rc=$?
  echo "[$(date +%F\ %T)] attempt $attempt finished rc=$rc" >> "$LOG"
  if [ "$rc" -eq 0 ]; then
    echo "[$(date +%F\ %T)] PROBE COMPLETE" >> "$LOG"
    exit 0
  fi
  sleep 120
done

echo "[$(date +%F\ %T)] gave up after $MAX_ATTEMPTS attempts" >> "$LOG"
exit 1
