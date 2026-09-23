#!/bin/bash
# Wait for the decisive rows of the narrow forensic 4.
#
# Exits when the "predictive" rows are ALL present (the two d4->d5 pairs are the
# ones that decide the next step), when the job dies, or on budget exhaustion.
# Polls every 4 minutes; reports the rows it has each time so a long wait still
# shows progress.
cd /d/RPBE_metan || exit 1

for i in $(seq 1 60); do
  sleep 240
  out=$(MSYS_NO_PATHCONV=1 python _box.py run "cat /root/autodl-tmp/sri_r3_seed0/forensic4_narrow.jsonl 2>/dev/null; echo __SEP__; pgrep -cf '[_]forensic4_narrow'" 2>/dev/null)
  rows=$(echo "$out" | sed '/__SEP__/,$d' | grep -c '"arm"')
  alive=$(echo "$out" | sed -n '/__SEP__/,$p' | tail -1 | tr -d '[:space:]')
  echo "[poll $i] rows=$rows alive=$alive"
  if [ "$rows" = "0" ] && [ "$alive" = "0" ]; then
    echo "=== JOB DIED with no rows ==="; exit 1
  fi
  if [ "$rows" -ge 4 ]; then
    echo "=== ENOUGH ROWS ($rows) ==="
    echo "$out" | sed '/__SEP__/,$d'
    exit 0
  fi
done
echo "=== watcher budget exhausted ==="
