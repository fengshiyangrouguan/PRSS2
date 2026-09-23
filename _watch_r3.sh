#!/bin/bash
# Watcher for the seed-0 chain. Exits when the chain reports a terminal marker.
cd /d/RPBE_metan 2>/dev/null || cd "$(dirname "$0")"
for i in $(seq 1 200); do
  sleep 300
  st=$(MSYS_NO_PATHCONV=1 python _box.py run "grep -c 'ALL_DONE\|CHAIN_GAVE_UP' /root/autodl-tmp/sri_r3_seed0.log 2>/dev/null; tail -1 /root/autodl-tmp/sri_r3_seed0.log 2>/dev/null; grep -E '^=== stage ' /root/autodl-tmp/sri_r3_seed0.log 2>/dev/null | tail -1; pgrep -cf '[r]un_sri_formal.py'" 2>/dev/null)
  if echo "$st" | head -1 | grep -qx '[1-9]'; then
    echo "=== CHAIN FINISHED (poll $i) ==="; echo "$st"; exit 0
  fi
done
echo "=== watcher budget exhausted ==="
