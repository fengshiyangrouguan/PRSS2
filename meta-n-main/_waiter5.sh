#!/bin/bash
# The batch2 scheduler was killed while s104 was in flight, so s105 would never
# start. Wait for s104 to exit, then run s105 with its PRE-REGISTERED name.
#
# The skip pattern uses `s6dev_r[8]` so this script's own command line (which
# contains the literal text) cannot match itself and spin forever.
for i in $(seq 1 600); do
  if ! pgrep -f "exp-name s6dev_r[8]" > /dev/null 2>&1; then
    echo "[waiter5] s104 finished"; break
  fi
  sleep 30
done
sleep 15
cd /root/autodl-tmp/rpbe-sri/meta-n-main
exec /root/miniconda3/bin/python -u -m scripts.run_phase_a_structural6 \
  --out-root /root/autodl-tmp/sri_s6_phasea \
  --bench-data-dir /root/autodl-tmp/meta-n-main/data/co_bench \
  --seeds 105 --exp-names s6dev_r1 --execute
