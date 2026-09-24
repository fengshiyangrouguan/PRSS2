#!/bin/bash
# Launch the Structural6 formal-shape SEARCH probe.
#
# Written as a FILE rather than an inline command because three rounds were lost
# to quoting/env drift in the inline form -- most expensively to a dropped
# META_N_EXTRA_HEADERS_JSON, which is what stops the relay from gzip-ing
# responses (the client's httpx2 decoder then dies with
# `process() takes no keyword arguments`). Everything is exported here, once.
set -euo pipefail

set -a
. /root/autodl-tmp/meta-n-main/.env
set +a

export LLM_BACKEND=relay
export ALLOW_PAID_API=YES_I_ACCEPT_REAL_COST
export RELAY_BASE_URL=https://api-key.xyz/api/v1
export META_N_EXTRA_HEADERS_JSON='{"Accept-Encoding": "identity"}'
export META_N_MAX_BACKEND_REQUESTS=400
export META_N_REQUEST_LEDGER=/root/autodl-tmp/sri_s6_probe/requests.jsonl
# NOT cleared: this script is also the RESUME path, and wiping the ledger would
# restart the request cap and hide what the first attempt already spent.

cd /root/autodl-tmp/rpbe-sri/meta-n-main
exec /root/miniconda3/bin/python -u _probe_formal_search.py --execute
