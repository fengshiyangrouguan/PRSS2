#!/bin/bash
# 远端采集脚本（部署到各卡 /root/watch_abl.sh）
# 用法: bash /root/watch_abl.sh <grp>    grp ∈ A1 | A2A6 | A3A7
# 输出为机器可解析的状态块：PROCS/ORCH 计数 + RUN| 行（完成的 summary）+ DONE 行
grp=${1:-A1}
cd /root/autodl-tmp/PRSS2_uci_v2/outputs/uci_formal_v2 || { echo "ERR=cd"; exit 9; }

echo "PROCS=$(pgrep -c -f '[p]ython -m scripts.train_uci_link')"
echo "ORCH=$(pgrep -c -f "[r]un_abl_${grp}")"
echo "ALLDONE=$(grep -c "ALL_DONE_ABL_${grp}" run_abl_${grp}.log 2>/dev/null)"
echo "NOW=$(date '+%m-%d %H:%M')"

for f in seed*_TGN_UCI_3L10N/*/summary.json; do
  [ -f "$f" ] || continue
  /root/miniconda3/bin/python - "$f" <<'PY'
import json, sys
p = sys.argv[1]
parts = p.split('/')
try:
    d = json.load(open(p))
except Exception as e:
    print("RUN|%s|%s|PARSE_ERR" % (parts[0], parts[1])); raise SystemExit
t = d.get('test', {}) or {}
fmt = lambda v: ("%.4f" % v) if isinstance(v, (int, float)) else str(v)
print("RUN|%s|%s|%s|%s|%s|%s|%s|%s" % (
    parts[0], parts[1], fmt(d.get('best_ap_all')), fmt(t.get('ap')),
    fmt(t.get('auc')), d.get('best_epoch'), d.get('stop_reason'),
    d.get('agg', '-')))
PY
done

echo "---DONES---"
grep -E "DONE |ALL_DONE|BUSY |RESET " run_abl_${grp}.log 2>/dev/null | tail -8
