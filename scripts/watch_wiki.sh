#!/bin/bash
# Watch two training runs.  Usage: watch_wiki.sh <log1> <log2>
# Streams every 25th epoch line plus error/early-stop/completion lines to
# stdout and exits (echoing ALL_PROCESSES_EXITED) once both python training
# processes are gone.
tail -F "$1" "$2" 2>/dev/null \
  | grep -E --line-buffered 'epoch=|early_stop|Traceback|Error|Killed|OOM|inference_isolation|test_mrr' \
  | /root/miniconda3/bin/python -u -c "
import sys
for line in sys.stdin:
    if 'epoch=' in line:
        try:
            e = int(line.split('epoch=')[1].split(' ')[0])
            if e % 25 != 0:
                continue
        except (ValueError, IndexError):
            pass
    sys.stdout.write(line)
    sys.stdout.flush()
" &
TAIL_PID=$!
while pgrep -f '^/root/miniconda3/bin/python scripts/train.py' > /dev/null 2>&1; do
  sleep 10
done
sleep 2
kill $TAIL_PID 2>/dev/null
echo ALL_PROCESSES_EXITED
