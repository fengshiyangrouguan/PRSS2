"""Scan the Claude Code transcripts for rollout results with their context.

The recovered_rollout.txt dump was de-duplicated and lost the run labels, so
this walks the raw .jsonl again and prints every place a rollout number appears
together with the text around it (command line, checkpoint path, run name).
"""
import json
import os
import re
import sys

DIR = r"C:\Users\unski\.claude\projects\D--RPBE"
PAT = re.compile(sys.argv[1] if len(sys.argv) > 1 else r"weighted_success")
WIN = int(sys.argv[2]) if len(sys.argv) > 2 else 260


def texts(obj):
    """Yield every string inside a nested JSON object."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from texts(v)


def main():
    seen = set()
    for fn in sorted(os.listdir(DIR)):
        if not fn.endswith(".jsonl"):
            continue
        path = os.path.join(DIR, fn)
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln, line in enumerate(f, 1):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                for s in texts(rec):
                    for m in PAT.finditer(s):
                        a = max(0, m.start() - WIN)
                        b = min(len(s), m.end() + WIN)
                        key = re.sub(r"\s+", " ", s[a:b])[:200]
                        if key in seen:
                            continue
                        seen.add(key)
                        print(f"\n=== {fn}:{ln} ===")
                        print(re.sub(r"\s+", " ", s[a:b]))


main()
