"""Count how many actions the policy server actually returned per request.

deploy.py's route does `print(answer)` where `answer` is the chunked list, so
the policy log literally contains the returned structure. Count the sub-lists
per response -- this is measured evidence, not config-string inference.
"""
import ast
import collections
import re
import sys
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/root/autodl-tmp/runs/t1_avg_seed42/rollout_t1_snapshot_15000.policy.log")

text = LOG.read_text(errors="replace")

# Each printed answer is a single line starting with '[['
answers = []
for line in text.splitlines():
    s = line.strip()
    if s.startswith("[[") and s.endswith("]]"):
        try:
            v = ast.literal_eval(s)
        except Exception:
            continue
        if isinstance(v, list) and v and isinstance(v[0], list):
            answers.append(v)

print(f"log            : {LOG}")
print(f"responses seen : {len(answers)}")
if not answers:
    print("!! no structured responses found")
    raise SystemExit(1)

lens = collections.Counter(len(a) for a in answers)
print(f"chunks/response histogram: {dict(lens)}")
widths = collections.Counter(len(r) for a in answers for r in a)
print(f"floats/chunk histogram   : {dict(widths)}")

n = len(answers)
tot_actions = sum(sum(len(r) for r in a) for a in answers)
print(f"total floats returned    : {tot_actions}")
print(f"mean chunks / response   : {sum(len(a) for a in answers)/n:.3f}")

# the key inference: how many env steps one response covers
per = sum(len(a) for a in answers) / n
print()
print(f"=> effective env-steps per request (if client executes all chunks): "
      f"{per:.2f}")
print("   official action_chunking_window is 8")

# also grab the reported episode bookkeeping
eps = [l for l in text.splitlines() if "ep " in l]
print(f"\n(server-side routes logged: "
      f"{len(re.findall(r'process_frame', text))} POSTs)")
