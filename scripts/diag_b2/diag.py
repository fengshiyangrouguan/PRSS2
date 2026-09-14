import pickle, numpy as np
p = "/root/autodl-tmp/PRSS2_uci_v2/outputs/uci_formal_v2/seed0_TGN_3hop/ours/retention_rows.pkl"
d = pickle.load(open(p, "rb"))
rows = d["audit"]
print("audit rows", len(rows))
from collections import defaultdict
agg = defaultdict(list)
cls = defaultdict(lambda: np.zeros(4, int))
for r in rows[:40000]:
    k = (r["line"], int(r["phys"]))
    agg[k].append(np.asarray(r["keep"], np.float64))
    cls[k][2 * int(r["y_s"]) + int(r["y_p"])] += 1
for k in sorted(agg, key=lambda x: (x[0], x[1])):
    A = np.stack(agg[k])
    print("%-7s phys%d  n=%5d  keep mean|.|=%.4g  std=%.4g  frac_zero=%.3f  classes=%s"
          % (k[0], k[1], len(A), np.abs(A).mean(), A.std(),
             float((np.abs(A).sum(1) == 0).mean()), cls[k].tolist()))
