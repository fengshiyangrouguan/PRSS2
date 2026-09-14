import sys, pickle
sys.path[:0] = ["/root/autodl-tmp/PRSS2_ret_tgb/scripts", "/root/autodl-tmp/PRSS2_ret_tgb/src"]
import numpy as np
from sklearn.linear_model import LogisticRegression
import retention_probe_joint as rj
import audit_rows_schema as ars
ROWS = "/root/autodl-tmp/PRSS2_uci_v2/outputs/uci_formal_v2/seed0_TGN_3hop/ours/retention_rows.pkl"
d = pickle.load(open(ROWS, "rb"))
man = {tuple(int(x) for x in m["pair_id"]): m for m in d["manifest"]}
LN = {"Y_leaf": ("leaf", "a2"), "Y_a2": ("a2", "a1"), "Y_a1": ("a1", "root")}
def idxmap(s, line):
    o = {}
    for r in d[s]:
        if r["line"] != line: continue
        o.setdefault(rj.pair_key(tuple(int(x) for x in r["pair_id"])), {})[int(r["phys"])] = r
    return o
def nll(Xtr, ytr, Xte, yte, C=0.1):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(C=C, max_iter=400).fit((Xtr - m) / s, ytr)
    P = clf.predict_proba((Xte - m) / s)
    col = {int(c): j for j, c in enumerate(clf.classes_)}
    hit = np.array([col.get(int(t), -1) for t in yte]); ok = hit >= 0
    return float(-np.mean(np.log(np.clip(P[np.arange(len(yte))[ok], hit[ok]], 1e-12, 1))))
U = np.log(4)
line = "Y_leaf"; sk, pk = LN[line]
ic, ia = idxmap("calib", line), idxmap("audit", line)
allp = sorted({p for v in ic.values() for p in v} & {p for v in ia.values() for p in v})
ck = sorted(k for k, v in ic.items() if set(v) == set(allp))
ak = sorted(k for k, v in ia.items() if set(v) == set(allp))
def pack(keys, im):
    ys, yp, C, Z = [], [], [], []
    for k in keys:
        m = man[tuple(int(x) for x in im[k][allp[0]]["pair_id"])]
        a, b, c, e = rj.ordered_pair_ids(m, sk, pk); ys.append(c); yp.append(e)
        C.append(ars.ctx_shared(im[k][allp[0]]["ctx"]))
        Z.append(np.asarray(im[k][allp[0]]["keep"], np.float64))
    return rj.joint_class(np.array(ys), np.array(yp)), np.stack(C), np.stack(Z)
yc, Cc, Zc = pack(ck, ic)
ya, Ca, Za = pack(ak, ia)
print("n_calib=%d n_audit=%d  uniform ln4=%.4f" % (len(yc), len(ya), U))
print("prior-only (no features):", round(nll(Cc, yc, np.zeros((len(ya), 1)), ya), 5))
rng = np.random.RandomState(0)
for tag, Xc, Xa in (("C+cand-ish C (32d)", Cc, Ca), ("Z (172d)", Zc, Za)):
    for trial in range(3):
        ysh = rng.permutation(yc)
        print("  NULL(shuffled calib labels) %-18s trial%d: NLL=%.5f  gain=%+.5f bits"
              % (tag, trial, nll(Xc, ysh, Xa, ya), (U - nll(Xc, ysh, Xa, ya)) / np.log(2)))
    print("  REAL %-32s        : NLL=%.5f  gain=%+.5f bits"
          % (tag, nll(Xc, yc, Xa, ya), (U - nll(Xc, yc, Xa, ya)) / np.log(2)))
    for C_ in (1e-4, 1e-2, 1.0):
        print("     REAL %-18s C=%-6g: NLL=%.5f" % (tag, C_, nll(Xc, yc, Xa, ya, C=C_)))
