import sys, pickle
sys.path[:0] = ["/root/autodl-tmp/PRSS2_ret_tgb/scripts", "/root/autodl-tmp/PRSS2_ret_tgb/src"]
import numpy as np
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
line = "Y_leaf"; sk, pk = LN[line]
ic, ia = idxmap("calib", line), idxmap("audit", line)
allp = sorted({p for v in ic.values() for p in v} & {p for v in ia.values() for p in v})
ck = sorted(k for k, v in ic.items() if set(v) == set(allp))
ak = sorted(k for k, v in ia.items() if set(v) == set(allp))
def lab(keys, im):
    ys, yp = [], []
    for k in keys:
        m = man[tuple(int(x) for x in im[k][allp[0]]["pair_id"])]
        a, b, c, e = rj.ordered_pair_ids(m, sk, pk); ys.append(c); yp.append(e)
    return rj.joint_class(np.array(ys), np.array(yp))
yc, ya = lab(ck, ic), lab(ak, ia)
print("nearest-centroid control (cannot overfit):")
for p in allp:
    Zc = np.stack([np.asarray(ic[k][p]["keep"], np.float64) for k in ck])
    Za = np.stack([np.asarray(ia[k][p]["keep"], np.float64) for k in ak])
    mu, sd = Zc.mean(0), Zc.std(0) + 1e-9
    C = (Zc - mu) / sd; A = (Za - mu) / sd
    cent = np.stack([C[yc == c].mean(0) for c in range(4)])
    def acc(Z, y):
        dist = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        return float((dist.argmin(1) == y).mean())
    # per-dim |corr| replication between blocks
    Zc0 = Zc - Zc.mean(0); Za0 = Za - Za.mean(0)
    cc = np.array([abs(np.corrcoef(Zc0[:, j], yc)[0, 1]) for j in range(Zc.shape[1])])
    ca = np.array([abs(np.corrcoef(Za0[:, j], ya)[0, 1]) for j in range(Za.shape[1])])
    top = np.argsort(-cc)[:10]
    print("  phys%d  centroid acc: calib-half2=%.4f  calib->audit=%.4f  (chance 0.25)  |  top-10 calib |corr| mean=%.4f, same dims on audit=%.4f, all-dim audit |corr| mean=%.4f"
          % (p, acc(C[len(ck)//2:], yc[len(ck)//2:]), acc(A, ya), cc[top].mean(), ca[top].mean(), ca.mean()))
