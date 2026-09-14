import sys, pickle
sys.path[:0] = ["/root/autodl-tmp/PRSS2_ret_tgb/scripts", "/root/autodl-tmp/PRSS2_ret_tgb/src"]
import numpy as np
from sklearn.linear_model import LogisticRegression
import retention_probe_joint as rj
import audit_rows_schema as ars
from rpbe.data.uci_link import UCILinkDataset
ROWS = "/root/autodl-tmp/PRSS2_uci_v2/outputs/uci_formal_v2/seed0_TGN_3hop/ours/retention_rows.pkl"
d = pickle.load(open(ROWS, "rb"))
man = {tuple(int(x) for x in m["pair_id"]): m for m in d["manifest"]}
LN = {"Y_leaf": ("leaf", "a2"), "Y_a2": ("a2", "a1"), "Y_a1": ("a1", "root")}
def idxmap(split, line):
    out = {}
    for r in d[split]:
        if r["line"] != line: continue
        out.setdefault(rj.pair_key(tuple(int(x) for x in r["pair_id"])), {})[int(r["phys"])] = r
    return out
def nll(Xtr, ytr, Xte, yte):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(C=0.1, max_iter=400).fit((Xtr - m) / s, ytr)
    P = clf.predict_proba((Xte - m) / s)
    col = {int(c): j for j, c in enumerate(clf.classes_)}
    hit = np.array([col.get(int(t), -1) for t in yte]); ok = hit >= 0
    return float(-np.mean(np.log(np.clip(P[np.arange(len(yte))[ok], hit[ok]], 1e-12, 1))))
for line in ("Y_leaf",):
    sk, pk = LN[line]
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
    h1 = len(ck) // 2
    print("calib classes:", np.bincount(yc, minlength=4).tolist())
    print("audit classes:", np.bincount(ya, minlength=4).tolist())
    for p in allp:
        Zc = np.stack([np.asarray(ic[k][p]["keep"], np.float64) for k in ck])
        Za = np.stack([np.asarray(ia[k][p]["keep"], np.float64) for k in ak])
        print(" phys%d  Z mean|.| calib=%.4f audit=%.4f  |  Zstd calib=%.4f audit=%.4f  |  per-dim mean shift (rms)=%.4f"
              % (p, np.abs(Zc).mean(), np.abs(Za).mean(), Zc.std(), Za.std(),
                 np.sqrt(((Zc.mean(0) - Za.mean(0)) ** 2).mean())))
        n_in = nll(Zc[:h1], yc[:h1], Zc[h1:], yc[h1:])
        n_au = nll(Zc, yc, Za, ya)
        print("        within-calib heldout: NLL=%.5f gain=%+.5f bits   |  calib->audit: NLL=%.5f gain=%+.5f bits"
              % (n_in, (np.log(4) - n_in) / np.log(2), n_au, (np.log(4) - n_au) / np.log(2)))
