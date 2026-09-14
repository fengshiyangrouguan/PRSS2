import sys, pickle
sys.path[:0] = ["/root/autodl-tmp/PRSS2_ret_tgb/scripts", "/root/autodl-tmp/PRSS2_ret_tgb/src"]
import numpy as np
from sklearn.linear_model import LogisticRegression
import retention_probe_joint as rj
import audit_rows_schema as ars
from rpbe.data.uci_link import UCILinkDataset

ROWS = "/root/autodl-tmp/PRSS2_uci_v2/outputs/uci_formal_v2/seed0_TGN_3hop/ours/retention_rows.pkl"
DATA = "/root/autodl-tmp/bt_data/benchtemp_datasets/uci"
d = pickle.load(open(ROWS, "rb"))
man = {tuple(int(x) for x in m["pair_id"]): m for m in d["manifest"]}
LN = {"Y_leaf": ("leaf", "a2"), "Y_a2": ("a2", "a1"), "Y_a1": ("a1", "root")}
ds = UCILinkDataset(DATA, data_name="uci")
cut = min(float(man[tuple(int(x) for x in r["pair_id"])]["t_root"]) for r in d["calib"])
kp = np.asarray(ds.train.timestamps, np.float64) < cut
E, _ = rj.build_svd_encoder(ds.train.sources[kp], ds.train.destinations[kp], int(ds.n_nodes))

def idxmap(split, line):
    out = {}
    for r in d[split]:
        if r["line"] != line:
            continue
        out.setdefault(rj.pair_key(tuple(int(x) for x in r["pair_id"])), {})[int(r["phys"])] = r
    return out

def nll_of(Xtr, ytr, Xte, yte):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(C=0.1, max_iter=400).fit((Xtr - m) / s, ytr)
    P = clf.predict_proba((Xte - m) / s)
    col = {int(c): j for j, c in enumerate(clf.classes_)}
    hit = np.array([col.get(int(t), -1) for t in yte])
    ok = hit >= 0
    return float(-np.mean(np.log(np.clip(P[np.arange(len(yte))[ok], hit[ok]], 1e-12, 1))))

for line in ("Y_leaf", "Y_a2", "Y_a1"):
    sk, pk = LN[line]
    ic, ia = idxmap("calib", line), idxmap("audit", line)
    allp = sorted({p for v in ic.values() for p in v} & {p for v in ia.values() for p in v})
    ck = sorted(k for k, v in ic.items() if set(v) == set(allp))
    ak = sorted(k for k, v in ia.items() if set(v) == set(allp))
    if not ck or not ak: continue
    def lab(im):
        ys, yp = [], []
        for k in sorted(im):
            m = man[tuple(int(x) for x in im[k][allp[0]]["pair_id"])]
            a, b, c, e = rj.ordered_pair_ids(m, sk, pk)
            ys.append(c); yp.append(e)
        return rj.joint_class(np.array(ys), np.array(yp))
    yc, ya = lab({k: ic[k] for k in ck}), lab({k: ia[k] for k in ak})
    Cc = np.stack([ars.ctx_shared(ic[k][allp[0]]["ctx"]) for k in ck])
    Ca = np.stack([ars.ctx_shared(ia[k][allp[0]]["ctx"]) for k in ak])
    print()
    print("== %s  calib %d / audit %d   uniform ln4 = %.4f" % (line, len(ck), len(ak), np.log(4)))
    for p in allp:
        Zc = np.stack([np.asarray(ic[k][p]["keep"], np.float64) for k in ck])
        Za = np.stack([np.asarray(ia[k][p]["keep"], np.float64) for k in ak])
        n_z = nll_of(Zc, yc, Za, ya)
        n_cz = nll_of(np.hstack([Cc, Zc]), yc, np.hstack([Ca, Za]), ya)
        print("   phys%d   Z only: NLL=%.5f gain=%+.5f bits | [C,Z]: NLL=%.5f gain=%+.5f bits"
              % (p, n_z, (np.log(4) - n_z) / np.log(2), n_cz, (np.log(4) - n_cz) / np.log(2)))
