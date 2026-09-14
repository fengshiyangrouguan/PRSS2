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
def idxmap(s, line):
    o = {}
    for r in d[s]:
        if r["line"] != line: continue
        o.setdefault(rj.pair_key(tuple(int(x) for x in r["pair_id"])), {})[int(r["phys"])] = r
    return o
def nll(Xtr, ytr, Xte, yte):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(C=0.1, max_iter=400).fit((Xtr - m) / s, ytr)
    P = clf.predict_proba((Xte - m) / s)
    col = {int(c): j for j, c in enumerate(clf.classes_)}
    hit = np.array([col.get(int(t), -1) for t in yte]); ok = hit >= 0
    return float(-np.mean(np.log(np.clip(P[np.arange(len(yte))[ok], hit[ok]], 1e-12, 1))))
line = "Y_leaf"; sk, pk = LN[line]
ic, ia = idxmap("calib", line), idxmap("audit", line)
allp = sorted({p for v in ic.values() for p in v} & {p for v in ia.values() for p in v})
ck = sorted(k for k, v in ic.items() if set(v) == set(allp))
ak = sorted(k for k, v in ia.items() if set(v) == set(allp))
def pack(keys, im):
    ys, yp, C, S, P, Z0 = [], [], [], [], [], []
    for k in keys:
        m = man[tuple(int(x) for x in im[k][allp[0]]["pair_id"])]
        a, b, c, e = rj.ordered_pair_ids(m, sk, pk); ys.append(c); yp.append(e)
        C.append(ars.ctx_shared(im[k][allp[0]]["ctx"]))
        S.append(np.concatenate([E[a[0]], E[a[1]]])); P.append(np.concatenate([E[b[0]], E[b[1]]]))
        Z0.append(np.asarray(im[k][allp[0]]["keep"], np.float64))
    return (np.array(ys), np.array(yp), np.stack(C), np.stack(S), np.stack(P), np.stack(Z0))
yc, ypc, Cc, Sc, Pc, Zc = pack(ck, ic)
ya, ypa, Ca, Sa, Pa, Za = pack(ak, ia)
lbs = {"calib": rj.joint_class(yc, ypc), "audit": rj.joint_class(ya, ypa)}
U = np.log(4)
def show(tag, Xc, Xa):
    n = nll(Xc, lbs["calib"], Xa, lbs["audit"])
    print("   %-34s audit NLL=%.5f  gain=%+.5f bits" % (tag, n, (U - n) / np.log(2)))
print("== Y_leaf phys0 ; uniform ln4=%.4f ; 4-class" % U)
show("C only (32d)", Cc, Ca)
show("cand E(s) only (32d)", Sc, Sa)
show("cand E(p) only (32d)", Pc, Pa)
show("cand E(s)+E(p) (64d)", np.hstack([Sc, Pc]), np.hstack([Sa, Pa]))
show("C + cand (96d)  [= base feature]", np.hstack([Cc, Sc, Pc]), np.hstack([Ca, Sa, Pa]))
show("Z only (172d)", Zc, Za)
rng = np.random.RandomState(0)
pass
