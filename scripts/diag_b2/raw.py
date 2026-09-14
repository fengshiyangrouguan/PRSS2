import sys, pickle
sys.path[:0] = ["/root/autodl-tmp/PRSS2_ret_tgb/scripts", "/root/autodl-tmp/PRSS2_ret_tgb/src"]
import numpy as np
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
print("encoder", E.shape, "cutoff", cut)

def idxmap(split, line):
    out = {}
    for r in d[split]:
        if r["line"] != line:
            continue
        out.setdefault(rj.pair_key(tuple(int(x) for x in r["pair_id"])), {})[int(r["phys"])] = r
    return out

def arrs(im, keys, sk, pk, phys):
    sp, pp, ys, yp, C = [], [], [], [], []
    for k in keys:
        rows = im[k]
        m = man[tuple(int(x) for x in rows[allp[0]]["pair_id"])]
        a, b, c, e = rj.ordered_pair_ids(m, sk, pk)
        sp.append(a); pp.append(b); ys.append(c); yp.append(b and e)
        C.append(ars.ctx_shared(rows[allp[0]]["ctx"]))
    Phi = rj.phi_tensor(E, np.array(sp), np.array(pp))
    y = rj.joint_class(np.array(ys), np.array(yp))
    Z = {p: np.stack([np.asarray(im[k][p]["keep"], np.float64) for k in keys]) for p in phys}
    return Phi, np.stack(C), y, Z

for line in ("Y_leaf", "Y_a2", "Y_a1"):
    sk, pk = LN[line]
    ic, ia = idxmap("calib", line), idxmap("audit", line)
    allp = sorted({p for v in ic.values() for p in v} & {p for v in ia.values() for p in v})
    ck = sorted(k for k, v in ic.items() if set(v) == set(allp))
    ak = sorted(k for k, v in ia.items() if set(v) == set(allp))
    if not ck or not ak:
        print(line, "no aligned pairs"); continue
    Phic, Cc, yc, Zc = arrs(ic, ck, sk, pk, allp)
    Phia, Ca, ya, Za = arrs(ia, ak, sk, pk, allp)
    base = rj.fit_joint(Phic, Cc, None, yc, 1e-2, use_state=False)
    nb = float(rj.joint_row_nll(base, Phia, Ca, None, ya).mean())
    print()
    print("== %s  calib_pairs=%d audit_pairs=%d  base audit NLL=%.6f (uniform ln4=%.4f)"
          % (line, len(ck), len(ak), nb, np.log(4)))
    for p in allp:
        best = None
        for lam in (1e-3, 1e-2, 1e-1, 1.0):
            f = rj.fit_joint(Phic, Cc, Zc[p], yc, lam, use_state=True)
            v = float(rj.joint_row_nll(f, Phia, Ca, Za[p], ya).mean())
            if best is None or v < best[1]:
                best = (lam, v)
        lam, v = best
        print("   phys%d  FULL audit NLL=%.6f (lam=%g)  vs base %.6f   raw J=%+.6f bits"
              % (p, v, lam, nb, (nb - v) / np.log(2.0)))
