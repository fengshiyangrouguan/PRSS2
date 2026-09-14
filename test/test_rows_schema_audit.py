"""Tests for the rows/schema audit (scripts/audit_rows_schema.py).

Builds two synthetic arm pkls whose SHARED fields agree but whose
model-dependent ctx blocks differ, then injects single-field mutations and
checks the auditor flags exactly those.
"""

import pickle

import numpy as np

import audit_rows_schema as ars

NODES = ("leaf", "a2", "a1", "root")
LINES = {"Y_leaf": ("leaf", "a2"), "Y_a2": ("a2", "a1"), "Y_a1": ("a1", "root")}


def _pair(i):
    nodes = {"leaf": 10 + i, "a2": 20 + i, "a1": 30 + i, "root": 40 + i}
    pid = (1000 + i, nodes["root"], nodes["leaf"], nodes["a2"], nodes["a1"])
    return pid, nodes


def _build_arm(n_pairs=4, mdep_shift=0.0):
    manifest = []
    man_by = {}
    for i in range(n_pairs):
        pid, nodes = _pair(i)
        pos = {k: 100 + i for k in NODES}
        neg = {k: 200 + i for k in NODES}
        pres = {k: (pos[k] if (i + len(k)) % 2 == 0 else neg[k]) for k in NODES}
        m = {"pair_id": list(pid), "root_event_id": pid[0],
             "root_node": nodes["root"], "t_root": 1.0 + i, "nodes": nodes,
             "pos_cand": pos, "neg_cand": neg, "presented": pres,
             "pos_future_event_id": {k: 500 + i for k in NODES},
             "candidate_seed": {k: 700 + i for k in NODES}}
        manifest.append(m)
        man_by[pid] = m
    rows = {"calib": [], "audit": [], "head": []}
    for i in range(n_pairs):
        pid, _ = _pair(i)
        m = man_by[pid]
        for line, (sk, pk) in LINES.items():
            for phys in (0, 1, 2, 3):
                ctx = np.zeros(80)
                for a, b in ((0, 16), (24, 40), (48, 64)):
                    ctx[a:b] = mdep_shift + i          # model-dependent
                for a, b in ((16, 24), (40, 48), (64, 80)):
                    ctx[a:b] = 0.5 + i                 # shared
                ys = 0 if m["presented"][sk] == m["pos_cand"][sk] else 1
                yp = 0 if m["presented"][pk] == m["pos_cand"][pk] else 1
                rows["audit"].append({
                    "pair_id": list(pid), "line": line, "phys": phys,
                    "ctx": ctx, "keep": np.zeros(172), "rem": np.zeros(172),
                    "y_s": ys, "y_p": yp, "pair_key": "k"})
    d = {"audit": rows["audit"], "calib": rows["calib"], "head": rows["head"],
         "manifest": manifest,
         "meta": {"n_layers": 3, "n_neighbors": 5, "bs": 64,
                  "data_name": "uci", "dataset_hash": "abc",
                  "manifest_sha": "sha", "model_kind": "ours"}}
    return d


def _mk_arms():
    a = _build_arm(mdep_shift=0.0)
    b = _build_arm(mdep_shift=9.9)          # differs ONLY in dropped blocks
    return a, b


def _audit(ds, tmpdir):
    paths = []
    for i, d in enumerate(ds):
        p = tmpdir / ("arm%d.pkl" % i)
        with open(p, "wb") as f:
            pickle.dump(d, f)
        paths.append(str(p))
    recs, intras = [], []
    for p in paths:
        with open(p, "rb") as f:
            dd = pickle.load(f)
        r, it = ars.arm_records(dd)
        recs.append(r)
        intras.append(it)
    return recs, intras


def test_shared_fields_match_despite_model_dep_blocks(tmp_path):
    ds = _mk_arms()
    recs, intras = _audit(ds, tmp_path)
    assert all(i["rows_missing_manifest"] == 0 for i in intras)
    assert all(i["label_conflict"] == 0 for i in intras)
    rep = ars.compare(recs, ["a", "b"])
    assert rep["ok"], rep["mismatch_counts"]
    assert rep["n_common_rows"] == 4 * 3 * 4
    # the kept blocks are 32-d
    assert recs[0][sorted(recs[0])[0]]["c_shared"].shape == (32,)


def test_presented_candidate_mismatch_flagged(tmp_path):
    a, b = _mk_arms()
    pid, _ = _pair(0)
    b["manifest"][0]["presented"]["leaf"] = (
        b["manifest"][0]["neg_cand"]["leaf"]
        if b["manifest"][0]["presented"]["leaf"]
        == b["manifest"][0]["pos_cand"]["leaf"]
        else b["manifest"][0]["pos_cand"]["leaf"])
    recs, _ = _audit([a, b], tmp_path)
    rep = ars.compare(recs, ["a", "b"])
    assert not rep["ok"]
    assert rep["mismatch_counts"]["presented"] > 0
    assert rep["mismatch_counts"]["ordered_s"] > 0


def test_future_event_id_mismatch_flagged(tmp_path):
    a, b = _mk_arms()
    b["manifest"][1]["pos_future_event_id"]["a2"] = 999999
    recs, _ = _audit([a, b], tmp_path)
    rep = ars.compare(recs, ["a", "b"])
    assert not rep["ok"]
    assert rep["mismatch_counts"]["future_event_id"] > 0


def test_label_conflict_detected_intra_arm(tmp_path):
    a, b = _mk_arms()
    a["audit"][0]["y_s"] = 1 - a["audit"][0]["y_s"]   # inconsistent with manifest
    recs, intras = _audit([a, b], tmp_path)
    assert intras[0]["label_conflict"] > 0


def test_row_without_manifest_is_reported(tmp_path):
    a, b = _mk_arms()
    a["manifest"] = [m for m in a["manifest"]
                     if tuple(m["pair_id"]) != _pair(2)[0]]
    recs, intras = _audit([a, b], tmp_path)
    assert intras[0]["rows_missing_manifest"] > 0


def test_main_exit_code_and_report(tmp_path, capsys):
    a, b = _mk_arms()
    pa, pb = tmp_path / "a.pkl", tmp_path / "b.pkl"
    for p, d in ((pa, a), (pb, b)):
        with open(p, "wb") as f:
            pickle.dump(d, f)
    out = tmp_path / "rep.json"
    import sys
    sys.argv = ["audit_rows_schema.py", "--rows", str(pa), str(pb),
                "--out", str(out)]
    assert ars.main() == 0
    assert "OK" in capsys.readouterr().out
    # now inject a mismatch -> nonzero exit
    b["manifest"][0]["candidate_seed"]["root"] = 123456
    with open(pb, "wb") as f:
        pickle.dump(b, f)
    sys.argv = ["audit_rows_schema.py", "--rows", str(pa), str(pb),
                "--out", str(out)]
    assert ars.main() == 1
