"""Tests for the rows/schema audit (scripts/audit_rows_schema.py).

Builds synthetic arm pkls whose SHARED fields agree but whose model-dependent
ctx blocks differ, then injects single faults and checks the auditor flags
exactly those.  Both the canonical and the legacy row-label conventions are
exercised, since existing extraction rows use the legacy one.
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


def _build_arm(n_pairs=4, mdep_shift=0.0, arm="a", seed=0, sha="deadbeef",
               label_kind="canonical", row_label_kind=None):
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
             "future_event_time": {k: 2.0 + i for k in NODES},
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
                true_at_0_s = m["presented"][sk] == m["pos_cand"][sk]
                true_at_0_p = m["presented"][pk] == m["pos_cand"][pk]
                if label_kind == "canonical":
                    ys = 0 if true_at_0_s else 1
                    yp = 0 if true_at_0_p else 1
                else:                                   # legacy extraction rows
                    ys = 1 if true_at_0_s else 0
                    yp = 1 if true_at_0_p else 0
                rows["audit"].append({
                    "pair_id": list(pid), "line": line, "phys": phys,
                    "ctx": ctx, "keep": np.zeros(172), "rem": np.zeros(172),
                    "y_s": ys, "y_p": yp, "pair_key": "k"})
    meta = {"n_layers": 3, "n_neighbors": 5, "bs": 64,
            "data_name": "uci", "dataset_hash": "abc",
            "manifest_sha": "sha", "model_kind": "ours",
            "arm": arm, "seed": seed, "ckpt_sha256": sha,
            "layout": {"audit_block": [1, 2]}}
    if row_label_kind is not None:
        meta["row_label_kind"] = row_label_kind
    return {"audit": rows["audit"], "calib": rows["calib"], "head": rows["head"],
            "manifest": manifest, "meta": meta}


def _mk_arms(label_kind="canonical", row_label_kind=None):
    a = _build_arm(mdep_shift=0.0, arm="ours", sha="aaaa",
                   label_kind=label_kind, row_label_kind=row_label_kind)
    b = _build_arm(mdep_shift=9.9, arm="taskonly", sha="bbbb",
                   label_kind=label_kind,
                   row_label_kind=row_label_kind)   # differs only in dropped ctx
    return a, b


def test_shared_fields_match_despite_model_dep_blocks():
    a, b = _mk_arms()
    rep = ars.full_audit([a, b])
    assert rep["ok"], (rep["problems"], rep["compare"]["mismatch_counts"])
    assert rep["compare"]["n_common_rows"] == 4 * 3 * 4
    assert rep["per_arm"][0]["label_kind"] == ars.rj.CANONICAL_LABEL_KIND
    assert rep["labels"] == ["ours|seed0|aaaa", "taskonly|seed0|bbbb"]
    # C_shared is the 32-d kept block
    recs, _ = ars.arm_records(a, ars.rj.CANONICAL_LABEL_KIND)
    assert recs[sorted(recs)[0]]["c_shared"].shape == (32,)


def test_legacy_rows_pass_without_a_flip():
    """Legacy extraction rows (Y=1 iff presented is the true one) must pass."""
    a, b = _mk_arms(label_kind="legacy")
    rep = ars.full_audit([a, b])
    assert rep["ok"], (rep["problems"], rep["compare"]["mismatch_counts"])
    assert rep["per_arm"][0]["label_kind"] == ars.rj.LEGACY_LABEL_KIND
    assert rep["per_arm"][0]["label_conflict"] == 0


def test_mixed_label_conventions_fail():
    a, b = _mk_arms()
    b["audit"] = [dict(r) for r in b["audit"]]
    for r in b["audit"]:                      # flip half the arm to legacy
        if r["phys"] == 0:
            r["y_s"], r["y_p"] = 1 - r["y_s"], 1 - r["y_p"]
    rep = ars.full_audit([a, b])
    assert not rep["ok"]


def test_presented_candidate_mismatch_flagged():
    a, b = _mk_arms()
    m = b["manifest"][0]
    m["presented"]["leaf"] = (m["neg_cand"]["leaf"]
                              if m["presented"]["leaf"] == m["pos_cand"]["leaf"]
                              else m["pos_cand"]["leaf"])
    rep = ars.full_audit([a, b])
    assert not rep["ok"]
    assert rep["compare"]["mismatch_counts"]["presented"] > 0
    assert rep["compare"]["mismatch_counts"]["ordered_s"] > 0


def test_future_event_id_mismatch_flagged():
    a, b = _mk_arms()
    b["manifest"][1]["pos_future_event_id"]["a2"] = 999999
    rep = ars.full_audit([a, b])
    assert not rep["ok"]
    assert rep["compare"]["mismatch_counts"]["future_event_id"] > 0


def test_future_event_time_mismatch_flagged():
    a, b = _mk_arms()
    b["manifest"][1]["future_event_time"]["a2"] = 123456.5
    rep = ars.full_audit([a, b])
    assert not rep["ok"]
    assert rep["compare"]["mismatch_counts"]["future_event_time"] > 0


def test_duplicate_manifest_pair_id_fails():
    a, b = _mk_arms()
    a["manifest"] = a["manifest"] + [dict(a["manifest"][0])]
    rep = ars.full_audit([a, b])
    assert not rep["ok"]
    assert any("manifest_duplicate_pair_ids" in p for p in rep["problems"]), \
        rep["problems"]


def test_label_conflict_detected_against_manifest():
    a, b = _mk_arms()
    a["audit"] = [dict(r) for r in a["audit"]]
    a["audit"][0]["y_s"] = 1 - a["audit"][0]["y_s"]   # inconsistent
    rep = ars.full_audit([a, b])
    assert not rep["ok"]
    assert any("label_conflict" in p for p in rep["problems"]), rep["problems"]


def test_row_without_manifest_is_reported():
    a, b = _mk_arms()
    a["manifest"] = [m for m in a["manifest"]
                     if tuple(m["pair_id"]) != _pair(2)[0]]
    rep = ars.full_audit([a, b])
    assert not rep["ok"]
    assert any("rows_missing_manifest" in p for p in rep["problems"])


def test_duplicate_row_key_and_split_overlap_fail():
    a, b = _mk_arms()
    a["audit"] = a["audit"] + [dict(a["audit"][0])]        # duplicate key
    rep = ars.full_audit([a, b])
    assert not rep["ok"]
    assert any("duplicate_row_keys" in p for p in rep["problems"])

    c, d = _mk_arms()
    c["calib"] = [dict(r) for r in c["audit"][:3]]         # same key, other split
    rep2 = ars.full_audit([c, d])
    assert not rep2["ok"]
    assert any("split_overlap" in p for p in rep2["problems"])


def test_missing_checkpoint_meta_and_duplicate_identity_fail():
    a, b = _mk_arms()
    b["meta"] = dict(b["meta"])
    del b["meta"]["ckpt_sha256"]
    rep = ars.full_audit([a, b])
    assert not rep["ok"]

    c, d = _mk_arms()
    d["meta"] = dict(d["meta"])
    d["meta"]["arm"], d["meta"]["ckpt_sha256"] = "ours", "aaaa"
    rep2 = ars.full_audit([c, d])
    assert not rep2["ok"]
    assert any("duplicate arm identities" in p for p in rep2["problems"])


def test_meta_layout_mismatch_fails():
    a, b = _mk_arms()
    b["meta"] = dict(b["meta"], layout={"audit_block": [9, 9]})
    rep = ars.full_audit([a, b])
    assert not rep["ok"]
    assert any("meta/layout mismatch" in p for p in rep["problems"])


def test_main_exit_code(tmp_path, capsys):
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
    b["manifest"][0]["candidate_seed"]["root"] = 123456
    with open(pb, "wb") as f:
        pickle.dump(b, f)
    sys.argv = ["audit_rows_schema.py", "--rows", str(pa), str(pb),
                "--out", str(out)]
    assert ars.main() == 1


def test_cli_can_declare_the_label_kind(tmp_path):
    import sys
    a, b = _mk_arms(label_kind="legacy")
    pa, pb = tmp_path / "a.pkl", tmp_path / "b.pkl"
    for p, d in ((pa, a), (pb, b)):
        with open(p, "wb") as f:
            pickle.dump(d, f)
    sys.argv = ["audit_rows_schema.py", "--rows", str(pa), str(pb),
                "--row-label-kind", "canonical",
                "--out", str(tmp_path / "r.json")]
    assert ars.main() == 1        # declared canonical but rows are legacy
