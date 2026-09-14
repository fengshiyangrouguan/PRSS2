"""End-to-end smoke test of the B2 joint-probe runner on synthetic rows.

Builds two arms whose shared fields agree; arm A's state carries the joint
label, arm B's is noise.  Checks the runner reports a positive RootGain for A,
~0 for B, and a paired R that is identifiable.
"""

import json
import pickle
import sys

import numpy as np

import retention_probe_joint as rj
import retention_probe_joint_run as run

NODES = ("leaf", "a2", "a1", "root")
LINES = {"Y_leaf": ("leaf", "a2"), "Y_a2": ("a2", "a1"), "Y_a1": ("a1", "root")}
PHYS = {"Y_leaf": (0, 1, 2, 3), "Y_a2": (1, 2, 3), "Y_a1": (2, 3)}
ORIGIN = {"Y_leaf": 0, "Y_a2": 1, "Y_a1": 2}


def _build_arm(seed, informative, n_pairs=80, n_offset=0):
    rng = np.random.RandomState(seed)
    manifest = []
    rows = {"calib": [], "audit": []}
    for i in range(n_pairs):
        # labels come from a per-pair seed independent of the arm, so both arms
        # share the SAME manifest (as they do in reality)
        ys, yp = np.random.RandomState(1000 + i).randint(2, size=2)
        nodes = {"leaf": 10 + i, "a2": 20 + i, "a1": 30 + i, "root": 40 + i}
        pid = (1000 + n_offset + i, nodes["root"], nodes["leaf"], nodes["a2"],
               nodes["a1"])
        pos = {k: 100 + i for k in NODES}
        neg = {k: 200 + i for k in NODES}
        # presented == neg -> Y_v = 1 ; presented == pos -> Y_v = 0
        pres = {k: (neg[k] if ys else pos[k]) for k in ("leaf", "a2")}
        pres.update({k: (neg[k] if yp else pos[k]) for k in ("a1", "root")})
        manifest.append({
            "pair_id": list(pid), "root_event_id": pid[0], "t_root": 10.0 + i,
            "nodes": nodes, "pos_cand": pos, "neg_cand": neg,
            "presented": pres,
            "pos_future_event_id": {k: 500 + i for k in NODES},
            "candidate_seed": {k: 700 + i for k in NODES}})
        # state: informative arms encode the joint class, else pure noise
        for split, base_i in (("calib", i), ("audit", i)):
            for line, (sk, pk) in LINES.items():
                label = 2 * (0 if pres[sk] == pos[sk] else 1) + \
                    (0 if pres[pk] == pos[pk] else 1)
                for phys in PHYS[line]:
                    if informative:
                        Z = np.zeros(172)
                        Z[:4] = 0.0
                        Z[label] = 2.5
                        Z += 0.1 * rng.randn(172)
                    else:
                        Z = rng.randn(172)
                    ctx = np.zeros(80)
                    for a, b in ((0, 16), (24, 40), (48, 64)):
                        ctx[a:b] = 0.3 * i + seed      # model-dependent
                    for a, b in ((16, 24), (40, 48), (64, 80)):
                        ctx[a:b] = 0.5 + 0.01 * i      # shared
                    rows[split].append({
                        "pair_id": list(pid), "line": line, "phys": phys,
                        "ctx": ctx, "keep": Z, "rem": np.zeros(172),
                        "y_s": 0 if pres[sk] == pos[sk] else 1,
                        "y_p": 0 if pres[pk] == pos[pk] else 1})
    d = {"audit": rows["audit"], "calib": rows["calib"], "head": [],
         "manifest": manifest,
         "meta": {"n_layers": 3, "n_neighbors": 5, "bs": 64,
                  "data_name": "uci", "dataset_hash": "abc",
                  "manifest_sha": "sha", "model_kind": "ours"}}
    return d


def _write(tmp_path, a, b, E):
    pa, pb = tmp_path / "a.pkl", tmp_path / "b.pkl"
    for p, d in ((pa, a), (pb, b)):
        with open(p, "wb") as f:
            pickle.dump(d, f)
    pe = tmp_path / "E.npz"
    np.savez(pe, E=E, svd_rank=16, embedding_hash="deadbeef", cutoff_time=0.0)
    return pa, pb, pe


def test_runner_end_to_end(tmp_path, capsys):
    rng = np.random.RandomState(0)
    E = rng.randn(400, 16)
    a = _build_arm(1, informative=True, n_offset=0)
    b = _build_arm(2, informative=False, n_offset=0)   # same events, other ckpt
    pa, pb, pe = _write(tmp_path, a, b, E)
    out = tmp_path / "out"
    sys.argv = ["retention_probe_joint_run.py", "--rows", str(pa), str(pb),
                "--labels", "ours", "taskonly", "--encoder-npz", str(pe),
                "--out-dir", str(out), "--n-boot", "150", "--n-blocks", "10"]
    rep = run.main()
    assert rep["schema_ok"], rep.get("schema")
    leaf = rep["arms"]["Y_leaf"]
    assert set(leaf) == {"ours", "taskonly"}
    assert leaf["ours"]["RootGain_millibits"] > 100, leaf["ours"]
    assert abs(leaf["taskonly"]["RootGain_millibits"]) < 100, leaf["taskonly"]
    # R for the source position is identically 1
    rsrc = leaf["ours"]["positions"][0]["R"]
    assert rsrc == 1.0 or abs(rsrc - 1.0) < 1e-9, rsrc
    assert (out / "joint_probe.json").exists()
    assert (out / "joint_probe_rows.pkl").exists()
    # the row-loss table allows the summary to be recomputed without refitting
    with open(out / "joint_probe_rows.pkl", "rb") as f:
        tab = pickle.load(f)
    k = ("ours", "Y_leaf", 0)
    assert k in tab["row_losses"]
    loss = tab["row_losses"][k]
    j = (loss["nll_base"].mean() - loss["nll_full"].mean()) / np.log(2.0)
    assert abs(j - leaf["ours"]["positions"][0]["J_point"]) < 1e-9


def test_runner_refuses_on_schema_mismatch(tmp_path):
    rng = np.random.RandomState(0)
    E = rng.randn(400, 16)
    a = _build_arm(1, informative=True, n_offset=0)
    b = _build_arm(2, informative=True, n_offset=0)
    b["manifest"][0]["pos_future_event_id"]["root"] = 987654   # mismatch
    pa, pb, pe = _write(tmp_path, a, b, E)
    sys.argv = ["retention_probe_joint_run.py", "--rows", str(pa), str(pb),
                "--encoder-npz", str(pe), "--out-dir", str(tmp_path / "o")]
    try:
        run.main()
        raise AssertionError("expected SystemExit on schema mismatch")
    except SystemExit as e:
        assert "schema audit FAILED" in str(e)
