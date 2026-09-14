"""End-to-end smoke test of the B2 joint-probe runner on synthetic rows.

Both arms share the same events (as they do in reality) and store LEGACY row
labels (Y=1 iff the presented candidate is the true one), so this also exercises
the label-convention flip.  Arm A's state carries the joint label; arm B's is
noise.  The calibration and audit blocks use disjoint events.
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


def _build_arm(seed, informative, n_cal=90, n_aud=90):
    rng = np.random.RandomState(seed)
    manifest = []
    rows = {"calib": [], "audit": []}
    for split, n, off in (("calib", n_cal, 0), ("audit", n_aud, 100000)):
        for i in range(n):
            # labels from a per-event seed, shared by both arms
            ys, yp = np.random.RandomState(1000 + off + i).randint(2, size=2)
            nodes = {"leaf": 10 + i, "a2": 20 + i, "a1": 30 + i, "root": 40 + i}
            pid = (1000 + off + i, nodes["root"], nodes["leaf"], nodes["a2"],
                   nodes["a1"])
            pos = {k: 100 + i for k in NODES}
            neg = {k: 200 + i for k in NODES}
            # presented == pos -> canonical 0 (legacy y = 1); else canonical 1
            pres = {k: (pos[k] if ys else neg[k]) for k in ("leaf", "a2")}
            pres.update({k: (pos[k] if yp else neg[k]) for k in ("a1", "root")})
            t_root = 10.0 + i
            manifest.append({
                "pair_id": list(pid), "root_event_id": pid[0], "t_root": t_root,
                "nodes": nodes, "pos_cand": pos, "neg_cand": neg,
                "presented": pres,
                "pos_future_event_id": {k: 500 + i for k in NODES},
                "future_event_time": {k: t_root + 0.5 for k in NODES},
                "candidate_seed": {k: 700 + i for k in NODES}})
            for line, (sk, pk) in LINES.items():
                leg_s = 1 if pres[sk] == pos[sk] else 0     # LEGACY convention
                leg_p = 1 if pres[pk] == pos[pk] else 0
                label = 2 * (1 - leg_s) + (1 - leg_p)       # canonical class
                for phys in PHYS[line]:
                    if informative:
                        Z = 0.1 * rng.randn(172)
                        Z[label] = 2.5
                    else:
                        Z = rng.randn(172)
                    ctx = np.zeros(80)
                    for a, b in ((0, 16), (24, 40), (48, 64)):
                        ctx[a:b] = 0.3 * i + seed       # model-dependent
                    for a, b in ((16, 24), (40, 48), (64, 80)):
                        ctx[a:b] = 0.5 + 0.01 * i       # shared
                    rows[split].append({
                        "pair_id": list(pid), "line": line, "phys": phys,
                        "ctx": ctx, "keep": Z, "rem": np.zeros(172),
                        "y_s": leg_s, "y_p": leg_p})
    d = {"audit": rows["audit"], "calib": rows["calib"], "head": [],
         "manifest": manifest,
         "meta": {"n_layers": 3, "n_neighbors": 5, "bs": 64,
                  "data_name": "uci", "dataset_hash": "abc",
                  "manifest_sha": "sha", "model_kind": "ours",
                  "row_label_kind": rj.LEGACY_LABEL_KIND,
                  "layout": {"audit_block": [1, 2]},
                  "arm": "ours" if informative else "taskonly",
                  "seed": 0,
                  "ckpt_sha256": "aaaa" if informative else "bbbb"}}
    return d


def _write(tmp_path, E, cutoff=0.0):
    a = _build_arm(1, informative=True)
    b = _build_arm(2, informative=False)
    pa, pb = tmp_path / "a.pkl", tmp_path / "b.pkl"
    for p, d in ((pa, a), (pb, b)):
        with open(p, "wb") as f:
            pickle.dump(d, f)
    pe = tmp_path / "E.npz"
    np.savez(pe, E=E, svd_rank=int(E.shape[1]),
             embedding_hash=rj._sha16(E), cutoff_time=cutoff)
    return pa, pb, pe


def _run(pa, pb, pe, out, extra=()):
    sys.argv = ["retention_probe_joint_run.py", "--rows", str(pa), str(pb),
                "--encoder-npz", str(pe), "--out-dir", str(out),
                "--n-boot", "150", "--n-blocks", "10"] + list(extra)
    return run.main()


def test_runner_end_to_end(tmp_path, capsys):
    rng = np.random.RandomState(0)
    E = rng.randn(400, 16)
    pa, pb, pe = _write(tmp_path, E)
    out = tmp_path / "out"
    rep = _run(pa, pb, pe, out)
    assert rep["schema"]["ok"]
    assert rep["labels"] == ["ours|seed0|aaaa", "taskonly|seed0|bbbb"]
    assert rep["schema"]["label_kinds"] == [rj.LEGACY_LABEL_KIND] * 2
    leaf = rep["arms"]["Y_leaf"]
    assert set(leaf) == {"ours|seed0|aaaa", "taskonly|seed0|bbbb"}
    ours = leaf["ours|seed0|aaaa"]
    other = leaf["taskonly|seed0|bbbb"]
    assert ours["RootGain_millibits"] > 100, ours
    assert abs(other["RootGain_millibits"]) < 100, other
    # R at the source position is identically 1
    assert abs(ours["positions"][0]["R"] - 1.0) < 1e-9
    # negative or unidentifiable ratios are reported, never clipped
    for e in ours["positions"].values():
        assert e["R"] == "ratio_not_identifiable" or np.isfinite(e["R"])
    # the CV is out-of-fold and the probe reports whether it fell back to base
    assert ours["positions"][0]["probe"]["cv_oof_nll"] > 0
    assert isinstance(ours["positions"][0]["probe"]["used_base_fallback"], bool)
    # paired ours-vs-other RootGain differences with Holm over the three hops
    diff = rep["rootgain_paired"]["ours|seed0|aaaa - taskonly|seed0|bbbb"]
    assert set(diff) == {"1", "2", "3"}
    assert diff["3"]["diff_point"] > 0.1
    assert diff["3"]["p_holm"] < 0.05 and diff["3"]["significant_05"]
    assert (out / "joint_probe.json").exists()
    assert (out / "joint_probe_rows.pkl").exists()
    with open(out / "joint_probe_rows.pkl", "rb") as f:
        tab = pickle.load(f)
    loss = tab["row_losses"][("ours|seed0|aaaa", "Y_leaf", 0)]
    j = (loss["nll_base"].mean() - loss["nll_full"].mean()) / np.log(2.0)
    assert abs(j - ours["positions"][0]["J_point"]) < 1e-9


def test_runner_refuses_on_schema_mismatch(tmp_path):
    rng = np.random.RandomState(0)
    pa, pb, pe = _write(tmp_path, rng.randn(400, 16))
    with open(pb, "rb") as f:
        b = pickle.load(f)
    b["manifest"][0]["pos_future_event_id"]["root"] = 987654
    with open(pb, "wb") as f:
        pickle.dump(b, f)
    try:
        _run(pa, pb, pe, tmp_path / "o")
        raise AssertionError("expected SystemExit on schema mismatch")
    except SystemExit as e:
        assert "schema audit FAILED" in str(e)


def test_runner_refuses_a_tampered_encoder(tmp_path):
    rng = np.random.RandomState(0)
    E = rng.randn(400, 16)
    pa, pb, pe = _write(tmp_path, E)
    np.savez(pe, E=E, svd_rank=16, embedding_hash="deadbeef", cutoff_time=0.0)
    try:
        _run(pa, pb, pe, tmp_path / "o")
        raise AssertionError("expected SystemExit on encoder mismatch")
    except SystemExit as e:
        assert "encoder verification FAILED" in str(e)


def test_runner_refuses_encoder_cutoff_after_calib(tmp_path):
    rng = np.random.RandomState(0)
    E = rng.randn(400, 16)
    pa, pb, pe = _write(tmp_path, E, cutoff=99999.0)   # after every calib row
    try:
        _run(pa, pb, pe, tmp_path / "o")
        raise AssertionError("expected SystemExit on late encoder cutoff")
    except SystemExit as e:
        assert "cutoff" in str(e)


def test_runner_refuses_without_future_times(tmp_path):
    """Without manifest future_event_time and without a dataset, purge is
    impossible and the runner must refuse rather than silently skip it."""
    rng = np.random.RandomState(0)
    pa, pb, pe = _write(tmp_path, rng.randn(400, 16))
    for p in (pa, pb):
        with open(p, "rb") as f:
            d = pickle.load(f)
        for m in d["manifest"]:
            m.pop("future_event_time", None)
        with open(p, "wb") as f:
            pickle.dump(d, f)
    try:
        _run(pa, pb, pe, tmp_path / "o")
        raise AssertionError("expected RuntimeError without future times")
    except RuntimeError as e:
        assert "future_event_time" in str(e) or "purge" in str(e)
