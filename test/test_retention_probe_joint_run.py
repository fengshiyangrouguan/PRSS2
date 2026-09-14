"""End-to-end smoke test of the B2 joint-probe runner on synthetic rows.

Both arms share the same events (as they do in reality) and store LEGACY row
labels (Y=1 iff the presented candidate is the true one), so this also exercises
the label-convention flip.  The four candidate bits are assigned from a
deterministic per-index pattern so every joint class is covered early in the
block, which is what the class-coverage gate requires.  Arm A's state carries
the joint label; arm B's is noise.  Calibration and audit use disjoint events.
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


def _bits(i):
    """Deterministic per-index candidate bits covering all four classes fast."""
    return {"leaf": i % 2, "a2": (i // 2) % 2, "a1": (i // 4) % 2,
            "root": (i // 8) % 2}


def _build_arm(noise_seed, informative, arm, sha, seed=0, n_cal=90, n_aud=90):
    rng = np.random.RandomState(noise_seed)
    manifest = []
    rows = {"calib": [], "audit": []}
    for split, n, off in (("calib", n_cal, 0), ("audit", n_aud, 100000)):
        for i in range(n):
            bits = _bits(i)
            nodes = {"leaf": 10 + i, "a2": 20 + i, "a1": 30 + i, "root": 40 + i}
            pid = (1000 + off + i, nodes["root"], nodes["leaf"], nodes["a2"],
                   nodes["a1"])
            pos = {k: 100 + i for k in NODES}
            neg = {k: 200 + i for k in NODES}
            # canonical label = position of the true candidate
            #   presented == pos -> canonical 0  ;  presented == neg -> canonical 1
            pres = {k: (pos[k] if bits[k] == 0 else neg[k]) for k in NODES}
            t_root = 10.0 + i
            manifest.append({
                "pair_id": list(pid), "root_event_id": pid[0], "t_root": t_root,
                "nodes": nodes, "pos_cand": pos, "neg_cand": neg,
                "presented": pres,
                "pos_future_event_id": {k: 500 + i for k in NODES},
                "future_event_time": {k: t_root + 0.5 for k in NODES},
                "candidate_seed": {k: 700 + i for k in NODES}})
            for line, (sk, pk) in LINES.items():
                label = 2 * bits[sk] + bits[pk]          # joint class
                leg_s = 1 - bits[sk]                     # LEGACY convention
                leg_p = 1 - bits[pk]
                for phys in PHYS[line]:
                    if informative:
                        Z = 0.1 * rng.randn(172)
                        Z[label] = 2.5
                    else:
                        Z = rng.randn(172)
                    ctx = np.zeros(80)
                    for a, b in ((0, 16), (24, 40), (48, 64)):
                        ctx[a:b] = 0.3 * i + noise_seed   # model-dependent
                    for a, b in ((16, 24), (40, 48), (64, 80)):
                        ctx[a:b] = 0.5 + 0.01 * i         # shared
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
                  "arm": arm, "seed": seed, "ckpt_sha256": sha}}
    return d


def _write(tmp_path, E, arms, cutoff=0.0, name="E.npz"):
    paths = []
    for i, d in enumerate(arms):
        p = tmp_path / ("arm%d.pkl" % i)
        with open(p, "wb") as f:
            pickle.dump(d, f)
        paths.append(str(p))
    pe = tmp_path / name
    np.savez(pe, E=E, svd_rank=int(E.shape[1]),
             embedding_hash=rj._sha16(E), cutoff_time=cutoff,
             dataset_hash="abc")        # matches the pkl meta
    return paths, str(pe)


def _run(paths, pe, out, extra=()):
    sys.argv = ["retention_probe_joint_run.py", "--rows"] + list(paths) + \
        ["--encoder-npz", str(pe), "--out-dir", str(out),
         "--n-boot", "150", "--n-blocks", "10"] + list(extra)
    return run.main()


def _two_arm(tmp_path, E):
    a = _build_arm(1, informative=True, arm="ours", sha="aaaa")
    b = _build_arm(2, informative=False, arm="taskonly", sha="bbbb")
    return _write(tmp_path, E, [a, b])


def test_runner_end_to_end(tmp_path, capsys):
    rng = np.random.RandomState(0)
    E = rng.randn(400, 16)
    paths, pe = _two_arm(tmp_path, E)
    out = tmp_path / "out"
    rep = _run(paths, pe, out)
    assert rep["schema"]["ok"]
    assert rep["labels"] == ["ours|seed0|aaaa", "taskonly|seed0|bbbb"]
    assert rep["schema"]["label_kinds"] == [rj.LEGACY_LABEL_KIND] * 2
    leaf = rep["arms"]["Y_leaf"]
    ours = leaf["ours|seed0|aaaa"]
    other = leaf["taskonly|seed0|bbbb"]
    assert ours["class_counts_calib"] == [22, 22, 23, 23] or \
        all(c > 0 for c in ours["class_counts_calib"])
    assert all(c > 0 for c in ours["class_counts_audit"])
    assert ours["RootGain_millibits"] > 100, ours
    assert abs(other["RootGain_millibits"]) < 100, other
    assert abs(ours["positions"][0]["retention_ratio"] - 1.0) < 1e-9
    for e in ours["positions"].values():
        rr = e["retention_ratio"]
        assert rr == "ratio_not_identifiable" or np.isfinite(rr)
    # lambda handling is recorded and no candidate was silently dropped
    assert ours["positions"][0]["probe"]["n_invalid_lambda"] >= 0
    assert isinstance(ours["positions"][0]["probe"]["lambda_trials"], list)
    # pre-registered, same-seed, one-direction comparison only
    assert list(rep["comparisons"]) == [
        "ours|seed0|aaaa - taskonly|seed0|bbbb (seed 0)"]
    hops = rep["comparisons"]["ours|seed0|aaaa - taskonly|seed0|bbbb (seed 0)"]
    assert set(hops) == {"1", "2", "3"}
    assert hops["3"]["diff_point"] > 0.1
    assert hops["3"]["p_signflip"] < 0.05 and hops["3"]["significant_05"]
    across = rep["comparisons_across_seed"]["ours - taskonly"]
    assert across["3"]["n_seeds"] == 1
    assert across["3"]["ci_claimed"] is False
    assert (out / "joint_probe.json").exists()
    assert (out / "joint_probe_rows.pkl").exists()
    with open(out / "joint_probe_rows.pkl", "rb") as f:
        tab = pickle.load(f)
    # the sign-flip p-value is computed from the RAW paired audit rows
    lo_ours = tab["row_losses"][("ours|seed0|aaaa", "Y_leaf", 3)]
    lo_oth = tab["row_losses"][("taskonly|seed0|bbbb", "Y_leaf", 3)]
    d_rows = (lo_oth["nll_full"] - lo_ours["nll_full"]) / np.log(2.0)
    blocks = np.asarray(ours["audit_block_ids"])
    assert len(blocks) == len(d_rows)
    p_check = rj.block_signflip_p(d_rows, blocks, n_perm=2000, seed=0 + 3,
                                  alternative="greater")
    assert abs(p_check - hops["3"]["p_signflip"]) < 1e-12, \
        (p_check, hops["3"]["p_signflip"])
    loss = tab["row_losses"][("ours|seed0|aaaa", "Y_leaf", 0)]
    j = (loss["nll_base"].mean() - loss["nll_full"].mean()) / np.log(2.0)
    assert abs(j - ours["positions"][0]["J_point"]) < 1e-9


def test_comparisons_are_same_seed_and_one_direction(tmp_path):
    rng = np.random.RandomState(4)
    E = rng.randn(400, 16)
    arms = [
        _build_arm(1, True, "ours", "aaa1", seed=0),
        _build_arm(2, False, "taskonly", "bbb1", seed=0),
        _build_arm(3, True, "ours", "aaa2", seed=1),
        _build_arm(4, False, "taskonly", "bbb2", seed=1),
    ]
    paths, pe = _write(tmp_path, E, arms)
    rep = _run(paths, pe, tmp_path / "out")
    keys = sorted(rep["comparisons"])
    assert keys == ["ours|seed0|aaa1 - taskonly|seed0|bbb1 (seed 0)",
                    "ours|seed1|aaa2 - taskonly|seed1|bbb2 (seed 1)"], keys
    for k in keys:                       # never the reverse direction
        assert not k.startswith("taskonly")
    # cross-seed aggregation is reported separately, one delta per seed
    across = rep["comparisons_across_seed"]["ours - taskonly"]
    assert across["3"]["n_seeds"] == 2
    assert across["3"]["ci_claimed"] is False
    per_seed = {x["seed"]: x["diff_point"] for x in across["3"]["per_seed"]}
    assert set(per_seed) == {0, 1}
    # the per-seed deltas are exactly the same-seed comparison point diffs --
    # no rows were pooled across seeds
    for k, hops in rep["comparisons"].items():
        seed = 0 if "(seed 0)" in k else 1
        assert abs(per_seed[seed] - hops["3"]["diff_point"]) < 1e-12
    assert abs(across["3"]["mean_diff_point"]
               - np.mean(list(per_seed.values()))) < 1e-12


class _Stub:
    pass


def _stub_ds(edge_idxs, timestamps):
    s = _Stub()
    s.edge_idxs = np.asarray(edge_idxs, np.int64)
    s.timestamps = np.asarray(timestamps, np.float64)
    d = _Stub()
    d.full = s
    return d


def test_eid_time_map_allows_padding_id_zero():
    """0 may be padding with real edges from 1; ids need not be contiguous."""
    ds = _stub_ds([1, 2, 3, 7], [10.0, 20.0, 30.0, 70.0])
    m = run.build_eid_time_map(ds)
    assert m.size == 8
    assert np.isnan(m[0])                     # padding slot is unassigned
    assert m[1] == 10.0 and m[7] == 70.0
    man = {"pos_future_event_id": {"leaf": 1, "a2": 2}}
    assert run._future_end(man, "leaf", "a2", m) == 20.0
    # an id the stream never produced must be an error, not a silent 0.0
    bad = {"pos_future_event_id": {"leaf": 0, "a2": 3}}
    try:
        run._future_end(bad, "leaf", "a2", m)
        raise AssertionError("expected RunnerError for an unassigned id")
    except run.RunnerError as e:
        assert "never occurred" in str(e) or "not an edge id" in str(e)


def test_eid_time_map_rejects_duplicate_ids():
    ds = _stub_ds([1, 1, 2], [10.0, 20.0, 30.0])
    try:
        run.build_eid_time_map(ds)
        raise AssertionError("expected RunnerError for duplicate edge ids")
    except run.RunnerError as e:
        assert "bijection" in str(e)


def test_runner_refuses_on_schema_mismatch(tmp_path):
    rng = np.random.RandomState(0)
    paths, pe = _two_arm(tmp_path, rng.randn(400, 16))
    with open(paths[1], "rb") as f:
        b = pickle.load(f)
    b["manifest"][0]["pos_future_event_id"]["root"] = 987654
    with open(paths[1], "wb") as f:
        pickle.dump(b, f)
    try:
        _run(paths, pe, tmp_path / "o")
        raise AssertionError("expected SystemExit on schema mismatch")
    except SystemExit as e:
        assert "schema audit FAILED" in str(e)


def test_runner_refuses_a_tampered_encoder(tmp_path):
    rng = np.random.RandomState(0)
    E = rng.randn(400, 16)
    paths, pe = _two_arm(tmp_path, E)
    np.savez(pe, E=E, svd_rank=16, embedding_hash="deadbeef", cutoff_time=0.0)
    try:
        _run(paths, pe, tmp_path / "o")
        raise AssertionError("expected SystemExit on encoder mismatch")
    except SystemExit as e:
        assert "encoder verification FAILED" in str(e)


def test_runner_requires_a_dataset_hash_on_the_encoder(tmp_path):
    """An external table that does not record dataset_hash is not acceptable."""
    rng = np.random.RandomState(0)
    E = rng.randn(400, 16)
    paths, pe = _two_arm(tmp_path, E)
    np.savez(pe, E=E, svd_rank=16, embedding_hash=rj._sha16(E),
             cutoff_time=0.0)          # dataset_hash deliberately absent
    try:
        _run(paths, pe, tmp_path / "o")
        raise AssertionError("expected SystemExit without dataset_hash")
    except SystemExit as e:
        assert "dataset_hash" in str(e)


def test_runner_refuses_encoder_cutoff_after_calib(tmp_path):
    rng = np.random.RandomState(0)
    E = rng.randn(400, 16)
    paths, pe = _two_arm(tmp_path, E)
    np.savez(pe, E=E, svd_rank=16, embedding_hash=rj._sha16(E),
             cutoff_time=99999.0)
    try:
        _run(paths, pe, tmp_path / "o")
        raise AssertionError("expected SystemExit on late encoder cutoff")
    except SystemExit as e:
        assert "cutoff" in str(e)


def test_runner_refuses_without_future_times(tmp_path):
    rng = np.random.RandomState(0)
    E = rng.randn(400, 16)
    paths, pe = _two_arm(tmp_path, E)
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        for m in d["manifest"]:
            m.pop("future_event_time", None)
        with open(p, "wb") as f:
            pickle.dump(d, f)
    try:
        _run(paths, pe, tmp_path / "o")
        raise AssertionError("expected RunnerError without future times")
    except run.RunnerError as e:
        assert "future_event_time" in str(e)


def test_runner_refuses_missing_positions(tmp_path):
    """A pair missing an expected position is an error, not a silent prune."""
    rng = np.random.RandomState(0)
    E = rng.randn(400, 16)
    paths, pe = _two_arm(tmp_path, E)
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        d["audit"] = [r for r in d["audit"]
                      if not (r["line"] == "Y_leaf" and r["phys"] == 2)]
        with open(p, "wb") as f:
            pickle.dump(d, f)
    try:
        _run(paths, pe, tmp_path / "o")
        raise AssertionError("expected RunnerError on missing positions")
    except run.RunnerError as e:
        assert "expected positions" in str(e)


def test_runner_refuses_missing_joint_class(tmp_path):
    """Dropping a joint class must fail rather than fit a separable problem."""
    rng = np.random.RandomState(0)
    E = rng.randn(400, 16)
    arms = [_build_arm(1, True, "ours", "aaaa"),
            _build_arm(2, False, "taskonly", "bbbb")]
    # keep only even per-block indices, i.e. bits leaf=0, so the Y_leaf line
    # only ever shows joint classes 0 and 1
    def _even(pid):
        return (int(pid[0]) % 100000) % 2 == 0

    for d in arms:
        for s in ("calib", "audit"):
            d[s] = [r for r in d[s] if _even(r["pair_id"])]
        d["manifest"] = [m for m in d["manifest"] if _even(m["pair_id"])]
    paths, pe = _write(tmp_path, E, arms)
    try:
        _run(paths, pe, tmp_path / "o")
        raise AssertionError("expected class-coverage failure")
    except (rj.ProbeFitError, run.RunnerError) as e:
        assert "missing" in str(e), str(e)
