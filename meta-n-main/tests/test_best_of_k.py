"""Tests for the best-of-K arm reducer (meta_n.analysis.best_of_k)."""

import json
from pathlib import Path

import pytest

from meta_n.analysis import best_of_k as bok


def _write_run(root: Path, name: str, *, oracle, best, test, chain, per_task, calls, status="completed"):
    d = root / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({
        "oracle_mean_score": oracle,
        "best_mean_score": best,
        "test_mean_score": test,
        "chain_test_mean_score": chain,
        "per_task_best_scores": per_task,
        "token_usage": {"outer_calls": calls, "inner_calls": 0},
        "run_status": status,
    }))
    return d


def test_load_run_extracts_metrics_and_compute(tmp_path):
    d = _write_run(tmp_path, "r", oracle=0.8, best=0.75, test=0.7, chain=0.65,
                   per_task={"a": 0.9, "b": 0.7}, calls=12)
    r = bok.load_run(d)
    assert r.best_of_search_dev == 0.8 and r.deployable_dev == 0.75
    assert r.best_of_search_test == 0.7 and r.deployable_test == 0.65
    assert r.per_task == {"a": 0.9, "b": 0.7} and r.calls == 12


def test_summary_found_via_exp_subdir(tmp_path):
    _write_run(tmp_path / "out", "smoke", oracle=0.5, best=0.5, test=0.5, chain=0.5,
               per_task={"a": 0.5}, calls=3)
    r = bok.load_run(tmp_path / "out")  # parent dir with a single run under it
    assert r.best_of_search_dev == 0.5


def test_best_of_k_picks_the_best_run(tmp_path):
    runs = [
        bok.load_run(_write_run(tmp_path, "s42", oracle=0.60, best=0.55, test=0.5, chain=0.5, per_task={"a": 0.6}, calls=20)),
        bok.load_run(_write_run(tmp_path, "s43", oracle=0.72, best=0.70, test=0.6, chain=0.6, per_task={"a": 0.72}, calls=20)),
        bok.load_run(_write_run(tmp_path, "s44", oracle=0.68, best=0.66, test=0.55, chain=0.55, per_task={"a": 0.68}, calls=20)),
    ]
    arm = bok.Arm("meta-n", runs)
    assert arm.k == 3
    val, who = arm.best_of_k("best_of_search_dev")
    assert val == 0.72 and who == "s43"  # run-level max, NOT per-task oracle
    assert arm.mean_calls() == 20.0


def test_per_task_best_of_k_is_oracle_across_runs(tmp_path):
    runs = [
        bok.load_run(_write_run(tmp_path, "s1", oracle=0.5, best=0.5, test=0.5, chain=0.5, per_task={"a": 0.9, "b": 0.3}, calls=3)),
        bok.load_run(_write_run(tmp_path, "s2", oracle=0.5, best=0.5, test=0.5, chain=0.5, per_task={"a": 0.4, "b": 0.8}, calls=3)),
    ]
    arm = bok.Arm("control", runs)
    # per-task max across the arm's runs
    assert arm.per_task_best_of_k() == {"a": 0.9, "b": 0.8}


def test_paired_ablation_wins_losses_and_sign_test():
    treat = {"a": 0.9, "b": 0.5, "c": 0.7, "d": 0.6}
    control = {"a": 0.6, "b": 0.5, "c": 0.4, "d": 0.8}
    res = bok.paired_per_task(treat, control)
    assert res["n_tasks"] == 4
    assert res["wins"] == 2 and res["losses"] == 1 and res["ties"] == 1  # a,c win; d loss; b tie
    assert res["mean_delta"] == pytest.approx((0.3 + 0.0 + 0.3 - 0.2) / 4)
    # 2 wins / 1 loss (ties excluded): two-sided binomial on n=3
    assert 0.0 < res["sign_test_p"] <= 1.0


def test_sign_test_symmetric_and_bounds():
    assert bok._sign_test_p(0, 0) == 1.0
    assert bok._sign_test_p(5, 5) == 1.0
    assert bok._sign_test_p(6, 0) == pytest.approx(2 / 64)  # 2 * (1/2)^6


def test_cli_renders_table_and_pair(tmp_path, capsys):
    t = [_write_run(tmp_path, f"treat_s4{i}", oracle=0.7 + 0.01 * i, best=0.65, test=0.6, chain=0.6,
                    per_task={"a": 0.7 + 0.01 * i, "b": 0.5}, calls=24) for i in range(3)]
    c = [_write_run(tmp_path, f"control_s{i}", oracle=0.6, best=0.6, test=0.55, chain=0.55,
                    per_task={"a": 0.6, "b": 0.45}, calls=3) for i in range(8)]
    rc = bok.main([
        "--arm", "meta-n", *[str(x) for x in t],
        "--arm", "control", *[str(x) for x in c],
        "--pair", "meta-n", "control",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "best-of-K arm comparison" in out
    assert "meta-n" in out and "control" in out
    assert "Thesis ablation" in out and "sign-test p" in out
