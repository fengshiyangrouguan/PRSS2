"""Smoke tests for the shared conftest factories (P0.1)."""


def test_disjoint_archive_is_the_oracle_fixture(make_disjoint_archive):
    arch = make_disjoint_archive(3)
    # Each task's best comes from a distinct lineage → oracle vector is all `win`.
    assert arch.per_task_best_scores() == {"task_0": 0.9, "task_1": 0.9, "task_2": 0.9}
    # ...and the oracle strictly dominates every single candidate's mean.
    assert all(c.mean_score < 0.9 for c in arch.candidates)
    assert len(arch) == 3


def test_continuous_archive_scale(make_continuous_archive):
    arch = make_continuous_archive()
    assert arch.best_mean_score == 40.0  # non-[0,1] scale preserved


def test_sentinel_archive_contains_sentinel(make_sentinel_archive):
    arch = make_sentinel_archive()
    means = [c.mean_score for c in arch.candidates]
    assert -1e9 in means
    # add() guards best with isfinite; -1e9 is finite but never the max here.
    assert arch.best_mean_score == 40.0


def test_make_candidate_defaults(make_candidate):
    c = make_candidate("c1", mean_score=0.5)
    assert c.candidate_id == "c1"
    assert c.per_task_scores == {"task_a": 0.5, "task_b": 0.5}
    assert len(c.traces) == 2


async def test_mock_omega_swallows_unknown_kwargs(make_mock_omega):
    omega = make_mock_omega()
    ic, tokens = await omega.generate(depth=2, no_code_library=True, some_future_param=123)
    assert ic.source_depth == 2
    assert tokens == 50


def test_write_candidate_dir_roundtrips(tmp_path, make_candidate, write_candidate_dir):
    from meta_n.core.archive import Archive

    c = make_candidate("gen0_seed", mean_score=0.5, iteration=0)
    write_candidate_dir(tmp_path / "archive", c)
    rebuilt = Archive.rebuild_from_disk(tmp_path / "archive")
    assert len(rebuilt) == 1
    assert rebuilt.get("gen0_seed").mean_score == 0.5
