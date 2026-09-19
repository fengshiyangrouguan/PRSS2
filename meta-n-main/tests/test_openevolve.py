"""Tests for OpenEvolve benchmark adapters (AlphaEvolve Math, Symbolic Regression, AlgoTune)."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meta_n.core.meta_layer import TaskDescription, Trace
from meta_n.integrations.benchmark import EvalResult
from meta_n.integrations.openevolve import (
    ALGOTUNE_TASKS,
    AlgoTuneAdapter,
    AlphaEvolveMathAdapter,
    OpenEvolveBaseAdapter,
    OpenEvolveExecutor,
    SymbolicRegressionAdapter,
)


# --- Helper fixtures ---


@pytest.fixture
def fake_math_dir():
    """Create a minimal fake AlphaEvolve math data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Simple problem: kissing_number
        kn_dir = base / "kissing_number"
        kn_dir.mkdir()
        (kn_dir / "evaluator.py").write_text(
            "BENCHMARK = 593\n"
            "def evaluate(program_path):\n"
            "    return {'combined_score': 0.003, 'num_points': 2}\n"
        )
        (kn_dir / "initial_program.py").write_text(
            "# EVOLVE-BLOCK-START\n"
            "import numpy as np\n\n"
            'def kissing_number11() -> "np.ndarray":\n'
            '    """Construct 11-dimensional points for kissing number.\n\n'
            "    Returns:\n"
            "        points: np.ndarray of shape (num_points, 11)\n"
            '    """\n'
            "    return np.zeros((2, 11))\n"
            "# EVOLVE-BLOCK-END\n"
        )
        (kn_dir / "requirements.txt").write_text("numpy\n")

        # Sub-variant: heilbronn_convex/13
        hc_dir = base / "heilbronn_convex" / "13"
        hc_dir.mkdir(parents=True)
        (hc_dir / "evaluator.py").write_text(
            "def evaluate(program_path):\n"
            "    return {'combined_score': 0.5}\n"
        )
        (hc_dir / "initial_program.py").write_text(
            "import numpy as np\n\n"
            "def heilbronn_convex13() -> 'np.ndarray':\n"
            '    """Place 13 points."""\n'
            "    return np.zeros((13, 2))\n"
        )

        yield tmpdir


@pytest.fixture
def fake_sr_dir():
    """Create a minimal fake symbolic regression data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Generated problem
        prob_dir = base / "physics_osc_001"
        prob_dir.mkdir()
        (prob_dir / "evaluator.py").write_text(
            "def evaluate(program_path):\n"
            "    return {'combined_score': 3.5, 'negative_mse': -0.0003}\n"
        )
        (prob_dir / "initial_program.py").write_text(
            "# EVOLVE-BLOCK-START\n"
            "import numpy as np\n\n"
            "def func(x, params):\n"
            '    """Predict y from x using params."""\n'
            "    return np.zeros(x.shape[0])\n"
            "# EVOLVE-BLOCK-END\n\n"
            "def run_search():\n"
            "    return func\n"
        )

        yield tmpdir


@pytest.fixture
def fake_algotune_dir():
    """Create a minimal fake AlgoTune data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        task_dir = base / "fft_convolution"
        task_dir.mkdir()
        (task_dir / "evaluator.py").write_text(
            "def evaluate(program_path):\n"
            "    return {'speedup_score': 2.5, 'correctness_score': 1.0}\n"
        )
        (task_dir / "initial_program.py").write_text(
            "import numpy as np\n\n"
            "def run_solver(problem):\n"
            '    """Solve FFT convolution."""\n'
            "    return {}\n"
        )

        yield tmpdir


# --- Function name extraction ---


class TestFunctionNameExtraction:
    def test_evolve_block(self):
        source = (
            "# EVOLVE-BLOCK-START\n"
            "import numpy as np\n\n"
            "def kissing_number11():\n"
            "    pass\n"
            "# EVOLVE-BLOCK-END\n"
        )
        assert OpenEvolveBaseAdapter._extract_function_name(source) == "kissing_number11"

    def test_no_evolve_block(self):
        source = "def run_search():\n    return func\n"
        assert OpenEvolveBaseAdapter._extract_function_name(source) == "run_search"

    def test_multiple_defs(self):
        source = (
            "# EVOLVE-BLOCK-START\n"
            "def helper():\n    pass\n\n"
            "def main_func(x):\n    pass\n"
            "# EVOLVE-BLOCK-END\n"
        )
        # Returns the first def in the block
        assert OpenEvolveBaseAdapter._extract_function_name(source) == "helper"

    def test_no_def(self):
        source = "x = 42\n"
        assert OpenEvolveBaseAdapter._extract_function_name(source) == "solve"

    def test_run_function(self):
        source = (
            "# EVOLVE-BLOCK-START\n"
            "def run():\n"
            '    """Run optimization."""\n'
            "    pass\n"
            "# EVOLVE-BLOCK-END\n"
        )
        assert OpenEvolveBaseAdapter._extract_function_name(source) == "run"


# --- AlphaEvolveMathAdapter ---


class TestAlphaEvolveMathAdapter:
    def test_properties(self):
        adapter = AlphaEvolveMathAdapter(data_dir="/nonexistent")
        assert adapter.name == "alphaevolve_math"

    def test_load_tasks(self, fake_math_dir):
        adapter = AlphaEvolveMathAdapter(data_dir=fake_math_dir)
        tasks = adapter.load_tasks()

        assert len(tasks) == 2
        task_ids = {t.task_id for t in tasks}
        assert "kissing_number" in task_ids
        # Sub-variant uses underscore-joined path
        assert any("heilbronn_convex" in tid for tid in task_ids)

    def test_load_tasks_metadata(self, fake_math_dir):
        adapter = AlphaEvolveMathAdapter(data_dir=fake_math_dir)
        tasks = adapter.load_tasks()

        kn_task = next(t for t in tasks if "kissing" in t.task_id)
        assert kn_task.metadata["benchmark"] == "alphaevolve_math"
        assert kn_task.metadata["function_name"] == "kissing_number11"
        assert kn_task.metadata["solution_language"] == "openevolve"
        assert "numpy" in kn_task.metadata["deps"]
        assert kn_task.metadata["score_key"] == "combined_score"

    def test_load_tasks_with_limit(self, fake_math_dir):
        adapter = AlphaEvolveMathAdapter(data_dir=fake_math_dir)
        tasks = adapter.load_tasks(limit=1)
        assert len(tasks) == 1

    def test_load_tasks_missing_dir(self):
        adapter = AlphaEvolveMathAdapter(data_dir="/nonexistent")
        tasks = adapter.load_tasks()
        assert len(tasks) == 0

    def test_load_tasks_with_filter(self, fake_math_dir):
        adapter = AlphaEvolveMathAdapter(
            data_dir=fake_math_dir, problem_names=["kissing"],
        )
        tasks = adapter.load_tasks()
        assert len(tasks) == 1
        assert "kissing" in tasks[0].task_id

    def test_task_description_content(self, fake_math_dir):
        adapter = AlphaEvolveMathAdapter(data_dir=fake_math_dir)
        tasks = adapter.load_tasks()
        kn_task = next(t for t in tasks if "kissing" in t.task_id)

        # Description should include full initial_program as reference
        assert "kissing_number11" in kn_task.description
        assert "Reference Implementation" in kn_task.description
        assert "numpy" in kn_task.description
        # Should include the full module structure, not just the signature
        assert "EVOLVE-BLOCK" in kn_task.description or "def kissing_number11" in kn_task.description

    @pytest.mark.asyncio
    async def test_evaluate_success(self, fake_math_dir):
        adapter = AlphaEvolveMathAdapter(data_dir=fake_math_dir)
        tasks = adapter.load_tasks()
        task = next(t for t in tasks if "kissing" in t.task_id)

        with patch(
            "meta_n.integrations.openevolve._run_openevolve_with_timeout",
            return_value=("ok", {"score": 0.85, "details": {"num_points": 504}}),
        ):
            result = await adapter.evaluate(task, "def kissing_number11(): ...")
            assert result.success is True
            assert result.score == 0.85
            assert result.raw_score == 0.85

    @pytest.mark.asyncio
    async def test_evaluate_failure(self, fake_math_dir):
        adapter = AlphaEvolveMathAdapter(data_dir=fake_math_dir)
        tasks = adapter.load_tasks()
        task = tasks[0]

        with patch(
            "meta_n.integrations.openevolve._run_openevolve_with_timeout",
            return_value=("error", "ImportError: No module named 'jax'"),
        ):
            result = await adapter.evaluate(task, "bad code")
            assert result.success is False
            assert result.score == 0.0
            assert "ImportError" in result.feedback

    @pytest.mark.asyncio
    async def test_evaluate_score_above_1(self, fake_math_dir):
        """Scores > 1.0 should pass through (beating the benchmark)."""
        adapter = AlphaEvolveMathAdapter(data_dir=fake_math_dir)
        tasks = adapter.load_tasks()
        task = tasks[0]

        with patch(
            "meta_n.integrations.openevolve._run_openevolve_with_timeout",
            return_value=("ok", {"score": 1.05, "details": {}}),
        ):
            result = await adapter.evaluate(task, "def kissing_number11(): ...")
            assert result.score == 1.05  # NOT clamped


# --- SymbolicRegressionAdapter ---


class TestSymbolicRegressionAdapter:
    def test_properties(self):
        adapter = SymbolicRegressionAdapter(data_dir="/nonexistent")
        assert adapter.name == "symbolic_regression"

    def test_load_tasks(self, fake_sr_dir):
        adapter = SymbolicRegressionAdapter(data_dir=fake_sr_dir)
        tasks = adapter.load_tasks()
        assert len(tasks) == 1
        assert "physics_osc_001" in tasks[0].task_id

    def test_load_tasks_missing_dir(self):
        adapter = SymbolicRegressionAdapter(data_dir="/nonexistent")
        tasks = adapter.load_tasks()
        assert len(tasks) == 0


# --- AlgoTuneAdapter ---


class TestAlgoTuneAdapter:
    def test_properties(self):
        adapter = AlgoTuneAdapter(data_dir="/nonexistent")
        assert adapter.name == "algotune"

    def test_task_list_constant(self):
        assert len(ALGOTUNE_TASKS) == 8
        assert "fft_convolution" in ALGOTUNE_TASKS

    def test_load_tasks(self, fake_algotune_dir):
        adapter = AlgoTuneAdapter(data_dir=fake_algotune_dir)
        adapter._available = True  # override dependency check for testing
        tasks = adapter.load_tasks()
        assert len(tasks) == 1
        assert "fft_convolution" in tasks[0].task_id

    def test_load_tasks_missing_deps(self):
        """If dependencies are missing, adapter returns empty list."""
        adapter = AlgoTuneAdapter(data_dir="/nonexistent")
        adapter._available = False
        tasks = adapter.load_tasks()
        assert len(tasks) == 0

    def test_score_key_is_speedup(self, fake_algotune_dir):
        adapter = AlgoTuneAdapter(data_dir=fake_algotune_dir)
        adapter._available = True  # override dependency check for testing
        tasks = adapter.load_tasks()
        assert tasks[0].metadata["score_key"] == "speedup_score"


# --- OpenEvolveExecutor ---


class TestOpenEvolveExecutor:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        adapter = MagicMock()
        adapter.evaluate = AsyncMock(
            return_value=EvalResult(
                success=True, score=0.85, raw_score=0.85, feedback="OK",
            )
        )

        executor = OpenEvolveExecutor(adapter)
        task = TaskDescription(
            task_id="kissing_number",
            description="Solve kissing number",
            metadata={"problem_name": "kissing_number"},
        )

        trace = await executor.execute("def kissing_number11(): ...", task)

        assert trace.success is True
        assert trace.score == 0.85
        assert "0.850000" in trace.stdout
        assert trace.exit_code == 0
        assert trace.duration_s > 0

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        adapter = MagicMock()
        adapter.evaluate = AsyncMock(
            return_value=EvalResult(
                success=False, score=0.0, feedback="Timeout (120s)",
            )
        )

        executor = OpenEvolveExecutor(adapter)
        task = TaskDescription(
            task_id="matmul",
            description="Tensor decomposition",
            metadata={"problem_name": "matmul"},
        )

        trace = await executor.execute("def run(): ...", task)

        assert trace.success is False
        assert trace.score == 0.0
        assert trace.exit_code == 1
        assert "Timeout" in trace.stderr

    @pytest.mark.asyncio
    async def test_execute_high_score(self):
        """Scores > 1.0 should pass through to Trace."""
        adapter = MagicMock()
        adapter.evaluate = AsyncMock(
            return_value=EvalResult(
                success=True, score=1.5, raw_score=1.5, feedback="Beat SOTA!",
            )
        )

        executor = OpenEvolveExecutor(adapter)
        task = TaskDescription(
            task_id="kissing_number",
            description="Solve",
            metadata={},
        )

        trace = await executor.execute("def kissing_number11(): ...", task)
        assert trace.score == 1.5  # NOT clamped


# --- Prompt routing ---


class TestPromptRouting:
    def test_openevolve_prompt_exists(self):
        from meta_n.core.prompts import SOLVER_PROMPT_OPENEVOLVE

        assert "mathematical" in SOLVER_PROMPT_OPENEVOLVE.lower()
        assert "numpy" in SOLVER_PROMPT_OPENEVOLVE.lower()
        # Should NOT restrict to stdlib
        assert "do NOT use numpy" not in SOLVER_PROMPT_OPENEVOLVE

    def test_openevolve_solver_lib_exists(self):
        from meta_n.core.prompts import SOLVER_LIB_SECTION_OPENEVOLVE

        assert "numpy" in SOLVER_LIB_SECTION_OPENEVOLVE.lower()
        assert "scipy" in SOLVER_LIB_SECTION_OPENEVOLVE.lower()
        # Should NOT restrict to stdlib
        assert "Only use Python standard library" not in SOLVER_LIB_SECTION_OPENEVOLVE

    def test_solver_uses_openevolve_prompt(self):
        """Verify solver.py routes 'openevolve' to the correct template."""
        from meta_n.core.prompts import SOLVER_PROMPT_OPENEVOLVE, SOLVER_PROMPT_PYTHON

        # The two prompts should be different
        assert SOLVER_PROMPT_OPENEVOLVE != SOLVER_PROMPT_PYTHON
        # OpenEvolve allows numpy; Python restricts to stdlib
        assert "do NOT use numpy" not in SOLVER_PROMPT_OPENEVOLVE


# --- Round-2 failure-ranking fix (OE-1 / OE-2 / OE-3) ---------------------
#
# These tests pin the SR-localized failure-ranking fix: a hard failure on the
# negative-capable symbolic_regression scale must rank to the scale FLOOR (so a
# valid negative fit out-ranks a crash), while the non-negative AlphaEvolve /
# AlgoTune scales — and the shared [0,1]/unit archive path — stay byte-identical.

from meta_n.core.archive import Archive, Candidate  # noqa: E402


def _sentinel_ok_result():
    """An 'ok' subprocess result whose -1e9 sentinel was clamped to 0.0 in the
    subprocess (``_coerce_eval_score``) and flagged via ``raw_failure_sentinel``."""
    return ("ok", {"score": 0.0, "details": {"raw_failure_sentinel": -1e9}})


class TestFailureRankingScopedSR:
    @pytest.mark.asyncio
    async def test_sr_sentinel_routes_to_reporting_floor(self, fake_sr_dir):
        """Reaudit #9: a sentinel failure on SR scores the CLAMPED reporting floor
        (0.0), NOT the -1e9 failure_sentinel — routing -1e9 into `score` poisoned
        every averaged/reported mean. ran-vs-crashed is preserved via `success`."""
        adapter = SymbolicRegressionAdapter(data_dir=fake_sr_dir)
        task = adapter.load_tasks()[0]
        with patch(
            "meta_n.integrations.openevolve._run_openevolve_with_timeout",
            return_value=_sentinel_ok_result(),
        ):
            result = await adapter.evaluate(task, "def f(): ...")
        assert result.score == 0.0             # clamped reporting floor (Reaudit #9)
        assert result.success is False         # ran-vs-crashed: sentinel = fail (OE-2)

    @pytest.mark.asyncio
    async def test_sr_crash_routes_to_reporting_floor(self, fake_sr_dir):
        """Reaudit #9: a hard crash (timeout/import error) on SR also scores the
        clamped reporting floor (0.0), never the -1e9 sentinel."""
        adapter = SymbolicRegressionAdapter(data_dir=fake_sr_dir)
        task = adapter.load_tasks()[0]
        with patch(
            "meta_n.integrations.openevolve._run_openevolve_with_timeout",
            return_value=("error", "Timeout (90s)"),
        ):
            result = await adapter.evaluate(task, "bad code")
        assert result.score == 0.0
        assert result.raw_score == 0.0
        assert result.success is False

    @pytest.mark.asyncio
    async def test_sr_valid_negative_fit_is_success(self, fake_sr_dir):
        """OE-2: a valid-but-poor negative fit RAN — it must be success=True
        (so Ω does not over-sample valid SR equations as failures) and keep its
        negative score (NOT routed to the floor)."""
        adapter = SymbolicRegressionAdapter(data_dir=fake_sr_dir)
        task = adapter.load_tasks()[0]
        with patch(
            "meta_n.integrations.openevolve._run_openevolve_with_timeout",
            return_value=("ok", {"score": -0.70, "details": {"mse": 5.0}}),
        ):
            result = await adapter.evaluate(task, "def f(): ...")
        assert result.score == -0.70
        assert result.success is True

    @pytest.mark.asyncio
    async def test_alphaevolve_failure_stays_zero(self, fake_math_dir):
        """Byte-identical: the non-negative AlphaEvolve scale keeps the historical
        0.0 failure score and success=raw_score>0 — both sentinel and crash."""
        adapter = AlphaEvolveMathAdapter(data_dir=fake_math_dir)
        task = adapter.load_tasks()[0]
        with patch(
            "meta_n.integrations.openevolve._run_openevolve_with_timeout",
            return_value=_sentinel_ok_result(),
        ):
            sent = await adapter.evaluate(task, "x")
        with patch(
            "meta_n.integrations.openevolve._run_openevolve_with_timeout",
            return_value=("error", "ImportError"),
        ):
            crash = await adapter.evaluate(task, "x")
        assert sent.score == 0.0 and sent.success is False
        assert crash.score == 0.0 and crash.success is False

    def test_base_hooks_are_byte_identical_defaults(self):
        """The base hooks preserve historical AlphaEvolve/AlgoTune behavior."""
        for cls, data in [
            (AlphaEvolveMathAdapter, "/nonexistent"),
            (AlgoTuneAdapter, "/nonexistent"),
        ]:
            a = cls(data_dir=data)
            assert a._failure_score() == 0.0
            assert a._eval_success(0.85, False) is True
            assert a._eval_success(0.0, False) is False
        # SR: failures score the clamped reporting floor 0.0 (Reaudit #9 — the
        # -1e9 failure_sentinel must NOT enter the averaged `score` channel), but
        # ran-vs-crashed is still distinguished via `success` (OE-2 preserved).
        sr = SymbolicRegressionAdapter(data_dir="/nonexistent")
        assert sr._failure_score() == 0.0           # clamped reporting floor (Reaudit #9)
        assert sr._eval_success(-0.70, False) is True   # valid neg fit RAN -> success
        assert sr._eval_success(0.0, True) is False     # sentinel failure -> not success


class TestArchivePerTaskBestNegativeScale:
    """Archive per-task-best ranking: the (success, score) dual channel.

    Reaudit #9 dual-channel resolution: the archive per-task-best now ranks by
    (success, score) — a trace that RAN out-ranks a crashed one regardless of
    score (`success` = ran-vs-crashed ranking channel), then score decides among
    successes. `score` stays the clean reporting channel (SR failures floor to
    0.0, no -1e9 sentinel poisoning any averaged metric). On a real non-negative
    [0,1] scale this is byte-identical to plain score-max (a failure floors to
    0.0 while any success scores >= 0), so the success term can only ever change
    the outcome on a negative-capable scale (SR)."""

    @staticmethod
    def _cand(cid, task_id, score, success):
        tr = Trace(task_id=task_id, script="s", success=success, score=score,
                   exit_code=0 if success else 1)
        return Candidate(candidate_id=cid, traces=[tr], mean_score=score,
                         per_task_scores={task_id: score})

    def test_sr_success_channel_lets_valid_negative_win(self):
        """On SR the valid −0.70 fit that RAN (success=True) becomes per-task-best
        over a 0.0-floored crash (success=False) — the success channel, not a
        -1e9 score sentinel, is what stops the crash out-ranking it."""
        arc = Archive()
        arc.add(self._cand("crash", "sr_task", 0.0, success=False))
        arc.add(self._cand("fit", "sr_task", -0.70, success=True))
        assert arc.best_score_for_task("sr_task") == -0.70
        assert arc.per_task_best_traces()["sr_task"].score == -0.70

    def test_unit_scale_per_task_best_is_score_max(self):
        """Real [0,1] scale: failures floor to 0.0 (success=False) and successes
        score >= 0 (success=True), so per-task-best == max score — byte-identical
        to plain score-max. (The success term only bites on negative-capable
        scales, where a 0.0 crash would otherwise beat a valid negative fit.)"""
        arc = Archive()
        arc.add(self._cand("crash", "unit_task", 0.0, success=False))
        arc.add(self._cand("fit", "unit_task", 0.30, success=True))
        assert arc.best_score_for_task("unit_task") == 0.30
