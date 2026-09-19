"""Tests for text classification benchmark adapter."""

import json

import pytest

from meta_n.core.meta_layer import TaskDescription
from meta_n.integrations.text_classification import (
    TextClassificationAdapter,
    TextClassificationExecutor,
    _normalize,
    _normalize_multi,
)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_basic(self):
        assert _normalize("Drug Reaction") == "drug reaction"

    def test_extra_whitespace(self):
        assert _normalize("  common   cold  ") == "common cold"

    def test_empty(self):
        assert _normalize("") == ""

    def test_case_insensitive(self):
        assert _normalize("ALLERGY") == _normalize("allergy")


class TestNormalizeMulti:
    def test_single_label(self):
        assert _normalize_multi("盗窃") == {"盗窃"}

    def test_multi_labels(self):
        assert _normalize_multi("盗窃;诈骗") == {"盗窃", "诈骗"}

    def test_whitespace_handling(self):
        assert _normalize_multi(" 盗窃 ; 诈骗 ") == {"盗窃", "诈骗"}

    def test_empty_parts_ignored(self):
        assert _normalize_multi("盗窃;;诈骗;") == {"盗窃", "诈骗"}

    def test_empty_string(self):
        assert _normalize_multi("") == set()


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


class TestScoringAccuracy:
    """Test accuracy scoring (Symptom2Disease)."""

    def _make_adapter(self) -> TextClassificationAdapter:
        adapter = TextClassificationAdapter.__new__(TextClassificationAdapter)
        adapter.dataset_name = "symptom2disease"
        adapter.data_dir = None
        adapter.n_few_shot = 0
        adapter.max_val = 50
        adapter.max_test = None
        adapter._data = None
        adapter._llm_client = None
        adapter._eval_workers = 1
        return adapter

    def test_perfect_score(self):
        adapter = self._make_adapter()
        examples = [
            {"text": "...", "label": "Allergy"},
            {"text": "...", "label": "Diabetes"},
        ]
        preds = {"case_0": "Allergy", "case_1": "Diabetes"}
        result = adapter._score_predictions(json.dumps(preds), examples, "accuracy")
        assert result.success
        assert result.score == 1.0
        # TC-2: raw_score is the fraction (0..1), matching _score_f1 — not the
        # 0..N correct-count it previously reported.
        assert result.raw_score == 1.0

    def test_partial_score(self):
        adapter = self._make_adapter()
        examples = [
            {"text": "...", "label": "Allergy"},
            {"text": "...", "label": "Diabetes"},
        ]
        preds = {"case_0": "Allergy", "case_1": "WRONG"}
        result = adapter._score_predictions(json.dumps(preds), examples, "accuracy")
        assert result.success
        assert result.score == 0.5

    def test_case_insensitive(self):
        adapter = self._make_adapter()
        examples = [{"text": "...", "label": "Drug Reaction"}]
        preds = {"case_0": "drug reaction"}
        result = adapter._score_predictions(json.dumps(preds), examples, "accuracy")
        assert result.score == 1.0

    def test_zero_score(self):
        adapter = self._make_adapter()
        examples = [{"text": "...", "label": "Allergy"}]
        preds = {"case_0": "Diabetes"}
        result = adapter._score_predictions(json.dumps(preds), examples, "accuracy")
        assert result.score == 0.0
        assert not result.success

    def test_missing_prediction(self):
        adapter = self._make_adapter()
        examples = [{"text": "...", "label": "Allergy"}]
        preds = {}  # no case_0
        result = adapter._score_predictions(json.dumps(preds), examples, "accuracy")
        assert result.score == 0.0

    def test_invalid_json(self):
        adapter = self._make_adapter()
        examples = [{"text": "...", "label": "Allergy"}]
        result = adapter._score_predictions("not json", examples, "accuracy")
        assert not result.success
        assert "Invalid JSON" in result.feedback

    def test_non_dict_json(self):
        adapter = self._make_adapter()
        examples = [{"text": "...", "label": "Allergy"}]
        result = adapter._score_predictions("[1, 2]", examples, "accuracy")
        assert not result.success
        assert "Expected JSON object" in result.feedback

    def test_mismatches_in_feedback(self):
        adapter = self._make_adapter()
        examples = [
            {"text": "...", "label": "Allergy"},
            {"text": "...", "label": "Diabetes"},
        ]
        preds = {"case_0": "WRONG", "case_1": "Diabetes"}
        result = adapter._score_predictions(json.dumps(preds), examples, "accuracy")
        assert "Sample mismatches" in result.feedback
        assert "case_0" in result.feedback


class TestScoringMicroF1:
    """Test per-example average F1 scoring (LawBench charge prediction)."""

    def _make_adapter(self) -> TextClassificationAdapter:
        adapter = TextClassificationAdapter.__new__(TextClassificationAdapter)
        adapter.dataset_name = "lawbench_charge"
        adapter.data_dir = None
        adapter.n_few_shot = 0
        adapter.max_val = 50
        adapter.max_test = None
        adapter._data = None
        adapter._llm_client = None
        adapter._eval_workers = 1
        return adapter

    def test_perfect_single_label(self):
        adapter = self._make_adapter()
        examples = [{"text": "...", "label": "盗窃"}]
        preds = {"case_0": "盗窃"}
        result = adapter._score_predictions(json.dumps(preds), examples, "micro_f1")
        assert result.score == 1.0

    def test_perfect_multi_label(self):
        adapter = self._make_adapter()
        examples = [{"text": "...", "label": "盗窃;诈骗"}]
        preds = {"case_0": "盗窃;诈骗"}
        result = adapter._score_predictions(json.dumps(preds), examples, "micro_f1")
        assert result.score == 1.0

    def test_partial_multi_label(self):
        adapter = self._make_adapter()
        examples = [{"text": "...", "label": "盗窃;诈骗"}]
        preds = {"case_0": "盗窃"}  # missed 诈骗
        result = adapter._score_predictions(json.dumps(preds), examples, "micro_f1")
        # TP=1, FP=0, FN=1 → P=1.0, R=0.5 → F1=2/3
        assert abs(result.score - 2 / 3) < 1e-6

    def test_wrong_prediction(self):
        adapter = self._make_adapter()
        examples = [{"text": "...", "label": "盗窃"}]
        preds = {"case_0": "诈骗"}  # wrong
        result = adapter._score_predictions(json.dumps(preds), examples, "micro_f1")
        # TP=0, FP=1, FN=1 → F1=0.0
        assert result.score == 0.0
        assert not result.success

    def test_empty_prediction(self):
        adapter = self._make_adapter()
        examples = [{"text": "...", "label": "盗窃"}]
        preds = {"case_0": ""}
        result = adapter._score_predictions(json.dumps(preds), examples, "micro_f1")
        assert result.score == 0.0

    def test_multi_case_aggregation(self):
        adapter = self._make_adapter()
        examples = [
            {"text": "...", "label": "盗窃"},
            {"text": "...", "label": "诈骗;抢劫"},
        ]
        preds = {"case_0": "盗窃", "case_1": "诈骗"}
        result = adapter._score_predictions(json.dumps(preds), examples, "micro_f1")
        # Per-example F1 (matching original LawBench evaluation):
        # case_0: P=1, R=1 → F1=1.0
        # case_1: P=1/1=1.0, R=1/2=0.5 → F1=2/3
        # Average: (1.0 + 2/3) / 2 = 5/6
        assert abs(result.score - 5 / 6) < 1e-6

    def test_feedback_format(self):
        adapter = self._make_adapter()
        examples = [{"text": "...", "label": "盗窃"}]
        preds = {"case_0": "盗窃"}
        result = adapter._score_predictions(json.dumps(preds), examples, "micro_f1")
        assert "F1:" in result.feedback
        assert "abstentions" in result.feedback


# ---------------------------------------------------------------------------
# Adapter load_tasks
# ---------------------------------------------------------------------------


class TestAdapterLoadTasks:
    def test_load_tasks_returns_one_task(self):
        """Each dataset produces exactly one task."""
        adapter = TextClassificationAdapter.__new__(TextClassificationAdapter)
        adapter.dataset_name = "symptom2disease"
        adapter.data_dir = None
        adapter.n_few_shot = 2
        adapter.max_val = 50
        adapter.max_test = None
        adapter._llm_client = None
        adapter._eval_workers = 1
        adapter._data = {
            "train": [
                {"text": "itching and rash", "label": "Fungal infection"},
                {"text": "sneezing and watery eyes", "label": "Allergy"},
                {"text": "headache", "label": "Migraine"},
            ],
            "val": [
                {"text": "fever and chills", "label": "Malaria"},
            ],
            "test": [
                {"text": "joint pain", "label": "Arthritis"},
            ],
            "labels": ["Allergy", "Arthritis", "Fungal infection", "Malaria", "Migraine"],
            "metric": "accuracy",
            "language": "en",
        }

        tasks = adapter.load_tasks()

        assert len(tasks) == 1
        task = tasks[0]
        assert task.task_id == "symptom2disease"
        assert task.metadata["benchmark"] == "text_classification"
        assert task.metadata["solution_language"] == "python"
        assert task.metadata["metric"] == "accuracy"
        # F127: val-split payloads must never enter task.metadata (Ω-injected
        # pre_process can read it and feed the solver prompt).
        assert "val_cases" not in task.metadata
        assert "val_labels" not in task.metadata
        assert len(task.metadata["few_shot_examples"]) == 2
        assert task.metadata["label_set"] == ["Allergy", "Arthritis", "Fungal infection", "Malaria", "Migraine"]
        assert "Allergy" in task.description
        assert "def solve(" in task.description

    def test_load_tasks_limit_zero(self):
        adapter = TextClassificationAdapter.__new__(TextClassificationAdapter)
        adapter.dataset_name = "symptom2disease"
        adapter.data_dir = None
        adapter.n_few_shot = 0
        adapter.max_val = 50
        adapter.max_test = None
        adapter._llm_client = None
        adapter._eval_workers = 1
        adapter._data = {
            "train": [], "val": [], "test": [],
            "labels": [], "metric": "accuracy", "language": "en",
        }

        tasks = adapter.load_tasks(limit=0)
        assert len(tasks) == 0


class TestAdapterProperties:
    def test_name(self):
        adapter = TextClassificationAdapter.__new__(TextClassificationAdapter)
        adapter.dataset_name = "symptom2disease"
        assert adapter.name == "classify_symptom2disease"

    def test_unknown_dataset(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            TextClassificationAdapter("nonexistent")


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class TestTextClassificationExecutor:
    def _make_adapter_with_data(self) -> TextClassificationAdapter:
        adapter = TextClassificationAdapter.__new__(TextClassificationAdapter)
        adapter.dataset_name = "symptom2disease"
        adapter.data_dir = None
        adapter.n_few_shot = 2
        adapter.max_val = 50
        adapter.max_test = None
        adapter._llm_client = None
        adapter._eval_workers = 1
        adapter._solve_timeout = 10
        adapter._subprocess_isolate = False  # in-process for fast tests
        adapter._data = {
            "train": [
                {"text": "itching", "label": "Fungal infection"},
                {"text": "sneezing", "label": "Allergy"},
            ],
            "val": [
                {"text": "fever", "label": "Malaria"},
                {"text": "rash", "label": "Allergy"},
            ],
            "test": [],
            "labels": ["Allergy", "Fungal infection", "Malaria"],
            "metric": "accuracy",
            "language": "en",
        }
        return adapter

    @pytest.mark.asyncio
    async def test_execute_success(self):
        adapter = self._make_adapter_with_data()
        executor = TextClassificationExecutor(adapter)
        task = TaskDescription(
            task_id="symptom2disease",
            description="Classify diseases",
            metadata={"metric": "accuracy"},
        )

        script = (
            "def solve(cases, labels, few_shot):\n"
            "    return {k: 'Malaria' if 'fever' in v else 'Allergy' for k, v in cases.items()}\n"
        )
        trace = await executor.execute(script, task)

        assert trace.success
        assert trace.score == 1.0
        assert trace.exit_code == 0
        assert trace.duration_s > 0
        assert trace.eval_feedback  # eval_feedback populated

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        adapter = self._make_adapter_with_data()
        executor = TextClassificationExecutor(adapter)
        task = TaskDescription(
            task_id="symptom2disease",
            description="Classify diseases",
            metadata={"metric": "accuracy"},
        )

        trace = await executor.execute("x = 1  # no solve function", task)

        assert not trace.success
        assert trace.score == 0.0
        assert trace.exit_code == 1
        assert "No solve() function" in trace.stderr


# ---------------------------------------------------------------------------
# _run_solve_code
# ---------------------------------------------------------------------------


class TestRunSolveCode:
    def _make_adapter(self, subprocess_isolate: bool = False) -> TextClassificationAdapter:
        # Default to in-process for easy debugging; subprocess-specific tests
        # opt in by passing subprocess_isolate=True. Production defaults to
        # subprocess (constructor default) — these tests bypass __init__.
        adapter = TextClassificationAdapter.__new__(TextClassificationAdapter)
        adapter.dataset_name = "symptom2disease"
        adapter.data_dir = None
        adapter.n_few_shot = 1
        adapter.max_val = 50
        adapter.max_test = None
        adapter._llm_client = None
        adapter._eval_workers = 1
        adapter._solve_timeout = 10
        adapter._subprocess_isolate = subprocess_isolate
        adapter._inner_log_path = None
        adapter._data = {
            "train": [{"text": "itching", "label": "Fungal infection"}],
            "val": [],
            "test": [],
            "labels": ["Allergy", "Fungal infection"],
            "metric": "accuracy",
            "language": "en",
        }
        return adapter

    def test_valid_solve(self):
        adapter = self._make_adapter()
        examples = [{"text": "rash", "label": "Allergy"}]
        code = "def solve(cases, labels, few_shot):\n    return {k: 'Allergy' for k in cases}\n"
        result = adapter._run_solve_code(code, examples, "accuracy")
        assert result.success
        assert result.score == 1.0

    def test_no_solve_function(self):
        adapter = self._make_adapter()
        examples = [{"text": "rash", "label": "Allergy"}]
        result = adapter._run_solve_code("x = 1", examples, "accuracy")
        assert not result.success
        assert "No solve() function" in result.feedback

    def test_crash(self):
        adapter = self._make_adapter()
        examples = [{"text": "rash", "label": "Allergy"}]
        code = "def solve(cases, labels, few_shot):\n    raise RuntimeError('boom')\n"
        result = adapter._run_solve_code(code, examples, "accuracy")
        assert not result.success
        assert "RuntimeError" in result.feedback

    def test_parallel_results_match_sequential(self):
        """Parallel eval should produce the same score as sequential."""
        adapter = self._make_adapter()
        examples = [
            {"text": "rash", "label": "Allergy"},
            {"text": "fever", "label": "Fungal infection"},
            {"text": "cough", "label": "Allergy"},
            {"text": "itch", "label": "Fungal infection"},
        ]
        code = (
            "def solve(cases, labels, few_shot):\n"
            "    return {k: labels[0] for k in cases}\n"
        )
        # Sequential
        adapter._eval_workers = 1
        result_seq = adapter._run_solve_code(code, examples, "accuracy")
        # Parallel (2 workers, 4 examples → 2 chunks of 2)
        adapter._eval_workers = 2
        result_par = adapter._run_solve_code(code, examples, "accuracy")

        assert result_seq.score == result_par.score
        assert result_seq.success == result_par.success

    def test_parallel_error_propagates(self):
        """If solve() crashes in a chunk, the error should be reported."""
        adapter = self._make_adapter()
        examples = [
            {"text": "rash", "label": "Allergy"},
            {"text": "fever", "label": "Allergy"},
            {"text": "cough", "label": "Allergy"},
            {"text": "itch", "label": "Allergy"},
        ]
        code = "def solve(cases, labels, few_shot):\n    raise ValueError('chunk crash')\n"
        adapter._eval_workers = 2
        result = adapter._run_solve_code(code, examples, "accuracy")
        assert not result.success
        assert "ValueError" in result.feedback

    def test_wrong_return_type(self):
        adapter = self._make_adapter()
        examples = [{"text": "rash", "label": "Allergy"}]
        code = "def solve(cases, labels, few_shot):\n    return ['Allergy']\n"
        result = adapter._run_solve_code(code, examples, "accuracy")
        assert not result.success
        assert "expected dict" in result.feedback

    def test_solve_with_llm_call(self):
        """solve() that calls llm() should work and track tokens (with split)."""
        from unittest.mock import AsyncMock, MagicMock

        client = MagicMock()
        # 9 prompt + 6 completion = 15 total — distinct so a swap shows up.
        # AsyncMock ignores unexpected kwargs (e.g. _suppress_io_log added
        # by llm_helpers to prevent inner/outer double-logging), so no
        # signature update is needed here.
        client.complete_with_breakdown = AsyncMock(
            return_value=("Allergy", 9, 6, 15)
        )

        adapter = self._make_adapter()
        adapter._llm_client = client
        examples = [{"text": "rash", "label": "Allergy"}]
        code = (
            "def solve(cases, labels, few_shot):\n"
            "    return {k: llm(f'Classify: {v}') for k, v in cases.items()}\n"
        )
        result = adapter._run_solve_code(code, examples, "accuracy")
        assert result.success
        assert result.score == 1.0
        assert "[inner-llm]" in result.feedback
        assert "1 calls" in result.feedback
        assert result.inner_tokens == 15
        assert result.inner_prompt_tokens == 9
        assert result.inner_completion_tokens == 6

    def test_solve_llm_exception_caught(self):
        """If llm() raises inside solve(), it should be caught gracefully."""
        from unittest.mock import AsyncMock, MagicMock

        client = MagicMock()
        client.complete_with_breakdown = AsyncMock(
            side_effect=RuntimeError("API down")
        )

        adapter = self._make_adapter()
        adapter._llm_client = client
        examples = [{"text": "rash", "label": "Allergy"}]
        code = (
            "def solve(cases, labels, few_shot):\n"
            "    return {k: llm('classify') for k, v in cases.items()}\n"
        )
        result = adapter._run_solve_code(code, examples, "accuracy")
        assert not result.success
        assert "RuntimeError" in result.feedback


# ---------------------------------------------------------------------------
# Subprocess isolation (production solver execution path)
# ---------------------------------------------------------------------------


class TestSubprocessIsolation:
    """Direct tests for _run_solve_with_timeout — the production path that
    isolates LLM-generated solver code in mp.Process subprocesses with hard
    timeouts. Mirrors co_bench's pattern.
    """

    def test_success_path(self):
        """Valid solver returns predictions; status='ok', usage reported."""
        from meta_n.integrations.text_classification import _run_solve_with_timeout

        code = (
            "def solve(cases, labels, few_shot):\n"
            "    return {k: labels[0] for k in cases}\n"
        )
        cases = {"case_0": "rash", "case_1": "fever"}
        labels = ["Allergy", "Fungal infection"]

        status, payload, usage = _run_solve_with_timeout(
            code, cases, labels, [], llm_config_dict=None, timeout=10,
        )
        assert status == "ok"
        assert payload == {"case_0": "Allergy", "case_1": "Allergy"}
        # No inner LLM calls — every counter should be zero.
        assert usage["total"] == 0
        assert usage["prompt"] == 0
        assert usage["completion"] == 0
        assert usage["calls"] == 0

    def test_no_solve_function(self):
        """Solver source with no solve() definition reports error."""
        from meta_n.integrations.text_classification import _run_solve_with_timeout

        status, payload, usage = _run_solve_with_timeout(
            "x = 1", {"case_0": "x"}, ["a"], [],
            llm_config_dict=None, timeout=5,
        )
        assert status == "error"
        assert "No solve() function" in payload
        assert usage["total"] == 0

    def test_solver_raises(self):
        """Solver raising at runtime is caught and reported."""
        from meta_n.integrations.text_classification import _run_solve_with_timeout

        code = (
            "def solve(cases, labels, few_shot):\n"
            "    raise RuntimeError('boom')\n"
        )
        status, payload, usage = _run_solve_with_timeout(
            code, {"case_0": "x"}, ["a"], [],
            llm_config_dict=None, timeout=5,
        )
        assert status == "error"
        assert "RuntimeError" in payload
        assert "boom" in payload
        assert usage["calls"] == 0

    def test_wrong_return_type(self):
        """Solver returning non-dict reports error and does not crash."""
        from meta_n.integrations.text_classification import _run_solve_with_timeout

        code = (
            "def solve(cases, labels, few_shot):\n"
            "    return ['not', 'a', 'dict']\n"
        )
        status, payload, usage = _run_solve_with_timeout(
            code, {"case_0": "x"}, ["a"], [],
            llm_config_dict=None, timeout=5,
        )
        assert status == "error"
        assert "expected dict" in payload
        assert usage["calls"] == 0

    def test_timeout_kills_runaway_solver(self):
        """A solver that hangs is killed within timeout + grace period."""
        import time as _time
        from meta_n.integrations.text_classification import _run_solve_with_timeout

        code = (
            "def solve(cases, labels, few_shot):\n"
            "    while True:\n"
            "        pass\n"
        )
        start = _time.time()
        status, payload, usage = _run_solve_with_timeout(
            code, {"case_0": "x"}, ["a"], [],
            llm_config_dict=None, timeout=1,
        )
        elapsed = _time.time() - start
        assert status == "error"
        assert "Timeout" in payload
        # Hard upper bound: 1 s timeout + 2 × 1 s post-kill joins + spawn overhead.
        assert elapsed < 6.0, f"kill cascade took too long: {elapsed:.1f}s"
        # Timeout returns the canonical empty-usage shape (subprocess killed
        # before it could sync any tokens back).
        assert set(usage) == {"total", "prompt", "completion", "calls"}

    def test_solver_cant_exfiltrate_parent_env(self):
        """Solver runs in subprocess: mutating sys.modules in child does not
        affect parent. Sanity check that isolation is real.
        """
        import sys as _sys
        from meta_n.integrations.text_classification import _run_solve_with_timeout

        code = (
            "def solve(cases, labels, few_shot):\n"
            "    import sys\n"
            "    sys.modules['__poisoned__'] = 'gotcha'\n"
            "    return {k: labels[0] for k in cases}\n"
        )
        status, payload, usage = _run_solve_with_timeout(
            code, {"case_0": "x"}, ["a"], [],
            llm_config_dict=None, timeout=5,
        )
        assert status == "ok"
        assert usage["total"] == 0
        # Child mutation must not leak into the parent's sys.modules.
        assert "__poisoned__" not in _sys.modules


# ---------------------------------------------------------------------------
# Solver JSON extraction (via solver module)
# ---------------------------------------------------------------------------


class TestSolverJsonExtraction:
    def test_extract_json_fenced(self):
        from meta_n.core.solver import Layer1Solver

        solver = Layer1Solver.__new__(Layer1Solver)
        response = 'Here are my predictions:\n```json\n{"case_0": "Allergy"}\n```'
        result = solver._extract_json(response)
        parsed = json.loads(result)
        assert parsed["case_0"] == "Allergy"

    def test_extract_json_unfenced(self):
        from meta_n.core.solver import Layer1Solver

        solver = Layer1Solver.__new__(Layer1Solver)
        response = '```\n{"case_0": "Allergy", "case_1": "Diabetes"}\n```'
        result = solver._extract_json(response)
        parsed = json.loads(result)
        assert parsed["case_0"] == "Allergy"

    def test_extract_json_raw(self):
        from meta_n.core.solver import Layer1Solver

        solver = Layer1Solver.__new__(Layer1Solver)
        response = 'The predictions are: {"case_0": "Allergy", "case_1": "Diabetes"}'
        result = solver._extract_json(response)
        parsed = json.loads(result)
        assert parsed["case_0"] == "Allergy"
