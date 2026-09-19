"""Tests for core data models."""

import json

from meta_n.core.meta_layer import (
    InjectedCode,
    LayerResult,
    TaskDescription,
    Trace,
)


class TestTaskDescription:
    def test_basic_creation(self):
        task = TaskDescription(
            task_id="test_001",
            description="Do something",
        )
        assert task.task_id == "test_001"
        assert task.description == "Do something"
        assert task.verification_script is None
        assert task.metadata == {}

    def test_with_verification(self):
        task = TaskDescription(
            task_id="test_002",
            description="Create a file",
            verification_script="test -f /tmp/file.txt",
            metadata={"category": "file_ops"},
        )
        assert task.verification_script == "test -f /tmp/file.txt"
        assert task.metadata["category"] == "file_ops"

    def test_serialization(self):
        task = TaskDescription(task_id="t1", description="hello")
        data = task.model_dump()
        restored = TaskDescription(**data)
        assert restored == task

    def test_json_roundtrip(self):
        task = TaskDescription(
            task_id="t1",
            description="test",
            verification_script="echo ok",
        )
        json_str = task.model_dump_json()
        restored = TaskDescription.model_validate_json(json_str)
        assert restored == task


class TestTrace:
    def test_basic_creation(self):
        trace = Trace(task_id="t1", depth=1, script="echo hello")
        assert trace.task_id == "t1"
        assert trace.depth == 1
        assert trace.success is False
        assert trace.exit_code == -1

    def test_code_hash(self):
        trace = Trace(task_id="t1", script="echo hello")
        assert len(trace.code_hash) == 16
        # Same script → same hash
        trace2 = Trace(task_id="t2", script="echo hello")
        assert trace.code_hash == trace2.code_hash
        # Different script → different hash
        trace3 = Trace(task_id="t3", script="echo world")
        assert trace.code_hash != trace3.code_hash

    def test_eval_feedback_default(self):
        trace = Trace(task_id="t1", script="echo hello")
        assert trace.eval_feedback == ""

    def test_eval_feedback_populated(self):
        trace = Trace(
            task_id="t1", script="pass",
            eval_feedback="Accuracy: 18/20 = 0.9000\nSample mismatches:\n  case_3: ...",
        )
        assert "Accuracy" in trace.eval_feedback
        assert "case_3" in trace.eval_feedback

    def test_successful_trace(self):
        trace = Trace(
            task_id="t1",
            depth=1,
            script="echo 42 > /tmp/answer.txt",
            stdout="",
            stderr="",
            exit_code=0,
            success=True,
            duration_s=0.5,
        )
        assert trace.success is True
        assert trace.exit_code == 0

    def test_serialization(self):
        trace = Trace(
            task_id="t1",
            depth=2,
            script="ls",
            stdout="file.txt\n",
            exit_code=0,
            success=True,
        )
        data = trace.model_dump()
        restored = Trace(**data)
        assert restored.task_id == trace.task_id
        assert restored.stdout == trace.stdout


class TestInjectedCode:
    def test_empty(self):
        code = InjectedCode()
        assert code.is_empty is True

    def test_with_pre_process(self):
        code = InjectedCode(
            pre_process="additional_context = 'hint: use echo'",
            rationale="Adding hints based on failure patterns",
            source_depth=2,
        )
        assert code.is_empty is False
        assert code.source_depth == 2

    def test_full(self):
        code = InjectedCode(
            pre_process="additional_context = 'be careful'",
            rationale="Full injection",
            source_depth=3,
        )
        assert code.is_empty is False
        assert code.pre_process is not None

    def test_serialization(self):
        code = InjectedCode(
            pre_process="x = 1",
            rationale="test",
        )
        json_str = code.model_dump_json()
        restored = InjectedCode.model_validate_json(json_str)
        assert restored == code


class TestLayerResult:
    def test_creation(self):
        traces = [
            Trace(task_id="t1", success=True, script="echo ok"),
            Trace(task_id="t2", success=False, script="exit 1"),
        ]
        result = LayerResult(depth=1, traces=traces, pass_at_1=0.5)
        assert result.depth == 1
        assert result.pass_at_1 == 0.5
        assert len(result.traces) == 2
        assert result.token_usage == 0

    def test_with_injected_code(self):
        code = InjectedCode(pre_process="x = 1", source_depth=2)
        result = LayerResult(
            depth=2,
            pass_at_1=0.75,
            injected_code=code,
            token_usage=1500,
        )
        assert result.injected_code is not None
        assert result.token_usage == 1500


class TestToyTasksFile:
    def test_load_toy_tasks(self):
        with open("examples/toy_tasks.json") as f:
            tasks_data = json.load(f)

        tasks = [TaskDescription(**t) for t in tasks_data]
        assert len(tasks) == 3
        assert tasks[0].task_id == "toy_001"
        assert tasks[0].verification_script is not None
        assert all(t.verification_script for t in tasks)
