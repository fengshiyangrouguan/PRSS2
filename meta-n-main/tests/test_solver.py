"""Tests for the Layer 1 solver."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.llm_client import LLMClient, LLMConfig
from meta_n.core.meta_layer import TaskDescription
from meta_n.core.solver import Layer1Solver


RESPONSE_WITH_BASH_FENCE = """Here's the script to create the file:

```bash
#!/bin/bash
set -e
echo 'Hello, World!' > /tmp/hello.txt
```

This creates the file with the exact content requested.
"""

RESPONSE_WITH_PLAIN_FENCE = """
```
echo 42 > /tmp/answer.txt
```
"""

RESPONSE_WITH_SH_FENCE = """
```sh
mkdir -p /tmp/testdir
touch /tmp/testdir/a.txt /tmp/testdir/b.txt /tmp/testdir/c.txt
```
"""

RESPONSE_NO_FENCE = """echo hello > /tmp/out.txt"""


@pytest.fixture
def mock_client():
    config = LLMConfig(api_key="test")
    client = LLMClient(config)
    return client


@pytest.fixture
def task():
    return TaskDescription(task_id="t1", description="Create hello.txt")


class TestLayer1Solver:
    @pytest.mark.asyncio
    async def test_solve_bash_fence(self, mock_client, task):
        mock_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(RESPONSE_WITH_BASH_FENCE, 100)
        )

        solver = Layer1Solver(mock_client)
        script, reasoning, tokens = await solver.solve(task)

        assert "echo 'Hello, World!' > /tmp/hello.txt" in script
        assert "set -e" in script
        assert tokens == 100
        assert reasoning == RESPONSE_WITH_BASH_FENCE

    @pytest.mark.asyncio
    async def test_solve_plain_fence(self, mock_client, task):
        mock_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(RESPONSE_WITH_PLAIN_FENCE, 50)
        )

        solver = Layer1Solver(mock_client)
        script, _, _ = await solver.solve(task)
        assert "echo 42" in script

    @pytest.mark.asyncio
    async def test_solve_sh_fence(self, mock_client, task):
        mock_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(RESPONSE_WITH_SH_FENCE, 60)
        )

        solver = Layer1Solver(mock_client)
        script, _, _ = await solver.solve(task)
        assert "mkdir -p" in script

    @pytest.mark.asyncio
    async def test_solve_no_fence(self, mock_client, task):
        mock_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(RESPONSE_NO_FENCE, 20)
        )

        solver = Layer1Solver(mock_client)
        script, _, _ = await solver.solve(task)
        assert "echo hello" in script

    @pytest.mark.asyncio
    async def test_solve_with_additional_context(self, mock_client, task):
        mock_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(RESPONSE_WITH_BASH_FENCE, 100)
        )

        solver = Layer1Solver(mock_client)
        await solver.solve(task, additional_context="Use mkdir -p first")

        # Verify the prompt included additional context
        call_args = mock_client._client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert "Use mkdir -p first" in messages[0]["content"]


RESPONSE_WITH_PYTHON_FENCE = """Here's the solve function:

```python
def solve(**kwargs):
    items = kwargs['items']
    capacity = kwargs['capacity']
    bins = []
    for item in items:
        placed = False
        for b in bins:
            if sum(b) + item <= capacity:
                b.append(item)
                placed = True
                break
        if not placed:
            bins.append([item])
    return {'bins': bins}
```
"""


class TestPythonSolver:
    @pytest.mark.asyncio
    async def test_solve_python_fence(self, mock_client):
        mock_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(RESPONSE_WITH_PYTHON_FENCE, 150)
        )
        task = TaskDescription(
            task_id="bp",
            description="Bin packing",
            metadata={"solution_language": "python"},
        )
        solver = Layer1Solver(mock_client, language="python")
        script, _, tokens = await solver.solve(task)

        assert "def solve(" in script
        assert "bins" in script
        assert tokens == 150

    @pytest.mark.asyncio
    async def test_task_metadata_overrides_language(self, mock_client):
        """Task metadata solution_language overrides solver default."""
        mock_client._client.chat.completions.create = AsyncMock(
            return_value=_mock_response(RESPONSE_WITH_PYTHON_FENCE, 100)
        )
        task = TaskDescription(
            task_id="bp",
            description="Bin packing",
            metadata={"solution_language": "python"},
        )
        # Solver defaults to bash, but task says python
        solver = Layer1Solver(mock_client, language="bash")
        await solver.solve(task)

        call_args = mock_client._client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "solve() function" in prompt  # Python prompt, not bash


def _mock_response(content: str, tokens: int):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage = MagicMock()
    resp.usage.total_tokens = tokens
    return resp
