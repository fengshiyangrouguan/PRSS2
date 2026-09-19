"""F172 regression — the ``solution_language`` property channel stays trimmed.

The SINGLE language channel is ``task.metadata["solution_language"]``, stamped
by each adapter's ``load_tasks`` and consumed by ``Layer1Solver.solve`` for
prompt selection (metadata stamps are pinned per adapter, e.g.
tests/test_benchmark.py / tests/test_openevolve.py load_tasks tests; routing is
also pinned by tests/test_solver.py::TestPythonSolver::
test_task_metadata_overrides_language). The former adapter-level property had
zero production reads and drifted from the metadata channel (the OpenEvolve
family answered "python" while stamping "openevolve"), so it was removed —
these tests keep it removed.
"""

from unittest.mock import AsyncMock, MagicMock

from meta_n.core.llm_client import LLMClient, LLMConfig
from meta_n.core.meta_layer import TaskDescription
from meta_n.core.solver import Layer1Solver
from meta_n.integrations.arc_agi import ARCAGI2Adapter
from meta_n.integrations.benchmark import BenchmarkAdapter
from meta_n.integrations.co_bench import COBenchAdapter
from meta_n.integrations.openevolve import (
    AlgoTuneAdapter,
    AlphaEvolveMathAdapter,
    OpenEvolveBaseAdapter,
    SymbolicRegressionAdapter,
)
from meta_n.integrations.swe_bench import SWEBenchVerifiedAdapter
from meta_n.integrations.terminal_bench.adapter import TerminalBenchAdapter
from meta_n.integrations.text_classification import TextClassificationAdapter


def test_solution_language_property_channel_trimmed():
    # No adapter-level language property anywhere in the hierarchy: metadata is
    # the single channel. (All these classes import without optional SDKs —
    # the subprocess-bridge invariant keeps SDK imports out of module scope.)
    for cls in (
        BenchmarkAdapter,
        COBenchAdapter,
        TextClassificationAdapter,
        TerminalBenchAdapter,
        SWEBenchVerifiedAdapter,
        OpenEvolveBaseAdapter,
        AlphaEvolveMathAdapter,
        SymbolicRegressionAdapter,
        AlgoTuneAdapter,
        ARCAGI2Adapter,
    ):
        assert not hasattr(cls, "solution_language"), cls.__name__


async def test_metadata_language_channel_still_routes():
    # task.metadata["solution_language"] still drives prompt selection in
    # Layer1Solver.solve (mocked LLM, no network): a bash-default solver must
    # pick the Python prompt when the task metadata says "python".
    client = LLMClient(LLMConfig(api_key="test"))
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = (
        "```python\ndef solve(**kwargs):\n    return {}\n```"
    )
    resp.usage = MagicMock()
    resp.usage.total_tokens = 10
    client._client.chat.completions.create = AsyncMock(return_value=resp)

    task = TaskDescription(
        task_id="t1",
        description="Bin packing",
        metadata={"solution_language": "python"},
    )
    solver = Layer1Solver(client, language="bash")
    await solver.solve(task)

    prompt = client._client.chat.completions.create.call_args[1]["messages"][0][
        "content"
    ]
    assert "solve() function" in prompt  # Python prompt, not bash
