"""Round-3 correctness regressions for meta_n/core/agentic_solver.py.

Fix covered:
* R3-E_omega_context-1 (AgenticSolver side) — the ``no_outer_context`` constructor
  kwarg completes the R1 wiring so E3 genuinely ablates the flattened inter-layer
  ``outer_context`` channel on the agentic path (matching native ``MetaLayer`` +
  ``run_pre_process(thread_outer_context=...)``).

  Contract:
  - Default ``no_outer_context=False`` => ``thread_outer_context=True`` =>
    ``_run_all_pre_process`` output is byte-identical to HEAD (a shallower block
    still SEES a deeper block's emission as ``outer_context``).
  - ``no_outer_context=True`` => ``thread_outer_context=False`` => the inter-layer
    INPUT channel is ablated; a shallower block sees the pinned seed ("") instead.

Only the AgenticSolver side is exercised here (its PRIMARY change lives in this
module). The InjectionMapper / ExternalAgentSolver / orchestrator-config wiring
and the main.py diagnostic collapse are landed in their own files.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from meta_n.core.agentic_solver import AgenticSolver
from meta_n.core.meta_layer import InjectedCode, TaskDescription


def _task() -> TaskDescription:
    return TaskDescription(task_id="t1", description="d")


# Two-layer probe (mirrors R1's ``_codes_input_probe``): the deeper block
# (source_depth=2) emits 'D3' and runs first (deepest-first); the shallower block
# (source_depth=1) echoes the ``outer_context`` it OBSERVED, so its emission
# reveals whether the deeper block's emission was threaded to it.
def _codes_input_probe() -> list[InjectedCode]:
    deep = InjectedCode(pre_process="additional_context = 'D3'", source_depth=2)
    shallow = InjectedCode(
        pre_process="additional_context = f'saw:{outer_context}'", source_depth=1
    )
    return [shallow, deep]


def _solver(injected_codes, *, no_outer_context: bool) -> AgenticSolver:
    return AgenticSolver(
        llm_client=AsyncMock(),
        executor=AsyncMock(),
        injected_codes=injected_codes,
        no_outer_context=no_outer_context,
    )


def test_no_outer_context_false_threads_inter_layer_channel():
    """Default (no_outer_context=False) => shallower block SEES the deeper 'D3'."""
    solver = _solver(_codes_input_probe(), no_outer_context=False)
    combined = solver._run_all_pre_process(_task())
    assert combined == "D3\nsaw:D3"


def test_no_outer_context_true_ablates_inter_layer_channel():
    """no_outer_context=True => shallower block sees outer_context='' (ablated)."""
    solver = _solver(_codes_input_probe(), no_outer_context=True)
    combined = solver._run_all_pre_process(_task())
    assert combined == "D3\nsaw:"
    assert "saw:D3" not in combined


def test_default_kwarg_matches_false_off_path_identity():
    """Omitting the kwarg == no_outer_context=False (OFF-path byte-identity)."""
    default_solver = AgenticSolver(
        llm_client=AsyncMock(),
        executor=AsyncMock(),
        injected_codes=_codes_input_probe(),
    )
    assert default_solver.no_outer_context is False
    assert default_solver._run_all_pre_process(_task()) == "D3\nsaw:D3"


def test_single_block_flag_is_inert_on_agentic_path():
    """Depth<=2 single-injected-code candidates have no inter-layer channel, so
    both flag values yield identical ``_run_all_pre_process`` output."""
    codes = [InjectedCode(pre_process="additional_context = 'only'", source_depth=1)]
    on = _solver(codes, no_outer_context=True)._run_all_pre_process(_task())
    off = _solver(codes, no_outer_context=False)._run_all_pre_process(_task())
    assert on == off == "only"
