"""run_pre_process — shared implementation, deepest-first, rejection (§12.1).

The same module-level ``run_pre_process`` is the single implementation behind
``MetaLayer``, ``AgenticSolver`` and the external-agents ``InjectionMapper``. We
verify its ordering, its accumulation contract, the ``validate_code`` rejection
path, and that the ``InjectionMapper`` call site produces the same context the
direct call does (golden-fixture parity).
"""

from __future__ import annotations

from meta_n.core.external_agents.injection import InjectionMapper
from meta_n.core.meta_layer import SandboxMarker, run_pre_process

from .conftest import make_injected, make_task


def test_deepest_first_order():
    # injected_codes is shallowest -> deepest; run_pre_process iterates reversed
    # (deepest first). Each block appends a marker so the order is observable.
    shallow = make_injected(
        pre_process="additional_context = additional_context + 'S'", source_depth=1
    )
    deep = make_injected(
        pre_process="additional_context = additional_context + 'D'", source_depth=2
    )
    ran, ctx = run_pre_process([shallow, deep], make_task(), "", outer_context="")
    assert ran is True
    # Deepest runs first -> 'D' emission precedes 'S' emission in the accumulation.
    assert ctx.splitlines() == ["D", "S"]


def test_empty_when_no_pre_process():
    injected = [make_injected(code_library={"f": "def f():\n    return 1\n"})]
    ran, ctx = run_pre_process(injected, make_task(), "", outer_context="")
    assert ran is False
    assert ctx == ""


def test_validate_code_rejection_path():
    # An import statement is rejected by validate_code -> the block is skipped and
    # contributes no context (ran stays False if it's the only block).
    bad = make_injected(
        pre_process="import os\nadditional_context = 'should not appear'",
        source_depth=1,
    )
    ran, ctx = run_pre_process([bad], make_task(), "", outer_context="")
    assert ran is False
    assert "should not appear" not in ctx


def test_runtime_error_block_is_skipped():
    # A block that raises at exec time is skipped (logged), not propagated.
    boom = make_injected(
        pre_process="additional_context = 1 / 0", source_depth=1
    )
    good = make_injected(
        pre_process="additional_context = 'ok'", source_depth=2
    )
    ran, ctx = run_pre_process([boom, good], make_task(), "", outer_context="")
    assert ran is True
    assert ctx == "ok"


def test_non_str_emission_is_skipped():
    # A block that sets additional_context to a non-str is rejected.
    bad = make_injected(
        pre_process="additional_context = 123", source_depth=1
    )
    ran, ctx = run_pre_process([bad], make_task(), "", outer_context="")
    assert ran is False
    assert ctx == ""


def test_injection_mapper_matches_direct_call():
    # Golden parity: InjectionMapper.build runs run_pre_process(outer_context="")
    # and folds the emission into system_suffix. The direct call with the same
    # arguments must produce the same context string.
    pp = "additional_context = 'CTX:' + task.task_id"
    injected = [make_injected(pre_process=pp, source_depth=1)]
    task = make_task(task_id="abc")

    ran_direct, ctx_direct = run_pre_process(injected, task, "", outer_context="")

    plan = InjectionMapper(injected, "python", SandboxMarker()).build(task)
    # No libraries staged, so the suffix IS exactly the pre_process context.
    assert plan.pre_process_ran == ran_direct
    assert plan.prompt.system_suffix == ctx_direct == "CTX:abc"


def test_metalayer_run_pre_process_matches_direct_call():
    # Parity: MetaLayer._run_pre_process(task, outer_context=X) must equal
    # run_pre_process([single_block], task, outer_context=X)[1]. This pins the
    # refactor (MetaLayer now delegates to the shared function) as behavior-
    # preserving for the single-block MetaLayer case, including the outer_context
    # seeding (the block reads outer_context, which is NOT prepended to the
    # result).
    from unittest.mock import MagicMock

    from meta_n.core.meta_layer import MetaLayer

    # A block that echoes both its task id AND the outer_context it received, so
    # the test observes that outer_context is seeded but excluded from the result.
    pp = "additional_context = 'OWN:' + task.task_id + '|saw:' + outer_context"
    block = make_injected(pre_process=pp, source_depth=2)
    task = make_task(task_id="t9")
    outer = "HIGHER"

    layer = MetaLayer(
        depth=2,
        injected_code=block,
        inner_solver=MagicMock(),
        executor=MagicMock(),
    )
    ml_ctx = layer._run_pre_process(task, outer_context=outer)

    _, direct_ctx = run_pre_process([block], task, outer_context=outer)

    assert ml_ctx == direct_ctx
    # The block saw the outer_context in its namespace, but the returned context
    # is only its own emission (outer_context not prepended).
    assert ml_ctx == "OWN:t9|saw:HIGHER"


def test_agentic_solver_run_all_pre_process_matches_direct_call():
    # Parity: AgenticSolver._run_all_pre_process(task) (multi-block, deepest
    # first, outer_context seeded as the running accumulation starting from "")
    # must equal run_pre_process(injected, task, "", outer_context="")[1]. Pins
    # the AgenticSolver migration to the shared implementation.
    from unittest.mock import MagicMock

    from meta_n.core.agentic_solver import AgenticSolver

    # Two layers: each appends a marker AND records the outer_context it saw, so
    # the deepest-first accumulation threading through outer_context is observable.
    shallow = make_injected(
        pre_process="additional_context = 'S(' + outer_context + ')'", source_depth=1
    )
    deep = make_injected(
        pre_process="additional_context = 'D(' + outer_context + ')'", source_depth=2
    )
    injected = [shallow, deep]
    task = make_task(task_id="tm")

    solver = AgenticSolver(
        llm_client=MagicMock(),
        executor=MagicMock(),
        injected_codes=injected,
    )
    as_ctx = solver._run_all_pre_process(task)

    _, direct_ctx = run_pre_process(injected, task, "", outer_context="")

    assert as_ctx == direct_ctx
    # Deepest (D) runs first with outer_context=""; its emission becomes the
    # outer_context the shallow (S) block sees.
    assert as_ctx.splitlines() == ["D()", "S(D())"]
