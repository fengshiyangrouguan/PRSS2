"""Round-1 feature-audit regression tests for meta_n/core/meta_layer.py.

Covers the confirmed ROUND-1 correctness fixes whose PRIMARY change lives in
``meta_n.core.meta_layer``. Each test pins the contract stated in the fix spec.

Fixes covered:
- R1-E_omega_context-2: ``run_pre_process(thread_outer_context=...)`` gates the
  intra-call inter-layer accumulation (the ``outer_context`` INPUT seen by inner
  blocks) while leaving the returned emission (``combined``) independent of the
  flag. This restores ``--no-outer-context`` as a genuine E3 ablation on the
  agentic/external paths, where the seed is already ``""``.
"""

from __future__ import annotations

from meta_n.core.meta_layer import (
    InjectedCode,
    TaskDescription,
    run_pre_process,
)


def _task() -> TaskDescription:
    return TaskDescription(task_id="t1", description="d")


# Two layers: shallower (source_depth=1, outermost/inner conditioning target) and
# deeper (source_depth=2). run_pre_process iterates deepest-first, so the deeper
# block runs first and its emission becomes the outer_context input for the
# shallower block WHEN threading is enabled.
def _codes_input_probe() -> list[InjectedCode]:
    deep = InjectedCode(
        pre_process="additional_context = 'D3'",
        source_depth=2,
    )
    # Shallower block echoes the outer_context it OBSERVED, so its emission
    # reveals whether the deeper block's emission was threaded to it.
    shallow = InjectedCode(
        pre_process="additional_context = f'saw:{outer_context}'",
        source_depth=1,
    )
    return [shallow, deep]


def _codes_fixed_emission() -> list[InjectedCode]:
    deep = InjectedCode(pre_process="additional_context = 'D3'", source_depth=2)
    # Fixed emission, independent of outer_context: proves ``combined`` is
    # byte-identical regardless of the flag.
    shallow = InjectedCode(pre_process="additional_context = 'S2'", source_depth=1)
    return [shallow, deep]


def test_thread_outer_context_true_conditions_inner_block():
    """threading ON -> shallower block SEES the deeper block's 'D3' emission."""
    ran, combined = run_pre_process(
        _codes_input_probe(), _task(), thread_outer_context=True
    )
    assert ran is True
    # deepest-first: 'D3' first, then the shallower block that saw outer_context='D3'
    assert combined == "D3\nsaw:D3"


def test_thread_outer_context_false_ablates_inner_channel():
    """threading OFF -> shallower block sees outer_context='' (seed), not 'D3'."""
    ran, combined = run_pre_process(
        _codes_input_probe(), _task(), thread_outer_context=False
    )
    assert ran is True
    # The inter-layer INPUT channel is ablated: the shallower block saw "".
    assert combined == "D3\nsaw:"
    assert "saw:D3" not in combined


def test_default_matches_thread_outer_context_true():
    """Default (omitted kwarg) == thread_outer_context=True — OFF-path identity."""
    default_ran, default_combined = run_pre_process(_codes_input_probe(), _task())
    true_ran, true_combined = run_pre_process(
        _codes_input_probe(), _task(), thread_outer_context=True
    )
    assert (default_ran, default_combined) == (true_ran, true_combined)


def test_combined_emission_independent_of_flag():
    """When block emissions do not depend on outer_context, ``combined`` (the
    returned emission that flows to the solver prompt) is byte-identical for
    both flag values — only the per-block outer_context INPUT changes."""
    _, combined_true = run_pre_process(
        _codes_fixed_emission(), _task(), thread_outer_context=True
    )
    _, combined_false = run_pre_process(
        _codes_fixed_emission(), _task(), thread_outer_context=False
    )
    assert combined_true == combined_false == "D3\nS2"


def test_single_block_flag_is_inert():
    """Depth<=2 single-injected-code candidates have no inter-layer channel, so
    the flag is inert: both values yield identical output."""
    codes = [InjectedCode(pre_process="additional_context = 'only'", source_depth=1)]
    on = run_pre_process(codes, _task(), thread_outer_context=True)
    off = run_pre_process(codes, _task(), thread_outer_context=False)
    assert on == off == (True, "only")


def test_seed_outer_context_pinned_when_not_threaded():
    """threading OFF pins outer_context at its SEED value for every block, so a
    non-empty seed is still visible to blocks but never grows across the chain."""
    ran, combined = run_pre_process(
        _codes_input_probe(),
        _task(),
        outer_context="SEED",
        thread_outer_context=False,
    )
    assert ran is True
    # Both the deeper and shallower blocks see the pinned seed; the shallower
    # block's echo shows the seed, NOT the seed + deeper emission.
    assert combined == "D3\nsaw:SEED"
