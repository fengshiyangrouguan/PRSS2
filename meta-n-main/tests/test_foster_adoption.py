"""Mechanism-0 adoption affordance (foster_adoption) tests.

The contract: foster_adoption defaults False and is byte-identical to the
legacy helper-advertising prose. When True it (1) inverts the "optional"
permission-to-ignore into a MUST-call imperative, (2) keeps the bare-name /
no-import convention that matches the deployed splice, and (3) emits a wired
solve() skeleton that already calls each helper by bare name.
"""

from meta_n.core.base_executor import LocalExecutor
from meta_n.core.meta_layer import (
    InjectedCode,
    MetaLayer,
    TaskDescription,
    format_bash_library_descriptions,
    format_python_library_descriptions,
)


class _Sandbox:
    is_sandboxed = True


_LIB = {
    "classify": (
        'def classify(case, labels):\n'
        '    """Classify a case into one of labels."""\n'
        '    return labels[0]'
    ),
    "validate_labels": (
        'def validate_labels(preds, labels):\n'
        '    """Coerce predictions to valid labels."""\n'
        '    return preds'
    ),
}


# --------------------------------------------------------------------------- #
# Default OFF == byte-identical
# --------------------------------------------------------------------------- #

def test_python_default_off_is_byte_identical():
    legacy = format_python_library_descriptions(_LIB, executor=_Sandbox())
    explicit_off = format_python_library_descriptions(
        _LIB, executor=_Sandbox(), foster_adoption=False
    )
    assert legacy == explicit_off
    # legacy retains the permission-to-ignore and the soft "just call them" header
    assert "optional" in legacy
    assert "just call them" in legacy
    assert "Wired solve() skeleton" not in legacy
    assert "MUST" not in legacy


def test_bash_default_off_is_byte_identical():
    legacy = format_bash_library_descriptions({}, {"setup": "setup() { echo ok; }"})
    explicit_off = format_bash_library_descriptions(
        {}, {"setup": "setup() { echo ok; }"}, foster_adoption=False
    )
    assert legacy == explicit_off
    assert "optional" in legacy
    assert "MUST" not in legacy


def test_empty_library_off_and_on_both_empty():
    assert format_python_library_descriptions({}, foster_adoption=True) == ""
    assert format_python_library_descriptions({}, foster_adoption=False) == ""


# --------------------------------------------------------------------------- #
# foster_adoption=True: imperative + convention + wired skeleton
# --------------------------------------------------------------------------- #

def test_python_foster_inverts_optional_to_imperative():
    desc = format_python_library_descriptions(
        _LIB, executor=_Sandbox(), foster_adoption=True
    )
    # (1) the permission-to-ignore is gone; replaced with a MUST imperative
    assert "optional" not in desc
    assert "MUST call" in desc
    assert "do NOT re-implement" in desc.replace("Do NOT", "do NOT")


def test_python_foster_keeps_bare_name_no_import_convention():
    # (2) advertised call form == deployed form (bare name, no solver_lib import)
    desc = format_python_library_descriptions(
        _LIB, executor=_Sandbox(), foster_adoption=True
    )
    assert "bare name" in desc
    assert "There is no module named `solver_lib`" in desc
    assert "from solver_lib import X" in desc  # the explicit anti-pattern warning


def test_python_foster_emits_wired_skeleton_calling_each_helper():
    # (3) wired solve() skeleton that already calls each helper by bare name
    desc = format_python_library_descriptions(
        _LIB, executor=_Sandbox(), foster_adoption=True
    )
    assert "Wired solve() skeleton" in desc
    assert "def solve(" in desc
    # every advertised helper appears as a bare-name call in the skeleton
    assert "classify(...)" in desc
    assert "validate_labels(...)" in desc
    # the skeleton must NOT advise importing solver_lib
    assert "import solver_lib" not in desc


def test_bash_foster_inverts_optional_to_imperative():
    desc = format_bash_library_descriptions(
        {}, {"setup": "setup() { echo ok; }"}, foster_adoption=True
    )
    assert "optional" not in desc
    assert "MUST call" in desc
    assert "do not re-implement" in desc.lower()


# --------------------------------------------------------------------------- #
# MetaLayer threads the flag end-to-end
# --------------------------------------------------------------------------- #

class _MockSolver:
    async def solve(self, task, additional_context=""):
        return "echo hi", "r", 0


def _layer(foster: bool) -> MetaLayer:
    return MetaLayer(
        depth=2,
        injected_code=InjectedCode(),
        inner_solver=_MockSolver(),
        executor=LocalExecutor(),
        merged_code_library=dict(_LIB),
        foster_adoption=foster,
    )


def test_metalayer_default_off_byte_identical():
    off = _layer(False)._format_library_descriptions()
    # default constructor (no kwarg) must equal explicit foster=False
    default = MetaLayer(
        depth=2,
        injected_code=InjectedCode(),
        inner_solver=_MockSolver(),
        executor=LocalExecutor(),
        merged_code_library=dict(_LIB),
    )._format_library_descriptions()
    assert off == default
    assert "Wired solve() skeleton" not in off


def test_metalayer_foster_on_injects_affordance():
    desc = _layer(True)._format_library_descriptions()
    assert "MUST call" in desc
    assert "Wired solve() skeleton" in desc
    assert "classify(...)" in desc


def test_metalayer_off_vs_on_differ():
    assert _layer(False)._format_library_descriptions() != _layer(True)._format_library_descriptions()


# --------------------------------------------------------------------------- #
# Config / CLI flag contract
# --------------------------------------------------------------------------- #

def test_config_has_foster_adoption_default_false():
    from meta_n.core.evolutionary_orchestrator import EvolutionaryConfig

    assert EvolutionaryConfig().foster_adoption is False
    assert EvolutionaryConfig(foster_adoption=True).foster_adoption is True


# --------------------------------------------------------------------------- #
# F5: --foster-adoption silent-no-op warning surface
# (foster_adoption_noop_reasons is pure; the warning fires iff it returns >0)
# --------------------------------------------------------------------------- #

from types import SimpleNamespace  # noqa: E402

from meta_n.main import foster_adoption_noop_reasons  # noqa: E402


class _LiveAdapter:
    def code_library_is_live(self):
        return True


class _DemotedAdapter:
    def code_library_is_live(self):
        return False


def _args(use_agentic=False, force_code_library_live=False):
    return SimpleNamespace(
        use_agentic=use_agentic,
        force_code_library_live=force_code_library_live,
    )


def test_foster_adoption_warns_with_use_agentic():
    # agentic solver path never consumes foster_adoption -> a no-op reason
    reasons = foster_adoption_noop_reasons(
        _args(use_agentic=True), _LiveAdapter(), external_spine=False
    )
    assert len(reasons) == 1
    assert "use-agentic" in reasons[0]


def test_foster_adoption_warns_with_external_spine():
    # external base-solver path ignores foster_adoption -> a no-op reason
    reasons = foster_adoption_noop_reasons(
        _args(), _LiveAdapter(), external_spine=True
    )
    assert len(reasons) == 1
    assert "external base-solver" in reasons[0]


def test_foster_adoption_warns_on_demoted_family_without_force_live():
    # demoted-helper family (code_library_is_live() False) without
    # --force-code-library-live -> helpers emptied, nothing to foster
    reasons = foster_adoption_noop_reasons(
        _args(force_code_library_live=False), _DemotedAdapter(), external_spine=False
    )
    assert len(reasons) == 1
    assert "demoted-helper" in reasons[0]


def test_foster_adoption_no_warn_when_force_live_undemotes_family():
    # --force-code-library-live un-demotes -> helpers are live -> no reason
    reasons = foster_adoption_noop_reasons(
        _args(force_code_library_live=True), _DemotedAdapter(), external_spine=False
    )
    assert reasons == []


def test_foster_adoption_no_warn_on_live_builtin():
    # live family + native builtin (no spine) + not agentic -> wired in -> []
    reasons = foster_adoption_noop_reasons(
        _args(), _LiveAdapter(), external_spine=False
    )
    assert reasons == []


def test_foster_adoption_warns_accumulates_multiple_reasons():
    # combined footguns each contribute a distinct reason
    reasons = foster_adoption_noop_reasons(
        _args(use_agentic=True), _DemotedAdapter(), external_spine=True
    )
    assert len(reasons) == 3


def test_foster_adoption_no_warn_when_adapter_none():
    # --tasks path (adapter is None): no demotion can be asserted -> only the
    # agentic/spine reasons fire; with neither, no reason at all
    reasons = foster_adoption_noop_reasons(_args(), None, external_spine=False)
    assert reasons == []
