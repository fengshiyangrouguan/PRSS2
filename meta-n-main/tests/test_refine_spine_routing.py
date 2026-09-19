"""F042: the shared external-spine routing predicate (core/spine_routing).

One predicate now backs BOTH main.py's startup guards and
``EvolutionaryOrchestrator._uses_external_spine``: external kinds always route
to the spine; ``builtin`` routes only when the adapter's
``advertises_spine_builtin()`` returns the literal ``True``; a missing /
non-callable / raising capability probe routes legacy (never crashes startup).
"""

from __future__ import annotations

from meta_n.core.spine_routing import uses_external_spine


class _AdvertisingAdapter:
    def advertises_spine_builtin(self) -> bool:
        return True


class _TruthyAdapter:
    def advertises_spine_builtin(self):
        return "yes"  # truthy but not the literal True


class _RaisingAdapter:
    def advertises_spine_builtin(self):
        raise RuntimeError("faulty capability probe")


class _NonCallableAdapter:
    advertises_spine_builtin = True  # attribute, not a method


def test_external_kinds_true_regardless_of_adapter():
    for adapter in (None, _AdvertisingAdapter(), _RaisingAdapter()):
        assert uses_external_spine("openhands", adapter) is True
        assert uses_external_spine("terminus2", adapter) is True


def test_none_base_solver_and_none_adapter_route_legacy():
    assert uses_external_spine(None, None) is False
    assert uses_external_spine(None, _AdvertisingAdapter()) is False
    assert uses_external_spine("builtin", None) is False


def test_builtin_advertising_literal_true_routes_spine():
    assert uses_external_spine("builtin", _AdvertisingAdapter()) is True


def test_builtin_truthy_non_true_routes_legacy():
    # Strict ``is True``: a truthy MagicMock-like return must not promote.
    assert uses_external_spine("builtin", _TruthyAdapter()) is False


def test_builtin_raising_probe_routes_legacy_not_crash():
    # F042: main.py previously crashed here; the shared predicate logs + routes
    # legacy (the orchestrator's documented semantics).
    assert uses_external_spine("builtin", _RaisingAdapter()) is False


def test_builtin_non_callable_probe_routes_legacy():
    assert uses_external_spine("builtin", _NonCallableAdapter()) is False
