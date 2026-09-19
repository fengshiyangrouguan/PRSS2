"""Per-adapter split-type surface (P0.4, PREREQ-A for the overfit cluster).

``split_type`` declares how a benchmark's dev split relates to deployment, so
the overfit cluster (3.3 / 6.2 / 6.4) can gate its behavior by property rather
than by dataset name. ``split_type`` is a stateless descriptor, so it is tested
on the class (no adapter construction / data dirs needed).
"""

import pytest

from meta_n.integrations.benchmark import BenchmarkAdapter
from meta_n.integrations.co_bench import COBenchAdapter
from meta_n.integrations.openevolve import OpenEvolveBaseAdapter
from meta_n.integrations.text_classification import TextClassificationAdapter
from meta_n.integrations.arc_agi import ARCAGI2Adapter


class _DummyAdapter(BenchmarkAdapter):
    @property
    def name(self):
        return "dummy"

    def load_tasks(self, limit=None):
        return []

    async def evaluate(self, task, solution):
        raise NotImplementedError


def test_default_split_type_is_none():
    assert _DummyAdapter().split_type() == "none"


def test_co_bench_is_held_out():
    # R5 flip: evaluate_test scores the disjoint non-dev complement — a real
    # held-out split (the in-loop get_test_task machinery still absent).
    assert COBenchAdapter.split_type(None) == "held_out"


def test_openevolve_is_dev_equals_test():
    assert OpenEvolveBaseAdapter.split_type(None) == "dev_equals_test"


def test_text_classification_is_held_out():
    assert TextClassificationAdapter.split_type(None) == "held_out"


def test_arc_agi_is_proxy():
    assert ARCAGI2Adapter.split_type(None) == "proxy"


def test_terminal_and_swe_inherit_none():
    try:
        from meta_n.integrations.terminal_bench import TerminalBenchAdapter
        from meta_n.integrations.swe_bench import SWEBenchVerifiedAdapter
    except Exception as e:  # harbor / docker extras may be absent
        pytest.skip(f"terminal_bench import unavailable: {e}")
    assert TerminalBenchAdapter.split_type(None) == "none"
    assert SWEBenchVerifiedAdapter.split_type(None) == "none"


def test_split_type_values_are_in_the_declared_set():
    allowed = {"dev_equals_test", "held_out", "proxy", "none"}
    assert COBenchAdapter.split_type(None) in allowed
    assert ARCAGI2Adapter.split_type(None) in allowed
    assert TextClassificationAdapter.split_type(None) in allowed
    assert OpenEvolveBaseAdapter.split_type(None) in allowed
