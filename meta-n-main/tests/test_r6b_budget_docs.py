"""§6b F236 doc-rot guard — the ledger undercount bound stays documented.

The daily cap bounds PROVEN spend only: a degraded run whose agent actually
started but returned no parseable result under-counts ``today_total_usd()``
(hard-killed runner / outer-envelope timeout / contract-violating raise — see
the ``budget.py`` module docstring). The true figure is unknowable host-side,
so the accepted remedy is the documented bound; these tests keep that
documentation from silently rotting away in a future refactor.
"""

from __future__ import annotations

import meta_n.core.external_agents.budget as budget
from meta_n.core.external_agents.budget import CostGuard


def test_budget_module_docstring_names_undercount_bound():
    doc = (budget.__doc__ or "").lower()
    assert "undercount" in doc
    # All three unledgered/underledgered degraded paths are named:
    # (a) the hard-killed runner ($0 fold; SIGKILL is uncatchable),
    assert "hard-kill" in doc
    assert "sigkill" in doc
    # (b) the outer last-resort envelope cancelling before the fold,
    assert "finish_timeout" in doc
    # (c) a contract-violating raise losing the result object.
    assert "contract-violating" in doc
    assert "finish_error" in doc


def test_record_docstring_points_at_bound():
    assert "undercount" in (CostGuard.record.__doc__ or "").lower()
