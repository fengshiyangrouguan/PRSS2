"""Regression tests for audit fixes in meta_n/analysis/emergent_roles.py.

Covers:
- Finding 25: `Optional` used in LayerRoleProfile annotations but never imported
  (latent NameError under type-hint introspection).
- Finding 44: documented CLI entry point (`python -m meta_n.analysis.emergent_roles`)
  did not exist (no main()/__main__).

All tests are offline / LLM-free (no LM Studio, no Docker, no network).
"""

import json
import sys
import typing
from pathlib import Path

from meta_n.analysis import emergent_roles
from meta_n.analysis.emergent_roles import EmergentRoleAnalyzer, LayerRoleProfile


def test_optional_resolves_in_type_hints():
    """Finding 25: get_type_hints must resolve the Optional annotations.

    On the un-fixed code (`Optional` never imported) this raises
    NameError: name 'Optional' is not defined because the lazy string
    annotations from `from __future__ import annotations` are resolved here.
    """
    hints = typing.get_type_hints(LayerRoleProfile)
    for field_name in ("rationale_embedding", "code_embedding", "combined_embedding"):
        assert field_name in hints
        # Optional[X] resolves to typing.Union[X, None]
        assert type(None) in typing.get_args(hints[field_name])


def test_module_exposes_main_entry_point():
    """Finding 44: the documented CLI must have a real main() callable."""
    assert hasattr(emergent_roles, "main")
    assert callable(emergent_roles.main)


def test_cli_main_runs_on_empty_experiment(tmp_path, monkeypatch, capsys):
    """Finding 44: invoking the documented CLI runs analyze() + save_results().

    On the un-fixed code there is no main(), so this AttributeErrors.
    With an empty experiment dir (no depth_*/injected_code.json) analyze()
    returns an empty result and save_results() must still write the JSON.
    """
    exp_dir = tmp_path / "run_001"
    exp_dir.mkdir()

    monkeypatch.setattr(
        sys, "argv",
        ["emergent_roles", "--experiment-dir", str(exp_dir)],
    )
    emergent_roles.main()

    out_json = exp_dir / "analysis" / "role_analysis.json"
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert data["experiment_dir"] == str(exp_dir)

    captured = capsys.readouterr()
    assert "role_analysis.json" in captured.out


def test_analyzer_still_constructs():
    """Sanity: the analyzer class is unaffected by the fixes."""
    analyzer = EmergentRoleAnalyzer("/nonexistent/dir")
    assert isinstance(analyzer.experiment_dir, Path)
