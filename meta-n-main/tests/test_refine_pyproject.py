"""Refinement guards for repo/packaging metadata (cluster C11_repo_meta).

F211 — the `docker` pip package must NOT be a core dependency: the
terminal_bench integration drives the Docker CLI via subprocess only. If the
SDK is ever imported again, the import guard forces the dependency to be
re-declared deliberately.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_ROOT = REPO_ROOT / "meta_n"

# Matches a module-level or nested `import docker` / `from docker import ...`.
_DOCKER_IMPORT_RE = re.compile(r"^\s*(import docker\b|from docker\b)")


def _dep_name(requirement: str) -> str:
    """Distribution name portion of a PEP 508 requirement string."""
    return re.split(r"[\[<>=!~;\s]", requirement.strip(), maxsplit=1)[0].lower()


def test_docker_pip_package_not_declared():
    # F211: no core dependency may be the docker SDK.
    with open(PYPROJECT, "rb") as fh:
        pyproject = tomllib.load(fh)
    core_deps = pyproject["project"]["dependencies"]
    docker_deps = [d for d in core_deps if _dep_name(d) == "docker"]
    assert docker_deps == [], (
        f"docker pip package declared as core dependency: {docker_deps}; "
        "the terminal_bench integration uses the Docker CLI via subprocess only"
    )


def test_pyyaml_is_core_dependency():
    # meta_n.main loads meta_n/configs/benchmark_features.yaml BY DEFAULT (the
    # metacognition stack is turned on via that file), so `import yaml` must
    # resolve on a clean `pip install -e .` — pyyaml is a CORE dependency, not
    # merely an openevolve extra.
    with open(PYPROJECT, "rb") as fh:
        pyproject = tomllib.load(fh)
    core_deps = pyproject["project"]["dependencies"]
    pyyaml_deps = [d for d in core_deps if _dep_name(d) == "pyyaml"]
    assert pyyaml_deps, (
        "pyyaml must be a core dependency: meta_n.main loads "
        "meta_n/configs/benchmark_features.yaml by default"
    )


def test_benchmark_features_yaml_is_shipped_package_data():
    # Y4-C_callability-7: --benchmark-config 'auto' resolves package-relatively,
    # so the YAML must (a) be declared as setuptools package data — otherwise
    # wheel/sdist installs lack the file — and (b) live inside the installed
    # meta_n package where the resolver points.
    import fnmatch

    import meta_n
    from meta_n.main import _benchmark_config_path

    # (a) pyproject declares meta_n/configs/*.yaml as package data.
    with open(PYPROJECT, "rb") as fh:
        pyproject = tomllib.load(fh)
    patterns = pyproject["tool"]["setuptools"]["package-data"]["meta_n"]
    assert any(
        fnmatch.fnmatch("configs/benchmark_features.yaml", pat) for pat in patterns
    ), (
        "meta_n package-data must include a pattern covering "
        f"configs/benchmark_features.yaml; got {patterns}"
    )

    # (b) 'auto' resolves inside the meta_n package and the file exists.
    pkg_root = Path(meta_n.__file__).resolve().parent
    p = _benchmark_config_path("auto")
    assert p is not None
    assert p.is_relative_to(pkg_root), (
        f"'auto' must resolve package-relatively (inside {pkg_root}); got {p}"
    )
    assert p.exists(), f"bundled benchmark_features.yaml missing at {p}"


def test_no_docker_sdk_imports_in_package():
    # F211 invariant: meta_n never imports the docker SDK (CLI-only contract).
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _DOCKER_IMPORT_RE.match(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert offenders == [], (
        "docker SDK import found in meta_n/ — re-declare the dependency in "
        f"pyproject.toml deliberately if this is intended: {offenders}"
    )


def test_analysis_telemetry_exports_unwrap_record_source_level():
    """The F162 public alias must exist even where pandas is absent.

    scripts/build_ab_summary.py imports ``unwrap_record`` from
    meta_n.analysis.telemetry at module scope, but that module needs pandas —
    so the behavioral test (test_refine_telemetry.py) is skipped in
    pandas-free environments. Pin the export at source level so a broken
    alias cannot hide behind the skip.
    """
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    src = (repo_root / "meta_n" / "analysis" / "telemetry.py").read_text()
    tree = ast.parse(src)
    names = {
        t.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    assert "unwrap_record" in names
    all_node = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
    )
    assert "unwrap_record" in ast.literal_eval(all_node)
