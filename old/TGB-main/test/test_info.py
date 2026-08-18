"""
Invariant tests for ``tgb/utils/info.py``.

These are the primary guard for URL-migration changes such as PR #128
(https://github.com/shenyangHuang/TGB/pull/128), which swaps every dataset
download URL to a new object-store host. They are fully offline: they only
inspect the module-level dictionaries, so they run anywhere with no dependencies
beyond ``tgb`` itself.

Everything is derived from the dictionaries (never a hardcoded dataset list) so
the checks automatically extend to datasets added later.
"""
import os
import os.path as osp
from urllib.parse import urlparse

import pytest

from tgb.utils.info import (
    PROJ_DIR,
    BColors,
    DATA_URL_DICT,
    DATA_VERSION_DICT,
    DATA_EVAL_METRIC_DICT,
    DATA_NS_STRATEGY_DICT,
    DATA_NUM_CLASSES,
)

VALID_EVAL_METRICS = {"mrr", "ndcg"}
VALID_NS_STRATEGIES = {
    "rnd",
    "hist_rnd",
    "time-filtered",
    "dst-time-filtered",
    "node-type-filtered",
}

DATASET_NAMES = sorted(DATA_URL_DICT.keys())


def _zip_basename(url: str) -> str:
    return osp.basename(urlparse(url).path)


# --------------------------------------------------------------------------- #
# cross-dictionary key consistency
# --------------------------------------------------------------------------- #
def test_url_version_eval_dicts_cover_the_same_datasets():
    """Every dataset must have a URL, a version and an eval metric."""
    url_keys = set(DATA_URL_DICT)
    assert url_keys == set(DATA_VERSION_DICT), (
        "mismatch between DATA_URL_DICT and DATA_VERSION_DICT: "
        f"{url_keys ^ set(DATA_VERSION_DICT)}"
    )
    assert url_keys == set(DATA_EVAL_METRIC_DICT), (
        "mismatch between DATA_URL_DICT and DATA_EVAL_METRIC_DICT: "
        f"{url_keys ^ set(DATA_EVAL_METRIC_DICT)}"
    )


def test_tgbn_datasets_have_num_classes():
    tgbn = {name for name in DATA_URL_DICT if name.startswith("tgbn-")}
    missing = tgbn - set(DATA_NUM_CLASSES)
    assert not missing, f"tgbn datasets missing from DATA_NUM_CLASSES: {missing}"
    # DATA_NUM_CLASSES should not reference unknown datasets
    assert set(DATA_NUM_CLASSES) <= set(DATA_URL_DICT)


def test_tkg_and_thg_datasets_have_ns_strategy():
    needs_strategy = {
        name
        for name in DATA_URL_DICT
        if name.startswith("tkgl-") or name.startswith("thgl-")
    }
    missing = needs_strategy - set(DATA_NS_STRATEGY_DICT)
    assert not missing, f"datasets missing from DATA_NS_STRATEGY_DICT: {missing}"
    assert set(DATA_NS_STRATEGY_DICT) <= set(DATA_URL_DICT)


# --------------------------------------------------------------------------- #
# URL well-formedness  (parametrized per dataset)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", DATASET_NAMES)
def test_url_is_well_formed(name):
    url = DATA_URL_DICT[name]
    parsed = urlparse(url)
    assert parsed.scheme == "https", f"{name}: url must be https, got {url!r}"
    assert parsed.netloc, f"{name}: url has no host: {url!r}"
    assert parsed.path.endswith(".zip"), f"{name}: url must point to a .zip: {url!r}"


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_url_filename_matches_dataset_name(name):
    """The archive filename must belong to this dataset (guards mis-pasted URLs)."""
    basename = _zip_basename(DATA_URL_DICT[name])
    assert basename.startswith(name), (
        f"{name}: url filename {basename!r} does not start with the dataset name; "
        "the URL may point at the wrong dataset"
    )


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_url_version_suffix_matches_version_dict(name):
    """A dataset at version > 1 must carry a matching ``-v<version>`` in its URL."""
    version = DATA_VERSION_DICT[name]
    if version > 1:
        basename = _zip_basename(DATA_URL_DICT[name])
        assert f"-v{version}" in basename, (
            f"{name}: version is {version} but url filename {basename!r} "
            f"does not contain '-v{version}'"
        )


def test_all_urls_share_a_single_host():
    """
    All datasets live in one object store. If a migration updates some URLs but
    misses others, the hosts diverge and this fails.
    """
    hosts = {urlparse(url).netloc for url in DATA_URL_DICT.values()}
    assert len(hosts) == 1, f"dataset URLs span multiple hosts: {sorted(hosts)}"


# --------------------------------------------------------------------------- #
# value validity
# --------------------------------------------------------------------------- #
def test_eval_metric_values_are_supported():
    bad = {k: v for k, v in DATA_EVAL_METRIC_DICT.items() if v not in VALID_EVAL_METRICS}
    assert not bad, f"unsupported eval metrics: {bad}"


def test_ns_strategy_values_are_supported():
    bad = {
        k: v for k, v in DATA_NS_STRATEGY_DICT.items() if v not in VALID_NS_STRATEGIES
    }
    assert not bad, f"unsupported negative-sampling strategies: {bad}"


@pytest.mark.parametrize("name", sorted(DATA_VERSION_DICT))
def test_version_is_positive_int(name):
    version = DATA_VERSION_DICT[name]
    assert isinstance(version, int) and version >= 1, f"{name}: bad version {version!r}"


@pytest.mark.parametrize("name", sorted(DATA_NUM_CLASSES))
def test_num_classes_is_positive_int(name):
    n = DATA_NUM_CLASSES[name]
    assert isinstance(n, int) and n > 0, f"{name}: bad num_classes {n!r}"


# --------------------------------------------------------------------------- #
# misc module-level constants
# --------------------------------------------------------------------------- #
def test_proj_dir_is_valid_directory():
    assert PROJ_DIR.endswith("/"), f"PROJ_DIR should end with '/': {PROJ_DIR!r}"
    assert osp.isdir(PROJ_DIR), f"PROJ_DIR is not a directory: {PROJ_DIR!r}"


def test_bcolors_has_expected_attributes():
    for attr in ("HEADER", "OKGREEN", "WARNING", "FAIL", "ENDC", "BOLD"):
        assert hasattr(BColors, attr), f"BColors missing {attr}"
        assert isinstance(getattr(BColors, attr), str)
