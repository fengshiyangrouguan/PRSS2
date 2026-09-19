"""Refinement regression tests for the SWE-bench adapter.

Covers:
  * F123/F198 — the harbor download preamble is consolidated into
    ``TerminalBenchAdapter.download`` (template method); SWE-bench customizes
    only via the ``_cache_log_label`` / ``_filter_task_configs`` /
    ``_stage_downloaded`` hooks. The full download path (filtering, staging,
    hard-fail) stays covered by tests/test_swe_bench.py and
    tests/test_terminal_bench.py::TestDownload.
  * F134 — ``_parse_task_toml`` parses the file exactly ONCE; SWE-bench's
    memory-string handling extends ``_extract_task_config`` (dict → dict)
    instead of re-opening and re-parsing the TOML the parent just parsed.
"""

from __future__ import annotations

import pytest

import meta_n.integrations.terminal_bench as tb_mod
from meta_n.integrations.swe_bench import SWEBenchVerifiedAdapter
from meta_n.integrations.terminal_bench import TerminalBenchAdapter


class TestDownloadConsolidation:
    def test_download_override_removed(self, tmp_path):
        """SWE-bench must NOT re-implement download(): it inherits the parent
        template and customizes via hooks only."""
        assert SWEBenchVerifiedAdapter.download is TerminalBenchAdapter.download

        # The parent's filter hook is the identity (TB downloads the whole
        # dataset); SWE-bench's override narrows it.
        a = TerminalBenchAdapter(task_cache_dir=str(tmp_path))
        try:
            sentinel = [object(), object()]
            assert a._filter_task_configs(sentinel) is sentinel
        finally:
            a.cleanup()

        # Both hooks + the log label are genuinely specialized on SWE-bench.
        assert (
            SWEBenchVerifiedAdapter._filter_task_configs
            is not TerminalBenchAdapter._filter_task_configs
        )
        assert (
            SWEBenchVerifiedAdapter._stage_downloaded
            is not TerminalBenchAdapter._stage_downloaded
        )
        assert SWEBenchVerifiedAdapter._cache_log_label == "SWE-bench task directory"
        assert TerminalBenchAdapter._cache_log_label == "task directory"


class TestParseTaskTomlSingleRead:
    def test_parse_task_toml_single_read(self, tmp_path, monkeypatch):
        """The SWE-bench memory-string extension must reuse the parent's
        already-parsed dict — exactly one tomllib.load per task.toml."""
        task_toml = tmp_path / "task.toml"
        task_toml.write_text(
            "[environment]\n"
            "cpus = 2\n"
            "memory = '4G'\n"
            "[agent]\n"
            "timeout_sec = 600\n"
        )

        real_load = tb_mod.tomllib.load
        calls: list[int] = []

        def counting_load(f):
            calls.append(1)
            return real_load(f)

        monkeypatch.setattr(tb_mod.tomllib, "load", counting_load)

        a = SWEBenchVerifiedAdapter(task_cache_dir=str(tmp_path))
        try:
            cfg = a._parse_task_toml(task_toml)
        finally:
            a.cleanup()

        assert len(calls) == 1, f"expected exactly 1 TOML parse, got {len(calls)}"
        assert cfg["memory_mb"] == 4096
        assert cfg["cpus"] == 2
        assert cfg["timeout_sec"] == 600

    def test_extract_task_config_memory_mb_wins(self, tmp_path):
        """Explicit memory_mb beats the memory string (contract unchanged)."""
        a = SWEBenchVerifiedAdapter(task_cache_dir=str(tmp_path))
        try:
            cfg = a._extract_task_config(
                {"environment": {"memory": "8G", "memory_mb": 1234}}
            )
        finally:
            a.cleanup()
        assert cfg["memory_mb"] == 1234
