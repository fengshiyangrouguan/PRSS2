"""CLI argparse smoke tests for the gemma-adoption-shortcircuit flags.

Each new flag MUST default OFF / None so the OFF path is byte-identical to HEAD.
"""
import sys
from unittest.mock import patch

from meta_n.main import parse_args


def _parse(argv):
    with patch.object(sys, "argv", ["meta-n", *argv]):
        return parse_args()


def test_request_timeout_defaults_none():
    args = _parse([])
    assert args.request_timeout is None


def test_request_timeout_parses_float():
    args = _parse(["--request-timeout", "1200"])
    assert args.request_timeout == 1200.0


def test_deploy_verified_code_defaults_off():
    args = _parse([])
    assert args.deploy_verified_code is False


def test_deploy_verified_code_flag_on():
    args = _parse(["--deploy-verified-code"])
    assert args.deploy_verified_code is True


def test_seed_code_library_defaults_none():
    args = _parse([])
    assert args.seed_code_library is None


def test_seed_code_library_parses_path():
    args = _parse(["--seed-code-library", "/tmp/crew.json"])
    assert args.seed_code_library == "/tmp/crew.json"


# --- F156 (§6b): symmetric Omega trace sampling flag ---
# Mirrors the omega-context-budget treatment: an OmegaEngine-constructor knob
# (not an evolutionary_run_kwargs key). Default OFF = historical sampling
# byte-identical.

def test_symmetric_trace_sampling_default_off():
    args = _parse([])
    assert args.symmetric_trace_sampling is False


def test_symmetric_trace_sampling_flag_on():
    args = _parse(["--symmetric-trace-sampling"])
    assert args.symmetric_trace_sampling is True


# --- C2.3: model-aware Omega INPUT-prompt budget flag ---

def test_omega_context_budget_defaults_none():
    args = _parse([])
    assert args.omega_context_budget is None


def test_omega_context_budget_parses_int():
    args = _parse(["--omega-context-budget", "32768"])
    assert args.omega_context_budget == 32768


def test_default_omega_engine_uses_builtin_100k_budget():
    # When the flag is omitted, OmegaEngine() is constructed with context_budget
    # None == the built-in ContextBudget (100k), keeping existing runs unchanged.
    from meta_n.core.omega import OmegaEngine
    from meta_n.core.llm_client import LLMClient, LLMConfig
    from meta_n.utils.context_manager import ContextBudget

    client = LLMClient(LLMConfig(api_key="test"))
    default_engine = OmegaEngine(client, context_budget=None)
    assert default_engine.context_manager.budget.max_tokens == 100_000

    custom = OmegaEngine(client, context_budget=ContextBudget(max_tokens=32768))
    assert custom.context_manager.budget.max_tokens == 32768
