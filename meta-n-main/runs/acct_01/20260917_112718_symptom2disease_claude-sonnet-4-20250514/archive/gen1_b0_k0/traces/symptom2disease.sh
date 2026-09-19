# --- injected code library ---

# lib:mock_helper

def mock_helper(x):
    return x

import types as _types
solver_lib = _types.SimpleNamespace(mock_helper=mock_helper)

# --- end injected code library ---

import sys


def solve(task, llm=None):
    """Mock solver. Deterministic, no network, no model."""
    return "mock"