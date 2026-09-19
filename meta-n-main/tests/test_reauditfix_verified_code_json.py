"""Re-audit regression test for finding #11.

_docker_run_json documents a "Never raises" fail-CLOSED contract: a timeout,
non-JSON, or crash must all map to a structured {"passed": False, ...} dict.
Before the fix, a VALID-but-non-object JSON last line (bare ``true`` / ``42`` /
``["ok"]``) parsed to a bool/int/list; the subsequent ``.setdefault`` raised
AttributeError, which was NOT in the caught ``(JSONDecodeError, ValueError)``
tuple and escaped the function, violating the contract and bypassing the
``no_json`` fail-closed branch.
"""

from unittest.mock import patch

import pytest

from meta_n.core.verified_code import _docker_run_json


class _FakeProc:
    def __init__(self, stdout: str, stderr: str = ""):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = 0


@pytest.mark.parametrize("last_line", ["true", "42", '["ok"]', "null", '"hi"'])
def test_valid_non_object_json_fails_closed_without_raising(last_line):
    """A valid-but-non-dict JSON last line must fail CLOSED, not raise."""
    with patch(
        "meta_n.core.verified_code.subprocess.run",
        return_value=_FakeProc(stdout=f"noise\n{last_line}\n"),
    ):
        # Pre-fix: AttributeError escapes here.
        result = _docker_run_json(["docker", "run"], timeout_s=5.0, container="c")

    assert isinstance(result, dict)
    assert result.get("passed") is False
    assert result.get("status") == "no_json"
    # structured record preserved (not swallowed by an upstream broad except)
    assert "raw_stdout" in result
    assert "raw_stderr" in result


def test_valid_object_json_still_passes_through():
    """A conforming JSON object last line keeps its keys and gains status=ok."""
    with patch(
        "meta_n.core.verified_code.subprocess.run",
        return_value=_FakeProc(stdout='{"passed": true}\n'),
    ):
        result = _docker_run_json(["docker", "run"], timeout_s=5.0, container="c")

    assert result == {"passed": True, "status": "ok"}
