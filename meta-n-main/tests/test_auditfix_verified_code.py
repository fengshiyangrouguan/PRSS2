"""Offline regression tests for audit findings 52 and 53 in
``meta_n/core/verified_code.py``.

Both tests are fully offline (no Docker, no LLM, no network):

#52 — ``SandboxedHeldoutVerifier.verify`` must report ``ran_in_sandbox=False``
      for a HOST-side crash that happens before the container is ever invoked.
#53 — ``_docker_run_json`` timeout-cleanup ``docker rm -f`` must be issued with
      a bounded ``timeout=`` so a wedged daemon cannot block past wall-clock.
"""

import subprocess

import pytest

from meta_n.core import verified_code
from meta_n.core.verified_code import SandboxedHeldoutVerifier, _docker_run_json


def test_finding52_host_side_crash_not_labeled_ran_in_sandbox():
    """A crash before the container runs must yield ran_in_sandbox=False.

    We force a host-side failure that occurs BEFORE ``_run_container`` is ever
    called by handing the verifier a non-str ``runner_src``: ``Path.write_text``
    raises ``TypeError`` while staging ``_runner.py`` (line ~193), well before the
    container invocation. ``run_container`` is a fail-fast sentinel so that if the
    code path somehow reached the container, the test would error loudly.

    On the ORIGINAL code the broad ``except`` returned ``ran_in_sandbox=True`` for
    this host crash; after the fix it honestly reports ``ran_in_sandbox=False``.
    """

    def _must_not_run(argv, timeout_s, container):  # pragma: no cover - guard
        raise AssertionError("container must not be invoked for a host-side crash")

    verifier = SandboxedHeldoutVerifier(
        image="dummy:latest",
        runner_src=b"not-a-str",  # bytes -> Path.write_text raises before container
        run_container=_must_not_run,
    )

    res = verifier.verify(
        name="helper",
        source="def helper():\n    return 1\n",
        task_id="t0",
        context_sources=[],
    )

    assert res.passed is False
    # The helper never executed inside the sandbox -> provenance must be honest.
    assert res.ran_in_sandbox is False


def test_finding52_real_container_run_still_labeled_ran_in_sandbox():
    """Sanity guard: a normal in-container run still reports ran_in_sandbox=True."""

    def _fake_run(argv, timeout_s, container):
        return {"status": "ok", "passed": True}

    verifier = SandboxedHeldoutVerifier(
        image="dummy:latest",
        runner_src="print('hi')",
        run_container=_fake_run,
    )
    res = verifier.verify(
        name="helper",
        source="def helper():\n    return 1\n",
        task_id="t0",
        context_sources=[],
    )
    assert res.passed is True
    assert res.ran_in_sandbox is True


def test_finding53_timeout_cleanup_is_bounded(monkeypatch):
    """The ``docker rm -f`` timeout-cleanup must pass a bounded ``timeout=``.

    We replace ``subprocess.run`` so the primary docker-run call raises
    ``TimeoutExpired`` and the subsequent cleanup call is recorded. On the
    ORIGINAL code the cleanup call had no ``timeout=`` kwarg (and so could hang on
    a wedged daemon); after the fix it carries a bounded timeout.
    """
    calls = []

    def _fake_run(argv, *args, **kwargs):
        calls.append((argv, kwargs))
        # First call = the primary `docker run ...` (bounded by timeout_s).
        if argv and argv[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))
        # Subsequent call = the `docker rm -f <container>` cleanup.
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(verified_code.subprocess, "run", _fake_run)

    result = _docker_run_json(
        ["docker", "run", "--rm", "--name", "c0", "img"],
        timeout_s=5.0,
        container="c0",
    )

    assert result == {"status": "timeout", "passed": False}

    cleanup = [c for c in calls if c[0][:3] == ["docker", "rm", "-f"]]
    assert cleanup, "expected a `docker rm -f` cleanup call after timeout"
    argv, kwargs = cleanup[0]
    assert argv == ["docker", "rm", "-f", "c0"]
    # The bug: cleanup had no bounded timeout. After the fix it must be bounded.
    assert "timeout" in kwargs and kwargs["timeout"] is not None
    assert kwargs["timeout"] > 0


def test_finding53_cleanup_timeout_does_not_propagate(monkeypatch):
    """If the cleanup itself times out, the function must still fail closed."""

    def _fake_run(argv, *args, **kwargs):
        if argv[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))
        # Cleanup also wedges -> must be swallowed, not raised.
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(verified_code.subprocess, "run", _fake_run)

    result = _docker_run_json(
        ["docker", "run", "--rm", "--name", "c1", "img"],
        timeout_s=5.0,
        container="c1",
    )
    assert result == {"status": "timeout", "passed": False}
