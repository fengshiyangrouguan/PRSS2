"""Regression test for SPINE finding #2 (proxy + docker daemon env passthrough).

Offline: no Docker / LLM / network. Verifies that :func:`scrubbed_child_env`
now carries the outbound-proxy and Docker-daemon connection config into the
child runner (so a proxied host / remote-or-rootless daemon still works), while
meta-n secrets are still kept out of the allowlisted base.
"""

from __future__ import annotations

import pytest

from meta_n.core.external_agents._bridge import SAFE_ENV_KEYS, scrubbed_child_env


PROXY_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)

DOCKER_VARS = (
    "DOCKER_HOST",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
    "XDG_RUNTIME_DIR",
)


@pytest.mark.parametrize("var", PROXY_VARS + DOCKER_VARS)
def test_config_var_in_allowlist(var: str) -> None:
    assert var in SAFE_ENV_KEYS


def test_representative_proxy_and_docker_host_survive_scrub(monkeypatch: pytest.MonkeyPatch) -> None:
    # Start from a clean env so we control exactly what os.environ carries.
    for k in list(SAFE_ENV_KEYS) + ["OPENAI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.setenv("https_proxy", "http://proxy.internal:8080")
    monkeypatch.setenv("DOCKER_HOST", "tcp://10.0.0.9:2376")
    # A meta-n secret that must NOT cross into the child from os.environ.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-sentinel-must-not-leak")

    env = scrubbed_child_env()

    # Egress / daemon CONFIG survives the scrub.
    assert env.get("HTTPS_PROXY") == "http://proxy.internal:8080"
    assert env.get("https_proxy") == "http://proxy.internal:8080"
    assert env.get("DOCKER_HOST") == "tcp://10.0.0.9:2376"

    # Secret is scrubbed from the allowlisted base (not sourced from os.environ).
    assert "OPENAI_API_KEY" not in env


def test_secret_still_injectable_via_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # Existing policy unchanged: the provider key is injected explicitly, not
    # inherited from os.environ.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = scrubbed_child_env({"OPENAI_API_KEY": "sk-explicit"})
    assert env["OPENAI_API_KEY"] == "sk-explicit"
