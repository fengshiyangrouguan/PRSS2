"""reasoning_effort config plumbing (LM Studio / reasoning-model control).

Contract: when set, ``reasoning_effort`` rides the ``extra_body`` passthrough to
every chat completion; when unset (default None) requests are byte-identical.
It lives on LLMConfig (not the client instance) so it survives ``asdict(config)``
into the subprocess-isolated evaluators — the reason a client-instance-only
override reached search but not test evaluation.
"""

from dataclasses import asdict

from meta_n.core.llm_client import LLMConfig, LLMClient


def _client(**kw):
    cfg = LLMConfig(
        model="google/gemma-4-31b-qat",
        base_url="http://localhost:1234/v1",
        api_key="dummy",
        **kw,
    )
    return LLMClient(cfg), cfg


def test_default_none_leaves_extra_body_unchanged():
    client, _ = _client()
    assert client._extra_body is None  # byte-identical: no field sent


def test_reasoning_effort_rides_extra_body():
    client, _ = _client(reasoning_effort="none")
    assert client._extra_body == {"reasoning_effort": "none"}


def test_reasoning_effort_merges_with_provider_routing():
    client, _ = _client(
        reasoning_effort="low", exclude_providers=["foo"]
    )
    assert client._extra_body == {
        "provider": {"ignore": ["foo"]},
        "reasoning_effort": "low",
    }


def test_survives_asdict_roundtrip_into_subprocess_config():
    # The text-classification evaluators rebuild the client in a spawned
    # subprocess from ``asdict(self._llm_client.config)``; the field must be
    # present in that dict, unlike an instance-only ``_extra_body`` override.
    _, cfg = _client(reasoning_effort="none")
    d = asdict(cfg)
    assert d["reasoning_effort"] == "none"
    rebuilt = LLMClient(LLMConfig(**d))
    assert rebuilt._extra_body == {"reasoning_effort": "none"}


def test_field_default_is_none():
    assert LLMConfig(model="m").reasoning_effort is None
