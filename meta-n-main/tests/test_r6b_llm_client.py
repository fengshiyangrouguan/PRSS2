"""§6b F068 — empty-content escalation cap is relative to the caller's scale.

The re-issue cap is ``min(empty_content_retry_max_tokens,
_EMPTY_ESCALATION_SCALE * eff_max_tokens)``: the absolute config cap (default
32768) was calibrated against the 16384 OUTER default, while inner ``llm()``
closures default to ``max_tokens=512`` — without the relative component every
fanned-out inner call from a systematically-empty model re-issued at 64x its
requested scale. Invariants pinned here:

* firing condition unchanged — ``min(cap, 4*eff) > eff`` iff ``cap > eff``;
* the config ``0`` disable knob unchanged;
* single-escalation bound unchanged;
* outer-default behaviour (16384 / 32768) byte-identical.
"""

from unittest.mock import AsyncMock, MagicMock

from meta_n.core.llm_client import _EMPTY_ESCALATION_SCALE, LLMClient, LLMConfig


def _resp(content, *, finish_reason, total_tokens=512) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    r.choices[0].finish_reason = finish_reason
    r.usage = MagicMock()
    r.usage.prompt_tokens = 100
    r.usage.completion_tokens = total_tokens
    r.usage.total_tokens = total_tokens + 100
    r.usage.prompt_tokens_details = None
    return r


def _client(*, max_tokens: int = 16384, escalate_cap: int = 32768) -> LLMClient:
    config = LLMConfig(
        api_key="test-key",
        max_tokens=max_tokens,
        empty_content_retry_max_tokens=escalate_cap,
    )
    return LLMClient(config)


class TestEscalationCapRelativeToCallerScale:
    async def test_escalation_cap_relative_to_caller_scale(self):
        """An inner-scale call (max_tokens=512) that comes back empty+length
        re-issues ONCE at 4x its own scale (2048), not at the 32768 config cap."""
        client = _client()
        empty = _resp("", finish_reason="length", total_tokens=512)
        recovered = _resp("flu", finish_reason="stop", total_tokens=50)
        mock_create = AsyncMock(side_effect=[empty, recovered])
        client._client.chat.completions.create = mock_create

        text, _, _, _ = await client.complete_with_breakdown(
            messages=[{"role": "user", "content": "classify"}], max_tokens=512,
        )

        assert mock_create.call_count == 2
        assert mock_create.call_args_list[0][1]["max_tokens"] == 512
        assert mock_create.call_args_list[1][1]["max_tokens"] == 2048
        assert mock_create.call_args_list[1][1]["max_tokens"] == (
            _EMPTY_ESCALATION_SCALE * 512
        )
        assert text == "flu"

    async def test_outer_default_escalation_cap_unchanged(self):
        """Outer-default scale (16384 / config 32768): min(32768, 65536) is a
        no-op, so the re-issue lands at exactly 32768 — byte-identical to the
        pre-F068 calibrated behaviour."""
        client = _client(max_tokens=16384, escalate_cap=32768)
        empty = _resp("", finish_reason="length", total_tokens=16384)
        recovered = _resp("#!/bin/bash\necho ok", finish_reason="stop",
                          total_tokens=2000)
        mock_create = AsyncMock(side_effect=[empty, recovered])
        client._client.chat.completions.create = mock_create

        text, _ = await client.complete(
            messages=[{"role": "user", "content": "author a script"}]
        )

        assert mock_create.call_count == 2
        assert mock_create.call_args_list[0][1]["max_tokens"] == 16384
        assert mock_create.call_args_list[1][1]["max_tokens"] == 32768
        assert text == "#!/bin/bash\necho ok"

    async def test_escalation_disabled_zero_config_small_caller(self):
        """Config cap 0 stays the full opt-out at every caller scale:
        min(0, 4*512) == 0 fails the gate, so a single call returns ''."""
        client = _client(escalate_cap=0)
        empty = _resp("", finish_reason="length", total_tokens=512)
        mock_create = AsyncMock(return_value=empty)
        client._client.chat.completions.create = mock_create

        text, _, _, _ = await client.complete_with_breakdown(
            messages=[{"role": "user", "content": "x"}], max_tokens=512,
        )
        assert mock_create.call_count == 1
        assert text == ""

    async def test_escalation_gate_still_requires_cap_above_request(self):
        """The min() must not widen the firing condition: config cap 8192 with
        a 16384-token request (min(8192, 65536) == 8192 <= 16384) never
        escalates, exactly as before."""
        client = _client(max_tokens=16384, escalate_cap=8192)
        empty = _resp("", finish_reason="length", total_tokens=16384)
        mock_create = AsyncMock(return_value=empty)
        client._client.chat.completions.create = mock_create

        text, _ = await client.complete(messages=[{"role": "user", "content": "x"}])
        assert mock_create.call_count == 1
        assert text == ""

    async def test_escalated_reissue_cannot_escalate_again_at_small_scale(self):
        """Single-escalation bound under the relative cap: the 2048 re-issue
        recomputes its own (higher) cap but _empty_retry_done blocks a second
        escalation — exactly 2 create() calls, '' returned."""
        client = _client()
        empty1 = _resp("", finish_reason="length", total_tokens=512)
        empty2 = _resp("", finish_reason="length", total_tokens=2048)
        mock_create = AsyncMock(side_effect=[empty1, empty2])
        client._client.chat.completions.create = mock_create

        text, _, _, _ = await client.complete_with_breakdown(
            messages=[{"role": "user", "content": "x"}], max_tokens=512,
        )
        assert mock_create.call_count == 2
        assert mock_create.call_args_list[1][1]["max_tokens"] == 2048
        assert text == ""
