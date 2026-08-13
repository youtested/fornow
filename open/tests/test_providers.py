"""Tests for the OpenAI-compatible provider: content-part deltas, the
stream_options flag, and flushing the SSE tail on stream end."""

import json
from unittest import mock

import pytest

from opencode_py.providers.base import ContextOverflowError, ProviderEvent, Usage
from opencode_py.providers.openai_compat import (
    OpenAICompatProvider,
    _content_to_text,
    _is_context_overflow_message,
)


class FakeResponse:
    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_bytes(self):
        yield from self._chunks


class FakeClient:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, *a, **k):
        return FakeResponse(self._chunks)


def provider(**kwargs):
    return OpenAICompatProvider(base_url="https://example.com", api_key="k", model="m", **kwargs)


def test_handle_content_string():
    p = provider()
    events = []
    evt = {"data": json.dumps({"choices": [{"delta": {"content": "hello"}}]})}
    p._handle_event(evt, events.append, {}, None)
    assert [e.text for e in events if e.kind == "text_delta"] == ["hello"]


def test_handle_content_list_parts():
    p = provider()
    events = []
    evt = {
        "data": json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {"type": "text", "text": "hello "},
                                {"type": "text", "text": "world"},
                            ]
                        }
                    }
                ]
            }
        )
    }
    p._handle_event(evt, events.append, {}, None)
    assert [e.text for e in events if e.kind == "text_delta"] == ["hello world"]


def test_content_to_text_variants():
    assert _content_to_text("plain") == "plain"
    assert _content_to_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert _content_to_text([{"text": "x"}]) == "x"
    assert _content_to_text(["raw"]) == "raw"
    assert _content_to_text({"unexpected": True}) == "{'unexpected': True}"


def test_handle_reasoning_content():
    p = provider()
    events = []
    evt = {"data": json.dumps({"choices": [{"delta": {"reasoning_content": "think..."}}]})}
    p._handle_event(evt, events.append, {}, None)
    assert [e.text for e in events if e.kind == "reasoning_delta"] == ["think..."]


def test_usage_in_every_chunk_keeps_content():
    # Some gateways (e.g. the Zen router) attach a usage object to every SSE
    # chunk alongside the content delta; the content must not be dropped.
    p = provider()
    events = []
    usage = Usage()
    for data in [
        {"choices": [{"delta": {"reasoning_content": "think"}}], "usage": {"total_tokens": 1}},
        {"choices": [{"delta": {"content": "hi"}}], "usage": {"total_tokens": 2}},
    ]:
        p._handle_event({"data": json.dumps(data)}, events.append, {}, usage)
    assert [e.text for e in events if e.kind == "reasoning_delta"] == ["think"]
    assert [e.text for e in events if e.kind == "text_delta"] == ["hi"]
    assert usage.total_tokens == 2


def test_usage_only_ping_no_content():
    p = provider()
    events = []
    usage = mock.MagicMock()
    usage.input_tokens = usage.output_tokens = usage.total_tokens = 0
    evt = {"data": json.dumps({"choices": [], "cost": "0", "usage": {"total_tokens": 5}})}
    p._handle_event(evt, events.append, {}, usage)
    assert events == []
    assert usage.total_tokens == 5


def test_build_payload_stream_options_on_by_default():
    p = provider()
    payload = p.build_payload([{"role": "user", "content": "hi"}])
    assert payload["stream_options"] == {"include_usage": True}


def test_build_payload_stream_options_disableable():
    p = provider(include_usage=False)
    payload = p.build_payload([{"role": "user", "content": "hi"}])
    assert "stream_options" not in payload


def test_stream_flushes_tail_without_newline():
    p = provider()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}',  # no trailing newline
    ]
    events = []
    with mock.patch("opencode_py.providers.openai_compat.httpx.Client", return_value=FakeClient(chunks)):
        p.stream_chat([], [], events.append)
    text = "".join(e.text for e in events if e.kind == "text_delta")
    assert text == "hello"


def test_stream_tail_done_sentinel_without_newline():
    p = provider()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        b"data: [DONE]",  # no trailing newline
    ]
    events = []
    with mock.patch("opencode_py.providers.openai_compat.httpx.Client", return_value=FakeClient(chunks)):
        p.stream_chat([], [], events.append)
    text = "".join(e.text for e in events if e.kind == "text_delta")
    assert text == "hi"


def test_context_overflow_marker_detection():
    assert _is_context_overflow_message("context_length_exceeded: too long")
    assert _is_context_overflow_message("this model's maximum context length is 8192 tokens")
    assert _is_context_overflow_message("reduce_other_history to keep the conversation under the token limit")
    assert not _is_context_overflow_message("rate limit reached")
    assert not _is_context_overflow_message("")


def test_context_overflow_broad_real_world_messages():
    from opencode_py.providers.classify import is_context_overflow

    assert is_context_overflow(
        "prompt was truncated because it exceeded the max context length of 128000 tokens"
    )
    assert is_context_overflow(
        "This model's maximum context length is 8192 tokens. However, your messages resulted in about 9000 tokens."
    )
    assert is_context_overflow("400 Bad Request: request_too_large")
    assert is_context_overflow("exceeds the context window")
    assert is_context_overflow("exceeds context window 200k")
    assert is_context_overflow("input token count exceeds max")
    assert is_context_overflow("input token count exceeds the maximum")
    assert is_context_overflow("exceeds the maximum allowed input length of 131072 tokens")
    assert is_context_overflow("error: {'code': 'model_context_window_exceeded'}")
    # rate-limit wording must NEVER be classified as overflow
    assert not is_context_overflow("throttling error: rate limit reached")
    assert not is_context_overflow("rate_limit_exceeded: too many requests this minute")
    assert not is_context_overflow("quota exceeded")
    assert not is_context_overflow("insufficient_quota")
    assert not is_context_overflow("")


def test_stream_error_event_context_overflow_raises():
    p = provider()
    events = []
    # context_length_exceeded must raise ContextOverflowError (not sink as error)
    with pytest.raises(ContextOverflowError):
        p._handle_event(
            {"data": json.dumps({"error": {"message": "budget too long", "code": "context_length_exceeded"}})},
            events.append,
            {},
            None,
        )


def test_context_overflow_by_message_text_raises():
    p = provider()
    events = []
    # some gateways only report the overflow in the message, without a code
    with pytest.raises(ContextOverflowError):
        p._handle_event(
            {"data": json.dumps({"error": {"message": "this model's maximum context length is 4000 tokens"}})},
            events.append,
            {},
            None,
        )


def test_non_context_error_still_sinks_error_event():
    p = provider()
    events = []
    p._handle_event(
        {"data": json.dumps({"error": {"message": "server hiccup", "code": "server_error"}})},
        events.append,
        {},
        None,
    )
    assert any(e.kind == "error" for e in events)


class RecordingClient:
    """Fake httpx.Client that captures the request JSON (like a real one,
    where json.dumps would reject non-serializable payload values)."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.request_payload = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, *a, json=None, **k):
        self.request_payload = json
        return FakeResponse(self._chunks)


def test_is_interrupted_not_leaked_into_request_payload():
    # Regression: the engine passes `is_interrupted` (a callback) down to the
    # provider. It must be popped before build_payload, or it lands in the JSON
    # body and real httpx fails with `TypeError: function is not JSON
    # serializable`, killing the stream.
    p = provider()
    client = RecordingClient([b"data: [DONE]\n\n"])
    with mock.patch("opencode_py.providers.openai_compat.httpx.Client", return_value=client):
        p.stream_chat(
            [{"role": "user", "content": "hi"}],
            [],
            events := [].append,
            is_interrupted=lambda: False,
        )
    assert "is_interrupted" not in client.request_payload
    json.dumps(client.request_payload)  # must be serializable


def test_is_interrupted_mid_stream_raises_stream_interrupted():
    from opencode_py.providers.base import StreamInterrupted

    p = provider()
    flag = {"interrupted": False}

    def on_event(evt):
        if evt.kind == "text_delta":
            flag["interrupted"] = True

    chunks = [
        b'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    with mock.patch("opencode_py.providers.openai_compat.httpx.Client", return_value=FakeClient(chunks)):
        with pytest.raises(StreamInterrupted):
            p.stream_chat(
                [{"role": "user", "content": "hi"}],
                [],
                on_event,
                is_interrupted=lambda: flag["interrupted"],
            )


def test_anthropic_is_interrupted_not_leaked_into_request_payload():
    from opencode_py.providers.anthropic import AnthropicProvider

    p = AnthropicProvider(base_url="https://api.anthropic.com/v1", api_key="k", model="claude")
    client = RecordingClient([b"data: [DONE]\n\n"])
    with mock.patch("opencode_py.providers.anthropic.httpx.Client", return_value=client):
        p.stream_chat(
            [{"role": "user", "content": "hi"}],
            [],
            [].append,
            is_interrupted=lambda: False,
        )
    assert "is_interrupted" not in client.request_payload
    json.dumps(client.request_payload)  # must be serializable


def test_anthropic_is_interrupted_mid_stream_raises_stream_interrupted():
    from opencode_py.providers.anthropic import AnthropicProvider
    from opencode_py.providers.base import StreamInterrupted

    p = AnthropicProvider(base_url="https://api.anthropic.com/v1", api_key="k", model="claude")
    flag = {"interrupted": False}

    def on_event(evt):
        if evt.kind == "text_delta":
            flag["interrupted"] = True

    chunks = [
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello "}}\n\n',
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}\n\n',
        b"data: [DONE]\n\n",
    ]
    with mock.patch("opencode_py.providers.anthropic.httpx.Client", return_value=FakeClient(chunks)):
        with pytest.raises(StreamInterrupted):
            p.stream_chat(
                [{"role": "user", "content": "hi"}],
                [],
                on_event,
                is_interrupted=lambda: flag["interrupted"],
            )
