from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


BENCH_ROOT = (Path(__file__).resolve().parents[2] / "src/roboweave/_vendor/bench")
LAB_ROOT = BENCH_ROOT / "lab"
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from anthropic_messages_adapter import (  # noqa: E402
    AnthropicMessagesAdapterError,
    assemble_anthropic_sse,
    chat_request_to_anthropic,
    anthropic_to_chat_completion,
    extract_anthropic_replay_content,
    parse_sse_json_events,
)


PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X89QAAAAASUVORK5CYII="
)


def test_stream_without_terminal_event_is_rejected():
    with pytest.raises(AnthropicMessagesAdapterError, match='terminal event'):
        assemble_anthropic_sse([
            {'type': 'message_start', 'message': {'content': [], 'usage': {}}},
            {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use'}},
        ])


def test_truncated_response_with_tool_keeps_length_finish_reason():
    result = anthropic_to_chat_completion({
        'stop_reason': 'max_tokens',
        'content': [{'type': 'tool_use', 'id': 't1', 'name': 'robot_action',
                     'input': {'action': 'wait'}}],
        'usage': {},
    })
    assert result['choices'][0]['finish_reason'] == 'length'


def _chat_request(**overrides):
    request = {
        "model": "claude-opus-5",
        "messages": [
            {"role": "system", "content": "You control an R1Pro robot."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Task: wait once"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{PNG}"},
                    },
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "robot_action",
                    "description": "Control the robot",
                    "parameters": {
                        "type": "object",
                        "properties": {"action": {"type": "string"}},
                        "required": ["action"],
                    },
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "robot_action"},
        },
        "temperature": 0.6,
        "top_p": 0.9,
        "parallel_tool_calls": False,
    }
    request.update(overrides)
    return request


def test_messages_adapter_sends_effort_max_stream_and_auto_tools():
    wire = chat_request_to_anthropic(
        _chat_request(),
        max_tokens=64000,
        effort="max",
    )
    assert wire["model"] == "claude-opus-5"
    assert wire["max_tokens"] == 64000
    assert wire["stream"] is True
    assert wire["output_config"] == {"effort": "max"}
    assert wire["tool_choice"] == {"type": "auto"}
    assert "temperature" not in wire
    assert "top_p" not in wire
    assert "thinking" not in wire
    assert "reasoning_context" not in wire
    assert wire["system"] == "You control an R1Pro robot."
    assert wire["messages"][0]["role"] == "user"
    image = wire["messages"][0]["content"][1]
    assert image["type"] == "image"
    assert image["source"]["media_type"] == "image/png"
    assert image["source"]["data"] == PNG
    assert wire["tools"][0]["name"] == "robot_action"
    assert wire["tools"][0]["input_schema"]["required"] == ["action"]


def test_messages_adapter_merges_tool_result_and_followup_image():
    request = _chat_request(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "robot_action",
                            "arguments": '{"action":"wait"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"status":"completed"}',
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "new observation"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{PNG}"},
                    },
                ],
            },
        ]
    )
    wire = chat_request_to_anthropic(request, max_tokens=64000, effort="max")
    assert [item["role"] for item in wire["messages"]] == ["user", "assistant", "user"]
    follow = wire["messages"][2]["content"]
    assert follow[0]["type"] == "tool_result"
    assert follow[0]["tool_use_id"] == "call_1"
    assert follow[1]["type"] == "text"
    assert follow[2]["type"] == "image"


def test_messages_adapter_replays_thinking_signature_blocks():
    replay = [
        {
            "type": "thinking",
            "thinking": "hidden",
            "signature": "sig-abc",
        },
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "robot_action",
            "input": {"action": "wait"},
        },
    ]
    request = _chat_request(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": None,
                "reasoning": "do not send this glm field",
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {
                            "name": "robot_action",
                            "arguments": '{"action":"wait"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "content": "{}"},
            {"role": "user", "content": "next"},
        ]
    )
    wire = chat_request_to_anthropic(
        request,
        max_tokens=64000,
        effort="max",
        replay_content_by_call_id={"toolu_1": replay},
    )
    assert wire["messages"][1]["content"] == replay
    assert "do not send this glm field" not in json.dumps(wire)


def test_messages_adapter_drops_glm_reasoning_without_replay():
    request = _chat_request(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": None,
                "reasoning": "private glm chain",
                "tool_calls": [
                    {
                        "id": "chatcmpl-tool-1",
                        "type": "function",
                        "function": {
                            "name": "robot_action",
                            "arguments": '{"action":"pick","target":6}',
                        },
                    }
                ],
            },
        ]
    )
    wire = chat_request_to_anthropic(request, max_tokens=64000, effort="max")
    dumped = json.dumps(wire)
    assert "private glm chain" not in dumped
    assert wire["messages"][1]["content"][0]["type"] == "tool_use"
    assert wire["messages"][1]["content"][0]["input"]["target"] == 6


def test_anthropic_response_maps_tool_use_and_length():
    normalized = anthropic_to_chat_completion(
        {
            "id": "msg_1",
            "model": "claude-fable-5-1",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_9",
                    "name": "robot_action",
                    "input": {"action": "wait"},
                }
            ],
            "usage": {"input_tokens": 12, "output_tokens": 8},
        }
    )
    choice = normalized["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["id"] == "toolu_9"
    length = anthropic_to_chat_completion(
        {
            "id": "msg_2",
            "model": "claude-opus-5",
            "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": "partial"}],
            "usage": {"input_tokens": 1, "output_tokens": 64000},
        }
    )
    assert length["choices"][0]["finish_reason"] == "length"


def test_anthropic_keeps_visible_commentary_out_of_reasoning():
    from Agent import parse_native_chat_completion

    normalized = anthropic_to_chat_completion(
        {
            "id": "msg_fold",
            "model": "claude-opus-5",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "wait is enough",
                },
                {
                    "type": "text",
                    "text": "I'll start by observing the scene.",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_wait",
                    "name": "robot_action",
                    "input": {"action": "wait"},
                },
            ],
            "usage": {"input_tokens": 8, "output_tokens": 12},
        }
    )
    message = normalized["choices"][0]["message"]
    assert message["content"] is None
    assert message["reasoning"] == "wait is enough"
    assert "I'll start by observing the scene." not in message["reasoning"]
    commentary = normalized["anthropic_visible_commentary"]
    assert commentary["label"] == "可见旁白"
    assert commentary["text"] == "I'll start by observing the scene."
    assert commentary["hidden_reasoning"] is False
    assert commentary["counted_as_reasoning_tokens"] is False
    assert commentary["echoed_as_thinking_block"] is False
    assert "reasoning_tokens" not in normalized["usage"]
    replay = extract_anthropic_replay_content(
        {
            "content": [
                {"type": "thinking", "thinking": "wait is enough", "signature": "sig"},
                {"type": "text", "text": "I'll start by observing the scene."},
                {
                    "type": "tool_use",
                    "id": "toolu_wait",
                    "name": "robot_action",
                    "input": {"action": "wait"},
                },
            ]
        }
    )
    assert replay[1] == {"type": "text", "text": "I'll start by observing the scene."}
    assert replay[0]["type"] == "thinking"
    decision = parse_native_chat_completion(normalized)
    assert decision.action == "wait"
    assert decision.target is None


def test_sse_assembler_keeps_signature_and_tool_json():
    payload = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"msg","model":"claude-opus-5","role":"assistant","content":[],"usage":{"input_tokens":3}}}\n\n'
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}\n\n'
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"robot_action","input":{}}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"action\\":\\"wait\\"}"}}\n\n'
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}\n\n'
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n\n'
    )
    message = assemble_anthropic_sse(parse_sse_json_events(payload))
    assert message["content"][0]["signature"] == "sig"
    assert message["content"][1]["input"] == {"action": "wait"}
    assert message["stop_reason"] == "tool_use"


def test_sse_assembler_finalizes_split_tool_json():
    payload = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"msg","role":"assistant","content":[],"usage":{}}}\n\n'
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_2","name":"robot_action","input":{}}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"act"}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ion\\":\\"wait\\"}"}}\n\n'
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
    )
    message = assemble_anthropic_sse(parse_sse_json_events(payload))
    assert message["content"][0]["input"] == {"action": "wait"}
    assert "_input_json" not in message["content"][0]


def test_adapter_rejects_gpt_reasoning_fields_by_not_copying_them():
    wire = chat_request_to_anthropic(_chat_request(), max_tokens=16, effort="high")
    for forbidden in (
        "reasoning_context",
        "reasoning_summary",
        "reasoning_effort",
        "text_verbosity",
        "seed",
        "temperature",
    ):
        assert forbidden not in wire
    with pytest.raises(AnthropicMessagesAdapterError):
        chat_request_to_anthropic(_chat_request(), max_tokens=16, effort="adaptive")
