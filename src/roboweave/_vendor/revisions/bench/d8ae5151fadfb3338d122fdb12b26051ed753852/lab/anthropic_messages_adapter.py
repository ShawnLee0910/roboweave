"""Chat Completions to Anthropic Messages adapter for Action20 Claude models.

The canonical Action20 loop stays on OpenAI-shaped messages. Only the provider
wire representation changes. Adaptive thinking is left at the API default: the
adapter never sends a thinking budget, never disables thinking, and never maps
GPT reasoning_context/summary onto Claude.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit


ADAPTER_VERSION = "esi.anthropic-messages-adapter/v1-visible-commentary"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
VISIBLE_COMMENTARY_LABEL = "可见旁白"
ANTHROPIC_STOP_TO_FINISH = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "stop_sequence": "stop",
    "refusal": "stop",
    "pause_turn": "stop",
}


class AnthropicMessagesAdapterError(ValueError):
    """The Chat Completions payload cannot be expressed as Anthropic Messages."""


def anthropic_messages_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path in {"", "/"}:
        path = "/v1"
    elif not path.endswith("/v1"):
        path = f"{path}/v1"
    return f"{parsed.scheme}://{parsed.netloc}{path}/messages"


def chat_request_to_anthropic(
    request: Mapping[str, Any],
    *,
    max_tokens: int,
    effort: str,
    replay_content_by_call_id: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise AnthropicMessagesAdapterError("native request must be a mapping")
    model = request.get("model")
    if not isinstance(model, str) or not model.strip():
        raise AnthropicMessagesAdapterError("native request model is invalid")
    if effort not in ANTHROPIC_EFFORTS:
        raise AnthropicMessagesAdapterError("Anthropic effort is invalid")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise AnthropicMessagesAdapterError("Anthropic max_tokens is invalid")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise AnthropicMessagesAdapterError("native request messages are invalid")
    system, anthropic_messages = _convert_messages(
        messages,
        replay_content_by_call_id=replay_content_by_call_id or {},
    )
    wire: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "output_config": {"effort": effort},
        "messages": anthropic_messages,
    }
    if system:
        wire["system"] = system
    tools = request.get("tools")
    if tools:
        wire["tools"] = [_convert_tool(tool) for tool in tools]
        wire["tool_choice"] = {"type": "auto"}
    return wire


def anthropic_to_chat_completion(response: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise AnthropicMessagesAdapterError("Anthropic response must be a mapping")
    content = response.get("content")
    if not isinstance(content, list):
        raise AnthropicMessagesAdapterError("Anthropic content is invalid")
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, Mapping):
            raise AnthropicMessagesAdapterError("Anthropic content block is invalid")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise AnthropicMessagesAdapterError("Anthropic text block is invalid")
            if text:
                text_parts.append(text)
        elif block_type in {"thinking", "redacted_thinking"}:
            thinking = block.get("thinking")
            if isinstance(thinking, str) and thinking:
                thinking_parts.append(thinking)
        elif block_type == "tool_use":
            tool_calls.append(_tool_use_to_chat(block))
        elif block_type in {"server_tool_use", "web_search_tool_result"}:
            raise AnthropicMessagesAdapterError(
                f"unsupported Anthropic block type {block_type}"
            )
    stop_reason = response.get("stop_reason")
    finish_reason = ANTHROPIC_STOP_TO_FINISH.get(
        stop_reason if isinstance(stop_reason, str) else "",
        "stop",
    )
    folded_visible_text = False
    visible_commentary = None
    if tool_calls:
        if text_parts:
            # Keep the original visible text. Action20's shared parser forbids
            # assistant content beside robot_action, so the OpenAI-shaped
            # message content is cleared. This is not hidden reasoning, not a
            # thinking-block echo, and not counted as reasoning tokens.
            visible_commentary = {
                "label": VISIBLE_COMMENTARY_LABEL,
                "text": "".join(text_parts),
                "hidden_reasoning": False,
                "counted_as_reasoning_tokens": False,
                "echoed_as_thinking_block": False,
            }
            text_parts = []
            folded_visible_text = True
    message: dict[str, Any] = {
        "role": "assistant",
        "content": None if not text_parts else "".join(text_parts),
        "tool_calls": tool_calls or None,
    }
    if thinking_parts:
        message["reasoning"] = "\n".join(thinking_parts)
    if message["tool_calls"] is None:
        message.pop("tool_calls")
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    prompt_tokens = _optional_int(usage.get("input_tokens"))
    completion_tokens = _optional_int(usage.get("output_tokens"))
    cache_creation = _optional_int(usage.get("cache_creation_input_tokens"))
    cache_read = _optional_int(usage.get("cache_read_input_tokens"))
    chat_usage: dict[str, Any] = {}
    if prompt_tokens is not None:
        chat_usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        chat_usage["completion_tokens"] = completion_tokens
    if prompt_tokens is not None and completion_tokens is not None:
        chat_usage["total_tokens"] = prompt_tokens + completion_tokens
    if cache_creation is not None:
        chat_usage["cache_creation_input_tokens"] = cache_creation
    if cache_read is not None:
        chat_usage["cache_read_input_tokens"] = cache_read
    record = {
        "id": response.get("id"),
        "object": "chat.completion",
        "model": response.get("model"),
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": message,
            }
        ],
        "usage": chat_usage,
        "stop_reason": stop_reason,
        "anthropic_adapter_version": ADAPTER_VERSION,
    }
    if folded_visible_text and visible_commentary is not None:
        record["anthropic_visible_commentary"] = visible_commentary
    return record


def extract_anthropic_replay_content(response: Mapping[str, Any]) -> list[Any]:
    content = response.get("content")
    if not isinstance(content, list):
        raise AnthropicMessagesAdapterError("Anthropic content is invalid")
    return json.loads(json.dumps(content))


def assemble_anthropic_sse(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] | None = None
    blocks: dict[int, dict[str, Any]] = {}
    output_tokens: int | None = None
    stop_reason: str | None = None
    saw_message_stop = False
    for event in events:
        if not isinstance(event, Mapping):
            raise AnthropicMessagesAdapterError("Anthropic SSE event is invalid")
        event_type = event.get("type")
        if event_type == "message_stop":
            saw_message_stop = True
            continue
        if event_type in {"ping", "content_block_stop"}:
            continue
        if event_type == "error":
            error = event.get("error")
            detail = error if isinstance(error, Mapping) else event
            raise AnthropicMessagesAdapterError(
                f"Anthropic stream error: {json.dumps(detail, ensure_ascii=False)[:512]}"
            )
        if event_type == "message_start":
            started = event.get("message")
            if not isinstance(started, Mapping):
                raise AnthropicMessagesAdapterError("message_start payload is invalid")
            message = json.loads(json.dumps(started))
            continue
        if event_type == "content_block_start":
            index = event.get("index")
            block = event.get("content_block")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not isinstance(block, Mapping)
            ):
                raise AnthropicMessagesAdapterError("content_block_start is invalid")
            blocks[index] = json.loads(json.dumps(block))
            continue
        if event_type == "content_block_delta":
            index = event.get("index")
            delta = event.get("delta")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not isinstance(delta, Mapping)
                or index not in blocks
            ):
                raise AnthropicMessagesAdapterError("content_block_delta is invalid")
            _apply_block_delta(blocks[index], delta)
            continue
        if event_type == "message_delta":
            delta = event.get("delta") if isinstance(event.get("delta"), Mapping) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop_reason = delta["stop_reason"]
            usage = event.get("usage") if isinstance(event.get("usage"), Mapping) else {}
            token_count = _optional_int(usage.get("output_tokens"))
            if token_count is not None:
                output_tokens = token_count
            continue
        raise AnthropicMessagesAdapterError(
            f"unsupported Anthropic SSE event {event_type}"
        )
    if message is None:
        raise AnthropicMessagesAdapterError("Anthropic stream omitted message_start")
    if not saw_message_stop or stop_reason is None:
        raise AnthropicMessagesAdapterError("Anthropic stream omitted terminal event")
    for block in blocks.values():
        leftover = block.pop("_input_json", None)
        if leftover:
            try:
                parsed = json.loads(leftover)
            except json.JSONDecodeError as exc:
                raise AnthropicMessagesAdapterError(
                    "tool_use input JSON is incomplete"
                ) from exc
            if not isinstance(parsed, Mapping):
                raise AnthropicMessagesAdapterError(
                    "tool_use input JSON is not an object"
                )
            block["input"] = dict(parsed)
    ordered = [blocks[index] for index in sorted(blocks)]
    message["content"] = ordered
    if stop_reason is not None:
        message["stop_reason"] = stop_reason
    usage = message.get("usage")
    if not isinstance(usage, dict):
        usage = {}
        message["usage"] = usage
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    return message


def parse_sse_json_events(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8")
    events: list[dict[str, Any]] = []
    current_event = "message"
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal current_event, data_lines
        if not data_lines:
            current_event = "message"
            return
        raw = "\n".join(data_lines)
        data_lines = []
        if raw.strip() == "[DONE]":
            current_event = "message"
            return
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AnthropicMessagesAdapterError(
                "Anthropic SSE chunk is not JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise AnthropicMessagesAdapterError("Anthropic SSE chunk is not an object")
        if "type" not in decoded and current_event != "message":
            decoded["type"] = current_event
        events.append(decoded)
        current_event = "message"

    for line in text.splitlines():
        if line == "":
            flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
    flush()
    return events


def _convert_messages(
    messages: Sequence[Any],
    *,
    replay_content_by_call_id: Mapping[str, Sequence[Any]],
) -> tuple[list[dict[str, str]] | str, list[dict[str, Any]]]:
    system_blocks: list[dict[str, str]] = []
    converted: list[dict[str, Any]] = []
    pending_user: list[dict[str, Any]] = []

    def flush_user() -> None:
        if pending_user:
            converted.append({"role": "user", "content": list(pending_user)})
            pending_user.clear()

    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise AnthropicMessagesAdapterError(f"message {index} is invalid")
        role = message.get("role")
        if role == "system":
            if converted or pending_user:
                raise AnthropicMessagesAdapterError(
                    "system message must precede conversation turns"
                )
            system_blocks.extend(_text_blocks(message.get("content")))
            continue
        if role == "user":
            pending_user.extend(_user_content_blocks(message.get("content")))
            continue
        if role == "tool":
            pending_user.append(_tool_result_block(message))
            continue
        if role == "assistant":
            flush_user()
            converted.append(
                {
                    "role": "assistant",
                    "content": _assistant_content(
                        message,
                        replay_content_by_call_id=replay_content_by_call_id,
                    ),
                }
            )
            continue
        raise AnthropicMessagesAdapterError(f"unsupported chat role {role}")
    flush_user()
    if not converted:
        raise AnthropicMessagesAdapterError("Anthropic request has no messages")
    if converted[0]["role"] != "user":
        raise AnthropicMessagesAdapterError("Anthropic messages must start with user")
    if system_blocks and all(block.get("type") == "text" for block in system_blocks):
        if len(system_blocks) == 1:
            return system_blocks[0]["text"], converted
        return system_blocks, converted
    return system_blocks, converted


def _text_blocks(content: Any) -> list[dict[str, str]]:
    if isinstance(content, str):
        if not content:
            raise AnthropicMessagesAdapterError("empty system text")
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        blocks: list[dict[str, str]] = []
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "text":
                raise AnthropicMessagesAdapterError("system content must be text")
            text = part.get("text")
            if not isinstance(text, str) or not text:
                raise AnthropicMessagesAdapterError("system text is invalid")
            blocks.append({"type": "text", "text": text})
        return blocks
    raise AnthropicMessagesAdapterError("system content is invalid")


def _user_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise AnthropicMessagesAdapterError("user content is invalid")
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, Mapping):
            raise AnthropicMessagesAdapterError("user content part is invalid")
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise AnthropicMessagesAdapterError("user text is invalid")
            blocks.append({"type": "text", "text": text})
            continue
        if part_type == "image_url":
            blocks.append(_image_block(part.get("image_url")))
            continue
        raise AnthropicMessagesAdapterError(f"unsupported user content type {part_type}")
    return blocks


def _image_block(image_url: Any) -> dict[str, Any]:
    url = None
    if isinstance(image_url, str):
        url = image_url
    elif isinstance(image_url, Mapping):
        url = image_url.get("url")
    if not isinstance(url, str) or not url.startswith("data:image/png;base64,"):
        raise AnthropicMessagesAdapterError("only PNG data URLs are supported")
    data = url.split(",", 1)[1]
    if not data:
        raise AnthropicMessagesAdapterError("image data URL is empty")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": data,
        },
    }


def _tool_result_block(message: Mapping[str, Any]) -> dict[str, Any]:
    tool_call_id = message.get("tool_call_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise AnthropicMessagesAdapterError("tool result id is invalid")
    content = message.get("content")
    if isinstance(content, str):
        result_content: str | list[dict[str, Any]] = content
    elif isinstance(content, list):
        result_content = _user_content_blocks(content)
    else:
        raise AnthropicMessagesAdapterError("tool result content is invalid")
    return {
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "content": result_content,
    }


def _assistant_content(
    message: Mapping[str, Any],
    *,
    replay_content_by_call_id: Mapping[str, Sequence[Any]],
) -> list[Any]:
    tool_calls = message.get("tool_calls") or []
    call_id = None
    if isinstance(tool_calls, list) and tool_calls:
        first = tool_calls[0]
        if isinstance(first, Mapping) and isinstance(first.get("id"), str):
            call_id = first["id"]
    if call_id and call_id in replay_content_by_call_id:
        return json.loads(json.dumps(list(replay_content_by_call_id[call_id])))
    blocks: list[Any] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        blocks.extend(_user_content_blocks(content))
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            blocks.append(_chat_tool_call_to_tool_use(tool_call))
    if not blocks:
        raise AnthropicMessagesAdapterError("assistant message has no content")
    return blocks


def _chat_tool_call_to_tool_use(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, Mapping) or tool_call.get("type") != "function":
        raise AnthropicMessagesAdapterError("tool call type is invalid")
    function = tool_call.get("function")
    if not isinstance(function, Mapping):
        raise AnthropicMessagesAdapterError("tool call function is invalid")
    name = function.get("name")
    arguments = function.get("arguments")
    call_id = tool_call.get("id")
    if not isinstance(name, str) or not name:
        raise AnthropicMessagesAdapterError("tool call name is invalid")
    if not isinstance(call_id, str) or not call_id:
        raise AnthropicMessagesAdapterError("tool call id is invalid")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise AnthropicMessagesAdapterError(
                "tool call arguments are not JSON"
            ) from exc
    elif isinstance(arguments, Mapping):
        parsed = dict(arguments)
    else:
        raise AnthropicMessagesAdapterError("tool call arguments are invalid")
    if not isinstance(parsed, Mapping):
        raise AnthropicMessagesAdapterError("tool call arguments must be an object")
    return {
        "type": "tool_use",
        "id": call_id,
        "name": name,
        "input": dict(parsed),
    }


def _tool_use_to_chat(block: Mapping[str, Any]) -> dict[str, Any]:
    call_id = block.get("id")
    name = block.get("name")
    tool_input = block.get("input")
    if not isinstance(call_id, str) or not call_id:
        raise AnthropicMessagesAdapterError("tool_use id is invalid")
    if not isinstance(name, str) or not name:
        raise AnthropicMessagesAdapterError("tool_use name is invalid")
    if not isinstance(tool_input, Mapping):
        raise AnthropicMessagesAdapterError("tool_use input is invalid")
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                dict(tool_input),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


def _convert_tool(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, Mapping) or tool.get("type") != "function":
        raise AnthropicMessagesAdapterError("tool definition type is invalid")
    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise AnthropicMessagesAdapterError("tool function is invalid")
    name = function.get("name")
    description = function.get("description")
    parameters = function.get("parameters")
    if not isinstance(name, str) or not name:
        raise AnthropicMessagesAdapterError("tool name is invalid")
    if not isinstance(description, str):
        raise AnthropicMessagesAdapterError("tool description is invalid")
    if not isinstance(parameters, Mapping):
        raise AnthropicMessagesAdapterError("tool parameters are invalid")
    return {
        "name": name,
        "description": description,
        "input_schema": json.loads(json.dumps(parameters)),
    }


def _apply_block_delta(block: dict[str, Any], delta: Mapping[str, Any]) -> None:
    delta_type = delta.get("type")
    if delta_type == "text_delta":
        text = delta.get("text")
        if not isinstance(text, str):
            raise AnthropicMessagesAdapterError("text_delta is invalid")
        block["text"] = str(block.get("text") or "") + text
        return
    if delta_type == "thinking_delta":
        thinking = delta.get("thinking")
        if not isinstance(thinking, str):
            raise AnthropicMessagesAdapterError("thinking_delta is invalid")
        block["thinking"] = str(block.get("thinking") or "") + thinking
        return
    if delta_type == "signature_delta":
        signature = delta.get("signature")
        if not isinstance(signature, str):
            raise AnthropicMessagesAdapterError("signature_delta is invalid")
        block["signature"] = str(block.get("signature") or "") + signature
        return
    if delta_type == "input_json_delta":
        partial = delta.get("partial_json")
        if not isinstance(partial, str):
            raise AnthropicMessagesAdapterError("input_json_delta is invalid")
        block["_input_json"] = str(block.get("_input_json") or "") + partial
        if block["_input_json"]:
            try:
                parsed = json.loads(block["_input_json"])
            except json.JSONDecodeError:
                return
            if isinstance(parsed, Mapping):
                block["input"] = dict(parsed)
                block.pop("_input_json", None)
        return
    raise AnthropicMessagesAdapterError(f"unsupported content delta {delta_type}")


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
