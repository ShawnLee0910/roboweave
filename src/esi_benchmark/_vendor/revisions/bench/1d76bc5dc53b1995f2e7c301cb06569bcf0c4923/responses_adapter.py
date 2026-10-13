"""Strict Chat-to-Responses transport adapter for ESI-Bench.

The benchmark keeps its canonical, replayable Chat Completions trajectory.
Only the provider wire representation changes.  The adapter deliberately
supports the narrow message and tool surface emitted by
``Agent.EpisodeConversation.build_native_request`` and fails closed on
provider-specific extensions.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
import json
from typing import Any


ADAPTER_VERSION = "esi.responses-transport-adapter/v4-normalized-opaque-replay"
RESPONSES_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)
RESPONSES_REASONING_CONTEXTS = frozenset(
    {"auto", "all_turns", "current_turn"}
)
RESPONSES_REASONING_MODES = frozenset({"standard", "pro"})
RESPONSES_REASONING_SUMMARIES = frozenset({"auto", "concise", "detailed"})
RESPONSES_TEXT_VERBOSITIES = frozenset({"low", "medium", "high"})


class ResponsesAdapterError(ValueError):
    """The canonical trajectory cannot be represented without ambiguity."""


def _clone(value: Any) -> Any:
    return copy.deepcopy(value)


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResponsesAdapterError(f"{field} must be a nonempty string")
    return value


def _strict_json_object(value: str, *, field: str) -> dict[str, Any]:
    """Parse one function-arguments object without duplicate or NaN keys."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ResponsesAdapterError(f"{field} contains a duplicate key")
            result[key] = item
        return result

    def reject_constant(constant: str) -> None:
        raise ResponsesAdapterError(
            f"{field} contains a non-JSON numeric constant: {constant}"
        )

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ResponsesAdapterError(f"{field} is not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise ResponsesAdapterError(f"{field} must be a JSON object")
    return parsed


def function_arguments_equivalent(actual: str, expected: str) -> bool:
    """Compare function arguments by strict JSON value, not wire whitespace."""

    actual_object = _strict_json_object(actual, field="response function arguments")
    expected_object = _strict_json_object(
        expected,
        field="canonical function arguments",
    )
    return (
        json.dumps(
            actual_object,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        == json.dumps(
            expected_object,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _convert_user_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list) or not content:
        raise ResponsesAdapterError("user content must be nonempty text or a list")
    converted: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, Mapping):
            raise ResponsesAdapterError("user content item must be an object")
        kind = item.get("type")
        if kind == "text":
            if set(item) != {"type", "text"}:
                raise ResponsesAdapterError("Chat text content fields changed")
            text = _require_nonempty_string(item.get("text"), field="text")
            converted.append({"type": "input_text", "text": text})
            continue
        if kind == "image_url":
            if set(item) != {"type", "image_url"}:
                raise ResponsesAdapterError("Chat image content fields changed")
            image = item.get("image_url")
            if not isinstance(image, Mapping) or not set(image).issubset(
                {"url", "detail"}
            ):
                raise ResponsesAdapterError("Chat image_url fields changed")
            url = _require_nonempty_string(image.get("url"), field="image URL")
            output: dict[str, Any] = {
                "type": "input_image",
                "image_url": url,
            }
            if "detail" in image:
                detail = image.get("detail")
                if detail not in {"auto", "low", "high", "original"}:
                    raise ResponsesAdapterError("image detail is invalid")
                output["detail"] = detail
            converted.append(output)
            continue
        raise ResponsesAdapterError(f"unsupported Chat content type: {kind!r}")
    return converted


def _convert_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list) or not tools:
        raise ResponsesAdapterError("at least one function tool is required")
    converted: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        if (
            not isinstance(tool, Mapping)
            or set(tool) != {"type", "function"}
            or tool.get("type") != "function"
            or not isinstance(tool.get("function"), Mapping)
        ):
            raise ResponsesAdapterError("Chat function tool shape changed")
        function = tool["function"]
        allowed = {"name", "description", "parameters", "strict"}
        if not set(function).issubset(allowed) or not {
            "name",
            "parameters",
        }.issubset(function):
            raise ResponsesAdapterError("Chat function definition fields changed")
        name = _require_nonempty_string(function.get("name"), field="tool name")
        if name in names:
            raise ResponsesAdapterError("function tool names must be unique")
        names.add(name)
        if not isinstance(function.get("parameters"), Mapping):
            raise ResponsesAdapterError("tool parameters must be an object")
        flattened = {"type": "function", **_clone(dict(function))}
        converted.append(flattened)
    return converted


def _convert_tool_choice(choice: Any, *, tool_names: set[str]) -> dict[str, str]:
    if (
        not isinstance(choice, Mapping)
        or set(choice) != {"type", "function"}
        or choice.get("type") != "function"
        or not isinstance(choice.get("function"), Mapping)
        or set(choice["function"]) != {"name"}
    ):
        raise ResponsesAdapterError("a forced Chat function tool_choice is required")
    name = _require_nonempty_string(
        choice["function"].get("name"), field="tool_choice name"
    )
    if name not in tool_names:
        raise ResponsesAdapterError("tool_choice names an unavailable function")
    return {"type": "function", "name": name}


def chat_request_to_responses(
    request: Mapping[str, Any],
    *,
    max_output_tokens: int | None = None,
    replay_items_by_call_id: Mapping[str, Sequence[Mapping[str, Any]]]
    | None = None,
    accepted_canonical_arguments_by_call_id: Mapping[str, str] | None = None,
    generation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one canonical ESI Chat request into a stateless Responses request.

    The single leading system message becomes the nonempty top-level
    ``instructions`` field.  It is not duplicated in ``input``.  Completed
    historical Chat tool transactions become paired ``function_call`` and
    ``function_call_output`` input items with the original call IDs.  When a
    raw Responses output bundle is supplied for a historical call, every
    opaque item in that bundle is replayed verbatim instead of reconstructing
    the function call.  This preserves encrypted reasoning without exposing it
    to the canonical Chat trajectory.  A trusted canonical-arguments mapping
    permits that verbatim replay after an accepted Action20 normalization.
    """

    if not isinstance(request, Mapping):
        raise ResponsesAdapterError("Chat request must be an object")
    expected = {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    }
    if set(request) != expected:
        extra = sorted(set(request) - expected)
        missing = sorted(expected - set(request))
        raise ResponsesAdapterError(
            f"canonical Chat request fields changed; extra={extra}, missing={missing}"
        )
    model = _require_nonempty_string(request.get("model"), field="model")
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ResponsesAdapterError("Chat request requires system and user messages")
    first = messages[0]
    if (
        not isinstance(first, Mapping)
        or set(first) != {"role", "content"}
        or first.get("role") != "system"
    ):
        raise ResponsesAdapterError("the first Chat message must be the only system message")
    instructions = _require_nonempty_string(
        first.get("content"), field="instructions"
    )

    if replay_items_by_call_id is None:
        replay_items_by_call_id = {}
    if not isinstance(replay_items_by_call_id, Mapping):
        raise ResponsesAdapterError("replay_items_by_call_id must be an object")
    for replay_call_id in replay_items_by_call_id:
        _require_nonempty_string(replay_call_id, field="replay call ID")
    if accepted_canonical_arguments_by_call_id is None:
        accepted_canonical_arguments_by_call_id = {}
    if not isinstance(accepted_canonical_arguments_by_call_id, Mapping):
        raise ResponsesAdapterError(
            "accepted_canonical_arguments_by_call_id must be an object"
        )
    for call_id, arguments in accepted_canonical_arguments_by_call_id.items():
        _require_nonempty_string(call_id, field="accepted replay call ID")
        _require_nonempty_string(arguments, field="accepted canonical arguments")

    input_items: list[dict[str, Any]] = []
    open_calls: set[str] = set()
    resolved_calls: set[str] = set()
    consumed_replay_calls: set[str] = set()
    consumed_accepted_calls: set[str] = set()
    for message in messages[1:]:
        if not isinstance(message, Mapping):
            raise ResponsesAdapterError("Chat message must be an object")
        role = message.get("role")
        if role in {"system", "developer"}:
            raise ResponsesAdapterError("only one leading system message is allowed")
        if role == "user":
            if set(message) != {"role", "content"}:
                raise ResponsesAdapterError("user message fields changed")
            input_items.append(
                {
                    "role": "user",
                    "content": _convert_user_content(message.get("content")),
                }
            )
            continue
        if role == "assistant":
            allowed = {
                "role",
                "content",
                "tool_calls",
                "reasoning_content",
                "reasoning",
            }
            if not set(message).issubset(allowed):
                raise ResponsesAdapterError("assistant message fields changed")
            if message.get("content") not in {None, ""}:
                raise ResponsesAdapterError("text assistant history is outside Action20")
            if message.get("reasoning_content") not in {None, ""} or message.get(
                "reasoning"
            ) not in {None, ""}:
                raise ResponsesAdapterError(
                    "plaintext reasoning cannot be losslessly converted to a Responses item"
                )
            calls = message.get("tool_calls")
            if not isinstance(calls, list) or len(calls) != 1:
                raise ResponsesAdapterError(
                    "exactly one historical Chat function call is required"
                )
            call = calls[0]
            if (
                not isinstance(call, Mapping)
                or not set(call).issubset({"id", "type", "function", "index"})
                or call.get("type") != "function"
                or not isinstance(call.get("function"), Mapping)
            ):
                raise ResponsesAdapterError("historical Chat tool call fields changed")
            call_id = _require_nonempty_string(call.get("id"), field="tool call ID")
            function = call["function"]
            if set(function) != {"name", "arguments"}:
                raise ResponsesAdapterError("historical Chat function fields changed")
            name = _require_nonempty_string(
                function.get("name"), field="historical function name"
            )
            arguments = _require_nonempty_string(
                function.get("arguments"), field="historical function arguments"
            )
            if call_id in open_calls or call_id in resolved_calls:
                raise ResponsesAdapterError("historical tool call ID was reused")
            open_calls.add(call_id)
            if call_id in replay_items_by_call_id:
                expected_replay_arguments: str | None = arguments
                if call_id in accepted_canonical_arguments_by_call_id:
                    accepted_arguments = accepted_canonical_arguments_by_call_id[
                        call_id
                    ]
                    if not function_arguments_equivalent(
                        arguments,
                        accepted_arguments,
                    ):
                        raise ResponsesAdapterError(
                            "accepted replay arguments do not match canonical history"
                        )
                    expected_replay_arguments = None
                    consumed_accepted_calls.add(call_id)
                replay_bundle = _validate_replay_bundle(
                    replay_items_by_call_id[call_id],
                    expected_call_id=call_id,
                    expected_name=name,
                    expected_arguments=expected_replay_arguments,
                )
                input_items.extend(replay_bundle)
                consumed_replay_calls.add(call_id)
            else:
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                    }
                )
            continue
        if role == "tool":
            if set(message) != {"role", "tool_call_id", "content"}:
                raise ResponsesAdapterError("tool result message fields changed")
            call_id = _require_nonempty_string(
                message.get("tool_call_id"), field="tool result call ID"
            )
            if call_id not in open_calls:
                raise ResponsesAdapterError("tool result does not match an open call")
            output = message.get("content")
            if not isinstance(output, str):
                raise ResponsesAdapterError("tool result content must be a string")
            open_calls.remove(call_id)
            resolved_calls.add(call_id)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            )
            continue
        raise ResponsesAdapterError(f"unsupported Chat role: {role!r}")
    if open_calls:
        raise ResponsesAdapterError("historical tool call is missing its tool result")
    unused_replay_calls = set(replay_items_by_call_id) - consumed_replay_calls
    if unused_replay_calls:
        raise ResponsesAdapterError(
            "replay bundle does not match canonical Chat history: "
            + ", ".join(sorted(unused_replay_calls))
        )
    unused_accepted_calls = (
        set(accepted_canonical_arguments_by_call_id) - consumed_accepted_calls
    )
    if unused_accepted_calls:
        raise ResponsesAdapterError(
            "accepted replay arguments do not match canonical Chat history: "
            + ", ".join(sorted(unused_accepted_calls))
        )

    tools = _convert_tools(request.get("tools"))
    tool_names = {str(tool["name"]) for tool in tools}
    if request.get("parallel_tool_calls") is not False:
        raise ResponsesAdapterError("parallel tool calls must remain disabled")
    result: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "tools": tools,
        "tool_choice": _convert_tool_choice(
            request.get("tool_choice"), tool_names=tool_names
        ),
        "parallel_tool_calls": False,
        "store": False,
    }
    if generation is not None:
        result.update(_convert_generation(generation))
    if max_output_tokens is not None:
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise ResponsesAdapterError("max_output_tokens must be a positive integer")
        result["max_output_tokens"] = max_output_tokens
    return result


def _convert_generation(generation: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(generation, Mapping) or set(generation) != {
        "reasoning_effort",
        "reasoning_context",
        "reasoning_mode",
        "reasoning_summary",
        "text_verbosity",
    }:
        raise ResponsesAdapterError("Responses generation fields changed")
    effort = generation.get("reasoning_effort")
    context = generation.get("reasoning_context")
    mode = generation.get("reasoning_mode")
    summary = generation.get("reasoning_summary")
    verbosity = generation.get("text_verbosity")
    if effort not in RESPONSES_REASONING_EFFORTS:
        raise ResponsesAdapterError("Responses reasoning effort is invalid")
    if context not in RESPONSES_REASONING_CONTEXTS:
        raise ResponsesAdapterError("Responses reasoning context is invalid")
    if mode not in RESPONSES_REASONING_MODES:
        raise ResponsesAdapterError("Responses reasoning mode is invalid")
    if summary is not None and summary not in RESPONSES_REASONING_SUMMARIES:
        raise ResponsesAdapterError("Responses reasoning summary is invalid")
    if verbosity not in RESPONSES_TEXT_VERBOSITIES:
        raise ResponsesAdapterError("Responses text verbosity is invalid")
    reasoning = {"effort": effort, "context": context}
    if summary is not None:
        reasoning["summary"] = summary
    if mode == "pro":
        reasoning["mode"] = "pro"
    return {
        "reasoning": reasoning,
        "text": {"verbosity": verbosity},
    }


def _validate_replay_bundle(
    items: Sequence[Mapping[str, Any]],
    *,
    expected_call_id: str | None = None,
    expected_name: str | None = None,
    expected_arguments: str | None = None,
) -> list[dict[str, Any]]:
    """Validate and clone one complete Responses output bundle.

    Reasoning payload fields are deliberately opaque: this function only
    checks each item's type and the function-call linkage needed to prevent a
    bundle from being attached to the wrong canonical turn.
    """

    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise ResponsesAdapterError("Responses replay bundle must be a list")
    if not items:
        raise ResponsesAdapterError("Responses replay bundle must not be empty")
    calls: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ResponsesAdapterError("Responses replay item must be an object")
        kind = item.get("type")
        if kind == "reasoning":
            continue
        if kind == "function_call":
            calls.append(item)
            continue
        raise ResponsesAdapterError(
            f"unexpected Responses output item in replay bundle: {kind!r}"
        )
    if len(calls) != 1:
        raise ResponsesAdapterError(
            "exactly one Responses function call is required in a replay bundle"
        )
    call = calls[0]
    call_id = _require_nonempty_string(call.get("call_id"), field="response call ID")
    name = _require_nonempty_string(call.get("name"), field="response function name")
    arguments = _require_nonempty_string(
        call.get("arguments"), field="response function arguments"
    )
    if expected_call_id is not None and call_id != expected_call_id:
        raise ResponsesAdapterError("replay function call ID does not match Chat history")
    if expected_name is not None and name != expected_name:
        raise ResponsesAdapterError("replay function name does not match Chat history")
    if expected_arguments is not None and not function_arguments_equivalent(
        arguments,
        expected_arguments,
    ):
        raise ResponsesAdapterError(
            "replay function arguments do not match Chat history"
        )
    return _clone(list(items))


def extract_responses_replay_items(
    response: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return a private, deep-cloned replay bundle from one Responses result.

    The encrypted reasoning payload is neither parsed nor converted.  Callers
    should persist this bundle only in a private artifact and associate it with
    the canonical assistant tool call's ``call_id``.
    """

    if not isinstance(response, Mapping):
        raise ResponsesAdapterError("Responses result must be an object")
    if response.get("object") not in {None, "response"}:
        raise ResponsesAdapterError("provider result is not a Responses object")
    if response.get("status") not in {None, "completed"}:
        raise ResponsesAdapterError("Responses request did not complete")
    output = response.get("output")
    if not isinstance(output, list):
        raise ResponsesAdapterError("Responses output must be a list")
    return _validate_replay_bundle(output)


def responses_to_chat_completion(response: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one Responses function call for the existing Agent parser.

    Raw provider output must be preserved separately by the caller.  Reasoning
    items are intentionally not converted into plaintext or assistant text.
    """

    output = extract_responses_replay_items(response)
    call = next(item for item in output if item.get("type") == "function_call")
    call_id = _require_nonempty_string(call.get("call_id"), field="response call ID")
    name = _require_nonempty_string(call.get("name"), field="response function name")
    arguments = _require_nonempty_string(
        call.get("arguments"), field="response function arguments"
    )
    model = response.get("model")
    if model is not None and not isinstance(model, str):
        raise ResponsesAdapterError("response model is invalid")
    normalized: dict[str, Any] = {
        "id": response.get("id"),
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        if all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (input_tokens, output_tokens, total_tokens)
        ):
            normalized["usage"] = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
    return normalized


__all__ = [
    "ADAPTER_VERSION",
    "RESPONSES_REASONING_CONTEXTS",
    "RESPONSES_REASONING_EFFORTS",
    "RESPONSES_REASONING_MODES",
    "RESPONSES_REASONING_SUMMARIES",
    "RESPONSES_TEXT_VERBOSITIES",
    "ResponsesAdapterError",
    "chat_request_to_responses",
    "extract_responses_replay_items",
    "function_arguments_equivalent",
    "responses_to_chat_completion",
]
