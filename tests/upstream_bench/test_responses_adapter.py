from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import sys

from PIL import Image
import pytest


BENCH_ROOT = (Path(__file__).resolve().parents[2] / "src/roboweave/_vendor/bench")
LAB_ROOT = BENCH_ROOT / "lab"
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

import Agent  # noqa: E402
import model_client  # noqa: E402
from responses_adapter import (  # noqa: E402
    ResponsesAdapterError,
    chat_request_to_responses,
    extract_responses_replay_items,
    responses_to_chat_completion,
)


def _request(*, messages: list[dict] | None = None) -> dict:
    schema = json.loads((BENCH_ROOT / "action20_tool_schema.json").read_text())
    return {
        "model": "gpt-5.6-sol",
        "messages": messages
        or [
            {"role": "system", "content": "ESI system prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Task: wait once"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo="
                        },
                    },
                ],
            },
        ],
        "tools": [schema],
        "tool_choice": {
            "type": "function",
            "function": {"name": "robot_action"},
        },
        "parallel_tool_calls": False,
    }


def _observation(tmp_path: Path, observation_id: str, color: tuple[int, int, int]):
    path = tmp_path / f"{observation_id}.png"
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(stream, format="PNG")
    raw = stream.getvalue()
    path.write_bytes(raw)
    return Agent.PublicObservation(
        observation_id=observation_id,
        images=(
            Agent.PublicImage(
                camera="head",
                path=path,
            ),
        ),
    )


def test_initial_request_moves_system_prompt_and_flattens_wire_shapes() -> None:
    converted = chat_request_to_responses(_request(), max_output_tokens=512)
    assert converted["instructions"] == "ESI system prompt"
    assert all(item.get("role") != "system" for item in converted["input"])
    assert converted["input"][0]["content"] == [
        {"type": "input_text", "text": "Task: wait once"},
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,iVBORw0KGgo=",
        },
    ]
    assert converted["tools"][0]["type"] == "function"
    assert "function" not in converted["tools"][0]
    assert converted["tools"][0]["name"] == "robot_action"
    assert converted["tool_choice"] == {
        "type": "function",
        "name": "robot_action",
    }
    assert converted["parallel_tool_calls"] is False
    assert converted["store"] is False
    assert "service_tier" not in converted
    max_alias = _request()
    max_alias["model"] = "gpt-5.6-sol-max"
    assert "service_tier" not in chat_request_to_responses(max_alias)
    assert converted["max_output_tokens"] == 512


def test_responses_generation_parameters_are_mapped_without_chat_sampling() -> None:
    converted = chat_request_to_responses(
        _request(),
        generation={
            "reasoning_effort": "high",
            "reasoning_context": "all_turns",
            "reasoning_mode": "standard",
            "reasoning_summary": "detailed",
            "text_verbosity": "low",
        },
    )
    assert converted["reasoning"] == {
        "effort": "high",
        "context": "all_turns",
        "summary": "detailed",
    }
    assert converted["text"] == {"verbosity": "low"}
    assert "temperature" not in converted
    assert "top_p" not in converted

    pro = chat_request_to_responses(
        _request(),
        generation={
            "reasoning_effort": "max",
            "reasoning_context": "current_turn",
            "reasoning_mode": "pro",
            "reasoning_summary": "concise",
            "text_verbosity": "medium",
        },
    )
    assert pro["reasoning"]["mode"] == "pro"


@pytest.mark.parametrize(
    "field,value",
    [
        ("reasoning_effort", "ultra"),
        ("reasoning_context", "forever"),
        ("reasoning_mode", "turbo"),
        ("reasoning_summary", "verbose"),
        ("text_verbosity", "verbose"),
    ],
)
def test_responses_generation_parameters_fail_closed(field: str, value: str) -> None:
    generation = {
        "reasoning_effort": "medium",
        "reasoning_context": "all_turns",
        "reasoning_mode": "standard",
        "reasoning_summary": "detailed",
        "text_verbosity": "low",
    }
    generation[field] = value
    with pytest.raises(ResponsesAdapterError, match="invalid"):
        chat_request_to_responses(_request(), generation=generation)


def test_completed_chat_tool_transaction_preserves_call_id() -> None:
    messages = [
        {"role": "system", "content": "ESI system prompt"},
        {"role": "user", "content": "Task: turn twice"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_first",
                    "type": "function",
                    "function": {
                        "name": "robot_action",
                        "arguments": '{"action":"turn_left"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_first",
            "content": '{"status":"completed"}',
        },
        {"role": "user", "content": "Current onboard observation."},
    ]
    converted = chat_request_to_responses(_request(messages=messages))
    assert converted["input"][1] == {
        "type": "function_call",
        "call_id": "call_first",
        "name": "robot_action",
        "arguments": '{"action":"turn_left"}',
    }
    assert converted["input"][2] == {
        "type": "function_call_output",
        "call_id": "call_first",
        "output": '{"status":"completed"}',
    }


def _opaque_response_bundle() -> dict:
    return {
        "id": "resp_opaque",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.6-sol",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_opaque",
                "encrypted_content": "gAAAAABopaque-provider-ciphertext",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "Private detailed reasoning summary.",
                    }
                ],
                "status": "completed",
            },
            {
                "type": "function_call",
                "id": "fc_opaque",
                "call_id": "call_first",
                "name": "robot_action",
                "arguments": '{"action":"turn_left"}',
                "status": "completed",
            },
        ],
    }


def _completed_history_request() -> dict:
    return _request(
        messages=[
            {"role": "system", "content": "ESI system prompt"},
            {"role": "user", "content": "Task: turn twice"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_first",
                        "type": "function",
                        "function": {
                            "name": "robot_action",
                            "arguments": '{"action":"turn_left"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_first",
                "content": '{"status":"completed"}',
            },
            {"role": "user", "content": "Current onboard observation."},
        ]
    )


def test_opaque_replay_bundle_is_deep_cloned_and_replayed_verbatim() -> None:
    raw = _opaque_response_bundle()
    original_output = copy.deepcopy(raw["output"])
    replay = extract_responses_replay_items(raw)
    assert replay == original_output
    assert replay is not raw["output"]
    assert replay[0] is not raw["output"][0]

    converted = chat_request_to_responses(
        _completed_history_request(),
        replay_items_by_call_id={"call_first": replay},
    )
    assert converted["input"][1:3] == original_output
    assert converted["input"][3] == {
        "type": "function_call_output",
        "call_id": "call_first",
        "output": '{"status":"completed"}',
    }
    assert "encrypted_content" not in json.dumps(
        _completed_history_request(), ensure_ascii=False
    )

    replay[0]["encrypted_content"] = "mutated"
    assert raw["output"] == original_output
    assert converted["input"][1:3] == original_output


def test_opaque_replay_accepts_equivalent_json_argument_serialization() -> None:
    replay = extract_responses_replay_items(_opaque_response_bundle())
    function_call = next(
        item for item in replay if item.get("type") == "function_call"
    )
    function_call["arguments"] = '{ "action" : "turn_left" }'

    converted = chat_request_to_responses(
        _completed_history_request(),
        replay_items_by_call_id={"call_first": replay},
    )

    assert function_call in converted["input"]


def test_opaque_replay_accepts_prevalidated_canonical_normalization() -> None:
    request = _completed_history_request()
    request["messages"][2]["tool_calls"][0]["function"][
        "arguments"
    ] = '{"action":"pick"}'
    replay = extract_responses_replay_items(_opaque_response_bundle())
    replay[1]["arguments"] = '{"action":"pick","target":7}'

    converted = chat_request_to_responses(
        request,
        replay_items_by_call_id={"call_first": replay},
        accepted_canonical_arguments_by_call_id={
            "call_first": '{"action":"pick"}'
        },
    )

    assert converted["input"][1:3] == replay

    with pytest.raises(ResponsesAdapterError, match="canonical history"):
        chat_request_to_responses(
            request,
            replay_items_by_call_id={"call_first": replay},
            accepted_canonical_arguments_by_call_id={
                "call_first": '{"action":"wait"}'
            },
        )


@pytest.mark.parametrize(
    "field,value,pattern",
    [
        ("call_id", "call_wrong", "call ID"),
        ("name", "wrong_tool", "function name"),
        ("arguments", '{"action":"wait"}', "function arguments"),
    ],
)
def test_opaque_replay_rejects_canonical_history_mismatch(
    field: str, value: str, pattern: str
) -> None:
    replay = extract_responses_replay_items(_opaque_response_bundle())
    function_call = next(
        item for item in replay if item.get("type") == "function_call"
    )
    function_call[field] = value
    with pytest.raises(ResponsesAdapterError, match=pattern):
        chat_request_to_responses(
            _completed_history_request(),
            replay_items_by_call_id={"call_first": replay},
        )


def test_opaque_replay_rejects_unused_or_non_action_output() -> None:
    replay = extract_responses_replay_items(_opaque_response_bundle())
    with pytest.raises(ResponsesAdapterError, match="does not match"):
        chat_request_to_responses(
            _completed_history_request(),
            replay_items_by_call_id={"call_first": replay, "call_unused": replay},
        )
    raw = _opaque_response_bundle()
    raw["output"].insert(1, {"type": "message", "content": []})
    with pytest.raises(ResponsesAdapterError, match="unexpected Responses output"):
        extract_responses_replay_items(raw)


def test_formal_provider_applies_parameters_and_persists_private_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="gpt-5.6-sol",
        base_url="http://10.16.123.234:51654/v1",
        timeout_seconds=30,
        api_key_file=None,
        maximum_output_tokens=2048,
        provider_api="responses",
        require_api_key=True,
        reasoning_effort="medium",
        reasoning_context="all_turns",
        reasoning_mode="standard",
        reasoning_summary="detailed",
        text_verbosity="low",
    )
    first_raw = _opaque_response_bundle()
    second_raw = {
        "id": "resp_second",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.6-sol",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_second",
                "name": "robot_action",
                "arguments": '{"action":"wait"}',
            }
        ],
    }
    responses = [first_raw, second_raw]
    observed: list[dict] = []

    def fake_post(request):
        observed.append(copy.deepcopy(request))
        return responses.pop(0)

    monkeypatch.setattr(provider, "_post", fake_post)
    output = tmp_path / "run"
    first = provider.complete(
        _request(),
        call_dir=output / "provider_calls" / "call_0001",
        seed=19506,
    )
    assert observed[0]["reasoning"] == {
        "effort": "medium",
        "context": "all_turns",
        "summary": "detailed",
    }
    assert observed[0]["text"] == {"verbosity": "low"}
    assert observed[0]["store"] is False
    assert "temperature" not in observed[0]
    private_raw = (
        output
        / "provider_calls"
        / "call_0001"
        / "private"
        / "responses_response.json"
    )
    assert private_raw.stat().st_mode & 0o077 == 0
    assert "gAAAAABopaque" in private_raw.read_text(encoding="utf-8")
    assert "Private detailed reasoning summary." in private_raw.read_text(
        encoding="utf-8"
    )
    assert "gAAAAABopaque" not in (
        output / "provider_calls" / "call_0001" / "response.json"
    ).read_text(encoding="utf-8")
    assert "Private detailed reasoning summary." not in (
        output / "provider_calls" / "call_0001" / "response.json"
    ).read_text(encoding="utf-8")
    assert "gAAAAABopaque" not in (
        output / "provider_calls" / "call_0001" / "request_manifest.json"
    ).read_text(encoding="utf-8")
    assert "iVBORw0KGgo=" not in (
        output / "provider_calls" / "call_0001" / "request_manifest.json"
    ).read_text(encoding="utf-8")

    provider.accept_response(first, output_dir=output)
    replay_path = model_client._responses_replay_path(output, "call_first")
    assert replay_path.stat().st_mode & 0o077 == 0

    provider.complete(
        _completed_history_request(),
        call_dir=output / "provider_calls" / "call_0002",
        seed=19507,
    )
    replay_bundle = extract_responses_replay_items(first_raw)
    assert any(
        observed[1]["input"][index : index + len(replay_bundle)]
        == replay_bundle
        for index in range(len(observed[1]["input"]) - len(replay_bundle) + 1)
    )


def test_provider_replays_reasoning_after_safe_agent_normalization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="gpt-5.6-sol",
        base_url="http://10.16.123.234:51654/v1",
        timeout_seconds=30,
        api_key_file=None,
        maximum_output_tokens=2048,
        provider_api="responses",
        require_api_key=True,
        reasoning_effort="medium",
        reasoning_context="all_turns",
        reasoning_mode="standard",
        reasoning_summary="detailed",
        text_verbosity="low",
    )
    raw = _opaque_response_bundle()
    raw["output"][1].update(
        {
            "call_id": "call_eager_pick",
            "arguments": '{"action":"pick","target":7}',
        }
    )
    monkeypatch.setattr(provider, "_post", lambda _: raw)
    output = tmp_path / "run"

    normalized = provider.complete(
        _request(),
        call_dir=output / "provider_calls" / "call_0001",
        seed=19506,
    )
    provider.accept_response(
        normalized,
        output_dir=output,
        canonical_function_arguments={"action": "pick"},
    )

    replay_path = model_client._responses_replay_path(
        output,
        "call_eager_pick",
    )
    record = json.loads(replay_path.read_text(encoding="utf-8"))
    assert record["replay_mode"] == "opaque_replay_accepted_normalization"
    assert record["output_items"] == raw["output"]
    replay, accepted = provider._load_responses_replay_items(
        ["call_eager_pick"],
        output_dir=output,
        canonical_arguments_by_call_id={
            "call_eager_pick": '{"action":"pick"}'
        },
    )
    assert replay == {"call_eager_pick": raw["output"]}
    assert accepted == {"call_eager_pick": '{"action":"pick"}'}

    with pytest.raises(model_client.ModelClientError, match="canonical history"):
        provider._load_responses_replay_items(
            ["call_eager_pick"],
            output_dir=output,
            canonical_arguments_by_call_id={
                "call_eager_pick": '{"action":"wait"}'
            },
        )


def test_responses_native_tool_preflight_tolerates_safe_eager_target_normalization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="gpt-5.6-sol",
        base_url="http://10.16.123.234:51654/v1",
        timeout_seconds=30,
        api_key_file=None,
        maximum_output_tokens=2048,
        provider_api="responses",
        require_api_key=True,
        reasoning_effort="medium",
        reasoning_context="all_turns",
        reasoning_mode="standard",
        reasoning_summary="detailed",
        text_verbosity="low",
    )
    arguments = [
        {"action": "wait"},
        {"action": "turn_left"},
        {"action": "turn_right"},
        {"action": "pick", "target": 7},
        {
            "action": "pick",
            "target": 3,
            "target_description": "red cup on counter",
        },
    ]
    requests: list[dict] = []

    def fake_post(request):
        index = len(requests)
        requests.append(copy.deepcopy(request))
        return {
            "id": f"resp_preflight_{index}",
            "object": "response",
            "status": "completed",
            "model": "gpt-5.6-sol",
            "output": [
                {
                    "type": "function_call",
                    "call_id": f"call_preflight_{index}",
                    "name": "robot_action",
                    "arguments": json.dumps(
                        arguments[index],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            ],
        }

    monkeypatch.setattr(provider, "_post", fake_post)
    prompt = Agent.load_common_system_prompt_revision(
        Agent.ACTION20_CORE_INSTRUCTIONS_PATH,
        physical_history_max=5,
    )

    report = provider.capability_preflight(
        prompt_revision=prompt,
        physical_history_max=5,
        artifact=tmp_path / "provider_preflight.json",
    )

    assert report["status"] == "PASS"
    assert report["model_call_count"] == 5
    assert len(requests) == 5
    assert all(
        item.get("type") != "reasoning"
        for request in requests
        for item in request["input"]
        if isinstance(item, dict)
    )


def test_responses_native_tool_preflight_retries_transient_provider_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="gpt-5.6-sol",
        base_url="http://10.16.123.234:51654/v1",
        timeout_seconds=30,
        api_key_file=None,
        maximum_output_tokens=2048,
        provider_api="responses",
        require_api_key=True,
        reasoning_effort="medium",
        reasoning_context="all_turns",
        reasoning_mode="standard",
        reasoning_summary="detailed",
        text_verbosity="low",
    )
    arguments = [
        {"action": "wait"},
        {"action": "turn_left"},
        {"action": "turn_right"},
        {"action": "pick", "target": 7},
        {
            "action": "pick",
            "target": 3,
            "target_description": "red cup on counter",
        },
    ]
    successful_requests: list[dict] = []
    attempts = 0

    def flaky_post(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise model_client.ProviderRequestError(
                "provider request timed out",
                failure_type="provider_timeout",
                retryable=True,
            )
        index = len(successful_requests)
        successful_requests.append(copy.deepcopy(request))
        return {
            "id": f"resp_retry_{index}",
            "object": "response",
            "status": "completed",
            "model": "gpt-5.6-sol",
            "output": [
                {
                    "type": "function_call",
                    "call_id": f"call_retry_{index}",
                    "name": "robot_action",
                    "arguments": json.dumps(
                        arguments[index],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            ],
        }

    monkeypatch.setattr(provider, "_post", flaky_post)
    prompt = Agent.load_common_system_prompt_revision(
        Agent.ACTION20_CORE_INSTRUCTIONS_PATH,
        physical_history_max=5,
    )

    report = provider.capability_preflight(
        prompt_revision=prompt,
        physical_history_max=5,
        artifact=tmp_path / "provider_preflight.json",
    )

    assert report["status"] == "PASS"
    assert report["provider_failures"] == 1
    assert report["provider_retries"] == 1
    assert report["provider_failure_attempts"][0]["failure_type"] == (
        "provider_timeout"
    )
    assert report["provider_failure_attempts"][0]["retry_allowed"] is True
    assert attempts == 6
    assert len(successful_requests) == 5


def test_chat_native_tool_preflight_retries_transient_provider_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider = model_client.OpenAICompatibleProvider(
        model="glm-flash",
        base_url="https://hpc-api.westlake.edu.cn/v1",
        timeout_seconds=30,
        api_key_file=None,
    )
    actions = ["wait", "turn_left", "turn_right", "pick", "pick"]
    attempts = 0

    def flaky_post(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise model_client.ProviderRequestError(
                "provider transport request failed",
                failure_type="provider_transport_error",
                retryable=True,
            )
        index = attempts - 2
        arguments = {"action": actions[index]}
        if index == 3:
            arguments["target"] = 7
        elif index == 4:
            arguments.update(target=3, target_description="red cup on counter")
        return {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": f"call_{index}", "type": "function",
                        "function": {
                            "name": "robot_action",
                            "arguments": json.dumps(arguments),
                        },
                    }],
                },
            }],
        }

    monkeypatch.setattr(provider, "_post", flaky_post)
    prompt = Agent.load_common_system_prompt_revision(
        Agent.ACTION20_CORE_INSTRUCTIONS_PATH,
        physical_history_max=5,
    )

    report = provider.capability_preflight(
        prompt_revision=prompt,
        physical_history_max=5,
        artifact=tmp_path / "provider_preflight.json",
    )

    assert report["status"] == "PASS"
    assert report["provider_failures"] == 1
    assert report["provider_retries"] == 1
    assert attempts == 6


def test_responses_function_call_is_accepted_by_existing_agent_parser() -> None:
    response = {
        "id": "resp_test",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.6-sol",
        "output": [
            {"type": "reasoning", "id": "rs_test", "summary": []},
            {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": "robot_action",
                "arguments": '{"action":"wait"}',
                "status": "completed",
            },
        ],
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    }
    normalized = responses_to_chat_completion(response)
    decision = Agent.parse_native_chat_completion(normalized)
    assert decision.action == "wait"
    assert decision.tool_call_id == "call_test"
    assert decision.reasoning_content is None
    assert normalized["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }


def test_real_episode_conversation_round_trip_preserves_prompt_image_and_history(
    tmp_path: Path,
) -> None:
    conversation = Agent.EpisodeConversation(
        Agent.PrivateEpisodeContract(
            episode_id="responses_adapter_round_trip",
            task_id="task0",
            split="train",
            instance_id="0",
            seed=19506,
            model="gpt-5.6-sol",
            project_root=tmp_path,
            instruction="Turn on the radio.",
            physical_history_max=5,
        ),
        _observation(tmp_path, "initial", (64, 64, 64)),
    )
    canonical = conversation.build_native_request()
    wire = chat_request_to_responses(canonical)
    assert wire["instructions"] == Agent.load_common_system_prompt_revision().text
    canonical_image = canonical["messages"][-1]["content"][-1]["image_url"]["url"]
    assert wire["input"][-1]["content"][-1] == {
        "type": "input_image",
        "image_url": canonical_image,
    }
    first_raw = {
        "object": "response",
        "status": "completed",
        "model": "gpt-5.6-sol",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_round_trip_1",
                "name": "robot_action",
                "arguments": '{"action":"wait"}',
            }
        ],
    }
    dispatch = conversation.append_assistant_completion(
        responses_to_chat_completion(first_raw)
    )
    assert dispatch.decision.action == "wait"
    conversation.append_tool_result(Agent.PublicToolResult("completed"))
    conversation.append_observation(
        _observation(tmp_path, "after_wait", (96, 96, 96)),
        kind="physical",
    )

    second_wire = chat_request_to_responses(conversation.build_native_request())
    function_call = next(
        item for item in second_wire["input"] if item.get("type") == "function_call"
    )
    function_output = next(
        item
        for item in second_wire["input"]
        if item.get("type") == "function_call_output"
    )
    assert function_call["call_id"] == "call_round_trip_1"
    assert function_output == {
        "type": "function_call_output",
        "call_id": "call_round_trip_1",
        "output": '{"status":"completed"}',
    }


def test_every_model_visible_action_survives_response_normalization() -> None:
    actions = Agent.ACTION20_TOOL_SCHEMA["function"]["parameters"]["properties"][
        "action"
    ]["enum"]
    assert len(actions) == 20
    for index, action in enumerate(actions):
        raw = {
            "object": "response",
            "status": "completed",
            "model": "gpt-5.6-sol",
            "output": [
                {
                    "type": "function_call",
                    "call_id": f"call_action_{index}",
                    "name": "robot_action",
                    "arguments": json.dumps({"action": action}),
                }
            ],
        }
        decision = Agent.parse_native_chat_completion(
            responses_to_chat_completion(raw)
        )
        assert decision.action == action


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (lambda value: value.update({"temperature": 0.1}), "fields changed"),
        (
            lambda value: value["messages"].append(
                {"role": "system", "content": "second system"}
            ),
            "only one",
        ),
    ],
)
def test_request_conversion_rejects_prompt_or_provider_drift(mutation, pattern) -> None:
    request = _request()
    mutation(request)
    with pytest.raises(ResponsesAdapterError, match=pattern):
        chat_request_to_responses(request)


def test_request_conversion_rejects_broken_tool_result_linkage() -> None:
    request = _request(
        messages=[
            {"role": "system", "content": "ESI system prompt"},
            {"role": "user", "content": "Task: test"},
            {
                "role": "tool",
                "tool_call_id": "missing",
                "content": '{"status":"completed"}',
            },
        ]
    )
    with pytest.raises(ResponsesAdapterError, match="does not match"):
        chat_request_to_responses(request)


def test_response_conversion_rejects_text_or_multiple_calls() -> None:
    with pytest.raises(ResponsesAdapterError, match="unexpected Responses output"):
        responses_to_chat_completion(
            {
                "object": "response",
                "status": "completed",
                "output": [{"type": "message", "content": []}],
            }
        )
    call = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "robot_action",
        "arguments": '{"action":"wait"}',
    }
    with pytest.raises(ResponsesAdapterError, match="exactly one"):
        responses_to_chat_completion(
            {
                "object": "response",
                "status": "completed",
                "output": [call, {**call, "call_id": "call_2"}],
            }
        )
