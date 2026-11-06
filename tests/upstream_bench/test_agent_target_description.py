from __future__ import annotations

import io
import json
from pathlib import Path
import sys

from PIL import Image
import pytest


BENCH_TRY_ROOT = (Path(__file__).resolve().parents[2] / "src/roboweave/_vendor/bench")
if str(BENCH_TRY_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_TRY_ROOT))

from Agent import (  # noqa: E402
    ACTION20_TOOL_SCHEMA,
    BLOCKED_OBSERVATION_REMINDER,
    load_common_system_prompt_revision,
    EpisodeConversation,
    PrivateEpisodeContract,
    PublicImage,
    PublicObservation,
    PublicToolResult,
    ACTION20_CORE_INSTRUCTIONS,
    parse_native_chat_completion,
)
from action20_protocol import (  # noqa: E402
    MODEL_VISIBLE_TOOL_ACTIONS,
    PUBLIC_TOOL_ACTIONS,
)
from agent_artifacts import build_chatml_artifact  # noqa: E402
from agent_protocol import ACTION20_BLOCKED_CATEGORIES  # noqa: E402


def _png(path: Path, color: tuple[int, int, int], edge: int) -> PublicImage:
    buffer = io.BytesIO()
    Image.new("RGB", (edge, edge), color).save(buffer, format="PNG")
    raw = buffer.getvalue()
    path.write_bytes(raw)
    return PublicImage(
        camera=path.stem,
        path=path,
    )


def _observation(root: Path, name: str) -> PublicObservation:
    directory = root / name
    directory.mkdir()
    return PublicObservation(
        observation_id=name,
        images=(
            _png(directory / "head.png", (0, 0, 255), 4),
            _png(directory / "left_wrist.png", (255, 0, 0), 2),
            _png(directory / "right_wrist.png", (0, 255, 0), 2),
        ),
    )


def _contract(tmp_path: Path) -> PrivateEpisodeContract:
    return PrivateEpisodeContract(
        episode_id="target_description_test",
        task_id="task",
        split="train",
        instance_id="0",
        seed=19506,
        model="model",
        project_root=tmp_path,
        instruction="Test task.",
        physical_history_max=5,
    )


def _conversation(tmp_path: Path) -> EpisodeConversation:
    return EpisodeConversation(
        _contract(tmp_path),
        _observation(tmp_path, "physical_0"),
    )


def test_default_conversation_reloads_prompt_and_schema(tmp_path, monkeypatch):
    import agent_prompt

    source = tmp_path / "core.txt"
    schema_path = tmp_path / "tools.json"
    source.write_text("Use robot_action. You have a budget of at most 150 action decisions for the entire task.")
    schema_path.write_text(json.dumps(ACTION20_TOOL_SCHEMA))
    monkeypatch.setattr(agent_prompt, "ACTION20_CORE_INSTRUCTIONS_PATH", source)
    monkeypatch.setattr(agent_prompt, "ACTION20_TOOL_SCHEMA_PATH", schema_path)
    first = EpisodeConversation(_contract(tmp_path), _observation(tmp_path, "first"))
    source.write_text(source.read_text() + "\nNew episode wording.")
    schema = json.loads(schema_path.read_text())
    schema["function"]["description"] += "\nUpdated tool wording."
    schema_path.write_text(json.dumps(schema))
    second = EpisodeConversation(_contract(tmp_path), _observation(tmp_path, "second"))
    assert "New episode wording" not in first.build_native_request()["messages"][0]["content"]
    assert "New episode wording" in second.build_native_request()["messages"][0]["content"]
    assert "Updated tool wording" in second.build_native_request()["tools"][0]["function"]["description"]
    source.write_text("Use robot_action without a budget sentence.")
    with pytest.raises(ValueError, match="budget contract"):
        EpisodeConversation(_contract(tmp_path), _observation(tmp_path, "third"))


def test_transcript_paths_are_run_relative_and_legacy_absolute_paths_restore(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    observations = run / "observations"
    observations.mkdir(parents=True)
    conversation = EpisodeConversation(
        _contract(tmp_path),
        _observation(observations, "physical_0"),
    )

    transcript = conversation.artifact_transcript(artifact_root=run)
    paths = [
        image["path"]
        for event in transcript["events"]
        if "observation" in event
        for image in event["observation"]["images"]
    ]
    assert paths == [
        "observations/physical_0/head.png",
        "observations/physical_0/left_wrist.png",
        "observations/physical_0/right_wrist.png",
    ]
    restored = EpisodeConversation.from_artifact_transcript(
        transcript,
        expected_contract=_contract(tmp_path),
        system_prompt_revision=load_common_system_prompt_revision(),
        artifact_root=run,
    )
    assert restored.artifact_transcript(artifact_root=run) == transcript
    chatml = restored.artifact_chatml(artifact_root=run)
    assert chatml["messages"][1]["content"][1]["image"] == (
        "observations/physical_0/three_view_composite.png"
    )

    legacy = conversation.artifact_transcript()
    assert Path(legacy["events"][1]["observation"]["images"][0]["path"]).is_absolute()
    EpisodeConversation.from_artifact_transcript(
        legacy,
        expected_contract=_contract(tmp_path),
        system_prompt_revision=load_common_system_prompt_revision(),
        artifact_root=run,
    )
    relocated = tmp_path / "relocated_run"
    run.rename(relocated)
    restored_legacy = EpisodeConversation.from_artifact_transcript(
        legacy,
        expected_contract=_contract(tmp_path),
        system_prompt_revision=load_common_system_prompt_revision(),
        artifact_root=relocated,
    )
    assert restored_legacy.artifact_transcript(artifact_root=relocated) == transcript
    legacy_chatml = build_chatml_artifact(
        legacy,
        system_prompt=ACTION20_CORE_INSTRUCTIONS,
        artifact_root=relocated,
    )
    assert legacy_chatml["messages"][1]["content"][1]["image"] == (
        "observations/physical_0/three_view_composite.png"
    )


def _completion(
    index: int,
    action: str,
    *,
    target: int | None = None,
    target_description: str | None = None,
    repeat: int | None = None,
    move_distance_m: float | None = None,
    turn_angle_deg: float | None = None,
) -> dict:
    arguments: dict[str, object] = {"action": action}
    if repeat is not None:
        arguments["repeat"] = repeat
    if move_distance_m is not None:
        arguments["move_distance_m"] = move_distance_m
    if turn_angle_deg is not None:
        arguments["turn_angle_deg"] = turn_angle_deg
    if target is not None:
        arguments["target"] = target
    if target_description is not None:
        arguments["target_description"] = target_description
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": "robot_action",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ]
    }


def _request_candidates(
    conversation: EpisodeConversation,
    tmp_path: Path,
    *,
    index: int,
    action: str,
    observation_name: str,
    targets: tuple[int, ...],
) -> None:
    conversation.append_assistant_completion(_completion(index, action))
    conversation.append_tool_result(
        PublicToolResult("selection_required", stage=1)
    )
    conversation.append_observation(
        _observation(tmp_path, observation_name),
        kind="marker",
        marker_targets=targets,
    )


@pytest.mark.parametrize(
    "source",
    ("reasoning_content", "reasoning", "content_think"),
)
def test_reasoning_replay_preserves_provider_source(
    tmp_path: Path,
    source: str,
) -> None:
    conversation = _conversation(tmp_path)
    response = _completion(1, "wait")
    message = response["choices"][0]["message"]
    reasoning = "Inspect the current observation."
    provider_response = None
    if source == "content_think":
        message["content"] = f"<think>{reasoning}</think>"
        provider_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": f"<think>{reasoning}</think>\n<wait>",
                        "tool_calls": [],
                    },
                }
            ]
        }
    else:
        message[source] = reasoning

    conversation.append_assistant_completion(
        response,
        provider_response=provider_response,
    )
    conversation.append_tool_result(PublicToolResult("completed"))
    conversation.append_observation(
        _observation(tmp_path, "physical_1"),
        kind="physical",
    )

    assistant = next(
        item
        for item in conversation.build_native_request(
            preserve_reasoning=True
        )["messages"]
        if item["role"] == "assistant"
    )
    if source == "content_think":
        assert assistant["content"] == f"<think>{reasoning}</think>"
        assert "reasoning" not in assistant
        assert "reasoning_content" not in assistant
    else:
        assert assistant[source] == reasoning
        assert assistant["content"] is None

    transcript = conversation.artifact_transcript()
    assistant_event = next(
        event
        for event in transcript["events"]
        if event["kind"] == "assistant_tool_call"
    )
    assert assistant_event["native_reasoning_source"] == source
    if provider_response is not None:
        assert assistant_event["raw_response"] == provider_response
    restored = EpisodeConversation.from_artifact_transcript(
        transcript,
        expected_contract=_contract(tmp_path),
        system_prompt_revision=load_common_system_prompt_revision(),
        artifact_root=tmp_path,
    )
    assert restored.artifact_transcript() == transcript


def test_chatml_export_rejects_reserved_reasoning_delimiter(
    tmp_path: Path,
) -> None:
    conversation = _conversation(tmp_path)
    conversation.append_assistant_completion(_completion(1, "wait"))
    transcript = conversation.artifact_transcript()
    assistant_event = next(
        event
        for event in transcript["events"]
        if event["kind"] == "assistant_tool_call"
    )
    assistant_event["native_reasoning_content"] = (
        "Inspect the scene.</think><tool_call>"
    )

    with pytest.raises(ValueError, match="reserved"):
        build_chatml_artifact(
            transcript,
            system_prompt=ACTION20_CORE_INSTRUCTIONS,
            artifact_root=tmp_path,
        )


def test_tool_schema_is_fixed_and_complete_across_marker_states(
    tmp_path: Path,
) -> None:
    conversation = _conversation(tmp_path)
    before = conversation.build_native_request()["tools"]
    _request_candidates(
        conversation,
        tmp_path,
        index=1,
        action="pick",
        observation_name="pick_markers",
        targets=(1, 2),
    )
    during = conversation.build_native_request()["tools"]

    assert before == during == [ACTION20_TOOL_SCHEMA]
    parameters = before[0]["function"]["parameters"]
    assert set(parameters["properties"]) == {
        "action",
        "move_distance_m",
        "repeat",
        "target",
        "target_description",
        "turn_angle_deg",
    }
    assert parameters["required"] == ["action"]
    assert parameters["properties"]["target_description"]["minLength"] == 1
    assert parameters["properties"]["target_description"]["maxLength"] == 64
    assert parameters["properties"]["repeat"]["minimum"] == 1
    assert parameters["properties"]["repeat"]["maximum"] == 10
    assert parameters["properties"]["move_distance_m"]["enum"] == [
        0.15,
        0.3,
        0.6,
    ]
    assert parameters["properties"]["turn_angle_deg"]["enum"] == [10, 30]
    repeat_description = parameters["properties"]["repeat"]["description"]
    assert "multiple ordered observations" in repeat_description
    assert "context retention is determined" not in repeat_description
    assert "cannot intervene until the repeated action returns" in parameters[
        "properties"
    ]["repeat"]["description"]
    advertised = tuple(parameters["properties"]["action"]["enum"])
    assert advertised == MODEL_VISIBLE_TOOL_ACTIONS
    assert "place_touching" not in advertised
    assert "place_touching" not in PUBLIC_TOOL_ACTIONS
    description = before[0]["function"]["description"]
    assert "action_decisions_remaining" in description
    assert "One-decision Marker actions (usually two calls)" in description
    assert "One-decision, two-stage Marker actions (usually three calls)" in description
    assert "place_on_top" in description
    assert "place_under" in description
    assert "place_touching" not in description
    assert "within about 1.7 m horizontally" in description
    assert "one three-view RGB composite" in description
    assert "at most 150 action decisions" not in description
    one_stage, two_stages = description.split(
        "One-decision, two-stage Marker actions (usually three calls):", 1
    )
    one_stage = one_stage.split("One-decision Marker actions (usually two calls):", 1)[1]
    assert "place_on_top" not in one_stage
    assert "place_under" in one_stage
    assert "place_on_top: request, select a support, then select a marked surface" in two_stages
    assert "if another selection_required is returned" in two_stages
    for category in ACTION20_BLOCKED_CATEGORIES:
        assert category in description
    for obsolete in (
        "no_visible_candidates", "target_outside_local_workspace",
        "navigation_path_blocked", "navigation_collision_blocked",
    ):
        assert obsolete not in description
    action_description = parameters["properties"]["action"]["description"]
    assert "move_distance_m" in action_description
    assert "turn_angle_deg" in action_description
    assert "In Symbolic mode, no physics steps advance" in action_description
    assert "Advance 30 simulator steps" not in action_description


def test_system_prompt_describes_behavior_without_backend_details() -> None:
    normalized = " ".join(ACTION20_CORE_INSTRUCTIONS.split())
    assert "Invoke only one `robot_action` at a time" in normalized
    assert "Do not batch or parallelize robot actions" in normalized
    assert "Ordinary assistant content must be empty" not in normalized
    assert "target_description" in normalized
    assert "select a suitable current C-label using the same action" in normalized
    assert "choose a different action without `target` to cancel" in normalized
    assert "`completed` confirms the action, not the full task" in normalized
    assert "Check the latest observation" in normalized
    assert "reassess the latest observation" in normalized
    assert "Arms and grippers may extend beyond the base footprint" in normalized
    assert "their pose is not directly controllable" in normalized
    assert "at most 150 action decisions for the entire task" in normalized
    assert "Do not repeat the same failed choice unchanged" in normalized
    assert (
        "A Marker action, from request through all same-action selections and final "
        "execution, counts as one decision"
    ) in normalized
    assert "Navigation with `repeat` counts as one decision" in normalized
    assert "cannot intervene until the repeated action returns" in normalized
    assert "Each result reports `action_decisions_remaining`" in normalized
    assert "At 0, only a pending Marker action may continue" in normalized
    assert "replacement as two decisions" in normalized
    assert "request fresh Markers before retrying a manipulation" in normalized
    assert "or when no viable recovery strategy remains" in normalized
    assert "is not by itself sufficient to call `done`" in normalized
    for forbidden in (
        "<tool_call>",
        "move_forward",
        "1.0 m",
        "10 degrees",
        "1.3 m",
        "dynamic schema",
        "internal hash",
        "Marker backend state machine",
    ):
        assert forbidden not in ACTION20_CORE_INSTRUCTIONS


def test_common_harness_keeps_tool_only_response_contract() -> None:
    normalized = " ".join(load_common_system_prompt_revision().text.split())
    assert "Common Tool-Calling Harness" in normalized
    assert "exactly one `robot_action` tool call" in normalized
    assert "no ordinary assistant content" in normalized


def test_blocked_result_adds_only_generic_current_observation_reminder(
    tmp_path: Path,
) -> None:
    conversation = _conversation(tmp_path)
    conversation.append_assistant_completion(_completion(1, "wait"))
    conversation.append_tool_result(PublicToolResult("blocked"))
    conversation.append_observation(
        _observation(tmp_path, "wait_blocked"),
        kind="physical",
    )

    messages = conversation.build_native_request()["messages"]
    tool_messages = [message for message in messages if message["role"] == "tool"]
    assert [message["content"] for message in tool_messages] == [
        '{"status":"blocked"}'
    ]
    current = messages[-1]
    assert current["role"] == "user"
    assert current["content"][0]["text"] == (
        "Current onboard observation.\n\n" + BLOCKED_OBSERVATION_REMINDER
    )
    assert "because" not in BLOCKED_OBSERVATION_REMINDER.lower()
    assert "move_" not in BLOCKED_OBSERVATION_REMINDER


def test_completed_result_does_not_add_blocked_reminder(tmp_path: Path) -> None:
    conversation = _conversation(tmp_path)
    conversation.append_assistant_completion(_completion(1, "wait"))
    conversation.append_tool_result(PublicToolResult("completed"))
    conversation.append_observation(
        _observation(tmp_path, "wait_completed"),
        kind="physical",
    )

    current = conversation.build_native_request()["messages"][-1]
    assert current["content"][0]["text"] == "Current onboard observation."
    assert BLOCKED_OBSERVATION_REMINDER not in current["content"][0]["text"]


def test_selection_required_is_temporary_and_stage_bound() -> None:
    assert PublicToolResult(
        "selection_required",
        stage=1,
    ).to_public_dict() == {
        "status": "selection_required",
        "stage": 1,
    }
    with pytest.raises(ValueError, match="stage 1 or 2"):
        PublicToolResult("selection_required")
    with pytest.raises(ValueError, match="cannot expose a selection stage"):
        PublicToolResult("completed", stage=1)


def test_public_result_validates_remaining_action_decisions() -> None:
    assert PublicToolResult(
        "completed",
        action_decisions_remaining=137,
    ).to_public_dict() == {
        "status": "completed",
        "action_decisions_remaining": 137,
    }
    for invalid in (-1, 151, True):
        with pytest.raises(ValueError, match="remaining action decisions"):
            PublicToolResult(
                "completed",
                action_decisions_remaining=invalid,
            )


@pytest.mark.parametrize(
    "category",
    (
        "articulated_part_blocked",
        "execution_unverified",
        "geometric_fit_blocked",
        "navigation_blocked",
        "no_selectable_candidates",
        "out_of_reach",
        "placement_collision_blocked",
        "placement_support_unavailable",
        "relation_not_realized",
        "requires_empty_left_hand",
        "requires_object_in_left_hand",
        "selection_unavailable",
        "view_limit_reached",
        "view_motion_blocked_while_holding",
    ),
)
def test_blocked_category_is_allowlisted_and_blocked_only(category: str) -> None:
    assert len(ACTION20_BLOCKED_CATEGORIES) == 14
    assert PublicToolResult(
        "blocked",
        blocked_category=category,
    ).to_public_dict() == {
        "status": "blocked",
        "blocked_category": category,
    }
    with pytest.raises(ValueError, match="blocked category"):
        PublicToolResult("completed", blocked_category=category)
    with pytest.raises(ValueError, match="blocked category"):
        PublicToolResult("selection_required", stage=1, blocked_category=category)
    repeated = PublicToolResult(
        "blocked",
        blocked_category=category,
        repeat_requested=2,
        repeat_executed=2,
        steps=(
            {
                "step_index": 1,
                "status": "completed",
                "body_odometry": {
                    "forward_m": 0.15,
                    "left_m": 0.0,
                    "yaw_deg": 0.0,
                },
            },
            {
                "step_index": 2,
                "status": "blocked",
                "blocked_category": category,
                "body_odometry": {
                    "forward_m": 0.0,
                    "left_m": 0.0,
                    "yaw_deg": 0.0,
                },
            },
        ),
    ).to_public_dict()
    assert repeated["blocked_category"] == category
    assert repeated["steps"][-1]["blocked_category"] == category


@pytest.mark.parametrize(
    "category",
    (
        "navigation_collision_blocked",
        "navigation_path_blocked",
        "no_visible_candidates",
        "target_outside_local_workspace",
        "too_far",
    ),
)
def test_private_blocked_aliases_are_rejected_at_public_boundary(category: str) -> None:
    with pytest.raises(ValueError, match="blocked category"):
        PublicToolResult("blocked", blocked_category=category)
    with pytest.raises(ValueError, match="repeated-step entry"):
        PublicToolResult(
            "blocked",
            blocked_category="navigation_blocked",
            repeat_requested=2,
            repeat_executed=1,
            steps=(
                {
                    "step_index": 1,
                    "status": "blocked",
                    "blocked_category": category,
                    "body_odometry": {
                        "forward_m": 0.0,
                        "left_m": 0.0,
                        "yaw_deg": 0.0,
                    },
                },
            ),
        )


def test_marker_observation_requires_selection_required(tmp_path: Path) -> None:
    completed_root = tmp_path / "completed"
    completed_root.mkdir()
    completed = _conversation(completed_root)
    completed.append_assistant_completion(_completion(1, "pick"))
    completed.append_tool_result(PublicToolResult("completed"))
    with pytest.raises(ValueError, match="only selection_required"):
        completed.append_observation(
            _observation(completed_root, "invalid_completed_marker"),
            kind="marker",
            marker_targets=(1,),
        )

    selection_root = tmp_path / "selection"
    selection_root.mkdir()
    selection = _conversation(selection_root)
    selection.append_assistant_completion(_completion(2, "pick"))
    selection.append_tool_result(
        PublicToolResult("selection_required", stage=1)
    )
    with pytest.raises(ValueError, match="must return a Marker observation"):
        selection.append_observation(
            _observation(selection_root, "invalid_selection_physical"),
            kind="physical",
        )


def test_non_target_and_candidate_request_arguments_remain_minimal() -> None:
    move = parse_native_chat_completion(_completion(1, "move_forward"))
    candidate_request = parse_native_chat_completion(_completion(2, "pick"))
    assert move.function_arguments() == {"action": "move_forward"}
    assert candidate_request.function_arguments() == {"action": "pick"}


@pytest.mark.parametrize(
    ("action", "repeat"),
    [
        ("move_forward", 1),
        ("move_backward", 10),
        ("move_left", 10),
        ("move_right", 1),
        ("turn_left", 10),
        ("turn_right", 3),
    ],
)
def test_navigation_repeat_is_preserved(action: str, repeat: int) -> None:
    decision = parse_native_chat_completion(
        _completion(1, action, repeat=repeat)
    )

    assert decision.repeat == repeat
    assert decision.function_arguments() == {
        "action": action,
        "repeat": repeat,
    }
    assert decision.runtime_action() == {"action": action}


def test_legacy_frozen_schema_keeps_repeat_disabled(tmp_path: Path) -> None:
    legacy_schema = json.loads(json.dumps(ACTION20_TOOL_SCHEMA))
    properties = legacy_schema["function"]["parameters"]["properties"]
    del properties["repeat"]
    del properties["move_distance_m"]
    del properties["turn_angle_deg"]
    conversation = EpisodeConversation(
        _contract(tmp_path),
        _observation(tmp_path, "legacy_initial"),
        tool_schema=legacy_schema,
    )

    assert "repeat" not in conversation.build_native_request()["tools"][0][
        "function"
    ]["parameters"]["properties"]
    with pytest.raises(ValueError, match="invalid navigation repeat"):
        conversation.append_assistant_completion(
            _completion(1, "turn_right", repeat=2)
        )


@pytest.mark.parametrize(
    ("action", "repeat"),
    [
        ("move_forward", 0),
        ("move_forward", 11),
        ("turn_left", 11),
    ],
)
def test_navigation_repeat_fails_closed(action: str, repeat: int) -> None:
    with pytest.raises(ValueError, match="invalid navigation repeat"):
        parse_native_chat_completion(_completion(1, action, repeat=repeat))


def test_non_navigation_ignores_structured_output_placeholders() -> None:
    decision = parse_native_chat_completion(
        _completion(
            1,
            "wait",
            repeat=1,
            move_distance_m=0.6,
            turn_angle_deg=30,
            target=1,
            target_description="x",
        )
    )

    assert decision.function_arguments() == {"action": "wait"}


@pytest.mark.parametrize(
    ("action", "step_field", "step_size", "expected_physical_repeat"),
    [
        ("move_forward", "move_distance_m", 0.15, 10),
        ("move_backward", "move_distance_m", 0.3, 20),
        ("move_left", "move_distance_m", 0.6, 40),
        ("turn_left", "turn_angle_deg", 10, 10),
        ("turn_right", "turn_angle_deg", 30, 30),
    ],
)
def test_navigation_step_size_expands_to_safe_physical_steps(
    action: str,
    step_field: str,
    step_size: float,
    expected_physical_repeat: int,
) -> None:
    decision = parse_native_chat_completion(
        _completion(1, action, repeat=10, **{step_field: step_size})
    )

    assert decision.physical_repeat == expected_physical_repeat
    assert decision.public_repeat == 10
    assert decision.navigation_substeps == expected_physical_repeat // 10
    assert decision.function_arguments()[step_field] == step_size


@pytest.mark.parametrize(
    ("action", "kwargs"),
    [
        ("move_forward", {"move_distance_m": 0.2}),
        ("turn_right", {"turn_angle_deg": 15}),
    ],
)
def test_navigation_step_size_fails_closed(
    action: str, kwargs: dict[str, float]
) -> None:
    with pytest.raises(ValueError, match="invalid (move distance|turn angle)"):
        parse_native_chat_completion(_completion(1, action, **kwargs))


def test_public_odometry_accepts_one_largest_navigation_step() -> None:
    assert PublicToolResult(
        "completed",
        body_odometry={"forward_m": 0.6, "left_m": 0.0, "yaw_deg": 0.0},
    ).to_public_dict()["body_odometry"]["forward_m"] == 0.6


def test_navigation_repeat_preserves_each_result_and_observation(
    tmp_path: Path,
) -> None:
    conversation = _conversation(tmp_path)
    conversation.append_assistant_completion(
        _completion(1, "turn_right", repeat=3)
    )
    conversation.append_tool_result(
        PublicToolResult(
            "blocked",
            blocked_category="navigation_blocked",
            repeat_requested=3,
            repeat_executed=3,
            steps=tuple(
                {
                    "step_index": index,
                    "status": "blocked" if index == 3 else "completed",
                    "body_odometry": {
                        "forward_m": 0.0,
                        "left_m": 0.0,
                        "yaw_deg": 0.0 if index == 3 else -10.0,
                    },
                    **(
                        {"blocked_category": "navigation_blocked"}
                        if index == 3
                        else {}
                    ),
                }
                for index in range(1, 4)
            ),
        )
    )
    for index in range(1, 4):
        conversation.append_observation(
            _observation(tmp_path, f"repeat_{index}"),
            kind="physical",
        )
        assert conversation.state == (
            "awaiting_assistant" if index == 3 else "awaiting_observation"
        )

    assert conversation.active_observation_ids == (
        "physical_0",
        "repeat_1",
        "repeat_2",
        "repeat_3",
    )
    messages = conversation.build_native_request()["messages"]
    tool_result = json.loads(
        next(message for message in messages if message["role"] == "tool")[
            "content"
        ]
    )
    assert tool_result["repeat_requested"] == 3
    assert tool_result["repeat_executed"] == 3
    assert tool_result["blocked_category"] == "navigation_blocked"
    assert [step["status"] for step in tool_result["steps"]] == [
        "completed",
        "completed",
        "blocked",
    ]
    assert tool_result["steps"][-1]["blocked_category"] == (
        "navigation_blocked"
    )
    observation_labels = [
        message["content"][0]["text"]
        for message in messages
        if message["role"] == "user"
        and isinstance(message.get("content"), list)
        and len(message["content"]) == 2
    ]
    assert observation_labels[-3:] == [
        "Historical completed physical observation. Repeat step 1 of 3.",
        "Historical completed physical observation. Repeat step 2 of 3.",
        (
            "Current onboard observation. Repeat step 3 of 3.\n\n"
            + BLOCKED_OBSERVATION_REMINDER
        ),
    ]
    transcript = conversation.artifact_transcript()
    restored = EpisodeConversation.from_artifact_transcript(
        transcript,
        expected_contract=_contract(tmp_path),
        system_prompt_revision=load_common_system_prompt_revision(),
        artifact_root=tmp_path,
    )
    assert restored.artifact_transcript() == transcript

    changed_prompt = json.loads(json.dumps(transcript))
    changed_prompt["system_prompt_revision"]["rendered_prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="resume transcript identity"):
        EpisodeConversation.from_artifact_transcript(
            changed_prompt, expected_contract=_contract(tmp_path),
            system_prompt_revision=load_common_system_prompt_revision(), artifact_root=tmp_path,
        )
    path_only = json.loads(json.dumps(transcript))
    path_only["system_prompt_revision"] = {"source_path": transcript["system_prompt_revision"]["source_path"]}
    restored_path_only = EpisodeConversation.from_artifact_transcript(
        path_only, expected_contract=_contract(tmp_path),
        system_prompt_revision=load_common_system_prompt_revision(), artifact_root=tmp_path,
    )
    assert restored_path_only.artifact_transcript() == transcript

    legacy = json.loads(json.dumps(transcript))
    legacy["transcript_version"] = (
        "behavior-hl-action20-private-transcript-v5-repeat"
    )
    legacy["interface_identity"] = {"legacy": "ignored"}
    legacy["system_prompt_revision"]["sha256"] = "ignored"
    for event in legacy["events"]:
        observation = event.get("observation")
        if observation is None:
            continue
        for image in observation["images"]:
            image["sha256"] = "ignored"
    restored_legacy = EpisodeConversation.from_artifact_transcript(
        legacy,
        expected_contract=_contract(tmp_path),
        system_prompt_revision=load_common_system_prompt_revision(),
        artifact_root=tmp_path,
    )
    assert restored_legacy.artifact_transcript() == transcript


@pytest.mark.parametrize(
    ("target", "description", "error"),
    [
        (1, "x" * 65, "1-64 printable"),
        (1, " leading space", "1-64 printable"),
        (1, "line\nbreak", "1-64 printable"),
    ],
)
def test_target_description_pair_and_shape_fail_closed(
    target: int | None,
    description: str | None,
    error: str,
) -> None:
    response = _completion(
        1,
        "pick",
        target=target,
        target_description=description,
    )
    with pytest.raises(ValueError, match=error):
        parse_native_chat_completion(response)


def test_orphan_target_description_is_a_minimal_candidate_request() -> None:
    decision = parse_native_chat_completion(
        _completion(
            1,
            "pick",
            target_description="left red apple",
        )
    )

    assert decision.target is None
    assert decision.target_description is None
    assert decision.function_arguments() == {"action": "pick"}
    assert decision.runtime_action() == {"action": "pick"}


def test_target_description_with_target_remains_attached() -> None:
    decision = parse_native_chat_completion(
        _completion(
            1,
            "pick",
            target=2,
            target_description="left red apple",
        )
    )

    assert decision.target == 2
    assert decision.target_description == "left red apple"
    assert decision.function_arguments() == {
        "action": "pick",
        "target": 2,
        "target_description": "left red apple",
    }


def test_exact_tool_call_envelope_inside_arguments_is_unwrapped() -> None:
    response = _completion(1, "wait")
    response["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ] = '<tool_call>{"name":"robot_action","arguments":{"action":"turn_left"}}</tool_call>'

    decision = parse_native_chat_completion(response)

    assert decision.function_arguments() == {"action": "turn_left"}


def test_orphan_target_description_does_not_select_a_pending_marker(
    tmp_path: Path,
) -> None:
    conversation = _conversation(tmp_path)
    _request_candidates(
        conversation,
        tmp_path,
        index=1,
        action="pick",
        observation_name="pick_markers_for_orphan_description",
        targets=(1, 2),
    )

    dispatch = conversation.append_assistant_completion(
        _completion(
            2,
            "pick",
            target_description="left red apple",
        )
    )

    assert dispatch.decision.target is None
    assert dispatch.decision.target_description is None
    assert dispatch.decision.function_arguments() == {"action": "pick"}
    assert dict(dispatch.runtime_action) == {"action": "pick"}
    assert (
        conversation.artifact_selected_target_visual_evidence()["record_count"]
        == 0
    )


def test_marker_target_without_description_is_valid_and_minimal(
    tmp_path: Path,
) -> None:
    decision = parse_native_chat_completion(_completion(1, "pick", target=1))
    assert decision.target == 1
    assert decision.target_description is None
    assert decision.function_arguments() == {
        "action": "pick",
        "target": 1,
    }

    conversation = _conversation(tmp_path)
    _request_candidates(
        conversation,
        tmp_path,
        index=2,
        action="pick",
        observation_name="pick_markers_without_description",
        targets=(1, 2),
    )
    dispatch = conversation.append_assistant_completion(
        _completion(3, "pick", target=1)
    )
    assert dict(dispatch.runtime_action) == {
        "action": "pick",
        "args": {"marker": "C1"},
    }
    conversation.append_tool_result(PublicToolResult("completed"))
    conversation.append_observation(
        _observation(tmp_path, "physical_after_descriptionless_pick"),
        kind="physical",
    )

    evidence = conversation.artifact_selected_target_visual_evidence()
    assert evidence["identity_source"] == "selected_marker_frame"
    assert evidence["target_description_optional"] is True
    assert evidence["records"][0]["target_description_provided"] is False
    assert "target_description" not in evidence["records"][0]

    chatml = conversation.artifact_chatml(artifact_root=tmp_path)
    serialized = json.dumps(chatml, ensure_ascii=False)
    assert "<parameter=target_description>" not in serialized
    assert "None" not in serialized


def test_optional_field_normalization_is_counted(tmp_path: Path) -> None:
    navigation = parse_native_chat_completion(
        _completion(
            1,
            "turn_left",
            target=1,
            target_description="x",
            repeat=1,
            move_distance_m=0.15,
            turn_angle_deg=10,
        )
    )
    assert navigation.ignored_target_fields == 2
    assert navigation.ignored_navigation_fields == 1
    assert navigation.eager_marker_target_removed is False

    conversation = _conversation(tmp_path)
    eager = conversation.append_assistant_completion(
        _completion(
            2,
            "pick",
            target=1,
            target_description="x",
            repeat=1,
            move_distance_m=0.15,
            turn_angle_deg=10,
        )
    ).decision
    assert eager.function_arguments() == {"action": "pick"}
    assert eager.ignored_target_fields == 2
    assert eager.ignored_navigation_fields == 3
    assert eager.eager_marker_target_removed is True


def test_explicit_null_marker_target_remains_invalid() -> None:
    response = _completion(1, "pick")
    function = response["choices"][0]["message"]["tool_calls"][0][
        "function"
    ]
    function["arguments"] = json.dumps({"action": "pick", "target": None})
    with pytest.raises(ValueError, match="invalid target"):
        parse_native_chat_completion(response)


def test_marker_target_validation_normalizes_absent_and_stale_but_rejects_wrong(
    tmp_path: Path,
) -> None:
    conversation = _conversation(tmp_path)
    request = conversation.append_assistant_completion(
        _completion(
            1,
            "pick",
            target=1,
            target_description="left red apple",
        )
    )
    assert request.runtime_action == {"action": "pick"}
    assert request.decision.target is None
    conversation.append_tool_result(
        PublicToolResult("selection_required", stage=1)
    )
    conversation.append_observation(
        _observation(tmp_path, "pick_markers"),
        kind="marker",
        marker_targets=(1, 2),
    )
    with pytest.raises(ValueError, match="current same-action Marker set"):
        conversation.append_assistant_completion(
            _completion(
                3,
                "open",
                target=1,
                target_description="right fridge door",
            )
        )
    with pytest.raises(ValueError, match="current same-action Marker set"):
        conversation.append_assistant_completion(
            _completion(
                4,
                "pick",
                target=27,
                target_description="left red apple",
            )
        )

    dispatch = conversation.append_assistant_completion(
        _completion(
            5,
            "pick",
            target=1,
            target_description="roasted chicken on white plate",
        )
    )
    assert dict(dispatch.runtime_action) == {
        "action": "pick",
        "args": {"marker": "C1"},
    }
    assert dispatch.decision.target_description == (
        "roasted chicken on white plate"
    )
    conversation.append_tool_result(PublicToolResult("blocked"))
    conversation.append_observation(
        _observation(tmp_path, "pick_blocked"),
        kind="transient",
    )

    stale = conversation.append_assistant_completion(
        _completion(
            6,
            "pick",
            target=1,
            target_description="roasted chicken on white plate",
        )
    )
    assert stale.runtime_action == {"action": "pick"}
    assert stale.decision.target is None


def test_raw_marker_trace_private_evidence_and_semantic_history_are_separate(
    tmp_path: Path,
) -> None:
    conversation = _conversation(tmp_path)

    _request_candidates(
        conversation,
        tmp_path,
        index=1,
        action="pick",
        observation_name="pick_markers",
        targets=(1, 2),
    )
    conversation.append_assistant_completion(
        _completion(
            2,
            "pick",
            target=1,
            target_description="roasted chicken on white plate",
        )
    )
    conversation.append_tool_result(PublicToolResult("completed"))
    conversation.append_observation(
        _observation(tmp_path, "physical_after_pick"),
        kind="physical",
    )

    _request_candidates(
        conversation,
        tmp_path,
        index=3,
        action="place_inside",
        observation_name="place_stage_1_markers",
        targets=(1, 4),
    )
    conversation.append_assistant_completion(
        _completion(
            4,
            "place_inside",
            target=1,
            target_description="open fridge",
        )
    )
    conversation.append_tool_result(
        PublicToolResult("selection_required", stage=2)
    )
    conversation.append_observation(
        _observation(tmp_path, "place_stage_2_markers"),
        kind="marker",
        marker_targets=(2, 3),
    )

    stage_two_messages = json.dumps(
        conversation.build_native_request()["messages"],
        ensure_ascii=False,
    )
    assert "open fridge" in stage_two_messages
    assert "selection_required" in stage_two_messages
    assert '"target": 1' not in stage_two_messages
    assert "C1" not in stage_two_messages
    assert conversation.pending_marker_stage == 2

    conversation.append_assistant_completion(
        _completion(
            5,
            "place_inside",
            target=2,
            target_description="middle fridge shelf",
        )
    )
    conversation.append_tool_result(PublicToolResult("completed"))
    conversation.append_observation(
        _observation(tmp_path, "physical_final"),
        kind="physical",
    )

    assert conversation.pending_marker_action is None
    assert conversation.pending_marker_targets == frozenset()
    assert conversation.active_observation_ids == (
        "physical_0",
        "physical_after_pick",
        "physical_final",
    )

    transcript = conversation.artifact_transcript()
    raw_arguments = [
        json.loads(event["message"]["tool_calls"][0]["function"]["arguments"])
        for event in transcript["events"]
        if event["kind"] == "assistant_tool_call"
    ]
    assert {
        "action": "pick",
        "target": 1,
        "target_description": "roasted chicken on white plate",
    } in raw_arguments
    assert {
        "action": "place_inside",
        "target": 1,
        "target_description": "open fridge",
    } in raw_arguments
    assert {
        "action": "place_inside",
        "target": 2,
        "target_description": "middle fridge shelf",
    } in raw_arguments

    private_evidence = conversation.artifact_selected_target_visual_evidence()
    assert private_evidence["record_count"] == 3
    records = private_evidence["records"]
    assert [record["stage"] for record in records] == [1, 1, 2]
    assert [record["selected_marker"] for record in records] == [
        "C1",
        "C1",
        "C2",
    ]
    assert [record["status"] for record in records] == [
        "completed",
        "selection_required",
        "completed",
    ]
    assert records[0]["target_description"] == (
        "roasted chicken on white plate"
    )
    assert records[1]["held_object_description_before"] == (
        "roasted chicken on white plate"
    )
    assert records[2]["held_object_description_before"] == (
        "roasted chicken on white plate"
    )
    assert records[2]["held_object_description_after"] is None
    assert records[0]["marker_frame"]["observation"]["observation_id"] == (
        "pick_markers"
    )
    assert "turkey_12" not in json.dumps(private_evidence)

    model_messages = json.dumps(
        conversation.build_native_request()["messages"],
        ensure_ascii=False,
    )
    assert "roasted chicken on white plate" in model_messages
    assert "open fridge" in model_messages
    assert "middle fridge shelf" in model_messages
    assert '"target": 1' not in model_messages
    assert '"target": 2' not in model_messages
    assert "C1" not in model_messages
    assert "C2" not in model_messages
    assert model_messages.count("Previous manipulation outcome:") == 2
    assert "Previous selected-target action:" not in model_messages
    assert '"status":"selection_required"' not in model_messages
    assert conversation.macro_action_count == 2


def test_blocked_pick_collapses_to_one_macro_outcome(tmp_path: Path) -> None:
    conversation = _conversation(tmp_path)
    _request_candidates(
        conversation,
        tmp_path,
        index=1,
        action="pick",
        observation_name="blocked_pick_markers",
        targets=(1, 2),
    )
    conversation.append_assistant_completion(
        _completion(
            2,
            "pick",
            target=1,
            target_description="blue sandal near bed",
        )
    )
    conversation.append_tool_result(
        PublicToolResult(
            "blocked",
            blocked_category="out_of_reach",
        )
    )
    conversation.append_observation(
        _observation(tmp_path, "blocked_pick_final"),
        kind="transient",
    )

    messages = conversation.build_native_request()["messages"]
    serialized = json.dumps(messages, ensure_ascii=False)
    assert serialized.count("Previous manipulation outcome:") == 1
    assert "blue sandal near bed" in serialized
    outcome = next(
        message["content"][0]["text"]
        for message in messages
        if message["role"] == "user"
        and message["content"][0]["text"].startswith(
            "Previous manipulation outcome:"
        )
    )
    assert json.loads(outcome.split(": ", 1)[1])["blocked_category"] == (
        "out_of_reach"
    )
    assert not [message for message in messages if message["role"] == "tool"]
    assert BLOCKED_OBSERVATION_REMINDER in serialized
    assert conversation.macro_action_count == 1


@pytest.mark.parametrize("blocked_stage", (1, 2))
def test_blocked_place_inside_history_preserves_category(
    tmp_path: Path,
    blocked_stage: int,
) -> None:
    conversation = _conversation(tmp_path)
    _request_candidates(
        conversation,
        tmp_path,
        index=1,
        action="place_inside",
        observation_name="container_markers",
        targets=(1,),
    )
    conversation.append_assistant_completion(
        _completion(
            2,
            "place_inside",
            target=1,
            target_description="open fridge",
        )
    )
    if blocked_stage == 2:
        conversation.append_tool_result(
            PublicToolResult("selection_required", stage=2)
        )
        conversation.append_observation(
            _observation(tmp_path, "compartment_markers"),
            kind="marker",
            marker_targets=(2,),
        )
        conversation.append_assistant_completion(
            _completion(
                3,
                "place_inside",
                target=2,
                target_description="middle fridge shelf",
            )
        )
    conversation.append_tool_result(
        PublicToolResult(
            "blocked",
            blocked_category="out_of_reach",
        )
    )
    conversation.append_observation(
        _observation(tmp_path, f"blocked_stage_{blocked_stage}"),
        kind="transient",
    )

    messages = conversation.build_native_request()["messages"]
    outcome = next(
        message["content"][0]["text"]
        for message in messages
        if message["role"] == "user"
        and message["content"][0]["text"].startswith("Previous ")
    )
    event = json.loads(outcome.split(": ", 1)[1])
    assert event["blocked_category"] == (
        "out_of_reach"
    )
    assert event["target_description"] == "open fridge"
    if blocked_stage == 2:
        assert event["placement_description"] == "middle fridge shelf"
