from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


BENCH_TRY_ROOT = (Path(__file__).resolve().parents[2] / "src/roboweave/_vendor/bench")
if str(BENCH_TRY_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_TRY_ROOT))

from Agent import (
    ACTION20_TOOL_SCHEMA,
    load_common_system_prompt_revision,
    EpisodeConversation,
    PublicToolResult,
    ACTION20_CORE_INSTRUCTIONS,
)
from test_agent_target_description import (
    _completion,
    _contract,
    _conversation,
    _observation,
    _request_candidates,
)


def test_marker_selection_can_be_cancelled_and_resumed(tmp_path) -> None:
    conversation = _conversation(tmp_path)
    _request_candidates(
        conversation,
        tmp_path,
        index=1,
        action="pick",
        observation_name="pick_markers",
        targets=(1, 2),
    )

    with pytest.raises(ValueError, match="two remaining action decisions"):
        conversation.append_assistant_completion(
            _completion(2, "wait"),
            remaining_action_decisions=1,
        )
    assert conversation.pending_marker_action == "pick"

    dispatch = conversation.append_assistant_completion(
        _completion(3, "wait"),
        remaining_action_decisions=2,
    )
    assert dispatch.runtime_action == {"action": "wait"}
    assert conversation.macro_action_count == 1
    conversation.append_tool_result(PublicToolResult("completed"))
    conversation.append_observation(
        _observation(tmp_path, "physical_after_cancel"),
        kind="physical",
    )

    messages = conversation.build_native_request()["messages"]
    serialized = json.dumps(messages, ensure_ascii=False)
    assert "selection_cancelled" in serialized
    assert "C1" not in serialized
    assert conversation.macro_action_count == 2

    restored = EpisodeConversation.from_artifact_transcript(
        conversation.artifact_transcript(),
        expected_contract=_contract(tmp_path),
        system_prompt_revision=load_common_system_prompt_revision(),
        artifact_root=tmp_path,
    )
    assert restored.macro_action_count == 2
    assert restored.build_native_request()["messages"] == messages


def test_final_budget_rejects_done_before_marker_state_changes(tmp_path) -> None:
    conversation = _conversation(tmp_path)
    _request_candidates(
        conversation,
        tmp_path,
        index=1,
        action="pick",
        observation_name="pick_markers",
        targets=(1,),
    )
    before = conversation.artifact_transcript()

    with pytest.raises(ValueError, match="two remaining action decisions"):
        conversation.append_assistant_completion(
            _completion(2, "done"),
            remaining_action_decisions=0,
        )

    assert conversation.artifact_transcript() == before
    conversation.append_assistant_completion(
        _completion(3, "pick", target=1, target_description="trash"),
        remaining_action_decisions=0,
    )
    conversation.append_tool_result(PublicToolResult("completed"))
    conversation.append_observation(
        _observation(tmp_path, "physical_after_pick"),
        kind="physical",
    )
    assert conversation.macro_action_count == 1


def test_stage_two_cancellation_keeps_held_object(tmp_path) -> None:
    conversation = _conversation(tmp_path)
    _request_candidates(
        conversation,
        tmp_path,
        index=1,
        action="pick",
        observation_name="pick_markers",
        targets=(1,),
    )
    conversation.append_assistant_completion(
        _completion(2, "pick", target=1, target_description="red cup")
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
        observation_name="place_markers",
        targets=(1,),
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
        _observation(tmp_path, "placement_markers"),
        kind="marker",
        marker_targets=(2,),
    )

    conversation.append_assistant_completion(
        _completion(5, "wait"),
        remaining_action_decisions=2,
    )
    conversation.append_tool_result(PublicToolResult("completed"))
    conversation.append_observation(
        _observation(tmp_path, "physical_after_cancel"),
        kind="physical",
    )

    transcript = conversation.artifact_transcript()
    assert transcript["agent_authored_state"]["held_object_description"] == (
        "red cup"
    )
    serialized = json.dumps(
        conversation.build_native_request()["messages"],
        ensure_ascii=False,
    )
    assert "selection_cancelled" in serialized
    assert "open fridge" in serialized
    assert "red cup" in serialized
    assert conversation.macro_action_count == 3


def test_fixed_prompt_and_schema_explain_marker_cancellation() -> None:
    assert "different action without `target`" in ACTION20_CORE_INSTRUCTIONS
    target_description = ACTION20_TOOL_SCHEMA["function"]["parameters"][
        "properties"
    ]["target"]["description"]
    assert "cancel the pending selection" in target_description
    assert "two action decisions total" in target_description
