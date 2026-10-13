import pytest

from Agent import EpisodeConversation, PublicToolResult, load_common_system_prompt_revision
from agent_protocol import ARTICULATED_BLOCKED_DETAILS
from lab.model_client import _public_tool_result_from_scene_step
from test_agent_target_description import _conversation, _contract, _observation, _request_candidates, _completion


@pytest.mark.parametrize("detail", sorted(ARTICULATED_BLOCKED_DETAILS))
def test_detail_reaches_model_result_and_restores_from_history(tmp_path, detail):
    conversation = _conversation(tmp_path)
    _request_candidates(conversation, tmp_path, index=0, action="open", observation_name="markers", targets=(1,))
    conversation.append_assistant_completion(_completion(1, "open", target=1, target_description="box"))
    payload = {"status": "blocked", "blocked_category": "articulated_part_blocked", "blocked_detail": detail}
    result = _public_tool_result_from_scene_step(
        payload, {"kind": "physical"}, expected_selection_stage=None,
        allow_blocked_category=True, allow_blocked_detail=True,
    )
    assert result.to_public_dict() == payload
    conversation.append_tool_result(result)
    conversation.append_observation(_observation(tmp_path, "after"), kind="physical")
    transcript = conversation.artifact_transcript()
    restored = EpisodeConversation.from_artifact_transcript(
        transcript, expected_contract=_contract(tmp_path),
        system_prompt_revision=load_common_system_prompt_revision(), artifact_root=tmp_path,
    )
    assert restored.artifact_transcript() == transcript
    # A resumed old schema that did not advertise detail keeps its old output.
    legacy = _public_tool_result_from_scene_step(payload, {"kind": "physical"},
        expected_selection_stage=None, allow_blocked_category=True)
    assert "blocked_detail" not in legacy.to_public_dict()


@pytest.mark.parametrize("status,category,detail", [
    ("completed", None, "robot_obstruction"),
    ("blocked", "out_of_reach", "robot_obstruction"),
    ("blocked", "articulated_part_blocked", "/World/robot/private"),
    ("blocked", "articulated_part_blocked", "ROBOT_OBSTRUCTION"),
])
def test_detail_rejects_untrusted_or_mismatched_values(status, category, detail):
    with pytest.raises(ValueError):
        PublicToolResult(status, blocked_category=category, blocked_detail=detail)
