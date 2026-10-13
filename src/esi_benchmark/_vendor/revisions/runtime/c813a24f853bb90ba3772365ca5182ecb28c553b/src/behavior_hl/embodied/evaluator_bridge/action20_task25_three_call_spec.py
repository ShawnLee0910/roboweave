"""Frozen Task25 Action20 three-call ``place_next_to`` contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from behavior_hl.embodied.evaluator_bridge.action20_task25_atomic_spec import (
    TASK25_BASE_EXECUTION,
    TASK25_HARD_REACH_M,
    TASK25_ID,
    TASK25_NAME,
    TASK25_OBSERVATION_PROFILE,
    TASK25_SCENE_MODEL,
    TASK25_SEED,
)


ACTION20_TASK25_THREE_CALL_SPEC_VERSION = (
    "behavior2026.action20-task25-place-next-to-three-call/v3"
)


@dataclass(frozen=True, slots=True)
class Task25ThreeCallTurnSpec:
    ordinal: int
    binding_stage: str
    expected_status: str
    expected_binding_kind: str | None
    held_before: bool
    held_after: bool

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Task25PlaceNextToThreeCallSpec:
    task_id: int = TASK25_ID
    task_name: str = TASK25_NAME
    scene_model: str = TASK25_SCENE_MODEL
    seed: int = TASK25_SEED
    action: str = "place_next_to"
    selector: str = "screen_right"
    canonical_predicate: str = "NextTo"
    base_execution: str = TASK25_BASE_EXECUTION
    observation_profile: str = TASK25_OBSERVATION_PROFILE
    preferred_station_m: float = 1.20
    hard_reach_m: float = TASK25_HARD_REACH_M
    scene_load_budget: int = 1
    setup_public_action_budget: int = 0
    setup_initial_observation_budget: int = 1
    public_turn_budget: int = 3
    fixture_pair_scope: str = "all_graspable_ordered_pairs"
    fixture_ranking_policy: str = "max_selected_candidate_support_margin"
    schema_version: str = ACTION20_TASK25_THREE_CALL_SPEC_VERSION
    turns: tuple[Task25ThreeCallTurnSpec, ...] = (
        Task25ThreeCallTurnSpec(
            1, "reference_request", "selection_required", "object", True, True
        ),
        Task25ThreeCallTurnSpec(
            2, "reference_selected", "selection_required", "position", True, True
        ),
        Task25ThreeCallTurnSpec(
            3, "position_selected", "completed", None, True, False
        ),
    )

    def payload(self, marker: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": self.action,
            "reasoning": "Action20 Task25 three-call atomic contract.",
        }
        if marker is not None:
            payload["args"] = {"marker": marker}
        return payload

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "scene_model": self.scene_model,
            "seed": self.seed,
            "action": self.action,
            "selector": self.selector,
            "canonical_predicate": self.canonical_predicate,
            "base_execution": self.base_execution,
            "observation_profile": self.observation_profile,
            "preferred_station_m": self.preferred_station_m,
            "hard_reach_m": self.hard_reach_m,
            "scene_load_budget": self.scene_load_budget,
            "setup_public_action_budget": self.setup_public_action_budget,
            "setup_initial_observation_budget": (
                self.setup_initial_observation_budget
            ),
            "public_turn_budget": self.public_turn_budget,
            "fixture_pair_scope": self.fixture_pair_scope,
            "fixture_ranking_policy": self.fixture_ranking_policy,
            "turns": [turn.to_record() for turn in self.turns],
            "private_fixture_setup": True,
            "fixture_policy": (
                "distinct initially-non-NextTo graspables; reference has one "
                "live OnTop support; selected screen position maximizes support margin"
            ),
            "setup_excluded_from_subject": True,
            "model_lane_authorized": False,
            "official_challenge_eligible": False,
            "execution_fidelity": "kinematic_research",
            "promotion_label": "STATIC_CONTRACT_ONLY",
        }


TASK25_PLACE_NEXT_TO_THREE_CALL_SPEC = Task25PlaceNextToThreeCallSpec()


def validate_task25_place_next_to_three_call_spec(
    spec: Task25PlaceNextToThreeCallSpec = TASK25_PLACE_NEXT_TO_THREE_CALL_SPEC,
) -> None:
    if spec.schema_version != ACTION20_TASK25_THREE_CALL_SPEC_VERSION:
        raise ValueError("Task25 three-call schema drifted")
    if (spec.task_id, spec.task_name, spec.scene_model, spec.seed) != (
        TASK25_ID,
        TASK25_NAME,
        TASK25_SCENE_MODEL,
        TASK25_SEED,
    ):
        raise ValueError("Task25 three-call task identity drifted")
    if (
        spec.action,
        spec.selector,
        spec.canonical_predicate,
        spec.base_execution,
        spec.observation_profile,
    ) != (
        "place_next_to",
        "screen_right",
        "NextTo",
        TASK25_BASE_EXECUTION,
        TASK25_OBSERVATION_PROFILE,
    ):
        raise ValueError("Task25 three-call action contract drifted")
    if not (0.0 < spec.preferred_station_m <= spec.hard_reach_m):
        raise ValueError("Task25 three-call station distances are invalid")
    if (
        spec.fixture_pair_scope,
        spec.fixture_ranking_policy,
    ) != (
        "all_graspable_ordered_pairs",
        "max_selected_candidate_support_margin",
    ):
        raise ValueError("Task25 three-call fixture policy drifted")
    if (
        spec.scene_load_budget,
        spec.setup_public_action_budget,
        spec.setup_initial_observation_budget,
        spec.public_turn_budget,
    ) != (1, 0, 1, 3):
        raise ValueError("Task25 three-call budgets drifted")
    expected = (
        (1, "reference_request", "selection_required", "object", True, True),
        (2, "reference_selected", "selection_required", "position", True, True),
        (3, "position_selected", "completed", None, True, False),
    )
    observed = tuple(
        (
            turn.ordinal,
            turn.binding_stage,
            turn.expected_status,
            turn.expected_binding_kind,
            turn.held_before,
            turn.held_after,
        )
        for turn in spec.turns
    )
    if observed != expected:
        raise ValueError("Task25 three-call public turns drifted")


def task25_place_next_to_three_call_manifest_record(
    spec: Task25PlaceNextToThreeCallSpec = TASK25_PLACE_NEXT_TO_THREE_CALL_SPEC,
) -> Mapping[str, Any]:
    validate_task25_place_next_to_three_call_spec(spec)
    return spec.to_record()


__all__ = [
    "ACTION20_TASK25_THREE_CALL_SPEC_VERSION",
    "TASK25_PLACE_NEXT_TO_THREE_CALL_SPEC",
    "Task25PlaceNextToThreeCallSpec",
    "Task25ThreeCallTurnSpec",
    "task25_place_next_to_three_call_manifest_record",
    "validate_task25_place_next_to_three_call_spec",
]
