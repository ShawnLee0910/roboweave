"""Pure contract for the Task25 far-visible Action20 reset canary.

The canary exercises one deliberately visible target just beyond Action20's
direct local workspace.  It repeats the same two-call public interaction on
both sides of one attested in-memory reset.  Marker labels are intentionally
absent from this spec: each target-bound label must come from the current
rendered selection frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from behavior_hl.embodied.runtime.public_action_feedback import (
    TARGET_OUTSIDE_LOCAL_WORKSPACE,
)


ACTION20_TASK25_DISTANCE_RESET_SPEC_VERSION = (
    "behavior2026.action20-task25-distance-reset/v1"
)
TASK25_ID = 25
TASK25_NAME = "clearing_food_from_table_into_fridge"
TASK25_SCENE_MODEL = "house_double_floor_lower"
TASK25_SEED = 19506
TASK25_BASE_EXECUTION = "restricted_kinematic"
TASK25_OBSERVATION_PROFILE = "r1pro_three_rgb_head_eval"
TASK25_LOCAL_REACH_M = 1.30
TASK25_FAR_VISIBLE_MIN_M = 1.34
TASK25_FAR_VISIBLE_MAX_M = 1.40
TASK25_FAR_VISIBLE_STATION_M = 1.37
TASK25_PRIVATE_SETUP_KIND = "symbolic_far_visible_front_view"


@dataclass(frozen=True, slots=True)
class Task25DistanceResetEpisodeSpec:
    """One marker request followed by one expected far-distance rejection."""

    ordinal: int
    selection_action: str = "pick"
    selection_status: str = "selection_required"
    target_action: str = "pick"
    target_status: str = "not_executed"
    target_reason_code: str = TARGET_OUTSIDE_LOCAL_WORKSPACE

    def selection_payload(self) -> dict[str, Any]:
        return {
            "action": self.selection_action,
            "reasoning": "Request current-frame Action20 pick targets.",
        }

    def target_payload(self, marker: str) -> dict[str, Any]:
        if not isinstance(marker, str) or not marker:
            raise ValueError("current-frame marker must be a non-empty string")
        return {
            "action": self.target_action,
            "args": {"marker": marker},
            "reasoning": "Attempt the selected visible Action20 target.",
        }

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Task25DistanceResetSpec:
    task_id: int
    task_name: str
    scene_model: str
    seed: int
    base_execution: str
    observation_profile: str
    episodes: tuple[Task25DistanceResetEpisodeSpec, ...]
    local_reach_m: float = TASK25_LOCAL_REACH_M
    far_visible_min_m: float = TASK25_FAR_VISIBLE_MIN_M
    far_visible_max_m: float = TASK25_FAR_VISIBLE_MAX_M
    far_visible_station_m: float = TASK25_FAR_VISIBLE_STATION_M
    private_setup_kind: str = TASK25_PRIVATE_SETUP_KIND
    scene_load_budget: int = 1
    in_memory_reset_count: int = 1
    public_action_budget: int = 4
    schema_version: str = ACTION20_TASK25_DISTANCE_RESET_SPEC_VERSION

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "scene_model": self.scene_model,
            "seed": self.seed,
            "base_execution": self.base_execution,
            "observation_profile": self.observation_profile,
            "scene_load_budget": self.scene_load_budget,
            "in_memory_reset_count": self.in_memory_reset_count,
            "public_action_budget": self.public_action_budget,
            "episodes": [episode.to_record() for episode in self.episodes],
            "local_reach_m": self.local_reach_m,
            "far_visible_min_m": self.far_visible_min_m,
            "far_visible_max_m": self.far_visible_max_m,
            "far_visible_station_m": self.far_visible_station_m,
            "private_setup_kind": self.private_setup_kind,
            "marker_policy": "dynamic_current_frame_target_binding",
            "same_target_semantics_across_reset": True,
            "setup_public_action_budget": 0,
            "setup_excluded_from_subject": True,
            "model_lane_authorized": False,
            "official_challenge_eligible": False,
            "execution_fidelity": "kinematic_research",
        }


TASK25_DISTANCE_RESET_SPEC = Task25DistanceResetSpec(
    task_id=TASK25_ID,
    task_name=TASK25_NAME,
    scene_model=TASK25_SCENE_MODEL,
    seed=TASK25_SEED,
    base_execution=TASK25_BASE_EXECUTION,
    observation_profile=TASK25_OBSERVATION_PROFILE,
    episodes=(
        Task25DistanceResetEpisodeSpec(ordinal=1),
        Task25DistanceResetEpisodeSpec(ordinal=2),
    ),
)


def validate_task25_distance_reset_spec(
    spec: Task25DistanceResetSpec = TASK25_DISTANCE_RESET_SPEC,
) -> None:
    """Fail closed before a simulator is imported or a scene is loaded."""

    if spec.schema_version != ACTION20_TASK25_DISTANCE_RESET_SPEC_VERSION:
        raise ValueError("Task25 distance/reset spec version is invalid")
    if (spec.task_id, spec.task_name, spec.scene_model, spec.seed) != (
        TASK25_ID,
        TASK25_NAME,
        TASK25_SCENE_MODEL,
        TASK25_SEED,
    ):
        raise ValueError("Task25 distance/reset identity drifted")
    if spec.base_execution != TASK25_BASE_EXECUTION:
        raise ValueError("Task25 distance/reset base execution drifted")
    if spec.observation_profile != TASK25_OBSERVATION_PROFILE:
        raise ValueError("Task25 distance/reset observation profile drifted")
    if spec.private_setup_kind != TASK25_PRIVATE_SETUP_KIND:
        raise ValueError("Task25 distance/reset private setup drifted")
    if (spec.scene_load_budget, spec.in_memory_reset_count) != (1, 1):
        raise ValueError("Task25 distance/reset requires one load and one reset")
    if tuple(episode.ordinal for episode in spec.episodes) != (1, 2):
        raise ValueError("Task25 distance/reset requires two ordered episodes")
    if spec.public_action_budget != 2 * len(spec.episodes):
        raise ValueError("Task25 distance/reset public action budget drifted")
    if spec.public_action_budget > 12:
        raise ValueError("Task25 distance/reset exceeds the bounded action budget")
    if not (
        0.0
        < spec.local_reach_m
        < spec.far_visible_min_m
        <= spec.far_visible_station_m
        <= spec.far_visible_max_m
    ):
        raise ValueError("Task25 far-visible distance band is invalid")
    for episode in spec.episodes:
        if (
            episode.selection_action != "pick"
            or episode.selection_status != "selection_required"
            or episode.target_action != "pick"
            or episode.target_status != "not_executed"
            or episode.target_reason_code
            != TARGET_OUTSIDE_LOCAL_WORKSPACE
        ):
            raise ValueError("Task25 distance/reset public semantics drifted")


def task25_distance_reset_manifest_record(
    spec: Task25DistanceResetSpec = TASK25_DISTANCE_RESET_SPEC,
) -> Mapping[str, Any]:
    validate_task25_distance_reset_spec(spec)
    return spec.to_record()


__all__ = [
    "ACTION20_TASK25_DISTANCE_RESET_SPEC_VERSION",
    "TASK25_DISTANCE_RESET_SPEC",
    "TASK25_FAR_VISIBLE_MAX_M",
    "TASK25_FAR_VISIBLE_MIN_M",
    "TASK25_FAR_VISIBLE_STATION_M",
    "TASK25_LOCAL_REACH_M",
    "Task25DistanceResetEpisodeSpec",
    "Task25DistanceResetSpec",
    "task25_distance_reset_manifest_record",
    "validate_task25_distance_reset_spec",
]
