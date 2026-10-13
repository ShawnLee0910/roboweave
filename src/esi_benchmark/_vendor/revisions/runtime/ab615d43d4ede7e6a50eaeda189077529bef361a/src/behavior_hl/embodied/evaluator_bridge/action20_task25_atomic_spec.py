"""Pure contract for the Task25 public Action20 carry canary.

The canary has one private workstation setup and one short public carry chain.
It deliberately does not claim task completion: its only purpose is to prove
that a completed pick is visually synchronized and remains held across the
restricted-kinematic base surface until a public floor placement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


ACTION20_TASK25_ATOMIC_SPEC_VERSION = (
    "behavior2026.action20-task25-carry-atomic/v2"
)
TASK25_ID = 25
TASK25_NAME = "clearing_food_from_table_into_fridge"
TASK25_SCENE_MODEL = "house_double_floor_lower"
TASK25_SEED = 19506
TASK25_MARKER = "C1"
TASK25_BASE_EXECUTION = "restricted_kinematic"
TASK25_OBSERVATION_PROFILE = "r1pro_three_rgb_head_eval"
TASK25_PREFERRED_STATION_M = 1.20
TASK25_HARD_REACH_M = 1.30
TASK25_PRIVATE_SETUP_KIND = "official_eroded_map_station"


@dataclass(frozen=True, slots=True)
class Task25AtomicTurnSpec:
    """One public subject turn and its carry boundary contract."""

    ordinal: int
    action: str
    allowed_statuses: tuple[str, ...]
    held_before: bool
    held_after: bool
    preserve_held_identity: bool = False
    marker: str | None = None

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": self.action,
            "reasoning": "Action20 Task25 atomic carry contract.",
        }
        if self.marker is not None:
            payload["args"] = {"marker": self.marker}
        return payload

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FuturePlacementTurnSpec:
    """A frozen follow-up capability that this carry canary does not execute."""

    action: str
    required_precondition: str
    enabled: bool = False

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Task25CarryAtomicSpec:
    task_id: int
    task_name: str
    scene_model: str
    seed: int
    base_execution: str
    observation_profile: str
    selection_action: str
    selection_status: str
    selection_marker: str
    turns: tuple[Task25AtomicTurnSpec, ...]
    future_placement_turns: tuple[FuturePlacementTurnSpec, ...]
    preferred_station_m: float = TASK25_PREFERRED_STATION_M
    hard_reach_m: float = TASK25_HARD_REACH_M
    private_setup_kind: str = TASK25_PRIVATE_SETUP_KIND
    setup_public_action_budget: int = 0
    setup_simulator_step_budget: int = 0
    setup_action_index_increment: int = 0
    scene_load_budget: int = 1
    schema_version: str = ACTION20_TASK25_ATOMIC_SPEC_VERSION

    def selection_payload(self) -> dict[str, Any]:
        return {
            "action": self.selection_action,
            "reasoning": "Request current-frame Action20 pick targets.",
        }

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
            "selection": {
                "action": self.selection_action,
                "required_status": self.selection_status,
                "marker": self.selection_marker,
            },
            "turns": [turn.to_record() for turn in self.turns],
            "future_placement_turns": [
                turn.to_record() for turn in self.future_placement_turns
            ],
            "preferred_station_m": self.preferred_station_m,
            "hard_reach_m": self.hard_reach_m,
            "private_setup_kind": self.private_setup_kind,
            "setup_public_action_budget": self.setup_public_action_budget,
            "setup_simulator_step_budget": self.setup_simulator_step_budget,
            "setup_action_index_increment": self.setup_action_index_increment,
            "private_fixture_setup": True,
            "setup_excluded_from_subject": True,
            "model_lane_authorized": False,
            "official_challenge_eligible": False,
            "execution_fidelity": "kinematic_research",
        }


TASK25_CARRY_ATOMIC_SPEC = Task25CarryAtomicSpec(
    task_id=TASK25_ID,
    task_name=TASK25_NAME,
    scene_model=TASK25_SCENE_MODEL,
    seed=TASK25_SEED,
    base_execution=TASK25_BASE_EXECUTION,
    observation_profile=TASK25_OBSERVATION_PROFILE,
    selection_action="pick",
    selection_status="selection_required",
    selection_marker=TASK25_MARKER,
    turns=(
        Task25AtomicTurnSpec(
            1,
            "pick",
            ("completed",),
            held_before=False,
            held_after=True,
            marker=TASK25_MARKER,
        ),
        Task25AtomicTurnSpec(
            2,
            "move_left",
            ("completed", "blocked"),
            held_before=True,
            held_after=True,
            preserve_held_identity=True,
        ),
        Task25AtomicTurnSpec(
            3,
            "turn_right",
            ("completed", "blocked"),
            held_before=True,
            held_after=True,
            preserve_held_identity=True,
        ),
        Task25AtomicTurnSpec(
            4,
            "move_forward",
            ("completed", "blocked"),
            held_before=True,
            held_after=True,
            preserve_held_identity=True,
        ),
        Task25AtomicTurnSpec(
            5,
            "move_backward",
            ("completed", "blocked"),
            held_before=True,
            held_after=True,
            preserve_held_identity=True,
        ),
        Task25AtomicTurnSpec(
            6,
            "place_on_floor",
            ("completed",),
            held_before=True,
            held_after=False,
        ),
    ),
    future_placement_turns=(
        FuturePlacementTurnSpec("place_inside", "held + visible container"),
        FuturePlacementTurnSpec("place_next_to", "held + visible peer"),
        FuturePlacementTurnSpec("place_touching", "held + visible peer"),
    ),
)


def validate_task25_carry_atomic_spec(
    spec: Task25CarryAtomicSpec = TASK25_CARRY_ATOMIC_SPEC,
) -> None:
    """Fail closed before any simulator process is started."""

    if spec.schema_version != ACTION20_TASK25_ATOMIC_SPEC_VERSION:
        raise ValueError("Task25 atomic spec version is invalid")
    if (spec.task_id, spec.task_name, spec.scene_model, spec.seed) != (
        TASK25_ID,
        TASK25_NAME,
        TASK25_SCENE_MODEL,
        TASK25_SEED,
    ):
        raise ValueError("Task25 atomic task identity drifted")
    if spec.base_execution != TASK25_BASE_EXECUTION:
        raise ValueError("Task25 atomic base execution must be restricted")
    if spec.observation_profile != TASK25_OBSERVATION_PROFILE:
        raise ValueError("Task25 atomic observation profile drifted")
    if spec.scene_load_budget != 1:
        raise ValueError("Task25 atomic canary permits exactly one scene load")
    if not (0.0 < spec.preferred_station_m <= spec.hard_reach_m):
        raise ValueError("Task25 atomic interaction distances are invalid")
    if spec.private_setup_kind != TASK25_PRIVATE_SETUP_KIND:
        raise ValueError("Task25 atomic private setup kind drifted")
    if (
        spec.setup_public_action_budget,
        spec.setup_simulator_step_budget,
        spec.setup_action_index_increment,
    ) != (0, 0, 0):
        raise ValueError("Task25 atomic setup must not consume public actions")
    if (
        spec.selection_action != "pick"
        or spec.selection_status != "selection_required"
        or spec.selection_marker != TASK25_MARKER
    ):
        raise ValueError("Task25 atomic selection contract drifted")
    expected_actions = (
        "pick",
        "move_left",
        "turn_right",
        "move_forward",
        "move_backward",
        "place_on_floor",
    )
    if tuple(turn.action for turn in spec.turns) != expected_actions:
        raise ValueError("Task25 atomic public turn order drifted")
    if tuple(turn.ordinal for turn in spec.turns) != tuple(
        range(1, len(spec.turns) + 1)
    ):
        raise ValueError("Task25 atomic turn ordinals are invalid")
    pick, *motions, floor_place = spec.turns
    if (
        pick.marker != TASK25_MARKER
        or pick.held_before
        or not pick.held_after
        or pick.allowed_statuses != ("completed",)
    ):
        raise ValueError("Task25 atomic pick contract drifted")
    if any(
        not turn.held_before
        or not turn.held_after
        or not turn.preserve_held_identity
        or turn.allowed_statuses != ("completed", "blocked")
        for turn in motions
    ):
        raise ValueError("Task25 atomic carry motion contract drifted")
    if (
        not floor_place.held_before
        or floor_place.held_after
        or floor_place.allowed_statuses != ("completed",)
    ):
        raise ValueError("Task25 atomic floor-placement contract drifted")
    future = tuple(turn.action for turn in spec.future_placement_turns)
    if future != ("place_inside", "place_next_to", "place_touching") or any(
        turn.enabled for turn in spec.future_placement_turns
    ):
        raise ValueError("Task25 future placement declarations drifted")


def task25_carry_atomic_manifest_record(
    spec: Task25CarryAtomicSpec = TASK25_CARRY_ATOMIC_SPEC,
) -> Mapping[str, Any]:
    validate_task25_carry_atomic_spec(spec)
    return spec.to_record()


__all__ = [
    "ACTION20_TASK25_ATOMIC_SPEC_VERSION",
    "FuturePlacementTurnSpec",
    "TASK25_CARRY_ATOMIC_SPEC",
    "TASK25_HARD_REACH_M",
    "TASK25_MARKER",
    "TASK25_NAME",
    "TASK25_PRIVATE_SETUP_KIND",
    "Task25AtomicTurnSpec",
    "Task25CarryAtomicSpec",
    "task25_carry_atomic_manifest_record",
    "validate_task25_carry_atomic_spec",
]
