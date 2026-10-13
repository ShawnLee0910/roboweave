"""Task-specific action whitelists without resolved instance information."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from behavior_hl.embodied.policy.action_specs import (
    base_motion_specs,
    grasp_at_spec,
    visible_official_grasp_at_spec,
    visible_object_official_grasp_spec,
    marker_symbolic_grasp_spec,
    marker_symbolic_action23_specs,
    marker_symbolic_put_back_spec,
    marker_symbolic_place_on_floor_spec,
    marker_symbolic_release_spec,
    marker_symbolic_place_inside_spec,
    marker_symbolic_v34_specs,
    mark_visible_objects_spec,
    place_inside_at_spec,
    release_spec,
    stop_spec,
    torso_pitch_specs,
    torso_action23_grid_specs,
    torso_action23_grid_specs,
    torso_look_specs,
    torso_vertical_specs,
    wait_spec,
)
from behavior_hl.embodied.policy.registry import ActionRegistry
from behavior_hl.embodied.policy.protocol import (
    ActionArguments,
    ActionProtocolError,
)


PICKING_UP_TOYS_SCHEMA_VERSION = "picking-up-toys-r1pro-hla-v1"
PICKING_UP_TOYS_RIGHT_CALIBRATED_SCHEMA_VERSION = (
    "picking-up-toys-r1pro-hla-right-calibrated-v1"
)
PICKING_UP_TOYS_VISIBLE_OFFICIAL_GRASP_SCHEMA_VERSION = (
    "picking-up-toys-r1pro-hla-visible-official-grasp-v1"
)
PICKING_UP_TOYS_VISIBLE_OBJECT_GRASP_SCHEMA_VERSION = (
    "picking-up-toys-r1pro-hla-visible-object-official-grasp-v1"
)
PICKING_UP_TOYS_MARKER_SYMBOLIC_GRASP_SCHEMA_VERSION = (
    "picking-up-toys-r1pro-hla-marker-symbolic-grasp-v1"
)
PICKING_UP_TOYS_MARKER_SYMBOLIC_V34_SCHEMA_VERSION = (
    "picking-up-toys-r1pro-esi-marker-symbolic-v110-place-inside"
)
PICKING_UP_TOYS_MARKER_SYMBOLIC_V34_SIMPLE_SPATIAL_SCHEMA_VERSION = (
    "picking-up-toys-r1pro-esi-marker-symbolic-v34-simple-spatial-support-v1"
)
MARKER_SYMBOLIC_ACTION23_SCHEMA_VERSION = (
    "behavior2026-r1pro-esi-marker-symbolic-action23-v1"
)
MARKER_SYMBOLIC_SIMPLE_SPATIAL_SCHEMA_VERSION = (
    "behavior2026-r1pro-esi-marker-symbolic-action23-simple-spatial-support-v2"
)
MARKER_SYMBOLIC_ACTION20_SCHEMA_VERSION = (
    "behavior2026-r1pro-esi-marker-symbolic-action20-"
    "move015-turn10-floor-workspace17-on-top-two-stage-multilevel-"
    "v13-zero-physics-final-collision"
)
ACTION20_LOCAL_WORKSPACE_M = 1.70
ACTION20_LOCAL_APPROACH_MAX_M = 0.40

ACTION20_LOCAL_APPROACH_MODES = frozenset(
    {"disabled", "bounded_0_4m"}
)


def _action20_base_motion_specs() -> tuple[Any, ...]:
    """Describe the exact restricted-kinematic public base contract."""

    descriptions = {
        "move_forward": (
            "Move the robot base forward by one symbolic 0.15 m step. The "
            "static map and live full-body guard check the destination pose; "
            "an unsafe destination is blocked."
        ),
        "move_backward": (
            "Move the robot base backward by one symbolic 0.15 m step. The "
            "static map and live full-body guard check the destination pose; "
            "an unsafe destination is blocked."
        ),
        "move_left": (
            "Move the holonomic robot base left by one symbolic 0.15 m step. "
            "An unsafe destination pose is blocked."
        ),
        "move_right": (
            "Move the holonomic robot base right by one symbolic 0.15 m step. "
            "An unsafe destination pose is blocked."
        ),
        "turn_left": (
            "Rotate the robot base left by one bounded 10-degree step."
        ),
        "turn_right": (
            "Rotate the robot base right by one bounded 10-degree step."
        ),
    }
    return tuple(
        replace(
            spec,
            model_description=descriptions[spec.name],
            required_capabilities=frozenset(
                {"holonomic_base", "official_traversability_map"}
            ),
            max_simulator_steps=1,
            invariants=frozenset(
                {
                    "planar_only",
                    "local_motion",
                    "kinematic_research",
                    "static_eroded_map_guard",
                    "live_full_body_collision_guard",
                    "no_long_range_teleport",
                    "zero_physics_state_write",
                }
            ),
        )
        for spec in base_motion_specs()
    )


# For placement actions the held object is already in hand; the 1.7 m reach
# bound applies to the selected destination, so each names its own referent.
_ACTION20_PLACE_REACH_REFERENTS = {
    "place_inside": "container or compartment",
    "place_on_top": "support",
    "place_next_to": "reference and nearby position",
    "place_under": "upper reference",
}


def _with_action20_local_approach_description(
    specs: Sequence[Any],
    *,
    local_approach_mode: str,
) -> tuple[Any, ...]:
    if local_approach_mode not in ACTION20_LOCAL_APPROACH_MODES:
        raise ValueError("local_approach_mode is invalid")
    target_bound = {
        "pick",
        "place_inside",
        "place_on_top",
        "place_next_to",
        "place_under",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
    }

    def _suffix(name: str) -> str:
        if name == "place_on_top":
            return (
                f" It tries within {ACTION20_LOCAL_WORKSPACE_M:g} m and may "
                "approach that support once; failure "
                "restores the base and leaves the object held."
            )
        if local_approach_mode != "disabled":
            return (
                f" If the same visible selection is outside the "
                f"{ACTION20_LOCAL_WORKSPACE_M:g} m direct workspace but no "
                f"farther than "
                f"{ACTION20_LOCAL_WORKSPACE_M + ACTION20_LOCAL_APPROACH_MAX_M:g} "
                f"m, the executor may approach it by at most "
                f"{ACTION20_LOCAL_APPROACH_MAX_M:g} m before executing."
            )
        referent = _ACTION20_PLACE_REACH_REFERENTS.get(name, "object")
        return (
            f" The selected {referent} must already be within the "
            f"{ACTION20_LOCAL_WORKSPACE_M:g} m "
            "direct local workspace; this action never moves the base."
        )

    return tuple(
        replace(
            spec,
            model_description=(
                (
                    "While holding an object, request support markers, select one, "
                    "then select a marked flat surface on it."
                )
                if spec.name == "place_on_top"
                else spec.model_description
            )
            + _suffix(spec.name),
            max_simulator_steps=(
                4096
                if spec.name == "place_on_top"
                else spec.max_simulator_steps
            ),
            required_capabilities=(
                spec.required_capabilities
                | {"holonomic_base", "official_traversability_map"}
                if spec.name == "place_on_top"
                else spec.required_capabilities
            ),
            invariants=(
                (spec.invariants - {"no_long_range_navigation"})
                | {
                    "same_selected_support_after_navigation",
                    "selected_support_then_horizontal_surface_cell",
                    "single_interaction_station",
                    "transactional_route_and_placement",
                }
                if spec.name == "place_on_top"
                else spec.invariants
            ),
        )
        if spec.name in target_bound
        else spec
        for spec in specs
    )


@dataclass(frozen=True, slots=True)
class TaskProfile:
    """Public task metadata fixed before an episode starts."""

    task_name: str
    task_id: int
    task_language: str
    schema_version: str


PICKING_UP_TOYS_PROFILE = TaskProfile(
    task_name="picking_up_toys",
    task_id=7,
    task_language=(
        "Pick up all board games, jigsaw puzzles, and the tennis ball, "
        "and place them completely inside the toy box."
    ),
    schema_version=PICKING_UP_TOYS_SCHEMA_VERSION,
)


def build_action23_registry(
    *,
    calibrated_hands: Sequence[str] = ("left",),
) -> ActionRegistry:
    """Build the opt-in 22-action surface plus the existing Mark action.

    Object actions use one executor-selected hand, so the hand never appears
    in the public schema.  Keeping this builder separate prevents Action23
    additions from changing the established QA and marker-symbolic-v34
    registry manifests.
    """

    hands = tuple(calibrated_hands)
    if (
        len(hands) != 1
        or hands[0] not in {"left", "right"}
    ):
        raise ValueError(
            "Action23 requires exactly one executor-selected left/right hand"
        )
    specs = (
        *base_motion_specs(),
        *torso_action23_grid_specs(),
        mark_visible_objects_spec(),
        *marker_symbolic_action23_specs(),
        marker_symbolic_release_spec(),
        wait_spec(),
    )
    return ActionRegistry(
        specs,
        schema_version=MARKER_SYMBOLIC_ACTION23_SCHEMA_VERSION,
    )


def build_picking_up_toys_registry(
    *,
    calibrated_hands: Sequence[str] = ("left", "right"),
    grasp_mode: str = "physical_local",
    simple_spatial_selection: bool = False,
    local_approach_mode: str = "disabled",
) -> ActionRegistry:
    if not isinstance(simple_spatial_selection, bool):
        raise TypeError("simple_spatial_selection must be boolean")
    hands = tuple(calibrated_hands)
    if (
        not hands
        or len(set(hands)) != len(hands)
        or not set(hands).issubset({"left", "right"})
    ):
        raise ValueError(
            "calibrated_hands must be unique left/right hand names"
        )
    if grasp_mode == "marker_symbolic_action23" and not simple_spatial_selection:
        return build_action23_registry(calibrated_hands=hands)
    if grasp_mode == "physical_local":
        grasp_spec = grasp_at_spec()
    elif grasp_mode == "visible_official_macro":
        grasp_spec = visible_official_grasp_at_spec()
    elif grasp_mode == "visible_object_official_macro":
        if len(hands) != 1:
            raise ValueError(
                "visible_object_official_macro requires exactly one "
                "calibrated hand because the hand is executor-selected"
            )
        grasp_spec = visible_object_official_grasp_spec()
    elif grasp_mode == "marker_symbolic_macro":
        if len(hands) != 1:
            raise ValueError(
                "marker_symbolic_macro requires exactly one executor-selected "
                "hand"
            )
        grasp_spec = marker_symbolic_grasp_spec()
    elif grasp_mode in {
        "marker_symbolic_v34",
        "marker_symbolic_action23",
        "marker_symbolic_action20",
    }:
        if len(hands) != 1:
            raise ValueError(
                "marker_symbolic_v34 requires exactly one executor-selected "
                "hand"
            )
        grasp_spec = None
    else:
        raise ValueError(
            "grasp_mode must be 'physical_local' or "
            "'visible_official_macro' or "
            "'visible_object_official_macro' or 'marker_symbolic_macro' or "
            "'marker_symbolic_v34', 'marker_symbolic_action23', or "
            "'marker_symbolic_action20'"
        )
    if grasp_mode == "marker_symbolic_action20" and not simple_spatial_selection:
        raise ValueError(
            "marker_symbolic_action20 requires internal action-first Marker selection"
        )
    if local_approach_mode not in ACTION20_LOCAL_APPROACH_MODES:
        raise ValueError("local_approach_mode is invalid")
    place_spec = (
        marker_symbolic_place_inside_spec()
        if grasp_mode == "marker_symbolic_macro"
        else place_inside_at_spec()
    )
    if grasp_mode == "marker_symbolic_action20":
        symbolic_torso = tuple(
            replace(
                spec,
                model_description=(
                    "Pitch the fixed onboard camera "
                    f"{'upward' if spec.name == 'look_up' else 'downward'} "
                    "by one 15-degree grid step using a zero-physics joint "
                    "endpoint write."
                ),
                required_capabilities=frozenset(
                    {"kinematic_torso", "head_rgbd"}
                ),
                max_simulator_steps=1,
                invariants=frozenset(
                    {
                        "physical_camera_only",
                        "bounded_grid",
                        "height_pitch_decoupled",
                        "zero_physics_state_write",
                    }
                ),
            )
            for spec in torso_pitch_specs()
        )
        symbolic_wait = replace(
            wait_spec(),
            model_description=(
                "Refresh the current observation without advancing simulator "
                "time or changing world state."
            ),
            required_capabilities=frozenset({"head_rgbd"}),
            max_simulator_steps=1,
            invariants=frozenset(
                {"zero_physics", "no_world_state_change", "fresh_observation"}
            ),
        )
        specs = (
            *_action20_base_motion_specs(),
            *symbolic_torso,
            *(
                spec
                for spec in marker_symbolic_v34_specs(
                    simple_spatial_selection=True
                )
                if spec.name != "place_touching"
            ),
            marker_symbolic_place_on_floor_spec(),
            symbolic_wait,
        )
        specs = _with_action20_local_approach_description(
            specs,
            local_approach_mode=local_approach_mode,
        )
        specs = tuple(
            replace(
                spec,
                invariants=spec.invariants
                | {"zero_physics"}
                | (
                    {"final_state_no_solid_overlap"}
                    if spec.name
                    in {
                        "place_inside",
                        "place_on_floor",
                        "place_on_top",
                        "place_next_to",
                        "place_under",
                    }
                    else set()
                ),
                model_description=spec.model_description.replace(
                    "tries stable poses",
                    "tries collision-free poses",
                ),
            )
            for spec in specs
        )
    elif grasp_mode == "marker_symbolic_action23":
        specs = (
            *base_motion_specs(),
            *torso_pitch_specs(),
            mark_visible_objects_spec(),
            *marker_symbolic_v34_specs(
                simple_spatial_selection=simple_spatial_selection,
            ),
            marker_symbolic_release_spec(),
            wait_spec(),
        )
    elif grasp_mode == "marker_symbolic_v34":
        specs = (
            *base_motion_specs(),
            *torso_vertical_specs(),
            mark_visible_objects_spec(),
            *marker_symbolic_v34_specs(
                simple_spatial_selection=simple_spatial_selection,
            ),
            marker_symbolic_put_back_spec(),
            stop_spec(),
        )
    else:
        specs = (
            *base_motion_specs(),
            *torso_look_specs(),
            *((mark_visible_objects_spec(),) if grasp_mode == "marker_symbolic_macro" else ()),
            grasp_spec,
            place_spec,
            release_spec(),
            stop_spec(),
        )
    if set(hands) != {"left", "right"}:
        specs = tuple(
            _restrict_spec_hands(spec, hands)
            if (
                spec.name in {"grasp_at", "place_inside_at"}
                or (
                    spec.name == "release"
                    and grasp_mode
                    not in {
                        "marker_symbolic_v34",
                        "marker_symbolic_action23",
                        "marker_symbolic_action20",
                    }
                )
            )
            else spec
            for spec in specs
        )
        if grasp_mode == "marker_symbolic_action20":
            schema_version = (
                f"{MARKER_SYMBOLIC_ACTION20_SCHEMA_VERSION}-"
                f"{local_approach_mode.replace('_', '-')}"
            )
        elif grasp_mode == "marker_symbolic_action23":
            schema_version = (
                MARKER_SYMBOLIC_SIMPLE_SPATIAL_SCHEMA_VERSION
                if simple_spatial_selection
                else MARKER_SYMBOLIC_ACTION23_SCHEMA_VERSION
            )
        elif grasp_mode == "marker_symbolic_v34":
            schema_version = (
                "picking-up-toys-r1pro-esi-marker-symbolic-v34-"
                f"{'-'.join(hands)}-calibrated"
                f"{'-simple-spatial-support-v1' if simple_spatial_selection else ''}"
            )
        elif grasp_mode == "marker_symbolic_macro":
            schema_version = (
                "picking-up-toys-r1pro-hla-marker-symbolic-grasp-"
                f"{'-'.join(hands)}-calibrated-v1"
            )
        elif grasp_mode == "visible_object_official_macro":
            schema_version = (
                "picking-up-toys-r1pro-hla-visible-object-official-grasp-"
                f"{'-'.join(hands)}-calibrated-v1"
            )
        elif grasp_mode == "visible_official_macro":
            schema_version = (
                "picking-up-toys-r1pro-hla-visible-official-grasp-"
                f"{'-'.join(hands)}-calibrated-v1"
            )
        elif hands == ("right",):
            schema_version = (
                PICKING_UP_TOYS_RIGHT_CALIBRATED_SCHEMA_VERSION
            )
        else:
            schema_version = (
                "picking-up-toys-r1pro-hla-"
                f"{'-'.join(hands)}-calibrated-v1"
            )
    elif grasp_mode == "marker_symbolic_action20":
        schema_version = (
            f"{MARKER_SYMBOLIC_ACTION20_SCHEMA_VERSION}-"
            f"{local_approach_mode.replace('_', '-')}"
        )
    elif grasp_mode == "marker_symbolic_action23":
        schema_version = (
            MARKER_SYMBOLIC_SIMPLE_SPATIAL_SCHEMA_VERSION
            if simple_spatial_selection
            else MARKER_SYMBOLIC_ACTION23_SCHEMA_VERSION
        )
    elif grasp_mode == "marker_symbolic_v34":
        schema_version = (
            PICKING_UP_TOYS_MARKER_SYMBOLIC_V34_SIMPLE_SPATIAL_SCHEMA_VERSION
            if simple_spatial_selection
            else PICKING_UP_TOYS_MARKER_SYMBOLIC_V34_SCHEMA_VERSION
        )
    elif grasp_mode == "marker_symbolic_macro":
        schema_version = PICKING_UP_TOYS_MARKER_SYMBOLIC_GRASP_SCHEMA_VERSION
    elif grasp_mode == "visible_object_official_macro":
        schema_version = PICKING_UP_TOYS_VISIBLE_OBJECT_GRASP_SCHEMA_VERSION
    elif grasp_mode == "visible_official_macro":
        schema_version = PICKING_UP_TOYS_VISIBLE_OFFICIAL_GRASP_SCHEMA_VERSION
    else:
        schema_version = PICKING_UP_TOYS_SCHEMA_VERSION
    return ActionRegistry(
        specs,
        schema_version=schema_version,
    )


def build_inside_task_registry(
    *,
    task_name: str,
    calibrated_hands: Sequence[str] = ("left", "right"),
    grasp_mode: str = "marker_symbolic_macro",
    base_execution: str = "physical_velocity",
    simple_spatial_selection: bool = False,
    local_approach_mode: str = "disabled",
) -> ActionRegistry:
    """Build the common Inside-task action surface with task-scoped identity.

    The action semantics are intentionally shared. Only the public task name
    is added to the schema identity; object roles and goal truth stay in the
    evaluator-private InsideTaskProfile.
    """

    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("task_name must be a non-empty string")
    if base_execution not in {
        "physical_velocity",
        "restricted_kinematic",
    }:
        raise ValueError("base_execution mode is invalid")
    if (
        grasp_mode == "marker_symbolic_action20"
        and base_execution != "restricted_kinematic"
    ):
        raise ValueError(
            "marker_symbolic_action20 requires restricted_kinematic base "
            "execution"
        )
    base = build_picking_up_toys_registry(
        calibrated_hands=calibrated_hands,
        grasp_mode=grasp_mode,
        simple_spatial_selection=simple_spatial_selection,
        local_approach_mode=local_approach_mode,
    )
    specs = tuple(base.spec(name) for name in base.names)
    if (
        base_execution == "restricted_kinematic"
        and grasp_mode != "marker_symbolic_action20"
    ):
        specs = tuple(
            replace(
                spec,
                model_description=(
                    "Move the robot base by one bounded local step (up to "
                    "0.15 m for translation or 15 degrees for rotation). "
                    "A translation may stop early at the final open prefix; "
                    "it is blocked when no meaningful direct robot-eroded "
                    "traversability-map prefix is available."
                ),
                required_capabilities=frozenset(
                    {"holonomic_base", "official_traversability_map"}
                ),
                max_simulator_steps=4,
                invariants=frozenset(
                    {
                        "planar_only",
                        "local_motion",
                        "kinematic_research",
                        "static_eroded_map_guard",
                        "no_long_range_teleport",
                    }
                ),
            )
            if spec.name
            in {
                "move_forward",
                "move_backward",
                "move_left",
                "move_right",
                "turn_left",
                "turn_right",
            }
            else spec
            for spec in specs
        )
    return ActionRegistry(
        specs,
        schema_version=(
            f"{base.schema_version}-task-{task_name}-inside-"
            f"{base_execution.replace('_', '-')}-v2"
        ),
    )


def _restrict_spec_hands(spec: Any, hands: tuple[str, ...]) -> Any:
    parser = spec.parse_args

    def parse_admitted(
        payload: Mapping[str, Any],
    ) -> ActionArguments:
        result = parser(payload)
        if result.hand not in hands:
            raise ActionProtocolError(
                f"'hand' is not calibrated for this episode; allowed: "
                f"{', '.join(hands)}"
            )
        return result

    schema = dict(spec.args_schema)
    properties = dict(schema["properties"])
    properties["hand"] = {
        "type": "string",
        "enum": list(hands),
    }
    schema["properties"] = MappingProxyType(properties)
    return replace(
        spec,
        parse_args=parse_admitted,
        args_schema=MappingProxyType(schema),
    )
