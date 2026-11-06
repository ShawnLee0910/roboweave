"""Torso-driven camera-view action metadata."""

from behavior_hl.embodied.policy.action_specs.base_motion import (
    EMPTY_ARGS_SCHEMA,
)
from behavior_hl.embodied.policy.protocol import parse_no_args
from behavior_hl.embodied.policy.registry import ActionSpec


def _look_spec(name: str, description: str, controller_key: str) -> ActionSpec:
    return ActionSpec(
        name=name,
        parse_args=parse_no_args,
        args_schema=EMPTY_ARGS_SCHEMA,
        args_required=False,
        model_description=description,
        required_capabilities=frozenset({"physical_torso", "head_rgbd"}),
        max_simulator_steps=180,
        public_statuses=(
            "completed",
            "unsafe_while_carrying",
            "joint_stalled",
        ),
        invariants=frozenset(
            {"physical_camera_only", "bounded_profile", "arm_safe"}
        ),
        controller_key=controller_key,
    )


def torso_look_specs() -> tuple[ActionSpec, ...]:
    return (
        _look_spec(
            "look_up",
            "Raise the onboard head view using bounded physical torso motion. "
            "While carrying, movement above the validated safe interval is rejected.",
            "torso.look_up",
        ),
        _look_spec(
            "look_down",
            "Lower the onboard head view using bounded physical torso motion. "
            "While carrying, only the validated carry-safe interval is allowed.",
            "torso.look_down",
        ),
        _look_spec(
            "look_center",
            "Return the physical torso and onboard head view to the validated "
            "official reset profile.",
            "torso.look_center",
        ),
    )


def torso_pitch_specs() -> tuple[ActionSpec, ...]:
    """Pitch-only Action20 actions that preserve the camera-height row."""

    return (
        _look_spec(
            "look_up",
            "Pitch the physical onboard head view upward by one bounded "
            "15-degree grid step while preserving the selected camera height.",
            "torso.look_up",
        ),
        _look_spec(
            "look_down",
            "Pitch the physical onboard head view downward by one bounded "
            "15-degree grid step while preserving the selected camera height.",
            "torso.look_down",
        ),
    )


def torso_vertical_specs() -> tuple[ActionSpec, ...]:
    """ESI-compatible names backed only by the physical R1Pro torso.

    These actions never translate the base in Z and never move a free viewer
    camera.  The fixed ZED camera follows ``torso_link4``.
    """

    return (
        _look_spec(
            "move_up",
            "Raise the onboard ZED view using bounded physical R1Pro torso "
            "articulation. The chassis never leaves the floor and no viewer "
            "camera is moved.",
            "torso.move_up",
        ),
        _look_spec(
            "move_down",
            "Lower the onboard ZED view using bounded physical R1Pro torso "
            "articulation. The chassis never leaves the floor and no viewer "
            "camera is moved.",
            "torso.move_down",
        ),
    )


def torso_action23_grid_specs() -> tuple[ActionSpec, ...]:
    """Bounded physical height and optical-pitch actions for Action23.

    The four actions share one physical 3x3 torso grid at runtime.  They are
    separate from the historical look/vertical specs so old registry hashes
    and public statuses remain unchanged.
    """

    descriptions = {
        "move_up": (
            "Raise the fixed onboard R1Pro camera by one bounded torso-height "
            "grid step without moving the base in Z or moving a viewer camera."
        ),
        "move_down": (
            "Lower the fixed onboard R1Pro camera by one bounded torso-height "
            "grid step without moving the base in Z or moving a viewer camera."
        ),
        "look_up": (
            "Pitch the fixed onboard camera upward by one bounded 15-degree "
            "grid step while preserving the selected torso height."
        ),
        "look_down": (
            "Pitch the fixed onboard camera downward by one bounded 15-degree "
            "grid step while preserving the selected torso height."
        ),
    }
    return tuple(
        ActionSpec(
            name=name,
            parse_args=parse_no_args,
            args_schema=EMPTY_ARGS_SCHEMA,
            args_required=False,
            model_description=description,
            required_capabilities=frozenset(
                {"physical_torso", "head_rgbd", "bounded_torso_grid"}
            ),
            max_simulator_steps=180,
            public_statuses=(
                "completed",
                "limit_reached",
                "unsafe_while_carrying",
                "joint_stalled",
            ),
            invariants=frozenset(
                {
                    "physical_camera_only",
                    "bounded_grid",
                    "height_pitch_decoupled",
                    "arm_safe",
                }
            ),
            controller_key=f"torso.{name}",
        )
        for name, description in descriptions.items()
    )
