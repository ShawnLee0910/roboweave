"""Image-point physical placement action metadata."""

from behavior_hl.embodied.policy.action_specs.grasp_at import (
    POINT_ARGS_SCHEMA,
)
from behavior_hl.embodied.policy.protocol import (
    ActionProtocolError,
    parse_point_action_args,
)
from behavior_hl.embodied.policy.registry import ActionSpec


def place_inside_at_spec() -> ActionSpec:
    return ActionSpec(
        name="place_inside_at",
        parse_args=parse_point_action_args,
        args_schema=POINT_ARGS_SCHEMA,
        args_required=True,
        model_description=(
            "Attempt a physical release above a visually selected point inside "
            "a currently visible container opening. It never moves the base "
            "and does not guarantee containment."
        ),
        required_capabilities=frozenset(
            {"rgbd_point", "arm_ik", "physical_gripper", "cavity_geometry"}
        ),
        max_simulator_steps=1_200,
        public_statuses=(
            "completed",
            "invalid_point",
            "invalid_depth",
            "out_of_reach",
            "ik_failed",
            "cavity_uncertain",
            "place_unstable",
        ),
        invariants=frozenset(
            {"no_base_motion", "sensor_target_only", "physical_release", "safe_retract"}
        ),
        controller_key="manipulation.place_inside_at",
    )


def visible_official_place_inside_at_spec() -> ActionSpec:
    """Opt-in head-image placement backed by official PLACE_INSIDE."""

    schema = dict(POINT_ARGS_SCHEMA)
    properties = dict(schema["properties"])
    properties["camera"] = {"type": "string", "enum": ["head"]}
    schema["properties"] = properties

    def parse_head_point(payload):
        args = parse_point_action_args(payload)
        if args.camera != "head":
            raise ActionProtocolError(
                "visible official place currently accepts head camera only"
            )
        return args

    return ActionSpec(
        name="place_inside_at",
        parse_args=parse_head_point,
        args_schema=schema,
        args_required=True,
        model_description=(
            "Attempt to place the carried object inside the container under "
            "a normalized point in the current head-camera image. Select a "
            "visible part of the container such as its rim or body. The "
            "robot may make only a small, bounded local base alignment; use "
            "move_* and turn_* for navigation. Judge success from the next "
            "onboard image, not text."
        ),
        required_capabilities=frozenset(
            {
                "current_frame_visible_container",
                "official_starter_place_inside",
                "bounded_base_alignment",
                "physical_gripper",
            }
        ),
        max_simulator_steps=1800,
        public_statuses=("completed", "place_uncertain", "place_noop"),
        invariants=frozenset(
            {
                "head_sensor_target_only",
                "private_visibility_gate",
                "bounded_base_alignment",
                "no_textual_failure_truth",
                "no_teleport",
                "private_inside_verification",
            }
        ),
        controller_key="manipulation.visible_official_place_inside_at",
    )
