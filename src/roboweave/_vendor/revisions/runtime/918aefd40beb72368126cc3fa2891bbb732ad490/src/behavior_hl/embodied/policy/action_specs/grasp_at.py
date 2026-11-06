"""Image-point physical grasp action metadata."""

from behavior_hl.embodied.policy.protocol import (
    ActionProtocolError,
    parse_point_action_args,
)
from behavior_hl.embodied.policy.registry import ActionSpec


POINT_ARGS_SCHEMA = {
    "type": "object",
    "properties": {
        "camera": {
            "type": "string",
            "enum": ["head", "left_wrist", "right_wrist"],
        },
        "point": {
            "type": "array",
            "prefixItems": [
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
            ],
            "minItems": 2,
            "maxItems": 2,
        },
        "hand": {"type": "string", "enum": ["left", "right"]},
    },
    "required": ["camera", "point", "hand"],
    "additionalProperties": False,
}


def grasp_at_spec() -> ActionSpec:
    return ActionSpec(
        name="grasp_at",
        parse_args=parse_point_action_args,
        args_schema=POINT_ARGS_SCHEMA,
        args_required=True,
        model_description=(
            "Attempt a physical grasp at a normalized point in the current "
            "onboard image. It never moves the base; invalid depth, reach, IK, "
            "or contact can make it fail."
        ),
        required_capabilities=frozenset(
            {"rgbd_point", "arm_ik", "physical_gripper"}
        ),
        max_simulator_steps=900,
        public_statuses=(
            "completed",
            "invalid_point",
            "invalid_depth",
            "out_of_reach",
            "ik_failed",
            "grasp_likely",
            "grasp_uncertain",
        ),
        invariants=frozenset(
            {"no_base_motion", "sensor_target_only", "safe_retract"}
        ),
        controller_key="manipulation.grasp_at",
    )


def visible_official_grasp_at_spec() -> ActionSpec:
    """Opt-in head-image grasp backed by BEHAVIOR's Starter primitive."""

    schema = dict(POINT_ARGS_SCHEMA)
    properties = dict(schema["properties"])
    properties["camera"] = {"type": "string", "enum": ["head"]}
    schema["properties"] = properties
    def parse_head_point(payload):
        args = parse_point_action_args(payload)
        if args.camera != "head":
            raise ActionProtocolError(
                "visible official grasp currently accepts head camera only"
            )
        return args

    return ActionSpec(
        name="grasp_at",
        parse_args=parse_head_point,
        args_schema=schema,
        args_required=True,
        model_description=(
            "Attempt to grasp the task object under a normalized point in "
            "the current head-camera image. The object must be visibly under "
            "the selected point. The robot may make only a small, bounded "
            "local base alignment; use move_* and turn_* for navigation. "
            "Judge success from the following onboard image, not text."
        ),
        required_capabilities=frozenset(
            {
                "current_frame_visible_target",
                "official_starter_grasp",
                "bounded_base_alignment",
                "physical_gripper",
            }
        ),
        max_simulator_steps=1800,
        public_statuses=("grasp_likely", "grasp_uncertain"),
        invariants=frozenset(
            {
                "head_sensor_target_only",
                "private_visibility_gate",
                "bounded_base_alignment",
                "no_textual_failure_truth",
                "no_teleport",
            }
        ),
        controller_key="manipulation.visible_official_grasp_at",
    )
