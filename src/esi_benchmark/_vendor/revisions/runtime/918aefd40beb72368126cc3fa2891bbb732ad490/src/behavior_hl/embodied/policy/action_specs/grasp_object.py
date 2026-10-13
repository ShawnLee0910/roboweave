"""Category-level physical grasp action metadata."""

from behavior_hl.embodied.policy.protocol import (
    VISIBLE_OBJECT_CATEGORIES,
    VISIBLE_OBJECT_SELECTORS,
    parse_visible_object_grasp_args,
)
from behavior_hl.embodied.policy.registry import ActionSpec


VISIBLE_OBJECT_ARGS_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {
            "type": "string",
            "enum": sorted(VISIBLE_OBJECT_CATEGORIES),
        },
        "selector": {
            "type": "string",
            "enum": sorted(VISIBLE_OBJECT_SELECTORS),
        },
    },
    "required": ["object"],
    "additionalProperties": False,
}


def visible_object_official_grasp_spec() -> ActionSpec:
    """Current-head visible category backed by official physical GRASP."""

    return ActionSpec(
        name="grasp",
        parse_args=parse_visible_object_grasp_args,
        args_schema=VISIBLE_OBJECT_ARGS_SCHEMA,
        args_required=True,
        model_description=(
            "Physically grasp one currently visible task object by category. "
            "If exactly one matching instance is visible, give only object. "
            "If several are visible, also give one image-relative selector: "
            "leftmost, rightmost, topmost, bottommost, or center. The robot "
            "may make a small bounded local base alignment, but navigation "
            "still requires move_* and turn_*. Judge success only from the "
            "following onboard images."
        ),
        required_capabilities=frozenset(
            {
                "current_frame_instance_visibility",
                "official_starter_grasp",
                "bounded_base_alignment",
                "physical_gripper",
            }
        ),
        max_simulator_steps=1800,
        public_statuses=("grasp_likely", "grasp_uncertain"),
        invariants=frozenset(
            {
                "head_sensor_category_only",
                "private_instance_binding",
                "bounded_base_alignment",
                "no_textual_failure_truth",
                "no_teleport",
            }
        ),
        controller_key="manipulation.visible_object_official_grasp",
    )
