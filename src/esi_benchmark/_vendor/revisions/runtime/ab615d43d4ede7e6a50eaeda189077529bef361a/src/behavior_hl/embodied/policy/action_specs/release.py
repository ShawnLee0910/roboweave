"""Physical gripper release action metadata."""

from behavior_hl.embodied.policy.protocol import parse_release_args
from behavior_hl.embodied.policy.registry import ActionSpec


RELEASE_ARGS_SCHEMA = {
    "type": "object",
    "properties": {
        "hand": {"type": "string", "enum": ["left", "right"]},
    },
    "required": ["hand"],
    "additionalProperties": False,
}


def release_spec() -> ActionSpec:
    return ActionSpec(
        name="release",
        parse_args=parse_release_args,
        args_schema=RELEASE_ARGS_SCHEMA,
        args_required=True,
        model_description=(
            "Open one physical gripper at its current pose. This does not move "
            "or automatically place the released item."
        ),
        required_capabilities=frozenset({"physical_gripper"}),
        max_simulator_steps=120,
        public_statuses=("completed", "joint_stalled"),
        invariants=frozenset({"no_base_motion", "physical_release"}),
        controller_key="manipulation.release",
    )
