"""Local physical-base movement and turning action metadata."""

from behavior_hl.embodied.policy.protocol import parse_no_args
from behavior_hl.embodied.policy.registry import ActionSpec


EMPTY_ARGS_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def _movement_spec(
    name: str,
    description: str,
    controller_key: str,
) -> ActionSpec:
    return ActionSpec(
        name=name,
        parse_args=parse_no_args,
        args_schema=EMPTY_ARGS_SCHEMA,
        args_required=False,
        model_description=description,
        required_capabilities=frozenset({"holonomic_base", "base_qvel"}),
        max_simulator_steps=120,
        public_statuses=("completed", "motion_partial", "blocked"),
        invariants=frozenset(
            {"planar_only", "local_motion", "sensor_odometry_only"}
        ),
        controller_key=controller_key,
    )


def base_motion_specs() -> tuple[ActionSpec, ...]:
    return (
        _movement_spec(
            "move_forward",
            "Move the physical base forward by up to 0.25 m. Front depth may "
            "slow or stop the action, and real collisions can block it.",
            "base.move_forward",
        ),
        _movement_spec(
            "move_backward",
            "Move the physical base backward by up to 0.20 m. The backward "
            "direction has no guaranteed depth coverage; execution uses a "
            "small step and velocity-stall detection.",
            "base.move_backward",
        ),
        _movement_spec(
            "move_left",
            "Translate the physical base left by up to 0.20 m. The leftward "
            "direction has no guaranteed depth coverage; execution uses a "
            "small step and velocity-stall detection.",
            "base.move_left",
        ),
        _movement_spec(
            "move_right",
            "Translate the physical base right by up to 0.20 m. The rightward "
            "direction has no guaranteed depth coverage; execution uses a "
            "small step and velocity-stall detection.",
            "base.move_right",
        ),
        _movement_spec(
            "turn_left",
            "Rotate the physical base left by up to 15 degrees. Real contact "
            "may stop the requested rotation.",
            "base.turn_left",
        ),
        _movement_spec(
            "turn_right",
            "Rotate the physical base right by up to 15 degrees. Real contact "
            "may stop the requested rotation.",
            "base.turn_right",
        ),
    )
