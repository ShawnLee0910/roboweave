"""Fixed-duration public wait action."""

from behavior_hl.embodied.policy.action_specs.base_motion import (
    EMPTY_ARGS_SCHEMA,
)
from behavior_hl.embodied.policy.protocol import parse_no_args
from behavior_hl.embodied.policy.registry import ActionSpec


WAIT_SIMULATOR_STEPS = 30


def wait_spec() -> ActionSpec:
    """Advance exactly 30 simulator steps without setting task state."""

    return ActionSpec(
        name="wait",
        parse_args=parse_no_args,
        args_schema=EMPTY_ARGS_SCHEMA,
        args_required=False,
        model_description=(
            "Wait for exactly 30 simulator steps (one fixed simulated "
            "second) while keeping the robot stationary. This action accepts "
            "no duration argument and does not set any task predicate."
        ),
        required_capabilities=frozenset(
            {"physical_robot_hold", "fixed_duration_simulation"}
        ),
        max_simulator_steps=WAIT_SIMULATOR_STEPS,
        public_statuses=("completed", "unsafe_while_carrying", "aborted"),
        invariants=frozenset(
            {
                "exactly_30_simulator_steps",
                "zero_base_command",
                "no_task_state_setter",
                "fresh_observation",
            }
        ),
        controller_key="temporal.wait",
    )
