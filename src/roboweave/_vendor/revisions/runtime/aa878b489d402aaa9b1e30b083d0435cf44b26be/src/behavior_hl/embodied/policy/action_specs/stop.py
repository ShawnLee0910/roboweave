"""Episode stop action metadata."""

from behavior_hl.embodied.policy.action_specs.base_motion import (
    EMPTY_ARGS_SCHEMA,
)
from behavior_hl.embodied.policy.protocol import parse_no_args
from behavior_hl.embodied.policy.registry import ActionSpec


def stop_spec() -> ActionSpec:
    return ActionSpec(
        name="stop",
        parse_args=parse_no_args,
        args_schema=EMPTY_ARGS_SCHEMA,
        args_required=False,
        model_description=(
            "Declare the task finished. If the private task goal is not yet "
            "satisfied, the episode ends as a failure."
        ),
        required_capabilities=frozenset({"episode_control"}),
        max_simulator_steps=1,
        public_statuses=("completed",),
        invariants=frozenset({"no_motion"}),
        controller_key="episode.stop",
    )
