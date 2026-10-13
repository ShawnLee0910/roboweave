"""Launch one isolated resident G-1 -> G0 -> G1 -> model campaign."""

import os

from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (
    get_inside_task_profile,
)
from behavior_hl.embodied.evaluator_bridge.launch_b1_marker_symbolic_rollout import (
    parse_worker_spec,
)
from behavior_hl.embodied.evaluator_bridge.launch_b1_single_load import (
    build_production_bindings,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
    make_marker_symbolic_full_rollout_callback,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    run_b1_single_load,
)


def main(argv: list[str] | None = None) -> int:
    worker = parse_worker_spec(argv)
    action_profile = os.environ.get(
        "BEHAVIOR_HL_MODEL_ACTION_PROFILE",
        "marker_symbolic_v34",
    )
    if action_profile not in {
        "marker_symbolic_v34",
        "marker_symbolic_action23",
        "marker_symbolic_action20",
    }:
        raise ValueError("BEHAVIOR_HL_MODEL_ACTION_PROFILE is invalid")
    maximum_model_calls = int(
        os.environ.get("BEHAVIOR_HL_MAXIMUM_MODEL_CALLS", "128")
    )
    if maximum_model_calls <= 0:
        raise ValueError("BEHAVIOR_HL_MAXIMUM_MODEL_CALLS must be positive")
    profile = get_inside_task_profile(worker.task_name)
    if profile.split != worker.split or profile.instance_id != worker.instance_id:
        raise ValueError(
            "worker split/instance does not match the frozen task profile"
        )
    result = run_b1_single_load(
        bindings=build_production_bindings(),
        worker=worker,
        strategy_sweep_callback=make_marker_symbolic_full_rollout_callback(
            seed=worker.seed,
            profile=profile,
            maximum_model_calls=maximum_model_calls,
            run_model_after_oracle=True,
            base_execution="restricted_kinematic",
            model_action_profile=action_profile,
        ),
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
