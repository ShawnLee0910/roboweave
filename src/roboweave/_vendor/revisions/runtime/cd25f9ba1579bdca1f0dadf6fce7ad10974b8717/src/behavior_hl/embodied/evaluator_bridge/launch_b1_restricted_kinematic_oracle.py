"""Launch one isolated G-1 -> G0 -> G1 restricted-kinematic Oracle."""

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
            run_model_after_oracle=False,
            base_execution="restricted_kinematic",
        ),
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

