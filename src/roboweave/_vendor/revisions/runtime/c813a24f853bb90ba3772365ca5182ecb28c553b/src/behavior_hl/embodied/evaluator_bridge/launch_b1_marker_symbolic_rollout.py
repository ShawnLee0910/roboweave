"""CLI for the isolated one-load marker Symbolic Oracle -> Qwen canary."""

from behavior_hl.embodied.evaluator_bridge.launch_b1_single_load import (
    build_production_bindings,
    parse_worker_spec,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
    make_marker_symbolic_rollout_callback,
)
from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (
    get_inside_task_profile,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    run_b1_single_load,
)


def main(argv=None) -> int:
    worker = parse_worker_spec(argv)
    profile = get_inside_task_profile(worker.task_name)
    result = run_b1_single_load(
        bindings=build_production_bindings(),
        worker=worker,
        strategy_sweep_callback=make_marker_symbolic_rollout_callback(
            seed=worker.seed,
            profile=profile,
        ),
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
