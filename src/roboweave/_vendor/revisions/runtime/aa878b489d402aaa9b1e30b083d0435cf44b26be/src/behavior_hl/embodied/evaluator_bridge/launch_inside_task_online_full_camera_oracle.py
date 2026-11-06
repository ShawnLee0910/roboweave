"""Launch one registered Inside task with the full R1Pro camera profile.

This is an explicit visual-evidence lane for the existing online-A* Symbolic
Oracle.  It deliberately leaves the stable head-only launcher unchanged while
selecting the already-supported full camera contract, whose action-boundary
composite is ordered ``left_wrist | right_wrist | head``.
"""

from __future__ import annotations

from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (
    get_inside_task_profile,
)
from behavior_hl.embodied.evaluator_bridge.launch_b1_single_load import (
    build_production_bindings,
    parse_worker_spec,
)
from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    FULL_CAMERA_PROFILE,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
    make_actionized_minimal_full_oracle_callback,
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
        strategy_sweep_callback=make_actionized_minimal_full_oracle_callback(
            seed=worker.seed,
            profile=profile,
            observation_profile=FULL_CAMERA_PROFILE,
        ),
        camera_profile=FULL_CAMERA_PROFILE,
        expected_task_id=profile.task_id,
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
