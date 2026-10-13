"""Launch the private four-boundary BEHAVIOR 2026 Oracle lane."""

import json
import os
from pathlib import Path

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
    make_actionized_minimal_full_oracle_callback,
    validate_precompiled_minimal_oracle_inputs,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    run_b1_single_load,
)
from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    HEAD_RGB_MINIMAL_CAMERA_PROFILE,
)


def main(argv: list[str] | None = None) -> int:
    worker = parse_worker_spec(argv)
    profile = get_inside_task_profile(worker.task_name)
    if profile.split != worker.split or profile.instance_id != worker.instance_id:
        raise ValueError(
            "worker split/instance does not match the frozen task profile"
        )
    route_value = os.environ.get("BEHAVIOR_HL_PRECOMPILED_ORACLE_ROUTE")
    viewpoints_value = os.environ.get("BEHAVIOR_HL_PRECOMPUTED_VIEWPOINTS")
    if bool(route_value) != bool(viewpoints_value):
        raise ValueError(
            "precompiled route and viewpoints environment paths must be paired"
        )
    precompiled_route = None
    precomputed_viewpoints = None
    if route_value and viewpoints_value:
        route_path = Path(route_value).resolve(strict=True)
        viewpoints_path = Path(viewpoints_value).resolve(strict=True)
        precompiled_route = json.loads(route_path.read_text(encoding="utf-8"))
        precomputed_viewpoints = json.loads(
            viewpoints_path.read_text(encoding="utf-8")
        )
        validate_precompiled_minimal_oracle_inputs(
            route=precompiled_route,
            viewpoints=precomputed_viewpoints,
        )
    if precompiled_route is None or precomputed_viewpoints is None:
        raise ValueError("v103 direct full Oracle requires frozen route and viewpoints")
    semantic_only_value = os.environ.get(
        "BEHAVIOR_HL_SEMANTIC_ONLY_PLACE_INSIDE", "0"
    )
    if semantic_only_value not in {"0", "1"}:
        raise ValueError(
            "BEHAVIOR_HL_SEMANTIC_ONLY_PLACE_INSIDE must be 0 or 1"
        )
    semantic_only_place_inside = semantic_only_value == "1"
    result = run_b1_single_load(
        bindings=build_production_bindings(),
        worker=worker,
        strategy_sweep_callback=make_actionized_minimal_full_oracle_callback(
            seed=worker.seed,
            profile=profile,
            precompiled_route=precompiled_route,
            precomputed_viewpoints=precomputed_viewpoints,
            observation_profile=HEAD_RGB_MINIMAL_CAMERA_PROFILE,
            semantic_only_place_inside=semantic_only_place_inside,
        ),
        camera_profile=HEAD_RGB_MINIMAL_CAMERA_PROFILE,
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
