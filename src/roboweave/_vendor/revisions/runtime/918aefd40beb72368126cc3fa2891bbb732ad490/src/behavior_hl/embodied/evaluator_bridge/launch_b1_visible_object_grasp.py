"""GPU-selected one-load Oracle + Qwen visible-object grasp canary."""

from behavior_hl.embodied.evaluator_bridge.launch_b1_single_load import (
    build_production_bindings,
    parse_worker_spec,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    run_b1_single_load,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_strategy_sweep import (
    make_visible_object_grasp_callback,
)


def main(argv: list[str] | None = None) -> int:
    worker = parse_worker_spec(argv)
    bindings = build_production_bindings()
    result = run_b1_single_load(
        bindings=bindings,
        worker=worker,
        strategy_sweep_callback=make_visible_object_grasp_callback(
            build_bindings=build_production_bindings,
        ),
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
