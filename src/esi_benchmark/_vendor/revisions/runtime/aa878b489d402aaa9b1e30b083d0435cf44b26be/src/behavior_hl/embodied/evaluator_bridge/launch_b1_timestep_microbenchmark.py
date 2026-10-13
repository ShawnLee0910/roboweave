"""Launch the isolated one-scene timestep microbenchmark."""

from behavior_hl.embodied.evaluator_bridge.launch_b1_single_load import (
    build_production_bindings,
    parse_worker_spec,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    run_b1_single_load,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_timestep_microbenchmark import (
    make_timestep_microbenchmark_callback,
)


def main(argv=None) -> int:
    worker = parse_worker_spec(argv)
    result = run_b1_single_load(
        bindings=build_production_bindings(),
        worker=worker,
        strategy_sweep_callback=make_timestep_microbenchmark_callback(),
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
