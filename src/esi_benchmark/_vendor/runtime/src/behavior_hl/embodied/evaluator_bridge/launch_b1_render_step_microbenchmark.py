"""Launch the isolated one-scene render-step microbenchmark."""

from __future__ import annotations

from behavior_hl.embodied.evaluator_bridge.launch_b1_single_load import (
    build_production_bindings,
    parse_worker_spec,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_render_step_microbenchmark import (
    make_render_step_microbenchmark_callback,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    run_b1_single_load,
)


def main(argv: list[str] | None = None) -> int:
    worker = parse_worker_spec(argv)
    result = run_b1_single_load(
        bindings=build_production_bindings(),
        worker=worker,
        strategy_sweep_callback=make_render_step_microbenchmark_callback(),
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
