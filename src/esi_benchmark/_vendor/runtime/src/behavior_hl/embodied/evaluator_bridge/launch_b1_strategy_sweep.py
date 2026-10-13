"""GPU-selected one-scene Oracle/Qwen strategy sweep launcher."""

from __future__ import annotations

from pathlib import Path

from behavior_hl.embodied.evaluator_bridge.launch_b1_single_load import (
    MODEL_PYTHON,
    build_production_bindings,
    parse_worker_spec,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_live_entrypoint import (
    R1ProB1LiveEntrypointError,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    run_b1_single_load,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_strategy_sweep import (
    make_strategy_sweep_callback,
)


def main(argv: list[str] | None = None) -> int:
    worker = parse_worker_spec(argv)
    bindings = build_production_bindings()
    callback = make_strategy_sweep_callback(
        build_bindings=build_production_bindings,
    )
    result = run_b1_single_load(
        bindings=bindings,
        worker=worker,
        strategy_sweep_callback=callback,
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

