"""Run a preconfigured-frequency hold canary, Oracle, then Qwen in one scene."""

from __future__ import annotations

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
from behavior_hl.embodied.evaluator_bridge.run_b1_render_step_microbenchmark import (
    _run_lane,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    run_b1_single_load,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_timestep_microbenchmark import (
    _base_pose,
    _pose_drift,
)
from behavior_hl.embodied.evaluator_bridge.run_g1_canary import _record_stage


def _wrap_with_hold_canary(*, worker, profile):
    full_callback = make_marker_symbolic_full_rollout_callback(
        seed=worker.seed,
        profile=profile,
    )

    def callback(*, evaluator, control, bindings, run_dir, stages):
        import omnigibson as og

        expected_frequency_hz = int(
            os.environ["BEHAVIOR_HL_SIM_CHECKPOINT_FREQUENCY_HZ"]
        )
        actual_frequency_hz = round(1.0 / og.sim.get_sim_step_dt())
        if actual_frequency_hz != expected_frequency_hz:
            raise RuntimeError(
                "live simulator checkpoint frequency does not match preflight: "
                f"expected={expected_frequency_hz}, actual={actual_frequency_hz}, "
                f"sim_step_dt={og.sim.get_sim_step_dt()}"
            )
        physics_frequency_hz = round(1.0 / og.sim.get_physics_dt())
        _record_stage(
            stages,
            "PRECONFIGURED_FREQUENCY_HOLD_CANARY_START",
            checkpoint_frequency_hz=actual_frequency_hz,
            physics_frequency_hz=physics_frequency_hz,
        )
        control.hold()
        before = _base_pose(evaluator)
        lane = dict(
            _run_lane(
                evaluator=evaluator,
                render_context=og.sim.render_on_step,
                schedule=((False, False),) * 19 + ((True, True),),
                lane="preconfigured_frequency_hold_canary",
            )
        )
        after = _base_pose(evaluator)
        drift = dict(_pose_drift(before, after))
        lane.update(
            {
                "checkpoint_frequency_hz": actual_frequency_hz,
                "physics_frequency_hz": physics_frequency_hz,
                "physics_substeps_per_checkpoint": (
                    physics_frequency_hz // actual_frequency_hz
                ),
                "simulated_seconds": 20.0 / actual_frequency_hz,
                "hold_base_pose_drift": drift,
            }
        )
        if (
            drift["translation_m"] > 0.01
            or abs(drift["z_m"]) > 0.005
            or drift["orientation_deg"] > 0.5
        ):
            raise RuntimeError(
                f"preconfigured frequency hold canary drifted: {drift}"
            )
        _record_stage(
            stages,
            "PRECONFIGURED_FREQUENCY_HOLD_CANARY_PASS",
            **lane,
        )

        # Restore the exact task-instance start in memory; this is not a scene
        # reconstruction and preserves the one-load budget.
        control.hold()
        evaluator.reset()
        _record_stage(
            stages,
            "PRECONFIGURED_FREQUENCY_POST_CANARY_RESET_PASS",
        )
        result = dict(
            full_callback(
                evaluator=evaluator,
                control=control,
                bindings=bindings,
                run_dir=run_dir,
                stages=stages,
            )
        )
        result["schema_version"] = (
            "behavior2026.preconfigured-frequency-full-oracle-qwen/v43"
        )
        result["preconfigured_frequency_hold_canary"] = lane
        result["checkpoint_frequency_hz"] = actual_frequency_hz
        result["physics_frequency_hz"] = physics_frequency_hz
        result["physics_substeps_per_checkpoint"] = (
            physics_frequency_hz // actual_frequency_hz
        )
        result["sparse_internal_rendering"] = True
        result["in_memory_resets"] = int(result.get("in_memory_resets", 0)) + 1
        result["strategies"] = [lane, *list(result.get("strategies", ())) ]
        return result

    return callback


def main(argv=None) -> int:
    worker = parse_worker_spec(argv)
    profile = get_inside_task_profile(worker.task_name)
    if profile.split != worker.split or profile.instance_id != worker.instance_id:
        raise ValueError("worker split/instance does not match task profile")
    result = run_b1_single_load(
        bindings=build_production_bindings(),
        worker=worker,
        strategy_sweep_callback=_wrap_with_hold_canary(
            worker=worker,
            profile=profile,
        ),
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
