"""One-scene benchmark for advancing more 120 Hz physics per checkpoint.

This is a research-only high-level executor experiment.  It uses the official
``Simulator.set_simulation_dt`` API, keeps the physics timestep unchanged, and
restores every timing value even when a lane raises.  Each measured step still
travels through ``Evaluator.step``; only the observation/checkpoint frequency
changes.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, ContextManager, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.run_b1_render_step_microbenchmark import (
    _run_lane,
)


TIMESTEP_MICROBENCHMARK_SCHEMA = "behavior2026.timestep-microbenchmark/v1"


@contextmanager
def simulation_checkpoint_frequency(simulator: Any, frequency_hz: int):
    """Set checkpoint frequency while preserving 120 Hz physics and restore."""

    if (
        isinstance(frequency_hz, bool)
        or not isinstance(frequency_hz, int)
        or frequency_hz <= 0
    ):
        raise ValueError("frequency_hz must be a positive integer")
    physics_dt = float(simulator.get_physics_dt())
    rendering_dt = float(simulator.get_rendering_dt())
    sim_step_dt = float(simulator.get_sim_step_dt())
    requested_dt = 1.0 / frequency_hz
    ratio = requested_dt / physics_dt
    if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
        raise ValueError("checkpoint dt must be an integer multiple of physics dt")
    simulator.set_simulation_dt(
        physics_dt=physics_dt,
        rendering_dt=requested_dt,
        sim_step_dt=requested_dt,
    )
    try:
        yield
    finally:
        simulator.set_simulation_dt(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            sim_step_dt=sim_step_dt,
        )


def _base_pose(evaluator: Any) -> tuple[np.ndarray, np.ndarray]:
    robot = getattr(evaluator, "robot", None)
    callback = getattr(robot, "get_position_orientation", None)
    if not callable(callback):
        raise RuntimeError("Evaluator robot pose callback is unavailable")
    position, quaternion = callback()
    position_array = _host_array(position)
    quaternion_array = _host_array(quaternion)
    if position_array.size != 3 or quaternion_array.size != 4:
        raise RuntimeError("Evaluator robot pose has an invalid shape")
    return position_array, quaternion_array


def _host_array(value: Any) -> np.ndarray:
    """Convert NumPy or CPU/CUDA torch-like values without importing torch."""

    item = value
    for method_name in ("detach", "cpu"):
        method = getattr(item, method_name, None)
        if callable(method):
            item = method()
    numpy_method = getattr(item, "numpy", None)
    if callable(numpy_method):
        item = numpy_method()
    return np.asarray(item, dtype=np.float64).reshape(-1)


def _pose_drift(
    before: tuple[np.ndarray, np.ndarray],
    after: tuple[np.ndarray, np.ndarray],
) -> Mapping[str, float]:
    before_position, before_quaternion = before
    after_position, after_quaternion = after
    quaternion_alignment = float(
        abs(np.dot(before_quaternion, after_quaternion))
        / (
            np.linalg.norm(before_quaternion)
            * np.linalg.norm(after_quaternion)
        )
    )
    quaternion_alignment = min(1.0, max(-1.0, quaternion_alignment))
    return {
        "translation_m": float(
            np.linalg.norm(after_position - before_position)
        ),
        "z_m": float(after_position[2] - before_position[2]),
        "orientation_deg": float(
            math.degrees(2.0 * math.acos(quaternion_alignment))
        ),
    }


def run_timestep_microbenchmark(
    *,
    evaluator: Any,
    control: Any,
    simulator: Any,
    render_context: Callable[[bool], ContextManager[Any]],
    frequencies_hz: tuple[int, ...] = (30, 10, 5, 2),
    steps_per_lane: int = 20,
) -> Mapping[str, Any]:
    if (
        isinstance(steps_per_lane, bool)
        or not isinstance(steps_per_lane, int)
        or steps_per_lane < 2
    ):
        raise ValueError("steps_per_lane must be an integer >= 2")
    if not frequencies_hz or len(set(frequencies_hz)) != len(frequencies_hz):
        raise ValueError("frequencies_hz must be non-empty and unique")
    if 30 not in frequencies_hz:
        raise ValueError("frequencies_hz must contain the 30 Hz baseline")

    # Pay one first-frame/video-writer initialization cost outside all lanes.
    control.hold()
    _run_lane(
        evaluator=evaluator,
        render_context=render_context,
        schedule=((True, True),),
        lane="unmeasured_render_video_prewarm",
    )

    physics_dt = float(simulator.get_physics_dt())
    original_rendering_dt = float(simulator.get_rendering_dt())
    original_sim_step_dt = float(simulator.get_sim_step_dt())
    lanes = []
    for frequency_hz in frequencies_hz:
        control.hold()
        before = _base_pose(evaluator)
        with simulation_checkpoint_frequency(simulator, frequency_hz):
            lane = dict(
                _run_lane(
                    evaluator=evaluator,
                    render_context=render_context,
                    schedule=(
                        ((False, False),) * (steps_per_lane - 1)
                        + ((True, True),)
                    ),
                    lane=f"checkpoint_{frequency_hz}hz",
                )
            )
        after = _base_pose(evaluator)
        simulated_seconds = steps_per_lane / frequency_hz
        lane.update(
            {
                "checkpoint_frequency_hz": frequency_hz,
                "physics_dt_seconds": physics_dt,
                "physics_substeps_per_checkpoint": int(
                    round((1.0 / frequency_hz) / physics_dt)
                ),
                "simulated_seconds": simulated_seconds,
                "simulated_seconds_per_wall_second": (
                    simulated_seconds / float(lane["wall_seconds"])
                ),
                "hold_base_pose_drift": _pose_drift(before, after),
            }
        )
        lanes.append(lane)

    if not math.isclose(simulator.get_physics_dt(), physics_dt):
        raise RuntimeError("physics dt was not restored")
    if not math.isclose(simulator.get_rendering_dt(), original_rendering_dt):
        raise RuntimeError("rendering dt was not restored")
    if not math.isclose(simulator.get_sim_step_dt(), original_sim_step_dt):
        raise RuntimeError("simulation step dt was not restored")

    baseline = next(
        lane for lane in lanes if lane["checkpoint_frequency_hz"] == 30
    )
    baseline_throughput = float(
        baseline["simulated_seconds_per_wall_second"]
    )
    for lane in lanes:
        lane["physical_time_throughput_speedup_vs_30hz"] = (
            float(lane["simulated_seconds_per_wall_second"])
            / baseline_throughput
        )
    return {
        "schema_version": TIMESTEP_MICROBENCHMARK_SCHEMA,
        "official_evaluator_step_preserved": True,
        "physics_dt_preserved": True,
        "physics_collisions_preserved": True,
        "checkpoint_frequency_modified": True,
        "official_challenge_eligible": False,
        "kinematic_research": True,
        "model_lane_attempted": False,
        "teleport_enabled": False,
        "viewer_camera_control": False,
        "privileged_policy_inputs": False,
        "scene_loads_attempted": 1,
        "successful_scene_loads": 1,
        "steps_per_lane": steps_per_lane,
        "unmeasured_prewarm_steps": 1,
        "total_simulator_checkpoints": (
            1 + steps_per_lane * len(frequencies_hz)
        ),
        "original_timing": {
            "physics_dt_seconds": physics_dt,
            "rendering_dt_seconds": original_rendering_dt,
            "sim_step_dt_seconds": original_sim_step_dt,
        },
        "strategies": lanes,
        "lanes": lanes,
    }


def make_timestep_microbenchmark_callback(
    *,
    frequencies_hz: tuple[int, ...] = (30, 10, 5, 2),
    steps_per_lane: int = 20,
):
    def callback(*, evaluator, control, bindings, run_dir, stages):
        del bindings, run_dir, stages
        import omnigibson as og

        return run_timestep_microbenchmark(
            evaluator=evaluator,
            control=control,
            simulator=og.sim,
            render_context=og.sim.render_on_step,
            frequencies_hz=frequencies_hz,
            steps_per_lane=steps_per_lane,
        )

    return callback
