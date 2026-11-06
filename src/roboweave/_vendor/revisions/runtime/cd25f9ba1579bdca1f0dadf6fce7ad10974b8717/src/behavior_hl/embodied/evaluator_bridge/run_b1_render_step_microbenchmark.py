"""One-scene wall-clock benchmark for supported sparse-render stepping.

The benchmark deliberately keeps the official ``Evaluator.step`` path.  It
only varies OmniGibson's supported ``Simulator.render_on_step`` context and
whether the evaluator writes a video frame for an internal physics step.
Physics, policy calls, task stepping, metrics, termination, and timeout
accounting therefore remain active in every lane.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from typing import Any, Callable, ContextManager, Mapping


RENDER_STEP_MICROBENCHMARK_SCHEMA = (
    "behavior2026.render-step-microbenchmark/v1"
)


def _require_nonterminal_step(evaluator: Any) -> None:
    outcome = evaluator.step()
    if (
        not isinstance(outcome, tuple)
        or len(outcome) != 2
        or not all(isinstance(value, bool) for value in outcome)
    ):
        raise RuntimeError(
            "Evaluator.step must return two native booleans during benchmark"
        )
    terminated, truncated = outcome
    if terminated or truncated:
        reason = "terminated" if terminated else "truncated"
        raise RuntimeError(f"benchmark evaluator unexpectedly {reason}")


@contextmanager
def _video_write_enabled(evaluator: Any, enabled: bool):
    """Temporarily suppress duplicate internal video frames, then restore.

    This does not close or replace the official writer.  The runner still owns
    start/stop recording.  Sparse checkpoints execute with the original path
    restored, so their rendered frame is written normally.
    """

    if enabled:
        yield
        return
    if not hasattr(evaluator, "_video_path"):
        raise RuntimeError("Evaluator video-path contract changed")
    original_path = evaluator._video_path
    evaluator._video_path = None
    try:
        yield
    finally:
        evaluator._video_path = original_path


def _run_lane(
    *,
    evaluator: Any,
    render_context: Callable[[bool], ContextManager[Any]],
    schedule: tuple[tuple[bool, bool], ...],
    lane: str,
) -> Mapping[str, Any]:
    durations: list[float] = []
    for render, write_video in schedule:
        started = time.perf_counter()
        with render_context(render):
            with _video_write_enabled(evaluator, write_video):
                _require_nonterminal_step(evaluator)
        durations.append(time.perf_counter() - started)
    total = sum(durations)
    return {
        "lane": lane,
        "simulator_steps": len(schedule),
        "rendered_steps": sum(render for render, _ in schedule),
        "video_written_steps": sum(write for _, write in schedule),
        "wall_seconds": total,
        "mean_step_seconds": statistics.fmean(durations),
        "median_step_seconds": statistics.median(durations),
        "min_step_seconds": min(durations),
        "max_step_seconds": max(durations),
        "per_step_seconds": durations,
    }


def run_render_step_microbenchmark(
    *,
    evaluator: Any,
    control: Any,
    render_context: Callable[[bool], ContextManager[Any]],
    steps_per_lane: int = 20,
) -> Mapping[str, Any]:
    """Measure full, render-disabled, and sparse-render official stepping."""

    if (
        isinstance(steps_per_lane, bool)
        or not isinstance(steps_per_lane, int)
        or steps_per_lane < 2
    ):
        raise ValueError("steps_per_lane must be an integer >= 2")
    if not callable(getattr(control, "hold", None)):
        raise TypeError("control must provide hold()")
    if not callable(render_context):
        raise TypeError("render_context must be callable")

    control.hold()
    full = _run_lane(
        evaluator=evaluator,
        render_context=render_context,
        schedule=((True, True),) * steps_per_lane,
        lane="full_render_and_video",
    )
    control.hold()
    disabled = _run_lane(
        evaluator=evaluator,
        render_context=render_context,
        schedule=((False, False),) * steps_per_lane,
        lane="render_and_video_disabled",
    )
    control.hold()
    sparse = _run_lane(
        evaluator=evaluator,
        render_context=render_context,
        schedule=(
            ((False, False),) * (steps_per_lane - 1)
            + ((True, True),)
        ),
        lane="sparse_render_final_checkpoint",
    )

    baseline_seconds = float(full["wall_seconds"])
    disabled_seconds = float(disabled["wall_seconds"])
    sparse_seconds = float(sparse["wall_seconds"])
    return {
        "schema_version": RENDER_STEP_MICROBENCHMARK_SCHEMA,
        "official_evaluator_step_preserved": True,
        "physics_preserved": True,
        "task_metric_timeout_accounting_preserved": True,
        "model_lane_attempted": False,
        "steps_per_lane": steps_per_lane,
        "total_simulator_steps": steps_per_lane * 3,
        "lanes": [full, disabled, sparse],
        "speedup_vs_full": {
            "render_and_video_disabled": (
                baseline_seconds / disabled_seconds
            ),
            "sparse_render_final_checkpoint": (
                baseline_seconds / sparse_seconds
            ),
        },
        "official_challenge_eligible": False,
        "kinematic_research": True,
        "teleport_enabled": False,
        "viewer_camera_control": False,
        "privileged_policy_inputs": False,
        "scene_loads_attempted": 1,
        "successful_scene_loads": 1,
        "strategies": [full, disabled, sparse],
    }


def make_render_step_microbenchmark_callback(
    *, steps_per_lane: int = 20
) -> Callable[..., Mapping[str, Any]]:
    """Build the callback consumed by the one-load production runner."""

    def callback(*, evaluator, control, bindings, run_dir, stages):
        del bindings, run_dir, stages
        import omnigibson as og

        return run_render_step_microbenchmark(
            evaluator=evaluator,
            control=control,
            render_context=og.sim.render_on_step,
            steps_per_lane=steps_per_lane,
        )

    return callback
