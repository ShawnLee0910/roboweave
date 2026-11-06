"""One-load Task25 far-visible rejection and reset/replay canary.

Private setup places the resident robot at a target-facing station inside the
frozen far-visible band.  Each public epoch uses only ``pick()`` followed by
``pick(marker)`` through the existing Action20 registry and executor.  One
canonical same-scene reset separates the epochs; no model or prompt exists in
this path.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.action20_task25_distance_reset_spec import (
    ACTION20_TASK25_DISTANCE_RESET_SPEC_VERSION,
    TASK25_DISTANCE_RESET_SPEC,
    Task25DistanceResetEpisodeSpec,
    Task25DistanceResetSpec,
    validate_task25_distance_reset_spec,
)
from behavior_hl.embodied.evaluator_bridge.run_action20_task25_atomic_canary import (
    _private_workbench_front_view_setup,
    resolve_task25_workbench_fixture,
)
from behavior_hl.embodied.evaluator_bridge.same_scene_reset_attestation import (
    PrivateResetSnapshot,
    SameSceneResetTolerances,
    attest_same_scene_reset,
)


ACTION20_TASK25_DISTANCE_RESET_RUNNER_VERSION = (
    "behavior2026.action20-task25-distance-reset-runner/v1"
)
PHASE_NAME = "action20_task25_distance_reset"


def _default_runtime_api() -> Any:
    return importlib.import_module(
        "behavior_hl.embodied.evaluator_bridge."
        "run_b1_marker_symbolic_rollout"
    )


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _safe_json(item())
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_json(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _record_stage(
    api: Any,
    stages: Path | None,
    name: str,
    **fields: Any,
) -> None:
    if stages is None:
        return
    recorder = getattr(api, "_record_stage", None)
    if callable(recorder):
        recorder(stages, name, **fields)


def _entity_name(entity: Any) -> str:
    name = str(getattr(entity, "name", ""))
    if not name:
        raise RuntimeError("Task25 distance/reset target is unnamed")
    return name


def _held_object(runtime: Any) -> Any | None:
    getter = getattr(runtime.symbolic_primitives, "_get_obj_in_hand", None)
    if not callable(getter):
        raise RuntimeError("Task25 distance/reset carry state is unavailable")
    return getter()


def _assert_empty_carry(runtime: Any, *, phase: str) -> None:
    if _held_object(runtime) is not None:
        raise RuntimeError(f"Task25 distance/reset held state is not empty at {phase}")
    snapshot = runtime.carry_tracker.snapshot()
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("Task25 distance/reset carry snapshot is invalid")
    if snapshot.get("likely") is not False or snapshot.get("active_hand") is not None:
        raise RuntimeError(f"Task25 distance/reset public carry is not empty at {phase}")
    tracked = getattr(runtime.carry_tracker, "held_object", None)
    if callable(tracked) and tracked() is not None:
        raise RuntimeError(f"Task25 distance/reset tracker retained carry at {phase}")


def _frame_boundary_record(frames: Sequence[Any]) -> dict[str, Any]:
    if len(frames) < 2:
        raise RuntimeError("Task25 distance/reset action lacks boundary frames")
    return {
        "frame_count": len(frames),
        "before_simulator_step": getattr(frames[0], "simulator_step", None),
        "after_simulator_step": getattr(frames[-1], "simulator_step", None),
        "first_physical_after_present": True,
    }


def _current_target_marker(runtime: Any, target: Any) -> tuple[str, Any]:
    snapshot = getattr(runtime.store, "active_snapshot", None)
    bindings = getattr(snapshot, "bindings", None)
    if not isinstance(bindings, Mapping):
        raise RuntimeError("Task25 distance/reset selection lacks a marker snapshot")
    matching = [
        (marker, binding)
        for marker, binding in bindings.items()
        if getattr(binding, "target_token", None) is target
        and getattr(binding, "binding_kind", "object") == "object"
    ]
    if len(matching) != 1:
        raise RuntimeError(
            "Task25 distance/reset target lacks one current-frame marker"
        )
    marker, binding = matching[0]
    if not isinstance(marker, str) or not marker:
        raise RuntimeError("Task25 distance/reset marker is invalid")
    binding_marker = getattr(binding, "marker", marker)
    if binding_marker != marker:
        raise RuntimeError("Task25 distance/reset marker identity is inconsistent")
    return marker, snapshot


def _clear_marker_state(runtime: Any, *, phase: str) -> None:
    clear = getattr(runtime.store, "clear", None)
    if not callable(clear):
        raise RuntimeError("Task25 distance/reset marker reset is unavailable")
    clear()
    if getattr(runtime.store, "active_snapshot", None) is not None:
        raise RuntimeError(f"Task25 distance/reset marker survived {phase}")


def _build_runtime(
    *,
    api: Any,
    evaluator: Any,
    control: Any,
    profile: Any,
    motion_safety_starter: Any,
    spec: Task25DistanceResetSpec,
) -> Any:
    runtime = api._build_marker_runtime(
        evaluator=evaluator,
        control=control,
        profile=profile,
        full_actions=True,
        motion_safety_starter=motion_safety_starter,
        suppress_held_object_environment_collisions=False,
        logical_inventory_carry=True,
        base_execution=spec.base_execution,
        observation_profile=spec.observation_profile,
        action_profile="marker_symbolic_action20",
    )
    if getattr(runtime.store, "active_snapshot", None) is not None:
        raise RuntimeError("Task25 distance/reset runtime began with a marker")
    _assert_empty_carry(runtime, phase="runtime_start")
    return runtime


def _private_far_visible_setup(
    *,
    api: Any,
    runtime: Any,
    evaluator: Any,
    target: Any,
    spec: Task25DistanceResetSpec,
    private_visibility_provider: Callable[..., Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    report = dict(
        _private_workbench_front_view_setup(
            runtime=runtime,
            evaluator=evaluator,
            target=target,
            desired_distance_m=spec.far_visible_station_m,
            hard_distance_m=spec.far_visible_max_m,
            observation_provider=api._capture_rendered_oracle_observation,
            visibility_provider=private_visibility_provider,
        )
    )
    distance = report.get("distance_m_private")
    if (
        isinstance(distance, bool)
        or not isinstance(distance, (int, float))
        or not spec.far_visible_min_m <= float(distance) <= spec.far_visible_max_m
    ):
        raise RuntimeError("Task25 target is outside the far-visible distance band")
    report.update(
        {
            "setup_kind": spec.private_setup_kind,
            "far_visible_band_pass": True,
            "local_reach_exceeded": float(distance) > spec.local_reach_m,
        }
    )
    if report["local_reach_exceeded"] is not True:
        raise RuntimeError("Task25 far-visible target remains within local reach")
    return report


def _run_public_epoch(
    *,
    api: Any,
    runtime: Any,
    evaluator: Any,
    profile: Any,
    role_provider: Callable[..., Any] | None,
    spec: Task25DistanceResetSpec,
    episode: Task25DistanceResetEpisodeSpec,
    phase_dir: Path,
    private_visibility_provider: Callable[..., Mapping[str, Any]] | None,
    expected_target: Any | None,
) -> tuple[Mapping[str, Any], Any]:
    fixture = resolve_task25_workbench_fixture(
        evaluator,
        profile,
        role_provider=role_provider,
    )
    target = fixture.station_target
    if expected_target is not None and target is not expected_target:
        raise RuntimeError("Task25 target identity changed across same-scene reset")
    epoch_dir = phase_dir / "episodes" / f"epoch_{episode.ordinal}"
    station_report = _private_far_visible_setup(
        api=api,
        runtime=runtime,
        evaluator=evaluator,
        target=target,
        spec=spec,
        private_visibility_provider=private_visibility_provider,
    )
    _write_json(
        epoch_dir / "private" / "far_visible_station.json",
        {
            **dict(station_report),
            "target_name_private": _entity_name(target),
            "private_only": True,
        },
    )
    _assert_empty_carry(runtime, phase="before_selection")

    selection = runtime.registry.parse_payload(episode.selection_payload())
    selection_result, selection_frames, selection_steps = api._execute(
        runtime=runtime,
        action=selection,
        action_index=0,
        phase_dir=epoch_dir,
    )
    if selection_result.status != episode.selection_status:
        raise RuntimeError("Task25 far-visible pick request did not return markers")
    _assert_empty_carry(runtime, phase="after_selection")
    marker, selection_snapshot = _current_target_marker(runtime, target)

    selected = runtime.registry.parse_payload(episode.target_payload(marker))
    blocked_result, blocked_frames, blocked_steps = api._execute(
        runtime=runtime,
        action=selected,
        action_index=1,
        phase_dir=epoch_dir,
    )
    feedback = dict(blocked_result.public_feedback)
    if (
        blocked_result.status != episode.target_status
        or feedback.get("reason_code") != episode.target_reason_code
        or feedback.get("object_acquired") is not False
    ):
        raise RuntimeError("Task25 far-visible pick was not blocked by local reach")
    _assert_empty_carry(runtime, phase="after_blocked_pick")
    remaining_snapshot = getattr(runtime.store, "active_snapshot", None)
    if remaining_snapshot is not None and remaining_snapshot is not selection_snapshot:
        raise RuntimeError("Task25 blocked pick replaced its marker snapshot")
    _clear_marker_state(runtime, phase="episode_cleanup")

    return (
        {
            "ordinal": episode.ordinal,
            "action_index_start": 0,
            "action_index_end": 1,
            "public_action_count": 2,
            "selection": {
                "action": episode.selection_action,
                "status": selection_result.status,
                "marker": marker,
                "simulator_steps": selection_steps,
                "frames": _frame_boundary_record(selection_frames),
            },
            "target_attempt": {
                "action": episode.target_action,
                "status": blocked_result.status,
                "blocked": True,
                "public_feedback": feedback,
                "held_after": False,
                "simulator_steps": blocked_steps,
                "frames": _frame_boundary_record(blocked_frames),
            },
            "marker_cleared": True,
            "held_cleared": True,
            "far_visible": True,
        },
        target,
    )


def _close_runtime(runtime: Any) -> None:
    environment = getattr(runtime, "environment", None)
    close = getattr(environment, "close", None)
    if callable(close):
        close()


def run_action20_task25_distance_reset_canary(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    profile: Any,
    run_dir: Path,
    stages: Path | None = None,
    spec: Task25DistanceResetSpec = TASK25_DISTANCE_RESET_SPEC,
    runtime_api: Any | None = None,
    role_provider: Callable[..., Any] | None = None,
    motion_safety_starter: Any | None = None,
    private_visibility_provider: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Run two dynamic-marker blocked picks around one attested reset."""

    validate_task25_distance_reset_spec(spec)
    api = runtime_api or _default_runtime_api()
    phase_dir = Path(run_dir) / "strategies" / PHASE_NAME
    phase_dir.mkdir(parents=True, exist_ok=False)
    _record_stage(api, stages, "ACTION20_TASK25_DISTANCE_RESET_START")
    result: dict[str, Any] = {
        "schema_version": ACTION20_TASK25_DISTANCE_RESET_SPEC_VERSION,
        "runner_version": ACTION20_TASK25_DISTANCE_RESET_RUNNER_VERSION,
        "status": "FAIL",
        "task_id": spec.task_id,
        "task_name": spec.task_name,
        "seed": spec.seed,
        "scene_load_count": 1,
        "in_memory_reset_count": 0,
        "public_action_budget": spec.public_action_budget,
        "public_actions_executed": 0,
        "public_epochs": [],
        "model_lane_authorized": False,
        "qwen_attempted": False,
        "official_challenge_eligible": False,
        "execution_fidelity": "kinematic_research",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(phase_dir / "distance_reset_spec.json", spec.to_record())
    runtimes: list[Any] = []
    closed_runtime_ids: set[int] = set()
    try:
        capture = getattr(bindings, "capture_private_reset_snapshot", None)
        if not callable(capture):
            raise RuntimeError("Task25 distance/reset snapshot binding is unavailable")
        pre_reset = capture(
            evaluator=evaluator,
            phase="pre_calibration",
            run_dir=run_dir,
        )
        if not isinstance(pre_reset, PrivateResetSnapshot):
            raise RuntimeError("Task25 distance/reset pre-reset snapshot is invalid")
        reset_dir = phase_dir / "private" / "reset_attestation"
        _write_json(
            reset_dir / "pre_reset.json",
            pre_reset.to_private_record(include_stable_state=True),
        )

        starter = motion_safety_starter
        if starter is None:
            starter = api._build_full_motion_safety_starter(evaluator)
        first_runtime = _build_runtime(
            api=api,
            evaluator=evaluator,
            control=control,
            profile=profile,
            motion_safety_starter=starter,
            spec=spec,
        )
        runtimes.append(first_runtime)
        first_epoch, target = _run_public_epoch(
            api=api,
            runtime=first_runtime,
            evaluator=evaluator,
            profile=profile,
            role_provider=role_provider,
            spec=spec,
            episode=spec.episodes[0],
            phase_dir=phase_dir,
            private_visibility_provider=private_visibility_provider,
            expected_target=None,
        )
        result["public_epochs"].append(dict(first_epoch))
        result["public_actions_executed"] = 2
        _close_runtime(first_runtime)
        closed_runtime_ids.add(id(first_runtime))

        control.hold()
        _record_stage(api, stages, "ACTION20_TASK25_DISTANCE_RESET_RESET_START")
        reset = getattr(evaluator, "reset", None)
        if not callable(reset):
            raise RuntimeError("Task25 distance/reset evaluator reset is unavailable")
        reset()
        _record_stage(api, stages, "ACTION20_TASK25_DISTANCE_RESET_RESET_PASS")
        post_reset = capture(
            evaluator=evaluator,
            phase="post_reset",
            run_dir=run_dir,
        )
        if not isinstance(post_reset, PrivateResetSnapshot):
            raise RuntimeError("Task25 distance/reset post-reset snapshot is invalid")
        _write_json(
            reset_dir / "post_reset.json",
            post_reset.to_private_record(include_stable_state=True),
        )
        attestation = attest_same_scene_reset(
            pre=pre_reset,
            post=post_reset,
            tolerances=SameSceneResetTolerances(),
        )
        _write_json(reset_dir / "report.json", attestation.to_private_record())
        attestation.require_pass()
        result["in_memory_reset_count"] = 1
        _record_stage(
            api,
            stages,
            "ACTION20_TASK25_DISTANCE_RESET_ATTESTATION_PASS",
        )

        second_runtime = _build_runtime(
            api=api,
            evaluator=evaluator,
            control=control,
            profile=profile,
            motion_safety_starter=starter,
            spec=spec,
        )
        runtimes.append(second_runtime)
        second_epoch, replay_target = _run_public_epoch(
            api=api,
            runtime=second_runtime,
            evaluator=evaluator,
            profile=profile,
            role_provider=role_provider,
            spec=spec,
            episode=spec.episodes[1],
            phase_dir=phase_dir,
            private_visibility_provider=private_visibility_provider,
            expected_target=target,
        )
        if replay_target is not target:
            raise RuntimeError("Task25 replay did not preserve target identity")
        result["public_epochs"].append(dict(second_epoch))
        result["public_actions_executed"] = 4
        _close_runtime(second_runtime)
        closed_runtime_ids.add(id(second_runtime))

        semantic_signatures = [
            (
                epoch["selection"]["action"],
                epoch["selection"]["status"],
                epoch["target_attempt"]["action"],
                epoch["target_attempt"]["status"],
                epoch["target_attempt"]["public_feedback"].get("reason_code"),
                epoch["target_attempt"]["held_after"],
            )
            for epoch in result["public_epochs"]
        ]
        if len(set(semantic_signatures)) != 1:
            raise RuntimeError("Task25 replay public semantics drifted")
        if result["public_actions_executed"] > spec.public_action_budget:
            raise RuntimeError("Task25 distance/reset public budget exceeded")

        result.update(
            {
                "status": "PASS",
                "reset_attestation_pass": True,
                "sequence_reset_pass": all(
                    epoch["action_index_start"] == 0
                    and epoch["action_index_end"] == 1
                    for epoch in result["public_epochs"]
                ),
                "marker_reset_pass": all(
                    epoch["marker_cleared"] for epoch in result["public_epochs"]
                ),
                "held_reset_pass": all(
                    epoch["held_cleared"] for epoch in result["public_epochs"]
                ),
                "same_target_semantics_replayed": True,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _record_stage(api, stages, "ACTION20_TASK25_DISTANCE_RESET_PASS")
        _write_json(phase_dir / "summary.json", result)
        return result
    except Exception as exc:
        result.update(
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _record_stage(
            api,
            stages,
            "ACTION20_TASK25_DISTANCE_RESET_FAIL",
            error_type=type(exc).__name__,
        )
        _write_json(phase_dir / "summary.json", result)
        raise
    finally:
        for runtime in runtimes:
            try:
                _clear_marker_state(runtime, phase="final_cleanup")
            except Exception:
                pass
            if id(runtime) not in closed_runtime_ids:
                try:
                    _close_runtime(runtime)
                except Exception:
                    pass


def make_action20_task25_distance_reset_callback(
    *,
    profile: Any | None = None,
    spec: Task25DistanceResetSpec = TASK25_DISTANCE_RESET_SPEC,
    runtime_api: Any | None = None,
    role_provider: Callable[..., Any] | None = None,
    private_visibility_provider: Callable[..., Mapping[str, Any]] | None = None,
) -> Callable[..., Mapping[str, Any]]:
    """Return the frozen standard one-load strategy callback."""

    validate_task25_distance_reset_spec(spec)
    resolved_profile = profile
    if resolved_profile is None:
        profiles = importlib.import_module(
            "behavior_hl.embodied.evaluator_bridge.inside_task_profiles"
        )
        resolved_profile = profiles.get_inside_task_profile(spec.task_name)

    def callback(
        *,
        evaluator: Any,
        control: Any,
        bindings: Any,
        run_dir: Path,
        stages: Path,
    ) -> Mapping[str, Any]:
        summary = run_action20_task25_distance_reset_canary(
            evaluator=evaluator,
            control=control,
            bindings=bindings,
            profile=resolved_profile,
            run_dir=run_dir,
            stages=stages,
            spec=spec,
            runtime_api=runtime_api,
            role_provider=role_provider,
            private_visibility_provider=private_visibility_provider,
        )
        success = summary.get("status") == "PASS"
        return {
            "schema_version": ACTION20_TASK25_DISTANCE_RESET_SPEC_VERSION,
            "strategies": [summary],
            "action20_task25_distance_reset_success": success,
            "oracle_success": success,
            "scene_load_count": 1,
            "in_memory_reset_count": 1,
            "model_lane_authorized": False,
            "qwen_attempted": False,
            "official_challenge_eligible": False,
            "execution_fidelity": "kinematic_research",
        }

    return callback


__all__ = [
    "ACTION20_TASK25_DISTANCE_RESET_RUNNER_VERSION",
    "PHASE_NAME",
    "make_action20_task25_distance_reset_callback",
    "run_action20_task25_distance_reset_canary",
]
