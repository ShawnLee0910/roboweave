"""Single-scene Oracle and Qwen diagnostic sweep for ``picking_up_toys``.

This module is deliberately a research diagnostic, not a scored evaluator
path.  It reuses the isolated B1 live bindings for the two Qwen arms of the
experiment and uses the official Starter primitive only in the private Oracle
arm.  All three arms share one Kit / one loaded scene and are separated by
in-memory ``Evaluator.reset()`` calls.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import hashlib
import traceback
from typing import Any, Callable, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    _record_stage,
)
from behavior_hl.embodied.policy.tokenplan_adapter import TOKENPLAN_MODEL
from behavior_hl.embodied.evaluator_bridge.private_observation_ingress import (
    capture_private_head_frame,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_observation_only_grasp_lift import (
    PublicApproachSnapshot,
    PublicApproachStep,
    SharedControlApproachMacroExecutor,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_visible_official_grasp_factory import (
    build_r1pro_visible_official_grasp_services,
    planar_base_pose_from_robot,
    private_picking_up_toys_eligible_objects,
    private_picking_up_toys_eligible_objects_by_category,
)
from behavior_hl.embodied.evaluator_bridge.run_visible_object_grasp_approach import (
    VisibleObjectGraspExecution,
    run_visible_object_grasp_approach,
)
from behavior_hl.embodied.policy.approach import ApproachLedger, ApproachLimits
from behavior_hl.embodied.policy.feedback import PublicStatus
from behavior_hl.embodied.policy.profiles import build_picking_up_toys_registry
from behavior_hl.embodied.policy.observations import CAMERA_SENSOR_NAMES
from behavior_hl.embodied.runtime.dispatch import (
    ActionExecutionContext,
    ActionPluginDispatcher,
)
from behavior_hl.embodied.runtime.plugins.visible_official_grasp import (
    VISIBLE_OFFICIAL_GRASP_BACKEND_SERVICE,
    VisibleObjectOfficialGraspPlugin,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def _as_bool(value: Any) -> bool:
    return bool(value.item() if hasattr(value, "item") else value)


def _scope_object(task: Any, prefix: str) -> Any:
    scope = getattr(task, "object_scope", None)
    if not isinstance(scope, Mapping):
        raise RuntimeError("BehaviorTask object_scope is unavailable")
    exact = scope.get(prefix)
    if exact is not None:
        return exact
    matches = [
        value
        for key, value in scope.items()
        if str(key).startswith(prefix) and value is not None
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one object for {prefix!r}, found {len(matches)}"
        )
    return matches[0]


def _reset_same_scene(
    *,
    evaluator: Any,
    control: Any,
    stages: Path,
    phase: str,
    run_dir: Path,
    reset_index: int,
) -> int:
    control.hold()
    _record_stage(
        stages,
        "STRATEGY_RESET_START",
        phase=phase,
        reset_index=reset_index,
    )
    evaluator.reset()
    _record_stage(
        stages,
        "STRATEGY_RESET_PASS",
        phase=phase,
        reset_index=reset_index,
    )
    _write_json(
        run_dir / "private" / f"{phase}_reset.json",
        {
            "phase": phase,
            "reset_index": reset_index,
            "scene_reconstructed": False,
            "evaluator_reset_only": True,
        },
    )
    return reset_index + 1


def _run_official_primitive(
    *,
    evaluator: Any,
    starter: Any,
    primitive: Any,
    target: Any,
    phase: str,
    trace_path: Path,
    maximum_steps: int = 12_000,
) -> dict[str, Any]:
    """Run one official Starter generator against the live environment."""

    steps = 0
    terminated = False
    truncated = False
    generator = starter.apply_ref(primitive, target, attempts=2)
    for low_level_action in generator:
        if low_level_action is None:
            continue
        if steps >= maximum_steps:
            raise RuntimeError(f"{phase} exceeded {maximum_steps} simulator steps")
        result = evaluator.env.step(low_level_action)
        if not isinstance(result, tuple) or len(result) < 5:
            raise RuntimeError(f"{phase} env.step returned an invalid result")
        _, _, terminated_value, truncated_value, info = result[:5]
        terminated = _as_bool(terminated_value)
        truncated = _as_bool(truncated_value)
        _append_jsonl(
            trace_path,
            {
                "phase": phase,
                "step": steps,
                "terminated": terminated,
                "truncated": truncated,
                "info_keys": sorted(info) if isinstance(info, Mapping) else [],
            },
        )
        steps += 1
        if terminated or truncated:
            break
    return {
        "status": "completed" if not (terminated or truncated) else "terminated",
        "simulator_steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "held_object": getattr(starter._get_obj_in_hand(), "name", None),
    }


def _oracle_phase(
    *,
    evaluator: Any,
    bindings: Any,
    run_dir: Path,
    stages: Path,
) -> dict[str, Any]:
    """Use private task objects only to test the official physical primitive."""

    _record_stage(stages, "ORACLE_START")
    starter = bindings._starter_for(evaluator)
    from omnigibson import object_states
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitiveSet,
    )

    task = evaluator.env.task
    ball = _scope_object(task, "tennis_ball.n.01_1")
    toy_box = _scope_object(task, "toy_box.n.01_1")
    trace_path = run_dir / "private" / "oracle_trace.jsonl"
    result: dict[str, Any] = {
        "method": "oracle_official_starter_grasp_place_inside",
        "target_category": str(getattr(ball, "category", "")),
        "container_category": str(getattr(toy_box, "category", "")),
        "physical_grasp_executions": 0,
        "physical_grasp_successes": 0,
        "physical_place_executions": 0,
        "grasp": None,
        "place": None,
    }
    approach_ledger: ApproachLedger | None = None
    try:
        grasp = _run_official_primitive(
            evaluator=evaluator,
            starter=starter,
            primitive=StarterSemanticActionPrimitiveSet.GRASP,
            target=ball,
            phase="oracle.grasp",
            trace_path=trace_path,
        )
        result["grasp"] = grasp
        held = starter._get_obj_in_hand()
        if held is not ball:
            raise RuntimeError(
                "Oracle grasp did not hold the expected tennis ball"
            )
        result["physical_grasp_executions"] = 1
        place = _run_official_primitive(
            evaluator=evaluator,
            starter=starter,
            primitive=StarterSemanticActionPrimitiveSet.PLACE_INSIDE,
            target=toy_box,
            phase="oracle.place_inside",
            trace_path=trace_path,
        )
        result["place"] = place
        result["physical_place_executions"] = 1
        result["inside_after_place"] = bool(
            ball.states[object_states.Inside].get_value(toy_box)
        )
        result["status"] = (
            "PASS" if result["inside_after_place"] is True else "FAIL_POSTCONDITION"
        )
    except Exception as exc:
        result.update(
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    _write_json(run_dir / "private" / "oracle_result.json", result)
    _record_stage(
        stages,
        "ORACLE_COMPLETE",
        status=result.get("status"),
        inside_after_place=result.get("inside_after_place"),
        simulator_steps=(
            int((result.get("grasp") or {}).get("simulator_steps", 0))
            + int((result.get("place") or {}).get("simulator_steps", 0))
        ),
    )
    return result


def _set_qwen_budget(bindings: Any, maximum_model_calls: int) -> None:
    """Use a bounded per-strategy public budget without changing the protocol."""

    callback = getattr(bindings, "_approach_callback", None)
    adapter = getattr(callback, "__self__", None)
    config = getattr(adapter, "_approach_config", None)
    if config is None:
        raise RuntimeError("Qwen approach adapter is unavailable")
    limits = replace(
        config.limits,
        max_model_calls=int(maximum_model_calls),
        max_simulator_steps=1600,
    )
    adapter._approach_config = replace(config, limits=limits)


def _qwen_phase(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    run_dir: Path,
    stages: Path,
    phase: str,
    maximum_model_calls: int = 16,
) -> dict[str, Any]:
    """Run public RGB-D Qwen approach followed by real B1 grasp/lift."""

    _record_stage(stages, "QWEN_STRATEGY_START", phase=phase)
    phase_dir = run_dir / "strategies" / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    _set_qwen_budget(bindings, maximum_model_calls)
    result: dict[str, Any] = {
        "method": "qwen_public_rgbd_observation_then_physical_grasp_lift",
        "phase": phase,
        "model": TOKENPLAN_MODEL,
        "status": "FAIL",
        "physical_grasp_executions": 0,
    }
    try:
        approach = bindings.observation_only_approach(
            evaluator=evaluator,
            control=control,
            run_dir=phase_dir,
        )
        proposal = bindings.propose_candidate(
            evaluator=evaluator,
            control=control,
            approach=approach,
            run_dir=phase_dir,
        )
        empty_wrist = bindings.calibrate_empty_wrist(
            evaluator=evaluator,
            control=control,
            approach=approach,
            proposal=proposal,
            minimum_trials=3,
            run_dir=phase_dir,
        )
        grasp_lift = bindings.physical_grasp_lift(
            evaluator=evaluator,
            control=control,
            approach=approach,
            proposal=proposal,
            empty_wrist=empty_wrist,
            run_dir=phase_dir,
        )
        result.update(
            {
                "status": "PASS",
                "physical_grasp_executions": int(
                    grasp_lift.physical_grasp_executions
                ),
                "observed_lift_m": float(grasp_lift.observed_lift_m),
                "approach_target_from_public_rgbd": bool(
                    approach.target_from_public_rgbd
                ),
                "approach_model_calls": int(
                    approach._ledger.model_calls
                    if hasattr(approach, "_ledger")
                    else -1
                ),
            }
        )
        starter = bindings._starter_for(evaluator)
        result["held_object"] = getattr(
            starter._get_obj_in_hand(), "name", None
        )
    except Exception as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    _write_json(phase_dir / "summary.json", result)
    _record_stage(
        stages,
        "QWEN_STRATEGY_COMPLETE",
        phase=phase,
        status=result.get("status"),
        physical_grasp_executions=result.get("physical_grasp_executions"),
    )
    return result


def _qwen_visible_object_grasp_phase(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    run_dir: Path,
    stages: Path,
    phase: str = "qwen_visible_object_grasp",
) -> dict[str, Any]:
    """Run the new category-level Qwen policy and official physical GRASP."""

    from behavior_hl.embodied.evaluator_bridge.r1pro_b1_live_entrypoint import (
        _write_public_snapshot,
    )
    from behavior_hl.embodied.evaluator_bridge.run_g4a_grasp_suite import (
        _run_tokenplan_approach_call,
    )

    _record_stage(stages, "QWEN_VISIBLE_OBJECT_START", phase=phase)
    phase_dir = run_dir / "strategies" / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "method": "qwen_visible_category_then_bounded_official_grasp",
        "phase": phase,
        "model": TOKENPLAN_MODEL,
        "status": "FAIL",
        "physical_grasp_executions": 0,
        "textual_executor_feedback_to_model": False,
        "model_object_coordinates": False,
        "model_instance_ids": False,
    }
    try:
        adapter = getattr(bindings._approach_callback, "__self__", None)
        approach_config = getattr(adapter, "_approach_config", None)
        base_contact_provider = getattr(adapter, "_base_contact_provider", None)
        if approach_config is None or not callable(base_contact_provider):
            raise RuntimeError("live approach configuration is unavailable")
        starter = bindings._starter_for(evaluator)
        hand = str(getattr(starter, "arm", ""))
        if hand not in {"left", "right"}:
            raise RuntimeError("official Starter has no calibrated hand")
        registry = build_picking_up_toys_registry(
            calibrated_hands=(hand,),
            grasp_mode="visible_object_official_macro",
        )
        plugin_dispatcher = ActionPluginDispatcher(
            [VisibleObjectOfficialGraspPlugin(hand=hand)]
        )
        eligible = private_picking_up_toys_eligible_objects(evaluator)
        eligible_by_category = (
            private_picking_up_toys_eligible_objects_by_category(evaluator)
        )
        capture_index = 0
        grasp_action_index = 0

        def capture() -> PublicApproachSnapshot:
            nonlocal capture_index
            capture_index += 1
            control.hold()
            bindings._raw_step(
                evaluator,
                f"{phase}.snapshot_{capture_index:03d}",
            )
            snapshot = bindings._snapshot(evaluator)
            path = (
                phase_dir
                / "policy"
                / "captures"
                / f"public_snapshot_{capture_index:03d}.npz"
            )
            _write_public_snapshot(path, snapshot)
            head = np.ascontiguousarray(
                snapshot["cameras"]["head"]["rgb"],
                dtype=np.uint8,
            )
            position, quaternion = evaluator.robot.get_position_orientation()
            return PublicApproachSnapshot(
                snapshot_path=path,
                snapshot=snapshot,
                source_head_rgb_sha256=hashlib.sha256(
                    head.tobytes()
                ).hexdigest(),
                robot_position=tuple(float(item) for item in position),
                robot_orientation_xyzw=tuple(
                    float(item) for item in quaternion
                ),
                simulator_steps=1,
            )

        def macro_step(label: str, allow_base_motion: bool) -> PublicApproachStep:
            if allow_base_motion is not True:
                raise RuntimeError("visible-object macro disabled base motion")
            bindings._raw_step(evaluator, f"{phase}.{label}")
            public = control.public_state()
            return PublicApproachStep(
                base_qvel=public["base_qvel"],
                head_forward_clearance_m=public[
                    "head_forward_clearance_m"
                ],
                torso_qpos=public["trunk_qpos"],
                base_contact=bool(base_contact_provider(evaluator)),
                simulator_steps=1,
            )

        macro = SharedControlApproachMacroExecutor(
            control=control,
            step=macro_step,
            base_config=approach_config.base_macro,
            command_frequency_hz=approach_config.command_frequency_hz,
            base_settle_steps=approach_config.base_settle_steps,
            look_maximum_steps=approach_config.look_maximum_steps,
            look_settle_steps=approach_config.look_settle_steps,
            look_tolerance_rad=approach_config.look_tolerance_rad,
            look_max_delta_per_step=approach_config.look_max_delta_per_step,
        )

        def execute_grasp(frame: PublicApproachSnapshot, action: Any):
            nonlocal grasp_action_index
            head_sensor = evaluator.robot.sensors[CAMERA_SENSOR_NAMES["head"]]
            instance_registry = getattr(head_sensor, "INSTANCE_REGISTRY", None)
            if not isinstance(instance_registry, Mapping):
                raise RuntimeError("head instance registry is unavailable")
            private_frame = capture_private_head_frame(
                raw=evaluator.obs,
                public_head_rgb=frame.snapshot["cameras"]["head"]["rgb"],
                simulator_step=int(
                    control.public_state()["observation_count"]
                ),
                instance_registry={
                    int(key): str(value)
                    for key, value in instance_registry.items()
                },
            )

            def command_step(command: Any, label: str) -> int:
                control.command_direct_action(command)
                return bindings._raw_step(evaluator, f"{phase}.{label}")

            services = build_r1pro_visible_official_grasp_services(
                starter_by_hand={hand: starter},
                private_frame_provider=lambda: private_frame,
                eligible_objects_provider=lambda: eligible,
                eligible_objects_by_category_provider=(
                    lambda: eligible_by_category
                ),
                command_step=command_step,
                base_pose_provider=lambda: planar_base_pose_from_robot(
                    evaluator.robot
                ),
                held_object_provider_by_hand={
                    hand: starter._get_obj_in_hand
                },
                hold_safe_by_hand={hand: control.hold},
                private_audit_writer=lambda payload: _append_jsonl(
                    phase_dir / "private" / "grasp_audit.jsonl",
                    payload,
                ),
            ).services
            context = ActionExecutionContext(
                episode_id=phase,
                action_index=grasp_action_index,
                services=services,
            )
            grasp_action_index += 1
            active = plugin_dispatcher.begin(registry, action, context)
            plugin_result = None
            for _ in range(registry.spec("grasp").max_simulator_steps + 2):
                plugin_result = plugin_dispatcher.tick(active)
                if plugin_result is not None:
                    break
            if plugin_result is None:
                plugin_result = plugin_dispatcher.abort(
                    active,
                    "controller_tick_budget_exhausted",
                )
            status = (
                PublicStatus.GRASP_LIKELY
                if plugin_result.status == "grasp_likely"
                else PublicStatus.GRASP_UNCERTAIN
            )
            return VisibleObjectGraspExecution(
                status=status,
                simulator_steps=active.simulator_steps,
            )

        def persist(item: Any, digest: str) -> None:
            _append_jsonl(
                phase_dir / "public" / "actions.jsonl",
                {
                    "sequence": item.sequence,
                    "action": dict(item.action),
                    "source_head_rgb_sha256": digest,
                },
            )

        limits = ApproachLimits(
            max_model_calls=24,
            max_invalid_responses=2,
            max_consecutive_same_action=6,
            max_consecutive_blocked=3,
            max_grasp_queries=6,
            max_grasp_executions=6,
            max_simulator_steps=4200,
            prompt_history_items=8,
            translations_after_out_of_reach=2,
            minimum_effective_translation_m=0.05,
        )
        approach_ledger = ApproachLedger(limits)
        approach = run_visible_object_grasp_approach(
            run_dir=phase_dir,
            capture_public_snapshot=capture,
            macro_executor=macro,
            execute_grasp=execute_grasp,
            limits=limits,
            tokenplan_wall_timeout_seconds=(
                approach_config.tokenplan_wall_timeout_seconds
            ),
            persist_public_result=persist,
            tokenplan_call=_run_tokenplan_approach_call,
            hand=hand,
            ledger=approach_ledger,
        )
        result.update(
            {
                "status": "PASS",
                "physical_grasp_executions": approach.grasp_attempts,
                "physical_grasp_successes": 1,
                "model_calls": approach.model_calls,
                "simulator_steps": approach.simulator_steps,
                "held_object": getattr(
                    starter._get_obj_in_hand(),
                    "name",
                    None,
                ),
                "final_action": approach.final_action.to_dict(),
            }
        )
    except Exception as exc:
        if approach_ledger is not None:
            result.update(
                {
                    "physical_grasp_executions": (
                        approach_ledger.grasp_executions
                    ),
                    "model_calls": approach_ledger.model_calls,
                    "simulator_steps": approach_ledger.simulator_steps,
                    "grasp_queries": approach_ledger.grasp_queries,
                    "ledger_summary": approach_ledger.public_summary(),
                }
            )
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    _write_json(phase_dir / "summary.json", result)
    _record_stage(
        stages,
        "QWEN_VISIBLE_OBJECT_COMPLETE",
        phase=phase,
        status=result.get("status"),
        physical_grasp_executions=result.get("physical_grasp_executions"),
    )
    return result


def _preposition_variant(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    run_dir: Path,
    phase: str,
) -> dict[str, Any]:
    """Try an ESI-style turn-right + forward preposition before Qwen sees RGB-D."""

    trace = run_dir / "strategies" / phase / "scripted_preposition.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        ("turn_right", (0.0, 0.0, -0.20), 36),
        ("move_forward", (0.20, 0.0, 0.0), 96),
    ]
    steps = 0
    completed = 0
    for action, command, count in commands:
        for index in range(count):
            control.command_base(command)
            bindings._raw_step(evaluator, f"{phase}.{action}.{index:03d}")
            steps += 1
        control.hold()
        completed += 1
    _write_json(
        run_dir / "strategies" / phase / "preposition_summary.json",
        {
            "method": "scripted_turn_right_forward_before_qwen",
            "status": "PASS",
            "macro_count": completed,
            "simulator_steps": steps,
            "uses_object_coordinates": False,
        },
    )
    return {
        "method": "scripted_turn_right_forward_before_qwen",
        "status": "PASS",
        "simulator_steps": steps,
        "macro_count": completed,
        "uses_object_coordinates": False,
    }


def make_strategy_sweep_callback(
    *,
    build_bindings: Callable[[], Any],
) -> Callable[..., Mapping[str, Any]]:
    """Build a callback suitable for ``run_b1_single_load``."""

    if not callable(build_bindings):
        raise TypeError("build_bindings must be callable")

    def callback(
        *,
        evaluator: Any,
        control: Any,
        bindings: Any,
        run_dir: Path,
        stages: Path,
    ) -> Mapping[str, Any]:
        reset_index = 0
        strategies: list[dict[str, Any]] = []

        oracle = _oracle_phase(
            evaluator=evaluator,
            bindings=bindings,
            run_dir=run_dir,
            stages=stages,
        )
        strategies.append(oracle)
        reset_index = _reset_same_scene(
            evaluator=evaluator,
            control=control,
            stages=stages,
            phase="after_oracle",
            run_dir=run_dir,
            reset_index=reset_index,
        )

        qwen_a = _qwen_phase(
            evaluator=evaluator,
            control=control,
            bindings=bindings,
            run_dir=run_dir,
            stages=stages,
            phase="qwen_default",
            # Give the no-GT policy enough calls to distinguish a genuinely
            # unreachable target from a budget exhausted after only four reach
            # queries.  The per-strategy simulator cap remains finite.
            maximum_model_calls=32,
        )
        strategies.append(qwen_a)
        reset_index = _reset_same_scene(
            evaluator=evaluator,
            control=control,
            stages=stages,
            phase="after_qwen_default",
            run_dir=run_dir,
            reset_index=reset_index,
        )

        bindings_b = build_bindings()
        preposition = _preposition_variant(
            evaluator=evaluator,
            control=control,
            bindings=bindings_b,
            run_dir=run_dir,
            phase="qwen_prepositioned",
        )
        qwen_b = _qwen_phase(
            evaluator=evaluator,
            control=control,
            bindings=bindings_b,
            run_dir=run_dir,
            stages=stages,
            phase="qwen_prepositioned",
            maximum_model_calls=24,
        )
        qwen_b = {**preposition, **qwen_b}
        strategies.append(qwen_b)
        reset_index = _reset_same_scene(
            evaluator=evaluator,
            control=control,
            stages=stages,
            phase="after_qwen_prepositioned",
            run_dir=run_dir,
            reset_index=reset_index,
        )
        return {
            "schema_version": "behavior2026.b1-single-load-strategy-sweep/v1",
            "task": "picking_up_toys",
            "split": "train",
            "instance_id": 0,
            "target_probe": "tennis_ball",
            "model": TOKENPLAN_MODEL,
            "scene_loads_attempted": 1,
            "successful_scene_loads": 1,
            "in_memory_resets": reset_index,
            "strategies": strategies,
            "oracle_success": oracle.get("status") == "PASS",
            "qwen_successes": sum(
                item.get("status") == "PASS"
                for item in strategies
                if item.get("method", "").startswith("qwen")
            ),
            "official_challenge_eligible": False,
            "kinematic_research": True,
            "privileged_policy_inputs": False,
            "teleport_enabled": False,
            "viewer_camera_control": False,
        }

    return callback


def make_visible_object_grasp_callback(
    *,
    build_bindings: Callable[[], Any],
) -> Callable[..., Mapping[str, Any]]:
    """One scene load: private Oracle first, then category-level Qwen grasp."""

    if not callable(build_bindings):
        raise TypeError("build_bindings must be callable")

    def callback(
        *,
        evaluator: Any,
        control: Any,
        bindings: Any,
        run_dir: Path,
        stages: Path,
    ) -> Mapping[str, Any]:
        oracle = _oracle_phase(
            evaluator=evaluator,
            bindings=bindings,
            run_dir=run_dir,
            stages=stages,
        )
        reset_index = _reset_same_scene(
            evaluator=evaluator,
            control=control,
            stages=stages,
            phase="after_visible_object_oracle",
            run_dir=run_dir,
            reset_index=0,
        )
        qwen = _qwen_visible_object_grasp_phase(
            evaluator=evaluator,
            control=control,
            bindings=bindings,
            run_dir=run_dir,
            stages=stages,
        )
        return {
            "schema_version": (
                "behavior2026.b1-visible-object-grasp-canary/v1"
            ),
            "task": "picking_up_toys",
            "split": "train",
            "instance_id": 0,
            "target_probe": "tennis_ball",
            "model": TOKENPLAN_MODEL,
            "scene_loads_attempted": 1,
            "successful_scene_loads": 1,
            "in_memory_resets": reset_index,
            "strategies": [oracle, qwen],
            "oracle_success": oracle.get("status") == "PASS",
            "qwen_successes": int(qwen.get("status") == "PASS"),
            "official_challenge_eligible": False,
            "kinematic_research": True,
            "privileged_policy_inputs": False,
            "teleport_enabled": False,
            "viewer_camera_control": False,
            "model_visible_segmentation": False,
            "model_visible_executor_text": False,
        }

    return callback
