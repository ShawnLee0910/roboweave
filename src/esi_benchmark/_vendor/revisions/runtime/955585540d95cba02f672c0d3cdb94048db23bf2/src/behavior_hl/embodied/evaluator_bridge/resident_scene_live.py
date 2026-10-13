"""Live task switching inside one already-constructed OmniGibson scene."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import os
import hashlib
import inspect
import json
import math
from pathlib import Path
from threading import Lock, Timer
from time import monotonic
from typing import Any, Callable, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
)
from behavior_hl.embodied.evaluator_bridge.l0_reset_pose_overlay import attest_l0_reset_pose
from behavior_hl.embodied.evaluator_bridge.resident_scene_batch import (
    RESIDENT_SCENE_BATCH_VERSION,
    ResidentTaskSpec,
    build_runtime_task_restore_document,
    write_json_atomic,
)
from behavior_hl.embodied.evaluator_bridge.resident_task_progress import SCHEMA as RESIDENT_TASK_PROGRESS_VERSION
from behavior_hl.embodied.evaluator_bridge.semantic_execution_profiles import (
    SEMANTIC_RESEARCH_V1,
)
from behavior_hl.embodied.evaluator_bridge.construction_material_adapter import (
    MaterialAdapterRollbackError,
    RUNNER_VRAY_MDL_PATH,
    RUNNER_VRAY_MATERIAL_NAME,
    apply_vray_material_rules,
    restore_material_changes,
)


from .oracle_episode_receipt import commit_episode_result, record_phase, exception_evidence, verification_scope


_FAILURE_CLASSES = frozenset(
    {
        "PUBLIC_PROCESS_REQUIREMENT_FAILED",
        "PREFLIGHT_ERROR",
        "TASKSPEC_BINDING_ERROR",
        "VERIFIER_ERROR",
        "ACTION20_PROTOCOL_ERROR",
        "ORACLE_PLANNER_ERROR",
        "NAVIGATION_ERROR",
        "MARKER_ERROR",
        "ASSET_OR_PHYSICS_INCOMPATIBLE",
        "RESTORE_ERROR",
        "TOO_COMPLEX_FOR_L0",
        "SEMANTIC_REJECT",
        "UNKNOWN",
    }
)


def _record(stages: Path, stage: str, **payload: Any) -> None:
    record = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    stages.parent.mkdir(parents=True, exist_ok=True)
    with stages.open("a", encoding="utf-8") as stream:
        import json

        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def public_oracle_phase_name(index: int, item_id: str, task_count: int) -> str:
    """One path identity for callback execution and resident evidence."""
    suffix = 'public_action20_marker_full_oracle'
    return f'resident_tasks/{index:02d}_{item_id}/{suffix}' if task_count > 1 else suffix


def _physical_canary_failed(contract, result):
    if (contract or {}).get('physical_batch_stop_after_first_failure') is False:
        return False
    if (contract or {}).get('required_qualification') != 'physical_template_review':
        return False
    return not (result.get('status') == 'PASS'
                and type(result.get('official_q')) in (int, float) and result['official_q'] == 1
                and (result.get('physical_template_review') or {}).get('qualification_status') == 'PASS')


def _write_resident_task_protocol(
    *, run_dir: Path, tasks: Sequence[ResidentTaskSpec], item_ids: Sequence[str],
    campaign_seed: Any = None,
    effective_execution_contract: Mapping[str, Any] | None = None,
) -> Path:
    identities = []
    for task in tasks:
        try:
            identities.append(asdict(task))
        except TypeError:
            identities.append({
                key: getattr(task, key, None)
                for key in ("task_name", "task_id", "scene_model", "split", "instance_id")
            })
    path = run_dir / "private" / "resident_task_protocol.json"
    payload = {
        "schema_version": RESIDENT_TASK_PROGRESS_VERSION,
        "seed": campaign_seed,
        "task_names": list(item_ids),
        "task_identities": identities,
        "created_at": _utc_timestamp(),
    }
    if effective_execution_contract is not None:
        from behavior_hl.embodied.evaluator_bridge.benchmark_execution_contract import execution_contract_sha256
        payload['effective_execution_contract'] = dict(effective_execution_contract)
        payload['effective_execution_contract_sha256'] = execution_contract_sha256(effective_execution_contract)
    write_json_atomic(path, payload)
    return path


class _ResidentTaskProgress:
    """Atomic, monotonic per-item lifecycle evidence."""

    def __init__(self, *, run_dir: Path, item_id: str, campaign_seed: Any = None) -> None:
        self.path = run_dir / "private" / "resident_tasks" / item_id / "task_progress.json"
        now = _utc_timestamp()
        self._phase_started = monotonic()
        self.payload: dict[str, Any] = {
            "schema_version": RESIDENT_TASK_PROGRESS_VERSION,
            "seed": campaign_seed,
            "task_name": item_id,
            "phase": "TASK_STARTED",
            "oracle_started": False,
            "timings_seconds": {},
            "timestamps": {
                "created_at": now,
                "updated_at": now,
                "TASK_STARTED": now,
            },
            "prepared_state": None,
            "first_frame": None,
            "execution_dir": str((run_dir / "private" / "resident_tasks" / item_id).resolve()),
            "executor_actions_path": None,
        }
        self._write()

    def _write(self) -> None:
        write_json_atomic(self.path, self.payload)

    def update(self, phase: str, *, oracle_started: bool | None = None, **fields: Any) -> None:
        now = _utc_timestamp()
        if phase != self.payload['phase']:
            elapsed = monotonic() - self._phase_started
            previous = self.payload['phase']
            self.payload['timings_seconds'][previous] = self.payload['timings_seconds'].get(previous, 0.) + elapsed
            self._phase_started = monotonic()
        self.payload["phase"] = phase
        if oracle_started is not None:
            self.payload["oracle_started"] = bool(oracle_started)
        self.payload["timestamps"]["updated_at"] = now
        self.payload["timestamps"][phase] = now
        for key, value in fields.items():
            if value is not None:
                self.payload[key] = value
        self._write()


def _run_oracle_with_deadline(
    callback: Any, *, timeout_seconds: float | None, item_id: str,
    run_dir: Path, stages: Path,
    progress_update: Callable[..., Any] | None = None,
    execution_phase_name: str | None = None,
    **kwargs: Any,
) -> Any:
    """Bound the entire callback; preserve final verification before its cleanup."""
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError('Oracle timeout must be positive and finite')
    inspect.signature(callback).bind(run_dir=run_dir, stages=stages, **kwargs)
    with verification_scope(run_dir, item_id):
        write_json_atomic(run_dir/'private/resident_tasks'/item_id/'oracle_entry.json',
                          {'oracle_invocation_started': True, 'deadline_scope': 'CALLBACK_INCLUDING_FINALIZATION'})
        return _invoke_oracle_callback(callback, timeout_seconds=timeout_seconds, item_id=item_id,
                                       run_dir=run_dir, stages=stages, progress_update=progress_update,
                                       execution_phase_name=execution_phase_name, **kwargs)


def _invoke_oracle_callback(callback, *, timeout_seconds, item_id, run_dir, stages,
                            progress_update=None, execution_phase_name=None, **kwargs):
    phase_name = execution_phase_name
    if progress_update is not None:
        progress_update('ORACLE_STARTED', oracle_started=True,
                        execution_dir=str((run_dir / 'strategies' / phase_name).resolve()) if phase_name else None,
                        executor_actions_path=str((run_dir / 'strategies' / phase_name / 'private/executor_actions.jsonl').resolve()) if phase_name else None)
    if timeout_seconds is None:
        return callback(run_dir=run_dir, stages=stages, **kwargs)
    started = monotonic()
    lock = Lock()
    finished = False

    def expire() -> None:
        with lock:
            if finished:
                return
            report = {
                "status": "TIMEOUT",
                "failure_class": "ORACLE_RESOURCE_EXHAUSTED",
                "reason": "oracle_wall_time_budget_exhausted",
                "resident_item_id": item_id,
                "oracle_started": True,
                "phase_name": phase_name,
                "oracle_timeout_seconds": timeout_seconds,
                "oracle_execution_seconds": monotonic() - started,
                "resident_state_reusable": False,
                "official_q": None,
                "oracle_execution_status": "INCOMPLETE",
                "evidence_status": "INCOMPLETE",
                "deadline_scope": "CALLBACK_INCLUDING_FINALIZATION",
            }
            if phase_name:
                report["execution_dir"] = str((run_dir / "strategies" / phase_name).resolve())
                report["executor_actions_path"] = str(
                    (run_dir / "strategies" / phase_name / "private" / "executor_actions.jsonl").resolve()
                )
            try:
                # Bound waiting for a concurrent verification writer before exit.
                # The execution-host supervisor can close a missing receipt after cleanup.
                report.update(commit_episode_result(run_dir, item_id, report, lock_timeout=.5))
                if progress_update is not None:
                    progress_update(
                        "TERMINAL",
                        oracle_started=True,
                        execution_dir=report.get("execution_dir"),
                        executor_actions_path=report.get("executor_actions_path"),
                        timeout=True,
                    )
                _record(stages, "ORACLE_RESOURCE_EXHAUSTED", **report)
            finally:
                # A Python exception could be swallowed by action recovery.
                os._exit(124)

    timer = Timer(timeout_seconds, expire)
    timer.daemon = True
    timer.start()
    try:
        return callback(run_dir=run_dir, stages=stages, **kwargs)
    finally:
        with lock:
            finished = True
            timer.cancel()


def switch_resident_task(
    *,
    evaluator: Any,
    control: Any,
    task: ResidentTaskSpec,
    union_template: Mapping[str, Any],
    target_template: Mapping[str, Any],
    runtime_superset: Mapping[str, Any],
    filter_embedded_scene_agents: bool,
    run_dir: Path,
    stages: Path,
    artifact_key: str | None = None,
    native_task_metric: bool = True,
) -> dict[str, Any]:
    """Restore one target task without reconstructing Kit or the scene stage."""

    if type(native_task_metric) is not bool:
        raise TypeError("native_task_metric must be boolean")
    started = monotonic()
    current_task = evaluator.env.task
    if str(current_task.scene_name) != task.scene_model:
        raise RuntimeError("resident task switch escaped the loaded scene model")
    restore_document, restore_report = build_runtime_task_restore_document(
        runtime_superset=runtime_superset,
        union_template=union_template,
        target_template=target_template,
        filter_embedded_scene_agents=filter_embedded_scene_agents,
    )
    artifact_key = task.task_name if artifact_key is None else artifact_key
    if Path(artifact_key).parts != (artifact_key,):
        raise ValueError("resident artifact key is not a safe path component")
    task_dir = run_dir / "private" / "resident_tasks" / artifact_key
    write_json_atomic(task_dir / "restore_plan.json", restore_report)
    _record(
        stages,
        "RESIDENT_CROSS_TASK_RESTORE_START",
        task=task.task_name,
        task_id=task.task_id,
        scene_model=task.scene_model,
    )
    control.hold()
    evaluator.env.scene.restore(restore_document, update_initial_file=True)

    current_task.update_activity(
        env=evaluator.env,
        activity_name=task.task_name,
        activity_definition_id=0,
    )
    current_task.activity_instance_id = 0
    success, feedback = current_task.initialize_activity(env=evaluator.env)
    current_task.feedback = feedback
    if not success:
        raise RuntimeError(f"resident target activity initialization failed: {feedback}")

    evaluator.cfg.task.name = task.task_name
    evaluator.policy._expected_task_id = task.task_id
    control.set_expected_task_id(task.task_id)
    from omnigibson.eval.utils.score_utils import load_human_stats
    from omnigibson.metrics import AgentMetric, TaskMetric

    evaluator.human_stats = load_human_stats(task.task_name)
    evaluator.metrics = [
        AgentMetric(evaluator.human_stats),
        TaskMetric(evaluator.human_stats),
    ]
    from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
        _initialize_loaded_instance,
    )

    _initialize_loaded_instance(
        evaluator,
        stages,
        instance_id=task.instance_id,
        native_task_metric=native_task_metric,
    )
    if evaluator.env.task.activity_name != task.task_name:
        raise RuntimeError("resident task name did not update")
    if evaluator.env.scene.object_registry("name", evaluator.robot.name) is not evaluator.robot:
        raise RuntimeError("resident restore replaced the evaluator robot")
    elapsed = monotonic() - started
    report = {
        "schema_version": RESIDENT_SCENE_BATCH_VERSION,
        "status": "PASS",
        "task": task.task_name,
        "task_id": task.task_id,
        "scene_model": task.scene_model,
        "instance_id": task.instance_id,
        "cross_task_restore_seconds": elapsed,
        "native_task_metric_enabled": native_task_metric,
        "object_add_remove_required": False,
        "robot_identity_preserved": True,
        **dict(restore_report),
    }
    write_json_atomic(task_dir / "restore_result.json", report)
    _record(
        stages,
        "RESIDENT_CROSS_TASK_RESTORE_PASS",
        task=task.task_name,
        elapsed_seconds=elapsed,
        inactive_union_object_count=restore_report[
            "inactive_union_object_count"
        ],
    )
    return report


def capture_construction_rgb(*, evaluator, control, profile, path):
    """Reuse the normal camera capture and PNG writer without executing actions."""
    from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
        _build_marker_runtime, _capture_rendered_oracle_observation, _save_frame,
    )
    from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
        THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    )
    runtime = _build_marker_runtime(evaluator=evaluator, control=control, profile=profile,
                                   observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE)
    observation = _capture_rendered_oracle_observation(runtime)
    from behavior_hl.embodied.evaluator_bridge.construction_camera_pose import capture_head_camera_pose
    camera_readback = dict(capture_head_camera_pose(evaluator), timing='AFTER_RENDER_ONLY_CAPTURE')
    return {**_save_frame(runtime=runtime, observation=observation, path=path),
            'head_camera_readback': camera_readback,
            'status': 'CAPTURED', 'visual_review': 'NOT_RUN',
            'pose_stage': 'CONSTRUCTION_REFERENCE_NOT_FINAL_FROZEN_POSE',
            'oracle_actions_executed': 0}


def capture_light_probe(*, evaluator, control, profile, path, rule):
    import omni.usd
    from behavior_hl.embodied.evaluator_bridge.construction_visual_transform import temporary_light_intensity
    before = json.dumps(json.loads(evaluator.env.scene.save()), sort_keys=True, allow_nan=False)
    with temporary_light_intensity(omni.usd.get_context().get_stage(), evaluator.env.scene, rule) as evidence:
        frame = capture_construction_rgb(evaluator=evaluator, control=control, profile=profile, path=path)
    after = json.dumps(json.loads(evaluator.env.scene.save()), sort_keys=True, allow_nan=False)
    if before != after:
        raise RuntimeError('render-only probe changed scene state')
    return dict(frame=frame, transform=evidence, physical_state_unchanged=True,
                oracle_status='NOT_RUN', visual_review='NOT_RUN')


_VRAY_DIAGNOSTIC_INPUTS = frozenset({
    'reflection_roughness_constant',
    'reflection_roughness_texture_influence',
})


def _material_diagnostic_requested(
    *, construction_only: bool, construction_report: Mapping[str, Any], probe: Any
) -> bool:
    if not construction_only or construction_report.get('calibration_only') is not True:
        return False
    if not isinstance(probe, Mapping) or str(probe.get('kind', '')).casefold() != 'material':
        return False
    return True


def _material_from_change(change: Any) -> Any | None:
    material = getattr(change, 'material', None)
    if material is not None:
        return material
    try:
        return change[0]
    except (IndexError, KeyError, TypeError):
        return None


def _material_path_from_change(change: Any) -> str | None:
    material = _material_from_change(change)
    path = getattr(material, 'prim_path', None)
    return None if path is None else str(path)


def _scene_digest(scene: Any) -> str:
    saved = scene.save()
    document = json.loads(saved) if isinstance(saved, str) else saved
    normalized = json.dumps(document, sort_keys=True, allow_nan=False)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def _vray_readback(material: Any, expected_inputs: Mapping[str, float]) -> dict[str, Any]:
    shader = material.shader
    asset = shader.GetSourceAsset('mdl')
    actual_asset = str(getattr(asset, 'path', asset) or '').replace('\\', '/')
    actual_subidentifier = str(shader.GetSourceAssetSubIdentifier('mdl') or '')
    expected_asset = str(RUNNER_VRAY_MDL_PATH).replace('\\', '/')
    inputs = {}
    passed = actual_asset == expected_asset and actual_subidentifier == RUNNER_VRAY_MATERIAL_NAME
    for key, expected in expected_inputs.items():
        actual = material.get_input(key)
        try:
            actual_float = float(actual)
            input_passed = math.isclose(actual_float, float(expected), rel_tol=1e-6, abs_tol=1e-6)
        except (TypeError, ValueError):
            actual_float, input_passed = None, False
        inputs[key] = {'expected': float(expected), 'actual': actual_float, 'passed': input_passed}
        passed = passed and input_passed
    return {
        'passed': bool(passed),
        'source': {
            'asset': actual_asset,
            'subidentifier': actual_subidentifier,
            'expected_asset': expected_asset,
            'expected_subidentifier': RUNNER_VRAY_MATERIAL_NAME,
            'passed': actual_asset == expected_asset and actual_subidentifier == RUNNER_VRAY_MATERIAL_NAME,
        },
        'inputs': inputs,
    }


def _material_rule_inputs(changes: Sequence[Any]) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    original: dict[str, dict[str, float]] = {}
    materials: dict[str, Any] = {}
    for change in changes:
        if not callable(getattr(change, 'restore', None)):
            continue
        material = _material_from_change(change)
        path = _material_path_from_change(change)
        key = str(getattr(change, 'key', ''))
        if material is None or not path or key not in _VRAY_DIAGNOSTIC_INPUTS:
            raise ValueError('material diagnostic requires only supported VRay transactions')
        if path not in original:
            original[path] = {}
            materials[path] = material
        if key in original[path]:
            raise ValueError(f'duplicate material diagnostic input: {path} {key}')
        value = getattr(change, 'value', None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('material diagnostic roughness value is not numeric')
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('material diagnostic roughness value is outside [0, 1]')
        original[path][key] = value
    for inputs in original.values():
        if 'reflection_roughness_constant' not in inputs:
            raise ValueError('material diagnostic needs reflection_roughness_constant')
        inputs.setdefault('reflection_roughness_texture_influence', 0.0)
    variant = {path: dict(inputs) for path, inputs in original.items()}
    for inputs in variant.values():
        inputs['reflection_roughness_constant'] = 1.0 - inputs['reflection_roughness_constant']
    return materials, original, variant


def _frame_evidence(frame: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: frame.get(key)
        for key in ('path', 'png_sha256', 'simulator_step', 'camera_sha256', 'head_camera_readback')
    }


def capture_material_paired_probe(
    *, evaluator: Any, control: Any, profile: Any, path: Path,
    baseline_frame: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture one same-pose complement while preserving the task's adaptation."""
    all_changes = tuple(getattr(evaluator, '_registered_material_reset', ()))
    materials, original_inputs, variant_inputs = _material_rule_inputs(all_changes)
    if not original_inputs:
        raise ValueError('material diagnostic target is not backed by VRay transactions')
    baseline_readback = {
        path: _vray_readback(materials[path], original_inputs[path])
        for path in sorted(original_inputs)
    }
    if not all(row['passed'] for row in baseline_readback.values()):
        raise RuntimeError('material diagnostic baseline source/input readback failed')
    before_digest = _scene_digest(evaluator.env.scene)

    variant_changes = []
    variant_frame = None
    variant_readback = None
    variant_restore = None
    try:
        for material_path in sorted(variant_inputs):
            variant_changes.extend(
                apply_vray_material_rules(materials[material_path], variant_inputs[material_path])
            )
        if not variant_changes:
            raise RuntimeError('material diagnostic produced no variant transaction')
        variant_readback = {
            material_path: _vray_readback(materials[material_path], variant_inputs[material_path])
            for material_path in sorted(variant_inputs)
        }
        if not all(row['passed'] for row in variant_readback.values()):
            raise RuntimeError('material diagnostic variant source/input readback failed')
        variant_frame = capture_construction_rgb(
            evaluator=evaluator, control=control, profile=profile, path=path
        )
    finally:
        if variant_changes:
            try:
                variant_restore = restore_material_changes(variant_changes)
            except Exception as exc:
                raise MaterialAdapterRollbackError(
                    f'material diagnostic rollback failed: {exc}'
                ) from exc
    restored_readback = None
    if variant_restore is not None:
        try:
            restored_readback = {
                material_path: _vray_readback(materials[material_path], original_inputs[material_path])
                for material_path in sorted(original_inputs)
            }
        except Exception as exc:
            raise MaterialAdapterRollbackError(
                f'material diagnostic original source/input restore readback failed: {exc}'
            ) from exc
        if not all(row['passed'] for row in restored_readback.values()):
            raise MaterialAdapterRollbackError('material diagnostic original source/input restore failed')
    baseline_camera = baseline_frame.get('head_camera_readback')
    variant_camera = variant_frame.get('head_camera_readback')
    same_camera = baseline_camera == variant_camera
    same_step = baseline_frame.get('simulator_step') == variant_frame.get('simulator_step')
    if not isinstance(baseline_camera, Mapping) or not isinstance(variant_camera, Mapping):
        raise RuntimeError('material diagnostic camera readback is missing')
    if not same_camera or not same_step:
        raise RuntimeError('material diagnostic pair changed pose/camera or simulator step')
    try:
        after_digest = _scene_digest(evaluator.env.scene)
    except Exception as exc:
        raise MaterialAdapterRollbackError(
            f'material diagnostic scene parity readback failed: {exc}'
        ) from exc
    parity = before_digest == after_digest
    if not parity:
        raise MaterialAdapterRollbackError('material diagnostic changed serialized scene state')
    return {
        'schema_version': 'behavior2026.material-diagnostic-paired-capture/v1',
        'status': 'PASS',
        'kind': 'material',
        'roughness_input': 'reflection_roughness_constant',
        'baseline': {
            'inputs': original_inputs,
            'frame': _frame_evidence(baseline_frame),
            'source_input_readback': baseline_readback,
        },
        'complement': {
            'inputs': variant_inputs,
            'frame': _frame_evidence(variant_frame),
            'source_input_readback': variant_readback,
        },
        'same_pose_camera': True,
        'same_simulator_step': True,
        'serialized_state_parity': {
            'passed': True,
            'before_sha256': before_digest,
            'after_sha256': after_digest,
            'scope': 'scene.save before variant and after nested original material restore',
        },
        'restore': {
            'variant': variant_restore,
            'original_source_input_readback': restored_readback,
            'registered_material_reset_preserved': True,
        },
        'oracle_actions_executed': 0,
        'physics_steps_executed': 0,
        'visual_effect_claim': 'UNPROVEN_PARAMETER_EFFECT',
    }


def _variant_base_block_reason(base: Mapping[str, Any]) -> str | None:
    """A valid initial state, not Base Oracle success, is the dependency."""
    if base.get('resident_state_reusable') is False:
        return 'base_resident_session_invalidated'
    if base.get('initial_state_status') == 'PASS' or base.get('marker_action_gate_attempted') is True:
        return None
    return 'base_initial_state_unavailable'


class OracleCallbackProtocolError(ValueError):
    pass


def prepare_episode_initial_state(provider, *, evaluator, control, bindings, task):
    """One initial-state interface; existing providers own construct/restore facts."""
    report = dict(provider(evaluator=evaluator, control=control, bindings=bindings, task=task))
    if report.get('construction_pass') is not True:
        from .registered_l0_task_runtime import RegisteredL0ConstructionError
        raise RegisteredL0ConstructionError('initial state provider did not pass', report)
    return report

def _explicit_pair_base_id(*, task: Any, template: Mapping[str, Any], construction_report: Mapping[str, Any]) -> str | None:
    value = getattr(task, "pair_base_task_spec_id", None)
    if value is None:
        value = template.get("pair_base_task_spec_id")
    if value is None:
        value = construction_report.get("pair_base_task_spec_id")
    if value is None:
        paired = construction_report.get("paired_initial_state")
        if isinstance(paired, Mapping) and paired.get("private") is True:
            value = paired.get("baseline_task_spec_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RuntimeError("PAIR_CONSISTENCY_ERROR: pair_base_task_spec_id is invalid")
    return value


def _physical_pair_digest(*, task: Any, template: Mapping[str, Any], construction_report: Mapping[str, Any]) -> str | None:
    paired = construction_report.get("paired_initial_state")
    if isinstance(paired, Mapping) and isinstance(paired.get("physical_contract_sha256"), str):
        return paired["physical_contract_sha256"]
    for key in ("physical_contract_sha256", "pair_physical_contract_sha256", "template_contract_sha256"):
        value = template.get(key)
        if isinstance(value, str) and value:
            return value
    value = getattr(task, "physical_contract_sha256", None)
    return value if isinstance(value, str) and value else None


def _runtime_vector_like(template: Any, values: Sequence[float]) -> Any:
    factory = getattr(template, "new_tensor", None)
    return factory(values) if callable(factory) else values


def _validated_prepared_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise RuntimeError(f"prepared initial {label} readback is malformed")
    values = [float(item) for item in value]
    if not all(math.isfinite(item) for item in values):
        raise RuntimeError(f"prepared initial {label} readback is non-finite")
    return values


def _validate_prepared_robot_record(prepared: Mapping[str, Any]) -> None:
    _validated_prepared_vector(prepared.get("robot_position"), 3, "position")
    orientation = _validated_prepared_vector(prepared.get("robot_orientation"), 4, "orientation")
    if math.sqrt(sum(item * item for item in orientation)) == 0:
        raise RuntimeError("prepared initial orientation is a zero quaternion")
    joints = prepared.get("robot_joint_positions")
    if not isinstance(joints, (list, tuple)) or not joints or not all(math.isfinite(float(item)) for item in joints):
        raise RuntimeError("prepared initial joint readback is malformed")


def _capture_prepared_initial_state(*, evaluator: Any, control: Any, profile: Any, run_dir: Path,
                                    item_id: str, initial_pose_validation_callback: Any | None,
                                    task: Any, existing_frame: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if initial_pose_validation_callback is not None and not initial_pose_validation_callback(
        evaluator=evaluator, task=task
    ):
        raise RuntimeError("prepared initial canonical semantics are not valid")
    robot = getattr(evaluator, "robot", None)
    if robot is None or not callable(getattr(robot, "get_position_orientation", None)):
        raise RuntimeError("prepared initial robot readback is unavailable")
    get_joints = getattr(robot, "get_joint_positions", None)
    if not callable(get_joints) or not callable(getattr(robot, "set_joint_positions", None)):
        raise RuntimeError("prepared initial joint readback/setter is unavailable")
    capture_started = monotonic()
    frame = dict(existing_frame) if existing_frame is not None else capture_construction_rgb(
        evaluator=evaluator,
        control=control,
        profile=profile,
        path=run_dir / "resident_tasks" / item_id / "prepared_initial.png",
    )
    if frame.get("status") != "CAPTURED" or not frame.get("path"):
        raise RuntimeError("prepared initial RGB capture did not complete")
    if not isinstance(frame.get("head_camera_readback"), Mapping):
        raise RuntimeError("prepared initial camera readback is unavailable")
    position, orientation = robot.get_position_orientation()
    joints = get_joints()
    joint_positions = joints.tolist() if hasattr(joints, "tolist") else list(joints)
    prepared = {
        "schema_version": "resident-prepared-initial-state/v1",
        "task_name": item_id,
        "first_frame_seconds": monotonic() - capture_started,
        "first_frame_reused": existing_frame is not None,
        "visual_review": "NOT_RUN",
        "robot_position": (position.tolist() if hasattr(position, "tolist") else list(position))
        if position is not None else None,
        "robot_orientation": (orientation.tolist() if hasattr(orientation, "tolist") else list(orientation))
        if orientation is not None else None,
        "robot_joint_positions": joint_positions,
        "head_camera_readback": frame.get("head_camera_readback"),
        "first_frame": frame.get("path"),
        "oracle_actions_executed": 0,
        "physics_steps_executed": 0,
    }
    _validate_prepared_robot_record(prepared)
    state_path = run_dir / "private" / "resident_tasks" / item_id / "prepared_initial_state.json"
    write_json_atomic(state_path, prepared)
    return {
        "state_path": str(state_path.resolve()),
        "frame_path": (
            str(Path(frame["path"]).resolve())
            if frame.get("path") else None
        ),
        "prepared": prepared,
    }


def _reuse_prepared_robot_pose(*, evaluator: Any, prepared: Mapping[str, Any]) -> None:
    robot = getattr(evaluator, "robot", None)
    if robot is None or not callable(getattr(robot, "get_position_orientation", None)):
        raise RuntimeError("paired prepared robot readback is unavailable")
    if not callable(getattr(robot, "set_position_orientation", None)):
        raise RuntimeError("paired prepared robot pose setter is unavailable")
    _validate_prepared_robot_record(prepared)
    position = prepared["robot_position"]
    orientation = prepared["robot_orientation"]
    current_position, current_orientation = robot.get_position_orientation()
    robot.set_position_orientation(
        _runtime_vector_like(current_position, position),
        _runtime_vector_like(current_orientation, orientation),
    )
    joint_positions = prepared.get("robot_joint_positions")
    set_joints = getattr(robot, "set_joint_positions", None)
    if not callable(set_joints) or not callable(getattr(robot, "get_joint_positions", None)):
        raise RuntimeError("paired prepared robot joint setter/readback is unavailable")
    current_joints = robot.get_joint_positions()
    set_joints(_runtime_vector_like(current_joints, joint_positions))


def _validate_reused_presentation(presentation: Mapping[str, Any]) -> None:
    if not isinstance(presentation, Mapping):
        raise RuntimeError("paired presentation configuration is malformed")
    statuses = []
    if "status" in presentation:
        statuses.append(presentation["status"])
    preflight = presentation.get("candidate_preflight")
    if isinstance(preflight, Mapping) and "status" in preflight:
        statuses.append(preflight["status"])
    if any(status in {"FAIL", "FAILED", "ERROR"} for status in statuses):
        raise RuntimeError("paired presentation configuration is not valid")


def make_resident_scene_batch_callback(
    *,
    tasks: Sequence[ResidentTaskSpec],
    profiles: Sequence[Any],
    templates: Sequence[Mapping[str, Any]],
    union_template: Mapping[str, Any],
    seed: int,
    filter_embedded_scene_agents: bool,
    construction_callback: Any | None = None,
    navigation_audit_callback: Any | None = None,
    oracle_callback: Any | None = None,
    item_ids: Sequence[str] | None = None,
    oracle_callbacks: Sequence[Any] | None = None,
    before_task_reset: Any | None = None,
    reset_pose_records: Mapping[str, Mapping[str, Any]] | None = None,
    oracle_timeout_seconds: float | None = None,
    initial_construction_timeout_seconds: float | None = None,
    construction_only: bool = False,
    present_inward_grippers: bool = False,
    nearby_initial_pose_search: bool = False,
    initial_pose_validation_callback: Any | None = None,
    candidate_preflight_callback: Any | None = None,
    variant_base_item_ids: Mapping[str, str] | None = None,

    effective_execution_contract: Mapping[str, Any] | None = None,
    l0_visual_variant_dependencies: Mapping[str, str] | None = None,
    model_callback: Any | None = None,
    model_prepared_states: Mapping[str, Any] | None = None,
    initial_task_facts_callback: Any | None = None,
    native_task_metric: bool = True,
) -> Any:
    """Run each task sequentially in one resident evaluator."""
    if type(native_task_metric) is not bool:
        raise TypeError("native_task_metric must be boolean")
    if model_prepared_states is not None and (model_callback is None or nearby_initial_pose_search):
        raise ValueError('recorded preparation requires a model and forbids pose search')

    if not isinstance(present_inward_grippers, bool):
        raise TypeError("present_inward_grippers must be bool")
    if candidate_preflight_callback is not None and (
        not callable(candidate_preflight_callback) or not present_inward_grippers or construction_only
    ):
        raise ValueError("candidate preflight requires the execution presentation phase")
    if nearby_initial_pose_search and (
        not present_inward_grippers
        or (not reset_pose_records and construction_callback is None)
        or (construction_callback is not None and not callable(initial_pose_validation_callback))
    ):
        raise ValueError("nearby initial search requires a fixed reset or validated registered construction")
    if construction_only and present_inward_grippers:
        raise ValueError("geometry compilation must defer execution presentation")

    if oracle_timeout_seconds is not None and (
        not math.isfinite(oracle_timeout_seconds) or oracle_timeout_seconds <= 0
    ):
        raise ValueError("Oracle timeout must be positive and finite")
    if initial_construction_timeout_seconds is not None and (
            not math.isfinite(initial_construction_timeout_seconds) or initial_construction_timeout_seconds <= 0):
        raise ValueError('construction timeout must be positive and finite')
    if not tasks or len(tasks) != len(profiles) or len(tasks) != len(templates):
        raise ValueError("resident batch inputs must be non-empty and aligned")
    scene_models = {task.scene_model for task in tasks}
    if len(scene_models) != 1:
        raise ValueError("resident batch must contain exactly one scene model")
    if any(profile.task_name != task.task_name for profile, task in zip(profiles, tasks)):
        raise ValueError("resident profiles do not match task specs")
    if construction_callback is not None and not callable(construction_callback):
        raise TypeError("construction_callback must be callable")
    if construction_only and construction_callback is None:
        raise ValueError('construction-only mode requires a construction callback')
    if navigation_audit_callback is not None and not callable(
        navigation_audit_callback
    ):
        raise TypeError("navigation_audit_callback must be callable")
    if oracle_callback is not None and not callable(oracle_callback):
        raise TypeError("oracle_callback must be callable")
    if oracle_callback is not None and oracle_callbacks is not None:
        raise ValueError("use one resident Oracle callback mode")
    resolved_item_ids = tuple(
        task.task_name for task in tasks
    ) if item_ids is None else tuple(item_ids)
    if len(resolved_item_ids) != len(tasks) or any(
        not item_id or Path(item_id).parts != (item_id,)
        for item_id in resolved_item_ids
    ):
        raise ValueError("resident item IDs must be aligned safe path components")
    variant_bases = dict(variant_base_item_ids or {})
    if construction_only and variant_bases:
        raise ValueError('Base Oracle gate is not a construction-only gate')
    for variant, base in variant_bases.items():
        if (variant not in resolved_item_ids or base not in resolved_item_ids
                or resolved_item_ids.index(base) >= resolved_item_ids.index(variant)
                or base in variant_bases):
            raise ValueError('variant requires its Base earlier in the same resident batch')
    resolved_oracle_callbacks = (
        None if oracle_callbacks is None else tuple(oracle_callbacks)
    )
    if before_task_reset is not None and not callable(before_task_reset):
        raise TypeError("before_task_reset must be callable")
    l0_visual_variant_dependencies = dict(l0_visual_variant_dependencies or {})
    if any(variant not in resolved_item_ids or not isinstance(base, str) or not base
           or base == variant for variant, base in l0_visual_variant_dependencies.items()):
        raise ValueError("invalid L0 visual Variant dependency")
    if reset_pose_records is not None and set(reset_pose_records) != set(resolved_item_ids):
        raise ValueError("fixed reset poses must align with resident item IDs")
    if resolved_oracle_callbacks is not None and (
        len(resolved_oracle_callbacks) != len(tasks)
        or any(not callable(callback) for callback in resolved_oracle_callbacks)
    ):
        raise ValueError("resident Oracle callbacks must align with tasks")
    for supplied in resolved_oracle_callbacks or (() if oracle_callback is None else (oracle_callback,)):
        # Validate the real interface at setup, before any scene is loaded.
        inspect.signature(supplied).bind(evaluator=None, control=None, bindings=None,
                                         run_dir=None, stages=None)

    def callback(
        *,
        evaluator: Any,
        control: Any,
        bindings: Any,
        run_dir: Path,
        stages: Path,
    ) -> Mapping[str, Any]:
        if evaluator.env.task.activity_name != tasks[0].task_name:
            raise RuntimeError("resident primary task does not match loaded evaluator")
        protocol_path = _write_resident_task_protocol(
            run_dir=run_dir, tasks=tasks, item_ids=resolved_item_ids,
            campaign_seed=seed,
            effective_execution_contract=effective_execution_contract,
        )
        runtime_superset = evaluator.env.scene.save()
        if isinstance(runtime_superset, str):
            runtime_superset = json.loads(runtime_superset)
        task_results: list[dict[str, Any]] = []
        restore_results: list[dict[str, Any]] = []
        construction_results: list[dict[str, Any]] = []
        navigation_results: list[dict[str, Any]] = []
        capture_paths = []
        def skip_variant(index: int, base_id: str, reason: str) -> None:
            item_id = resolved_item_ids[index]
            base = task_results[resolved_item_ids.index(base_id)]
            row = dict(status='SKIPPED_BASE_BLOCKED', official_q=None, oracle_status='NOT_RUN',
                       reason=reason, blocked_by_base=base_id, base_status=base.get('status'),
                       base_failure_class=base.get('failure_class'), resident_item_id=item_id,
                       resident_task_index=index, actions_executed=0, oracle_execution_seconds=0,
                       route_gate_attempted=False, marker_action_gate_attempted=False,
                       base_result_path=str(run_dir/'private/resident_tasks'/base_id/'oracle_result.json'))
            task_results.append(row)
            row.update(commit_episode_result(run_dir, item_id, row))
            _record(stages, 'VARIANT_SKIPPED_BASE_BLOCKED', **row)

        pair_preparation_cache: dict[tuple[str, str], dict[str, Any]] = {}
        pair_preparation_digests: dict[str, str] = {}
        from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
            _minimal_four_boundary_oracle_phase,
        )

        def run_resident_task_episode(index, task, profile, template):
            item_id = resolved_item_ids[index]
            record_phase(run_dir, item_id, 'DEPENDENCY_CHECK')
            base_id = variant_bases.get(item_id)
            if base_id is not None:
                reason = _variant_base_block_reason(task_results[resolved_item_ids.index(base_id)])
                if reason is not None:
                    skip_variant(index, base_id, reason)
                    return
            progress = _ResidentTaskProgress(
                run_dir=run_dir, item_id=item_id, campaign_seed=seed,
            )
            record_phase(run_dir, item_id, 'RESTORE')
            try:
                if before_task_reset is not None:
                    before_task_reset(item_id)
                if index:
                    restore_results.append(
                        switch_resident_task(
                            evaluator=evaluator,
                            control=control,
                            task=task,
                            union_template=union_template,
                            target_template=template,
                            runtime_superset=runtime_superset,
                            filter_embedded_scene_agents=filter_embedded_scene_agents,
                            run_dir=run_dir,
                            stages=stages,
                            artifact_key=item_id,
                            native_task_metric=native_task_metric,
                        )
                    )
            except Exception as exc:
                task_results.append(commit_episode_result(run_dir, item_id, {
                    'status': 'FAIL', 'failure_class': 'RESTORE_ERROR',
                    'resident_item_id': item_id, 'oracle_started': False,
                    'oracle_status': 'NOT_RUN', 'resident_state_reusable': False,
                    'error': f'{type(exc).__name__}: {exc}', 'failed_phase': 'TASK_RESTORE',
                }))
                progress.update("TERMINAL", oracle_started=False, terminal_status="FAIL")
                raise
            if model_prepared_states is not None:
                # Frozen camera-relative conditions must be checked from the
                # recorded view, not the loader's temporary default robot pose.
                from .prepared_model_state import restore_prepared_model_state
                report = restore_prepared_model_state(evaluator=evaluator,
                    bundle=model_prepared_states[item_id], instruction_id=item_id,
                    seed=seed, observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE)
                write_json_atomic(run_dir / 'private/resident_tasks' / item_id / 'prepared_view_restore.json', report)
            progress.update("CONSTRUCTING")
            construction_report: dict[str, Any] = {}
            skip_construction = (
                model_prepared_states is not None
                and (effective_execution_contract or {}).get('execution_mode') == 'PREPARED_MODEL_REPLAY'
            )
            if construction_callback is not None and not skip_construction:
                record_phase(run_dir, item_id, 'INITIAL_STATE_CONSTRUCTION')
                construction_path = (
                    run_dir
                    / "private"
                    / "resident_tasks"
                    / item_id
                    / "live_initial_construction.json"
                )
                try:
                    from .construction_progress import construction_scope
                    with construction_scope(run_dir, item_id, initial_construction_timeout_seconds):
                        construction_report = prepare_episode_initial_state(
                            construction_callback, evaluator=evaluator, control=control,
                            bindings=bindings, task=task)
                    if construction_only and construction_report.get('construction_pass') is True:
                        # Preserve the physical capture even if rendering fails.
                        write_json_atomic(construction_path, construction_report)
                        construction_report['construction_rgb'] = capture_construction_rgb(
                            evaluator=evaluator, control=control, profile=profile,
                            path=run_dir / 'resident_tasks' / item_id / 'construction_initial.png')
                        probe = construction_report.get('calibration_probe', {})
                        if probe.get('render_transform'):
                            construction_report['visual_probe'] = capture_light_probe(
                                evaluator=evaluator, control=control, profile=profile,
                                path=run_dir / 'resident_tasks' / item_id / 'construction_variant.png',
                                rule=probe['render_transform'])
                        if _material_diagnostic_requested(
                            construction_only=construction_only,
                            construction_report=construction_report,
                            probe=probe,
                        ):
                            try:
                                construction_report['material_diagnostic_probe'] = capture_material_paired_probe(
                                    evaluator=evaluator,
                                    control=control,
                                    profile=profile,
                                    path=run_dir / 'resident_tasks' / item_id / 'construction_material_complement.png',
                                    baseline_frame=construction_report['construction_rgb'],
                                )
                            except Exception as exc:
                                from behavior_hl.embodied.evaluator_bridge.registered_l0_task_runtime import (
                                    RegisteredL0ConstructionError,
                                )
                                failure_report = {
                                    **construction_report,
                                    'status': 'FAIL',
                                    'construction_pass': False,
                                    'failure_class': 'MATERIAL_DIAGNOSTIC_ERROR',
                                    'material_diagnostic_error': f'{type(exc).__name__}: {exc}',
                                }
                                if getattr(exc, 'session_invalidated', False):
                                    failure_report['resident_session_invalidated'] = True
                                wrapped = RegisteredL0ConstructionError(
                                    'material paired diagnostic failed', failure_report
                                )
                                if getattr(exc, 'session_invalidated', False):
                                    wrapped.session_invalidated = True
                                raise wrapped from exc
                except Exception as exc:
                    failure_report = getattr(exc, "report", None)
                    diagnostic = {}
                    if construction_only:
                        try:
                            diagnostic['diagnostic_rgb'] = capture_construction_rgb(
                                evaluator=evaluator, control=control, profile=profile,
                                path=run_dir / 'resident_tasks' / item_id / 'construction_failed.png')
                        except Exception as capture_error:
                            diagnostic['diagnostic_rgb_error'] = f'{type(capture_error).__name__}: {capture_error}'
                    failure_report = {
                        **(dict(failure_report) if isinstance(failure_report, Mapping) else {}),
                        **diagnostic,
                        "status": "FAIL",
                        "construction_pass": False,
                        "error": (failure_report.get("error") or str(exc)
                                  if isinstance(failure_report, Mapping) else str(exc)),
                    }
                    write_json_atomic(construction_path, failure_report)
                    construction_results.append(dict(failure_report))
                    # A construction exception is not evidence of asset infeasibility,
                    # including when construction and Oracle share one scene load.
                    default_failure_class = 'INITIAL_STATE_COMPILATION_ERROR'
                    task_result = {
                        "status": (failure_report.get('status') if isinstance(failure_report, Mapping)
                                   and failure_report.get('status') == 'SKIPPED_BASE_BLOCKED' else 'FAIL'),
                        "failure_class": (failure_report.get('failure_class', default_failure_class)
                            if isinstance(failure_report, Mapping) else default_failure_class),
                        "failed_phase": "CONSTRUCTION",
                        "construction_status": "FAIL",
                        "oracle_status": "NOT_RUN",
                        "construction_evidence": str(construction_path.resolve()),
                        "construction_error_type": type(exc).__name__,
                        **exception_evidence(exc),
                        "initial_state_status": failure_report.get('initial_state_status', 'FAIL')
                            if isinstance(failure_report, Mapping) else 'FAIL',
                        "construction_error": str(exc),
                        "error": failure_report['error'],
                        "route_gate_attempted": False,
                        "marker_action_gate_attempted": False,
                        "oracle_started": False,
                        "resident_item_id": item_id,
                        "resident_task_index": index,
                        **diagnostic,
                    }
                    if isinstance(failure_report, Mapping) and failure_report.get('blocked_by_base'):
                        task_result['blocked_by_base'] = failure_report['blocked_by_base']
                    task_results.append(task_result)
                    task_result.update(commit_episode_result(run_dir, item_id, task_result))
                    progress.update("TERMINAL", oracle_started=False, terminal_status="FAIL",
                                    failed_phase="CONSTRUCTION", construction_status="FAIL",
                                    oracle_status="NOT_RUN",
                                    construction_evidence=str(construction_path.resolve()))
                    if getattr(exc, 'session_invalidated', False) or (
                        isinstance(failure_report, Mapping)
                        and failure_report.get('resident_session_invalidated') is True
                    ) or (
                        construction_only and (
                            'USD edit detected outside of og.sim.editing_usd()' in str(exc)
                            or (isinstance(failure_report, Mapping) and failure_report.get('material_restore_error'))
                        )
                    ):
                        for later in range(index + 1, len(tasks)):
                            if variant_bases.get(resolved_item_ids[later]) == item_id:
                                skip_variant(later, item_id, 'base_resident_session_invalidated')
                        _record(stages, 'RESIDENT_SESSION_INVALIDATED',
                                      task=task.task_name, reason='construction synchronization, rollback or private sensor failure')
                        raise RuntimeError('construction invalidated resident scene; new attempt required') from exc
                    return
                write_json_atomic(construction_path, construction_report)
                construction_results.append(construction_report)

            progress.update(
                "PREPARING",
                prepared_state=(str(construction_path.resolve()) if construction_callback is not None else None),
            )
            pair_key = None
            try:
                pair_base_id = _explicit_pair_base_id(
                    task=task, template=template, construction_report=construction_report,
                )
                if pair_base_id is not None:
                    physical_digest = _physical_pair_digest(
                        task=task, template=template, construction_report=construction_report,
                    )
                    if physical_digest is None:
                        raise RuntimeError(
                            "PAIR_CONSISTENCY_ERROR: physical contract identity is unavailable"
                        )
                    previous_digest = pair_preparation_digests.get(pair_base_id)
                    if previous_digest is not None and previous_digest != physical_digest:
                        raise RuntimeError(
                            "PAIR_CONSISTENCY_ERROR: paired physical contract differs"
                        )
                    pair_preparation_digests[pair_base_id] = physical_digest
                    pair_key = (pair_base_id, physical_digest)
            except Exception as exc:
                task_result = {
                    "status": "FAIL",
                    "failure_class": "PREFLIGHT_ERROR",
                    "reason": str(exc),
                    "oracle_status": "NOT_RUN",
                    "oracle_started": False,
                    "route_gate_attempted": False,
                    "marker_action_gate_attempted": False,
                    "resident_task_index": index,
                    "resident_item_id": item_id,
                    "pair_consistency": "FAIL",
                }
                task_results.append(task_result)
                progress.update("TERMINAL", oracle_started=False, terminal_status="FAIL")
                task_result.update(commit_episode_result(run_dir, item_id, task_result))
                return
            if reset_pose_records is not None and model_prepared_states is None:
                record_phase(run_dir, item_id, 'INITIAL_STATE_VALIDATION')
                try:
                    pose_report = attest_l0_reset_pose(
                        evaluator.robot, reset_pose_records[item_id]
                    )
                except (TypeError, ValueError) as exc:
                    pose_report = {"status": "FAIL", "reason": str(exc)}
                write_json_atomic(
                    run_dir / "private" / "resident_tasks" / item_id
                    / "fixed_reset_pose_attestation.json",
                    pose_report,
                )
                if pose_report["status"] != "PASS":
                    task_result = {
                        "status": "FAIL",
                        "failure_class": "RESTORE_ERROR",
                        "reason": "invalid_fixed_reset_pose_readback",
                        "fixed_reset_pose_attestation": pose_report,
                        "route_gate_attempted": False,
                        "marker_action_gate_attempted": False,
                        "resident_task_index": index,
                        "resident_item_id": item_id,
                    }
                    task_results.append(task_result)
                    progress.update("TERMINAL", oracle_started=False, terminal_status="FAIL")
                    task_result.update(commit_episode_result(run_dir, item_id, task_result))
                    return
            if construction_only:
                prepared = _capture_prepared_initial_state(
                    evaluator=evaluator,
                    control=control,
                    profile=profile,
                    run_dir=run_dir,
                    item_id=item_id,
                    initial_pose_validation_callback=initial_pose_validation_callback,
                    task=task,
                    existing_frame=construction_report.get("construction_rgb"),
                )
                progress.update(
                    "READY",
                    prepared_state=prepared["state_path"],
                    first_frame=prepared["frame_path"],
                )
                capture_paths.append({'path': str(construction_path.resolve())})
                index_path = run_dir / 'private' / 'initial_state_capture_index.jsonl'
                index_path.parent.mkdir(parents=True, exist_ok=True)
                index_path.write_text(''.join(json.dumps(r) + '\n' for r in capture_paths))
                task_results.append(commit_episode_result(run_dir, item_id, {
                    'status': 'CONSTRUCTION_CAPTURED', 'initial_state_status': 'PASS',
                    'oracle_status': 'NOT_RUN', 'resident_task_index': index}))
                progress.update('TERMINAL', oracle_started=False, terminal_status='CONSTRUCTION_CAPTURED')
                return

            cached_preparation = pair_preparation_cache.get(pair_key) if pair_key is not None else None
            if present_inward_grippers:
                record_phase(run_dir, item_id, 'PREPARE_ROBOT')
                from behavior_hl.embodied.evaluator_bridge.moderate_inward_gripper_presentation import (
                    apply_untucked_inward_gripper_presentation,
                )
                _record(stages, "RESIDENT_GRIPPER_PRESENTATION_START", resident_item_id=item_id)
                try:
                    if model_prepared_states is not None:
                        from .prepared_model_state import restore_prepared_model_state
                        presentation = restore_prepared_model_state(
                            evaluator=evaluator, bundle=model_prepared_states[item_id],
                            instruction_id=item_id, seed=seed,
                            observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
                            semantic_check=(lambda: initial_pose_validation_callback(evaluator=evaluator, task=task))
                            if initial_pose_validation_callback is not None else None)
                    elif cached_preparation is not None:
                        _validate_reused_presentation(cached_preparation['presentation'])
                        _reuse_prepared_robot_pose(evaluator=evaluator, prepared=cached_preparation.get('prepared', {}))
                        presentation = dict(cached_preparation['presentation'])
                        presentation.update(resident_item_id=item_id, pair_preparation_reused=True,
                                            pair_base_task_spec_id=pair_key[0])
                        _record(stages, 'RESIDENT_GRIPPER_PRESENTATION_REUSED', resident_item_id=item_id)
                    elif candidate_preflight_callback is not None:
                        preflight = dict(candidate_preflight_callback(evaluator=evaluator, task=task))
                        write_json_atomic(run_dir / "private/resident_tasks" / item_id / "candidate_preflight.json", preflight)
                        presentation = dict(preflight["preparation"])
                        presentation["candidate_preflight"] = {key: value for key, value in preflight.items() if key != "preparation"}
                    elif nearby_initial_pose_search:
                        from .nearby_initial_pose import prepare_nearby_initial_pose
                        from .registered_l0_task_runtime import registered_preparation_room
                        presentation = dict(prepare_nearby_initial_pose(evaluator,
                            expected_room=registered_preparation_room(profile.task_name),
                            initial_state_valid=(lambda: initial_pose_validation_callback(evaluator=evaluator, task=task))
                            if initial_pose_validation_callback is not None else (lambda: True)))
                    else:
                        presentation = dict(apply_untucked_inward_gripper_presentation(evaluator))
                except Exception as exc:
                    # A partially restored robot is not safe to reuse. Persist
                    # this task's terminal evidence before abandoning the worker.
                    session_invalid = model_prepared_states is not None or cached_preparation is not None or bool(getattr(exc, 'session_invalid', False))
                    presentation_failure = getattr(exc, "private_report", {})
                    presentation_failure = dict(presentation_failure) if isinstance(presentation_failure, Mapping) else {}
                    presentation_failure.update(status="FAIL", error_type=type(exc).__name__,
                                                error=str(exc), resident_item_id=item_id)
                    try:
                        write_json_atomic(run_dir / "private" / "resident_tasks" / item_id
                                          / "gripper_presentation.json", presentation_failure)
                    except Exception as diagnostic_exc:
                        presentation_failure["persistence_error"] = str(diagnostic_exc)
                    task_result = dict(status="FAIL", failure_class="RESTORE_ERROR", initial_state_status="PASS",
                        reason="initial_gripper_presentation_failed", error_type=type(exc).__name__,
                        error=str(exc), resident_item_id=item_id, resident_task_index=index,
                        route_gate_attempted=False, marker_action_gate_attempted=False, official_q=None,
                        oracle_started=False, oracle_status='NOT_RUN', failed_phase='PREPARATION',
                        preparation_status='FAIL', resident_session_invalidated=session_invalid)
                    if candidate_preflight_callback is not None and cached_preparation is None:
                        task_result.update(failure_class="PREFLIGHT_ERROR", reason=str(exc), oracle_status="NOT_RUN")
                        write_json_atomic(run_dir / "private/resident_tasks" / item_id / "candidate_preflight.json", presentation_failure)
                    if "persistence_error" in presentation_failure:
                        task_result["presentation_diagnostic_persistence_error"] = presentation_failure["persistence_error"]
                    task_result["preparation_failure_type"] = presentation_failure.get("failure_type")
                    task_result["retry_scope"] = presentation_failure.get("retry_scope")
                    if task_result["preparation_failure_type"] == "RESOURCE_EXHAUSTED":
                        task_result["failure_class"] = "RESOURCE_EXHAUSTED"
                    task_results.append(task_result)
                    task_result.update(commit_episode_result(run_dir, item_id, task_result))
                    progress.update("TERMINAL", oracle_started=False, terminal_status="FAIL",
                                    oracle_status='NOT_RUN', failed_phase='PREPARATION', preparation_status='FAIL')
                    _record(stages, "RESIDENT_GRIPPER_PRESENTATION_FAILED", **task_result)
                    if session_invalid:
                        _record(stages, 'PREFLIGHT_SESSION_INVALID', resident_item_id=item_id, error=str(exc))
                        raise
                    return
                presentation.update(applied_after_task_reset=True, untucked_preamble_applied=model_prepared_states is None,
                                    resident_item_id=item_id)
                write_json_atomic(run_dir / "private" / "resident_tasks" / item_id / "gripper_presentation.json", presentation)
                _record(stages, "RESIDENT_GRIPPER_PRESENTATION_PASS", resident_item_id=item_id)

            try:
                prepared = _capture_prepared_initial_state(
                    evaluator=evaluator,
                    control=control,
                    profile=profile,
                    run_dir=run_dir,
                    item_id=item_id,
                    initial_pose_validation_callback=initial_pose_validation_callback,
                    task=task,
                )
                if initial_task_facts_callback is not None:
                    facts = initial_task_facts_callback(evaluator=evaluator, task=task)
                    if facts is not None:
                        prepared['prepared']['initial_task_facts'] = facts
                        write_json_atomic(Path(prepared['state_path']), prepared['prepared'])
            except Exception as exc:
                task_result = {
                    "status": "FAIL",
                    "failure_class": "PREFLIGHT_ERROR",
                    "reason": "prepared_initial_state_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "oracle_status": "NOT_RUN",
                    "oracle_started": False,
                    "route_gate_attempted": False,
                    "marker_action_gate_attempted": False,
                    "resident_task_index": index,
                    "resident_item_id": item_id,
                }
                task_results.append(task_result)
                progress.update("TERMINAL", oracle_started=False, terminal_status="FAIL")
                task_result.update(commit_episode_result(run_dir, item_id, task_result))
                return
            if pair_key is not None and cached_preparation is None:
                pair_preparation_cache[pair_key] = {
                    "presentation": dict(presentation) if present_inward_grippers else {},
                    "prepared": prepared["prepared"],
                    "prepared_state": prepared["state_path"],
                }
            progress.update(
                "READY",
                prepared_state=prepared["state_path"],
                first_frame=prepared["frame_path"],
            )

            navigation_report = None
            # Oracle admission requires its paired base in this sweep. Model
            # serving uses the source handoff's Oracle gate and a frozen initial
            # package; it must not rerun an Oracle before the model episode.
            base_id = ((l0_visual_variant_dependencies or {}).get(item_id)
                       if model_callback is None else None)
            if base_id is not None:
                base_result = next((row for row in reversed(task_results)
                                    if row.get("resident_item_id") == base_id), None)
                base_q1 = (base_result is not None
                           and base_result.get("oracle_started") is True
                           and base_result.get("verification_completed") is True
                           and base_result.get("goal_status") == "PASS"
                           and base_result.get("official_success") is not False
                           and not isinstance(base_result.get("official_q"), bool)
                           and base_result.get("official_q") == 1)
                if not base_q1:
                    task_result = {
                        "status": "RENDERED_ONLY", "resident_item_id": item_id,
                        "resident_task_index": index, "oracle_started": False,
                        "oracle_status": "NOT_RUN", "official_q": None,
                        "goal_status": "UNKNOWN", "failure_class": None,
                        "oracle_skip_reason": "L0_VISUAL_VARIANT_BASE_NO_VERIFIED_Q1",
                        "base_task_spec_id": base_id,
                        "base_evidence_scope": "current_resident_attempt",
                        "base_result_available": base_result is not None,
                        "base_result_path": str(run_dir / "private/resident_tasks" / base_id / "oracle_result.json"),
                        "construction_status": "PASS", "preparation_status": "PASS",
                        "prepared_state": prepared["state_path"], "first_frame": prepared["frame_path"],
                        "route_gate_attempted": False, "marker_action_gate_attempted": False,
                    }
                    task_result.update(commit_episode_result(run_dir, item_id, task_result))
                    task_results.append(task_result)
                    progress.update("TERMINAL", oracle_started=False, oracle_status="NOT_RUN",
                                    terminal_status="RENDERED_ONLY", oracle_skip_reason=task_result["oracle_skip_reason"])
                    return
            if model_callback is not None:
                # Model serving starts only after the same prepared-state gates
                # as the Oracle. Never execute or report an Oracle for this lane.
                result = model_callback(
                    item_id=item_id, profile=profile, evaluator=evaluator,
                    control=control, bindings=bindings, run_dir=run_dir,
                    stages=stages, prepared=prepared,
                    **({'initial_state_validator':
                        (lambda: initial_pose_validation_callback(evaluator=evaluator, task=task))
                        if initial_pose_validation_callback is not None else None}
                       if model_prepared_states is not None else {}),
                )
                task_result = {
                    "resident_item_id": item_id, "execution_phase": "MODEL",
                    "oracle_started": False, "status": "MODEL_SERVER_STOPPED",
                    "prepared_state": prepared["state_path"],
                    "first_frame": prepared["frame_path"], "result": result,
                }
                write_json_atomic(run_dir / "private/resident_tasks" / item_id / "model_server_result.json", task_result)
                task_results.append(task_result)
                return
            if navigation_audit_callback is not None:
                navigation_path = (
                    run_dir
                    / "private"
                    / "resident_tasks"
                    / item_id
                    / "navigation_route_audit.json"
                )
                try:
                    navigation_report = dict(
                        navigation_audit_callback(
                            evaluator=evaluator,
                            task=task,
                        )
                    )
                except Exception as exc:
                    navigation_report = {
                        "status": "FAIL",
                        "route_exists": False,
                        "failure_class": "NAVIGATION_ERROR",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                write_json_atomic(navigation_path, navigation_report)
                navigation_results.append(navigation_report)
            phase_started = monotonic()
            robot = getattr(evaluator.env, 'robots', [getattr(evaluator, 'robot', None)])[0]
            initial_pose = None
            if robot is not None:
                position, orientation = robot.get_position_orientation()
                initial_pose = {'position': [float(v) for v in position],
                                'orientation_xyzw': [float(v) for v in orientation]}
                write_json_atomic(run_dir/'private/resident_tasks'/item_id/'execution_initial_pose.json', initial_pose)
            record_phase(run_dir, item_id, 'RUN_ORACLE')
            active_oracle_callback = (
                resolved_oracle_callbacks[index]
                if resolved_oracle_callbacks is not None
                else oracle_callback
            )
            if active_oracle_callback is None:
                phase_name = (
                    f"resident_tasks/{index:02d}_{item_id}/"
                    "minimal_four_boundary_full_oracle_g1"
                )
            else:
                phase_name = public_oracle_phase_name(index, item_id, len(tasks))
            execution_dir = str((run_dir / "strategies" / phase_name).resolve())
            executor_actions_path = str(
                (run_dir / "strategies" / phase_name / "private" / "executor_actions.jsonl").resolve()
            )
            phase_started = monotonic()
            if active_oracle_callback is None:
                result = _run_oracle_with_deadline(
                    _minimal_four_boundary_oracle_phase,
                    timeout_seconds=oracle_timeout_seconds,
                    item_id=item_id,
                    evaluator=evaluator,
                    control=control,
                    run_dir=run_dir,
                    stages=stages,
                    progress_update=progress.update,
                    execution_phase_name=phase_name,
                    seed=seed,
                    profile=profile,
                    phase_name=phase_name,
                    observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
                    suppress_held_object_environment_collisions=False,
                    logical_inventory_carry=True,
                    semantic_only_place_inside=False,
                    semantic_execution_profile=SEMANTIC_RESEARCH_V1,
                )
            else:
                try:
                    raw_receipt = _run_oracle_with_deadline(
                            active_oracle_callback,
                            timeout_seconds=oracle_timeout_seconds,
                            item_id=item_id,
                            evaluator=evaluator,
                            control=control,
                            bindings=bindings,
                            run_dir=run_dir,
                            stages=stages,
                            progress_update=progress.update,
                            execution_phase_name=phase_name,
                        )
                    if not isinstance(raw_receipt, Mapping):
                        raise OracleCallbackProtocolError('custom oracle returned no mapping receipt')
                    receipt = dict(raw_receipt)
                    strategies = receipt.get("strategies")
                    if (
                        not isinstance(strategies, list)
                        or len(strategies) != 1
                        or not isinstance(strategies[0], Mapping)
                    ):
                        raise OracleCallbackProtocolError(
                            "custom oracle returned no single strategy"
                        )
                    q = strategies[0].get("official_q")
                    if q is not None and (
                        isinstance(q, bool) or not isinstance(q, (int, float))
                        or not math.isfinite(q) or not 0 <= q <= 1
                    ):
                        raise OracleCallbackProtocolError("custom oracle returned invalid official Q")
                    result = {
                        **dict(strategies[0]),
                        "public_action20_marker_oracle_receipt": receipt,
                    }
                except Exception as exc:
                    result = {
                        "status": "FAIL_EXCEPTION",
                        "oracle_execution_status": "ERROR",
                        "evidence_status": "INCOMPLETE",
                        "failure_class": "UNKNOWN",
                        **exception_evidence(exc),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    if isinstance(exc, OracleCallbackProtocolError):
                        result.update(failure_owner='RUNTIME_CODE', failure_code='INVALID_CALLBACK_ENVELOPE',
                                      failure_stage='ORACLE_RESULT_DECODING')
            marker_action_pass = result.get("status") == "PASS"
            marker_failure_class = None
            if not marker_action_pass:
                marker_failure_class = str(
                    result.get("failure_class") or "UNKNOWN"
                ).upper()
                if marker_failure_class not in _FAILURE_CLASSES:
                    marker_failure_class = "UNKNOWN"
            route_status = ('NOT_RUN' if navigation_report is None else 'PASS'
                if navigation_report.get('route_exists') is True else 'FAIL'
                if navigation_report.get('route_exists') is False else 'ERROR')
            route_pass = True if route_status == 'PASS' else False if route_status == 'FAIL' else None
            admission_pass = marker_action_pass and route_status in {'PASS', 'NOT_RUN'}
            task_result = {
                **dict(result),
                "status": "PASS" if marker_action_pass else "FAIL",
                "failure_class": (
                    None
                    if marker_action_pass
                    else marker_failure_class
                ),
                "admission_failure_class": ('NAVIGATION_ERROR' if route_status == 'FAIL' else
                    'NAVIGATION_AUDIT_ERROR' if route_status == 'ERROR' else marker_failure_class),
                "route_gate_attempted": navigation_report is not None,
                "route_gate_pass": route_pass,
                "route_gate_status": route_status,
                "admission_status": 'PASS' if admission_pass else 'FAIL',
                "navigation_route_audit": navigation_report,
                "marker_action_gate_attempted": True,
                "marker_action_gate_pass": marker_action_pass,
                "marker_action_failure_class": marker_failure_class,
                "marker_action_oracle_status": result.get("status"),
                "admission_double_gate_pass": admission_pass,
                "resident_task_index": index,
                "resident_item_id": item_id,
                "oracle_execution_seconds": monotonic() - phase_started,
                "execution_initial_pose": initial_pose,
                "initial_state_identity": (construction_report.get('paired_initial_state')
                    or {'mode': construction_report.get('initialization_mode')})
                    if construction_callback is not None else None,
                "phase_name": phase_name,
                "oracle_started": True,
                "execution_dir": execution_dir,
                "executor_actions_path": executor_actions_path,
            }
            task_results.append(task_result)
            task_result.update(commit_episode_result(run_dir, item_id, task_result))

            progress.update(
                "TERMINAL",
                oracle_started=True,
                terminal_status=task_result["status"],
                execution_dir=execution_dir,
                executor_actions_path=executor_actions_path,
            )

        for index, (task, profile, template) in enumerate(zip(tasks, profiles, templates)):
            try:
                run_resident_task_episode(index, task, profile, template)
                if index == 0 and model_callback is None and _physical_canary_failed(effective_execution_contract, task_results[-1]):
                    write_json_atomic(run_dir / 'private/physical_canary_stop.json', {
                        'reason': 'FIRST_TASK_PHYSICAL_CANARY_NOT_CERTIFIED',
                        'completed_task_id': resolved_item_ids[index],
                        'not_run_task_ids': list(resolved_item_ids[index + 1:]),
                    })
                    break
                if model_callback is not None and task_results[-1].get('result', {}).get('stop_batch'):
                    break
            except Exception as exc:
                item_id = resolved_item_ids[index]
                if model_callback is not None:
                    write_json_atomic(run_dir / "private/resident_tasks" / item_id / "model_server_error.json",
                                      {"execution_phase": "MODEL", "oracle_started": False,
                                       "error_type": type(exc).__name__, "error": str(exc)})
                    raise
                receipt = run_dir/'private/resident_tasks'/item_id/'oracle_result.json'
                if not receipt.exists():
                    task_results.append(commit_episode_result(run_dir, item_id, dict(
                        status='FAIL', official_q=None, error=f'{type(exc).__name__}: {exc}',
                        **exception_evidence(exc),
                        resident_state_reusable=False, resident_item_id=item_id)))
                raise  # The session is unsafe; preserve completed tasks, stop this worker.
        if model_callback is not None:
            return {"execution_phase": "MODEL", "private_oracle_executed": False,
                    "resident_task_results": task_results,
                    "resident_restore_results": restore_results,
                    "scene_loads_attempted": 1, "successful_scene_loads": 1}
        admissions = [row.get('admission_status', 'NOT_RUN') for row in task_results]
        admission_status = ('NOT_RUN' if all(s == 'NOT_RUN' for s in admissions) else 'FAIL'
                            if 'FAIL' in admissions else 'PASS' if all(s == 'PASS' for s in admissions) else 'PARTIAL')
        admission_all_pass = True if admission_status == 'PASS' else False if admission_status == 'FAIL' else None
        executions = [row['oracle_execution_status'] for row in task_results]
        entered = sum(row.get('oracle_entered') is True for row in task_results)
        oracle_status = ('NOT_RUN' if not entered else 'PARTIAL' if entered < len(tasks)
                         else 'COMPLETED' if all(s == 'COMPLETED' for s in executions) else 'INCOMPLETE')
        route_statuses = [row.get('route_gate_status', 'NOT_RUN') for row in task_results]
        route_status = ('NOT_RUN' if all(s == 'NOT_RUN' for s in route_statuses) else 'ERROR'
            if 'ERROR' in route_statuses else 'FAIL' if 'FAIL' in route_statuses else 'PASS'
            if all(s == 'PASS' for s in route_statuses) else 'PARTIAL')
        route_all_pass = True if route_status == 'PASS' else False if route_status == 'FAIL' else None
        goals = [row['goal_status'] for row in task_results]
        official_success = False if 'FAIL' in goals else None if 'UNKNOWN' in goals else True
        marker_all_pass = len(task_results) == len(tasks) and all(
            result.get("marker_action_gate_pass") is True
            for result in task_results
        )
        oracle_started_count = sum(
            1 for result in task_results if result.get("oracle_started") is True
        )
        return {
            "schema_version": "behavior2026.resident-scene-task-sweep/v1",
            "execution_phase": "INITIAL_STATE_COMPILATION" if construction_only else "ORACLE",
            "construction_capture_success": (all(r.get('status') == 'CONSTRUCTION_CAPTURED' for r in task_results) if construction_only else None),
            "oracle_status": oracle_status,
            "oracle_entered_count": entered,
            "oracle_execution_status_counts": {s: executions.count(s) for s in set(executions)},
            "goal_status_counts": {s: goals.count(s) for s in set(goals)},
            "admission_status": admission_status,
            "admission_status_counts": {s: admissions.count(s) for s in set(admissions)},
            "evidence_status_counts": {s: sum(row.get('evidence_status') == s for row in task_results)
                                       for s in {row.get('evidence_status') for row in task_results}},
            "q1_count": goals.count('PASS'),
            "oracle_started_count": oracle_started_count,
            "rendered_only_count": sum(row.get("status") == "RENDERED_ONLY" for row in task_results),
            "l0_visual_variant_dependencies": l0_visual_variant_dependencies,
            "resident_task_protocol": str(protocol_path.resolve()),
            "initial_state_capture_index": str(run_dir / 'private' / 'initial_state_capture_index.jsonl') if construction_only and capture_paths else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": tasks[0].task_name,
            "resident_scene_model": tasks[0].scene_model,
            "resident_task_count": len(tasks),
            "variant_base_gate_version": 'base-oracle-admission/v1' if variant_bases else None,
            "base_blocked_variant_count": sum(r.get('status') == 'SKIPPED_BASE_BLOCKED' for r in task_results),
            "resident_item_ids": list(resolved_item_ids),
            "scene_loads_attempted": 1,
            "successful_scene_loads": 1,
            "cross_task_state_restore_count": len(restore_results),
            "in_memory_resets": len(restore_results),
            "strategies": task_results,
            "resident_task_results": task_results,
            "resident_restore_results": restore_results,
            "resident_construction_results": construction_results,
            "resident_navigation_results": navigation_results,
            "route_gate_success": route_all_pass,
            "route_gate_status": route_status,
            "marker_action_oracle_success": marker_all_pass if entered else None,
            "oracle_success": marker_all_pass if entered else None,
            "official_success": official_success,
            "admission_double_gate_success": admission_all_pass,
            "all_tasks_passed": official_success,
            "teleport_enabled_inside_symbolic_primitive": True,
            "official_challenge_eligible": False,
            "strict_production_eligible": False,
            "kinematic_research": True,
        }

    return callback
