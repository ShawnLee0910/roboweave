"""Launch several same-scene Oracles in one resident Kit process."""

from __future__ import annotations

import argparse
from dataclasses import replace
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (
    get_inside_task_profile,
)
from behavior_hl.embodied.evaluator_bridge.construction_camera_pose import (
    CAMERA_CAPTURE_SCHEMA,
    camera_pose_at_robot_pose,
    capture_head_camera_pose,
)
from behavior_hl.embodied.evaluator_bridge.launch_b1_single_load import (
    build_production_bindings,
)
from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
)
from behavior_hl.embodied.policy.observations import CAMERA_SENSOR_NAMES
from behavior_hl.embodied.evaluator_bridge.l0_taskspec_goal_plan import (
    make_l0_taskspec_action20_oracle_callback,
)
from behavior_hl.embodied.evaluator_bridge.l0_reset_pose_overlay import (
    configure_official_l0_reset_pose_runtime,
    load_l0_reset_pose_overlay,
    patch_scene_template_reset_pose,
    patch_tro_state_reset_pose,
)
from behavior_hl.embodied.evaluator_bridge.resident_scene_batch import (
    ResidentTaskSpec,
    build_union_scene_document,
    load_scene_document,
    write_json_atomic,
)
from behavior_hl.embodied.evaluator_bridge.resident_scene_live import (
    make_resident_scene_batch_callback,
    public_oracle_phase_name,
)
from behavior_hl.embodied.evaluator_bridge.resident_scene_state import (
    install_native_initial_state_completion,
    native_initial_state_completion_report,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    B1SingleLoadWorkerSpec,
    run_b1_single_load,
)
from behavior_hl.embodied.evaluator_bridge.gripper_visual_profiles import (
    GRIPPER_VISUAL_PROFILE_INWARD_12CM_OFFICIAL_CAMERA,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
    make_public_action20_marker_oracle_callback,
)
from behavior_hl.embodied.evaluator_bridge.registered_l0_task_runtime import (
    _matching_vector,
    configure_registered_l0_tasks,
    load_registered_l0_tasks,
    make_registered_l0_construction_callback,
    make_registered_l0_navigation_audit_callback,
)


BENCHMARK_PACKAGE_SCHEMA = "action20-capability-benchmark.package/v1"
RUNNABLE_PACKAGE_STAGES = {
    "COMPILED",
    "LIVE_BINDING_PASS",
    "ORACLE_PASS",
    "ADMITTED",
}

CALIBRATION_HEAD_SEGMENTATION_MIN_PIXELS = 8
CALIBRATION_HEAD_SEGMENTATION_CANDIDATE_BUDGET = 16


def _registered_authored_object_names(tasks: Mapping[str, Any]) -> frozenset[str]:
    """Resolve authored object names from the frozen construction contracts."""

    names: set[str] = set()
    for task in tasks.values():
        contract = task.template_contract()
        bindings = contract.get("scope_bindings")
        if not isinstance(bindings, Mapping):
            raise ValueError(
                f"registered task {task.task_name} has no scope bindings"
            )
        for binding in bindings.values():
            if not isinstance(binding, Mapping):
                continue
            if binding.get("binding_kind") != "AUTHORED_OBJECT":
                continue
            actual_name = binding.get("actual_name")
            if not isinstance(actual_name, str) or not actual_name:
                raise ValueError(
                    f"registered task {task.task_name} has an incomplete authored binding"
                )
            names.add(actual_name)
    return frozenset(names)


def _calibration_head_segmentation_channel(evaluator):
    """Reuse the existing private head channel without changing public obs."""
    from behavior_hl.embodied.evaluator_bridge.private_head_segmentation_side_channel import (
        PrivateHeadSegmentationSideChannelError,
        get_preinstalled_private_head_segmentation_side_channel,
    )

    sensor_name = CAMERA_SENSOR_NAMES['head']
    try:
        return get_preinstalled_private_head_segmentation_side_channel(
            evaluator=evaluator,
            head_sensor_name=sensor_name,
        )
    except PrivateHeadSegmentationSideChannelError as exc:
        raise RuntimeError(
            'calibration requires the environment-wrapper-installed '
            'private head segmentation side channel'
        ) from exc


def _calibration_candidate_pose(option):
    yaw = math.radians(float(option['yaw']))
    world_xy = option['members'][0]['world_xy']
    return {
        'position': [float(world_xy[0]), float(world_xy[1]), .005],
        'orientation': [0., 0., math.sin(yaw / 2.), math.cos(yaw / 2.)],
    }


def _calibration_head_candidate_acceptance(*, evaluator, task, torso, control, profile):
    """Build a bounded, render-only source visibility acceptance callback."""
    state = {'channel': None, 'channel_error': None}
    contract = task.template_contract()

    def accept(option, rank):
        result = {
            'accepted': False,
            'visibility_evidence': 'LIVE_HEAD_INSTANCE_SEGMENTATION',
            'mask_threshold_pixels': CALIBRATION_HEAD_SEGMENTATION_MIN_PIXELS,
            'semantic_rgb_review': 'NOT_RUN',
            'candidate_rank': int(rank),
            'source_mask_pixels': {},
            'source_mask_sources': {},
        }
        try:
            if state['channel_error'] is not None:
                result['failure_reason'] = 'HEAD_SEGMENTATION_UNAVAILABLE'
                result['error'] = state['channel_error']
                return result
            if state['channel'] is None:
                try:
                    state['channel'] = _calibration_head_segmentation_channel(evaluator)
                except Exception as exc:
                    state['channel_error'] = f'{type(exc).__name__}: {exc}'
                    result['failure_reason'] = 'HEAD_SEGMENTATION_UNAVAILABLE'
                    result['error'] = state['channel_error']
                    return result

            focus_scopes = tuple(
                option['members'][0].get('calibration_focus_scopes', ())
            )
            cofocus_scopes = tuple(
                option['members'][0].get('calibration_cofocus_scopes', focus_scopes)
            )
            if not focus_scopes:
                result['failure_reason'] = 'NO_CALIBRATION_SOURCE_SCOPE'
                return result
            result['calibration_cofocus_scopes'] = list(cofocus_scopes)
            result['calibration_co_view_required'] = bool(
                option['members'][0].get('calibration_cofocus_required', False)
            )
            result['foil_mask_pixels'] = {}
            result['foil_mask_sources'] = {}
            result['missing_foil_scopes'] = []
            robot = evaluator.robot
            robot.reset()
            current_position, current_orientation = robot.get_position_orientation()
            pose = _calibration_candidate_pose(option)
            robot.set_position_orientation(
                position=_matching_vector(pose['position'], current_position),
                orientation=_matching_vector(pose['orientation'], current_orientation),
                frame='world',
            )
            _set_calibration_torso(robot, torso)
            invalidate_cache = getattr(state['channel'], 'invalidate_cache', None)
            if not callable(invalidate_cache):
                raise RuntimeError('private head segmentation cache invalidation is unavailable')
            invalidate_cache()
            from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
                _build_marker_runtime,
                _capture_rendered_oracle_observation,
                _current_public_action20_target_visibility,
            )
            runtime = _build_marker_runtime(
                evaluator=evaluator,
                control=control,
                profile=profile,
                observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            )
            observation = _capture_rendered_oracle_observation(runtime)
            scene = evaluator.env.scene
            for scope in cofocus_scopes:
                binding = contract['scope_bindings'][scope]
                target = scene.object_registry('name', binding['actual_name'])
                if target is None:
                    if scope in focus_scopes:
                        raise RuntimeError(f'calibration source object is missing: {scope}')
                    result['missing_foil_scopes'].append(scope)
                    continue
                visibility = _current_public_action20_target_visibility(
                    runtime=runtime,
                    target=target,
                    target_name_private=str(target.name),
                )
                pixels = int(visibility['visible_pixels_private'])
                if scope in focus_scopes:
                    result['source_mask_pixels'][scope] = pixels
                    result['source_mask_sources'][scope] = visibility['mask_source_private']
                else:
                    result['foil_mask_pixels'][scope] = pixels
                    result['foil_mask_sources'][scope] = visibility['mask_source_private']
            result['accepted'] = all(
                value >= CALIBRATION_HEAD_SEGMENTATION_MIN_PIXELS
                for value in result['source_mask_pixels'].values()
            )
            result['calibration_co_view_pass'] = bool(
                result['accepted']
                and (
                    not result['calibration_co_view_required']
                    or (
                        bool(result['foil_mask_pixels'])
                        and all(
                            value >= CALIBRATION_HEAD_SEGMENTATION_MIN_PIXELS
                            for value in result['foil_mask_pixels'].values()
                        )
                        and not result['missing_foil_scopes']
                    )
                )
            )
            result['calibration_co_view_fallback'] = bool(
                result['calibration_co_view_required']
                and not result['calibration_co_view_pass']
            )
            result['render_simulator_step'] = int(observation.simulator_step)
            if not result['accepted']:
                result['failure_reason'] = 'CALIBRATION_SOURCE_BELOW_MARKER_PIXEL_GATE'
            elif result['calibration_co_view_fallback']:
                result['failure_reason'] = 'CALIBRATION_SOURCE_ONLY_FALLBACK'
            return result
        except Exception as exc:
            result.update(
                failure_reason='HEAD_SEGMENTATION_CAPTURE_ERROR',
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return result

    return accept


def _prepare_calibration_observation(
    evaluator, task, capture, maps_by_scene, *, control=None, profile=None
):
    """Reuse measured-geometry reset selection; never promote a probe to a task."""
    import numpy as np
    from finalize_registered_initial_state import measured_task, selector, l0
    from behavior_hl.embodied.evaluator_bridge.registered_l0_task_runtime import object_pose, quaternion_angle_rad
    measured = measured_task(task.record, capture, task.template_contract())
    measured['calibration_only'] = bool(task.record.get('construction', {}).get('calibration_only'))
    scene = measured['scene_model']
    if scene not in maps_by_scene:
        data = Path(os.environ['OMNIGIBSON_DATA_PATH'])
        assets = data / 'behavior-1k-assets'
        maps_by_scene[scene] = l0.SceneMaps(scene_model=scene, assets_root=assets,
            room_categories_path=assets / 'metadata/room_categories.txt',
            robot_metadata_path=data / 'omnigibson-robot-assets/models/r1pro/misc/metadata.json')
    candidate_acceptance = None
    if measured['calibration_only']:
        if control is None or profile is None:
            raise RuntimeError(
                'calibration head segmentation requires the live control and task profile'
            )
        candidate_acceptance = _calibration_head_candidate_acceptance(
            evaluator=evaluator,
            task=task,
            torso=l0.EMPTY_TORSO_GRID[('neutral', 'center')],
            control=control,
            profile=profile,
        )
    row = selector.compile_family(
        [measured],
        maps_by_scene[scene],
        64,
        **({
            'candidate_acceptance': candidate_acceptance,
            'candidate_acceptance_budget': CALIBRATION_HEAD_SEGMENTATION_CANDIDATE_BUDGET,
        } if candidate_acceptance is not None else {}),
    )[0]
    diagnostic = row.get('best_candidate_diagnostic') or {}
    row.update(evidence_level=(
                   'LIVE_GEOMETRY_RENDER_CAMERA_PROJECTION'
                   if diagnostic.get('visibility_evidence') == 'LIVE_GEOMETRY_RENDER_CAMERA_PROJECTION'
                   else 'LIVE_GEOMETRY_STATIC_CAMERA_PREDICTION'),
               route_model='GOAL_DAG_WITH_LIVE_INITIAL_TRUTH',
               purpose='CALIBRATION_OBSERVATION_ONLY', formal_task_reset_enabled=False)
    pose = row['fixed_robot_pose']
    if pose is None:
        if (diagnostic.get('visibility_evidence') == 'LIVE_GEOMETRY_RENDER_CAMERA_PROJECTION'
                and not diagnostic.get('calibration_focus_visible', True)):
            failures = row.setdefault('failure_reasons', [])
            if 'NO_CALIBRATION_FOCUS_IN_VIEW' not in failures:
                failures.append('NO_CALIBRATION_FOCUS_IN_VIEW')
        return row
    robot = evaluator.robot
    robot.reset()
    position, orientation = robot.get_position_orientation()
    robot.set_position_orientation(position=_matching_vector(pose['position'], position),
        orientation=_matching_vector(pose['orientation'], orientation), frame='world')
    torso = row['initial_camera_view_state']['torso_qpos']
    _set_calibration_torso(robot, torso)
    actual_position, actual_orientation = object_pose(robot)
    actual_torso = robot.get_joint_positions()[robot.trunk_control_idx].detach().cpu().numpy()
    if (not np.allclose(actual_position, pose['position'], atol=1e-4, rtol=0)
            or quaternion_angle_rad(actual_orientation, pose['orientation']) > 1e-4
            or not np.allclose(actual_torso, torso, atol=1e-4, rtol=0)):
        raise RuntimeError('calibration observation robot/view reset readback failed')
    row['robot_and_view_readback_pass'] = True
    row['physics_steps_executed_for_view'] = 0
    if (isinstance(capture.get('camera_pose_capture'), Mapping)
            and capture['camera_pose_capture'].get('schema_version') == CAMERA_CAPTURE_SCHEMA):
        row['actual_camera_capture'], row['actual_camera_projection'] = _calibration_camera_readback(
            evaluator=evaluator, initial_capture=capture, selected=row,
        )
    else:
        row['camera_readback_status'] = 'NOT_RUN_INCOMPLETE_MEASURED_CAPTURE'
    return row


def _set_calibration_torso(robot, torso):
    """Apply the official neutral/center torso without advancing physics."""
    joints = robot.get_joint_positions().clone()
    joints[robot.trunk_control_idx] = joints.new_tensor(torso)
    robot.set_joint_positions(joints, drive=False)
    robot.set_joint_velocities(joints * 0, drive=False)
    robot.keep_still()


def _capture_calibration_reference(*, evaluator, torso):
    """Set neutral/center before taking the measured camera reference."""
    _set_calibration_torso(evaluator.robot, torso)
    return capture_head_camera_pose(evaluator)


def _calibration_camera_readback(*, evaluator, initial_capture, selected):
    """Read back the real head sensor after the calibration-only reset."""
    import numpy as np
    from behavior_hl.embodied.evaluator_bridge.on_top_support_layers import (
        quaternion_angle_rad,
    )
    from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
        _private_aabb_corners,
        _live_head_camera_sensor,
        _project_world_points_to_head_pixels,
    )

    measured = initial_capture.get('camera_pose_capture')
    if not isinstance(measured, Mapping) or measured.get('schema_version') != CAMERA_CAPTURE_SCHEMA:
        raise RuntimeError('calibration observation requires complete measured camera capture')
    actual = {
        **capture_head_camera_pose(evaluator),
        'capture_phase': 'CALIBRATION_AFTER_NEUTRAL_CENTER_RESET',
        'physics_steps_executed': 0,
    }
    image_shape = actual.get('image_shape')
    if not isinstance(image_shape, list) or len(image_shape) != 2:
        raise RuntimeError('calibration observation camera image shape is unavailable')
    pose = selected.get('fixed_robot_pose')
    diagnostic = selected.get('best_candidate_diagnostic') or {}
    if not isinstance(pose, Mapping) or not isinstance(diagnostic, Mapping):
        raise RuntimeError('calibration observation selection lacks a camera pose')
    expected_position, expected_orientation = camera_pose_at_robot_pose(
        measured, pose['position'][:2], diagnostic['yaw_deg']
    )
    actual_position = np.asarray(actual['head_position'], dtype=float)
    actual_orientation = np.asarray(actual['head_orientation'], dtype=float)
    translation_error = float(np.linalg.norm(actual_position - expected_position))
    rotation_error = float(quaternion_angle_rad(actual_orientation, expected_orientation))
    pose_readback = {
        'expected_head_position': expected_position.tolist(),
        'expected_head_orientation': expected_orientation.tolist(),
        'actual_head_position': actual_position.tolist(),
        'actual_head_orientation': actual_orientation.tolist(),
        'translation_error_m': translation_error,
        'rotation_error_rad': rotation_error,
        'translation_tolerance_m': 0.03,
        'rotation_tolerance_rad': float(np.deg2rad(2.0)),
        'passed': bool(translation_error <= 0.03 and rotation_error <= np.deg2rad(2.0)),
    }
    if not pose_readback['passed']:
        from behavior_hl.embodied.evaluator_bridge.registered_l0_task_runtime import RegisteredL0ConstructionError
        raise RegisteredL0ConstructionError('calibration observation head-camera pose readback failed',
            dict(status='FAIL', construction_pass=False, failure_class='CALIBRATION_CAMERA_PARITY',
                 actual_camera_capture=actual, camera_pose_readback=pose_readback))
    sensor_name, sensor = _live_head_camera_sensor(evaluator)
    projected_pixels = {}
    projected_aabb_pixels = {}
    visible = []
    focus_scopes = list(diagnostic.get('calibration_focus_scopes', ()))
    cofocus_scopes = list(
        diagnostic.get('calibration_cofocus_scopes', focus_scopes)
    )
    focus_aabb_visible = {}
    cofocus_aabb_visible = {}
    for scope, geometry in initial_capture.get('scope_geometry', {}).items():
        pixels = _project_world_points_to_head_pixels(
            sensor, [geometry['position']], image_shape
        )
        if pixels is not None:
            projected_pixels[scope] = list(pixels[0])
            visible.append(scope)
        if scope in cofocus_scopes:
            try:
                lower = np.asarray(geometry['aabb_min'], dtype=float)
                upper = np.asarray(geometry['aabb_max'], dtype=float)
                if (lower.shape != (3,) or upper.shape != (3,)
                        or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                        or np.any(lower > upper)):
                    raise ValueError('invalid calibration focus AABB')
                aabb_pixels = _project_world_points_to_head_pixels(
                    sensor, _private_aabb_corners(lower, upper), image_shape
                )
            except (IndexError, KeyError, TypeError, ValueError):
                aabb_pixels = None
            cofocus_aabb_visible[scope] = aabb_pixels is not None
            if scope in focus_scopes:
                focus_aabb_visible[scope] = aabb_pixels is not None
            if aabb_pixels is not None:
                projected_aabb_pixels[scope] = [list(point) for point in aabb_pixels]
    focus_visible = all(
        scope in visible and focus_aabb_visible.get(scope, False)
        for scope in focus_scopes
    )
    cofocus_required = bool(diagnostic.get('calibration_cofocus_required', False))
    cofocus_visible = all(
        scope in visible and cofocus_aabb_visible.get(scope, False)
        for scope in cofocus_scopes
    )
    projection = {
        'schema_version': 'registered-calibration-camera-projection-readback/v1',
        'sensor_name': sensor_name,
        'image_shape': image_shape,
        'intrinsic_matrix': actual['intrinsic_matrix'],
        'visible_scope_names': visible,
        'projected_pixels': projected_pixels,
        'projected_aabb_pixels': projected_aabb_pixels,
        'calibration_focus_scopes': focus_scopes,
        'calibration_cofocus_scopes': cofocus_scopes,
        'calibration_focus_visible': focus_visible,
        'calibration_focus_aabb_visible': all(
            focus_aabb_visible.get(scope, False) for scope in focus_scopes
        ),
        'calibration_cofocus_required': cofocus_required,
        'calibration_cofocus_visible': (
            not cofocus_required or cofocus_visible
        ),
        'calibration_cofocus_aabb_visible': all(
            cofocus_aabb_visible.get(scope, False) for scope in cofocus_scopes
        ),
        'calibration_co_view_fallback': bool(
            cofocus_required and not cofocus_visible
        ),
        'focus_projection_evidence': 'WHOLE_AABB',
        'projection_evidence': 'WHOLE_AABB',
        'pose_readback': pose_readback,
    }
    if not focus_visible:
        from behavior_hl.embodied.evaluator_bridge.registered_l0_task_runtime import RegisteredL0ConstructionError
        raise RegisteredL0ConstructionError('NO_CALIBRATION_FOCUS_IN_VIEW',
            dict(status='FAIL', construction_pass=False, failure_class='CALIBRATION_CAMERA_FOCUS',
                 actual_camera_capture=actual, actual_camera_projection=projection))
    return actual, projection


def _verified_package_file(
    package_path: Path,
    reference: Any,
    field: str,
) -> Path:
    if not isinstance(reference, Mapping):
        raise ValueError(f"benchmark package {field} must be a file reference")
    raw_path = reference.get("path")
    expected_sha256 = reference.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_sha256, str):
        raise ValueError(f"benchmark package {field} file reference is incomplete")
    path = Path(raw_path)
    if not path.is_absolute():
        path = package_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"benchmark package {field} SHA256 mismatch")
    return path


def runtime_registration_from_benchmark_package(
    package_path: Path,
    task_names: tuple[str, ...],
    *, construction_only: bool = False, construct_and_oracle: bool = False, prepared_model_replay: bool = False,
) -> Path:
    package_path = package_path.resolve()
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if package.get('purpose') == 'CALIBRATION_ONLY' and not construction_only:
        raise ValueError('calibration specimen package cannot enter Oracle')
    if package.get("schema_version") != BENCHMARK_PACKAGE_SCHEMA:
        raise ValueError("unsupported benchmark package schema")
    if package.get("benchmark_level") not in {"L0", "L1"}:
        raise ValueError(
            f"runner adapter is unavailable for {package.get('benchmark_level')}"
        )
    if package.get("episode_mode") != "goal_completion":
        raise ValueError("benchmark package is not a goal-completion episode")
    if construction_only and construct_and_oracle:
        raise ValueError('construction-only and direct Oracle modes are exclusive')
    if construct_and_oracle and package.get('purpose') != 'TASK_CONSTRUCTION':
        raise ValueError('direct construction requires a TASK_CONSTRUCTION package')
    if package.get("stage") not in RUNNABLE_PACKAGE_STAGES and not ((construction_only or construct_and_oracle) and package.get('stage') == 'CONSTRUCTION_PENDING'):
        raise ValueError("benchmark package has not reached COMPILED")
    state = package.get('initial_state_compilation', {})
    if prepared_model_replay and (package.get('model_rollout_allowed') is not True
            or 'PREPARED_MODEL_REPLAY' not in package.get('allowed_execution_modes',[])
            or state.get('reset_mode') != 'RESTORE_AND_VERIFY_NO_RECONSTRUCTION'):
        raise ValueError('invalid prepared model replay package')
    if state.get('required') and state.get('status') != 'FROZEN' and not (construction_only or construct_and_oracle or prepared_model_replay):
        raise ValueError('benchmark package initial state has not been frozen')
    task_spec_ids = package.get("task_spec_ids")
    if (
        not isinstance(task_spec_ids, list)
        or any(not isinstance(item, str) for item in task_spec_ids)
        or len(task_spec_ids) != len(set(task_spec_ids))
        or package.get("task_count") != len(task_spec_ids)
    ):
        raise ValueError("benchmark package task membership is invalid")
    missing = sorted(set(task_names) - set(task_spec_ids))
    if missing:
        raise ValueError(f"task is absent from benchmark package: {missing[0]}")
    task_specs_path = _verified_package_file(
        package_path,
        package.get("task_specs"),
        "task_specs",
    )
    task_spec_rows = [
        json.loads(line)
        for line in task_specs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [row.get("task_spec_id") for row in task_spec_rows] != task_spec_ids:
        raise ValueError("benchmark package TaskSpec membership drifted")
    if any(row.get("construction_blockers") for row in task_spec_rows):
        raise ValueError("benchmark package contains unresolved construction blockers")
    runtime = package.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("adapter") != "registered_l0":
        raise ValueError("unsupported benchmark runtime adapter")
    catalog_path = _verified_package_file(
        package_path,
        runtime.get("registration_catalog"),
        "runtime.registration_catalog",
    )
    from behavior_hl.embodied.evaluator_bridge.benchmark_package_correspondence import validate_package_correspondence
    validate_package_correspondence(package_path, task_spec_rows, catalog_path, task_names)
    return catalog_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=19506)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--task-names", nargs="+", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--benchmark-package", type=Path)
    parser.add_argument("--v-package", type=Path, help="Validated V envelope referencing an unchanged L0 package")
    v_mode = parser.add_mutually_exclusive_group()
    v_mode.add_argument("--v-branch", choices=("D0", "D1"))
    v_mode.add_argument("--v-suite", action="store_true", help="One native resident load: source baseline, D0, D1")
    parser.add_argument("--construct-and-oracle", action="store_true",
                        help="Construct declared initial state and run Oracle in the same scene load; does not claim frozen admission")
    parser.add_argument('--placement-physical-consequences', action='store_true')
    parser.add_argument('--prepared-model-replay', action='store_true',
                        help='Restore frozen prepared state for a model; forbids construction and Oracle')
    parser.add_argument('--compile-initial-state', action='store_true',
                        help='Construct and capture initial state only; do not execute Oracle')
    parser.add_argument("--runtime-registration", type=Path)
    parser.add_argument("--l0-taskspecs", type=Path)
    parser.add_argument("--reset-pose-overlay", type=Path)
    parser.add_argument("--nearby-initial-pose-search", action="store_true")
    parser.add_argument("--oracle-preflight-state-dir", type=Path)
    parser.add_argument("--oracle-timeout-seconds", type=float, default=None)
    parser.add_argument("--initial-construction-timeout-seconds", type=float, default=None)
    parser.add_argument("--expected-execution-contract-sha256")
    parser.add_argument("--partial-scene-load", action="store_true")
    navigation = parser.add_mutually_exclusive_group()
    navigation.add_argument(
        "--registered-public-action20-marker-navigation",
        action="store_true",
    )
    navigation.add_argument(
        "--registered-public-action20-marker-gt-navigation",
        action="store_true",
    )
    navigation.add_argument(
        "--registered-public-action20-marker-se2-navigation",
        action="store_true",
    )
    navigation.add_argument(
        "--registered-admission-double-gate",
        action="store_true",
    )
    return parser


def task_template_path(
    *, data_root: Path, scene_model: str, task_name: str, split: str
) -> Path:
    split_dir = {
        "train": Path("scenes"),
        "public_test": Path("scene_test/public"),
        "hidden_test": Path("scene_test/private"),
    }.get(split)
    if split_dir is None:
        raise ValueError(f"unsupported split: {split}")
    filename = f"{scene_model}_task_{task_name}_0_0_template.json"
    path = (
        data_root
        / "2026-challenge-task-instances"
        / split_dir
        / scene_model
        / "json"
        / filename
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _l0_visual_variant_dependencies(package_path, registered):
    """Only explicit L0 visual pairs opt into the render-only fallback."""
    if package_path is None or not registered:
        return {}
    package = json.loads(package_path.read_text())
    if package.get("benchmark_level") != "L0":
        return {}
    dependencies = {}
    for identifier, task in registered.items():
        contract = task.template_contract()
        base = contract.get("pair_base_task_spec_id")
        if base and base != identifier and contract.get("visual_rules"):
            dependencies[identifier] = base
    return dependencies




def main(argv: list[str] | None = None, *, model_callback: Any | None = None,
         model_prepared_states: Mapping[str, Any] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.v_package is not None and args.placement_physical_consequences:
        raise ValueError('V intervention packages require Symbolic execution')
    if args.prepared_model_replay:
        if not args.placement_physical_consequences:
            raise ValueError('prepared-model-replay requires Physical execution')
        if model_callback is None:
            raise ValueError('Physical model replay requires a model callback')
        if args.construct_and_oracle or args.compile_initial_state:
            raise ValueError('Physical model replay forbids Oracle construction')
        if args.nearby_initial_pose_search:
            raise ValueError('Physical model replay forbids nearby pose search')
        if args.benchmark_package is None:
            raise ValueError('Physical model replay requires a package-backed contract')
        if model_prepared_states is None or set(model_prepared_states) != set(args.task_names):
            raise ValueError('Physical model replay requires prepared states for every Instruction')
    elif args.placement_physical_consequences and (not args.construct_and_oracle or model_callback is not None):
        raise ValueError('Physical placement requires a package-backed scripted Oracle')
    v_config = None
    oracle_prepared_states = None
    if (args.v_package is not None) != (args.v_branch is not None or args.v_suite):
        raise ValueError("--v-package requires exactly one of --v-branch or --v-suite")
    if args.v_suite:
        raise ValueError('unified V collection requires an explicit --v-branch D0 or D1; suite expansion is not enabled')
    if args.v_package is not None:
        from .v_protocol import require_supported
        require_supported(json.loads(args.v_package.read_text()))
        if any((args.benchmark_package, args.l0_taskspecs, args.runtime_registration,
                args.reset_pose_overlay, args.compile_initial_state, args.construct_and_oracle,
                args.partial_scene_load, args.nearby_initial_pose_search)):
            raise ValueError("V inherits the frozen source package and forbids conflicting inputs")
        from .v_intervention import load_package
        v_config, args.l0_taskspecs, args.reset_pose_overlay, policy = load_package(
            args.v_package, task_names=args.task_names, seed=args.seed)
        if v_config.get('_source_adapter') == 'registered_l0':
            args.benchmark_package = Path(v_config['_base_package_path'])
            args.l0_taskspecs = None
            if args.v_suite:
                raise ValueError('registered V requires one explicit branch, not a projected suite')
        if v_config.get('schema_version') == 'behavior.v-r4-package/v1' and args.v_suite:
            raise ValueError('R4 requires --v-branch D0/D1, not a projected suite')
        if model_callback is not None and v_config.get('_prepared_states'):
            if model_prepared_states is None:
                model_prepared_states = v_config['_prepared_states']
            elif model_prepared_states != v_config['_prepared_states']:
                raise ValueError('model preparation must match the explicitly enabled V donor bundle')
        if v_config.get('_prepared_states') and model_callback is None:
            oracle_prepared_states = v_config['_prepared_states']
            if set(oracle_prepared_states) != set(args.task_names):
                raise ValueError('V Oracle prepared membership mismatch')
            from .prepared_model_state import validate_bundle
            for task_id, bundle in oracle_prepared_states.items():
                validate_bundle(bundle, instruction_id=task_id, seed=args.seed,
                                observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE)
        expected_timeout = policy['oracle_timeout_seconds_per_task']
        if args.oracle_timeout_seconds is not None and args.oracle_timeout_seconds != expected_timeout:
            raise ValueError("V cannot change the inherited Oracle budget")
        args.oracle_timeout_seconds = expected_timeout
    if model_prepared_states is not None:
        if model_callback is None or set(model_prepared_states) != set(args.task_names):
            raise ValueError('prepared replay requires exact model Instruction membership')
        if args.nearby_initial_pose_search:
            raise ValueError('prepared replay forbids nearby pose search')
    effective_execution_contract = None
    if args.expected_execution_contract_sha256 is not None and args.benchmark_package is None:
        raise ValueError('expected execution contract requires a benchmark package')
    if args.construct_and_oracle and (args.compile_initial_state or args.benchmark_package is None):
        raise ValueError('direct Oracle requires --benchmark-package and excludes --compile-initial-state')
    if args.nearby_initial_pose_search and (
        args.compile_initial_state
        or (not args.runtime_registration and not args.benchmark_package
            and (not args.l0_taskspecs or not args.reset_pose_overlay))
    ):
        raise ValueError("nearby search requires an Oracle task source and an existing initial state")
    if args.reset_pose_overlay is not None and (
        args.runtime_registration is None and args.l0_taskspecs is None and args.benchmark_package is None
    ):
        raise ValueError("reset-pose overlay requires registered L0 or L0 TaskSpecs")
    selected_registered = None
    selected_l0 = None
    preflight_rows = {}
    preflight_source_hash = None
    task_sources = (
        args.benchmark_package,
        args.runtime_registration,
        args.l0_taskspecs,
    )
    if sum(source is not None for source in task_sources) > 1:
        raise ValueError(
            "benchmark package, runtime registration, and L0 TaskSpecs are exclusive"
        )
    if args.benchmark_package is not None:
        from behavior_hl.embodied.evaluator_bridge.benchmark_execution_contract import (
            resolve_execution_contract, execution_contract_sha256, runner_identity,
        )
        effective_execution_contract = resolve_execution_contract(
            args, package_path=args.benchmark_package, selected_task_ids=args.task_names,
            runner_source_identity=runner_identity(Path(__file__).resolve().parents[4]),
        )
        args.oracle_timeout_seconds = effective_execution_contract['oracle_timeout_seconds_per_task']
        expected_contract = args.expected_execution_contract_sha256
        if expected_contract is not None and expected_contract != execution_contract_sha256(effective_execution_contract):
            raise ValueError('Worker execution contract differs from campaign plan')
        from behavior_hl.embodied.evaluator_bridge.oracle_preflight_candidates import load_oracle_preflight_candidates
        preflight_rows = load_oracle_preflight_candidates(args.benchmark_package)
        if preflight_rows and not args.compile_initial_state:
            if args.oracle_preflight_state_dir is None:
                raise ValueError("exported candidates require a campaign-scoped --oracle-preflight-state-dir")
            package_payload = json.loads(args.benchmark_package.read_text())
            preflight_source_hash = package_payload['oracle_preflight']['candidates']['sha256']
        args.runtime_registration = runtime_registration_from_benchmark_package(
            args.benchmark_package,
            tuple(args.task_names),
            construction_only=args.compile_initial_state,
            construct_and_oracle=args.construct_and_oracle,
            prepared_model_replay=args.prepared_model_replay,
        )
    if args.l0_taskspecs is not None:
        rows = {
            row["task_spec_id"]: row
            for line in args.l0_taskspecs.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in (json.loads(line),)
        }
        try:
            selected_l0 = tuple(rows[candidate_id] for candidate_id in args.task_names)
        except KeyError as exc:
            raise ValueError(
                f"candidate ID is absent from L0 TaskSpecs: {exc.args[0]}"
            ) from exc
        profiles = tuple(
            SimpleNamespace(
                **task_spec["private"]["source_task_identity"],
                task_language=task_spec["public"]["instruction"],
            )
            for task_spec in selected_l0
        )
    elif args.runtime_registration is not None:
        registered = load_registered_l0_tasks(
            args.runtime_registration,
            reset_pose_overlay=args.reset_pose_overlay,
        )
        try:
            selected_registered = {
                candidate_id: registered[candidate_id]
                for candidate_id in args.task_names
            }
        except KeyError as exc:
            raise ValueError(
                f"candidate ID is absent from runtime registration: {exc.args[0]}"
            ) from exc
        profiles = tuple(
            task.profile() for task in selected_registered.values()
        )
    else:
        if len(args.task_names) < 2:
            raise ValueError("resident canary requires at least two same-scene tasks")
        profiles = tuple(get_inside_task_profile(name) for name in args.task_names)
    public_marker_oracle = bool(
        args.registered_public_action20_marker_navigation
        or args.registered_public_action20_marker_gt_navigation
        or args.registered_public_action20_marker_se2_navigation
        or args.registered_admission_double_gate
    )
    if args.compile_initial_state and (selected_registered is None or public_marker_oracle):
        raise ValueError('initial-state compilation requires registered tasks and no Oracle mode flag')
    if args.benchmark_package is not None:
        package_state = json.loads(args.benchmark_package.read_text()).get('initial_state_compilation', {})
        if package_state.get('required') and package_state.get('status') != 'FROZEN' and not (args.compile_initial_state or args.construct_and_oracle or args.prepared_model_replay):
            raise ValueError('package initial state is not frozen; run construction compilation first')
    if args.construct_and_oracle and not public_marker_oracle:
        raise ValueError('direct construction requires a public Action20 Oracle mode')
    if selected_l0 is not None and (
        args.registered_public_action20_marker_gt_navigation
        or args.registered_admission_double_gate
    ):
        raise ValueError("L0 TaskSpecs already select GT admission navigation")
    if (
        public_marker_oracle
        and model_callback is None
        and not args.registered_admission_double_gate
        and selected_l0 is None
        and not args.construct_and_oracle
        and oracle_prepared_states is None
        and (
            selected_registered is None or len(profiles) != 1
        )
    ):
        raise ValueError(
            "registered public Action20+Marker Oracle requires one registered task"
        )
    if args.registered_admission_double_gate and selected_registered is None:
        raise ValueError("registered admission double gate requires registered tasks")
    scene_models = {profile.scene_model for profile in profiles}
    if len(scene_models) != 1:
        raise ValueError("resident task names do not share one scene model")
    scene_model = next(iter(scene_models))
    if any(profile.split != args.split for profile in profiles):
        raise ValueError("resident task profile split mismatch")
    task_specs = tuple(
        ResidentTaskSpec(
            task_name=profile.task_name,
            task_id=profile.task_id,
            scene_model=profile.scene_model,
            split=profile.split,
            instance_id=profile.instance_id,
        )
        for profile in profiles
    )
    data_root = Path(os.environ["OMNIGIBSON_DATA_PATH"]).resolve()
    variant_spec = v_config.get('_variant_task_spec') if v_config is not None else None
    if variant_spec is not None:
        from .v_variant_registration import variant_artifact
    template_paths = (
        (variant_artifact(variant_spec, 'template'),) if variant_spec is not None else
        tuple(task.template_path for task in selected_registered.values())
        if selected_registered is not None
        else tuple(
            task_template_path(
                data_root=data_root,
                scene_model=scene_model,
                task_name=task.task_name,
                split=task.split,
            )
            for task in task_specs
        )
    )
    artifact_root = args.artifact_root.resolve()
    generated = artifact_root / "generated"
    templates = tuple(load_scene_document(path) for path in template_paths)
    official_reset_runtime = None
    reset_records_by_item = None
    before_task_reset = None
    if args.reset_pose_overlay is not None:
        overlay = load_l0_reset_pose_overlay(args.reset_pose_overlay)
        missing = [task_id for task_id in args.task_names if task_id not in overlay]
        if missing:
            raise ValueError(f"selected tasks have no fixed reset pose: {missing}")
        reset_records = tuple(overlay[task_id] for task_id in args.task_names)
        reset_records_by_item = dict(zip(args.task_names, reset_records, strict=True))
        templates = tuple(
            patch_scene_template_reset_pose(template, record)
            for template, record in zip(templates, reset_records, strict=True)
        )
        source_tro_paths = (
            tuple(task.tro_state_path for task in selected_registered.values())
            if selected_registered is not None
            else tuple(
                Path(task_spec["private"]["source_artifacts"]["tro_state"]["path"])
                for task_spec in selected_l0
            )
        )
        reset_tro_dir = generated / "l0_reset_pose_tro_states"
        patched_tro_paths = []
        for task_id, source_path, record in zip(
            args.task_names, source_tro_paths, reset_records, strict=True
        ):
            output_path = reset_tro_dir / f"{task_id}.json"
            write_json_atomic(
                output_path,
                patch_tro_state_reset_pose(
                    json.loads(source_path.read_text(encoding="utf-8")), record
                ),
            )
            patched_tro_paths.append(output_path)
        if selected_registered is not None:
            selected_registered = {
                task_id: replace(task, tro_state_path=tro_path)
                for (task_id, task), tro_path in zip(
                    selected_registered.items(), patched_tro_paths, strict=True
                )
            }
        else:
            active_reset_item = args.task_names[0]

            def before_task_reset(item_id: str, *, _source_records=reset_records_by_item) -> None:
                nonlocal active_reset_item
                if item_id not in _source_records:
                    raise ValueError(f"missing fixed reset pose for {item_id}")
                active_reset_item = item_id

            official_reset_runtime = {
                "records_by_task_spec_id": reset_records_by_item,
                "tro_paths_by_task_spec_id": dict(
                    zip(args.task_names, patched_tro_paths, strict=True)
                ),
                "current_task_spec_id": lambda: active_reset_item,
            }
    union, union_report = build_union_scene_document(templates)
    union_path = generated / f"{scene_model}_resident_union.json"
    write_json_atomic(union_path, union)
    write_json_atomic(
        generated / "resident_batch_manifest.json",
        {
            **union_report,
            "scene_model": scene_model,
            "task_names": list(args.task_names),
            "task_templates": [str(path) for path in template_paths],
            "physical_gpu": args.physical_gpu,
            "server_port": args.server_port,
            "seed": args.seed,
            "reset_pose_overlay": (
                None
                if args.reset_pose_overlay is None
                else str(args.reset_pose_overlay.resolve())
            ),
        },
    )
    primary = task_specs[0]
    filter_embedded_scene_agents = not args.partial_scene_load
    worker = B1SingleLoadWorkerSpec(
        worker_id=args.worker_id,
        physical_gpu=args.physical_gpu,
        server_port=args.server_port,
        seed=args.seed,
        artifact_root=artifact_root,
        task_name=primary.task_name,
        split=primary.split,
        instance_id=primary.instance_id,
    )
    oracle_callback = None
    oracle_callbacks = None
    navigation_audit_callback = None
    item_ids = None
    if selected_l0 is not None:
        motion_safety_starter_cache = {}
        item_ids = tuple(task_spec["task_spec_id"] for task_spec in selected_l0)
        oracle_callbacks = tuple(
            make_l0_taskspec_action20_oracle_callback(
                task_spec=task_spec,
                seed=args.seed,
                phase_name=public_oracle_phase_name(index, task_spec['task_spec_id'], len(profiles)),
                motion_safety_starter_cache=motion_safety_starter_cache,
                admission_gt_macro_navigation=(
                    not (
                        args.registered_public_action20_marker_navigation
                        or args.registered_public_action20_marker_se2_navigation
                    )
                ),
                navigation_lane=(
                    "se2_navigation_leg_v1"
                    if args.registered_public_action20_marker_se2_navigation
                    else "legacy_xy_recovery_v1"
                ),
            )
            for index, task_spec in enumerate(selected_l0)
        )
    elif (args.construct_and_oracle or oracle_prepared_states is not None) and selected_registered is not None:
        motion_safety_starter_cache = {}
        item_ids = tuple(selected_registered)
        oracle_callbacks = tuple(
            make_public_action20_marker_oracle_callback(
                seed=effective_execution_contract['seed'], profile=profile, observation_profile=effective_execution_contract['camera_profile'],
                task_contract_owner=selected_registered[item_ids[index]],
                admission_gt_macro_navigation=False,
                navigation_lane=effective_execution_contract['navigation_lane'],
                phase_name=public_oracle_phase_name(index, item_ids[index], len(profiles)),
                motion_safety_starter_cache=motion_safety_starter_cache,
                placement_physical_consequences_enabled=args.placement_physical_consequences,
            ) for index, profile in enumerate(profiles)
        )
    elif args.registered_admission_double_gate:
        motion_safety_starter_cache = {}
        callbacks = {
            profile.task_name: make_public_action20_marker_oracle_callback(
                seed=args.seed,
                profile=profile,
                observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
                admission_gt_macro_navigation=True,
                phase_name=public_oracle_phase_name(index, profile.task_name, len(profiles)),
                motion_safety_starter_cache=motion_safety_starter_cache,
            )
            for index, profile in enumerate(profiles)
        }

        def dispatch_registered_oracle(**kwargs: Any) -> Mapping[str, Any]:
            task_name = kwargs["evaluator"].env.task.activity_name
            return callbacks[task_name](**kwargs)

        oracle_callback = dispatch_registered_oracle
        navigation_audit_callback = make_registered_l0_navigation_audit_callback(
            selected_registered
        )
    elif public_marker_oracle:
        oracle_callback = make_public_action20_marker_oracle_callback(
            seed=args.seed,
            profile=profiles[0],
            observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            admission_gt_macro_navigation=(
                args.registered_public_action20_marker_gt_navigation
            ),
            navigation_lane=(
                "se2_navigation_leg_v1"
                if args.registered_public_action20_marker_se2_navigation
                else "legacy_xy_recovery_v1"
            ),
        )
    if v_config is not None and selected_registered is not None:
        if any(not task.record.get('construction', {}).get('frozen_initial_state')
               for task in selected_registered.values()):
            raise ValueError('registered V requires frozen object state; live construction is forbidden')
    construction_callback = None
    initial_task_facts_callback = None
    initial_pose_validation_callback = None
    candidate_preflight_callback = None
    install_private_head_segmentation = model_callback is not None
    if selected_registered is not None:
        initial_predicate_mode = (
            "action20_symbolic" if public_marker_oracle and not args.placement_physical_consequences else "official"
        )
        if (effective_execution_contract or {}).get('initial_predicate_mode') == 'action20_symbolic':
            if not all(t.record.get('construction', {}).get('frozen_initial_state')
                       for t in selected_registered.values()):
                raise ValueError('legacy Physical initial reader requires frozen task states')
            initial_predicate_mode = 'action20_symbolic'
        if any(t.record.get('task_contract') for t in selected_registered.values()):
            from .registered_l0_task_runtime import capture_l1_prepared_facts
            facts_tasks = {t.profile().task_name: t for t in selected_registered.values()}
            def initial_task_facts_callback(*, evaluator, task):
                return capture_l1_prepared_facts(evaluator, facts_tasks[task.task_name], predicate_mode=initial_predicate_mode)
        if (model_prepared_states is not None or oracle_prepared_states is not None or args.nearby_initial_pose_search
                or any(t.record.get('_prepared_oracle_state') for t in selected_registered.values())
                or (preflight_rows and not args.compile_initial_state)):
            from behavior_hl.embodied.evaluator_bridge.registered_l0_task_runtime import validate_registered_initial_semantics
            registered_by_name = {task.profile().task_name: task for task in selected_registered.values()}
            def initial_pose_validation_callback(*, evaluator: Any, task: Any) -> bool:
                return validate_registered_initial_semantics(evaluator=evaluator, task=registered_by_name[task.task_name], predicate_mode=initial_predicate_mode)
        if preflight_rows and not args.compile_initial_state:
            from behavior_hl.embodied.evaluator_bridge.oracle_candidate_preflight import prepare_exported_start
            def candidate_preflight_callback(*, evaluator: Any, task: Any) -> Mapping[str, Any]:
                registered = registered_by_name[task.task_name]
                row = preflight_rows[registered.candidate_id]
                members = [task_id for task_id, member in preflight_rows.items()
                           if member['source_family_id'] == row['source_family_id']]
                return prepare_exported_start(evaluator, row=row, family_members=members,
                    state_dir=args.oracle_preflight_state_dir, source_hash=preflight_source_hash,
                    initial_state_valid=lambda: initial_pose_validation_callback(evaluator=evaluator, task=task))
        construction_callback = None if args.prepared_model_replay else make_registered_l0_construction_callback(
            selected_registered,
            predicate_mode=initial_predicate_mode,
            **({'allow_pending_initial_state': True} if args.compile_initial_state else {}),
            **({'allow_live_oracle_construction': True} if args.construct_and_oracle else {}))
        if args.compile_initial_state or args.construct_and_oracle:
            install_private_head_segmentation = args.placement_physical_consequences or any(
                bool(task.record.get('construction', {}).get('calibration_only'))
                for task in selected_registered.values()
            )
            from behavior_hl.embodied.evaluator_bridge.registered_l0_task_runtime import (
                RegisteredL0ConstructionError,
                capture_registered_initial_state,
            )
            original_construction_callback = construction_callback
            registered_by_activity = {task.task_name: task for task in selected_registered.values()}
            profile_by_activity = {profile.task_name: profile for profile in profiles}
            calibration_maps = {}

            def capture_initial_state(*, evaluator, task):
                try:
                    capture = dict(
                        capture_registered_initial_state(
                            evaluator=evaluator,
                            task=task,
                            predicate_mode=initial_predicate_mode,
                        )
                    )
                except RegisteredL0ConstructionError as exc:
                    failure_report = dict(exc.report)
                    native_report = native_initial_state_completion_report(
                        evaluator.env.scene
                    )
                    if native_report is not None:
                        failure_report["native_initial_state_completion"] = dict(
                            native_report
                        )
                    raise RegisteredL0ConstructionError(
                        str(exc), failure_report
                    ) from exc
                native_report = native_initial_state_completion_report(
                    evaluator.env.scene
                )
                if native_report is not None:
                    capture["native_initial_state_completion"] = dict(native_report)
                return capture

            def construction_callback(**kwargs: Any) -> Mapping[str, Any]:
                try:
                    report = dict(original_construction_callback(**kwargs))
                except RegisteredL0ConstructionError as exc:
                    failure_report = dict(exc.report)
                    native_report = native_initial_state_completion_report(
                        kwargs["evaluator"].env.scene
                    )
                    if native_report is not None:
                        failure_report["native_initial_state_completion"] = dict(
                            native_report
                        )
                    raise RegisteredL0ConstructionError(
                        str(exc), failure_report
                    ) from exc
                registered = registered_by_activity[kwargs['task'].task_name]
                if registered.record['construction'].get('calibration_only'):
                    from finalize_registered_initial_state import l0
                    calibration_torso = l0.EMPTY_TORSO_GRID[('neutral', 'center')]
                    reference_capture = _capture_calibration_reference(
                        evaluator=kwargs['evaluator'], torso=calibration_torso,
                    )
                report['initial_state_capture'] = capture_initial_state(
                    evaluator=kwargs['evaluator'], task=registered)
                if registered.record['construction'].get('calibration_only'):
                    capture = dict(report['initial_state_capture'])
                    capture['camera_pose_capture'] = reference_capture
                    report['initial_state_capture'] = capture
                    try:
                        report['calibration_observation_pose'] = _prepare_calibration_observation(
                            kwargs['evaluator'], registered, capture, calibration_maps,
                            control=kwargs['control'],
                            profile=profile_by_activity[registered.task_name],
                        )
                    except Exception as exc:
                        diagnostic = getattr(exc, 'report', None)
                        failure_report = dict(report)
                        failure_report.update(
                            status='FAIL',
                            construction_pass=False,
                            failure_class='CALIBRATION_HEAD_SEGMENTATION_ERROR',
                            calibration_observation_error_type=type(exc).__name__,
                            calibration_observation_error=str(exc),
                        )
                        if isinstance(diagnostic, Mapping):
                            failure_report['calibration_observation_error_report'] = dict(diagnostic)
                            if diagnostic.get('failure_class') == 'CALIBRATION_HEAD_SEGMENTATION_ERROR':
                                failure_report['resident_session_invalidated'] = True
                        raise RegisteredL0ConstructionError(
                            'calibration head segmentation candidate acceptance failed',
                            failure_report,
                        ) from exc
                    refreshed = capture_initial_state(
                        evaluator=kwargs['evaluator'], task=registered
                    )
                    if refreshed['scope_geometry'] != capture['scope_geometry']:
                        raise RuntimeError('calibration view reset changed task object geometry')
                    actual_camera = report['calibration_observation_pose'].get('actual_camera_capture')
                    if actual_camera is not None:
                        refreshed['camera_pose_capture'] = actual_camera
                    report['initial_state_capture'] = refreshed
                return report
    evaluator_module_configurer = None
    if selected_registered is not None:
        if not (args.compile_initial_state or args.construct_and_oracle):
            evaluator_module_configurer = (
                lambda module: configure_registered_l0_tasks(
                    module,
                    selected_registered,
                    full_scene_load=not args.partial_scene_load,
                    defer_process_goal_termination=(public_marker_oracle or model_callback is not None),
                )
            )
        else:
            authored_object_names = _registered_authored_object_names(selected_registered)

            def configure_evaluator_module(module: Any):
                stack = ExitStack()
                try:
                    registration_cleanup = configure_registered_l0_tasks(
                        module,
                        selected_registered,
                        full_scene_load=not args.partial_scene_load,
                        defer_process_goal_termination=(public_marker_oracle or model_callback is not None),
                    )
                    stack.callback(registration_cleanup)
                    if authored_object_names:
                        stack.enter_context(
                            install_native_initial_state_completion(
                                authored_object_names=authored_object_names,
                            )
                        )
                except BaseException:
                    stack.close()
                    raise
                return stack.pop_all().close

            evaluator_module_configurer = configure_evaluator_module
    elif variant_spec is not None:
        from .v_variant_registration import configure_variant_task
        variant_prepared = (oracle_prepared_states or model_prepared_states)[variant_spec['task_spec_id']]['prepared']
        variant_tro = official_reset_runtime['tro_paths_by_task_spec_id'][variant_spec['task_spec_id']] if official_reset_runtime else None
        evaluator_module_configurer = lambda module: configure_variant_task(
            module, variant_spec, variant_prepared, tro_state_path=variant_tro)
    elif official_reset_runtime is not None:
        evaluator_module_configurer = lambda module: configure_official_l0_reset_pose_runtime(module, **official_reset_runtime)

    v_branches = None
    visual_dependencies = _l0_visual_variant_dependencies(args.benchmark_package, selected_registered)

    if (
        oracle_prepared_states is not None
        and selected_l0 is not None
        and v_config is not None
        and v_config.get('schema_version') == 'behavior.v-r4-package/v1'
    ):
        from .r4_package_gate import enabled
        if enabled(v_config):
            from .v_r4_intervention import make_native_r4_world_verification_callback, native_r4_initial_semantics_ok
        else:
            from .v_r4_legacy import make_native_r4_world_verification_callback, native_r4_initial_semantics_ok
        if construction_callback is not None:
            raise ValueError('native R4 prepared replay cannot mix registered construction')
        if args.nearby_initial_pose_search:
            raise ValueError('R4 prepared replay forbids nearby pose search')
        construction_callback = make_native_r4_world_verification_callback(v_config)

        def initial_pose_validation_callback(*, evaluator: Any, task: Any, _config=v_config) -> bool:
            return native_r4_initial_semantics_ok(evaluator=evaluator, task=task, config=_config)

    result = run_b1_single_load(
        bindings=build_production_bindings(),
        worker=worker,
        strategy_sweep_callback=make_resident_scene_batch_callback(
            tasks=task_specs,
            native_task_metric=selected_l0 is None,
            profiles=profiles,
            templates=templates,
            union_template=union,
            seed=args.seed,
            filter_embedded_scene_agents=filter_embedded_scene_agents,
            construction_callback=construction_callback,
            construction_only=args.compile_initial_state,
            navigation_audit_callback=navigation_audit_callback,
            oracle_callback=oracle_callback,
            item_ids=item_ids,
            oracle_callbacks=oracle_callbacks,
            before_task_reset=before_task_reset,
            reset_pose_records=reset_records_by_item,
            oracle_timeout_seconds=args.oracle_timeout_seconds,
            initial_construction_timeout_seconds=args.initial_construction_timeout_seconds,
            present_inward_grippers=not args.compile_initial_state,
            nearby_initial_pose_search=args.nearby_initial_pose_search,
            initial_pose_validation_callback=initial_pose_validation_callback,
            candidate_preflight_callback=candidate_preflight_callback,
            variant_base_item_ids={
                name: registered.record['construction']['pair_base_task_spec_id']
                for name, registered in (selected_registered or {}).items()
                if args.construct_and_oracle
                and registered.record.get('construction', {}).get('pair_base_task_spec_id')
                and registered.record['construction']['pair_base_task_spec_id'] != name
                and registered.record['construction']['pair_base_task_spec_id'] in selected_registered
            },
            effective_execution_contract=effective_execution_contract,
            model_callback=model_callback,
            model_prepared_states=model_prepared_states,
            oracle_prepared_states=oracle_prepared_states,
            v_config=v_config,
            v_branch=args.v_branch,
            v_branches=v_branches,
            l0_visual_variant_dependencies=visual_dependencies,
            initial_task_facts_callback=initial_task_facts_callback,
        ),
        camera_profile=(effective_execution_contract['camera_profile'] if effective_execution_contract else THREE_RGB_HEAD_EVAL_CAMERA_PROFILE),
        gripper_visual_profile=(effective_execution_contract['gripper_visual_profile'] if effective_execution_contract else GRIPPER_VISUAL_PROFILE_INWARD_12CM_OFFICIAL_CAMERA),
        # Resident resets replace the arms. The callback applies the approved
        # profile per task, after reset; geometry compilation has no Oracle frame.
        defer_inward_arm_presentation=True,
        install_private_head_segmentation_side_channel=install_private_head_segmentation,
        expected_task_id=primary.task_id,
        partial_scene_load=args.partial_scene_load,
        filter_embedded_scene_agents=filter_embedded_scene_agents,
        use_wrist_evidence_reset_pose=model_prepared_states is None and oracle_prepared_states is None,
        resident_scene_file=union_path,
        resident_scene_model=scene_model,
        evaluator_module_configurer=evaluator_module_configurer,
        native_task_metric=selected_l0 is None,
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
