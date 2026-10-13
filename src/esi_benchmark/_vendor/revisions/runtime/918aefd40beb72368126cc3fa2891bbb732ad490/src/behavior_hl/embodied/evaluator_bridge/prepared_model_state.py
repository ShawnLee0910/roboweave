"""Replay recorded robot/camera state. No pose search, untuck, or IK fallback."""
from collections.abc import Mapping
import math

import numpy as np


class PreparedModelStateError(RuntimeError):
    session_invalid = True


def require_semantic_reset(attestation):
    numerical = {'error_m', 'error_rad', 'max_error_rad', 'numeric_error'}
    failures = [dict(f) for f in attestation.failures if not numerical.intersection(f)]
    if failures:
        raise PreparedModelStateError(f'reset identity or semantics changed: {failures}')
    return {'passed': True, 'failures': [],
            'numeric_diagnostic_failures': [dict(f) for f in attestation.failures],
            'admission_policy': 'oracle_prepared_identity_and_semantics_v1'}


def array(value, shape, name):
    for method in ('detach', 'cpu'):
        if callable(getattr(value, method, None)):
            value = getattr(value, method)()
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise PreparedModelStateError(f'{name} must have finite values and shape {shape}')
    return result


def validate_bundle(bundle, *, instruction_id, seed, observation_profile):
    if bundle.get('schema_version') != 'oracle-prepared-model-state/v1' or bundle.get('instruction_id') != instruction_id:
        raise PreparedModelStateError('prepared Instruction identity mismatch')
    if type(bundle.get('seed')) is not int or bundle['seed'] != seed:
        raise PreparedModelStateError('prepared seed mismatch')
    contract = bundle['execution_contract']
    if contract.get('action_mode') == 'Physical':
        expected = dict(action_mode='Physical', physics_steps_per_action=None, scene_load='Full',
                        observation_profile=observation_profile, overhead_view_mode='off')
    else:
        expected = dict(action_mode='Symbolic', physics_steps_per_action=0, scene_load='Full',
                        observation_profile=observation_profile, overhead_view_mode='off')
        if type(contract.get('physics_steps_per_action')) is not int:
            raise PreparedModelStateError('prepared execution contract mismatch')
    if any(contract.get(k) != v for k, v in expected.items()):
        raise PreparedModelStateError('prepared execution contract mismatch')
    if bundle.get('gripper_visual_profile') != 'inward_12cm_head_dominant_official_camera_v1':
        raise PreparedModelStateError('recorded layout/profile differs from the model launcher')
    prepared = bundle['prepared']
    if (prepared.get('schema_version') != 'resident-prepared-initial-state/v1'
            or prepared.get('task_name') != instruction_id
            or prepared.get('oracle_actions_executed') != 0
            or prepared.get('physics_steps_executed') != 0):
        raise PreparedModelStateError('input is not an Oracle pre-action state')
    for name, count in [('robot_position', 3), ('robot_orientation', 4), ('robot_joint_positions', 28)]:
        array(prepared[name], (count,), name)
    if not np.linalg.norm(prepared['robot_orientation']):
        raise PreparedModelStateError('zero robot quaternion')
    structure = bundle['camera_structure']
    if structure.get('camera_profile') != observation_profile or set(structure.get('cameras', {})) != {'head', 'left_wrist', 'right_wrist'}:
        raise PreparedModelStateError('prepared camera configuration mismatch')
    census = structure.get('robot_census', [])
    if len(census) != 1 or census[0].get('model', '').lower() != 'r1pro':
        raise PreparedModelStateError('prepared robot is not the declared R1Pro')
    head = prepared['head_camera_readback']
    if head.get('sensor_name') != structure['cameras']['head']['sensor_name']:
        raise PreparedModelStateError('prepared head sensor identity mismatch')
    for name, size in [('head_position', 3), ('head_orientation', 4)]:
        array(head[name], (size,), name)
    if not np.linalg.norm(head['head_orientation']):
        raise PreparedModelStateError('zero head quaternion')
    intrinsic = array(head['intrinsic_matrix'], (3, 3), 'head intrinsic')
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise PreparedModelStateError('invalid focal projection')
    for camera in structure['cameras'].values():
        if len(camera['resolution']) != 2 or any(type(n) is not int or n <= 0 for n in camera['resolution']):
            raise PreparedModelStateError('invalid image dimensions')
        if 'rgb' not in camera.get('modalities', []):
            raise PreparedModelStateError('required RGB camera missing')
    return prepared


def restore_prepared_model_state(*, evaluator, bundle, instruction_id, seed,
                                 observation_profile, render=None, semantic_check=None):
    """Apply recorded values once; numerical deltas are diagnostics, not gates."""
    try:
        prepared = validate_bundle(bundle, instruction_id=instruction_id, seed=seed,
                                   observation_profile=observation_profile)
        robot = evaluator.robot
        structure = bundle['camera_structure']
        model = str(getattr(robot, 'model', '')).lower()
        expected_model = structure['robot_census'][0]['model'].lower()
        if robot.name != structure['selected_robot_name'] or model != expected_model:
            raise PreparedModelStateError(
                f'live robot identity mismatch: name={robot.name!r}, model={model!r}')
        current_joints = robot.get_joint_positions()
        array(current_joints, (28,), 'live joint layout')
        sensors = robot.sensors
        for role, camera in structure['cameras'].items():
            sensor = sensors.get(camera['sensor_name'])
            if sensor is None or [sensor.image_height, sensor.image_width] != camera['resolution']:
                raise PreparedModelStateError(f'live {role} sensor/size differs from recorded configuration')
        head_record = prepared['head_camera_readback']
        head = sensors[head_record['sensor_name']]
        for obj, methods in [(robot, ['set_joint_positions', 'set_position_orientation']),
                             (head, ['set_position_orientation', 'set_attribute'])]:
            if any(not callable(getattr(obj, name, None)) for name in methods):
                raise PreparedModelStateError('recorded-state setter unavailable')
        focal = float(head.focal_length)
        if not math.isfinite(focal) or focal <= 0:
            raise PreparedModelStateError('live camera focal length is invalid')
        def like(reference, values):
            if callable(getattr(reference, 'new_tensor', None)):
                return reference.new_tensor(values)
            return np.asarray(values, dtype=float)
        # Root/world pose is applied after joints so floating-base DOFs cannot
        # displace the requested world pose as a side effect of joint restore.
        robot.set_joint_positions(like(current_joints, prepared['robot_joint_positions']), drive=False)
        position, orientation = robot.get_position_orientation()
        robot.set_position_orientation(like(position, prepared['robot_position']),
                                       like(orientation, prepared['robot_orientation']))
        tuning = bundle['camera_tuning']
        if tuning.get('applied') is True:
            for role, camera in structure['cameras'].items():
                value = float(tuning['cameras'][role]['focal_length_after_mm'])
                if not math.isfinite(value) or value <= 0:
                    raise PreparedModelStateError('invalid recorded camera focal length')
                sensors[camera['sensor_name']].focal_length = value
        elif tuning.get('applied') is not False:
            raise PreparedModelStateError('camera tuning state is missing')
        else:
            # False means no post-load tuning, not that declared focal lengths
            # may be ignored when moving this prepared state to another run.
            for role, camera in structure['cameras'].items():
                prefix = 'head' if role == 'head' else 'wrist'
                if tuning.get(prefix + '_focal_length_predeclared') is True:
                    value = float(tuning[prefix + '_focal_length_mm'])
                    if not math.isfinite(value) or value <= 0:
                        raise PreparedModelStateError('invalid predeclared camera focal length')
                    sensors[camera['sensor_name']].focal_length = value
        focal = float(head.focal_length)
        # Reconstruct the recorded projection through the native camera API.
        k = array(head_record['intrinsic_matrix'], (3, 3), 'head intrinsic')
        h, w = structure['cameras']['head']['resolution']
        aperture_x, aperture_y = w * focal / k[0, 0], h * focal / k[1, 1]
        for name, value in [('horizontalAperture', aperture_x), ('verticalAperture', aperture_y),
                            ('horizontalApertureOffset', aperture_x * (0.5 - k[0, 2] / w)),
                            ('verticalApertureOffset', aperture_y * (0.5 - k[1, 2] / h))]:
            head.set_attribute(name, float(value))
        hp, hq = head.get_position_orientation()
        head.set_position_orientation(like(hp, head_record['head_position']),
                                      like(hq, head_record['head_orientation']))
        if render is None:
            import omnigibson as og
            render = og.sim.render
        render()  # Render only; no physics step or planner is invoked.
        if semantic_check is not None and semantic_check() is not True:
            raise PreparedModelStateError('declared initial semantics are not satisfied')
        actual_p, actual_q = robot.get_position_orientation()
        actual_joints = array(robot.get_joint_positions(), (28,), 'restored joints')
        actual_p = array(actual_p, (3,), 'restored position')
        array(actual_q, (4,), 'restored orientation')
        array(head.intrinsic_matrix, (3, 3), 'restored camera projection')
        return {'status': 'PASS', 'policy': 'ORACLE_PREPARED_ONLY',
                'instruction_id': instruction_id, 'pose_search': False,
                'untucked_preamble_applied': False, 'inward_planning': False,
                'numerical_deltas_are_diagnostic_only': True,
                'position_delta_m': float(np.linalg.norm(actual_p - prepared['robot_position'])),
                'maximum_joint_readback_delta': float(np.max(np.abs(actual_joints - prepared['robot_joint_positions'])))}
    except Exception as exc:
        if isinstance(exc, PreparedModelStateError):
            raise
        raise PreparedModelStateError(f'prepared-state restore failed: {type(exc).__name__}: {exc}') from exc
