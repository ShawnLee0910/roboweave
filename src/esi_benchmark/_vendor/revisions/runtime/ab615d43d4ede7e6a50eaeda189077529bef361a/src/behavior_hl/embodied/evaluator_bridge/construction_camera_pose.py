"""Measured R1Pro head-camera geometry for construction-only calibration."""
import math
from typing import Any, Mapping

import numpy as np

from behavior_hl.embodied.policy.observations import CAMERA_SENSOR_NAMES
from behavior_hl.embodied.evaluator_bridge.on_top_support_layers import (
    quaternion_to_matrix_xyzw, matrix_to_quaternion_xyzw, quaternion_angle_rad,
)


CAMERA_CAPTURE_SCHEMA = "registered-camera-pose-capture/v1"


def _array(value: Any, shape: tuple[int, ...], field: str) -> np.ndarray:
    item = value
    for method_name in ("detach", "cpu"):
        method = getattr(item, method_name, None)
        if callable(method):
            item = method()
    numpy = getattr(item, "numpy", None)
    if callable(numpy):
        item = numpy()
    result = np.asarray(item, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{field} must be finite with shape {shape}")
    return result


def _pose_matrix(position: Any, orientation: Any) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = quaternion_to_matrix_xyzw(
        _array(orientation, (4,), "orientation")
    )
    result[:3, 3] = _array(position, (3,), "position")
    return result


def _image_shape(evaluator: Any, sensor: Any, sensor_name: str) -> list[int] | None:
    height = getattr(sensor, "image_height", None)
    width = getattr(sensor, "image_width", None)
    try:
        height, width = int(height), int(width)
    except (TypeError, ValueError):
        height = width = 0
    if height > 0 and width > 0:
        return [height, width]
    # Only fall back to an already-materialized RGB observation.  The live
    # observation may be CUDA-backed, so the official sensor resolution must
    # be preferred and conversion failures must not hide the real readback.
    observation = getattr(evaluator, "obs", None)
    if isinstance(observation, Mapping):
        for key, value in observation.items():
            if not (isinstance(key, str) and key.endswith("::rgb") and sensor_name in key):
                continue
            try:
                array = np.asarray(value)
            except (TypeError, ValueError):
                continue
            if array.ndim >= 2:
                return [int(array.shape[0]), int(array.shape[1])]
    return None


def capture_head_camera_pose(evaluator: Any) -> dict[str, Any]:
    """Capture the live R1Pro base/head extrinsic and head intrinsics."""

    from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
        _capture_live_reset_robot_rig,
        _live_head_camera_sensor,
    )
    rig = _capture_live_reset_robot_rig(evaluator)
    sensor_name, sensor = _live_head_camera_sensor(evaluator)
    robot_position = _array(rig["base_world_position"], (3,), "robot_position")
    robot_orientation = _array(rig["base_quaternion_xyzw"], (4,), "robot_orientation")
    head_position = _array(rig["camera_world_position"], (3,), "head_position")
    head_orientation = _array(rig["camera_quaternion_xyzw"], (4,), "head_orientation")
    intrinsic = _array(getattr(sensor, "intrinsic_matrix", None), (3, 3), "intrinsic_matrix")
    return {
        "schema_version": CAMERA_CAPTURE_SCHEMA,
        "sensor_name": sensor_name,
        "robot_position": robot_position.tolist(),
        "robot_orientation": robot_orientation.tolist(),
        "head_position": head_position.tolist(),
        "head_orientation": head_orientation.tolist(),
        "intrinsic_matrix": intrinsic.tolist(),
        "image_shape": _image_shape(evaluator, sensor, sensor_name),
        "robot_yaw_deg": float(rig["base_yaw_deg"]),
        "camera_yaw_deg": float(rig["camera_yaw_deg"]),
        "camera_pitch_deg": float(rig["camera_pitch_deg"]),
        "camera_forward_world": list(rig["camera_forward_world"]),
        "base_from_camera_matrix": rig["base_from_camera_matrix"],
        "camera_forward_axis_local": list(rig["camera_forward_axis_local"]),
        "source": rig["source"],
    }


def camera_pose_at_robot_pose(
    camera_capture: Mapping[str, Any], world_xy: Any, yaw_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly move a measured base-to-head sensor transform."""

    base_position = _array(camera_capture["robot_position"], (3,), "robot_position")
    base_orientation = _array(camera_capture["robot_orientation"], (4,), "robot_orientation")
    head_position = _array(camera_capture["head_position"], (3,), "head_position")
    head_orientation = _array(camera_capture["head_orientation"], (4,), "head_orientation")
    xy = _array(world_xy, (2,), "world_xy")
    if not math.isfinite(float(yaw_deg)):
        raise ValueError("yaw_deg must be finite")
    world_from_base = _pose_matrix(base_position, base_orientation)
    world_from_head = _pose_matrix(head_position, head_orientation)
    base_from_head = np.linalg.inv(world_from_base) @ world_from_head
    target = np.eye(4, dtype=float)
    yaw = math.radians(float(yaw_deg))
    target[:3, :3] = quaternion_to_matrix_xyzw(
        [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
    )
    target[:3, 3] = [xy[0], xy[1], base_position[2]]
    world_from_target_head = target @ base_from_head
    return world_from_target_head[:3, 3], matrix_to_quaternion_xyzw(
        world_from_target_head[:3, :3]
    )


class MeasuredHeadCamera:
    """Minimal sensor adapter accepted by the existing renderer projector."""

    def __init__(self, camera_capture: Mapping[str, Any], world_xy: Any, yaw_deg: float):
        position, orientation = camera_pose_at_robot_pose(camera_capture, world_xy, yaw_deg)
        self._position = position
        self._orientation = orientation
        self.intrinsic_matrix = _array(
            camera_capture["intrinsic_matrix"], (3, 3), "intrinsic_matrix"
        )

    def get_position_orientation(self):
        return self._position, self._orientation


def measured_head_camera(
    camera_capture: Mapping[str, Any], world_xy: Any, yaw_deg: float
) -> MeasuredHeadCamera:
    return MeasuredHeadCamera(camera_capture, world_xy, yaw_deg)


def head_camera_position(evaluator):
    sensor = evaluator.env.robots[0].sensors[CAMERA_SENSOR_NAMES['head']]
    position, _ = sensor.get_position_orientation()
    if hasattr(position, 'detach'):
        position = position.detach().cpu().numpy()
    value = np.asarray(position, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('invalid live head camera position')
    return value


def orientation_toward_camera(reference_xyzw, object_xyz, camera_xyz):
    """Reference quaternion is calibrated with the observer on world +X."""
    reference = np.asarray(reference_xyzw, dtype=float)
    obj, camera = np.asarray(object_xyz, dtype=float), np.asarray(camera_xyz, dtype=float)
    if (reference.shape != (4,) or not np.isfinite(reference).all()
            or not np.isclose(np.linalg.norm(reference), 1)
            or obj.shape != (3,) or camera.shape != (3,)
            or not np.isfinite([obj, camera]).all()):
        raise ValueError('invalid camera-relative pose geometry')
    delta = camera[:2] - obj[:2]
    if np.linalg.norm(delta) < 1e-6:
        raise ValueError('camera-relative yaw is undefined above the object')
    yaw = math.atan2(delta[1], delta[0])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    return matrix_to_quaternion_xyzw(rotation @ quaternion_to_matrix_xyzw(reference))


def camera_pose_readback(rules, camera_xyz, role_pose):
    reports = []
    for role, rule in rules.items():
        tolerance = rule['maximum_error_rad']
        if type(tolerance) not in (float, int) or not math.isfinite(tolerance) or not 0 < tolerance < math.pi / 4:
            raise ValueError('camera pose tolerance must be calibrated in (0, pi/4)')
        position, orientation = role_pose(role)
        expected = orientation_toward_camera(rule['reference_orientation_xyzw'], position, camera_xyz)
        error = quaternion_angle_rad(orientation, expected)
        reports.append(dict(role=role, angular_error_rad=error, maximum_error_rad=tolerance,
                            passed=bool(error <= tolerance)))
    return reports


def camera_at_robot_pose(camera_capture, world_xy, yaw_deg):
    """Rigidly transform the measured head offset; no second camera model."""
    if "head_orientation" in camera_capture:
        return camera_pose_at_robot_pose(camera_capture, world_xy, yaw_deg)[0]
    position = np.asarray(camera_capture['robot_position'], dtype=float)
    rotation = quaternion_to_matrix_xyzw(camera_capture['robot_orientation'])
    offset = rotation.T @ (np.asarray(camera_capture['head_position']) - position)
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([world_xy[0], world_xy[1], position[2]]) + np.array(
        [[c, -s, 0], [s, c, 0], [0, 0, 1.]]) @ offset
