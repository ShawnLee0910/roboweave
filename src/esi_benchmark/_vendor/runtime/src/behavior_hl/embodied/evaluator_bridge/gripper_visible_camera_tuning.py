"""Declared camera framing that exposes more of R1Pro's two grippers.

The tuning is intentionally independent from the Symbolic hold geometry.  It
widens the two RGB-only wrist cameras without moving their initialized sensor
prims, and pitches the head camera slightly downward.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    qualify_r1pro_camera_sensor_keys,
)


GRIPPER_VISIBLE_CAMERA_TUNING_VERSION = (
    "behavior2026.gripper-visible-camera-tuning/v4"
)
WRIST_FOCAL_LENGTH_MM = 14.5
HEAD_FOCAL_LENGTH_MM = 12.0
# Live translation of an initialized R1Pro wrist VisionSensor can place its
# optical center inside the wrist housing (the v138 evidence became uniformly
# black in both wrist panels).  Widening the lens is sufficient to expose more
# gripper without moving the authored sensor prim.
WRIST_RETREAT_ALONG_LOCAL_Z_M = 0.0
# A moderate downward look retains scene context without filling the lower
# edge with the robot shell.  The predeclared wider head lens supplies the
# horizontal coverage needed for both hands.
HEAD_DOWNWARD_PITCH_DEG = 10.0


class GripperVisibleCameraTuningError(RuntimeError):
    """Raised when the live three-camera rig cannot be tuned safely."""


def _cpu_array(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    return np.asarray(value, dtype=np.float64)


def _vector(value: Any, *, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise GripperVisibleCameraTuningError(
            f"{name} must contain {length} finite values"
        )
    return array


def _quat_to_matrix_xyzw(value: Any) -> np.ndarray:
    x, y, z, w = _vector(value, length=4, name="quaternion")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1.0e-8:
        raise GripperVisibleCameraTuningError("quaternion cannot be normalized")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_xyzw(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise GripperVisibleCameraTuningError(
            "rotation matrix must be finite and 3x3"
        )
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= float(np.linalg.norm(quaternion))
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def compose_parent_camera_tuning(
    *,
    parent_position: Sequence[float],
    parent_orientation_xyzw: Sequence[float],
    local_translation_m: Sequence[float] = (0.0, 0.0, 0.0),
    local_rotation_x_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-compose one camera-local translation and X-axis rotation."""

    position = _vector(parent_position, length=3, name="parent_position")
    rotation = _quat_to_matrix_xyzw(parent_orientation_xyzw)
    translation = _vector(
        local_translation_m,
        length=3,
        name="local_translation_m",
    )
    angle = math.radians(float(local_rotation_x_deg))
    if not math.isfinite(angle):
        raise GripperVisibleCameraTuningError(
            "local_rotation_x_deg must be finite"
        )
    cosine, sine = math.cos(angle), math.sin(angle)
    local_rotation = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )
    return (
        position + rotation @ translation,
        _matrix_to_quat_xyzw(rotation @ local_rotation),
    )


def gripper_visible_camera_tuning_contract() -> Mapping[str, Any]:
    return {
        "version": GRIPPER_VISIBLE_CAMERA_TUNING_VERSION,
        "wrist_focal_length_mm": WRIST_FOCAL_LENGTH_MM,
        "head_focal_length_mm": HEAD_FOCAL_LENGTH_MM,
        "wrist_retreat_along_local_z_m": WRIST_RETREAT_ALONG_LOCAL_Z_M,
        "head_downward_pitch_deg": HEAD_DOWNWARD_PITCH_DEG,
        "symbolic_hold_changed": False,
        "object_pose_changed": False,
        "camera_conditioned_grasp": False,
        "wrist_focal_length_predeclared": True,
        "head_focal_length_predeclared": True,
    }


def apply_gripper_visible_camera_tuning(evaluator: Any) -> Mapping[str, Any]:
    """Apply the declared local offsets to one already-constructed R1Pro."""

    robot = getattr(evaluator, "robot", None)
    sensors = getattr(robot, "sensors", None)
    robot_name = str(getattr(robot, "name", ""))
    if robot is None or not isinstance(sensors, Mapping) or not robot_name:
        raise GripperVisibleCameraTuningError(
            "live evaluator R1Pro sensors are unavailable"
        )
    names = qualify_r1pro_camera_sensor_keys(robot_name)
    missing = sorted(name for name in names.values() if name not in sensors)
    if missing:
        raise GripperVisibleCameraTuningError(
            f"live camera sensors are missing: {missing}"
        )

    reports: dict[str, Any] = {}
    for role in ("left_wrist", "right_wrist", "head"):
        sensor = sensors[names[role]]
        read_pose = getattr(sensor, "get_position_orientation", None)
        write_pose = getattr(sensor, "set_position_orientation", None)
        if not callable(read_pose) or not callable(write_pose):
            raise GripperVisibleCameraTuningError(
                f"camera {role!r} lacks a parent-pose API"
            )
        before_position, before_orientation = read_pose(frame="parent")
        before_position = _vector(
            _cpu_array(before_position),
            length=3,
            name=f"{role} parent position",
        )
        before_orientation = _vector(
            _cpu_array(before_orientation),
            length=4,
            name=f"{role} parent orientation",
        )
        before_focal = float(sensor.focal_length)
        before_aperture = float(sensor.horizontal_aperture)
        if role in {"left_wrist", "right_wrist"}:
            after_position, after_orientation = compose_parent_camera_tuning(
                parent_position=before_position,
                parent_orientation_xyzw=before_orientation,
                local_translation_m=(0.0, 0.0, 0.0),
            )
            if not math.isclose(
                before_focal,
                WRIST_FOCAL_LENGTH_MM,
                rel_tol=0.0,
                abs_tol=1.0e-5,
            ):
                raise GripperVisibleCameraTuningError(
                    f"camera {role!r} focal length was not predeclared"
                )
            pose_written = False
        else:
            after_position, after_orientation = compose_parent_camera_tuning(
                parent_position=before_position,
                parent_orientation_xyzw=before_orientation,
                # USD cameras look along local -Z with +Y up.  A negative
                # local-X rotation pitches the optical axis downward.
                local_rotation_x_deg=-HEAD_DOWNWARD_PITCH_DEG,
            )
            write_pose(
                position=after_position,
                orientation=after_orientation,
                frame="parent",
            )
            if not math.isclose(
                before_focal,
                HEAD_FOCAL_LENGTH_MM,
                rel_tol=0.0,
                abs_tol=1.0e-5,
            ):
                raise GripperVisibleCameraTuningError(
                    "camera 'head' focal length was not predeclared"
                )
            pose_written = True
        observed_position, observed_orientation = read_pose(frame="parent")
        observed_position = _cpu_array(observed_position)
        observed_orientation = _cpu_array(observed_orientation)
        if not np.allclose(observed_position, after_position, atol=1.0e-5):
            raise GripperVisibleCameraTuningError(
                f"camera {role!r} position tuning did not persist"
            )
        if not (
            np.allclose(observed_orientation, after_orientation, atol=1.0e-5)
            or np.allclose(observed_orientation, -after_orientation, atol=1.0e-5)
        ):
            raise GripperVisibleCameraTuningError(
                f"camera {role!r} orientation tuning did not persist"
            )
        reports[role] = {
            "sensor_name": names[role],
            "parent_position_before_m": before_position.tolist(),
            "parent_orientation_before_xyzw": before_orientation.tolist(),
            "parent_position_after_m": after_position.tolist(),
            "parent_orientation_after_xyzw": after_orientation.tolist(),
            "focal_length_before_mm": before_focal,
            "focal_length_after_mm": float(sensor.focal_length),
            "horizontal_aperture_before_mm": before_aperture,
            "horizontal_aperture_after_mm": float(sensor.horizontal_aperture),
            "live_pose_written": pose_written,
        }

    import omnigibson as og

    og.sim.render()
    return {
        **dict(gripper_visible_camera_tuning_contract()),
        "applied": True,
        "camera_count": len(reports),
        "cameras": reports,
    }


__all__ = [
    "GRIPPER_VISIBLE_CAMERA_TUNING_VERSION",
    "GripperVisibleCameraTuningError",
    "HEAD_DOWNWARD_PITCH_DEG",
    "HEAD_FOCAL_LENGTH_MM",
    "WRIST_FOCAL_LENGTH_MM",
    "WRIST_RETREAT_ALONG_LOCAL_Z_M",
    "apply_gripper_visible_camera_tuning",
    "compose_parent_camera_tuning",
    "gripper_visible_camera_tuning_contract",
]
