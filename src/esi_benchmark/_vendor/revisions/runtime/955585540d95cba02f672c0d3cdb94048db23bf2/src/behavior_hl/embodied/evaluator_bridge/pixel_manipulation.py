"""Pure helpers for observation-grounded physical manipulation canaries."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from behavior_hl.embodied.policy.observations import CAMERA_RESOLUTIONS


def load_public_snapshot(path: Path) -> dict:
    """Load the policy-side snapshot without accepting pickle payloads."""

    with np.load(path, allow_pickle=False) as snapshot:
        expected = {"proprio", "cam_rel_poses"}
        for role in CAMERA_RESOLUTIONS:
            expected.add(f"{role}_rgb")
            expected.add(f"{role}_depth_linear")
        if set(snapshot.files) != expected:
            raise ValueError("public snapshot fields do not match the whitelist")
        return {
            "proprio": np.array(snapshot["proprio"], copy=True),
            "cam_rel_poses": np.array(
                snapshot["cam_rel_poses"],
                copy=True,
            ),
            "cameras": {
                role: {
                    "rgb": np.array(snapshot[f"{role}_rgb"], copy=True),
                    "depth_linear": np.array(
                        snapshot[f"{role}_depth_linear"],
                        copy=True,
                    ),
                }
                for role in CAMERA_RESOLUTIONS
            },
        }


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("robot quaternion must be a finite xyzw vector")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        raise ValueError("robot quaternion has zero norm")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def robot_base_point_to_world(
    point: Sequence[float],
    *,
    robot_position: Sequence[float],
    robot_orientation_xyzw: Sequence[float],
) -> np.ndarray:
    point_array = np.asarray(point, dtype=np.float64)
    position = np.asarray(robot_position, dtype=np.float64)
    if point_array.shape != (3,) or not np.isfinite(point_array).all():
        raise ValueError("robot-base point must be finite and 3D")
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("robot position must be finite and 3D")
    rotation = quaternion_xyzw_to_matrix(robot_orientation_xyzw)
    return position + rotation @ point_array


def validate_local_manipulation_target(
    point: Sequence[float],
    *,
    maximum_horizontal_reach_m: float,
    minimum_height_m: float,
    maximum_height_m: float,
) -> dict[str, float]:
    point_array = np.asarray(point, dtype=np.float64)
    if point_array.shape != (3,) or not np.isfinite(point_array).all():
        raise ValueError("manipulation target must be finite and 3D")
    horizontal_reach = float(np.linalg.norm(point_array[:2]))
    if horizontal_reach > maximum_horizontal_reach_m:
        raise ValueError(
            f"target is outside local arm reach: {horizontal_reach:.6f} m"
        )
    if not minimum_height_m <= float(point_array[2]) <= maximum_height_m:
        raise ValueError(
            f"target height is outside the safe interval: {point_array[2]:.6f} m"
        )
    return {
        "horizontal_reach_m": horizontal_reach,
        "height_m": float(point_array[2]),
    }


def public_snapshot_metadata(snapshot: Mapping) -> dict:
    return {
        "proprio_shape": list(np.asarray(snapshot["proprio"]).shape),
        "cam_rel_poses_shape": list(
            np.asarray(snapshot["cam_rel_poses"]).shape
        ),
        "camera_shapes": {
            role: {
                "rgb": list(np.asarray(camera["rgb"]).shape),
                "depth_linear": list(
                    np.asarray(camera["depth_linear"]).shape
                ),
            }
            for role, camera in snapshot["cameras"].items()
        },
    }
