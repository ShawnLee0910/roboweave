"""Observation-only image-point to robot-base 3D grounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from behavior_hl.embodied.policy.observations import CAMERA_RESOLUTIONS


CAMERA_ORDER = ("left_wrist", "right_wrist", "head")


class RGBDGroundingError(ValueError):
    """Raised when a public image point cannot be grounded safely."""


@dataclass(frozen=True, slots=True)
class GroundedPoint:
    camera: str
    normalized_point: tuple[float, float]
    pixel_xy: tuple[int, int]
    depth_m: float
    camera_xyz_m: tuple[float, float, float]
    robot_base_xyz_m: tuple[float, float, float]
    depth_sample_count: int
    depth_spread_m: float


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise RGBDGroundingError("camera quaternion must be a finite xyzw vector")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        raise RGBDGroundingError("camera quaternion has zero norm")
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


def normalized_to_pixel(
    point: Sequence[float],
    *,
    width: int,
    height: int,
) -> tuple[int, int]:
    normalized = np.asarray(point, dtype=np.float64)
    if (
        normalized.shape != (2,)
        or not np.isfinite(normalized).all()
        or np.any(normalized < 0.0)
        or np.any(normalized > 1.0)
    ):
        raise RGBDGroundingError("normalized point must be finite and in [0, 1]")
    x = int(np.rint(normalized[0] * (width - 1)))
    y = int(np.rint(normalized[1] * (height - 1)))
    return x, y


def _camera_pose(
    cam_rel_poses: Sequence[float],
    camera: str,
) -> tuple[np.ndarray, np.ndarray]:
    if camera not in CAMERA_ORDER:
        raise RGBDGroundingError(f"unknown camera role: {camera}")
    poses = np.asarray(cam_rel_poses, dtype=np.float64)
    if poses.shape != (21,) or not np.isfinite(poses).all():
        raise RGBDGroundingError("cam_rel_poses must be a finite 21D vector")
    start = CAMERA_ORDER.index(camera) * 7
    pose = poses[start : start + 7]
    return pose[:3], quaternion_xyzw_to_matrix(pose[3:])


def ground_image_point(
    observation: Mapping,
    *,
    camera: str,
    point: Sequence[float],
    intrinsic_matrix: Sequence[Sequence[float]],
    depth_radius_px: int = 1,
    minimum_depth_m: float = 0.05,
    maximum_depth_m: float = 5.0,
    maximum_depth_spread_m: float = 0.08,
) -> GroundedPoint:
    """Ground a selected public RGB-D pixel in the robot base frame."""

    if camera not in CAMERA_RESOLUTIONS:
        raise RGBDGroundingError(f"unknown camera role: {camera}")
    height, width = CAMERA_RESOLUTIONS[camera]
    depth = np.asarray(
        observation["cameras"][camera]["depth_linear"],
        dtype=np.float64,
    ).squeeze()
    if depth.shape != (height, width):
        raise RGBDGroundingError(
            f"{camera} depth shape {depth.shape} is not {(height, width)}"
        )
    if depth_radius_px < 0 or depth_radius_px > 3:
        raise RGBDGroundingError("depth_radius_px must be in [0, 3]")

    x, y = normalized_to_pixel(point, width=width, height=height)
    x0 = max(0, x - depth_radius_px)
    x1 = min(width, x + depth_radius_px + 1)
    y0 = max(0, y - depth_radius_px)
    y1 = min(height, y + depth_radius_px + 1)
    samples = depth[y0:y1, x0:x1].reshape(-1)
    samples = samples[
        np.isfinite(samples)
        & (samples >= minimum_depth_m)
        & (samples <= maximum_depth_m)
    ]
    if samples.size == 0:
        raise RGBDGroundingError("selected pixel has no valid local depth")
    depth_m = float(np.median(samples))
    spread = float(np.max(np.abs(samples - depth_m)))
    if spread > maximum_depth_spread_m:
        raise RGBDGroundingError(
            f"selected depth neighborhood crosses a discontinuity: {spread:.6f} m"
        )

    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise RGBDGroundingError("intrinsic_matrix must be finite and 3x3")
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise RGBDGroundingError("camera focal lengths must be positive")
    camera_point = np.asarray(
        [
            (float(x) - cx) * depth_m / fx,
            (float(y) - cy) * depth_m / fy,
            depth_m,
        ],
        dtype=np.float64,
    )

    relative_position, relative_rotation = _camera_pose(
        observation["cam_rel_poses"],
        camera,
    )
    # Match OmniGibson eval.utils.obs_utils.depth_to_pcd: the renderer's
    # camera convention needs a 180-degree rotation around camera X.
    renderer_to_camera = np.diag([1.0, -1.0, -1.0])
    robot_point = (
        relative_position
        + relative_rotation @ renderer_to_camera @ camera_point
    )
    return GroundedPoint(
        camera=camera,
        normalized_point=(float(point[0]), float(point[1])),
        pixel_xy=(x, y),
        depth_m=depth_m,
        camera_xyz_m=tuple(float(value) for value in camera_point),
        robot_base_xyz_m=tuple(float(value) for value in robot_point),
        depth_sample_count=int(samples.size),
        depth_spread_m=spread,
    )
