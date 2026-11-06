"""Pure geometry for a visually grounded Symbolic two-finger hold.

One point inside the held object is anchored to the midpoint of BEHAVIOR's
official assisted-grasp rays.  The point is near the object's proximal edge,
so most of the object extends toward the fingertips instead of backward into
the wrist camera.  Cameras never participate in the pose calculation, so an
object cannot become a camera-relative billboard.  The thinnest object
bounding-box axis is aligned with the parallel-jaw closing axis, which gives
one generic rule for books, batteries, balls, and box-shaped household
objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


SYMBOLIC_GRIPPER_HOLD_VERSION = "behavior2026.symbolic-gripper-hold/v8"
MAX_PROXIMAL_EDGE_INSET_M = 0.01


class SymbolicGripperHoldError(ValueError):
    """Raised when live gripper geometry cannot produce a finite hold pose."""


@dataclass(frozen=True, slots=True)
class SymbolicGripperHold:
    eef_from_object: np.ndarray
    grasp_center_in_eef_m: np.ndarray
    closing_axis_in_eef: np.ndarray
    object_thin_axis_index: int
    object_extent_m: np.ndarray
    object_grasp_point_in_object_frame_m: np.ndarray
    object_center_offset_from_grasp_m: np.ndarray
    proximal_edge_inset_m: float
    world_orientation_preserved: bool = False
    version: str = SYMBOLIC_GRIPPER_HOLD_VERSION

    def __post_init__(self) -> None:
        matrix = _matrix4(self.eef_from_object, "eef_from_object")
        center = _vector3(self.grasp_center_in_eef_m, "grasp_center_in_eef_m")
        closing = _unit_vector3(self.closing_axis_in_eef, "closing_axis_in_eef")
        extent = _positive_vector3(self.object_extent_m, "object_extent_m")
        object_grasp_point = _vector3(
            self.object_grasp_point_in_object_frame_m,
            "object_grasp_point_in_object_frame_m",
        )
        center_offset = _vector3(
            self.object_center_offset_from_grasp_m,
            "object_center_offset_from_grasp_m",
        )
        edge_inset = float(self.proximal_edge_inset_m)
        if not math.isfinite(edge_inset) or edge_inset <= 0.0:
            raise SymbolicGripperHoldError(
                "proximal_edge_inset_m must be finite and positive"
            )
        thin_axis = int(self.object_thin_axis_index)
        if thin_axis not in (0, 1, 2):
            raise SymbolicGripperHoldError(
                "object_thin_axis_index must be 0, 1, or 2"
            )
        object.__setattr__(self, "eef_from_object", matrix)
        object.__setattr__(self, "grasp_center_in_eef_m", center)
        object.__setattr__(self, "closing_axis_in_eef", closing)
        object.__setattr__(self, "object_extent_m", extent)
        object.__setattr__(
            self,
            "object_grasp_point_in_object_frame_m",
            object_grasp_point,
        )
        object.__setattr__(
            self,
            "object_center_offset_from_grasp_m",
            center_offset,
        )
        object.__setattr__(self, "proximal_edge_inset_m", edge_inset)

    def private_report(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "anchor_source": "official_assisted_grasp_ray_midpoint",
            "object_grasp_point_rule": (
                "proximal_edge_inset_along_object_long_axis"
            ),
            "orientation_rule": (
                "preserve_original_world_orientation"
                if self.world_orientation_preserved
                else "thinnest_bbox_axis_to_parallel_jaw_closing_axis"
            ),
            "grasp_center_in_eef_m": self.grasp_center_in_eef_m.tolist(),
            "closing_axis_in_eef": self.closing_axis_in_eef.tolist(),
            "object_thin_axis_index": self.object_thin_axis_index,
            "object_extent_m": self.object_extent_m.tolist(),
            "object_grasp_point_in_object_frame_m": (
                self.object_grasp_point_in_object_frame_m.tolist()
            ),
            "object_center_offset_from_grasp_m": (
                self.object_center_offset_from_grasp_m.tolist()
            ),
            "proximal_edge_inset_m": self.proximal_edge_inset_m,
            "world_orientation_preserved": self.world_orientation_preserved,
            "eef_from_object_matrix": self.eef_from_object.tolist(),
            "camera_conditioned": False,
            "task_or_category_conditioned": False,
            "carry_pose_rule": (
                "fixed_eef_to_object_transform_from_grasp_until_release"
            ),
            "carry_relative_transform_frozen": True,
            "first_after_capture_warmup_projection_cycle": True,
            "first_after_capture_rig_pose_reassertion": True,
            "intermediate_grasp_frames": 0,
        }


def build_symbolic_gripper_hold(
    *,
    world_from_eef: Any,
    assisted_grasp_start_points_world: Sequence[Sequence[float]],
    assisted_grasp_end_points_world: Sequence[Sequence[float]],
    object_extent_m: Sequence[float],
    object_bbox_center_in_object_frame_m: Sequence[float],
    original_world_from_object: Any | None = None,
    preserve_world_orientation: bool = False,
) -> SymbolicGripperHold:
    """Build an EEF-relative object pose held near its proximal edge.

    Start and end points are the official assisted-grasp ray endpoints on the
    opposing fingers.  Their joint midpoint defines the object contact point;
    the vector between their centroids defines the closing axis.  The EEF +Z
    axis supplies the along-finger direction.  The contact point lies a small,
    bounded distance inside the object's proximal edge.  This keeps most of a
    large object in front of the gripper while remaining camera-independent.
    No camera transform is accepted.  Callers can preserve the root object's
    original world orientation while using the same translated grasp anchor;
    callers that omit it retain the thin-axis alignment rule.
    """

    eef = _matrix4(world_from_eef, "world_from_eef")
    starts = _point_cloud3(
        assisted_grasp_start_points_world,
        "assisted_grasp_start_points_world",
    )
    ends = _point_cloud3(
        assisted_grasp_end_points_world,
        "assisted_grasp_end_points_world",
    )
    extent = _positive_vector3(object_extent_m, "object_extent_m")
    bbox_center_local = _vector3(
        object_bbox_center_in_object_frame_m,
        "object_bbox_center_in_object_frame_m",
    )

    start_center = starts.mean(axis=0)
    end_center = ends.mean(axis=0)
    closing_world = _unit_vector3(
        end_center - start_center,
        "assisted-grasp closing axis",
    )
    grasp_center_world = np.concatenate([starts, ends], axis=0).mean(axis=0)

    finger_world = eef[:3, 2].copy()
    finger_world = finger_world - closing_world * float(
        np.dot(finger_world, closing_world)
    )
    finger_world = _unit_vector3(finger_world, "EEF along-finger axis")
    lateral_world = _unit_vector3(
        np.cross(closing_world, finger_world),
        "gripper lateral axis",
    )
    # Re-orthogonalize after floating-point projection.  This preserves the
    # official EEF convention: X=lateral, Y=closing, Z=along-finger.
    finger_world = _unit_vector3(
        np.cross(lateral_world, closing_world),
        "orthogonalized along-finger axis",
    )

    order = np.argsort(extent, kind="stable")
    thin_axis = int(order[0])
    middle_axis = int(order[1])
    long_axis = int(order[2])
    if preserve_world_orientation:
        if original_world_from_object is None:
            raise SymbolicGripperHoldError(
                "original_world_from_object is required when preserving orientation"
            )
        world_from_object_rotation = _matrix4(
            original_world_from_object,
            "original_world_from_object",
        )[:3, :3]
    else:
        world_from_object_rotation = np.zeros((3, 3), dtype=np.float64)
        world_from_object_rotation[:, thin_axis] = closing_world
        world_from_object_rotation[:, middle_axis] = lateral_world
        world_from_object_rotation[:, long_axis] = finger_world
        if float(np.linalg.det(world_from_object_rotation)) < 0.0:
            world_from_object_rotation[:, middle_axis] *= -1.0
    if not np.allclose(
        world_from_object_rotation.T @ world_from_object_rotation,
        np.eye(3),
        atol=1e-7,
    ) or not math.isclose(
        float(np.linalg.det(world_from_object_rotation)),
        1.0,
        abs_tol=1e-7,
    ):
        raise SymbolicGripperHoldError(
            "computed object hold orientation is not a proper rotation"
        )

    # Pinch near the proximal edge instead of placing the complete object
    # center at the EEF origin.  Centering a large book at the origin makes
    # half of it extend backward through the wrist camera.  The inset is
    # bounded by both an absolute limit and one quarter of the object's length
    # so the rule also remains valid for small objects.
    proximal_edge_inset = min(
        MAX_PROXIMAL_EDGE_INSET_M,
        float(extent[long_axis]) * 0.25,
    )
    object_grasp_point_local = bbox_center_local.copy()
    object_grasp_point_local[long_axis] -= (
        float(extent[long_axis]) * 0.5 - proximal_edge_inset
    )

    world_from_object = np.eye(4, dtype=np.float64)
    world_from_object[:3, :3] = world_from_object_rotation
    world_from_object[:3, 3] = (
        grasp_center_world
        - world_from_object_rotation @ object_grasp_point_local
    )
    eef_from_object = np.linalg.inv(eef) @ world_from_object
    grasp_center_in_eef = (
        np.linalg.inv(eef)
        @ np.asarray([*grasp_center_world, 1.0], dtype=np.float64)
    )[:3]
    closing_axis_in_eef = eef[:3, :3].T @ closing_world
    object_center_world = (
        world_from_object
        @ np.asarray([*bbox_center_local, 1.0], dtype=np.float64)
    )[:3]
    object_center_offset_from_grasp = (
        eef[:3, :3].T @ (object_center_world - grasp_center_world)
    )
    return SymbolicGripperHold(
        eef_from_object=eef_from_object,
        grasp_center_in_eef_m=grasp_center_in_eef,
        closing_axis_in_eef=closing_axis_in_eef,
        object_thin_axis_index=thin_axis,
        object_extent_m=extent,
        object_grasp_point_in_object_frame_m=object_grasp_point_local,
        object_center_offset_from_grasp_m=object_center_offset_from_grasp,
        proximal_edge_inset_m=proximal_edge_inset,
        world_orientation_preserved=preserve_world_orientation,
    )


def _matrix4(value: Any, name: str) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise SymbolicGripperHoldError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise SymbolicGripperHoldError(f"{name} is not a homogeneous transform")
    return matrix.copy()


def _vector3(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise SymbolicGripperHoldError(f"{name} must contain three finite values")
    return vector.copy()


def _unit_vector3(value: Any, name: str) -> np.ndarray:
    vector = _vector3(value, name)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise SymbolicGripperHoldError(f"{name} cannot be normalized")
    return vector / norm


def _positive_vector3(value: Any, name: str) -> np.ndarray:
    vector = _vector3(value, name)
    if np.any(vector <= 0.0):
        raise SymbolicGripperHoldError(f"{name} must contain positive values")
    return vector


def _point_cloud3(value: Any, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) == 0
        or not np.isfinite(points).all()
    ):
        raise SymbolicGripperHoldError(
            f"{name} must be a non-empty finite Nx3 point cloud"
        )
    return points.copy()


__all__ = [
    "SYMBOLIC_GRIPPER_HOLD_VERSION",
    "MAX_PROXIMAL_EDGE_INSET_M",
    "SymbolicGripperHold",
    "SymbolicGripperHoldError",
    "build_symbolic_gripper_hold",
]
