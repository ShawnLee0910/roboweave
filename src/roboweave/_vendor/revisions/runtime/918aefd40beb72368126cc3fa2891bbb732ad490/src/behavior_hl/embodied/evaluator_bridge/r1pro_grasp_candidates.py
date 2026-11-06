"""Pure, geometry-directed grasp candidates for private Oracle authoring.

This module deliberately knows neither BDDL object names nor simulator types.
The live Oracle supplies an object's private geometry and consumes the poses;
none of these fields are policy-visible observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import numpy as np


class GraspCandidateError(ValueError):
    """Raised when private candidate geometry is malformed."""


@dataclass(frozen=True, slots=True)
class GraspCandidate:
    """One bounded physical grasp proposal in the world frame."""

    candidate_id: str
    geometry_family: str
    source: str
    quaternion_convention: str
    approach_direction_convention: str
    pregrasp_position_xyz: tuple[float, float, float]
    contact_position_xyz: tuple[float, float, float]
    top_down_quaternion_xyzw: tuple[float, float, float, float]
    approach_direction_world: tuple[float, float, float]
    closure_axis_world: tuple[float, float, float]
    lift_position_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "geometry_family",
            "source",
            "quaternion_convention",
            "approach_direction_convention",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise GraspCandidateError(f"{name} must be non-empty")
        for name in (
            "pregrasp_position_xyz",
            "contact_position_xyz",
            "approach_direction_world",
            "closure_axis_world",
            "lift_position_xyz",
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name))
        quaternion = np.asarray(
            _vector(
                self.top_down_quaternion_xyzw,
                4,
                "top_down_quaternion_xyzw",
            ),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-8:
            raise GraspCandidateError("grasp quaternion has zero norm")
        object.__setattr__(
            self,
            "top_down_quaternion_xyzw",
            tuple(float(value) for value in quaternion / norm),
        )
        for name in ("approach_direction_world", "closure_axis_world"):
            vector = np.asarray(getattr(self, name), dtype=np.float64)
            if not math.isclose(float(np.linalg.norm(vector)), 1.0, abs_tol=1e-6):
                raise GraspCandidateError(f"{name} must be unit length")


GraspCandidateGenerator = Callable[..., tuple[GraspCandidate, ...]]


def sphere_grasp_candidates(
    *,
    object_center_xyz: Sequence[float],
    object_extent_xyz: Sequence[float],
    fingertip_offset_m: float,
    approach_clearance_m: float = 0.01,
    lift_m: float = 0.10,
) -> tuple[GraspCandidate, ...]:
    """Return official-top, equator, and one bounded equator refinement.

    BEHAVIOR's official Starter defines ``approach_dir = R(q) @ (0,0,-1)``
    and ``pregrasp = grasp - approach_dir * (fingertip + 0.01)``.  For the
    R1Pro top-down x-flip, approach is therefore world ``-z``.  B/C preserve
    the official top pregrasp height but use the motion-constrained final
    approach to reach the sphere equator instead of stopping at its AABB top.
    """

    center = np.asarray(_vector(object_center_xyz, 3, "object_center_xyz"))
    extent = np.asarray(_vector(object_extent_xyz, 3, "object_extent_xyz"))
    if np.any(extent <= 0.0):
        raise GraspCandidateError("object_extent_xyz must be positive")
    fingertip = _positive(fingertip_offset_m, "fingertip_offset_m")
    clearance = _positive(approach_clearance_m, "approach_clearance_m")
    lift = _positive(lift_m, "lift_m")
    if lift < 0.05:
        raise GraspCandidateError("lift_m must be at least 0.05")

    top = center + np.asarray((0.0, 0.0, 0.5 * extent[2]))
    pregrasp = top + np.asarray((0.0, 0.0, fingertip + clearance))
    approach = (0.0, 0.0, -1.0)
    definitions = (
        (
            "A_official_top",
            top,
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            "official_starter_top_surface_baseline",
        ),
        (
            "B_sphere_equator",
            center,
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            "sphere_equator_motion_constrained_approach",
        ),
        (
            "C_sphere_equator_zp5mm_yaw90",
            center + np.asarray((0.0, 0.0, 0.005)),
            (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
            (1.0, 0.0, 0.0),
            "bounded_sphere_equator_refinement",
        ),
    )
    candidates = []
    for candidate_id, contact, quaternion, closure, source in definitions:
        candidates.append(
            GraspCandidate(
                candidate_id=candidate_id,
                geometry_family="sphere",
                source=source,
                quaternion_convention="final_r1pro_eef_xyzw_post_x_flip",
                approach_direction_convention=(
                    "robot_agnostic_grasp_quaternion_pre_r1pro_x_flip"
                ),
                pregrasp_position_xyz=tuple(float(value) for value in pregrasp),
                contact_position_xyz=tuple(float(value) for value in contact),
                top_down_quaternion_xyzw=quaternion,
                approach_direction_world=approach,
                closure_axis_world=closure,
                lift_position_xyz=tuple(
                    float(value)
                    for value in contact + np.asarray((0.0, 0.0, lift))
                ),
            )
        )
    return tuple(candidates)


GRASP_CANDIDATE_GENERATORS: Mapping[str, GraspCandidateGenerator] = {
    "sphere": sphere_grasp_candidates,
}


def generate_grasp_candidates(
    geometry_family: str,
    **geometry: object,
) -> tuple[GraspCandidate, ...]:
    """Dispatch through the explicit geometry-family registry."""

    generator = GRASP_CANDIDATE_GENERATORS.get(geometry_family)
    if generator is None:
        raise GraspCandidateError(
            f"unsupported grasp geometry family: {geometry_family!r}"
        )
    return generator(**geometry)


def _vector(value: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise GraspCandidateError(f"{name} must be a finite {size}D vector")
    return tuple(float(item) for item in array)


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraspCandidateError(f"{name} must be finite and positive")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise GraspCandidateError(f"{name} must be finite and positive")
    return converted
