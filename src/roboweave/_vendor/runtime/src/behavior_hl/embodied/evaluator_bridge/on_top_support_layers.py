"""Task-neutral direct-plane candidate authoring for ``OnTop`` placements.

This module is deliberately simulator-free.  Live AABBs and base-aligned
bounding boxes are proposal evidence only.  The caller must still run a
collision-shape query and the active mode's canonical support readback:
zero-physics geometry for Symbolic, contact / settle for Physical.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    CandidateAdmissionReport,
    DEFAULT_MAX_ANGULAR_SPEED_RAD_S,
    DEFAULT_MAX_LINEAR_SPEED_M_S,
    DEFAULT_STABILITY_WINDOW,
    SIMPLE_NEXT_TO_SCREEN_SELECTORS,
    SpatialCandidatePlan,
    SpatialPoseCandidate,
    SpatialStabilitySample,
    next_to_gap_for_extents,
    object_aabb,
    object_pose,
    summarize_spatial_stability,
)


ON_TOP_SUPPORT_LAYER_VERSION = (
    "behavior2026.symbolic-on-top-support-layers/v11"
)
DEFAULT_SUPPORT_CLEARANCE_M = 0.003
DEFAULT_FOOTPRINT_MARGIN_M = 0.001
DEFAULT_MAX_CANDIDATES = 96
DEFAULT_MAX_SYMBOLIC_SUPPORT_PENETRATION_M = 0.030
# Shared final-pose tolerance for Inside and spatial placement readback.
DEFAULT_PROTECTED_TRANSLATION_M = 0.005
DEFAULT_PROTECTED_ROTATION_RAD = np.deg2rad(2.0)
_SURFACE_FRACTIONS = (
    ("center", 0.50, 0.50),
    ("left_center", 0.25, 0.50),
    ("right_center", 0.75, 0.50),
    ("center_front", 0.50, 0.25),
    ("center_back", 0.50, 0.75),
    ("left_front", 0.25, 0.25),
    ("right_front", 0.75, 0.25),
    ("left_back", 0.25, 0.75),
    ("right_back", 0.75, 0.75),
)


@dataclass(frozen=True, slots=True)
class BaseAlignedBox:
    center_world: np.ndarray
    extent_local: np.ndarray
    center_local: np.ndarray


@dataclass(frozen=True, slots=True)
class OnTopCandidateBinding:
    candidate: SpatialPoseCandidate
    proposal_kind: str
    broad_footprint_fit: bool
    broad_footprint_free: bool
    thin_axis: int
    orientation_quarter_turn: int
    future_aabb_min_world: np.ndarray
    future_aabb_max_world: np.ndarray
    height_from_collision_points: bool = False

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.name,
            "proposal_kind": self.proposal_kind,
            "broad_footprint_fit": self.broad_footprint_fit,
            "broad_footprint_free": self.broad_footprint_free,
            "thin_axis": self.thin_axis,
            "orientation_quarter_turn": self.orientation_quarter_turn,
            "future_aabb_min_world_private": (
                self.future_aabb_min_world.tolist()
            ),
            "future_aabb_max_world_private": (
                self.future_aabb_max_world.tolist()
            ),
            "future_aabb_source_private": (
                "base_aligned_xy_and_collision_point_z_at_candidate_rotation"
                if self.height_from_collision_points else "base_aligned_box_at_candidate_rotation"
            ),
        }


@dataclass(frozen=True, slots=True)
class OnTopSupportLayerPlan:
    candidate_plan: SpatialCandidatePlan
    bindings: tuple[OnTopCandidateBinding, ...]

    @property
    def candidates(self) -> tuple[SpatialPoseCandidate, ...]:
        return self.candidate_plan.candidates

    def binding_for(self, candidate_name: str) -> OnTopCandidateBinding:
        matches = tuple(
            item for item in self.bindings if item.candidate.name == candidate_name
        )
        if len(matches) != 1:
            raise KeyError(f"candidate binding is not unique: {candidate_name!r}")
        return matches[0]


@dataclass(frozen=True, slots=True)
class SupportPlaneLateralBinding:
    """One screen-relative lateral pose with orientation-aware future bounds."""

    selector: str
    candidate: SpatialPoseCandidate
    future_aabb_min_world: np.ndarray
    future_aabb_max_world: np.ndarray
    orientation_rank: int

    @property
    def center_xy(self) -> np.ndarray:
        return (
            self.future_aabb_min_world[:2]
            + self.future_aabb_max_world[:2]
        ) / 2.0

    @property
    def extent_xy(self) -> np.ndarray:
        return (
            self.future_aabb_max_world[:2]
            - self.future_aabb_min_world[:2]
        )


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    for method in ("detach", "cpu"):
        callback = getattr(value, method, None)
        if callable(callback):
            value = callback()
    numpy_value = getattr(value, "numpy", None)
    if callable(numpy_value):
        value = numpy_value()
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != size or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite length-{size} vector")
    return result


def quaternion_angle_rad(first: Any, second: Any) -> float:
    left = _vector(first, 4, "first quaternion")
    right = _vector(second, 4, "second quaternion")
    left /= float(np.linalg.norm(left))
    right /= float(np.linalg.norm(right))
    return float(
        2.0
        * np.arccos(np.clip(abs(float(np.dot(left, right))), 0.0, 1.0))
    )


def on_top_binding_horizontal_distance(
    binding: OnTopCandidateBinding,
    base_xy: Any,
) -> float:
    """Return planar point-to-future-footprint distance in meters."""

    point = _vector(base_xy, 2, "base XY")
    lower = _vector(binding.future_aabb_min_world, 3, "future AABB minimum")
    upper = _vector(binding.future_aabb_max_world, 3, "future AABB maximum")
    if np.any(lower > upper):
        raise ValueError("future AABB minimum must not exceed maximum")
    offset = np.maximum(np.maximum(lower[:2] - point, point - upper[:2]), 0.0)
    return float(np.linalg.norm(offset))


def restrict_and_rank_on_top_plan(
    plan: OnTopSupportLayerPlan,
    base_xy: Any,
    maximum_distance_m: float,
) -> OnTopSupportLayerPlan:
    """Keep reachable bindings and rank them by footprint distance."""

    point = _vector(base_xy, 2, "base XY")
    maximum_distance = float(maximum_distance_m)
    if not np.isfinite(maximum_distance) or maximum_distance < 0.0:
        raise ValueError("maximum distance must be finite and non-negative")

    measured = tuple(
        (
            on_top_binding_horizontal_distance(binding, point),
            index,
            binding,
        )
        for index, binding in enumerate(plan.bindings)
    )
    reachable = tuple(
        item[2]
        for item in sorted(
            (
                item
                for item in measured
                if item[0] <= maximum_distance + 1.0e-9
            ),
            key=lambda item: (
                item[2].orientation_quarter_turn != -1,
                item[0],
                item[1],
            ),
        )
    )
    unreachable_names = {
        binding.candidate.name
        for distance, _index, binding in measured
        if distance > maximum_distance + 1.0e-9
    }
    reports = tuple(
        replace(
            report,
            accepted=False,
            reason="future_footprint_outside_local_workspace",
        )
        if report.candidate in unreachable_names
        else report
        for report in plan.candidate_plan.admission_reports
    )
    metadata = dict(plan.candidate_plan.planning_metadata_private or {})
    metadata.setdefault(
        "original_plane_candidate_count",
        metadata.get("plane_candidate_count", len(plan.bindings)),
    )
    metadata.setdefault(
        "original_bindings_private",
        list(metadata.get("bindings_private", ())),
    )
    for key in (
        "broad_free_plane_candidate_count",
        "broad_footprint_fit_plane_candidate_count",
        "broad_overhang_plane_candidate_count",
    ):
        metadata.setdefault(f"original_{key}", metadata.get(key, 0))
    metadata.update(
        {
            "local_workspace_base_xy_private": point.tolist(),
            "maximum_local_reach_m": maximum_distance,
            "candidate_horizontal_distances_m_private": [
                {
                    "candidate": binding.candidate.name,
                    "distance_m": distance,
                }
                for distance, _index, binding in measured
            ],
            "reachable_candidate_count": len(reachable),
            "plane_candidate_count": len(reachable),
            "bindings_private": [
                binding.to_private_dict() for binding in reachable
            ],
            "candidate_attempt_order_private": [
                binding.candidate.name for binding in reachable
            ],
            "broad_free_plane_candidate_count": sum(
                binding.broad_footprint_free for binding in reachable
            ),
            "broad_footprint_fit_plane_candidate_count": sum(
                binding.broad_footprint_fit for binding in reachable
            ),
            "broad_overhang_plane_candidate_count": sum(
                not binding.broad_footprint_fit for binding in reachable
            ),
        }
    )
    return OnTopSupportLayerPlan(
        candidate_plan=SpatialCandidatePlan(
            candidates=tuple(binding.candidate for binding in reachable),
            admission_reports=reports,
            planning_error=(
                plan.candidate_plan.planning_error
                if reachable or plan.candidate_plan.planning_error is not None
                else "no OnTop candidates within local workspace"
            ),
            planning_metadata_private=metadata,
        ),
        bindings=reachable,
    )


def quaternion_to_matrix_xyzw(value: Any) -> np.ndarray:
    x, y, z, w = _vector(value, 4, "quaternion")
    norm = float(np.linalg.norm((x, y, z, w)))
    if norm <= 1.0e-12:
        raise ValueError("quaternion has zero norm")
    x, y, z, w = (np.asarray((x, y, z, w)) / norm).tolist()
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrix must be finite 3x3")
    # Project tiny numerical drift back onto SO(3) before conversion.
    u, _, vh = np.linalg.svd(matrix)
    matrix = u @ vh
    if np.linalg.det(matrix) < 0.0:
        u[:, -1] *= -1.0
        matrix = u @ vh
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            )
        )
    else:
        diagonal = int(np.argmax(np.diag(matrix)))
        if diagonal == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(((0.25 * scale), (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale))
        elif diagonal == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(((matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale))
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(((matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale))
    quat /= float(np.linalg.norm(quat))
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def read_base_aligned_box(obj: Any) -> BaseAlignedBox:
    callback = getattr(obj, "get_base_aligned_bbox", None)
    if not callable(callback):
        raise ValueError(
            "thin-axis OnTop authoring requires a base-aligned bounding box"
        )
    center, _orientation, extent, center_local = callback()
    result = BaseAlignedBox(
        center_world=_vector(center, 3, "base-aligned bbox center"),
        extent_local=_vector(extent, 3, "base-aligned bbox extent"),
        center_local=_vector(center_local, 3, "base-aligned bbox local center"),
    )
    if np.any(result.extent_local <= 0.0):
        raise ValueError("base-aligned bbox extent must be positive")
    return result


def thin_axis_orientation_matrices(
    extent_local: Any,
    support_normal_world: Any,
) -> tuple[np.ndarray, ...]:
    """Return four deterministic rotations with the thinnest axis normal."""

    extent = _vector(extent_local, 3, "bbox extent")
    if np.any(extent <= 0.0):
        raise ValueError("bbox extent must be positive")
    normal = _vector(support_normal_world, 3, "support normal")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-12:
        raise ValueError("support normal has zero norm")
    normal /= normal_norm
    reference = np.asarray((1.0, 0.0, 0.0))
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.asarray((0.0, 1.0, 0.0))
    tangent0 = reference - float(np.dot(reference, normal)) * normal
    tangent0 /= float(np.linalg.norm(tangent0))
    tangent1 = np.cross(normal, tangent0)
    thin_axis = int(np.argmin(extent))
    results = []
    for quarter_turn in range(4):
        angle = quarter_turn * np.pi / 2.0
        tangent = np.cos(angle) * tangent0 + np.sin(angle) * tangent1
        columns: list[np.ndarray | None] = [None, None, None]
        columns[thin_axis] = normal
        if thin_axis == 0:
            columns[1] = tangent
            columns[2] = np.cross(columns[0], columns[1])
        elif thin_axis == 1:
            columns[2] = tangent
            columns[0] = np.cross(columns[1], columns[2])
        else:
            columns[0] = tangent
            columns[1] = np.cross(columns[2], columns[0])
        results.append(np.column_stack(columns))
    return tuple(results)


def on_top_orientation_matrices(
    extent_local: Any,
    preferred_orientation_xyzw: Any | None = None,
) -> tuple[np.ndarray, ...]:
    """Shared publication/execution orientation order; no additional rotations."""
    rotations = thin_axis_orientation_matrices(extent_local, (0.0, 0.0, 1.0))
    if preferred_orientation_xyzw is None:
        return rotations
    preferred = quaternion_to_matrix_xyzw(preferred_orientation_xyzw)
    return (preferred, *(item for item in rotations if not np.allclose(preferred, item)))


def plan_support_plane_lateral_bindings(
    held: Any,
    target: Any,
    *,
    observer_xy: Any,
    relation: str = "NextTo",
    preferred_orientation_xyzw: Any | None = None,
) -> tuple[SupportPlaneLateralBinding, ...]:
    """Author eight physical support-plane positions and stable orientations.

    The positions are the eight camera-relative compass points around target.
    Every fallback orientation keeps the same XY center, so a selected visual
    point never turns into a different placement position.  Z remains only a
    seed until the existing five-ray local-support transaction rebases it.
    """

    if relation != "NextTo":
        raise ValueError("support-plane lateral bindings require NextTo")
    held_box = read_base_aligned_box(held)
    target_min, target_max = object_aabb(target)
    target_center = (target_min + target_max) / 2.0
    target_extent = target_max - target_min

    if preferred_orientation_xyzw is None:
        _, preferred_orientation_xyzw = object_pose(held)
    preferred_rotation = quaternion_to_matrix_xyzw(
        preferred_orientation_xyzw
    )
    rotations: list[tuple[str, np.ndarray]] = [
        ("preferred", preferred_rotation)
    ]
    for quarter_turn, rotation in enumerate(
        thin_axis_orientation_matrices(
            held_box.extent_local,
            (0.0, 0.0, 1.0),
        )
    ):
        if any(np.allclose(rotation, existing) for _, existing in rotations):
            continue
        rotations.append((f"thin_q{quarter_turn}", rotation))

    observer = _vector(observer_xy, 2, "observer XY")
    near = observer - target_center[:2]
    near_norm = float(np.linalg.norm(near))
    if near_norm <= 1.0e-9:
        raise ValueError("observer XY must differ from target center XY")
    near /= near_norm
    right = np.asarray((-near[1], near[0]), dtype=np.float64)
    raw_directions = (
        -right,
        -right - near,
        -near,
        right - near,
        right,
        right + near,
        near,
        -right + near,
    )
    directions = tuple(
        (selector, direction / float(np.linalg.norm(direction)))
        for selector, direction in zip(
            SIMPLE_NEXT_TO_SCREEN_SELECTORS,
            raw_directions,
            strict=True,
        )
    )

    preferred_half = (
        np.abs(preferred_rotation) @ held_box.extent_local
    ) / 2.0
    preferred_extent = preferred_half * 2.0
    gap = next_to_gap_for_extents(preferred_extent, target_extent)
    target_half_xy = target_extent[:2] / 2.0
    fixed_centers_xy = {}
    combined_half = target_half_xy + preferred_half[:2]
    for selector, direction in directions:
        absolute = np.abs(direction)
        active = absolute > 1.0e-12
        # Solve actual AABB separation along the selected ray, rather than
        # confusing projected box radii with Euclidean nearest-box distance.
        lower = float(np.min(combined_half[active] / absolute[active]))
        upper = float(np.min((combined_half[active] + gap) / absolute[active]))
        for _ in range(48):
            radius = (lower + upper) / 2.0
            separation = np.maximum(radius * absolute - combined_half, 0.0)
            if np.linalg.norm(separation) > gap:
                upper = radius
            else:
                lower = radius
        fixed_centers_xy[selector] = target_center[:2] + direction * ((lower + upper) / 2.0)

    bindings = []
    for orientation_rank, (orientation_kind, rotation) in enumerate(rotations):
        world_half = (np.abs(rotation) @ held_box.extent_local) / 2.0
        center_offset = rotation @ held_box.center_local
        for selector, _direction in directions:
            desired_center = np.asarray(
                (
                    *fixed_centers_xy[selector],
                    float(target_min[2])
                    + float(world_half[2])
                    + DEFAULT_SUPPORT_CLEARANCE_M,
                ),
                dtype=np.float64,
            )
            root = desired_center - center_offset
            lower, upper = _future_aabb(
                held_box,
                root,
                rotation,
            )
            bindings.append(
                SupportPlaneLateralBinding(
                    selector=selector,
                    candidate=SpatialPoseCandidate(
                        name=(
                            f"next_to_{orientation_kind}_{selector}"
                        ),
                        position=root,
                        orientation_xyzw=matrix_to_quaternion_xyzw(rotation),
                    ),
                    future_aabb_min_world=lower,
                    future_aabb_max_world=upper,
                    orientation_rank=orientation_rank,
                )
            )
    return tuple(bindings)


def fitting_inside_orientation_matrices(
    extent_local: Any,
    fillable_extent: Any,
    *,
    preferred_orientation_xyzw: Any | None = None,
    clearance_m: float = 0.005,
) -> tuple[np.ndarray, ...]:
    """Return fitting orientations, preferring the pre-pick pose."""

    extent = _vector(extent_local, 3, "bbox extent")
    fillable = _vector(fillable_extent, 3, "fillable extent")
    if np.any(extent <= 0.0) or np.any(fillable <= 0.0):
        raise ValueError("bbox and fillable extents must be positive")
    clearance = float(clearance_m)
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance must be finite and non-negative")
    rotations = thin_axis_orientation_matrices(extent, (0.0, 0.0, 1.0))
    preferred_rotation_count = 0
    if preferred_orientation_xyzw is not None:
        preferred = quaternion_to_matrix_xyzw(preferred_orientation_xyzw)
        rotations = (
            preferred,
            *(item for item in rotations if not np.allclose(preferred, item)),
        )
        preferred_rotation_count = 1
    ranked = []
    for index, rotation in enumerate(rotations):
        oriented_extent = np.abs(rotation) @ extent
        minimum_clearance = float(np.min((fillable - oriented_extent) / 2.0))
        if minimum_clearance + 1.0e-12 >= clearance:
            ranked.append(
                (
                    not (preferred_rotation_count and index == 0),
                    -minimum_clearance,
                    index,
                    rotation,
                )
            )
    ranked.sort(key=lambda item: item[:3])
    return tuple(
        rotation
        for _preferred, _clearance, _index, rotation in ranked
    )


def _future_aabb(
    box: BaseAlignedBox,
    root_position: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = root_position + rotation @ box.center_local
    half = np.abs(rotation) @ (box.extent_local / 2.0)
    return center - half, center + half


def _overlaps_any(
    lower: np.ndarray,
    upper: np.ndarray,
    blockers: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    margin_m: float,
) -> bool:
    for other_lower, other_upper in blockers:
        overlap = np.minimum(upper, other_upper) - np.maximum(lower, other_lower)
        if bool(np.all(overlap > margin_m)):
            return True
    return False


def _plane_candidates(
    *,
    held_box: BaseAlignedBox,
    destination: Any,
    rotations: Sequence[np.ndarray],
    preferred_rotation_count: int = 0,
    blocker_aabbs: Sequence[tuple[np.ndarray, np.ndarray]],
    region_center_xy: Any | None = None,
    region_extent_xy: Any | None = None,
    surface_z_m: float | None = None,
    include_support_center: bool = False,
    collision_points_in_root: np.ndarray | None = None,
) -> tuple[OnTopCandidateBinding, ...]:
    support_min, support_max = object_aabb(destination)
    support_z = float(support_max[2])
    if surface_z_m is not None:
        support_z = float(surface_z_m)
        if not np.isfinite(support_z):
            raise ValueError("support surface height must be finite")
        if not (
            support_min[2] - DEFAULT_FOOTPRINT_MARGIN_M
            <= support_z
            <= support_max[2] + DEFAULT_FOOTPRINT_MARGIN_M
        ):
            raise ValueError("support surface height lies outside the support AABB")
    center_min = support_min[:2].copy()
    center_max = support_max[:2].copy()
    if (region_center_xy is None) != (region_extent_xy is None):
        raise ValueError("region center and extent must be supplied together")
    if region_center_xy is not None:
        center = _vector(region_center_xy, 2, "region center")
        extent = _vector(region_extent_xy, 2, "region extent")
        if np.any(extent <= 0.0):
            raise ValueError("region extent must be positive")
        center_min = np.maximum(center_min, center - extent / 2.0)
        center_max = np.minimum(center_max, center + extent / 2.0)
        if np.any(center_min >= center_max):
            return ()
    bindings = []
    for orientation_index, rotation in enumerate(rotations):
        preserves_original_orientation = orientation_index < preferred_rotation_count
        world_half = np.abs(rotation) @ (held_box.extent_local / 2.0)
        # Irregular objects need not touch the lowest corner of their rotated box.
        height_offsets = None
        if collision_points_in_root is not None:
            heights = collision_points_in_root @ rotation[2]
            height_offsets = np.array([heights.min(), heights.max()])
        available_min = np.maximum(
            center_min,
            support_min[:2] + world_half[:2] + DEFAULT_FOOTPRINT_MARGIN_M,
        )
        available_max = np.minimum(
            center_max,
            support_max[:2] - world_half[:2] - DEFAULT_FOOTPRINT_MARGIN_M,
        )
        footprint_fit = bool(np.all(available_min <= available_max))
        if not footprint_fit:
            # A whole-object AABB is not the benchmark relation authority.
            # Keep a bounded set of overhang proposals rooted over the live
            # support footprint; every one must still pass the live narrow
            # collision query and a complete official OnTop+Touching window.
            # This is essential for thin/irregular pairs whose conservative
            # AABBs say "too large" even though their real contact geometry is
            # a valid stable placement.
            available_min = np.maximum(
                center_min,
                support_min[:2] + DEFAULT_FOOTPRINT_MARGIN_M,
            )
            available_max = np.minimum(
                center_max,
                support_max[:2] - DEFAULT_FOOTPRINT_MARGIN_M,
            )
            for axis in range(2):
                if available_min[axis] > available_max[axis]:
                    midpoint = (center_min[axis] + center_max[axis]) / 2.0
                    available_min[axis] = midpoint
                    available_max[axis] = midpoint
        centers = [
            (label, available_min + np.asarray((fx, fy)) * (available_max - available_min))
            for label, fx, fy in _SURFACE_FRACTIONS
        ]
        support_center = (support_min[:2] + support_max[:2]) / 2.0
        if (
            include_support_center
            and np.all((available_min <= support_center) & (support_center <= available_max))
            and not any(np.allclose(support_center, xy) for _, xy in centers)
        ):
            # A grid boundary can contain the support centre without any of
            # the ordinary interior samples landing there. Never leave the
            # selected cell to recover it.
            centers.insert(0, ("support_center", support_center))
        for label, center_xy in centers:
            desired_center = np.asarray(
                (
                    center_xy[0],
                    center_xy[1],
                    support_z + world_half[2] + DEFAULT_SUPPORT_CLEARANCE_M,
                )
            )
            root = desired_center - rotation @ held_box.center_local
            if height_offsets is not None:
                root[2] = support_z + DEFAULT_SUPPORT_CLEARANCE_M - height_offsets[0]
            lower, upper = _future_aabb(held_box, root, rotation)
            if height_offsets is not None:
                lower[2], upper[2] = root[2] + height_offsets
            free = footprint_fit and not _overlaps_any(
                lower,
                upper,
                blocker_aabbs,
                margin_m=DEFAULT_FOOTPRINT_MARGIN_M,
            )
            bindings.append(
                OnTopCandidateBinding(
                    candidate=SpatialPoseCandidate(
                        name=(
                            f"on_top_original_{label}"
                            if preserves_original_orientation
                            else f"on_top_thin_q{orientation_index - preferred_rotation_count}_{label}"
                        ),
                        position=root,
                        orientation_xyzw=matrix_to_quaternion_xyzw(rotation),
                    ),
                    proposal_kind=(
                        "original_orientation_plane"
                        if preserves_original_orientation
                        else "original_destination_plane"
                    ),
                    broad_footprint_fit=footprint_fit,
                    broad_footprint_free=free,
                    thin_axis=int(np.argmin(held_box.extent_local)),
                    orientation_quarter_turn=(
                        -1
                        if preserves_original_orientation
                        else orientation_index - preferred_rotation_count
                    ),
                    height_from_collision_points=height_offsets is not None,
                    future_aabb_min_world=lower,
                    future_aabb_max_world=upper,
                )
            )
    # Preserve the pre-pick pose first. AABB remains ranking evidence only
    # within each orientation; every pose still needs the narrow query.
    return tuple(
        sorted(
            bindings,
            key=lambda item: (
                item.orientation_quarter_turn != -1,
                not item.broad_footprint_free,
                not item.candidate.name.endswith("_support_center"),
                item.candidate.name,
            ),
        )
    )


def plan_on_top_support_layer_candidates(
    held: Any,
    original_destination: Any,
    *,
    scene_objects: Sequence[Any] = (),
    broad_diagnostic_objects: Sequence[Any] | None = None,
    preferred_orientation_xyzw: Any | None = None,
    region_center_xy: Any | None = None,
    region_extent_xy: Any | None = None,
    surface_z_m: float | None = None,
    allow_overhang: bool = True,
    include_support_center: bool = False,
    maximum_candidates: int = DEFAULT_MAX_CANDIDATES,
    collision_points_in_root: np.ndarray | None = None,
) -> OnTopSupportLayerPlan:
    """Try the pre-pick orientation, then stable thin-axis fallbacks."""

    if maximum_candidates <= 0:
        raise ValueError("maximum_candidates must be positive")
    held_box = read_base_aligned_box(held)
    rotations = on_top_orientation_matrices(held_box.extent_local, preferred_orientation_xyzw)
    preferred_rotation_count = int(preferred_orientation_xyzw is not None)
    excluded = {id(held), id(original_destination)}
    blocker_aabbs = []
    unreadable = 0
    diagnostic_objects = (
        scene_objects
        if broad_diagnostic_objects is None
        else broad_diagnostic_objects
    )
    for other in diagnostic_objects:
        if other is None or id(other) in excluded:
            continue
        try:
            blocker_aabbs.append(object_aabb(other))
        except Exception:
            unreadable += 1
    plane = _plane_candidates(
        held_box=held_box,
        destination=original_destination,
        rotations=rotations,
        preferred_rotation_count=preferred_rotation_count,
        blocker_aabbs=tuple(blocker_aabbs),
        region_center_xy=region_center_xy,
        region_extent_xy=region_extent_xy,
        surface_z_m=surface_z_m,
        include_support_center=include_support_center,
        collision_points_in_root=collision_points_in_root,
    )
    bindings = tuple(
        item for item in plane if allow_overhang or item.broad_footprint_fit
    )[:maximum_candidates]
    reports = tuple(
        CandidateAdmissionReport(
            candidate=item.candidate.name,
            accepted=True,
            reason=(
                None
                if item.broad_footprint_fit and item.broad_footprint_free
                else (
                    "broad support footprint overhang diagnostic; live narrow "
                    "query and official readback required"
                    if not item.broad_footprint_fit
                    else "broad occupancy diagnostic; live narrow query required"
                )
            ),
            blockers=(),
            bottom_to_support_gap_m=None,
        )
        for item in bindings
    )
    metadata: Mapping[str, Any] = {
        "planner": ON_TOP_SUPPORT_LAYER_VERSION,
        "thin_axis": int(np.argmin(held_box.extent_local)),
        "thin_axis_orientation_first": not preferred_rotation_count,
        "original_orientation_first": bool(preferred_rotation_count),
        "region_center_xy": (
            None
            if region_center_xy is None
            else _vector(region_center_xy, 2, "region center").tolist()
        ),
        "region_extent_xy": (
            None
            if region_extent_xy is None
            else _vector(region_extent_xy, 2, "region extent").tolist()
        ),
        "surface_z_m": (
            None if surface_z_m is None else float(surface_z_m)
        ),
        "plane_candidate_count": min(len(plane), len(bindings)),
        "broad_free_plane_candidate_count": sum(
            item.broad_footprint_free for item in plane
        ),
        "broad_footprint_fit_plane_candidate_count": sum(
            item.broad_footprint_fit for item in plane
        ),
        "broad_overhang_plane_candidate_count": sum(
            not item.broad_footprint_fit for item in plane
        ),
        "unreadable_geometry_count": unreadable,
        "broad_diagnostic_object_count": len(diagnostic_objects),
        "aabb_authoritative": not allow_overhang,
        "overhang_candidates_allowed": allow_overhang,
        "support_center_candidate_enabled": include_support_center,
        "narrow_collision_and_canonical_readback_required": True,
        "bindings_private": [item.to_private_dict() for item in bindings],
    }
    return OnTopSupportLayerPlan(
        candidate_plan=SpatialCandidatePlan(
            candidates=tuple(item.candidate for item in bindings),
            admission_reports=reports,
            planning_error=(
                None
                if bindings
                else (
                    "Selected region has no stable support candidate"
                    if plane and not allow_overhang
                    else "no bounded OnTop candidates"
                )
            ),
            planning_metadata_private=metadata,
        ),
        bindings=bindings,
    )


def summarize_original_destination_on_top(
    samples: Sequence[SpatialStabilitySample],
    *,
    required_samples: int = DEFAULT_STABILITY_WINDOW,
    max_support_penetration_m: float = (
        DEFAULT_MAX_SYMBOLIC_SUPPORT_PENETRATION_M
    ),
) -> dict[str, Any]:
    """Reject unstable poses and only gross support penetration."""

    report = summarize_spatial_stability(
        samples,
        required_samples=required_samples,
        # Keep separation permissive for layered / irregular supports.  This
        # is a symbolic gate: only obvious penetration below the support's
        # live AABB top is rejected.
        max_support_gap_m=float("inf"),
        max_support_penetration_m=max_support_penetration_m,
        max_linear_speed_m_s=DEFAULT_MAX_LINEAR_SPEED_M_S,
        max_angular_speed_rad_s=DEFAULT_MAX_ANGULAR_SPEED_RAD_S,
    )
    report.update(
        {
            "verification_policy": (
                "behavior2026.original-destination-on-top-strict/v2"
            ),
            "official_on_top_original_destination_required": True,
            "official_touching_original_destination_required": True,
            "complete_settle_window_required": True,
            "finite_pose_and_velocity_required": True,
            "aabb_gap_diagnostic_only": False,
            "aabb_separation_diagnostic_only": True,
            "gross_support_penetration_rejected": True,
            "max_symbolic_support_penetration_m": (
                max_support_penetration_m
            ),
        }
    )
    return report


__all__ = [
    "on_top_orientation_matrices",
    "BaseAlignedBox",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_PROTECTED_ROTATION_RAD",
    "ON_TOP_SUPPORT_LAYER_VERSION",
    "OnTopCandidateBinding",
    "OnTopSupportLayerPlan",
    "SupportPlaneLateralBinding",
    "fitting_inside_orientation_matrices",
    "matrix_to_quaternion_xyzw",
    "on_top_binding_horizontal_distance",
    "plan_on_top_support_layer_candidates",
    "plan_support_plane_lateral_bindings",
    "quaternion_to_matrix_xyzw",
    "read_base_aligned_box",
    "restrict_and_rank_on_top_plan",
    "summarize_original_destination_on_top",
    "thin_axis_orientation_matrices",
]
