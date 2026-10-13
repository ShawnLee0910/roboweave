"""Goal-neutral geometry helpers for restricted Symbolic spatial placement.

The public action selects only a current-frame marker.  Side / region choice
is executor-private and deterministic.  Candidate geometry has no OmniGibson
dependency; only the live OnTop readback fallback lazily imports its ray query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


def placement_predicate_mode(
    *, symbolic_zero_physics: bool, physical_consequences_enabled: bool,
) -> str:
    """Choose readback from execution flags, never the action profile name."""

    return (
        "action20_symbolic"
        if symbolic_zero_physics and not physical_consequences_enabled
        else "official"
    )


SPATIAL_SYMBOLIC_ACTION_VERSION = (
    "behavior2026.marker-symbolic-spatial-default-region/v13-geometric-nextto"
)
SPATIAL_ACTION_TO_PREDICATE = {
    "place_on_floor": "OnTop",
    "place_on_top": "OnTop",
    "place_next_to": "NextTo",
    "place_under": "Under",
    "place_touching": "Touching",
}
DEFAULT_LATERAL_ORDER = (
    ("positive_x", 0, 1.0),
    ("negative_x", 0, -1.0),
    ("positive_y", 1, 1.0),
    ("negative_y", 1, -1.0),
)
DEFAULT_LATERAL_FACE_OFFSETS = (
    ("face_near_lower", -0.46),
    ("face_lower", -0.30),
    ("face_mid_lower", -0.15),
    ("face_mid_upper", 0.15),
    ("face_upper", 0.30),
    ("face_near_upper", 0.46),
)
SIMPLE_TOUCHING_SCREEN_SELECTORS = (
    "screen_left",
    "screen_right",
    "screen_near",
    "screen_far",
)
SIMPLE_NEXT_TO_SCREEN_SELECTORS = (
    "screen_left",
    "screen_far_left",
    "screen_far",
    "screen_far_right",
    "screen_right",
    "screen_near_right",
    "screen_near",
    "screen_near_left",
)
DEFAULT_NEXT_TO_GAP_M = 0.03
OFFICIAL_NEXT_TO_DISTANCE_RATIO = 1.0 / 6.0
DEFAULT_NEXT_TO_THRESHOLD_FRACTION = 0.75
DEFAULT_TOUCHING_APPROACH_M = 0.0005
DEFAULT_SUPPORT_CLEARANCE_M = 0.003
DEFAULT_SUPPORT_INSET_M = 0.005
DEFAULT_BLOCKER_MARGIN_M = 0.005
DEFAULT_PLACEMENT_BLOCKER_MARGIN_M = 0.001
DEFAULT_COLLISION_QUERY_Z_INSET_M = 0.0005
DEFAULT_SOLID_OVERLAP_TOLERANCE_M = 0.0005
DEFAULT_LOCAL_SUPPORT_RAY_ABOVE_M = 0.05
DEFAULT_LOCAL_SUPPORT_RAY_BELOW_M = 0.15
DEFAULT_LOCAL_SUPPORT_NORMAL_Z_MIN = 0.95
DEFAULT_LOCAL_SUPPORT_COPLANAR_TOLERANCE_M = 0.005
DEFAULT_LOCAL_SUPPORT_MIN_CONSISTENT_HITS = 3
LOCAL_SUPPORT_RAY_SAMPLE_NAMES = (
    "footprint_center",
    "footprint_negative_x_negative_y",
    "footprint_negative_x_positive_y",
    "footprint_positive_x_negative_y",
    "footprint_positive_x_positive_y",
)
# These are executor roles, not object names, categories, or task identities.
# The live adapter should assign them from identity (held / relation target /
# robot) and structural collision evidence.  Traversable terrain or another
# geometrically valid local support layer receives a non-forbidden role.
DEFAULT_FORBIDDEN_LOCAL_SUPPORT_ENTITY_ROLES = frozenset(
    {
        "held_object",
        "relation_target",
        "robot",
        "structural_obstacle",
        "non_support_obstacle",
    }
)
DEFAULT_SETTLE_STEPS = 30
DEFAULT_STABILITY_WINDOW = 10
DEFAULT_SYMBOLIC_POSTCONDITION_WINDOW = 3
DEFAULT_SYMBOLIC_CANDIDATE_TRIAL_LIMIT = 8
DEFAULT_MAX_SUPPORT_GAP_M = 0.015
DEFAULT_MAX_SUPPORT_PENETRATION_M = 0.010
DEFAULT_MAX_LINEAR_SPEED_M_S = 0.02
DEFAULT_MAX_ANGULAR_SPEED_RAD_S = 0.20
# Deliberately broad gross-failure gates for lateral relations.  These are
# not precision placement tolerances: ordinary bounce and settling remain
# legal, while falling out of the room or remaining in obvious free flight
# cannot be accepted merely because a horizontal relation ray stays true.
DEFAULT_SUPPORTED_LATERAL_GROSS_FALL_M = 0.25
DEFAULT_SUPPORTED_LATERAL_GROSS_PLANAR_ESCAPE_M = 1.0
DEFAULT_SUPPORTED_LATERAL_MAX_LINEAR_SPEED_M_S = 5.0
DEFAULT_SUPPORTED_LATERAL_MAX_ANGULAR_SPEED_RAD_S = 12.0
DEFAULT_MIN_UNDER_HEADROOM_MARGIN_M = 0.015
DEFAULT_PLACEMENT_FRACTIONS = (
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
class SpatialPoseCandidate:
    name: str
    position: np.ndarray
    orientation_xyzw: np.ndarray


@dataclass(frozen=True, slots=True)
class DownwardSupportRayRequest:
    """One of five deterministic footprint probes for local support height."""

    sample_name: str
    start_xyz: np.ndarray
    end_xyz: np.ndarray

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "sample_name": self.sample_name,
            "start_xyz": _private_jsonable(self.start_xyz),
            "end_xyz": _private_jsonable(self.end_xyz),
        }


@dataclass(frozen=True, slots=True)
class DownwardSupportRayEvidence:
    """Caller-owned live result for one downward footprint probe.

    ``entity_role`` is a role assigned by the executor adapter.  The geometry
    kernel never branches on an object name, category, model, task, or scene.
    ``hit_entity_key`` is retained only as private audit evidence.
    """

    sample_name: str
    start_xyz: Any
    end_xyz: Any
    hit_position_xyz: Any | None
    hit_normal_xyz: Any | None
    hit_entity_key: str | None
    entity_role: str | None
    error: str | None = None

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "sample_name": self.sample_name,
            "start_xyz": _private_jsonable(self.start_xyz),
            "end_xyz": _private_jsonable(self.end_xyz),
            "hit_position_xyz": _private_jsonable(self.hit_position_xyz),
            "hit_normal_xyz": _private_jsonable(self.hit_normal_xyz),
            "hit_entity_key_private": self.hit_entity_key,
            "entity_role_private": self.entity_role,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class LocalSupportRebaseReport:
    """Fail-closed result of rebasing a candidate onto a measured plane.

    This report admits only the *candidate geometry*.  It does not claim the
    official ``NextTo`` (or any other) relation.  The caller must feed an
    admitted candidate through a live collision-shape query, execute it,
    settle, and then use the canonical official predicate readback.  An
    object-level AABB overlap may still be retained as broad-phase telemetry,
    but it is not sufficiently precise to reject a candidate.
    """

    accepted: bool
    reason: str | None
    original_candidate: SpatialPoseCandidate
    rebased_candidate: SpatialPoseCandidate | None
    original_bottom_z_m: float | None
    median_hit_z_m: float | None
    rebased_bottom_z_m: float | None
    coherent_hit_count: int
    required_coherent_hit_count: int
    ray_evidence: tuple[DownwardSupportRayEvidence, ...]
    forbidden_roles_seen: tuple[str, ...] = ()

    def to_private_dict(self) -> dict[str, Any]:
        rebased = self.rebased_candidate
        return {
            "accepted_for_blocker_check": self.accepted,
            "official_relation_declared": False,
            "rebase_reason": self.reason,
            "original_candidate": self.original_candidate.name,
            "original_candidate_position": _private_jsonable(
                self.original_candidate.position
            ),
            "rebased_candidate_position": (
                None if rebased is None else _private_jsonable(rebased.position)
            ),
            "original_bottom_z_m": self.original_bottom_z_m,
            "median_hit_z_m": self.median_hit_z_m,
            "rebased_bottom_z_m": self.rebased_bottom_z_m,
            "coherent_hit_count": self.coherent_hit_count,
            "required_coherent_hit_count": self.required_coherent_hit_count,
            "forbidden_roles_seen_private": list(self.forbidden_roles_seen),
            "support_rays_private": [
                item.to_private_dict() for item in self.ray_evidence
            ],
        }


@dataclass(frozen=True, slots=True)
class CandidateCollisionQueryBox:
    """World-axis query box enclosing one future held-object candidate."""

    center_xyz: np.ndarray
    half_extent_xyz: np.ndarray
    rotation_xyzw: np.ndarray
    xy_margin_m: float
    xy_inset_m: float
    z_inset_m: float
    geometry_source: str

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "center_xyz": _private_jsonable(self.center_xyz),
            "half_extent_xyz": _private_jsonable(self.half_extent_xyz),
            "rotation_xyzw": _private_jsonable(self.rotation_xyzw),
            "xy_margin_m": self.xy_margin_m,
            "xy_inset_m": self.xy_inset_m,
            "z_inset_m": self.z_inset_m,
            "geometry_source": self.geometry_source,
        }


@dataclass(frozen=True, slots=True)
class CollisionOverlapHitEvidence:
    """One raw PhysX overlap hit, kept independent of task/object labels."""

    rigid_body_path: str | None
    collision_path: str | None

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "rigid_body_path_private": self.rigid_body_path,
            "collision_path_private": self.collision_path,
        }


@dataclass(frozen=True, slots=True)
class CollisionOverlapAdmissionReport:
    """Fail-closed collision-shape admission for a spatial candidate."""

    accepted: bool
    reason: str | None
    query_box: CandidateCollisionQueryBox | None
    raw_hits: tuple[CollisionOverlapHitEvidence, ...]
    excluded_hits: tuple[CollisionOverlapHitEvidence, ...]
    blocking_hits: tuple[CollisionOverlapHitEvidence, ...]
    query_error: str | None = None

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "accepted_for_candidate_execution": self.accepted,
            "official_relation_declared": False,
            "collision_query_reason": self.reason,
            "collision_query_box_private": (
                None if self.query_box is None else self.query_box.to_private_dict()
            ),
            "raw_overlap_hits_private": [
                item.to_private_dict() for item in self.raw_hits
            ],
            "excluded_overlap_hits_private": [
                item.to_private_dict() for item in self.excluded_hits
            ],
            "blocking_overlap_hits_private": [
                item.to_private_dict() for item in self.blocking_hits
            ],
            "query_error": self.query_error,
        }


@dataclass(frozen=True, slots=True)
class VerticalPlacementEvidence:
    """Live ray / adjacency evidence for one proposed Under placement.

    A caller-owned live probe should cast several rays over the proposed held
    footprint.  The first upward hit must be the Under target and the first
    downward hit must be the selected support.  ``vertical_adjacency`` may be
    ``None`` for a pre-placement ray-only probe; canonical Under is still
    required by the post-settle stability window.
    """

    target_first_above: bool
    support_first_below: bool
    clear_underside: bool
    vertical_clearance_m: float | None
    vertical_adjacency: bool | None = None
    upward_hit_names: tuple[str, ...] = ()
    downward_hit_names: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateAdmissionReport:
    candidate: str
    accepted: bool
    reason: str | None
    blockers: tuple[str, ...] = ()
    bottom_to_support_gap_m: float | None = None
    vertical_evidence: VerticalPlacementEvidence | None = None

    def to_private_dict(self) -> dict[str, Any]:
        evidence = self.vertical_evidence
        return {
            "candidate": self.candidate,
            "accepted_for_settle": self.accepted,
            "admission_reason": self.reason,
            "blocked_by_private": list(self.blockers),
            "bottom_to_support_gap_m": self.bottom_to_support_gap_m,
            "vertical_evidence_private": (
                None
                if evidence is None
                else {
                    "target_first_above": evidence.target_first_above,
                    "support_first_below": evidence.support_first_below,
                    "clear_underside": evidence.clear_underside,
                    "vertical_clearance_m": evidence.vertical_clearance_m,
                    "vertical_adjacency": evidence.vertical_adjacency,
                    "upward_hit_names": list(evidence.upward_hit_names),
                    "downward_hit_names": list(evidence.downward_hit_names),
                    "error": evidence.error,
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class SpatialCandidatePlan:
    """Bounded candidate list plus reports for every admitted or rejected pose."""

    candidates: tuple[SpatialPoseCandidate, ...]
    admission_reports: tuple[CandidateAdmissionReport, ...]
    planning_error: str | None = None
    planning_metadata_private: Mapping[str, Any] | None = None

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": len(self.admission_reports),
            "admitted_candidate_count": len(self.candidates),
            "planning_error": self.planning_error,
            "planning_metadata_private": _private_jsonable(
                self.planning_metadata_private
            ),
            "candidate_admission_private": [
                report.to_private_dict() for report in self.admission_reports
            ],
        }


@dataclass(frozen=True, slots=True)
class UnderSupportPlan:
    support: Any
    candidate_plan: SpatialCandidatePlan

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "support_private": str(getattr(self.support, "name", "")) or None,
            **self.candidate_plan.to_private_dict(),
        }


@dataclass(frozen=True, slots=True)
class SpatialStabilitySample:
    relation: bool
    support_on_top: bool
    support_touching: bool | None
    bottom_to_support_gap_m: float
    linear_speed_m_s: float
    angular_speed_rad_s: float
    tilt_rad: float
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    bottom_z_m: float | None = None

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "support_on_top": self.support_on_top,
            "support_touching": self.support_touching,
            "bottom_to_support_gap_m": self.bottom_to_support_gap_m,
            "linear_speed_m_s": self.linear_speed_m_s,
            "angular_speed_rad_s": self.angular_speed_rad_s,
            "tilt_rad": self.tilt_rad,
            "position": list(self.position),
            "orientation_xyzw": list(self.orientation_xyzw),
            "bottom_z_m": self.bottom_z_m,
        }


VerticalProbe = Callable[
    [SpatialPoseCandidate, Any, Any, Any],
    VerticalPlacementEvidence,
]


def spatial_exact_step_budget(operation: str) -> int:
    """Return a bounded budget large enough for every default candidate.

    Under has one official setter attempt followed by at most nine fallback
    regions.  Each fallback region uses 30 settle steps and a 10-sample
    canonical-state window, so the former shared 256-step cap truncated a
    valid deterministic search before it could finish.
    """

    if operation in {"place_on_floor", "place_on_top"}:
        # Thin-axis authoring is bounded to 96 poses.  Every applied pose owns
        # 30 settle steps and a 10-sample canonical readback window; leave a
        # small allowance for release / finalization.  This is an execution
        # ceiling, not a promise that all poses will be applied.
        return 4096
    if operation == "place_under":
        return 512
    if operation in {"place_next_to", "place_touching"}:
        # Four face-centre candidates plus six deterministic offsets along
        # every face.  This covers furniture whose only horizontal adjacency
        # rays hit near legs / face edges instead of the AABB face centre.
        # Each candidate owns 30 settle steps and a 10-sample readback window,
        # plus a small release / finalization allowance.
        return 1280
    raise ValueError(f"unsupported spatial operation: {operation}")


def _vector(value: Any, size: int) -> np.ndarray:
    for method in ("detach", "cpu"):
        callback = getattr(value, method, None)
        if callable(callback):
            value = callback()
    callback = getattr(value, "numpy", None)
    if callable(callback):
        value = callback()
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"expected finite vector with {size} entries")
    return result


def object_pose(obj: Any) -> tuple[np.ndarray, np.ndarray]:
    getter = getattr(obj, "get_position_orientation", None)
    if not callable(getter):
        raise RuntimeError("object pose is unavailable")
    try:
        position, orientation = getter()
    except TypeError:
        position, orientation = getter(frame="scene")
    return _vector(position, 3), _vector(orientation, 4)


def object_aabb(obj: Any) -> tuple[np.ndarray, np.ndarray]:
    raw = getattr(obj, "aabb", None)
    raw = raw() if callable(raw) else raw
    if raw is None:
        raise RuntimeError("object AABB is unavailable")
    lower, upper = (_vector(item, 3) for item in raw)
    if np.any(upper <= lower):
        raise RuntimeError("object AABB is invalid")
    return lower, upper


def object_bottom_offset_z(obj: Any) -> float:
    """Return the live world-pose origin's height above the live AABB bottom."""

    position, _ = object_pose(obj)
    lower, _ = object_aabb(obj)
    offset = float(position[2] - lower[2])
    if not np.isfinite(offset):
        raise RuntimeError("object bottom offset is invalid")
    return offset


def candidate_aabb(
    obj: Any,
    candidate: SpatialPoseCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate an object's live AABB to one proposed origin pose."""

    position, _ = object_pose(obj)
    lower, upper = object_aabb(obj)
    delta = candidate.position - position
    return lower + delta, upper + delta


def build_candidate_collision_query_box(
    held: Any,
    candidate: SpatialPoseCandidate,
    *,
    candidate_aabb_world: tuple[Any, Any] | None = None,
    xy_margin_m: float = DEFAULT_BLOCKER_MARGIN_M,
    xy_inset_m: float = 0.0,
    z_inset_m: float = DEFAULT_COLLISION_QUERY_Z_INSET_M,
) -> CandidateCollisionQueryBox:
    """Build a conservative world-axis box for a live PhysX overlap query.

    ``candidate_aabb_world`` is caller-owned proposal geometry for operations
    that author a new orientation from a live base-aligned box.  Without it,
    the legacy translation-only AABB remains available for candidates that do
    not rotate the held object.  Both paths receive identical margins and
    strict finite/positive-extent validation.

    XY retains the existing obstacle clearance.  Z is inset slightly so an
    ordinary resting contact with a support plane is not promoted to a
    volumetric collision.  The query remains a broad box around the held
    object, but PhysX tests it against actual collision shapes rather than
    another object's possibly disconnected whole-object AABB.
    """

    if xy_margin_m < 0.0 or xy_inset_m < 0.0 or z_inset_m < 0.0:
        raise ValueError("collision query margins and insets must be non-negative")
    if candidate_aabb_world is None:
        lower, upper = candidate_aabb(held, candidate)
        geometry_source = "translated_live_aabb"
    else:
        if (
            not isinstance(candidate_aabb_world, tuple)
            or len(candidate_aabb_world) != 2
        ):
            raise TypeError("candidate_aabb_world must be a lower/upper tuple")
        lower, upper = (
            _vector(value, 3) for value in candidate_aabb_world
        )
        if np.any(upper <= lower):
            raise RuntimeError("candidate_aabb_world is invalid")
        geometry_source = "orientation_aware_candidate_aabb"
    query_lower = np.asarray(lower, dtype=np.float64).copy()
    query_upper = np.asarray(upper, dtype=np.float64).copy()
    query_lower[:2] -= xy_margin_m
    query_upper[:2] += xy_margin_m
    query_lower[:2] += xy_inset_m
    query_upper[:2] -= xy_inset_m
    query_lower[2] += z_inset_m
    query_upper[2] -= z_inset_m
    extent = query_upper - query_lower
    if not np.isfinite(np.concatenate((query_lower, query_upper))).all():
        raise RuntimeError("collision query box contains non-finite values")
    if np.any(extent <= 0.0):
        raise RuntimeError("collision query box has non-positive extent")
    return CandidateCollisionQueryBox(
        center_xyz=(query_lower + query_upper) / 2.0,
        half_extent_xyz=extent / 2.0,
        rotation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
        xy_margin_m=float(xy_margin_m),
        xy_inset_m=float(xy_inset_m),
        z_inset_m=float(z_inset_m),
        geometry_source=geometry_source,
    )


def _normalized_collision_path(value: Any) -> str | None:
    if value is None:
        return None
    path = str(value).strip()
    return path if path.startswith("/") else None


def _collision_path_matches_identity(
    path: str | None,
    identity_paths: Sequence[str],
) -> bool:
    if path is None:
        return False
    return any(path == root or path.startswith(f"{root}/") for root in identity_paths)


def summarize_candidate_collision_overlap(
    *,
    query_box: CandidateCollisionQueryBox | None,
    raw_hits: Sequence[CollisionOverlapHitEvidence],
    excluded_identity_paths: Sequence[str],
    query_error: BaseException | str | None = None,
) -> CollisionOverlapAdmissionReport:
    """Classify raw PhysX overlap hits without object/category special cases.

    Exclusions are exact runtime identities owned by the action transaction:
    the held closure, relation target, and selected support.  Every other
    collision-shape hit is a blocker, including unregistered structural
    geometry.  Malformed evidence and query errors fail closed.
    """

    normalized_exclusions = tuple(
        sorted(
            {
                path
                for item in excluded_identity_paths
                if (path := _normalized_collision_path(item)) is not None
            }
        )
    )
    normalized_hits = tuple(
        sorted(
            {
                CollisionOverlapHitEvidence(
                    rigid_body_path=_normalized_collision_path(
                        hit.rigid_body_path
                    ),
                    collision_path=_normalized_collision_path(
                        hit.collision_path
                    ),
                )
                for hit in raw_hits
            },
            key=lambda hit: (
                hit.rigid_body_path or "",
                hit.collision_path or "",
            ),
        )
    )
    error_text = None if query_error is None else str(query_error)
    if query_box is None or error_text is not None:
        return CollisionOverlapAdmissionReport(
            accepted=False,
            reason=(
                "collision_query_unavailable"
                if query_box is None
                else "collision_query_error"
            ),
            query_box=query_box,
            raw_hits=normalized_hits,
            excluded_hits=(),
            blocking_hits=normalized_hits,
            query_error=error_text,
        )

    excluded: list[CollisionOverlapHitEvidence] = []
    blocking: list[CollisionOverlapHitEvidence] = []
    malformed = False
    for hit in normalized_hits:
        paths = (hit.rigid_body_path, hit.collision_path)
        if not any(paths):
            malformed = True
            blocking.append(hit)
            continue
        if any(
            _collision_path_matches_identity(path, normalized_exclusions)
            for path in paths
        ):
            excluded.append(hit)
        else:
            blocking.append(hit)
    accepted = not blocking and not malformed
    return CollisionOverlapAdmissionReport(
        accepted=accepted,
        reason=(
            "clear_collision_shape_overlap"
            if accepted
            else (
                "malformed_collision_overlap_hit"
                if malformed
                else "blocking_collision_shape_overlap"
            )
        ),
        query_box=query_box,
        raw_hits=normalized_hits,
        excluded_hits=tuple(excluded),
        blocking_hits=tuple(blocking),
        query_error=None,
    )


def build_downward_support_ray_requests(
    held: Any,
    candidate: SpatialPoseCandidate,
    *,
    ray_above_m: float = DEFAULT_LOCAL_SUPPORT_RAY_ABOVE_M,
    ray_below_m: float = DEFAULT_LOCAL_SUPPORT_RAY_BELOW_M,
) -> tuple[DownwardSupportRayRequest, ...]:
    """Return center + four-corner downward probes over the held footprint.

    This function only constructs ray vectors; it performs no simulator call.
    The corner points are slightly inset so a numerical edge hit does not
    randomly select a neighbouring vertical face.  Exactly these five sample
    names are required by :func:`rebase_candidate_to_local_support`.
    """

    if ray_above_m <= 0.0 or ray_below_m <= 0.0:
        raise ValueError("local support ray distances must be positive")
    candidate_min, candidate_max = candidate_aabb(held, candidate)
    extent_xy = candidate_max[:2] - candidate_min[:2]
    if np.any(extent_xy <= 0.0):
        raise RuntimeError("candidate footprint is invalid")
    inset_xy = np.minimum(
        np.full(2, DEFAULT_SUPPORT_INSET_M, dtype=np.float64),
        extent_xy * 0.20,
    )
    lower_xy = candidate_min[:2] + inset_xy
    upper_xy = candidate_max[:2] - inset_xy
    center_xy = (candidate_min[:2] + candidate_max[:2]) / 2.0
    sample_xy = (
        center_xy,
        np.asarray((lower_xy[0], lower_xy[1]), dtype=np.float64),
        np.asarray((lower_xy[0], upper_xy[1]), dtype=np.float64),
        np.asarray((upper_xy[0], lower_xy[1]), dtype=np.float64),
        np.asarray((upper_xy[0], upper_xy[1]), dtype=np.float64),
    )
    start_z = float(candidate_min[2] + ray_above_m)
    end_z = float(candidate_min[2] - ray_below_m)
    return tuple(
        DownwardSupportRayRequest(
            sample_name=name,
            start_xyz=np.asarray((xy[0], xy[1], start_z), dtype=np.float64),
            end_xyz=np.asarray((xy[0], xy[1], end_z), dtype=np.float64),
        )
        for name, xy in zip(LOCAL_SUPPORT_RAY_SAMPLE_NAMES, sample_xy, strict=True)
    )


def rebase_candidate_to_local_support(
    held: Any,
    candidate: SpatialPoseCandidate,
    ray_evidence: Sequence[DownwardSupportRayEvidence],
    *,
    forbidden_entity_roles: Iterable[str] = (
        DEFAULT_FORBIDDEN_LOCAL_SUPPORT_ENTITY_ROLES
    ),
    normal_z_min: float = DEFAULT_LOCAL_SUPPORT_NORMAL_Z_MIN,
    coplanar_tolerance_m: float = DEFAULT_LOCAL_SUPPORT_COPLANAR_TOLERANCE_M,
    min_consistent_hits: int = DEFAULT_LOCAL_SUPPORT_MIN_CONSISTENT_HITS,
    clearance_m: float = DEFAULT_SUPPORT_CLEARANCE_M,
    allow_overhang: bool = False,
) -> LocalSupportRebaseReport:
    """Rebase a candidate to a ray-measured, locally horizontal support.

    All five footprint rays must return finite hits.  Every hit must be on a
    horizontal face and outside the caller-declared forbidden entity roles.
    At least ``min_consistent_hits`` hit heights must agree with the five-hit
    median within ``coplanar_tolerance_m``.  The candidate bottom is then set
    to ``median(hit_z) + clearance_m``.  Object identity is audit-only and is
    never used for admission.

    Symbolic overhang uses the centre hit's coplanar cluster; corner rays may
    miss or hit a lower surface. Faults still reject the candidate.
    Successful rebasing is deliberately not a semantic commit.  The rebased
    candidate still needs the ordinary blocker check, execution / settle, and
    official relation readback.
    """

    evidence = tuple(ray_evidence)
    normalized_forbidden = frozenset(
        str(item).strip().lower()
        for item in forbidden_entity_roles
        if str(item).strip()
    )
    required = int(min_consistent_hits)

    if not 0.0 < normal_z_min <= 1.0:
        raise ValueError("local support normal threshold must be in (0, 1]")
    if coplanar_tolerance_m < 0.0 or clearance_m < 0.0:
        raise ValueError("local support tolerances must be non-negative")
    if not 3 <= required <= len(LOCAL_SUPPORT_RAY_SAMPLE_NAMES):
        raise ValueError("local support consistent-hit count must be in [3, 5]")

    original_bottom_z: float | None
    try:
        candidate_min, _ = candidate_aabb(held, candidate)
        original_bottom_z = float(candidate_min[2])
    except Exception:
        original_bottom_z = None

    def rejected(
        reason: str,
        *,
        median_hit_z: float | None = None,
        coherent_hit_count: int = 0,
        forbidden_roles_seen: Iterable[str] = (),
    ) -> LocalSupportRebaseReport:
        return LocalSupportRebaseReport(
            accepted=False,
            reason=reason,
            original_candidate=candidate,
            rebased_candidate=None,
            original_bottom_z_m=original_bottom_z,
            median_hit_z_m=median_hit_z,
            rebased_bottom_z_m=None,
            coherent_hit_count=coherent_hit_count,
            required_coherent_hit_count=required,
            ray_evidence=evidence,
            forbidden_roles_seen=tuple(sorted(set(forbidden_roles_seen))),
        )

    if original_bottom_z is None:
        return rejected("candidate_geometry_unavailable")
    if len(evidence) != len(LOCAL_SUPPORT_RAY_SAMPLE_NAMES):
        return rejected("expected_exactly_five_support_rays")
    evidence_by_name = {item.sample_name: item for item in evidence}
    if (
        len(evidence_by_name) != len(evidence)
        or set(evidence_by_name) != set(LOCAL_SUPPORT_RAY_SAMPLE_NAMES)
    ):
        return rejected("support_ray_samples_incomplete_or_duplicated")
    ordered = tuple(evidence_by_name[name] for name in LOCAL_SUPPORT_RAY_SAMPLE_NAMES)
    if any(item.error is not None for item in ordered):
        return rejected("support_ray_error")
    if any((item.hit_position_xyz is None) != (item.hit_normal_xyz is None) for item in ordered):
        return rejected("support_ray_vector_invalid")

    if allow_overhang and ordered[0].hit_position_xyz is None:
        return rejected("support_ray_no_hit")
    hit_zs: list[float] = []
    forbidden_seen: list[str] = []
    for item in ordered:
        if allow_overhang and item.hit_position_xyz is None and item.hit_normal_xyz is None:
            continue
        if item.hit_position_xyz is None or item.hit_normal_xyz is None:
            return rejected("support_ray_no_hit")
        if item.entity_role is None or not str(item.entity_role).strip():
            return rejected("support_ray_entity_role_missing")
        role = str(item.entity_role).strip().lower()
        if role in normalized_forbidden:
            forbidden_seen.append(role)
        try:
            start = _vector(item.start_xyz, 3)
            end = _vector(item.end_xyz, 3)
            hit = _vector(item.hit_position_xyz, 3)
            normal = _vector(item.hit_normal_xyz, 3)
        except Exception:
            return rejected("support_ray_vector_invalid")
        if (
            not end[2] < start[2]
            or not np.allclose(start[:2], end[:2], atol=1e-6, rtol=0.0)
            or not np.allclose(start[:2], hit[:2], atol=1e-4, rtol=0.0)
            or hit[2] < end[2] - 1e-4
            or hit[2] > start[2] + 1e-4
        ):
            return rejected("support_ray_vector_invalid")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            return rejected("support_ray_vector_invalid")
        if float(normal[2] / normal_norm) < normal_z_min:
            return rejected("support_ray_hit_is_not_horizontal")
        hit_zs.append(float(hit[2]))

    if forbidden_seen:
        return rejected(
            "support_ray_hit_forbidden_entity_role",
            forbidden_roles_seen=forbidden_seen,
        )

    if allow_overhang:
        center_z = hit_zs[0]
        hit_zs = [z for z in hit_zs if abs(z - center_z) <= coplanar_tolerance_m]
        required = 1
    median_hit_z = float(np.median(hit_zs))
    coherent_count = sum(
        abs(value - median_hit_z) <= coplanar_tolerance_m for value in hit_zs
    )
    if coherent_count < required:
        return rejected(
            "support_ray_hits_lack_coplanar_majority",
            median_hit_z=median_hit_z,
            coherent_hit_count=coherent_count,
        )

    rebased_bottom_z = median_hit_z + clearance_m
    rebased_position = _vector(candidate.position, 3).copy()
    rebased_position[2] += rebased_bottom_z - original_bottom_z
    rebased = SpatialPoseCandidate(
        name=candidate.name,
        position=rebased_position,
        orientation_xyzw=_vector(candidate.orientation_xyzw, 4).copy(),
    )
    return LocalSupportRebaseReport(
        accepted=True,
        reason=None,
        original_candidate=candidate,
        rebased_candidate=rebased,
        original_bottom_z_m=original_bottom_z,
        median_hit_z_m=median_hit_z,
        rebased_bottom_z_m=rebased_bottom_z,
        coherent_hit_count=coherent_count,
        required_coherent_hit_count=required,
        ray_evidence=evidence,
    )


def _screen_axes_xy(
    *,
    observer_xy: Any,
    target_center_xy: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return screen-right and screen-near axes on the world XY plane.

    These axes intentionally use only the observer and target positions.  The
    visual marker is a coarse screen-relative choice, so reconstructing a
    camera ray or estimating a surface normal would add precision that the
    public contract neither exposes nor needs.
    """

    observer = _vector(observer_xy, 2)
    target = _vector(target_center_xy, 2)
    near = observer - target
    norm = float(np.linalg.norm(near))
    if norm <= 1e-9:
        raise ValueError("observer XY must be distinct from target center XY")
    near /= norm
    right = np.asarray((-near[1], near[0]), dtype=np.float64)
    return right, near


def shelf_screen_cell_aabb(
    container_low: Any,
    container_high: Any,
    *,
    observer_xy: Any,
    row_index: int,
    row_count: int,
    column_index: int,
    column_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a container AABB into one top-down, screen-left-to-right cubby."""

    low = _vector(container_low, 3).copy()
    high = _vector(container_high, 3).copy()
    if np.any(high <= low):
        raise ValueError("container AABB must have positive extent")
    for index, count, name in (
        (row_index, row_count, "row"),
        (column_index, column_count, "column"),
    ):
        if (
            isinstance(index, bool)
            or isinstance(count, bool)
            or not isinstance(index, int)
            or not isinstance(count, int)
            or count < 1
            or not 1 <= index <= count
        ):
            raise ValueError(f"invalid shelf {name} selector")

    result_low, result_high = low.copy(), high.copy()
    row_height = (high[2] - low[2]) / row_count
    result_low[2] = low[2] + (row_count - row_index) * row_height
    result_high[2] = result_low[2] + row_height

    screen_right, _ = _screen_axes_xy(
        observer_xy=observer_xy,
        target_center_xy=(low[:2] + high[:2]) / 2.0,
    )
    horizontal_extent = high[:2] - low[:2]
    axis = int(np.argmax(np.abs(screen_right) * horizontal_extent))
    world_column = (
        column_index - 1
        if screen_right[axis] >= 0.0
        else column_count - column_index
    )
    column_width = horizontal_extent[axis] / column_count
    result_low[axis] = low[axis] + world_column * column_width
    result_high[axis] = result_low[axis] + column_width
    return result_low, result_high


def _candidate_center_xy(
    held: Any,
    candidate: SpatialPoseCandidate,
) -> np.ndarray:
    lower, upper = candidate_aabb(held, candidate)
    return (lower[:2] + upper[:2]) / 2.0


def select_touching_screen_candidate(
    candidates: Sequence[SpatialPoseCandidate],
    *,
    held: Any,
    target: Any,
    observer_xy: Any,
    selector: str,
) -> tuple[SpatialPoseCandidate, ...]:
    """Map one coarse screen direction to exactly one lateral candidate.

    The four existing world-axis candidates remain the only executable poses.
    This function merely ranks them from the current observer.  If every
    admitted candidate lies on the opposite half-plane, it returns no
    candidate instead of silently executing a different direction.
    """

    if selector not in SIMPLE_TOUCHING_SCREEN_SELECTORS:
        raise ValueError(f"unsupported Touching screen selector: {selector!r}")
    admitted = tuple(candidates)
    if not admitted:
        return ()

    target_min, target_max = object_aabb(target)
    target_center_xy = (target_min[:2] + target_max[:2]) / 2.0
    screen_right, screen_near = _screen_axes_xy(
        observer_xy=observer_xy,
        target_center_xy=target_center_xy,
    )
    desired = {
        "screen_left": -screen_right,
        "screen_right": screen_right,
        "screen_near": screen_near,
        "screen_far": -screen_near,
    }[selector]
    ranking: list[tuple[float, str, int]] = []
    for index, candidate in enumerate(admitted):
        delta = _candidate_center_xy(held, candidate) - target_center_xy
        norm = float(np.linalg.norm(delta))
        alignment = (
            float(np.dot(delta / norm, desired)) if norm > 1e-9 else -1.0
        )
        ranking.append((-alignment, candidate.name, index))
    ranking.sort()
    best_alignment = -ranking[0][0]
    if best_alignment <= 1e-9:
        return ()
    return (admitted[ranking[0][2]],)


def select_next_to_screen_candidate(
    candidates: Sequence[SpatialPoseCandidate],
    *,
    held: Any,
    target: Any,
    observer_xy: Any,
    selector: str,
) -> tuple[SpatialPoseCandidate, ...]:
    """Map one of eight screen-relative positions to one admitted candidate."""

    if selector not in SIMPLE_NEXT_TO_SCREEN_SELECTORS:
        raise ValueError(f"unsupported NextTo screen selector: {selector!r}")
    admitted = tuple(candidates)
    if not admitted:
        return ()

    target_min, target_max = object_aabb(target)
    target_center_xy = (target_min[:2] + target_max[:2]) / 2.0
    screen_right, screen_near = _screen_axes_xy(
        observer_xy=observer_xy,
        target_center_xy=target_center_xy,
    )
    desired = {
        "screen_left": -screen_right,
        "screen_far_left": -screen_right - screen_near,
        "screen_far": -screen_near,
        "screen_far_right": screen_right - screen_near,
        "screen_right": screen_right,
        "screen_near_right": screen_right + screen_near,
        "screen_near": screen_near,
        "screen_near_left": -screen_right + screen_near,
    }[selector]
    desired /= float(np.linalg.norm(desired))
    ranking: list[tuple[float, str, int]] = []
    for index, candidate in enumerate(admitted):
        delta = _candidate_center_xy(held, candidate) - target_center_xy
        norm = float(np.linalg.norm(delta))
        alignment = (
            float(np.dot(delta / norm, desired)) if norm > 1e-9 else -1.0
        )
        ranking.append((-alignment, candidate.name, index))
    ranking.sort()
    best_alignment = -ranking[0][0]
    if best_alignment <= 1e-9:
        return ()
    return (admitted[ranking[0][2]],)


def _object_velocity(obj: Any, kind: str) -> np.ndarray:
    getter_name = f"get_{kind}_velocity"
    owners = (obj, getattr(obj, "root_link", None))
    for owner in owners:
        getter = getattr(owner, getter_name, None)
        if not callable(getter):
            continue
        try:
            value = getter()
        except TypeError:
            value = getter(frame="scene")
        return _vector(value, 3)
    raise RuntimeError(f"object {kind} velocity is unavailable")


def quaternion_tilt_rad(orientation_xyzw: Any) -> float:
    """Return the angle between the object's local +Z and world +Z."""

    orientation = _vector(orientation_xyzw, 4)
    norm = float(np.linalg.norm(orientation))
    if norm <= 1e-12:
        raise ValueError("orientation quaternion has zero norm")
    x, y, _, _ = orientation / norm
    world_up_dot = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    return float(np.arccos(world_up_dot))


def state_by_name(obj: Any, state_name: str) -> Any | None:
    expected = state_name.casefold()
    for key, state in (getattr(obj, "states", {}) or {}).items():
        name = getattr(key, "__name__", type(key).__name__).casefold()
        if name == expected:
            return state
    return None


def canonical_relation(obj: Any, predicate: str, other: Any) -> bool:
    state = state_by_name(obj, predicate)
    getter = getattr(state, "get_value", None)
    if not callable(getter):
        raise RuntimeError(f"canonical {predicate} state is unavailable")
    return bool(getter(other))


def read_action_relation(obj: Any, predicate: str, other: Any = None, *, predicate_mode: str = "official") -> bool:
    """One relation contract for construction, actions and task evaluation."""
    if predicate_mode not in {"official", "action20_symbolic"}:
        raise ValueError("unknown predicate mode")
    if predicate_mode == "action20_symbolic" and predicate.casefold() == "ontop":
        return symbolic_on_top_relation(obj, other)
    if predicate_mode == "action20_symbolic" and predicate.casefold() == "nextto":
        return symbolic_next_to_relation(obj, other)
    state = state_by_name(obj, predicate)
    getter = getattr(state, "get_value", None)
    if not callable(getter):
        raise RuntimeError(f"canonical {predicate} state is unavailable")
    return bool(getter() if other is None else getter(other))


def symbolic_next_to_relation(obj: Any, other: Any) -> bool:
    """AABB proximity beside the reference, without native horizontal rays.

    Keep the official size-dependent 3D distance gate. Require separation on
    at least one horizontal axis and intersecting height ranges (no minimum
    overlap ratio). One millimetre only absorbs pose/AABB rounding at edges.
    Collision and actual support remain separate execution admission gates.
    """
    if obj is other:
        return False
    lower, upper = object_aabb(obj)
    other_lower, other_upper = object_aabb(other)
    separation = np.maximum(lower, other_lower) - np.minimum(upper, other_upper)
    limit = float(np.mean((upper - lower) + (other_upper - other_lower))) * OFFICIAL_NEXT_TO_DISTANCE_RATIO
    edge_tolerance_m = 0.001
    return bool(
        np.linalg.norm(np.maximum(separation, 0.0)) <= limit
        and np.any(separation[:2] >= -edge_tolerance_m)
        and separation[2] <= edge_tolerance_m
    )


def _query_on_top_surface_z(
    obj: Any,
    support: Any,
    *,
    maximum_gap_m: float,
    maximum_penetration_m: float,
) -> float | None:
    """Return nearby surface Z, or None for a valid miss; query faults raise."""

    from omnigibson.utils.sampling_utils import raytest_batch

    # Use the simulator's live link identities, not object/category names.
    ignored = tuple(obj.link_prim_paths)
    support_paths = tuple(support.link_prim_paths)
    if not ignored or not support_paths:
        raise RuntimeError("OnTop surface query requires live link identities")
    position, orientation = object_pose(obj)
    obj_min, obj_max = object_aabb(obj)
    requests = build_downward_support_ray_requests(
        obj,
        SpatialPoseCandidate("on_top_readback", position, orientation),
        # Start above the held object, not above a cabinet roof that would
        # occlude its internal shelf. Do not start inside a cushion at its foot.
        ray_above_m=float(obj_max[2] - obj_min[2] + 0.01),
        ray_below_m=maximum_gap_m + DEFAULT_SOLID_OVERLAP_TOLERANCE_M,
    )
    results = raytest_batch(
        [request.start_xyz for request in requests],
        [request.end_xyz for request in requests],
        only_closest=True,
        ignore_bodies=ignored,
        ignore_collisions=ignored,
    )
    if len(results) != len(requests):
        raise RuntimeError("OnTop surface query returned an incomplete batch")
    heights = []
    for result in results:
        if not isinstance(result.get("hit"), (bool, np.bool_)):
            raise ValueError("OnTop surface query returned an invalid hit flag")
        if not result["hit"]:
            continue
        path = _normalized_collision_path(result.get("rigidBody"))
        if path is None:
            raise ValueError("OnTop surface hit has no live rigid-body identity")
        if not _collision_path_matches_identity(path, support_paths):
            continue
        hit_position = _vector(result.get("position"), 3)
        normal = _vector(result.get("normal"), 3)
        if normal[2] > 0.0:
            heights.append(float(hit_position[2]))
    # ponytail: five footprint samples, not a full contact manifold. Keep
    # Symbolic permissive; require no all-corner contact or coplanarity.
    return max(heights, default=None)


def symbolic_on_top_relation(
    obj: Any,
    support: Any,
    *,
    maximum_gap_m: float = DEFAULT_MAX_SUPPORT_GAP_M,
    maximum_penetration_m: float = DEFAULT_SOLID_OVERLAP_TOLERANCE_M,
    footprint_tolerance_m: float = DEFAULT_SOLID_OVERLAP_TOLERANCE_M,
    support_surface_z_m: float | None = None,
) -> bool:
    """Read Action20 ``OnTop`` from final geometry without contact state.

    Action20 commits poses without advancing physics, so PhysX ``Touching``
    cannot be an authoritative postcondition.  Candidate planning and the
    shared final-overlap gate remain responsible for exact surface and solid
    collision admission. A selected surface height is a planning hint, not
    proof of support under the final pose. All readers query the same nearby
    actual surface with the same gap limits; failures propagate to the verifier.
    """

    if obj is support:
        return False
    tolerances = (
        float(maximum_gap_m),
        float(maximum_penetration_m),
        float(footprint_tolerance_m),
    )
    if not np.isfinite(tolerances).all() or any(value < 0.0 for value in tolerances):
        raise ValueError("symbolic OnTop tolerances must be finite and non-negative")

    obj_min, obj_max = object_aabb(obj)
    support_min, support_max = object_aabb(support)
    # Overhang is legal in Symbolic. A positive horizontal intersection is
    # only a broad precheck; the selected plane / live rays prove support.
    footprint_overlap = np.minimum(obj_max[:2], support_max[:2]) - np.maximum(
        obj_min[:2], support_min[:2]
    )
    # A selected plane is a planner hint, never a second relation definition.
    if np.any(footprint_overlap <= tolerances[2]):
        return False
    if not (
        support_min[2] - tolerances[1]
        <= obj_min[2]
        <= support_max[2] + tolerances[0]
    ):
        return False
    surface_z = _query_on_top_surface_z(
        obj,
        support,
        maximum_gap_m=tolerances[0],
        maximum_penetration_m=tolerances[1],
    )
    if surface_z is None:
        return False
    if not (
        support_min[2] - tolerances[1]
        <= surface_z
        <= support_max[2] + tolerances[0]
    ):
        return False
    # The global AABB minimum is not the local contact point of an irregular
    # object. Permit a surface within its vertical envelope; the shared solid
    # collision gate, not a second AABB penetration rule, owns nonpenetration.
    return bool(obj_min[2] - tolerances[0] <= surface_z <= obj_max[2] + tolerances[1])


def infer_unique_on_top_support(
    target: Any,
    candidates: Iterable[Any],
) -> tuple[Any | None, tuple[str, ...]]:
    """Return the sole canonical support; ambiguity fails closed."""

    supports: list[Any] = []
    for candidate in candidates:
        if candidate is None or candidate is target:
            continue
        try:
            if canonical_relation(target, "OnTop", candidate):
                supports.append(candidate)
        except Exception:
            continue
    names = tuple(
        sorted(str(getattr(candidate, "name", "")) for candidate in supports)
    )
    return (supports[0] if len(supports) == 1 else None), names


def next_to_gap_for_extents(
    held_extent: Any,
    target_extent: Any,
) -> float:
    """Choose a gap below BEHAVIOR's size-dependent NextTo distance gate.

    The 2025 executor used a fixed 3 cm gap.  BEHAVIOR 2026 first rejects
    pairs whose AABB separation is greater than
    ``mean(held_extent + target_extent) / 6``.  A fixed 3 cm therefore makes
    NextTo impossible for small objects such as batteries.  Keep 3 cm as the
    upper bound, while reserving numerical margin below the official gate.
    """

    held = _vector(held_extent, 3)
    target = _vector(target_extent, 3)
    official_limit = float(
        np.mean(held + target) * OFFICIAL_NEXT_TO_DISTANCE_RATIO
    )
    return max(
        0.0,
        min(
            DEFAULT_NEXT_TO_GAP_M,
            official_limit * DEFAULT_NEXT_TO_THRESHOLD_FRACTION,
        ),
    )


def _bounded_surface_candidates(
    held: Any,
    *,
    surface_min_xy: np.ndarray,
    surface_max_xy: np.ndarray,
    support_top_z: float,
    name_prefix: str,
) -> tuple[SpatialPoseCandidate, ...]:
    """Generate a deterministic, bounded live-surface search.

    The candidate origin Z is computed from the held object's live origin to
    AABB-bottom offset.  This remains correct for assets whose origin is not
    at the AABB center.
    """

    held_position, held_orientation = object_pose(held)
    held_min, held_max = object_aabb(held)
    held_half_xy = (held_max - held_min)[:2] / 2.0
    held_center_offset_xy = ((held_min + held_max) / 2.0 - held_position)[:2]
    available_min = _vector(surface_min_xy, 2) + held_half_xy + DEFAULT_SUPPORT_INSET_M
    available_max = _vector(surface_max_xy, 2) - held_half_xy - DEFAULT_SUPPORT_INSET_M
    if np.any(available_min > available_max):
        return ()

    origin_z = (
        float(support_top_z)
        + object_bottom_offset_z(held)
        + DEFAULT_SUPPORT_CLEARANCE_M
    )
    output: list[SpatialPoseCandidate] = []
    for name, x_fraction, y_fraction in DEFAULT_PLACEMENT_FRACTIONS:
        center_xy = available_min + np.asarray(
            (x_fraction, y_fraction), dtype=np.float64
        ) * (available_max - available_min)
        origin_xy = center_xy - held_center_offset_xy
        output.append(
            SpatialPoseCandidate(
                name=f"{name_prefix}{name}",
                position=np.asarray((origin_xy[0], origin_xy[1], origin_z)),
                orientation_xyzw=held_orientation.copy(),
            )
        )
    return tuple(output)


def plan_default_on_top_candidates(
    held: Any,
    support: Any,
    *,
    scene_objects: Sequence[Any] = (),
) -> SpatialCandidatePlan:
    """Plan bounded OnTop poses from live AABBs without guessed origin Z."""

    support_min, support_max = object_aabb(support)
    raw_candidates = _bounded_surface_candidates(
        held,
        surface_min_xy=support_min[:2],
        surface_max_xy=support_max[:2],
        support_top_z=float(support_max[2]),
        name_prefix="on_top_",
    )
    if not raw_candidates:
        return SpatialCandidatePlan(
            candidates=(),
            admission_reports=(),
            planning_error="held footprint does not fit inside live support AABB",
        )

    candidates: list[SpatialPoseCandidate] = []
    reports: list[CandidateAdmissionReport] = []
    for candidate in raw_candidates:
        blockers = candidate_blockers(
            held=held,
            target=support,
            support=support,
            candidate=candidate,
            scene_objects=scene_objects,
            margin_m=DEFAULT_PLACEMENT_BLOCKER_MARGIN_M,
        )
        candidate_min, _ = candidate_aabb(held, candidate)
        gap = float(candidate_min[2] - support_max[2])
        accepted = not blockers
        reports.append(
            CandidateAdmissionReport(
                candidate=candidate.name,
                accepted=accepted,
                reason=None if accepted else "candidate AABB intersects a scene blocker",
                blockers=blockers,
                bottom_to_support_gap_m=gap,
            )
        )
        if accepted:
            candidates.append(candidate)
    return SpatialCandidatePlan(
        candidates=tuple(candidates),
        admission_reports=tuple(reports),
    )


def rank_default_under_supports(
    held: Any,
    target: Any,
    candidates: Iterable[Any],
    *,
    vertical_probe: VerticalProbe | None = None,
    scene_objects: Sequence[Any] = (),
) -> tuple[Any, ...]:
    """Rank only supports with ray-proven, clear Under candidates.

    Missing vertical evidence fails closed.  This deliberately removes the
    old behavior that treated a furniture object's whole AABB footprint as a
    usable underside.
    """

    return tuple(
        item.support
        for item in plan_default_under_supports(
            held,
            target,
            candidates,
            vertical_probe=vertical_probe,
            scene_objects=scene_objects,
        )
        if item.candidate_plan.candidates
    )


def default_lateral_candidates(
    held: Any,
    target: Any,
    support: Any,
    *,
    relation: str,
) -> tuple[SpatialPoseCandidate, ...]:
    """Build the 2025-compatible +X,-X,+Y,-Y default candidate order."""

    if relation not in {"NextTo", "Touching"}:
        raise ValueError("lateral relation must be NextTo or Touching")
    held_position, held_orientation = object_pose(held)
    held_min, held_max = object_aabb(held)
    target_min, target_max = object_aabb(target)
    support_min, support_max = object_aabb(support)
    held_extent = held_max - held_min
    target_extent = target_max - target_min
    held_center_offset = (held_min + held_max) / 2.0 - held_position
    target_center = (target_min + target_max) / 2.0
    gap = (
        -DEFAULT_TOUCHING_APPROACH_M
        if relation == "Touching"
        else next_to_gap_for_extents(held_extent, target_extent)
    )
    output: list[SpatialPoseCandidate] = []
    for name, axis, sign in DEFAULT_LATERAL_ORDER:
        center = target_center.copy()
        center[axis] += sign * (
            (target_extent[axis] + held_extent[axis]) / 2.0 + gap
        )
        center[2] = (
            float(support_max[2])
            + float(held_extent[2]) / 2.0
            + DEFAULT_SUPPORT_CLEARANCE_M
        )
        if np.any(
            center[:2] < support_min[:2] + DEFAULT_SUPPORT_INSET_M
        ) or np.any(
            center[:2] > support_max[:2] - DEFAULT_SUPPORT_INSET_M
        ):
            continue
        output.append(
            SpatialPoseCandidate(
                name=name,
                position=center - held_center_offset,
                orientation_xyzw=held_orientation.copy(),
            )
        )
    return tuple(output)


def expanded_lateral_candidates(
    held: Any,
    target: Any,
    support: Any,
    *,
    relation: str,
) -> tuple[SpatialPoseCandidate, ...]:
    """Search each target face from its centre to both near-edge regions.

    The legacy four face-centre poses are always attempted first.  Large
    furniture often has a leg, wall, or another object on the face centre;
    failing the whole relation in that case is an avoidable false negative.
    The extra candidates slide only along the same face, remain on the live
    support AABB, and preserve the official size-dependent NextTo gap.  Near
    edge samples are important for furniture whose low horizontal geometry is
    concentrated in legs rather than the face centre.  No task, category,
    object-name, or instance information participates in the search.
    """

    if relation not in {"NextTo", "Touching"}:
        raise ValueError("lateral relation must be NextTo or Touching")

    held_position, held_orientation = object_pose(held)
    held_min, held_max = object_aabb(held)
    target_min, target_max = object_aabb(target)
    support_min, support_max = object_aabb(support)
    held_extent = held_max - held_min
    target_extent = target_max - target_min
    held_center_offset = (held_min + held_max) / 2.0 - held_position
    target_center = (target_min + target_max) / 2.0
    gap = (
        -DEFAULT_TOUCHING_APPROACH_M
        if relation == "Touching"
        else next_to_gap_for_extents(held_extent, target_extent)
    )

    available_min = support_min[:2] + held_extent[:2] / 2.0 + DEFAULT_SUPPORT_INSET_M
    available_max = support_max[:2] - held_extent[:2] / 2.0 - DEFAULT_SUPPORT_INSET_M

    output: list[SpatialPoseCandidate] = []
    offsets = (("center", 0.0), *DEFAULT_LATERAL_FACE_OFFSETS)
    # Preserve the historical four-candidate prefix before trying offsets.
    for offset_name, offset_fraction in offsets:
        for face_name, axis, sign in DEFAULT_LATERAL_ORDER:
            perpendicular_axis = 1 - axis
            center = target_center.copy()
            center[axis] += sign * (
                (target_extent[axis] + held_extent[axis]) / 2.0 + gap
            )
            center[perpendicular_axis] += (
                offset_fraction * target_extent[perpendicular_axis]
            )
            center[2] = (
                float(support_max[2])
                + float(held_extent[2]) / 2.0
                + DEFAULT_SUPPORT_CLEARANCE_M
            )
            if np.any(center[:2] < available_min) or np.any(
                center[:2] > available_max
            ):
                continue
            candidate_name = (
                face_name
                if offset_name == "center"
                else f"{face_name}_{offset_name}"
            )
            output.append(
                SpatialPoseCandidate(
                    name=candidate_name,
                    position=center - held_center_offset,
                    orientation_xyzw=held_orientation.copy(),
                )
            )
    return tuple(output)


def expanded_support_agnostic_lateral_candidates(
    held: Any,
    target: Any,
    *,
    relation: str,
) -> tuple[SpatialPoseCandidate, ...]:
    """Generate lateral XY candidates before discovering local support.

    ``NextTo`` and ``Touching`` are relations between the held object and the
    target, not between the target and one globally inferred support object.
    Large furniture and objects spanning multiple floor pieces frequently do
    not have a unique canonical ``OnTop`` support.  Requiring one before
    searching therefore rejects valid lateral placements.

    This planner preserves the deterministic 28-candidate face / edge order
    of :func:`expanded_lateral_candidates`, but deliberately applies no
    support-AABB clipping.  Its Z coordinate is only a ray-casting seed: the
    held bottom starts just above the target's live lower AABB plane.  The
    executor must rebase every candidate with the five-ray local-support
    kernel before collision admission or execution.  Consequently this
    function does not itself authorize placement or declare a relation.
    """

    if relation not in {"NextTo", "Touching"}:
        raise ValueError("lateral relation must be NextTo or Touching")

    held_position, held_orientation = object_pose(held)
    held_min, held_max = object_aabb(held)
    target_min, target_max = object_aabb(target)
    held_extent = held_max - held_min
    target_extent = target_max - target_min
    held_center_offset = (held_min + held_max) / 2.0 - held_position
    target_center = (target_min + target_max) / 2.0
    gap = (
        -DEFAULT_TOUCHING_APPROACH_M
        if relation == "Touching"
        else next_to_gap_for_extents(held_extent, target_extent)
    )

    output: list[SpatialPoseCandidate] = []
    offsets = (("center", 0.0), *DEFAULT_LATERAL_FACE_OFFSETS)
    for offset_name, offset_fraction in offsets:
        for face_name, axis, sign in DEFAULT_LATERAL_ORDER:
            perpendicular_axis = 1 - axis
            center = target_center.copy()
            center[axis] += sign * (
                (target_extent[axis] + held_extent[axis]) / 2.0 + gap
            )
            center[perpendicular_axis] += (
                offset_fraction * target_extent[perpendicular_axis]
            )
            # Seed the five downward rays near the target's lower plane.  A
            # successful local-support report replaces this nominal Z before
            # collision admission and execution.
            center[2] = (
                float(target_min[2])
                + float(held_extent[2]) / 2.0
                + DEFAULT_SUPPORT_CLEARANCE_M
            )
            candidate_name = (
                face_name
                if offset_name == "center"
                else f"{face_name}_{offset_name}"
            )
            output.append(
                SpatialPoseCandidate(
                    name=candidate_name,
                    position=center - held_center_offset,
                    orientation_xyzw=held_orientation.copy(),
                )
            )
    return tuple(output)


def default_under_candidates(
    held: Any,
    target: Any,
    support: Any,
    *,
    vertical_probe: VerticalProbe | None = None,
    scene_objects: Sequence[Any] = (),
) -> tuple[SpatialPoseCandidate, ...]:
    """Compatibility projection of the ray-aware Under candidate plan."""

    return plan_default_under_candidates(
        held,
        target,
        support,
        vertical_probe=vertical_probe,
        scene_objects=scene_objects,
    ).candidates


def plan_default_under_candidates(
    held: Any,
    target: Any,
    support: Any,
    *,
    vertical_probe: VerticalProbe | None,
    scene_objects: Sequence[Any] = (),
    allow_support_overhang: bool = False,
) -> SpatialCandidatePlan:
    """Plan only Under candidates with live vertical-ray evidence.

    The target and support AABBs provide bounded XY search limits, not proof
    of an underside.  Every admitted pose additionally needs a caller-owned
    live probe showing that the target is the first object above the complete
    held footprint, the selected support is first below it, and the vertical
    corridor is clear.  Canonical Under and support OnTop remain post-settle
    requirements.
    """

    if vertical_probe is None:
        return SpatialCandidatePlan(
            candidates=(),
            admission_reports=(),
            planning_error="live vertical probe is required for Under placement",
            planning_metadata_private={"vertical_probe_available": False},
        )

    held_min, held_max = object_aabb(held)
    target_min, target_max = object_aabb(target)
    support_min, support_max = object_aabb(support)
    support_overhang = (held_max[:2] - held_min[:2]) / 2.0 if allow_support_overhang else 0.0
    # Keep the complete object under the target. Only the lower support may
    # be narrower than the object; the live support probe remains mandatory.
    search_min = np.maximum(target_min[:2], support_min[:2] - support_overhang)
    search_max = np.minimum(target_max[:2], support_max[:2] + support_overhang)
    raw_candidates = _bounded_surface_candidates(
        held,
        surface_min_xy=search_min,
        surface_max_xy=search_max,
        support_top_z=float(support_max[2]),
        name_prefix="under_ray_",
    )
    if not raw_candidates:
        return SpatialCandidatePlan(
            candidates=(),
            admission_reports=(),
            planning_error="held footprint does not fit target/support XY intersection",
            planning_metadata_private={"footprint_fits": False},
        )

    held_height = float(held_max[2] - held_min[2])
    required_clearance = held_height + DEFAULT_MIN_UNDER_HEADROOM_MARGIN_M
    candidates: list[SpatialPoseCandidate] = []
    reports: list[CandidateAdmissionReport] = []
    for candidate in raw_candidates:
        candidate_min, _ = candidate_aabb(held, candidate)
        gap = float(candidate_min[2] - support_max[2])
        blockers = candidate_blockers(
            held=held,
            target=target,
            support=support,
            candidate=candidate,
            scene_objects=scene_objects,
            margin_m=DEFAULT_PLACEMENT_BLOCKER_MARGIN_M,
        )
        if blockers:
            reports.append(
                CandidateAdmissionReport(
                    candidate=candidate.name,
                    accepted=False,
                    reason="candidate AABB intersects a scene blocker",
                    blockers=blockers,
                    bottom_to_support_gap_m=gap,
                )
            )
            continue

        try:
            evidence = vertical_probe(candidate, held, target, support)
            if not isinstance(evidence, VerticalPlacementEvidence):
                raise TypeError("vertical probe returned an invalid evidence type")
        except Exception as exc:
            evidence = VerticalPlacementEvidence(
                target_first_above=False,
                support_first_below=False,
                clear_underside=False,
                vertical_clearance_m=None,
                error=f"{type(exc).__name__}: {exc}",
            )

        reason: str | None = None
        clearance = evidence.vertical_clearance_m
        if evidence.error is not None:
            reason = "vertical probe failed"
        elif not evidence.target_first_above:
            reason = "target is not the first live ray hit above candidate"
        elif not evidence.support_first_below:
            reason = "selected support is not the first live ray hit below candidate"
        elif not evidence.clear_underside:
            reason = "held footprint does not have a clear underside corridor"
        elif evidence.vertical_adjacency is False:
            reason = "live VerticalAdjacency evidence rejected candidate"
        elif clearance is None or not np.isfinite(clearance):
            reason = "vertical clearance is unavailable"
        elif float(clearance) < required_clearance:
            reason = "vertical clearance is too small for held object"

        accepted = reason is None
        reports.append(
            CandidateAdmissionReport(
                candidate=candidate.name,
                accepted=accepted,
                reason=reason,
                bottom_to_support_gap_m=gap,
                vertical_evidence=evidence,
            )
        )
        if accepted:
            candidates.append(candidate)

    return SpatialCandidatePlan(
        candidates=tuple(candidates),
        admission_reports=tuple(reports),
        planning_metadata_private={"required_clearance_m": required_clearance},
    )


def plan_default_under_supports(
    held: Any,
    target: Any,
    supports: Iterable[Any],
    *,
    vertical_probe: VerticalProbe | None,
    scene_objects: Sequence[Any] = (),
    allow_support_overhang: bool = False,
) -> tuple[UnderSupportPlan, ...]:
    """Plan every support and rank ray-proven candidates before rejections."""

    ranked: list[tuple[int, int, float, str, UnderSupportPlan]] = []
    for support in supports:
        if support is None or support is held or support is target:
            continue
        name = str(getattr(support, "name", "")) or "<unnamed>"
        try:
            _, support_max = object_aabb(support)
            plan = plan_default_under_candidates(
                held,
                target,
                support,
                vertical_probe=vertical_probe,
                scene_objects=scene_objects,
                allow_support_overhang=allow_support_overhang,
            )
            support_top = float(support_max[2])
        except Exception as exc:
            plan = SpatialCandidatePlan(
                candidates=(),
                admission_reports=(),
                planning_error=f"{type(exc).__name__}: {exc}",
                planning_metadata_private={"geometry_query_failed": True},
            )
            support_top = float("-inf")
        support_plan = UnderSupportPlan(support=support, candidate_plan=plan)
        ranked.append(
            (
                0 if plan.candidates else 1,
                -len(plan.candidates),
                -support_top,
                name,
                support_plan,
            )
        )
    ranked.sort(key=lambda item: item[:4])
    return tuple(item[4] for item in ranked)


def _aabb_overlap(
    first_min: np.ndarray,
    first_max: np.ndarray,
    second_min: np.ndarray,
    second_max: np.ndarray,
    *,
    margin_m: float,
) -> bool:
    """Return volumetric overlap with clearance applied only in XY.

    Placement candidates intentionally rest on a live support surface.  A
    positive clearance margin on the vertical axis turns an ordinary
    support contact (and overlapping floor / terrain layers at the same
    height) into a false collision.  Horizontal clearance is still useful
    for avoiding walls, furniture, and neighbouring objects, while the Z
    axis must show an actual positive-volume intersection before another
    object is a blocker.
    """

    return bool(
        np.all(first_max[:2] + margin_m > second_min[:2])
        and np.all(second_max[:2] + margin_m > first_min[:2])
        and first_max[2] > second_min[2]
        and second_max[2] > first_min[2]
    )


def candidate_blockers(
    *,
    held: Any,
    target: Any,
    support: Any,
    candidate: SpatialPoseCandidate,
    scene_objects: Sequence[Any],
    margin_m: float = DEFAULT_BLOCKER_MARGIN_M,
) -> tuple[str, ...]:
    """Reject default candidates intersecting another live scene object."""

    if margin_m < 0.0:
        raise ValueError("candidate blocker margin must be non-negative")

    held_position, _ = object_pose(held)
    held_min, held_max = object_aabb(held)
    delta = candidate.position - held_position
    candidate_min = held_min + delta
    candidate_max = held_max + delta
    excluded = {id(held), id(target), id(support)}
    blockers: list[str] = []
    for other in scene_objects:
        if other is None or id(other) in excluded:
            continue
        try:
            other_min, other_max = object_aabb(other)
        except Exception:
            continue
        if _aabb_overlap(
            candidate_min,
            candidate_max,
            other_min,
            other_max,
            margin_m=margin_m,
        ):
            blockers.append(str(getattr(other, "name", "")) or "<unnamed>")
    return tuple(sorted(set(blockers)))


def pair_penetration_depth(first: Any, second: Any) -> np.ndarray:
    first_min, first_max = object_aabb(first)
    second_min, second_max = object_aabb(second)
    return np.maximum(
        0.0,
        np.minimum(first_max, second_max) - np.maximum(first_min, second_min),
    )


def sample_spatial_stability(
    held: Any,
    target: Any,
    *,
    relation: str,
    support: Any,
    symbolic_on_top: bool = False,
    support_surface_z_m: float | None = None,
) -> SpatialStabilitySample:
    """Read one canonical and physical post-placement sample from live state."""

    if not isinstance(symbolic_on_top, bool):
        raise TypeError("symbolic_on_top must be boolean")
    position, orientation = object_pose(held)
    held_min, _ = object_aabb(held)
    _, support_max = object_aabb(support)
    support_z = (
        float(support_max[2])
        if support_surface_z_m is None
        else float(support_surface_z_m)
    )
    linear_velocity = _object_velocity(held, "linear")
    angular_velocity = _object_velocity(held, "angular")
    relation_value = (
        symbolic_on_top_relation(
            held,
            target,
            support_surface_z_m=support_surface_z_m,
        )
        if symbolic_on_top and relation == "OnTop"
        else read_action_relation(held, relation, target,
            predicate_mode="action20_symbolic" if symbolic_on_top else "official")
    )
    support_on_top = (
        relation_value
        if symbolic_on_top and relation == "OnTop" and support is target
        else symbolic_on_top_relation(
            held,
            support,
            support_surface_z_m=support_surface_z_m,
        )
        if symbolic_on_top
        else canonical_relation(held, "OnTop", support)
    )
    return SpatialStabilitySample(
        relation=relation_value,
        support_on_top=support_on_top,
        support_touching=None if symbolic_on_top else canonical_relation(held, "Touching", support),
        bottom_to_support_gap_m=float(held_min[2] - support_z),
        linear_speed_m_s=float(np.linalg.norm(linear_velocity)),
        angular_speed_rad_s=float(np.linalg.norm(angular_velocity)),
        tilt_rad=quaternion_tilt_rad(orientation),
        position=tuple(float(item) for item in position),
        orientation_xyzw=tuple(float(item) for item in orientation),
        bottom_z_m=float(held_min[2]),
    )


def summarize_spatial_stability(
    samples: Sequence[SpatialStabilitySample],
    *,
    required_samples: int = DEFAULT_STABILITY_WINDOW,
    max_support_gap_m: float = DEFAULT_MAX_SUPPORT_GAP_M,
    max_support_penetration_m: float = DEFAULT_MAX_SUPPORT_PENETRATION_M,
    max_linear_speed_m_s: float = DEFAULT_MAX_LINEAR_SPEED_M_S,
    max_angular_speed_rad_s: float = DEFAULT_MAX_ANGULAR_SPEED_RAD_S,
    max_tilt_rad: float | None = None,
) -> dict[str, Any]:
    """Require a complete 10/10 canonical/support/contact stability window."""

    if required_samples <= 0:
        raise ValueError("required stability sample count must be positive")
    if max_support_gap_m < 0.0 or max_support_penetration_m < 0.0:
        raise ValueError("support gap tolerances must be non-negative")
    if max_linear_speed_m_s < 0.0 or max_angular_speed_rad_s < 0.0:
        raise ValueError("velocity tolerances must be non-negative")
    if max_tilt_rad is not None and max_tilt_rad < 0.0:
        raise ValueError("tilt tolerance must be non-negative")

    window = tuple(samples[-required_samples:])
    complete = len(window) == required_samples
    relation_window = [sample.relation for sample in window]
    support_window = [sample.support_on_top for sample in window]
    touching_window = [sample.support_touching for sample in window]
    gaps = [sample.bottom_to_support_gap_m for sample in window]
    linear_speeds = [sample.linear_speed_m_s for sample in window]
    angular_speeds = [sample.angular_speed_rad_s for sample in window]
    tilts = [sample.tilt_rad for sample in window]
    bottom_heights = [sample.bottom_z_m for sample in window]

    failure_reasons: list[str] = []
    finite_samples = all(
        np.isfinite(
            [
                sample.bottom_to_support_gap_m,
                sample.linear_speed_m_s,
                sample.angular_speed_rad_s,
                sample.tilt_rad,
                *sample.position,
                *sample.orientation_xyzw,
            ]
        ).all()
        for sample in window
    )
    if not complete:
        failure_reasons.append("incomplete_stability_window")
    if not finite_samples:
        failure_reasons.append("non_finite_stability_sample")
    if not relation_window or not all(relation_window):
        failure_reasons.append("canonical_relation_not_stable")
    if not support_window or not all(support_window):
        failure_reasons.append("canonical_support_not_stable")
    if not touching_window or not all(touching_window):
        failure_reasons.append("support_contact_not_stable")
    if not gaps or not all(
        -max_support_penetration_m <= gap <= max_support_gap_m for gap in gaps
    ):
        failure_reasons.append("support_gap_out_of_bounds")
    if not linear_speeds or not all(
        speed <= max_linear_speed_m_s for speed in linear_speeds
    ):
        failure_reasons.append("linear_velocity_not_stable")
    if not angular_speeds or not all(
        speed <= max_angular_speed_rad_s for speed in angular_speeds
    ):
        failure_reasons.append("angular_velocity_not_stable")
    if max_tilt_rad is not None and (
        not tilts or not all(tilt <= max_tilt_rad for tilt in tilts)
    ):
        failure_reasons.append("tilt_out_of_bounds")

    return {
        "accepted": not failure_reasons,
        "settle_steps_required": DEFAULT_SETTLE_STEPS,
        "required_samples": required_samples,
        "observed_samples": len(window),
        "canonical_relation_window": relation_window,
        "canonical_support_window": support_window,
        "support_touching_window": touching_window,
        "bottom_to_support_gap_m": gaps,
        "linear_speed_m_s": linear_speeds,
        "angular_speed_rad_s": angular_speeds,
        "tilt_rad": tilts,
        "bottom_z_m": bottom_heights,
        "max_support_gap_m": max_support_gap_m,
        "max_support_penetration_m": max_support_penetration_m,
        "max_linear_speed_m_s": max_linear_speed_m_s,
        "max_angular_speed_rad_s": max_angular_speed_rad_s,
        "max_tilt_rad": max_tilt_rad,
        "failure_reasons": failure_reasons,
        "final_sample_private": (
            window[-1].to_private_dict() if window else None
        ),
    }


def summarize_symbolic_spatial_postcondition(
    samples: Sequence[SpatialStabilitySample],
    *,
    required_samples: int = DEFAULT_SYMBOLIC_POSTCONDITION_WINDOW,
    max_support_gap_m: float = DEFAULT_MAX_SUPPORT_GAP_M,
    max_support_penetration_m: float = DEFAULT_MAX_SUPPORT_PENETRATION_M,
) -> dict[str, Any]:
    """Validate a high-level Symbolic placement without gating on residual speed.

    The official Symbolic ``PLACE_ON_TOP`` primitive settles and then checks its
    canonical predicate; it does not require every sample in a physical velocity
    window to be below a controller threshold.  Our restricted Symbolic spatial
    actions additionally retain live support, contact, and gap checks for the
    final three samples.  Linear and angular speeds remain in the report as
    diagnostics, but do not reject an otherwise valid semantic placement.
    """

    report = summarize_spatial_stability(
        samples,
        required_samples=required_samples,
        max_support_gap_m=max_support_gap_m,
        max_support_penetration_m=max_support_penetration_m,
        max_linear_speed_m_s=float("inf"),
        max_angular_speed_rad_s=float("inf"),
    )
    report.update(
        {
            "verification_policy": (
                "behavior2026.symbolic-spatial-final-postcondition/v1"
            ),
            "velocity_diagnostic_only": True,
            "max_linear_speed_m_s": None,
            "max_angular_speed_rad_s": None,
        }
    )
    return report


def summarize_relation_authoritative_postcondition(
    samples: Sequence[SpatialStabilitySample],
    *,
    required_samples: int = DEFAULT_SYMBOLIC_POSTCONDITION_WINDOW,
    require_geometric_support: bool = False,
) -> dict[str, Any]:
    """Use the selected mode's relation as the commit gate.

    Symbolic NextTo uses AABB geometry; Physical retains native adjacency.
    Requiring a separate velocity envelope and simultaneous ``OnTop`` plus
    ``Touching`` window can reject a pose whose relation is already true.
    Keep those physical observations in private
    diagnostics, but commit only when the canonical relation is true for the
    complete final window and all samples are finite. Zero-physics callers
    can require geometric support without requiring PhysX Touching or speed.
    """

    report = summarize_spatial_stability(
        samples,
        required_samples=required_samples,
        max_linear_speed_m_s=float("inf"),
        max_angular_speed_rad_s=float("inf"),
    )
    diagnostic_failure_reasons = list(report["failure_reasons"])
    authoritative_reasons = [
        reason
        for reason in diagnostic_failure_reasons
        if reason
        in {
            "incomplete_stability_window",
            "non_finite_stability_sample",
            "canonical_relation_not_stable",
        }
        or (require_geometric_support and reason == "canonical_support_not_stable")
    ]
    report.update(
        {
            "accepted": not authoritative_reasons,
            "failure_reasons": authoritative_reasons,
            "diagnostic_failure_reasons": diagnostic_failure_reasons,
            "verification_policy": (
                "behavior2026.official-relation-authoritative/v1"
            ),
            "official_relation_authoritative": True,
            "support_and_velocity_diagnostic_only": not require_geometric_support,
            "geometric_support_required": require_geometric_support,
            "max_linear_speed_m_s": None,
            "max_angular_speed_rad_s": None,
        }
    )
    return report


def summarize_supported_lateral_postcondition(
    samples: Sequence[SpatialStabilitySample],
    *,
    expected_support_bottom_z_m: float | None,
    expected_candidate_xy: Sequence[float] | None,
    room_membership_window: Sequence[bool | None] | None,
    post_settle_support_valid: bool | None,
    required_samples: int = DEFAULT_SYMBOLIC_POSTCONDITION_WINDOW,
    gross_fall_tolerance_m: float = DEFAULT_SUPPORTED_LATERAL_GROSS_FALL_M,
    gross_planar_escape_m: float = (
        DEFAULT_SUPPORTED_LATERAL_GROSS_PLANAR_ESCAPE_M
    ),
    max_linear_speed_m_s: float = (
        DEFAULT_SUPPORTED_LATERAL_MAX_LINEAR_SPEED_M_S
    ),
    max_angular_speed_rad_s: float = (
        DEFAULT_SUPPORTED_LATERAL_MAX_ANGULAR_SPEED_RAD_S
    ),
) -> dict[str, Any]:
    """Require official lateral truth plus broad live support integrity.

    Canonical ``NextTo`` / ``Touching`` remains mandatory for the requested
    relation.  It is not by itself proof that a released object remains in
    the scene: an adjacency ray may stay true while the object free-falls far
    below a large reference.  Add only metre-scale, task-neutral trajectory
    invariants: finite pose, no gross fall or planar / room escape, no obvious
    high-speed motion, and a valid five-ray local support after settling.

    These are deliberately not centimetre-level placement thresholds.  No
    task, asset, object, category, model, or room name is inspected.
    """

    for name, value in (
        ("gross_fall_tolerance_m", gross_fall_tolerance_m),
        ("gross_planar_escape_m", gross_planar_escape_m),
        ("max_linear_speed_m_s", max_linear_speed_m_s),
        ("max_angular_speed_rad_s", max_angular_speed_rad_s),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    report = summarize_relation_authoritative_postcondition(
        samples,
        required_samples=required_samples,
    )
    window = tuple(samples[-required_samples:])
    reasons = list(report["failure_reasons"])

    expected_bottom = (
        None
        if expected_support_bottom_z_m is None
        else float(expected_support_bottom_z_m)
    )
    if expected_bottom is None or not np.isfinite(expected_bottom):
        reasons.append("expected_local_support_height_unavailable")
    bottom_deltas: list[float | None] = []
    for sample in window:
        bottom = sample.bottom_z_m
        if bottom is None or not np.isfinite(bottom) or expected_bottom is None:
            bottom_deltas.append(None)
        else:
            bottom_deltas.append(float(bottom) - expected_bottom)
    if (
        len(bottom_deltas) != required_samples
        or any(value is None for value in bottom_deltas)
    ):
        reasons.append("live_bottom_height_unavailable")
    elif any(
        float(value) < -gross_fall_tolerance_m
        for value in bottom_deltas
        if value is not None
    ):
        reasons.append("gross_fall_below_local_support")

    expected_xy = None
    if expected_candidate_xy is not None:
        vector = np.asarray(expected_candidate_xy, dtype=np.float64).reshape(-1)
        if vector.size >= 2 and np.isfinite(vector[:2]).all():
            expected_xy = vector[:2]
    planar_drift: list[float | None] = []
    for sample in window:
        if expected_xy is None:
            planar_drift.append(None)
            continue
        position = np.asarray(sample.position, dtype=np.float64)
        planar_drift.append(float(np.linalg.norm(position[:2] - expected_xy)))
    if expected_xy is None:
        reasons.append("expected_candidate_xy_unavailable")
    elif (
        len(planar_drift) != required_samples
        or any(value is None or not np.isfinite(value) for value in planar_drift)
    ):
        reasons.append("live_planar_position_unavailable")
    elif any(
        float(value) > gross_planar_escape_m
        for value in planar_drift
        if value is not None
    ):
        reasons.append("gross_planar_escape")

    normalized_rooms = (
        None
        if room_membership_window is None
        else tuple(room_membership_window[-required_samples:])
    )
    if (
        normalized_rooms is None
        or len(normalized_rooms) != required_samples
        or any(value is None for value in normalized_rooms)
    ):
        reasons.append("post_settle_room_evidence_unavailable")
    elif not all(bool(value) for value in normalized_rooms):
        reasons.append("post_settle_room_escape")

    linear_speeds = [sample.linear_speed_m_s for sample in window]
    angular_speeds = [sample.angular_speed_rad_s for sample in window]
    if not linear_speeds or not all(
        np.isfinite(value) and value <= max_linear_speed_m_s
        for value in linear_speeds
    ):
        reasons.append("gross_linear_velocity")
    if not angular_speeds or not all(
        np.isfinite(value) and value <= max_angular_speed_rad_s
        for value in angular_speeds
    ):
        reasons.append("gross_angular_velocity")

    if post_settle_support_valid is None:
        reasons.append("post_settle_local_support_unavailable")
    elif not post_settle_support_valid:
        reasons.append("post_settle_local_support_invalid")

    reasons = list(dict.fromkeys(reasons))
    report.update(
        {
            "accepted": not reasons,
            "failure_reasons": reasons,
            "verification_policy": (
                "behavior2026.official-relation-plus-supported-lateral/v2"
            ),
            "official_relation_authoritative": True,
            "supported_lateral_commit_required": True,
            "support_and_velocity_diagnostic_only": False,
            "expected_support_bottom_z_m": expected_bottom,
            "bottom_delta_from_local_support_m": bottom_deltas,
            "expected_candidate_xy": (
                None if expected_xy is None else expected_xy.tolist()
            ),
            "planar_drift_from_candidate_m": planar_drift,
            "room_membership_window": (
                None if normalized_rooms is None else list(normalized_rooms)
            ),
            "post_settle_support_valid": post_settle_support_valid,
            "gross_fall_tolerance_m": gross_fall_tolerance_m,
            "gross_planar_escape_m": gross_planar_escape_m,
            "max_linear_speed_m_s": max_linear_speed_m_s,
            "max_angular_speed_rad_s": max_angular_speed_rad_s,
        }
    )
    return report


def _private_jsonable(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy_value = getattr(value, "numpy", None)
        if callable(numpy_value):
            value = numpy_value()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _private_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_private_jsonable(item) for item in value]
    return value


def build_spatial_partial_report(
    *,
    operation: str,
    candidate_plan: SpatialCandidatePlan | None,
    candidate_attempts: Sequence[Mapping[str, Any]],
    selected_candidate: str | None = None,
    selected_support: Any | None = None,
    stability_report: Mapping[str, Any] | None = None,
    error: BaseException | str | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe report that retains every failed candidate.

    The executor should publish this report from both success and exception
    paths.  Keeping planning admissions separate from settled attempts makes
    blocker, ray-probe, predicate, support, and physics failures distinguishable.
    """

    error_text = None if error is None else str(error)
    plan_report = (
        candidate_plan.to_private_dict()
        if candidate_plan is not None
        else {
            "candidate_count": 0,
            "admitted_candidate_count": 0,
            "planning_error": None,
            "candidate_admission_private": [],
        }
    )
    return {
        "operation": operation,
        "executor_version": SPATIAL_SYMBOLIC_ACTION_VERSION,
        "partial_report": error is not None or selected_candidate is None,
        **plan_report,
        "candidate_attempts_private": _private_jsonable(candidate_attempts),
        "selected_candidate_private": selected_candidate,
        "selected_support_private": (
            str(getattr(selected_support, "name", "")) or None
            if selected_support is not None
            else None
        ),
        "stability_private": _private_jsonable(stability_report),
        "error": error_text,
        "error_type": (
            type(error).__name__ if isinstance(error, BaseException) else None
        ),
    }


__all__ = [
    "CandidateAdmissionReport",
    "CandidateCollisionQueryBox",
    "CollisionOverlapAdmissionReport",
    "CollisionOverlapHitEvidence",
    "DEFAULT_FORBIDDEN_LOCAL_SUPPORT_ENTITY_ROLES",
    "DEFAULT_LATERAL_ORDER",
    "DEFAULT_LOCAL_SUPPORT_COPLANAR_TOLERANCE_M",
    "DEFAULT_LOCAL_SUPPORT_MIN_CONSISTENT_HITS",
    "DEFAULT_LOCAL_SUPPORT_NORMAL_Z_MIN",
    "DEFAULT_COLLISION_QUERY_Z_INSET_M",
    "DEFAULT_SOLID_OVERLAP_TOLERANCE_M",
    "DEFAULT_MAX_ANGULAR_SPEED_RAD_S",
    "DEFAULT_MAX_LINEAR_SPEED_M_S",
    "DEFAULT_MAX_SUPPORT_GAP_M",
    "DEFAULT_MAX_SUPPORT_PENETRATION_M",
    "DEFAULT_SUPPORTED_LATERAL_GROSS_FALL_M",
    "DEFAULT_SUPPORTED_LATERAL_GROSS_PLANAR_ESCAPE_M",
    "DEFAULT_SUPPORTED_LATERAL_MAX_ANGULAR_SPEED_RAD_S",
    "DEFAULT_SUPPORTED_LATERAL_MAX_LINEAR_SPEED_M_S",
    "DEFAULT_SETTLE_STEPS",
    "DEFAULT_STABILITY_WINDOW",
    "DEFAULT_SYMBOLIC_POSTCONDITION_WINDOW",
    "DownwardSupportRayEvidence",
    "DownwardSupportRayRequest",
    "LOCAL_SUPPORT_RAY_SAMPLE_NAMES",
    "LocalSupportRebaseReport",
    "SIMPLE_NEXT_TO_SCREEN_SELECTORS",
    "SIMPLE_TOUCHING_SCREEN_SELECTORS",
    "SPATIAL_ACTION_TO_PREDICATE",
    "SPATIAL_SYMBOLIC_ACTION_VERSION",
    "SpatialCandidatePlan",
    "SpatialPoseCandidate",
    "SpatialStabilitySample",
    "UnderSupportPlan",
    "VerticalPlacementEvidence",
    "build_downward_support_ray_requests",
    "build_candidate_collision_query_box",
    "build_spatial_partial_report",
    "candidate_aabb",
    "candidate_blockers",
    "canonical_relation",
    "default_lateral_candidates",
    "expanded_lateral_candidates",
    "expanded_support_agnostic_lateral_candidates",
    "default_under_candidates",
    "infer_unique_on_top_support",
    "next_to_gap_for_extents",
    "symbolic_next_to_relation",
    "object_aabb",
    "object_bottom_offset_z",
    "object_pose",
    "pair_penetration_depth",
    "plan_default_on_top_candidates",
    "plan_default_under_candidates",
    "plan_default_under_supports",
    "quaternion_tilt_rad",
    "rank_default_under_supports",
    "rebase_candidate_to_local_support",
    "sample_spatial_stability",
    "select_next_to_screen_candidate",
    "select_touching_screen_candidate",
    "shelf_screen_cell_aabb",
    "spatial_exact_step_budget",
    "state_by_name",
    "summarize_spatial_stability",
    "summarize_relation_authoritative_postcondition",
    "summarize_supported_lateral_postcondition",
    "summarize_candidate_collision_overlap",
    "summarize_symbolic_spatial_postcondition",
    "symbolic_on_top_relation",
]
