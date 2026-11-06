"""Shared, task-neutral admission policy for symbolic spatial candidates.

Candidate generation and canonical predicate readback remain owned by the
existing spatial kernel.  This module only combines two kinds of collision
evidence without letting an imprecise whole-object AABB veto an otherwise
clear live collision-shape query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.symbolic_contact_geometry import (
    target_solid_intersects_query_box,
)

from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    CollisionOverlapAdmissionReport,
    CollisionOverlapHitEvidence,
    DEFAULT_BLOCKER_MARGIN_M,
    DEFAULT_COLLISION_QUERY_Z_INSET_M,
    DEFAULT_SOLID_OVERLAP_TOLERANCE_M,
    LocalSupportRebaseReport,
    SpatialPoseCandidate,
    build_candidate_collision_query_box,
    object_pose,
    summarize_candidate_collision_overlap,
)


ADMISSION_POLICY_VERSION = (
    "behavior2026.symbolic-spatial-candidate-admission/v2"
)
FINAL_PLACEMENT_COLLISION_POLICY_VERSION = (
    "behavior2026.symbolic-placement-final-collision/v2"
)


def spatial_object_link_paths(obj: Any) -> tuple[str, ...]:
    """Return the live collision identities owned by one object."""

    paths = getattr(obj, "link_prim_paths", ())
    paths = paths() if callable(paths) else paths
    if paths:
        return tuple(str(path) for path in paths)
    links = getattr(obj, "links", {}) or {}
    linked = tuple(
        str(getattr(link, "prim_path", ""))
        for link in links.values()
        if str(getattr(link, "prim_path", ""))
    )
    if linked:
        return linked
    prim_path = str(getattr(obj, "prim_path", ""))
    return (prim_path,) if prim_path else ()


def query_spatial_candidate_collision_overlap(
    *,
    omnigibson: Any,
    held: Any,
    closure_objects: Sequence[Any],
    candidate: SpatialPoseCandidate,
    geometry_certified_objects: Sequence[Any] = (),
    contact_objects: Sequence[Any] = (),
    candidate_aabb_world: tuple[Any, Any] | None = None,
    xy_margin_m: float = DEFAULT_BLOCKER_MARGIN_M,
    xy_inset_m: float = 0.0,
    z_inset_m: float = DEFAULT_COLLISION_QUERY_Z_INSET_M,
) -> CollisionOverlapAdmissionReport:
    """Read collision overlap for a future pose without applying that pose.

    Target contact candidates require a live surface check before a coarse
    hull hit can be excluded. An AABB/center predicate alone is not a certificate.
    ``geometry_certified_objects`` retains the separate Physical lane's policy;
    Symbolic callers supply ``contact_objects`` instead.
    """

    query_box = None
    raw_hits: list[CollisionOverlapHitEvidence] = []
    query_error = None
    excluded_paths = tuple(
        path
        for item in (*closure_objects, *geometry_certified_objects)
        if item is not None
        for path in spatial_object_link_paths(item)
    )
    try:
        query_box = build_candidate_collision_query_box(
            held,
            candidate,
            candidate_aabb_world=candidate_aabb_world,
            xy_margin_m=xy_margin_m,
            xy_inset_m=xy_inset_m,
            z_inset_m=z_inset_m,
        )

        def overlap_callback(hit: Any) -> bool:
            raw_hits.append(
                CollisionOverlapHitEvidence(
                    rigid_body_path=getattr(hit, "rigid_body", None),
                    collision_path=getattr(hit, "collision", None),
                )
            )
            return True

        omnigibson.sim.psqi.overlap_box(
            halfExtent=query_box.half_extent_xyz.tolist(),
            pos=query_box.center_xyz.tolist(),
            rot=query_box.rotation_xyzw.tolist(),
            reportFn=overlap_callback,
        )
        for target in contact_objects:
            if target is None:
                continue
            paths = spatial_object_link_paths(target)
            target_hit = any(
                value is not None and (str(value) == path or str(value).startswith(path + "/"))
                for hit in raw_hits
                for value in (hit.rigid_body_path, hit.collision_path)
                for path in paths
            )
            if target_hit and not target_solid_intersects_query_box(target, query_box):
                excluded_paths += paths
    except Exception as exc:
        query_error = exc
    return summarize_candidate_collision_overlap(
        query_box=query_box,
        raw_hits=raw_hits,
        excluded_identity_paths=excluded_paths,
        query_error=query_error,
    )


@dataclass(frozen=True, slots=True)
class FinalPlacementCollisionReport:
    accepted: bool
    solid_overlap_tolerance_m: float
    object_reports: tuple[
        tuple[str, CollisionOverlapAdmissionReport], ...
    ]

    def to_private_dict(self) -> dict[str, object]:
        return {
            "version": FINAL_PLACEMENT_COLLISION_POLICY_VERSION,
            "accepted": self.accepted,
            "solid_overlap_tolerance_m": self.solid_overlap_tolerance_m,
            "moving_object_count": len(self.object_reports),
            "moving_object_collision_reports_private": [
                {
                    "object_private": name,
                    **report.to_private_dict(),
                }
                for name, report in self.object_reports
            ],
        }


def query_symbolic_placement_final_collision(
    *,
    omnigibson: Any,
    moving_objects: Sequence[Any],
    contact_objects: Sequence[Any] = (),
    solid_overlap_tolerance_m: float = (
        DEFAULT_SOLID_OVERLAP_TOLERANCE_M
    ),
) -> FinalPlacementCollisionReport:
    """Fail closed when a zero-physics final placement penetrates a solid."""

    tolerance = float(solid_overlap_tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("solid overlap tolerance must be finite and non-negative")
    moving_list: list[Any] = []
    seen_ids: set[int] = set()
    for obj in moving_objects:
        if obj is None or id(obj) in seen_ids:
            continue
        seen_ids.add(id(obj))
        moving_list.append(obj)
    moving = tuple(moving_list)
    if not moving:
        raise ValueError("moving_objects must not be empty")

    reports: list[tuple[str, CollisionOverlapAdmissionReport]] = []
    for index, obj in enumerate(moving):
        name = str(getattr(obj, "name", "")) or f"moving_object_{index}"
        try:
            position, orientation = object_pose(obj)
            candidate = SpatialPoseCandidate(
                name=f"{name}:final_symbolic_pose",
                position=position,
                orientation_xyzw=orientation,
            )
            report = query_spatial_candidate_collision_overlap(
                omnigibson=omnigibson,
                held=obj,
                closure_objects=moving,
                contact_objects=contact_objects,
                candidate=candidate,
                xy_margin_m=0.0,
                xy_inset_m=tolerance,
                z_inset_m=tolerance,
            )
        except Exception as exc:
            report = CollisionOverlapAdmissionReport(
                accepted=False,
                reason="collision_query_error",
                query_box=None,
                raw_hits=(),
                excluded_hits=(),
                blocking_hits=(),
                query_error=f"{type(exc).__name__}: {exc}",
            )
        reports.append((name, report))
    return FinalPlacementCollisionReport(
        accepted=all(report.accepted for _name, report in reports),
        solid_overlap_tolerance_m=tolerance,
        object_reports=tuple(reports),
    )


@dataclass(frozen=True, slots=True)
class SpatialCandidateAdmissionDecision:
    accepted_for_execution: bool
    reason: str
    candidate: SpatialPoseCandidate | None
    broad_blockers: tuple[str, ...]
    local_support_required: bool
    local_support_accepted: bool | None
    collision_shape_accepted: bool

    def to_private_dict(self) -> dict[str, object]:
        return {
            "version": ADMISSION_POLICY_VERSION,
            "accepted_for_execution": self.accepted_for_execution,
            "reason": self.reason,
            "candidate": None if self.candidate is None else self.candidate.name,
            "broad_blockers_private": list(self.broad_blockers),
            "local_support_required": self.local_support_required,
            "local_support_accepted": self.local_support_accepted,
            "collision_shape_accepted": self.collision_shape_accepted,
            "official_relation_declared": False,
        }


@dataclass(frozen=True, slots=True)
class SpatialCandidateExecutionResult:
    """One fail-closed narrow-query transaction for a candidate."""

    decision: SpatialCandidateAdmissionDecision
    collision_shape_report: CollisionOverlapAdmissionReport
    applied: bool


def decide_spatial_candidate_admission(
    *,
    candidate: SpatialPoseCandidate,
    broad_blockers: Sequence[str],
    collision_shape_report: CollisionOverlapAdmissionReport,
    local_support_report: LocalSupportRebaseReport | None = None,
    local_support_required: bool = False,
) -> SpatialCandidateAdmissionDecision:
    """Return whether a candidate may be executed and canonically read back.

    Broad object AABBs are diagnostic only.  Actual PhysX collision-shape
    overlap is authoritative for collision admission.  A relation that needs
    candidate-local support must additionally supply an accepted five-ray
    support report.  No task, object, category, model, or asset identity is
    inspected here.
    """

    if not isinstance(candidate, SpatialPoseCandidate):
        raise TypeError("candidate must be a SpatialPoseCandidate")
    broad = tuple(sorted({str(value) for value in broad_blockers}))
    support_accepted = (
        None if local_support_report is None else bool(local_support_report.accepted)
    )
    effective_candidate = candidate
    if local_support_required:
        if local_support_report is None:
            return SpatialCandidateAdmissionDecision(
                accepted_for_execution=False,
                reason="local_support_evidence_required",
                candidate=None,
                broad_blockers=broad,
                local_support_required=True,
                local_support_accepted=None,
                collision_shape_accepted=bool(collision_shape_report.accepted),
            )
        if not local_support_report.accepted:
            return SpatialCandidateAdmissionDecision(
                accepted_for_execution=False,
                reason="local_support_rejected",
                candidate=None,
                broad_blockers=broad,
                local_support_required=True,
                local_support_accepted=False,
                collision_shape_accepted=bool(collision_shape_report.accepted),
            )
        effective_candidate = local_support_report.rebased_candidate
        if effective_candidate is None:
            raise RuntimeError("accepted local support report lacks a candidate")
    if not collision_shape_report.accepted:
        return SpatialCandidateAdmissionDecision(
            accepted_for_execution=False,
            reason="collision_shape_overlap_rejected",
            candidate=None,
            broad_blockers=broad,
            local_support_required=local_support_required,
            local_support_accepted=support_accepted,
            collision_shape_accepted=False,
        )
    return SpatialCandidateAdmissionDecision(
        accepted_for_execution=True,
        reason="collision_shape_clear",
        candidate=effective_candidate,
        broad_blockers=broad,
        local_support_required=local_support_required,
        local_support_accepted=support_accepted,
        collision_shape_accepted=True,
    )


def execute_admitted_spatial_candidate(
    *,
    candidate: SpatialPoseCandidate,
    broad_blockers: Sequence[str],
    collision_shape_query: Callable[
        [SpatialPoseCandidate], CollisionOverlapAdmissionReport
    ],
    apply_candidate: Callable[[SpatialPoseCandidate], None],
    local_support_report: LocalSupportRebaseReport | None = None,
    local_support_required: bool = False,
) -> SpatialCandidateExecutionResult:
    """Run narrow collision admission before applying one candidate.

    When local support is required, the collision query receives only the
    rebased candidate.  A missing or rejected support report prevents both
    the query and pose application.  Query exceptions and malformed report
    types propagate before ``apply_candidate`` and therefore fail closed.
    """

    effective_candidate = candidate
    if local_support_required:
        if local_support_report is None or not local_support_report.accepted:
            not_queried = CollisionOverlapAdmissionReport(
                accepted=False,
                reason="collision_query_not_run_without_local_support",
                query_box=None,
                raw_hits=(),
                excluded_hits=(),
                blocking_hits=(),
                query_error=None,
            )
            decision = decide_spatial_candidate_admission(
                candidate=candidate,
                broad_blockers=broad_blockers,
                collision_shape_report=not_queried,
                local_support_report=local_support_report,
                local_support_required=True,
            )
            return SpatialCandidateExecutionResult(
                decision=decision,
                collision_shape_report=not_queried,
                applied=False,
            )
        effective_candidate = local_support_report.rebased_candidate
        if effective_candidate is None:
            raise RuntimeError("accepted local support report lacks a candidate")

    collision_shape_report = collision_shape_query(effective_candidate)
    if not isinstance(collision_shape_report, CollisionOverlapAdmissionReport):
        raise TypeError(
            "collision_shape_query must return CollisionOverlapAdmissionReport"
        )
    decision = decide_spatial_candidate_admission(
        candidate=candidate,
        broad_blockers=broad_blockers,
        collision_shape_report=collision_shape_report,
        local_support_report=local_support_report,
        local_support_required=local_support_required,
    )
    if decision.accepted_for_execution:
        if decision.candidate is None:
            raise RuntimeError("accepted spatial admission lacks a candidate")
        apply_candidate(decision.candidate)
        return SpatialCandidateExecutionResult(
            decision=decision,
            collision_shape_report=collision_shape_report,
            applied=True,
        )
    return SpatialCandidateExecutionResult(
        decision=decision,
        collision_shape_report=collision_shape_report,
        applied=False,
    )


def first_collision_free_approach_candidate(
    *,
    start: SpatialPoseCandidate,
    target: SpatialPoseCandidate,
    collision_shape_query: Callable[
        [SpatialPoseCandidate], CollisionOverlapAdmissionReport
    ],
    sample_count: int = 12,
) -> tuple[
    SpatialPoseCandidate | None,
    tuple[CollisionOverlapAdmissionReport, ...],
]:
    """Return the first clear release pose between the gripper and target."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    reports: list[CollisionOverlapAdmissionReport] = []
    for index in range(sample_count + 1):
        fraction = index / sample_count
        candidate = SpatialPoseCandidate(
            name=f"{target.name}:collision_free_approach_{index}",
            position=(
                np.asarray(start.position, dtype=np.float64)
                + fraction
                * (
                    np.asarray(target.position, dtype=np.float64)
                    - np.asarray(start.position, dtype=np.float64)
                )
            ),
            # A failed placement should approach with the already-held pose;
            # rotating inside the target would create a second collision path.
            orientation_xyzw=np.asarray(
                start.orientation_xyzw, dtype=np.float64
            ).copy(),
        )
        report = collision_shape_query(candidate)
        if not isinstance(report, CollisionOverlapAdmissionReport):
            raise TypeError(
                "collision_shape_query must return CollisionOverlapAdmissionReport"
            )
        reports.append(report)
        if index > 0 and report.accepted:
            return candidate, tuple(reports)
    return None, tuple(reports)


__all__ = [
    "ADMISSION_POLICY_VERSION",
    "FINAL_PLACEMENT_COLLISION_POLICY_VERSION",
    "FinalPlacementCollisionReport",
    "SpatialCandidateAdmissionDecision",
    "SpatialCandidateExecutionResult",
    "decide_spatial_candidate_admission",
    "execute_admitted_spatial_candidate",
    "first_collision_free_approach_candidate",
    "query_spatial_candidate_collision_overlap",
    "query_symbolic_placement_final_collision",
    "spatial_object_link_paths",
]
