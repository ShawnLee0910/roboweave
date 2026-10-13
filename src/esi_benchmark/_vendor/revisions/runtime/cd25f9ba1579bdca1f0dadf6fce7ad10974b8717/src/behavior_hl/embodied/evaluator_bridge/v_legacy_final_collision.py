"""Frozen R1/R3 intervention final collision query from f099876.

All seven historical V Runtime revisions share these exact three definitions.
Only legacy R1/R3 intervention code imports this compatibility module; normal
Action20 placement and explicit new R4 retain the mainline final collision gate.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Sequence
import numpy as np
from .spatial_candidate_admission import query_spatial_candidate_collision_overlap
from .spatial_symbolic_actions import (CollisionOverlapAdmissionReport,
    DEFAULT_SOLID_OVERLAP_TOLERANCE_M, SpatialPoseCandidate, object_pose)
FINAL_PLACEMENT_COLLISION_POLICY_VERSION = "behavior2026.symbolic-placement-final-collision/v3-component-refinement"

@dataclass(frozen=True, slots=True)
class FinalPlacementCollisionReport:
    accepted: bool
    solid_overlap_tolerance_m: float
    object_reports: tuple[
        tuple[str, CollisionOverlapAdmissionReport], ...
    ]
    component_refinements: tuple[dict[str, Any], ...] = ()

    def to_private_dict(self) -> dict[str, object]:
        return {
            "version": FINAL_PLACEMENT_COLLISION_POLICY_VERSION,
            "accepted": self.accepted,
            "solid_overlap_tolerance_m": self.solid_overlap_tolerance_m,
            "moving_object_count": len(self.object_reports),
            "component_refinements_private": list(self.component_refinements),
            # Keep decisive evidence shallow enough for bounded private JSON.
            "blocking_bodies_private": sorted({hit.rigid_body_path or hit.collision_path or "UNKNOWN"
                for _name, report in self.object_reports for hit in report.blocking_hits}),
            "component_refinement_status_private": [r["status"] for r in self.component_refinements],
            "moving_object_collision_reports_private": [
                {
                    "object_private": name,
                    **report.to_private_dict(),
                }
                for name, report in self.object_reports
            ],
        }


def _refine_final_object_overlap(omnigibson, obj, report):
    """Query current collision parts only after an inset AABB was blocked.

    Uses the same native API as OG's robot collision queries. PhysX cooks a
    convex approximation per mesh, so this remains conservative for concave
    individual parts; it does not fill all space in the object's whole AABB.
    No pose update, physics step, target exemption, or tolerance change.
    Missing/invalid queries retain the original refusal, never grant admission.
    """
    detail = {"object_private": str(getattr(obj, "name", "")),
              "method": "native_collision_parts_at_current_pose",
              "mesh_approximation": "physx_convex_per_authored_collision_part",
              "queried_parts": [], "confirmed_blockers": [],
              "status": "NOT_NEEDED"}
    if report.accepted or report.query_error or report.reason != "blocking_collision_shape_overlap":
        return report, detail
    try:
        from pxr import PhysicsSchemaTools
        parts = [mesh for link in obj.links.values() for mesh in link.collision_meshes.values()]
        if not parts:
            raise ValueError("moving object collision parts unavailable")
        hits = []
        for mesh in parts:
            prim = mesh.prim
            if not prim.IsValid():
                raise ValueError("invalid moving collision prim")
            path = str(prim.GetPath())
            if not path.startswith("/") or path == "/" or path in detail["queried_parts"]:
                raise ValueError("missing or duplicate collision prim identity")
            kind = prim.GetTypeName()
            if kind not in {"Mesh", "Cube", "Sphere", "Capsule", "Cylinder", "Cone"}:
                raise ValueError("unsupported moving collision prim type: " + str(kind))
            method = "overlap_mesh" if kind == "Mesh" else "overlap_shape"
            query = getattr(omnigibson.sim.psqi, method, None)
            if not callable(query):
                raise ValueError("native component overlap query unavailable")
            part_hits = []
            invalid_hits = []

            def collect(hit):
                body, shape = getattr(hit, "rigid_body", None), getattr(hit, "collision", None)
                identities = [p for p in (body, shape) if p is not None]
                if not identities or any(not isinstance(p, str) or not p.startswith("/") or p == "/"
                                         for p in identities):
                    invalid_hits.append(True)
                    return True
                part_hits.append((body, shape))
                return True

            count = query(*PhysicsSchemaTools.encodeSdfPath(path), reportFn=collect)
            if invalid_hits or type(count) is not int or count < 0 or count != len(part_hits):
                raise ValueError("component query hit count is inconsistent")
            hits.extend(part_hits)
            detail["queried_parts"].append(path)

        def confirmed(hit):
            # Compare rigid bodies when available: another shape on the same
            # body must not be silently cleared by shape-path differences.
            return any((hit.rigid_body_path is not None and
                        (body == hit.rigid_body_path or
                         (isinstance(shape, str) and shape.startswith(hit.rigid_body_path + "/"))))
                       or (hit.collision_path is not None and
                           (shape == hit.collision_path or
                            (isinstance(body, str) and hit.collision_path.startswith(body + "/"))))
                       for body, shape in hits)

        remaining = tuple(hit for hit in report.blocking_hits if confirmed(hit))
        cleared = tuple(hit for hit in report.blocking_hits if not confirmed(hit))
        detail.update(status="CONFIRMED_BLOCKED" if remaining else "CLEAR",
                      confirmed_blockers=[hit.to_private_dict() for hit in remaining],
                      cleared_box_hit_count=len(cleared))
        return replace(report, accepted=not remaining,
                       reason="blocking_collision_shape_overlap" if remaining else "clear_component_overlap",
                       excluded_hits=(*report.excluded_hits, *cleared), blocking_hits=remaining), detail
    except Exception as exc:
        detail.update(status="UNAVAILABLE_RETAINED_REJECTION", error=f"{type(exc).__name__}: {exc}")
        return report, detail


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
    refinements = []
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
            report, refinement = _refine_final_object_overlap(omnigibson, obj, report)
            refinements.append(refinement)
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
        component_refinements=tuple(refinements),
    )
