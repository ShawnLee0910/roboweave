"""Read-only placement hints before pick, never an Action admission gate.

These finite broad-geometry proposals neither prove availability nor prove
infeasibility.  Only a live public placement action performs actual admission.
"""

from __future__ import annotations

from typing import Any, Sequence

from .container_packing_order import live_object_extent_m
from .deterministic_inside_placement import Aabb3, plan_symbolic_inside_candidates
from .on_top_support_layers import (
    plan_on_top_support_layer_candidates, restrict_and_rank_on_top_plan,
    on_top_binding_horizontal_distance,
)
from .private_visual_markers import SpatialMarkerTargetToken


def reachable_on_top_surface_markers(
    bindings, *, held, base_xy, maximum_distance_m,
    preferred_orientation_xyzw=None, include_support_center=True, physical=False,
):
    """Use Action20's selected-region proposals and exact workspace kernel.

    This is a station-local reach check, not collision admission or task
    infeasibility. Return all plans too, for the same controller's next station.
    No Marker is consumed and no simulator state is changed.
    """
    if held is None:
        raise RuntimeError("surface workspace selection requires held object")
    if physical:
        # Physical selected-column transport does not need Symbolic fit/reach sampling.
        # Keep every published region; prefer the explicit, ray-verified central Mark.
        from .physical_ontop_drop import physical_region_rank
        ordered = sorted(bindings, key=physical_region_rank)
        return tuple(ordered), (), [dict(marker=b.marker, selector=b.target_token.selector,
            selection_policy='PHYSICAL_CENTER_THEN_INWARD',
            priority=list(physical_region_rank(b))) for b in ordered]
    ranked, plans, reports = [], [], []
    for binding in bindings:
        token = binding.target_token
        if not isinstance(token, SpatialMarkerTargetToken) or token.operation != "place_on_top":
            raise RuntimeError("OnTop surface Marker lacks its Action20 region token")
        plan = plan_on_top_support_layer_candidates(
            held, token.target_token, preferred_orientation_xyzw=preferred_orientation_xyzw,
            region_center_xy=token.region_center_xy,
            region_extent_xy=token.region_extent_xy, surface_z_m=token.surface_z_m,
            include_support_center=include_support_center,
        )
        plans.append(plan)
        reachable = restrict_and_rank_on_top_plan(plan, base_xy, maximum_distance_m)
        distance = min((on_top_binding_horizontal_distance(item, base_xy)
                        for item in plan.bindings), default=None)
        reports.append({
            "marker": binding.marker, "selector": token.selector,
            "candidate_count": len(plan.bindings),
            "workspace_candidate_count": len(reachable.bindings),
            "minimum_future_footprint_distance_m": distance,
            "maximum_distance_m": maximum_distance_m,
            "distance_reference": "base_xy_to_future_object_footprint",
        })
        if reachable.bindings:
            ranked.append((distance, int(binding.marker[1:]), binding))
    return tuple(item[2] for item in sorted(ranked, key=lambda item: item[:2])), tuple(plans), reports


def probe_public_placement_before_grasp(
    *,
    operation: str,
    subject: Any,
    destination: Any,
    scene_objects: Sequence[Any] = (),
    fillable_regions: Sequence[Aabb3] | None = None,
    maximum_candidates: int = 48,
) -> dict[str, Any]:
    """Give a task-neutral advisory ordering hint without requiring held state.

    ``fillable_regions`` must come from actual fillable meta links, never the
    container's outer AABB. Missing Inside geometry is UNKNOWN. The caller can
    move a NO_PROPOSALS candidate behind other ready candidates, but must retain
    it for normal execution; ``can_reject`` is always false. UNKNOWN and positive
    hints have equal rank so missing geometry does not demote a candidate.

    No object setters, simulator steps, collision queries or state restore are
    performed. Geometry/data errors remain diagnostics, while unexpected API
    errors (including TypeError) propagate rather than masquerading as no fit.
    """
    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or maximum_candidates <= 0
    ):
        raise ValueError("maximum_candidates must be a positive integer")
    report: dict[str, Any] = {
        "version": "public-oracle-pregrasp/v1",
        "operation": operation,
        "status": "UNKNOWN",
        "candidate_count": None,
        "advisory_rank": 0,
        "may_defer": False,
        "can_reject": False,
        "simulator_state_mutated": False,
        "candidate_budget": maximum_candidates,
        "truncated": False,
        "actual_placement_admission_required": True,
    }
    if operation not in {"place_on_top", "place_inside"}:
        return {**report, "reason": "operation_has_no_shared_readonly_probe"}
    if operation == "place_inside" and not fillable_regions:
        return {**report, "reason": "official_fillable_geometry_unavailable"}
    try:
        if operation == "place_on_top":
            plan = plan_on_top_support_layer_candidates(
                subject,
                destination,
                scene_objects=scene_objects,
                allow_overhang=True,
                maximum_candidates=maximum_candidates + 1,
            )
            count = len(plan.candidates)
            report["geometry_source"] = "shared_ontop_broad_proposals"
        else:
            extent = live_object_extent_m(subject)
            count = 0
            regions = tuple(fillable_regions or ())
            for index, region in enumerate(regions):
                plan = plan_symbolic_inside_candidates(
                    held_extent_m=extent,
                    fillable_aabb=region,
                    max_candidates=maximum_candidates + 1 - count,
                )
                count += len(plan.candidates)
                if count > maximum_candidates:
                    report["unvisited_region_count"] = len(regions) - index - 1
                    break
            report["geometry_source"] = "shared_symbolic_inside_fillable_proposals"
        report.update(
            status="CANDIDATES_AVAILABLE" if count else "NO_PROPOSALS",
            candidate_count=min(count, maximum_candidates),
            truncated=count > maximum_candidates,
            advisory_rank=0 if count else 1,
            may_defer=not bool(count),
            reason=(
                "finite_broad_proposals_require_public_action_admission"
                if count
                else "finite_sampler_empty_not_a_proof_of_infeasibility"
            ),
        )
    except (ValueError, RuntimeError) as exc:
        report.update(
            reason="geometry_unavailable_or_invalid",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    return report
