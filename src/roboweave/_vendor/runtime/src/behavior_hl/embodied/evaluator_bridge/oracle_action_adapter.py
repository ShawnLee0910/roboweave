"""Thin Oracle-to-Action20 binding; no navigation or recovery policy.

Action20 owns request validation, execution, diagnostics and Marker publication.
Oracle owns candidate selection, retry policy and measured-state validation.
The runner injects its existing executor (including frame/receipt recording),
so this module never imports the runner or implements another dispatcher.
"""

from __future__ import annotations

from behavior_hl.embodied.policy.protocol import HOUSEHOLD_ACTIONS

from pathlib import Path
from collections import Counter
from functools import lru_cache
from time import perf_counter
from typing import Any, Callable, Mapping

from behavior_hl.embodied.runtime.dispatch import ActionPluginResult
from behavior_hl.navigation import PlanarPose


# Stages consumed after the empty-argument request. C-labels are still resolved
# from each fresh runtime snapshot by Oracle; this table never selects a label.
PUBLIC_ACTION20_ORACLE_MARKER_STAGES = {
    "pick": (("target", ("object", "bundle")),),
    "place_inside": (
        ("reference", ("object",)),
        ("inside_cell", ("inside_cell",)),
    ),
    "place_on_top": (
        ("reference", ("object",)),
        ("surface_cell", ("surface_cell",)),
    ),
    "place_under": (("target", ("object",)),),
    "place_next_to": (
        ("reference", ("object",)),
        ("position", ("position",)),
    ),
    "place_touching": (
        ("reference", ("object",)),
        ("direction", ("direction",)),
    ),
    "open": (("target", ("part", "object")),),
    "close": (("target", ("part", "object")),),
    "toggle_on": (("target", ("part", "object")),),
    "toggle_off": (("target", ("part", "object")),),
    "wipe": (("target", ("object",)),),
    **{name: (("target", ("object",)),) for name in HOUSEHOLD_ACTIONS},
}


def execute_oracle_action(
    *,
    runtime: Any,
    execute: Callable[..., tuple[ActionPluginResult, tuple[Any, ...], int]],
    phase_dir: Path,
    action_index: int,
    name: str,
    reasoning: str,
    args: Mapping[str, Any] | None = None,
    enforce_public_carry_contract: bool = False,
    terminal_success_check: Callable[[], Any] | None = None,
    capture_before_frame: bool = True,
    capture_after_frame: bool = True,
    execution_diagnostic_private: dict[str, Any] | None = None,
) -> tuple[ActionPluginResult, int]:
    """Parse with the live registry and execute exactly once, without retries."""

    payload: dict[str, Any] = {"action": name, "reasoning": reasoning}
    if args is not None:
        payload["args"] = dict(args)
    action = runtime.registry.parse_payload(payload)
    result, _, steps = execute(
        runtime=runtime,
        action=action,
        action_index=action_index,
        phase_dir=phase_dir,
        enforce_public_carry_contract=enforce_public_carry_contract,
        terminal_success_check=terminal_success_check,
        capture_before_frame=capture_before_frame,
        capture_after_frame=capture_after_frame,
        execution_diagnostic_private=execution_diagnostic_private,
    )
    return result, steps


def navigation_execution_receipt(
    result: ActionPluginResult,
    measured_pose: PlanarPose,
    route_receipt: Mapping[str, Any],
) -> tuple[str, PlanarPose, Any]:
    """Pass executor facts through; the controller validates their types/state.

    Do not infer clipping from float32 pose readback or coerce missing/malformed
    evidence to False. Those receipts must still fail closed in the controller.
    """

    return (
        result.status,
        measured_pose,
        route_receipt.get("executor_translation_clipped_private"),
    )


def marker_publication_evidence(*, diagnostic, operation, stage):
    """Normalize only reasons explicitly emitted by this publication call.

    The caller supplies a newly allocated dispatch diagnostic, never a cached
    backend report. Counts do not establish why a candidate was filtered.
    """
    report = {"publication_filter_reason": "UNKNOWN", "publication_recovery_scope": "UNKNOWN",
              "publication_diagnostic_private": {}}
    if not isinstance(diagnostic, Mapping) or diagnostic.get("operation") != operation:
        return report
    source_stage = diagnostic.get("stage")
    if not isinstance(source_stage, str):
        return report
    second_stage = stage in {"surface_cell", "inside_cell", "position", "direction"}
    scopes = {"marker_frame_unavailable": "UNVERIFIED"}
    if second_stage:
        scopes["reference_no_longer_visible"] = "VIEW_OR_STATION"
    if operation == "place_next_to" and stage == "position":
        scopes["reference_geometry_unavailable"] = "UNVERIFIED"
    # Preserve bounded publication facts, not an arbitrary old action report.
    publication_stages = {"pick_marker_publication", "place_on_top_support_publication",
                          "place_on_top_surface_publication"}
    if source_stage not in scopes and source_stage not in publication_stages:
        return report
    report["publication_diagnostic_private"] = {
        key: value for key, value in diagnostic.items()
        if key in {"operation", "stage", "surface_count", "support_count", "visual_fallback_count",
                   "generic_candidate_count", "relation_bundle_count", "candidate_coverage_miss"}
        and isinstance(value, (str, int, bool))
    }
    if source_stage in scopes:
        report.update(publication_filter_reason=source_stage,
                      publication_recovery_scope=scopes[source_stage])
    return report


def static_endpoint_predicate(*, scene, robot, cell_resolver, corridor, start_pose=None):
    """Action20's exact static endpoint guard, cached only for one A* call.

    No corridor/eroded-map substitution, swept edge, clearance, or dynamic
    collision check is introduced. Unknown evidence stays optimistic so the
    real Action remains authoritative. Exact pose keys avoid merging distinct
    subcell positions or headings; the bounded cache dies before dispatch.
    """
    from .private_traversability_route import prepare_official_endpoint_check

    stats = {"version": "oracle-static-endpoint-parity/v1", "requests": 0,
             "queries": 0, "static_rejected": 0, "unknown": 0,
             "query_seconds": 0.0, "reason_counts": Counter(),
             "cache_scope": "planner_call", "cache_max_entries": 4096,
             "map_source": "action_scene_trav_map", "edge_is_open": None}
    prepared = False
    endpoint_check = None

    @lru_cache(maxsize=4096)
    def check(x, y, z, yaw):
        nonlocal prepared, endpoint_check
        started = perf_counter()
        stats["queries"] += 1
        try:
            endpoint = PlanarPose(x, y, z, yaw)
            if not prepared:
                prepared = True
                endpoint_check, report = prepare_official_endpoint_check(scene=scene, robot=robot, pose=endpoint)
                stats["static_context_private"] = report
            if endpoint_check is None:
                stats["unknown"] += 1
                return True
            opened = bool(endpoint_check(endpoint))
        except (AttributeError, ImportError, RuntimeError, ValueError) as exc:
            stats["reason_counts"][type(exc).__name__] += 1
            stats["unknown"] += 1
            return True
        finally:
            stats["query_seconds"] += perf_counter() - started
        stats["reason_counts"]["base_footprint_segment_open" if opened else "navigation_map_blocked"] += 1
        if not opened:
            stats["static_rejected"] += 1
            return False
        return True

    def pose_is_open(pose):
        stats["requests"] += 1
        if cell_resolver(pose) not in corridor:
            return False
        # The measured initial pose is not an attempted endpoint. Preserve
        # endpoint-only escape from a bad source; every generated successor
        # (including a newly constructed equal pose) still passes the guard.
        if pose is start_pose:
            return True
        return check(float(pose.x), float(pose.y), float(pose.z), float(pose.yaw))

    return pose_is_open, stats
