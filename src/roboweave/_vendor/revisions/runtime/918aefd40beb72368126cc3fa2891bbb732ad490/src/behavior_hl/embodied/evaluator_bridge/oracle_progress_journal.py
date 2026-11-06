"""Append-only private progress evidence for long-running Oracle lanes.

The evaluator runner has several nested execution paths.  An exception in a
nested path must not make already executed actions disappear merely because
the nested function did not return its aggregate tuple.  This module owns a
small, simulator-independent journal which is updated at accepted action and
authoritative verifier boundaries.

The JSONL records are private artifacts.  :meth:`public_record` intentionally
contains no predicate truth, Q score, selected option, object identity, or
route detail and is safe to expose as an evidence-completeness statement.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.private_json_sanitizer import (
    PRIVATE_JSON_SANITIZER_VERSION,
    sanitize_private_json,
)


SCHEMA_VERSION = "behavior2026.oracle-progress-journal/v1"
_NONFINITE_ERROR = "progress evidence cannot contain NaN or infinity"
_ALLOWED_DIAGNOSTIC_SANITIZER_EVENTS = frozenset(
    {"non_finite_number", "unordered_container"}
)


def json_safe(value: Any) -> Any:
    """Return a strict-JSON representation without simulator imports.

    NumPy arrays and Torch tensors are handled through their public conversion
    protocols.  Importing either package here would make CPU-only reporting
    tests unnecessarily depend on the simulator environment.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(_NONFINITE_ERROR)
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted_items = [json_safe(item) for item in value]
        return sorted(
            converted_items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    converted = value
    detach = getattr(converted, "detach", None)
    if callable(detach):
        converted = detach()
    cpu = getattr(converted, "cpu", None)
    if callable(cpu):
        converted = cpu()
    tolist = getattr(converted, "tolist", None)
    if callable(tolist):
        listed = tolist()
        if listed is not value:
            return json_safe(listed)
    item = getattr(converted, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value:
            return json_safe(scalar)
    raise TypeError(
        "progress evidence contains an unsupported value of type "
        f"{type(value).__name__}"
    )


def _private_diagnostic_json_safe_raw(
    value: Any,
) -> tuple[Any, Mapping[str, Any] | None]:
    """Encode non-finite private diagnostics without weakening strict fields.

    The ordinary converter remains the primary path, preserving all existing
    supported shapes and failure behavior.  Only its explicit non-finite
    failure activates the bounded private sanitizer.  Unsupported values,
    cycles, or applied size/depth limits still fail journal persistence rather
    than being silently accepted under this narrowly scoped recovery.
    """

    try:
        return json_safe(value), None
    except ValueError as exc:
        if str(exc) != _NONFINITE_ERROR:
            raise

    result = sanitize_private_json(value)
    event_counts = dict(result.health.get("event_counts", {}))
    unexpected = sorted(
        kind
        for kind, count in event_counts.items()
        if count and kind not in _ALLOWED_DIAGNOSTIC_SANITIZER_EVENTS
    )
    if unexpected:
        raise TypeError(
            "progress diagnostic recovery encountered unsupported sanitizer "
            f"events: {unexpected}"
        )
    if int(event_counts.get("non_finite_number", 0)) < 1:
        raise ValueError(_NONFINITE_ERROR)
    return result.value, dict(result.health)


_BOUNDED_CANDIDATE_COLLECTION_KEYS = frozenset(
    {"candidate_attempts_private", "candidate_admission_private"}
)
_BOUNDED_ROUTE_PROBE_COLLECTION_KEYS = frozenset(
    {"portal_route_probes_private"}
)
_BOUNDED_REPEATED_ACTION_COLLECTION_KEYS = frozenset(
    {"place_attempts", "placement_attempts"}
)
_BOUNDED_CANDIDATE_SCALAR_KEYS = (
    "candidate",
    "accepted",
    "status",
    "candidate_applied",
    "official_postcondition_readback_complete",
    "complete_settle_window_required",
    "finite_pose_and_velocity_required",
    "observed_samples",
    "required_samples",
    "failure_reasons",
)
_BOUNDED_CANDIDATE_WINDOW_KEYS = (
    "canonical_relation_window",
    "canonical_support_window",
    "support_touching_window",
)

_BOUNDED_ROUTE_PROVIDER_SCALAR_KEYS = (
    "status",
    "provider_version",
    "floor",
    "map_with_objects",
    "source_snap_m",
    "source_snap_required",
    "station_index",
    "station_distance_m",
    "geodesic_m",
    "astar_calls",
    "component_prefilter",
    "official_waypoint_cell_count",
    "official_cell_path_length",
    "official_cell_path_kind",
    "executable_cell_path_length",
    "executable_cell_path_kind",
    "station_selection_policy",
    "exact_cell_astar_provider",
    "station_selection_geometry",
    "interaction_anchor_attempted_count",
    "interaction_anchor_room_mismatch_rejection_count",
    "interaction_anchor_without_station_rejection_count",
    "interaction_anchor_terminal_allowed_cell_count",
    "region_support_pair_used",
    "region_support_candidate_station_count",
    "region_support_candidate_anchor_count",
    "station_target_room_source",
    "floor_map_override_applied",
    "floor_map_override_provenance",
    "dynamic_overlay_policy",
    "dynamic_overlay_cell_count",
    "excluded_station_approach_count",
)
_BOUNDED_ROUTE_PROVIDER_POINT_KEYS = (
    "source_requested_official_cell",
    "source_snapped_official_cell",
    "station_target_official_cell",
    "station_xy",
    "station_approach_key",
)
_BOUNDED_ACTION_SCALAR_KEYS = (
    "sequence",
    "operation",
    "status",
    "held_object_retained",
    "object_acquired",
    "placement_committed",
    "recoverable_executor_noop",
    "accepted_official_success_termination",
    "simulator_steps",
)


def _validate_supported_progress_value(
    value: Any,
    *,
    active_container_ids: set[int] | None = None,
) -> None:
    """Visit omitted diagnostics so summaries never hide unsupported values."""

    if active_container_ids is None:
        active_container_ids = set()
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        identity = id(value)
        if identity in active_container_ids:
            raise TypeError("progress diagnostic contains a cycle")
        active_container_ids.add(identity)
        try:
            items = value.values() if isinstance(value, Mapping) else value
            for item in items:
                _validate_supported_progress_value(
                    item,
                    active_container_ids=active_container_ids,
                )
        finally:
            active_container_ids.remove(identity)
        return
    _private_diagnostic_json_safe_raw(value)


def _bounded_route_provider_summary(value: Any) -> Mapping[str, Any]:
    """Keep route outcome/provenance while bounding planner search detail."""

    if not isinstance(value, Mapping):
        raise TypeError("route provider progress diagnostic must be a mapping")
    _validate_supported_progress_value(value)
    summary: dict[str, Any] = {
        "schema_version": "behavior2026.progress-route-provider-summary/v1",
        "original_field_count": len(value),
    }
    for key in _BOUNDED_ROUTE_PROVIDER_SCALAR_KEYS:
        if key in value:
            summary[key] = json_safe(value[key])
    for key in _BOUNDED_ROUTE_PROVIDER_POINT_KEYS:
        if key in value:
            summary[key] = json_safe(value[key])
    collection_counts: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(item, (Mapping, list, tuple, set, frozenset)):
            collection_counts[str(key)] = len(item)
    if collection_counts:
        summary["collection_item_counts_private"] = dict(
            sorted(collection_counts.items())
        )
    for key in ("official_cell_path", "executable_cell_path"):
        path = value.get(key)
        if isinstance(path, (list, tuple)):
            summary[f"{key}_endpoints_private"] = {
                "item_count": len(path),
                "first": json_safe(path[0]) if path else None,
                "last": json_safe(path[-1]) if path else None,
            }
    return summary


def _bounded_action_summary(value: Any) -> Mapping[str, Any]:
    """Summarize a repeated action attempt; the primary action stays exact."""

    if not isinstance(value, Mapping):
        raise TypeError("repeated action progress diagnostic must be a mapping")
    _validate_supported_progress_value(value)
    summary: dict[str, Any] = {
        "schema_version": "behavior2026.progress-action-summary/v1",
        "original_field_count": len(value),
    }
    for key in _BOUNDED_ACTION_SCALAR_KEYS:
        if key in value:
            summary[key] = json_safe(value[key])
    primitive = value.get("primitive_report_private")
    if isinstance(primitive, Mapping):
        summary["primitive_report_private"] = {
            key: json_safe(primitive[key])
            for key in (
                "operation",
                "executor_version",
                "candidate_count",
                "admitted_candidate_count",
                "fresh_candidate_generation_count",
                "placement_committed",
                "held_object_retained",
                "recoverable_executor_noop",
                "terminal_success_finalize",
            )
            if key in primitive
        }
    return summary


def _candidate_window_summary(value: Any) -> Mapping[str, Any]:
    safe = json_safe(value)
    if not isinstance(safe, list) or any(not isinstance(item, bool) for item in safe):
        raise TypeError("candidate verification window must be a Boolean sequence")
    return {
        "sample_count": len(safe),
        "true_count": sum(1 for item in safe if item),
        "all_true": bool(safe) and all(safe),
    }


def _bounded_candidate_summary(
    value: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    safe, health = _private_diagnostic_json_safe_raw(value)
    if not isinstance(safe, Mapping):
        raise TypeError("candidate progress diagnostic must be a mapping")
    summary: dict[str, Any] = {
        "schema_version": "behavior2026.progress-candidate-summary/v1"
    }
    for key in _BOUNDED_CANDIDATE_SCALAR_KEYS:
        if key in safe:
            summary[key] = safe[key]
    for key in _BOUNDED_CANDIDATE_WINDOW_KEYS:
        if key in safe:
            summary[key] = _candidate_window_summary(safe[key])
    collision = safe.get("collision_shape_admission_private")
    if isinstance(collision, Mapping):
        query_box = collision.get("collision_query_box_private")
        summary["collision_shape_admission_private"] = {
            "accepted_for_candidate_execution": collision.get(
                "accepted_for_candidate_execution"
            ),
            "collision_query_reason": collision.get("collision_query_reason"),
            "blocking_overlap_hit_count": len(
                collision.get("blocking_overlap_hits_private", ())
            ),
            "geometry_source": (
                query_box.get("geometry_source")
                if isinstance(query_box, Mapping)
                else None
            ),
        }
    reprojection = safe.get("rejected_candidate_carry_reprojection_private")
    if isinstance(reprojection, Mapping):
        summary["rejected_candidate_carry_reprojection_private"] = {
            key: reprojection.get(key)
            for key in ("version", "status", "error_type", "error")
            if key in reprojection
        }
    return summary, health


def _merge_sanitizer_health(
    health_records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not health_records:
        return None
    event_counts: dict[str, int] = {}
    for health in health_records:
        for kind, count in dict(health.get("event_counts", {})).items():
            event_counts[kind] = event_counts.get(kind, 0) + int(count)
    return {
        "schema_version": PRIVATE_JSON_SANITIZER_VERSION,
        "status": "SANITIZED",
        "strict_json_pass": True,
        "event_count": sum(event_counts.values()),
        "event_counts": dict(sorted(event_counts.items())),
        "bounded_component_count": len(health_records),
    }


def _bounded_progress_collections(
    value: Any,
) -> tuple[Any, list[Mapping[str, Any]]]:
    health_records: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text == "route_provider_private":
                converted[key_text] = _bounded_route_provider_summary(item)
                continue
            if key_text in _BOUNDED_ROUTE_PROBE_COLLECTION_KEYS:
                if not isinstance(item, Mapping):
                    raise TypeError(f"{key_text} must be a route-probe mapping")
                converted[key_text] = {
                    str(probe_name): _bounded_route_provider_summary(probe)
                    for probe_name, probe in item.items()
                }
                continue
            if key_text in _BOUNDED_REPEATED_ACTION_COLLECTION_KEYS:
                if not isinstance(item, Sequence) or isinstance(
                    item, (str, bytes, bytearray)
                ):
                    raise TypeError(f"{key_text} must be an action sequence")
                converted[key_text] = [
                    _bounded_action_summary(action) for action in item
                ]
                continue
            if key_text in _BOUNDED_CANDIDATE_COLLECTION_KEYS:
                if not isinstance(item, Sequence) or isinstance(
                    item, (str, bytes, bytearray)
                ):
                    raise TypeError(
                        f"{key_text} must be a candidate sequence"
                    )
                if len(item) <= 8:
                    converted_item, nested_health = (
                        _bounded_progress_collections(list(item))
                    )
                    converted[key_text] = converted_item
                    health_records.extend(nested_health)
                    continue
                summaries: list[Mapping[str, Any]] = []
                for candidate in item:
                    summary, health = _bounded_candidate_summary(candidate)
                    summaries.append(summary)
                    if health is not None:
                        health_records.append(health)
                converted[key_text] = summaries
                continue
            converted_item, nested_health = _bounded_progress_collections(item)
            converted[key_text] = converted_item
            health_records.extend(nested_health)
        return converted, health_records
    if isinstance(value, (list, tuple)):
        converted_items: list[Any] = []
        for item in value:
            converted_item, nested_health = _bounded_progress_collections(item)
            converted_items.append(converted_item)
            health_records.extend(nested_health)
        return converted_items, health_records
    return value, health_records


def _private_diagnostic_json_safe(
    value: Any,
) -> tuple[Any, Mapping[str, Any] | None]:
    bounded, component_health = _bounded_progress_collections(value)
    safe, outer_health = _private_diagnostic_json_safe_raw(bounded)
    if outer_health is not None:
        component_health.append(outer_health)
    return safe, _merge_sanitizer_health(component_health)


def _private_route_reports_json_safe(
    route_reports: Sequence[Mapping[str, Any]],
) -> tuple[list[Any], Mapping[str, Any] | None]:
    """Sanitize each report independently so a long prefix stays bounded.

    A single live OnTop report is already a deep, thousand-node diagnostic.
    Sanitizing an ever-growing route list as one value would eventually hit a
    whole-container resource bound and recreate the persistence outage.  Per-
    report conversion keeps token paths deterministic across later appends and
    records the original report index alongside each sanitizer health record.
    """

    converted: list[Any] = []
    reports_health: list[dict[str, Any]] = []
    event_counts: dict[str, int] = {}
    for index, report in enumerate(route_reports):
        safe_report, health = _private_diagnostic_json_safe(report)
        if not isinstance(safe_report, Mapping):
            raise TypeError("route report sanitization must preserve a mapping")
        converted.append(safe_report)
        if health is None:
            continue
        reports_health.append(
            {
                "route_report_index_private": index,
                "sanitizer_private": health,
            }
        )
        for event_kind, count in dict(health.get("event_counts", {})).items():
            event_counts[event_kind] = event_counts.get(event_kind, 0) + int(
                count
            )
    if not reports_health:
        return converted, None
    return converted, {
        "schema_version": PRIVATE_JSON_SANITIZER_VERSION,
        "status": "SANITIZED",
        "strict_json_pass": True,
        "event_count": sum(event_counts.values()),
        "event_counts": dict(sorted(event_counts.items())),
        "route_report_sanitizers_private": reports_health,
    }


def _non_negative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _snapshot_private_record(
    snapshot: Any,
    *,
    selected_leaf_ids: Sequence[str] | None,
    selected_option_index: int | None,
) -> dict[str, Any]:
    ok = getattr(snapshot, "ok", None)
    if ok is not True:
        raise ValueError("only an operational verifier snapshot is authoritative")
    official_q = getattr(snapshot, "official_q", None)
    official_success = getattr(snapshot, "official_success", None)
    leaf_truth = getattr(snapshot, "leaf_truth", None)
    if (
        not isinstance(official_q, (int, float))
        or isinstance(official_q, bool)
        or not math.isfinite(float(official_q))
        or not 0.0 <= float(official_q) <= 1.0
    ):
        raise ValueError("authoritative verifier snapshot has invalid official_q")
    if not isinstance(official_success, bool):
        raise ValueError("authoritative verifier snapshot has invalid success")
    if not isinstance(leaf_truth, Mapping) or not leaf_truth:
        raise ValueError("authoritative verifier snapshot has no leaf truth")
    if any(
        not isinstance(leaf_id, str)
        or not leaf_id
        or not isinstance(truth, bool)
        for leaf_id, truth in leaf_truth.items()
    ):
        raise ValueError("authoritative verifier snapshot has invalid leaf truth")

    if selected_leaf_ids is None:
        selected_truth = dict(leaf_truth)
    else:
        selected_ids = tuple(selected_leaf_ids)
        missing = [leaf_id for leaf_id in selected_ids if leaf_id not in leaf_truth]
        if missing:
            raise ValueError(
                "authoritative selected-option truth is incomplete: "
                f"{missing}"
            )
        selected_truth = {leaf_id: leaf_truth[leaf_id] for leaf_id in selected_ids}
    return {
        "official_q": float(official_q),
        "official_success": official_success,
        "selected_option_index": selected_option_index,
        "selected_option_leaf_truth": selected_truth,
    }


class OracleProgressJournal:
    """Keep exact execution counters and the last authoritative goal state."""

    def __init__(self, path: Path, *, lane: str) -> None:
        if not isinstance(lane, str) or not lane:
            raise ValueError("progress journal lane must be non-empty")
        self.path = Path(path)
        self.lane = lane
        self._sequence = 0
        self._actions_executed: int | None = None
        self._simulator_steps: int | None = None
        self._visual_frame_count: int | None = None
        self._route_reports: list[dict[str, Any]] = []
        self._last_authoritative: dict[str, Any] | None = None
        self._last_kind: str | None = None
        self._health_errors: list[dict[str, Any]] = []
        self._diagnostic_sanitization_event_count = 0
        self._diagnostic_sanitization_event_counts: dict[str, int] = {}
        self._sanitized_route_report_counted_indices: set[int] = set()

    def record(
        self,
        *,
        kind: str,
        actions_executed: int,
        simulator_steps: int,
        route_reports: Sequence[Mapping[str, Any]],
        visual_frame_count: int,
        snapshot: Any | None = None,
        selected_leaf_ids: Sequence[str] | None = None,
        selected_option_index: int | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if not isinstance(kind, str) or not kind:
            raise ValueError("progress event kind must be non-empty")
        actions = _non_negative_integer(actions_executed, field="actions_executed")
        steps = _non_negative_integer(simulator_steps, field="simulator_steps")
        frames = _non_negative_integer(
            visual_frame_count, field="visual_frame_count"
        )
        if self._actions_executed is not None and actions < self._actions_executed:
            raise ValueError("actions_executed cannot regress")
        if self._simulator_steps is not None and steps < self._simulator_steps:
            raise ValueError("simulator_steps cannot regress")
        if self._visual_frame_count is not None and frames < self._visual_frame_count:
            raise ValueError("visual_frame_count cannot regress")
        safe_routes, route_sanitization = _private_route_reports_json_safe(
            route_reports
        )
        if len(safe_routes) < len(self._route_reports):
            raise ValueError("route report sequence cannot regress")
        if safe_routes[: len(self._route_reports)] != self._route_reports:
            raise ValueError("previous route reports cannot be rewritten")
        new_routes = safe_routes[len(self._route_reports) :]

        authoritative = None
        if snapshot is not None:
            authoritative = _snapshot_private_record(
                snapshot,
                selected_leaf_ids=selected_leaf_ids,
                selected_option_index=selected_option_index,
            )

        safe_detail: Any = None
        detail_sanitization: Mapping[str, Any] | None = None
        if detail is not None:
            safe_detail, detail_sanitization = _private_diagnostic_json_safe(
                detail
            )
            if not isinstance(safe_detail, Mapping):
                raise TypeError("detail sanitization must preserve a mapping")

        diagnostic_sanitization = {
            "schema_version": PRIVATE_JSON_SANITIZER_VERSION,
            "route_reports_private": route_sanitization,
            "detail_private": detail_sanitization,
        }
        if route_sanitization is None and detail_sanitization is None:
            diagnostic_sanitization = None

        event = {
            "schema_version": SCHEMA_VERSION,
            "sequence": self._sequence,
            "lane": self.lane,
            "kind": kind,
            "actions_executed": actions,
            "simulator_steps": steps,
            "route_report_count": len(safe_routes),
            "new_route_reports_private": new_routes,
            "visual_frame_count": frames,
            "authoritative_official_state_private": authoritative,
            "detail_private": safe_detail,
            "diagnostic_json_sanitization_private": diagnostic_sanitization,
        }
        safe_event = json_safe(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(safe_event, ensure_ascii=False, allow_nan=False) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

        self._sequence += 1
        self._actions_executed = actions
        self._simulator_steps = steps
        self._visual_frame_count = frames
        self._route_reports = safe_routes
        if authoritative is not None:
            self._last_authoritative = authoritative
        self._last_kind = kind
        if route_sanitization is not None:
            for report_health in route_sanitization.get(
                "route_report_sanitizers_private", ()
            ):
                report_index = int(
                    report_health["route_report_index_private"]
                )
                if report_index in self._sanitized_route_report_counted_indices:
                    continue
                self._sanitized_route_report_counted_indices.add(report_index)
                self._count_diagnostic_sanitization_health(
                    report_health["sanitizer_private"]
                )
        if detail_sanitization is not None:
            self._count_diagnostic_sanitization_health(detail_sanitization)
        return safe_event

    def _count_diagnostic_sanitization_health(
        self, health: Mapping[str, Any]
    ) -> None:
        for event_kind, count in dict(health.get("event_counts", {})).items():
            numeric_count = int(count)
            self._diagnostic_sanitization_event_count += numeric_count
            self._diagnostic_sanitization_event_counts[event_kind] = (
                self._diagnostic_sanitization_event_counts.get(event_kind, 0)
                + numeric_count
            )

    def safe_record(self, **kwargs: Any) -> bool:
        """Best-effort journal append which never changes Oracle control flow.

        Reporting is a side channel.  Serialization, filesystem, flush, or
        fsync failures are recorded as private journal-health evidence and
        must not replace an executor result or mask its original exception.
        A JSON-safe volatile checkpoint is retained when possible so the
        in-process terminal summary can still report exact work, while its
        completeness explicitly states that durable persistence degraded.
        """

        try:
            self.record(**kwargs)
            return True
        except Exception as exc:  # reporting must never fail the episode
            health_error = {
                "schema_version": SCHEMA_VERSION,
                "health_status": "DEGRADED",
                "failed_event_kind": str(kwargs.get("kind", "unknown")),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self._health_errors.append(health_error)
            self._append_health_error_best_effort(health_error)
            self._retain_volatile_checkpoint_best_effort(kwargs)
            return False

    def _append_health_error_best_effort(
        self, health_error: Mapping[str, Any]
    ) -> None:
        health_path = self.path.with_name(
            f"{self.path.stem}_health.jsonl"
        )
        try:
            health_path.parent.mkdir(parents=True, exist_ok=True)
            with health_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        dict(health_error),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            # There is intentionally no third reporting channel whose own
            # failure could recurse or affect the Oracle.
            return

    def _retain_volatile_checkpoint_best_effort(
        self, kwargs: Mapping[str, Any]
    ) -> None:
        try:
            actions = _non_negative_integer(
                kwargs.get("actions_executed"), field="actions_executed"
            )
            steps = _non_negative_integer(
                kwargs.get("simulator_steps"), field="simulator_steps"
            )
            frames = _non_negative_integer(
                kwargs.get("visual_frame_count"), field="visual_frame_count"
            )
            if self._actions_executed is not None:
                actions = max(actions, self._actions_executed)
            if self._simulator_steps is not None:
                steps = max(steps, self._simulator_steps)
            if self._visual_frame_count is not None:
                frames = max(frames, self._visual_frame_count)
            self._actions_executed = actions
            self._simulator_steps = steps
            self._visual_frame_count = frames
        except Exception:
            pass

        try:
            safe_routes, _health = _private_route_reports_json_safe(
                kwargs.get("route_reports", ())
            )
            if (
                len(safe_routes) >= len(self._route_reports)
                and safe_routes[: len(self._route_reports)]
                == self._route_reports
            ):
                self._route_reports = safe_routes
        except Exception:
            pass

        try:
            snapshot = kwargs.get("snapshot")
            if snapshot is not None:
                self._last_authoritative = _snapshot_private_record(
                    snapshot,
                    selected_leaf_ids=kwargs.get("selected_leaf_ids"),
                    selected_option_index=kwargs.get(
                        "selected_option_index"
                    ),
                )
        except Exception:
            pass
        kind = kwargs.get("kind")
        if isinstance(kind, str) and kind:
            self._last_kind = kind
        # Reserve the failed event's sequence number.  The durable JSONL may
        # contain a gap, which is preferable to a duplicate sequence when a
        # write succeeded but its fsync raised.
        self._sequence += 1

    def private_summary_fields(self) -> dict[str, Any]:
        """Return recovery fields for a private terminal summary.

        No authoritative snapshot means Q/success are ``None``.  The caller
        must never convert missing evidence to a fabricated zero.
        """

        authoritative = self._last_authoritative
        durable = not self._health_errors
        return {
            "actions_executed": self._actions_executed,
            "simulator_steps": self._simulator_steps,
            "route_reports": list(self._route_reports),
            "visual_frame_count": self._visual_frame_count,
            "official_q": (
                None if authoritative is None else authoritative["official_q"]
            ),
            "official_success": (
                None
                if authoritative is None
                else authoritative["official_success"]
            ),
            "last_authoritative_selected_option_truth_private": (
                None
                if authoritative is None
                else dict(authoritative["selected_option_leaf_truth"])
            ),
            "last_authoritative_selected_option_index_private": (
                None
                if authoritative is None
                else authoritative["selected_option_index"]
            ),
            "progress_evidence_completeness": {
                "execution_counters": (
                    (
                        "exact_append_only_checkpoint"
                        if durable
                        else "exact_volatile_checkpoint"
                    )
                    if self._actions_executed is not None
                    else "unavailable"
                ),
                "route_reports": (
                    (
                        "exact_append_only_prefix"
                        if durable
                        else "volatile_prefix_persistence_degraded"
                    )
                    if self._actions_executed is not None
                    else "unavailable"
                ),
                "visual_frames": (
                    (
                        "exact_append_only_checkpoint"
                        if durable
                        else "exact_volatile_checkpoint"
                    )
                    if self._visual_frame_count is not None
                    else "unavailable"
                ),
                "authoritative_official_state": (
                    "last_checkpoint"
                    if authoritative is not None
                    else "unavailable"
                ),
                "terminal_authoritative_official_state": bool(
                    authoritative is not None
                    and self._last_kind in {"terminal", "postcondition"}
                ),
            },
            "progress_journal_private": str(self.path),
            "progress_journal_health_private": {
                "healthy": durable,
                "error_count": len(self._health_errors),
                "errors": list(self._health_errors),
                "diagnostic_sanitization_private": {
                    "schema_version": PRIVATE_JSON_SANITIZER_VERSION,
                    "event_count": self._diagnostic_sanitization_event_count,
                    "event_counts": dict(
                        sorted(
                            self._diagnostic_sanitization_event_counts.items()
                        )
                    ),
                },
                "health_artifact": str(
                    self.path.with_name(
                        f"{self.path.stem}_health.jsonl"
                    )
                ),
            },
        }

    def public_record(self) -> dict[str, Any]:
        """Return evidence availability without leaking private GT."""

        return {
            "schema_version": SCHEMA_VERSION,
            "event_count": self._sequence,
            "journal_healthy": not self._health_errors,
            "journal_health_error_count": len(self._health_errors),
            "execution_checkpoint_available": self._actions_executed is not None,
            "authoritative_verifier_checkpoint_available": (
                self._last_authoritative is not None
            ),
        }
