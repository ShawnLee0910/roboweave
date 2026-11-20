"""Live adapter for official bbox marking and official Symbolic GRASP."""

from __future__ import annotations

from behavior_hl.embodied.policy.protocol import HOUSEHOLD_ACTIONS

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
)
from behavior_hl.embodied.evaluator_bridge.private_drawer_parts import (
    DrawerPartTargetToken,
)
from behavior_hl.embodied.evaluator_bridge.private_visual_markers import (
    MarkerPublicFailureReason,
    MarkerStatus,
    PickBundleTargetToken,
    PrivateMarkerBinding,
    PrivateMarkerResolution,
    PrivateMarkerSnapshot,
    PrivateVisualMarkerStore,
    SpatialMarkerTargetToken,
    build_private_container_compartment_snapshot,
    build_private_drawer_part_snapshot,
    build_private_marker_snapshot,
    build_private_pick_snapshot,
    build_private_next_to_position_snapshot,
    build_private_on_top_support_snapshot,
    build_private_touching_direction_snapshot,
    build_private_toggle_part_snapshot,
    current_target_visibility,
    render_marker_overlay,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_resident_environment import (
    ResidentEvaluatorTerminalError,
)
from behavior_hl.embodied.runtime.plugins.marker_symbolic_grasp import (
    SymbolicGraspStatus,
    SymbolicGraspTick,
)
from behavior_hl.embodied.runtime.public_action_feedback import (
    ACTION20_BLOCKED_CATEGORIES,
    ACTION20_BLOCKED_CATEGORY_METADATA_KEY,
    EXECUTION_UNVERIFIED,
    PLACEMENT_COLLISION_BLOCKED,
    RELATION_NOT_REALIZED,
    action20_blocked_detail,
)
from behavior_hl.embodied.runtime.action_integrity import (
    ActionIntegrityError, execution_integrity_failure, invalidate_execution,
)


MARKER_SYMBOLIC_BACKEND_VERSION = "behavior2026.marker-symbolic-grasp-backend/v145"

AUTOMATIC_INTERACTION_APPROACH_OPERATIONS = frozenset(
    {
        "pick",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
        "place_inside",
        "place_next_to",
        "place_under",
        "place_on_top",
    }
)

_SPATIAL_OPERATIONS = frozenset(
    {
        "place_on_top",
        "place_next_to",
        "place_under",
        "place_touching",
    }
)
_PLACEMENT_OR_RELEASE_OPERATIONS = frozenset(
    {"place_inside", "release", "place_on_floor", *_SPATIAL_OPERATIONS}
)
_RECOVERABLE_PLACEMENT_OPERATIONS = frozenset(
    {"place_inside", "place_on_floor", *_SPATIAL_OPERATIONS}
)
_PHYSICAL_CONSEQUENCE_PLACEMENT_OPERATIONS = frozenset(
    {
        "place_inside",
        "place_on_top",
        "place_on_floor",
        "place_next_to",
        "place_under",
    }
)
_DETERMINISTIC_ACTION_PRIMITIVE_REASONS = frozenset(
    {"PRE_CONDITION_ERROR", "SAMPLING_ERROR", "POST_CONDITION_ERROR"}
)
_PRIVATE_NAVIGATION_BLOCKED_CATEGORIES = frozenset(
    {"navigation_path_blocked", "navigation_collision_blocked"}
)
_TRUSTED_ACTION20_BLOCKED_CATEGORIES = (
    ACTION20_BLOCKED_CATEGORIES | _PRIVATE_NAVIGATION_BLOCKED_CATEGORIES
)
_TWO_STAGE_OPERATIONS = frozenset(
    {"place_inside", "place_on_top", "place_next_to", "place_touching"}
)


def _verified_preparation_rollback(report: Any) -> bool:
    if not isinstance(report, Mapping):
        return False
    logical = report.get("logical_restore_private")
    poses = report.get("protected_pose_restore_private")
    return (
        report.get("status") == "rolled_back"
        and report.get("rollback_applied") is True
        and report.get("base_pose_restored") is True
        and report.get("inside_edges_restored") is True
        and isinstance(poses, Mapping) and poses.get("accepted") is True
        and isinstance(logical, Mapping)
        and logical.get("held_identity_restored") is True
        and logical.get("logical_inventory_identity_restored") is True
    )


def _base_marker_target(target: Any) -> Any:
    while isinstance(
        target,
        (SpatialMarkerTargetToken, DrawerPartTargetToken, PickBundleTargetToken),
    ):
        target = target.target_token
    return target


def _has_official_fillable_meta_link(target: Any) -> bool:
    links = getattr(target, "links", None)
    return isinstance(links, Mapping) and any(
        bool(getattr(link, "is_meta_link", False))
        and "fillable" in str(getattr(link, "meta_link_type", "")).casefold()
        for link in links.values()
    )


class MarkerSymbolicBackendError(RuntimeError):
    pass


class MarkerInputUnavailableError(MarkerSymbolicBackendError):
    """A selected target has no current geometry to admit safely."""


class Action20CommitRejectedError(RuntimeError):
    def __init__(self, message: str, *, private_report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.private_report = dict(private_report)


class Action23PostconditionError(MarkerSymbolicBackendError):
    pass


@dataclass(slots=True)
class _SymbolicToken:
    generator: Any | None
    placement_rollback: Callable[[], Mapping[str, Any]] | None = field(default=None, repr=False)
    placement_boundary_started: bool = False
    preparation: Any | None = field(default=None, repr=False)
    generator_factory: Callable[[Any], Any] | None = field(
        default=None,
        repr=False,
    )
    preparation_finalize: Callable[[bool], Mapping[str, Any]] | None = field(
        default=None,
        repr=False,
    )
    preparation_report_private: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
    )
    preparation_finalization_private: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
    )
    symbolic_failure_private: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
    )
    articulation_result_private: Mapping[str, Any] | None = field(default=None, repr=False)
    operation: str | None = None
    target_token: Any | None = field(default=None, repr=False)
    acted_object: Any | None = field(default=None, repr=False)
    preparation_steps: int = 0
    simulator_steps: int = 0
    symbolic_generator_start_count: int = 0
    phase: str = "local_prepare"
    terminal: bool = False
    error_type: str | None = None
    error: str | None = None
    outcome_verifier_private: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
    )
    started_at_private: float = field(default_factory=perf_counter, repr=False)
    preparation_seconds_private: float | None = field(default=None, repr=False)
    symbolic_started_at_private: float | None = field(default=None, repr=False)
    symbolic_generator_seconds_private: float | None = field(
        default=None,
        repr=False,
    )
    outcome_verifier_seconds_private: float | None = field(
        default=None,
        repr=False,
    )
    finalization_seconds_private: float | None = field(default=None, repr=False)
    terminal_seconds_private: float | None = field(default=None, repr=False)

    def private_diagnostic(self) -> dict[str, Any]:
        report = {
            "phase": self.phase,
            "preparation_steps": self.preparation_steps,
            "terminal": self.terminal,
            "error_type": self.error_type,
            "error": self.error,
            "symbolic_generator_start_count": (
                self.symbolic_generator_start_count
            ),
        }
        if self.outcome_verifier_private is not None:
            report["outcome_verifier_private"] = dict(
                self.outcome_verifier_private
            )
        if self.preparation_report_private is not None:
            report["preparation_report_private"] = dict(
                self.preparation_report_private
            )
        if self.preparation_finalization_private is not None:
            report["preparation_finalization_private"] = dict(
                self.preparation_finalization_private
            )
        if self.symbolic_failure_private is not None:
            report["symbolic_failure_private"] = dict(
                self.symbolic_failure_private
            )
        if self.articulation_result_private is not None:
            report["articulation_result_private"] = dict(self.articulation_result_private)
        if self.preparation_seconds_private is not None:
            report["phase_timing_private"] = {
                "version": "behavior2026.symbolic-preparation-timing/v1",
                "preparation_seconds": self.preparation_seconds_private,
                "symbolic_generator_seconds": (
                    self.symbolic_generator_seconds_private
                ),
                "outcome_verifier_seconds": (
                    self.outcome_verifier_seconds_private
                ),
                "finalization_seconds": self.finalization_seconds_private,
                "total_seconds": (
                    self.terminal_seconds_private
                    if self.terminal_seconds_private is not None
                    else perf_counter() - self.started_at_private
                ),
            }
        return report


@dataclass(frozen=True, slots=True)
class SymbolicPreparationResult:
    """Audit plus one action-lifetime commit / rollback callback."""

    report: Mapping[str, Any]
    finalize: Callable[[bool], Mapping[str, Any]] | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.report, Mapping):
            raise TypeError("preparation report must be a mapping")
        if self.finalize is not None and not callable(self.finalize):
            raise TypeError("preparation finalizer must be callable")
        object.__setattr__(self, "report", dict(self.report))


class MarkerSymbolicGraspBackendAdapter:
    """Bind C-labels to a frozen frame, then run only that object target."""

    api_contract_version = MARKER_SYMBOLIC_BACKEND_VERSION

    def __init__(
        self,
        *,
        marker_store: PrivateVisualMarkerStore,
        private_frame_provider: Callable[[], PrivateInstanceSegmentationFrame],
        raw_head_rgb_provider: Callable[[], np.ndarray],
        eligible_objects_provider: Callable[[], Mapping[str, Any]],
        graspable_object_names_provider: Callable[[], Sequence[str]],
        place_container_names_provider: Callable[[], Sequence[str]],
        official_bbox_provider: Callable[[PrivateInstanceSegmentationFrame], Sequence[Sequence[int]]],
        symbolic_generator_factory: Callable[[Any], Any],
        symbolic_place_generator_factory: Callable[[Any], Any],
        command_step: Callable[[Any], int],
        hold_safe: Callable[[], Any],
        local_reach_admission: Callable[[Any, str], tuple[bool, Mapping[str, Any]]]
        | None = None,
        local_prepare_factory: Callable[[Any, str], Any] | None = None,
        prepare_place_on_top: bool = False,
        automatic_interaction_approach_enabled: bool = False,
        placement_physical_consequences_enabled: bool = False,
        openable_object_names_provider: Callable[[], Sequence[str]] | None = None,
        support_object_names_provider: Callable[[], Sequence[str]] | None = None,
        operation_marker_objects_provider: Callable[
            [str], Mapping[str, Any]
        ]
        | None = None,
        under_marker_objects_provider: Callable[[Callable[[str, Any], bool]], Mapping[str, Any]] | None = None,
        on_top_snapshot_provider: Callable[..., PrivateMarkerSnapshot] | None = None,
        next_to_snapshot_provider: Callable[..., PrivateMarkerSnapshot] | None = None,
        toggleable_object_names_provider: Callable[[], Sequence[str]] | None = None,
        symbolic_operation_factories: Mapping[
            str, Callable[[Any | None], Any]
        ]
        | None = None,
        held_object_provider: Callable[[], Any | None] | None = None,
        private_outcome_verifier: Callable[
            [str, Any | None, Any | None, int], Mapping[str, Any]
        ]
        | None = None,
        pick_bundle_targets_provider: Callable[[], Sequence[PickBundleTargetToken]]
        | None = None,
        execution_integrity_state: Any | None = None,
        placement_rollback_factory: Callable[[Any], Callable[[], Mapping[str, Any]]] | None = None,
    ) -> None:
        if not isinstance(marker_store, PrivateVisualMarkerStore):
            raise TypeError("marker_store must be PrivateVisualMarkerStore")
        for callback, name in (
            (private_frame_provider, "private_frame_provider"),
            (raw_head_rgb_provider, "raw_head_rgb_provider"),
            (eligible_objects_provider, "eligible_objects_provider"),
            (graspable_object_names_provider, "graspable_object_names_provider"),
            (place_container_names_provider, "place_container_names_provider"),
            (official_bbox_provider, "official_bbox_provider"),
            (symbolic_generator_factory, "symbolic_generator_factory"),
            (symbolic_place_generator_factory, "symbolic_place_generator_factory"),
            (command_step, "command_step"),
            (hold_safe, "hold_safe"),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        self._store = marker_store
        self._frame = private_frame_provider
        self._raw_rgb = raw_head_rgb_provider
        self._eligible = eligible_objects_provider
        self._graspable_names = graspable_object_names_provider
        self._container_names = place_container_names_provider
        if openable_object_names_provider is not None and not callable(
            openable_object_names_provider
        ):
            raise TypeError("openable_object_names_provider must be callable")
        self._openable_names = openable_object_names_provider or (lambda: ())
        if support_object_names_provider is not None and not callable(
            support_object_names_provider
        ):
            raise TypeError("support_object_names_provider must be callable")
        self._support_names = support_object_names_provider or (lambda: ())
        if operation_marker_objects_provider is not None and not callable(
            operation_marker_objects_provider
        ):
            raise TypeError("operation_marker_objects_provider must be callable")
        self._operation_marker_objects = operation_marker_objects_provider
        self._under_marker_objects = under_marker_objects_provider
        if on_top_snapshot_provider is not None and not callable(
            on_top_snapshot_provider
        ):
            raise TypeError("on_top_snapshot_provider must be callable")
        self._on_top_snapshot = on_top_snapshot_provider
        if next_to_snapshot_provider is not None and not callable(
            next_to_snapshot_provider
        ):
            raise TypeError("next_to_snapshot_provider must be callable")
        self._next_to_snapshot = next_to_snapshot_provider
        if toggleable_object_names_provider is not None and not callable(
            toggleable_object_names_provider
        ):
            raise TypeError("toggleable_object_names_provider must be callable")
        self._toggleable_names = toggleable_object_names_provider or (lambda: ())
        self._official_bboxes = official_bbox_provider
        self._generator = symbolic_generator_factory
        self._place_generator = symbolic_place_generator_factory
        self._command_step = command_step
        self._hold_safe = hold_safe
        self._execution_integrity_state = execution_integrity_state
        self._placement_rollback_factory = placement_rollback_factory
        self._integrity_failure: str | None = None
        if local_reach_admission is not None and not callable(
            local_reach_admission
        ):
            raise TypeError("local_reach_admission must be callable")
        self._local_reach_admission = local_reach_admission
        if local_prepare_factory is not None and not callable(
            local_prepare_factory
        ):
            raise TypeError("local_prepare_factory must be callable")
        if not isinstance(prepare_place_on_top, bool):
            raise TypeError("prepare_place_on_top must be boolean")
        if not isinstance(automatic_interaction_approach_enabled, bool):
            raise TypeError(
                "automatic_interaction_approach_enabled must be boolean"
            )
        if (
            automatic_interaction_approach_enabled
            and local_prepare_factory is None
        ):
            raise ValueError(
                "automatic interaction approach requires local preparation"
            )
        if not isinstance(placement_physical_consequences_enabled, bool):
            raise TypeError(
                "placement_physical_consequences_enabled must be boolean"
            )
        self._local_prepare = local_prepare_factory
        self._prepare_place_on_top = prepare_place_on_top
        self._automatic_interaction_approach_enabled = (
            automatic_interaction_approach_enabled
        )
        self._placement_physical_consequences_enabled = (
            placement_physical_consequences_enabled
        )
        operation_factories = dict(symbolic_operation_factories or {})
        allowed_operations = {
            "open",
            "close",
            "toggle_on",
            "toggle_off",
            "wipe",
            *HOUSEHOLD_ACTIONS,
            "put_back",
            "release",
            "place_on_floor",
            *_SPATIAL_OPERATIONS,
        }
        if set(operation_factories).difference(allowed_operations):
            raise ValueError("unknown v34 symbolic operation factory")
        if any(not callable(value) for value in operation_factories.values()):
            raise TypeError("symbolic operation factories must be callable")
        self._operation_factories = operation_factories
        if held_object_provider is not None and not callable(
            held_object_provider
        ):
            raise TypeError("held_object_provider must be callable")
        if private_outcome_verifier is not None and not callable(
            private_outcome_verifier
        ):
            raise TypeError("private_outcome_verifier must be callable")
        self._held_object = held_object_provider or (lambda: None)
        self._private_outcome_verifier = private_outcome_verifier
        if pick_bundle_targets_provider is not None and not callable(
            pick_bundle_targets_provider
        ):
            raise TypeError("pick_bundle_targets_provider must be callable")
        self._pick_bundle_targets = pick_bundle_targets_provider or (lambda: ())
        self._last_private_diagnostic: dict[str, Any] = {
            "stage": "idle",
        }

    def publish_visible_markers(self, *, action_index: int) -> int:
        self._require_integrity()
        self._last_private_diagnostic = {"stage": "visible_marker_publication", "action_index": action_index}
        frame = self._frame()
        if not frame.instance_id_to_name and not frame.part_id_to_prim_path:
            self._store.clear()
            self._last_private_diagnostic = {
                "stage": "marker_frame_unavailable",
            }
            return 0
        snapshot = build_private_marker_snapshot(
            frame=frame,
            public_head_rgb=self._raw_rgb(),
            eligible_objects_by_name=self._eligible(),
            official_bboxes=self._official_bboxes(frame),
            created_action_index=action_index,
        )
        self._store.publish(snapshot)
        self._last_private_diagnostic["candidate_count"] = len(snapshot.bindings)
        return len(snapshot.bindings)

    def publish_operation_markers(
        self, *, operation: str, action_index: int,
        reference_binding: PrivateMarkerBinding | None = None,
    ) -> int:
        """Own a fresh diagnostic for every publication, including early exits."""
        publication = {
            "version": "marker-publication/v1", "operation": operation,
            "action_index": action_index,
            "stage": ("region" if reference_binding is not None else
                      "reference" if operation in _TWO_STAGE_OPERATIONS else "target"),
            "target_private": getattr(reference_binding, "object_name", None),
            "candidate_count": None, "outcome": "UNKNOWN",
            "target_filters_private": [],
        }
        self._last_private_diagnostic = {"operation": operation, "stage": "marker_publication"}
        self._publication_report = publication
        try:
            count = self._publish_operation_markers(
                operation=operation, action_index=action_index, reference_binding=reference_binding,
            )
            publication.update(candidate_count=count, outcome="PUBLISHED" if count else "EMPTY")
            if not count:
                publication["empty_reason"] = {
                    "marker_frame_unavailable": "FRAME_UNAVAILABLE",
                    "reference_no_longer_visible": "REFERENCE_NOT_VISIBLE",
                    "reference_geometry_unavailable": "REFERENCE_GEOMETRY_UNAVAILABLE",
                }.get(self._last_private_diagnostic.get("stage"), "UNKNOWN")
                if reference_binding is not None:
                    publication["target_filters_private"].append({
                        "target_private": reference_binding.object_name,
                        "stage": publication["stage"], "reason": publication["empty_reason"],
                    })
            return count
        except Exception as exc:
            publication.update(outcome="ERROR", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            provider = getattr(self._operation_marker_objects, "publication_diagnostic", None)
            if isinstance(provider, Mapping) and provider.get("operation") == operation and publication.get("provider_called"):
                publication["provider_private"] = dict(provider)
            if operation == "place_on_top" and self._on_top_snapshot is not None and publication.get("surface_provider_called"):
                detail = getattr(self._on_top_snapshot, "publication_diagnostic", None)
                if isinstance(detail, Mapping) and detail.get("action_index") == action_index:
                    publication["surface_provider_private"] = dict(detail)
            filters = publication["target_filters_private"]
            publication["target_filter_count"] = len(filters)
            publication["target_filters_truncated"] = len(filters) > 200
            publication["target_filters_private"] = filters[:200]
            provider = publication.get("provider_private")
            if isinstance(provider, dict):
                rows = provider.get("filters_private", [])
                provider["filter_count"] = len(rows)
                provider["filters_truncated"] = len(rows) > 200
                provider["filters_private"] = rows[:200]
            self._last_private_diagnostic["marker_publication_private"] = publication
            self._publication_report = None

    def _publish_operation_markers(
        self,
        *,
        operation: str,
        action_index: int,
        reference_binding: PrivateMarkerBinding | None = None,
    ) -> int:
        """Publish only current-frame choices relevant to one Action20 call."""

        self._require_integrity()
        supported = {
            "pick",
            "place_inside",
            "place_on_top",
            "place_next_to",
            "place_under",
            "place_touching",
            "open",
            "close",
            "toggle_on",
            "toggle_off",
        }
        supported.update(self._operation_factories.keys() & {"wipe", *HOUSEHOLD_ACTIONS})
        if operation not in supported:
            raise MarkerSymbolicBackendError(
                f"operation-specific markers are unsupported for {operation!r}"
            )
        if operation not in _TWO_STAGE_OPERATIONS and reference_binding is not None:
            raise MarkerSymbolicBackendError(
                f"{operation} does not accept a reference binding"
            )
        if reference_binding is not None and not isinstance(
            reference_binding, PrivateMarkerBinding
        ):
            raise TypeError("reference_binding must be PrivateMarkerBinding")

        frame = self._frame()
        if not frame.instance_id_to_name and not frame.part_id_to_prim_path:
            self._store.clear()
            self._last_private_diagnostic = {
                "stage": "marker_frame_unavailable",
                "operation": operation,
            }
            return 0
        rgb = self._raw_rgb()
        if reference_binding is None:
            if operation == "place_under":
                # Mesh pixels can belong to nested prim roots. Keep partial masks
                # for the snapshot to resolve ownership across admitted objects.
                min_pixels = 1 if frame.part_segmentation is not None and frame.part_id_to_prim_path else 8

                def visible(name, target):
                    pixels, _ = current_target_visibility(
                        frame=frame, target=target, object_name=name, minimum_visible_pixels=1,
                    )
                    return pixels >= min_pixels

                marker_objects = self._operation_marker_objects_for(operation, include_under_target=visible)
            else:
                marker_objects = self._operation_marker_objects_for(operation)
            if operation == "place_on_top":
                self._publication_report["surface_provider_called"] = self._on_top_snapshot is not None
                arguments = {
                    "frame": frame,
                    "public_head_rgb": rgb,
                    "eligible_supports_by_name": marker_objects,
                    "official_bboxes": self._official_bboxes(frame),
                    "created_action_index": action_index,
                }
                snapshot = (
                    self._on_top_snapshot(**arguments)
                    if self._on_top_snapshot is not None
                    else build_private_on_top_support_snapshot(**arguments)
                )
            elif operation == "pick":
                snapshot = build_private_pick_snapshot(
                    frame=frame,
                    public_head_rgb=rgb,
                    eligible_objects_by_name=marker_objects,
                    bundle_targets=tuple(self._pick_bundle_targets()),
                    official_bboxes=self._official_bboxes(frame),
                    created_action_index=action_index,
                )
            else:
                snapshot = (
                    build_private_drawer_part_snapshot(
                        frame=frame,
                        public_head_rgb=rgb,
                        eligible_openables_by_name=marker_objects,
                        official_bboxes=self._official_bboxes(frame),
                        created_action_index=action_index,
                        operation=operation,
                        diagnostics=self._publication_report.setdefault('articulated_parts_private', []),
                    )
                    if operation in {"open", "close"}
                    else None
                )
                if operation in {"toggle_on", "toggle_off"}:
                    snapshot = build_private_toggle_part_snapshot(
                        frame=frame,
                        public_head_rgb=rgb,
                        eligible_toggleables_by_name=marker_objects,
                        created_action_index=action_index,
                        operation=operation,
                    )
                # The articulated snapshot already supplies whole-object
                # markers for single-part targets; empty means no safe part choice.
                if snapshot is None:
                    snapshot = build_private_marker_snapshot(
                        frame=frame,
                        public_head_rgb=rgb,
                        eligible_objects_by_name=marker_objects,
                        official_bboxes=self._official_bboxes(frame),
                        created_action_index=action_index,
                    )
                    snapshot = replace(
                        snapshot,
                        operation=operation,
                        stage=(
                            "reference"
                            if operation in _TWO_STAGE_OPERATIONS
                            else "target"
                        ),
                    )
        elif reference_binding.binding_kind != "object":
            raise MarkerSymbolicBackendError(
                "second-stage spatial markers require an object reference"
            )
        elif current_target_visibility(
            frame=frame,
            target=_base_marker_target(reference_binding.target_token),
            minimum_visible_pixels=1,
        )[0] == 0:
            self._store.clear()
            self._last_private_diagnostic = {
                "stage": "reference_no_longer_visible",
                "operation": operation,
            }
            return 0
        elif operation == "place_inside":
            snapshot = build_private_container_compartment_snapshot(
                frame=frame,
                public_head_rgb=rgb,
                reference_binding=reference_binding,
                created_action_index=action_index,
            )
        elif operation == "place_on_top":
            if self._on_top_snapshot is None:
                raise MarkerSymbolicBackendError(
                    "second-stage OnTop markers require a surface provider"
                )
            self._publication_report["surface_provider_called"] = True
            snapshot = self._on_top_snapshot(
                frame=frame,
                public_head_rgb=rgb,
                eligible_supports_by_name={
                    reference_binding.object_name: _base_marker_target(
                        reference_binding.target_token
                    )
                },
                official_bboxes=self._official_bboxes(frame),
                created_action_index=action_index,
                reference_binding=reference_binding,
            )
        elif operation == "place_next_to":
            arguments = {
                "frame": frame,
                "public_head_rgb": rgb,
                "reference_binding": reference_binding,
                "created_action_index": action_index,
            }
            try:
                snapshot = (
                    self._next_to_snapshot(**arguments)
                    if self._next_to_snapshot is not None
                    else build_private_next_to_position_snapshot(**arguments)
                )
            except MarkerInputUnavailableError:
                self._store.clear()
                self._last_private_diagnostic = {
                    "stage": "reference_geometry_unavailable",
                    "operation": operation,
                }
                return 0
        else:
            snapshot = build_private_touching_direction_snapshot(
                frame=frame,
                public_head_rgb=rgb,
                reference_binding=reference_binding,
                created_action_index=action_index,
                operation=operation,
            )
        self._store.publish(snapshot)
        publication = self._publication_report
        if operation in {'open', 'close'}:
            publication['bindings_private'] = [dict(marker=b.marker,object_name=b.object_name,
                binding_kind=b.binding_kind,link_prim_path=getattr(b.target_token,'link_prim_path',None),
                joint=getattr(b.target_token,'selector',None),visible_pixels=b.visible_pixels)
                for b in snapshot.bindings.values()]
        publication["stage"] = snapshot.stage or "target"
        if reference_binding is None:
            published_names = {binding.object_name for binding in snapshot.bindings.values()}
            for name in marker_objects:
                if name not in published_names:
                    try:
                        ids = [key for key, value in frame.instance_id_to_name.items() if value == name]
                        visible_pixels = int(np.count_nonzero(np.isin(frame.segmentation, ids)))
                    except Exception:
                        visible_pixels = None
                    object_pixels, mask_source = current_target_visibility(frame=frame, target=marker_objects[name])
                    publication["target_filters_private"].append({
                        "target_private": name, "stage": "snapshot", "reason": "UNKNOWN",
                        "instance_visible_pixels": visible_pixels,
                        "object_visible_pixels": object_pixels,
                        "object_mask_source_private": mask_source,
                        "visibility_evidence": "NO_OBJECT_PIXELS" if object_pixels == 0 else "OBJECT_PIXELS_PRESENT",
                    })
        if operation == "pick":
            sources = tuple(
                binding.candidate_source
                for binding in snapshot.bindings.values()
            )
            self._last_private_diagnostic = {
                "stage": "pick_marker_publication",
                "operation": "pick",
                "generic_candidate_count": sources.count("generic"),
                "relation_bundle_count": sources.count("relation_bundle"),
                "candidate_coverage_miss": (
                    "visual_fallback" in sources
                ),
            }
        elif operation == "place_on_top":
            if snapshot.stage == "reference":
                self._last_private_diagnostic = {
                    "stage": "place_on_top_support_publication",
                    "operation": "place_on_top",
                    "support_count": len(snapshot.bindings),
                }
            else:
                self._last_private_diagnostic = {
                    "stage": "place_on_top_surface_publication",
                    "operation": "place_on_top",
                    "surface_count": len(snapshot.bindings),
                }
        self._last_private_diagnostic["visual_fallback_count"] = sum(
            binding.candidate_source == "visual_fallback"
            for binding in snapshot.bindings.values()
        )
        return len(snapshot.bindings)

    def consume_marker(
        self, *, marker: str, action_index: int, operation: str
    ) -> PrivateMarkerResolution:
        frame = self._frame()
        if not frame.instance_id_to_name and not frame.part_id_to_prim_path:
            self._store.clear()
            self._last_private_diagnostic = {
                "stage": "marker_frame_unavailable",
                "operation": operation,
            }
            return PrivateMarkerResolution(MarkerStatus.UNKNOWN)
        active_snapshot = self._store.active_snapshot
        strict_spatial_context = (
            active_snapshot is not None
            and (
                active_snapshot.operation is not None
                or active_snapshot.stage is not None
            )
        )
        resolution = self._store.consume(
            marker=marker,
            action_index=action_index,
            raw_rgb=self._raw_rgb(),
            simulator_step=frame.simulator_step,
            operation=operation if strict_spatial_context else None,
            stage=(
                active_snapshot.stage
                if strict_spatial_context and active_snapshot is not None
                else None
            ),
            current_segmentation=(
                frame.segmentation if strict_spatial_context else None
            ),
        )
        if resolution.status is not MarkerStatus.READY:
            return resolution
        if (
            operation in _PLACEMENT_OR_RELEASE_OPERATIONS
            and _base_marker_target(resolution.binding.target_token)
            is self._held_object()
        ):
            self._last_private_diagnostic = {
                "stage": "self_target_admission",
                "operation": operation,
                "admitted": False,
            }
            return PrivateMarkerResolution(
                MarkerStatus.UNKNOWN,
                public_failure_reason=MarkerPublicFailureReason.SELECTION_UNAVAILABLE,
            )
        if operation in {"grasp", "pick"}:
            admitted = set(self._graspable_names())
        elif operation == "place_inside":
            admitted = set(self._container_names())
        elif operation in {"open", "close"}:
            admitted = set(self._openable_names())
        elif operation == "place_on_top":
            admitted = set(self._support_names())
        elif operation == "place_under":
            binding = resolution.binding
            admitted = self._operation_marker_objects_for(
                operation,
                include_under_target=lambda name, obj: (name == binding.object_name and obj is binding.target_token),
            )
        elif operation in _SPATIAL_OPERATIONS or operation in {"wipe", *HOUSEHOLD_ACTIONS}:
            admitted = set(self._operation_marker_objects_for(operation))
        elif operation in {"toggle_on", "toggle_off"}:
            admitted = set(self._toggleable_names())
        else:
            raise MarkerSymbolicBackendError(f"unknown marker operation {operation!r}")
        if resolution.binding.object_name not in admitted:
            self._last_private_diagnostic = {
                "stage": "role_admission",
                "operation": operation,
                "admitted": False,
            }
            return PrivateMarkerResolution(
                MarkerStatus.UNKNOWN,
                public_failure_reason=MarkerPublicFailureReason.SELECTION_UNAVAILABLE,
            )
        is_reference_selection = (
            operation in _TWO_STAGE_OPERATIONS
            and active_snapshot is not None
            and active_snapshot.stage == "reference"
        )
        if (
            self._local_reach_admission is not None
            and not is_reference_selection
        ):
            admission_target = _base_marker_target(
                resolution.binding.target_token
            )
            try:
                admitted_local, diagnostic = self._local_reach_admission(
                    admission_target,
                    operation,
                )
            except MarkerInputUnavailableError:
                self._store.clear()
                self._last_private_diagnostic = {
                    "stage": "local_reach_admission",
                    "operation": operation,
                    "admitted": False,
                    "target_geometry_available": False,
                }
                return PrivateMarkerResolution(
                    MarkerStatus.UNKNOWN,
                    public_failure_reason=MarkerPublicFailureReason.EXECUTION_UNVERIFIED,
                )
            if not isinstance(admitted_local, bool) or not isinstance(
                diagnostic, Mapping
            ):
                raise MarkerSymbolicBackendError(
                    "local reach admission returned an invalid result"
                )
            self._last_private_diagnostic = {
                "stage": "local_reach_admission",
                "operation": operation,
                "admitted": bool(
                    admitted_local
                    or (
                        self._automatic_interaction_approach_enabled
                        and operation
                        in AUTOMATIC_INTERACTION_APPROACH_OPERATIONS
                    )
                ),
                "direct_local_reach_admitted": admitted_local,
                "automatic_interaction_approach_required": bool(
                    not admitted_local
                    and self._automatic_interaction_approach_enabled
                    and operation
                    in AUTOMATIC_INTERACTION_APPROACH_OPERATIONS
                ),
                **dict(diagnostic),
            }
            if not admitted_local and not (
                self._automatic_interaction_approach_enabled
                and operation in AUTOMATIC_INTERACTION_APPROACH_OPERATIONS
            ):
                return PrivateMarkerResolution(
                    MarkerStatus.UNKNOWN,
                    public_failure_reason=(
                        MarkerPublicFailureReason.TARGET_OUTSIDE_LOCAL_WORKSPACE
                    ),
                )
        return resolution

    def _operation_marker_objects_for(
        self, operation: str, *, include_under_target: Callable[[str, Any], bool] | None = None,
    ) -> dict[str, Any]:
        scoped_under = include_under_target is not None and self._under_marker_objects is not None
        if scoped_under or self._operation_marker_objects is not None:
            if getattr(self, "_publication_report", None) is not None:
                self._publication_report["provider_called"] = True
            if scoped_under:
                eligible = dict(self._under_marker_objects(include_under_target))
            else:
                eligible = dict(self._operation_marker_objects(operation))
            report = getattr(self._operation_marker_objects, "publication_diagnostic", None)
            if isinstance(report, Mapping) and getattr(self, "_publication_report", None) is not None:
                self._publication_report["provider_private"] = dict(report)
            if any(
                not isinstance(name, str) or not name or token is None
                for name, token in eligible.items()
            ):
                raise MarkerSymbolicBackendError(
                    "operation marker provider returned invalid live objects"
                )
        else:
            eligible = dict(self._eligible())
        if include_under_target is not None and not scoped_under:
            eligible = {name: obj for name, obj in eligible.items() if include_under_target(name, obj)}
        if operation == "pick":
            admitted_names = set(self._graspable_names())
        elif operation == "place_inside":
            admitted_names = set(self._container_names())
        elif operation in {"open", "close"}:
            admitted_names = set(self._openable_names())
        elif operation == "place_on_top":
            admitted_names = set(self._support_names())
        elif operation in _SPATIAL_OPERATIONS or operation in {"wipe", *HOUSEHOLD_ACTIONS}:
            admitted_names = set(eligible)
        elif operation in {"toggle_on", "toggle_off"}:
            admitted_names = set(self._toggleable_names())
        else:
            raise MarkerSymbolicBackendError(
                f"unsupported operation marker role {operation!r}"
            )
        held = (
            self._held_object()
            if operation in _PLACEMENT_OR_RELEASE_OPERATIONS
            else None
        )
        admitted = {}
        publication = getattr(self, "_publication_report", None)
        for name, token in eligible.items():
            reason = (
                "ROLE_NOT_ADMITTED" if name not in admitted_names else
                "HELD_OBJECT" if token is held else
                "NO_OFFICIAL_FILLABLE_REGION" if operation == "place_inside"
                and not _has_official_fillable_meta_link(_base_marker_target(token)) else None
            )
            if reason is None:
                admitted[name] = token
            elif publication is not None:
                publication["target_filters_private"].append({"target_private": name, "stage": "role", "reason": reason})
        if publication is not None:
            publication.update(input_candidate_count=len(eligible), role_candidate_count=len(admitted))
        return admitted

    def private_diagnostic(self) -> dict[str, Any]:
        """Return executor-only admission evidence for artifact auditing."""

        return dict(self._last_private_diagnostic)

    def begin_symbolic_grasp(self, *, target_token: Any) -> _SymbolicToken:
        if target_token is None:
            raise MarkerSymbolicBackendError("marker target token is missing")
        return self._begin(target_token, "grasp", self._generator)

    def tick_symbolic_grasp(self, state_token: Any) -> SymbolicGraspTick:
        self._require_integrity()
        token = self._require_token(state_token)
        if token.terminal:
            raise MarkerSymbolicBackendError("symbolic grasp is already terminal")
        if token.preparation is not None:
            try:
                steps = next(token.preparation)
            except StopIteration as stop:
                token.preparation = None
                token.preparation_seconds_private = (
                    perf_counter() - token.started_at_private
                )
                try:
                    self._accept_preparation_result(token, stop.value)
                except Exception as exc:
                    return self._fail_token(token, exc)
                token.phase = "prepared"
                return SymbolicGraspTick(SymbolicGraspStatus.RUNNING, 0)
            except Exception as exc:
                return self._fail_token(token, exc)
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                error = MarkerSymbolicBackendError(
                    "local preparation returned invalid step count"
                )
                preparation = token.preparation
                token.preparation = None
                close = getattr(preparation, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as exc:
                        token.terminal = True
                        token.phase = "rollback_failed"
                        token.error_type = type(exc).__name__
                        token.error = str(exc)
                        self._halt_integrity_failure(token, exc, message="local preparation cleanup failed")
                return self._fail_token(token, error)
            token.preparation_steps += steps
            token.simulator_steps += steps
            return SymbolicGraspTick(SymbolicGraspStatus.RUNNING, steps)
        if token.generator is None:
            factory = token.generator_factory
            if factory is None:
                return self._fail_token(
                    token,
                    MarkerSymbolicBackendError(
                        "prepared symbolic operation lacks a generator factory"
                    ),
                )
            try:
                token.symbolic_started_at_private = perf_counter()
                token.symbolic_generator_start_count += 1
                token.generator = iter(factory(token.target_token))
            except Exception as exc:
                return self._fail_token(token, exc)
            token.generator_factory = None
            token.phase = "symbolic"
            return SymbolicGraspTick(SymbolicGraspStatus.RUNNING, 0)
        try:
            if not token.placement_boundary_started:
                token.placement_boundary_started = True
                capture = getattr(self, "_placement_rollback_factory", None)
                if (
                    capture is not None
                    and not self._placement_physical_consequences_enabled
                    and token.operation in {
                        "place_inside", "place_on_top", "place_on_floor",
                        "place_next_to", "place_under",
                    }
                    and token.preparation_finalize is None
                ):
                    # Keep the existing checkpoint alive through backend attestation.
                    # Prepared operations already own an outer transaction.
                    token.placement_rollback = capture(token.target_token)
            command = next(token.generator)
        except StopIteration as finished:
            self._require_integrity()
            if token.operation in {"open", "close"} and isinstance(finished.value, Mapping):
                token.articulation_result_private = dict(finished.value)
            if token.symbolic_started_at_private is not None:
                token.symbolic_generator_seconds_private = (
                    perf_counter() - token.symbolic_started_at_private
                )
            physical_consequence_committed = False
            try:
                if token.operation in {"wipe", *HOUSEHOLD_ACTIONS} or self._private_outcome_verifier is not None:
                    outcome_verifier_started = perf_counter()
                    if token.operation in {"wipe", *HOUSEHOLD_ACTIONS}:
                        report = finished.value
                    else:
                        report = self._private_outcome_verifier(
                            str(token.operation),
                            token.target_token,
                            token.acted_object,
                            token.simulator_steps,
                        )
                    if not isinstance(report, Mapping) or not isinstance(
                        report.get("semantic_state_pass"), bool
                    ):
                        raise MarkerSymbolicBackendError(
                            "private outcome verifier returned an invalid report"
                        )
                    token.outcome_verifier_private = dict(report)
                    self._last_private_diagnostic = {
                        **dict(self._last_private_diagnostic),
                        "outcome_verifier_private": dict(report),
                    }
                    physical_consequence_committed = bool(
                        not report["semantic_state_pass"]
                        and report.get("physical_consequence_committed") is True
                        and self._placement_physical_consequences_enabled
                        and token.operation
                        in _PHYSICAL_CONSEQUENCE_PLACEMENT_OPERATIONS
                        and token.acted_object is not None
                        and self._held_object() is None
                    )
                    if physical_consequence_committed:
                        token.outcome_verifier_private[
                            "physical_consequence_committed"
                        ] = True
                        self._last_private_diagnostic[
                            "outcome_verifier_private"
                        ] = dict(token.outcome_verifier_private)
                    elif not report["semantic_state_pass"]:
                        raise Action23PostconditionError(
                            "canonical action postcondition is false"
                        )
                    token.outcome_verifier_seconds_private = (
                        perf_counter() - outcome_verifier_started
                    )
            except Exception as exc:
                return self._fail_token(token, exc)
            if (
                token.operation in _PLACEMENT_OR_RELEASE_OPERATIONS
                and self._held_object() is not None
            ):
                return self._fail_token(
                    token, Action23PostconditionError("canonical carry remained after release")
                )
            finalization_started = perf_counter()
            try:
                self._finalize_preparation(token, success=True)
            except BaseException as exc:
                token.finalization_seconds_private = (
                    perf_counter() - finalization_started
                )
                self._record_finalization_exception_report(token, exc)
                private_report = getattr(exc, "private_report", None)
                rollback_report = (
                    private_report.get("rollback_private")
                    if isinstance(private_report, Mapping)
                    else None
                )
                restore_attested = bool(
                    isinstance(private_report, Mapping)
                    and private_report.get("status") == "commit_rejected"
                    and _verified_preparation_rollback(rollback_report)
                    and token.acted_object is not None
                    and self._held_object() is token.acted_object
                )
                token.terminal = True
                token.error_type = type(exc).__name__
                token.error = str(exc)
                if not restore_attested:
                    self._halt_integrity_failure(token, exc, message="preparation commit finalizer failed")
                self._hold_safe()
                if (restore_attested and type(exc) is Action20CommitRejectedError
                    and token.operation == "place_on_top"
                    and private_report.get("base_stayed_at_prepared_pose") is False):
                    token.phase = "blocked"
                    token.terminal_seconds_private = (
                        perf_counter() - token.started_at_private
                    )
                    return SymbolicGraspTick(
                        SymbolicGraspStatus.BLOCKED, 0,
                        blocked_category=EXECUTION_UNVERIFIED,
                    )
                token.phase = "commit_failed"
                raise MarkerSymbolicBackendError(
                    "preparation commit finalizer failed"
                ) from exc
            token.finalization_seconds_private = (
                perf_counter() - finalization_started
            )
            token.terminal = True
            token.phase = (
                "physical_consequence_committed"
                if physical_consequence_committed
                else "succeeded"
            )
            token.terminal_seconds_private = (
                perf_counter() - token.started_at_private
            )
            return SymbolicGraspTick(SymbolicGraspStatus.SUCCEEDED, 0)
        except GeneratorExit:
            self._finalize_preparation(token, success=False)
            raise
        except Exception as exc:
            return self._fail_token(token, exc)
        try:
            steps = self._command_step(command)
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                raise MarkerSymbolicBackendError(
                    "command_step returned invalid count"
                )
        except ResidentEvaluatorTerminalError:
            raise
        except Exception as exc:
            return self._fail_token(token, exc)
        token.simulator_steps += steps
        return SymbolicGraspTick(SymbolicGraspStatus.RUNNING, steps)

    def abort_symbolic_grasp(self, state_token: Any) -> SymbolicGraspTick:
        token = self._require_token(state_token)
        # A terminal evaluator step can interrupt command_step while the
        # official generator is suspended at its settle yield.  Close both
        # iterators explicitly so generator ``finally`` cleanup runs before the
        # action lifecycle is declared closed.  This is zero-step cleanup.
        close_error: BaseException | None = None
        for iterator in (token.preparation, token.generator):
            close = getattr(iterator, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as exc:
                close_error = exc
                break
        token.preparation = None
        token.generator = None
        token.generator_factory = None
        try:
            self._finalize_preparation(token, success=False)
        except BaseException as exc:
            self._record_finalization_exception_report(token, exc)
            token.terminal = True
            token.phase = "rollback_failed"
            token.error_type = type(exc).__name__
            token.error = str(exc)
            self._halt_integrity_failure(token, exc)
        if close_error is not None:
            token.terminal = True
            token.phase = "rollback_failed"
            token.error_type = type(close_error).__name__
            token.error = str(close_error)
            self._halt_integrity_failure(token, close_error)
        self._hold_safe()
        token.terminal = True
        token.phase = "aborted"
        return SymbolicGraspTick(SymbolicGraspStatus.ABORTED, 0)

    def begin_symbolic_place_inside(self, *, target_token: Any) -> _SymbolicToken:
        if target_token is None:
            raise MarkerSymbolicBackendError("marker container token is missing")
        return self._begin(
            target_token,
            "place_inside",
            self._place_generator,
            lazy_after_prepare=self._automatic_interaction_approach_enabled,
        )

    def tick_symbolic_place_inside(self, state_token: Any) -> SymbolicGraspTick:
        return self.tick_symbolic_grasp(state_token)

    def abort_symbolic_place_inside(self, state_token: Any) -> SymbolicGraspTick:
        return self.abort_symbolic_grasp(state_token)

    def begin_symbolic_operation(
        self,
        *,
        operation: str,
        target_token: Any | None,
    ) -> _SymbolicToken:
        self._last_private_diagnostic = {
            "stage": "symbolic_operation",
            "operation": operation,
            "direct_state_modification": (
                operation in {"open", "close", "put_back", "place_on_floor"}
                or operation in _SPATIAL_OPERATIONS
            ),
            "direct_state_modification_scope": (
                "private_pre_pick_pose_restore"
                if operation == "put_back"
                else "collision_guarded_official_open_endpoint"
                if operation == "open"
                else "collision_guarded_official_close_endpoint"
                if operation == "close"
                else "validated_p5_spatial_kernel"
                if operation == "place_on_floor"
                or operation in _SPATIAL_OPERATIONS
                else "official_symbolic_primitive"
            ),
            "kinematic_research": True,
            "official_challenge_eligible": False,
        }
        if operation == "pick":
            if target_token is None:
                raise MarkerSymbolicBackendError("marker target token is missing")
            return self._begin(
                target_token,
                "pick",
                self._generator,
                lazy_after_prepare=(
                    self._automatic_interaction_approach_enabled
                ),
            )
        if operation == "place_inside":
            return self.begin_symbolic_place_inside(target_token=target_token)
        try:
            factory = self._operation_factories[operation]
        except KeyError as exc:
            raise MarkerSymbolicBackendError(
                f"unsupported v34 symbolic operation {operation!r}"
            ) from exc
        if operation not in {
            "put_back",
            "release",
            "place_on_floor",
        } and target_token is None:
            raise MarkerSymbolicBackendError("marker target token is missing")
        if operation == "place_on_top":
            return self._begin(
                target_token,
                operation,
                factory,
                prepare=(
                    self._prepare_place_on_top
                    or self._automatic_interaction_approach_enabled
                ),
                lazy_after_prepare=True,
            )
        if (
            self._automatic_interaction_approach_enabled
            and operation in AUTOMATIC_INTERACTION_APPROACH_OPERATIONS
        ):
            return self._begin(
                target_token,
                operation,
                factory,
                lazy_after_prepare=True,
            )
        return _SymbolicToken(
            generator=iter(factory(target_token)),
            preparation=None,
            operation=operation,
            target_token=target_token,
            acted_object=(
                self._held_object()
                if operation in _PLACEMENT_OR_RELEASE_OPERATIONS
                else None
            ),
            phase="symbolic",
        )

    def tick_symbolic_operation(self, state_token: Any) -> SymbolicGraspTick:
        return self.tick_symbolic_grasp(state_token)

    def abort_symbolic_operation(self, state_token: Any) -> SymbolicGraspTick:
        return self.abort_symbolic_grasp(state_token)

    @staticmethod
    def _require_token(value: Any) -> _SymbolicToken:
        if not isinstance(value, _SymbolicToken):
            raise MarkerSymbolicBackendError("invalid symbolic grasp token")
        return value

    def _begin(
        self,
        target_token: Any,
        operation: str,
        generator_factory: Callable[[Any], Any],
        *,
        prepare: bool = True,
        lazy_after_prepare: bool = False,
    ) -> _SymbolicToken:
        self._require_integrity()
        preparation = (
            None
            if self._local_prepare is None or not prepare
            else iter(self._local_prepare(target_token, operation))
        )
        lazy = bool(preparation is not None and lazy_after_prepare)
        return _SymbolicToken(
            generator=(None if lazy else iter(generator_factory(target_token))),
            preparation=preparation,
            generator_factory=(generator_factory if lazy else None),
            operation=operation,
            target_token=target_token,
            acted_object=(
                self._held_object()
                if operation in _PLACEMENT_OR_RELEASE_OPERATIONS
                else None
            ),
            phase="symbolic" if preparation is None else "local_prepare",
        )

    def _require_integrity(self) -> None:
        failure = self._integrity_failure or execution_integrity_failure(self._execution_integrity_state)
        if failure is not None:
            raise MarkerSymbolicBackendError(f"execution stopped after integrity failure: {failure}")

    def _halt_integrity_failure(self, token: _SymbolicToken, error: BaseException, *, message: str = "execution integrity failure"):
        self._integrity_failure = f"{message}: {type(error).__name__}: {error}"
        if self._execution_integrity_state is not None:
            invalidate_execution(self._execution_integrity_state, error)
        token.terminal = True
        token.phase = "rollback_failed"
        token.error_type = type(error).__name__
        token.error = self._integrity_failure
        self._store.clear()
        try:
            self._hold_safe()
        finally:
            raise MarkerSymbolicBackendError(self._integrity_failure) from error

    def _fail_token(
        self,
        token: _SymbolicToken,
        exc: Exception,
    ) -> SymbolicGraspTick:
        boundary_rollback = token.placement_rollback is not None
        private_report = getattr(exc, "private_report", None)
        if not isinstance(private_report, Mapping) and type(exc).__name__ == "ActionPrimitiveErrorGroup":
            # apply_ref wraps the current primitive error after its cleanup.
            # Preserve every attempt as evidence, not as boundary rollback proof.
            attempts = getattr(exc, "exceptions", ())
            if isinstance(attempts, (tuple, list)):
                private_report = {
                    "primitive_attempt_failures_private": [
                        {
                            "error_type": type(attempt).__name__,
                            "error": str(attempt),
                            "private_report": (
                                dict(attempt.private_report)
                                if isinstance(getattr(attempt, "private_report", None), Mapping)
                                else None
                            ),
                        }
                        for attempt in attempts
                    ],
                }
        if isinstance(private_report, Mapping):
            key = (
                "preparation_report_private"
                if token.phase == "local_prepare"
                else "symbolic_failure_private"
            )
            copied = dict(private_report)
            if key == "preparation_report_private":
                token.preparation_report_private = copied
            else:
                token.symbolic_failure_private = copied
            self._last_private_diagnostic = {
                **dict(self._last_private_diagnostic),
                key: copied,
            }
        close_error: BaseException | None = None
        for iterator in (token.preparation, token.generator):
            close = getattr(iterator, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as iterator_exc:
                close_error = iterator_exc
                break
        token.preparation = None
        token.generator = None
        token.generator_factory = None
        try:
            preserve_articulation = bool(
                token.operation in {"open", "close"}
                and isinstance(private_report, Mapping)
                and private_report.get("last_safe_state_verified") is True
                and private_report.get("partial_progress_preserved") is True
                and getattr(exc, "blocked_category", None) != EXECUTION_UNVERIFIED
            )
            self._finalize_preparation(token, success=preserve_articulation)
        except BaseException as finalize_exc:
            self._record_finalization_exception_report(token, finalize_exc)
            token.terminal = True
            token.phase = "rollback_failed"
            token.error_type = type(finalize_exc).__name__
            token.error = (
                "preparation rollback failed after "
                f"{type(exc).__name__}: {exc}: "
                f"{type(finalize_exc).__name__}: {finalize_exc}"
            )
            self._halt_integrity_failure(token, finalize_exc)
        if close_error is not None:
            token.terminal = True
            token.phase = "rollback_failed"
            token.error_type = type(close_error).__name__
            token.error = (
                "symbolic iterator cleanup failed after "
                f"{type(exc).__name__}: {exc}: "
                f"{type(close_error).__name__}: {close_error}"
            )
            self._halt_integrity_failure(token, close_error)
        if isinstance(exc, ActionIntegrityError) or execution_integrity_failure(self._execution_integrity_state) is not None:
            self._halt_integrity_failure(token, exc)
        token.terminal = True
        token.phase = "failed"
        token.error_type = type(exc).__name__
        token.error = str(exc)
        self._hold_safe()
        blocked_category = getattr(exc, "blocked_category", None)
        if (
            blocked_category is None
            and isinstance(exc, Action23PostconditionError)
            and token.operation in _RECOVERABLE_PLACEMENT_OPERATIONS
            and isinstance(token.outcome_verifier_private, Mapping)
            and token.outcome_verifier_private.get("predicate_mode") == "action20_symbolic"
            and token.outcome_verifier_private.get("semantic_state_pass") is False
        ):
            blocked_category = RELATION_NOT_REALIZED
        metadata = getattr(exc, "metadata", None)
        if blocked_category is None and isinstance(metadata, Mapping):
            blocked_category = metadata.get(
                ACTION20_BLOCKED_CATEGORY_METADATA_KEY
            )
        if (
            not isinstance(blocked_category, str)
            or blocked_category not in _TRUSTED_ACTION20_BLOCKED_CATEGORIES
        ):
            blocked_category = None
        placement_restored = bool(
            token.operation in _RECOVERABLE_PLACEMENT_OPERATIONS
            and token.acted_object is not None
            and self._held_object() is token.acted_object
        )
        if (
            blocked_category is None and token.operation == "place_inside"
            and placement_restored and type(exc).__name__ == "ActionPrimitiveErrorGroup"
            and not self._placement_physical_consequences_enabled
            and (boundary_rollback or _verified_preparation_rollback(token.preparation_finalization_private))
        ):
            # The boundary finalizer above verified restoration; children only supply the cause.
            attempts = getattr(exc, "exceptions", ())
            if isinstance(attempts, (tuple, list)) and attempts and all(
                type(attempt).__name__ == "ActionPrimitiveError"
                and isinstance(getattr(attempt, "metadata", None), Mapping)
                and attempt.metadata.get(ACTION20_BLOCKED_CATEGORY_METADATA_KEY) == PLACEMENT_COLLISION_BLOCKED
                for attempt in attempts
            ):
                blocked_category = PLACEMENT_COLLISION_BLOCKED
        action_primitive_reason = getattr(
            getattr(exc, "reason", None),
            "name",
            None,
        )
        deterministic_blocked = bool(
            (
                blocked_category in ACTION20_BLOCKED_CATEGORIES
                and blocked_category != EXECUTION_UNVERIFIED
            )
            or (
                blocked_category is None
                and placement_restored
                and type(exc).__name__ == "ActionPrimitiveError"
                and action_primitive_reason
                in _DETERMINISTIC_ACTION_PRIMITIVE_REASONS
            )
        )
        if deterministic_blocked:
            token.phase = "blocked"
        return SymbolicGraspTick(
            (
                SymbolicGraspStatus.BLOCKED
                if deterministic_blocked
                else SymbolicGraspStatus.FAILED
            ),
            0,
            blocked_category=(
                blocked_category
                if blocked_category in _TRUSTED_ACTION20_BLOCKED_CATEGORIES
                else None
            ),
            blocked_detail=action20_blocked_detail(blocked_category, getattr(exc, "blocked_detail", None)),
        )

    def _accept_preparation_result(
        self,
        token: _SymbolicToken,
        result: Any,
    ) -> None:
        if result is None:
            return
        if isinstance(result, SymbolicPreparationResult):
            report = dict(result.report)
            token.preparation_finalize = result.finalize
        elif isinstance(result, Mapping):
            report = dict(result)
        else:
            raise MarkerSymbolicBackendError(
                "local preparation returned an invalid terminal result"
            )
        token.preparation_report_private = report
        self._last_private_diagnostic = {
            **dict(self._last_private_diagnostic),
            "preparation_report_private": report,
        }

    def _finalize_preparation(
        self,
        token: _SymbolicToken,
        *,
        success: bool,
    ) -> None:
        rollback = token.placement_rollback
        token.placement_rollback = None
        if rollback is not None and not success:
            report = rollback()
            previous_failure = token.symbolic_failure_private
            token.symbolic_failure_private = {"rollback_private": dict(report) if isinstance(report, Mapping) else None}
            if previous_failure is not None:
                # Preserve the original cause without promoting its pre-rollback
                # fields into the authoritative boundary rollback namespace.
                token.symbolic_failure_private["failure_before_boundary_rollback_private"] = dict(previous_failure)
            if not isinstance(report, Mapping) or report.get("accepted") is not True:
                raise ActionIntegrityError("placement boundary rollback was not verified")
        callback = token.preparation_finalize
        if callback is None:
            return
        # Clear before invoking so success, failure, abort, and rollback-error
        # paths can never run the action transaction twice.
        token.preparation_finalize = None
        report = callback(bool(success))
        if not isinstance(report, Mapping):
            raise MarkerSymbolicBackendError(
                "preparation finalizer returned an invalid report"
            )
        finalized = dict(report)
        token.preparation_finalization_private = finalized
        self._last_private_diagnostic = {
            **dict(self._last_private_diagnostic),
            "preparation_finalization_private": finalized,
        }

    def _record_finalization_exception_report(
        self,
        token: _SymbolicToken,
        exc: BaseException,
    ) -> None:
        report = getattr(exc, "private_report", None)
        if not isinstance(report, Mapping):
            return
        finalized = dict(report)
        token.preparation_finalization_private = finalized
        self._last_private_diagnostic = {
            **dict(self._last_private_diagnostic),
            "preparation_finalization_private": finalized,
        }


def official_instance_bboxes(
    frame: PrivateInstanceSegmentationFrame,
) -> list[tuple[int, int, int, int, int]]:
    """Call BEHAVIOR 2026's official ``instance_to_bbox`` lazily."""

    import torch as th
    from omnigibson.eval.utils.obs_utils import instance_to_bbox

    ids = [int(value) for value in np.unique(frame.segmentation)]
    result = instance_to_bbox(
        th.as_tensor(np.asarray(frame.segmentation), dtype=th.int64),
        dict(frame.instance_id_to_name),
        ids,
    )
    if not isinstance(result, list) or len(result) != 1:
        raise MarkerSymbolicBackendError("official instance_to_bbox returned invalid output")
    return [tuple(int(value) for value in box) for box in result[0]]
