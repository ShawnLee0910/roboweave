"""Private same-frame bindings for temporary public visual markers.

Primary boxes come from the live evaluator's ``instance_to_bbox``. Missing
boxes are recovered only from the same frame's instance or visual-mesh masks.
This module only assigns opaque labels and draws a non-occluding overlay.
"""

from __future__ import annotations

import hashlib
import random
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from behavior_hl.embodied.evaluator_bridge.logical_inventory import (
    LogicalInventoryRelation,
)
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
)
from behavior_hl.embodied.evaluator_bridge.private_drawer_parts import (
    drawer_part_tokens,
    drawer_fillable_aabbs,
    visible_drawer_parts,
)
from behavior_hl.embodied.evaluator_bridge.simple_spatial_markers import (
    generate_next_to_position_markers,
    generate_touching_direction_markers,
)
from behavior_hl.embodied.runtime.public_action_feedback import (
    EXECUTION_UNVERIFIED as EXECUTION_UNVERIFIED_REASON,
    SELECTION_UNAVAILABLE as SELECTION_UNAVAILABLE_REASON,
    TARGET_OUTSIDE_LOCAL_WORKSPACE as TARGET_OUTSIDE_LOCAL_WORKSPACE_REASON,
)


PRIVATE_VISUAL_MARKER_VERSION = "behavior2026.private-visual-marker/v11"
_MASK_SOURCES = frozenset(
    {"seg_instance", "seg_instance_id_prefix", "seg_instance_or_prefix"}
)
_BINDING_KINDS = frozenset(
    {
        "object",
        "bundle",
        "part",
        "inside_cell",
        "surface_cell",
        "position",
        "direction",
    }
)


class MarkerStatus(str, Enum):
    READY = "ready"
    UNKNOWN = "unknown"
    STALE = "stale"
    CONSUMED = "consumed"


class MarkerPublicFailureReason(str, Enum):
    """Small allowlist of non-directive marker failures safe for the model."""

    SELECTION_UNAVAILABLE = SELECTION_UNAVAILABLE_REASON
    TARGET_OUTSIDE_LOCAL_WORKSPACE = TARGET_OUTSIDE_LOCAL_WORKSPACE_REASON
    EXECUTION_UNVERIFIED = EXECUTION_UNVERIFIED_REASON


@dataclass(frozen=True, slots=True)
class PrivateMarkerBinding:
    marker: str
    object_name: str
    instance_id: int
    visible_pixels: int
    official_bbox_xyxy: tuple[int, int, int, int]
    target_token: Any = field(repr=False)
    binding_kind: str = "object"
    selector: str | None = None
    anchor_uv: tuple[int, int] | None = None
    polygon_uv: tuple[tuple[int, int], ...] | None = None
    mask_source: str = "seg_instance"
    member_anchors_uv: tuple[tuple[int, int], ...] | None = None
    candidate_source: str = "standard"

    def __post_init__(self) -> None:
        if self.mask_source not in _MASK_SOURCES:
            raise ValueError("unknown private marker mask source")
        if self.binding_kind not in _BINDING_KINDS:
            raise ValueError(
                "binding_kind must be object, bundle, part, inside_cell, "
                "surface_cell, position, or direction"
            )
        if not isinstance(self.candidate_source, str) or not self.candidate_source:
            raise ValueError("candidate_source must be a non-empty string")
        if self.binding_kind == "bundle":
            anchors = self.member_anchors_uv
            if anchors is None or len(anchors) < 2 or any(
                not isinstance(point, tuple)
                or len(point) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in point
                )
                for point in anchors
            ):
                raise ValueError("bundle marker requires at least two member anchors")
            return
        if self.binding_kind in {"object", "part"}:
            return
        if not isinstance(self.selector, str) or not self.selector:
            raise ValueError("spatial marker binding requires a selector")
        if (
            not isinstance(self.anchor_uv, tuple)
            or len(self.anchor_uv) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.anchor_uv)
            or any(value < 0 for value in self.anchor_uv)
        ):
            raise ValueError("spatial marker binding requires a non-negative integer anchor_uv")
        if self.polygon_uv is not None and (
            len(self.polygon_uv) < 3
            or any(
                not isinstance(point, tuple)
                or len(point) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in point
                )
                for point in self.polygon_uv
            )
        ):
            raise ValueError(
                "polygon_uv must contain at least three non-negative integer points"
            )


@dataclass(frozen=True, slots=True)
class OnTopSurfaceRegion:
    """One ray-verified horizontal support cell in image and world space."""

    object_name: str
    selector: str
    polygon_uv: tuple[tuple[int, int], ...]
    anchor_uv: tuple[int, int]
    region_center_xy: tuple[float, float]
    region_extent_xy: tuple[float, float]
    surface_z_m: float
    anchor_candidates_uv: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.object_name or not self.selector:
            raise ValueError("surface region identity must be non-empty")
        if len(self.polygon_uv) < 3:
            raise ValueError("surface region polygon must contain at least three points")
        for point in self.polygon_uv:
            if (
                not isinstance(point, tuple)
                or len(point) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in point
                )
            ):
                raise ValueError("surface region polygon points must be integer pairs")
        for anchor in (self.anchor_uv, *self.anchor_candidates_uv):
            if (
                not isinstance(anchor, tuple)
                or len(anchor) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in anchor
                )
            ):
                raise ValueError("surface region anchor must be an integer pair")
        center = np.asarray(self.region_center_xy, dtype=np.float64)
        extent = np.asarray(self.region_extent_xy, dtype=np.float64)
        if center.shape != (2,) or extent.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("surface region center and extent must be finite 2-vectors")
        if not np.isfinite(extent).all() or np.any(extent <= 0.0):
            raise ValueError("surface region extent must be finite and positive")
        if not np.isfinite(float(self.surface_z_m)):
            raise ValueError("surface region height must be finite")


@dataclass(frozen=True, slots=True)
class NextToSupportPosition:
    """One ray-verified nearby support point in image and world space."""

    selector: str
    anchor_uv: tuple[int, int]
    region_center_xy: tuple[float, float]
    region_extent_xy: tuple[float, float]


class SpatialMarkerTargetToken:
    """Opaque executor-only wrapper for one selected spatial proposal.

    The wrapped simulator token is deliberately absent from ``repr`` and the
    class rejects pickle serialization.  Public artifacts should retain only
    the corresponding C-label; an executor may privately inspect the target,
    selector, operation, and optional region after consuming a same-frame
    snapshot.
    """

    __slots__ = (
        "_target_token",
        "_operation",
        "_selector",
        "_region_center_xy",
        "_region_extent_xy",
        "_surface_z_m",
    )

    def __init__(
        self,
        target_token: Any,
        operation: str,
        selector: str,
        *,
        region_center_xy: Sequence[float] | None = None,
        region_extent_xy: Sequence[float] | None = None,
        surface_z_m: float | None = None,
    ) -> None:
        if target_token is None:
            raise ValueError("target_token must not be None")
        if not isinstance(operation, str) or not operation:
            raise ValueError("operation must be a non-empty string")
        if not isinstance(selector, str) or not selector:
            raise ValueError("selector must be a non-empty string")
        if (region_center_xy is None) != (region_extent_xy is None):
            raise ValueError("spatial marker region must be complete")
        if surface_z_m is not None and region_center_xy is None:
            raise ValueError("surface height requires a spatial marker region")
        center = None
        extent = None
        if region_center_xy is not None:
            center_array = np.asarray(region_center_xy, dtype=np.float64)
            extent_array = np.asarray(region_extent_xy, dtype=np.float64)
            if (
                center_array.shape != (2,)
                or extent_array.shape != (2,)
                or not np.isfinite(center_array).all()
                or not np.isfinite(extent_array).all()
                or np.any(extent_array <= 0.0)
            ):
                raise ValueError(
                    "spatial marker region must contain finite positive 2-vectors"
                )
            center = tuple(float(value) for value in center_array)
            extent = tuple(float(value) for value in extent_array)
        self._target_token = target_token
        self._operation = operation
        self._selector = selector
        self._region_center_xy = center
        self._region_extent_xy = extent
        self._surface_z_m = (
            None if surface_z_m is None else float(surface_z_m)
        )
        if self._surface_z_m is not None and not np.isfinite(self._surface_z_m):
            raise ValueError("surface height must be finite")

    @property
    def target_token(self) -> Any:
        return self._target_token

    @property
    def operation(self) -> str:
        return self._operation

    @property
    def selector(self) -> str:
        return self._selector

    @property
    def region_center_xy(self) -> tuple[float, float] | None:
        return self._region_center_xy

    @property
    def region_extent_xy(self) -> tuple[float, float] | None:
        return self._region_extent_xy

    @property
    def surface_z_m(self) -> float | None:
        return self._surface_z_m

    def __repr__(self) -> str:
        return "<SpatialMarkerTargetToken redacted>"

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("private spatial marker tokens must not be serialized")


@dataclass(frozen=True, slots=True)
class PickBundleTargetToken:
    """Opaque root plus the exact live relations selected by one C-marker."""

    root_token: Any = field(repr=False)
    relations: tuple[LogicalInventoryRelation, ...] = field(repr=False)
    exclusive: bool = False

    def __post_init__(self) -> None:
        if self.root_token is None:
            raise ValueError("bundle root must not be None")
        relations = tuple(self.relations)
        if not relations or any(
            not isinstance(relation, LogicalInventoryRelation)
            or relation.kind not in {"on_top", "attached_to"}
            for relation in relations
        ):
            raise ValueError("pick bundle requires OnTop or AttachedTo relations")
        if not isinstance(self.exclusive, bool):
            raise TypeError("exclusive must be boolean")
        members = self.member_tokens
        if len(members) < 2 or all(
            relation.child is not self.root_token
            and relation.parent is not self.root_token
            for relation in relations
        ):
            raise ValueError("bundle relations must include their root")
        connected = {id(self.root_token)}
        changed = True
        while changed:
            changed = False
            for relation in relations:
                child_id = id(relation.child)
                parent_id = id(relation.parent)
                if parent_id in connected and child_id not in connected:
                    connected.add(child_id)
                    changed = True
                if child_id in connected and parent_id not in connected:
                    connected.add(parent_id)
                    changed = True
        if len(connected) != len(members):
            raise ValueError("bundle relations must form one connected component")
        object.__setattr__(self, "relations", relations)

    @property
    def target_token(self) -> Any:
        return self.root_token

    @property
    def member_tokens(self) -> tuple[Any, ...]:
        ordered = [self.root_token]
        seen = {id(self.root_token)}
        for relation in self.relations:
            for member in (relation.parent, relation.child):
                if id(member) in seen:
                    continue
                seen.add(id(member))
                ordered.append(member)
        return tuple(ordered)

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(str(getattr(member, "name", "")) for member in self.member_tokens)

    def __repr__(self) -> str:
        return "<PickBundleTargetToken redacted>"

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("private pick bundle tokens must not be serialized")


@dataclass(frozen=True, slots=True)
class PrivateMarkerSnapshot:
    camera: str
    simulator_step: int
    source_rgb_sha256: str
    created_action_index: int
    marked_rgb: np.ndarray = field(repr=False)
    bindings: Mapping[str, PrivateMarkerBinding] = field(repr=False)
    operation: str | None = None
    stage: str | None = None
    source_segmentation_sha256: str | None = None

    def __post_init__(self) -> None:
        marked = np.asarray(self.marked_rgb)
        if marked.ndim != 3 or marked.shape[2] != 3 or marked.dtype != np.uint8:
            raise ValueError("marked_rgb must be HxWx3 uint8")
        frozen = np.array(marked, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "marked_rgb", frozen)
        object.__setattr__(
            self,
            "bindings",
            MappingProxyType(dict(self.bindings)),
        )
        for name, value in (("operation", self.operation), ("stage", self.stage)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be None or a non-empty string")
        digest = self.source_segmentation_sha256
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(
                "source_segmentation_sha256 must be None or lowercase SHA256"
            )


@dataclass(frozen=True, slots=True)
class PrivateMarkerResolution:
    status: MarkerStatus
    binding: PrivateMarkerBinding | None = field(default=None, repr=False)
    public_failure_reason: MarkerPublicFailureReason | None = None

    def __post_init__(self) -> None:
        if self.status is MarkerStatus.READY:
            if self.binding is None:
                raise ValueError("ready marker resolution requires a binding")
            if self.public_failure_reason is not None:
                raise ValueError("ready marker resolution cannot have a failure reason")
        elif self.binding is not None:
            raise ValueError("non-ready marker resolution cannot have a binding")


def _selection_unavailable(status: MarkerStatus) -> PrivateMarkerResolution:
    return PrivateMarkerResolution(
        status,
        public_failure_reason=MarkerPublicFailureReason.SELECTION_UNAVAILABLE,
    )


MARKER_LABEL_COUNTERBALANCE = ContextVar('marker_label_counterbalance', default=None)


def _build_marker_snapshot(*, rgb, bindings, **metadata):
    """Apply an opt-in answer-blind label permutation to every marker kind."""
    policy = MARKER_LABEL_COUNTERBALANCE.get()
    if policy is not None:
        if not isinstance(policy, Mapping) or set(policy) != {'seed', 'rotation'} or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in policy.values()
        ):
            raise ValueError('invalid marker counterbalance policy')
        values = list(bindings.values())
        labels = list(range(1, len(values) + 1))
        random.Random(policy['seed']).shuffle(labels)
        if labels:
            offset = policy['rotation'] % len(labels)
            labels = labels[offset:] + labels[:offset]
        bindings = {f'C{label}': replace(binding, marker=f'C{label}')
                    for label, binding in zip(labels, values)}
    return PrivateMarkerSnapshot(marked_rgb=render_marker_overlay(rgb, bindings),
                                 bindings=bindings, **metadata)


class PrivateVisualMarkerStore:
    """Own one one-shot marker snapshot; never serialize object bindings."""

    def __init__(self) -> None:
        self._snapshot: PrivateMarkerSnapshot | None = None
        self._consumed = False
        self._selected_marker: str | None = None

    @property
    def active_snapshot(self) -> PrivateMarkerSnapshot | None:
        return self._snapshot

    def publish(self, snapshot: PrivateMarkerSnapshot) -> None:
        if not isinstance(snapshot, PrivateMarkerSnapshot):
            raise TypeError("snapshot must be PrivateMarkerSnapshot")
        self._snapshot = snapshot
        self._consumed = False
        self._selected_marker = None

    def clear(self) -> None:
        self._snapshot = None
        self._consumed = False
        self._selected_marker = None

    def highlight(self, marker: str) -> None:
        """Highlight one active choice without consuming or changing it."""

        snapshot = self._snapshot
        if snapshot is None or self._consumed or marker not in snapshot.bindings:
            raise ValueError("cannot highlight an unavailable marker")
        self._selected_marker = marker

    def overlay_for(
        self,
        *,
        raw_rgb: np.ndarray,
        simulator_step: int,
    ) -> np.ndarray | None:
        snapshot = self._snapshot
        if snapshot is None or self._consumed:
            return None
        rgb = np.ascontiguousarray(np.asarray(raw_rgb, dtype=np.uint8))
        digest = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
        if (
            simulator_step != snapshot.simulator_step
            or digest != snapshot.source_rgb_sha256
        ):
            return None
        if self._selected_marker is None:
            return np.array(snapshot.marked_rgb, copy=True)
        return render_marker_overlay(
            rgb,
            snapshot.bindings,
            selected_marker=self._selected_marker,
        )

    def consume(
        self,
        *,
        marker: str,
        action_index: int,
        raw_rgb: np.ndarray,
        simulator_step: int,
        operation: str | None = None,
        stage: str | None = None,
        current_segmentation: np.ndarray | None = None,
    ) -> PrivateMarkerResolution:
        snapshot = self._snapshot
        if snapshot is None:
            return _selection_unavailable(MarkerStatus.UNKNOWN)
        if self._consumed:
            return _selection_unavailable(MarkerStatus.CONSUMED)
        rgb = np.ascontiguousarray(np.asarray(raw_rgb, dtype=np.uint8))
        digest = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
        strict_spatial_context = (
            snapshot.operation is not None or snapshot.stage is not None
        )
        context_mismatch = False
        if snapshot.operation is not None and operation != snapshot.operation:
            context_mismatch = True
        if snapshot.stage is not None and stage != snapshot.stage:
            context_mismatch = True
        if current_segmentation is not None:
            segmentation_digest = _segmentation_sha256(current_segmentation)
            if (
                snapshot.source_segmentation_sha256 is not None
                and segmentation_digest != snapshot.source_segmentation_sha256
            ):
                context_mismatch = True
        elif strict_spatial_context and snapshot.source_segmentation_sha256 is not None:
            context_mismatch = True
        if (
            action_index != snapshot.created_action_index + 1
            or simulator_step != snapshot.simulator_step
            or digest != snapshot.source_rgb_sha256
            or context_mismatch
        ):
            self.clear()
            return _selection_unavailable(MarkerStatus.STALE)
        # Every syntactically valid marker-consuming attempt is one-shot,
        # including an unknown label.  This also removes the overlay before
        # the executor captures the post-action frame.
        self._consumed = True
        binding = snapshot.bindings.get(marker)
        if binding is None:
            return _selection_unavailable(MarkerStatus.UNKNOWN)
        return PrivateMarkerResolution(MarkerStatus.READY, binding)


def build_private_marker_snapshot(
    *,
    frame: PrivateInstanceSegmentationFrame,
    public_head_rgb: np.ndarray,
    eligible_objects_by_name: Mapping[str, Any],
    official_bboxes: Sequence[Sequence[int]],
    created_action_index: int,
    minimum_visible_pixels: int = 8,
    minimum_bbox_span_pixels: int = 1,
    largest_component_only: bool = False,
) -> PrivateMarkerSnapshot:
    """Label eligible objects, completing omissions from current-frame masks."""

    rgb = np.ascontiguousarray(np.asarray(public_head_rgb, dtype=np.uint8))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("public_head_rgb must be HxWx3")
    digest = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
    frame.require_rgb_sha256(digest)
    segmentation_digest = _segmentation_sha256(frame.segmentation)
    if created_action_index < 0:
        raise ValueError("created_action_index must be non-negative")
    bindings = _object_marker_bindings(
        frame=frame,
        rgb=rgb,
        eligible_objects_by_name=eligible_objects_by_name,
        official_bboxes=official_bboxes,
        minimum_visible_pixels=minimum_visible_pixels,
        minimum_bbox_span_pixels=minimum_bbox_span_pixels,
        largest_component_only=largest_component_only,
    )

    return _build_marker_snapshot(
        camera=frame.camera,
        simulator_step=frame.simulator_step,
        source_rgb_sha256=digest,
        created_action_index=created_action_index,
        rgb=rgb,
        bindings=bindings,
        source_segmentation_sha256=segmentation_digest,
    )


def build_private_pick_snapshot(
    *,
    frame: PrivateInstanceSegmentationFrame,
    public_head_rgb: np.ndarray,
    eligible_objects_by_name: Mapping[str, Any],
    bundle_targets: Sequence[PickBundleTargetToken],
    official_bboxes: Sequence[Sequence[int]],
    created_action_index: int,
    minimum_visible_pixels: int = 16,
    minimum_bbox_span_pixels: int = 3,
) -> PrivateMarkerSnapshot:
    """Use shared object coverage, then add explicit relation bundles."""

    rgb, rgb_digest, segmentation_digest = _validated_snapshot_inputs(
        frame=frame,
        public_head_rgb=public_head_rgb,
        created_action_index=created_action_index,
    )
    eligible = dict(eligible_objects_by_name)
    object_bindings = _object_marker_bindings(
        frame=frame,
        rgb=rgb,
        eligible_objects_by_name=eligible,
        official_bboxes=official_bboxes,
        minimum_visible_pixels=minimum_visible_pixels,
        minimum_bbox_span_pixels=minimum_bbox_span_pixels,
        largest_component_only=True,
    )

    bundles = tuple(bundle_targets)
    if any(not isinstance(bundle, PickBundleTargetToken) for bundle in bundles):
        raise TypeError("bundle_targets must contain PickBundleTargetToken values")
    bundle_members = {
        name: member
        for bundle in bundles
        for name, member in zip(
            bundle.member_names,
            bundle.member_tokens,
            strict=True,
        )
        if name
    }
    visible_by_name = {
        item[5]: item
        for item in _eligible_bbox_candidates(
            frame=frame,
            rgb=rgb,
            eligible_objects_by_name={**eligible, **bundle_members},
            official_bboxes=official_bboxes,
            minimum_visible_pixels=1,
            minimum_bbox_span_pixels=1,
            largest_component_only=True,
        )
    }
    renderable_bundles: list[tuple[PickBundleTargetToken, tuple[Any, ...]]] = []
    for bundle in bundles:
        visible = tuple(
            visible_by_name[name]
            for name in bundle.member_names
            if name in visible_by_name
        )
        if len(visible) < 2:
            continue
        if not bundle.exclusive and len(visible) != len(bundle.member_tokens):
            continue
        renderable_bundles.append((bundle, visible))

    exclusive_names = {
        name
        for bundle in bundles
        if bundle.exclusive
        for name in bundle.member_names
    }
    bindings: dict[str, PrivateMarkerBinding] = {}
    for binding in object_bindings.values():
        if binding.object_name in exclusive_names:
            continue
        marker = f"C{len(bindings) + 1}"
        bindings[marker] = replace(binding, marker=marker)
    for bundle, visible in renderable_bundles:
        marker = f"C{len(bindings) + 1}"
        masks = tuple(item[8] for item in visible)
        union_mask = np.logical_or.reduce(masks)
        ys, xs = np.nonzero(union_mask)
        anchors = tuple(
            _mask_anchor(item[8], item[:4])
            for item in visible
        )
        root_name = str(getattr(bundle.root_token, "name", ""))
        root_item = visible_by_name.get(root_name, visible[0])
        bindings[marker] = PrivateMarkerBinding(
            marker=marker,
            object_name=root_name,
            instance_id=int(root_item[4]),
            visible_pixels=int(len(xs)),
            official_bbox_xyxy=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            ),
            target_token=bundle,
            binding_kind="bundle",
            anchor_uv=_mask_anchor(
                union_mask,
                (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            ),
            mask_source="seg_instance_or_prefix",
            member_anchors_uv=anchors,
            candidate_source="relation_bundle",
        )
    return _build_marker_snapshot(
        camera=frame.camera,
        simulator_step=frame.simulator_step,
        source_rgb_sha256=rgb_digest,
        created_action_index=created_action_index,
        rgb=rgb,
        bindings=bindings,
        operation="pick",
        stage="target",
        source_segmentation_sha256=segmentation_digest,
    )


def _bbox_candidate_binding(
    marker: str,
    item: tuple[Any, ...],
    *,
    candidate_source: str = "standard",
) -> PrivateMarkerBinding:
    (
        x0,
        y0,
        x1,
        y1,
        instance_id,
        object_name,
        visible_pixels,
        token,
        target_mask,
        mask_source,
    ) = item
    return PrivateMarkerBinding(
        marker=marker,
        object_name=object_name,
        instance_id=instance_id,
        visible_pixels=visible_pixels,
        official_bbox_xyxy=(x0, y0, x1, y1),
        target_token=token,
        anchor_uv=_mask_anchor(target_mask, (x0, y0, x1, y1)),
        mask_source=mask_source,
        candidate_source=candidate_source,
    )


def _mask_anchor(
    mask: np.ndarray,
    bbox_xyxy: Sequence[int],
) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("marker mask must contain a visible pixel")
    x0, y0, x1, y1 = (int(value) for value in bbox_xyxy)
    centre_x = (x0 + x1) / 2
    centre_y = (y0 + y1) / 2
    index = int(np.argmin((xs - centre_x) ** 2 + (ys - centre_y) ** 2))
    return int(xs[index]), int(ys[index])


def build_private_drawer_part_snapshot(
    *,
    frame: PrivateInstanceSegmentationFrame,
    public_head_rgb: np.ndarray,
    eligible_openables_by_name: Mapping[str, Any],
    official_bboxes: Sequence[Sequence[int]],
    created_action_index: int,
    operation: str,
    minimum_visible_pixels: int = 8,
    diagnostics: list | None = None,
) -> PrivateMarkerSnapshot:
    """Draw movable parts and whole-object markers for single-part openables."""

    rgb, rgb_digest, segmentation_digest = _validated_snapshot_inputs(
        frame=frame,
        public_head_rgb=public_head_rgb,
        created_action_index=created_action_index,
    )
    candidates = visible_drawer_parts(
        frame=frame,
        eligible_objects_by_name=eligible_openables_by_name,
        operation=operation,
        minimum_visible_pixels=minimum_visible_pixels,
        diagnostics=diagnostics,
    )
    bindings: dict[str, PrivateMarkerBinding] = {}
    for candidate in candidates:
        marker = f"C{len(bindings) + 1}"
        selector = f"drawer_compartment_{len(bindings) + 1}"
        # The public instance ID is not used for part identity. Retain one
        # matching whole-object ID solely for legacy role/freshness metadata.
        instance_id = next(
            (
                raw_id
                for raw_id, name in frame.instance_id_to_name.items()
                if name == candidate.object_name
            ),
            -1,
        )
        bindings[marker] = PrivateMarkerBinding(
            marker=marker,
            object_name=candidate.object_name,
            instance_id=int(instance_id),
            visible_pixels=candidate.visible_pixels,
            official_bbox_xyxy=candidate.bbox_xyxy,
            target_token=candidate.target_token,
            binding_kind="part",
            selector=selector,
            anchor_uv=candidate.anchor_uv,
        )
    whole_snapshot = build_private_marker_snapshot(
        frame=frame,
        public_head_rgb=rgb,
        eligible_objects_by_name={
            name: target
            for name, target in eligible_openables_by_name.items()
            if not drawer_part_tokens(target)
        },
        official_bboxes=official_bboxes,
        created_action_index=created_action_index,
        minimum_visible_pixels=minimum_visible_pixels,
    )
    for binding in whole_snapshot.bindings.values():
        marker = f"C{len(bindings) + 1}"
        bindings[marker] = replace(binding, marker=marker)
    return _build_marker_snapshot(
        camera=frame.camera,
        simulator_step=frame.simulator_step,
        source_rgb_sha256=rgb_digest,
        created_action_index=created_action_index,
        rgb=rgb,
        bindings=bindings,
        operation=operation,
        stage="target",
        source_segmentation_sha256=segmentation_digest,
    )


def build_private_toggle_part_snapshot(
    *,
    frame: PrivateInstanceSegmentationFrame,
    public_head_rgb: np.ndarray,
    eligible_toggleables_by_name: Mapping[str, Any],
    created_action_index: int,
    operation: str,
    minimum_visible_pixels: int = 1,
) -> PrivateMarkerSnapshot:
    """Mark the visible toggle-button mesh of each eligible object.

    OmniGibson's ``ToggledOn`` state exposes its private toggle-button visual
    (and its owning link).  The public marker is still derived exclusively
    from pixels in the current ``seg_instance_id`` frame.  The binding keeps
    the whole object as its executor token because the official Symbolic
    primitive is object-level.
    """

    if operation not in {"toggle_on", "toggle_off"}:
        raise ValueError("toggle part snapshot requires toggle_on or toggle_off")
    if minimum_visible_pixels < 1:
        raise ValueError("minimum_visible_pixels must be positive")
    rgb, rgb_digest, segmentation_digest = _validated_snapshot_inputs(
        frame=frame,
        public_head_rgb=public_head_rgb,
        created_action_index=created_action_index,
    )
    part_segmentation = frame.part_segmentation
    part_registry = frame.part_id_to_prim_path
    candidates: list[
        tuple[int, int, str, Any, int, tuple[int, int, int, int], tuple[int, int]]
    ] = []
    if part_segmentation is not None:
        for object_name, token in sorted(eligible_toggleables_by_name.items()):
            states = getattr(token, "states", {})
            toggle_state = next(
                (
                    state
                    for state_type, state in states.items()
                    if getattr(state_type, "__name__", "") == "ToggledOn"
                    or type(state).__name__ == "ToggledOn"
                ),
                None,
            )
            if toggle_state is None:
                continue
            visual_marker = getattr(toggle_state, "visual_marker", None)
            toggle_link = getattr(toggle_state, "link", None)
            path = getattr(visual_marker, "prim_path", None) or getattr(
                toggle_link, "prim_path", None
            )
            if not isinstance(path, str) or not path:
                continue
            matching_ids = [
                int(instance_id)
                for instance_id, prim_path in part_registry.items()
                if isinstance(prim_path, str)
                and (prim_path == path or prim_path.startswith(f"{path}/"))
            ]
            if not matching_ids:
                continue
            mask = np.isin(part_segmentation, matching_ids)
            points = np.argwhere(mask)
            if points.shape[0] < minimum_visible_pixels:
                continue
            centre = points.astype(np.float64).mean(axis=0)
            anchor_index = int(
                np.argmin(np.square(points - centre).sum(axis=1))
            )
            anchor_y, anchor_x = (int(value) for value in points[anchor_index])
            ys, xs = points[:, 0], points[:, 1]
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            candidates.append(
                (
                    anchor_y,
                    anchor_x,
                    object_name,
                    token,
                    int(points.shape[0]),
                    bbox,
                    (anchor_x, anchor_y),
                )
            )

    bindings: dict[str, PrivateMarkerBinding] = {}
    for _, _, object_name, token, visible_pixels, bbox, anchor_uv in sorted(
        candidates
    ):
        marker = f"C{len(bindings) + 1}"
        bindings[marker] = PrivateMarkerBinding(
            marker=marker,
            object_name=object_name,
            instance_id=-1,
            visible_pixels=visible_pixels,
            official_bbox_xyxy=bbox,
            target_token=token,
            binding_kind="part",
            selector="togglebutton",
            anchor_uv=anchor_uv,
            mask_source="seg_instance_id_prefix",
        )
    # A visible button on one object must not suppress coverage of another.
    # Toggle execution is object-level, so a visible whole-object binding is
    # unambiguous. Multi-joint Open/Close deliberately retains its part tokens.
    covered_names = {binding.object_name for binding in bindings.values()}
    for binding in _object_marker_bindings(
        frame=frame,
        rgb=rgb,
        eligible_objects_by_name=eligible_toggleables_by_name,
        official_bboxes=(),
        minimum_visible_pixels=8,
    ).values():
        if binding.object_name in covered_names:
            continue
        marker = f"C{len(bindings) + 1}"
        bindings[marker] = replace(
            binding, marker=marker, candidate_source="visual_fallback",
        )
    return _build_marker_snapshot(
        camera=frame.camera,
        simulator_step=frame.simulator_step,
        source_rgb_sha256=rgb_digest,
        created_action_index=created_action_index,
        rgb=rgb,
        bindings=bindings,
        operation=operation,
        stage="target",
        source_segmentation_sha256=segmentation_digest,
    )


def build_private_container_compartment_snapshot(
    *,
    frame: PrivateInstanceSegmentationFrame,
    public_head_rgb: np.ndarray,
    reference_binding: PrivateMarkerBinding,
    created_action_index: int,
    minimum_visible_pixels: int = 8,
    operation: str = "place_inside",
    stage: str = "inside_cell",
) -> PrivateMarkerSnapshot:
    """Publish real open drawers or one whole-container second-stage target."""

    if not isinstance(reference_binding, PrivateMarkerBinding):
        raise TypeError("reference_binding must be PrivateMarkerBinding")
    rgb, rgb_digest, segmentation_digest = _validated_snapshot_inputs(
        frame=frame,
        public_head_rgb=public_head_rgb,
        created_action_index=created_action_index,
    )
    target = reference_binding.target_token
    if isinstance(target, SpatialMarkerTargetToken):
        target = target.target_token
    drawer_candidates = tuple(
        candidate
        for candidate in visible_drawer_parts(
            frame=frame,
            eligible_objects_by_name={reference_binding.object_name: target},
            operation="close",
            minimum_visible_pixels=minimum_visible_pixels,
        )
        if _has_fillable_drawer_region(candidate.target_token)
    )
    bindings: dict[str, PrivateMarkerBinding] = {}
    for drawer_index, candidate in enumerate(drawer_candidates, start=1):
        marker = f"C{len(bindings) + 1}"
        selector = f"drawer_compartment_{drawer_index}"
        bindings[marker] = PrivateMarkerBinding(
            marker=marker,
            object_name=reference_binding.object_name,
            instance_id=reference_binding.instance_id,
            visible_pixels=candidate.visible_pixels,
            official_bbox_xyxy=candidate.bbox_xyxy,
            target_token=SpatialMarkerTargetToken(
                candidate.target_token,
                operation,
                selector,
            ),
            binding_kind="inside_cell",
            selector=selector,
            anchor_uv=candidate.anchor_uv,
        )

    if not bindings:
        mask = _binding_target_mask(frame, reference_binding)
        ys, xs = np.nonzero(mask)
        if len(xs) >= minimum_visible_pixels:
            centre_x = float(xs.mean())
            centre_y = float(ys.mean())
            anchor_index = int(
                np.argmin((xs - centre_x) ** 2 + (ys - centre_y) ** 2)
            )
            selector = "container_default"
            bindings["C1"] = PrivateMarkerBinding(
                marker="C1",
                object_name=reference_binding.object_name,
                instance_id=reference_binding.instance_id,
                visible_pixels=int(len(xs)),
                official_bbox_xyxy=reference_binding.official_bbox_xyxy,
                target_token=SpatialMarkerTargetToken(
                    target,
                    operation,
                    selector,
                ),
                binding_kind="inside_cell",
                selector=selector,
                anchor_uv=(
                    int(xs[anchor_index]),
                    int(ys[anchor_index]),
                ),
            )

    return _build_marker_snapshot(
        camera=frame.camera,
        simulator_step=frame.simulator_step,
        source_rgb_sha256=rgb_digest,
        created_action_index=created_action_index,
        rgb=rgb,
        bindings=bindings,
        operation=operation,
        stage=stage,
        source_segmentation_sha256=segmentation_digest,
    )


def _has_fillable_drawer_region(target: Any) -> bool:
    """Exclude articulated doors or panels that cannot contain an object."""

    try:
        return bool(drawer_fillable_aabbs(target))
    except RuntimeError:
        return False


def build_private_on_top_support_snapshot(
    *,
    frame: PrivateInstanceSegmentationFrame,
    public_head_rgb: np.ndarray,
    eligible_supports_by_name: Mapping[str, Any],
    official_bboxes: Sequence[Sequence[int]],
    created_action_index: int,
    minimum_visible_pixels: int = 8,
    operation: str = "place_on_top",
    stage: str = "target",
) -> PrivateMarkerSnapshot:
    """Label each visible support object once without implying a drop point."""

    return replace(
        build_private_marker_snapshot(
            frame=frame,
            public_head_rgb=public_head_rgb,
            eligible_objects_by_name=eligible_supports_by_name,
            official_bboxes=official_bboxes,
            created_action_index=created_action_index,
            minimum_visible_pixels=minimum_visible_pixels,
        ),
        operation=operation,
        stage=stage,
    )


def build_private_on_top_surface_snapshot(
    *,
    frame: PrivateInstanceSegmentationFrame,
    public_head_rgb: np.ndarray,
    eligible_supports_by_name: Mapping[str, Any],
    official_bboxes: Sequence[Sequence[int]],
    surface_regions: Sequence[OnTopSurfaceRegion],
    created_action_index: int,
    minimum_visible_pixels: int = 8,
    minimum_cell_pixels: int = 1,
    operation: str = "place_on_top",
    stage: str = "surface_cell",
) -> PrivateMarkerSnapshot:
    """Bind only ray-verified cells on the already selected support."""

    rgb, rgb_digest, segmentation_digest = _validated_snapshot_inputs(
        frame=frame,
        public_head_rgb=public_head_rgb,
        created_action_index=created_action_index,
    )
    if minimum_cell_pixels <= 0:
        raise ValueError("minimum_cell_pixels must be positive")
    supports = dict(eligible_supports_by_name)
    visible = _eligible_bbox_candidates(
        frame=frame,
        rgb=rgb,
        eligible_objects_by_name=supports,
        official_bboxes=official_bboxes,
        minimum_visible_pixels=minimum_visible_pixels,
    )
    prefix_masks = _part_prefix_masks(
        frame=frame,
        eligible_objects_by_name=supports,
    )
    visible_by_name = {}
    for item in visible:
        target_mask = item[8]
        prefix_mask = prefix_masks.get(item[5])
        mask_source = item[9]
        if prefix_mask is not None and np.any(prefix_mask & ~target_mask):
            target_mask = target_mask | prefix_mask
            mask_source = "seg_instance_or_prefix"
        visible_by_name[item[5]] = (*item[:8], target_mask, mask_source)

    ordered_regions = sorted(
        surface_regions,
        key=lambda region: (
            sum(point[1] for point in region.polygon_uv) / len(region.polygon_uv),
            sum(point[0] for point in region.polygon_uv) / len(region.polygon_uv),
            region.object_name,
            region.selector,
        ),
    )
    bindings: dict[str, PrivateMarkerBinding] = {}
    seen_regions: set[tuple[str, str]] = set()
    for region in ordered_regions:
        if not isinstance(region, OnTopSurfaceRegion):
            raise TypeError("surface_regions must contain OnTopSurfaceRegion")
        identity = (region.object_name, region.selector)
        if identity in seen_regions:
            raise ValueError("surface region selectors must be unique per support")
        seen_regions.add(identity)
        support = visible_by_name.get(region.object_name)
        if support is None:
            continue
        x0, y0, x1, y1, instance_id, object_name, _, target, target_mask, mask_source = support
        polygon = region.polygon_uv
        if any(
            x < 0 or y < 0 or x >= rgb.shape[1] or y >= rgb.shape[0]
            for x, y in polygon
        ):
            continue
        anchor = next((
            point for point in (region.anchor_uv, *region.anchor_candidates_uv)
            if 0 <= point[0] < rgb.shape[1] and 0 <= point[1] < rgb.shape[0]
            and bool(target_mask[point[1], point[0]])
        ), None)
        if anchor is None:
            continue
        cell_image = Image.new("1", (rgb.shape[1], rgb.shape[0]))
        ImageDraw.Draw(cell_image).polygon(polygon, fill=1)
        cell_mask = np.asarray(cell_image, dtype=np.bool_)
        cell_pixels = int(np.count_nonzero(cell_mask & target_mask))
        if cell_pixels < minimum_cell_pixels:
            continue
        marker = f"C{len(bindings) + 1}"
        bindings[marker] = PrivateMarkerBinding(
            marker=marker,
            object_name=object_name,
            instance_id=instance_id,
            visible_pixels=cell_pixels,
            official_bbox_xyxy=(x0, y0, x1, y1),
            target_token=SpatialMarkerTargetToken(
                target,
                operation,
                region.selector,
                region_center_xy=region.region_center_xy,
                region_extent_xy=region.region_extent_xy,
                surface_z_m=region.surface_z_m,
            ),
            binding_kind="surface_cell",
            selector=region.selector,
            anchor_uv=anchor,
            polygon_uv=polygon,
            mask_source=mask_source,
            candidate_source="generic",
        )
    return _build_marker_snapshot(
        camera=frame.camera,
        simulator_step=frame.simulator_step,
        source_rgb_sha256=rgb_digest,
        created_action_index=created_action_index,
        rgb=rgb,
        bindings=bindings,
        operation=operation,
        stage=stage,
        source_segmentation_sha256=segmentation_digest,
    )


def build_private_next_to_position_snapshot(
    *,
    frame: PrivateInstanceSegmentationFrame,
    public_head_rgb: np.ndarray,
    reference_binding: PrivateMarkerBinding,
    created_action_index: int,
    minimum_visible_pixels: int = 8,
    support_positions: Sequence[NextToSupportPosition] | None = None,
    operation: str = "place_next_to",
    stage: str = "position",
) -> PrivateMarkerSnapshot:
    """Build legacy image points or exact physical support positions."""

    if not isinstance(reference_binding, PrivateMarkerBinding):
        raise TypeError("reference_binding must be PrivateMarkerBinding")
    rgb, rgb_digest, segmentation_digest = _validated_snapshot_inputs(
        frame=frame,
        public_head_rgb=public_head_rgb,
        created_action_index=created_action_index,
    )
    target_mask = _binding_target_mask(frame, reference_binding)
    if not np.any(target_mask):
        raise ValueError("reference binding does not belong to the current frame")
    target = reference_binding.target_token
    if isinstance(target, SpatialMarkerTargetToken):
        target = target.target_token
    if support_positions is None:
        batch = generate_next_to_position_markers(
            target_mask=target_mask,
            official_bbox_xyxy=reference_binding.official_bbox_xyxy,
            minimum_visible_pixels=minimum_visible_pixels,
        )
        candidates = tuple(
            (item.position, item.anchor_uv, None, None)
            for item in batch.candidates
        )
        visible_pixels = batch.visible_pixels
    else:
        candidates = tuple(
            (
                item.selector,
                item.anchor_uv,
                item.region_center_xy,
                item.region_extent_xy,
            )
            for item in support_positions
        )
        visible_pixels = int(np.count_nonzero(target_mask))
    bindings: dict[str, PrivateMarkerBinding] = {}
    for selector, anchor, center_xy, extent_xy in candidates:
        if not (0 <= anchor[0] < rgb.shape[1] and 0 <= anchor[1] < rgb.shape[0]):
            continue
        marker = f"C{len(bindings) + 1}"
        bindings[marker] = PrivateMarkerBinding(
            marker=marker,
            object_name=reference_binding.object_name,
            instance_id=reference_binding.instance_id,
            visible_pixels=visible_pixels,
            official_bbox_xyxy=reference_binding.official_bbox_xyxy,
            target_token=SpatialMarkerTargetToken(
                target,
                operation,
                selector,
                region_center_xy=center_xy,
                region_extent_xy=extent_xy,
            ),
            binding_kind="position",
            selector=selector,
            anchor_uv=anchor,
            mask_source=reference_binding.mask_source,
        )
    return _build_marker_snapshot(
        camera=frame.camera,
        simulator_step=frame.simulator_step,
        source_rgb_sha256=rgb_digest,
        created_action_index=created_action_index,
        rgb=rgb,
        bindings=bindings,
        operation=operation,
        stage=stage,
        source_segmentation_sha256=segmentation_digest,
    )


def build_private_touching_direction_snapshot(
    *,
    frame: PrivateInstanceSegmentationFrame,
    public_head_rgb: np.ndarray,
    reference_binding: PrivateMarkerBinding,
    created_action_index: int,
    minimum_visible_pixels: int = 8,
    operation: str = "place_touching",
    stage: str = "direction",
) -> PrivateMarkerSnapshot:
    """Build four screen-relative choices for one visible reference object."""

    if not isinstance(reference_binding, PrivateMarkerBinding):
        raise TypeError("reference_binding must be PrivateMarkerBinding")
    rgb, rgb_digest, segmentation_digest = _validated_snapshot_inputs(
        frame=frame,
        public_head_rgb=public_head_rgb,
        created_action_index=created_action_index,
    )
    target_mask = _binding_target_mask(frame, reference_binding)
    if not np.any(target_mask):
        raise ValueError("reference binding does not belong to the current frame")
    target = reference_binding.target_token
    if isinstance(target, SpatialMarkerTargetToken):
        target = target.target_token
    batch = generate_touching_direction_markers(
        target_mask=target_mask,
        official_bbox_xyxy=reference_binding.official_bbox_xyxy,
        minimum_visible_pixels=minimum_visible_pixels,
    )
    bindings: dict[str, PrivateMarkerBinding] = {}
    for candidate in batch.candidates:
        marker = f"C{len(bindings) + 1}"
        selector = candidate.direction
        bindings[marker] = PrivateMarkerBinding(
            marker=marker,
            object_name=reference_binding.object_name,
            instance_id=reference_binding.instance_id,
            visible_pixels=batch.visible_pixels,
            official_bbox_xyxy=reference_binding.official_bbox_xyxy,
            target_token=SpatialMarkerTargetToken(target, operation, selector),
            binding_kind="direction",
            selector=selector,
            anchor_uv=candidate.anchor_uv,
            mask_source=reference_binding.mask_source,
        )
    return _build_marker_snapshot(
        camera=frame.camera,
        simulator_step=frame.simulator_step,
        source_rgb_sha256=rgb_digest,
        created_action_index=created_action_index,
        rgb=rgb,
        bindings=bindings,
        operation=operation,
        stage=stage,
        source_segmentation_sha256=segmentation_digest,
    )


def render_marker_overlay(
    rgb: np.ndarray,
    bindings: Mapping[str, PrivateMarkerBinding],
    *,
    selected_marker: str | None = None,
) -> np.ndarray:
    """Draw object boxes, verified surface polygons, or direction arrows.

    ``selected_marker`` is artifact-only presentation state.  It highlights
    the already selected visible region and never changes its private token.
    """

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15
        )
    except OSError:
        font = ImageFont.load_default()
    colors = (
        (0, 205, 225),
        (255, 185, 0),
        (245, 75, 135),
        (75, 225, 105),
        (165, 105, 245),
    )
    occupied_badges: list[tuple[int, int, int, int]] = []
    for index, (marker, binding) in enumerate(bindings.items()):
        color = colors[index % len(colors)]
        if binding.binding_kind == "bundle":
            _draw_bundle_marker(
                draw,
                marker,
                binding,
                color,
                font,
                image.size,
                selected=marker == selected_marker,
                occupied_badges=occupied_badges,
            )
            continue
        if binding.binding_kind == "surface_cell":
            _draw_surface_region(
                image,
                draw,
                marker,
                binding,
                color,
                font,
                selected=marker == selected_marker,
                occupied_badges=occupied_badges,
            )
            continue
        if binding.binding_kind == "part":
            x0, y0, x1, y1 = binding.official_bbox_xyxy
            part_box = (
                max(0, x0 - 3),
                max(0, y0 - 3),
                min(image.width - 1, x1 + 3),
                min(image.height - 1, y1 + 3),
            )
            _draw_corner_box(draw, part_box, color)
            _draw_external_badge(
                draw,
                marker,
                part_box,
                color,
                font,
                image.size,
                target=binding.anchor_uv,
                selected=marker == selected_marker,
                occupied_badges=occupied_badges,
            )
            continue
        if binding.binding_kind == "inside_cell":
            x0, y0, x1, y1 = binding.official_bbox_xyxy
            cell_box = (
                max(0, x0 - 2),
                max(0, y0 - 2),
                min(image.width - 1, x1 + 2),
                min(image.height - 1, y1 + 2),
            )
            _draw_corner_box(draw, cell_box, color)
            _draw_external_badge(
                draw,
                marker,
                cell_box,
                color,
                font,
                image.size,
                target=binding.anchor_uv,
                selected=marker == selected_marker,
                occupied_badges=occupied_badges,
            )
            continue
        if binding.binding_kind == "position":
            _draw_direction_marker(
                draw,
                marker,
                binding.anchor_uv,
                binding.selector,
                color,
                font,
                image.size,
                selected=marker == selected_marker,
                occupied_badges=occupied_badges,
            )
            continue
        if binding.binding_kind == "direction":
            _draw_direction_marker(
                draw,
                marker,
                binding.anchor_uv,
                binding.selector,
                color,
                font,
                image.size,
                occupied_badges=occupied_badges,
            )
            continue
        x0, y0, x1, y1 = binding.official_bbox_xyxy
        short = max(1, min(x1 - x0, y1 - y0))
        # The black under-stroke is four pixels wide. Keep its full footprint
        # outside the visible object instead of letting small-object corners
        # overwrite one or two target pixels.
        padding = max(4, min(7, round(short * 0.08)))
        box = (
            max(0, x0 - padding),
            max(0, y0 - padding),
            min(image.width - 1, x1 + padding),
            min(image.height - 1, y1 + padding),
        )
        _draw_corner_box(draw, box, color)
        _draw_external_badge(
            draw,
            marker,
            box,
            color,
            font,
            image.size,
            target=binding.anchor_uv,
            selected=marker == selected_marker,
            occupied_badges=occupied_badges,
        )
    return np.asarray(image, dtype=np.uint8)


def _draw_bundle_marker(
    draw: ImageDraw.ImageDraw,
    marker: str,
    binding: PrivateMarkerBinding,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    image_size: tuple[int, int],
    *,
    selected: bool,
    occupied_badges: list[tuple[int, int, int, int]],
) -> None:
    x0, y0, x1, y1 = binding.official_bbox_xyxy
    box = (
        max(0, x0 - 5),
        max(0, y0 - 5),
        min(image_size[0] - 1, x1 + 5),
        min(image_size[1] - 1, y1 + 5),
    )
    _draw_corner_box(draw, box, color)
    anchors = binding.member_anchors_uv or ()
    root = anchors[0]
    for anchor in anchors[1:]:
        draw.line((root, anchor), fill=(0, 0, 0), width=5)
        draw.line((root, anchor), fill=color, width=2)
    for x, y in anchors:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(0, 0, 0))
    _draw_external_badge(
        draw,
        marker,
        box,
        color,
        font,
        image_size,
        target=binding.anchor_uv,
        selected=selected,
        occupied_badges=occupied_badges,
    )


def _draw_surface_region(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    marker: str,
    binding: PrivateMarkerBinding,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    *,
    selected: bool,
    occupied_badges: list[tuple[int, int, int, int]],
) -> None:
    polygon = binding.polygon_uv
    if polygon is None:
        _draw_spatial_point(
            draw,
            marker,
            binding.anchor_uv,
            color,
            font,
            image.size,
            selected=selected,
            occupied_badges=occupied_badges,
        )
        return
    points = tuple((int(x), int(y)) for x, y in polygon)
    if selected:
        tint = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ImageDraw.Draw(tint).polygon(points, fill=(*color, 42))
        image.paste(tint, (0, 0), tint)
        draw = ImageDraw.Draw(image)
    closed = points + (points[0],)
    draw.line(closed, fill=(0, 0, 0), width=5 if selected else 4, joint="curve")
    draw.line(closed, fill=color, width=3 if selected else 2, joint="curve")
    x0 = min(point[0] for point in points)
    x1 = max(point[0] for point in points)
    y0 = min(point[1] for point in points)
    y1 = max(point[1] for point in points)
    row = 1 if (binding.selector or "").startswith("on_top_r1_") else 0
    _draw_compact_badge(
        draw,
        marker,
        ((x0 + x1) // 2 - 10, y0 - 22 if row == 0 else y1 + 5),
        color,
        font,
        image.size,
        selected=selected,
        leader_target=binding.anchor_uv,
        occupied_badges=occupied_badges,
    )


def _validated_snapshot_inputs(
    *,
    frame: PrivateInstanceSegmentationFrame,
    public_head_rgb: np.ndarray,
    created_action_index: int,
) -> tuple[np.ndarray, str, str]:
    rgb = np.ascontiguousarray(np.asarray(public_head_rgb, dtype=np.uint8))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("public_head_rgb must be HxWx3")
    if tuple(frame.segmentation.shape) != tuple(rgb.shape[:2]):
        raise ValueError("private segmentation must match public_head_rgb dimensions")
    if isinstance(created_action_index, bool) or not isinstance(created_action_index, int):
        raise TypeError("created_action_index must be a non-negative integer")
    if created_action_index < 0:
        raise ValueError("created_action_index must be non-negative")
    rgb_digest = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
    frame.require_rgb_sha256(rgb_digest)
    return rgb, rgb_digest, _segmentation_sha256(frame.segmentation)


def _segmentation_sha256(segmentation: np.ndarray) -> str:
    raw = np.asarray(segmentation)
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim != 2 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("current_segmentation must be a non-empty integer 2D image")
    normalized = np.ascontiguousarray(raw, dtype=np.int64)
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _object_marker_bindings(
    *,
    frame: PrivateInstanceSegmentationFrame,
    rgb: np.ndarray,
    eligible_objects_by_name: Mapping[str, Any],
    official_bboxes: Sequence[Sequence[int]],
    minimum_visible_pixels: int = 8,
    minimum_bbox_span_pixels: int = 1,
    largest_component_only: bool = False,
) -> dict[str, PrivateMarkerBinding]:
    """Complete each eligible object, never inventing targets or spatial cells.

    The shared fallback keeps at least eight visible pixels and a one-pixel
    span. Pick uses stricter primary thresholds; missed non-task objects get
    the same recovery as task objects, without relaxing their role admission.
    """

    candidates = _eligible_bbox_candidates(
        frame=frame, rgb=rgb,
        eligible_objects_by_name=eligible_objects_by_name,
        official_bboxes=official_bboxes,
        minimum_visible_pixels=minimum_visible_pixels,
        minimum_bbox_span_pixels=minimum_bbox_span_pixels,
        largest_component_only=largest_component_only,
    )
    primary_names = {item[5] for item in candidates}
    missing = set(eligible_objects_by_name).difference(primary_names)
    if missing and (minimum_visible_pixels > 8 or minimum_bbox_span_pixels > 1):
        # Resolve masks against the complete eligible set so nested prim roots
        # cannot donate their pixels to an omitted parent object.
        candidates.extend(
            item for item in _eligible_bbox_candidates(
                frame=frame, rgb=rgb,
                eligible_objects_by_name=eligible_objects_by_name,
                official_bboxes=official_bboxes,
                minimum_visible_pixels=min(8, minimum_visible_pixels),
                minimum_bbox_span_pixels=1,
                largest_component_only=largest_component_only,
            ) if item[5] in missing
        )
    candidates.sort(key=lambda item: (
        (item[0] + item[2]) / 2, (item[1] + item[3]) / 2, item[4],
    ))
    boxed_ids = {int(box[4]) for box in official_bboxes}
    return {
        f"C{index}": _bbox_candidate_binding(
            f"C{index}", item,
            candidate_source=(
                "generic"
                if item[5] in primary_names and item[4] in boxed_ids
                and item[9] == "seg_instance"
                else "visual_fallback"
            ),
        )
        for index, item in enumerate(candidates, start=1)
    }


def _eligible_bbox_candidates(
    *,
    frame: PrivateInstanceSegmentationFrame,
    rgb: np.ndarray,
    eligible_objects_by_name: Mapping[str, Any],
    official_bboxes: Sequence[Sequence[int]],
    minimum_visible_pixels: int,
    minimum_bbox_span_pixels: int = 1,
    largest_component_only: bool = False,
) -> list[
    tuple[
        int,
        int,
        int,
        int,
        int,
        str,
        int,
        Any,
        np.ndarray,
        str,
    ]
]:
    eligible = dict(eligible_objects_by_name)
    candidates: list[
        tuple[
            int,
            int,
            int,
            int,
            int,
            str,
            int,
            Any,
            np.ndarray,
            str,
        ]
    ] = []
    seen_instance_ids: set[int] = set()
    seen_object_names: set[str] = set()
    for raw_box in official_bboxes:
        if len(raw_box) != 5:
            raise ValueError("official bbox must be x0,y0,x1,y1,instance_id")
        x0, y0, x1, y1, instance_id = (int(value) for value in raw_box)
        object_name = frame.instance_id_to_name.get(instance_id)
        if (
            object_name not in eligible
            or instance_id in seen_instance_ids
            or object_name in seen_object_names
        ):
            continue
        target_mask = frame.segmentation == instance_id
        if largest_component_only:
            target_mask = _largest_visible_component(target_mask)
        ys, xs = np.nonzero(target_mask)
        visible_pixels = int(len(xs))
        if visible_pixels < minimum_visible_pixels:
            continue
        if (
            int(xs.max() - xs.min() + 1) < minimum_bbox_span_pixels
            or int(ys.max() - ys.min() + 1) < minimum_bbox_span_pixels
        ):
            continue
        if not (0 <= x0 <= x1 < rgb.shape[1] and 0 <= y0 <= y1 < rgb.shape[0]):
            raise ValueError("official bbox falls outside the head image")
        if largest_component_only:
            x0, y0, x1, y1 = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            )
        seen_instance_ids.add(instance_id)
        seen_object_names.add(object_name)
        candidates.append(
            (
                x0,
                y0,
                x1,
                y1,
                instance_id,
                object_name,
                visible_pixels,
                eligible[object_name],
                target_mask,
                "seg_instance",
            )
        )

    # A missing bbox is not a missing object. Recover its current instance
    # pixels directly, or its visible meshes if the object registry missed it.
    # Neither source creates a target outside the caller's admitted live set.
    fallback_names = set(eligible).difference(seen_object_names)
    fallback_masks = _part_prefix_masks(
        frame=frame,
        eligible_objects_by_name=eligible,
    ) if fallback_names else {}
    for synthetic_index, object_name in enumerate(
        sorted(fallback_names),
        start=1,
    ):
        instance_ids = sorted(
            instance_id for instance_id, name in frame.instance_id_to_name.items()
            if name == object_name
        )
        prefix_mask = fallback_masks.get(object_name)
        if not instance_ids and prefix_mask is None:
            continue
        target_mask = np.isin(frame.segmentation, instance_ids)
        has_instance_pixels = bool(np.any(target_mask))
        mask_source = "seg_instance"
        if prefix_mask is not None:
            if np.any(prefix_mask & ~target_mask):
                mask_source = (
                    "seg_instance_or_prefix" if has_instance_pixels
                    else "seg_instance_id_prefix"
                )
            target_mask |= prefix_mask
        if largest_component_only:
            target_mask = _largest_visible_component(target_mask)
        ys, xs = np.nonzero(target_mask)
        visible_pixels = int(len(xs))
        if visible_pixels < minimum_visible_pixels:
            continue
        if (
            int(xs.max() - xs.min() + 1) < minimum_bbox_span_pixels
            or int(ys.max() - ys.min() + 1) < minimum_bbox_span_pixels
        ):
            continue
        candidates.append(
            (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
                instance_ids[0] if has_instance_pixels else -synthetic_index,
                object_name,
                visible_pixels,
                eligible[object_name],
                target_mask,
                mask_source,
            )
        )
    candidates.sort(
        key=lambda item: (
            (item[0] + item[2]) / 2,
            (item[1] + item[3]) / 2,
            item[4],
        )
    )
    return candidates


def _largest_visible_component(mask: np.ndarray) -> np.ndarray:
    """Keep one coherent current-frame region instead of scattered ID noise."""

    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=np.bool_)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def _part_prefix_masks(
    *,
    frame: PrivateInstanceSegmentationFrame,
    eligible_objects_by_name: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Group current-frame mesh pixels under the best matching object root."""

    part_segmentation = frame.part_segmentation
    if part_segmentation is None or not frame.part_id_to_prim_path:
        return {}
    roots: list[tuple[str, str]] = []
    for object_name, target in eligible_objects_by_name.items():
        if isinstance(target, SpatialMarkerTargetToken):
            target = target.target_token
        root = str(getattr(target, "prim_path", "")).rstrip("/")
        if root.startswith("/") and root:
            roots.append((object_name, root))
    if not roots:
        return {}

    part_ids_by_name: dict[str, list[int]] = {}
    for part_id, part_path in frame.part_id_to_prim_path.items():
        matches = [
            (len(root), object_name)
            for object_name, root in roots
            if part_path == root or part_path.startswith(f"{root}/")
        ]
        if not matches:
            continue
        _, object_name = max(matches, key=lambda item: (item[0], item[1]))
        part_ids_by_name.setdefault(object_name, []).append(int(part_id))
    return {
        object_name: np.isin(part_segmentation, sorted(part_ids))
        for object_name, part_ids in part_ids_by_name.items()
    }


def current_target_visibility(
    *,
    frame: PrivateInstanceSegmentationFrame,
    target: Any,
    minimum_visible_pixels: int = 8,
) -> tuple[int, str | None]:
    """Count one live target's current pixels using the marker fallback."""

    object_name = str(getattr(target, "name", ""))
    if not object_name:
        return 0, None
    instance_mask = np.zeros(frame.segmentation.shape, dtype=np.bool_)
    for instance_id, name in frame.instance_id_to_name.items():
        if name == object_name:
            instance_mask |= frame.segmentation == instance_id
    instance_pixels = int(np.count_nonzero(instance_mask))
    prefix_mask = _part_prefix_masks(
        frame=frame,
        eligible_objects_by_name={object_name: target},
    ).get(object_name)
    if prefix_mask is not None:
        combined = instance_mask | prefix_mask
        combined_pixels = int(np.count_nonzero(combined))
        if combined_pixels >= minimum_visible_pixels:
            return (
                combined_pixels,
                "seg_instance_or_prefix"
                if instance_pixels and np.any(prefix_mask & ~instance_mask)
                else "seg_instance_id_prefix"
                if not instance_pixels
                else "seg_instance",
            )
    if instance_pixels:
        return instance_pixels, "seg_instance"
    return 0, None


def _binding_target_mask(
    frame: PrivateInstanceSegmentationFrame,
    binding: PrivateMarkerBinding,
) -> np.ndarray:
    """Resolve the current-frame mask represented by one private binding."""

    if binding.mask_source in {"seg_instance", "seg_instance_or_prefix"}:
        if (
            frame.instance_id_to_name.get(binding.instance_id)
            != binding.object_name
        ):
            instance_mask = np.zeros(frame.segmentation.shape, dtype=np.bool_)
        else:
            instance_mask = np.isin(frame.segmentation, [
                instance_id
                for instance_id, name in frame.instance_id_to_name.items()
                if name == binding.object_name
            ])
        if binding.mask_source == "seg_instance":
            return instance_mask
    else:
        instance_mask = np.zeros(frame.segmentation.shape, dtype=np.bool_)
    masks = _part_prefix_masks(
        frame=frame,
        eligible_objects_by_name={
            binding.object_name: binding.target_token,
        },
    )
    prefix_mask = masks.get(
        binding.object_name,
        np.zeros(frame.segmentation.shape, dtype=np.bool_),
    )
    return instance_mask | prefix_mask


def _draw_spatial_point(
    draw,
    marker,
    anchor_uv,
    color,
    font,
    image_size,
    *,
    selected: bool = False,
    occupied_badges: list[tuple[int, int, int, int]],
) -> None:
    if anchor_uv is None:
        return
    x, y = anchor_uv
    if selected:
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(0, 0, 0))
        draw.ellipse(
            (x - 7, y - 7, x + 7, y + 7),
            outline=color,
            width=3,
        )
    _draw_compact_badge(
        draw,
        marker,
        (x + 7, y - 7),
        color,
        font,
        image_size,
        leader_target=(x, y),
        occupied_badges=occupied_badges,
    )


def _draw_direction_marker(
    draw,
    marker,
    anchor_uv,
    selector,
    color,
    font,
    image_size,
    *,
    selected: bool = False,
    occupied_badges: list[tuple[int, int, int, int]],
) -> None:
    if anchor_uv is None:
        return
    delta = {
        "screen_left": (-1, 0),
        "screen_far_left": (-1, -1),
        "screen_far_right": (1, -1),
        "screen_right": (1, 0),
        "screen_near_right": (1, 1),
        "screen_near_left": (-1, 1),
        "screen_near": (0, 1),
        "screen_far": (0, -1),
    }.get(selector, (0, 0))
    x, y = anchor_uv
    # Keep direction labels readable around small references.  The
    # anchor stays on a real visible boundary pixel; only the public arrow and
    # badge extend outwards.
    end = (x + delta[0] * 26, y + delta[1] * 26)
    text_box = draw.textbbox((0, 0), marker, font=font)
    badge_w = text_box[2] - text_box[0] + 6
    badge_h = text_box[3] - text_box[1] + 2
    badge_x = (
        end[0] - badge_w - 4
        if delta[0] < 0
        else end[0] + 4
        if delta[0] > 0
        else end[0] - badge_w // 2
    )
    badge_y = (
        end[1] - badge_h - 4
        if delta[1] < 0
        else end[1] + 4
        if delta[1] > 0
        else end[1] - badge_h // 2
    )
    draw.line(((x, y), end), fill=(0, 0, 0), width=5)
    draw.line(((x, y), end), fill=color, width=2)
    if selected:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=3)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=(0, 0, 0), width=1)
    _draw_compact_badge(
        draw,
        marker,
        (badge_x, badge_y),
        color,
        font,
        image_size,
        selected=selected,
        leader_target=end,
        occupied_badges=occupied_badges,
    )


def _allocate_badge(
    origin,
    width: int,
    height: int,
    image_size,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    max_x = max(0, int(image_size[0]) - width - 1)
    max_y = max(0, int(image_size[1]) - height - 1)
    start_x = max(0, min(int(origin[0]), max_x))
    start_y = max(0, min(int(origin[1]), max_y))
    xs = sorted(
        range(0, max_x + 1, max(1, width + 3)),
        key=lambda value: (abs(value - start_x), value),
    )
    ys = sorted(
        range(0, max_y + 1, max(1, height + 3)),
        key=lambda value: (abs(value - start_y), value),
    )
    for y in (start_y, *ys):
        for x in (start_x, *xs):
            box = (x, y, x + width, y + height)
            if all(
                box[2] + 2 <= other[0]
                or other[2] + 2 <= box[0]
                or box[3] + 2 <= other[1]
                or other[3] + 2 <= box[1]
                for other in occupied
            ):
                occupied.append(box)
                return x, y
    occupied.append((start_x, start_y, start_x + width, start_y + height))
    return start_x, start_y


def _draw_compact_badge(
    draw,
    marker,
    origin,
    color,
    font,
    image_size,
    *,
    selected: bool = False,
    leader_target: tuple[int, int] | None = None,
    occupied_badges: list[tuple[int, int, int, int]],
) -> None:
    text_box = draw.textbbox((0, 0), marker, font=font)
    text_w, text_h = text_box[2] - text_box[0], text_box[3] - text_box[1]
    width, height = text_w + 6, text_h + 2
    x, y = _allocate_badge(
        origin,
        width,
        height,
        image_size,
        occupied_badges,
    )
    badge_box = (x, y, x + width, y + height)
    if leader_target is not None:
        _draw_high_contrast_leader(draw, badge_box, leader_target, color)
    draw.rounded_rectangle(
        badge_box,
        radius=2,
        fill=color if selected else (18, 18, 18),
        outline=color,
        width=1,
    )
    draw.text(
        (x + 3, y + 1 - text_box[1]),
        marker,
        fill=(0, 0, 0) if selected else (255, 255, 255),
        font=font,
    )


def _draw_corner_box(draw, box, color) -> None:
    x0, y0, x1, y1 = box
    corner = max(5, min(12, round(min(x1 - x0, y1 - y0) * 0.35)))
    segments = (
        ((x0, y0), (x0 + corner, y0)), ((x0, y0), (x0, y0 + corner)),
        ((x1, y0), (x1 - corner, y0)), ((x1, y0), (x1, y0 + corner)),
        ((x0, y1), (x0 + corner, y1)), ((x0, y1), (x0, y1 - corner)),
        ((x1, y1), (x1 - corner, y1)), ((x1, y1), (x1, y1 - corner)),
    )
    for start, end in segments:
        draw.line((start, end), fill=(0, 0, 0), width=4)
        draw.line((start, end), fill=color, width=2)


def _draw_high_contrast_leader(draw, badge_box, target, color) -> None:
    """Connect an avoided label to its exact target on bright or dark RGB."""

    x0, y0, x1, y1 = badge_box
    start = ((x0 + x1) // 2, (y0 + y1) // 2)
    target = (int(target[0]), int(target[1]))
    draw.line((start, target), fill=(0, 0, 0), width=4)
    draw.line((start, target), fill=color, width=2)
    draw.ellipse(
        (target[0] - 5, target[1] - 5, target[0] + 5, target[1] + 5),
        fill=(0, 0, 0),
    )
    draw.ellipse(
        (target[0] - 3, target[1] - 3, target[0] + 3, target[1] + 3),
        fill=color,
    )


def _draw_external_badge(
    draw,
    marker,
    box,
    color,
    font,
    image_size,
    *,
    target: tuple[int, int] | None = None,
    selected: bool = False,
    occupied_badges: list[tuple[int, int, int, int]],
) -> None:
    x0, y0, x1, _ = box
    text_box = draw.textbbox((0, 0), marker, font=font)
    text_w, text_h = text_box[2] - text_box[0], text_box[3] - text_box[1]
    badge_w, badge_h = text_w + 8, text_h + 4
    target_x = (x0 + x1) // 2
    badge_x, badge_y = _allocate_badge(
        (target_x - badge_w // 2, y0 - badge_h - 7),
        badge_w,
        badge_h,
        image_size,
        occupied_badges,
    )
    badge_box = (badge_x, badge_y, badge_x + badge_w, badge_y + badge_h)
    leader_target = target or (target_x, max(0, y0))
    _draw_high_contrast_leader(
        draw,
        badge_box,
        leader_target,
        color,
    )
    draw.rounded_rectangle(
        badge_box,
        radius=3,
        fill=color if selected else (18, 18, 18),
        outline=color,
        width=2,
    )
    draw.text(
        (badge_x + 4, badge_y + 2 - text_box[1]),
        marker,
        fill=(0, 0, 0) if selected else (255, 255, 255),
        font=font,
    )
