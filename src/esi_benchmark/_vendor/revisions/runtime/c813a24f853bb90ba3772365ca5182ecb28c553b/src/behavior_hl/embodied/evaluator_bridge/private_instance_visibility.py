"""Private same-frame instance-segmentation gate for point actions.

This module deliberately has no OmniGibson imports.  A live evaluator adapter
may populate :class:`PrivateInstanceSegmentationFrame` from the head
``VisionSensor`` and its remapped ``INSTANCE_REGISTRY``.  The frame and the
resolved simulator object are executor-only values and must never enter a
model prompt or public action result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


PRIVATE_INSTANCE_VISIBILITY_VERSION = (
    "behavior2026.private-instance-visibility/v1"
)
IMAGE_RELATIVE_SELECTORS = frozenset(
    {"bottommost", "center", "leftmost", "rightmost", "topmost"}
)


class VisibilityStatus(str, Enum):
    """Private outcome of resolving one selected image point."""

    VISIBLE = "visible"
    INVALID_POINT = "invalid_point"
    NO_VISIBLE_TASK_OBJECT = "no_visible_task_object"
    AMBIGUOUS_POINT = "ambiguous_point"
    INSUFFICIENT_VISIBLE_PIXELS = "insufficient_visible_pixels"


@dataclass(frozen=True, slots=True)
class PrivateVisibilityConfig:
    """Small anti-speckle thresholds, not a semantic reachability gate."""

    minimum_visible_pixels: int = 8
    point_search_radius_px: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_visible_pixels, bool)
            or not isinstance(self.minimum_visible_pixels, int)
            or self.minimum_visible_pixels <= 0
        ):
            raise ValueError("minimum_visible_pixels must be positive")
        if (
            isinstance(self.point_search_radius_px, bool)
            or not isinstance(self.point_search_radius_px, int)
            or not 0 <= self.point_search_radius_px <= 8
        ):
            raise ValueError("point_search_radius_px must be in [0, 8]")


@dataclass(frozen=True, slots=True)
class PrivateInstanceSegmentationFrame:
    """One segmentation image synchronized to the policy RGB frame."""

    camera: str
    simulator_step: int
    source_rgb_sha256: str
    segmentation: np.ndarray = field(repr=False)
    instance_id_to_name: Mapping[int, str] = field(repr=False)
    part_segmentation: np.ndarray | None = field(default=None, repr=False)
    part_id_to_prim_path: Mapping[int, str] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.camera, str) or not self.camera:
            raise ValueError("camera must be a non-empty string")
        if (
            isinstance(self.simulator_step, bool)
            or not isinstance(self.simulator_step, int)
            or self.simulator_step < 0
        ):
            raise ValueError("simulator_step must be non-negative")
        digest = self.source_rgb_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("source_rgb_sha256 must be lowercase SHA256")
        segmentation = np.asarray(self.segmentation)
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[..., 0]
        if segmentation.ndim != 2 or segmentation.size == 0:
            raise ValueError("segmentation must be a non-empty 2D image")
        if not np.issubdtype(segmentation.dtype, np.integer):
            raise ValueError("segmentation must contain integer instance IDs")
        frozen_segmentation = np.array(segmentation, dtype=np.int64, copy=True)
        frozen_segmentation.setflags(write=False)
        object.__setattr__(self, "segmentation", frozen_segmentation)

        mapping: dict[int, str] = {}
        for raw_id, raw_name in self.instance_id_to_name.items():
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise ValueError("instance registry IDs must be integers")
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError("instance registry names must be non-empty")
            mapping[int(raw_id)] = raw_name
        object.__setattr__(
            self,
            "instance_id_to_name",
            MappingProxyType(mapping),
        )

        part_segmentation = self.part_segmentation
        part_mapping: dict[int, str] = {}
        if part_segmentation is not None:
            part_array = np.asarray(part_segmentation)
            if part_array.ndim == 3 and part_array.shape[-1] == 1:
                part_array = part_array[..., 0]
            if part_array.shape != frozen_segmentation.shape:
                raise ValueError(
                    "part_segmentation must match object segmentation shape"
                )
            if not np.issubdtype(part_array.dtype, np.integer):
                raise ValueError(
                    "part_segmentation must contain integer instance IDs"
                )
            frozen_parts = np.array(part_array, dtype=np.int64, copy=True)
            frozen_parts.setflags(write=False)
            object.__setattr__(self, "part_segmentation", frozen_parts)
            for raw_id, raw_path in self.part_id_to_prim_path.items():
                if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                    raise ValueError("part registry IDs must be integers")
                if not isinstance(raw_path, str) or not raw_path.startswith("/"):
                    raise ValueError(
                        "part registry values must be absolute prim paths"
                    )
                part_mapping[int(raw_id)] = raw_path
        elif self.part_id_to_prim_path:
            raise ValueError(
                "part_id_to_prim_path requires part_segmentation"
            )
        object.__setattr__(
            self,
            "part_id_to_prim_path",
            MappingProxyType(part_mapping),
        )

    def require_rgb_sha256(self, digest: str) -> None:
        """Fail closed when a caller tries to use this with another RGB."""

        if digest != self.source_rgb_sha256:
            raise ValueError(
                "private segmentation is not synchronized to the public RGB"
            )


@dataclass(frozen=True, slots=True)
class PrivateVisibleTarget:
    """Opaque executor-only object binding selected from a visible mask."""

    object_name: str
    instance_id: int
    visible_pixels: int
    selected_pixel_xy: tuple[int, int]
    target_token: Any = field(repr=False)

    def __post_init__(self) -> None:
        if not self.object_name:
            raise ValueError("object_name must not be empty")
        if self.visible_pixels <= 0:
            raise ValueError("visible_pixels must be positive")
        if self.target_token is None:
            raise ValueError("target_token must not be None")


@dataclass(frozen=True, slots=True)
class PrivateVisibilityResolution:
    """Private gate result; failed results never retain an object token."""

    status: VisibilityStatus
    target: PrivateVisibleTarget | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.status is VisibilityStatus.VISIBLE:
            if not isinstance(self.target, PrivateVisibleTarget):
                raise ValueError("visible resolution requires a target")
        elif self.target is not None:
            raise ValueError("failed visibility resolution cannot retain target")


def resolve_visible_task_instance(
    *,
    frame: PrivateInstanceSegmentationFrame,
    normalized_point: Sequence[float],
    task_objects_by_name: Mapping[str, Any],
    config: PrivateVisibilityConfig | None = None,
) -> PrivateVisibilityResolution:
    """Bind a selected point only to a sufficiently visible task object.

    ``task_objects_by_name`` is a private simulator mapping.  The returned
    token is intended for an executor such as the official Starter ``GRASP``
    primitive and must not be serialized into public artifacts.
    """

    if not isinstance(frame, PrivateInstanceSegmentationFrame):
        raise TypeError("frame must be PrivateInstanceSegmentationFrame")
    active_config = config or PrivateVisibilityConfig()
    if not isinstance(active_config, PrivateVisibilityConfig):
        raise TypeError("config must be PrivateVisibilityConfig")
    if not isinstance(task_objects_by_name, Mapping):
        raise TypeError("task_objects_by_name must be a mapping")
    objects = dict(task_objects_by_name)
    if any(
        not isinstance(name, str) or not name or token is None
        for name, token in objects.items()
    ):
        raise ValueError("task object mapping must contain named live objects")

    point = np.asarray(tuple(normalized_point), dtype=np.float64)
    if (
        point.shape != (2,)
        or not np.isfinite(point).all()
        or np.any(point < 0.0)
        or np.any(point > 1.0)
    ):
        return PrivateVisibilityResolution(VisibilityStatus.INVALID_POINT)

    height, width = frame.segmentation.shape
    center_x = int(round(float(point[0]) * (width - 1)))
    center_y = int(round(float(point[1]) * (height - 1)))
    candidates: list[tuple[float, int, int, int]] = []
    radius = active_config.point_search_radius_px
    for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
        for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
            instance_id = int(frame.segmentation[y, x])
            name = frame.instance_id_to_name.get(instance_id)
            if name not in objects:
                continue
            distance = math.hypot(x - center_x, y - center_y)
            candidates.append((distance, instance_id, x, y))
    if not candidates:
        return PrivateVisibilityResolution(
            VisibilityStatus.NO_VISIBLE_TASK_OBJECT
        )

    nearest_distance = min(item[0] for item in candidates)
    nearest = [
        item for item in candidates if abs(item[0] - nearest_distance) <= 1e-9
    ]
    nearest_ids = {item[1] for item in nearest}
    if len(nearest_ids) != 1:
        return PrivateVisibilityResolution(VisibilityStatus.AMBIGUOUS_POINT)
    chosen_id = next(iter(nearest_ids))
    chosen = min(
        (item for item in nearest if item[1] == chosen_id),
        key=lambda item: (item[3], item[2]),
    )
    visible_pixels = int(np.count_nonzero(frame.segmentation == chosen_id))
    if visible_pixels < active_config.minimum_visible_pixels:
        return PrivateVisibilityResolution(
            VisibilityStatus.INSUFFICIENT_VISIBLE_PIXELS
        )
    object_name = frame.instance_id_to_name[chosen_id]
    return PrivateVisibilityResolution(
        VisibilityStatus.VISIBLE,
        PrivateVisibleTarget(
            object_name=object_name,
            instance_id=chosen_id,
            visible_pixels=visible_pixels,
            selected_pixel_xy=(chosen[2], chosen[3]),
            target_token=objects[object_name],
        ),
    )


def resolve_visible_task_category(
    *,
    frame: PrivateInstanceSegmentationFrame,
    object_category: str,
    selector: str | None,
    task_objects_by_category: Mapping[str, Mapping[str, Any]],
    config: PrivateVisibilityConfig | None = None,
) -> PrivateVisibilityResolution:
    """Privately bind a public category and image-relative selector.

    Selection uses only masks in the synchronized current head frame.  It
    never reads world coordinates, object poses, BDDL goal truth, or depth.
    Omitting ``selector`` is admitted only when exactly one sufficiently
    visible matching instance exists.
    """

    if not isinstance(frame, PrivateInstanceSegmentationFrame):
        raise TypeError("frame must be PrivateInstanceSegmentationFrame")
    active_config = config or PrivateVisibilityConfig()
    if not isinstance(active_config, PrivateVisibilityConfig):
        raise TypeError("config must be PrivateVisibilityConfig")
    if not isinstance(object_category, str) or not object_category:
        raise ValueError("object_category must be a non-empty string")
    if selector is not None and selector not in IMAGE_RELATIVE_SELECTORS:
        raise ValueError("selector is not an admitted image-relative selector")
    if not isinstance(task_objects_by_category, Mapping):
        raise TypeError("task_objects_by_category must be a mapping")

    category_objects = task_objects_by_category.get(object_category)
    if not isinstance(category_objects, Mapping) or not category_objects:
        return PrivateVisibilityResolution(
            VisibilityStatus.NO_VISIBLE_TASK_OBJECT
        )
    objects = dict(category_objects)
    if any(
        not isinstance(name, str) or not name or token is None
        for name, token in objects.items()
    ):
        raise ValueError("category object mapping contains an invalid object")

    candidates: list[dict[str, Any]] = []
    saw_matching_pixels = False
    for instance_id, object_name in frame.instance_id_to_name.items():
        if object_name not in objects:
            continue
        ys, xs = np.nonzero(frame.segmentation == instance_id)
        visible_pixels = int(xs.size)
        if visible_pixels == 0:
            continue
        saw_matching_pixels = True
        if visible_pixels < active_config.minimum_visible_pixels:
            continue
        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))
        nearest_index = int(
            np.argmin(
                np.square(xs.astype(np.float64) - centroid_x)
                + np.square(ys.astype(np.float64) - centroid_y)
            )
        )
        candidates.append(
            {
                "object_name": object_name,
                "instance_id": int(instance_id),
                "visible_pixels": visible_pixels,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "selected_pixel_xy": (
                    int(xs[nearest_index]),
                    int(ys[nearest_index]),
                ),
                "target_token": objects[object_name],
            }
        )

    if not candidates:
        return PrivateVisibilityResolution(
            VisibilityStatus.INSUFFICIENT_VISIBLE_PIXELS
            if saw_matching_pixels
            else VisibilityStatus.NO_VISIBLE_TASK_OBJECT
        )
    if len(candidates) > 1:
        if selector is None:
            return PrivateVisibilityResolution(
                VisibilityStatus.AMBIGUOUS_POINT
            )
        metrics = [
            _selector_metric(
                selector,
                centroid_x=item["centroid_x"],
                centroid_y=item["centroid_y"],
                width=frame.segmentation.shape[1],
                height=frame.segmentation.shape[0],
            )
            for item in candidates
        ]
        best = min(metrics)
        winners = [
            item
            for item, metric in zip(candidates, metrics)
            if math.isclose(metric, best, rel_tol=0.0, abs_tol=1e-9)
        ]
        if len(winners) != 1:
            return PrivateVisibilityResolution(
                VisibilityStatus.AMBIGUOUS_POINT
            )
        chosen = winners[0]
    else:
        chosen = candidates[0]

    return PrivateVisibilityResolution(
        VisibilityStatus.VISIBLE,
        PrivateVisibleTarget(
            object_name=chosen["object_name"],
            instance_id=chosen["instance_id"],
            visible_pixels=chosen["visible_pixels"],
            selected_pixel_xy=chosen["selected_pixel_xy"],
            target_token=chosen["target_token"],
        ),
    )


def _selector_metric(
    selector: str,
    *,
    centroid_x: float,
    centroid_y: float,
    width: int,
    height: int,
) -> float:
    if selector == "leftmost":
        return centroid_x
    if selector == "rightmost":
        return -centroid_x
    if selector == "topmost":
        return centroid_y
    if selector == "bottommost":
        return -centroid_y
    if selector == "center":
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        return math.hypot(centroid_x - center_x, centroid_y - center_y)
    raise ValueError("unknown selector")
