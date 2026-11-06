"""Private adapter for BEHAVIOR's official traversability-map A*.

The adapter is used only by the private Oracle author.  It reads the live
scene map and measured robot pose, but returns no value to the public policy
context.  Execution of the returned path remains ordinary registered
``move_*`` / ``turn_*`` actions with physical collision and stability gates.
"""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral
import os
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.interaction_station_geometry import (
    compute_source_geodesic_distances,
    nearest_interaction_target_xy,
    normalize_target_aabb_xy,
    select_interaction_station_cells,
)
from behavior_hl.embodied.evaluator_bridge.private_base_route import XYRoutePlan
from behavior_hl.embodied.evaluator_bridge.safe_path_geometry import (
    greedy_farthest_visible_path_simplification,
)


_PROVIDER_VERSION = "behavior2026.generic-indoor-route-station-yaw/v6-interaction-side-room"
_LOCAL_GUARD_VERSION = "behavior2026.base-footprint-local-segment-guard/v2"
_STATION_HEADING_BIN_COUNT = 36


def structural_terminal_cells_from_aabbs(
    *,
    aabbs_xy: Sequence[Sequence[float]],
    world_to_cell: Callable[[Sequence[float]], tuple[int, int]],
    grid_shape: Sequence[int],
) -> frozenset[tuple[int, int]]:
    """Rasterize live interaction-terminal footprints on one structural grid.

    Planning and the final local-interaction gate must exempt exactly the same
    target/support cells.  Keeping this conversion in one pure helper prevents
    a route from admitting a station that the executor later rejects merely
    because the target occupies several terminal map cells.
    """

    shape = tuple(int(value) for value in grid_shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("grid_shape must contain two positive dimensions")
    cells: set[tuple[int, int]] = set()
    for raw_bounds in aabbs_xy:
        values = tuple(float(value) for value in raw_bounds)
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise ValueError("structural terminal AABB must be finite XY bounds")
        x0, y0, x1, y1 = values
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        corners = (
            world_to_cell((x0, y0)),
            world_to_cell((x0, y1)),
            world_to_cell((x1, y0)),
            world_to_cell((x1, y1)),
        )
        rows = [int(cell[0]) for cell in corners]
        cols = [int(cell[1]) for cell in corners]
        for row in range(max(0, min(rows)), min(shape[0] - 1, max(rows)) + 1):
            for col in range(
                max(0, min(cols)), min(shape[1] - 1, max(cols)) + 1
            ):
                cells.add((row, col))
    return frozenset(cells)


def _station_heading_bin(yaw_rad: float) -> int:
    """Quantize a terminal manipulation heading to the public 10 degree lattice."""

    yaw = float(yaw_rad)
    if not math.isfinite(yaw):
        raise ValueError("station heading must be finite")
    quantum = 2.0 * math.pi / _STATION_HEADING_BIN_COUNT
    wrapped = (yaw + math.pi) % (2.0 * math.pi) - math.pi
    return int(round(wrapped / quantum)) % _STATION_HEADING_BIN_COUNT


def _station_approach_key(
    *,
    station_cell: tuple[int, int],
    station_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> tuple[int, int, int]:
    """Name one terminal station together with the heading needed to interact."""

    final_yaw = math.atan2(
        float(target_xy[1]) - float(station_xy[1]),
        float(target_xy[0]) - float(station_xy[0]),
    )
    return (
        int(station_cell[0]),
        int(station_cell[1]),
        _station_heading_bin(final_yaw),
    )


@dataclass(frozen=True, slots=True)
class _CanonicalMapCellAdapter:
    """One map-cell contract shared by planning, validation, and telemetry.

    A live OmniGibson scene must use the map object's own Torch-backed
    transforms.  The arithmetic fallback exists only for pure-Python doubles
    that do not expose those methods; its use is always reported.
    """

    resolution: float
    map_size: float
    world_to_map: Callable[[Any], Any] | None
    map_to_world: Callable[[Any], Any] | None
    tensor_factory: Callable[[Sequence[float]], Any]

    @property
    def transform_source(self) -> str:
        return (
            "official_world_to_map"
            if callable(self.world_to_map) and callable(self.map_to_world)
            else "test_fallback"
        )

    def world_to_cell(self, point: Sequence[float]) -> tuple[int, int]:
        xy = _xy(point, "world point")
        if callable(self.world_to_map):
            values = _flat_pair(
                self.world_to_map(self.tensor_factory(xy)),
                "official world_to_map",
            )
            return int(values[0]), int(values[1])
        return _world_to_cell_fallback(xy, self.resolution, self.map_size)

    def cell_to_world(self, cell: Sequence[float]) -> tuple[float, float]:
        row_col = _xy(cell, "map cell")
        if callable(self.map_to_world):
            values = _flat_pair(
                self.map_to_world(self.tensor_factory(row_col)),
                "official map_to_world",
            )
            return float(values[0]), float(values[1])
        return _cell_to_world_fallback(
            (int(row_col[0]), int(row_col[1])),
            self.resolution,
            self.map_size,
        )


@dataclass(frozen=True, slots=True)
class _OfficialPathSanitizationResult:
    status: Literal[
        "valid",
        "official_astar_returned_none",
        "official_path_empty",
        "official_path_shape_invalid",
        "official_path_nonfinite",
        "official_path_local_validation_mismatch",
        "cell_transform_error",
    ]
    waypoints: tuple[tuple[float, float], ...] | None
    source_snap_required: bool
    private_report: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class OfficialTraversabilityRouteResult:
    """One private route-provider outcome.

    ``unavailable`` means the evaluator is a pure test double or the scene has
    no official traversability API, so a caller may use its legacy test-only
    fallback.  ``no_route`` means the live official map was available and
    failed closed; callers must not silently drive a direct line instead.
    """

    status: Literal["planned", "unavailable", "no_route"]
    plan: XYRoutePlan | None
    private_report: Mapping[str, Any]
    reachable_official_cells: frozenset[tuple[int, int]] | None = None
    station_candidates: tuple[
        tuple[tuple[int, int], tuple[float, float]], ...
    ] = ()


@dataclass(frozen=True, slots=True)
class OfficialLocalSegmentGuardResult:
    """Private static-map decision for one bounded local base segment."""

    traversable: bool
    private_report: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _StableRobotFootprintResult:
    """Private, arm-pose-invariant base footprint used for map erosion.

    OmniGibson's historical ``reset_joint_pos_aabb_extent`` is documented as
    a chassis extent, but it is populated from the complete articulated robot
    AABB.  A presampled reset pose with an extended arm can therefore make the
    same mobile base appear much wider in one task than another.  This result
    keeps the chosen extent and its provenance inseparable so route artifacts
    cannot silently describe that mutable full-body AABB as a base footprint.
    """

    extent_xy_m: tuple[float, float] | None
    private_report: Mapping[str, Any]


def official_world_to_map_cell(
    *,
    trav_map: Any,
    point: Sequence[float],
    tensor_factory: Callable[[Sequence[float]], Any] | None = None,
) -> tuple[int, int] | None:
    """Convert one live world point with the canonical official contract."""

    resolution = float(getattr(trav_map, "map_resolution", float("nan")))
    map_size = float(getattr(trav_map, "map_size", float("nan")))
    if (
        not math.isfinite(resolution)
        or resolution <= 0.0
        or not math.isfinite(map_size)
        or map_size <= 0.0
    ):
        return None
    adapter = _canonical_cell_adapter(
        trav_map=trav_map,
        resolution=resolution,
        map_size=map_size,
        tensor_factory=tensor_factory or _torch_tensor,
        allow_test_fallback=False,
    )
    return None if adapter is None else adapter.world_to_cell(point)


def check_official_local_segment(
    *,
    scene: Any,
    robot: Any,
    start_xy: Sequence[float],
    target_xy: Sequence[float],
    robot_z: float,
    start_yaw_rad: float = 0.0,
    target_yaw_rad: float | None = None,
    allow_bounded_source_snap: bool = True,
    allow_test_fallback: bool = False,
) -> OfficialLocalSegmentGuardResult:
    """Check one local base action against the official traversability map.

    Translation uses the oriented base collision rectangle. Rotation uses its
    circumscribed circle. Arms, grippers, held objects, and arbitrary clearance
    margins are deliberately excluded here; the later live CuRobo guard checks
    the complete current robot. This function performs no A* detour and never
    mutates simulator state.
    """

    trav_map = getattr(scene, "trav_map", None)
    floor_maps = getattr(trav_map, "floor_map", None)
    floor_heights = getattr(trav_map, "floor_heights", None)
    if (
        trav_map is None
        or floor_maps is None
        or floor_heights is None
        or len(floor_maps) == 0
        or len(floor_heights) == 0
    ):
        return OfficialLocalSegmentGuardResult(
            False,
            {
                "guard_version": _LOCAL_GUARD_VERSION,
                "reason": "official_traversability_api_unavailable",
            },
        )
    start = _xy(start_xy, "start_xy")
    target = _xy(target_xy, "target_xy")
    start_yaw = float(start_yaw_rad)
    target_yaw = start_yaw if target_yaw_rad is None else float(target_yaw_rad)
    if not math.isfinite(start_yaw) or not math.isfinite(target_yaw):
        raise ValueError("local segment yaw must be finite")
    if not isinstance(allow_bounded_source_snap, bool):
        raise TypeError("allow_bounded_source_snap must be boolean")
    if not isinstance(allow_test_fallback, bool):
        raise TypeError("allow_test_fallback must be boolean")
    floor = min(
        range(len(floor_heights)),
        key=lambda index: abs(float(floor_heights[index]) - float(robot_z)),
    )
    try:
        grid = np.asarray(_numpy(floor_maps[floor]))
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        return OfficialLocalSegmentGuardResult(
            False,
            {
                "guard_version": _LOCAL_GUARD_VERSION,
                "floor": int(floor),
                "reason": "official_floor_map_failed",
                "error_type": type(exc).__name__,
            },
        )
    resolution = float(getattr(trav_map, "map_resolution", float("nan")))
    map_size = float(getattr(trav_map, "map_size", grid.shape[0]))
    if (
        grid.ndim != 2
        or grid.size == 0
        or not math.isfinite(resolution)
        or resolution <= 0.0
        or not math.isfinite(map_size)
        or map_size <= 0.0
    ):
        return OfficialLocalSegmentGuardResult(
            False,
            {
                "guard_version": _LOCAL_GUARD_VERSION,
                "floor": int(floor),
                "reason": "invalid_official_floor_map",
            },
        )

    base_bounds, base_report = _base_collision_bounds(robot)
    if base_bounds is None:
        return OfficialLocalSegmentGuardResult(
            False,
            {
                "guard_version": _LOCAL_GUARD_VERSION,
                "floor": int(floor),
                "reason": "base_collision_geometry_unavailable",
                **base_report,
            },
        )

    cell_adapter = _canonical_cell_adapter(
        trav_map=trav_map,
        resolution=resolution,
        map_size=map_size,
        tensor_factory=_torch_tensor,
        allow_test_fallback=allow_test_fallback,
    )
    if cell_adapter is None:
        return OfficialLocalSegmentGuardResult(
            False,
            {
                "guard_version": _LOCAL_GUARD_VERSION,
                "floor": int(floor),
                "reason": "official_map_transform_unavailable",
                "cell_transform_source": "unavailable",
            },
        )
    map_telemetry = _eroded_map_telemetry(grid, cell_adapter)

    length = _distance(start, target)
    yaw_delta = math.atan2(
        math.sin(target_yaw - start_yaw),
        math.cos(target_yaw - start_yaw),
    )
    turning = abs(yaw_delta) > 1.0e-9
    sample_count = max(
        1,
        int(
            math.ceil(
                length / max(0.01, min(0.05, resolution * 0.5))
            )
        ),
    )
    open_samples: list[bool] = []
    sampled_cells: list[tuple[int, int]] = []
    try:
        for index in range(sample_count + 1):
            fraction = index / sample_count
            point = (
                start[0] + fraction * (target[0] - start[0]),
                start[1] + fraction * (target[1] - start[1]),
            )
            row, col = cell_adapter.world_to_cell(point)
            sampled_cells.append((row, col))
            open_samples.append(
                _base_turn_circle_open(
                    grid=grid,
                    center_cell=(row, col),
                    bounds_xy=base_bounds,
                    resolution=resolution,
                )
                if turning
                else _oriented_base_rectangle_open(
                    grid=grid,
                    center_xy=point,
                    yaw_rad=start_yaw,
                    bounds_xy=base_bounds,
                    world_to_cell=cell_adapter.world_to_cell,
                )
            )
    except (
        AttributeError,
        ImportError,
        IndexError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return OfficialLocalSegmentGuardResult(
            False,
            {
                "guard_version": _LOCAL_GUARD_VERSION,
                "floor": int(floor),
                "reason": "cell_transform_error",
                "error_type": type(exc).__name__,
                **map_telemetry,
            },
        )

    source_open = open_samples[0]
    safe_prefix_last_index = -1
    if source_open:
        for index, sample_open in enumerate(open_samples):
            if not sample_open:
                break
            safe_prefix_last_index = index
    safe_prefix_fraction = (
        0.0
        if safe_prefix_last_index < 0
        else safe_prefix_last_index / sample_count
    )
    safe_prefix_length_m = length * safe_prefix_fraction
    bounded_source_snap = False
    traversable = all(open_samples)
    reason = "base_footprint_segment_open" if traversable else "navigation_map_blocked"
    if not traversable and not source_open and allow_bounded_source_snap:
        try:
            first_open = open_samples.index(True)
        except ValueError:
            first_open = -1
        if first_open > 0 and all(open_samples[first_open:]):
            traversable = True
            bounded_source_snap = True
            reason = "bounded_source_snap_then_open"

    return OfficialLocalSegmentGuardResult(
        traversable,
        {
            "guard_version": _LOCAL_GUARD_VERSION,
            "floor": int(floor),
            "map_with_objects": bool(
                getattr(trav_map, "trav_map_with_objects", False)
            ),
            "segment_length_m": float(length),
            "sample_count": int(sample_count + 1),
            "source_official_cell": list(sampled_cells[0]),
            "target_official_cell": list(sampled_cells[-1]),
            "source_open": bool(source_open),
            "safe_prefix_fraction": float(safe_prefix_fraction),
            "safe_prefix_length_m": float(safe_prefix_length_m),
            "bounded_source_snap": bounded_source_snap,
            "static_map_policy": (
                "base_circumscribed_circle"
                if turning
                else "oriented_base_collision_rectangle"
            ),
            "static_clearance_margin_m": 0.0,
            "legacy_full_robot_aabb_erosion_used": False,
            "base_collision_bounds_xy_m": list(base_bounds),
            "base_collision_extent_xy_m": [
                float(base_bounds[2] - base_bounds[0]),
                float(base_bounds[3] - base_bounds[1]),
            ],
            "requested_yaw_delta_deg": float(math.degrees(yaw_delta)),
            "dynamic_collision_guard_claimed": False,
            "reason": reason,
            **base_report,
            **map_telemetry,
        },
    )


def prepare_official_endpoint_check(*, scene, robot, pose):
    """Snapshot the Action static context for one synchronous planner call.

    The public guard performs all metadata validation once. Its same measured
    asymmetric bounds, raw floor map, official transform and rectangle kernel
    are then reused, without repeated whole-map telemetry scans per successor.
    No caller may retain this closure across an executed action or reset.
    """
    initial = check_official_local_segment(
        scene=scene, robot=robot, start_xy=(pose.x, pose.y), target_xy=(pose.x, pose.y),
        robot_z=pose.z, start_yaw_rad=pose.yaw, target_yaw_rad=pose.yaw,
        allow_bounded_source_snap=False,
    )
    report = dict(initial.private_report)
    if report.get("reason") not in {"base_footprint_segment_open", "navigation_map_blocked"}:
        return None, report
    trav_map = scene.trav_map
    grid = np.array(_numpy(trav_map.floor_map[report["floor"]]), copy=True)
    adapter = _canonical_cell_adapter(
        trav_map=trav_map, resolution=float(trav_map.map_resolution),
        map_size=float(getattr(trav_map, "map_size", grid.shape[0])),
        tensor_factory=_torch_tensor, allow_test_fallback=False,
    )
    if adapter is None:
        return None, {**report, "reason": "official_map_transform_unavailable"}
    bounds = tuple(report["base_collision_bounds_xy_m"])

    def endpoint_open(endpoint):
        if endpoint.z != pose.z:
            raise ValueError("endpoint static snapshot cannot cross floors")
        return _oriented_base_rectangle_open(
            grid=grid, center_xy=(endpoint.x, endpoint.y), yaw_rad=endpoint.yaw,
            bounds_xy=bounds, world_to_cell=adapter.world_to_cell,
        )

    return endpoint_open, report


@lru_cache(maxsize=16)
def _load_structural_open_grid(
    scene_dir: str,
    floor: int,
    map_size: int,
) -> np.ndarray | None:
    """Load the official no-object floor map used only for wall LOS checks.

    The navigation map intentionally keeps scene objects.  This second map has
    one narrower responsibility: reject an interaction station whose straight
    station-to-target ray crosses permanent building structure.  Keeping this
    immutable layout map cached avoids one disk read for every task leaf.
    """

    if not scene_dir or floor < 0 or map_size <= 0:
        return None
    path = Path(scene_dir) / "layout" / f"floor_trav_no_obj_{floor}.png"
    if not path.is_file():
        return None
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.ndim != 2:
            return None
        # Match TraversableMap._load_map exactly: OpenCV's default linear
        # resize followed by the strict 255 threshold.  Nearest-neighbour
        # resizing can move a thin wall boundary by one navigation cell and
        # make the station-to-target LOS gate disagree with the official map.
        resized = cv2.resize(image, (map_size, map_size))
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return None
    result = np.asarray(resized == 255, dtype=bool)
    result.setflags(write=False)
    return result


def _scene_room_instance_grid(
    scene: Any,
    *,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray | None, Mapping[str, int]]:
    """Return the official room-instance raster in navigation-map resolution."""

    segmentation_map = getattr(scene, "seg_map", None)
    if segmentation_map is None:
        segmentation_map = getattr(scene, "_seg_map", None)
    raw = getattr(segmentation_map, "room_ins_map", None)
    name_to_id = getattr(segmentation_map, "room_ins_name_to_ins_id", None)
    if raw is None:
        return None, {}
    try:
        values = np.asarray(_numpy(raw))
        if values.ndim != 2:
            return None, {}
        if values.shape != expected_shape:
            import cv2

            values = cv2.resize(
                values,
                (expected_shape[1], expected_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        result = np.asarray(values, dtype=np.int64)
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return None, {}
    mapping = (
        {str(name): int(value) for name, value in name_to_id.items()}
        if isinstance(name_to_id, Mapping)
        else {}
    )
    return result, mapping


def _mask_to_loaded_rooms(
    grid: np.ndarray,
    *,
    scene: Any,
    room_instance_grid: np.ndarray | None,
    room_name_to_id: Mapping[str, int],
    structural_open_grid: np.ndarray | None,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Restrict a private route to rooms actually loaded for this scene.

    A one-cell structural halo is admitted only on background-labelled pixels
    adjacent to an authorised room.  It preserves doorway / archway boundary
    pixels without opening an unrequested neighbouring room or the exterior.
    """

    loaded_names = getattr(scene, "load_room_instances", None)
    if (
        room_instance_grid is None
        or not isinstance(loaded_names, Sequence)
        or isinstance(loaded_names, (str, bytes))
        or not loaded_names
    ):
        return np.array(grid, copy=True), {
            "loaded_room_mask_applied": False,
            "loaded_room_instances": [],
            "loaded_room_mask_reason": "scene_loaded_room_filter_unavailable",
        }
    loaded_ids = tuple(
        sorted(
            {
                int(room_name_to_id[str(name)])
                for name in loaded_names
                if str(name) in room_name_to_id
            }
        )
    )
    if not loaded_ids:
        return np.array(grid, copy=True), {
            "loaded_room_mask_applied": False,
            "loaded_room_instances": [str(name) for name in loaded_names],
            "loaded_room_mask_reason": "loaded_room_ids_unresolved",
        }
    authorised = np.isin(room_instance_grid, loaded_ids)
    portal_halo = np.zeros_like(authorised, dtype=bool)
    if structural_open_grid is not None:
        try:
            import cv2

            adjacent = cv2.dilate(
                authorised.astype(np.uint8),
                np.ones((3, 3), dtype=np.uint8),
            ).astype(bool)
            portal_halo = (
                adjacent
                & (room_instance_grid == 0)
                & np.asarray(structural_open_grid, dtype=bool)
            )
        except (ImportError, RuntimeError, TypeError, ValueError):
            portal_halo = np.zeros_like(authorised, dtype=bool)
    allowed = authorised | portal_halo
    masked = np.where(allowed, grid, 0).astype(grid.dtype, copy=False)
    return masked, {
        "loaded_room_mask_applied": True,
        "loaded_room_instances": [str(name) for name in loaded_names],
        "loaded_room_ids": list(loaded_ids),
        "loaded_room_allowed_cell_count": int(np.count_nonzero(allowed)),
        "loaded_room_portal_halo_cell_count": int(np.count_nonzero(portal_halo)),
    }


def _finite_xy_extent(value: Any) -> tuple[float, float] | None:
    """Normalize one candidate extent without importing simulator types."""

    try:
        values = _numpy(value).astype(np.float64, copy=False).reshape(-1)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if values.size < 2 or not np.isfinite(values[:2]).all():
        return None
    result = (float(values[0]), float(values[1]))
    if result[0] <= 0.0 or result[1] <= 0.0:
        return None
    return result


def _finite_collision_points(value: Any) -> np.ndarray | None:
    """Return finite collision points with at least XY coordinates."""

    try:
        points = _numpy(value).astype(np.float64, copy=False)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if (
        points.ndim != 2
        or points.shape[0] < 2
        or points.shape[1] < 2
        or not np.isfinite(points[:, : min(3, points.shape[1])]).all()
    ):
        return None
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(points.shape[0])))
    return np.asarray(points[:, :3], dtype=np.float64)


def _finite_transform(value: Any) -> np.ndarray | None:
    try:
        transform = _numpy(value).astype(np.float64, copy=False)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return None
    return transform


def _oriented_base_rectangle_open(
    *,
    grid: np.ndarray,
    center_xy: tuple[float, float],
    yaw_rad: float,
    bounds_xy: tuple[float, float, float, float],
    world_to_cell: Callable[[Sequence[float]], tuple[int, int]],
) -> bool:
    """Raster-check the exact oriented base rectangle with no added margin."""

    import cv2

    x0, y0, x1, y1 = bounds_xy
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    cells = []
    for local_x, local_y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        cells.append(
            world_to_cell(
                (
                    center_xy[0] + cosine * local_x - sine * local_y,
                    center_xy[1] + sine * local_x + cosine * local_y,
                )
            )
        )
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    row0, row1 = min(rows), max(rows)
    col0, col1 = min(cols), max(cols)
    if row0 < 0 or col0 < 0 or row1 >= grid.shape[0] or col1 >= grid.shape[1]:
        return False
    mask = np.zeros((row1 - row0 + 1, col1 - col0 + 1), dtype=np.uint8)
    polygon = np.asarray(
        [(col - col0, row - row0) for row, col in cells],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, polygon, 1)
    return bool(np.all(grid[row0 : row1 + 1, col0 : col1 + 1][mask != 0] != 0))


def _base_turn_circle_open(
    *,
    grid: np.ndarray,
    center_cell: tuple[int, int],
    bounds_xy: tuple[float, float, float, float],
    resolution: float,
) -> bool:
    """Check every map-cell center inside the base's rotation envelope."""

    x0, y0, x1, y1 = bounds_xy
    radius_cells = max(
        math.hypot(x, y) for x in (x0, x1) for y in (y0, y1)
    ) / resolution
    row, col = center_cell
    radius_limit = int(math.ceil(radius_cells))
    row0, row1 = row - radius_limit, row + radius_limit
    col0, col1 = col - radius_limit, col + radius_limit
    if row0 < 0 or col0 < 0 or row1 >= grid.shape[0] or col1 >= grid.shape[1]:
        return False
    rows, cols = np.ogrid[row0 : row1 + 1, col0 : col1 + 1]
    mask = (rows - row) ** 2 + (cols - col) ** 2 <= radius_cells**2
    return bool(np.all(grid[row0 : row1 + 1, col0 : col1 + 1][mask] != 0))


def _base_collision_bounds(robot: Any) -> tuple[
    tuple[float, float, float, float] | None,
    dict[str, Any],
]:
    """Measure base collision bounds in the base link's local frame.

    The base link is always eligible.  Floor-touching links (normally wheels)
    are included only when both transforms can be resolved, so geometry from a
    different local frame can never be unioned accidentally.  Arm, gripper,
    torso, and held-object links are deliberately never enumerated.
    """

    try:
        base_link = getattr(robot, "base_footprint_link")
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        base_link = None
    if base_link is None:
        return None, {
            "base_collision_geometry_available": False,
            "base_collision_geometry_reason": "base_footprint_link_unavailable",
        }

    try:
        base_points = _finite_collision_points(
            getattr(base_link, "collision_boundary_points_local")
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        base_points = None
    if base_points is None:
        return None, {
            "base_collision_geometry_available": False,
            "base_collision_geometry_reason": (
                "base_footprint_link_collision_points_unavailable"
            ),
        }

    point_sets = [base_points]
    included_links = [str(getattr(robot, "base_footprint_link_name", "base"))]
    skipped_links: list[str] = []
    base_transform = _finite_transform(getattr(base_link, "scaled_transform", None))
    base_inverse: np.ndarray | None = None
    if base_transform is not None:
        try:
            base_inverse = np.linalg.inv(base_transform)
        except np.linalg.LinAlgError:
            base_inverse = None

    # Prefer the official resolved link objects.  A names+links fallback keeps
    # this adapter usable with lightweight doubles and older OmniGibson APIs.
    try:
        auxiliary_links = tuple(getattr(robot, "floor_touching_base_links") or ())
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        auxiliary_links = ()
    if not auxiliary_links:
        try:
            names = tuple(getattr(robot, "floor_touching_base_link_names") or ())
            links = getattr(robot, "links")
            auxiliary_links = tuple(links[name] for name in names if name in links)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            auxiliary_links = ()

    for index, link in enumerate(auxiliary_links):
        if link is base_link:
            continue
        link_name = str(getattr(link, "name", f"floor_touching_base_link_{index}"))
        try:
            points = _finite_collision_points(
                getattr(link, "collision_boundary_points_local")
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            points = None
        link_transform = _finite_transform(getattr(link, "scaled_transform", None))
        if points is None or base_inverse is None or link_transform is None:
            skipped_links.append(link_name)
            continue
        homogeneous = np.column_stack((points, np.ones(points.shape[0])))
        points_in_base = (homogeneous @ (base_inverse @ link_transform).T)[:, :3]
        if not np.isfinite(points_in_base).all():
            skipped_links.append(link_name)
            continue
        point_sets.append(points_in_base)
        included_links.append(link_name)

    all_points = np.concatenate(point_sets, axis=0)
    lower = np.min(all_points[:, :2], axis=0)
    upper = np.max(all_points[:, :2], axis=0)
    extent = upper - lower
    normalized = _finite_xy_extent(extent)
    if normalized is None:
        return None, {
            "base_collision_geometry_available": False,
            "base_collision_geometry_reason": "base_collision_extent_invalid",
        }
    bounds = (
        float(lower[0]),
        float(lower[1]),
        float(upper[0]),
        float(upper[1]),
    )
    return bounds, {
        "base_collision_geometry_available": True,
        "base_collision_frame": "base_footprint_link_local",
        "base_collision_included_links": included_links,
        "base_collision_skipped_auxiliary_links": skipped_links,
    }


def _select_stable_robot_footprint(robot: Any) -> _StableRobotFootprintResult:
    """Select a stable chassis footprint and record all relevant provenance.

    Selection order is intentionally independent of task/category:

    1. collision geometry attached to the official base footprint link;
    2. an explicit reset/stable *base* extent exposed by a robot adapter;
    3. OmniGibson's cached reset-joint AABB as a compatibility fallback.

    The mutable ``robot.aabb_extent`` is never read.  Consequently an arm pose
    or a held object cannot alter the erosion kernel during an episode.
    """

    base_bounds, base_report = _base_collision_bounds(robot)
    base_extent = (
        None
        if base_bounds is None
        else (
            base_bounds[2] - base_bounds[0],
            base_bounds[3] - base_bounds[1],
        )
    )
    reset_extent: tuple[float, float] | None = None
    reset_error: str | None = None
    try:
        reset_extent = _finite_xy_extent(
            getattr(robot, "reset_joint_pos_aabb_extent")
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        reset_error = type(exc).__name__

    explicit_candidates: list[tuple[str, tuple[float, float]]] = []
    for attribute in (
        "reset_base_footprint_extent",
        "stable_base_footprint_extent",
        "base_collision_aabb_extent",
    ):
        try:
            extent = _finite_xy_extent(getattr(robot, attribute))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            extent = None
        if extent is not None:
            explicit_candidates.append((attribute, extent))

    rejected: list[dict[str, Any]] = [
        {
            "source": "robot.aabb_extent",
            "reason": "mutable_full_body_aabb_not_read",
        }
    ]
    selected_extent: tuple[float, float] | None = None
    selected_source: str | None = None
    if base_extent is not None:
        selected_extent = base_extent
        selected_source = "base_collision_geometry_local"
        if reset_extent is not None:
            reset_diagonal = math.hypot(*reset_extent)
            base_diagonal = math.hypot(*base_extent)
            reset_is_inflated = reset_diagonal > base_diagonal * 1.20
            rejected.append(
                {
                    "source": "reset_joint_pos_aabb_extent",
                    "extent_xy_m": list(reset_extent),
                    "reason": (
                        "likely_articulated_full_body_aabb"
                        if reset_is_inflated
                        else "lower_priority_than_base_collision_geometry"
                    ),
                    "diagonal_ratio_to_selected": float(
                        reset_diagonal / base_diagonal
                    ),
                }
            )
    elif explicit_candidates:
        selected_source, selected_extent = explicit_candidates[0]
        selected_source = f"robot_adapter.{selected_source}"
    elif reset_extent is not None:
        # Compatibility with OmniGibson robots that do not expose usable base
        # collision points.  This property is cached at robot initialization,
        # so it is stable within an episode even though its scope is less
        # precise than the preferred base-only source.
        selected_extent = reset_extent
        selected_source = "omnigibson.reset_joint_pos_aabb_extent_fallback"

    report: dict[str, Any] = {
        **base_report,
        "robot_footprint_source": selected_source,
        "robot_footprint_extent_m": (
            None if selected_extent is None else list(selected_extent)
        ),
        "robot_footprint_arm_pose_invariant": selected_extent is not None,
        "robot_mutable_full_body_aabb_used": False,
        "robot_reset_joint_aabb_extent_m": (
            None if reset_extent is None else list(reset_extent)
        ),
        "robot_reset_joint_aabb_read_error": reset_error,
        "robot_footprint_rejected_sources": rejected,
    }
    if selected_extent is None:
        report["robot_footprint_selection_reason"] = (
            "no_stable_base_or_reset_footprint_source"
        )
    elif selected_source == "omnigibson.reset_joint_pos_aabb_extent_fallback":
        report["robot_footprint_selection_reason"] = (
            "base_collision_unavailable_compatibility_fallback"
        )
    else:
        report["robot_footprint_selection_reason"] = "stable_base_source_selected"
    return _StableRobotFootprintResult(selected_extent, report)


def plan_official_traversability_route(
    *,
    scene: Any,
    robot: Any,
    start_xy: Sequence[float],
    target_xy: Sequence[float],
    target_aabb_xy: Sequence[float] | None = None,
    target_room_names: Sequence[str] | None = None,
    structural_terminal_aabbs_xy: Sequence[Sequence[float]] = (),
    robot_z: float,
    desired_distance_m: float,
    maximum_station_distance_m: float = 1.26,
    maximum_endpoint_snap_m: float = 0.30,
    angular_samples: int = 32,
    maximum_astar_calls: int = 8,
    tensor_factory: Callable[[Sequence[float]], Any] | None = None,
    followup_target_xy: Sequence[float] | None = None,
    followup_target_aabb_xy: Sequence[float] | None = None,
    followup_target_room_names: Sequence[str] | None = None,
    followup_desired_distance_m: float | None = None,
    official_cell_astar: Callable[[Any, tuple[int, int], tuple[int, int]], Any]
    | None = None,
    official_four_connected_astar: Callable[
        [Any, tuple[int, int], tuple[int, int]], Any
    ]
    | None = None,
    additional_blocked_cells: Sequence[Sequence[int]] = (),
    excluded_station_approaches: Sequence[Sequence[int]] = (),
    floor_map_override: Any | None = None,
    floor_map_override_provenance: str | None = None,
    robot_erosion_margin_m: float | None = None,
    robot_erosion_kernel_pixels: int | None = None,
    allow_cross_room_station_when_target_room_known: bool = False,
    allow_test_fallback: bool = False,
) -> OfficialTraversabilityRouteResult:
    """Plan to a manipulation station with the official eroded floor map.

    ``floor_map_override`` is a private, run-local clone of the official floor
    map.  It is used for state-dependent topology such as one door whose live
    official ``Open`` postcondition has already been verified.  It must match
    the official map shape and is cloned again before robot-footprint erosion;
    the scene map is never mutated.

    ``robot_erosion_margin_m`` is an explicit research-mode override for the
    extra clearance around a stable base-only footprint. ``None`` preserves
    OmniGibson's official erosion. A numeric value rebuilds the grid from the
    same official floor map; walls and furniture remain blocked. The mutable
    live full-body AABB is never used because arm and held-object poses must
    not change global map connectivity.

    ``robot_erosion_kernel_pixels`` is a narrower research override used by
    the high-level semantic executor.  It fixes the base-only morphological
    kernel independently of the map discretisation's ceil boundary.  It may
    only be supplied together with an explicit margin and never disables the
    later structural swept-segment guard.

    ``target_room_names`` and ``followup_target_room_names`` bind the official
    object ``in_rooms`` metadata to the segmentation-map instance ids.  A
    target-centre pixel is only a fallback when no official room metadata was
    supplied; wall-mounted targets must not inherit the room on the other side
    of the wall.

    ``structural_terminal_aabbs_xy`` describes live target/support footprints
    that may occupy the terminal part of an otherwise clear interaction ray.
    This lets an elevated object on a table be approached to the table edge
    without treating that table as a wall.  Only blocked cells inside these
    live AABBs are exempted; an intermediate wall remains a hard gate.

    ``additional_blocked_cells`` is a private, run-local overlay containing
    only locations that the live CuRobo candidate-pose guard actually queried
    and predicted colliding.  A merely checked successor endpoint is not
    eligible when a midpoint query short-circuited the guard.  The overlay is
    applied only to a clone of the official robot-eroded map; neither the
    scene map nor any public observation is mutated.  The measured source cell
    is always preserved so a collision report can never manufacture a source
    snap.

    ``excluded_station_approaches`` contains terminal
    ``(row, col, final_yaw_bin)`` keys that already failed live full-body
    validation during this navigation leg.  Excluding the approach, rather
    than the entire XY cell, allows the same traversable cell to remain usable
    at another interaction heading.
    """

    trav_map = getattr(scene, "trav_map", None)
    shortest_path = getattr(scene, "get_shortest_path", None)
    floor_maps = getattr(trav_map, "floor_map", None)
    floor_heights = getattr(trav_map, "floor_heights", None)
    if (
        trav_map is None
        or not callable(shortest_path)
        or floor_maps is None
        or floor_heights is None
        or len(floor_maps) == 0
        or len(floor_heights) == 0
    ):
        return OfficialTraversabilityRouteResult(
            status="unavailable",
            plan=None,
            private_report={
                "provider_version": _PROVIDER_VERSION,
                "reason": "official_traversability_api_unavailable",
            },
        )
    start = _xy(start_xy, "start_xy")
    target = _xy(target_xy, "target_xy")
    target_bounds = normalize_target_aabb_xy(target_aabb_xy)
    desired = _positive(desired_distance_m, "desired_distance_m")
    maximum = _positive(
        maximum_station_distance_m,
        "maximum_station_distance_m",
    )
    snap_limit = _positive(maximum_endpoint_snap_m, "maximum_endpoint_snap_m")
    if not isinstance(allow_test_fallback, bool):
        raise TypeError("allow_test_fallback must be boolean")
    if not isinstance(allow_cross_room_station_when_target_room_known, bool):
        raise TypeError(
            "allow_cross_room_station_when_target_room_known must be boolean"
        )
    if robot_erosion_margin_m is not None:
        if (
            isinstance(robot_erosion_margin_m, bool)
            or not isinstance(robot_erosion_margin_m, (int, float))
            or not math.isfinite(float(robot_erosion_margin_m))
            or float(robot_erosion_margin_m) < 0.0
        ):
            raise ValueError("robot_erosion_margin_m must be finite and non-negative")
    if robot_erosion_kernel_pixels is not None:
        if robot_erosion_margin_m is None:
            raise ValueError(
                "robot_erosion_kernel_pixels requires an explicit erosion margin"
            )
        if (
            isinstance(robot_erosion_kernel_pixels, bool)
            or not isinstance(robot_erosion_kernel_pixels, int)
            or robot_erosion_kernel_pixels < 1
        ):
            raise ValueError("robot_erosion_kernel_pixels must be a positive integer")
    if (followup_target_xy is None) != (followup_desired_distance_m is None):
        raise ValueError(
            "followup target and desired distance must be provided together"
        )
    if followup_target_xy is None and followup_target_room_names:
        raise ValueError("followup room names require a followup target")
    if followup_target_xy is None and followup_target_aabb_xy is not None:
        raise ValueError("followup target AABB requires a followup target")

    def normalize_room_names(
        values: Sequence[str] | None,
        *,
        argument: str,
    ) -> tuple[str, ...]:
        if values is None:
            return ()
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{argument} must be a string sequence")
        result = tuple(
            sorted(
                {
                    str(value)
                    for value in values
                    if isinstance(value, str) and value
                }
            )
        )
        if len(result) != len(tuple(values)):
            raise ValueError(f"{argument} contains invalid or duplicate names")
        return result

    target_rooms = normalize_room_names(
        target_room_names,
        argument="target_room_names",
    )
    followup_target_rooms = normalize_room_names(
        followup_target_room_names,
        argument="followup_target_room_names",
    )
    followup_target = (
        None
        if followup_target_xy is None
        else _xy(followup_target_xy, "followup_target_xy")
    )
    followup_target_bounds = normalize_target_aabb_xy(
        followup_target_aabb_xy
    )
    followup_desired = (
        None
        if followup_desired_distance_m is None
        else _positive(
            followup_desired_distance_m,
            "followup_desired_distance_m",
        )
    )
    if desired > maximum:
        raise ValueError("desired station distance exceeds maximum")
    if angular_samples < 8 or maximum_astar_calls < 1:
        raise ValueError("route sampling budgets are invalid")
    floor = min(
        range(len(floor_heights)),
        key=lambda index: abs(float(floor_heights[index]) - float(robot_z)),
    )
    try:
        floor_map = floor_maps[floor]
        if floor_map_override is not None:
            official_shape = np.asarray(_numpy(floor_map)).shape
            override_array = np.asarray(_numpy(floor_map_override))
            if override_array.shape != official_shape:
                raise ValueError("floor map override shape does not match official map")
            new_tensor = getattr(floor_map, "new_tensor", None)
            floor_map = (
                new_tensor(override_array)
                if callable(new_tensor)
                else np.array(override_array, copy=True)
            )
        cloned = (
            floor_map.clone()
            if callable(getattr(floor_map, "clone", None))
            else np.array(_numpy(floor_map), copy=True)
        )
        raw_navigation_grid = np.asarray(_numpy(cloned) != 0, dtype=bool)
        if robot_erosion_margin_m is None:
            eroded = np.array(
                _numpy(trav_map._erode_trav_map(cloned, robot=robot)),
                copy=True,
            )
            erosion_telemetry = {
                "erosion_policy": "official_robot_eroded_map",
                "robot_erosion_margin_m": None,
            }
        else:
            import cv2

            resolution_for_erosion = float(getattr(trav_map, "map_resolution"))
            footprint = _select_stable_robot_footprint(robot)
            extent = footprint.extent_xy_m
            if extent is None:
                return OfficialTraversabilityRouteResult(
                    status="no_route",
                    plan=None,
                    private_report={
                        "provider_version": _PROVIDER_VERSION,
                        "floor": int(floor),
                        "reason": "stable_robot_footprint_unavailable",
                        **dict(footprint.private_report),
                    },
                )
            if (
                not math.isfinite(resolution_for_erosion)
                or resolution_for_erosion <= 0.0
            ):
                raise ValueError("robot footprint erosion inputs are invalid")
            radius_m = float(math.hypot(*extent) / 2.0) + float(
                robot_erosion_margin_m
            )
            kernel_pixels = (
                max(1, int(math.ceil(radius_m / resolution_for_erosion)))
                if robot_erosion_kernel_pixels is None
                else int(robot_erosion_kernel_pixels)
            )
            eroded = cv2.erode(
                np.where(_numpy(cloned) != 0, np.uint8(255), np.uint8(0)),
                np.ones((kernel_pixels, kernel_pixels), dtype=np.uint8),
            )
            erosion_telemetry = {
                "erosion_policy": "stable_base_footprint_with_explicit_margin",
                "robot_erosion_margin_m": float(robot_erosion_margin_m),
                "robot_footprint_extent_m": [float(extent[0]), float(extent[1])],
                "robot_erosion_radius_m": radius_m,
                "robot_erosion_kernel_pixels": kernel_pixels,
                "robot_erosion_kernel_override": (
                    None
                    if robot_erosion_kernel_pixels is None
                    else int(robot_erosion_kernel_pixels)
                ),
                **dict(footprint.private_report),
            }
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        return OfficialTraversabilityRouteResult(
            status="no_route",
            plan=None,
            private_report={
                "provider_version": _PROVIDER_VERSION,
                "floor": int(floor),
                "reason": "eroded_floor_map_failed",
                "error_type": type(exc).__name__,
            },
        )
    if eroded.ndim != 2 or eroded.size == 0:
        return _no_route(floor, "invalid_eroded_floor_map")
    eroded_before_loaded_room_mask = np.array(eroded != 0, copy=True)
    resolution = float(getattr(trav_map, "map_resolution", float("nan")))
    map_size = float(getattr(trav_map, "map_size", eroded.shape[0]))
    if (
        not math.isfinite(resolution)
        or resolution <= 0.0
        or not math.isfinite(map_size)
        or map_size <= 0.0
    ):
        return _no_route(floor, "invalid_map_geometry")
    make_tensor = tensor_factory or _torch_tensor
    cell_adapter = _canonical_cell_adapter(
        trav_map=trav_map,
        resolution=resolution,
        map_size=map_size,
        tensor_factory=make_tensor,
        allow_test_fallback=allow_test_fallback,
    )
    if cell_adapter is None:
        return OfficialTraversabilityRouteResult(
            status="unavailable",
            plan=None,
            private_report={
                "provider_version": _PROVIDER_VERSION,
                "floor": int(floor),
                "reason": "official_map_transform_unavailable",
                "cell_transform_source": "unavailable",
            },
        )
    try:
        terminal_structural_cells = structural_terminal_cells_from_aabbs(
            aabbs_xy=structural_terminal_aabbs_xy,
            world_to_cell=cell_adapter.world_to_cell,
            grid_shape=eroded.shape,
        )
    except (TypeError, ValueError) as exc:
        return _no_route(
            floor,
            "invalid_structural_terminal_aabb",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
    room_instance_grid, room_name_to_id = _scene_room_instance_grid(
        scene,
        expected_shape=eroded.shape,
    )
    structural_open_grid = _load_structural_open_grid(
        str(getattr(scene, "scene_dir", "")),
        int(floor),
        int(eroded.shape[0]),
    )
    target_room_ids: tuple[int, ...] = ()
    followup_target_room_ids: tuple[int, ...] = ()
    if room_instance_grid is not None:
        unresolved_target_rooms = tuple(
            room for room in target_rooms if room not in room_name_to_id
        )
        unresolved_followup_rooms = tuple(
            room for room in followup_target_rooms if room not in room_name_to_id
        )
        if unresolved_target_rooms or unresolved_followup_rooms:
            return _no_route(
                floor,
                "official_target_room_names_unresolved",
                {
                    "target_room_names": list(target_rooms),
                    "unresolved_target_room_names": list(unresolved_target_rooms),
                    "followup_target_room_names": list(followup_target_rooms),
                    "unresolved_followup_target_room_names": list(
                        unresolved_followup_rooms
                    ),
                },
            )
        target_room_ids = tuple(
            int(room_name_to_id[name]) for name in target_rooms
        )
        followup_target_room_ids = tuple(
            int(room_name_to_id[name]) for name in followup_target_rooms
        )
    eroded, loaded_room_telemetry = _mask_to_loaded_rooms(
        eroded,
        scene=scene,
        room_instance_grid=room_instance_grid,
        room_name_to_id=room_name_to_id,
        structural_open_grid=structural_open_grid,
    )
    loaded_room_masked_grid = np.array(eroded != 0, copy=True)
    station_geometry_available = bool(
        room_instance_grid is not None and structural_open_grid is not None
    )
    max_snap_cells = max(1, int(math.ceil(snap_limit / resolution)))
    try:
        start_cell_requested = cell_adapter.world_to_cell(start)
        blocked_cells_requested = _validated_blocked_cells(
            additional_blocked_cells,
            shape=eroded.shape,
        )
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        return _no_route(
            floor,
            "invalid_dynamic_blocked_cells",
            {
                "error_type": type(exc).__name__,
                "source_requested_xy": list(start),
                **_eroded_map_telemetry(eroded, cell_adapter),
            },
        )
    try:
        excluded_approaches_requested = _validated_station_approaches(
            excluded_station_approaches,
            shape=eroded.shape,
        )
    except (IndexError, TypeError, ValueError) as exc:
        return _no_route(
            floor,
            "invalid_station_approach_exclusions",
            {
                "error_type": type(exc).__name__,
                "source_requested_xy": list(start),
                **_eroded_map_telemetry(eroded, cell_adapter),
            },
        )
    source_cell_preserved = start_cell_requested in blocked_cells_requested
    blocked_cells_applied = tuple(
        cell
        for cell in blocked_cells_requested
        if cell != start_cell_requested
    )
    for row, col in blocked_cells_applied:
        eroded[row, col] = 0
    final_navigation_grid = np.array(eroded != 0, copy=True)
    map_telemetry = {
        **_eroded_map_telemetry(eroded, cell_adapter),
        **erosion_telemetry,
        **dict(loaded_room_telemetry),
        "interaction_station_geometry_available": station_geometry_available,
        "interaction_station_room_source": (
            "scene.seg_map.room_ins_map"
            if room_instance_grid is not None
            else None
        ),
        "target_room_names_private": list(target_rooms),
        "target_room_ids_private": list(target_room_ids),
        "target_room_binding_source": (
            "official_object_in_rooms"
            if target_room_ids
            else "target_center_pixel_fallback"
        ),
        "target_distance_reference": (
            "aabb_nearest_face"
            if target_bounds is not None
            else "object_origin_fallback"
        ),
        "target_aabb_xy_private": (
            None if target_bounds is None else list(target_bounds)
        ),
        "followup_target_room_names_private": list(followup_target_rooms),
        "followup_target_room_ids_private": list(followup_target_room_ids),
        "interaction_station_structural_map_source": (
            "scene_dir/layout/floor_trav_no_obj_<floor>.png"
            if structural_open_grid is not None
            else None
        ),
        "floor_map_override_applied": floor_map_override is not None,
        "floor_map_override_provenance": (
            None
            if floor_map_override is None
            else str(floor_map_override_provenance or "private_unspecified")
        ),
        "dynamic_overlay_policy": "run_local_curobo_positive_sample_cells",
        "dynamic_overlay_requested_cells": [
            list(cell) for cell in blocked_cells_requested
        ],
        "dynamic_overlay_applied_cells": [
            list(cell) for cell in blocked_cells_applied
        ],
        "dynamic_overlay_cell_count": len(blocked_cells_applied),
        "dynamic_overlay_source_cell_preserved": source_cell_preserved,
        "excluded_station_approaches": [
            list(key) for key in excluded_approaches_requested
        ],
        "excluded_station_approach_count": len(
            excluded_approaches_requested
        ),
    }
    if blocked_cells_applied and official_cell_astar is None:
        return _no_route(
            floor,
            "dynamic_overlay_requires_exact_astar",
            {
                "source_requested_xy": list(start),
                "source_requested_official_cell": list(start_cell_requested),
                **map_telemetry,
            },
        )
    try:
        source_snap_required = not _cell_open(eroded, *start_cell_requested)
        start_cell = (
            _nearest_traversable_cell(
                eroded,
                start_cell_requested,
                maximum_radius_cells=max_snap_cells,
            )
            if source_snap_required
            else start_cell_requested
        )
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        return _no_route(
            floor,
            "cell_transform_error",
            {
                **map_telemetry,
                "error_type": type(exc).__name__,
                "source_requested_xy": list(start),
            },
        )
    if start_cell is None:
        return _no_route(
            floor,
            "source_snap_failed",
            {
                **map_telemetry,
                "source_requested_xy": list(start),
                "source_requested_official_cell": list(start_cell_requested),
            },
        )
    try:
        source = (
            cell_adapter.cell_to_world(start_cell)
            if source_snap_required
            else start
        )
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        return _no_route(
            floor,
            "cell_transform_error",
            {
                **map_telemetry,
                "error_type": type(exc).__name__,
                "source_requested_xy": list(start),
                "source_requested_official_cell": list(start_cell_requested),
            },
        )
    source_snap_m = _distance(start, source)
    if source_snap_m > snap_limit + resolution * math.sqrt(2.0) + 1e-9:
        return _no_route(
            floor,
            "source_snap_exceeds_limit",
            {
                **map_telemetry,
                "source_requested_xy": list(start),
                "source_requested_official_cell": list(start_cell_requested),
                "source_snapped_xy": list(source),
                "source_snapped_official_cell": list(start_cell),
                "source_snap_m": source_snap_m,
            },
        )

    reachable_cells = _reachable_cells_8(eroded, start_cell)
    source_geodesic_distances: np.ndarray | None = None
    station_selection_telemetry: dict[str, Any] = {
        "station_selection_geometry": "legacy_radial_snap",
    }
    raw_candidates: list[
        tuple[float, float, int, tuple[float, float], tuple[int, int]]
    ] = []
    seen_cells: set[tuple[int, int]] = set()

    def expanded_aabb_cells(bounds: Sequence[float]) -> tuple[tuple[int, int], ...]:
        x0, y0, x1, y1 = bounds
        corners = (
            cell_adapter.world_to_cell((x0 - maximum, y0 - maximum)),
            cell_adapter.world_to_cell((x1 + maximum, y1 + maximum)),
        )
        row_min = max(0, min(cell[0] for cell in corners) - 2)
        row_max = min(eroded.shape[0], max(cell[0] for cell in corners) + 3)
        col_min = max(0, min(cell[1] for cell in corners) - 2)
        col_max = min(eroded.shape[1], max(cell[1] for cell in corners) + 3)
        return tuple(
            (row, col)
            for row in range(row_min, row_max)
            for col in range(col_min, col_max)
            if bool(eroded[row, col])
        )

    try:
        if station_geometry_available:
            source_geodesic_distances = compute_source_geodesic_distances(
                np.asarray(eroded != 0, dtype=bool),
                start_cell,
                resolution=resolution,
            )
            reachable_rows, reachable_cols = np.where(
                np.isfinite(source_geodesic_distances)
            )
            reachable_cells = {
                (int(row), int(col))
                for row, col in zip(reachable_rows, reachable_cols)
            }
            component_labels = np.asarray(
                np.isfinite(source_geodesic_distances),
                dtype=np.int32,
            )
            selection = select_interaction_station_cells(
                target_xy=target,
                target_aabb_xy=target_bounds,
                navigation_open_grid=np.asarray(eroded != 0, dtype=bool),
                structural_open_grid=np.asarray(
                    structural_open_grid,
                    dtype=bool,
                ),
                component_labels=component_labels,
                source_component_label=1,
                resolution=resolution,
                map_size=int(eroded.shape[0]),
                maximum_center_distance_m=maximum,
                room_instance_map=room_instance_grid,
                target_room_ids=(target_room_ids or None),
                prefer_same_room=True,
                allow_cross_room_fallback_when_target_room_known=(
                    allow_cross_room_station_when_target_room_known
                ),
                source_geodesic_distance_grid=source_geodesic_distances,
                allowed_blocked_target_cells=frozenset(
                    terminal_structural_cells
                ),
            )
            station_selection_telemetry = {
                "station_selection_geometry": (
                    "same_room_structural_los_source_geodesic"
                ),
                "station_target_official_cell": list(selection.target_cell),
                "station_target_room_id": selection.target_room_id,
                "station_target_center_room_id": (
                    selection.target_center_room_id
                ),
                "station_target_room_ids": list(selection.target_room_ids),
                "station_target_room_source": selection.target_room_source,
                "station_room_match_policy": selection.room_match_policy,
                "station_raw_open_within_radius_count": (
                    selection.raw_open_within_radius_count
                ),
                "station_raw_open_connected_count": (
                    selection.raw_open_connected_count
                ),
                "station_structural_occlusion_rejection_count": (
                    selection.structural_occlusion_rejection_count
                ),
                "station_same_room_candidate_count": (
                    selection.same_room_candidate_count
                ),
                "station_cross_room_candidate_count": (
                    selection.cross_room_candidate_count
                ),
                "station_same_room_preference_applied": (
                    selection.same_room_preference_applied
                ),
                "station_cross_room_fallback_used": (
                    selection.cross_room_fallback_used
                ),
                "station_known_target_room_constraint_applied": (
                    selection.known_target_room_constraint_applied
                ),
                "station_source_geodesic_sort_applied": (
                    selection.source_geodesic_sort_applied
                ),
                "station_structural_terminal_allowed_cell_count": len(
                    terminal_structural_cells
                ),
                "station_cross_room_boundary_opt_in": bool(
                    allow_cross_room_station_when_target_room_known
                ),
                "station_same_room_candidate_cells": [
                    list(cell) for cell in selection.same_room_candidate_cells
                ],
                "station_cross_room_candidate_cells": [
                    {
                        "cell": list(cell),
                        "room_id": (
                            None
                            if room_instance_grid is None
                            else int(room_instance_grid[cell])
                        ),
                    }
                    for cell in selection.cross_room_candidate_cells
                ],
                "station_structural_occlusion_witnesses": [
                    {
                        "station_cell": list(station_cell),
                        "interaction_target_cell": list(target_cell),
                        "first_blocked_cell": list(blocked_cell),
                        "blocked_cell_room_id": (
                            None
                            if room_instance_grid is None
                            or not (
                                0 <= blocked_cell[0] < room_instance_grid.shape[0]
                                and 0 <= blocked_cell[1] < room_instance_grid.shape[1]
                            )
                            else int(room_instance_grid[blocked_cell])
                        ),
                        "blocked_cell_is_interaction_target": (
                            blocked_cell == target_cell
                        ),
                        "blocked_cell_is_allowed_terminal": (
                            blocked_cell in terminal_structural_cells
                        ),
                    }
                    for station_cell, target_cell, blocked_cell
                    in selection.structural_occlusion_witnesses
                ],
            }
            candidate_cells = selection.cells
        elif target_bounds is not None:
            candidate_cells = expanded_aabb_cells(target_bounds)
        else:
            radial_angle = math.atan2(
                start[1] - target[1],
                start[0] - target[0],
            )
            legacy_cells: list[tuple[int, int]] = []
            for radius in _station_radii(desired, maximum):
                for index in range(angular_samples):
                    angle = (
                        radial_angle
                        + 2.0 * math.pi * index / angular_samples
                    )
                    requested = (
                        target[0] + radius * math.cos(angle),
                        target[1] + radius * math.sin(angle),
                    )
                    requested_cell = cell_adapter.world_to_cell(requested)
                    station_cell = _nearest_traversable_cell(
                        eroded,
                        requested_cell,
                        maximum_radius_cells=max_snap_cells,
                    )
                    if station_cell is not None:
                        legacy_cells.append(station_cell)
            candidate_cells = tuple(legacy_cells)
        for index, station_cell in enumerate(candidate_cells):
            if station_cell is None or station_cell in seen_cells:
                continue
            station = cell_adapter.cell_to_world(station_cell)
            interaction_target, _ = nearest_interaction_target_xy(
                station,
                target,
                target_aabb_xy=target_bounds,
            )
            approach_key = _station_approach_key(
                station_cell=station_cell,
                station_xy=station,
                target_xy=interaction_target,
            )
            if approach_key in excluded_approaches_requested:
                continue
            station_distance = _distance(station, interaction_target)
            if (
                station_distance < desired - resolution
                or station_distance > maximum + 1e-9
            ):
                continue
            seen_cells.add(station_cell)
            source_distance = (
                float(source_geodesic_distances[station_cell])
                if source_geodesic_distances is not None
                else _distance(source, station)
            )
            raw_candidates.append(
                (
                    station_distance,
                    source_distance,
                    index,
                    station,
                    station_cell,
                )
            )
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        return _no_route(
            floor,
            "cell_transform_error",
            {
                **map_telemetry,
                "error_type": type(exc).__name__,
                "source_requested_xy": list(start),
                "source_requested_official_cell": list(start_cell_requested),
                **station_selection_telemetry,
            },
        )
    # The first key is source geodesic, not target proximity.  A station just
    # across a wall can be geometrically close to the object yet require a
    # house-scale detour; same-room / structural LOS has already rejected that
    # case when live layout geometry is available.
    raw_candidates.sort(key=lambda row: (row[1], row[0], row[2], row[4]))
    if not raw_candidates:
        return _no_route(
            floor,
            "no_traversable_approach_station",
            {
                **map_telemetry,
                "source_requested_xy": list(start),
                "source_requested_official_cell": list(start_cell_requested),
                "source_snapped_xy": list(source),
                "source_snapped_official_cell": list(start_cell),
                **station_selection_telemetry,
                **_route_frontend_failure_diagnostic(
                    grids={
                        "g0_official_raw": raw_navigation_grid,
                        "g1_robot_eroded": eroded_before_loaded_room_mask,
                        "g2_loaded_room_masked": loaded_room_masked_grid,
                        "g3_dynamic_overlay": final_navigation_grid,
                    },
                    source_requested_cell=start_cell_requested,
                    maximum_source_snap_cells=max_snap_cells,
                    target_xy=target,
                    target_aabb_xy=target_bounds,
                    resolution=resolution,
                    map_size=int(eroded.shape[0]),
                    maximum_station_distance_m=maximum,
                ),
            },
        )

    # OmniGibson's official A* rebuilds a Python g-score entry for every map
    # cell on every call.  Calling it for every pair of manipulation stations
    # made the two-leg Oracle spend minutes in nested A*.  The official search
    # uses 8-connected neighbors, so an inexpensive flood fill on the exact
    # same robot-eroded map is a sound reachability prefilter.  A* is still
    # used to author and validate the actual path, once per selected leg.
    raw_candidates = [
        row for row in raw_candidates if row[4] in reachable_cells
    ]
    if not raw_candidates:
        return _no_route(
            floor,
            "no_connected_approach_station",
            {
                **map_telemetry,
                "source_requested_xy": list(start),
                "source_requested_official_cell": list(start_cell_requested),
                "source_snapped_xy": list(source),
                "source_snapped_official_cell": list(start_cell),
                **station_selection_telemetry,
            },
        )

    followup_connected_station: tuple[tuple[float, float], tuple[int, int]] | None = None
    if followup_target is not None and followup_desired is not None:
        followup_connected = False
        followup_seen: set[tuple[int, int]] = set()
        followup_selection_telemetry: dict[str, Any] = {}
        try:
            if station_geometry_available:
                followup_selection = select_interaction_station_cells(
                    target_xy=followup_target,
                    target_aabb_xy=followup_target_bounds,
                    navigation_open_grid=np.asarray(eroded != 0, dtype=bool),
                    structural_open_grid=np.asarray(
                        structural_open_grid,
                        dtype=bool,
                    ),
                    component_labels=component_labels,
                    source_component_label=1,
                    resolution=resolution,
                    map_size=int(eroded.shape[0]),
                    maximum_center_distance_m=maximum,
                    room_instance_map=room_instance_grid,
                    target_room_ids=(followup_target_room_ids or None),
                    prefer_same_room=True,
                    allow_cross_room_fallback_when_target_room_known=False,
                    source_geodesic_distance_grid=source_geodesic_distances,
                )
                followup_selection_telemetry = {
                    "followup_station_raw_open_within_radius_count": (
                        followup_selection.raw_open_within_radius_count
                    ),
                    "followup_station_raw_open_connected_count": (
                        followup_selection.raw_open_connected_count
                    ),
                    "followup_station_structural_occlusion_rejection_count": (
                        followup_selection.structural_occlusion_rejection_count
                    ),
                    "followup_station_same_room_candidate_count": (
                        followup_selection.same_room_candidate_count
                    ),
                    "followup_station_cross_room_candidate_count": (
                        followup_selection.cross_room_candidate_count
                    ),
                    "followup_station_target_room_ids": list(
                        followup_selection.target_room_ids
                    ),
                    "followup_station_room_match_policy": (
                        followup_selection.room_match_policy
                    ),
                    "followup_station_cross_room_candidate_cells": [
                        {
                            "cell": list(cell),
                            "room_id": (
                                None
                                if room_instance_grid is None
                                else int(room_instance_grid[cell])
                            ),
                        }
                        for cell in followup_selection.cross_room_candidate_cells
                    ],
                    "followup_station_structural_occlusion_witnesses": [
                        {
                            "station_cell": list(station_cell),
                            "interaction_target_cell": list(target_cell),
                            "first_blocked_cell": list(blocked_cell),
                            "blocked_cell_room_id": (
                                None
                                if room_instance_grid is None
                                or not (
                                    0 <= blocked_cell[0] < room_instance_grid.shape[0]
                                    and 0 <= blocked_cell[1] < room_instance_grid.shape[1]
                                )
                                else int(room_instance_grid[blocked_cell])
                            ),
                            "blocked_cell_is_interaction_target": (
                                blocked_cell == target_cell
                            ),
                        }
                        for station_cell, target_cell, blocked_cell
                        in followup_selection.structural_occlusion_witnesses
                    ],
                }
                followup_candidate_cells = followup_selection.cells
            elif followup_target_bounds is not None:
                followup_candidate_cells = expanded_aabb_cells(
                    followup_target_bounds
                )
            else:
                followup_radial_angle = math.atan2(
                    source[1] - followup_target[1],
                    source[0] - followup_target[0],
                )
                fallback_cells: list[tuple[int, int]] = []
                for radius in _station_radii(followup_desired, maximum):
                    for index in range(angular_samples):
                        angle = (
                            followup_radial_angle
                            + 2.0 * math.pi * index / angular_samples
                        )
                        requested_cell = cell_adapter.world_to_cell(
                            (
                                followup_target[0] + radius * math.cos(angle),
                                followup_target[1] + radius * math.sin(angle),
                            )
                        )
                        station_cell = _nearest_traversable_cell(
                            eroded,
                            requested_cell,
                            maximum_radius_cells=max_snap_cells,
                        )
                        if station_cell is not None:
                            fallback_cells.append(station_cell)
                followup_candidate_cells = tuple(fallback_cells)
            for station_cell in followup_candidate_cells:
                if station_cell is None or station_cell in followup_seen:
                    continue
                followup_seen.add(station_cell)
                station = cell_adapter.cell_to_world(station_cell)
                followup_interaction_target, _ = nearest_interaction_target_xy(
                    station,
                    followup_target,
                    target_aabb_xy=followup_target_bounds,
                )
                station_distance = _distance(
                    station,
                    followup_interaction_target,
                )
                if (
                    followup_desired - resolution
                    <= station_distance
                    <= maximum + resolution
                    and station_cell in reachable_cells
                ):
                    followup_connected = True
                    followup_connected_station = (station, station_cell)
                    break
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
            return _no_route(
                floor,
                "cell_transform_error",
                {
                    **map_telemetry,
                    "error_type": type(exc).__name__,
                    "source_requested_xy": list(start),
                    "source_requested_official_cell": list(start_cell_requested),
                    "followup_target_xy": list(followup_target),
                },
            )
        if not followup_connected:
            return OfficialTraversabilityRouteResult(
                status="no_route",
                plan=None,
                private_report={
                    "provider_version": _PROVIDER_VERSION,
                    "floor": int(floor),
                    "reason": "no_initial_station_with_followup_route",
                    "source_requested_xy": list(start),
                    "source_requested_official_cell": list(start_cell_requested),
                    "source_snapped_xy": list(source),
                    "source_snapped_official_cell": list(start_cell),
                    "source_snap_m": source_snap_m,
                    "component_prefilter": "8_connected_eroded_map",
                    "followup_target_xy": list(followup_target),
                    "followup_route_validation": "same_8_connected_component",
                    "followup_station_found": False,
                    "followup_route_attempts": [],
                    **followup_selection_telemetry,
                    **_route_frontend_failure_diagnostic(
                        grids={
                            "g0_official_raw": raw_navigation_grid,
                            "g1_robot_eroded": eroded_before_loaded_room_mask,
                            "g2_loaded_room_masked": loaded_room_masked_grid,
                            "g3_dynamic_overlay": final_navigation_grid,
                        },
                        source_requested_cell=start_cell_requested,
                        maximum_source_snap_cells=max_snap_cells,
                        target_xy=followup_target,
                        target_aabb_xy=followup_target_bounds,
                        resolution=resolution,
                        map_size=int(eroded.shape[0]),
                        maximum_station_distance_m=maximum,
                    ),
                    **map_telemetry,
                },
            )
    attempts: list[dict[str, Any]] = []
    feasible: list[tuple[float, float, int, XYRoutePlan, dict[str, Any]]] = []
    for station_distance, _, station_index, station, station_cell in raw_candidates[
        :maximum_astar_calls
    ]:
        interaction_target, _ = nearest_interaction_target_xy(
            station,
            target,
            target_aabb_xy=target_bounds,
        )
        attempt: dict[str, Any] = {
            "source_requested_xy": list(start),
            "source_requested_official_cell": list(start_cell_requested),
            "source_snapped_xy": list(source),
            "source_snapped_official_cell": list(start_cell),
            "station_index": int(station_index),
            "station_xy": list(station),
            "station_official_cell": list(station_cell),
            "station_distance_m": float(station_distance),
            "station_approach_key": list(
                _station_approach_key(
                    station_cell=station_cell,
                    station_xy=station,
                    target_xy=interaction_target,
                )
            ),
            "interaction_target_xy": list(interaction_target),
        }
        try:
            if official_cell_astar is not None:
                exact_path = official_cell_astar(eroded, start_cell, station_cell)
                sanitized = _sanitize_official_cell_path(
                    path=exact_path,
                    source_cell=start_cell,
                    target_cell=station_cell,
                    source_snapped=source,
                    target_snapped=station,
                    source_snap_required=source_snap_required,
                    eroded=eroded,
                    cell_adapter=cell_adapter,
                )
                eight_connected_rejection = None
                if (
                    sanitized.status
                    == "official_path_local_validation_mismatch"
                    and sanitized.private_report.get("reason")
                    == "diagonal_corner_has_no_open_cardinal_decomposition"
                    and official_four_connected_astar is not None
                ):
                    # OmniGibson's official 8-connected A* permits corner
                    # cutting and returns only one shortest path.  Rejecting
                    # that one path does not prove that the station lacks a
                    # path executable as cardinal public actions.  Reuse the
                    # same official A* implementation in its 4-connected mode
                    # and validate its result with the unchanged sanitizer.
                    # This is a strict rescue on the same eroded map, not a
                    # second planner or a diagonal safety relaxation.
                    eight_connected_rejection = dict(
                        sanitized.private_report
                    )
                    cardinal_path = official_four_connected_astar(
                        eroded,
                        start_cell,
                        station_cell,
                    )
                    sanitized = _sanitize_official_cell_path(
                        path=cardinal_path,
                        source_cell=start_cell,
                        target_cell=station_cell,
                        source_snapped=source,
                        target_snapped=station,
                        source_snap_required=source_snap_required,
                        eroded=eroded,
                        cell_adapter=cell_adapter,
                    )
                exact_cells = sanitized.private_report.get(
                    "official_cell_path",
                    [],
                )
                geodesic_value = (
                    sum(
                        math.hypot(
                            float(right[0]) - float(left[0]),
                            float(right[1]) - float(left[1]),
                        )
                        * resolution
                        for left, right in zip(exact_cells, exact_cells[1:])
                    )
                    if sanitized.status == "valid"
                    else None
                )
                if eight_connected_rejection is not None:
                    attempt.update(
                        four_connected_rescue_attempted=True,
                        eight_connected_rejection=eight_connected_rejection,
                        four_connected_rescue_status=sanitized.status,
                    )
            else:
                path, geodesic = shortest_path(
                    floor=floor,
                    source_world=make_tensor(source),
                    target_world=make_tensor(station),
                    entire_path=True,
                    robot=robot,
                )
                geodesic_value = _scalar(geodesic)
                sanitized = _sanitize_official_path(
                    path=path,
                    source_requested=start,
                    source_snapped=source,
                    target_snapped=station,
                    source_snap_required=source_snap_required,
                    eroded=eroded,
                    resolution=resolution,
                    cell_adapter=cell_adapter,
                )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            attempt.update(
                status="official_astar_error",
                official_path_exists=None,
                error_type=type(exc).__name__,
            )
            attempts.append(attempt)
            continue
        if sanitized.status != "valid":
            attempt.update(
                status=sanitized.status,
                official_path_exists=(
                    None
                    if sanitized.status == "cell_transform_error"
                    else sanitized.status != "official_astar_returned_none"
                ),
                **dict(sanitized.private_report),
            )
            attempts.append(attempt)
            continue
        if geodesic_value is None:
            attempt.update(
                status="official_geodesic_nonfinite",
                official_path_exists=True,
            )
            attempts.append(attempt)
            continue
        if sanitized.waypoints is None:
            raise RuntimeError("valid official path has no waypoints")
        waypoints = sanitized.waypoints
        source_snap_required = sanitized.source_snap_required
        if not waypoints:
            attempt.update(status="official_path_empty", official_path_exists=False)
            attempts.append(attempt)
            continue
        path_length = _polyline_length(start, waypoints)
        if path_length <= 1e-9:
            attempt.update(status="zero_length_path", official_path_exists=True)
            attempts.append(attempt)
            continue
        plan = XYRoutePlan(
            goal_xy=station,
            waypoints=waypoints,
            target_distance_m=station_distance,
            path_length_m=path_length,
            start_escape_required=source_snap_required,
        )
        attempt.update(
            status="planned",
            geodesic_m=geodesic_value,
            official_path_exists=True,
            **dict(sanitized.private_report),
        )
        attempts.append(attempt)
        feasible.append(
            (station_distance, geodesic_value, station_index, plan, attempt)
        )
        # Without a follow-up, every remaining candidate is in the same
        # 8-connected component and one official path is sufficient.  With a
        # follow-up, retain the bounded candidate set: the official Torch
        # world-to-map conversion can put a decimal boundary coordinate in an
        # adjacent cell, so each candidate still needs final official
        # validation until one two-leg pair succeeds.
        if followup_target is None:
            break
    if not feasible:
        return OfficialTraversabilityRouteResult(
            status="no_route",
            plan=None,
            private_report={
                "provider_version": _PROVIDER_VERSION,
                "floor": int(floor),
                "reason": "official_astar_no_valid_station",
                "source_requested_xy": list(start),
                "source_requested_official_cell": list(start_cell_requested),
                "source_snapped_xy": list(source),
                "source_snapped_official_cell": list(start_cell),
                "source_snap_m": source_snap_m,
                "astar_calls": len(attempts),
                "station_attempts": attempts,
                **map_telemetry,
            },
        )
    # A follow-up target only constrains destination selection to the same
    # exact 8-connected eroded-map component.  Do not recursively call A* for
    # every initial station: every station retained above is in the source
    # component, and the follow-up prefilter has already found a valid station
    # in that component.  The actual second leg will receive one fresh A* from
    # its measured source after the first leg executes.
    chosen_row = min(
        feasible,
        key=lambda row: (row[1], row[0], row[2], row[3].goal_xy),
    )
    if chosen_row is None:
        raise RuntimeError("official route selection produced no candidate")
    _, geodesic_value, station_index, plan, chosen_attempt = chosen_row
    sparse_api_attestation: dict[str, Any] | None = None
    if official_cell_astar is not None and robot_erosion_margin_m is None:
        try:
            sparse_path, sparse_geodesic = shortest_path(
                floor=floor,
                source_world=make_tensor(source),
                target_world=make_tensor(plan.goal_xy),
                entire_path=True,
                robot=robot,
            )
            sparse_sanitized = _sanitize_official_path(
                path=sparse_path,
                source_requested=start,
                source_snapped=source,
                target_snapped=plan.goal_xy,
                source_snap_required=source_snap_required,
                eroded=eroded,
                resolution=resolution,
                cell_adapter=cell_adapter,
            )
            sparse_api_attestation = {
                "status": sparse_sanitized.status,
                "geodesic_m": _scalar(sparse_geodesic),
                **dict(sparse_sanitized.private_report),
            }
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            sparse_api_attestation = {
                "status": "official_astar_error",
                "error_type": type(exc).__name__,
            }
    elif official_cell_astar is not None:
        # scene.get_shortest_path(robot=...) always rebuilds the grid with
        # OmniGibson's fixed +0.20 m clearance.  Comparing that result against
        # a route authored with an explicit research margin is comparing two
        # different graphs; historically this produced a misleading
        # `official_astar_returned_none` even when the selected relaxed route
        # was valid.  Keep the diagnostic explicit and do not run the extra A*.
        sparse_api_attestation = {
            "status": "not_comparable_different_erosion_policy",
            "route_robot_erosion_margin_m": float(robot_erosion_margin_m),
            "official_helper_extra_margin_m": 0.20,
        }
    private_report: dict[str, Any] = {
        "provider_version": _PROVIDER_VERSION,
        "floor": int(floor),
        "map_with_objects": bool(
            getattr(trav_map, "trav_map_with_objects", False)
        ),
        "source_requested_xy": list(start),
        "source_requested_official_cell": list(start_cell_requested),
        "source_snapped_xy": list(source),
        "source_snapped_official_cell": list(start_cell),
        "source_snap_m": source_snap_m,
        "source_snap_required": source_snap_required,
        "station_index": int(station_index),
        "station_xy": list(plan.goal_xy),
        "station_distance_m": float(plan.target_distance_m),
        "station_approach_key": list(
            _station_approach_key(
                station_cell=tuple(
                    int(value)
                    for value in chosen_attempt["station_official_cell"]
                ),
                station_xy=(float(plan.goal_xy[0]), float(plan.goal_xy[1])),
                target_xy=nearest_interaction_target_xy(
                    plan.goal_xy,
                    target,
                    target_aabb_xy=target_bounds,
                )[0],
            )
        ),
        "geodesic_m": float(geodesic_value),
        "astar_calls": len(attempts),
        "component_prefilter": (
            "no_corner_source_geodesic_grid"
            if station_geometry_available
            else "8_connected_eroded_map"
        ),
        "station_attempts": attempts,
        "official_waypoint_cells": chosen_attempt.get(
            "official_waypoint_cells",
            [],
        ),
        "official_waypoint_cell_count": chosen_attempt.get(
            "official_waypoint_cell_count",
            0,
        ),
        "official_cell_path": chosen_attempt.get("official_cell_path", []),
        "official_cell_path_length": chosen_attempt.get(
            "official_cell_path_length",
            0,
        ),
        "official_cell_path_kind": chosen_attempt.get(
            "official_cell_path_kind",
            "official_sparse_waypoint_cells",
        ),
        "executable_cell_path": chosen_attempt.get(
            "executable_cell_path",
            chosen_attempt.get("official_cell_path", []),
        ),
        "executable_cell_path_length": chosen_attempt.get(
            "executable_cell_path_length",
            chosen_attempt.get("official_cell_path_length", 0),
        ),
        "executable_cell_path_kind": chosen_attempt.get(
            "executable_cell_path_kind",
            "official_sparse_waypoint_cells",
        ),
        "station_selection_policy": (
            "same_room_structural_los_then_source_geodesic"
            if station_geometry_available
            else "legacy_nearest_object_within_local_reach"
        ),
        "sparse_public_api_attestation": sparse_api_attestation,
        "exact_cell_astar_provider": (
            "omnigibson.utils.motion_planning_utils.astar"
            if official_cell_astar is not None
            else None
        ),
        **station_selection_telemetry,
        **map_telemetry,
    }
    if followup_target is not None:
        if followup_connected_station is None:
            raise RuntimeError("follow-up component validation lost its station")
        followup_station, followup_cell = followup_connected_station
        private_report.update(
            {
                "followup_route_required": True,
                "followup_target_xy": list(followup_target),
                "followup_route_validation": "same_8_connected_component",
                "followup_station_xy": list(followup_station),
                "followup_station_official_cell": list(followup_cell),
                "followup_route_attempts": [],
                "chosen_followup_route_private": {
                    "status": "component_prevalidated",
                    "station_xy": list(followup_station),
                    "station_official_cell": list(followup_cell),
                    "astar_deferred_until_measured_second_leg": True,
                },
            }
        )
    return OfficialTraversabilityRouteResult(
        status="planned",
        plan=plan,
        private_report=private_report,
        reachable_official_cells=frozenset(reachable_cells),
        station_candidates=tuple(
            (station_cell, station)
            for _, _, _, station, station_cell in raw_candidates
        ),
    )


def _sanitize_official_path(
    *,
    path: Any,
    source_requested: tuple[float, float],
    source_snapped: tuple[float, float],
    target_snapped: tuple[float, float],
    source_snap_required: bool,
    eroded: np.ndarray,
    resolution: float,
    cell_adapter: _CanonicalMapCellAdapter,
) -> _OfficialPathSanitizationResult:
    if path is None:
        return _OfficialPathSanitizationResult(
            status="official_astar_returned_none",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={},
        )
    try:
        values = _numpy(path)
    except (RuntimeError, TypeError, ValueError):
        return _OfficialPathSanitizationResult(
            status="official_path_shape_invalid",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={"official_path_shape": None},
        )
    if values.ndim == 2 and values.shape[0] == 0:
        return _OfficialPathSanitizationResult(
            status="official_path_empty",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={"official_path_shape": list(values.shape)},
        )
    if values.ndim != 2 or values.shape[1] < 2:
        return _OfficialPathSanitizationResult(
            status="official_path_shape_invalid",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={"official_path_shape": list(values.shape)},
        )
    try:
        values = values[:, :2].astype(float, copy=False)
    except (TypeError, ValueError):
        return _OfficialPathSanitizationResult(
            status="official_path_shape_invalid",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={"official_path_shape": list(values.shape)},
        )
    if not np.isfinite(values).all():
        return _OfficialPathSanitizationResult(
            status="official_path_nonfinite",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={},
        )
    points: list[tuple[float, float]] = []
    if source_snap_required:
        points.append(source_snapped)
    for row in values:
        point = (float(row[0]), float(row[1]))
        if (
            _distance(point, source_snapped)
            <= max(0.03, resolution * math.sqrt(2.0))
            and not points
        ):
            continue
        if not points or _distance(point, points[-1]) > 0.03:
            points.append(point)
    if not points or _distance(points[-1], target_snapped) > 0.03:
        points.append(target_snapped)
    else:
        points[-1] = target_snapped
    # `get_shortest_path(..., entire_path=True)` still returns a waypoint-
    # interval subsample of the official A* cell path.  Reconstructing a
    # straight continuous segment between those sparse points can cut across
    # a corner that the real A* path correctly went around.  Therefore this
    # adapter records the official waypoint cells but does not second-guess
    # global connectivity.  The bounded local swept guard independently
    # validates every configured public execution step.
    official_waypoint_cells: list[tuple[int, int]] = []
    try:
        for point in points:
            cell = cell_adapter.world_to_cell(point)
            if not official_waypoint_cells or official_waypoint_cells[-1] != cell:
                official_waypoint_cells.append(cell)
        source_cell = cell_adapter.world_to_cell(source_snapped)
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        return _OfficialPathSanitizationResult(
            status="cell_transform_error",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={"error_type": type(exc).__name__},
        )
    reachable_cells = _reachable_cells_8(eroded, source_cell)
    invalid_cells = [
        cell for cell in official_waypoint_cells if cell not in reachable_cells
    ]
    if invalid_cells:
        return _OfficialPathSanitizationResult(
            status="official_path_local_validation_mismatch",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={
                "official_cell_path": [
                    list(cell) for cell in official_waypoint_cells
                ],
                "official_cell_path_length": len(official_waypoint_cells),
                "official_cell_path_kind": "official_sparse_waypoint_cells",
                "invalid_official_cells": [
                    list(cell) for cell in invalid_cells
                ],
                "local_validation": "waypoint_cells_in_source_component",
            },
        )
    return _OfficialPathSanitizationResult(
        status="valid",
        waypoints=tuple(points),
        source_snap_required=source_snap_required,
        private_report={
            "official_waypoint_cells": [
                list(cell) for cell in official_waypoint_cells
            ],
            "official_waypoint_cell_count": len(official_waypoint_cells),
            # Canonical artifact-schema names.  The official API returns a
            # waypoint-interval subsample, not every raw A* cell; preserve
            # that qualification instead of claiming denser evidence.
            "official_cell_path": [
                list(cell) for cell in official_waypoint_cells
            ],
            "official_cell_path_length": len(official_waypoint_cells),
            "official_cell_path_kind": "official_sparse_waypoint_cells",
        },
    )


def _sanitize_official_cell_path(
    *,
    path: Any,
    source_cell: tuple[int, int],
    target_cell: tuple[int, int],
    source_snapped: tuple[float, float],
    target_snapped: tuple[float, float],
    source_snap_required: bool,
    eroded: np.ndarray,
    cell_adapter: _CanonicalMapCellAdapter,
) -> _OfficialPathSanitizationResult:
    """Validate the complete cell sequence returned by official ``astar``.

    ``TraversableMap.get_shortest_path(..., entire_path=True)`` still applies
    ``waypoint_interval`` after running A*.  Those sparse points are useful as
    an API attestation, but joining them with straight world-space segments
    can cut a corner.  The caller therefore supplies the exact underlying
    OmniGibson ``motion_planning_utils.astar`` callable for live execution.
    This helper validates that exact result and converts every cell transition
    into a canonical official-map waypoint.
    """

    if path is None:
        return _OfficialPathSanitizationResult(
            status="official_astar_returned_none",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={"official_cell_path_kind": "official_exact_astar_cells"},
        )
    try:
        values = _numpy(path)
    except (RuntimeError, TypeError, ValueError):
        values = np.asarray([], dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        return _OfficialPathSanitizationResult(
            status=(
                "official_path_empty"
                if values.ndim == 2 and values.shape[0] == 0
                else "official_path_shape_invalid"
            ),
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={
                "official_path_shape": list(values.shape),
                "official_cell_path_kind": "official_exact_astar_cells",
            },
        )
    try:
        numeric = values[:, :2].astype(float, copy=False)
    except (TypeError, ValueError):
        numeric = np.asarray([], dtype=float)
    if numeric.ndim != 2 or numeric.shape[1:] != (2,) or not np.isfinite(numeric).all():
        return _OfficialPathSanitizationResult(
            status="official_path_nonfinite",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={"official_cell_path_kind": "official_exact_astar_cells"},
        )
    rounded = np.rint(numeric)
    if not np.allclose(numeric, rounded, rtol=0.0, atol=1e-6):
        return _OfficialPathSanitizationResult(
            status="official_path_shape_invalid",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={
                "reason": "official_cell_path_contains_non_integer_cell",
                "official_cell_path_kind": "official_exact_astar_cells",
            },
        )
    cells = [tuple(int(value) for value in row) for row in rounded]
    invalid_reason: str | None = None
    if cells[0] != source_cell:
        invalid_reason = "official_cell_path_source_mismatch"
    elif cells[-1] != target_cell:
        invalid_reason = "official_cell_path_target_mismatch"
    else:
        for index, cell in enumerate(cells):
            if not _cell_open(eroded, *cell):
                invalid_reason = "official_cell_path_contains_blocked_cell"
                break
            if index > 0:
                previous = cells[index - 1]
                row_step = abs(cell[0] - previous[0])
                col_step = abs(cell[1] - previous[1])
                if max(row_step, col_step) != 1:
                    invalid_reason = "official_cell_path_not_8_connected"
                    break
    if invalid_reason is not None:
        return _OfficialPathSanitizationResult(
            status="official_path_local_validation_mismatch",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={
                "reason": invalid_reason,
                "official_cell_path": [list(cell) for cell in cells],
                "official_cell_path_length": len(cells),
                "official_cell_path_kind": "official_exact_astar_cells",
            },
        )
    for previous, cell in zip(cells, cells[1:]):
        row_step = cell[0] - previous[0]
        col_step = cell[1] - previous[1]
        if abs(row_step) == 1 and abs(col_step) == 1:
            # OmniGibson's point-grid A* permits a diagonal whenever only the
            # destination cell is open.  The swept footprint may use that edge
            # only when the complete supercover is open; otherwise retry with
            # the existing official four-connected rescue.
            side_cells = (
                (cell[0], previous[1]),
                (previous[0], cell[1]),
            )
            if not all(_cell_open(eroded, *candidate) for candidate in side_cells):
                return _OfficialPathSanitizationResult(
                    status="official_path_local_validation_mismatch",
                    waypoints=None,
                    source_snap_required=source_snap_required,
                    private_report={
                        "reason": (
                            "diagonal_corner_has_no_open_cardinal_decomposition"
                        ),
                        "official_cell_path": [list(item) for item in cells],
                        "official_cell_path_length": len(cells),
                        "official_cell_path_kind": "official_exact_astar_cells",
                        "blocked_diagonal_edge": [
                            list(previous),
                            list(cell),
                        ],
                    },
                )
    try:
        executable_cells = list(
            greedy_farthest_visible_path_simplification(
                cells,
                np.asarray(eroded != 0, dtype=bool),
            )
        )
    except ValueError as exc:
        return _OfficialPathSanitizationResult(
            status="official_path_local_validation_mismatch",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={
                "reason": "safe_path_simplification_failed",
                "error": str(exc),
                "official_cell_path": [list(item) for item in cells],
                "official_cell_path_length": len(cells),
                "official_cell_path_kind": "official_exact_astar_cells",
            },
        )
    try:
        cell_world = [
            cell_adapter.cell_to_world(cell) for cell in executable_cells
        ]
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        return _OfficialPathSanitizationResult(
            status="cell_transform_error",
            waypoints=None,
            source_snap_required=source_snap_required,
            private_report={
                "error_type": type(exc).__name__,
                "official_cell_path_kind": "official_exact_astar_cells",
            },
        )
    points: list[tuple[float, float]] = []
    if source_snap_required:
        points.append(source_snapped)
    for point in cell_world[1:]:
        if not points or _distance(points[-1], point) > 1e-9:
            points.append(point)
    if not points or _distance(points[-1], target_snapped) > 1e-9:
        points.append(target_snapped)
    else:
        points[-1] = target_snapped
    waypoint_cells = (
        executable_cells[1:]
        if len(executable_cells) > 1
        else [executable_cells[-1]]
    )
    return _OfficialPathSanitizationResult(
        status="valid",
        waypoints=tuple(points),
        source_snap_required=source_snap_required,
        private_report={
            "official_waypoint_cells": [list(cell) for cell in waypoint_cells],
            "official_waypoint_cell_count": len(waypoint_cells),
            "official_cell_path": [list(cell) for cell in cells],
            "official_cell_path_length": len(cells),
            "official_cell_path_kind": "official_exact_astar_cells",
            "executable_cell_path": [
                list(cell) for cell in executable_cells
            ],
            "executable_cell_path_length": len(executable_cells),
            "executable_cell_path_kind": (
                "supercover_los_simplified_official_exact_astar_cells"
            ),
            "local_validation": (
                "exact_open_no_corner_cut_then_supercover_los_simplified"
            ),
        },
    )


def _nearest_traversable_cell(
    grid: np.ndarray,
    requested: tuple[int, int],
    *,
    maximum_radius_cells: int,
) -> tuple[int, int] | None:
    row, col = requested
    candidates: list[tuple[int, int, int]] = []
    for dr in range(-maximum_radius_cells, maximum_radius_cells + 1):
        for dc in range(-maximum_radius_cells, maximum_radius_cells + 1):
            if dr * dr + dc * dc > maximum_radius_cells * maximum_radius_cells:
                continue
            candidate = (row + dr, col + dc)
            if _cell_open(grid, *candidate):
                candidates.append((dr * dr + dc * dc, candidate[0], candidate[1]))
    if not candidates:
        return None
    _, nearest_row, nearest_col = min(candidates)
    return nearest_row, nearest_col


def _reachable_cells_8(
    grid: np.ndarray,
    source: tuple[int, int],
) -> set[tuple[int, int]]:
    """Return the official A* 8-connected component containing ``source``."""

    if not _cell_open(grid, *source):
        return set()
    reachable = {source}
    frontier: deque[tuple[int, int]] = deque([source])
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    while frontier:
        row, col = frontier.popleft()
        for dr, dc in offsets:
            candidate = (row + dr, col + dc)
            if candidate in reachable or not _cell_open(grid, *candidate):
                continue
            reachable.add(candidate)
            frontier.append(candidate)
    return reachable


def _cell_open(grid: np.ndarray, row: int, col: int) -> bool:
    return (
        0 <= row < grid.shape[0]
        and 0 <= col < grid.shape[1]
        and int(grid[row, col]) != 0
    )


def _world_to_cell_fallback(
    point: tuple[float, float],
    resolution: float,
    map_size: float,
) -> tuple[int, int]:
    return (
        int(point[1] / resolution + map_size / 2.0),
        int(point[0] / resolution + map_size / 2.0),
    )


def _cell_to_world_fallback(
    cell: tuple[int, int],
    resolution: float,
    map_size: float,
) -> tuple[float, float]:
    return (
        (float(cell[1]) - map_size / 2.0) * resolution,
        (float(cell[0]) - map_size / 2.0) * resolution,
    )


def _canonical_cell_adapter(
    *,
    trav_map: Any,
    resolution: float,
    map_size: float,
    tensor_factory: Callable[[Sequence[float]], Any],
    allow_test_fallback: bool,
) -> _CanonicalMapCellAdapter | None:
    world_to_map = getattr(trav_map, "world_to_map", None)
    map_to_world = getattr(trav_map, "map_to_world", None)
    # A half-official adapter is more dangerous than an explicit fallback:
    # station generation and terminal checks would otherwise use different
    # coordinate contracts.  Live OmniGibson provides both methods.
    if not callable(world_to_map) or not callable(map_to_world):
        if not allow_test_fallback:
            return None
        world_to_map = None
        map_to_world = None
    return _CanonicalMapCellAdapter(
        resolution=resolution,
        map_size=map_size,
        world_to_map=world_to_map,
        map_to_world=map_to_world,
        tensor_factory=tensor_factory,
    )


def _flat_pair(value: Any, name: str) -> tuple[float, float]:
    values = _numpy(value).reshape(-1)
    if values.size != 2:
        raise ValueError(f"{name} must return exactly two values")
    result = (float(values[0]), float(values[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} returned non-finite values")
    return result


def _eroded_map_telemetry(
    eroded: np.ndarray,
    cell_adapter: _CanonicalMapCellAdapter,
) -> dict[str, Any]:
    values = np.unique(eroded)
    return {
        "cell_transform_source": cell_adapter.transform_source,
        "cell_open_semantics": "nonzero_matches_official_astar",
        "eroded_map_distinct_value_count": int(values.size),
        "eroded_map_distinct_values": [
            int(value) if np.issubdtype(values.dtype, np.integer) else float(value)
            for value in values[:256]
        ],
        "eroded_map_distinct_values_truncated": bool(values.size > 256),
    }


def _packed_binary_grid(grid: np.ndarray) -> dict[str, Any]:
    values = np.asarray(grid, dtype=bool)
    packed = np.packbits(values.reshape(-1), bitorder="little")
    return {
        "shape": [int(value) for value in values.shape],
        "encoding": "numpy.packbits-little-base64",
        "open_cell_count": int(np.count_nonzero(values)),
        "data": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _route_frontend_failure_diagnostic(
    *,
    grids: Mapping[str, np.ndarray],
    source_requested_cell: tuple[int, int],
    maximum_source_snap_cells: int,
    target_xy: tuple[float, float],
    target_aabb_xy: tuple[float, float, float, float] | None,
    resolution: float,
    map_size: int,
    maximum_station_distance_m: float,
) -> dict[str, Any]:
    """Explain a zero-station result without changing route decisions."""

    stages: dict[str, Any] = {}
    for name, raw_grid in grids.items():
        grid = np.asarray(raw_grid, dtype=bool)
        source = (
            source_requested_cell
            if _cell_open(grid, *source_requested_cell)
            else _nearest_traversable_cell(
                grid,
                source_requested_cell,
                maximum_radius_cells=maximum_source_snap_cells,
            )
        )
        if source is None:
            stages[name] = {
                "source_requested_open": False,
                "source_snapped_cell": None,
                "reachable_cell_count": 0,
                "target_near_open_cell_count": 0,
                "target_near_connected_cell_count": 0,
            }
            continue
        reachable = _reachable_cells_8(grid, source)
        labels = np.zeros(grid.shape, dtype=np.int8)
        if reachable:
            rows, cols = zip(*reachable)
            labels[rows, cols] = 1
        selection = select_interaction_station_cells(
            target_xy=target_xy,
            target_aabb_xy=target_aabb_xy,
            navigation_open_grid=grid,
            structural_open_grid=np.ones(grid.shape, dtype=bool),
            component_labels=labels,
            source_component_label=1,
            resolution=resolution,
            map_size=map_size,
            maximum_center_distance_m=maximum_station_distance_m,
        )
        stages[name] = {
            "source_requested_open": bool(
                _cell_open(grid, *source_requested_cell)
            ),
            "source_snapped_cell": list(source),
            "reachable_cell_count": len(reachable),
            "target_near_open_cell_count": (
                selection.raw_open_within_radius_count
            ),
            "target_near_connected_cell_count": (
                selection.raw_open_connected_count
            ),
        }
    result: dict[str, Any] = {"route_frontend_map_stages": stages}
    if os.environ.get("ACTION20_ROUTEFRONT_MAP_BUNDLE") == "1":
        result["route_frontend_packed_grids"] = {
            name: _packed_binary_grid(grid) for name, grid in grids.items()
        }
    return result


def _station_radii(desired: float, maximum: float) -> tuple[float, ...]:
    values = [desired]
    for fraction in (0.25, 0.50, 0.75, 1.0):
        value = desired + fraction * (maximum - desired)
        if all(abs(value - old) > 1e-6 for old in values):
            values.append(value)
    return tuple(values)


def _polyline_length(
    start: tuple[float, float],
    points: Sequence[tuple[float, float]],
) -> float:
    total = 0.0
    previous = start
    for point in points:
        total += _distance(previous, point)
        previous = point
    return total


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _validated_blocked_cells(
    cells: Sequence[Sequence[int]],
    *,
    shape: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    """Validate and deterministically deduplicate a private map overlay."""

    if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence):
        raise TypeError("additional_blocked_cells must be a sequence")
    if len(shape) != 2:
        raise ValueError("blocked-cell overlay requires a 2D map")
    normalized: set[tuple[int, int]] = set()
    for cell in cells:
        if (
            isinstance(cell, (str, bytes))
            or not isinstance(cell, Sequence)
            or len(cell) != 2
        ):
            raise TypeError("each blocked cell must contain exactly two integers")
        row_value, col_value = cell
        if (
            isinstance(row_value, bool)
            or isinstance(col_value, bool)
            or not isinstance(row_value, Integral)
            or not isinstance(col_value, Integral)
        ):
            raise TypeError("blocked-cell coordinates must be integers")
        row, col = int(row_value), int(col_value)
        if row < 0 or col < 0 or row >= shape[0] or col >= shape[1]:
            raise ValueError("blocked-cell coordinate is outside the eroded map")
        normalized.add((row, col))
    return tuple(sorted(normalized))


def _validated_station_approaches(
    approaches: Sequence[Sequence[int]],
    *,
    shape: tuple[int, ...],
) -> frozenset[tuple[int, int, int]]:
    """Validate failed terminal ``(row, col, yaw_bin)`` approaches."""

    if isinstance(approaches, (str, bytes)) or not isinstance(
        approaches, Sequence
    ):
        raise TypeError("excluded_station_approaches must be a sequence")
    if len(shape) != 2:
        raise ValueError("station-approach exclusions require a 2D map")
    normalized: set[tuple[int, int, int]] = set()
    for approach in approaches:
        if (
            isinstance(approach, (str, bytes))
            or not isinstance(approach, Sequence)
            or len(approach) != 3
        ):
            raise TypeError(
                "each excluded station approach must contain row, col, yaw_bin"
            )
        row_value, col_value, yaw_value = approach
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (row_value, col_value, yaw_value)
        ):
            raise TypeError("station-approach values must be integers")
        row, col, yaw_bin = (
            int(row_value),
            int(col_value),
            int(yaw_value),
        )
        if row < 0 or col < 0 or row >= shape[0] or col >= shape[1]:
            raise ValueError("station-approach cell is outside the eroded map")
        if yaw_bin < 0 or yaw_bin >= _STATION_HEADING_BIN_COUNT:
            raise ValueError("station-approach yaw bin is outside [0, 24)")
        normalized.add((row, col, yaw_bin))
    return frozenset(normalized)


def _xy(value: Sequence[float], name: str) -> tuple[float, float]:
    try:
        result = (float(value[0]), float(value[1]))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain two finite numbers")
    return result


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _numpy(value: Any) -> np.ndarray:
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "numpy", None)):
        value = value.numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if callable(getattr(value, "item", None)):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _torch_tensor(value: Sequence[float]) -> Any:
    import torch

    return torch.tensor(value, dtype=torch.float32)


def _no_route(
    floor: int,
    reason: str,
    private_report: Mapping[str, Any] | None = None,
) -> OfficialTraversabilityRouteResult:
    return OfficialTraversabilityRouteResult(
        status="no_route",
        plan=None,
        private_report={
            "provider_version": _PROVIDER_VERSION,
            "floor": int(floor),
            "reason": reason,
            **({} if private_report is None else dict(private_report)),
        },
    )
