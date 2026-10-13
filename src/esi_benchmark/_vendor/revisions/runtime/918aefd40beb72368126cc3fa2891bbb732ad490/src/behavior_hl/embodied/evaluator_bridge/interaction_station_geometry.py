"""Task-agnostic 2-D interaction-station geometry.

The functions in this module are deliberately simulator-free.  They operate on
the official BEHAVIOR floor maps after the caller has applied the same resize,
threshold, and robot-footprint erosion as ``TraversableMap``.

Two maps have different jobs:

* ``navigation_open_grid`` includes scene objects and defines legal base cells.
* ``structural_open_grid`` comes from ``floor_trav_no_obj_<floor>.png`` and is
  used for station-to-target line-of-sight.  It rejects walls without treating
  the target's support table as an opaque wall.

No task names, object categories, room names, or fixed coordinates belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

import numpy as np


Cell = tuple[int, int]
AabbXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class InteractionStationSelection:
    """Filtered interaction stations and auditable rejection statistics."""

    cells: tuple[Cell, ...]
    target_cell: Cell
    target_room_id: int | None
    raw_open_within_radius_count: int
    raw_open_connected_count: int
    structural_occlusion_rejection_count: int
    same_room_candidate_count: int
    cross_room_candidate_count: int
    same_room_preference_applied: bool
    cross_room_fallback_used: bool
    known_target_room_constraint_applied: bool = False
    source_geodesic_sort_applied: bool = False
    source_geodesic_unreachable_rejection_count: int = 0
    target_center_room_id: int | None = None
    target_room_ids: tuple[int, ...] = ()
    target_room_source: str = "target_center_pixel"
    room_match_policy: str = "target_room_ids"
    same_room_candidate_cells: tuple[Cell, ...] = ()
    cross_room_candidate_cells: tuple[Cell, ...] = ()
    structural_occlusion_witnesses: tuple[tuple[Cell, Cell, Cell], ...] = ()


def world_to_cell(
    xy: Sequence[float], *, resolution: float, map_size: int
) -> Cell:
    """Match official ``BaseMap.world_to_map`` for one world-space XY point."""

    return (
        int(float(xy[1]) / resolution + map_size / 2.0),
        int(float(xy[0]) / resolution + map_size / 2.0),
    )


def cell_to_world(
    cell: Sequence[int], *, resolution: float, map_size: int
) -> tuple[float, float]:
    """Match official ``BaseMap.map_to_world`` for one map cell."""

    return (
        (int(cell[1]) - map_size / 2.0) * resolution,
        (int(cell[0]) - map_size / 2.0) * resolution,
    )


def _finite_xy(value: Sequence[float], name: str) -> tuple[float, float]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite numbers") from exc
    if len(result) != 2 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain two finite numbers")
    return result


def normalize_target_aabb_xy(
    value: Sequence[float] | None,
) -> AabbXY | None:
    """Return finite ordered XY bounds, or ``None`` for origin fallback."""

    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        bounds = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if (
        len(bounds) != 4
        or not all(math.isfinite(item) for item in bounds)
        or bounds[0] > bounds[2]
        or bounds[1] > bounds[3]
    ):
        return None
    return bounds


def nearest_interaction_target_xy(
    observer_xy: Sequence[float],
    target_xy: Sequence[float],
    *,
    target_aabb_xy: Sequence[float] | None = None,
) -> tuple[tuple[float, float], str]:
    """Resolve the target point nearest one candidate base position."""

    observer = _finite_xy(observer_xy, "observer_xy")
    target = _finite_xy(target_xy, "target_xy")
    bounds = normalize_target_aabb_xy(target_aabb_xy)
    if bounds is None:
        return target, "object_origin_fallback"
    x0, y0, x1, y1 = bounds
    return (
        min(max(observer[0], x0), x1),
        min(max(observer[1], y0), y1),
    ), "aabb_nearest_face"


def supercover_line_cells(start: Cell, end: Cell) -> tuple[Cell, ...]:
    """Return every grid cell touched by the closed segment ``start -> end``.

    Corner crossings include both side-adjacent cells.  This conservative
    supercover prevents a zero-width ray from slipping diagonally through the
    corner shared by two wall cells.
    """

    row, col = int(start[0]), int(start[1])
    end_row, end_col = int(end[0]), int(end[1])
    dcol = end_col - col
    drow = end_row - row
    ncol = abs(dcol)
    nrow = abs(drow)
    step_col = 0 if dcol == 0 else (1 if dcol > 0 else -1)
    step_row = 0 if drow == 0 else (1 if drow > 0 else -1)
    col_steps = 0
    row_steps = 0
    cells: list[Cell] = [(row, col)]

    def append(cell: Cell) -> None:
        if cell != cells[-1]:
            cells.append(cell)

    while col_steps < ncol or row_steps < nrow:
        decision = (1 + 2 * col_steps) * nrow - (1 + 2 * row_steps) * ncol
        if decision == 0:
            # The segment crosses a grid corner.  Both adjacent cells are part
            # of the supercover and must be open for an unobstructed segment.
            if col_steps < ncol:
                append((row, col + step_col))
            if row_steps < nrow:
                append((row + step_row, col))
            col += step_col
            row += step_row
            col_steps += 1
            row_steps += 1
            append((row, col))
        elif decision < 0:
            col += step_col
            col_steps += 1
            append((row, col))
        else:
            row += step_row
            row_steps += 1
            append((row, col))
    return tuple(cells)


def has_structural_line_of_sight(
    structural_open_grid: np.ndarray,
    start: Cell,
    end: Cell,
    *,
    allow_blocked_target_endpoint: bool = False,
    allowed_blocked_target_cells: frozenset[Cell] = frozenset(),
) -> bool:
    """Whether an interaction ray is unobstructed by permanent structure.

    Ordinary navigation checks require the complete supercover to be open.
    For an interaction target mounted *on* a wall, callers may allow only the
    final target-centre cell to be blocked.  Every preceding cell remains a
    hard gate, so this cannot turn an intermediate wall into line of sight.
    """

    return first_structural_line_of_sight_blocker(
        structural_open_grid,
        start,
        end,
        allow_blocked_target_endpoint=allow_blocked_target_endpoint,
        allowed_blocked_target_cells=allowed_blocked_target_cells,
    ) is None


def first_structural_line_of_sight_blocker(
    structural_open_grid: np.ndarray,
    start: Cell,
    end: Cell,
    *,
    allow_blocked_target_endpoint: bool = False,
    allowed_blocked_target_cells: frozenset[Cell] = frozenset(),
) -> Cell | None:
    """Return the first cell that rejects the same LOS contract as above."""

    if structural_open_grid.ndim != 2:
        raise ValueError("structural_open_grid must be 2-D")
    if not isinstance(allow_blocked_target_endpoint, bool):
        raise TypeError("allow_blocked_target_endpoint must be boolean")
    if not isinstance(allowed_blocked_target_cells, frozenset):
        raise TypeError("allowed_blocked_target_cells must be a frozenset")
    height, width = structural_open_grid.shape
    cells = supercover_line_cells(start, end)
    for index, (row, col) in enumerate(cells):
        if not (0 <= row < height and 0 <= col < width):
            return row, col
        if not bool(structural_open_grid[row, col]):
            if (row, col) in allowed_blocked_target_cells:
                continue
            if allow_blocked_target_endpoint and index == len(cells) - 1:
                continue
            return row, col
    return None


def facing_yaw(station_xy: Sequence[float], target_xy: Sequence[float]) -> float:
    """World yaw that faces the interaction target from a station."""

    return math.atan2(
        float(target_xy[1]) - float(station_xy[1]),
        float(target_xy[0]) - float(station_xy[0]),
    )


def compute_source_geodesic_distances(
    navigation_open_grid: np.ndarray,
    source_cell: Cell,
    *,
    resolution: float = 1.0,
    allow_diagonal: bool = True,
) -> np.ndarray:
    """Compute conservative grid geodesics from ``source_cell``.

    The result is simulator-independent and contains distances in world-space
    metres.  Diagonal moves are allowed by default, but may not cut between two
    blocked side-adjacent cells.  This matches the structural safety contract
    used by :func:`supercover_line_cells` instead of treating a point robot as
    able to slip through an occupied corner.

    Callers selecting several interaction targets from the same source should
    compute this grid once and pass it to
    :func:`select_interaction_station_cells`.
    """

    if navigation_open_grid.ndim != 2:
        raise ValueError("navigation_open_grid must be 2-D")
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    height, width = navigation_open_grid.shape
    source = (int(source_cell[0]), int(source_cell[1]))
    if not (0 <= source[0] < height and 0 <= source[1] < width):
        raise ValueError("source_cell lies outside navigation_open_grid")
    if not bool(navigation_open_grid[source]):
        raise ValueError("source_cell must be open")

    distances = np.full((height, width), np.inf, dtype=np.float64)
    distances[source] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, source[0], source[1])]
    cardinal = ((-1, 0), (0, -1), (0, 1), (1, 0))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    neighbours = cardinal + diagonal if allow_diagonal else cardinal

    while queue:
        distance, row, col = heapq.heappop(queue)
        if distance > float(distances[row, col]):
            continue
        for drow, dcol in neighbours:
            next_row, next_col = row + drow, col + dcol
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if not bool(navigation_open_grid[next_row, next_col]):
                continue
            is_diagonal = drow != 0 and dcol != 0
            if is_diagonal and (
                not bool(navigation_open_grid[row + drow, col])
                or not bool(navigation_open_grid[row, col + dcol])
            ):
                continue
            step_cost = resolution * (math.sqrt(2.0) if is_diagonal else 1.0)
            candidate_distance = distance + step_cost
            if candidate_distance + 1.0e-12 < float(distances[next_row, next_col]):
                distances[next_row, next_col] = candidate_distance
                heapq.heappush(queue, (candidate_distance, next_row, next_col))
    return distances


def select_interaction_station_cells(
    *,
    target_xy: Sequence[float],
    target_aabb_xy: Sequence[float] | None = None,
    navigation_open_grid: np.ndarray,
    structural_open_grid: np.ndarray,
    component_labels: np.ndarray,
    source_component_label: int,
    resolution: float,
    map_size: int,
    maximum_center_distance_m: float,
    room_instance_map: np.ndarray | None = None,
    target_room_ids: Sequence[int] | None = None,
    prefer_same_room: bool = True,
    allow_cross_room_fallback_when_target_room_known: bool = False,
    allowed_blocked_target_cells: frozenset[Cell] = frozenset(),
    source_cell: Cell | None = None,
    source_geodesic_distance_grid: np.ndarray | None = None,
) -> InteractionStationSelection:
    """Enumerate connected, structurally visible interaction stations.

    When the target has a known official room, prefer an interaction station
    in the room on its side of the target's nearest AABB surface.  If none
    exists, retain structurally visible cross-room candidates instead of
    treating room metadata as a wall.  Structural line-of-sight remains the
    wall-crossing gate.

    When ``source_cell`` or a precomputed
    ``source_geodesic_distance_grid`` is provided, candidates within the chosen
    room group are ranked by reachable path length from the source, not merely
    by target proximity.  Pass a precomputed grid when selecting many targets
    from the same source.  Supplying both forms is rejected to keep the source
    of the ordering unambiguous.
    """

    if maximum_center_distance_m <= 0.0:
        raise ValueError("maximum_center_distance_m must be positive")
    if not isinstance(allow_cross_room_fallback_when_target_room_known, bool):
        raise TypeError(
            "allow_cross_room_fallback_when_target_room_known must be boolean"
        )
    expected_shape = (map_size, map_size)
    for name, grid in (
        ("navigation_open_grid", navigation_open_grid),
        ("structural_open_grid", structural_open_grid),
        ("component_labels", component_labels),
    ):
        if grid.shape != expected_shape:
            raise ValueError(f"{name} shape does not match map_size")
    if room_instance_map is not None and room_instance_map.shape != expected_shape:
        raise ValueError("room_instance_map shape does not match map_size")
    if target_room_ids is not None:
        if room_instance_map is None:
            raise ValueError("target_room_ids require room_instance_map")
        if isinstance(target_room_ids, (str, bytes)):
            raise TypeError("target_room_ids must be an integer sequence")
        normalized_target_room_ids = tuple(
            sorted({int(value) for value in target_room_ids})
        )
        if not normalized_target_room_ids or any(
            value <= 0 for value in normalized_target_room_ids
        ):
            raise ValueError("target_room_ids must contain positive room ids")
    else:
        normalized_target_room_ids = ()
    if source_cell is not None and source_geodesic_distance_grid is not None:
        raise ValueError(
            "provide source_cell or source_geodesic_distance_grid, not both"
        )
    if source_geodesic_distance_grid is not None:
        if source_geodesic_distance_grid.shape != expected_shape:
            raise ValueError(
                "source_geodesic_distance_grid shape does not match map_size"
            )
        geodesic_distances = np.asarray(
            source_geodesic_distance_grid, dtype=np.float64
        )
    elif source_cell is not None:
        geodesic_distances = compute_source_geodesic_distances(
            navigation_open_grid,
            source_cell,
            resolution=resolution,
        )
    else:
        geodesic_distances = None

    target = _finite_xy(target_xy, "target_xy")
    target_bounds = normalize_target_aabb_xy(target_aabb_xy)
    target_cell = world_to_cell(target, resolution=resolution, map_size=map_size)
    if not (
        0 <= target_cell[0] < map_size and 0 <= target_cell[1] < map_size
    ):
        raise ValueError("target_xy maps outside the supplied grids")
    target_center_room_id: int | None = None
    if room_instance_map is not None:
        target_center_room_id = int(room_instance_map[target_cell])
        if target_center_room_id == 0:
            target_center_room_id = None
    effective_target_room_ids = (
        normalized_target_room_ids
        if normalized_target_room_ids
        else (() if target_center_room_id is None else (target_center_room_id,))
    )
    target_room_id = (
        effective_target_room_ids[0]
        if len(effective_target_room_ids) == 1
        else target_center_room_id
    )
    target_room_source = (
        "official_object_in_rooms"
        if normalized_target_room_ids
        else (
            "target_center_pixel"
            if target_center_room_id is not None
            else "unresolved"
        )
    )

    x0, y0, x1, y1 = target_bounds or (*target, *target)
    search_corners = (
        world_to_cell(
            (x0 - maximum_center_distance_m, y0 - maximum_center_distance_m),
            resolution=resolution,
            map_size=map_size,
        ),
        world_to_cell(
            (x1 + maximum_center_distance_m, y1 + maximum_center_distance_m),
            resolution=resolution,
            map_size=map_size,
        ),
    )
    row_min = max(0, min(cell[0] for cell in search_corners) - 2)
    row_max = min(map_size, max(cell[0] for cell in search_corners) + 3)
    col_min = max(0, min(cell[1] for cell in search_corners) - 2)
    col_max = min(map_size, max(cell[1] for cell in search_corners) + 3)
    same_room: list[tuple[float, float, Cell]] = []
    cross_room: list[tuple[float, float, Cell]] = []
    raw_open_count = 0
    raw_connected_count = 0
    occluded_count = 0
    occlusion_witnesses: list[tuple[Cell, Cell, Cell]] = []
    unreachable_count = 0
    for row in range(row_min, row_max):
        for col in range(col_min, col_max):
            cell = (row, col)
            if not bool(navigation_open_grid[cell]):
                continue
            world = cell_to_world(cell, resolution=resolution, map_size=map_size)
            interaction_target, _ = nearest_interaction_target_xy(
                world,
                target,
                target_aabb_xy=target_bounds,
            )
            distance = math.dist(world, interaction_target)
            if distance > maximum_center_distance_m + 1.0e-7:
                continue
            raw_open_count += 1
            if int(component_labels[cell]) != int(source_component_label):
                continue
            raw_connected_count += 1
            interaction_target_cell = world_to_cell(
                interaction_target,
                resolution=resolution,
                map_size=map_size,
            )
            blocker = first_structural_line_of_sight_blocker(
                structural_open_grid,
                cell,
                interaction_target_cell,
                allow_blocked_target_endpoint=True,
                allowed_blocked_target_cells=allowed_blocked_target_cells,
            )
            if blocker is not None:
                occluded_count += 1
                occlusion_witnesses.append(
                    (cell, interaction_target_cell, blocker)
                )
                continue
            if geodesic_distances is not None:
                source_distance = float(geodesic_distances[cell])
                if not math.isfinite(source_distance):
                    unreachable_count += 1
                    continue
            else:
                # Preserve the legacy target-proximity ordering when no source
                # geodesic information is requested.
                source_distance = distance
            interaction_side_room_id: int | None = None
            if room_instance_map is not None:
                ray_cells = supercover_line_cells(cell, interaction_target_cell)
                cells_before_surface = (
                    ray_cells[:-1] if len(ray_cells) > 1 else ray_cells
                )
                for side_cell in reversed(cells_before_surface):
                    if (
                        0 <= side_cell[0] < map_size
                        and 0 <= side_cell[1] < map_size
                        and bool(structural_open_grid[side_cell])
                    ):
                        room_id = int(room_instance_map[side_cell])
                        if room_id > 0:
                            interaction_side_room_id = room_id
                            break
            accepted_room_ids = (
                (interaction_side_room_id,)
                if interaction_side_room_id is not None
                else effective_target_room_ids
            )
            is_same_room = bool(
                room_instance_map is not None
                and accepted_room_ids
                and int(room_instance_map[cell]) in accepted_room_ids
            )
            (same_room if is_same_room else cross_room).append(
                (source_distance, distance, cell)
            )

    same_room.sort(key=lambda item: (item[0], item[1], item[2]))
    cross_room.sort(key=lambda item: (item[0], item[1], item[2]))
    known_target_room_constraint = bool(
        prefer_same_room
        and effective_target_room_ids
        and same_room
        and not allow_cross_room_fallback_when_target_room_known
    )
    apply_same_room = bool(known_target_room_constraint and same_room)
    selected = (
        same_room
        if known_target_room_constraint
        else [*same_room, *cross_room]
    )
    return InteractionStationSelection(
        cells=tuple(cell for _, _, cell in selected),
        target_cell=target_cell,
        target_room_id=target_room_id,
        raw_open_within_radius_count=raw_open_count,
        raw_open_connected_count=raw_connected_count,
        structural_occlusion_rejection_count=occluded_count,
        same_room_candidate_count=len(same_room),
        cross_room_candidate_count=len(cross_room),
        same_room_preference_applied=apply_same_room,
        cross_room_fallback_used=bool(
            not known_target_room_constraint and cross_room
        ),
        known_target_room_constraint_applied=known_target_room_constraint,
        source_geodesic_sort_applied=geodesic_distances is not None,
        source_geodesic_unreachable_rejection_count=unreachable_count,
        target_center_room_id=target_center_room_id,
        target_room_ids=effective_target_room_ids,
        target_room_source=target_room_source,
        room_match_policy="interaction_side_ray_then_target_room_fallback",
        same_room_candidate_cells=tuple(cell for _, _, cell in same_room),
        cross_room_candidate_cells=tuple(cell for _, _, cell in cross_room),
        structural_occlusion_witnesses=tuple(occlusion_witnesses),
    )
