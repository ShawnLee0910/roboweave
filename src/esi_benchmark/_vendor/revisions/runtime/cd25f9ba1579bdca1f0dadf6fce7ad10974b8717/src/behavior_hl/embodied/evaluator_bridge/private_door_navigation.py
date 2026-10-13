"""Private room / portal dependencies for door-aware Oracle navigation.

The module is deliberately simulator-agnostic.  Live adapters bind official
room labels, door objects, and traversability arrays in the resident runner;
the public policy never receives any of these values.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


# Mirrors OmniGibson's official ``STRUCTURAL_DOOR_CATEGORIES`` without
# importing the simulator in CPU-only contract tests.  BEHAVIOR scene assets
# also use the ordinary ``door`` category for openable two-room portals (for
# example, office doors).  Keep the official set available for provenance,
# but use the expanded set when building the private room graph.  Admission
# still requires a live Open state and exactly two official ``in_rooms``
# bindings, so an arbitrary object named "door" cannot become a portal.
DOOR_CATEGORIES = frozenset({"sliding_door", "garage_door"})
OPENABLE_ROOM_PORTAL_CATEGORIES = frozenset({*DOOR_CATEGORIES, "door"})
DOOR_DEPENDENCY_VERSION = "behavior2026.private-room-door-dependency/v1"
DOOR_MAP_OVERLAY_VERSION = "behavior2026.private-open-door-local-overlay/v1"


def route_provider_report(
    route_report: Mapping[str, object],
    *,
    maximum_depth: int = 4,
) -> Mapping[str, object]:
    """Return the bounded provider payload used for route classification.

    Navigation wrappers append execution status around the underlying route
    provider report.  Classifiers must inspect that provider payload rather
    than silently treating absent wrapper fields as a different geometry
    signature.  The traversal is deliberately bounded and cycle checked so a
    malformed private diagnostic cannot expand control flow or recurse.
    """

    if not isinstance(route_report, Mapping):
        raise TypeError("route_report must be a mapping")
    if (
        isinstance(maximum_depth, bool)
        or not isinstance(maximum_depth, int)
        or maximum_depth < 0
    ):
        raise ValueError("maximum_depth must be a non-negative integer")
    current: Mapping[str, object] = route_report
    seen: set[int] = set()
    for _ in range(maximum_depth + 1):
        identity = id(current)
        if identity in seen:
            raise ValueError("route provider report contains a cycle")
        seen.add(identity)
        if all(
            field in current
            for field in (
                "reason",
                "station_raw_open_connected_count",
                "station_structural_terminal_allowed_cell_count",
                "station_structural_occlusion_rejection_count",
            )
        ):
            return current
        nested = current.get("route_provider_private")
        if nested is None:
            return current
        if not isinstance(nested, Mapping):
            raise TypeError("nested route_provider_private must be a mapping")
        current = nested
    raise ValueError("route provider report nesting exceeds maximum_depth")


def same_room_component_partition_requires_portal_chain(
    *,
    current_room: str,
    target_room: str,
    route_report: Mapping[str, object],
) -> bool:
    """Recognize the strict same-room / different-component signature."""

    provider = route_provider_report(route_report)
    raw = provider.get("station_raw_open_connected_count")
    terminal = provider.get("station_structural_terminal_allowed_cell_count")
    occluded = provider.get("station_structural_occlusion_rejection_count")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (
        raw,
        terminal,
        occluded,
    )):
        return False
    return bool(
        current_room == target_room
        and provider.get("reason") == "no_traversable_approach_station"
        and raw == 0
        and terminal > 0
        and occluded == 0
    )


@dataclass(frozen=True, slots=True)
class DoorPortal:
    """One live door edge in the private room graph."""

    name: str
    rooms: tuple[str, str]
    center_xy: tuple[float, float]
    is_open: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("door portal name must be non-empty")
        if len(self.rooms) != 2 or any(
            not isinstance(room, str) or not room for room in self.rooms
        ):
            raise ValueError("door portal must connect exactly two named rooms")
        if self.rooms[0] == self.rooms[1]:
            raise ValueError("door portal rooms must be distinct")
        if len(self.center_xy) != 2 or not all(
            math.isfinite(float(value)) for value in self.center_xy
        ):
            raise ValueError("door portal center must be a finite XY pair")
        if not isinstance(self.is_open, bool):
            raise TypeError("door portal open state must be boolean")


@dataclass(frozen=True, slots=True)
class DoorDependencyPlan:
    """Deterministic shortest room path represented by its portal actions."""

    start_room: str
    target_room: str
    room_path: tuple[str, ...]
    portals: tuple[DoorPortal, ...]
    version: str = DOOR_DEPENDENCY_VERSION


def plan_room_door_dependencies(
    *,
    start_room: str,
    target_room: str,
    portals: Iterable[DoorPortal],
) -> DoorDependencyPlan | None:
    """Return the deterministic shortest portal chain, or ``None``.

    Door state does not affect topological connectivity: a closed door remains
    a valid edge because opening it is precisely the action dependency being
    authored.  Ties are resolved by door name and then destination room.
    """

    if not isinstance(start_room, str) or not start_room:
        raise ValueError("start_room must be non-empty")
    if not isinstance(target_room, str) or not target_room:
        raise ValueError("target_room must be non-empty")
    ordered = tuple(sorted(tuple(portals), key=lambda item: item.name))
    if len({portal.name for portal in ordered}) != len(ordered):
        raise ValueError("door portal names must be unique")
    if start_room == target_room:
        return DoorDependencyPlan(
            start_room=start_room,
            target_room=target_room,
            room_path=(start_room,),
            portals=(),
        )

    adjacency: dict[str, list[tuple[str, DoorPortal]]] = {}
    for portal in ordered:
        left, right = portal.rooms
        adjacency.setdefault(left, []).append((right, portal))
        adjacency.setdefault(right, []).append((left, portal))
    for edges in adjacency.values():
        edges.sort(key=lambda row: (row[1].name, row[0]))

    frontier: deque[str] = deque([start_room])
    previous: dict[str, tuple[str, DoorPortal] | None] = {start_room: None}
    while frontier:
        room = frontier.popleft()
        for neighbor, portal in adjacency.get(room, ()):
            if neighbor in previous:
                continue
            previous[neighbor] = (room, portal)
            if neighbor == target_room:
                frontier.clear()
                break
            frontier.append(neighbor)
    if target_room not in previous:
        return None

    reversed_rooms = [target_room]
    reversed_portals: list[DoorPortal] = []
    cursor = target_room
    while cursor != start_room:
        edge = previous[cursor]
        if edge is None:
            raise RuntimeError("door dependency predecessor chain is incomplete")
        prior, portal = edge
        reversed_portals.append(portal)
        reversed_rooms.append(prior)
        cursor = prior
    return DoorDependencyPlan(
        start_room=start_room,
        target_room=target_room,
        room_path=tuple(reversed(reversed_rooms)),
        portals=tuple(reversed(reversed_portals)),
    )


def plan_room_door_dependencies_from_reachable_portals(
    *,
    start_room: str,
    target_room: str,
    portals: Iterable[DoorPortal],
    reachable_portal_names: Iterable[str],
) -> DoorDependencyPlan | None:
    """Plan from the current traversability component to ``target_room``.

    Room segmentation is not itself a connectivity graph: two named rooms can
    share an open passage without a door object.  Consequently, using only
    ``start_room`` as the graph root incorrectly rejects routes such as
    copy-room -> corridor -> closed office door.  The live adapter first
    proves which portal objects are approachable on the *current* robot-eroded
    map.  This function then treats each proven portal as a deterministic
    entry edge and finds the shortest remaining room/door chain.

    No portal is considered reachable from metadata alone.  That property is
    supplied by the caller after the ordinary route planner has returned a
    concrete path (or the local-reach gate has passed), so open archways are
    handled without inventing a second geometric map implementation here.
    """

    ordered = tuple(sorted(tuple(portals), key=lambda item: item.name))
    by_name = {portal.name: portal for portal in ordered}
    if len(by_name) != len(ordered):
        raise ValueError("door portal names must be unique")
    reachable = tuple(sorted({str(name) for name in reachable_portal_names}))
    unknown = tuple(name for name in reachable if name not in by_name)
    if unknown:
        raise ValueError(f"reachable portal names are unknown: {unknown}")
    if start_room == target_room:
        return DoorDependencyPlan(
            start_room=start_room,
            target_room=target_room,
            room_path=(start_room,),
            portals=(),
        )

    candidates: list[DoorDependencyPlan] = []
    for name in reachable:
        first = by_name[name]
        remaining_portals = tuple(
            portal for portal in ordered if portal.name != first.name
        )
        for downstream_room in first.rooms:
            if downstream_room == target_room:
                suffix_rooms = (target_room,)
                suffix_portals: tuple[DoorPortal, ...] = ()
            else:
                suffix = plan_room_door_dependencies(
                    start_room=downstream_room,
                    target_room=target_room,
                    portals=remaining_portals,
                )
                if suffix is None:
                    continue
                suffix_rooms = suffix.room_path
                suffix_portals = suffix.portals
            room_path = [start_room]
            for room in suffix_rooms:
                if room != room_path[-1]:
                    room_path.append(room)
            candidates.append(
                DoorDependencyPlan(
                    start_room=start_room,
                    target_room=target_room,
                    room_path=tuple(room_path),
                    portals=(first, *suffix_portals),
                )
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda plan: (
            len(plan.portals),
            tuple(portal.name for portal in plan.portals),
            plan.room_path,
        ),
    )


def plan_room_door_dependencies_from_accessible_sides(
    *,
    target_room: str,
    portals: Iterable[DoorPortal],
    accessible_room_by_portal: Mapping[str, str],
) -> DoorDependencyPlan | None:
    """Plan from route-proven portal sides, including split same-room components.

    A concrete route to a door proves only the side containing its selected
    station.  Crossing that exact portal enters its unique other room; the
    ordinary room graph may then find a suffix back into ``target_room``.
    Removing the first portal from the suffix prevents metadata from
    inventing a return through the same edge.  Missing or ambiguous side
    evidence fails closed.
    """

    if not isinstance(target_room, str) or not target_room:
        raise ValueError("target_room must be non-empty")
    ordered = tuple(sorted(tuple(portals), key=lambda item: item.name))
    by_name = {portal.name: portal for portal in ordered}
    if len(by_name) != len(ordered):
        raise ValueError("door portal names must be unique")
    if not isinstance(accessible_room_by_portal, Mapping):
        raise TypeError("accessible_room_by_portal must be a mapping")
    unknown = tuple(
        sorted(name for name in accessible_room_by_portal if name not in by_name)
    )
    if unknown:
        raise ValueError(f"accessible portal names are unknown: {unknown}")

    candidates: list[DoorDependencyPlan] = []
    for name in sorted(accessible_room_by_portal):
        portal = by_name[name]
        accessible_room = accessible_room_by_portal[name]
        if not isinstance(accessible_room, str) or not accessible_room:
            raise ValueError("accessible portal room must be non-empty")
        if accessible_room not in portal.rooms:
            raise ValueError(
                "accessible portal room is not one of its exact two rooms"
            )
        downstream_room = (
            portal.rooms[1]
            if portal.rooms[0] == accessible_room
            else portal.rooms[0]
        )
        remaining = tuple(item for item in ordered if item.name != name)
        suffix = plan_room_door_dependencies(
            start_room=downstream_room,
            target_room=target_room,
            portals=remaining,
        )
        if suffix is None:
            continue
        candidates.append(
            DoorDependencyPlan(
                start_room=accessible_room,
                target_room=target_room,
                room_path=(accessible_room, *suffix.room_path),
                portals=(portal, *suffix.portals),
            )
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda plan: (
            len(plan.portals),
            tuple(portal.name for portal in plan.portals),
            plan.room_path,
        ),
    )


def apply_open_door_local_overlay(
    *,
    current_floor_map: np.ndarray,
    all_doors_open_floor_map: np.ndarray,
    official_door_removed_floor_map: np.ndarray | None = None,
    verified_open: bool,
    door_center_xy: Sequence[float],
    world_to_cell: Callable[[Sequence[float]], Sequence[int]],
    map_resolution_m: float,
    locality_radius_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply only the official open-map pixels local to one proven-open door.

    ``all_doors_open_floor_map`` is an official BEHAVIOR asset, but some
    structural sliding-door footprints are represented only by the official
    ``floor_trav_no_door`` asset. Copying either asset wholesale would silently
    make unopened doors traversable. This function runs only after a live
    ``Open=True`` readback and copies only the center-connected, monotonic
    opening delta inside the existing circular door neighborhood.
    """

    if verified_open is not True:
        raise RuntimeError("door overlay requires live Open=True readback")

    current = np.asarray(current_floor_map)
    opened = np.asarray(all_doors_open_floor_map)
    if current.ndim != 2 or current.size == 0:
        raise ValueError("current floor map must be a non-empty 2D array")
    if opened.shape != current.shape:
        raise ValueError("open-door floor map shape must match the current map")
    removed = (
        None
        if official_door_removed_floor_map is None
        else np.asarray(official_door_removed_floor_map)
    )
    if removed is not None and removed.shape != current.shape:
        raise ValueError("door-removed floor map shape must match the current map")
    resolution = float(map_resolution_m)
    radius = float(locality_radius_m)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("map resolution must be positive and finite")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("door overlay radius must be positive and finite")
    center = tuple(int(value) for value in world_to_cell(door_center_xy))
    if len(center) != 2:
        raise ValueError("world_to_cell must return a row/column pair")
    row, col = center
    if not (0 <= row < current.shape[0] and 0 <= col < current.shape[1]):
        raise ValueError("door center lies outside the official floor map")

    radius_cells = max(1, int(math.ceil(radius / resolution)))
    row_lo, row_hi = max(0, row - radius_cells), min(
        current.shape[0], row + radius_cells + 1
    )
    col_lo, col_hi = max(0, col - radius_cells), min(
        current.shape[1], col + radius_cells + 1
    )
    rr, cc = np.ogrid[row_lo:row_hi, col_lo:col_hi]
    local_mask = (rr - row) ** 2 + (cc - col) ** 2 <= radius_cells**2
    result = np.array(current, copy=True)
    before = result[row_lo:row_hi, col_lo:col_hi]
    source = opened[row_lo:row_hi, col_lo:col_hi]
    changed_mask = local_mask & (before != source)
    changed_count = int(np.count_nonzero(changed_mask))
    monotonic_opening_only = bool(
        np.all((before[changed_mask] == 0) & (source[changed_mask] == 255))
    )
    if changed_count == 0:
        if removed is None:
            raise RuntimeError(
                "official open-door map has no local delta for portal"
            )
        source = removed[row_lo:row_hi, col_lo:col_hi]
        changed_mask = local_mask & (before != source)
        if not bool(changed_mask[row - row_lo, col - col_lo]):
            raise RuntimeError(
                "official door-removed delta does not contain portal center"
            )
        if bool(np.any(changed_mask & ~((before == 0) & (source == 255)))):
            raise RuntimeError(
                "official door-removed delta is not monotonic opening-only"
            )
        center_local = (row - row_lo, col - col_lo)
        connected = np.zeros(changed_mask.shape, dtype=bool)
        frontier: deque[tuple[int, int]] = deque([center_local])
        connected[center_local] = True
        while frontier:
            local_row, local_col = frontier.popleft()
            for next_row, next_col in (
                (local_row - 1, local_col),
                (local_row + 1, local_col),
                (local_row, local_col - 1),
                (local_row, local_col + 1),
            ):
                if not (
                    0 <= next_row < changed_mask.shape[0]
                    and 0 <= next_col < changed_mask.shape[1]
                    and changed_mask[next_row, next_col]
                    and not connected[next_row, next_col]
                ):
                    continue
                connected[next_row, next_col] = True
                frontier.append((next_row, next_col))
        changed_mask = connected
        changed_count = int(np.count_nonzero(changed_mask))
        monotonic_opening_only = True
        source_name = "floor_trav_no_door"
        fallback_reason = "all_open_map_has_no_local_delta"
    else:
        source_name = "floor_trav_open_door"
        fallback_reason = None
    before[changed_mask] = source[changed_mask]
    return result, {
        "overlay_version": DOOR_MAP_OVERLAY_VERSION,
        "door_center_official_cell": [row, col],
        "locality_radius_m": radius,
        "locality_radius_cells": radius_cells,
        "changed_cell_count": changed_count,
        "official_map_source": source_name,
        "fallback_reason": fallback_reason,
        "center_cell_changed": bool(
            changed_mask[row - row_lo, col - col_lo]
        ),
        "monotonic_opening_only": monotonic_opening_only,
        "other_door_regions_preserved": True,
    }
