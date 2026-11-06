"""Private, geometry-only base-route planning for the Oracle.

This module is deliberately simulator independent.  The live Oracle may read
axis-aligned bounds from the simulator and pass them here, but the returned
waypoints are only consumed to author ordinary relative ``move_*`` and
``turn_*`` actions.  No route, object coordinate, or obstacle map is exposed
to a policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Sequence


class PrivateBaseRouteError(ValueError):
    """Raised when private route geometry is malformed."""


@dataclass(frozen=True, slots=True)
class XYObstacle:
    """An axis-aligned obstacle in the private world XY plane."""

    name: str
    minimum: tuple[float, float]
    maximum: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PrivateBaseRouteError("obstacle name must be non-empty")
        minimum = _vector(self.minimum, "minimum")
        maximum = _vector(self.maximum, "maximum")
        if not all(b > a for a, b in zip(minimum, maximum)):
            raise PrivateBaseRouteError("obstacle bounds must have positive area")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)


@dataclass(frozen=True, slots=True)
class XYRoutePlan:
    """A collision-free polyline ending at a reachable manipulation station."""

    goal_xy: tuple[float, float]
    waypoints: tuple[tuple[float, float], ...]
    target_distance_m: float
    path_length_m: float
    start_escape_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_xy", _vector(self.goal_xy, "goal_xy"))
        points = tuple(_vector(point, "waypoint") for point in self.waypoints)
        if not points or points[-1] != self.goal_xy:
            raise PrivateBaseRouteError("waypoints must end at goal_xy")
        object.__setattr__(self, "waypoints", points)
        distance = _finite_positive(self.target_distance_m, "target_distance_m")
        length = _finite_positive(self.path_length_m, "path_length_m")
        object.__setattr__(self, "target_distance_m", distance)
        object.__setattr__(self, "path_length_m", length)
        if not isinstance(self.start_escape_required, bool):
            raise PrivateBaseRouteError("start_escape_required must be boolean")


def plan_xy_route(
    *,
    start_xy: Sequence[float],
    target_xy: Sequence[float],
    desired_distance_m: float,
    obstacles: Iterable[XYObstacle] = (),
    clearance_m: float = 0.42,
    maximum_station_distance_m: float = 1.26,
    angular_samples: int = 32,
) -> XYRoutePlan | None:
    """Find a short collision-free route to a ring around ``target_xy``.

    Candidate stations start at the requested arm distance and extend to the
    local symbolic workspace.  The graph contains only expanded AABB corners,
    so it is deterministic, cheap, and independent of a privileged occupancy
    map.  ``None`` means the live caller should retain its conservative direct
    fallback and let the physical action gate decide.
    """

    start = _vector(start_xy, "start_xy")
    target = _vector(target_xy, "target_xy")
    desired = _finite_positive(desired_distance_m, "desired_distance_m")
    clearance = _finite_positive(clearance_m, "clearance_m")
    maximum = _finite_positive(
        maximum_station_distance_m,
        "maximum_station_distance_m",
    )
    if desired > maximum:
        raise PrivateBaseRouteError("desired station distance exceeds maximum")
    if isinstance(angular_samples, bool) or not isinstance(angular_samples, int):
        raise PrivateBaseRouteError("angular_samples must be an integer")
    if angular_samples < 8:
        raise PrivateBaseRouteError("angular_samples must be at least 8")

    clean_obstacles = tuple(obstacles)
    if any(not isinstance(item, XYObstacle) for item in clean_obstacles):
        raise PrivateBaseRouteError("obstacles must contain XYObstacle values")
    # Prefer the current radial direction, then enumerate the full ring.  A
    # few extra radii handle a target whose support furniture occupies the
    # exact desired-radius ring (e.g. a toy box on a desk).
    radial = (start[0] - target[0], start[1] - target[1])
    radial_angle = math.atan2(radial[1], radial[0])
    radii = _station_radii(desired, maximum)
    candidates: list[
        tuple[
            float,
            float,
            tuple[float, float],
            tuple[tuple[float, float], ...],
            bool,
        ]
    ] = []
    for radius in radii:
        for index in range(angular_samples):
            angle = radial_angle + 2.0 * math.pi * index / angular_samples
            goal = (
                target[0] + radius * math.cos(angle),
                target[1] + radius * math.sin(angle),
            )
            if _point_blocked(goal, clean_obstacles, clearance):
                continue
            path, start_escape_required = _shortest_polyline(
                start,
                goal,
                clean_obstacles,
                clearance,
            )
            if path is None:
                continue
            length = _polyline_length(start, path)
            # Prefer the nearest reachable manipulation ring.  A path-only
            # score incorrectly selected the maximum 1.26 m ring in an empty
            # room because that shortens the base path.
            candidates.append(
                (radius, length, goal, path, start_escape_required)
            )
    if not candidates:
        return None
    _, _, goal, path, start_escape_required = min(
        candidates,
        key=lambda row: (row[0], row[1], row[2]),
    )
    return XYRoutePlan(
        goal_xy=goal,
        waypoints=path,
        target_distance_m=_distance(goal, target),
        path_length_m=_polyline_length(start, path),
        start_escape_required=start_escape_required,
    )


def _station_radii(desired: float, maximum: float) -> tuple[float, ...]:
    values = [desired]
    for fraction in (0.25, 0.50, 0.75, 1.0):
        value = desired + fraction * (maximum - desired)
        if value <= maximum + 1e-9 and all(abs(value - old) > 1e-6 for old in values):
            values.append(value)
    return tuple(values)


def _shortest_polyline(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: tuple[XYObstacle, ...],
    clearance: float,
) -> tuple[tuple[tuple[float, float], ...] | None, bool]:
    escape_points = _start_escape_points(start, obstacles, clearance)
    start_escape_required = bool(escape_points)
    if not start_escape_required and _segment_clear(start, goal, obstacles, clearance):
        return (goal,), False
    nodes: list[tuple[float, float]] = [start, goal]
    for obstacle in obstacles:
        minimum = (obstacle.minimum[0] - clearance - 0.025, obstacle.minimum[1] - clearance - 0.025)
        maximum = (obstacle.maximum[0] + clearance + 0.025, obstacle.maximum[1] + clearance + 0.025)
        nodes.extend(
            (
                (minimum[0], minimum[1]),
                (minimum[0], maximum[1]),
                (maximum[0], minimum[1]),
                (maximum[0], maximum[1]),
            )
        )
    # If the base starts inside an inflated clearance band but outside the
    # raw furniture AABB, require an explicit outward escape waypoint. This
    # avoids the old ``None -> direct route`` fallback that drove straight
    # back into the same table edge.
    nodes.extend(escape_points)
    # A quadratic visibility graph is small in household scenes. Keep only
    # nodes near the start-goal corridor without ever dropping/reordering the
    # two distinguished endpoints.
    corridor = max(3.0, _distance(start, goal) + 2.0)
    nodes = [
        start,
        goal,
        *(
            point
            for point in nodes[2:]
            if _distance(point, start) <= corridor
            or _distance(point, goal) <= corridor
        ),
    ]
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if left == 0:
                # Do not jump from an expanded-obstacle start directly to the
                # goal. Only a mandatory escape point may be reached first.
                if start_escape_required and nodes[right] not in escape_points:
                    continue
                clear = (
                    _start_escape_segment_clear(start, nodes[right], obstacles)
                    if start_escape_required
                    else _segment_clear(
                        nodes[left],
                        nodes[right],
                        obstacles,
                        clearance,
                    )
                )
            else:
                clear = _segment_clear(
                    nodes[left],
                    nodes[right],
                    obstacles,
                    clearance,
                )
            if not clear:
                continue
            cost = _distance(nodes[left], nodes[right])
            adjacency[left].append((right, cost))
            adjacency[right].append((left, cost))
    distances = [float("inf")] * len(nodes)
    previous: list[int | None] = [None] * len(nodes)
    distances[0] = 0.0
    queue: list[tuple[float, int]] = [(0.0, 0)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node] + 1e-12:
            continue
        if node == 1:
            break
        for neighbor, edge in adjacency[node]:
            candidate = distance + edge
            if candidate < distances[neighbor] - 1e-12:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    if not math.isfinite(distances[1]):
        return None, start_escape_required
    indices: list[int] = []
    current: int | None = 1
    while current is not None:
        indices.append(current)
        current = previous[current]
    indices.reverse()
    return tuple(nodes[index] for index in indices[1:]), start_escape_required


def _start_escape_points(
    start: tuple[float, float],
    obstacles: tuple[XYObstacle, ...],
    clearance: float,
) -> tuple[tuple[float, float], ...]:
    """Return outward points for a start inside an expanded AABB only.

    This occurs when the robot is already just outside a desk/table but the
    carried-object safety margin overlaps its footprint. The first segment is
    still executed by registered physical ``move_*`` actions; the relaxed
    check below only permits leaving the already-occupied clearance band.
    """

    points: list[tuple[float, float]] = []
    # Keep enough *net* room to rotate the full R1Pro footprint after the
    # caller's 4 cm waypoint-reached tolerance is applied.  The old 4 cm
    # outward margin was cancelled exactly by that tolerance, so the live
    # v34f robot began its first turn on the inflated desk boundary and
    # immediately raised a real base-contact witness.  A 25 cm waypoint
    # margin leaves at least 21 cm beyond the planning boundary without
    # weakening any physical contact gate.
    margin = 0.25
    for obstacle in obstacles:
        raw_minimum = obstacle.minimum
        raw_maximum = obstacle.maximum
        expanded_minimum = (
            raw_minimum[0] - clearance,
            raw_minimum[1] - clearance,
        )
        expanded_maximum = (
            raw_maximum[0] + clearance,
            raw_maximum[1] + clearance,
        )
        if not _point_in_rect(start, expanded_minimum, expanded_maximum):
            continue
        if _point_in_rect(start, raw_minimum, raw_maximum):
            # The base center is actually inside furniture; no geometric
            # planner should fabricate a safe route from that state.
            return ()
        candidates = (
            (start[0], expanded_minimum[1] - margin),
            (start[0], expanded_maximum[1] + margin),
            (expanded_minimum[0] - margin, start[1]),
            (expanded_maximum[0] + margin, start[1]),
        )
        distances = tuple(_distance(start, point) for point in candidates)
        for index in sorted(range(len(candidates)), key=lambda item: distances[item]):
            point = candidates[index]
            if _point_in_rect(point, expanded_minimum, expanded_maximum):
                continue
            # v34h proved that the nearest outward point can be physically
            # blocked by geometry omitted from the coarse AABB set.  Retain
            # every cardinal exit so the visibility graph can choose the
            # shortest complete route, rather than committing to the nearest
            # local exit before considering the downstream path.
            if point not in points:
                points.append(point)
    return tuple(points)


def _start_escape_segment_clear(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: tuple[XYObstacle, ...],
) -> bool:
    """Allow only an outward retreat that does not enter raw furniture."""

    return all(
        not _segment_intersects_rect(
            start,
            goal,
            obstacle.minimum,
            obstacle.maximum,
        )
        for obstacle in obstacles
    )


def _segment_clear(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: tuple[XYObstacle, ...],
    clearance: float,
) -> bool:
    for obstacle in obstacles:
        minimum = (obstacle.minimum[0] - clearance, obstacle.minimum[1] - clearance)
        maximum = (obstacle.maximum[0] + clearance, obstacle.maximum[1] + clearance)
        if _segment_intersects_rect(start, goal, minimum, maximum):
            return False
    return True


def _point_blocked(
    point: tuple[float, float],
    obstacles: tuple[XYObstacle, ...],
    clearance: float,
) -> bool:
    for obstacle in obstacles:
        if (
            obstacle.minimum[0] - clearance <= point[0] <= obstacle.maximum[0] + clearance
            and obstacle.minimum[1] - clearance <= point[1] <= obstacle.maximum[1] + clearance
        ):
            return True
    return False


def _segment_intersects_rect(
    start: tuple[float, float],
    goal: tuple[float, float],
    minimum: tuple[float, float],
    maximum: tuple[float, float],
) -> bool:
    if _point_in_rect(start, minimum, maximum) or _point_in_rect(goal, minimum, maximum):
        return True
    t0, t1 = 0.0, 1.0
    for origin, delta, lower, upper in zip(
        start,
        (goal[0] - start[0], goal[1] - start[1]),
        minimum,
        maximum,
    ):
        if abs(delta) <= 1e-12:
            if lower <= origin <= upper:
                continue
            return False
            continue
        enter = (lower - origin) / delta
        exit = (upper - origin) / delta
        if enter > exit:
            enter, exit = exit, enter
        t0 = max(t0, enter)
        t1 = min(t1, exit)
        if t0 >= t1:
            return False
    return t0 < t1


def _point_in_rect(
    point: tuple[float, float],
    minimum: tuple[float, float],
    maximum: tuple[float, float],
) -> bool:
    return (
        minimum[0] <= point[0] <= maximum[0]
        and minimum[1] <= point[1] <= maximum[1]
    )


def _polyline_length(
    start: tuple[float, float],
    waypoints: Sequence[tuple[float, float]],
) -> float:
    previous = start
    total = 0.0
    for point in waypoints:
        total += _distance(previous, point)
        previous = point
    return total


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _vector(value: Sequence[float], name: str) -> tuple[float, float]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise PrivateBaseRouteError(f"{name} must be a finite 2-vector") from exc
    if len(values) != 2 or not all(math.isfinite(item) for item in values):
        raise PrivateBaseRouteError(f"{name} must be a finite 2-vector")
    return values


def _finite_positive(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PrivateBaseRouteError(f"{name} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise PrivateBaseRouteError(f"{name} must be positive and finite")
    return result
