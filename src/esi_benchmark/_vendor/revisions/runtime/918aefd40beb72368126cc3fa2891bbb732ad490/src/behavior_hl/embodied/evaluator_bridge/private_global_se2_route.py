"""Global private SE(2) route authoring for the kinematic-research lane.

The official BEHAVIOR traversability map remains the topology source, but a
base route is no longer represented by XY alone.  Search nodes retain the
measured 5 cm sub-cell position, the public 10 degree heading bin, and a map
epoch.  Edges are exactly the six registered public base actions.

The search is deliberately lazy with respect to expensive live collision
queries.  A cheap oriented-footprint guard authors a complete candidate
route.  The caller validates that route with the live full-body provider;
any rejected edge is added to ``blocked_transitions`` and the same global
planner is called again.  Thus a rejection is remembered by the global
planner instead of being forgotten when a local recovery returns to XY A*.

Nothing in this module imports OmniGibson, mutates simulator state, renders,
or exposes private route information to the model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import itertools
import math
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from behavior_hl.embodied.runtime.plugins.restricted_kinematic_base import (
    ACTION20_RESTRICTED_KINEMATIC_PUBLIC_TURN_DEG,
    RESTRICTED_KINEMATIC_BASE_ACTIONS,
    RESTRICTED_KINEMATIC_PUBLIC_MOVE_M,
    restricted_kinematic_action_target,
)
from behavior_hl.embodied.evaluator_bridge.interaction_station_geometry import (
    facing_yaw,
    nearest_interaction_target_xy,
)
from behavior_hl.navigation import PlanarPose, normalize_angle


GLOBAL_SE2_ROUTE_VERSION = "behavior2026.global-se2-lazy-public-actions/v5-official-cell-corridor"
GLOBAL_SE2_HEADING_BIN_COUNT = 36
GLOBAL_SE2_XY_QUANTUM_M = 0.05
GLOBAL_SE2_MOVE_M = RESTRICTED_KINEMATIC_PUBLIC_MOVE_M
GLOBAL_SE2_TURN_DEG = ACTION20_RESTRICTED_KINEMATIC_PUBLIC_TURN_DEG

_ACTIONS = (
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
)

GlobalSE2StateKey = tuple[int, int, int, int, int, int]
GlobalSE2TransitionKey = tuple[int, int, int, int, int, int, str]
CellResolver = Callable[[PlanarPose], tuple[int, int]]
PoseGuard = Callable[[PlanarPose], bool]
EdgeGuard = Callable[[PlanarPose, PlanarPose], bool]


def official_cell_corridor(
    *,
    open_cells: Iterable[Sequence[int]],
    start_cell: Sequence[int],
    goal_cells: Iterable[Sequence[int]],
    radius_cells: int,
) -> frozenset[tuple[int, int]]:
    """Return a bounded 4-connected corridor to all reachable goal cells."""

    if (
        isinstance(radius_cells, bool)
        or not isinstance(radius_cells, int)
        or radius_cells < 0
    ):
        raise ValueError("radius_cells must be a non-negative integer")
    open_set = frozenset((int(cell[0]), int(cell[1])) for cell in open_cells)
    start = (int(start_cell[0]), int(start_cell[1]))
    goals = frozenset((int(cell[0]), int(cell[1])) for cell in goal_cells) & open_set
    if start not in open_set or not goals:
        return frozenset()

    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = deque((start,))
    remaining = set(goals)
    remaining.discard(start)
    while queue and remaining:
        row, col = queue.popleft()
        for neighbor in (
            (row - 1, col),
            (row, col - 1),
            (row, col + 1),
            (row + 1, col),
        ):
            if neighbor not in open_set or neighbor in parents:
                continue
            parents[neighbor] = (row, col)
            remaining.discard(neighbor)
            queue.append(neighbor)

    paths = {start}
    for goal in goals - remaining:
        cursor: tuple[int, int] | None = goal
        while cursor is not None and cursor not in paths:
            paths.add(cursor)
            cursor = parents[cursor]
    return frozenset(
        (row + dr, col + dc)
        for row, col in paths
        for dr in range(-radius_cells, radius_cells + 1)
        for dc in range(-radius_cells, radius_cells + 1)
        if (row + dr, col + dc) in open_set
    )


def official_cell_action_estimates(
    *,
    open_cells: Iterable[Sequence[int]],
    goal_cells: Iterable[Sequence[int]],
    cell_size_m: float,
    move_distance_m: float,
) -> dict[tuple[int, int], float]:
    """Estimate remaining public moves from an 8-connected cell distance."""

    cell_size = _positive(cell_size_m, "cell_size_m")
    move_distance = _positive(move_distance_m, "move_distance_m")
    open_set = frozenset((int(cell[0]), int(cell[1])) for cell in open_cells)
    goals = frozenset((int(cell[0]), int(cell[1])) for cell in goal_cells) & open_set
    steps = {cell: 0 for cell in goals}
    queue = deque(sorted(goals))
    while queue:
        row, col = queue.popleft()
        next_steps = steps[(row, col)] + 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = (row + dr, col + dc)
                if neighbor in open_set and neighbor not in steps:
                    steps[neighbor] = next_steps
                    queue.append(neighbor)
    return {
        cell: value * cell_size / move_distance
        for cell, value in steps.items()
    }


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def global_se2_heading_bin(
    yaw_rad: float,
    *,
    heading_bin_count: int = GLOBAL_SE2_HEADING_BIN_COUNT,
) -> int:
    """Quantize a yaw to the public turn lattice."""

    if (
        isinstance(heading_bin_count, bool)
        or not isinstance(heading_bin_count, int)
        or heading_bin_count < 4
    ):
        raise ValueError("heading_bin_count must be an integer >= 4")
    yaw = _finite(yaw_rad, "yaw_rad")
    quantum = 2.0 * math.pi / heading_bin_count
    return int(round(normalize_angle(yaw) / quantum)) % heading_bin_count


def global_se2_state_key(
    *,
    pose: PlanarPose,
    official_cell: Sequence[int],
    map_epoch: int,
    xy_quantum_m: float = GLOBAL_SE2_XY_QUANTUM_M,
    heading_bin_count: int = GLOBAL_SE2_HEADING_BIN_COUNT,
) -> GlobalSE2StateKey:
    """Return the canonical global state identity.

    The official cell preserves compatibility with route artifacts.  The
    quantized world position prevents two poses in the same 10 cm map cell
    from being collapsed, which was the deterministic v115 failure mode.
    """

    if not isinstance(pose, PlanarPose):
        raise TypeError("global SE2 pose must be PlanarPose")
    if len(official_cell) != 2:
        raise ValueError("official_cell must have two values")
    row, col = official_cell
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (row, col)):
        raise TypeError("official_cell values must be integers")
    if row < 0 or col < 0:
        raise ValueError("official_cell values must be non-negative")
    if isinstance(map_epoch, bool) or not isinstance(map_epoch, int) or map_epoch < 0:
        raise ValueError("map_epoch must be a non-negative integer")
    quantum = _positive(xy_quantum_m, "xy_quantum_m")
    for name, value in (("pose.x", pose.x), ("pose.y", pose.y), ("pose.z", pose.z), ("pose.yaw", pose.yaw)):
        _finite(value, name)
    return (
        int(row),
        int(col),
        global_se2_heading_bin(pose.yaw, heading_bin_count=heading_bin_count),
        int(round(float(pose.x) / quantum)),
        int(round(float(pose.y) / quantum)),
        int(map_epoch),
    )


def global_se2_transition_key(
    *,
    pose: PlanarPose,
    official_cell: Sequence[int],
    map_epoch: int,
    action: str,
    xy_quantum_m: float = GLOBAL_SE2_XY_QUANTUM_M,
    heading_bin_count: int = GLOBAL_SE2_HEADING_BIN_COUNT,
) -> GlobalSE2TransitionKey:
    if action not in RESTRICTED_KINEMATIC_BASE_ACTIONS:
        raise ValueError("global SE2 transition action is not registered")
    return (
        *global_se2_state_key(
            pose=pose,
            official_cell=official_cell,
            map_epoch=map_epoch,
            xy_quantum_m=xy_quantum_m,
            heading_bin_count=heading_bin_count,
        ),
        str(action),
    )


@dataclass(frozen=True, slots=True)
class GlobalSE2Goal:
    x: float
    y: float
    yaw_rad: float
    position_tolerance_m: float = 0.12
    yaw_tolerance_deg: float = 7.6
    name: str | None = None
    interaction_target_xy: tuple[float, float] | None = None
    minimum_interaction_distance_m: float | None = None
    maximum_interaction_distance_m: float | None = None
    official_station_cell: tuple[int, int] | None = None
    allowed_yaw_bins: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("x", "y", "yaw_rad"):
            object.__setattr__(self, field_name, _finite(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "position_tolerance_m",
            _positive(self.position_tolerance_m, "position_tolerance_m"),
        )
        yaw_tolerance = _positive(self.yaw_tolerance_deg, "yaw_tolerance_deg")
        if yaw_tolerance > 180.0:
            raise ValueError("yaw_tolerance_deg must not exceed 180")
        object.__setattr__(self, "yaw_tolerance_deg", yaw_tolerance)
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError("goal name must be a non-empty string when provided")
        discrete = self.official_station_cell is not None
        if discrete != bool(self.allowed_yaw_bins):
            raise ValueError(
                "official_station_cell and allowed_yaw_bins must be provided together"
            )
        if discrete:
            cell = self.official_station_cell
            if (
                isinstance(cell, (str, bytes))
                or not isinstance(cell, Sequence)
                or len(cell) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in cell)
                or min(cell) < 0
            ):
                raise ValueError("official_station_cell must contain two non-negative integers")
            if (
                isinstance(self.allowed_yaw_bins, (str, bytes))
                or not isinstance(self.allowed_yaw_bins, Sequence)
            ):
                raise ValueError("allowed_yaw_bins must contain non-negative integers")
            raw_bins = tuple(self.allowed_yaw_bins)
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in raw_bins
            ):
                raise ValueError("allowed_yaw_bins must contain non-negative integers")
            bins = tuple(sorted(set(raw_bins)))
            object.__setattr__(self, "official_station_cell", tuple(cell))
            object.__setattr__(self, "allowed_yaw_bins", bins)
            if self.interaction_target_xy is not None:
                raise ValueError(
                    "discrete station goals cannot use continuous interaction bounds"
                )
        radial_values = (
            self.minimum_interaction_distance_m,
            self.maximum_interaction_distance_m,
        )
        if self.interaction_target_xy is None:
            if any(value is not None for value in radial_values):
                raise ValueError("interaction distance bounds require a target")
        else:
            if len(self.interaction_target_xy) != 2:
                raise ValueError("interaction_target_xy must have two values")
            target = (
                _finite(self.interaction_target_xy[0], "interaction_target_x"),
                _finite(self.interaction_target_xy[1], "interaction_target_y"),
            )
            object.__setattr__(self, "interaction_target_xy", target)
            if any(value is None for value in radial_values):
                raise ValueError("both interaction distance bounds are required")
            minimum = _positive(radial_values[0], "minimum_interaction_distance_m")
            maximum = _positive(radial_values[1], "maximum_interaction_distance_m")
            if minimum > maximum:
                raise ValueError("interaction distance bounds are inverted")
            object.__setattr__(self, "minimum_interaction_distance_m", minimum)
            object.__setattr__(self, "maximum_interaction_distance_m", maximum)


@dataclass(frozen=True, slots=True)
class GlobalSE2RoutePlan:
    status: str
    actions: tuple[str, ...]
    poses: tuple[PlanarPose, ...]
    state_keys: tuple[GlobalSE2StateKey, ...]
    selected_goal_index: int | None
    expanded_states: int
    generated_states: int
    path_cost: float | None
    reason: str
    private_report: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"planned", "no_route"}:
            raise ValueError("global SE2 status is invalid")
        if self.status == "planned":
            if len(self.poses) != len(self.actions) + 1:
                raise ValueError("planned global SE2 poses/actions do not align")
            if len(self.state_keys) != len(self.poses):
                raise ValueError("planned global SE2 state keys do not align")
            if self.selected_goal_index is None or self.path_cost is None:
                raise ValueError("planned global SE2 route lacks goal or cost")
        elif self.actions or self.poses or self.state_keys:
            raise ValueError("no-route global SE2 result must not contain a path")
        if any(action not in RESTRICTED_KINEMATIC_BASE_ACTIONS for action in self.actions):
            raise ValueError("global SE2 route contains a non-public action")
        for value_name in ("expanded_states", "generated_states"):
            value = getattr(self, value_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{value_name} must be a non-negative integer")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("global SE2 reason must be non-empty")
        object.__setattr__(self, "private_report", MappingProxyType(dict(self.private_report)))


def _goal_error(pose: PlanarPose, goal: GlobalSE2Goal) -> tuple[float, float]:
    return (
        math.hypot(float(pose.x) - goal.x, float(pose.y) - goal.y),
        abs(normalize_angle(float(pose.yaw) - goal.yaw_rad)),
    )


def _goal_reached(
    pose: PlanarPose,
    state_key: GlobalSE2StateKey,
    goal: GlobalSE2Goal,
) -> bool:
    if goal.official_station_cell is not None:
        return bool(
            tuple(state_key[:2]) == goal.official_station_cell
            and state_key[2] in goal.allowed_yaw_bins
        )
    position_error, yaw_error = _goal_error(pose, goal)
    local_pose_match = (
        position_error <= goal.position_tolerance_m + 1.0e-9
        and yaw_error <= math.radians(goal.yaw_tolerance_deg) + 1.0e-9
    )
    if not local_pose_match or goal.interaction_target_xy is None:
        return local_pose_match
    interaction_distance = math.hypot(
        float(pose.x) - goal.interaction_target_xy[0],
        float(pose.y) - goal.interaction_target_xy[1],
    )
    return bool(
        goal.minimum_interaction_distance_m - 1.0e-9
        <= interaction_distance
        <= goal.maximum_interaction_distance_m + 1.0e-9
    )


def _heuristic(
    pose: PlanarPose,
    goals: Sequence[GlobalSE2Goal],
    *,
    official_cell: Sequence[int],
    move_distance_m: float,
    turn_angle_deg: float,
    heading_bin_count: int,
    official_cell_action_estimates: Mapping[tuple[int, int], float] | None = None,
) -> float:
    turn_rad = math.radians(turn_angle_deg)
    values = []
    for goal in goals:
        position_error, yaw_error = _goal_error(pose, goal)
        discrete_station = goal.official_station_cell is not None
        if discrete_station:
            yaw_error = min(
                abs(
                    normalize_angle(
                        float(pose.yaw)
                        - yaw_bin * (2.0 * math.pi / heading_bin_count)
                    )
                )
                for yaw_bin in goal.allowed_yaw_bins
            )
            # Admission is the exact official cell below.  This computes the
            # distance to that cell's square, not to its center.
            position_error = (
                0.0
                if tuple(official_cell) == goal.official_station_cell
                else math.hypot(
                    max(abs(pose.x - goal.x) - goal.position_tolerance_m, 0.0),
                    max(abs(pose.y - goal.y) - goal.position_tolerance_m, 0.0),
                )
            )
            position_tolerance_m = 0.0
            yaw_tolerance_deg = 0.0
        else:
            position_tolerance_m = goal.position_tolerance_m
            yaw_tolerance_deg = goal.yaw_tolerance_deg
        translation_steps = max(
            0.0,
            (position_error - position_tolerance_m) / move_distance_m,
        )
        if discrete_station and official_cell_action_estimates is not None:
            translation_steps = max(
                translation_steps,
                float(
                    official_cell_action_estimates.get(
                        (int(official_cell[0]), int(official_cell[1])),
                        translation_steps,
                    )
                ),
            )
        turn_steps = max(
            0.0,
            (yaw_error - math.radians(yaw_tolerance_deg)) / turn_rad,
        )
        values.append(translation_steps + turn_steps)
    return min(values)


def _validated_blocked_transitions(
    values: Iterable[Sequence[Any]],
    *,
    heading_bin_count: int,
) -> frozenset[GlobalSE2TransitionKey]:
    result: set[GlobalSE2TransitionKey] = set()
    for value in values:
        if isinstance(value, (str, bytes)) or len(value) != 7:
            raise ValueError("blocked global SE2 transition must contain seven values")
        numeric = value[:6]
        action = value[6]
        if any(isinstance(item, bool) or not isinstance(item, int) for item in numeric):
            raise TypeError("blocked global SE2 transition state must be integer")
        row, col, yaw_bin, _quantized_x, _quantized_y, map_epoch = numeric
        if row < 0 or col < 0 or map_epoch < 0 or not 0 <= yaw_bin < heading_bin_count:
            raise ValueError("blocked global SE2 transition state is outside bounds")
        if action not in RESTRICTED_KINEMATIC_BASE_ACTIONS:
            raise ValueError("blocked global SE2 transition action is invalid")
        result.add((*tuple(int(item) for item in numeric), str(action)))
    return frozenset(result)


def plan_global_se2_route(
    *,
    start_pose: PlanarPose,
    goals: Sequence[GlobalSE2Goal],
    cell_resolver: CellResolver,
    pose_is_open: PoseGuard,
    edge_is_open: EdgeGuard | None = None,
    map_epoch: int = 0,
    blocked_transitions: Iterable[Sequence[Any]] = (),
    move_distance_m: float = GLOBAL_SE2_MOVE_M,
    turn_angle_deg: float = GLOBAL_SE2_TURN_DEG,
    xy_quantum_m: float = GLOBAL_SE2_XY_QUANTUM_M,
    heading_bin_count: int = GLOBAL_SE2_HEADING_BIN_COUNT,
    maximum_states: int = 100_000,
    official_cell_action_estimates: Mapping[tuple[int, int], float] | None = None,
    heuristic_weight: float = 1.0,
) -> GlobalSE2RoutePlan:
    """Run deterministic A* over the public-action SE(2) state space.

    Omitting ``edge_is_open`` is the endpoint-only symbolic contract: target
    poses are still checked, but the planner adds no stricter swept-edge gate.
    """

    if not isinstance(start_pose, PlanarPose):
        raise TypeError("start_pose must be PlanarPose")
    if not goals or any(not isinstance(goal, GlobalSE2Goal) for goal in goals):
        raise ValueError("at least one valid global SE2 goal is required")
    for callback_name, callback in (
        ("cell_resolver", cell_resolver),
        ("pose_is_open", pose_is_open),
    ):
        if not callable(callback):
            raise TypeError(f"{callback_name} must be callable")
    move_distance = _positive(move_distance_m, "move_distance_m")
    turn_degrees = _positive(turn_angle_deg, "turn_angle_deg")
    if turn_degrees > 180.0:
        raise ValueError("turn_angle_deg must not exceed 180")
    quantum = _positive(xy_quantum_m, "xy_quantum_m")
    weight = _positive(heuristic_weight, "heuristic_weight")
    if weight < 1.0:
        raise ValueError("heuristic_weight must be at least one")
    if (
        isinstance(heading_bin_count, bool)
        or not isinstance(heading_bin_count, int)
        or heading_bin_count < 4
    ):
        raise ValueError("heading_bin_count must be an integer >= 4")
    if isinstance(maximum_states, bool) or not isinstance(maximum_states, int) or maximum_states <= 0:
        raise ValueError("maximum_states must be a positive integer")
    if edge_is_open is not None and not callable(edge_is_open):
        raise TypeError("edge_is_open must be callable when provided")
    blocked = _validated_blocked_transitions(
        blocked_transitions,
        heading_bin_count=heading_bin_count,
    )
    if any(
        goal.official_station_cell is not None
        and max(goal.allowed_yaw_bins) >= heading_bin_count
        for goal in goals
    ):
        raise ValueError("discrete goal yaw bin is outside heading_bin_count")

    start_cell = tuple(int(value) for value in cell_resolver(start_pose))
    start_key = global_se2_state_key(
        pose=start_pose,
        official_cell=start_cell,
        map_epoch=map_epoch,
        xy_quantum_m=quantum,
        heading_bin_count=heading_bin_count,
    )
    if not bool(pose_is_open(start_pose)):
        return GlobalSE2RoutePlan(
            "no_route",
            (),
            (),
            (),
            None,
            0,
            1,
            None,
            "start_pose_footprint_blocked",
            {
                "planner_version": GLOBAL_SE2_ROUTE_VERSION,
                "start_state": list(start_key),
                "map_epoch": int(map_epoch),
                "blocked_transition_count": len(blocked),
                "heuristic_weight": weight,
            },
        )

    pose_by_key: dict[GlobalSE2StateKey, PlanarPose] = {start_key: start_pose}
    g_score: dict[GlobalSE2StateKey, float] = {start_key: 0.0}
    parent: dict[GlobalSE2StateKey, tuple[GlobalSE2StateKey, str]] = {}
    closed: set[GlobalSE2StateKey] = set()
    counter = itertools.count()
    queue: list[tuple[float, float, float, int, GlobalSE2StateKey]] = []
    start_h = _heuristic(
        start_pose,
        goals,
        official_cell=start_cell,
        move_distance_m=move_distance,
        turn_angle_deg=turn_degrees,
        heading_bin_count=heading_bin_count,
        official_cell_action_estimates=official_cell_action_estimates,
    )
    heapq.heappush(
        queue,
        (
            weight * start_h,
            start_h,
            0.0,
            next(counter),
            start_key,
        ),
    )
    expanded = 0
    selected_key: GlobalSE2StateKey | None = None
    selected_goal_index: int | None = None

    while queue and expanded < maximum_states:
        _, _, queued_g, _, key = heapq.heappop(queue)
        if key in closed or queued_g > g_score.get(key, math.inf) + 1.0e-12:
            continue
        pose = pose_by_key[key]
        for goal_index, goal in enumerate(goals):
            if _goal_reached(pose, key, goal):
                selected_key = key
                selected_goal_index = goal_index
                break
        if selected_key is not None:
            break
        closed.add(key)
        expanded += 1

        candidates: list[tuple[float, str, PlanarPose, tuple[int, int], GlobalSE2StateKey]] = []
        for action in _ACTIONS:
            transition = (*key, action)
            if transition in blocked:
                continue
            target = restricted_kinematic_action_target(
                pose,
                action,
                move_distance_m=move_distance,
                turn_angle_deg=turn_degrees,
            )
            if not bool(pose_is_open(target)) or (
                edge_is_open is not None and not bool(edge_is_open(pose, target))
            ):
                continue
            target_cell = tuple(int(value) for value in cell_resolver(target))
            target_key = global_se2_state_key(
                pose=target,
                official_cell=target_cell,
                map_epoch=map_epoch,
                xy_quantum_m=quantum,
                heading_bin_count=heading_bin_count,
            )
            if target_key == key:
                continue
            estimate = _heuristic(
                target,
                goals,
                official_cell=target_cell,
                move_distance_m=move_distance,
                turn_angle_deg=turn_degrees,
                heading_bin_count=heading_bin_count,
                official_cell_action_estimates=official_cell_action_estimates,
            )
            candidates.append((estimate, action, target, target_cell, target_key))
        candidates.sort(key=lambda row: (row[0], _ACTIONS.index(row[1]), row[4]))
        for estimate, action, target, _, target_key in candidates:
            # Count each public action equally.  This directly optimizes the
            # Oracle/model decision surface instead of hiding turns as free.
            tentative_g = g_score[key] + 1.0
            if tentative_g + 1.0e-12 >= g_score.get(target_key, math.inf):
                continue
            g_score[target_key] = tentative_g
            pose_by_key[target_key] = target
            parent[target_key] = (key, action)
            heapq.heappush(
                queue,
                (
                    tentative_g + weight * estimate,
                    estimate,
                    tentative_g,
                    next(counter),
                    target_key,
                ),
            )

    if selected_key is None or selected_goal_index is None:
        reason = "maximum_state_budget_exhausted" if expanded >= maximum_states else "open_set_exhausted"
        return GlobalSE2RoutePlan(
            "no_route",
            (),
            (),
            (),
            None,
            expanded,
            len(pose_by_key),
            None,
            reason,
            {
                "planner_version": GLOBAL_SE2_ROUTE_VERSION,
                "start_state": list(start_key),
                "map_epoch": int(map_epoch),
                "goal_count": len(goals),
                "blocked_transition_count": len(blocked),
                "maximum_states": int(maximum_states),
                "heuristic_weight": weight,
            },
        )

    reversed_keys = [selected_key]
    reversed_actions: list[str] = []
    cursor = selected_key
    while cursor != start_key:
        previous, action = parent[cursor]
        reversed_actions.append(action)
        reversed_keys.append(previous)
        cursor = previous
    state_keys = tuple(reversed(reversed_keys))
    actions = tuple(reversed(reversed_actions))
    poses = tuple(pose_by_key[key] for key in state_keys)
    selected_goal = goals[selected_goal_index]
    final_position_error, final_yaw_error = _goal_error(poses[-1], selected_goal)
    final_interaction_distance = (
        None
        if selected_goal.interaction_target_xy is None
        else math.hypot(
            float(poses[-1].x) - selected_goal.interaction_target_xy[0],
            float(poses[-1].y) - selected_goal.interaction_target_xy[1],
        )
    )
    return GlobalSE2RoutePlan(
        "planned",
        actions,
        poses,
        state_keys,
        selected_goal_index,
        expanded,
        len(pose_by_key),
        float(g_score[selected_key]),
        "goal_reached",
        {
            "planner_version": GLOBAL_SE2_ROUTE_VERSION,
            "start_state": list(start_key),
            "final_state": list(selected_key),
            "map_epoch": int(map_epoch),
            "goal_count": len(goals),
            "selected_goal_index": int(selected_goal_index),
            "selected_goal_name": selected_goal.name,
            "action_count": len(actions),
            "move_action_count": sum(action.startswith("move_") for action in actions),
            "turn_action_count": sum(action.startswith("turn_") for action in actions),
            "expanded_states": expanded,
            "generated_states": len(pose_by_key),
            "blocked_transition_count": len(blocked),
            "final_position_error_m": float(final_position_error),
            "final_yaw_error_deg": math.degrees(final_yaw_error),
            "final_interaction_distance_m": final_interaction_distance,
            "move_distance_m": move_distance,
            "turn_angle_deg": turn_degrees,
            "xy_quantum_m": quantum,
            "heading_bin_count": int(heading_bin_count),
            "heuristic_weight": weight,
        },
    )


def interaction_station_goals(
    *,
    station_cells: Sequence[Sequence[int]],
    station_cell_to_world: Callable[[Sequence[int]], Sequence[float]],
    target_xy: Sequence[float],
    target_aabb_xy: Sequence[float] | None,
    pose_z: float,
    map_resolution_m: float,
) -> tuple[GlobalSE2Goal, ...]:
    """Adapt prefiltered interaction cells to exact public-action goals."""

    if not callable(station_cell_to_world):
        raise TypeError("station_cell_to_world must be callable")
    if len(target_xy) != 2:
        raise ValueError("target_xy must have two coordinates")
    target = (_finite(target_xy[0], "target_x"), _finite(target_xy[1], "target_y"))
    z = _finite(pose_z, "pose_z")
    cell_half_width_m = _positive(map_resolution_m, "map_resolution_m") / 2.0
    goals: list[GlobalSE2Goal] = []
    seen: set[tuple[int, int]] = set()
    for value in station_cells:
        if (
            isinstance(value, (str, bytes))
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise ValueError("station cell must contain two integers")
        cell = (int(value[0]), int(value[1]))
        if min(cell) < 0:
            raise ValueError("station cell must be non-negative")
        if cell in seen:
            continue
        seen.add(cell)
        world = tuple(float(item) for item in station_cell_to_world(cell))
        if len(world) != 2 or not all(math.isfinite(item) for item in world):
            raise ValueError("station_cell_to_world returned an invalid point")
        interaction_target, _ = nearest_interaction_target_xy(
            world,
            target,
            target_aabb_xy=target_aabb_xy,
        )
        yaw_bin = global_se2_heading_bin(facing_yaw(world, interaction_target))
        quantized_yaw = normalize_angle(
            yaw_bin * (2.0 * math.pi / GLOBAL_SE2_HEADING_BIN_COUNT)
        )
        goals.append(
            GlobalSE2Goal(
                x=world[0],
                y=world[1],
                yaw_rad=quantized_yaw,
                position_tolerance_m=cell_half_width_m,
                name=f"station_{cell[0]}_{cell[1]}",
                official_station_cell=cell,
                allowed_yaw_bins=(yaw_bin,),
            )
        )
    return tuple(goals)


@dataclass(slots=True)
class OrientedFootprintGridGuard:
    """Cheap static oriented-footprint and swept-edge guard.

    ``open_grid`` is the current run-local official floor map *before* the
    isotropic robot erosion.  The rectangle represents the actual R1Pro base
    footprint plus a small explicit margin; yaw therefore changes feasibility.
    """

    open_grid: np.ndarray
    world_to_cell: Callable[[Sequence[float]], tuple[int, int]] = field(repr=False)
    half_forward_m: float = 0.318
    half_left_m: float = 0.338
    safety_margin_m: float = 0.03
    translation_sample_m: float = 0.05
    rotation_sample_deg: float = 5.0
    _pose_cache: dict[tuple[int, int, int], bool] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        grid = np.asarray(self.open_grid)
        if grid.ndim != 2 or grid.size == 0:
            raise ValueError("open_grid must be a non-empty 2-D array")
        self.open_grid = np.where(grid != 0, np.uint8(255), np.uint8(0))
        if not callable(self.world_to_cell):
            raise TypeError("world_to_cell must be callable")
        for name in (
            "half_forward_m",
            "half_left_m",
            "translation_sample_m",
            "rotation_sample_deg",
        ):
            setattr(self, name, _positive(getattr(self, name), name))
        self.safety_margin_m = _finite(self.safety_margin_m, "safety_margin_m")
        if self.safety_margin_m < 0.0:
            raise ValueError("safety_margin_m must be non-negative")
        if self.rotation_sample_deg > 45.0:
            raise ValueError("rotation_sample_deg is too coarse")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.open_grid.shape[0]), int(self.open_grid.shape[1])

    def cell(self, pose: PlanarPose) -> tuple[int, int]:
        row, col = self.world_to_cell((float(pose.x), float(pose.y)))
        return int(row), int(col)

    def _cache_key(self, pose: PlanarPose) -> tuple[int, int, int]:
        return (
            int(round(float(pose.x) / 0.025)),
            int(round(float(pose.y) / 0.025)),
            int(round(math.degrees(normalize_angle(float(pose.yaw))) / self.rotation_sample_deg)),
        )

    def pose_open(self, pose: PlanarPose) -> bool:
        if not isinstance(pose, PlanarPose):
            raise TypeError("footprint pose must be PlanarPose")
        cache_key = self._cache_key(pose)
        cached = self._pose_cache.get(cache_key)
        if cached is not None:
            return cached
        forward = self.half_forward_m + self.safety_margin_m
        left = self.half_left_m + self.safety_margin_m
        cosine = math.cos(float(pose.yaw))
        sine = math.sin(float(pose.yaw))
        corners_world = []
        for local_forward, local_left in (
            (forward, left),
            (forward, -left),
            (-forward, -left),
            (-forward, left),
        ):
            corners_world.append(
                (
                    float(pose.x) + cosine * local_forward - sine * local_left,
                    float(pose.y) + sine * local_forward + cosine * local_left,
                )
            )
        corners_cell = [
            tuple(int(value) for value in self.world_to_cell(point))
            for point in corners_world
        ]
        rows = [cell[0] for cell in corners_cell]
        cols = [cell[1] for cell in corners_cell]
        row_min, row_max = min(rows), max(rows)
        col_min, col_max = min(cols), max(cols)
        height, width = self.shape
        if row_min < 0 or col_min < 0 or row_max >= height or col_max >= width:
            self._pose_cache[cache_key] = False
            return False
        mask = np.zeros((row_max - row_min + 1, col_max - col_min + 1), dtype=np.uint8)
        polygon = np.asarray(
            [
                [col - col_min, row - row_min]
                for row, col in corners_cell
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, polygon, 1)
        region = self.open_grid[row_min : row_max + 1, col_min : col_max + 1]
        result = bool(np.all(region[mask != 0] != 0))
        self._pose_cache[cache_key] = result
        return result

    def edge_open(self, start: PlanarPose, target: PlanarPose) -> bool:
        if not isinstance(start, PlanarPose) or not isinstance(target, PlanarPose):
            raise TypeError("footprint edge endpoints must be PlanarPose")
        translation = math.hypot(float(target.x) - float(start.x), float(target.y) - float(start.y))
        yaw_delta = normalize_angle(float(target.yaw) - float(start.yaw))
        sample_count = max(
            1,
            int(math.ceil(translation / self.translation_sample_m)),
            int(math.ceil(abs(math.degrees(yaw_delta)) / self.rotation_sample_deg)),
        )
        for index in range(sample_count + 1):
            fraction = index / sample_count
            pose = PlanarPose(
                float(start.x) + fraction * (float(target.x) - float(start.x)),
                float(start.y) + fraction * (float(target.y) - float(start.y)),
                float(start.z) + fraction * (float(target.z) - float(start.z)),
                normalize_angle(float(start.yaw) + fraction * yaw_delta),
            )
            if not self.pose_open(pose):
                return False
        return True


@dataclass(slots=True)
class SweptCenterGridGuard:
    """Guard public translations on an already robot-eroded official map.

    OmniGibson's ``TraversableMap._erode_trav_map`` is the canonical static
    footprint approximation used by the official shortest-path API.  Once
    that erosion has been applied, rasterizing the robot's reset-pose AABB a
    second time is both redundant and unsound: that AABB includes articulated
    links and can reject interaction stations that the same robot has already
    occupied successfully.  This guard therefore checks the base centre at
    <=5 cm intervals.  Heading remains part of the global search state, while
    the live CuRobo edge provider is authoritative for yaw-dependent and
    full-body collision.
    """

    open_grid: np.ndarray
    world_to_cell: Callable[[Sequence[float]], tuple[int, int]] = field(
        repr=False
    )
    translation_sample_m: float = 0.05

    def __post_init__(self) -> None:
        grid = np.asarray(self.open_grid)
        if grid.ndim != 2 or grid.size == 0:
            raise ValueError("open_grid must be a non-empty 2-D array")
        self.open_grid = np.where(grid != 0, np.uint8(255), np.uint8(0))
        if not callable(self.world_to_cell):
            raise TypeError("world_to_cell must be callable")
        self.translation_sample_m = _positive(
            self.translation_sample_m,
            "translation_sample_m",
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.open_grid.shape[0]), int(self.open_grid.shape[1])

    def cell(self, pose: PlanarPose) -> tuple[int, int]:
        if not isinstance(pose, PlanarPose):
            raise TypeError("center-guard pose must be PlanarPose")
        row, col = self.world_to_cell((float(pose.x), float(pose.y)))
        return int(row), int(col)

    def pose_open(self, pose: PlanarPose) -> bool:
        row, col = self.cell(pose)
        height, width = self.shape
        return bool(
            0 <= row < height
            and 0 <= col < width
            and self.open_grid[row, col] != 0
        )

    def edge_open(self, start: PlanarPose, target: PlanarPose) -> bool:
        if not isinstance(start, PlanarPose) or not isinstance(target, PlanarPose):
            raise TypeError("center-guard edge endpoints must be PlanarPose")
        translation = math.hypot(
            float(target.x) - float(start.x),
            float(target.y) - float(start.y),
        )
        sample_count = max(
            1,
            int(math.ceil(translation / self.translation_sample_m)),
        )
        for index in range(sample_count + 1):
            fraction = index / sample_count
            pose = PlanarPose(
                float(start.x)
                + fraction * (float(target.x) - float(start.x)),
                float(start.y)
                + fraction * (float(target.y) - float(start.y)),
                float(start.z)
                + fraction * (float(target.z) - float(start.z)),
                normalize_angle(
                    float(start.yaw)
                    + fraction
                    * normalize_angle(float(target.yaw) - float(start.yaw))
                ),
            )
            if not self.pose_open(pose):
                return False
        return True


def radial_interaction_goals(
    *,
    target_xy: Sequence[float],
    minimum_distance_m: float,
    maximum_distance_m: float,
    pose_z: float,
    pose_is_open: PoseGuard,
    angular_samples: int = 64,
    radial_step_m: float = 0.05,
) -> tuple[GlobalSE2Goal, ...]:
    """Enumerate target-facing interaction goals without choosing one early."""

    if len(target_xy) != 2:
        raise ValueError("target_xy must have two coordinates")
    target = (_finite(target_xy[0], "target_x"), _finite(target_xy[1], "target_y"))
    minimum = _positive(minimum_distance_m, "minimum_distance_m")
    maximum = _positive(maximum_distance_m, "maximum_distance_m")
    if minimum > maximum:
        raise ValueError("minimum interaction distance exceeds maximum")
    if isinstance(angular_samples, bool) or not isinstance(angular_samples, int) or angular_samples < 8:
        raise ValueError("angular_samples must be an integer >= 8")
    radial_step = _positive(radial_step_m, "radial_step_m")
    z = _finite(pose_z, "pose_z")
    radii: list[float] = []
    radius = maximum
    while radius >= minimum - 1.0e-9:
        radii.append(max(minimum, radius))
        radius -= radial_step
    result: list[GlobalSE2Goal] = []
    seen: set[tuple[int, int, int]] = set()
    for radius in radii:
        for index in range(angular_samples):
            angle = 2.0 * math.pi * index / angular_samples
            x = target[0] + radius * math.cos(angle)
            y = target[1] + radius * math.sin(angle)
            yaw = normalize_angle(math.atan2(target[1] - y, target[0] - x))
            yaw_bin = global_se2_heading_bin(yaw)
            quantized_yaw = normalize_angle(
                yaw_bin * (2.0 * math.pi / GLOBAL_SE2_HEADING_BIN_COUNT)
            )
            pose = PlanarPose(x, y, z, quantized_yaw)
            if not bool(pose_is_open(pose)):
                continue
            key = (
                int(round(x / GLOBAL_SE2_XY_QUANTUM_M)),
                int(round(y / GLOBAL_SE2_XY_QUANTUM_M)),
                yaw_bin,
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(
                GlobalSE2Goal(
                    x=x,
                    y=y,
                    yaw_rad=quantized_yaw,
                    position_tolerance_m=0.12,
                    yaw_tolerance_deg=7.6,
                    name=f"ring_{radius:.2f}_angle_{index:02d}",
                    interaction_target_xy=target,
                    minimum_interaction_distance_m=minimum,
                    maximum_interaction_distance_m=maximum,
                )
            )
    return tuple(result)
