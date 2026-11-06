"""Bounded closed-loop navigation over public Action20 base actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from behavior_hl.embodied.evaluator_bridge.private_global_se2_route import (
    GlobalSE2Goal,
    GlobalSE2RoutePlan,
    GlobalSE2StateKey,
    GlobalSE2TransitionKey,
    global_se2_state_key,
)
from behavior_hl.navigation import PlanarPose


RoutePlanner = Callable[
    [
        PlanarPose,
        Sequence[GlobalSE2Goal],
        frozenset[GlobalSE2TransitionKey],
        int,
    ],
    GlobalSE2RoutePlan,
]
ActionExecutor = Callable[[str], tuple[str, PlanarPose, bool]]
StationValidator = Callable[[int, GlobalSE2Goal, PlanarPose], bool]

_ORACLE_ACTION_COUNT_DIAGNOSTIC = 150


class NavigationProgress:
    """Measured-state coverage, independent of planner calls and goal labels."""

    def __init__(self, start_key):
        self.seen = {start_key}
        self.revisited_actions = 0
        self.maximum_revisited_actions = 0

    def observe(self, key):
        self.revisited_actions = self.revisited_actions + 1 if key in self.seen else 0
        self.seen.add(key)
        self.maximum_revisited_actions = max(self.maximum_revisited_actions, self.revisited_actions)


@dataclass(frozen=True, slots=True)
class NavigationLegResult:
    status: str
    reason: str
    failure_category: str | None
    final_pose: PlanarPose
    selected_station_index: int | None
    action_count: int
    planner_call_count: int
    planner_expanded_state_count: int
    planner_generated_state_count: int
    blocked_transition_count: int
    blocked_transitions: tuple[GlobalSE2TransitionKey, ...]
    rejected_station_indices: tuple[int, ...]
    oracle_action_count_over_150: bool
    measured_state_count: int = 0
    maximum_revisited_actions: int = 0
    no_progress_station_switches: int = 0


class NavigationLegController:
    """Own route state, failed edges, station rejection, and termination."""

    def __init__(
        self,
        *,
        goals: Sequence[GlobalSE2Goal],
        cell_resolver: Callable[[PlanarPose], tuple[int, int]],
        route_planner: RoutePlanner,
        execute_action: ActionExecutor,
        validate_station: StationValidator,
        map_epoch: int,
        maximum_actions: int,
        maximum_planner_calls: int,
        maximum_expanded_states: int,
        initial_blocked_transitions: Sequence[GlobalSE2TransitionKey] = (),
        expand_search: Callable[[], Sequence[GlobalSE2Goal] | None] | None = None,
        maximum_revisited_actions: int = 16,
    ) -> None:
        if not goals or any(not isinstance(goal, GlobalSE2Goal) for goal in goals):
            raise ValueError("navigation leg requires at least one valid goal")
        for name, callback in (
            ("cell_resolver", cell_resolver),
            ("route_planner", route_planner),
            ("execute_action", execute_action),
            ("validate_station", validate_station),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        for name, value in (
            ("map_epoch", map_epoch),
            ("maximum_actions", maximum_actions),
            ("maximum_planner_calls", maximum_planner_calls),
            ("maximum_expanded_states", maximum_expanded_states),
            ("maximum_revisited_actions", maximum_revisited_actions),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if map_epoch < 0:
            raise ValueError("map_epoch must be non-negative")
        if (
            maximum_actions <= 0
            or maximum_planner_calls <= 0
            or maximum_expanded_states <= 0
            or maximum_revisited_actions <= 0
        ):
            raise ValueError("navigation leg budgets must be positive")
        self.goals = tuple(goals)
        self.cell_resolver = cell_resolver
        self.route_planner = route_planner
        self.execute_action = execute_action
        self.validate_station = validate_station
        self.map_epoch = map_epoch
        self.maximum_actions = maximum_actions
        self.maximum_planner_calls = maximum_planner_calls
        self.maximum_expanded_states = maximum_expanded_states
        self.initial_blocked_transitions = frozenset(initial_blocked_transitions)
        if expand_search is not None and not callable(expand_search):
            raise TypeError("expand_search must be callable")
        self.expand_search = expand_search
        self.maximum_revisited_actions = maximum_revisited_actions

    def navigate(self, start_pose: PlanarPose) -> NavigationLegResult:
        pose = start_pose
        blocked: set[GlobalSE2TransitionKey] = set(
            self.initial_blocked_transitions
        )
        rejected_stations: set[int] = set()
        visited: set[tuple[int, GlobalSE2StateKey, int]] = set()
        rejection_revision = 0
        action_count = 0
        planner_calls = 0
        planner_expanded_states = 0
        planner_generated_states = 0
        plan: GlobalSE2RoutePlan | None = None
        route_cursor = 0
        selected_station: int | None = None
        progress = NavigationProgress(self._state_key(start_pose))
        no_progress_station_switches = 0

        def done(
            status: str,
            reason: str,
            failure_category: str | None,
        ) -> NavigationLegResult:
            return NavigationLegResult(
                status=status,
                reason=reason,
                failure_category=failure_category,
                final_pose=pose,
                selected_station_index=selected_station,
                action_count=action_count,
                planner_call_count=planner_calls,
                planner_expanded_state_count=planner_expanded_states,
                planner_generated_state_count=planner_generated_states,
                blocked_transition_count=len(blocked),
                blocked_transitions=tuple(sorted(blocked)),
                rejected_station_indices=tuple(sorted(rejected_stations)),
                oracle_action_count_over_150=(
                    action_count > _ORACLE_ACTION_COUNT_DIAGNOSTIC
                ),
                measured_state_count=len(progress.seen),
                maximum_revisited_actions=progress.maximum_revisited_actions,
                no_progress_station_switches=no_progress_station_switches,
            )

        while True:
            actual_key = self._state_key(pose)
            if plan is None:
                remaining = tuple(
                    (index, goal)
                    for index, goal in enumerate(self.goals)
                    if index not in rejected_stations
                )
                if not remaining:
                    # A preferred subset may be reachable yet unusable. It is
                    # not the full station frontier until expansion is checked.
                    # Do not expand after resource exhaustion or reset budgets.
                    if (self.expand_search is not None
                            and planner_calls < self.maximum_planner_calls
                            and planner_expanded_states < self.maximum_expanded_states
                            and action_count < self.maximum_actions):
                        expanded = self.expand_search()
                        if expanded is not None:
                            expanded = tuple(expanded)
                            if (expanded[:len(self.goals)] != self.goals
                                    or any(not isinstance(goal, GlobalSE2Goal) for goal in expanded)):
                                return done("planner_contract_error", "search_expansion_changed_station_identity",
                                            "ORACLE_PLANNER_ERROR")
                            if len(expanded) > len(self.goals):
                                self.goals = expanded
                                selected_station = None
                                continue
                            # Enlarging only the route domain cannot resurrect
                            # already rejected goals; avoid a zero-work loop.
                    if no_progress_station_switches:
                        return done('navigation_no_progress', 'active_station_frontier_stalled', 'NAVIGATION_ERROR')
                    return done(
                        "all_stations_rejected",
                        "station_candidates_exhausted",
                        "NAVIGATION_ERROR",
                    )
                if planner_calls >= self.maximum_planner_calls:
                    return done(
                        "controller_safety_budget_exhausted",
                        "planner_call_budget_exhausted",
                        "ORACLE_PLANNER_ERROR",
                    )
                remaining_expansions = (
                    self.maximum_expanded_states - planner_expanded_states
                )
                if remaining_expansions <= 0:
                    return done(
                        "planner_resource_exhausted",
                        "cumulative_expanded_state_budget_exhausted",
                        "ORACLE_PLANNER_ERROR",
                    )
                plan = self.route_planner(
                    pose,
                    tuple(goal for _, goal in remaining),
                    frozenset(blocked),
                    remaining_expansions,
                )
                planner_calls += 1
                if not isinstance(plan, GlobalSE2RoutePlan):
                    return done(
                        "planner_contract_error",
                        "route_planner_returned_invalid_result",
                        "ORACLE_PLANNER_ERROR",
                    )
                if plan.expanded_states > remaining_expansions:
                    return done(
                        "planner_contract_error",
                        "route_planner_exceeded_expanded_state_budget",
                        "ORACLE_PLANNER_ERROR",
                    )
                planner_expanded_states += plan.expanded_states
                planner_generated_states += plan.generated_states
                if plan.status == "no_route":
                    if plan.reason == "open_set_exhausted" and self.expand_search is not None:
                        expanded = self.expand_search()
                        if expanded is not None:
                            expanded = tuple(expanded)
                            # Append-only identities keep rejected station indices
                            # valid. All measured state, failed edges and budgets
                            # stay in this controller; no new leg is created.
                            if (expanded[:len(self.goals)] != self.goals
                                    or any(not isinstance(goal, GlobalSE2Goal) for goal in expanded)):
                                return done("planner_contract_error", "search_expansion_changed_station_identity",
                                            "ORACLE_PLANNER_ERROR")
                            self.goals = expanded
                            plan = None
                            selected_station = None
                            continue
                    resource_exhausted = plan.reason == "maximum_state_budget_exhausted"
                    return done(
                        "planner_resource_exhausted" if resource_exhausted else "no_route",
                        plan.reason,
                        "ORACLE_PLANNER_ERROR" if resource_exhausted else "NAVIGATION_ERROR",
                    )
                if plan.selected_goal_index is None or not 0 <= plan.selected_goal_index < len(remaining):
                    return done(
                        "planner_contract_error",
                        "selected_goal_index_out_of_range",
                        "ORACLE_PLANNER_ERROR",
                    )
                selected_station = remaining[plan.selected_goal_index][0]
                route_cursor = 0
                if not plan.state_keys or plan.state_keys[0] != actual_key:
                    return done(
                        "planner_contract_error",
                        "route_start_state_mismatch",
                        "ORACLE_PLANNER_ERROR",
                    )

            if selected_station is None:
                raise AssertionError("planned navigation route lacks a selected station")
            knowledge_state = (selected_station, actual_key, rejection_revision)
            if knowledge_state in visited:
                rejected_stations.add(selected_station)
                plan = None
                selected_station = None
                continue
            visited.add(knowledge_state)

            if route_cursor == len(plan.actions):
                if self.validate_station(
                    selected_station,
                    self.goals[selected_station],
                    pose,
                ):
                    return done(
                        "arrived",
                        "station_validated",
                        None,
                    )
                rejected_stations.add(selected_station)
                plan = None
                selected_station = None
                continue

            if plan.state_keys[route_cursor] != actual_key:
                plan = None
                selected_station = None
                continue
            if action_count >= self.maximum_actions:
                return done(
                    "controller_safety_budget_exhausted",
                    "public_action_budget_exhausted",
                    "ORACLE_PLANNER_ERROR",
                )
            action = plan.actions[route_cursor]
            transition = (*actual_key, action)
            if transition in blocked:
                return done(
                    "planner_contract_error",
                    "planner_reused_rejected_transition",
                    "ORACLE_PLANNER_ERROR",
                )
            receipt = self.execute_action(action)
            action_count += 1
            if (
                not isinstance(receipt, tuple)
                or len(receipt) != 3
                or receipt[0] not in {"completed", "blocked"}
                or not isinstance(receipt[1], PlanarPose)
                or not isinstance(receipt[2], bool)
            ):
                return done(
                    "execution_state_desync",
                    "invalid_action20_execution_receipt",
                    "ACTION20_PROTOCOL_ERROR",
                )
            status, measured_after, translation_clipped = receipt
            pose = measured_after
            if status == "blocked":
                if self._state_key(pose) != actual_key:
                    return done(
                        "execution_state_desync",
                        "blocked_action_changed_measured_state",
                        "ACTION20_PROTOCOL_ERROR",
                    )
                blocked.add(transition)
                rejection_revision += 1
            progress.observe(self._state_key(pose))
            if (status == 'completed' and not translation_clipped
                    and route_cursor + 1 == len(plan.actions)
                    and self._state_key(pose) == plan.state_keys[route_cursor + 1]
                    and progress.revisited_actions >= self.maximum_revisited_actions):
                if self.validate_station(selected_station, self.goals[selected_station], pose):
                    return done('arrived', 'station_validated', None)
            if progress.revisited_actions >= self.maximum_revisited_actions:
                if no_progress_station_switches:
                    return done('navigation_no_progress', 'repeated_measured_states_after_station_switch',
                                'NAVIGATION_ERROR')
                # Reject only this station under this leg's fixed context.
                # Replanning/new blocked edges do not mint a fresh progress budget.
                rejected_stations.add(selected_station)
                no_progress_station_switches += 1
                progress.revisited_actions = 0
                plan = None
                selected_station = None
                continue
            if status == "blocked":
                plan = None
                selected_station = None
                continue
            expected_key = plan.state_keys[route_cursor + 1]
            measured_key = self._state_key(pose)
            if translation_clipped or measured_key != expected_key:
                if measured_key == actual_key:
                    return done(
                        "execution_state_desync",
                        "completed_action_made_no_state_progress",
                        "ACTION20_PROTOCOL_ERROR",
                    )
                plan = None
                selected_station = None
                continue
            route_cursor += 1

    def _state_key(self, pose: PlanarPose) -> GlobalSE2StateKey:
        return global_se2_state_key(
            pose=pose,
            official_cell=self.cell_resolver(pose),
            map_epoch=self.map_epoch,
        )
