"""Adapter from the public high-level protocol to OmniGibson action primitives.

OmniGibson imports are intentionally lazy. Importing this module therefore does
not launch Isaac Sim and the pure protocol tests can run in an ordinary Python
environment.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Literal

from behavior_hl.actions import ActionName, EGO_NAVIGATION_ACTIONS, HighLevelAction
from behavior_hl.navigation import NavigationBlocked, PlanarNavigator, PlanarPose
from behavior_hl.scoring import q_score


BackendMode = Literal["symbolic", "physics"]


class BackendActionError(RuntimeError):
    """A structured high-level execution failure."""

    def __init__(
        self,
        reason: str,
        message: str,
        details: dict[str, Any] | None = None,
        *,
        consumed_steps: int = 0,
    ) -> None:
        if consumed_steps < 0:
            raise ValueError("consumed_steps must be non-negative")
        self.reason = reason
        self.details = details or {}
        self.consumed_steps = consumed_steps
        self.details.setdefault("consumed_steps", consumed_steps)
        super().__init__(f"{reason}: {message}")


class OmniGibsonBackend:
    """Execute high-level actions against one loaded ``BehaviorTask``.

    ``symbolic`` uses OmniGibson's official symbolic primitives for rapid
    high-level-method experiments. ``physics`` uses the WIP R1/R1Pro/Tiago
    starter primitives and CuRobo. Both modes use the original task predicates
    and the original Q-score formula.
    """

    def __init__(
        self,
        env: Any,
        *,
        mode: BackendMode = "symbolic",
        move_distance: float = 0.25,
        turn_angle_degrees: float = 15.0,
        max_reach_xy: float = 1.5,
        primitive_attempts: int = 3,
        max_simulator_steps_per_macro: int = 10_000,
        max_total_simulator_steps: int = 100_000,
    ) -> None:
        if mode not in ("symbolic", "physics"):
            raise ValueError("mode must be 'symbolic' or 'physics'")
        if max_reach_xy <= 0:
            raise ValueError("max_reach_xy must be positive")
        if primitive_attempts <= 0:
            raise ValueError("primitive_attempts must be positive")
        if max_simulator_steps_per_macro <= 0:
            raise ValueError("max_simulator_steps_per_macro must be positive")
        if max_total_simulator_steps <= 0:
            raise ValueError("max_total_simulator_steps must be positive")

        # These imports must happen only after OmniGibson / Isaac Sim has been
        # configured and launched by the runner.
        import torch as th

        import omnigibson.utils.transform_utils as transform_utils
        from omnigibson import object_states

        self.env = env
        self.robot = env.robots[0]
        self.mode = mode
        self.max_reach_xy = float(max_reach_xy)
        self.primitive_attempts = primitive_attempts
        self.max_simulator_steps_per_macro = max_simulator_steps_per_macro
        self.max_total_simulator_steps = max_total_simulator_steps
        self._th = th
        self._T = transform_utils
        self._object_states = object_states
        self._last_info: dict[str, Any] = {}
        self._total_simulator_steps = 0
        self._active_macro_start_steps: int | None = None

        if mode == "symbolic":
            from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
                SymbolicSemanticActionPrimitives,
                SymbolicSemanticActionPrimitiveSet,
            )

            self._primitive_enum = SymbolicSemanticActionPrimitiveSet
            self._primitives = SymbolicSemanticActionPrimitives(env, self.robot)
        else:
            from omnigibson.action_primitives.starter_semantic_action_primitives import (
                StarterSemanticActionPrimitives,
                StarterSemanticActionPrimitiveSet,
            )

            self._primitive_enum = StarterSemanticActionPrimitiveSet
            self._primitives = StarterSemanticActionPrimitives(
                env,
                self.robot,
                enable_head_tracking=False,
                task_relevant_objects_only=True,
            )

        self._navigator = PlanarNavigator(
            move_distance=move_distance,
            turn_angle_degrees=turn_angle_degrees,
            guard=self._local_path_is_traversable,
        )
        self._initial_predicate_states = self._predicate_states()

    def addressable_objects(self) -> dict[str, str]:
        """Return BDDL instance ID -> simulator name for task-relevant objects."""

        objects: dict[str, str] = {}
        for instance_id, entity in self.env.task.object_scope.items():
            if entity is None or entity is self.robot:
                continue
            if hasattr(entity, "name"):
                objects[instance_id] = entity.name
        return dict(sorted(objects.items()))

    def current_planar_pose(self) -> PlanarPose:
        """Expose a read-only base pose for the restricted ESI adapter."""

        return self._current_planar_pose()

    def same_floor_motion(self, start: PlanarPose, target: PlanarPose) -> bool:
        current = self._current_planar_pose()
        if not self._poses_match(current, start):
            return False
        if abs(target.z - start.z) > 1e-6:
            return False
        return self._floor_index_for_z(start.z) == self._floor_index_for_z(target.z)

    def guarded_local_motion(self, start: PlanarPose, target: PlanarPose) -> bool:
        """Atomically re-read pose and apply the backend local motion guard."""

        current = self._current_planar_pose()
        return self._poses_match(current, start) and self._local_path_is_traversable(start, target)

    def resolve_object_label(self, label: str) -> str | None:
        """Resolve only exact IDs or unique scope-local names/categories."""

        normalized = self._normalize_label(label)
        matches = []
        for instance_id, entity in self.env.task.object_scope.items():
            if entity is None or entity is self.robot:
                continue
            aliases = {
                self._normalize_label(instance_id),
                self._normalize_label(str(getattr(entity, "name", ""))),
                self._normalize_label(str(getattr(entity, "category", ""))),
            }
            if normalized in aliases:
                matches.append(instance_id)
        unique = sorted(set(matches))
        return unique[0] if len(unique) == 1 else None

    def held_object(self) -> str | None:
        """Return the canonical BDDL instance ID currently held, if any."""

        obj = self._primitives._get_obj_in_hand()
        if obj is None:
            return None
        for instance_id, entity in self.env.task.object_scope.items():
            if entity is obj:
                return instance_id
        raise BackendActionError(
            "held_object_not_in_scope",
            f"held simulator object {obj.name!r} has no BDDL instance ID",
        )

    def success(self) -> bool:
        return any(all(option) for option in self._predicate_states())

    def q_score(self) -> float:
        current = self._predicate_states()
        return q_score(self._initial_predicate_states, current, success=any(all(option) for option in current))

    def goal_status(self) -> dict[str, Any]:
        options = []
        for predicates, values in zip(self.env.task.ground_goal_state_options, self._predicate_states()):
            options.append(
                {
                    "satisfied": [str(predicate) for predicate, value in zip(predicates, values) if value],
                    "unsatisfied": [str(predicate) for predicate, value in zip(predicates, values) if not value],
                }
            )
        return {"options": options}

    def execute(self, action: HighLevelAction) -> tuple[int, dict[str, Any]]:
        before_steps = self._total_simulator_steps
        if self._active_macro_start_steps is not None:
            raise RuntimeError("backend.execute cannot be nested")
        self._active_macro_start_steps = before_steps
        try:
            if action.name in EGO_NAVIGATION_ACTIONS:
                metadata = self._execute_ego_navigation(action.name)
            elif action.name is ActionName.NAVIGATE_TO:
                target = self._resolve_object(action.target)
                metadata = self._navigate_to_object(target)
            else:
                target = self._resolve_object(action.target)
                self._require_reachable(target)
                if action.name is ActionName.PLACE_NEXT_TO:
                    if self.mode != "symbolic":
                        raise BackendActionError(
                            "unsupported_in_physics",
                            "PLACE_NEXT_TO is not present in StarterSemanticActionPrimitives",
                        )
                    support = self._resolve_object(action.support)
                    self._require_reachable(support)
                    metadata = self._place_next_to(target=target, support=support)
                else:
                    if self.mode == "physics" and action.name in (
                        ActionName.OPEN,
                        ActionName.CLOSE,
                        ActionName.TOGGLE_ON,
                        ActionName.TOGGLE_OFF,
                    ):
                        raise BackendActionError(
                            "unsupported_in_physics",
                            "the upstream physical open/close and toggle implementations "
                            "currently raise NotImplementedError",
                        )
                    metadata = self._execute_official_primitive(action.name, target)
        except BackendActionError as exc:
            consumed = self._total_simulator_steps - before_steps
            exc.consumed_steps = max(exc.consumed_steps, consumed)
            exc.details["consumed_steps"] = exc.consumed_steps
            raise
        except NavigationBlocked as exc:
            consumed = self._total_simulator_steps - before_steps
            raise BackendActionError(
                "navigation_blocked",
                str(exc),
                consumed_steps=consumed,
            ) from exc
        except Exception as exc:
            consumed = self._total_simulator_steps - before_steps
            reason = getattr(getattr(exc, "reason", None), "name", "primitive_failed").lower()
            raise BackendActionError(
                reason,
                str(exc),
                consumed_steps=consumed,
            ) from exc
        finally:
            self._active_macro_start_steps = None

        return self._total_simulator_steps - before_steps, metadata

    def _predicate_states(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(
            tuple(bool(predicate.evaluate(self.env.task._evaluate_predicate)) for predicate in option)
            for option in self.env.task.ground_goal_state_options
        )

    def _resolve_object(self, instance_id: str | None) -> Any:
        if instance_id is None:
            raise BackendActionError("missing_target", "an object target is required")
        if instance_id not in self.env.task.object_scope:
            raise BackendActionError(
                "unknown_target",
                f"{instance_id!r} is not in BehaviorTask.object_scope",
                {"allowed_targets": sorted(self.addressable_objects())},
            )
        obj = self.env.task.object_scope[instance_id]
        if obj is None:
            raise BackendActionError("missing_object", f"{instance_id!r} does not currently resolve to an entity")
        if obj is self.robot:
            raise BackendActionError("invalid_target", "the robot cannot be a manipulation target")
        return obj

    def _floor_index_for_z(self, z: float) -> int:
        floor_heights = self.env.scene.trav_map.floor_heights
        if floor_heights is None or len(floor_heights) == 0:
            raise BackendActionError("floor_map_unavailable", "scene has no floor-height metadata")
        return min(
            range(len(floor_heights)),
            key=lambda index: abs(float(floor_heights[index]) - z),
        )

    def _object_floor_index(self, obj: Any) -> int:
        # The AABB bottom identifies the supporting floor more reliably than the
        # origin or center for tall cabinets and objects placed on counters.
        try:
            reference_z = float(obj.aabb_center[2]) - 0.5 * float(obj.aabb_extent[2])
        except (AttributeError, IndexError, RuntimeError, TypeError):
            reference_z = float(obj.get_position_orientation()[0][2])
        return self._floor_index_for_z(reference_z)

    def _require_same_floor(self, obj: Any) -> int:
        robot_z = float(self.robot.get_position_orientation()[0][2])
        robot_floor = self._floor_index_for_z(robot_z)
        object_floor = self._object_floor_index(obj)
        if object_floor != robot_floor:
            raise BackendActionError(
                "different_floor",
                f"target {obj.name!r} is on floor {object_floor}, robot is on floor {robot_floor}",
                {
                    "robot_floor": robot_floor,
                    "target_floor": object_floor,
                    "robot_z": robot_z,
                },
            )
        return robot_floor

    def _require_reachable(self, obj: Any) -> None:
        self._require_same_floor(obj)

        # A same-floor floor object is spatially extended; its object origin is
        # not a meaningful reach distance. Placement sampling validates the pose.
        category = str(getattr(obj, "category", "")).lower()
        if category in {"floor", "ground"}:
            return

        robot_pos = self.robot.get_position_orientation()[0]
        object_pos = obj.get_position_orientation()[0]
        distance = float(self._th.linalg.vector_norm(robot_pos[:2] - object_pos[:2]).item())
        if distance > self.max_reach_xy:
            raise BackendActionError(
                "target_out_of_reach",
                f"target {obj.name!r} is {distance:.2f} m away; navigate first",
                {"distance_xy": distance, "max_reach_xy": self.max_reach_xy},
            )

    def _execute_official_primitive(self, action_name: ActionName, target: Any) -> dict[str, Any]:
        enum_names = {
            ActionName.PICK_UP: "GRASP",
            ActionName.PLACE_INSIDE: "PLACE_INSIDE",
            ActionName.PLACE_ON_TOP: "PLACE_ON_TOP",
            ActionName.OPEN: "OPEN",
            ActionName.CLOSE: "CLOSE",
            ActionName.TOGGLE_ON: "TOGGLE_ON",
            ActionName.TOGGLE_OFF: "TOGGLE_OFF",
        }
        try:
            primitive = getattr(self._primitive_enum, enum_names[action_name])
        except KeyError as exc:
            raise BackendActionError("unsupported_action", f"no primitive mapping for {action_name.value}") from exc

        generator = self._primitives.apply_ref(primitive, target, attempts=self.primitive_attempts)
        steps = self._run_generator(generator)
        return {
            "backend_mode": self.mode,
            "primitive": primitive.name,
            "target": target.name,
            "low_level_steps": steps,
            "held_object": self.held_object(),
        }

    def _current_planar_pose(self) -> PlanarPose:
        position, orientation = self.robot.get_position_orientation()
        yaw = float(self._T.quat2euler(orientation)[2].item())
        return PlanarPose(float(position[0]), float(position[1]), float(position[2]), yaw)

    def _eroded_floor_map(self, floor: int) -> Any:
        trav_map = self.env.scene.trav_map
        if trav_map.floor_map is None or not 0 <= floor < len(trav_map.floor_map):
            raise BackendActionError("floor_map_unavailable", f"no traversability map for floor {floor}")
        return trav_map._erode_trav_map(self._th.clone(trav_map.floor_map[floor]), robot=self.robot)

    def _cell_is_traversable(self, eroded: Any, x: float, y: float) -> bool:
        cell = self.env.scene.trav_map.world_to_map(
            self._th.tensor([x, y], dtype=self._th.float32)
        )
        row, col = int(cell[0]), int(cell[1])
        return (
            0 <= row < eroded.shape[0]
            and 0 <= col < eroded.shape[1]
            and int(eroded[row, col]) == 255
        )

    def _safe_route(
        self,
        start_xy: tuple[float, float],
        target_xy: tuple[float, float],
        floor: int,
    ) -> tuple[list[tuple[float, float]], float] | None:
        source = self._th.tensor(start_xy, dtype=self._th.float32)
        target = self._th.tensor(target_xy, dtype=self._th.float32)
        path, distance = self.env.scene.get_shortest_path(
            floor=floor,
            source_world=source,
            target_world=target,
            entire_path=True,
            robot=self.robot,
        )
        if path is None or distance is None:
            return None

        points = [start_xy]
        points.extend((float(point[0]), float(point[1])) for point in path)
        if math.hypot(points[-1][0] - target_xy[0], points[-1][1] - target_xy[1]) > 1e-6:
            points.append(target_xy)

        deduplicated = [points[0]]
        for point in points[1:]:
            if math.hypot(point[0] - deduplicated[-1][0], point[1] - deduplicated[-1][1]) > 1e-6:
                deduplicated.append(point)

        eroded = self._eroded_floor_map(floor)
        sample_step = max(min(float(self.env.scene.trav_map.map_resolution) * 0.5, 0.05), 0.01)
        for segment_start, segment_end in zip(deduplicated, deduplicated[1:]):
            length = math.hypot(
                segment_end[0] - segment_start[0],
                segment_end[1] - segment_start[1],
            )
            sample_count = max(1, math.ceil(length / sample_step))
            for sample_index in range(sample_count + 1):
                fraction = sample_index / sample_count
                x = segment_start[0] + fraction * (segment_end[0] - segment_start[0])
                y = segment_start[1] + fraction * (segment_end[1] - segment_start[1])
                if not self._cell_is_traversable(eroded, x, y):
                    return None

        geodesic = float(distance.item() if hasattr(distance, "item") else distance)
        return deduplicated[1:], geodesic

    def _local_path_is_traversable(self, start: PlanarPose, target: PlanarPose) -> bool:
        if abs(target.z - start.z) > 1e-6:
            return False
        if self._floor_index_for_z(start.z) != self._floor_index_for_z(target.z):
            return False

        floor = self._floor_index_for_z(start.z)
        displacement = math.hypot(target.x - start.x, target.y - start.y)
        if displacement < 1e-8:
            return self._cell_is_traversable(self._eroded_floor_map(floor), start.x, start.y)

        route = self._safe_route((start.x, start.y), (target.x, target.y), floor)
        if route is None:
            return False
        _, geodesic = route
        # A local ego move must not silently turn into a long detour around a wall.
        tolerance = max(
            displacement * 1.7,
            displacement + 2.0 * float(self.env.scene.trav_map.map_resolution),
        )
        return geodesic <= tolerance

    def _run_symbolic_route(
        self,
        waypoints: list[tuple[float, float]],
        final_yaw: float,
    ) -> int:
        current = self._current_planar_pose()
        if not waypoints:
            return self._run_generator(self._primitives._navigate_to_pose((current.x, current.y, final_yaw)))

        steps = 0
        previous_x, previous_y = current.x, current.y
        for index, (x, y) in enumerate(waypoints):
            yaw = final_yaw if index == len(waypoints) - 1 else math.atan2(y - previous_y, x - previous_x)
            steps += self._run_generator(self._primitives._navigate_to_pose((x, y, yaw)))
            previous_x, previous_y = x, y
        return steps

    def _execute_ego_navigation(self, action_name: ActionName) -> dict[str, Any]:
        start = self._current_planar_pose()
        target = self._navigator.transition(start, action_name)
        displacement = math.hypot(target.x - start.x, target.y - start.y)
        if self.mode == "symbolic":
            if displacement < 1e-8:
                waypoints: list[tuple[float, float]] = []
            else:
                route = self._safe_route(
                    (start.x, start.y),
                    (target.x, target.y),
                    self._floor_index_for_z(start.z),
                )
                if route is None:
                    raise BackendActionError("navigation_blocked", "ego route failed swept-map validation")
                waypoints, _ = route
            steps = self._run_symbolic_route(waypoints, target.yaw)
        else:
            steps = self._run_generator(self._primitives._navigate_to_pose((target.x, target.y, target.yaw)))

        end = self._current_planar_pose()
        if abs(end.z - start.z) > 1e-3:
            raise BackendActionError(
                "vertical_drift",
                "robot base z changed during a planar action",
                {"start_z": start.z, "end_z": end.z},
            )
        return {
            "backend_mode": self.mode,
            "start_pose": self._pose_dict(start),
            "requested_pose": self._pose_dict(target),
            "end_pose": self._pose_dict(end),
            "low_level_steps": steps,
        }

    def _navigate_to_object(self, target: Any) -> dict[str, Any]:
        floor = self._require_same_floor(target)
        if self.mode == "physics":
            primitive = getattr(self._primitive_enum, "NAVIGATE_TO")
            steps = self._run_generator(
                self._primitives.apply_ref(primitive, target, attempts=self.primitive_attempts)
            )
            return {
                "backend_mode": self.mode,
                "primitive": primitive.name,
                "target": target.name,
                "low_level_steps": steps,
            }

        start = self._current_planar_pose()
        target_position = target.get_position_orientation()[0]
        target_x, target_y = float(target_position[0]), float(target_position[1])
        distance_to_target = math.hypot(target_x - start.x, target_y - start.y)

        if distance_to_target <= self.max_reach_xy:
            chosen_x, chosen_y = start.x, start.y
            waypoints = []
        else:
            candidate = self._nearest_reachable_standoff(target_x, target_y, start, floor)
            if candidate is None:
                raise BackendActionError(
                    "navigation_blocked",
                    f"could not find a traversable standoff pose near {target.name!r}",
                )
            chosen_x, chosen_y, waypoints = candidate

        chosen_yaw = math.atan2(target_y - chosen_y, target_x - chosen_x)
        requested = PlanarPose(chosen_x, chosen_y, start.z, chosen_yaw)
        steps = self._run_symbolic_route(waypoints, chosen_yaw)
        end = self._current_planar_pose()
        if abs(end.z - start.z) > 1e-3:
            raise BackendActionError("vertical_drift", "object navigation changed the robot base floor")
        return {
            "backend_mode": self.mode,
            "target": target.name,
            "start_pose": self._pose_dict(start),
            "requested_pose": self._pose_dict(requested),
            "end_pose": self._pose_dict(end),
            "route_waypoints": len(waypoints),
            "low_level_steps": steps,
        }

    def _nearest_reachable_standoff(
        self,
        target_x: float,
        target_y: float,
        start: PlanarPose,
        floor: int,
    ) -> tuple[float, float, list[tuple[float, float]]] | None:
        best: tuple[float, float, float, list[tuple[float, float]]] | None = None
        for radius in (0.70, 0.90, 1.10, 1.30):
            for index in range(24):
                angle = 2.0 * math.pi * index / 24.0
                x = target_x + radius * math.cos(angle)
                y = target_y + radius * math.sin(angle)
                route = self._safe_route((start.x, start.y), (x, y), floor)
                if route is None:
                    continue
                waypoints, geodesic = route
                if best is None or geodesic < best[0]:
                    best = (geodesic, x, y, waypoints)
        return None if best is None else (best[1], best[2], best[3])

    def _place_next_to(self, *, target: Any, support: Any) -> dict[str, Any]:
        held = self._primitives._get_obj_in_hand()
        if held is None:
            raise BackendActionError("pre_condition_error", "PLACE_NEXT_TO requires an object in hand")

        on_top = self._object_states.OnTop
        next_to = self._object_states.NextTo
        target_position = target.get_position_orientation()[0]
        chosen_pose = None

        # Upstream has no NextTo setter. Sample a physically valid OnTop pose on
        # the supplied support and retain samples near the reference object.
        for _ in range(200):
            pose = self._primitives._sample_pose_with_object_and_predicate(on_top, held, support)
            separation = float(self._th.linalg.vector_norm(pose[0][:2] - target_position[:2]).item())
            if 0.10 <= separation <= 0.75:
                chosen_pose = pose
                break
        if chosen_pose is None:
            raise BackendActionError(
                "sampling_error",
                f"could not sample an OnTop({support.name}) pose near {target.name}",
            )

        self._run_generator(self._primitives._release())
        held.set_position_orientation(*chosen_pose)
        settle_steps = self._run_generator(self._primitives._settle_robot())

        is_supported = bool(held.states[on_top].get_value(support))
        is_next_to = bool(held.states[next_to].get_value(target))
        if not (is_supported and is_next_to):
            raise BackendActionError(
                "post_condition_error",
                "sampled placement did not satisfy both OnTop(support) and NextTo(target)",
                {"on_top": is_supported, "next_to": is_next_to},
            )
        return {
            "backend_mode": self.mode,
            "primitive": "PLACE_NEXT_TO",
            "placed_object": held.name,
            "target": target.name,
            "support": support.name,
            "low_level_steps": settle_steps,
            "held_object": self.held_object(),
        }

    def _ensure_step_budget(self) -> None:
        macro_start = (
            self._total_simulator_steps
            if self._active_macro_start_steps is None
            else self._active_macro_start_steps
        )
        macro_steps = self._total_simulator_steps - macro_start
        if macro_steps >= self.max_simulator_steps_per_macro:
            raise BackendActionError(
                "macro_simulator_step_limit",
                f"macro exceeded {self.max_simulator_steps_per_macro} simulator steps",
            )
        if self._total_simulator_steps >= self.max_total_simulator_steps:
            raise BackendActionError(
                "episode_simulator_step_limit",
                f"episode exceeded {self.max_total_simulator_steps} simulator steps",
            )

    @staticmethod
    def _as_bool(value: Any) -> bool:
        return bool(value.item() if hasattr(value, "item") else value)

    def _run_generator(self, generator: Iterable[Any]) -> int:
        steps = 0
        for low_level_action in generator:
            self._ensure_step_budget()
            _, _, terminated, truncated, info = self.env.step(low_level_action)
            self._last_info = info
            steps += 1
            self._total_simulator_steps += 1

            is_terminated = self._as_bool(terminated)
            is_truncated = self._as_bool(truncated)
            if is_truncated and not is_terminated:
                raise BackendActionError(
                    "environment_truncated",
                    "environment truncated while executing a high-level macro",
                )
            if is_terminated:
                close = getattr(generator, "close", None)
                if callable(close):
                    close()
                break

        if steps == 0:
            # Refresh task termination / goal status after a no-op primitive.
            self._ensure_step_budget()
            _, _, terminated, truncated, info = self.env.step(self._primitives._empty_action())
            self._last_info = info
            steps = 1
            self._total_simulator_steps += 1
            if self._as_bool(truncated) and not self._as_bool(terminated):
                raise BackendActionError(
                    "environment_truncated",
                    "environment truncated during a no-op primitive refresh",
                )
        return steps

    @staticmethod
    def _normalize_label(value: str) -> str:
        return " ".join(value.strip().lower().replace("__", " ").replace("_", " ").split())

    @staticmethod
    def _poses_match(first: PlanarPose, second: PlanarPose, tolerance: float = 1e-4) -> bool:
        return (
            abs(first.x - second.x) <= tolerance
            and abs(first.y - second.y) <= tolerance
            and abs(first.z - second.z) <= tolerance
            and abs(first.yaw - second.yaw) <= tolerance
        )

    @staticmethod
    def _pose_dict(pose: PlanarPose) -> dict[str, float]:
        return {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw}

