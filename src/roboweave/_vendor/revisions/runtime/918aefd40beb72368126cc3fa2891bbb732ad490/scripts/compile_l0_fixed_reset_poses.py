#!/usr/bin/env python3
"""Compile additive fixed reset poses for Original-100 plus Expansion-80 on CPU."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from behavior_hl.embodied.evaluator_bridge.canonical_bddl_grounder import (
    iter_ground_options,
    object_map_from_scope,
    problem_goal,
)
from behavior_hl.embodied.evaluator_bridge.interaction_station_geometry import (
    cell_to_world,
    compute_source_geodesic_distances,
    has_structural_line_of_sight,
    select_interaction_station_cells,
    world_to_cell,
)
from behavior_hl.embodied.evaluator_bridge.l0_reset_pose_overlay import (
    OVERLAY_VERSION,
    choose_fixed_reset_pose_candidate,
)
from behavior_hl.embodied.evaluator_bridge.l0_taskspec_goal_plan import (
    compile_projected_goal_plan,
)
from behavior_hl.embodied.evaluator_bridge.mixed_goal_graph import (
    compile_grounded_mixed_option,
)
from behavior_hl.embodied.runtime.plugins.restricted_kinematic_base import (
    ACTION20_RESTRICTED_KINEMATIC_PUBLIC_TURN_DEG,
    RESTRICTED_KINEMATIC_PUBLIC_MOVE_M,
)
from behavior_hl.embodied.runtime.plugins.torso import (
    EMPTY_TORSO_GRID,
    r1pro_head_link_height_m,
    r1pro_head_optical_pitch_deg,
)
from compile_picking_up_toys_reset_routes import (
    _load_resized_binary_map,
    _load_resized_label_map,
    _official_room_name_to_id,
)
from compile_picking_up_toys_reset_viewpoints import (
    HEAD_FOCAL_LENGTH_MM,
    HEAD_HORIZONTAL_APERTURE_MM,
)


MAP_RESOLUTION_M = 0.1
SOURCE_MAP_RESOLUTION_M = 0.01
ROBOT_PADDING_M = 0.2
MAXIMUM_STATION_DISTANCE_M = 1.26
DESIRED_SOURCE_DISTANCE_M = 0.76
DESIRED_DESTINATION_DISTANCE_M = 0.78
MAXIMUM_LEG_M = 3.0
MAXIMUM_TOTAL_M = 6.0
START_LATTICE_CELLS = 2
STATION_CANDIDATE_LIMIT = 16
HEAD_ASPECT_WIDTH = 1280.0
HEAD_ASPECT_HEIGHT = 720.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"expected JSON objects: {path}")
    return rows


def _plan_operations(plan: Any) -> list[dict[str, Any]]:
    return [
        {
            "node_id": str(node.node_id),
            "action": str(node.action),
            "target": str(node.target),
            "destination": None if node.destination is None else str(node.destination),
        }
        for node in plan.action_nodes
    ]


def _scope_positions(
    *, template: Mapping[str, Any], tro: Mapping[str, Any], bindings: Mapping[str, Any]
) -> dict[str, list[float]]:
    states = template["state"]["registry"]["object_registry"]
    positions: dict[str, list[float]] = {}
    for scope_name, binding in bindings.items():
        if binding.get("binding_kind") == "RUNTIME_ROBOT":
            continue
        state = tro.get(scope_name) or states.get(binding.get("actual_name"))
        position = None if not isinstance(state, Mapping) else state.get("root_link", {}).get("pos")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(f"missing reset position for {scope_name}")
        positions[str(scope_name)] = [float(value) for value in position]
    return positions


def load_original_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = []
    for row in _read_jsonl(path):
        private = row["private"]
        identity = private["source_task_identity"]
        template_path = Path(private["source_artifacts"]["template"]["path"])
        tro_path = Path(private["source_artifacts"]["tro_state"]["path"])
        template, tro = _read_json(template_path), _read_json(tro_path)
        if private["verifier"]["route"] == "official_full":
            expression = problem_goal(Path(private["source_artifacts"]["bddl"]["path"]).read_text(encoding="utf-8"))
            option = next(iter_ground_options(expression, object_map_from_scope(private["scope_bindings"])))
            plan = compile_grounded_mixed_option(option)
        else:
            plan = compile_projected_goal_plan(row, lambda *_: False)
        tasks.append(
            {
                "task_spec_id": row["task_spec_id"],
                "task_name": identity["task_name"],
                "scene_model": identity["scene_model"],
                "public_instruction": row["public"]["instruction"],
                "rooms": list(private["rooms"]),
                "scope_bindings": deepcopy(private["scope_bindings"]),
                "positions": _scope_positions(
                    template=template,
                    tro=tro,
                    bindings=private["scope_bindings"],
                ),
                "operations": _plan_operations(plan),
                "template_path": str(template_path.resolve()),
                "tro_state_path": str(tro_path.resolve()),
                "source_family": "ORIGINAL_100",
                "initial_goal_false": True,
            }
        )
    return tasks


def load_expansion_tasks(root: Path) -> list[dict[str, Any]]:
    tasks = []
    registry_paths = sorted((Path(root) / "benchmark/l0/expansion_v1/official_compatible").glob("*/registry.jsonl"))
    for registry_path in registry_paths:
        catalog_path = registry_path.parent / "runtime_registration/catalog.json"
        catalog = _read_json(catalog_path)
        catalog_by_id = {record["candidate_id"]: record for record in catalog["tasks"]}
        for row in _read_jsonl(registry_path):
            registered = catalog_by_id[row["candidate_id"]]
            contract = _read_json(Path(row["template_contract"]["path"]))
            template_path = Path(registered["template"]["path"])
            tro_path = Path(registered["tro_state"]["path"])
            template, tro = _read_json(template_path), _read_json(tro_path)
            positions = _scope_positions(
                template=template,
                tro=tro,
                bindings=row["scope_bindings"],
            )
            # Authored objects are materialized on their declared staging/support
            # anchors.  Reuse that exact construction relation as the navigation
            # and visibility XY proxy; do not invent a static object pose.
            role_to_scope = row["role_to_scope"]
            for recipe in contract.get("initial_state_recipe", ()):
                subject = role_to_scope.get(recipe.get("subject_role"))
                reference = role_to_scope.get(recipe.get("object_role"))
                if subject in positions and reference in positions:
                    positions[subject] = list(positions[reference])
            plan = compile_grounded_mixed_option(
                tuple(registered["profile"]["canonical_goal_leaves"])
            )
            tasks.append(
                {
                    "task_spec_id": row["candidate_id"],
                    "task_name": registered["activity_name"],
                    "scene_model": row["scene_model"],
                    "public_instruction": row["public_instruction"],
                    "rooms": list(row["rooms"]),
                    "scope_bindings": deepcopy(row["scope_bindings"]),
                    "positions": positions,
                    "operations": _plan_operations(plan),
                    "template_path": str(template_path.resolve()),
                    "tro_state_path": str(tro_path.resolve()),
                    "source_family": "EXPANSION_80",
                    "initial_goal_false": row["initial_goal_false_cpu"] is True,
                }
            )
    return tasks


class SceneMaps:
    def __init__(self, *, scene_model: str, assets_root: Path, room_categories_path: Path, robot_metadata_path: Path):
        layout = Path(assets_root) / "scenes" / scene_model / "layout"
        floor_path = layout / "floor_trav_0.png"
        image = cv2.imread(str(floor_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.ndim != 2 or image.shape[0] != image.shape[1]:
            raise ValueError(f"invalid official traversability map: {floor_path}")
        self.scene_model = scene_model
        self.map_size = int(image.shape[0] * SOURCE_MAP_RESOLUTION_M / MAP_RESOLUTION_M)
        floor = cv2.resize(image, (self.map_size, self.map_size))
        floor[floor < 255] = 0
        bbox = _read_json(robot_metadata_path)["bbox_size"]
        erosion_radius = math.hypot(float(bbox[0]), float(bbox[1])) / 2.0 + ROBOT_PADDING_M
        self.erosion_kernel_pixels = int(math.ceil(erosion_radius / MAP_RESOLUTION_M))
        self.open_grid = cv2.erode(
            floor,
            np.ones((self.erosion_kernel_pixels, self.erosion_kernel_pixels), dtype=np.uint8),
        ) != 0
        self.structural_grid = _load_resized_binary_map(
            layout / "floor_trav_no_obj_0.png", map_size=self.map_size
        )
        self.room_instance_map = _load_resized_label_map(
            layout / "floor_insseg_0.png", map_size=self.map_size
        )
        room_semantic_map = _load_resized_label_map(
            layout / "floor_semseg_0.png", map_size=self.map_size
        )
        categories = Path(room_categories_path).read_text(encoding="utf-8").splitlines()
        self.room_name_to_id = _official_room_name_to_id(
            room_instance_map=self.room_instance_map,
            room_semantic_map=room_semantic_map,
            room_categories=categories,
        )
        _, self.labels = cv2.connectedComponents(self.open_grid.astype(np.uint8), connectivity=4)
        self.map_paths = {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in {
                "floor_trav": floor_path,
                "floor_trav_no_obj": layout / "floor_trav_no_obj_0.png",
                "floor_insseg": layout / "floor_insseg_0.png",
                "floor_semseg": layout / "floor_semseg_0.png",
            }.items()
        }


def _is_floor(scope: str, bindings: Mapping[str, Any]) -> bool:
    binding = bindings.get(scope, {})
    return binding.get("category") in {"floor", "floors"} or str(
        binding.get("synset", "")
    ).startswith("floor.n.")


def _route_visits(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    positions = deepcopy(task["positions"])
    visits = []
    for operation in task["operations"]:
        target, destination = operation["target"], operation["destination"]
        visits.append(
            {
                "node_id": operation["node_id"],
                "action": operation["action"],
                "role": "source",
                "scope": target,
                "xyz": list(positions[target]),
            }
        )
        if destination is not None:
            visits.append(
                {
                    "node_id": operation["node_id"],
                    "action": operation["action"],
                    "role": "destination",
                    "scope": destination,
                    "xyz": list(positions[destination]),
                    "local_floor_support": _is_floor(
                        destination, task["scope_bindings"]
                    ),
                }
            )
        if operation["action"].startswith("place_") and destination is not None:
            positions[target] = list(positions[destination])
    return visits


def _station_selection(
    *,
    maps: SceneMaps,
    visit: Mapping[str, Any],
    component: int,
    distance_grid: np.ndarray | None,
) -> tuple[tuple[int, int] | None, Mapping[str, Any]]:
    room_names = visit["room_names"]
    room_ids = tuple(
        maps.room_name_to_id[name]
        for name in room_names
        if name in maps.room_name_to_id
    )
    desired = (
        DESIRED_DESTINATION_DISTANCE_M
        if visit["role"] == "destination"
        else DESIRED_SOURCE_DISTANCE_M
    )
    selection = select_interaction_station_cells(
        target_xy=visit["xyz"][:2],
        navigation_open_grid=maps.open_grid,
        structural_open_grid=maps.structural_grid,
        component_labels=maps.labels,
        source_component_label=component,
        resolution=MAP_RESOLUTION_M,
        map_size=maps.map_size,
        maximum_center_distance_m=MAXIMUM_STATION_DISTANCE_M,
        room_instance_map=maps.room_instance_map,
        target_room_ids=room_ids or None,
        prefer_same_room=True,
        source_geodesic_distance_grid=distance_grid,
    )
    minimum = max(0.0, desired - MAP_RESOLUTION_M)
    cells = [
        cell
        for cell in selection.cells
        if math.dist(
            cell_to_world(
                cell, resolution=MAP_RESOLUTION_M, map_size=maps.map_size
            ),
            visit["xyz"][:2],
        )
        >= minimum - 1.0e-9
    ]
    return (cells[0] if cells else None), {
        "candidate_cells": [list(cell) for cell in cells[:STATION_CANDIDATE_LIMIT]],
        "candidate_count": len(cells),
        "candidate_limit": STATION_CANDIDATE_LIMIT,
        "candidate_search_complete": len(cells) <= STATION_CANDIDATE_LIMIT,
        "raw_open_connected_count": selection.raw_open_connected_count,
        "same_room_candidate_count": selection.same_room_candidate_count,
        "structural_occlusion_rejection_count": selection.structural_occlusion_rejection_count,
        "source_geodesic_unreachable_rejection_count": selection.source_geodesic_unreachable_rejection_count,
        "target_room_ids": list(selection.target_room_ids),
    }


def _compile_station_chain(task: Mapping[str, Any], maps: SceneMaps, component: int) -> Mapping[str, Any]:
    visits = _route_visits(task)
    for visit in visits:
        binding = task["scope_bindings"][visit["scope"]]
        visit["room_names"] = list(binding.get("in_rooms") or task["rooms"])
    current: tuple[int, int] | None = None
    legs = []
    failures = []
    failure_details = []
    distance_cache: dict[tuple[int, int], np.ndarray] = {}
    for index, visit in enumerate(visits):
        if visit.get("local_floor_support") and current is not None:
            station, audit = current, {"local_floor_support_reuses_source_station": True}
        else:
            distances = None
            if current is not None:
                distances = distance_cache.setdefault(
                    current,
                    compute_source_geodesic_distances(maps.open_grid, current, resolution=MAP_RESOLUTION_M),
                )
            station, audit = _station_selection(
                maps=maps,
                visit=visit,
                component=component,
                distance_grid=distances,
            )
        if station is None:
            failures.append("NO_INTERACTION_STATION")
            failure_details.append(
                {
                    "code": "NO_INTERACTION_STATION",
                    "scope_name": visit["scope"],
                    "instance_id": task["scope_bindings"][visit["scope"]]["actual_name"],
                    "role": visit["role"],
                    "station_audit": audit,
                }
            )
            break
        distance = (
            None
            if current is None
            else 0.0
            if station == current
            else float(distance_cache[current][station])
        )
        legs.append(
            {
                "index": index,
                "node_id": visit["node_id"],
                "action": visit["action"],
                "role": visit["role"],
                "scope_name": visit["scope"],
                "instance_id": task["scope_bindings"][visit["scope"]]["actual_name"],
                "target_xyz": visit["xyz"],
                "station_cell": list(station),
                "station_xy": list(cell_to_world(station, resolution=MAP_RESOLUTION_M, map_size=maps.map_size)),
                "path_length_m": distance,
                "station_audit": audit,
            }
        )
        current = station
    return {
        "status": "PASS" if len(legs) == len(visits) else "FAIL",
        "component": component,
        "legs": legs,
        "failures": failures,
        "failure_details": failure_details,
    }


def _egress_world_directions(
    maps: SceneMaps, cell: tuple[int, int]
) -> list[tuple[int, int]]:
    start = cell_to_world(cell, resolution=MAP_RESOLUTION_M, map_size=maps.map_size)
    result = []
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        end = world_to_cell(
            (
                start[0] + dx * RESTRICTED_KINEMATIC_PUBLIC_MOVE_M,
                start[1] + dy * RESTRICTED_KINEMATIC_PUBLIC_MOVE_M,
            ),
            resolution=MAP_RESOLUTION_M,
            map_size=maps.map_size,
        )
        if (
            0 <= end[0] < maps.map_size
            and 0 <= end[1] < maps.map_size
            and has_structural_line_of_sight(maps.open_grid, cell, end)
        ):
            result.append((dx, dy))
    return result


def _wrapped_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _visibility_at_start(
    *, task: Mapping[str, Any], maps: SceneMaps, cell: tuple[int, int],
    fixed_yaw_deg: float | None = None,
) -> Mapping[str, Any]:
    source_scopes = {operation["target"] for operation in task["operations"]}
    destination_scopes = {
        operation["destination"]
        for operation in task["operations"]
        if operation["destination"] is not None
        and not _is_floor(operation["destination"], task["scope_bindings"])
    }
    visibility_roles = task.get("visibility_roles")
    relevant = sorted(visibility_roles if visibility_roles is not None else source_scopes | destination_scopes)
    qpos = EMPTY_TORSO_GRID[("neutral", "center")]
    camera_z = 0.005 + r1pro_head_link_height_m(qpos)
    camera_pitch = r1pro_head_optical_pitch_deg(qpos)
    horizontal_half = math.degrees(
        math.atan(HEAD_HORIZONTAL_APERTURE_MM / (2.0 * HEAD_FOCAL_LENGTH_MM))
    )
    vertical_aperture = (
        HEAD_HORIZONTAL_APERTURE_MM * HEAD_ASPECT_HEIGHT / HEAD_ASPECT_WIDTH
    )
    vertical_half = math.degrees(
        math.atan(vertical_aperture / (2.0 * HEAD_FOCAL_LENGTH_MM))
    )
    xy = cell_to_world(cell, resolution=MAP_RESOLUTION_M, map_size=maps.map_size)
    sightlines = []
    for scope in relevant:
        xyz = task["positions"][scope]
        target_cell = world_to_cell(
            xyz[:2], resolution=MAP_RESOLUTION_M, map_size=maps.map_size
        )
        clear = has_structural_line_of_sight(
            maps.structural_grid,
            cell,
            target_cell,
            allow_blocked_target_endpoint=True,
        )
        dx, dy, dz = xyz[0] - xy[0], xyz[1] - xy[1], xyz[2] - camera_z
        bearing = math.degrees(math.atan2(dy, dx))
        pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
        sightlines.append(
            {
                "scope": scope,
                "instance_id": task["scope_bindings"][scope]["actual_name"],
                "roles": sorted(visibility_roles[scope]) if visibility_roles is not None else sorted(
                    ({"source"} if scope in source_scopes else set())
                    | ({"destination"} if scope in destination_scopes else set())
                ),
                "bearing_deg": bearing,
                "pitch_deg": pitch,
                "structural_los": clear,
                "vertical_in_frame": abs(pitch - camera_pitch) <= vertical_half + 1.0e-9,
            }
        )
    yaw_candidates = {0.0}
    step = ACTION20_RESTRICTED_KINEMATIC_PUBLIC_TURN_DEG
    for sightline in sightlines:
        for value in (
            sightline["bearing_deg"],
            sightline["bearing_deg"] - horizontal_half,
            sightline["bearing_deg"] + horizontal_half,
        ):
            yaw_candidates.add(_wrapped_degrees(round(value / step) * step))
    options = []
    if fixed_yaw_deg is not None:
        yaw_candidates = {float(fixed_yaw_deg)}
    for yaw in sorted(yaw_candidates):
        visible = [
            line
            for line in sightlines
            if line["structural_los"]
            and line["vertical_in_frame"]
            and abs(_wrapped_degrees(line["bearing_deg"] - yaw)) <= horizontal_half + 1.0e-9
        ]
        roles = sorted({role for line in visible for role in line["roles"]})
        options.append(( -len(visible), -len(roles), abs(yaw), yaw, visible, roles))
    _, _, _, yaw, visible, roles = min(options)
    return {
        "yaw_deg": float(yaw),
        "visible_relevant_instance_count": len(visible),
        "visible_source_count": sum("source" in line["roles"] for line in visible),
        "visible_destination_count": sum("destination" in line["roles"] for line in visible),
        "visible_foil_count": sum("foil" in line["roles"] for line in visible),
        "visible_landmark_count": sum("landmark" in line["roles"] for line in visible),
        "visible_role_kind_count": len(roles),
        "predicted_visible_instance_ids": [line["instance_id"] for line in visible],
        "predicted_visible_scope_names": [line["scope"] for line in visible],
        "camera_pitch_deg": camera_pitch,
        "camera_height_m": camera_z,
        "horizontal_fov_deg": 2.0 * horizontal_half,
        "vertical_fov_deg": 2.0 * vertical_half,
    }


def _candidate_records(task: Mapping[str, Any], maps: SceneMaps, chain: Mapping[str, Any], *, route_only: bool = False) -> list[dict[str, Any]]:
    if chain["status"] != "PASS":
        return []
    first = tuple(chain["legs"][0]["station_cell"])
    distances = compute_source_geodesic_distances(maps.open_grid, first, resolution=MAP_RESOLUTION_M)
    suffix_lengths = [float(leg["path_length_m"] or 0.0) for leg in chain["legs"][1:]]
    suffix = sum(suffix_lengths)
    component = int(chain["component"])
    raw_cells = np.argwhere(maps.labels == component)
    cells = {
        (int(row), int(col))
        for row, col in raw_cells
        if int(row) % START_LATTICE_CELLS == 0 and int(col) % START_LATTICE_CELLS == 0
    }
    cells.add(first)
    candidates = []
    for cell in sorted(cells):
        first_leg = float(distances[cell])
        if not math.isfinite(first_leg):
            continue
        egress = _egress_world_directions(maps, cell)
        if not egress:
            continue
        leg_lengths = [first_leg, *suffix_lengths]
        total = first_leg + suffix
        candidates.append(
            {
                "cell": list(cell),
                "world_xy": list(cell_to_world(cell, resolution=MAP_RESOLUTION_M, map_size=maps.map_size)),
                "yaw_deg": 0.0,
                "egress_world_directions": [list(value) for value in egress],
                "all_required_routes_exist": True,
                "under_l0_route_target": bool(
                    max(leg_lengths, default=0.0) <= MAXIMUM_LEG_M + 1.0e-9
                    and total <= MAXIMUM_TOTAL_M + 1.0e-9
                    and len(task["rooms"]) <= 2
                ),
                "route_leg_lengths_m": leg_lengths,
                "total_path_length_m": total,
                "initial_turn_look_action_count": 0,
                "stable_coordinate_tiebreak": [cell[0], cell[1], 0.0],
            }
        )
    if route_only:
        return candidates
    eligible = [candidate for candidate in candidates if candidate["under_l0_route_target"]]
    if not eligible and candidates:
        eligible = [
            min(
                candidates,
                key=lambda candidate: (
                    candidate["total_path_length_m"],
                    candidate["cell"],
                ),
            )
        ]
    for candidate in eligible:
        visible = _visibility_at_start(
            task=task, maps=maps, cell=tuple(candidate["cell"])
        )
        candidate.update(visible)
        candidate["stable_coordinate_tiebreak"][-1] = visible["yaw_deg"]
    candidates = eligible
    return candidates


def _partial_route_report(
    task: Mapping[str, Any],
    maps: SceneMaps,
    cells: Sequence[tuple[int, int]],
) -> Mapping[str, Any]:
    room_ids = [
        maps.room_name_to_id[name]
        for name in task["rooms"]
        if name in maps.room_name_to_id
    ]
    if len(room_ids) != len(set(task["rooms"])):
        return {
            "status": "UNAVAILABLE",
            "failure_reasons": ["TASK_ROOM_NOT_IN_OFFICIAL_SEGMENTATION"],
        }
    partial = maps.open_grid & np.isin(maps.room_instance_map, room_ids)
    lengths = []
    for source, target in zip(cells, cells[1:]):
        if not bool(partial[source]) or not bool(partial[target]):
            return {
                "status": "FAIL",
                "route_exists": False,
                "leg_lengths_m": lengths,
                "failure_reasons": ["PARTIAL_ROOM_MASK_ENDPOINT_CLOSED"],
            }
        distance = float(
            compute_source_geodesic_distances(
                partial, source, resolution=MAP_RESOLUTION_M
            )[target]
        )
        if not math.isfinite(distance):
            return {
                "status": "FAIL",
                "route_exists": False,
                "leg_lengths_m": lengths,
                "failure_reasons": ["PARTIAL_ROOM_MASK_DISCONNECT"],
            }
        lengths.append(distance)
    return {
        "status": "PASS",
        "route_exists": True,
        "leg_lengths_m": lengths,
        "total_path_length_m": sum(lengths),
        "failure_reasons": [],
    }


def compile_task(task: Mapping[str, Any], maps: SceneMaps) -> dict[str, Any]:
    visits = _route_visits(task)
    first_xy = visits[0]["xyz"][:2]
    first_cell = world_to_cell(
        first_xy, resolution=MAP_RESOLUTION_M, map_size=maps.map_size
    )
    radius_cells = int(math.ceil(MAXIMUM_STATION_DISTANCE_M / MAP_RESOLUTION_M)) + 2
    labels = {
        int(maps.labels[row, col])
        for row in range(
            max(0, first_cell[0] - radius_cells),
            min(maps.map_size, first_cell[0] + radius_cells + 1),
        )
        for col in range(
            max(0, first_cell[1] - radius_cells),
            min(maps.map_size, first_cell[1] + radius_cells + 1),
        )
        if int(maps.labels[row, col]) > 0
    }
    chains = [_compile_station_chain(task, maps, component) for component in sorted(labels)]
    passing = [chain for chain in chains if chain["status"] == "PASS"]
    if not passing:
        failures = Counter(reason for chain in chains for reason in chain["failures"])
        return {
            "schema_version": OVERLAY_VERSION,
            "task_spec_id": task["task_spec_id"],
            "task_name": task["task_name"],
            "scene_model": task["scene_model"],
            "source_family": task["source_family"],
            "selection_status": "NO_CANDIDATE_UNDER_L0_TARGET",
            "fixed_robot_pose": None,
            "initial_camera_view_state": None,
            "visible_relevant_instance_count": None,
            "visible_source_count": None,
            "visible_destination_count": None,
            "predicted_visible_instance_ids": [],
            "route_legs": [],
            "total_path_length_m": None,
            "total_move015_lower_bound": None,
            "egress_directions": [],
            "full_map_route_status": "FAIL",
            "partial_map_route_status": "NOT_COMPUTED",
            "partial_map_route": {"status": "NOT_COMPUTED"},
            "selection_reason": None,
            "failure_reasons": list(failures) or ["NO_CONNECTED_COMPONENT_NEAR_FIRST_TARGET"],
            "best_candidate_diagnostic": {
                "station_chain_failures": [
                    detail
                    for chain in chains
                    for detail in chain["failure_details"]
                ]
            },
            "goal_initial_false_preserved": task["initial_goal_false"] is True,
            "goal_dag_operations": task["operations"],
            "map_contract": {
                "map_resolution_m": MAP_RESOLUTION_M,
                "source_map_resolution_m": SOURCE_MAP_RESOLUTION_M,
                "robot_erosion_kernel_pixels": maps.erosion_kernel_pixels,
                "official_maps": maps.map_paths,
            },
        }
    chain = min(
        passing,
        key=lambda value: (
            sum(float(leg["path_length_m"] or 0.0) for leg in value["legs"][1:]),
            tuple(value["legs"][0]["station_cell"]),
            value["component"],
        ),
    )
    candidates = _candidate_records(task, maps, chain)
    selected = choose_fixed_reset_pose_candidate(candidates)
    diagnostic = min(
        candidates,
        key=lambda candidate: (
            not candidate["all_required_routes_exist"],
            not candidate["under_l0_route_target"],
            -candidate["visible_relevant_instance_count"],
            candidate["total_path_length_m"],
            candidate["stable_coordinate_tiebreak"],
        ),
        default=None,
    )
    chosen = selected or diagnostic
    if chosen is None:
        failure_reasons = ["NO_TRAVERSABLE_START_WITH_PUBLIC_TRANSLATION_EGRESS"]
    elif selected is None:
        failure_reasons = []
        if chosen["visible_relevant_instance_count"] == 0:
            failure_reasons.append("NO_RELEVANT_INSTANCE_PREDICTED_VISIBLE")
        if (
            max(chosen["route_leg_lengths_m"], default=0.0)
            > MAXIMUM_LEG_M + 1.0e-9
        ):
            failure_reasons.append("LEG_LENGTH_EXCEEDS_3M_TARGET")
        if chosen["total_path_length_m"] > MAXIMUM_TOTAL_M + 1.0e-9:
            failure_reasons.append("TOTAL_LENGTH_EXCEEDS_6M_TARGET")
        if len(task["rooms"]) > 2:
            failure_reasons.append("MORE_THAN_ONE_ADJACENT_ROOM")
        failure_reasons = failure_reasons or ["NO_CANDIDATE_UNDER_L0_TARGET"]
    else:
        failure_reasons = []
    full_cells = (
        []
        if chosen is None
        else [
            tuple(chosen["cell"]),
            *(tuple(leg["station_cell"]) for leg in chain["legs"]),
        ]
    )
    partial = (
        {"status": "NOT_COMPUTED"}
        if not full_cells
        else _partial_route_report(task, maps, full_cells)
    )
    fixed_pose = None
    view = None
    if selected is not None:
        yaw = math.radians(selected["yaw_deg"])
        fixed_pose = {
            "position": [*selected["world_xy"], 0.005],
            "orientation": [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
            "yaw_deg": selected["yaw_deg"],
            "map_cell": selected["cell"],
        }
        view = {
            "robot_model": "R1Pro",
            "torso_height_state": "neutral",
            "pitch_state": "center",
            "torso_qpos": list(EMPTY_TORSO_GRID[("neutral", "center")]),
            "camera_pitch_deg": selected["camera_pitch_deg"],
            "camera_height_m": selected["camera_height_m"],
            "initial_view_actions": [],
            "reset_contract": (
                "existing R1Pro neutral/center reset; no synthetic head joint "
                "and no pitch in base quaternion"
            ),
        }
    return {
        "schema_version": OVERLAY_VERSION,
        "task_spec_id": task["task_spec_id"],
        "task_name": task["task_name"],
        "scene_model": task["scene_model"],
        "source_family": task["source_family"],
        "public_instruction": task["public_instruction"],
        "selection_status": (
            "SELECTED"
            if selected is not None
            else "NO_CANDIDATE_UNDER_L0_TARGET"
        ),
        "fixed_robot_pose": fixed_pose,
        "initial_camera_view_state": view,
        "visible_relevant_instance_count": (
            None if chosen is None else chosen["visible_relevant_instance_count"]
        ),
        "visible_source_count": None if chosen is None else chosen["visible_source_count"],
        "visible_destination_count": None if chosen is None else chosen["visible_destination_count"],
        "predicted_visible_instance_ids": (
            [] if chosen is None else chosen["predicted_visible_instance_ids"]
        ),
        "route_legs": (
            []
            if chosen is None
            else [
                {**leg, "path_length_m": chosen["route_leg_lengths_m"][index]}
                for index, leg in enumerate(chain["legs"])
            ]
        ),
        "total_path_length_m": None if chosen is None else chosen["total_path_length_m"],
        "total_move015_lower_bound": (
            None
            if chosen is None
            else int(
                math.ceil(
                    chosen["total_path_length_m"]
                    / RESTRICTED_KINEMATIC_PUBLIC_MOVE_M
                    - 1.0e-12
                )
            )
        ),
        "egress_directions": [] if chosen is None else chosen["egress_world_directions"],
        "full_map_route_status": chain["status"],
        "partial_map_route_status": partial["status"],
        "partial_map_route": partial,
        "selection_reason": (
            None
            if selected is None
            else "A complete Goal-DAG route and L0 distance target passed; "
            "maximize visible instances, then visible role kinds, then minimize "
            "route length, initial view actions, and stable cell/yaw."
        ),
        "failure_reasons": failure_reasons,
        "best_candidate_diagnostic": diagnostic,
        "goal_initial_false_preserved": task["initial_goal_false"] is True,
        "goal_dag_operations": task["operations"],
        "map_contract": {
            "map_resolution_m": MAP_RESOLUTION_M,
            "source_map_resolution_m": SOURCE_MAP_RESOLUTION_M,
            "robot_erosion_kernel_pixels": maps.erosion_kernel_pixels,
            "official_maps": maps.map_paths,
        },
    }


def _distribution(values: Iterable[float], bins: Sequence[float]) -> dict[str, int]:
    values = list(values)
    result = {}
    lower = -math.inf
    for upper in bins:
        result[f"({lower},{upper}]"] = sum(lower < value <= upper for value in values)
        lower = upper
    result[f"({lower},inf)"] = sum(value > lower for value in values)
    return result


def build_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["selection_status"] == "SELECTED"]
    failed = [row for row in rows if row["selection_status"] != "SELECTED"]
    leg_lengths = [float(leg["path_length_m"]) for row in selected for leg in row["route_legs"]]
    totals = [float(row["total_path_length_m"]) for row in selected]
    failures = Counter(reason for row in failed for reason in row["failure_reasons"])
    visibility = Counter(int(row["visible_relevant_instance_count"]) for row in selected)
    return {
        "schema_version": "action20-capability-benchmark.l0-reset-pose-summary/v1",
        "cpu_only": True,
        "simulator_started": False,
        "live_executed": False,
        "task_count": len(rows),
        "original_task_count": sum(row["source_family"] == "ORIGINAL_100" for row in rows),
        "expansion_task_count": sum(row["source_family"] == "EXPANSION_80" for row in rows),
        "selected_count": len(selected),
        "no_candidate_under_l0_target_count": len(failed),
        "failure_histogram": dict(sorted(failures.items())),
        "visible_relevant_instance_count_distribution": {
            str(key): value for key, value in sorted(visibility.items())
        },
        "leg_length_distribution_m": _distribution(
            leg_lengths, (1.0, 2.0, 3.0, 4.0, 6.0)
        ),
        "total_length_distribution_m": _distribution(
            totals, (2.0, 4.0, 6.0, 8.0, 12.0)
        ),
        "all_legs_at_most_3m_count": sum(
            max(
                (leg["path_length_m"] for leg in row["route_legs"]),
                default=0.0,
            )
            <= 3.0
            for row in selected
        ),
        "total_at_most_6m_count": sum(
            row["total_path_length_m"] <= 6.0 for row in selected
        ),
        "full_map_pass_count": sum(
            row["full_map_route_status"] == "PASS" for row in rows
        ),
        "partial_map_pass_count": sum(
            row["partial_map_route_status"] == "PASS" for row in rows
        ),
        "partial_map_fail_count": sum(
            row["partial_map_route_status"] == "FAIL" for row in rows
        ),
        "partial_map_unavailable_count": sum(
            row["partial_map_route_status"] in {"UNAVAILABLE", "NOT_COMPUTED"}
            for row in rows
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-taskspecs", type=Path, required=True)
    parser.add_argument("--expansion-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--room-categories", type=Path, required=True)
    parser.add_argument("--robot-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    tasks = [
        *load_original_tasks(args.original_taskspecs),
        *load_expansion_tasks(args.expansion_root),
    ]
    if len(tasks) != 180 or len({task["task_spec_id"] for task in tasks}) != 180:
        raise ValueError("combined L0 inputs must contain exactly 180 unique tasks")
    scene_cache = {
        scene: SceneMaps(
            scene_model=scene,
            assets_root=args.assets_root,
            room_categories_path=args.room_categories,
            robot_metadata_path=args.robot_metadata,
        )
        for scene in sorted({task["scene_model"] for task in tasks})
    }
    rows = [compile_task(task, scene_cache[task["scene_model"]]) for task in tasks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    summary = build_summary(rows)
    summary["inputs"] = {
        "original_taskspecs": {
            "path": str(args.original_taskspecs.resolve()),
            "sha256": _sha256(args.original_taskspecs),
        },
        "expansion_root": str(args.expansion_root.resolve()),
        "assets_root": str(args.assets_root.resolve()),
        "room_categories": {
            "path": str(args.room_categories.resolve()),
            "sha256": _sha256(args.room_categories),
        },
        "robot_metadata": {
            "path": str(args.robot_metadata.resolve()),
            "sha256": _sha256(args.robot_metadata),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
