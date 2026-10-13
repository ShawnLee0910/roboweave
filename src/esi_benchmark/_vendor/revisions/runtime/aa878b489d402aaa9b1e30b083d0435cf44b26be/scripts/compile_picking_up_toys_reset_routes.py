#!/usr/bin/env python3
"""Compile reset-state interaction stations and routes without Isaac Sim.

This private-Oracle utility reproduces the BEHAVIOR traversability-map resize,
threshold, world/cell transform, and robot-footprint erosion using only CPU
libraries.  It enumerates every open station within the configured local
workspace of each ``picking_up_toys`` goal object and the toy box, then builds
a globally ordered six-object route on a four-connected grid.  Four-connected
paths are used deliberately: every edge maps directly to one public
``move_*`` direction and cannot cut a diagonal corner.

The output is valid only for the reset-state object positions and the exact
input hashes recorded in the manifest.  A live runner must fail closed if any
of those inputs or positions differ.
"""

from __future__ import annotations

import argparse
from collections import deque
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from behavior_hl.embodied.evaluator_bridge.interaction_station_geometry import (
    facing_yaw,
    select_interaction_station_cells,
)


NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
TARGET_PREFIXES = (
    "board_game.",
    "jigsaw_puzzle.",
    "tennis_ball.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _world_to_cell(
    xy: tuple[float, float], *, resolution: float, map_size: int
) -> tuple[int, int]:
    # Exact arithmetic equivalent of BaseMap.world_to_map for a 1-D [x, y].
    return (
        int(xy[1] / resolution + map_size / 2.0),
        int(xy[0] / resolution + map_size / 2.0),
    )


def _cell_to_world(
    cell: tuple[int, int], *, resolution: float, map_size: int
) -> tuple[float, float]:
    return (
        (cell[1] - map_size / 2.0) * resolution,
        (cell[0] - map_size / 2.0) * resolution,
    )


def _bfs(
    open_grid: np.ndarray, source: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    height, width = open_grid.shape
    distance = np.full((height, width), -1, dtype=np.int32)
    parent = np.full((height, width, 2), -1, dtype=np.int16)
    if not open_grid[source]:
        return distance, parent
    distance[source] = 0
    frontier: deque[tuple[int, int]] = deque([source])
    while frontier:
        row, col = frontier.popleft()
        next_distance = int(distance[row, col]) + 1
        for dr, dc in NEIGHBORS_4:
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < height
                and 0 <= nc < width
                and open_grid[nr, nc]
                and distance[nr, nc] < 0
            ):
                distance[nr, nc] = next_distance
                parent[nr, nc] = (row, col)
                frontier.append((nr, nc))
    return distance, parent


def _path_from_root(
    parent: np.ndarray,
    *,
    root: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]]:
    reverse_path = [target]
    cursor = target
    while cursor != root:
        values = parent[cursor]
        predecessor = (int(values[0]), int(values[1]))
        if predecessor == (-1, -1):
            raise RuntimeError(f"no parent path from {root} to {target}")
        reverse_path.append(predecessor)
        cursor = predecessor
    reverse_path.reverse()
    return reverse_path


def _path_to_root(
    parent: np.ndarray,
    *,
    source: tuple[int, int],
    root: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [source]
    cursor = source
    while cursor != root:
        values = parent[cursor]
        successor = (int(values[0]), int(values[1]))
        if successor == (-1, -1):
            raise RuntimeError(f"no parent path from {source} to {root}")
        path.append(successor)
        cursor = successor
    return path


def _station_cells(
    *,
    target_xy: tuple[float, float],
    open_grid: np.ndarray,
    component_labels: np.ndarray,
    source_label: int,
    resolution: float,
    map_size: int,
    maximum_distance_m: float,
    structural_open_grid: np.ndarray,
    room_instance_map: np.ndarray | None,
    target_room_ids: tuple[int, ...] | None,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    selection = select_interaction_station_cells(
        target_xy=target_xy,
        navigation_open_grid=open_grid,
        structural_open_grid=structural_open_grid,
        component_labels=component_labels,
        source_component_label=source_label,
        resolution=resolution,
        map_size=map_size,
        maximum_center_distance_m=maximum_distance_m,
        room_instance_map=room_instance_map,
        target_room_ids=target_room_ids,
        prefer_same_room=True,
    )
    return list(selection.cells), {
        "target_cell": list(selection.target_cell),
        "target_room_instance_id": selection.target_room_id,
        "target_center_room_instance_id": selection.target_center_room_id,
        "target_room_instance_ids": list(selection.target_room_ids),
        "target_room_source": selection.target_room_source,
        "raw_open_connected_count": selection.raw_open_connected_count,
        "structural_occlusion_rejection_count": (
            selection.structural_occlusion_rejection_count
        ),
        "same_room_candidate_count": selection.same_room_candidate_count,
        "cross_room_candidate_count": selection.cross_room_candidate_count,
        "same_room_preference_applied": selection.same_room_preference_applied,
        "cross_room_fallback_used": selection.cross_room_fallback_used,
    }


def _cell_record(
    cell: tuple[int, int],
    *,
    resolution: float,
    map_size: int,
    target_xy: tuple[float, float] | None = None,
) -> dict[str, Any]:
    world = _cell_to_world(cell, resolution=resolution, map_size=map_size)
    record: dict[str, Any] = {
        "cell": list(cell),
        "world_xy": [float(world[0]), float(world[1])],
    }
    if target_xy is not None:
        record["target_distance_m"] = float(math.dist(world, target_xy))
        record["target_yaw_rad"] = float(facing_yaw(world, target_xy))
    return record


def _load_resized_binary_map(path: Path, *, map_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"invalid square grayscale map: {path}")
    resized = cv2.resize(image, (map_size, map_size))
    resized[resized < 255] = 0
    return resized != 0


def _load_resized_label_map(path: Path, *, map_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"invalid square label map: {path}")
    return cv2.resize(
        image, (map_size, map_size), interpolation=cv2.INTER_NEAREST
    )


def _official_room_name_to_id(
    *,
    room_instance_map: np.ndarray,
    room_semantic_map: np.ndarray,
    room_categories: list[str],
) -> dict[str, int]:
    """Reconstruct the official ``<category>_<index>`` room-name binding."""

    semantic_to_instances: dict[int, list[int]] = {}
    for instance_id in sorted(
        int(value) for value in np.unique(room_instance_map) if int(value) > 0
    ):
        first = np.argwhere(room_instance_map == instance_id)
        if first.size == 0:
            raise RuntimeError(f"room instance {instance_id} has no pixels")
        row, col = (int(value) for value in first[0])
        semantic_id = int(room_semantic_map[row, col])
        if semantic_id <= 0 or semantic_id > len(room_categories):
            raise RuntimeError(
                f"room instance {instance_id} has invalid semantic id {semantic_id}"
            )
        semantic_to_instances.setdefault(semantic_id, []).append(instance_id)

    result: dict[str, int] = {}
    for semantic_id, instance_ids in sorted(semantic_to_instances.items()):
        category = room_categories[semantic_id - 1]
        for index, instance_id in enumerate(sorted(instance_ids)):
            result[f"{category}_{index}"] = instance_id
    return result


def _official_task_scope_rooms(
    *,
    scene_template_path: Path,
    scope_names: list[str],
) -> dict[str, tuple[str, ...]]:
    """Read official object ``in_rooms`` without importing OmniGibson."""

    scene_template = json.loads(scene_template_path.read_text(encoding="utf-8"))
    task_metadata = scene_template.get("metadata", {}).get("task", {})
    scope_to_scene_name = task_metadata.get("inst_to_name", {})
    init_info = scene_template.get("objects_info", {}).get("init_info", {})
    if not isinstance(scope_to_scene_name, dict) or not isinstance(init_info, dict):
        raise ValueError("scene template lacks official task/object bindings")

    result: dict[str, tuple[str, ...]] = {}
    for scope_name in scope_names:
        scene_name = scope_to_scene_name.get(scope_name)
        payload = init_info.get(scene_name, {}) if isinstance(scene_name, str) else {}
        room_names = payload.get("args", {}).get("in_rooms", [])
        if not isinstance(room_names, list) or not room_names or not all(
            isinstance(value, str) and value for value in room_names
        ):
            raise ValueError(
                f"official in_rooms binding is missing for {scope_name}: {scene_name}"
            )
        result[scope_name] = tuple(sorted(set(room_names)))
    return result


def _path_record(
    path: list[tuple[int, int]], *, resolution: float, map_size: int
) -> dict[str, Any]:
    return {
        "connectivity": 4,
        "cell_count": len(path),
        "edge_count": max(0, len(path) - 1),
        "distance_m": max(0, len(path) - 1) * resolution,
        "cells": [list(cell) for cell in path],
        "world_xy": [
            list(_cell_to_world(cell, resolution=resolution, map_size=map_size))
            for cell in path
        ],
    }


def compile_routes(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    map_png = args.map_png.resolve()
    reset_attestation = args.reset_attestation.resolve()
    robot_metadata = args.robot_metadata.resolve()
    structural_map_png = args.structural_map_png.resolve()
    room_instance_map_png = args.room_instance_map.resolve()
    room_semantic_map_png = args.room_semantic_map.resolve()
    room_categories_path = args.room_categories.resolve()
    scene_template_path = args.scene_template.resolve()
    image = cv2.imread(str(map_png), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"invalid square grayscale map: {map_png}")

    map_size = int(
        image.shape[0] * args.map_default_resolution / args.map_resolution
    )
    floor_map = cv2.resize(image, (map_size, map_size))
    floor_map[floor_map < 255] = 0

    metadata = json.loads(robot_metadata.read_text(encoding="utf-8"))
    bbox_size = [float(value) for value in metadata["bbox_size"]]
    erosion_radius_m = math.hypot(*bbox_size[:2]) / 2.0 + args.robot_padding_m
    erosion_kernel_pixels = int(math.ceil(erosion_radius_m / args.map_resolution))
    eroded = cv2.erode(
        floor_map,
        np.ones((erosion_kernel_pixels, erosion_kernel_pixels), dtype=np.uint8),
    )
    open_grid = eroded != 0
    structural_open_grid = _load_resized_binary_map(
        structural_map_png, map_size=map_size
    )
    room_instance_map = _load_resized_label_map(
        room_instance_map_png, map_size=map_size
    )
    room_semantic_map = _load_resized_label_map(
        room_semantic_map_png, map_size=map_size
    )
    room_categories = room_categories_path.read_text(encoding="utf-8").splitlines()
    room_name_to_id = _official_room_name_to_id(
        room_instance_map=room_instance_map,
        room_semantic_map=room_semantic_map,
        room_categories=room_categories,
    )
    loaded_room_types = tuple(dict.fromkeys(args.loaded_room_type))
    if loaded_room_types:
        unknown = [name for name in loaded_room_types if name not in room_categories]
        if unknown:
            raise ValueError(f"unknown loaded room types: {unknown}")
        loaded_semantic_ids = [room_categories.index(name) + 1 for name in loaded_room_types]
        loaded_room_mask = np.isin(room_semantic_map, loaded_semantic_ids)
        open_grid &= loaded_room_mask
    else:
        loaded_semantic_ids = []

    reset = json.loads(reset_attestation.read_text(encoding="utf-8"))
    state = reset["stable_serialized_state"]
    scope = state["task_scope"]
    robot_xy = tuple(float(value) for value in reset["robot_base_position"][:2])
    object_xy = {
        name: tuple(float(value) for value in payload["position"][:2])
        for name, payload in scope.items()
        if name.startswith(TARGET_PREFIXES)
    }
    container_names = [name for name in scope if name.startswith("toy_box.")]
    if len(object_xy) != 6 or len(container_names) != 1:
        raise ValueError(
            "expected exactly six goal objects and one toy_box in reset state"
        )
    container_name = container_names[0]
    container_xy = tuple(
        float(value) for value in scope[container_name]["position"][:2]
    )
    target_names = [*sorted(object_xy), container_name]
    official_target_rooms = _official_task_scope_rooms(
        scene_template_path=scene_template_path,
        scope_names=target_names,
    )
    unresolved_rooms = sorted(
        {
            room_name
            for room_names in official_target_rooms.values()
            for room_name in room_names
            if room_name not in room_name_to_id
        }
    )
    if unresolved_rooms:
        raise RuntimeError(
            f"official target rooms are absent from room maps: {unresolved_rooms}"
        )
    official_target_room_ids = {
        name: tuple(room_name_to_id[room] for room in official_target_rooms[name])
        for name in target_names
    }
    start_cell = _world_to_cell(
        robot_xy, resolution=args.map_resolution, map_size=map_size
    )
    if not open_grid[start_cell]:
        raise RuntimeError(f"reset robot cell is closed after erosion: {start_cell}")

    _, labels = cv2.connectedComponents(open_grid.astype(np.uint8), connectivity=4)
    source_label = int(labels[start_cell])
    station_results = {
        name: _station_cells(
            target_xy=xy,
            open_grid=open_grid,
            component_labels=labels,
            source_label=source_label,
            resolution=args.map_resolution,
            map_size=map_size,
            maximum_distance_m=args.maximum_object_interaction_distance_m,
            structural_open_grid=structural_open_grid,
            room_instance_map=room_instance_map,
            target_room_ids=official_target_room_ids[name],
        )
        for name, xy in object_xy.items()
    }
    station_cells = {name: result[0] for name, result in station_results.items()}
    station_audits = {name: result[1] for name, result in station_results.items()}
    container_cells, container_audit = _station_cells(
        target_xy=container_xy,
        open_grid=open_grid,
        component_labels=labels,
        source_label=source_label,
        resolution=args.map_resolution,
        map_size=map_size,
        maximum_distance_m=args.maximum_container_interaction_distance_m,
        structural_open_grid=structural_open_grid,
        room_instance_map=room_instance_map,
        target_room_ids=official_target_room_ids[container_name],
    )
    empty = [name for name, cells in station_cells.items() if not cells]
    if empty or not container_cells:
        raise RuntimeError(
            f"missing reset-state station set: objects={empty}, container={len(container_cells)}"
        )

    roots = [start_cell, *container_cells]
    bfs_results = [_bfs(open_grid, root) for root in roots]
    object_names = sorted(object_xy)
    transitions: dict[tuple[int, int, int], tuple[int, int, int, tuple[int, int]]] = {}
    for source_index in range(len(roots)):
        source_distances = bfs_results[source_index][0]
        for object_index, name in enumerate(object_names):
            for box_index in range(len(container_cells)):
                box_distances = bfs_results[1 + box_index][0]
                options = []
                for station in station_cells[name]:
                    first = int(source_distances[station])
                    second = int(box_distances[station])
                    if first >= 0 and second >= 0:
                        options.append((first + second, first, second, station))
                if not options:
                    raise RuntimeError(
                        f"no route through {name} from root {source_index} to box {box_index}"
                    )
                transitions[(source_index, object_index, box_index)] = min(options)

    @lru_cache(maxsize=None)
    def solve(mask: int, source_index: int) -> tuple[int, tuple[Any, ...]]:
        if mask == 0:
            return 0, ()
        best: tuple[int, tuple[Any, ...]] | None = None
        for object_index, name in enumerate(object_names):
            if not mask & (1 << object_index):
                continue
            for box_index in range(len(container_cells)):
                transition = transitions[(source_index, object_index, box_index)]
                tail_cost, tail = solve(mask ^ (1 << object_index), 1 + box_index)
                candidate = (
                    transition[0] + tail_cost,
                    ((object_index, box_index, transition), *tail),
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            raise RuntimeError("route dynamic program found no complete solution")
        return best

    total_steps, sequence = solve((1 << len(object_names)) - 1, 0)
    route_legs = []
    action_sequence = []
    current_root_index = 0
    for cycle_index, (object_index, box_index, transition) in enumerate(sequence, 1):
        _, to_object_steps, to_box_steps, object_station = transition
        object_name = object_names[object_index]
        source_cell = roots[current_root_index]
        box_cell = container_cells[box_index]
        to_object = _path_from_root(
            bfs_results[current_root_index][1],
            root=source_cell,
            target=object_station,
        )
        to_box = _path_to_root(
            bfs_results[1 + box_index][1],
            source=object_station,
            root=box_cell,
        )
        if len(to_object) - 1 != to_object_steps or len(to_box) - 1 != to_box_steps:
            raise AssertionError("compiled path length disagrees with dynamic program")
        route_legs.append(
            {
                "cycle": cycle_index,
                "object": object_name,
                "object_reset_xy": list(object_xy[object_name]),
                "object_station": _cell_record(
                    object_station,
                    resolution=args.map_resolution,
                    map_size=map_size,
                    target_xy=object_xy[object_name],
                ),
                "to_object": _path_record(
                    to_object, resolution=args.map_resolution, map_size=map_size
                ),
                "container": container_name,
                "container_reset_xy": list(container_xy),
                "container_station": _cell_record(
                    box_cell,
                    resolution=args.map_resolution,
                    map_size=map_size,
                    target_xy=container_xy,
                ),
                "to_container": _path_record(
                    to_box, resolution=args.map_resolution, map_size=map_size
                ),
            }
        )
        action_sequence.extend(
            [
                {"action": "follow_precompiled_path", "leg": "to_object", "cycle": cycle_index},
                {"action": "grasp_private_gt", "object": object_name},
                {"action": "follow_precompiled_path", "leg": "to_container", "cycle": cycle_index},
                {"action": "place_inside_private_gt", "container": container_name},
            ]
        )
        current_root_index = 1 + box_index

    route_segments: list[dict[str, Any]] = []
    for item in route_legs:
        for leg_name, action_name, target_key, station_key, distance_limit in (
            (
                "to_object",
                "grasp",
                "object",
                "object_station",
                args.maximum_object_interaction_distance_m,
            ),
            (
                "to_container",
                "place_inside",
                "container",
                "container_station",
                args.maximum_container_interaction_distance_m,
            ),
        ):
            target_center_key = (
                "object_reset_xy" if target_key == "object" else "container_reset_xy"
            )
            target_center = item[target_center_key]
            target_cell = _world_to_cell(
                tuple(target_center), resolution=args.map_resolution, map_size=map_size
            )
            route_segments.append(
                {
                    "segment_id": f"cycle_{int(item['cycle']):02d}_{leg_name}",
                    "cycle": int(item["cycle"]),
                    "purpose": leg_name,
                    "interaction_action": action_name,
                    "target_private_id": item[target_key],
                    "target_center_world_xy": list(target_center),
                    "target_center_room_instance_id": int(
                        room_instance_map[target_cell]
                    ),
                    "target_room_instance_id": int(
                        official_target_room_ids[item[target_key]][0]
                    ),
                    "target_room_instance_ids": list(
                        official_target_room_ids[item[target_key]]
                    ),
                    "target_room_names_private": list(
                        official_target_rooms[item[target_key]]
                    ),
                    "target_room_binding_source": "official_object_in_rooms",
                    "maximum_center_distance_m": float(distance_limit),
                    "station": dict(item[station_key]),
                    "planned_path": dict(item[leg_name]),
                    "structural_line_of_sight_verified": True,
                    "station_faces_target": True,
                }
            )

    common = {
        "schema_version": "behavior2026.picking-up-toys-offline-route/v2",
        "activity_name": state["activity_name"],
        "scene_name": state["scene_name"],
        "execution_fidelity": "kinematic_research",
        "official_challenge_eligible": False,
        "planner": "pure_cpu_reset_state_four_connected_dynamic_program",
        "navigation_contract": "behavior2026.generic-indoor-navigation/v122",
        "maximum_object_interaction_distance_m": (
            args.maximum_object_interaction_distance_m
        ),
        "maximum_container_interaction_distance_m": (
            args.maximum_container_interaction_distance_m
        ),
        "interaction_distance_reference": "target_object_world_xy_center",
        "structural_line_of_sight_required": True,
        "structural_line_of_sight_map_kind": "floor_trav_no_obj",
        "same_room_station_preferred": True,
        "cross_room_station_allowed_only_without_structural_occlusion": True,
        "loaded_room_types": list(loaded_room_types),
        "loaded_room_semantic_ids": loaded_semantic_ids,
        "map_resolution_m": args.map_resolution,
        "map_size_cells": map_size,
        "map_resize_contract": "official_cv2_resize_then_values_below_255_blocked",
        "map_transform_contract": "official_BaseMap_world_to_map_equivalent",
        "erosion_contract": "official_formula_with_r1pro_metadata_bbox_proxy",
        "robot_bbox_xy_m": bbox_size[:2],
        "robot_padding_m": args.robot_padding_m,
        "erosion_radius_m": erosion_radius_m,
        "erosion_kernel_pixels": erosion_kernel_pixels,
        "input_paths": {
            "map_png": str(map_png),
            "reset_attestation": str(reset_attestation),
            "robot_metadata": str(robot_metadata),
            "structural_map_png": str(structural_map_png),
            "room_instance_map": str(room_instance_map_png),
            "room_semantic_map": str(room_semantic_map_png),
            "room_categories": str(room_categories_path),
            "scene_template": str(scene_template_path),
        },
        "input_sha256": {
            "map_png": _sha256(map_png),
            "reset_attestation": _sha256(reset_attestation),
            "robot_metadata": _sha256(robot_metadata),
            "structural_map_png": _sha256(structural_map_png),
            "room_instance_map": _sha256(room_instance_map_png),
            "room_semantic_map": _sha256(room_semantic_map_png),
            "room_categories": _sha256(room_categories_path),
            "scene_template": _sha256(scene_template_path),
        },
        "reset_state_sha256": reset.get("stable_state_sha256"),
        "reset_robot": _cell_record(
            start_cell,
            resolution=args.map_resolution,
            map_size=map_size,
        )
        | {"measured_world_xy": list(robot_xy)},
        "source_component_open_cell_count": int(np.count_nonzero(labels == source_label)),
        "official_target_rooms_private": {
            name: list(official_target_rooms[name]) for name in target_names
        },
        "official_target_room_ids_private": {
            name: list(official_target_room_ids[name]) for name in target_names
        },
        "cache_contract": {
            "valid_only_for_exact_input_hashes": True,
            "valid_only_for_reset_object_xy": True,
            "live_object_position_mismatch": "fail_closed_and_recompile_from_state_without_rendering",
            "live_map_or_robot_footprint_mismatch": "fail_closed",
        },
    }
    catalog = common | {
        "entities": {
            **{
                name: {
                    "kind": "goal_object",
                    "reset_world_xy": list(object_xy[name]),
                    "reset_cell": list(
                        _world_to_cell(
                            object_xy[name],
                            resolution=args.map_resolution,
                            map_size=map_size,
                        )
                    ),
                    "station_count": len(station_cells[name]),
                    "station_selection": station_audits[name],
                    "stations": [
                        _cell_record(
                            cell,
                            resolution=args.map_resolution,
                            map_size=map_size,
                            target_xy=object_xy[name],
                        )
                        for cell in station_cells[name]
                    ],
                }
                for name in object_names
            },
            container_name: {
                "kind": "goal_container",
                "reset_world_xy": list(container_xy),
                "reset_cell": list(
                    _world_to_cell(
                        container_xy,
                        resolution=args.map_resolution,
                        map_size=map_size,
                    )
                ),
                "station_count": len(container_cells),
                "station_selection": container_audit,
                "stations": [
                    _cell_record(
                        cell,
                        resolution=args.map_resolution,
                        map_size=map_size,
                        target_xy=container_xy,
                    )
                    for cell in container_cells
                ],
            },
        }
    }
    route = common | {
        "coverage": {
            "goal_object_count": len(object_names),
            "compiled_goal_object_count": len(route_legs),
            "all_goal_objects_have_station": all(station_cells.values()),
            "container_has_station": bool(container_cells),
            "complete_route_exists": len(route_legs) == len(object_names),
        },
        "optimization": {
            "objective": "minimum_total_cardinal_grid_steps_over_object_order_and_two_container_stations",
            "total_cardinal_edges": total_steps,
            "total_translation_distance_m": total_steps * args.map_resolution,
            "rotation_cost_included": False,
        },
        "route_cycles": route_legs,
        "route_segments": route_segments,
        "private_oracle_action_sequence": action_sequence,
    }
    return catalog, route


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-png", type=Path, required=True)
    parser.add_argument("--reset-attestation", type=Path, required=True)
    parser.add_argument("--robot-metadata", type=Path, required=True)
    parser.add_argument("--structural-map-png", type=Path, required=True)
    parser.add_argument("--room-instance-map", type=Path, required=True)
    parser.add_argument("--room-semantic-map", type=Path, required=True)
    parser.add_argument("--room-categories", type=Path, required=True)
    parser.add_argument("--scene-template", type=Path, required=True)
    parser.add_argument(
        "--loaded-room-type",
        action="append",
        default=[],
        help="Repeat for every room type loaded by the partial-scene evaluator.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--map-resolution", type=float, default=0.1)
    parser.add_argument("--map-default-resolution", type=float, default=0.01)
    parser.add_argument("--robot-padding-m", type=float, default=0.2)
    parser.add_argument(
        "--maximum-object-interaction-distance-m", type=float, default=1.3
    )
    parser.add_argument(
        "--maximum-container-interaction-distance-m", type=float, default=1.3
    )
    args = parser.parse_args()
    if args.maximum_object_interaction_distance_m <= 0.0:
        parser.error("--maximum-object-interaction-distance-m must be positive")
    if args.maximum_container_interaction_distance_m <= 0.0:
        parser.error("--maximum-container-interaction-distance-m must be positive")
    catalog, route = compile_routes(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output_dir / "station_catalog.json"
    route_path = args.output_dir / "compiled_oracle_route.json"
    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    route_path.write_text(
        json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "station_catalog": str(catalog_path),
                "compiled_oracle_route": str(route_path),
                "station_counts": {
                    name: payload["station_count"]
                    for name, payload in catalog["entities"].items()
                },
                "complete_route_exists": route["coverage"]["complete_route_exists"],
                "total_translation_distance_m": route["optimization"]["total_translation_distance_m"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
