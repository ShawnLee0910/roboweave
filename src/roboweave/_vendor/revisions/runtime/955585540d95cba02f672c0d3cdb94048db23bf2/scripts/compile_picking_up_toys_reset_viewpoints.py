#!/usr/bin/env python3
"""Compile the reset R1Pro camera rig and discrete interaction views on CPU.

This is the second stage of the private ``picking_up_toys`` Oracle compiler.
It consumes the reset attestation and the already-compiled reset route, derives
the R1Pro head-camera pose from the official URDF and camera import config, and
quantizes every object/container view into 15-degree turn/look actions.

No simulator, renderer, GPU, segmentation truth, or online pose sampler is
used.  The result remains kinematic-research infrastructure and must be parity
checked against the live camera transform before its first replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Iterable

import numpy as np
import yaml


TORSO_JOINT_NAMES = (
    "torso_joint1",
    "torso_joint2",
    "torso_joint3",
    "torso_joint4",
)
TORSO_QPOS_START = 6
ZED_JOINT_NAME = "zed_joint"
ZED_CAMERA_LINK_NAME = "zed_link"

# The R1Pro source config says its [-1, 0, 0, 0] generated-camera offset
# aligns the camera with robot/world +X.  After applying the official URDF
# chain, local -Y is the only camera axis that has the base yaw and the zed
# link's physical downward pitch.  Local -Z would incorrectly point 62 degrees
# upward.  The first live replay must still compare this derived pose with the
# sensor's cameraViewTransform and fail closed on disagreement.
CAMERA_FORWARD_AXIS_LOCAL = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
HEAD_HORIZONTAL_APERTURE_MM = 40.0
HEAD_FOCAL_LENGTH_MM = 17.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_vector(text: str | None, *, length: int) -> np.ndarray:
    values = [float(value) for value in (text or "").split()]
    if len(values) != length:
        raise ValueError(f"expected {length} values, got {values}")
    return np.asarray(values, dtype=np.float64)


def _rotation_from_rpy(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        if abs(angle) > 1e-12:
            raise ValueError("nonzero rotation around a zero axis")
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c, s, one_minus_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def _transform(position: Iterable[float], rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(list(position), dtype=np.float64)
    return result


def _quaternion_xyzw_to_rotation(quaternion: Iterable[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _yaw_pitch(vector: Iterable[float]) -> tuple[float, float]:
    x, y, z = (float(value) for value in vector)
    return math.atan2(y, x), math.atan2(z, math.hypot(x, y))


def _wrap_radians(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _angle_degrees(angle: float) -> float:
    return math.degrees(_wrap_radians(angle))


def _quantized_steps(delta_radians: float, step_degrees: float) -> int:
    scaled = math.degrees(_wrap_radians(delta_radians)) / step_degrees
    # Deliberately avoid Python's ties-to-even rounding.
    magnitude = math.floor(abs(scaled) + 0.5 + 1e-12)
    return magnitude if scaled >= 0.0 else -magnitude


def _repeat_action(positive: str, negative: str, signed_steps: int) -> list[str]:
    action = positive if signed_steps >= 0 else negative
    return [action] * abs(signed_steps)


def _inverse_actions(actions: Iterable[str]) -> list[str]:
    inverse = {
        "turn_left": "turn_right",
        "turn_right": "turn_left",
        "look_up": "look_down",
        "look_down": "look_up",
    }
    return [inverse[action] for action in reversed(list(actions))]


def _joint_specifications(urdf_path: Path) -> dict[str, dict[str, np.ndarray]]:
    root = ET.parse(urdf_path).getroot()
    result: dict[str, dict[str, np.ndarray]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in {*TORSO_JOINT_NAMES, ZED_JOINT_NAME}:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        result[name] = {
            "xyz": _parse_vector(origin.attrib.get("xyz") if origin is not None else "0 0 0", length=3),
            "rpy": _parse_vector(origin.attrib.get("rpy") if origin is not None else "0 0 0", length=3),
            "axis": _parse_vector(axis.attrib.get("xyz") if axis is not None else "0 0 0", length=3),
        }
    expected = {*TORSO_JOINT_NAMES, ZED_JOINT_NAME}
    if set(result) != expected:
        raise ValueError(f"missing R1Pro camera-chain joints: {sorted(expected - set(result))}")
    return result


def _camera_offset(source_cfg_path: Path) -> tuple[np.ndarray, np.ndarray]:
    config = yaml.safe_load(source_cfg_path.read_text(encoding="utf-8"))
    candidates = [
        item for item in config.get("camera_links", []) if item.get("link") == ZED_CAMERA_LINK_NAME
    ]
    if len(candidates) != 1:
        raise ValueError("expected exactly one zed_link camera in R1Pro source config")
    offset = candidates[0]["offset"]
    position = np.asarray(offset["position"], dtype=np.float64)
    orientation = _quaternion_xyzw_to_rotation(offset["orientation"])
    return position, orientation


def derive_reset_camera_rig(
    reset: dict[str, Any], *, urdf_path: Path, source_cfg_path: Path
) -> dict[str, Any]:
    qpos = [float(value) for value in reset["robot_joint_positions"]]
    if len(qpos) < TORSO_QPOS_START + len(TORSO_JOINT_NAMES):
        raise ValueError("reset attestation does not contain the R1Pro torso qpos slice")
    torso_qpos = qpos[TORSO_QPOS_START : TORSO_QPOS_START + len(TORSO_JOINT_NAMES)]
    specs = _joint_specifications(urdf_path)

    base_position = np.asarray(reset["robot_base_position"], dtype=np.float64)
    base_rotation = _quaternion_xyzw_to_rotation(reset["robot_base_quaternion_xyzw"])
    world_from_base = _transform(base_position, base_rotation)
    world_from_link = world_from_base.copy()
    for name, position in zip(TORSO_JOINT_NAMES, torso_qpos):
        spec = specs[name]
        origin = _transform(spec["xyz"], _rotation_from_rpy(spec["rpy"]))
        motion = _transform((0.0, 0.0, 0.0), _rotation_from_axis_angle(spec["axis"], position))
        world_from_link = world_from_link @ origin @ motion
    zed = specs[ZED_JOINT_NAME]
    world_from_zed = world_from_link @ _transform(zed["xyz"], _rotation_from_rpy(zed["rpy"]))
    camera_position, camera_orientation = _camera_offset(source_cfg_path)
    world_from_camera = world_from_zed @ _transform(camera_position, camera_orientation)
    base_from_camera = np.linalg.inv(world_from_base) @ world_from_camera

    forward_world = world_from_camera[:3, :3] @ CAMERA_FORWARD_AXIS_LOCAL
    yaw, pitch = _yaw_pitch(forward_world)
    base_yaw, _ = _yaw_pitch(base_rotation[:, 0])
    return {
        "base_world_position": base_position.tolist(),
        "base_quaternion_xyzw": [float(value) for value in reset["robot_base_quaternion_xyzw"]],
        "base_yaw_deg": math.degrees(base_yaw),
        "torso_qpos_indices": [TORSO_QPOS_START, TORSO_QPOS_START + len(TORSO_JOINT_NAMES)],
        "torso_qpos": torso_qpos,
        "camera_world_position": world_from_camera[:3, 3].tolist(),
        "camera_forward_world": forward_world.tolist(),
        "camera_yaw_deg": math.degrees(yaw),
        "camera_pitch_deg": math.degrees(pitch),
        "base_from_camera_matrix": base_from_camera.tolist(),
        "camera_forward_axis_local": CAMERA_FORWARD_AXIS_LOCAL.tolist(),
    }


def _world_from_base_at_station(
    reset_rig: dict[str, Any], station_xy: Iterable[float]
) -> np.ndarray:
    base_rotation = _quaternion_xyzw_to_rotation(reset_rig["base_quaternion_xyzw"])
    base_position = np.asarray(reset_rig["base_world_position"], dtype=np.float64)
    station_position = base_position.copy()
    station_position[:2] = np.asarray(list(station_xy), dtype=np.float64)
    return _transform(station_position, base_rotation)


def compile_view(
    *,
    reset_rig: dict[str, Any],
    station: dict[str, Any],
    target_name: str,
    target_position: Iterable[float],
    maximum_interaction_distance_m: float,
    turn_step_degrees: float,
    look_step_degrees: float,
) -> dict[str, Any]:
    world_from_base = _world_from_base_at_station(reset_rig, station["world_xy"])
    base_from_camera = np.asarray(reset_rig["base_from_camera_matrix"], dtype=np.float64)
    world_from_camera = world_from_base @ base_from_camera
    camera_position = world_from_camera[:3, 3]
    canonical_forward = world_from_camera[:3, :3] @ CAMERA_FORWARD_AXIS_LOCAL
    canonical_yaw, canonical_pitch = _yaw_pitch(canonical_forward)
    target_position_array = np.asarray(list(target_position), dtype=np.float64)
    target_vector = target_position_array - camera_position
    target_yaw, target_pitch = _yaw_pitch(target_vector)

    turn_steps = _quantized_steps(target_yaw - canonical_yaw, turn_step_degrees)
    look_steps = _quantized_steps(target_pitch - canonical_pitch, look_step_degrees)
    turn_actions = _repeat_action("turn_left", "turn_right", turn_steps)
    look_actions = _repeat_action("look_up", "look_down", look_steps)
    approach_actions = [*turn_actions, *look_actions]
    quantized_yaw = canonical_yaw + math.radians(turn_steps * turn_step_degrees)
    quantized_pitch = canonical_pitch + math.radians(look_steps * look_step_degrees)
    horizontal_residual = _angle_degrees(target_yaw - quantized_yaw)
    vertical_residual = math.degrees(target_pitch - quantized_pitch)
    camera_horizontal_distance = float(np.linalg.norm(target_vector[:2]))
    base_target_distance = float(station["target_distance_m"])
    return {
        "target": target_name,
        "station": station,
        "target_world_position": target_position_array.tolist(),
        "camera_world_position_at_canonical_rig": camera_position.tolist(),
        "canonical_camera_yaw_deg": math.degrees(canonical_yaw),
        "canonical_camera_pitch_deg": math.degrees(canonical_pitch),
        "exact_target_yaw_deg": math.degrees(target_yaw),
        "exact_target_pitch_deg": math.degrees(target_pitch),
        "base_target_distance_m": base_target_distance,
        "camera_horizontal_target_distance_m": camera_horizontal_distance,
        "three_dimensional_distance_m": float(np.linalg.norm(target_vector)),
        "turn_steps_signed": turn_steps,
        "look_steps_signed": look_steps,
        "approach_actions": approach_actions,
        "restore_actions": _inverse_actions(approach_actions),
        "quantized_camera_yaw_deg": math.degrees(quantized_yaw),
        "quantized_camera_pitch_deg": math.degrees(quantized_pitch),
        "horizontal_residual_deg": horizontal_residual,
        "vertical_residual_deg": vertical_residual,
        "maximum_interaction_distance_m": float(maximum_interaction_distance_m),
        "within_interaction_distance": (
            base_target_distance <= maximum_interaction_distance_m + 1e-7
        ),
        "within_half_quantization_step": (
            abs(horizontal_residual) <= turn_step_degrees / 2.0 + 1e-7
            and abs(vertical_residual) <= look_step_degrees / 2.0 + 1e-7
        ),
        "canonical_rig_restored_after_interaction": True,
    }


def compile_viewpoints(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_path = args.reset_attestation.resolve()
    route_path = args.compiled_route.resolve()
    urdf_path = args.robot_urdf.resolve()
    source_cfg_path = args.robot_source_cfg.resolve()
    reset = json.loads(reset_path.read_text(encoding="utf-8"))
    route = json.loads(route_path.read_text(encoding="utf-8"))
    state = reset["stable_serialized_state"]
    scope = state["task_scope"]
    if route["reset_state_sha256"] != reset.get("stable_state_sha256"):
        raise ValueError("compiled route and reset attestation state hashes differ")

    reset_rig = derive_reset_camera_rig(
        reset, urdf_path=urdf_path, source_cfg_path=source_cfg_path
    )
    cycles: list[dict[str, Any]] = []
    all_views: dict[str, dict[str, Any]] = {}
    action_plan: list[dict[str, Any]] = []
    for route_cycle in route["route_cycles"]:
        cycle = int(route_cycle["cycle"])
        object_name = route_cycle["object"]
        container_name = route_cycle["container"]
        object_view = compile_view(
            reset_rig=reset_rig,
            station=route_cycle["object_station"],
            target_name=object_name,
            target_position=scope[object_name]["position"],
            maximum_interaction_distance_m=float(
                route["maximum_object_interaction_distance_m"]
            ),
            turn_step_degrees=args.turn_step_degrees,
            look_step_degrees=args.look_step_degrees,
        )
        container_view = compile_view(
            reset_rig=reset_rig,
            station=route_cycle["container_station"],
            target_name=container_name,
            target_position=scope[container_name]["position"],
            maximum_interaction_distance_m=float(
                route["maximum_container_interaction_distance_m"]
            ),
            turn_step_degrees=args.turn_step_degrees,
            look_step_degrees=args.look_step_degrees,
        )
        cycles.append(
            {
                "cycle": cycle,
                "object_view": object_view,
                "container_view": container_view,
            }
        )
        all_views[f"cycle_{cycle:02d}_object"] = object_view
        all_views[f"cycle_{cycle:02d}_container"] = container_view

        def append(kind: str, **payload: Any) -> None:
            action_plan.append({"index": len(action_plan), "action": kind, **payload})

        append("follow_precompiled_path", cycle=cycle, leg="to_object", private_gt=True)
        for action in object_view["approach_actions"]:
            append(action, cycle=cycle, purpose="view_object", public_discrete_action=True)
        append("grasp_private_gt", cycle=cycle, object=object_name, private_gt=True)
        for action in object_view["restore_actions"]:
            append(action, cycle=cycle, purpose="restore_canonical_rig", public_discrete_action=True)
        append("follow_precompiled_path", cycle=cycle, leg="to_container", private_gt=True)
        for action in container_view["approach_actions"]:
            append(action, cycle=cycle, purpose="view_container", public_discrete_action=True)
        append("place_inside_private_gt", cycle=cycle, container=container_name, private_gt=True)
        for action in container_view["restore_actions"]:
            append(action, cycle=cycle, purpose="restore_canonical_rig", public_discrete_action=True)

    common = {
        "schema_version": "behavior2026.picking-up-toys-reset-viewpoints/v1",
        "activity_name": state["activity_name"],
        "scene_name": state["scene_name"],
        "execution_fidelity": "kinematic_research",
        "official_challenge_eligible": False,
        "marker_required_for_private_oracle": False,
        "turn_step_degrees": args.turn_step_degrees,
        "look_step_degrees": args.look_step_degrees,
        "canonical_rig_contract": (
            "reset R1Pro base orientation and torso pose; after every interaction, "
            "invert discrete turn/look actions before following the next route leg"
        ),
        "camera_contract": {
            "source": "official R1Pro source URDF forward kinematics plus official camera import offset",
            "head_camera_is_fixed_to_torso_link4": True,
            "r1pro_has_independent_head_camera_joint": False,
            "look_actions_are_virtual_rig_actions": True,
            "look_actions_are_not_official_r1pro_joint_control": True,
            "horizontal_aperture_mm": HEAD_HORIZONTAL_APERTURE_MM,
            "focal_length_mm": HEAD_FOCAL_LENGTH_MM,
            "nominal_horizontal_fov_deg": math.degrees(
                2.0
                * math.atan(
                    HEAD_HORIZONTAL_APERTURE_MM / (2.0 * HEAD_FOCAL_LENGTH_MM)
                )
            ),
            "first_live_replay_requires_camera_transform_parity_check": True,
            "parity_failure": "fail_closed_without_executing_oracle_actions",
        },
        "reset_camera_rig": reset_rig,
        "input_paths": {
            "reset_attestation": str(reset_path),
            "compiled_route": str(route_path),
            "robot_urdf": str(urdf_path),
            "robot_source_cfg": str(source_cfg_path),
        },
        "input_sha256": {
            "reset_attestation": _sha256(reset_path),
            "compiled_route": _sha256(route_path),
            "robot_urdf": _sha256(urdf_path),
            "robot_source_cfg": _sha256(source_cfg_path),
        },
        "reset_state_sha256": reset.get("stable_state_sha256"),
    }
    viewpoints = common | {
        "coverage": {
            "interaction_view_count": len(all_views),
            "all_within_interaction_distance": all(
                view["within_interaction_distance"] for view in all_views.values()
            ),
            "all_quantization_residuals_bounded": all(
                view["within_half_quantization_step"] for view in all_views.values()
            ),
            "all_restore_canonical_rig": all(
                view["canonical_rig_restored_after_interaction"] for view in all_views.values()
            ),
        },
        "route_cycles": cycles,
        "interaction_views": all_views,
    }
    plan = common | {
        "route_reference": str(route_path),
        "route_cycle_count": len(cycles),
        "action_count_excluding_path_internal_moves": len(action_plan),
        "path_internal_cardinal_edges": route["optimization"]["total_cardinal_edges"],
        "actions": action_plan,
        "runtime_contract": {
            "follow_precompiled_path_uses_recorded_world_waypoints": True,
            "navigation_preserves_canonical_rig_orientation": True,
            "interaction_uses_private_gt_binding": True,
            "after_each_navigation_leg_render_one_frame": True,
            "after_each_discrete_view_action_render_one_frame": True,
            "live_segmentation_is_validation_not_planning": True,
            "on_target_not_visible_after_compiled_view": "fail_closed_and_record_frame",
        },
    }
    return viewpoints, plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-attestation", type=Path, required=True)
    parser.add_argument("--compiled-route", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--robot-source-cfg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--turn-step-degrees", type=float, default=15.0)
    parser.add_argument("--look-step-degrees", type=float, default=15.0)
    args = parser.parse_args()
    if args.turn_step_degrees <= 0.0 or args.look_step_degrees <= 0.0:
        parser.error("turn/look step sizes must be positive")
    viewpoints, plan = compile_viewpoints(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    viewpoints_path = args.output_dir / "interaction_viewpoints.json"
    plan_path = args.output_dir / "compiled_oracle_action_plan.json"
    viewpoints_path.write_text(
        json.dumps(viewpoints, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "interaction_viewpoints": str(viewpoints_path),
                "compiled_oracle_action_plan": str(plan_path),
                "reset_camera_world_position": viewpoints["reset_camera_rig"]["camera_world_position"],
                "reset_camera_yaw_deg": viewpoints["reset_camera_rig"]["camera_yaw_deg"],
                "reset_camera_pitch_deg": viewpoints["reset_camera_rig"]["camera_pitch_deg"],
                "interaction_view_count": viewpoints["coverage"]["interaction_view_count"],
                "all_quantization_residuals_bounded": viewpoints["coverage"]["all_quantization_residuals_bounded"],
                "action_count_excluding_path_internal_moves": plan["action_count_excluding_path_internal_moves"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
