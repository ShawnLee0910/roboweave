"""Run G2 base and G3a torso canaries in one resident Kit scene."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

from behavior_hl.embodied.evaluator_bridge.private_invariants import (
    capture_reset_snapshot,
    compare_reset_snapshots,
)
from behavior_hl.embodied.evaluator_bridge.run_g1_canary import (
    BEHAVIOR_ROOT,
    CONFIG_PATH,
    EFFECTIVE_WRAPPER,
    PROJECT_ROOT,
    _append_jsonl,
    _configure_process_environment,
    _git_head,
    _load_and_validate_config,
    _record_stage,
    _sha256,
    _source_aggregate,
    _validate_launch_contract,
    _validate_video,
    _wait_for_port,
    _write_json,
)
from behavior_hl.embodied.policy.canary_server import (
    SharedCanaryControl,
    server_process_main,
)
from behavior_hl.embodied.policy.observations import schema_sha256
from behavior_hl.embodied.policy.profiles import (
    build_picking_up_toys_registry,
)
from behavior_hl.embodied.policy.r1pro.command_policy import (
    TORSO_PROFILES,
    torso_profile_from_translate,
)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value, dtype=np.float64)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _quaternion_to_rpy(quaternion: Any) -> tuple[float, float, float]:
    x, y, z, w = _array(quaternion).tolist()
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(sin_pitch, -1.0, 1.0)))
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


def _robot_pose(robot: Any) -> dict[str, Any]:
    position, orientation = robot.get_position_orientation()
    roll, pitch, yaw = _quaternion_to_rpy(orientation)
    return {
        "position": _array(position).tolist(),
        "orientation_xyzw": _array(orientation).tolist(),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


def _base_contact(evaluator: Any, rigid_contact_api: Any) -> bool:
    return bool(
        rigid_contact_api.is_in_contact(
            evaluator.env.scene.idx,
            set(evaluator.robot.non_floor_touching_base_links),
            with_set=None,
            ignore_set=None,
            current_only=True,
        )
    )


def _trunk_arm_contact(evaluator: Any, rigid_contact_api: Any) -> bool:
    arm_links = set()
    for links in evaluator.robot.arm_links.values():
        arm_links.update(links)
    return bool(
        rigid_contact_api.is_in_contact(
            evaluator.env.scene.idx,
            set(evaluator.robot.trunk_links),
            with_set=arm_links,
            ignore_set=None,
            current_only=True,
        )
    )


def _local_displacement(
    start_pose: dict[str, Any],
    end_pose: dict[str, Any],
) -> tuple[float, float]:
    delta = (
        np.asarray(end_pose["position"][:2])
        - np.asarray(start_pose["position"][:2])
    )
    yaw = float(start_pose["yaw"])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        float(cosine * delta[0] + sine * delta[1]),
        float(-sine * delta[0] + cosine * delta[1]),
    )


def _load_step_summaries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_suite() -> Path:
    run_config = _load_and_validate_config()
    g2 = run_config["g2"]
    g3a = run_config["g3a"]
    reuse_g2_pass = bool(run_config["runtime"].get("reuse_g2_pass", False))
    if int(g2["collision_contact_probe_steps"]) != int(
        g2["blocked_window_steps"]
    ):
        raise ValueError(
            "collision_contact_probe_steps must equal blocked_window_steps"
        )
    if not (
        0.0
        < float(g2["collision_probe_command"])
        < float(g2["translation_command"])
    ):
        raise ValueError(
            "collision_probe_command must be positive and lower than the "
            "free-motion translation command"
        )
    artifact_root = Path(run_config["runtime"]["artifact_root"])
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_g2_g3a_{os.getpid()}"
    )
    run_dir = artifact_root / run_id
    for relative in ("private", "policy", "video", "logs", "canary"):
        (run_dir / relative).mkdir(parents=True, exist_ok=False)

    _configure_process_environment(run_dir)
    stage_path = run_dir / "canary" / "stages.jsonl"
    checkpoint_path = run_dir / "canary" / "gate_checkpoints.jsonl"
    trace_path = run_dir / "private" / "physical_trace.jsonl"
    reset_path = run_dir / "private" / "reset_invariants.jsonl"
    _record_stage(stage_path, "RUN_DIRECTORY_READY")
    launch_contract = _validate_launch_contract()
    _write_json(run_dir / "private" / "launch_contract.json", launch_contract)
    _record_stage(stage_path, "ZERO_SCENE_LAUNCH_CONTRACT_PASS")

    registry = build_picking_up_toys_registry()
    initial_manifest = {
        "run_id": run_id,
        "status": "RUNNING",
        "gates": ["G3a"] if reuse_g2_pass else ["G2", "G3a"],
        "task": "picking_up_toys",
        "mode": "train",
        "instance_id": 0,
        "simulator_seed": 0,
        "gpu": 0,
        "scene_load_budget": 1,
        "scene_loads_attempted": 0,
        "resident_single_scene": True,
        "wrapper": run_config["runtime"]["wrapper"],
        "effective_wrapper": EFFECTIVE_WRAPPER,
        "config_sha256": _sha256(CONFIG_PATH),
        "registry_sha256": registry.sha256(),
        "observation_schema_sha256": schema_sha256(),
        "embodied_source_sha256": _source_aggregate(
            PROJECT_ROOT / "src" / "behavior_hl" / "embodied"
        ),
        "viewer_camera_control": False,
        "teleport_enabled": False,
        "privileged_policy_inputs": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "manifest.json", initial_manifest)

    context = multiprocessing.get_context("spawn")
    control = SharedCanaryControl(context)
    ready_queue = context.Queue()
    server_process = context.Process(
        target=server_process_main,
        args=(
            str(run_config["g1"]["server_host"]),
            int(run_config["g1"]["server_port"]),
            str(run_dir / "policy"),
            control,
            float(run_config["observation"]["minimum_depth_finite_ratio"]),
            ready_queue,
        ),
        daemon=False,
    )
    server_process.start()
    evaluator_context = None
    scene_loads_attempted = 0
    try:
        host = str(run_config["g1"]["server_host"])
        actual_port = int(ready_queue.get(timeout=20))
        _wait_for_port(host, actual_port)
        _record_stage(
            stage_path,
            "POLICY_SERVER_READY",
            host=host,
            port=actual_port,
        )

        _record_stage(stage_path, "OFFICIAL_RUNTIME_IMPORT_START")
        from omegaconf import OmegaConf
        from omnigibson.eval.evaluator import Evaluator
        from omnigibson.eval.utils.eval_utils import seed_everything
        from omnigibson.macros import gm
        from omnigibson.utils.usd_utils import RigidContactAPI

        expected_appdata = Path(
            os.environ["OMNIGIBSON_APPDATA_PATH"]
        ).resolve()
        if Path(gm.APPDATA_PATH).resolve() != expected_appdata:
            raise RuntimeError("OmniGibson appdata escaped isolation")
        _record_stage(stage_path, "OFFICIAL_RUNTIME_IMPORT_PASS")

        gm.HEADLESS = True
        seed_everything(0)
        cfg = OmegaConf.create(
            {
                "env_wrapper": {"_target_": EFFECTIVE_WRAPPER},
                "policy_name": "behavior_hl_resident_public_canary",
                "model": {
                    "_target_": (
                        "behavior_hl.embodied.evaluator_bridge."
                        "whitelist_policy.WhitelistedWebsocketPolicy"
                    ),
                    "host": host,
                    "port": actual_port,
                    "allow_reconnect": False,
                },
                "headless": True,
                "partial_scene_load": run_config["runtime"][
                    "partial_scene_load"
                ],
                "max_steps": None,
                "write_video": True,
                "mode": "train",
                "seed": 0,
                "task": {"name": "picking_up_toys"},
                "robot": None,
            }
        )
        scene_loads_attempted = 1
        _record_stage(
            stage_path,
            "SCENE_CONSTRUCTION_START",
            scene_load_number=1,
        )
        evaluator_context = Evaluator(cfg)
        with nullcontext(evaluator_context.__enter__()) as evaluator:
            _record_stage(
                stage_path,
                "SCENE_CONSTRUCTION_PASS",
                scene_load_number=1,
            )
            timeout_steps = int(
                evaluator.env.task._termination_conditions["timeout"]._max_steps
            )
            if int(evaluator.robot.action_dim) != 23:
                raise RuntimeError("R1Pro action dimension is not 23")
            if timeout_steps != 28_335:
                raise RuntimeError(
                    f"unexpected resolved timeout: {timeout_steps}"
                )

            control.hold()
            evaluator.reset()
            evaluator.load_task_instance(0)
            evaluator.reset()
            baseline_snapshot = capture_reset_snapshot(evaluator)
            _append_jsonl(
                reset_path,
                {
                    "label": "baseline_after_task_instance_load",
                    "snapshot": baseline_snapshot,
                },
            )
            _record_stage(stage_path, "SUITE_INITIALIZATION_PASS")

            reset_cfg = g2["reset_invariant"]

            def reset_and_check(label: str) -> None:
                control.hold()
                evaluator.reset()
                actual = capture_reset_snapshot(evaluator)
                report = compare_reset_snapshots(
                    baseline_snapshot,
                    actual,
                    position_tolerance_m=float(
                        reset_cfg["position_tolerance_m"]
                    ),
                    orientation_tolerance_rad=float(
                        reset_cfg["orientation_tolerance_rad"]
                    ),
                    joint_tolerance_rad=float(
                        reset_cfg["joint_tolerance_rad"]
                    ),
                )
                _append_jsonl(
                    reset_path,
                    {
                        "label": label,
                        "report": report,
                        "snapshot": actual,
                    },
                )
                if not report["passed"]:
                    raise RuntimeError(
                        f"private reset invariant failed at {label}"
                    )

            suite_step = 0

            def one_step(segment: str) -> dict[str, Any]:
                nonlocal suite_step
                terminated, truncated = evaluator.step()
                pose = _robot_pose(evaluator.robot)
                public_state = control.public_state()
                record = {
                    "suite_step": suite_step,
                    "segment": segment,
                    "robot_pose": pose,
                    "public_state": public_state,
                    "base_contact": _base_contact(
                        evaluator,
                        RigidContactAPI,
                    ),
                    "trunk_arm_contact": _trunk_arm_contact(
                        evaluator,
                        RigidContactAPI,
                    ),
                }
                _append_jsonl(trace_path, record)
                suite_step += 1
                if terminated or truncated:
                    raise RuntimeError(
                        "task terminated during physical canary"
                    )
                return record

            def hold_steps(segment: str, steps: int) -> list[dict[str, Any]]:
                control.hold()
                return [one_step(segment) for _ in range(steps)]

            video_path = run_dir / "video" / "official_rollout.mp4"
            evaluator.start_recording(
                str(video_path),
                rate=int(run_config["runtime"]["video_fps"]),
            )
            try:
                _record_stage(stage_path, "G2_START")
                translation_command = float(g2["translation_command"])
                rotation_command = float(g2["rotation_command"])
                motion_definitions = () if reuse_g2_pass else (
                    (
                        "move_forward",
                        (translation_command, 0.0, 0.0),
                        "x",
                        1.0,
                    ),
                    (
                        "move_backward",
                        (-translation_command, 0.0, 0.0),
                        "x",
                        -1.0,
                    ),
                    (
                        "move_left",
                        (0.0, translation_command, 0.0),
                        "y",
                        1.0,
                    ),
                    (
                        "move_right",
                        (0.0, -translation_command, 0.0),
                        "y",
                        -1.0,
                    ),
                    (
                        "turn_left",
                        (0.0, 0.0, rotation_command),
                        "yaw",
                        1.0,
                    ),
                    (
                        "turn_right",
                        (0.0, 0.0, -rotation_command),
                        "yaw",
                        -1.0,
                    ),
                )
                motion_results = {}
                for name, command, axis, expected_sign in motion_definitions:
                    reset_and_check(f"before_{name}")
                    hold_steps(
                        f"{name}.pre_hold",
                        int(g2["stop_steps"]),
                    )
                    start_pose = _robot_pose(evaluator.robot)
                    control.command_base(command)
                    steps = (
                        int(g2["rotation_steps"])
                        if axis == "yaw"
                        else int(g2["translation_steps"])
                    )
                    trace = [one_step(name) for _ in range(steps)]
                    hold_steps(
                        f"{name}.post_hold",
                        int(g2["stop_steps"]),
                    )
                    end_pose = _robot_pose(evaluator.robot)
                    local_x, local_y = _local_displacement(
                        start_pose,
                        end_pose,
                    )
                    yaw_delta = _wrap_angle(
                        float(end_pose["yaw"]) - float(start_pose["yaw"])
                    )
                    values = {
                        "x": local_x,
                        "y": local_y,
                        "yaw": yaw_delta,
                    }
                    signed_progress = expected_sign * values[axis]
                    threshold = (
                        float(g2["minimum_rotation_rad"])
                        if axis == "yaw"
                        else float(g2["minimum_translation_m"])
                    )
                    max_z_drift = max(
                        abs(
                            record["robot_pose"]["position"][2]
                            - start_pose["position"][2]
                        )
                        for record in trace
                    )
                    max_roll_pitch = max(
                        max(
                            abs(record["robot_pose"]["roll"]),
                            abs(record["robot_pose"]["pitch"]),
                        )
                        for record in trace
                    )
                    public_qvel = np.asarray(
                        [
                            record["public_state"]["base_qvel"]
                            for record in trace
                        ],
                        dtype=np.float64,
                    )
                    passed = (
                        signed_progress >= threshold
                        and max_z_drift
                        <= float(g2["maximum_z_drift_m"])
                        and max_roll_pitch
                        <= float(g2["maximum_roll_pitch_rad"])
                    )
                    motion_results[name] = {
                        "status": "PASS" if passed else "FAIL",
                        "command": list(command),
                        "local_displacement_m": [local_x, local_y],
                        "yaw_delta_rad": yaw_delta,
                        "signed_progress": signed_progress,
                        "required_progress": threshold,
                        "max_z_drift_m": max_z_drift,
                        "max_abs_roll_pitch_rad": max_roll_pitch,
                        "mean_public_base_qvel": public_qvel.mean(
                            axis=0
                        ).tolist(),
                        "integrated_public_base_qvel": (
                            public_qvel.sum(axis=0)
                            / float(g2["command_frequency_hz"])
                        ).tolist(),
                    }
                    if not passed:
                        raise RuntimeError(f"G2 motion failed: {name}")

                reset_and_check("before_blocked_collision")
                hold_steps(
                    "blocked_collision.pre_hold",
                    int(g2["stop_steps"]),
                )
                collision_start_pose = _robot_pose(evaluator.robot)
                if not reuse_g2_pass:
                    control.command_base(
                        (float(g2["collision_probe_command"]), 0.0, 0.0)
                    )
                window_size = int(g2["blocked_window_steps"])
                window: deque[dict[str, Any]] = deque(maxlen=window_size)
                blocked_result = (
                    {
                        "status": "REUSED_PASS",
                        "evidence_run": "20260730T124848Z_g2_g3a_65680",
                        "new_motion_steps": 0,
                    }
                    if reuse_g2_pass
                    else None
                )
                contact_started = False
                first_contact_clearance = None
                slowed_for_clearance = False
                blocked_search_steps = (
                    0
                    if reuse_g2_pass
                    else int(g2["blocked_search_max_steps"])
                )
                for _ in range(blocked_search_steps):
                    record = one_step("blocked_collision")
                    tilt = max(
                        abs(record["robot_pose"]["roll"]),
                        abs(record["robot_pose"]["pitch"]),
                    )
                    z_drift = abs(
                        record["robot_pose"]["position"][2]
                        - collision_start_pose["position"][2]
                    )
                    if (
                        tilt > float(g2["maximum_roll_pitch_rad"])
                        or z_drift > float(g2["maximum_z_drift_m"])
                    ):
                        control.hold()
                        raise RuntimeError(
                            "G2 collision safety guard stopped unsafe "
                            f"tilt/z drift: tilt={tilt:.6f}, "
                            f"z_drift={z_drift:.6f}"
                        )

                    clearance = float(
                        record["public_state"][
                            "head_forward_clearance_m"
                        ]
                    )
                    if (
                        not contact_started
                        and not slowed_for_clearance
                        and clearance
                        <= float(g2["collision_slow_clearance_m"])
                    ):
                        # The collision approach is already constrained to
                        # collision_probe_command.  This flag proves that the
                        # public head-depth signal was available without
                        # changing the physical command.
                        slowed_for_clearance = True

                    if not contact_started and record["base_contact"]:
                        contact_started = True
                        first_contact_clearance = clearance
                        control.command_base(
                            (
                                float(g2["collision_probe_command"]),
                                0.0,
                                0.0,
                            )
                        )
                        # The action that produced first contact was selected
                        # before contact was observable.  Do not count that
                        # higher-speed step in the bounded low-speed probe.
                        # The next evaluator step is the first one commanded
                        # at collision_probe_command.
                        continue
                    if not contact_started:
                        continue
                    window.append(record)
                    if len(window) < int(
                        g2["collision_contact_probe_steps"]
                    ):
                        continue

                    public_speed = float(
                        np.mean(
                            [
                                np.linalg.norm(
                                    item["public_state"]["base_qvel"][:2]
                                )
                                for item in window
                            ]
                        )
                    )
                    first_position = np.asarray(
                        window[0]["robot_pose"]["position"][:2]
                    )
                    last_position = np.asarray(
                        window[-1]["robot_pose"]["position"][:2]
                    )
                    displacement = float(
                        np.linalg.norm(last_position - first_position)
                    )
                    contact = any(
                        bool(item["base_contact"]) for item in window
                    )
                    control.hold()
                    if (
                        public_speed
                        <= float(g2["blocked_max_public_speed"])
                        and displacement
                        <= float(g2["blocked_max_displacement_m"])
                        and contact
                    ):
                        blocked_result = {
                            "status": "PASS",
                            "suite_step": suite_step,
                            "window_steps": window_size,
                            "mean_public_speed": public_speed,
                            "private_displacement_m": displacement,
                            "private_base_contact": contact,
                            "slowed_from_public_depth": (
                                slowed_for_clearance
                            ),
                            "first_contact_clearance_m": (
                                first_contact_clearance
                            ),
                        }
                    else:
                        raise RuntimeError(
                            "G2 real contact did not produce a safe public "
                            "blocked signal within the bounded low-speed "
                            f"probe: speed={public_speed:.6f}, "
                            f"displacement={displacement:.6f}"
                        )
                    break
                hold_steps(
                    "blocked_collision.post_hold",
                    int(g2["stop_steps"]),
                )
                if blocked_result is None:
                    raise RuntimeError(
                        "G2 failed to observe contact-confirmed blocked motion"
                    )
                g2_result = {
                    "status": "REUSED_PASS" if reuse_g2_pass else "PASS",
                    "motion": motion_results,
                    "blocked_collision": blocked_result,
                    "base_qvel_calibration": {
                        "command_frame": "robot_local",
                        "observed_proprio_frame": (
                            "canonical_joint_velocity_pending_numeric_trace"
                        ),
                        "frequency_hz": int(
                            g2["command_frequency_hz"]
                        ),
                    },
                }
                _write_json(run_dir / "private" / "g2_result.json", g2_result)
                _append_jsonl(
                    checkpoint_path,
                    {
                        "gate": "G2",
                        "status": (
                            "REUSED_PASS" if reuse_g2_pass else "PASS"
                        ),
                        "end_step": suite_step,
                        "scene_load_number": 1,
                        "result_path": "private/g2_result.json",
                    },
                )
                _record_stage(
                    stage_path,
                    "G2_REUSED_PASS" if reuse_g2_pass else "G2_PASS",
                    end_step=suite_step,
                )

                _record_stage(stage_path, "G3A_START")
                reset_and_check("g2_to_g3a")
                hold_steps(
                    "g3a.neutral_warmup",
                    int(g3a["settle_steps"]),
                )
                initial_joints = _array(
                    evaluator.robot.get_joint_positions()
                )
                profile_results: dict[str, Any] = {}
                capture_ids = {
                    "neutral_start": 1,
                    "incremental_down": 2,
                    "neutral_mid": 3,
                    "incremental_up": 4,
                    "neutral_end": 5,
                }
                profile_sequence = (
                    ("neutral_start", "neutral"),
                    ("incremental_down", "incremental_down"),
                    ("neutral_mid", "neutral"),
                    ("incremental_up", "incremental_up"),
                    ("neutral_end", "neutral"),
                )
                g3a_start_pose = _robot_pose(evaluator.robot)
                for label, profile_name in profile_sequence:
                    translate = float(g3a["translate"][profile_name])
                    target = torso_profile_from_translate(translate)
                    expected_neutral = TORSO_PROFILES["neutral"]
                    if (
                        profile_name == "neutral"
                        and not np.allclose(target, expected_neutral)
                    ):
                        raise RuntimeError(
                            "official neutral torso interpolation drift"
                        )
                    if not 0.0 <= translate <= 1.0:
                        raise RuntimeError(
                            f"unsafe torso translate: {profile_name}"
                        )
                    configured_neutral = float(
                        g3a["translate"]["neutral"]
                    )
                    if (
                        profile_name == "incremental_down"
                        and translate <= configured_neutral
                    ) or (
                        profile_name == "incremental_up"
                        and translate >= configured_neutral
                    ):
                        raise RuntimeError(
                            f"torso translate direction drift: {profile_name}"
                        )
                    control.command_torso(
                        target,
                        max_delta_per_step=float(
                            g3a["max_torso_delta_per_step"]
                        ),
                    )
                    consecutive = 0
                    trace = []
                    for _ in range(int(g3a["max_profile_steps"])):
                        record = one_step(f"g3a.{label}.move")
                        trace.append(record)
                        tilt = max(
                            abs(record["robot_pose"]["roll"]),
                            abs(record["robot_pose"]["pitch"]),
                        )
                        z_drift = abs(
                            record["robot_pose"]["position"][2]
                            - g3a_start_pose["position"][2]
                        )
                        if (
                            tilt
                            > float(g3a["maximum_roll_pitch_rad"])
                            or z_drift > float(g3a["maximum_z_drift_m"])
                            or bool(record["trunk_arm_contact"])
                        ):
                            control.hold()
                            raise RuntimeError(
                                "G3a safety guard stopped torso motion: "
                                f"label={label}, tilt={tilt:.6f}, "
                                f"z_drift={z_drift:.6f}, "
                                "trunk_arm_contact="
                                f"{record['trunk_arm_contact']}"
                            )
                        error = float(
                            np.max(
                                np.abs(
                                    np.asarray(
                                        record["public_state"][
                                            "trunk_qpos"
                                        ]
                                    )
                                    - target
                                )
                            )
                        )
                        consecutive = (
                            consecutive + 1
                            if error
                            <= float(g3a["profile_tolerance_rad"])
                            else 0
                        )
                        if consecutive >= 3:
                            break
                    else:
                        raise RuntimeError(
                            f"G3a profile did not converge: {label}"
                        )
                    for _ in range(int(g3a["settle_steps"])):
                        trace.append(one_step(f"g3a.{label}.settle"))
                    control.request_capture(capture_ids[label])
                    trace.append(one_step(f"g3a.{label}.capture"))
                    final_error = float(
                        np.max(
                            np.abs(
                                np.asarray(
                                    trace[-1]["public_state"]["trunk_qpos"]
                                )
                                - target
                            )
                        )
                    )
                    trunk_arm_contact = any(
                        bool(item["trunk_arm_contact"]) for item in trace
                    )
                    profile_results[label] = {
                        "profile": profile_name,
                        "translate": translate,
                        "target": target.astype(float).tolist(),
                        "steps": len(trace),
                        "final_error_rad": final_error,
                        "trunk_arm_contact": trunk_arm_contact,
                        "capture": (
                            f"policy/captures/head_rgb_"
                            f"{capture_ids[label]:03d}.npy"
                        ),
                    }
                    if trunk_arm_contact:
                        raise RuntimeError(
                            f"G3a trunk-arm contact: {label}"
                        )

                final_joints = _array(evaluator.robot.get_joint_positions())
                arm_indices = np.concatenate(
                    [
                        _array(evaluator.robot.arm_control_idx[arm]).astype(int)
                        for arm in evaluator.robot.arm_names
                    ]
                )
                max_arm_drift = float(
                    np.max(
                        np.abs(
                            final_joints[arm_indices]
                            - initial_joints[arm_indices]
                        )
                    )
                )
                captures = {
                    label: np.load(
                        run_dir
                        / "policy"
                        / "captures"
                        / f"head_rgb_{capture_ids[label]:03d}.npy",
                        allow_pickle=False,
                    ).astype(np.float32)
                    for label in capture_ids
                }
                incremental_down_mad = float(
                    np.mean(
                        np.abs(
                            captures["incremental_down"]
                            - captures["neutral_start"]
                        )
                    )
                )
                incremental_up_mad = float(
                    np.mean(
                        np.abs(
                            captures["incremental_up"]
                            - captures["neutral_mid"]
                        )
                    )
                )
                neutral_return_mad = float(
                    np.mean(
                        np.abs(
                            captures["neutral_end"]
                            - captures["neutral_start"]
                        )
                    )
                )
                g3a_passed = (
                    incremental_down_mad
                    >= float(g3a["minimum_head_rgb_mad"])
                    and incremental_up_mad >= float(
                        g3a["minimum_head_rgb_mad"]
                    )
                    and max_arm_drift
                    <= float(g3a["maximum_arm_drift_rad"])
                )
                g3a_result = {
                    "status": "PASS" if g3a_passed else "FAIL",
                    "profiles": profile_results,
                    "head_rgb_mad": {
                        "incremental_down_vs_neutral": incremental_down_mad,
                        "incremental_up_vs_neutral": incremental_up_mad,
                        "neutral_end_vs_start": neutral_return_mad,
                        "minimum_required": float(
                            g3a["minimum_head_rgb_mad"]
                        ),
                    },
                    "max_arm_joint_drift_rad": max_arm_drift,
                    "viewer_camera_control": False,
                    "physical_torso_only": True,
                    "g3_status": "PARTIAL",
                    "g3b_required_after_tennis_ball_grasp": True,
                }
                _write_json(
                    run_dir / "private" / "g3a_result.json",
                    g3a_result,
                )
                if not g3a_passed:
                    raise RuntimeError("G3a visual/joint checks failed")
                _append_jsonl(
                    checkpoint_path,
                    {
                        "gate": "G3a",
                        "status": "PASS",
                        "end_step": suite_step,
                        "scene_load_number": 1,
                        "result_path": "private/g3a_result.json",
                    },
                )
                _record_stage(stage_path, "G3A_PASS", end_step=suite_step)
            finally:
                control.hold()
                evaluator.stop_recording()

            video = _validate_video(video_path)
            public_steps = _load_step_summaries(
                run_dir / "policy" / "public_step_summaries.jsonl"
            )
            result = {
                "status": "PASS",
                "g2": "REUSED_PASS" if reuse_g2_pass else "PASS",
                "g3a": "PASS",
                "g3": "PARTIAL",
                "scene_loads_attempted": 1,
                "suite_steps": suite_step,
                "public_observation_messages": len(public_steps),
                "resolved_timeout_steps": timeout_steps,
                "video": video,
            }
            _write_json(run_dir / "private" / "suite_result.json", result)
            final_manifest = {
                **initial_manifest,
                "status": "PASS",
                "gate_status": {
                    "G2": "REUSED_PASS" if reuse_g2_pass else "PASS",
                    "G3a": "PASS",
                    "G3": "PARTIAL",
                },
                "scene_loads_attempted": scene_loads_attempted,
                "suite_steps": suite_step,
                "resolved_timeout_steps": timeout_steps,
                "video_sha256": video["sha256"],
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "source_files": {
                    "behavior_head": _git_head(BEHAVIOR_ROOT),
                    "config_sha256": _sha256(CONFIG_PATH),
                    "runner_sha256": _sha256(Path(__file__)),
                },
            }
            _write_json(run_dir / "manifest.json", final_manifest)
            _record_stage(
                stage_path,
                "RESULTS_PERSISTED_BEFORE_EVALUATOR_EXIT",
                status="PASS",
            )
            return run_dir
    except Exception as exc:
        _record_stage(
            stage_path,
            "PYTHON_EXCEPTION",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        _write_json(
            run_dir / "failure.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        _write_json(
            run_dir / "manifest.json",
            {
                **initial_manifest,
                "status": "FAIL",
                "scene_loads_attempted": scene_loads_attempted,
                "failure_type": type(exc).__name__,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    finally:
        if server_process.is_alive():
            server_process.terminate()
        server_process.join(timeout=10)
        if evaluator_context is not None:
            evaluator_context.__exit__(*sys.exc_info())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(run_suite())


if __name__ == "__main__":
    main()
