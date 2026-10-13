"""Observation-only policy server for resident physical-action canaries."""

from __future__ import annotations

import asyncio
import hashlib
import http
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import websockets
import websockets.asyncio.server as websocket_server

from behavior_hl.embodied.policy.network_codec import packb, unpackb
from behavior_hl.embodied.policy.observations import (
    HEAD_RGB_MINIMAL_OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    THREE_RGB_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION,
    THREE_RGBD_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION,
    TASK_ID,
    head_rgb_minimal_schema_sha256,
    schema_sha256,
    three_rgb_head_eval_schema_sha256,
    validate_head_rgb_minimal_observation,
    validate_public_observation,
    validate_three_rgb_head_eval_observation,
)
from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    FULL_CAMERA_PROFILE,
    HEAD_RGB_MINIMAL_CAMERA_PROFILE,
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
)
from behavior_hl.embodied.evaluator_bridge.oracle_websocket_keepalive import (
    oracle_websocket_keepalive_manifest,
    oracle_websocket_server_options,
)
from behavior_hl.embodied.policy.profiles import (
    build_picking_up_toys_registry,
)
from behavior_hl.embodied.policy.r1pro.command_policy import (
    TORSO_PROFILES,
    build_public_command_action,
)


MODE_HOLD = 0
MODE_BASE = 1
MODE_TORSO = 2
MODE_DIRECT_ACTION = 3
MODE_CARRY_ACTION = 4
MODE_NAVIGATION_ACTION = 5
MODE_NAMES = {
    MODE_HOLD: "hold",
    MODE_BASE: "base",
    MODE_TORSO: "torso",
    MODE_DIRECT_ACTION: "direct_action",
    MODE_CARRY_ACTION: "carry",
    MODE_NAVIGATION_ACTION: "official_starter_navigation",
}


class SharedCanaryControl:
    """Small synchronized control plane shared with the canary orchestrator."""

    def __init__(self, context) -> None:
        self.mode = context.Value("i", MODE_HOLD)
        self.base_command = context.Array("d", [0.0, 0.0, 0.0])
        self.torso_target = context.Array(
            "d",
            TORSO_PROFILES["neutral"].astype(float).tolist(),
        )
        self.direct_action = context.Array("d", [0.0] * 23)
        self.carry_hand = context.Value("i", -1)
        self.max_torso_delta = context.Value("d", 0.025)
        self.capture_id = context.Value("i", 0)
        self.snapshot_id = context.Value("i", 0)
        self.observation_count = context.Value("i", 0)
        self.last_base_qvel = context.Array("d", [0.0, 0.0, 0.0])
        self.last_trunk_qpos = context.Array("d", [0.0, 0.0, 0.0, 0.0])
        self.last_head_forward_clearance = context.Value("d", float("inf"))
        # A resident scene batch may switch the active BDDL task without
        # restarting this policy process.  Keep the public task-id validator
        # synchronized through the existing multiprocessing control plane.
        self.expected_task_id = context.Value("i", -1)

    def set_expected_task_id(self, task_id: int) -> None:
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
            raise ValueError("expected task id must be a non-negative integer")
        with self.expected_task_id.get_lock():
            self.expected_task_id.value = task_id

    def current_expected_task_id(self, fallback: int) -> int:
        with self.expected_task_id.get_lock():
            value = int(self.expected_task_id.value)
        return fallback if value < 0 else value

    @staticmethod
    def _write_array(target, values) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(target),) or not np.isfinite(array).all():
            raise ValueError("shared control vector is invalid")
        with target.get_lock():
            target[:] = array.tolist()

    @staticmethod
    def _read_array(source) -> np.ndarray:
        with source.get_lock():
            return np.asarray(source[:], dtype=np.float32)

    def hold(self) -> None:
        self._write_array(self.base_command, (0.0, 0.0, 0.0))
        with self.mode.get_lock():
            self.mode.value = MODE_HOLD

    def command_base(self, command) -> None:
        self._write_array(self.base_command, command)
        with self.mode.get_lock():
            self.mode.value = MODE_BASE

    def command_torso(
        self,
        target,
        *,
        max_delta_per_step: float = 0.025,
    ) -> None:
        self._write_array(self.base_command, (0.0, 0.0, 0.0))
        self._write_array(self.torso_target, target)
        if (
            not np.isfinite(max_delta_per_step)
            or max_delta_per_step <= 0.0
            or max_delta_per_step > 0.05
        ):
            raise ValueError("invalid torso interpolation rate")
        with self.max_torso_delta.get_lock():
            self.max_torso_delta.value = float(max_delta_per_step)
        with self.mode.get_lock():
            self.mode.value = MODE_TORSO

    def command_direct_action(self, action) -> None:
        self._write_array(self.direct_action, action)
        with self.mode.get_lock():
            self.mode.value = MODE_DIRECT_ACTION

    def command_navigation_action(self, action) -> None:
        """Forward one private-Oracle official Starter BASE action.

        ``command_direct_action`` deliberately suppresses base motion because
        it is the arm-local manipulation lane.  Official Starter navigation
        also emits a full 23D action, so it needs an explicit, separately
        auditable mode which preserves the three base components instead of
        silently erasing them.  This method is not exposed to the model action
        registry.
        """

        self._write_array(self.direct_action, action)
        with self.mode.get_lock():
            self.mode.value = MODE_NAVIGATION_ACTION

    def command_carry_action(self, action, *, active_hand: str) -> None:
        """Command base+body jointly while forcing the active hand closed."""

        if active_hand not in {"left", "right"}:
            raise ValueError("active_hand must be left or right")
        candidate = np.asarray(action, dtype=np.float64)
        if candidate.shape != (23,) or not np.isfinite(candidate).all():
            raise ValueError("carry physical action must be finite and 23D")
        gripper_index = 14 if active_hand == "left" else 22
        if candidate[gripper_index] > -0.5:
            raise ValueError("carry action must keep the active gripper closed")
        self._write_array(self.direct_action, candidate)
        with self.carry_hand.get_lock():
            self.carry_hand.value = 0 if active_hand == "left" else 1
        with self.mode.get_lock():
            self.mode.value = MODE_CARRY_ACTION

    def request_capture(self, capture_id: int) -> None:
        if capture_id <= 0:
            raise ValueError("capture_id must be positive")
        with self.capture_id.get_lock():
            self.capture_id.value = capture_id

    def request_public_snapshot(self, snapshot_id: int) -> None:
        if snapshot_id <= 0:
            raise ValueError("snapshot_id must be positive")
        with self.snapshot_id.get_lock():
            self.snapshot_id.value = snapshot_id

    def public_state(self) -> dict[str, Any]:
        with self.observation_count.get_lock():
            observation_count = int(self.observation_count.value)
        with self.last_head_forward_clearance.get_lock():
            head_forward_clearance = float(
                self.last_head_forward_clearance.value
            )
        return {
            "observation_count": observation_count,
            "base_qvel": self._read_array(self.last_base_qvel).tolist(),
            "trunk_qpos": self._read_array(self.last_trunk_qpos).tolist(),
            "head_forward_clearance_m": head_forward_clearance,
        }


class ResidentCanaryState:
    def __init__(
        self,
        artifact_dir: Path,
        control: SharedCanaryControl,
        *,
        minimum_depth_finite_ratio: float,
        observation_profile: str = FULL_CAMERA_PROFILE,
        expected_task_id: int = TASK_ID,
    ) -> None:
        if observation_profile not in {
            FULL_CAMERA_PROFILE,
            HEAD_RGB_MINIMAL_CAMERA_PROFILE,
            THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
        }:
            raise ValueError("unknown canary observation profile")
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.capture_dir = self.artifact_dir / "captures"
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.artifact_dir / "server_events.jsonl"
        self.steps_path = self.artifact_dir / "public_step_summaries.jsonl"
        self.control = control
        self.minimum_depth_finite_ratio = minimum_depth_finite_ratio
        self.observation_profile = observation_profile
        self.expected_task_id = expected_task_id
        self.reset_epoch = -1
        self.observation_index = 0
        self.total_observations = 0
        self.last_capture_id = 0
        self.last_snapshot_id = 0

    def reset(self) -> None:
        self.reset_epoch += 1
        self.observation_index = 0
        self._append(
            self.events_path,
            {
                "event": "reset",
                "reset_epoch": self.reset_epoch,
                "monotonic_s": time.monotonic(),
            },
        )

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        expected_task_id = self.control.current_expected_task_id(
            self.expected_task_id
        )
        if self.observation_profile == HEAD_RGB_MINIMAL_CAMERA_PROFILE:
            if "simulator_step" not in observation:
                raise ValueError("head-RGB observation omits simulator_step")
            validate_head_rgb_minimal_observation(
                {
                    key: value
                    for key, value in observation.items()
                    if key != "simulator_step"
                },
                expected_task_id=expected_task_id,
            )
        elif self.observation_profile == THREE_RGB_HEAD_EVAL_CAMERA_PROFILE:
            validate_three_rgb_head_eval_observation(
                observation,
                require_simulator_step=True,
                expected_task_id=expected_task_id,
            )
        elif self.observation_profile == THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE:
            validate_public_observation(
                observation,
                require_simulator_step=True,
                expected_task_id=expected_task_id,
                head_eval=True,
            )
        else:
            validate_public_observation(
                observation,
                require_simulator_step=True,
                expected_task_id=expected_task_id,
            )
        if observation["simulator_step"] != self.observation_index:
            raise ValueError("public simulator_step is not reset-consistent")
        proprio = np.asarray(observation["proprio"], dtype=np.float32)
        base_qvel = proprio[:3]
        trunk_qpos = proprio[53:57]
        if not np.isfinite(base_qvel).all() or not np.isfinite(
            trunk_qpos
        ).all():
            raise ValueError("public proprio contains non-finite state")

        with self.control.mode.get_lock():
            mode = int(self.control.mode.value)
        base_command = self.control._read_array(self.control.base_command)
        torso_target = self.control._read_array(self.control.torso_target)
        with self.control.max_torso_delta.get_lock():
            max_torso_delta = float(self.control.max_torso_delta.value)
        carry_hand_label = None
        active_gripper_command = None
        if mode == MODE_HOLD:
            base_command = np.zeros(3, dtype=np.float32)
            active_torso_target = None
        elif mode == MODE_BASE:
            active_torso_target = None
        elif mode == MODE_TORSO:
            base_command = np.zeros(3, dtype=np.float32)
            active_torso_target = torso_target
        elif mode in {
            MODE_DIRECT_ACTION,
            MODE_CARRY_ACTION,
            MODE_NAVIGATION_ACTION,
        }:
            action = self.control._read_array(self.control.direct_action)
            if action.shape != (23,) or not np.isfinite(action).all():
                raise ValueError("direct physical action must be finite and 23D")
            if mode == MODE_DIRECT_ACTION:
                # The image-point manipulation executor is arm-local. Never
                # allow its full joint target to move the velocity base or
                # silently change the camera-bearing torso.
                action[:3] = 0.0
                action[3:7] = proprio[53:57]
            elif mode == MODE_CARRY_ACTION:
                with self.control.carry_hand.get_lock():
                    carry_hand = int(self.control.carry_hand.value)
                if carry_hand not in {0, 1}:
                    raise ValueError("carry action has no active hand")
                gripper_index = 14 if carry_hand == 0 else 22
                carry_hand_label = "left" if carry_hand == 0 else "right"
                # Recheck at the server boundary after shared-memory transfer.
                if action[gripper_index] > -0.5:
                    raise ValueError(
                        "carry action attempted to open the active gripper"
                    )
                active_gripper_command = float(action[gripper_index])
            else:
                # Private Oracle only: retain the complete official Starter
                # BASE command.  The planner supplies collision-checked base
                # waypoints and current articulation targets.
                pass
            active_torso_target = None
        else:
            raise ValueError(f"unknown canary control mode: {mode}")

        if mode not in {
            MODE_DIRECT_ACTION,
            MODE_CARRY_ACTION,
            MODE_NAVIGATION_ACTION,
        }:
            action = build_public_command_action(
                observation,
                base_command=base_command,
                torso_target=active_torso_target,
                max_torso_delta_per_step=max_torso_delta,
                observation_profile=self.observation_profile,
                expected_task_id=expected_task_id,
            )
        head = observation["cameras"]["head"]
        head_rgb = np.asarray(head["rgb"], dtype=np.uint8)
        head_depth_value = head.get("depth_linear")
        if head_depth_value is None:
            head_depth = None
            depth_finite_ratio = None
            depth_quality_degraded = False
        else:
            head_depth = np.asarray(head_depth_value, dtype=np.float32)
            finite_depth = np.isfinite(head_depth)
            depth_finite_ratio = (
                float(finite_depth.mean()) if head_depth.size else 0.0
            )
            depth_quality_degraded = (
                depth_finite_ratio < self.minimum_depth_finite_ratio
            )
        # A transient incomplete depth render must not tear down the resident
        # websocket and discard an otherwise valid rollout.  Depth is not
        # needed for HOLD / torso / arm-only settling.  When a base command is
        # active, fail closed at the actuator boundary for this frame by
        # freezing the base.  The next observation is still delivered to the
        # policy, with no privileged textual explanation or simulator truth.
        base_command_suppressed_due_to_depth_quality = bool(
            depth_quality_degraded
            and np.linalg.norm(np.asarray(action[:3], dtype=np.float32))
            > 1e-8
        )
        if depth_quality_degraded:
            action[:3] = 0.0
            # Keep the persisted command evidence aligned with the action
            # actually returned across the websocket boundary.
            base_command = np.zeros(3, dtype=np.float32)
        if head_depth is None:
            center_depth = None
            forward_values = np.asarray([], dtype=np.float32)
        else:
            center_y = head_depth.shape[0] // 2
            center_x = head_depth.shape[1] // 2
            center_window = head_depth[
                center_y - 2 : center_y + 3,
                center_x - 2 : center_x + 3,
            ]
            center_values = center_window[np.isfinite(center_window)]
            center_depth = (
                float(np.median(center_values))
                if center_values.size
                else None
            )
            height, width = head_depth.shape[:2]
            forward_region = head_depth[
                int(height * 0.45) : int(height * 0.75),
                int(width * 0.40) : int(width * 0.60),
            ]
            forward_values = forward_region[np.isfinite(forward_region)]
        # A degraded depth image cannot justify positive free space.  Publish
        # a conservative finite zero so downstream fail-closed actuator
        # validation remains usable instead of poisoning the resident worker
        # with infinity.  A later valid frame naturally replaces this value.
        forward_clearance = (
            0.0
            if depth_quality_degraded
            else float(np.percentile(forward_values, 10.0))
            if forward_values.size
            else 0.0
        )

        with self.control.last_base_qvel.get_lock():
            self.control.last_base_qvel[:] = base_qvel.astype(float).tolist()
        with self.control.last_trunk_qpos.get_lock():
            self.control.last_trunk_qpos[:] = trunk_qpos.astype(float).tolist()
        with self.control.observation_count.get_lock():
            self.control.observation_count.value += 1
        with self.control.last_head_forward_clearance.get_lock():
            self.control.last_head_forward_clearance.value = (
                forward_clearance
            )
        with self.control.capture_id.get_lock():
            capture_id = int(self.control.capture_id.value)
        captured = None
        if capture_id > 0 and capture_id != self.last_capture_id:
            captured = f"captures/head_rgb_{capture_id:03d}.npy"
            np.save(self.artifact_dir / captured, head_rgb, allow_pickle=False)
            self.last_capture_id = capture_id
        with self.control.snapshot_id.get_lock():
            snapshot_id = int(self.control.snapshot_id.value)
        snapshot = None
        if snapshot_id > 0 and snapshot_id != self.last_snapshot_id:
            snapshot = f"captures/public_snapshot_{snapshot_id:03d}.npz"
            arrays = {
                "proprio": np.asarray(
                    observation["proprio"],
                    dtype=np.float32,
                ),
                "cam_rel_poses": np.asarray(
                    observation["cam_rel_poses"],
                    dtype=np.float32,
                ),
            }
            for role, camera in observation["cameras"].items():
                arrays[f"{role}_rgb"] = np.asarray(
                    camera["rgb"],
                    dtype=np.uint8,
                )
                if "depth_linear" in camera:
                    arrays[f"{role}_depth_linear"] = np.asarray(
                        camera["depth_linear"],
                        dtype=np.float32,
                    )
            np.savez_compressed(self.artifact_dir / snapshot, **arrays)
            self.last_snapshot_id = snapshot_id

        self._append(
            self.steps_path,
            {
                "reset_epoch": self.reset_epoch,
                "observation_index": self.observation_index,
                "simulator_step": observation["simulator_step"],
                "mode": mode,
                "mode_name": MODE_NAMES[mode],
                "carry_hand": carry_hand_label,
                "active_gripper_command": active_gripper_command,
                "base_command": base_command.astype(float).tolist(),
                "base_qvel": base_qvel.astype(float).tolist(),
                "trunk_qpos": trunk_qpos.astype(float).tolist(),
                "torso_target": (
                    None
                    if active_torso_target is None
                    else active_torso_target.astype(float).tolist()
                ),
                "head_depth_finite_ratio": depth_finite_ratio,
                "head_depth_minimum_finite_ratio": (
                    self.minimum_depth_finite_ratio
                ),
                "head_depth_quality": (
                    "degraded" if depth_quality_degraded else "valid"
                ),
                "base_command_suppressed_due_to_depth_quality": (
                    base_command_suppressed_due_to_depth_quality
                ),
                "head_center_depth": center_depth,
                "head_forward_clearance_m": forward_clearance,
                "head_rgb_sha256": hashlib.sha256(
                    head_rgb.tobytes()
                ).hexdigest(),
                "capture": captured,
                "public_snapshot": snapshot,
                "action_sha256": hashlib.sha256(
                    action.tobytes()
                ).hexdigest(),
            },
        )
        self.observation_index += 1
        self.total_observations += 1
        return action

    @staticmethod
    def _append(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
            )


async def _health_check(connection, request):
    if getattr(request, "path", None) == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return (
            http.HTTPStatus.OK,
            {"Content-Type": "text/plain"},
            b"OK\n",
        )
    return None


async def serve(
    *,
    host: str,
    port: int,
    artifact_dir: Path,
    control: SharedCanaryControl,
    minimum_depth_finite_ratio: float = 0.95,
    ready_queue=None,
    observation_profile: str = FULL_CAMERA_PROFILE,
    expected_task_id: int = TASK_ID,
) -> None:
    state = ResidentCanaryState(
        artifact_dir,
        control,
        minimum_depth_finite_ratio=minimum_depth_finite_ratio,
        observation_profile=observation_profile,
        expected_task_id=expected_task_id,
    )
    if observation_profile == HEAD_RGB_MINIMAL_CAMERA_PROFILE:
        observation_schema_version = (
            HEAD_RGB_MINIMAL_OBSERVATION_SCHEMA_VERSION
        )
        observation_schema_sha256 = head_rgb_minimal_schema_sha256(
            expected_task_id=expected_task_id,
        )
    elif observation_profile == THREE_RGB_HEAD_EVAL_CAMERA_PROFILE:
        observation_schema_version = (
            THREE_RGB_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION
        )
        observation_schema_sha256 = three_rgb_head_eval_schema_sha256(
            expected_task_id=expected_task_id,
        )
    elif observation_profile == THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE:
        observation_schema_version = (
            THREE_RGBD_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION
        )
        observation_schema_sha256 = schema_sha256(
            expected_task_id=expected_task_id,
            head_eval=True,
        )
    else:
        observation_schema_version = OBSERVATION_SCHEMA_VERSION
        observation_schema_sha256 = schema_sha256(
            expected_task_id=expected_task_id,
        )
    metadata = {
        "mode": "resident_public_action_canary",
        "action_dim": 23,
        "observation_schema_version": observation_schema_version,
        "observation_schema_sha256": observation_schema_sha256,
        "registry_sha256": build_picking_up_toys_registry().sha256(),
        "websocket_keepalive": oracle_websocket_keepalive_manifest(),
    }

    async def handler(websocket):
        await websocket.send(packb(metadata))
        state._append(
            state.events_path,
            {
                "event": "connection_opened",
                "monotonic_s": time.monotonic(),
                "websocket_keepalive": oracle_websocket_keepalive_manifest(),
            },
        )
        try:
            while True:
                message = unpackb(
                    await websocket.recv(),
                    strict_map_key=False,
                )
                if isinstance(message, dict) and "reset" in message:
                    if message != {"reset": True}:
                        raise ValueError("invalid reset message")
                    state.reset()
                    continue
                if not isinstance(message, dict):
                    raise ValueError("observation message must be a mapping")
                action = state.act(message)
                await websocket.send(packb({"action": action}))
        except websockets.ConnectionClosed:
            state._append(
                state.events_path,
                {
                    "event": "connection_closed",
                    "monotonic_s": time.monotonic(),
                    "total_observations": state.total_observations,
                },
            )

    async with websocket_server.serve(
        handler,
        host,
        port,
        compression=None,
        max_size=None,
        process_request=_health_check,
        # This is a private Oracle endpoint.  CPU-only route and nested
        # inventory precomputation may legitimately block the handler longer
        # than the official 300-second policy timeout.  The matching local
        # client disables the timeout as well; QA never imports this server.
        **oracle_websocket_server_options(),
    ) as server:
        actual_port = int(server.sockets[0].getsockname()[1])
        if ready_queue is not None:
            ready_queue.put(actual_port)
        await server.serve_forever()


def server_process_main(
    host: str,
    port: int,
    artifact_dir: str,
    control: SharedCanaryControl,
    minimum_depth_finite_ratio: float = 0.95,
    ready_queue=None,
    observation_profile: str = FULL_CAMERA_PROFILE,
    expected_task_id: int = TASK_ID,
) -> None:
    asyncio.run(
        serve(
            host=host,
            port=port,
            artifact_dir=Path(artifact_dir),
            control=control,
            minimum_depth_finite_ratio=minimum_depth_finite_ratio,
            observation_profile=observation_profile,
            expected_task_id=expected_task_id,
            ready_queue=ready_queue,
        )
    )
