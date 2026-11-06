"""Standalone observation-only websocket server for the G1 hold canary."""

from __future__ import annotations

import argparse
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

from behavior_hl.embodied.policy.hold_policy import (
    build_open_gripper_hold_action,
)
from behavior_hl.embodied.policy.network_codec import packb, unpackb
from behavior_hl.embodied.policy.observations import (
    OBSERVATION_SCHEMA_VERSION,
    schema_sha256,
    validate_public_observation,
)
from behavior_hl.embodied.policy.profiles import (
    build_picking_up_toys_registry,
)


class HoldCanaryState:
    def __init__(
        self,
        artifact_dir: Path,
        *,
        minimum_depth_finite_ratio: float,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.artifact_dir / "server_events.jsonl"
        self.observations_path = (
            self.artifact_dir / "observation_summaries.jsonl"
        )
        self.minimum_depth_finite_ratio = minimum_depth_finite_ratio
        self.reset_epoch = -1
        self.observation_index = 0
        self.total_observations = 0

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
        validate_public_observation(
            observation,
            require_simulator_step=True,
        )
        simulator_step = observation["simulator_step"]
        if simulator_step != self.observation_index:
            raise ValueError(
                f"unexpected simulator_step {simulator_step}; "
                f"expected {self.observation_index}"
            )

        summary = {
            "reset_epoch": self.reset_epoch,
            "observation_index": self.observation_index,
            "simulator_step": simulator_step,
            "arrival_monotonic_s": time.monotonic(),
            "arrays": {},
        }
        arrays = {
            "proprio": observation["proprio"],
            "cam_rel_poses": observation["cam_rel_poses"],
        }
        for role, camera in observation["cameras"].items():
            arrays[f"{role}.rgb"] = camera["rgb"]
            arrays[f"{role}.depth_linear"] = camera["depth_linear"]

        for field, value in arrays.items():
            array = np.asarray(value)
            finite = np.isfinite(array)
            finite_ratio = float(finite.mean()) if array.size else 0.0
            summary["arrays"][field] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "finite_ratio": finite_ratio,
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
            if field.endswith(".depth_linear"):
                if finite_ratio < self.minimum_depth_finite_ratio:
                    raise ValueError(
                        f"{field} finite ratio {finite_ratio:.6f} is below "
                        f"{self.minimum_depth_finite_ratio:.6f}"
                    )
            elif finite_ratio != 1.0:
                raise ValueError(f"{field} contains non-finite values")

        action = build_open_gripper_hold_action(observation)
        summary["action_sha256"] = hashlib.sha256(action.tobytes()).hexdigest()
        self._append(self.observations_path, summary)
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
    minimum_depth_finite_ratio: float = 0.95,
    ready_queue=None,
) -> None:
    state = HoldCanaryState(
        artifact_dir,
        minimum_depth_finite_ratio=minimum_depth_finite_ratio,
    )
    registry = build_picking_up_toys_registry()
    metadata = {
        "mode": "g1_open_gripper_hold",
        "action_dim": 23,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_schema_sha256": schema_sha256(),
        "registry_sha256": registry.sha256(),
    }

    async def handler(websocket):
        await websocket.send(packb(metadata))
        state._append(
            state.events_path,
            {
                "event": "connection_opened",
                "monotonic_s": time.monotonic(),
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
    ) as server:
        actual_port = int(server.sockets[0].getsockname()[1])
        if ready_queue is not None:
            ready_queue.put(actual_port)
        await server.serve_forever()


def server_process_main(
    host: str,
    port: int,
    artifact_dir: str,
    minimum_depth_finite_ratio: float = 0.95,
    ready_queue=None,
) -> None:
    asyncio.run(
        serve(
            host=host,
            port=port,
            artifact_dir=Path(artifact_dir),
            minimum_depth_finite_ratio=minimum_depth_finite_ratio,
            ready_queue=ready_queue,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--minimum-depth-finite-ratio",
        type=float,
        default=0.95,
    )
    args = parser.parse_args()
    asyncio.run(
        serve(
            host=args.host,
            port=args.port,
            artifact_dir=args.artifact_dir,
            minimum_depth_finite_ratio=args.minimum_depth_finite_ratio,
        )
    )


if __name__ == "__main__":
    main()
