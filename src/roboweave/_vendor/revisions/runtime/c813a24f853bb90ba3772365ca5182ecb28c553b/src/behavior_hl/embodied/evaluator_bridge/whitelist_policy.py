"""Evaluator-side observation filter in front of the private Oracle client.

The official BEHAVIOR websocket client has a fixed 60-second ping interval and
300-second ping timeout.  That is a reasonable default for a normal policy
server, but it is not appropriate for this local, private Oracle boundary:
some Oracle actions perform a long CPU-only route/inventory precomputation in
the server process.  A websocket ping timeout during that computation would
turn a valid resident scene into a false transport failure.

This module therefore uses a local client subclass with keepalive disabled for
the private Oracle connection only.  The official ``WebsocketClientPolicy``
and the QA path are never modified.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from urllib.parse import urlparse

import requests
import torch as th
import websockets
import websockets.sync.client

from behavior_hl.embodied.evaluator_bridge.private_observation_ingress import (
    declassify_official_observation,
)
from behavior_hl.embodied.evaluator_bridge.oracle_websocket_keepalive import (
    ORACLE_WEBSOCKET_PING_INTERVAL_S,
    ORACLE_WEBSOCKET_PING_TIMEOUT_S,
)
from behavior_hl.embodied.policy.observations import (
    TASK_ID,
    sanitize_head_rgb_minimal_observation,
    sanitize_official_observation,
    sanitize_three_rgb_head_eval_observation,
)
from behavior_hl.embodied.policy.network_codec import unpackb
from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    FULL_CAMERA_PROFILE,
    HEAD_RGB_MINIMAL_CAMERA_PROFILE,
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
)
from behavior_hl.embodied.policy.profiles import (
    build_picking_up_toys_registry,
)
from omnigibson.eval.utils.network_utils import WebsocketClientPolicy


LOGGER = logging.getLogger(__name__)


class OracleWebsocketClientPolicy(WebsocketClientPolicy):
    """Private local client with no ping timeout and fail-closed transport.

    The implementation intentionally subclasses the official client instead
    of editing or monkeypatching its source.  Only the two connection options
    differ.  ``act`` narrows the official reconnect behavior so that a dead
    private server cannot be silently treated as an action or trigger an
    unbounded reconnect when ``allow_reconnect`` is false.
    """

    def _wait_for_server(self):
        parsed = urlparse(self._uri)
        host = parsed.hostname
        port = parsed.port
        http_scheme = "https" if parsed.scheme == "wss" else "http"
        health_url = f"{http_scheme}://{host}:{port}/healthz"

        # Keep the official startup behavior: the private launcher owns the
        # server lifecycle, so waiting before the first connection is valid.
        while True:
            try:
                response = requests.get(health_url, timeout=2)
                if response.ok:
                    LOGGER.info(
                        "Private Oracle health check passed; connecting..."
                    )
                    break
            except Exception:
                pass
            time.sleep(5)

        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=ORACLE_WEBSOCKET_PING_INTERVAL_S,
                    ping_timeout=ORACLE_WEBSOCKET_PING_TIMEOUT_S,
                )
                metadata = unpackb(conn.recv())
                LOGGER.info("Connected to private Oracle server")
                return conn, metadata
            except (
                ConnectionRefusedError,
                websockets.exceptions.InvalidMessage,
                EOFError,
            ) as exc:
                LOGGER.info(
                    "Private Oracle websocket connection failed (%s); retrying",
                    exc,
                )
                time.sleep(5)

    def act(self, obs: dict) -> th.Tensor:
        if self._ws is None:
            self._ws, self._server_metadata = self._wait_for_server()

        data = self._packer.pack(obs)
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                self._ws.send(data)
                response = self._ws.recv()
                if isinstance(response, str):
                    raise RuntimeError(
                        f"Error in private Oracle server:\n{response}"
                    )
                action_dict = unpackb(response)
                if "action" not in action_dict:
                    if attempt < max_retries:
                        LOGGER.warning(
                            "Private Oracle server response missing action; "
                            "retrying (%s/%s)",
                            attempt + 1,
                            max_retries,
                        )
                        continue
                    raise RuntimeError(
                        "Private Oracle server response missing action: "
                        f"{action_dict}"
                    )
                return th.from_numpy(
                    deepcopy(action_dict["action"])
                ).to(th.float32)
            except websockets.exceptions.ConnectionClosed as exc:
                if self._allow_reconnect and attempt < max_retries:
                    LOGGER.warning(
                        "Private Oracle connection lost; reconnecting "
                        "(%s/%s)",
                        attempt + 1,
                        max_retries,
                    )
                    self._ws, self._server_metadata = self._wait_for_server()
                    continue
                raise RuntimeError(
                    f"Private Oracle websocket server unavailable: {exc}"
                ) from exc
            except (ConnectionError, EOFError, OSError) as exc:
                raise RuntimeError(
                    f"Private Oracle websocket transport failed: {exc}"
                ) from exc

        raise RuntimeError("private Oracle websocket action retries exhausted")

    def reset(self) -> None:
        if self._ws is None:
            self._ws, self._server_metadata = self._wait_for_server()
        try:
            self._ws.send(self._packer.pack({"reset": True}))
        except websockets.exceptions.ConnectionClosed as exc:
            raise RuntimeError(
                f"Private Oracle websocket server unavailable during reset: {exc}"
            ) from exc
        except (ConnectionError, EOFError, OSError) as exc:
            raise RuntimeError(
                f"Private Oracle websocket reset transport failed: {exc}"
            ) from exc


class WhitelistedWebsocketPolicy:
    """Send only the frozen normalized observation schema to the policy process."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        allow_reconnect: bool = False,
        observation_profile: str = FULL_CAMERA_PROFILE,
        expected_task_id: int = TASK_ID,
    ) -> None:
        if observation_profile not in {
            FULL_CAMERA_PROFILE,
            HEAD_RGB_MINIMAL_CAMERA_PROFILE,
            THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
        }:
            raise ValueError("unknown whitelist observation profile")
        self._client = OracleWebsocketClientPolicy(
            host=host,
            port=port,
            allow_reconnect=allow_reconnect,
        )
        self._action_dim: int | None = None
        self._simulator_step = 0
        self._observation_profile = observation_profile
        self._expected_task_id = expected_task_id

    def set_action_dim(self, action_dim: int) -> None:
        if action_dim != 23:
            raise ValueError(f"expected R1Pro action_dim=23, got {action_dim}")
        self._action_dim = action_dim

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        if self._action_dim != 23:
            raise RuntimeError("action dimension was not initialized")
        # The live wrapper may carry the exact head semantic / instance pair
        # for evaluator-private point-action binding.  Strip that pair at the
        # policy boundary, while continuing to fail closed on every other
        # undeclared field.
        declassified = declassify_official_observation(obs)
        if self._observation_profile == HEAD_RGB_MINIMAL_CAMERA_PROFILE:
            public_observation = sanitize_head_rgb_minimal_observation(
                declassified,
                expected_task_id=self._expected_task_id,
            )
        elif self._observation_profile == THREE_RGB_HEAD_EVAL_CAMERA_PROFILE:
            public_observation = sanitize_three_rgb_head_eval_observation(
                declassified,
                expected_task_id=self._expected_task_id,
            )
        elif self._observation_profile == THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE:
            public_observation = sanitize_official_observation(
                declassified,
                expected_task_id=self._expected_task_id,
                head_eval=True,
            )
        else:
            public_observation = sanitize_official_observation(
                declassified,
                expected_task_id=self._expected_task_id,
            )
        public_observation["simulator_step"] = self._simulator_step
        action = self._client.act(public_observation)
        if tuple(action.shape) != (23,) or not bool(th.isfinite(action).all()):
            raise ValueError("policy server returned a non-finite or non-23D action")
        self._simulator_step += 1
        return action

    def reset(self) -> None:
        self._simulator_step = 0
        self._client.reset()
        metadata = self._client.get_server_metadata()
        expected_registry = build_picking_up_toys_registry().sha256()
        if metadata.get("action_dim") != 23:
            raise ValueError("policy metadata action_dim does not match")
        if metadata.get("registry_sha256") != expected_registry:
            raise ValueError("policy metadata registry hash does not match")
