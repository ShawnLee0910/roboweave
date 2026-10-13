"""Bench-only reset-safe three-RGB wrapper for the resident Task25 lab.

The production Action20 rollout may attach its private head segmentation
channel after the final reset because it runs exactly one model episode.  The
interactive lab resets one resident scene between experiments, so its channel
must instead be installed during Environment-wrapper construction.

It temporarily replaces the camera-profile resolver only around one
``run_b1_single_load`` call and restores the original resolver in ``finally``.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
from multiprocessing.queues import Queue as MultiprocessingQueue
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from behavior_hl.embodied.evaluator_bridge.camera_only_rgbd_wrapper import (
    CameraOnlyThreeRGBHeadEvalWrapper,
    CameraOnlyThreeRGBDHeadEvalWrapper,
)
from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGB_HEAD_EVAL_CAMERA_WRAPPER_TARGET,
    THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGBD_HEAD_EVAL_CAMERA_WRAPPER_TARGET,
    camera_profile_contract as _shared_camera_profile_contract,
)
from behavior_hl.embodied.evaluator_bridge.private_head_segmentation_side_channel import (
    install_private_head_segmentation_side_channel,
)
from behavior_hl.embodied.policy.observations import CAMERA_SENSOR_NAMES


RESET_SAFE_THREE_RGB_WRAPPER_VERSION = (
    "behavior-hl-bench-try-reset-safe-three-rgb-wrapper/v1"
)
RESET_SAFE_THREE_RGB_WRAPPER_TARGET = (
    "lab.reset_safe_three_rgb_wrapper."
    "ResetSafeCameraOnlyThreeRGBHeadEvalWrapper"
)
RESET_SAFE_RGBD_WRAPPER_TARGET = (
    "lab.reset_safe_three_rgb_wrapper."
    "ResetSafeCameraOnlyThreeRGBDHeadEvalWrapper"
)
_SINGLE_LOAD_MODULE = (
    "behavior_hl.embodied.evaluator_bridge.run_b1_single_load"
)
_POLICY_READY_TIMEOUT_SECONDS = 300


def _install_reset_safe_channel(environment: Any) -> Mapping[str, Any]:
    channel = install_private_head_segmentation_side_channel(
        # Evaluator.env is this wrapper, not the wrapped raw Environment.
        # Ownership must therefore be published on the wrapper identity.
        environment=environment,
        head_sensor_name=CAMERA_SENSOR_NAMES["head"],
    )
    report = dict(channel.last_report)
    if (
        channel.started is not True
        or report.get("status") != "started"
        or report.get("installation_phase")
        != "environment_wrapper_construction"
        or report.get("installed_before_evaluator_handles") is not True
        or report.get("post_attach_evaluator_reset_budget") is not None
        or report.get("resident_modalities")
        not in [
            ["rgb"],
            ["depth_linear", "rgb"],
        ]
        or report.get("vision_sensor_modality_mutation") is not False
        or report.get("evaluator_segmentation_fields_written") is not False
    ):
        raise RuntimeError(
            "reset-safe private segmentation installation contract failed"
        )
    return {
        "schema_version": RESET_SAFE_THREE_RGB_WRAPPER_VERSION,
        "wrapper_target": (
            RESET_SAFE_RGBD_WRAPPER_TARGET
            if "depth_linear" in report["resident_modalities"]
            else RESET_SAFE_THREE_RGB_WRAPPER_TARGET
        ),
        "head_sensor_name_private": CAMERA_SENSOR_NAMES["head"],
        "model_visible_segmentation": False,
        "public_observation_modified": False,
        "channel_private": report,
    }


def _attestation_copy(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {**value, "channel_private": dict(value["channel_private"])}
    )


class ResetSafeCameraOnlyThreeRGBHeadEvalWrapper(
    CameraOnlyThreeRGBHeadEvalWrapper
):
    """Three public RGB cameras plus one persistent private head channel."""

    def __init__(self, env: Any):
        super().__init__(env)
        self._reset_safe_private_segmentation_attestation = (
            _install_reset_safe_channel(self)
        )

    @property
    def reset_safe_private_segmentation_attestation(self) -> Mapping[str, Any]:
        return _attestation_copy(
            self._reset_safe_private_segmentation_attestation
        )


class ResetSafeCameraOnlyThreeRGBDHeadEvalWrapper(
    CameraOnlyThreeRGBDHeadEvalWrapper
):
    """Three RGB cameras plus aligned head depth and the Marker channel."""

    def __init__(self, env: Any):
        super().__init__(env)
        self._reset_safe_private_segmentation_attestation = (
            _install_reset_safe_channel(self)
        )

    @property
    def reset_safe_private_segmentation_attestation(self) -> Mapping[str, Any]:
        return _attestation_copy(
            self._reset_safe_private_segmentation_attestation
        )


def resident_camera_profile_contract(camera_profile: str) -> Mapping[str, Any]:
    """Select the lab wrapper for exactly the frozen three-RGB profile."""

    contract = dict(_shared_camera_profile_contract(camera_profile))
    if camera_profile == THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE:
        if contract.get("wrapper") != THREE_RGBD_HEAD_EVAL_CAMERA_WRAPPER_TARGET:
            raise RuntimeError("shared RGB-D wrapper target drifted")
        contract["wrapper"] = RESET_SAFE_RGBD_WRAPPER_TARGET
        return MappingProxyType(contract)
    if camera_profile != THREE_RGB_HEAD_EVAL_CAMERA_PROFILE:
        return MappingProxyType(contract)
    if contract.get("wrapper") != THREE_RGB_HEAD_EVAL_CAMERA_WRAPPER_TARGET:
        raise RuntimeError("shared three-RGB wrapper target drifted")
    contract["wrapper"] = RESET_SAFE_THREE_RGB_WRAPPER_TARGET
    return MappingProxyType(contract)


@contextmanager
def _resident_camera_profile_override(runner_module: Any) -> Iterator[None]:
    """Install one process-local resolver override and always restore it."""

    original = getattr(runner_module, "camera_profile_contract", None)
    if original is not _shared_camera_profile_contract:
        raise RuntimeError("single-load camera profile resolver identity drifted")
    runner_module.camera_profile_contract = resident_camera_profile_contract
    try:
        yield
    finally:
        current = getattr(runner_module, "camera_profile_contract", None)
        runner_module.camera_profile_contract = original
        if current is not resident_camera_profile_contract:
            raise RuntimeError(
                "single-load camera profile resolver changed during resident run"
            )


@contextmanager
def _policy_ready_timeout_override() -> Iterator[None]:
    """Extend only the shared runner's cold spawned-policy ready wait.

    The shared single-load runner uses ``Queue.get(timeout=120)`` before it
    imports OmniGibson or consumes the one-scene-load budget.  On this node a
    fresh spawned interpreter can spend more than 120 seconds importing the
    frozen source tree.  Keep the shared EXEC tree immutable and widen exactly
    that one wait in this bench-only process; every other queue timeout keeps
    its original value.
    """

    original = MultiprocessingQueue.get

    def get_with_cold_start_allowance(
        queue: Any,
        block: bool = True,
        timeout: float | None = None,
    ) -> Any:
        if block is True and timeout == 120:
            timeout = _POLICY_READY_TIMEOUT_SECONDS
        return original(queue, block=block, timeout=timeout)

    MultiprocessingQueue.get = get_with_cold_start_allowance
    try:
        yield
    finally:
        current = MultiprocessingQueue.get
        MultiprocessingQueue.get = original
        if current is not get_with_cold_start_allowance:
            raise RuntimeError("multiprocessing queue getter changed during resident run")


def run_b1_single_load_with_reset_safe_three_rgb(**kwargs: Any) -> Any:
    """Run one frozen single-load worker with the bench-only wrapper."""

    if kwargs.get("camera_profile") not in {
        THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
        THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
    }:
        raise ValueError(
            "resident scene requires the frozen three-RGB or RGB-D profile"
        )
    runner_module = importlib.import_module(_SINGLE_LOAD_MODULE)
    with (
        _resident_camera_profile_override(runner_module),
        _policy_ready_timeout_override(),
    ):
        return runner_module.run_b1_single_load(**kwargs)


__all__ = [
    "RESET_SAFE_RGBD_WRAPPER_TARGET",
    "RESET_SAFE_THREE_RGB_WRAPPER_TARGET",
    "RESET_SAFE_THREE_RGB_WRAPPER_VERSION",
    "ResetSafeCameraOnlyThreeRGBDHeadEvalWrapper",
    "ResetSafeCameraOnlyThreeRGBHeadEvalWrapper",
    "resident_camera_profile_contract",
    "run_b1_single_load_with_reset_safe_three_rgb",
]
