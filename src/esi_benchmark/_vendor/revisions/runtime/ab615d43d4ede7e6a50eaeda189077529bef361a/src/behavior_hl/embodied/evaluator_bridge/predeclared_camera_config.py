"""Pure configuration helper for one-shot R1Pro camera construction.

This module deliberately does not import OmniGibson, Isaac Sim, Replicator,
or Warp.  The official evaluator robot YAML is copied before ``Evaluator``
construction and only the three evaluated camera entries are overridden.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


R1PRO_CAMERA_SENSOR_KEYS = {
    "left_wrist": "left_realsense_link:Camera:0",
    "right_wrist": "right_realsense_link:Camera:0",
    "head": "zed_link:Camera:0",
}
R1PRO_HEAD_CAMERA_PRIM_PATH_FILTER = "zed_link/Camera"
FULL_CAMERA_PROFILE = "r1pro_rgbd_fullres"
HEAD_RGB_MINIMAL_CAMERA_PROFILE = "r1pro_head_rgb_minimal"
THREE_RGB_HEAD_EVAL_CAMERA_PROFILE = "r1pro_three_rgb_head_eval"
THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE = "r1pro_three_rgbd_head_eval"
# Official robot-asset ``mobile_manipulation.untucked_default_joint_pos``
# arm slices (indices 10:17 and 17:24).  The evaluator YAML otherwise resets
# both arms to all-zero joints, which points the wrist cameras into the robot
# shell.  This pose is robot-level and category/task independent.
R1PRO_OFFICIAL_UNTUCKED_LEFT_ARM = (
    0.0,
    0.0,
    1.57,
    -1.57,
    0.0,
    0.0,
    -1.57,
)
R1PRO_OFFICIAL_UNTUCKED_RIGHT_ARM = (
    -1.57,
    1.57,
    -1.57,
    0.0,
    0.0,
    0.0,
    0.0,
)
# Private side-channel key carrying the pre-replacement official YAML arm
# slices out of ``build_predeclared_r1pro_camera_robot_config`` when
# ``use_wrist_evidence_reset_pose`` rewrites them.  The caller must pop this
# key before handing the config to the official Evaluator so the live robot
# configuration remains byte-identical to today.
ORIGINAL_RESET_ARM_JOINTS_CONFIG_KEY = (
    "behavior_hl_private_original_reset_arm_joints"
)
FULL_CAMERA_WRAPPER_TARGET = (
    "behavior_hl.embodied.evaluator_bridge.camera_only_rgbd_wrapper."
    "CameraOnlyRGBDFullResWrapper"
)
HEAD_RGB_MINIMAL_CAMERA_WRAPPER_TARGET = (
    "behavior_hl.embodied.evaluator_bridge.camera_only_rgbd_wrapper."
    "CameraOnlyHeadRGBWrapper"
)
THREE_RGB_HEAD_EVAL_CAMERA_WRAPPER_TARGET = (
    "behavior_hl.embodied.evaluator_bridge.camera_only_rgbd_wrapper."
    "CameraOnlyThreeRGBHeadEvalWrapper"
)
THREE_RGBD_HEAD_EVAL_CAMERA_WRAPPER_TARGET = (
    "behavior_hl.embodied.evaluator_bridge.camera_only_rgbd_wrapper."
    "CameraOnlyThreeRGBDHeadEvalWrapper"
)


class PredeclaredCameraConfigError(ValueError):
    """Raised when the official robot config cannot be copied safely."""


def qualify_r1pro_camera_sensor_keys(robot_name: str) -> dict[str, str]:
    """Return the live ``robot.sensors`` keys for one named R1Pro.

    ``sensor_config`` is indexed by the unqualified link key, whereas
    OmniGibson stores instantiated sensors under ``<robot.name>:<link-key>``.
    Keeping that distinction explicit prevents head-only evaluator metadata
    from accidentally hiding the two RGB-only wrist evidence sensors.
    """

    if not isinstance(robot_name, str) or not robot_name:
        raise PredeclaredCameraConfigError(
            "robot_name must be a non-empty string"
        )
    return {
        role: f"{robot_name}:{sensor_name}"
        for role, sensor_name in R1PRO_CAMERA_SENSOR_KEYS.items()
    }


def camera_profile_contract(camera_profile: str) -> Mapping[str, Any]:
    """Return pure launch metadata for one explicit camera profile."""

    if camera_profile == FULL_CAMERA_PROFILE:
        return {
            "camera_profile": FULL_CAMERA_PROFILE,
            "wrapper": FULL_CAMERA_WRAPPER_TARGET,
            "camera_roles": ("head", "left_wrist", "right_wrist"),
            "write_video": True,
        }
    if camera_profile == HEAD_RGB_MINIMAL_CAMERA_PROFILE:
        return {
            "camera_profile": HEAD_RGB_MINIMAL_CAMERA_PROFILE,
            "wrapper": HEAD_RGB_MINIMAL_CAMERA_WRAPPER_TARGET,
            "camera_roles": ("head",),
            "write_video": False,
        }
    if camera_profile == THREE_RGB_HEAD_EVAL_CAMERA_PROFILE:
        return {
            "camera_profile": THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            "wrapper": THREE_RGB_HEAD_EVAL_CAMERA_WRAPPER_TARGET,
            # Only this set is allowed to create evaluator camera-parameter
            # annotators. Wrist sensors remain RGB evidence sensors.
            "camera_roles": ("head",),
            "artifact_camera_roles": (
                "left_wrist",
                "right_wrist",
                "head",
            ),
            "write_video": False,
        }
    if camera_profile == THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE:
        return {
            "camera_profile": THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
            "wrapper": THREE_RGBD_HEAD_EVAL_CAMERA_WRAPPER_TARGET,
            "camera_roles": ("head",),
            "artifact_camera_roles": (
                "left_wrist",
                "right_wrist",
                "head",
            ),
            "write_video": False,
        }
    raise PredeclaredCameraConfigError(
        "camera_profile must be one of the supported explicit profiles"
    )


def _resolution_pair(
    value: Sequence[int],
    *,
    name: str,
) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise PredeclaredCameraConfigError(
            f"{name} must contain exactly height and width"
        )
    height, width = value
    if (
        isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
    ):
        raise PredeclaredCameraConfigError(
            f"{name} height and width must be positive integers"
        )
    return height, width


def _sensor_entry(
    *,
    modalities: Sequence[str],
    resolution: tuple[int, int],
) -> dict[str, Any]:
    height, width = resolution
    return {
        "modalities": list(modalities),
        "sensor_kwargs": {
            "image_height": height,
            "image_width": width,
        },
    }


def build_predeclared_r1pro_camera_robot_config(
    official_robot_config_path: str | Path,
    *,
    head_resolution: Sequence[int],
    wrist_resolution: Sequence[int],
    camera_profile: str = FULL_CAMERA_PROFILE,
    use_wrist_evidence_reset_pose: bool = False,
    wrist_focal_length_mm: float | None = None,
    head_focal_length_mm: float | None = None,
) -> dict[str, Any]:
    """Copy the official robot YAML and apply one explicit camera profile.

    ``FULL_CAMERA_PROFILE`` preserves the historical three-camera RGB-D plus
    head-segmentation contract byte-for-byte.  The Oracle-only
    ``HEAD_RGB_MINIMAL_CAMERA_PROFILE`` instead whitelists the official head
    sensor, requests RGB only, and removes wrist sensors before Environment
    construction.  No post-construction modality mutation is used.
    """

    if camera_profile not in {
        FULL_CAMERA_PROFILE,
        HEAD_RGB_MINIMAL_CAMERA_PROFILE,
        THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
        THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
    }:
        raise PredeclaredCameraConfigError(
            "camera_profile must be one of the supported explicit profiles"
        )
    for name, value in (
        ("wrist_focal_length_mm", wrist_focal_length_mm),
        ("head_focal_length_mm", head_focal_length_mm),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise PredeclaredCameraConfigError(
                f"{name} must be finite and positive"
            )

    path = Path(official_robot_config_path).resolve()
    if not path.is_file():
        raise PredeclaredCameraConfigError(
            f"official robot config does not exist: {path}"
        )
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, Mapping):
        raise PredeclaredCameraConfigError(
            "official robot config must contain a mapping"
        )

    config = deepcopy(dict(loaded))
    if not isinstance(use_wrist_evidence_reset_pose, bool):
        raise PredeclaredCameraConfigError(
            "use_wrist_evidence_reset_pose must be bool"
        )
    if use_wrist_evidence_reset_pose:
        reset_joint_pos = config.get("reset_joint_pos")
        if (
            not isinstance(reset_joint_pos, list)
            or len(reset_joint_pos) != 28
        ):
            raise PredeclaredCameraConfigError(
                "official R1Pro reset_joint_pos must contain 28 joints"
            )
        reset_joint_pos = list(reset_joint_pos)
        # Preserve the pre-replacement official YAML arm slices for callers
        # (the arm-presentation ladder's stowed endpoint).  This key is a
        # private side channel and never reaches the official robot loader.
        config[ORIGINAL_RESET_ARM_JOINTS_CONFIG_KEY] = {
            "left": [float(value) for value in reset_joint_pos[10:17]],
            "right": [float(value) for value in reset_joint_pos[17:24]],
        }
        reset_joint_pos[10:17] = R1PRO_OFFICIAL_UNTUCKED_LEFT_ARM
        reset_joint_pos[17:24] = R1PRO_OFFICIAL_UNTUCKED_RIGHT_ARM
        config["reset_joint_pos"] = reset_joint_pos
    sensor_config = config.get("sensor_config")
    if not isinstance(sensor_config, Mapping):
        raise PredeclaredCameraConfigError(
            "official robot config must contain sensor_config"
        )
    sensor_config = deepcopy(dict(sensor_config))

    head = _resolution_pair(head_resolution, name="head_resolution")
    wrist = _resolution_pair(
        wrist_resolution,
        name="wrist_resolution",
    )
    if camera_profile == FULL_CAMERA_PROFILE:
        sensor_config[R1PRO_CAMERA_SENSOR_KEYS["left_wrist"]] = _sensor_entry(
            modalities=("rgb", "depth_linear"),
            resolution=wrist,
        )
        sensor_config[R1PRO_CAMERA_SENSOR_KEYS["right_wrist"]] = _sensor_entry(
            modalities=("rgb", "depth_linear"),
            resolution=wrist,
        )
        sensor_config[R1PRO_CAMERA_SENSOR_KEYS["head"]] = _sensor_entry(
            modalities=(
                "rgb",
                "depth_linear",
                "seg_semantic",
                "seg_instance",
            ),
            resolution=head,
        )
    elif camera_profile == HEAD_RGB_MINIMAL_CAMERA_PROFILE:
        # The robot asset still contains wrist camera prims, but the official
        # Robot loader must never instantiate them in this profile.  The
        # include filter is applied during robot construction, before any
        # render product or annotator can be created.
        sensor_config[R1PRO_CAMERA_SENSOR_KEYS["head"]] = _sensor_entry(
            modalities=("rgb",),
            resolution=head,
        )
        config["include_sensor_names"] = [
            R1PRO_HEAD_CAMERA_PRIM_PATH_FILTER,
        ]
        eval_config = config.get("eval")
        if not isinstance(eval_config, Mapping):
            raise PredeclaredCameraConfigError(
                "official robot config must contain eval mapping"
            )
        eval_config = deepcopy(dict(eval_config))
        camera_sensor_names = eval_config.get("camera_sensor_names")
        if not isinstance(camera_sensor_names, Mapping):
            raise PredeclaredCameraConfigError(
                "official robot eval config must contain camera_sensor_names"
            )
        eval_config["camera_sensor_names"] = {
            "head": camera_sensor_names.get(
                "head", R1PRO_CAMERA_SENSOR_KEYS["head"]
            )
        }
        config["eval"] = eval_config
    else:
        for role, resolution in (
            ("left_wrist", wrist),
            ("right_wrist", wrist),
            ("head", head),
        ):
            sensor_config[R1PRO_CAMERA_SENSOR_KEYS[role]] = _sensor_entry(
                modalities=(
                    ("rgb", "depth_linear")
                    if camera_profile == THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE
                    and role == "head"
                    else ("rgb",)
                ),
                resolution=resolution,
            )
        if wrist_focal_length_mm is not None:
            for role in ("left_wrist", "right_wrist"):
                sensor_config[R1PRO_CAMERA_SENSOR_KEYS[role]][
                    "sensor_kwargs"
                ]["focal_length"] = float(wrist_focal_length_mm)
        if head_focal_length_mm is not None:
            sensor_config[R1PRO_CAMERA_SENSOR_KEYS["head"]][
                "sensor_kwargs"
            ]["focal_length"] = float(head_focal_length_mm)
        # Evaluator._preprocess_obs reads camera_parameters for exactly these
        # roles. Keep wrists out of that lazy SyntheticData graph while their
        # RGB observations remain available in the flattened robot payload.
        eval_config = config.get("eval")
        if not isinstance(eval_config, Mapping):
            raise PredeclaredCameraConfigError(
                "official robot config must contain eval mapping"
            )
        eval_config = deepcopy(dict(eval_config))
        camera_sensor_names = eval_config.get("camera_sensor_names")
        if not isinstance(camera_sensor_names, Mapping):
            raise PredeclaredCameraConfigError(
                "official robot eval config must contain camera_sensor_names"
            )
        eval_config["camera_sensor_names"] = {
            "head": camera_sensor_names.get(
                "head", R1PRO_CAMERA_SENSOR_KEYS["head"]
            )
        }
        config["eval"] = eval_config
    config["sensor_config"] = sensor_config
    return config


__all__ = [
    "ORIGINAL_RESET_ARM_JOINTS_CONFIG_KEY",
    "PredeclaredCameraConfigError",
    "R1PRO_CAMERA_SENSOR_KEYS",
    "R1PRO_HEAD_CAMERA_PRIM_PATH_FILTER",
    "R1PRO_OFFICIAL_UNTUCKED_LEFT_ARM",
    "R1PRO_OFFICIAL_UNTUCKED_RIGHT_ARM",
    "qualify_r1pro_camera_sensor_keys",
    "FULL_CAMERA_PROFILE",
    "HEAD_RGB_MINIMAL_CAMERA_PROFILE",
    "THREE_RGB_HEAD_EVAL_CAMERA_PROFILE",
    "THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE",
    "FULL_CAMERA_WRAPPER_TARGET",
    "HEAD_RGB_MINIMAL_CAMERA_WRAPPER_TARGET",
    "THREE_RGB_HEAD_EVAL_CAMERA_WRAPPER_TARGET",
    "THREE_RGBD_HEAD_EVAL_CAMERA_WRAPPER_TARGET",
    "camera_profile_contract",
    "build_predeclared_r1pro_camera_robot_config",
]
