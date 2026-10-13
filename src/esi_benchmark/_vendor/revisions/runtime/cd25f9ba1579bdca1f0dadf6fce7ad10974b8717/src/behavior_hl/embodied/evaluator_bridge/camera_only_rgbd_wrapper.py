"""Read-only validation wrapper for predeclared R1Pro camera sensors."""

from __future__ import annotations

from collections.abc import Mapping

from omnigibson.envs import EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import (
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    get_robot_camera_names,
)
from omnigibson.eval.wrappers import RGBDFullResWrapper

from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    FULL_CAMERA_PROFILE,
    HEAD_RGB_MINIMAL_CAMERA_PROFILE,
    R1PRO_CAMERA_SENSOR_KEYS,
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
    qualify_r1pro_camera_sensor_keys,
)
from behavior_hl.embodied.policy.observations import CAMERA_SENSOR_NAMES


class PredeclaredCameraStructureError(RuntimeError):
    """Raised when a live camera differs from its preconstruction contract."""


def _space_keys(space):
    spaces = getattr(space, "spaces", None)
    if spaces is None:
        raise PredeclaredCameraStructureError(
            "camera observation space has no nested spaces mapping"
        )
    return set(spaces)


def _resolve_three_rgb_sensor_names(robot):
    """Resolve R1Pro camera roles from live sensor suffixes, uniquely.

    A complete scene can rename the robot prim to avoid a scene-level name
    collision while the camera sensor keys retain the authored owner prefix.
    Prefer the canonical ``robot.name`` qualification, then accept exactly
    one live key with the same official link/camera suffix.  Ambiguity and
    genuinely missing cameras remain hard failures.
    """

    live_names = set(robot.sensors)
    resolved = {}
    sources = {}
    for role, suffix in R1PRO_CAMERA_SENSOR_KEYS.items():
        preferred = f"{robot.name}:{suffix}"
        if preferred in live_names:
            resolved[role] = preferred
            sources[role] = "robot_name_qualified"
            continue
        matches = sorted(
            name
            for name in live_names
            if name == suffix or name.endswith(f":{suffix}")
        )
        if len(matches) != 1:
            raise PredeclaredCameraStructureError(
                f"camera role {role} resolved to {len(matches)} live "
                f"sensor keys for suffix {suffix!r}"
            )
        resolved[role] = matches[0]
        sources[role] = "unique_live_suffix"
    return resolved, sources


def _select_camera_complete_r1pro(env, *, required_roles):
    """Select the unique configured R1Pro, never ``env.robots[0]``.

    Full task templates may serialize a camera-less historical Robot before
    the evaluator adds ``robot_r1``.  Selection is therefore based on the
    official R1/R1Pro model plus the complete predeclared camera suffix set.
    """

    robots = tuple(getattr(env, "robots", ()))
    required_suffixes = {
        R1PRO_CAMERA_SENSOR_KEYS[role] for role in required_roles
    }
    census = []
    candidates = []
    for robot in robots:
        name = str(getattr(robot, "name", ""))
        model = str(getattr(robot, "model", "")).lower()
        sensors = getattr(robot, "sensors", {})
        sensor_names = set(sensors) if isinstance(sensors, Mapping) else set()
        complete = all(
            any(
                live == suffix or live.endswith(f":{suffix}")
                for live in sensor_names
            )
            for suffix in required_suffixes
        )
        census.append(
            {
                "name": name,
                "model": model,
                "sensor_count": len(sensor_names),
                "required_camera_suffixes_complete": complete,
            }
        )
        if model in {"r1", "r1pro"} and complete:
            candidates.append(robot)
    preferred = [robot for robot in candidates if robot.name == "robot_r1"]
    if len(preferred) == 1:
        return preferred[0], census
    if len(candidates) != 1:
        raise PredeclaredCameraStructureError(
            "expected one camera-complete configured R1Pro; "
            f"found {len(candidates)} among robots {census}"
        )
    return candidates[0], census


def attest_predeclared_camera_sensors(
    env,
    *,
    camera_profile: str = FULL_CAMERA_PROFILE,
):
    """Validate camera structure without mutating Replicator or observations."""

    if camera_profile not in {
        FULL_CAMERA_PROFILE,
        HEAD_RGB_MINIMAL_CAMERA_PROFILE,
        THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
        THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
    }:
        raise PredeclaredCameraStructureError(
            "unknown predeclared camera profile"
        )

    robot_eval_config = getattr(env, "_eval_robot_config", {})
    expected_roles = (
        ("left_wrist", "right_wrist", "head")
        if camera_profile
        in {
            FULL_CAMERA_PROFILE,
            THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
        }
        else ("head",)
    )
    robot, robot_census = _select_camera_complete_r1pro(
        env,
        required_roles=expected_roles,
    )
    if camera_profile in {
        THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
        THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
    }:
        role_to_sensor_name, sensor_name_sources = (
            _resolve_three_rgb_sensor_names(robot)
        )
    else:
        role_to_sensor_name = {
            camera_id: camera_name.split("::")[1]
            for camera_id, camera_name in get_robot_camera_names(
                robot.name,
                robot_eval_config,
            ).items()
        }
        sensor_name_sources = {
            role: "official_eval_config" for role in role_to_sensor_name
        }
    expected = (
        {
            "left_wrist": {
                "modalities": {"rgb", "depth_linear"},
                "resolution": tuple(WRIST_RESOLUTION),
            },
            "right_wrist": {
                "modalities": {"rgb", "depth_linear"},
                "resolution": tuple(WRIST_RESOLUTION),
            },
            "head": {
                "modalities": {
                    "rgb",
                    "depth_linear",
                    "seg_semantic",
                    "seg_instance",
                },
                "resolution": tuple(HEAD_RESOLUTION),
            },
        }
        if camera_profile == FULL_CAMERA_PROFILE
        else {
            "left_wrist": {
                "modalities": {"rgb"},
                "resolution": tuple(WRIST_RESOLUTION),
            },
            "right_wrist": {
                "modalities": {"rgb"},
                "resolution": tuple(WRIST_RESOLUTION),
            },
            "head": {
                "modalities": {"rgb", "depth_linear"},
                "resolution": tuple(HEAD_RESOLUTION),
            },
        }
        if camera_profile == THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE
        else {
            "head": {
                "modalities": {"rgb"},
                "resolution": tuple(HEAD_RESOLUTION),
            }
        }
        if camera_profile == HEAD_RGB_MINIMAL_CAMERA_PROFILE
        else {
            "left_wrist": {
                "modalities": {"rgb"},
                "resolution": tuple(WRIST_RESOLUTION),
            },
            "right_wrist": {
                "modalities": {"rgb"},
                "resolution": tuple(WRIST_RESOLUTION),
            },
            "head": {
                "modalities": {"rgb"},
                "resolution": tuple(HEAD_RESOLUTION),
            },
        }
    )
    if set(role_to_sensor_name) != set(expected):
        raise PredeclaredCameraStructureError(
            "eval camera roles do not match the predeclared contract"
        )

    observation_space = getattr(env, "observation_space", None)
    robot_spaces = getattr(observation_space, "spaces", {}).get(robot.name)
    observation_space_source = "environment"
    if robot_spaces is None:
        # Full-scene construction can expose the already-initialized robot
        # before Environment has composed its top-level Dict.  The robot's
        # own observation space is the canonical source used by
        # Environment._load_observation_space, so validate that same object
        # without mutating sensors or creating new annotators.
        robot_spaces = getattr(robot, "observation_space", None)
        observation_space_source = "robot"
    if robot_spaces is None:
        raise PredeclaredCameraStructureError(
            "robot observation space is unavailable from both environment "
            "and robot"
        )
    sensor_spaces = getattr(robot_spaces, "spaces", None)
    if sensor_spaces is None:
        raise PredeclaredCameraStructureError(
            "robot observation space has no sensor mapping"
        )

    report = {
        "schema_version": "behavior2026.predeclared-camera/v1",
        "camera_profile": camera_profile,
        "observation_space_source": observation_space_source,
        "selected_robot_name": robot.name,
        "robot_census": robot_census,
        "cameras": {},
    }
    for role, contract in expected.items():
        sensor_name = role_to_sensor_name[role]
        sensor = robot.sensors.get(sensor_name)
        if sensor is None:
            raise PredeclaredCameraStructureError(
                f"missing predeclared camera sensor for role {role}"
            )
        modalities = set(sensor.modalities)
        if modalities != contract["modalities"]:
            raise PredeclaredCameraStructureError(
                f"{role} modalities differ from the predeclared contract"
            )
        resolution = (int(sensor.image_height), int(sensor.image_width))
        if resolution != contract["resolution"]:
            raise PredeclaredCameraStructureError(
                f"{role} resolution differs from the predeclared contract"
            )
        annotators = getattr(sensor, "_annotators", None)
        if not isinstance(annotators, dict) or any(
            annotators.get(modality) is None for modality in modalities
        ):
            raise PredeclaredCameraStructureError(
                f"{role} annotators are not initialized"
            )
        render_product = getattr(sensor, "_render_product", None)
        if render_product is None:
            raise PredeclaredCameraStructureError(
                f"{role} render product is unavailable"
            )
        sensor_space = sensor_spaces.get(sensor_name)
        if sensor_space is None or not modalities.issubset(
            _space_keys(sensor_space)
        ):
            raise PredeclaredCameraStructureError(
                f"{role} observation space omits required modalities"
            )
        report["cameras"][role] = {
            "sensor_name": sensor_name,
            "sensor_name_source": sensor_name_sources[role],
            "modalities": sorted(modalities),
            "resolution": list(resolution),
            "annotators_initialized": True,
            "render_product_present": True,
        }
    return report


class CameraOnlyRGBDFullResWrapper(RGBDFullResWrapper):
    """Validate camera configuration created before ``Evaluator`` exists."""

    def __init__(self, env):
        EnvironmentWrapper.__init__(self, env=env)
        self._predeclared_camera_attestation = (
            attest_predeclared_camera_sensors(env)
        )

    @property
    def predeclared_camera_attestation(self):
        return {
            "schema_version": self._predeclared_camera_attestation[
                "schema_version"
            ],
            "observation_space_source": self._predeclared_camera_attestation[
                "observation_space_source"
            ],
            "selected_robot_name": self._predeclared_camera_attestation[
                "selected_robot_name"
            ],
            "robot_census": [
                dict(record)
                for record in self._predeclared_camera_attestation[
                    "robot_census"
                ]
            ],
            "cameras": {
                role: dict(record)
                for role, record in self._predeclared_camera_attestation[
                    "cameras"
                ].items()
            },
        }


class CameraOnlyHeadRGBWrapper(EnvironmentWrapper):
    """Oracle-only wrapper for the predeclared head-RGB profile.

    Unlike ``RGBDFullResWrapper``, this wrapper does not add depth or mutate
    any sensor after construction.  The robot config has already excluded
    wrist sensors and declared the head RGB modality before ``Evaluator``
    creates the environment.
    """

    def __init__(self, env):
        EnvironmentWrapper.__init__(self, env=env)
        self._predeclared_camera_attestation = (
            attest_predeclared_camera_sensors(
                env,
                camera_profile=HEAD_RGB_MINIMAL_CAMERA_PROFILE,
            )
        )

    @property
    def predeclared_camera_attestation(self):
        return {
            "schema_version": self._predeclared_camera_attestation[
                "schema_version"
            ],
            "camera_profile": self._predeclared_camera_attestation[
                "camera_profile"
            ],
            "observation_space_source": self._predeclared_camera_attestation[
                "observation_space_source"
            ],
            "selected_robot_name": self._predeclared_camera_attestation[
                "selected_robot_name"
            ],
            "robot_census": [
                dict(record)
                for record in self._predeclared_camera_attestation[
                    "robot_census"
                ]
            ],
            "cameras": {
                role: dict(record)
                for role, record in self._predeclared_camera_attestation[
                    "cameras"
                ].items()
            },
        }


class CameraOnlyThreeRGBHeadEvalWrapper(EnvironmentWrapper):
    """Three RGB sensors with head-only evaluator camera parameters."""

    def __init__(
        self,
        env,
        install_private_head_segmentation_side_channel=False,
    ):
        EnvironmentWrapper.__init__(self, env=env)
        if not isinstance(install_private_head_segmentation_side_channel, bool):
            raise TypeError(
                "install_private_head_segmentation_side_channel must be bool"
            )
        if install_private_head_segmentation_side_channel:
            from behavior_hl.embodied.evaluator_bridge.private_head_segmentation_side_channel import (
                install_private_head_segmentation_side_channel as _install_private_head_segmentation_side_channel,
            )

            _install_private_head_segmentation_side_channel(
                environment=self,
                head_sensor_name=CAMERA_SENSOR_NAMES["head"],
            )
        self._predeclared_camera_attestation = (
            attest_predeclared_camera_sensors(
                env,
                camera_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            )
        )

    @property
    def predeclared_camera_attestation(self):
        return {
            "schema_version": self._predeclared_camera_attestation[
                "schema_version"
            ],
            "camera_profile": self._predeclared_camera_attestation[
                "camera_profile"
            ],
            "observation_space_source": self._predeclared_camera_attestation[
                "observation_space_source"
            ],
            "selected_robot_name": self._predeclared_camera_attestation[
                "selected_robot_name"
            ],
            "robot_census": [
                dict(record)
                for record in self._predeclared_camera_attestation[
                    "robot_census"
                ]
            ],
            "cameras": {
                role: dict(record)
                for role, record in self._predeclared_camera_attestation[
                    "cameras"
                ].items()
            },
        }


class CameraOnlyThreeRGBDHeadEvalWrapper(EnvironmentWrapper):
    """Three RGB sensors plus aligned head depth and head-only metadata."""

    def __init__(self, env):
        EnvironmentWrapper.__init__(self, env=env)
        self._predeclared_camera_attestation = (
            attest_predeclared_camera_sensors(
                env,
                camera_profile=THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
            )
        )

    @property
    def predeclared_camera_attestation(self):
        return {
            **self._predeclared_camera_attestation,
            "robot_census": [
                dict(record)
                for record in self._predeclared_camera_attestation[
                    "robot_census"
                ]
            ],
            "cameras": {
                role: dict(record)
                for role, record in self._predeclared_camera_attestation[
                    "cameras"
                ].items()
            },
        }


__all__ = [
    "CameraOnlyHeadRGBWrapper",
    "CameraOnlyThreeRGBHeadEvalWrapper",
    "CameraOnlyThreeRGBDHeadEvalWrapper",
    "CameraOnlyRGBDFullResWrapper",
    "PredeclaredCameraStructureError",
    "attest_predeclared_camera_sensors",
]
