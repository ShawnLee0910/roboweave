"""Exact ingress contract for official flattened R1Pro observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


OBSERVATION_SCHEMA_VERSION = "r1pro-rgbd-observation-v1"
# The full RGB-D contract above is the default model-facing contract.  This
# deliberately separate schema is used only by the direct, private Oracle
# lane.  Keeping a distinct schema makes an accidental head-only observation
# fail closed at every full-policy ingress instead of silently filling missing
# wrist/depth fields.
HEAD_RGB_MINIMAL_OBSERVATION_SCHEMA_VERSION = (
    "behavior2026.oracle-head-rgb-observation/v1"
)
HEAD_RGB_MINIMAL_CAMERA_PROFILE = "r1pro_head_rgb_minimal"
THREE_RGB_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION = (
    "behavior2026.oracle-three-rgb-head-eval-observation/v1"
)
THREE_RGB_HEAD_EVAL_CAMERA_PROFILE = "r1pro_three_rgb_head_eval"
THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE = "r1pro_three_rgbd_head_eval"
THREE_RGBD_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION = (
    "behavior2026.three-rgbd-head-eval-observation/v1"
)
ROBOT_NAME = "robot_r1"
TASK_ID = 7
PROPRIO_DIM = 61
CAM_REL_POSE_DIM = 21
HEAD_RGB_MINIMAL_CAM_REL_POSE_DIM = 7

CAMERA_SENSOR_NAMES = {
    "left_wrist": "robot_r1:left_realsense_link:Camera:0",
    "right_wrist": "robot_r1:right_realsense_link:Camera:0",
    "head": "robot_r1:zed_link:Camera:0",
}
CAMERA_RESOLUTIONS = {
    "left_wrist": (480, 480),
    "right_wrist": (480, 480),
    "head": (720, 720),
}
HEAD_RGB_MINIMAL_CAMERA_RESOLUTIONS = {
    "head": CAMERA_RESOLUTIONS["head"],
}


class ObservationContractError(ValueError):
    """Raised when evaluator output deviates from the frozen whitelist."""


@dataclass(frozen=True, slots=True)
class ArrayContract:
    shapes: tuple[tuple[int, ...], ...]

    def validate(self, value: Any, field: str) -> None:
        shape = getattr(value, "shape", None)
        if shape is None:
            raise ObservationContractError(f"{field!r} has no array shape")
        actual = tuple(int(part) for part in shape)
        if actual not in self.shapes:
            raise ObservationContractError(
                f"{field!r} shape {actual} is not one of {self.shapes}"
            )


def _camera_key(role: str, modality: str) -> str:
    return f"{ROBOT_NAME}::{CAMERA_SENSOR_NAMES[role]}::{modality}"


PROPRIO_KEY = f"{ROBOT_NAME}::proprio"
CAM_REL_POSES_KEY = f"{ROBOT_NAME}::cam_rel_poses"
TASK_ID_KEY = "task_id"


def official_field_contracts(*, head_eval: bool = False) -> dict[str, ArrayContract]:
    contracts = {
        PROPRIO_KEY: ArrayContract(((PROPRIO_DIM,),)),
        CAM_REL_POSES_KEY: ArrayContract(
            ((HEAD_RGB_MINIMAL_CAM_REL_POSE_DIM if head_eval else CAM_REL_POSE_DIM,),)
        ),
        TASK_ID_KEY: ArrayContract(((1,),)),
    }
    for role, (height, width) in CAMERA_RESOLUTIONS.items():
        contracts[_camera_key(role, "rgb")] = ArrayContract(
            ((height, width, 3), (height, width, 4))
        )
        if not head_eval or role == "head":
            contracts[_camera_key(role, "depth_linear")] = ArrayContract(
                ((height, width), (height, width, 1))
            )
    return contracts


def head_rgb_minimal_field_contracts() -> dict[str, ArrayContract]:
    """Return the exact private-Oracle head-RGB observation contract.

    This is intentionally not a relaxation of :func:`official_field_contracts`:
    the minimal lane has a different schema and contains no wrist, depth, or
    segmentation fields.  The direct Oracle does not use a public policy
    ingress, so its head-only schema can be kept narrow without weakening the
    normal full RGB-D model boundary.
    """

    head_prefix = _camera_key("head", "rgb")
    return {
        PROPRIO_KEY: ArrayContract(((PROPRIO_DIM,),)),
        CAM_REL_POSES_KEY: ArrayContract(
            ((HEAD_RGB_MINIMAL_CAM_REL_POSE_DIM,),)
        ),
        TASK_ID_KEY: ArrayContract(((1,),)),
        head_prefix: ArrayContract(
            (
                (
                    HEAD_RGB_MINIMAL_CAMERA_RESOLUTIONS["head"][0],
                    HEAD_RGB_MINIMAL_CAMERA_RESOLUTIONS["head"][1],
                    3,
                ),
                (
                    HEAD_RGB_MINIMAL_CAMERA_RESOLUTIONS["head"][0],
                    HEAD_RGB_MINIMAL_CAMERA_RESOLUTIONS["head"][1],
                    4,
                ),
            )
        ),
    }


def three_rgb_head_eval_field_contracts() -> dict[str, ArrayContract]:
    """Return the exact three-RGB / head-only-evaluator contract.

    All three physical R1Pro cameras expose RGB for trajectory evidence, but
    only the head camera belongs to ``Evaluator.robot_camera_names``.  The
    resulting camera-pose vector is therefore seven-dimensional and neither
    wrist creates a lazy ``camera_params`` annotator.
    """

    contracts = {
        PROPRIO_KEY: ArrayContract(((PROPRIO_DIM,),)),
        CAM_REL_POSES_KEY: ArrayContract(
            ((HEAD_RGB_MINIMAL_CAM_REL_POSE_DIM,),)
        ),
        TASK_ID_KEY: ArrayContract(((1,),)),
    }
    for role, (height, width) in CAMERA_RESOLUTIONS.items():
        contracts[_camera_key(role, "rgb")] = ArrayContract(
            ((height, width, 3), (height, width, 4))
        )
    return contracts


def sanitize_official_observation(
    observation: Mapping[str, Any],
    *,
    expected_task_id: int = TASK_ID,
    head_eval: bool = False,
) -> dict[str, Any]:
    """Validate exact fields, then map simulator-specific names to public roles."""

    if not isinstance(observation, Mapping):
        raise ObservationContractError("observation must be a mapping")
    contracts = official_field_contracts(head_eval=head_eval)
    actual = set(observation)
    expected = set(contracts)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ObservationContractError(
            f"missing observation fields: {sorted(missing)}"
        )
    if unknown:
        raise ObservationContractError(
            f"unknown observation fields: {sorted(unknown)}"
        )
    for field, contract in contracts.items():
        contract.validate(observation[field], field)

    task_id = _single_integer(observation[TASK_ID_KEY], TASK_ID_KEY)
    if task_id != expected_task_id:
        raise ObservationContractError(
            f"unexpected task id {task_id}; expected {expected_task_id}"
        )

    cameras = {}
    for role in CAMERA_SENSOR_NAMES:
        rgb = observation[_camera_key(role, "rgb")]
        if tuple(int(part) for part in rgb.shape)[-1] == 4:
            rgb = rgb[..., :3]
        cameras[role] = {"rgb": rgb}
        if not head_eval or role == "head":
            cameras[role]["depth_linear"] = observation[
                _camera_key(role, "depth_linear")
            ]
    public = {
        "schema_version": (
            THREE_RGBD_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION
            if head_eval
            else OBSERVATION_SCHEMA_VERSION
        ),
        "task_id": task_id,
        "proprio": observation[PROPRIO_KEY],
        "cam_rel_poses": observation[CAM_REL_POSES_KEY],
        "cameras": cameras,
    }
    validate_public_observation(
        public,
        expected_task_id=expected_task_id,
        head_eval=head_eval,
    )
    return public


def sanitize_head_rgb_minimal_observation(
    observation: Mapping[str, Any],
    *,
    expected_task_id: int = TASK_ID,
) -> dict[str, Any]:
    """Validate and normalize one Oracle-only head-RGB observation.

    No implicit zero/empty wrist or depth values are accepted.  The exact
    field set and the seven-dimensional single-camera pose are part of this
    separate schema, so a full observation or a malformed legacy observation
    is rejected rather than coerced.
    """

    if not isinstance(observation, Mapping):
        raise ObservationContractError("observation must be a mapping")
    contracts = head_rgb_minimal_field_contracts()
    actual = set(observation)
    expected = set(contracts)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ObservationContractError(
            f"missing head-RGB minimal observation fields: {sorted(missing)}"
        )
    if unknown:
        raise ObservationContractError(
            f"unknown head-RGB minimal observation fields: {sorted(unknown)}"
        )
    for field, contract in contracts.items():
        contract.validate(observation[field], field)

    task_id = _single_integer(observation[TASK_ID_KEY], TASK_ID_KEY)
    if task_id != expected_task_id:
        raise ObservationContractError(
            f"unexpected task id {task_id}; expected {expected_task_id}"
        )
    rgb = observation[_camera_key("head", "rgb")]
    if tuple(int(part) for part in rgb.shape)[-1] == 4:
        rgb = rgb[..., :3]
    public = {
        "schema_version": HEAD_RGB_MINIMAL_OBSERVATION_SCHEMA_VERSION,
        "task_id": task_id,
        "proprio": observation[PROPRIO_KEY],
        "cam_rel_poses": observation[CAM_REL_POSES_KEY],
        "cameras": {"head": {"rgb": rgb}},
    }
    validate_head_rgb_minimal_observation(
        public,
        expected_task_id=expected_task_id,
    )
    return public


def sanitize_three_rgb_head_eval_observation(
    observation: Mapping[str, Any],
    *,
    expected_task_id: int = TASK_ID,
) -> dict[str, Any]:
    """Validate three RGB sensors with only one evaluator camera pose."""

    if not isinstance(observation, Mapping):
        raise ObservationContractError("observation must be a mapping")
    contracts = three_rgb_head_eval_field_contracts()
    actual = set(observation)
    expected = set(contracts)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ObservationContractError(
            f"missing three-RGB observation fields: {sorted(missing)}"
        )
    if unknown:
        raise ObservationContractError(
            f"unknown three-RGB observation fields: {sorted(unknown)}"
        )
    for field, contract in contracts.items():
        contract.validate(observation[field], field)
    task_id = _single_integer(observation[TASK_ID_KEY], TASK_ID_KEY)
    if task_id != expected_task_id:
        raise ObservationContractError(
            f"unexpected task id {task_id}; expected {expected_task_id}"
        )
    cameras: dict[str, dict[str, Any]] = {}
    for role in CAMERA_SENSOR_NAMES:
        rgb = observation[_camera_key(role, "rgb")]
        if tuple(int(part) for part in rgb.shape)[-1] == 4:
            rgb = rgb[..., :3]
        cameras[role] = {"rgb": rgb}
    public = {
        "schema_version": THREE_RGB_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION,
        "task_id": task_id,
        "proprio": observation[PROPRIO_KEY],
        "cam_rel_poses": observation[CAM_REL_POSES_KEY],
        "cameras": cameras,
    }
    validate_three_rgb_head_eval_observation(
        public,
        expected_task_id=expected_task_id,
    )
    return public


def validate_public_observation(
    observation: Mapping[str, Any],
    *,
    require_simulator_step: bool = False,
    expected_task_id: int = TASK_ID,
    head_eval: bool = False,
) -> None:
    """Validate the normalized message received by the policy server."""

    if not isinstance(observation, Mapping):
        raise ObservationContractError("public observation must be a mapping")
    expected = {
        "schema_version",
        "task_id",
        "proprio",
        "cam_rel_poses",
        "cameras",
    }
    if require_simulator_step:
        expected.add("simulator_step")
    actual = set(observation)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ObservationContractError(
            f"missing public observation fields: {sorted(missing)}"
        )
    if unknown:
        raise ObservationContractError(
            f"unknown public observation fields: {sorted(unknown)}"
        )
    expected_schema_version = (
        THREE_RGBD_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION
        if head_eval
        else OBSERVATION_SCHEMA_VERSION
    )
    if observation["schema_version"] != expected_schema_version:
        raise ObservationContractError("unexpected observation schema version")
    if observation["task_id"] != expected_task_id:
        raise ObservationContractError("unexpected public task id")
    if require_simulator_step:
        step = observation["simulator_step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ObservationContractError(
                "simulator_step must be a non-negative integer"
            )

    ArrayContract(((PROPRIO_DIM,),)).validate(
        observation["proprio"],
        "proprio",
    )
    ArrayContract(
        ((HEAD_RGB_MINIMAL_CAM_REL_POSE_DIM if head_eval else CAM_REL_POSE_DIM,),)
    ).validate(
        observation["cam_rel_poses"],
        "cam_rel_poses",
    )
    cameras = observation["cameras"]
    if not isinstance(cameras, Mapping):
        raise ObservationContractError("cameras must be a mapping")
    if set(cameras) != set(CAMERA_RESOLUTIONS):
        raise ObservationContractError("public camera roles do not match")
    for role, (height, width) in CAMERA_RESOLUTIONS.items():
        camera = cameras[role]
        if not isinstance(camera, Mapping):
            raise ObservationContractError(f"camera {role!r} must be a mapping")
        expected_modalities = (
            {"rgb", "depth_linear"}
            if not head_eval or role == "head"
            else {"rgb"}
        )
        if set(camera) != expected_modalities:
            raise ObservationContractError(
                f"camera {role!r} modalities do not match"
            )
        ArrayContract(((height, width, 3),)).validate(
            camera["rgb"],
            f"{role}.rgb",
        )
        if not head_eval or role == "head":
            ArrayContract(((height, width), (height, width, 1))).validate(
                camera["depth_linear"],
                f"{role}.depth_linear",
            )


def validate_head_rgb_minimal_observation(
    observation: Mapping[str, Any],
    *,
    expected_task_id: int = TASK_ID,
) -> None:
    """Validate the normalized private-Oracle head-only public payload."""

    if not isinstance(observation, Mapping):
        raise ObservationContractError("public observation must be a mapping")
    expected = {
        "schema_version",
        "task_id",
        "proprio",
        "cam_rel_poses",
        "cameras",
    }
    actual = set(observation)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ObservationContractError(
            f"missing head-RGB minimal public fields: {sorted(missing)}"
        )
    if unknown:
        raise ObservationContractError(
            f"unknown head-RGB minimal public fields: {sorted(unknown)}"
        )
    if observation["schema_version"] != (
        HEAD_RGB_MINIMAL_OBSERVATION_SCHEMA_VERSION
    ):
        raise ObservationContractError(
            "unexpected head-RGB minimal observation schema version"
        )
    if observation["task_id"] != expected_task_id:
        raise ObservationContractError("unexpected head-RGB minimal task id")
    ArrayContract(((PROPRIO_DIM,),)).validate(
        observation["proprio"],
        "proprio",
    )
    ArrayContract(((HEAD_RGB_MINIMAL_CAM_REL_POSE_DIM,),)).validate(
        observation["cam_rel_poses"],
        "cam_rel_poses",
    )
    cameras = observation["cameras"]
    if not isinstance(cameras, Mapping) or set(cameras) != {"head"}:
        raise ObservationContractError(
            "head-RGB minimal public camera roles do not match"
        )
    camera = cameras["head"]
    if not isinstance(camera, Mapping) or set(camera) != {"rgb"}:
        raise ObservationContractError(
            "head-RGB minimal camera modalities do not match"
        )
    height, width = HEAD_RGB_MINIMAL_CAMERA_RESOLUTIONS["head"]
    ArrayContract(((height, width, 3),)).validate(
        camera["rgb"],
        "head.rgb",
    )


def validate_three_rgb_head_eval_observation(
    observation: Mapping[str, Any],
    *,
    require_simulator_step: bool = False,
    expected_task_id: int = TASK_ID,
) -> None:
    """Validate the public three-camera RGB-only trajectory payload."""

    if not isinstance(observation, Mapping):
        raise ObservationContractError("public observation must be a mapping")
    expected = {
        "schema_version",
        "task_id",
        "proprio",
        "cam_rel_poses",
        "cameras",
    }
    if require_simulator_step:
        expected.add("simulator_step")
    actual = set(observation)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ObservationContractError(
            f"missing three-RGB public fields: {sorted(missing)}"
        )
    if unknown:
        raise ObservationContractError(
            f"unknown three-RGB public fields: {sorted(unknown)}"
        )
    if observation["schema_version"] != (
        THREE_RGB_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION
    ):
        raise ObservationContractError(
            "unexpected three-RGB observation schema version"
        )
    if observation["task_id"] != expected_task_id:
        raise ObservationContractError("unexpected three-RGB task id")
    if require_simulator_step:
        step = observation["simulator_step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ObservationContractError(
                "simulator_step must be a non-negative integer"
            )
    ArrayContract(((PROPRIO_DIM,),)).validate(
        observation["proprio"], "proprio"
    )
    ArrayContract(((HEAD_RGB_MINIMAL_CAM_REL_POSE_DIM,),)).validate(
        observation["cam_rel_poses"], "cam_rel_poses"
    )
    cameras = observation["cameras"]
    if not isinstance(cameras, Mapping) or set(cameras) != set(
        CAMERA_RESOLUTIONS
    ):
        raise ObservationContractError(
            "three-RGB public camera roles do not match"
        )
    for role, (height, width) in CAMERA_RESOLUTIONS.items():
        camera = cameras[role]
        if not isinstance(camera, Mapping) or set(camera) != {"rgb"}:
            raise ObservationContractError(
                f"three-RGB camera {role!r} modalities do not match"
            )
        ArrayContract(((height, width, 3),)).validate(
            camera["rgb"], f"{role}.rgb"
        )


def schema_snapshot(
    *,
    expected_task_id: int = TASK_ID,
    head_eval: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": (
            THREE_RGBD_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION
            if head_eval
            else OBSERVATION_SCHEMA_VERSION
        ),
        "task_id": expected_task_id,
        "fields": {
            field: [list(shape) for shape in contract.shapes]
            for field, contract in sorted(
                official_field_contracts(head_eval=head_eval).items()
            )
        },
    }


def schema_sha256(
    *,
    expected_task_id: int = TASK_ID,
    head_eval: bool = False,
) -> str:
    canonical = json.dumps(
        schema_snapshot(
            expected_task_id=expected_task_id,
            head_eval=head_eval,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def head_rgb_minimal_schema_snapshot(
    *,
    expected_task_id: int = TASK_ID,
) -> dict[str, Any]:
    return {
        "schema_version": HEAD_RGB_MINIMAL_OBSERVATION_SCHEMA_VERSION,
        "task_id": expected_task_id,
        "fields": {
            field: [list(shape) for shape in contract.shapes]
            for field, contract in sorted(
                head_rgb_minimal_field_contracts().items()
            )
        },
    }


def head_rgb_minimal_schema_sha256(
    *,
    expected_task_id: int = TASK_ID,
) -> str:
    canonical = json.dumps(
        head_rgb_minimal_schema_snapshot(
            expected_task_id=expected_task_id,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def three_rgb_head_eval_schema_snapshot(
    *,
    expected_task_id: int = TASK_ID,
) -> dict[str, Any]:
    return {
        "schema_version": THREE_RGB_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION,
        "task_id": expected_task_id,
        "fields": {
            field: [list(shape) for shape in contract.shapes]
            for field, contract in sorted(
                three_rgb_head_eval_field_contracts().items()
            )
        },
    }


def three_rgb_head_eval_schema_sha256(
    *,
    expected_task_id: int = TASK_ID,
) -> str:
    canonical = json.dumps(
        three_rgb_head_eval_schema_snapshot(
            expected_task_id=expected_task_id,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _single_integer(value: Any, field: str) -> int:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list) or len(value) != 1:
        raise ObservationContractError(f"{field!r} must contain one integer")
    item = value[0]
    if isinstance(item, bool) or not isinstance(item, int):
        raise ObservationContractError(f"{field!r} must contain one integer")
    return item
