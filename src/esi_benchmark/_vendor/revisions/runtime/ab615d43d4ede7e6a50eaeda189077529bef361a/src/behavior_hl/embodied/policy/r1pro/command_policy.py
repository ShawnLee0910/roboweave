"""Safe 23D command synthesis from the public R1Pro observation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from behavior_hl.embodied.policy.hold_policy import (
    LEFT_ARM_QPOS,
    RIGHT_ARM_QPOS,
    TRUNK_QPOS,
    build_open_gripper_hold_action,
)
from behavior_hl.embodied.policy.observations import (
    TASK_ID,
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
from behavior_hl.embodied.policy.r1pro.action_layout import (
    R1PRO_ACTION_LAYOUT,
)


BASE_QVEL = slice(0, 3)
TRUNK_QVEL = slice(57, 61)
MAX_BASE_COMMAND_ABS = np.array([1.0, 1.0, 1.0], dtype=np.float32)

TORSO_PROFILES = {
    "upright": np.array([0.45, -0.40, 0.00, 0.00], dtype=np.float32),
    "neutral": np.array([1.025, -1.45, -0.47, 0.00], dtype=np.float32),
    "downward": np.array([1.60, -2.50, -0.94, 0.00], dtype=np.float32),
}


def torso_profile_from_translate(translate: float) -> np.ndarray:
    """Match JoyLo's official upright-to-downward interpolation."""

    if not np.isfinite(translate) or not 0.0 <= translate <= 1.0:
        raise ValueError("torso translate must be finite and in [0, 1]")
    return (
        (1.0 - float(translate)) * TORSO_PROFILES["upright"]
        + float(translate) * TORSO_PROFILES["downward"]
    ).astype(np.float32)


def _finite_vector(
    value: Sequence[float] | np.ndarray,
    *,
    length: int,
    field: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (length,):
        raise ValueError(f"{field} must have shape ({length},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains non-finite values")
    return array


def build_public_command_action(
    public_observation: dict,
    *,
    base_command: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    torso_target: Sequence[float] | np.ndarray | None = None,
    max_torso_delta_per_step: float = 0.025,
    left_gripper_command: float = 1.0,
    right_gripper_command: float = 1.0,
    observation_profile: str = FULL_CAMERA_PROFILE,
    expected_task_id: int = TASK_ID,
) -> np.ndarray:
    """Build one physical controller command without simulator-side state."""

    if observation_profile == HEAD_RGB_MINIMAL_CAMERA_PROFILE:
        if "simulator_step" not in public_observation:
            raise ValueError("head-RGB observation omits simulator_step")
        validate_head_rgb_minimal_observation(
            {
                key: value
                for key, value in public_observation.items()
                if key != "simulator_step"
            },
            expected_task_id=expected_task_id,
        )
    elif observation_profile == THREE_RGB_HEAD_EVAL_CAMERA_PROFILE:
        validate_three_rgb_head_eval_observation(
            public_observation,
            require_simulator_step=True,
            expected_task_id=expected_task_id,
        )
    elif observation_profile in {
        FULL_CAMERA_PROFILE,
        THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
    }:
        validate_public_observation(
            public_observation,
            require_simulator_step=True,
            expected_task_id=expected_task_id,
            head_eval=(
                observation_profile == THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE
            ),
        )
    else:
        raise ValueError("unknown command observation profile")
    base = _finite_vector(
        base_command,
        length=3,
        field="base_command",
    )
    if np.any(np.abs(base) > MAX_BASE_COMMAND_ABS):
        raise ValueError("base_command exceeds the official normalized limits")
    if (
        not np.isfinite(max_torso_delta_per_step)
        or max_torso_delta_per_step <= 0.0
        or max_torso_delta_per_step > 0.05
    ):
        raise ValueError(
            "max_torso_delta_per_step must be in the interval (0, 0.05]"
        )
    grippers = _finite_vector(
        (left_gripper_command, right_gripper_command),
        length=2,
        field="gripper_commands",
    )
    if np.any(np.abs(grippers) > 1.0):
        raise ValueError("gripper commands exceed normalized limits")

    action = build_open_gripper_hold_action(
        public_observation,
        observation_profile=observation_profile,
        expected_task_id=expected_task_id,
    )
    action[R1PRO_ACTION_LAYOUT.slice_for("base")] = base
    if torso_target is not None:
        target = _finite_vector(
            torso_target,
            length=4,
            field="torso_target",
        )
        proprio = np.asarray(
            public_observation["proprio"],
            dtype=np.float32,
        )
        current = proprio[TRUNK_QPOS]
        delta = np.clip(
            target - current,
            -max_torso_delta_per_step,
            max_torso_delta_per_step,
        )
        action[R1PRO_ACTION_LAYOUT.slice_for("torso")] = current + delta
    action[R1PRO_ACTION_LAYOUT.slice_for("left_gripper")] = grippers[0]
    action[R1PRO_ACTION_LAYOUT.slice_for("right_gripper")] = grippers[1]

    # Arm position controllers always hold the observed positions unless a
    # later manipulation macro explicitly supplies bounded IK targets.
    proprio = np.asarray(public_observation["proprio"], dtype=np.float32)
    action[R1PRO_ACTION_LAYOUT.slice_for("left_arm")] = proprio[LEFT_ARM_QPOS]
    action[R1PRO_ACTION_LAYOUT.slice_for("right_arm")] = proprio[
        RIGHT_ARM_QPOS
    ]
    if action.shape != (23,) or not np.isfinite(action).all():
        raise ValueError("generated command is not a finite 23D vector")
    return action
