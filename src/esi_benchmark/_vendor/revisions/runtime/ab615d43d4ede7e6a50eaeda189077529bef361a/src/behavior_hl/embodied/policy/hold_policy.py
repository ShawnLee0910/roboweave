"""Sensor-derived 23D no-motion action for the G1 observation canary."""

from __future__ import annotations

import numpy as np

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


LEFT_ARM_QPOS = slice(3, 10)
RIGHT_ARM_QPOS = slice(28, 35)
TRUNK_QPOS = slice(53, 57)


def build_open_gripper_hold_action(
    public_observation: dict,
    *,
    observation_profile: str = FULL_CAMERA_PROFILE,
    expected_task_id: int = TASK_ID,
) -> np.ndarray:
    """Hold current arm/torso joints, stop the base, and keep grippers open."""

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
        raise ValueError("unknown hold observation profile")
    proprio = np.asarray(public_observation["proprio"], dtype=np.float32)
    if not np.isfinite(proprio).all():
        raise ValueError("proprio contains non-finite values")

    action = np.zeros(R1PRO_ACTION_LAYOUT.dimension, dtype=np.float32)
    action[R1PRO_ACTION_LAYOUT.slice_for("torso")] = proprio[TRUNK_QPOS]
    action[R1PRO_ACTION_LAYOUT.slice_for("left_arm")] = proprio[LEFT_ARM_QPOS]
    action[R1PRO_ACTION_LAYOUT.slice_for("right_arm")] = proprio[RIGHT_ARM_QPOS]

    # The official smooth position grippers map +1 to their open joint limit.
    # G1 starts from the official open reset configuration and performs no grasp.
    action[R1PRO_ACTION_LAYOUT.slice_for("left_gripper")] = 1.0
    action[R1PRO_ACTION_LAYOUT.slice_for("right_gripper")] = 1.0

    if action.shape != (23,) or not np.isfinite(action).all():
        raise ValueError("generated hold action is not a finite 23D vector")
    return action
