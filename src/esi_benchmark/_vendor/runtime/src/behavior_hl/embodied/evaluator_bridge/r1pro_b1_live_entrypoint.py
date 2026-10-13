"""Zero-scene composition boundary for the R1Pro B1 single-load runner.

This module deliberately imports no OmniGibson package at module import time.
It turns the generic :mod:`run_b1_single_load` callbacks into strict live
callbacks for:

* one official ``Evaluator.step`` per requested simulator tick;
* sanitized onboard RGB-D / proprio observations;
* private same-scene reset snapshots;
* observation-only TokenPlan approach followed by one physical grasp/lift.

The post-reset episode plan is injected.  In particular, this module does not
invent a ``navigate_to`` witness, does not treat a resident TokenPlan policy as
an Oracle, and does not claim that G3b calibration completes a toy-box leaf.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
import hashlib
import gc
import inspect
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.private_invariants import (
    capture_reset_snapshot,
)
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
)
from behavior_hl.embodied.evaluator_bridge.private_observation_ingress import (
    capture_private_head_frame,
    declassify_official_observation,
)
from behavior_hl.embodied.evaluator_bridge.official_predicate_identity import (
    build_official_goal_leaf_ids,
)
from behavior_hl.embodied.evaluator_bridge.inside_goal_graph import (
    InsideGoalGraphError,
    grounded_goal_option_identity,
)
from behavior_hl.embodied.evaluator_bridge.grounded_option_identity_owner import (
    GroundedOptionIdentityOwnerError,
    resolve_grounded_option_identity_owner,
)
from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (
    get_inside_task_profile,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_single_load_bindings import (
    R1ProB1SingleLoadCalibrationConfig,
    R1ProB1SingleLoadLiveBindings,
    R1ProFrozenBasePoseToken,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_observation_only_grasp_lift import (
    ObservationOnlyApproachResult,
    ObservationOnlyGraspLiftError,
    PhysicalGraspLiftConfig,
    PublicApproachSnapshot,
    PublicApproachStep,
    PublicRGBDGroundingConfig,
    SharedControlApproachMacroExecutor,
    run_observation_only_approach,
    run_physical_grasp_lift,
)
from behavior_hl.embodied.evaluator_bridge.same_scene_reset_attestation import (
    PrivateResetSnapshot,
)
from behavior_hl.embodied.policy.approach import ApproachLimits
from behavior_hl.embodied.policy.approach_execution import BaseMacroConfig
from behavior_hl.embodied.policy.observations import (
    CAMERA_RESOLUTIONS,
    CAMERA_SENSOR_NAMES,
    HEAD_RGB_MINIMAL_CAMERA_PROFILE,
    TASK_ID,
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGBD_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION,
    sanitize_head_rgb_minimal_observation,
    sanitize_official_observation,
    sanitize_three_rgb_head_eval_observation,
    validate_head_rgb_minimal_observation,
    validate_public_observation,
    validate_three_rgb_head_eval_observation,
)
from behavior_hl.embodied.runtime.public_single_leaf_witness import (
    PublicSingleLeafWitness,
    PublicWitnessDecision,
)
from behavior_hl.embodied.runtime.observation import PublicRobotObservation
from behavior_hl.embodied.policy.profiles import (
    build_picking_up_toys_registry,
)


_ACTIVE_STARTER_JOINT_MOTION_MACRO: Any | None = None
_STARTER_JOINT_MOTION_POISONED_REASON: str | None = None
_FULL_OBSERVATION_PROFILE = "r1pro_rgbd_fullres"


def _set_starter_macro_value(macro: Any, name: str, value: Any) -> None:
    """Set one Starter macro through OmniGibson's supported unlock scope.

    ``starter_semantic_action_primitives.m`` is a ``MacroDict`` in the live
    runtime.  Once a macro has been read, direct assignment is intentionally
    rejected.  Lightweight unit-test doubles are plain namespaces, so retain
    that path without importing OmniGibson into this zero-scene module.
    """

    unlocked = getattr(macro, "unlocked", None)
    if callable(unlocked):
        with unlocked():
            setattr(macro, name, value)
    else:
        setattr(macro, name, value)


@contextmanager
def _temporary_starter_joint_motion_steps(
    macro: Any,
    *,
    expected_steps: int = 10,
    temporary_steps: int = 30,
) -> Iterator[dict[str, Any]]:
    """Temporarily extend official per-waypoint settling, then restore it.

    The official 0.005 joint threshold and motion implementation are left
    untouched.  The guard is intentionally non-reentrant because the macro is
    process-global even though the isolated worker is single threaded.
    """

    if isinstance(expected_steps, bool) or expected_steps <= 0:
        raise R1ProB1LiveEntrypointError("expected_steps must be positive")
    if isinstance(temporary_steps, bool) or temporary_steps <= expected_steps:
        raise R1ProB1LiveEntrypointError(
            "temporary_steps must be greater than expected_steps"
        )
    global _ACTIVE_STARTER_JOINT_MOTION_MACRO
    global _STARTER_JOINT_MOTION_POISONED_REASON
    if _STARTER_JOINT_MOTION_POISONED_REASON is not None:
        raise R1ProB1LiveEntrypointError(
            "Starter joint-motion convergence scope is permanently poisoned "
            f"for this process: {_STARTER_JOINT_MOTION_POISONED_REASON}"
        )
    if _ACTIVE_STARTER_JOINT_MOTION_MACRO is not None:
        raise R1ProB1LiveEntrypointError(
            "Starter joint-motion step override may not be nested"
        )
    observed = getattr(macro, "MAX_STEPS_FOR_JOINT_MOTION", None)
    if observed != expected_steps:
        raise R1ProB1LiveEntrypointError(
            "official MAX_STEPS_FOR_JOINT_MOTION drifted: "
            f"observed={observed!r}, expected={expected_steps}"
        )
    observed_threshold = getattr(macro, "JOINT_POS_DIFF_THRESHOLD", None)
    if observed_threshold != 0.005:
        raise R1ProB1LiveEntrypointError(
            "official JOINT_POS_DIFF_THRESHOLD drifted: "
            f"observed={observed_threshold!r}, expected=0.005"
        )
    record = {
        "official_steps_before": expected_steps,
        "temporary_steps": temporary_steps,
        "official_joint_threshold_before": observed_threshold,
        "official_joint_threshold_unchanged": True,
        "restored": False,
    }
    # Do not publish the process-global active marker until the live MacroDict
    # accepted the temporary value.  A failed acquisition must leave the
    # process able to perform a safe restore or a later independent attempt.
    _set_starter_macro_value(
        macro,
        "MAX_STEPS_FOR_JOINT_MOTION",
        temporary_steps,
    )
    _ACTIVE_STARTER_JOINT_MOTION_MACRO = macro
    try:
        yield record
    finally:
        restoration_error: Exception | None = None
        try:
            _set_starter_macro_value(
                macro,
                "MAX_STEPS_FOR_JOINT_MOTION",
                expected_steps,
            )
        except Exception as exc:  # pragma: no cover - live fail-closed guard
            restoration_error = exc
        steps_after = getattr(macro, "MAX_STEPS_FOR_JOINT_MOTION", None)
        threshold_after = getattr(macro, "JOINT_POS_DIFF_THRESHOLD", None)
        record["official_steps_after"] = steps_after
        record["restored"] = (
            restoration_error is None and steps_after == expected_steps
        )
        record["official_joint_threshold_after"] = threshold_after
        record["official_joint_threshold_unchanged"] = (
            threshold_after == observed_threshold == 0.005
        )
        record["fatal_threshold_drift"] = not record[
            "official_joint_threshold_unchanged"
        ]
        if restoration_error is not None or not record["restored"]:
            _STARTER_JOINT_MOTION_POISONED_REASON = (
                "MAX_STEPS_FOR_JOINT_MOTION could not be restored exactly"
            )
        elif record["fatal_threshold_drift"]:
            _STARTER_JOINT_MOTION_POISONED_REASON = (
                "JOINT_POS_DIFF_THRESHOLD changed inside a live scope"
            )
        # Clear this guard before raising any restoration error.  The poison
        # marker, not a stale nested-scope marker, owns all subsequent refusal.
        _ACTIVE_STARTER_JOINT_MOTION_MACRO = None
        if restoration_error is not None or not record["restored"]:
            detail = (
                repr(restoration_error)
                if restoration_error is not None
                else f"observed={steps_after!r}"
            )
            raise R1ProB1LiveEntrypointError(
                "official MAX_STEPS_FOR_JOINT_MOTION restoration failed: "
                f"{detail}"
            ) from restoration_error
        if not record["official_joint_threshold_unchanged"]:
            raise R1ProB1LiveEntrypointError(
                "official JOINT_POS_DIFF_THRESHOLD changed inside hand-motion scope"
            )


R1PRO_B1_LIVE_ENTRYPOINT_VERSION = (
    "behavior2026.r1pro-b1-live-entrypoint/v2"
)
R1PRO_EVAL_MIXED_Q_TO_ACTION_ADAPTER_VERSION = (
    "behavior2026.r1pro-eval-mixed-q-to-action/v2"
)
_EXCLUDED_RESET_FIELDS = (
    "simulator_clock",
    "frame_counter",
    "timestamp",
    "random_generator_state",
    "renderer_state",
)
_AUDITED_CUROBO_SOURCE_SHA256 = {
    "types/math.py": (
        "6865d911444ddc5ee8e10e86e12eebf757e36793ed07cf6c5d733d1e0a1ab9a4"
    ),
    "cuda_robot_model/kinematics_parser.py": (
        "3da973965de968115b1a23de0a3a755675108aa01edb46563e74a3845c2d6516"
    ),
    "wrap/reacher/motion_gen.py": (
        "f5c0fae61dda206025f30b9747e2c30bbda820b5e43f3ec16e2c92d3c0c6d2e9"
    ),
    "graph/graph_base.py": (
        "e0b33ea59d15cb93c26957c8db1f64a2683b2df6d64c8a79e0131572de2bb5f9"
    ),
    "rollout/arm_base.py": (
        "a3e632a66f274604d6adf3f1ab59ea0ac1333933710976d893633b0ee8a510f3"
    ),
    "cuda_robot_model/usd_kinematics_parser.py": (
        "100a55234da5b8b1a325c5bdf97809696622676a407aeb296c3406c1065b4b8e"
    ),
    "cuda_robot_model/cuda_robot_generator.py": (
        "260df6aa18ddf9ae2a7c9dee098613a079d9d8ad5bed92fba394f0515c40599d"
    ),
}
_AUDITED_CUROBO_POSE_REDIRECT_SITES = {
    ("motion_gen.py", 4284),
    ("motion_gen.py", 4286),
    ("kinematics_parser.py", 65),
}


class R1ProB1LiveEntrypointError(RuntimeError):
    """Fail-closed live callback or composition error."""


def _world_base_candidate_joint_position(
    *,
    robot: Any,
    pose_2d: Any,
    current_joint_pos: Any,
    transform_utils: Any,
    torch_module: Any,
) -> Any:
    """Convert world ``x,y,yaw`` to R1Pro root-relative base joints.

    This is the motionless tensor-only portion of official
    ``Robot.set_position_orientation(frame="world")``.  Keeping it at module
    scope makes the exact runtime conversion numerically unit-testable without
    importing OmniGibson or constructing a scene.
    """

    candidate = torch_module.as_tensor(
        pose_2d,
        dtype=current_joint_pos.dtype,
        device=current_joint_pos.device,
    )
    if candidate.shape != (3,) or not bool(torch_module.isfinite(candidate).all()):
        raise R1ProB1LiveEntrypointError(
            "sampled world base candidate must be finite x,y,yaw"
        )
    # Mirror StarterSemanticActionPrimitives._get_robot_pose_from_2d_pose
    # exactly: a holonomic base candidate changes only world x/y/yaw while
    # preserving the currently measured base z/roll/pitch.  The v87 helper
    # accidentally collision-checked a levelled pose even though the official
    # Symbolic apply path preserves measured tilt.
    current_base_joint_pos = current_joint_pos[robot.base_idx]
    desired_position = torch_module.stack(
        (candidate[0], candidate[1], current_base_joint_pos[2])
    )
    desired_orientation = transform_utils.mat2quat(
        transform_utils.euler_intrinsic2mat(
            torch_module.stack(
                (
                    current_base_joint_pos[3],
                    current_base_joint_pos[4],
                    candidate[2],
                )
            )
        )
    )
    root_position, root_orientation = robot.root_link.get_position_orientation()
    inverse_root_position, inverse_root_orientation = (
        transform_utils.invert_pose_transform(root_position, root_orientation)
    )
    relative_position, relative_orientation = transform_utils.pose_transform(
        inverse_root_position,
        inverse_root_orientation,
        desired_position,
        desired_orientation,
    )
    relative_intrinsic_eulers = transform_utils.mat2euler_intrinsic(
        transform_utils.quat2mat(relative_orientation)
    )
    candidate_joint_pos = current_joint_pos.clone()
    candidate_joint_pos[robot.base_idx] = torch_module.concatenate(
        (relative_position, relative_intrinsic_eulers)
    )
    return candidate_joint_pos


def _install_r1pro_eval_mixed_q_to_action_adapter(
    robot: Any,
    *,
    _runtime: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Install a worker-local adapter for official Starter joint trajectories.

    The official 2026 evaluation configuration combines an absolute-position
    arm / trunk, a velocity-driven holonomic base, and scalar smooth
    ``MultiFingerGripperController`` instances.  ``Robot.q_to_action`` still
    asserts that every controller is an absolute ``JointController``.  This
    adapter preserves the official conversion verbatim for all joint/base
    controllers and adds only the representational two-finger-to-one-scalar
    conversion required by the official 23D grippers.  Motion trajectories do
    not plan gripper DOFs, so each gripper uses its controller-native no-op.
    The local Starter subclass drives explicit close/open commands separately;
    this avoids lossy projection when real contact makes two finger positions
    asymmetric.
    """

    if _runtime is None:
        import torch as th
        import omnigibson.utils.transform_utils as T
        from omnigibson.controllers import (
            ControllerView,
            HolonomicBaseJointController,
            JointController,
            MultiFingerGripperController,
        )
        from omnigibson.utils.geometry_utils import wrap_angle
    else:
        required = {
            "torch",
            "transform_utils",
            "controller_view",
            "holonomic_controller_type",
            "joint_controller_type",
            "multifinger_controller_type",
            "gripper_metadata",
            "wrap_angle",
        }
        if set(_runtime) != required:
            raise ValueError(
                "injected mixed-controller runtime keys are invalid"
            )
        th = _runtime["torch"]
        T = _runtime["transform_utils"]
        ControllerView = _runtime["controller_view"]
        HolonomicBaseJointController = _runtime[
            "holonomic_controller_type"
        ]
        JointController = _runtime["joint_controller_type"]
        MultiFingerGripperController = _runtime[
            "multifinger_controller_type"
        ]
        gripper_metadata = _runtime["gripper_metadata"]
        wrap_angle = _runtime["wrap_angle"]
    if _runtime is None:
        def gripper_metadata(group_key: str) -> Mapping[str, Any]:
            controller = ControllerView._controller_groups[group_key]
            return {
                "mode": controller._mode,
                "motor_type": controller._motor_type,
                "inverted": bool(controller._inverted),
            }

    if getattr(robot, "_behavior_hl_mixed_q_to_action_installed", False):
        raise R1ProB1LiveEntrypointError(
            "mixed q_to_action adapter may be installed only once"
        )
    if int(getattr(robot, "action_dim", -1)) != 23:
        raise R1ProB1LiveEntrypointError(
            "mixed q_to_action adapter requires the official 23D action"
        )
    if getattr(robot, "_grasping_direction", None) != "lower":
        raise R1ProB1LiveEntrypointError(
            "R1Pro eval gripper must grasp toward lower joint limits"
        )

    controller_report: dict[str, Mapping[str, Any]] = {}
    gripper_names = {"gripper_left", "gripper_right"}
    expected_order = (
        "base",
        "trunk",
        "arm_left",
        "gripper_left",
        "arm_right",
        "gripper_right",
    )
    if tuple(robot.controllers) != expected_order:
        raise R1ProB1LiveEntrypointError(
            "R1Pro controller order drifted from the frozen 23D contract: "
            f"{tuple(robot.controllers)!r}"
        )
    expected_action_indices = {
        "base": (0, 1, 2),
        "trunk": (3, 4, 5, 6),
        "arm_left": tuple(range(7, 14)),
        "gripper_left": (14,),
        "arm_right": tuple(range(15, 22)),
        "gripper_right": (22,),
    }
    expected_dof_joint_names = {
        "base": (
            "base_footprint_x_joint",
            "base_footprint_y_joint",
            "base_footprint_rz_joint",
        ),
        "trunk": tuple(f"torso_joint{index}" for index in range(1, 5)),
        "arm_left": tuple(
            f"left_arm_joint{index}" for index in range(1, 8)
        ),
        "gripper_left": (
            "left_gripper_finger_joint1",
            "left_gripper_finger_joint2",
        ),
        "arm_right": tuple(
            f"right_arm_joint{index}" for index in range(1, 8)
        ),
        "gripper_right": (
            "right_gripper_finger_joint1",
            "right_gripper_finger_joint2",
        ),
    }
    expected_command_dims = {
        "base": 3,
        "trunk": 4,
        "arm_left": 7,
        "gripper_left": 1,
        "arm_right": 7,
        "gripper_right": 1,
    }
    robot_joint_names = tuple(robot.joints.keys())
    controller_joint_names = tuple(
        getattr(robot, "dof_names_ordered", ())
    )
    if controller_joint_names != robot_joint_names:
        raise R1ProB1LiveEntrypointError(
            "R1Pro controller DOF-name order disagrees with the live "
            "articulation joint order"
        )
    if len(robot_joint_names) != 28 or len(set(robot_joint_names)) != 28:
        raise R1ProB1LiveEntrypointError(
            "R1Pro low-level joint order must contain 28 unique names"
        )
    base_idx_tuple = tuple(int(value) for value in robot.base_idx.detach().cpu())
    base_joint_local_indices = {
        name: base_idx_tuple.index(robot_joint_names.index(name))
        for name in (
            "base_footprint_x_joint",
            "base_footprint_y_joint",
            "base_footprint_rz_joint",
        )
    }
    controlled_dof_indices: set[int] = set()
    for name, (group_key, _) in robot.controllers.items():
        controller_type = ControllerView.get_controller_type_str(group_key)
        command_dim = int(ControllerView.get_command_dim(group_key))
        motor_type: str
        use_delta_commands: bool | None = None
        private_gripper_metadata: Mapping[str, Any] | None = None
        if name in gripper_names:
            private_gripper_metadata = gripper_metadata(group_key)
            if set(private_gripper_metadata) != {
                "mode",
                "motor_type",
                "inverted",
            }:
                raise R1ProB1LiveEntrypointError(
                    f"{name} private controller metadata is invalid"
                )
            motor_type = str(private_gripper_metadata["motor_type"])
        else:
            motor_type = ControllerView.get_motor_type(group_key)
            use_delta_commands = bool(
                ControllerView.get_use_delta_commands(group_key)
            )
        controller_specific_report: dict[str, Any] = {}
        dof_indices = tuple(
            int(value)
            for value in ControllerView.get_dof_idx(
                group_key
            ).detach().cpu()
        )
        if any(
            index < 0 or index >= len(robot_joint_names)
            for index in dof_indices
        ):
            raise R1ProB1LiveEntrypointError(
                f"{name} exposes an out-of-range DOF index: "
                f"{dof_indices!r}"
            )
        dof_joint_names = tuple(
            robot_joint_names[index] for index in dof_indices
        )
        action_indices = tuple(
            int(value)
            for value in robot.controller_action_idx[name].detach().cpu()
        )
        if action_indices != expected_action_indices[name]:
            raise R1ProB1LiveEntrypointError(
                f"{name} action indices drifted: {action_indices!r}"
            )
        if command_dim != expected_command_dims[name]:
            raise R1ProB1LiveEntrypointError(
                f"{name} command dimension drifted: dim={command_dim}"
            )
        if dof_joint_names != expected_dof_joint_names[name]:
            raise R1ProB1LiveEntrypointError(
                f"{name} DOF joint names drifted: "
                f"names={dof_joint_names!r}, indices={dof_indices!r}"
            )
        overlap = controlled_dof_indices.intersection(dof_indices)
        if overlap:
            raise R1ProB1LiveEntrypointError(
                f"{name} overlaps already-attested controller DOFs: "
                f"{tuple(sorted(overlap))!r}"
            )
        controlled_dof_indices.update(dof_indices)
        if name in gripper_names:
            if (
                controller_type != "MultiFingerGripperController"
                or not ControllerView.is_controller_type(
                    group_key,
                    MultiFingerGripperController,
                )
                or private_gripper_metadata["mode"] != "smooth"
                or motor_type != "position"
                or command_dim != 1
            ):
                raise R1ProB1LiveEntrypointError(
                    f"{name} is not the audited scalar smooth position "
                    "MultiFingerGripperController"
                )
            dof_count = int(ControllerView.get_dof_idx(group_key).numel())
            if dof_count != 2:
                raise R1ProB1LiveEntrypointError(
                    f"{name} must control exactly two finger joints"
                )
            input_limits = ControllerView.get_command_input_limits(group_key)
            if input_limits is None:
                raise R1ProB1LiveEntrypointError(
                    f"{name} lacks normalized command input limits"
                )
            low = np.asarray(
                input_limits[0].detach().cpu(),
                dtype=np.float64,
            ).reshape(-1)
            high = np.asarray(
                input_limits[1].detach().cpu(),
                dtype=np.float64,
            ).reshape(-1)
            if (
                low.shape != (1,)
                or high.shape != (1,)
                or not np.array_equal(low, np.asarray((-1.0,)))
                or not np.array_equal(high, np.asarray((1.0,)))
                or private_gripper_metadata["inverted"] is not False
            ):
                raise R1ProB1LiveEntrypointError(
                    f"{name} must use non-inverted normalized [-1, 1] "
                    "smooth commands"
                )
            controller_specific_report = {
                "mode": "smooth",
                "command_input_limits": [-1.0, 1.0],
                "inverted": False,
            }
        elif name == "base":
            if (
                not ControllerView.is_controller_type(
                    group_key,
                    HolonomicBaseJointController,
                )
                or motor_type != "velocity"
                or use_delta_commands
            ):
                raise R1ProB1LiveEntrypointError(
                    "base is not the audited velocity holonomic controller"
                )
        elif (
            controller_type != "JointController"
            or not ControllerView.is_controller_type(
                group_key,
                JointController,
            )
            or motor_type != "position"
            or use_delta_commands
        ):
            raise R1ProB1LiveEntrypointError(
                f"unsupported non-gripper controller {name}: "
                f"{controller_type}"
            )
        controller_report[name] = {
            "controller_type": controller_type,
            "command_dim": command_dim,
            "motor_type": motor_type,
            "use_delta_commands": use_delta_commands,
            "dof_indices": dof_indices,
            "dof_joint_names": dof_joint_names,
            "action_indices": action_indices,
            **controller_specific_report,
        }
    if not gripper_names.issubset(controller_report):
        raise R1ProB1LiveEntrypointError(
            "official left and right gripper controllers are required"
        )
    expected_controlled_joint_names = {
        joint_name
        for names in expected_dof_joint_names.values()
        for joint_name in names
    }
    actual_controlled_joint_names = {
        robot_joint_names[index] for index in controlled_dof_indices
    }
    if (
        len(controlled_dof_indices) != 25
        or actual_controlled_joint_names != expected_controlled_joint_names
    ):
        raise R1ProB1LiveEntrypointError(
            "R1Pro controller groups must cover exactly the 25 audited "
            "semantic DOFs"
        )
    base_control_indices = tuple(
        int(value) for value in robot.base_control_idx.detach().cpu()
    )
    if (
        len(base_control_indices) != 3
        or base_control_indices != controller_report["base"]["dof_indices"]
    ):
        raise R1ProB1LiveEntrypointError(
            "R1Pro base_control_idx must match the attested three-DOF base"
        )

    original_q_to_action = robot.q_to_action
    had_instance_q_to_action = "q_to_action" in vars(robot)
    original_instance_q_to_action = vars(robot).get("q_to_action")

    def restore_original_q_to_action() -> None:
        if had_instance_q_to_action:
            robot.q_to_action = original_instance_q_to_action
        elif "q_to_action" in vars(robot):
            delattr(robot, "q_to_action")
        for field in (
            "_behavior_hl_mixed_q_to_action_installed",
            "_behavior_hl_original_q_to_action",
            "_behavior_hl_restore_q_to_action",
        ):
            if hasattr(robot, field):
                delattr(robot, field)

    def mixed_q_to_action(instance: Any, q: Any) -> Any:
        if (
            not bool(th.is_tensor(q))
            or
            tuple(getattr(q, "shape", ())) != (28,)
            or not bool(th.isfinite(q).all().item())
        ):
            raise R1ProB1LiveEntrypointError(
                "mixed q_to_action requires a finite 28D low-level target"
            )
        if getattr(instance, "_behavior_hl_motion_convergence_active", False):
            freeze_active = bool(
                getattr(instance, "_behavior_hl_freeze_base_q_to_action", False)
            )
            freeze_authorized = bool(
                getattr(
                    instance,
                    "_behavior_hl_authorized_frozen_base_arm_motion",
                    False,
                )
            )
            if freeze_active and not freeze_authorized:
                raise R1ProB1LiveEntrypointError(
                    "frozen-base arm motion is active outside its authorized scope"
                )
            current_q = instance.get_joint_positions()
            if (
                not bool(th.is_tensor(current_q))
                or tuple(getattr(current_q, "shape", ())) != (28,)
                or not bool(th.isfinite(current_q).all().item())
            ):
                raise R1ProB1LiveEntrypointError(
                    "motion convergence telemetry requires finite live 28D joints"
                )
            target_native = np.asarray(q.detach().cpu(), dtype=np.float32)
            target_token = hashlib.sha256(target_native.tobytes()).hexdigest()[:16]
            diagnostic_state = getattr(
                instance, "_behavior_hl_motion_convergence_state", None
            )
            if not isinstance(diagnostic_state, dict):
                diagnostic_state = {"target_token": None, "target_call": 0}
                instance._behavior_hl_motion_convergence_state = diagnostic_state
            if diagnostic_state["target_token"] != target_token:
                diagnostic_state["target_token"] = target_token
                diagnostic_state["target_call"] = 0
            diagnostic_state["target_call"] += 1
            target_call = int(diagnostic_state["target_call"])
            if target_call in {1, 11, 21, 30}:
                articulation_indices = tuple(
                    sorted(
                        index
                        for controller_name in ("trunk", "arm_left", "arm_right")
                        for index in controller_report[controller_name]["dof_indices"]
                    )
                )
                diff = q - current_q
                articulation_abs = th.abs(diff[list(articulation_indices)])
                worst_local = int(th.argmax(articulation_abs).item())
                worst_index = articulation_indices[worst_local]
                base_diff = diff[list(base_control_indices)]
                samples = getattr(
                    instance, "_behavior_hl_motion_convergence_samples", None
                )
                if not isinstance(samples, list):
                    samples = []
                    instance._behavior_hl_motion_convergence_samples = samples
                if len(samples) < 4096:
                    samples.append(
                        {
                            "target_token": target_token,
                            "target_call_pre_step": target_call,
                            "would_have_exhausted_official_10_steps": target_call == 11,
                            "max_articulation_joint_diff_pre_step": float(
                                articulation_abs[worst_local].item()
                            ),
                            "worst_joint_index": worst_index,
                            "worst_joint_name": robot_joint_names[worst_index],
                            "max_base_position_diff_pre_step": float(
                                th.max(th.abs(base_diff[:2])).item()
                            ),
                            "base_yaw_diff_pre_step": float(
                                th.abs(wrap_angle(base_diff[2])).item()
                            ),
                        }
                    )
        action_parts = []
        for name, (group_key, controller_idx) in instance.controllers.items():
            dof_idx = ControllerView.get_dof_idx(group_key)
            if name in gripper_names:
                # compute_no_op_action is already returned in normalized
                # action/input space.  Do not reverse-preprocess it twice.
                action_parts.append(
                    ControllerView.compute_no_op_action(
                        group_key,
                        controller_idx,
                    )
                )
                continue
            command = q[dof_idx]
            if ControllerView.is_controller_type(
                group_key,
                HolonomicBaseJointController,
            ):
                base_joint_pos = instance.get_joint_positions()[
                    instance.base_idx
                ]
                # CuRobo ARM plans are reconstructed as full 28D joint
                # waypoints, but their locked base entries can be a stale
                # planner snapshot.  After the candidate boundary the base
                # is deliberately frozen; replace only the base q target in
                # this scoped context before applying the normal local-frame
                # velocity conversion.  Candidate-before navigation never
                # sets this flag and retains the official base behavior.
                if getattr(
                    instance,
                    "_behavior_hl_freeze_base_q_to_action",
                    False,
                ):
                    command = th.stack(
                        (
                            base_joint_pos[
                                base_joint_local_indices["base_footprint_x_joint"]
                            ],
                            base_joint_pos[
                                base_joint_local_indices["base_footprint_y_joint"]
                            ],
                            base_joint_pos[
                                base_joint_local_indices["base_footprint_rz_joint"]
                            ],
                        )
                    )
                current_rz = base_joint_pos[5]
                delta_q = wrap_angle(command[2] - current_rz)
                body_pos = base_joint_pos[:3]
                body_quat = T.mat2quat(
                    T.euler_intrinsic2mat(base_joint_pos[3:6])
                )
                canonical_pos = th.stack(
                    (command[0], command[1], body_pos[2])
                )
                local_pos = T.relative_pose_transform(
                    canonical_pos,
                    body_quat.new_tensor((0.0, 0.0, 0.0, 1.0)),
                    body_pos,
                    body_quat,
                )[0]
                command = th.stack((local_pos[0], local_pos[1], delta_q))
            action_parts.append(
                ControllerView.reverse_preprocess_command(
                    group_key,
                    command,
                )
            )
        action = th.cat(action_parts, dim=0)
        if (
            tuple(action.shape) != (23,)
            or not bool(th.isfinite(action).all().item())
        ):
            raise R1ProB1LiveEntrypointError(
                "mixed q_to_action adapter produced a non-finite or non-23D "
                "action"
            )
        return action

    robot.q_to_action = MethodType(mixed_q_to_action, robot)
    try:
        dry_action = robot.q_to_action(robot.get_joint_positions())
    except Exception:
        restore_original_q_to_action()
        raise
    if (
        tuple(dry_action.shape) != (23,)
        or not bool(th.isfinite(dry_action).all().item())
    ):
        restore_original_q_to_action()
        raise R1ProB1LiveEntrypointError(
            "mixed q_to_action dry-run attestation failed"
        )
    robot._behavior_hl_mixed_q_to_action_installed = True
    robot._behavior_hl_original_q_to_action = original_q_to_action
    robot._behavior_hl_restore_q_to_action = restore_original_q_to_action
    return {
        "schema_version": R1PRO_EVAL_MIXED_Q_TO_ACTION_ADAPTER_VERSION,
        "action_dim": 23,
        "gripper_motion_plan_policy": "controller_native_no_op",
        "scope": "resident_worker_robot_instance",
        "exact_restore_available": True,
        "controlled_dof_count": len(controlled_dof_indices),
        "controllers": controller_report,
        "dry_run_finite": True,
    }


def _attest_curobo_gripper_joints_locked(
    *,
    motion_generator: Any,
    robot: Any,
) -> Mapping[str, Any]:
    """Prove every live CuRobo embodiment treats both grippers as locked."""

    robot_joint_names = tuple(robot.joints.keys())
    expected_gripper_joints = {
        robot_joint_names[int(index)]
        for arm in ("left", "right")
        for index in robot.gripper_control_idx[arm].detach().cpu()
    }
    if len(expected_gripper_joints) != 4:
        raise R1ProB1LiveEntrypointError(
            "R1Pro must expose four distinct gripper joint names"
        )
    embodiments: dict[str, Mapping[str, Any]] = {}
    generators = getattr(motion_generator, "mg", None)
    if not isinstance(generators, Mapping) or not generators:
        raise R1ProB1LiveEntrypointError(
            "CuRobo motion generator exposes no embodiments"
        )
    for embodiment, generator in generators.items():
        kinematics = getattr(generator, "kinematics", None)
        active = set(getattr(kinematics, "joint_names", ()))
        config = getattr(kinematics, "kinematics_config", None)
        lock_state = getattr(config, "lock_jointstate", None)
        locked = set(getattr(lock_state, "joint_names", ()))
        missing = expected_gripper_joints.difference(locked)
        planned = expected_gripper_joints.intersection(active)
        if missing or planned:
            raise R1ProB1LiveEntrypointError(
                "CuRobo gripper joint contract drifted for embodiment "
                f"{embodiment!s}: missing_locked={sorted(missing)}, "
                f"planned={sorted(planned)}"
            )
        embodiments[str(embodiment)] = {
            "locked_gripper_joints": sorted(expected_gripper_joints),
            "active_gripper_joints": [],
        }
    return {
        "expected_gripper_joints": sorted(expected_gripper_joints),
        "embodiments": embodiments,
        "all_grippers_locked_and_unplanned": True,
    }


@contextmanager
def _selected_device_curobo_pose_from_list(
    *,
    pose_type: Any,
    selected_tensor_args: Any,
) -> Iterator[list[dict[str, Any]]]:
    """Scope CuRobo's hard-coded ``Pose.from_list`` cuda0 defaults.

    The pinned CuRobo fork constructs two persistent default ``Pose`` values
    while importing ``motion_gen`` and constructs R1Pro ``extra_links`` through
    the same API.  Both paths omit the caller's tensor device (or explicitly
    create the default ``TensorDeviceType()``, which is cuda0).  Temporarily
    route only non-selected CUDA inputs to the already-attested worker tensor
    device, then restore the exact class descriptor.  CPU inputs are preserved.
    """

    original_descriptor = pose_type.__dict__.get("from_list")
    if not isinstance(original_descriptor, classmethod):
        raise R1ProB1LiveEntrypointError(
            "CuRobo Pose.from_list descriptor is not the audited classmethod"
        )
    original_bound = pose_type.from_list
    selected_device = str(selected_tensor_args.device)
    redirect_events: list[dict[str, Any]] = []

    @classmethod
    def selected_from_list(
        cls: Any,
        pose: Any,
        tensor_args: Any = None,
        q_xyzw: bool = False,
    ) -> Any:
        effective = tensor_args
        effective_device = str(getattr(effective, "device", ""))
        if effective is None or (
            effective_device.startswith("cuda")
            and effective_device != selected_device
        ):
            caller = inspect.currentframe()
            caller = None if caller is None else caller.f_back
            site = (
                "" if caller is None else Path(caller.f_code.co_filename).name,
                -1 if caller is None else int(caller.f_lineno),
            )
            if site not in _AUDITED_CUROBO_POSE_REDIRECT_SITES:
                raise R1ProB1LiveEntrypointError(
                    "unaudited CuRobo Pose.from_list CUDA-device redirect "
                    f"requested at {site}"
                )
            effective = selected_tensor_args
            redirect_events.append(
                {
                    "filename": site[0],
                    "line": site[1],
                    "from_device": effective_device or None,
                    "to_device": selected_device,
                }
            )
        return original_bound(
            pose,
            tensor_args=effective,
            q_xyzw=q_xyzw,
        )

    pose_type.from_list = selected_from_list
    try:
        yield redirect_events
    finally:
        pose_type.from_list = original_descriptor


def _attest_audited_curobo_sources(*, pose_type: Any) -> dict[str, str]:
    """Hash the exact installed CuRobo sources covered by the local shim."""

    math_path = Path(inspect.getfile(pose_type)).resolve()
    curobo_root = math_path.parents[1]
    observed: dict[str, str] = {}
    for relative_path, expected_hash in _AUDITED_CUROBO_SOURCE_SHA256.items():
        source_path = curobo_root / relative_path
        if not source_path.is_file():
            raise R1ProB1LiveEntrypointError(
                f"audited CuRobo source is unavailable: {relative_path}"
            )
        observed_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if observed_hash != expected_hash:
            raise R1ProB1LiveEntrypointError(
                f"audited CuRobo source hash drifted: {relative_path}"
            )
        observed[relative_path] = observed_hash
    return observed


@contextmanager
def _selected_device_curobo_arm_base_from_dict(
    *,
    arm_base_config_type: Any,
    selected_tensor_args: Any,
) -> Iterator[list[dict[str, Any]]]:
    """Repair the audited GraphConfig -> ArmBaseConfig device omission."""

    descriptor = arm_base_config_type.__dict__.get("from_dict")
    if not isinstance(descriptor, classmethod):
        raise R1ProB1LiveEntrypointError(
            "CuRobo ArmBaseConfig.from_dict is not the audited classmethod"
        )
    original_function = descriptor.__func__
    selected_device = str(selected_tensor_args.device)
    events: list[dict[str, Any]] = []

    @classmethod
    def selected_from_dict(cls: Any, *args: Any, **kwargs: Any) -> Any:
        supplied = kwargs.get("tensor_args")
        supplied_device = str(getattr(supplied, "device", ""))
        if supplied is None or (
            supplied_device.startswith("cuda")
            and supplied_device != selected_device
        ):
            caller = inspect.currentframe()
            caller = None if caller is None else caller.f_back
            site = (
                "" if caller is None else Path(caller.f_code.co_filename).name,
                -1 if caller is None else int(caller.f_lineno),
            )
            if site != ("graph_base.py", 160):
                raise R1ProB1LiveEntrypointError(
                    "unaudited ArmBaseConfig tensor-device redirect "
                    f"requested at {site}"
                )
            kwargs["tensor_args"] = selected_tensor_args
            events.append(
                {
                    "filename": site[0],
                    "line": site[1],
                    "from_device": supplied_device or None,
                    "to_device": selected_device,
                }
            )
        # Preserve classmethod polymorphism.  CuRobo's derived rollout
        # configs inherit this descriptor and rely on ``cls.cost_from_dict``
        # to construct their richer cost types.  Calling a method already
        # bound to ArmBaseConfig here would silently down-cast
        # ArmReacherConfig into ArmBaseConfig.
        return original_function(cls, *args, **kwargs)

    arm_base_config_type.from_dict = selected_from_dict
    try:
        yield events
    finally:
        arm_base_config_type.from_dict = descriptor


@contextmanager
def _selected_device_curobo_usd_parser_init(
    *,
    usd_parser_type: Any,
    selected_tensor_args: Any,
) -> Iterator[list[dict[str, Any]]]:
    """Repair the audited CudaRobotGenerator -> USD parser device omission."""

    original_init = usd_parser_type.__dict__.get("__init__")
    if not callable(original_init):
        raise R1ProB1LiveEntrypointError(
            "CuRobo UsdKinematicsParser.__init__ is unavailable"
        )
    selected_device = str(selected_tensor_args.device)
    events: list[dict[str, Any]] = []

    def selected_init(self: Any, *args: Any, **kwargs: Any) -> None:
        supplied = kwargs.get("tensor_args")
        supplied_device = str(getattr(supplied, "device", ""))
        if supplied is None or (
            supplied_device.startswith("cuda")
            and supplied_device != selected_device
        ):
            caller = inspect.currentframe()
            caller = None if caller is None else caller.f_back
            site = (
                "" if caller is None else Path(caller.f_code.co_filename).name,
                -1 if caller is None else int(caller.f_lineno),
            )
            if site != ("cuda_robot_generator.py", 319):
                raise R1ProB1LiveEntrypointError(
                    "unaudited UsdKinematicsParser tensor-device redirect "
                    f"requested at {site}"
                )
            kwargs["tensor_args"] = selected_tensor_args
            events.append(
                {
                    "filename": site[0],
                    "line": site[1],
                    "from_device": supplied_device or None,
                    "to_device": selected_device,
                }
            )
        original_init(self, *args, **kwargs)

    usd_parser_type.__init__ = selected_init
    try:
        yield events
    finally:
        usd_parser_type.__init__ = original_init


def _tracked_cuda_tensor_snapshot(
    *,
    torch_module: Any,
    device_indices: set[int],
) -> dict[int, dict[str, Any]]:
    """Describe GC-tracked live CUDA tensors on selected physical ordinals."""

    snapshot: dict[int, dict[str, Any]] = {}
    for value in gc.get_objects():
        try:
            if not isinstance(value, torch_module.Tensor):
                continue
            device = value.device
            if device.type != "cuda" or device.index not in device_indices:
                continue
            snapshot[id(value)] = {
                "device": str(device),
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
                "numel": int(value.numel()),
                "element_size": int(value.element_size()),
                "bytes": int(value.numel() * value.element_size()),
                "requires_grad": bool(value.requires_grad),
            }
        except (ReferenceError, RuntimeError):
            continue
    return snapshot


def _find_tensor_owner_paths(
    *,
    root: Any,
    target_ids: set[int],
    torch_module: Any,
    max_objects: int = 200_000,
    max_depth: int = 18,
) -> dict[int, list[str]]:
    """Find non-property object paths from a root to target Torch tensors."""

    found: dict[int, list[str]] = {target: [] for target in target_ids}
    queue: list[tuple[str, Any, int]] = [("motion_generator", root, 0)]
    seen: set[int] = set()
    visited = 0
    while queue and visited < max_objects:
        path, value, depth = queue.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        visited += 1
        if identity in target_ids:
            found[identity].append(path)
            continue
        if isinstance(value, torch_module.Tensor) or depth >= max_depth:
            continue
        children: list[tuple[str, Any]] = []
        if isinstance(value, Mapping):
            children.extend(
                (f"[{key!r}]", child) for key, child in value.items()
            )
        elif isinstance(value, (list, tuple)):
            children.extend(
                (f"[{index}]", child)
                for index, child in enumerate(value)
            )
        elif isinstance(value, set):
            children.extend(
                (f"[set:{index}]", child)
                for index, child in enumerate(value)
            )
        elif hasattr(value, "__dict__") and not (
            inspect.ismodule(value)
            or inspect.isclass(value)
            or callable(value)
        ):
            children.extend(
                (f".{key}", child)
                for key, child in vars(value).items()
            )
        for suffix, child in children:
            queue.append((path + suffix, child, depth + 1))
    return {key: paths for key, paths in found.items() if paths}


@dataclass(frozen=True, slots=True)
class R1ProObservationApproachConfig:
    """Frozen public approach controls; all geometry is onboard RGB-D."""

    grounding: PublicRGBDGroundingConfig
    limits: ApproachLimits
    base_macro: BaseMacroConfig
    command_frequency_hz: float
    base_settle_steps: int
    look_maximum_steps: int
    look_settle_steps: int
    look_tolerance_rad: float
    look_max_delta_per_step: float
    tokenplan_wall_timeout_seconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.grounding, PublicRGBDGroundingConfig):
            raise TypeError("grounding has the wrong type")
        if not isinstance(self.limits, ApproachLimits):
            raise TypeError("limits has the wrong type")
        if self.limits.max_grasp_queries != 8:
            raise ValueError("B1 approach requires exactly 8 grasp queries")
        if not isinstance(self.base_macro, BaseMacroConfig):
            raise TypeError("base_macro has the wrong type")
        for name in (
            "command_frequency_hz",
            "look_tolerance_rad",
            "look_max_delta_per_step",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        for name, minimum in (
            ("base_settle_steps", 0),
            ("look_maximum_steps", 1),
            ("look_settle_steps", 0),
            ("tokenplan_wall_timeout_seconds", 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")


class StrictEvaluatorStepOnce:
    """Advance one bound Evaluator exactly once and poison on ambiguity."""

    def __init__(
        self,
        *,
        event_writer: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if event_writer is not None and not callable(event_writer):
            raise TypeError("event_writer must be callable")
        self._event_writer = event_writer
        self._evaluator: Any | None = None
        self._step_count = 0
        self._poisoned = False

    @property
    def step_count(self) -> int:
        return self._step_count

    def __call__(self, evaluator: Any, label: str) -> int:
        if self._poisoned:
            raise R1ProB1LiveEntrypointError("stepper is poisoned")
        if self._evaluator is None:
            self._evaluator = evaluator
        elif evaluator is not self._evaluator:
            self._poisoned = True
            raise R1ProB1LiveEntrypointError(
                "stepper cannot switch Evaluator instances"
            )
        if not isinstance(label, str) or not label:
            self._poisoned = True
            raise R1ProB1LiveEntrypointError("step label must be non-empty")
        step = getattr(evaluator, "step", None)
        if not callable(step):
            self._poisoned = True
            raise R1ProB1LiveEntrypointError(
                "Evaluator does not expose step()"
            )
        try:
            result = step()
        except Exception:
            self._poisoned = True
            raise
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or any(not isinstance(item, (bool, np.bool_)) for item in result)
        ):
            self._poisoned = True
            raise R1ProB1LiveEntrypointError(
                "Evaluator.step() must return (terminated, truncated)"
            )
        self._step_count += 1
        terminated, truncated = (bool(item) for item in result)
        if self._event_writer is not None:
            self._event_writer(
                {
                    "event": "evaluator_step",
                    "step": self._step_count,
                    "label": label,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
        if terminated or truncated:
            self._poisoned = True
            raise R1ProB1LiveEntrypointError(
                "task terminated during unscored calibration"
            )
        return 1


def capture_sanitized_public_observation(
    evaluator: Any,
    *,
    expected_task_id: int = TASK_ID,
    observation_profile: str = _FULL_OBSERVATION_PROFILE,
) -> Mapping[str, Any]:
    """Copy one frozen task's profile-bound observation into public schema.

    ``expected_task_id`` is evaluator-private binding metadata supplied by the
    active frozen task profile.  ``observation_profile`` selects one complete,
    exact schema; it never fills omitted cameras or modalities.  Both schema
    identity and equality with the observed task ID remain fail-closed.
    """

    raw = getattr(evaluator, "obs", None)
    try:
        if observation_profile not in {
            _FULL_OBSERVATION_PROFILE,
            HEAD_RGB_MINIMAL_CAMERA_PROFILE,
            THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
        }:
            raise ValueError(
                f"unknown observation profile: {observation_profile!r}"
            )
        # The visibility-gated opt-in wrapper adds exactly two evaluator-only
        # head modalities.  They are removed at this single declassification
        # boundary.  Every other unknown field remains a hard error.
        public_raw = declassify_official_observation(raw)
        if observation_profile == HEAD_RGB_MINIMAL_CAMERA_PROFILE:
            public = sanitize_head_rgb_minimal_observation(
                public_raw,
                expected_task_id=expected_task_id,
            )
            copied = {
                "schema_version": public["schema_version"],
                "task_id": int(public["task_id"]),
                "proprio": _numpy_copy(public["proprio"]),
                "cam_rel_poses": _numpy_copy(public["cam_rel_poses"]),
                "cameras": {
                    "head": {
                        "rgb": _numpy_copy(
                            public["cameras"]["head"]["rgb"]
                        ),
                    }
                },
            }
            validate_head_rgb_minimal_observation(
                copied,
                expected_task_id=expected_task_id,
            )
        elif observation_profile == THREE_RGB_HEAD_EVAL_CAMERA_PROFILE:
            public = sanitize_three_rgb_head_eval_observation(
                public_raw,
                expected_task_id=expected_task_id,
            )
            copied = {
                "schema_version": public["schema_version"],
                "task_id": int(public["task_id"]),
                "proprio": _numpy_copy(public["proprio"]),
                "cam_rel_poses": _numpy_copy(public["cam_rel_poses"]),
                "cameras": {
                    role: {
                        "rgb": _numpy_copy(
                            public["cameras"][role]["rgb"]
                        ),
                    }
                    for role in CAMERA_RESOLUTIONS
                },
            }
            validate_three_rgb_head_eval_observation(
                copied,
                expected_task_id=expected_task_id,
            )
        else:
            head_eval = (
                observation_profile == THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE
            )
            public = sanitize_official_observation(
                public_raw,
                expected_task_id=expected_task_id,
                head_eval=head_eval,
            )
            copied = {
                "schema_version": public["schema_version"],
                "task_id": int(public["task_id"]),
                "proprio": _numpy_copy(public["proprio"]),
                "cam_rel_poses": _numpy_copy(public["cam_rel_poses"]),
                "cameras": {
                    role: {
                        modality: _numpy_copy(value)
                        for modality, value in public["cameras"][role].items()
                    }
                    for role in CAMERA_RESOLUTIONS
                },
            }
            validate_public_observation(
                copied,
                expected_task_id=expected_task_id,
                head_eval=head_eval,
            )
    except Exception as exc:
        raise R1ProB1LiveEntrypointError(
            "official observation failed the public RGB-D whitelist"
        ) from exc
    return copied


def capture_private_head_instance_segmentation(
    evaluator: Any,
    *,
    simulator_step: int,
    expected_task_id: int = TASK_ID,
    observation_profile: str = _FULL_OBSERVATION_PROFILE,
) -> PrivateInstanceSegmentationFrame:
    """Capture private same-observation instance masks for the head camera.

    This function is evaluator-only.  Its return value contains simulator
    instance names and must never be attached to a public observation, action
    result, model prompt, or public artifact.  RGB-only profiles obtain the
    private mask from the preinstalled side channel rather than weakening the
    exact public observation schema or synthesizing depth.
    """

    raw = getattr(evaluator, "obs", None)
    try:
        # Reuse the strict public boundary to reject any unexpected field.
        public = capture_sanitized_public_observation(
            evaluator,
            expected_task_id=expected_task_id,
            observation_profile=observation_profile,
        )
        if observation_profile in {
            THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
        }:
            from behavior_hl.embodied.evaluator_bridge.private_head_segmentation_side_channel import (
                get_preinstalled_private_head_segmentation_side_channel,
            )

            channel = get_preinstalled_private_head_segmentation_side_channel(
                evaluator=evaluator,
                head_sensor_name=CAMERA_SENSOR_NAMES["head"],
            )
            return channel.capture(
                simulator_step=simulator_step,
                public_head_rgb=public["cameras"]["head"]["rgb"],
            )
        # Imported only in a live evaluator process, never policy-side.
        from omnigibson.sensors.vision_sensor import VisionSensor

        registry = {
            int(instance_id): str(name)
            for instance_id, name in VisionSensor.INSTANCE_REGISTRY.items()
        }
        return capture_private_head_frame(
            raw=raw,
            public_head_rgb=public["cameras"]["head"]["rgb"],
            simulator_step=simulator_step,
            instance_registry=registry,
        )
    except Exception as exc:
        raise R1ProB1LiveEntrypointError(
            "private head segmentation capture failed closed"
        ) from exc


def capture_private_reset_snapshot(
    *,
    evaluator: Any,
    phase: str,
    run_dir: Path,
) -> PrivateResetSnapshot:
    """Capture evaluator-only restoration evidence and all official leaves."""

    del run_dir
    robot = getattr(evaluator, "robot", None)
    env = getattr(evaluator, "env", None)
    task = getattr(env, "task", None)
    if robot is None or task is None:
        raise R1ProB1LiveEntrypointError(
            "reset snapshot requires evaluator.robot and evaluator.env.task"
        )
    position, quaternion = _robot_pose(robot)
    joints = _finite_vector(
        robot.get_joint_positions(),
        name="robot joint positions",
    )
    held = getattr(robot, "_ag_obj_in_hand", None)
    arm_names = tuple(getattr(robot, "arm_names", ()))
    if (
        not isinstance(held, Mapping)
        or set(held) != set(arm_names)
        or set(arm_names) != {"left", "right"}
    ):
        raise R1ProB1LiveEntrypointError(
            "official assisted-grasp hand state is unavailable"
        )
    held_none = all(held[hand] is None for hand in arm_names)

    options = getattr(task, "ground_goal_state_options", None)
    evaluate = getattr(task, "_evaluate_predicate", None)
    if (
        not isinstance(options, SequenceABC)
        or isinstance(options, (str, bytes))
        or not options
        or not callable(evaluate)
    ):
        raise R1ProB1LiveEntrypointError(
            "Inside task must expose non-empty official goal options"
    )
    selected_option_index = 0
    selected_option: SequenceABC[Any] | None = None
    selected_option_selection_mode = "single_official_option"
    selected_option_identity_sha256: str | None = None
    if len(options) > 1:
        activity_name = str(getattr(task, "activity_name", ""))
        try:
            profile = get_inside_task_profile(activity_name)
            owner = resolve_grounded_option_identity_owner(
                evaluator=evaluator,
                profile=profile,
            )
            selected_option_index = owner.witness.option_index
            selected_option = owner.witness.option
            selected_option_selection_mode = owner.selection_mode
            selected_option_identity_sha256 = owner.witness.identity_sha256
        except (
            KeyError,
            InsideGoalGraphError,
            GroundedOptionIdentityOwnerError,
        ) as exc:
            raise R1ProB1LiveEntrypointError(
                "official goal options do not contain the frozen canonical "
                "profile identity or one feasible actual indexed identity"
            ) from exc
    else:
        selected_option = options[0]
    if (
        selected_option is None
        or not isinstance(selected_option, (list, tuple))
        or not selected_option
    ):
        raise R1ProB1LiveEntrypointError(
            "official grounded option selection produced no predicate set"
        )
    leaves = tuple(
        bool(predicate.evaluate(evaluate)) for predicate in selected_option
    )
    leaf_ids = build_official_goal_leaf_ids((selected_option,))[0]
    if selected_option_identity_sha256 is None:
        try:
            selected_identity = grounded_goal_option_identity(selected_option)
        except InsideGoalGraphError:
            selected_identity = tuple(leaf_ids)
        selected_option_identity_sha256 = hashlib.sha256(
            json.dumps(
                list(selected_identity),
                sort_keys=False,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    stable = capture_reset_snapshot(evaluator)
    if not isinstance(stable, Mapping):
        raise R1ProB1LiveEntrypointError(
            "stable private reset state capture failed"
        )
    stable_state = {
        "activity_name": str(getattr(task, "activity_name", "")),
        "scene_name": str(getattr(task, "scene_name", "")),
        "task_scope": stable.get("task_scope"),
        "skipped_scope": stable.get("skipped_scope"),
    }
    return PrivateResetSnapshot(
        phase=phase,
        robot_base_position=position,
        robot_base_quaternion_xyzw=quaternion,
        robot_joint_positions=joints,
        held_none=held_none,
        inside_leaf_ids=leaf_ids,
        initial_inside_leaves=leaves,
        stable_serialized_state=stable_state,
        excluded_volatile_fields=_EXCLUDED_RESET_FIELDS,
        grounded_option_index=selected_option_index,
        grounded_option_identity_sha256=selected_option_identity_sha256,
        grounded_option_selection_mode=selected_option_selection_mode,
    )


class R1ProObservationPhysicalAdapter:
    """Stateful adapter joining public approach to the physical grasp stage."""

    def __init__(
        self,
        *,
        approach_config: R1ProObservationApproachConfig,
        physical_config: PhysicalGraspLiftConfig,
        tokenplan_call: Callable[..., Any],
        calibration_live_sample_provider_factory: Callable[
            [Any, Any], Callable[..., Any]
        ],
        base_contact_provider: Callable[[Any], bool] | None = None,
    ) -> None:
        if not isinstance(
            approach_config,
            R1ProObservationApproachConfig,
        ):
            raise TypeError("approach_config has the wrong type")
        if not isinstance(physical_config, PhysicalGraspLiftConfig):
            raise TypeError("physical_config has the wrong type")
        for name, callback in (
            ("tokenplan_call", tokenplan_call),
            (
                "calibration_live_sample_provider_factory",
                calibration_live_sample_provider_factory,
            ),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        if base_contact_provider is not None and not callable(
            base_contact_provider
        ):
            raise TypeError("base_contact_provider must be callable")
        self._approach_config = approach_config
        self._physical_config = physical_config
        self._tokenplan_call = tokenplan_call
        self._sample_factory = calibration_live_sample_provider_factory
        self._base_contact_provider = (
            base_contact_provider or _official_base_contact
        )
        self._approach: ObservationOnlyApproachResult | None = None
        self._captured_public_observations: list[
            tuple[str, PublicRobotObservation]
        ] = []
        self._cached_approach_executions: list[
            tuple[Mapping[str, Any], PublicRobotObservation, str, int]
        ] = []
        self._physical_grasp_steps: int | None = None
        self._live_starter: Any | None = None
        self._poisoned = False

    def observation_only_approach(self, **kwargs: Any) -> Any:
        if self._poisoned or self._approach is not None:
            raise R1ProB1LiveEntrypointError(
                "observation approach is unavailable or already consumed"
            )
        required = {
            "evaluator",
            "control",
            "run_dir",
            "live_public_snapshot",
            "step_once",
            "capture_final_state",
        }
        if set(kwargs) != required:
            raise R1ProB1LiveEntrypointError(
                "observation approach callback fields changed"
            )
        evaluator = kwargs["evaluator"]
        control = kwargs["control"]
        run_dir = Path(kwargs["run_dir"])
        live_snapshot = kwargs["live_public_snapshot"]
        step_once = kwargs["step_once"]
        capture_index = 0

        def capture() -> PublicApproachSnapshot:
            nonlocal capture_index
            capture_index += 1
            control.hold()
            result = step_once(
                f"b1.approach.snapshot_{capture_index:03d}"
            )
            if result != 1 or isinstance(result, bool):
                raise R1ProB1LiveEntrypointError(
                    "snapshot capture did not advance exactly one step"
                )
            snapshot = live_snapshot()
            path = (
                run_dir
                / "policy"
                / "captures"
                / f"public_snapshot_{capture_index:03d}.npz"
            )
            _write_public_snapshot(path, snapshot)
            position, quaternion = _robot_pose(evaluator.robot)
            head = np.ascontiguousarray(
                snapshot["cameras"]["head"]["rgb"],
                dtype=np.uint8,
            )
            public_observation = _public_robot_observation(
                snapshot,
                simulator_step=capture_index,
            )
            self._captured_public_observations.append(
                (
                    hashlib.sha256(head.tobytes()).hexdigest(),
                    public_observation,
                )
            )
            return PublicApproachSnapshot(
                snapshot_path=path,
                snapshot=snapshot,
                source_head_rgb_sha256=hashlib.sha256(
                    head.tobytes()
                ).hexdigest(),
                robot_position=position,
                robot_orientation_xyzw=quaternion,
                simulator_steps=1,
            )

        def macro_step(
            label: str,
            allow_base_motion: bool,
        ) -> PublicApproachStep:
            if allow_base_motion is not True:
                raise R1ProB1LiveEntrypointError(
                    "approach macro unexpectedly disabled base motion"
                )
            if step_once(label) != 1:
                raise R1ProB1LiveEntrypointError(
                    "approach macro did not advance exactly one step"
                )
            public = control.public_state()
            if not isinstance(public, Mapping):
                raise R1ProB1LiveEntrypointError(
                    "SharedCanaryControl public_state is unavailable"
                )
            return PublicApproachStep(
                base_qvel=public["base_qvel"],
                head_forward_clearance_m=public[
                    "head_forward_clearance_m"
                ],
                torso_qpos=public["trunk_qpos"],
                base_contact=bool(
                    self._base_contact_provider(evaluator)
                ),
                simulator_steps=1,
            )

        macro = SharedControlApproachMacroExecutor(
            control=control,
            step=macro_step,
            base_config=self._approach_config.base_macro,
            command_frequency_hz=(
                self._approach_config.command_frequency_hz
            ),
            base_settle_steps=(
                self._approach_config.base_settle_steps
            ),
            look_maximum_steps=(
                self._approach_config.look_maximum_steps
            ),
            look_settle_steps=(
                self._approach_config.look_settle_steps
            ),
            look_tolerance_rad=(
                self._approach_config.look_tolerance_rad
            ),
            look_max_delta_per_step=(
                self._approach_config.look_max_delta_per_step
            ),
        )
        try:
            def persist_public_result(item: Any, digest: str) -> None:
                _append_jsonl(
                    run_dir / "public" / "approach_actions.jsonl",
                    {
                        **item.to_dict(),
                        "source_head_rgb_sha256": digest,
                    },
                )
                action = dict(item.action)
                if action.get("action") == "grasp_at":
                    return
                matches = [
                    observation
                    for source_hash, observation
                    in self._captured_public_observations
                    if source_hash == digest
                ]
                if not matches:
                    raise R1ProB1LiveEntrypointError(
                        "approach result has no matching public observation"
                    )
                result_mapping = dict(item.result)
                self._cached_approach_executions.append(
                    (
                        action,
                        matches[-1],
                        str(result_mapping["status"]),
                        int(result_mapping["simulator_steps"]),
                    )
                )

            result = run_observation_only_approach(
                run_dir=run_dir,
                capture_public_snapshot=capture,
                capture_final_state=kwargs["capture_final_state"],
                macro_executor=macro,
                grounding_config=self._approach_config.grounding,
                limits=self._approach_config.limits,
                tokenplan_wall_timeout_seconds=(
                    self._approach_config.tokenplan_wall_timeout_seconds
                ),
                persist_public_result=persist_public_result,
                tokenplan_call=self._tokenplan_call,
            )
        except Exception:
            self._poisoned = True
            raise
        self._approach = result
        return result.runner_evidence

    def physical_grasp_lift(self, **kwargs: Any) -> Any:
        if self._poisoned or self._approach is None:
            raise R1ProB1LiveEntrypointError(
                "physical grasp requires one completed approach"
            )
        required = {
            "evaluator",
            "control",
            "approach",
            "proposal",
            "empty_wrist",
            "run_dir",
            "starter",
            "live_public_snapshot",
            "step_once",
        }
        if set(kwargs) != required:
            raise R1ProB1LiveEntrypointError(
                "physical grasp callback fields changed"
            )
        if kwargs["approach"] != self._approach.runner_evidence:
            self._poisoned = True
            raise R1ProB1LiveEntrypointError(
                "physical grasp received a foreign public approach"
            )
        evaluator = kwargs["evaluator"]
        starter = kwargs["starter"]
        run_dir = Path(kwargs["run_dir"])
        probe = _OfficialPrivateGraspProbe(starter)
        sample_provider = self._sample_factory(
            evaluator,
            starter,
            live_public_snapshot=kwargs["live_public_snapshot"],
            private_lift_provider=probe.lift,
        )
        if not callable(sample_provider):
            self._poisoned = True
            raise R1ProB1LiveEntrypointError(
                "calibration sample factory returned a non-callable"
            )

        def base_token() -> str:
            position, quaternion = _robot_pose(evaluator.robot)
            return R1ProFrozenBasePoseToken(
                hand=self._approach.target.hand,
                base_world_position=position,
                base_world_quaternion_xyzw=quaternion,
            ).pose_sha256

        def frozen(phase: str) -> Any:
            from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
                B1FrozenBasePoseEvidence,
            )

            position, quaternion = _robot_pose(evaluator.robot)
            return B1FrozenBasePoseEvidence(
                phase=phase,
                frozen_base_pose_token=base_token(),
                position=position,
                quaternion_xyzw=quaternion,
            )

        def public_joint_state() -> Mapping[str, Any]:
            proprio = np.asarray(
                kwargs["live_public_snapshot"]()["proprio"],
                dtype=np.float64,
            )
            if proprio.shape != (61,) or not np.isfinite(proprio).all():
                raise R1ProB1LiveEntrypointError(
                    "public R1Pro proprio is invalid"
                )
            return {
                "left_arm_qpos": tuple(proprio[3:10]),
                "right_arm_qpos": tuple(proprio[28:35]),
                "torso_qpos": tuple(proprio[53:57]),
            }

        physical_step_count = 0

        def counted_step(label: str) -> int:
            nonlocal physical_step_count
            result = kwargs["step_once"](label)
            if result == 1 and not isinstance(result, bool):
                physical_step_count += 1
            return result

        try:
            result = run_physical_grasp_lift(
                approach=self._approach,
                proposal=kwargs["proposal"],
                bootstrap=kwargs["empty_wrist"].bootstrap,
                starter=starter,
                control=kwargs["control"],
                current_base_pose_token=base_token,
                capture_frozen_base_pose=frozen,
                one_step=counted_step,
                live_public_snapshot_provider=kwargs[
                    "live_public_snapshot"
                ],
                private_contact_provider=probe.qualified_contact,
                private_assisted_provider=probe.assisted,
                private_held_provider=probe.held,
                private_lift_provider=probe.lift,
                private_audit_writer=lambda record: _append_jsonl(
                    run_dir / "private" / "physical_grasp_trace.jsonl",
                    record,
                ),
                public_joint_state=public_joint_state,
                calibration_live_sample_provider=sample_provider,
                config=self._physical_config,
            )
            self._physical_grasp_steps = physical_step_count
            self._live_starter = starter
            return result
        except Exception:
            self._poisoned = True
            raise

    def cached_public_witness_prefix(
        self,
        *,
        calibrated_hand: str,
    ) -> tuple[PublicWitnessDecision, ...]:
        """Return cached public approach actions plus the one physical grasp."""

        if (
            self._poisoned
            or self._approach is None
            or self._physical_grasp_steps is None
            or not self._captured_public_observations
            or calibrated_hand != self._approach.target.hand
        ):
            raise R1ProB1LiveEntrypointError(
                "cached public grasp witness prefix is unavailable"
            )
        registry = build_picking_up_toys_registry(
            calibrated_hands=(calibrated_hand,)
        )
        decisions: list[PublicWitnessDecision] = []
        for action, observation, status, steps in (
            self._cached_approach_executions
        ):
            decisions.append(
                PublicWitnessDecision.from_public_execution(
                    sequence=len(decisions),
                    registry=registry,
                    action=action,
                    observation=observation,
                    public_result_status=status,
                    simulator_steps=steps,
                )
            )
        decisions.append(
            PublicWitnessDecision.from_public_execution(
                sequence=len(decisions),
                registry=registry,
                action=self._approach.pending_grasp.to_dict(),
                observation=self._captured_public_observations[-1][1],
                public_result_status="grasp_likely",
                simulator_steps=self._physical_grasp_steps,
            )
        )
        return tuple(decisions)

    def live_starter(self, *, calibrated_hand: str) -> Any:
        """Return the same selected-arm Starter used by physical grasp."""

        starter = self._live_starter
        if (
            self._poisoned
            or starter is None
            or getattr(starter, "arm", None) != calibrated_hand
        ):
            raise R1ProB1LiveEntrypointError(
                "live selected-arm Starter is unavailable"
            )
        return starter


def build_r1pro_b1_live_bindings(
    *,
    calibration_config: R1ProB1SingleLoadCalibrationConfig,
    approach_config: R1ProObservationApproachConfig,
    physical_config: PhysicalGraspLiftConfig,
    calibration_live_sample_provider_factory: Callable[
        [Any, Any], Callable[..., Any]
    ],
    author_and_freeze_single_leaf_witness_callback: Callable[..., Any],
    witness_author_identity: str,
    build_post_reset_plan_callback: Callable[..., Any],
    tokenplan_call: Callable[..., Any] | None = None,
    base_contact_provider: Callable[[Any], bool] | None = None,
    event_writer: Callable[[Mapping[str, Any]], None] | None = None,
) -> R1ProB1SingleLoadLiveBindings:
    """Compose all pre-reset live callbacks without importing OmniGibson.

    ``author_and_freeze_single_leaf_witness_callback`` must continue from the held G3b
    state through public carry / toy-box grounding / physical placement, then
    freeze a typed ``PublicSingleLeafWitness`` after the private one-leaf
    canary passes.  The runner performs its sole reset after that authoring
    phase.  ``build_post_reset_plan_callback`` must replay the frozen witness;
    this function supplies no fallback plan and no default Oracle witness.
    """

    if not callable(build_post_reset_plan_callback):
        raise TypeError("build_post_reset_plan_callback must be callable")
    if not callable(author_and_freeze_single_leaf_witness_callback):
        raise TypeError(
            "author_and_freeze_single_leaf_witness_callback must be callable"
        )
    if (
        not isinstance(witness_author_identity, str)
        or not witness_author_identity.strip()
    ):
        raise ValueError("witness_author_identity must be explicit")
    call_model = tokenplan_call or _lazy_tokenplan_call
    adapter = R1ProObservationPhysicalAdapter(
        approach_config=approach_config,
        physical_config=physical_config,
        tokenplan_call=call_model,
        calibration_live_sample_provider_factory=(
            calibration_live_sample_provider_factory
        ),
        base_contact_provider=base_contact_provider,
    )
    stepper = StrictEvaluatorStepOnce(event_writer=event_writer)

    def author_and_freeze_single_leaf_witness(
        *,
        evaluator: Any,
        control: Any,
        approach: Any,
        formal_wrist: Any,
        safe_carry: Any,
        run_dir: Path,
    ) -> PublicSingleLeafWitness:
        result = author_and_freeze_single_leaf_witness_callback(
            evaluator=evaluator,
            control=control,
            approach=approach,
            formal_wrist=formal_wrist,
            safe_carry=safe_carry,
            run_dir=Path(run_dir),
            live_public_snapshot=lambda: (
                capture_sanitized_public_observation(evaluator)
            ),
            step_once=lambda label: stepper(evaluator, label),
            witness_author_identity=witness_author_identity,
            cached_public_witness_prefix=(
                adapter.cached_public_witness_prefix(
                    calibrated_hand=safe_carry.candidate.hand,
                )
            ),
            starter=adapter.live_starter(
                calibrated_hand=safe_carry.candidate.hand,
            ),
        )
        if not isinstance(result, PublicSingleLeafWitness):
            raise R1ProB1LiveEntrypointError(
                "witness author did not return PublicSingleLeafWitness"
            )
        if result.witness_author != witness_author_identity:
            raise R1ProB1LiveEntrypointError(
                "frozen witness author identity changed"
            )
        return result

    bindings = R1ProB1SingleLoadLiveBindings(
        config=calibration_config,
        arm_selectable_starter_factory=_arm_selectable_starter_factory,
        live_public_snapshot=capture_sanitized_public_observation,
        step_once=stepper,
        capture_private_reset_snapshot_callback=(
            capture_private_reset_snapshot
        ),
        observation_only_approach_callback=(
            adapter.observation_only_approach
        ),
        physical_grasp_lift_callback=adapter.physical_grasp_lift,
        author_and_freeze_single_leaf_witness_callback=(
            author_and_freeze_single_leaf_witness
        ),
        build_post_reset_plan_callback=build_post_reset_plan_callback,
    )
    setattr(
        bindings,
        "witness_author_identity",
        witness_author_identity,
    )
    return bindings


def _arm_selectable_starter_factory(
    *args: Any,
    selected_arm: str,
    **kwargs: Any,
) -> Any:
    """Import and specialize the official Starter only in a live process.

    Isaac and CUDA are intentionally unmasked so their physical device
    ordinals agree.  The official Starter does not expose CuRobo's ``device``
    argument and therefore defaults to ``cuda:0``.  Constructing its official
    motion generator explicitly is required to keep a GPU2 worker on GPU2.
    """

    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitives,
    )
    from omnigibson.action_primitives import (
        starter_semantic_action_primitives as starter_module,
    )
    from omnigibson.action_primitives.curobo import (
        CuRoboMotionGenerator,
        m as curobo_macros,
    )
    from omnigibson.utils import transform_utils as T
    import omnigibson as og
    import torch as th
    from curobo.cuda_robot_model.usd_kinematics_parser import (
        UsdKinematicsParser,
    )
    from curobo.rollout.arm_base import ArmBaseConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.wrap.reacher.evaluator import TrajEvaluatorConfig

    raw_gpu = os.environ.get("OMNIGIBSON_GPU_ID")
    try:
        physical_gpu = int(raw_gpu) if raw_gpu is not None else -1
    except ValueError as exc:
        raise R1ProB1LiveEntrypointError(
            "OMNIGIBSON_GPU_ID must be an integer"
        ) from exc
    if physical_gpu < 0:
        raise R1ProB1LiveEntrypointError(
            "OMNIGIBSON_GPU_ID must select a physical GPU"
        )

    curobo_device = f"cuda:{physical_gpu}"
    simulator_tensor_device = str(og.sim.device)
    if og.sim.floor_plane is not None:
        raise R1ProB1LiveEntrypointError(
            "CuRobo GPU2 adapter requires an InteractiveTraversableScene "
            "without a simulator floor plane"
        )

    _require_current_warp_cuda_runtime(
        physical_gpu=physical_gpu,
    )
    tensor_args = TensorDeviceType(device=th.device(curobo_device))
    curobo_source_hashes = _attest_audited_curobo_sources(
        pose_type=Pose,
    )
    # CuRobo's MotionGenConfig currently omits tensor_args when it creates
    # the default TrajEvaluatorConfig, which silently puts min_dt/max_dt on
    # cuda:0.  Supplying the equivalent official config explicitly keeps all
    # warmup tensors on this worker's selected CUDA ordinal.
    traj_evaluator_config = TrajEvaluatorConfig.from_basic(
        dof=1,
        min_dt=float(og.sim.get_sim_step_dt()),
        max_dt=0.15,
        tensor_args=tensor_args,
    )

    class ArmSelectableStarter(StarterSemanticActionPrimitives):
        def __init__(self, *inner_args: Any, **inner_kwargs: Any) -> None:
            self._selected_arm = selected_arm
            batch_size = int(inner_kwargs.get("curobo_batch_size", 3))
            if inner_kwargs.get("skip_curobo_initilization", False):
                raise R1ProB1LiveEntrypointError(
                    "production Starter may not skip CuRobo initialization"
                )
            inner_kwargs["skip_curobo_initilization"] = True
            super().__init__(*inner_args, **inner_kwargs)
            nonselected_before = {
                index: {
                    "allocated": int(th.cuda.memory_allocated(index)),
                    "reserved": int(th.cuda.memory_reserved(index)),
                }
                for index in range(th.cuda.device_count())
                if index != physical_gpu
            }
            tracked_nonselected_before = _tracked_cuda_tensor_snapshot(
                torch_module=th,
                device_indices=set(nonselected_before),
            )
            # Several CuRobo warmup internals (notably CUDA graph creation)
            # consult Torch's *current* CUDA device instead of the explicit
            # TensorDeviceType.  Keep the complete constructor and warmup
            # inside the selected physical-device context.  CUDA graphs are
            # disabled for this correctness-first backend gate; graph-free
            # planning has identical task semantics and avoids implicit
            # allocations on Torch's process-default cuda:0.
            with (
                th.cuda.device(physical_gpu),
                _selected_device_curobo_pose_from_list(
                    pose_type=Pose,
                    selected_tensor_args=tensor_args,
                ) as pose_redirect_events,
                _selected_device_curobo_arm_base_from_dict(
                    arm_base_config_type=ArmBaseConfig,
                    selected_tensor_args=tensor_args,
                ) as arm_base_redirect_events,
                _selected_device_curobo_usd_parser_init(
                    usd_parser_type=UsdKinematicsParser,
                    selected_tensor_args=tensor_args,
                ) as usd_parser_redirect_events,
            ):
                self._motion_generator = CuRoboMotionGenerator(
                    robot=self.robot,
                    device=curobo_device,
                    batch_size=batch_size,
                    use_cuda_graph=False,
                    collision_activation_distance=(
                        curobo_macros.DEFAULT_COLLISION_ACTIVATION_DISTANCE
                    ),
                    motion_cfg_kwargs={
                        "traj_evaluator_config": traj_evaluator_config,
                    },
                )
            redirect_counts = {
                site: sum(
                    event["filename"] == site[0]
                    and event["line"] == site[1]
                    for event in pose_redirect_events
                )
                for site in _AUDITED_CUROBO_POSE_REDIRECT_SITES
            }
            # ``motion_gen.py`` constructs its two persistent default Pose
            # values only on the first MotionGen construction in a resident
            # Kit process.  Later same-scene strategy resets reuse the
            # imported module and therefore legitimately observe zero events
            # at those sites.  The per-robot kinematics parser is constructed
            # for every Starter and must still redirect exactly twice per
            # embodiment.  Accept both lifecycle states while rejecting any
            # unexpected call-site/count drift.
            expected_kinematics_count = 2 * len(self._motion_generator.mg)
            motion_gen_sites = {
                ("motion_gen.py", 4284),
                ("motion_gen.py", 4286),
            }
            unexpected_motion_gen = {
                site: count
                for site, count in redirect_counts.items()
                if site in motion_gen_sites and count not in {0, 1}
            }
            observed_kinematics_count = redirect_counts.get(
                ("kinematics_parser.py", 65),
                0,
            )
            # CuRobo's kinematics parser defaults to cuda:0. On a GPU0
            # worker that device already matches the selected device, so the
            # compatibility redirect is correctly not invoked (count=0). On
            # non-zero workers the shim redirects every embodiment twice.
            # Accept only these two fully-attested lifecycle states; reject
            # partial counts and unknown call-site drift.
            valid_kinematics_counts = {0, expected_kinematics_count}
            if (
                unexpected_motion_gen
                or observed_kinematics_count not in valid_kinematics_counts
            ):
                raise R1ProB1LiveEntrypointError(
                    "CuRobo Pose.from_list compatibility redirect count "
                    f"drifted: observed={redirect_counts}, "
                    "expected motion_gen counts in {0,1} and "
                    f"kinematics_parser count {expected_kinematics_count}"
                )
            embodiment_count = len(self._motion_generator.mg)
            if len(arm_base_redirect_events) != embodiment_count:
                raise R1ProB1LiveEntrypointError(
                    "CuRobo ArmBaseConfig compatibility redirect count "
                    f"drifted: observed={len(arm_base_redirect_events)}, "
                    f"expected={embodiment_count}"
                )
            if len(usd_parser_redirect_events) != embodiment_count:
                raise R1ProB1LiveEntrypointError(
                    "CuRobo UsdKinematicsParser compatibility redirect count "
                    f"drifted: observed={len(usd_parser_redirect_events)}, "
                    f"expected={embodiment_count}"
                )
            nonselected_after = {
                index: {
                    "allocated": int(th.cuda.memory_allocated(index)),
                    "reserved": int(th.cuda.memory_reserved(index)),
                }
                for index in nonselected_before
            }
            if nonselected_after != nonselected_before:
                tracked_nonselected_after = _tracked_cuda_tensor_snapshot(
                    torch_module=th,
                    device_indices=set(nonselected_before),
                )
                new_tensor_ids = set(tracked_nonselected_after).difference(
                    tracked_nonselected_before
                )
                new_tensor_inventory = [
                    {
                        "tensor_id": tensor_id,
                        **tracked_nonselected_after[tensor_id],
                    }
                    for tensor_id in sorted(new_tensor_ids)
                ]
                owner_paths = _find_tensor_owner_paths(
                    root=self._motion_generator,
                    target_ids=new_tensor_ids,
                    torch_module=th,
                )
                nonselected_delta = {
                    index: {
                        key: nonselected_after[index][key]
                        - nonselected_before[index][key]
                        for key in ("allocated", "reserved")
                    }
                    for index in nonselected_before
                    if nonselected_after[index] != nonselected_before[index]
                }
                raise R1ProB1LiveEntrypointError(
                    "CuRobo allocated or reserved Torch memory on a "
                    f"non-selected GPU: {nonselected_delta}; "
                    f"new_tracked_tensors={new_tensor_inventory}; "
                    f"motion_generator_owner_paths={owner_paths}"
                )
            _attest_curobo_motion_generator_device(
                motion_generator=self._motion_generator,
                expected_device=curobo_device,
            )
            curobo_gripper_lock_attestation = (
                _attest_curobo_gripper_joints_locked(
                    motion_generator=self._motion_generator,
                    robot=self.robot,
                )
            )
            self._behavior_hl_curobo_device = curobo_device
            self._behavior_hl_simulator_tensor_device = (
                simulator_tensor_device
            )
            self._behavior_hl_nonselected_torch_allocated_bytes = (
                nonselected_after
            )
            self._behavior_hl_curobo_cuda_graphs_enabled = False
            self._behavior_hl_floor_plane_absent = True
            self._behavior_hl_curobo_pose_device_compatibility = True
            self._behavior_hl_curobo_pose_redirect_attestation = {
                "redirect_counts": {
                    f"{site[0]}:{site[1]}": count
                    for site, count in redirect_counts.items()
                },
                "kinematics_count": observed_kinematics_count,
                "kinematics_expected_if_device_mismatch": expected_kinematics_count,
                "kinematics_device_already_matched": (
                    observed_kinematics_count == 0
                ),
                "selected_device": curobo_device,
                "physical_gpu": physical_gpu,
            }
            self._behavior_hl_curobo_pose_redirect_events = (
                pose_redirect_events
            )
            self._behavior_hl_curobo_arm_base_redirect_events = (
                arm_base_redirect_events
            )
            self._behavior_hl_curobo_usd_parser_redirect_events = (
                usd_parser_redirect_events
            )
            self._behavior_hl_curobo_source_hashes = curobo_source_hashes
            self._behavior_hl_curobo_gripper_lock_attestation = (
                curobo_gripper_lock_attestation
            )
            # The R1Pro articulation is resident across same-scene
            # ``Evaluator.reset()`` calls.  The mixed 28D->23D adapter is an
            # intentionally single-install patch on that articulation, so a
            # later Starter must reuse (and re-attest) the existing immutable
            # report instead of trying to install a second bound method.
            if getattr(
                self.robot,
                "_behavior_hl_mixed_q_to_action_installed",
                False,
            ):
                mixed_q_to_action_attestation = getattr(
                    self.robot,
                    "_behavior_hl_mixed_q_to_action_attestation",
                    None,
                )
                if not isinstance(mixed_q_to_action_attestation, Mapping):
                    raise R1ProB1LiveEntrypointError(
                        "resident mixed q_to_action adapter lacks its "
                        "attestation"
                    )
            else:
                mixed_q_to_action_attestation = (
                    _install_r1pro_eval_mixed_q_to_action_adapter(
                        self.robot
                    )
                )
            self._behavior_hl_mixed_q_to_action_attestation = (
                mixed_q_to_action_attestation
            )
            # Keep the immutable report on the resident articulation as well
            # as on this Starter instance so later same-scene Starter objects
            # can reuse the one-time adapter safely.
            self.robot._behavior_hl_mixed_q_to_action_attestation = (
                mixed_q_to_action_attestation
            )

        @property
        def arm(self) -> str:
            return self._selected_arm

        def _move_hand(self, *move_args: Any, **move_kwargs: Any) -> Iterator[Any]:
            """Run the unmodified official hand motion with longer settling."""

            robot = self.robot
            if getattr(robot, "_behavior_hl_motion_convergence_active", False):
                raise R1ProB1LiveEntrypointError(
                    "R1Pro hand-motion convergence scope may not be nested"
                )
            freeze_active = bool(
                getattr(robot, "_behavior_hl_freeze_base_q_to_action", False)
            )
            freeze_authorized = bool(
                getattr(
                    robot,
                    "_behavior_hl_authorized_frozen_base_arm_motion",
                    False,
                )
            )
            if freeze_active and not freeze_authorized:
                raise R1ProB1LiveEntrypointError(
                    "frozen-base arm motion is active outside its authorized scope"
                )
            low_precision_requested = move_kwargs.get("low_precision", False)
            if not isinstance(low_precision_requested, bool):
                raise R1ProB1LiveEntrypointError(
                    "Starter low_precision must be an explicit bool"
                )
            low_precision_threshold = getattr(
                starter_module.m,
                "LOW_PRECISION_JOINT_POS_DIFF_THRESHOLD",
                None,
            )
            if low_precision_threshold != 0.05:
                raise R1ProB1LiveEntrypointError(
                    "official LOW_PRECISION_JOINT_POS_DIFF_THRESHOLD drifted: "
                    f"observed={low_precision_threshold!r}, expected=0.05"
                )
            samples = getattr(robot, "_behavior_hl_motion_convergence_samples", None)
            if not isinstance(samples, list):
                samples = []
                robot._behavior_hl_motion_convergence_samples = samples
            sample_start = len(samples)
            scope_record: dict[str, Any]
            with _temporary_starter_joint_motion_steps(
                starter_module.m,
                expected_steps=10,
                temporary_steps=30,
            ) as scope_record:
                scope_record["frozen_base_arm_motion_active"] = freeze_active
                scope_record["frozen_base_arm_motion_authorized"] = (
                    freeze_authorized
                )
                scope_record["low_precision_requested"] = (
                    low_precision_requested
                )
                scope_record["effective_articulation_threshold"] = (
                    low_precision_threshold
                    if low_precision_requested
                    else 0.005
                )
                scope_record["official_low_precision_threshold"] = (
                    low_precision_threshold
                )
                robot._behavior_hl_motion_convergence_active = True
                robot._behavior_hl_motion_convergence_state = {
                    "target_token": None,
                    "target_call": 0,
                }
                try:
                    yield from super()._move_hand(*move_args, **move_kwargs)
                    scope_record["status"] = "completed"
                except Exception as exc:
                    scope_record["status"] = "exception"
                    scope_record["error_type"] = type(exc).__name__
                    scope_record["error"] = str(exc)
                    raise
                finally:
                    robot._behavior_hl_motion_convergence_active = False
                    robot._behavior_hl_motion_convergence_state = None
                    scope_record["diagnostic_sample_start"] = sample_start
                    scope_record["diagnostic_sample_end"] = len(samples)
                    records = getattr(
                        self, "_behavior_hl_motion_convergence_scopes", None
                    )
                    if not isinstance(records, list):
                        records = []
                        self._behavior_hl_motion_convergence_scopes = records
                    # Append the live mapping so the context manager's
                    # subsequent ``restored=True`` update is retained even
                    # when the official generator raises.
                    records.append(scope_record)

        def _behavior_hl_world_base_candidate_joint_position(
            self,
            pose_2d: Any,
            current_joint_pos: Any,
        ) -> Any:
            """Express a world base-footprint pose in root-relative joints.

            Official ``_validate_poses`` receives sampled world ``x,y,yaw``
            but writes those values directly into ``base_control_idx``.  R1Pro
            defines its six virtual base joints relative to ``root_link``;
            ``Robot.set_position_orientation`` therefore performs this exact
            inverse-root transform before writing ``base_idx``.  Mirror that
            coordinate conversion only for motionless candidate validation.
            The robot is never moved by this helper.
            """

            return _world_base_candidate_joint_position(
                robot=self.robot,
                pose_2d=pose_2d,
                current_joint_pos=current_joint_pos,
                transform_utils=T,
                torch_module=th,
            )

        def _validate_poses(
            self,
            candidate_poses: Any,
            eef_pose: Any = None,
            plan_with_open_gripper: bool = False,
            skip_obstacle_update: bool = False,
        ) -> Any:
            """Official validation with corrected R1Pro world/root frames."""

            if plan_with_open_gripper:
                current_joint_pos = self._get_joint_position_with_fingers_at_limit(
                    "upper"
                )
            else:
                current_joint_pos = self.robot.get_joint_positions()
            candidate_joint_positions = th.stack(
                [
                    self._behavior_hl_world_base_candidate_joint_position(
                        pose,
                        current_joint_pos,
                    )
                    for pose in candidate_poses
                ]
            )
            obj_in_hand = self._get_obj_in_hand()
            attached_obj = (
                {self.robot.eef_link_names[self.arm]: obj_in_hand.root_link}
                if obj_in_hand is not None
                else None
            )
            invalid_results = self._motion_generator.check_collisions(
                candidate_joint_positions,
                self_collision_check=False,
                skip_obstacle_update=skip_obstacle_update,
                attached_obj=attached_obj,
            ).cpu()
            collision_invalid_results = invalid_results.clone()
            reachability_checked = 0
            reachability_failed = 0
            for index in range(len(candidate_poses)):
                if invalid_results[index].item() or eef_pose is None:
                    continue
                reachability_checked += 1
                if not self._target_in_reach_of_robot(
                    eef_pose,
                    initial_joint_pos=candidate_joint_positions[index],
                    skip_obstacle_update=skip_obstacle_update,
                ):
                    invalid_results[index] = True
                    reachability_failed += 1
            self._behavior_hl_world_root_validation_calls = int(
                getattr(self, "_behavior_hl_world_root_validation_calls", 0)
            ) + 1
            self._behavior_hl_last_world_root_validation = {
                "candidate_pose_frame": "world",
                "candidate_count": int(len(candidate_poses)),
                "base_collision_count": int(
                    collision_invalid_results.count_nonzero().item()
                ),
                "official_reachability_checked_count": reachability_checked,
                "official_reachability_failed_count": reachability_failed,
                "base_idx": [
                    int(value)
                    for value in self.robot.base_idx.detach().cpu().tolist()
                ],
                "base_control_idx": [
                    int(value)
                    for value in self.robot.base_control_idx.detach().cpu().tolist()
                ],
            }
            return ~invalid_results

        def _move_fingers_to_limit(self, limit_type: str) -> Iterator[Any]:
            """Drive the official scalar smooth gripper without q_to_action.

            The parent implementation converts a 28D low-level target through
            ``Robot.q_to_action`` and therefore cannot represent the official
            2026 one-scalar-per-gripper controller.  All non-gripper fields
            come from Starter's controller-native no-op; only the selected
            gripper action slot is overwritten with the official normalized
            close / open extreme.
            """

            if self.robot._grasping_direction != "lower":
                raise R1ProB1LiveEntrypointError(
                    "R1Pro eval smooth gripper must grasp toward lower limits"
                )
            if limit_type not in {"lower", "upper"}:
                raise R1ProB1LiveEntrypointError(
                    "gripper limit_type must be lower or upper"
                )
            target_joint_positions = (
                self._get_joint_position_with_fingers_at_limit(limit_type)
            )
            action_index = self.robot.controller_action_idx[
                f"gripper_{self.arm}"
            ]
            if tuple(int(value) for value in action_index.detach().cpu()) not in {
                (14,),
                (22,),
            }:
                raise R1ProB1LiveEntrypointError(
                    "selected R1Pro gripper action index drifted"
                )
            command = -1.0 if limit_type == "lower" else 1.0
            for _ in range(
                starter_module.m.MAX_STEPS_FOR_GRASP_OR_RELEASE
            ):
                action = self._empty_action()
                action[action_index] = command
                yield self._postprocess_action(action)
                current_joint_positions = self.robot.get_joint_positions()
                if th.allclose(
                    current_joint_positions,
                    target_joint_positions,
                    rtol=0.0,
                    atol=starter_module.m.JOINT_POS_DIFF_THRESHOLD,
                ):
                    break
                if (
                    limit_type == "lower"
                    and self._get_obj_in_hand() is not None
                ):
                    break

    return ArmSelectableStarter(*args, **kwargs)


def _attest_curobo_motion_generator_device(
    *,
    motion_generator: Any,
    expected_device: str,
) -> None:
    """Verify every constructed embodiment and trajectory dt tensor."""

    tensor_args = getattr(motion_generator, "_tensor_args", None)
    if str(getattr(tensor_args, "device", None)) != expected_device:
        raise R1ProB1LiveEntrypointError(
            "CuRobo root tensor device differs from selected GPU"
        )
    generators = getattr(motion_generator, "mg", None)
    if not isinstance(generators, Mapping) or not generators:
        raise R1ProB1LiveEntrypointError(
            "CuRobo motion-generator embodiments are unavailable"
        )
    first_generator = next(iter(generators.values()))
    plan_grasp = getattr(type(first_generator), "plan_grasp", None)
    defaults = getattr(plan_grasp, "__defaults__", None)
    if not isinstance(defaults, tuple):
        raise R1ProB1LiveEntrypointError(
            "CuRobo MotionGen.plan_grasp defaults are unavailable"
        )
    default_poses = [
        value
        for value in defaults
        if hasattr(value, "position") and hasattr(value, "quaternion")
    ]
    if len(default_poses) != 2:
        raise R1ProB1LiveEntrypointError(
            "CuRobo MotionGen.plan_grasp default Pose count drifted"
        )
    for pose_index, pose in enumerate(default_poses):
        for field in ("position", "quaternion", "rotation"):
            tensor = getattr(pose, field, None)
            if tensor is not None and str(
                getattr(tensor, "device", None)
            ) != expected_device:
                raise R1ProB1LiveEntrypointError(
                    "CuRobo MotionGen.plan_grasp default "
                    f"Pose[{pose_index}].{field} device mismatch"
                )
    for embodiment, generator in generators.items():
        if str(getattr(generator.tensor_args, "device", None)) != (
            expected_device
        ):
            raise R1ProB1LiveEntrypointError(
                f"CuRobo {embodiment} tensor device mismatch"
            )
        graph_planner = getattr(generator, "graph_planner", None)
        for rollout_name in ("rollout_fn", "safety_rollout_fn"):
            rollout = getattr(graph_planner, rollout_name, None)
            if str(
                getattr(
                    getattr(rollout, "tensor_args", None),
                    "device",
                    None,
                )
            ) != expected_device:
                raise R1ProB1LiveEntrypointError(
                    f"CuRobo {embodiment} graph {rollout_name} "
                    "device mismatch"
                )
        for solver_name in (
            "trajopt_solver",
            "finetune_trajopt_solver",
        ):
            solver = getattr(generator, solver_name, None)
            evaluator = getattr(solver, "traj_evaluator", None)
            for field in ("min_dt", "max_dt"):
                tensor = getattr(evaluator, field, None)
                if str(getattr(tensor, "device", None)) != expected_device:
                    raise R1ProB1LiveEntrypointError(
                        f"CuRobo {embodiment} {solver_name}.{field} "
                        "device mismatch"
                    )


def _require_current_warp_cuda_runtime(
    *,
    physical_gpu: int,
    context_module: Any | None = None,
) -> None:
    """Fail closed unless Kit's already-attested Warp runtime is ready."""

    if (
        isinstance(physical_gpu, bool)
        or not isinstance(physical_gpu, int)
        or physical_gpu < 0
    ):
        raise R1ProB1LiveEntrypointError(
            "physical_gpu must be a non-negative integer"
        )

    if context_module is None:
        import warp.context as context
    else:
        context = context_module

    if context is None:
        raise R1ProB1LiveEntrypointError(
            "current Warp context is unavailable"
        )
    runtime = getattr(context, "runtime", None)
    cuda_devices = getattr(runtime, "cuda_devices", None)
    if runtime is None or cuda_devices is None:
        raise R1ProB1LiveEntrypointError(
            "current Warp CUDA runtime is unavailable"
        )
    if physical_gpu >= len(cuda_devices):
        raise R1ProB1LiveEntrypointError(
            "current Warp runtime does not expose the selected physical GPU"
        )


def _lazy_tokenplan_call(**kwargs: Any) -> Any:
    from behavior_hl.embodied.evaluator_bridge.run_g4a_grasp_suite import (
        _run_tokenplan_approach_call,
    )

    return _run_tokenplan_approach_call(**kwargs)


def _official_base_contact(evaluator: Any) -> bool:
    """Use the existing private contact audit, imported only after Kit."""

    from omnigibson.utils.usd_utils import RigidContactAPI

    from behavior_hl.embodied.evaluator_bridge.run_g2_g3a_suite import (
        _base_contact,
    )

    return bool(_base_contact(evaluator, RigidContactAPI))


class _OfficialPrivateGraspProbe:
    """Read official assisted-grasp state only for private calibration audit."""

    def __init__(self, starter: Any) -> None:
        self._starter = starter
        self._robot = getattr(starter, "robot", None)
        self._start_height: dict[str, float] = {}
        if self._robot is None or not callable(
            getattr(starter, "_get_obj_in_hand", None)
        ):
            raise R1ProB1LiveEntrypointError(
                "official Starter held-object API is unavailable"
            )

    def held(self, hand: str) -> Any:
        self._require_hand(hand)
        value = self._starter._get_obj_in_hand()
        if value is not None and hand not in self._start_height:
            self._start_height[hand] = float(
                _finite_vector(
                    value.get_position_orientation()[0],
                    size=3,
                    name="held object position",
                )[2]
            )
        return value

    def begin_target(self, hand: str, target: Any) -> None:
        """Freeze the evaluator-only pre-close height for lift evidence."""

        self._require_hand(hand)
        self._start_height[hand] = float(
            _finite_vector(
                target.get_position_orientation()[0],
                size=3,
                name="target baseline position",
            )[2]
        )

    def assisted(self, hand: str) -> bool:
        self._require_hand(hand)
        state = getattr(self._robot, "_ag_obj_in_hand", None)
        if not isinstance(state, Mapping) or hand not in state:
            raise R1ProB1LiveEntrypointError(
                "official assisted-grasp state is unavailable"
            )
        return state[hand] is not None

    def qualified_contact(self, hand: str) -> bool:
        # OmniGibson creates _ag_obj_in_hand only after its own finger-contact
        # and raycast checks.  This is private audit evidence, never a public
        # model input.
        return self.assisted(hand) and self.held(hand) is not None

    def contact_witness(
        self,
        hand: str,
        *,
        target: Any | None = None,
    ) -> dict[str, Any]:
        """Return read-only official finger/contact/raycast evidence.

        The report is evaluator-private.  It calls the same robot helpers used
        by ``_calculate_in_hand_object`` and never mutates grasp state.
        Individual probes fail closed into an error field so diagnostics do
        not abort the physical controller at the instant of contact.
        """

        self._require_hand(hand)
        report: dict[str, Any] = {
            "finger_contact_body_paths": [],
            "finger_contact_links_by_body": {},
            "raycast_body_paths": [],
            "calculated_in_hand_object": None,
            "calculated_in_hand_is_target": False,
            "assisted_object": None,
            "assisted_object_is_target": False,
            "grasping_mode": getattr(self._robot, "grasping_mode", None),
            "target_name": getattr(target, "name", None),
            "target_body_paths": [],
            "target_in_finger_contacts": False,
            "target_in_raycast": False,
        }
        target_paths: set[str] = set()
        if target is not None:
            links = getattr(target, "links", None)
            if isinstance(links, Mapping):
                target_paths.update(
                    str(getattr(link, "prim_path"))
                    for link in links.values()
                    if getattr(link, "prim_path", None) is not None
                )
            root_link = getattr(target, "root_link", None)
            if getattr(root_link, "prim_path", None) is not None:
                target_paths.add(str(root_link.prim_path))
        report["target_body_paths"] = sorted(target_paths)
        try:
            contacts, robot_links = self._robot._find_gripper_contacts(
                arm=hand
            )
            report["finger_contact_body_paths"] = sorted(
                str(value) for value in contacts
            )
            report["finger_contact_links_by_body"] = {
                str(body): sorted(str(link) for link in links)
                for body, links in robot_links.items()
            }
            report["target_in_finger_contacts"] = bool(
                target_paths.intersection(str(value) for value in contacts)
            )
        except Exception as exc:
            report["finger_contact_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        try:
            raycast_paths = {
                str(value)
                for value in self._robot._find_gripper_raycast_collisions(
                    arm=hand
                )
            }
            report["raycast_body_paths"] = sorted(raycast_paths)
            report["target_in_raycast"] = bool(
                target_paths.intersection(raycast_paths)
            )
        except Exception as exc:
            report["raycast_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        try:
            calculated = self._robot._calculate_in_hand_object(arm=hand)
            if calculated is not None:
                obj, link_name = calculated
                report["calculated_in_hand_object"] = {
                    "object_name": getattr(obj, "name", None),
                    "link_name": str(link_name),
                }
                report["calculated_in_hand_is_target"] = obj is target
        except Exception as exc:
            report["calculated_in_hand_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        state = getattr(self._robot, "_ag_obj_in_hand", None)
        if isinstance(state, Mapping) and hand in state and state[hand] is not None:
            report["assisted_object"] = getattr(state[hand], "name", None)
            report["assisted_object_is_target"] = state[hand] is target
        return report

    def lift(self, hand: str) -> float:
        value = self.held(hand)
        if value is None or hand not in self._start_height:
            return 0.0
        height = _finite_vector(
            value.get_position_orientation()[0],
            size=3,
            name="held object position",
        )[2]
        return max(0.0, float(height) - self._start_height[hand])

    def _require_hand(self, hand: str) -> None:
        if hand != getattr(self._starter, "arm", None):
            raise R1ProB1LiveEntrypointError(
                "private grasp probe hand changed"
            )


def _write_public_snapshot(
    path: Path,
    snapshot: Mapping[str, Any],
) -> None:
    head_eval = (
        snapshot.get("schema_version")
        == THREE_RGBD_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION
    )
    validate_public_observation(snapshot, head_eval=head_eval)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, Any] = {
        "proprio": snapshot["proprio"],
        "cam_rel_poses": snapshot["cam_rel_poses"],
    }
    for role in CAMERA_RESOLUTIONS:
        for modality, value in snapshot["cameras"][role].items():
            fields[f"{role}_{modality}"] = value
    temporary = path.with_name(f".{path.name}.tmp.npz")
    if temporary.exists():
        raise R1ProB1LiveEntrypointError(
            "stale public snapshot temporary file exists"
        )
    try:
        np.savez_compressed(temporary, **fields)
        temporary.replace(path)
    except Exception:
        if temporary.is_file():
            temporary.unlink()
        raise


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(dict(record), sort_keys=True, default=float) + "\n"
        )


def _robot_pose(robot: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    getter = getattr(robot, "get_position_orientation", None)
    if not callable(getter):
        raise R1ProB1LiveEntrypointError(
            "robot pose API is unavailable"
        )
    try:
        position, quaternion = getter(frame="world", clone=True)
    except TypeError:
        position, quaternion = getter()
    return (
        _finite_vector(position, size=3, name="robot position"),
        _finite_vector(quaternion, size=4, name="robot quaternion"),
    )


def _numpy_copy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    result = np.array(value, copy=True)
    if not np.issubdtype(result.dtype, np.number):
        raise ValueError("public observation contains a non-numeric array")
    return result


def _public_robot_observation(
    snapshot: Mapping[str, Any],
    *,
    simulator_step: int,
) -> PublicRobotObservation:
    head_eval = (
        snapshot.get("schema_version")
        == THREE_RGBD_HEAD_EVAL_OBSERVATION_SCHEMA_VERSION
    )
    validate_public_observation(snapshot, head_eval=head_eval)
    return PublicRobotObservation(
        schema_version=str(snapshot["schema_version"]),
        simulator_step=simulator_step,
        values={
            "task_id": int(snapshot["task_id"]),
            "proprio": _numpy_copy(snapshot["proprio"]),
            "cam_rel_poses": _numpy_copy(snapshot["cam_rel_poses"]),
            "cameras": {
                role: {
                    modality: _numpy_copy(value)
                    for modality, value in snapshot["cameras"][role].items()
                }
                for role in CAMERA_RESOLUTIONS
            },
        },
    )


def _finite_vector(
    value: Any,
    *,
    name: str,
    size: int | None = None,
) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 1
        or (size is not None and array.shape != (size,))
        or not array.size
        or not np.isfinite(array).all()
    ):
        suffix = "" if size is None else f" {size}D"
        raise R1ProB1LiveEntrypointError(
            f"{name} must be a finite{suffix} vector"
        )
    return tuple(float(item) for item in array)
