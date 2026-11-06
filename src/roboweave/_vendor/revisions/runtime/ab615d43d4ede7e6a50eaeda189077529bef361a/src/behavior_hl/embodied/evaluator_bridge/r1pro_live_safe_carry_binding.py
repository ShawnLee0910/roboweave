"""Concrete 2026 R1Pro callbacks for live safe-carry proposal.

The caller imports and constructs OmniGibson before instantiating these
callbacks.  This module itself has no simulator import: it binds only the
documented live Robot and Starter object surface.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_live_safe_carry_candidate import (
    LIVE_POSE_FAMILY_SOURCE,
    LIVE_ROBOT_STATE_SOURCE,
    LiveEEFPoseCandidate,
    LiveRobotCarryState,
)


R1PRO_LIVE_SAFE_CARRY_BINDING_VERSION = (
    "behavior2026.r1pro-live-safe-carry-binding/v2"
)


class R1ProLiveSafeCarryBindingError(RuntimeError):
    """Raised when the live Starter/Robot surface violates assumptions."""


@dataclass(frozen=True, slots=True)
class R1ProDynamicCandidateFamilyConfig:
    """Dimensionless live-geometry search fractions and local boundaries."""

    minimum_reference_horizontal_radius_m: float = 0.15
    maximum_body_reach_m: float = 1.20
    inner_radius_fraction: float = 0.68
    center_radius_fraction: float = 0.78
    outer_radius_fraction: float = 0.88
    low_height_fraction: float = 0.88
    center_height_fraction: float = 0.94
    workspace_yaw_fraction: float = 0.30

    def __post_init__(self) -> None:
        for name in (
            "minimum_reference_horizontal_radius_m",
            "maximum_body_reach_m",
        ):
            if _positive(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
            self.maximum_body_reach_m
            <= self.minimum_reference_horizontal_radius_m
        ):
            raise ValueError("body reach interval is invalid")
        radius_fractions = (
            self.inner_radius_fraction,
            self.center_radius_fraction,
            self.outer_radius_fraction,
        )
        if not all(
            0.0 < _finite(value, "radius_fraction") <= 1.0
            for value in radius_fractions
        ):
            raise ValueError("radius fractions must be in (0, 1]")
        if not (
            self.inner_radius_fraction
            < self.center_radius_fraction
            < self.outer_radius_fraction
        ):
            raise ValueError("radius fractions must be strictly ordered")
        if not (
            0.0
            < _finite(
                self.low_height_fraction,
                "low_height_fraction",
            )
            < self.center_height_fraction
            <= 1.0
        ):
            raise ValueError("height fractions are invalid")
        yaw_fraction = _finite(
            self.workspace_yaw_fraction,
            "workspace_yaw_fraction",
        )
        if not 0.0 < yaw_fraction <= 0.5:
            raise ValueError("workspace_yaw_fraction must be in (0, 0.5]")


@dataclass(frozen=True, slots=True)
class R1ProLiveSafeCarryProposalBindings:
    """Callbacks passed directly to propose_live_safe_carry_candidate."""

    live_state_provider: Callable[[Any, str], LiveRobotCarryState]
    candidate_pose_family: Callable[
        [LiveRobotCarryState], Sequence[LiveEEFPoseCandidate]
    ]
    official_pose_adapter: Callable[
        [Any, LiveEEFPoseCandidate], tuple[Any, Any]
    ]
    arm_target_from_official_ik: Callable[
        [Any, Any, str], Sequence[float]
    ]
    version: str = R1PRO_LIVE_SAFE_CARRY_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.version != R1PRO_LIVE_SAFE_CARRY_BINDING_VERSION:
            raise ValueError("binding version mismatch")
        for name in (
            "live_state_provider",
            "candidate_pose_family",
            "official_pose_adapter",
            "arm_target_from_official_ik",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")

    def proposal_kwargs(self) -> Mapping[str, Callable[..., Any]]:
        return MappingProxyType(
            {
                "live_state_provider": self.live_state_provider,
                "candidate_pose_family": self.candidate_pose_family,
                "official_pose_adapter": self.official_pose_adapter,
                "arm_target_from_official_ik": (
                    self.arm_target_from_official_ik
                ),
            }
        )


def build_r1pro_live_safe_carry_proposal_bindings(
    *,
    starter: Any,
    hand: str,
    config: R1ProDynamicCandidateFamilyConfig | None = None,
) -> R1ProLiveSafeCarryProposalBindings:
    """Preflight current 2026 Starter/Robot capabilities without mutation."""

    _require_starter(starter, hand)
    family = R1ProDynamicCandidateFamily(
        starter=starter,
        hand=hand,
        config=config or R1ProDynamicCandidateFamilyConfig(),
    )
    return R1ProLiveSafeCarryProposalBindings(
        live_state_provider=read_r1pro_live_robot_carry_state,
        candidate_pose_family=family,
        official_pose_adapter=adapt_r1pro_official_pose,
        arm_target_from_official_ik=(
            extract_r1pro_arm_target_from_official_ik
        ),
    )


def read_r1pro_live_robot_carry_state(
    starter: Any,
    hand: str,
) -> LiveRobotCarryState:
    """Read current body, EEF, full joints, selected arm, and live limits."""

    robot = _require_starter(starter, hand)
    try:
        body_position, body_quaternion = robot.get_position_orientation(
            frame="world",
            clone=True,
        )
        eef_position, eef_quaternion = robot.get_eef_pose(arm=hand)
        full_joint_pos = robot.get_joint_positions()
        arm_indices_native = robot.arm_control_idx[hand]
        arm_indices = _indices(arm_indices_native, "arm_control_idx")
        arm_qpos = _take(full_joint_pos, arm_indices, "arm_qpos")
        arm_lower = _take(
            robot.joint_lower_limits,
            arm_indices,
            "arm_lower_limits",
        )
        arm_upper = _take(
            robot.joint_upper_limits,
            arm_indices,
            "arm_upper_limits",
        )
    except Exception as exc:
        raise R1ProLiveSafeCarryBindingError(
            f"live R1Pro state read failed: {type(exc).__name__}: {exc}"
        ) from exc
    if len(arm_indices) != 7:
        raise R1ProLiveSafeCarryBindingError(
            "selected R1Pro arm_control_idx must contain exactly 7 joints"
        )
    body = _vector(body_position, 3, "body_world_position")
    body_quat = _vector(
        body_quaternion,
        4,
        "body_world_quaternion_xyzw",
    )
    eef = _vector(eef_position, 3, "eef_world_position")
    eef_quat = _vector(
        eef_quaternion,
        4,
        "eef_world_quaternion_xyzw",
    )
    arm = _vector(arm_qpos, 7, "arm_qpos")
    lower = _vector(arm_lower, 7, "arm_lower_limits")
    upper = _vector(arm_upper, 7, "arm_upper_limits")
    full = _native_array(full_joint_pos, "full_joint_pos")
    token = _live_state_token(
        hand=hand,
        body=body,
        body_quaternion=body_quat,
        eef=eef,
        eef_quaternion=eef_quat,
        full_joints=full,
    )
    return LiveRobotCarryState(
        hand=hand,
        state_token=token,
        body_world_position=body,
        body_world_quaternion_xyzw=body_quat,
        eef_world_position=eef,
        eef_world_quaternion_xyzw=eef_quat,
        arm_qpos=arm,
        arm_lower_limits=lower,
        arm_upper_limits=upper,
        official_initial_joint_pos=_clone_native(full_joint_pos),
        source=LIVE_ROBOT_STATE_SOURCE,
    )


def adapt_r1pro_official_pose(
    starter: Any,
    candidate: LiveEEFPoseCandidate,
) -> tuple[Any, Any]:
    """Create Starter-native tensors matching its current live EEF tensors."""

    if not isinstance(candidate, LiveEEFPoseCandidate):
        raise TypeError("candidate must be LiveEEFPoseCandidate")
    robot = _require_starter(starter, getattr(starter, "arm", None))
    try:
        position_reference, quaternion_reference = robot.get_eef_pose(
            arm=starter.arm
        )
        position = _native_like(
            position_reference,
            candidate.eef_world_position,
        )
        quaternion = _native_like(
            quaternion_reference,
            candidate.eef_world_quaternion_xyzw,
        )
    except Exception as exc:
        raise R1ProLiveSafeCarryBindingError(
            f"official pose adaptation failed: {type(exc).__name__}: {exc}"
        ) from exc
    return position, quaternion


def extract_r1pro_arm_target_from_official_ik(
    starter: Any,
    official_solution: Any,
    hand: str,
) -> tuple[float, ...]:
    """Map Starter's manipulation-order IK result to selected arm order."""

    robot = _require_starter(starter, hand)
    manipulation_method = getattr(
        starter,
        "_manipulation_control_idx",
        None,
    )
    if not callable(manipulation_method):
        raise R1ProLiveSafeCarryBindingError(
            "Starter lacks _manipulation_control_idx"
        )
    try:
        manipulation_indices = _indices(
            manipulation_method(),
            "manipulation_control_idx",
        )
        arm_indices = _indices(
            robot.arm_control_idx[hand],
            "arm_control_idx",
        )
        solution = _native_array(
            official_solution,
            "official IK solution",
        )
    except Exception as exc:
        raise R1ProLiveSafeCarryBindingError(
            f"official IK layout read failed: {type(exc).__name__}: {exc}"
        ) from exc
    if solution.ndim != 1 or solution.shape[0] != len(
        manipulation_indices
    ):
        raise R1ProLiveSafeCarryBindingError(
            "official IK solution shape differs from manipulation index order"
        )
    if len(arm_indices) != 7 or len(set(arm_indices)) != 7:
        raise R1ProLiveSafeCarryBindingError(
            "selected R1Pro arm must map to exactly 7 unique joints"
        )
    if len(set(manipulation_indices)) != len(manipulation_indices):
        raise R1ProLiveSafeCarryBindingError(
            "manipulation control indices must be unique"
        )
    position_by_joint = {
        joint_index: position
        for position, joint_index in enumerate(manipulation_indices)
    }
    if any(index not in position_by_joint for index in arm_indices):
        raise R1ProLiveSafeCarryBindingError(
            "selected arm indices are absent from official IK layout"
        )
    target = solution[
        [position_by_joint[index] for index in arm_indices]
    ]
    return _vector(target, 7, "selected arm IK target")


class R1ProDynamicCandidateFamily:
    """Generate a local multi-pose family from current body/EEF/workspace."""

    def __init__(
        self,
        *,
        starter: Any,
        hand: str,
        config: R1ProDynamicCandidateFamilyConfig,
    ) -> None:
        _require_starter(starter, hand)
        if not isinstance(config, R1ProDynamicCandidateFamilyConfig):
            raise TypeError(
                "config must be R1ProDynamicCandidateFamilyConfig"
            )
        self._starter = starter
        self._hand = hand
        self._config = config

    def __call__(
        self,
        state: LiveRobotCarryState,
    ) -> tuple[LiveEEFPoseCandidate, ...]:
        if not isinstance(state, LiveRobotCarryState):
            raise TypeError("state must be LiveRobotCarryState")
        if state.hand != self._hand:
            raise R1ProLiveSafeCarryBindingError(
                "candidate family hand differs from live state"
            )
        robot = _require_starter(self._starter, self._hand)
        workspace = getattr(robot, "arm_workspace_range", None)
        if not isinstance(workspace, Mapping) or self._hand not in workspace:
            raise R1ProLiveSafeCarryBindingError(
                "R1Pro arm workspace range is unavailable"
            )
        yaw_bounds = _native_array(
            workspace[self._hand],
            "arm workspace range",
        )
        if (
            yaw_bounds.shape != (2,)
            or not np.isfinite(yaw_bounds).all()
            or yaw_bounds[1] <= yaw_bounds[0]
        ):
            raise R1ProLiveSafeCarryBindingError(
                "arm workspace range must be finite [lower, upper]"
            )

        rotation = _rotation_matrix(
            state.body_world_quaternion_xyzw
        )
        body = np.asarray(state.body_world_position, dtype=np.float64)
        relative_world = (
            np.asarray(state.eef_world_position, dtype=np.float64) - body
        )
        relative_body = rotation.T @ relative_world
        horizontal_radius = float(np.linalg.norm(relative_body[:2]))
        if np.linalg.norm(relative_body) > self._config.maximum_body_reach_m:
            raise R1ProLiveSafeCarryBindingError(
                "live EEF is outside admitted local arm workspace"
            )
        # The official arm_workspace_range describes target workspace angles;
        # it does not require the reset / tucked EEF itself to be inside that
        # range.  R1Pro's zero-joint reset leaves each EEF beside the body
        # (about +/-65 degrees) while the official target range is
        # [-45, +45] degrees.  Use live yaw only when it is already admitted;
        # otherwise seed the collision-aware IK family at the workspace
        # midpoint.  No candidate is accepted without the downstream local
        # displacement, body-reach, official IK, and joint-margin gates.
        live_yaw = math.atan2(relative_body[1], relative_body[0])
        yaw = (
            live_yaw
            if yaw_bounds[0] <= live_yaw <= yaw_bounds[1]
            else float((yaw_bounds[0] + yaw_bounds[1]) / 2.0)
        )
        reference_horizontal_radius = max(
            horizontal_radius,
            self._config.minimum_reference_horizontal_radius_m,
        )
        joint_margin = _minimum_joint_margin(state)
        comfort = min(max(joint_margin / 0.5, 0.0), 1.0)
        center_radius_scale = (
            self._config.center_radius_fraction
            + (
                self._config.outer_radius_fraction
                - self._config.center_radius_fraction
            )
            * comfort
        )
        radii = (
            reference_horizontal_radius
            * self._config.inner_radius_fraction,
            reference_horizontal_radius * center_radius_scale,
            reference_horizontal_radius
            * self._config.outer_radius_fraction,
        )
        center_height_scale = (
            self._config.low_height_fraction
            + (
                self._config.center_height_fraction
                - self._config.low_height_fraction
            )
            * comfort
        )
        heights = (
            relative_body[2] * self._config.low_height_fraction,
            relative_body[2] * center_height_scale,
            relative_body[2] * self._config.center_height_fraction,
        )
        left_room = float(yaw_bounds[1] - yaw)
        right_room = float(yaw - yaw_bounds[0])
        left_yaw = yaw + left_room * self._config.workspace_yaw_fraction
        right_yaw = yaw - right_room * self._config.workspace_yaw_fraction
        angles = (yaw, left_yaw, right_yaw)

        specifications = (
            ("center", radii[1], angles[0], heights[1]),
            ("inner", radii[0], angles[0], heights[2]),
            ("outer", radii[2], angles[0], heights[0]),
            ("workspace_left", radii[1], angles[1], heights[1]),
            ("workspace_right", radii[1], angles[2], heights[1]),
        )
        candidates: list[LiveEEFPoseCandidate] = []
        seen: set[tuple[float, ...]] = set()
        for label, radius, angle, height in specifications:
            body_point = np.asarray(
                (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    height,
                ),
                dtype=np.float64,
            )
            world_point = body + rotation @ body_point
            signature = tuple(round(float(value), 10) for value in world_point)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(
                LiveEEFPoseCandidate(
                    candidate_id=(
                        f"{state.state_token[:12]}:{label}"
                    ),
                    state_token=state.state_token,
                    eef_world_position=tuple(
                        float(value) for value in world_point
                    ),
                    eef_world_quaternion_xyzw=(
                        state.eef_world_quaternion_xyzw
                    ),
                    source=LIVE_POSE_FAMILY_SOURCE,
                )
            )
        if len(candidates) < 3:
            raise R1ProLiveSafeCarryBindingError(
                "dynamic live candidate family produced fewer than 3 poses"
            )
        return tuple(candidates)


def _require_starter(starter: Any, hand: Any) -> Any:
    if hand not in {"left", "right"}:
        raise R1ProLiveSafeCarryBindingError(
            "hand must be left or right"
        )
    if getattr(starter, "arm", None) != hand:
        raise R1ProLiveSafeCarryBindingError(
            "Starter arm differs from requested hand"
        )
    robot = getattr(starter, "robot", None)
    if robot is None:
        raise R1ProLiveSafeCarryBindingError("Starter robot is unavailable")
    for method in (
        "_ik_solver_cartesian_to_joint_space",
        "_manipulation_control_idx",
    ):
        if not callable(getattr(starter, method, None)):
            raise R1ProLiveSafeCarryBindingError(
                f"Starter lacks {method}"
            )
    for method in (
        "get_position_orientation",
        "get_eef_pose",
        "get_joint_positions",
    ):
        if not callable(getattr(robot, method, None)):
            raise R1ProLiveSafeCarryBindingError(
                f"R1Pro robot lacks {method}"
            )
    for attribute in (
        "arm_control_idx",
        "joint_lower_limits",
        "joint_upper_limits",
    ):
        if not hasattr(robot, attribute):
            raise R1ProLiveSafeCarryBindingError(
                f"R1Pro robot lacks {attribute}"
            )
    return robot


def _minimum_joint_margin(state: LiveRobotCarryState) -> float:
    qpos = np.asarray(state.arm_qpos)
    lower = np.asarray(state.arm_lower_limits)
    upper = np.asarray(state.arm_upper_limits)
    span = upper - lower
    return float(
        np.min(np.minimum((qpos - lower) / span, (upper - qpos) / span))
    )


def _rotation_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = _vector(
        quaternion_xyzw,
        4,
        "body_world_quaternion_xyzw",
    )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-8:
        raise R1ProLiveSafeCarryBindingError(
            "body quaternion has zero norm"
        )
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def _live_state_token(
    *,
    hand: str,
    body: Sequence[float],
    body_quaternion: Sequence[float],
    eef: Sequence[float],
    eef_quaternion: Sequence[float],
    full_joints: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(R1PRO_LIVE_SAFE_CARRY_BINDING_VERSION.encode())
    digest.update(hand.encode())
    for value in (
        body,
        body_quaternion,
        eef,
        eef_quaternion,
        full_joints,
    ):
        digest.update(
            np.ascontiguousarray(value, dtype=np.float64).tobytes()
        )
    return f"live-{digest.hexdigest()}"


def _indices(value: Any, name: str) -> tuple[int, ...]:
    array = _native_array(value, name)
    if array.ndim != 1 or not np.equal(array, np.floor(array)).all():
        raise R1ProLiveSafeCarryBindingError(
            f"{name} must be a 1D integer index vector"
        )
    output = tuple(int(item) for item in array)
    if any(item < 0 for item in output):
        raise R1ProLiveSafeCarryBindingError(
            f"{name} cannot contain negative indices"
        )
    return output


def _take(value: Any, indices: tuple[int, ...], name: str) -> np.ndarray:
    array = _native_array(value, name)
    if array.ndim != 1 or any(index >= len(array) for index in indices):
        raise R1ProLiveSafeCarryBindingError(
            f"{name} cannot be indexed by selected arm indices"
        )
    return array[list(indices)]


def _native_array(value: Any, name: str) -> np.ndarray:
    item = value
    for method in ("detach", "cpu"):
        callback = getattr(item, method, None)
        if callable(callback):
            item = callback()
    numpy_method = getattr(item, "numpy", None)
    if callable(numpy_method):
        item = numpy_method()
    array = np.asarray(item)
    if not np.issubdtype(array.dtype, np.number):
        raise R1ProLiveSafeCarryBindingError(f"{name} must be numeric")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise R1ProLiveSafeCarryBindingError(f"{name} must be finite")
    return array


def _clone_native(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    return np.array(value, copy=True)


def _native_like(reference: Any, values: Sequence[float]) -> Any:
    new_tensor = getattr(reference, "new_tensor", None)
    if callable(new_tensor):
        return new_tensor(values)
    reference_array = np.asarray(reference)
    return np.asarray(values, dtype=reference_array.dtype)


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    array = _native_array(value, name)
    if array.shape != (size,):
        raise R1ProLiveSafeCarryBindingError(
            f"{name} must be a finite {size}-vector"
        )
    return tuple(float(item) for item in array)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result
