"""Live, collision-aware SafeCarryCandidate proposal for R1Pro.

This module is an injection-only adapter around the already-live Starter.  A
caller supplies one current robot/body/EEF snapshot, a multi-pose candidate
family derived from that snapshot, and an extractor for the arm portion of an
official IK solution.  The adapter asks Starter's official collision-aware IK
method about every admitted pose and ranks successful solutions by normalized
joint-limit margin.

No simulator package is imported and no physics step or robot command occurs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
    SafeCarryCandidate,
)


LIVE_SAFE_CARRY_PROPOSAL_VERSION = (
    "behavior2026.r1pro-live-safe-carry-proposal/v1"
)
LIVE_SAFE_CARRY_CANDIDATE_SOURCE = (
    "live_collision_aware_joint_margin_search"
)
LIVE_ROBOT_STATE_SOURCE = "live_robot_body_eef_snapshot"
LIVE_POSE_FAMILY_SOURCE = "live_body_eef_candidate_family"
OFFICIAL_STARTER_IK_METHOD = "_ik_solver_cartesian_to_joint_space"


class R1ProLiveSafeCarryProposalError(RuntimeError):
    """Raised when no fully evidenced live candidate can be selected."""


@dataclass(frozen=True, slots=True)
class LiveRobotCarryState:
    """One immutable current-state seed for a proposal search."""

    hand: str
    state_token: str
    body_world_position: tuple[float, ...]
    body_world_quaternion_xyzw: tuple[float, ...]
    eef_world_position: tuple[float, ...]
    eef_world_quaternion_xyzw: tuple[float, ...]
    arm_qpos: tuple[float, ...]
    arm_lower_limits: tuple[float, ...]
    arm_upper_limits: tuple[float, ...]
    official_initial_joint_pos: Any = field(repr=False, compare=False)
    source: str = LIVE_ROBOT_STATE_SOURCE

    def __post_init__(self) -> None:
        if self.hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        _token(self.state_token, "state_token")
        if self.source != LIVE_ROBOT_STATE_SOURCE:
            raise ValueError("live state source contract mismatch")
        for name, size in (
            ("body_world_position", 3),
            ("body_world_quaternion_xyzw", 4),
            ("eef_world_position", 3),
            ("eef_world_quaternion_xyzw", 4),
            ("arm_qpos", 7),
            ("arm_lower_limits", 7),
            ("arm_upper_limits", 7),
        ):
            object.__setattr__(
                self,
                name,
                _vector(getattr(self, name), size, name),
            )
        _quaternion(
            self.body_world_quaternion_xyzw,
            "body_world_quaternion_xyzw",
        )
        _quaternion(
            self.eef_world_quaternion_xyzw,
            "eef_world_quaternion_xyzw",
        )
        lower = np.asarray(self.arm_lower_limits)
        upper = np.asarray(self.arm_upper_limits)
        qpos = np.asarray(self.arm_qpos)
        if np.any(upper <= lower):
            raise ValueError("arm joint-limit intervals must be positive")
        if np.any(qpos < lower) or np.any(qpos > upper):
            violations = [
                {"joint_index": int(i), "qpos": float(qpos[i]),
                 "lower": float(lower[i]), "upper": float(upper[i]),
                 "excess_rad": float(max(lower[i] - qpos[i], qpos[i] - upper[i]))}
                for i in np.flatnonzero((qpos < lower) | (qpos > upper))
            ]
            raise ValueError(
                f"live arm qpos is outside its joint limits: hand={self.hand}, "
                f"violations={violations}"
            )
        if self.official_initial_joint_pos is None:
            raise ValueError("official live joint seed is required")


@dataclass(frozen=True, slots=True)
class LiveEEFPoseCandidate:
    """One member of a live-state-derived EEF pose family."""

    candidate_id: str
    state_token: str
    eef_world_position: tuple[float, ...]
    eef_world_quaternion_xyzw: tuple[float, ...]
    source: str = LIVE_POSE_FAMILY_SOURCE

    def __post_init__(self) -> None:
        _token(self.candidate_id, "candidate_id")
        _token(self.state_token, "state_token")
        if self.source != LIVE_POSE_FAMILY_SOURCE:
            raise ValueError("live pose-family source contract mismatch")
        object.__setattr__(
            self,
            "eef_world_position",
            _vector(
                self.eef_world_position,
                3,
                "eef_world_position",
            ),
        )
        object.__setattr__(
            self,
            "eef_world_quaternion_xyzw",
            _vector(
                self.eef_world_quaternion_xyzw,
                4,
                "eef_world_quaternion_xyzw",
            ),
        )
        _quaternion(
            self.eef_world_quaternion_xyzw,
            "eef_world_quaternion_xyzw",
        )


@dataclass(frozen=True, slots=True)
class LiveSafeCarryProposalLimits:
    minimum_pose_candidates: int = 3
    maximum_pose_candidates: int = 32
    maximum_eef_displacement_from_live_m: float = 0.45
    maximum_body_to_candidate_reach_m: float = 1.20
    minimum_joint_margin_fraction: float = 0.05
    pose_match_position_tolerance_m: float = 1e-6
    pose_match_orientation_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "minimum_pose_candidates",
            "maximum_pose_candidates",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_pose_candidates < self.minimum_pose_candidates:
            raise ValueError("candidate-count interval is invalid")
        for name in (
            "maximum_eef_displacement_from_live_m",
            "maximum_body_to_candidate_reach_m",
            "pose_match_position_tolerance_m",
            "pose_match_orientation_tolerance",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        margin = _finite(
            self.minimum_joint_margin_fraction,
            "minimum_joint_margin_fraction",
        )
        if not 0.0 <= margin < 0.5:
            raise ValueError(
                "minimum_joint_margin_fraction must be in [0, 0.5)"
            )


@dataclass(frozen=True, slots=True)
class LiveSafeCarryProposalReport:
    """Private construction evidence for one selected candidate."""

    version: str
    candidate_source: str
    official_ik_method: str
    collision_aware: bool
    state_token: str
    attempted_candidates: int
    feasible_candidates: int
    selected_candidate_id: str
    selected_joint_margin_fraction: float
    selected_normalized_motion: float
    joint_margin_by_candidate: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.version != LIVE_SAFE_CARRY_PROPOSAL_VERSION:
            raise ValueError("proposal report version mismatch")
        if self.candidate_source != LIVE_SAFE_CARRY_CANDIDATE_SOURCE:
            raise ValueError("candidate source mismatch")
        if self.official_ik_method != OFFICIAL_STARTER_IK_METHOD:
            raise ValueError("official IK method mismatch")
        if self.collision_aware is not True:
            raise ValueError("proposal must be collision-aware")
        _token(self.state_token, "state_token")
        _token(self.selected_candidate_id, "selected_candidate_id")
        for name in ("attempted_candidates", "feasible_candidates"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if self.feasible_candidates > self.attempted_candidates:
            raise ValueError("feasible candidate count is invalid")
        selected_margin = _finite(
            self.selected_joint_margin_fraction,
            "selected_joint_margin_fraction",
        )
        selected_motion = _finite(
            self.selected_normalized_motion,
            "selected_normalized_motion",
        )
        if selected_margin < 0.0 or selected_motion < 0.0:
            raise ValueError("selected proposal scores must be non-negative")
        margins = dict(self.joint_margin_by_candidate)
        if (
            len(margins) != self.feasible_candidates
            or self.selected_candidate_id not in margins
            or any(not isinstance(key, str) or not key for key in margins)
        ):
            raise ValueError("joint-margin report keys are inconsistent")
        for candidate_id, margin in margins.items():
            if _finite(margin, f"joint_margin[{candidate_id}]") < 0.0:
                raise ValueError("joint-margin score must be non-negative")
        if not math.isclose(
            margins[self.selected_candidate_id],
            selected_margin,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("selected joint-margin score is inconsistent")
        object.__setattr__(
            self,
            "joint_margin_by_candidate",
            MappingProxyType(margins),
        )


@dataclass(frozen=True, slots=True)
class LiveSafeCarryProposal:
    candidate: SafeCarryCandidate
    report: LiveSafeCarryProposalReport

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, SafeCarryCandidate):
            raise TypeError("candidate must be SafeCarryCandidate")
        if not isinstance(self.report, LiveSafeCarryProposalReport):
            raise TypeError("report must be LiveSafeCarryProposalReport")


@dataclass(frozen=True, slots=True)
class _Feasible:
    index: int
    pose: LiveEEFPoseCandidate
    arm_target: tuple[float, ...]
    joint_margin: float
    normalized_motion: float


def propose_live_safe_carry_candidate(
    *,
    starter: Any,
    hand: str,
    live_state_provider: Callable[[Any, str], LiveRobotCarryState],
    candidate_pose_family: Callable[
        [LiveRobotCarryState], Sequence[LiveEEFPoseCandidate]
    ],
    official_pose_adapter: Callable[
        [Any, LiveEEFPoseCandidate], tuple[Any, Any]
    ],
    arm_target_from_official_ik: Callable[
        [Any, Any, str], Sequence[float]
    ],
    limits: LiveSafeCarryProposalLimits | None = None,
) -> LiveSafeCarryProposal:
    """Search current-scene EEF candidates through official Starter IK."""

    if hand not in {"left", "right"}:
        raise R1ProLiveSafeCarryProposalError(
            "hand must be left or right"
        )
    if getattr(starter, "arm", None) != hand:
        raise R1ProLiveSafeCarryProposalError(
            "requested hand does not match live Starter arm"
        )
    solver = getattr(starter, OFFICIAL_STARTER_IK_METHOD, None)
    if not callable(solver):
        raise R1ProLiveSafeCarryProposalError(
            f"Starter lacks {OFFICIAL_STARTER_IK_METHOD}"
        )
    for callback, name in (
        (live_state_provider, "live_state_provider"),
        (candidate_pose_family, "candidate_pose_family"),
        (official_pose_adapter, "official_pose_adapter"),
        (
            arm_target_from_official_ik,
            "arm_target_from_official_ik",
        ),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")
    active_limits = limits or LiveSafeCarryProposalLimits()
    if not isinstance(active_limits, LiveSafeCarryProposalLimits):
        raise TypeError("limits must be LiveSafeCarryProposalLimits")

    try:
        state = live_state_provider(starter, hand)
    except Exception as exc:
        raise R1ProLiveSafeCarryProposalError(
            f"live robot-state read failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(state, LiveRobotCarryState):
        raise R1ProLiveSafeCarryProposalError(
            "live_state_provider must return LiveRobotCarryState"
        )
    if state.hand != hand:
        raise R1ProLiveSafeCarryProposalError(
            "live robot state hand mismatch"
        )

    try:
        poses = tuple(candidate_pose_family(state))
    except Exception as exc:
        raise R1ProLiveSafeCarryProposalError(
            f"live candidate-family generation failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    _validate_pose_family(poses, state=state, limits=active_limits)

    feasible: list[_Feasible] = []
    margins: dict[str, float] = {}
    for index, pose in enumerate(poses):
        try:
            official_pose = official_pose_adapter(starter, pose)
            _validate_official_pose(
                official_pose,
                candidate=pose,
                limits=active_limits,
            )
            raw_solution = solver(
                official_pose,
                initial_joint_pos=state.official_initial_joint_pos,
                skip_obstacle_update=False,
            )
        except Exception as exc:
            raise R1ProLiveSafeCarryProposalError(
                f"official collision-aware IK failed operationally for "
                f"{pose.candidate_id!r}: {type(exc).__name__}: {exc}"
            ) from exc
        if raw_solution is None:
            continue
        try:
            arm_target = _vector(
                arm_target_from_official_ik(
                    starter,
                    raw_solution,
                    hand,
                ),
                7,
                "official IK arm target",
            )
        except Exception as exc:
            raise R1ProLiveSafeCarryProposalError(
                f"official IK arm-target extraction failed for "
                f"{pose.candidate_id!r}: {type(exc).__name__}: {exc}"
            ) from exc
        joint_margin, normalized_motion = _score_solution(
            arm_target,
            state=state,
        )
        margins[pose.candidate_id] = joint_margin
        feasible.append(
            _Feasible(
                index=index,
                pose=pose,
                arm_target=arm_target,
                joint_margin=joint_margin,
                normalized_motion=normalized_motion,
            )
        )

    if not feasible:
        raise R1ProLiveSafeCarryProposalError(
            "all live collision-aware IK candidates failed"
        )
    selected = max(
        feasible,
        key=lambda item: (
            item.joint_margin,
            -item.normalized_motion,
            -item.index,
        ),
    )
    if (
        selected.joint_margin
        < active_limits.minimum_joint_margin_fraction
    ):
        raise R1ProLiveSafeCarryProposalError(
            "best official IK candidate lacks required joint margin"
        )
    candidate = SafeCarryCandidate(
        hand=hand,
        arm_target=selected.arm_target,
        eef_world_position=selected.pose.eef_world_position,
        eef_world_quaternion_xyzw=(
            selected.pose.eef_world_quaternion_xyzw
        ),
        source=LIVE_SAFE_CARRY_CANDIDATE_SOURCE,
    )
    report = LiveSafeCarryProposalReport(
        version=LIVE_SAFE_CARRY_PROPOSAL_VERSION,
        candidate_source=LIVE_SAFE_CARRY_CANDIDATE_SOURCE,
        official_ik_method=OFFICIAL_STARTER_IK_METHOD,
        collision_aware=True,
        state_token=state.state_token,
        attempted_candidates=len(poses),
        feasible_candidates=len(feasible),
        selected_candidate_id=selected.pose.candidate_id,
        selected_joint_margin_fraction=selected.joint_margin,
        selected_normalized_motion=selected.normalized_motion,
        joint_margin_by_candidate=margins,
    )
    return LiveSafeCarryProposal(candidate=candidate, report=report)


def _validate_pose_family(
    poses: tuple[Any, ...],
    *,
    state: LiveRobotCarryState,
    limits: LiveSafeCarryProposalLimits,
) -> None:
    if not (
        limits.minimum_pose_candidates
        <= len(poses)
        <= limits.maximum_pose_candidates
    ):
        raise R1ProLiveSafeCarryProposalError(
            "live pose family candidate count is outside limits"
        )
    if any(not isinstance(item, LiveEEFPoseCandidate) for item in poses):
        raise R1ProLiveSafeCarryProposalError(
            "candidate_pose_family returned an invalid member"
        )
    ids = [item.candidate_id for item in poses]
    if len(set(ids)) != len(ids):
        raise R1ProLiveSafeCarryProposalError(
            "live pose candidate IDs must be unique"
        )
    if any(item.state_token != state.state_token for item in poses):
        raise R1ProLiveSafeCarryProposalError(
            "live pose candidate uses a stale state token"
        )
    body = np.asarray(state.body_world_position)
    live_eef = np.asarray(state.eef_world_position)
    signatures: set[tuple[float, ...]] = set()
    for item in poses:
        position = np.asarray(item.eef_world_position)
        if (
            np.linalg.norm(position - live_eef)
            > limits.maximum_eef_displacement_from_live_m
        ):
            raise R1ProLiveSafeCarryProposalError(
                f"candidate {item.candidate_id!r} is not local to live EEF"
            )
        if (
            np.linalg.norm(position - body)
            > limits.maximum_body_to_candidate_reach_m
        ):
            raise R1ProLiveSafeCarryProposalError(
                f"candidate {item.candidate_id!r} exceeds live body reach"
            )
        quaternion = _canonical_quaternion(
            item.eef_world_quaternion_xyzw
        )
        signature = tuple(
            round(value, 10)
            for value in (*item.eef_world_position, *quaternion)
        )
        signatures.add(signature)
    if len(signatures) < limits.minimum_pose_candidates:
        raise R1ProLiveSafeCarryProposalError(
            "live pose family contains duplicate fixed poses"
        )


def _validate_official_pose(
    value: Any,
    *,
    candidate: LiveEEFPoseCandidate,
    limits: LiveSafeCarryProposalLimits,
) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise R1ProLiveSafeCarryProposalError(
            "official_pose_adapter must return (position, quaternion)"
        )
    position = _native_vector(value[0], 3, "official pose position")
    quaternion = _native_vector(value[1], 4, "official pose quaternion")
    if not np.allclose(
        position,
        candidate.eef_world_position,
        atol=limits.pose_match_position_tolerance_m,
        rtol=0.0,
    ):
        raise R1ProLiveSafeCarryProposalError(
            "official IK pose position differs from its candidate"
        )
    if not _quaternion_close(
        quaternion,
        candidate.eef_world_quaternion_xyzw,
        tolerance=limits.pose_match_orientation_tolerance,
    ):
        raise R1ProLiveSafeCarryProposalError(
            "official IK pose orientation differs from its candidate"
        )


def _score_solution(
    arm_target: tuple[float, ...],
    *,
    state: LiveRobotCarryState,
) -> tuple[float, float]:
    target = np.asarray(arm_target)
    current = np.asarray(state.arm_qpos)
    lower = np.asarray(state.arm_lower_limits)
    upper = np.asarray(state.arm_upper_limits)
    if np.any(target < lower) or np.any(target > upper):
        raise R1ProLiveSafeCarryProposalError(
            "official IK arm target is outside live joint limits"
        )
    span = upper - lower
    lower_margin = (target - lower) / span
    upper_margin = (upper - target) / span
    margin = float(np.min(np.minimum(lower_margin, upper_margin)))
    motion = float(np.linalg.norm((target - current) / span))
    return margin, motion


def _native_vector(value: Any, size: int, name: str) -> np.ndarray:
    item = value
    for method in ("detach", "cpu"):
        callback = getattr(item, method, None)
        if callable(callback):
            item = callback()
    numpy_method = getattr(item, "numpy", None)
    if callable(numpy_method):
        item = numpy_method()
    array = np.asarray(item, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise R1ProLiveSafeCarryProposalError(
            f"{name} must be a finite {size}-vector"
        )
    return array


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    array = _native_vector(value, size, name)
    return tuple(float(item) for item in array)


def _quaternion(value: Sequence[float], name: str) -> None:
    if np.linalg.norm(np.asarray(value, dtype=np.float64)) <= 1e-8:
        raise ValueError(f"{name} has zero norm")


def _canonical_quaternion(value: Sequence[float]) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    array = array / np.linalg.norm(array)
    for component in reversed(array):
        if abs(component) > 1e-12:
            if component < 0.0:
                array = -array
            break
    return tuple(float(item) for item in array)


def _quaternion_close(
    first: Sequence[float],
    second: Sequence[float],
    *,
    tolerance: float,
) -> bool:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return bool(
        min(np.linalg.norm(a - b), np.linalg.norm(a + b))
        <= tolerance
    )


def _token(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
