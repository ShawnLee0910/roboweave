"""Concrete R1Pro bindings for the B1 one-scene calibration lifecycle.

The expensive simulator and official Starter class are injected after the
live runtime imports.  This module has no OmniGibson import.  It owns the
strict transition:

``observation-only approach -> frozen live state -> candidate proposal
-> exact-candidate empty-wrist calibration``.

After proposal, base commands are rejected and the live base pose is checked
before and after every calibration step.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_empty_wrist_calibration import (
    EmptyWristCalibrationConfig,
    R1ProEmptyWristCalibrationActuator,
    run_empty_wrist_negative_calibration,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_live_safe_carry_binding import (
    R1ProDynamicCandidateFamilyConfig,
    build_r1pro_live_safe_carry_proposal_bindings,
    read_r1pro_live_robot_carry_state,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_live_safe_carry_candidate import (
    LiveRobotCarryState,
    LiveSafeCarryProposal,
    LiveSafeCarryProposalLimits,
    propose_live_safe_carry_candidate,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_public_wrist_evidence import (
    BootstrapPublicWristCalibrationRecord,
    PublicWristCalibrationRecord,
    WristVisualThresholds,
    safe_carry_candidate_sha256,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    B1_SINGLE_LOAD_RUNNER_VERSION,
    B1CalibrationGraspLift,
    B1EmptyWristCalibration,
    B1FrozenBasePoseEvidence,
    B1ObservationOnlyApproach,
    B1PostResetAssemblyPlan,
)
from behavior_hl.embodied.evaluator_bridge.same_scene_reset_attestation import (
    PrivateResetSnapshot,
)
from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
    SafeCarryCalibrationRecord,
)
from behavior_hl.embodied.policy.observations import (
    validate_public_observation,
)
from behavior_hl.embodied.policy.r1pro.action_layout import (
    R1PRO_ACTION_LAYOUT,
)
from behavior_hl.embodied.runtime.public_single_leaf_witness import (
    PublicSingleLeafWitness,
)


R1PRO_B1_SINGLE_LOAD_BINDINGS_VERSION = (
    "behavior2026.r1pro-b1-single-load-bindings/v1"
)
R1PRO_APPROACH_STATE_TOKEN_VERSION = (
    "behavior2026.r1pro-approach-state-token/v1"
)
R1PRO_CANDIDATE_SOURCE_STATE_TOKEN_VERSION = (
    "behavior2026.r1pro-candidate-source-state-token/v1"
)
R1PRO_FROZEN_BASE_POSE_TOKEN_VERSION = (
    "behavior2026.r1pro-frozen-base-pose-token/v1"
)


class R1ProB1SingleLoadBindingError(RuntimeError):
    """Fail-closed live composition or frozen-base error."""


@dataclass(frozen=True, slots=True)
class R1ProCandidateSourceStateToken:
    """Full robot-state token valid only at candidate proposal start."""

    hand: str
    live_robot_state_token: str
    version: str = R1PRO_CANDIDATE_SOURCE_STATE_TOKEN_VERSION

    def __post_init__(self) -> None:
        if self.version != R1PRO_CANDIDATE_SOURCE_STATE_TOKEN_VERSION:
            raise ValueError("candidate source token version mismatch")
        if self.hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        if (
            not isinstance(self.live_robot_state_token, str)
            or not self.live_robot_state_token.startswith("live-")
            or len(self.live_robot_state_token) != 69
        ):
            raise ValueError("live_robot_state_token is invalid")


@dataclass(frozen=True, slots=True)
class R1ProFrozenBasePoseToken:
    """Base-only pose token that remains valid while the arm moves."""

    hand: str
    base_world_position: tuple[float, ...]
    base_world_quaternion_xyzw: tuple[float, ...]
    pose_sha256: str = field(init=False)
    version: str = R1PRO_FROZEN_BASE_POSE_TOKEN_VERSION

    def __post_init__(self) -> None:
        if self.version != R1PRO_FROZEN_BASE_POSE_TOKEN_VERSION:
            raise ValueError("frozen base pose token version mismatch")
        if self.hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        object.__setattr__(
            self,
            "base_world_position",
            _vector(
                self.base_world_position,
                3,
                "base_world_position",
            ),
        )
        quaternion = _vector(
            self.base_world_quaternion_xyzw,
            4,
            "base_world_quaternion_xyzw",
        )
        if np.linalg.norm(quaternion) <= 1e-8:
            raise ValueError("base quaternion has zero norm")
        object.__setattr__(
            self,
            "base_world_quaternion_xyzw",
            quaternion,
        )
        digest = hashlib.sha256()
        digest.update(R1PRO_FROZEN_BASE_POSE_TOKEN_VERSION.encode())
        digest.update(self.hand.encode())
        digest.update(
            np.ascontiguousarray(
                self.base_world_position,
                dtype=np.float64,
            ).tobytes()
        )
        digest.update(
            np.ascontiguousarray(
                self.base_world_quaternion_xyzw,
                dtype=np.float64,
            ).tobytes()
        )
        object.__setattr__(self, "pose_sha256", digest.hexdigest())


@dataclass(frozen=True, slots=True)
class R1ProApproachStateToken:
    """Two-scope attestation captured at the final approach state."""

    candidate_source_state_token: R1ProCandidateSourceStateToken
    frozen_base_pose_token: R1ProFrozenBasePoseToken
    version: str = R1PRO_APPROACH_STATE_TOKEN_VERSION

    def __post_init__(self) -> None:
        if self.version != R1PRO_APPROACH_STATE_TOKEN_VERSION:
            raise ValueError("approach state token version mismatch")
        if not isinstance(
            self.candidate_source_state_token,
            R1ProCandidateSourceStateToken,
        ):
            raise TypeError(
                "candidate_source_state_token has the wrong type"
            )
        if not isinstance(
            self.frozen_base_pose_token,
            R1ProFrozenBasePoseToken,
        ):
            raise TypeError("frozen_base_pose_token has the wrong type")
        if (
            self.candidate_source_state_token.hand
            != self.frozen_base_pose_token.hand
        ):
            raise ValueError("approach token hand scopes differ")

    @property
    def hand(self) -> str:
        return self.candidate_source_state_token.hand

    @property
    def live_robot_state_token(self) -> str:
        return self.candidate_source_state_token.live_robot_state_token

    @property
    def base_world_position(self) -> tuple[float, ...]:
        return self.frozen_base_pose_token.base_world_position

    @property
    def base_world_quaternion_xyzw(self) -> tuple[float, ...]:
        return self.frozen_base_pose_token.base_world_quaternion_xyzw


@dataclass(frozen=True, slots=True)
class R1ProB1SingleLoadCalibrationConfig:
    """Explicit candidate, wrist, and frozen-base calibration policy."""

    hand: str
    roi_xyxy: tuple[int, int, int, int]
    wrist_thresholds: WristVisualThresholds
    proposal_limits: LiveSafeCarryProposalLimits = field(
        default_factory=LiveSafeCarryProposalLimits
    )
    candidate_family: R1ProDynamicCandidateFamilyConfig = field(
        default_factory=R1ProDynamicCandidateFamilyConfig
    )
    empty_wrist_trials: int = 3
    stable_samples_per_trial: int = 3
    max_steps_per_phase: int = 256
    empty_qpos_tolerance: float = 0.003
    eef_position_tolerance_m: float = 0.01
    eef_orientation_tolerance_deg: float = 2.0
    maximum_base_position_drift_m: float = 1e-4
    maximum_base_orientation_drift_rad: float = 1e-4
    base_command_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        if not isinstance(self.wrist_thresholds, WristVisualThresholds):
            raise TypeError(
                "wrist_thresholds must be WristVisualThresholds"
            )
        if not isinstance(
            self.proposal_limits,
            LiveSafeCarryProposalLimits,
        ):
            raise TypeError(
                "proposal_limits must be LiveSafeCarryProposalLimits"
            )
        if not isinstance(
            self.candidate_family,
            R1ProDynamicCandidateFamilyConfig,
        ):
            raise TypeError(
                "candidate_family must be "
                "R1ProDynamicCandidateFamilyConfig"
            )
        # Reuse the canonical calibration config validator with a sentinel
        # candidate digest; the live digest is supplied only after proposal.
        EmptyWristCalibrationConfig(
            hand=self.hand,
            roi_xyxy=self.roi_xyxy,
            thresholds=self.wrist_thresholds,
            safe_carry_candidate_sha256="0" * 64,
            trials=self.empty_wrist_trials,
            stable_samples_per_trial=self.stable_samples_per_trial,
            max_steps_per_phase=self.max_steps_per_phase,
            empty_qpos_tolerance=self.empty_qpos_tolerance,
            eef_position_tolerance_m=self.eef_position_tolerance_m,
            eef_orientation_tolerance_deg=(
                self.eef_orientation_tolerance_deg
            ),
        )
        for name in (
            "maximum_base_position_drift_m",
            "maximum_base_orientation_drift_rad",
            "base_command_epsilon",
        ):
            value = _finite_scalar(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

    def empty_wrist_config(
        self,
        *,
        proposal: LiveSafeCarryProposal,
        minimum_trials: int,
    ) -> EmptyWristCalibrationConfig:
        if (
            isinstance(minimum_trials, bool)
            or not isinstance(minimum_trials, int)
            or minimum_trials < 3
        ):
            raise ValueError("minimum_trials must be an integer >= 3")
        return EmptyWristCalibrationConfig(
            hand=self.hand,
            roi_xyxy=self.roi_xyxy,
            thresholds=self.wrist_thresholds,
            safe_carry_candidate_sha256=safe_carry_candidate_sha256(
                proposal.candidate
            ),
            trials=max(self.empty_wrist_trials, minimum_trials),
            stable_samples_per_trial=self.stable_samples_per_trial,
            max_steps_per_phase=self.max_steps_per_phase,
            empty_qpos_tolerance=self.empty_qpos_tolerance,
            eef_position_tolerance_m=self.eef_position_tolerance_m,
            eef_orientation_tolerance_deg=(
                self.eef_orientation_tolerance_deg
            ),
        )


def capture_r1pro_approach_state_token(
    *,
    starter: Any,
    hand: str,
) -> R1ProApproachStateToken:
    """Capture the current robot state after the final approach step."""

    state = read_r1pro_live_robot_carry_state(starter, hand)
    return _token_from_live_state(state)


def validate_r1pro_approach_state_token(
    *,
    starter: Any,
    token: R1ProApproachStateToken,
) -> LiveRobotCarryState:
    """Re-read the live robot and reject a stale or foreign approach token."""

    if not isinstance(token, R1ProApproachStateToken):
        raise TypeError("token must be R1ProApproachStateToken")
    state = read_r1pro_live_robot_carry_state(starter, token.hand)
    if state.state_token != token.live_robot_state_token:
        raise R1ProB1SingleLoadBindingError(
            "approach state token is stale"
        )
    if not np.array_equal(
        np.asarray(state.body_world_position),
        np.asarray(token.base_world_position),
    ):
        raise R1ProB1SingleLoadBindingError(
            "approach token base position differs from live state"
        )
    if not _same_quaternion(
        state.body_world_quaternion_xyzw,
        token.base_world_quaternion_xyzw,
        tolerance_rad=1e-12,
    ):
        raise R1ProB1SingleLoadBindingError(
            "approach token base orientation differs from live state"
        )
    return state


class R1ProNoBaseMotionGuard:
    """Reject base commands and attest base pose around every live step."""

    def __init__(
        self,
        *,
        robot: Any,
        token: R1ProApproachStateToken,
        maximum_position_drift_m: float,
        maximum_orientation_drift_rad: float,
        base_command_epsilon: float,
    ) -> None:
        if robot is None or not callable(
            getattr(robot, "get_position_orientation", None)
        ):
            raise R1ProB1SingleLoadBindingError(
                "robot lacks get_position_orientation"
            )
        if not isinstance(token, R1ProApproachStateToken):
            raise TypeError("token must be R1ProApproachStateToken")
        self._robot = robot
        self._token = token
        self._position_tolerance = _nonnegative(
            maximum_position_drift_m,
            "maximum_position_drift_m",
        )
        self._orientation_tolerance = _nonnegative(
            maximum_orientation_drift_rad,
            "maximum_orientation_drift_rad",
        )
        self._base_command_epsilon = _nonnegative(
            base_command_epsilon,
            "base_command_epsilon",
        )
        self.assert_base_pose()

    @property
    def token(self) -> R1ProApproachStateToken:
        return self._token

    def assert_base_pose(self) -> None:
        try:
            position, quaternion = self._robot.get_position_orientation(
                frame="world",
                clone=True,
            )
            position = np.asarray(position, dtype=np.float64)
            quaternion = np.asarray(quaternion, dtype=np.float64)
        except Exception as exc:
            raise R1ProB1SingleLoadBindingError(
                "frozen base pose read failed"
            ) from exc
        if (
            position.shape != (3,)
            or quaternion.shape != (4,)
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all()
        ):
            raise R1ProB1SingleLoadBindingError(
                "frozen base pose is invalid"
            )
        position_drift = float(
            np.linalg.norm(
                position - np.asarray(self._token.base_world_position)
            )
        )
        if position_drift > self._position_tolerance:
            raise R1ProB1SingleLoadBindingError(
                "base position moved after candidate proposal"
            )
        if not _same_quaternion(
            quaternion,
            self._token.base_world_quaternion_xyzw,
            tolerance_rad=self._orientation_tolerance,
        ):
            raise R1ProB1SingleLoadBindingError(
                "base orientation moved after candidate proposal"
            )

    def validate_action(self, action: Any) -> np.ndarray:
        array = np.asarray(action, dtype=np.float64)
        if (
            array.shape != (R1PRO_ACTION_LAYOUT.dimension,)
            or not np.isfinite(array).all()
        ):
            raise R1ProB1SingleLoadBindingError(
                "post-candidate action must be finite R1Pro action"
            )
        base = array[R1PRO_ACTION_LAYOUT.slice_for("base")]
        if np.any(np.abs(base) > self._base_command_epsilon):
            raise R1ProB1SingleLoadBindingError(
                "base motion command is forbidden after candidate proposal; "
                f"base_action={base.tolist()}"
            )
        return np.asarray(array, dtype=np.float32)

    def step_once(
        self,
        callback: Callable[[str], int],
        label: str,
    ) -> int:
        if not callable(callback):
            raise TypeError("step callback must be callable")
        self.assert_base_pose()
        result = callback(label)
        if isinstance(result, bool) or not isinstance(result, int) or result != 1:
            raise R1ProB1SingleLoadBindingError(
                "step_once must confirm exactly one simulator step"
            )
        self.assert_base_pose()
        return 1


class _NoBaseMotionControl:
    def __init__(self, delegate: Any, guard: R1ProNoBaseMotionGuard) -> None:
        for method in (
            "command_direct_action",
            "command_carry_action",
            "hold",
        ):
            if not callable(getattr(delegate, method, None)):
                raise R1ProB1SingleLoadBindingError(
                    f"control is missing {method}"
                )
        self._delegate = delegate
        self._guard = guard

    def command_direct_action(self, action: Any) -> Any:
        self._guard.assert_base_pose()
        return self._delegate.command_direct_action(
            self._guard.validate_action(action)
        )

    def command_carry_action(
        self,
        action: Any,
        *,
        active_hand: str,
    ) -> Any:
        if active_hand not in {"left", "right"}:
            raise R1ProB1SingleLoadBindingError(
                "active_hand must be left or right"
            )
        self._guard.assert_base_pose()
        candidate = self._guard.validate_action(action)
        # Do not rewrite the active gripper command.  The official control
        # method remains responsible for enforcing physical closure.
        return self._delegate.command_carry_action(
            candidate,
            active_hand=active_hand,
        )

    def hold(self) -> Any:
        self._guard.assert_base_pose()
        return self._delegate.hold()


class R1ProB1SingleLoadLiveBindings:
    """Complete v3 runner binding with explicit witness authoring."""

    api_contract_version = B1_SINGLE_LOAD_RUNNER_VERSION
    implementation_version = R1PRO_B1_SINGLE_LOAD_BINDINGS_VERSION

    def __init__(
        self,
        *,
        config: R1ProB1SingleLoadCalibrationConfig,
        arm_selectable_starter_factory: Callable[..., Any],
        live_public_snapshot: Callable[[Any], Mapping[str, Any]],
        step_once: Callable[[Any, str], int],
        capture_private_reset_snapshot_callback: Callable[..., Any],
        observation_only_approach_callback: Callable[..., Any],
        physical_grasp_lift_callback: Callable[..., Any],
        author_and_freeze_single_leaf_witness_callback: Callable[
            ..., Any
        ],
        build_post_reset_plan_callback: Callable[..., Any],
    ) -> None:
        if not isinstance(config, R1ProB1SingleLoadCalibrationConfig):
            raise TypeError(
                "config must be R1ProB1SingleLoadCalibrationConfig"
            )
        callbacks = {
            "arm_selectable_starter_factory": (
                arm_selectable_starter_factory
            ),
            "live_public_snapshot": live_public_snapshot,
            "step_once": step_once,
            "capture_private_reset_snapshot_callback": (
                capture_private_reset_snapshot_callback
            ),
            "observation_only_approach_callback": (
                observation_only_approach_callback
            ),
            "physical_grasp_lift_callback": (
                physical_grasp_lift_callback
            ),
            "author_and_freeze_single_leaf_witness_callback": (
                author_and_freeze_single_leaf_witness_callback
            ),
            "build_post_reset_plan_callback": (
                build_post_reset_plan_callback
            ),
        }
        for name, callback in callbacks.items():
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        self._config = config
        self._starter_factory = arm_selectable_starter_factory
        self._live_public_snapshot = live_public_snapshot
        self._step_callback = step_once
        self._capture_reset_callback = (
            capture_private_reset_snapshot_callback
        )
        self._approach_callback = observation_only_approach_callback
        self._grasp_callback = physical_grasp_lift_callback
        self._witness_author_callback = (
            author_and_freeze_single_leaf_witness_callback
        )
        self._post_reset_callback = build_post_reset_plan_callback
        self._evaluator: Any | None = None
        self._starter: Any | None = None
        self._approach: B1ObservationOnlyApproach | None = None
        self._approach_token: R1ProApproachStateToken | None = None
        self._proposal: LiveSafeCarryProposal | None = None
        self._empty_wrist: B1EmptyWristCalibration | None = None
        self._guard: R1ProNoBaseMotionGuard | None = None
        self._witness: PublicSingleLeafWitness | None = None
        self._poisoned = False

    def capture_private_reset_snapshot(
        self,
        *,
        evaluator: Any,
        phase: str,
        run_dir: Path,
    ) -> PrivateResetSnapshot:
        self._require_operational()
        result = self._capture_reset_callback(
            evaluator=evaluator,
            phase=phase,
            run_dir=run_dir,
        )
        if not isinstance(result, PrivateResetSnapshot):
            raise R1ProB1SingleLoadBindingError(
                "reset snapshot callback returned the wrong type"
            )
        return result

    def observation_only_approach(
        self,
        *,
        evaluator: Any,
        control: Any,
        run_dir: Path,
    ) -> B1ObservationOnlyApproach:
        self._require_operational()
        if self._approach is not None:
            raise R1ProB1SingleLoadBindingError(
                "observation-only approach may run only once"
            )

        # Fail before spending any model calls or simulator approach steps if
        # the physical manipulation backend cannot be constructed.  The same
        # cached Starter is reused after the final public RGB-D grasp query.
        before_starter = _capture_starter_neutral_robot_state(
            evaluator=evaluator,
            hand=self._config.hand,
        )
        starter = self._starter_for(evaluator)
        after_starter = _capture_starter_neutral_robot_state(
            evaluator=evaluator,
            hand=self._config.hand,
        )
        unchanged = before_starter == after_starter
        neutrality_path = (
            Path(run_dir) / "private" / "starter_state_neutrality.json"
        )
        neutrality_path.parent.mkdir(parents=True, exist_ok=True)
        neutrality_path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "behavior2026.starter-state-neutrality/v1"
                    ),
                    "unchanged": unchanged,
                    "before_sha256": _canonical_sha256(before_starter),
                    "after_sha256": _canonical_sha256(after_starter),
                    "curobo_device": getattr(
                        starter,
                        "_behavior_hl_curobo_device",
                        None,
                    ),
                    "simulator_tensor_device": getattr(
                        starter,
                        "_behavior_hl_simulator_tensor_device",
                        None,
                    ),
                    "nonselected_torch_allocated_bytes": getattr(
                        starter,
                        (
                            "_behavior_hl_nonselected_torch_"
                            "allocated_bytes"
                        ),
                        None,
                    ),
                    "curobo_cuda_graphs_enabled": getattr(
                        starter,
                        "_behavior_hl_curobo_cuda_graphs_enabled",
                        None,
                    ),
                    "floor_plane_absent": getattr(
                        starter,
                        "_behavior_hl_floor_plane_absent",
                        None,
                    ),
                    "curobo_pose_device_compatibility": getattr(
                        starter,
                        (
                            "_behavior_hl_curobo_pose_device_"
                            "compatibility"
                        ),
                        None,
                    ),
                    "curobo_pose_redirect_events": getattr(
                        starter,
                        "_behavior_hl_curobo_pose_redirect_events",
                        None,
                    ),
                    "curobo_arm_base_redirect_events": getattr(
                        starter,
                        (
                            "_behavior_hl_curobo_arm_base_redirect_"
                            "events"
                        ),
                        None,
                    ),
                    "curobo_usd_parser_redirect_events": getattr(
                        starter,
                        (
                            "_behavior_hl_curobo_usd_parser_redirect_"
                            "events"
                        ),
                        None,
                    ),
                    "curobo_source_hashes": getattr(
                        starter,
                        "_behavior_hl_curobo_source_hashes",
                        None,
                    ),
                    "curobo_gripper_lock_attestation": getattr(
                        starter,
                        (
                            "_behavior_hl_curobo_gripper_lock_"
                            "attestation"
                        ),
                        None,
                    ),
                    "mixed_q_to_action_attestation": getattr(
                        starter,
                        (
                            "_behavior_hl_mixed_q_to_action_"
                            "attestation"
                        ),
                        None,
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if not unchanged:
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "Starter construction changed live robot state"
            )
        captures: list[R1ProApproachStateToken] = []

        def capture_final_state() -> R1ProApproachStateToken:
            token = capture_r1pro_approach_state_token(
                starter=self._starter_for(evaluator),
                hand=self._config.hand,
            )
            captures.append(token)
            return token

        result = self._approach_callback(
            evaluator=evaluator,
            control=control,
            run_dir=run_dir,
            live_public_snapshot=lambda: self._snapshot(evaluator),
            step_once=lambda label: self._raw_step(evaluator, label),
            capture_final_state=capture_final_state,
        )
        if not isinstance(result, B1ObservationOnlyApproach):
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "approach callback returned the wrong type"
            )
        if len(captures) != 1:
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "approach must capture final live state exactly once"
            )
        token = _token_from_approach(
            approach=result,
            hand=self._config.hand,
        )
        if token != captures[0]:
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "approach result changed the captured final live state"
            )
        validate_r1pro_approach_state_token(
            starter=self._starter_for(evaluator),
            token=token,
        )
        self._approach = result
        self._approach_token = token
        return result

    def propose_candidate(
        self,
        *,
        evaluator: Any,
        control: Any,
        approach: B1ObservationOnlyApproach,
        run_dir: Path,
    ) -> LiveSafeCarryProposal:
        del control, run_dir
        self._require_approach(evaluator, approach)
        if self._proposal is not None:
            raise R1ProB1SingleLoadBindingError(
                "candidate proposal may run only once"
            )
        starter = self._starter_for(evaluator)
        token = self._required_token()
        validate_r1pro_approach_state_token(
            starter=starter,
            token=token,
        )
        callbacks = build_r1pro_live_safe_carry_proposal_bindings(
            starter=starter,
            hand=self._config.hand,
            config=self._config.candidate_family,
        )
        proposal = propose_live_safe_carry_candidate(
            starter=starter,
            hand=self._config.hand,
            limits=self._config.proposal_limits,
            **callbacks.proposal_kwargs(),
        )
        if proposal.report.state_token != token.live_robot_state_token:
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "candidate does not derive from final approach state"
            )
        guard = R1ProNoBaseMotionGuard(
            robot=starter.robot,
            token=token,
            maximum_position_drift_m=(
                self._config.maximum_base_position_drift_m
            ),
            maximum_orientation_drift_rad=(
                self._config.maximum_base_orientation_drift_rad
            ),
            base_command_epsilon=self._config.base_command_epsilon,
        )
        self._proposal = proposal
        self._guard = guard
        return proposal

    def calibrate_empty_wrist(
        self,
        *,
        evaluator: Any,
        control: Any,
        approach: B1ObservationOnlyApproach,
        proposal: LiveSafeCarryProposal,
        minimum_trials: int,
        run_dir: Path,
    ) -> B1EmptyWristCalibration:
        self._require_proposal(evaluator, approach, proposal)
        if self._empty_wrist is not None:
            raise R1ProB1SingleLoadBindingError(
                "empty wrist calibration may run only once"
            )
        starter = self._starter_for(evaluator)
        guard = self._required_guard()
        # The full candidate-source token is intentionally not revalidated
        # here.  Candidate proposal is its only validity boundary; arm motion
        # after proposal is legal.  From this point onward only the frozen
        # base token and exact candidate provenance remain invariant.
        guard.assert_base_pose()
        guarded_control = _NoBaseMotionControl(control, guard)
        actuator = R1ProEmptyWristCalibrationActuator(
            starter=starter,
            control=guarded_control,
            evaluator_step=lambda label: guard.step_once(
                lambda inner_label: self._raw_step(
                    evaluator,
                    inner_label,
                ),
                label,
            ),
            live_public_snapshot=lambda: self._snapshot(evaluator),
            eef_position_tolerance_m=(
                self._config.eef_position_tolerance_m
            ),
            eef_orientation_tolerance_deg=(
                self._config.eef_orientation_tolerance_deg
            ),
        )
        # Official Starter ARM trajectories are full 28D waypoints whose
        # locked base entries may reflect a slightly stale CuRobo snapshot.
        # Empty-wrist calibration is the post-candidate frozen-base phase, so
        # scope the q_to_action base-target correction to this call only. The
        # normal adapter conversion and the no-base guard remain active.
        had_freeze_base_context = hasattr(
            starter.robot,
            "_behavior_hl_freeze_base_q_to_action",
        )
        previous_freeze_base_context = getattr(
            starter.robot,
            "_behavior_hl_freeze_base_q_to_action",
            False,
        )
        had_freeze_base_authorization = hasattr(
            starter.robot,
            "_behavior_hl_authorized_frozen_base_arm_motion",
        )
        previous_freeze_base_authorization = getattr(
            starter.robot,
            "_behavior_hl_authorized_frozen_base_arm_motion",
            False,
        )
        starter.robot._behavior_hl_freeze_base_q_to_action = True
        starter.robot._behavior_hl_authorized_frozen_base_arm_motion = True
        try:
            result = run_empty_wrist_negative_calibration(
                candidate=proposal.candidate,
                config=self._config.empty_wrist_config(
                    proposal=proposal,
                    minimum_trials=minimum_trials,
                ),
                actuator=actuator,
            )
        finally:
            if had_freeze_base_context:
                starter.robot._behavior_hl_freeze_base_q_to_action = (
                    previous_freeze_base_context
                )
            else:
                delattr(
                    starter.robot,
                    "_behavior_hl_freeze_base_q_to_action",
                )
            if had_freeze_base_authorization:
                starter.robot._behavior_hl_authorized_frozen_base_arm_motion = (
                    previous_freeze_base_authorization
                )
            else:
                delattr(
                    starter.robot,
                    "_behavior_hl_authorized_frozen_base_arm_motion",
                )
        diagnostic_path = (
            Path(run_dir) / "private" / "empty_wrist_calibration.json"
        )
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(
            json.dumps(
                dict(result.to_manifest()),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if not result.passed or result.bootstrap_record is None:
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "empty wrist calibration failed: "
                f"{result.failure_reason or 'unknown failure'}"
            )
        guard.assert_base_pose()
        empty_wrist = B1EmptyWristCalibration(
            bootstrap=result.bootstrap_record,
            frozen_base=self._frozen_base_evidence(
                evaluator=evaluator,
                phase="after_empty_wrist",
            ),
            base_motion_commanded=False,
        )
        self._empty_wrist = empty_wrist
        return empty_wrist

    def physical_grasp_lift(
        self,
        *,
        evaluator: Any,
        control: Any,
        approach: B1ObservationOnlyApproach,
        proposal: LiveSafeCarryProposal,
        empty_wrist: B1EmptyWristCalibration,
        run_dir: Path,
    ) -> B1CalibrationGraspLift:
        self._require_proposal(evaluator, approach, proposal)
        if empty_wrist is not self._empty_wrist:
            raise R1ProB1SingleLoadBindingError(
                "physical grasp received a foreign empty-wrist record"
            )
        guard = self._required_guard()
        guard.assert_base_pose()
        result = self._grasp_callback(
            evaluator=evaluator,
            control=_NoBaseMotionControl(control, guard),
            approach=approach,
            proposal=proposal,
            empty_wrist=empty_wrist,
            run_dir=run_dir,
            starter=self._starter_for(evaluator),
            live_public_snapshot=lambda: self._snapshot(evaluator),
            step_once=lambda label: guard.step_once(
                lambda inner_label: self._raw_step(
                    evaluator,
                    inner_label,
                ),
                label,
            ),
        )
        guard.assert_base_pose()
        if not isinstance(result, B1CalibrationGraspLift):
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "physical grasp callback returned the wrong type"
            )
        if (
            result.consumed_target_token != approach.target_token
            or result.consumed_candidate_source_state_token
            != proposal.report.state_token
            or result.base_motion_commanded
        ):
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "physical grasp violated frozen target/base contract"
            )
        return result

    def capture_frozen_base_pose(
        self,
        *,
        evaluator: Any,
        phase: str,
        run_dir: Path,
    ) -> B1FrozenBasePoseEvidence:
        del run_dir
        self._require_operational()
        return self._frozen_base_evidence(
            evaluator=evaluator,
            phase=phase,
        )

    def author_and_freeze_single_leaf_witness(
        self,
        *,
        evaluator: Any,
        control: Any,
        approach: B1ObservationOnlyApproach,
        formal_wrist: PublicWristCalibrationRecord,
        safe_carry: SafeCarryCalibrationRecord,
        run_dir: Path,
    ) -> PublicSingleLeafWitness:
        self._require_approach(evaluator, approach)
        if self._witness is not None:
            raise R1ProB1SingleLoadBindingError(
                "single-leaf witness may be authored only once"
            )
        if (
            not isinstance(formal_wrist, PublicWristCalibrationRecord)
            or not isinstance(safe_carry, SafeCarryCalibrationRecord)
            or not safe_carry.passed
            or formal_wrist.hand != self._config.hand
            or safe_carry.candidate.hand != self._config.hand
            or formal_wrist.safe_carry_calibration_sha256
            != safe_carry.sha256
        ):
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "witness author requires the matching passing formal "
                "wrist/safe-carry calibration"
            )
        result = self._witness_author_callback(
            evaluator=evaluator,
            control=control,
            approach=approach,
            formal_wrist=formal_wrist,
            safe_carry=safe_carry,
            run_dir=run_dir,
        )
        if not isinstance(result, PublicSingleLeafWitness):
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "witness author returned the wrong type"
            )
        if result.calibrated_hand != self._config.hand:
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "witness changed the calibrated hand"
            )
        self._witness = result
        return result

    def build_post_reset_plan(
        self,
        *,
        evaluator: Any,
        control: Any,
        formal_wrist: Any,
        safe_carry: Any,
        witness: PublicSingleLeafWitness,
        run_dir: Path,
    ) -> B1PostResetAssemblyPlan:
        self._require_evaluator(evaluator)
        if (
            self._witness is None
            or not isinstance(witness, PublicSingleLeafWitness)
            or witness is not self._witness
            or witness.witness_sha256 != self._witness.witness_sha256
        ):
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "post-reset plan requires the exact witness authored "
                "before reset"
            )
        result = self._post_reset_callback(
            evaluator=evaluator,
            control=control,
            formal_wrist=formal_wrist,
            safe_carry=safe_carry,
            witness=witness,
            run_dir=run_dir,
            starter=self._starter_for(evaluator),
            live_public_snapshot=lambda: self._snapshot(evaluator),
        )
        if not isinstance(result, B1PostResetAssemblyPlan):
            raise R1ProB1SingleLoadBindingError(
                "post-reset callback returned the wrong type"
            )
        return result

    def _starter_for(self, evaluator: Any) -> Any:
        self._require_evaluator(evaluator)
        if self._starter is not None:
            return self._starter
        env = getattr(evaluator, "env", None)
        robot = getattr(evaluator, "robot", None)
        if env is None or robot is None:
            raise R1ProB1SingleLoadBindingError(
                "evaluator must expose live env and robot"
            )
        try:
            starter = self._starter_factory(
                env,
                robot,
                selected_arm=self._config.hand,
                enable_head_tracking=False,
                task_relevant_objects_only=False,
                curobo_batch_size=1,
            )
        except Exception as exc:
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "ArmSelectable Starter construction failed"
            ) from exc
        if (
            getattr(starter, "arm", None) != self._config.hand
            or getattr(starter, "robot", None) is not robot
        ):
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "ArmSelectable Starter is bound to the wrong robot or hand"
            )
        self._starter = starter
        return starter

    def _snapshot(self, evaluator: Any) -> Mapping[str, Any]:
        self._require_evaluator(evaluator)
        snapshot = self._live_public_snapshot(evaluator)
        try:
            validate_public_observation(snapshot)
        except Exception as exc:
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "snapshot provider did not return sanitized public RGB-D"
            ) from exc
        return snapshot

    def _raw_step(self, evaluator: Any, label: str) -> int:
        self._require_evaluator(evaluator)
        result = self._step_callback(evaluator, label)
        if isinstance(result, bool) or not isinstance(result, int) or result != 1:
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "step_once must confirm exactly one simulator step"
            )
        return 1

    def _frozen_base_evidence(
        self,
        *,
        evaluator: Any,
        phase: str,
    ) -> B1FrozenBasePoseEvidence:
        self._require_evaluator(evaluator)
        guard = self._required_guard()
        guard.assert_base_pose()
        token = guard.token.frozen_base_pose_token
        position, quaternion = self._starter_for(
            evaluator
        ).robot.get_position_orientation(
            frame="world",
            clone=True,
        )
        evidence = B1FrozenBasePoseEvidence(
            phase=phase,
            frozen_base_pose_token=token.pose_sha256,
            position=_vector(position, 3, "base position"),
            quaternion_xyzw=_vector(
                quaternion,
                4,
                "base quaternion",
            ),
        )
        guard.assert_base_pose()
        return evidence

    def _require_evaluator(self, evaluator: Any) -> None:
        if self._evaluator is None:
            self._evaluator = evaluator
        elif evaluator is not self._evaluator:
            self._poisoned = True
            raise R1ProB1SingleLoadBindingError(
                "binding cannot cross evaluator or scene instances"
            )

    def _require_approach(
        self,
        evaluator: Any,
        approach: B1ObservationOnlyApproach,
    ) -> None:
        self._require_operational()
        self._require_evaluator(evaluator)
        if approach is not self._approach or self._approach_token is None:
            raise R1ProB1SingleLoadBindingError(
                "candidate requires this binding's completed approach"
            )

    def _require_proposal(
        self,
        evaluator: Any,
        approach: B1ObservationOnlyApproach,
        proposal: LiveSafeCarryProposal,
    ) -> None:
        self._require_approach(evaluator, approach)
        if proposal is not self._proposal or self._guard is None:
            raise R1ProB1SingleLoadBindingError(
                "calibration requires this binding's exact proposal"
            )

    def _required_token(self) -> R1ProApproachStateToken:
        if self._approach_token is None:
            raise R1ProB1SingleLoadBindingError(
                "approach state token is unavailable"
            )
        return self._approach_token

    def _required_guard(self) -> R1ProNoBaseMotionGuard:
        if self._guard is None:
            raise R1ProB1SingleLoadBindingError(
                "base-motion guard is unavailable"
            )
        return self._guard

    def _require_operational(self) -> None:
        if self._poisoned:
            raise R1ProB1SingleLoadBindingError("binding is poisoned")


def _token_from_live_state(
    state: LiveRobotCarryState,
) -> R1ProApproachStateToken:
    if not isinstance(state, LiveRobotCarryState):
        raise TypeError("state must be LiveRobotCarryState")
    return R1ProApproachStateToken(
        candidate_source_state_token=R1ProCandidateSourceStateToken(
            hand=state.hand,
            live_robot_state_token=state.state_token,
        ),
        frozen_base_pose_token=R1ProFrozenBasePoseToken(
            hand=state.hand,
            base_world_position=state.body_world_position,
            base_world_quaternion_xyzw=(
                state.body_world_quaternion_xyzw
            ),
        ),
    )


def _capture_starter_neutral_robot_state(
    *,
    evaluator: Any,
    hand: str,
) -> Mapping[str, Any]:
    """Read exact robot state around Starter construction without stepping."""

    robot = getattr(evaluator, "robot", None)
    if robot is None:
        raise R1ProB1SingleLoadBindingError(
            "evaluator robot is unavailable"
        )
    try:
        body_position, body_quaternion = robot.get_position_orientation(
            frame="world",
            clone=True,
        )
        eef_position, eef_quaternion = robot.get_eef_pose(arm=hand)
        joints = robot.get_joint_positions()
        held = getattr(robot, "_ag_obj_in_hand")
    except Exception as exc:
        raise R1ProB1SingleLoadBindingError(
            "Starter neutrality state read failed"
        ) from exc
    if not isinstance(held, Mapping):
        raise R1ProB1SingleLoadBindingError(
            "assisted-grasp state is unavailable"
        )
    return {
        "body_position": _vector(body_position, 3, "body_position"),
        "body_quaternion": _vector(
            body_quaternion,
            4,
            "body_quaternion",
        ),
        "eef_position": _vector(eef_position, 3, "eef_position"),
        "eef_quaternion": _vector(
            eef_quaternion,
            4,
            "eef_quaternion",
        ),
        "joint_positions": tuple(
            float(item)
            for item in np.asarray(
                joints,
                dtype=np.float64,
            ).reshape(-1)
        ),
        "held_none": all(value is None for value in held.values()),
    }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _token_from_approach(
    *,
    approach: B1ObservationOnlyApproach,
    hand: str,
) -> R1ProApproachStateToken:
    if not isinstance(approach, B1ObservationOnlyApproach):
        raise TypeError("approach must be B1ObservationOnlyApproach")
    token = R1ProApproachStateToken(
        candidate_source_state_token=R1ProCandidateSourceStateToken(
            hand=hand,
            live_robot_state_token=(
                approach.candidate_source_state_token
            ),
        ),
        frozen_base_pose_token=R1ProFrozenBasePoseToken(
            hand=hand,
            base_world_position=approach.frozen_base.position,
            base_world_quaternion_xyzw=(
                approach.frozen_base.quaternion_xyzw
            ),
        ),
    )
    if (
        approach.frozen_base.frozen_base_pose_token
        != token.frozen_base_pose_token.pose_sha256
    ):
        raise R1ProB1SingleLoadBindingError(
            "approach frozen-base token differs from its numeric pose"
        )
    return token


def _same_quaternion(
    first: Any,
    second: Any,
    *,
    tolerance_rad: float,
) -> bool:
    lhs = np.asarray(first, dtype=np.float64)
    rhs = np.asarray(second, dtype=np.float64)
    if (
        lhs.shape != (4,)
        or rhs.shape != (4,)
        or not np.isfinite(lhs).all()
        or not np.isfinite(rhs).all()
    ):
        return False
    # A token is captured from the same live read and the state-token digest
    # already attests exact quaternion bytes.  Returning early here avoids a
    # spurious non-zero angle from normalizing an identical non-axis-aligned
    # quaternion and then passing it through acos at a 1e-12-rad threshold.
    if np.array_equal(lhs, rhs):
        return True
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= 1e-8 or rhs_norm <= 1e-8:
        return False
    cosine = float(
        np.clip(
            abs(np.dot(lhs / lhs_norm, rhs / rhs_norm)),
            -1.0,
            1.0,
        )
    )
    angle = 2.0 * math.acos(cosine)
    return angle <= tolerance_rad


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}D vector")
    return tuple(float(item) for item in array)


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result
