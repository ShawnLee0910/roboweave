"""Two-stage observation-only approach and physical grasp/lift binding.

The approach stage may move the physical base and torso, but consumes only
public snapshots and public action results.  It returns a frozen RGB-D target
plus two deliberately different attestations: a full robot-state token used
only as candidate-proposal provenance, and a base-only token that remains
valid while empty-wrist calibration and grasp move the arm.

The physical stage first verifies that same token, then permits only direct
23D Starter actions.  It executes reach, close, contact-authorized assisted
grasp, and lift; it never executes the driver's safe-retract phase.  Private
contact / held / lift evidence is written to audit and cannot enter the
already-terminated TokenPlan prompt loop.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.pixel_manipulation import (
    robot_base_point_to_world,
    validate_local_manipulation_target,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_public_wrist_evidence import (
    BootstrapPublicGripperWristEvidenceProvider,
    BootstrapPublicWristCalibrationRecord,
    safe_carry_candidate_sha256,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_safe_carry_calibration_actuator import (
    build_r1pro_safe_carry_calibration_actuator,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_safe_carry_stage_kinematics import (
    ExactG3BCompletionCalibrationActuator,
    R1ProCarryKinematicsLimits,
    R1ProSafeCarryStageCommandBuilder,
    build_candidate_r1pro_calibration_compensator,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_starter_grasp_driver import (
    PrivateGraspEvidence,
    R1ProStarterGraspDriver,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    B1CalibrationGraspLift,
    B1FrozenBasePoseEvidence,
    B1ObservationOnlyApproach,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_live_safe_carry_candidate import (
    LiveSafeCarryProposal,
)
from behavior_hl.embodied.policy.approach import (
    APPROACH_BASE_ACTIONS,
    APPROACH_LOOK_ACTIONS,
    ApproachBudgetExceeded,
    ApproachLedger,
    ApproachLimits,
    ApproachStopped,
)
from behavior_hl.embodied.policy.approach_execution import (
    BaseMacroConfig,
    BaseMacroProgress,
    classify_macro_displacement,
    look_translate_for_action,
)
from behavior_hl.embodied.policy.feedback import (
    PublicActionResult,
    PublicStatus,
    movement_result,
)
from behavior_hl.embodied.policy.profiles import (
    build_picking_up_toys_registry,
)
from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    PointActionArgs,
)
from behavior_hl.embodied.policy.r1pro.command_policy import (
    torso_profile_from_translate,
)
from behavior_hl.embodied.policy.rgbd_grounding import (
    ground_image_point,
)
from behavior_hl.embodied.runtime.plugins.grasp import (
    GraspMotionPhase,
)


R1PRO_OBSERVATION_ONLY_GRASP_LIFT_VERSION = (
    "behavior2026.r1pro-observation-only-grasp-lift/v1"
)
_FORBIDDEN_PUBLIC_FRAGMENTS = (
    "bddl",
    "contact",
    "goal",
    "held",
    "object_scope",
    "segmentation",
    "success",
    "task_state",
)


class ObservationOnlyGraspLiftError(RuntimeError):
    """Fail-closed approach, token, grasp, or calibration binding error."""


class PublicGroundingRejected(ObservationOnlyGraspLiftError):
    def __init__(self, status: PublicStatus, message: str) -> None:
        if status not in {
            PublicStatus.INVALID_DEPTH,
            PublicStatus.OUT_OF_REACH,
        }:
            raise ValueError("grounding rejection status is invalid")
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class PublicApproachSnapshot:
    snapshot_path: Path
    snapshot: Mapping[str, Any] = field(repr=False)
    source_head_rgb_sha256: str
    robot_position: tuple[float, ...] = field(repr=False)
    robot_orientation_xyzw: tuple[float, ...] = field(repr=False)
    simulator_steps: int = 1

    def __post_init__(self) -> None:
        path = Path(self.snapshot_path)
        if not path.is_file():
            raise ValueError("public snapshot path does not exist")
        object.__setattr__(self, "snapshot_path", path)
        _validate_public_mapping(self.snapshot)
        _sha256(self.source_head_rgb_sha256, "head RGB")
        object.__setattr__(
            self,
            "robot_position",
            _vector(self.robot_position, 3, "robot_position"),
        )
        quaternion = _vector(
            self.robot_orientation_xyzw,
            4,
            "robot_orientation_xyzw",
        )
        if np.linalg.norm(quaternion) <= 1e-8:
            raise ValueError("robot orientation has zero norm")
        object.__setattr__(
            self,
            "robot_orientation_xyzw",
            quaternion,
        )
        if self.simulator_steps != 1:
            raise ValueError(
                "each public snapshot capture must advance exactly one step"
            )


@dataclass(frozen=True, slots=True)
class FrozenPublicGraspTarget:
    target_token: str
    camera: str
    hand: str
    normalized_point: tuple[float, float]
    robot_base_xyz_m: tuple[float, ...]
    world_point_m: tuple[float, ...] = field(repr=False)
    horizontal_reach_m: float
    source_head_rgb_sha256: str
    candidate_source_state_token: str
    frozen_base: B1FrozenBasePoseEvidence
    target_from_public_rgbd: bool = True

    def __post_init__(self) -> None:
        _sha256(self.target_token, "target")
        if self.camera != "head" or self.hand != "right":
            raise ValueError("frozen approach target must be head/right")
        point = _vector(self.normalized_point, 2, "normalized_point")
        if any(item < 0.0 or item > 1.0 for item in point):
            raise ValueError("normalized point must be in [0, 1]")
        object.__setattr__(self, "normalized_point", point)
        object.__setattr__(
            self,
            "robot_base_xyz_m",
            _vector(self.robot_base_xyz_m, 3, "robot_base_xyz_m"),
        )
        object.__setattr__(
            self,
            "world_point_m",
            _vector(self.world_point_m, 3, "world_point_m"),
        )
        reach = _positive(self.horizontal_reach_m, "horizontal_reach_m")
        object.__setattr__(self, "horizontal_reach_m", reach)
        _sha256(self.source_head_rgb_sha256, "head RGB")
        if (
            not isinstance(self.candidate_source_state_token, str)
            or not self.candidate_source_state_token
        ):
            raise ValueError(
                "candidate_source_state_token must be non-empty"
            )
        if (
            not isinstance(self.frozen_base, B1FrozenBasePoseEvidence)
            or self.frozen_base.phase != "final_approach"
        ):
            raise TypeError(
                "target requires final_approach frozen-base evidence"
            )
        if self.target_from_public_rgbd is not True:
            raise ValueError("target must originate from public RGB-D")


@dataclass(frozen=True, slots=True)
class ObservationOnlyApproachResult:
    target: FrozenPublicGraspTarget
    ledger: ApproachLedger = field(repr=False)
    pending_grasp: ActionEnvelope = field(repr=False)
    model_loop_terminated: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.target, FrozenPublicGraspTarget):
            raise TypeError("target must be FrozenPublicGraspTarget")
        if not isinstance(self.ledger, ApproachLedger):
            raise TypeError("ledger must be ApproachLedger")
        if (
            not isinstance(self.pending_grasp, ActionEnvelope)
            or self.pending_grasp.action != "grasp_at"
        ):
            raise TypeError("pending_grasp must be grasp_at")
        if self.model_loop_terminated is not True:
            raise ValueError("model loop must terminate at a valid target")

    @property
    def runner_evidence(self) -> B1ObservationOnlyApproach:
        """Return the public typed boundary consumed by the B1 runner."""

        return B1ObservationOnlyApproach(
            target_token=self.target.target_token,
            public_frame_sha256=(
                self.target.source_head_rgb_sha256
            ),
            candidate_source_state_token=(
                self.target.candidate_source_state_token
            ),
            frozen_base=self.target.frozen_base,
            target_from_public_rgbd=True,
            physical_grasp_executions=0,
            private_truth_exposed_to_policy=False,
        )


@dataclass(frozen=True, slots=True)
class PublicRGBDGroundingConfig:
    intrinsic_matrix: np.ndarray = field(repr=False)
    depth_radius_px: int = 1
    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 5.0
    maximum_depth_spread_m: float = 0.10
    maximum_horizontal_reach_m: float = 1.30
    minimum_target_height_m: float = 0.20
    maximum_target_height_m: float = 1.50

    def __post_init__(self) -> None:
        intrinsic = np.asarray(self.intrinsic_matrix, dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("intrinsic_matrix must be finite 3x3")
        copy = np.array(intrinsic, copy=True)
        copy.setflags(write=False)
        object.__setattr__(self, "intrinsic_matrix", copy)
        if (
            isinstance(self.depth_radius_px, bool)
            or not isinstance(self.depth_radius_px, int)
            or self.depth_radius_px < 0
        ):
            raise ValueError("depth_radius_px must be non-negative")
        for name in (
            "minimum_depth_m",
            "maximum_depth_m",
            "maximum_depth_spread_m",
            "maximum_horizontal_reach_m",
            "minimum_target_height_m",
            "maximum_target_height_m",
        ):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), name),
            )


@dataclass(frozen=True, slots=True)
class PublicApproachStep:
    base_qvel: tuple[float, ...]
    head_forward_clearance_m: float
    torso_qpos: tuple[float, ...]
    base_contact: bool = False
    simulator_steps: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_qvel",
            _vector(self.base_qvel, 3, "base_qvel"),
        )
        object.__setattr__(
            self,
            "torso_qpos",
            _vector(self.torso_qpos, 4, "torso_qpos"),
        )
        clearance = _nonnegative(
            self.head_forward_clearance_m,
            "head_forward_clearance_m",
        )
        object.__setattr__(
            self,
            "head_forward_clearance_m",
            clearance,
        )
        if not isinstance(self.base_contact, bool):
            raise ValueError("base_contact must be bool")
        if self.simulator_steps != 1:
            raise ValueError("public approach step must be exactly one")


class SharedControlApproachMacroExecutor:
    """Concrete public-only base/look macros with per-tick slow-forward."""

    def __init__(
        self,
        *,
        control: Any,
        step: Callable[[str, bool], PublicApproachStep],
        base_config: BaseMacroConfig,
        command_frequency_hz: float,
        base_settle_steps: int,
        look_maximum_steps: int,
        look_settle_steps: int,
        look_tolerance_rad: float,
        look_max_delta_per_step: float,
    ) -> None:
        for method in ("command_base", "command_torso", "hold"):
            if not callable(getattr(control, method, None)):
                raise TypeError(f"control is missing {method}")
        if not callable(step):
            raise TypeError("step must be callable")
        if not isinstance(base_config, BaseMacroConfig):
            raise TypeError("base_config must be BaseMacroConfig")
        self._control = control
        self._step = step
        self.base_config = base_config
        self._frequency = _positive(
            command_frequency_hz,
            "command_frequency_hz",
        )
        self._base_settle = _nonnegative_int(
            base_settle_steps,
            "base_settle_steps",
        )
        self._look_maximum = _positive_int(
            look_maximum_steps,
            "look_maximum_steps",
        )
        self._look_settle = _nonnegative_int(
            look_settle_steps,
            "look_settle_steps",
        )
        self._look_tolerance = _positive(
            look_tolerance_rad,
            "look_tolerance_rad",
        )
        self._look_delta = _positive(
            look_max_delta_per_step,
            "look_max_delta_per_step",
        )

    def execute_base(self, action: str) -> PublicActionResult:
        progress = BaseMacroProgress(action, self.base_config)
        integrated = np.zeros(3, dtype=np.float64)
        status = None
        contact = False
        hard_stop = False
        active_steps = 0
        for _ in range(self.base_config.maximum_step_budget(action)):
            state = self._control.public_state()
            clearance = _nonnegative(
                state["head_forward_clearance_m"],
                "head_forward_clearance_m",
            )
            decision = self.base_config.step_decision(
                action,
                head_forward_clearance_m=clearance,
            )
            if decision.hard_stop:
                hard_stop = True
                status = PublicStatus.BLOCKED
                break
            self._control.command_base(decision.command)
            sample = self._require_step(
                f"b1.approach.{action}",
                True,
            )
            active_steps += 1
            integrated += np.asarray(sample.base_qvel) / self._frequency
            contact = sample.base_contact
            status = progress.observe(
                sample.base_qvel,
                head_forward_clearance_m=(
                    sample.head_forward_clearance_m
                ),
                contact_detected=contact,
            )
            if status is not None:
                break
        if status is None:
            status = PublicStatus.BLOCKED
        self._control.hold()
        for _ in range(self._base_settle):
            self._require_step(f"b1.approach.{action}.settle", True)
        classified = classify_macro_displacement(
            action,
            forward_m=float(integrated[0]),
            left_m=float(integrated[1]),
            yaw_deg=float(np.degrees(integrated[2])),
            config=self.base_config,
            contact_detected=contact,
            clearance_hard_stop=hard_stop,
            stall_detected=(
                status is PublicStatus.BLOCKED
                and not contact
                and not hard_stop
            ),
        )
        return movement_result(
            status=classified,
            simulator_steps=active_steps + self._base_settle,
            forward_m=float(integrated[0]),
            left_m=float(integrated[1]),
            yaw_deg=float(np.degrees(integrated[2])),
        )

    def execute_look(self, action: str) -> PublicActionResult:
        target = torso_profile_from_translate(
            look_translate_for_action(action)
        )
        converged = False
        active_steps = 0
        for _ in range(self._look_maximum):
            self._control.command_torso(
                target,
                max_delta_per_step=self._look_delta,
            )
            sample = self._require_step(
                f"b1.approach.{action}",
                True,
            )
            active_steps += 1
            if np.max(np.abs(np.asarray(sample.torso_qpos) - target)) <= (
                self._look_tolerance
            ):
                converged = True
                break
        self._control.hold()
        for _ in range(self._look_settle):
            self._require_step(f"b1.approach.{action}.settle", True)
        return PublicActionResult(
            status=(
                PublicStatus.COMPLETED
                if converged
                else PublicStatus.JOINT_STALLED
            ),
            simulator_steps=active_steps + self._look_settle,
        )

    def _require_step(
        self,
        label: str,
        allow_base_motion: bool,
    ) -> PublicApproachStep:
        sample = self._step(label, allow_base_motion)
        if not isinstance(sample, PublicApproachStep):
            raise ObservationOnlyGraspLiftError(
                "approach step returned invalid public sample"
            )
        return sample


@dataclass(frozen=True, slots=True)
class PhysicalGraspLiftConfig:
    max_reach_ticks: int
    max_close_ticks: int
    max_lift_ticks: int
    minimum_lift_m: float
    g3b_samples_per_stage: int
    inactive_gripper_command: float

    def __post_init__(self) -> None:
        for name in (
            "max_reach_ticks",
            "max_close_ticks",
            "max_lift_ticks",
            "g3b_samples_per_stage",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        lift = _positive(self.minimum_lift_m, "minimum_lift_m")
        if lift < 0.05:
            raise ValueError("minimum_lift_m must be at least 0.05")
        object.__setattr__(self, "minimum_lift_m", lift)
        gripper = _finite(
            self.inactive_gripper_command,
            "inactive_gripper_command",
        )
        if abs(gripper) > 1.0:
            raise ValueError(
                "inactive_gripper_command exceeds normalized limits"
            )
        object.__setattr__(
            self,
            "inactive_gripper_command",
            gripper,
        )


def ground_frozen_public_target(
    *,
    frame: PublicApproachSnapshot,
    envelope: ActionEnvelope,
    config: PublicRGBDGroundingConfig,
    final_state: Any,
) -> FrozenPublicGraspTarget:
    """Ground one strict head/right point without object or task truth."""

    candidate, reach, world = _ground_public_geometry(
        frame=frame,
        envelope=envelope,
        config=config,
    )
    candidate_source_token, frozen_base = _final_state_evidence(
        final_state=final_state,
        frame=frame,
    )
    target_token = _target_token(
        frame_sha256=frame.source_head_rgb_sha256,
        normalized_point=candidate.normalized_point,
        robot_base_xyz_m=candidate.robot_base_xyz_m,
        candidate_source_state_token=candidate_source_token,
        frozen_base_pose_token=frozen_base.frozen_base_pose_token,
    )
    return FrozenPublicGraspTarget(
        target_token=target_token,
        camera=candidate.camera,
        hand=envelope.args.hand,
        normalized_point=tuple(candidate.normalized_point),
        robot_base_xyz_m=tuple(candidate.robot_base_xyz_m),
        world_point_m=tuple(float(item) for item in world),
        horizontal_reach_m=float(reach["horizontal_reach_m"]),
        source_head_rgb_sha256=frame.source_head_rgb_sha256,
        candidate_source_state_token=candidate_source_token,
        frozen_base=frozen_base,
    )


def _ground_public_geometry(
    *,
    frame: PublicApproachSnapshot,
    envelope: ActionEnvelope,
    config: PublicRGBDGroundingConfig,
) -> tuple[Any, Mapping[str, Any], np.ndarray]:
    if (
        not isinstance(envelope.args, PointActionArgs)
        or envelope.action != "grasp_at"
    ):
        raise TypeError("grounding requires a parsed grasp_at action")
    try:
        candidate = ground_image_point(
            frame.snapshot,
            camera=envelope.args.camera,
            point=envelope.args.point,
            intrinsic_matrix=config.intrinsic_matrix,
            depth_radius_px=config.depth_radius_px,
            minimum_depth_m=config.minimum_depth_m,
            maximum_depth_m=config.maximum_depth_m,
            maximum_depth_spread_m=config.maximum_depth_spread_m,
        )
    except ValueError as exc:
        raise PublicGroundingRejected(
            PublicStatus.INVALID_DEPTH,
            str(exc),
        ) from exc
    try:
        reach = validate_local_manipulation_target(
            candidate.robot_base_xyz_m,
            maximum_horizontal_reach_m=(
                config.maximum_horizontal_reach_m
            ),
            minimum_height_m=config.minimum_target_height_m,
            maximum_height_m=config.maximum_target_height_m,
        )
    except ValueError as exc:
        raise PublicGroundingRejected(
            PublicStatus.OUT_OF_REACH,
            str(exc),
        ) from exc
    world = robot_base_point_to_world(
        candidate.robot_base_xyz_m,
        robot_position=frame.robot_position,
        robot_orientation_xyzw=frame.robot_orientation_xyzw,
    )
    return candidate, reach, np.asarray(world, dtype=np.float64)


def run_observation_only_approach(
    *,
    run_dir: Path,
    capture_public_snapshot: Callable[[], PublicApproachSnapshot],
    capture_final_state: Callable[[], Any],
    macro_executor: SharedControlApproachMacroExecutor,
    grounding_config: PublicRGBDGroundingConfig,
    limits: ApproachLimits,
    tokenplan_wall_timeout_seconds: int,
    persist_public_result: Callable[[Any, str], None] | None = None,
    tokenplan_call: Callable[..., tuple[Any, Mapping[str, Any], str]]
    | None = None,
) -> ObservationOnlyApproachResult:
    """Run exact Qwen approach and terminate immediately on valid RGB-D target."""

    if not callable(capture_public_snapshot):
        raise TypeError("capture_public_snapshot must be callable")
    if not callable(capture_final_state):
        raise TypeError("capture_final_state must be callable")
    if not isinstance(
        macro_executor,
        SharedControlApproachMacroExecutor,
    ):
        raise TypeError("macro_executor has the wrong type")
    if not isinstance(grounding_config, PublicRGBDGroundingConfig):
        raise TypeError("grounding_config has the wrong type")
    if not isinstance(limits, ApproachLimits):
        raise TypeError("limits must be ApproachLimits")
    if limits.max_grasp_queries != 8:
        raise ValueError("single-load approach requires exactly 8 reach queries")
    if persist_public_result is not None and not callable(
        persist_public_result
    ):
        raise TypeError("persist_public_result must be callable")
    call_model = tokenplan_call or _default_tokenplan_call
    if not callable(call_model):
        raise TypeError("tokenplan_call must be callable")

    registry = build_picking_up_toys_registry()
    ledger = ApproachLedger(limits)
    # Match ESI-Bench's temporal input contract: retain the five most recent
    # pre-action public snapshots and send them oldest-to-newest before the
    # current view.  The current frame is remembered only after a policy
    # action has been admitted, so protocol repairs reuse the identical image
    # sequence without inventing another observation.
    history_snapshot_paths: list[Path] = []

    def remember_current(path: Path) -> None:
        history_snapshot_paths.append(Path(path))
        del history_snapshot_paths[:-5]

    while True:
        if ledger.grasp_query_budget_exhausted:
            raise ApproachBudgetExceeded(
                "grasp_at reach-query budget exhausted before model call"
            )
        frame = capture_public_snapshot()
        if not isinstance(frame, PublicApproachSnapshot):
            raise ObservationOnlyGraspLiftError(
                "capture callback returned an invalid public snapshot"
            )
        ledger.note_simulator_steps(frame.simulator_steps)
        envelope = None
        frame_hash = ""
        while envelope is None:
            envelope, _, frame_hash = call_model(
                run_dir=Path(run_dir),
                snapshot_path=frame.snapshot_path,
                history_snapshot_paths=tuple(history_snapshot_paths),
                registry=registry,
                ledger=ledger,
                wall_timeout_seconds=tokenplan_wall_timeout_seconds,
            )
        if frame_hash != frame.source_head_rgb_sha256:
            raise ObservationOnlyGraspLiftError(
                "TokenPlan decision frame differs from frozen public target"
            )
        immediate = ledger.begin_action(envelope)
        if immediate is not None:
            _persist(
                persist_public_result,
                ledger.history[-1],
                frame_hash,
            )
            remember_current(frame.snapshot_path)
            continue
        if envelope.action in APPROACH_BASE_ACTIONS:
            item = ledger.finish_action(
                macro_executor.execute_base(envelope.action)
            )
            _persist(persist_public_result, item, frame_hash)
            remember_current(frame.snapshot_path)
            continue
        if envelope.action in APPROACH_LOOK_ACTIONS:
            item = ledger.finish_action(
                macro_executor.execute_look(envelope.action)
            )
            _persist(persist_public_result, item, frame_hash)
            remember_current(frame.snapshot_path)
            continue
        if envelope.action == "stop":
            item = ledger.finish_action(
                PublicActionResult(
                    status=PublicStatus.COMPLETED,
                    simulator_steps=0,
                )
            )
            _persist(persist_public_result, item, frame_hash)
            raise ApproachStopped("Qwen stopped before physical grasp")
        if envelope.action != "grasp_at":
            raise ObservationOnlyGraspLiftError(
                "approach dispatched an unknown action"
            )
        try:
            # Validate public RGB-D reachability before capturing the sole
            # full-state proposal token.  Invalid grasp queries therefore do
            # not consume or multiply the final-state attestation.
            _ground_public_geometry(
                frame=frame,
                envelope=envelope,
                config=grounding_config,
            )
        except PublicGroundingRejected as exc:
            item = ledger.finish_action(
                PublicActionResult(
                    status=exc.status,
                    simulator_steps=0,
                )
            )
            _persist(persist_public_result, item, frame_hash)
            if ledger.grasp_query_budget_exhausted:
                raise ApproachBudgetExceeded(
                    "grasp_at reach-query budget exhausted immediately"
                ) from exc
            remember_current(frame.snapshot_path)
            continue
        target = ground_frozen_public_target(
            frame=frame,
            envelope=envelope,
            config=grounding_config,
            final_state=capture_final_state(),
        )
        # Do not call the model again.  The grasp action remains pending until
        # the post-candidate physical stage executes exactly one grasp.
        return ObservationOnlyApproachResult(
            target=target,
            ledger=ledger,
            pending_grasp=envelope,
        )


def run_physical_grasp_lift(
    *,
    approach: ObservationOnlyApproachResult,
    proposal: LiveSafeCarryProposal,
    bootstrap: BootstrapPublicWristCalibrationRecord,
    starter: Any,
    control: Any,
    current_base_pose_token: Callable[[], str],
    capture_frozen_base_pose: Callable[
        [str], B1FrozenBasePoseEvidence
    ],
    one_step: Callable[[str], int],
    live_public_snapshot_provider: Callable[[], Mapping[str, Any]],
    private_contact_provider: Callable[[str], bool],
    private_assisted_provider: Callable[[str], bool],
    private_held_provider: Callable[[str], Any],
    private_lift_provider: Callable[[str], float],
    private_audit_writer: Callable[[Mapping[str, Any]], None],
    public_joint_state: Callable[[], Mapping[str, Any]],
    calibration_live_sample_provider: Callable[..., Any],
    config: PhysicalGraspLiftConfig,
    kinematics_limits: R1ProCarryKinematicsLimits | None = None,
) -> B1CalibrationGraspLift:
    """Execute no-base reach/close/lift, then return a real calibration actuator."""

    if not isinstance(approach, ObservationOnlyApproachResult):
        raise TypeError("approach has the wrong type")
    if not isinstance(proposal, LiveSafeCarryProposal):
        raise TypeError("proposal has the wrong type")
    if not isinstance(bootstrap, BootstrapPublicWristCalibrationRecord):
        raise TypeError("bootstrap has the wrong type")
    if not isinstance(config, PhysicalGraspLiftConfig):
        raise TypeError("config has the wrong type")
    for name, callback in (
        ("current_base_pose_token", current_base_pose_token),
        ("capture_frozen_base_pose", capture_frozen_base_pose),
        ("one_step", one_step),
        ("live_public_snapshot_provider", live_public_snapshot_provider),
        ("private_contact_provider", private_contact_provider),
        ("private_assisted_provider", private_assisted_provider),
        ("private_held_provider", private_held_provider),
        ("private_lift_provider", private_lift_provider),
        ("private_audit_writer", private_audit_writer),
        ("public_joint_state", public_joint_state),
        (
            "calibration_live_sample_provider",
            calibration_live_sample_provider,
        ),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")
    for method in ("command_direct_action", "command_carry_action", "hold"):
        if not callable(getattr(control, method, None)):
            raise TypeError(f"control is missing {method}")
    target = approach.target
    _require_same_base_token(current_base_pose_token, target)
    if (
        proposal.candidate.hand != target.hand
        or bootstrap.hand != target.hand
        or proposal.report.state_token
        != target.candidate_source_state_token
        or bootstrap.safe_carry_candidate_sha256
        != safe_carry_candidate_sha256(proposal.candidate)
    ):
        raise ObservationOnlyGraspLiftError(
            "proposal / bootstrap hand or candidate changed"
        )

    wrist = BootstrapPublicGripperWristEvidenceProvider(
        snapshot_provider=live_public_snapshot_provider,
        calibrations={target.hand: bootstrap},
    )
    physical_steps = 0

    def step_direct(action: np.ndarray, label: str) -> int:
        nonlocal physical_steps
        _require_same_base_token(current_base_pose_token, target)
        array = np.asarray(action, dtype=np.float64)
        if array.shape != (23,) or not np.isfinite(array).all():
            raise ObservationOnlyGraspLiftError(
                "Starter yielded invalid physical action"
            )
        control.command_direct_action(array)
        result = one_step(label)
        if isinstance(result, bool) or result != 1:
            raise ObservationOnlyGraspLiftError(
                "physical grasp callback must advance exactly one step"
            )
        physical_steps += 1
        _require_same_base_token(current_base_pose_token, target)
        return 1

    def hold_one_step(hand: str) -> int:
        nonlocal physical_steps
        del hand
        control.hold()
        result = one_step("grasp.safe_hold")
        if isinstance(result, bool) or result != 1:
            raise ObservationOnlyGraspLiftError(
                "physical safe hold must advance exactly one step"
            )
        physical_steps += 1
        _require_same_base_token(current_base_pose_token, target)
        return 1

    def private_evidence(
        hand: str,
        phase: GraspMotionPhase,
    ) -> PrivateGraspEvidence:
        contact = private_contact_provider(hand)
        assisted = private_assisted_provider(hand)
        held_token = private_held_provider(hand)
        lift = (
            private_lift_provider(hand)
            if phase is GraspMotionPhase.LIFT
            else 0.0
        )
        private_audit_writer(
            {
                "event": "physical_grasp_sample",
                "phase": phase.value,
                "qualified_contact": bool(contact),
                "assisted_grasp_active": bool(assisted),
                "held_present": held_token is not None,
                "payload_lift_m": float(lift),
            }
        )
        return PrivateGraspEvidence(
            qualified_contact=contact,
            assisted_grasp_active=assisted,
            held_token=held_token,
            payload_lift_m=lift,
        )

    driver = R1ProStarterGraspDriver(
        starter=starter,
        command_step=step_direct,
        public_evidence_provider=wrist,
        private_audit_provider=private_evidence,
        safe_hold=hold_one_step,
        safe_carry_pose_provider=lambda **kwargs: (
            proposal.candidate.eef_world_position,
            proposal.candidate.eef_world_quaternion_xyzw,
        ),
    )
    plan = driver.preflight_ik(
        world_point=target.world_point_m,
        hand=target.hand,
    )
    if plan is None:
        wrist.cancel_grasp(target.hand)
        raise ObservationOnlyGraspLiftError(
            "official grasp IK rejected frozen public target"
        )
    approach.ledger.note_grasp_execution()
    observed_lift = 0.0
    assisted_authorized = False
    try:
        for phase, budget in (
            (GraspMotionPhase.REACH, config.max_reach_ticks),
            (GraspMotionPhase.CLOSE, config.max_close_ticks),
            (GraspMotionPhase.LIFT, config.max_lift_ticks),
        ):
            for _ in range(budget):
                sample = driver.tick_motion(
                    phase=phase,
                    plan_token=plan,
                    hand=target.hand,
                )
                if phase is GraspMotionPhase.CLOSE and sample.contact_event:
                    assisted_authorized = driver.enable_assisted_grasp(
                        plan_token=plan,
                        hand=target.hand,
                    )
                    if assisted_authorized is not True:
                        raise ObservationOnlyGraspLiftError(
                            "contact did not authorize assisted grasp"
                        )
                if phase is GraspMotionPhase.LIFT:
                    observed_lift = max(
                        observed_lift,
                        float(sample.payload_lift_m),
                    )
                if sample.completed:
                    break
            else:
                raise ObservationOnlyGraspLiftError(
                    f"{phase.value} exhausted its physical tick budget"
                )
            if phase is GraspMotionPhase.CLOSE and not assisted_authorized:
                raise ObservationOnlyGraspLiftError(
                    "physical close completed without authorized contact"
                )
        if observed_lift < config.minimum_lift_m:
            raise ObservationOnlyGraspLiftError(
                "physical payload lift is below 0.05 m"
            )
        positive_window = wrist.positive_window(target.hand)
        bind_public_proxy = getattr(
            calibration_live_sample_provider,
            "bind_public_grasp_proxy",
            None,
        )
        if callable(bind_public_proxy):
            bind_public_proxy(
                lambda: bool(
                    wrist.sample(target.hand).grasp_proxy_likely
                )
            )
    except Exception:
        wrist.cancel_grasp(target.hand)
        driver.hold_safe(hand=target.hand)
        raise

    item = approach.ledger.finish_action(
        PublicActionResult(
            status=PublicStatus.GRASP_UNCERTAIN,
            simulator_steps=physical_steps,
            hand=target.hand,
        )
    )
    del item

    stage_builder = R1ProSafeCarryStageCommandBuilder(
        hand=target.hand,
        public_joint_state=public_joint_state,
        samples_per_stage=config.g3b_samples_per_stage,
        inactive_gripper_command=config.inactive_gripper_command,
        kinematics_limits=kinematics_limits,
    )
    compensator = build_candidate_r1pro_calibration_compensator(
        starter=starter,
        candidate=proposal.candidate,
        kinematics_limits=kinematics_limits,
    )
    actuator = build_r1pro_safe_carry_calibration_actuator(
        starter=starter,
        control=control,
        evaluator_step=one_step,
        stage_command_builder=stage_builder,
        active_arm_compensator=compensator,
        live_sample_provider=calibration_live_sample_provider,
    )
    completing_actuator = ExactG3BCompletionCalibrationActuator(
        actuator=actuator,
        stage_builder=stage_builder,
        samples_per_stage=config.g3b_samples_per_stage,
    )
    frozen_base = capture_frozen_base_pose("after_grasp_lift")
    if (
        not isinstance(frozen_base, B1FrozenBasePoseEvidence)
        or frozen_base.phase != "after_grasp_lift"
    ):
        raise ObservationOnlyGraspLiftError(
            "physical grasp returned invalid frozen-base evidence"
        )
    if (
        frozen_base.frozen_base_pose_token
        != target.frozen_base.frozen_base_pose_token
    ):
        raise ObservationOnlyGraspLiftError(
            "base pose token changed during physical grasp"
        )
    return B1CalibrationGraspLift(
        positive_window=positive_window,
        safe_carry_actuator=completing_actuator,
        consumed_target_token=target.target_token,
        consumed_candidate_source_state_token=(
            target.candidate_source_state_token
        ),
        frozen_base=frozen_base,
        physical_grasp_executions=1,
        observed_lift_m=observed_lift,
        base_motion_commanded=False,
        private_truth_exposed_to_policy=False,
    )


def _default_tokenplan_call(**kwargs: Any) -> Any:
    # Historical runner code remains untouched; only its strict file-boundary
    # helper is reused.
    from behavior_hl.embodied.evaluator_bridge.run_g4a_grasp_suite import (
        _run_tokenplan_approach_call,
    )

    return _run_tokenplan_approach_call(**kwargs)


def _persist(
    callback: Callable[[Any, str], None] | None,
    item: Any,
    frame_hash: str,
) -> None:
    if callback is not None:
        callback(item, frame_hash)


def _require_same_base_token(
    provider: Callable[[], str],
    target: FrozenPublicGraspTarget,
) -> None:
    value = provider()
    if value != target.frozen_base.frozen_base_pose_token:
        raise ObservationOnlyGraspLiftError(
            "base pose changed after public target was frozen"
        )


def _final_state_evidence(
    *,
    final_state: Any,
    frame: PublicApproachSnapshot,
) -> tuple[str, B1FrozenBasePoseEvidence]:
    """Convert the binding's typed two-scope token without importing it."""

    candidate_token = getattr(
        final_state,
        "live_robot_state_token",
        None,
    )
    base_scope = getattr(final_state, "frozen_base_pose_token", None)
    base_token = getattr(base_scope, "pose_sha256", None)
    position = getattr(final_state, "base_world_position", None)
    quaternion = getattr(
        final_state,
        "base_world_quaternion_xyzw",
        None,
    )
    if not isinstance(candidate_token, str) or not candidate_token:
        raise ObservationOnlyGraspLiftError(
            "final state lacks full robot-state candidate provenance"
        )
    _sha256(base_token, "frozen base pose")
    position_tuple = _vector(position, 3, "final base position")
    quaternion_tuple = _vector(
        quaternion,
        4,
        "final base quaternion",
    )
    if not np.allclose(
        position_tuple,
        frame.robot_position,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ObservationOnlyGraspLiftError(
            "public frame and final-state base positions differ"
        )
    lhs = np.asarray(quaternion_tuple, dtype=np.float64)
    rhs = np.asarray(frame.robot_orientation_xyzw, dtype=np.float64)
    lhs /= np.linalg.norm(lhs)
    rhs /= np.linalg.norm(rhs)
    if not np.allclose(
        lhs,
        rhs,
        rtol=0.0,
        atol=1e-8,
    ) and not np.allclose(
        lhs,
        -rhs,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ObservationOnlyGraspLiftError(
            "public frame and final-state base orientations differ"
        )
    return candidate_token, B1FrozenBasePoseEvidence(
        phase="final_approach",
        frozen_base_pose_token=base_token,
        position=position_tuple,
        quaternion_xyzw=quaternion_tuple,
    )


def _target_token(
    *,
    frame_sha256: str,
    normalized_point: Any,
    robot_base_xyz_m: Any,
    candidate_source_state_token: str,
    frozen_base_pose_token: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(R1PRO_OBSERVATION_ONLY_GRASP_LIFT_VERSION.encode())
    digest.update(frame_sha256.encode())
    digest.update(candidate_source_state_token.encode())
    digest.update(frozen_base_pose_token.encode())
    digest.update(
        np.ascontiguousarray(
            normalized_point,
            dtype=np.float64,
        ).tobytes()
    )
    digest.update(
        np.ascontiguousarray(
            robot_base_xyz_m,
            dtype=np.float64,
        ).tobytes()
    )
    return digest.hexdigest()


def _validate_public_mapping(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("public snapshot must be a mapping")
    stack = [value]
    while stack:
        current = stack.pop()
        for key, item in current.items():
            folded = str(key).casefold()
            if any(fragment in folded for fragment in _FORBIDDEN_PUBLIC_FRAGMENTS):
                raise ValueError(
                    f"public snapshot contains private field {key!r}"
                )
            if isinstance(item, Mapping):
                stack.append(item)
    if "proprio" not in value or "cameras" not in value:
        raise ValueError("public snapshot lacks proprio or cameras")


def _sha256(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} token must be lowercase SHA256")


def _vector(value: Any, size: int, field: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{field} must be a finite {size}-vector")
    return tuple(float(item) for item in array)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value
