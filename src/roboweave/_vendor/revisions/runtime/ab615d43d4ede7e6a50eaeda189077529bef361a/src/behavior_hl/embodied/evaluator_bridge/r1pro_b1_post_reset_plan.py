"""Concrete post-reset assembly for the first B1 single-leaf physical chain.

This module performs no scene load and has no OmniGibson import.  It binds a
passing same-scene safe-carry calibration and formal public wrist calibration
to the existing dispatcher, six component builders, official six-leaf
verifier, resident environment factories, and crash-resilient artifact sink.

The first pilot is deliberately *single-hand* according to the calibration
record.  The action registry exposes only that hand.  No fallback arm target
is invented for an uncalibrated hand.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.official_predicate_verifier import (
    OfficialPredicateVerifier,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders import (
    R1ProBaseComponentBuilder,
    R1ProCarryComponentBuilder,
    R1ProGraspComponentBuilder,
    R1ProPlaceComponentBuilder,
    R1ProReleaseComponentBuilder,
    R1ProTorsoComponentBuilder,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_grasp_binding import (
    RobotPose,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_place_services import (
    LocalRGBPatchAssessment,
    R1ProPlaceMotionLimits,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_public_wrist_evidence import (
    CalibratedPublicGripperWristEvidenceProvider,
    PublicWristCalibrationRecord,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_safe_carry_stage_kinematics import (
    R1ProCarryKinematicsLimits,
    build_validated_r1pro_carry_compensator,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    B1PostResetAssemblyPlan,
)
from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
    SafeCarryCalibrationRecord,
    provider_from_validated_calibration,
)
from behavior_hl.embodied.policy.approach_execution import BaseMacroConfig
from behavior_hl.embodied.policy.feedback import GraspProxyThresholds
from behavior_hl.embodied.policy.profiles import (
    PICKING_UP_TOYS_PROFILE,
    build_picking_up_toys_registry,
)
from behavior_hl.embodied.runtime.b1_dispatcher_factory import (
    BASE_ACTIONS,
    CarryBaseParameters,
)
from behavior_hl.embodied.runtime.episode import EpisodeSpec
from behavior_hl.embodied.runtime.plugins.carry import CarrySafetyLimits
from behavior_hl.embodied.runtime.plugins.grasp import PhysicalGraspConfig
from behavior_hl.embodied.runtime.plugins.local_base import (
    LocalBaseSafetyLimits,
)
from behavior_hl.embodied.runtime.plugins.place import PlaceConfig
from behavior_hl.embodied.runtime.plugins.release import (
    PhysicalReleaseConfig,
)
from behavior_hl.embodied.runtime.plugins.torso import EmptyTorsoLimits
from behavior_hl.embodied.runtime.public_carry_state import (
    PublicCarryStateTracker,
)
from behavior_hl.embodied.runtime.resident_artifact_adapter import (
    ResidentArtifactSinkAdapter,
)
from behavior_hl.embodied.runtime.resident_artifacts import (
    ResidentEpisodeArtifactWriter,
)
from behavior_hl.embodied.runtime.resident_episode import (
    ResidentEpisodeLimits,
)
from behavior_hl.embodied.runtime.public_single_leaf_witness import (
    PublicSingleLeafReplayModel,
    PublicSingleLeafWitness,
)


R1PRO_B1_POST_RESET_PLAN_VERSION = (
    "behavior2026.r1pro-b1-post-reset-single-leaf/v1"
)


class R1ProB1PostResetPlanError(RuntimeError):
    """Raised before a malformed post-reset plan can command the robot."""


@dataclass(frozen=True, slots=True)
class R1ProB1PostResetLiveDependencies:
    """Live, evaluator-owned callbacks with no policy-facing return path."""

    starter: Any
    camera_intrinsics: Mapping[str, Any]
    private_robot_pose_provider: Callable[[Any], RobotPose]
    private_contact_provider: Callable[[str], bool]
    private_assisted_provider: Callable[[str], bool]
    private_held_provider: Callable[[str], Any]
    private_lift_provider: Callable[[str], float]
    target_pose_builder: Callable[
        [tuple[float, float, float], str], tuple[Any, Any]
    ]
    safe_hold_action: Callable[[str, bool], Sequence[float]]
    public_stability: Callable[[], Mapping[str, Any]]
    private_safety: Callable[[], Mapping[str, Any]]
    capture_private_carry_endpoint: Callable[..., Mapping[str, Any]]
    command_frequency_hz: float = 30.0

    def __post_init__(self) -> None:
        if self.starter is None:
            raise ValueError("starter must be a live selected-arm Starter")
        intrinsics = dict(self.camera_intrinsics)
        if not intrinsics:
            raise ValueError("camera_intrinsics must not be empty")
        object.__setattr__(
            self,
            "camera_intrinsics",
            MappingProxyType(intrinsics),
        )
        for name in (
            "private_robot_pose_provider",
            "private_contact_provider",
            "private_assisted_provider",
            "private_held_provider",
            "private_lift_provider",
            "target_pose_builder",
            "safe_hold_action",
            "public_stability",
            "private_safety",
            "capture_private_carry_endpoint",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if (
            isinstance(self.command_frequency_hz, bool)
            or not isinstance(self.command_frequency_hz, (int, float))
            or float(self.command_frequency_hz) <= 0.0
        ):
            raise ValueError("command_frequency_hz must be positive")
        object.__setattr__(
            self,
            "command_frequency_hz",
            float(self.command_frequency_hz),
        )


@dataclass(frozen=True, slots=True)
class SingleTennisBallPostResetConfig:
    """Frozen diagnostic episode/controller policy.

    The witness is intentionally absent: it can only be authored and frozen
    from the live pre-reset state, so the runner supplies it explicitly to the
    post-reset callback.
    """

    release_config: PhysicalReleaseConfig
    carry_base_parameters: Mapping[str, CarryBaseParameters]
    episode_id: str = (
        "behavior_hl_b1_v1/picking_up_toys/train/0/"
        "single_tennis_ball_physical_chain"
    )
    split: str = "train"
    instance_id: str = "0"
    seed: int = 0
    empty_base_config: BaseMacroConfig = field(
        default_factory=BaseMacroConfig
    )
    empty_base_safety: LocalBaseSafetyLimits = field(
        default_factory=LocalBaseSafetyLimits
    )
    empty_torso_limits: EmptyTorsoLimits = field(
        default_factory=EmptyTorsoLimits
    )
    carry_limits: CarrySafetyLimits = field(
        default_factory=CarrySafetyLimits
    )
    carry_kinematics_limits: R1ProCarryKinematicsLimits = field(
        default_factory=R1ProCarryKinematicsLimits
    )
    carry_look_fractions: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType(
            {
                "look_up": 0.0,
                "look_center": 0.5,
                "look_down": 1.0,
            }
        )
    )
    place_config: PlaceConfig = field(default_factory=PlaceConfig)
    place_motion_limits: R1ProPlaceMotionLimits | None = None
    place_visual_config: LocalRGBPatchAssessment | None = None
    grasp_binding_config_overrides: Mapping[str, Any] | None = None
    limits: ResidentEpisodeLimits = field(
        default_factory=lambda: ResidentEpisodeLimits(
            max_model_calls=32,
            max_simulator_steps=2_400,
        )
    )
    settle_steps: int = 30

    def __post_init__(self) -> None:
        for name, expected in (
            ("release_config", PhysicalReleaseConfig),
            ("empty_base_config", BaseMacroConfig),
            ("empty_base_safety", LocalBaseSafetyLimits),
            ("empty_torso_limits", EmptyTorsoLimits),
            ("carry_limits", CarrySafetyLimits),
            ("carry_kinematics_limits", R1ProCarryKinematicsLimits),
            ("place_config", PlaceConfig),
            ("limits", ResidentEpisodeLimits),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        parameters = dict(self.carry_base_parameters)
        if set(parameters) != set(BASE_ACTIONS) or any(
            not isinstance(item, CarryBaseParameters)
            for item in parameters.values()
        ):
            raise ValueError(
                "carry_base_parameters must cover all six base actions"
            )
        object.__setattr__(
            self,
            "carry_base_parameters",
            MappingProxyType(parameters),
        )
        fractions = dict(self.carry_look_fractions)
        if set(fractions) != {"look_up", "look_center", "look_down"}:
            raise ValueError(
                "carry_look_fractions must cover all three look actions"
            )
        object.__setattr__(
            self,
            "carry_look_fractions",
            MappingProxyType(fractions),
        )
        if (
            isinstance(self.settle_steps, bool)
            or not isinstance(self.settle_steps, int)
            or self.settle_steps < 0
        ):
            raise ValueError("settle_steps must be non-negative")


class _PublicSnapshotBridge:
    """Late-bind the resident environment without weakening provider typing."""

    def __init__(self) -> None:
        self._provider: Callable[[], Mapping[str, Any]] | None = None

    def bind(self, provider: Callable[[], Mapping[str, Any]]) -> None:
        if self._provider is not None:
            raise R1ProB1PostResetPlanError(
                "public snapshot bridge was bound more than once"
            )
        if not callable(provider):
            raise TypeError("public snapshot provider must be callable")
        self._provider = provider

    def __call__(self) -> Mapping[str, Any]:
        if self._provider is None:
            raise R1ProB1PostResetPlanError(
                "resident public snapshot is not bound"
            )
        return self._provider()


class _AtomicPrivateAudit:
    """Small fsync-backed private stream shared by grasp/carry adapters."""

    def __init__(
        self,
        *,
        path: Path,
        carry_capture: Callable[..., Mapping[str, Any]],
    ) -> None:
        self._path = Path(path)
        self._capture = carry_capture
        self._events: list[dict[str, Any]] = []

    def __call__(self, event: Mapping[str, Any]) -> None:
        self._append("grasp", event)

    def capture_endpoint(
        self,
        *,
        stage: str,
        episode_id: str,
        action_index: int,
    ) -> None:
        value = self._capture(
            stage=stage,
            episode_id=episode_id,
            action_index=action_index,
        )
        if not isinstance(value, Mapping):
            raise R1ProB1PostResetPlanError(
                "private carry endpoint capture must return a mapping"
            )
        self._append("carry", value)

    def _append(self, source: str, event: Mapping[str, Any]) -> None:
        try:
            copied = json.loads(
                json.dumps(
                    dict(event),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise R1ProB1PostResetPlanError(
                "private executor audit is not strict JSON"
            ) from exc
        self._events.append({"source": source, "event": copied})
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(
            f".{self._path.name}.tmp-{os.getpid()}"
        )
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                self._events,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self._path)
        directory_fd = os.open(self._path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def build_single_tennis_ball_post_reset_plan(
    *,
    evaluator: Any,
    control: Any,
    formal_wrist: PublicWristCalibrationRecord,
    safe_carry: SafeCarryCalibrationRecord,
    witness: PublicSingleLeafWitness,
    run_dir: Path,
    dependencies: R1ProB1PostResetLiveDependencies,
    config: SingleTennisBallPostResetConfig,
) -> B1PostResetAssemblyPlan:
    """Build, but do not execute, the post-reset single-leaf resident plan."""

    del control
    if evaluator is None:
        raise ValueError("evaluator must be the live post-reset evaluator")
    if not isinstance(
        dependencies,
        R1ProB1PostResetLiveDependencies,
    ):
        raise TypeError(
            "dependencies must be R1ProB1PostResetLiveDependencies"
        )
    if not isinstance(config, SingleTennisBallPostResetConfig):
        raise TypeError("config must be SingleTennisBallPostResetConfig")
    if not isinstance(witness, PublicSingleLeafWitness):
        raise TypeError("witness must be PublicSingleLeafWitness")
    if (
        not isinstance(formal_wrist, PublicWristCalibrationRecord)
        or not isinstance(safe_carry, SafeCarryCalibrationRecord)
        or not safe_carry.passed
        or formal_wrist.hand != safe_carry.candidate.hand
        or formal_wrist.safe_carry_calibration_sha256
        != safe_carry.sha256
    ):
        raise R1ProB1PostResetPlanError(
            "post-reset plan requires matching passing wrist/carry records"
        )
    hand = safe_carry.candidate.hand
    if witness.calibrated_hand != hand:
        raise R1ProB1PostResetPlanError(
            "Oracle witness hand differs from calibrated carry hand"
        )
    if (
        getattr(dependencies.starter, "arm", None) != hand
        or getattr(dependencies.starter, "robot", None)
        is not getattr(evaluator, "robot", None)
    ):
        raise R1ProB1PostResetPlanError(
            "selected Starter is not bound to the live evaluator/hand"
        )

    registry = build_picking_up_toys_registry(
        calibrated_hands=(hand,)
    )
    witness.validate_against(registry)
    public_bridge = _PublicSnapshotBridge()
    public_wrist = CalibratedPublicGripperWristEvidenceProvider(
        snapshot_provider=public_bridge,
        calibrations={hand: formal_wrist},
    )
    carry_compensator = build_validated_r1pro_carry_compensator(
        starter=dependencies.starter,
        calibration=safe_carry,
        kinematics_limits=config.carry_kinematics_limits,
    )
    safe_pose_provider = provider_from_validated_calibration(safe_carry)
    audit = _AtomicPrivateAudit(
        path=Path(run_dir) / "private" / "executor_audit.json",
        carry_capture=dependencies.capture_private_carry_endpoint,
    )
    motion_kwargs = {
        "public_stability": dependencies.public_stability,
        "private_safety": dependencies.private_safety,
        "command_frequency_hz": dependencies.command_frequency_hz,
    }
    starter_by_hand = {hand: dependencies.starter}
    builders = {
        "grasp_builder": R1ProGraspComponentBuilder(
            starter_by_hand=starter_by_hand,
            allowed_hands=(hand,),
            camera_intrinsics=dependencies.camera_intrinsics,
            private_robot_pose_provider=(
                dependencies.private_robot_pose_provider
            ),
            public_evidence_provider=public_wrist,
            private_contact_provider=dependencies.private_contact_provider,
            private_assisted_provider=(
                dependencies.private_assisted_provider
            ),
            private_held_provider=dependencies.private_held_provider,
            private_lift_provider=dependencies.private_lift_provider,
            private_audit_writer=audit,
            safe_carry_pose_provider=safe_pose_provider,
            binding_config_overrides=(
                config.grasp_binding_config_overrides
            ),
        ),
        "base_builder": R1ProBaseComponentBuilder(**motion_kwargs),
        "carry_builder": R1ProCarryComponentBuilder(
            **motion_kwargs,
            compensator=carry_compensator,
            private_audit=audit,
        ),
        "torso_builder": R1ProTorsoComponentBuilder(**motion_kwargs),
        "place_builder": R1ProPlaceComponentBuilder(
            starter_by_hand=starter_by_hand,
            allowed_hands=(hand,),
            camera_intrinsics=dependencies.camera_intrinsics,
            target_pose_builder=dependencies.target_pose_builder,
            safe_hold_action=dependencies.safe_hold_action,
            limits=config.place_motion_limits,
            visual_config=config.place_visual_config,
        ),
        "release_builder": R1ProReleaseComponentBuilder(**motion_kwargs),
    }
    grasp_thresholds = formal_wrist.thresholds
    dispatcher_kwargs = {
        "grasp_config": PhysicalGraspConfig(
            proxy_thresholds=GraspProxyThresholds(
                empty_close_qpos=grasp_thresholds.empty_close_qpos,
                minimum_qpos_gap=grasp_thresholds.minimum_qpos_gap,
                qvel_epsilon=grasp_thresholds.qvel_epsilon,
                stable_steps=grasp_thresholds.consecutive_frames,
            ),
            minimum_payload_lift_m=(
                safe_carry.limits.minimum_object_lift_m
            ),
        ),
        "place_config": config.place_config,
        "release_config": config.release_config,
        "empty_base_config": config.empty_base_config,
        "empty_base_safety": config.empty_base_safety,
        "empty_torso_limits": config.empty_torso_limits,
        "carry_limits": config.carry_limits,
        "carry_safe_arm_targets": {
            hand: safe_carry.candidate.arm_target,
        },
        "calibrated_carry_hands": (hand,),
        "carry_compensator": carry_compensator,
        "carry_base_parameters": config.carry_base_parameters,
        "carry_look_fractions": config.carry_look_fractions,
        "registry": registry,
    }
    episode = EpisodeSpec(
        episode_id=config.episode_id,
        task_id=PICKING_UP_TOYS_PROFILE.task_name,
        split=config.split,
        instance_id=config.instance_id,
        seed=config.seed,
        instruction=PICKING_UP_TOYS_PROFILE.task_language,
        action_profile="r1pro_esi_single_calibrated_hand_v1",
        model_profile="observation_only_oracle_witness_v1",
        action_schema_version=registry.schema_version,
    )

    def model_factory(environment: Any) -> Any:
        public_bridge.bind(environment.public_snapshot)
        return PublicSingleLeafReplayModel(
            registry=registry,
            witness=witness,
        )

    def artifact_factory(
        episode_run_dir: Path,
        environment: Any,
    ) -> ResidentArtifactSinkAdapter:
        del environment
        writer = ResidentEpisodeArtifactWriter(
            episode_run_dir,
            run_id=episode.episode_id,
            metadata={
                "pilot_kind": "single_leaf_physical_chain",
                "promotion_label": "STATIC_CONTRACT_ONLY",
                "execution_fidelity": "physical_r1pro_research",
                "official_challenge_eligible": False,
                "target_new_leaf_count": 1,
                "full_official_leaf_count": 6,
                "action_registry_sha256": registry.sha256(),
                "witness_sha256": witness.witness_sha256,
                "public_witness": witness.to_manifest(),
                "safe_carry_calibration_sha256": safe_carry.sha256,
                "public_wrist_calibration_sha256": (
                    formal_wrist.calibration_sha256
                ),
                "calibrated_hand": hand,
            },
        )
        return ResidentArtifactSinkAdapter(writer)

    return B1PostResetAssemblyPlan(
        dispatcher_kwargs=dispatcher_kwargs,
        component_builders=builders,
        public_carry_tracker=PublicCarryStateTracker(),
        episode=episode,
        model_factory=model_factory,
        verifier_factory=lambda live_evaluator: (
            OfficialPredicateVerifier(require_single_option=True)
        ),
        artifact_factory=artifact_factory,
        limits=config.limits,
        carry_kinematics_limits=config.carry_kinematics_limits,
        settle_steps=config.settle_steps,
        diagnostic_target_new_leaf_count=1,
    )


def make_single_tennis_ball_post_reset_callback(
    *,
    dependencies: R1ProB1PostResetLiveDependencies,
    config: SingleTennisBallPostResetConfig,
) -> Callable[..., B1PostResetAssemblyPlan]:
    """Adapt the plan builder to the runner's live-witness callback contract."""

    def callback(**kwargs: Any) -> B1PostResetAssemblyPlan:
        core = {
            "evaluator",
            "control",
            "formal_wrist",
            "safe_carry",
            "witness",
            "run_dir",
        }
        extended = core | {"starter", "live_public_snapshot"}
        keys = frozenset(kwargs)
        if keys not in {frozenset(core), frozenset(extended)}:
            raise R1ProB1PostResetPlanError(
                "post-reset callback fields differ from runner contract"
            )
        if keys == frozenset(extended):
            if kwargs["starter"] is not dependencies.starter:
                raise R1ProB1PostResetPlanError(
                    "post-reset callback Starter differs from dependencies"
                )
            if not callable(kwargs["live_public_snapshot"]):
                raise R1ProB1PostResetPlanError(
                    "live_public_snapshot must be callable"
                )
            kwargs.pop("starter")
            kwargs.pop("live_public_snapshot")
        return build_single_tennis_ball_post_reset_plan(
            **kwargs,
            dependencies=dependencies,
            config=config,
        )

    return callback
