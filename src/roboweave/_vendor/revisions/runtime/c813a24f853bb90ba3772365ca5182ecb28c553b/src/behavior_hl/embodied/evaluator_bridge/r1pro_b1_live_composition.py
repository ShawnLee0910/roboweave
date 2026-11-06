"""Zero-placeholder composition for B1 witness authoring and replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders import (
    R1ProBaseComponentBuilder,
    R1ProCarryComponentBuilder,
    R1ProGraspComponentBuilder,
    R1ProPlaceComponentBuilder,
    R1ProReleaseComponentBuilder,
    R1ProTorsoComponentBuilder,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_live_dependencies import (
    R1ProB1PostResetLiveDependencyFactory,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_post_reset_plan import (
    SingleTennisBallPostResetConfig,
    _AtomicPrivateAudit,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_public_wrist_evidence import (
    CalibratedPublicGripperWristEvidenceProvider,
    PublicWristCalibrationRecord,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_resident_environment import (
    R1ProResidentEnvironment,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_safe_carry_stage_kinematics import (
    build_validated_r1pro_carry_compensator,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_single_leaf_witness_author import (
    PublicHLAEnvironmentFacade,
    R1ProSingleLeafAuthorRuntime,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_service_factory import (
    R1ProB1ServiceFactory,
)
from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
    SafeCarryCalibrationRecord,
    provider_from_validated_calibration,
)
from behavior_hl.embodied.policy.feedback import GraspProxyThresholds
from behavior_hl.embodied.policy.profiles import (
    build_picking_up_toys_registry,
)
from behavior_hl.embodied.runtime.b1_dispatcher_factory import (
    build_b1_picking_up_toys_dispatcher,
)
from behavior_hl.embodied.runtime.dispatch import ActionPluginResult
from behavior_hl.embodied.runtime.plugins.grasp import PhysicalGraspConfig
from behavior_hl.embodied.runtime.public_carry_state import (
    PublicCarryStateTracker,
)
from behavior_hl.embodied.runtime.public_single_leaf_witness import (
    PublicWitnessDecision,
)
from behavior_hl.embodied.runtime.tokenplan_resident_model import (
    TokenPlanResidentModel,
)


R1PRO_B1_LIVE_COMPOSITION_VERSION = (
    "behavior2026.r1pro-b1-live-composition/v1"
)


class R1ProB1LiveCompositionError(RuntimeError):
    """Fail-closed production composition error."""


class R1ProSingleLeafAuthorRuntimeFactory:
    """Build the normal resident dispatcher/environment in the held state."""

    api_contract_version = R1PRO_B1_LIVE_COMPOSITION_VERSION

    def __init__(
        self,
        *,
        dependency_factory: R1ProB1PostResetLiveDependencyFactory,
        config: SingleTennisBallPostResetConfig,
        model_python_executable: Path,
        model_wall_timeout_seconds: int = 210,
    ) -> None:
        if not isinstance(
            dependency_factory,
            R1ProB1PostResetLiveDependencyFactory,
        ):
            raise TypeError("dependency_factory has the wrong type")
        if not isinstance(config, SingleTennisBallPostResetConfig):
            raise TypeError("config has the wrong type")
        if (
            isinstance(model_wall_timeout_seconds, bool)
            or not isinstance(model_wall_timeout_seconds, int)
            or model_wall_timeout_seconds <= 0
        ):
            raise ValueError("model_wall_timeout_seconds must be positive")
        self._dependency_factory = dependency_factory
        self._config = config
        self._python = Path(model_python_executable)
        self._timeout = model_wall_timeout_seconds

    def __call__(self, **kwargs: Any) -> R1ProSingleLeafAuthorRuntime:
        expected = {
            "evaluator",
            "control",
            "formal_wrist",
            "safe_carry",
            "run_dir",
            "starter",
            "live_public_snapshot",
            "cached_public_witness_prefix",
        }
        if set(kwargs) != expected:
            raise R1ProB1LiveCompositionError(
                "author runtime factory fields changed"
            )
        evaluator = kwargs["evaluator"]
        control = kwargs["control"]
        formal = kwargs["formal_wrist"]
        safe_carry = kwargs["safe_carry"]
        starter = kwargs["starter"]
        run_dir = Path(kwargs["run_dir"])
        prefix = tuple(kwargs["cached_public_witness_prefix"])
        if (
            not isinstance(formal, PublicWristCalibrationRecord)
            or not isinstance(safe_carry, SafeCarryCalibrationRecord)
            or not safe_carry.passed
            or formal.hand != safe_carry.candidate.hand
            or formal.safe_carry_calibration_sha256 != safe_carry.sha256
            or getattr(starter, "arm", None) != formal.hand
        ):
            raise R1ProB1LiveCompositionError(
                "author runtime calibration/Starter identity changed"
            )
        if (
            not prefix
            or any(
                not isinstance(item, PublicWitnessDecision)
                for item in prefix
            )
            or prefix[-1].action.get("action") != "grasp_at"
            or prefix[-1].public_result_status != "grasp_likely"
        ):
            raise R1ProB1LiveCompositionError(
                "author runtime requires a likely public grasp prefix"
            )
        dependencies = self._dependency_factory(
            evaluator=evaluator,
            starter=starter,
            live_public_snapshot=kwargs["live_public_snapshot"],
        )
        registry, dispatcher, services, tracker = (
            _build_author_execution_components(
                evaluator=evaluator,
                formal_wrist=formal,
                safe_carry=safe_carry,
                dependencies=dependencies,
                config=self._config,
                prefix=prefix,
                run_dir=run_dir,
                live_public_snapshot=kwargs["live_public_snapshot"],
            )
        )
        del registry, tracker
        environment = R1ProResidentEnvironment(
            evaluator=evaluator,
            control=control,
            service_factory=services,
            settle_steps=self._config.settle_steps,
        )
        model = TokenPlanResidentModel(
            frame_artifact_provider=environment.frame_artifact,
            policy_dir=run_dir / "policy" / "witness_author",
            python_executable=self._python,
            wall_timeout_seconds=self._timeout,
        )
        return R1ProSingleLeafAuthorRuntime(
            registry=dispatcher.registry,
            dispatcher=dispatcher.dispatcher,
            environment=PublicHLAEnvironmentFacade(environment),
            model=model,
            episode_id=(
                self._config.episode_id + "/pre_reset_author"
            ),
            max_model_calls=24,
            max_controller_ticks=10_000,
            max_simulator_steps=2_000,
        )


def _build_author_execution_components(
    *,
    evaluator: Any,
    formal_wrist: PublicWristCalibrationRecord,
    safe_carry: SafeCarryCalibrationRecord,
    dependencies: Any,
    config: SingleTennisBallPostResetConfig,
    prefix: Sequence[PublicWitnessDecision],
    run_dir: Path,
    live_public_snapshot: Any,
) -> tuple[Any, Any, R1ProB1ServiceFactory, PublicCarryStateTracker]:
    hand = safe_carry.candidate.hand
    registry = build_picking_up_toys_registry(calibrated_hands=(hand,))
    tracker = PublicCarryStateTracker()
    grasp = registry.parse_payload(prefix[-1].action)
    tracker.update(
        grasp,
        ActionPluginResult(
            status="grasp_likely",
            public_feedback={"grasp_proxy_likely": True},
        ),
    )
    public_wrist = CalibratedPublicGripperWristEvidenceProvider(
        snapshot_provider=live_public_snapshot,
        calibrations={hand: formal_wrist},
    )
    compensator = build_validated_r1pro_carry_compensator(
        starter=dependencies.starter,
        calibration=safe_carry,
        kinematics_limits=config.carry_kinematics_limits,
    )
    audit = _AtomicPrivateAudit(
        path=run_dir / "private" / "witness_author_executor_audit.json",
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
            safe_carry_pose_provider=provider_from_validated_calibration(
                safe_carry
            ),
            binding_config_overrides=(
                config.grasp_binding_config_overrides
            ),
        ),
        "base_builder": R1ProBaseComponentBuilder(**motion_kwargs),
        "carry_builder": R1ProCarryComponentBuilder(
            **motion_kwargs,
            compensator=compensator,
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
    thresholds = formal_wrist.thresholds
    dispatcher = build_b1_picking_up_toys_dispatcher(
        grasp_config=PhysicalGraspConfig(
            proxy_thresholds=GraspProxyThresholds(
                empty_close_qpos=thresholds.empty_close_qpos,
                minimum_qpos_gap=thresholds.minimum_qpos_gap,
                qvel_epsilon=thresholds.qvel_epsilon,
                stable_steps=thresholds.consecutive_frames,
            ),
            minimum_payload_lift_m=(
                safe_carry.limits.minimum_object_lift_m
            ),
        ),
        place_config=config.place_config,
        release_config=config.release_config,
        empty_base_config=config.empty_base_config,
        empty_base_safety=config.empty_base_safety,
        empty_torso_limits=config.empty_torso_limits,
        carry_limits=config.carry_limits,
        carry_safe_arm_targets={
            hand: safe_carry.candidate.arm_target
        },
        calibrated_carry_hands=(hand,),
        carry_compensator=compensator,
        carry_base_parameters=config.carry_base_parameters,
        carry_look_fractions=config.carry_look_fractions,
        registry=registry,
    )
    services = R1ProB1ServiceFactory(
        public_carry_state_source=tracker,
        **builders,
    )
    return registry, dispatcher, services, tracker
