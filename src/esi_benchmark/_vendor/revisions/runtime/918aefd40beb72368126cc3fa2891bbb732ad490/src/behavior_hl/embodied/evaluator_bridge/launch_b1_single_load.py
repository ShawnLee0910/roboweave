"""Production composition entrypoint for one isolated one-load B1 worker."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_b1_live_composition import (
    R1ProSingleLeafAuthorRuntimeFactory,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_live_dependencies import (
    R1ProB1PostResetLiveDependencyFactory,
    R1ProCalibrationPrivateSafetyProviderFactory,
    make_concrete_single_tennis_ball_post_reset_callback,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_live_entrypoint import (
    R1ProObservationApproachConfig,
    build_r1pro_b1_live_bindings,
    capture_sanitized_public_observation,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_post_reset_plan import (
    SingleTennisBallPostResetConfig,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_single_load_bindings import (
    R1ProB1SingleLoadCalibrationConfig,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_observation_only_grasp_lift import (
    PhysicalGraspLiftConfig,
    PublicRGBDGroundingConfig,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_public_wrist_evidence import (
    WristVisualThresholds,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_safe_carry_live_sample import (
    R1ProSafeCarryLiveSampleProviderFactory,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_single_leaf_witness_author import (
    make_r1pro_single_leaf_witness_author,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (
    B1SingleLoadWorkerSpec,
    run_b1_single_load,
)
from behavior_hl.embodied.policy.approach import ApproachLimits
from behavior_hl.embodied.policy.approach_execution import BaseMacroConfig
from behavior_hl.embodied.policy.tokenplan_adapter import TOKENPLAN_MODEL
from behavior_hl.embodied.runtime.b1_dispatcher_factory import (
    CarryBaseParameters,
)
from behavior_hl.embodied.runtime.plugins.release import (
    PhysicalReleaseConfig,
)


PRODUCTION_COMPOSITION_VERSION = (
    "behavior2026.b1-single-load-production-composition/v1"
)
MODEL_PYTHON = Path(os.environ.get("ESI_MODEL_PYTHON", sys.executable))


def build_production_bindings():
    """Build every runner callback without constructing an Evaluator."""

    post_config = SingleTennisBallPostResetConfig(
        release_config=PhysicalReleaseConfig(
            open_qpos=(0.05, 0.05),
        ),
        carry_base_parameters=_carry_base_parameters(),
    )
    dependency_factory = R1ProB1PostResetLiveDependencyFactory(
        command_frequency_hz=30.0,
    )
    author_runtime_factory = R1ProSingleLeafAuthorRuntimeFactory(
        dependency_factory=dependency_factory,
        config=post_config,
        model_python_executable=MODEL_PYTHON,
        model_wall_timeout_seconds=210,
    )
    author = make_r1pro_single_leaf_witness_author(
        runtime_factory=author_runtime_factory,
    )
    post_reset = make_concrete_single_tennis_ball_post_reset_callback(
        dependency_factory=dependency_factory,
        config=post_config,
    )
    calibration_safety = R1ProCalibrationPrivateSafetyProviderFactory(
        live_public_snapshot_factory=lambda evaluator: (
            lambda: capture_sanitized_public_observation(evaluator)
        ),
    )
    sample_factory = R1ProSafeCarryLiveSampleProviderFactory(
        private_safety_provider_factory=calibration_safety,
    )
    return build_r1pro_b1_live_bindings(
        calibration_config=_calibration_config(),
        approach_config=_approach_config(),
        physical_config=PhysicalGraspLiftConfig(
            max_reach_ticks=900,
            max_close_ticks=900,
            max_lift_ticks=900,
            minimum_lift_m=0.05,
            g3b_samples_per_stage=50,
            inactive_gripper_command=1.0,
        ),
        calibration_live_sample_provider_factory=sample_factory,
        author_and_freeze_single_leaf_witness_callback=author,
        witness_author_identity=f"TokenPlan:{TOKENPLAN_MODEL}",
        build_post_reset_plan_callback=post_reset,
    )


def _approach_config() -> R1ProObservationApproachConfig:
    # Research-only override for the isolated grasp gate experiment.  The
    # default remains the conservative public protocol value; the override is
    # never read by the official BEHAVIOR or ESI trees.
    max_horizontal_reach_m = float(
        os.environ.get("BEHAVIOR_HL_MAX_HORIZONTAL_REACH_M", "1.30")
    )
    if not np.isfinite(max_horizontal_reach_m) or max_horizontal_reach_m <= 0.0:
        raise ValueError("BEHAVIOR_HL_MAX_HORIZONTAL_REACH_M must be positive")
    return R1ProObservationApproachConfig(
        grounding=PublicRGBDGroundingConfig(
            intrinsic_matrix=np.asarray(
                (
                    (306.0000085830689, 0.0, 360.0),
                    (0.0, 306.0000085830689, 360.0),
                    (0.0, 0.0, 1.0),
                ),
                dtype=np.float64,
            ),
            maximum_horizontal_reach_m=max_horizontal_reach_m,
        ),
        limits=ApproachLimits(
            max_model_calls=32,
            max_invalid_responses=2,
            max_consecutive_same_action=6,
            max_consecutive_blocked=3,
            max_grasp_queries=8,
            max_grasp_executions=1,
            max_simulator_steps=2400,
            prompt_history_items=8,
            translations_after_out_of_reach=2,
            minimum_effective_translation_m=0.05,
        ),
        base_macro=BaseMacroConfig(),
        command_frequency_hz=30.0,
        base_settle_steps=4,
        look_maximum_steps=90,
        look_settle_steps=4,
        look_tolerance_rad=0.04,
        look_max_delta_per_step=0.025,
        tokenplan_wall_timeout_seconds=180,
    )


def _calibration_config() -> R1ProB1SingleLoadCalibrationConfig:
    return R1ProB1SingleLoadCalibrationConfig(
        hand="right",
        # Frozen central wrist crop for the first tennis-ball pilot. This is
        # calibrated by >=3 empty trials before any positive admission.
        roi_xyxy=(144, 144, 336, 336),
        wrist_thresholds=WristVisualThresholds(
            empty_close_qpos=(0.0, 0.0),
            minimum_qpos_gap=0.01,
            qvel_epsilon=0.001,
            minimum_depth_m=0.04,
            maximum_depth_m=0.30,
            minimum_foreground_depth_delta_m=0.08,
            minimum_valid_depth_fraction=0.95,
            minimum_foreground_fraction=0.50,
            minimum_empty_rgb_delta=20.0,
            minimum_temporal_foreground_iou=0.80,
            maximum_temporal_depth_delta_m=0.01,
            maximum_temporal_rgb_delta=5.0,
            consecutive_frames=5,
        ),
        empty_wrist_trials=3,
        stable_samples_per_trial=5,
        max_steps_per_phase=256,
    )


def _carry_base_parameters():
    return {
        "move_forward": CarryBaseParameters(
            (0.12, 0.0, 0.0), (0.07, 0.0, 0.0)
        ),
        "move_backward": CarryBaseParameters(
            (-0.12, 0.0, 0.0), (-0.07, 0.0, 0.0)
        ),
        "move_left": CarryBaseParameters(
            (0.0, 0.12, 0.0), (0.0, 0.07, 0.0)
        ),
        "move_right": CarryBaseParameters(
            (0.0, -0.12, 0.0), (0.0, -0.07, 0.0)
        ),
        "turn_left": CarryBaseParameters(
            (0.0, 0.0, 0.12), (0.0, 0.0, 10.0)
        ),
        "turn_right": CarryBaseParameters(
            (0.0, 0.0, -0.12), (0.0, 0.0, -10.0)
        ),
    }


def parse_worker_spec(argv: list[str] | None = None) -> B1SingleLoadWorkerSpec:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--server-port", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--task-name", default="picking_up_toys")
    parser.add_argument("--split", default="train")
    parser.add_argument("--instance-id", default=0, type=int)
    args = parser.parse_args(argv)
    return B1SingleLoadWorkerSpec(
        worker_id=args.worker_id,
        physical_gpu=args.physical_gpu,
        server_port=args.server_port,
        seed=args.seed,
        artifact_root=args.artifact_root,
        task_name=args.task_name,
        split=args.split,
        instance_id=args.instance_id,
    )


def main(argv: list[str] | None = None) -> int:
    worker = parse_worker_spec(argv)
    result = run_b1_single_load(
        bindings=build_production_bindings(),
        worker=worker,
    )
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
