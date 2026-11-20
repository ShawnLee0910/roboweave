"""One-construction B1 calibration and resident-episode runner.

This module owns the expensive lifecycle boundary, not the simulator-specific
calibration details.  A concrete live binding must be injected and must
produce typed evidence for each calibration stage.  The runner then:

1. constructs exactly one official ``Evaluator``;
2. performs ``reset -> load_task_instance(0) -> reset``;
3. finishes an observation-only base approach and freezes its public target;
4. proposes a live collision-aware candidate from that final base pose;
5. calibrates the empty wrist at that exact candidate pose;
6. performs one physical grasp / lift without moving the base again;
7. validates the held candidate through the safe-carry G3b contract;
8. freezes all calibration records before resetting;
9. calls ``evaluator.reset()`` and privately attests restoration without
   reconstructing the scene;
10. assembles the B1 dispatcher, composite services, resident environment, and
   resident episode; and
11. runs the episode in the same Kit process and scene construction.

There is deliberately no default live binding.  Missing live callbacks fail
before ``Evaluator`` construction; no placeholder PASS record or fixed carry
pose is admitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from behavior_hl.embodied.evaluator_bridge.camera_parameter_prewarm import (
    prewarm_evaluator_camera_parameters,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_b1_service_factory import (
    R1ProB1ServiceFactory,
)
from behavior_hl.embodied.evaluator_bridge.official_repository_identity import (
    capture_official_repository_identity,
    require_unchanged_official_repository_identity,
)
from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    FULL_CAMERA_PROFILE,
    HEAD_RGB_MINIMAL_CAMERA_PROFILE,
    ORIGINAL_RESET_ARM_JOINTS_CONFIG_KEY,
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
    build_predeclared_r1pro_camera_robot_config,
    camera_profile_contract,
)
from behavior_hl.embodied.evaluator_bridge.arm_presentation_ladder_runtime import (
    ARM_PRESENTATION_LADDER_CONFIG_ATTRIBUTE,
    DEGRADED_TO_UNTUCKED_STATUS,
    apply_load_time_inward_presentation_with_ladder_grading,
    build_arm_presentation_ladder_config,
)
from behavior_hl.embodied.evaluator_bridge.gripper_visible_camera_tuning import (
    HEAD_FOCAL_LENGTH_MM,
    WRIST_FOCAL_LENGTH_MM,
    apply_gripper_visible_camera_tuning,
    gripper_visible_camera_tuning_contract,
)
from behavior_hl.embodied.evaluator_bridge.gripper_visual_profiles import (
    GRIPPER_VISUAL_PROFILE_DISABLED,
    gripper_visual_profile_contract,
)
from behavior_hl.embodied.evaluator_bridge.moderate_inward_gripper_presentation import (
    MODERATE_INWARD_GRIPPER_PRESENTATION_VERSION,
    apply_moderate_inward_gripper_presentation,
)
from behavior_hl.embodied.evaluator_bridge.full_scene_robot_filter import (
    require_full_scene_robot_filter_report,
    temporary_full_scene_robot_filter,
)
from behavior_hl.embodied.evaluator_bridge.preconstruction_checkpoint_frequency import (
    configure_preconstruction_checkpoint_frequency,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_live_safe_carry_candidate import (
    LiveSafeCarryProposal,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_public_wrist_evidence import (
    BootstrapPublicWristCalibrationRecord,
    PublicWristCalibrationRecord,
    PublicWristPositiveWindow,
    promote_bootstrap_wrist_calibration,
    safe_carry_candidate_sha256,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_resident_environment import (
    HEAD_DOMINANT_CAMERA_LAYOUT_PROFILE,
    NATIVE_CAMERA_LAYOUT_PROFILE,
    R1ProResidentEnvironment,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_safe_carry_stage_kinematics import (
    R1ProCarryKinematicsLimits,
    audit_r1pro_g3b_stage_reachability,
)
from behavior_hl.embodied.evaluator_bridge.run_g1_canary import (
    BEHAVIOR_ROOT,
    CONFIG_PATH,
    EFFECTIVE_WRAPPER,
    PROJECT_ROOT,
    _configure_process_environment,
    _git_head,
    _load_and_validate_config,
    _record_stage,
    _sha256,
    _validate_launch_contract,
    _wait_for_port,
    _write_json,
)
from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
    SafeCarryCalibrationActuator,
    SafeCarryCalibrationLimits,
    SafeCarryCalibrationRecord,
    run_safe_carry_calibration,
)
from behavior_hl.embodied.evaluator_bridge.same_scene_reset_attestation import (
    PrivateResetSnapshot,
    SameSceneResetTolerances,
    attest_same_scene_reset,
)
from behavior_hl.embodied.policy.canary_server import (
    SharedCanaryControl,
    server_process_main,
)
from behavior_hl.embodied.evaluator_bridge.oracle_websocket_keepalive import (
    oracle_websocket_keepalive_manifest,
)
from behavior_hl.embodied.policy.observations import schema_sha256
from behavior_hl.embodied.policy.profiles import (
    build_picking_up_toys_registry,
)
from behavior_hl.embodied.runtime.b1_dispatcher_factory import (
    B1DispatcherBundle,
    build_b1_picking_up_toys_dispatcher,
)
from behavior_hl.embodied.runtime.b1_episode_factory import (
    B1ResidentEpisodeBundle,
    build_b1_resident_episode,
)
from behavior_hl.embodied.runtime.b1_live_readiness import (
    B1_REQUIRED_ACTIONS,
    audit_b1_live_readiness,
)
from behavior_hl.embodied.runtime.episode import EpisodeSpec
from behavior_hl.embodied.runtime.public_carry_state import (
    PublicCarryStateTracker,
)
from behavior_hl.embodied.runtime.public_single_leaf_witness import (
    PublicSingleLeafReplayModel,
    PublicSingleLeafWitness,
)
from behavior_hl.embodied.runtime.termination import TerminationClass
from behavior_hl.project_identity import capture_project_identity
from behavior_hl.embodied.runtime.resident_episode import (
    ResidentEpisodeLimits,
    ResidentEpisodeResult,
)


def _seed_before_kit_without_warp(
    seed: int,
    *,
    physical_gpu: int,
) -> int:
    """Seed scene RNGs without importing Warp before Kit owns its runtime.

    The official ``seed_everything`` helper also imports Warp.  Calling it
    before Isaac Kit has loaded ``omni.warp.core`` can leave the overlay
    ``warp-lang`` package and Kit's Warp extension mixed in one process.
    Scene construction still needs deterministic Python / NumPy / Torch
    state, so this pre-Kit phase intentionally seeds only those RNGs.  Warp
    provenance and runtime readiness are attested after the Evaluator exists;
    no claim is made that a process-global Warp RNG was seeded.
    """

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch as th

    th.random.default_generator.manual_seed(seed)
    if th.cuda.is_available():
        with th.cuda.device(physical_gpu):
            th.cuda.manual_seed(seed)
    if hasattr(th.backends, "cudnn"):
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True
    return seed


def _require_warp_absent_before_kit() -> None:
    """Fail if any library imported Warp before Kit selected its extension."""

    imported = sorted(
        name
        for name in sys.modules
        if name == "warp" or name.startswith("warp.")
    )
    if imported:
        raise B1SingleLoadRunnerError(
            "Warp was imported before Kit runtime construction: "
            f"{imported[:8]}"
        )


def _attest_kit_warp_runtime(
    *,
    physical_gpu: int,
) -> dict[str, Any]:
    """Attest one Isaac Warp package and initialize its current context."""

    import warp as wp
    import warp.context as wp_context
    import warp.torch as wp_torch
    from curobo.util.warp import init_warp

    paths = {
        "warp": str(Path(wp.__file__).resolve()),
        "warp_context": str(Path(wp_context.__file__).resolve()),
        "warp_torch": str(Path(wp_torch.__file__).resolve()),
    }
    expected_marker = "isaacsim/extscache/omni.warp.core-"
    if any(expected_marker not in path for path in paths.values()):
        raise B1SingleLoadRunnerError(
            f"Warp modules are not from one Isaac extension: {paths}"
        )
    package_roots = {
        path.split("/warp/", 1)[0]
        for path in paths.values()
        if "/warp/" in path
    }
    if len(package_roots) != 1:
        raise B1SingleLoadRunnerError(
            f"Warp module package roots differ: {paths}"
        )

    init_warp()
    runtime = getattr(wp_context, "runtime", None)
    cuda_devices = getattr(runtime, "cuda_devices", None)
    if runtime is None or cuda_devices is None:
        raise B1SingleLoadRunnerError(
            "Isaac Warp CUDA runtime is unavailable after init_warp"
        )
    if physical_gpu >= len(cuda_devices):
        raise B1SingleLoadRunnerError(
            "Isaac Warp runtime does not expose the selected physical GPU"
        )
    selected = cuda_devices[physical_gpu]
    expected_alias = f"cuda:{physical_gpu}"
    if (
        getattr(selected, "ordinal", None) != physical_gpu
        or getattr(selected, "alias", None) != expected_alias
    ):
        raise B1SingleLoadRunnerError(
            "Isaac Warp selected device ordinal / alias mismatch"
        )
    return {
        **paths,
        "physical_gpu": physical_gpu,
        "cuda_device_count": len(cuda_devices),
        "selected_alias": str(selected.alias),
        "selected_ordinal": int(selected.ordinal),
        "selected_uuid": getattr(selected, "uuid", None),
        "selected_pci_bus_id": getattr(selected, "pci_bus_id", None),
    }
from behavior_hl.embodied.runtime.plugins.carry import CarrySafetyLimits


B1_SINGLE_LOAD_RUNNER_VERSION = "behavior2026.b1-single-load-runner/v3"
B1_SINGLE_LOAD_CONTRACT_PATH = (
    PROJECT_ROOT / "configs" / "suites" / "b1_single_load.yaml"
)
B1_SINGLE_LOAD_STAGE_ORDER = (
    "capture_pre_calibration_reset_state",
    "observation_only_approach",
    "candidate_proposal",
    "empty_wrist_negative_calibration",
    "physical_grasp_lift",
    "safe_carry_g3b",
    "freeze_calibration_records",
    "author_and_freeze_single_leaf_witness",
    "same_scene_episode_reset",
    "capture_post_reset_state",
    "private_reset_attestation",
    "build_b1_dispatcher",
    "build_b1_service_factory",
    "build_b1_resident_episode",
    "run_b1_resident_episode",
)
_COMPONENT_BUILDER_KEYS = frozenset(
    {
        "grasp_builder",
        "base_builder",
        "carry_builder",
        "torso_builder",
        "place_builder",
        "release_builder",
    }
)


class B1SingleLoadRunnerError(RuntimeError):
    """Fail-closed runner construction or lifecycle error."""


@contextmanager
def temporary_evaluator_scene_file_override(
    evaluator_module: Any,
    *,
    scene_file: Path | None,
    expected_scene_model: str | None,
):
    """Point one Evaluator construction at a generated scene superset.

    The override is process-local and restored immediately after construction;
    official files and the evaluator's default behavior are unchanged.
    """

    if scene_file is None:
        yield {"enabled": False}
        return
    resolved = Path(scene_file).resolve()
    if not resolved.is_file() or resolved.suffix != ".json":
        raise B1SingleLoadRunnerError("resident scene override must be a JSON file")
    original = evaluator_module.generate_basic_environment_config

    def generate(task_name: str, task_cfg: Mapping[str, Any]):
        cfg = original(task_name=task_name, task_cfg=task_cfg)
        actual_scene_model = str(task_cfg["scene_model"])
        if expected_scene_model is not None and actual_scene_model != expected_scene_model:
            raise B1SingleLoadRunnerError(
                "resident scene override task escaped its scene-affinity group"
            )
        cfg["scene"]["scene_file"] = str(resolved)
        cfg["scene"]["scene_instance"] = None
        cfg["scene"]["scene_model"] = actual_scene_model
        return cfg

    evaluator_module.generate_basic_environment_config = generate
    try:
        yield {
            "enabled": True,
            "scene_file": str(resolved),
            "expected_scene_model": expected_scene_model,
        }
    finally:
        evaluator_module.generate_basic_environment_config = original


@dataclass(frozen=True, slots=True)
class B1SingleLoadWorkerSpec:
    """Immutable ownership and reproducibility boundary for one live worker."""

    worker_id: str
    physical_gpu: int
    server_port: int
    seed: int
    artifact_root: Path
    task_name: str = "picking_up_toys"
    split: str = "train"
    instance_id: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.worker_id, str)
            or not self.worker_id
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for character in self.worker_id
            )
        ):
            raise ValueError("worker_id must be a non-empty safe path token")
        for name in ("physical_gpu", "server_port", "seed"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if not 1 <= self.server_port <= 65535:
            raise ValueError("server_port must be in [1, 65535]")
        for name in ("task_name", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if (
            isinstance(self.instance_id, bool)
            or not isinstance(self.instance_id, int)
            or self.instance_id < 0
        ):
            raise ValueError("instance_id must be a non-negative integer")
        root = Path(self.artifact_root).resolve()
        if root == Path("/"):
            raise ValueError("artifact_root must not be filesystem root")
        object.__setattr__(self, "artifact_root", root)


@dataclass(frozen=True, slots=True)
class B1FrozenBasePoseEvidence:
    """Base-only token; arm motion must not change its meaning."""

    phase: str
    frozen_base_pose_token: str
    position: tuple[float, ...]
    quaternion_xyzw: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.phase not in {
            "final_approach",
            "after_empty_wrist",
            "after_grasp_lift",
            "after_g3b",
        }:
            raise ValueError("frozen base evidence phase is invalid")
        if (
            not isinstance(self.frozen_base_pose_token, str)
            or not self.frozen_base_pose_token
        ):
            raise ValueError("frozen_base_pose_token must be non-empty")
        object.__setattr__(
            self,
            "position",
            _finite_vector(self.position, 3, "base position"),
        )
        quaternion = _finite_vector(
            self.quaternion_xyzw,
            4,
            "base quaternion",
        )
        if math.sqrt(sum(value * value for value in quaternion)) <= 1e-8:
            raise ValueError("base quaternion has zero norm")
        object.__setattr__(self, "quaternion_xyzw", quaternion)


@dataclass(frozen=True, slots=True)
class B1ObservationOnlyApproach:
    """Frozen public target and final robot/base state before proposal."""

    target_token: str
    public_frame_sha256: str
    candidate_source_state_token: str
    frozen_base: B1FrozenBasePoseEvidence
    target_from_public_rgbd: bool
    physical_grasp_executions: int
    private_truth_exposed_to_policy: bool

    def __post_init__(self) -> None:
        for name in (
            "target_token",
            "public_frame_sha256",
            "candidate_source_state_token",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        if (
            len(self.public_frame_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.public_frame_sha256
            )
        ):
            raise ValueError("public_frame_sha256 must be sha256")
        if (
            not isinstance(self.frozen_base, B1FrozenBasePoseEvidence)
            or self.frozen_base.phase != "final_approach"
        ):
            raise TypeError(
                "approach requires final_approach base evidence"
            )
        if self.target_from_public_rgbd is not True:
            raise ValueError("approach target must come from public RGB-D")
        if self.physical_grasp_executions != 0:
            raise ValueError("approach phase must not execute a grasp")
        if self.private_truth_exposed_to_policy is not False:
            raise ValueError("approach policy must not receive private truth")


@dataclass(frozen=True, slots=True)
class B1EmptyWristCalibration:
    """Exact candidate-pose empty-hand baseline with base-only continuity."""

    bootstrap: BootstrapPublicWristCalibrationRecord
    frozen_base: B1FrozenBasePoseEvidence
    base_motion_commanded: bool

    def __post_init__(self) -> None:
        if not isinstance(
            self.bootstrap,
            BootstrapPublicWristCalibrationRecord,
        ):
            raise TypeError("bootstrap wrist calibration is invalid")
        if (
            not isinstance(self.frozen_base, B1FrozenBasePoseEvidence)
            or self.frozen_base.phase != "after_empty_wrist"
        ):
            raise TypeError(
                "empty wrist requires after_empty_wrist base evidence"
            )
        if self.base_motion_commanded is not False:
            raise ValueError(
                "base movement is forbidden during empty wrist calibration"
            )


@dataclass(frozen=True, slots=True)
class B1CalibrationGraspLift:
    """One unscored physical grasp after the base pose has been frozen."""

    positive_window: PublicWristPositiveWindow
    safe_carry_actuator: SafeCarryCalibrationActuator
    consumed_target_token: str
    consumed_candidate_source_state_token: str
    frozen_base: B1FrozenBasePoseEvidence
    physical_grasp_executions: int
    observed_lift_m: float
    base_motion_commanded: bool
    private_truth_exposed_to_policy: bool

    def __post_init__(self) -> None:
        if not isinstance(
            self.positive_window,
            PublicWristPositiveWindow,
        ):
            raise TypeError(
                "positive_window must be PublicWristPositiveWindow"
            )
        for name in (
            "ik_only_valid",
            "step_toward",
            "step_g3b",
            "safe_hold",
        ):
            if not callable(getattr(self.safe_carry_actuator, name, None)):
                raise TypeError(
                    f"safe_carry_actuator is missing {name}"
                )
        for name in (
            "consumed_target_token",
            "consumed_candidate_source_state_token",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        if (
            not isinstance(self.frozen_base, B1FrozenBasePoseEvidence)
            or self.frozen_base.phase != "after_grasp_lift"
        ):
            raise TypeError(
                "grasp lift requires after_grasp_lift base evidence"
            )
        if self.physical_grasp_executions != 1:
            raise ValueError(
                "calibration must execute exactly one physical grasp"
            )
        if (
            isinstance(self.observed_lift_m, bool)
            or not isinstance(self.observed_lift_m, (int, float))
            or float(self.observed_lift_m) < 0.05
        ):
            raise ValueError(
                "calibration grasp must observe at least 0.05 m lift"
            )
        if self.base_motion_commanded is not False:
            raise ValueError(
                "base movement is forbidden after candidate proposal"
            )
        if self.private_truth_exposed_to_policy is not False:
            raise ValueError(
                "calibration policy must not receive private truth"
            )


@dataclass(frozen=True, slots=True)
class B1PostResetAssemblyPlan:
    """All explicit dependencies needed after the in-scene reset."""

    dispatcher_kwargs: Mapping[str, Any]
    component_builders: Mapping[str, Any]
    public_carry_tracker: PublicCarryStateTracker
    episode: EpisodeSpec
    model_factory: Callable[[R1ProResidentEnvironment], Any]
    verifier_factory: Callable[[Any], Any]
    artifact_factory: Callable[
        [Path, R1ProResidentEnvironment], Any
    ]
    limits: ResidentEpisodeLimits
    carry_kinematics_limits: R1ProCarryKinematicsLimits
    settle_steps: int
    diagnostic_target_new_leaf_count: int | None = None
    settle_step_policy: Callable[[Any, Any], int] | None = None

    def __post_init__(self) -> None:
        dispatcher_kwargs = dict(self.dispatcher_kwargs)
        if not dispatcher_kwargs:
            raise ValueError("dispatcher_kwargs must not be empty")
        object.__setattr__(
            self,
            "dispatcher_kwargs",
            MappingProxyType(dispatcher_kwargs),
        )
        builders = dict(self.component_builders)
        if set(builders) != _COMPONENT_BUILDER_KEYS:
            raise ValueError(
                "component_builders must contain exactly "
                f"{sorted(_COMPONENT_BUILDER_KEYS)}"
            )
        object.__setattr__(
            self,
            "component_builders",
            MappingProxyType(builders),
        )
        if not isinstance(
            self.public_carry_tracker,
            PublicCarryStateTracker,
        ):
            raise TypeError(
                "public_carry_tracker must be PublicCarryStateTracker"
            )
        if not isinstance(self.episode, EpisodeSpec):
            raise TypeError("episode must be EpisodeSpec")
        if self.episode.task_id != "picking_up_toys":
            raise ValueError("single-load B1 task must be picking_up_toys")
        for callback_name in (
            "model_factory",
            "verifier_factory",
            "artifact_factory",
        ):
            if not callable(getattr(self, callback_name)):
                raise TypeError(f"{callback_name} must be callable")
        if not isinstance(self.limits, ResidentEpisodeLimits):
            raise TypeError("limits must be ResidentEpisodeLimits")
        if not isinstance(
            self.carry_kinematics_limits,
            R1ProCarryKinematicsLimits,
        ):
            raise TypeError(
                "carry_kinematics_limits must be "
                "R1ProCarryKinematicsLimits"
            )
        if (
            isinstance(self.settle_steps, bool)
            or not isinstance(self.settle_steps, int)
            or self.settle_steps < 0
        ):
            raise ValueError("settle_steps must be non-negative")
        if self.diagnostic_target_new_leaf_count is not None and (
            isinstance(self.diagnostic_target_new_leaf_count, bool)
            or not isinstance(
                self.diagnostic_target_new_leaf_count,
                int,
            )
            or self.diagnostic_target_new_leaf_count <= 0
            or self.diagnostic_target_new_leaf_count >= 6
        ):
            raise ValueError(
                "diagnostic_target_new_leaf_count must be in [1, 5]"
            )
        if self.settle_step_policy is not None and not callable(
            self.settle_step_policy
        ):
            raise TypeError("settle_step_policy must be callable")


class B1SingleLoadLiveBindings(Protocol):
    """Simulator-specific construction edge required by this runner."""

    api_contract_version: str

    def capture_private_reset_snapshot(
        self,
        *,
        evaluator: Any,
        phase: str,
        run_dir: Path,
    ) -> PrivateResetSnapshot:
        """Capture evaluator-only stable state for tolerant reset checking."""

    def observation_only_approach(
        self,
        *,
        evaluator: Any,
        control: SharedCanaryControl,
        run_dir: Path,
    ) -> B1ObservationOnlyApproach:
        """Move the base, freeze a public RGB-D target, and do not grasp."""

    def propose_candidate(
        self,
        *,
        evaluator: Any,
        control: SharedCanaryControl,
        approach: B1ObservationOnlyApproach,
        run_dir: Path,
    ) -> LiveSafeCarryProposal:
        """Propose a live collision-aware candidate without physics steps."""

    def calibrate_empty_wrist(
        self,
        *,
        evaluator: Any,
        control: SharedCanaryControl,
        approach: B1ObservationOnlyApproach,
        proposal: LiveSafeCarryProposal,
        minimum_trials: int,
        run_dir: Path,
    ) -> B1EmptyWristCalibration:
        """Move empty hand to exact candidate pose and prove zero FP."""

    def physical_grasp_lift(
        self,
        *,
        evaluator: Any,
        control: SharedCanaryControl,
        approach: B1ObservationOnlyApproach,
        proposal: LiveSafeCarryProposal,
        empty_wrist: B1EmptyWristCalibration,
        run_dir: Path,
    ) -> B1CalibrationGraspLift:
        """Consume the frozen target; base movement is now forbidden."""

    def capture_frozen_base_pose(
        self,
        *,
        evaluator: Any,
        phase: str,
        run_dir: Path,
    ) -> B1FrozenBasePoseEvidence:
        """Capture base-only continuity after held G3b arm/torso motion."""

    def author_and_freeze_single_leaf_witness(
        self,
        *,
        evaluator: Any,
        control: SharedCanaryControl,
        approach: B1ObservationOnlyApproach,
        formal_wrist: PublicWristCalibrationRecord,
        safe_carry: SafeCarryCalibrationRecord,
        run_dir: Path,
    ) -> PublicSingleLeafWitness:
        """Continue from held G3b, place one ball, and freeze a public witness."""

    def build_post_reset_plan(
        self,
        *,
        evaluator: Any,
        control: SharedCanaryControl,
        formal_wrist: PublicWristCalibrationRecord,
        safe_carry: SafeCarryCalibrationRecord,
        witness: PublicSingleLeafWitness,
        run_dir: Path,
    ) -> B1PostResetAssemblyPlan:
        """Return explicit post-reset dependencies; do not step simulator."""


@dataclass(frozen=True, slots=True)
class B1SingleLoadRunResult:
    run_dir: Path
    calibration: SafeCarryCalibrationRecord
    formal_wrist: PublicWristCalibrationRecord
    witness: PublicSingleLeafWitness
    episode: ResidentEpisodeResult
    scene_loads_attempted: int = 1
    successful_scene_loads: int = 1
    in_memory_resets_after_calibration: int = 1


@dataclass(frozen=True, slots=True)
class B1StrategySweepRunResult:
    """Completed same-scene diagnostic sweep result.

    This is intentionally separate from the normal B1 benchmark result: a
    strategy sweep may contain Oracle diagnostics and failed Qwen pilots and
    therefore must never be interpreted as a scored resident episode.
    """

    run_dir: Path
    summary: Mapping[str, Any]
    scene_loads_attempted: int = 1
    successful_scene_loads: int = 1


@dataclass(frozen=True, slots=True)
class _AssembledResidentEpisode:
    bundle: B1ResidentEpisodeBundle
    recording_artifacts: "_RecordingFinalizingArtifactSink"
    model: Any


class _RecordingFinalizingArtifactSink:
    """Close the official MP4 before the atomic sink hashes it.

    ``ResidentEpisodeOrchestrator`` calls ``artifacts.finalize`` before
    returning.  Therefore the evaluator recording must be closed inside this
    narrow adapter, before the delegate registers and hashes ``episode.mp4``.
    """

    def __init__(
        self,
        *,
        evaluator: Any,
        delegate: Any,
        video_path: Path,
        video_fps: int,
    ) -> None:
        for name in ("record", "finalize"):
            if not callable(getattr(delegate, name, None)):
                raise TypeError(f"artifact delegate is missing {name}")
        if (
            isinstance(video_fps, bool)
            or not isinstance(video_fps, int)
            or video_fps <= 0
        ):
            raise ValueError("video_fps must be positive")
        self._evaluator = evaluator
        self._delegate = delegate
        self._video_path = Path(video_path)
        self._video_fps = video_fps
        self._recording = False

    def start(self) -> None:
        if self._recording:
            raise B1SingleLoadRunnerError(
                "episode recording is already active"
            )
        self._evaluator.start_recording(
            str(self._video_path),
            rate=self._video_fps,
        )
        self._recording = True

    def stop_if_active(self) -> None:
        if self._recording:
            self._evaluator.stop_recording()
            self._recording = False

    def record(self, **kwargs: Any) -> None:
        self._delegate.record(**kwargs)

    def finalize(self, final_payload: Mapping[str, Any]) -> Any:
        self.stop_if_active()
        return self._delegate.finalize(final_payload)


def validate_single_load_bindings(bindings: Any) -> None:
    """Reject incomplete live bindings before any runtime import or scene."""

    if (
        getattr(bindings, "api_contract_version", None)
        != B1_SINGLE_LOAD_RUNNER_VERSION
    ):
        raise B1SingleLoadRunnerError(
            "live binding version does not match the single-load runner"
        )
    missing = [
        name
        for name in (
            "capture_private_reset_snapshot",
            "observation_only_approach",
            "propose_candidate",
            "calibrate_empty_wrist",
            "physical_grasp_lift",
            "capture_frozen_base_pose",
            "author_and_freeze_single_leaf_witness",
            "build_post_reset_plan",
        )
        if not callable(getattr(bindings, name, None))
    ]
    if missing:
        raise B1SingleLoadRunnerError(
            f"single-load live bindings are incomplete: {missing}"
        )


def load_single_load_contract() -> Mapping[str, Any]:
    """Load and strictly validate the static one-construction contract."""

    import yaml

    with B1_SINGLE_LOAD_CONTRACT_PATH.open(
        "r",
        encoding="utf-8",
    ) as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise B1SingleLoadRunnerError(
            "single-load contract must be a mapping"
        )
    expected = {
        "schema_version": B1_SINGLE_LOAD_RUNNER_VERSION,
        "physical_gpu": "worker_spec",
        "scene_load_budget": 1,
        "evaluator_instances": 1,
        "automatic_retry": False,
        "empty_close_minimum_trials": 3,
        "empty_close_maximum_false_positives": 0,
        "post_calibration_in_memory_resets": 1,
        "official_challenge_eligible": False,
        "maximum_grasp_base_position_drift_m": 0.01,
        "maximum_grasp_base_orientation_drift_rad": 0.01,
    }
    mismatches = {
        key: {"actual": value.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    expected_keepalive = oracle_websocket_keepalive_manifest()
    if value.get("websocket_keepalive") != expected_keepalive:
        mismatches["websocket_keepalive"] = {
            "actual": value.get("websocket_keepalive"),
            "expected": expected_keepalive,
        }
    if tuple(value.get("stage_order", ())) != B1_SINGLE_LOAD_STAGE_ORDER:
        mismatches["stage_order"] = {
            "actual": value.get("stage_order"),
            "expected": B1_SINGLE_LOAD_STAGE_ORDER,
        }
    limits = value.get("safe_carry_limits")
    if not isinstance(limits, Mapping):
        mismatches["safe_carry_limits"] = {
            "actual": type(limits).__name__,
            "expected": "mapping",
        }
    reset_tolerances = value.get("reset_tolerances")
    if not isinstance(reset_tolerances, Mapping):
        mismatches["reset_tolerances"] = {
            "actual": type(reset_tolerances).__name__,
            "expected": "mapping",
        }
    reset_sources = value.get("official_reset_semantics_source_refs")
    if not isinstance(reset_sources, Mapping) or set(reset_sources) != {
        "evaluator_reset",
        "instance_initial_file_update",
        "task_reset",
        "task_reset_scene",
        "scene_hard_reset",
    }:
        mismatches["official_reset_semantics_source_refs"] = {
            "actual": reset_sources,
            "expected": "five frozen official source references",
        }
    if mismatches:
        raise B1SingleLoadRunnerError(
            f"single-load contract mismatch: {mismatches}"
        )
    return MappingProxyType(dict(value))


def run_b1_single_load(
    *,
    bindings: B1SingleLoadLiveBindings,
    worker: B1SingleLoadWorkerSpec,
    strategy_sweep_callback: Callable[..., Mapping[str, Any]] | None = None,
    camera_profile: str = FULL_CAMERA_PROFILE,
    install_private_head_segmentation_side_channel: bool = False,
    expected_task_id: int = 7,
    partial_scene_load: bool | None = None,
    filter_embedded_scene_agents: bool = False,
    use_wrist_evidence_reset_pose: bool = False,
    gripper_visual_profile: str = GRIPPER_VISUAL_PROFILE_DISABLED,
    defer_inward_arm_presentation: bool = False,
    resident_scene_file: Path | None = None,
    resident_scene_model: str | None = None,
    evaluator_module_configurer: Callable[[Any], Callable[[], None]] | None = None,
    native_task_metric: bool = True,
) -> B1SingleLoadRunResult | B1StrategySweepRunResult:
    """Run calibration and one B1 episode with one official scene creation."""

    # These checks must precede environment setup, runtime imports, server
    # startup, and therefore any possible scene construction.
    validate_single_load_bindings(bindings)
    selected_camera_profile = camera_profile_contract(camera_profile)
    selected_gripper_visual_profile = gripper_visual_profile_contract(
        gripper_visual_profile
    )
    contract = load_single_load_contract()
    run_config = _load_and_validate_config()
    effective_partial_scene_load = (
        bool(run_config["runtime"]["partial_scene_load"])
        if partial_scene_load is None
        else bool(partial_scene_load)
    )
    if filter_embedded_scene_agents and effective_partial_scene_load:
        raise B1SingleLoadRunnerError(
            "embedded scene-agent filtering is reserved for full-scene loads"
        )
    if not isinstance(use_wrist_evidence_reset_pose, bool):
        raise B1SingleLoadRunnerError(
            "use_wrist_evidence_reset_pose must be bool"
        )
    if not isinstance(install_private_head_segmentation_side_channel, bool):
        raise B1SingleLoadRunnerError(
            "install_private_head_segmentation_side_channel must be bool"
        )
    if not isinstance(defer_inward_arm_presentation, bool):
        raise B1SingleLoadRunnerError(
            "defer_inward_arm_presentation must be bool"
        )
    if type(native_task_metric) is not bool:
        raise B1SingleLoadRunnerError("native_task_metric must be boolean")
    if evaluator_module_configurer is not None and not callable(
        evaluator_module_configurer
    ):
        raise B1SingleLoadRunnerError(
            "evaluator_module_configurer must be callable"
        )
    if (
        defer_inward_arm_presentation
        and not selected_gripper_visual_profile["inward_arm_presentation"]
    ):
        raise B1SingleLoadRunnerError(
            "deferring the inward arm presentation requires a gripper visual "
            "profile that includes it"
        )
    if (
        selected_gripper_visual_profile["camera_tuning"]
        and camera_profile
        not in {
            THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
            FULL_CAMERA_PROFILE,
        }
    ):
        raise B1SingleLoadRunnerError(
            "gripper visual profiles require a three-camera RGB profile"
        )
    limits = _safe_carry_limits(contract)
    reset_tolerances = _reset_tolerances(contract)
    if strategy_sweep_callback is None and worker.task_name != "picking_up_toys":
        raise B1SingleLoadRunnerError(
            "the legacy physical-calibration path supports picking_up_toys only"
        )

    artifact_root = worker.artifact_root / "workers" / worker.worker_id
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_b1_single_load_{os.getpid()}"
    )
    run_dir = artifact_root / run_id
    for relative in (
        "private",
        "policy",
        "video",
        "logs",
        "lifecycle",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=False)

    stages = run_dir / "lifecycle" / "stages.jsonl"
    try:
        (
            source_freeze,
            official_repository_before,
            initial_manifest,
        ) = _run_zero_scene_preflight(
            run_dir=run_dir,
            run_id=run_id,
            stages=stages,
            contract=contract,
            reset_tolerances=reset_tolerances,
            worker=worker,
            camera_profile=selected_camera_profile["camera_profile"],
            gripper_visual_profile=selected_gripper_visual_profile["name"],
        )
    except Exception as exc:
        _write_preflight_failure(
            run_dir=run_dir,
            run_id=run_id,
            stages=stages,
            exc=exc,
        )
        raise

    context = multiprocessing.get_context("spawn")
    control = SharedCanaryControl(context)
    control.set_expected_task_id(expected_task_id)
    ready_queue = context.Queue()
    server_process = context.Process(
        target=server_process_main,
        args=(
            str(run_config["g1"]["server_host"]),
            worker.server_port,
            str(run_dir / "policy"),
            control,
            float(run_config["observation"][
                "minimum_depth_finite_ratio"
            ]),
            ready_queue,
            camera_profile,
            expected_task_id,
        ),
        daemon=False,
    )
    evaluator_context = None
    evaluator_module_cleanup = None
    recording = False
    scene_loads_attempted = 0
    successful_scene_loads = 0
    try:
        server_process.start()
        host = str(run_config["g1"]["server_host"])
        # Importing the isolated policy process can exceed 20 seconds when the
        # shared repository tree is cold or the backing filesystem is under
        # load.  This wait happens before OmniGibson is imported and before the
        # single scene-load budget is touched, so a longer fail-fast window is
        # safe and avoids discarding a valid run during storage jitter.
        port = int(ready_queue.get(timeout=120))
        _wait_for_port(host, port)
        _record_stage(stages, "POLICY_SERVER_READY", host=host, port=port)

        # Seed scene construction without importing Warp before Kit has
        # selected its own omni.warp.core module.
        _require_warp_absent_before_kit()
        _seed_before_kit_without_warp(
            worker.seed,
            physical_gpu=worker.physical_gpu,
        )
        _require_warp_absent_before_kit()

        # Official runtime imports are intentionally confined to the live run.
        # Configure global timing before importing Evaluator: importing the
        # evaluator transitively defines Environment.default_config and freezes
        # its action/render frequencies.
        from omegaconf import OmegaConf
        from omnigibson.macros import gm, macros

        if os.environ.get("ROBOWEAVE_EXPERIMENTAL_HOUSEHOLD_ACTIONS") == "1":
            with gm.unlocked():
                gm.USE_GPU_DYNAMICS = True

        gm.HEADLESS = True
        checkpoint_timing = configure_preconstruction_checkpoint_frequency(
            global_macros=gm,
            macros_module=macros,
        )
        import omnigibson.eval.evaluator as evaluator_module

        if evaluator_module_configurer is not None:
            evaluator_module_cleanup = evaluator_module_configurer(
                evaluator_module
            )
            if not callable(evaluator_module_cleanup):
                raise B1SingleLoadRunnerError(
                    "evaluator module configurer must return cleanup"
                )

        DEFAULT_ROBOT_CONFIG_PATH = evaluator_module.DEFAULT_ROBOT_CONFIG_PATH
        Evaluator = evaluator_module.Evaluator
        from omnigibson.eval.utils.eval_utils import (
            HEAD_RESOLUTION,
            WRIST_RESOLUTION,
        )
        _write_json(
            run_dir / "private" / "preconstruction_checkpoint_timing.json",
            checkpoint_timing,
        )
        _record_stage(
            stages,
            "PRECONSTRUCTION_CHECKPOINT_FREQUENCY_PASS",
            **dict(checkpoint_timing),
        )

        expected_appdata = Path(
            os.environ["OMNIGIBSON_APPDATA_PATH"]
        ).resolve()
        if Path(gm.APPDATA_PATH).resolve() != expected_appdata:
            raise B1SingleLoadRunnerError(
                "OmniGibson appdata escaped worker isolation"
            )
        robot_config = build_predeclared_r1pro_camera_robot_config(
            DEFAULT_ROBOT_CONFIG_PATH,
            head_resolution=HEAD_RESOLUTION,
            wrist_resolution=WRIST_RESOLUTION,
            camera_profile=camera_profile,
            use_wrist_evidence_reset_pose=(
                use_wrist_evidence_reset_pose
            ),
            wrist_focal_length_mm=(
                WRIST_FOCAL_LENGTH_MM
                if selected_gripper_visual_profile["camera_tuning"]
                else None
            ),
            head_focal_length_mm=(
                HEAD_FOCAL_LENGTH_MM
                if selected_gripper_visual_profile["camera_tuning"]
                else None
            ),
        )
        # The pre-replacement official YAML arm slices are a private side
        # channel for the arm-presentation ladder's stowed endpoint.  Pop
        # them before Evaluator construction so the live robot config stays
        # byte-identical to the historical contract.
        original_reset_arm_joints = robot_config.pop(
            ORIGINAL_RESET_ARM_JOINTS_CONFIG_KEY,
            None,
        )
        if original_reset_arm_joints is None:
            official_reset_joint_pos = robot_config.get("reset_joint_pos")
            if (
                isinstance(official_reset_joint_pos, list)
                and len(official_reset_joint_pos) == 28
            ):
                original_reset_arm_joints = {
                    "left": [
                        float(value)
                        for value in official_reset_joint_pos[10:17]
                    ],
                    "right": [
                        float(value)
                        for value in official_reset_joint_pos[17:24]
                    ],
                }
        if (
            selected_gripper_visual_profile["arm_motion_stow_ladder"]
            and original_reset_arm_joints is None
        ):
            raise B1SingleLoadRunnerError(
                "arm presentation ladder requires the official 28-joint "
                "reset vector for its stowed endpoint"
            )
        env_wrapper_config = {
            "_target_": selected_camera_profile["wrapper"],
        }
        if install_private_head_segmentation_side_channel:
            env_wrapper_config[
                "install_private_head_segmentation_side_channel"
            ] = True
        cfg = OmegaConf.create(
            {
                "env_wrapper": env_wrapper_config,
                "policy_name": "behavior_hl_b1_single_load",
                "model": {
                    "_target_": (
                        "behavior_hl.embodied.evaluator_bridge."
                        "whitelist_policy.WhitelistedWebsocketPolicy"
                    ),
                    "host": host,
                    "port": port,
                    "allow_reconnect": False,
                    "observation_profile": camera_profile,
                    "expected_task_id": expected_task_id,
                },
                "headless": True,
                "partial_scene_load": effective_partial_scene_load,
                "max_steps": None,
                "write_video": selected_camera_profile["write_video"],
                "mode": worker.split,
                "seed": worker.seed,
                "task": {"name": worker.task_name},
                "robot": robot_config,
            }
        )

        scene_loads_attempted = 1
        _record_stage(
            stages,
            "SCENE_COVERAGE_CONTRACT_PASS",
            scene_coverage=(
                "task_relevant_partial"
                if effective_partial_scene_load
                else "full"
            ),
            partial_scene_load=effective_partial_scene_load,
            goal_completion_only=True,
        )
        _record_stage(
            stages,
            "SCENE_CONSTRUCTION_START",
            scene_load_number=1,
        )
        with temporary_full_scene_robot_filter(
            evaluator_module,
            enabled=filter_embedded_scene_agents,
        ) as full_scene_robot_filter_report, temporary_evaluator_scene_file_override(
            evaluator_module,
            scene_file=resident_scene_file,
            expected_scene_model=resident_scene_model,
        ) as resident_scene_override_report:
            evaluator_context = Evaluator(cfg)
        require_full_scene_robot_filter_report(
            full_scene_robot_filter_report,
            enabled=filter_embedded_scene_agents,
        )
        _write_json(
            run_dir / "private" / "full_scene_robot_filter.json",
            full_scene_robot_filter_report,
        )
        _write_json(
            run_dir / "private" / "resident_scene_override.json",
            resident_scene_override_report,
        )
        _record_stage(
            stages,
            "FULL_SCENE_ROBOT_FILTER_PASS",
            **dict(full_scene_robot_filter_report),
            wrist_evidence_reset_pose=use_wrist_evidence_reset_pose,
        )
        with nullcontext(evaluator_context.__enter__()) as evaluator:
            successful_scene_loads = 1
            camera_layout_profile = (
                HEAD_DOMINANT_CAMERA_LAYOUT_PROFILE
                if selected_gripper_visual_profile[
                    "head_dominant_layout"
                ]
                else NATIVE_CAMERA_LAYOUT_PROFILE
            )
            setattr(
                evaluator,
                "_behavior_hl_camera_layout_profile",
                camera_layout_profile,
            )
            _record_stage(
                stages,
                "SCENE_CONSTRUCTION_PASS",
                scene_load_number=1,
            )

            # Kit now owns Warp.  Attest one module family and the selected
            # CUDA ordinal without resetting RNG streams consumed by scene
            # construction.
            warp_attestation = _attest_kit_warp_runtime(
                physical_gpu=worker.physical_gpu,
            )
            _write_json(
                run_dir / "private" / "warp_runtime_attestation.json",
                warp_attestation,
            )
            _record_stage(
                stages,
                "WARP_RUNTIME_ATTESTATION_PASS",
                physical_gpu=worker.physical_gpu,
                cuda_device_count=warp_attestation[
                    "cuda_device_count"
                ],
                selected_alias=warp_attestation["selected_alias"],
                selected_uuid=warp_attestation["selected_uuid"],
                selected_pci_bus_id=warp_attestation[
                    "selected_pci_bus_id"
                ],
            )
            camera_attestation = getattr(
                evaluator.env,
                "predeclared_camera_attestation",
                None,
            )
            if not isinstance(camera_attestation, Mapping):
                raise B1SingleLoadRunnerError(
                    "predeclared camera structure attestation is unavailable"
                )
            _write_json(
                run_dir
                / "private"
                / "predeclared_camera_structure_attestation.json",
                dict(camera_attestation),
            )
            _record_stage(
                stages,
                "PREDECLARED_CAMERA_STRUCTURE_ATTESTATION_PASS",
                camera_profile=selected_camera_profile["camera_profile"],
                camera_roles=sorted(camera_attestation["cameras"]),
            )
            _record_stage(
                stages,
                "CAMERA_PARAMETER_ANNOTATORS_PREWARM_START",
            )
            if camera_profile == FULL_CAMERA_PROFILE:
                # Preserve the established full-camera call path byte-for-byte:
                # this remains the default QA / full-observation behavior.
                camera_parameter_attestation = (
                    prewarm_evaluator_camera_parameters(evaluator)
                )
            else:
                camera_parameter_attestation = (
                    prewarm_evaluator_camera_parameters(
                        evaluator,
                        camera_roles=frozenset(
                            selected_camera_profile["camera_roles"]
                        ),
                    )
                )
            _write_json(
                run_dir
                / "private"
                / "camera_parameter_annotators_prewarm.json",
                camera_parameter_attestation,
            )
            _record_stage(
                stages,
                "CAMERA_PARAMETER_ANNOTATORS_PREWARM_PASS",
                camera_profile=selected_camera_profile["camera_profile"],
                camera_roles=sorted(
                    camera_parameter_attestation["cameras"]
                ),
            )
            _initialize_loaded_instance(
                evaluator,
                stages,
                instance_id=worker.instance_id,
                native_task_metric=native_task_metric,
            )
            # Apply presentation only after the official reset/load/reset
            # sequence so neither the tuned sensor transform nor the planned
            # arm pose is silently replaced by task-instance restoration.
            if selected_gripper_visual_profile["camera_tuning"]:
                _record_stage(
                    stages,
                    "GRIPPER_VISIBLE_CAMERA_TUNING_START",
                )
                camera_tuning_report = apply_gripper_visible_camera_tuning(
                    evaluator
                )
                _record_stage(
                    stages,
                    "GRIPPER_VISIBLE_CAMERA_TUNING_PASS",
                    camera_count=camera_tuning_report["camera_count"],
                )
            else:
                camera_tuning_report = {
                    **dict(gripper_visible_camera_tuning_contract()),
                    "applied": False,
                    "camera_count": 0,
                    "cameras": {},
                }
            _write_json(
                run_dir
                / "private"
                / "gripper_visible_camera_tuning.json",
                camera_tuning_report,
            )
            if (
                selected_gripper_visual_profile["inward_arm_presentation"]
                and defer_inward_arm_presentation
            ):
                # Model lanes admit the Oracle from the untouched official
                # reset state (v186-proven); the approved inward presentation
                # is applied by the rollout callback after the reset
                # attestation, immediately before the first model frame.
                gripper_presentation_report = {
                    "version": MODERATE_INWARD_GRIPPER_PRESENTATION_VERSION,
                    "status": "DEFERRED_TO_EXECUTION_PHASE",
                    "presentation_only": True,
                    "direct_kinematic_joint_application": False,
                }
                _record_stage(
                    stages,
                    "MODERATE_INWARD_GRIPPER_PRESENTATION_DEFERRED",
                )
            elif selected_gripper_visual_profile[
                "inward_arm_presentation"
            ]:
                _record_stage(
                    stages,
                    "MODERATE_INWARD_GRIPPER_PRESENTATION_START",
                )
                # Without the ladder flag this helper calls the presentation
                # directly and every failure propagates exactly as before.
                gripper_presentation_report = (
                    apply_load_time_inward_presentation_with_ladder_grading(
                        apply_presentation=lambda: (
                            apply_moderate_inward_gripper_presentation(
                                evaluator
                            )
                        ),
                        arm_motion_stow_ladder=bool(
                            selected_gripper_visual_profile[
                                "arm_motion_stow_ladder"
                            ]
                        ),
                    )
                )
                if (
                    gripper_presentation_report["status"]
                    == DEGRADED_TO_UNTUCKED_STATUS
                ):
                    _record_stage(
                        stages,
                        "MODERATE_INWARD_GRIPPER_PRESENTATION_DEGRADED",
                        error_type=gripper_presentation_report[
                            "error_type"
                        ],
                    )
                else:
                    _record_stage(
                        stages,
                        "MODERATE_INWARD_GRIPPER_PRESENTATION_PASS",
                        selected_shift_m=gripper_presentation_report[
                            "selected_shift_m"
                        ],
                    )
            else:
                gripper_presentation_report = {
                    "version": MODERATE_INWARD_GRIPPER_PRESENTATION_VERSION,
                    "status": "DISABLED",
                    "presentation_only": True,
                    "direct_kinematic_joint_application": False,
                }
            _write_json(
                run_dir
                / "private"
                / "moderate_inward_gripper_presentation.json",
                gripper_presentation_report,
            )
            if selected_gripper_visual_profile["arm_motion_stow_ladder"]:
                # Thread the ladder endpoints and the profile flag into the
                # rollout phases through the same private evaluator-attribute
                # channel already used by the camera layout profile.  A
                # deferred or degraded presentation resolves to the official
                # untucked constants; a later model-phase presentation may
                # rebind the presented endpoints on this same attribute.
                arm_presentation_ladder_config = (
                    build_arm_presentation_ladder_config(
                        profile_name=selected_gripper_visual_profile[
                            "name"
                        ],
                        presentation_report=gripper_presentation_report,
                        stowed_left_arm_qpos=original_reset_arm_joints[
                            "left"
                        ],
                        stowed_right_arm_qpos=original_reset_arm_joints[
                            "right"
                        ],
                    )
                )
                setattr(
                    evaluator,
                    ARM_PRESENTATION_LADDER_CONFIG_ATTRIBUTE,
                    arm_presentation_ladder_config,
                )
                _write_json(
                    run_dir
                    / "private"
                    / "arm_presentation_ladder_config.json",
                    {
                        key: (
                            list(value)
                            if isinstance(value, tuple)
                            else value
                        )
                        for key, value in (
                            arm_presentation_ladder_config.items()
                        )
                    },
                )
                _record_stage(
                    stages,
                    "ARM_PRESENTATION_LADDER_CONFIG_PASS",
                    presented_source=arm_presentation_ladder_config[
                        "presented_source"
                    ],
                )
            task_instance_identity = _capture_loaded_task_instance_identity(
                evaluator=evaluator,
                mode=worker.split,
                instance_id=worker.instance_id,
            )
            _write_json(
                run_dir / "private" / "task_instance_identity.json",
                task_instance_identity,
            )
            _record_stage(
                stages,
                "CAPTURE_PRE_CALIBRATION_RESET_STATE_START",
            )
            pre_reset = bindings.capture_private_reset_snapshot(
                evaluator=evaluator,
                phase="pre_calibration",
                run_dir=run_dir,
            )
            if (
                not isinstance(pre_reset, PrivateResetSnapshot)
                or pre_reset.phase != "pre_calibration"
            ):
                raise B1SingleLoadRunnerError(
                    "pre-calibration reset snapshot is invalid"
                )
            _write_json(
                run_dir
                / "private"
                / "reset_attestation"
                / "pre_calibration.json",
                pre_reset.to_private_record(include_stable_state=True),
            )
            _record_stage(
                stages,
                "CAPTURE_PRE_CALIBRATION_RESET_STATE_PASS",
            )

            calibration_video = (
                run_dir / "private" / "calibration_rollout.mp4"
            )
            evaluator.start_recording(
                str(calibration_video),
                rate=int(run_config["runtime"]["video_fps"]),
            )
            recording = True
            if strategy_sweep_callback is not None:
                _record_stage(stages, "SAME_SCENE_STRATEGY_SWEEP_START")
                strategy_summary = strategy_sweep_callback(
                    evaluator=evaluator,
                    control=control,
                    bindings=bindings,
                    run_dir=run_dir,
                    stages=stages,
                )
                if not isinstance(strategy_summary, Mapping):
                    raise B1SingleLoadRunnerError(
                        "strategy sweep callback must return a mapping"
                    )
                _write_json(
                    run_dir / "private" / "strategy_sweep_summary.json",
                    dict(strategy_summary),
                )
                _record_stage(
                    stages,
                    "SAME_SCENE_STRATEGY_SWEEP_PASS",
                    strategy_count=len(strategy_summary.get("strategies", ())),
                )
                # Keep the ordinary episode lifecycle marker unique for the
                # static protocol test; this diagnostic branch is a separate
                # same-scene sweep finalizer.
                getattr(evaluator, "stop_recording")()
                recording = False
                resident_hold_requested = _resident_scene_hold_requested()
                resident_reset_attestation = None
                if resident_hold_requested:
                    resident_reset_attestation = (
                        _prepare_resident_scene_for_hold(
                            bindings=bindings,
                            evaluator=evaluator,
                            control=control,
                            pre_reset=pre_reset,
                            reset_tolerances=reset_tolerances,
                            run_dir=run_dir,
                            stages=stages,
                        )
                    )
                source_freeze_after = capture_source_freeze(PROJECT_ROOT)
                _write_json(
                    run_dir / "private" / "source_freeze_after_run.json",
                    source_freeze_after,
                )
                require_unchanged_source_freeze(
                    before=source_freeze,
                    after=source_freeze_after,
                )
                official_repository_after = (
                    capture_official_repository_identity(BEHAVIOR_ROOT)
                )
                _write_json(
                    run_dir / "private" / "official_repository_after_run.json",
                    official_repository_after,
                )
                require_unchanged_official_repository_identity(
                    before=official_repository_before,
                    after=official_repository_after,
                )
                final_manifest = {
                    **initial_manifest,
                    "status": (
                        "RESIDENT_READY"
                        if resident_hold_requested
                        else "COMPLETE"
                    ),
                    "strategy_sweep": True,
                    "camera_profile": selected_camera_profile[
                        "camera_profile"
                    ],
                    "write_video": bool(
                        selected_camera_profile["write_video"]
                    ),
                    # A diagnostic callback may deliberately use the official
                    # Symbolic primitive set.  Never leave the generic
                    # physical-run default in the completed manifest when the
                    # callback explicitly attests that kinematic teleport was
                    # enabled inside that primitive.
                    "teleport_enabled": bool(
                        strategy_summary.get(
                            "teleport_enabled_inside_symbolic_primitive",
                            initial_manifest["teleport_enabled"],
                        )
                    ),
                    "execution_fidelity": (
                        "official_symbolic_kinematic_research"
                        if strategy_summary.get(
                            "teleport_enabled_inside_symbolic_primitive"
                        )
                        else "physical_r1pro_research"
                    ),
                    "scene_loads_attempted": scene_loads_attempted,
                    "successful_scene_loads": successful_scene_loads,
                    "in_memory_strategy_resets": int(
                        strategy_summary.get("in_memory_resets", 0)
                    ),
                    "resident_scene_hold": resident_hold_requested,
                    "resident_ready_state_restore": bool(
                        resident_reset_attestation
                        and resident_reset_attestation.passed
                    ),
                    "source_freeze_revalidated": True,
                    "official_repository_revalidated": True,
                    "official_behavior_head": official_repository_after[
                        "head"
                    ],
                    "strategy_summary": dict(strategy_summary),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "source_files": {
                        "behavior_head": official_repository_after["head"],
                        "runner_sha256": _sha256(Path(__file__)),
                    },
                }
                _write_json(run_dir / "manifest.json", final_manifest)
                if resident_hold_requested:
                    _record_stage(
                        stages,
                        "RESULTS_PERSISTED_RESIDENT_READY",
                        status="RESIDENT_READY",
                    )
                    resident_release = _hold_resident_scene_until_release(
                        evaluator=evaluator,
                        control=control,
                        run_dir=run_dir,
                        stages=stages,
                        physical_gpu=worker.physical_gpu,
                        strategy_summary=strategy_summary,
                    )
                    final_manifest = {
                        **final_manifest,
                        "status": "COMPLETE",
                        "resident_release": dict(resident_release),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }
                    _write_json(run_dir / "manifest.json", final_manifest)
                _record_stage(
                    stages,
                    "RESULTS_PERSISTED_BEFORE_EVALUATOR_EXIT",
                    status="COMPLETE",
                )
                return B1StrategySweepRunResult(
                    run_dir=run_dir,
                    summary=dict(strategy_summary),
                    scene_loads_attempted=scene_loads_attempted,
                    successful_scene_loads=successful_scene_loads,
                )
            (
                approach,
                proposal,
                bootstrap,
                grasp_lift,
                safe_carry,
                formal_wrist,
            ) = (
                _run_calibration_lifecycle(
                    evaluator=evaluator,
                    control=control,
                    bindings=bindings,
                    run_dir=run_dir,
                    stages=stages,
                    limits=limits,
                    minimum_empty_trials=int(
                        contract["empty_close_minimum_trials"]
                    ),
                    maximum_grasp_base_position_drift_m=float(
                        contract[
                            "maximum_grasp_base_position_drift_m"
                        ]
                    ),
                    maximum_grasp_base_orientation_drift_rad=float(
                        contract[
                            "maximum_grasp_base_orientation_drift_rad"
                        ]
                    ),
                )
            )
            _record_stage(
                stages,
                "AUTHOR_AND_FREEZE_SINGLE_LEAF_WITNESS_START",
            )
            witness = bindings.author_and_freeze_single_leaf_witness(
                evaluator=evaluator,
                control=control,
                approach=approach,
                formal_wrist=formal_wrist,
                safe_carry=safe_carry,
                run_dir=run_dir,
            )
            if not isinstance(witness, PublicSingleLeafWitness):
                raise B1SingleLoadRunnerError(
                    "witness author returned the wrong type"
                )
            if witness.calibrated_hand != safe_carry.candidate.hand:
                raise B1SingleLoadRunnerError(
                    "witness changed the calibrated carry hand"
                )
            witness_registry = build_picking_up_toys_registry(
                calibrated_hands=(safe_carry.candidate.hand,),
            )
            witness.validate_against(witness_registry)
            _write_json_atomic(
                run_dir
                / "private"
                / "witness"
                / "public_single_leaf_witness.json",
                witness.to_manifest(),
            )
            _record_stage(
                stages,
                "AUTHOR_AND_FREEZE_SINGLE_LEAF_WITNESS_PASS",
                witness_sha256=witness.witness_sha256,
                witness_author=witness.witness_author,
                decision_count=len(witness.decisions),
            )
            evaluator.stop_recording()
            recording = False

            control.hold()
            _record_stage(stages, "SAME_SCENE_EPISODE_RESET_START")
            evaluator.reset()
            _record_stage(stages, "SAME_SCENE_EPISODE_RESET_PASS")
            _record_stage(
                stages,
                "CAPTURE_POST_RESET_STATE_START",
            )
            post_reset = bindings.capture_private_reset_snapshot(
                evaluator=evaluator,
                phase="post_reset",
                run_dir=run_dir,
            )
            if (
                not isinstance(post_reset, PrivateResetSnapshot)
                or post_reset.phase != "post_reset"
            ):
                raise B1SingleLoadRunnerError(
                    "post-reset private snapshot is invalid"
                )
            _write_json(
                run_dir
                / "private"
                / "reset_attestation"
                / "post_reset.json",
                post_reset.to_private_record(include_stable_state=True),
            )
            _record_stage(stages, "CAPTURE_POST_RESET_STATE_PASS")
            reset_attestation = attest_same_scene_reset(
                pre=pre_reset,
                post=post_reset,
                tolerances=reset_tolerances,
            )
            _write_json(
                run_dir
                / "private"
                / "reset_attestation"
                / "attestation.json",
                reset_attestation.to_private_record(),
            )
            reset_attestation.require_pass()
            _record_stage(
                stages,
                "PRIVATE_RESET_ATTESTATION_PASS",
            )

            # No post-reset runtime object may be constructed before the
            # private restoration attestation above passes.
            plan = bindings.build_post_reset_plan(
                evaluator=evaluator,
                control=control,
                formal_wrist=formal_wrist,
                safe_carry=safe_carry,
                witness=witness,
                run_dir=run_dir,
            )
            if not isinstance(plan, B1PostResetAssemblyPlan):
                raise B1SingleLoadRunnerError(
                    "build_post_reset_plan returned the wrong type"
                )
            assembled = _assemble_resident_episode(
                evaluator=evaluator,
                control=control,
                plan=plan,
                safe_carry=safe_carry,
                run_dir=run_dir,
                stages=stages,
                video_fps=int(run_config["runtime"]["video_fps"]),
            )

            assembled.recording_artifacts.start()
            _record_stage(stages, "RUN_B1_RESIDENT_EPISODE_START")
            try:
                episode_result = assembled.bundle.orchestrator.run()
            finally:
                assembled.recording_artifacts.stop_if_active()
            _record_stage(
                stages,
                "RUN_B1_RESIDENT_EPISODE_COMPLETE",
                termination=(
                    episode_result.termination.classification.value
                ),
                action_count=episode_result.action_count,
                model_calls=episode_result.model_calls,
                simulator_steps=episode_result.simulator_steps,
                artifact_complete=episode_result.artifact_complete,
            )
            source_freeze_after = capture_source_freeze(PROJECT_ROOT)
            _write_json(
                run_dir / "private" / "source_freeze_after_run.json",
                source_freeze_after,
            )
            require_unchanged_source_freeze(
                before=source_freeze,
                after=source_freeze_after,
            )
            official_repository_after = (
                capture_official_repository_identity(BEHAVIOR_ROOT)
            )
            _write_json(
                run_dir / "private" / "official_repository_after_run.json",
                official_repository_after,
            )
            require_unchanged_official_repository_identity(
                before=official_repository_before,
                after=official_repository_after,
            )
            _record_stage(
                stages,
                "SOURCE_FREEZE_AFTER_RUN_PASS",
                aggregate_sha256=source_freeze_after[
                    "aggregate_sha256"
                ],
            )
            _require_replay_single_leaf_completion(
                episode_result=episode_result,
                model=assembled.model,
                run_dir=run_dir,
            )
            _record_stage(
                stages,
                "REPLAY_SINGLE_LEAF_COMPLETION_GATE_PASS",
            )

            final_manifest = {
                **initial_manifest,
                "status": "COMPLETE",
                "scene_loads_attempted": scene_loads_attempted,
                "successful_scene_loads": successful_scene_loads,
                "post_calibration_in_memory_resets": 1,
                "same_scene_reset_attestation_passed": True,
                "source_freeze_revalidated": True,
                "official_repository_revalidated": True,
                "official_behavior_head": official_repository_after["head"],
                "task_instance_file_sha256": (
                    task_instance_identity["sha256"]
                ),
                "approach_target_token_sha256": hashlib.sha256(
                    approach.target_token.encode("utf-8")
                ).hexdigest(),
                "calibration_candidate_sha256": (
                    safe_carry_candidate_sha256(proposal.candidate)
                ),
                "bootstrap_wrist_calibration_sha256": (
                    bootstrap.calibration_sha256
                ),
                "public_positive_window_sha256": (
                    grasp_lift.positive_window.window_sha256
                ),
                "safe_carry_calibration_sha256": safe_carry.sha256,
                "formal_wrist_calibration_sha256": (
                    formal_wrist.calibration_sha256
                ),
                "public_single_leaf_witness_sha256": (
                    witness.witness_sha256
                ),
                "witness_author": witness.witness_author,
                "witness_decision_count": len(witness.decisions),
                "episode_termination": (
                    episode_result.termination.classification.value
                ),
                "episode_artifact_complete": (
                    episode_result.artifact_complete
                ),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "source_files": {
                    "behavior_head": official_repository_after["head"],
                    "runner_sha256": _sha256(Path(__file__)),
                },
            }
            _write_json(run_dir / "manifest.json", final_manifest)
            _record_stage(
                stages,
                "RESULTS_PERSISTED_BEFORE_EVALUATOR_EXIT",
            )
            return B1SingleLoadRunResult(
                run_dir=run_dir,
                calibration=safe_carry,
                formal_wrist=formal_wrist,
                witness=witness,
                episode=episode_result,
            )
    except Exception as exc:
        _record_stage(
            stages,
            "PYTHON_EXCEPTION",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        _write_json(
            run_dir / "failure.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        _write_json(
            run_dir / "manifest.json",
            {
                **initial_manifest,
                "status": "FAIL",
                "scene_loads_attempted": scene_loads_attempted,
                "successful_scene_loads": successful_scene_loads,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    finally:
        pending_exc_info = sys.exc_info()
        cleanup_failures = _cleanup_live_resources(
            control=control,
            recording=recording,
            evaluator_context=evaluator_context,
            server_process=server_process,
            evaluator_exit_exc_info=pending_exc_info,
        )
        if evaluator_module_cleanup is not None:
            try:
                evaluator_module_cleanup()
            except BaseException as exc:
                cleanup_failures.append(
                    {
                        "operation": "evaluator_module_registration.cleanup",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        if cleanup_failures:
            try:
                _write_json(
                    run_dir / "private" / "cleanup_failures.json",
                    {
                        "visibility": "private",
                        "failures": cleanup_failures,
                    },
                )
            except Exception:
                pass
            if pending_exc_info[0] is None:
                raise B1SingleLoadRunnerError(
                    "live resource cleanup failed: "
                    + ", ".join(
                        item["operation"] for item in cleanup_failures
                    )
                )


def _run_zero_scene_preflight(
    *,
    run_dir: Path,
    run_id: str,
    stages: Path,
    contract: Mapping[str, Any],
    reset_tolerances: SameSceneResetTolerances,
    worker: B1SingleLoadWorkerSpec,
    camera_profile: str = FULL_CAMERA_PROFILE,
    gripper_visual_profile: str = GRIPPER_VISUAL_PROFILE_DISABLED,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Freeze launch/source/official identity before any process starts."""

    _configure_process_environment(
        run_dir,
        physical_gpu=worker.physical_gpu,
    )
    _record_stage(stages, "RUN_DIRECTORY_READY")
    launch_contract = _validate_launch_contract(
        physical_gpu=worker.physical_gpu,
    )
    _write_json(
        run_dir / "private" / "launch_contract.json",
        launch_contract,
    )
    _record_stage(stages, "ZERO_SCENE_LAUNCH_CONTRACT_PASS")
    project_identity = capture_project_identity(
        expected_root=PROJECT_ROOT,
        module_files={"run_b1_single_load": Path(__file__)},
        require_environment=True,
    )
    _write_json(
        run_dir / "private" / "project_identity_before_scene.json",
        project_identity,
    )
    source_freeze = capture_source_freeze(PROJECT_ROOT)
    _write_json(
        run_dir / "private" / "source_freeze_before_scene.json",
        source_freeze,
    )
    official_repository_before = capture_official_repository_identity(
        BEHAVIOR_ROOT
    )
    _write_json(
        run_dir / "private" / "official_repository_before_scene.json",
        official_repository_before,
    )
    _write_json(
        run_dir
        / "private"
        / "reset_attestation"
        / "contract.json",
        {
            "visibility": "private",
            "comparison": "tolerant_canonical_v1",
            "full_state_exact_hash_claimed": False,
            "tolerances": {
                name: getattr(reset_tolerances, name)
                for name in reset_tolerances.__dataclass_fields__
            },
            "official_source_refs": dict(
                contract["official_reset_semantics_source_refs"]
            ),
        },
    )
    _record_stage(
        stages,
        "SOURCE_FREEZE_BEFORE_SCENE_PASS",
        aggregate_sha256=source_freeze["aggregate_sha256"],
        file_count=source_freeze["file_count"],
    )
    initial_manifest = {
        "schema_version": B1_SINGLE_LOAD_RUNNER_VERSION,
        "run_id": run_id,
        "status": "RUNNING",
        "worker": worker.worker_id,
        "task": worker.task_name,
        "split": worker.split,
        "instance_id": worker.instance_id,
        "seed": worker.seed,
        "physical_gpu": worker.physical_gpu,
        "server_port": worker.server_port,
        "scene_load_budget": 1,
        "scene_loads_attempted": 0,
        "successful_scene_loads": 0,
        "evaluator_instances": 1,
        "resident_single_scene": True,
        "automatic_retry": False,
        "hot_patch": True,
        "hot_patch_scope": (
            "scoped_curobo_and_r1pro_eval_controller_compatibility_v3"
        ),
        "viewer_camera_control": False,
        "teleport_enabled": False,
        "privileged_policy_inputs": False,
        "official_challenge_eligible": False,
        "camera_profile": camera_profile,
        "gripper_visual_profile": gripper_visual_profile,
        "websocket_keepalive": oracle_websocket_keepalive_manifest(),
        "effective_wrapper": camera_profile_contract(camera_profile)[
            "wrapper"
        ],
        "write_video": bool(
            camera_profile_contract(camera_profile)["write_video"]
        ),
        "config_sha256": _sha256(CONFIG_PATH),
        "single_load_contract_sha256": _sha256(
            B1_SINGLE_LOAD_CONTRACT_PATH
        ),
        "observation_schema_sha256": schema_sha256(),
        "source_freeze_aggregate_sha256": (
            source_freeze["aggregate_sha256"]
        ),
        "source_freeze_file_count": source_freeze["file_count"],
        "source_freeze_manifest": (
            "private/source_freeze_before_scene.json"
        ),
        "project_root": project_identity["project_root"],
        "project_identity_manifest": (
            "private/project_identity_before_scene.json"
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "manifest.json", initial_manifest)
    return (
        source_freeze,
        official_repository_before,
        initial_manifest,
    )


def _write_preflight_failure(
    *,
    run_dir: Path,
    run_id: str,
    stages: Path,
    exc: BaseException,
) -> None:
    """Best-effort failure artifacts for every post-run-dir preflight error."""

    try:
        _record_stage(
            stages,
            "ZERO_SCENE_PREFLIGHT_FAILURE",
            error_type=type(exc).__name__,
            message=str(exc),
        )
    except Exception:
        pass
    payload = {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "phase": "zero_scene_preflight",
    }
    try:
        _write_json(run_dir / "failure.json", payload)
    except Exception:
        pass
    try:
        _write_json(
            run_dir / "manifest.json",
            {
                "schema_version": B1_SINGLE_LOAD_RUNNER_VERSION,
                "run_id": run_id,
                "status": "FAIL",
                "phase": "zero_scene_preflight",
                "websocket_keepalive": oracle_websocket_keepalive_manifest(),
                "scene_loads_attempted": 0,
                "successful_scene_loads": 0,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception:
        pass


def _cleanup_live_resources(
    *,
    control: Any,
    recording: bool,
    evaluator_context: Any,
    server_process: Any,
    evaluator_exit_exc_info: tuple[Any, Any, Any],
) -> list[dict[str, str]]:
    """Attempt every shutdown operation even when an earlier one fails."""

    failures: list[dict[str, str]] = []

    def attempt(operation: str, callback: Callable[[], Any]) -> None:
        try:
            callback()
        except BaseException as exc:
            failures.append(
                {
                    "operation": operation,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    attempt("control.hold", control.hold)
    if recording and evaluator_context is not None:
        attempt(
            "evaluator.stop_recording",
            evaluator_context.stop_recording,
        )

    def terminate_server_if_alive() -> None:
        if server_process.is_alive():
            server_process.terminate()

    attempt("policy_server.terminate", terminate_server_if_alive)
    attempt("policy_server.join", lambda: server_process.join(timeout=10))
    if evaluator_context is not None:
        attempt(
            "evaluator.__exit__",
            lambda: evaluator_context.__exit__(
                *evaluator_exit_exc_info
            ),
        )
    return failures


def _resident_scene_hold_requested() -> bool:
    value = os.environ.get("BEHAVIOR_HL_RESIDENT_SCENE_HOLD", "0")
    if value not in {"0", "1"}:
        raise B1SingleLoadRunnerError(
            "BEHAVIOR_HL_RESIDENT_SCENE_HOLD must be 0 or 1"
        )
    return value == "1"


def _reset_evaluator_for_resident_hold(value: Any) -> None:
    reset = getattr(value, "reset", None)
    if not callable(reset):
        raise B1SingleLoadRunnerError(
            "resident scene evaluator reset is unavailable"
        )
    reset()


def _prepare_resident_scene_for_hold(
    *,
    bindings: Any,
    evaluator: Any,
    control: Any,
    pre_reset: PrivateResetSnapshot,
    reset_tolerances: SameSceneResetTolerances,
    run_dir: Path,
    stages: Path,
) -> Any:
    """Restore and attest the loaded task state before an idle hold."""

    control.hold()
    _record_stage(stages, "RESIDENT_SCENE_PREPARE_RESET_START")
    _reset_evaluator_for_resident_hold(evaluator)
    _record_stage(stages, "RESIDENT_SCENE_PREPARE_RESET_PASS")
    resident_reset = bindings.capture_private_reset_snapshot(
        evaluator=evaluator,
        phase="post_reset",
        run_dir=run_dir,
    )
    resident_reset_attestation = attest_same_scene_reset(
        pre=pre_reset,
        post=resident_reset,
        tolerances=reset_tolerances,
    )
    _write_json(
        run_dir
        / "private"
        / "reset_attestation"
        / "resident_ready.json",
        resident_reset_attestation.to_private_record(),
    )
    resident_reset_attestation.require_pass()
    _record_stage(stages, "RESIDENT_SCENE_RESET_ATTESTATION_PASS")
    return resident_reset_attestation


def _hold_resident_scene_until_release(
    *,
    evaluator: Any,
    control: Any,
    run_dir: Path,
    stages: Path,
    physical_gpu: int,
    strategy_summary: Mapping[str, Any],
    poll_seconds: float = 5.0,
) -> Mapping[str, Any]:
    """Keep one already-loaded evaluator alive until an owned release file."""

    if poll_seconds <= 0.0 or not math.isfinite(poll_seconds):
        raise ValueError("resident hold poll_seconds must be positive")
    maximum_seconds = float(
        os.environ.get("BEHAVIOR_HL_RESIDENT_SCENE_MAX_SECONDS", "604800")
    )
    if maximum_seconds <= 0.0 or not math.isfinite(maximum_seconds):
        raise B1SingleLoadRunnerError(
            "resident scene maximum seconds must be positive"
        )
    resident_dir = run_dir / "resident"
    resident_dir.mkdir(parents=True, exist_ok=False)
    release_path = resident_dir / "release.request"
    heartbeat_path = resident_dir / "heartbeat.json"
    ready_path = resident_dir / "ready.json"
    started_monotonic = time.monotonic()
    ready = {
        "schema_version": "behavior2026.resident-scene-hold/v1",
        "status": "READY",
        "pid": os.getpid(),
        "physical_gpu": int(physical_gpu),
        "run_dir": str(run_dir),
        "release_file": str(release_path),
        "scene_load_count": 1,
        "state_restored_before_hold": True,
        "oracle_success": bool(strategy_summary.get("oracle_success")),
        "g0_success": bool(strategy_summary.get("g0_success")),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "maximum_hold_seconds": maximum_seconds,
    }
    _write_json(ready_path, ready)
    _record_stage(
        stages,
        "RESIDENT_SCENE_HOLD_READY",
        pid=os.getpid(),
        physical_gpu=int(physical_gpu),
        release_file=str(release_path),
    )
    control.hold()
    while True:
        elapsed = time.monotonic() - started_monotonic
        heartbeat = {
            **ready,
            "status": "READY" if not release_path.exists() else "RELEASING",
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "owned_process_alive": True,
        }
        _write_json(heartbeat_path, heartbeat)
        if release_path.exists():
            released = {
                **heartbeat,
                "status": "RELEASED",
                "released_at": datetime.now(timezone.utc).isoformat(),
                "release_reason": "owned release.request observed",
            }
            _write_json(resident_dir / "released.json", released)
            _record_stage(stages, "RESIDENT_SCENE_HOLD_RELEASED")
            return released
        if elapsed >= maximum_seconds:
            expired = {
                **heartbeat,
                "status": "EXPIRED",
                "released_at": datetime.now(timezone.utc).isoformat(),
                "release_reason": "maximum resident hold duration reached",
            }
            _write_json(resident_dir / "released.json", expired)
            _record_stage(stages, "RESIDENT_SCENE_HOLD_EXPIRED")
            return expired
        time.sleep(min(poll_seconds, maximum_seconds - elapsed))


def _initialize_loaded_instance(
    evaluator: Any,
    stages: Path,
    *,
    instance_id: int = 0,
    native_task_metric: bool = True,
) -> None:
    """Perform the official initial sequence without constructing again."""

    if type(native_task_metric) is not bool:
        raise TypeError("native_task_metric must be boolean")
    if not native_task_metric:
        from omnigibson.metrics import TaskMetric
        # L0 TaskSpecs have their own verifier. Native whole-activity progress
        # is unrelated here and can enumerate millions of alternative goals.
        before = len(evaluator.metrics)
        evaluator.metrics = [metric for metric in evaluator.metrics if type(metric) is not TaskMetric]
        _record_stage(stages, "NATIVE_TASK_METRIC_DISABLED", removed_count=before - len(evaluator.metrics),
                      reason="l0_taskspec_has_independent_verifier")
    _record_stage(stages, "INITIAL_RESET_START")
    evaluator.reset()
    _record_stage(stages, "INITIAL_RESET_PASS")
    _record_stage(stages, "TASK_INSTANCE_STATE_LOAD_START")
    evaluator.load_task_instance(instance_id)
    _record_stage(stages, "TASK_INSTANCE_STATE_LOAD_PASS")
    _record_stage(stages, "POST_LOAD_RESET_START")
    evaluator.reset()
    _record_stage(stages, "POST_LOAD_RESET_PASS")


def _run_calibration_lifecycle(
    *,
    evaluator: Any,
    control: SharedCanaryControl,
    bindings: B1SingleLoadLiveBindings,
    run_dir: Path,
    stages: Path,
    limits: SafeCarryCalibrationLimits,
    minimum_empty_trials: int,
    maximum_grasp_base_position_drift_m: float,
    maximum_grasp_base_orientation_drift_rad: float,
) -> tuple[
    B1ObservationOnlyApproach,
    LiveSafeCarryProposal,
    BootstrapPublicWristCalibrationRecord,
    B1CalibrationGraspLift,
    SafeCarryCalibrationRecord,
    PublicWristCalibrationRecord,
]:
    _record_stage(stages, "OBSERVATION_ONLY_APPROACH_START")
    approach = bindings.observation_only_approach(
        evaluator=evaluator,
        control=control,
        run_dir=run_dir,
    )
    if not isinstance(approach, B1ObservationOnlyApproach):
        raise B1SingleLoadRunnerError(
            "observation-only approach returned the wrong type"
        )
    _record_stage(
        stages,
        "OBSERVATION_ONLY_APPROACH_PASS",
        target_token_sha256=hashlib.sha256(
            approach.target_token.encode("utf-8")
        ).hexdigest(),
        public_frame_sha256=approach.public_frame_sha256,
    )

    _record_stage(stages, "CANDIDATE_PROPOSAL_START")
    proposal = bindings.propose_candidate(
        evaluator=evaluator,
        control=control,
        approach=approach,
        run_dir=run_dir,
    )
    if not isinstance(proposal, LiveSafeCarryProposal):
        raise B1SingleLoadRunnerError(
            "candidate proposal returned the wrong type"
        )
    if proposal.report.collision_aware is not True:
        raise B1SingleLoadRunnerError(
            "candidate proposal is not collision-aware"
        )
    if (
        proposal.report.state_token
        != approach.candidate_source_state_token
    ):
        raise B1SingleLoadRunnerError(
            "candidate was not proposed from the final approach state"
        )
    _record_stage(stages, "CANDIDATE_PROPOSAL_PASS")

    _record_stage(stages, "EMPTY_WRIST_NEGATIVE_CALIBRATION_START")
    empty_wrist = bindings.calibrate_empty_wrist(
        evaluator=evaluator,
        control=control,
        approach=approach,
        proposal=proposal,
        minimum_trials=minimum_empty_trials,
        run_dir=run_dir,
    )
    if not isinstance(empty_wrist, B1EmptyWristCalibration):
        raise B1SingleLoadRunnerError(
            "empty wrist calibration returned the wrong type"
        )
    bootstrap = empty_wrist.bootstrap
    _require_same_frozen_base(
        expected=approach.frozen_base,
        actual=empty_wrist.frozen_base,
        position_tolerance_m=maximum_grasp_base_position_drift_m,
        orientation_tolerance_rad=(
            maximum_grasp_base_orientation_drift_rad
        ),
    )
    if (
        bootstrap.empty_close_trials < minimum_empty_trials
        or bootstrap.empty_close_false_positives != 0
        or bootstrap.safe_carry_candidate_sha256
        != safe_carry_candidate_sha256(proposal.candidate)
    ):
        raise B1SingleLoadRunnerError(
            "empty wrist calibration does not match the exact candidate "
            "or zero-false-positive contract"
        )
    _record_stage(
        stages,
        "EMPTY_WRIST_NEGATIVE_CALIBRATION_PASS",
        trials=bootstrap.empty_close_trials,
        false_positives=bootstrap.empty_close_false_positives,
    )

    _record_stage(stages, "PHYSICAL_GRASP_LIFT_START")
    grasp_lift = bindings.physical_grasp_lift(
        evaluator=evaluator,
        control=control,
        approach=approach,
        proposal=proposal,
        empty_wrist=empty_wrist,
        run_dir=run_dir,
    )
    if not isinstance(grasp_lift, B1CalibrationGraspLift):
        raise B1SingleLoadRunnerError(
            "physical grasp lift returned the wrong type"
        )
    if (
        grasp_lift.positive_window.hand != proposal.candidate.hand
        or grasp_lift.positive_window.bootstrap_calibration_sha256
        != bootstrap.calibration_sha256
    ):
        raise B1SingleLoadRunnerError(
            "public-positive wrist window changed hand or bootstrap"
        )
    if (
        grasp_lift.consumed_target_token != approach.target_token
        or grasp_lift.consumed_candidate_source_state_token
        != proposal.report.state_token
    ):
        raise B1SingleLoadRunnerError(
            "grasp did not consume the frozen approach target/state token"
        )
    _require_same_frozen_base(
        expected=approach.frozen_base,
        actual=grasp_lift.frozen_base,
        position_tolerance_m=maximum_grasp_base_position_drift_m,
        orientation_tolerance_rad=(
            maximum_grasp_base_orientation_drift_rad
        ),
    )
    _record_stage(
        stages,
        "PHYSICAL_GRASP_LIFT_PASS",
        physical_grasp_executions=1,
        observed_lift_m=float(grasp_lift.observed_lift_m),
    )

    _record_stage(stages, "SAFE_CARRY_G3B_START")
    private_calibration_path = (
        run_dir / "private" / "safe_carry_samples.jsonl"
    )

    def write_private_sample(payload: Mapping[str, Any]) -> None:
        from behavior_hl.embodied.evaluator_bridge.run_g1_canary import (
            _append_jsonl,
        )

        _append_jsonl(private_calibration_path, dict(payload))

    safe_carry = run_safe_carry_calibration(
        candidate=proposal.candidate,
        limits=limits,
        actuator=grasp_lift.safe_carry_actuator,
        private_audit_writer=write_private_sample,
    )
    if not safe_carry.passed:
        raise B1SingleLoadRunnerError(
            "safe-carry G3b calibration failed: "
            f"{safe_carry.failure_reason}"
        )
    post_g3b_base = bindings.capture_frozen_base_pose(
        evaluator=evaluator,
        phase="after_g3b",
        run_dir=run_dir,
    )
    if not isinstance(post_g3b_base, B1FrozenBasePoseEvidence):
        raise B1SingleLoadRunnerError(
            "post-G3b frozen base evidence has wrong type"
        )
    _require_same_frozen_base(
        expected=approach.frozen_base,
        actual=post_g3b_base,
        position_tolerance_m=maximum_grasp_base_position_drift_m,
        orientation_tolerance_rad=(
            maximum_grasp_base_orientation_drift_rad
        ),
    )
    _record_stage(stages, "SAFE_CARRY_G3B_PASS")

    formal_wrist = promote_bootstrap_wrist_calibration(
        bootstrap=bootstrap,
        positive_window=grasp_lift.positive_window,
        safe_carry_record=safe_carry,
    )
    _freeze_calibration_records(
        run_dir=run_dir,
        approach=approach,
        proposal=proposal,
        bootstrap=bootstrap,
        positive_window=grasp_lift.positive_window,
        safe_carry=safe_carry,
        formal_wrist=formal_wrist,
    )
    _record_stage(stages, "CALIBRATION_RECORDS_FROZEN")
    return (
        approach,
        proposal,
        bootstrap,
        grasp_lift,
        safe_carry,
        formal_wrist,
    )


def _assemble_resident_episode(
    *,
    evaluator: Any,
    control: SharedCanaryControl,
    plan: B1PostResetAssemblyPlan,
    safe_carry: SafeCarryCalibrationRecord,
    run_dir: Path,
    stages: Path,
    video_fps: int,
) -> _AssembledResidentEpisode:
    _record_stage(stages, "BUILD_B1_DISPATCHER_START")
    carry_limits = plan.dispatcher_kwargs.get("carry_limits")
    reachability = audit_r1pro_g3b_stage_reachability(
        samples_per_stage=safe_carry.limits.samples_per_stage,
        carry_limits=carry_limits,
        kinematics_limits=plan.carry_kinematics_limits,
    )
    _write_json(
        run_dir / "private" / "g3b_reachability.json",
        {
            "passed": reachability.passed,
            "samples_per_stage": reachability.samples_per_stage,
            "max_torso_delta_per_tick": (
                reachability.max_torso_delta_per_tick
            ),
            "required_ticks_by_transition": dict(
                reachability.required_ticks_by_transition
            ),
            "failures": list(reachability.failures),
        },
    )
    reachability.require_reachable()
    dispatcher = build_b1_picking_up_toys_dispatcher(
        **dict(plan.dispatcher_kwargs)
    )
    if not isinstance(dispatcher, B1DispatcherBundle):
        raise B1SingleLoadRunnerError(
            "B1 dispatcher factory returned the wrong type"
        )
    _record_stage(stages, "BUILD_B1_DISPATCHER_PASS")

    _record_stage(stages, "BUILD_B1_SERVICE_FACTORY_START")
    services = R1ProB1ServiceFactory(
        public_carry_state_source=plan.public_carry_tracker,
        **dict(plan.component_builders),
    )
    readiness = audit_b1_live_readiness(
        registry=dispatcher.registry,
        dispatcher=dispatcher.dispatcher,
        resident_service_factory=services,
        public_carry_tracker=plan.public_carry_tracker,
        safe_carry_calibration=safe_carry,
        required_actions=B1_REQUIRED_ACTIONS,
    )
    _write_json(
        run_dir / "private" / "b1_live_readiness.json",
        readiness.to_dict(),
    )
    readiness.require_ready()
    _record_stage(stages, "BUILD_B1_SERVICE_FACTORY_PASS")

    environment = R1ProResidentEnvironment(
        evaluator=evaluator,
        control=control,
        service_factory=services,
        settle_steps=plan.settle_steps,
        settle_step_policy=plan.settle_step_policy,
    )
    model = plan.model_factory(environment)
    verifier = plan.verifier_factory(evaluator)
    artifact_delegate = plan.artifact_factory(run_dir, environment)
    artifacts = _RecordingFinalizingArtifactSink(
        evaluator=evaluator,
        delegate=artifact_delegate,
        video_path=run_dir / "video" / "episode.mp4",
        video_fps=video_fps,
    )

    _record_stage(stages, "BUILD_B1_RESIDENT_EPISODE_START")
    bundle = build_b1_resident_episode(
        episode=plan.episode,
        dispatcher_bundle=dispatcher,
        model=model,
        environment=environment,
        verifier=verifier,
        artifacts=artifacts,
        limits=plan.limits,
        diagnostic_target_new_leaf_count=(
            plan.diagnostic_target_new_leaf_count
        ),
    )
    _record_stage(stages, "BUILD_B1_RESIDENT_EPISODE_PASS")
    return _AssembledResidentEpisode(
        bundle=bundle,
        recording_artifacts=artifacts,
        model=model,
    )


def _require_replay_single_leaf_completion(
    *,
    episode_result: ResidentEpisodeResult,
    model: Any,
    run_dir: Path,
) -> None:
    """Require exact replay consumption and the bounded 1/6 terminal fact."""

    if not isinstance(model, PublicSingleLeafReplayModel):
        raise B1SingleLoadRunnerError(
            "single-leaf validation must use PublicSingleLeafReplayModel"
        )
    model.require_fully_consumed()
    checkpoints = tuple(episode_result.checkpoints)
    if not checkpoints:
        raise B1SingleLoadRunnerError(
            "single-leaf validation produced no private checkpoints"
        )
    final = checkpoints[-1]
    expected_q = 1.0 / 6.0
    if (
        episode_result.termination.classification
        is not TerminationClass.CANARY_TARGET_REACHED
        or episode_result.termination.success is not False
        or final.snapshot.official_success is not False
        or len(final.snapshot.leaf_truth) != 6
        or len(final.active_new_leaves) != 1
        or sum(final.snapshot.leaf_truth.values()) != 1
        or final.snapshot.official_q is None
        or not math.isfinite(final.snapshot.official_q)
        or not math.isclose(
            final.snapshot.official_q,
            expected_q,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or final.research_partial_q is None
        or not math.isfinite(final.research_partial_q)
        or not math.isclose(
            final.research_partial_q,
            expected_q,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or episode_result.artifact_complete is not True
    ):
        raise B1SingleLoadRunnerError(
            "single-leaf validation did not prove exact 1/6 "
            "CANARY_TARGET_REACHED with complete artifacts"
        )
    _write_json_atomic(
        Path(run_dir)
        / "private"
        / "replay_consumption_diagnostics.json",
        {
            "visibility": "private",
            "termination": (
                episode_result.termination.classification.value
            ),
            "termination_success": episode_result.termination.success,
            "official_success": final.snapshot.official_success,
            "official_leaf_count": len(final.snapshot.leaf_truth),
            "active_new_leaf_count": len(final.active_new_leaves),
            "official_q": final.snapshot.official_q,
            "research_partial_q": final.research_partial_q,
            "artifact_complete": episode_result.artifact_complete,
            "model": dict(model.consumption_diagnostics),
        },
    )


def _safe_carry_limits(
    contract: Mapping[str, Any],
) -> SafeCarryCalibrationLimits:
    raw = contract["safe_carry_limits"]
    try:
        result = SafeCarryCalibrationLimits(
            max_retract_steps=int(raw["max_retract_steps"]),
            g3b_stages=tuple(raw["g3b_stages"]),
            samples_per_stage=int(raw["samples_per_stage"]),
            arm_tolerance_rad=float(raw["arm_tolerance_rad"]),
            eef_position_tolerance_m=float(
                raw["eef_position_tolerance_m"]
            ),
            eef_orientation_tolerance_deg=float(
                raw["eef_orientation_tolerance_deg"]
            ),
            minimum_object_lift_m=float(
                raw["minimum_object_lift_m"]
            ),
            maximum_payload_linear_speed_mps=float(
                raw["maximum_payload_linear_speed_mps"]
            ),
            maximum_payload_angular_speed_radps=float(
                raw["maximum_payload_angular_speed_radps"]
            ),
            maximum_base_z_drift_m=float(
                raw["maximum_base_z_drift_m"]
            ),
            maximum_abs_roll_pitch_deg=float(
                raw["maximum_abs_roll_pitch_deg"]
            ),
        )
        audit_r1pro_g3b_stage_reachability(
            samples_per_stage=result.samples_per_stage,
            carry_limits=CarrySafetyLimits(),
            kinematics_limits=R1ProCarryKinematicsLimits(),
        ).require_reachable()
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise B1SingleLoadRunnerError(
            f"invalid safe-carry limit contract: {exc}"
        ) from exc


def _reset_tolerances(
    contract: Mapping[str, Any],
) -> SameSceneResetTolerances:
    raw = contract["reset_tolerances"]
    try:
        return SameSceneResetTolerances(
            base_position_m=float(raw["base_position_m"]),
            base_orientation_rad=float(raw["base_orientation_rad"]),
            joint_position_rad=float(raw["joint_position_rad"]),
            stable_state_numeric_abs=float(
                raw["stable_state_numeric_abs"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise B1SingleLoadRunnerError(
            f"invalid reset tolerance contract: {exc}"
        ) from exc


def _require_same_frozen_base(
    *,
    expected: B1FrozenBasePoseEvidence,
    actual: B1FrozenBasePoseEvidence,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> None:
    if not isinstance(expected, B1FrozenBasePoseEvidence) or not isinstance(
        actual,
        B1FrozenBasePoseEvidence,
    ):
        raise TypeError("frozen base evidence is invalid")
    if (
        actual.frozen_base_pose_token
        != expected.frozen_base_pose_token
    ):
        raise B1SingleLoadRunnerError(
            "frozen base pose token changed after final approach"
        )
    for name, value in (
        ("position_tolerance_m", position_tolerance_m),
        ("orientation_tolerance_rad", orientation_tolerance_rad),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    position_error = math.sqrt(
        sum(
            (actual_value - expected_value) ** 2
            for actual_value, expected_value in zip(
                actual.position,
                expected.position,
            )
        )
    )
    first = _normalized_quaternion(expected.quaternion_xyzw)
    second = _normalized_quaternion(actual.quaternion_xyzw)
    dot = min(1.0, max(-1.0, abs(sum(a * b for a, b in zip(first, second)))))
    orientation_error = 2.0 * math.acos(dot)
    if (
        position_error > position_tolerance_m
        or orientation_error > orientation_tolerance_rad
    ):
        raise B1SingleLoadRunnerError(
            "base pose changed after final approach / candidate proposal: "
            f"position_error_m={position_error}, "
            f"orientation_error_rad={orientation_error}"
        )


def _finite_vector(
    value: Any,
    size: int,
    name: str,
) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite and {size}D")
    return result


def _normalized_quaternion(
    value: tuple[float, ...],
) -> tuple[float, ...]:
    norm = math.sqrt(sum(item * item for item in value))
    if norm <= 1e-8:
        raise ValueError("quaternion has zero norm")
    return tuple(item / norm for item in value)


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Atomically freeze one strict-JSON lifecycle artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(
        f"{destination.suffix}.tmp-{os.getpid()}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                dict(payload),
                stream,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def capture_source_freeze(project_root: Path) -> dict[str, Any]:
    """Freeze every embodied source/config file by relative path and sha256."""

    root = Path(project_root).resolve()
    candidates = set(
        (root / "src" / "behavior_hl" / "embodied").rglob("*.py")
    )
    project_identity = root / "src" / "behavior_hl" / "project_identity.py"
    if project_identity.is_file():
        candidates.add(project_identity)
    candidates.update(
        path
        for path in (root / "configs").rglob("*")
        if path.is_file()
    )
    candidates.update(
        path
        for pattern in ("*.py", "*.sh")
        for path in (root / "scripts").rglob(pattern)
        if path.is_file()
    )
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        candidates.add(pyproject)
    files: list[dict[str, Any]] = []
    for path in sorted(candidates):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise B1SingleLoadRunnerError(
                f"source-freeze path escaped project root: {resolved}"
            ) from exc
        files.append(
            {
                "relative_path": relative,
                "byte_size": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
    if not files:
        raise B1SingleLoadRunnerError("source freeze captured no files")
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(item["relative_path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(item["sha256"].encode("ascii"))
        aggregate.update(b"\n")
    return {
        "schema_version": "behavior2026.source-freeze/v1",
        "project_root": str(root),
        "roots": [
            "src/behavior_hl/embodied/**/*.py",
            "src/behavior_hl/project_identity.py",
            "configs/**/*",
            "scripts/**/*.py",
            "scripts/**/*.sh",
            "pyproject.toml",
        ],
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": aggregate.hexdigest(),
    }


def require_unchanged_source_freeze(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    """Fail a live result if any frozen source/config file changed."""

    if (
        before.get("schema_version") != after.get("schema_version")
        or before.get("aggregate_sha256") != after.get("aggregate_sha256")
        or before.get("files") != after.get("files")
    ):
        before_files = {
            item["relative_path"]: item["sha256"]
            for item in before.get("files", ())
        }
        after_files = {
            item["relative_path"]: item["sha256"]
            for item in after.get("files", ())
        }
        changed = sorted(
            path
            for path in set(before_files) | set(after_files)
            if before_files.get(path) != after_files.get(path)
        )
        raise B1SingleLoadRunnerError(
            "source/config changed during live run: "
            f"{changed[:50]}"
        )


def _capture_loaded_task_instance_identity(
    *,
    evaluator: Any,
    mode: str,
    instance_id: int,
) -> dict[str, Any]:
    """Hash the exact 2026 task-instance JSON selected by the live evaluator."""

    from omnigibson.utils.asset_utils import get_task_instance_path

    task = getattr(getattr(evaluator, "env", None), "task", None)
    if task is None:
        raise B1SingleLoadRunnerError(
            "loaded evaluator task is unavailable for data identity"
        )
    scene_name = getattr(task, "scene_name", None)
    activity_name = getattr(task, "activity_name", None)
    definition_id = getattr(task, "activity_definition_id", None)
    filename_builder = getattr(
        task,
        "get_cached_activity_scene_filename",
        None,
    )
    if (
        not isinstance(scene_name, str)
        or not isinstance(activity_name, str)
        or not callable(filename_builder)
    ):
        raise B1SingleLoadRunnerError(
            "official task-instance locator metadata is unavailable"
        )
    filename = filename_builder(
        scene_model=scene_name,
        activity_name=activity_name,
        activity_definition_id=definition_id,
        activity_instance_id=instance_id,
    )
    path_value = get_task_instance_path(
        scene_name,
        f"{scene_name}_task_{activity_name}_instances/"
        f"{filename}-tro_state",
        mode=mode,
    )
    if path_value is None:
        raise B1SingleLoadRunnerError(
            "official loaded task-instance JSON cannot be resolved"
        )
    path = Path(path_value).resolve()
    if not path.is_file():
        raise B1SingleLoadRunnerError(
            "official loaded task-instance JSON is missing"
        )
    return {
        "schema_version": "behavior2026.task-instance-identity/v1",
        "mode": mode,
        "instance_id": instance_id,
        "scene_name": scene_name,
        "activity_name": activity_name,
        "activity_definition_id": definition_id,
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": _sha256(path),
        "asset_tree_hash_scope": "P1_not_full_asset_tree",
    }


def _freeze_calibration_records(
    *,
    run_dir: Path,
    approach: B1ObservationOnlyApproach,
    proposal: LiveSafeCarryProposal,
    bootstrap: BootstrapPublicWristCalibrationRecord,
    positive_window: PublicWristPositiveWindow,
    safe_carry: SafeCarryCalibrationRecord,
    formal_wrist: PublicWristCalibrationRecord,
) -> None:
    root = run_dir / "private" / "calibration"
    root.mkdir(parents=True, exist_ok=False)
    _write_json(
        root / "approach.json",
        {
            "target_token_sha256": hashlib.sha256(
                approach.target_token.encode("utf-8")
            ).hexdigest(),
            "public_frame_sha256": approach.public_frame_sha256,
            "candidate_source_state_token_sha256": hashlib.sha256(
                approach.candidate_source_state_token.encode("utf-8")
            ).hexdigest(),
            "frozen_base_pose_token_sha256": hashlib.sha256(
                approach.frozen_base.frozen_base_pose_token.encode(
                    "utf-8"
                )
            ).hexdigest(),
            "final_base_position": approach.frozen_base.position,
            "final_base_quaternion_xyzw": (
                approach.frozen_base.quaternion_xyzw
            ),
            "target_from_public_rgbd": (
                approach.target_from_public_rgbd
            ),
            "physical_grasp_executions": (
                approach.physical_grasp_executions
            ),
            "private_truth_exposed_to_policy": (
                approach.private_truth_exposed_to_policy
            ),
        },
    )
    _write_json(
        root / "proposal.json",
        {
            "version": proposal.report.version,
            "candidate_source": proposal.report.candidate_source,
            "official_ik_method": proposal.report.official_ik_method,
            "collision_aware": proposal.report.collision_aware,
            "attempted_candidates": proposal.report.attempted_candidates,
            "feasible_candidates": proposal.report.feasible_candidates,
            "selected_candidate_id": (
                proposal.report.selected_candidate_id
            ),
            "selected_joint_margin_fraction": (
                proposal.report.selected_joint_margin_fraction
            ),
            "selected_normalized_motion": (
                proposal.report.selected_normalized_motion
            ),
            "candidate_sha256": safe_carry_candidate_sha256(
                proposal.candidate
            ),
        },
    )
    _write_json(
        root / "bootstrap_wrist.json",
        dict(bootstrap.to_manifest()),
    )
    _write_json(
        root / "public_positive_window.json",
        dict(positive_window.to_manifest()),
    )
    _write_json(
        root / "safe_carry.json",
        {
            "schema_version": safe_carry.schema_version,
            "candidate_sha256": safe_carry_candidate_sha256(
                safe_carry.candidate
            ),
            "ik_only_pass": safe_carry.ik_only_pass,
            "retract_steps": safe_carry.retract_steps,
            "stage_records": [
                {
                    "stage": item.stage,
                    "samples": item.samples,
                    "passed": item.passed,
                }
                for item in safe_carry.stage_records
            ],
            "held_continuous": safe_carry.held_continuous,
            "grasp_proxy_continuous": (
                safe_carry.grasp_proxy_continuous
            ),
            "body_eef_position": safe_carry.body_eef_position,
            "body_eef_quaternion_xyzw": (
                safe_carry.body_eef_quaternion_xyzw
            ),
            "passed": safe_carry.passed,
            "failure_reason": safe_carry.failure_reason,
            "sha256": safe_carry.sha256,
        },
    )
    _write_json(
        root / "formal_wrist.json",
        dict(formal_wrist.to_manifest()),
    )
