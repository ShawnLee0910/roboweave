#!/usr/bin/env python3
"""Model-independent resident Action20 scene server.

The process owns exactly one OmniGibson scene, one GPU, and one Unix-domain
control socket.  It never reads a Prompt, opens a model endpoint, starts SSH,
or records a model identity.  Clients submit only public Action20 decisions
and receive only allowlisted action feedback plus public RGB artifact paths.
Official predicate values are written privately and never returned over the
socket.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import sys
import traceback
from typing import Any, Mapping, Sequence


BENCH_TRY_ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = Path(__file__).resolve().parent
if str(BENCH_TRY_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_TRY_ROOT))

from action20_protocol import (  # noqa: E402
    BASE_TRANSLATION_ACTIONS,
    MARKER_ACTIONS,
    STRICT_ACTION20,
    THREE_CALL_MARKER_ACTIONS,
)
from lab.l1_episode import audited_step, bind as bind_process_verifier  # noqa: E402
from lab.reset_safe_three_rgb_wrapper import (  # noqa: E402
    RESET_SAFE_RGBD_WRAPPER_TARGET,
    RESET_SAFE_THREE_RGB_WRAPPER_TARGET,
    run_b1_single_load_with_reset_safe_three_rgb,
)
from agent_observation import (  # noqa: E402
    RGBD_PROTOCOL_VERSION,
    marker_depth_records,
    validate_public_marker_depth,
)
from lab.action20_atomic_validation import (  # noqa: E402
    AtomicActionCard,
    MARKER_ACTIONS as ATOMIC_MARKER_ACTIONS,
    marker_number_for_binding,
    normalize_atomic_cards,
)
from lab.atomic_fixture_inventory import (  # noqa: E402
    AtomicFixtureInventoryCache,
    build_atomic_fixture_inventory,
)
from lab.scene_action_preconditions import (  # noqa: E402
    public_precondition_block_reason,
)
from lab.scene_socket_transport import (  # noqa: E402
    send_response_or_client_gone,
)
from lab.scene_protocol import control_socket_path  # noqa: E402
from lab.scene_source_identity import validate_source_identity  # noqa: E402
from lab.rolling_episode_checkpoints import (  # noqa: E402
    DEFAULT_ACTION_INTERVAL,
    DEFAULT_MAXIMUM_RETAINED,
    RollingCheckpointError,
    RollingEpisodeCheckpointRing,
)
from lab.persistent_scene_checkpoints import (  # noqa: E402
    PERSISTENT_SCENE_CHECKPOINT_VERSION,
    PersistentCheckpointError,
    decode_portable_value,
    encode_portable_value,
    load_persistent_checkpoint,
    write_persistent_checkpoint,
)

from behavior_hl.embodied.evaluator_bridge.gripper_visual_profiles import (  # noqa: E402
    GRIPPER_VISUAL_PROFILE_INWARD_12CM_OFFICIAL_CAMERA,
)
from behavior_hl.embodied.evaluator_bridge.grounded_option_identity_owner import (  # noqa: E402
    resolve_grounded_option_identity_owner,
)
from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (  # noqa: E402
    get_inside_task_profile,
    private_inside_task_roles,
)
from behavior_hl.embodied.evaluator_bridge.toggle_task_profiles import (  # noqa: E402
    ToggleTaskProfile,
    get_toggle_task_profile,
    private_toggle_task_roles,
    private_toggle_task_targets,
)
from behavior_hl.embodied.evaluator_bridge.launch_b1_single_load import (  # noqa: E402
    build_production_bindings,
)
from behavior_hl.embodied.evaluator_bridge.logical_inventory import (  # noqa: E402
    LogicalInventoryEntry,
    LogicalInventorySnapshot,
)
from behavior_hl.embodied.evaluator_bridge.marker_symbolic_grasp_backend import (  # noqa: E402
    official_instance_bboxes,
)
from behavior_hl.embodied.evaluator_bridge.marker_symbolic_spatial_execution import (  # noqa: E402
    plan_live_under_supports,
)
from behavior_hl.embodied.evaluator_bridge.official_predicate_verifier import (  # noqa: E402
    OfficialPredicateVerifier,
)
from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (  # noqa: E402
    THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
)
from behavior_hl.embodied.evaluator_bridge.private_head_segmentation_side_channel import (  # noqa: E402
    get_preinstalled_private_head_segmentation_side_channel,
)
from behavior_hl.embodied.evaluator_bridge.private_visual_markers import (  # noqa: E402
    build_private_marker_snapshot,
)
from behavior_hl.embodied.evaluator_bridge.private_drawer_parts import (  # noqa: E402
    drawer_fillable_aabbs,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (  # noqa: E402
    _apply_model_phase_gripper_presentation,
    _build_full_motion_safety_starter,
    _build_marker_runtime,
    _capture_private_inside_action_logical_checkpoint,
    _capture_action_boundary_observation,
    _episode,
    _execute,
    _new_leaf_count,
    _object_xy,
    _oracle_execute_exact_private_symbolic,
    _oracle_navigate_near_with_privileged_symbolic,
    _private_official_oracle_route_plan,
    _privileged_symbolic_station_pose,
    _require_loaded_profile,
    _restore_private_inside_action_logical_checkpoint,
    _save_frame,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_single_load import (  # noqa: E402
    B1SingleLoadWorkerSpec,
)
from behavior_hl.embodied.evaluator_bridge.run_b1_strategy_sweep import (  # noqa: E402
    _reset_same_scene,
)
from behavior_hl.embodied.evaluator_bridge.same_scene_reset_attestation import (  # noqa: E402
    SameSceneResetTolerances,
    attest_same_scene_reset,
)
from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (  # noqa: E402
    object_aabb,
    object_pose,
    read_action_relation,
)
from behavior_hl.embodied.policy.observations import CAMERA_SENSOR_NAMES  # noqa: E402
from behavior_hl.embodied.runtime.dispatch import ActionPluginResult  # noqa: E402
from behavior_hl.embodied.runtime.action_integrity import (  # noqa: E402
    execution_integrity_failure,
    invalidate_execution,
    require_execution_integrity,
)
from behavior_hl.embodied.runtime.public_action_feedback import (  # noqa: E402
    ACTION20_BLOCKED_CATEGORIES,
    model_visible_action20_last_result,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_visible_official_grasp_factory import (  # noqa: E402
    planar_base_pose_from_robot,
)


SCENE_PROTOCOL_VERSION = "behavior-hl-bench-try-scene-control/v1"
SCENE_SERVER_VERSION = (
    "behavior-hl-bench-try-resident-inside-scene-server/"
    "v9-persistent-checkpoint"
)
DEFAULT_RUNTIME_ROOT = BENCH_TRY_ROOT / "runtime" / "task25"
DEFAULT_ARTIFACT_ROOT = DEFAULT_RUNTIME_ROOT / "simulator"
DEFAULT_SOCKET_PATH = control_socket_path(DEFAULT_RUNTIME_ROOT)
DEFAULT_READY_PATH = DEFAULT_RUNTIME_ROOT / "ready.json"
SCENE_MANIFEST_PATH = LAB_ROOT / "scene_manifest.json"
DEFAULT_APPDATA_SLOT = Path(
    "/tmp/behavior2026_og_appdata_isaac5_v1/slot_000"
)
MAX_REQUEST_BYTES = 64 * 1024
PRIVATE_ERROR_LOG_RELATIVE_PATH = Path("private") / "scene_server_errors.jsonl"
MAX_PRIVATE_ERROR_ENTRY_BYTES = 128 * 1024
MAX_PRIVATE_ERROR_LOG_BYTES = 4 * 1024 * 1024
BASE_ACTIONS = frozenset(
    {
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
    }
)
OVERHEAD_VIEW_MODES = frozenset({"off", "always", "on_blocked"})


class SceneProtocolError(ValueError):
    """A malformed or out-of-order local scene command."""


class SceneIntegrationError(RuntimeError):
    """An internal Action20/Marker contract violation, never model fault."""


def _resolve_task_profile(task_name: str) -> Any:
    """Resolve every task registered by the shared resident runtime."""

    try:
        return get_inside_task_profile(task_name)
    except KeyError:
        return get_toggle_task_profile(task_name)


@dataclass(frozen=True, slots=True)
class PublicSceneObservation:
    observation_id: str
    kind: str
    images: tuple[Mapping[str, str], ...]
    marker_targets: tuple[int, ...] = ()
    marker_depth: tuple[Mapping[str, Any], ...] | None = None

    def to_record(self) -> dict[str, Any]:
        record = {
            "observation_id": self.observation_id,
            "kind": self.kind,
            "images": [dict(item) for item in self.images],
            "marker_targets": list(self.marker_targets),
        }
        if self.marker_depth is not None:
            record["marker_depth"] = [dict(item) for item in self.marker_depth]
        return record


_CARRY_TRACKER_CHECKPOINT_ATTRIBUTES = (
    "_confidence",
    "_active_hand",
    "_acquired_via",
    "_transition_count",
    "_last_transition",
)


@dataclass(frozen=True, slots=True)
class _ResidentRuntimeCheckpoint:
    """One same-process rollback point at a clean public action boundary."""

    episode_id: str
    episode_dir: Path
    runtime: Any
    episode: Any
    verifier: Any
    initial_verifier: Any
    last_verifier: Any
    sequence: int
    simulator_state: Any
    simulator_state_bytes_private: int
    logical_state: Any
    carry_tracker_state: tuple[tuple[str, Any], ...]


def _checkpoint_simulator() -> Any:
    """Resolve the already-live simulator without importing it at CPU test time."""

    import omnigibson as og

    simulator = getattr(og, "sim", None)
    if simulator is None:
        raise RuntimeError("resident rolling checkpoint simulator is unavailable")
    return simulator


def _checkpoint_payload_bytes(value: Any, *, seen: set[int] | None = None) -> int:
    """Best-effort tensor/array payload size for private memory telemetry."""

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, Mapping):
        return sum(
            _checkpoint_payload_bytes(item, seen=seen)
            for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return sum(_checkpoint_payload_bytes(item, seen=seen) for item in value)
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and nbytes >= 0:
        return nbytes
    element_size = getattr(value, "element_size", None)
    nelement = getattr(value, "nelement", None)
    if callable(element_size) and callable(nelement):
        try:
            return int(element_size()) * int(nelement())
        except Exception:
            return 0
    return 0


def _checkpoint_state_numpy(value: Any) -> Any:
    """Copy an already-serialized simulator tensor into portable CPU memory."""

    import numpy as np

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    result = np.ascontiguousarray(candidate)
    if result.ndim != 1 or result.dtype.hasobject:
        raise RuntimeError("serialized simulator checkpoint is not a numeric vector")
    return result


class ResidentInsideTaskScene:
    """One loaded inside-task scene with no provider or conversation dependency."""

    def __init__(
        self,
        *,
        evaluator: Any,
        control: Any,
        bindings: Any,
        run_dir: Path,
        stages: Path,
        seed: int,
        profile: Any,
        fixture_rules: Mapping[str, Any] | None,
        observation_profile: str = THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
        rgbd_protocol_version: str | None = None,
        scene_load_mode: str = "partial",
        rolling_checkpoint_action_interval: int = DEFAULT_ACTION_INTERVAL,
        rolling_checkpoint_maximum_retained: int = DEFAULT_MAXIMUM_RETAINED,
        l0_task_spec: Mapping[str, Any] | None = None,
        registered_task: Any | None = None,
        public_task_id: int | None = None,
        public_instruction_id: str | None = None,
        preserve_prepared_initial_state: bool = False,
        oracle_prepared_state: Mapping[str, Any] | None = None,
        initial_state_validator: Any | None = None,
        placement_physical_consequences_enabled: bool = False,
        native_q_alone_is_success: bool = True,
        runtime_registry: Path | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.control = control
        self.bindings = bindings
        self.run_dir = Path(run_dir).resolve()
        self._runtime_registry = (BENCH_TRY_ROOT / "runtime" if runtime_registry is None else Path(runtime_registry)).resolve()
        self.stages = Path(stages).resolve()
        self.seed = seed
        self.profile = profile
        self._public_task_id = public_task_id
        self._public_instruction_id = public_instruction_id
        self._prepared_scene_document = None
        self._oracle_prepared_state = oracle_prepared_state
        self._initial_state_validator = initial_state_validator
        self._placement_physical_consequences_enabled = bool(
            placement_physical_consequences_enabled
        )
        self._native_q_alone_is_success = bool(native_q_alone_is_success)
        if self._placement_physical_consequences_enabled and self._native_q_alone_is_success:
            raise ValueError('Physical model scoring forbids native Q as success')
        self._physical_audit = None
        self._physical_observe_cm = None
        if oracle_prepared_state is not None:
            from behavior_hl.embodied.evaluator_bridge.prepared_model_state import validate_bundle
            validate_bundle(oracle_prepared_state, instruction_id=public_instruction_id,
                            seed=seed, observation_profile=observation_profile)
        self._l0_task_spec = None if l0_task_spec is None else dict(l0_task_spec)
        self._registered_task = registered_task
        self._process_audit = None
        if observation_profile not in {
            THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
            THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE,
        }:
            raise ValueError("resident observation profile is invalid")
        if rgbd_protocol_version not in {None, RGBD_PROTOCOL_VERSION}:
            raise ValueError("resident RGB-D protocol version is invalid")
        if (rgbd_protocol_version is None) != (
            observation_profile == THREE_RGB_HEAD_EVAL_CAMERA_PROFILE
        ):
            raise ValueError("resident RGB-D profile binding is inconsistent")
        self.observation_profile = observation_profile
        self.rgbd_protocol_version = rgbd_protocol_version
        self._marker_depth_protocol: str | None = None
        self._overhead_view_mode = "off"
        self._automatic_interaction_approach_enabled = False
        if scene_load_mode not in {"partial", "full"}:
            raise ValueError("resident scene load mode is invalid")
        self.scene_load_mode = scene_load_mode
        if (
            isinstance(rolling_checkpoint_action_interval, bool)
            or not isinstance(rolling_checkpoint_action_interval, int)
            or rolling_checkpoint_action_interval <= 0
            or isinstance(rolling_checkpoint_maximum_retained, bool)
            or not isinstance(rolling_checkpoint_maximum_retained, int)
            or rolling_checkpoint_maximum_retained <= 0
        ):
            raise ValueError("resident rolling checkpoint policy is invalid")
        self._rolling_checkpoint_action_interval = (
            rolling_checkpoint_action_interval
        )
        self._rolling_checkpoint_maximum_retained = (
            rolling_checkpoint_maximum_retained
        )
        _require_loaded_profile(self.evaluator, self.profile)
        self._toggle_task = isinstance(self.profile, ToggleTaskProfile)
        self._fixture_rules = (
            None if fixture_rules is None else dict(fixture_rules)
        )
        fixtures = () if self._fixture_rules is None else self._fixture_rules["fixtures"]
        self._fixtures = {
            str(item["fixture_id"]): dict(item)
            for item in fixtures
        }
        self._default_fixture_id = (
            None
            if self._fixture_rules is None
            else str(self._fixture_rules["default_fixture_id"])
        )
        atomic_rules = (
            None
            if self._fixture_rules is None
            else self._fixture_rules.get("atomic_cards")
        )
        atomic_cards = (
            () if atomic_rules is None else normalize_atomic_cards(atomic_rules)
        )
        self._atomic_cards = {card.card_id: card for card in atomic_cards}
        self._atomic_inventory_cache = AtomicFixtureInventoryCache()
        self._active_atomic_card: AtomicActionCard | None = None
        self._atomic_run_count = 0
        self._calibrated_fixtures: dict[str, Mapping[str, Any]] = {}
        self._baseline = self.bindings.capture_private_reset_snapshot(
            evaluator=self.evaluator,
            phase="pre_calibration",
            run_dir=self.run_dir,
        )
        if preserve_prepared_initial_state:
            from copy import deepcopy
            self._prepared_scene_document = deepcopy(self.evaluator.env.scene.save(as_dict=True))
        self._option_owner = (
            None
            if self._toggle_task
            else resolve_grounded_option_identity_owner(
                evaluator=self.evaluator,
                profile=self.profile,
            )
        )
        self._motion_safety_starter = _build_full_motion_safety_starter(
            self.evaluator
        )
        segmentation_attestation = (
            self._require_reset_safe_segmentation_channel()
        )
        self._reset_count = 0
        self._runtime: Any | None = None
        self._episode: Any | None = None
        self._episode_id: str | None = None
        self._episode_dir: Path | None = None
        self._verifier: Any | None = None
        self._initial_verifier: Any | None = None
        self._last_verifier: Any | None = None
        self._sequence = 0
        self._action_artifact_dir: Path | None = None
        self._observation_id_prefix: str | None = None
        self._pending_marker_action: str | None = None
        self._pending_marker_targets: frozenset[int] = frozenset()
        self._pending_marker_stage: int | None = None
        self._rolling_checkpoints: RollingEpisodeCheckpointRing | None = None
        self._session_valid = True
        private = self.run_dir / "private" / "resident_scene"
        private.mkdir(parents=True, exist_ok=True)
        _write_json(
            private / "baseline.json",
            self._baseline.to_private_record(include_stable_state=True),
        )
        _write_json(
            private / "reset_safe_private_segmentation.json",
            segmentation_attestation,
        )

    @property
    def episode_active(self) -> bool:
        return self._runtime is not None

    def status(self) -> Mapping[str, Any]:
        rolling = self._rolling_checkpoints
        return {
            "scene_server_version": SCENE_SERVER_VERSION,
            "scene_resident": True,
            "scene_load_attempts": 1,
            "successful_scene_loads": 1,
            "in_memory_resets": self._reset_count,
            "episode_active": self.episode_active,
            "session_valid": self._session_valid,
            "episode_id": self._episode_id,
            "next_sequence": self._sequence,
            "rolling_checkpoint_action_interval": (
                self._rolling_checkpoint_action_interval
            ),
            "rolling_checkpoint_maximum_retained": (
                self._rolling_checkpoint_maximum_retained
            ),
            "rolling_checkpoint_action_counts": (
                [] if rolling is None else list(rolling.captured_action_counts)
            ),
            "model_bound": False,
            "prompt_bound": False,
            "provider_bound": False,
            "private_oracle_executed": False,
            "task_id": self.profile.task_id if self._public_task_id is None else self._public_task_id,
            "task_name": self.profile.task_name if self._public_task_id is None else (self._public_instruction_id or self._l0_task_spec['task_spec_id']),
            "task_split": self.profile.split,
            "task_instance_id": str(self.profile.instance_id),
            "l0_task_spec_id": (
                None
                if self._l0_task_spec is None
                else self._l0_task_spec["task_spec_id"]
            ),
            "scene_load_mode": self.scene_load_mode,
            "observation_profile": self.observation_profile,
            "rgbd_protocol_version": self.rgbd_protocol_version,
            "marker_depth_default_enabled": False,
            "marker_depth_episode_enabled": (
                self._marker_depth_protocol == RGBD_PROTOCOL_VERSION
            ),
            "automatic_interaction_approach_enabled": (
                self._automatic_interaction_approach_enabled
            ),
            "default_fixture_id": self._default_fixture_id,
            "fixture_ids": sorted(self._fixtures),
            "atomic_card_ids": sorted(self._atomic_cards),
            "atomic_action_count": len(self._atomic_cards),
            "atomic_inventory_prepared": (
                self._atomic_inventory_cache.value is not None
            ),
        }

    def prepare_atomic_inventory(self) -> Mapping[str, Any]:
        """Freeze one private live fixture inventory without a public turn."""

        if not self._session_valid:
            raise SceneProtocolError("resident scene session is invalid")
        if self.episode_active:
            raise SceneProtocolError(
                "atomic inventory cannot be prepared during an active episode"
            )
        if self._toggle_task:
            raise SceneProtocolError(
                "atomic fixture inventory is unavailable for toggle tasks"
            )
        cached = self._atomic_inventory_cache.value
        if cached is not None:
            return cached.to_public_record(idempotent=True)
        if self._atomic_cards:
            raise SceneProtocolError(
                "atomic cards are already frozen by the lane configuration"
            )
        graspables, containers = private_inside_task_roles(
            self.evaluator,
            self.profile,
        )
        private_path = (
            self.run_dir
            / "private"
            / "resident_scene"
            / "atomic_fixture_inventory.json"
        )

        def persist_prepared_inventory(inventory: Any) -> None:
            cards = {card.card_id: card for card in inventory.cards}
            if len(cards) != len(inventory.cards):
                raise RuntimeError("prepared atomic card IDs are ambiguous")
            _write_json(private_path, inventory.to_private_record())
            self._atomic_cards = cards

        inventory, idempotent = self._atomic_inventory_cache.get_or_build(
            lambda: build_atomic_fixture_inventory(
                scene_objects=self._scene_objects_by_name(self.evaluator),
                graspables=graspables,
                containers=containers,
                under_feasibility_query=self._atomic_under_feasibility,
                station_feasibility_query=self._atomic_station_feasibility,
                aabb_reader=object_aabb,
            ),
            before_commit=persist_prepared_inventory,
        )
        if idempotent:
            return inventory.to_public_record(idempotent=True)
        return inventory.to_public_record(idempotent=False)

    def _atomic_under_feasibility(
        self, source: Any, target: Any, support: Any,
    ) -> Mapping[str, Any]:
        """Use the public Under marker/executor kernel before freezing a fixture."""

        plans = plan_live_under_supports(
            held=source,
            target=target,
            supports=(support,),
            robot=self.evaluator.robot,
        )
        for plan in plans:
            if plan.support is support and plan.candidate_plan.candidates:
                candidate = plan.candidate_plan.candidates[0]
                return {
                    "accepted": True,
                    "selector": candidate.name,
                    "support": support.name,
                    "candidate": {
                        "name": candidate.name,
                        "position": [float(x) for x in candidate.position],
                        "orientation_xyzw": [float(x) for x in candidate.orientation_xyzw],
                    },
                }
        return {"accepted": False}

    def _atomic_station_feasibility(self, target: Any) -> Mapping[str, Any]:
        """Read-only official-map check for one prospective atomic target."""

        tx, ty = _object_xy(target)
        attempts: list[dict[str, Any]] = []
        for desired_distance_m in (0.76, 1.00, 1.20):
            route = _private_official_oracle_route_plan(
                evaluator=self.evaluator,
                target=target,
                desired_distance_m=desired_distance_m,
            )
            try:
                _pose, station_distance_m, selection = (
                    _privileged_symbolic_station_pose(
                        target_xy=(tx, ty),
                        route=route,
                    )
                )
            except RuntimeError:
                attempts.append(
                    {
                        "desired_distance_m": desired_distance_m,
                        "route_status": str(route.status),
                        "accepted": False,
                    }
                )
                continue
            accepted = (
                math.isfinite(station_distance_m)
                and station_distance_m <= 1.70
            )
            attempts.append(
                {
                    "desired_distance_m": desired_distance_m,
                    "route_status": str(route.status),
                    "station_distance_m": station_distance_m,
                    "station_selection": selection,
                    "accepted": accepted,
                }
            )
            if accepted:
                return {
                    "accepted": True,
                    "desired_distance_m": desired_distance_m,
                    "station_distance_m": station_distance_m,
                    "station_selection": selection,
                    "attempts": attempts,
                }
        return {"accepted": False, "attempts": attempts}

    @staticmethod
    def _scene_objects_by_name(evaluator: Any) -> Mapping[str, Any]:
        scene = getattr(getattr(evaluator, "env", None), "scene", None)
        objects = getattr(scene, "objects", None)
        values = objects.values() if isinstance(objects, Mapping) else objects
        result: dict[str, Any] = {}
        for entity in tuple(values or ()):
            name = str(getattr(entity, "name", ""))
            if not name:
                continue
            if name in result and result[name] is not entity:
                raise RuntimeError("live scene object names are ambiguous")
            result[name] = entity
        if not result:
            raise RuntimeError("live scene has no named objects")
        return result

    def _all_scene_objects_by_name(self, evaluator: Any) -> Mapping[str, Any]:
        return self._scene_objects_by_name(evaluator)

    def _active_atomic_operation_marker_objects(
        self,
        evaluator: Any,
        operation: str,
    ) -> Mapping[str, Any]:
        """Expose only the declared subject target for one atomic card.

        Atomic cards validate one already-chosen public action against one
        private fixture.  Letting the spatial Marker backend enumerate every
        object in the loaded house turns one ``place_on_top`` request into
        hundreds of mesh-cell labels and makes the visual evidence unusable.
        This provider is installed only while ``run_atomic_card`` owns the
        episode; ordinary resident/model episodes keep the normal Action20
        candidate provider.
        """

        card = self._active_atomic_card
        if card is None:
            raise RuntimeError("atomic marker provider has no active card")
        if operation != card.action:
            raise RuntimeError("atomic marker operation does not match the card")
        target_name = card.target_name_private
        if target_name is None:
            return {}
        target = self._scene_objects_by_name(evaluator).get(target_name)
        if target is None:
            raise RuntimeError("atomic marker target is absent from the scene")
        return {target_name: target}

    @staticmethod
    def _state_by_name(entity: Any, state_name: str) -> Any:
        states = getattr(entity, "states", None)
        if not isinstance(states, Mapping):
            raise RuntimeError("atomic target has no state registry")
        matches = [
            state
            for state_type, state in states.items()
            if getattr(state_type, "__name__", "") == state_name
        ]
        if len(matches) != 1:
            raise RuntimeError(f"atomic target lacks unique {state_name} state")
        return matches[0]

    def _private_atomic_exact(
        self,
        *,
        runtime: Any,
        episode_dir: Path,
        action_index: int,
        operation: str,
        target: Any,
    ) -> tuple[int, int, Mapping[str, Any]]:
        return _oracle_execute_exact_private_symbolic(
            runtime=runtime,
            phase_dir=episode_dir / "private" / "atomic_setup",
            action_index=action_index,
            target=target,
            operation=operation,
        )

    def _place_source_in_selected_drawer(
        self,
        *,
        source: Any,
        token: Any,
    ) -> Mapping[str, Any]:
        """Move released test content into the selected official drawer volume."""

        regions = drawer_fillable_aabbs(token)
        region_lower, region_upper = max(
            regions,
            key=lambda bounds: float(math.prod(bounds[1] - bounds[0])),
        )
        region_extent = region_upper - region_lower
        import numpy as np

        bbox_provider = getattr(source, "get_base_aligned_bbox", None)
        if not callable(bbox_provider):
            raise RuntimeError("atomic source base-aligned bbox is unavailable")
        _, _, local_extent, local_center = bbox_provider()
        for method in ("detach", "cpu"):
            callback = getattr(local_extent, method, None)
            if callable(callback):
                local_extent = callback()
            callback = getattr(local_center, method, None)
            if callable(callback):
                local_center = callback()
        local_extent = np.asarray(local_extent, dtype=np.float64)
        local_center = np.asarray(local_center, dtype=np.float64)
        if local_extent.shape != (3,) or local_center.shape != (3,):
            raise RuntimeError("atomic source base-aligned bbox is invalid")

        half_turn = math.sqrt(0.5)
        orientations = (
            (
                "axis_aligned",
                np.asarray((0.0, 0.0, 0.0, 1.0)),
                local_extent,
            ),
            (
                "side_x",
                np.asarray((half_turn, 0.0, 0.0, half_turn)),
                local_extent[[0, 2, 1]],
            ),
            (
                "side_y",
                np.asarray((0.0, half_turn, 0.0, half_turn)),
                local_extent[[2, 1, 0]],
            ),
        )
        fits = [
            item
            for item in orientations
            if all(item[2] <= region_extent + 0.010)
        ]
        if not fits:
            raise RuntimeError(
                f"atomic source does not fit drawer extent {region_extent.tolist()}; "
                f"base-aligned extent {local_extent.tolist()}"
            )
        orientation_name, orientation, predicted_extent = max(
            fits,
            key=lambda item: float(
                np.min(region_extent + 0.010 - item[2])
            ),
        )
        desired_center = (region_lower + region_upper) / 2
        xyz = orientation[:3]
        rotated_center = (
            local_center
            + 2 * orientation[3] * np.cross(xyz, local_center)
            + 2 * np.cross(xyz, np.cross(xyz, local_center))
        )
        source.set_position_orientation(
            position=desired_center - rotated_center,
            orientation=orientation,
        )
        keep_still = getattr(source, "keep_still", None)
        if callable(keep_still):
            keep_still()
        return {
            "operation": "place_inside_selected_drawer",
            "selected_joint_private": token.selector,
            "selected_link_private": token.link_prim_path,
            "placement_orientation_private": orientation_name,
            "placed_extent_private": predicted_extent.tolist(),
            "simulator_steps": 0,
        }

    def _prepare_atomic_card(
        self,
        *,
        card: AtomicActionCard,
        artifact_name: str = "atomic_setup",
    ) -> tuple[Any | None, Any | None, Mapping[str, Any]]:
        runtime = self._require_active_runtime()
        assert self._episode_dir is not None
        objects = self._scene_objects_by_name(self.evaluator)
        target = (
            None
            if card.target_name_private is None
            else objects.get(card.target_name_private)
        )
        source = (
            None
            if card.source_name_private is None
            else objects.get(card.source_name_private)
        )
        if card.target_name_private is not None and target is None:
            raise RuntimeError("atomic target is absent from the loaded scene")
        if card.source_name_private is not None and source is None:
            raise RuntimeError("atomic source is absent from the loaded scene")

        private_index = 0
        private_steps = 0
        operations: list[dict[str, Any]] = []
        for setup in card.private_setup:
            setup_target = source if setup == "hold_source" else target
            if setup_target is None:
                raise RuntimeError("atomic setup target is unavailable")
            if setup == "hold_source":
                operation = "grasp"
                required = runtime.symbolic_primitives._get_obj_in_hand() is not source
            elif setup == "source_inside_target":
                if runtime.symbolic_primitives._get_obj_in_hand() is not source:
                    raise RuntimeError(
                        "atomic inside setup requires the declared held source"
                    )
                operation = "place_inside"
                required = True
            else:
                state_name = "Open" if setup in {
                    "target_open",
                    "target_closed",
                } else "ToggledOn"
                desired = setup in {"target_open", "target_on"}
                getter = getattr(
                    self._state_by_name(setup_target, state_name),
                    "get_value",
                    None,
                )
                if not callable(getter):
                    raise RuntimeError("atomic setup state is unreadable")
                required = bool(getter()) is not desired
                operation = (
                    "open" if setup == "target_open"
                    else "close" if setup == "target_closed"
                    else "toggle_on" if setup == "target_on"
                    else "toggle_off"
                )
            if required:
                private_index, used, report = self._private_atomic_exact(
                    runtime=runtime,
                    episode_dir=self._episode_dir,
                    action_index=private_index,
                    operation=operation,
                    target=setup_target,
                )
                private_steps += used
                operations.append(
                    {
                        "setup": setup,
                        "operation": operation,
                        "status": str(report.get("status", "")),
                        "simulator_steps": used,
                    }
                )
            else:
                operations.append(
                    {
                        "setup": setup,
                        "operation": operation,
                        "status": "already_satisfied",
                        "simulator_steps": 0,
                    }
                )

            if setup == "source_inside_target":
                getter = getattr(
                    self._state_by_name(source, "Inside"), "get_value", None
                )
                if (
                    not callable(getter)
                    or not bool(getter(target))
                    or runtime.symbolic_primitives._get_obj_in_hand() is not None
                ):
                    raise RuntimeError("atomic inside setup did not persist")
        navigation: Mapping[str, Any] | None = None
        navigation_attempts: list[dict[str, Any]] = []
        if target is not None:
            # A fixed centre distance is brittle across objects of very
            # different footprints (a small glass, a cabinet, and a fridge).
            # Try three still-local stations and let the official map planner
            # choose the first traversable one.  Every accepted station remains
            # inside the public 1.70 m direct-workspace limit.
            for desired_distance_m in (0.76, 1.00, 1.20):
                private_index, used, attempted_navigation = (
                    _oracle_navigate_near_with_privileged_symbolic(
                        runtime=runtime,
                        evaluator=self.evaluator,
                        target=target,
                        phase_dir=(
                            self._episode_dir / "private" / artifact_name
                        ),
                        action_index=private_index,
                        desired_distance_m=desired_distance_m,
                    )
                )
                private_steps += used
                navigation_attempts.append(
                    {
                        "desired_distance_m": desired_distance_m,
                        "status": str(attempted_navigation.get("status", "")),
                        "simulator_steps": used,
                    }
                )
                if attempted_navigation.get("status") == "near":
                    navigation = attempted_navigation
                    break
            if navigation is None:
                raise RuntimeError("atomic private station is unavailable")

        # Private setup can move the base, open a target, or establish a
        # logical hold without updating evaluator.obs.  Render the resulting
        # physical state before the first public action so Marker geometry and
        # the RGB returned to the caller share one current camera pose.  This
        # is render-only and does not advance the public Action20 sequence.
        observation = _capture_action_boundary_observation(
            runtime,
            force_render=True,
        )
        report = {
            "schema_version": "behavior-hl-private-action20-atomic-setup/v1",
            "card": card.to_private_record(),
            "private_setup_operations": operations,
            "private_setup_action_count": private_index,
            "private_setup_simulator_steps": private_steps,
            "navigation": None if navigation is None else dict(navigation),
            "navigation_attempts": navigation_attempts,
            "public_action_index_after_setup": self._sequence,
            "observation_simulator_step": observation.simulator_step,
        }
        _write_private_json(
            self._episode_dir / "private" / f"{artifact_name}.json",
            report,
        )
        return target, source, report

    def prepare_cumulative_inside_subject(
        self,
        *,
        source_name_private: str,
        target_name_private: str,
    ) -> Mapping[str, Any]:
        """Privately hold one task object for cumulative PlaceInside Live tests."""

        if (
            not isinstance(source_name_private, str)
            or not isinstance(target_name_private, str)
            or source_name_private == target_name_private
        ):
            raise SceneProtocolError("cumulative inside object binding is invalid")
        runtime = self._require_active_runtime()
        if runtime.symbolic_primitives._get_obj_in_hand() is not None:
            raise SceneProtocolError("cumulative inside setup requires an empty hand")
        graspables, containers = private_inside_task_roles(
            self.evaluator,
            self.profile,
        )
        if (
            source_name_private not in containers
            and source_name_private not in graspables
        ):
            raise SceneProtocolError("cumulative inside source is not a task object")
        if target_name_private not in containers:
            raise SceneProtocolError("cumulative inside target is not a task container")
        artifact_name = f"cumulative_inside_{self._sequence:03d}"
        card = AtomicActionCard(
            card_id=artifact_name,
            action="place_inside",
            target_name_private=target_name_private,
            source_name_private=source_name_private,
            private_setup=("hold_source",),
            selector_private=None,
        )
        _target, _source, setup = self._prepare_atomic_card(
            card=card,
            artifact_name=artifact_name,
        )
        return {
            "status": "PASS",
            "prepared": True,
            "public_action_index_after_setup": setup[
                "public_action_index_after_setup"
            ],
            "private_setup_simulator_steps": setup[
                "private_setup_simulator_steps"
            ],
        }

    def _atomic_semantic_postcondition(
        self,
        *,
        card: AtomicActionCard,
        target: Any | None,
        source: Any | None,
    ) -> Mapping[str, Any]:
        runtime = self._require_active_runtime()
        held = runtime.symbolic_primitives._get_obj_in_hand()
        record: dict[str, Any] = {
            "action": card.action,
            "logical_hand_empty": held is None,
        }
        if card.action == "pick":
            passed = held is target
            record["held_object_is_target"] = passed
        elif card.action == "place_on_floor":
            passed = held is None
        elif card.action.startswith("place_"):
            predicate = {
                "place_inside": "Inside",
                "place_on_top": "OnTop",
                "place_under": "Under",
                "place_next_to": "NextTo",
            }[card.action]
            value = read_action_relation(source, predicate, target, predicate_mode=runtime.predicate_mode)
            record["predicate_mode"] = runtime.predicate_mode
            try:
                record["raw_official_predicate_value"] = (
                    value if runtime.predicate_mode == "official"
                    else read_action_relation(source, predicate, target, predicate_mode="official")
                )
            except Exception as exc:
                record["raw_official_predicate_value"] = None
                record["raw_official_predicate_error"] = f"{type(exc).__name__}: {exc}"
            passed = value and held is None
            record.update(
                {"canonical_predicate": predicate, "predicate_value": value}
            )
        elif card.action in {"open", "close"}:
            # ``completed`` is emitted only after the runtime's private
            # outcome verifier reads the exact selected whole-object or drawer
            # part state.  Requiring the aggregate Open state of a multi-drawer
            # cabinet would incorrectly reject a successfully closed selected
            # drawer merely because another drawer remains open.
            passed = True
            record["selected_target_state_verified_by_executor"] = True
            if card.action == "close" and source is not None:
                drawer_report = getattr(
                    runtime.symbolic_primitives,
                    "_last_marker_symbolic_drawer_part_report",
                    {},
                )
                getter = getattr(
                    self._state_by_name(source, "Inside"), "get_value", None
                )
                inside = bool(callable(getter) and getter(target))
                content_count = int(drawer_report.get("content_count_private", 0))
                passed = (
                    inside
                    and content_count > 0
                    and drawer_report.get("blocked") is False
                )
                record.update(
                    {
                        "source_remains_inside": inside,
                        "selected_drawer_content_count_private": content_count,
                    }
                )
        elif card.action in {"toggle_on", "toggle_off"}:
            getter = getattr(
                self._state_by_name(target, "ToggledOn"), "get_value", None
            )
            value = bool(callable(getter) and getter())
            expected = card.action == "toggle_on"
            passed = value is expected
            record.update({"toggled_on_value": value, "expected": expected})
        else:
            # Motion, look, and wait are certified by a completed public
            # result plus a fresh finite RGB action-after observation.
            passed = True
        record["passed"] = bool(passed)
        return record

    def _execute_atomic_subject(
        self,
        *,
        card: AtomicActionCard,
        target: Any | None,
        source: Any | None,
        setup: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        turns: list[Mapping[str, Any]] = []
        if card.action not in ATOMIC_MARKER_ACTIONS:
            turns.append(
                self._step_impl(
                    sequence=self._sequence,
                    action=card.action,
                    target=None,
                )
            )
            return turns

        first = self._step_impl(
            sequence=self._sequence,
            action=card.action,
            target=None,
        )
        turns.append(first)
        if first["observation"]["kind"] != "marker":
            return turns
        snapshot = self._require_active_runtime().store.active_snapshot
        first_target = marker_number_for_binding(
            snapshot,
            target=target,
            selector=(
                card.selector_private
                if card.action in {"open", "close"}
                else None
            ),
            binding_kinds=("object", "part"),
            require_unique=(
                card.action not in {"open", "close"}
                or card.selector_private is not None
            ),
        )
        if card.action == "close" and source is not None:
            binding = snapshot.bindings[f"C{first_target}"]
            selected_report = self._place_source_in_selected_drawer(
                source=source,
                token=binding.target_token,
            )
            updated_setup = dict(setup)
            updated_setup["private_setup_operations"] = [
                *setup["private_setup_operations"],
                dict(selected_report),
            ]
            assert self._episode_dir is not None
            _write_private_json(
                self._episode_dir / "private" / "atomic_setup.json",
                updated_setup,
            )
        selected = self._step_impl(
            sequence=self._sequence,
            action=card.action,
            target=first_target,
        )
        turns.append(selected)
        if selected["observation"]["kind"] != "marker":
            return turns
        second_snapshot = self._require_active_runtime().store.active_snapshot
        second_kind = {
            "place_inside": "inside_cell",
            "place_on_top": "surface_cell",
            "place_next_to": "position",
        }.get(card.action)
        if second_kind is None:
            raise RuntimeError("atomic action returned an unexpected Marker stage")
        second_target = marker_number_for_binding(
            second_snapshot,
            target=target,
            selector=card.selector_private,
            binding_kinds=(second_kind,),
            require_unique=card.selector_private is not None,
            fallback_to_any_selector=True,
        )
        turns.append(
            self._step_impl(
                sequence=self._sequence,
                action=card.action,
                target=second_target,
            )
        )
        return turns

    def _finish_atomic_validation_failure(
        self,
        *,
        card: AtomicActionCard,
        setup: Mapping[str, Any],
        turns: Sequence[Mapping[str, Any]],
        failure_stage: str,
        semantic: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Return one diagnostic card failure without an environment error."""

        assert failure_stage in {"public_action_status", "semantic_postcondition"}
        assert self._episode_dir is not None
        summary: dict[str, Any] = {
            "schema_version": "behavior-hl-action20-atomic-result/v1",
            "status": "FAIL",
            "failure_type": "ATOMIC_VALIDATION_FAILED",
            "failure_stage": failure_stage,
            "card_id": card.card_id,
            "action": card.action,
            "public_turn_count": len(turns),
            "public_action_statuses": [
                turn["action_result"]["status"] for turn in turns
            ],
            "same_scene_reset": True,
            "scene_reconstructed": False,
            "artifact_dir": str(self._episode_dir),
            "private_setup_simulator_steps": setup[
                "private_setup_simulator_steps"
            ],
        }
        if semantic is not None:
            summary["semantic_postcondition"] = dict(semantic)
        _write_json(self._episode_dir / "atomic_result.json", summary)
        end = self.end_episode(reason="client_error")
        return {**summary, "session_valid": end["session_valid"]}

    def run_atomic_card(self, *, card_id: str) -> Mapping[str, Any]:
        if not self._session_valid:
            raise SceneProtocolError("resident scene session is invalid")
        if self.episode_active:
            raise SceneProtocolError("an episode is already active")
        card = self._atomic_cards.get(card_id)
        if card is None:
            raise SceneProtocolError("atomic card is not declared for this scene")
        self._atomic_run_count += 1
        episode_id = f"atomic_{self._atomic_run_count:03d}_{card.card_id}"
        self._active_atomic_card = card
        try:
            self._begin_episode_impl(episode_id=episode_id, fixture_id=None)
            target, source, setup = self._prepare_atomic_card(card=card)
            turns = self._execute_atomic_subject(
                card=card,
                target=target,
                source=source,
                setup=setup,
            )
            final_statuses = [
                turn["action_result"]["status"] for turn in turns
            ]
            if not final_statuses or final_statuses[-1] != "completed":
                return self._finish_atomic_validation_failure(
                    card=card,
                    setup=setup,
                    turns=turns,
                    failure_stage="public_action_status",
                )
            semantic = self._atomic_semantic_postcondition(
                card=card,
                target=target,
                source=source,
            )
            if semantic.get("passed") is not True:
                return self._finish_atomic_validation_failure(
                    card=card,
                    setup=setup,
                    turns=turns,
                    failure_stage="semantic_postcondition",
                    semantic=semantic,
                )
            assert self._episode_dir is not None
            artifact_dir = str(self._episode_dir)
            summary = {
                "schema_version": "behavior-hl-action20-atomic-result/v1",
                "status": "PASS",
                "card_id": card.card_id,
                "action": card.action,
                "public_turn_count": len(turns),
                "public_action_statuses": final_statuses,
                "semantic_postcondition": dict(semantic),
                "same_scene_reset": True,
                "scene_reconstructed": False,
                "artifact_dir": artifact_dir,
                "private_setup_simulator_steps": setup[
                    "private_setup_simulator_steps"
                ],
            }
            _write_json(self._episode_dir / "atomic_result.json", summary)
            end = self.end_episode(reason="done")
            return {**summary, "session_valid": end["session_valid"]}
        except Exception:
            if self.episode_active:
                self._session_valid = self._recover_failed_episode_runtime(
                    reason="atomic_card_environment_error"
                )
            raise
        finally:
            self._active_atomic_card = None

    def _rolling_checkpoint_index_path(self) -> Path:
        ring = self._rolling_checkpoints
        if ring is None:
            raise RuntimeError("rolling checkpoint ring is unavailable")
        # A retained checkpoint may outlive end_episode, so derive the path
        # from the retained payload instead of the cleared active fields.
        latest = ring.require()
        payload = latest.payload
        if not isinstance(payload, _ResidentRuntimeCheckpoint):
            raise RuntimeError("rolling checkpoint payload type changed")
        return (
            payload.episode_dir
            / "private"
            / "rolling_checkpoints"
            / "index.json"
        )

    def _write_rolling_checkpoint_index(self) -> None:
        ring = self._rolling_checkpoints
        if ring is None or not ring.captured_action_counts:
            return
        record = dict(ring.to_private_record())
        enriched = []
        for item in record["retained"]:
            action_count = item["captured_action_count"]
            payload = ring.require(action_count).payload
            if not isinstance(payload, _ResidentRuntimeCheckpoint):
                raise RuntimeError("rolling checkpoint payload type changed")
            enriched.append(
                {
                    **dict(item),
                    "simulator_state_bytes_private": (
                        payload.simulator_state_bytes_private
                    ),
                }
            )
        record["retained"] = enriched
        record["retained_simulator_state_bytes_private"] = sum(
            item["simulator_state_bytes_private"] for item in enriched
        )
        _write_private_json(
            self._rolling_checkpoint_index_path(),
            record,
        )

    def _rolling_checkpoint_safe_boundary(self, runtime: Any) -> bool:
        primitives = runtime.symbolic_primitives
        return bool(
            self._pending_marker_action is None
            and not self._pending_marker_targets
            and self._pending_marker_stage is None
            and getattr(runtime.store, "active_snapshot", None) is None
            and getattr(
                primitives,
                "_logical_inventory_visual_projection_active",
                False,
            )
            is False
            and getattr(primitives, "_placing_with_predicate", False) is False
            and getattr(primitives, "_private_exact_operation_active", None)
            is None
        )

    def _capture_resident_runtime_checkpoint(
        self,
    ) -> _ResidentRuntimeCheckpoint:
        runtime = self._require_active_runtime()
        if not self._rolling_checkpoint_safe_boundary(runtime):
            raise RuntimeError("rolling checkpoint boundary is not clean")
        if (
            self._episode_id is None
            or self._episode_dir is None
            or self._episode is None
            or self._verifier is None
            or self._initial_verifier is None
            or self._last_verifier is None
            or self._sequence <= 0
        ):
            raise RuntimeError("rolling checkpoint episode state is incomplete")
        # A serialized tensor is equally valid for same-process rollback and
        # can also be copied to disk for a later resident process.  Keeping a
        # single representation avoids two simulator snapshots at milestone
        # boundaries.
        simulator_state = _checkpoint_simulator().dump_state(serialized=True)
        logical_state = _capture_private_inside_action_logical_checkpoint(
            runtime.symbolic_primitives
        )
        tracker = runtime.carry_tracker
        missing = tuple(
            name
            for name in _CARRY_TRACKER_CHECKPOINT_ATTRIBUTES
            if not hasattr(tracker, name)
        )
        if missing:
            raise RuntimeError(
                "rolling checkpoint carry tracker contract changed"
            )
        return _ResidentRuntimeCheckpoint(
            episode_id=self._episode_id,
            episode_dir=self._episode_dir,
            runtime=runtime,
            episode=self._episode,
            verifier=self._verifier,
            initial_verifier=self._initial_verifier,
            last_verifier=self._last_verifier,
            sequence=self._sequence,
            simulator_state=simulator_state,
            simulator_state_bytes_private=_checkpoint_payload_bytes(
                simulator_state
            ),
            logical_state=logical_state,
            carry_tracker_state=tuple(
                (name, getattr(tracker, name))
                for name in _CARRY_TRACKER_CHECKPOINT_ATTRIBUTES
            ),
        )

    def _persistent_checkpoint_identity(self) -> Mapping[str, Any]:
        return {
            "task_id": int(self.profile.task_id),
            "task_name": str(self.profile.task_name),
            "task_split": str(self.profile.split),
            "task_instance_id": str(self.profile.instance_id),
            "task_seed": int(self.seed),
            "scene_model": str(self.profile.scene_model),
            "scene_load_mode": self.scene_load_mode,
        }

    def _encode_checkpoint_runtime_value(self, value: Any) -> Any:
        objects = self._all_scene_objects_by_name(self.evaluator)
        names_by_identity = {id(entity): name for name, entity in objects.items()}

        def object_name_for(item: Any) -> str | None:
            return names_by_identity.get(id(item))

        def dataclass_encoder(
            item: Any,
        ) -> tuple[str, Mapping[str, Any]] | None:
            if isinstance(item, LogicalInventoryEntry):
                return (
                    "LogicalInventoryEntry/v1",
                    {
                        "object_ref": item.object_ref,
                        "origin_pose": item.origin_pose,
                        "relative_to_root": item.relative_to_root,
                    },
                )
            if isinstance(item, LogicalInventorySnapshot):
                return (
                    "LogicalInventorySnapshot/v2",
                    {
                        "root": item.root,
                        "root_origin_pose": item.root_origin_pose,
                        "entries": item.entries,
                        "containment_edges": item.containment_edges,
                        "root_private_pose": item.root_private_pose,
                        "sync_count": item.sync_count,
                    },
                )
            return None

        return encode_portable_value(
            value,
            object_name_for=object_name_for,
            dataclass_encoder=dataclass_encoder,
        )

    def _decode_checkpoint_runtime_value(self, value: Any) -> Any:
        objects = self._all_scene_objects_by_name(self.evaluator)

        def object_for_name(name: str) -> Any:
            return objects.get(name)

        def dataclass_decoder(tag: str, fields: Mapping[str, Any]) -> Any:
            if tag == "LogicalInventoryEntry/v1" and set(fields) == {
                "object_ref", "origin_pose", "relative_to_root"
            }:
                return LogicalInventoryEntry(**fields)
            if tag == "LogicalInventorySnapshot/v2" and set(fields) == {
                "root", "root_origin_pose", "entries", "containment_edges",
                "root_private_pose", "sync_count",
            }:
                return LogicalInventorySnapshot(**fields)
            raise PersistentCheckpointError(
                "persistent checkpoint dataclass tag is unsupported"
            )

        def tensor_factory(array: Any) -> Any:
            import torch

            return torch.from_numpy(array.copy())

        return decode_portable_value(
            value,
            object_for_name=object_for_name,
            dataclass_decoder=dataclass_decoder,
            tensor_factory=tensor_factory,
        )

    def _persist_resident_runtime_checkpoint(
        self,
        *,
        checkpoint: _ResidentRuntimeCheckpoint,
        retained_action_counts: list[int],
    ) -> Mapping[str, Any]:
        logical_record = self._encode_checkpoint_runtime_value(
            {
                "held_object": checkpoint.logical_state.held_object,
                "attribute_values": checkpoint.logical_state.attribute_values,
            }
        )
        carry_values: dict[str, Any] = {}
        for name, value in checkpoint.carry_tracker_state:
            enum_value = getattr(value, "value", None)
            carry_values[name] = (
                enum_value
                if name in {"_confidence", "_acquired_via"}
                and isinstance(enum_value, str)
                else value
            )
        carry_record = self._encode_checkpoint_runtime_value(carry_values)
        return write_persistent_checkpoint(
            root=(
                checkpoint.episode_dir
                / "private"
                / "rolling_checkpoints"
                / "persistent"
            ),
            captured_action_count=checkpoint.sequence,
            source_episode_id=checkpoint.episode_id,
            identity=self._persistent_checkpoint_identity(),
            compatibility={
                "persistent_checkpoint_version": (
                    PERSISTENT_SCENE_CHECKPOINT_VERSION
                ),
                "scene_server_version_diagnostic": SCENE_SERVER_VERSION,
                "rolling_action_interval": self._rolling_checkpoint_action_interval,
            },
            simulator_state=_checkpoint_state_numpy(
                checkpoint.simulator_state
            ),
            logical_state=logical_record,
            carry_tracker_state=carry_record,
            verifier=checkpoint.last_verifier.to_private_record(),
            retained_action_counts=retained_action_counts,
        )

    def _maybe_capture_rolling_checkpoint(self) -> Mapping[str, Any] | None:
        # ponytail: temporal rollback needs event-state serialization before enabling it.
        if getattr(self, '_process_audit', None) is not None:
            return None
        ring = self._rolling_checkpoints
        if ring is None:
            raise RuntimeError("active episode lacks its checkpoint ring")
        runtime = self._require_active_runtime()
        safe = self._rolling_checkpoint_safe_boundary(runtime)
        try:
            entry = ring.capture_if_due(
                completed_action_count=self._sequence,
                safe_boundary=safe,
                capture=self._capture_resident_runtime_checkpoint,
            )
        except Exception as exc:
            assert self._episode_dir is not None
            _append_jsonl(
                self._episode_dir
                / "private"
                / "rolling_checkpoints"
                / "capture_failures.jsonl",
                {
                    "sequence": self._sequence,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retained_action_counts": list(
                        ring.captured_action_counts
                    ),
                },
            )
            return None
        if entry is None:
            return None
        self._write_rolling_checkpoint_index()
        payload = entry.payload
        assert isinstance(payload, _ResidentRuntimeCheckpoint)
        persistent: Mapping[str, Any] | None = None
        try:
            persistent = self._persist_resident_runtime_checkpoint(
                checkpoint=payload,
                retained_action_counts=list(ring.captured_action_counts),
            )
        except Exception as exc:
            assert self._episode_dir is not None
            _append_jsonl(
                self._episode_dir
                / "private"
                / "rolling_checkpoints"
                / "persistent_capture_failures.jsonl",
                {
                    "sequence": self._sequence,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "in_memory_checkpoint_preserved": True,
                },
            )
        return {
            "milestone_action_count": entry.milestone_action_count,
            "captured_action_count": entry.captured_action_count,
            "retained_action_counts": list(ring.captured_action_counts),
            "simulator_state_bytes_private": (
                payload.simulator_state_bytes_private
            ),
            "persistent_checkpoint": (
                None if persistent is None else dict(persistent)
            ),
        }

    @staticmethod
    def _verifier_checkpoint_matches(expected: Any, actual: Any) -> bool:
        return bool(
            getattr(expected, "leaf_truth", None)
            == getattr(actual, "leaf_truth", None)
            and getattr(expected, "official_success", None)
            == getattr(actual, "official_success", None)
            and getattr(expected, "official_q", None)
            == getattr(actual, "official_q", None)
        )

    def restore_checkpoint(
        self,
        *,
        captured_action_count: int | None = None,
    ) -> Mapping[str, Any]:
        """Roll the resident process back to one retained clean boundary.

        This is an owner-only control operation.  It restores simulator and
        Python-owned carry state in the same live process; it is never exposed
        to the model as an Action20 tool.
        """

        if not self._session_valid:
            raise SceneProtocolError("resident scene session is invalid")
        if getattr(self, '_process_audit', None) is not None:
            raise SceneProtocolError("process-history checkpoint restore is not supported")
        ring = self._rolling_checkpoints
        if ring is None:
            raise SceneProtocolError("resident episode has no checkpoint ring")
        try:
            entry = ring.require(captured_action_count)
        except RollingCheckpointError as exc:
            raise SceneProtocolError(str(exc)) from exc
        checkpoint = entry.payload
        if not isinstance(checkpoint, _ResidentRuntimeCheckpoint):
            raise RuntimeError("rolling checkpoint payload type changed")
        if self.episode_active and self._episode_id != checkpoint.episode_id:
            raise SceneProtocolError("checkpoint belongs to another active episode")
        mutation_started = False
        try:
            runtime = checkpoint.runtime
            cleanup = runtime.symbolic_primitives.cleanup_symbolic_collision_filter()
            if cleanup.get("active") is not False:
                raise RuntimeError(
                    "rolling checkpoint collision filter cleanup failed"
                )
            mutation_started = True
            _checkpoint_simulator().load_state(
                checkpoint.simulator_state,
                serialized=True,
            )
            _restore_private_inside_action_logical_checkpoint(
                runtime.symbolic_primitives,
                checkpoint.logical_state,
            )
            for name, value in checkpoint.carry_tracker_state:
                setattr(runtime.carry_tracker, name, value)
            runtime.store.clear()
            # Never reuse a frame/controller cache created after the rollback
            # point.  The public simulator step counter remains monotonic.
            environment = runtime.environment
            environment._latest = None
            environment._controller_snapshot = None
            environment._frozen = {}
            self._runtime = runtime
            self._episode = checkpoint.episode
            self._episode_id = checkpoint.episode_id
            self._episode_dir = checkpoint.episode_dir
            self._verifier = checkpoint.verifier
            self._initial_verifier = checkpoint.initial_verifier
            self._last_verifier = checkpoint.last_verifier
            self._sequence = checkpoint.sequence
            self._clear_marker_state()
            restore_index = ring.record_restore(
                captured_action_count=checkpoint.sequence
            )
            restore_id = (
                f"restore_{checkpoint.sequence:04d}_{restore_index:02d}"
            )
            self._action_artifact_dir = (
                checkpoint.episode_dir / "branches" / restore_id
            )
            self._observation_id_prefix = (
                f"{checkpoint.episode_id}_{restore_id}"
            )
            # OmniGibson explicitly requires one propagation step after
            # load_state before spatial predicates become authoritative.
            propagation_steps = environment.checkpoint_once()
            synchronize_hold = getattr(
                runtime.symbolic_primitives,
                "sync_marker_kinematic_hold",
                None,
            )
            if callable(synchronize_hold):
                synchronize_hold()
            observation = _capture_action_boundary_observation(
                runtime,
                force_render=True,
            )
            actual = self._verifier.evaluate(self._episode, self.evaluator)
            if not self._verifier_checkpoint_matches(
                checkpoint.last_verifier,
                actual,
            ):
                raise RuntimeError(
                    "rolling checkpoint verifier state changed after restore"
                )
            self._last_verifier = actual
            public = _freeze_public_observation(
                runtime=runtime,
                observation=observation,
                observation_id=(
                    f"{checkpoint.episode_id}_{restore_id}_physical"
                ),
                directory=(
                    self._action_artifact_dir / "observations" / "restored"
                ),
                kind="physical",
                marker_targets=(),
            )
            _write_private_json(
                checkpoint.episode_dir
                / "private"
                / "rolling_checkpoints"
                / "restores"
                / f"{restore_id}.json",
                {
                    "status": "PASS",
                    "episode_id": checkpoint.episode_id,
                    "captured_action_count": checkpoint.sequence,
                    "restore_index": restore_index,
                    "post_load_propagation_steps": propagation_steps,
                    "verifier_exact_match": True,
                    "artifact_branch": str(self._action_artifact_dir),
                },
            )
            self._write_rolling_checkpoint_index()
            return {
                "status": "PASS",
                "episode_id": checkpoint.episode_id,
                "next_sequence": checkpoint.sequence,
                "restored_action_count": checkpoint.sequence,
                "post_load_propagation_steps": propagation_steps,
                "observation": public.to_record(),
            }
        except Exception:
            if mutation_started:
                self._session_valid = False
            raise

    @staticmethod
    def _verifier_record_matches(
        expected: Mapping[str, Any],
        actual: Any,
    ) -> bool:
        return bool(
            expected.get("ok") is True
            and expected.get("official_success")
            == getattr(actual, "official_success", None)
            and expected.get("official_q")
            == getattr(actual, "official_q", None)
            and expected.get("leaf_truth")
            == dict(getattr(actual, "leaf_truth", {}))
        )

    def restore_persistent_checkpoint(
        self,
        *,
        manifest_path: str,
        manifest_sha256: str,
        episode_id: str,
    ) -> Mapping[str, Any]:
        """Restore a verified milestone into this newly loaded resident.

        The scene must already be loaded normally.  This operation is private
        control-plane functionality; no checkpoint path or state is exposed
        through the model's Action20 interface.
        """

        if not self._session_valid:
            raise SceneProtocolError("resident scene session is invalid")
        if self.episode_active:
            raise SceneProtocolError(
                "persistent checkpoint restore requires no active episode"
            )
        task = getattr(self, '_registered_task', None)
        if task is not None and task.record.get('task_contract', {}).get('process_requirements') is not None:
            raise SceneProtocolError("process-history checkpoint restore is not supported")
        if not isinstance(manifest_path, str) or not manifest_path:
            raise SceneProtocolError("persistent checkpoint path is invalid")
        if not isinstance(manifest_sha256, str):
            raise SceneProtocolError("persistent checkpoint hash is invalid")
        loaded = load_persistent_checkpoint(
            manifest_path=Path(manifest_path),
            expected_manifest_sha256=manifest_sha256,
            expected_identity=self._persistent_checkpoint_identity(),
            allowed_root=self._runtime_registry,
        )

        # Reuse all existing begin-episode gates to construct the current
        # runtime, verifier, cameras, safety services, and clean baseline.
        self.begin_episode(episode_id=episode_id, fixture_id=None)
        mutation_started = False
        try:
            runtime = self._require_active_runtime()
            logical = self._decode_checkpoint_runtime_value(
                loaded.logical_state
            )
            carry = self._decode_checkpoint_runtime_value(
                loaded.carry_tracker_state
            )
            if (
                not isinstance(logical, Mapping)
                or set(logical) != {"held_object", "attribute_values"}
                or not isinstance(logical["attribute_values"], tuple)
                or not isinstance(carry, Mapping)
                or set(carry) != set(_CARRY_TRACKER_CHECKPOINT_ATTRIBUTES)
            ):
                raise PersistentCheckpointError(
                    "persistent Python runtime state contract changed"
                )
            logical_template = (
                _capture_private_inside_action_logical_checkpoint(
                    runtime.symbolic_primitives
                )
            )
            logical_checkpoint = type(logical_template)(
                held_object=logical["held_object"],
                attribute_values=logical["attribute_values"],
            )

            import torch

            simulator_state = torch.from_numpy(
                loaded.simulator_state.copy()
            )
            mutation_started = True
            _checkpoint_simulator().load_state(
                simulator_state,
                serialized=True,
            )
            _restore_private_inside_action_logical_checkpoint(
                runtime.symbolic_primitives,
                logical_checkpoint,
            )
            for name in _CARRY_TRACKER_CHECKPOINT_ATTRIBUTES:
                value = carry[name]
                if name in {"_confidence", "_acquired_via"}:
                    current = getattr(runtime.carry_tracker, name)
                    value = type(current)(value)
                setattr(runtime.carry_tracker, name, value)
            runtime.store.clear()
            environment = runtime.environment
            environment._latest = None
            environment._controller_snapshot = None
            environment._frozen = {}
            self._sequence = loaded.captured_action_count
            self._clear_marker_state()
            restore_id = f"persistent_restore_{self._sequence:04d}"
            assert self._episode_dir is not None
            assert self._episode_id is not None
            assert self._verifier is not None
            assert self._episode is not None
            self._action_artifact_dir = (
                self._episode_dir / "branches" / restore_id
            )
            self._observation_id_prefix = (
                f"{self._episode_id}_{restore_id}"
            )
            propagation_steps = environment.checkpoint_once()
            synchronize_hold = getattr(
                runtime.symbolic_primitives,
                "sync_marker_kinematic_hold",
                None,
            )
            if callable(synchronize_hold):
                synchronize_hold()
            observation = _capture_action_boundary_observation(
                runtime,
                force_render=True,
            )
            actual = self._verifier.evaluate(self._episode, self.evaluator)
            if not self._verifier_record_matches(loaded.verifier, actual):
                raise RuntimeError(
                    "persistent checkpoint official verifier readback changed"
                )
            self._last_verifier = actual
            ring = self._rolling_checkpoints
            if ring is None:
                raise RuntimeError("persistent restore lacks checkpoint ring")
            adopted_payload = self._capture_resident_runtime_checkpoint()
            ring.adopt_restored_payload(
                captured_action_count=self._sequence,
                payload=adopted_payload,
            )
            public = _freeze_public_observation(
                runtime=runtime,
                observation=observation,
                observation_id=(
                    f"{self._episode_id}_{restore_id}_physical"
                ),
                directory=(
                    self._action_artifact_dir
                    / "observations"
                    / "restored"
                ),
                kind="physical",
                marker_targets=(),
            )
            record = {
                "status": "PASS",
                "source_episode_id": loaded.source_episode_id,
                "new_episode_id": self._episode_id,
                "captured_action_count": self._sequence,
                "manifest_path": str(loaded.manifest_path),
                "manifest_sha256": loaded.manifest_sha256,
                "post_load_propagation_steps": propagation_steps,
                "verifier_exact_match": True,
                "task_identity_exact_match": True,
                "source_version_is_diagnostic": True,
            }
            _write_private_json(
                self._episode_dir
                / "private"
                / "persistent_checkpoint_restore.json",
                record,
            )
            self._write_rolling_checkpoint_index()
            return {
                "status": "PASS",
                "episode_id": self._episode_id,
                "next_sequence": self._sequence,
                "restored_action_count": self._sequence,
                "post_load_propagation_steps": propagation_steps,
                "observation": public.to_record(),
            }
        except Exception:
            if self.episode_active:
                recoverable = self._recover_failed_episode_runtime(
                    reason="persistent_checkpoint_restore_error"
                )
                self._session_valid = bool(recoverable and not mutation_started)
            raise

    def _make_l0_verifier(self, episode_id: str, reset_count: int) -> Any:
        from behavior_hl.embodied.evaluator_bridge.l0_taskspec_live_adapter import (
            L0TaskSpecEpisodeVerifier,
        )

        assert self._l0_task_spec is not None
        return L0TaskSpecEpisodeVerifier(
            self._l0_task_spec,
            evaluator_state=self.evaluator,
            runtime_identity={
                "task_id": self.profile.task_id,
                "task_name": self.profile.task_name,
                "scene_model": self.profile.scene_model,
                "split": self.profile.split,
                "instance_id": int(self.profile.instance_id),
            },
            marker_session_id=f"{episode_id}:reset{reset_count}",
            predicate_mode=self._runtime.predicate_mode,
        )

    def begin_episode(
        self,
        *,
        episode_id: str,
        fixture_id: str | None = None,
        marker_depth_protocol: str | None = None,
        overhead_view_mode: str = "off",
        automatic_interaction_approach_enabled: bool = False,
    ) -> Mapping[str, Any]:
        require_execution_integrity(self.evaluator)
        if not self._session_valid:
            raise SceneProtocolError("resident scene session is invalid")
        if self.episode_active:
            raise SceneProtocolError("an episode is already active")
        if marker_depth_protocol not in {None, RGBD_PROTOCOL_VERSION}:
            raise SceneProtocolError("marker depth protocol is invalid")
        if marker_depth_protocol is not None and self.rgbd_protocol_version is None:
            raise SceneProtocolError("marker depth requires an RGB-D resident")
        if overhead_view_mode not in OVERHEAD_VIEW_MODES:
            raise SceneProtocolError("overhead view mode is invalid")
        if (
            overhead_view_mode != "off"
            and self.observation_profile != THREE_RGB_HEAD_EVAL_CAMERA_PROFILE
        ):
            raise SceneProtocolError("overhead view requires an RGB-only resident")
        if not isinstance(automatic_interaction_approach_enabled, bool):
            raise SceneProtocolError(
                "automatic interaction approach flag is invalid"
            )
        if (
            not isinstance(episode_id, str)
            or not episode_id
            or len(episode_id) > 96
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in episode_id)
        ):
            raise SceneProtocolError("episode_id is invalid")
        if (
            fixture_id is not None
            and (
                not isinstance(fixture_id, str)
                or fixture_id not in self._fixtures
            )
        ):
            raise SceneProtocolError("fixture_id is not declared for this scene")
        try:
            self._marker_depth_protocol = marker_depth_protocol
            self._overhead_view_mode = overhead_view_mode
            self._automatic_interaction_approach_enabled = (
                automatic_interaction_approach_enabled
            )
            return self._begin_episode_impl(
                episode_id=episode_id,
                fixture_id=fixture_id,
            )
        except SceneProtocolError:
            self._marker_depth_protocol = None
            self._overhead_view_mode = "off"
            self._automatic_interaction_approach_enabled = False
            raise
        except Exception:
            self._marker_depth_protocol = None
            self._overhead_view_mode = "off"
            self._automatic_interaction_approach_enabled = False
            self._session_valid = self._recover_failed_episode_runtime(
                reason="begin_episode_environment_error"
            )
            raise

    def _begin_episode_impl(
        self,
        *,
        episode_id: str,
        fixture_id: str | None,
    ) -> Mapping[str, Any]:
        episode_dir = self.run_dir / "episodes" / episode_id
        if episode_dir.exists():
            raise SceneProtocolError("episode artifact directory already exists")
        episode_dir.mkdir(parents=True)
        self._process_audit = None
        if self._rolling_checkpoints is not None:
            # A same-scene reset invalidates every checkpoint from the prior
            # episode.  Release their opaque simulator payloads before reset;
            # their private metadata and images remain in the old artifacts.
            self._rolling_checkpoints.release_payloads()
        self._rolling_checkpoints = RollingEpisodeCheckpointRing(
            episode_id=episode_id,
            action_interval=self._rolling_checkpoint_action_interval,
            maximum_retained=self._rolling_checkpoint_maximum_retained,
        )
        next_count = _reset_same_scene(
            evaluator=self.evaluator,
            control=self.control,
            stages=self.stages,
            phase=f"external_scene_episode_{self._reset_count + 1:04d}",
            run_dir=episode_dir,
            reset_index=self._reset_count,
        )
        if self._prepared_scene_document is not None:
            from copy import deepcopy
            self.evaluator.env.scene.restore(deepcopy(self._prepared_scene_document), update_initial_file=False)
        if self._oracle_prepared_state is not None:
            from behavior_hl.embodied.evaluator_bridge.prepared_model_state import restore_prepared_model_state
            report = restore_prepared_model_state(evaluator=self.evaluator,
                bundle=self._oracle_prepared_state, instruction_id=self._public_instruction_id,
                seed=self.seed, observation_profile=self.observation_profile,
                semantic_check=self._initial_state_validator)
            _write_private_json(episode_dir / 'private/oracle_prepared_restore.json', report)
        post = self.bindings.capture_private_reset_snapshot(
            evaluator=self.evaluator,
            phase="post_reset",
            run_dir=episode_dir,
        )
        strict_attestation = attest_same_scene_reset(
            pre=self._baseline,
            post=post,
            tolerances=SameSceneResetTolerances(
                stable_state_numeric_abs=1e-4,
                stable_state_orientation_rad=5e-3,
            ),
        )
        # The original thresholds remain as private diagnostics, but natural
        # post-reset settling must not invalidate an otherwise readable
        # resident scene.  Admission uses deliberately broad sanity bounds;
        # exact identities, goal truth, schemas, finite values, and vector
        # shapes remain hard requirements inside the shared attestation.
        attestation = attest_same_scene_reset(
            pre=self._baseline,
            post=post,
            tolerances=SameSceneResetTolerances(
                base_position_m=0.30,
                base_orientation_rad=0.50,
                joint_position_rad=0.50,
                stable_state_numeric_abs=1e-2,
                stable_state_orientation_rad=0.50,
            ),
        )
        attestation_record = attestation.to_private_record()
        attestation_record.update(
            {
                "admission_policy": (
                    "broad_numeric_sanity_with_exact_structure_and_semantics_v1"
                ),
                "strict_diagnostic_pass": strict_attestation.passed,
                "strict_diagnostic": strict_attestation.to_private_record(),
                "numeric_drift_is_private_diagnostic": True,
            }
        )
        if self._oracle_prepared_state is not None:
            from behavior_hl.embodied.evaluator_bridge.prepared_model_state import require_semantic_reset
            try:
                attestation_record.update(require_semantic_reset(attestation))
            except Exception:
                _write_json(episode_dir / 'private_reset_attestation.json', attestation_record)
                raise
        _write_json(
            episode_dir / "private_reset_attestation.json",
            attestation_record,
        )
        if self._oracle_prepared_state is None:
            attestation.require_pass()
        if self._option_owner is not None:
            owner = resolve_grounded_option_identity_owner(
                evaluator=self.evaluator,
                profile=self.profile,
            )
            if (
                owner.witness.option_index
                != self._option_owner.witness.option_index
                or owner.witness.identity != self._option_owner.witness.identity
            ):
                raise RuntimeError(
                    "selected option identity drifted across scene reset"
                )
        _write_json(
            episode_dir / "private_segmentation_attestation.json",
            self._require_reset_safe_segmentation_channel(),
        )
        if self._oracle_prepared_state is None:
            _apply_model_phase_gripper_presentation(
                evaluator=self.evaluator, run_dir=episode_dir, stages=self.stages)
        marker_objects_provider = None
        marker_roles_provider = None
        operation_marker_objects_provider = None
        if self._toggle_task:
            marker_objects_provider = lambda active_evaluator: (
                private_toggle_task_targets(active_evaluator, self.profile)
            )
            marker_roles_provider = lambda active_evaluator: (
                private_toggle_task_roles(active_evaluator, self.profile)
            )
        elif self._active_atomic_card is not None:
            marker_objects_provider = self._all_scene_objects_by_name
            operation_marker_objects_provider = (
                self._active_atomic_operation_marker_objects
            )
        runtime = _build_marker_runtime(
            evaluator=self.evaluator,
            control=self.control,
            profile=self.profile,
            full_actions=True,
            motion_safety_starter=self._motion_safety_starter,
            # Public Action20 uses the Oracle-proven logical inventory carry:
            # physics steps keep the held closure off-scene, while RGB-only
            # action boundaries temporarily project it into the frozen
            # two-finger hold.  Never mutate collision schemas after Isaac's
            # process-wide Physics Tensor View has been created.
            suppress_held_object_environment_collisions=False,
            logical_inventory_carry=True,
            base_execution="restricted_kinematic",
            action_profile="marker_symbolic_action20",
            observation_profile=self.observation_profile,
            marker_objects_provider=marker_objects_provider,
            marker_roles_provider=marker_roles_provider,
            operation_marker_objects_provider=operation_marker_objects_provider,
            overhead_view_mode=self._overhead_view_mode,
            automatic_interaction_approach_enabled=(
                self._automatic_interaction_approach_enabled
            ),
            placement_physical_consequences_enabled=(
                self._placement_physical_consequences_enabled
            ),
        )
        episode = _episode(
            self.profile,
            self.seed,
            episode_id,
            runtime.registry,
            observation_profile=self.observation_profile,
        )
        # Publish the runtime before TaskSpec binding so the episode recovery
        # boundary can clean symbolic state if fail-closed admission rejects it.
        self._runtime = runtime
        self._episode = episode
        self._episode_id = episode_id
        self._episode_dir = episode_dir
        self._action_artifact_dir = episode_dir
        self._observation_id_prefix = episode_id
        verifier = (
            OfficialPredicateVerifier(require_single_option=False, predicate_mode=runtime.predicate_mode)
            if self._l0_task_spec is None
            else self._make_l0_verifier(episode_id, next_count)
        )
        if self._registered_task is not None:
            verifier = bind_process_verifier(self, verifier, episode_dir)
        # Publish the local runtime into the owned scene object before any
        # fixture work. If a private setup or first-observation gate fails,
        # the exception boundary can now cleanly remove symbolic collision
        # filters and permit the next begin_episode to perform a fresh,
        # attested same-scene reset.
        self._verifier = verifier
        self._initial_verifier = None
        self._last_verifier = None
        self._sequence = 0
        self._reset_count = next_count
        self._clear_marker_state()
        fixture_public = None
        fixture_initial = None
        fixture_target_name = None
        fixture_target = None
        if fixture_id is not None:
            fixture_initial = verifier.evaluate(episode, self.evaluator)
            fixture_public, fixture_target_name, fixture_target = self._apply_fixture(
                runtime=runtime,
                episode_dir=episode_dir,
                fixture_id=fixture_id,
            )
        # Reset/prepared-state restoration can leave the evaluator camera cache
        # stale even without a fixture. Refresh before freezing the first frame.
        observation = _capture_action_boundary_observation(runtime, force_render=True)
        if fixture_id is not None:
            assert fixture_target_name is not None
            assert fixture_target is not None
            self._attest_fixture_visibility(
                observation=observation,
                episode_dir=episode_dir,
                fixture_id=fixture_id,
                target_name=fixture_target_name,
                target=fixture_target,
            )
        public = _freeze_public_observation(
            runtime=runtime,
            observation=observation,
            observation_id=f"{episode_id}_initial",
            directory=episode_dir / "observations" / "initial",
            kind="physical",
            marker_targets=(),
        )
        initial = verifier.evaluate(episode, self.evaluator)
        if fixture_initial is not None:
            if (
                fixture_initial.leaf_truth != initial.leaf_truth
                or fixture_initial.official_success != initial.official_success
                or fixture_initial.official_q != initial.official_q
            ):
                raise RuntimeError("fixture setup changed official task truth")
            _write_private_json(
                episode_dir / "private" / "fixture_predicate_attestation.json",
                {
                    "fixture_id": fixture_id,
                    "official_truth_unchanged": True,
                    "before": fixture_initial.to_private_record(),
                    "after": initial.to_private_record(),
                },
            )
        _write_json(
            episode_dir / "private" / "initial_verifier.json",
            initial.to_private_record(),
        )
        self._initial_verifier = initial
        self._last_verifier = initial
        return {
            "status": "PASS",
            "episode_id": episode_id,
            "reset_index": next_count,
            "reset_attestation_pass": True,
            "scene_reconstructed": False,
            "fixture": fixture_public,
            "observation": public.to_record(),
        }

    def _apply_fixture(
        self,
        *,
        runtime: Any,
        episode_dir: Path,
        fixture_id: str,
    ) -> tuple[Mapping[str, Any], str, Any]:
        """Apply one private, task-truth-preserving operator station."""

        if self._sequence != 0:
            raise RuntimeError("fixture setup must precede every public action")
        rule = self._fixtures[fixture_id]
        carry = runtime.carry_tracker.snapshot()
        held_provider = getattr(runtime.symbolic_primitives, "_get_obj_in_hand", None)
        if (
            not callable(held_provider)
            or held_provider() is not None
            or carry.get("likely") is not False
            or carry.get("active_hand") is not None
            or carry.get("canonical_held_object_present") not in {False, None}
        ):
            raise RuntimeError("fixture setup requires a canonical empty hand")
        role = rule["target_role"]
        if role == "primary_toggle":
            if not self._toggle_task:
                raise RuntimeError("toggle fixture requires a toggle task")
            candidates = private_toggle_task_targets(
                self.evaluator,
                self.profile,
            )
        else:
            if self._toggle_task:
                raise RuntimeError(
                    "inside-task fixture is unavailable for a toggle task"
                )
            graspables, containers = private_inside_task_roles(
                self.evaluator,
                self.profile,
            )
            candidates = (
                graspables if role == "primary_graspable" else containers
            )
        ordered = sorted(candidates.items(), key=lambda item: item[0])
        target_index = int(rule["target_index"])
        if target_index >= len(ordered):
            raise RuntimeError("fixture target index is unavailable in the live task")
        target_name, target = ordered[target_index]
        _private_action_index, simulator_steps, navigation = (
            _oracle_navigate_near_with_privileged_symbolic(
                runtime=runtime,
                evaluator=self.evaluator,
                target=target,
                phase_dir=episode_dir / "private" / "fixture_setup",
                action_index=0,
                desired_distance_m=float(rule["desired_distance_m"]),
            )
        )
        pose = planar_base_pose_from_robot(self.evaluator.robot)
        tx, ty = _object_xy(target)
        distance = math.hypot(tx - pose.x, ty - pose.y)
        heading_error = math.atan2(
            math.sin(math.atan2(ty - pose.y, tx - pose.x) - pose.yaw_rad),
            math.cos(math.atan2(ty - pose.y, tx - pose.x) - pose.yaw_rad),
        )
        # This is an Action20 smoke fixture, not a station-calibration
        # benchmark.  The official planner owns traversability and the
        # subsequent private RGB/segmentation gate owns target visibility.
        # Do not reject a valid official station merely because it missed an
        # arbitrary preferred-distance band.
        if (
            navigation.get("status") != "near"
            or not math.isfinite(distance)
            or distance > float(rule["hard_workspace_limit_m"])
        ):
            raise RuntimeError("fixture official-map station failed its live gate")
        current = {
            "target_name": target_name,
            "robot_pose": [pose.x, pose.y, pose.yaw_rad],
        }
        calibration = (
            "observed"
            if fixture_id not in self._calibrated_fixtures
            else "reobserved"
        )
        self._calibrated_fixtures[fixture_id] = current
        _write_private_json(
            episode_dir / "private" / "fixture_setup.json",
            {
                "fixture_id": fixture_id,
                "fixture_rule": dict(rule),
                "target_role": role,
                "target_name_private": target_name,
                "calibration": calibration,
                "robot_pose_private": current["robot_pose"],
                "target_distance_m": distance,
                "heading_error_rad": heading_error,
                "simulator_steps": simulator_steps,
                "navigation_private": dict(navigation),
                "public_action_index_after_setup": 0,
                "official_task_truth_checked_after_setup": True,
            },
        )
        return (
            {
                "fixture_id": fixture_id,
                "setup_kind": "official_map_symbolic_station",
                "setup_pass": True,
                "calibration": calibration,
                "public_action_index": 0,
            },
            target_name,
            target,
        )

    def _attest_fixture_visibility(
        self,
        *,
        observation: Any,
        episode_dir: Path,
        fixture_id: str,
        target_name: str,
        target: Any,
    ) -> None:
        """Require an explicitly requested fixture in the first head RGB.

        Generic model episodes omit ``fixture_id`` and therefore begin from the
        reset pose without this atomic-canary gate.  This is a private setup
        gate only.  It does not install a Marker
        overlay, mutate the public observation, consume a simulator step, or
        advance the public action sequence.
        """

        cameras = observation.values.get("cameras")
        if not isinstance(cameras, Mapping) or "head" not in cameras:
            raise RuntimeError("fixture visibility requires the public head camera")
        head = cameras["head"]
        if not isinstance(head, Mapping) or set(head) != {"rgb"}:
            raise RuntimeError("fixture visibility requires RGB-only head input")
        public_head_rgb = head["rgb"]
        channel = self._reset_safe_segmentation_channel
        frame = channel.capture(
            simulator_step=observation.simulator_step,
            public_head_rgb=public_head_rgb,
        )
        snapshot = build_private_marker_snapshot(
            frame=frame,
            public_head_rgb=public_head_rgb,
            eligible_objects_by_name={target_name: target},
            official_bboxes=official_instance_bboxes(frame),
            created_action_index=0,
        )
        matching = [
            binding
            for binding in snapshot.bindings.values()
            if binding.object_name == target_name
        ]
        if len(matching) != 1 or matching[0].visible_pixels < 8:
            raise RuntimeError(
                "fixture target is not Marker-selectable in the initial head view"
            )
        _write_private_json(
            episode_dir / "private" / "fixture_visibility.json",
            {
                "fixture_id": fixture_id,
                "target_name_private": target_name,
                "marker_selectable": True,
                "visible_pixels": matching[0].visible_pixels,
                "mask_source": matching[0].mask_source,
                "simulator_step": observation.simulator_step,
                "public_action_index": 0,
                "marker_overlay_installed": False,
            },
        )

    @audited_step
    def step(
        self,
        *,
        sequence: int,
        action: str,
        target: int | None,
        navigation_substeps: int = 1,
    ) -> Mapping[str, Any]:
        if not self._session_valid:
            raise SceneProtocolError("resident scene session is invalid")
        try:
            result = self._step_impl(
                sequence=sequence,
                action=action,
                target=target,
                navigation_substeps=navigation_substeps,
            )
            if (
                self._last_verifier is not None
                and self._last_verifier.official_success
            ):
                from lab.l1p_model_entry import physical_model_may_auto_end
                diagnostics = dict(self._last_verifier.research_diagnostics or {})
                qualification = dict(
                    diagnostics.get('physical_template_review') or {}
                ).get('qualification_status')
                if not physical_model_may_auto_end(
                    official_success=True,
                    native_q_alone_is_success=self._native_q_alone_is_success,
                    qualification_status=qualification,
                ):
                    return result
                end = self.end_episode(reason="done")
                return {
                    **dict(result),
                    "episode_terminal": True,
                    "session_valid": end["session_valid"],
                }
            return result
        except SceneProtocolError:
            raise
        except Exception:
            self._session_valid = self._recover_failed_episode_runtime(
                reason="step_environment_error"
            )
            raise

    def _step_impl(
        self,
        *,
        sequence: int,
        action: str,
        target: int | None,
        navigation_substeps: int = 1,
    ) -> Mapping[str, Any]:
        require_execution_integrity(self.evaluator)
        runtime = self._require_active_runtime()
        if sequence != self._sequence:
            raise SceneProtocolError("action sequence is not the next sequence")
        if action not in STRICT_ACTION20:
            raise SceneProtocolError("scene accepts only physical Action20 actions")
        maximum_substeps = 4 if action in BASE_TRANSLATION_ACTIONS else 3
        if (
            isinstance(navigation_substeps, bool)
            or not isinstance(navigation_substeps, int)
            or not 1 <= navigation_substeps <= maximum_substeps
            or (action not in BASE_ACTIONS and navigation_substeps != 1)
        ):
            raise SceneProtocolError("navigation substeps are invalid")
        if target is not None and (
            action not in MARKER_ACTIONS
            or isinstance(target, bool)
            or not isinstance(target, int)
            or not 1 <= target <= 999
        ):
            raise SceneProtocolError("target is invalid for this action")
        if target is None:
            inflight_stage = 0 if action in MARKER_ACTIONS else None
            payload: dict[str, Any] = {"action": action}
        else:
            if (
                self._pending_marker_action != action
                or target not in self._pending_marker_targets
                or self._pending_marker_stage not in {1, 2}
            ):
                raise SceneProtocolError(
                    "target is not in the current same-action Marker set"
                )
            inflight_stage = self._pending_marker_stage
            payload = {"action": action, "args": {"marker": f"C{target}"}}
        self._clear_marker_state()
        parsed = runtime.registry.parse_payload(payload)
        assert self._episode_dir is not None
        assert self._action_artifact_dir is not None
        assert self._observation_id_prefix is not None
        action_artifact_dir = self._action_artifact_dir
        precondition_block = public_precondition_block_reason(
            runtime=runtime,
            action=action,
        )
        navigation_substep_records: list[dict[str, Any]] | None = None
        if precondition_block is None:
            assert self._verifier is not None
            assert self._episode is not None
            if navigation_substeps == 1:
                result, action_frames, simulator_steps = _execute(
                    runtime=runtime,
                    action=parsed,
                    action_index=sequence,
                    phase_dir=action_artifact_dir,
                    terminal_success_check=lambda: self._verifier.evaluate(
                        self._episode,
                        self.evaluator,
                    ),
                )
            else:
                before = _capture_action_boundary_observation(runtime)
                _save_frame(
                    runtime=runtime,
                    observation=before,
                    path=(
                        action_artifact_dir
                        / "frames"
                        / f"action_{sequence:03d}_before.png"
                    ),
                )
                simulator_steps = 0
                navigation_substep_records = []
                for substep_index in range(1, navigation_substeps + 1):
                    result, internal_frames, used_steps = _execute(
                        runtime=runtime,
                        action=parsed,
                        action_index=sequence,
                        phase_dir=action_artifact_dir,
                        terminal_success_check=lambda: self._verifier.evaluate(
                            self._episode,
                            self.evaluator,
                        ),
                        capture_before_frame=False,
                        capture_after_frame=False,
                    )
                    if internal_frames:
                        raise RuntimeError(
                            "internal navigation substep captured a public frame"
                        )
                    simulator_steps += used_steps
                    substep_public = model_visible_action20_last_result(
                        {"status": result.status, **dict(result.public_feedback)},
                        action=action,
                    )
                    if (
                        substep_public is None
                        or substep_public.get("status")
                        not in {"completed", "blocked"}
                        or not isinstance(
                            substep_public.get("body_odometry"), Mapping
                        )
                    ):
                        raise SceneIntegrationError(
                            "navigation substep public result changed"
                        )
                    navigation_substep_records.append(
                        {"substep_index": substep_index, **substep_public}
                    )
                    internal_snapshot = self._verifier.evaluate(
                        self._episode,
                        self.evaluator,
                    )
                    if (
                        substep_public["status"] == "blocked"
                        or internal_snapshot.official_success
                    ):
                        break
                if not internal_snapshot.official_success:
                    simulator_steps += runtime.environment.checkpoint_once()
                    synchronize_hold = getattr(
                        runtime.symbolic_primitives,
                        "sync_marker_kinematic_hold",
                        None,
                    )
                    if callable(synchronize_hold):
                        synchronize_hold()
                after = _capture_action_boundary_observation(runtime)
                _save_frame(
                    runtime=runtime,
                    observation=after,
                    path=(
                        action_artifact_dir
                        / "frames"
                        / f"action_{sequence:03d}_after.png"
                    ),
                )
                action_frames = (before, after)
        else:
            # Selecting a syntactically valid Action20 tool that cannot run in
            # the current public carry state is an ordinary blocked action, not
            # a scene/protocol failure.  Keep the detailed cause private while
            # returning a fresh observation and the allowlisted public status.
            clear_markers = getattr(runtime.store, "clear", None)
            if callable(clear_markers):
                clear_markers()
            result = ActionPluginResult(
                "blocked",
                {
                    "blocked_category": precondition_block,
                }
                if precondition_block in ACTION20_BLOCKED_CATEGORIES
                else {},
            )
            simulator_steps = 0
            if self._overhead_view_mode == "on_blocked":
                runtime.environment.record_public_action_result(parsed, result)
            observation = runtime.environment.observe()
            action_frames = (observation,)
            if self._overhead_view_mode != "on_blocked":
                runtime.environment.record_public_action_result(parsed, result)
            _save_frame(
                runtime=runtime,
                observation=observation,
                path=(
                    action_artifact_dir
                    / "frames"
                    / f"action_{sequence:03d}_before.png"
                ),
            )
            _save_frame(
                runtime=runtime,
                observation=observation,
                path=(
                    action_artifact_dir
                    / "frames"
                    / f"action_{sequence:03d}_after.png"
                ),
            )
            _append_jsonl(
                action_artifact_dir / "private" / "executor_actions.jsonl",
                {
                    "sequence": sequence,
                    "action": parsed.to_dict(),
                    "executor_status_private_to_rollout": "blocked",
                    "controller_status_before_settle": "blocked",
                    "controller_public_feedback_private": dict(
                        result.public_feedback
                    ),
                    "simulator_steps": 0,
                    "precondition_block_private": precondition_block,
                },
            )
        if navigation_substep_records is None:
            public_result, kind, targets, next_stage = self._public_transition(
                result=result,
                action=action,
                inflight_stage=inflight_stage,
                runtime=runtime,
            )
        else:
            final_substep = navigation_substep_records[-1]
            odometry = {
                field: sum(
                    float(step["body_odometry"][field])
                    for step in navigation_substep_records
                )
                for field in ("forward_m", "left_m", "yaw_deg")
            }
            public_result = {
                "status": final_substep["status"],
                "body_odometry": odometry,
            }
            if final_substep.get("blocked_category") is not None:
                public_result["blocked_category"] = final_substep[
                    "blocked_category"
                ]
            kind, targets, next_stage = "physical", (), None
            _append_jsonl(
                action_artifact_dir / "private" / "navigation_substeps.jsonl",
                {
                    "sequence": sequence,
                    "action": action,
                    "substeps_requested": navigation_substeps,
                    "substeps_executed": len(navigation_substep_records),
                    "substeps": navigation_substep_records,
                },
            )
        if next_stage is not None:
            self._pending_marker_action = action
            self._pending_marker_targets = frozenset(targets)
            self._pending_marker_stage = next_stage
        public_observation = _freeze_public_observation(
            runtime=runtime,
            observation=action_frames[-1],
            observation_id=(
                f"{self._observation_id_prefix}_action_{sequence:04d}_{kind}"
            ),
            directory=action_artifact_dir / "observations" / f"action_{sequence:04d}_{kind}",
            kind=kind,
            marker_targets=targets,
            marker_depth=(
                _marker_depth_for_snapshot(
                    observation=action_frames[-1],
                    snapshot=runtime.store.active_snapshot,
                )
                if kind == "marker"
                and self._marker_depth_protocol == RGBD_PROTOCOL_VERSION
                else None
            ),
        )
        assert self._verifier is not None
        assert self._episode is not None
        assert self._initial_verifier is not None
        snapshot = self._verifier.evaluate(self._episode, self.evaluator)
        self._last_verifier = snapshot
        _append_jsonl(
            action_artifact_dir / "private" / "verifier.jsonl",
            {
                "sequence": sequence,
                "snapshot": snapshot.to_private_record(),
                "new_official_goal_leaves": _new_leaf_count(
                    self._initial_verifier,
                    snapshot,
                ),
            },
        )
        self._sequence += 1
        # A successful evaluator is terminal and must never be propagated by a
        # checkpoint step or offered as a rollback target.  The atomic owner
        # performs its read-only semantic attestation and end_episode directly;
        # the ordinary step wrapper auto-ends after returning this completed
        # final action, also without another simulator step.
        rolling_checkpoint = (
            None
            if snapshot.official_success
            else self._maybe_capture_rolling_checkpoint()
        )
        return {
            "status": "PASS",
            "episode_id": self._episode_id,
            "sequence": sequence,
            "action_result": public_result,
            "observation": public_observation.to_record(),
            "simulator_steps": simulator_steps,
            "rolling_checkpoint": (
                None
                if rolling_checkpoint is None
                else {
                    "milestone_action_count": rolling_checkpoint[
                        "milestone_action_count"
                    ],
                    "captured_action_count": rolling_checkpoint[
                        "captured_action_count"
                    ],
                    "retained_action_counts": rolling_checkpoint[
                        "retained_action_counts"
                    ],
                    "persistent_checkpoint": rolling_checkpoint[
                        "persistent_checkpoint"
                    ],
                }
            ),
        }

    def end_episode(self, *, reason: str) -> Mapping[str, Any]:
        self._require_active_runtime()
        if reason not in {"done", "budget_exhausted", "client_error", "manual"}:
            raise SceneProtocolError("episode end reason is invalid")
        assert self._episode_dir is not None
        assert self._last_verifier is not None
        assert self._initial_verifier is not None
        try:
            cleanup = self._runtime.symbolic_primitives.cleanup_symbolic_collision_filter()
            if cleanup.get("active") is not False:
                invalidate_execution(self.evaluator, RuntimeError("collision-filter cleanup was not verified"))
        except Exception as exc:
            invalidate_execution(self.evaluator, exc)
            cleanup = {"active": None, "error": f"{type(exc).__name__}: {exc}"}
        integrity_failure = execution_integrity_failure(self.evaluator)
        settle_final = getattr(self._verifier, "settle_final", None)
        if callable(settle_final):
            self._last_verifier = settle_final(self.evaluator)
            _write_private_json(
                self._episode_dir / "private" / "settle_final_verifier.json",
                self._last_verifier.to_private_record(),
            )
        private_summary = {
            "episode_id": self._episode_id,
            "termination_reason": reason,
            "actions_executed": self._sequence,
            "official_q": None if integrity_failure else self._last_verifier.official_q,
            "official_success": False if integrity_failure else self._last_verifier.official_success,
            "verifier_ok": integrity_failure is None and self._last_verifier.ok,
            "integrity_failure": integrity_failure,
            "new_official_goal_leaves": None if integrity_failure else _new_leaf_count(
                self._initial_verifier,
                self._last_verifier,
            ),
            "private_oracle_executed": False,
            "l0_task_spec_id": (
                None
                if self._l0_task_spec is None
                else self._l0_task_spec["task_spec_id"]
            ),
        }
        diagnostics = {} if integrity_failure else dict(
            getattr(self._last_verifier, 'research_diagnostics', None) or {}
        )
        review = diagnostics.get('physical_template_review') or {}
        if review:
            private_summary['native_official_q'] = diagnostics.get('native_official_q', private_summary['official_q'])
            private_summary['native_proxy_success'] = diagnostics.get('native_proxy_success')
            private_summary['full_goal_success'] = bool(diagnostics.get('full_goal_success'))
            private_summary['physical_qualification'] = review.get('qualification_status')
            private_summary['process_constraints'] = diagnostics.get('process_constraints') or diagnostics.get('public_process_report')
            private_summary['physics_substeps_observed'] = review.get('physics_substeps_observed', 0)
            private_summary['physics_substeps_are_not_qualification'] = True
            private_summary['native_q_alone_is_success'] = False
            private_summary['task_success'] = (
                private_summary['official_success']
                and private_summary['verifier_ok']
                and review.get('qualification_status') == 'PASS'
            )
        private_summary['exception'] = integrity_failure
        if self._process_audit is not None:
            report = self._process_audit.report()
            private_summary['public_process_report'] = report
            private_summary['canonical_final_success'] = self._last_verifier.research_diagnostics.get('canonical_final_success')
            private_summary['official_success'] &= report['public_process_satisfied']
            private_summary['verifier_ok'] &= report['operational']
            if not review:
                private_summary['task_success'] = private_summary['official_success'] and private_summary['verifier_ok']
            _write_private_json(self._episode_dir / 'private' / 'task_process_result.json', report)
        if review:
            private_summary['task_success'] = (
                bool(private_summary['official_success'])
                and bool(private_summary['verifier_ok'])
                and review.get('qualification_status') == 'PASS'
            )
        _write_json(
            self._episode_dir / "private" / "final_score.json",
            private_summary,
        )
        _write_json(
            self._episode_dir / "private" / "collision_filter_cleanup.json",
            dict(cleanup),
        )
        session_valid = cleanup.get("active") is False and integrity_failure is None
        self._session_valid = self._session_valid and session_valid
        episode_id = self._episode_id
        actions = self._sequence
        self._runtime = None
        self._episode = None
        self._episode_id = None
        self._episode_dir = None
        retire = getattr(self._verifier, "retire", None)
        if callable(retire):
            try:
                retire()
            except Exception:
                self._session_valid = False
        cm = getattr(self, '_physical_observe_cm', None)
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                self._session_valid = False
            self._physical_observe_cm = None
        self._physical_audit = None
        self._verifier = None
        self._process_audit = None
        self._initial_verifier = None
        self._last_verifier = None
        self._sequence = 0
        self._action_artifact_dir = None
        self._observation_id_prefix = None
        self._clear_marker_state()
        self._marker_depth_protocol = None
        self._overhead_view_mode = "off"
        self._automatic_interaction_approach_enabled = False
        return {
            "status": "PASS" if session_valid else "FAIL",
            "episode_id": episode_id,
            "actions_executed": actions,
            "session_valid": self._session_valid,
            "private_score_returned": False,
        }

    def close(self) -> Mapping[str, Any]:
        if self.episode_active:
            return self.end_episode(reason="manual")
        return {
            "status": "PASS" if self._session_valid else "FAIL",
            "session_valid": self._session_valid,
        }

    def _public_transition(
        self,
        *,
        result: Any,
        action: str,
        inflight_stage: int | None,
        runtime: Any,
    ) -> tuple[dict[str, Any], str, tuple[int, ...], int | None]:
        if result.status == "selection_required":
            if action not in MARKER_ACTIONS:
                raise SceneIntegrationError(
                    "non-Marker action returned Marker candidates"
                )
            if inflight_stage == 0:
                next_stage = 1
            elif action in THREE_CALL_MARKER_ACTIONS and inflight_stage == 1:
                next_stage = 2
            else:
                raise SceneIntegrationError(
                    "final Marker selection returned another stage"
                )
            snapshot = runtime.store.active_snapshot
            if snapshot is None or snapshot.operation != action:
                raise SceneIntegrationError(
                    "Marker result has no same-action snapshot"
                )
            targets = tuple(
                sorted(
                    int(marker[1:])
                    for marker in snapshot.bindings
                    if marker.startswith("C") and marker[1:].isdigit()
                )
            )
            if not targets:
                raise SceneIntegrationError(
                    "completed Marker request has no candidates"
                )
            return {"status": "completed"}, "marker", targets, next_stage
        payload = model_visible_action20_last_result(
            {"status": result.status, **dict(result.public_feedback)},
            action=action,
        )
        if payload is None:
            raise SceneIntegrationError(
                "executor result cannot cross the public boundary"
            )
        if payload["status"] == "completed":
            if action in MARKER_ACTIONS and inflight_stage == 0:
                raise SceneIntegrationError(
                    "completed Marker request did not return candidates"
                )
            if action in THREE_CALL_MARKER_ACTIONS and inflight_stage == 1:
                raise SceneIntegrationError(
                    "three-call intermediate selection skipped candidates"
                )
        kind = (
            "transient"
            if action in MARKER_ACTIONS and payload["status"] == "blocked"
            else "physical"
        )
        return payload, kind, (), None

    def _clear_marker_state(self) -> None:
        self._pending_marker_action = None
        self._pending_marker_targets = frozenset()
        self._pending_marker_stage = None

    def _recover_failed_episode_runtime(self, *, reason: str) -> bool:
        """Abort one failed episode while preserving the resident scene.

        Recovery is admitted only when the owned symbolic collision filter
        reports itself fully inactive. The next begin_episode always performs
        the normal same-scene reset and reset attestation before another
        public action can run.
        """

        runtime = self._runtime
        if runtime is None:
            return False
        episode_dir = self._episode_dir
        try:
            cleanup = runtime.symbolic_primitives.cleanup_symbolic_collision_filter()
            recovered = cleanup.get("active") is False
        except Exception as exc:
            cleanup = {
                "active": None,
                "cleanup_error_type": type(exc).__name__,
                "cleanup_error": str(exc),
            }
            invalidate_execution(self.evaluator, exc)
            recovered = False
        integrity_failure = execution_integrity_failure(self.evaluator)
        if integrity_failure is not None:
            recovered = False
        if episode_dir is not None:
            try:
                audit = getattr(self, '_process_audit', None)
                if audit is not None:
                    _write_private_json(episode_dir / 'private' / 'task_process_result.json', audit.report())
                if integrity_failure is not None:
                    _write_private_json(
                        episode_dir / "private" / "final_score.json",
                        {"episode_id": self._episode_id, "verifier_ok": False,
                         "official_q": None, "official_success": False,
                         "integrity_failure": integrity_failure,
                         "termination_reason": "execution_integrity_failure"},
                    )
                _write_private_json(
                    episode_dir / "private" / "aborted_after_environment_error.json",
                    {
                        "reason": reason,
                        "public_actions_executed": self._sequence,
                        "symbolic_collision_filter_cleanup": dict(cleanup),
                        "same_scene_reset_required_before_retry": True,
                        "recovery_admitted": recovered,
                    },
                )
            except Exception:
                recovered = False
        retire = getattr(self._verifier, "retire", None)
        if callable(retire):
            try:
                retire()
            except Exception:
                recovered = False
        self._runtime = None
        self._episode = None
        self._episode_id = None
        self._overhead_view_mode = "off"
        self._automatic_interaction_approach_enabled = False
        self._episode_dir = None
        self._verifier = None
        self._process_audit = None
        self._initial_verifier = None
        self._last_verifier = None
        self._sequence = 0
        self._action_artifact_dir = None
        self._observation_id_prefix = None
        self._clear_marker_state()
        return recovered

    def _require_active_runtime(self) -> Any:
        if self._runtime is None:
            raise SceneProtocolError("no episode is active")
        return self._runtime

    def _require_reset_safe_segmentation_channel(self) -> Mapping[str, Any]:
        channel = get_preinstalled_private_head_segmentation_side_channel(
            evaluator=self.evaluator,
            head_sensor_name=CAMERA_SENSOR_NAMES["head"],
        )
        previous = getattr(self, "_reset_safe_segmentation_channel", None)
        if previous is not None and channel is not previous:
            raise RuntimeError(
                "private segmentation channel identity changed across reset"
            )
        self._reset_safe_segmentation_channel = channel
        report = dict(channel.last_report)
        if (
            channel.started is not True
            or report.get("installation_phase")
            != "environment_wrapper_construction"
            or report.get("post_attach_evaluator_reset_budget") is not None
        ):
            raise RuntimeError(
                "resident scene requires reset-safe preinstalled segmentation"
            )
        return {
            "schema_version": (
                "behavior-hl-bench-try-reset-safe-segmentation-attestation/v1"
            ),
            "status": "PASS",
            "same_channel_across_reset": previous is None or channel is previous,
            "model_visible_segmentation": False,
            "public_observation_modified": False,
            "channel_private": report,
        }


# Compatibility for retained CPU tests and old Task25 artifacts. New code uses
# the task-generic class above.
ResidentTask25Scene = ResidentInsideTaskScene


class SceneCommandProcessor:
    """Strict JSON command boundary, independently testable without a simulator."""

    def __init__(self, *, scene: Any, control_token: str, next_instruction_id: str | None = None) -> None:
        if not isinstance(control_token, str) or len(control_token) < 32:
            raise ValueError("control token is invalid")
        self.scene = scene
        self.control_token = control_token
        self.stop_requested = False
        self.next_instruction_id = next_instruction_id
        self.advance_requested = False

    def handle(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(request, Mapping):
            raise SceneProtocolError("request must be a JSON object")
        if request.get("protocol_version") != SCENE_PROTOCOL_VERSION:
            raise SceneProtocolError("scene protocol version changed")
        if request.get("control_token") != self.control_token:
            raise SceneProtocolError("scene control token is invalid")
        command = request.get("command")
        common = {"protocol_version", "control_token", "command"}
        if command == "status":
            _require_exact_keys(request, common)
            result = self.scene.status()
            if self.next_instruction_id is not None:
                result = {**result, 'next_instruction_id': self.next_instruction_id}
        elif command == "begin_episode":
            allowed = common | {
                "episode_id",
                "fixture_id",
                "marker_depth_protocol",
                "overhead_view_mode",
                "automatic_interaction_approach_enabled",
            }
            if "fixture_id" not in request:
                allowed.remove("fixture_id")
            if "marker_depth_protocol" not in request:
                allowed.remove("marker_depth_protocol")
            if "overhead_view_mode" not in request:
                allowed.remove("overhead_view_mode")
            if "automatic_interaction_approach_enabled" not in request:
                allowed.remove("automatic_interaction_approach_enabled")
            _require_exact_keys(request, allowed)
            begin_fields = {
                "episode_id": request["episode_id"],
                "fixture_id": request.get("fixture_id"),
            }
            if "marker_depth_protocol" in request:
                begin_fields["marker_depth_protocol"] = request[
                    "marker_depth_protocol"
                ]
            if "overhead_view_mode" in request:
                begin_fields["overhead_view_mode"] = request[
                    "overhead_view_mode"
                ]
            if "automatic_interaction_approach_enabled" in request:
                begin_fields["automatic_interaction_approach_enabled"] = request[
                    "automatic_interaction_approach_enabled"
                ]
            result = self.scene.begin_episode(**begin_fields)
        elif command == "step":
            allowed = common | {
                "sequence",
                "action",
                "target",
                "navigation_substeps",
            }
            if "target" not in request:
                allowed.remove("target")
            if "navigation_substeps" not in request:
                allowed.remove("navigation_substeps")
            _require_exact_keys(request, allowed)
            result = self.scene.step(
                sequence=request["sequence"],
                action=request["action"],
                target=request.get("target"),
                navigation_substeps=request.get("navigation_substeps", 1),
            )
        elif command == "run_atomic_card":
            _require_exact_keys(request, common | {"card_id"})
            result = self.scene.run_atomic_card(card_id=request["card_id"])
        elif command == "prepare_atomic_inventory":
            _require_exact_keys(request, common)
            result = self.scene.prepare_atomic_inventory()
        elif command == "prepare_cumulative_inside_subject":
            _require_exact_keys(
                request,
                common | {"source_name_private", "target_name_private"},
            )
            result = self.scene.prepare_cumulative_inside_subject(
                source_name_private=request["source_name_private"],
                target_name_private=request["target_name_private"],
            )
        elif command == "restore_checkpoint":
            allowed = common | {"captured_action_count"}
            if "captured_action_count" not in request:
                allowed.remove("captured_action_count")
            _require_exact_keys(request, allowed)
            result = self.scene.restore_checkpoint(
                captured_action_count=request.get("captured_action_count"),
            )
        elif command == "restore_persistent_checkpoint":
            _require_exact_keys(
                request,
                common
                | {
                    "manifest_path",
                    "manifest_sha256",
                    "episode_id",
                },
            )
            result = self.scene.restore_persistent_checkpoint(
                manifest_path=request["manifest_path"],
                manifest_sha256=request["manifest_sha256"],
                episode_id=request["episode_id"],
            )
        elif command == "end_episode":
            _require_exact_keys(request, common | {"reason"})
            result = self.scene.end_episode(reason=request["reason"])
        elif command == 'advance_instruction':
            _require_exact_keys(request, common | {'instruction_id'})
            if self.next_instruction_id is None or request['instruction_id'] != self.next_instruction_id:
                raise SceneProtocolError('Instruction is outside the frozen resident continuation')
            current = self.scene.status()
            if current.get('episode_active') is True or current.get('session_valid') is not True:
                raise SceneProtocolError('Instruction switch requires a valid inactive episode')
            result = self.scene.close()
            self.advance_requested = True
            self.stop_requested = True
        elif command == "stop_scene":
            _require_exact_keys(request, common)
            if self.scene.status().get("episode_active") is True:
                raise SceneProtocolError("end the active episode before stop_scene")
            result = self.scene.close()
            self.stop_requested = True
        else:
            raise SceneProtocolError("unknown scene command")
        return {
            "protocol_version": SCENE_PROTOCOL_VERSION,
            "command": command,
            "result": dict(result),
        }


_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)((?:\"|')?(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"control[_-]?token|password|passwd|secret|credential|authorization|cookie)"
    r"(?:\"|')?\s*[:=]\s*)"
    r"(?:bearer\s+)?(?:[^\s,;\]}]+|\"[^\"]*\"|'[^']*')"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_URL_CREDENTIAL_PATTERN = re.compile(r"(://)[^/@\s:]+:[^/@\s]+@")
_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?"
    r"-----END [^-\n]*PRIVATE KEY-----",
    re.DOTALL,
)
_KNOWN_REQUEST_OPERATIONS = frozenset(
    {
        "status",
        "begin_episode",
        "step",
        "run_atomic_card",
        "prepare_atomic_inventory",
        "prepare_cumulative_inside_subject",
        "restore_checkpoint",
        "restore_persistent_checkpoint",
        "end_episode",
        "stop_scene",
    }
)


def _redact_private_error_text(
    value: str,
    *,
    sensitive_values: tuple[str, ...],
) -> str:
    redacted = str(value)
    for secret_value in sensitive_values:
        if isinstance(secret_value, str) and len(secret_value) >= 8:
            redacted = redacted.replace(secret_value, "[REDACTED]")
    redacted = _PRIVATE_KEY_PATTERN.sub("[REDACTED_PRIVATE_KEY]", redacted)
    redacted = _SECRET_ASSIGNMENT_PATTERN.sub(r"\1[REDACTED]", redacted)
    redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", redacted)
    redacted = _URL_CREDENTIAL_PATTERN.sub(r"\1[REDACTED]@", redacted)
    return redacted


def _bounded_private_error_payload(value: Mapping[str, Any]) -> bytes:
    record = dict(value)
    payload = _canonical_json_bytes(record) + b"\n"
    if len(payload) <= MAX_PRIVATE_ERROR_ENTRY_BYTES:
        return payload
    record["entry_truncated"] = True
    original_message = str(record.get("message", ""))
    original_traceback = str(record.get("traceback", ""))
    message = original_message
    trace = original_traceback
    for _ in range(64):
        record["message"] = message
        record["traceback"] = trace
        record["message_truncated"] = message != original_message
        record["traceback_truncated"] = trace != original_traceback
        payload = _canonical_json_bytes(record) + b"\n"
        if len(payload) <= MAX_PRIVATE_ERROR_ENTRY_BYTES:
            return payload
        if len(trace) >= len(message) and trace:
            trace = trace[: max(0, len(trace) // 2)]
        elif message:
            message = message[: max(0, len(message) // 2)]
        else:
            break
    raise ValueError("private error metadata exceeds the per-entry bound")


def _append_bounded_private_jsonl(path: Path, value: Mapping[str, Any]) -> bool:
    payload = _bounded_private_error_payload(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("private error log is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if os.fstat(descriptor).st_size + len(payload) > MAX_PRIVATE_ERROR_LOG_BYTES:
            return False
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("private error log append made no progress")
            view = view[written:]
        return True
    finally:
        os.close(descriptor)


def _private_request_operation(
    request: Any,
    *,
    fallback: str,
) -> str:
    if isinstance(request, Mapping):
        command = request.get("command")
        if command in _KNOWN_REQUEST_OPERATIONS:
            return str(command)
        return "unknown"
    return fallback


def _exception_fingerprint(
    exception: BaseException,
    *,
    request_operation: str,
) -> str:
    frames = traceback.extract_tb(exception.__traceback__)
    root = frames[-1] if frames else None
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "exception_module": type(exception).__module__,
                "exception_type": type(exception).__qualname__,
                "request_operation": request_operation,
                "root_file": None if root is None else Path(root.filename).name,
                "root_function": None if root is None else root.name,
                "root_line": None if root is None else root.lineno,
            }
        )
    ).hexdigest()


class PrivateSceneErrorRecorder:
    """Append-only, size-bounded diagnostics that never enter public replies."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        sensitive_values: tuple[str, ...] = (),
    ) -> None:
        self.path = Path(runtime_root).resolve() / PRIVATE_ERROR_LOG_RELATIVE_PATH
        self.sensitive_values = tuple(sensitive_values)

    def record(
        self,
        *,
        error_id: str,
        exception: BaseException,
        request_operation: str,
    ) -> bool:
        message = _redact_private_error_text(
            str(exception),
            sensitive_values=self.sensitive_values,
        )
        formatted_traceback = _redact_private_error_text(
            "".join(
                traceback.format_exception(
                    type(exception),
                    exception,
                    exception.__traceback__,
                )
            ),
            sensitive_values=self.sensitive_values,
        )
        record = {
            "schema_version": "behavior-hl-private-scene-error/v1",
            "error_id": error_id,
            "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
            "request_operation": request_operation,
            "exception_type": type(exception).__name__,
            "exception_module": type(exception).__module__,
            "message": message,
            "traceback": formatted_traceback,
        }
        return _append_bounded_private_jsonl(self.path, record)


class UnixSceneServer:
    def __init__(
        self,
        *,
        processor: SceneCommandProcessor,
        socket_path: Path,
        ready_path: Path,
        run_dir: Path,
        runtime_root: Path | None = None,
    ) -> None:
        self.processor = processor
        self.socket_path = Path(socket_path).resolve()
        self.ready_path = Path(ready_path).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.runtime_root = Path(
            runtime_root if runtime_root is not None else self.run_dir.parent
        ).resolve()
        self.private_errors = PrivateSceneErrorRecorder(
            runtime_root=self.runtime_root,
            sensitive_values=(self.processor.control_token,),
        )

    def _error_response_for_exception(
        self,
        exception: Exception,
        *,
        request_operation: str,
    ) -> Mapping[str, Any]:
        error_id = secrets.token_hex(16)
        try:
            self.private_errors.record(
                error_id=error_id,
                exception=exception,
                request_operation=request_operation,
            )
        except Exception:
            # Error reporting must never take down the resident simulator.
            pass
        recovery: Mapping[str, Any] | None = None
        error_type = "INTERNAL_ENV_ERROR"
        message = "INTERNAL_ENV_ERROR"
        scene = getattr(self.processor, "scene", None)
        status_provider = getattr(scene, "status", None)
        if callable(status_provider):
            try:
                status = status_provider()
                if isinstance(status, Mapping):
                    recovery = {
                        "scene_resident": status.get("scene_resident") is True,
                        "session_valid": status.get("session_valid") is True,
                        "episode_active": status.get("episode_active") is True,
                        "episode_id": status.get("episode_id"),
                        "next_sequence": status.get("next_sequence"),
                        "rolling_checkpoint_action_counts": list(
                            status.get("rolling_checkpoint_action_counts", ())
                        ),
                        "error_fingerprint": _exception_fingerprint(
                            exception,
                            request_operation=request_operation,
                        ),
                    }
            except Exception:
                # A failed status read is itself evidence that automatic
                # same-process recovery must not be advertised.
                recovery = None
        if isinstance(exception, SceneProtocolError):
            error_type = "SCENE_PROTOCOL_ERROR"
            message = "SCENE_PROTOCOL_ERROR"
        elif isinstance(exception, SceneIntegrationError):
            if recovery is not None and not recovery["session_valid"]:
                error_type = "SESSION_INVALID"
                message = "SESSION_INVALID"
            else:
                error_type = "INTEGRATION_ERROR"
                message = "INTEGRATION_ERROR"
        elif recovery is not None:
            if recovery["session_valid"]:
                error_type = "RECOVERABLE_SCENE_ERROR"
                message = "RECOVERABLE_SCENE_ERROR"
            else:
                error_type = "SESSION_INVALID"
                message = "SESSION_INVALID"
        return _error_response(
            error_id=error_id,
            error_type=error_type,
            message=message,
            recovery=recovery,
        )

    def serve(self) -> Mapping[str, Any]:
        if self.socket_path.exists() or self.ready_path.exists():
            raise RuntimeError("fresh scene control paths already exist")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        commands = 0
        try:
            server.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            server.listen(1)
            _write_private_json(
                self.ready_path,
                {
                    "protocol_version": SCENE_PROTOCOL_VERSION,
                    "scene_server_version": SCENE_SERVER_VERSION,
                    "socket_path": str(self.socket_path),
                    "control_token": self.processor.control_token,
                    "run_dir": str(self.run_dir),
                    "ready_at": datetime.now(timezone.utc).isoformat(),
                    "pid": os.getpid(),
                    "model_bound": False,
                    "prompt_bound": False,
                    "provider_bound": False,
                },
            )
            while not self.processor.stop_requested:
                connection, _ = server.accept()
                with connection:
                    request: Any = None
                    request_operation = "read_request"
                    try:
                        with connection.makefile("rb") as stream:
                            line = stream.readline(MAX_REQUEST_BYTES + 1)
                        if len(line) > MAX_REQUEST_BYTES:
                            raise SceneProtocolError("request exceeds size limit")
                        request_operation = "decode_request"
                        request = json.loads(line.decode("utf-8"))
                        request_operation = _private_request_operation(
                            request,
                            fallback="decode_request",
                        )
                        response = self.processor.handle(request)
                        commands += 1
                    except Exception as exc:
                        response = self._error_response_for_exception(
                            exc,
                            request_operation=request_operation,
                        )
                    if not send_response_or_client_gone(
                        connection,
                        _canonical_json_bytes(response) + b"\n",
                    ):
                        continue
            return {
                "status": "PASS",
                "scene_server_version": SCENE_SERVER_VERSION,
                "commands_processed": commands,
                "successful_scene_loads": 1,
                "model_bound": False,
                "advance_instruction": self.processor.next_instruction_id if getattr(self.processor, 'advance_requested', False) else None,
            }
        finally:
            server.close()
            if self.ready_path.is_file():
                self.ready_path.unlink()
            if self.socket_path.exists():
                self.socket_path.unlink()


def _freeze_public_observation(
    *,
    runtime: Any,
    observation: Any,
    observation_id: str,
    directory: Path,
    kind: str,
    marker_targets: tuple[int, ...],
    marker_depth: tuple[Mapping[str, Any], ...] | None = None,
) -> PublicSceneObservation:
    artifacts = runtime.environment.png_artifacts(observation)
    camera_roles = (
        frozenset(artifacts) if isinstance(artifacts, Mapping) else frozenset()
    )
    if camera_roles not in {
        frozenset({"head", "left_wrist", "right_wrist"}),
        frozenset({"head", "left_wrist", "right_wrist", "overhead"}),
    }:
        raise RuntimeError("runtime returned an invalid public camera set")
    directory.mkdir(parents=True, exist_ok=False)
    preferred = ("head", "left_wrist", "right_wrist", "overhead")
    ordered = [camera for camera in preferred if camera in artifacts]
    if set(ordered) != set(artifacts):
        raise RuntimeError("runtime public camera identity changed")
    images: list[dict[str, str]] = []
    for camera in ordered:
        artifact = artifacts[camera]
        value = bytes(artifact.png_bytes)
        digest = hashlib.sha256(value).hexdigest()
        if digest != artifact.png_sha256:
            raise RuntimeError("runtime public PNG identity changed")
        path = directory / f"{camera}.png"
        path.write_bytes(value)
        images.append({"camera": camera, "path": str(path), "sha256": digest})
    return PublicSceneObservation(
        observation_id=observation_id,
        kind=kind,
        images=tuple(images),
        marker_targets=marker_targets,
        marker_depth=(
            None
            if marker_depth is None
            else validate_public_marker_depth([dict(item) for item in marker_depth])
        ),
    )


def _marker_depth_for_snapshot(*, observation: Any, snapshot: Any) -> tuple[dict[str, Any], ...]:
    """Use only the current public Marker geometry and aligned onboard depth."""

    if snapshot is None:
        raise RuntimeError("Marker depth requires the current Marker snapshot")
    anchors: dict[str, tuple[int, int]] = {}
    for marker, binding in snapshot.bindings.items():
        anchor = binding.anchor_uv
        if anchor is None:
            x0, y0, x1, y1 = binding.official_bbox_xyxy
            anchor = ((int(x0) + int(x1)) // 2, (int(y0) + int(y1)) // 2)
        anchors[str(marker)] = (int(anchor[0]), int(anchor[1]))
    cameras = observation.values.get("cameras")
    head = cameras.get("head") if isinstance(cameras, Mapping) else None
    if not isinstance(head, Mapping):
        raise RuntimeError("RGB-D observation has no head camera")
    return marker_depth_records(
        head_camera=head,
        anchors=anchors,
    )


def _require_exact_keys(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise SceneProtocolError("scene command fields changed")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _error_response(
    *,
    error_id: str,
    error_type: str = "INTERNAL_ENV_ERROR",
    message: str = "INTERNAL_ENV_ERROR",
    recovery: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if error_type not in {
        "INTERNAL_ENV_ERROR",
        "SCENE_PROTOCOL_ERROR",
        "RECOVERABLE_SCENE_ERROR",
        "INTEGRATION_ERROR",
        "SESSION_INVALID",
    }:
        raise ValueError("public scene error type changed")
    response: dict[str, Any] = {
        "protocol_version": SCENE_PROTOCOL_VERSION,
        "status": "ERROR",
        "error_type": error_type,
        "message": message,
        "error_id": error_id,
    }
    if recovery is not None:
        response["recovery"] = dict(recovery)
    return response


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _write_private_json(path: Path, value: Any) -> None:
    _write_json(path, value)
    os.chmod(path, 0o600)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(_canonical_json_bytes(value) + b"\n")


def make_scene_server_callback(
    *,
    runtime_root: Path,
    socket_path: Path,
    ready_path: Path,
    seed: int,
    profile: Any,
    fixture_rules: Mapping[str, Any] | None,
    observation_profile: str = THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    rgbd_protocol_version: str | None = None,
    scene_load_mode: str = "partial",
    rolling_checkpoint_action_interval: int = DEFAULT_ACTION_INTERVAL,
    rolling_checkpoint_maximum_retained: int = DEFAULT_MAXIMUM_RETAINED,
    l0_task_spec: Mapping[str, Any] | None = None,
    registered_task: Any | None = None,
    public_task_id: int | None = None,
    public_instruction_id: str | None = None,
    preserve_prepared_initial_state: bool = False,
    oracle_prepared_state: Mapping[str, Any] | None = None,
    initial_state_validator: Any | None = None,
    next_instruction_id: str | None = None,
    placement_physical_consequences_enabled: bool = False,
    native_q_alone_is_success: bool = True,
) -> Any:
    def callback(
        *,
        evaluator: Any,
        control: Any,
        bindings: Any,
        run_dir: Path,
        stages: Path,
    ) -> Mapping[str, Any]:
        scene = ResidentInsideTaskScene(
            evaluator=evaluator,
            control=control,
            bindings=bindings,
            run_dir=run_dir,
            stages=stages,
            seed=seed,
            profile=profile,
            fixture_rules=fixture_rules,
            observation_profile=observation_profile,
            rgbd_protocol_version=rgbd_protocol_version,
            scene_load_mode=scene_load_mode,
            rolling_checkpoint_action_interval=(
                rolling_checkpoint_action_interval
            ),
            rolling_checkpoint_maximum_retained=(
                rolling_checkpoint_maximum_retained
            ),
            l0_task_spec=l0_task_spec,
            registered_task=registered_task,
            public_task_id=public_task_id,
            public_instruction_id=public_instruction_id,
            preserve_prepared_initial_state=preserve_prepared_initial_state,
            oracle_prepared_state=oracle_prepared_state,
            initial_state_validator=initial_state_validator,
            placement_physical_consequences_enabled=placement_physical_consequences_enabled,
            native_q_alone_is_success=native_q_alone_is_success,
            runtime_registry=Path(runtime_root).resolve().parent,
        )
        token = secrets.token_hex(32)
        server = UnixSceneServer(
            processor=SceneCommandProcessor(scene=scene, control_token=token, next_instruction_id=next_instruction_id),
            socket_path=socket_path,
            ready_path=ready_path,
            run_dir=run_dir,
            runtime_root=runtime_root,
        )
        return server.serve()

    return callback


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--server-port", type=int, default=9500)
    parser.add_argument("--seed", type=int, default=19506)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--appdata-slot", type=Path, default=DEFAULT_APPDATA_SLOT)
    return parser.parse_args(argv)


def _validate_frozen_scene_manifest(
    *,
    lane_config: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    manifest = json.loads(SCENE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("frozen scene manifest must be an object")
    if lane_config is None:
        if (
            manifest.get("schema_version")
            != "behavior-hl-bench-try-scene-launch/v2"
            or manifest.get("task_id") != 25
            or manifest.get("seed") != 19506
            or manifest.get("scene_load_budget") != 1
            or manifest.get("model_bound") is not False
            or manifest.get("prompt_bound") is not False
            or manifest.get("provider_bound") is not False
            or manifest.get("reset_safe_private_segmentation") is not True
            or manifest.get("environment_wrapper_target")
            != RESET_SAFE_THREE_RGB_WRAPPER_TARGET
            or manifest.get("runtime_root") != str(DEFAULT_RUNTIME_ROOT)
        ):
            raise ValueError("legacy frozen scene manifest contract changed")
    else:
        lanes = manifest.get("lanes")
        lane_key = lane_config.get("lane_key")
        selected = None
        if isinstance(lanes, list):
            selected = next(
                (
                    item
                    for item in lanes
                    if isinstance(item, Mapping)
                    and item.get("lane_key") == lane_key
                ),
                None,
            )
        expected_fixture_ids = sorted(
            item["fixture_id"]
            for item in lane_config["fixture_rules"]["fixtures"]
        )
        expected_atomic_card_ids = sorted(
            item["card_id"]
            for item in lane_config["fixture_rules"].get(
                "atomic_cards", {}
            ).get("cards", [])
        )
        config_path = Path(str(lane_config.get("config_path", ""))).resolve()
        expected_load_mode = str(lane_config.get("scene_load_mode", "partial"))
        expected_rgbd_protocol = lane_config.get("rgbd_protocol_version")
        expected_observation_profile = (
            "r1pro_three_rgbd_head_eval"
            if expected_rgbd_protocol == "rgbd_v1"
            else "r1pro_three_rgb_head_eval"
        )
        expected_wrapper_target = (
            "lab.reset_safe_three_rgb_wrapper."
            "ResetSafeCameraOnlyThreeRGBDHeadEvalWrapper"
            if expected_rgbd_protocol == "rgbd_v1"
            else "lab.reset_safe_three_rgb_wrapper.ResetSafeCameraOnlyThreeRGBHeadEvalWrapper"
        )
        expected_resident_slot = int(lane_config.get("resident_slot", 0))
        expected_checkpoint_interval = int(
            lane_config.get(
                "rolling_checkpoint_action_interval",
                DEFAULT_ACTION_INTERVAL,
            )
        )
        expected_checkpoint_maximum = int(
            lane_config.get(
                "rolling_checkpoint_maximum_retained",
                DEFAULT_MAXIMUM_RETAINED,
            )
        )
        if (
            manifest.get("schema_version")
            not in {
                "behavior-hl-bench-try-scene-launch/v3",
                "behavior-hl-bench-try-scene-launch/v4-source-identity",
                "behavior-hl-bench-try-scene-launch/v5-official-source-identity",
            }
            or manifest.get("scene_load_budget_per_lane") != 1
            or manifest.get("total_scene_load_budget") != 1
            or not isinstance(lanes, list)
            or len(lanes) != 1
            or manifest.get("model_bound") is not False
            or manifest.get("prompt_bound") is not False
            or manifest.get("provider_bound") is not False
            or manifest.get("private_oracle_enabled") is not False
            or manifest.get("reset_safe_private_segmentation") is not True
            or manifest.get("environment_wrapper_target")
            != expected_wrapper_target
            or manifest.get("observation_profile")
            != expected_observation_profile
            or manifest.get("rgbd_protocol_version")
            != expected_rgbd_protocol
            or manifest.get("config_path") != str(config_path)
            or manifest.get("config_sha256")
            != hashlib.sha256(config_path.read_bytes()).hexdigest()
            or not isinstance(selected, Mapping)
            or selected.get("task_id") != lane_config.get("task_id")
            or selected.get("task_name") != lane_config.get("task_name")
            or selected.get("scene_model") != lane_config.get("scene_model")
            or selected.get("split") != lane_config.get("split")
            or str(selected.get("instance_id")) != lane_config.get("instance_id")
            or selected.get("seed") != lane_config.get("seed")
            or selected.get("physical_gpu") != lane_config.get("physical_gpu")
            or selected.get("server_port") != lane_config.get("server_port")
            or selected.get("runtime_root") != lane_config.get("runtime_root")
            or selected.get("rgbd_protocol_version")
            != expected_rgbd_protocol
            or manifest.get("resident_slot", 0) != expected_resident_slot
            or selected.get("resident_slot", 0) != expected_resident_slot
            or manifest.get(
                "rolling_checkpoint_action_interval",
                DEFAULT_ACTION_INTERVAL,
            )
            != expected_checkpoint_interval
            or manifest.get(
                "rolling_checkpoint_maximum_retained",
                DEFAULT_MAXIMUM_RETAINED,
            )
            != expected_checkpoint_maximum
            or selected.get(
                "rolling_checkpoint_action_interval",
                DEFAULT_ACTION_INTERVAL,
            )
            != expected_checkpoint_interval
            or selected.get(
                "rolling_checkpoint_maximum_retained",
                DEFAULT_MAXIMUM_RETAINED,
            )
            != expected_checkpoint_maximum
            or manifest.get("scene_load_mode", "partial") != expected_load_mode
            or selected.get("scene_load_mode", "partial") != expected_load_mode
            or selected.get("default_fixture_id")
            != lane_config["fixture_rules"]["default_fixture_id"]
            or selected.get("fixture_ids") != expected_fixture_ids
            or selected.get("atomic_card_ids", [])
            != expected_atomic_card_ids
            or selected.get("l0_task_spec_id")
            != (
                lane_config["l0_task_spec"]["task_spec_id"]
                if lane_config.get("l0_task_spec") is not None
                else None
            )
            or selected.get("l0_task_spec_sha256")
            != lane_config.get("l0_task_spec_sha256")
        ):
            raise ValueError("resident action-lab manifest contract changed")
    if manifest.get("schema_version") == "behavior-hl-bench-try-scene-launch/v3":
        source_hashes = manifest.get("source_sha256")
        if not isinstance(source_hashes, Mapping) or not source_hashes:
            raise ValueError("frozen scene source hashes are missing")
        for raw_path, expected in source_hashes.items():
            path = Path(raw_path)
            if (
                not isinstance(raw_path, str)
                or not isinstance(expected, str)
                or len(expected) != 64
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError(f"frozen scene source drifted: {raw_path}")
    else:
        identities = manifest.get("source_identity")
        expected_identities = {"bench", "simulator"}
        if manifest.get("schema_version") == (
            "behavior-hl-bench-try-scene-launch/v5-official-source-identity"
        ):
            expected_identities.add("official")
        if (
            not isinstance(identities, Mapping)
            or set(identities) != expected_identities
        ):
            raise ValueError("resident source identities are missing")
        bench_identity = validate_source_identity(identities["bench"])
        simulator_identity = validate_source_identity(identities["simulator"])
        if (
            Path(bench_identity["root"]) != BENCH_TRY_ROOT.resolve()
            or lane_config is None
            or Path(simulator_identity["root"])
            != Path(lane_config["project_root"]).resolve()
        ):
            raise ValueError("resident source root does not match its lane")
        if "official" in identities:
            official_identity = validate_source_identity(identities["official"])
            expected_official_root = (
                Path(lane_config["bundle_root"]) / "BEHAVIOR-1K-2026"
            ).resolve()
            if Path(official_identity["root"]) != expected_official_root:
                raise ValueError("official source root does not match its lane")
    return manifest


def main(
    argv: list[str] | None = None,
    *,
    lane_config: Mapping[str, Any] | None = None,
) -> int:
    args = parse_args(argv)
    _validate_frozen_scene_manifest(lane_config=lane_config)
    runtime_root = args.runtime_root.resolve()
    expected_runtime_root = (
        DEFAULT_RUNTIME_ROOT.resolve()
        if lane_config is None
        else Path(lane_config["runtime_root"]).resolve()
    )
    if runtime_root != expected_runtime_root:
        raise ValueError("scene runtime must use the frozen bench_try runtime root")
    if runtime_root.exists():
        raise ValueError("fresh scene runtime root already exists")
    artifact_root = runtime_root / "simulator"
    # Resolve and validate this before the expensive scene load.
    socket_path = control_socket_path(runtime_root)
    ready_path = runtime_root / "ready.json"
    task_name = (
        "clearing_food_from_table_into_fridge"
        if lane_config is None
        else str(lane_config["task_name"])
    )
    profile = _resolve_task_profile(task_name)
    if lane_config is not None and (
        profile.task_id != lane_config["task_id"]
        or profile.task_name != lane_config["task_name"]
        or profile.scene_model != lane_config["scene_model"]
        or profile.split != lane_config["split"]
        or str(profile.instance_id) != lane_config["instance_id"]
        or args.seed != lane_config["seed"]
        or args.physical_gpu != lane_config["physical_gpu"]
        or args.server_port != lane_config["server_port"]
    ):
        raise ValueError("live inside-task profile does not match the frozen lane")
    worker = B1SingleLoadWorkerSpec(
        worker_id=args.worker_id,
        physical_gpu=args.physical_gpu,
        server_port=args.server_port,
        seed=args.seed,
        artifact_root=artifact_root,
        task_name=profile.task_name,
        split=profile.split,
        instance_id=profile.instance_id,
    )
    callback = make_scene_server_callback(
        runtime_root=runtime_root,
        socket_path=socket_path,
        ready_path=ready_path,
        seed=args.seed,
        profile=profile,
        fixture_rules=(
            None if lane_config is None else lane_config["fixture_rules"]
        ),
        observation_profile=(
            THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE
            if lane_config is not None
            and lane_config.get("rgbd_protocol_version") == RGBD_PROTOCOL_VERSION
            else THREE_RGB_HEAD_EVAL_CAMERA_PROFILE
        ),
        rgbd_protocol_version=(
            None if lane_config is None else lane_config.get("rgbd_protocol_version")
        ),
        scene_load_mode=(
            "partial"
            if lane_config is None
            else str(lane_config.get("scene_load_mode", "partial"))
        ),
        rolling_checkpoint_action_interval=(
            DEFAULT_ACTION_INTERVAL
            if lane_config is None
            else int(lane_config["rolling_checkpoint_action_interval"])
        ),
        rolling_checkpoint_maximum_retained=(
            DEFAULT_MAXIMUM_RETAINED
            if lane_config is None
            else int(lane_config["rolling_checkpoint_maximum_retained"])
        ),
        l0_task_spec=(
            None if lane_config is None else lane_config.get("l0_task_spec")
        ),
    )
    scene_load_mode = (
        "partial"
        if lane_config is None
        else str(lane_config.get("scene_load_mode", "partial"))
    )
    observation_profile = (
        THREE_RGBD_HEAD_EVAL_CAMERA_PROFILE
        if lane_config is not None
        and lane_config.get("rgbd_protocol_version") == RGBD_PROTOCOL_VERSION
        else THREE_RGB_HEAD_EVAL_CAMERA_PROFILE
    )
    result = run_b1_single_load_with_reset_safe_three_rgb(
        bindings=build_production_bindings(),
        worker=worker,
        expected_task_id=profile.task_id,
        camera_profile=observation_profile,
        gripper_visual_profile=(
            GRIPPER_VISUAL_PROFILE_INWARD_12CM_OFFICIAL_CAMERA
        ),
        defer_inward_arm_presentation=True,
        partial_scene_load=scene_load_mode == "partial",
        filter_embedded_scene_agents=scene_load_mode == "full",
        use_wrist_evidence_reset_pose=scene_load_mode == "full",
        strategy_sweep_callback=callback,
    )
    print(
        json.dumps(
            {
                "event": "SCENE_SERVER_STOPPED",
                "run_dir": str(result.run_dir),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
