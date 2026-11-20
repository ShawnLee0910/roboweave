"""Zero-scene, fail-closed readiness gate for the B1 resident runtime."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
    G3BStageRecord,
    SafeCarryCalibrationLimits,
    SafeCarryCalibrationRecord,
    SafeCarryCandidate,
)
from behavior_hl.embodied.policy.registry import ActionRegistry
from behavior_hl.embodied.runtime.dispatch import ActionPluginDispatcher
from behavior_hl.embodied.runtime.public_carry_state import (
    PUBLIC_CARRY_STATE_VERSION,
    PublicCarryStateTracker,
)


B1_LIVE_READINESS_VERSION = "behavior2026.b1-live-readiness/v2"
REQUIRED_RESIDENT_FACTORY_VERSION = (
    "behavior2026.r1pro-resident-services/v2"
)
B1_REQUIRED_ACTIONS = (
    "move_backward",
    "move_forward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "look_up",
    "look_center",
    "look_down",
    "grasp_at",
    "place_inside_at",
    "release",
    "stop",
)
B1_MARKER_SYMBOLIC_V34_REQUIRED_ACTIONS = (
    "move_backward",
    "move_forward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "move_up",
    "move_down",
    "mark_visible_objects",
    "pick",
    "place_inside",
    "put_back",
    "open",
    "close",
    "stop",
)
B1_MARKER_SYMBOLIC_V34_SPATIAL_REQUIRED_ACTIONS = (
    *B1_MARKER_SYMBOLIC_V34_REQUIRED_ACTIONS[:-1],
    "place_on_top",
    "place_next_to",
    "place_under",
    "place_touching",
    "toggle_on",
    "toggle_off",
    "stop",
)
B1_MARKER_SYMBOLIC_ACTION23_REQUIRED_ACTIONS = (
    "move_backward",
    "move_forward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "mark_visible_objects",
    "pick",
    "release",
    "place_inside",
    "place_on_top",
    "place_next_to",
    "place_under",
    "place_touching",
    "open",
    "close",
    "toggle_on",
    "toggle_off",
    "wait",
)
B1_MARKER_SYMBOLIC_ACTION20_REQUIRED_ACTIONS = tuple(
    "place_on_floor" if action == "release" else action
    for action in B1_MARKER_SYMBOLIC_ACTION23_REQUIRED_ACTIONS
    if action != "mark_visible_objects"
)
B1_MARKER_SYMBOLIC_LEGACY_ACTION23_REQUIRED_ACTIONS = (
    *B1_MARKER_SYMBOLIC_ACTION23_REQUIRED_ACTIONS[:8],
    "move_up",
    "move_down",
    *B1_MARKER_SYMBOLIC_ACTION23_REQUIRED_ACTIONS[8:],
)
B1_G3B_REQUIRED_STAGES = (
    "neutral",
    "downward",
    "neutral_return",
)

_SOURCE_HASH = re.compile(r"^[0-9a-fA-F]{64}$")
_INVALID_PROVENANCE_MARKER = re.compile(
    r"(?<![a-z0-9])"
    r"(?:placeholder|fake|dummy|mock|stub|unvalidated|todo|tbd|"
    r"not[_ -]?run)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class B1LiveReadinessReport:
    """Machine-readable evidence for deciding whether a live run may start."""

    ready: bool
    gates: Mapping[str, bool]
    failed_gates: tuple[str, ...]
    required_actions: tuple[str, ...]
    action_to_controller: Mapping[str, str]
    dispatcher_controller_keys: tuple[str, ...]
    factory_version: str | None
    tracker_snapshot: Mapping[str, Any]
    calibration_summary: Mapping[str, Any]
    schema_version: str = B1_LIVE_READINESS_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "gates", MappingProxyType(dict(self.gates)))
        object.__setattr__(
            self,
            "failed_gates",
            tuple(self.failed_gates),
        )
        object.__setattr__(
            self,
            "required_actions",
            tuple(self.required_actions),
        )
        object.__setattr__(
            self,
            "action_to_controller",
            MappingProxyType(dict(self.action_to_controller)),
        )
        object.__setattr__(
            self,
            "dispatcher_controller_keys",
            tuple(self.dispatcher_controller_keys),
        )
        object.__setattr__(
            self,
            "tracker_snapshot",
            MappingProxyType(dict(self.tracker_snapshot)),
        )
        object.__setattr__(
            self,
            "calibration_summary",
            MappingProxyType(dict(self.calibration_summary)),
        )

    def require_ready(self) -> None:
        if not self.ready:
            raise RuntimeError(
                "B1 live readiness failed closed: "
                f"{list(self.failed_gates)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ready": self.ready,
            "gates": dict(self.gates),
            "failed_gates": list(self.failed_gates),
            "required_actions": list(self.required_actions),
            "action_to_controller": dict(self.action_to_controller),
            "dispatcher_controller_keys": list(
                self.dispatcher_controller_keys
            ),
            "factory_version": self.factory_version,
            "tracker_snapshot": dict(self.tracker_snapshot),
            "calibration_summary": dict(self.calibration_summary),
        }


def audit_b1_live_readiness(
    *,
    registry: Any,
    dispatcher: Any,
    resident_service_factory: Any,
    public_carry_tracker: Any,
    safe_carry_calibration: SafeCarryCalibrationRecord,
    required_actions: Sequence[str],
) -> B1LiveReadinessReport:
    """Inspect B1 contracts without importing or touching a simulator."""

    gates: dict[str, bool] = {}
    actions = _validated_action_sequence(required_actions)
    admitted_profiles = (
        B1_REQUIRED_ACTIONS,
        B1_MARKER_SYMBOLIC_V34_REQUIRED_ACTIONS,
        B1_MARKER_SYMBOLIC_V34_SPATIAL_REQUIRED_ACTIONS,
        B1_MARKER_SYMBOLIC_ACTION23_REQUIRED_ACTIONS,
        B1_MARKER_SYMBOLIC_LEGACY_ACTION23_REQUIRED_ACTIONS,
        B1_MARKER_SYMBOLIC_ACTION20_REQUIRED_ACTIONS,
        (*B1_MARKER_SYMBOLIC_ACTION20_REQUIRED_ACTIONS, "wipe"),
    )
    gates["required_action_profile"] = actions is not None and any(
        len(actions) == len(profile) and set(actions) == set(profile)
        for profile in admitted_profiles
    )
    report_actions = actions or ()

    action_to_controller: dict[str, str] = {}
    registry_valid = (
        isinstance(registry, ActionRegistry)
        and gates["required_action_profile"]
    )
    if registry_valid:
        try:
            registry_valid = set(registry.names) == set(report_actions)
            if registry_valid:
                action_to_controller = {
                    action: registry.spec(action).controller_key
                    for action in report_actions
                }
                registry_valid = all(
                    isinstance(key, str) and bool(key)
                    for key in action_to_controller.values()
                )
        except Exception:
            registry_valid = False
            action_to_controller = {}
    gates["registry_action_profile"] = registry_valid

    non_stop_keys: tuple[str, ...] = ()
    twelve_unique = False
    intercepted_count = 1 if "stop" in report_actions else 0
    expected_controller_count = max(
        0, len(report_actions) - intercepted_count
    )
    if registry_valid:
        non_stop_keys = tuple(
            action_to_controller[action]
            for action in report_actions
            if action != "stop"
        )
        twelve_unique = (
            len(non_stop_keys) == expected_controller_count
            and len(set(non_stop_keys)) == expected_controller_count
        )
    gates["twelve_unique_controller_keys"] = twelve_unique
    gates["unique_controller_keys"] = twelve_unique
    # Keep the legacy key stable while exposing an accurately named v34
    # attestation for the fourteen non-stop controllers.
    if expected_controller_count == 14:
        gates["fourteen_unique_controller_keys"] = twelve_unique

    dispatcher_keys: tuple[str, ...] = ()
    dispatcher_valid = isinstance(dispatcher, ActionPluginDispatcher)
    if dispatcher_valid:
        try:
            dispatcher_keys = tuple(dispatcher.controller_keys)
            dispatcher_valid = (
                twelve_unique
                and len(dispatcher_keys) == expected_controller_count
                and set(dispatcher_keys) == set(non_stop_keys)
                and (
                    "stop" not in action_to_controller
                    or action_to_controller["stop"] not in set(dispatcher_keys)
                )
            )
        except Exception:
            dispatcher_valid = False
            dispatcher_keys = ()
    gates["dispatcher_controller_coverage"] = dispatcher_valid
    gates["controller_coverage_12"] = (
        registry_valid and twelve_unique and dispatcher_valid
    )
    if expected_controller_count == 14:
        gates["controller_coverage_14"] = (
            registry_valid and twelve_unique and dispatcher_valid
        )

    factory_version = getattr(
        resident_service_factory,
        "api_contract_version",
        None,
    )
    gates["factory_v2"] = (
        factory_version == REQUIRED_RESIDENT_FACTORY_VERSION
    )
    gates["factory_build"] = callable(
        getattr(resident_service_factory, "build", None)
    )
    gates["factory_public_carry_snapshot"] = callable(
        getattr(
            resident_service_factory,
            "public_carry_snapshot",
            None,
        )
    )
    gates["factory_public_result_update"] = callable(
        getattr(
            resident_service_factory,
            "record_public_action_result",
            None,
        )
    )

    tracker_snapshot: dict[str, Any] = {}
    tracker_valid = isinstance(
        public_carry_tracker,
        PublicCarryStateTracker,
    )
    if tracker_valid:
        try:
            tracker_snapshot = dict(public_carry_tracker.snapshot())
            tracker_valid = (
                tracker_snapshot.get("schema_version")
                == PUBLIC_CARRY_STATE_VERSION
                and tracker_snapshot.get("confidence") == "not_likely"
                and tracker_snapshot.get("likely") is False
                and tracker_snapshot.get("active_hand") is None
                and tracker_snapshot.get("transition_count") == 0
                and tracker_snapshot.get("last_transition") == "initial"
                and public_carry_tracker.likely() is False
                and public_carry_tracker.active_hand() is None
            )
        except Exception:
            tracker_valid = False
            tracker_snapshot = {}
    gates["tracker_initial_not_likely"] = tracker_valid

    calibration_gates, calibration_summary = _audit_calibration(
        safe_carry_calibration
    )
    gates.update(calibration_gates)

    failed = tuple(name for name, passed in gates.items() if not passed)
    return B1LiveReadinessReport(
        ready=not failed,
        gates=gates,
        failed_gates=failed,
        required_actions=report_actions,
        action_to_controller=action_to_controller,
        dispatcher_controller_keys=dispatcher_keys,
        factory_version=(
            factory_version if isinstance(factory_version, str) else None
        ),
        tracker_snapshot=tracker_snapshot,
        calibration_summary=calibration_summary,
    )


def _validated_action_sequence(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    try:
        result = tuple(value)
    except Exception:
        return None
    if (
        not result
        or any(not isinstance(item, str) or not item for item in result)
        or len(set(result)) != len(result)
    ):
        return None
    return result


def _audit_calibration(
    record: Any,
) -> tuple[dict[str, bool], dict[str, Any]]:
    record_type = isinstance(record, SafeCarryCalibrationRecord)
    gates = {"calibration_record_type": record_type}
    if not record_type:
        gates.update(
            {
                "calibration_schema": False,
                "calibration_passed": False,
                "calibration_failure_reason_clear": False,
                "calibration_ik_only_pass": False,
                "calibration_held_continuous": False,
                "calibration_grasp_proxy_continuous": False,
                "calibration_candidate": False,
                "calibration_provenance_clean": False,
                "calibration_body_relative_eef": False,
                "calibration_retract_sample_count": False,
                "calibration_stage_sequence_exact": False,
                "calibration_stage_sample_counts": False,
                "calibration_sha256": False,
            }
        )
        return gates, {}

    gates["calibration_schema"] = (
        record.schema_version
        == "behavior2026.safe-carry-calibration/v1"
    )
    gates["calibration_passed"] = record.passed is True
    gates["calibration_failure_reason_clear"] = (
        record.failure_reason is None
    )
    gates["calibration_ik_only_pass"] = record.ik_only_pass is True
    gates["calibration_held_continuous"] = (
        record.held_continuous is True
    )
    gates["calibration_grasp_proxy_continuous"] = (
        record.grasp_proxy_continuous is True
    )
    candidate = record.candidate
    candidate_valid = isinstance(candidate, SafeCarryCandidate)
    arm_target = None
    candidate_position = None
    candidate_quaternion = None
    if candidate_valid:
        arm_target = _finite_vector(candidate.arm_target, 7)
        candidate_position = _finite_vector(
            candidate.eef_world_position,
            3,
        )
        candidate_quaternion = _finite_nonzero_quaternion(
            candidate.eef_world_quaternion_xyzw
        )
        candidate_valid = (
            candidate.hand in {"left", "right"}
            and arm_target is not None
            and candidate_position is not None
            and candidate_quaternion is not None
        )
    gates["calibration_candidate"] = candidate_valid
    gates["calibration_provenance_clean"] = (
        candidate_valid
        and isinstance(candidate.source, str)
        and bool(candidate.source.strip())
        and _INVALID_PROVENANCE_MARKER.search(candidate.source) is None
    )

    body_position = _finite_vector(record.body_eef_position, 3)
    body_quaternion = _finite_nonzero_quaternion(
        record.body_eef_quaternion_xyzw
    )
    gates["calibration_body_relative_eef"] = (
        body_position is not None and body_quaternion is not None
    )

    retract_steps = record.retract_steps
    limits_valid = isinstance(
        record.limits,
        SafeCarryCalibrationLimits,
    )
    gates["calibration_retract_sample_count"] = (
        limits_valid
        and isinstance(retract_steps, int)
        and not isinstance(retract_steps, bool)
        and retract_steps > 0
        and retract_steps <= record.limits.max_retract_steps
    )
    stage_records_valid = (
        isinstance(record.stage_records, tuple)
        and all(
            isinstance(item, G3BStageRecord)
            for item in record.stage_records
        )
    )
    actual_stages = (
        tuple(item.stage for item in record.stage_records)
        if stage_records_valid
        else ()
    )
    gates["calibration_stage_sequence_exact"] = (
        limits_valid
        and stage_records_valid
        and tuple(record.limits.g3b_stages)
        == B1_G3B_REQUIRED_STAGES
        and actual_stages == B1_G3B_REQUIRED_STAGES
    )
    gates["calibration_stage_sample_counts"] = (
        gates["calibration_stage_sequence_exact"]
        and all(
            item.passed is True
            and isinstance(item.samples, int)
            and not isinstance(item.samples, bool)
            and item.samples == record.limits.samples_per_stage
            for item in record.stage_records
        )
    )

    record_sha256 = None
    try:
        claimed_hash = record.sha256
        if (
            isinstance(claimed_hash, str)
            and _SOURCE_HASH.fullmatch(claimed_hash) is not None
            and set(claimed_hash.casefold()) != {"0"}
        ):
            record_sha256 = claimed_hash.casefold()
    except Exception:
        record_sha256 = None
    gates["calibration_sha256"] = record_sha256 is not None

    stage_summary = (
        tuple(
            {
                "stage": item.stage,
                "samples": item.samples,
                "passed": item.passed,
            }
            for item in record.stage_records
        )
        if stage_records_valid
        else ()
    )
    return gates, {
        "record_schema_version": record.schema_version,
        "record_sha256": record_sha256,
        "candidate": {
            "hand": candidate.hand if candidate_valid else None,
            "source": candidate.source if candidate_valid else None,
            "arm_target": arm_target,
            "eef_world_position": candidate_position,
            "eef_world_quaternion_xyzw": candidate_quaternion,
        },
        "body_relative_eef": {
            "position": body_position,
            "quaternion_xyzw": body_quaternion,
        },
        "ik_only_pass": record.ik_only_pass is True,
        "held_continuous": record.held_continuous is True,
        "grasp_proxy_continuous": (
            record.grasp_proxy_continuous is True
        ),
        "retract_steps": (
            retract_steps
            if gates["calibration_retract_sample_count"]
            else None
        ),
        "samples_per_stage": (
            record.limits.samples_per_stage
            if limits_valid
            else None
        ),
        "stages": stage_summary,
        "total_g3b_samples": (
            sum(item.samples for item in record.stage_records)
            if gates["calibration_stage_sample_counts"]
            else None
        ),
    }


def _finite_vector(value: Any, dimensions: int) -> tuple[float, ...] | None:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != dimensions
    ):
        return None
    converted: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        converted.append(number)
    return tuple(converted)


def _finite_nonzero_quaternion(
    value: Any,
) -> tuple[float, ...] | None:
    quaternion = _finite_vector(value, 4)
    if (
        quaternion is None
        or math.sqrt(sum(item * item for item in quaternion)) <= 1e-12
    ):
        return None
    return quaternion
