"""Public-only frozen witness for the first single-leaf physical canary.

The witness is authored in the calibration segment from onboard public
RGB-D/proprioception and normalized image-point high-level actions.  It
contains no simulator object identity, pose, BDDL scope, map, or predicate
truth.  A private official verifier may approve freezing the witness, but
that judgement is deliberately not an input to replay.

Source observation hashes prove what the author saw.  They are diagnostics,
not byte-exact replay gates: a reset renderer may produce harmless pixel
differences.  During replay the normal ``grasp_at`` / ``place_inside_at``
plugins re-ground the normalized point in the current public RGB-D and
repeat their depth, workspace, collision-aware IK, and physical safety
checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from behavior_hl.embodied.policy.protocol import PointActionArgs
from behavior_hl.embodied.policy.registry import ActionRegistry
from behavior_hl.embodied.runtime.observation import PublicRobotObservation


PUBLIC_SINGLE_LEAF_WITNESS_VERSION = (
    "behavior2026.public-single-leaf-witness/v2"
)
PUBLIC_SINGLE_LEAF_REPLAY_VERSION = (
    "behavior2026.public-single-leaf-replay/v1"
)
PRIVATE_SINGLE_LEAF_ADMISSION_VERSION = (
    "behavior2026.private-single-leaf-admission/v1"
)
PUBLIC_SINGLE_LEAF_ADMISSION_SUMMARY_VERSION = (
    "behavior2026.public-single-leaf-admission-summary/v1"
)
_POINT_ACTIONS = frozenset({"grasp_at", "place_inside_at"})
_BASE_ACTIONS = frozenset(
    {
        "move_backward",
        "move_forward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
    }
)
_FORBIDDEN_SERIALIZED_FRAGMENTS = (
    "bddl",
    "goal_truth",
    "object_scope",
    "occupancy_map",
    "resolved_object",
    "seg_instance",
    "success_truth",
    "task_scope",
    "trav_map",
    "world_pose",
)
_SUMMARY_ADMISSION = object()
_WITNESS_FREEZE_ADMISSION = object()


class PublicSingleLeafWitnessError(RuntimeError):
    """Fail-closed witness authoring or replay error."""


@dataclass(frozen=True, slots=True)
class PublicSingleLeafAdmissionSummary:
    """De-identified proof shape safe to bind into the public witness."""

    attestation_sha256: str
    leaf_count: int = 6
    before_true_count: int = 0
    after_true_count: int = 1
    newly_true_count: int = 1
    unchanged_leaf_count: int = 5
    _admission: object = field(repr=False, compare=False, default=None)
    schema_version: str = (
        PUBLIC_SINGLE_LEAF_ADMISSION_SUMMARY_VERSION
    )

    def __post_init__(self) -> None:
        if self._admission is not _SUMMARY_ADMISSION:
            raise ValueError(
                "admission summary must derive from typed private evidence"
            )
        if (
            self.schema_version
            != PUBLIC_SINGLE_LEAF_ADMISSION_SUMMARY_VERSION
        ):
            raise ValueError("admission summary version mismatch")
        _require_sha256(
            self.attestation_sha256,
            "attestation_sha256",
        )
        expected = {
            "leaf_count": 6,
            "before_true_count": 0,
            "after_true_count": 1,
            "newly_true_count": 1,
            "unchanged_leaf_count": 5,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"single-leaf admission summary {name} must be {value}"
                )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attestation_sha256": self.attestation_sha256,
            "leaf_count": self.leaf_count,
            "before_true_count": self.before_true_count,
            "after_true_count": self.after_true_count,
            "newly_true_count": self.newly_true_count,
            "unchanged_leaf_count": self.unchanged_leaf_count,
            "leaf_identifiers_included": False,
        }


@dataclass(frozen=True, slots=True)
class PrivateSingleLeafAdmissionEvidence:
    """Evaluator-only six-leaf transition evidence used to admit freezing.

    Leaf identifiers and truth vectors never enter the public witness.  The
    public side receives only :meth:`public_summary`, whose hash commits to
    this complete private record without identifying the transitioned leaf.
    """

    before_leaf_truth: Mapping[str, bool]
    after_leaf_truth: Mapping[str, bool]
    verifier_version: str
    schema_version: str = PRIVATE_SINGLE_LEAF_ADMISSION_VERSION
    attestation_sha256: str = field(init=False)
    _newly_true_leaf_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != PRIVATE_SINGLE_LEAF_ADMISSION_VERSION:
            raise ValueError("private admission evidence version mismatch")
        if (
            not isinstance(self.verifier_version, str)
            or not self.verifier_version.strip()
        ):
            raise ValueError("verifier_version must be explicit")
        before = _strict_leaf_truth(
            self.before_leaf_truth,
            "before_leaf_truth",
        )
        after = _strict_leaf_truth(
            self.after_leaf_truth,
            "after_leaf_truth",
        )
        if set(before) != set(after):
            raise PublicSingleLeafWitnessError(
                "before/after leaf identifiers changed"
            )
        if len(before) != 6:
            raise PublicSingleLeafWitnessError(
                "private admission requires exactly six official leaves"
            )
        if any(before.values()):
            raise PublicSingleLeafWitnessError(
                "all six official leaves must be false before authoring"
            )
        newly_true = tuple(
            leaf_id
            for leaf_id in sorted(before)
            if before[leaf_id] is False and after[leaf_id] is True
        )
        changed_other = tuple(
            leaf_id
            for leaf_id in sorted(before)
            if before[leaf_id] != after[leaf_id]
            and leaf_id not in newly_true
        )
        if len(newly_true) != 1 or changed_other:
            raise PublicSingleLeafWitnessError(
                "private admission requires exactly one false-to-true "
                "transition and five unchanged leaves"
            )
        if sum(after.values()) != 1:
            raise PublicSingleLeafWitnessError(
                "exactly one official leaf must be true after authoring"
            )
        object.__setattr__(
            self,
            "before_leaf_truth",
            MappingProxyType(before),
        )
        object.__setattr__(
            self,
            "after_leaf_truth",
            MappingProxyType(after),
        )
        object.__setattr__(
            self,
            "_newly_true_leaf_id",
            newly_true[0],
        )
        private_record = {
            "schema_version": self.schema_version,
            "verifier_version": self.verifier_version,
            "leaf_ids": sorted(before),
            "before_leaf_truth": {
                key: before[key] for key in sorted(before)
            },
            "after_leaf_truth": {
                key: after[key] for key in sorted(after)
            },
            "newly_true_leaf_id": newly_true[0],
        }
        object.__setattr__(
            self,
            "attestation_sha256",
            hashlib.sha256(
                _canonical_json(private_record).encode("utf-8")
            ).hexdigest(),
        )

    def public_summary(self) -> PublicSingleLeafAdmissionSummary:
        return PublicSingleLeafAdmissionSummary(
            attestation_sha256=self.attestation_sha256,
            _admission=_SUMMARY_ADMISSION,
        )

    def to_private_record(self) -> dict[str, Any]:
        """Return complete evidence for a private evaluator artifact only."""

        return {
            "schema_version": self.schema_version,
            "verifier_version": self.verifier_version,
            "leaf_ids": sorted(self.before_leaf_truth),
            "before_leaf_truth": dict(self.before_leaf_truth),
            "after_leaf_truth": dict(self.after_leaf_truth),
            "newly_true_leaf_id": self._newly_true_leaf_id,
            "attestation_sha256": self.attestation_sha256,
            "visibility": "private",
        }


@dataclass(frozen=True, slots=True)
class PublicWitnessDecision:
    """One executed public action and its public-only provenance.

    ``source_frame_sha256`` is the hash of the exact public observation from
    which the author chose the action.  It is retained for audit and drift
    diagnostics only.  Point actions additionally name the public camera
    whose normalized image point they used.
    """

    sequence: int
    action: Mapping[str, Any]
    source_frame_sha256: str
    source_observation_schema_version: str
    public_result_status: str
    simulator_steps: int
    point_source_camera: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer")
        payload = _strict_json_object(self.action, "action")
        _require_sha256(
            self.source_frame_sha256,
            "source_frame_sha256",
        )
        if (
            not isinstance(self.source_observation_schema_version, str)
            or not self.source_observation_schema_version
        ):
            raise ValueError(
                "source_observation_schema_version must be non-empty"
            )
        if (
            not isinstance(self.public_result_status, str)
            or not self.public_result_status
        ):
            raise ValueError("public_result_status must be non-empty")
        if (
            isinstance(self.simulator_steps, bool)
            or not isinstance(self.simulator_steps, int)
            or self.simulator_steps < 0
        ):
            raise ValueError(
                "simulator_steps must be a non-negative integer"
            )
        if self.point_source_camera is not None and (
            self.point_source_camera
            not in {"head", "left_wrist", "right_wrist"}
        ):
            raise ValueError("point_source_camera is not public")
        object.__setattr__(
            self,
            "action",
            MappingProxyType(payload),
        )

    @classmethod
    def from_public_execution(
        cls,
        *,
        sequence: int,
        registry: ActionRegistry,
        action: Mapping[str, Any],
        observation: PublicRobotObservation,
        public_result_status: str,
        simulator_steps: int,
    ) -> "PublicWitnessDecision":
        """Bind an executed action to the exact public author observation."""

        if not isinstance(registry, ActionRegistry):
            raise TypeError("registry must be an ActionRegistry")
        if not isinstance(observation, PublicRobotObservation):
            raise TypeError(
                "observation must be PublicRobotObservation"
            )
        parsed = registry.parse_payload(action)
        spec = registry.spec(parsed.action)
        if public_result_status not in spec.public_statuses:
            raise PublicSingleLeafWitnessError(
                f"{parsed.action} returned an unregistered public status"
            )
        camera: str | None = None
        if parsed.action in _POINT_ACTIONS:
            if not isinstance(parsed.args, PointActionArgs):
                raise PublicSingleLeafWitnessError(
                    "point action did not parse as PointActionArgs"
                )
            camera = parsed.args.camera
            _require_public_point_camera(observation, camera)
        return cls(
            sequence=sequence,
            action=parsed.to_dict(),
            source_frame_sha256=public_observation_sha256(observation),
            source_observation_schema_version=(
                observation.schema_version
            ),
            public_result_status=public_result_status,
            simulator_steps=simulator_steps,
            point_source_camera=camera,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "action": dict(self.action),
            "source_frame_sha256": self.source_frame_sha256,
            "source_observation_schema_version": (
                self.source_observation_schema_version
            ),
            "public_result_status": self.public_result_status,
            "simulator_steps": self.simulator_steps,
            "point_source_camera": self.point_source_camera,
        }


@dataclass(frozen=True, slots=True)
class PublicSingleLeafWitness:
    """Frozen full grasp/carry/place action witness for one official leaf."""

    decisions: tuple[PublicWitnessDecision, ...]
    calibrated_hand: str
    witness_author: str
    registry_schema_version: str
    registry_sha256: str
    admission_summary: PublicSingleLeafAdmissionSummary
    _admission: object = field(repr=False, compare=False)
    pilot_kind: str = "single_leaf_physical_chain"
    schema_version: str = PUBLIC_SINGLE_LEAF_WITNESS_VERSION
    witness_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self._admission is not _WITNESS_FREEZE_ADMISSION:
            raise ValueError(
                "public witness must use typed private admission freeze"
            )
        if self.schema_version != PUBLIC_SINGLE_LEAF_WITNESS_VERSION:
            raise ValueError("witness schema version mismatch")
        if self.pilot_kind != "single_leaf_physical_chain":
            raise ValueError("pilot_kind must be single_leaf_physical_chain")
        if self.calibrated_hand not in {"left", "right"}:
            raise ValueError("calibrated_hand must be left or right")
        if (
            not isinstance(self.witness_author, str)
            or not self.witness_author.strip()
        ):
            raise ValueError("witness_author must be explicit")
        if (
            not isinstance(self.registry_schema_version, str)
            or not self.registry_schema_version
        ):
            raise ValueError("registry_schema_version must be non-empty")
        _require_sha256(self.registry_sha256, "registry_sha256")
        if not isinstance(
            self.admission_summary,
            PublicSingleLeafAdmissionSummary,
        ):
            raise TypeError(
                "admission_summary must be "
                "PublicSingleLeafAdmissionSummary"
            )
        decisions = tuple(self.decisions)
        if not decisions or any(
            not isinstance(item, PublicWitnessDecision)
            for item in decisions
        ):
            raise ValueError("decisions must be PublicWitnessDecision values")
        if tuple(item.sequence for item in decisions) != tuple(
            range(len(decisions))
        ):
            raise ValueError("decision sequences must be contiguous from zero")
        object.__setattr__(self, "decisions", decisions)
        payload = self._unhashed_manifest()
        serialized = _canonical_json(payload)
        folded = serialized.casefold()
        leaked = [
            fragment
            for fragment in _FORBIDDEN_SERIALIZED_FRAGMENTS
            if fragment in folded
        ]
        if leaked:
            raise ValueError(
                "witness contains forbidden private fragments: "
                f"{sorted(leaked)}"
            )
        object.__setattr__(
            self,
            "witness_sha256",
            hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def freeze_after_private_canary(
        cls,
        *,
        decisions: Sequence[PublicWitnessDecision],
        calibrated_hand: str,
        witness_author: str,
        registry: ActionRegistry,
        private_admission_evidence: (
            PrivateSingleLeafAdmissionEvidence
        ),
    ) -> "PublicSingleLeafWitness":
        """Freeze only after the private official one-leaf judge passes.

        Complete private leaf identities/truth remain evaluator-only.  The
        witness binds only their de-identified attestation summary and hash.
        """

        if not isinstance(
            private_admission_evidence,
            PrivateSingleLeafAdmissionEvidence,
        ):
            raise TypeError(
                "private_admission_evidence must be "
                "PrivateSingleLeafAdmissionEvidence"
            )
        if not isinstance(registry, ActionRegistry):
            raise TypeError("registry must be an ActionRegistry")
        witness = cls(
            decisions=tuple(decisions),
            calibrated_hand=calibrated_hand,
            witness_author=witness_author,
            registry_schema_version=registry.schema_version,
            registry_sha256=_registry_sha256(registry),
            admission_summary=(
                private_admission_evidence.public_summary()
            ),
            _admission=_WITNESS_FREEZE_ADMISSION,
        )
        witness.validate_against(registry)
        return witness

    def validate_against(self, registry: ActionRegistry) -> None:
        """Validate registry identity and the complete public action chain."""

        if not isinstance(registry, ActionRegistry):
            raise TypeError("registry must be an ActionRegistry")
        if (
            registry.schema_version != self.registry_schema_version
            or _registry_sha256(registry) != self.registry_sha256
        ):
            raise PublicSingleLeafWitnessError(
                "witness registry identity changed"
            )
        parsed = [
            registry.parse_payload(decision.action)
            for decision in self.decisions
        ]
        names = tuple(item.action for item in parsed)
        if "stop" in names or "release" in names:
            raise PublicSingleLeafWitnessError(
                "witness must end through place_inside_at and verifier "
                "termination, not stop/release"
            )
        if names.count("grasp_at") != 1:
            raise PublicSingleLeafWitnessError(
                "witness requires exactly one grasp_at"
            )
        if names.count("place_inside_at") != 1:
            raise PublicSingleLeafWitnessError(
                "witness requires exactly one place_inside_at"
            )
        grasp_index = names.index("grasp_at")
        place_index = names.index("place_inside_at")
        if grasp_index >= place_index or place_index != len(names) - 1:
            raise PublicSingleLeafWitnessError(
                "witness must grasp before a terminal place_inside_at"
            )
        carrying_actions = names[grasp_index + 1 : place_index]
        if not any(name in _BASE_ACTIONS for name in carrying_actions):
            raise PublicSingleLeafWitnessError(
                "witness must contain physical carrying base motion"
            )
        for index, (decision, action) in enumerate(
            zip(self.decisions, parsed)
        ):
            spec = registry.spec(action.action)
            if decision.public_result_status not in spec.public_statuses:
                raise PublicSingleLeafWitnessError(
                    f"decision {index} has an unregistered public status"
                )
            if action.action in _POINT_ACTIONS:
                if (
                    not isinstance(action.args, PointActionArgs)
                    or action.args.hand != self.calibrated_hand
                    or decision.point_source_camera
                    != action.args.camera
                ):
                    raise PublicSingleLeafWitnessError(
                        "point action changed calibrated hand/camera"
                    )
            elif decision.point_source_camera is not None:
                raise PublicSingleLeafWitnessError(
                    "non-point action cannot claim a point camera"
                )
        grasp = self.decisions[grasp_index]
        place = self.decisions[place_index]
        if grasp.public_result_status != "grasp_likely":
            raise PublicSingleLeafWitnessError(
                "public grasp proxy did not report grasp_likely"
            )
        if place.public_result_status != "completed":
            raise PublicSingleLeafWitnessError(
                "public place action did not complete"
            )
        if not any(
            self.decisions[index].public_result_status
            in {"completed", "motion_partial"}
            for index in range(grasp_index + 1, place_index)
            if names[index] in _BASE_ACTIONS
        ):
            raise PublicSingleLeafWitnessError(
                "carrying base motion produced no public displacement"
            )

    def _unhashed_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pilot_kind": self.pilot_kind,
            "calibrated_hand": self.calibrated_hand,
            "witness_author": self.witness_author,
            "registry_schema_version": self.registry_schema_version,
            "registry_sha256": self.registry_sha256,
            "private_admission_attestation": (
                self.admission_summary.to_manifest()
            ),
            "decisions": [
                item.to_manifest() for item in self.decisions
            ],
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            **self._unhashed_manifest(),
            "decision_count": len(self.decisions),
            "witness_sha256": self.witness_sha256,
            "source_hash_role": "audit_only_not_replay_admission",
            "replay_grounding": (
                "current_public_rgbd_depth_workspace_collision_aware_ik"
            ),
        }


class PublicSingleLeafReplayModel:
    """Replay the exact frozen actions without consulting private truth."""

    replay_version = PUBLIC_SINGLE_LEAF_REPLAY_VERSION

    def __init__(
        self,
        *,
        registry: ActionRegistry,
        witness: PublicSingleLeafWitness,
    ) -> None:
        if not isinstance(registry, ActionRegistry):
            raise TypeError("registry must be an ActionRegistry")
        if not isinstance(witness, PublicSingleLeafWitness):
            raise TypeError("witness must be PublicSingleLeafWitness")
        witness.validate_against(registry)
        self.registry = registry
        self.witness = witness
        self._index = 0
        self._poisoned = False
        self._source_frame_diagnostics: list[dict[str, Any]] = []

    @property
    def consumed_count(self) -> int:
        return self._index

    @property
    def consumption_diagnostics(self) -> Mapping[str, Any]:
        """Return public, action-free exact-replay consumption progress."""

        total = len(self.witness.decisions)
        return MappingProxyType(
            {
                "schema_version": self.replay_version,
                "witness_sha256": self.witness.witness_sha256,
                "consumed_count": self._index,
                "total_count": total,
                "remaining_count": total - self._index,
                "fully_consumed": self._index == total,
                "poisoned": self._poisoned,
            }
        )

    @property
    def source_frame_diagnostics(
        self,
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            MappingProxyType(dict(item))
            for item in self._source_frame_diagnostics
        )

    def complete(
        self,
        prompt: str,
        observation: PublicRobotObservation,
    ) -> str:
        if self._poisoned:
            raise PublicSingleLeafWitnessError("replay model is poisoned")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be non-empty")
        if not isinstance(observation, PublicRobotObservation):
            raise TypeError(
                "observation must be PublicRobotObservation"
            )
        if self._index >= len(self.witness.decisions):
            self._poisoned = True
            raise PublicSingleLeafWitnessError(
                "witness exhausted before private verifier termination"
            )
        decision = self.witness.decisions[self._index]
        if (
            observation.schema_version
            != decision.source_observation_schema_version
        ):
            self._poisoned = True
            raise PublicSingleLeafWitnessError(
                "public observation schema changed during replay"
            )
        # This comparison is diagnostic only.  In particular, it must not
        # prevent the current public RGB-D controller from re-validating a
        # normalized grasp/place point after harmless render differences.
        current_sha256 = public_observation_sha256(observation)
        self._source_frame_diagnostics.append(
            {
                "sequence": self._index,
                "source_frame_sha256": (
                    decision.source_frame_sha256
                ),
                "replay_frame_sha256": current_sha256,
                "exact_match": (
                    current_sha256 == decision.source_frame_sha256
                ),
                "admission_effect": "diagnostic_only",
            }
        )
        try:
            action = self.registry.parse_payload(decision.action)
            if action.action in _POINT_ACTIONS:
                assert isinstance(action.args, PointActionArgs)
                _require_public_point_camera(
                    observation,
                    action.args.camera,
                )
        except Exception:
            self._poisoned = True
            raise
        self._index += 1
        return _canonical_json(action.to_dict())

    def require_fully_consumed(self) -> None:
        if self._index != len(self.witness.decisions):
            raise PublicSingleLeafWitnessError(
                "private verifier terminated before exact witness replay "
                f"completed: {self._index}/{len(self.witness.decisions)}"
            )


def public_observation_sha256(
    observation: PublicRobotObservation,
) -> str:
    """Hash the complete immutable public observation, excluding step ID."""

    if not isinstance(observation, PublicRobotObservation):
        raise TypeError("observation must be PublicRobotObservation")
    digest = hashlib.sha256()
    _update_public_hash(
        digest,
        {
            "schema_version": observation.schema_version,
            "values": observation.values,
        },
    )
    return digest.hexdigest()


def _update_public_hash(digest: Any, value: Any) -> None:
    if isinstance(value, Mapping):
        digest.update(b"M")
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("public hash mapping keys must be strings")
            encoded = key.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            _update_public_hash(digest, value[key])
        digest.update(b"m")
        return
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.kind in {"O", "V", "c"}:
            raise TypeError("unsupported public array dtype")
        contiguous = np.ascontiguousarray(array)
        digest.update(b"A")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(_canonical_json(list(contiguous.shape)).encode())
        digest.update(contiguous.tobytes())
        return
    if isinstance(value, tuple):
        digest.update(b"T")
        for item in value:
            _update_public_hash(digest, item)
        digest.update(b"t")
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        digest.update(b"S")
        digest.update(_canonical_json(value).encode("utf-8"))
        return
    raise TypeError(
        f"unsupported public observation value: {type(value).__name__}"
    )


def _require_public_point_camera(
    observation: PublicRobotObservation,
    camera: str,
) -> None:
    try:
        selected = observation.values["cameras"][camera]
        rgb = np.asarray(selected["rgb"])
        depth = np.asarray(selected["depth_linear"])
        poses = np.asarray(observation.values["cam_rel_poses"])
    except Exception as exc:
        raise PublicSingleLeafWitnessError(
            f"public observation lacks {camera} RGB-D grounding"
        ) from exc
    if (
        rgb.ndim != 3
        or rgb.shape[-1] != 3
        or rgb.dtype.kind not in {"u", "i"}
        or depth.ndim not in {2, 3}
        or not np.isfinite(depth).any()
        or poses.ndim != 1
        or not np.isfinite(poses).all()
    ):
        raise PublicSingleLeafWitnessError(
            f"{camera} public RGB-D grounding is malformed"
        )


def _strict_json_object(
    value: Any,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        result = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be strict JSON") from exc
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a JSON object")
    return result


def _strict_leaf_truth(
    value: Any,
    name: str,
) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, bool] = {}
    for leaf_id, truth in value.items():
        if not isinstance(leaf_id, str) or not leaf_id.strip():
            raise ValueError(
                f"{name} leaf identifiers must be non-empty strings"
            )
        if not isinstance(truth, bool):
            raise TypeError(f"{name}[{leaf_id!r}] must be bool")
        result[leaf_id] = truth
    return result


def _registry_sha256(registry: ActionRegistry) -> str:
    """Hash a registry after recursively thawing immutable schema mappings."""

    if not isinstance(registry, ActionRegistry):
        raise TypeError("registry must be an ActionRegistry")
    return hashlib.sha256(
        _canonical_json(_thaw_json(registry.manifest())).encode("utf-8")
    ).hexdigest()


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not strict JSON: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _require_sha256(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase sha256")
