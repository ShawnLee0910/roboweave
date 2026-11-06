"""Private, tolerant attestation for restoring one resident B1 scene.

The snapshot contract intentionally excludes simulator clocks, frame counters,
timestamps, random generator internals, and other volatile values.  A live
binding must serialize only stable task / environment state needed to detect
calibration carryover.  Admission uses explicit numeric tolerances and exact
boolean / structural checks; recorded hashes are audit identities, not a
claim that an unstable full simulator dump is byte-for-byte reproducible.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SAME_SCENE_RESET_ATTESTATION_VERSION = (
    "behavior2026.same-scene-reset-attestation/v1"
)
PRIVATE_RESET_SNAPSHOT_VERSION = (
    "behavior2026.private-reset-snapshot/v2"
)
_SNAPSHOT_SOURCE = "private_evaluator_tolerant_canonical_state_v1"
_PHASES = {"pre_calibration", "post_reset"}


class SameSceneResetAttestationError(ValueError):
    """Malformed private reset evidence or comparison contract."""


@dataclass(frozen=True, slots=True)
class SameSceneResetTolerances:
    base_position_m: float = 0.03
    base_orientation_rad: float = 0.05
    joint_position_rad: float = 0.05
    stable_state_numeric_abs: float = 1e-5
    stable_state_orientation_rad: float = 1e-3

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True)
class PrivateResetSnapshot:
    """Evaluator-owned state that must never enter public/model artifacts."""

    phase: str
    robot_base_position: tuple[float, ...]
    robot_base_quaternion_xyzw: tuple[float, ...]
    robot_joint_positions: tuple[float, ...]
    held_none: bool
    inside_leaf_ids: tuple[str, ...]
    initial_inside_leaves: tuple[bool, ...]
    stable_serialized_state: Mapping[str, Any] = field(repr=False)
    excluded_volatile_fields: tuple[str, ...]
    grounded_option_index: int = 0
    grounded_option_identity_sha256: str | None = None
    grounded_option_selection_mode: str = "legacy_single_or_bounded_profile"
    source: str = _SNAPSHOT_SOURCE
    schema_version: str = PRIVATE_RESET_SNAPSHOT_VERSION
    stable_state_sha256: str = field(init=False)
    stable_state_schema_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PRIVATE_RESET_SNAPSHOT_VERSION:
            raise ValueError("unexpected private reset snapshot version")
        if self.phase not in _PHASES:
            raise ValueError("snapshot phase is invalid")
        if self.source != _SNAPSHOT_SOURCE:
            raise ValueError("snapshot source is invalid")
        object.__setattr__(
            self,
            "robot_base_position",
            _finite_vector(
                self.robot_base_position,
                3,
                "robot_base_position",
            ),
        )
        quaternion = _finite_vector(
            self.robot_base_quaternion_xyzw,
            4,
            "robot_base_quaternion_xyzw",
        )
        if float(np.linalg.norm(quaternion)) <= 1e-8:
            raise ValueError("robot base quaternion has zero norm")
        object.__setattr__(
            self,
            "robot_base_quaternion_xyzw",
            quaternion,
        )
        joints = tuple(float(item) for item in self.robot_joint_positions)
        if not joints or not all(math.isfinite(item) for item in joints):
            raise ValueError(
                "robot_joint_positions must be a non-empty finite vector"
            )
        object.__setattr__(self, "robot_joint_positions", joints)
        if self.held_none is not True:
            raise ValueError(
                "reset snapshot requires no object held by either hand"
            )
        leaf_ids = tuple(self.inside_leaf_ids)
        if (
            not leaf_ids
            or len(set(leaf_ids)) != len(leaf_ids)
            or any(not isinstance(item, str) or not item for item in leaf_ids)
        ):
            raise ValueError("inside_leaf_ids must name one or more unique leaves")
        object.__setattr__(self, "inside_leaf_ids", leaf_ids)
        leaves = tuple(self.initial_inside_leaves)
        if len(leaves) != len(leaf_ids) or any(
            not isinstance(item, bool) for item in leaves
        ):
            raise ValueError(
                "initial_inside_leaves must match inside_leaf_ids"
            )
        object.__setattr__(self, "initial_inside_leaves", leaves)
        if (
            isinstance(self.grounded_option_index, bool)
            or not isinstance(self.grounded_option_index, int)
            or self.grounded_option_index < 0
        ):
            raise ValueError("grounded_option_index must be non-negative")
        option_identity_sha256 = self.grounded_option_identity_sha256
        if option_identity_sha256 is None:
            option_identity_sha256 = _json_sha256(list(leaf_ids))
        if (
            not isinstance(option_identity_sha256, str)
            or len(option_identity_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in option_identity_sha256
            )
        ):
            raise ValueError(
                "grounded_option_identity_sha256 must be lowercase SHA-256"
            )
        object.__setattr__(
            self,
            "grounded_option_identity_sha256",
            option_identity_sha256,
        )
        if (
            not isinstance(self.grounded_option_selection_mode, str)
            or not self.grounded_option_selection_mode
        ):
            raise ValueError(
                "grounded_option_selection_mode must be non-empty"
            )
        excluded = tuple(self.excluded_volatile_fields)
        if (
            not excluded
            or len(set(excluded)) != len(excluded)
            or any(not isinstance(item, str) or not item for item in excluded)
        ):
            raise ValueError(
                "excluded_volatile_fields must explicitly name omissions"
            )
        object.__setattr__(self, "excluded_volatile_fields", excluded)
        stable = _canonical_json_mapping(self.stable_serialized_state)
        object.__setattr__(
            self,
            "stable_serialized_state",
            MappingProxyType(stable),
        )
        object.__setattr__(
            self,
            "stable_state_sha256",
            _json_sha256(stable),
        )
        object.__setattr__(
            self,
            "stable_state_schema_sha256",
            _json_sha256(_schema_signature(stable)),
        )

    def to_private_record(self, *, include_stable_state: bool) -> dict[str, Any]:
        record = {
            "schema_version": self.schema_version,
            "source": self.source,
            "visibility": "private",
            "phase": self.phase,
            "robot_base_position": self.robot_base_position,
            "robot_base_quaternion_xyzw": (
                self.robot_base_quaternion_xyzw
            ),
            "robot_joint_positions": self.robot_joint_positions,
            "held_none": self.held_none,
            "inside_leaf_ids": self.inside_leaf_ids,
            "initial_inside_leaves": self.initial_inside_leaves,
            "goal_leaf_count": len(self.inside_leaf_ids),
            "grounded_option_index": self.grounded_option_index,
            "grounded_option_identity_sha256": (
                self.grounded_option_identity_sha256
            ),
            "grounded_option_selection_mode": (
                self.grounded_option_selection_mode
            ),
            "excluded_volatile_fields": self.excluded_volatile_fields,
            "stable_state_sha256": self.stable_state_sha256,
            "stable_state_schema_sha256": (
                self.stable_state_schema_sha256
            ),
            "full_state_exact_hash_claimed": False,
        }
        if include_stable_state:
            record["stable_serialized_state"] = _thaw(
                self.stable_serialized_state
            )
        return record


@dataclass(frozen=True, slots=True)
class SameSceneResetAttestation:
    passed: bool
    failures: tuple[Mapping[str, Any], ...]
    tolerances: SameSceneResetTolerances
    pre_snapshot_sha256: str
    post_snapshot_sha256: str
    stable_schema_match: bool
    stable_hash_match: bool
    hash_equality_required: bool = False
    schema_version: str = SAME_SCENE_RESET_ATTESTATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SAME_SCENE_RESET_ATTESTATION_VERSION:
            raise ValueError("unexpected reset attestation version")
        failures = tuple(MappingProxyType(dict(item)) for item in self.failures)
        object.__setattr__(self, "failures", failures)
        if self.passed != (not failures):
            raise ValueError("attestation pass flag disagrees with failures")
        if not isinstance(self.tolerances, SameSceneResetTolerances):
            raise TypeError("tolerances are invalid")
        for name in ("pre_snapshot_sha256", "post_snapshot_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} is not sha256")
        if self.hash_equality_required is not False:
            raise ValueError(
                "tolerant reset attestation cannot require exact state hash"
            )

    def require_pass(self) -> None:
        if not self.passed:
            raise RuntimeError(
                "same-scene reset attestation failed closed: "
                f"{[dict(item) for item in self.failures]}"
            )

    def to_private_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "visibility": "private",
            "passed": self.passed,
            "failures": [dict(item) for item in self.failures],
            "tolerances": {
                name: getattr(self.tolerances, name)
                for name in self.tolerances.__dataclass_fields__
            },
            "pre_snapshot_sha256": self.pre_snapshot_sha256,
            "post_snapshot_sha256": self.post_snapshot_sha256,
            "stable_schema_match": self.stable_schema_match,
            "stable_hash_match": self.stable_hash_match,
            "hash_equality_required": self.hash_equality_required,
            "full_state_exact_hash_claimed": False,
        }


def attest_same_scene_reset(
    *,
    pre: PrivateResetSnapshot,
    post: PrivateResetSnapshot,
    tolerances: SameSceneResetTolerances,
) -> SameSceneResetAttestation:
    """Compare stable private reset evidence with numeric tolerances."""

    if not isinstance(pre, PrivateResetSnapshot) or not isinstance(
        post,
        PrivateResetSnapshot,
    ):
        raise TypeError("pre and post must be PrivateResetSnapshot")
    if pre.phase != "pre_calibration" or post.phase != "post_reset":
        raise SameSceneResetAttestationError(
            "reset snapshots have incorrect phases"
        )
    if not isinstance(tolerances, SameSceneResetTolerances):
        raise TypeError("tolerances must be SameSceneResetTolerances")

    failures: list[dict[str, Any]] = []
    position_error = float(
        np.linalg.norm(
            np.asarray(pre.robot_base_position)
            - np.asarray(post.robot_base_position)
        )
    )
    if position_error > tolerances.base_position_m:
        failures.append(
            {
                "field": "robot_base_position",
                "error_m": position_error,
                "tolerance_m": tolerances.base_position_m,
            }
        )
    orientation_error = _quaternion_angle(
        pre.robot_base_quaternion_xyzw,
        post.robot_base_quaternion_xyzw,
    )
    if orientation_error > tolerances.base_orientation_rad:
        failures.append(
            {
                "field": "robot_base_orientation",
                "error_rad": orientation_error,
                "tolerance_rad": tolerances.base_orientation_rad,
            }
        )
    pre_joints = np.asarray(pre.robot_joint_positions)
    post_joints = np.asarray(post.robot_joint_positions)
    if pre_joints.shape != post_joints.shape:
        failures.append(
            {
                "field": "robot_joint_positions.shape",
                "pre": list(pre_joints.shape),
                "post": list(post_joints.shape),
            }
        )
    else:
        joint_error = float(np.max(np.abs(pre_joints - post_joints)))
        if joint_error > tolerances.joint_position_rad:
            failures.append(
                {
                    "field": "robot_joint_positions",
                    "max_error_rad": joint_error,
                    "tolerance_rad": tolerances.joint_position_rad,
                }
            )
    if not pre.held_none or not post.held_none:
        failures.append({"field": "held_none", "required": True})
    if pre.inside_leaf_ids != post.inside_leaf_ids:
        failures.append(
            {
                "field": "inside_leaf_ids",
                "reason": "official Inside leaf identity/order changed",
            }
        )
    if pre.initial_inside_leaves != post.initial_inside_leaves:
        failures.append(
            {
                "field": "initial_inside_leaves",
                "reason": "official canonical goal truth changed across reset",
                "pre": list(pre.initial_inside_leaves),
                "post": list(post.initial_inside_leaves),
            }
        )
    if pre.grounded_option_index != post.grounded_option_index:
        failures.append(
            {
                "field": "grounded_option_index",
                "reason": "selected official option index changed across reset",
                "pre": pre.grounded_option_index,
                "post": post.grounded_option_index,
            }
        )
    if (
        pre.grounded_option_identity_sha256
        != post.grounded_option_identity_sha256
    ):
        failures.append(
            {
                "field": "grounded_option_identity_sha256",
                "reason": "selected official option identity changed across reset",
                "pre": pre.grounded_option_identity_sha256,
                "post": post.grounded_option_identity_sha256,
            }
        )
    if (
        pre.grounded_option_selection_mode
        != post.grounded_option_selection_mode
    ):
        failures.append(
            {
                "field": "grounded_option_selection_mode",
                "reason": "selected option owner mode changed across reset",
            }
        )
    if pre.excluded_volatile_fields != post.excluded_volatile_fields:
        failures.append(
            {
                "field": "excluded_volatile_fields",
                "reason": "capture policy changed across reset",
            }
        )

    stable_schema_match = (
        pre.stable_state_schema_sha256
        == post.stable_state_schema_sha256
    )
    if not stable_schema_match:
        failures.append(
            {
                "field": "stable_serialized_state.schema",
                "reason": "stable state structure changed",
            }
        )
    else:
        _compare_stable(
            pre.stable_serialized_state,
            post.stable_serialized_state,
            path="stable_serialized_state",
            numeric_tolerance=tolerances.stable_state_numeric_abs,
            orientation_tolerance_rad=(
                tolerances.stable_state_orientation_rad
            ),
            failures=failures,
        )

    return SameSceneResetAttestation(
        passed=not failures,
        failures=tuple(failures),
        tolerances=tolerances,
        pre_snapshot_sha256=_snapshot_sha256(pre),
        post_snapshot_sha256=_snapshot_sha256(post),
        stable_schema_match=stable_schema_match,
        stable_hash_match=(
            pre.stable_state_sha256 == post.stable_state_sha256
        ),
        hash_equality_required=False,
    )


def _compare_stable(
    left: Any,
    right: Any,
    *,
    path: str,
    numeric_tolerance: float,
    orientation_tolerance_rad: float,
    failures: list[dict[str, Any]],
) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            failures.append(
                {"field": path, "reason": "mapping keys changed"}
            )
            return
        for key in sorted(left):
            _compare_stable(
                left[key],
                right[key],
                path=f"{path}.{key}",
                numeric_tolerance=numeric_tolerance,
                orientation_tolerance_rad=orientation_tolerance_rad,
                failures=failures,
            )
        return
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            failures.append(
                {"field": path, "reason": "sequence length changed"}
            )
            return
        if (
            path.endswith(".orientation_xyzw")
            and len(left) == 4
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in (*left, *right)
            )
        ):
            error_rad = _quaternion_angle(left, right)
            if error_rad > orientation_tolerance_rad:
                failures.append(
                    {
                        "field": path,
                        "error_rad": error_rad,
                        "tolerance_rad": orientation_tolerance_rad,
                    }
                )
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _compare_stable(
                left_item,
                right_item,
                path=f"{path}[{index}]",
                numeric_tolerance=numeric_tolerance,
                orientation_tolerance_rad=orientation_tolerance_rad,
                failures=failures,
            )
        return
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        error = abs(float(left) - float(right))
        if error > numeric_tolerance:
            failures.append(
                {
                    "field": path,
                    "numeric_error": error,
                    "tolerance": numeric_tolerance,
                }
            )
        return
    if left != right:
        failures.append(
            {
                "field": path,
                "pre": left,
                "post": right,
            }
        )


def _canonical_json_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "stable_serialized_state must be a non-empty mapping"
        )
    result: dict[str, Any] = {}
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        if not isinstance(key, str) or not key:
            raise ValueError(
                "stable_serialized_state has invalid mapping key"
            )
        result[key] = _canonical_json_value(item, path=key)
    return result


def _canonical_json_value(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        if not value:
            return MappingProxyType({})
        result: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} has invalid mapping key")
            result[key] = _canonical_json_value(
                item,
                path=f"{path}.{key}",
            )
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{path} is not finite")
        return result
    raise ValueError(f"{path} is not stable JSON state")


def _schema_signature(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _schema_signature(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, tuple):
        return [_schema_signature(item) for item in value]
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, float):
        return "number"
    if value is None:
        return "null"
    return "string"


def _snapshot_sha256(snapshot: PrivateResetSnapshot) -> str:
    return _json_sha256(snapshot.to_private_record(include_stable_state=True))


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _thaw(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _finite_vector(
    value: Sequence[float],
    size: int,
    name: str,
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and {size}D")
    return tuple(float(item) for item in array)


def _quaternion_angle(left: Sequence[float], right: Sequence[float]) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    dot = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
    return 2.0 * math.acos(dot)
