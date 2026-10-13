"""Public-only R1Pro gripper and wrist-image evidence.

The live grasp driver needs a narrow callback after every physical simulator
step.  This module deliberately consumes only the already public R1Pro
proprioception and wrist RGB-D frame.  It does not import OmniGibson and has no
path to task state, object scope, contacts, held-object APIs, world poses, or
goal truth.

The visual proxy is intentionally conservative.  It compares the current
wrist ROI with an explicit empty-close calibration baseline and then requires
the foreground mask, depth, and RGB appearance to remain consistent across a
bounded temporal window.  Missing or malformed public evidence raises instead
of silently returning a positive result.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.r1pro_starter_grasp_driver import (
    PublicGripperWristEvidence,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_torso_release_binding import (
    LEFT_GRIPPER_QPOS,
    LEFT_GRIPPER_QVEL,
    RIGHT_GRIPPER_QPOS,
    RIGHT_GRIPPER_QVEL,
)
from behavior_hl.embodied.policy.observations import (
    CAMERA_RESOLUTIONS,
    PROPRIO_DIM,
)


PUBLIC_WRIST_EVIDENCE_PROVIDER_VERSION = (
    "behavior2026.r1pro-public-wrist-evidence/v1"
)
BOOTSTRAP_PUBLIC_WRIST_EVIDENCE_PROVIDER_VERSION = (
    "behavior2026.r1pro-bootstrap-public-wrist-evidence/v1"
)
PUBLIC_WRIST_CALIBRATION_VERSION = (
    "behavior2026.r1pro-public-wrist-calibration/v1"
)
BOOTSTRAP_PUBLIC_WRIST_CALIBRATION_VERSION = (
    "behavior2026.r1pro-bootstrap-public-wrist-calibration/v1"
)
PUBLIC_WRIST_POSITIVE_WINDOW_VERSION = (
    "behavior2026.r1pro-public-wrist-positive-window/v1"
)

_FORMAL_ADMISSION = object()

_HAND_CAMERA = {
    "left": "left_wrist",
    "right": "right_wrist",
}
_PUBLIC_SNAPSHOT_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "simulator_step",
        "proprio",
        "cam_rel_poses",
        "cameras",
    }
)


class PublicWristEvidenceError(RuntimeError):
    """Fail-closed public-evidence or calibration error."""


@dataclass(frozen=True, slots=True)
class WristVisualThresholds:
    """Frozen public RGB-D thresholds selected during live calibration."""

    empty_close_qpos: tuple[float, float]
    minimum_qpos_gap: float
    qvel_epsilon: float
    minimum_depth_m: float
    maximum_depth_m: float
    minimum_foreground_depth_delta_m: float
    minimum_valid_depth_fraction: float
    minimum_foreground_fraction: float
    minimum_empty_rgb_delta: float
    minimum_temporal_foreground_iou: float
    maximum_temporal_depth_delta_m: float
    maximum_temporal_rgb_delta: float
    consecutive_frames: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "empty_close_qpos",
            _finite_pair(self.empty_close_qpos, "empty_close_qpos"),
        )
        object.__setattr__(
            self,
            "minimum_qpos_gap",
            _finite_positive(
                self.minimum_qpos_gap,
                "minimum_qpos_gap",
            ),
        )
        object.__setattr__(
            self,
            "qvel_epsilon",
            _finite_positive(self.qvel_epsilon, "qvel_epsilon"),
        )
        minimum_depth = _finite_positive(
            self.minimum_depth_m,
            "minimum_depth_m",
        )
        maximum_depth = _finite_positive(
            self.maximum_depth_m,
            "maximum_depth_m",
        )
        if maximum_depth <= minimum_depth:
            raise ValueError(
                "maximum_depth_m must exceed minimum_depth_m"
            )
        object.__setattr__(self, "minimum_depth_m", minimum_depth)
        object.__setattr__(self, "maximum_depth_m", maximum_depth)
        object.__setattr__(
            self,
            "minimum_foreground_depth_delta_m",
            _finite_positive(
                self.minimum_foreground_depth_delta_m,
                "minimum_foreground_depth_delta_m",
            ),
        )
        for name in (
            "minimum_valid_depth_fraction",
            "minimum_foreground_fraction",
            "minimum_temporal_foreground_iou",
        ):
            object.__setattr__(
                self,
                name,
                _positive_fraction(getattr(self, name), name),
            )
        minimum_empty_rgb_delta = _finite_positive(
            self.minimum_empty_rgb_delta,
            "minimum_empty_rgb_delta",
        )
        if minimum_empty_rgb_delta > 255.0:
            raise ValueError(
                "minimum_empty_rgb_delta must not exceed 255"
            )
        object.__setattr__(
            self,
            "minimum_empty_rgb_delta",
            minimum_empty_rgb_delta,
        )
        for name in ("maximum_temporal_rgb_delta",):
            value = _finite_nonnegative(
                getattr(self, name),
                name,
            )
            if value > 255.0:
                raise ValueError(f"{name} must not exceed 255")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "maximum_temporal_depth_delta_m",
            _finite_positive(
                self.maximum_temporal_depth_delta_m,
                "maximum_temporal_depth_delta_m",
            ),
        )
        if (
            isinstance(self.consecutive_frames, bool)
            or not isinstance(self.consecutive_frames, int)
            or self.consecutive_frames < 2
        ):
            raise ValueError("consecutive_frames must be an integer >= 2")

    def to_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                field_name: getattr(self, field_name)
                for field_name in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class BootstrapPublicWristCalibrationRecord:
    """Empty-close negative calibration for the first physical grasp.

    This record is intentionally *not* a formal calibration.  It admits only
    the bootstrap provider used to collect a real public-positive window while
    validating the safe-carry candidate.  It cannot construct the provider
    used by benchmark episodes.
    """

    hand: str
    roi_xyxy: tuple[int, int, int, int]
    empty_rgb: np.ndarray = field(repr=False)
    empty_depth_m: np.ndarray = field(repr=False)
    thresholds: WristVisualThresholds
    safe_carry_candidate_sha256: str
    candidate_ik_only_passed: bool
    empty_close_trials: int
    empty_close_false_positives: int
    baseline_context: str
    source: str = "same_resident_scene_public_rgbd_proprio"
    schema_version: str = BOOTSTRAP_PUBLIC_WRIST_CALIBRATION_VERSION
    calibration_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.hand not in _HAND_CAMERA:
            raise ValueError("hand must be left or right")
        if self.schema_version != BOOTSTRAP_PUBLIC_WRIST_CALIBRATION_VERSION:
            raise ValueError(
                "unexpected bootstrap wrist calibration version"
            )
        if self.source != "same_resident_scene_public_rgbd_proprio":
            raise ValueError("unsupported public wrist calibration source")
        if (
            self.baseline_context
            != "ik_passed_safe_carry_candidate_exact_pose_empty_close"
        ):
            raise ValueError(
                "bootstrap baseline must be captured empty-handed at the "
                "same-scene, collision-aware IK-passed safe-carry candidate "
                "exact pose"
            )
        _sha256(
            self.safe_carry_candidate_sha256,
            "safe-carry candidate",
        )
        if self.candidate_ik_only_passed is not True:
            raise ValueError(
                "bootstrap baseline requires official candidate IK-only PASS"
            )
        roi = _roi(self.roi_xyxy, _HAND_CAMERA[self.hand])
        object.__setattr__(self, "roi_xyxy", roi)
        if not isinstance(self.thresholds, WristVisualThresholds):
            raise TypeError("thresholds must be WristVisualThresholds")
        height = roi[3] - roi[1]
        width = roi[2] - roi[0]
        rgb = _rgb_array(
            self.empty_rgb,
            expected_shape=(height, width, 3),
            name="empty_rgb",
        )
        depth = _depth_array(
            self.empty_depth_m,
            expected_shape=(height, width),
            name="empty_depth_m",
        )
        if np.count_nonzero(_valid_depth(depth)) == 0:
            raise ValueError("empty_depth_m contains no valid depth")
        object.__setattr__(self, "empty_rgb", _read_only(rgb))
        object.__setattr__(self, "empty_depth_m", _read_only(depth))
        for name in (
            "empty_close_trials",
            "empty_close_false_positives",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.empty_close_trials < 3:
            raise ValueError("at least three empty-close trials are required")
        if self.empty_close_false_positives != 0:
            raise ValueError(
                "empty-close calibration has a public false positive"
            )
        canonical = {
            "schema_version": self.schema_version,
            "source": self.source,
            "baseline_context": self.baseline_context,
            "hand": self.hand,
            "roi_xyxy": list(roi),
            "thresholds": dict(self.thresholds.to_manifest()),
            "safe_carry_candidate_sha256": (
                self.safe_carry_candidate_sha256
            ),
            "candidate_ik_only_passed": self.candidate_ik_only_passed,
            "empty_close_trials": self.empty_close_trials,
            "empty_close_false_positives": (
                self.empty_close_false_positives
            ),
            "empty_rgb_sha256": _array_sha256(rgb),
            "empty_depth_sha256": _array_sha256(depth),
        }
        digest = hashlib.sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "calibration_sha256", digest)

    @property
    def camera(self) -> str:
        return _HAND_CAMERA[self.hand]

    def to_manifest(self) -> Mapping[str, Any]:
        """Return calibration provenance without embedding baseline pixels."""

        return MappingProxyType(
            {
                "schema_version": self.schema_version,
                "admission_level": "bootstrap_empty_negative_only",
                "source": self.source,
                "baseline_context": self.baseline_context,
                "hand": self.hand,
                "camera": self.camera,
                "roi_xyxy": self.roi_xyxy,
                "thresholds": dict(self.thresholds.to_manifest()),
                "safe_carry_candidate_sha256": (
                    self.safe_carry_candidate_sha256
                ),
                "candidate_ik_only_passed": (
                    self.candidate_ik_only_passed
                ),
                "empty_close_trials": self.empty_close_trials,
                "empty_close_false_positives": (
                    self.empty_close_false_positives
                ),
                "empty_rgb_sha256": _array_sha256(self.empty_rgb),
                "empty_depth_sha256": _array_sha256(self.empty_depth_m),
                "calibration_sha256": self.calibration_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class PublicWristPositiveWindow:
    """Hashed consecutive evidence computed from public inputs only."""

    hand: str
    bootstrap_calibration_sha256: str
    sample_sha256: tuple[str, ...]
    schema_version: str = PUBLIC_WRIST_POSITIVE_WINDOW_VERSION
    window_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.hand not in _HAND_CAMERA:
            raise ValueError("hand must be left or right")
        if self.schema_version != PUBLIC_WRIST_POSITIVE_WINDOW_VERSION:
            raise ValueError("unexpected public wrist window version")
        _sha256(self.bootstrap_calibration_sha256, "bootstrap calibration")
        samples = tuple(self.sample_sha256)
        if len(samples) < 2:
            raise ValueError(
                "public-positive window requires at least two samples"
            )
        for value in samples:
            _sha256(value, "public-positive sample")
        if len(set(samples)) != len(samples):
            raise ValueError(
                "public-positive window cannot reuse a sampled observation"
            )
        object.__setattr__(self, "sample_sha256", samples)
        payload = {
            "schema_version": self.schema_version,
            "hand": self.hand,
            "bootstrap_calibration_sha256": (
                self.bootstrap_calibration_sha256
            ),
            "sample_sha256": samples,
        }
        object.__setattr__(
            self,
            "window_sha256",
            hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )

    def to_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": self.schema_version,
                "admission_level": "formal_g3b_promoted",
                "hand": self.hand,
                "bootstrap_calibration_sha256": (
                    self.bootstrap_calibration_sha256
                ),
                "sample_sha256": self.sample_sha256,
                "window_sha256": self.window_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class PublicWristCalibrationRecord:
    """Formal record promoted only after safe-carry G3b passes."""

    bootstrap: BootstrapPublicWristCalibrationRecord
    positive_window: PublicWristPositiveWindow
    safe_carry_calibration_sha256: str
    _admission: object = field(repr=False)
    schema_version: str = PUBLIC_WRIST_CALIBRATION_VERSION
    calibration_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self._admission is not _FORMAL_ADMISSION:
            raise ValueError(
                "formal wrist calibration must use the promotion API"
            )
        if self.schema_version != PUBLIC_WRIST_CALIBRATION_VERSION:
            raise ValueError("unexpected public wrist calibration version")
        if not isinstance(
            self.bootstrap,
            BootstrapPublicWristCalibrationRecord,
        ):
            raise TypeError("bootstrap record is invalid")
        if not isinstance(
            self.positive_window,
            PublicWristPositiveWindow,
        ):
            raise TypeError("public-positive window is invalid")
        if self.positive_window.hand != self.bootstrap.hand:
            raise ValueError("public-positive window hand changed")
        if (
            self.positive_window.bootstrap_calibration_sha256
            != self.bootstrap.calibration_sha256
        ):
            raise ValueError(
                "public-positive window bootstrap hash changed"
            )
        if (
            len(self.positive_window.sample_sha256)
            < self.bootstrap.thresholds.consecutive_frames
        ):
            raise ValueError(
                "public-positive window is shorter than the frozen threshold"
            )
        _sha256(
            self.safe_carry_calibration_sha256,
            "safe-carry calibration",
        )
        canonical = {
            "schema_version": self.schema_version,
            "bootstrap_calibration_sha256": (
                self.bootstrap.calibration_sha256
            ),
            "positive_window_sha256": (
                self.positive_window.window_sha256
            ),
            "safe_carry_calibration_sha256": (
                self.safe_carry_calibration_sha256
            ),
        }
        object.__setattr__(
            self,
            "calibration_sha256",
            hashlib.sha256(
                json.dumps(
                    canonical,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )

    @property
    def hand(self) -> str:
        return self.bootstrap.hand

    @property
    def camera(self) -> str:
        return self.bootstrap.camera

    @property
    def roi_xyxy(self) -> tuple[int, int, int, int]:
        return self.bootstrap.roi_xyxy

    @property
    def empty_rgb(self) -> np.ndarray:
        return self.bootstrap.empty_rgb

    @property
    def empty_depth_m(self) -> np.ndarray:
        return self.bootstrap.empty_depth_m

    @property
    def thresholds(self) -> WristVisualThresholds:
        return self.bootstrap.thresholds

    def to_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": self.schema_version,
                "hand": self.hand,
                "camera": self.camera,
                "bootstrap": dict(self.bootstrap.to_manifest()),
                "positive_window": dict(
                    self.positive_window.to_manifest()
                ),
                "safe_carry_calibration_sha256": (
                    self.safe_carry_calibration_sha256
                ),
                "calibration_sha256": self.calibration_sha256,
            }
        )


def promote_bootstrap_wrist_calibration(
    *,
    bootstrap: BootstrapPublicWristCalibrationRecord,
    positive_window: PublicWristPositiveWindow,
    safe_carry_record: Any,
) -> PublicWristCalibrationRecord:
    """Promote only after the same hand completes strict safe-carry G3b.

    The safe-carry record is inspected only at this evaluator-owned admission
    boundary.  It is reduced to a hash before the formal public provider is
    constructed; private held/contact evidence never enters the provider.
    """

    from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
        SafeCarryCalibrationRecord,
    )

    if not isinstance(
        bootstrap,
        BootstrapPublicWristCalibrationRecord,
    ):
        raise PublicWristEvidenceError(
            "promotion requires a bootstrap wrist calibration"
        )
    if not isinstance(positive_window, PublicWristPositiveWindow):
        raise PublicWristEvidenceError(
            "promotion requires a public-positive window"
        )
    if (
        not isinstance(safe_carry_record, SafeCarryCalibrationRecord)
        or not safe_carry_record.passed
        or not safe_carry_record.ik_only_pass
        or not safe_carry_record.held_continuous
        or not safe_carry_record.grasp_proxy_continuous
        or safe_carry_record.candidate.hand != bootstrap.hand
        or not safe_carry_record.stage_records
        or not all(
            stage.passed
            and stage.samples
            == safe_carry_record.limits.samples_per_stage
            for stage in safe_carry_record.stage_records
        )
    ):
        raise PublicWristEvidenceError(
            "formal wrist calibration requires a passing same-hand G3b "
            "safe-carry record"
        )
    if (
        bootstrap.safe_carry_candidate_sha256
        != safe_carry_candidate_sha256(safe_carry_record.candidate)
    ):
        raise PublicWristEvidenceError(
            "G3b safe-carry candidate differs from the empty-close baseline"
        )
    if (
        positive_window.hand != bootstrap.hand
        or positive_window.bootstrap_calibration_sha256
        != bootstrap.calibration_sha256
        or len(positive_window.sample_sha256)
        < bootstrap.thresholds.consecutive_frames
    ):
        raise PublicWristEvidenceError(
            "public-positive window does not match the bootstrap record"
        )
    return PublicWristCalibrationRecord(
        bootstrap=bootstrap,
        positive_window=positive_window,
        safe_carry_calibration_sha256=safe_carry_record.sha256,
        _admission=_FORMAL_ADMISSION,
    )


def safe_carry_candidate_sha256(candidate: Any) -> str:
    """Return an opaque hash linking empty-close and held G3b pose trials."""

    from behavior_hl.embodied.evaluator_bridge.safe_carry_calibration import (
        SafeCarryCandidate,
    )

    if not isinstance(candidate, SafeCarryCandidate):
        raise PublicWristEvidenceError(
            "candidate must be a SafeCarryCandidate"
        )
    payload = {
        "hand": candidate.hand,
        "arm_target": candidate.arm_target,
        "eef_world_position": candidate.eef_world_position,
        "eef_world_quaternion_xyzw": (
            candidate.eef_world_quaternion_xyzw
        ),
        "source": candidate.source,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(slots=True)
class _VisualState:
    active: bool = False
    sample_index: int = 0
    consecutive: int = 0
    previous_foreground: np.ndarray | None = field(
        default=None,
        repr=False,
    )
    previous_depth_median: float | None = None
    previous_rgb: np.ndarray | None = field(default=None, repr=False)
    recent_sample_sha256: list[str] = field(
        default_factory=list,
        repr=False,
    )
    positive_window: PublicWristPositiveWindow | None = None

    def reset(self, *, active: bool) -> None:
        self.active = active
        self.sample_index = 0
        self.consecutive = 0
        self.previous_foreground = None
        self.previous_depth_median = None
        self.previous_rgb = None
        self.recent_sample_sha256.clear()
        self.positive_window = None


class _BasePublicGripperWristEvidenceProvider:
    """Shared callable implementation for bootstrap and formal providers.

    ``snapshot_provider`` must return the current post-step public observation,
    not an action-start frozen snapshot.  :meth:`begin_grasp` resets all
    temporal state and is invoked automatically by the Starter driver when
    this object itself is passed as ``public_evidence_provider``.
    """

    _record_type: type[Any]

    def __init__(
        self,
        *,
        snapshot_provider: Callable[[], Mapping[str, Any]],
        calibrations: Mapping[str, Any],
    ) -> None:
        if not callable(snapshot_provider):
            raise TypeError("snapshot_provider must be callable")
        if not isinstance(calibrations, Mapping) or not calibrations:
            raise PublicWristEvidenceError(
                "at least one hand calibration is required"
            )
        records: dict[str, Any] = {}
        for hand, record in calibrations.items():
            if hand not in _HAND_CAMERA:
                raise PublicWristEvidenceError(
                    "calibration key must be left or right"
                )
            if (
                not isinstance(record, self._record_type)
                or record.hand != hand
            ):
                raise PublicWristEvidenceError(
                    "calibration key and record hand disagree"
                )
            records[hand] = record
        self._snapshot_provider = snapshot_provider
        self._calibrations = MappingProxyType(records)
        self._states = {
            hand: _VisualState()
            for hand in records
        }

    @property
    def calibration_manifest(self) -> Mapping[str, Mapping[str, Any]]:
        return MappingProxyType(
            {
                hand: dict(record.to_manifest())
                for hand, record in self._calibrations.items()
            }
        )

    def begin_grasp(self, hand: str) -> None:
        """Start a fresh temporal window before any physical grasp motion."""

        state = self._state(hand)
        for other_hand, other_state in self._states.items():
            other_state.reset(active=other_hand == hand)
        state.active = True

    def cancel_grasp(self, hand: str) -> None:
        """Invalidate accumulated visual evidence after an aborted attempt."""

        self._state(hand).reset(active=False)

    def __call__(self, hand: str) -> PublicGripperWristEvidence:
        state = self._state(hand)
        if not state.active:
            raise PublicWristEvidenceError(
                "begin_grasp must be called before sampling wrist evidence"
            )
        snapshot = self._snapshot()
        qpos, qvel = _gripper_proprio(snapshot["proprio"], hand)
        record = self._calibrations[hand]
        camera_payload = snapshot["cameras"][record.camera]
        rgb = _rgb_array(
            camera_payload["rgb"],
            expected_shape=(
                *CAMERA_RESOLUTIONS[record.camera],
                3,
            ),
            name=f"{record.camera}.rgb",
        )
        depth = _depth_array(
            camera_payload["depth_linear"],
            expected_shape=CAMERA_RESOLUTIONS[record.camera],
            name=f"{record.camera}.depth_linear",
        )
        state.sample_index += 1
        visual = _update_visual_state(
            state,
            record,
            rgb=rgb,
            depth_m=depth,
            qpos=qpos,
            qvel=qvel,
            sample_index=state.sample_index,
        )
        return PublicGripperWristEvidence(
            qpos=qpos,
            qvel=qvel,
            wrist_visual_consistent=visual,
        )

    def _state(self, hand: str) -> _VisualState:
        if hand not in _HAND_CAMERA:
            raise PublicWristEvidenceError("hand must be left or right")
        try:
            return self._states[hand]
        except KeyError as exc:
            raise PublicWristEvidenceError(
                f"no admitted public calibration for {hand} hand"
            ) from exc

    def _snapshot(self) -> Mapping[str, Any]:
        value = self._snapshot_provider()
        if not isinstance(value, Mapping):
            raise PublicWristEvidenceError(
                "public snapshot provider must return a mapping"
            )
        unknown = set(value).difference(_PUBLIC_SNAPSHOT_FIELDS)
        missing = {"proprio", "cameras"}.difference(value)
        if missing or unknown:
            raise PublicWristEvidenceError(
                "public snapshot fields mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        cameras = value["cameras"]
        if not isinstance(cameras, Mapping):
            raise PublicWristEvidenceError("cameras must be a mapping")
        if set(cameras) != set(CAMERA_RESOLUTIONS):
            raise PublicWristEvidenceError(
                "public camera roles do not match"
            )
        for role in CAMERA_RESOLUTIONS:
            payload = cameras[role]
            if (
                not isinstance(payload, Mapping)
                or set(payload) != {"rgb", "depth_linear"}
            ):
                raise PublicWristEvidenceError(
                    f"public camera modalities do not match for {role}"
                )
        return value


class BootstrapPublicGripperWristEvidenceProvider(
    _BasePublicGripperWristEvidenceProvider
):
    """Negative-calibrated provider used only inside live calibration."""

    api_contract_version = (
        BOOTSTRAP_PUBLIC_WRIST_EVIDENCE_PROVIDER_VERSION
    )
    _record_type = BootstrapPublicWristCalibrationRecord

    def positive_window(self, hand: str) -> PublicWristPositiveWindow:
        """Return the latest consecutive public-positive evidence window."""

        state = self._state(hand)
        if state.positive_window is None:
            raise PublicWristEvidenceError(
                "no consecutive public-positive window has been observed"
            )
        return state.positive_window


class CalibratedPublicGripperWristEvidenceProvider(
    _BasePublicGripperWristEvidenceProvider
):
    """Formal provider admitted only by promoted G3b calibration records."""

    api_contract_version = PUBLIC_WRIST_EVIDENCE_PROVIDER_VERSION
    _record_type = PublicWristCalibrationRecord


def _update_visual_state(
    state: _VisualState,
    record: (
        BootstrapPublicWristCalibrationRecord
        | PublicWristCalibrationRecord
    ),
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    qpos: tuple[float, float],
    qvel: tuple[float, float],
    sample_index: int,
) -> bool:
    x0, y0, x1, y1 = record.roi_xyxy
    current_rgb = np.array(rgb[y0:y1, x0:x1], copy=True)
    current_depth = np.array(depth_m[y0:y1, x0:x1], copy=True)
    thresholds = record.thresholds

    valid = _valid_depth(current_depth) & _valid_depth(
        record.empty_depth_m
    )
    valid_fraction = float(np.mean(valid))
    foreground = (
        valid
        & (current_depth >= thresholds.minimum_depth_m)
        & (current_depth <= thresholds.maximum_depth_m)
        & (
            record.empty_depth_m - current_depth
            >= thresholds.minimum_foreground_depth_delta_m
        )
    )
    foreground_fraction = float(np.mean(foreground))
    if np.any(foreground):
        depth_median = float(np.median(current_depth[foreground]))
        empty_rgb_delta = float(
            np.mean(
                np.abs(
                    current_rgb[foreground].astype(np.float32)
                    - record.empty_rgb[foreground].astype(np.float32)
                )
            )
        )
    else:
        depth_median = math.nan
        empty_rgb_delta = 0.0
    spatial = (
        valid_fraction >= thresholds.minimum_valid_depth_fraction
        and foreground_fraction >= thresholds.minimum_foreground_fraction
        and empty_rgb_delta >= thresholds.minimum_empty_rgb_delta
        and math.isfinite(depth_median)
        and max(
            abs(qpos[index] - thresholds.empty_close_qpos[index])
            for index in range(2)
        )
        >= thresholds.minimum_qpos_gap
        and max(abs(value) for value in qvel)
        <= thresholds.qvel_epsilon
    )

    temporal = False
    if spatial and state.previous_foreground is not None:
        intersection = foreground & state.previous_foreground
        union = foreground | state.previous_foreground
        iou = (
            float(np.count_nonzero(intersection))
            / float(np.count_nonzero(union))
            if np.any(union)
            else 0.0
        )
        depth_delta = abs(
            depth_median - float(state.previous_depth_median)
        )
        rgb_delta = (
            float(
                np.mean(
                    np.abs(
                        current_rgb[intersection].astype(np.float32)
                        - state.previous_rgb[
                            intersection
                        ].astype(np.float32)
                    )
                )
            )
            if np.any(intersection)
            else math.inf
        )
        temporal = (
            iou >= thresholds.minimum_temporal_foreground_iou
            and depth_delta
            <= thresholds.maximum_temporal_depth_delta_m
            and rgb_delta <= thresholds.maximum_temporal_rgb_delta
        )

    if not spatial:
        state.consecutive = 0
        state.previous_foreground = None
        state.previous_depth_median = None
        state.previous_rgb = None
        state.recent_sample_sha256.clear()
        state.positive_window = None
        return False
    sample_sha256 = _public_sample_sha256(
        rgb=current_rgb,
        depth_m=current_depth,
        qpos=qpos,
        qvel=qvel,
        sample_index=sample_index,
    )
    if temporal:
        state.consecutive += 1
        state.recent_sample_sha256.append(sample_sha256)
    else:
        state.consecutive = 1
        state.recent_sample_sha256 = [sample_sha256]
    state.previous_foreground = np.array(foreground, copy=True)
    state.previous_depth_median = depth_median
    state.previous_rgb = current_rgb
    required = thresholds.consecutive_frames
    if len(state.recent_sample_sha256) > required:
        state.recent_sample_sha256 = state.recent_sample_sha256[-required:]
    visual = (
        state.consecutive >= required
        and len(state.recent_sample_sha256) == required
    )
    if visual:
        bootstrap_sha256 = (
            record.calibration_sha256
            if isinstance(
                record,
                BootstrapPublicWristCalibrationRecord,
            )
            else record.bootstrap.calibration_sha256
        )
        state.positive_window = PublicWristPositiveWindow(
            hand=record.hand,
            bootstrap_calibration_sha256=bootstrap_sha256,
            sample_sha256=tuple(state.recent_sample_sha256),
        )
    return visual


def _gripper_proprio(
    value: Any,
    hand: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (PROPRIO_DIM,) or not np.isfinite(array).all():
        raise PublicWristEvidenceError(
            f"public R1Pro proprio must be finite {PROPRIO_DIM}D"
        )
    qpos_slice, qvel_slice = {
        "left": (LEFT_GRIPPER_QPOS, LEFT_GRIPPER_QVEL),
        "right": (RIGHT_GRIPPER_QPOS, RIGHT_GRIPPER_QVEL),
    }[hand]
    return (
        tuple(float(item) for item in array[qpos_slice]),
        tuple(float(item) for item in array[qvel_slice]),
    )


def _roi(
    value: Any,
    camera: str,
) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 4
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in value
        )
    ):
        raise ValueError("roi_xyxy must contain four integers")
    x0, y0, x1, y1 = map(int, value)
    height, width = CAMERA_RESOLUTIONS[camera]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("roi_xyxy is outside the native wrist frame")
    return x0, y0, x1, y1


def _rgb_array(
    value: Any,
    *,
    expected_shape: tuple[int, int, int],
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape or array.dtype != np.uint8:
        raise PublicWristEvidenceError(
            f"{name} must be uint8 with shape {expected_shape}"
        )
    return np.array(array, copy=True)


def _depth_array(
    value: Any,
    *,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (*expected_shape, 1):
        array = array[..., 0]
    if array.shape != expected_shape or array.dtype.kind not in {"f"}:
        raise PublicWristEvidenceError(
            f"{name} must be floating depth with shape {expected_shape}"
        )
    if np.isinf(array).any():
        raise PublicWristEvidenceError(f"{name} contains infinite depth")
    return np.array(array, dtype=np.float32, copy=True)


def _valid_depth(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0.0)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _public_sample_sha256(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    qpos: tuple[float, float],
    qvel: tuple[float, float],
    sample_index: int,
) -> str:
    """Hash one provider-cadence sample without any private field."""

    if (
        isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or sample_index <= 0
    ):
        raise PublicWristEvidenceError(
            "public sample index must be positive"
        )
    payload = {
        "sample_index": sample_index,
        "rgb_sha256": _array_sha256(rgb),
        "depth_sha256": _array_sha256(depth_m),
        "qpos": qpos,
        "qvel": qvel,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} sha256 must be lowercase hexadecimal")
    return value


def _read_only(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, copy=True)
    copied.setflags(write=False)
    return copied


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain two finite values")
    return (
        _finite_number(value[0], f"{name}[0]"),
        _finite_number(value[1], f"{name}[1]"),
    )


def _finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _finite_nonnegative(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _fraction(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _positive_fraction(value: Any, name: str) -> float:
    result = _fraction(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return result
