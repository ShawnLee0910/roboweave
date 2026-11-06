"""Strict capability gate for hard-trajectory scene grounding.

This module does not import OmniGibson.  A simulator integration supplies
snapshot, rollback, intervention, settle, and oracle callbacks.  Missing
visual or physics capabilities fail closed before a probe is exposed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from behavior_hl.hard_trajectories import (
    DiagnosticProbe,
    EvidenceTier,
    HardTrajectorySpec,
    InterventionAdapterKey,
    OracleMode,
    SceneIntervention,
    TrajectoryPhase,
)


class HardGroundingError(RuntimeError):
    """Base class for strict hard-trajectory grounding failures."""


class MissingGroundingCapability(HardGroundingError):
    """A required evidence tier, intervention adapter, or oracle is absent."""


class InterventionRejected(HardGroundingError):
    """A deterministic intervention failed validation and must be rolled back."""


@dataclass(frozen=True, slots=True)
class InterventionOutcome:
    """Validated intervention result and its evaluator-owned simulator cost."""

    accepted: bool = True
    simulator_steps: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool")
        if (
            not isinstance(self.simulator_steps, int)
            or isinstance(self.simulator_steps, bool)
            or self.simulator_steps < 0
        ):
            raise ValueError("simulator_steps must be a non-negative integer")


InterventionHandler = Callable[
    [SceneIntervention], InterventionOutcome | bool | None
]
OracleResolver = Callable[[TrajectoryPhase, DiagnosticProbe], str]
PublicEvidenceResolver = Callable[
    [TrajectoryPhase, DiagnosticProbe], Mapping[str, Any]
]
SnapshotHandler = Callable[[], object]
CommitHandler = Callable[[object], None]
RollbackHandler = Callable[[object], None]
SettleHandler = Callable[[int], int | None]


@dataclass(frozen=True, slots=True)
class PhaseCapability:
    """Evaluator-side readiness for one phase.

    Adapter keys and resolver identifiers stay private.  ``to_public_dict``
    exposes only generic missing-capability classes.
    """

    phase_id: str
    runnable: bool
    missing_tiers: tuple[EvidenceTier, ...] = ()
    missing_intervention_adapter: bool = False
    missing_oracle_resolver: bool = False
    missing_public_evidence: bool = False
    final_success: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        missing: list[str] = []
        if self.missing_tiers:
            missing.append("evidence")
        if self.missing_intervention_adapter:
            missing.append("scene_intervention")
        if self.missing_oracle_resolver:
            missing.append("ground_truth_resolver")
        if self.missing_public_evidence:
            missing.append("public_evidence")
        return {
            "phase_id": self.phase_id,
            "runnable": self.runnable,
            "final_success": self.final_success,
            "missing_capability_classes": missing,
            "missing_evidence_tiers": [tier.value for tier in self.missing_tiers],
        }


@dataclass(frozen=True, slots=True)
class TrajectoryCapabilityReport:
    task_name: str
    variant: str
    phases: tuple[PhaseCapability, ...]

    @property
    def fully_runnable(self) -> bool:
        return all(phase.runnable for phase in self.phases)

    @property
    def runnable_probe_count(self) -> int:
        return sum(phase.runnable and not phase.final_success for phase in self.phases)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "variant": self.variant,
            "fully_runnable": self.fully_runnable,
            "runnable_probe_count": self.runnable_probe_count,
            "phases": [phase.to_public_dict() for phase in self.phases],
        }


def probe_requires_runtime_resolver(phase: TrajectoryPhase) -> bool:
    """Return whether actual episode state must determine or verify the answer."""

    probe = phase.probe
    if probe is None:
        return False
    return (
        probe.oracle.mode is OracleMode.RUNTIME_GROUNDED
        or phase.intervention is not None
    )


def audit_trajectory_capabilities(
    trajectory: HardTrajectorySpec,
    *,
    supported_tiers: frozenset[EvidenceTier],
    intervention_adapters: frozenset[InterventionAdapterKey],
    oracle_resolvers: frozenset[str],
    public_evidence_resolvers: frozenset[str] = frozenset(),
) -> TrajectoryCapabilityReport:
    """Compute fail-closed readiness without launching a simulator."""

    reports: list[PhaseCapability] = []
    for phase in trajectory.phases:
        if phase.final_success:
            reports.append(
                PhaseCapability(
                    phase_id=phase.id,
                    runnable=True,
                    final_success=True,
                )
            )
            continue

        probe = phase.probe
        if probe is None:
            raise ValueError(f"non-final phase {phase.id!r} has no probe")
        missing_tiers = tuple(
            sorted(probe.evidence_tiers - supported_tiers, key=lambda tier: tier.value)
        )
        missing_intervention = (
            phase.intervention is not None
            and phase.intervention.adapter_key not in intervention_adapters
        )
        missing_resolver = (
            probe_requires_runtime_resolver(phase)
            and probe.id not in oracle_resolvers
        )
        missing_public_evidence = probe.id not in public_evidence_resolvers
        reports.append(
            PhaseCapability(
                phase_id=phase.id,
                runnable=not (
                    missing_tiers
                    or missing_intervention
                    or missing_resolver
                    or missing_public_evidence
                ),
                missing_tiers=missing_tiers,
                missing_intervention_adapter=missing_intervention,
                missing_oracle_resolver=missing_resolver,
                missing_public_evidence=missing_public_evidence,
            )
        )
    return TrajectoryCapabilityReport(
        task_name=trajectory.task_name,
        variant=trajectory.variant,
        phases=tuple(reports),
    )


class StrictHardGrounder:
    """Callback-backed grounder used by the hybrid episode controller.

    Deterministic questions without a scene intervention may use their
    manifest expectation.  Every runtime outcome and every intervention must
    be resolved from the frozen simulator state.  If a deterministic
    intervention produces the wrong option, grounding fails so the controller
    can roll back the snapshot before exposing the question.
    """

    def __init__(
        self,
        *,
        supported_tiers: frozenset[EvidenceTier],
        intervention_handlers: Mapping[InterventionAdapterKey, InterventionHandler],
        oracle_resolvers: Mapping[str, OracleResolver],
        snapshot_handler: SnapshotHandler,
        rollback_handler: RollbackHandler,
        settle_handler: SettleHandler,
        public_evidence_resolvers: Mapping[str, PublicEvidenceResolver] | None = None,
        commit_handler: CommitHandler | None = None,
    ) -> None:
        if public_evidence_resolvers is None:
            public_evidence_resolvers = {}
        if any(not isinstance(tier, EvidenceTier) for tier in supported_tiers):
            raise TypeError("supported_tiers must contain EvidenceTier values")
        if any(not isinstance(key, InterventionAdapterKey) for key in intervention_handlers):
            raise TypeError("intervention handler keys must be InterventionAdapterKey values")
        if any(not isinstance(key, str) or not key for key in oracle_resolvers):
            raise TypeError("oracle resolver keys must be non-empty probe ids")
        if not isinstance(public_evidence_resolvers, Mapping):
            raise TypeError("public_evidence_resolvers must be a mapping")
        if any(
            not isinstance(key, str) or not key.strip()
            for key in public_evidence_resolvers
        ):
            raise TypeError(
                "public evidence resolver keys must be non-empty probe ids"
            )
        for label, handler in (
            ("snapshot_handler", snapshot_handler),
            ("rollback_handler", rollback_handler),
            ("settle_handler", settle_handler),
        ):
            if not callable(handler):
                raise TypeError(f"{label} must be callable")
        if commit_handler is not None and not callable(commit_handler):
            raise TypeError("commit_handler must be callable or None")
        if any(not callable(handler) for handler in intervention_handlers.values()):
            raise TypeError("every intervention handler must be callable")
        if any(not callable(resolver) for resolver in oracle_resolvers.values()):
            raise TypeError("every oracle resolver must be callable")
        if any(
            not callable(resolver)
            for resolver in public_evidence_resolvers.values()
        ):
            raise TypeError("every public evidence resolver must be callable")

        self.supported_tiers = supported_tiers
        self.intervention_handlers = dict(intervention_handlers)
        self.oracle_resolvers = dict(oracle_resolvers)
        self.public_evidence_resolvers = dict(public_evidence_resolvers)
        self._snapshot_handler = snapshot_handler
        # Some snapshot providers need no success-path release.  Keeping an
        # explicit no-op preserves the old constructor while making commit()
        # available to every transaction controller.
        self._commit_handler = _noop_commit if commit_handler is None else commit_handler
        self._rollback_handler = rollback_handler
        self._settle_handler = settle_handler

    def capability_report(self, trajectory: HardTrajectorySpec) -> TrajectoryCapabilityReport:
        return audit_trajectory_capabilities(
            trajectory,
            supported_tiers=self.supported_tiers,
            intervention_adapters=frozenset(self.intervention_handlers),
            oracle_resolvers=frozenset(self.oracle_resolvers),
            public_evidence_resolvers=frozenset(self.public_evidence_resolvers),
        )

    def require_trajectory(self, trajectory: HardTrajectorySpec) -> None:
        report = self.capability_report(trajectory)
        if not report.fully_runnable:
            blocked = [phase.phase_id for phase in report.phases if not phase.runnable]
            raise MissingGroundingCapability(
                f"{trajectory.task_name}/{trajectory.variant} is missing capabilities "
                f"for {len(blocked)} phase(s): {blocked}"
            )

    def snapshot(self) -> object:
        return self._snapshot_handler()

    def apply_intervention(self, intervention: SceneIntervention | None) -> int:
        if intervention is None:
            return 0
        self._require_tiers(intervention.required_tiers, context=intervention.id)
        try:
            handler = self.intervention_handlers[intervention.adapter_key]
        except KeyError as exc:
            raise MissingGroundingCapability(
                f"checkpoint intervention {intervention.id!r} has no registered adapter"
            ) from exc
        result = handler(intervention)
        if isinstance(result, InterventionOutcome):
            accepted = result.accepted
            simulator_steps = result.simulator_steps
        elif isinstance(result, bool):
            accepted = result
            simulator_steps = 0
        elif result is None:
            accepted = True
            simulator_steps = 0
        else:
            raise TypeError(
                "intervention handler must return InterventionOutcome, bool, or None"
            )
        if not accepted:
            raise InterventionRejected(
                f"checkpoint intervention {intervention.id!r} failed its validity checks"
            )
        return simulator_steps

    def settle(self, steps: int = 0) -> int:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("settle steps must be a non-negative integer")
        consumed = self._settle_handler(steps)
        if consumed is None:
            return steps
        if (
            not isinstance(consumed, int)
            or isinstance(consumed, bool)
            or consumed < 0
        ):
            raise ValueError(
                "settle handler must return a non-negative simulator step count or None"
            )
        return consumed

    def freeze_ground_truth(self, phase: TrajectoryPhase) -> Mapping[str, str]:
        probe = phase.probe
        if probe is None:
            return {}
        self._require_tiers(probe.evidence_tiers, context=probe.id)

        resolved: str | None = None
        if probe_requires_runtime_resolver(phase):
            try:
                resolver = self.oracle_resolvers[probe.id]
            except KeyError as exc:
                raise MissingGroundingCapability(
                    f"probe {probe.id!r} has no registered ground-truth resolver"
                ) from exc
            resolved = resolver(phase, probe)

        frozen = probe.freeze_ground_truth(resolved)
        if (
            phase.intervention is not None
            and frozen.mode is OracleMode.DETERMINISTIC_EXPECTED
            and frozen.matches_manifest_expectation is not True
        ):
            raise InterventionRejected(
                f"intervention for probe {probe.id!r} did not create its declared truth"
            )
        return {probe.id: frozen.option_id}

    def freeze_public_evidence(
        self,
        phase: TrajectoryPhase,
    ) -> Mapping[str, Mapping[str, Any]]:
        """Freeze a detached, policy-safe sensor bundle for one probe."""

        probe = phase.probe
        if probe is None:
            return {}
        self._require_tiers(probe.evidence_tiers, context=probe.id)
        try:
            resolver = self.public_evidence_resolvers[probe.id]
        except KeyError as exc:
            raise MissingGroundingCapability(
                f"probe {probe.id!r} has no registered public-evidence resolver"
            ) from exc

        bundle = _validate_public_evidence_bundle(
            resolver(phase, probe),
            probe_id=probe.id,
        )
        return {probe.id: bundle}

    def commit(self, snapshot: object) -> None:
        """Release a successfully committed snapshot without restoring it."""

        self._commit_handler(snapshot)

    def rollback(self, snapshot: object) -> None:
        self._rollback_handler(snapshot)

    def _require_tiers(
        self,
        required: frozenset[EvidenceTier],
        *,
        context: str,
    ) -> None:
        missing = required - self.supported_tiers
        if missing:
            names = sorted(tier.value for tier in missing)
            raise MissingGroundingCapability(
                f"{context!r} requires unavailable evidence tiers: {names}"
            )


def _noop_commit(_snapshot: object) -> None:
    """Default for snapshot providers that do not allocate release handles."""


_EVALUATOR_ONLY_EVIDENCE_KEY_PARTS = (
    "oracle",
    "answer",
    "expected",
    "correct",
    "q_score",
    "qscore",
    "goal_status",
    "goalstatus",
    "predicate",
    "ground_truth",
    "groundtruth",
    "deterministic_seed",
    "deterministicseed",
    "success",
    "bddl",
    "inside",
    "ontop",
    "nextto",
    "touching",
    "toggled_on",
    "toggledon",
    "open_state",
    "openstate",
    "contact",
    "object_state",
    "objectstate",
    "reward",
    "score",
)


_PUBLIC_SENSOR_MODALITIES = frozenset(
    {
        "rgb",
        "depth",
        "instance_mask",
        "instance_segmentation",
        "proprio",
        "proprioception",
    }
)


def _validate_public_evidence_bundle(
    value: Any,
    *,
    probe_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"public evidence resolver for {probe_id!r} must return a mapping"
        )
    bundle = _json_safe_evidence_copy(value)

    if set(bundle) != {"capture_id", "modalities"}:
        raise ValueError(
            f"public evidence for {probe_id!r} must contain exactly capture_id and modalities"
        )

    capture_id = bundle["capture_id"]
    if not isinstance(capture_id, str) or not capture_id.strip():
        raise ValueError(
            f"public evidence for {probe_id!r} requires a non-empty capture_id"
        )

    modalities = bundle["modalities"]
    if not isinstance(modalities, Mapping) or not modalities:
        raise ValueError(
            f"public evidence for {probe_id!r} requires non-empty modalities"
        )

    seen_modalities: set[str] = set()
    for name, handles in modalities.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("public evidence modality names must be non-empty")
        normalized = _normalize_evidence_key(name)
        if normalized not in _PUBLIC_SENSOR_MODALITIES:
            raise ValueError(
                f"unsupported public evidence modality for {probe_id!r}: {name!r}"
            )
        if normalized in seen_modalities:
            raise ValueError("public evidence modality names must be unique")
        seen_modalities.add(normalized)
        _validate_sensor_handles(handles, modality=normalized)
    return bundle


def _validate_sensor_handles(
    handles: Any,
    *,
    modality: str,
) -> None:
    if isinstance(handles, str):
        if not handles.strip():
            raise ValueError(
                f"public evidence handle for {modality!r} must be non-empty"
            )
        return
    if isinstance(handles, list) and handles:
        if all(isinstance(handle, str) and handle.strip() for handle in handles):
            return
    raise ValueError(
        f"public evidence for {modality!r} must use opaque string handles"
    )


def _normalize_evidence_key(key: str) -> str:
    return (
        key.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _json_safe_evidence_copy(value: Any) -> Any:
    """Return a detached JSON value while rejecting evaluator-private keys."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("public evidence numbers must be finite")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("public evidence object keys must be strings")
            normalized = _normalize_evidence_key(key)
            if normalized.startswith("hidden") or any(
                part in normalized
                for part in _EVALUATOR_ONLY_EVIDENCE_KEY_PARTS
            ):
                raise ValueError(
                    f"evaluator-only public evidence key is forbidden: {key!r}"
                )
            copied[key] = _json_safe_evidence_copy(item)
        return copied
    if isinstance(value, (list, tuple)):
        return [_json_safe_evidence_copy(item) for item in value]
    raise ValueError(
        f"public evidence value of type {type(value).__name__} is not JSON-safe"
    )
