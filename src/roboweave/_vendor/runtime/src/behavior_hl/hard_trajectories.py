"""Declarative hard-v1 trajectories with evaluator-only ESI-style probes.

The manifests in this module do not replace the official BDDL goals.  Every
trajectory ends with the complete canonical BDDL goal from :mod:`behavior_hl.tasks`.
Probe answers, verifiers, and scene-mutation operations are evaluator-only and
are deliberately excluded from all public serialization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from behavior_hl.actions import ActionName, HighLevelAction
from behavior_hl.tasks import FIVE_TASKS, get_task_spec


class ManifestValidationError(ValueError):
    """Raised when a hard-trajectory manifest is internally inconsistent."""


class ProbeFamily(str, Enum):
    ACTION_ORDER_INFERENCE = "Action Order Inference"
    PHYSICAL_CONTACT = "Physical Contact"
    PARTIAL_OCCLUSION = "Partial Occlusion"
    VIEW_HALLUCINATION = "View Hallucination"
    UNOBSERVED_CHANGE = "Unobserved Change"
    LONG_TERM_NAVIGATION = "Long-Term Navigation"
    TRAVERSABLE_PASSAGE = "Traversable Passage"
    RIGID_CONTAINMENT = "Rigid Containment"
    STRUCTURAL_ENCLOSURE = "Structural Enclosure"
    STACKING_AND_STABILITY = "Stacking & Stability"


class EvidenceTier(str, Enum):
    """Where an evaluator obtains ground truth for a probe."""

    SYMBOLIC = "symbolic_state"
    VISUAL = "visual_observation"
    PHYSICS = "physics_state"


class TriggerKind(str, Enum):
    EPISODE_START = "episode_start"
    BEFORE_ACTION = "before_action"
    AFTER_ACTION = "after_action"
    AFTER_REVISIT = "after_revisit"
    BEFORE_FINAL_SUCCESS = "before_final_success"
    FINAL_SUCCESS_CHECK = "final_success_check"


class RevisitAnchorKind(str, Enum):
    """Evaluator-only ways to bind a concrete spatial revisit anchor."""

    INSTANCE = "instance"
    ROLE = "role"


@dataclass(frozen=True, slots=True)
class RevisitAnchor:
    """A typed private selector resolved and frozen when a revisit is armed."""

    kind: RevisitAnchorKind
    value: str

    @property
    def selector(self) -> str:
        return f"{self.kind.value}:{self.value}"


@dataclass(frozen=True, slots=True)
class RevisitTriggerSpec:
    """Private geometric and temporal criteria for an evaluator revisit event.

    The evaluator must observe the robot leave ``leave_radius`` around the
    frozen anchor, remain absent for ``min_absence_steps``, and subsequently
    enter ``reenter_radius``. The smaller re-entry radius provides hysteresis.
    """

    anchor: RevisitAnchor
    leave_radius: float
    reenter_radius: float
    min_absence_steps: int


class InterventionStatus(str, Enum):
    """Execution readiness of a scene intervention.

    ``DECLARED_REQUIRES_ADAPTER`` is intentionally explicit: the mutation is a
    deterministic benchmark design, but an OmniGibson adapter must implement
    and verify it before a physics rollout may claim that it occurred.
    """

    OBSERVATION_ONLY = "observation_only"
    RUNTIME_NATIVE = "runtime_native"
    DECLARED_REQUIRES_ADAPTER = "declared_requires_simulator_adapter"


class InterventionAdapterKey(str, Enum):
    """Stable capability keys consumed by a simulator intervention adapter."""

    OCCLUDE_WITH_DISTRACTOR = "occlude_with_distractor"
    MISLEADING_VIEW = "misleading_view"
    OBSERVE_RUNTIME = "observe_runtime"
    RELOCATE_ON_SUPPORT = "relocate_on_support"
    NARROW_PASSAGE = "narrow_passage"
    CONTAINMENT_NEAR_MISS = "containment_near_miss"
    DISTURB_RELATION = "disturb_relation"
    SETTLE_ONLY = "settle_only"
    THRESHOLD_NEAR_MISS = "threshold_near_miss"
    CARRIED_ORIENTATION = "carried_orientation"


_COMMON_INTERVENTION_PARAMETER_SCHEMA: dict[str, tuple[type, ...]] = {
    "target_selector": (str,),
    "postcondition": (str,),
}


def _adapter_schema(**specific: tuple[type, ...]) -> dict[str, tuple[type, ...]]:
    return {**_COMMON_INTERVENTION_PARAMETER_SCHEMA, **specific}


INTERVENTION_PARAMETER_SCHEMAS: dict[
    InterventionAdapterKey, dict[str, tuple[type, ...]]
] = {
    InterventionAdapterKey.OCCLUDE_WITH_DISTRACTOR: _adapter_schema(
        occluder_selector=(str,),
        initial_view_id=(str,),
        verification_view_id=(str,),
        min_visibility_fraction=(float,),
        max_visibility_fraction=(float,),
    ),
    InterventionAdapterKey.MISLEADING_VIEW: _adapter_schema(
        initial_view_id=(str,),
        verification_view_id=(str,),
        min_projected_overlap_fraction=(float,),
    ),
    InterventionAdapterKey.OBSERVE_RUNTIME: _adapter_schema(
        measurement_name=(str,),
        window_steps=(int,),
    ),
    InterventionAdapterKey.RELOCATE_ON_SUPPORT: _adapter_schema(
        support_selector=(str,),
        displacement_m=(float,),
        settle_steps=(int,),
    ),
    InterventionAdapterKey.NARROW_PASSAGE: _adapter_schema(
        passage_selector=(str,),
        blocker_selector=(str,),
        min_clearance_m=(float,),
        alternate_route_selector=(str,),
    ),
    InterventionAdapterKey.CONTAINMENT_NEAR_MISS: _adapter_schema(
        container_selector=(str,),
        protrusion_m=(float,),
    ),
    InterventionAdapterKey.DISTURB_RELATION: _adapter_schema(
        relation=(str,),
        impulse_ns=(float,),
        displacement_m=(float,),
    ),
    InterventionAdapterKey.SETTLE_ONLY: _adapter_schema(
        measurement_name=(str,),
        window_steps=(int,),
        linear_velocity_threshold_mps=(float,),
        angular_velocity_threshold_rps=(float,),
    ),
    InterventionAdapterKey.THRESHOLD_NEAR_MISS: _adapter_schema(
        support_selector=(str,),
        relation=(str,),
        displacement_m=(float,),
    ),
    InterventionAdapterKey.CARRIED_ORIENTATION: _adapter_schema(
        passage_selector=(str,),
        initial_orientation=(str,),
        required_orientation=(str,),
        carried_clearance_m=(float,),
    ),
}


class OracleMode(str, Enum):
    """How the evaluator obtains the answer for one concrete checkpoint."""

    DETERMINISTIC_EXPECTED = "deterministic_expected"
    RUNTIME_GROUNDED = "runtime_grounded"


@dataclass(frozen=True, slots=True)
class ESIFamilyMapping:
    big_task: str
    small_task: str
    runner_task: str
    adaptation: str


ESI_FAMILY_MAPPINGS: dict[ProbeFamily, ESIFamilyMapping] = {
    ProbeFamily.ACTION_ORDER_INFERENCE: ESIFamilyMapping(
        big_task="Action Sequencing",
        small_task="Action Order Inference",
        runner_task="action",
        adaptation="BEHAVIOR adaptation: dependency-aware macro-action ordering, not only the original ESI stacking relation.",
    ),
    ProbeFamily.PHYSICAL_CONTACT: ESIFamilyMapping(
        big_task="Spatial Relations",
        small_task="Physical Contact",
        runner_task="touching",
        adaptation="BEHAVIOR adaptation: contact and attachment are grounded in the current robot rollout.",
    ),
    ProbeFamily.PARTIAL_OCCLUSION: ESIFamilyMapping(
        big_task="Perceptual Grounding",
        small_task="Partial Occlusion",
        runner_task="occlusion",
        adaptation="BEHAVIOR adaptation: active multi-view identification of a BDDL-scoped object.",
    ),
    ProbeFamily.VIEW_HALLUCINATION: ESIFamilyMapping(
        big_task="Perceptual Grounding",
        small_task="View Hallucination",
        runner_task="angle_confusion",
        adaptation="BEHAVIOR adaptation: a second view checks a task relation, not only object category.",
    ),
    ProbeFamily.UNOBSERVED_CHANGE: ESIFamilyMapping(
        big_task="Temporal Understanding",
        small_task="Unobserved Change",
        runner_task="unobserved_changes",
        adaptation="BEHAVIOR adaptation: before/after task-state snapshots ground the changed relation.",
    ),
    ProbeFamily.LONG_TERM_NAVIGATION: ESIFamilyMapping(
        big_task="Cognitive Mapping",
        small_task="Long-Term Navigation",
        runner_task="cognitivemap",
        adaptation="BEHAVIOR adaptation: route choice is conditioned on long-horizon task progress.",
    ),
    ProbeFamily.TRAVERSABLE_PASSAGE: ESIFamilyMapping(
        big_task="Cognitive Mapping",
        small_task="Traversable Passage",
        runner_task="cognitivemap",
        adaptation="BEHAVIOR adaptation: passage truth includes the robot, held object, and dynamic obstacle.",
    ),
    ProbeFamily.RIGID_CONTAINMENT: ESIFamilyMapping(
        big_task="Physical Structure",
        small_task="Rigid Containment",
        runner_task="storage",
        adaptation="BEHAVIOR adaptation: official Inside is reported separately from stricter full-volume containment.",
    ),
    ProbeFamily.STRUCTURAL_ENCLOSURE: ESIFamilyMapping(
        big_task="Enumerative Perception",
        small_task="Structural Enclosure",
        runner_task="counting",
        adaptation="BEHAVIOR adaptation: opening a closed structure reveals countable storage regions or objects.",
    ),
    ProbeFamily.STACKING_AND_STABILITY: ESIFamilyMapping(
        big_task="Physical Dynamics",
        small_task="Stacking & Stability",
        runner_task="stacking",
        adaptation="BEHAVIOR adaptation: support is checked after a physics settle window.",
    ),
}


def _matches_instance_constraint(
    value: str | None,
    instances: tuple[str, ...],
    prefixes: tuple[str, ...],
) -> bool:
    return value is not None and (value in instances or any(value.startswith(prefix) for prefix in prefixes))


@dataclass(frozen=True, slots=True)
class ProbeTrigger:
    kind: TriggerKind
    action: ActionName | None = None
    occurrence: int = 1
    settle_steps: int = 0
    target_instances: tuple[str, ...] = ()
    target_prefixes: tuple[str, ...] = ()
    support_instances: tuple[str, ...] = ()
    support_prefixes: tuple[str, ...] = ()
    evaluator_revisit: RevisitTriggerSpec | None = None

    def matches_action(self, candidate: HighLevelAction) -> bool:
        """Match action name and BDDL role; occurrence is tracked by the caller."""

        if not isinstance(candidate, HighLevelAction):
            raise TypeError("candidate must be HighLevelAction")
        if self.kind not in {TriggerKind.BEFORE_ACTION, TriggerKind.AFTER_ACTION}:
            return False
        if candidate.name is not self.action:
            return False
        if not _matches_instance_constraint(
            candidate.target,
            self.target_instances,
            self.target_prefixes,
        ):
            return False
        if self.support_instances or self.support_prefixes:
            return _matches_instance_constraint(
                candidate.support,
                self.support_instances,
                self.support_prefixes,
            )
        return True

    def to_public_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind.value,
            "occurrence": self.occurrence,
            "settle_steps": self.settle_steps,
        }
        if self.action is not None:
            result["action"] = self.action.value
            result["target_instances"] = list(self.target_instances)
            result["target_prefixes"] = list(self.target_prefixes)
            if self.support_instances or self.support_prefixes:
                result["support_instances"] = list(self.support_instances)
                result["support_prefixes"] = list(self.support_prefixes)
        return result


@dataclass(frozen=True, slots=True)
class SceneIntervention:
    """A checkpoint intervention.

    Fields prefixed with ``hidden_`` are evaluator-only.  They may identify the
    changed object or recovery operation and must never be exposed to a policy.
    """

    id: str
    status: InterventionStatus
    public_notice: str
    required_tiers: frozenset[EvidenceTier]
    adapter_key: InterventionAdapterKey
    hidden_operation: str
    hidden_recovery: str
    hidden_parameters: tuple[tuple[str, str | int | float | bool], ...] = ()
    deterministic_seed: int | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "notice": self.public_notice,
            "required_evidence": sorted(tier.value for tier in self.required_tiers),
        }


@dataclass(frozen=True, slots=True)
class ProbeOption:
    id: str
    text: str

    def to_public_dict(self) -> dict[str, str]:
        return {"id": self.id, "text": self.text}


@dataclass(frozen=True, slots=True)
class ProbeOracle:
    """Evaluator-only answer and verification recipe."""

    mode: OracleMode
    expected_option_id: str | None
    source: str
    verifier: str
    resolver: str


@dataclass(frozen=True, slots=True)
class FrozenProbeGroundTruth:
    """Ground truth frozen for one concrete episode checkpoint."""

    probe_id: str
    option_id: str
    mode: OracleMode
    expected_option_id: str | None
    matches_manifest_expectation: bool | None


@dataclass(frozen=True, slots=True)
class DiagnosticProbe:
    id: str
    family: ProbeFamily
    prompt: str
    options: tuple[ProbeOption, ...]
    evidence_tiers: frozenset[EvidenceTier]
    oracle: ProbeOracle

    def freeze_ground_truth(self, resolved_option_id: str | None = None) -> FrozenProbeGroundTruth:
        """Freeze actual checkpoint truth without exposing it in public output.

        Runtime-grounded probes require a result from their resolver. A
        deterministic probe uses its expected answer by default, while an
        adapter may supply an observed result to reveal a failed intervention.
        """

        option_ids = {option.id for option in self.options}
        if self.oracle.mode is OracleMode.RUNTIME_GROUNDED:
            if resolved_option_id is None:
                raise ManifestValidationError(
                    f"{self.id}: runtime-grounded oracle requires a resolved option id"
                )
            selected = resolved_option_id
            expectation_match: bool | None = None
        else:
            selected = resolved_option_id or self.oracle.expected_option_id
            expectation_match = selected == self.oracle.expected_option_id
        if selected not in option_ids:
            raise ManifestValidationError(
                f"{self.id}: resolved oracle answer {selected!r} is not a public option id"
            )
        return FrozenProbeGroundTruth(
            probe_id=self.id,
            option_id=selected,
            mode=self.oracle.mode,
            expected_option_id=self.oracle.expected_option_id,
            matches_manifest_expectation=expectation_match,
        )

    def to_public_dict(self) -> dict[str, Any]:
        esi = ESI_FAMILY_MAPPINGS[self.family]
        return {
            "id": self.id,
            "family": self.family.value,
            "big_task": esi.big_task,
            "small_task": esi.small_task,
            "runner_task": esi.runner_task,
            "adaptation": esi.adaptation,
            "prompt": self.prompt,
            "options": [option.to_public_dict() for option in self.options],
            "evidence_tiers": sorted(tier.value for tier in self.evidence_tiers),
        }


@dataclass(frozen=True, slots=True)
class TrajectoryPhase:
    id: str
    ordinal: int
    objective: str
    trigger: ProbeTrigger
    probe: DiagnosticProbe | None
    intervention: SceneIntervention | None = None
    hidden_recovery: str = ""
    final_success: bool = False
    required_goal_fragments: tuple[str, ...] = ()
    required_canonical_goal: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "ordinal": self.ordinal,
            "objective": self.objective,
            "trigger": self.trigger.to_public_dict(),
            "final_success": self.final_success,
        }
        if self.probe is not None:
            result["probe"] = self.probe.to_public_dict()
        if self.intervention is not None:
            result["intervention"] = self.intervention.to_public_dict()
        if self.final_success:
            if self.required_goal_fragments:
                result["goal_leaf_templates"] = list(self.required_goal_fragments)
            if self.required_canonical_goal is not None:
                result["required_canonical_goal"] = self.required_canonical_goal
        return result


@dataclass(frozen=True, slots=True)
class HardTrajectorySpec:
    task_name: str
    variant: str
    title: str
    goal_leaf_templates: tuple[str, ...]
    phases: tuple[TrajectoryPhase, ...]
    official_canonical_goal: str = ""

    @property
    def checkpoint_count(self) -> int:
        return sum(phase.probe is not None for phase in self.phases)

    @property
    def families(self) -> frozenset[ProbeFamily]:
        return frozenset(
            phase.probe.family for phase in self.phases if phase.probe is not None
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "variant": self.variant,
            "title": self.title,
            "checkpoint_count": self.checkpoint_count,
            "families": sorted(family.value for family in self.families),
            "goal_leaf_templates": list(self.goal_leaf_templates),
            "official_canonical_goal": self.official_canonical_goal,
            "phases": [phase.to_public_dict() for phase in self.phases],
        }


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_ACTION_TRIGGERS = {TriggerKind.BEFORE_ACTION, TriggerKind.AFTER_ACTION}


def _fail(message: str) -> None:
    raise ManifestValidationError(message)


def _validate_id(value: str, label: str) -> None:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        _fail(f"{label} must match {_ID_PATTERN.pattern!r}, got {value!r}")


def _validate_constraint_values(
    instances: tuple[str, ...],
    prefixes: tuple[str, ...],
    *,
    task_name: str,
    label: str,
) -> None:
    if not isinstance(instances, tuple) or not isinstance(prefixes, tuple):
        _fail(f"{task_name}: {label} constraints must be tuples")
    for collection, kind in ((instances, "instance"), (prefixes, "prefix")):
        if len(collection) != len(set(collection)):
            _fail(f"{task_name}: duplicate {label} {kind} constraints")
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or any(char.isspace() for char in value)
            or value.startswith("?")
            for value in collection
        ):
            _fail(f"{task_name}: {label} {kind} constraints must be canonical runtime BDDL ids")


def _validate_trigger(trigger: ProbeTrigger, *, final: bool, task_name: str) -> None:
    if not isinstance(trigger.kind, TriggerKind):
        _fail(f"{task_name}: trigger kind must be TriggerKind")
    if not isinstance(trigger.occurrence, int) or isinstance(trigger.occurrence, bool) or trigger.occurrence < 1:
        _fail(f"{task_name}: trigger occurrence must be a positive integer")
    if not isinstance(trigger.settle_steps, int) or isinstance(trigger.settle_steps, bool) or trigger.settle_steps < 0:
        _fail(f"{task_name}: settle_steps must be a non-negative integer")
    _validate_constraint_values(
        trigger.target_instances, trigger.target_prefixes, task_name=task_name, label="target"
    )
    _validate_constraint_values(
        trigger.support_instances, trigger.support_prefixes, task_name=task_name, label="support"
    )
    has_target_constraint = bool(trigger.target_instances or trigger.target_prefixes)
    has_support_constraint = bool(trigger.support_instances or trigger.support_prefixes)
    if trigger.kind in _ACTION_TRIGGERS:
        if not isinstance(trigger.action, ActionName):
            _fail(f"{task_name}: {trigger.kind.value} requires an ActionName")
        if trigger.action not in get_task_spec(task_name).allowed_actions:
            _fail(f"{task_name}: trigger action {trigger.action.value!r} is not allowed")
        if not has_target_constraint:
            _fail(f"{task_name}: action trigger requires a target instance or prefix constraint")
        if trigger.action is ActionName.PLACE_NEXT_TO:
            if not has_support_constraint:
                _fail(f"{task_name}: place_next_to trigger also requires a support constraint")
        elif has_support_constraint:
            _fail(f"{task_name}: only place_next_to triggers may carry support constraints")
    elif trigger.action is not None or has_target_constraint or has_support_constraint:
        _fail(f"{task_name}: {trigger.kind.value} must not carry action role constraints")
    if trigger.kind is TriggerKind.AFTER_REVISIT:
        revisit = trigger.evaluator_revisit
        if not isinstance(revisit, RevisitTriggerSpec):
            _fail(f"{task_name}: after_revisit requires evaluator-only RevisitTriggerSpec")
        if not isinstance(revisit.anchor, RevisitAnchor):
            _fail(f"{task_name}: revisit anchor must be RevisitAnchor")
        if not isinstance(revisit.anchor.kind, RevisitAnchorKind):
            _fail(f"{task_name}: revisit anchor kind must be RevisitAnchorKind")
        anchor_value = revisit.anchor.value
        if (
            not isinstance(anchor_value, str)
            or not anchor_value
            or anchor_value != anchor_value.strip()
            or any(char.isspace() for char in anchor_value)
            or ":" in anchor_value
        ):
            _fail(f"{task_name}: revisit anchor value must be a private typed selector value")
        if (
            revisit.anchor.kind is RevisitAnchorKind.ROLE
            and _ID_PATTERN.fullmatch(anchor_value) is None
        ):
            _fail(f"{task_name}: revisit role anchor must match {_ID_PATTERN.pattern!r}")
        for radius_name in ("leave_radius", "reenter_radius"):
            radius = getattr(revisit, radius_name)
            if type(radius) is not float or radius <= 0.0:
                _fail(f"{task_name}: revisit {radius_name} must be a positive float in metres")
        if revisit.reenter_radius >= revisit.leave_radius:
            _fail(f"{task_name}: revisit reenter_radius must be smaller than leave_radius")
        if type(revisit.min_absence_steps) is not int or revisit.min_absence_steps < 1:
            _fail(f"{task_name}: revisit min_absence_steps must be a positive integer")
    elif trigger.evaluator_revisit is not None:
        _fail(f"{task_name}: only after_revisit may carry evaluator-only revisit criteria")
    if final and trigger.kind is not TriggerKind.FINAL_SUCCESS_CHECK:
        _fail(f"{task_name}: final phase must use final_success_check")
    if not final and trigger.kind is TriggerKind.FINAL_SUCCESS_CHECK:
        _fail(f"{task_name}: only the final phase may use final_success_check")


def _validate_probe(probe: DiagnosticProbe, task_name: str) -> None:
    _validate_id(probe.id, f"{task_name} probe id")
    if not isinstance(probe.family, ProbeFamily):
        _fail(f"{task_name}/{probe.id}: family must be ProbeFamily")
    if not isinstance(probe.prompt, str) or not probe.prompt.strip():
        _fail(f"{task_name}/{probe.id}: prompt must be non-empty")
    if len(probe.options) < 2:
        _fail(f"{task_name}/{probe.id}: at least two options are required")
    option_ids = [option.id for option in probe.options]
    option_texts = [option.text for option in probe.options]
    if len(option_ids) != len(set(option_ids)):
        _fail(f"{task_name}/{probe.id}: option ids must be unique")
    if len(option_texts) != len(set(option_texts)):
        _fail(f"{task_name}/{probe.id}: option text must be unique")
    for option in probe.options:
        _validate_id(option.id, f"{task_name}/{probe.id} option id")
        if not isinstance(option.text, str) or not option.text.strip():
            _fail(f"{task_name}/{probe.id}: option text must be non-empty")
    if not isinstance(probe.oracle, ProbeOracle):
        _fail(f"{task_name}/{probe.id}: oracle must be ProbeOracle")
    if not isinstance(probe.oracle.mode, OracleMode):
        _fail(f"{task_name}/{probe.id}: oracle mode must be OracleMode")
    if probe.oracle.mode is OracleMode.DETERMINISTIC_EXPECTED:
        if probe.oracle.expected_option_id not in option_ids:
            _fail(f"{task_name}/{probe.id}: expected oracle answer is not one of the option ids")
    elif probe.oracle.expected_option_id is not None:
        _fail(f"{task_name}/{probe.id}: runtime-grounded oracle cannot predeclare an answer")
    if not probe.oracle.source.strip() or not probe.oracle.verifier.strip() or not probe.oracle.resolver.strip():
        _fail(f"{task_name}/{probe.id}: oracle source, verifier, and resolver must be non-empty")
    if not probe.evidence_tiers:
        _fail(f"{task_name}/{probe.id}: at least one evidence tier is required")
    if any(not isinstance(tier, EvidenceTier) for tier in probe.evidence_tiers):
        _fail(f"{task_name}/{probe.id}: invalid evidence tier")


def _validate_intervention(
    intervention: SceneIntervention,
    *,
    probe: DiagnosticProbe,
    task_name: str,
) -> None:
    _validate_id(intervention.id, f"{task_name} intervention id")
    if not isinstance(intervention.status, InterventionStatus):
        _fail(f"{task_name}/{intervention.id}: invalid intervention status")
    if not isinstance(intervention.adapter_key, InterventionAdapterKey):
        _fail(f"{task_name}/{intervention.id}: adapter_key must be InterventionAdapterKey")
    if not isinstance(intervention.hidden_parameters, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2
        for item in intervention.hidden_parameters
    ):
        _fail(f"{task_name}/{intervention.id}: hidden parameters must be (name, scalar) tuples")
    parameter_names = [name for name, _ in intervention.hidden_parameters]
    if len(parameter_names) != len(set(parameter_names)) or any(
        not isinstance(name, str) or not name.strip() for name in parameter_names
    ):
        _fail(f"{task_name}/{intervention.id}: hidden parameter names must be non-empty and unique")
    schema = INTERVENTION_PARAMETER_SCHEMAS[intervention.adapter_key]
    missing = set(schema) - set(parameter_names)
    unknown = set(parameter_names) - set(schema)
    if missing or unknown:
        _fail(
            f"{task_name}/{intervention.id}: adapter parameter schema mismatch; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    parameters = dict(intervention.hidden_parameters)
    for name, allowed_types in schema.items():
        value = parameters[name]
        if type(value) not in allowed_types:
            expected = ", ".join(item.__name__ for item in allowed_types)
            _fail(
                f"{task_name}/{intervention.id}: parameter {name!r} must have exact type {expected}"
            )
        if type(value) is str and not value.strip():
            _fail(f"{task_name}/{intervention.id}: string parameter {name!r} must be non-empty")
        if name.endswith("_selector") and not value.startswith(("instance:", "prefix:", "role:")):
            _fail(
                f"{task_name}/{intervention.id}: selector {name!r} must use instance:, prefix:, or role:"
            )
    for name, value in parameters.items():
        if type(value) is float and (
            name.endswith("_m")
            or name.endswith("_mps")
            or name.endswith("_rps")
            or name.endswith("_ns")
        ) and value < 0.0:
            _fail(f"{task_name}/{intervention.id}: numeric parameter {name!r} must be non-negative")
    if "window_steps" in parameters and parameters["window_steps"] <= 0:
        _fail(f"{task_name}/{intervention.id}: window_steps must be positive")
    if "settle_steps" in parameters and parameters["settle_steps"] <= 0:
        _fail(f"{task_name}/{intervention.id}: settle_steps must be positive")
    for fraction_name in (
        "min_visibility_fraction",
        "max_visibility_fraction",
        "min_projected_overlap_fraction",
    ):
        if fraction_name in parameters and not 0.0 <= parameters[fraction_name] <= 1.0:
            _fail(f"{task_name}/{intervention.id}: {fraction_name} must be in [0, 1]")
    if (
        "min_visibility_fraction" in parameters
        and parameters["min_visibility_fraction"] > parameters["max_visibility_fraction"]
    ):
        _fail(f"{task_name}/{intervention.id}: visibility bounds are reversed")
    if not intervention.public_notice.strip():
        _fail(f"{task_name}/{intervention.id}: public notice must be non-empty")
    if not intervention.hidden_operation.strip() or not intervention.hidden_recovery.strip():
        _fail(f"{task_name}/{intervention.id}: evaluator operation and recovery must be non-empty")
    if not intervention.required_tiers:
        _fail(f"{task_name}/{intervention.id}: required tiers must be non-empty")
    if any(not isinstance(tier, EvidenceTier) for tier in intervention.required_tiers):
        _fail(f"{task_name}/{intervention.id}: invalid required tier")
    if not intervention.required_tiers.issubset(probe.evidence_tiers):
        _fail(f"{task_name}/{intervention.id}: intervention tiers must be included in probe evidence")
    if intervention.status is InterventionStatus.RUNTIME_NATIVE:
        if probe.oracle.mode is not OracleMode.RUNTIME_GROUNDED:
            _fail(f"{task_name}/{intervention.id}: runtime-native outcome must use a runtime-grounded oracle")
    if intervention.status is InterventionStatus.DECLARED_REQUIRES_ADAPTER:
        if (
            not isinstance(intervention.deterministic_seed, int)
            or isinstance(intervention.deterministic_seed, bool)
            or intervention.deterministic_seed < 0
        ):
            _fail(f"{task_name}/{intervention.id}: declared mutation requires a non-negative seed")
    elif intervention.deterministic_seed is not None:
        _fail(f"{task_name}/{intervention.id}: only declared mutations may carry a seed")
    if intervention.status is InterventionStatus.OBSERVATION_ONLY:
        if intervention.required_tiers != frozenset({EvidenceTier.VISUAL}):
            _fail(f"{task_name}/{intervention.id}: observation-only intervention must be visual-only")


def validate_hard_trajectory(spec: HardTrajectorySpec) -> None:
    """Validate one manifest or raise :class:`ManifestValidationError`."""

    if spec.task_name not in FIVE_TASKS:
        _fail(f"unknown pilot task {spec.task_name!r}")
    if not isinstance(spec.variant, str) or not spec.variant.strip():
        _fail(f"{spec.task_name}: variant must be non-empty")
    if not isinstance(spec.title, str) or not spec.title.strip():
        _fail(f"{spec.task_name}: title must be non-empty")
    official = get_task_spec(spec.task_name)
    if spec.goal_leaf_templates != official.goal_leaf_templates:
        _fail(f"{spec.task_name}: goal leaf templates do not match the task hints")
    if spec.official_canonical_goal != official.official_canonical_goal:
        _fail(f"{spec.task_name}: hard trajectory must preserve the complete canonical BDDL :goal")
    if not 6 <= spec.checkpoint_count <= 8:
        _fail(f"{spec.task_name}: expected 6-8 checkpoints, got {spec.checkpoint_count}")
    if len(spec.families) < 5:
        _fail(f"{spec.task_name}: expected at least five probe families")
    if not spec.phases:
        _fail(f"{spec.task_name}: phases must not be empty")
    if [phase.ordinal for phase in spec.phases] != list(range(len(spec.phases))):
        _fail(f"{spec.task_name}: phase ordinals must be contiguous and start at zero")

    final_phases = [phase for phase in spec.phases if phase.final_success]
    if len(final_phases) != 1 or final_phases[0] is not spec.phases[-1]:
        _fail(f"{spec.task_name}: exactly one final-success phase is required and it must be last")

    phase_ids: set[str] = set()
    probe_ids: set[str] = set()
    intervention_ids: set[str] = set()
    for phase in spec.phases:
        _validate_id(phase.id, f"{spec.task_name} phase id")
        if phase.id in phase_ids:
            _fail(f"{spec.task_name}: duplicate phase id {phase.id!r}")
        phase_ids.add(phase.id)
        if not isinstance(phase.objective, str) or not phase.objective.strip():
            _fail(f"{spec.task_name}/{phase.id}: objective must be non-empty")
        _validate_trigger(phase.trigger, final=phase.final_success, task_name=spec.task_name)

        if phase.final_success:
            if phase.probe is not None or phase.intervention is not None:
                _fail(f"{spec.task_name}/{phase.id}: final phase cannot contain a probe or intervention")
            if phase.hidden_recovery:
                _fail(f"{spec.task_name}/{phase.id}: final phase cannot contain a recovery recipe")
            if phase.required_goal_fragments:
                _fail(f"{spec.task_name}/{phase.id}: final check must not substitute leaf templates for the complete goal")
            if phase.required_canonical_goal != spec.official_canonical_goal:
                _fail(f"{spec.task_name}/{phase.id}: final check must bind the complete canonical BDDL :goal")
            continue

        if phase.probe is None:
            _fail(f"{spec.task_name}/{phase.id}: every pre-final phase must contain a probe")
        if phase.required_goal_fragments or phase.required_canonical_goal is not None:
            _fail(f"{spec.task_name}/{phase.id}: only final phase may carry goal requirements")
        if not phase.hidden_recovery.strip():
            _fail(f"{spec.task_name}/{phase.id}: evaluator recovery recipe must be non-empty")
        _validate_probe(phase.probe, spec.task_name)
        if phase.probe.id in probe_ids:
            _fail(f"{spec.task_name}: duplicate probe id {phase.probe.id!r}")
        probe_ids.add(phase.probe.id)
        if phase.intervention is not None:
            _validate_intervention(phase.intervention, probe=phase.probe, task_name=spec.task_name)
            if phase.intervention.id in intervention_ids:
                _fail(f"{spec.task_name}: duplicate intervention id {phase.intervention.id!r}")
            intervention_ids.add(phase.intervention.id)

    all_ids = phase_ids | probe_ids | intervention_ids
    if len(all_ids) != len(phase_ids) + len(probe_ids) + len(intervention_ids):
        _fail(f"{spec.task_name}: phase, probe, and intervention ids must be mutually unique")


def validate_hard_trajectory_registry(
    registry: Mapping[str, HardTrajectorySpec] | None = None,
) -> None:
    """Validate the five-task registry and its exact ten-family coverage."""

    current = HARD_V1_TRAJECTORIES if registry is None else registry
    if set(INTERVENTION_PARAMETER_SCHEMAS) != set(InterventionAdapterKey):
        _fail("intervention parameter schemas must cover every adapter key")
    if set(ESI_FAMILY_MAPPINGS) != set(ProbeFamily):
        _fail("ESI family metadata must cover exactly the ten probe families")
    if set(current) != set(FIVE_TASKS):
        _fail("hard-v1 registry must contain exactly the five pilot task names")
    global_ids: set[str] = set()
    covered: set[ProbeFamily] = set()
    for task_name, spec in current.items():
        if task_name != spec.task_name:
            _fail(f"registry key {task_name!r} does not match manifest task {spec.task_name!r}")
        if spec.variant != "hard_v1":
            _fail(f"{task_name}: registry variant must be 'hard_v1'")
        validate_hard_trajectory(spec)
        covered.update(spec.families)
        ids = {
            item_id
            for phase in spec.phases
            for item_id in (
                phase.id,
                phase.probe.id if phase.probe is not None else None,
                phase.intervention.id if phase.intervention is not None else None,
            )
            if item_id is not None
        }
        duplicates = global_ids.intersection(ids)
        if duplicates:
            _fail(f"hard-v1 ids must be globally unique: {sorted(duplicates)}")
        global_ids.update(ids)
    if covered != set(ProbeFamily):
        missing = sorted(family.value for family in set(ProbeFamily) - covered)
        extra = sorted(str(family) for family in covered - set(ProbeFamily))
        _fail(f"hard-v1 must cover exactly the ten probe families; missing={missing}, extra={extra}")


def get_hard_trajectory(task_name: str, variant: str = "hard_v1") -> HardTrajectorySpec:
    if variant != "hard_v1":
        raise ValueError(f"unsupported hard trajectory variant {variant!r}; expected 'hard_v1'")
    try:
        return HARD_V1_TRAJECTORIES[task_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported hard trajectory task {task_name!r}; choose one of {sorted(HARD_V1_TRAJECTORIES)}"
        ) from exc


def describe_hard_trajectory(task_name: str, variant: str = "hard_v1") -> dict[str, Any]:
    """Return a policy-safe description with all evaluator secrets removed."""

    return get_hard_trajectory(task_name, variant).to_public_dict()


def _opts(*texts: str) -> tuple[ProbeOption, ...]:
    return tuple(ProbeOption(id=f"option_{index}", text=text) for index, text in enumerate(texts, start=1))


def _probe(
    *,
    probe_id: str,
    family: ProbeFamily,
    prompt: str,
    options: tuple[ProbeOption, ...],
    tiers: frozenset[EvidenceTier],
    source: str,
    verifier: str,
    correct: int | None = None,
    runtime_grounded: bool = False,
) -> DiagnosticProbe:
    if runtime_grounded and correct is not None:
        _fail(f"{probe_id}: runtime-grounded probes must not predeclare a correct option")
    if not runtime_grounded and correct is None:
        _fail(f"{probe_id}: deterministic probes require a correct option")
    mode = OracleMode.RUNTIME_GROUNDED if runtime_grounded else OracleMode.DETERMINISTIC_EXPECTED
    return DiagnosticProbe(
        id=probe_id,
        family=family,
        prompt=prompt,
        options=options,
        evidence_tiers=tiers,
        oracle=ProbeOracle(
            mode=mode,
            expected_option_id=None if runtime_grounded else f"option_{correct}",
            source=source,
            verifier=verifier,
            resolver=f"resolve {probe_id} from frozen checkpoint state using: {verifier}",
        ),
    )


def _intervention(
    *,
    intervention_id: str,
    status: InterventionStatus,
    notice: str,
    tiers: frozenset[EvidenceTier],
    adapter_key: InterventionAdapterKey,
    operation: str,
    recovery: str,
    parameters: tuple[tuple[str, str | int | float | bool], ...] = (),
    seed: int | None = None,
) -> SceneIntervention:
    return SceneIntervention(
        id=intervention_id,
        status=status,
        public_notice=notice,
        required_tiers=tiers,
        adapter_key=adapter_key,
        hidden_operation=operation,
        hidden_recovery=recovery,
        hidden_parameters=parameters,
        deterministic_seed=seed,
    )


def _revisit_trigger(
    *,
    anchor: RevisitAnchor,
    occurrence: int = 1,
    leave_radius: float = 2.0,
    reenter_radius: float = 1.0,
    min_absence_steps: int = 30,
) -> ProbeTrigger:
    """Build private revisit criteria; this does not detect or signal an event."""

    return ProbeTrigger(
        kind=TriggerKind.AFTER_REVISIT,
        occurrence=occurrence,
        evaluator_revisit=RevisitTriggerSpec(
            anchor=anchor,
            leave_radius=leave_radius,
            reenter_radius=reenter_radius,
            min_absence_steps=min_absence_steps,
        ),
    )


def _phase(
    *,
    phase_id: str,
    ordinal: int,
    objective: str,
    trigger: ProbeTrigger,
    probe: DiagnosticProbe,
    recovery: str,
    intervention: SceneIntervention | None = None,
) -> TrajectoryPhase:
    return TrajectoryPhase(
        id=phase_id,
        ordinal=ordinal,
        objective=objective,
        trigger=trigger,
        probe=probe,
        intervention=intervention,
        hidden_recovery=recovery,
    )


def _final_phase(prefix: str, ordinal: int, task_name: str) -> TrajectoryPhase:
    canonical_goal = get_task_spec(task_name).official_canonical_goal
    return TrajectoryPhase(
        id=f"{prefix}_final_success",
        ordinal=ordinal,
        objective="Evaluate the unchanged official BDDL goal and task Q-score.",
        trigger=ProbeTrigger(kind=TriggerKind.FINAL_SUCCESS_CHECK),
        probe=None,
        final_success=True,
        required_canonical_goal=canonical_goal,
    )


S = frozenset({EvidenceTier.SYMBOLIC})
V = frozenset({EvidenceTier.VISUAL})
P = frozenset({EvidenceTier.PHYSICS})
SV = frozenset({EvidenceTier.SYMBOLIC, EvidenceTier.VISUAL})
SP = frozenset({EvidenceTier.SYMBOLIC, EvidenceTier.PHYSICS})
VP = frozenset({EvidenceTier.VISUAL, EvidenceTier.PHYSICS})
SVP = frozenset({EvidenceTier.SYMBOLIC, EvidenceTier.VISUAL, EvidenceTier.PHYSICS})


def _radio_manifest() -> HardTrajectorySpec:
    task_name = "turning_on_radio"
    phases = (
        _phase(
            phase_id="radio_phase_01",
            ordinal=0,
            objective="Identify the interaction target from an initially incomplete view.",
            trigger=ProbeTrigger(TriggerKind.EPISODE_START),
            probe=_probe(
                probe_id="radio_probe_01",
                family=ProbeFamily.PARTIAL_OCCLUSION,
                prompt="Which labeled candidate matches the scoped radio receiver across the available views?",
                options=_opts("candidate A", "candidate B"),
                correct=1,
                tiers=V,
                source="seeded camera visibility, candidate-label map, and simulator instance identity",
                verifier="multi-view instance segmentation associates candidate A with the scoped radio receiver",
            ),
            intervention=_intervention(
                intervention_id="radio_intervention_01",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="A seeded, task-preserving visual occluder is present at episode start.",
                tiers=V,
                adapter_key=InterventionAdapterKey.OCCLUDE_WITH_DISTRACTOR,
                operation="place a non-task distractor so the radio remains on its original support but is partially occluded",
                recovery="remove only the distractor if the required alternate view is not reachable",
                parameters=(
                    ('target_selector', 'instance:radio_receiver.n.01_1'),
                    ('postcondition', 'visibility_fraction_between_bounds'),
                    ('occluder_selector', 'role:seeded_radio_distractor'),
                    ('initial_view_id', 'radio_front_initial'),
                    ('verification_view_id', 'radio_lateral_verification'),
                    ('min_visibility_fraction', 0.15),
                    ('max_visibility_fraction', 0.55),
                ),
                seed=1101,
            ),
            recovery="permit lateral base motion and a second camera view; never reveal the instance id",
        ),
        _phase(
            phase_id="radio_phase_02",
            ordinal=1,
            objective="Compare the initial and alternate views before approaching.",
            trigger=ProbeTrigger(TriggerKind.BEFORE_ACTION, action=ActionName.NAVIGATE_TO, target_instances=("radio_receiver.n.01_1",)),
            probe=_probe(
                probe_id="radio_probe_02",
                family=ProbeFamily.VIEW_HALLUCINATION,
                prompt="After changing viewpoint, is the apparent front control part of the radio or only aligned in projection?",
                options=_opts("it is part of the radio", "it only overlaps in the original view"),
                correct=1,
                tiers=V,
                source="two calibrated RGB-D views of the radio and distractor",
                verifier="projected instance masks separate after a viewpoint change",
            ),
            intervention=_intervention(
                intervention_id="radio_intervention_02",
                status=InterventionStatus.OBSERVATION_ONLY,
                notice="The first view is intentionally ambiguous; active viewpoint change is allowed.",
                tiers=V,
                adapter_key=InterventionAdapterKey.MISLEADING_VIEW,
                operation="select a low-angle camera pose with projected overlap",
                recovery="allow one side-step viewpoint before requiring an answer",
                parameters=(
                    ('target_selector', 'instance:radio_receiver.n.01_1'),
                    ('postcondition', 'multi_view_identity_resolved'),
                    ('initial_view_id', 'radio_low_overlap'),
                    ('verification_view_id', 'radio_side_separated'),
                    ('min_projected_overlap_fraction', 0.65),
                ),
            ),
            recovery="request a side view and re-identify the receiver before navigation",
        ),
        _phase(
            phase_id="radio_phase_03",
            ordinal=2,
            objective="Choose a valid ordering for target acquisition and operation.",
            trigger=ProbeTrigger(TriggerKind.BEFORE_ACTION, action=ActionName.NAVIGATE_TO, target_instances=("radio_receiver.n.01_1",)),
            probe=_probe(
                probe_id="radio_probe_03",
                family=ProbeFamily.ACTION_ORDER_INFERENCE,
                prompt="Which plan satisfies both target-localization and contact preconditions?",
                options=_opts("plan A: refresh the target pose, approach it, then attempt the control", "plan B: approach the cached pose, then refresh the target pose before attempting the control"),
                correct=1,
                tiers=S,
                source="high-level action preconditions and current target-localization state",
                verifier="navigate_to must precede the contact-dependent toggle attempt",
            ),
            recovery="reject a premature toggle and return control to target acquisition",
        ),
        _phase(
            phase_id="radio_phase_04",
            ordinal=3,
            objective="Verify physical readiness at the control surface.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.NAVIGATE_TO, target_instances=("radio_receiver.n.01_1",)),
            probe=_probe(
                probe_id="radio_probe_04",
                family=ProbeFamily.PHYSICAL_CONTACT,
                prompt="Does the current end-effector pose satisfy the physical contact-readiness criterion?",
                options=_opts("yes", "no"),
                tiers=VP,
                source="end-effector/control geometry and simulator contact state",
                verifier="interaction pose is within tolerance and a contact-capable approach is collision free",
                runtime_grounded=True,
            ),
            intervention=_intervention(
                intervention_id="radio_intervention_04",
                status=InterventionStatus.RUNTIME_NATIVE,
                notice="Readiness is measured from the executed approach, not from image overlap.",
                tiers=VP,
                adapter_key=InterventionAdapterKey.OBSERVE_RUNTIME,
                operation="observe the policy-created end-effector pose without teleporting either body",
                recovery="back off and repeat navigate_to with a reachable interaction pose",
                parameters=(
                    ('target_selector', 'instance:radio_receiver.n.01_1'),
                    ('postcondition', 'runtime_measurement_frozen'),
                    ('measurement_name', 'contact_readiness'),
                    ('window_steps', 8),
                ),
            ),
            recovery="repeat the approach if geometry or contact readiness is false",
        ),
        _phase(
            phase_id="radio_phase_05",
            ordinal=4,
            objective="Compare remembered and current target state after a revisit.",
            trigger=_revisit_trigger(
                anchor=RevisitAnchor(RevisitAnchorKind.INSTANCE, "radio_receiver.n.01_1")
            ),
            probe=_probe(
                probe_id="radio_probe_05",
                family=ProbeFamily.UNOBSERVED_CHANGE,
                prompt="Which before/after pose relation is reported for the radio?",
                options=_opts("the radio moved to a second marked pose on the same table", "the radio remained at its remembered pose"),
                correct=1,
                tiers=SV,
                source="before/after object-pose snapshots keyed by the scoped radio instance",
                verifier="pose delta exceeds threshold while OnTop(radio, original support) remains true",
            ),
            intervention=_intervention(
                intervention_id="radio_intervention_05",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="A seeded evaluator intervention may or may not have changed task-relevant state while unobserved.",
                tiers=SV,
                adapter_key=InterventionAdapterKey.RELOCATE_ON_SUPPORT,
                operation="move the radio between two validated stable poses on its original table",
                recovery="restore the second stable pose and require reacquisition rather than using stale coordinates",
                parameters=(
                    ('target_selector', 'instance:radio_receiver.n.01_1'),
                    ('postcondition', 'target_stable_on_original_support'),
                    ('support_selector', 'instance:table.n.02_1'),
                    ('displacement_m', 0.35),
                    ('settle_steps', 30),
                ),
                seed=1105,
            ),
            recovery="invalidate cached target coordinates and visually reacquire the radio",
        ),
        _phase(
            phase_id="radio_phase_06",
            ordinal=5,
            objective="Choose a valid post-revisit action order.",
            trigger=ProbeTrigger(TriggerKind.BEFORE_ACTION, action=ActionName.NAVIGATE_TO, occurrence=2, target_instances=("radio_receiver.n.01_1",)),
            probe=_probe(
                probe_id="radio_probe_06",
                family=ProbeFamily.ACTION_ORDER_INFERENCE,
                prompt="Which plan consumes the newest target pose before the approach?",
                options=_opts("plan A: refresh pose, approach the refreshed pose, check readiness, then toggle", "plan B: approach the cached pose, refresh pose in place, check readiness, then toggle"),
                correct=1,
                tiers=SV,
                source="current target pose plus navigation and toggle preconditions",
                verifier="the selected order consumes the updated pose before toggle_on",
            ),
            recovery="clear the stale plan and regenerate the approach from the current observation",
        ),
        _phase(
            phase_id="radio_phase_07",
            ordinal=6,
            objective="Perform a final observation-based target check before the only toggle.",
            trigger=ProbeTrigger(TriggerKind.BEFORE_ACTION, action=ActionName.TOGGLE_ON, target_instances=("radio_receiver.n.01_1",)),
            probe=_probe(
                probe_id="radio_probe_07",
                family=ProbeFamily.VIEW_HALLUCINATION,
                prompt="How should the close-view object be associated with the side-view instance track?",
                options=_opts("the same instance track", "a different instance track"),
                tiers=V,
                source="multi-view instance association immediately before toggle",
                verifier="both views resolve to the BDDL-scoped radio receiver",
                runtime_grounded=True,
            ),
            recovery="block toggle_on and reacquire if multi-view identity is inconsistent",
        ),
        _final_phase("radio", 7, task_name),
    )
    return HardTrajectorySpec(
        task_name=task_name,
        variant="hard_v1",
        title="Radio operation with active perception, memory, and contact checks",
        goal_leaf_templates=get_task_spec(task_name).goal_leaf_templates,
        official_canonical_goal=get_task_spec(task_name).official_canonical_goal,
        phases=phases,
    )


def _trash_manifest() -> HardTrajectorySpec:
    task_name = "picking_up_trash"
    phases = (
        _phase(
            phase_id="trash_phase_01",
            ordinal=0,
            objective="Select a task-completing collection order.",
            trigger=ProbeTrigger(TriggerKind.EPISODE_START),
            probe=_probe(
                probe_id="trash_probe_01",
                family=ProbeFamily.ACTION_ORDER_INFERENCE,
                prompt="Which symbolic sequence satisfies the pickup and containment preconditions?",
                options=_opts("sequence A: locate → pick_up → navigate_to(ashcan) → place_inside", "sequence B: locate → navigate_to(ashcan) → pick_up → place_inside"),
                correct=1,
                tiers=S,
                source="pick_up and place_inside preconditions",
                verifier="the selected sequence binds a held scoped can before containment",
            ),
            recovery="reject invalid dependency order and retain all cans as pending",
        ),
        _phase(
            phase_id="trash_phase_02",
            ordinal=1,
            objective="Plan repeated travel between trash locations and the ashcan.",
            trigger=ProbeTrigger(TriggerKind.BEFORE_ACTION, action=ActionName.NAVIGATE_TO, target_prefixes=("can__of__soda.n.01_",)),
            probe=_probe(
                probe_id="trash_probe_02",
                family=ProbeFamily.LONG_TERM_NAVIGATION,
                prompt="Which endpoint-selection policy satisfies the collection-state invariant after each ashcan visit?",
                options=_opts("policy A: derive the next endpoints from the current per-instance state table", "policy B: replay the episode-start endpoint itinerary after each ashcan visit"),
                correct=1,
                tiers=S,
                source="scene traversability graph and per-instance pending/held/contained state",
                verifier="route endpoints correspond to an uncontained can and the scoped ashcan",
            ),
            recovery="recompute the next endpoint from current per-instance state",
        ),
        _phase(
            phase_id="trash_phase_03",
            ordinal=2,
            objective="Recover a target hidden by ordinary clutter.",
            trigger=_revisit_trigger(
                anchor=RevisitAnchor(RevisitAnchorKind.ROLE, "active_pending_can")
            ),
            probe=_probe(
                probe_id="trash_probe_03",
                family=ProbeFamily.PARTIAL_OCCLUSION,
                prompt="Which labeled cylinder is associated with the scoped soda-can instance after the alternate view?",
                options=_opts("candidate A", "candidate B"),
                correct=1,
                tiers=V,
                source="seeded clutter placement, candidate-label map, and multi-view instance segmentation",
                verifier="alternate view associates candidate A with the scoped can without moving it off its initial support",
            ),
            intervention=_intervention(
                intervention_id="trash_intervention_03",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="One target is partially occluded by seeded non-task clutter.",
                tiers=V,
                adapter_key=InterventionAdapterKey.OCCLUDE_WITH_DISTRACTOR,
                operation="place a distractor between the initial camera and one scoped soda can",
                recovery="remove the distractor only if no collision-free alternate view exists",
                parameters=(
                    ('target_selector', 'prefix:can__of__soda.n.01_'),
                    ('postcondition', 'visibility_fraction_between_bounds'),
                    ('occluder_selector', 'role:seeded_trash_clutter'),
                    ('initial_view_id', 'trash_floor_initial'),
                    ('verification_view_id', 'trash_floor_lateral'),
                    ('min_visibility_fraction', 0.1),
                    ('max_visibility_fraction', 0.5),
                ),
                seed=1203,
            ),
            recovery="move laterally for a second view before classifying the target",
        ),
        _phase(
            phase_id="trash_phase_04",
            ordinal=3,
            objective="Confirm that grasp appearance corresponds to physical attachment.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.PICK_UP, settle_steps=8, target_prefixes=("can__of__soda.n.01_",)),
            probe=_probe(
                probe_id="trash_probe_04",
                family=ProbeFamily.PHYSICAL_CONTACT,
                prompt="After lifting and settling, is the soda can physically held?",
                options=_opts("yes", "no"),
                tiers=SP,
                source="held-object binding, gripper contacts, and relative-pose persistence",
                verifier="the scoped can remains attached through the post-grasp settle window",
                runtime_grounded=True,
            ),
            intervention=_intervention(
                intervention_id="trash_intervention_04",
                status=InterventionStatus.RUNTIME_NATIVE,
                notice="Grasp truth is measured after the executed pickup settles.",
                tiers=SP,
                adapter_key=InterventionAdapterKey.OBSERVE_RUNTIME,
                operation="sample contacts and relative transform after the policy pickup",
                recovery="release any false attachment, re-approach, and retry pick_up",
                parameters=(
                    ('target_selector', 'prefix:can__of__soda.n.01_'),
                    ('postcondition', 'runtime_measurement_frozen'),
                    ('measurement_name', 'held_attachment'),
                    ('window_steps', 8),
                ),
            ),
            recovery="retry the grasp if contact and held-object state disagree",
        ),
        _phase(
            phase_id="trash_phase_05",
            ordinal=4,
            objective="Evaluate strict containment after the first placement.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.PLACE_INSIDE, settle_steps=12, target_instances=("ashcan.n.01_1",)),
            probe=_probe(
                probe_id="trash_probe_05",
                family=ProbeFamily.RIGID_CONTAINMENT,
                prompt="Is the can fully contained by the ashcan after settling?",
                options=_opts("yes, it is inside", "no, it is supported on or crossing the rim"),
                correct=2,
                tiers=SVP,
                source="seeded rim placement plus official Inside and strict full-volume containment checks",
                verifier="official Inside(can, ashcan) is false and strict full-volume containment is false after settling",
            ),
            intervention=_intervention(
                intervention_id="trash_intervention_05",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="A seeded evaluator-controlled placement perturbation is applied before verification.",
                tiers=SVP,
                adapter_key=InterventionAdapterKey.CONTAINMENT_NEAR_MISS,
                operation="place the first can in a stable rim-crossing pose that fails official Inside and strict full-volume containment",
                recovery="return the can to the gripper and permit a deeper place_inside retry",
                parameters=(
                    ('target_selector', 'prefix:can__of__soda.n.01_'),
                    ('postcondition', 'official_inside_false_and_strict_full_volume_containment_false'),
                    ('container_selector', 'instance:ashcan.n.01_1'),
                    ('protrusion_m', 0.04),
                ),
                seed=1205,
            ),
            recovery="re-grasp and place deeper when strict containment is false",
        ),
        _phase(
            phase_id="trash_phase_06",
            ordinal=5,
            objective="Compare front and side/depth evidence for the first placement.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.PLACE_INSIDE, settle_steps=12, target_instances=("ashcan.n.01_1",)),
            probe=_probe(
                probe_id="trash_probe_06",
                family=ProbeFamily.VIEW_HALLUCINATION,
                prompt="The can looks inside from the front. What does the side/depth view show?",
                options=_opts("the full can is below the rim", "part of the can still crosses the rim"),
                correct=2,
                tiers=VP,
                source="paired front and side RGB-D views of the deterministic near-miss",
                verifier="side depth reveals can geometry above and outside the container interior",
            ),
            intervention=_intervention(
                intervention_id="trash_intervention_06",
                status=InterventionStatus.OBSERVATION_ONLY,
                notice="The initial verification view is intentionally misleading; a side view is available.",
                tiers=V,
                adapter_key=InterventionAdapterKey.MISLEADING_VIEW,
                operation="select a front camera pose where the near-miss projects inside the ashcan silhouette",
                recovery="provide a reachable side view with depth before answer submission",
                parameters=(
                    ('target_selector', 'prefix:can__of__soda.n.01_'),
                    ('postcondition', 'multi_view_containment_resolved'),
                    ('initial_view_id', 'trash_ashcan_front'),
                    ('verification_view_id', 'trash_ashcan_side_depth'),
                    ('min_projected_overlap_fraction', 0.7),
                ),
            ),
            recovery="change viewpoint, then execute the containment recovery from the prior checkpoint",
        ),
        _phase(
            phase_id="trash_phase_07",
            ordinal=6,
            objective="Revalidate earlier progress after leaving the container area.",
            trigger=_revisit_trigger(
                anchor=RevisitAnchor(RevisitAnchorKind.INSTANCE, "ashcan.n.01_1")
            ),
            probe=_probe(
                probe_id="trash_probe_07",
                family=ProbeFamily.UNOBSERVED_CHANGE,
                prompt="Which current-state report matches the before/after containment snapshots?",
                options=_opts("report A: the first can is Inside(ashcan)", "report B: the first can is not Inside(ashcan)"),
                correct=2,
                tiers=SVP,
                source="before/after Inside snapshots and settled object pose",
                verifier="the first can no longer satisfies containment after the seeded disturbance",
            ),
            intervention=_intervention(
                intervention_id="trash_intervention_07",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="A seeded evaluator intervention may or may not alter prior progress while unobserved.",
                tiers=SVP,
                adapter_key=InterventionAdapterKey.DISTURB_RELATION,
                operation="apply a bounded impulse that moves the first can out of the ashcan after the robot leaves",
                recovery="mark the affected can pending and require another verified place_inside",
                parameters=(
                    ('target_selector', 'prefix:can__of__soda.n.01_'),
                    ('postcondition', 'goal_relation_changed'),
                    ('relation', 'inside(prefix:can__of__soda.n.01_,instance:ashcan.n.01_1)'),
                    ('impulse_ns', 0.35),
                    ('displacement_m', 0.12),
                ),
                seed=1207,
            ),
            recovery="prioritize the invalidated can before stopping",
        ),
        _final_phase("trash", 7, task_name),
    )
    return HardTrajectorySpec(
        task_name=task_name,
        variant="hard_v1",
        title="Trash collection with active perception, containment, and memory checks",
        goal_leaf_templates=get_task_spec(task_name).goal_leaf_templates,
        official_canonical_goal=get_task_spec(task_name).official_canonical_goal,
        phases=phases,
    )


def _coffee_manifest() -> HardTrajectorySpec:
    task_name = "set_up_a_coffee_station_in_your_kitchen"
    phases = (
        _phase(
            phase_id="coffee_phase_01",
            ordinal=0,
            objective="Identify the coffee bottle despite shelf clutter.",
            trigger=ProbeTrigger(TriggerKind.EPISODE_START),
            probe=_probe(
                probe_id="coffee_probe_01",
                family=ProbeFamily.PARTIAL_OCCLUSION,
                prompt="Which labeled bottle is associated with the scoped coffee-bottle instance after the alternate view?",
                options=_opts("candidate A", "candidate B"),
                correct=1,
                tiers=V,
                source="seeded shelf arrangement, candidate-label map, and multi-view instance identity",
                verifier="alternate view associates candidate A with the BDDL-scoped bottle instance",
            ),
            intervention=_intervention(
                intervention_id="coffee_intervention_01",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="Shelf clutter creates a seeded partial occlusion without changing task predicates.",
                tiers=V,
                adapter_key=InterventionAdapterKey.OCCLUDE_WITH_DISTRACTOR,
                operation="place a non-task bottle in front of the scoped coffee bottle while preserving its OnTop shelf state",
                recovery="remove only the distractor if no reachable disambiguating view exists",
                parameters=(
                    ('target_selector', 'instance:bottle__of__coffee.n.01_1'),
                    ('postcondition', 'visibility_fraction_between_bounds'),
                    ('occluder_selector', 'role:seeded_shelf_distractor'),
                    ('initial_view_id', 'coffee_shelf_initial'),
                    ('verification_view_id', 'coffee_shelf_side'),
                    ('min_visibility_fraction', 0.15),
                    ('max_visibility_fraction', 0.55),
                ),
                seed=1301,
            ),
            recovery="obtain a side view and bind the scoped bottle before planning a grasp",
        ),
        _phase(
            phase_id="coffee_phase_02",
            ordinal=1,
            objective="Respect the support dependency between saucer and cup.",
            trigger=ProbeTrigger(TriggerKind.BEFORE_ACTION, action=ActionName.PICK_UP, target_instances=("coffee_cup.n.01_1", "saucer.n.02_1")),
            probe=_probe(
                probe_id="coffee_probe_02",
                family=ProbeFamily.ACTION_ORDER_INFERENCE,
                prompt="Which manipulation sequence satisfies the cup-on-saucer support dependency?",
                options=_opts("sequence A: establish the saucer goal pose, then place the cup on it", "sequence B: hold the cup at the goal pose while attempting to place the saucer beneath it"),
                correct=1,
                tiers=S,
                source="official goal fragment OnTop(coffee_cup, saucer)",
                verifier="support dependency orders saucer placement before cup placement",
            ),
            recovery="keep the cup pending until the saucer has a verified stable support pose",
        ),
        _phase(
            phase_id="coffee_phase_03",
            ordinal=2,
            objective="Verify that the selected object remains held after pickup.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.PICK_UP, settle_steps=8, target_instances=("coffee_cup.n.01_1", "saucer.n.02_1")),
            probe=_probe(
                probe_id="coffee_probe_03",
                family=ProbeFamily.PHYSICAL_CONTACT,
                prompt="Did the picked object remain physically attached through the lift?",
                options=_opts("yes", "no"),
                tiers=SP,
                source="held-object binding, contacts, and relative transform during lift",
                verifier="the scoped object remains attached for the full settle interval",
                runtime_grounded=True,
            ),
            intervention=_intervention(
                intervention_id="coffee_intervention_03",
                status=InterventionStatus.RUNTIME_NATIVE,
                notice="Pickup is verified from settled physical state.",
                tiers=SP,
                adapter_key=InterventionAdapterKey.OBSERVE_RUNTIME,
                operation="observe contacts and attachment generated by the policy pickup",
                recovery="release a false attachment and retry from a collision-free grasp pose",
                parameters=(
                    ('target_selector', 'role:trigger_target'),
                    ('postcondition', 'runtime_measurement_frozen'),
                    ('measurement_name', 'held_attachment'),
                    ('window_steps', 8),
                ),
            ),
            recovery="retry pickup when visual overlap is not backed by attachment and contact",
        ),
        _phase(
            phase_id="coffee_phase_04",
            ordinal=3,
            objective="Choose a collision-free return route while carrying an object.",
            trigger=ProbeTrigger(TriggerKind.BEFORE_ACTION, action=ActionName.NAVIGATE_TO, occurrence=2, target_instances=("coffee_maker.n.01_1",)),
            probe=_probe(
                probe_id="coffee_probe_04",
                family=ProbeFamily.TRAVERSABLE_PASSAGE,
                prompt="Which labeled return route passes the robot-plus-held-object swept-volume check?",
                options=_opts("route A (left marker)", "route B (right marker)"),
                correct=1,
                tiers=SP,
                source="dynamic obstacle state and robot-plus-held-object footprint",
                verifier="route A passes the swept-volume check and route B intersects the appliance door",
            ),
            intervention=_intervention(
                intervention_id="coffee_intervention_04",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="A seeded dynamic obstacle narrows one return passage.",
                tiers=SP,
                adapter_key=InterventionAdapterKey.NARROW_PASSAGE,
                operation="open a non-goal appliance door into one passage while retaining a collision-free alternate side",
                recovery="close the obstacle if the alternate route fails validation",
                parameters=(
                    ('target_selector', 'role:robot_with_held_coffee_item'),
                    ('postcondition', 'alternate_route_traversable'),
                    ('passage_selector', 'role:kitchen_return_passage'),
                    ('blocker_selector', 'instance:dishwasher.n.01_1'),
                    ('min_clearance_m', 0.12),
                    ('alternate_route_selector', 'role:kitchen_counter_clear_side'),
                ),
                seed=1304,
            ),
            recovery="retreat to the staging pose and replan through the verified clear side",
        ),
        _phase(
            phase_id="coffee_phase_05",
            ordinal=4,
            objective="Verify filter support from more than one view.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.PLACE_ON_TOP, settle_steps=10, target_instances=("coffee_maker.n.01_1",)),
            probe=_probe(
                probe_id="coffee_probe_05",
                family=ProbeFamily.VIEW_HALLUCINATION,
                prompt="The filter overlaps the coffee maker in the low view. Is it actually OnTop?",
                options=_opts("Touching and VerticalAdjacency are both true", "at least one OnTop component is false"),
                correct=2,
                tiers=SVP,
                source="paired camera views plus Touching and VerticalAdjacency state",
                verifier="the seeded edge pose lacks the contact/support relation required by OnTop",
            ),
            intervention=_intervention(
                intervention_id="coffee_intervention_05",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="A seeded evaluator-controlled placement perturbation is applied before verification.",
                tiers=SVP,
                adapter_key=InterventionAdapterKey.THRESHOLD_NEAR_MISS,
                operation="place the filter at an edge pose that overlaps in the low view but does not satisfy OnTop",
                recovery="return the filter for a centered place_on_top retry",
                parameters=(
                    ('target_selector', 'instance:paper_coffee_filter.n.01_1'),
                    ('postcondition', 'target_relation_false'),
                    ('support_selector', 'instance:coffee_maker.n.01_1'),
                    ('relation', 'ontop(instance:paper_coffee_filter.n.01_1,instance:coffee_maker.n.01_1)'),
                    ('displacement_m', 0.06),
                ),
                seed=1305,
            ),
            recovery="inspect from the side and re-place the filter when OnTop is false",
        ),
        _phase(
            phase_id="coffee_phase_06",
            ordinal=5,
            objective="Re-evaluate a previously established goal relation after a revisit.",
            trigger=_revisit_trigger(
                anchor=RevisitAnchor(RevisitAnchorKind.ROLE, "coffee_cup_saucer_stack")
            ),
            probe=_probe(
                probe_id="coffee_probe_06",
                family=ProbeFamily.UNOBSERVED_CHANGE,
                prompt="Which relation-state report matches the before/after counter snapshots?",
                options=_opts("report A: OnTop(cup, saucer) is false", "report B: every previously true goal relation remains true"),
                correct=1,
                tiers=SVP,
                source="before/after truth values for the six official goal fragments",
                verifier="only OnTop(coffee_cup, saucer) transitions from true to false",
            ),
            intervention=_intervention(
                intervention_id="coffee_intervention_06",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="A seeded evaluator intervention may or may not alter a completed relation while unobserved.",
                tiers=SVP,
                adapter_key=InterventionAdapterKey.DISTURB_RELATION,
                operation="apply a bounded disturbance that moves the cup off the saucer",
                recovery="mark the cup relation pending and preserve unaffected goal relations",
                parameters=(
                    ('target_selector', 'instance:coffee_cup.n.01_1'),
                    ('postcondition', 'goal_relation_changed'),
                    ('relation', 'ontop(instance:coffee_cup.n.01_1,instance:saucer.n.02_1)'),
                    ('impulse_ns', 0.2),
                    ('displacement_m', 0.08),
                ),
                seed=1306,
            ),
            recovery="restore only the invalidated cup-on-saucer relation",
        ),
        _phase(
            phase_id="coffee_phase_07",
            ordinal=6,
            objective="Wait for delayed motion before accepting the arrangement.",
            trigger=ProbeTrigger(TriggerKind.BEFORE_FINAL_SUCCESS, settle_steps=30),
            probe=_probe(
                probe_id="coffee_probe_07",
                family=ProbeFamily.STACKING_AND_STABILITY,
                prompt="After the settle window, which statement describes the cup-and-saucer stack?",
                options=_opts("support and motion thresholds are satisfied", "at least one support or motion threshold is exceeded"),
                tiers=SP,
                source="OnTop state, body velocities, tilt, and contact persistence over the settle window",
                verifier="support remains true and all stability thresholds hold for 30 steps",
                runtime_grounded=True,
            ),
            intervention=_intervention(
                intervention_id="coffee_intervention_07",
                status=InterventionStatus.RUNTIME_NATIVE,
                notice="Final support relations are evaluated only after physical settling.",
                tiers=SP,
                adapter_key=InterventionAdapterKey.SETTLE_ONLY,
                operation="advance simulation without policy actions and measure delayed motion",
                recovery="re-place only unstable items and repeat the settle window",
                parameters=(
                    ('target_selector', 'role:coffee_cup_saucer_stack'),
                    ('postcondition', 'settled_below_motion_thresholds'),
                    ('measurement_name', 'stack_stability'),
                    ('window_steps', 30),
                    ('linear_velocity_threshold_mps', 0.02),
                    ('angular_velocity_threshold_rps', 0.05),
                ),
            ),
            recovery="replace an unstable item and restart the stability timer",
        ),
        _final_phase("coffee", 7, task_name),
    )
    return HardTrajectorySpec(
        task_name=task_name,
        variant="hard_v1",
        title="Coffee-station setup with support, route, and memory checks",
        goal_leaf_templates=get_task_spec(task_name).goal_leaf_templates,
        official_canonical_goal=get_task_spec(task_name).official_canonical_goal,
        phases=phases,
    )


def _dishes_manifest() -> HardTrajectorySpec:
    task_name = "putting_dishes_away_after_cleaning"
    phases = (
        _phase(
            phase_id="dishes_phase_01",
            ordinal=0,
            objective="Choose one valid destination cabinet before distributing plates.",
            trigger=ProbeTrigger(TriggerKind.EPISODE_START),
            probe=_probe(
                probe_id="dishes_probe_01",
                family=ProbeFamily.ACTION_ORDER_INFERENCE,
                prompt="Which plan satisfies the shared-cabinet quantifier and final closed-state requirement?",
                options=_opts("plan A: bind one cabinet, store every plate there, then close it", "plan B: bind a destination independently for each plate and close each destination after use"),
                correct=1,
                tiers=S,
                source="official existential-cabinet and universal-plate goal structure",
                verifier="one cabinet is bound consistently for all plate Inside predicates before Close",
            ),
            recovery="retain a single destination binding and reopen it if closed prematurely",
        ),
        _phase(
            phase_id="dishes_phase_02",
            ordinal=1,
            objective="Inspect a closed destination structure before placing objects.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.OPEN, target_prefixes=("cabinet.n.01_",)),
            probe=_probe(
                probe_id="dishes_probe_02",
                family=ProbeFamily.STRUCTURAL_ENCLOSURE,
                prompt="How many labeled plate-storage regions are reported after opening the selected cabinet?",
                options=_opts("one region", "zero regions", "two regions"),
                tiers=SV,
                source="cabinet open state, reveal links, and storage-region metadata",
                verifier="the selected cabinet exposes exactly one validated interior placement region",
                runtime_grounded=True,
            ),
            intervention=_intervention(
                intervention_id="dishes_intervention_02",
                status=InterventionStatus.RUNTIME_NATIVE,
                notice="The interior is observable only after the policy opens the selected cabinet.",
                tiers=SV,
                adapter_key=InterventionAdapterKey.OBSERVE_RUNTIME,
                operation="observe reveal links and storage regions created by the open action",
                recovery="reopen and reacquire the same cabinet if the interior is not visible",
                parameters=(
                    ('target_selector', 'prefix:cabinet.n.01_'),
                    ('postcondition', 'runtime_measurement_frozen'),
                    ('measurement_name', 'revealed_storage_region_count'),
                    ('window_steps', 1),
                ),
            ),
            recovery="do not place plates until the selected cabinet interior is verified",
        ),
        _phase(
            phase_id="dishes_phase_03",
            ordinal=2,
            objective="Maintain progress while visiting plates on separated counters.",
            trigger=ProbeTrigger(TriggerKind.BEFORE_ACTION, action=ActionName.NAVIGATE_TO, target_prefixes=("plate.n.04_", "cabinet.n.01_")),
            probe=_probe(
                probe_id="dishes_probe_03",
                family=ProbeFamily.LONG_TERM_NAVIGATION,
                prompt="Which endpoint-selection policy satisfies the per-plate state invariant across cabinet trips?",
                options=_opts("policy A: derive each endpoint from the current per-plate state table", "policy B: replay the episode-start endpoint itinerary after every cabinet visit"),
                correct=1,
                tiers=S,
                source="scene route graph and per-plate pending/held/contained state",
                verifier="each route terminates at a currently pending plate or the bound cabinet",
            ),
            recovery="recompute the next endpoint from the current per-plate state table",
        ),
        _phase(
            phase_id="dishes_phase_04",
            ordinal=3,
            objective="Account for door geometry while carrying a plate.",
            trigger=ProbeTrigger(TriggerKind.BEFORE_ACTION, action=ActionName.NAVIGATE_TO, occurrence=2, target_prefixes=("plate.n.04_", "cabinet.n.01_")),
            probe=_probe(
                probe_id="dishes_probe_04",
                family=ProbeFamily.TRAVERSABLE_PASSAGE,
                prompt="Which labeled cabinet approach passes the robot-plus-plate swept-volume check?",
                options=_opts("approach A (direct marker)", "approach B (side marker)"),
                correct=2,
                tiers=SP,
                source="open-door joint state and carried-object swept volume",
                verifier="approach A intersects the door and approach B passes the swept-volume check",
            ),
            intervention=_intervention(
                intervention_id="dishes_intervention_04",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="The selected cabinet door narrows the direct approach while open.",
                tiers=SP,
                adapter_key=InterventionAdapterKey.NARROW_PASSAGE,
                operation="validate an open angle that blocks the direct route but leaves a safe alternate approach",
                recovery="close or reposition the door only if the alternate approach is infeasible",
                parameters=(
                    ('target_selector', 'role:robot_with_held_plate'),
                    ('postcondition', 'alternate_route_traversable'),
                    ('passage_selector', 'role:selected_cabinet_approach'),
                    ('blocker_selector', 'prefix:cabinet.n.01_'),
                    ('min_clearance_m', 0.1),
                    ('alternate_route_selector', 'role:selected_cabinet_side_approach'),
                ),
                seed=1404,
            ),
            recovery="retreat and approach the cabinet through the swept-volume-safe side",
        ),
        _phase(
            phase_id="dishes_phase_05",
            ordinal=4,
            objective="Evaluate full-volume containment at the cabinet threshold.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.PLACE_INSIDE, settle_steps=12, target_prefixes=("cabinet.n.01_",)),
            probe=_probe(
                probe_id="dishes_probe_05",
                family=ProbeFamily.RIGID_CONTAINMENT,
                prompt="Is the plate fully inside the cabinet interior rather than crossing its threshold?",
                options=_opts("yes, the full plate volume is inside", "no, part of the plate remains outside"),
                correct=2,
                tiers=SVP,
                source="seeded threshold pose, official Inside state, and strict full-volume containment",
                verifier="one or more plate volume samples remain outside the cabinet interior",
            ),
            intervention=_intervention(
                intervention_id="dishes_intervention_05",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="A seeded evaluator-controlled placement perturbation is applied before verification.",
                tiers=SVP,
                adapter_key=InterventionAdapterKey.CONTAINMENT_NEAR_MISS,
                operation="place one plate across the cabinet threshold without satisfying strict containment",
                recovery="return that plate to the gripper for a deeper placement",
                parameters=(
                    ('target_selector', 'prefix:plate.n.04_'),
                    ('postcondition', 'strict_containment_false'),
                    ('container_selector', 'prefix:cabinet.n.01_'),
                    ('protrusion_m', 0.05),
                ),
                seed=1405,
            ),
            recovery="re-grasp and place the plate farther into the verified storage region",
        ),
        _phase(
            phase_id="dishes_phase_06",
            ordinal=5,
            objective="Challenge the threshold judgment with a misleading view.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.PLACE_INSIDE, settle_steps=12, target_prefixes=("cabinet.n.01_",)),
            probe=_probe(
                probe_id="dishes_probe_06",
                family=ProbeFamily.VIEW_HALLUCINATION,
                prompt="The plate looks enclosed from the front. What does the side/depth view establish?",
                options=_opts("the plate is fully beyond the cabinet threshold", "the plate still crosses the threshold"),
                correct=2,
                tiers=VP,
                source="paired front and side RGB-D views of the seeded threshold pose",
                verifier="side depth exposes plate geometry outside the cabinet interior",
            ),
            intervention=_intervention(
                intervention_id="dishes_intervention_06",
                status=InterventionStatus.OBSERVATION_ONLY,
                notice="The first verification view is intentionally ambiguous; a side view is available.",
                tiers=V,
                adapter_key=InterventionAdapterKey.MISLEADING_VIEW,
                operation="select a frontal camera pose where the plate projects within the cabinet opening",
                recovery="provide a collision-free side view with depth",
                parameters=(
                    ('target_selector', 'prefix:plate.n.04_'),
                    ('postcondition', 'multi_view_containment_resolved'),
                    ('initial_view_id', 'dishes_cabinet_front'),
                    ('verification_view_id', 'dishes_cabinet_side_depth'),
                    ('min_projected_overlap_fraction', 0.7),
                ),
            ),
            recovery="use the side view, then apply the containment recovery from the prior checkpoint",
        ),
        _phase(
            phase_id="dishes_phase_07",
            ordinal=6,
            objective="Evaluate the latest plate after a physical settle window.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.PLACE_INSIDE, occurrence=2, settle_steps=30, target_prefixes=("cabinet.n.01_",)),
            probe=_probe(
                probe_id="dishes_probe_07",
                family=ProbeFamily.STACKING_AND_STABILITY,
                prompt="After settling, is the latest plate stably supported without sliding into another plate?",
                options=_opts("yes", "no"),
                tiers=SP,
                source="support contacts, velocities, tilt, and inter-plate collision state",
                verifier="the plate remains supported and below motion thresholds for 30 steps",
                runtime_grounded=True,
            ),
            intervention=_intervention(
                intervention_id="dishes_intervention_07",
                status=InterventionStatus.RUNTIME_NATIVE,
                notice="Placement is accepted only after a physical settle window.",
                tiers=SP,
                adapter_key=InterventionAdapterKey.SETTLE_ONLY,
                operation="advance simulation and measure delayed plate motion and contacts",
                recovery="remove and re-place the unstable plate in an unoccupied interior region",
                parameters=(
                    ('target_selector', 'role:latest_placed_plate'),
                    ('postcondition', 'settled_below_motion_thresholds'),
                    ('measurement_name', 'plate_stability'),
                    ('window_steps', 30),
                    ('linear_velocity_threshold_mps', 0.015),
                    ('angular_velocity_threshold_rps', 0.04),
                ),
            ),
            recovery="re-place unstable plates and repeat the settle window",
        ),
        _phase(
            phase_id="dishes_phase_08",
            ordinal=7,
            objective="Compare remembered and current plate/cabinet state after a revisit.",
            trigger=_revisit_trigger(
                anchor=RevisitAnchor(RevisitAnchorKind.ROLE, "bound_destination_cabinet")
            ),
            probe=_probe(
                probe_id="dishes_probe_08",
                family=ProbeFamily.UNOBSERVED_CHANGE,
                prompt="Which state report matches the before/after plate and cabinet snapshots?",
                options=_opts("report A: one prior Inside predicate is now false", "report B: all prior plate and cabinet predicates retain their values"),
                correct=1,
                tiers=SVP,
                source="before/after Inside and Open snapshots for the bound cabinet and scoped plates",
                verifier="exactly one previously contained plate transitions to not Inside",
            ),
            intervention=_intervention(
                intervention_id="dishes_intervention_08",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="A seeded evaluator intervention may or may not alter prior storage progress while unobserved.",
                tiers=SVP,
                adapter_key=InterventionAdapterKey.DISTURB_RELATION,
                operation="move one stored plate to a stable pose outside the cabinet while the robot is away",
                recovery="mark only that plate pending and keep the destination cabinet binding",
                parameters=(
                    ('target_selector', 'prefix:plate.n.04_'),
                    ('postcondition', 'goal_relation_changed'),
                    ('relation', 'inside(prefix:plate.n.04_,prefix:cabinet.n.01_)'),
                    ('impulse_ns', 0.0),
                    ('displacement_m', 0.25),
                ),
                seed=1408,
            ),
            recovery="recover the affected plate, verify every plate Inside, then close the cabinet",
        ),
        _final_phase("dishes", 8, task_name),
    )
    return HardTrajectorySpec(
        task_name=task_name,
        variant="hard_v1",
        title="Dish storage with enclosure, passage, and stability checks",
        goal_leaf_templates=get_task_spec(task_name).goal_leaf_templates,
        official_canonical_goal=get_task_spec(task_name).official_canonical_goal,
        phases=phases,
    )


def _wood_manifest() -> HardTrajectorySpec:
    task_name = "bringing_in_wood"
    phases = (
        _phase(
            phase_id="wood_phase_01",
            ordinal=0,
            objective="Inventory all scoped plywood despite garden occlusion.",
            trigger=ProbeTrigger(TriggerKind.EPISODE_START),
            probe=_probe(
                probe_id="wood_probe_01",
                family=ProbeFamily.PARTIAL_OCCLUSION,
                prompt="How many scoped plywood pieces are present after checking behind the garden occluder?",
                options=_opts("one", "two", "three"),
                correct=3,
                tiers=SV,
                source="BDDL object scope and seeded multi-view visibility",
                verifier="multi-view instance association finds all three scoped plywood instances",
            ),
            intervention=_intervention(
                intervention_id="wood_intervention_01",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="One plywood piece is partly hidden by seeded garden clutter without changing its floor support.",
                tiers=SV,
                adapter_key=InterventionAdapterKey.OCCLUDE_WITH_DISTRACTOR,
                operation="position non-task garden furniture to occlude one plywood while preserving OnTop garden floor",
                recovery="remove only the occluder if no collision-free inspection view exists",
                parameters=(
                    ('target_selector', 'prefix:plywood.n.01_'),
                    ('postcondition', 'visibility_fraction_between_bounds'),
                    ('occluder_selector', 'role:seeded_garden_furniture'),
                    ('initial_view_id', 'wood_garden_initial'),
                    ('verification_view_id', 'wood_garden_occluder_side'),
                    ('min_visibility_fraction', 0.1),
                    ('max_visibility_fraction', 0.5),
                ),
                seed=1501,
            ),
            recovery="circle the occluder and complete the three-instance inventory",
        ),
        _phase(
            phase_id="wood_phase_02",
            ordinal=1,
            objective="Choose an order that keeps later corridor placements reachable.",
            trigger=ProbeTrigger(TriggerKind.BEFORE_ACTION, action=ActionName.PICK_UP, target_prefixes=("plywood.n.01_",)),
            probe=_probe(
                probe_id="wood_probe_02",
                family=ProbeFamily.ACTION_ORDER_INFERENCE,
                prompt="Which labeled drop-zone sequence preserves a feasible path for every remaining delivery?",
                options=_opts("sequence A: far zone → middle zone → near zone", "sequence B: near zone → middle zone → far zone"),
                tiers=SP,
                source="corridor occupancy plan and carried-plywood swept-volume feasibility",
                verifier="evaluate both labeled sequences against current occupancy and select the one preserving every remaining delivery path",
                runtime_grounded=True,
            ),
            recovery="reserve the doorway and nearest drop zone until later deliveries are complete",
        ),
        _phase(
            phase_id="wood_phase_03",
            ordinal=2,
            objective="Verify a large-object grasp after lift.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.PICK_UP, settle_steps=12, target_prefixes=("plywood.n.01_",)),
            probe=_probe(
                probe_id="wood_probe_03",
                family=ProbeFamily.PHYSICAL_CONTACT,
                prompt="Does the plywood remain physically secured after lifting and settling?",
                options=_opts("yes", "no"),
                tiers=SP,
                source="gripper contacts, held-object binding, and relative-pose persistence",
                verifier="the scoped plywood remains attached through the full settle interval",
                runtime_grounded=True,
            ),
            intervention=_intervention(
                intervention_id="wood_intervention_03",
                status=InterventionStatus.RUNTIME_NATIVE,
                notice="The policy-created grasp is verified after a settle window.",
                tiers=SP,
                adapter_key=InterventionAdapterKey.OBSERVE_RUNTIME,
                operation="observe attachment and contact state during the executed lift",
                recovery="lower safely, release, and retry with a more central grasp",
                parameters=(
                    ('target_selector', 'prefix:plywood.n.01_'),
                    ('postcondition', 'runtime_measurement_frozen'),
                    ('measurement_name', 'held_attachment'),
                    ('window_steps', 12),
                ),
            ),
            recovery="retry from a central grasp when attachment or contact is lost",
        ),
        _phase(
            phase_id="wood_phase_04",
            ordinal=3,
            objective="Account for plywood orientation at the garden-to-corridor passage.",
            trigger=ProbeTrigger(TriggerKind.BEFORE_ACTION, action=ActionName.NAVIGATE_TO, occurrence=2, target_instances=("floor.n.01_2",)),
            probe=_probe(
                probe_id="wood_probe_04",
                family=ProbeFamily.TRAVERSABLE_PASSAGE,
                prompt="Which transport configuration passes the carried-plywood swept-volume check?",
                options=_opts("configuration A: retain the transverse orientation", "configuration B: rotate to longitudinal at the staging zone"),
                correct=2,
                tiers=SP,
                source="doorway geometry and robot-plus-plywood swept volume",
                verifier="configuration A intersects the doorway envelope and configuration B passes the swept-volume check",
            ),
            intervention=_intervention(
                intervention_id="wood_intervention_04",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="Doorway feasibility is evaluated with the held plywood included in the footprint.",
                tiers=SP,
                adapter_key=InterventionAdapterKey.CARRIED_ORIENTATION,
                operation="present the first carried plywood in a validated non-traversable sideways orientation",
                recovery="return to the staging zone and use guarded turn_left/turn_right steps until longitudinal; the adapter must check held-object swept volume",
                parameters=(
                    ('target_selector', 'prefix:plywood.n.01_'),
                    ('postcondition', 'longitudinal_swept_volume_clear'),
                    ('passage_selector', 'role:garden_to_corridor_doorway'),
                    ('initial_orientation', 'sideways'),
                    ('required_orientation', 'longitudinal'),
                    ('carried_clearance_m', 0.08),
                ),
                seed=1504,
            ),
            recovery="retreat to the staging zone, use existing guarded base turns until longitudinal, then retry the passage",
        ),
        _phase(
            phase_id="wood_phase_05",
            ordinal=4,
            objective="Evaluate the official support goal from doorway and side views.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.NAVIGATE_TO, occurrence=2, target_instances=("floor.n.01_2",)),
            probe=_probe(
                probe_id="wood_probe_05",
                family=ProbeFamily.VIEW_HALLUCINATION,
                prompt="The front edge crossed the doorway. Does the plywood already satisfy OnTop(corridor floor)?",
                options=_opts("OnTop(plywood, corridor_floor) is true", "OnTop(plywood, corridor_floor) is false"),
                correct=2,
                tiers=SVP,
                source="side/depth observation and official OnTop(plywood, corridor_floor) state",
                verifier="the plywood remains held or supported by the garden side and OnTop corridor floor is false",
            ),
            intervention=_intervention(
                intervention_id="wood_intervention_05",
                status=InterventionStatus.OBSERVATION_ONLY,
                notice="The initial doorway view is intentionally misleading; side/depth verification is available.",
                tiers=V,
                adapter_key=InterventionAdapterKey.MISLEADING_VIEW,
                operation="select a frontal view where the plywood front edge projects across the doorway",
                recovery="provide a side view and the official predicate state after answer submission",
                parameters=(
                    ('target_selector', 'prefix:plywood.n.01_'),
                    ('postcondition', 'multi_view_support_resolved'),
                    ('initial_view_id', 'wood_doorway_front'),
                    ('verification_view_id', 'wood_doorway_side_depth'),
                    ('min_projected_overlap_fraction', 0.6),
                ),
            ),
            recovery="continue carrying to a corridor drop zone; do not count projected threshold crossing",
        ),
        _phase(
            phase_id="wood_phase_06",
            ordinal=5,
            objective="Verify that each plywood piece settles directly on corridor floor.",
            trigger=ProbeTrigger(TriggerKind.AFTER_ACTION, action=ActionName.PLACE_ON_TOP, settle_steps=30, target_instances=("floor.n.01_2",)),
            probe=_probe(
                probe_id="wood_probe_06",
                family=ProbeFamily.STACKING_AND_STABILITY,
                prompt="After settling, is this plywood flat and independently supported by corridor floor?",
                options=_opts("yes", "no, it tilted, slid, or rests on another plywood"),
                tiers=SP,
                source="OnTop state, support contacts, tilt, velocity, and plywood-plywood contacts",
                verifier="the piece remains directly OnTop corridor floor and below stability thresholds",
                runtime_grounded=True,
            ),
            intervention=_intervention(
                intervention_id="wood_intervention_06",
                status=InterventionStatus.RUNTIME_NATIVE,
                notice="Each drop is judged only after physical settling.",
                tiers=SP,
                adapter_key=InterventionAdapterKey.SETTLE_ONLY,
                operation="advance simulation and measure direct floor support and delayed motion",
                recovery="re-grasp and place the piece flat in an unoccupied corridor zone",
                parameters=(
                    ('target_selector', 'role:latest_placed_plywood'),
                    ('postcondition', 'settled_below_motion_thresholds'),
                    ('measurement_name', 'plywood_floor_stability'),
                    ('window_steps', 30),
                    ('linear_velocity_threshold_mps', 0.02),
                    ('angular_velocity_threshold_rps', 0.04),
                ),
            ),
            recovery="re-place any unstable or stacked piece directly on corridor floor",
        ),
        _phase(
            phase_id="wood_phase_07",
            ordinal=6,
            objective="Maintain instance state across repeated garden-corridor trips.",
            trigger=_revisit_trigger(
                anchor=RevisitAnchor(RevisitAnchorKind.ROLE, "active_pending_plywood")
            ),
            probe=_probe(
                probe_id="wood_probe_07",
                family=ProbeFamily.LONG_TERM_NAVIGATION,
                prompt="Which labeled route endpoint is selected by the current per-plywood state table?",
                options=_opts("endpoint A (garden plywood track)", "endpoint B (corridor plywood track)"),
                tiers=S,
                source="scene route graph and per-plywood pending/held/placed state",
                verifier="route endpoint binds a scoped instance whose corridor-floor goal is still false",
                runtime_grounded=True,
            ),
            recovery="replan from the pending instance set instead of replaying the previous endpoint",
        ),
        _phase(
            phase_id="wood_phase_08",
            ordinal=7,
            objective="Compare remembered and current delivery state after a revisit.",
            trigger=_revisit_trigger(
                anchor=RevisitAnchor(RevisitAnchorKind.INSTANCE, "floor.n.01_2"),
                occurrence=2,
            ),
            probe=_probe(
                probe_id="wood_probe_08",
                family=ProbeFamily.UNOBSERVED_CHANGE,
                prompt="Which state report matches the before/after plywood support snapshots?",
                options=_opts("report A: one prior OnTop(corridor_floor) predicate is now false", "report B: all prior plywood support predicates retain their values"),
                correct=1,
                tiers=SVP,
                source="before/after OnTop snapshots, poses, and settled support contacts",
                verifier="exactly one delivered plywood transitions from OnTop corridor floor to false",
            ),
            intervention=_intervention(
                intervention_id="wood_intervention_08",
                status=InterventionStatus.DECLARED_REQUIRES_ADAPTER,
                notice="A seeded evaluator intervention may or may not alter a prior delivery while unobserved.",
                tiers=SVP,
                adapter_key=InterventionAdapterKey.DISTURB_RELATION,
                operation="apply a bounded impulse that slides one delivered plywood back to the doorway threshold",
                recovery="mark that plywood pending and preserve other verified deliveries",
                parameters=(
                    ('target_selector', 'prefix:plywood.n.01_'),
                    ('postcondition', 'goal_relation_changed'),
                    ('relation', 'ontop(prefix:plywood.n.01_,instance:floor.n.01_2)'),
                    ('impulse_ns', 1.2),
                    ('displacement_m', 0.3),
                ),
                seed=1508,
            ),
            recovery="restore the invalidated plywood before completing remaining deliveries",
        ),
        _final_phase("wood", 8, task_name),
    )
    return HardTrajectorySpec(
        task_name=task_name,
        variant="hard_v1",
        title="Wood delivery with carried-object passage, memory, and support checks",
        goal_leaf_templates=get_task_spec(task_name).goal_leaf_templates,
        official_canonical_goal=get_task_spec(task_name).official_canonical_goal,
        phases=phases,
    )


HARD_V1_TRAJECTORIES: dict[str, HardTrajectorySpec] = {
    spec.task_name: spec
    for spec in (
        _radio_manifest(),
        _trash_manifest(),
        _coffee_manifest(),
        _dishes_manifest(),
        _wood_manifest(),
    )
}


# Fail fast if a future edit makes a policy-facing or evaluator-facing manifest invalid.
validate_hard_trajectory_registry()

