"""Single private owner for one selected official grounded option.

The same evaluator can capture multiple reset snapshots and then enter the
Oracle.  Exponentially large official option tables must not be independently
searched at each boundary.  This module freezes one actual official option,
stores it on the evaluator, and identity-attests that exact option on every
reuse without enumerating the table.

The owner is evaluator-private.  Neither its option bodies, object IDs, index,
nor identity may enter a model observation or public action result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from behavior_hl.embodied.evaluator_bridge.inside_goal_graph import (
    compile_grounded_inside_option,
    grounded_goal_option_identity,
)
from behavior_hl.embodied.evaluator_bridge.executable_grounded_option_admission import (
    build_action23_executable_admission,
)
from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (
    InsideTaskProfile,
    get_inside_task_profile,
)
from behavior_hl.embodied.evaluator_bridge.lazy_grounded_option_solver import (
    MAXIMUM_LAZY_ASSIGNMENT_CANDIDATES,
    LazyGroundedOptionWitness,
    attest_indexed_official_option_witness,
    attest_official_option_witness,
)
from behavior_hl.embodied.evaluator_bridge.mixed_goal_graph import (
    GroundedMixedGoalPlan,
    compile_grounded_mixed_option,
)
from behavior_hl.embodied.evaluator_bridge.official_actual_option_mapping import (
    OfficialActualOptionMapping,
    OfficialActualOptionMappingError,
    load_official_actual_option_mapping,
)
from behavior_hl.embodied.policy.profiles import build_action23_registry


GROUNDED_OPTION_IDENTITY_OWNER_VERSION = (
    "behavior2026.grounded-option-identity-owner/v1"
)
_OWNER_ATTRIBUTE = "_behavior_hl_grounded_option_identity_owner_v1"


class GroundedOptionIdentityOwnerError(RuntimeError):
    """The selected official option cannot be frozen or safely reused."""


def _official_task(evaluator: Any) -> Any:
    task = getattr(getattr(evaluator, "env", None), "task", None)
    if task is None:
        raise GroundedOptionIdentityOwnerError(
            "evaluator does not expose one official task"
        )
    return task


def _official_options(task: Any) -> Sequence[Sequence[Any]]:
    options = getattr(task, "ground_goal_state_options", None)
    if (
        not isinstance(options, Sequence)
        or isinstance(options, (str, bytes))
        or len(options) < 1
    ):
        raise GroundedOptionIdentityOwnerError(
            "official grounded goal options are unavailable"
        )
    return options


def _openable_scope_names(task: Any) -> frozenset[str]:
    """Return concrete scope IDs exposing exactly one official Open state."""

    scope = getattr(task, "object_scope", None)
    if not isinstance(scope, Mapping):
        return frozenset()
    result: set[str] = set()
    for scope_name, entity in scope.items():
        states = getattr(entity, "states", None)
        if not isinstance(states, Mapping):
            continue
        matches = tuple(
            state
            for state_type, state in states.items()
            if getattr(state_type, "__name__", "") == "Open"
        )
        if len(matches) > 1:
            raise GroundedOptionIdentityOwnerError(
                "live object exposes multiple Open state entries"
            )
        if not matches:
            continue
        if not isinstance(scope_name, str) or not scope_name:
            raise GroundedOptionIdentityOwnerError(
                "openable scope entry has an invalid official object ID"
            )
        result.add(scope_name)
    return frozenset(result)


def _immovable_scope_names(task: Any) -> frozenset[str]:
    """Return live scope IDs whose official object has a fixed base."""

    scope = getattr(task, "object_scope", None)
    if not isinstance(scope, Mapping):
        return frozenset()
    result: set[str] = set()
    for scope_name, entity in scope.items():
        fixed = getattr(entity, "fixed_base", None)
        if fixed is None:
            fixed = getattr(entity, "_fixed_base", False)
        if bool(fixed):
            if not isinstance(scope_name, str) or not scope_name:
                raise GroundedOptionIdentityOwnerError(
                    "immovable scope entry has an invalid official object ID"
                )
            result.add(scope_name)
    return frozenset(result)


def _candidate_compiler(
    *,
    task: Any,
    profile: InsideTaskProfile,
):
    openable_targets = _openable_scope_names(task)
    immovable_targets = _immovable_scope_names(task)
    if any(
        signature.startswith("Mixed")
        for signature in profile.goal_signatures
    ):
        return lambda option, option_index: compile_grounded_mixed_option(
            option,
            option_index=option_index,
            openable_targets=openable_targets,
            immovable_targets=immovable_targets,
        )
    return lambda option, option_index: compile_grounded_inside_option(
        option,
        option_index=option_index,
        openable_targets=openable_targets,
    )


def _require_executable_witness(
    witness: LazyGroundedOptionWitness,
) -> None:
    """Fail closed unless the frozen official option is Action23-executable."""

    admission = build_action23_executable_admission(
        build_action23_registry()
    )
    accepted, reason = admission.compiled_candidate_gate(
        witness.option,
        witness.compiled_plan,
    )
    if not accepted:
        raise GroundedOptionIdentityOwnerError(
            "frozen official option is structurally grounded but not "
            f"executable by the active action registry: {reason}"
        )


@dataclass(frozen=True, slots=True)
class GroundedOptionIdentityOwner:
    """One selected actual official option shared by reset and Oracle."""

    activity_name: str
    task_object_identity: int
    official_option_count: int
    witness: LazyGroundedOptionWitness
    selection_mode: str
    catalog_identity_match: bool
    actual_index_mapping: OfficialActualOptionMapping | None = None
    schema_version: str = GROUNDED_OPTION_IDENTITY_OWNER_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GROUNDED_OPTION_IDENTITY_OWNER_VERSION:
            raise ValueError("unexpected grounded option owner version")
        if not self.activity_name:
            raise ValueError("grounded option owner activity name is empty")
        if self.official_option_count < 1:
            raise ValueError("grounded option owner option count is invalid")
        if not isinstance(self.witness, LazyGroundedOptionWitness):
            raise TypeError("grounded option owner witness is invalid")
        if not self.witness.official_identity_attested:
            raise ValueError("grounded option owner witness is not attested")
        if self.selection_mode not in {
            "legacy_canonical_identity",
            "frozen_official_actual_index_identity",
        }:
            raise ValueError("grounded option owner selection mode is invalid")
        if (
            self.selection_mode == "frozen_official_actual_index_identity"
        ) != (self.actual_index_mapping is not None):
            raise ValueError(
                "grounded option owner actual-index mapping is inconsistent"
            )

    def require_matches(
        self,
        task: Any,
        *,
        allow_task_object_rebind: bool = False,
    ) -> None:
        """Re-attest the frozen index by actual identity in O(1)."""

        if (
            id(task) != self.task_object_identity
            and not allow_task_object_rebind
        ):
            raise GroundedOptionIdentityOwnerError(
                "grounded option owner belongs to another official task"
            )
        activity_name = str(getattr(task, "activity_name", ""))
        if activity_name != self.activity_name:
            raise GroundedOptionIdentityOwnerError(
                "grounded option owner activity changed"
            )
        options = _official_options(task)
        if len(options) != self.official_option_count:
            raise GroundedOptionIdentityOwnerError(
                "official option count changed after owner freeze"
            )
        index = self.witness.option_index
        if index >= len(options):
            raise GroundedOptionIdentityOwnerError(
                "frozen official option index is out of bounds"
            )
        try:
            actual_identity = grounded_goal_option_identity(options[index])
        except Exception as exc:
            raise GroundedOptionIdentityOwnerError(
                "frozen official option identity is unreadable"
            ) from exc
        if actual_identity != self.witness.identity:
            raise GroundedOptionIdentityOwnerError(
                "frozen official option identity changed after owner freeze"
            )

    def compile_live_mixed_plan(
        self,
        *,
        task: Any,
        initially_true_relations: Iterable[Any],
        source_openable_ancestors_by_subject: Mapping[str, Iterable[str]],
        precedence_relations=(),
    ) -> GroundedMixedGoalPlan:
        """Refresh execution prerequisites without selecting a new witness.

        The caller supplies current canonical truth and live containment
        evidence.  The cached witness attests semantic identity, not a plan
        valid forever after reset or state changes.  Keep it immutable and
        compile all its original goals, including already satisfied ones.
        """

        self.require_matches(task)
        if not isinstance(self.witness.compiled_plan, GroundedMixedGoalPlan):
            raise GroundedOptionIdentityOwnerError(
                "live mixed compilation requires a mixed goal witness"
            )
        return compile_grounded_mixed_option(
            self.witness.option,
            option_index=self.witness.option_index,
            openable_targets=_openable_scope_names(task),
            immovable_targets=_immovable_scope_names(task),
            initially_true_relations=initially_true_relations,
            source_openable_ancestors_by_subject=(
                source_openable_ancestors_by_subject
            ),
            precedence_relations=precedence_relations,
        )

    def private_audit_record(self) -> dict[str, Any]:
        record = self.witness.private_audit_record(
            raw_option_count=self.official_option_count
        )
        record.update(
            {
                "owner_schema_version": self.schema_version,
                "selection_mode": self.selection_mode,
                "catalog_identity_match": self.catalog_identity_match,
                "single_identity_owner": True,
                "identity_reattested_on_reuse": True,
                "official_task_success_remains_global": True,
            }
        )
        if self.actual_index_mapping is not None:
            record["official_actual_index_mapping"] = (
                self.actual_index_mapping.private_audit_record()
            )
        return record


def resolve_grounded_option_identity_owner(
    *,
    evaluator: Any,
    profile: InsideTaskProfile | None = None,
) -> GroundedOptionIdentityOwner:
    """Return the evaluator's only selected-option owner.

    Small tables preserve the legacy canonical identity selector.  Large
    tables freeze one exact official indexed option and its *actual* identity;
    they never scan for a catalog identity whose raw table index is unstable.
    """

    task = _official_task(evaluator)
    activity_name = str(getattr(task, "activity_name", ""))
    existing = getattr(evaluator, _OWNER_ATTRIBUTE, None)
    if isinstance(existing, GroundedOptionIdentityOwner):
        if existing.activity_name == activity_name:
            existing.require_matches(
                task,
                allow_task_object_rebind=True,
            )
            if id(task) == existing.task_object_identity:
                return existing
            rebound = GroundedOptionIdentityOwner(
                activity_name=existing.activity_name,
                task_object_identity=id(task),
                official_option_count=existing.official_option_count,
                witness=existing.witness,
                selection_mode=existing.selection_mode,
                catalog_identity_match=existing.catalog_identity_match,
                actual_index_mapping=existing.actual_index_mapping,
            )
            setattr(evaluator, _OWNER_ATTRIBUTE, rebound)
            return rebound
    if profile is None:
        try:
            profile = get_inside_task_profile(activity_name)
        except KeyError as exc:
            raise GroundedOptionIdentityOwnerError(
                "official task has no registered grounded goal profile"
            ) from exc
    elif profile.task_name != activity_name:
        raise GroundedOptionIdentityOwnerError(
            "grounded option profile does not match the live task"
        )
    options = _official_options(task)
    try:
        actual_index_mapping = load_official_actual_option_mapping(
            activity_name,
            # Runtime correctness is owned by the exact live option count,
            # actual index, and semantic-identity checks below.  Requiring all
            # authoring-source hashes to remain byte-identical turns unrelated
            # implementation edits into scene-start failures without adding a
            # semantic safety guarantee.
            verify_provenance=False,
        )
    except OfficialActualOptionMappingError as exc:
        raise GroundedOptionIdentityOwnerError(
            "official actual-index mapping provenance failed closed"
        ) from exc
    preferred_index = (
        profile.canonical_live_option_index
        if profile.canonical_live_option_index is not None
        else profile.canonical_option_index
    )
    compiler = _candidate_compiler(task=task, profile=profile)
    try:
        if actual_index_mapping is not None:
            if len(options) != actual_index_mapping.official_option_count:
                raise GroundedOptionIdentityOwnerError(
                    "live official option count does not match frozen mapping"
                )
            witness = attest_indexed_official_option_witness(
                official_options=options,
                selected_option_index=(
                    actual_index_mapping.actual_option_index
                ),
                compile_candidate=compiler,
            )
            if witness.identity != actual_index_mapping.semantic_identity:
                raise GroundedOptionIdentityOwnerError(
                    "frozen actual index does not carry the authored "
                    "semantic identity"
                )
            selection_mode = "frozen_official_actual_index_identity"
        elif len(options) > MAXIMUM_LAZY_ASSIGNMENT_CANDIDATES:
            raise GroundedOptionIdentityOwnerError(
                "large official goal table has no authored actual-index "
                "mapping; CPU lazy raw indices are not live indices"
            )
        else:
            witness = attest_official_option_witness(
                official_options=options,
                canonical_goal_leaves=profile.canonical_goal_leaves,
                preferred_option_index=preferred_index,
                compile_candidate=compiler,
                allow_bounded_fallback=(
                    profile.canonical_live_option_index is None
                ),
            )
            selection_mode = "legacy_canonical_identity"
    except Exception as exc:
        raise GroundedOptionIdentityOwnerError(
            "official grounded option owner could not freeze one feasible "
            "identity"
        ) from exc
    _require_executable_witness(witness)
    owner = GroundedOptionIdentityOwner(
        activity_name=activity_name,
        task_object_identity=id(task),
        official_option_count=len(options),
        witness=witness,
        selection_mode=selection_mode,
        catalog_identity_match=(
            witness.identity == tuple(sorted(profile.canonical_goal_leaves))
        ),
        actual_index_mapping=actual_index_mapping,
    )
    owner.require_matches(task)
    setattr(evaluator, _OWNER_ATTRIBUTE, owner)
    return owner
