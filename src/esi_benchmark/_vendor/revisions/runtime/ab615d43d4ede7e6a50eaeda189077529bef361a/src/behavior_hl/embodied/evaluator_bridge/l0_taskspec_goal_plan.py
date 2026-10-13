"""Materialize bounded L0 witnesses as existing Goal-DAG plans."""

from __future__ import annotations

import json
import itertools
import math
import re
from functools import cache, partial
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.action23_outcome_verifier import (
    Action23OutcomeVerifier,
)
from behavior_hl.embodied.evaluator_bridge.inside_goal_graph import (
    GroundedInsideGoalPlan,
    feasible_grounded_inside_plans,
    grounded_goal_option_identity,
)
from behavior_hl.embodied.evaluator_bridge.mixed_goal_graph import (
    GroundedMixedGoalPlan,
    MixedGoalGraphError,
    MixedGoalRelation,
    compile_grounded_mixed_option,
    feasible_grounded_mixed_plans,
    relation_from_grounded_leaf,
)
from behavior_hl.embodied.runtime.verification import PrivatePredicateSnapshot
from behavior_hl.embodied.runtime.episode import EpisodeSpec
from behavior_hl.embodied.runtime.prompt import ACTION20_PUBLIC_PROFILE


_AT_LEAST = re.compile(r"AT_LEAST_(\d+)(?:_TO_ONE_DESTINATION)?\Z")
_PREDICATE_BY_KIND = {
    "inside": "inside",
    "open": "open",
    "closed": "closed",
    "on_top": "ontop",
    "under": "under",
    "next_to": "nextto",
    "touching": "touching",
    "toggled_on": "toggled_on",
    "not_toggled_on": "toggled_off",
}
_MAXIMUM_PLAN_CANDIDATES = 256
_ROUTE_RANKING_PREFIX = 8


def _route_audit_blocks_candidate(
    audit: Mapping[str, Any], navigation_lane: str
) -> bool:
    return (
        navigation_lane != "se2_navigation_leg_v1"
        and audit.get("route_exists") is not True
    )


def _rank_projected_candidate_prefix(
    rows: Sequence[Any],
    audit_candidate: Callable[[Any], Mapping[str, Any]],
) -> tuple[list[tuple[int, Any]], dict[int, Mapping[str, Any]]]:
    """Spend at most eight advisory audits before the first public action."""

    indexed = list(enumerate(rows))
    prefix = indexed[:_ROUTE_RANKING_PREFIX]
    audits = {index: audit_candidate(row[0]) for index, row in prefix}

    def rank(item: tuple[int, Any]) -> tuple[bool, float, int]:
        index, _ = item
        report = audits[index]
        cost = report.get("total_move015_lower_bound")
        finite_cost = (
            float(cost)
            if isinstance(cost, (int, float)) and not isinstance(cost, bool)
            and math.isfinite(cost) and cost >= 0
            else math.inf
        )
        return (report.get("route_exists") is not True, finite_cost, index)

    return sorted(prefix, key=rank) + indexed[_ROUTE_RANKING_PREFIX:], audits


def _required_count(quantifier: str, target_count: int) -> int:
    if quantifier in {
        "ALL_LISTED",
        "ALL_LISTED_TO_ONE_DESTINATION",
        "ALL_TEMPLATE_ROOM_CATEGORY_MEMBERS",
    }:
        return target_count
    if quantifier in {"EXACTLY_1", "AT_LEAST_1_PUBLIC_EQUIVALENT_PAIR"}:
        return 1
    match = _AT_LEAST.fullmatch(quantifier)
    if match is None:
        raise ValueError(f"unsupported L0 quantifier: {quantifier!r}")
    required = int(match.group(1))
    if not 1 <= required <= target_count:
        raise ValueError(f"invalid L0 quantifier count: {quantifier!r}")
    return required


def _destinations(clause: Mapping[str, Any]) -> tuple[str | None, ...]:
    declared = tuple(clause.get("destination_scopes", ()))
    if clause.get("destination_equivalence_policy") == (
        "ANY_TEMPLATE_ROOM_CATEGORY_EQUIVALENT"
    ):
        declared = tuple(clause.get("destination_equivalence_members", declared))
    return declared or (None,)


def _leaf(predicate: str, target: str, destination: str | None) -> list[Any]:
    if predicate == "closed":
        return ["not", ["open", target]]
    if predicate == "toggled_off":
        return ["not", ["toggled_on", target]]
    if destination is None:
        return [predicate, target]
    return [predicate, target, destination]


def projected_witness_leaves(
    task_spec: Mapping[str, Any],
    truth_reader: Callable[[str, str, str | None], bool],
) -> tuple[list[Any], ...]:
    """Choose a complete deterministic witness, including preservation goals."""

    return projected_witness_leaf_options(task_spec, truth_reader)[0]


def projected_witness_leaf_options(
    task_spec: Mapping[str, Any],
    truth_reader: Callable[[str, str, str | None], bool],
    *,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[tuple[list[Any], ...], ...]:
    """Enumerate the bounded legal witness frontier without task special cases."""

    verifier = task_spec.get("private", {}).get("verifier", {})
    route = verifier.get("route")
    if route not in {"official_projection", "canonical_fallback"}:
        raise ValueError("projected witness requires a projected verifier route")
    clause_options: list[tuple[tuple[list[Any], ...], ...]] = []
    truncated = False
    for index, clause in enumerate(verifier.get("clauses", ())):
        predicate = clause.get("canonical_predicate")
        quantifier = clause.get("quantifier")
        targets = tuple(clause.get("target_scopes", ()))
        if not isinstance(predicate, str) or not isinstance(quantifier, str) or not targets:
            raise ValueError(f"malformed projected clause {index}")
        destinations = _destinations(clause)
        truth = {
            (target, destination): truth_reader(predicate, target, destination)
            for target in targets
            for destination in destinations
            if predicate not in {"nextto", "touching"} or target != destination
        }
        if any(not isinstance(value, bool) for value in truth.values()):
            raise TypeError("projected truth reader must return bool")
        required = _required_count(quantifier, len(targets))

        shared_destination = bool(
            quantifier.endswith("_TO_ONE_DESTINATION")
            or task_spec.get("private", {})
            .get("goal_spec", {})
            .get("v1_quantifier_policy", {})
            .get("shared_destination")
            == "EXISTS_ONE_SHARED_DESTINATION_FOR_ALL_TARGETS"
        )

        def target_sets(available: Sequence[str], matched: set[str]) -> Iterable[tuple[str, ...]]:
            preserved_count = min(required, len(matched))
            for preserved in itertools.combinations(
                (target for target in available if target in matched), preserved_count
            ):
                for missing in itertools.combinations(
                    (target for target in available if target not in matched),
                    required - preserved_count,
                ):
                    yield preserved + missing

        def clause_witnesses() -> Iterable[tuple[list[Any], ...]]:
            # Keep exactly the required witness, not every currently true
            # alternative. True relations stay in the selected goal set so
            # the compiler can preserve them while executing the missing ones.
            if shared_destination:
                score = lambda item: sum(
                    truth.get((target, item), False) for target in targets
                )
                ranked_destinations = (
                    sorted(destinations, key=lambda item: (score(item), str(item)), reverse=True)
                    if quantifier.endswith("_TO_ONE_DESTINATION")
                    else sorted(destinations, key=lambda item: -score(item))
                )
                for destination in ranked_destinations:
                    available = tuple(
                        target for target in targets if (target, destination) in truth
                    )
                    matched = {target for target in available if truth[target, destination]}
                    for selected in target_sets(available, matched):
                        yield tuple(_leaf(predicate, target, destination) for target in selected)
                return

            matched_targets = {
                target for target in targets
                if any(truth.get((target, destination), False) for destination in destinations)
            }
            if quantifier == "EXACTLY_1" and len(matched_targets) > 1:
                raise ValueError("EXACTLY_1 is already over-satisfied and needs a negative action")
            for selected in target_sets(targets, matched_targets):
                choices = tuple(
                    tuple(sorted(
                        (destination for destination in destinations
                         if (target, destination) in truth
                         and (target not in matched_targets or truth[target, destination])),
                        key=lambda destination: not truth[target, destination],
                    ))
                    for target in selected
                )
                for assignment in itertools.product(*choices):
                    yield tuple(
                        _leaf(predicate, target, destination)
                        for target, destination in zip(selected, assignment, strict=True)
                    )

        options = tuple(itertools.islice(clause_witnesses(), _MAXIMUM_PLAN_CANDIDATES + 1))
        if not options:
            raise ValueError(f"clause {index} has no positive witness")
        truncated |= len(options) > _MAXIMUM_PLAN_CANDIDATES
        clause_options.append(options[:_MAXIMUM_PLAN_CANDIDATES])

    result: list[tuple[list[Any], ...]] = []
    seen: set[tuple[str, ...]] = set()
    # Bound examined combinations too: duplicate clauses must not hide an
    # exponential Cartesian traversal behind the unique-result limit.
    combinations = itertools.islice(
        itertools.product(*clause_options), _MAXIMUM_PLAN_CANDIDATES + 1
    )
    for selection_index, clause_selection in enumerate(combinations):
        if selection_index == _MAXIMUM_PLAN_CANDIDATES:
            truncated = True
            break
        unique = {
            json.dumps(leaf, separators=(",", ":")): leaf
            for leaves in clause_selection
            for leaf in leaves
        }
        key = tuple(sorted(unique))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(tuple(unique[item] for item in key))
    if diagnostics is not None:
        diagnostics.update(
            frontier_truncated=truncated,
            candidate_limit=_MAXIMUM_PLAN_CANDIDATES,
            enumerated_candidate_count=len(result),
            exhaustive=not truncated,
        )
    if not result:
        raise ValueError("projected goal has no positive witness")
    return tuple(result)


def _initially_true_leaves(
    leaves: Iterable[Any], truth_reader: Callable[[str, str, str | None], bool]
) -> tuple[Any, ...]:
    result = []
    for leaf in leaves:
        relation = relation_from_grounded_leaf(leaf)
        value = truth_reader(
            _PREDICATE_BY_KIND[relation.kind], relation.subject, relation.reference
        )
        if not isinstance(value, bool):
            raise TypeError("projected truth reader must return bool")
        if value:
            result.append(leaf)
    return tuple(result)


def compile_projected_goal_plan(
    task_spec: Mapping[str, Any],
    truth_reader: Callable[[str, str, str | None], bool],
    *,
    openable_targets: Iterable[str] = (),
    immovable_targets: Iterable[str] = (),
    source_openable_ancestors_by_subject: Mapping[str, Iterable[str]] | None = None,
) -> GroundedMixedGoalPlan:
    return compile_projected_goal_plans(
        task_spec,
        truth_reader,
        openable_targets=openable_targets,
        immovable_targets=immovable_targets,
        source_openable_ancestors_by_subject=source_openable_ancestors_by_subject,
    )[0]


def compile_projected_goal_plans(
    task_spec: Mapping[str, Any],
    truth_reader: Callable[[str, str, str | None], bool],
    *,
    openable_targets: Iterable[str] = (),
    immovable_targets: Iterable[str] = (),
    source_openable_ancestors_by_subject: Mapping[str, Iterable[str]] | None = None,
    witness_options: Sequence[tuple[list[Any], ...]] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[GroundedMixedGoalPlan, ...]:
    """Compile a bounded frontier; structural rejection is candidate-local."""

    # Compilation is read-only. Share one measured truth per grounded relation
    # across enumeration and all candidate compiles, not thousands of readbacks.
    truth_reader = cache(truth_reader)
    if witness_options is None:
        witness_options = projected_witness_leaf_options(
            task_spec, truth_reader, diagnostics=diagnostics
        )
    openable_targets = tuple(openable_targets)
    immovable_targets = tuple(immovable_targets)
    plans = []
    rejected = []
    for index, leaves in enumerate(witness_options):
        try:
            plans.append(compile_grounded_mixed_option(
                leaves,
                option_index=index,
                openable_targets=openable_targets,
                immovable_targets=immovable_targets,
                initially_true_relations=_initially_true_leaves(leaves, truth_reader),
                source_openable_ancestors_by_subject=source_openable_ancestors_by_subject,
            ))
        except MixedGoalGraphError as exc:
            rejected.append({"candidate_index": index, "error": str(exc)})
    if diagnostics is not None:
        diagnostics.update(
            compiled_candidate_count=len(plans),
            compilation_rejections=rejected,
        )
    if not plans:
        raise MixedGoalGraphError("no compilable candidate in bounded projected frontier")
    return tuple(plans)


def candidate_profile(
    task_spec: Mapping[str, Any], plan: GroundedMixedGoalPlan
) -> Any:
    identity = task_spec["private"]["source_task_identity"]
    return SimpleNamespace(
        task_name=identity["task_name"],
        task_id=identity["task_id"],
        task_language=task_spec["public"]["instruction"],
        scene_model=identity["scene_model"],
        split=identity["split"],
        instance_id=identity["instance_id"],
        expected_goal_leaf_count=plan.leaf_count,
        canonical_option_index=0,
        canonical_live_option_index=None,
        canonical_goal_leaves=tuple(
            json.dumps(relation.to_official_body(), separators=(",", ":"))
            for relation in plan.goal_relations
        ),
        goal_signatures=("MixedAction23",),
    )


def plan_scope_names(plan: GroundedMixedGoalPlan) -> frozenset[str]:
    return frozenset(
        value
        for node in plan.action_nodes
        for value in (node.target, node.destination)
        if value is not None
    )


def _first_plan_operation(
    plan: GroundedMixedGoalPlan | GroundedInsideGoalPlan,
) -> tuple[str, str] | None:
    if not plan.action_nodes:
        return None
    node = plan.action_nodes[0]
    if node.action.startswith("place_"):
        return ("pick", node.target)
    return (node.action, node.target)


def audit_goal_plan_navigation_route(
    *,
    evaluator: Any,
    plan: GroundedMixedGoalPlan,
    route_planner: Callable[..., Any],
) -> Mapping[str, Any]:
    """Read-only route check following the existing DAG action order."""

    scope = evaluator.env.task.object_scope
    start_xy = None
    legs = []
    for node in plan.action_nodes:
        names = (
            (node.target, node.destination)
            if node.destination is not None
            else (node.target,)
        )
        for index, scope_name in enumerate(names):
            target = scope.get(scope_name)
            if target is None:
                raise RuntimeError(f"route scope is unbound: {scope_name}")
            route = route_planner(
                evaluator=evaluator,
                target=target,
                desired_distance_m=(
                    0.76 if index == 0 and len(names) == 2 else 0.78
                ),
                relaxed_symbolic_navigation=True,
                start_xy_override=start_xy,
            )
            path_length = (
                None
                if route.plan is None
                else float(route.plan.path_length_m)
            )
            legs.append(
                {
                    "node_id": node.node_id,
                    "action": node.action,
                    "scope_name_private": scope_name,
                    "status": route.status,
                    "path_length_m": path_length,
                    "route_provider_private": dict(route.private_report),
                }
            )
            if route.status != "planned" or route.plan is None:
                return {
                    "status": "FAIL",
                    "route_exists": False,
                    "legs": legs,
                    "simulator_state_mutated": False,
                }
            start_xy = route.plan.goal_xy
    total = sum(float(leg["path_length_m"]) for leg in legs)
    return {
        "status": "PASS",
        "route_exists": True,
        "legs": legs,
        "total_path_length_m": total,
        "total_move015_lower_bound": int(math.ceil(total / 0.15)),
        "simulator_state_mutated": False,
    }


class L0TaskSpecPredicateVerifier:
    """Expose the TaskSpec adapter through the existing Oracle verifier API."""

    def __init__(
        self,
        *,
        adapter: Any,
        plan: GroundedMixedGoalPlan,
        initial_report: Mapping[str, Any],
    ) -> None:
        self._adapter = adapter
        self._relations = tuple(plan.goal_relations)

    def _leaf_truth(self) -> dict[str, bool]:
        reader = self._adapter.truth_reader()
        return {
            json.dumps(relation.identity, separators=(",", ":")): reader(
                _PREDICATE_BY_KIND[relation.kind],
                relation.subject,
                relation.reference,
            )
            for relation in self._relations
        }

    def evaluate(self, episode: Any, evaluator: Any) -> PrivatePredicateSnapshot:
        # Candidate verifiers may be created before earlier candidates run.
        # Never combine their old initial Q with fresh per-leaf truth.
        report = dict(
            self._adapter.snapshot(episode, evaluator, phase="after_action")
        )
        if report.get("operational") is not True:
            return PrivatePredicateSnapshot(
                ok=False,
                official_success=False,
                official_q=None,
                error="; ".join(map(str, report.get("errors", ())))
                or str(report.get("failure_code") or "TaskSpec verifier failed"),
            )
        return PrivatePredicateSnapshot(
            ok=True,
            official_success=bool(report.get("success")),
            official_q=float(report.get("q_score", 0.0)),
            leaf_truth=self._leaf_truth(),
            research_diagnostics={"task_spec_id": self._adapter.task_spec_id},
        )


def make_l0_taskspec_action20_oracle_callback(
    *,
    task_spec: Mapping[str, Any],
    seed: int,
    phase_name: str,
    motion_safety_starter_cache: dict[int, Any] | None = None,
    admission_gt_macro_navigation: bool = True,
    navigation_lane: str = "legacy_xy_recovery_v1",
) -> Any:
    """Bind the finite legal candidate frontier to the Goal-DAG Oracle."""

    if not isinstance(admission_gt_macro_navigation, bool):
        raise TypeError("admission_gt_macro_navigation must be boolean")

    from behavior_hl.embodied.evaluator_bridge.l0_taskspec_live_adapter import (
        L0TaskSpecLiveAdapter,
    )
    from behavior_hl.embodied.evaluator_bridge.official_predicate_verifier import (
        OfficialPredicateVerifier,
    )
    from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
        THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    )
    from behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders import (
        _read_marker_symbolic_canonical_state,
    )
    from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
        placement_predicate_mode,
    )

    route = task_spec["private"]["verifier"]["route"]
    candidate_id = str(task_spec["task_spec_id"])
    identity = dict(task_spec["private"]["source_task_identity"])
    # This callback runs the default zero-physics public Action20 lane. Keep
    # initial, per-action and final verification on that same shared contract.
    predicate_mode = placement_predicate_mode(
        symbolic_zero_physics=True, physical_consequences_enabled=False,
    )
    verifier_episode = EpisodeSpec(
        episode_id=f"{candidate_id}/{phase_name}/{seed}",
        task_id=identity["task_name"],
        split=identity["split"],
        instance_id=str(identity["instance_id"]),
        seed=seed,
        instruction=task_spec["public"]["instruction"],
        action_profile=ACTION20_PUBLIC_PROFILE,
        observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
    )
    starter_cache = (
        {} if motion_safety_starter_cache is None else motion_safety_starter_cache
    )

    def callback(
        *,
        evaluator: Any,
        control: Any,
        bindings: Any,
        run_dir: Any,
        stages: Any,
    ) -> Mapping[str, Any]:
        del bindings
        from behavior_hl.embodied.evaluator_bridge.grounded_option_identity_owner import (
            resolve_grounded_option_identity_owner,
        )
        from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import (
            get_inside_task_profile,
        )
        from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
            _live_immovable_scope_names,
            _live_openable_scope_names,
            _openable_containment_ancestor_scope_names,
            _private_official_oracle_route_plan,
            make_public_action20_marker_oracle_callback,
        )

        adapter = L0TaskSpecLiveAdapter(
            task_spec,
            stability_window=1,
            predicate_mode=predicate_mode,
            action_outcome_verifier=(
                None
                if route == "official_full"
                else Action23OutcomeVerifier(
                    read_state=partial(
                        _read_marker_symbolic_canonical_state,
                        symbolic_on_top=predicate_mode == "action20_symbolic",
                    ),
                    read_held=lambda: None,
                )
            ),
        )
        initial: Mapping[str, Any] | None = None
        final: Mapping[str, Any] | None = None
        oracle: Mapping[str, Any] | None = None
        navigation: Mapping[str, Any] | None = None
        plan: GroundedMixedGoalPlan | GroundedInsideGoalPlan | None = None
        candidate_attempts: list[Mapping[str, Any]] = []
        frontier_audit: dict[str, Any] = {}
        cached_navigation: dict[int, Mapping[str, Any]] = {}
        failure_class = "UNKNOWN"
        failure_code = None
        error = None
        try:
            binding = adapter.bind(
                evaluator,
                runtime_identity=identity,
                marker_session_id=f"l0:{candidate_id}:live",
            )
            initial = adapter.initialize(verifier_episode, evaluator)
            if (
                binding.get("binding_pass") is not True
                or initial.get("admission_pass") is not True
            ):
                if (binding.get("binding_pass") is True
                        and initial.get("failure_code") == "INITIAL_GOAL_ALREADY_SATISFIED"):
                    failure_code = "INITIAL_GOAL_ALREADY_SATISFIED"
                    raise RuntimeError("initial TaskSpec goal already satisfied; Oracle not run")
                failure_class = "TASKSPEC_BINDING_ERROR"
                raise RuntimeError("initial TaskSpec binding/admission failed")

            task = evaluator.env.task
            scope = task.object_scope
            candidate_rows: list[tuple[Any, Any, Any, Mapping[str, Any]]] = []
            if route == "official_full":
                profile = get_inside_task_profile(identity["task_name"])
                owner = resolve_grounded_option_identity_owner(
                    evaluator=evaluator, profile=profile
                )
                options = task.ground_goal_state_options
                plans: Sequence[Any] = (owner.witness.compiled_plan,)
                if len(options) <= _MAXIMUM_PLAN_CANDIDATES:
                    if isinstance(owner.witness.compiled_plan, GroundedMixedGoalPlan):
                        plans = feasible_grounded_mixed_plans(
                            options,
                            openable_targets=_live_openable_scope_names(scope),
                            immovable_targets=_live_immovable_scope_names(scope),
                        )
                    else:
                        plans = feasible_grounded_inside_plans(
                            options,
                            openable_targets=_live_openable_scope_names(scope),
                        )
                    plans = (
                        owner.witness.compiled_plan,
                        *(candidate for candidate in plans if candidate.option_index != owner.witness.option_index),
                    )
                for candidate in plans:
                    option_identity = grounded_goal_option_identity(
                        options[candidate.option_index]
                    )
                    candidate_rows.append(
                        (
                            candidate,
                            profile,
                            OfficialPredicateVerifier(
                                require_single_option=True,
                                predicate_mode=predicate_mode,
                                selected_option_index=candidate.option_index,
                                selected_option_identity=option_identity,
                            ),
                            (
                                owner.private_audit_record()
                                if candidate.option_index == owner.witness.option_index
                                else {
                                    "route": route,
                                    "bounded_legal_plan_frontier": True,
                                    "selected_option_index": candidate.option_index,
                                    "selected_option_identity": option_identity,
                                }
                            ),
                        )
                    )
            else:
                reader = cache(adapter.truth_reader())
                witness_options = projected_witness_leaf_options(
                    task_spec, reader, diagnostics=frontier_audit
                )
                access_subjects = {
                    relation.subject
                    for leaves in witness_options
                    for leaf in leaves
                    for relation in (relation_from_grounded_leaf(leaf),)
                    if relation.kind
                    in {
                        "inside",
                        "open",
                        "on_top",
                        "under",
                        "next_to",
                        "touching",
                        "toggled_on",
                        "not_toggled_on",
                    }
                }
                ancestors = {
                    subject: chain
                    for subject in sorted(access_subjects)
                    for chain in (
                        _openable_containment_ancestor_scope_names(
                            scope=scope,
                            child_scope_name=subject,
                        ),
                    )
                    if chain
                }
                plans = compile_projected_goal_plans(
                    task_spec,
                    reader,
                    openable_targets=_live_openable_scope_names(scope),
                    immovable_targets=_live_immovable_scope_names(scope),
                    source_openable_ancestors_by_subject=ancestors,
                    witness_options=witness_options,
                    diagnostics=frontier_audit,
                )
                for candidate in plans:
                    candidate_rows.append(
                        (
                            candidate,
                            candidate_profile(task_spec, candidate),
                            L0TaskSpecPredicateVerifier(
                                adapter=adapter,
                                plan=candidate,
                                initial_report=initial,
                            ),
                            {
                                "task_spec_id": candidate_id,
                                "route": route,
                                "projection_materialized_from_live_truth": True,
                                "candidate_plan_index": candidate.option_index,
                                "candidate_plan_count": len(plans),
                                "frontier_private": frontier_audit,
                            },
                        )
                    )

            if not candidate_rows:
                raise RuntimeError("L0 TaskSpec has no executable plan candidate")
            rejected_first_operations: set[tuple[str, str]] = set()
            navigation_attempts: list[Mapping[str, Any]] = []
            all_strategies: list[Mapping[str, Any]] = []
            selected_oracle: Mapping[str, Any] | None = None
            selected_navigation: Mapping[str, Any] | None = None
            indexed_candidates = list(enumerate(candidate_rows))
            if route != "official_full":
                indexed_candidates, cached_navigation = _rank_projected_candidate_prefix(
                    candidate_rows,
                    lambda candidate: audit_goal_plan_navigation_route(
                        evaluator=evaluator,
                        plan=candidate,
                        route_planner=_private_official_oracle_route_plan,
                    ),
                )
            for candidate_index, (
                candidate,
                candidate_profile_value,
                candidate_verifier,
                candidate_audit,
            ) in indexed_candidates:
                first_operation = _first_plan_operation(candidate)
                if first_operation is None:
                    candidate_attempts.append({
                        "candidate_index": candidate_index,
                        "option_index": candidate.option_index,
                        "status": "SKIPPED_EMPTY_GOAL_PLAN",
                    })
                    continue
                if first_operation in rejected_first_operations:
                    candidate_attempts.append(
                        {
                            "candidate_index": candidate_index,
                            "option_index": candidate.option_index,
                            "status": "SKIPPED_REJECTED_FIRST_OPERATION",
                            "first_operation_private": first_operation,
                        }
                    )
                    continue
                candidate_navigation = cached_navigation.get(candidate_index)
                if candidate_navigation is None:
                    candidate_navigation = audit_goal_plan_navigation_route(
                        evaluator=evaluator,
                        plan=candidate,
                        route_planner=_private_official_oracle_route_plan,
                    )
                navigation_attempts.append(
                    {
                        "candidate_index": candidate_index,
                        "option_index": candidate.option_index,
                        **dict(candidate_navigation),
                    }
                )
                if _route_audit_blocks_candidate(
                    candidate_navigation, navigation_lane
                ):
                    candidate_attempts.append(
                        {
                            "candidate_index": candidate_index,
                            "option_index": candidate.option_index,
                            "status": "ROUTE_REJECTED",
                            "first_operation_private": first_operation,
                        }
                    )
                    continue
                candidate_phase_name = (
                    phase_name
                    if selected_oracle is None
                    else f"{phase_name}_candidate_{candidate_index:03d}"
                )
                current_oracle = make_public_action20_marker_oracle_callback(
                    seed=seed,
                    profile=candidate_profile_value,
                    observation_profile=THREE_RGB_HEAD_EVAL_CAMERA_PROFILE,
                    admission_gt_macro_navigation=admission_gt_macro_navigation,
                    navigation_lane=navigation_lane,
                    phase_name=candidate_phase_name,
                    motion_safety_starter_cache=starter_cache,
                    public_goal_plan=candidate,
                    predicate_verifier=candidate_verifier,
                    goal_dag_audit=candidate_audit,
                )(
                    evaluator=evaluator,
                    control=control,
                    bindings=None,
                    run_dir=run_dir,
                    stages=stages,
                )
                selected_oracle = current_oracle
                selected_navigation = candidate_navigation
                plan = candidate
                current_strategies = [
                    {
                        **dict(strategy),
                        "candidate_plan_index_private": candidate_index,
                        "candidate_option_index_private": candidate.option_index,
                    }
                    for strategy in current_oracle.get("strategies", ())
                ]
                all_strategies.extend(current_strategies)
                terminal_strategy = current_strategies[-1] if current_strategies else {}
                candidate_attempts.append(
                    {
                        "candidate_index": candidate_index,
                        "option_index": candidate.option_index,
                        "status": (
                            "PASS"
                            if current_oracle.get("oracle_success") is True
                            else "FAIL"
                        ),
                        "first_operation_private": first_operation,
                        "alternative_plan_retry_safe": bool(
                            terminal_strategy.get("alternative_plan_retry_safe")
                        ),
                    }
                )
                if current_oracle.get("oracle_success") is True:
                    break
                rejection = terminal_strategy.get("candidate_rejection_private")
                if not (
                    terminal_strategy.get("alternative_plan_retry_safe") is True
                    and isinstance(rejection, Mapping)
                ):
                    break
                rejected_operation = (
                    str(rejection.get("operation", "")),
                    str(rejection.get("scope_name_private", "")),
                )
                if not all(rejected_operation):
                    break
                rejected_first_operations.add(rejected_operation)

            oracle = (
                None
                if selected_oracle is None
                else {
                    **dict(selected_oracle),
                    "strategies": all_strategies,
                    "candidate_plan_attempts_private": candidate_attempts,
                }
            )
            navigation = {
                "route_exists": (
                    None if selected_navigation is None
                    else selected_navigation.get("route_exists")
                ),
                "selected_private": selected_navigation,
                "attempts_private": navigation_attempts,
            }
            final = adapter.snapshot(verifier_episode, evaluator, phase="settle_final")
            passed = bool(
                oracle is not None
                and oracle.get("oracle_success") is True
                and final.get("success") is True
                and selected_navigation is not None
            )
            if not passed:
                failures = [
                    strategy.get("failure_class")
                    for strategy in all_strategies
                    if strategy.get("failure_class")
                ]
                failure_class = str(
                    failures[-1]
                    if failures
                    else "NAVIGATION_ERROR"
                    if selected_navigation is None
                    else final.get("failure_code") or "UNKNOWN"
                ).upper()
                if candidate_attempts and all(
                    item["status"] == "SKIPPED_EMPTY_GOAL_PLAN" for item in candidate_attempts
                ):
                    failure_class = "UNKNOWN"
                    failure_code = "EMPTY_GOAL_PLAN"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            passed = False
            code = str(getattr(exc, "code", "")).upper()
            if code in {"TASKSPEC_BINDING_ERROR", "VERIFIER_ERROR"}:
                failure_class = code
            elif plan is None and failure_class == "UNKNOWN" and failure_code is None:
                failure_class = "ORACLE_PLANNER_ERROR"
        finally:
            try:
                adapter.retire()
            except Exception:
                pass

        strategies = (
            list(oracle.get("strategies", ()))
            if oracle is not None
            else [
                {
                    "status": "FAIL",
                    "failure_class": failure_class,
                    "error": error,
                }
            ]
        )
        # Candidate attempts belong to one resident strategy, not separate runs.
        terminal = dict(strategies[-1]) if strategies else {}
        strategy_result = {
            **terminal,
            "underlying_oracle_status": terminal.get("status"),
            "status": "PASS" if passed else "FAIL",
            "failure_class": None if passed else failure_class,
            "failure_code": None if passed else failure_code or terminal.get("failure_code"),
            "error": error or terminal.get("error"),
            "official_q": None if final is None else final.get("q_score"),
            **{
                key: sum(int(strategy.get(key) or 0) for strategy in strategies)
                for key in ("actions_executed", "simulator_steps")
            },
        }
        return {
            "schema_version": "behavior2026.l0-taskspec-action20-oracle/v1",
            "task": identity["task_name"],
            "task_spec_id": candidate_id,
            "verifier_route": route,
            "predicate_mode": predicate_mode,
            "seed": seed,
            "strategies": [strategy_result],
            "candidate_strategy_results_private": strategies,
            "oracle_success": passed,
            "official_success": passed,
            "failure_class": None if passed else failure_class,
            "failure_code": strategy_result.get("failure_code"),
            "error": error,
            "initial_taskspec_snapshot_private": initial,
            "final_taskspec_snapshot_private": final,
            "goal_dag_action_count": (
                None if plan is None else len(plan.action_nodes)
            ),
            "candidate_plan_attempts_private": candidate_attempts,
            "projected_frontier_private": frontier_audit or None,
            "candidate_ranking_audits_private": [
                {"candidate_index": index, **dict(audit)}
                for index, audit in cached_navigation.items()
            ],
            "navigation_route_audit_private": navigation,
            "route_gate_pass": bool(
                navigation and navigation.get("route_exists") is True
            ),
            "route_audit_mode": (
                "ADVISORY" if navigation_lane == "se2_navigation_leg_v1" else "REQUIRED"
            ),
            "marker_action_gate_pass": bool(
                oracle
                and oracle.get("oracle_success") is True
                and final
                and final.get("success") is True
            ),
            "public_action20_runtime_dispatch_execution_proven": passed,
            "marker_runtime_transaction_execution_proven": passed,
            "admission_gt_macro_navigation_enabled": (
                admission_gt_macro_navigation
            ),
            "public_oracle_navigation_lane": navigation_lane,
            "official_challenge_eligible": False,
        }

    return callback


__all__ = [
    "L0TaskSpecPredicateVerifier",
    "audit_goal_plan_navigation_route",
    "candidate_profile",
    "compile_projected_goal_plan",
    "compile_projected_goal_plans",
    "make_l0_taskspec_action20_oracle_callback",
    "plan_scope_names",
    "projected_witness_leaves",
    "projected_witness_leaf_options",
]
