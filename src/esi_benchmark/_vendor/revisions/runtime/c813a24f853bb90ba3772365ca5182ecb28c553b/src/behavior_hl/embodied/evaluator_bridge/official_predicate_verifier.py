"""Private BEHAVIOR task-state adapter for the resident episode runtime.

This module intentionally imports no OmniGibson modules.  It is instantiated
inside the evaluator process and receives the already-live evaluator as an
opaque object.  Its output must only be written to private artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from behavior_hl.embodied.evaluator_bridge.inside_goal_graph import (
    grounded_goal_option_identity,
)
from behavior_hl.embodied.evaluator_bridge.official_predicate_identity import (
    build_official_goal_leaf_ids,
)
from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    read_action_relation,
)
from behavior_hl.embodied.runtime.episode import EpisodeSpec
from behavior_hl.embodied.runtime.action_integrity import execution_integrity_failure
from behavior_hl.embodied.runtime.verification import PrivatePredicateSnapshot


@dataclass(slots=True)
class OfficialPredicateVerifier:
    """Re-evaluate official goal predicates using the TaskMetric convention.

    The B1 pilot freezes ``picking_up_toys`` to one grounded goal option with
    six leaves.  Requiring one option avoids inventing a cross-option research
    partial score that would not have the same meaning as the official metric.
    """

    require_single_option: bool = True
    selected_option_index: int | None = None
    selected_option_identity: tuple[str, ...] | None = None
    predicate_mode: str = "official"
    _initial_states: tuple[tuple[bool, ...], ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _leaf_ids: tuple[tuple[str, ...], ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _initial_raw_official_states: tuple[tuple[bool, ...], ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @staticmethod
    def _resolve_task(evaluator_state: Any) -> Any:
        env = getattr(evaluator_state, "env", evaluator_state)
        task = getattr(env, "task", None)
        if task is None:
            raise TypeError("evaluator state does not expose env.task")
        return task

    def __post_init__(self) -> None:
        if self.predicate_mode not in ("official", "action20_symbolic"):
            raise ValueError("predicate_mode must be official or action20_symbolic")
        if self.selected_option_index is not None and (
            isinstance(self.selected_option_index, bool)
            or not isinstance(self.selected_option_index, int)
            or self.selected_option_index < 0
        ):
            raise ValueError("selected option index must be non-negative")
        if self.selected_option_identity is not None:
            identity = tuple(self.selected_option_identity)
            if (
                not identity
                or any(not isinstance(item, str) or not item for item in identity)
            ):
                raise ValueError("selected option identity must be non-empty")
            if self.selected_option_index is None:
                raise ValueError(
                    "selected option identity requires one selected index"
                )
            self.selected_option_identity = identity

    def _selected_options(self, task: Any) -> tuple[Any, ...]:
        options = getattr(task, "ground_goal_state_options", None)
        if (
            not isinstance(options, Sequence)
            or isinstance(options, (str, bytes))
            or not options
        ):
            raise TypeError("official task predicate API is unavailable")
        if self.selected_option_index is None:
            return tuple(options)
        if self.selected_option_index >= len(options):
            raise ValueError("selected official goal option index is out of bounds")
        return (options[self.selected_option_index],)

    @staticmethod
    def _evaluate_options(
        task: Any,
        options: tuple[Any, ...],
        *,
        predicate_evaluator: Any | None = None,
    ) -> tuple[tuple[bool, ...], ...]:
        if predicate_evaluator is None:
            predicate_evaluator = getattr(task, "_evaluate_predicate", None)
        if not callable(predicate_evaluator):
            raise TypeError("official task predicate API is unavailable")
        evaluated = tuple(
            tuple(bool(predicate.evaluate(predicate_evaluator)) for predicate in option)
            for option in options
        )
        if not evaluated or any(not option for option in evaluated):
            raise ValueError("official task has no grounded goal leaves")
        return evaluated

    @staticmethod
    def _action20_predicate_evaluator(task: Any):
        official = getattr(task, "_evaluate_predicate", None)
        scope = getattr(task, "object_scope", None)
        if not callable(official) or not isinstance(scope, Mapping):
            raise TypeError("official task predicate API is unavailable")

        def evaluate(predicate_type: Any, *entity_names: str, **kwargs: Any) -> bool:
            relation = {"ontop": "OnTop", "nextto": "NextTo"}.get(
                getattr(predicate_type, "__name__", "").casefold())
            if relation is None:
                return bool(official(predicate_type, *entity_names, **kwargs))
            if len(entity_names) != 2:
                raise ValueError(f"{relation} readback requires two bound entities")
            obj, support = (scope.get(name) for name in entity_names)
            if (
                obj is None
                or support is None
                or not bool(getattr(obj, "initialized", False))
                or not bool(getattr(support, "initialized", False))
            ):
                raise RuntimeError(f"{relation} readback requires initialized live objects")
            return read_action_relation(obj, relation, support, predicate_mode="action20_symbolic")

        return evaluate

    def _score_options(
        self,
        current: tuple[tuple[bool, ...], ...],
        initial: tuple[tuple[bool, ...], ...],
    ) -> tuple[bool, float]:
        # Options have already been restricted to the caller's scoring scope.
        # task.success is only updated by Task.step, not symbolic pose writes.
        success = any(all(option) for option in current)
        if success:
            return True, 1.0
        q = max(
            sum(
                int(not initially_true and truth)
                for truth, initially_true in zip(
                    option,
                    initial_option,
                    strict=True,
                )
            )
            / len(option)
            for option, initial_option in zip(current, initial, strict=True)
        )
        return False, float(q)

    def evaluate(
        self,
        episode: EpisodeSpec,
        evaluator_state: Any,
    ) -> PrivatePredicateSnapshot:
        failure = execution_integrity_failure(evaluator_state)
        if failure is not None:
            return PrivatePredicateSnapshot(
                ok=False, official_success=False, official_q=None,
                error=f"world integrity invalid: {failure}",
            )
        task = self._resolve_task(evaluator_state)
        all_goal_options = getattr(task, "ground_goal_state_options", None)
        goal_options = self._selected_options(task)
        if self.selected_option_identity is not None:
            actual_identity = grounded_goal_option_identity(goal_options[0])
            if actual_identity != self.selected_option_identity:
                raise ValueError(
                    "selected official goal option identity changed"
                )
        current_leaf_ids = build_official_goal_leaf_ids(goal_options)
        action20_symbolic = self.predicate_mode == "action20_symbolic"
        raw_official_error = None
        try:
            raw_official_current = self._evaluate_options(task, goal_options)
        except Exception as exc:
            if not action20_symbolic:
                raise
            raw_official_current = None
            raw_official_error = f"{type(exc).__name__}: {exc}"
        if action20_symbolic:
            current = self._evaluate_options(
                task,
                goal_options,
                predicate_evaluator=self._action20_predicate_evaluator(task),
            )
        else:
            current = raw_official_current
        if self.require_single_option and len(current) != 1:
            raise ValueError(
                "B1 verifier requires exactly one grounded goal option"
            )

        if self._initial_states is None:
            self._initial_states = current
            self._initial_raw_official_states = raw_official_current
            self._leaf_ids = current_leaf_ids
        elif tuple(map(len, current)) != tuple(map(len, self._initial_states)):
            raise ValueError("official grounded goal leaf structure changed")
        elif current_leaf_ids != self._leaf_ids:
            raise ValueError(
                "official grounded goal predicate identity/order changed"
            )

        assert self._initial_states is not None
        assert self._leaf_ids is not None
        official_success, official_q = self._score_options(
            current,
            self._initial_states,
        )
        raw_official_success, raw_official_q = None, None
        if raw_official_current is not None and self._initial_raw_official_states is not None:
            raw_official_success, raw_official_q = self._score_options(
                raw_official_current, self._initial_raw_official_states,
            )

        leaf_truth = {
            leaf_id: truth
            for option_ids, option_truth in zip(
                self._leaf_ids,
                current,
                strict=True,
            )
            for leaf_id, truth in zip(
                option_ids,
                option_truth,
                strict=True,
            )
        }
        return PrivatePredicateSnapshot(
            ok=True,
            official_success=official_success,
            official_q=float(official_q),
            leaf_truth=leaf_truth,
            research_diagnostics={
                "ground_goal_option_count": len(current),
                "ground_goal_total_option_count": len(all_goal_options),
                "ground_goal_leaf_counts": [len(option) for option in current],
                "selected_ground_goal_option_index": self.selected_option_index,
                "selected_ground_goal_option_identity_attested": (
                    self.selected_option_identity is not None
                ),
                "official_task_success_scope": (
                    "global_all_options"
                    if self.selected_option_index is None
                    else "selected_option"
                ),
                "source": "live_official_predicate_evaluation",
                "predicate_mode": self.predicate_mode,
                "authoritative_score_source": (
                    "action20_symbolic_geometry"
                    if action20_symbolic
                    else "live_official_predicate_evaluation"
                ),
                "action20_symbolic_on_top": action20_symbolic,
                "action20_symbolic_q": (
                    float(official_q) if action20_symbolic else None
                ),
                "raw_official_success_private": raw_official_success,
                "raw_official_q_private": raw_official_q,
                "raw_official_error_private": raw_official_error,
                "raw_official_baseline_available_private": self._initial_raw_official_states is not None,
            },
        )
