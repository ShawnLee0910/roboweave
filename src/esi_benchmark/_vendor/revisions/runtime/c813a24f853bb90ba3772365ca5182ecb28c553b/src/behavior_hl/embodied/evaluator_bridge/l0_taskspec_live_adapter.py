"""CPU-safe live binding and verifier routing for L0 TaskSpecs.

This module owns no predicate thresholds.  Official goals delegate to
``OfficialPredicateVerifier``; projected goals delegate their aggregation to
the adapter named by the TaskSpec and read live truth through the same
``Action23OutcomeVerifier.read_state`` callback used by Action20 outcomes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache, partial
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from behavior_hl.embodied.evaluator_bridge.action23_outcome_verifier import (
    Action23OutcomeVerifier,
)
from behavior_hl.embodied.evaluator_bridge.official_predicate_verifier import (
    OfficialPredicateVerifier,
)
from behavior_hl.embodied.runtime.verification import PrivatePredicateSnapshot


TASKSPEC_VERSION = "action20-capability-benchmark.l0-task/v1"
RECEIPT_VERSION = "behavior2026.l0-taskspec-live-adapter/v1"
OFFICIAL_FULL = "official_full"
OFFICIAL_PROJECTION = "official_projection"
CANONICAL_FALLBACK = "canonical_fallback"
_ROUTES = frozenset({OFFICIAL_FULL, OFFICIAL_PROJECTION, CANONICAL_FALLBACK})
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_FIELDS = ("task_id", "task_name", "scene_model", "split", "instance_id")
_PREDICATE_STATE = MappingProxyType(
    {
        "inside": ("Inside", True),
        "ontop": ("OnTop", True),
        "under": ("Under", True),
        "nextto": ("NextTo", True),
        "open": ("Open", True),
        "closed": ("Open", False),
        "toggled_on": ("ToggledOn", True),
        "toggled_off": ("ToggledOn", False),
    }
)


class L0LiveAdapterError(RuntimeError):
    """One typed, fail-closed TaskSpec binding or verifier error."""

    def __init__(self, code: str, message: str) -> None:
        if not isinstance(code, str) or not code:
            raise ValueError("L0 adapter error code must be non-empty")
        super().__init__(message)
        self.code = code


def canonical_task_spec_sha256(value: Mapping[str, Any]) -> str:
    """Hash one TaskSpec using the compiler's canonical JSON convention."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise L0LiveAdapterError("INVALID_TASKSPEC", f"TaskSpec is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise L0LiveAdapterError("INVALID_TASKSPEC", f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise L0LiveAdapterError("INVALID_TASKSPEC", f"{field} must be a sequence")
    return value


def _validated_file_ref(value: Any, field: str) -> tuple[Path, str]:
    record = _mapping(value, field)
    raw_path = record.get("path")
    expected = record.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(expected, str)
        or not _HEX_SHA256.fullmatch(expected)
    ):
        raise L0LiveAdapterError("INVALID_TASKSPEC", f"{field} must contain path and SHA256")
    path = Path(raw_path)
    return path, expected


@lru_cache(maxsize=None)
def _import_projected_verifier_class(raw_path: str, digest: str) -> type[Any]:
    del raw_path  # Frozen source path is provenance; deployment uses its exact vendored bytes.
    path = Path(__file__).with_name("l0_verifier_adapter.py")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise L0LiveAdapterError(
            "PROVENANCE_MISMATCH", "projected verifier SHA256 differs"
        )
    module_name = f"_l0_projected_verifier_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise L0LiveAdapterError("PROVENANCE_MISMATCH", "projected verifier module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise L0LiveAdapterError("PROVENANCE_MISMATCH", f"projected verifier import failed: {exc}") from exc
    verifier_class = getattr(module, "ProjectedGoalVerifier", None)
    if not isinstance(verifier_class, type):
        raise L0LiveAdapterError("PROVENANCE_MISMATCH", "ProjectedGoalVerifier is unavailable")
    return verifier_class


def _task_from(evaluator_state: Any) -> Any:
    env = getattr(evaluator_state, "env", evaluator_state)
    task = getattr(env, "task", None)
    if task is None:
        raise L0LiveAdapterError("TASKSPEC_BINDING_ERROR", "evaluator state does not expose env.task")
    return task


def _entity_rooms(entity: Any) -> tuple[str, ...]:
    rooms = getattr(entity, "in_rooms", None)
    if not isinstance(rooms, Iterable) or isinstance(rooms, (str, bytes, Mapping)):
        raise L0LiveAdapterError("TASKSPEC_BINDING_ERROR", "live object in_rooms is unavailable")
    return tuple(sorted(str(room) for room in rooms))


@dataclass(frozen=True, slots=True)
class _Binding:
    generation: int
    marker_session_id: str
    task: Any
    scopes: Mapping[str, Any]
    receipt: Mapping[str, Any]


class L0TaskSpecLiveAdapter:
    """Bind one frozen L0 TaskSpec and route its authoritative verifier."""

    def __init__(
        self,
        task_spec: Mapping[str, Any],
        *,
        expected_task_spec_sha256: str | None = None,
        stability_window: int,
        action_outcome_verifier: Action23OutcomeVerifier | None = None,
        predicate_mode: str = "official",
    ) -> None:
        if not isinstance(task_spec, Mapping):
            raise L0LiveAdapterError("INVALID_TASKSPEC", "TaskSpec must be a mapping")
        if expected_task_spec_sha256 is not None and (
            not isinstance(expected_task_spec_sha256, str)
            or not _HEX_SHA256.fullmatch(expected_task_spec_sha256)
        ):
            raise L0LiveAdapterError("INVALID_TASKSPEC_HASH", "expected TaskSpec SHA256 is invalid")
        if isinstance(stability_window, bool) or not isinstance(stability_window, int) or stability_window < 1:
            raise L0LiveAdapterError("INVALID_STABILITY_POLICY", "stability_window must be positive")
        if predicate_mode not in ("official", "action20_symbolic"):
            raise ValueError("predicate_mode must be official or action20_symbolic")

        self.task_spec = deepcopy(dict(task_spec))
        self.task_spec_sha256 = canonical_task_spec_sha256(self.task_spec)
        if (
            expected_task_spec_sha256 is not None
            and self.task_spec_sha256 != expected_task_spec_sha256
        ):
            raise L0LiveAdapterError("TASKSPEC_HASH_MISMATCH", "TaskSpec canonical SHA256 mismatch")
        self.stability_window = stability_window
        self.predicate_mode = predicate_mode
        self._action_outcome_verifier = action_outcome_verifier
        self._projected_verifier_class: type[Any] | None = None
        self._binding: _Binding | None = None
        self._retired_generation = 0
        self._retired_marker_session_id: str | None = None
        self._verifier: Any | None = None
        self._initialized = False
        self._initial_goal_satisfied = False
        self._official_streak = 0
        self._validate_task_spec()

    @property
    def task_spec_id(self) -> str:
        return str(self.task_spec["task_spec_id"])

    @property
    def route(self) -> str:
        return str(self.task_spec["private"]["verifier"]["route"])

    def _validate_task_spec(self) -> None:
        spec = self.task_spec
        if spec.get("task_spec_version") != TASKSPEC_VERSION or spec.get("episode_mode") != "goal_completion":
            raise L0LiveAdapterError("INVALID_TASKSPEC", "unexpected TaskSpec version or episode mode")
        if not isinstance(spec.get("task_spec_id"), str) or not spec["task_spec_id"]:
            raise L0LiveAdapterError("INVALID_TASKSPEC", "task_spec_id must be non-empty")
        candidate_hash = spec.get("candidate_contract_sha256")
        if not isinstance(candidate_hash, str) or not _HEX_SHA256.fullmatch(candidate_hash):
            raise L0LiveAdapterError("INVALID_TASKSPEC", "candidate contract SHA256 is invalid")
        private = _mapping(spec.get("private"), "private")
        identity = _mapping(private.get("source_task_identity"), "private.source_task_identity")
        if any(field not in identity for field in _IDENTITY_FIELDS):
            raise L0LiveAdapterError("INVALID_TASKSPEC", "source task identity is incomplete")
        evaluation = _mapping(spec.get("evaluation"), "evaluation")
        if (
            evaluation.get("require_initial_goal_incomplete") is not True
            or evaluation.get("initially_true_leaves_earn_credit") is not False
        ):
            raise L0LiveAdapterError("INVALID_TASKSPEC", "initial-goal credit policy is not fail-closed")
        if (
            evaluation.get("execution_fidelity") != "kinematic_research"
            or evaluation.get("official_challenge_eligible") is not False
            or evaluation.get("strict_production_eligible") is not False
        ):
            raise L0LiveAdapterError(
                "INVALID_EXECUTION_FIDELITY",
                "TaskSpec does not declare the frozen F1 kinematic lane",
            )
        stability_policy = _mapping(
            evaluation.get("stability_policy"),
            "evaluation.stability_policy",
        )
        required_snapshots = stability_policy.get(
            "required_consecutive_true_snapshots"
        )
        if (
            set(stability_policy)
            != {
                "status",
                "lane",
                "required_consecutive_true_snapshots",
            }
            or stability_policy.get("status") != "BOUND"
            or stability_policy.get("lane") != "SYMBOLIC_CANONICAL"
            or isinstance(required_snapshots, bool)
            or not isinstance(required_snapshots, int)
            or required_snapshots != 1
        ):
            raise L0LiveAdapterError(
                "INVALID_STABILITY_POLICY",
                "TaskSpec symbolic stability policy differs from the frozen F1 contract",
            )
        if (
            self.stability_window
            != stability_policy["required_consecutive_true_snapshots"]
        ):
            raise L0LiveAdapterError(
                "INVALID_STABILITY_POLICY",
                "runtime stability_window differs from TaskSpec",
            )
        settle_policy = _mapping(
            evaluation.get("settle_policy"),
            "evaluation.settle_policy",
        )
        terminal_settle_steps = settle_policy.get(
            "terminal_incomplete_max_physics_steps"
        )
        if (
            set(settle_policy)
            != {
                "status",
                "terminal_incomplete_max_physics_steps",
                "new_high_level_actions_allowed",
                "new_marker_requests_allowed",
            }
            or settle_policy.get("status") != "BOUND"
            or isinstance(terminal_settle_steps, bool)
            or not isinstance(terminal_settle_steps, int)
            or terminal_settle_steps != 30
            or settle_policy.get("new_high_level_actions_allowed") is not False
            or settle_policy.get("new_marker_requests_allowed") is not False
        ):
            raise L0LiveAdapterError(
                "INVALID_SETTLE_POLICY",
                "TaskSpec settle policy differs from the frozen F1 contract",
            )
        verifier = _mapping(private.get("verifier"), "private.verifier")
        route = verifier.get("route")
        if route not in _ROUTES or verifier.get("fail_closed") is not True:
            raise L0LiveAdapterError("INVALID_TASKSPEC", "unsupported or non-fail-closed verifier route")
        clauses = tuple(_sequence(verifier.get("clauses"), "private.verifier.clauses"))
        if any(not isinstance(clause, Mapping) for clause in clauses):
            raise L0LiveAdapterError("INVALID_TASKSPEC", "verifier clauses must be mappings")
        unsupported_predicates = {
            clause.get("canonical_predicate")
            for clause in clauses
            if clause.get("canonical_predicate") not in _PREDICATE_STATE
        }
        if unsupported_predicates:
            raise L0LiveAdapterError(
                "INVALID_TASKSPEC",
                f"unsupported canonical predicates: {sorted(map(str, unsupported_predicates))}",
            )
        goal_spec = _mapping(private.get("goal_spec"), "private.goal_spec")
        goal_clauses = tuple(goal_spec.get("clauses", ()))
        if clauses != goal_clauses:
            raise L0LiveAdapterError("INVALID_TASKSPEC", "goal and verifier clauses differ")
        scope_bindings = _mapping(private.get("scope_bindings"), "private.scope_bindings")
        for scope_name, binding in scope_bindings.items():
            if not isinstance(scope_name, str) or not scope_name or not isinstance(binding, Mapping):
                raise L0LiveAdapterError("INVALID_TASKSPEC", "scope binding is malformed")
            rooms = binding.get("in_rooms")
            if (
                not isinstance(rooms, Sequence)
                or isinstance(rooms, (str, bytes))
                or any(not isinstance(room, str) or not room for room in rooms)
            ):
                raise L0LiveAdapterError("INVALID_TASKSPEC", f"scope {scope_name!r} rooms are malformed")

        if route == OFFICIAL_FULL:
            config = _mapping(verifier.get("official_verifier_config"), "official_verifier_config")
            if clauses or config.get("require_single_option") is not False:
                raise L0LiveAdapterError("INVALID_TASKSPEC", "official_full must retain the full disjunction")
        else:
            if not clauses or verifier.get("official_verifier_config") is not None:
                raise L0LiveAdapterError("INVALID_TASKSPEC", "projected verifier clauses are unavailable")
            referenced = {
                scope
                for clause in clauses
                for field in ("target_scopes", "destination_scopes", "destination_equivalence_members")
                for scope in tuple(clause.get(field, ()))
            }
            missing = referenced - set(scope_bindings)
            if missing:
                raise L0LiveAdapterError("INVALID_TASKSPEC", f"unbound projected scopes: {sorted(missing)}")
        if route == OFFICIAL_PROJECTION and verifier.get("missing_official_final_goal_edges"):
            raise L0LiveAdapterError("INVALID_TASKSPEC", "official projection has missing official goal edges")
        _validated_file_ref(verifier.get("official_verifier"), "private.verifier.official_verifier")
        local_official = Path(__file__).with_name("official_predicate_verifier.py")
        if not local_official.is_file():
            raise L0LiveAdapterError(
                "PROVENANCE_MISMATCH",
                "runtime OfficialPredicateVerifier is unavailable",
            )
        if route != OFFICIAL_FULL:
            if self._action_outcome_verifier is None or not isinstance(
                self._action_outcome_verifier, Action23OutcomeVerifier
            ):
                raise L0LiveAdapterError(
                    "STRICT_CANONICAL_READER_REQUIRED",
                    "projected routes require the shared Action23OutcomeVerifier reader",
                )
            aggregation_path, aggregation_digest = _validated_file_ref(
                verifier.get("aggregation_adapter"),
                "private.verifier.aggregation_adapter",
            )
            self._projected_verifier_class = _import_projected_verifier_class(
                str(aggregation_path),
                aggregation_digest,
            )

    def _validate_runtime_identity(self, task: Any, runtime_identity: Mapping[str, Any]) -> dict[str, Any]:
        expected = dict(self.task_spec["private"]["source_task_identity"])
        observed = {field: runtime_identity.get(field) for field in _IDENTITY_FIELDS}
        if observed != {field: expected[field] for field in _IDENTITY_FIELDS}:
            raise L0LiveAdapterError("TASKSPEC_BINDING_ERROR", "runtime source identity differs from TaskSpec")
        task_observed = {
            "task_name": getattr(task, "activity_name", None),
            "scene_model": getattr(task, "scene_name", None),
        }
        if task_observed != {"task_name": expected["task_name"], "scene_model": expected["scene_model"]}:
            raise L0LiveAdapterError("TASKSPEC_BINDING_ERROR", "loaded task/scene differs from TaskSpec")
        return observed

    def _bind_scopes(self, task: Any) -> tuple[Mapping[str, Any], list[dict[str, Any]], int]:
        live_scope = getattr(task, "object_scope", None)
        if not isinstance(live_scope, Mapping) or not live_scope:
            raise L0LiveAdapterError("TASKSPEC_BINDING_ERROR", "live task object_scope is unavailable")
        expected_bindings = self.task_spec["private"]["scope_bindings"]
        bound: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        object_ids: set[int] = set()
        actual_names: set[str] = set()
        for scope_name, expected in sorted(expected_bindings.items()):
            entity = live_scope.get(scope_name)
            if entity is None:
                raise L0LiveAdapterError("TASKSPEC_BINDING_ERROR", f"scope {scope_name!r} is unbound")
            actual = {
                "actual_name": str(getattr(entity, "name", "")),
                "category": str(getattr(entity, "category", "")),
                "model": str(getattr(entity, "model", "")),
                "in_rooms": list(_entity_rooms(entity)),
            }
            wanted = {
                "actual_name": expected.get("actual_name"),
                "category": expected.get("category"),
                "model": expected.get("model"),
                "in_rooms": sorted(expected.get("in_rooms", ())),
            }
            if actual != wanted:
                raise L0LiveAdapterError(
                    "TASKSPEC_BINDING_ERROR",
                    f"scope {scope_name!r} metadata differs: {actual!r} != {wanted!r}",
                )
            if id(entity) in object_ids or actual["actual_name"] in actual_names:
                raise L0LiveAdapterError("TASKSPEC_BINDING_ERROR", "declared scopes do not bind unique live objects")
            object_ids.add(id(entity))
            actual_names.add(actual["actual_name"])
            bound[scope_name] = entity
            rows.append({"scope": scope_name, **actual})
        return MappingProxyType(bound), rows, len(live_scope)

    def bind(
        self,
        evaluator_state: Any,
        *,
        runtime_identity: Mapping[str, Any],
        marker_session_id: str,
    ) -> Mapping[str, Any]:
        """Bind the current runtime once; repeated calls must use ``rebind``."""

        if self._binding is not None or self._retired_generation:
            raise L0LiveAdapterError("STALE_BINDING", "adapter is already bound; use rebind after reset")
        return self._bind(evaluator_state, runtime_identity=runtime_identity, marker_session_id=marker_session_id)

    def _bind(
        self,
        evaluator_state: Any,
        *,
        runtime_identity: Mapping[str, Any],
        marker_session_id: str,
    ) -> Mapping[str, Any]:
        if not isinstance(runtime_identity, Mapping):
            raise L0LiveAdapterError("TASKSPEC_BINDING_ERROR", "runtime_identity must be a mapping")
        if not isinstance(marker_session_id, str) or not marker_session_id:
            raise L0LiveAdapterError("STALE_MARKER_SESSION", "marker_session_id must be non-empty")
        task = _task_from(evaluator_state)
        identity = self._validate_runtime_identity(task, runtime_identity)
        scopes, rows, live_scope_count = self._bind_scopes(task)
        previous_generation = (
            self._binding.generation
            if self._binding is not None
            else self._retired_generation
        )
        generation = previous_generation + 1
        scope_binding_deferred = self.route == OFFICIAL_FULL and not rows
        receipt = {
            "schema_version": RECEIPT_VERSION,
            "private": True,
            "task_spec_id": self.task_spec_id,
            "task_spec_sha256": self.task_spec_sha256,
            "route": self.route,
            "binding_generation": generation,
            "marker_session_id": marker_session_id,
            "source_identity": identity,
            "declared_scope_count": len(rows),
            "live_task_scope_count": live_scope_count,
            "scope_bindings": rows,
            "binding_status": (
                "DEFERRED_OFFICIAL_SCOPE_BINDINGS"
                if scope_binding_deferred
                else "PASS"
            ),
            "binding_pass": not scope_binding_deferred,
            "previous_binding_invalidated": generation > 1,
            "previous_marker_session_invalidated": generation > 1,
        }
        self._binding = _Binding(generation, marker_session_id, task, scopes, MappingProxyType(receipt))
        self._retired_generation = 0
        self._retired_marker_session_id = None
        self._verifier = None
        self._initialized = False
        self._initial_goal_satisfied = False
        self._official_streak = 0
        return deepcopy(dict(self._binding.receipt))

    def retire(self) -> Mapping[str, Any]:
        """Invalidate the current live epoch before any world mutation."""

        binding = self._binding
        if binding is None:
            raise L0LiveAdapterError("STALE_BINDING", "adapter has no active binding to retire")
        self._retired_generation = binding.generation
        self._retired_marker_session_id = binding.marker_session_id
        self._binding = None
        self._verifier = None
        self._initialized = False
        self._initial_goal_satisfied = False
        self._official_streak = 0
        return {
            "schema_version": RECEIPT_VERSION,
            "private": True,
            "task_spec_id": self.task_spec_id,
            "retired_binding_generation": self._retired_generation,
            "retired_marker_session_id": self._retired_marker_session_id,
        }

    def rebind(
        self,
        evaluator_state: Any,
        *,
        runtime_identity: Mapping[str, Any],
        marker_session_id: str,
    ) -> Mapping[str, Any]:
        """Invalidate all old handles/readers and bind a fresh reset epoch."""

        if self._binding is None and not self._retired_generation:
            raise L0LiveAdapterError("STALE_BINDING", "adapter has no binding to reset")
        previous_marker_session_id = (
            self._binding.marker_session_id
            if self._binding is not None
            else self._retired_marker_session_id
        )
        if marker_session_id == previous_marker_session_id:
            raise L0LiveAdapterError("STALE_MARKER_SESSION", "reset reused the previous Marker session")
        if self._binding is not None:
            self.retire()
        return self._bind(evaluator_state, runtime_identity=runtime_identity, marker_session_id=marker_session_id)

    def truth_reader(self) -> Any:
        """Return a generation-bound projected-goal reader."""

        binding = self._binding
        if binding is None:
            raise L0LiveAdapterError("STALE_BINDING", "adapter is not bound")
        generation = binding.generation
        scopes = binding.scopes

        def read(predicate: str, target_scope: str, destination_scope: str | None) -> bool:
            current = self._binding
            if current is None or current.generation != generation:
                raise L0LiveAdapterError("STALE_BINDING", "projected truth reader belongs to an old reset epoch")
            mapping = _PREDICATE_STATE.get(predicate)
            if mapping is None:
                raise L0LiveAdapterError("VERIFIER_ERROR", f"unsupported canonical predicate: {predicate!r}")
            target = scopes.get(target_scope)
            destination = None if destination_scope is None else scopes.get(destination_scope)
            if target is None or (destination_scope is not None and destination is None):
                raise L0LiveAdapterError("TASKSPEC_BINDING_ERROR", "predicate references an unbound scope")
            state_name, desired = mapping
            assert self._action_outcome_verifier is not None
            observed = self._action_outcome_verifier.read_state(target, state_name, destination)
            if not isinstance(observed, bool):
                raise L0LiveAdapterError("VERIFIER_ERROR", "canonical state reader did not return bool")
            return observed is desired

        return read

    def initialize(self, episode: Any, evaluator_state: Any) -> Mapping[str, Any]:
        """Freeze initial truth and return the admission snapshot."""

        if self._binding is None:
            raise L0LiveAdapterError("STALE_BINDING", "adapter is not bound")
        if _task_from(evaluator_state) is not self._binding.task:
            raise L0LiveAdapterError("STALE_BINDING", "evaluator task changed without rebind")
        try:
            if self.route == OFFICIAL_FULL:
                self._verifier = OfficialPredicateVerifier(
                    require_single_option=False, predicate_mode=self.predicate_mode,
                )
                raw = self._verifier.evaluate(episode, evaluator_state)
                report = self._official_report(raw, phase="initial", done_requested=False)
            else:
                assert self._projected_verifier_class is not None
                reader = self.truth_reader()
                goal_spec = {
                    "route": self.route,
                    "clauses": deepcopy(self.task_spec["private"]["verifier"]["clauses"]),
                }
                self._verifier = self._projected_verifier_class(
                    goal_spec,
                    reader,
                    stability_window=self.stability_window,
                )
                report = self._projected_report(
                    self._verifier.snapshot(reader),
                    phase="initial",
                )
        except Exception as exc:
            report = self._fail_closed_report(exc, phase="initial", done_requested=False)
        self._initialized = True
        self._initial_goal_satisfied = bool(report.get("initial_goal_satisfied"))
        return report

    def snapshot(
        self,
        episode: Any,
        evaluator_state: Any,
        *,
        phase: str = "after_action",
        done_requested: bool = False,
    ) -> Mapping[str, Any]:
        """Re-read live truth and return one private, fail-closed snapshot."""

        if not self._initialized or self._binding is None or self._verifier is None:
            raise L0LiveAdapterError("STALE_BINDING", "adapter verifier is not initialized")
        if phase not in {"after_action", "settle_final"}:
            raise L0LiveAdapterError("VERIFIER_ERROR", f"unsupported verification phase: {phase!r}")
        if _task_from(evaluator_state) is not self._binding.task:
            return self._fail_closed_report(
                L0LiveAdapterError("STALE_BINDING", "evaluator task changed without rebind"),
                phase=phase,
                done_requested=done_requested,
            )
        try:
            if self.route == OFFICIAL_FULL:
                raw = self._verifier.evaluate(episode, evaluator_state)
                return self._official_report(raw, phase=phase, done_requested=done_requested)
            return self._projected_report(
                self._verifier.snapshot(self.truth_reader(), done_requested=done_requested),
                phase=phase,
            )
        except Exception as exc:
            return self._fail_closed_report(exc, phase=phase, done_requested=done_requested)

    def _official_report(self, raw: Any, *, phase: str, done_requested: bool) -> Mapping[str, Any]:
        operational = bool(getattr(raw, "ok", False))
        if not operational:
            raise L0LiveAdapterError(
                "VERIFIER_ERROR",
                str(getattr(raw, "error", None) or "official verifier failed closed"),
            )
        binding_pass = bool(
            self._binding is not None
            and self._binding.receipt.get("binding_pass") is True
        )
        current_goal = operational and bool(getattr(raw, "official_success", False))
        if phase == "initial":
            self._initial_goal_satisfied = current_goal
            self._official_streak = 0
        else:
            candidate = (
                binding_pass
                and current_goal
                and not self._initial_goal_satisfied
            )
            self._official_streak = self._official_streak + 1 if candidate else 0
        success = bool(
            binding_pass
            and phase != "initial"
            and current_goal
            and not self._initial_goal_satisfied
            and self._official_streak >= self.stability_window
        )
        q_score = (
            0.0
            if self._initial_goal_satisfied or not binding_pass
            else float(getattr(raw, "official_q", 0.0))
        )
        return {
            "schema_version": RECEIPT_VERSION,
            "private": True,
            "task_spec_id": self.task_spec_id,
            "route": self.route,
            "phase": phase,
            "predicate_mode": self.predicate_mode,
            "operational": operational,
            "admission_pass": bool(
                binding_pass
                and operational
                and phase == "initial"
                and not current_goal
            ),
            "success": success,
            "q_score": 1.0 if success else q_score,
            "initial_goal_satisfied": self._initial_goal_satisfied,
            "current_goal_satisfied": current_goal,
            "leaf_truth": dict(getattr(raw, "leaf_truth", {})),
            "stability": {
                "required": self.stability_window,
                "consecutive": self._official_streak,
                "current_goal_satisfied": current_goal,
                "complete": self._official_streak >= self.stability_window,
            },
            "early_done": bool(done_requested and not success),
            "failure_code": (
                "LIVE_SCOPE_BINDING_DEFERRED"
                if not binding_pass
                else "INITIAL_GOAL_ALREADY_SATISFIED"
                if phase == "initial" and current_goal
                else None
            ),
            "errors": [],
        }

    def _projected_report(self, report: Mapping[str, Any], *, phase: str) -> Mapping[str, Any]:
        result = dict(report)
        result.update(
            {
                "schema_version": RECEIPT_VERSION,
                "private": True,
                "task_spec_id": self.task_spec_id,
                "route": self.route,
                "phase": phase,
                "predicate_mode": self.predicate_mode,
                "admission_pass": bool(
                    phase == "initial"
                    and result.get("operational") is True
                    and not result.get("initial_goal_satisfied")
                ),
                "failure_code": (
                    "INITIAL_GOAL_ALREADY_SATISFIED"
                    if phase == "initial" and result.get("initial_goal_satisfied")
                    else "VERIFIER_ERROR"
                    if result.get("operational") is not True
                    else None
                ),
            }
        )
        return result

    def _fail_closed_report(
        self,
        exc: Exception,
        *,
        phase: str,
        done_requested: bool,
    ) -> Mapping[str, Any]:
        code = exc.code if isinstance(exc, L0LiveAdapterError) else "VERIFIER_ERROR"
        return {
            "schema_version": RECEIPT_VERSION,
            "private": True,
            "task_spec_id": self.task_spec_id,
            "route": self.route,
            "phase": phase,
            "operational": False,
            "admission_pass": False,
            "success": False,
            "q_score": 0.0,
            "initial_goal_satisfied": self._initial_goal_satisfied,
            "current_goal_satisfied": False,
            "stability": {
                "required": self.stability_window,
                "consecutive": 0,
                "current_goal_satisfied": False,
                "complete": False,
            },
            "early_done": bool(done_requested),
            "failure_code": code,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }


class L0TaskSpecEpisodeVerifier:
    """Expose one bound TaskSpec through the resident verifier API."""

    def __init__(
        self,
        task_spec: Mapping[str, Any],
        *,
        evaluator_state: Any,
        runtime_identity: Mapping[str, Any],
        marker_session_id: str,
        predicate_mode: str = "official",
    ) -> None:
        from behavior_hl.embodied.evaluator_bridge.r1pro_b1_component_builders import (
            _read_marker_symbolic_canonical_state,
        )

        route = task_spec.get("private", {}).get("verifier", {}).get("route")
        self.adapter = L0TaskSpecLiveAdapter(
            task_spec,
            expected_task_spec_sha256=canonical_task_spec_sha256(task_spec),
            stability_window=1,
            predicate_mode=predicate_mode,
            action_outcome_verifier=(
                None
                if route == OFFICIAL_FULL
                else Action23OutcomeVerifier(
                    read_state=partial(
                        _read_marker_symbolic_canonical_state,
                        symbolic_on_top=predicate_mode == "action20_symbolic",
                    ),
                    read_held=lambda: None,
                )
            ),
        )
        binding = self.adapter.bind(
            evaluator_state,
            runtime_identity=runtime_identity,
            marker_session_id=marker_session_id,
        )
        if binding.get("binding_pass") is not True:
            raise L0LiveAdapterError(
                "TASKSPEC_BINDING_ERROR", "TaskSpec live binding did not pass"
            )
        self._initial_report = self.adapter.initialize(None, evaluator_state)
        if self._initial_report.get("admission_pass") is not True:
            raise L0LiveAdapterError(
                "TASKSPEC_BINDING_ERROR", "TaskSpec initial admission did not pass"
            )
        self._first = True

    @staticmethod
    def _snapshot(report: Mapping[str, Any]) -> PrivatePredicateSnapshot:
        if report.get("operational") is not True:
            return PrivatePredicateSnapshot(
                ok=False,
                official_success=False,
                official_q=None,
                error="; ".join(map(str, report.get("errors", ())))
                or str(report.get("failure_code") or "TaskSpec verifier failed"),
            )
        leaf_truth = dict(report.get("leaf_truth", {}))
        if not leaf_truth:
            leaf_truth = {
                f"l0_clause/{row['index']}": bool(row.get("satisfied"))
                for row in report.get("clause_reports", ())
            }
        if not leaf_truth:
            raise L0LiveAdapterError(
                "VERIFIER_ERROR", "TaskSpec verifier returned no goal leaves"
            )
        return PrivatePredicateSnapshot(
            ok=True,
            official_success=bool(report.get("success")),
            official_q=float(report.get("q_score", 0.0)),
            leaf_truth=leaf_truth,
            research_diagnostics={"l0_taskspec_report": deepcopy(dict(report))},
        )

    def evaluate(self, _episode: Any, evaluator_state: Any) -> PrivatePredicateSnapshot:
        if self._first:
            self._first = False
            report = self._initial_report
        else:
            report = self.adapter.snapshot(None, evaluator_state, phase="after_action")
        return self._snapshot(report)

    def settle_final(self, evaluator_state: Any) -> PrivatePredicateSnapshot:
        return self._snapshot(
            self.adapter.snapshot(None, evaluator_state, phase="settle_final")
        )

    def retire(self) -> None:
        self.adapter.retire()


__all__ = [
    "L0LiveAdapterError",
    "L0TaskSpecEpisodeVerifier",
    "L0TaskSpecLiveAdapter",
    "canonical_task_spec_sha256",
]
