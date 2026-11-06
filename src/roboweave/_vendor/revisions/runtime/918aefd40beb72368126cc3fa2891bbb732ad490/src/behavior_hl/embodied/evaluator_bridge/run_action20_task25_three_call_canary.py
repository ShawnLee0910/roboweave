"""One-load Task25 Action20 ``place_next_to`` three-call atomic canary."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.action20_task25_three_call_spec import (
    ACTION20_TASK25_THREE_CALL_SPEC_VERSION,
    TASK25_PLACE_NEXT_TO_THREE_CALL_SPEC,
    Task25PlaceNextToThreeCallSpec,
    validate_task25_place_next_to_three_call_spec,
)
from behavior_hl.embodied.evaluator_bridge.run_action20_task25_atomic_canary import (
    _assert_carry_boundary,
    _default_role_provider,
    _entity_name,
    _entity_xy_yaw,
    _frame_boundary_record,
    _private_workbench_front_view_setup,
    _record_stage,
    _scene_objects,
    _write_json,
)
from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    candidate_aabb,
    canonical_relation,
    default_lateral_candidates,
    infer_unique_on_top_support,
    object_aabb,
    select_next_to_screen_candidate,
)


ACTION20_TASK25_THREE_CALL_RUNNER_VERSION = (
    "behavior2026.action20-task25-place-next-to-three-call-runner/v3"
)
PHASE_NAME = "action20_task25_place_next_to_three_call"
_THREE_RGB_CAMERA_ROLES = ("left_wrist", "right_wrist", "head")


@dataclass(frozen=True, slots=True)
class Task25ThreeCallFixture:
    source: Any = field(repr=False)
    reference: Any = field(repr=False)
    support: Any = field(repr=False)
    source_name_private: str
    reference_name_private: str
    support_name_private: str
    selector_private: str
    initial_next_to_private: bool
    selected_candidate_private: str
    support_margin_m_private: float
    pair_probe_private: tuple[Mapping[str, Any], ...] = field(repr=False)

    def to_private_record(self) -> dict[str, Any]:
        return {
            "schema_version": "behavior2026.task25-three-call-fixture/v1",
            "source_name_private": self.source_name_private,
            "reference_name_private": self.reference_name_private,
            "support_name_private": self.support_name_private,
            "selector_private": self.selector_private,
            "initial_next_to_private": self.initial_next_to_private,
            "selected_candidate_private": self.selected_candidate_private,
            "support_margin_m_private": self.support_margin_m_private,
            "source_policy": "all eligible graspables",
            "reference_policy": (
                "distinct initially-non-NextTo graspable with one canonical support"
            ),
            "ranking_policy": "max_selected_candidate_support_margin",
            "private_only": True,
        }

    def probe_record(self) -> dict[str, Any]:
        accepted = tuple(
            item for item in self.pair_probe_private if item.get("status") == "ACCEPT"
        )
        return {
            "schema_version": "behavior2026.task25-three-call-fixture-probe/v1",
            "status": "PASS",
            "pair_scope": "all_graspable_ordered_pairs",
            "pair_count": len(self.pair_probe_private),
            "accepted_pair_count": len(accepted),
            "ranking_policy": "max_selected_candidate_support_margin",
            "selected_source_name_private": self.source_name_private,
            "selected_reference_name_private": self.reference_name_private,
            "selected_support_name_private": self.support_name_private,
            "selected_candidate_private": self.selected_candidate_private,
            "selected_support_margin_m_private": self.support_margin_m_private,
            "pairs_private": [dict(item) for item in self.pair_probe_private],
            "private_only": True,
        }


class Task25ThreeCallFixtureError(RuntimeError):
    def __init__(self, message: str, probe_private: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.probe_private = dict(probe_private)


def _failed_fixture_probe(
    pair_records: tuple[Mapping[str, Any], ...],
    *,
    error: str,
) -> dict[str, Any]:
    return {
        "schema_version": "behavior2026.task25-three-call-fixture-probe/v1",
        "status": "FAIL",
        "pair_scope": "all_graspable_ordered_pairs",
        "pair_count": len(pair_records),
        "accepted_pair_count": sum(
            item.get("status") == "ACCEPT" for item in pair_records
        ),
        "ranking_policy": "max_selected_candidate_support_margin",
        "error_private": error,
        "pairs_private": [dict(item) for item in pair_records],
        "private_only": True,
    }


def _default_runtime_api() -> Any:
    return importlib.import_module(
        "behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout"
    )


def resolve_task25_three_call_fixture(
    evaluator: Any,
    profile: Any,
    *,
    role_provider: Callable[
        [Any, Any], tuple[Mapping[str, Any], Mapping[str, Any]]
    ]
    | None = None,
    selector: str = "screen_right",
    preferred_station_m: float = 1.20,
) -> Task25ThreeCallFixture:
    spec = TASK25_PLACE_NEXT_TO_THREE_CALL_SPEC
    if (
        getattr(profile, "task_id", None),
        getattr(profile, "task_name", None),
        getattr(profile, "scene_model", None),
    ) != (spec.task_id, spec.task_name, spec.scene_model):
        probe = _failed_fixture_probe((), error="loaded task identity drifted")
        raise Task25ThreeCallFixtureError(
            "loaded task differs from Task25 three-call spec",
            probe,
        )
    provider = role_provider or _default_role_provider
    graspables, _containers = provider(evaluator, profile)
    eligible = tuple(sorted(graspables.values(), key=_entity_name))
    pair_records: list[dict[str, Any]] = []
    if len(eligible) < 2 or len({id(item) for item in eligible}) != len(eligible):
        probe = _failed_fixture_probe(
            (),
            error="fixture needs two distinct graspables",
        )
        raise Task25ThreeCallFixtureError(
            "Task25 three-call fixture needs two distinct graspables",
            probe,
        )

    scene_objects = _scene_objects(evaluator)
    ranked: list[
        tuple[
            float,
            int,
            str,
            str,
            str,
            Any,
            Any,
            Any,
            str,
        ]
    ] = []
    for source in eligible:
        source_name = _entity_name(source)
        for reference in eligible:
            if reference is source:
                continue
            reference_name = _entity_name(reference)
            base_record: dict[str, Any] = {
                "source_name_private": source_name,
                "reference_name_private": reference_name,
                "selector_private": selector,
            }
            try:
                support, support_names = infer_unique_on_top_support(
                    reference,
                    scene_objects,
                )
            except Exception as exc:
                pair_records.append(
                    {
                        **base_record,
                        "status": "REJECT",
                        "reject_reason": "reference_support_query_failed",
                        "error_type_private": type(exc).__name__,
                    }
                )
                continue
            if support is None:
                pair_records.append(
                    {
                        **base_record,
                        "status": "REJECT",
                        "reject_reason": "reference_support_not_unique",
                        "reference_support_names_private": list(support_names),
                    }
                )
                continue
            support_name = _entity_name(support)
            base_record.update(
                {
                    "support_name_private": support_name,
                    "reference_support_names_private": list(support_names),
                }
            )
            try:
                initial_next_to = canonical_relation(source, "NextTo", reference)
            except Exception as exc:
                pair_records.append(
                    {
                        **base_record,
                        "status": "REJECT",
                        "reject_reason": "initial_next_to_query_failed",
                        "error_type_private": type(exc).__name__,
                    }
                )
                continue
            if initial_next_to:
                pair_records.append(
                    {
                        **base_record,
                        "status": "REJECT",
                        "reject_reason": "initial_next_to_already_true",
                        "initial_next_to_private": True,
                    }
                )
                continue
            try:
                tx, ty, target_yaw = _entity_xy_yaw(reference)
                observer_xy = (
                    tx + preferred_station_m * math.cos(target_yaw),
                    ty + preferred_station_m * math.sin(target_yaw),
                )
                candidates = default_lateral_candidates(
                    source,
                    reference,
                    support,
                    relation="NextTo",
                )
                selected = select_next_to_screen_candidate(
                    candidates,
                    held=source,
                    target=reference,
                    observer_xy=observer_xy,
                    selector=selector,
                )
            except Exception as exc:
                pair_records.append(
                    {
                        **base_record,
                        "status": "REJECT",
                        "reject_reason": "selector_geometry_query_failed",
                        "initial_next_to_private": False,
                        "error_type_private": type(exc).__name__,
                    }
                )
                continue
            if len(selected) != 1:
                pair_records.append(
                    {
                        **base_record,
                        "status": "REJECT",
                        "reject_reason": "screen_selector_not_unique",
                        "initial_next_to_private": False,
                        "candidate_names_private": [
                            candidate.name for candidate in candidates
                        ],
                        "selected_candidate_count_private": len(selected),
                    }
                )
                continue
            selected_candidate = selected[0]
            try:
                proposed_min, proposed_max = candidate_aabb(
                    source,
                    selected_candidate,
                )
                support_min, support_max = object_aabb(support)
                edge_clearances = (
                    float(proposed_min[0] - support_min[0]),
                    float(proposed_min[1] - support_min[1]),
                    float(support_max[0] - proposed_max[0]),
                    float(support_max[1] - proposed_max[1]),
                )
                support_margin = min(edge_clearances)
                if not math.isfinite(support_margin) or support_margin < 0.0:
                    raise ValueError("selected candidate exceeds its support AABB")
            except Exception as exc:
                pair_records.append(
                    {
                        **base_record,
                        "status": "REJECT",
                        "reject_reason": "support_margin_invalid",
                        "initial_next_to_private": False,
                        "selected_candidate_private": selected_candidate.name,
                        "error_type_private": type(exc).__name__,
                    }
                )
                continue
            pair_records.append(
                {
                    **base_record,
                    "status": "ACCEPT",
                    "initial_next_to_private": False,
                    "candidate_names_private": [
                        candidate.name for candidate in candidates
                    ],
                    "selected_candidate_private": selected_candidate.name,
                    "support_edge_clearances_m_private": list(edge_clearances),
                    "support_margin_m_private": support_margin,
                }
            )
            ranked.append(
                (
                    -support_margin,
                    -len(candidates),
                    source_name,
                    reference_name,
                    support_name,
                    source,
                    reference,
                    support,
                    selected_candidate.name,
                )
            )

    frozen_records = tuple(pair_records)
    if not ranked:
        probe = _failed_fixture_probe(
            frozen_records,
            error="no feasible initially-non-NextTo pair",
        )
        raise Task25ThreeCallFixtureError(
            "Task25 three-call fixture has no feasible initially-non-NextTo pair",
            probe,
        )
    (
        negative_margin,
        _negative_candidate_count,
        source_name,
        reference_name,
        support_name,
        source,
        reference,
        support,
        selected_candidate_name,
    ) = sorted(ranked, key=lambda item: item[:5])[0]
    return Task25ThreeCallFixture(
        source=source,
        reference=reference,
        support=support,
        source_name_private=source_name,
        reference_name_private=reference_name,
        support_name_private=support_name,
        selector_private=selector,
        initial_next_to_private=False,
        selected_candidate_private=selected_candidate_name,
        support_margin_m_private=-negative_margin,
        pair_probe_private=frozen_records,
    )


def _snapshot_binding(
    runtime: Any,
    *,
    action_index: int,
    binding_kind: str,
    stage: str,
    target_identity: Any | None = None,
    selector: str | None = None,
) -> Any:
    snapshot = getattr(runtime.store, "active_snapshot", None)
    if (
        snapshot is None
        or getattr(snapshot, "created_action_index", None) != action_index
    ):
        raise RuntimeError("Task25 three-call marker namespace is not fresh")
    if getattr(snapshot, "operation", None) != "place_next_to":
        raise RuntimeError("Task25 three-call marker operation drifted")
    if getattr(snapshot, "stage", None) != stage:
        raise RuntimeError("Task25 three-call marker stage drifted")
    bindings = getattr(snapshot, "bindings", None)
    if not isinstance(bindings, Mapping):
        raise RuntimeError("Task25 three-call marker bindings are invalid")
    matches = [
        binding
        for binding in bindings.values()
        if getattr(binding, "binding_kind", None) == binding_kind
        and (
            target_identity is None
            or getattr(binding, "target_token", None) is target_identity
            or getattr(
                getattr(binding, "target_token", None),
                "target_token",
                None,
            )
            is target_identity
        )
        and (selector is None or getattr(binding, "selector", None) == selector)
    ]
    if len(matches) != 1:
        raise RuntimeError("Task25 three-call marker binding is not unique")
    marker = getattr(matches[0], "marker", None)
    if not isinstance(marker, str) or not marker.startswith("C"):
        raise RuntimeError("Task25 three-call marker label is invalid")
    return matches[0]


def _canonical_next_to(source: Any, reference: Any) -> bool:
    return canonical_relation(source, "NextTo", reference)


def _selected_relation_report(runtime: Any) -> Mapping[str, Any]:
    getter = getattr(
        runtime.symbolic_primitives,
        "private_oracle_exact_report",
        None,
    )
    if not callable(getter):
        raise RuntimeError("Task25 three-call relation report is unavailable")
    report = getter()
    if not isinstance(report, Mapping):
        raise RuntimeError("Task25 three-call relation report is invalid")
    required = {
        "operation": "place_next_to",
        "partial_report": False,
        "placement_committed": True,
        "error": None,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise RuntimeError("Task25 selected-relation verifier failed")
    selected = report.get("selected_candidate_private")
    if not isinstance(selected, str) or not selected:
        raise RuntimeError("Task25 selected relation has no committed candidate")
    stability = report.get("stability_private")
    if not isinstance(stability, Mapping) or stability.get("accepted") is not True:
        raise RuntimeError("Task25 selected relation lacks accepted stability")
    relation_window = stability.get("canonical_relation_window")
    if not isinstance(relation_window, list) or not relation_window or not all(
        value is True for value in relation_window
    ):
        raise RuntimeError("Task25 canonical NextTo window is incomplete")
    return dict(report)


def _private_initial_three_rgb_observation_record(
    observation: Any,
) -> Mapping[str, Any]:
    simulator_step = getattr(observation, "simulator_step", None)
    if (
        isinstance(simulator_step, bool)
        or not isinstance(simulator_step, int)
        or simulator_step < 0
    ):
        raise RuntimeError("Task25 initial observation lacks a simulator step")
    values = getattr(observation, "values", None)
    if not isinstance(values, Mapping):
        raise RuntimeError("Task25 initial observation values are invalid")
    cameras = values.get("cameras")
    if not isinstance(cameras, Mapping) or tuple(cameras) != _THREE_RGB_CAMERA_ROLES:
        raise RuntimeError("Task25 initial observation is not exact three-RGB")
    camera_records: dict[str, Mapping[str, Any]] = {}
    for role in _THREE_RGB_CAMERA_ROLES:
        modalities = cameras.get(role)
        if not isinstance(modalities, Mapping) or set(modalities) != {"rgb"}:
            raise RuntimeError("Task25 initial observation camera is not RGB-only")
        rgb = np.asarray(modalities["rgb"])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise RuntimeError("Task25 initial observation RGB is invalid")
        frozen = np.ascontiguousarray(rgb)
        camera_records[role] = {
            "shape": [int(value) for value in frozen.shape],
            "dtype": str(frozen.dtype),
            "rgb_sha256": hashlib.sha256(frozen.tobytes()).hexdigest(),
        }
    return {
        "schema_version": "behavior2026.task25-private-initial-three-rgb/v1",
        "status": "PASS",
        "sequence": 0,
        "operation": "resident_initial_observe",
        "simulator_step": simulator_step,
        "camera_roles": list(_THREE_RGB_CAMERA_ROLES),
        "cameras": camera_records,
        "observation_call_count": 1,
        "public_action_index_before": 0,
        "public_action_index_after": 0,
        "private_action_count_before": 0,
        "next_operation": "private_grasp",
        "raw_rgb_persisted": False,
        "private_only": True,
    }


def run_action20_task25_three_call_canary(
    *,
    evaluator: Any,
    control: Any,
    profile: Any,
    run_dir: Path,
    stages: Path | None = None,
    spec: Task25PlaceNextToThreeCallSpec = TASK25_PLACE_NEXT_TO_THREE_CALL_SPEC,
    runtime_api: Any | None = None,
    role_provider: Callable[
        [Any, Any], tuple[Mapping[str, Any], Mapping[str, Any]]
    ]
    | None = None,
    private_visibility_provider: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Run private setup followed by exactly three public placement turns."""

    validate_task25_place_next_to_three_call_spec(spec)
    api = runtime_api or _default_runtime_api()
    phase_dir = Path(run_dir) / "strategies" / PHASE_NAME
    phase_dir.mkdir(parents=True, exist_ok=False)
    _record_stage(api, stages, "ACTION20_TASK25_THREE_CALL_START")
    result: dict[str, Any] = {
        "schema_version": ACTION20_TASK25_THREE_CALL_SPEC_VERSION,
        "runner_version": ACTION20_TASK25_THREE_CALL_RUNNER_VERSION,
        "status": "FAIL",
        "public_turns": [],
        "scene_load_count": 1,
        "setup_public_action_count": 0,
        "official_challenge_eligible": False,
        "execution_fidelity": "kinematic_research",
        "promotion_label": "STATIC_CONTRACT_ONLY",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(phase_dir / "atomic_spec.json", spec.to_record())
    runtime = None
    try:
        try:
            fixture = resolve_task25_three_call_fixture(
                evaluator,
                profile,
                role_provider=role_provider,
                selector=spec.selector,
                preferred_station_m=spec.preferred_station_m,
            )
        except Task25ThreeCallFixtureError as exc:
            _write_json(
                phase_dir / "private" / "fixture_probe.json",
                exc.probe_private,
            )
            raise
        _write_json(
            phase_dir / "private" / "fixture_probe.json",
            fixture.probe_record(),
        )
        _write_json(
            phase_dir / "private" / "fixture.json",
            fixture.to_private_record(),
        )
        starter = api._build_full_motion_safety_starter(evaluator)
        runtime = api._build_marker_runtime(
            evaluator=evaluator,
            control=control,
            profile=profile,
            full_actions=True,
            motion_safety_starter=starter,
            suppress_held_object_environment_collisions=False,
            logical_inventory_carry=True,
            base_execution=spec.base_execution,
            observation_profile=spec.observation_profile,
            action_profile="marker_symbolic_action20",
        )
        _assert_carry_boundary(runtime, expected_held=False, expected_identity=None)

        initial_observation = runtime.environment.observe()
        initial_observation_record = _private_initial_three_rgb_observation_record(
            initial_observation
        )
        _write_json(
            phase_dir / "private" / "setup" / "initial_three_rgb_observation.json",
            initial_observation_record,
        )

        private_index = 0
        private_steps = 0
        setup_actions: list[dict[str, Any]] = []
        private_index, used, grasp_report = api._oracle_execute_exact_private_symbolic(
            runtime=runtime,
            phase_dir=phase_dir / "private" / "setup",
            action_index=private_index,
            target=fixture.source,
            operation="grasp",
        )
        if grasp_report.get("status") != "completed":
            raise RuntimeError("Task25 private source grasp failed")
        private_steps += used
        setup_actions.append({"operation": "grasp", "report": dict(grasp_report)})
        _assert_carry_boundary(
            runtime,
            expected_held=True,
            expected_identity=fixture.source,
        )
        station_report = _private_workbench_front_view_setup(
            runtime=runtime,
            evaluator=evaluator,
            target=fixture.reference,
            desired_distance_m=spec.preferred_station_m,
            hard_distance_m=spec.hard_reach_m,
            observation_provider=api._capture_rendered_oracle_observation,
            visibility_provider=private_visibility_provider,
        )
        _write_json(
            phase_dir / "private" / "setup.json",
            {
                "status": "PASS",
                "private_action_count": len(setup_actions),
                "public_action_count": 0,
                "public_action_index_increment": 0,
                "initial_observation_count": 1,
                "initial_observation_artifact": (
                    "setup/initial_three_rgb_observation.json"
                ),
                "initial_observation_simulator_step": (
                    initial_observation_record["simulator_step"]
                ),
                "private_simulator_steps": private_steps,
                "actions_private": setup_actions,
                "station_private": dict(station_report),
                "subject_recording_started_after_setup": True,
            },
        )

        action_index = 0
        held = fixture.source

        def dispatch(marker: str | None) -> tuple[Any, tuple[Any, ...], int]:
            nonlocal action_index
            _assert_carry_boundary(
                runtime,
                expected_held=True,
                expected_identity=held,
            )
            action = runtime.registry.parse_payload(spec.payload(marker))
            action_result, frames, steps = api._execute(
                runtime=runtime,
                action=action,
                action_index=action_index,
                phase_dir=phase_dir / "subject",
            )
            turn = spec.turns[action_index]
            if action_result.status != turn.expected_status:
                raise RuntimeError("Task25 three-call public status drifted")
            result["public_turns"].append(
                {
                    "ordinal": turn.ordinal,
                    "action_index": action_index,
                    "action": spec.action,
                    "marker": marker,
                    "status": action_result.status,
                    "simulator_steps": steps,
                    "public_feedback": dict(action_result.public_feedback),
                    "frames": _frame_boundary_record(frames),
                }
            )
            action_index += 1
            return action_result, frames, steps

        dispatch(None)
        reference_binding = _snapshot_binding(
            runtime,
            action_index=0,
            binding_kind="object",
            stage="reference",
            target_identity=fixture.reference,
        )
        dispatch(reference_binding.marker)
        position_binding = _snapshot_binding(
            runtime,
            action_index=1,
            binding_kind="position",
            stage="position",
            target_identity=fixture.reference,
            selector=spec.selector,
        )
        dispatch(position_binding.marker)
        if action_index != spec.public_turn_budget:
            raise RuntimeError("Task25 three-call turn count drifted")
        _assert_carry_boundary(runtime, expected_held=False, expected_identity=None)
        if not _canonical_next_to(fixture.source, fixture.reference):
            raise RuntimeError("Task25 canonical NextTo verifier failed")
        relation_report = _selected_relation_report(runtime)
        _write_json(
            phase_dir / "private" / "verifier.json",
            {
                "canonical_next_to": True,
                "hand_empty": True,
                "tracker_clear": True,
                "selected_position_pass": True,
                "selected_relation_report_private": relation_report,
            },
        )
        result.update(
            {
                "status": "PASS",
                "protocol_pass": True,
                "verifier_pass": True,
                "public_turn_count": len(result["public_turns"]),
                "setup_private_simulator_steps": private_steps,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _record_stage(api, stages, "ACTION20_TASK25_THREE_CALL_PASS")
        _write_json(phase_dir / "summary.json", result)
        return result
    except Exception as exc:
        result.update(
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _record_stage(
            api,
            stages,
            "ACTION20_TASK25_THREE_CALL_FAIL",
            error_type=type(exc).__name__,
        )
        _write_json(phase_dir / "summary.json", result)
        raise
    finally:
        if runtime is not None:
            close = getattr(runtime.environment, "close", None)
            if callable(close):
                close()


def make_action20_task25_three_call_callback(
    *,
    profile: Any | None = None,
    spec: Task25PlaceNextToThreeCallSpec = TASK25_PLACE_NEXT_TO_THREE_CALL_SPEC,
    runtime_api: Any | None = None,
    role_provider: Callable[
        [Any, Any], tuple[Mapping[str, Any], Mapping[str, Any]]
    ]
    | None = None,
    private_visibility_provider: Callable[..., Mapping[str, Any]] | None = None,
) -> Callable[..., Mapping[str, Any]]:
    validate_task25_place_next_to_three_call_spec(spec)
    resolved_profile = profile
    if resolved_profile is None:
        profiles = importlib.import_module(
            "behavior_hl.embodied.evaluator_bridge.inside_task_profiles"
        )
        resolved_profile = profiles.get_inside_task_profile(spec.task_name)

    def callback(
        *,
        evaluator: Any,
        control: Any,
        bindings: Any,
        run_dir: Path,
        stages: Path,
    ) -> Mapping[str, Any]:
        del bindings
        summary = run_action20_task25_three_call_canary(
            evaluator=evaluator,
            control=control,
            profile=resolved_profile,
            run_dir=run_dir,
            stages=stages,
            spec=spec,
            runtime_api=runtime_api,
            role_provider=role_provider,
            private_visibility_provider=private_visibility_provider,
        )
        success = summary.get("status") == "PASS"
        return {
            "schema_version": ACTION20_TASK25_THREE_CALL_SPEC_VERSION,
            "strategies": [summary],
            "action20_task25_three_call_success": success,
            "oracle_success": success,
            "model_lane_authorized": False,
            "qwen_attempted": False,
            "official_challenge_eligible": False,
            "execution_fidelity": "kinematic_research",
        }

    return callback


__all__ = [
    "ACTION20_TASK25_THREE_CALL_RUNNER_VERSION",
    "PHASE_NAME",
    "Task25ThreeCallFixture",
    "Task25ThreeCallFixtureError",
    "make_action20_task25_three_call_callback",
    "resolve_task25_three_call_fixture",
    "run_action20_task25_three_call_canary",
]
