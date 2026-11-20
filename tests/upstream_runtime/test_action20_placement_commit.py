from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from behavior_hl.embodied.evaluator_bridge import (
    run_b1_marker_symbolic_rollout as marker_rollout,
)
from behavior_hl.embodied.evaluator_bridge.deterministic_inside_placement import (
    Aabb3,
    plan_deterministic_inside_candidates,
    summarize_symbolic_inside_window,
)
from behavior_hl.embodied.evaluator_bridge.logical_inventory import (
    LogicalInventorySnapshot,
)
from behavior_hl.embodied.evaluator_bridge.marker_symbolic_inside_execution import (
    _empty_inside_plan_category,
    _inside_candidate_for_execution,
)
from behavior_hl.embodied.evaluator_bridge.marker_symbolic_grasp_backend import (
    MarkerSymbolicGraspBackendAdapter,
    _SymbolicToken,
)
from behavior_hl.embodied.evaluator_bridge.physical_inside_execution import (
    INSIDE_PHYSICAL_RELEASE_STEPS,
    PhysicalInsideExecution,
    _inside_entry_path,
    inside_entry_direction_candidates,
    physical_inside_consequence_commit_safe,
)
from behavior_hl.embodied.evaluator_bridge.semantic_inside_ledger import (
    SemanticInsideLedger,
)
from behavior_hl.embodied.evaluator_bridge.spatial_candidate_admission import (
    first_collision_free_approach_candidate,
    query_spatial_candidate_collision_overlap,
    query_symbolic_placement_final_collision,
)
from behavior_hl.embodied.evaluator_bridge.marker_symbolic_spatial_execution import (
    _PrivateVerifiedSpatialTerminalSuccess,
    _candidate_geometry_for_aabb,
    _candidate_from_spatial_execution,
    _empty_on_top_plan_category,
    _empty_under_plans_category,
    _fresh_official_inside_containment,
    _keep_logical_inventory_closure_still,
    _logical_inventory_descendants_stable,
    _ordered_floor_candidates,
    _placement_relation_target,
    _relation_target_role,
    _rollback_spatial_candidate_trial,
    _spatial_collision_failure_category,
    marker_symbolic_place_on_floor_generator,
    plan_live_under_supports,
)
from behavior_hl.embodied.evaluator_bridge.on_top_support_layers import (
    OnTopCandidateBinding,
    OnTopSupportLayerPlan,
    fitting_inside_orientation_matrices,
    on_top_binding_horizontal_distance,
    plan_on_top_support_layer_candidates,
    quaternion_to_matrix_xyzw,
    restrict_and_rank_on_top_plan,
    summarize_original_destination_on_top,
)
from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    DEFAULT_SETTLE_STEPS,
    DEFAULT_STABILITY_WINDOW,
    CandidateAdmissionReport,
    CollisionOverlapAdmissionReport,
    DownwardSupportRayEvidence,
    SpatialCandidatePlan,
    SpatialPoseCandidate,
    SpatialStabilitySample,
    build_downward_support_ray_requests,
    rebase_candidate_to_local_support,
    sample_spatial_stability,
    summarize_relation_authoritative_postcondition,
    summarize_symbolic_spatial_postcondition,
    symbolic_on_top_relation,
)
from behavior_hl.embodied.runtime.plugins.marker_symbolic_grasp import MarkerSymbolicV34OperationPlugin
from behavior_hl.embodied.runtime.public_action_feedback import (
    model_visible_action20_last_result,
    placement_failure_category,
)


def test_inside_orientation_ranking_prefers_a_closeable_flat_pose():
    extent = (0.08, 0.08, 0.30)
    rotations = fitting_inside_orientation_matrices(
        extent,
        (0.40, 0.30, 0.12),
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    assert rotations
    oriented_height = sum(
        abs(rotations[0][2, axis]) * extent[axis] for axis in range(3)
    )
    assert oriented_height <= 0.11


def test_inside_physical_consequence_requires_complete_finite_readback():
    report = {
        "physics_step_count": INSIDE_PHYSICAL_RELEASE_STEPS,
        "readback_samples": 1,
        "official_inside_window": [False],
        "nested_relations_window": [False],
        "pose_finite_window": [True],
        "failure_reasons": ["official_inside_not_satisfied"],
    }

    assert physical_inside_consequence_commit_safe(report) is True
    report["pose_finite_window"][0] = False
    assert physical_inside_consequence_commit_safe(report) is False

    penetration, overlap, opening_is_bounded = (
        PhysicalInsideExecution.plan_options()
    )
    assert penetration == np.nextafter(1.0, 0.0)
    assert overlap == np.nextafter(1.0, 0.0)
    assert opening_is_bounded is False


def test_physical_inside_entry_path_uses_slanted_opening_axes():
    desired = SpatialPoseCandidate(
        "inside", np.zeros(3), np.asarray((0.0, 0.0, 0.0, 1.0))
    )
    half_angle = np.pi / 8.0
    directions = inside_entry_direction_candidates(
        source_position=np.zeros(3),
        target_position=np.zeros(3),
        opening_orientation_xyzw=np.asarray(
            (0.0, 0.0, np.sin(half_angle), np.cos(half_angle))
        ),
    )

    slanted = next(
        direction
        for direction in directions
        if np.allclose(direction, (np.sqrt(0.5), np.sqrt(0.5), 0.0))
    )
    path = _inside_entry_path(
        desired,
        candidate_aabb=Aabb3.from_center_extent((0.0, 0.0, 0.0), (0.2,) * 3),
        fillable_aabb=Aabb3((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
        outward_direction=slanted,
    )

    assert path[0][2] is False
    assert path[-1][2] is True
    np.testing.assert_allclose(path[-1][0].position, desired.position)
    assert all(
        np.linalg.norm(right[0].position - left[0].position) <= 0.1 + 1.0e-9
        for left, right in zip(path[:-1], path[1:], strict=True)
    )


@pytest.mark.parametrize("object_extent", [0.2, 1.2])
def test_physical_inside_checks_the_complete_entry_path(monkeypatch, object_extent):
    import sys
    from behavior_hl.embodied.evaluator_bridge import physical_inside_execution

    desired = SpatialPoseCandidate(
        "inside",
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.0, 0.0, 0.0, 1.0)),
    )
    candidate_aabb = Aabb3.from_center_extent((0.0, 0.0, 0.0), (object_extent,) * 3)
    fillable_aabb = Aabb3((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    expected_path = _inside_entry_path(
        desired,
        candidate_aabb=candidate_aabb,
        fillable_aabb=fillable_aabb,
        outward_direction=(1.0, 0.0, 0.0),
    )
    first_center_inside = next(
        index
        for index, (_candidate, aabb, _inside) in enumerate(expected_path)
        if all(-0.5 <= value <= 0.5 for value in aabb.center)
    )
    release_index = next(
        (index for index, (_candidate, _aabb, inside) in enumerate(expected_path) if inside),
        first_center_inside,
    )
    if object_extent < 1.0:
        assert release_index > first_center_inside
    seen = []

    def clear_query(**kwargs):
        seen.append(kwargs["candidate"].name)
        return CollisionOverlapAdmissionReport(
            accepted=True,
            reason="clear",
            query_box=None,
            raw_hits=(),
            excluded_hits=(),
            blocking_hits=(),
        )

    monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace())
    monkeypatch.setattr(
        physical_inside_execution,
        "inside_entry_direction_candidates",
        lambda **_kwargs: (np.asarray((1.0, 0.0, 0.0)),),
    )
    monkeypatch.setattr(
        physical_inside_execution,
        "query_spatial_candidate_collision_overlap",
        clear_query,
    )
    posed = SimpleNamespace(
        get_position_orientation=lambda: (
            np.zeros(3),
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        )
    )
    physical = PhysicalInsideExecution(
        SimpleNamespace(_logical_inventory_snapshot=None)
    )
    admitted, reports, used_path = physical.admit_candidate(
        posed,
        posed,
        desired,
        candidate_aabb=candidate_aabb,
        fillable_aabb=fillable_aabb,
    )

    assert admitted is not None
    assert 0.0 <= admitted.position[0] <= 0.5
    assert used_path is True
    assert len(reports) == release_index + 1
    assert len(seen) == release_index + 1
    assert reports[0]["entry_release_fully_inside_private"] is (object_extent < 1.0)
    np.testing.assert_allclose(admitted.position, expected_path[release_index][0].position)

    def blocked_query(**_kwargs):
        return CollisionOverlapAdmissionReport(
            accepted=False,
            reason="collision_shape_overlap",
            query_box=None,
            raw_hits=(),
            excluded_hits=(),
            blocking_hits=(),
        )

    monkeypatch.setattr(
        physical_inside_execution,
        "query_spatial_candidate_collision_overlap",
        blocked_query,
    )
    blocked, blocked_reports, used_path = PhysicalInsideExecution(
        SimpleNamespace(_logical_inventory_snapshot=None)
    ).admit_candidate(
        posed,
        posed,
        desired,
        candidate_aabb=candidate_aabb,
        fillable_aabb=fillable_aabb,
    )

    assert blocked is None
    assert used_path is False
    assert len(blocked_reports) == 1


def test_symbolic_inside_candidate_defers_to_final_collision_admission():
    candidate = SpatialPoseCandidate(
        "inside", np.zeros(3), np.asarray((0.0, 0.0, 0.0, 1.0))
    )
    result = _inside_candidate_for_execution(
        None,
        SimpleNamespace(),
        SimpleNamespace(),
        candidate,
        candidate_aabb=Aabb3((-0.5,) * 3, (0.5,) * 3),
        fillable_aabb=Aabb3((-1.0,) * 3, (1.0,) * 3),
    )
    assert result[0] is candidate
    assert result[1:] == ((), False)


def test_physical_approach_uses_first_clear_release_pose():
    def collision_query(candidate):
        accepted = bool(0.25 <= candidate.position[0] < 0.75)
        return CollisionOverlapAdmissionReport(
            accepted=accepted,
            reason=None if accepted else "blocked",
            query_box=None,
            raw_hits=(),
            excluded_hits=(),
            blocking_hits=(),
        )

    candidate, reports = first_collision_free_approach_candidate(
        start=SpatialPoseCandidate(
            "start",
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        ),
        target=SpatialPoseCandidate(
            "target",
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0, 0.0)),
        ),
        collision_shape_query=collision_query,
        sample_count=4,
    )

    assert candidate is not None
    np.testing.assert_allclose(candidate.position, (0.25, 0.0, 0.0))
    np.testing.assert_allclose(candidate.orientation_xyzw, (0.0, 0.0, 0.0, 1.0))
    assert [report.accepted for report in reports] == [False, True]


def test_inside_orientation_ranking_keeps_a_fitting_pre_pick_pose_first():
    rotations = fitting_inside_orientation_matrices(
        (0.08, 0.08, 0.30),
        (0.40, 0.40, 0.50),
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(rotations[0], np.eye(3))


def test_nested_container_pick_normalizes_only_near_rigid_pose_drift():
    pose = np.eye(4)
    pose[:3, :3] *= 1.0 + 1.0e-5
    pose[:3, 3] = (1.0, 2.0, 3.0)

    normalized = marker_rollout._normalize_near_rigid_pose(pose)

    np.testing.assert_allclose(normalized[:3, 3], pose[:3, 3])
    np.testing.assert_allclose(
        normalized[:3, :3].T @ normalized[:3, :3],
        np.eye(3),
        atol=1.0e-12,
    )
    assert np.linalg.det(normalized[:3, :3]) == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="not near-rigid"):
        marker_rollout._normalize_near_rigid_pose(
            np.diag((2.0, 1.0, 1.0, 1.0))
        )


def test_spatial_trial_rollback_restores_checkpoint_before_carry():
    events = []
    world = {"spectator": "moved"}

    class Simulator:
        def load_state(self, checkpoint, *, serialized):
            assert checkpoint == {"spectator": "original"}
            assert serialized is False
            world.update(checkpoint)
            events.append("checkpoint")

    def reproject():
        assert world["spectator"] == "original"
        events.append("carry")
        return {"status": "PASS"}

    report = _rollback_spatial_candidate_trial(
        omnigibson=SimpleNamespace(sim=Simulator()),
        checkpoint={"spectator": "original"},
        reproject_carry=reproject,
    )

    assert events == ["checkpoint", "carry"]
    assert report == {"status": "PASS"}

    class BrokenSimulator:
        def load_state(self, _checkpoint, *, serialized):
            assert serialized is False
            events.append("broken_checkpoint")
            raise RuntimeError("restore failed")

    with pytest.raises(RuntimeError, match="restore failed"):
        _rollback_spatial_candidate_trial(
            omnigibson=SimpleNamespace(sim=BrokenSimulator()),
            checkpoint={},
            reproject_carry=lambda: events.append("recovered_carry") or {},
        )
    assert events[-2:] == ["broken_checkpoint", "recovered_carry"]


def test_spatial_admission_cannot_fail_open_after_decision():
    candidate = object()
    accepted = SimpleNamespace(
        decision=SimpleNamespace(
            accepted_for_execution=True,
            candidate=candidate,
        ),
        applied=True,
    )
    assert _candidate_from_spatial_execution(accepted) is candidate

    with pytest.raises(RuntimeError, match="did not apply"):
        _candidate_from_spatial_execution(
            SimpleNamespace(decision=accepted.decision, applied=False)
        )
    with pytest.raises(RuntimeError, match="rejected.*applied"):
        _candidate_from_spatial_execution(
            SimpleNamespace(
                decision=SimpleNamespace(
                    accepted_for_execution=False,
                    candidate=None,
                ),
                applied=True,
            )
        )


def test_spatial_collision_query_is_read_only_and_fails_closed():
    class PSQI:
        def __init__(self, *, error=None):
            self.error = error
            self.calls = []

        def overlap_box(self, **kwargs):
            self.calls.append(kwargs)
            if self.error is not None:
                raise self.error
            for hit in (
                SimpleNamespace(rigid_body="/held/link", collision=None),
                SimpleNamespace(rigid_body=None, collision="/target/collision"),
                SimpleNamespace(rigid_body="/support/child", collision=None),
                SimpleNamespace(rigid_body="/structural_blocker", collision=None),
            ):
                assert kwargs["reportFn"](hit) is True

    held = SimpleNamespace(link_prim_paths=lambda: ("/held",))
    child = SimpleNamespace(link_prim_paths=("/child",))
    target = SimpleNamespace(
        links={"main": SimpleNamespace(prim_path="/target")}
    )
    support = SimpleNamespace(prim_path="/support")
    candidate = SpatialPoseCandidate(
        name="candidate",
        position=np.zeros(3),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
    )
    bounds = (np.zeros(3), np.ones(3))
    psqi = PSQI()
    simulator = SimpleNamespace(psqi=psqi)

    report = query_spatial_candidate_collision_overlap(
        omnigibson=SimpleNamespace(sim=simulator),
        held=held,
        closure_objects=(held, child),
        geometry_certified_objects=(target, support),
        candidate=candidate,
        candidate_aabb_world=bounds,
    )

    assert len(psqi.calls) == 1
    assert report.accepted is False
    assert report.reason == "blocking_collision_shape_overlap"
    assert len(report.excluded_hits) == 3
    assert [item.rigid_body_path for item in report.blocking_hits] == [
        "/structural_blocker"
    ]
    assert report.query_box.geometry_source == "orientation_aware_candidate_aabb"
    assert _spatial_collision_failure_category(report) == (
        "placement_collision_blocked"
    )
    np.testing.assert_allclose(psqi.calls[0]["pos"], (0.5, 0.5, 0.5))

    broken = query_spatial_candidate_collision_overlap(
        omnigibson=SimpleNamespace(
            sim=SimpleNamespace(psqi=PSQI(error=RuntimeError("query failed")))
        ),
        held=held,
        closure_objects=(held, child),
        geometry_certified_objects=(target, support),
        candidate=candidate,
        candidate_aabb_world=bounds,
    )
    assert broken.accepted is False
    assert broken.reason == "collision_query_error"
    assert broken.query_error == "query failed"
    assert _spatial_collision_failure_category(broken) == (
        "execution_unverified"
    )


def test_empty_inside_plan_reports_only_exclusive_failure_causes():
    def plan(*, overlap=0, outside=0, orientation=0, intrusion=0):
        return SimpleNamespace(
            rejected_initial_overlap=overlap,
            rejected_outside_volume=outside,
            rejected_non_fitting_orientation=orientation,
            rejected_gross_intrusion=intrusion,
        )

    assert _empty_inside_plan_category(plan(overlap=4)) == (
        "placement_collision_blocked"
    )
    assert _empty_inside_plan_category(plan(intrusion=4)) == (
        "geometric_fit_blocked"
    )
    assert _empty_inside_plan_category(plan(overlap=2, intrusion=2)) is None
    assert placement_failure_category({"geometric_fit_blocked", "placement_collision_blocked"}) is None
    assert placement_failure_category({"execution_unverified", "placement_collision_blocked", None}) == "execution_unverified"


@pytest.mark.parametrize("operation", ("place_inside", "place_on_top", "place_on_floor", "place_next_to", "place_under"))
def test_placement_query_failure_survives_backend_plugin_and_public_feedback(operation):
    class ActionPrimitiveError(Exception):
        reason = SimpleNamespace(name="SAMPLING_ERROR")

    def failed_query(**_kwargs):
        raise RuntimeError("opaque private query failure")

    held = _Box((0, 0, 0), (0.1, 0.1, 0.1))
    report = query_spatial_candidate_collision_overlap(
        omnigibson=SimpleNamespace(sim=SimpleNamespace(psqi=SimpleNamespace(overlap_box=failed_query))),
        held=held, closure_objects=(held,),
        candidate=SpatialPoseCandidate("candidate", np.zeros(3), np.array((0, 0, 0, 1))),
    )
    error = ActionPrimitiveError("opaque placement failure")
    error.metadata = {"action20_blocked_category": _spatial_collision_failure_category(report)}
    backend = object.__new__(MarkerSymbolicGraspBackendAdapter)
    backend._execution_integrity_state = None
    backend._integrity_failure = None
    backend._held_object = lambda: held
    backend._hold_safe = lambda: None
    backend._last_private_diagnostic = {}
    token = _SymbolicToken(generator=None, operation=operation, acted_object=held, phase="symbolic")
    tick = backend._fail_token(token, error)
    state = SimpleNamespace(
        status=None, stage="symbolic", operation=operation, driver_token=token,
        backend=SimpleNamespace(tick_symbolic_operation=lambda _token: tick),
    )
    plugin = MarkerSymbolicV34OperationPlugin(operation)
    plugin.tick(state, None)
    result = plugin.finish(state, None)
    assert result.status == "uncertain"
    assert model_visible_action20_last_result(
        {"status": result.status, **result.public_feedback}, action=operation,
    ) == {"status": "blocked", "blocked_category": "execution_unverified"}


def test_symbolic_final_collision_checks_every_moving_object_and_target(monkeypatch):
    monkeypatch.setattr(
        "behavior_hl.embodied.evaluator_bridge.spatial_candidate_admission.target_solid_intersects_query_box",
        lambda *_: False,
    )
    class PSQI:
        def __init__(self, hits_by_call):
            self.hits_by_call = list(hits_by_call)
            self.calls = []

        def overlap_box(self, **kwargs):
            self.calls.append(kwargs)
            for path in self.hits_by_call.pop(0):
                assert kwargs["reportFn"](
                    SimpleNamespace(rigid_body=path, collision=None)
                ) is True

    def obj(name, path, lower, upper):
        return SimpleNamespace(
            name=name,
            prim_path=path,
            aabb=(np.asarray(lower), np.asarray(upper)),
            get_position_orientation=lambda: (
                np.asarray((0.5, 0.5, 0.5)),
                np.asarray((0.0, 0.0, 0.0, 1.0)),
            ),
        )

    root = obj("root", "/root", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    child = obj("child", "/child", (0.2, 0.2, 0.2), (0.4, 0.4, 0.4))
    target = obj("target", "/target", (-1.0, -1.0, -1.0), (2.0, 2.0, 0.0))
    psqi = PSQI((("/root/link", "/target/link"), ("/child/link",)))

    blocked = query_symbolic_placement_final_collision(
        omnigibson=SimpleNamespace(sim=SimpleNamespace(psqi=psqi)),
        moving_objects=(root, child),
    )

    assert blocked.accepted is False
    assert _spatial_collision_failure_category(blocked) == (
        "placement_collision_blocked"
    )
    assert len(blocked.object_reports) == 2
    assert blocked.object_reports[0][1].blocking_hits[0].rigid_body_path == (
        "/target/link"
    )
    assert psqi.calls[0]["halfExtent"] == pytest.approx(
        (0.4995, 0.4995, 0.4995)
    )

    inside_psqi = PSQI((("/root/link", "/target/link"), ("/child/link",)))
    inside = query_symbolic_placement_final_collision(
        omnigibson=SimpleNamespace(sim=SimpleNamespace(psqi=inside_psqi)),
        moving_objects=(root, child),
        contact_objects=(target,),
    )

    assert inside.accepted is True
    assert _spatial_collision_failure_category(inside) is None
    assert all(report.accepted for _name, report in inside.object_reports)

    occupied_psqi = PSQI(
        (("/target/link", "/existing_content/link"), ("/child/link",))
    )
    occupied = query_symbolic_placement_final_collision(
        omnigibson=SimpleNamespace(sim=SimpleNamespace(psqi=occupied_psqi)),
        moving_objects=(root, child),
        contact_objects=(target,),
    )

    assert occupied.accepted is False
    assert occupied.object_reports[0][1].blocking_hits[0].rigid_body_path == (
        "/existing_content/link"
    )


def test_only_verified_spatial_terminal_bypasses_generator_close_rollback():
    events = []

    def transaction():
        try:
            yield None
        except _PrivateVerifiedSpatialTerminalSuccess:
            events.append("verified_terminal")
            raise
        except GeneratorExit:
            events.append("rollback")
            raise

    ordinary = transaction()
    next(ordinary)
    marker_rollout._terminate_private_exact_generator(
        ordinary,
        preserve_verified_spatial_terminal=False,
    )
    verified = transaction()
    next(verified)
    marker_rollout._terminate_private_exact_generator(
        verified,
        preserve_verified_spatial_terminal=True,
    )

    assert events == ["rollback", "verified_terminal"]


def test_inside_candidates_fit_completely_without_overlapping_contents():
    fillable = Aabb3((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    occupied = Aabb3((0.4, 0.4, 0.0), (0.6, 0.6, 1.0))
    plan = plan_deterministic_inside_candidates(
        held_extent_m=(0.2, 0.2, 0.2),
        fillable_aabb=fillable,
        existing_content_aabbs={"occupied": occupied},
    )

    assert plan.candidates
    assert all(candidate.wall_clearance_m >= 0.0 for candidate in plan.candidates)
    assert all(
        not candidate.initial_aabb.intersects(occupied, clearance_m=0.002)
        for candidate in plan.candidates
    )
    assert not plan_deterministic_inside_candidates(
        held_extent_m=(1.1, 0.2, 0.2),
        fillable_aabb=fillable,
    ).candidates


def test_inside_commit_requires_three_consecutive_symbolic_samples():
    stable = summarize_symbolic_inside_window(
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
    )
    transient = summarize_symbolic_inside_window(
        [True, True, False],
        [True, True, True],
        [True, True, True],
        [True, True, True],
    )

    assert stable["accepted"] is True
    assert transient["accepted"] is False


def test_on_top_commit_rejects_motion_and_penetration_over_three_centimeters():
    moving = SpatialStabilitySample(
        relation=True,
        support_on_top=True,
        support_touching=True,
        bottom_to_support_gap_m=0.0,
        linear_speed_m_s=0.5,
        angular_speed_rad_s=0.5,
        tilt_rad=0.0,
        position=(0.0, 0.0, 1.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        bottom_z_m=1.0,
    )
    samples = [moving] * 10

    assert summarize_symbolic_spatial_postcondition(samples)["accepted"] is True
    strict = summarize_original_destination_on_top(samples)
    assert strict["accepted"] is False
    assert "linear_velocity_not_stable" in strict["failure_reasons"]

    stable = replace(moving, linear_speed_m_s=0.0, angular_speed_rad_s=0.0)
    shallow = summarize_original_destination_on_top(
        [replace(stable, bottom_to_support_gap_m=-0.02)] * 10
    )
    gross = summarize_original_destination_on_top(
        [replace(stable, bottom_to_support_gap_m=-0.04)] * 10
    )
    assert shallow["accepted"] is True
    assert gross["accepted"] is False
    assert "support_gap_out_of_bounds" in gross["failure_reasons"]


class _Box:
    def __init__(self, center, extent):
        self._center = center
        self._extent = extent
        self.states = {}

    def get_base_aligned_bbox(self):
        return self._center, (0.0, 0.0, 0.0, 1.0), self._extent, (0.0, 0.0, 0.0)

    def get_position_orientation(self):
        return self._center, (0.0, 0.0, 0.0, 1.0)

    def get_linear_velocity(self):
        return (0.0, 0.0, 0.0)

    def get_angular_velocity(self):
        return (0.0, 0.0, 0.0)

    @property
    def aabb(self):
        half = tuple(value / 2.0 for value in self._extent)
        return tuple(a - b for a, b in zip(self._center, half)), tuple(
            a + b for a, b in zip(self._center, half)
        )


@pytest.mark.parametrize("support_aabb_pad", (0.0, 0.05))
@pytest.mark.parametrize(
    "ceiling_z, floor_z, expected_clearance, accepted",
    (
        ((0.1,) * 5, (0.0,) * 5, 0.1, False),
        ((0.3,) * 5, (0.0,) * 5, 0.3, True),
        ((0.3, 0.3, 0.3, 0.3, 0.1), (0.0,) * 5, 0.1, False),
        ((0.3,) * 5, (0.0, 0.0, 0.0, 0.0, 0.1), 0.2, False),
        ((0.3, 0.5, 0.5, 0.5, 0.5), (0.0, 0.2, 0.2, 0.2, 0.2), 0.1, False),
    ),
)
def test_under_clearance_uses_actual_footprint_hit_heights(
    ceiling_z, floor_z, expected_clearance, accepted, support_aabb_pad,
):
    held = _Box((2.0, 2.0, 1.0), (0.1, 0.1, 0.2))
    target = _Box((0.0, 0.0, 0.4), (1.0, 1.0, 0.8))
    # Neither the furniture top nor the support's overall AABB top is the
    # local hit surface. Their heights may only bound the query/search.
    support = _Box(
        (0.0, 0.0, max(floor_z) - 0.5 + support_aabb_pad), (2.0, 2.0, 1.0),
    )
    for name, obj in (("held", held), ("target", target), ("support", support)):
        obj.link_prim_paths = (f"/World/{name}/base_link",)

    ray_calls = []

    def raytest_batch(starts, ends, **kwargs):
        upward = ends[0][2] > starts[0][2]
        ray_calls.append(upward)
        hit_object = target if upward else support
        heights = ceiling_z if upward else floor_z
        return [
            {
                "hit": True,
                "rigidBody": hit_object.link_prim_paths[0],
                "position": (start[0], start[1], z),
            }
            for start, z in zip(starts, heights, strict=True)
        ]

    plan, = plan_live_under_supports(
        held=held, target=target, supports=(support,), robot=object(),
        raytest_batch=raytest_batch,
    )
    reports = plan.candidate_plan.admission_reports
    assert len(reports) == 9
    assert ray_calls == [True, False] * 9
    assert bool(plan.candidate_plan.candidates) is accepted
    for report in reports:
        assert report.vertical_evidence.vertical_clearance_m == pytest.approx(
            expected_clearance,
        )
        assert report.accepted is accepted
        if not accepted:
            assert report.reason == "vertical clearance is too small for held object"
    if not accepted:
        assert _empty_under_plans_category((plan,)) == "geometric_fit_blocked"


def test_under_marker_existence_check_stops_after_first_usable_support():
    held = _Box((2.0, 2.0, 1.0), (0.1, 0.1, 0.2))
    target = _Box((0.0, 0.0, 0.4), (1.0, 1.0, 0.8))
    supports = tuple(
        _Box((0.0, 0.0, -0.5 - index), (2.0, 2.0, 1.0))
        for index in range(3)
    )
    for index, obj in enumerate((held, target, *supports)):
        obj.link_prim_paths = (f"/World/object_{index}/base_link",)
    ray_calls = []

    def raytest_batch(starts, ends, **kwargs):
        upward = ends[0][2] > starts[0][2]
        ray_calls.append(upward)
        hit_object = target if upward else supports[0]
        return [
            {
                "hit": True,
                "rigidBody": hit_object.link_prim_paths[0],
                "position": (start[0], start[1], 0.3 if upward else 0.0),
            }
            for start in starts
        ]

    plans = plan_live_under_supports(
        held=held,
        target=target,
        supports=supports,
        robot=object(),
        raytest_batch=raytest_batch,
        stop_after_first_usable=True,
    )
    assert len(plans) == 1
    assert plans[0].candidate_plan.candidates
    assert ray_calls == [True, False] * len(plans[0].candidate_plan.candidates)


@pytest.mark.parametrize("bad_hit", ("missing_position", "nan", "infinite", "short_batch"))
def test_under_clearance_fails_closed_without_complete_finite_hits(bad_hit):
    held = _Box((2.0, 2.0, 1.0), (0.1, 0.1, 0.2))
    target = _Box((0.0, 0.0, 0.4), (1.0, 1.0, 0.8))
    support = _Box((0.0, 0.0, -0.5), (2.0, 2.0, 1.0))
    target.link_prim_paths = ("/World/target/base_link",)
    support.link_prim_paths = ("/World/support/base_link",)

    def raytest_batch(starts, ends, **kwargs):
        upward = ends[0][2] > starts[0][2]
        obj = target if upward else support
        hits = [
            {
                "hit": True, "rigidBody": obj.link_prim_paths[0],
                "position": (start[0], start[1], 0.3 if upward else 0.0),
            }
            for start in starts
        ]
        if upward:
            if bad_hit == "missing_position":
                del hits[0]["position"]
            elif bad_hit == "short_batch":
                hits.pop()
            else:
                hits[0]["position"] = (
                    0.0, 0.0, float("nan" if bad_hit == "nan" else "inf"),
                )
        return hits

    plan, = plan_live_under_supports(
        held=held, target=target, supports=(support,), robot=object(),
        raytest_batch=raytest_batch,
    )
    assert plan.candidate_plan.admission_reports
    assert not plan.candidate_plan.candidates
    assert all(not report.accepted for report in plan.candidate_plan.admission_reports)
    assert _empty_under_plans_category((plan,)) == "execution_unverified"


@pytest.mark.parametrize("case,expected", (
    ("footprint", "geometric_fit_blocked"), ("no_hit", "placement_support_unavailable"),
    ("geometry_error", "execution_unverified"), ("query_error", "execution_unverified"),
))
def test_empty_under_plans_preserve_structured_rejection_evidence(case, expected):
    class UnreadableSupport:
        @property
        def aabb(self):
            raise RuntimeError("opaque geometry failure")

    held = _Box((2, 2, 1), (2, 2, 0.2) if case == "footprint" else (0.1, 0.1, 0.2))
    target = _Box((0, 0, 0.4), (1, 1, 0.8))
    support = UnreadableSupport() if case == "geometry_error" else _Box((0, 0, -0.5), (2, 2, 1))

    def rays(starts, ends, **_kwargs):
        if case == "query_error":
            raise RuntimeError("opaque ray failure")
        return [{"hit": False} for _ in starts]

    plans = plan_live_under_supports(held=held, target=target, supports=(support,), robot=object(), raytest_batch=rays)
    assert not any(plan.candidate_plan.candidates for plan in plans)
    assert _empty_under_plans_category(plans) == expected


def test_under_collision_rejection_is_not_missing_support():
    plan = SpatialCandidatePlan((), (
        CandidateAdmissionReport(
            candidate="pose", accepted=False, reason=None, blockers=("scene_object",),
        ),
    ))
    assert _empty_under_plans_category((SimpleNamespace(candidate_plan=plan),)) == "placement_collision_blocked"


@pytest.mark.parametrize("physical", (False, True))
@pytest.mark.parametrize("unreadable", (False, True))
def test_floor_absence_is_distinct_from_unreadable_geometry(physical, unreadable):
    class ActionPrimitiveError(Exception):
        Reason = SimpleNamespace(SAMPLING_ERROR="sampling")

        def __init__(self, reason, message, metadata):
            self.metadata = metadata

    class UnreadableFloor:
        category = "floor"

        @property
        def aabb(self):
            raise RuntimeError("opaque geometry failure")

    owner = SimpleNamespace(
        robot=SimpleNamespace(get_position_orientation=lambda: ((0, 0, 0), (0, 0, 0, 1))),
        env=SimpleNamespace(scene=SimpleNamespace(objects=(UnreadableFloor(),) if unreadable else ())),
        _placement_physical_consequences_enabled=physical,
    )
    with pytest.raises(ActionPrimitiveError) as caught:
        next(marker_symbolic_place_on_floor_generator(owner, action_primitive_error=ActionPrimitiveError))
    assert caught.value.metadata["unreadable_floor_geometry_count"] == int(unreadable)
    assert caught.value.metadata.get("action20_blocked_category") == (
        None if physical else "execution_unverified" if unreadable else "placement_support_unavailable"
    )


def test_symbolic_on_top_readback_does_not_require_physx_touching(monkeypatch):
    class Touching:
        pass

    class _FalseState:
        @staticmethod
        def get_value(_other):
            return False

    held = _Box((0.0, 0.0, 0.553), (0.1, 0.1, 0.1))
    support = _Box((0.0, 0.0, 0.25), (1.0, 1.0, 0.5))
    held.states[type("VerticalAdjacency", (), {})] = SimpleNamespace(
        get_value=lambda: SimpleNamespace(negative_neighbors={support}, positive_neighbors=set())
    )
    held.states[Touching] = _FalseState()
    held.link_prim_paths = ("/World/held/base_link",)
    support.link_prim_paths = ("/World/support/base_link",)
    monkeypatch.setitem(
        sys.modules, "omnigibson.utils.sampling_utils",
        SimpleNamespace(raytest_batch=lambda starts, ends, **_kwargs: [
            {"hit": True, "rigidBody": support.link_prim_paths[0],
             "position": (start[0], start[1], 0.5), "normal": (0, 0, 1)}
            for start in starts
        ]),
    )

    assert symbolic_on_top_relation(held, support) is True
    sample = sample_spatial_stability(
        held,
        support,
        relation="OnTop",
        support=support,
        symbolic_on_top=True,
    )
    assert sample.relation is True
    assert sample.support_on_top is True
    assert sample.support_touching is None

    held._center = (0.6, 0.0, 0.553)
    assert symbolic_on_top_relation(held, support) is False


def test_symbolic_on_top_readback_accepts_a_ray_verified_internal_surface(monkeypatch):
    class VerticalAdjacency:
        pass

    class Touching:
        pass

    class _AdjacencyState:
        @staticmethod
        def get_value():
            return SimpleNamespace(
                negative_neighbors={support},
                positive_neighbors=set(),
            )

    support = _Box((0.0, 0.0, 0.5), (1.0, 1.0, 1.0))
    held = _Box((0.0, 0.0, 0.35), (0.1, 0.1, 0.1))
    support.link_prim_paths = ("/World/support/base_link",)
    held.link_prim_paths = ("/World/held/base_link",)
    held.states[VerticalAdjacency] = _AdjacencyState()
    held.states[Touching] = SimpleNamespace(
        get_value=lambda _other: False
    )
    monkeypatch.setitem(
        sys.modules,
        "omnigibson.utils.sampling_utils",
        SimpleNamespace(
            raytest_batch=lambda starts, ends, **kwargs: [
                {
                    "hit": True,
                    "rigidBody": support.link_prim_paths[0],
                    "position": (start[0], start[1], 0.3),
                    "normal": (0.0, 0.0, 1.0),
                }
                for start in starts
            ]
        ),
    )

    assert symbolic_on_top_relation(
        held,
        support,
        support_surface_z_m=0.3,
    )
    assert symbolic_on_top_relation(held, support)
    sample = sample_spatial_stability(
        held,
        support,
        relation="OnTop",
        support=support,
        symbolic_on_top=True,
        support_surface_z_m=0.3,
    )
    assert sample.relation is True
    assert np.isclose(sample.bottom_to_support_gap_m, 0.0)


def test_floor_candidates_try_the_original_orientation_first():
    def binding(name, center, *, free=True, original=False):
        center = np.asarray(center, dtype=np.float64)
        return SimpleNamespace(
            candidate=SimpleNamespace(name=name),
            future_aabb_min_world=np.r_[center - 0.05, 0.0],
            future_aabb_max_world=np.r_[center + 0.05, 0.1],
            broad_footprint_free=free,
            orientation_quarter_turn=-1 if original else 0,
        )

    ordered = _ordered_floor_candidates(
        (
            binding("original_edge", (0.2, 0.0), original=True),
            binding("thin_center", (0.0, 0.0)),
            binding("original_center", (0.0, 0.0), original=True),
            binding("blocked_center", (0.0, 0.0), free=False),
        ),
        (0.0, 0.0),
    )

    assert [item.name for item in ordered] == [
        "original_center",
        "original_edge",
        "thin_center",
        "blocked_center",
    ]


def test_floor_placement_verifies_the_actual_local_support():
    floor = object()
    rug = object()

    assert _relation_target_role("place_on_floor") == "declared_support"
    assert _placement_relation_target("place_on_floor", floor, rug) is rug
    assert _relation_target_role("place_on_top") == "relation_target"
    assert _placement_relation_target("place_on_top", floor, rug) is floor


def test_floor_support_rays_use_the_authored_candidate_aabb():
    held = _Box((0.0, 0.0, 1.0), (0.2, 0.2, 1.0))
    candidate = SpatialPoseCandidate(
        name="floor_rotated",
        position=np.asarray((1.0, 2.0, 0.3)),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
    )
    bounds = (
        np.asarray((0.6, 1.8, 0.2)),
        np.asarray((1.4, 2.2, 0.4)),
    )
    geometry = _candidate_geometry_for_aabb(held, candidate, bounds)
    requests = build_downward_support_ray_requests(geometry, candidate)

    assert _candidate_geometry_for_aabb(held, candidate, None) is held
    assert {round(request.start_xyz[2], 3) for request in requests} == {0.25}
    assert np.isclose(min(request.start_xyz[0] for request in requests), 0.605)
    assert np.isclose(max(request.start_xyz[0] for request in requests), 1.395)

    report = rebase_candidate_to_local_support(
        geometry,
        candidate,
        tuple(
            DownwardSupportRayEvidence(
                sample_name=request.sample_name,
                start_xyz=request.start_xyz,
                end_xyz=request.end_xyz,
                hit_position_xyz=(*request.start_xyz[:2], 0.1),
                hit_normal_xyz=(0.0, 0.0, 1.0),
                hit_entity_key="rug",
                entity_role="support_candidate",
            )
            for request in requests
        ),
    )

    assert report.accepted is True
    assert np.isclose(report.original_bottom_z_m, 0.2)
    assert np.isclose(report.rebased_bottom_z_m, 0.103)
    assert np.isclose(report.rebased_candidate.position[2], 0.203)


def test_on_top_candidates_try_original_orientation_before_thin_axis_fallback():
    held = _Box((0.0, 0.0, 0.5), (0.1, 0.1, 0.4))
    support = _Box((0.0, 0.0, 0.0), (1.0, 1.0, 0.1))
    original = (0.0, 0.0, 0.0, 1.0)

    plan = plan_on_top_support_layer_candidates(
        held,
        support,
        preferred_orientation_xyzw=original,
    )

    assert plan.candidates[0].name.startswith("on_top_original_")
    assert any("on_top_thin_" in candidate.name for candidate in plan.candidates)
    assert all(
        binding.proposal_kind
        in {"original_orientation_plane", "original_destination_plane"}
        for binding in plan.bindings
    )
    raw_metadata = plan.candidate_plan.planning_metadata_private
    unreachable = restrict_and_rank_on_top_plan(plan, (10.0, 10.0), 0.1)
    filtered_metadata = unreachable.candidate_plan.planning_metadata_private
    assert filtered_metadata["plane_candidate_count"] == 0
    assert filtered_metadata["bindings_private"] == []
    for key in (
        "broad_free_plane_candidate_count",
        "broad_footprint_fit_plane_candidate_count",
        "broad_overhang_plane_candidate_count",
    ):
        assert filtered_metadata[key] == 0
        assert filtered_metadata[f"original_{key}"] == raw_metadata[key]


def test_on_top_candidates_label_a_matching_thin_pose_as_original():
    plate = _Box((0.0, 0.0, 0.5), (0.4, 0.3, 0.05))
    support = _Box((0.0, 0.0, 0.0), (1.0, 1.0, 0.1))

    plan = plan_on_top_support_layer_candidates(
        plate,
        support,
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    assert plan.bindings[0].orientation_quarter_turn == -1
    assert plan.candidates[0].name.startswith("on_top_original_")


def test_on_top_candidates_can_be_limited_to_a_local_floor_region():
    held = _Box((0.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    floor = _Box((0.0, 0.0, -0.05), (20.0, 20.0, 0.1))

    plan = plan_on_top_support_layer_candidates(
        held,
        floor,
        region_center_xy=(2.0, 3.0),
        region_extent_xy=(0.6, 0.6),
    )

    assert plan.candidates
    assert all(
        1.7 <= candidate.position[0] <= 2.3
        and 2.7 <= candidate.position[1] <= 3.3
        for candidate in plan.candidates
    )


def test_on_top_candidates_use_the_selected_surface_height():
    held = _Box((0.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    support = _Box((0.0, 0.0, 0.5), (1.0, 1.0, 1.0))

    # The six collision vertices leave the corners of the base-aligned box empty.
    points = np.concatenate((np.eye(3), -np.eye(3))) * 0.05
    tilted = (0.0, np.sin(np.pi / 8), 0.0, np.cos(np.pi / 8))
    plan = plan_on_top_support_layer_candidates(
        held,
        support,
        region_center_xy=(0.0, 0.0),
        region_extent_xy=(0.5, 0.5),
        surface_z_m=0.3,
        allow_overhang=False,
        preferred_orientation_xyzw=tilted,
        collision_points_in_root=points,
    )

    assert plan.candidates
    assert all(
        0.3 < binding.future_aabb_min_world[2] < 0.31
        for binding in plan.bindings
    )
    assert plan.candidate_plan.planning_metadata_private["surface_z_m"] == 0.3

    for binding in plan.bindings:
        pose = binding.candidate
        world = points @ quaternion_to_matrix_xyzw(pose.orientation_xyzw).T + pose.position
        assert world[:, 2].min() == pytest.approx(0.303)
        assert binding.future_aabb_min_world[2] == pytest.approx(world[:, 2].min())
        assert binding.future_aabb_max_world[2] == pytest.approx(world[:, 2].max())


def test_on_top_region_uses_full_support_and_can_opt_into_overhang():
    held = _Box((0.0, 0.0, 0.5), (0.4, 0.2, 0.1))
    support = _Box((0.0, 0.0, 0.0), (1.0, 1.0, 0.1))
    plan = plan_on_top_support_layer_candidates(
        held,
        support,
        preferred_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        region_center_xy=(0.35, 0.0),
        region_extent_xy=(0.3, 0.4),
        allow_overhang=False,
    )

    assert plan.bindings
    assert all(binding.broad_footprint_fit for binding in plan.bindings)
    assert all(
        0.2 <= binding.candidate.position[0] <= 0.5
        for binding in plan.bindings
    )
    assert all(
        np.all(binding.future_aabb_min_world[:2] >= (-0.5, -0.5))
        and np.all(binding.future_aabb_max_world[:2] <= (0.5, 0.5))
        for binding in plan.bindings
    )

    oversized = _Box((0.0, 0.0, 0.5), (1.2, 1.2, 1.2))
    blocked = plan_on_top_support_layer_candidates(
        oversized,
        support,
        region_center_xy=(0.35, 0.0),
        region_extent_xy=(0.3, 0.4),
        allow_overhang=False,
    )
    physical = plan_on_top_support_layer_candidates(
        oversized,
        support,
        region_center_xy=(0.35, 0.0),
        region_extent_xy=(0.3, 0.4),
        allow_overhang=True,
    )

    assert not blocked.candidates
    assert blocked.candidate_plan.planning_error == (
        "Selected region has no stable support candidate"
    )
    assert physical.candidates
    assert all(not binding.broad_footprint_fit for binding in physical.bindings)


def test_symbolic_center_candidate_recovers_grid_boundary_without_leaving_selection():
    held = _Box((0, 0, 0.5), (0.28, 0.20, 0.02))
    support = _Box((0, 0, 0), (0.32, 0.24, 0.02))
    kwargs = dict(preferred_orientation_xyzw=(0, 0, 0, 1),
                  region_center_xy=(0, 0.06), region_extent_xy=(0.32, 0.12))
    legacy = plan_on_top_support_layer_candidates(held, support, **kwargs)
    centered = plan_on_top_support_layer_candidates(
        held, support, **kwargs, include_support_center=True,
    )
    assert not any(np.allclose(c.position[:2], (0, 0)) for c in legacy.candidates)
    assert np.allclose(centered.candidates[0].position[:2], (0, 0))
    assert all(c.position[1] >= 0 for c in centered.candidates)

    side = plan_on_top_support_layer_candidates(
        held, support, region_center_xy=(0.12, 0.06), region_extent_xy=(0.08, 0.12),
        include_support_center=True,
    )
    assert side.candidates
    assert all(0.08 <= c.position[0] <= 0.16 for c in side.candidates)
    assert not any(c.name.endswith("_support_center") for c in side.candidates)


@pytest.mark.parametrize("supported", (False, True))
def test_symbolic_commit_requires_real_support_but_not_physx_contact(supported):
    sample = SpatialStabilitySample(
        relation=True, support_on_top=supported, support_touching=False,
        bottom_to_support_gap_m=0.003, linear_speed_m_s=0.5, angular_speed_rad_s=0.5,
        tilt_rad=0.0, position=(0, 0, 0.1), orientation_xyzw=(0, 0, 0, 1),
    )
    assert summarize_relation_authoritative_postcondition([sample], required_samples=1)["accepted"]
    report = summarize_relation_authoritative_postcondition(
        [sample], required_samples=1, require_geometric_support=True,
    )
    assert report["accepted"] is supported
    assert report["failure_reasons"] == ([] if supported else ["canonical_support_not_stable"])


@pytest.mark.parametrize("corners", ("miss", "lower_surface"))
def test_symbolic_support_rebase_keeps_center_height_when_corners_overhang(corners):
    held = _Box((0, 0, 0.2), (0.3, 0.3, 0.1))
    candidate = SpatialPoseCandidate("next_to_or_floor", np.array((0, 0, 0.2)), np.array((0, 0, 0, 1)))
    rays = build_downward_support_ray_requests(held, candidate)
    evidence = tuple(DownwardSupportRayEvidence(
        sample_name=ray.sample_name, start_xyz=ray.start_xyz, end_xyz=ray.end_xyz,
        hit_position_xyz=(*ray.start_xyz[:2], 0.1 if index == 0 else 0.0)
                         if index == 0 or corners == "lower_surface" else None,
        hit_normal_xyz=(0, 0, 1) if index == 0 or corners == "lower_surface" else None,
        hit_entity_key="small_support" if index == 0 else "floor",
        entity_role="support_candidate",
    ) for index, ray in enumerate(rays))
    report = rebase_candidate_to_local_support(held, candidate, evidence, allow_overhang=True)
    assert report.accepted
    assert report.median_hit_z_m == 0.1
    assert np.isclose(report.rebased_bottom_z_m, 0.103)
    legacy = rebase_candidate_to_local_support(held, candidate, evidence)
    assert not legacy.accepted or legacy.median_hit_z_m != 0.1
    missing_center = (replace(evidence[0], hit_position_xyz=None, hit_normal_xyz=None), *evidence[1:])
    assert not rebase_candidate_to_local_support(held, candidate, missing_center, allow_overhang=True).accepted
    fault = (replace(evidence[0], error="ray query failed"), *evidence[1:])
    assert not rebase_candidate_to_local_support(held, candidate, fault, allow_overhang=True).accepted


@pytest.mark.parametrize("clearance", (0.5, 0.05))
def test_under_allows_small_lower_support_but_preserves_upper_clearance(clearance):
    held = _Box((0, 0, 1.0), (0.3, 0.3, 0.1))
    support = _Box((0, 0, -0.05), (0.12, 0.12, 0.1))
    target = _Box((0, 0, clearance + 0.05), (1.2, 1.2, 0.1))
    for obj, name in ((held, "held"), (support, "support"), (target, "target")):
        obj.name = name
        obj.link_prim_paths = (f"/World/{name}",)

    def raytest_batch(starts, ends, **_kwargs):
        upward = starts[0][2] < ends[0][2]
        return [
            {"hit": True, "rigidBody": target.link_prim_paths[0] if upward else support.link_prim_paths[0],
             "position": (start[0], start[1], clearance if upward else 0.0), "normal": (0, 0, -1 if upward else 1)}
            if upward or (abs(start[0]) <= 0.06 and abs(start[1]) <= 0.06) else {"hit": False}
            for start in starts
        ]

    kwargs = dict(held=held, target=target, supports=(support,), robot=object(), raytest_batch=raytest_batch)
    physical, = plan_live_under_supports(**kwargs)
    symbolic, = plan_live_under_supports(**kwargs, symbolic_support=True)
    assert not physical.candidate_plan.candidates
    assert bool(symbolic.candidate_plan.candidates) is (clearance == 0.5)


def test_on_top_plan_uses_pose_priority_then_stable_footprint_distance():
    def binding(name, lower_x, upper_x, *, original=False):
        candidate = SpatialPoseCandidate(
            name=name,
            position=np.zeros(3),
            orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        )
        return OnTopCandidateBinding(
            candidate=candidate,
            proposal_kind="original_destination_plane",
            broad_footprint_fit=True,
            broad_footprint_free=True,
            thin_axis=2,
            orientation_quarter_turn=-1 if original else 0,
            future_aabb_min_world=np.asarray((lower_x, -0.1, 0.0)),
            future_aabb_max_world=np.asarray((upper_x, 0.1, 0.2)),
        )

    bindings = (
        binding("equal_left", -1.1, -1.0, original=True),
        binding("near", 0.5, 0.6),
        binding("outside", 2.0, 2.1),
        binding("equal_right", 1.0, 1.1),
        binding("boundary", 1.7, 1.8),
    )
    reports = tuple(
        CandidateAdmissionReport(
            candidate=item.candidate.name,
            accepted=True,
            reason=f"original:{item.candidate.name}",
            blockers=("diagnostic",),
            bottom_to_support_gap_m=0.01,
        )
        for item in bindings
    )
    plan = OnTopSupportLayerPlan(
        candidate_plan=SpatialCandidatePlan(
            candidates=tuple(item.candidate for item in bindings),
            admission_reports=reports,
            planning_metadata_private={"sentinel": "kept"},
        ),
        bindings=bindings,
    )

    assert on_top_binding_horizontal_distance(bindings[1], (0.0, 0.0)) == 0.5
    assert on_top_binding_horizontal_distance(
        bindings[4], (0.0, 0.0)
    ) == pytest.approx(1.7)
    ranked = restrict_and_rank_on_top_plan(plan, (0.0, 0.0), 1.7)

    assert tuple(item.candidate.name for item in ranked.bindings) == (
        "equal_left",
        "near",
        "equal_right",
        "boundary",
    )
    assert ranked.candidates == tuple(item.candidate for item in ranked.bindings)
    assert len(ranked.candidate_plan.admission_reports) == len(reports)
    assert ranked.candidate_plan.admission_reports[0] is reports[0]
    outside = ranked.candidate_plan.admission_reports[2]
    assert outside.accepted is False
    assert outside.reason == "future_footprint_outside_local_workspace"
    assert outside.blockers == ("diagnostic",)
    assert outside.bottom_to_support_gap_m == 0.01
    metadata = ranked.candidate_plan.planning_metadata_private
    assert metadata["sentinel"] == "kept"
    assert metadata["local_workspace_base_xy_private"] == [0.0, 0.0]
    assert metadata["maximum_local_reach_m"] == 1.7
    assert metadata["reachable_candidate_count"] == 4
    assert [
        item["candidate"]
        for item in metadata["candidate_horizontal_distances_m_private"]
    ] == [item.candidate.name for item in bindings]

    unreachable = restrict_and_rank_on_top_plan(plan, (0.0, 0.0), 0.1)
    assert unreachable.bindings == ()
    assert unreachable.candidates == ()
    assert _empty_on_top_plan_category(unreachable.candidate_plan) == "out_of_reach"
    assert _empty_on_top_plan_category(SpatialCandidatePlan((), ())) == "geometric_fit_blocked"
    assert unreachable.candidate_plan.planning_error
    assert len(unreachable.candidate_plan.admission_reports) == len(reports)
    assert all(
        report.reason == "future_footprint_outside_local_workspace"
        for report in unreachable.candidate_plan.admission_reports
    )


def test_live_spatial_execution_is_owned_by_the_focused_module():
    bridge = (
        __import__("behavior_hl.project_identity", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT
        / "src/behavior_hl/embodied/evaluator_bridge"
    )
    execution = (bridge / "marker_symbolic_spatial_execution.py").read_text()
    runtime = (bridge / "run_b1_marker_symbolic_rollout.py").read_text()
    wrapper = runtime.split(
        "        def marker_symbolic_spatial_generator(", 1
    )[1].split("        def marker_symbolic_place_on_floor_generator", 1)[0]

    assert "def marker_symbolic_spatial_generator(" in execution
    assert "execute_admitted_spatial_candidate(" in execution
    assert "or owner._placement_physical_consequences_enabled" in execution
    assert "on_top_layer_plan" not in execution
    assert "intended_contact_support" not in execution
    assert "def contact_targets(support):" in execution
    assert execution.count(
        "contact_objects=contact_targets("
    ) == 2
    assert "geometry_certified_objects=(target, selected_support)" not in execution
    assert "_execute_spatial_placement(" in wrapper
    assert "execute_admitted_spatial_candidate(" not in wrapper


def test_generic_inside_uses_zero_physics_without_physical_consequences():
    bridge = (
        __import__("behavior_hl.project_identity", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT
        / "src/behavior_hl/embodied/evaluator_bridge"
    )
    spatial = (bridge / "marker_symbolic_spatial_execution.py").read_text()
    inside = (bridge / "marker_symbolic_inside_execution.py").read_text()
    physical = (bridge / "physical_inside_execution.py").read_text()
    runtime = (bridge / "run_b1_marker_symbolic_rollout.py").read_text()

    physics_gate = runtime.split("    physical_placement = bool(", 1)[1].split(
        "    return (", 1
    )[0]
    assert '"placement_physical_consequences_enabled"' in physics_gate
    assert '"place_inside"' in physics_gate
    assert "and not physical_placement" in physics_gate

    assert DEFAULT_SETTLE_STEPS > DEFAULT_STABILITY_WINDOW
    assert (
        "DEFAULT_SETTLE_STEPS - DEFAULT_STABILITY_WINDOW" in spatial
    )
    selected = inside.split(
        "def _marker_symbolic_selected_inside_band_generator(", 1
    )[1].split("def marker_symbolic_inside_compartment_generator(", 1)[0]
    generic = inside.split(
        "def _try_high_level_symbolic_inside_candidates(", 1
    )[1].split("def place_inside_with_predicate(", 1)[0]
    placement = inside.split("def place_inside_with_predicate(", 1)[1]
    symbolic_readback = inside.split(
        "def _symbolic_inside_readback(", 1
    )[1].split("def _inside_release_readback(", 1)[0]
    release_dispatch = inside.split(
        "def _inside_release_readback(", 1
    )[1].split(
        "def _marker_symbolic_selected_inside_band_generator(", 1
    )[0]

    assert "og.sim.step_physics()" not in generic
    assert "collision_shape_clear_entry_path" in physical
    assert '"collision_shape_overlap_is_rejection_gate": True' in generic
    assert "_final_inside_collision_admission(" in selected
    assert "_final_inside_collision_admission(" in generic
    assert "plan_symbolic_inside_candidates(" in selected
    assert "existing_content_aabbs=existing" in selected
    assert "fitting_inside_orientation_matrices" not in selected
    assert "DRAWER_CONTENT_TOLERANCE_M" not in selected
    assert '"candidate_plan_private": dict(plan.private_record())' in selected
    assert "existing_content_aabbs=existing" in generic
    assert '"container_motion_thresholds_required": False' in generic
    assert "DEFAULT_MAX_BLOCKED_FACE_PENETRATION_RATIO" in generic
    assert (
        "maximum_blocked_face_penetration_ratio=penetration_limit" in generic
    )
    assert (
        "include_opening_face_penetration=opening_face_is_bounded"
        in generic
    )
    assert '"blocked_face_penetration_is_rejection_gate": True' in generic
    assert "closeable_container or len(fillable_regions) > 1" in generic
    assert '"opening_face_penetration_is_rejection_gate": (' in generic
    assert "maximum_content_overlap_ratio=content_overlap_limit" in generic
    assert '"content_overlap_is_rejection_gate": True' in generic
    assert placement.count("_semantic_inside_ledger.commit(") == 2
    assert '"official_inside_readback_is_rejection_gate": True' in placement
    assert "yield from _inside_release_readback(" in placement
    assert "physical = _physical_inside_execution(owner)" in placement
    assert "probably dropped" not in placement
    assert "og.sim.step_physics()" not in selected
    assert "yield from _inside_release_readback(" in selected
    assert "disable_gravity" not in selected
    assert "set_collision" not in selected
    assert "orientation_preserved" not in selected
    assert "DEFAULT_PROTECTED_ROTATION_RAD" not in selected
    assert "yield from owner._settle_robot()" not in inside
    assert "yield from physical.release_readback(" in release_dispatch
    assert "yield None" not in release_dispatch
    assert '"physics_step_count": 0' in symbolic_readback
    assert '"stability_sample_steps": 0' in symbolic_readback
    assert "yield" not in symbolic_readback
    assert "for _ in range(INSIDE_PHYSICAL_RELEASE_STEPS):" in physical
    assert physical.count("yield None") == 1
    assert "_placement_physical_consequences_enabled" not in physical
    assert inside.count("_placement_physical_consequences_enabled") == 1
    assert "_object_velocity" not in inside
    assert "linear_velocity_not_stable" not in inside
    assert "angular_velocity_not_stable" not in inside
    assert "_refresh_inside_readback_without_physics(" in inside
    assert '"free_settle_steps": 0' in inside
    assert '"stability_sample_steps"' in spatial
    assert '"stability_sample_steps"' in inside
    assert "owner._skip_next_symbolic_settle = True" in inside
    settle = runtime.split("        def _settle_robot(self):", 1)[1].split(
        "        def _grasp(self, obj):", 1
    )[0]
    assert "if action20_enabled or self._skip_next_symbolic_settle:" in settle
    assert settle.index(
        "if action20_enabled or self._skip_next_symbolic_settle:"
    ) < settle.index("for _ in range(8):")


def test_spatial_placement_defaults_to_one_symbolic_readback():
    execution = (
        __import__("behavior_hl.project_identity", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT
        / "src/behavior_hl/embodied/evaluator_bridge"
        / "marker_symbolic_spatial_execution.py"
    ).read_text()

    acceptance = execution.split("    def candidate_is_accepted(", 1)[1].split(
        "\n    try:\n        yield from owner._detach_held_object_at_current_pose()",
        1,
    )[0]
    candidate_loop = execution.split(
        "        candidate_count = len(candidates)", 1
    )[1].split("        if not placement_committed:", 1)[0]
    assert '"place_touching"' not in execution.split(
        "    symbolic_direct_readback = bool(", 1
    )[1].split("    )", 1)[0]
    assert "sample_count = 1 if symbolic_direct_readback" in acceptance
    assert "_refresh_inside_readback_without_physics(" in acceptance
    assert "query_symbolic_placement_final_collision(" in acceptance
    assert "summarize_relation_authoritative_postcondition(" in acceptance
    assert '"settle_steps_total": (' in acceptance
    assert '"candidate_orientation_is_rejection_gate": (' in acceptance
    assert candidate_loop.count("yield from candidate_is_accepted(") == 1
    assert "applied_candidate_attempt_count += 1" in candidate_loop
    assert "physics_candidate_attempt_count += 1" in candidate_loop
    assert 'operation != "place_next_to"' in candidate_loop
    assert "applied_candidate_attempt_count >= 2" in candidate_loop


def test_next_to_revalidates_selector_without_float_identity_gate():
    execution = (
        __import__("behavior_hl.project_identity", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT
        / "src/behavior_hl/embodied/evaluator_bridge"
        / "marker_symbolic_spatial_execution.py"
    ).read_text()
    next_to = execution.split(
        'if operation == "place_next_to" and placement_region_center_xy is not None:',
        2,
    )[2].split("            else:", 1)[0]

    assert "physical_next_to_bindings()" in next_to
    assert "Selected NextTo support point is unavailable" in next_to
    assert "Selected NextTo support point is stale" not in next_to
    assert "np.allclose(" not in next_to


def test_committed_inside_survives_transient_readback_until_explicit_pick():
    ledger = SemanticInsideLedger()
    ledger.commit(object_name="rose", container_name="vase")

    assert ledger.contains(object_name="rose", container_name="vase")

    root = SimpleNamespace(name="vase")
    child = SimpleNamespace(name="rose")
    committed = set(ledger.relations())
    snapshot = LogicalInventorySnapshot.capture(
        root=root,
        candidates=(root, child),
        get_pose=lambda _obj: 0,
        is_inside=lambda candidate, parent: (
            (candidate.name, parent.name) in committed
        ),
        copy_pose=lambda value: value,
        inverse_pose=lambda value: value,
        compose_pose=lambda _left, right: right,
        valid_pose=lambda _value: True,
    )
    assert tuple(entry.object_ref for entry in snapshot.entries) == (child,)

    ledger.discard(object_name="rose")
    assert not ledger.contains(object_name="rose", container_name="vase")

    runtime = (
        __import__("behavior_hl.project_identity", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT
        / "src/behavior_hl/embodied/evaluator_bridge"
        / "run_b1_marker_symbolic_rollout.py"
    ).read_text()
    capture = runtime.split(
        "        def _capture_logical_inventory(", 1
    )[1].split("        def _apply_logical_inventory_root_pose(", 1)[0]
    assert "_semantic_inside_ledger.relations()" in capture
    assert "or child.states[object_states.Inside].get_value(parent)" in capture


def test_logical_inventory_refreshes_official_inside_before_capture():
    class Candidate:
        def __init__(self, *, live_inside=False, cached_inside=False):
            self.live_inside = live_inside
            self.cached_inside = cached_inside

            self.clear_count = 0

        def clear_states_cache(self):
            self.clear_count += 1
            self.cached_inside = self.live_inside

    root = Candidate()
    stale = Candidate(
        live_inside=False,
        cached_inside=True,
    )
    nested = Candidate(
        live_inside=True,
        cached_inside=True,
    )
    candidates = marker_rollout._refresh_logical_inventory_inside_candidates(
        (root, stale, nested, nested)
    )

    snapshot = LogicalInventorySnapshot.capture(
        root=root,
        candidates=candidates,
        get_pose=lambda _candidate: 0,
        is_inside=lambda child, parent: bool(
            parent is root and child.cached_inside
        ),
        copy_pose=lambda value: value,
        inverse_pose=lambda value: value,
        compose_pose=lambda _left, right: right,
        valid_pose=lambda _value: True,
    )

    assert tuple(entry.object_ref for entry in snapshot.entries) == (nested,)
    assert (root.clear_count, stale.clear_count, nested.clear_count) == (1, 1, 1)


def test_on_top_containment_refresh_rejects_a_spilled_nested_object():
    inside_state_type = object()

    class State:
        def __init__(self):
            self.live = False
            self.cached = True

        def get_value(self, _parent):
            return self.cached

    class Object:
        def __init__(self, state=None):
            self.states = {} if state is None else {inside_state_type: state}
            self.clear_count = 0

        def clear_states_cache(self):
            self.clear_count += 1
            for state in self.states.values():
                state.cached = state.live

    state = State()
    container = Object()
    child = Object(state)
    snapshot = SimpleNamespace(
        object_refs=(container, child),
        containment_edges=((child, container),),
    )

    preserved, cleared = _fresh_official_inside_containment(
        snapshot,
        inside_state_type,
    )
    assert preserved is False
    assert cleared == 2

    state.live = True
    preserved, cleared = _fresh_official_inside_containment(
        snapshot,
        inside_state_type,
    )
    assert preserved is True
    assert cleared == 2
    assert (container.clear_count, child.clear_count) == (2, 2)


def test_on_top_nested_closure_drops_carry_velocity_and_requires_stability():
    class Object:
        def __init__(self, name, linear=(0.0, 0.0, 0.0)):
            self.name = name
            self.linear = linear
            self.keep_still_count = 0

        def keep_still(self):
            self.keep_still_count += 1
            self.linear = (0.0, 0.0, 0.0)

        def get_linear_velocity(self):
            return self.linear

        def get_angular_velocity(self):
            return (0.0, 0.0, 0.0)

    root = Object("box")
    child = Object("chicken", linear=(0.5, 0.0, 0.0))
    snapshot = SimpleNamespace(
        object_refs=(root, child),
        entries=(SimpleNamespace(object_ref=child),),
    )

    vector_reader = lambda value, **_kwargs: np.asarray(value, dtype=float)
    stable, reports = _logical_inventory_descendants_stable(
        snapshot,
        vector_reader=vector_reader,
    )
    assert stable is False
    assert reports[0]["stable"] is False
    assert _keep_logical_inventory_closure_still(snapshot) == 2
    assert (root.keep_still_count, child.keep_still_count) == (1, 1)
    assert _logical_inventory_descendants_stable(
        snapshot,
        vector_reader=vector_reader,
    )[0] is True
