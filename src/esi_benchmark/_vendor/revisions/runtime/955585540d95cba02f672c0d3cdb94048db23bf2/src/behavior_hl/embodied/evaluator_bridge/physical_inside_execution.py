"""Physical-release mechanics for experimental Inside placement."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.deterministic_inside_placement import (
    Aabb3,
)
from behavior_hl.embodied.evaluator_bridge.on_top_support_layers import (
    quaternion_to_matrix_xyzw,
)
from behavior_hl.embodied.evaluator_bridge.spatial_candidate_admission import (
    query_spatial_candidate_collision_overlap,
)
from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    SpatialPoseCandidate,
    object_pose,
)


INSIDE_PHYSICAL_ENTRY_CLEARANCE_M = 0.01
INSIDE_PHYSICAL_ENTRY_MAX_SAMPLES = 96
INSIDE_PHYSICAL_ENTRY_QUERY_BUDGET = 384
INSIDE_PHYSICAL_RELEASE_STEPS = 8


def inside_entry_direction_candidates(
    *,
    source_position: Any,
    target_position: Any,
    opening_orientation_xyzw: Any,
) -> tuple[np.ndarray, ...]:
    """Return one approach direction followed by the opening frame axes."""

    source = np.asarray(source_position, dtype=np.float64).reshape(-1)
    target = np.asarray(target_position, dtype=np.float64).reshape(-1)
    if source.shape != (3,) or target.shape != (3,):
        raise ValueError("Inside entry positions must be 3-D")
    if not np.isfinite(np.concatenate((source, target))).all():
        raise ValueError("Inside entry positions must be finite")
    rotation = quaternion_to_matrix_xyzw(opening_orientation_xyzw)
    preferred = source - target
    preferred_norm = float(np.linalg.norm(preferred))
    preferred = (
        preferred / preferred_norm
        if preferred_norm > 1.0e-9
        else rotation[:, 2]
    )
    axes = [
        sign * rotation[:, axis]
        for axis in range(3)
        for sign in (1.0, -1.0)
    ]
    axes.sort(
        key=lambda direction: (
            -float(np.dot(direction, preferred)),
            tuple(float(value) for value in direction),
        )
    )
    directions: list[np.ndarray] = []
    for direction in (preferred, *axes):
        unit = direction / float(np.linalg.norm(direction))
        if any(np.allclose(unit, item, atol=1.0e-7) for item in directions):
            continue
        directions.append(unit.copy())
    return tuple(directions)


def _inside_entry_path(
    desired: SpatialPoseCandidate,
    *,
    candidate_aabb: Aabb3,
    fillable_aabb: Aabb3,
    outward_direction: Any,
) -> tuple[tuple[SpatialPoseCandidate, Aabb3, bool], ...]:
    """Sample one straight, fixed-orientation path from outside to inside."""

    direction = np.asarray(outward_direction, dtype=np.float64).reshape(-1)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("Inside entry direction must be a finite 3-D vector")
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        raise ValueError("Inside entry direction must be non-zero")
    direction /= norm
    fill_center = np.asarray(fillable_aabb.center, dtype=np.float64)
    fill_half = np.asarray(fillable_aabb.extent, dtype=np.float64) / 2.0
    candidate_center = np.asarray(candidate_aabb.center, dtype=np.float64)
    candidate_half = np.asarray(candidate_aabb.extent, dtype=np.float64) / 2.0
    fillable_outer = float(
        np.dot(fill_center, direction)
        + np.dot(fill_half, np.abs(direction))
    )
    candidate_inner = float(
        np.dot(candidate_center, direction)
        - np.dot(candidate_half, np.abs(direction))
    )
    distance = max(
        INSIDE_PHYSICAL_ENTRY_CLEARANCE_M,
        fillable_outer
        + INSIDE_PHYSICAL_ENTRY_CLEARANCE_M
        - candidate_inner,
    )
    step_m = max(min(candidate_aabb.extent) * 0.5, 1.0e-6)
    sample_count = max(1, int(np.ceil(distance / step_m)))
    if sample_count > INSIDE_PHYSICAL_ENTRY_MAX_SAMPLES:
        raise ValueError("Inside entry path exceeds its bounded sample count")

    desired_position = np.asarray(desired.position, dtype=np.float64)
    stage_position = desired_position + direction * distance
    fill_lower = np.asarray(fillable_aabb.lower, dtype=np.float64)
    fill_upper = np.asarray(fillable_aabb.upper, dtype=np.float64)
    candidate_lower = np.asarray(candidate_aabb.lower, dtype=np.float64)
    candidate_upper = np.asarray(candidate_aabb.upper, dtype=np.float64)
    samples = []
    for index in range(sample_count + 1):
        fraction = index / sample_count
        position = stage_position + fraction * (desired_position - stage_position)
        delta = position - desired_position
        aabb = Aabb3(
            tuple((candidate_lower + delta).tolist()),
            tuple((candidate_upper + delta).tolist()),
        )
        fully_inside = bool(
            np.all(np.asarray(aabb.lower) >= fill_lower - 1.0e-6)
            and np.all(np.asarray(aabb.upper) <= fill_upper + 1.0e-6)
        )
        samples.append((
            SpatialPoseCandidate(
                name=f"{desired.name}:entry_{index}_of_{sample_count}",
                position=position,
                orientation_xyzw=np.asarray(
                    desired.orientation_xyzw,
                    dtype=np.float64,
                ).copy(),
            ),
            aabb,
            fully_inside,
        ))
    return tuple(samples)


def physical_inside_consequence_commit_safe(
    report: Mapping[str, Any],
) -> bool:
    """Return whether one complete physical trial has finite readback."""

    return bool(
        report.get("physics_step_count") == INSIDE_PHYSICAL_RELEASE_STEPS
        and report.get("readback_samples") == 1
        and len(report.get("pose_finite_window", ())) == 1
        and all(report["pose_finite_window"])
    )


class PhysicalInsideExecution:
    """Own the dormant physical-release policy for one Inside action."""

    __slots__ = ("owner", "_remaining_collision_queries")

    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self._remaining_collision_queries = INSIDE_PHYSICAL_ENTRY_QUERY_BUDGET

    @staticmethod
    def plan_options() -> tuple[float, float, bool]:
        """Return the legacy physical candidate policy unchanged."""

        permissive = np.nextafter(1.0, 0.0)
        return permissive, permissive, False

    def begin(self):
        """Begin the legacy physical transaction without changing behavior."""

        return (yield from self.owner._detach_held_object_at_current_pose())

    def admit_candidate(
        self,
        held: Any,
        container: Any,
        desired: SpatialPoseCandidate,
        *,
        candidate_aabb: Aabb3,
        fillable_aabb: Aabb3,
        opening_frame: Any | None = None,
        point_in_fillable_volume: Callable[[tuple[float, float, float]], bool]
        | None = None,
    ) -> tuple[
        SpatialPoseCandidate | None,
        tuple[Mapping[str, Any], ...],
        bool,
    ]:
        """Admit a complete collision-free entry path to one release pose."""

        import omnigibson as og

        snapshot = self.owner._logical_inventory_snapshot
        closure = snapshot.object_refs if snapshot is not None else (held,)
        source_position, _ = object_pose(held)
        _, opening_orientation = object_pose(
            container if opening_frame is None else opening_frame
        )
        directions = inside_entry_direction_candidates(
            source_position=source_position,
            target_position=desired.position,
            opening_orientation_xyzw=opening_orientation,
        )
        reports: list[Mapping[str, Any]] = []
        for path_index, direction in enumerate(directions):
            try:
                samples = _inside_entry_path(
                    desired,
                    candidate_aabb=candidate_aabb,
                    fillable_aabb=fillable_aabb,
                    outward_direction=direction,
                )
            except ValueError as exc:
                reports.append({
                    "accepted": False,
                    "reason": "entry_path_geometry_rejected",
                    "entry_path_index_private": path_index,
                    "entry_direction_xyz_private": direction.tolist(),
                    "query_error": f"{type(exc).__name__}: {exc}",
                })
                continue

            fill_lower = np.asarray(fillable_aabb.lower, dtype=np.float64)
            fill_upper = np.asarray(fillable_aabb.upper, dtype=np.float64)
            release_index = None
            try:
                for index, (_candidate, sample_aabb, fully_inside) in enumerate(samples):
                    center = np.asarray(sample_aabb.center, dtype=np.float64)
                    center_inside = bool(
                        point_in_fillable_volume(tuple(center.tolist()))
                        if point_in_fillable_volume is not None
                        else np.all(center >= fill_lower - 1.0e-6)
                        and np.all(center <= fill_upper + 1.0e-6)
                    )
                    if center_inside:
                        if release_index is None:
                            release_index = index
                        # Keep supporting a fitting object until it clears the
                        # entrance, including for sideways/slanted insertion.
                        # Oversized candidates retain the partial-entry trial;
                        # the same collision gate still has to admit it.
                        if fully_inside:
                            release_index = index
                            break
            except Exception as exc:
                reports.append({
                    "accepted": False,
                    "reason": "entry_path_fillable_probe_failed",
                    "entry_path_index_private": path_index,
                    "query_error": f"{type(exc).__name__}: {exc}",
                })
                continue
            if release_index is None:
                reports.append({
                    "accepted": False,
                    "reason": "entry_path_never_enters_fillable_volume",
                    "entry_path_index_private": path_index,
                })
                continue

            query_order = tuple(dict.fromkeys((
                release_index,
                0,
                *range(release_index + 1),
            )))
            path_clear = True
            for sample_index in query_order:
                if self._remaining_collision_queries <= 0:
                    reports.append({
                        "accepted": False,
                        "reason": "entry_path_query_budget_exhausted",
                        "entry_path_index_private": path_index,
                        "entry_sample_index_private": sample_index,
                    })
                    return None, tuple(reports), False
                self._remaining_collision_queries -= 1
                sample, sample_aabb, fully_inside = samples[sample_index]
                report = query_spatial_candidate_collision_overlap(
                    omnigibson=og,
                    held=held,
                    closure_objects=closure,
                    geometry_certified_objects=(container,) if fully_inside else (),
                    candidate=sample,
                    candidate_aabb_world=(sample_aabb.lower, sample_aabb.upper),
                    xy_margin_m=0.0,
                    xy_inset_m=0.0,
                    z_inset_m=0.0,
                )
                reports.append({
                    "phase": "physical_inside_entry_path",
                    "entry_path_index_private": path_index,
                    "entry_sample_index_private": sample_index,
                    "entry_sample_count_private": len(samples),
                    "entry_release_index_private": release_index,
                    "entry_release_fully_inside_private": samples[release_index][2],
                    "entry_direction_xyz_private": direction.tolist(),
                    "container_geometry_certified_private": fully_inside,
                    **report.to_private_dict(),
                })
                if not report.accepted:
                    path_clear = False
                    break
            if path_clear:
                return samples[release_index][0], tuple(reports), True
        return None, tuple(reports), False

    @staticmethod
    def pre_release_ready(*, pose_finite: bool, nested_preserved: bool) -> bool:
        return bool(pose_finite and nested_preserved)

    @staticmethod
    def candidate_report_policy() -> Mapping[str, Any]:
        return {
            "semantic_admission_private": "collision_shape_clear_entry_path",
            "official_inside_readback_required": False,
            "blocked_face_penetration_is_rejection_gate": False,
            "final_state_collision_readback_required": False,
            "content_overlap_is_rejection_gate": False,
        }

    def release_readback(
        self,
        held: Any,
        container: Any,
        inside_state_type: Any,
        logical_snapshot: Any,
        *,
        cpu_vector: Callable[..., np.ndarray],
        refresh_without_physics: Callable[..., Mapping[str, Any]],
    ):
        """Advance one bounded natural release and read its resulting state."""

        active = self.owner._logical_inventory_snapshot or logical_snapshot
        # The physical release owns the real world from here, including errors.
        self.owner._clear_logical_inventory_state()
        from .physical_template_runtime import ACTIVE
        audit=ACTIVE.get()
        if audit is not None:audit.on_release(held,container)
        for carried in (held,) if active is None else active.object_refs:
            keep_still = getattr(carried, "keep_still", None)
            if callable(keep_still):
                keep_still()
        try:
            for _ in range(INSIDE_PHYSICAL_RELEASE_STEPS):
                yield None
        finally:
            if audit is not None:audit.end_release()
        protected = [held, container]
        if active is not None:
            protected.extend(active.object_refs)
        refresh = refresh_without_physics(self.owner, *protected)
        position, orientation = held.get_position_orientation()
        nested_preserved, _ = self.owner._logical_inventory_relations_preserved(
            active
        )
        official_inside = bool(
            held.states[inside_state_type].get_value(container)
        )
        pose_finite = bool(
            np.isfinite(
                cpu_vector(position, length=3, name="Inside position")
            ).all()
            and np.isfinite(
                cpu_vector(
                    orientation,
                    length=4,
                    name="Inside orientation",
                )
            ).all()
        )
        failures = []
        if not official_inside:
            failures.append("official_inside_not_satisfied")
        if not nested_preserved:
            failures.append("nested_relations_not_preserved")
        if not pose_finite:
            failures.append("non_finite_pose")
        return {
            "accepted": not failures,
            "readback_samples": 1,
            "physics_step_count": INSIDE_PHYSICAL_RELEASE_STEPS,
            "settle_steps_total": INSIDE_PHYSICAL_RELEASE_STEPS,
            "stability_sample_steps": 0,
            "official_inside_window": [official_inside],
            "nested_relations_window": [bool(nested_preserved)],
            "pose_finite_window": [pose_finite],
            "linear_speed_m_s": [],
            "angular_speed_rad_s": [],
            "failure_reasons": failures,
            "refresh_private": dict(refresh),
        }

    @staticmethod
    def consequence_commit_safe(report: Mapping[str, Any]) -> bool:
        return physical_inside_consequence_commit_safe(report)
