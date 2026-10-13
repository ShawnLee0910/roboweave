"""Geometry-only planning and recoverable execution for ``Inside`` actions.

The simulator-facing worker remains responsible for reading live AABBs,
applying poses, stepping physics, and reading the official ``Inside`` state.
This module deliberately has no OmniGibson or Torch imports.  It supplies the
three pieces that need to be deterministic and task-independent:

* rank feasible grounded goal options by live container load;
* generate collision-free initial pose candidates from a fillable AABB and
  its current contents; and
* retry candidates without losing the held object after a failed attempt.

Object and container names are used only as stable identity tie-breakers.
Task names, semantic categories, and instance-specific assignment rules are
not accepted by any public function in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import math
from typing import Any, Callable, Iterable, Mapping, Sequence


DETERMINISTIC_INSIDE_PLACEMENT_VERSION = (
    "behavior2026.deterministic-inside-placement/v1"
)
SYMBOLIC_INSIDE_PLACEMENT_VERSION = (
    "behavior2026.high-level-symbolic-inside/v8-preferred-orientation"
)
INSIDE_CANDIDATE_GOAL_PRESERVATION_VERSION = (
    "behavior2026.inside-action-goal-preservation/v2"
)
DEFAULT_SYMBOLIC_INSIDE_WINDOW = 3

_EPSILON = 1e-12
DEFAULT_MAX_BLOCKED_FACE_PENETRATION_RATIO = 0.0
DEFAULT_MAX_CONTENT_OVERLAP_RATIO = 0.0
DEFAULT_MAX_CLOSEABLE_FACE_PENETRATION_RATIO = 0.0

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

_IDENTITY_MAT3: Mat3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class InsidePlacementGeometryError(ValueError):
    """Raised when required live geometry is missing or invalid."""


class InsidePlacementRecoveryError(RuntimeError):
    """Raised when a failed candidate cannot prove that hold was restored."""


def summarize_symbolic_inside_window(
    inside: Sequence[bool],
    contained: Sequence[bool],
    nested: Sequence[bool],
    finite: Sequence[bool],
    *,
    required_samples: int = DEFAULT_SYMBOLIC_INSIDE_WINDOW,
) -> dict[str, Any]:
    """Require a short consecutive Symbolic Inside postcondition window."""

    if required_samples <= 0:
        raise ValueError("required Inside samples must be positive")
    windows = tuple(tuple(values[-required_samples:]) for values in (
        inside,
        contained,
        nested,
        finite,
    ))
    complete = all(len(values) == required_samples for values in windows)
    accepted = complete and all(all(values) for values in windows)
    return {
        "accepted": accepted,
        "required_samples": required_samples,
        "observed_samples": min((len(values) for values in windows), default=0),
        "official_inside_window": list(windows[0]),
        "complete_containment_window": list(windows[1]),
        "nested_inside_window": list(windows[2]),
        "finite_pose_window": list(windows[3]),
    }


def _determinant3(matrix: Mat3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _axis_aligned_rotations() -> tuple[Mat3, ...]:
    """Return the 24 proper rotations of an axis-aligned object bbox."""

    matrices: list[Mat3] = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            rows = [[0.0, 0.0, 0.0] for _ in range(3)]
            for world_axis, local_axis in enumerate(permutation):
                rows[world_axis][local_axis] = signs[world_axis]
            matrix = tuple(tuple(row) for row in rows)
            if abs(_determinant3(matrix) - 1.0) <= _EPSILON:
                matrices.append(matrix)  # type: ignore[arg-type]
    matrices.sort(key=lambda matrix: (matrix != _IDENTITY_MAT3, matrix))
    return tuple(matrices)


_AXIS_ALIGNED_ROTATIONS = _axis_aligned_rotations()


def _yaw_rotation_matrix(quarter_turns: int) -> Mat3:
    return (
        _IDENTITY_MAT3,
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )[quarter_turns]


def _oriented_extent(extent: Vec3, rotation: Mat3) -> Vec3:
    return tuple(
        sum(
            abs(rotation[world_axis][local_axis]) * extent[local_axis]
            for local_axis in range(3)
        )
        for world_axis in range(3)
    )  # type: ignore[return-value]


def _finite_vec3(value: Any, *, label: str, positive: bool = False) -> Vec3:
    try:
        items = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise InsidePlacementGeometryError(f"{label} must be a finite 3-vector") from exc
    if len(items) != 3 or any(not math.isfinite(item) for item in items):
        raise InsidePlacementGeometryError(f"{label} must be a finite 3-vector")
    if positive and any(item <= 0.0 for item in items):
        raise InsidePlacementGeometryError(f"{label} must be strictly positive")
    return items  # type: ignore[return-value]


def _positive_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise InsidePlacementGeometryError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InsidePlacementGeometryError(
            f"{label} must be finite and positive"
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise InsidePlacementGeometryError(f"{label} must be finite and positive")
    return result


@dataclass(frozen=True, slots=True)
class Aabb3:
    """A validated world-frame axis-aligned bounding box."""

    lower: Vec3
    upper: Vec3

    def __post_init__(self) -> None:
        lower = _finite_vec3(self.lower, label="AABB lower")
        upper = _finite_vec3(self.upper, label="AABB upper")
        if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
            raise InsidePlacementGeometryError(
                "AABB upper must be greater than lower on every axis"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def from_center_extent(cls, center: Sequence[float], extent: Sequence[float]) -> "Aabb3":
        center_vec = _finite_vec3(center, label="AABB center")
        extent_vec = _finite_vec3(extent, label="AABB extent", positive=True)
        half = tuple(value / 2.0 for value in extent_vec)
        return cls(
            tuple(c - h for c, h in zip(center_vec, half, strict=True)),
            tuple(c + h for c, h in zip(center_vec, half, strict=True)),
        )

    @property
    def extent(self) -> Vec3:
        return tuple(
            hi - lo for lo, hi in zip(self.lower, self.upper, strict=True)
        )  # type: ignore[return-value]

    @property
    def center(self) -> Vec3:
        return tuple(
            (lo + hi) / 2.0
            for lo, hi in zip(self.lower, self.upper, strict=True)
        )  # type: ignore[return-value]

    @property
    def volume(self) -> float:
        x, y, z = self.extent
        return x * y * z

    def intersects(self, other: "Aabb3", *, clearance_m: float = 0.0) -> bool:
        clearance = float(clearance_m)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise InsidePlacementGeometryError(
                "AABB clearance must be finite and non-negative"
            )
        return all(
            self.upper[axis] + clearance > other.lower[axis]
            and other.upper[axis] + clearance > self.lower[axis]
            for axis in range(3)
        )

    def separation_m(self, other: "Aabb3") -> float:
        gaps = []
        for axis in range(3):
            gaps.append(
                max(
                    other.lower[axis] - self.upper[axis],
                    self.lower[axis] - other.upper[axis],
                    0.0,
                )
            )
        return math.sqrt(sum(value * value for value in gaps))

    def intersection_volume_m3(self, other: "Aabb3") -> float:
        """Return positive-volume overlap without treating contact as overlap."""

        overlap = tuple(
            max(
                0.0,
                min(self.upper[axis], other.upper[axis])
                - max(self.lower[axis], other.lower[axis]),
            )
            for axis in range(3)
        )
        return overlap[0] * overlap[1] * overlap[2]


def blocked_face_penetration_ratio(
    candidate_aabb: Aabb3,
    fillable_aabb: Aabb3,
    *,
    include_opening_face: bool = False,
) -> float:
    """Measure relative penetration through blocked container faces."""

    if not isinstance(include_opening_face, bool):
        raise InsidePlacementGeometryError(
            "include_opening_face must be boolean"
        )

    extent = candidate_aabb.extent
    return (
        max(fillable_aabb.lower[0] - candidate_aabb.lower[0], 0.0) / extent[0]
        + max(candidate_aabb.upper[0] - fillable_aabb.upper[0], 0.0) / extent[0]
        + max(fillable_aabb.lower[1] - candidate_aabb.lower[1], 0.0) / extent[1]
        + max(candidate_aabb.upper[1] - fillable_aabb.upper[1], 0.0) / extent[1]
        + max(fillable_aabb.lower[2] - candidate_aabb.lower[2], 0.0) / extent[2]
        + (
            max(candidate_aabb.upper[2] - fillable_aabb.upper[2], 0.0)
            / extent[2]
            if include_opening_face
            else 0.0
        )
    )


def _strict_leaf_truth(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, bool]:
    if not isinstance(value, Mapping) or not value:
        raise InsidePlacementGeometryError(
            f"{label} must be a non-empty leaf-truth mapping"
        )
    result: dict[str, bool] = {}
    for leaf_id, truth in value.items():
        if not isinstance(leaf_id, str) or not leaf_id:
            raise InsidePlacementGeometryError(
                f"{label} has an invalid goal-leaf identity"
            )
        if not isinstance(truth, bool):
            raise InsidePlacementGeometryError(
                f"{label} values must be booleans"
            )
        result[leaf_id] = truth
    return result


@dataclass(frozen=True, slots=True)
class InsideCandidateGoalPreservationDecision:
    """Commit decision against one action-start selected-option baseline."""

    attempted_leaf_id: str
    before_true_leaf_ids: tuple[str, ...]
    regressed_leaf_ids: tuple[str, ...]
    attempted_leaf_true: bool
    accepted: bool

    def private_record(self) -> Mapping[str, Any]:
        return {
            "version": INSIDE_CANDIDATE_GOAL_PRESERVATION_VERSION,
            "attempted_leaf_id_private": self.attempted_leaf_id,
            "before_true_leaf_ids_private": list(self.before_true_leaf_ids),
            "regressed_leaf_ids_private": list(self.regressed_leaf_ids),
            "attempted_leaf_true": self.attempted_leaf_true,
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class InsideActionGoalBaseline:
    """Immutable selected-option truth captured at one action boundary.

    This is deliberately not the episode-initial truth.  A new baseline is
    captured for every scheduled action so leaves achieved by an earlier
    action become preservation requirements for the next action.
    """

    leaf_truth_items: tuple[tuple[str, bool], ...]

    @property
    def leaf_truth(self) -> Mapping[str, bool]:
        return dict(self.leaf_truth_items)

    @property
    def true_leaf_ids(self) -> tuple[str, ...]:
        return tuple(
            leaf_id for leaf_id, truth in self.leaf_truth_items if truth
        )

    def private_record(self) -> Mapping[str, Any]:
        return {
            "version": INSIDE_CANDIDATE_GOAL_PRESERVATION_VERSION,
            "capture_boundary": "immediate_action_start",
            "leaf_ids_private": [
                leaf_id for leaf_id, _truth in self.leaf_truth_items
            ],
            "true_leaf_ids_private": list(self.true_leaf_ids),
        }


@dataclass(frozen=True, slots=True)
class InsideActionBaselinePreservationDecision:
    """Read-only pre-candidate decision for an in-flight Inside action."""

    action_start_true_leaf_ids: tuple[str, ...]
    regressed_leaf_ids: tuple[str, ...]
    accepted: bool

    def private_record(self) -> Mapping[str, Any]:
        return {
            "version": INSIDE_CANDIDATE_GOAL_PRESERVATION_VERSION,
            "capture_boundary": "immediate_action_start",
            "action_start_true_leaf_ids_private": list(
                self.action_start_true_leaf_ids
            ),
            "regressed_leaf_ids_private": list(self.regressed_leaf_ids),
            "accepted": self.accepted,
        }


def freeze_inside_action_goal_baseline(
    selected_leaf_truth: Mapping[str, Any],
) -> InsideActionGoalBaseline:
    """Freeze one selected option at the immediate action-start boundary."""

    normalized = _strict_leaf_truth(
        selected_leaf_truth,
        label="action-start selected-option truth",
    )
    return InsideActionGoalBaseline(tuple(normalized.items()))


def evaluate_inside_action_baseline_preservation(
    *,
    action_start_baseline: InsideActionGoalBaseline,
    current_selected_leaf_truth: Mapping[str, Any],
) -> InsideActionBaselinePreservationDecision:
    """Reject navigation / grasp drift before any candidate is applied."""

    if not isinstance(action_start_baseline, InsideActionGoalBaseline):
        raise InsidePlacementGeometryError(
            "action-start baseline has an invalid type"
        )
    before = dict(action_start_baseline.leaf_truth_items)
    current = _strict_leaf_truth(
        current_selected_leaf_truth,
        label="current selected-option truth",
    )
    if set(before) != set(current):
        raise InsidePlacementGeometryError(
            "selected-option goal-leaf identity changed during action"
        )
    regressed = tuple(
        leaf_id
        for leaf_id, truth in action_start_baseline.leaf_truth_items
        if truth and not current[leaf_id]
    )
    return InsideActionBaselinePreservationDecision(
        action_start_true_leaf_ids=action_start_baseline.true_leaf_ids,
        regressed_leaf_ids=regressed,
        accepted=not regressed,
    )


def evaluate_inside_candidate_goal_preservation(
    *,
    before_selected_leaf_truth: Mapping[str, Any],
    after_selected_leaf_truth: Mapping[str, Any],
    attempted_leaf_id: str,
) -> InsideCandidateGoalPreservationDecision:
    """Require target progress without regressing any action-start true leaf.

    The caller supplies the immediate action-start mapping, not a mapping
    captured after navigation or grasp.  Both mappings describe the same
    selected official option.  Missing, additional, or non-boolean leaves are
    contract failures rather than an invitation to weaken preservation.  The
    function is task-, category-, object-, and container-name agnostic.
    """

    before = _strict_leaf_truth(
        before_selected_leaf_truth,
        label="before selected-option truth",
    )
    after = _strict_leaf_truth(
        after_selected_leaf_truth,
        label="after selected-option truth",
    )
    if set(before) != set(after):
        raise InsidePlacementGeometryError(
            "selected-option goal-leaf identity changed during candidate"
        )
    if not isinstance(attempted_leaf_id, str) or attempted_leaf_id not in before:
        raise InsidePlacementGeometryError(
            "attempted Inside goal leaf is absent from the selected option"
        )
    before_true = tuple(sorted(leaf_id for leaf_id, truth in before.items() if truth))
    regressed = tuple(
        leaf_id for leaf_id in before_true if not after[leaf_id]
    )
    attempted_true = after[attempted_leaf_id]
    return InsideCandidateGoalPreservationDecision(
        attempted_leaf_id=attempted_leaf_id,
        before_true_leaf_ids=before_true,
        regressed_leaf_ids=regressed,
        attempted_leaf_true=attempted_true,
        accepted=bool(attempted_true and not regressed),
    )


@dataclass(frozen=True, slots=True)
class ContainerCapacityHint:
    """Live fillable geometry used to compare grounded assignments.

    ``usable_volume_m3`` should be supplied when the live fillable shape has a
    better volume estimate than its AABB (for example, a bowl).  Otherwise the
    interior AABB volume is used as a conservative, deterministic proxy.
    """

    interior_extent_m: Vec3
    usable_volume_m3: float | None = None

    def __post_init__(self) -> None:
        extent = _finite_vec3(
            self.interior_extent_m,
            label="container interior extent",
            positive=True,
        )
        object.__setattr__(self, "interior_extent_m", extent)
        if self.usable_volume_m3 is not None:
            object.__setattr__(
                self,
                "usable_volume_m3",
                _positive_finite(
                    self.usable_volume_m3,
                    label="container usable volume",
                ),
            )

    @property
    def capacity_m3(self) -> float:
        if self.usable_volume_m3 is not None:
            return self.usable_volume_m3
        x, y, z = self.interior_extent_m
        return x * y * z


@dataclass(frozen=True, slots=True)
class GroundedPlanPlacementRank:
    """Private geometry score for one official grounded goal option."""

    plan: Any
    geometry_feasible: bool
    oversized_relation_count: int
    overflow_container_count: int
    overflow_ratio_sum: float
    maximum_container_load_ratio: float
    load_ratio_square_sum: float
    maximum_assigned_object_count: int
    assigned_count_square_sum: int
    assignment_identity: tuple[tuple[str, str], ...]
    container_load_ratios: tuple[tuple[str, float], ...]
    container_assigned_counts: tuple[tuple[str, int], ...]

    @property
    def option_index(self) -> int:
        value = getattr(self.plan, "option_index", None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InsidePlacementGeometryError(
                "grounded plan option_index must be a non-negative integer"
            )
        return value

    @property
    def sort_key(self) -> tuple[Any, ...]:
        """A stable key: hard feasibility first, then balanced live load."""

        return (
            not self.geometry_feasible,
            self.oversized_relation_count,
            self.overflow_container_count,
            self.overflow_ratio_sum,
            self.maximum_container_load_ratio,
            self.load_ratio_square_sum,
            self.maximum_assigned_object_count,
            self.assigned_count_square_sum,
            self.assignment_identity,
            self.option_index,
        )

    def private_record(self) -> Mapping[str, Any]:
        return {
            "version": DETERMINISTIC_INSIDE_PLACEMENT_VERSION,
            "option_index": self.option_index,
            "geometry_feasible": self.geometry_feasible,
            "oversized_relation_count": self.oversized_relation_count,
            "overflow_container_count": self.overflow_container_count,
            "overflow_ratio_sum": self.overflow_ratio_sum,
            "maximum_container_load_ratio": self.maximum_container_load_ratio,
            "load_ratio_square_sum": self.load_ratio_square_sum,
            "maximum_assigned_object_count": self.maximum_assigned_object_count,
            "assigned_count_square_sum": self.assigned_count_square_sum,
            "assignment_identity_private": [list(item) for item in self.assignment_identity],
            "container_load_ratios_private": dict(self.container_load_ratios),
            "container_assigned_counts_private": dict(
                self.container_assigned_counts
            ),
        }


def _extent_volume(extent: Sequence[float], *, label: str) -> tuple[Vec3, float]:
    checked = _finite_vec3(extent, label=label, positive=True)
    return checked, checked[0] * checked[1] * checked[2]


def _can_fit_under_axis_permutation(
    object_extent: Vec3,
    container_extent: Vec3,
) -> bool:
    return all(
        object_dim <= container_dim + _EPSILON
        for object_dim, container_dim in zip(
            sorted(object_extent),
            sorted(container_extent),
            strict=True,
        )
    )


def rank_grounded_inside_plans(
    plans: Iterable[Any],
    *,
    object_extents_m: Mapping[str, Sequence[float]],
    container_capacity_hints: Mapping[str, ContainerCapacityHint],
    existing_contents: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> tuple[GroundedPlanPlacementRank, ...]:
    """Rank official options using only assignments and current live geometry.

    Existing contents form the baseline load.  If an assigned child is
    already present in any supplied container it is first removed from that
    baseline and then added to its goal parent, so rehoming is not double
    counted.  Missing geometry raises instead of silently choosing a frozen or
    category-specific option.
    """

    plan_items = tuple(plans)
    if not plan_items:
        raise InsidePlacementGeometryError("at least one grounded plan is required")
    if not container_capacity_hints:
        raise InsidePlacementGeometryError(
            "at least one live container capacity hint is required"
        )

    capacities: dict[str, ContainerCapacityHint] = {}
    for name, hint in container_capacity_hints.items():
        if not isinstance(name, str) or not name:
            raise InsidePlacementGeometryError(
                "container identities must be non-empty strings"
            )
        if not isinstance(hint, ContainerCapacityHint):
            raise InsidePlacementGeometryError(
                f"container {name!r} has an invalid capacity hint"
            )
        capacities[name] = hint

    object_extents: dict[str, Vec3] = {}
    object_volumes: dict[str, float] = {}
    for name, extent in object_extents_m.items():
        if not isinstance(name, str) or not name:
            raise InsidePlacementGeometryError(
                "object identities must be non-empty strings"
            )
        checked, volume = _extent_volume(
            extent,
            label=f"object {name!r} extent",
        )
        object_extents[name] = checked
        object_volumes[name] = volume

    contents = {} if existing_contents is None else existing_contents
    baseline_volume = {name: 0.0 for name in capacities}
    baseline_count = {name: 0 for name in capacities}
    current_parent_by_child: dict[str, str] = {}
    current_volume_by_child: dict[str, float] = {}
    for parent, children in contents.items():
        if parent not in capacities:
            raise InsidePlacementGeometryError(
                f"existing-content container {parent!r} has no capacity hint"
            )
        if not isinstance(children, Mapping):
            raise InsidePlacementGeometryError(
                f"existing contents for {parent!r} must be a mapping"
            )
        for child, extent in children.items():
            if not isinstance(child, str) or not child:
                raise InsidePlacementGeometryError(
                    "existing-content identities must be non-empty strings"
                )
            if child in current_parent_by_child:
                raise InsidePlacementGeometryError(
                    f"existing content {child!r} appears in multiple containers"
                )
            checked, volume = _extent_volume(
                extent,
                label=f"existing content {child!r} extent",
            )
            if child in object_extents and any(
                abs(left - right) > _EPSILON
                for left, right in zip(
                    checked,
                    object_extents[child],
                    strict=True,
                )
            ):
                raise InsidePlacementGeometryError(
                    f"object {child!r} has inconsistent live extents"
                )
            object_extents.setdefault(child, checked)
            object_volumes.setdefault(child, volume)
            current_parent_by_child[child] = parent
            current_volume_by_child[child] = volume
            baseline_volume[parent] += volume
            baseline_count[parent] += 1

    ranked: list[GroundedPlanPlacementRank] = []
    for plan in plan_items:
        option_index = getattr(plan, "option_index", None)
        if isinstance(option_index, bool) or not isinstance(option_index, int) or option_index < 0:
            raise InsidePlacementGeometryError(
                "grounded plan option_index must be a non-negative integer"
            )
        relations = getattr(plan, "positive_inside", None)
        if not isinstance(relations, (tuple, list)) or not relations:
            raise InsidePlacementGeometryError(
                f"grounded plan {option_index} has no positive Inside relations"
            )
        assignments: list[tuple[str, str]] = []
        parent_by_child: dict[str, str] = {}
        oversized = 0
        for relation in relations:
            child = getattr(relation, "child", None)
            parent = getattr(relation, "parent", None)
            if not all(isinstance(value, str) and value for value in (child, parent)):
                raise InsidePlacementGeometryError(
                    f"grounded plan {option_index} has an invalid Inside relation"
                )
            previous = parent_by_child.setdefault(child, parent)
            if previous != parent:
                raise InsidePlacementGeometryError(
                    f"grounded plan {option_index} assigns {child!r} twice"
                )
            if child not in object_extents:
                raise InsidePlacementGeometryError(
                    f"grounded plan {option_index} lacks live geometry for {child!r}"
                )
            if parent not in capacities:
                raise InsidePlacementGeometryError(
                    f"grounded plan {option_index} lacks capacity geometry for {parent!r}"
                )
            assignments.append((child, parent))
            if not _can_fit_under_axis_permutation(
                object_extents[child],
                capacities[parent].interior_extent_m,
            ):
                oversized += 1

        assignments = sorted(set(assignments))
        post_volume = dict(baseline_volume)
        post_count = dict(baseline_count)
        assigned_count = {name: 0 for name in capacities}
        for child, parent in assignments:
            current_parent = current_parent_by_child.get(child)
            if current_parent is not None:
                post_volume[current_parent] -= current_volume_by_child[child]
                post_count[current_parent] -= 1
            post_volume[parent] += object_volumes[child]
            post_count[parent] += 1
            assigned_count[parent] += 1

        load_ratios = {
            name: max(0.0, post_volume[name]) / hint.capacity_m3
            for name, hint in capacities.items()
        }
        overflowing = {
            name: ratio for name, ratio in load_ratios.items() if ratio > 1.0 + _EPSILON
        }
        overflow_sum = sum(ratio - 1.0 for ratio in overflowing.values())
        maximum_load = max(load_ratios.values(), default=0.0)
        load_square_sum = sum(value * value for value in load_ratios.values())
        maximum_count = max(assigned_count.values(), default=0)
        count_square_sum = sum(value * value for value in assigned_count.values())
        ranked.append(
            GroundedPlanPlacementRank(
                plan=plan,
                geometry_feasible=not oversized and not overflowing,
                oversized_relation_count=oversized,
                overflow_container_count=len(overflowing),
                overflow_ratio_sum=overflow_sum,
                maximum_container_load_ratio=maximum_load,
                load_ratio_square_sum=load_square_sum,
                maximum_assigned_object_count=maximum_count,
                assigned_count_square_sum=count_square_sum,
                assignment_identity=tuple(assignments),
                container_load_ratios=tuple(sorted(load_ratios.items())),
                container_assigned_counts=tuple(sorted(assigned_count.items())),
            )
        )
    ranked.sort(key=lambda item: item.sort_key)
    return tuple(ranked)


@dataclass(frozen=True, slots=True)
class InsidePlacementCandidate:
    """One deterministic world-frame initial pose candidate."""

    candidate_id: str
    position_m: Vec3
    yaw_quarter_turns: int
    oriented_extent_m: Vec3
    initial_aabb: Aabb3
    minimum_content_separation_m: float
    wall_clearance_m: float
    normalized_height: float
    orientation_matrix: Mat3 = _IDENTITY_MAT3
    blocked_face_penetration_ratio: float = 0.0
    opening_overflow_ratio: float = 0.0
    content_overlap_ratio: float = 0.0
    physical_intrusion_score: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise InsidePlacementGeometryError(
                "Inside candidate identity must be non-empty"
            )
        if self.yaw_quarter_turns not in {0, 1, 2, 3}:
            raise InsidePlacementGeometryError(
                "Inside candidate yaw must be a quarter turn"
            )
        if abs(_determinant3(self.orientation_matrix) - 1.0) > _EPSILON:
            raise InsidePlacementGeometryError(
                "Inside candidate orientation must be a proper rotation"
            )
        for label, value in (
            ("blocked-face penetration ratio", self.blocked_face_penetration_ratio),
            ("opening overflow ratio", self.opening_overflow_ratio),
            ("content overlap ratio", self.content_overlap_ratio),
            ("physical intrusion score", self.physical_intrusion_score),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise InsidePlacementGeometryError(
                    f"Inside candidate {label} must be finite and non-negative"
                )


@dataclass(frozen=True, slots=True)
class InsideCandidatePlan:
    version: str
    fillable_aabb: Aabb3
    held_extent_m: Vec3
    existing_content_count: int
    candidates: tuple[InsidePlacementCandidate, ...]
    rejected_outside_volume: int
    rejected_initial_overlap: int
    rejected_non_fitting_orientation: int
    rejected_gross_intrusion: int = 0

    def private_record(self) -> Mapping[str, Any]:
        return {
            "version": self.version,
            "fillable_aabb_private": {
                "lower": list(self.fillable_aabb.lower),
                "upper": list(self.fillable_aabb.upper),
            },
            "held_extent_m_private": list(self.held_extent_m),
            "existing_content_count_private": self.existing_content_count,
            "candidate_count": len(self.candidates),
            "rejected_outside_volume": self.rejected_outside_volume,
            "rejected_initial_overlap": self.rejected_initial_overlap,
            "rejected_non_fitting_orientation": (
                self.rejected_non_fitting_orientation
            ),
            "rejected_gross_intrusion": self.rejected_gross_intrusion,
        }


def _grid_value(lower: float, upper: float, fraction: float) -> float:
    return lower + fraction * (upper - lower)


def plan_deterministic_inside_candidates(
    *,
    held_extent_m: Sequence[float],
    fillable_aabb: Aabb3,
    existing_content_aabbs: Mapping[str, Aabb3] | None = None,
    point_in_fillable_volume: Callable[[Vec3], bool] | None = None,
    clearance_m: float = 0.002,
    max_candidates: int = 64,
) -> InsideCandidatePlan:
    """Build a stable, content-aware candidate sequence.

    Candidates start collision-free inside the inset fillable AABB.  Higher
    spawn positions and positions farther from existing content rank first so
    physics may settle the held object into remaining free volume.  A live
    fillable-volume callback can reject AABB corners that are outside a bowl
    or another non-box container.
    """

    extent = _finite_vec3(
        held_extent_m,
        label="held object extent",
        positive=True,
    )
    if not isinstance(fillable_aabb, Aabb3):
        raise InsidePlacementGeometryError("fillable_aabb must be Aabb3")
    clearance = float(clearance_m)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise InsidePlacementGeometryError(
            "candidate clearance must be finite and non-negative"
        )
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
        raise InsidePlacementGeometryError("max_candidates must be a positive integer")
    if point_in_fillable_volume is not None and not callable(point_in_fillable_volume):
        raise TypeError("point_in_fillable_volume must be callable")

    contents: list[tuple[str, Aabb3]] = []
    for name, aabb in (existing_content_aabbs or {}).items():
        if not isinstance(name, str) or not name:
            raise InsidePlacementGeometryError(
                "existing-content identities must be non-empty"
            )
        if not isinstance(aabb, Aabb3):
            raise InsidePlacementGeometryError(
                f"existing content {name!r} must provide Aabb3"
            )
        contents.append((name, aabb))
    contents.sort(key=lambda item: item[0])

    xy_fractions = (0.5, 0.25, 0.75, 0.0, 1.0)
    z_fractions = (1.0, 0.75, 0.5, 0.25, 0.0)
    rejected_outside = 0
    rejected_overlap = 0
    rejected_orientation = 0
    raw: list[tuple[tuple[Any, ...], InsidePlacementCandidate]] = []
    fill_center = fillable_aabb.center
    fill_diagonal = math.sqrt(
        sum(value * value for value in fillable_aabb.extent)
    )

    for yaw_quarter_turns in range(4):
        oriented = (
            extent
            if yaw_quarter_turns % 2 == 0
            else (extent[1], extent[0], extent[2])
        )
        allowed_lower = tuple(
            fillable_aabb.lower[axis] + oriented[axis] / 2.0 + clearance
            for axis in range(3)
        )
        allowed_upper = tuple(
            fillable_aabb.upper[axis] - oriented[axis] / 2.0 - clearance
            for axis in range(3)
        )
        if any(lo > hi + _EPSILON for lo, hi in zip(allowed_lower, allowed_upper, strict=True)):
            rejected_orientation += 1
            continue
        for z_fraction in z_fractions:
            for x_fraction in xy_fractions:
                for y_fraction in xy_fractions:
                    position = (
                        _grid_value(allowed_lower[0], allowed_upper[0], x_fraction),
                        _grid_value(allowed_lower[1], allowed_upper[1], y_fraction),
                        _grid_value(allowed_lower[2], allowed_upper[2], z_fraction),
                    )
                    if point_in_fillable_volume is not None:
                        try:
                            in_volume = point_in_fillable_volume(position)
                        except Exception as exc:
                            raise InsidePlacementGeometryError(
                                "live fillable-volume probe failed"
                            ) from exc
                        if not isinstance(in_volume, bool):
                            raise InsidePlacementGeometryError(
                                "live fillable-volume probe must return bool"
                            )
                        if not in_volume:
                            rejected_outside += 1
                            continue
                    candidate_aabb = Aabb3.from_center_extent(position, oriented)
                    if any(
                        candidate_aabb.intersects(aabb, clearance_m=clearance)
                        for _name, aabb in contents
                    ):
                        rejected_overlap += 1
                        continue
                    minimum_separation = (
                        fill_diagonal
                        if not contents
                        else min(
                            candidate_aabb.separation_m(aabb)
                            for _name, aabb in contents
                        )
                    )
                    wall_clearance = min(
                        candidate_aabb.lower[axis] - fillable_aabb.lower[axis]
                        for axis in range(3)
                    )
                    wall_clearance = min(
                        wall_clearance,
                        *(
                            fillable_aabb.upper[axis]
                            - candidate_aabb.upper[axis]
                            for axis in range(3)
                        ),
                    )
                    radial_distance = math.hypot(
                        position[0] - fill_center[0],
                        position[1] - fill_center[1],
                    )
                    candidate_id = (
                        f"yaw{yaw_quarter_turns}:"
                        f"z{z_fraction:.2f}:x{x_fraction:.2f}:y{y_fraction:.2f}"
                    )
                    candidate = InsidePlacementCandidate(
                        candidate_id=candidate_id,
                        position_m=position,
                        yaw_quarter_turns=yaw_quarter_turns,
                        oriented_extent_m=oriented,
                        initial_aabb=candidate_aabb,
                        minimum_content_separation_m=minimum_separation,
                        wall_clearance_m=wall_clearance,
                        normalized_height=z_fraction,
                        orientation_matrix=_yaw_rotation_matrix(
                            yaw_quarter_turns
                        ),
                    )
                    # Maximize free-space separation, spawn height, and wall
                    # clearance.  Prefer the center and the original yaw for
                    # exact deterministic ties.
                    key = (
                        -minimum_separation,
                        -z_fraction,
                        -wall_clearance,
                        radial_distance,
                        yaw_quarter_turns,
                        candidate_id,
                    )
                    raw.append((key, candidate))

    raw.sort(key=lambda item: item[0])
    candidates = tuple(candidate for _key, candidate in raw[:max_candidates])
    return InsideCandidatePlan(
        version=DETERMINISTIC_INSIDE_PLACEMENT_VERSION,
        fillable_aabb=fillable_aabb,
        held_extent_m=extent,
        existing_content_count=len(contents),
        candidates=candidates,
        rejected_outside_volume=rejected_outside,
        rejected_initial_overlap=rejected_overlap,
        rejected_non_fitting_orientation=rejected_orientation,
    )


def plan_symbolic_inside_candidates(
    *,
    held_extent_m: Sequence[float],
    fillable_aabb: Aabb3,
    existing_content_aabbs: Mapping[str, Aabb3] | None = None,
    point_in_fillable_volume: Callable[[Vec3], bool] | None = None,
    max_candidates: int = 48,
    maximum_blocked_face_penetration_ratio: float = (
        DEFAULT_MAX_BLOCKED_FACE_PENETRATION_RATIO
    ),
    maximum_content_overlap_ratio: float = DEFAULT_MAX_CONTENT_OVERLAP_RATIO,
    include_opening_face_penetration: bool = False,
    preferred_orientation_xyzw: Sequence[float] | None = None,
) -> InsideCandidatePlan:
    """Generate high-level Inside poses without solid-volume overlap.

    BEHAVIOR's final ``Inside`` predicate is a centre-in-fillable-volume
    relation.  Its official setter is substantially stricter: it first tries
    to fit the complete object AABB inside the fillable AABB and rejects any
    initial contact.  That makes long but semantically valid objects (for
    example an item resting across a bowl rim) impossible to place.

    By default this planner keeps the *candidate centre* inside the live
    fillable volume and rejects penetration through blocked faces or existing
    contents. Overflow through an explicitly open top remains legal. The
    simulator adapter accepts a candidate only after official ``Inside``
    readback succeeds.
    """

    extent = _finite_vec3(
        held_extent_m,
        label="held object extent",
        positive=True,
    )
    if not isinstance(fillable_aabb, Aabb3):
        raise InsidePlacementGeometryError("fillable_aabb must be Aabb3")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
        raise InsidePlacementGeometryError("max_candidates must be a positive integer")
    maximum_intrusion = float(maximum_blocked_face_penetration_ratio)
    if not math.isfinite(maximum_intrusion) or not 0.0 <= maximum_intrusion < 1.0:
        raise InsidePlacementGeometryError(
            "maximum blocked-face penetration ratio must be in [0, 1)"
        )
    maximum_content_overlap = float(maximum_content_overlap_ratio)
    if (
        not math.isfinite(maximum_content_overlap)
        or not 0.0 <= maximum_content_overlap < 1.0
    ):
        raise InsidePlacementGeometryError(
            "maximum content-overlap ratio must be in [0, 1)"
        )
    if not isinstance(include_opening_face_penetration, bool):
        raise InsidePlacementGeometryError(
            "include_opening_face_penetration must be boolean"
        )
    if point_in_fillable_volume is not None and not callable(point_in_fillable_volume):
        raise TypeError("point_in_fillable_volume must be callable")

    contents: list[tuple[str, Aabb3]] = []
    for name, aabb in (existing_content_aabbs or {}).items():
        if not isinstance(name, str) or not name:
            raise InsidePlacementGeometryError(
                "existing-content identities must be non-empty"
            )
        if not isinstance(aabb, Aabb3):
            raise InsidePlacementGeometryError(
                f"existing content {name!r} must provide Aabb3"
            )
        contents.append((name, aabb))
    contents.sort(key=lambda item: item[0])

    # Centre-first, then bounded offsets. Prefer a grounded centre and retain
    # higher candidates only when existing contents occupy the bottom.
    # Intrusion remains in the sort key for explicitly non-default research
    # callers. The Action20 default admits only zero-overlap candidates.
    xy_fractions = (0.50, 0.35, 0.65, 0.20, 0.80)
    rejected_outside = 0
    rejected_overlap = 0
    rejected_gross_intrusion = 0
    raw: list[tuple[tuple[Any, ...], InsidePlacementCandidate]] = []
    rotations = _AXIS_ALIGNED_ROTATIONS
    if preferred_orientation_xyzw is not None:
        from .pixel_manipulation import quaternion_xyzw_to_matrix
        preferred = tuple(tuple(float(v) for v in row) for row in
                          quaternion_xyzw_to_matrix(preferred_orientation_xyzw))
        rotations = (preferred, *(rotation for rotation in rotations
            if max(abs(rotation[i][j] - preferred[i][j]) for i in range(3) for j in range(3)) > 1e-6))
    longest_local_axis = max(range(3), key=lambda axis: extent[axis])
    for orientation_index, orientation_matrix in enumerate(
        rotations
    ):
        oriented = _oriented_extent(extent, orientation_matrix)
        fillable_height = fillable_aabb.extent[2]
        grounded_fraction = oriented[2] / (2.0 * fillable_height)
        z_fractions = tuple(dict.fromkeys((
            grounded_fraction,
            0.35,
            0.50,
            0.65,
            0.80,
            0.95,
        )))
        for z_fraction in z_fractions:
            if not 0.0 <= z_fraction <= 1.0:
                rejected_outside += len(xy_fractions) ** 2
                continue
            for x_fraction in xy_fractions:
                for y_fraction in xy_fractions:
                    position = (
                        _grid_value(fillable_aabb.lower[0], fillable_aabb.upper[0], x_fraction),
                        _grid_value(fillable_aabb.lower[1], fillable_aabb.upper[1], y_fraction),
                        _grid_value(fillable_aabb.lower[2], fillable_aabb.upper[2], z_fraction),
                    )
                    if point_in_fillable_volume is not None:
                        try:
                            in_volume = point_in_fillable_volume(position)
                        except Exception as exc:
                            raise InsidePlacementGeometryError(
                                "live fillable-volume probe failed"
                            ) from exc
                        if not isinstance(in_volume, bool):
                            raise InsidePlacementGeometryError(
                                "live fillable-volume probe must return bool"
                            )
                        if not in_volume:
                            rejected_outside += 1
                            continue
                    candidate_aabb = Aabb3.from_center_extent(position, oriented)
                    overlap_volumes = tuple(
                        candidate_aabb.intersection_volume_m3(aabb)
                        for _name, aabb in contents
                    )
                    overlap_count = sum(value > 0.0 for value in overlap_volumes)
                    overlap_volume_m3 = sum(overlap_volumes)
                    content_overlap_ratio = min(
                        1.0,
                        overlap_volume_m3 / candidate_aabb.volume,
                    )
                    if content_overlap_ratio > maximum_content_overlap + _EPSILON:
                        rejected_overlap += 1
                        continue
                    minimum_separation = (
                        math.inf
                        if not contents
                        else min(
                            candidate_aabb.separation_m(aabb)
                            for _name, aabb in contents
                        )
                    )
                    wall_clearance = min(
                        *(
                            candidate_aabb.lower[axis] - fillable_aabb.lower[axis]
                            for axis in range(3)
                        ),
                        *(
                            fillable_aabb.upper[axis] - candidate_aabb.upper[axis]
                            for axis in range(3)
                        ),
                    )
                    blocked_penetration = blocked_face_penetration_ratio(
                        candidate_aabb,
                        fillable_aabb,
                        include_opening_face=(
                            include_opening_face_penetration
                        ),
                    )
                    if blocked_penetration > maximum_intrusion + _EPSILON:
                        rejected_gross_intrusion += 1
                        continue
                    opening_overflow_ratio = max(
                        candidate_aabb.upper[2] - fillable_aabb.upper[2],
                        0.0,
                    ) / oriented[2]
                    physical_intrusion_score = (
                        blocked_penetration + content_overlap_ratio
                    )
                    candidate_id = (
                        f"symbolic:r{orientation_index}:"
                        f"z{z_fraction:.2f}:x{x_fraction:.2f}:y{y_fraction:.2f}"
                    )
                    candidate = InsidePlacementCandidate(
                        candidate_id=candidate_id,
                        position_m=position,
                        yaw_quarter_turns=0,
                        oriented_extent_m=oriented,
                        initial_aabb=candidate_aabb,
                        minimum_content_separation_m=minimum_separation,
                        wall_clearance_m=wall_clearance,
                        normalized_height=z_fraction,
                        orientation_matrix=orientation_matrix,
                        blocked_face_penetration_ratio=blocked_penetration,
                        opening_overflow_ratio=opening_overflow_ratio,
                        content_overlap_ratio=content_overlap_ratio,
                        physical_intrusion_score=physical_intrusion_score,
                    )
                    positive_long_axis_up = orientation_matrix[2][
                        longest_local_axis
                    ]
                    # Prefer the least total intrusion, then the lowest
                    # collision-free pose. Upper overflow is legal for an
                    # open container and only breaks late ties.
                    raw.append(
                        (
                            (
                                physical_intrusion_score,
                                blocked_penetration,
                                content_overlap_ratio,
                                overlap_count,
                                overlap_volume_m3,
                                opening_overflow_ratio,
                                z_fraction,
                                -minimum_separation,
                                abs(x_fraction - 0.5) + abs(y_fraction - 0.5),
                                orientation_index != 0,
                                -positive_long_axis_up,
                                orientation_index,
                                candidate_id,
                            ),
                            candidate,
                        )
                    )

    raw.sort(key=lambda item: item[0])
    if preferred_orientation_xyzw is not None:
        preferred_rows = [row for row in raw if row[1].orientation_matrix == preferred]
        fallback_rows = [row for row in raw if row[1].orientation_matrix != preferred]
        # Try several original-orientation positions first, without starving
        # fallback orientations of the existing finite candidate budget.
        preferred_count = min(len(preferred_rows), max(1, max_candidates // 2)) if fallback_rows else max_candidates
        selected = preferred_rows[:preferred_count] + fallback_rows[:max_candidates - preferred_count]
        if len(selected) < max_candidates:
            selected += preferred_rows[preferred_count:preferred_count + max_candidates - len(selected)]
            selected.sort(key=lambda row: row[1].orientation_matrix != preferred)
        raw = selected
    return InsideCandidatePlan(
        version=SYMBOLIC_INSIDE_PLACEMENT_VERSION,
        fillable_aabb=fillable_aabb,
        held_extent_m=extent,
        existing_content_count=len(contents),
        candidates=tuple(candidate for _key, candidate in raw[:max_candidates]),
        rejected_outside_volume=rejected_outside,
        rejected_initial_overlap=rejected_overlap,
        rejected_non_fitting_orientation=0,
        rejected_gross_intrusion=rejected_gross_intrusion,
    )


@dataclass(frozen=True, slots=True)
class InsidePlacementAttempt:
    attempt_index: int
    candidate_id: str
    attempt_returned: bool
    inside_verified: bool
    error: str | None
    recovery_called: bool
    held_after_recovery: bool | None


@dataclass(frozen=True, slots=True)
class InsidePlacementExecutionReport:
    version: str
    success: bool
    selected_candidate_id: str | None
    attempts: tuple[InsidePlacementAttempt, ...]
    held_object_retained: bool
    recoverable_executor_noop: bool
    failure_reason: str | None

    def private_record(self) -> Mapping[str, Any]:
        return {
            "version": self.version,
            "success": self.success,
            "selected_candidate_id_private": self.selected_candidate_id,
            "attempt_count": len(self.attempts),
            "attempts_private": [
                {
                    "attempt_index": item.attempt_index,
                    "candidate_id": item.candidate_id,
                    "attempt_returned": item.attempt_returned,
                    "inside_verified": item.inside_verified,
                    "error": item.error,
                    "recovery_called": item.recovery_called,
                    "held_after_recovery": item.held_after_recovery,
                }
                for item in self.attempts
            ],
            "held_object_retained": self.held_object_retained,
            "recoverable_executor_noop": self.recoverable_executor_noop,
            "failure_reason": self.failure_reason,
        }


def _callback_bool(
    callback: Callable[..., Any],
    *args: Any,
    label: str,
) -> bool:
    value = callback(*args)
    if not isinstance(value, bool):
        raise TypeError(f"{label} must return bool")
    return value


def execute_recoverable_inside_candidates(
    candidates: Iterable[InsidePlacementCandidate],
    *,
    attempt_candidate: Callable[[InsidePlacementCandidate], bool],
    verify_official_inside: Callable[[], bool],
    recover_held: Callable[[InsidePlacementCandidate], None],
    held_is_retained: Callable[[], bool],
) -> InsidePlacementExecutionReport:
    """Try candidates and preserve hold after every unsuccessful mutation.

    ``attempt_candidate`` may apply a pose and settle physics.  Success is
    accepted only from ``verify_official_inside``; the attempt callback's
    return is diagnostic.  Any ordinary attempt/readback exception is treated
    as a failed candidate and followed by recovery.  If recovery raises or
    held-object readback is false, the function fails closed with
    :class:`InsidePlacementRecoveryError` instead of claiming the episode can
    continue.
    """

    for name, callback in (
        ("attempt_candidate", attempt_candidate),
        ("verify_official_inside", verify_official_inside),
        ("recover_held", recover_held),
        ("held_is_retained", held_is_retained),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")
    try:
        initially_held = _callback_bool(
            held_is_retained,
            label="held_is_retained",
        )
    except Exception as exc:
        raise InsidePlacementRecoveryError(
            "held-object precondition readback failed"
        ) from exc
    if not initially_held:
        raise InsidePlacementRecoveryError(
            "deterministic Inside placement requires a retained held object"
        )

    candidate_items = tuple(candidates)
    if any(not isinstance(item, InsidePlacementCandidate) for item in candidate_items):
        raise TypeError("all Inside candidates must be InsidePlacementCandidate")
    candidate_ids = tuple(item.candidate_id for item in candidate_items)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Inside candidate identities must be unique")

    records: list[InsidePlacementAttempt] = []
    for attempt_index, candidate in enumerate(candidate_items):
        attempt_returned = False
        inside_verified = False
        errors: list[str] = []
        try:
            attempt_returned = _callback_bool(
                attempt_candidate,
                candidate,
                label="attempt_candidate",
            )
        except Exception as exc:
            errors.append(f"attempt:{type(exc).__name__}: {exc}")
        try:
            inside_verified = _callback_bool(
                verify_official_inside,
                label="verify_official_inside",
            )
        except Exception as exc:
            errors.append(f"verify:{type(exc).__name__}: {exc}")
        if inside_verified:
            records.append(
                InsidePlacementAttempt(
                    attempt_index=attempt_index,
                    candidate_id=candidate.candidate_id,
                    attempt_returned=attempt_returned,
                    inside_verified=True,
                    error="; ".join(errors) or None,
                    recovery_called=False,
                    held_after_recovery=None,
                )
            )
            return InsidePlacementExecutionReport(
                version=DETERMINISTIC_INSIDE_PLACEMENT_VERSION,
                success=True,
                selected_candidate_id=candidate.candidate_id,
                attempts=tuple(records),
                held_object_retained=False,
                recoverable_executor_noop=False,
                failure_reason=None,
            )

        try:
            recover_held(candidate)
        except Exception as exc:
            raise InsidePlacementRecoveryError(
                f"candidate {candidate.candidate_id!r} recovery raised"
            ) from exc
        try:
            held_after_recovery = _callback_bool(
                held_is_retained,
                label="held_is_retained",
            )
        except Exception as exc:
            raise InsidePlacementRecoveryError(
                f"candidate {candidate.candidate_id!r} hold readback failed"
            ) from exc
        records.append(
            InsidePlacementAttempt(
                attempt_index=attempt_index,
                candidate_id=candidate.candidate_id,
                attempt_returned=attempt_returned,
                inside_verified=False,
                error="; ".join(errors) or None,
                recovery_called=True,
                held_after_recovery=held_after_recovery,
            )
        )
        if not held_after_recovery:
            raise InsidePlacementRecoveryError(
                f"candidate {candidate.candidate_id!r} did not restore hold"
            )

    return InsidePlacementExecutionReport(
        version=DETERMINISTIC_INSIDE_PLACEMENT_VERSION,
        success=False,
        selected_candidate_id=None,
        attempts=tuple(records),
        held_object_retained=True,
        recoverable_executor_noop=True,
        failure_reason=(
            "no_geometry_candidate"
            if not candidate_items
            else "all_geometry_candidates_failed"
        ),
    )
