"""Pure geometry for the private object-directed R1Pro Oracle.

The module intentionally has no simulator imports and never chooses an
object.  Live code supplies the private object position and uses the returned
station geometry only to issue ordinary velocity commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


class PrivateOracleGeometryError(ValueError):
    """Raised when a private station geometry value is malformed."""


@dataclass(frozen=True, slots=True)
class RadialStation:
    """A desired collision-aware base station around a private target."""

    station_id: str
    object_xy: tuple[float, float]
    base_xy: tuple[float, float]
    radius_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        if not isinstance(self.station_id, str) or not self.station_id:
            raise PrivateOracleGeometryError("station_id must be non-empty")
        object.__setattr__(self, "object_xy", _vector(self.object_xy, 2, "object_xy"))
        object.__setattr__(self, "base_xy", _vector(self.base_xy, 2, "base_xy"))
        radius = _finite_positive(self.radius_m, "radius_m")
        if radius < 0.20:
            raise PrivateOracleGeometryError("station radius is unsafe")
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "yaw_rad", wrap_angle(_finite(self.yaw_rad, "yaw_rad")))


@dataclass(frozen=True, slots=True)
class TableEdgePlacement:
    """Private calibration placement that remains supported by a tabletop."""

    placement_id: str
    object_center_xyz: tuple[float, float, float]
    table_edge: str
    inset_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.placement_id, str) or not self.placement_id:
            raise PrivateOracleGeometryError("placement_id must be non-empty")
        object.__setattr__(
            self,
            "object_center_xyz",
            _vector(self.object_center_xyz, 3, "object_center_xyz"),
        )
        if self.table_edge not in {"x_min", "x_max", "y_min", "y_max"}:
            raise PrivateOracleGeometryError("table_edge is invalid")
        object.__setattr__(self, "inset_m", _finite_positive(self.inset_m, "inset_m"))


@dataclass(frozen=True, slots=True)
class LocalSupportPlacement:
    """Small private tabletop perturbation toward the current robot base."""

    placement_id: str
    object_center_xyz: tuple[float, float, float]
    displacement_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.placement_id, str) or not self.placement_id:
            raise PrivateOracleGeometryError("placement_id must be non-empty")
        object.__setattr__(
            self,
            "object_center_xyz",
            _vector(self.object_center_xyz, 3, "object_center_xyz"),
        )
        object.__setattr__(
            self,
            "displacement_m",
            _finite_positive(self.displacement_m, "displacement_m"),
        )


@dataclass(frozen=True, slots=True)
class TennisBallGraspVariant:
    """Deterministic private sphere grasp geometry for the R1Pro EEF."""

    variant_id: str
    close_position_xyz: tuple[float, float, float]
    top_down_quaternion_xyzw: tuple[float, float, float, float]
    height_fraction: float
    yaw_rad: float

    def __post_init__(self) -> None:
        if not isinstance(self.variant_id, str) or not self.variant_id:
            raise PrivateOracleGeometryError("variant_id must be non-empty")
        object.__setattr__(
            self,
            "close_position_xyz",
            _vector(self.close_position_xyz, 3, "close_position_xyz"),
        )
        quaternion = np.asarray(
            _vector(
                self.top_down_quaternion_xyzw,
                4,
                "top_down_quaternion_xyzw",
            ),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-8:
            raise PrivateOracleGeometryError("grasp quaternion has zero norm")
        object.__setattr__(
            self,
            "top_down_quaternion_xyzw",
            tuple(float(value) for value in quaternion / norm),
        )
        object.__setattr__(
            self, "height_fraction", _finite(self.height_fraction, "height_fraction")
        )
        object.__setattr__(self, "yaw_rad", wrap_angle(self.yaw_rad))


def tennis_ball_center_grasp_variants(
    *,
    object_center_xyz: Sequence[float],
    object_extent_xyz: Sequence[float],
    table_edge: str,
) -> tuple[TennisBallGraspVariant, ...]:
    """Return center/equator-biased, edge-aligned top-down sphere grasps.

    R1Pro's EEF y-axis runs from the left finger to the right finger, so its
    horizontal closure axis is aligned to the table-edge tangent.  The
    quaternion includes the official R1Pro top-down x-axis flip used by
    ``StarterSemanticActionPrimitives._sample_grasp_pose``.
    """

    center = np.asarray(
        _vector(object_center_xyz, 3, "object_center_xyz"), dtype=np.float64
    )
    extent = np.asarray(
        _vector(object_extent_xyz, 3, "object_extent_xyz"), dtype=np.float64
    )
    if np.any(extent <= 0.0):
        raise PrivateOracleGeometryError("object extent must be positive")
    if table_edge not in {"x_min", "x_max", "y_min", "y_max"}:
        raise PrivateOracleGeometryError("table_edge is invalid")
    # EEF y is the closure axis.  y-aligned closure is tangent to x edges;
    # rotate 90 degrees to make it x-aligned for y edges.
    edge_yaw = 0.0 if table_edge.startswith("x_") else math.pi / 2.0
    definitions = (
        ("center_official_top_down", 0.0, 0.0),
        ("center_edge_aligned", 0.0, edge_yaw),
        ("center_edge_aligned_reverse", 0.0, edge_yaw + math.pi),
        ("center_edge_diagonal", 0.0, edge_yaw + math.pi / 4.0),
        ("upper_quarter_edge_aligned", 0.25, edge_yaw),
        ("lower_quarter_edge_aligned", -0.25, edge_yaw),
    )
    variants: list[TennisBallGraspVariant] = []
    for variant_id, height_fraction, yaw in definitions:
        position = center.copy()
        position[2] += height_fraction * extent[2]
        half = yaw * 0.5
        # q_world_yaw * q_r1pro_top_down_x_flip in xyzw convention.
        quaternion = (math.cos(half), math.sin(half), 0.0, 0.0)
        variants.append(
            TennisBallGraspVariant(
                variant_id=variant_id,
                close_position_xyz=tuple(float(value) for value in position),
                top_down_quaternion_xyzw=quaternion,
                height_fraction=height_fraction,
                yaw_rad=yaw,
            )
        )
    return tuple(variants)


def wrap_angle(angle_rad: float) -> float:
    """Return an angle in [-pi, pi)."""

    value = _finite(angle_rad, "angle_rad")
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_xy_quaternion(quaternion_xyzw: Sequence[float]) -> float:
    """Extract planar yaw from an xyzw quaternion."""

    x, y, z, w = _vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-8:
        raise PrivateOracleGeometryError("quaternion has zero norm")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def radial_station_candidates(
    *,
    object_xy: Sequence[float],
    current_base_xy: Sequence[float],
    radii_m: Iterable[float] = (0.62, 0.72, 0.82),
    angular_offsets_rad: Iterable[float] = (0.0, 0.55, -0.55, 1.10, -1.10),
    arm_workspace_yaw_rad: float = 0.0,
) -> tuple[RadialStation, ...]:
    """Generate deterministic stations, starting along the current approach.

    The returned positions are *targets for velocity navigation*, not state
    writes.  Duplicate positions/radii are removed while preserving order.
    """

    target = np.asarray(_vector(object_xy, 2, "object_xy"), dtype=np.float64)
    base = np.asarray(_vector(current_base_xy, 2, "current_base_xy"), dtype=np.float64)
    radial = base - target
    if float(np.linalg.norm(radial)) <= 1e-6:
        radial = np.asarray((1.0, 0.0), dtype=np.float64)
    reference_angle = math.atan2(float(radial[1]), float(radial[0]))
    stations: list[RadialStation] = []
    seen: set[tuple[float, float]] = set()
    for radius in radii_m:
        radius_value = _finite_positive(radius, "radius_m")
        for offset in angular_offsets_rad:
            angle = reference_angle + _finite(offset, "angular_offset_rad")
            station_xy = target + radius_value * np.asarray(
                (math.cos(angle), math.sin(angle)), dtype=np.float64
            )
            key = tuple(round(float(value), 6) for value in station_xy)
            if key in seen:
                continue
            seen.add(key)
            # Match StarterSemanticActionPrimitives._sample_pose_near_object:
            # the base faces the target minus the mean arm-workspace yaw so
            # the selected arm, rather than the chassis centreline, is aimed
            # at the manipulation target.
            yaw = math.atan2(
                float(target[1] - station_xy[1]),
                float(target[0] - station_xy[0]),
            ) - _finite(arm_workspace_yaw_rad, "arm_workspace_yaw_rad")
            stations.append(
                RadialStation(
                    station_id=f"r{radius_value:.2f}_a{offset:+.2f}",
                    object_xy=tuple(float(value) for value in target),
                    base_xy=tuple(float(value) for value in station_xy),
                    radius_m=radius_value,
                    yaw_rad=yaw,
                )
            )
    if not stations:
        raise PrivateOracleGeometryError("station generator produced no candidates")
    return tuple(stations)


def rank_radial_stations_by_measured_cost(
    stations: Iterable[RadialStation],
    *,
    current_base_xy: Sequence[float],
    current_yaw_rad: float,
    planar_speed_mps: float = 0.20,
    yaw_speed_radps: float = 0.18,
) -> tuple[tuple[RadialStation, float], ...]:
    """Rank stations by a deterministic measured-pose travel-time proxy.

    This helper does not inspect an occupancy map or mutate simulator state.
    It uses only the robot's measured world pose and the already-validated
    station poses.  Stable ``station_id`` tie-breaking makes the private
    Oracle audit reproducible.
    """

    base = np.asarray(_vector(current_base_xy, 2, "current_base_xy"), dtype=np.float64)
    yaw = wrap_angle(_finite(current_yaw_rad, "current_yaw_rad"))
    planar_speed = _finite_positive(planar_speed_mps, "planar_speed_mps")
    yaw_speed = _finite_positive(yaw_speed_radps, "yaw_speed_radps")
    ranked: list[tuple[RadialStation, float]] = []
    seen_ids: set[str] = set()
    for station in stations:
        if not isinstance(station, RadialStation):
            raise PrivateOracleGeometryError("stations must contain RadialStation values")
        if station.station_id in seen_ids:
            raise PrivateOracleGeometryError("station_id values must be unique")
        seen_ids.add(station.station_id)
        distance = float(np.linalg.norm(np.asarray(station.base_xy) - base))
        yaw_error = abs(wrap_angle(station.yaw_rad - yaw))
        cost = distance / planar_speed + yaw_error / yaw_speed
        ranked.append((station, float(cost)))
    if not ranked:
        raise PrivateOracleGeometryError("at least one station is required")
    ranked.sort(key=lambda item: (item[1], item[0].station_id))
    return tuple(ranked)


def table_edge_placements(
    *,
    table_aabb_min: Sequence[float],
    table_aabb_max: Sequence[float],
    object_extent_xyz: Sequence[float],
    current_base_xy: Sequence[float],
    inset_m: float = 0.08,
    clearance_m: float = 0.004,
    lateral_fractions: Iterable[float] = (0.0, -0.35, 0.35),
) -> tuple[TableEdgePlacement, ...]:
    """Return deterministic, fully supported tabletop edge placements.

    These positions are for a private, explicitly non-scored calibration
    variant.  The object remains inside the tabletop XY bounds by its half
    extent plus ``inset_m`` and is placed immediately above the table top.
    Candidates nearest the robot are returned first.
    """

    minimum = np.asarray(_vector(table_aabb_min, 3, "table_aabb_min"), dtype=np.float64)
    maximum = np.asarray(_vector(table_aabb_max, 3, "table_aabb_max"), dtype=np.float64)
    extent = np.asarray(_vector(object_extent_xyz, 3, "object_extent_xyz"), dtype=np.float64)
    base = np.asarray(_vector(current_base_xy, 2, "current_base_xy"), dtype=np.float64)
    inset = _finite_positive(inset_m, "inset_m")
    clearance = _finite(clearance_m, "clearance_m")
    if clearance < 0.0:
        raise PrivateOracleGeometryError("clearance_m must be non-negative")
    if np.any(maximum <= minimum):
        raise PrivateOracleGeometryError("table AABB must have positive extent")
    if np.any(extent <= 0.0):
        raise PrivateOracleGeometryError("object extent must be positive")

    half = extent * 0.5
    safe_min = minimum[:2] + half[:2] + inset
    safe_max = maximum[:2] - half[:2] - inset
    if np.any(safe_max <= safe_min):
        raise PrivateOracleGeometryError("tabletop is too small for requested inset")
    center = (safe_min + safe_max) * 0.5
    half_span = (safe_max - safe_min) * 0.5
    z = float(maximum[2] + half[2] + clearance)

    placements: list[TableEdgePlacement] = []
    seen: set[tuple[float, float]] = set()
    for fraction in lateral_fractions:
        fraction_value = _finite(fraction, "lateral_fraction")
        if abs(fraction_value) > 1.0:
            raise PrivateOracleGeometryError("lateral fractions must be in [-1, 1]")
        candidates = (
            ("x_min", safe_min[0], center[1] + fraction_value * half_span[1]),
            ("x_max", safe_max[0], center[1] + fraction_value * half_span[1]),
            ("y_min", center[0] + fraction_value * half_span[0], safe_min[1]),
            ("y_max", center[0] + fraction_value * half_span[0], safe_max[1]),
        )
        for edge, x, y in candidates:
            key = (round(float(x), 6), round(float(y), 6))
            if key in seen:
                continue
            seen.add(key)
            placements.append(
                TableEdgePlacement(
                    placement_id=f"{edge}_f{fraction_value:+.2f}",
                    object_center_xyz=(float(x), float(y), z),
                    table_edge=edge,
                    inset_m=inset,
                )
            )
    placements.sort(
        key=lambda item: float(
            np.linalg.norm(np.asarray(item.object_center_xyz[:2]) - base)
        )
    )
    if not placements:
        raise PrivateOracleGeometryError("table edge placement generator produced no candidates")
    return tuple(placements)


def local_support_placements_toward_robot(
    *,
    object_center_xyz: Sequence[float],
    current_base_xy: Sequence[float],
    displacements_m: Iterable[float] = (0.08, 0.12, 0.16, 0.20),
) -> tuple[LocalSupportPlacement, ...]:
    """Move a supported object a small planar distance toward the robot.

    The original Z coordinate is preserved exactly.  Live code must still
    settle the object and require the canonical ``OnTop`` predicate before a
    physical primitive is allowed to run.
    """

    center = np.asarray(
        _vector(object_center_xyz, 3, "object_center_xyz"), dtype=np.float64
    )
    base = np.asarray(_vector(current_base_xy, 2, "current_base_xy"), dtype=np.float64)
    direction = base - center[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        raise PrivateOracleGeometryError("robot and object XY positions coincide")
    direction /= norm
    placements: list[LocalSupportPlacement] = []
    previous = 0.0
    for displacement in displacements_m:
        distance = _finite_positive(displacement, "displacement_m")
        if distance <= previous:
            raise PrivateOracleGeometryError(
                "displacements_m must be strictly increasing"
            )
        previous = distance
        position = center.copy()
        position[:2] += direction * distance
        placements.append(
            LocalSupportPlacement(
                placement_id=f"toward_robot_d{distance:.2f}",
                object_center_xyz=tuple(float(value) for value in position),
                displacement_m=distance,
            )
        )
    if not placements:
        raise PrivateOracleGeometryError("local support generator produced no candidates")
    return tuple(placements)


def local_planar_error(
    *,
    current_base_xy: Sequence[float],
    current_yaw_rad: float,
    station: RadialStation,
) -> tuple[float, float, float]:
    """Return station error in robot-local x/y and yaw error."""

    current = np.asarray(_vector(current_base_xy, 2, "current_base_xy"), dtype=np.float64)
    delta_world = np.asarray(station.base_xy, dtype=np.float64) - current
    cosine = math.cos(current_yaw_rad)
    sine = math.sin(current_yaw_rad)
    local_x = cosine * delta_world[0] + sine * delta_world[1]
    local_y = -sine * delta_world[0] + cosine * delta_world[1]
    return float(local_x), float(local_y), wrap_angle(station.yaw_rad - current_yaw_rad)


def _vector(value: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise PrivateOracleGeometryError(f"{name} must be a finite {size}D vector")
    return tuple(float(item) for item in array)


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrivateOracleGeometryError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise PrivateOracleGeometryError(f"{name} must be finite")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise PrivateOracleGeometryError(f"{name} must be positive")
    return result
