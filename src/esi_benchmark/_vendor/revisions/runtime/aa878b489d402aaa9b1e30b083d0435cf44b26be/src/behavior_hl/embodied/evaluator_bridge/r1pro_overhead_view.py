"""Lazy RGB-only overhead camera for the resident R1Pro."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from behavior_hl.embodied.evaluator_bridge.predeclared_camera_config import (
    R1PRO_CAMERA_SENSOR_KEYS,
)


OVERHEAD_CAMERA_HEIGHT_ABOVE_HEAD_M = 1.0
OVERHEAD_CAMERA_FORWARD_OFFSET_M = 0.12
OVERHEAD_CAMERA_IMAGE_SIZE_PX = 720
OVERHEAD_CAMERA_FOCAL_LENGTH_MM = 12.0
_OVERHEAD_CAMERA_COLLISION_HALF_EXTENT_M = 0.03
_OVERHEAD_CAMERA_HEIGHT_CANDIDATES_M = (
    OVERHEAD_CAMERA_HEIGHT_ABOVE_HEAD_M,
    0.75,
    0.5,
    0.25,
    0.1,
)
_OVERHEAD_CAMERA_ATTR = "_behavior_hl_action20_overhead_camera"
_OVERHEAD_CAMERA_PARENT_ORIENTATION = (
    0.0,
    0.0,
    -math.sqrt(0.5),
    math.sqrt(0.5),
)


def _xyz(value: Sequence[Any], *, name: str) -> tuple[float, float, float]:
    if len(value) < 3:
        raise RuntimeError(f"{name} position is incomplete")
    result = tuple(float(value[index]) for index in range(3))
    if not all(math.isfinite(number) for number in result):
        raise RuntimeError(f"{name} position is non-finite")
    return result  # type: ignore[return-value]


def _head_sensor(robot: Any) -> Any:
    sensors = getattr(robot, "sensors", None)
    if not isinstance(sensors, Mapping):
        raise RuntimeError("R1Pro sensors are unavailable")
    suffix = R1PRO_CAMERA_SENSOR_KEYS["head"]
    matches = [
        sensor
        for name, sensor in sensors.items()
        if name == suffix or str(name).endswith(f":{suffix}")
    ]
    if len(matches) != 1:
        raise RuntimeError("R1Pro head sensor is not uniquely resolved")
    return matches[0]


def _parent_pose(
    robot: Any,
    *,
    forward_offset_m: float = OVERHEAD_CAMERA_FORWARD_OFFSET_M,
    height_above_head_m: float = OVERHEAD_CAMERA_HEIGHT_ABOVE_HEAD_M,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    head_position, _ = _head_sensor(robot).get_position_orientation()
    robot_position, _ = robot.get_position_orientation()
    head = _xyz(head_position, name="head sensor")
    base = _xyz(robot_position, name="R1Pro")
    height = head[2] - base[2] + height_above_head_m
    if not math.isfinite(height) or height <= 0.0:
        raise RuntimeError("R1Pro overhead camera height is invalid")
    return (
        (forward_offset_m, 0.0, height),
        _OVERHEAD_CAMERA_PARENT_ORIENTATION,
    )


def _robot_link_paths(robot: Any) -> tuple[str, ...]:
    paths = getattr(robot, "link_prim_paths", ())
    paths = paths() if callable(paths) else paths
    if paths:
        return tuple(str(path) for path in paths)
    links = getattr(robot, "links", {}) or {}
    linked = tuple(
        str(getattr(link, "prim_path", ""))
        for link in links.values()
        if str(getattr(link, "prim_path", ""))
    )
    if linked:
        return linked
    path = str(getattr(robot, "prim_path", ""))
    return (path,) if path else ()


def _camera_position_is_clear(
    world_position: Sequence[Any],
    *,
    overlap_box: Callable[..., Any],
    robot_paths: Sequence[str],
) -> bool:
    external_hit = False

    def record_hit(hit: Any) -> bool:
        nonlocal external_hit
        for attribute in ("rigid_body", "collision"):
            path = str(getattr(hit, attribute, ""))
            if path and not any(
                path == robot_path or path.startswith(f"{robot_path}/")
                for robot_path in robot_paths
            ):
                external_hit = True
        return True

    overlap_box(
        halfExtent=[_OVERHEAD_CAMERA_COLLISION_HALF_EXTENT_M] * 3,
        pos=list(world_position),
        rot=[0.0, 0.0, 0.0, 1.0],
        reportFn=record_hit,
    )
    return not external_hit


def _place_camera_with_clearance(
    robot: Any,
    sensor: Any,
    *,
    overlap_box: Callable[..., Any],
) -> None:
    """Use the highest unoccupied camera point without moving scene objects."""

    robot_paths = _robot_link_paths(robot)
    for height in _OVERHEAD_CAMERA_HEIGHT_CANDIDATES_M:
        for forward in (OVERHEAD_CAMERA_FORWARD_OFFSET_M, 0.0):
            pose = _parent_pose(
                robot,
                forward_offset_m=forward,
                height_above_head_m=height,
            )
            sensor.set_position_orientation(*pose, frame="parent")
            world_position, _ = sensor.get_position_orientation()
            if _camera_position_is_clear(
                _xyz(world_position, name="overhead camera"),
                overlap_box=overlap_box,
                robot_paths=robot_paths,
            ):
                return


def ensure_r1pro_overhead_camera(
    robot: Any,
    *,
    sensor_factory: Callable[..., Any] | None = None,
    path_converter: Callable[[Any, str], str] | None = None,
) -> Any:
    """Create one base-attached RGB VisionSensor on first use, then reuse it."""

    existing = getattr(robot, _OVERHEAD_CAMERA_ATTR, None)
    if existing is not None:
        return existing
    if sensor_factory is None:
        from omnigibson.sensors import VisionSensor

        sensor_factory = VisionSensor
    if path_converter is None:
        from omnigibson.utils.usd_utils import (
            absolute_prim_path_to_scene_relative,
        )

        path_converter = absolute_prim_path_to_scene_relative
    scene = getattr(robot, "scene", None)
    base_link = getattr(robot, "base_footprint_link", None)
    base_path = getattr(base_link, "prim_path", None)
    robot_name = getattr(robot, "name", None)
    if scene is None or not isinstance(base_path, str) or not base_path:
        raise RuntimeError("R1Pro base link or scene is unavailable")
    if not isinstance(robot_name, str) or not robot_name:
        raise RuntimeError("R1Pro name is unavailable")
    sensor = sensor_factory(
        relative_prim_path=path_converter(
            scene,
            f"{base_path.rstrip('/')}/action20_overhead_camera",
        ),
        name=f"{robot_name}:action20_overhead_camera:Camera:0",
        modalities=("rgb",),
        image_height=OVERHEAD_CAMERA_IMAGE_SIZE_PX,
        image_width=OVERHEAD_CAMERA_IMAGE_SIZE_PX,
        focal_length=OVERHEAD_CAMERA_FOCAL_LENGTH_MM,
    )
    sensor.load(scene)
    sensor.set_position_orientation(*_parent_pose(robot), frame="parent")
    sensor.initialize()
    sensor.load_observation_space()
    setattr(robot, _OVERHEAD_CAMERA_ATTR, sensor)
    return sensor


def capture_r1pro_overhead_rgb(
    robot: Any,
    *,
    render: Callable[[], Any] | None = None,
    overlap_box: Callable[..., Any] | None = None,
) -> Any:
    """Capture only RGB; the parented sensor follows the live robot base."""

    if render is None:
        import omnigibson as og

        render = og.sim.render
    if overlap_box is None:
        import omnigibson as og

        overlap_box = og.sim.psqi.overlap_box
    sensor = ensure_r1pro_overhead_camera(robot)
    try:
        _place_camera_with_clearance(
            robot,
            sensor,
            overlap_box=overlap_box,
        )
    except Exception:
        sensor.set_position_orientation(*_parent_pose(robot), frame="parent")
    for _ in range(4):
        render()
    payload = sensor.get_obs()
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise RuntimeError("overhead VisionSensor observation is invalid")
    rgb = payload[0].get("rgb") if isinstance(payload[0], Mapping) else None
    shape = getattr(rgb, "shape", ())
    if len(shape) != 3 or int(shape[-1]) < 3:
        raise RuntimeError("overhead VisionSensor RGB frame is invalid")
    return rgb


__all__ = [
    "OVERHEAD_CAMERA_FORWARD_OFFSET_M",
    "OVERHEAD_CAMERA_HEIGHT_ABOVE_HEAD_M",
    "OVERHEAD_CAMERA_IMAGE_SIZE_PX",
    "capture_r1pro_overhead_rgb",
    "ensure_r1pro_overhead_camera",
]
