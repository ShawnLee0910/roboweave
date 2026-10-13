"""Private visible-part bindings for semantic open / close actions.

The public model only receives temporary C labels.  Simulator object, link,
and joint identities remain inside :class:`DrawerPartTargetToken` and are
never serialized.  This module intentionally has no OmniGibson imports so the
mesh-to-link mapping and endpoint logic can be tested without loading Kit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
)
from behavior_hl.embodied.evaluator_bridge.on_top_support_layers import (
    matrix_to_quaternion_xyzw,
    quaternion_angle_rad,
    quaternion_to_matrix_xyzw,
)
from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    object_aabb,
    object_pose,
)
from behavior_hl.embodied.runtime.public_action_feedback import (
    ARTICULATED_PART_BLOCKED,
    EXECUTION_UNVERIFIED,
)
from behavior_hl.embodied.runtime.action_integrity import ActionIntegrityError


PRIVATE_DRAWER_PART_VERSION = (
    "behavior2026.private-drawer-part/v8-zero-physics-atomic"
)
DRAWER_TRANSLATION_MAX_STEP_M = 0.05
REVOLUTE_MOVE_MAX_STEPS = 20
DRAWER_CONTENT_TOLERANCE_M = 0.005
DRAWER_FINAL_STABILITY_STEPS = 3
ARTICULATED_ENDPOINT_TOLERANCE_RATIO = 0.005


class DrawerPartBlockedError(RuntimeError):
    """Raised when an articulated part cannot move safely."""

    def __init__(
        self,
        report: Mapping[str, Any],
        *,
        blocked_category: str = ARTICULATED_PART_BLOCKED,
    ) -> None:
        super().__init__("articulated part motion is obstructed")
        self.report = dict(report)
        self.blocked_category = blocked_category


@dataclass(frozen=True, slots=True)
class ArticulatedPartTransactionBoundary:
    """One public action boundary after private physics-only motion."""

    simulator_steps: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.simulator_steps, bool)
            or not isinstance(self.simulator_steps, int)
            or self.simulator_steps <= 0
        ):
            raise ValueError("simulator_steps must be a positive integer")


def _scalar(value: Any) -> float:
    for method in ("detach", "cpu"):
        callback = getattr(value, method, None)
        if callable(callback):
            value = callback()
    callback = getattr(value, "item", None)
    if callable(callback):
        value = callback()
    return float(value)


def _vector3(value: Any) -> np.ndarray:
    for method in ("detach", "cpu"):
        callback = getattr(value, method, None)
        if callable(callback):
            value = callback()
    callback = getattr(value, "numpy", None)
    if callable(callback):
        value = callback()
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("expected a finite 3-vector")
    return result


def _visual_aabb(link: Any) -> tuple[np.ndarray, np.ndarray]:
    raw = getattr(link, "visual_aabb", None)
    raw = raw() if callable(raw) else raw
    if raw is None:
        raise RuntimeError("drawer link visual AABB is unavailable")
    lower, upper = (_vector3(item) for item in raw)
    if np.any(upper <= lower):
        raise RuntimeError("drawer link visual AABB is invalid")
    return lower, upper


def _strictly_inside(
    inner: tuple[np.ndarray, np.ndarray],
    outer: tuple[np.ndarray, np.ndarray],
) -> bool:
    inner_lower, inner_upper = inner
    outer_lower, outer_upper = outer
    tolerance = DRAWER_CONTENT_TOLERANCE_M
    return bool(
        np.all(inner_lower >= outer_lower - tolerance)
        and np.all(inner_upper <= outer_upper + tolerance)
    )


def drawer_fillable_aabbs(
    token: "DrawerPartTargetToken",
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return the official fillable volumes owned by one moving drawer."""

    link_name = str(getattr(token.link_token, "name", "")).rsplit(":", 1)[-1]
    if not link_name:
        raise RuntimeError("drawer link name is unavailable")
    needle = f"meta__{link_name}_fillable"
    links = getattr(token.target_token, "links", None)
    if not isinstance(links, Mapping):
        raise RuntimeError("drawer object links are unavailable")
    matches = [
        link
        for link in links.values()
        if bool(getattr(link, "is_meta_link", False))
        and "fillable" in str(getattr(link, "meta_link_type", "")).casefold()
        and needle in str(getattr(link, "name", ""))
    ]
    if not matches:
        raise RuntimeError("selected drawer has no official fillable volume")
    return tuple(
        _visual_aabb(link)
        for link in sorted(matches, key=lambda link: str(getattr(link, "name", "")))
    )


def _inside_any(
    inner: tuple[np.ndarray, np.ndarray],
    outers: Iterable[tuple[np.ndarray, np.ndarray]],
) -> bool:
    return any(_strictly_inside(inner, outer) for outer in outers)


def _drawer_contents(
    token: "DrawerPartTargetToken",
    scene_objects: Iterable[Any],
) -> tuple[Any, ...]:
    """Find movable objects intersecting the selected drawer compartment."""

    try:
        fillable_regions = drawer_fillable_aabbs(token)
    except RuntimeError:
        # Hinged doors and sliding panels have no fillable compartment of
        # their own. They still move independently, just without contents.
        return ()
    contents: list[Any] = []
    for obj in scene_objects:
        if obj is token.target_token:
            continue
        fixed = getattr(obj, "fixed_base", None)
        if fixed is None:
            fixed = getattr(obj, "_fixed_base", False)
        if bool(fixed) or not callable(getattr(obj, "set_position_orientation", None)):
            continue
        try:
            lower, upper = object_aabb(obj)
        except (RuntimeError, ValueError, TypeError):
            continue
        if any(
            np.all(
                np.minimum(upper, region_upper)
                - np.maximum(lower, region_lower)
                > DRAWER_CONTENT_TOLERANCE_M
            )
            for region_lower, region_upper in fillable_regions
        ):
            contents.append(obj)
    return tuple(contents)


def _is_prismatic(joint: Any) -> bool:
    value = getattr(joint, "joint_type", None)
    pieces = [str(value).lower()]
    name = getattr(value, "name", None)
    if name is not None:
        pieces.append(str(name).lower())
    return any("prismatic" in piece for piece in pieces)


def _is_revolute(joint: Any) -> bool:
    value = getattr(joint, "joint_type", None)
    pieces = [str(value).lower()]
    name = getattr(value, "name", None)
    if name is not None:
        pieces.append(str(name).lower())
    return any("revolute" in piece for piece in pieces)


def _open_state(obj: Any) -> Any | None:
    for state_type, state in getattr(obj, "states", {}).items():
        if getattr(state_type, "__name__", "") == "Open":
            return state
    return None


def _link_for_body1(obj: Any, body1: str) -> Any | None:
    for link in getattr(obj, "links", {}).values():
        if str(getattr(link, "prim_path", "")) == body1:
            return link
    return None


class DrawerPartTargetToken:
    """Opaque executor-only binding to one drawer joint and moving link."""

    __slots__ = (
        "_object",
        "_joint",
        "_link",
        "_joint_direction",
    )

    def __init__(
        self,
        *,
        object_token: Any,
        joint_token: Any,
        link_token: Any,
        joint_direction: int,
    ) -> None:
        if object_token is None or joint_token is None or link_token is None:
            raise ValueError("drawer part tokens must be live simulator values")
        if joint_direction not in {-1, 1}:
            raise ValueError("joint_direction must be -1 or 1")
        self._object = object_token
        self._joint = joint_token
        self._link = link_token
        self._joint_direction = joint_direction

    @property
    def target_token(self) -> Any:
        """Whole object used only for shared role and reach admission."""

        return self._object

    @property
    def joint_token(self) -> Any:
        return self._joint

    @property
    def link_token(self) -> Any:
        return self._link

    @property
    def joint_direction(self) -> int:
        return self._joint_direction

    @property
    def selector(self) -> str:
        return str(getattr(self._joint, "name", "drawer_joint"))

    @property
    def link_prim_path(self) -> str:
        return str(getattr(self._link, "prim_path", ""))

    def endpoints(self) -> tuple[float, float]:
        lower = _scalar(getattr(self._joint, "lower_limit"))
        upper = _scalar(getattr(self._joint, "upper_limit"))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("drawer joint limits are invalid")
        if self._joint_direction == 1:
            return upper, lower
        return lower, upper

    def current_position(self) -> float:
        state = self._joint.get_state()
        value = state[0] if isinstance(state, (tuple, list)) else state
        return _scalar(value)

    def is_open(self) -> bool:
        open_end, closed_end = self.endpoints()
        threshold = 0.95 * closed_end + 0.05 * open_end
        position = self.current_position()
        return position > threshold if open_end > threshold else position < threshold

    def __repr__(self) -> str:
        return "<DrawerPartTargetToken redacted>"

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("private drawer part tokens must not be serialized")


@dataclass(frozen=True, slots=True)
class VisibleDrawerPart:
    object_name: str
    visible_pixels: int
    bbox_xyxy: tuple[int, int, int, int]
    anchor_uv: tuple[int, int]
    mask_ids: tuple[int, ...]
    target_token: DrawerPartTargetToken = field(repr=False)


def drawer_part_tokens(obj: Any) -> tuple[DrawerPartTargetToken, ...]:
    """Return independently selectable joints for a multi-part openable."""

    state = _open_state(obj)
    info = getattr(state, "relevant_joints_info", None)
    if not isinstance(info, (tuple, list)) or len(info) != 3:
        return ()
    _, joints, directions = info
    if len(joints) != len(directions):
        return ()
    tokens: list[DrawerPartTargetToken] = []
    for joint, direction in zip(joints, directions):
        if not (_is_prismatic(joint) or _is_revolute(joint)):
            continue
        body1 = str(getattr(joint, "body1", ""))
        link = _link_for_body1(obj, body1)
        if link is None:
            continue
        tokens.append(
            DrawerPartTargetToken(
                object_token=obj,
                joint_token=joint,
                link_token=link,
                joint_direction=int(direction),
            )
        )
    # A single door/drawer continues using the established whole-object path.
    return tuple(tokens) if len(tokens) >= 2 else ()


def visible_drawer_parts(
    *,
    frame: PrivateInstanceSegmentationFrame,
    eligible_objects_by_name: Mapping[str, Any],
    operation: str,
    minimum_visible_pixels: int = 8,
    diagnostics: list | None = None,
) -> tuple[VisibleDrawerPart, ...]:
    """Resolve current-frame visual meshes to movable drawer links."""

    if operation not in {"open", "close"}:
        raise ValueError("drawer part operation must be open or close")
    if minimum_visible_pixels <= 0:
        raise ValueError("minimum_visible_pixels must be positive")
    part_image = frame.part_segmentation
    if part_image is None:
        if diagnostics is not None:
            diagnostics.extend(dict(object_name=name, reason='PART_SEGMENTATION_UNAVAILABLE')
                               for name in eligible_objects_by_name)
        return ()
    candidates: list[VisibleDrawerPart] = []
    for object_name, obj in eligible_objects_by_name.items():
        for token in drawer_part_tokens(obj):
            is_open = token.is_open()
            row = dict(object_name=object_name, joint=token.selector, link_prim_path=token.link_prim_path,
                       is_open=is_open, visible_pixels=None, mask_ids=[], reason='STATE_PRECONDITION_MISMATCH')
            if diagnostics is not None:
                diagnostics.append(row)
            if operation == "open" and is_open:
                continue
            if operation == "close" and not is_open:
                continue
            link_path = token.link_prim_path.rstrip("/")
            ids = tuple(
                sorted(
                    part_id
                    for part_id, prim_path in frame.part_id_to_prim_path.items()
                    if prim_path == link_path
                    or prim_path.startswith(link_path + "/")
                )
            )
            if not ids:
                row.update(reason='PART_SEGMENTATION_IDS_MISSING', visible_pixels=0)
                continue
            mask = np.isin(part_image, np.asarray(ids, dtype=np.int64))
            ys, xs = np.nonzero(mask)
            row.update(mask_ids=list(ids), visible_pixels=int(len(xs)), reason='INSUFFICIENT_PART_PIXELS')
            if len(xs) < minimum_visible_pixels:
                continue
            row['reason']='PUBLISHED'
            # Anchor the public point to a real visible part pixel.  The
            # arithmetic centroid can fall between pixels or in a hole in a
            # drawer handle, so choose the nearest pixel from the frozen mask.
            centre_x = float(xs.mean())
            centre_y = float(ys.mean())
            anchor_index = int(
                np.argmin((xs - centre_x) ** 2 + (ys - centre_y) ** 2)
            )
            candidates.append(
                VisibleDrawerPart(
                    object_name=object_name,
                    visible_pixels=int(len(xs)),
                    bbox_xyxy=(
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    ),
                    anchor_uv=(
                        int(xs[anchor_index]),
                        int(ys[anchor_index]),
                    ),
                    mask_ids=ids,
                    target_token=token,
                )
            )
    # Top-to-bottom then left-to-right is intuitive for banks of drawers.
    candidates.sort(
        key=lambda item: (
            (item.bbox_xyxy[1] + item.bbox_xyxy[3]) / 2.0,
            (item.bbox_xyxy[0] + item.bbox_xyxy[2]) / 2.0,
            item.object_name,
            item.target_token.selector,
        )
    )
    return tuple(candidates)


def _set_drawer_part_position(
    token: DrawerPartTargetToken,
    commanded: float,
) -> tuple[float, float, int]:
    """Move one drawer joint without changing any of its peers."""

    if not isinstance(token, DrawerPartTargetToken):
        raise TypeError("token must be DrawerPartTargetToken")
    obj = token.target_token
    peers = drawer_part_tokens(obj) or (token,)
    before = {peer.selector: peer.current_position() for peer in peers}
    open_end, closed_end = token.endpoints()
    tolerance = max(1e-5, abs(open_end - closed_end) * 1e-3)
    if not min(open_end, closed_end) - tolerance <= commanded <= max(
        open_end, closed_end
    ) + tolerance:
        raise ValueError("drawer joint command is outside its annotated limits")
    token.joint_token.set_pos(commanded)
    clear_states_cache = getattr(obj, "clear_states_cache", None)
    if callable(clear_states_cache):
        clear_states_cache()
    after = {peer.selector: peer.current_position() for peer in peers}
    target_after = after[token.selector]
    if abs(target_after - commanded) > tolerance:
        raise RuntimeError("selected drawer joint did not reach its command")
    changed_other = [
        name
        for name, position in before.items()
        if name != token.selector and abs(after[name] - position) > tolerance
    ]
    if changed_other:
        raise RuntimeError("an unselected drawer joint changed")
    return before[token.selector], target_after, max(0, len(peers) - 1)


def _pose_after_link_motion(
    pose: tuple[np.ndarray, np.ndarray],
    *,
    link_before: tuple[np.ndarray, np.ndarray],
    link_after: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Carry one payload through the selected link's rigid transform."""

    position, orientation = pose
    before_position, before_orientation = link_before
    after_position, after_orientation = link_after
    before_rotation = quaternion_to_matrix_xyzw(before_orientation)
    after_rotation = quaternion_to_matrix_xyzw(after_orientation)
    payload_rotation = quaternion_to_matrix_xyzw(orientation)
    relative_position = before_rotation.T @ (position - before_position)
    relative_rotation = before_rotation.T @ payload_rotation
    return (
        after_position + after_rotation @ relative_position,
        matrix_to_quaternion_xyzw(after_rotation @ relative_rotation),
    )


def _restore_articulated_step(
    token: DrawerPartTargetToken,
    *,
    joint_position: float,
    content_poses: Mapping[int, tuple[np.ndarray, np.ndarray]],
    contents: Iterable[Any],
) -> None:
    _set_drawer_part_position(token, joint_position)
    for obj in contents:
        position, orientation = content_poses[id(obj)]
        obj.set_position_orientation(position=position, orientation=orientation)
        keep_still = getattr(obj, "keep_still", None)
        if callable(keep_still):
            keep_still()


def move_drawer_part_incrementally(
    token: DrawerPartTargetToken,
    *,
    opened: bool,
    scene_objects: Iterable[Any],
    external_contact_pairs: Callable[[Iterable[Any]], frozenset[tuple[str, str]]]
    | None = None,
    physics_step_once: Callable[[], int] | None = None,
    zero_physics: bool = False,
    render_once: Callable[[], Any] | None = None,
):
    """Preserve verified partial progress; only faults roll back the action."""

    if not isinstance(zero_physics, bool):
        raise TypeError("zero_physics must be boolean")
    objects = tuple(scene_objects)
    if not zero_physics:
        return (yield from _move_drawer_part_incrementally(
            token, opened=opened, scene_objects=objects,
            external_contact_pairs=external_contact_pairs, physics_step_once=physics_step_once,
            zero_physics=zero_physics, render_once=render_once,
        ))
    if physics_step_once is not None:
        raise ValueError("zero-physics articulation cannot receive a physics step")
    if not callable(render_once):
        raise TypeError("zero-physics articulation requires render_once")
    peers = drawer_part_tokens(token.target_token) or (token,)
    start = token.current_position()
    joints = tuple((peer, peer.current_position()) for peer in peers)
    contents = _drawer_contents(token, objects)
    poses = tuple((obj, object_pose(obj)) for obj in (*contents, *(peer.link_token for peer in peers)))
    safe_joints, safe_poses = joints, poses

    def checkpoint_safe():
        nonlocal safe_joints, safe_poses
        safe_joints = tuple((peer, peer.current_position()) for peer in peers)
        safe_poses = tuple((obj, object_pose(obj)) for obj, _pose in poses)

    def restored(joint_snapshot, pose_snapshot):
        return all(abs(peer.current_position() - value) <= 1e-5 for peer, value in joint_snapshot) and all(
            np.linalg.norm(object_pose(obj)[0] - position) <= 1e-4
            and quaternion_angle_rad(object_pose(obj)[1], orientation) <= 1e-3
            for obj, (position, orientation) in pose_snapshot
        )

    try:
        return (yield from _move_drawer_part_incrementally(
            token, opened=opened, scene_objects=objects,
            external_contact_pairs=external_contact_pairs, zero_physics=True, render_once=render_once,
            checkpoint_safe=checkpoint_safe,
        ))
    except BaseException as exc:
        normal_block = isinstance(exc, DrawerPartBlockedError) and exc.blocked_category != EXECUTION_UNVERIFIED
        restore_joints, restore_poses = (safe_joints, safe_poses) if normal_block else (joints, poses)
        try:
            if not restored(restore_joints, restore_poses):
                for peer, value in restore_joints:
                    peer.joint_token.set_pos(value)
                for obj, (position, orientation) in restore_poses[:len(contents)]:
                    obj.set_position_orientation(position=position, orientation=orientation)
                clearer = getattr(token.target_token, "clear_states_cache", None)
                if callable(clearer):
                    clearer()
                render_once()
                if not restored(restore_joints, restore_poses):
                    raise RuntimeError("articulated state did not restore")
            if normal_block:
                desired_reached = token.is_open() is opened
                # Reporting only; success is determined by the Open predicate.
                changed = token.current_position() != start
                exc.report.update(
                    blocked=not desired_reached, requested_state_satisfied=desired_reached,
                    partial_progress_preserved=changed, task_state_unchanged=not changed,
                    recoverable_executor_noop=not changed, last_safe_state_verified=True,
                    rollback_applied=True, rollback_verified=True,
                    rollback_scope="last_safe_step", position_after=token.current_position(),
                )
                if desired_reached:
                    return exc.report
        except BaseException as restore_error:
            raise ActionIntegrityError(
                "Symbolic articulation rollback failed",
                private_report={"rollback_applied": False, "restore_error": str(restore_error)},
            ) from restore_error
        if isinstance(exc, DrawerPartBlockedError):
            if not normal_block:
                exc.report.update(rollback_applied=True, rollback_verified=True,
                                  rollback_scope="action_start", position_after=token.current_position())
            raise
        if not isinstance(exc, Exception):
            raise
        raise DrawerPartBlockedError(
            {"operation": "open" if opened else "close", "blocked": True,
             "rollback_applied": True, "rollback_verified": True,
             "position_before": start, "position_after": token.current_position(),
             "physics_steps_executed": 0, "error": f"{type(exc).__name__}: {exc}"},
            blocked_category=EXECUTION_UNVERIFIED,
        ) from exc


def _move_drawer_part_incrementally(
    token: DrawerPartTargetToken,
    *,
    opened: bool,
    scene_objects: Iterable[Any],
    external_contact_pairs: Callable[[Iterable[Any]], frozenset[tuple[str, str]]] | None = None,
    physics_step_once: Callable[[], int] | None = None,
    zero_physics: bool = False,
    render_once: Callable[[], Any] | None = None,
    checkpoint_safe: Callable[[], None] | None = None,
):
    """Move a drawer and its contents, stopping at the first obstruction."""

    start = token.current_position()
    open_end, closed_end = token.endpoints()
    destination = open_end if opened else closed_end
    operation = "open" if opened else "close"
    if physics_step_once is not None and not callable(physics_step_once):
        raise TypeError("physics_step_once must be callable")
    if not isinstance(zero_physics, bool):
        raise TypeError("zero_physics must be boolean")
    if zero_physics:
        if physics_step_once is not None:
            raise ValueError("zero-physics articulation cannot receive a physics step")
        if not callable(render_once):
            raise TypeError("zero-physics articulation requires render_once")
    distance = abs(destination - start)
    if _is_prismatic(token.joint_token):
        step_count = max(
            1,
            int(math.ceil(distance / DRAWER_TRANSLATION_MAX_STEP_M)),
        )
    elif _is_revolute(token.joint_token):
        full_range = abs(open_end - closed_end)
        step_count = max(
            1,
            min(
                REVOLUTE_MOVE_MAX_STEPS,
                int(math.ceil(distance / full_range * REVOLUTE_MOVE_MAX_STEPS)),
            ),
        )
    else:
        raise ValueError("selected articulated joint type is unsupported")
    positions = np.linspace(start, destination, step_count + 1)[1:]
    contents = _drawer_contents(token, scene_objects)
    final_containment_window: list[bool] = []
    final_endpoint_window: list[bool] = []
    endpoint_tolerance = max(
        1e-5,
        abs(open_end - closed_end) * ARTICULATED_ENDPOINT_TOLERANCE_RATIO,
    )
    pending_physics_steps = 0
    physics_steps_executed = 0
    render_refresh_count = 0

    def at_destination() -> bool:
        return abs(token.current_position() - destination) <= endpoint_tolerance

    def contents_contained() -> bool:
        return all(
            _inside_any(object_aabb(obj), drawer_fillable_aabbs(token))
            for obj in contents
        )

    def advance_physics(*, expose_boundary: bool = False):
        nonlocal pending_physics_steps, physics_steps_executed
        nonlocal render_refresh_count
        if zero_physics:
            assert render_once is not None
            render_once()
            render_refresh_count += 1
            return
        if physics_step_once is None:
            yield None
            return
        steps = physics_step_once()
        if isinstance(steps, bool) or not isinstance(steps, int) or steps != 1:
            raise RuntimeError("physics_step_once must advance exactly one step")
        pending_physics_steps += steps
        physics_steps_executed += steps
        if expose_boundary:
            boundary_steps = pending_physics_steps
            pending_physics_steps = 0
            yield ArticulatedPartTransactionBoundary(boundary_steps)

    def expose_transaction_boundary():
        nonlocal pending_physics_steps
        if physics_step_once is not None and pending_physics_steps:
            boundary_steps = pending_physics_steps
            pending_physics_steps = 0
            yield ArticulatedPartTransactionBoundary(boundary_steps)

    def restore_failure(
        *,
        joint_position: float,
        content_poses: Mapping[int, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        _restore_articulated_step(
            token,
            joint_position=joint_position,
            content_poses=content_poses,
            contents=contents,
        )

    def report(
        *,
        blocked: bool,
        executed: int,
        obstructing: tuple[Any, ...] = (),
        blocking_contacts: tuple[tuple[str, str], ...] = (),
    ):
        return {
            "schema_version": PRIVATE_DRAWER_PART_VERSION,
            "operation": operation,
            "selected_joint": token.selector,
            "selected_link": token.link_prim_path,
            "position_before": start,
            "position_commanded": destination,
            "position_after": token.current_position(),
            "move_steps_requested": step_count,
            "move_steps_executed": executed,
            "physics_steps_executed": physics_steps_executed,
            "render_refresh_count": render_refresh_count,
            "final_stability_steps_required": (
                0 if zero_physics else DRAWER_FINAL_STABILITY_STEPS
            ),
            "final_containment_window": tuple(final_containment_window),
            "final_endpoint_window": tuple(final_endpoint_window),
            "endpoint_tolerance": endpoint_tolerance,
            "content_count_private": len(contents),
            "obstructing_objects_private": tuple(
                str(getattr(obj, "name", "unnamed")) for obj in obstructing
            ),
            "blocking_contact_pairs_private": blocking_contacts,
            "blocked": blocked,
            "direct_state_modification": True,
            "kinematic_research": True,
            "official_challenge_eligible": False,
        }

    protruding = tuple(
        obj
        for obj in contents
        if not _inside_any(object_aabb(obj), drawer_fillable_aabbs(token))
    )
    if protruding:
        raise DrawerPartBlockedError(
            report(blocked=True, executed=0, obstructing=protruding)
        )

    final_rollback: tuple[
        float,
        dict[int, tuple[np.ndarray, np.ndarray]],
    ] | None = None
    for executed, commanded in enumerate(positions, start=1):
        joint_before = token.current_position()
        link_before = object_pose(token.link_token)
        content_poses = {
            id(obj): object_pose(obj)
            for obj in contents
        }
        try:
            contacts_before = (
                external_contact_pairs(contents)
                if external_contact_pairs is not None and not zero_physics
                else frozenset()
            )
        except Exception as exc:
            raise DrawerPartBlockedError(
                report(blocked=True, executed=executed - 1),
                blocked_category=EXECUTION_UNVERIFIED,
            ) from exc
        _set_drawer_part_position(token, float(commanded))
        # Isaac updates some articulated link transforms only after physics.
        # Read the real link displacement after that update, then carry the
        # contents by the same delta before checking containment.
        yield from advance_physics()
        link_after = object_pose(token.link_token)
        if external_contact_pairs is not None and not zero_physics:
            try:
                blocking_contacts = tuple(
                    sorted(external_contact_pairs(contents) - contacts_before)
                )
            except Exception as exc:
                restore_failure(
                    joint_position=joint_before,
                    content_poses=content_poses,
                )
                yield from advance_physics(expose_boundary=True)
                raise DrawerPartBlockedError(
                    report(blocked=True, executed=executed - 1),
                    blocked_category=EXECUTION_UNVERIFIED,
                ) from exc
            if blocking_contacts:
                restore_failure(
                    joint_position=joint_before,
                    content_poses=content_poses,
                )
                yield from advance_physics(expose_boundary=True)
                raise DrawerPartBlockedError(
                    report(
                        blocked=True,
                        executed=executed - 1,
                        blocking_contacts=blocking_contacts,
                    )
                )
        delta = link_after[0] - link_before[0]
        expected_positions: dict[int, np.ndarray] = {}
        obstructing = []
        for obj in contents:
            position, orientation = content_poses[id(obj)]
            drift = object_pose(obj)[0] - position
            distance = float(np.linalg.norm(delta))
            if distance:
                direction = delta / distance
                along = float(np.dot(drift, direction))
                transverse = drift - along * direction
                if (
                    np.linalg.norm(transverse) > DRAWER_CONTENT_TOLERANCE_M
                    or along < -DRAWER_CONTENT_TOLERANCE_M
                    or along > distance + DRAWER_CONTENT_TOLERANCE_M
                ):
                    obstructing.append(obj)
                    continue
            expected, expected_orientation = _pose_after_link_motion(
                (position, orientation),
                link_before=link_before,
                link_after=link_after,
            )
            obj.set_position_orientation(
                position=expected,
                orientation=expected_orientation,
            )
            keep_still = getattr(obj, "keep_still", None)
            if callable(keep_still):
                keep_still()
            expected_positions[id(obj)] = expected.copy()

        if zero_physics and external_contact_pairs is not None:
            # Query after the complete moving assembly has reached this step.
            # The outer zero-physics transaction restores any query/setter error.
            if contents:
                yield from advance_physics()
            try:
                blocking_contacts = tuple(sorted(external_contact_pairs(contents)))
            except Exception as exc:
                raise DrawerPartBlockedError(
                    report(blocked=True, executed=executed - 1),
                    blocked_category=EXECUTION_UNVERIFIED,
                ) from exc
            if blocking_contacts:
                raise DrawerPartBlockedError(report(
                    blocked=True, executed=executed - 1,
                    blocking_contacts=blocking_contacts,
                ))

        fillable_regions = drawer_fillable_aabbs(token) if contents else ()
        obstructing.extend(
            obj
            for obj in contents
            if obj not in obstructing
            and (
                not _inside_any(object_aabb(obj), fillable_regions)
                or np.linalg.norm(object_pose(obj)[0] - expected_positions[id(obj)])
                > DRAWER_CONTENT_TOLERANCE_M
            )
        )
        if obstructing:
            restore_failure(
                joint_position=joint_before,
                content_poses=content_poses,
            )
            yield from advance_physics(expose_boundary=True)
            raise DrawerPartBlockedError(
                report(
                    blocked=True,
                    executed=executed - 1,
                    obstructing=tuple(obstructing),
                )
            )
        if executed == step_count:
            final_rollback = (joint_before, content_poses)
        if checkpoint_safe is not None:
            checkpoint_safe()

    if not at_destination():
        if final_rollback is None:
            raise RuntimeError("final drawer rollback checkpoint is unavailable")
        joint_before, content_poses = final_rollback
        restore_failure(
            joint_position=joint_before,
            content_poses=content_poses,
        )
        yield from advance_physics(expose_boundary=True)
        raise DrawerPartBlockedError(
            report(blocked=True, executed=max(0, step_count - 1))
        )
    final_read_count = 1 if zero_physics else DRAWER_FINAL_STABILITY_STEPS
    for _ in range(final_read_count):
        if not zero_physics:
            yield from advance_physics()
        endpoint_reached = at_destination()
        contained = contents_contained()
        final_endpoint_window.append(endpoint_reached)
        final_containment_window.append(contained)
        if not endpoint_reached or not contained:
            if final_rollback is None:
                raise RuntimeError("final drawer rollback checkpoint is unavailable")
            obstructing = tuple(
                obj
                for obj in contents
                if not _inside_any(
                    object_aabb(obj), drawer_fillable_aabbs(token)
                )
            )
            joint_before, content_poses = final_rollback
            restore_failure(
                joint_position=joint_before,
                content_poses=content_poses,
            )
            yield from advance_physics(expose_boundary=True)
            raise DrawerPartBlockedError(
                report(
                    blocked=True,
                    executed=max(0, step_count - 1),
                    obstructing=obstructing,
                )
            )
    yield from expose_transaction_boundary()
    if (zero_physics or physics_step_once is not None) and (
        not at_destination() or not contents_contained()
    ):
        if final_rollback is None:
            raise RuntimeError("final drawer rollback checkpoint is unavailable")
        obstructing = tuple(
            obj
            for obj in contents
            if not _inside_any(
                object_aabb(obj), drawer_fillable_aabbs(token)
            )
        )
        joint_before, content_poses = final_rollback
        restore_failure(
            joint_position=joint_before,
            content_poses=content_poses,
        )
        yield from advance_physics(expose_boundary=True)
        raise DrawerPartBlockedError(
            report(
                blocked=True,
                executed=max(0, step_count - 1),
                obstructing=obstructing,
            )
        )
    return report(blocked=False, executed=step_count)


__all__ = [
    "PRIVATE_DRAWER_PART_VERSION",
    "ArticulatedPartTransactionBoundary",
    "DrawerPartBlockedError",
    "DrawerPartTargetToken",
    "VisibleDrawerPart",
    "drawer_fillable_aabbs",
    "drawer_part_tokens",
    "move_drawer_part_incrementally",
    "visible_drawer_parts",
]
