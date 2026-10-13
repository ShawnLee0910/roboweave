"""Live-only factory for the BEHAVIOR 2026 official GRASP adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from behavior_hl.embodied.evaluator_bridge.bounded_official_grasp import (
    BoundedOfficialGraspConfig,
    BoundedOfficialGraspDriver,
    PlanarBasePose,
)
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
    PrivateVisibilityConfig,
)
from behavior_hl.embodied.evaluator_bridge.visible_official_grasp_backend import (
    VisibleOfficialGraspBackendAdapter,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_grasp_binding import (
    PrivateGraspArtifactSink,
)
from behavior_hl.embodied.runtime.plugins.grasp import (
    PRIVATE_GRASP_AUDIT_SERVICE,
)
from behavior_hl.embodied.runtime.plugins.visible_official_grasp import (
    VISIBLE_OFFICIAL_GRASP_BACKEND_SERVICE,
)


class R1ProVisibleOfficialGraspFactoryError(RuntimeError):
    """Raised before mutation when the live adapter cannot be assembled."""


@dataclass(frozen=True, slots=True)
class R1ProVisibleOfficialGraspServices:
    services: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "services", MappingProxyType(dict(self.services)))


PICKING_UP_TOYS_GRASPABLE_SYNSETS = frozenset(
    {"board_game.n.01", "jigsaw_puzzle.n.01", "tennis_ball.n.01"}
)
PICKING_UP_TOYS_CONTAINER_SYNSETS = frozenset({"toy_box.n.01"})
PICKING_UP_TOYS_PUBLIC_CATEGORY_BY_SYNSET = MappingProxyType(
    {
        "board_game.n.01": "board_game",
        "jigsaw_puzzle.n.01": "jigsaw_puzzle",
        "tennis_ball.n.01": "tennis_ball",
    }
)


def private_picking_up_toys_eligible_objects(
    evaluator: Any,
) -> Mapping[str, Any]:
    """Map only the six BDDL toy leaves to their segmentation names.

    The returned simulator handles are private executor inputs.  Furniture,
    the toy box, floor, and agent are deliberately excluded even if visible.
    """

    try:
        scope = evaluator.env.task.object_scope
    except Exception as exc:
        raise R1ProVisibleOfficialGraspFactoryError(
            "picking_up_toys object_scope is unavailable"
        ) from exc
    if not isinstance(scope, Mapping):
        raise R1ProVisibleOfficialGraspFactoryError(
            "picking_up_toys object_scope must be a mapping"
        )
    result: dict[str, Any] = {}
    for scope_name, entity in scope.items():
        if entity is None or not isinstance(scope_name, str):
            continue
        synset, separator, instance = scope_name.rpartition("_")
        if (
            not separator
            or not instance.isdigit()
            or synset not in PICKING_UP_TOYS_GRASPABLE_SYNSETS
        ):
            continue
        entity_name = getattr(entity, "name", None)
        if not isinstance(entity_name, str) or not entity_name:
            raise R1ProVisibleOfficialGraspFactoryError(
                f"eligible object {scope_name!r} has no simulator name"
            )
        if entity_name in result and result[entity_name] is not entity:
            raise R1ProVisibleOfficialGraspFactoryError(
                f"duplicate simulator object name {entity_name!r}"
            )
        result[entity_name] = entity
    if len(result) != 6:
        raise R1ProVisibleOfficialGraspFactoryError(
            f"expected exactly six graspable toys; found {len(result)}"
        )
    return MappingProxyType(result)


def private_picking_up_toys_marker_objects(
    evaluator: Any,
) -> Mapping[str, Any]:
    """Return the six toy leaves plus the toy box for public C-labeling."""

    try:
        scope = evaluator.env.task.object_scope
    except Exception as exc:
        raise R1ProVisibleOfficialGraspFactoryError(
            "picking_up_toys object_scope is unavailable"
        ) from exc
    admitted = PICKING_UP_TOYS_GRASPABLE_SYNSETS.union(
        PICKING_UP_TOYS_CONTAINER_SYNSETS
    )
    result: dict[str, Any] = {}
    for scope_name, entity in scope.items():
        if entity is None or not isinstance(scope_name, str):
            continue
        synset, separator, instance = scope_name.rpartition("_")
        if not separator or not instance.isdigit() or synset not in admitted:
            continue
        name = getattr(entity, "name", None)
        if not isinstance(name, str) or not name:
            raise R1ProVisibleOfficialGraspFactoryError(
                f"marker object {scope_name!r} has no simulator name"
            )
        result[name] = entity
    if len(result) != 7:
        raise R1ProVisibleOfficialGraspFactoryError(
            f"expected six toys and one toy box; found {len(result)}"
        )
    return MappingProxyType(result)


def private_picking_up_toys_marker_roles(
    evaluator: Any,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return private simulator names admitted for grasp and placement."""

    all_objects = private_picking_up_toys_marker_objects(evaluator)
    graspable = private_picking_up_toys_eligible_objects(evaluator)
    container_names = frozenset(set(all_objects).difference(graspable))
    if len(container_names) != 1:
        raise R1ProVisibleOfficialGraspFactoryError(
            "expected exactly one marker-visible toy box"
        )
    return frozenset(graspable), container_names


def private_picking_up_toys_eligible_objects_by_category(
    evaluator: Any,
) -> Mapping[str, Mapping[str, Any]]:
    """Return the same six handles grouped by public visual category."""

    try:
        scope = evaluator.env.task.object_scope
    except Exception as exc:
        raise R1ProVisibleOfficialGraspFactoryError(
            "picking_up_toys object_scope is unavailable"
        ) from exc
    if not isinstance(scope, Mapping):
        raise R1ProVisibleOfficialGraspFactoryError(
            "picking_up_toys object_scope must be a mapping"
        )
    result: dict[str, dict[str, Any]] = {
        category: {}
        for category in PICKING_UP_TOYS_PUBLIC_CATEGORY_BY_SYNSET.values()
    }
    for scope_name, entity in scope.items():
        if entity is None or not isinstance(scope_name, str):
            continue
        synset, separator, instance = scope_name.rpartition("_")
        category = PICKING_UP_TOYS_PUBLIC_CATEGORY_BY_SYNSET.get(synset)
        if not separator or not instance.isdigit() or category is None:
            continue
        entity_name = getattr(entity, "name", None)
        if not isinstance(entity_name, str) or not entity_name:
            raise R1ProVisibleOfficialGraspFactoryError(
                f"eligible object {scope_name!r} has no simulator name"
            )
        category_objects = result[category]
        if (
            entity_name in category_objects
            and category_objects[entity_name] is not entity
        ):
            raise R1ProVisibleOfficialGraspFactoryError(
                f"duplicate simulator object name {entity_name!r}"
            )
        category_objects[entity_name] = entity
    if sum(len(items) for items in result.values()) != 6:
        raise R1ProVisibleOfficialGraspFactoryError(
            "expected exactly six categorized graspable toys"
        )
    return MappingProxyType(
        {
            category: MappingProxyType(dict(items))
            for category, items in result.items()
        }
    )


def build_r1pro_visible_official_grasp_services(
    *,
    starter_by_hand: Mapping[str, Any],
    private_frame_provider: Callable[[], PrivateInstanceSegmentationFrame],
    eligible_objects_provider: Callable[[], Mapping[str, Any]],
    eligible_objects_by_category_provider: (
        Callable[[], Mapping[str, Mapping[str, Any]]] | None
    ) = None,
    command_step: Callable[[Any, str], int],
    base_pose_provider: Callable[[], PlanarBasePose],
    held_object_provider_by_hand: Mapping[str, Callable[[], Any]],
    hold_safe_by_hand: Mapping[str, Callable[[], Any]],
    private_audit_writer: Callable[[Mapping[str, Any]], None],
    grasp_config: BoundedOfficialGraspConfig | None = None,
    visibility_config: PrivateVisibilityConfig | None = None,
) -> R1ProVisibleOfficialGraspServices:
    """Build services without importing OmniGibson until GRASP is invoked."""

    starters = dict(starter_by_hand)
    held = dict(held_object_provider_by_hand)
    holds = dict(hold_safe_by_hand)
    if not starters or set(starters) != set(held) or set(starters) != set(holds):
        raise R1ProVisibleOfficialGraspFactoryError(
            "starter, held-provider, and hold maps must share hand keys"
        )
    if not set(starters).issubset({"left", "right"}):
        raise R1ProVisibleOfficialGraspFactoryError(
            "only left/right hand keys are supported"
        )
    for callback, name in (
        (private_frame_provider, "private_frame_provider"),
        (eligible_objects_provider, "eligible_objects_provider"),
        (command_step, "command_step"),
        (base_pose_provider, "base_pose_provider"),
        (private_audit_writer, "private_audit_writer"),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")
    for mapping, label in ((held, "held"), (holds, "hold")):
        if any(not callable(value) for value in mapping.values()):
            raise TypeError(f"all {label} providers must be callable")

    active_config = grasp_config or BoundedOfficialGraspConfig()
    drivers = {}
    for hand, starter in starters.items():
        if not callable(getattr(starter, "apply_ref", None)):
            raise R1ProVisibleOfficialGraspFactoryError(
                f"Starter for {hand!r} lacks apply_ref()"
            )

        def generator_factory(
            target: Any,
            attempts: int,
            *,
            active_starter: Any = starter,
        ):
            # Lazy import keeps pure tests and policy processes Isaac-free.
            from omnigibson.action_primitives.starter_semantic_action_primitives import (
                StarterSemanticActionPrimitiveSet,
            )

            return active_starter.apply_ref(
                StarterSemanticActionPrimitiveSet.GRASP,
                target,
                attempts=attempts,
            )

        drivers[hand] = BoundedOfficialGraspDriver(
            generator_factory=generator_factory,
            command_step=command_step,
            base_pose_provider=base_pose_provider,
            held_object_provider=held[hand],
            hold_safe=holds[hand],
            config=active_config,
        )
    backend = VisibleOfficialGraspBackendAdapter(
        private_frame_provider=private_frame_provider,
        eligible_objects_provider=eligible_objects_provider,
        eligible_objects_by_category_provider=(
            eligible_objects_by_category_provider
        ),
        driver_by_hand=drivers,
        visibility_config=visibility_config,
    )
    audit = PrivateGraspArtifactSink(private_audit_writer)
    return R1ProVisibleOfficialGraspServices(
        services={
            VISIBLE_OFFICIAL_GRASP_BACKEND_SERVICE: backend,
            PRIVATE_GRASP_AUDIT_SERVICE: audit,
        }
    )


def planar_base_pose_from_robot(robot: Any) -> PlanarBasePose:
    """Read R1Pro's current world pose without changing simulator state."""

    try:
        position, quaternion = robot.get_position_orientation()
        x, y = float(position[0]), float(position[1])
        qx, qy, qz, qw = (float(value) for value in quaternion)
    except Exception as exc:
        raise R1ProVisibleOfficialGraspFactoryError(
            "failed to read R1Pro world pose"
        ) from exc
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return PlanarBasePose(x=x, y=y, yaw_rad=yaw)
