"""Live-only factory for BEHAVIOR 2026 official PLACE_INSIDE."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from behavior_hl.embodied.evaluator_bridge.bounded_official_grasp import (
    PlanarBasePose,
)
from behavior_hl.embodied.evaluator_bridge.bounded_official_place import (
    BoundedOfficialPlaceConfig,
    BoundedOfficialPlaceDriver,
)
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
    PrivateVisibilityConfig,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_grasp_binding import (
    PrivateGraspArtifactSink,
)
from behavior_hl.embodied.evaluator_bridge.visible_official_place_backend import (
    VisibleOfficialPlaceBackendAdapter,
)
from behavior_hl.embodied.runtime.plugins.grasp import (
    PRIVATE_GRASP_AUDIT_SERVICE,
)
from behavior_hl.embodied.runtime.plugins.visible_official_place import (
    VISIBLE_OFFICIAL_PLACE_BACKEND_SERVICE,
)


class R1ProVisibleOfficialPlaceFactoryError(RuntimeError):
    """Raised before mutation when the live adapter cannot be assembled."""


@dataclass(frozen=True, slots=True)
class R1ProVisibleOfficialPlaceServices:
    services: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "services", MappingProxyType(dict(self.services)))


def private_picking_up_toys_container_objects(
    evaluator: Any,
) -> Mapping[str, Any]:
    """Return exactly the private toy-box handle keyed by segmentation name."""

    try:
        scope = evaluator.env.task.object_scope
    except Exception as exc:
        raise R1ProVisibleOfficialPlaceFactoryError(
            "picking_up_toys object_scope is unavailable"
        ) from exc
    if not isinstance(scope, Mapping):
        raise R1ProVisibleOfficialPlaceFactoryError(
            "picking_up_toys object_scope must be a mapping"
        )
    matches = [
        entity
        for scope_name, entity in scope.items()
        if isinstance(scope_name, str)
        and scope_name.startswith("toy_box.n.01_")
        and entity is not None
    ]
    if len(matches) != 1:
        raise R1ProVisibleOfficialPlaceFactoryError(
            f"expected exactly one toy box; found {len(matches)}"
        )
    container = matches[0]
    name = getattr(container, "name", None)
    if not isinstance(name, str) or not name:
        raise R1ProVisibleOfficialPlaceFactoryError(
            "toy box has no simulator name"
        )
    return MappingProxyType({name: container})


def build_r1pro_visible_official_place_services(
    *,
    starter_by_hand: Mapping[str, Any],
    private_frame_provider: Callable[[], PrivateInstanceSegmentationFrame],
    eligible_containers_provider: Callable[[], Mapping[str, Any]],
    command_step: Callable[[Any, str], int],
    base_pose_provider: Callable[[], PlanarBasePose],
    held_object_provider_by_hand: Mapping[str, Callable[[], Any]],
    hold_safe_by_hand: Mapping[str, Callable[[], Any]],
    inside_checker: Callable[[Any, Any], bool],
    private_audit_writer: Callable[[Mapping[str, Any]], None],
    place_config: BoundedOfficialPlaceConfig | None = None,
    visibility_config: PrivateVisibilityConfig | None = None,
) -> R1ProVisibleOfficialPlaceServices:
    starters = dict(starter_by_hand)
    held = dict(held_object_provider_by_hand)
    holds = dict(hold_safe_by_hand)
    if not starters or set(starters) != set(held) or set(starters) != set(holds):
        raise R1ProVisibleOfficialPlaceFactoryError(
            "starter, held-provider, and hold maps must share hand keys"
        )
    if not set(starters).issubset({"left", "right"}):
        raise R1ProVisibleOfficialPlaceFactoryError(
            "only left/right hand keys are supported"
        )
    for callback, name in (
        (private_frame_provider, "private_frame_provider"),
        (eligible_containers_provider, "eligible_containers_provider"),
        (command_step, "command_step"),
        (base_pose_provider, "base_pose_provider"),
        (inside_checker, "inside_checker"),
        (private_audit_writer, "private_audit_writer"),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")
    for mapping, label in ((held, "held"), (holds, "hold")):
        if any(not callable(value) for value in mapping.values()):
            raise TypeError(f"all {label} providers must be callable")

    active_config = place_config or BoundedOfficialPlaceConfig()
    drivers = {}
    for hand, starter in starters.items():
        if not callable(getattr(starter, "apply_ref", None)):
            raise R1ProVisibleOfficialPlaceFactoryError(
                f"Starter for {hand!r} lacks apply_ref()"
            )

        def generator_factory(
            target: Any,
            attempts: int,
            *,
            active_starter: Any = starter,
        ):
            from omnigibson.action_primitives.starter_semantic_action_primitives import (
                StarterSemanticActionPrimitiveSet,
            )

            return active_starter.apply_ref(
                StarterSemanticActionPrimitiveSet.PLACE_INSIDE,
                target,
                attempts=attempts,
            )

        drivers[hand] = BoundedOfficialPlaceDriver(
            generator_factory=generator_factory,
            command_step=command_step,
            base_pose_provider=base_pose_provider,
            held_object_provider=held[hand],
            inside_checker=inside_checker,
            hold_safe=holds[hand],
            config=active_config,
        )
    backend = VisibleOfficialPlaceBackendAdapter(
        private_frame_provider=private_frame_provider,
        eligible_containers_provider=eligible_containers_provider,
        driver_by_hand=drivers,
        visibility_config=visibility_config,
    )
    return R1ProVisibleOfficialPlaceServices(
        services={
            VISIBLE_OFFICIAL_PLACE_BACKEND_SERVICE: backend,
            PRIVATE_GRASP_AUDIT_SERVICE: PrivateGraspArtifactSink(
                private_audit_writer
            ),
        }
    )


def private_inside_checker(payload: Any, container: Any) -> bool:
    """Read the canonical live Inside predicate without serializing it."""

    try:
        from omnigibson import object_states

        return bool(payload.states[object_states.Inside].get_value(container))
    except Exception as exc:
        raise R1ProVisibleOfficialPlaceFactoryError(
            "failed to evaluate the private Inside predicate"
        ) from exc
