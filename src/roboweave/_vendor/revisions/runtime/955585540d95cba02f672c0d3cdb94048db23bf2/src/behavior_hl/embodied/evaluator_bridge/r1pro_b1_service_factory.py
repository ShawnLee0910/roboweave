"""Action-aware resident service composition for the R1Pro B1 profile.

The factory owns no simulator capability.  Six versioned component builders
are injected explicitly and receive the already-live evaluator/control
objects.  Controller selection is based only on a public carry-state tracker;
the factory never queries held-object, task, predicate, or world-pose truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from behavior_hl.embodied.evaluator_bridge.r1pro_actuator_binding import (
    R1PRO_ACTUATOR_BINDING_VERSION,
    R1PRO_LOCAL_BASE_BINDING_VERSION,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_grasp_binding import (
    R1PRO_GRASP_DRIVER_VERSION,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_place_services import (
    R1PRO_PLACE_SERVICES_VERSION,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_resident_environment import (
    R1PRO_RESIDENT_SERVICE_FACTORY_VERSION,
)
from behavior_hl.embodied.evaluator_bridge.r1pro_torso_release_binding import (
    R1PRO_TORSO_RELEASE_BINDING_VERSION,
)
from behavior_hl.embodied.policy.protocol import (
    ActionEnvelope,
    MarkerPlaceInsideArgs,
    MarkerTargetArgs,
    MarkerTargetOrRequestArgs,
    NoArgs,
    PointActionArgs,
    ReleaseArgs,
)
from behavior_hl.embodied.runtime.dispatch import ActionPluginResult
from behavior_hl.embodied.runtime.plugins.local_base import (
    PUBLIC_GRASP_PROXY_SERVICE as LOCAL_BASE_PUBLIC_GRASP_PROXY_SERVICE,
)
from behavior_hl.embodied.runtime.plugins.torso import (
    PUBLIC_GRASP_PROXY_SERVICE as TORSO_PUBLIC_GRASP_PROXY_SERVICE,
)


R1PRO_B1_SERVICE_FACTORY_IMPLEMENTATION_VERSION = (
    "behavior2026.r1pro-b1-composite-services/v5-action20-floor"
)

GRASP_COMPONENT = "grasp"
BASE_COMPONENT = "base"
CARRY_COMPONENT = "carry"
TORSO_COMPONENT = "torso"
PLACE_COMPONENT = "place"
RELEASE_COMPONENT = "release"

R1PRO_B1_COMPONENT_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        GRASP_COMPONENT: R1PRO_GRASP_DRIVER_VERSION,
        BASE_COMPONENT: R1PRO_LOCAL_BASE_BINDING_VERSION,
        CARRY_COMPONENT: R1PRO_ACTUATOR_BINDING_VERSION,
        TORSO_COMPONENT: R1PRO_TORSO_RELEASE_BINDING_VERSION,
        PLACE_COMPONENT: R1PRO_PLACE_SERVICES_VERSION,
        RELEASE_COMPONENT: R1PRO_TORSO_RELEASE_BINDING_VERSION,
    }
)

_BASE_ACTIONS = frozenset(
    {
        "move_backward",
        "move_forward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
    }
)
_LOOK_ACTIONS = frozenset(
    {"look_up", "look_center", "look_down", "move_up", "move_down"}
)
_KNOWN_ACTIONS = frozenset(
    {
        "grasp_at",
        "grasp",
        "mark_visible_objects",
        "place_inside",
        "place_inside_at",
        "release",
        "place_on_floor",
        "pick",
        "put_back",
        "open",
        "close",
        "place_on_top",
        "place_next_to",
        "place_under",
        "place_touching",
        "toggle_on",
        "toggle_off",
        "wait",
        "stop",
        *_BASE_ACTIONS,
        *_LOOK_ACTIONS,
    }
)
_CONFIDENCE_NOT_LIKELY = "not_likely"
_CONFIDENCE_LIKELY = "likely"
_CONFIDENCE_UNKNOWN = "unknown"


class R1ProB1ServiceFactoryError(RuntimeError):
    """Raised before a malformed resident service set can reach a plugin."""


class VersionedR1ProB1ComponentBuilder(Protocol):
    """One explicitly injected, simulator-specific service builder."""

    api_contract_version: str

    def build(
        self,
        *,
        action_index: int,
        action: ActionEnvelope,
        evaluator: Any,
        control: Any,
        public_snapshot_provider: Callable[[], Mapping[str, Any]],
        step_once: Callable[[], int],
        physics_step_once: Callable[[], int] | None = None,
        mark_boundary_already_checkpointed: Callable[[], None] | None = None,
        public_carry_snapshot: Callable[[], Mapping[str, Any]],
        public_grasp_proxy_source: "PublicCarryProxySource",
        active_hand: str | None,
    ) -> Mapping[str, Any]:
        """Return only this component's services."""


@dataclass(frozen=True, slots=True)
class R1ProB1ServiceFactoryPreflight:
    """Read-only component and public-state admission evidence."""

    version: str
    resident_api_contract_version: str
    available: bool
    missing_capabilities: tuple[str, ...]
    component_versions: Mapping[str, str]
    public_carry_snapshot: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "missing_capabilities",
            tuple(self.missing_capabilities),
        )
        object.__setattr__(
            self,
            "component_versions",
            MappingProxyType(dict(self.component_versions)),
        )
        object.__setattr__(
            self,
            "public_carry_snapshot",
            MappingProxyType(dict(self.public_carry_snapshot)),
        )

    def require_available(self) -> None:
        if not self.available:
            raise R1ProB1ServiceFactoryError(
                "R1Pro B1 composite service preflight failed: "
                f"{list(self.missing_capabilities)}"
            )


class PublicCarryProxySource:
    """One public tracker view usable by both adaptive plugin contracts."""

    def __init__(self, factory: "R1ProB1ServiceFactory") -> None:
        self._factory = factory

    def __call__(self) -> bool:
        """Satisfy the local-base callable-bool selector contract."""

        return self.grasp_proxy_likely()

    def grasp_proxy_likely(self) -> bool:
        """Satisfy the torso public-proxy source contract."""

        value = self._factory.public_carry_snapshot()[
            "grasp_proxy_likely"
        ]
        if not isinstance(value, bool):
            raise R1ProB1ServiceFactoryError(
                "public grasp proxy must be boolean"
            )
        return value

    def carry_mode(self) -> str:
        """Return the executor-only controller carry mode, never identity."""

        resolver = getattr(self._factory, "controller_carry_mode", None)
        mode = (
            resolver()
            if callable(resolver)
            else "physical"
            if self.grasp_proxy_likely()
            else "empty"
        )
        if mode not in {"empty", "physical", "logical_inventory"}:
            raise R1ProB1ServiceFactoryError(
                "controller carry mode is invalid"
            )
        return mode


class R1ProB1ServiceFactory:
    """Compose the minimum safe service set for one parsed B1 action."""

    api_contract_version = R1PRO_RESIDENT_SERVICE_FACTORY_VERSION
    implementation_version = R1PRO_B1_SERVICE_FACTORY_IMPLEMENTATION_VERSION

    def __init__(
        self,
        *,
        public_carry_state_source: Any,
        grasp_builder: VersionedR1ProB1ComponentBuilder,
        base_builder: VersionedR1ProB1ComponentBuilder,
        carry_builder: VersionedR1ProB1ComponentBuilder,
        torso_builder: VersionedR1ProB1ComponentBuilder,
        place_builder: VersionedR1ProB1ComponentBuilder,
        release_builder: VersionedR1ProB1ComponentBuilder,
        component_versions: Mapping[str, str] | None = None,
        wait_requires_actuator: bool = True,
    ) -> None:
        if not isinstance(wait_requires_actuator, bool):
            raise TypeError("wait_requires_actuator must be boolean")
        self._public_carry_state_source = public_carry_state_source
        self._wait_requires_actuator = wait_requires_actuator
        self._builders = MappingProxyType(
            {
                GRASP_COMPONENT: grasp_builder,
                BASE_COMPONENT: base_builder,
                CARRY_COMPONENT: carry_builder,
                TORSO_COMPONENT: torso_builder,
                PLACE_COMPONENT: place_builder,
                RELEASE_COMPONENT: release_builder,
            }
        )
        versions = dict(component_versions or R1PRO_B1_COMPONENT_VERSIONS)
        if set(versions) != set(self._builders) or any(
            not isinstance(value, str) or not value
            for value in versions.values()
        ):
            raise R1ProB1ServiceFactoryError(
                "component_versions must cover exactly all six components"
            )
        self._component_versions = MappingProxyType(versions)
        self._public_proxy_source = PublicCarryProxySource(self)
        self.capability_report = self.preflight()
        self.capability_report.require_available()

    def preflight(self) -> R1ProB1ServiceFactoryPreflight:
        """Inspect versions and public carry state without building or stepping."""

        missing: list[str] = []
        versions: dict[str, str] = {}
        for name, builder in self._builders.items():
            version = getattr(builder, "api_contract_version", None)
            versions[name] = version if isinstance(version, str) else ""
            if version != self._component_versions[name]:
                missing.append(f"{name}.api_contract_version")
            if not callable(getattr(builder, "build", None)):
                missing.append(f"{name}.build")

        source = self._public_carry_state_source
        for method in ("snapshot", "update"):
            if not callable(getattr(source, method, None)):
                missing.append(f"public_carry_state_source.{method}")
        required_implicit_hand = getattr(
            self._builders[GRASP_COMPONENT],
            "implicit_object_grasp_hand",
            None,
        )
        if required_implicit_hand is not None:
            tracker_implicit_hand = getattr(
                source,
                "implicit_grasp_hand",
                None,
            )
            if tracker_implicit_hand != required_implicit_hand:
                missing.append(
                    "public_carry_state_source.implicit_grasp_hand"
                )

        public_snapshot: Mapping[str, Any]
        try:
            public_snapshot = self.public_carry_snapshot()
        except Exception as exc:
            missing.append(
                "public_carry_state_source.valid_snapshot:"
                f"{type(exc).__name__}"
            )
            public_snapshot = MappingProxyType(
                {
                    "grasp_proxy_likely": False,
                    "active_hand": None,
                }
            )
        return R1ProB1ServiceFactoryPreflight(
            version=self.implementation_version,
            resident_api_contract_version=self.api_contract_version,
            available=not missing,
            missing_capabilities=tuple(missing),
            component_versions=versions,
            public_carry_snapshot=public_snapshot,
        )

    def public_carry_snapshot(self) -> Mapping[str, Any]:
        """Return exactly the two controller-safe public carry fields."""

        snapshot_method = getattr(
            self._public_carry_state_source,
            "snapshot",
            None,
        )
        if not callable(snapshot_method):
            raise R1ProB1ServiceFactoryError(
                "public carry-state source lacks snapshot"
            )
        raw = snapshot_method()
        if not isinstance(raw, Mapping):
            raise R1ProB1ServiceFactoryError(
                "public carry-state snapshot must be a mapping"
            )
        confidence = raw.get("confidence")
        likely = raw.get("likely")
        active_hand = raw.get("active_hand")
        if confidence == _CONFIDENCE_UNKNOWN:
            raise R1ProB1ServiceFactoryError(
                "public carry-state confidence is unknown"
            )
        if confidence not in {
            _CONFIDENCE_NOT_LIKELY,
            _CONFIDENCE_LIKELY,
        }:
            raise R1ProB1ServiceFactoryError(
                "public carry-state confidence is invalid"
            )
        if not isinstance(likely, bool):
            raise R1ProB1ServiceFactoryError(
                "public carry-state likely must be boolean"
            )
        if active_hand not in {None, "left", "right"}:
            raise R1ProB1ServiceFactoryError(
                "public carry-state active_hand is invalid"
            )
        if confidence == _CONFIDENCE_LIKELY:
            if not likely or active_hand not in {"left", "right"}:
                raise R1ProB1ServiceFactoryError(
                    "likely carry state requires one active hand"
                )
        elif likely or active_hand is not None:
            raise R1ProB1ServiceFactoryError(
                "not-likely carry state cannot retain an active hand"
            )
        return MappingProxyType(
            {
                "grasp_proxy_likely": likely,
                "active_hand": active_hand,
            }
        )

    def record_public_action_result(
        self,
        action: ActionEnvelope,
        result: ActionPluginResult,
    ) -> None:
        """Delegate one public result, rejecting ambiguous carry state."""

        if not isinstance(action, ActionEnvelope):
            raise TypeError("action must be ActionEnvelope")
        if not isinstance(result, ActionPluginResult):
            raise TypeError("result must be ActionPluginResult")
        update = getattr(self._public_carry_state_source, "update", None)
        if not callable(update):
            raise R1ProB1ServiceFactoryError(
                "public carry-state source lacks update"
            )
        updated = update(action, result)
        if not isinstance(updated, Mapping):
            raise R1ProB1ServiceFactoryError(
                "public carry-state update must return a mapping"
            )
        # The tracker is deliberately left poisoned if it transitioned to
        # unknown; this call and every future service build fail closed.
        self.public_carry_snapshot()

    def build(
        self,
        *,
        action_index: int,
        action: ActionEnvelope,
        evaluator: Any,
        control: Any,
        public_snapshot_provider: Callable[[], Mapping[str, Any]],
        step_once: Callable[[], int],
        physics_step_once: Callable[[], int] | None = None,
        mark_boundary_already_checkpointed: Callable[[], None] | None = None,
    ) -> Mapping[str, Any]:
        """Build the action's minimum service set without simulator mutation."""

        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or action_index < 0
        ):
            raise ValueError("action_index must be a non-negative integer")
        if not isinstance(action, ActionEnvelope):
            raise TypeError("action must be ActionEnvelope")
        if action.action not in _KNOWN_ACTIONS:
            raise R1ProB1ServiceFactoryError(
                f"unsupported B1 action: {action.action!r}"
            )
        if not callable(public_snapshot_provider):
            raise TypeError("public_snapshot_provider must be callable")
        if not callable(step_once):
            raise TypeError("step_once must be callable")

        # Recheck versions on every action so a mutable injected builder cannot
        # drift after constructor preflight.
        self.preflight().require_available()
        public_snapshot = public_snapshot_provider()
        if not isinstance(public_snapshot, Mapping):
            raise R1ProB1ServiceFactoryError(
                "public_snapshot_provider must return a mapping"
            )
        carry_snapshot = self.public_carry_snapshot()
        component = self._component_for(
            action,
            carry_snapshot=carry_snapshot,
        )
        carry_mode = self._public_proxy_source.carry_mode()

        services: dict[str, Any] = {
            LOCAL_BASE_PUBLIC_GRASP_PROXY_SERVICE: self._public_proxy_source,
            TORSO_PUBLIC_GRASP_PROXY_SERVICE: self._public_proxy_source,
        }
        if component is None:
            return MappingProxyType(services)

        builder = self._builders[component]
        built = builder.build(
            action_index=action_index,
            action=action,
            evaluator=evaluator,
            control=control,
            public_snapshot_provider=public_snapshot_provider,
            step_once=step_once,
            physics_step_once=physics_step_once,
            mark_boundary_already_checkpointed=mark_boundary_already_checkpointed,
            public_carry_snapshot=self.public_carry_snapshot,
            public_grasp_proxy_source=self._public_proxy_source,
            # A logical-inventory hold is intentionally absent from physics
            # while a torso look runs.  It therefore uses the same physical
            # torso service as an empty-hand look; the adaptive plugin still
            # observes ``logical_inventory`` from the public proxy and keeps
            # the held object synchronized at the rendered boundary.
            active_hand=(
                None
                if component == TORSO_COMPONENT
                and carry_mode == "logical_inventory"
                else carry_snapshot["active_hand"]
            ),
        )
        if not isinstance(built, Mapping):
            raise R1ProB1ServiceFactoryError(
                f"{component} builder must return a mapping"
            )
        for key in built:
            if not isinstance(key, str) or not key:
                raise R1ProB1ServiceFactoryError(
                    f"{component} builder returned an invalid service key"
                )
            if key in services:
                raise R1ProB1ServiceFactoryError(
                    f"duplicate resident service key: {key!r}"
                )
        services.update(built)
        return MappingProxyType(services)

    def _component_for(
        self,
        action: ActionEnvelope,
        *,
        carry_snapshot: Mapping[str, Any],
    ) -> str | None:
        likely = carry_snapshot["grasp_proxy_likely"]
        active_hand = carry_snapshot["active_hand"]
        if action.action == "stop":
            return None
        if action.action == "wait":
            if not self._wait_requires_actuator:
                return None
            return CARRY_COMPONENT if likely else BASE_COMPONENT
        if action.action in {
            "grasp_at",
            "grasp",
            "pick",
            "mark_visible_objects",
            "open",
            "close",
            "toggle_on",
            "toggle_off",
        }:
            if action.action == "pick" and likely:
                raise R1ProB1ServiceFactoryError(
                    "pick is unavailable while carrying"
                )
            if action.action in {"open", "close", "toggle_on", "toggle_off"} and likely:
                raise R1ProB1ServiceFactoryError(
                    f"{action.action} is unavailable while carrying"
                )
            return GRASP_COMPONENT
        if action.action in _BASE_ACTIONS:
            return CARRY_COMPONENT if likely else BASE_COMPONENT
        if action.action in _LOOK_ACTIONS:
            if not likely:
                return TORSO_COMPONENT
            return (
                TORSO_COMPONENT
                if self._public_proxy_source.carry_mode()
                == "logical_inventory"
                else CARRY_COMPONENT
            )
        if action.action == "place_inside_at":
            _require_matching_active_hand(
                action,
                likely=likely,
                active_hand=active_hand,
            )
            return PLACE_COMPONENT
        if action.action in {
            "place_inside",
            "place_on_top",
            "place_next_to",
            "place_under",
            "place_touching",
        }:
            if not likely or active_hand not in {"left", "right"}:
                raise R1ProB1ServiceFactoryError(
                    f"{action.action} requires a likely implicit-hand carry state"
                )
            if not isinstance(
                action.args,
                (
                    MarkerPlaceInsideArgs,
                    MarkerTargetArgs,
                    MarkerTargetOrRequestArgs,
                ),
            ):
                raise R1ProB1ServiceFactoryError(
                    f"{action.action} requires parsed Marker placement arguments"
                )
            return GRASP_COMPONENT
        if action.action == "put_back":
            if (
                not likely
                or active_hand not in {"left", "right"}
                or not isinstance(action.args, NoArgs)
            ):
                raise R1ProB1ServiceFactoryError(
                    "put_back requires a likely implicit-hand carry state"
                )
            return GRASP_COMPONENT
        if action.action == "place_on_floor":
            if (
                not likely
                or active_hand not in {"left", "right"}
                or not isinstance(action.args, NoArgs)
            ):
                raise R1ProB1ServiceFactoryError(
                    "place_on_floor requires a likely implicit-hand carry state"
                )
            return GRASP_COMPONENT
        if action.action == "release":
            if isinstance(action.args, NoArgs):
                if not likely or active_hand not in {"left", "right"}:
                    raise R1ProB1ServiceFactoryError(
                        "marker release requires a likely implicit-hand carry state"
                    )
                return GRASP_COMPONENT
            _require_matching_active_hand(
                action,
                likely=likely,
                active_hand=active_hand,
            )
            return RELEASE_COMPONENT
        raise R1ProB1ServiceFactoryError(
            f"unsupported B1 action: {action.action!r}"
        )


def _require_matching_active_hand(
    action: ActionEnvelope,
    *,
    likely: bool,
    active_hand: str | None,
) -> None:
    if not likely or active_hand not in {"left", "right"}:
        raise R1ProB1ServiceFactoryError(
            f"{action.action} requires a likely public carry state"
        )
    if action.action == "release":
        if not isinstance(action.args, ReleaseArgs):
            raise R1ProB1ServiceFactoryError(
                "release requires parsed ReleaseArgs"
            )
    elif not isinstance(action.args, PointActionArgs):
        raise R1ProB1ServiceFactoryError(
            "place_inside_at requires parsed PointActionArgs"
        )
    if action.args.hand != active_hand:
        raise R1ProB1ServiceFactoryError(
            f"{action.action} hand differs from public active hand"
        )
