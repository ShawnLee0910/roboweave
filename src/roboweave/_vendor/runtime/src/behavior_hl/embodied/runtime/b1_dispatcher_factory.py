"""Pure B1 dispatcher assembly for the picking-up-toys action profile."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from behavior_hl.embodied.policy.approach_execution import BaseMacroConfig
from behavior_hl.embodied.policy.profiles import (
    build_picking_up_toys_registry,
)
from behavior_hl.embodied.policy.registry import ActionRegistry
from behavior_hl.embodied.runtime.dispatch import ActionPluginDispatcher
from behavior_hl.embodied.runtime.plugins.carry import (
    CarrySafeLookPlugin,
    CarrySafetyLimits,
    CarryingBaseMotionPlugin,
)
from behavior_hl.embodied.runtime.plugins.grasp import (
    PhysicalGraspConfig,
    PhysicalGraspPlugin,
)
from behavior_hl.embodied.runtime.plugins.local_base import (
    AdaptivePhysicalActionPlugin as AdaptiveBasePlugin,
)
from behavior_hl.embodied.runtime.plugins.local_base import (
    LocalBaseMotionPlugin,
    LocalBaseSafetyLimits,
)
from behavior_hl.embodied.runtime.plugins.place import (
    PlaceConfig,
    PlaceInsideAtPlugin,
)
from behavior_hl.embodied.runtime.plugins.marker_symbolic_grasp import (
    MarkerSymbolicGraspPlugin,
    MarkerSymbolicPlaceInsidePlugin,
    MarkerSymbolicV34OperationPlugin,
    MarkVisibleObjectsPlugin,
)
from behavior_hl.embodied.runtime.plugins.release import (
    PhysicalReleaseConfig,
    PhysicalReleasePlugin,
)
from behavior_hl.embodied.runtime.plugins.torso import (
    AdaptivePhysicalActionPlugin as AdaptiveTorsoPlugin,
)
from behavior_hl.embodied.runtime.plugins.torso import (
    EmptyTorsoLimits,
    EmptyTorsoGridPlugin,
    EmptyTorsoLookPlugin,
    RejectCarryingTorsoGridPlugin,
)
from behavior_hl.embodied.runtime.plugins.wait import (
    FixedDurationWaitPlugin,
    SymbolicWaitPlugin,
)


BASE_ACTIONS = (
    "move_backward",
    "move_forward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
)
LOOK_ACTION_PROFILES = MappingProxyType(
    {
        "look_up": "up",
        "look_center": "center",
        "look_down": "down",
    }
)
VERTICAL_ACTION_PROFILES = MappingProxyType(
    {
        "move_up": "up",
        "move_down": "down",
    }
)
ACTION23_TORSO_ACTIONS = MappingProxyType(
    {
        "move_up": ("height", 1),
        "move_down": ("height", -1),
        "look_up": ("pitch", 1),
        "look_down": ("pitch", -1),
    }
)
CARRY_COMPENSATOR_SERVICE = "carry_arm_compensator"


class B1DispatcherFactoryError(ValueError):
    """Raised when B1 plugin assembly would rely on an implicit capability."""


@dataclass(frozen=True, slots=True)
class CarryBaseParameters:
    """Explicit velocity and odometry target for one carrying base action."""

    base_velocity: tuple[float, float, float]
    target_odometry: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class B1DispatcherBundle:
    registry: ActionRegistry
    dispatcher: ActionPluginDispatcher
    injected_services: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "injected_services",
            MappingProxyType(dict(self.injected_services)),
        )


def build_b1_picking_up_toys_dispatcher(
    *,
    grasp_config: PhysicalGraspConfig,
    place_config: PlaceConfig,
    release_config: PhysicalReleaseConfig,
    empty_base_config: BaseMacroConfig,
    empty_base_safety: LocalBaseSafetyLimits,
    empty_torso_limits: EmptyTorsoLimits,
    carry_limits: CarrySafetyLimits,
    carry_safe_arm_targets: Mapping[str, Sequence[float]],
    carry_compensator: Any,
    carry_base_parameters: Mapping[str, CarryBaseParameters],
    carry_look_fractions: Mapping[str, float],
    calibrated_carry_hands: Sequence[str] = ("left", "right"),
    registry: ActionRegistry | None = None,
    grasp_plugin: Any | None = None,
    place_plugin: Any | None = None,
    mark_plugin: Any | None = None,
    base_plugins: Mapping[str, Any] | None = None,
    torso_plugins: Mapping[str, Any] | None = None,
) -> B1DispatcherBundle:
    """Assemble every registry non-stop plugin without simulator access."""

    _require_instance(grasp_config, PhysicalGraspConfig, "grasp_config")
    _require_instance(place_config, PlaceConfig, "place_config")
    _require_instance(
        release_config,
        PhysicalReleaseConfig,
        "release_config",
    )
    _require_instance(
        empty_base_config,
        BaseMacroConfig,
        "empty_base_config",
    )
    _require_instance(
        empty_base_safety,
        LocalBaseSafetyLimits,
        "empty_base_safety",
    )
    _require_instance(
        empty_torso_limits,
        EmptyTorsoLimits,
        "empty_torso_limits",
    )
    _require_instance(
        carry_limits,
        CarrySafetyLimits,
        "carry_limits",
    )
    if not callable(
        getattr(carry_compensator, "target_for_torso_step", None)
    ):
        raise B1DispatcherFactoryError(
            "carry_compensator must explicitly implement "
            "target_for_torso_step"
        )
    calibrated_hands = tuple(calibrated_carry_hands)
    if (
        not calibrated_hands
        or len(set(calibrated_hands)) != len(calibrated_hands)
        or not set(calibrated_hands).issubset({"left", "right"})
    ):
        raise B1DispatcherFactoryError(
            "calibrated_carry_hands must be unique left/right hand names"
        )
    if set(carry_safe_arm_targets) != set(calibrated_hands):
        if set(calibrated_hands) == {"left", "right"}:
            requirement = "left and right calibrated hands"
        else:
            requirement = "exactly the calibrated_carry_hands"
        raise B1DispatcherFactoryError(
            "carry_safe_arm_targets must explicitly define "
            f"{requirement}"
        )
    if set(carry_base_parameters) != set(BASE_ACTIONS):
        raise B1DispatcherFactoryError(
            "carry_base_parameters must explicitly cover all six base actions"
        )
    if frozenset(carry_look_fractions) not in {
        frozenset(LOOK_ACTION_PROFILES),
        frozenset(VERTICAL_ACTION_PROFILES),
    }:
        raise B1DispatcherFactoryError(
            "carry_look_fractions must explicitly cover all three look actions "
            "or both vertical actions"
        )
    for action, parameters in carry_base_parameters.items():
        if not isinstance(parameters, CarryBaseParameters):
            raise B1DispatcherFactoryError(
                f"carry_base_parameters[{action!r}] has invalid type"
            )
    for action, value in carry_look_fractions.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise B1DispatcherFactoryError(
                f"carry_look_fractions[{action!r}] must be numeric"
            )

    active_registry = registry or build_picking_up_toys_registry()
    _require_instance(
        active_registry,
        ActionRegistry,
        "registry",
    )
    expected_by_action = {
        name: active_registry.spec(name).controller_key
        for name in active_registry.names
        if name != "stop"
    }
    v34_mode = "pick" in expected_by_action
    # ``wait`` is the stable runtime discriminator shared by the legacy
    # Action23 surfaces and the strict Action20 surface.  Do not key this off
    # one schema-version string: Action23 intentionally has both the original
    # 23-action grid profile and the later 21-action simple-spatial profile,
    # while Action20 removes public Mark and the two vertical actions.
    symbolic_wait_mode = "wait" in expected_by_action
    action20_mode = symbolic_wait_mode and (
        "mark_visible_objects" not in expected_by_action
    )
    action23_mode = symbolic_wait_mode and not action20_mode
    markerless_place_action = (
        "place_on_floor" if action20_mode else "release"
    )
    grasp_actions = set(expected_by_action).intersection(
        {"grasp_at", "grasp", "pick"}
    )
    if len(grasp_actions) != 1:
        raise B1DispatcherFactoryError(
            "registry must contain exactly one supported grasp action"
        )
    grasp_action = next(iter(grasp_actions))
    place_actions = set(expected_by_action).intersection(
        {"place_inside_at", "place_inside"}
    )
    if len(place_actions) != 1:
        raise B1DispatcherFactoryError(
            "registry must contain exactly one supported place-inside action"
        )
    place_action = next(iter(place_actions))
    marker_mode = "mark_visible_objects" in expected_by_action
    symbolic_torso_actions = set(expected_by_action).intersection(
        ACTION23_TORSO_ACTIONS
    )
    valid_symbolic_torso_surfaces = (
        {"look_up", "look_down"},
        {"look_up", "look_down", "move_up", "move_down"},
    )
    if (
        symbolic_wait_mode
        and symbolic_torso_actions not in valid_symbolic_torso_surfaces
    ):
        raise B1DispatcherFactoryError(
            "symbolic wait profile has an invalid torso action surface"
        )
    if action20_mode and symbolic_torso_actions != {
        "look_up",
        "look_down",
    }:
        raise B1DispatcherFactoryError(
            "Action20 must not expose move_up or move_down"
        )
    torso_profiles = (
        {
            action: ACTION23_TORSO_ACTIONS[action]
            for action in ACTION23_TORSO_ACTIONS
            if action in symbolic_torso_actions
        }
        if symbolic_wait_mode
        else VERTICAL_ACTION_PROFILES
        if v34_mode
        else LOOK_ACTION_PROFILES
    )
    required_carry_fraction_actions = (
        VERTICAL_ACTION_PROFILES if symbolic_wait_mode else torso_profiles
    )
    if frozenset(carry_look_fractions) != frozenset(
        required_carry_fraction_actions
    ):
        raise B1DispatcherFactoryError(
            "carry_look_fractions does not match the active registry torso "
            "action profile"
        )
    symbolic_wait_spatial_actions = (
        "place_on_top",
        "place_next_to",
        "place_under",
        *(("place_touching",) if action23_mode else ()),
    )
    wipe_actions = ("wipe",) if action20_mode and "wipe" in expected_by_action else ()
    if symbolic_wait_mode:
        expected_actions = {
            *BASE_ACTIONS,
            *torso_profiles,
            "pick",
            markerless_place_action,
            "place_inside",
            *symbolic_wait_spatial_actions,
            "open",
            "close",
            "toggle_on",
            "toggle_off",
            *wipe_actions,
            "wait",
        }
        if action23_mode:
            expected_actions.add("mark_visible_objects")
    elif v34_mode:
        expected_actions = {
            *BASE_ACTIONS,
            *torso_profiles,
            "mark_visible_objects",
            "pick",
            "place_inside",
            "put_back",
            "open",
            "close",
            "place_on_top",
            "place_next_to",
            "place_under",
            "place_touching",
            "toggle_on",
            "toggle_off",
        }
    else:
        expected_actions = {
            grasp_action,
            place_action,
            *BASE_ACTIONS,
            *torso_profiles,
            "release",
        }
        if marker_mode:
            expected_actions.add("mark_visible_objects")
    if set(expected_by_action) != expected_actions:
        raise B1DispatcherFactoryError(
            "registry non-stop action coverage differs from B1"
        )
    if len(set(expected_by_action.values())) != len(expected_actions):
        raise B1DispatcherFactoryError(
            "B1 requires one unique controller key per non-stop action"
        )

    active_grasp_plugin = grasp_plugin or (
        MarkerSymbolicV34OperationPlugin("pick")
        if v34_mode
        else
        MarkerSymbolicGraspPlugin()
        if marker_mode
        else PhysicalGraspPlugin(grasp_config)
    )
    if (
        getattr(active_grasp_plugin, "controller_key", None)
        != expected_by_action[grasp_action]
    ):
        raise B1DispatcherFactoryError(
            "grasp plugin controller_key does not match registry"
        )
    active_place_plugin = place_plugin or (
        MarkerSymbolicV34OperationPlugin("place_inside")
        if v34_mode
        else
        MarkerSymbolicPlaceInsidePlugin()
        if marker_mode
        else PlaceInsideAtPlugin(place_config)
    )
    if (
        getattr(active_place_plugin, "controller_key", None)
        != expected_by_action[place_action]
    ):
        raise B1DispatcherFactoryError(
            "place plugin controller_key does not match registry"
        )
    plugins: list[Any] = [active_grasp_plugin, active_place_plugin]
    if symbolic_wait_mode:
        plugins.extend(
            MarkerSymbolicV34OperationPlugin(
                name,
                two_stage_on_top=(
                    action20_mode and name == "place_on_top"
                ),
            )
            for name in (
                *symbolic_wait_spatial_actions,
                "open",
                "close",
                "toggle_on",
                "toggle_off",
                *wipe_actions,
                markerless_place_action,
            )
        )
    elif v34_mode:
        plugins.extend(
            MarkerSymbolicV34OperationPlugin(name)
            for name in (
                "put_back",
                "open",
                "close",
                "place_on_top",
                "place_next_to",
                "place_under",
                "place_touching",
                "toggle_on",
                "toggle_off",
            )
        )
    else:
        plugins.append(
            PhysicalReleasePlugin(
                controller_key=expected_by_action["release"],
                config=release_config,
            )
        )
    if marker_mode:
        active_mark_plugin = mark_plugin or MarkVisibleObjectsPlugin()
        if (
            getattr(active_mark_plugin, "controller_key", None)
            != expected_by_action["mark_visible_objects"]
        ):
            raise B1DispatcherFactoryError(
                "mark plugin controller_key does not match registry"
            )
        plugins.append(active_mark_plugin)
    if base_plugins is not None:
        if set(base_plugins) != set(BASE_ACTIONS):
            raise B1DispatcherFactoryError(
                "base_plugins must explicitly cover all six base actions"
            )
        for action in BASE_ACTIONS:
            plugin = base_plugins[action]
            if (
                getattr(plugin, "controller_key", None)
                != expected_by_action[action]
            ):
                raise B1DispatcherFactoryError(
                    f"base_plugins[{action!r}] controller_key does not match registry"
                )
            if getattr(plugin, "action_name", None) != action:
                raise B1DispatcherFactoryError(
                    f"base_plugins[{action!r}] action_name does not match"
                )
            plugins.append(plugin)
    else:
        for action in BASE_ACTIONS:
            key = expected_by_action[action]
            parameters = carry_base_parameters[action]
            empty = LocalBaseMotionPlugin(
                controller_key=key,
                action_name=action,
                config=empty_base_config,
                safety=empty_base_safety,
            )
            carrying = CarryingBaseMotionPlugin(
                controller_key=key,
                base_velocity=parameters.base_velocity,
                target_odometry=parameters.target_odometry,
                safe_arm_targets=carry_safe_arm_targets,
                limits=carry_limits,
            )
            plugins.append(
                AdaptiveBasePlugin(
                    controller_key=key,
                    empty_plugin=empty,
                    carrying_plugin=carrying,
                )
            )
    if torso_plugins is not None:
        if set(torso_plugins) != set(torso_profiles):
            raise B1DispatcherFactoryError(
                "torso_plugins must explicitly cover the active torso actions"
            )
        for action in torso_profiles:
            plugin = torso_plugins[action]
            if (
                getattr(plugin, "controller_key", None)
                != expected_by_action[action]
                or getattr(plugin, "action_name", None) != action
            ):
                raise B1DispatcherFactoryError(
                    f"torso_plugins[{action!r}] does not match the registry"
                )
            plugins.append(plugin)
    else:
        for action, profile in torso_profiles.items():
            key = expected_by_action[action]
            if symbolic_wait_mode:
                empty = EmptyTorsoGridPlugin(
                    controller_key=key,
                    action_name=action,
                    limits=empty_torso_limits,
                )
                carrying = RejectCarryingTorsoGridPlugin(
                    controller_key=key,
                    action_name=action,
                )
            else:
                empty = EmptyTorsoLookPlugin(
                    controller_key=key,
                    target_profile=profile,
                    limits=empty_torso_limits,
                )
                carrying = CarrySafeLookPlugin(
                    controller_key=key,
                    target_fraction=float(carry_look_fractions[action]),
                    limits=carry_limits,
                )
            plugins.append(
                AdaptiveTorsoPlugin(
                    controller_key=key,
                    empty_plugin=empty,
                    carrying_plugin=carrying,
                    # Symbolic carry is synchronized to the EEF by the resident
                    # runtime.  It may use the grid look motion; physical carry
                    # remains on the separately rejecting/safe delegate.
                    logical_carry_plugin=(
                        empty if symbolic_wait_mode else None
                    ),
                )
            )
    if symbolic_wait_mode:
        plugins.append(
            (SymbolicWaitPlugin if action20_mode else FixedDurationWaitPlugin)(
                controller_key=expected_by_action["wait"]
            )
        )

    dispatcher = ActionPluginDispatcher(plugins)
    expected_keys = frozenset(expected_by_action.values())
    if frozenset(dispatcher.controller_keys) != expected_keys:
        raise B1DispatcherFactoryError(
            "assembled dispatcher controller coverage is incomplete"
        )
    return B1DispatcherBundle(
        registry=active_registry,
        dispatcher=dispatcher,
        injected_services={
            CARRY_COMPENSATOR_SERVICE: carry_compensator,
        },
    )


def _require_instance(value: Any, expected: type, name: str) -> None:
    if not isinstance(value, expected):
        raise B1DispatcherFactoryError(
            f"{name} must be explicitly supplied as {expected.__name__}"
        )
