"""Evaluator-only composition for visibility-gated official grasping."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from behavior_hl.embodied.evaluator_bridge.bounded_official_grasp import (
    BoundedOfficialGraspDriver,
    OfficialGraspTick,
)
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
    PrivateVisibilityConfig,
    PrivateVisibilityResolution,
    VisibilityStatus,
    resolve_visible_task_category,
    resolve_visible_task_instance,
)


VISIBLE_OFFICIAL_GRASP_BACKEND_VERSION = (
    "behavior2026.visible-official-grasp-backend/v1"
)


class VisibleOfficialGraspBackendError(RuntimeError):
    """Raised before mutation when private inputs are inconsistent."""


@dataclass(frozen=True, slots=True)
class _DriverToken:
    driver: BoundedOfficialGraspDriver
    state: Any


class VisibleOfficialGraspBackendAdapter:
    """Resolve current masks and drive one hand-specific official primitive."""

    api_contract_version = VISIBLE_OFFICIAL_GRASP_BACKEND_VERSION

    def __init__(
        self,
        *,
        private_frame_provider: Callable[[], PrivateInstanceSegmentationFrame],
        eligible_objects_provider: Callable[[], Mapping[str, Any]],
        eligible_objects_by_category_provider: (
            Callable[[], Mapping[str, Mapping[str, Any]]] | None
        ) = None,
        driver_by_hand: Mapping[str, BoundedOfficialGraspDriver],
        visibility_config: PrivateVisibilityConfig | None = None,
    ) -> None:
        if not callable(private_frame_provider):
            raise TypeError("private_frame_provider must be callable")
        if not callable(eligible_objects_provider):
            raise TypeError("eligible_objects_provider must be callable")
        if (
            eligible_objects_by_category_provider is not None
            and not callable(eligible_objects_by_category_provider)
        ):
            raise TypeError(
                "eligible_objects_by_category_provider must be callable"
            )
        drivers = dict(driver_by_hand)
        if not drivers or not set(drivers).issubset({"left", "right"}):
            raise ValueError("driver_by_hand must contain left/right keys")
        if any(
            not isinstance(driver, BoundedOfficialGraspDriver)
            for driver in drivers.values()
        ):
            raise TypeError(
                "driver_by_hand values must be BoundedOfficialGraspDriver"
            )
        self._private_frame_provider = private_frame_provider
        self._eligible_objects_provider = eligible_objects_provider
        self._eligible_objects_by_category_provider = (
            eligible_objects_by_category_provider
        )
        self._driver_by_hand = MappingProxyType(drivers)
        self.visibility_config = visibility_config or PrivateVisibilityConfig()

    def resolve_visible_target(
        self,
        *,
        camera: str,
        point: tuple[float, float],
        hand: str,
    ) -> PrivateVisibilityResolution:
        # The initial research mode deliberately exposes only the head-camera
        # gate.  Wrist segmentation can be added as another explicit mode.
        if camera != "head" or hand not in self._driver_by_hand:
            return PrivateVisibilityResolution(VisibilityStatus.INVALID_POINT)
        frame = self._private_frame_provider()
        if not isinstance(frame, PrivateInstanceSegmentationFrame):
            raise VisibleOfficialGraspBackendError(
                "private frame provider returned the wrong type"
            )
        if frame.camera != camera:
            raise VisibleOfficialGraspBackendError(
                "private segmentation camera is stale or mismatched"
            )
        objects = self._eligible_objects_provider()
        if not isinstance(objects, Mapping):
            raise VisibleOfficialGraspBackendError(
                "eligible object provider must return a mapping"
            )
        return resolve_visible_task_instance(
            frame=frame,
            normalized_point=point,
            task_objects_by_name=objects,
            config=self.visibility_config,
        )

    def resolve_visible_object(
        self,
        *,
        object_category: str,
        selector: str | None,
    ) -> PrivateVisibilityResolution:
        provider = self._eligible_objects_by_category_provider
        if provider is None:
            raise VisibleOfficialGraspBackendError(
                "category-level visible-object provider is not configured"
            )
        frame = self._private_frame_provider()
        if not isinstance(frame, PrivateInstanceSegmentationFrame):
            raise VisibleOfficialGraspBackendError(
                "private frame provider returned the wrong type"
            )
        if frame.camera != "head":
            raise VisibleOfficialGraspBackendError(
                "private segmentation camera is stale or mismatched"
            )
        objects_by_category = provider()
        if not isinstance(objects_by_category, Mapping):
            raise VisibleOfficialGraspBackendError(
                "category object provider must return a mapping"
            )
        return resolve_visible_task_category(
            frame=frame,
            object_category=object_category,
            selector=selector,
            task_objects_by_category=objects_by_category,
            config=self.visibility_config,
        )

    def begin_official_grasp(self, *, target_token: Any, hand: str) -> Any:
        try:
            driver = self._driver_by_hand[hand]
        except KeyError as exc:
            raise VisibleOfficialGraspBackendError(
                f"no official grasp driver admitted for hand {hand!r}"
            ) from exc
        return _DriverToken(
            driver=driver,
            state=driver.begin(target_token=target_token, hand=hand),
        )

    def tick_official_grasp(self, state_token: Any) -> OfficialGraspTick:
        token = _require_token(state_token)
        return token.driver.tick(token.state)

    def abort_official_grasp(self, state_token: Any) -> OfficialGraspTick:
        token = _require_token(state_token)
        return token.driver.abort(token.state)


def _require_token(value: Any) -> _DriverToken:
    if not isinstance(value, _DriverToken):
        raise VisibleOfficialGraspBackendError(
            "invalid visible official grasp state token"
        )
    return value
