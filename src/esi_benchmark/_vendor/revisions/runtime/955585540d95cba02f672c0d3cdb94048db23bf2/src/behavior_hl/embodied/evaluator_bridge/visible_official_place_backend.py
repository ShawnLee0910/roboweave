"""Evaluator-only composition for visibility-gated official placement."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from behavior_hl.embodied.evaluator_bridge.bounded_official_place import (
    BoundedOfficialPlaceDriver,
    OfficialPlaceTick,
)
from behavior_hl.embodied.evaluator_bridge.private_instance_visibility import (
    PrivateInstanceSegmentationFrame,
    PrivateVisibilityConfig,
    PrivateVisibilityResolution,
    VisibilityStatus,
    resolve_visible_task_instance,
)


VISIBLE_OFFICIAL_PLACE_BACKEND_VERSION = (
    "behavior2026.visible-official-place-inside-backend/v1"
)


class VisibleOfficialPlaceBackendError(RuntimeError):
    """Raised before mutation when private placement inputs are inconsistent."""


@dataclass(frozen=True, slots=True)
class _DriverToken:
    driver: BoundedOfficialPlaceDriver
    state: Any


class VisibleOfficialPlaceBackendAdapter:
    """Resolve a visible container and drive one official PLACE_INSIDE."""

    api_contract_version = VISIBLE_OFFICIAL_PLACE_BACKEND_VERSION

    def __init__(
        self,
        *,
        private_frame_provider: Callable[[], PrivateInstanceSegmentationFrame],
        eligible_containers_provider: Callable[[], Mapping[str, Any]],
        driver_by_hand: Mapping[str, BoundedOfficialPlaceDriver],
        visibility_config: PrivateVisibilityConfig | None = None,
    ) -> None:
        if not callable(private_frame_provider):
            raise TypeError("private_frame_provider must be callable")
        if not callable(eligible_containers_provider):
            raise TypeError("eligible_containers_provider must be callable")
        drivers = dict(driver_by_hand)
        if not drivers or not set(drivers).issubset({"left", "right"}):
            raise ValueError("driver_by_hand must contain left/right keys")
        if any(
            not isinstance(driver, BoundedOfficialPlaceDriver)
            for driver in drivers.values()
        ):
            raise TypeError(
                "driver_by_hand values must be BoundedOfficialPlaceDriver"
            )
        self._private_frame_provider = private_frame_provider
        self._eligible_containers_provider = eligible_containers_provider
        self._driver_by_hand = MappingProxyType(drivers)
        self.visibility_config = visibility_config or PrivateVisibilityConfig(
            minimum_visible_pixels=4,
            point_search_radius_px=4,
        )

    def resolve_visible_container(
        self,
        *,
        camera: str,
        point: tuple[float, float],
        hand: str,
    ) -> PrivateVisibilityResolution:
        if camera != "head" or hand not in self._driver_by_hand:
            return PrivateVisibilityResolution(VisibilityStatus.INVALID_POINT)
        frame = self._private_frame_provider()
        if not isinstance(frame, PrivateInstanceSegmentationFrame):
            raise VisibleOfficialPlaceBackendError(
                "private frame provider returned the wrong type"
            )
        if frame.camera != camera:
            raise VisibleOfficialPlaceBackendError(
                "private segmentation camera is stale or mismatched"
            )
        containers = self._eligible_containers_provider()
        if not isinstance(containers, Mapping):
            raise VisibleOfficialPlaceBackendError(
                "eligible container provider must return a mapping"
            )
        return resolve_visible_task_instance(
            frame=frame,
            normalized_point=point,
            task_objects_by_name=containers,
            config=self.visibility_config,
        )

    def begin_official_place(self, *, container_token: Any, hand: str) -> Any:
        try:
            driver = self._driver_by_hand[hand]
        except KeyError as exc:
            raise VisibleOfficialPlaceBackendError(
                f"no official place driver admitted for hand {hand!r}"
            ) from exc
        return _DriverToken(
            driver=driver,
            state=driver.begin(container_token=container_token, hand=hand),
        )

    def tick_official_place(self, state_token: Any) -> OfficialPlaceTick:
        token = _require_token(state_token)
        return token.driver.tick(token.state)

    def abort_official_place(self, state_token: Any) -> OfficialPlaceTick:
        token = _require_token(state_token)
        return token.driver.abort(token.state)


def _require_token(value: Any) -> _DriverToken:
    if not isinstance(value, _DriverToken):
        raise VisibleOfficialPlaceBackendError(
            "invalid visible official place state token"
        )
    return value
