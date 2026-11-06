"""Deterministic geometry-only ordering for container placement.

The Oracle may use private live geometry, but it must not encode task names,
object categories, or instance-specific orders.  This helper ranks awkward,
large-footprint objects before compact objects so a strict official Inside
sampler is less likely to discover late that the remaining free volume cannot
accept a flat or bulky item.

It intentionally does not move or rearrange an already placed object.  The
result is only a planning order; every placement is still accepted solely by
the official Inside state and verifier.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


PACKING_ORDER_VERSION = "behavior2026.geometry-only-packing-order/v1"


def _finite_extent(value: Any) -> tuple[float, float, float] | None:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if result.size != 3 or not np.isfinite(result).all() or np.any(result <= 0.0):
        return None
    return tuple(float(item) for item in result)


def live_object_extent_m(obj: Any) -> tuple[float, float, float]:
    """Read one live object's world AABB extent without simulator imports."""

    extent = _finite_extent(getattr(obj, "aabb_extent", None))
    if extent is not None:
        return extent
    aabb = getattr(obj, "aabb", None)
    if isinstance(aabb, (tuple, list)) and len(aabb) == 2:
        lower = np.asarray(aabb[0], dtype=np.float64).reshape(-1)
        upper = np.asarray(aabb[1], dtype=np.float64).reshape(-1)
        extent = _finite_extent(upper - lower)
        if extent is not None:
            return extent
    raise ValueError("live object exposes no finite positive 3-D extent")


def packing_priority_from_extent(
    extent_m: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Return a descending difficulty key using geometry only.

    The first component combines the largest possible support footprint with
    thinness.  A flat board therefore precedes a compact ball even when its
    volume is smaller.  Remaining components make ties deterministic and put
    larger objects first.
    """

    dims = tuple(sorted(float(value) for value in extent_m))
    if any(not math.isfinite(value) or value <= 0.0 for value in dims):
        raise ValueError("packing extent must be finite and positive")
    thickness, middle, longest = dims
    footprint = middle * longest
    flatness = longest / thickness
    volume = thickness * middle * longest
    difficulty = footprint * math.sqrt(flatness)
    return difficulty, footprint, volume, longest


def rank_live_objects_for_packing(
    objects: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    """Rank live objects least-placeable first, with stable name tie-breaks."""

    measured = []
    for name, obj in objects.items():
        if not isinstance(name, str) or not name:
            raise ValueError("packing object names must be non-empty strings")
        extent = live_object_extent_m(obj)
        measured.append((name, obj, packing_priority_from_extent(extent)))
    measured.sort(
        key=lambda item: tuple(-value for value in item[2]) + (item[0],)
    )
    return tuple((name, obj) for name, obj, _priority in measured)

