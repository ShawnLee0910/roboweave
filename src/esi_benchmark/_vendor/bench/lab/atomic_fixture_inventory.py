"""Private, task-neutral live fixture selection for Action20 atomic cards.

The inventory is built from one already-loaded evaluator scene.  Object names
and capability evidence are private artifacts; callers expose only card IDs
and action names over the resident control socket.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

try:
    from action20_atomic_validation import (
        ACTION20,
        ATOMIC_CARD_SCHEMA_VERSION,
        AtomicActionCard,
        normalize_atomic_cards,
    )
except ModuleNotFoundError:  # Package imports expose the module through lab.
    from lab.action20_atomic_validation import (
        ACTION20,
        ATOMIC_CARD_SCHEMA_VERSION,
        AtomicActionCard,
        normalize_atomic_cards,
    )


ATOMIC_FIXTURE_INVENTORY_VERSION = (
    "behavior-hl-bench-try-private-atomic-fixture-inventory/v1"
)


class AtomicFixtureInventoryError(RuntimeError):
    """The live scene cannot support the complete Action20 card set."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics_private: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics_private = dict(diagnostics_private or {})


class AtomicFixtureInventoryCache:
    """One-shot resident cache; repeated prepare calls never rescan live state."""

    def __init__(self) -> None:
        self._value: AtomicFixtureInventory | None = None

    @property
    def value(self) -> AtomicFixtureInventory | None:
        return self._value

    def get_or_build(
        self,
        builder: Callable[[], AtomicFixtureInventory],
        *,
        before_commit: Callable[[AtomicFixtureInventory], None] | None = None,
    ) -> tuple[AtomicFixtureInventory, bool]:
        if self._value is not None:
            return self._value, True
        value = builder()
        if not isinstance(value, AtomicFixtureInventory):
            raise TypeError("atomic inventory builder returned an invalid value")
        if before_commit is not None:
            before_commit(value)
        self._value = value
        return value, False


@dataclass(frozen=True, slots=True)
class AtomicFixtureInventory:
    cards: tuple[AtomicActionCard, ...]
    evidence_private: Mapping[str, Any]

    def to_private_record(self) -> dict[str, Any]:
        record = {
            "schema_version": ATOMIC_FIXTURE_INVENTORY_VERSION,
            "status": "PASS",
            "card_count": len(self.cards),
            "cards_private": [card.to_private_record() for card in self.cards],
            "capability_evidence_private": dict(self.evidence_private),
            "model_visible": False,
            "prompt_visible": False,
            "public_action_index_after_prepare": 0,
        }
        canonical = json.dumps(
            record, ensure_ascii=False, allow_nan=False, sort_keys=True
        ).encode("utf-8")
        record["inventory_sha256"] = hashlib.sha256(canonical).hexdigest()
        return record

    def to_public_record(self, *, idempotent: bool) -> dict[str, Any]:
        return {
            "status": "PASS",
            "prepared": True,
            "idempotent": bool(idempotent),
            "card_count": len(self.cards),
            "card_ids": [card.card_id for card in self.cards],
            "actions": [card.action for card in self.cards],
        }


def _name(entity: Any) -> str:
    value = getattr(entity, "name", None)
    if not isinstance(value, str) or not value or len(value) > 160:
        raise AtomicFixtureInventoryError("live fixture object has no safe name")
    return value


def _named_objects(objects: Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
    values = objects.values() if isinstance(objects, Mapping) else objects
    result: dict[str, Any] = {}
    for entity in tuple(values):
        name = _name(entity)
        if name in result and result[name] is not entity:
            raise AtomicFixtureInventoryError("live object names are ambiguous")
        result[name] = entity
    if not result:
        raise AtomicFixtureInventoryError("live scene has no named objects")
    return result


def _role_objects(value: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    result = _named_objects(value)
    if set(result) != set(value):
        raise AtomicFixtureInventoryError(f"{role} role names drifted from live names")
    return result


def _state(entity: Any, state_name: str, *, required: bool = False) -> Any | None:
    states = getattr(entity, "states", None)
    matches = []
    if isinstance(states, Mapping):
        matches = [
            state
            for state_type, state in states.items()
            if getattr(state_type, "__name__", "") == state_name
        ]
    if len(matches) == 1:
        return matches[0]
    if matches or required:
        raise AtomicFixtureInventoryError(
            f"live object lacks one unique {state_name} state"
        )
    return None


def _read_boolean_state(entity: Any, state_name: str, *args: Any) -> bool:
    state = _state(entity, state_name, required=True)
    getter = getattr(state, "get_value", None)
    if not callable(getter):
        raise AtomicFixtureInventoryError(f"{state_name} state is unreadable")
    try:
        value = getter(*args)
    except Exception as exc:
        raise AtomicFixtureInventoryError(f"{state_name} state read failed") from exc
    if not isinstance(value, bool):
        raise AtomicFixtureInventoryError(f"{state_name} state is not boolean")
    return value


def _vector3(value: Any, *, field: str) -> tuple[float, float, float]:
    try:
        converted = tuple(float(item) for item in value)
    except Exception as exc:
        raise AtomicFixtureInventoryError(f"{field} is unreadable") from exc
    if len(converted) != 3 or not all(math.isfinite(item) for item in converted):
        raise AtomicFixtureInventoryError(f"{field} is not one finite xyz vector")
    return converted


def _aabb(
    entity: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    raw = getattr(entity, "aabb", None)
    if raw is None:
        state = _state(entity, "AABB", required=True)
        getter = getattr(state, "get_value", None)
        if not callable(getter):
            raise AtomicFixtureInventoryError("AABB state is unreadable")
        raw = getter()
    try:
        low_raw, high_raw = raw
    except Exception as exc:
        raise AtomicFixtureInventoryError("AABB value is malformed") from exc
    low = _vector3(low_raw, field="AABB lower")
    high = _vector3(high_raw, field="AABB upper")
    if any(high[index] <= low[index] for index in range(3)):
        raise AtomicFixtureInventoryError("AABB has no positive volume")
    return low, high


def _stable(entity: Any) -> bool:
    fixed = getattr(entity, "fixed_base", False)
    asleep = getattr(entity, "is_asleep", False)
    return fixed is True or asleep is True


def _fixed_support(entity: Any) -> bool:
    return getattr(entity, "fixed_base", False) is True


def _read_aabb(
    entity: Any,
    reader: Callable[[Any], Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if reader is _aabb:
        return _aabb(entity)
    try:
        low_raw, high_raw = reader(entity)
    except Exception as exc:
        raise AtomicFixtureInventoryError("shared AABB reader failed") from exc
    low = _vector3(low_raw, field="AABB lower")
    high = _vector3(high_raw, field="AABB upper")
    if any(high[index] <= low[index] for index in range(3)):
        raise AtomicFixtureInventoryError("AABB has no positive volume")
    return low, high


def _surface_rank(
    *, source: Any, candidate: Any, aabb_reader: Callable[[Any], Any]
) -> tuple[float, float, str] | None:
    source_low, source_high = _read_aabb(source, aabb_reader)
    low, high = _read_aabb(candidate, aabb_reader)
    source_x = source_high[0] - source_low[0]
    source_y = source_high[1] - source_low[1]
    width = high[0] - low[0]
    depth = high[1] - low[1]
    top = high[2]
    if (
        not _fixed_support(candidate)
        or width < source_x + 0.16
        or depth < source_y + 0.16
        or not 0.20 <= top <= 1.30
    ):
        return None
    return (-(width * depth), abs(top - 0.75), _name(candidate))


def _select_capability_target(
    objects: Mapping[str, Any],
    *,
    state_name: str,
    aabb_reader: Callable[[Any], Any],
    station_feasibility_query: Callable[[Any], Mapping[str, Any]] | None,
) -> tuple[Any, bool, Mapping[str, Any]]:
    candidates: list[
        tuple[int, int, float, float, str, Any, bool, Mapping[str, Any] | None]
    ] = []
    scanned: list[str] = []
    rejected: list[Mapping[str, Any]] = []
    station_evidence: list[Mapping[str, Any]] = []
    for name, entity in sorted(objects.items()):
        try:
            state = _state(entity, state_name)
        except AtomicFixtureInventoryError:
            rejected.append({"name_private": name, "reason": "ambiguous_state"})
            continue
        if state is None:
            continue
        scanned.append(name)
        getter = getattr(state, "get_value", None)
        if not callable(getter):
            rejected.append({"name_private": name, "reason": "missing_getter"})
            continue
        try:
            value = getter()
        except Exception:
            rejected.append({"name_private": name, "reason": "getter_failed"})
            continue
        if not isinstance(value, bool):
            rejected.append({"name_private": name, "reason": "non_boolean_value"})
            continue
        try:
            low, high = _read_aabb(entity, aabb_reader)
            footprint = (high[0] - low[0]) * (high[1] - low[1])
            height = high[2] - low[2]
        except AtomicFixtureInventoryError:
            rejected.append({"name_private": name, "reason": "invalid_aabb"})
            continue
        operation_center_z = (low[2] + high[2]) * 0.5
        if not 0.10 <= operation_center_z <= 1.80:
            rejected.append(
                {"name_private": name, "reason": "operation_center_height"}
            )
            continue
        station_record = None
        if station_feasibility_query is not None:
            try:
                raw_station_record = station_feasibility_query(entity)
                if not isinstance(raw_station_record, Mapping):
                    raise TypeError("station query did not return a mapping")
                station_record = dict(raw_station_record)
                json.dumps(
                    station_record,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
            except Exception as exc:
                rejected.append(
                    {
                        "name_private": name,
                        "reason": "station_feasibility_query_failed",
                        "error_type_private": type(exc).__name__,
                    }
                )
                continue
            station_evidence.append(
                {
                    "name_private": name,
                    "evidence_private": station_record,
                }
            )
            if station_record.get("accepted") is not True:
                rejected.append(
                    {
                        "name_private": name,
                        "reason": "station_feasibility_rejected",
                        "station_evidence_private": station_record,
                    }
                )
                continue
        stability_rank = (
            0 if _fixed_support(entity) else (1 if _stable(entity) else 2)
        )
        candidates.append(
            (
                stability_rank,
                0 if height <= 1.40 else 1,
                abs(operation_center_z - 0.80),
                footprint,
                name,
                entity,
                value,
                station_record,
            )
        )
    if not candidates:
        raise AtomicFixtureInventoryError(
            f"live scene has no readable unique {state_name} capability",
            diagnostics_private={
                "stage": "capability_selection",
                "state_name": state_name,
                "scanned_names_private": scanned,
                "rejected_candidates_private": rejected,
            },
        )
    candidates.sort(key=lambda item: item[:5])
    _, _, _, _, _, selected, value, selected_station = candidates[0]
    return selected, value, {
        "eligible_names_private": list(scanned),
        "rejected_candidates_private": rejected,
        "station_feasibility_query_used": station_feasibility_query is not None,
        "selected_station_feasibility_private": selected_station,
        "candidate_station_feasibility_private": station_evidence,
        "unique_state_instance": True,
    }


def _select_inside_pair(
    graspables: Mapping[str, Any], containers: Mapping[str, Any]
) -> tuple[Any, Any]:
    for _source_name, source in sorted(graspables.items()):
        for _target_name, target in sorted(containers.items()):
            if not _read_boolean_state(source, "Inside", target):
                return source, target
    raise AtomicFixtureInventoryError(
        "every task graspable is already Inside every task container"
    )


def _unique_on_top_support(
    reference: Any, objects: Mapping[str, Any]
) -> tuple[Any | None, str | None]:
    try:
        state = _state(reference, "OnTop")
    except AtomicFixtureInventoryError:
        return None, "ambiguous_on_top_state"
    if state is None:
        return None, "missing_on_top_state"
    getter = getattr(state, "get_value", None)
    if not callable(getter):
        return None, "unreadable_on_top_state"
    matches = []
    for name, candidate in sorted(objects.items()):
        if candidate is reference or not _fixed_support(candidate):
            continue
        try:
            supported = getter(candidate)
        except Exception:
            return None, "unreadable_fixed_support_relation"
        if not isinstance(supported, bool):
            return None, "non_boolean_fixed_support_relation"
        if supported:
            matches.append((name, candidate))
    if len(matches) != 1:
        return None, "fixed_support_count_not_one"
    return matches[0][1], None


def _select_lateral_pair(
    graspables: Mapping[str, Any],
    objects: Mapping[str, Any],
    *,
    relation: str,
    aabb_reader: Callable[[Any], Any],
    station_feasibility_query: Callable[[Any], Mapping[str, Any]] | None,
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    if relation not in {"NextTo", "Touching"}:
        raise ValueError("lateral fixture relation is invalid")
    candidates = []
    rejected: list[Mapping[str, Any]] = []
    graspable_ids = frozenset(id(entity) for entity in graspables.values())
    for reference_name, reference in sorted(objects.items()):
        reference_is_task_graspable = id(reference) in graspable_ids
        if relation == "NextTo" and reference_is_task_graspable:
            rejected.append(
                {
                    "reference_name_private": reference_name,
                    "reason": "task_graspable_reference_excluded",
                }
            )
            continue
        support, support_error = _unique_on_top_support(reference, objects)
        if support is None:
            rejected.append(
                {
                    "reference_name_private": reference_name,
                    "reason": support_error,
                }
            )
            continue
        support_name = _name(support)
        try:
            reference_low, reference_high = _read_aabb(reference, aabb_reader)
            support_low, support_high = _read_aabb(support, aabb_reader)
        except AtomicFixtureInventoryError:
            rejected.append(
                {
                    "reference_name_private": reference_name,
                    "support_name_private": support_name,
                    "reason": "invalid_reference_or_support_aabb",
                }
            )
            continue
        width = reference_high[0] - reference_low[0]
        depth = reference_high[1] - reference_low[1]
        height = reference_high[2] - reference_low[2]
        maximum_extent = max(width, depth, height)
        center_z = (reference_low[2] + reference_high[2]) * 0.5
        if (
            width > 1.25
            or depth > 1.25
            or height > 1.80
            or not 0.05 <= center_z <= 1.80
        ):
            rejected.append(
                {
                    "reference_name_private": reference_name,
                    "support_name_private": support_name,
                    "reason": "reference_geometry_not_moderate",
                }
            )
            continue
        edge_clearances = (
            reference_low[0] - support_low[0],
            support_high[0] - reference_high[0],
            reference_low[1] - support_low[1],
            support_high[1] - reference_high[1],
        )
        clearance = min(edge_clearances)
        if clearance < 0.0:
            rejected.append(
                {
                    "reference_name_private": reference_name,
                    "support_name_private": support_name,
                    "reason": "reference_exceeds_support_edges",
                }
            )
            continue
        station_record = None
        if station_feasibility_query is not None:
            try:
                raw_station_record = station_feasibility_query(reference)
                if not isinstance(raw_station_record, Mapping):
                    raise TypeError("station query did not return a mapping")
                station_record = dict(raw_station_record)
                json.dumps(
                    station_record,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
            except Exception as exc:
                rejected.append(
                    {
                        "reference_name_private": reference_name,
                        "support_name_private": support_name,
                        "reason": "station_feasibility_query_failed",
                        "error_type_private": type(exc).__name__,
                    }
                )
                continue
            if station_record.get("accepted") is not True:
                rejected.append(
                    {
                        "reference_name_private": reference_name,
                        "support_name_private": support_name,
                        "reason": "station_feasibility_rejected",
                        "station_evidence_private": station_record,
                    }
                )
                continue
        for source_name, source in sorted(graspables.items()):
            if source is reference or source_name == reference_name:
                continue
            try:
                initial_next_to = _read_boolean_state(source, "NextTo", reference)
                initial_touching = _read_boolean_state(
                    source, "Touching", reference
                )
            except AtomicFixtureInventoryError:
                rejected.append(
                    {
                        "source_name_private": source_name,
                        "reference_name_private": reference_name,
                        "reason": "lateral_predicate_unreadable",
                    }
                )
                continue
            if initial_next_to or initial_touching:
                rejected.append(
                    {
                        "source_name_private": source_name,
                        "reference_name_private": reference_name,
                        "reason": "initial_lateral_predicate_true",
                        "initial_next_to": initial_next_to,
                        "initial_touching": initial_touching,
                    }
                )
                continue
            candidates.append(
                (
                    (
                        0
                        if relation == "Touching"
                        and reference_is_task_graspable
                        else 1
                    ),
                    maximum_extent if relation == "Touching" else 0.0,
                    -clearance,
                    reference_name,
                    source_name,
                    source,
                    reference,
                    support,
                    edge_clearances,
                    station_record,
                )
            )
    if candidates:
        candidates.sort(key=lambda item: item[:5])
        (
            reference_priority,
            reference_maximum_extent,
            neg_clearance,
            reference_name,
            source_name,
            source,
            reference,
            support,
            edge_clearances,
            selected_station,
        ) = candidates[0]
        return source, reference, support, {
            "source_name_private": source_name,
            "target_name_private": reference_name,
            "support_name_private": _name(support),
            "initial_next_to": False,
            "initial_touching": False,
            "relation": relation,
            "reference_is_task_graspable": reference_priority == 0,
            "reference_maximum_extent_m": reference_maximum_extent,
            "minimum_edge_clearance_m": -neg_clearance,
            "edge_clearances_m_private": list(edge_clearances),
            "selector_private": "screen_right",
            "selector_runtime_fallback_allowed": True,
            "station_feasibility_query_used": (
                station_feasibility_query is not None
            ),
            "selected_station_feasibility_private": selected_station,
            "rejected_candidates_private": rejected,
        }
    raise AtomicFixtureInventoryError(
        f"live task lacks an initially non-{relation} reference on one unique support",
        diagnostics_private={
            "stage": f"{relation.lower()}_reference_selection",
            "rejected_candidates_private": rejected,
        },
    )


def _select_on_top_support(
    source: Any,
    objects: Mapping[str, Any],
    *,
    excluded_ids: frozenset[int],
    aabb_reader: Callable[[Any], Any],
) -> tuple[Any, Mapping[str, Any]]:
    on_top = _state(source, "OnTop", required=True)
    getter = getattr(on_top, "get_value", None)
    if not callable(getter):
        raise AtomicFixtureInventoryError("OnTop state is unreadable")
    candidates = []
    rejected: list[Mapping[str, str]] = []
    for name, entity in sorted(objects.items()):
        if id(entity) in excluded_ids:
            continue
        if not _fixed_support(entity):
            rejected.append({"name_private": name, "reason": "not_fixed_base"})
            continue
        try:
            rank = _surface_rank(
                source=source, candidate=entity, aabb_reader=aabb_reader
            )
        except AtomicFixtureInventoryError:
            rejected.append({"name_private": name, "reason": "invalid_aabb"})
            continue
        if rank is None:
            rejected.append(
                {"name_private": name, "reason": "surface_geometry_ineligible"}
            )
            continue
        try:
            initially_on_top = getter(entity)
        except Exception:
            rejected.append({"name_private": name, "reason": "relation_read_failed"})
            continue
        if not isinstance(initially_on_top, bool) or initially_on_top:
            rejected.append(
                {"name_private": name, "reason": "initial_on_top_not_false"}
            )
            continue
        candidates.append((rank, name, entity))
    if not candidates:
        raise AtomicFixtureInventoryError(
            "live scene lacks a wide fixed initially-false OnTop support",
            diagnostics_private={
                "stage": "on_top_support_selection",
                "rejected_candidates_private": rejected,
            },
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    rank, name, entity = candidates[0]
    low, high = _read_aabb(entity, aabb_reader)
    return entity, {
        "target_name_private": name,
        "aabb_private": [list(low), list(high)],
        "surface_area_m2": -rank[0],
        "initial_on_top": False,
        "fixed_base": True,
        "stable_support": True,
        "selector_private": "on_top_r1_c1",
        "selector_runtime_probe_performed": False,
        "selector_risk_private": (
            "inventory geometry does not prove runtime marker-cell feasibility; "
            "the execution kernel must revalidate the selector"
        ),
        "rejected_candidates_private": rejected,
    }


def _select_under_reference(
    source: Any,
    objects: Mapping[str, Any],
    *,
    excluded_ids: frozenset[int],
    feasibility_query: Callable[[Any, Any, Any], Mapping[str, Any]] | None,
    aabb_reader: Callable[[Any], Any],
) -> tuple[Any, Mapping[str, Any]]:
    source_low, source_high = _read_aabb(source, aabb_reader)
    source_width = source_high[0] - source_low[0]
    source_depth = source_high[1] - source_low[1]
    under = _state(source, "Under", required=True)
    under_get = getattr(under, "get_value", None)
    if not callable(under_get):
        raise AtomicFixtureInventoryError("Under state is unreadable")
    supports = [
        entity
        for entity in objects.values()
        if id(entity) not in excluded_ids and _fixed_support(entity)
    ]
    candidates = []
    rejected: list[Mapping[str, Any]] = []
    for target_name, target in sorted(objects.items()):
        if id(target) in excluded_ids:
            continue
        try:
            initially_under = under_get(target)
            if not isinstance(initially_under, bool) or initially_under:
                rejected.append(
                    {
                        "target_name_private": target_name,
                        "reason": "initial_under_not_false",
                    }
                )
                continue
            target_low, target_high = _read_aabb(target, aabb_reader)
        except Exception:
            rejected.append(
                {"target_name_private": target_name, "reason": "target_unreadable"}
            )
            continue
        target_width = target_high[0] - target_low[0]
        target_depth = target_high[1] - target_low[1]
        target_center_z = (target_low[2] + target_high[2]) * 0.5
        if (
            not 0.20 <= target_high[2] <= 3.0
            or not 0.10 <= target_center_z <= 1.80
            or target_width < source_width + 0.04
            or target_depth < source_depth + 0.04
        ):
            rejected.append(
                {
                    "target_name_private": target_name,
                    "reason": "target_geometry_ineligible",
                }
            )
            continue
        for support in supports:
            if support is target:
                continue
            support_name = _name(support)
            try:
                support_low, support_high = _read_aabb(support, aabb_reader)
            except AtomicFixtureInventoryError:
                rejected.append(
                    {
                        "target_name_private": target_name,
                        "support_name_private": support_name,
                        "reason": "support_unreadable",
                    }
                )
                continue
            search_low_x = max(target_low[0], support_low[0])
            search_low_y = max(target_low[1], support_low[1])
            search_high_x = min(target_high[0], support_high[0])
            search_high_y = min(target_high[1], support_high[1])
            overlap_x = search_high_x - search_low_x
            overlap_y = search_high_y - search_low_y
            if overlap_x <= 0.0 or overlap_y <= 0.0:
                rejected.append(
                    {
                        "target_name_private": target_name,
                        "support_name_private": support_name,
                        "reason": "no_xy_overlap",
                    }
                )
                continue
            if feasibility_query is None:
                live = {
                    "evidence_kind": "coarse_prefilter_pending_live_kernel",
                    "support": support_name,
                    "candidate": {
                        "target_name_private": target_name,
                        "xy_overlap_m2": overlap_x * overlap_y,
                    },
                }
            else:
                try:
                    live = dict(feasibility_query(source, target, support))
                except Exception:
                    rejected.append(
                        {
                            "target_name_private": target_name,
                            "support_name_private": support_name,
                            "reason": "feasibility_query_failed",
                        }
                    )
                    continue
                if live.get("accepted") is not True:
                    rejected.append(
                        {
                            "target_name_private": target_name,
                            "support_name_private": support_name,
                            "reason": "feasibility_rejected",
                        }
                    )
                    continue
                selector = live.get("selector")
                selected_support = live.get("support")
                selected_candidate = live.get("candidate")
                if (
                    not isinstance(selector, str)
                    or not selector
                    or selected_support != support_name
                    or selected_candidate is None
                ):
                    rejected.append(
                        {
                            "target_name_private": target_name,
                            "support_name_private": support_name,
                            "reason": "incomplete_or_unbound_feasibility_evidence",
                        }
                    )
                    continue
            try:
                json.dumps(live, ensure_ascii=False, allow_nan=False, sort_keys=True)
            except (TypeError, ValueError):
                rejected.append(
                    {
                        "target_name_private": target_name,
                        "support_name_private": support_name,
                        "reason": "non_json_feasibility_evidence",
                    }
                )
                continue
            overlap = overlap_x * overlap_y
            candidates.append(
                (
                    -overlap,
                    target_name,
                    support_name,
                    target,
                    live,
                )
            )
    if not candidates:
        raise AtomicFixtureInventoryError(
            "live scene lacks an eligible Under reference/support candidate",
            diagnostics_private={
                "stage": "under_reference_selection",
                "rejected_candidates_private": rejected,
            },
        )
    candidates.sort(key=lambda item: item[:3])
    neg_overlap, target_name, support_name, target, live = candidates[0]
    return target, {
        "target_name_private": target_name,
        "support_name_private": support_name,
        "initial_under": False,
        "xy_overlap_m2": -neg_overlap,
        "feasibility_private": live,
        "rejected_candidates_private": rejected,
    }


def build_atomic_fixture_inventory(
    *,
    scene_objects: Mapping[str, Any] | Sequence[Any],
    graspables: Mapping[str, Any],
    containers: Mapping[str, Any],
    under_feasibility_query: Callable[[Any, Any, Any], Mapping[str, Any]]
    | None = None,
    station_feasibility_query: Callable[[Any], Mapping[str, Any]] | None = None,
    aabb_reader: Callable[[Any], Any] = _aabb,
) -> AtomicFixtureInventory:
    """Select one deterministic, complete private Action20 fixture set."""

    objects = _named_objects(scene_objects)
    graspable = _role_objects(graspables, role="primary_graspable")
    container = _role_objects(containers, role="primary_container")
    if not set(graspable).issubset(objects) or not set(container).issubset(objects):
        raise AtomicFixtureInventoryError("task roles are absent from live scene")
    if not graspable or not container:
        raise AtomicFixtureInventoryError(
            "complete Action20 inventory needs one graspable and one container"
        )

    source, inside_target = _select_inside_pair(graspable, container)
    lateral_source, lateral_target, _lateral_support, lateral_evidence = (
        _select_lateral_pair(
            graspable,
            objects,
            relation="NextTo",
            aabb_reader=aabb_reader,
            station_feasibility_query=station_feasibility_query,
        )
    )
    excluded_ids = frozenset(
        id(item) for item in (*graspable.values(), *container.values())
    )
    open_target, open_value, open_evidence = (
        _select_capability_target(
            objects,
            state_name="Open",
            aabb_reader=aabb_reader,
            station_feasibility_query=station_feasibility_query,
        )
    )
    toggle_target, toggle_value, toggle_evidence = (
        _select_capability_target(
            objects,
            state_name="ToggledOn",
            aabb_reader=aabb_reader,
            station_feasibility_query=station_feasibility_query,
        )
    )
    on_top_target, on_top_evidence = _select_on_top_support(
        source,
        objects,
        excluded_ids=excluded_ids,
        aabb_reader=aabb_reader,
    )
    under_target, under_evidence = _select_under_reference(
        source,
        objects,
        excluded_ids=excluded_ids,
        feasibility_query=under_feasibility_query,
        aabb_reader=aabb_reader,
    )

    source_name = _name(source)
    inside_name = _name(inside_target)
    open_name = _name(open_target)
    toggle_name = _name(toggle_target)
    on_top_name = _name(on_top_target)
    under_name = _name(under_target)
    lateral_source_name = _name(lateral_source)
    lateral_target_name = _name(lateral_target)
    cards = []
    for action in ACTION20:
        target_name = None
        source_card_name = None
        setup: tuple[str, ...] = ()
        selector = None
        if action == "pick":
            target_name = source_name
        elif action == "open":
            target_name = open_name
            setup = ("target_closed",)
        elif action == "close":
            target_name = open_name
            setup = ("target_open",)
        elif action == "toggle_on":
            target_name = toggle_name
            setup = ("target_off",)
        elif action == "toggle_off":
            target_name = toggle_name
            setup = ("target_on",)
        elif action == "place_on_floor":
            source_card_name = source_name
            setup = ("hold_source",)
        elif action == "place_inside":
            target_name = inside_name
            source_card_name = source_name
            target_open = _state(inside_target, "Open") is not None
            setup = (("target_open",) if target_open else ()) + ("hold_source",)
        elif action == "place_on_top":
            target_name = on_top_name
            source_card_name = source_name
            setup = ("hold_source",)
            selector = "on_top_r1_c1"
        elif action == "place_under":
            target_name = under_name
            source_card_name = source_name
            setup = ("hold_source",)
        elif action == "place_next_to":
            target_name = lateral_target_name
            source_card_name = lateral_source_name
            setup = ("hold_source",)
            selector = "screen_right"
        cards.append(
            AtomicActionCard(
                card_id=action,
                action=action,
                target_name_private=target_name,
                source_name_private=source_card_name,
                private_setup=setup,
                selector_private=selector,
            )
        )
    if tuple(card.action for card in cards) != ACTION20:
        raise AtomicFixtureInventoryError("generated Action20 order drifted")
    validated_cards = normalize_atomic_cards(
        {
            "schema_version": ATOMIC_CARD_SCHEMA_VERSION,
            "cards": [
                {
                    "card_id": card.card_id,
                    "action": card.action,
                    "target_name_private": card.target_name_private,
                    "source_name_private": card.source_name_private,
                    "private_setup": list(card.private_setup),
                    "selector_private": card.selector_private,
                }
                for card in cards
            ],
        }
    )

    return AtomicFixtureInventory(
        cards=validated_cards,
        evidence_private={
            "task_roles": {
                "graspable_names_private": sorted(graspable),
                "container_names_private": sorted(container),
                "inside_source_name_private": source_name,
                "inside_target_name_private": inside_name,
                "initial_inside": False,
            },
            "open": {
                "target_name_private": open_name,
                "initial_value": open_value,
                **dict(open_evidence),
            },
            "toggle": {
                "target_name_private": toggle_name,
                "initial_value": toggle_value,
                **dict(toggle_evidence),
            },
            "on_top": dict(on_top_evidence),
            "under": dict(under_evidence),
            "lateral": dict(lateral_evidence),
        },
    )


__all__ = [
    "ATOMIC_FIXTURE_INVENTORY_VERSION",
    "AtomicFixtureInventory",
    "AtomicFixtureInventoryCache",
    "AtomicFixtureInventoryError",
    "build_atomic_fixture_inventory",
]
