"""Small, reusable contract for resident Action20 atomic validation.

An atomic card resets one already-loaded scene, establishes only the private
precondition needed by one public action, and then executes that action through
the normal Action20 registry.  Cards are test fixtures, not task solutions and
are never model-visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ATOMIC_CARD_SCHEMA_VERSION = (
    "behavior-hl-bench-try-action20-atomic-cards/v1"
)

ACTION20 = (
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "wait",
    "pick",
    "open",
    "close",
    "toggle_on",
    "toggle_off",
    "place_on_floor",
    "place_inside",
    "place_on_top",
    "place_under",
    "place_next_to",
)
ACTION20_SET = frozenset(ACTION20)
MARKER_ACTIONS = frozenset(
    {
        "pick",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
        "place_inside",
        "place_on_top",
        "place_under",
        "place_next_to",
    }
)
PLACEMENT_ACTIONS = frozenset(
    {
        "place_inside",
        "place_on_floor",
        "place_on_top",
        "place_under",
        "place_next_to",
    }
)
PRIVATE_SETUP_OPERATIONS = frozenset(
    {"hold_source", "target_open", "target_closed", "target_on", "target_off"}
)


@dataclass(frozen=True, slots=True)
class AtomicActionCard:
    card_id: str
    action: str
    target_name_private: str | None
    source_name_private: str | None
    private_setup: tuple[str, ...]
    selector_private: str | None

    def to_private_record(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "action": self.action,
            "target_name_private": self.target_name_private,
            "source_name_private": self.source_name_private,
            "private_setup": list(self.private_setup),
            "selector_private": self.selector_private,
            "private_only": True,
        }


@dataclass(frozen=True, slots=True)
class PreparedAtomicCard:
    """Name-free public handle for one resident-prepared private card."""

    card_id: str
    action: str


def normalize_prepared_atomic_cards(value: Any) -> tuple[PreparedAtomicCard, ...]:
    """Validate the allowlisted response from ``prepare_atomic_inventory``."""

    expected = {
        "status",
        "prepared",
        "idempotent",
        "card_count",
        "card_ids",
        "actions",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("prepared atomic inventory public fields changed")
    identifiers = value.get("card_ids")
    actions = value.get("actions")
    if (
        value.get("status") != "PASS"
        or value.get("prepared") is not True
        or not isinstance(value.get("idempotent"), bool)
        or value.get("card_count") != len(ACTION20)
        or not isinstance(identifiers, list)
        or not isinstance(actions, list)
        or tuple(actions) != ACTION20
        or len(identifiers) != len(ACTION20)
        or len(set(identifiers)) != len(identifiers)
        or any(
            not isinstance(identifier, str)
            or not identifier
            or len(identifier) > 96
            or not identifier.replace("_", "").replace("-", "").isalnum()
            for identifier in identifiers
        )
    ):
        raise ValueError("prepared atomic inventory is invalid")
    return tuple(
        PreparedAtomicCard(card_id=identifier, action=action)
        for identifier, action in zip(identifiers, actions, strict=True)
    )


def _optional_name(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 160:
        raise ValueError(f"atomic card {field} is invalid")
    return value


def normalize_atomic_cards(value: Any) -> tuple[AtomicActionCard, ...]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "cards",
    }:
        raise ValueError("atomic card rule fields changed")
    if value.get("schema_version") != ATOMIC_CARD_SCHEMA_VERSION:
        raise ValueError("atomic card schema changed")
    records = value.get("cards")
    if not isinstance(records, list) or not 1 <= len(records) <= len(ACTION20):
        raise ValueError("atomic card list is invalid")
    expected = {
        "card_id",
        "action",
        "target_name_private",
        "source_name_private",
        "private_setup",
        "selector_private",
    }
    cards: list[AtomicActionCard] = []
    identifiers: set[str] = set()
    actions: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ValueError("atomic card fields changed")
        card_id = record.get("card_id")
        action = record.get("action")
        setup = record.get("private_setup")
        if (
            not isinstance(card_id, str)
            or not card_id
            or len(card_id) > 96
            or not card_id.replace("_", "").replace("-", "").isalnum()
            or card_id in identifiers
            or action not in ACTION20_SET
            or action in actions
            or not isinstance(setup, list)
            or len(set(setup)) != len(setup)
            or any(item not in PRIVATE_SETUP_OPERATIONS for item in setup)
        ):
            raise ValueError("atomic card identity or setup is invalid")
        target = _optional_name(
            record.get("target_name_private"), field="target name"
        )
        source = _optional_name(
            record.get("source_name_private"), field="source name"
        )
        selector = _optional_name(
            record.get("selector_private"), field="selector"
        )
        if (
            (action in MARKER_ACTIONS) is not (target is not None)
            or ("hold_source" in setup) is not (source is not None)
            or (action in PLACEMENT_ACTIONS and "hold_source" not in setup)
            or (
                action not in {
                    "place_inside",
                    "place_on_top",
                    "place_next_to",
                }
                and selector is not None
            )
            or (action == "place_next_to" and selector is None)
            or any(
                item.startswith("target_") for item in setup
            ) and target is None
        ):
            raise ValueError(f"atomic card precondition is invalid: {card_id}")
        identifiers.add(card_id)
        actions.add(str(action))
        cards.append(
            AtomicActionCard(
                card_id=card_id,
                action=str(action),
                target_name_private=target,
                source_name_private=source,
                private_setup=tuple(str(item) for item in setup),
                selector_private=selector,
            )
        )
    return tuple(cards)


def marker_number_for_binding(
    snapshot: Any,
    *,
    target: Any | None = None,
    selector: str | None = None,
    binding_kinds: Sequence[str] = ("object", "part", "surface_cell"),
    require_unique: bool = True,
    fallback_to_any_selector: bool = False,
) -> int:
    """Resolve one current Marker by private identity, never by a fixed C number.

    Atomic smoke cards may prefer a spatial selector for stable evidence, but
    the selector is not the Action20 capability being certified.  A
    current-frame candidate for the same target and binding kind is therefore
    an admissible fallback when that preferred cell or side is unavailable.
    """

    bindings = getattr(snapshot, "bindings", None)
    if not isinstance(bindings, Mapping):
        raise RuntimeError("atomic Marker snapshot is unavailable")
    matches = []
    for marker, binding in bindings.items():
        kind = str(getattr(binding, "binding_kind", "object"))
        if kind not in set(binding_kinds):
            continue
        if target is not None:
            token = getattr(binding, "target_token", None)
            for _ in range(2):
                nested = getattr(token, "target_token", None)
                if nested is None:
                    break
                token = nested
            target_name = str(getattr(target, "name", ""))
            if token is not target and str(
                getattr(binding, "object_name", "")
            ) != target_name:
                continue
        if selector is not None and getattr(binding, "selector", None) != selector:
            continue
        if not isinstance(marker, str) or not marker.startswith("C"):
            continue
        suffix = marker[1:]
        if suffix.isdigit() and int(suffix) >= 1:
            matches.append(int(suffix))
    if not matches and selector is not None and fallback_to_any_selector:
        return marker_number_for_binding(
            snapshot,
            target=target,
            selector=None,
            binding_kinds=binding_kinds,
            require_unique=False,
            fallback_to_any_selector=False,
        )
    if not matches or (require_unique and len(matches) != 1):
        raise RuntimeError(
            f"atomic Marker binding is not unique: count={len(matches)}"
        )
    return min(matches)


__all__ = [
    "ACTION20",
    "ATOMIC_CARD_SCHEMA_VERSION",
    "AtomicActionCard",
    "MARKER_ACTIONS",
    "PLACEMENT_ACTIONS",
    "PreparedAtomicCard",
    "marker_number_for_binding",
    "normalize_atomic_cards",
    "normalize_prepared_atomic_cards",
]
