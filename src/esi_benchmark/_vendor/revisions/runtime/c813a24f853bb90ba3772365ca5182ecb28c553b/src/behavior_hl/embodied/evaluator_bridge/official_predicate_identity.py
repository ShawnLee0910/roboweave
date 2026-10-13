"""Stable private identities for official grounded BDDL goal leaves.

BEHAVIOR 2026 exposes each ``ground_goal_state_options`` leaf as a compiled
``HEAD`` wrapper, not necessarily as the inner predicate instance.  The
identity therefore accepts either the direct ``STATE_NAME`` / ``inputs``
form or the wrapper's declarative ``body`` form.  Simulator entities, Python
object addresses, resolved object handles, predicate truth, and positional
indices never enter the identifier.  Reset attestation and live verification
share this module so a reordered, replaced, or duplicated goal leaf cannot
hide behind a positional ``leaf_###`` label.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


class OfficialPredicateIdentityError(RuntimeError):
    """Raised when a grounded goal predicate has no stable private identity."""


def official_predicate_leaf_id(predicate: Any) -> str:
    """Return one stable, opaque ID from declarative predicate metadata."""

    predicate_type = (
        f"{type(predicate).__module__}.{type(predicate).__qualname__}"
    )
    state_name = getattr(predicate, "STATE_NAME", None)
    inputs = getattr(predicate, "inputs", None)
    if (
        isinstance(state_name, str)
        and state_name.strip()
        and isinstance(inputs, (list, tuple))
    ):
        predicate_token = state_name
        declarative = {
            "form": "predicate_state_inputs",
            "state_name": state_name,
            "inputs": _canonical_public_bddl_value(
                list(inputs),
                "inputs",
            ),
        }
    else:
        body = getattr(predicate, "body", None)
        if not isinstance(body, (list, tuple)) or not body:
            raise OfficialPredicateIdentityError(
                "official goal leaf has neither STATE_NAME/inputs nor body"
            )
        predicate_token = _grounded_body_token(body)
        declarative = {
            "form": "compiled_head_body",
            "body": _canonical_public_bddl_value(list(body), "body"),
        }
    canonical = {
        "leaf_wrapper_type": predicate_type,
        "declarative": declarative,
    }
    serialized = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    token = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in predicate_token.casefold()
    ).strip("_")
    if not token:
        token = "predicate"
    return f"official_predicate/v2/{token}/{digest}"


def _grounded_body_token(body: Sequence[Any]) -> str:
    """Extract an auditable predicate token from one grounded HEAD body."""

    current: Any = list(body)
    if (
        len(current) == 2
        and current[0] == "not"
        and isinstance(current[1], (list, tuple))
    ):
        current = list(current[1])
    if (
        not current
        or not isinstance(current[0], str)
        or not current[0].strip()
    ):
        raise OfficialPredicateIdentityError(
            "official grounded body has no predicate token"
        )
    return current[0]


def build_official_goal_leaf_ids(
    options: Sequence[Sequence[Any]],
) -> tuple[tuple[str, ...], ...]:
    """Build ordered IDs and reject duplicate identities within an option."""

    if not isinstance(options, (list, tuple)) or not options:
        raise OfficialPredicateIdentityError(
            "official grounded goal options are unavailable"
        )
    result: list[tuple[str, ...]] = []
    for option_index, option in enumerate(options):
        if not isinstance(option, (list, tuple)) or not option:
            raise OfficialPredicateIdentityError(
                f"official goal option {option_index} has no leaves"
            )
        identities = tuple(
            official_predicate_leaf_id(predicate) for predicate in option
        )
        if len(set(identities)) != len(identities):
            raise OfficialPredicateIdentityError(
                f"official goal option {option_index} has duplicate "
                "predicate identities"
            )
        result.append(identities)
    return tuple(result)


def _canonical_public_bddl_value(value: Any, path: str) -> Any:
    """Accept strict declarative JSON values and reject runtime objects."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # json.dumps(..., allow_nan=False) performs the finite check.
        try:
            json.dumps(value, allow_nan=False)
        except ValueError as exc:
            raise OfficialPredicateIdentityError(
                f"{path} contains a non-finite float"
            ) from exc
        return value
    if isinstance(value, (list, tuple)):
        return [
            _canonical_public_bddl_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise OfficialPredicateIdentityError(
                    f"{path} contains a non-string mapping key"
                )
            result[key] = _canonical_public_bddl_value(
                value[key],
                f"{path}.{key}",
            )
        return result
    raise OfficialPredicateIdentityError(
        f"{path} contains runtime value {type(value).__name__}"
    )
