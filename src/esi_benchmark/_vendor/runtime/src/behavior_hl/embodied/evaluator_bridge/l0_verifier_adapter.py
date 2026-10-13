"""CPU-pure aggregation for projected L0 goal clauses.

The adapter never decides whether a spatial predicate is true.  Callers pass
one of the existing official-leaf readers or a canonical-kernel callback with
this signature::

    reader(predicate: str, target: str, destination: str | None) -> bool

``official_full`` tasks bypass this class and use the existing
``OfficialPredicateVerifier``.  The other two routes share the quantifier,
initial-credit, stability, and fail-closed behavior implemented here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import re
from typing import Any


OFFICIAL_FULL = "official_full"
OFFICIAL_PROJECTION = "official_projection"
CANONICAL_FALLBACK = "canonical_fallback"
PROJECTED_ROUTES = frozenset({OFFICIAL_PROJECTION, CANONICAL_FALLBACK})

_FIXED_QUANTIFIERS = frozenset(
    {
        "ALL_LISTED",
        "ALL_LISTED_TO_ONE_DESTINATION",
        "ALL_TEMPLATE_ROOM_CATEGORY_MEMBERS",
        "EXACTLY_1",
        "AT_LEAST_1_PUBLIC_EQUIVALENT_PAIR",
    }
)
_AT_LEAST = re.compile(r"AT_LEAST_(\d+)(?:_TO_ONE_DESTINATION)?\Z")
_SHARED_DESTINATION = re.compile(
    r"(?:ALL_LISTED_TO_ONE_DESTINATION|ALL_TEMPLATE_ROOM_CATEGORY_MEMBERS|"
    r"AT_LEAST_\d+_TO_ONE_DESTINATION|AT_LEAST_1_PUBLIC_EQUIVALENT_PAIR)\Z"
)

TruthReader = Callable[[str, str, str | None], bool]

_NO_DESTINATION_EQUIVALENCE = "NONE"
_ROOM_CATEGORY_EQUIVALENCE = "ANY_TEMPLATE_ROOM_CATEGORY_EQUIVALENT"


def _string_list(value: Any, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or (not value and not allow_empty)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{field} must be {qualifier} of non-empty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicate scopes")
    return result


def _required_count(quantifier: str, target_count: int) -> int:
    if quantifier in {
        "ALL_LISTED",
        "ALL_LISTED_TO_ONE_DESTINATION",
        "ALL_TEMPLATE_ROOM_CATEGORY_MEMBERS",
    }:
        return target_count
    if quantifier in {"EXACTLY_1", "AT_LEAST_1_PUBLIC_EQUIVALENT_PAIR"}:
        return 1
    match = _AT_LEAST.fullmatch(quantifier)
    if match is None:
        raise ValueError(f"unsupported quantifier: {quantifier!r}")
    required = int(match.group(1))
    if required < 1 or required > target_count:
        raise ValueError(
            f"quantifier {quantifier!r} cannot be satisfied by {target_count} targets"
        )
    return required


def _candidate_destinations(
    clause: Mapping[str, Any],
    destinations: tuple[str, ...],
    index: int,
) -> tuple[str | None, ...]:
    policy = clause.get(
        "destination_equivalence_policy",
        _NO_DESTINATION_EQUIVALENCE,
    )
    if policy == _NO_DESTINATION_EQUIVALENCE:
        members = clause.get("destination_equivalence_members")
        if members is not None and set(
            _string_list(
                members,
                "destination_equivalence_members",
                allow_empty=True,
            )
        ) != set(destinations):
            raise ValueError(
                f"clause {index} NONE destination equivalence must match "
                "destination_scopes"
            )
        return destinations or (None,)
    if policy != _ROOM_CATEGORY_EQUIVALENCE:
        raise ValueError(
            f"clause {index} has unsupported destination equivalence policy {policy!r}"
        )
    if not destinations:
        raise ValueError(
            f"clause {index} room/category destination equivalence requires destinations"
        )
    members = _string_list(
        clause.get("destination_equivalence_members"),
        "destination_equivalence_members",
        allow_empty=False,
    )
    if not set(destinations).issubset(members):
        raise ValueError(
            f"clause {index} destination equivalence members must include "
            "destination_scopes"
        )
    return members


def _read_truth(reader: TruthReader, predicate: str, target: str, destination: str | None) -> bool:
    truth = reader(predicate, target, destination)
    if not isinstance(truth, bool):
        raise TypeError(
            "truth reader must return bool for "
            f"({predicate!r}, {target!r}, {destination!r})"
        )
    return truth


def _evaluate_clause(
    clause: Mapping[str, Any],
    reader: TruthReader,
    index: int,
) -> dict[str, Any]:
    predicate = clause.get("canonical_predicate")
    quantifier = clause.get("quantifier")
    if not isinstance(predicate, str) or not predicate:
        raise ValueError(f"clause {index} lacks canonical_predicate")
    if not isinstance(quantifier, str) or (
        quantifier not in _FIXED_QUANTIFIERS
        and _AT_LEAST.fullmatch(quantifier) is None
    ):
        raise ValueError(f"clause {index} has unsupported quantifier {quantifier!r}")

    targets = _string_list(clause.get("target_scopes"), "target_scopes", allow_empty=False)
    destinations = _string_list(
        clause.get("destination_scopes"),
        "destination_scopes",
        allow_empty=True,
    )
    required = _required_count(quantifier, len(targets))

    if (
        quantifier.endswith("_TO_ONE_DESTINATION")
        or quantifier == "AT_LEAST_1_PUBLIC_EQUIVALENT_PAIR"
    ) and not destinations:
        raise ValueError(f"clause {index} quantifier {quantifier!r} requires destinations")

    candidate_destinations = _candidate_destinations(clause, destinations, index)
    per_destination = []
    per_target_any = {target: False for target in targets}
    for destination in candidate_destinations:
        matched_targets = []
        for target in targets:
            truth = _read_truth(reader, predicate, target, destination)
            per_target_any[target] = per_target_any[target] or truth
            if truth:
                matched_targets.append(target)
        per_destination.append((destination, tuple(matched_targets)))

    shared_destination = bool(destinations) and bool(
        _SHARED_DESTINATION.fullmatch(quantifier)
    )
    if shared_destination:
        matched_destination, matched_targets = max(
            per_destination,
            key=lambda item: (len(item[1]), str(item[0])),
        )
        matched_count = len(matched_targets)
    else:
        matched_destination = None
        matched_targets = tuple(
            target for target in targets if per_target_any[target]
        )
        matched_count = len(matched_targets)

    if quantifier == "EXACTLY_1":
        satisfied = matched_count == 1
    else:
        satisfied = matched_count >= required

    return {
        "index": index,
        "canonical_predicate": predicate,
        "quantifier": quantifier,
        "required_count": required,
        "matched_count": matched_count,
        "matched_targets": list(matched_targets),
        "matched_destination": matched_destination,
        "satisfied": satisfied,
        "_true_by_destination": tuple(per_destination),
    }


def _credit_report(
    initial: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Return identity-sensitive new credit for one quantified clause."""

    required = int(current["required_count"])
    quantifier = str(current["quantifier"])
    initial_by_destination = {
        destination: frozenset(targets)
        for destination, targets in initial["_true_by_destination"]
    }
    current_by_destination = {
        destination: frozenset(targets)
        for destination, targets in current["_true_by_destination"]
    }

    if quantifier == "AT_LEAST_1_PUBLIC_EQUIVALENT_PAIR":
        initial_pairs = {
            (target, destination)
            for destination, targets in initial_by_destination.items()
            for target in targets
        }
        current_pairs = {
            (target, destination)
            for destination, targets in current_by_destination.items()
            for target in targets
        }
        new_pairs = sorted(
            current_pairs - initial_pairs,
            key=lambda pair: (pair[0], str(pair[1])),
        )[:required]
        credit_destination = new_pairs[0][1] if new_pairs else None
    elif bool(current["matched_destination"] is not None) and bool(
        _SHARED_DESTINATION.fullmatch(quantifier)
    ):
        credit_destination = max(
            current_by_destination,
            key=lambda destination: (
                min(
                    len(
                        current_by_destination[destination]
                        - initial_by_destination[destination]
                    ),
                    required,
                ),
                str(destination),
            ),
        )
        initial_targets = initial_by_destination[credit_destination]
        new_pairs = [
            (target, credit_destination)
            for target in sorted(
                current_by_destination[credit_destination] - initial_targets
            )[:required]
        ]
    else:
        new_pairs = []
        for destination in sorted(current_by_destination, key=lambda value: str(value)):
            for target in sorted(
                current_by_destination[destination]
                - initial_by_destination[destination]
            ):
                if all(existing_target != target for existing_target, _ in new_pairs):
                    new_pairs.append((target, destination))
        new_pairs = new_pairs[:required]
        credit_destination = None

    return {
        "eligible_count": required,
        "new_credit": len(new_pairs),
        "credit_destination": credit_destination,
        "new_credit_relations": [
            {"target": target, "destination": destination}
            for target, destination in new_pairs
        ],
    }


class ProjectedGoalVerifier:
    """Aggregate official or canonical relation truth without implementing it."""

    def __init__(
        self,
        goal_spec: Mapping[str, Any],
        initial_reader: TruthReader,
        *,
        stability_window: int = 1,
    ) -> None:
        self._stable_streak = 0
        self._regressions: set[tuple[int, str, str | None]] = set()
        self._previous_reports: list[dict[str, Any]] | None = None
        self._configuration_errors: list[str] = []
        self._initial_reports: list[dict[str, Any]] | None = None
        self._initial_goal_satisfied = False

        if (
            isinstance(stability_window, bool)
            or not isinstance(stability_window, int)
            or stability_window < 1
        ):
            self._configuration_error("stability_window must be a positive integer")
            self._stability_window = 1
        else:
            self._stability_window = stability_window

        try:
            if not isinstance(goal_spec, Mapping):
                raise TypeError("goal_spec must be a mapping")
            self.route = goal_spec.get("route")
            if self.route == OFFICIAL_FULL:
                raise ValueError(
                    "official_full must use the existing OfficialPredicateVerifier"
                )
            if self.route not in PROJECTED_ROUTES:
                raise ValueError(f"unsupported verifier route: {self.route!r}")
            clauses = goal_spec.get("clauses")
            if (
                not isinstance(clauses, Sequence)
                or isinstance(clauses, (str, bytes))
                or not clauses
                or any(not isinstance(clause, Mapping) for clause in clauses)
            ):
                raise ValueError("projected goal_spec clauses must be a non-empty list")
            self._clauses = tuple(clauses)
            if not callable(initial_reader):
                raise TypeError("initial_reader must be callable")
            self._initial_reports = self._read(initial_reader)
            self._initial_goal_satisfied = all(
                report["satisfied"] for report in self._initial_reports
            )
            self._previous_reports = self._initial_reports
        except Exception as exc:
            self._configuration_error(f"{type(exc).__name__}: {exc}")
            self._clauses = tuple(getattr(self, "_clauses", ()))

    def _configuration_error(self, error: str) -> None:
        self._configuration_errors.append(error)

    def _read(self, reader: TruthReader) -> list[dict[str, Any]]:
        return [
            _evaluate_clause(clause, reader, index)
            for index, clause in enumerate(self._clauses)
        ]

    def _stability(self, current_goal_satisfied: bool = False) -> dict[str, Any]:
        return {
            "required": self._stability_window,
            "consecutive": self._stable_streak,
            "current_goal_satisfied": current_goal_satisfied,
            "complete": self._stable_streak >= self._stability_window,
        }

    def _regression_report(self) -> list[dict[str, Any]]:
        return [
            {"clause_index": index, "target": target, "destination": destination}
            for index, target, destination in sorted(
                self._regressions,
                key=lambda item: (item[0], item[1], str(item[2])),
            )
        ]

    def _fail_closed(self, exc: Exception | None, done_requested: bool) -> dict[str, Any]:
        self._stable_streak = 0
        errors = list(self._configuration_errors)
        if exc is not None:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(error)
        return {
            "operational": False,
            "success": False,
            "q_score": 0.0,
            "initial_goal_satisfied": self._initial_goal_satisfied,
            "current_goal_satisfied": False,
            "clause_reports": [],
            "stability": self._stability(),
            "regressions": self._regression_report(),
            "early_done": bool(done_requested),
            "errors": errors,
        }

    def snapshot(
        self,
        reader: TruthReader,
        *,
        done_requested: bool = False,
    ) -> dict[str, Any]:
        """Return one private, fail-closed verifier snapshot."""

        if self._configuration_errors or self._initial_reports is None:
            return self._fail_closed(None, done_requested)
        try:
            if not callable(reader):
                raise TypeError("truth reader must be callable")
            reports = self._read(reader)
        except Exception as exc:
            return self._fail_closed(exc, done_requested)

        assert self._previous_reports is not None
        for previous, current in zip(self._previous_reports, reports, strict=True):
            previous_pairs = {
                (target, destination)
                for destination, targets in previous["_true_by_destination"]
                for target in targets
            }
            current_pairs = {
                (target, destination)
                for destination, targets in current["_true_by_destination"]
                for target in targets
            }
            self._regressions.update(
                (current["index"], target, destination)
                for target, destination in previous_pairs - current_pairs
            )
        self._previous_reports = reports

        current_goal_satisfied = all(report["satisfied"] for report in reports)
        stable_candidate = current_goal_satisfied and not self._initial_goal_satisfied
        self._stable_streak = self._stable_streak + 1 if stable_candidate else 0
        stability = self._stability(current_goal_satisfied)
        success = stable_candidate and stability["complete"]

        numerator = 0
        denominator = 0
        rendered_reports = []
        for initial, current in zip(self._initial_reports, reports, strict=True):
            credit = _credit_report(initial, current)
            numerator += credit["new_credit"]
            denominator += current["required_count"]
            rendered_reports.append(
                {
                    **{
                        key: value
                        for key, value in current.items()
                        if not key.startswith("_")
                    },
                    "initial_matched_count": initial["matched_count"],
                    "required_slot_count": current["required_count"],
                    **credit,
                }
            )

        return {
            "operational": True,
            "success": bool(success),
            "q_score": 1.0 if success else numerator / denominator if denominator else 0.0,
            "initial_goal_satisfied": self._initial_goal_satisfied,
            "current_goal_satisfied": current_goal_satisfied,
            "clause_reports": rendered_reports,
            "stability": stability,
            "regressions": self._regression_report(),
            "early_done": bool(done_requested and not success),
            "errors": [],
        }
