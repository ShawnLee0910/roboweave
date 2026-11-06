"""Lazily ground one deterministic BDDL goal option without OmniGibson.

The official BDDL implementation materializes every satisfying ground option.
That is unnecessary for an Oracle author and becomes exponential for goals
such as ``forall item: exists cabinet``.  This module follows the official
quantifier semantics but yields options lazily, allowing callers to stop at
the first structurally feasible option.
"""

from __future__ import annotations

from itertools import combinations, permutations
import re
from typing import Callable, Iterable, Iterator, Mapping, Sequence


class CanonicalGroundingError(ValueError):
    """Raised when a BDDL expression cannot be grounded safely."""


def tokenize_bddl(text: str) -> list[str]:
    clean = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    return re.findall(r"\(|\)|[^\s()]+", clean)


def parse_bddl(text: str) -> list[object]:
    tokens = tokenize_bddl(text)
    cursor = 0

    def parse_one() -> object:
        nonlocal cursor
        if cursor >= len(tokens):
            raise CanonicalGroundingError("unexpected end of BDDL")
        token = tokens[cursor]
        cursor += 1
        if token != "(":
            if token == ")":
                raise CanonicalGroundingError("unexpected closing parenthesis")
            return token
        result: list[object] = []
        while cursor < len(tokens) and tokens[cursor] != ")":
            result.append(parse_one())
        if cursor >= len(tokens):
            raise CanonicalGroundingError("unclosed BDDL expression")
        cursor += 1
        return result

    result: list[object] = []
    while cursor < len(tokens):
        result.append(parse_one())
    return result


def find_section(define_expression: Sequence[object], name: str) -> list[object]:
    for item in define_expression:
        if isinstance(item, list) and item and item[0] == name:
            return item
    raise CanonicalGroundingError(f"missing BDDL section {name!r}")


def problem_goal(text: str) -> list[object]:
    expressions = parse_bddl(text)
    if len(expressions) != 1 or not isinstance(expressions[0], list):
        raise CanonicalGroundingError("expected one BDDL define expression")
    goal_section = find_section(expressions[0], ":goal")
    # Match bddl.parsing.parse_problem: only the first formula after :goal is
    # consumed.  A malformed trailing formula must not silently change runtime.
    if len(goal_section) < 2 or not isinstance(goal_section[1], list):
        raise CanonicalGroundingError("BDDL :goal has no formula")
    return goal_section[1]


def _natural_object_key(name: str) -> tuple[str, int, str]:
    stem, separator, suffix = name.rpartition("_")
    return (stem, int(suffix), name) if separator and suffix.isdigit() else (name, -1, name)


def object_map_from_scope(scope_names: Iterable[str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for name in scope_names:
        if name == "robot_poses":
            continue
        synset, separator, suffix = name.rpartition("_")
        if not separator or not suffix.isdigit():
            raise CanonicalGroundingError(f"invalid scope object {name!r}")
        grouped.setdefault(synset, []).append(name)
    return {
        synset: tuple(sorted(names, key=_natural_object_key))
        for synset, names in sorted(grouped.items())
    }


GroundLeaf = list[object]
GroundOption = list[GroundLeaf]
AtomicGroundLeafGate = Callable[[GroundLeaf], bool]


def _declaration(value: object) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 3 or value[1] != "-":
        raise CanonicalGroundingError(f"invalid quantifier declaration {value!r}")
    variable, category = value[0], value[2]
    if not isinstance(variable, str) or not variable.startswith("?") or not isinstance(category, str):
        raise CanonicalGroundingError(f"invalid quantifier declaration {value!r}")
    return variable, category


def _substitute(value: object, bindings: Mapping[str, str]) -> object:
    if isinstance(value, str):
        if value in bindings:
            return bindings[value]
        # BDDL writes both quantified variables and fixed object references
        # with a leading '?'.  Quantified variables were resolved above;
        # unresolved names are therefore fixed scope IDs.
        return value[1:] if value.startswith("?") else value
    if isinstance(value, list):
        return [_substitute(item, bindings) for item in value]
    raise CanonicalGroundingError(f"unexpected BDDL token {value!r}")


def _product_options(
    expressions: Sequence[object],
    object_map: Mapping[str, Sequence[str]],
    bindings: Mapping[str, str],
) -> Iterator[GroundOption]:
    if not expressions:
        yield []
        return
    head, *tail = expressions
    for left in iter_ground_options(head, object_map, bindings):
        for right in _product_options(tail, object_map, bindings):
            yield left + right


def _objects(object_map: Mapping[str, Sequence[str]], category: str) -> tuple[str, ...]:
    result = tuple(object_map.get(category, ()))
    if not result:
        raise CanonicalGroundingError(f"no concrete objects for category {category!r}")
    return result


def iter_ground_options(
    expression: object,
    object_map: Mapping[str, Sequence[str]],
    bindings: Mapping[str, str] | None = None,
) -> Iterator[GroundOption]:
    """Yield satisfying atomic options in official order, without materializing all."""

    if bindings is None:
        bindings = {}
    if not isinstance(expression, list) or not expression or not isinstance(expression[0], str):
        raise CanonicalGroundingError(f"invalid BDDL expression {expression!r}")
    operator = expression[0]
    if operator == "and":
        yield from _product_options(expression[1:], object_map, bindings)
        return
    if operator == "or":
        for child in expression[1:]:
            yield from iter_ground_options(child, object_map, bindings)
        return
    if operator == "not":
        if len(expression) != 2 or not isinstance(expression[1], list):
            raise CanonicalGroundingError("only well-formed negation is supported")
        child = expression[1]
        if child and child[0] in {"and", "or", "forall", "exists", "forn", "forpairs", "fornpairs", "not"}:
            raise CanonicalGroundingError("compound negation is outside the first-tier contract")
        yield [["not", _substitute(child, bindings)]]
        return
    if operator in {"forall", "exists"}:
        if len(expression) != 3:
            raise CanonicalGroundingError(f"invalid {operator} expression")
        variable, category = _declaration(expression[1])
        instances = _objects(object_map, category)
        if operator == "exists":
            for instance in instances:
                yield from iter_ground_options(
                    expression[2], object_map, {**bindings, variable: instance}
                )
        else:
            bound_expressions = [
                (expression[2], {**bindings, variable: instance})
                for instance in instances
            ]

            def universal_product(index: int) -> Iterator[GroundOption]:
                if index == len(bound_expressions):
                    yield []
                    return
                child, child_bindings = bound_expressions[index]
                for left in iter_ground_options(child, object_map, child_bindings):
                    for right in universal_product(index + 1):
                        yield left + right

            yield from universal_product(0)
        return
    if operator == "forn":
        if len(expression) != 4 or not isinstance(expression[1], list) or len(expression[1]) != 1:
            raise CanonicalGroundingError("invalid forn expression")
        count = int(expression[1][0])
        variable, category = _declaration(expression[2])
        instances = _objects(object_map, category)
        if count < 0 or count > len(instances):
            raise CanonicalGroundingError("forn count exceeds available objects")
        for chosen in combinations(instances, count):
            bound = [
                (expression[3], {**bindings, variable: instance})
                for instance in chosen
            ]

            def chosen_product(index: int) -> Iterator[GroundOption]:
                if index == len(bound):
                    yield []
                    return
                child, child_bindings = bound[index]
                for left in iter_ground_options(child, object_map, child_bindings):
                    for right in chosen_product(index + 1):
                        yield left + right

            yield from chosen_product(0)
        return
    if operator in {"forpairs", "fornpairs"}:
        offset = 1
        pair_count: int | None = None
        if operator == "fornpairs":
            if not isinstance(expression[1], list) or len(expression[1]) != 1:
                raise CanonicalGroundingError("invalid fornpairs count")
            pair_count = int(expression[1][0])
            offset = 2
        expected_len = offset + 3
        if len(expression) != expected_len:
            raise CanonicalGroundingError(f"invalid {operator} expression")
        variable1, category1 = _declaration(expression[offset])
        variable2, category2 = _declaration(expression[offset + 1])
        body = expression[offset + 2]
        instances1 = _objects(object_map, category1)
        instances2 = _objects(object_map, category2)
        count = min(len(instances1), len(instances2)) if pair_count is None else pair_count
        if count < 0 or count > min(len(instances1), len(instances2)):
            raise CanonicalGroundingError(f"{operator} count exceeds available objects")
        for selected1 in combinations(instances1, count):
            for selected2 in permutations(instances2, count):
                if any(left == right for left, right in zip(selected1, selected2)):
                    continue
                bound = [
                    (body, {**bindings, variable1: left, variable2: right})
                    for left, right in zip(selected1, selected2)
                ]

                def pair_product(index: int) -> Iterator[GroundOption]:
                    if index == len(bound):
                        yield []
                        return
                    child, child_bindings = bound[index]
                    for left in iter_ground_options(child, object_map, child_bindings):
                        for right in pair_product(index + 1):
                            yield left + right

                yield from pair_product(0)
        return
    # Atomic predicate.
    yield [[_substitute(item, bindings) for item in expression]]


def _ground_option_count(
    expression: object,
    object_map: Mapping[str, Sequence[str]],
    bindings: Mapping[str, str],
) -> int:
    """Count the unfiltered official-order options for one expression.

    The count is used only to retain the *raw* official option index while an
    authoring-only atomic gate prunes impossible branches.  It never
    materializes an option table.
    """

    if (
        not isinstance(expression, list)
        or not expression
        or not isinstance(expression[0], str)
    ):
        raise CanonicalGroundingError(f"invalid BDDL expression {expression!r}")
    operator = expression[0]
    if operator == "and":
        result = 1
        for child in expression[1:]:
            result *= _ground_option_count(child, object_map, bindings)
        return result
    if operator == "or":
        return sum(
            _ground_option_count(child, object_map, bindings)
            for child in expression[1:]
        )
    if operator == "not":
        if len(expression) != 2 or not isinstance(expression[1], list):
            raise CanonicalGroundingError("only well-formed negation is supported")
        child = expression[1]
        if child and child[0] in {
            "and",
            "or",
            "forall",
            "exists",
            "forn",
            "forpairs",
            "fornpairs",
            "not",
        }:
            raise CanonicalGroundingError(
                "compound negation is outside the first-tier contract"
            )
        return 1
    if operator in {"forall", "exists"}:
        if len(expression) != 3:
            raise CanonicalGroundingError(f"invalid {operator} expression")
        variable, category = _declaration(expression[1])
        instances = _objects(object_map, category)
        child_counts = tuple(
            _ground_option_count(
                expression[2],
                object_map,
                {**bindings, variable: instance},
            )
            for instance in instances
        )
        if operator == "exists":
            return sum(child_counts)
        result = 1
        for child_count in child_counts:
            result *= child_count
        return result
    if operator == "forn":
        if (
            len(expression) != 4
            or not isinstance(expression[1], list)
            or len(expression[1]) != 1
        ):
            raise CanonicalGroundingError("invalid forn expression")
        count = int(expression[1][0])
        variable, category = _declaration(expression[2])
        instances = _objects(object_map, category)
        if count < 0 or count > len(instances):
            raise CanonicalGroundingError("forn count exceeds available objects")
        total = 0
        for chosen in combinations(instances, count):
            product = 1
            for instance in chosen:
                product *= _ground_option_count(
                    expression[3],
                    object_map,
                    {**bindings, variable: instance},
                )
            total += product
        return total
    if operator in {"forpairs", "fornpairs"}:
        offset = 1
        pair_count: int | None = None
        if operator == "fornpairs":
            if not isinstance(expression[1], list) or len(expression[1]) != 1:
                raise CanonicalGroundingError("invalid fornpairs count")
            pair_count = int(expression[1][0])
            offset = 2
        if len(expression) != offset + 3:
            raise CanonicalGroundingError(f"invalid {operator} expression")
        variable1, category1 = _declaration(expression[offset])
        variable2, category2 = _declaration(expression[offset + 1])
        body = expression[offset + 2]
        instances1 = _objects(object_map, category1)
        instances2 = _objects(object_map, category2)
        count = (
            min(len(instances1), len(instances2))
            if pair_count is None
            else pair_count
        )
        if count < 0 or count > min(len(instances1), len(instances2)):
            raise CanonicalGroundingError(f"{operator} count exceeds available objects")
        total = 0
        for selected1 in combinations(instances1, count):
            for selected2 in permutations(instances2, count):
                if any(left == right for left, right in zip(selected1, selected2)):
                    continue
                product = 1
                for left, right in zip(selected1, selected2):
                    product *= _ground_option_count(
                        body,
                        object_map,
                        {**bindings, variable1: left, variable2: right},
                    )
                total += product
        return total
    return 1


def _indexed_product_options(
    expressions: Sequence[tuple[object, Mapping[str, str]]],
    object_map: Mapping[str, Sequence[str]],
    atomic_leaf_gate: AtomicGroundLeafGate | None,
) -> Iterator[tuple[int, GroundOption]]:
    if not expressions:
        yield 0, []
        return
    (head, head_bindings), *tail = expressions
    tail_total = 1
    for child, child_bindings in tail:
        tail_total *= _ground_option_count(child, object_map, child_bindings)
    for left_index, left in iter_indexed_ground_options(
        head,
        object_map,
        head_bindings,
        atomic_leaf_gate=atomic_leaf_gate,
    ):
        for right_index, right in _indexed_product_options(
            tail,
            object_map,
            atomic_leaf_gate,
        ):
            yield left_index * tail_total + right_index, left + right


def iter_indexed_ground_options(
    expression: object,
    object_map: Mapping[str, Sequence[str]],
    bindings: Mapping[str, str] | None = None,
    *,
    atomic_leaf_gate: AtomicGroundLeafGate | None = None,
) -> Iterator[tuple[int, GroundOption]]:
    """Yield admitted options with their unchanged official raw index.

    ``atomic_leaf_gate`` is an authoring-only pruning hook.  A rejected atom
    removes its whole branch before Cartesian products are expanded, while
    raw indices continue to count every unfiltered official branch.  With no
    gate this iterator is exactly ``enumerate(iter_ground_options(...))``.
    """

    if bindings is None:
        bindings = {}
    if atomic_leaf_gate is not None and not callable(atomic_leaf_gate):
        raise TypeError("atomic_leaf_gate must be callable")
    if (
        not isinstance(expression, list)
        or not expression
        or not isinstance(expression[0], str)
    ):
        raise CanonicalGroundingError(f"invalid BDDL expression {expression!r}")
    operator = expression[0]
    if operator == "and":
        yield from _indexed_product_options(
            tuple((child, bindings) for child in expression[1:]),
            object_map,
            atomic_leaf_gate,
        )
        return
    if operator == "or":
        raw_offset = 0
        for child in expression[1:]:
            for child_index, option in iter_indexed_ground_options(
                child,
                object_map,
                bindings,
                atomic_leaf_gate=atomic_leaf_gate,
            ):
                yield raw_offset + child_index, option
            raw_offset += _ground_option_count(child, object_map, bindings)
        return
    if operator == "not":
        if len(expression) != 2 or not isinstance(expression[1], list):
            raise CanonicalGroundingError("only well-formed negation is supported")
        child = expression[1]
        if child and child[0] in {
            "and",
            "or",
            "forall",
            "exists",
            "forn",
            "forpairs",
            "fornpairs",
            "not",
        }:
            raise CanonicalGroundingError(
                "compound negation is outside the first-tier contract"
            )
        leaf = ["not", _substitute(child, bindings)]
        if atomic_leaf_gate is None or atomic_leaf_gate(leaf):
            yield 0, [leaf]
        return
    if operator in {"forall", "exists"}:
        if len(expression) != 3:
            raise CanonicalGroundingError(f"invalid {operator} expression")
        variable, category = _declaration(expression[1])
        instances = _objects(object_map, category)
        bound = tuple(
            (expression[2], {**bindings, variable: instance})
            for instance in instances
        )
        if operator == "forall":
            yield from _indexed_product_options(
                bound,
                object_map,
                atomic_leaf_gate,
            )
            return
        raw_offset = 0
        for child, child_bindings in bound:
            for child_index, option in iter_indexed_ground_options(
                child,
                object_map,
                child_bindings,
                atomic_leaf_gate=atomic_leaf_gate,
            ):
                yield raw_offset + child_index, option
            raw_offset += _ground_option_count(
                child,
                object_map,
                child_bindings,
            )
        return
    if operator == "forn":
        if (
            len(expression) != 4
            or not isinstance(expression[1], list)
            or len(expression[1]) != 1
        ):
            raise CanonicalGroundingError("invalid forn expression")
        count = int(expression[1][0])
        variable, category = _declaration(expression[2])
        instances = _objects(object_map, category)
        if count < 0 or count > len(instances):
            raise CanonicalGroundingError("forn count exceeds available objects")
        raw_offset = 0
        for chosen in combinations(instances, count):
            bound = tuple(
                (expression[3], {**bindings, variable: instance})
                for instance in chosen
            )
            group_total = 1
            for child, child_bindings in bound:
                group_total *= _ground_option_count(
                    child,
                    object_map,
                    child_bindings,
                )
            for child_index, option in _indexed_product_options(
                bound,
                object_map,
                atomic_leaf_gate,
            ):
                yield raw_offset + child_index, option
            raw_offset += group_total
        return
    if operator in {"forpairs", "fornpairs"}:
        offset = 1
        pair_count: int | None = None
        if operator == "fornpairs":
            if not isinstance(expression[1], list) or len(expression[1]) != 1:
                raise CanonicalGroundingError("invalid fornpairs count")
            pair_count = int(expression[1][0])
            offset = 2
        if len(expression) != offset + 3:
            raise CanonicalGroundingError(f"invalid {operator} expression")
        variable1, category1 = _declaration(expression[offset])
        variable2, category2 = _declaration(expression[offset + 1])
        body = expression[offset + 2]
        instances1 = _objects(object_map, category1)
        instances2 = _objects(object_map, category2)
        count = (
            min(len(instances1), len(instances2))
            if pair_count is None
            else pair_count
        )
        if count < 0 or count > min(len(instances1), len(instances2)):
            raise CanonicalGroundingError(f"{operator} count exceeds available objects")
        raw_offset = 0
        for selected1 in combinations(instances1, count):
            for selected2 in permutations(instances2, count):
                if any(left == right for left, right in zip(selected1, selected2)):
                    continue
                bound = tuple(
                    (
                        body,
                        {
                            **bindings,
                            variable1: left,
                            variable2: right,
                        },
                    )
                    for left, right in zip(selected1, selected2)
                )
                group_total = 1
                for child, child_bindings in bound:
                    group_total *= _ground_option_count(
                        child,
                        object_map,
                        child_bindings,
                    )
                for child_index, option in _indexed_product_options(
                    bound,
                    object_map,
                    atomic_leaf_gate,
                ):
                    yield raw_offset + child_index, option
                raw_offset += group_total
        return
    leaf = [_substitute(item, bindings) for item in expression]
    if atomic_leaf_gate is None or atomic_leaf_gate(leaf):
        yield 0, [leaf]
