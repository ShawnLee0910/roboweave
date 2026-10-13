"""Verify complete canonical ``:goal`` forms against a BEHAVIOR checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

from behavior_hl.tasks import FIVE_TASKS


class BDDLGoalParseError(ValueError):
    """Raised when a problem does not contain one balanced ``:goal`` form."""


def _without_comments(text: str) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        quote = False
        escaped = False
        kept = []
        for char in line:
            if quote:
                kept.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quote = False
            elif char == '"':
                quote = True
                kept.append(char)
            elif char == ";":
                if line.endswith("\n"):
                    kept.append("\n")
                break
            else:
                kept.append(char)
        lines.append("".join(kept))
    return "".join(lines)


def _form_starts(text: str, keyword: str) -> list[int]:
    starts: list[int] = []
    quote = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
            index += 1
            continue
        if char == '"':
            quote = True
            index += 1
            continue
        if char != "(":
            index += 1
            continue
        cursor = index + 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        end = cursor + len(keyword)
        if text[cursor:end] == keyword and (
            end == len(text) or text[end].isspace() or text[end] in "()"
        ):
            starts.append(index)
        index += 1
    return starts


def extract_goal_form(problem_text: str) -> str:
    """Extract exactly one balanced ``(:goal ...)`` form."""

    uncommented = _without_comments(problem_text)
    starts = _form_starts(uncommented, ":goal")
    if len(starts) != 1:
        raise BDDLGoalParseError(f"expected exactly one :goal form, found {len(starts)}")
    start = starts[0]
    depth = 0
    quote = False
    escaped = False
    for index in range(start, len(uncommented)):
        char = uncommented[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
            continue
        if char == '"':
            quote = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return uncommented[start : index + 1]
            if depth < 0:
                break
    raise BDDLGoalParseError("unbalanced :goal form")


def canonicalize_bddl_form(form: str) -> str:
    """Return a whitespace/comment-insensitive, token-preserving form."""

    text = _without_comments(form)
    tokens: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in "()":
            tokens.append(char)
            index += 1
            continue
        if char == '"':
            start = index
            index += 1
            escaped = False
            while index < len(text):
                current = text[index]
                index += 1
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    break
            else:
                raise BDDLGoalParseError("unterminated string literal")
            tokens.append(text[start:index])
            continue
        start = index
        while index < len(text) and not text[index].isspace() and text[index] not in "()":
            index += 1
        tokens.append(text[start:index])
    return " ".join(tokens)


def extract_canonical_goal(problem_text: str) -> str:
    return canonicalize_bddl_form(extract_goal_form(problem_text))


def normalize_bddl(text: str) -> str:
    """Backward-compatible alias for token-preserving normalization."""

    return canonicalize_bddl_form(text)


def verify_behavior_root(behavior_root: Path) -> list[str]:
    errors: list[str] = []
    definitions = behavior_root / "bddl3" / "bddl" / "activity_definitions"
    for spec in FIVE_TASKS.values():
        bddl_path = definitions / spec.name / "problem0.bddl"
        if not bddl_path.is_file():
            errors.append(f"{spec.name}: missing {bddl_path}")
            continue
        try:
            actual_goal = extract_canonical_goal(bddl_path.read_text(encoding="utf-8"))
        except BDDLGoalParseError as exc:
            errors.append(f"{spec.name}: invalid :goal form: {exc}")
            continue
        if actual_goal != spec.official_canonical_goal:
            errors.append(
                f"{spec.name}: canonical :goal mismatch: "
                f"expected {spec.official_canonical_goal!r}, got {actual_goal!r}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("behavior_root", type=Path)
    args = parser.parse_args()
    errors = verify_behavior_root(args.behavior_root)
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    print(f"OK: all {len(FIVE_TASKS)} pilot task specs match complete canonical :goal forms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
