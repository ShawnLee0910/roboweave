"""Static policy-tree audit used by G0."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


FORBIDDEN_IMPORT_ROOTS = frozenset({"omnigibson"})
FORBIDDEN_SOURCE_TOKENS = (
    "object_scope",
    "ground_goal_state_options",
    "trav_map",
    "get_shortest_path",
    "seg_map",
    "addressable_objects",
    "goal_status",
    "get_position_orientation",
    "RigidContactAPI",
    "set_position_orientation",
    "viewer_camera",
    "object_states.Inside",
    "private_goal_trace.jsonl",
)


@dataclass(frozen=True, slots=True)
class AuditViolation:
    path: Path
    detail: str


def audit_policy_tree(policy_root: Path) -> tuple[AuditViolation, ...]:
    """Return all import, token, and syntax violations in a policy package."""

    policy_root = policy_root.resolve()
    violations: list[AuditViolation] = []
    if not policy_root.is_dir():
        return (AuditViolation(policy_root, "policy root does not exist"),)

    for path in sorted(policy_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_SOURCE_TOKENS:
            if token in text:
                violations.append(
                    AuditViolation(path, f"forbidden source token: {token}")
                )
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            violations.append(
                AuditViolation(path, f"syntax error: {exc.msg}")
            )
            continue
        for node in ast.walk(tree):
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots.extend(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.append(node.module.split(".", 1)[0])
            for root in roots:
                if root in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(
                        AuditViolation(path, f"forbidden import root: {root}")
                    )
    return tuple(violations)
