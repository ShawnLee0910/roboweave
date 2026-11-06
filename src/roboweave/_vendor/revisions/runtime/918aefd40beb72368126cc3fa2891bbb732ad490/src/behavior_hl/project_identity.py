"""Fail-closed project-root identity for isolated BEHAVIOR-HL copies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


PROJECT_IDENTITY_VERSION = "behavior-hl.project-identity/v1"
PROJECT_ROOT_ENV = "BEHAVIOR_HL_PROJECT_ROOT"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = PROJECT_ROOT.parent


class ProjectIdentityError(RuntimeError):
    """Raised before scene construction when project roots do not agree."""


def capture_project_identity(
    *,
    expected_root: Path | None = None,
    module_files: Mapping[str, Path] | None = None,
    require_environment: bool = False,
) -> dict[str, Any]:
    """Validate and describe the exact source tree imported by this process."""

    root = PROJECT_ROOT.resolve()
    expected = root if expected_root is None else Path(expected_root).resolve()
    if expected != root:
        raise ProjectIdentityError(
            f"expected project root {expected} differs from imported root {root}"
        )

    environment_value = os.environ.get(PROJECT_ROOT_ENV)
    if require_environment and not environment_value:
        raise ProjectIdentityError(f"{PROJECT_ROOT_ENV} is required")
    environment_root = (
        None if not environment_value else Path(environment_value).resolve()
    )
    if environment_root is not None and environment_root != root:
        raise ProjectIdentityError(
            f"{PROJECT_ROOT_ENV}={environment_root} differs from imported root {root}"
        )

    marker_paths = {
        "pyproject": root / "pyproject.toml",
        "package_init": root / "src" / "behavior_hl" / "__init__.py",
        "config": root / "configs" / "picking_up_toys_r1pro_hla.yaml",
        "approach_child": root / "scripts" / "tokenplan_approach_decision.py",
        "resident_child": root / "scripts" / "tokenplan_resident_decision.py",
    }
    missing = [name for name, path in marker_paths.items() if not path.is_file()]
    if missing:
        raise ProjectIdentityError(
            f"project root is missing required markers: {sorted(missing)}"
        )
    for name, path in marker_paths.items():
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ProjectIdentityError(
                f"project marker {name} escaped project root: {resolved}"
            ) from exc

    checked_modules: dict[str, str] = {}
    for name, value in dict(module_files or {}).items():
        resolved = Path(value).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ProjectIdentityError(
                f"imported module {name} escaped project root: {resolved}"
            ) from exc
        if not resolved.is_file():
            raise ProjectIdentityError(
                f"imported module {name} is not a file: {resolved}"
            )
        checked_modules[str(name)] = str(resolved)

    return {
        "schema_version": PROJECT_IDENTITY_VERSION,
        "candidate_source_identity": (
            json.loads((root / "SOURCE_IDENTITY.json").read_text())
            if (root / "SOURCE_IDENTITY.json").is_file() else None
        ),
        "project_root": str(root),
        "bundle_root": str(BUNDLE_ROOT.resolve()),
        "environment_variable": PROJECT_ROOT_ENV,
        "environment_project_root": (
            None if environment_root is None else str(environment_root)
        ),
        "identity_module": str(Path(__file__).resolve()),
        "markers": {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in marker_paths.items()
        },
        "imported_modules": checked_modules,
    }


def bind_project_root_environment() -> str:
    """Bind child processes to this imported source tree without overriding drift."""

    capture_project_identity(expected_root=PROJECT_ROOT)
    os.environ[PROJECT_ROOT_ENV] = str(PROJECT_ROOT.resolve())
    return os.environ[PROJECT_ROOT_ENV]
