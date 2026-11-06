"""Read-only identity and drift checks for the official BEHAVIOR checkout."""

from __future__ import annotations

import subprocess
import os
from pathlib import Path
from typing import Any, Callable, Mapping


class OfficialRepositoryIdentityError(RuntimeError):
    """Raised when official source is dirty, unavailable, or changes in-run."""


def capture_official_repository_identity(
    repo: Path,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Capture exact git HEAD and require a clean tracked/untracked status."""

    root = Path(repo).resolve()
    if not root.is_dir() or not callable(runner):
        raise OfficialRepositoryIdentityError(
            "official repository path or git runner is invalid"
        )

    def git(*arguments: str) -> str:
        try:
            timeout_seconds = float(
                os.environ.get("BEHAVIOR_HL_OFFICIAL_GIT_TIMEOUT_SECONDS", "120")
            )
            if timeout_seconds <= 0:
                raise ValueError("BEHAVIOR_HL_OFFICIAL_GIT_TIMEOUT_SECONDS must be positive")
            completed = runner(
                ["git", "-C", str(root), *arguments],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise OfficialRepositoryIdentityError(
                "official git identity command failed"
            ) from exc
        if getattr(completed, "returncode", None) != 0:
            raise OfficialRepositoryIdentityError(
                "official git identity command returned non-zero"
            )
        stdout = getattr(completed, "stdout", None)
        if not isinstance(stdout, str):
            raise OfficialRepositoryIdentityError(
                "official git identity output is invalid"
            )
        return stdout.strip()

    head = git("rev-parse", "HEAD")
    if (
        len(head) != 40
        or any(character not in "0123456789abcdef" for character in head)
    ):
        raise OfficialRepositoryIdentityError(
            "official git HEAD is not a lowercase commit id"
        )
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    if status:
        raise OfficialRepositoryIdentityError(
            "official BEHAVIOR checkout is not clean"
        )
    return {
        "schema_version": "behavior2026.official-repository-identity/v1",
        "repository_root": str(root),
        "head": head,
        "clean": True,
        "status_porcelain": "",
    }


def require_unchanged_official_repository_identity(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    """Reject a result when the official checkout changed during the run."""

    required = {
        "schema_version",
        "repository_root",
        "head",
        "clean",
        "status_porcelain",
    }
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or set(before) != required
        or set(after) != required
        or before != after
        or before.get("clean") is not True
        or after.get("clean") is not True
    ):
        raise OfficialRepositoryIdentityError(
            "official BEHAVIOR repository identity changed during live run"
        )
