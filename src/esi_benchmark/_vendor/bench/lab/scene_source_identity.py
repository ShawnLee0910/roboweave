"""Small Git identity contract for one resident launch."""

from __future__ import annotations

from functools import cache
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


_COMMIT = re.compile(r"[0-9a-f]{40}")


def git_source_head(root: Path) -> str:
    source = Path(root).resolve()
    if not source.is_dir():
        raise ValueError(f"source root is unavailable: {source}")
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
        check=False,
    )
    commit = revision.stdout.strip()
    if revision.returncode != 0 or not _COMMIT.fullmatch(commit):
        raise ValueError(f"source root is not a readable Git worktree: {source}")
    return commit


@cache
def _cached_git_source_identity(source: str) -> tuple[str, bool]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            source,
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
        check=False,
    )
    commit = git_source_head(Path(source))
    if completed.returncode != 0:
        raise ValueError(f"source root is not a readable Git worktree: {source}")
    return commit, bool(completed.stdout.strip())


def git_source_identity(root: Path) -> dict[str, Any]:
    source = Path(root).resolve()
    if not source.is_dir():
        raise ValueError(f"source root is unavailable: {source}")
    commit, dirty = _cached_git_source_identity(str(source))
    return {
        "root": str(source),
        "git_commit": commit,
        "dirty": dirty,
    }


def validate_source_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "root",
        "git_commit",
        "dirty",
    }:
        raise ValueError("resident source identity fields changed")
    root = value.get("root")
    commit = value.get("git_commit")
    dirty = value.get("dirty")
    if (
        not isinstance(root, str)
        or not Path(root).is_absolute()
        or not isinstance(commit, str)
        or not _COMMIT.fullmatch(commit)
        or not isinstance(dirty, bool)
    ):
        raise ValueError("resident source identity is invalid")
    recorded = {
        "root": str(Path(root).resolve()),
        "git_commit": commit,
        "dirty": dirty,
    }
    if dirty or git_source_head(Path(root)) != commit:
        raise ValueError(f"resident source changed after materialization: {root}")
    return recorded


__all__ = ["git_source_head", "git_source_identity", "validate_source_identity"]
