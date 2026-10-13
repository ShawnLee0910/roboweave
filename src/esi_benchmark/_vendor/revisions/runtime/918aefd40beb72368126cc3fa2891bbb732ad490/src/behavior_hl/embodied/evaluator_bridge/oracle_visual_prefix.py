"""Recover already-rendered Oracle prefix frames after an exception.

Goal-graph execution returns its in-memory frame list only on a normal return.
If a later route or verifier operation raises, the action boundary PNGs are
still durable under ``steps/<index>/action_after.png``.  This module rebuilds
only that existing prefix.  It does not render, synthesize a terminal frame,
change Q, or claim visual completeness.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


VISUAL_PREFIX_RECOVERY_VERSION = (
    "behavior2026.oracle-visual-prefix-recovery/v1"
)
DEFAULT_MAX_PREFIX_STEP_DIRECTORIES = 10_000


@dataclass(frozen=True, slots=True)
class VisualPrefixRecovery:
    frames: tuple[Mapping[str, Any], ...]
    discovered_action_after_count: int
    recovered_action_after_count: int

    def private_report(self) -> dict[str, Any]:
        return {
            "version": VISUAL_PREFIX_RECOVERY_VERSION,
            "discovered_action_after_count": self.discovered_action_after_count,
            "recovered_action_after_count": self.recovered_action_after_count,
            "frame_count_after_recovery": len(self.frames),
            "terminal_frame_synthesized": False,
            "q_or_verifier_state_modified": False,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recover_existing_action_after_prefix(
    *,
    phase_dir: Path,
    existing_frames: Sequence[Mapping[str, Any]],
    maximum_step_directories: int = DEFAULT_MAX_PREFIX_STEP_DIRECTORIES,
) -> VisualPrefixRecovery:
    """Merge durable ``action_after`` PNGs into a chronological prefix.

    The scan is deliberately bounded to immediate numeric children of the
    exact phase ``steps`` directory.  Existing richer entries win by path;
    recovery supplies only missing action boundaries and marks them as
    recovered.  No final frame is invented, so callers must keep
    ``complete=false`` for exception paths.
    """

    phase_dir = Path(phase_dir).resolve()
    if maximum_step_directories <= 0:
        raise ValueError("maximum_step_directories must be positive")
    steps_dir = phase_dir / "steps"
    numeric_steps: list[tuple[int, Path]] = []
    if steps_dir.is_dir():
        for entry in steps_dir.iterdir():
            if not entry.is_dir() or not entry.name.isdigit():
                continue
            numeric_steps.append((int(entry.name), entry))
            if len(numeric_steps) > maximum_step_directories:
                raise RuntimeError("visual prefix step-directory bound exceeded")
    numeric_steps.sort(key=lambda item: (item[0], item[1].name))

    normalized_existing = [dict(frame) for frame in existing_frames]
    existing_by_path: dict[str, dict[str, Any]] = {}
    for frame in normalized_existing:
        raw_path = frame.get("path")
        if isinstance(raw_path, str) and raw_path:
            existing_by_path[str(Path(raw_path).resolve())] = frame

    discovered = 0
    recovered = 0
    action_entries: dict[int, dict[str, Any]] = {}
    for frame in normalized_existing:
        index = frame.get("action_index")
        if isinstance(index, int) and index >= 0:
            action_entries.setdefault(index, frame)
    for action_index, step_dir in numeric_steps:
        frame_path = (step_dir / "action_after.png").resolve()
        if not frame_path.is_file():
            continue
        if phase_dir not in frame_path.parents:
            raise RuntimeError("visual prefix frame escaped phase directory")
        discovered += 1
        existing = existing_by_path.get(str(frame_path))
        if existing is not None:
            action_entries.setdefault(action_index, existing)
            continue
        recovered += 1
        action_entries.setdefault(
            action_index,
            {
                "kind": "action_after",
                "action_index": action_index,
                "path": str(frame_path),
                "png_sha256": _sha256(frame_path),
                "recovered_after_exception": True,
                "recovery_version": VISUAL_PREFIX_RECOVERY_VERSION,
            },
        )

    prefix = [
        frame
        for frame in normalized_existing
        if frame.get("kind") == "initial"
        and not isinstance(frame.get("action_index"), int)
    ]
    suffix = [
        frame
        for frame in normalized_existing
        if frame.get("kind") != "initial"
        and not isinstance(frame.get("action_index"), int)
    ]
    merged = (*prefix, *(action_entries[key] for key in sorted(action_entries)), *suffix)
    return VisualPrefixRecovery(
        frames=tuple(merged),
        discovered_action_after_count=discovered,
        recovered_action_after_count=recovered,
    )


__all__ = [
    "DEFAULT_MAX_PREFIX_STEP_DIRECTORIES",
    "VISUAL_PREFIX_RECOVERY_VERSION",
    "VisualPrefixRecovery",
    "recover_existing_action_after_prefix",
]
