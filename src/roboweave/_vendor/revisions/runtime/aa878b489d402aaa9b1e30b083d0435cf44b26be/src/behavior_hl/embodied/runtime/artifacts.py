"""Pure artifact inventory contracts; no filesystem writes occur here."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Iterable


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ArtifactVisibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    MEDIA = "media"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One logical artifact and its run-relative location."""

    logical_name: str
    relative_path: str
    visibility: ArtifactVisibility
    byte_size: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.logical_name:
            raise ValueError("logical_name must not be empty")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
            raise ValueError("relative_path must stay within the run directory")
        if self.byte_size is not None:
            if (
                isinstance(self.byte_size, bool)
                or not isinstance(self.byte_size, int)
                or self.byte_size < 0
            ):
                raise ValueError("byte_size must be a non-negative integer")
        if self.sha256 is not None and not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_name": self.logical_name,
            "relative_path": self.relative_path,
            "visibility": self.visibility.value,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


class ArtifactInventory:
    """Validated immutable inventory with required-artifact checks."""

    def __init__(self, records: Iterable[ArtifactRecord] = ()) -> None:
        by_name: dict[str, ArtifactRecord] = {}
        paths: set[str] = set()
        for record in records:
            if record.logical_name in by_name:
                raise ValueError(
                    f"duplicate artifact logical_name: {record.logical_name!r}"
                )
            if record.relative_path in paths:
                raise ValueError(
                    f"duplicate artifact relative_path: {record.relative_path!r}"
                )
            by_name[record.logical_name] = record
            paths.add(record.relative_path)
        self._records = dict(sorted(by_name.items()))

    @property
    def logical_names(self) -> tuple[str, ...]:
        return tuple(self._records)

    @property
    def records(self) -> tuple[ArtifactRecord, ...]:
        return tuple(self._records.values())

    def get(self, logical_name: str) -> ArtifactRecord:
        return self._records[logical_name]

    def missing(self, required_logical_names: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            sorted(set(required_logical_names).difference(self._records))
        )

    def require(self, required_logical_names: Iterable[str]) -> None:
        missing = self.missing(required_logical_names)
        if missing:
            raise ValueError(f"missing required artifacts: {list(missing)}")

    def to_manifest(self) -> list[dict[str, object]]:
        return [record.to_dict() for record in self.records]
