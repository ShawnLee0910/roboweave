"""Run-scoped artifact writer with atomic JSON replacement."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from behavior_hl.embodied.runtime.artifacts import (
    ArtifactInventory,
    ArtifactRecord,
    ArtifactVisibility,
)


class ArtifactWriter:
    """Write only beneath one explicit run directory and track hashes."""

    def __init__(self, run_dir: Path) -> None:
        resolved = Path(run_dir).resolve()
        if resolved == Path(resolved.anchor):
            raise ValueError("run_dir cannot be a filesystem root")
        resolved.mkdir(parents=True, exist_ok=True)
        self.run_dir = resolved
        self._records: dict[str, ArtifactRecord] = {}

    @property
    def inventory(self) -> ArtifactInventory:
        return ArtifactInventory(self._records.values())

    def write_json(
        self,
        *,
        logical_name: str,
        relative_path: str,
        visibility: ArtifactVisibility,
        payload: Any,
    ) -> ArtifactRecord:
        target = self._target(
            logical_name,
            relative_path,
            visibility,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return self._record(
            logical_name,
            relative_path,
            visibility,
            target,
        )

    def append_jsonl(
        self,
        *,
        logical_name: str,
        relative_path: str,
        visibility: ArtifactVisibility,
        payload: Any,
    ) -> ArtifactRecord:
        target = self._target(
            logical_name,
            relative_path,
            visibility,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return self._record(
            logical_name,
            relative_path,
            visibility,
            target,
        )

    def _target(
        self,
        logical_name: str,
        relative_path: str,
        visibility: ArtifactVisibility,
    ) -> Path:
        ArtifactRecord(
            logical_name,
            relative_path,
            visibility,
        )
        expected_prefix = {
            ArtifactVisibility.PUBLIC: "public",
            ArtifactVisibility.PRIVATE: "private",
            ArtifactVisibility.MEDIA: "video",
            ArtifactVisibility.SYSTEM: "system",
        }[visibility]
        path = Path(relative_path)
        if not path.parts or path.parts[0] != expected_prefix:
            raise ValueError(
                f"{visibility.value} artifact must use {expected_prefix}/"
            )
        target = (self.run_dir / path).resolve()
        if self.run_dir not in target.parents:
            raise ValueError("artifact path escaped the run directory")
        existing = self._records.get(logical_name)
        if existing is not None and existing.relative_path != relative_path:
            raise ValueError(
                "logical artifact name cannot change relative_path"
            )
        for name, record in self._records.items():
            if name != logical_name and record.relative_path == relative_path:
                raise ValueError(
                    "artifact relative_path already belongs to another name"
                )
        return target

    def _record(
        self,
        logical_name: str,
        relative_path: str,
        visibility: ArtifactVisibility,
        target: Path,
    ) -> ArtifactRecord:
        content = target.read_bytes()
        record = ArtifactRecord(
            logical_name=logical_name,
            relative_path=relative_path,
            visibility=visibility,
            byte_size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        self._records[logical_name] = record
        return record
