"""CPU predicate-record replay in an isolated, source-frozen backend process."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .engine import call
from .profiles import DEFAULT_PROFILE

SCHEMA = "esi-benchmark/score-record-v1"
SCHEMA_V2 = "esi-benchmark/score-record-v2"


def score_record(record: Mapping[str, Any], *, process_directory: Path | None = None,
                 profile: str | None = None) -> dict[str, Any]:
    """Score supplied facts without loading a native source bundle in this process."""
    if not isinstance(record, Mapping) or record.get("schema_version") not in (SCHEMA, SCHEMA_V2):
        raise ValueError(f"score record must use {SCHEMA} or {SCHEMA_V2}")
    if profile is not None and "profile" in record and profile != record["profile"]:
        raise ValueError("score profile differs from the record profile")
    selected = record.get("profile", DEFAULT_PROFILE) if profile is None else profile
    if not isinstance(selected, str) or not selected:
        raise ValueError("score profile must be a nonempty string")
    return call("score", {"record": dict(record),
                          "process_directory": str(process_directory) if process_directory is not None else None},
                profile=selected)
