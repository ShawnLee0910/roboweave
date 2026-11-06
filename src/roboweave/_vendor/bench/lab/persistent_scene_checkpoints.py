"""Bounded, portable files for resident scene checkpoints.

The simulator tensor and the small amount of Python-owned carry state are
written without pickle.  A new resident process may read the files only after
it has loaded the same task identity and rebound simulator object references
by their exact live names.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import numpy as np


PERSISTENT_SCENE_CHECKPOINT_VERSION = (
    "behavior-hl-persistent-scene-checkpoint/v1"
)
PORTABLE_VALUE_VERSION = "behavior-hl-portable-runtime-value/v1"
MAXIMUM_MANIFEST_BYTES = 8 * 1024 * 1024
MAXIMUM_SIMULATOR_STATE_BYTES = 128 * 1024 * 1024
MAXIMUM_PORTABLE_ARRAY_BYTES = 8 * 1024 * 1024
MAXIMUM_PORTABLE_DEPTH = 32
MAXIMUM_PORTABLE_NODES = 100_000
_MANIFEST_NAME = re.compile(r"checkpoint_action_([0-9]{4,})\.json")


class PersistentCheckpointError(RuntimeError):
    """A durable checkpoint failed a safety or compatibility gate."""


@dataclass(frozen=True, slots=True)
class LoadedPersistentCheckpoint:
    manifest_path: Path
    manifest_sha256: str
    captured_action_count: int
    source_episode_id: str
    identity: Mapping[str, Any]
    compatibility: Mapping[str, Any]
    simulator_state: np.ndarray
    logical_state: Any
    carry_tracker_state: Any
    verifier: Mapping[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_private_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _safe_relative_manifest(path: Path, *, allowed_root: Path) -> Path:
    source = Path(path)
    root = Path(allowed_root).resolve()
    if source.is_symlink():
        raise PersistentCheckpointError("checkpoint manifest may not be a symlink")
    resolved = source.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise PersistentCheckpointError("checkpoint manifest is outside the runtime root")
    if _MANIFEST_NAME.fullmatch(resolved.name) is None:
        raise PersistentCheckpointError("checkpoint manifest filename is invalid")
    return resolved


def encode_portable_value(
    value: Any,
    *,
    object_name_for: Callable[[Any], str | None],
    dataclass_encoder: Callable[[Any], tuple[str, Mapping[str, Any]] | None],
) -> Any:
    """Encode an allowlisted runtime value without executable serialization."""

    nodes = 0

    def encode(item: Any, depth: int) -> Any:
        nonlocal nodes
        nodes += 1
        if depth > MAXIMUM_PORTABLE_DEPTH or nodes > MAXIMUM_PORTABLE_NODES:
            raise PersistentCheckpointError("portable runtime value is too complex")
        if item is None or isinstance(item, (bool, str, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise PersistentCheckpointError("portable float is not finite")
            return item
        if isinstance(item, np.generic):
            return encode(item.item(), depth + 1)

        object_name = object_name_for(item)
        if object_name is not None:
            if not isinstance(object_name, str) or not object_name:
                raise PersistentCheckpointError("portable object name is invalid")
            return {"kind": "object_ref", "name": object_name}

        custom = dataclass_encoder(item)
        if custom is not None:
            tag, fields = custom
            if not isinstance(tag, str) or not tag or not isinstance(fields, Mapping):
                raise PersistentCheckpointError("portable dataclass encoder changed")
            return {
                "kind": "dataclass",
                "tag": tag,
                "fields": encode(dict(fields), depth + 1),
            }

        array: np.ndarray | None = None
        array_kind = "ndarray"
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
        else:
            detach = getattr(item, "detach", None)
            cpu = getattr(item, "cpu", None)
            numpy_method = getattr(item, "numpy", None)
            if callable(detach) and callable(cpu):
                candidate = detach().cpu()
                numpy_method = getattr(candidate, "numpy", None)
                if callable(numpy_method):
                    array = np.ascontiguousarray(numpy_method())
                    array_kind = "tensor"
        if array is not None:
            if array.dtype.hasobject or array.nbytes > MAXIMUM_PORTABLE_ARRAY_BYTES:
                raise PersistentCheckpointError("portable array is unsafe or too large")
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                raise PersistentCheckpointError("portable array is not finite")
            return {
                "kind": array_kind,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
            }
        if isinstance(item, tuple):
            return {"kind": "tuple", "items": [encode(v, depth + 1) for v in item]}
        if isinstance(item, list):
            return {"kind": "list", "items": [encode(v, depth + 1) for v in item]}
        if isinstance(item, (set, frozenset)):
            ordered = sorted(item, key=lambda value: repr(value))
            return {
                "kind": "frozenset" if isinstance(item, frozenset) else "set",
                "items": [encode(v, depth + 1) for v in ordered],
            }
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise PersistentCheckpointError("portable mapping keys must be strings")
            return {
                "kind": "mapping",
                "items": [[key, encode(item[key], depth + 1)] for key in sorted(item)],
            }
        raise PersistentCheckpointError(
            f"unsupported portable runtime type: {type(item).__module__}.{type(item).__qualname__}"
        )

    return {"schema_version": PORTABLE_VALUE_VERSION, "value": encode(value, 0)}


def decode_portable_value(
    record: Any,
    *,
    object_for_name: Callable[[str], Any],
    dataclass_decoder: Callable[[str, Mapping[str, Any]], Any],
    tensor_factory: Callable[[np.ndarray], Any] | None = None,
) -> Any:
    """Decode only the tagged value language emitted above."""

    if (
        not isinstance(record, Mapping)
        or set(record) != {"schema_version", "value"}
        or record.get("schema_version") != PORTABLE_VALUE_VERSION
    ):
        raise PersistentCheckpointError("portable runtime value header changed")
    nodes = 0

    def decode(item: Any, depth: int) -> Any:
        nonlocal nodes
        nodes += 1
        if depth > MAXIMUM_PORTABLE_DEPTH or nodes > MAXIMUM_PORTABLE_NODES:
            raise PersistentCheckpointError("portable runtime value is too complex")
        if item is None or isinstance(item, (bool, str, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise PersistentCheckpointError("portable float is not finite")
            return item
        if not isinstance(item, Mapping) or not isinstance(item.get("kind"), str):
            raise PersistentCheckpointError("portable runtime tag is invalid")
        kind = item["kind"]
        if kind == "object_ref" and set(item) == {"kind", "name"}:
            name = item["name"]
            if not isinstance(name, str) or not name:
                raise PersistentCheckpointError("portable object name is invalid")
            value = object_for_name(name)
            if value is None:
                raise PersistentCheckpointError("portable object is absent from live scene")
            return value
        if kind in {"ndarray", "tensor"} and set(item) == {
            "kind", "dtype", "shape", "data"
        }:
            if (
                not isinstance(item["dtype"], str)
                or not isinstance(item["shape"], list)
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in item["shape"])
                or not isinstance(item["data"], str)
            ):
                raise PersistentCheckpointError("portable array header is invalid")
            try:
                dtype = np.dtype(item["dtype"])
                raw = base64.b64decode(item["data"], validate=True)
            except Exception as exc:
                raise PersistentCheckpointError("portable array encoding is invalid") from exc
            if dtype.hasobject or len(raw) > MAXIMUM_PORTABLE_ARRAY_BYTES:
                raise PersistentCheckpointError("portable array is unsafe or too large")
            expected = int(np.prod(item["shape"], dtype=np.int64)) * dtype.itemsize
            if expected != len(raw):
                raise PersistentCheckpointError("portable array byte length changed")
            array = np.frombuffer(raw, dtype=dtype).reshape(item["shape"]).copy()
            if np.issubdtype(dtype, np.number) and not np.isfinite(array).all():
                raise PersistentCheckpointError("portable array is not finite")
            if kind == "tensor":
                if tensor_factory is None:
                    raise PersistentCheckpointError("portable tensor factory is unavailable")
                return tensor_factory(array)
            return array
        if kind in {"tuple", "list", "set", "frozenset"} and set(item) == {"kind", "items"}:
            values = item["items"]
            if not isinstance(values, list):
                raise PersistentCheckpointError("portable sequence is invalid")
            decoded = [decode(value, depth + 1) for value in values]
            if kind == "tuple":
                return tuple(decoded)
            if kind == "list":
                return decoded
            if kind == "set":
                return set(decoded)
            return frozenset(decoded)
        if kind == "mapping" and set(item) == {"kind", "items"}:
            pairs = item["items"]
            if not isinstance(pairs, list):
                raise PersistentCheckpointError("portable mapping is invalid")
            result: dict[str, Any] = {}
            for pair in pairs:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or not isinstance(pair[0], str)
                    or pair[0] in result
                ):
                    raise PersistentCheckpointError("portable mapping item is invalid")
                result[pair[0]] = decode(pair[1], depth + 1)
            return result
        if kind == "dataclass" and set(item) == {"kind", "tag", "fields"}:
            tag = item["tag"]
            if not isinstance(tag, str) or not tag:
                raise PersistentCheckpointError("portable dataclass tag is invalid")
            fields = decode(item["fields"], depth + 1)
            if not isinstance(fields, Mapping):
                raise PersistentCheckpointError("portable dataclass fields are invalid")
            return dataclass_decoder(tag, fields)
        raise PersistentCheckpointError("portable runtime tag is unsupported")

    return decode(record["value"], 0)


def write_persistent_checkpoint(
    *,
    root: Path,
    captured_action_count: int,
    source_episode_id: str,
    identity: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    simulator_state: np.ndarray,
    logical_state: Any,
    carry_tracker_state: Any,
    verifier: Mapping[str, Any],
    retained_action_counts: list[int],
) -> Mapping[str, Any]:
    if (
        isinstance(captured_action_count, bool)
        or not isinstance(captured_action_count, int)
        or captured_action_count <= 0
        or not isinstance(source_episode_id, str)
        or not source_episode_id
    ):
        raise PersistentCheckpointError("persistent checkpoint identity is invalid")
    if captured_action_count not in retained_action_counts or len(retained_action_counts) > 2:
        raise PersistentCheckpointError("persistent checkpoint retention changed")
    state = np.ascontiguousarray(simulator_state)
    if state.ndim != 1 or state.dtype.hasobject:
        raise PersistentCheckpointError("serialized simulator state must be a numeric vector")
    if state.nbytes <= 0 or state.nbytes > MAXIMUM_SIMULATOR_STATE_BYTES:
        raise PersistentCheckpointError("serialized simulator state size is invalid")
    if np.issubdtype(state.dtype, np.number) and not np.isfinite(state).all():
        raise PersistentCheckpointError("serialized simulator state is not finite")

    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination, 0o700)
    stem = f"checkpoint_action_{captured_action_count:04d}"
    state_path = destination / f"{stem}.npy"
    stream = io.BytesIO()
    np.save(stream, state, allow_pickle=False)
    state_bytes = stream.getvalue()
    _write_private_atomic(state_path, state_bytes)
    state_sha256 = _sha256(state_bytes)
    manifest = {
        "schema_version": PERSISTENT_SCENE_CHECKPOINT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "captured_action_count": captured_action_count,
        "source_episode_id": source_episode_id,
        "identity": dict(identity),
        "compatibility": dict(compatibility),
        "simulator_state": {
            "file": state_path.name,
            "sha256": state_sha256,
            "bytes": len(state_bytes),
            "dtype": state.dtype.str,
            "shape": list(state.shape),
        },
        "logical_state": logical_state,
        "carry_tracker_state": carry_tracker_state,
        "verifier": dict(verifier),
    }
    manifest_bytes = _canonical_json_bytes(manifest) + b"\n"
    if len(manifest_bytes) > MAXIMUM_MANIFEST_BYTES:
        state_path.unlink(missing_ok=True)
        raise PersistentCheckpointError("persistent checkpoint manifest is too large")
    manifest_path = destination / f"{stem}.json"
    _write_private_atomic(manifest_path, manifest_bytes)
    manifest_sha256 = _sha256(manifest_bytes)

    retained = set(retained_action_counts)
    for path in tuple(destination.iterdir()):
        match = re.fullmatch(r"checkpoint_action_([0-9]{4,})\.(json|npy)", path.name)
        if match is not None and int(match.group(1)) not in retained:
            path.unlink(missing_ok=True)
    return {
        "schema_version": PERSISTENT_SCENE_CHECKPOINT_VERSION,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "captured_action_count": captured_action_count,
        "simulator_state_bytes": len(state_bytes),
    }


def load_persistent_checkpoint(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_identity: Mapping[str, Any],
    allowed_root: Path,
) -> LoadedPersistentCheckpoint:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256 or ""):
        raise PersistentCheckpointError("checkpoint manifest hash is invalid")
    source = _safe_relative_manifest(manifest_path, allowed_root=allowed_root)
    manifest_bytes = source.read_bytes()
    if len(manifest_bytes) > MAXIMUM_MANIFEST_BYTES:
        raise PersistentCheckpointError("checkpoint manifest is too large")
    manifest_sha256 = _sha256(manifest_bytes)
    if manifest_sha256 != expected_manifest_sha256:
        raise PersistentCheckpointError("checkpoint manifest hash changed")
    try:
        manifest = json.loads(manifest_bytes)
    except Exception as exc:
        raise PersistentCheckpointError("checkpoint manifest is invalid JSON") from exc
    expected_fields = {
        "schema_version", "created_at", "captured_action_count",
        "source_episode_id", "identity", "compatibility", "simulator_state",
        "logical_state", "carry_tracker_state", "verifier",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != expected_fields
        or manifest.get("schema_version") != PERSISTENT_SCENE_CHECKPOINT_VERSION
        or manifest.get("identity") != dict(expected_identity)
    ):
        raise PersistentCheckpointError("checkpoint manifest contract or identity changed")
    captured = manifest["captured_action_count"]
    match = _MANIFEST_NAME.fullmatch(source.name)
    if (
        isinstance(captured, bool)
        or not isinstance(captured, int)
        or captured <= 0
        or match is None
        or int(match.group(1)) != captured
        or not isinstance(manifest["source_episode_id"], str)
        or not manifest["source_episode_id"]
        or not isinstance(manifest["compatibility"], Mapping)
        or not isinstance(manifest["verifier"], Mapping)
    ):
        raise PersistentCheckpointError("checkpoint manifest fields are invalid")
    state_record = manifest["simulator_state"]
    if (
        not isinstance(state_record, Mapping)
        or set(state_record) != {"file", "sha256", "bytes", "dtype", "shape"}
        or state_record.get("file") != f"checkpoint_action_{captured:04d}.npy"
        or not re.fullmatch(r"[0-9a-f]{64}", str(state_record.get("sha256", "")))
        or isinstance(state_record.get("bytes"), bool)
        or not isinstance(state_record.get("bytes"), int)
        or state_record["bytes"] <= 0
        or state_record["bytes"] > MAXIMUM_SIMULATOR_STATE_BYTES
    ):
        raise PersistentCheckpointError("checkpoint simulator-state manifest is invalid")
    state_path = source.parent / state_record["file"]
    if state_path.is_symlink() or not state_path.is_file():
        raise PersistentCheckpointError("checkpoint simulator state is unavailable")
    state_bytes = state_path.read_bytes()
    if len(state_bytes) != state_record["bytes"] or _sha256(state_bytes) != state_record["sha256"]:
        raise PersistentCheckpointError("checkpoint simulator state hash changed")
    try:
        state = np.load(io.BytesIO(state_bytes), allow_pickle=False)
    except Exception as exc:
        raise PersistentCheckpointError("checkpoint simulator state is invalid") from exc
    if (
        not isinstance(state, np.ndarray)
        or state.ndim != 1
        or state.dtype.hasobject
        or state.dtype.str != state_record["dtype"]
        or list(state.shape) != state_record["shape"]
        or (np.issubdtype(state.dtype, np.number) and not np.isfinite(state).all())
    ):
        raise PersistentCheckpointError("checkpoint simulator state contract changed")
    return LoadedPersistentCheckpoint(
        manifest_path=source,
        manifest_sha256=manifest_sha256,
        captured_action_count=captured,
        source_episode_id=manifest["source_episode_id"],
        identity=dict(manifest["identity"]),
        compatibility=dict(manifest["compatibility"]),
        simulator_state=np.ascontiguousarray(state),
        logical_state=manifest["logical_state"],
        carry_tracker_state=manifest["carry_tracker_state"],
        verifier=dict(manifest["verifier"]),
    )
