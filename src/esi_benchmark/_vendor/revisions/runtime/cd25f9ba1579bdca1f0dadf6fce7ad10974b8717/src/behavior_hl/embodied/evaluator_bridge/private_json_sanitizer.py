"""Bounded canonical JSON conversion for private diagnostic artifacts.

This module is intentionally independent of the simulator and public runtime.
It converts diagnostic values into strict RFC-8259-compatible JSON values
without silently replacing non-finite measurements with ``null``.  Every
lossy or bounded conversion is represented by an explicit private token and
is repeated in the returned health record with its original field path.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from typing import Any, Mapping


PRIVATE_JSON_SANITIZER_VERSION = "behavior2026.private-json-sanitizer/v1"
PRIVATE_JSON_TOKEN_KEY = "__behavior2026_private_json_token__"


@dataclass(frozen=True, slots=True)
class PrivateJsonSanitizationResult:
    """One strict JSON value plus bounded private conversion health."""

    value: Any
    health: Mapping[str, Any]


def _source_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _child_path(path: str, key: str) -> str:
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


class _PrivateJsonSanitizer:
    def __init__(
        self,
        *,
        maximum_depth: int,
        maximum_nodes: int,
        maximum_container_items: int,
        maximum_total_string_characters: int,
        maximum_string_characters: int,
        maximum_serialized_bytes: int,
        maximum_health_events: int,
    ) -> None:
        for label, value in (
            ("maximum_depth", maximum_depth),
            ("maximum_nodes", maximum_nodes),
            ("maximum_container_items", maximum_container_items),
            (
                "maximum_total_string_characters",
                maximum_total_string_characters,
            ),
            ("maximum_string_characters", maximum_string_characters),
            ("maximum_serialized_bytes", maximum_serialized_bytes),
            ("maximum_health_events", maximum_health_events),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.maximum_depth = maximum_depth
        self.maximum_nodes = maximum_nodes
        self.maximum_container_items = maximum_container_items
        self.maximum_total_string_characters = maximum_total_string_characters
        self.maximum_string_characters = maximum_string_characters
        self.maximum_serialized_bytes = maximum_serialized_bytes
        self.maximum_health_events = maximum_health_events
        self.nodes_seen = 0
        self.maximum_depth_observed = 0
        self.total_string_characters = 0
        self._active_container_ids: set[int] = set()
        self._event_counts: Counter[str] = Counter()
        self._events: list[dict[str, Any]] = []

    def _record_event(
        self,
        *,
        kind: str,
        path: str,
        source_type: str,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        self._event_counts[kind] += 1
        if len(self._events) >= self.maximum_health_events:
            return
        event: dict[str, Any] = {
            "kind": kind,
            "field_path_private": path,
            "source_type_private": source_type,
        }
        if detail:
            event.update(detail)
        self._events.append(event)

    def _token(
        self,
        *,
        kind: str,
        path: str,
        source_type: str,
        detail: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self._record_event(
            kind=kind,
            path=path,
            source_type=source_type,
            detail=detail,
        )
        body: dict[str, Any] = {
            "kind": kind,
            "field_path_private": path,
            "source_type_private": source_type,
        }
        if detail:
            body.update(detail)
        return {PRIVATE_JSON_TOKEN_KEY: body}

    def _finite_number(self, value: float, *, path: str, source: str) -> Any:
        if math.isfinite(value):
            return value
        classification = (
            "nan"
            if math.isnan(value)
            else ("positive_infinity" if value > 0.0 else "negative_infinity")
        )
        return self._token(
            kind="non_finite_number",
            path=path,
            source_type=source,
            detail={"classification": classification},
        )

    def _bounded_string(self, value: str, *, path: str, source: str) -> Any:
        value_length = len(value)
        if (
            value_length > self.maximum_string_characters
            or self.total_string_characters + value_length
            > self.maximum_total_string_characters
        ):
            return self._token(
                kind="string_size_limit",
                path=path,
                source_type=source,
                detail={
                    "character_count": value_length,
                    "sha256_private": hashlib.sha256(
                        value.encode("utf-8", errors="surrogatepass")
                    ).hexdigest(),
                },
            )
        self.total_string_characters += value_length
        return value

    def _size_token(
        self,
        value: Any,
        *,
        path: str,
        reason: str,
        item_count: int | None = None,
    ) -> Mapping[str, Any]:
        detail: dict[str, Any] = {"reason": reason}
        if item_count is not None:
            detail["item_count"] = item_count
        return self._token(
            kind="size_limit",
            path=path,
            source_type=_source_type(value),
            detail=detail,
        )

    def _enter_container(self, value: Any, *, path: str) -> Mapping[str, Any] | None:
        marker = id(value)
        if marker in self._active_container_ids:
            return self._token(
                kind="cycle",
                path=path,
                source_type=_source_type(value),
            )
        self._active_container_ids.add(marker)
        return None

    def _leave_container(self, value: Any) -> None:
        self._active_container_ids.discard(id(value))

    @staticmethod
    def _shape(value: Any) -> tuple[int, ...] | None:
        try:
            raw = getattr(value, "shape")
            shape = tuple(int(item) for item in raw)
        except Exception:
            return None
        if any(item < 0 for item in shape):
            return None
        return shape

    @staticmethod
    def _tensor_item_count(value: Any, shape: tuple[int, ...] | None) -> int | None:
        try:
            numel = getattr(value, "numel", None)
            if callable(numel):
                count = int(numel())
                if count >= 0:
                    return count
            size = getattr(value, "size", None)
            if isinstance(size, int) and size >= 0:
                return size
        except Exception:
            return None
        if shape is None:
            return None
        count = 1
        for dimension in shape:
            count *= dimension
        return count

    def _convert_array_like(
        self,
        value: Any,
        *,
        path: str,
        depth: int,
    ) -> Any | None:
        source = _source_type(value)
        module = type(value).__module__.split(".", 1)[0]
        has_tensor_interface = all(
            hasattr(value, attribute) for attribute in ("shape", "tolist")
        )
        if module not in {"numpy", "torch"} and not has_tensor_interface:
            return None
        shape = self._shape(value)
        item_count = self._tensor_item_count(value, shape)
        if item_count == 1 or shape == ():
            try:
                scalar = value.item()
            except Exception as exc:
                return self._token(
                    kind="scalar_conversion_error",
                    path=path,
                    source_type=source,
                    detail={"error_type_private": type(exc).__name__},
                )
            if isinstance(scalar, float):
                return self._finite_number(scalar, path=path, source=source)
            return self.convert(scalar, path=path, depth=depth + 1)
        if item_count is None:
            return self._token(
                kind="array_shape_unavailable",
                path=path,
                source_type=source,
            )
        if item_count > self.maximum_container_items:
            return self._size_token(
                value,
                path=path,
                reason="array_item_limit",
                item_count=item_count,
            )
        try:
            materialized = value
            detach = getattr(materialized, "detach", None)
            if callable(detach):
                materialized = detach()
            cpu = getattr(materialized, "cpu", None)
            if callable(cpu):
                materialized = cpu()
            items = materialized.tolist()
        except Exception as exc:
            return self._token(
                kind="array_conversion_error",
                path=path,
                source_type=source,
                detail={"error_type_private": type(exc).__name__},
            )
        converted = self.convert(items, path=path, depth=depth + 1)
        return {
            PRIVATE_JSON_TOKEN_KEY: {
                "kind": "array",
                "source_type_private": source,
                "shape": list(shape or ()),
                "items": converted,
            }
        }

    def convert(self, value: Any, *, path: str = "$", depth: int = 0) -> Any:
        self.maximum_depth_observed = max(self.maximum_depth_observed, depth)
        if depth > self.maximum_depth:
            return self._token(
                kind="depth_limit",
                path=path,
                source_type=_source_type(value),
                detail={"maximum_depth": self.maximum_depth},
            )
        if self.nodes_seen >= self.maximum_nodes:
            return self._size_token(value, path=path, reason="node_limit")
        self.nodes_seen += 1

        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            return self._bounded_string(value, path=path, source=_source_type(value))
        if isinstance(value, int):
            if value.bit_length() > 4096:
                return self._size_token(value, path=path, reason="integer_bit_limit")
            return value
        if isinstance(value, float):
            return self._finite_number(value, path=path, source=_source_type(value))

        converted_array = self._convert_array_like(
            value,
            path=path,
            depth=depth,
        )
        if converted_array is not None:
            return converted_array

        if isinstance(value, bytes):
            return self._token(
                kind="binary",
                path=path,
                source_type=_source_type(value),
                detail={
                    "byte_count": len(value),
                    "sha256_private": hashlib.sha256(value).hexdigest(),
                },
            )

        if is_dataclass(value) and not isinstance(value, type):
            cycle = self._enter_container(value, path=path)
            if cycle is not None:
                return cycle
            try:
                dataclass_fields = sorted(fields(value), key=lambda item: item.name)
                if len(dataclass_fields) > self.maximum_container_items:
                    return self._size_token(
                        value,
                        path=path,
                        reason="dataclass_field_limit",
                        item_count=len(dataclass_fields),
                    )
                result: dict[str, Any] = {}
                for item in dataclass_fields:
                    child = _child_path(path, item.name)
                    try:
                        field_value = getattr(value, item.name)
                    except Exception as exc:
                        result[item.name] = self._token(
                            kind="field_read_error",
                            path=child,
                            source_type=_source_type(value),
                            detail={"error_type_private": type(exc).__name__},
                        )
                    else:
                        result[item.name] = self.convert(
                            field_value,
                            path=child,
                            depth=depth + 1,
                        )
                return result
            finally:
                self._leave_container(value)

        if isinstance(value, Mapping):
            cycle = self._enter_container(value, path=path)
            if cycle is not None:
                return cycle
            try:
                try:
                    items = list(value.items())
                except Exception as exc:
                    return self._token(
                        kind="mapping_read_error",
                        path=path,
                        source_type=_source_type(value),
                        detail={"error_type_private": type(exc).__name__},
                    )
                if len(items) > self.maximum_container_items:
                    return self._size_token(
                        value,
                        path=path,
                        reason="mapping_item_limit",
                        item_count=len(items),
                    )
                if any(not isinstance(key, str) for key, _item in items):
                    entry_values = []
                    for key, item_value in items:
                        entry_values.append(
                            {
                                "key": self.convert(
                                    key,
                                    path=f"{path}.entries[*].key",
                                    depth=depth + 2,
                                ),
                                "value": self.convert(
                                    item_value,
                                    path=f"{path}.entries[*].value",
                                    depth=depth + 2,
                                ),
                            }
                        )
                    entry_values.sort(
                        key=lambda item: json.dumps(
                            item,
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                    )
                    self._record_event(
                        kind="non_string_mapping_keys",
                        path=path,
                        source_type=_source_type(value),
                    )
                    return {
                        PRIVATE_JSON_TOKEN_KEY: {
                            "kind": "mapping_entries",
                            "source_type_private": _source_type(value),
                            "entries": entry_values,
                        }
                    }
                result = {}
                for key, item_value in sorted(items, key=lambda item: item[0]):
                    bounded_key = self._bounded_string(
                        key,
                        path=f"{path}.__key__",
                        source=_source_type(key),
                    )
                    if not isinstance(bounded_key, str):
                        return self._size_token(
                            value,
                            path=path,
                            reason="mapping_key_string_limit",
                            item_count=len(items),
                        )
                    result[key] = self.convert(
                        item_value,
                        path=_child_path(path, key),
                        depth=depth + 1,
                    )
                return result
            finally:
                self._leave_container(value)

        if isinstance(value, (list, tuple, deque, set, frozenset)):
            cycle = self._enter_container(value, path=path)
            if cycle is not None:
                return cycle
            try:
                items = list(value)
                if len(items) > self.maximum_container_items:
                    return self._size_token(
                        value,
                        path=path,
                        reason="container_item_limit",
                        item_count=len(items),
                    )
                if isinstance(value, (set, frozenset)):
                    converted = [
                        self.convert(
                            item,
                            path=f"{path}[*]",
                            depth=depth + 1,
                        )
                        for item in items
                    ]
                    converted.sort(
                        key=lambda item: json.dumps(
                            item,
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                    )
                    self._record_event(
                        kind="unordered_container",
                        path=path,
                        source_type=_source_type(value),
                    )
                else:
                    converted = [
                        self.convert(
                            item,
                            path=f"{path}[{index}]",
                            depth=depth + 1,
                        )
                        for index, item in enumerate(items)
                    ]
                return converted
            finally:
                self._leave_container(value)

        return self._token(
            kind="unsupported_type",
            path=path,
            source_type=_source_type(value),
        )

    def finish(self, value: Any) -> PrivateJsonSanitizationResult:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        serialized_bytes = len(canonical.encode("utf-8"))
        if serialized_bytes > self.maximum_serialized_bytes:
            value = self._token(
                kind="serialized_size_limit",
                path="$",
                source_type="sanitized_payload",
                detail={
                    "serialized_byte_count": serialized_bytes,
                    "maximum_serialized_bytes": self.maximum_serialized_bytes,
                },
            )
            canonical = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            serialized_bytes = len(canonical.encode("utf-8"))
        total_events = sum(self._event_counts.values())
        ordered_events = sorted(
            self._events,
            key=lambda item: (
                str(item.get("field_path_private", "")),
                str(item.get("kind", "")),
                str(item.get("source_type_private", "")),
                json.dumps(item, ensure_ascii=False, sort_keys=True),
            ),
        )
        health = {
            "schema_version": PRIVATE_JSON_SANITIZER_VERSION,
            "status": "SANITIZED" if total_events else "CLEAN",
            "strict_json_pass": True,
            "nodes_seen": self.nodes_seen,
            "maximum_depth_observed": self.maximum_depth_observed,
            "serialized_byte_count": serialized_bytes,
            "event_count": total_events,
            "event_counts": dict(sorted(self._event_counts.items())),
            "events_private": ordered_events,
            "events_omitted_count": max(0, total_events - len(self._events)),
            "bounds": {
                "maximum_depth": self.maximum_depth,
                "maximum_nodes": self.maximum_nodes,
                "maximum_container_items": self.maximum_container_items,
                "maximum_total_string_characters": (
                    self.maximum_total_string_characters
                ),
                "maximum_string_characters": self.maximum_string_characters,
                "maximum_serialized_bytes": self.maximum_serialized_bytes,
                "maximum_health_events": self.maximum_health_events,
            },
        }
        return PrivateJsonSanitizationResult(value=value, health=health)


def sanitize_private_json(
    value: Any,
    *,
    maximum_depth: int = 12,
    maximum_nodes: int = 4096,
    maximum_container_items: int = 512,
    maximum_total_string_characters: int = 262_144,
    maximum_string_characters: int = 16_384,
    maximum_serialized_bytes: int = 1_048_576,
    maximum_health_events: int = 256,
) -> PrivateJsonSanitizationResult:
    """Return a deterministic, bounded strict-JSON representation.

    This function never mutates ``value``.  Non-finite numbers, unsupported
    types, cycles, and every applied resource bound remain explicit in both
    the converted private payload and its private health report.
    """

    sanitizer = _PrivateJsonSanitizer(
        maximum_depth=maximum_depth,
        maximum_nodes=maximum_nodes,
        maximum_container_items=maximum_container_items,
        maximum_total_string_characters=maximum_total_string_characters,
        maximum_string_characters=maximum_string_characters,
        maximum_serialized_bytes=maximum_serialized_bytes,
        maximum_health_events=maximum_health_events,
    )
    converted = sanitizer.convert(value)
    return sanitizer.finish(converted)
