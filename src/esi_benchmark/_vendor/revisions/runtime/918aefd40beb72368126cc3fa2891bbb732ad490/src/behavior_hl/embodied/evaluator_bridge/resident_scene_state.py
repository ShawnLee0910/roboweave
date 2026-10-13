"""Construction-only completion of authored resident-scene state.

The native scene loader does not synthesize joint state for an articulated
object.  During initial-state compilation only, this module lets the native
object dump its own current state fill *missing* joint fields.  It never
overwrites authored fields and never invents values for an object whose native
dump does not expose joints.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterator


JOINT_STATE_KEYS = ("joint_pos", "joint_vel")
RESIDENT_SCENE_STATE_VERSION = "behavior2026.resident-scene-state/v1"
_ACTIVE_INITIAL_STATE_COMPLETION: "_InitialStateCompletionStore | None" = None


class ResidentSceneStateCompletionError(RuntimeError):
    """Raised when native state cannot safely complete authored state."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = dict(report or {})


def _jsonable(value: Any) -> Any:
    """Convert native tensor/array values to finite JSON evidence."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("native joint state contains a non-finite value")
        return value
    raise TypeError(f"native joint state is not JSON-readable: {type(value).__name__}")


def _scene_objects(scene: Any) -> dict[str, Any]:
    raw_objects = getattr(scene, "objects", None)
    if isinstance(raw_objects, Mapping):
        objects = dict(raw_objects)
    else:
        objects = {
            str(getattr(obj, "name")): obj
            for obj in (raw_objects or ())
            if getattr(obj, "name", None) is not None
        }
    return objects


def _state_registry(state: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        registry = state["registry"]["object_registry"]
    except (KeyError, TypeError) as exc:
        raise ResidentSceneStateCompletionError(
            "scene state has no registry.object_registry"
        ) from exc
    if not isinstance(registry, Mapping):
        raise ResidentSceneStateCompletionError("scene object state registry is not a mapping")
    return registry


def complete_missing_joint_state(
    *,
    scene: Any,
    state: Mapping[str, Any],
    authored_object_names: Collection[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Complete missing authored joint fields from native ``dump_state``.

    ``state`` is copied.  Existing ``joint_pos`` / ``joint_vel`` fields and
    every non-joint field, especially ``root_link``, remain untouched.  A
    native dump that has no joint fields is treated as a rigid object and does
    not cause synthetic state to be added.  An articulated object outside the
    authored scope is rejected instead of silently repaired.
    """

    if not isinstance(state, Mapping):
        raise ResidentSceneStateCompletionError("scene state must be a mapping")
    authored = {str(name) for name in authored_object_names}
    copied = deepcopy(dict(state))
    registry = _state_registry(copied)
    objects = _scene_objects(scene)
    filled: list[dict[str, Any]] = []
    preserved: list[dict[str, Any]] = []

    for name, raw_entry in registry.items():
        if not isinstance(raw_entry, Mapping):
            raise ResidentSceneStateCompletionError(
                f"state for object {name!r} is not a mapping"
            )
        # OG EntityPrim._load_state has an explicit backwards-compatible
        # branch for this legacy representation; preserve it verbatim rather
        # than mixing modern fields into the same object state.
        if "joints" in raw_entry:
            continue
        missing = [key for key in JOINT_STATE_KEYS if key not in raw_entry]
        if not missing:
            continue

        obj = objects.get(str(name))
        if obj is None:
            raise ResidentSceneStateCompletionError(
                f"live object for incomplete state is unavailable: {name!r}"
            )
        dump_state = getattr(obj, "dump_state", None)
        if not callable(dump_state):
            raise ResidentSceneStateCompletionError(
                f"native dump_state is unavailable for incomplete object: {name!r}"
            )
        try:
            native = dump_state(serialized=False)
        except Exception as exc:
            raise ResidentSceneStateCompletionError(
                f"native dump_state failed for incomplete object: {name!r}"
            ) from exc
        if not isinstance(native, Mapping):
            raise ResidentSceneStateCompletionError(
                f"native dump_state is not a mapping for object: {name!r}"
            )
        native_joint_keys = [key for key in JOINT_STATE_KEYS if key in native]
        if not native_joint_keys:
            # Rigid objects legitimately have only root/non-kinematic state.
            continue
        if str(name) not in authored:
            raise ResidentSceneStateCompletionError(
                f"incomplete articulated state is outside authored scope: {name!r}"
            )
        if any(key not in native for key in JOINT_STATE_KEYS):
            raise ResidentSceneStateCompletionError(
                f"native dump is incomplete for articulated object: {name!r}"
            )

        entry = dict(raw_entry)
        filled_fields: list[str] = []
        native_defaults: dict[str, Any] = {}
        for key in missing:
            value = native[key]
            if value is None:
                raise ResidentSceneStateCompletionError(
                    f"native dump returned null {key} for object: {name!r}"
                )
            entry[key] = deepcopy(value)
            native_defaults[key] = _jsonable(value)
            filled_fields.append(key)
        registry[name] = entry
        filled.append(
            {
                "object": str(name),
                "fields": filled_fields,
                "native_defaults": native_defaults,
            }
        )
        existing = [key for key in JOINT_STATE_KEYS if key in raw_entry]
        if existing:
            preserved.append({"object": str(name), "fields": existing})

    report = {
        "schema_version": RESIDENT_SCENE_STATE_VERSION,
        "scope": "construction-only/authored-object-missing-joint-fields",
        "filled_object_count": len(filled),
        "filled_field_count": sum(len(item["fields"]) for item in filled),
        "filled": filled,
        "existing_joint_fields_preserved": preserved,
    }
    return copied, report


def native_initial_state_completion_report(scene: Any) -> Mapping[str, Any] | None:
    """Return the process-local report attached by the initialization hook."""

    report = getattr(scene, "_resident_scene_state_completion_report", None)
    return None if report is None else dict(report)


class _InitialStateCompletionStore:
    """Keep the completed construction state available to the TRO path hook.

    The store deliberately owns no live-object reads after ``set_state``.  A
    TRO restore happens after the scene has been reset/settled, so reading a
    live object's state there would make the result depend on the previous
    task.  The only source accepted here is the state passed through the
    construction-time Scene.load_state completion above.
    """

    def __init__(self) -> None:
        self._completed_state: Mapping[str, Any] | None = None
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="behavior2026-tro-state-"
        )
        self._materialized: dict[str, Path] = {}
        self.report: dict[str, Any] = {
            "source": "initial_native_scene_completion",
            "materialized_tro_count": 0,
            "materialized_tro_field_count": 0,
            "materialized_tro": [],
        }

    def close(self) -> None:
        self._temporary_directory.cleanup()

    def set_state(self, state: Mapping[str, Any]) -> None:
        if self._completed_state is not None:
            raise ResidentSceneStateCompletionError(
                "native initial scene completion was invoked more than once"
            )
        self._completed_state = deepcopy(dict(state))

    def materialize(
        self,
        *,
        tro_state_path: Path,
        scope_bindings: Mapping[str, Any],
    ) -> Path:
        if self._completed_state is None:
            raise ResidentSceneStateCompletionError(
                "TRO restore requested before native initial scene completion"
            )
        source = Path(tro_state_path).resolve()
        try:
            source_bytes = source.read_bytes()
            source_document = json.loads(source_bytes.decode("utf-8"))
        except Exception as exc:
            raise ResidentSceneStateCompletionError(
                f"registered TRO state is unreadable: {source}"
            ) from exc
        if not isinstance(source_document, Mapping):
            raise ResidentSceneStateCompletionError(
                f"registered TRO state is not a mapping: {source}"
            )
        registry = _state_registry(self._completed_state)

        binding_fingerprint = json.dumps(
            scope_bindings, sort_keys=True, separators=(",", ":")
        )
        cache_key = f"{source}\n{binding_fingerprint}"
        cached = self._materialized.get(cache_key)
        if cached is not None:
            return cached

        document = deepcopy(dict(source_document))
        filled: list[dict[str, Any]] = []
        for scope, raw_entry in document.items():
            if scope == "robot_poses":
                continue
            if not isinstance(raw_entry, Mapping):
                raise ResidentSceneStateCompletionError(
                    f"TRO state for scope {scope!r} is not a mapping"
                )
            # The native loader has a separate legacy branch for this schema;
            # adding modern joint fields would change its semantics.
            if "joints" in raw_entry:
                continue
            missing = [key for key in JOINT_STATE_KEYS if key not in raw_entry]
            if not missing:
                continue

            binding = scope_bindings.get(scope)
            if not isinstance(binding, Mapping):
                raise ResidentSceneStateCompletionError(
                    f"TRO scope has no registered binding: {scope!r}"
                )
            actual_name = binding.get("actual_name")
            if not actual_name:
                raise ResidentSceneStateCompletionError(
                    f"TRO scope has no actual object name: {scope!r}"
                )
            native_entry = registry.get(str(actual_name))
            if not isinstance(native_entry, Mapping):
                raise ResidentSceneStateCompletionError(
                    "initial completed scene state has no object "
                    f"{actual_name!r} for TRO scope {scope!r}"
                )
            native_joint_keys = [key for key in JOINT_STATE_KEYS if key in native_entry]
            if not native_joint_keys:
                # The object is rigid in the frozen initial state.  Its TRO
                # omission is valid and must remain untouched.
                continue
            if any(key not in native_entry for key in JOINT_STATE_KEYS):
                raise ResidentSceneStateCompletionError(
                    f"initial completed state has incomplete joints for {actual_name!r}"
                )

            entry = dict(raw_entry)
            joint_values: dict[str, Any] = {}
            for key in missing:
                value = native_entry[key]
                if value is None:
                    raise ResidentSceneStateCompletionError(
                        f"initial completed state has null {key} for {actual_name!r}"
                    )
                entry[key] = _jsonable(value)
                joint_values[key] = deepcopy(entry[key])
            document[scope] = entry
            filled.append(
                {
                    "scope": str(scope),
                    "object": str(actual_name),
                    "fields": list(missing),
                    "joint_values": joint_values,
                }
            )

        if not filled:
            self._materialized[cache_key] = source
            return source

        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        destination = Path(self._temporary_directory.name) / f"{digest}.json"
        derived_bytes = json.dumps(document, indent=2, allow_nan=False).encode("utf-8")
        destination.write_bytes(derived_bytes)
        self._materialized[cache_key] = destination
        self.report["materialized_tro_count"] += 1
        self.report["materialized_tro_field_count"] += sum(
            len(item["fields"]) for item in filled
        )
        self.report["materialized_tro"].append(
            {
                "source": str(source),
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "derived_sha256": hashlib.sha256(derived_bytes).hexdigest(),
                "fields": filled,
            }
        )
        return destination


def materialize_tro_state_from_initial_state(
    *,
    tro_state_path: Path,
    scope_bindings: Mapping[str, Any],
) -> Path | None:
    """Return a runner-owned TRO copy with initial joint fields completed.

    ``None`` means no construction completion context is active; callers may
    then use their normal frozen TRO path.  Once the context is active, a
    missing initial state is an error rather than a silent fallback.
    """

    store = _ACTIVE_INITIAL_STATE_COMPLETION
    if store is None:
        return None
    return store.materialize(
        tro_state_path=tro_state_path,
        scope_bindings=scope_bindings,
    )


@contextmanager
def install_native_initial_state_completion(
    *,
    authored_object_names: Collection[str],
    scene_class: type[Any] | None = None,
) -> Iterator[None]:
    """Patch native ``Scene.load_state`` for construction-only initialization.

    The patch is process-local and restored immediately.  It only runs for the
    first non-serialized state load performed by ``Scene.initialize``; later
    task restores and Oracle actions are untouched.
    """

    if scene_class is None:
        from omnigibson.scenes.scene_base import Scene as scene_class

    global _ACTIVE_INITIAL_STATE_COMPLETION

    original = scene_class.load_state
    authored = frozenset(str(name) for name in authored_object_names)
    if _ACTIVE_INITIAL_STATE_COMPLETION is not None:
        raise RuntimeError("native initial scene completion is already installed")
    store = _InitialStateCompletionStore()
    _ACTIVE_INITIAL_STATE_COMPLETION = store

    def load_state(self: Any, state: Any, serialized: bool = False) -> Any:
        initial_load = (
            not serialized
            and bool(getattr(self, "_initialized", False))
            and getattr(self, "_initial_file", None) is None
            and not bool(getattr(self, "_resident_scene_state_completed", False))
        )
        if initial_load:
            completed, report = complete_missing_joint_state(
                scene=self,
                state=state,
                authored_object_names=authored,
            )
            store.set_state(completed)
            report["tro_state_completion"] = store.report
            self._resident_scene_state_completion_report = report
            self._resident_scene_state_completed = True
            state = completed
        return original(self, state, serialized=serialized)

    scene_class.load_state = load_state
    try:
        yield
    finally:
        scene_class.load_state = original
        _ACTIVE_INITIAL_STATE_COMPLETION = None
        store.close()
