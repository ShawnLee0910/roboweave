"""Session-only VRay roughness adapter for registered construction tasks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


VRAY_MDL_BASENAME = "omnigibson_vray_mtl.mdl"
VRAY_MATERIAL_NAME = "OmniGibsonVRayMtl"
RUNNER_VRAY_MATERIAL_NAME = "BehaviorHLVRayMtl"
RUNNER_VRAY_MDL_PATH = Path(os.environ.get(
    "ESI_RUNTIME_MDL_PATH",
    str(Path(__file__).resolve().parents[4] / "configs" / "materials" / "behavior_hl_vray_roughness.mdl"),
))
VRAY_ROUGHNESS_INPUTS = frozenset(
    {"reflection_roughness_constant", "reflection_roughness_texture_influence"}
)
VRAY_ALBEDO_INPUTS = frozenset({"albedo_add"})
VRAY_RENDER_INPUTS = VRAY_ROUGHNESS_INPUTS | VRAY_ALBEDO_INPUTS
_SOURCE_ASSET_ATTRIBUTE = "info:mdl:sourceAsset"
_SOURCE_SUBIDENTIFIER_ATTRIBUTE = "info:mdl:sourceAsset:subIdentifier"
_IMPLEMENTATION_SOURCE_ATTRIBUTE = "info:implementationSource"


class MaterialAdapterError(ValueError):
    """Raised when an adaptation cannot be applied and restored safely."""


class MaterialAdapterRollbackError(MaterialAdapterError):
    """The session transaction could not be restored; the resident is invalid."""

    session_invalidated = True


def _bounded(value: Any, *, low: float, high: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterialAdapterError(f"VRay {label} values must be finite numbers in [{low:g}, {high:g}]")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise MaterialAdapterError(f"VRay {label} values must be finite numbers in [{low:g}, {high:g}]")
    return value


def _unit(value: Any) -> float:
    return _bounded(value, low=0.0, high=1.0, label="roughness")


def _albedo_add(value: Any) -> float:
    return _bounded(value, low=-1.0, high=1.0, label="albedo_add")


def vray_roughness_from_glossiness(glossiness: Any) -> float:
    """Match ``vray_materials.mdl``: roughness = (1 - glossiness)^2."""

    glossiness = _unit(glossiness)
    return (1.0 - glossiness) ** 2


def vray_glossiness_from_roughness(roughness: Any) -> float:
    """Inverse mapping used by the runner-owned material wrapper."""

    return 1.0 - math.sqrt(_unit(roughness))


@dataclass(frozen=True)
class MaterialChange:
    """Report-compatible change backed by one idempotent VRay transaction."""

    material: Any
    key: str
    previous: Any
    value: Any
    _transaction: "_VRayTransaction"

    def __iter__(self):
        yield self.material
        yield self.key
        yield self.previous
        yield self.value

    @property
    def new(self) -> Any:
        """Report-friendly alias for the fourth tuple item."""

        return self.value

    def restore(self) -> Mapping[str, Any]:
        return self._transaction.restore()


def _source_asset_path(asset: Any) -> str:
    path = getattr(asset, "path", asset)
    return "" if path is None else str(path).replace("\\", "/")


def _source_identity(material: Any) -> tuple[Any, Any, str]:
    try:
        shader = material.shader
        asset = shader.GetSourceAsset("mdl")
        subidentifier = shader.GetSourceAssetSubIdentifier("mdl")
    except Exception as exc:  # USD wrappers differ in how invalid metadata is reported.
        raise MaterialAdapterError("material shader source metadata is unavailable") from exc
    return shader, asset, "" if subidentifier is None else str(subidentifier)


def is_vray_material(material: Any) -> bool:
    """Recognize only the exact OG VRay source/subidentifier pair."""

    try:
        _, asset, subidentifier = _source_identity(material)
    except MaterialAdapterError:
        return False
    path = _source_asset_path(asset)
    return (path == VRAY_MDL_BASENAME or path.endswith("/" + VRAY_MDL_BASENAME)) and (
        subidentifier == VRAY_MATERIAL_NAME
    )


def _is_runner_vray_material(material: Any) -> bool:
    try:
        _, asset, subidentifier = _source_identity(material)
    except MaterialAdapterError:
        return False
    return (
        _source_asset_path(asset) == _source_asset_path(RUNNER_VRAY_MDL_PATH)
        and subidentifier == RUNNER_VRAY_MATERIAL_NAME
    )


def validate_vray_material_rules(material: Any, inputs: Mapping[str, Any]) -> None:
    """Validate an exact VRay rule without writing USD."""

    if not (is_vray_material(material) or _is_runner_vray_material(material)):
        raise MaterialAdapterError("material is not the supported OG VRay shader")
    if not isinstance(inputs, Mapping) or not inputs or not set(inputs) <= VRAY_RENDER_INPUTS:
        raise MaterialAdapterError("unsupported VRay render-only material inputs")
    for key, value in inputs.items():
        if key == "albedo_add":
            _albedo_add(value)
        else:
            _unit(value)
    if (
        "reflection_roughness_texture_influence" in inputs
        and "reflection_roughness_constant" not in inputs
        and _unit(inputs["reflection_roughness_texture_influence"]) == 0
    ):
        raise MaterialAdapterError(
            "VRay roughness override needs reflection_roughness_constant when texture influence is 0"
        )
    if not RUNNER_VRAY_MDL_PATH.is_file():
        raise MaterialAdapterError(f"runner-owned VRay MDL is unavailable: {RUNNER_VRAY_MDL_PATH}")


def _stage(shader: Any) -> Any:
    try:
        stage = shader.GetPrim().GetStage()
    except Exception as exc:
        raise MaterialAdapterError("VRay shader stage is unavailable") from exc
    if stage is None:
        # ponytail: no CPU production fallback; real USD edits must be guarded.
        raise MaterialAdapterError("VRay shader stage is unavailable")
    return stage


@contextmanager
def _session_edit_stage(stage: Any):
    """Guard USD and Fabric edits for one complete, non-yielding transaction."""

    try:
        from pxr import Usd
        import omnigibson as og
    except ImportError as exc:  # pragma: no cover - Isaac Sim supplies both modules.
        raise MaterialAdapterError("OmniGibson USD editing support is unavailable") from exc
    with og.sim.editing_usd():
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            yield


@contextmanager
def _session_edit(shader: Any):
    with _session_edit_stage(_stage(shader)):
        yield


def _float_type() -> Any:
    try:
        from pxr import Sdf
    except ImportError as exc:  # pragma: no cover - Isaac Sim supplies pxr.
        raise MaterialAdapterError("pxr is required for VRay input creation") from exc
    return Sdf.ValueTypeNames.Float


def _asset_value() -> Any:
    try:
        from pxr import Sdf
    except ImportError as exc:  # pragma: no cover - Isaac Sim supplies pxr.
        raise MaterialAdapterError("pxr is required for VRay source conversion") from exc
    return Sdf.AssetPath(str(RUNNER_VRAY_MDL_PATH))


def _source_matches(shader: Any, asset: Any, subidentifier: str) -> bool:
    return (
        _source_asset_path(shader.GetSourceAsset("mdl")) == _source_asset_path(asset)
        and str(shader.GetSourceAssetSubIdentifier("mdl") or "") == subidentifier
    )


def _set_source(shader: Any, asset: Any, subidentifier: str) -> None:
    if shader.SetSourceAsset(asset, "mdl") is False:
        raise MaterialAdapterError("VRay shader source asset write failed")
    if shader.SetSourceAssetSubIdentifier(subidentifier, "mdl") is False:
        raise MaterialAdapterError("VRay shader source subidentifier write failed")
    if not _source_matches(shader, asset, subidentifier):
        raise MaterialAdapterError("VRay shader source write/readback mismatch")


def _valid(attr: Any) -> bool:
    try:
        return bool(attr.IsValid())
    except Exception as exc:
        raise MaterialAdapterError("VRay shader input validity is unavailable") from exc


def _property_path(prim_path: Any, name: str) -> Any:
    try:
        return prim_path.AppendProperty(name)
    except Exception as exc:  # pragma: no cover - malformed USD prims are fail-closed.
        raise MaterialAdapterError(f"VRay shader property path is unavailable: {name}") from exc


def _session_property_names() -> tuple[str, ...]:
    return (
        _IMPLEMENTATION_SOURCE_ATTRIBUTE,
        _SOURCE_ASSET_ATTRIBUTE,
        _SOURCE_SUBIDENTIFIER_ATTRIBUTE,
        *(f"inputs:{key}" for key in sorted(VRAY_RENDER_INPUTS)),
    )


def _remove_property(layer: Any, path: Any) -> None:
    spec = layer.GetPropertyAtPath(path)
    if spec is None:
        return
    prim = layer.GetPrimAtPath(path.GetPrimPath())
    if prim is None or prim.RemoveProperty(spec) is False:
        raise MaterialAdapterError(f"VRay session property removal failed: {path.name}")


def _property_fingerprint(layer: Any, path: Any) -> Any:
    """Return a bounded Sdf property value/metadata fingerprint.

    The backup is already a per-property Sdf layer, so ``GetAsText`` gives a
    compact comparison that includes the authored value and property metadata
    without taking or traversing a whole-stage snapshot.
    """

    spec = layer.GetPropertyAtPath(path)
    if spec is None:
        return None
    try:
        return ("sdf", str(spec.GetAsText()))
    except Exception as exc:
        raise MaterialAdapterError(
            f"VRay session property fingerprint unavailable: {path.name}"
        ) from exc


@dataclass
class _SessionSnapshot:
    layer: Any
    prim_path: Any
    property_names: tuple[str, ...]
    present: frozenset[str]
    backup: Any

    def restore(self) -> Mapping[str, Any]:
        try:
            from pxr import Sdf
        except ImportError as exc:  # pragma: no cover - Isaac Sim supplies pxr.
            raise MaterialAdapterError("pxr is required for VRay session rollback") from exc
        for name in self.property_names:
            path = _property_path(self.prim_path, name)
            _remove_property(self.layer, path)
            if name in self.present:
                result = Sdf.CopySpec(self.backup, path, self.layer, path)
                if result is False:
                    raise MaterialAdapterError(f"VRay session property restore failed: {name}")
        for name in self.property_names:
            path = _property_path(self.prim_path, name)
            present = self.layer.GetPropertyAtPath(path) is not None
            if present != (name in self.present):
                raise MaterialAdapterError(f"VRay session property readback failed: {name}")
            if present:
                expected = _property_fingerprint(self.backup, path)
                actual = _property_fingerprint(self.layer, path)
                if actual != expected:
                    raise MaterialAdapterError(
                        f"VRay session property value/metadata readback failed: {name}"
                    )
        input_names = tuple(name for name in self.property_names if name.startswith("inputs:"))
        return {
            "property_count": len(self.property_names),
            "restored_property_count": len(self.present),
            "property_readback_passed": True,
            "input_readback_passed": True,
            "input_property_count": len(input_names),
        }


def _snapshot_properties(stage: Any, prim_path: Any, property_names: Iterable[str]) -> _SessionSnapshot:
    """Snapshot only selected session properties for any USD prim.

    Light transforms use this same bounded snapshot instead of creating a second
    transaction implementation.  The caller must still perform writes inside
    ``_session_edit_stage``.
    """

    try:
        from pxr import Sdf
    except ImportError as exc:  # pragma: no cover - Isaac Sim supplies pxr.
        raise MaterialAdapterError("pxr is required for VRay session rollback") from exc
    layer = stage.GetSessionLayer()
    backup = Sdf.Layer.CreateAnonymous("behavior_hl_vray_material_backup")
    Sdf.CreatePrimInLayer(backup, prim_path)
    names = tuple(property_names)
    present = set()
    for name in names:
        path = _property_path(prim_path, name)
        if layer.GetPropertyAtPath(path) is not None:
            result = Sdf.CopySpec(layer, path, backup, path)
            if result is False:
                raise MaterialAdapterError(f"VRay session property backup failed: {name}")
            present.add(name)
    return _SessionSnapshot(layer, prim_path, names, frozenset(present), backup)


def _snapshot_session(shader: Any) -> _SessionSnapshot:
    return _snapshot_properties(_stage(shader), shader.GetPrim().GetPath(), _session_property_names())


def read_vray_input(material: Any, key: str, *, default: float | None = None) -> float | None:
    """Read a numeric VRay input, falling back to the native default when absent."""

    value = None
    getter = getattr(material, "get_input", None)
    if callable(getter):
        try:
            value = getter(key)
        except Exception:
            value = None
    if value is None:
        try:
            input_obj = material.shader.GetInput(key)
            attr = input_obj.GetAttr() if input_obj is not None else None
            if attr is not None and _valid(attr):
                value = input_obj.Get()
        except Exception:
            value = None
    if value is None:
        return default
    if key == "albedo_add":
        return _albedo_add(value)
    return _unit(value)


def _create_input(shader: Any, key: str, value: float) -> None:
    input_obj = shader.CreateInput(key, _float_type())
    if input_obj is None or not _valid(input_obj.GetAttr()):
        raise MaterialAdapterError(f"VRay shader input creation failed: {key}")
    if input_obj.Set(value) is False:
        raise MaterialAdapterError(f"VRay shader input write failed: {key}")
    actual = input_obj.Get()
    if actual is None or not math.isclose(float(actual), value, rel_tol=1e-6, abs_tol=1e-6):
        raise MaterialAdapterError(f"VRay shader input readback mismatch: {key}")


@dataclass
class _VRayTransaction:
    shader: Any
    original_asset: Any
    original_subidentifier: str
    snapshot: _SessionSnapshot
    _restored: bool = False
    _restore_evidence: Mapping[str, Any] | None = None
    kind: str = "VRAY"

    def restore(self) -> Mapping[str, Any]:
        if self._restored:
            return dict(self._restore_evidence or {})
        try:
            with _session_edit(self.shader):
                snapshot_evidence = self.snapshot.restore()
                source_readback_passed = _source_matches(
                    self.shader, self.original_asset, self.original_subidentifier
                )
                if not source_readback_passed:
                    raise MaterialAdapterError("VRay shader source rollback readback failed")
        except MaterialAdapterRollbackError:
            raise
        except Exception as exc:
            raise MaterialAdapterRollbackError(f"VRay shader rollback failed: {exc}") from exc
        self._restore_evidence = {
            "source_readback_passed": source_readback_passed,
            "input_readback_passed": bool(snapshot_evidence["input_readback_passed"]),
            "session_property_readback_passed": bool(snapshot_evidence["property_readback_passed"]),
            "restored_property_count": int(snapshot_evidence["restored_property_count"]),
        }
        self._restored = True
        return dict(self._restore_evidence)


def apply_vray_material_rules(material: Any, inputs: Mapping[str, Any]) -> list[MaterialChange]:
    """Apply one exact VRay rule in the USD session layer and restore it later."""

    validate_vray_material_rules(material, inputs)
    shader, original_asset, original_subidentifier = _source_identity(material)
    snapshot = _snapshot_session(shader)

    values = {
        key: (_albedo_add(value) if key == "albedo_add" else _unit(value))
        for key, value in inputs.items()
    }
    if "reflection_roughness_constant" in values and "reflection_roughness_texture_influence" not in values:
        values["reflection_roughness_texture_influence"] = 0.0
    previous_values = {
        key: read_vray_input(
            material,
            key,
            default=0.0 if key in {"albedo_add", "reflection_roughness_texture_influence"} else None,
        )
        for key in values
    }
    transaction = _VRayTransaction(
        shader=shader,
        original_asset=original_asset,
        original_subidentifier=original_subidentifier,
        snapshot=snapshot,
    )
    try:
        with _session_edit(shader):
            _set_source(shader, _asset_value(), RUNNER_VRAY_MATERIAL_NAME)
            for key, value in sorted(values.items()):
                _create_input(shader, key, value)
    except Exception:
        try:
            transaction.restore()
        except Exception as restore_error:
            raise MaterialAdapterRollbackError(
                f"VRay adaptation rollback failed: {restore_error}"
            ) from restore_error
        raise
    return [
        MaterialChange(
            material,
            key,
            previous_values[key],
            value,
            transaction,
        )
        for key, value in sorted(values.items())
    ]


def restore_material_changes(changes: Iterable[Any]) -> Mapping[str, Any]:
    """Restore old numeric changes and VRay transactions in reverse order.

    The return value is intentionally small and safe for old callers to ignore.
    """

    changes = tuple(changes)
    report = {
        "scope": "VRAY_ONLY",
        "change_count": len(changes),
        "numeric_change_count": 0,
        "vray_transaction_count": 0,
        "source_readback_passed": True,
        "input_readback_passed": True,
        "session_property_readback_passed": True,
    }
    seen_transactions: set[int] = set()
    for change in reversed(changes):
        restore = getattr(change, "restore", None)
        if callable(restore):
            evidence = restore()
            transaction = getattr(change, "_transaction", None)
            transaction_id = id(transaction) if transaction is not None else id(change)
            if transaction_id not in seen_transactions:
                seen_transactions.add(transaction_id)
                if getattr(transaction, "kind", "VRAY") == "VRAY":
                    report["vray_transaction_count"] += 1
                else:
                    report.setdefault("session_property_transaction_count", 0)
                    report["session_property_transaction_count"] += 1
            if isinstance(evidence, Mapping):
                for key in (
                    "source_readback_passed",
                    "input_readback_passed",
                    "session_property_readback_passed",
                ):
                    report[key] = bool(report[key] and evidence.get(key, True))
            continue
        material, key, previous, _ = change
        material.set_input(key, previous)
        report["numeric_change_count"] += 1
    return report
