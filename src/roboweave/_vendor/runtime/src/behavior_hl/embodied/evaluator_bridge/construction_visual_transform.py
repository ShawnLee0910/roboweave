"""Bounded render-only visual transforms for construction calibration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any

from .construction_material_adapter import (
    MaterialAdapterError,
    MaterialAdapterRollbackError,
    _session_edit_stage,
    _snapshot_properties,
    apply_vray_material_rules,
    is_vray_material,
    read_vray_input,
    restore_material_changes,
    validate_vray_material_rules,
)


_LIGHT_RULE_FIELDS = frozenset({"kind", "light_type", "factor"})
_BACKGROUND_RULE_FIELDS = frozenset({"kind", "offset"})
_LIGHT_TYPES = frozenset({"DomeLight", "DistantLight"})


def _objects(scene: Any) -> tuple[Any, ...]:
    objects = getattr(scene, "objects", ())
    if isinstance(objects, Mapping):
        return tuple(objects.values())
    return tuple(objects)


def _prim_path(value: Any) -> str:
    path = getattr(value, "prim_path", value)
    path = str(path).rstrip("/")
    if not path.startswith("/") or path == "/":
        raise MaterialAdapterError("scene object prim inventory unavailable")
    return path


def _scene_object_paths(scene: Any) -> tuple[str, ...]:
    paths = tuple(_prim_path(obj) for obj in _objects(scene))
    if len(paths) != len(set(paths)):
        raise MaterialAdapterError("scene object prim inventory contains duplicate paths")
    return paths


def _stage_from_evaluator(evaluator: Any) -> Any:
    for owner in (evaluator, getattr(evaluator, "env", None)):
        stage = getattr(owner, "stage", None)
        if stage is not None:
            return stage
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    except Exception as exc:  # pragma: no cover - only live Kit supplies omni.usd.
        raise MaterialAdapterError("USD stage is unavailable for visual transform") from exc
    if stage is None:
        raise MaterialAdapterError("USD stage is unavailable for visual transform")
    return stage


def _environment_light(stage: Any, scene: Any, light_type: str) -> Any:
    if light_type not in _LIGHT_TYPES:
        raise MaterialAdapterError("only noninteractive environment light types are supported")
    paths = _scene_object_paths(scene)
    candidates = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        active = prim.IsActive() if callable(getattr(prim, "IsActive", None)) else True
        if active and prim.GetTypeName() == light_type and not any(
            path == owner or path.startswith(owner + "/") for owner in paths
        ):
            candidates.append(prim)
    if len(candidates) != 1:
        raise MaterialAdapterError(
            f"expected one noninteractive environment light, found {len(candidates)}"
        )
    return candidates[0]


def _light_attribute(prim: Any, attribute_name: str, *, allow_zero: bool) -> tuple[Any, float]:
    attr = prim.GetAttribute(attribute_name)
    if attr is None:
        raise MaterialAdapterError(f"light attribute is missing: {attribute_name}")
    try:
        if not attr or attr.GetNumTimeSamples() != 0 or attr.GetConnections():
            raise MaterialAdapterError(
                f"light attribute is missing, animated or connected: {attribute_name}"
            )
        previous = attr.Get()
    except MaterialAdapterError:
        raise
    except Exception as exc:
        raise MaterialAdapterError(f"light attribute readback is unavailable: {attribute_name}") from exc
    if isinstance(previous, bool) or not isinstance(previous, (int, float)):
        raise MaterialAdapterError(f"light baseline is not numeric: {attribute_name}")
    previous = float(previous)
    if not math.isfinite(previous) or (previous < 0 if allow_zero else previous <= 0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise MaterialAdapterError(f"light baseline must be finite and {qualifier}: {attribute_name}")
    return attr, previous


def _factor(value: Any, *, allow_reduction_only: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterialAdapterError("light factor must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0 or (allow_reduction_only and value >= 1):
        raise MaterialAdapterError("light factor must be a finite positive number")
    return value


def _normalize_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(rule, Mapping) or not isinstance(rule.get("kind"), str):
        raise MaterialAdapterError("visual transform rule must be a mapping with kind")
    kind = rule["kind"]
    if kind == "light_intensity":
        if not set(rule) <= _LIGHT_RULE_FIELDS:
            raise MaterialAdapterError("light_intensity has unsupported fields")
        light_type = rule.get("light_type", "DomeLight")
        if not isinstance(light_type, str) or light_type not in _LIGHT_TYPES:
            raise MaterialAdapterError("light_intensity has unsupported light_type")
        return {
            "kind": kind,
            "light_type": light_type,
            "factor": _factor(rule.get("factor", 0.5), allow_reduction_only=True),
        }
    if kind == "light_specular":
        if not set(rule) <= _LIGHT_RULE_FIELDS:
            raise MaterialAdapterError("light_specular has unsupported fields")
        light_type = rule.get("light_type", "DomeLight")
        if not isinstance(light_type, str) or light_type not in _LIGHT_TYPES:
            raise MaterialAdapterError("light_specular has unsupported light_type")
        return {
            "kind": kind,
            "light_type": light_type,
            "factor": _factor(rule.get("factor", 2.0), allow_reduction_only=False),
        }
    if kind == "background_albedo_offset":
        if not set(rule) <= _BACKGROUND_RULE_FIELDS:
            raise MaterialAdapterError("background_albedo_offset has unsupported fields")
        offset = rule.get("offset", -0.12)
        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            raise MaterialAdapterError("background albedo offset must be finite in [-1, 1]")
        offset = float(offset)
        if not math.isfinite(offset) or not -1 <= offset <= 1:
            raise MaterialAdapterError("background albedo offset must be finite in [-1, 1]")
        return {"kind": kind, "offset": offset}
    raise MaterialAdapterError(f"unsupported visual transform kind: {kind}")


@dataclass(frozen=True)
class _LightMaterial:
    prim_path: str


class _SessionPropertyTransaction:
    """One session-layer property transaction shared by light transforms."""

    kind = "SESSION_PROPERTY"

    def __init__(self, prim: Any, attribute_name: str, previous: float, *, stage: Any = None):
        self.prim = prim
        self.stage = stage if stage is not None else prim.GetStage()
        self.attribute_name = attribute_name
        self.previous = previous
        self.snapshot = _snapshot_properties(
            self.stage, prim.GetPath(), (attribute_name,)
        )
        self._restored = False
        self._restore_evidence: Mapping[str, Any] | None = None

    def apply(self, value: float) -> None:
        attr = self.prim.GetAttribute(self.attribute_name)
        try:
            with _session_edit_stage(self.stage):
                written = attr.Set(value)
                actual = attr.Get()
                if written is False or not math.isclose(
                    float(actual), value, rel_tol=1e-6, abs_tol=1e-6
                ):
                    raise MaterialAdapterError(
                        f"light transform write/readback failed: {self.attribute_name}"
                    )
        except Exception:
            try:
                self.restore()
            except Exception as restore_error:
                raise MaterialAdapterRollbackError(
                    f"light transform rollback failed: {restore_error}"
                ) from restore_error
            raise

    def restore(self) -> Mapping[str, Any]:
        if self._restored:
            return dict(self._restore_evidence or {})
        try:
            attr = self.prim.GetAttribute(self.attribute_name)
            with _session_edit_stage(self.stage):
                snapshot_evidence = self.snapshot.restore()
                actual = attr.Get()
                if actual is None or not math.isclose(
                    float(actual), self.previous, rel_tol=1e-6, abs_tol=1e-6
                ):
                    raise MaterialAdapterError(
                        f"light transform restore readback failed: {self.attribute_name}"
                    )
        except MaterialAdapterRollbackError:
            raise
        except Exception as exc:
            raise MaterialAdapterRollbackError(
                f"light transform rollback failed: {exc}"
            ) from exc
        self._restore_evidence = {
            "session_property_readback_passed": bool(
                snapshot_evidence["property_readback_passed"]
            ),
            "input_readback_passed": bool(snapshot_evidence["input_readback_passed"]),
            "source_readback_passed": True,
        }
        self._restored = True
        return dict(self._restore_evidence)


@dataclass
class _VisualChange:
    """Tuple-compatible visual change consumed by ``restore_material_changes``."""

    material: Any
    key: str
    previous: Any
    value: Any
    _transaction: _SessionPropertyTransaction
    evidence: Mapping[str, Any]

    @property
    def new(self) -> Any:
        return self.value

    def __iter__(self):
        yield self.material
        yield self.key
        yield self.previous
        yield self.value

    def restore(self) -> Mapping[str, Any]:
        evidence = self._transaction.restore()
        if isinstance(self.evidence, dict):
            self.evidence["restored"] = True
        return evidence


def _light_change(
    prim: Any,
    attribute_name: str,
    previous: float,
    value: float,
    evidence: Mapping[str, Any],
    *,
    stage: Any = None,
) -> _VisualChange:
    transaction = _SessionPropertyTransaction(prim, attribute_name, previous, stage=stage)
    transaction.apply(value)
    return _VisualChange(
        material=_LightMaterial(str(prim.GetPath())),
        key=attribute_name,
        previous=previous,
        value=value,
        _transaction=transaction,
        evidence=dict(evidence),
    )


def _light_plan(stage: Any, scene: Any, rule: Mapping[str, Any]):
    prim = _environment_light(stage, scene, rule["light_type"])
    attribute_name = "inputs:intensity" if rule["kind"] == "light_intensity" else "inputs:specular"
    _attr, previous = _light_attribute(
        prim, attribute_name, allow_zero=rule["kind"] == "light_specular"
    )
    value = float(previous * rule["factor"])
    if not math.isfinite(value) or value < 0:
        raise MaterialAdapterError("light transform result is not finite and nonnegative")
    evidence = {
        "kind": rule["kind"],
        "prim_path": str(prim.GetPath()),
        "attribute": attribute_name,
        "before": previous,
        "after": value,
        "factor": rule["factor"],
        "restored": False,
        "visual_review": "NOT_RUN",
    }
    if rule["kind"] == "light_specular":
        evidence["visual_effective"] = previous != 0.0
        evidence["baseline_zero"] = previous == 0.0
    return prim, attribute_name, previous, value, evidence


def _protected_keys(protected_objects: Iterable[Any] | None) -> tuple[set[int], set[str]]:
    identities: set[int] = set()
    names: set[str] = set()
    if isinstance(protected_objects, (str, bytes)):
        protected_objects = (protected_objects,)
    for obj in protected_objects or ():
        identities.add(id(obj))
        for attr in ("name", "prim_path"):
            value = getattr(obj, attr, None)
            if value is not None:
                names.add(str(value).rstrip("/"))
        if isinstance(obj, str):
            names.add(obj.rstrip("/"))
    return identities, names


def _is_protected(obj: Any, identities: set[int], names: set[str]) -> bool:
    if id(obj) in identities:
        return True
    values = {
        str(value).rstrip("/")
        for value in (getattr(obj, "name", None), getattr(obj, "prim_path", None))
        if value is not None
    }
    return bool(values & names)


def _fixed_base(obj: Any) -> bool:
    fixed = getattr(obj, "fixed_base", None)
    if fixed is None:
        fixed = getattr(obj, "_fixed_base", False)
    return bool(fixed)


def _materials(obj: Any) -> tuple[Any, ...]:
    values = getattr(obj, "materials", ())
    if isinstance(values, Mapping):
        values = values.values()
    return tuple(values)


def _background_material_plans(scene: Any, offset: float, protected_objects: Iterable[Any] | None):
    objects = _objects(scene)
    identities, names = _protected_keys(protected_objects)
    fixed_objects = [
        obj for obj in objects
        if _fixed_base(obj) and not _is_protected(obj, identities, names)
    ]
    if not fixed_objects:
        raise MaterialAdapterError("no eligible fixed-base background object")
    usage: dict[str, int] = {}
    for obj in objects:
        for material in _materials(obj):
            path = getattr(material, "prim_path", None)
            if path is not None:
                usage[str(path)] = usage.get(str(path), 0) + 1
    plans = []
    unsupported = False
    shared = False
    for obj in fixed_objects:
        for material in _materials(obj):
            path = getattr(material, "prim_path", None)
            if path is None:
                unsupported = True
                continue
            if usage.get(str(path), 0) != 1:
                shared = True
                continue
            if not is_vray_material(material):
                unsupported = True
                continue
            baseline = read_vray_input(material, "albedo_add", default=0.0)
            if baseline is None:
                baseline = 0.0
            desired = baseline + offset
            validate_vray_material_rules(material, {"albedo_add": desired})
            plans.append((material, {"albedo_add": desired}, baseline, offset))
    if not plans:
        if unsupported:
            raise MaterialAdapterError("unsupported fixed-base background material")
        if shared:
            raise MaterialAdapterError("all fixed-base background materials are shared")
        raise MaterialAdapterError("no eligible VRay fixed-base background material")
    return plans


def apply_visual_rules(
    evaluator: Any,
    rules: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
    protected_objects: Iterable[Any] | None,
) -> list[Any]:
    """Apply validated render-only rules and return reversible change records."""

    if rules is None:
        return []
    if isinstance(rules, Mapping):
        rules = (rules,)
    else:
        rules = tuple(rules)
    normalized = tuple(_normalize_rule(rule) for rule in rules)
    if not normalized:
        return []
    scene = getattr(getattr(evaluator, "env", None), "scene", None)
    if scene is None:
        raise MaterialAdapterError("evaluator scene is unavailable for visual transform")
    stage = (
        _stage_from_evaluator(evaluator)
        if any(rule["kind"] in {"light_intensity", "light_specular"} for rule in normalized)
        else None
    )
    plans = []
    seen_properties: set[tuple[str, str]] = set()
    for rule in normalized:
        if rule["kind"] in {"light_intensity", "light_specular"}:
            prim, attribute_name, previous, value, evidence = _light_plan(stage, scene, rule)
            property_key = (str(prim.GetPath()), attribute_name)
            if property_key in seen_properties:
                raise MaterialAdapterError(f"duplicate visual transform property: {property_key}")
            seen_properties.add(property_key)
            plans.append(("light", prim, attribute_name, previous, value, evidence))
        else:
            for material, inputs, baseline, offset in _background_material_plans(
                scene, rule["offset"], protected_objects
            ):
                property_key = (str(material.prim_path), "albedo_add")
                if property_key in seen_properties:
                    raise MaterialAdapterError(f"duplicate visual transform property: {property_key}")
                seen_properties.add(property_key)
                plans.append(("material", material, inputs, baseline, offset))
    changes: list[Any] = []
    try:
        for plan in plans:
            if plan[0] == "light":
                _, prim, attribute_name, previous, value, evidence = plan
                changes.append(
                    _light_change(
                        prim, attribute_name, previous, value, evidence, stage=stage
                    )
                )
            else:
                _, material, inputs, _baseline, _offset = plan
                new_changes = apply_vray_material_rules(material, inputs)
                changes.extend(new_changes)
    except Exception as exc:
        if changes:
            try:
                restore_material_changes(changes)
            except Exception as restore_error:
                raise MaterialAdapterRollbackError(
                    f"visual transform rollback failed: {restore_error}"
                ) from restore_error
        raise
    return changes


@contextmanager
def _temporary_light_property(stage: Any, scene: Any, rule: Mapping[str, Any], *, kind: str):
    normalized = _normalize_rule(rule)
    if normalized["kind"] != kind:
        raise ValueError(f"expected {kind} rule")
    prim, attribute_name, previous, value, evidence = _light_plan(stage, scene, normalized)
    try:
        from pxr import Usd
        import omnigibson as og
    except ImportError as exc:  # pragma: no cover - Isaac Sim supplies both modules.
        raise MaterialAdapterError("OmniGibson USD editing support is unavailable") from exc
    try:
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            with og.sim.editing_usd():
                written = prim.GetAttribute(attribute_name).Set(value)
            actual = prim.GetAttribute(attribute_name).Get()
            if written is False or not math.isclose(
                float(actual), value, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise MaterialAdapterError(
                    f"light transform write/readback failed: {attribute_name}"
                )
        yield evidence
    finally:
        try:
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                with og.sim.editing_usd():
                    restored = prim.GetAttribute(attribute_name).Set(previous)
                actual = prim.GetAttribute(attribute_name).Get()
                if restored is False or not math.isclose(
                    float(actual), previous, rel_tol=1e-6, abs_tol=1e-6
                ):
                    raise MaterialAdapterError(
                        f"light transform restore readback failed: {attribute_name}"
                    )
        except MaterialAdapterRollbackError:
            raise
        except Exception as exc:
            raise MaterialAdapterRollbackError(
                f"light transform rollback failed: {exc}"
            ) from exc
        evidence["restored"] = True


@contextmanager
def temporary_light_intensity(stage, scene, rule):
    """Compatibility context manager for one render-only light intensity edit."""

    with _temporary_light_property(stage, scene, rule, kind="light_intensity") as evidence:
        yield evidence


@contextmanager
def temporary_light_specular(stage, scene, rule):
    """Render-only environment-light specular edit with the same rollback contract."""

    with _temporary_light_property(stage, scene, rule, kind="light_specular") as evidence:
        yield evidence
