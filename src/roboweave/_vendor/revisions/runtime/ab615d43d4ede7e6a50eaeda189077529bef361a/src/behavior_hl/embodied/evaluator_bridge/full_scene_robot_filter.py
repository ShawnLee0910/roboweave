"""Isolated full-scene filtering for stale task-template robots.

Some BEHAVIOR 2026 task templates contain a serialized generic ``Robot``
object in category ``agent``.  The official evaluator separately supplies
the configured R1Pro.  During a full-scene load the serialized agent can
otherwise occupy the scene's only robot slot and suppress construction of
the configured evaluator robot.

This module is pure Python.  It temporarily wraps the function referenced by
``omnigibson.eval.evaluator`` only while one Evaluator is constructed.  The
official source tree and the QA launch path remain unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, MutableMapping


EMBEDDED_SCENE_AGENT_CATEGORY = "agent"
FULL_SCENE_ROBOT_FILTER_VERSION = (
    "behavior2026.full-scene-robot-filter/v3"
)
SERIALIZED_GENERIC_ROBOT_CLASS_NAMES = frozenset({"Robot", "BaseRobot"})


class FullSceneRobotFilterError(RuntimeError):
    """Raised when the temporary evaluator hook cannot be installed safely."""


@dataclass(frozen=True, slots=True)
class FullSceneRobotFilterReport:
    enabled: bool
    filtered_category: str | None
    generated_config_count: int
    restored_scene_init_count: int
    filtered_scene_robot_count: int
    filtered_scene_robot_models: tuple[str, ...]
    object_filter_installed: bool
    object_filter_restored: bool
    restored: bool
    version: str = FULL_SCENE_ROBOT_FILTER_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "enabled": self.enabled,
            "filtered_category": self.filtered_category,
            "generated_config_count": self.generated_config_count,
            "restored_scene_init_count": self.restored_scene_init_count,
            "filtered_scene_robot_count": self.filtered_scene_robot_count,
            "filtered_scene_robot_models": list(
                self.filtered_scene_robot_models
            ),
            "object_filter_installed": self.object_filter_installed,
            "object_filter_restored": self.object_filter_restored,
            "restored": self.restored,
        }


def serialized_generic_robot_model(
    obj_info: Mapping[str, Any],
) -> str | None:
    """Return the model of a legacy generic serialized robot, if present.

    Some 2026 cached task templates serialize an R1Pro as ``class_name:
    Robot`` with the concrete class only in ``args.model``.  OmniGibson's
    normal ``include_robots=False`` guard checks ``class_name`` against the
    registered concrete robot classes, so this legacy representation slips
    through and suppresses the evaluator-configured robot.

    This detector is deliberately narrow: ordinary DatasetObject entries
    also carry a model identifier and must never be treated as robots.
    """

    if not isinstance(obj_info, Mapping):
        raise FullSceneRobotFilterError("scene object init info must be a mapping")
    class_name = obj_info.get("class_name")
    if class_name not in SERIALIZED_GENERIC_ROBOT_CLASS_NAMES:
        return None
    args = obj_info.get("args")
    if not isinstance(args, Mapping):
        raise FullSceneRobotFilterError(
            "serialized generic robot has no args mapping"
        )
    model = args.get("model", args.get("type"))
    if not isinstance(model, str) or not model.strip():
        raise FullSceneRobotFilterError(
            "serialized generic robot has no concrete model"
        )
    return model.strip().lower()


def add_scene_object_category_blacklist(
    config: MutableMapping[str, Any],
    category: str,
) -> MutableMapping[str, Any]:
    """Add one category without removing any official scene exclusions."""

    if not isinstance(config, MutableMapping):
        raise FullSceneRobotFilterError("environment config must be mutable")
    if not isinstance(category, str) or not category:
        raise FullSceneRobotFilterError("category must be a non-empty string")
    scene = config.get("scene")
    if not isinstance(scene, MutableMapping):
        raise FullSceneRobotFilterError(
            "environment config has no mutable scene mapping"
        )
    existing = scene.get("not_load_object_categories")
    if existing is None:
        values: list[str] = []
    elif isinstance(existing, str):
        values = [existing]
    else:
        try:
            values = list(existing)
        except TypeError as exc:
            raise FullSceneRobotFilterError(
                "scene category blacklist is not iterable"
            ) from exc
    if any(not isinstance(value, str) or not value for value in values):
        raise FullSceneRobotFilterError(
            "scene category blacklist contains an invalid entry"
        )
    if category not in values:
        values.append(category)
    scene["not_load_object_categories"] = values
    return config


@contextmanager
def temporary_full_scene_robot_filter(
    evaluator_module: Any,
    *,
    enabled: bool,
) -> Iterator[dict[str, Any]]:
    """Filter serialized agents for one full-scene Evaluator construction.

    The yielded dictionary is finalized on context exit so callers can write
    it directly into private lifecycle evidence after construction.
    """

    report = {
        "version": FULL_SCENE_ROBOT_FILTER_VERSION,
        "enabled": bool(enabled),
        "filtered_category": (
            EMBEDDED_SCENE_AGENT_CATEGORY if enabled else None
        ),
        "generated_config_count": 0,
        "restored_scene_init_count": 0,
        "filtered_scene_robot_count": 0,
        "filtered_scene_robot_models": [],
        "object_filter_installed": False,
        "object_filter_restored": True,
        "restored": True,
    }
    if not enabled:
        yield report
        return

    original = getattr(
        evaluator_module,
        "generate_basic_environment_config",
        None,
    )
    if not callable(original):
        raise FullSceneRobotFilterError(
            "evaluator generate_basic_environment_config is unavailable"
        )

    def wrapped(*args: Any, **kwargs: Any):
        config = original(*args, **kwargs)
        add_scene_object_category_blacklist(
            config,
            EMBEDDED_SCENE_AGENT_CATEGORY,
        )
        report["generated_config_count"] += 1
        return config

    evaluator_module.generate_basic_environment_config = wrapped

    # Cached BEHAVIOR task scenes are restored from JSON.  The official
    # InteractiveTraversableScene restoration hook unconditionally rewrites
    # ``include_robots=True`` even when the evaluator config requested
    # ``False``.  Tasks whose BDDL contains an agent therefore resurrect the
    # serialized, camera-less authoring robot and it can replace the
    # evaluator's configured R1Pro.  Patch only the registered scene class for
    # this one construction and restore its descriptor afterwards.  The
    # evaluator-provided robot remains the sole live agent and receives the
    # predeclared camera config.
    scene_cls = None
    scene_descriptor = None
    should_load_descriptor = None
    og_module = getattr(evaluator_module, "og", None)
    registered_scenes = getattr(og_module, "REGISTERED_SCENES", None)
    if isinstance(registered_scenes, Mapping):
        scene_cls = registered_scenes.get("InteractiveTraversableScene")
    if scene_cls is None:
        evaluator_module.generate_basic_environment_config = original
        raise FullSceneRobotFilterError(
            "registered InteractiveTraversableScene is unavailable"
        )

    scene_descriptor = vars(scene_cls).get(
        "modify_init_info_for_restoring"
    )
    original_scene_hook = getattr(
        scene_cls, "modify_init_info_for_restoring", None
    )
    should_load_descriptor = vars(scene_cls).get("_should_load_object")
    original_should_load_object = getattr(
        scene_cls, "_should_load_object", None
    )
    if (
        not callable(original_scene_hook)
        or scene_descriptor is None
        or not callable(original_should_load_object)
        or should_load_descriptor is None
    ):
        evaluator_module.generate_basic_environment_config = original
        raise FullSceneRobotFilterError(
            "registered scene robot filtering hooks are unavailable"
        )

    def filtered_scene_restore_hook(cls, init_info):
        original_scene_hook(init_info=init_info)
        args = init_info.get("args")
        if not isinstance(args, MutableMapping):
            raise FullSceneRobotFilterError(
                "restored scene init_info has no mutable args mapping"
            )
        args["include_robots"] = False
        report["restored_scene_init_count"] += 1

    def filtered_should_load_object(self, obj_info, task_metadata):
        model = serialized_generic_robot_model(obj_info)
        if model is not None:
            report["filtered_scene_robot_count"] += 1
            if model not in report["filtered_scene_robot_models"]:
                report["filtered_scene_robot_models"].append(model)
            return False
        return original_should_load_object(
            self,
            obj_info,
            task_metadata,
        )

    scene_cls.modify_init_info_for_restoring = classmethod(
        filtered_scene_restore_hook
    )
    scene_cls._should_load_object = filtered_should_load_object
    report["object_filter_installed"] = True
    report["object_filter_restored"] = False
    report["restored"] = False
    try:
        yield report
    finally:
        scene_cls.modify_init_info_for_restoring = scene_descriptor
        scene_cls._should_load_object = should_load_descriptor
        report["object_filter_restored"] = True
        evaluator_module.generate_basic_environment_config = original
        report["filtered_scene_robot_models"].sort()
        report["restored"] = True


def require_full_scene_robot_filter_report(
    report: Mapping[str, Any],
    *,
    enabled: bool,
) -> None:
    """Fail closed if the one-construction hook did not run exactly once."""

    expected_count = 1 if enabled else 0
    restored_scene_init_count = report.get("restored_scene_init_count")
    filtered_scene_robot_count = report.get("filtered_scene_robot_count")
    filtered_scene_robot_models = report.get("filtered_scene_robot_models")
    if (
        report.get("version") != FULL_SCENE_ROBOT_FILTER_VERSION
        or report.get("enabled") is not bool(enabled)
        or report.get("generated_config_count") != expected_count
        or not isinstance(restored_scene_init_count, int)
        or restored_scene_init_count < 0
        or not isinstance(filtered_scene_robot_count, int)
        or filtered_scene_robot_count < 0
        or not isinstance(filtered_scene_robot_models, list)
        or any(
            not isinstance(model, str) or not model
            for model in filtered_scene_robot_models
        )
        or report.get("object_filter_installed") is not bool(enabled)
        or report.get("object_filter_restored") is not True
        or report.get("restored") is not True
    ):
        raise FullSceneRobotFilterError(
            "full-scene robot filter attestation is incomplete"
        )


__all__ = [
    "EMBEDDED_SCENE_AGENT_CATEGORY",
    "FULL_SCENE_ROBOT_FILTER_VERSION",
    "FullSceneRobotFilterError",
    "add_scene_object_category_blacklist",
    "require_full_scene_robot_filter_report",
    "serialized_generic_robot_model",
    "temporary_full_scene_robot_filter",
]
