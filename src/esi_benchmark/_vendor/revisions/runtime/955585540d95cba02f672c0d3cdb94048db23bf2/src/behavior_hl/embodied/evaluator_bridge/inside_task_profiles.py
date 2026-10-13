"""Frozen public/private profiles for marker-bound Inside tasks.

The public instruction is safe for the policy. Synsets, object counts, scene
models, and goal-leaf counts are evaluator-private routing metadata and must
never be serialized into a model prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


INSIDE_TASK_PROFILE_VERSION = "behavior2026.inside-task-profile/v2"


@dataclass(frozen=True, slots=True)
class InsideTaskProfile:
    task_name: str
    task_id: int
    task_language: str
    scene_model: str
    split: str
    instance_id: int
    graspable_synsets: frozenset[str]
    container_synsets: frozenset[str]
    public_category_by_synset: Mapping[str, str]
    expected_graspable_count: int
    expected_container_count: int
    expected_admitted_count: int
    expected_goal_leaf_count: int
    canonical_option_index: int = 0
    canonical_live_option_index: int | None = None
    canonical_goal_leaves: tuple[str, ...] = ()
    goal_signatures: tuple[str, ...] = ("Inside",)
    schema_version: str = INSIDE_TASK_PROFILE_VERSION

    def __post_init__(self) -> None:
        for name in ("task_name", "task_language", "scene_model", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "task_id",
            "instance_id",
            "expected_graspable_count",
            "expected_container_count",
            "expected_admitted_count",
            "expected_goal_leaf_count",
        ):
            value = getattr(self, name)
            minimum = 0 if name in {"task_id", "instance_id"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        graspable = frozenset(self.graspable_synsets)
        containers = frozenset(self.container_synsets)
        if not graspable or not containers:
            raise ValueError("graspable and container synsets must be non-empty")
        object.__setattr__(self, "graspable_synsets", graspable)
        object.__setattr__(self, "container_synsets", containers)
        categories = dict(self.public_category_by_synset)
        if set(categories) != set(graspable):
            raise ValueError("public categories must cover exactly the graspable synsets")
        if any(not isinstance(value, str) or not value for value in categories.values()):
            raise ValueError("public category labels must be non-empty strings")
        object.__setattr__(
            self,
            "public_category_by_synset",
            MappingProxyType(categories),
        )
        if self.expected_admitted_count > (
            self.expected_graspable_count + self.expected_container_count
        ):
            raise ValueError("admitted count exceeds the union count upper bound")
        if self.canonical_option_index < 0:
            raise ValueError("canonical option index must be non-negative")
        if self.canonical_live_option_index is not None and (
            isinstance(self.canonical_live_option_index, bool)
            or not isinstance(self.canonical_live_option_index, int)
            or self.canonical_live_option_index < 0
        ):
            raise ValueError("canonical live option index must be non-negative")
        canonical_leaves = tuple(self.canonical_goal_leaves)
        if len(canonical_leaves) != self.expected_goal_leaf_count or any(
            not isinstance(leaf, str) or not leaf for leaf in canonical_leaves
        ):
            raise ValueError("canonical goal leaves must cover one grounded option")
        object.__setattr__(self, "canonical_goal_leaves", canonical_leaves)
        signatures = tuple(self.goal_signatures)
        if not signatures or any(
            signature
            not in {
                "Inside",
                "Closed+Inside",
                "Inside+NotInside",
                "Inside+PreservedOnTop",
                "MixedInsideOnTopClosed",
                "MixedInsideNextToClosed",
                "MixedAction23",
            }
            for signature in signatures
        ):
            raise ValueError("profile contains an unsupported first-tier signature")
        object.__setattr__(self, "goal_signatures", signatures)
        if self.schema_version != INSIDE_TASK_PROFILE_VERSION:
            raise ValueError("unexpected Inside task profile version")

    def public_record(self) -> Mapping[str, Any]:
        """Return only fields allowed to cross the evaluator-policy boundary."""

        return MappingProxyType(
            {
                "task_name": self.task_name,
                "task_language": self.task_language,
            }
        )

    def private_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": self.schema_version,
                "task_name": self.task_name,
                "task_id": self.task_id,
                "scene_model": self.scene_model,
                "split": self.split,
                "instance_id": self.instance_id,
                "graspable_synsets": sorted(self.graspable_synsets),
                "container_synsets": sorted(self.container_synsets),
                "public_category_by_synset": dict(self.public_category_by_synset),
                "expected_graspable_count": self.expected_graspable_count,
                "expected_container_count": self.expected_container_count,
                "expected_admitted_count": self.expected_admitted_count,
                "expected_goal_leaf_count": self.expected_goal_leaf_count,
                "canonical_option_index": self.canonical_option_index,
                "canonical_live_option_index": self.canonical_live_option_index,
                "canonical_goal_leaves": list(self.canonical_goal_leaves),
                "goal_signatures": list(self.goal_signatures),
            }
        )


def _profile(
    *,
    task_name: str,
    task_id: int,
    task_language: str,
    scene_model: str,
    graspable: Mapping[str, str],
    containers: frozenset[str],
    graspable_count: int,
    container_count: int = 1,
    instance_id: int = 0,
) -> InsideTaskProfile:
    return InsideTaskProfile(
        task_name=task_name,
        task_id=task_id,
        task_language=task_language,
        scene_model=scene_model,
        split="train",
        instance_id=instance_id,
        graspable_synsets=frozenset(graspable),
        container_synsets=containers,
        public_category_by_synset=graspable,
        expected_graspable_count=graspable_count,
        expected_container_count=container_count,
        expected_admitted_count=graspable_count + container_count,
        expected_goal_leaf_count=graspable_count,
        canonical_goal_leaves=tuple(
            f'[["inside","legacy_{index}","legacy_container"]]'
            for index in range(graspable_count)
        ),
    )


_LEGACY_INSIDE_TASK_PROFILES = MappingProxyType(
    {
        "picking_up_toys": _profile(
            task_name="picking_up_toys",
            task_id=7,
            task_language=(
                "Pick up all board games, jigsaw puzzles, and the tennis ball, "
                "and place them completely inside a toy box."
            ),
            scene_model="house_single_floor",
            graspable={
                "board_game.n.01": "board_game",
                "jigsaw_puzzle.n.01": "jigsaw_puzzle",
                "tennis_ball.n.01": "tennis_ball",
            },
            containers=frozenset({"toy_box.n.01"}),
            graspable_count=6,
        ),
        "composting_waste": _profile(
            task_name="composting_waste",
            task_id=90,
            task_language=(
                "Put the banana half and pomegranate half completely inside "
                "the ashcan for composting."
            ),
            scene_model="Rs_int",
            graspable={
                "half__banana.n.01": "banana_half",
                "half__pomegranate.n.01": "pomegranate_half",
            },
            containers=frozenset({"ashcan.n.01"}),
            graspable_count=2,
            instance_id=1,
        ),
        "picking_up_trash": _profile(
            task_name="picking_up_trash",
            task_id=1,
            task_language="Put all three soda cans completely inside the ashcan.",
            scene_model="house_double_floor_lower",
            graspable={"can__of__soda.n.01": "soda_can"},
            containers=frozenset({"ashcan.n.01"}),
            graspable_count=3,
        ),
        "putting_dirty_dishes_in_sink": _profile(
            task_name="putting_dirty_dishes_in_sink",
            task_id=85,
            task_language=(
                "Put all bowls and plates completely inside the sink."
            ),
            scene_model="restaurant_diner",
            graspable={"bowl.n.01": "bowl", "plate.n.04": "plate"},
            containers=frozenset({"sink.n.01"}),
            graspable_count=4,
            instance_id=1,
        ),
        "collecting_aluminum_cans": _profile(
            task_name="collecting_aluminum_cans",
            task_id=87,
            task_language="Collect all six soda cans completely inside the bucket.",
            scene_model="hotel_suite_large",
            graspable={"can__of__soda.n.01": "soda_can"},
            containers=frozenset({"bucket.n.01"}),
            graspable_count=6,
            instance_id=1,
        ),
        "putting_away_toys": _profile(
            task_name="putting_away_toys",
            task_id=54,
            task_language="Put every toy figure completely inside either toy box.",
            scene_model="house_single_floor",
            graspable={"toy_figure.n.01": "toy_figure"},
            containers=frozenset({"toy_box.n.01"}),
            graspable_count=8,
            container_count=2,
            instance_id=1,
        ),
        "boxing_books_up_for_storage": _profile(
            task_name="boxing_books_up_for_storage",
            task_id=23,
            task_language="Put all six books completely inside the storage box.",
            scene_model="house_double_floor_upper",
            graspable={"book.n.02": "book"},
            containers=frozenset({"box.n.01"}),
            graspable_count=6,
        ),
        "dispose_of_glass": _profile(
            task_name="dispose_of_glass",
            task_id=76,
            task_language="Put all four water glasses completely inside the ashcan.",
            scene_model="hotel_suite_large",
            graspable={"water_glass.n.02": "water_glass"},
            containers=frozenset({"ashcan.n.01"}),
            graspable_count=4,
            instance_id=1,
        ),
        "clean_up_broken_glass": _profile(
            task_name="clean_up_broken_glass",
            task_id=80,
            task_language=(
                "Put all three visible pieces of broken glass completely "
                "inside the ashcan."
            ),
            scene_model="restaurant_diner",
            graspable={"broken__glass.n.01": "broken_glass"},
            containers=frozenset({"ashcan.n.01"}),
            graspable_count=3,
            instance_id=1,
        ),
    }
)


def _load_b100_inside27_profiles() -> Mapping[str, InsideTaskProfile]:
    path = (
        Path(__file__).resolve().parents[4]
        / "configs"
        / "b100_inside27_profiles.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "behavior2026.inside27-profile-catalog/v1":
        raise RuntimeError("unexpected B100 Inside27 catalog schema")
    records = payload.get("tasks")
    declared_count = payload.get("task_count")
    if (
        not isinstance(records, list)
        or isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count < 1
        or len(records) != declared_count
    ):
        raise RuntimeError(
            "B100 Inside profile catalog task_count does not match its records"
        )
    result: dict[str, InsideTaskProfile] = {}
    for record in records:
        name = record["task_name"]
        legacy = _LEGACY_INSIDE_TASK_PROFILES.get(name)
        profile = InsideTaskProfile(
            task_name=name,
            task_id=int(record["task_id"]),
            task_language=(
                legacy.task_language if legacy is not None else record["task_language"]
            ),
            scene_model=record["scene_model"],
            split=record["split"],
            instance_id=int(record["instance_id"]),
            graspable_synsets=frozenset(record["graspable_synsets"]),
            container_synsets=frozenset(record["container_synsets"]),
            public_category_by_synset=record["public_category_by_synset"],
            expected_graspable_count=int(record["expected_graspable_count"]),
            expected_container_count=int(record["expected_container_count"]),
            expected_admitted_count=int(record["expected_admitted_count"]),
            expected_goal_leaf_count=int(record["expected_goal_leaf_count"]),
            canonical_option_index=int(record["canonical_option_index"]),
            canonical_live_option_index=(
                None
                if record.get("canonical_live_option_index") is None
                else int(record["canonical_live_option_index"])
            ),
            canonical_goal_leaves=tuple(
                json.dumps(leaf, sort_keys=True, separators=(",", ":"))
                for leaf in record["canonical_goal_leaves"]
            ),
            goal_signatures=tuple(record["goal_signatures"]),
        )
        if name in result:
            raise RuntimeError(f"duplicate Inside27 profile {name!r}")
        result[name] = profile
    return MappingProxyType(result)


def _load_priority_mixed_profiles() -> Mapping[str, InsideTaskProfile]:
    path = (
        Path(__file__).resolve().parents[4]
        / "configs"
        / "b100_priority_mixed_profiles.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "behavior2026.priority-mixed-profile-catalog/v1":
        raise RuntimeError("unexpected priority mixed profile catalog schema")
    records = payload.get("tasks")
    if not isinstance(records, list) or len(records) != payload.get("task_count"):
        raise RuntimeError("priority mixed profile task_count does not match records")
    result: dict[str, InsideTaskProfile] = {}
    for record in records:
        profile = InsideTaskProfile(
            task_name=record["task_name"],
            task_id=int(record["task_id"]),
            task_language=record["task_language"],
            scene_model=record["scene_model"],
            split=record["split"],
            instance_id=int(record["instance_id"]),
            graspable_synsets=frozenset(record["graspable_synsets"]),
            container_synsets=frozenset(record["container_synsets"]),
            public_category_by_synset=record["public_category_by_synset"],
            expected_graspable_count=int(record["expected_graspable_count"]),
            expected_container_count=int(record["expected_container_count"]),
            expected_admitted_count=int(record["expected_admitted_count"]),
            expected_goal_leaf_count=int(record["expected_goal_leaf_count"]),
            canonical_option_index=int(record["canonical_option_index"]),
            canonical_live_option_index=(
                None
                if record.get("canonical_live_option_index") is None
                else int(record["canonical_live_option_index"])
            ),
            canonical_goal_leaves=tuple(
                json.dumps(leaf, sort_keys=True, separators=(",", ":"))
                for leaf in record["canonical_goal_leaves"]
            ),
            goal_signatures=tuple(record["goal_signatures"]),
        )
        if profile.task_name in result:
            raise RuntimeError(f"duplicate priority mixed profile {profile.task_name!r}")
        result[profile.task_name] = profile
    return MappingProxyType(result)


def _load_action23_oracle_profiles() -> Mapping[str, InsideTaskProfile]:
    """Load task-agnostic profiles generated from official BDDL + TRO state."""

    path = (
        Path(__file__).resolve().parents[4]
        / "configs"
        / "b100_action23_oracle_profiles.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "behavior2026.action23-oracle-profile-catalog/v1"
    ):
        raise RuntimeError("unexpected Action23 Oracle profile catalog schema")
    records = payload.get("tasks")
    if not isinstance(records, list) or len(records) != payload.get("task_count"):
        raise RuntimeError("Action23 Oracle profile task_count does not match records")
    result: dict[str, InsideTaskProfile] = {}
    for record in records:
        profile = InsideTaskProfile(
            task_name=record["task_name"],
            task_id=int(record["task_id"]),
            task_language=record["task_language"],
            scene_model=record["scene_model"],
            split=record["split"],
            instance_id=int(record["instance_id"]),
            graspable_synsets=frozenset(record["graspable_synsets"]),
            container_synsets=frozenset(record["container_synsets"]),
            public_category_by_synset=record["public_category_by_synset"],
            expected_graspable_count=int(record["expected_graspable_count"]),
            expected_container_count=int(record["expected_container_count"]),
            expected_admitted_count=int(record["expected_admitted_count"]),
            expected_goal_leaf_count=int(record["expected_goal_leaf_count"]),
            canonical_option_index=int(record["canonical_option_index"]),
            canonical_live_option_index=(
                None
                if record.get("canonical_live_option_index") is None
                else int(record["canonical_live_option_index"])
            ),
            canonical_goal_leaves=tuple(
                json.dumps(leaf, sort_keys=True, separators=(",", ":"))
                for leaf in record["canonical_goal_leaves"]
            ),
            goal_signatures=tuple(record["goal_signatures"]),
        )
        if profile.task_name in result:
            raise RuntimeError(
                f"duplicate Action23 Oracle profile {profile.task_name!r}"
            )
        result[profile.task_name] = profile
    return MappingProxyType(result)


_inside_profiles = dict(_load_b100_inside27_profiles())
for _task_name, _profile_value in _load_priority_mixed_profiles().items():
    if _task_name in _inside_profiles:
        raise RuntimeError(f"duplicate task profile {_task_name!r}")
    _inside_profiles[_task_name] = _profile_value
for _task_name, _profile_value in _load_action23_oracle_profiles().items():
    if _task_name in _inside_profiles:
        raise RuntimeError(f"duplicate task profile {_task_name!r}")
    _inside_profiles[_task_name] = _profile_value
INSIDE_TASK_PROFILES = MappingProxyType(_inside_profiles)


def get_inside_task_profile(task_name: str) -> InsideTaskProfile:
    try:
        return INSIDE_TASK_PROFILES[task_name]
    except KeyError as exc:
        raise KeyError(f"unsupported Inside task profile: {task_name!r}") from exc


def _scope(evaluator: Any) -> Mapping[str, Any]:
    try:
        scope = evaluator.env.task.object_scope
    except Exception as exc:
        raise RuntimeError("Inside task object_scope is unavailable") from exc
    if not isinstance(scope, Mapping):
        raise RuntimeError("Inside task object_scope must be a mapping")
    return scope


def _synset(scope_name: str) -> str | None:
    synset, separator, instance = scope_name.rpartition("_")
    if not separator or not instance.isdigit():
        return None
    return synset


def private_inside_task_objects(
    evaluator: Any,
    profile: InsideTaskProfile,
) -> Mapping[str, Any]:
    admitted = profile.graspable_synsets.union(profile.container_synsets)
    result: dict[str, Any] = {}
    for scope_name, entity in _scope(evaluator).items():
        if entity is None or not isinstance(scope_name, str):
            continue
        if _synset(scope_name) not in admitted:
            continue
        name = getattr(entity, "name", None)
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"Inside task object {scope_name!r} has no simulator name")
        if name in result and result[name] is not entity:
            raise RuntimeError(f"duplicate simulator object name {name!r}")
        result[name] = entity
    expected = profile.expected_admitted_count
    if len(result) != expected:
        raise RuntimeError(
            f"{profile.task_name} expected {expected} admitted objects; found {len(result)}"
        )
    return MappingProxyType(result)


def private_inside_task_roles(
    evaluator: Any,
    profile: InsideTaskProfile,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    all_objects = private_inside_task_objects(evaluator, profile)
    graspable: dict[str, Any] = {}
    containers: dict[str, Any] = {}
    for scope_name, entity in _scope(evaluator).items():
        if entity is None or not isinstance(scope_name, str):
            continue
        synset = _synset(scope_name)
        name = getattr(entity, "name", None)
        if name not in all_objects:
            continue
        if synset in profile.graspable_synsets:
            graspable[name] = entity
        if synset in profile.container_synsets:
            containers[name] = entity
    if len(graspable) != profile.expected_graspable_count:
        raise RuntimeError(
            f"{profile.task_name} expected {profile.expected_graspable_count} graspables; "
            f"found {len(graspable)}"
        )
    if len(containers) != profile.expected_container_count:
        raise RuntimeError(
            f"{profile.task_name} expected {profile.expected_container_count} containers; "
            f"found {len(containers)}"
        )
    return MappingProxyType(graspable), MappingProxyType(containers)


def private_inside_task_objects_by_category(
    evaluator: Any,
    profile: InsideTaskProfile,
) -> Mapping[str, Mapping[str, Any]]:
    graspable, _ = private_inside_task_roles(evaluator, profile)
    result: dict[str, dict[str, Any]] = {
        category: {} for category in profile.public_category_by_synset.values()
    }
    for scope_name, entity in _scope(evaluator).items():
        if entity is None or not isinstance(scope_name, str):
            continue
        category = profile.public_category_by_synset.get(_synset(scope_name))
        name = getattr(entity, "name", None)
        if category is not None and name in graspable:
            result[category][name] = entity
    return MappingProxyType(
        {key: MappingProxyType(value) for key, value in result.items()}
    )
