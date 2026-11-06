"""Validate split profile composition without importing Hydra or a simulator."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from behavior_hl.embodied.runtime.episode import EpisodeSpec


@dataclass(frozen=True, slots=True)
class EpisodeProfiles:
    robot: Mapping[str, Any]
    observation: Mapping[str, Any]
    actions: Mapping[str, Any]
    model: Mapping[str, Any]
    task: Mapping[str, Any]
    suite: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "robot",
            "observation",
            "actions",
            "model",
            "task",
            "suite",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{field_name} profile must be a mapping")
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(value)),
            )
        expected_task = self.suite.get("task_profile")
        if expected_task != self.task.get("profile_id"):
            raise ValueError("suite task_profile does not match task profile")
        for name, profile in (
            ("robot", self.robot),
            ("observation", self.observation),
            ("actions", self.actions),
            ("model", self.model),
        ):
            expected = self.suite.get(f"{name}_profile")
            if expected != profile.get("profile_id"):
                raise ValueError(
                    f"suite {name}_profile does not match {name} profile"
                )

    def episode(
        self,
        *,
        split: str,
        instance_id: str,
        seed: int,
    ) -> EpisodeSpec:
        task_id = str(self.task["task_id"])
        suite_id = str(self.suite["profile_id"])
        return EpisodeSpec(
            episode_id=f"{suite_id}/{task_id}/{split}/{instance_id}/{seed}",
            task_id=task_id,
            split=split,
            instance_id=instance_id,
            seed=seed,
            instruction=str(self.task["instruction"]),
            action_profile=str(self.actions["profile_id"]),
            robot_profile=str(self.robot["profile_id"]),
            observation_profile=str(self.observation["profile_id"]),
            model_profile=str(self.model["profile_id"]),
            action_schema_version=str(
                self.actions["action_schema_version"]
            ),
        )
