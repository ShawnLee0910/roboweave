"""Required episode artifact inventory for a physical task rollout."""

from __future__ import annotations

from dataclasses import dataclass

from behavior_hl.embodied.runtime.artifacts import ArtifactInventory


_EPISODE_REQUIRED = frozenset(
    {
        "initial_frame",
        "events",
        "model_raw_responses",
        "parsed_actions",
        "initial_predicates",
        "final_predicate_report",
        "final_state",
        "episode_video",
        "manifest",
    }
)


@dataclass(frozen=True, slots=True)
class EpisodeArtifactContract:
    action_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.action_count, bool)
            or not isinstance(self.action_count, int)
            or self.action_count < 0
        ):
            raise ValueError("action_count must be non-negative")

    @property
    def required_logical_names(self) -> tuple[str, ...]:
        names = set(_EPISODE_REQUIRED)
        for index in range(self.action_count):
            prefix = f"step_{index:04d}"
            names.update(
                {
                    f"{prefix}_action_before",
                    f"{prefix}_action_after",
                    f"{prefix}_settle_final",
                    f"{prefix}_verifier",
                }
            )
        return tuple(sorted(names))

    def require_complete(self, inventory: ArtifactInventory) -> None:
        inventory.require(self.required_logical_names)
