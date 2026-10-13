"""Task-independent committed ``Inside`` relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


def _validated_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(slots=True)
class SemanticInsideLedger:
    """Record the latest private container choice for each object."""

    _object_to_container: dict[str, str] = field(default_factory=dict)

    VERSION = "private_semantic_inside_ledger/v1"

    def commit(self, *, object_name: str, container_name: str) -> None:
        object_key = _validated_name(object_name, field_name="object_name")
        container_key = _validated_name(
            container_name, field_name="container_name"
        )
        if object_key == container_key:
            raise ValueError("an object cannot be semantically inside itself")
        self._object_to_container[object_key] = container_key

    def contains(self, *, object_name: str, container_name: str) -> bool:
        object_key = _validated_name(object_name, field_name="object_name")
        container_key = _validated_name(
            container_name, field_name="container_name"
        )
        return self._object_to_container.get(object_key) == container_key

    def discard(self, *, object_name: str) -> None:
        """Forget the old parent when an explicit action picks the child."""

        object_key = _validated_name(object_name, field_name="object_name")
        self._object_to_container.pop(object_key, None)

    @property
    def relation_count(self) -> int:
        return len(self._object_to_container)

    def relations(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._object_to_container.items()))

    def private_record(self) -> Mapping[str, object]:
        return {
            "verifier_version": self.VERSION,
            "relation_count": self.relation_count,
            "relations": [
                {"object": object_name, "container": container_name}
                for object_name, container_name in self.relations()
            ],
            "official_predicate_verified": False,
            "physical_placement_verified": False,
            "official_q": None,
        }
