"""Extensible action registry and schema generation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from behavior_hl.embodied.policy.protocol import (
    ActionArguments,
    ActionEnvelope,
    ActionProtocolError,
    decode_json_object,
    parse_reasoning,
)


ArgParser = Callable[[Mapping[str, Any]], ActionArguments]
ACTION_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """One independently registered public action."""

    name: str
    parse_args: ArgParser
    args_schema: Mapping[str, Any]
    args_required: bool
    model_description: str
    required_capabilities: frozenset[str]
    max_simulator_steps: int
    public_statuses: tuple[str, ...]
    invariants: frozenset[str]
    controller_key: str

    def __post_init__(self) -> None:
        if not ACTION_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid action name: {self.name!r}")
        if not self.model_description.strip():
            raise ValueError(f"action {self.name!r} has no model description")
        if self.max_simulator_steps <= 0:
            raise ValueError(
                f"action {self.name!r} must have a positive simulator-step limit"
            )

    def manifest(self) -> dict[str, Any]:
        manifest = {
            "name": self.name,
            "args_schema": dict(self.args_schema),
            "args_required": self.args_required,
            "model_description": self.model_description,
            "required_capabilities": sorted(self.required_capabilities),
            "max_simulator_steps": self.max_simulator_steps,
            "public_statuses": list(self.public_statuses),
            "invariants": sorted(self.invariants),
            "controller_key": self.controller_key,
        }
        if "kinematic_research" in self.invariants:
            manifest.update(
                {
                    "kinematic_research": True,
                    "official_challenge_eligible": False,
                }
            )
        return manifest

    def model_manifest(self) -> dict[str, Any]:
        """Return action syntax without executor outcomes or private routing."""

        return {
            "name": self.name,
            "args_schema": dict(self.args_schema),
            "args_required": self.args_required,
            "model_description": self.model_description,
        }


class ActionRegistry:
    """Frozen set of task-enabled ActionSpec plugins."""

    def __init__(
        self,
        specs: Iterable[ActionSpec],
        *,
        schema_version: str,
    ) -> None:
        if not schema_version.strip():
            raise ValueError("schema_version must not be empty")
        by_name: dict[str, ActionSpec] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"duplicate action registration: {spec.name!r}")
            by_name[spec.name] = spec
        if not by_name:
            raise ValueError("registry must contain at least one action")
        self._specs = dict(sorted(by_name.items()))
        self.schema_version = schema_version

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def spec(self, name: str) -> ActionSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            allowed = ", ".join(self.names)
            raise ActionProtocolError(
                f"unknown action {name!r}; allowed: {allowed}"
            ) from exc

    def parse_json(self, text: str) -> ActionEnvelope:
        return self.parse_payload(decode_json_object(text))

    def parse_payload(self, payload: Mapping[str, Any]) -> ActionEnvelope:
        if not isinstance(payload, Mapping):
            raise ActionProtocolError("action payload must be one JSON object")
        unknown = set(payload) - {"action", "args", "reasoning"}
        if unknown:
            raise ActionProtocolError(f"unknown action fields: {sorted(unknown)}")
        if "action" not in payload:
            raise ActionProtocolError("missing required field 'action'")
        action_name = payload["action"]
        if not isinstance(action_name, str) or not action_name:
            raise ActionProtocolError("'action' must be a non-empty string")

        spec = self.spec(action_name)
        if "args" not in payload:
            if spec.args_required:
                raise ActionProtocolError(
                    f"action {action_name!r} requires an 'args' object"
                )
            raw_args: Mapping[str, Any] = {}
        else:
            raw_args = payload["args"]
            if not isinstance(raw_args, Mapping):
                raise ActionProtocolError("'args' must be a JSON object")

        args = spec.parse_args(raw_args)
        reasoning = parse_reasoning(payload.get("reasoning"))
        return ActionEnvelope(
            action=action_name,
            args=args,
            reasoning=reasoning,
        )

    def json_schema(self) -> dict[str, Any]:
        variants = []
        for spec in self._specs.values():
            properties: dict[str, Any] = {
                "action": {"const": spec.name},
                "args": dict(spec.args_schema),
                "reasoning": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 2_000,
                },
            }
            required = ["action"]
            if spec.args_required:
                required.append("args")
            variants.append(
                {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                }
            )
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "BehaviorHLAction",
            "oneOf": variants,
        }

    def manifest(self) -> dict[str, Any]:
        return _json_thaw({
            "schema_version": self.schema_version,
            "actions": [
                self._specs[name].manifest()
                for name in self.names
            ],
        })

    def sha256(self) -> str:
        canonical = json.dumps(
            self.manifest(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _json_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_thaw(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_thaw(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        "action registry manifest contains a non-JSON value: "
        f"{type(value).__name__}"
    )
