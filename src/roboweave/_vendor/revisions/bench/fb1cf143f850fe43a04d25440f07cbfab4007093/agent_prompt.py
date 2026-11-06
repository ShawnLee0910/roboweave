"""Prompt and native tool-schema loading for the Action20 agent."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from action20_protocol import (
    MODEL_VISIBLE_TOOL_ACTIONS,
    NAVIGATION_MOVE_DISTANCES_M,
    NAVIGATION_REPEAT_MAX,
    NAVIGATION_TURN_ANGLES_DEG,
)
from agent_common import PHYSICAL_HISTORY_MAX, validate_physical_history_max


INTERFACE_ROOT = Path(__file__).resolve().parent
ACTION20_CORE_INSTRUCTIONS_PATH = INTERFACE_ROOT / "action20_core_instructions.txt"
ACTION20_TOOL_SCHEMA_PATH = INTERFACE_ROOT / "action20_tool_schema.json"
_SECRET_KEY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])s" r"k-[A-Za-z0-9._-]{8,}",
    re.IGNORECASE,
)
_ACTION_DECISION_BUDGET_PATTERN = re.compile(
    r"You have a budget of at most \d+ action decisions for the entire task\."
)
COMMON_HARNESS_ADAPTER_INSTRUCTIONS = (
    "Common Tool-Calling Harness:\n"
    "Each assistant response must contain exactly one `robot_action` tool call "
    "and no ordinary assistant content."
)
_THREE_VIEW_OBSERVATION_RULE = (
    "Each onboard observation is one three-view RGB composite: the head camera "
    "fills the large right panel, with the left-wrist view above the right-wrist "
    "view on the left."
)
_OVERHEAD_VIEW_OBSERVATION_RULES = {
    "always": (
        "Each onboard observation is one four-view RGB composite: the head "
        "camera fills the large right panel; the left column shows left-wrist, "
        "right-wrist, and overhead top-down views from top to bottom."
    ),
    "on_blocked": (
        "Observations normally use the three-view RGB composite: head on the "
        "large right, with left-wrist above right-wrist on the left. After a "
        "blocked action, an overhead top-down view may appear at the bottom of "
        "the left column."
    ),
}
_LOCAL_WORKSPACE_RULE = (
    "Other than OnTop's one-time support approach, manipulation does not automatically "
    "move the base. Selected targets must be within about 1.7 m horizontally; "
    "OnTop measures reach to the future placement footprint. "
    "A visible Marker does not guarantee reachability or a feasible placement."
)
_AUTOMATIC_INTERACTION_APPROACH_RULE = (
    "Automatic interaction approach is enabled: after you select a manipulation "
    "target, the action may first navigate to a safe nearby pose; if none is "
    "reachable, it returns `blocked` with out_of_reach. "
    "A visible Marker does not guarantee a feasible placement."
)


def load_action20_tool_schema(
    path: Path | None = None,
) -> dict[str, Any]:
    """Validate schema structure; semantic wording is covered by contract tests."""

    path = ACTION20_TOOL_SCHEMA_PATH if path is None else path
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Action20 tool schema is unreadable") from exc
    if not isinstance(value, dict) or set(value) != {"type", "function"}:
        raise ValueError("Action20 tool schema envelope changed")
    function = value.get("function")
    if (
        value.get("type") != "function"
        or not isinstance(function, dict)
        or set(function) != {"name", "description", "parameters"}
        or function.get("name") != "robot_action"
        or not isinstance(function.get("description"), str)
        or not function["description"].strip()
    ):
        raise ValueError("Action20 tool identity changed")
    parameters = function.get("parameters")
    properties = (
        parameters.get("properties", {})
        if isinstance(parameters, dict)
        else {}
    )
    if (
        not isinstance(parameters, dict)
        or parameters.get("type") != "object"
        or parameters.get("additionalProperties") is not False
        or parameters.get("required") != ["action"]
        or frozenset(properties)
        not in {
            frozenset({"action", "target", "target_description"}),
            frozenset({"action", "repeat", "target", "target_description"}),
            frozenset(
                {
                    "action",
                    "move_distance_m",
                    "repeat",
                    "target",
                    "target_description",
                    "turn_angle_deg",
                }
            ),
        }
    ):
        raise ValueError("Action20 tool parameter contract changed")
    action = properties["action"]
    repeat = properties.get("repeat")
    move_distance = properties.get("move_distance_m")
    turn_angle = properties.get("turn_angle_deg")
    target = properties["target"]
    target_description = properties["target_description"]
    if (
        not isinstance(action, dict)
        or action.get("type") != "string"
        or tuple(action.get("enum", ())) != MODEL_VISIBLE_TOOL_ACTIONS
        or not isinstance(action.get("description"), str)
        or not action["description"].strip()
        or (
            repeat is not None
            and (
                not isinstance(repeat, dict)
                or repeat.get("type") != "integer"
                or repeat.get("minimum") != 1
                or repeat.get("maximum") != NAVIGATION_REPEAT_MAX
                or not isinstance(repeat.get("description"), str)
                or not repeat["description"].strip()
            )
        )
        or (
            move_distance is not None
            and (
                not isinstance(move_distance, dict)
                or move_distance.get("type") != "number"
                or tuple(move_distance.get("enum", ()))
                != NAVIGATION_MOVE_DISTANCES_M
                or not isinstance(move_distance.get("description"), str)
                or not move_distance["description"].strip()
            )
        )
        or (
            turn_angle is not None
            and (
                not isinstance(turn_angle, dict)
                or turn_angle.get("type") != "number"
                or tuple(turn_angle.get("enum", ()))
                != NAVIGATION_TURN_ANGLES_DEG
                or not isinstance(turn_angle.get("description"), str)
                or not turn_angle["description"].strip()
            )
        )
        or not isinstance(target, dict)
        or target.get("type") != "integer"
        or target.get("minimum") != 1
        or target.get("maximum") != 999
        or not isinstance(target.get("description"), str)
        or not target["description"].strip()
        or not isinstance(target_description, dict)
        or target_description.get("type") != "string"
        or target_description.get("minLength") != 1
        or target_description.get("maxLength") != 64
        or not isinstance(target_description.get("description"), str)
        or not target_description["description"].strip()
    ):
        raise ValueError("Action20 action or target schema changed")
    return value


def load_action20_core_instructions(
    prompt_path: Path = ACTION20_CORE_INSTRUCTIONS_PATH,
) -> str:
    """Load the shared model-visible Action20 rules."""

    try:
        template = Path(prompt_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("Action20 core instructions are unreadable") from exc
    _validate_experimental_system_instruction(template)
    return template


def with_automatic_interaction_approach_tool_schema(
    tool_schema: dict[str, Any],
) -> dict[str, Any]:
    """Replace the direct-reach contract for an automatic-approach episode."""

    configured = deepcopy(tool_schema)
    description = configured["function"]["description"]
    if description.count(_LOCAL_WORKSPACE_RULE) != 1:
        raise ValueError("Action20 local workspace rule changed")
    configured["function"]["description"] = description.replace(
        _LOCAL_WORKSPACE_RULE,
        _AUTOMATIC_INTERACTION_APPROACH_RULE,
    )
    return configured


def with_overhead_view_tool_schema(
    tool_schema: dict[str, Any],
    overhead_view_mode: str,
) -> dict[str, Any]:
    """Describe the active model-visible composite without changing actions."""

    try:
        observation_rule = _OVERHEAD_VIEW_OBSERVATION_RULES[overhead_view_mode]
    except KeyError as exc:
        raise ValueError("overhead view mode must be always or on_blocked") from exc
    configured = deepcopy(tool_schema)
    description = configured["function"]["description"]
    if description.count(_THREE_VIEW_OBSERVATION_RULE) != 1:
        raise ValueError("Action20 observation layout rule changed")
    configured["function"]["description"] = description.replace(
        _THREE_VIEW_OBSERVATION_RULE,
        observation_rule,
    )
    return configured


def _validate_experimental_system_instruction(text: str) -> None:
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text.encode("utf-8")) > 131_072
        or text.lstrip().startswith("```")
        or text.rstrip().endswith("```")
        or "{{ACTION20_TOOL_SCHEMA}}" in text
        or "<tool_call>" in text
        or "robot_action" not in text
    ):
        raise ValueError("experimental system Prompt contract changed")
    lowered = text.lower()
    for forbidden in (
        "selected_option",
        "official_q",
        "api_key",
        "authorization: bearer",
    ):
        if forbidden in lowered:
            raise ValueError("experimental system Prompt contains private data")
    if _SECRET_KEY_PATTERN.search(text):
        raise ValueError("experimental system Prompt contains private data")


ACTION20_TOOL_SCHEMA = load_action20_tool_schema()
ACTION20_CORE_INSTRUCTIONS = load_action20_core_instructions()


@dataclass(frozen=True)
class SystemPromptRevision:
    """One explicitly loaded Prompt revision for a single fresh episode."""

    text: str
    source_path: Path
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_experimental_system_instruction(self.text)
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        if self.source_sha256 is not None and (
            not isinstance(self.source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None
        ):
            raise ValueError("Prompt source hash must be SHA-256")

    def artifact_record(self) -> dict[str, str]:
        return {
            "source_path": str(self.source_path),
            "rendered_prompt_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
        }


def prompt_provenance_record(
    revision: SystemPromptRevision,
    tool_schema: Mapping[str, Any],
    *,
    instruction: str,
    maximum_action_decisions: int,
    harness_adapter: str,
    overhead_view_mode: str,
    automatic_interaction_approach_enabled: bool,
    physical_history_max: int,
) -> dict[str, Any]:
    """Bind existing run artifacts to the actual rendered text and conditions."""

    schema_bytes = json.dumps(
        tool_schema, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return {
        "contract_version": "action20-prompt-provenance/v1",
        **revision.artifact_record(),
        "source_sha256": revision.source_sha256,
        "tool_schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "maximum_action_decisions": maximum_action_decisions,
        "harness_adapter": harness_adapter,
        "overhead_view_mode": overhead_view_mode,
        "automatic_interaction_approach_enabled": automatic_interaction_approach_enabled,
        "physical_history_max": physical_history_max,
    }


def load_prompt_revision(
    path: Path | None = None,
    *,
    physical_history_max: int = PHYSICAL_HISTORY_MAX,
    maximum_action_decisions: int = 150,
) -> SystemPromptRevision:
    """Read one stable on-disk revision without caching it across episodes."""

    validate_physical_history_max(physical_history_max)
    if (
        isinstance(maximum_action_decisions, bool)
        or not isinstance(maximum_action_decisions, int)
        or not 1 <= maximum_action_decisions <= 150
    ):
        raise ValueError("maximum action decisions must be in [1, 150]")
    source = Path(ACTION20_CORE_INSTRUCTIONS_PATH if path is None else path).resolve()
    try:
        before = source.stat()
        raw = source.read_bytes()
        after = source.stat()
    except OSError as exc:
        raise ValueError("system Prompt revision is unreadable") from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise ValueError("system Prompt changed while it was being read")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("system Prompt must be UTF-8") from exc
    text, replacements = _ACTION_DECISION_BUDGET_PATTERN.subn(
        "You have a budget of at most "
        f"{maximum_action_decisions} action decisions for the entire task.",
        text,
    )
    if replacements != 1:
        raise ValueError("system Prompt action-decision budget contract changed")
    _validate_experimental_system_instruction(text)
    return SystemPromptRevision(
        text=text,
        source_path=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def with_common_harness_adapter(
    revision: SystemPromptRevision,
) -> SystemPromptRevision:
    return replace(
        revision,
        text=(
            revision.text.rstrip()
            + "\n\n"
            + COMMON_HARNESS_ADAPTER_INSTRUCTIONS
            + "\n"
        ),
    )


def load_common_system_prompt_revision(
    path: Path | None = None,
    *,
    physical_history_max: int = PHYSICAL_HISTORY_MAX,
    maximum_action_decisions: int = 150,
) -> SystemPromptRevision:
    return with_common_harness_adapter(
        load_prompt_revision(
            path,
            physical_history_max=physical_history_max,
            maximum_action_decisions=maximum_action_decisions,
        )
    )
