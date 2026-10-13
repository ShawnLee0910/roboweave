"""Readable ChatML export derived from private Action20 transcripts."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping

from agent_common import BLOCKED_OBSERVATION_REMINDER, json_clone
from agent_observation import (
    PublicObservation,
    THREE_VIEW_COMPOSITE_FILENAME,
    model_visible_marker_depth_text,
)
from agent_protocol import parse_native_tool_message, validate_native_reasoning_text


CHATML_TRANSCRIPT_VERSION = "behavior-hl-action20-chatml-v3"


def link_or_copy(source: Path, destination: Path) -> None:
    """Hardlink immutable image artifacts, copying only when unsupported."""

    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno not in {
            errno.EXDEV,
            errno.ENOSYS,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
            errno.EPERM,
        }:
            raise
        shutil.copy2(source, destination)


def publish_action_after_view(
    observation: PublicObservation,
    *,
    output_dir: Path,
    step_index: int,
    action_name: str,
    hash_file: Callable[[Path], str],
    error_type: type[Exception],
    publish_file: Callable[[Path, Path], None] = link_or_copy,
    recovery_branch_index: int = 0,
) -> Path:
    """Publish one chronological, human-readable post-action composite."""

    if (
        isinstance(step_index, bool)
        or not isinstance(step_index, int)
        or step_index < 0
        or not isinstance(action_name, str)
        or not re.fullmatch(r"[a-z][a-z0-9_]*", action_name)
        or isinstance(recovery_branch_index, bool)
        or not isinstance(recovery_branch_index, int)
        or recovery_branch_index < 0
    ):
        raise error_type("action-after artifact identity is invalid")
    source = (
        Path(output_dir)
        / "observations"
        / observation.observation_id
        / THREE_VIEW_COMPOSITE_FILENAME
    )
    if not source.is_file():
        raise error_type("model-visible composite is missing")
    action_after_dir = Path(output_dir) / "action_after"
    if recovery_branch_index:
        action_after_dir = (
            Path(output_dir)
            / "action_after_branches"
            / f"recovery_{recovery_branch_index:04d}"
        )
    action_after_dir.mkdir(parents=True, exist_ok=True)
    destination = action_after_dir / f"{step_index:03d}_{action_name}.png"
    if destination.exists():
        raise error_type("action-after artifact already exists")
    publish_file(source, destination)
    if hash_file(destination) != hash_file(source):
        raise error_type("action-after artifact copy changed")
    return destination


def write_conversation_views(
    *,
    output_dir: Path,
    conversation: Any,
) -> None:
    """Write the private transcript and its readable ChatML projection."""

    root = Path(output_dir)
    _write_artifact_json(
        root / "conversation_private.json",
        conversation.artifact_transcript(artifact_root=root),
    )
    _write_artifact_json(
        root / "conversation_chatml.json",
        conversation.artifact_chatml(artifact_root=root),
        pretty=True,
    )


def _write_artifact_json(path: Path, value: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    options = (
        {"ensure_ascii": False, "indent": 2, "allow_nan": False}
        if pretty
        else {
            "ensure_ascii": False,
            "sort_keys": True,
            "separators": (",", ":"),
            "allow_nan": False,
        }
    )
    temporary.write_bytes(json.dumps(value, **options).encode("utf-8") + b"\n")
    temporary.replace(path)


def build_chatml_artifact(
    transcript: Mapping[str, Any],
    *,
    system_prompt: str,
    artifact_root: Path,
) -> dict[str, Any]:
    """Convert one complete private transcript into a readable ChatML view.

    The output preserves the chronological native Marker micro-trace using
    data that was model-visible at each original step, plus the model's own
    native reasoning.  Backend identity, raw provider responses, and private
    episode metadata stay exclusively in the source transcript.
    """

    if not isinstance(transcript, Mapping):
        raise ValueError("private transcript must be a mapping")
    source_version = transcript.get("transcript_version")
    state = transcript.get("conversation_state")
    events = transcript.get("events")
    if (
        not isinstance(source_version, str)
        or not source_version
        or state not in {
            "awaiting_assistant",
            "awaiting_tool_result",
            "awaiting_observation",
            "terminal",
        }
        or not isinstance(events, list)
        or not events
    ):
        raise ValueError("private transcript envelope is invalid")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("ChatML system prompt must be nonempty")
    root = Path(artifact_root).resolve()

    normalized_events: list[Mapping[str, Any]] = []
    for index, event in enumerate(events):
        if (
            not isinstance(event, Mapping)
            or event.get("event_index") != index
            or not isinstance(event.get("kind"), str)
        ):
            raise ValueError("private transcript event sequence is invalid")
        normalized_events.append(event)

    first = normalized_events[0]
    if (
        first.get("kind") != "task"
        or first.get("role") != "user"
        or not isinstance(first.get("instruction"), str)
        or not first["instruction"].strip()
    ):
        raise ValueError("private transcript task event is invalid")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    index = 1
    if index < len(normalized_events) and normalized_events[index]["kind"] == "observation":
        initial = normalized_events[index]
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Task: {first['instruction']}\n\n"
                            + _chatml_observation_label(initial)
                        ),
                    },
                    _chatml_image_item(initial, artifact_root=root),
                ],
            }
        )
        index += 1
    else:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Task: {first['instruction']}"}
                ],
            }
        )

    while index < len(normalized_events):
        event = normalized_events[index]
        kind = event["kind"]
        if kind == "assistant_tool_call":
            messages.append(_chatml_assistant_message(event))
        elif kind == "tool_result":
            content = _chatml_tool_result_content(event)
            if (
                index + 1 < len(normalized_events)
                and normalized_events[index + 1]["kind"] == "observation"
            ):
                observation = normalized_events[index + 1]
                content[0]["text"] += (
                    "\n\n" + _chatml_observation_label(observation) + "\n"
                )
                content.extend(
                    [
                        _chatml_image_item(observation, artifact_root=root),
                        {"type": "text", "text": "\n</tool_response>"},
                    ]
                )
                index += 1
            else:
                content[0]["text"] += "\n</tool_response>"
            messages.append({"role": "user", "content": content})
        elif kind == "observation":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _chatml_observation_label(event),
                        },
                        _chatml_image_item(event, artifact_root=root),
                    ],
                }
            )
        elif kind != "episode_end":
            raise ValueError(f"unsupported private transcript event kind: {kind}")
        index += 1

    visual_history = transcript.get("visual_history")
    return {
        "type": "chatml",
        "schema_version": CHATML_TRANSCRIPT_VERSION,
        "source": "conversation_private.json",
        "source_transcript_version": source_version,
        "conversation_state": state,
        "visual_history": json_clone(visual_history),
        "messages": messages,
    }


def _chatml_assistant_message(event: Mapping[str, Any]) -> dict[str, Any]:
    message = event.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("assistant transcript event is invalid")
    decision = parse_native_tool_message(message)
    content: list[dict[str, str]] = []
    reasoning = event.get("native_reasoning_content")
    if reasoning is not None:
        reasoning = validate_native_reasoning_text(reasoning)
        if reasoning:
            content.append(
                {
                    "type": "text",
                    "text": f"<think>\n{reasoning}\n</think>\n\n",
                }
            )
    lines = [
        "<tool_call>",
        "<function=robot_action>",
        "<parameter=action>",
        decision.action,
        "</parameter>",
    ]
    if decision.repeat is not None:
        lines.extend(
            [
                "<parameter=repeat>",
                str(decision.repeat),
                "</parameter>",
            ]
        )
    if decision.target is not None:
        lines.extend(
            [
                "<parameter=target>",
                str(decision.target),
                "</parameter>",
            ]
        )
        if decision.target_description is not None:
            lines.extend(
                [
                    "<parameter=target_description>",
                    decision.target_description,
                    "</parameter>",
                ]
            )
    lines.extend(["</function>", "</tool_call>"])
    content.append({"type": "text", "text": "\n".join(lines)})
    return {"role": "assistant", "content": content}


def _chatml_tool_result_content(
    event: Mapping[str, Any],
) -> list[dict[str, str]]:
    message = event.get("message")
    if (
        event.get("role") != "tool"
        or not isinstance(message, Mapping)
        or message.get("role") != "tool"
        or not isinstance(message.get("content"), str)
    ):
        raise ValueError("tool-result transcript event is invalid")
    try:
        public_result = json.loads(message["content"])
    except json.JSONDecodeError as exc:
        raise ValueError("tool-result transcript content is invalid") from exc
    canonical_result = json.dumps(
        public_result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if public_result.get("status") == "blocked":
        canonical_result += "\n\n" + BLOCKED_OBSERVATION_REMINDER
    return [
        {
            "type": "text",
            "text": f"<tool_response>\n{canonical_result}",
        }
    ]


def _chatml_observation_label(event: Mapping[str, Any]) -> str:
    kind = event.get("observation_kind")
    if kind == "marker":
        stage = event.get("marker_stage")
        targets = event.get("marker_targets")
        label = f"Marker observation (stage {stage}, candidates {targets})."
        observation = event.get("observation")
        if isinstance(observation, Mapping) and "marker_depth" in observation:
            label += "\n" + model_visible_marker_depth_text(
                observation["marker_depth"]
            )
        return label
    if kind == "physical":
        label = "Onboard physical observation."
        if event.get("repeat_step_index") is not None:
            label += (
                f" Repeat step {event['repeat_step_index']} of "
                f"{event.get('repeat_step_count')}."
            )
        return label
    if kind == "transient":
        return "Onboard transient observation."
    raise ValueError("observation transcript kind is invalid")


def _chatml_image_item(
    event: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    observation = event.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("observation transcript event is invalid")
    observation_id = observation.get("observation_id")
    images = observation.get("images")
    if (
        not isinstance(observation_id, str)
        or not observation_id
        or not isinstance(images, list)
        or not images
        or not isinstance(images[0], Mapping)
        or not isinstance(images[0].get("path"), str)
    ):
        raise ValueError("observation transcript record is invalid")
    image = Path(images[0]["path"])
    if not image.is_absolute():
        image = artifact_root / image
    elif (
        not image.resolve().is_relative_to(artifact_root)
        and image.parts[-3:]
        == ("observations", observation_id, image.name)
    ):
        relocated = (
            artifact_root / "observations" / observation_id / image.name
        )
        if relocated.is_file():
            image = relocated
    composite = image.resolve().parent / THREE_VIEW_COMPOSITE_FILENAME
    try:
        image_path = composite.relative_to(artifact_root).as_posix()
    except ValueError:
        image_path = str(composite)
    item: dict[str, Any] = {
        "type": "image",
        "image": image_path,
        "observation_id": observation_id,
        "observation_kind": event.get("observation_kind"),
    }
    if event.get("observation_kind") == "marker":
        item["marker_stage"] = event.get("marker_stage")
        item["marker_targets"] = json_clone(event.get("marker_targets"))
    return item
