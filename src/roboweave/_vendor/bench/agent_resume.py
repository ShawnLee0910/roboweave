"""Validated restoration of committed Action20 conversation transcripts."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from action20_protocol import BASE_ODOMETRY_ACTIONS, navigation_repeat_max
from agent_artifacts import write_conversation_views
from agent_common import json_clone
from agent_observation import PublicImage, PublicObservation
from agent_protocol import PublicToolResult, replay_native_reasoning


PRIVATE_TRANSCRIPT_VERSION_V4 = "behavior-hl-action20-private-transcript-v4"
PRIVATE_TRANSCRIPT_VERSION_V5 = (
    "behavior-hl-action20-private-transcript-v5-repeat"
)
PRIVATE_TRANSCRIPT_VERSION = (
    "behavior-hl-action20-private-transcript-v6-path-only"
)
MAX_RESUME_ARTIFACT_BYTES = 32 * 1024 * 1024
RESUME_CHECKPOINT_VERSION_V3 = (
    "behavior-hl-model-resume-checkpoint/v3-macro-action-budget"
)
RESUME_CHECKPOINT_VERSION = "behavior-hl-model-resume-checkpoint/v4-repeat"
MODEL_ROLLING_CHECKPOINT_VERSION_V1 = (
    "behavior-hl-model-rolling-checkpoint/v1"
)
MODEL_ROLLING_CHECKPOINT_VERSION = (
    "behavior-hl-model-rolling-checkpoint/v2-persistent-scene"
)
ROLLING_CHECKPOINT_MAXIMUM_RETAINED = 2
PAUSED_CLIENT_STATUSES = frozenset(
    {"PAUSED_PROVIDER", "PAUSED_MODEL_PROTOCOL", "PAUSED_INTERRUPTED"}
)
RESUME_COUNTER_FIELDS = (
    "calls",
    "actions",
    "macro_actions_resolved",
    "invalid_model_responses",
    "protocol_retries",
    "decisions_recovered_after_retry",
    "provider_failures",
    "provider_retries",
    "decisions_recovered_after_provider_retry",
    "action_tag_compatibility_applied",
    "marker_selection_retries",
    "scene_step_attempts",
    "scene_recovery_count",
    "decision_only_recoveries",
    "checkpoint_restores",
    "same_scene_restarts",
    "rolled_back_completed_actions",
)

ConversationT = TypeVar("ConversationT")


def resume_client_manifest_identity(
    record: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return semantic resume identity without diagnostic or retry controls."""

    identity = dict(record)
    identity.pop("source_sha256", None)
    identity.pop("maximum_provider_attempts_per_protocol_attempt", None)
    identity.pop("provider_retry_policy", None)
    identity.pop("maximum_automatic_provider_retry_windows", None)
    identity.pop("automatic_provider_retry_window_delay_seconds", None)
    generation = identity.get("generation")
    if isinstance(generation, Mapping):
        normalized_generation = dict(generation)
        normalized_generation.setdefault("reasoning_summary", None)
        identity["generation"] = normalized_generation
    return identity


def validate_repeat_inflight(
    value: Mapping[str, Any] | None,
    *,
    next_sequence: int,
    error_type: type[Exception],
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "action",
        "repeat_requested",
        "steps",
    }:
        raise error_type("repeat checkpoint contract changed")
    action = value.get("action")
    requested = value.get("repeat_requested")
    steps = value.get("steps")
    if (
        action not in BASE_ODOMETRY_ACTIONS
        or isinstance(requested, bool)
        or not isinstance(requested, int)
        or not 2 <= requested <= navigation_repeat_max(action)
        or not isinstance(steps, list)
        or not 1 <= len(steps) <= requested
    ):
        raise error_type("repeat checkpoint identity changed")
    first_sequence = next_sequence - len(steps)
    if first_sequence < 0:
        raise error_type("repeat checkpoint sequence changed")
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping) or set(step) != {
            "step_index",
            "sequence",
            "result",
            "observation",
            "episode_terminal",
        }:
            raise error_type("repeat checkpoint step changed")
        result = step.get("result")
        observation = step.get("observation")
        result_keys = set(result) if isinstance(result, Mapping) else None
        if (
            step.get("step_index") != index
            or step.get("sequence") != first_sequence + index - 1
            or not isinstance(result, Mapping)
            or result_keys
            not in (
                {"status", "body_odometry"},
                {"status", "body_odometry", "blocked_category"},
            )
            or result.get("status") not in {"completed", "blocked"}
            or (index < len(steps) and result.get("status") != "completed")
            or not isinstance(observation, Mapping)
            or set(observation) != {"observation_id", "images"}
            or not isinstance(observation.get("images"), list)
            or not isinstance(step.get("episode_terminal"), bool)
            or (index < len(steps) and step.get("episode_terminal") is True)
        ):
            raise error_type("repeat checkpoint step identity changed")
        try:
            PublicToolResult(
                status=result["status"],
                body_odometry=result["body_odometry"],
                blocked_category=result.get("blocked_category"),
            )
        except ValueError as exc:
            raise error_type("repeat checkpoint result changed") from exc
    return value


def write_resume_checkpoint(
    *,
    output_dir: Path,
    scene: Any,
    experiment_id: str,
    scene_episode_id: str,
    prompt_sha256: str,
    conversation: Any,
    counters: Mapping[str, int],
    resume_session_index: int,
    state: str,
    write_json: Callable[[Path, Any], None],
    hash_file: Callable[[Path], str],
    error_type: type[Exception],
    write_views: Callable[..., None] = write_conversation_views,
    pause_status: str | None = None,
    terminal_protocol_failure: str | None = None,
    terminal_provider_failure: str | None = None,
    repeat_inflight: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if state not in {"ACTIVE", "PAUSED", "TERMINAL"}:
        raise ValueError("resume checkpoint state is invalid")
    if state == "PAUSED" and pause_status not in PAUSED_CLIENT_STATUSES:
        raise ValueError("resume pause status is invalid")
    if state == "ACTIVE" and pause_status is not None:
        raise ValueError("active checkpoint cannot have a pause status")
    if state == "TERMINAL" and pause_status is not None:
        raise ValueError("terminal checkpoint cannot have a pause status")
    normalized: dict[str, int] = {}
    if set(counters) != set(RESUME_COUNTER_FIELDS):
        raise ValueError("resume counters changed")
    for name in RESUME_COUNTER_FIELDS:
        value = counters[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("resume counter is invalid")
        normalized[name] = value
    client_manifest = Path(output_dir) / "client_manifest.json"
    if not client_manifest.is_file():
        raise error_type("resume client manifest is unavailable")
    record = {
        "schema_version": RESUME_CHECKPOINT_VERSION,
        "checkpointed_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "pause_status": pause_status,
        "experiment_id": experiment_id,
        "episode_id": scene_episode_id,
        "scene_run_dir": str(Path(scene.scene_run_dir).resolve()),
        "next_sequence": normalized["actions"],
        "resume_session_index": resume_session_index,
        "prompt_sha256": prompt_sha256,
        "client_manifest_sha256": hash_file(client_manifest),
        "terminal_protocol_failure": terminal_protocol_failure,
        "terminal_provider_failure": terminal_provider_failure,
        "repeat_inflight": validate_repeat_inflight(
            repeat_inflight,
            next_sequence=normalized["actions"],
            error_type=error_type,
        ),
        "counters": normalized,
        "conversation": conversation.artifact_transcript(
            artifact_root=output_dir
        ),
    }
    write_json(Path(output_dir) / "resume_checkpoint.json", record)
    write_views(output_dir=output_dir, conversation=conversation)
    return record


def load_resume_checkpoint(
    path: Path,
    *,
    read_json: Callable[..., Mapping[str, Any]],
    error_type: type[Exception],
    allow_active_for_persistent_migration: bool = False,
) -> Mapping[str, Any]:
    value = read_json(path, max_bytes=MAX_RESUME_ARTIFACT_BYTES)
    expected = {
        "schema_version",
        "checkpointed_at",
        "state",
        "pause_status",
        "experiment_id",
        "episode_id",
        "scene_run_dir",
        "next_sequence",
        "resume_session_index",
        "prompt_sha256",
        "client_manifest_sha256",
        "terminal_protocol_failure",
        "terminal_provider_failure",
        "counters",
        "conversation",
    }
    version = value.get("schema_version")
    if version == RESUME_CHECKPOINT_VERSION:
        expected.add("repeat_inflight")
    state = value.get("state")
    state_valid = bool(
        (state == "PAUSED" and value.get("pause_status") in PAUSED_CLIENT_STATUSES)
        or (
            allow_active_for_persistent_migration
            and state == "ACTIVE"
            and value.get("pause_status") is None
            and value.get("terminal_protocol_failure") is None
            and value.get("terminal_provider_failure") is None
        )
    )
    if (
        set(value) != expected
        or version not in {RESUME_CHECKPOINT_VERSION_V3, RESUME_CHECKPOINT_VERSION}
        or not state_valid
        or not isinstance(value.get("counters"), Mapping)
        or set(value["counters"]) != set(RESUME_COUNTER_FIELDS)
        or not isinstance(value.get("conversation"), Mapping)
    ):
        raise error_type("resume checkpoint contract changed")
    for name in RESUME_COUNTER_FIELDS:
        counter = value["counters"][name]
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise error_type("resume checkpoint counter is invalid")
    if (
        value.get("next_sequence") != value["counters"]["actions"]
        or isinstance(value.get("resume_session_index"), bool)
        or not isinstance(value.get("resume_session_index"), int)
        or value["resume_session_index"] < 0
    ):
        raise error_type("resume checkpoint sequence changed")
    validate_repeat_inflight(
        value.get("repeat_inflight"),
        next_sequence=value["next_sequence"],
        error_type=error_type,
    )
    return value


def write_model_rolling_checkpoint(
    *,
    output_dir: Path,
    resume_checkpoint: Mapping[str, Any],
    scene_checkpoint: Mapping[str, Any],
    write_json: Callable[[Path, Any], None],
    error_type: type[Exception],
) -> Path:
    """Atomically retain the conversation paired with a scene checkpoint."""

    expected_scene_fields = {
        "milestone_action_count",
        "captured_action_count",
        "retained_action_counts",
    }
    if frozenset(scene_checkpoint) not in {
        frozenset(expected_scene_fields),
        frozenset(expected_scene_fields | {"persistent_checkpoint"}),
    }:
        raise error_type("scene rolling checkpoint contract changed")
    normalized_scene_checkpoint = {
        **dict(scene_checkpoint),
        "persistent_checkpoint": scene_checkpoint.get("persistent_checkpoint"),
    }
    captured = scene_checkpoint["captured_action_count"]
    retained = scene_checkpoint["retained_action_counts"]
    if (
        isinstance(captured, bool)
        or not isinstance(captured, int)
        or captured <= 0
        or resume_checkpoint.get("state") != "ACTIVE"
        or resume_checkpoint.get("next_sequence") != captured
        or not isinstance(retained, list)
        or len(retained) > ROLLING_CHECKPOINT_MAXIMUM_RETAINED
        or captured not in retained
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in retained
        )
    ):
        raise error_type("scene and model rolling checkpoints diverged")
    root = Path(output_dir) / "rolling_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"checkpoint_action_{captured:04d}.json"
    write_json(
        destination,
        {
            "schema_version": MODEL_ROLLING_CHECKPOINT_VERSION,
            "scene_checkpoint": normalized_scene_checkpoint,
            "resume_checkpoint": dict(resume_checkpoint),
        },
    )
    for path in sorted(root.glob("checkpoint_action_*.json")):
        match = re.fullmatch(r"checkpoint_action_([0-9]{4,})\.json", path.name)
        if match is None:
            raise error_type("rolling checkpoint filename changed")
        action_count = int(match.group(1))
        if action_count not in retained:
            path.unlink()
    write_json(
        root / "index.json",
        {
            "schema_version": MODEL_ROLLING_CHECKPOINT_VERSION,
            "maximum_retained": ROLLING_CHECKPOINT_MAXIMUM_RETAINED,
            "retained_action_counts": retained,
            "latest_action_count": captured,
        },
    )
    return destination


def load_latest_model_rolling_checkpoint(
    *,
    output_dir: Path,
    resident_action_counts: list[int],
    read_json: Callable[..., Mapping[str, Any]],
    error_type: type[Exception],
) -> Mapping[str, Any]:
    """Load the newest model transcript paired with retained scene state."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in resident_action_counts
    ):
        raise error_type("resident rolling checkpoint index changed")
    root = Path(output_dir) / "rolling_checkpoints"
    index = read_json(root / "index.json", max_bytes=64 * 1024)
    if (
        set(index)
        != {
            "schema_version",
            "maximum_retained",
            "retained_action_counts",
            "latest_action_count",
        }
        or index.get("schema_version")
        not in {
            MODEL_ROLLING_CHECKPOINT_VERSION_V1,
            MODEL_ROLLING_CHECKPOINT_VERSION,
        }
        or index.get("maximum_retained") != ROLLING_CHECKPOINT_MAXIMUM_RETAINED
        or not isinstance(index.get("retained_action_counts"), list)
    ):
        raise error_type("model rolling checkpoint index changed")
    common = sorted(
        set(resident_action_counts).intersection(index["retained_action_counts"])
    )
    if not common:
        raise error_type("scene and model have no common rolling checkpoint")
    captured = common[-1]
    record = read_json(
        root / f"checkpoint_action_{captured:04d}.json",
        max_bytes=MAX_RESUME_ARTIFACT_BYTES,
    )
    if (
        set(record) != {"schema_version", "scene_checkpoint", "resume_checkpoint"}
        or record.get("schema_version")
        not in {
            MODEL_ROLLING_CHECKPOINT_VERSION_V1,
            MODEL_ROLLING_CHECKPOINT_VERSION,
        }
        or not isinstance(record.get("scene_checkpoint"), Mapping)
        or not isinstance(record.get("resume_checkpoint"), Mapping)
        or record["scene_checkpoint"].get("captured_action_count") != captured
        or record["resume_checkpoint"].get("state") != "ACTIVE"
        or record["resume_checkpoint"].get("next_sequence") != captured
    ):
        raise error_type("model rolling checkpoint payload changed")
    return record


def load_latest_persistent_model_checkpoint(
    *,
    output_dir: Path,
    maximum_action_count: int,
    read_json: Callable[..., Mapping[str, Any]],
    error_type: type[Exception],
) -> Mapping[str, Any]:
    """Select the newest paired model/physical checkpoint for migration."""

    if (
        isinstance(maximum_action_count, bool)
        or not isinstance(maximum_action_count, int)
        or maximum_action_count <= 0
    ):
        raise error_type("persistent resume action bound is invalid")
    root = Path(output_dir) / "rolling_checkpoints"
    candidates: list[tuple[int, Mapping[str, Any]]] = []
    for path in sorted(root.glob("checkpoint_action_*.json")):
        match = re.fullmatch(r"checkpoint_action_([0-9]{4,})\.json", path.name)
        if match is None:
            raise error_type("rolling checkpoint filename changed")
        captured = int(match.group(1))
        if captured > maximum_action_count:
            continue
        record = read_json(path, max_bytes=MAX_RESUME_ARTIFACT_BYTES)
        if (
            not isinstance(record, Mapping)
            or record.get("schema_version") != MODEL_ROLLING_CHECKPOINT_VERSION
            or not isinstance(record.get("scene_checkpoint"), Mapping)
            or not isinstance(record.get("resume_checkpoint"), Mapping)
            or record["scene_checkpoint"].get("captured_action_count") != captured
            or record["resume_checkpoint"].get("state") != "ACTIVE"
            or record["resume_checkpoint"].get("next_sequence") != captured
        ):
            continue
        persistent = record["scene_checkpoint"].get("persistent_checkpoint")
        if (
            not isinstance(persistent, Mapping)
            or set(persistent)
            != {
                "schema_version",
                "manifest_path",
                "manifest_sha256",
                "captured_action_count",
                "simulator_state_bytes",
            }
            or persistent.get("captured_action_count") != captured
            or not isinstance(persistent.get("manifest_path"), str)
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(persistent.get("manifest_sha256", ""))
            )
        ):
            continue
        candidates.append((captured, record))
    if not candidates:
        raise error_type(
            "model run has no portable physical checkpoint at or before pause"
        )
    return candidates[-1][1]


def restore_episode_conversation(
    conversation_type: type[ConversationT],
    transcript: Mapping[str, Any],
    *,
    expected_contract: Any,
    system_prompt_revision: Any,
    artifact_root: Path,
    tool_schema: Mapping[str, Any] | None = None,
) -> ConversationT:
    """Validate and replay one committed transcript without simulator calls."""

    common_keys = {
        "transcript_version",
        "private_episode_contract",
        "system_prompt_revision",
        "conversation_state",
        "visual_history",
        "active_observation_ids",
        "pending_marker",
        "inflight_marker_stage",
        "agent_authored_state",
        "events",
    }
    if not isinstance(transcript, Mapping):
        raise ValueError("resume transcript envelope changed")
    source_version = transcript.get("transcript_version")
    legacy_version = source_version in {
        PRIVATE_TRANSCRIPT_VERSION_V4,
        PRIVATE_TRANSCRIPT_VERSION_V5,
    }
    expected_keys = common_keys | (
        {"interface_identity"} if legacy_version else set()
    )
    prompt_record = transcript.get("system_prompt_revision")
    if (
        source_version
        not in {
            PRIVATE_TRANSCRIPT_VERSION_V4,
            PRIVATE_TRANSCRIPT_VERSION_V5,
            PRIVATE_TRANSCRIPT_VERSION,
        }
        or set(transcript) != expected_keys
        or transcript.get("private_episode_contract")
        != expected_contract.artifact_record()
        or not isinstance(prompt_record, Mapping)
        or prompt_record.get("source_path")
        != str(system_prompt_revision.source_path)
        or (
            not legacy_version
            and prompt_record != {"source_path": str(system_prompt_revision.source_path)}
            and prompt_record != system_prompt_revision.artifact_record()
        )
        or transcript.get("conversation_state")
        not in {"awaiting_assistant", "awaiting_tool_result"}
    ):
        raise ValueError("resume transcript identity changed")

    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ValueError("resume artifact root is unavailable")
    events = transcript.get("events")
    if not isinstance(events, list) or len(events) < 2:
        raise ValueError("resume transcript events are incomplete")

    def observation_from_event(event: Mapping[str, Any]) -> PublicObservation:
        record = event.get("observation")
        if (
            not isinstance(record, Mapping)
            or set(record)
            not in (
                {"observation_id", "images"},
                {"observation_id", "images", "marker_depth"},
            )
            or not isinstance(record.get("images"), list)
        ):
            raise ValueError("resume observation record changed")
        images: list[PublicImage] = []
        for item in record["images"]:
            required_image_keys = {"camera", "path"}
            item_keys = set(item) if isinstance(item, Mapping) else set()
            if (
                not isinstance(item, Mapping)
                or not required_image_keys.issubset(item_keys)
                or (
                    legacy_version
                    and len(item_keys - required_image_keys) != 1
                )
                or (
                    not legacy_version
                    and item_keys != required_image_keys
                )
            ):
                raise ValueError("resume image record changed")
            path = Path(item["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("resume image escaped its artifact root")
            image = PublicImage(camera=item["camera"], path=path)
            image.read_png()
            images.append(image)
        return PublicObservation(
            observation_id=record["observation_id"],
            images=tuple(images),
            marker_depth=(
                None
                if "marker_depth" not in record
                else tuple(record["marker_depth"])
            ),
        )

    initial_event = events[1]
    if (
        not isinstance(initial_event, Mapping)
        or initial_event.get("kind") != "observation"
        or initial_event.get("observation_kind") != "physical"
        or initial_event.get("marker_targets") != []
    ):
        raise ValueError("resume initial observation changed")
    restored = conversation_type(
        expected_contract,
        observation_from_event(initial_event),
        system_prompt_revision=system_prompt_revision,
        tool_schema=tool_schema,
    )
    for event in events[2:]:
        if not isinstance(event, Mapping):
            raise ValueError("resume event is not an object")
        kind = event.get("kind")
        if kind == "assistant_tool_call":
            provider_response = event.get("raw_response")
            message = json_clone(event.get("message"))
            if not isinstance(provider_response, Mapping) or not isinstance(
                message, dict
            ):
                raise ValueError("resume assistant response is missing")
            replay_native_reasoning(
                message,
                event.get("native_reasoning_content"),
                event.get("native_reasoning_source"),
            )
            restored.append_assistant_completion(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": message,
                        }
                    ]
                },
                provider_response=provider_response,
            )
        elif kind == "tool_result":
            message = event.get("message")
            if not isinstance(message, Mapping) or not isinstance(
                message.get("content"), str
            ):
                raise ValueError("resume tool result changed")
            try:
                result = json.loads(message["content"])
            except json.JSONDecodeError as exc:
                raise ValueError("resume tool result is not JSON") from exc
            if not isinstance(result, Mapping) or not set(result).issubset(
                {
                    "status",
                    "body_odometry",
                    "stage",
                    "repeat_requested",
                    "repeat_executed",
                    "steps",
                    "blocked_category",
                    "blocked_detail",
                    "action_decisions_remaining",
                }
            ):
                raise ValueError("resume tool result fields changed")
            restored.append_tool_result(
                PublicToolResult(
                    status=result.get("status"),
                    body_odometry=result.get("body_odometry"),
                    stage=result.get("stage"),
                    repeat_requested=result.get("repeat_requested"),
                    repeat_executed=result.get("repeat_executed"),
                    blocked_category=result.get("blocked_category"),
                    blocked_detail=result.get("blocked_detail"),
                    action_decisions_remaining=result.get(
                        "action_decisions_remaining"
                    ),
                    steps=(
                        tuple(result["steps"])
                        if "steps" in result
                        else None
                    ),
                )
            )
        elif kind == "observation":
            observation_kind = event.get("observation_kind")
            marker_targets = event.get("marker_targets")
            if (
                observation_kind not in {"physical", "marker", "transient"}
                or not isinstance(marker_targets, list)
            ):
                raise ValueError("resume observation event changed")
            restored.append_observation(
                observation_from_event(event),
                kind=observation_kind,
                marker_targets=marker_targets,
            )
        else:
            raise ValueError("resume transcript event kind changed")

    if _resume_transcript_projection(
        restored.artifact_transcript()
    ) != _resume_transcript_projection(transcript):
        raise ValueError("resume transcript did not replay exactly")
    return restored


def _resume_transcript_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove legacy integrity fields before replay comparison."""

    def keep_public_image_fields(observation: Any) -> None:
        if not isinstance(observation, dict):
            return
        images = observation.get("images")
        if not isinstance(images, list):
            return
        observation["images"] = [
            {
                "camera": image.get("camera"),
                "path": image.get("path"),
            }
            if isinstance(image, dict)
            else image
            for image in images
        ]

    projected = json_clone(value)
    projected["transcript_version"] = PRIVATE_TRANSCRIPT_VERSION
    projected.pop("interface_identity", None)
    prompt = projected.get("system_prompt_revision")
    if isinstance(prompt, dict):
        projected["system_prompt_revision"] = {
            "source_path": prompt.get("source_path")
        }
    for event in projected.get("events", ()):
        if not isinstance(event, dict):
            continue
        keep_public_image_fields(event.get("observation"))
        evidence = event.get("selected_target_visual_evidence_private")
        marker_frame = (
            evidence.get("marker_frame")
            if isinstance(evidence, dict)
            else None
        )
        if isinstance(marker_frame, dict):
            marker_observation = marker_frame.get("observation")
            keep_public_image_fields(marker_observation)
            evidence["marker_frame"] = {"observation": marker_observation}
    return projected
