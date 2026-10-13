"""Generic resident-episode sink adapter for the atomic artifact writer."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from behavior_hl.embodied.runtime.artifacts import (
    ArtifactInventory,
    ArtifactVisibility,
)
from behavior_hl.embodied.runtime.resident_artifacts import (
    ResidentArtifactError,
    ResidentEpisodeArtifactWriter,
)


FRAME_PROTOCOL_BLOCKER = (
    "ResidentEpisodeOrchestrator._record_frame currently supplies hashes only. "
    "It must receive exact encoded onboard PNG bytes from a new narrow "
    "ResidentEnvironment frame-artifact method and add png_bytes plus "
    "png_sha256 to the sink payload. A hash is not image evidence."
)

_STEP = re.compile(
    r"^step_(?P<index>\d{4})_(?P<kind>"
    r"action_before|action_after|settle_final|verifier)$"
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ResidentArtifactAdapterError(ResidentArtifactError):
    """Raised when the generic sink stream violates its ordering contract."""


class FrameContentUnavailableError(ResidentArtifactAdapterError):
    """Raised instead of turning a frame hash into fake image content."""


class ResidentArtifactSinkAdapter:
    """Map ``ResidentArtifactSink.record`` calls onto atomic typed methods."""

    def __init__(
        self,
        writer: ResidentEpisodeArtifactWriter,
        *,
        episode_video_relative_path: str = "video/episode.mp4",
    ) -> None:
        if not isinstance(writer, ResidentEpisodeArtifactWriter):
            raise TypeError("writer must be ResidentEpisodeArtifactWriter")
        self.writer = writer
        self.episode_video_relative_path = episode_video_relative_path
        self._pending: dict[int, dict[str, Any]] = {}
        self._committed: dict[int, tuple[Any, Any, Any]] = {}
        self._events: list[Any] = []
        self._final_state: Mapping[str, Any] | None = None

    def record(
        self,
        *,
        logical_name: str,
        visibility: ArtifactVisibility,
        payload: Any,
        append: bool = False,
    ) -> None:
        if not isinstance(visibility, ArtifactVisibility):
            raise TypeError("visibility must be ArtifactVisibility")
        if logical_name in {"model_raw_responses", "parsed_actions"}:
            self._record_model_stream(
                logical_name,
                visibility,
                payload,
                append,
            )
            return
        if logical_name == "events":
            self._record_event(visibility, payload, append)
            return
        if append:
            raise ResidentArtifactAdapterError(
                f"{logical_name} is not an append stream"
            )
        if logical_name == "initial_frame":
            self._require_visibility(
                visibility, ArtifactVisibility.MEDIA, logical_name
            )
            self.writer.record_initial_observation(
                self._exact_png(payload, logical_name)
            )
            return
        if logical_name == "initial_predicates":
            self._require_visibility(
                visibility, ArtifactVisibility.PRIVATE, logical_name
            )
            self.writer.record_initial_predicates(payload)
            return
        if logical_name == "final_predicate_report":
            self._require_visibility(
                visibility, ArtifactVisibility.PRIVATE, logical_name
            )
            self.writer.record_final_predicate_report(payload)
            return
        if logical_name == "final_state":
            self._require_visibility(
                visibility, ArtifactVisibility.SYSTEM, logical_name
            )
            if not isinstance(payload, Mapping):
                raise ResidentArtifactAdapterError(
                    "final_state must be a mapping"
                )
            # ResidentEpisode emits terminal accounting, not another sensor
            # observation. Preserve its declared SYSTEM visibility instead of
            # mislabelling verifier-derived success as public observation.
            self.writer._write_json(  # noqa: SLF001 - internal runtime bridge
                logical_name="final_state",
                relative_path="system/final_state.json",
                visibility=ArtifactVisibility.SYSTEM,
                payload=dict(payload),
                immutable=True,
            )
            self._final_state = dict(payload)
            return

        match = _STEP.fullmatch(logical_name)
        if match is None:
            raise ResidentArtifactAdapterError(
                f"unknown resident artifact: {logical_name}"
            )
        index = int(match.group("index"))
        kind = match.group("kind")
        if kind == "verifier":
            self._require_visibility(
                visibility, ArtifactVisibility.PRIVATE, logical_name
            )
            self.writer.record_private_verifier(index, payload)
            return
        self._require_visibility(
            visibility, ArtifactVisibility.MEDIA, logical_name
        )
        png = self._exact_png(payload, logical_name)
        {
            "action_before": self.writer.record_action_before,
            "action_after": self.writer.record_action_after,
            "settle_final": self.writer.record_settle_final,
        }[kind](index, png)

    def finalize(
        self,
        final_payload: Mapping[str, Any],
    ) -> ArtifactInventory:
        if not isinstance(final_payload, Mapping):
            raise ResidentArtifactAdapterError(
                "final_payload must be a mapping"
            )
        unresolved = {
            index: sorted(parts)
            for index, parts in self._pending.items()
            if set(parts) != {"model_input", "raw_response"}
        }
        if unresolved:
            raise ResidentArtifactAdapterError(
                "parsed-without-raw or malformed model records remain: "
                f"{unresolved}"
            )
        if self._final_state is None:
            raise ResidentArtifactAdapterError(
                "final_state was not recorded before finalize"
            )
        if dict(final_payload) != dict(self._final_state):
            raise ResidentArtifactAdapterError(
                "final_payload does not match recorded final_state"
            )
        video = self.writer.run_dir / self.episode_video_relative_path
        if not video.is_file():
            raise ResidentArtifactAdapterError(
                f"episode MP4 is missing: {self.episode_video_relative_path}"
            )
        self.writer.register_episode_video(
            self.episode_video_relative_path
        )
        status = final_payload.get("termination")
        if not isinstance(status, str) or not status:
            raise ResidentArtifactAdapterError(
                "final_payload.termination must be non-empty"
            )
        self.writer.finalize(status=status.upper())
        return self.writer.inventory

    def _record_model_stream(
        self,
        logical_name: str,
        visibility: ArtifactVisibility,
        payload: Any,
        append: bool,
    ) -> None:
        self._require_visibility(
            visibility, ArtifactVisibility.PUBLIC, logical_name
        )
        if not append:
            if payload != []:
                raise ResidentArtifactAdapterError(
                    f"{logical_name} initialization must be an empty list"
                )
            # The typed writer creates both histories at construction time.
            return
        if not isinstance(payload, Mapping):
            raise ResidentArtifactAdapterError(
                f"{logical_name} append payload must be a mapping"
            )
        sequence = payload.get("sequence")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise ResidentArtifactAdapterError(
                "model stream sequence must be non-negative"
            )
        pending = self._pending.setdefault(sequence, {})
        if logical_name == "model_raw_responses":
            if set(payload) != {"sequence", "prompt", "raw_response"}:
                raise ResidentArtifactAdapterError(
                    "raw model record fields do not match"
                )
            part = {
                "model_input": {"prompt": payload["prompt"]},
                "raw_response": payload["raw_response"],
            }
            self._merge_pending(sequence, pending, part)
            self.writer.record_model_attempt(
                sequence=sequence,
                model_input=part["model_input"],
                raw_response=part["raw_response"],
            )
        else:
            if set(payload) != {"sequence", "action"}:
                raise ResidentArtifactAdapterError(
                    "parsed action record fields do not match"
                )
            self._merge_pending(
                sequence,
                pending,
                {"parsed_action": payload["action"]},
            )
        required = {"model_input", "raw_response", "parsed_action"}
        if set(pending) == required:
            triple = (
                pending["model_input"],
                pending["raw_response"],
                pending["parsed_action"],
            )
            previous = self._committed.get(sequence)
            if previous is not None and previous != triple:
                raise ResidentArtifactAdapterError(
                    "model exchange changed after commit"
                )
            self.writer.record_model_exchange(
                action_index=sequence,
                model_input=triple[0],
                raw_response=triple[1],
                parsed_action=triple[2],
            )
            self._committed[sequence] = triple
            self._pending.pop(sequence)

    def _record_event(
        self,
        visibility: ArtifactVisibility,
        payload: Any,
        append: bool,
    ) -> None:
        self._require_visibility(
            visibility, ArtifactVisibility.SYSTEM, "events"
        )
        if not append:
            if payload not in ([], {}):
                raise ResidentArtifactAdapterError(
                    "events initialization must be empty"
                )
            return
        if not isinstance(payload, Mapping):
            raise ResidentArtifactAdapterError(
                "event append payload must be a mapping"
            )
        self._events.append(dict(payload))
        # The typed writer intentionally owns the required lifecycle `events`
        # record.  This bridge adds the complete generic orchestrator stream as
        # a separate system artifact so neither stream overwrites the other.
        self.writer._write_json(  # noqa: SLF001 - internal runtime bridge
            logical_name="orchestrator_events",
            relative_path="system/orchestrator_events.json",
            visibility=ArtifactVisibility.SYSTEM,
            payload=self._events,
        )

    @staticmethod
    def _merge_pending(
        sequence: int,
        pending: dict[str, Any],
        new_parts: Mapping[str, Any],
    ) -> None:
        for key, value in new_parts.items():
            if key in pending and pending[key] != value:
                raise ResidentArtifactAdapterError(
                    f"model stream sequence {sequence} changed {key}"
                )
            pending[key] = value

    @staticmethod
    def _require_visibility(
        actual: ArtifactVisibility,
        expected: ArtifactVisibility,
        logical_name: str,
    ) -> None:
        if actual is not expected:
            raise ResidentArtifactAdapterError(
                f"{logical_name} requires {expected.value} visibility"
            )

    @staticmethod
    def _exact_png(payload: Any, logical_name: str) -> bytes:
        if not isinstance(payload, Mapping):
            raise FrameContentUnavailableError(
                f"{logical_name}: {FRAME_PROTOCOL_BLOCKER}"
            )
        content = payload.get("png_bytes")
        digest = payload.get("png_sha256")
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise FrameContentUnavailableError(
                f"{logical_name}: {FRAME_PROTOCOL_BLOCKER}"
            )
        png = bytes(content)
        if (
            not png.startswith(_PNG_SIGNATURE)
            or b"IHDR" not in png[:64]
            or b"IEND" not in png[-32:]
        ):
            raise FrameContentUnavailableError(
                f"{logical_name}: png_bytes is not an encoded PNG"
            )
        actual = hashlib.sha256(png).hexdigest()
        if digest != actual:
            raise FrameContentUnavailableError(
                f"{logical_name}: png_sha256 does not match png_bytes"
            )
        return png
