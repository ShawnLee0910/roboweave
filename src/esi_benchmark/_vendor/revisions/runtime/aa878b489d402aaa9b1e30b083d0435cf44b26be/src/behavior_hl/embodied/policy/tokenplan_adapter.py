"""File-boundary validation for the isolated TokenPlan vision process."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from behavior_hl.embodied.policy.approach import (
    prompt_sha256,
    validate_approach_decision,
)
from behavior_hl.embodied.policy.protocol import ActionEnvelope
from behavior_hl.embodied.policy.registry import ActionRegistry


TOKENPLAN_MODEL = os.environ.get(
    "BEHAVIOR_HL_TOKENPLAN_MODEL",
    "qwen3.8-max-preview",
).strip()
if not TOKENPLAN_MODEL:
    raise RuntimeError("BEHAVIOR_HL_TOKENPLAN_MODEL must not be empty")
TOKENPLAN_TEMPERATURE = 0.1
TOKENPLAN_TOP_P = 0.9
TOKENPLAN_MAX_OUTPUT_TOKENS = 2048
TOKENPLAN_WALL_TIMEOUT_SECONDS = 180
# The benchmark owns retry accounting.  Hidden SDK retries can outlive the
# per-call subprocess wall and produce an un-auditable timeout without a
# response artifact.
TOKENPLAN_MAX_RETRIES = int(
    os.environ.get("BEHAVIOR_HL_TOKENPLAN_MAX_RETRIES", "0")
)
if TOKENPLAN_MAX_RETRIES < 0:
    raise RuntimeError("BEHAVIOR_HL_TOKENPLAN_MAX_RETRIES must be non-negative")
REQUEST_FIELDS = frozenset(
    {
        "call_index",
        "prompt",
        "prompt_sha256",
        "frame_sha256",
        "history_frame_sha256s",
        "registry_schema_version",
    }
)
MAX_HISTORY_FRAMES = 5
PUBLIC_SNAPSHOT_FIELDS = frozenset(
    {
        "proprio",
        "cam_rel_poses",
        "left_wrist_rgb",
        "left_wrist_depth_linear",
        "right_wrist_rgb",
        "right_wrist_depth_linear",
        "head_rgb",
        "head_depth_linear",
    }
)


@dataclass(frozen=True, slots=True)
class TokenPlanApproachRequest:
    call_index: int
    prompt: str
    prompt_sha256: str
    frame_sha256: str
    registry_schema_version: str
    history_frame_sha256s: tuple[str, ...] = ()

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "TokenPlanApproachRequest":
        if not isinstance(payload, Mapping):
            raise ValueError("TokenPlan request must be a JSON object")
        if set(payload) != REQUEST_FIELDS:
            raise ValueError("TokenPlan request fields do not match contract")
        call_index = payload["call_index"]
        if (
            isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or call_index <= 0
        ):
            raise ValueError("call_index must be a positive integer")
        prompt = payload["prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        expected_prompt_hash = prompt_sha256(prompt)
        if payload["prompt_sha256"] != expected_prompt_hash:
            raise ValueError("prompt_sha256 does not match prompt")
        frame_hash = _validate_sha256(
            payload["frame_sha256"],
            "frame_sha256",
        )
        raw_history = payload["history_frame_sha256s"]
        if not isinstance(raw_history, list):
            raise ValueError("history_frame_sha256s must be a JSON list")
        if len(raw_history) > MAX_HISTORY_FRAMES:
            raise ValueError(
                f"history_frame_sha256s exceeds {MAX_HISTORY_FRAMES} frames"
            )
        history_hashes = tuple(
            _validate_sha256(value, "history_frame_sha256s item")
            for value in raw_history
        )
        registry_schema_version = payload["registry_schema_version"]
        if (
            not isinstance(registry_schema_version, str)
            or not registry_schema_version.strip()
        ):
            raise ValueError(
                "registry_schema_version must be a non-empty string"
            )
        return cls(
            call_index=call_index,
            prompt=prompt,
            prompt_sha256=expected_prompt_hash,
            frame_sha256=frame_hash,
            history_frame_sha256s=history_hashes,
            registry_schema_version=registry_schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "prompt": self.prompt,
            "prompt_sha256": self.prompt_sha256,
            "frame_sha256": self.frame_sha256,
            "history_frame_sha256s": list(self.history_frame_sha256s),
            "registry_schema_version": self.registry_schema_version,
        }


def build_tokenplan_registry(schema_version: str) -> ActionRegistry:
    """Rebuild the exact public registry selected by the resident parent.

    The vision API runs in an isolated child process.  It must not silently
    fall back to the legacy point-grasp profile when the parent prompt exposes
    category-level ``grasp``.
    """

    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError("schema_version must be a non-empty string")
    from behavior_hl.embodied.policy.profiles import (
        build_picking_up_toys_registry,
    )

    candidates: list[ActionRegistry] = []
    for hands in (("left", "right"), ("right",), ("left",)):
        for grasp_mode in (
            "physical_local",
            "visible_official_macro",
            "visible_object_official_macro",
        ):
            try:
                candidate = build_picking_up_toys_registry(
                    calibrated_hands=hands,
                    grasp_mode=grasp_mode,
                )
            except ValueError:
                continue
            candidates.append(candidate)
    matches = [
        candidate
        for candidate in candidates
        if candidate.schema_version == schema_version
    ]
    if len(matches) != 1:
        raise ValueError(
            "unsupported or ambiguous TokenPlan registry schema version: "
            f"{schema_version!r}"
        )
    return matches[0]


def validate_tokenplan_prompt_registry(
    prompt: str,
    *,
    schema_version: str,
) -> ActionRegistry:
    """Fail before an API call if the prompt and child parser can drift."""

    import json
    from behavior_hl.embodied.policy.approach import (
        approach_action_manifest,
    )

    registry = build_tokenplan_registry(schema_version)
    try:
        prompt_payload = json.loads(prompt)
    except json.JSONDecodeError as exc:
        raise ValueError("approach prompt must be valid JSON") from exc
    if not isinstance(prompt_payload, dict):
        raise ValueError("approach prompt must be a JSON object")
    if prompt_payload.get("action_manifest") != approach_action_manifest(
        registry
    ):
        raise ValueError(
            "prompt action manifest differs from child parser registry"
        )
    return registry


def _validate_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def load_verified_head_rgb(
    snapshot_path: Path,
    *,
    expected_frame_sha256: str,
) -> np.ndarray:
    """Read only the frozen public snapshot and verify the exact head frame."""

    with np.load(snapshot_path, allow_pickle=False) as archive:
        if set(archive.files) != PUBLIC_SNAPSHOT_FIELDS:
            raise ValueError("public snapshot fields do not match contract")
        image = np.ascontiguousarray(
            archive["head_rgb"],
            dtype=np.uint8,
        )
    if image.shape != (720, 720, 3):
        raise ValueError("head RGB must have shape (720, 720, 3)")
    actual_hash = hashlib.sha256(image.tobytes()).hexdigest()
    if actual_hash != expected_frame_sha256:
        raise ValueError("head RGB hash does not match request")
    return image


def parse_strict_approach_response(
    raw_response: str,
    *,
    registry: ActionRegistry,
) -> ActionEnvelope:
    """Use the raw model text, never a tolerant JSON recovery path."""

    envelope = registry.parse_json(raw_response)
    validate_approach_decision(envelope)
    return envelope
