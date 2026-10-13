#!/usr/bin/env python3
"""Run one isolated TokenPlan Qwen approach decision from a public snapshot."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROJECT_ROOT = os.environ.get("BEHAVIOR_HL_PROJECT_ROOT")
if (
    EXPECTED_PROJECT_ROOT
    and Path(EXPECTED_PROJECT_ROOT).resolve() != PROJECT_ROOT
):
    raise RuntimeError(
        "TokenPlan child project root differs from its parent contract"
    )
ENV_FILE = (Path(os.environ["ESI_MODEL_ENV_FILE"]).expanduser()
            if os.environ.get("ESI_MODEL_ENV_FILE") else None)

for directory in (PROJECT_ROOT / "src",):
    sys.path.insert(0, str(directory))

from behavior_hl.embodied.policy.approach import (  # noqa: E402
    APPROACH_SYSTEM_INSTRUCTION,
)
from behavior_hl.embodied.policy.tokenplan_adapter import (  # noqa: E402
    TOKENPLAN_MAX_OUTPUT_TOKENS,
    TOKENPLAN_MAX_RETRIES,
    TOKENPLAN_MODEL,
    TOKENPLAN_TEMPERATURE,
    TOKENPLAN_TOP_P,
    TOKENPLAN_WALL_TIMEOUT_SECONDS,
    TokenPlanApproachRequest,
    load_verified_head_rgb,
    parse_strict_approach_response,
    validate_tokenplan_prompt_registry,
)


def _load_environment(io_log: Path) -> None:
    loaded: dict[str, str] = {}
    for raw in (ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE is not None else []):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        loaded[key.strip()] = value.strip().strip("'\"")
    # The launcher may provide a run-scoped endpoint / credential.  Prefer
    # that protected process environment and keep the legacy secret file as
    # a compatibility fallback.
    token_plan_key = (
        os.environ.get("TOKEN_PLAN_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or loaded.get("TOKEN_PLAN_API_KEY")
    )
    token_plan_url = (
        os.environ.get("TOKEN_PLAN_BASE_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or loaded.get("TOKEN_PLAN_BASE_URL")
    )
    if not token_plan_key or not token_plan_url:
        raise ValueError("TokenPlan secret file is incomplete")
    os.environ["TOKEN_PLAN_API_KEY"] = token_plan_key
    os.environ["TOKEN_PLAN_BASE_URL"] = token_plan_url
    os.environ["OPENAI_API_KEY"] = token_plan_key
    os.environ["OPENAI_BASE_URL"] = token_plan_url
    os.environ["ESI_DASHSCOPE_ENABLE_THINKING"] = "true"
    os.environ["EB_OPENAI_REQUEST_TIMEOUT_SECONDS"] = str(
        TOKENPLAN_WALL_TIMEOUT_SECONDS
    )
    os.environ["ESI_LLM_IO_LOG"] = str(io_log)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _image_part(path: Path) -> dict[str, Any]:
    image_url = (
        "data:image/png;base64,"
        + base64.b64encode(path.read_bytes()).decode("ascii")
    )
    return {"type": "image_url", "image_url": {"url": image_url}}


def _build_temporal_user_content(
    *,
    prompt: str,
    history_frame_paths: tuple[Path, ...],
    frame_path: Path,
) -> list[dict[str, Any]]:
    if len(history_frame_paths) > 5:
        raise ValueError("temporal user content exceeds five past views")
    contents: list[dict[str, Any]] = []
    for index, history_path in enumerate(history_frame_paths, start=1):
        contents.append(
            {
                "type": "text",
                "text": (
                    f"[PAST VIEW {index}/{len(history_frame_paths)} - "
                    "chronological order]"
                ),
            }
        )
        contents.append(_image_part(history_path))
    contents.append(
        {"type": "text", "text": "[CURRENT VIEW - always last]"}
    )
    contents.append(_image_part(frame_path))
    contents.append({"type": "text", "text": prompt})
    return contents


def _call_tokenplan(
    *,
    prompt: str,
    history_frame_paths: tuple[Path, ...],
    frame_path: Path,
    io_log: Path,
) -> tuple[
    dict[str, Any],
    str,
    str | None,
    dict[str, int | None],
    Any,
    Any,
]:
    """Call the endpoint without ESI's generic answer/confidence instruction."""

    import httpx
    from openai import OpenAI

    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or ""
    )
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or ""
    ).rstrip("/")
    if not api_key or not base_url:
        raise ValueError("TokenPlan API key or base URL is missing")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(
            trust_env=False,
            timeout=float(TOKENPLAN_WALL_TIMEOUT_SECONDS),
        ),
        max_retries=TOKENPLAN_MAX_RETRIES,
    )
    strict_instruction = (
        f"{APPROACH_SYSTEM_INSTRUCTION}\n\n"
        "The only allowed top-level JSON fields are action, args, and "
        "reasoning. reasoning is required and must preserve your original "
        "decision text. Do not emit answer, confidence, markdown, or any other "
        "field. Omit args for argument-free actions. Follow the action "
        "manifest exactly for actions that require args."
    )
    # Preserve the proven ESI-Bench ordering: up to five recent past views,
    # oldest first, then the current view last, followed by the task prompt.
    user_content = _build_temporal_user_content(
        prompt=prompt,
        history_frame_paths=history_frame_paths,
        frame_path=frame_path,
    )
    messages = [
        {"role": "system", "content": strict_instruction},
        {"role": "user", "content": user_content},
    ]
    started = time.monotonic()
    response = client.chat.completions.create(
        model=TOKENPLAN_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=TOKENPLAN_MAX_OUTPUT_TOKENS,
        temperature=TOKENPLAN_TEMPERATURE,
        top_p=TOKENPLAN_TOP_P,
        extra_body={"enable_thinking": True},
    )
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    raw = content if isinstance(content, str) else ""
    provider_reasoning_content = getattr(
        message,
        "reasoning_content",
        None,
    )
    provider_reasoning = getattr(message, "reasoning", None)
    parsed_diagnostic: dict[str, Any]
    try:
        parsed_value = json.loads(raw)
        parsed_diagnostic = (
            parsed_value
            if isinstance(parsed_value, dict)
            else {"non_object_response": parsed_value}
        )
    except json.JSONDecodeError:
        parsed_diagnostic = {"json_decode_failed": True}
    finish_reason = str(getattr(choice, "finish_reason", None))
    response_usage = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": getattr(response_usage, "prompt_tokens", None),
        "completion_tokens": getattr(
            response_usage,
            "completion_tokens",
            None,
        ),
        "total_tokens": getattr(response_usage, "total_tokens", None),
    }
    _append_jsonl(
        io_log,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": TOKENPLAN_MODEL,
            "base_url": base_url,
            "trust_env": False,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "finish_reason": finish_reason,
            "system_instruction": strict_instruction,
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "image_path": str(frame_path),
            "history_image_paths": [
                str(path) for path in history_frame_paths
            ],
            "ordered_image_paths": [
                *(str(path) for path in history_frame_paths),
                str(frame_path),
            ],
            "raw_response": raw,
            "original_reasoning": (
                parsed_diagnostic.get("reasoning")
                if isinstance(parsed_diagnostic, dict)
                else None
            ),
            "provider_reasoning_content": provider_reasoning_content,
            "provider_reasoning": provider_reasoning,
            "parsed_diagnostic": parsed_diagnostic,
            "usage": usage,
        },
    )
    return (
        parsed_diagnostic,
        raw,
        finish_reason,
        usage,
        provider_reasoning_content,
        provider_reasoning,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-snapshot",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot_path = args.snapshot.resolve()
    request_path = args.request.resolve()
    output_path = args.output.resolve()
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    request = TokenPlanApproachRequest.from_payload(request_payload)
    registry = validate_tokenplan_prompt_registry(
        request.prompt,
        schema_version=request.registry_schema_version,
    )

    history_snapshot_paths = tuple(
        path.resolve() for path in args.history_snapshot
    )
    if len(history_snapshot_paths) != len(request.history_frame_sha256s):
        raise ValueError(
            "history snapshot count differs from request hash contract"
        )
    history_frames = tuple(
        load_verified_head_rgb(path, expected_frame_sha256=expected_hash)
        for path, expected_hash in zip(
            history_snapshot_paths,
            request.history_frame_sha256s,
            strict=True,
        )
    )

    frame = load_verified_head_rgb(
        snapshot_path,
        expected_frame_sha256=request.frame_sha256,
    )
    frame_dir = output_path.parent / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    history_frame_paths: list[Path] = []
    for history_index, history_frame in enumerate(history_frames, start=1):
        history_path = frame_dir / (
            f"call_{request.call_index:03d}_past_{history_index:02d}.png"
        )
        Image.fromarray(history_frame).save(history_path)
        history_frame_paths.append(history_path)
    frame_path = frame_dir / f"call_{request.call_index:03d}_head.png"
    Image.fromarray(frame).save(frame_path)
    frame_png_sha256 = hashlib.sha256(frame_path.read_bytes()).hexdigest()

    io_log = (
        output_path.parent
        / "io"
        / f"call_{request.call_index:03d}.jsonl"
    )
    io_log.parent.mkdir(parents=True, exist_ok=True)
    _load_environment(io_log)

    base_record = {
        "model": TOKENPLAN_MODEL,
        "call_index": request.call_index,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request_path": str(request_path),
        "snapshot_path": str(snapshot_path),
        "history_snapshot_paths": [
            str(path) for path in history_snapshot_paths
        ],
        "prompt_sha256": request.prompt_sha256,
        "head_rgb_sha256": request.frame_sha256,
        "history_head_rgb_sha256s": list(
            request.history_frame_sha256s
        ),
        "ordered_head_rgb_sha256s": [
            *request.history_frame_sha256s,
            request.frame_sha256,
        ],
        "head_png_path": str(frame_path),
        "history_head_png_paths": [
            str(path) for path in history_frame_paths
        ],
        "head_png_sha256": frame_png_sha256,
        "temperature": TOKENPLAN_TEMPERATURE,
        "top_p": TOKENPLAN_TOP_P,
        "max_output_tokens": TOKENPLAN_MAX_OUTPUT_TOKENS,
        "wall_timeout_seconds": TOKENPLAN_WALL_TIMEOUT_SECONDS,
        "max_retries": TOKENPLAN_MAX_RETRIES,
        "trust_env": False,
        "project_root": str(PROJECT_ROOT),
        "registry_schema_version": request.registry_schema_version,
    }
    started = time.monotonic()
    try:
        (
            parsed,
            raw,
            finish_reason,
            usage,
            provider_reasoning_content,
            provider_reasoning,
        ) = _call_tokenplan(
            prompt=request.prompt,
            history_frame_paths=tuple(history_frame_paths),
            frame_path=frame_path,
            io_log=io_log,
        )
    except Exception as exc:
        _write_json(
            output_path,
            {
                **base_record,
                "status": "infrastructure_error",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return 2

    original_reasoning = (
        parsed.get("reasoning")
        if isinstance(parsed, dict)
        else None
    )
    try:
        if (
            not isinstance(original_reasoning, str)
            or not original_reasoning.strip()
        ):
            raise ValueError(
                "approach response requires original reasoning"
            )
        envelope = parse_strict_approach_response(
            raw,
            registry=registry,
        )
    except Exception as exc:
        _write_json(
            output_path,
            {
                **base_record,
                "status": "invalid_response",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "finish_reason": finish_reason,
                "usage": usage,
                "provider_parsed_diagnostic": parsed,
                "raw_response": raw,
                "original_reasoning": original_reasoning,
                "provider_reasoning_content": provider_reasoning_content,
                "provider_reasoning": provider_reasoning,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return 0

    _write_json(
        output_path,
        {
            **base_record,
            "status": "valid",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "finish_reason": finish_reason,
            "usage": usage,
            "action": envelope.to_dict(),
            "raw_response": raw,
            "original_reasoning": original_reasoning,
            "provider_reasoning_content": provider_reasoning_content,
            "provider_reasoning": provider_reasoning,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
