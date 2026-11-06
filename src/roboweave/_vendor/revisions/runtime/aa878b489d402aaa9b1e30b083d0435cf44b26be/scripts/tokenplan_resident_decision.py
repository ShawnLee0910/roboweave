#!/usr/bin/env python3
"""One isolated model decision over ordered public resident PNG frames."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


MODEL = os.environ.get(
    "BEHAVIOR_HL_TOKENPLAN_MODEL",
    "qwen3.8-max-preview",
).strip()
if not MODEL:
    raise RuntimeError("BEHAVIOR_HL_TOKENPLAN_MODEL must not be empty")
TOKENPLAN_MAX_RETRIES = int(
    os.environ.get("BEHAVIOR_HL_TOKENPLAN_MAX_RETRIES", "0")
)
if TOKENPLAN_MAX_RETRIES < 0:
    raise RuntimeError("BEHAVIOR_HL_TOKENPLAN_MAX_RETRIES must be non-negative")
MODEL_API = os.environ.get("BEHAVIOR_HL_MODEL_API", "chat_completions").strip()
if MODEL_API not in {"chat_completions", "responses"}:
    raise RuntimeError("BEHAVIOR_HL_MODEL_API must be chat_completions or responses")
REASONING_EFFORT = os.environ.get(
    "BEHAVIOR_HL_OPENAI_REASONING_EFFORT", ""
).strip()
if MODEL_API == "responses" and REASONING_EFFORT not in {
    "none", "low", "medium", "high", "xhigh", "max"
}:
    raise RuntimeError("Responses reasoning effort is invalid")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = (Path(os.environ["ESI_MODEL_ENV_FILE"]).expanduser()
            if os.environ.get("ESI_MODEL_ENV_FILE") else None)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SYSTEM_INSTRUCTION = (
    "You control a physical R1Pro. Use the task, ordered observations, prior "
    "actions, and action manifest in the request. If the request contains "
    "privileged_decision_brief or privileged_truth, use its live coordinates, "
    "relations, goal truth, "
    "executor diagnostics, and navigation hints as authoritative. Otherwise, "
    "never assume a map, object coordinates, segmentation, BDDL scope, held "
    "truth, goal truth, or executor outcome labels, and infer outcomes from "
    "the ordered images. Return exactly one JSON object using "
    "action, reasoning, optional args, and optional repeat only when the request "
    "declares navigation_repeat. The reasoning field is "
    "required and must preserve your original decision text. Do not emit markdown, "
    "answer, confidence, or additional fields."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    expected_project_root = os.environ.get("BEHAVIOR_HL_PROJECT_ROOT")
    if not expected_project_root:
        raise ValueError("BEHAVIOR_HL_PROJECT_ROOT is required")
    if Path(expected_project_root).resolve() != PROJECT_ROOT:
        raise ValueError("resident child project root differs from parent")
    if request.get("project_root") != str(PROJECT_ROOT):
        raise ValueError("resident request project root differs from child")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    base = _validate_request(request)
    pngs = []
    for path, expected_hash in zip(
        request["png_paths"],
        request["png_sha256s"],
        strict=True,
    ):
        png = Path(path).read_bytes()
        if (
            not png.startswith(PNG_SIGNATURE)
            or hashlib.sha256(png).hexdigest() != expected_hash
        ):
            raise ValueError("resident PNG identity changed")
        pngs.append(png)
    secrets = _load_secret()
    os.environ["OPENAI_API_KEY"] = secrets["TOKEN_PLAN_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = secrets["TOKEN_PLAN_BASE_URL"]

    started = time.monotonic()
    try:
        (
            raw,
            finish_reason,
            usage,
            provider_reasoning_content,
            provider_reasoning,
        ) = _call(
            prompt=request["prompt"],
            pngs=pngs,
            io_log=output.with_name(f"{output.stem}_io.jsonl"),
        )
    except Exception as exc:
        _write_json(
            output,
            {
                **base,
                "status": "infrastructure_error",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return 2
    try:
        parsed = json.loads(raw)
        valid_json_object = isinstance(parsed, dict)
    except json.JSONDecodeError:
        parsed = None
        valid_json_object = False
    original_reasoning = (
        parsed.get("reasoning")
        if isinstance(parsed, dict)
        else None
    )
    if (
        valid_json_object
        and (
            not isinstance(original_reasoning, str)
            or not original_reasoning.strip()
        )
    ):
        valid_json_object = False
    _write_json(
        output,
        {
            **base,
            "status": (
                "valid_json_object"
                if valid_json_object
                else "invalid_response"
            ),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "finish_reason": finish_reason,
            "usage": usage,
            "raw_response": raw,
            "original_reasoning": original_reasoning,
            "provider_reasoning_content": provider_reasoning_content,
            "provider_reasoning": provider_reasoning,
        },
    )
    return 0


def _validate_request(request):
    required = {
        "schema_version",
        "model",
        "call_index",
        "project_root",
        "prompt",
        "prompt_sha256",
        "png_paths",
        "png_sha256s",
    }
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError("resident request fields do not match")
    if request["schema_version"] != (
        "behavior-hl-tokenplan-resident-request-v2"
    ):
        raise ValueError("resident request schema changed")
    if request["model"] != MODEL:
        raise ValueError("resident model changed")
    if request["project_root"] != str(PROJECT_ROOT):
        raise ValueError("resident request project root changed")
    if (
        isinstance(request["call_index"], bool)
        or not isinstance(request["call_index"], int)
        or request["call_index"] <= 0
    ):
        raise ValueError("call_index must be positive")
    prompt = request["prompt"]
    if (
        not isinstance(prompt, str)
        or not prompt
        or hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        != request["prompt_sha256"]
    ):
        raise ValueError("prompt identity changed")
    paths = request["png_paths"]
    hashes = request["png_sha256s"]
    if (
        not isinstance(paths, list)
        or not isinstance(hashes, list)
        or not 1 <= len(paths) <= 8
        or len(paths) != len(hashes)
        or any(not isinstance(path, str) or not path for path in paths)
    ):
        raise ValueError("resident ordered PNG contract changed")
    for name, value in [
        ("prompt_sha256", request["prompt_sha256"]),
        *[("png_sha256", item) for item in hashes],
    ]:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} is not a lowercase SHA-256")
    return {
        "model": MODEL,
        "model_api": MODEL_API,
        "reasoning_effort": REASONING_EFFORT or None,
        "call_index": request["call_index"],
        "project_root": str(PROJECT_ROOT),
        "prompt_sha256": request["prompt_sha256"],
        "png_sha256s": list(hashes),
        "max_retries": TOKENPLAN_MAX_RETRIES,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_secret():
    loaded = {}
    for raw in (
        ENV_FILE.read_text(encoding="utf-8").splitlines()
        if ENV_FILE is not None and ENV_FILE.is_file()
        else ()
    ):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        loaded[key.strip()] = value.strip().strip("'\"")
    key = (
        os.environ.get("TOKEN_PLAN_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or loaded.get("TOKEN_PLAN_API_KEY")
    )
    base_url = (
        os.environ.get("TOKEN_PLAN_BASE_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or loaded.get("TOKEN_PLAN_BASE_URL")
    )
    if not key or not base_url:
        raise ValueError("TokenPlan secret file is incomplete")
    return {
        "TOKEN_PLAN_API_KEY": key,
        "TOKEN_PLAN_BASE_URL": base_url,
    }


def _call(*, prompt, pngs, io_log):
    import httpx
    from openai import OpenAI

    base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=base_url,
        # Default 180 s is the frozen value; the override exists for
        # authorized slow agent-bridge lanes and must exceed the bridge's own
        # agent timeout so the HTTP client never gives up first.
        http_client=httpx.Client(
            trust_env=False,
            timeout=float(
                os.environ.get(
                    "BEHAVIOR_HL_TOKENPLAN_HTTP_TIMEOUT_SECONDS", "180"
                )
            ),
        ),
        max_retries=TOKENPLAN_MAX_RETRIES,
    )
    user_content = [{"type": "text", "text": prompt}]
    for order, png in enumerate(pngs):
        image_url = (
            "data:image/png;base64,"
            + base64.b64encode(png).decode("ascii")
        )
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": f"Onboard observation frame {order + 1} of {len(pngs)}.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ]
        )
    started = time.monotonic()
    if MODEL_API == "responses":
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_INSTRUCTION,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": item["text"]}
                    if item["type"] == "text"
                    else {
                        "type": "input_image",
                        "image_url": item["image_url"]["url"],
                    }
                    for item in user_content
                ],
            }],
            reasoning={
                "effort": REASONING_EFFORT,
                "context": "all_turns",
                "summary": "detailed",
            },
            text={"verbosity": "low"},
            max_output_tokens=2048,
            store=False,
        )
        raw = response.output_text if isinstance(response.output_text, str) else ""
        finish_reason = str(response.status)
        provider_reasoning_content = None
        provider_reasoning = None
        usage_record = getattr(response, "usage", None)
        usage = {
            "prompt_tokens": getattr(usage_record, "input_tokens", None),
            "completion_tokens": getattr(usage_record, "output_tokens", None),
            "total_tokens": getattr(usage_record, "total_tokens", None),
        }
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=2048,
            temperature=0.1,
            top_p=0.9,
            extra_body={"enable_thinking": True},
        )
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        raw = content if isinstance(content, str) else ""
        provider_reasoning_content = getattr(
            choice.message,
            "reasoning_content",
            None,
        )
        provider_reasoning = getattr(choice.message, "reasoning", None)
        usage_record = getattr(response, "usage", None)
        usage = {
            "prompt_tokens": getattr(usage_record, "prompt_tokens", None),
            "completion_tokens": getattr(
                usage_record, "completion_tokens", None
            ),
            "total_tokens": getattr(usage_record, "total_tokens", None),
        }
        finish_reason = str(choice.finish_reason)
    _append_jsonl(
        io_log,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "model_api": MODEL_API,
            "reasoning_effort": REASONING_EFFORT or None,
            "base_url": base_url,
            "trust_env": False,
            "max_retries": TOKENPLAN_MAX_RETRIES,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "finish_reason": finish_reason,
            "system_instruction": SYSTEM_INSTRUCTION,
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "png_sha256s": [
                hashlib.sha256(png).hexdigest() for png in pngs
            ],
            "raw_response": raw,
            "provider_reasoning_content": provider_reasoning_content,
            "provider_reasoning": provider_reasoning,
            "usage": usage,
        },
    )
    return (
        raw,
        finish_reason,
        usage,
        provider_reasoning_content,
        provider_reasoning,
    )


def _write_json(path, payload):
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
