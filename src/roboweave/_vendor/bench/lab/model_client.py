#!/usr/bin/env python3
"""One model episode client for the model-independent resident scene.

This process owns Prompt loading, model/provider routing, native tool calls,
conversation history, and model artifacts.  It never imports simulator
libraries and can be stopped or replaced without touching the resident scene.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


from Agent import (
    ACTION20_CORE_INSTRUCTIONS_PATH,
    ACTION20_TOOL_SCHEMA_PATH,
    EpisodeConversation,
    PrivateEpisodeContract,
    PublicObservation,
    PublicToolResult,
    PHYSICAL_HISTORY_MAX,
    PHYSICAL_HISTORY_OPTIONS,
    SystemPromptRevision,
    TERMINATION_ACTION,
    load_action20_tool_schema,
    load_common_system_prompt_revision,
    load_prompt_revision,
    with_automatic_interaction_approach_tool_schema,
    with_overhead_view_tool_schema,
)
from action20_protocol import (
    ACTION_DECISION_LIMIT,
    MARKER_ACTIONS,
    PUBLIC_TOOL_ACTIONS,
)
from agent_protocol import isolate_native_reasoning, replay_native_reasoning
from agent_artifacts import (
    link_or_copy as _link_or_copy,
    publish_action_after_view,
    write_conversation_views as _write_conversation_views,
)
from agent_observation import materialize_observation, observation_from_artifact
from agent_resume import (
    MAX_RESUME_ARTIFACT_BYTES,
    MODEL_ROLLING_CHECKPOINT_VERSION,
    MODEL_ROLLING_CHECKPOINT_VERSION_V1,
    PAUSED_CLIENT_STATUSES,
    RESUME_CHECKPOINT_VERSION,
    RESUME_CHECKPOINT_VERSION_V3,
    RESUME_COUNTER_FIELDS as _RESUME_COUNTER_FIELDS,
    ROLLING_CHECKPOINT_MAXIMUM_RETAINED,
    load_latest_model_rolling_checkpoint,
    load_latest_persistent_model_checkpoint,
    load_resume_checkpoint,
    resume_client_manifest_identity as _resume_client_manifest_identity,
    validate_repeat_inflight,
    write_model_rolling_checkpoint,
    write_resume_checkpoint,
)
from scene_protocol import (
    DEFAULT_READY_PATH,
    SCENE_PROTOCOL_VERSION,
    ModelClientError,
    SceneClient,
    SceneRequestError,
    _canonical_json_bytes,
    _read_strict_json,
    _write_json,
)
from native_tool_preflight import run_native_tool_preflight
from campaign_manifest import CampaignManifestError, bind_campaign_run
from run_artifact_layout import (
    allocate_incomplete_run,
    finalize_completed_run,
)
from provider_retry import ProviderRetryPolicy, parse_retry_after_seconds
from anthropic_messages_adapter import (
    ADAPTER_VERSION as ANTHROPIC_ADAPTER_VERSION,
    ANTHROPIC_EFFORTS,
    ANTHROPIC_VERSION,
    AnthropicMessagesAdapterError,
    anthropic_messages_url,
    anthropic_to_chat_completion,
    assemble_anthropic_sse,
    chat_request_to_anthropic,
    extract_anthropic_replay_content,
    parse_sse_json_events,
)
from responses_adapter import (
    ADAPTER_VERSION as RESPONSES_ADAPTER_VERSION,
    RESPONSES_REASONING_CONTEXTS,
    RESPONSES_REASONING_EFFORTS,
    RESPONSES_REASONING_MODES,
    RESPONSES_REASONING_SUMMARIES,
    RESPONSES_TEXT_VERBOSITIES,
    ResponsesAdapterError,
    chat_request_to_responses,
    extract_responses_replay_items,
    function_arguments_equivalent,
    responses_to_chat_completion,
)
from agent_observation import RGBD_PROTOCOL_VERSION
from agent_prompt import prompt_provenance_record


BENCH_TRY_ROOT = Path(__file__).resolve().parents[1]
_AGENT_SOURCE_PATHS = tuple(
    BENCH_TRY_ROOT / name
    for name in (
        "Agent.py",
        "agent_artifacts.py",
        "agent_common.py",
        "agent_observation.py",
        "agent_prompt.py",
        "agent_protocol.py",
        "agent_resume.py",
        "episode_conversation.py",
    )
)
DEFAULT_RUNS_ROOT = BENCH_TRY_ROOT / "runs"
DEFAULT_MAXIMUM_ACTION_DECISIONS = ACTION_DECISION_LIMIT
DEFAULT_MAXIMUM_MODEL_CALLS = 3000
MAXIMUM_ACTION_DECISIONS = ACTION_DECISION_LIMIT
MAXIMUM_MODEL_CALLS = 30_000
DEFAULT_MAXIMUM_OUTPUT_TOKENS = None
DEFAULT_MAXIMUM_CONSECUTIVE_BLOCKED = 30
MAXIMUM_PROTOCOL_ATTEMPTS_PER_DECISION = 10
DEFAULT_MAXIMUM_PROVIDER_ATTEMPTS_PER_PROTOCOL_ATTEMPT = 10
MAXIMUM_RESPONSES_PREFLIGHT_PROVIDER_ATTEMPTS = 3
MAXIMUM_AUTOMATIC_DECISION_REPAIRS_PER_SEQUENCE = 3
MAXIMUM_AUTOMATIC_CHECKPOINT_RESTORES = 3
MAXIMUM_AUTOMATIC_SAME_SCENE_RESETS = 5
MAXIMUM_AUTOMATIC_PROVIDER_RETRY_WINDOWS = 5
AUTOMATIC_PROVIDER_RETRY_WINDOW_DELAY_SECONDS = 60.0
PROVIDER_RETRY_STATE_VERSION = "behavior-hl-provider-retry-state/v1"
NORMALIZATION_STATS_VERSION = "behavior-hl-model-normalization-stats/v1"
_NORMALIZATION_STAT_FIELDS = (
    "normalized_model_responses",
    "ignored_target_fields",
    "ignored_navigation_fields",
    "eager_marker_targets_removed",
)
PROTOCOL_ATTEMPT_SEEDS = tuple(range(19506, 19516))
MAX_PROVIDER_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_PROVIDER_ERROR_BODY_BYTES = 16 * 1024
MAX_PROVIDER_ERROR_PREVIEW_CHARS = 2 * 1024
MAX_INSTRUCTION_REVIEW_BYTES = 4 * 1024 * 1024
SSH_PASSWORD_ENV = "BENCH_SSH_PASSWORD"
SSH_ASKPASS_PATH = Path(__file__).resolve().parent / "ssh_password_askpass.py"
RESUME_INVOCATION_VERSION_V1 = "behavior-hl-model-resume-invocation/v1"
RESUME_INVOCATION_VERSION_V2 = "behavior-hl-model-resume-invocation/v2-campaign"
RESUME_INVOCATION_VERSION_V3 = "behavior-hl-model-resume-invocation/v3-generation"
RESUME_INVOCATION_VERSION_V4 = "behavior-hl-model-resume-invocation/v4-responses"
RESUME_INVOCATION_VERSION_V5 = (
    "behavior-hl-model-resume-invocation/v5-provider-retry-policy"
)
RESUME_INVOCATION_VERSION_V6 = (
    "behavior-hl-model-resume-invocation/v6-reasoning-summary"
)
RESUME_INVOCATION_VERSION_V7 = (
    "behavior-hl-model-resume-invocation/v7-provider-backoff"
)
RESUME_INVOCATION_VERSION_V8 = (
    "behavior-hl-model-resume-invocation/v8-macro-action-budget"
)
RESUME_INVOCATION_VERSION_V9 = "behavior-hl-model-resume-invocation/v9-wire-model"
RESUME_INVOCATION_VERSION_V10 = "behavior-hl-model-resume-invocation/v10-rgbd-v1"
RESUME_INVOCATION_VERSION_V11 = (
    "behavior-hl-model-resume-invocation/v11-overhead-view"
)
RESUME_INVOCATION_VERSION_V12 = (
    "behavior-hl-model-resume-invocation/v12-automatic-interaction-approach"
)
RESUME_INVOCATION_VERSION = (
    "behavior-hl-model-resume-invocation/v13-responses-image-proxy"
)
R2_IMAGE_TRANSPORT_ENV = None  # Private upload transport is not part of the public runner.
R2_SIGNED_URL_SECONDS = 86_400
OVERHEAD_VIEW_MODES = frozenset({"off", "always", "on_blocked"})
CHAT_COMPLETIONS_REASONING_EFFORTS = frozenset({"low", "high", "max"})


def _gemini_requires_function_response_name(model: str, base_url: str) -> bool:
    return (
        model == "gemini-3.8-flash"
        and base_url.rstrip("/") == "https://api.apikey.fan/v1"
    )


def _with_function_response_names(request: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Gemini OpenAI-compat history: role=model plus tool-result name.

    This gateway translates ``role=assistant`` plus ``tool_calls`` into a
    GenerateContent ``function_response`` and then rejects it for a missing
    name. This workaround is scoped to the tested apikey.fan model route.
    """

    payload = json.loads(json.dumps(request, ensure_ascii=False, allow_nan=False))
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload
    names_by_id: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            message["role"] = "model"
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id")
                function = call.get("function")
                name = function.get("name") if isinstance(function, dict) else None
                if isinstance(call_id, str) and isinstance(name, str) and name:
                    names_by_id[call_id] = name
        elif message.get("role") == "model":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id")
                function = call.get("function")
                name = function.get("name") if isinstance(function, dict) else None
                if isinstance(call_id, str) and isinstance(name, str) and name:
                    names_by_id[call_id] = name
        elif message.get("role") == "tool":
            existing = message.get("name")
            if isinstance(existing, str) and existing:
                continue
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str) and call_id in names_by_id:
                message["name"] = names_by_id[call_id]
            else:
                message["name"] = "robot_action"
    return payload
RESPONSES_REPLAY_VERSION = "behavior-hl-responses-replay/v2-normalization-aware"
RESPONSES_REPLAY_COMPATIBLE_ADAPTER_VERSIONS = frozenset(
    {
        RESPONSES_ADAPTER_VERSION,
        "esi.responses-transport-adapter/v3-reasoning-summary",
        "esi.responses-transport-adapter/v2-opaque-reasoning-replay",
    }
)
PROJECT_ROOT = Path(os.environ.get("BEHAVIOR_HL_PROJECT_ROOT", str(Path(__file__).resolve().parents[2] / "runtime"))).resolve()
_SINGLE_ACTION_TAG = re.compile(r"\A\s*<([a-z][a-z0-9_]*)>\s*\Z")
_RYNNBRAIN_TARGETED_ACTION_BLOCK = re.compile(
    r"\A\s*"
    r"<robot_action>\s*"
    r"<parameter=action>\s*"
    r"(?P<action>[a-z][a-z0-9_]*)\s*"
    r"</parameter>\s*"
    r"<parameter=target>\s*"
    r"(?P<target>[1-9][0-9]{0,2})\s*"
    r"</parameter>\s*"
    r"<parameter=target_description>\s*"
    r"(?P<target_description>[^\r\n<>]{1,64})\s*"
    r"</parameter>\s*"
    r"</function>\s*"
    r"</tool_call>\s*\Z"
)


class ProviderRequestError(ModelClientError):
    """A provider failure classified before any simulator action is issued."""

    def __init__(
        self,
        message: str,
        *,
        failure_type: str,
        retryable: bool,
        status_code: int | None = None,
        response_body: bytes = b"",
        response_body_truncated: bool = False,
        request_body_sha256: str | None = None,
        request_body_path: str | None = None,
        retry_after_seconds: float | None = None,
        redaction_values: tuple[str, ...] = (),
    ) -> None:
        super().__init__(_redact_provider_error_text(message, redaction_values))
        self._redaction_values = redaction_values
        self.failure_type = failure_type
        self.retryable = retryable
        self.status_code = status_code
        self.response_body = bytes(response_body)
        self.response_body_truncated = response_body_truncated
        self.request_body_sha256 = request_body_sha256
        self.request_body_path = request_body_path
        self.retry_after_seconds = retry_after_seconds

    def artifact_fields(self) -> dict[str, Any]:
        preview = self.response_body.decode("utf-8", errors="replace")
        return {
            "failure_type": self.failure_type,
            "exception_type": type(self).__name__,
            "exception_message": _redact_provider_error_text(str(self), self._redaction_values),
            "retryable": self.retryable,
            "http_status": self.status_code,
            "response_body_bytes": len(self.response_body),
            "response_body_sha256": hashlib.sha256(self.response_body).hexdigest(),
            "response_body_truncated": self.response_body_truncated,
            "response_body_preview": _redact_provider_error_text(preview, self._redaction_values),
            "request_body_sha256": self.request_body_sha256,
            "request_body_path": self.request_body_path,
            "retry_after_seconds": self.retry_after_seconds,
        }


class SshLocalForward:
    def __init__(
        self,
        *,
        ssh_host: str,
        local_port: int,
        remote_port: int,
        log_path: Path,
    ) -> None:
        self.ssh_host = ssh_host
        self.local_port = local_port
        self.remote_port = remote_port
        self.log_path = Path(log_path)
        self._process: subprocess.Popen[bytes] | None = None
        self._stream: Any | None = None

    def start(self) -> Mapping[str, Any]:
        if self._process is not None or _tcp_open(self.local_port):
            raise ModelClientError("SSH local-forward port is unavailable")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.log_path.open("xb")
        environment = dict(os.environ)
        for name in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
        ):
            environment.pop(name, None)
        password_auth = bool(environment.get(SSH_PASSWORD_ENV))
        if password_auth:
            if not SSH_ASKPASS_PATH.is_file():
                raise ModelClientError("SSH password askpass helper is unavailable")
            environment["SSH_ASKPASS"] = str(SSH_ASKPASS_PATH)
            environment["SSH_ASKPASS_REQUIRE"] = "force"
            environment.setdefault("DISPLAY", "bench-ssh-password:0")
        self._process = subprocess.Popen(
            [
                "ssh", "-N", "-T",
                "-o", f"BatchMode={'no' if password_auth else 'yes'}",
                "-o", "NumberOfPasswordPrompts=1",
                "-o", "ExitOnForwardFailure=yes",
                "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=2",
                "-o", "ProxyCommand=none",
                "-o", "ProxyJump=none",
                "-L", f"127.0.0.1:{self.local_port}:127.0.0.1:{self.remote_port}",
                self.ssh_host,
            ],
            stdin=subprocess.DEVNULL,
            stdout=self._stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        for _ in range(100):
            if self._process.poll() is not None:
                code = self._process.returncode
                self.stop()
                raise ModelClientError(f"SSH forward exited during startup: {code}")
            if _tcp_open(self.local_port):
                return {
                    "status": "PASS",
                    "pid": self._process.pid,
                    "pgid": os.getpgid(self._process.pid),
                    "ssh_host": self.ssh_host,
                    "local_port": self.local_port,
                    "remote_port": self.remote_port,
                    "authentication": (
                        "explicit_password_askpass" if password_auth else "key_only"
                    ),
                    "http_proxy_used": False,
                }
            time.sleep(0.1)
        self.stop()
        raise ModelClientError("SSH forward did not become ready")

    def stop(self) -> Mapping[str, Any]:
        process = self._process
        signals: list[str] = []
        if process is not None and process.poll() is None:
            pgid = os.getpgid(process.pid)
            if pgid != process.pid or pgid == os.getpgrp():
                raise ModelClientError("SSH forward ownership changed")
            os.killpg(pgid, signal.SIGTERM)
            signals.append("SIGTERM")
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                signals.append("SIGKILL")
                process.wait(timeout=5)
        code = None if process is None else process.poll()
        if self._stream is not None:
            self._stream.close()
        self._process = None
        self._stream = None
        return {
            "status": "PASS",
            "signals_sent": signals,
            "exit_code": code,
            "local_port_closed": not _tcp_open(self.local_port),
        }


def _provider_metrics(
    response: Mapping[str, Any] | None,
    provider_wall_seconds: float,
) -> Mapping[str, Any]:
    usage = response.get("usage") if isinstance(response, Mapping) else None
    usage = usage if isinstance(usage, Mapping) else {}

    def token(*paths: tuple[str, ...]) -> int | None:
        for path in paths:
            value: Any = usage
            for field in path:
                value = value.get(field) if isinstance(value, Mapping) else None
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    input_tokens = token(("input_tokens",), ("prompt_tokens",))
    cached_input_tokens = token(
        ("input_tokens_details", "cached_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
    )
    if (
        input_tokens is None
        or cached_input_tokens is None
        or cached_input_tokens > input_tokens
    ):
        uncached_input_tokens = None
    else:
        uncached_input_tokens = input_tokens - cached_input_tokens
    return {
        "schema_version": "behavior-hl-provider-metrics/v1",
        "response_received": isinstance(response, Mapping),
        "provider_wall_seconds": round(max(0.0, provider_wall_seconds), 6),
        "server_inference_seconds": None,
        "network_excluded": False,
        "usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "uncached_input_tokens": uncached_input_tokens,
            "output_tokens": token(("output_tokens",), ("completion_tokens",)),
            "reasoning_output_tokens": token(
                ("output_tokens_details", "reasoning_tokens"),
                ("completion_tokens_details", "reasoning_tokens"),
            ),
        },
    }


def _aggregate_provider_metrics(
    output_dir: Path,
    call_count: int,
) -> Mapping[str, Any]:
    records: list[Mapping[str, Any]] = []
    for index in range(1, call_count + 1):
        path = output_dir / "provider_calls" / f"call_{index:04d}" / "provider_metrics.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, Mapping):
            records.append(record)

    walls = sorted(
        float(record["provider_wall_seconds"])
        for record in records
        if isinstance(record.get("provider_wall_seconds"), (int, float))
        and not isinstance(record.get("provider_wall_seconds"), bool)
    )
    response_count = sum(record.get("response_received") is True for record in records)
    fields = (
        "input_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    )
    totals: dict[str, int | None] = {}
    reported: dict[str, int] = {}
    for field in fields:
        values = [
            usage[field]
            for record in records
            if isinstance((usage := record.get("usage")), Mapping)
            and isinstance(usage.get(field), int)
            and not isinstance(usage[field], bool)
            and usage[field] >= 0
        ]
        totals[field] = sum(values) if values else None
        reported[field] = len(values)
    cache_hit_rate = None
    if (
        response_count > 0
        and reported["input_tokens"] == response_count
        and reported["cached_input_tokens"] == response_count
        and totals["input_tokens"]
    ):
        cache_hit_rate = round(
            totals["cached_input_tokens"] / totals["input_tokens"], 6
        )
    return {
        "schema_version": "behavior-hl-provider-metrics-summary/v1",
        "expected_call_count": call_count,
        "recorded_call_count": len(records),
        "response_received_count": response_count,
        "provider_wall_seconds_total": round(sum(walls), 6),
        "provider_wall_seconds_p50": (
            walls[(len(walls) - 1) // 2] if walls else None
        ),
        "provider_wall_seconds_p95": (
            walls[max(0, (95 * len(walls) + 99) // 100 - 1)] if walls else None
        ),
        "server_inference_seconds": None,
        "network_excluded": False,
        "usage": totals,
        "usage_reported_call_count": reported,
        "cache_hit_rate": cache_hit_rate,
        "monetary_cost": {
            "status": "UNPRICED",
            "currency": None,
            "estimated_cost": None,
            "reason": "pricing_snapshot_not_supplied",
        },
    }


class _NoProviderRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward credentials or request bodies beyond the configured endpoint.
        return None


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        model: str,
        wire_model: str | None = None,
        base_url: str,
        timeout_seconds: int,
        api_key_file: Path | None,
        api_key_env: str | None = None,
        maximum_output_tokens: int | None = DEFAULT_MAXIMUM_OUTPUT_TOKENS,
        thinking_tool_choice_auto: bool = False,
        action_tag_compatibility: bool = False,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        preserve_thinking: bool = False,
        provider_api: str = "chat_completions",
        require_api_key: bool = False,
        reasoning_effort: str | None = None,
        reasoning_context: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_summary: str | None = None,
        text_verbosity: str | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be nonempty")
        if wire_model is not None and (
            not isinstance(wire_model, str) or not wire_model.strip()
        ):
            raise ValueError("wire model must be nonempty")
        parsed = urllib.parse.urlsplit(base_url)
        loopback_http = (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        )
        try:
            address = ipaddress.ip_address(parsed.hostname or "")
        except ValueError:
            address = None
        private_http = (
            parsed.scheme == "http"
            and isinstance(address, ipaddress.IPv4Address)
            and address.is_private
            and not address.is_loopback
            and not address.is_link_local
            and not address.is_multicast
            and not address.is_unspecified
        )
        if parsed.scheme != "https" and not loopback_http and not private_http:
            raise ValueError(
                "plaintext model HTTP is allowed only through loopback or private IPv4"
            )
        if not 1 <= timeout_seconds <= 900:
            raise ValueError("provider timeout is invalid")
        if maximum_output_tokens is not None and (
            isinstance(maximum_output_tokens, bool)
            or not isinstance(maximum_output_tokens, int)
            or maximum_output_tokens <= 0
        ):
            raise ValueError("maximum output tokens must be null or a positive integer")
        if not isinstance(thinking_tool_choice_auto, bool):
            raise TypeError("thinking tool-choice mode must be boolean")
        if not isinstance(action_tag_compatibility, bool):
            raise TypeError("action-tag compatibility mode must be boolean")
        if action_tag_compatibility and not thinking_tool_choice_auto:
            raise ValueError("action-tag compatibility requires tool_choice=auto")
        if provider_api not in {"chat_completions", "responses", "anthropic_messages"}:
            raise ValueError("provider API is invalid")
        if not isinstance(require_api_key, bool):
            raise TypeError("require_api_key must be boolean")
        responses_generation = {
            "reasoning_effort": reasoning_effort,
            "reasoning_context": reasoning_context,
            "reasoning_mode": reasoning_mode,
            "reasoning_summary": reasoning_summary,
            "text_verbosity": text_verbosity,
        }
        responses_only_generation = {
            key: value
            for key, value in responses_generation.items()
            if key != "reasoning_effort"
        }
        if provider_api == "responses":
            if (
                reasoning_effort not in RESPONSES_REASONING_EFFORTS
                or reasoning_context not in RESPONSES_REASONING_CONTEXTS
                or reasoning_mode not in RESPONSES_REASONING_MODES
                or (
                    reasoning_summary is not None
                    and reasoning_summary not in RESPONSES_REASONING_SUMMARIES
                )
                or text_verbosity not in RESPONSES_TEXT_VERBOSITIES
            ):
                raise ValueError("Responses generation parameters are invalid")
            if thinking_tool_choice_auto or action_tag_compatibility:
                raise ValueError(
                    "Responses provider requires the forced Action20 tool choice"
                )
        elif provider_api == "anthropic_messages":
            if reasoning_effort not in ANTHROPIC_EFFORTS:
                raise ValueError("Anthropic Messages effort is invalid")
            if any(value is not None for value in responses_only_generation.values()):
                raise ValueError(
                    "Anthropic Messages does not accept GPT Responses generation fields"
                )
            if thinking_tool_choice_auto or action_tag_compatibility:
                raise ValueError(
                    "Anthropic Messages keeps protocol-side Action20 tool enforcement"
                )
        elif any(value is not None for value in responses_only_generation.values()):
            raise ValueError(
                "Responses generation parameters require provider_api=responses"
            )
        elif (
            reasoning_effort is not None
            and reasoning_effort not in CHAT_COMPLETIONS_REASONING_EFFORTS
        ):
            raise ValueError("Chat Completions reasoning effort is invalid")
        elif reasoning_effort is not None and (
            thinking_tool_choice_auto or action_tag_compatibility
        ):
            raise ValueError(
                "GLM reasoning requires the frozen Action20 tool choice"
            )
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= float(temperature) <= 2
            or isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not 0 < float(top_p) <= 1
            or (
                top_k is not None
                and (
                    isinstance(top_k, bool)
                    or not isinstance(top_k, int)
                    or not 0 <= top_k <= 100
                )
            )
            or (
                repetition_penalty is not None
                and (
                    isinstance(repetition_penalty, bool)
                    or not isinstance(repetition_penalty, (int, float))
                    or not 0 < float(repetition_penalty) <= 2
                )
            )
            or not isinstance(preserve_thinking, bool)
        ):
            raise ValueError("provider generation parameters are invalid")
        self.model = model.strip()
        self.wire_model = self.model if wire_model is None else wire_model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        if api_key_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError("API key environment variable name is invalid")
        self.api_key_env = api_key_env
        self.api_key_file = None if api_key_file is None else Path(api_key_file).resolve()
        self.maximum_output_tokens = maximum_output_tokens
        self.thinking_tool_choice_auto = thinking_tool_choice_auto
        self.action_tag_compatibility = action_tag_compatibility
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = top_k
        self.repetition_penalty = (
            None if repetition_penalty is None else float(repetition_penalty)
        )
        self.preserve_thinking = preserve_thinking
        self.provider_api = provider_api
        self.responses_generation = responses_generation
        self.chat_reasoning_effort = (
            reasoning_effort if provider_api == "chat_completions" else None
        )
        self.anthropic_effort = (
            reasoning_effort if provider_api == "anthropic_messages" else None
        )
        self.requires_api_key = require_api_key or not (loopback_http or private_http)
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoProviderRedirect()
        )
        self._pending_responses_replay: dict[str, Mapping[str, Any]] = {}
        self._anthropic_replay: dict[str, list[Any]] = {}

    def capability_preflight(
        self,
        *,
        prompt_revision: Any,
        physical_history_max: int,
        artifact: Path,
        maximum_action_decisions: int = DEFAULT_MAXIMUM_ACTION_DECISIONS,
        maximum_provider_attempts: int = (
            MAXIMUM_RESPONSES_PREFLIGHT_PROVIDER_ATTEMPTS
        ),
        provider_retry_policy: ProviderRetryPolicy | None = None,
        tool_schema: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        revision = load_common_system_prompt_revision(
            ACTION20_CORE_INSTRUCTIONS_PATH,
            physical_history_max=physical_history_max,
            maximum_action_decisions=maximum_action_decisions,
        )
        if revision.text != prompt_revision.text:
            raise ModelClientError("Prompt changed during provider preflight")
        preflight_provider_failures: list[dict[str, Any]] = []
        retry_policy = provider_retry_policy or ProviderRetryPolicy()

        def retry_provider_preflight_post(
            request: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            retry_started_at = time.monotonic()
            jitter_key = hashlib.sha256(
                _canonical_json_bytes(request)
            ).hexdigest()
            for attempt_index in range(
                1, maximum_provider_attempts + 1
            ):
                try:
                    return self._post(request)
                except ProviderRequestError as exc:
                    retry_elapsed_seconds = max(
                        0.0,
                        time.monotonic() - retry_started_at,
                    )
                    retry_delay_seconds = retry_policy.delay_seconds(
                        retry_index=attempt_index,
                        jitter_key=f"preflight:{self.model}:{jitter_key}",
                        retry_after_seconds=(
                            exc.retry_after_seconds
                            if retry_policy.enabled
                            else None
                        ),
                    )
                    retry_time_budget_exhausted = (
                        retry_policy.enabled
                        and retry_elapsed_seconds + retry_delay_seconds
                        > retry_policy.maximum_elapsed_seconds
                    )
                    retry_allowed = (
                        exc.retryable
                        and attempt_index
                        < maximum_provider_attempts
                        and not retry_time_budget_exhausted
                    )
                    preflight_provider_failures.append(
                        {
                            "attempt_index": attempt_index,
                            "maximum_attempts": maximum_provider_attempts,
                            "retry_allowed": retry_allowed,
                            "retry_delay_seconds": (
                                retry_delay_seconds if retry_allowed else 0.0
                            ),
                            "retry_elapsed_seconds": round(
                                retry_elapsed_seconds,
                                6,
                            ),
                            "retry_time_budget_exhausted": (
                                retry_time_budget_exhausted
                            ),
                            **exc.artifact_fields(),
                        }
                    )
                    if not retry_allowed:
                        raise
                    if retry_delay_seconds > 0.0:
                        time.sleep(retry_delay_seconds)
            raise AssertionError("provider preflight retry loop exhausted")

        preflight_post = retry_provider_preflight_post
        if self.provider_api == "chat_completions":
            preflight_response_index = 0

            def chat_preflight_post(request: Mapping[str, Any]) -> Mapping[str, Any]:
                nonlocal preflight_response_index
                raw = retry_provider_preflight_post(request)
                preflight_response_index += 1
                return self._adapt_gemini_response(
                    raw,
                    artifact.parent / "private" / "gemini_preflight"
                    / f"response_{preflight_response_index:04d}.json",
                )

            preflight_post = chat_preflight_post
        if self.provider_api == "responses":
            def responses_preflight_post(
                request: Mapping[str, Any],
            ) -> Mapping[str, Any]:
                canonical = {
                    field: request[field]
                    for field in (
                        "model",
                        "messages",
                        "tools",
                        "tool_choice",
                        "parallel_tool_calls",
                    )
                }
                wire = chat_request_to_responses(
                    canonical,
                    max_output_tokens=int(request["max_tokens"]),
                    generation=self.responses_generation,
                )
                raw = retry_provider_preflight_post(wire)
                normalized = responses_to_chat_completion(raw)
                return normalized

            preflight_post = responses_preflight_post
        elif self.provider_api == "anthropic_messages":
            def anthropic_preflight_post(
                request: Mapping[str, Any],
            ) -> Mapping[str, Any]:
                if self.maximum_output_tokens is None or self.anthropic_effort is None:
                    raise ModelClientError("Anthropic Messages preflight budget is incomplete")
                try:
                    wire = chat_request_to_anthropic(
                        request,
                        max_tokens=int(self.maximum_output_tokens),
                        effort=self.anthropic_effort,
                        replay_content_by_call_id=self._anthropic_replay,
                    )
                except AnthropicMessagesAdapterError as exc:
                    raise ModelClientError(
                        f"Anthropic request adaptation failed: {exc}"
                    ) from exc
                raw = retry_provider_preflight_post(wire)
                try:
                    normalized = anthropic_to_chat_completion(raw)
                    replay = extract_anthropic_replay_content(raw)
                except AnthropicMessagesAdapterError as exc:
                    raise ModelClientError(
                        f"Anthropic result adaptation failed: {exc}"
                    ) from exc
                self._remember_anthropic_replay(replay)
                return normalized

            preflight_post = anthropic_preflight_post
        report = dict(run_native_tool_preflight(
            post=preflight_post,
            model=self.model,
            prompt_revision=revision,
            thinking_tool_choice_auto=self.thinking_tool_choice_auto,
            tool_schema=tool_schema,
        ))
        report["provider_retry_policy"] = {
            "maximum_attempts_per_request": maximum_provider_attempts,
            "retryable_failures_only": True,
            **retry_policy.to_record(),
        }
        report["provider_failures"] = len(preflight_provider_failures)
        report["provider_retries"] = sum(
            1
            for failure in preflight_provider_failures
            if failure["retry_allowed"]
        )
        report["provider_failure_attempts"] = preflight_provider_failures
        _write_json(artifact, report)
        return report

    def complete(
        self,
        request: Mapping[str, Any],
        *,
        call_dir: Path,
        seed: int,
    ) -> Mapping[str, Any]:
        call_dir.mkdir(parents=True, exist_ok=False)
        if self.provider_api == "responses":
            historical_arguments = _historical_chat_call_arguments(request)
            replay_items, accepted_canonical_arguments = (
                self._load_responses_replay_items(
                    list(historical_arguments),
                    output_dir=call_dir.parent.parent,
                    canonical_arguments_by_call_id=historical_arguments,
                )
            )
            try:
                provider_request = chat_request_to_responses(
                    request,
                    max_output_tokens=self.maximum_output_tokens,
                    replay_items_by_call_id=replay_items,
                    accepted_canonical_arguments_by_call_id=(
                        accepted_canonical_arguments
                    ),
                    generation=self.responses_generation,
                )
            except ResponsesAdapterError as exc:
                raise ModelClientError(
                    f"Responses request adaptation failed: {exc}"
                ) from exc
        elif self.provider_api == "anthropic_messages":
            if self.maximum_output_tokens is None or self.anthropic_effort is None:
                raise ModelClientError("Anthropic Messages output budget is incomplete")
            try:
                provider_request = chat_request_to_anthropic(
                    request,
                    max_tokens=int(self.maximum_output_tokens),
                    effort=self.anthropic_effort,
                    replay_content_by_call_id=self._anthropic_replay,
                )
            except AnthropicMessagesAdapterError as exc:
                raise ModelClientError(
                    f"Anthropic request adaptation failed: {exc}"
                ) from exc
        else:
            provider_request = {
                **request,
                "temperature": self.temperature,
                "seed": seed,
                "stream": False,
            }
            provider_request.pop('max_tokens', None)
            provider_request.pop('max_completion_tokens', None)
            if self.maximum_output_tokens is not None:
                provider_request['max_tokens'] = self.maximum_output_tokens
            if self.temperature > 0:
                provider_request["top_p"] = self.top_p
                if self.top_k is not None:
                    provider_request["top_k"] = self.top_k
                if self.repetition_penalty is not None:
                    provider_request["repetition_penalty"] = self.repetition_penalty
            if self.chat_reasoning_effort is not None:
                provider_request.update(
                    {
                        "reasoning_effort": self.chat_reasoning_effort,
                        "thinking": {
                            "type": "enabled",
                            "clear_thinking": False,
                        },
                        "chat_template_kwargs": {
                            "reasoning_effort": self.chat_reasoning_effort,
                            "clear_thinking": False,
                        },
                    }
                )
            elif self.thinking_tool_choice_auto:
                provider_request.update(
                    {
                        "tool_choice": "auto",
                        "enable_thinking": True,
                    }
                )
                provider_request.pop("chat_template_kwargs", None)
            else:
                provider_request["chat_template_kwargs"] = {"enable_thinking": True}
            if self.preserve_thinking and self.chat_reasoning_effort is None:
                provider_request["preserve_thinking"] = True
            if _gemini_requires_function_response_name(self.wire_model, self.base_url):
                provider_request = _with_function_response_names(provider_request)
        _write_json(
            call_dir / "request_manifest.json",
            _redact_data_urls(provider_request),
        )
        provider_started_at = time.monotonic()
        try:
            response = self._post(provider_request)
        except ProviderRequestError as exc:
            _write_json(
                call_dir / "provider_metrics.json",
                _provider_metrics(None, time.monotonic() - provider_started_at),
            )
            request_bytes = _canonical_json_bytes(provider_request)
            exc.request_body_sha256 = hashlib.sha256(request_bytes).hexdigest()
            exc.request_body_path = None
            raise
        _write_json(
            call_dir / "provider_metrics.json",
            _provider_metrics(response, time.monotonic() - provider_started_at),
        )
        if self.provider_api == "responses":
            try:
                replay_bundle = extract_responses_replay_items(response)
                normalized = responses_to_chat_completion(response)
            except ResponsesAdapterError as exc:
                _write_private_json(
                    call_dir / "private" / "responses_response.json",
                    response,
                )
                raise ModelClientError(
                    f"Responses result adaptation failed: {exc}"
                ) from exc
            _write_private_json(
                call_dir / "private" / "responses_response.json",
                response,
            )
            _write_json(call_dir / "response.json", normalized)
            call_id = normalized["choices"][0]["message"]["tool_calls"][0]["id"]
            self._pending_responses_replay[call_id] = {
                "schema_version": RESPONSES_REPLAY_VERSION,
                "adapter_version": RESPONSES_ADAPTER_VERSION,
                "call_id": call_id,
                "response_id": response.get("id"),
                "model": response.get("model"),
                "output_items": replay_bundle,
            }
            return normalized
        if self.provider_api == "anthropic_messages":
            try:
                replay = extract_anthropic_replay_content(response)
                normalized = anthropic_to_chat_completion(response)
            except AnthropicMessagesAdapterError as exc:
                _write_private_json(
                    call_dir / "private" / "anthropic_response.json",
                    _redact_data_urls(response),
                )
                raise ModelClientError(
                    f"Anthropic result adaptation failed: {exc}"
                ) from exc
            _write_private_json(
                call_dir / "private" / "anthropic_response.json",
                _redact_data_urls(response),
            )
            _write_json(call_dir / "response.json", _redact_data_urls(normalized))
            self._remember_anthropic_replay(replay)
            return normalized
        response = self._adapt_gemini_response(
            response, call_dir / "private" / "gemini_response.json"
        )
        _write_json(call_dir / "response.json", response)
        return response

    def _adapt_gemini_response(
        self, response: Mapping[str, Any], raw_path: Path
    ) -> Mapping[str, Any]:
        """Archive accompanying prose; only native tool_calls supply actions."""
        if not _gemini_requires_function_response_name(self.wire_model, self.base_url):
            return response
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            return response
        message = choices[0].get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            return response
        calls = message.get("tool_calls")
        if not message["content"] or not isinstance(calls, list) or len(calls) != 1:
            return response
        _write_private_json(raw_path, response)
        # Do not relabel prose as native reasoning or change tool arguments.
        return {**response, "choices": [{**choices[0], "message": {**message, "content": None}}]}

    def _remember_anthropic_replay(self, content: Sequence[Any]) -> None:
        replay = json.loads(json.dumps(list(content)))
        for block in replay:
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                call_id = block.get("id")
                if isinstance(call_id, str) and call_id:
                    self._anthropic_replay[call_id] = replay
                    return

    def accept_response(
        self,
        response: Mapping[str, Any],
        *,
        output_dir: Path,
        canonical_function_arguments: Mapping[str, Any] | None = None,
    ) -> None:
        """Commit only a protocol-accepted Responses bundle for future replay."""

        if self.provider_api != "responses":
            return
        call_id = _normalized_chat_call_id(response)
        record = self._pending_responses_replay.pop(call_id, None)
        if record is None:
            raise ModelClientError("accepted Responses call has no pending replay bundle")
        output_items = record.get("output_items")
        if not isinstance(output_items, list):
            raise ModelClientError("pending Responses replay bundle changed")
        call = next(
            (
                item
                for item in output_items
                if isinstance(item, Mapping)
                and item.get("type") == "function_call"
            ),
            None,
        )
        provider_arguments = call.get("arguments") if call is not None else None
        if not isinstance(provider_arguments, str):
            raise ModelClientError("pending Responses function arguments changed")
        if canonical_function_arguments is None:
            canonical_arguments = provider_arguments
        else:
            canonical_arguments = json.dumps(
                dict(canonical_function_arguments),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        try:
            replay_exact = function_arguments_equivalent(
                provider_arguments,
                canonical_arguments,
            )
        except ResponsesAdapterError as exc:
            raise ModelClientError(
                f"accepted Responses arguments are invalid: {exc}"
            ) from exc
        persisted = {
            **record,
            "replay_mode": (
                "opaque_replay"
                if replay_exact
                else "opaque_replay_accepted_normalization"
            ),
            "canonical_arguments_sha256": hashlib.sha256(
                canonical_arguments.encode("utf-8")
            ).hexdigest(),
            "output_items": output_items,
        }
        path = _responses_replay_path(Path(output_dir), call_id)
        if path.exists():
            raise ModelClientError("Responses replay call ID was reused")
        _write_private_json(path, persisted)

    def _load_responses_replay_items(
        self,
        call_ids: list[str],
        *,
        output_dir: Path,
        canonical_arguments_by_call_id: Mapping[str, str] | None = None,
    ) -> tuple[
        dict[str, list[dict[str, Any]]],
        dict[str, str],
    ]:
        replay: dict[str, list[dict[str, Any]]] = {}
        accepted_canonical_arguments: dict[str, str] = {}
        canonical_arguments_by_call_id = canonical_arguments_by_call_id or {}
        for call_id in call_ids:
            path = _responses_replay_path(Path(output_dir), call_id)
            record = _read_private_json(path, max_bytes=MAX_PROVIDER_RESPONSE_BYTES)
            if (
                set(record)
                != {
                    "schema_version",
                    "adapter_version",
                    "call_id",
                    "response_id",
                    "model",
                    "replay_mode",
                    "canonical_arguments_sha256",
                    "output_items",
                }
                or record.get("schema_version") != RESPONSES_REPLAY_VERSION
                or record.get("adapter_version")
                not in RESPONSES_REPLAY_COMPATIBLE_ADAPTER_VERSIONS
                or record.get("call_id") != call_id
                or record.get("replay_mode")
                not in {
                    "opaque_replay",
                    "opaque_replay_accepted_normalization",
                    "canonical_reconstruction",
                }
                or not isinstance(record.get("canonical_arguments_sha256"), str)
                or len(record["canonical_arguments_sha256"]) != 64
            ):
                raise ModelClientError("Responses replay artifact contract changed")
            if record["replay_mode"] == "canonical_reconstruction":
                if record.get("output_items") is not None:
                    raise ModelClientError(
                        "canonical Responses reconstruction retained opaque output"
                    )
                continue
            if not isinstance(record.get("output_items"), list):
                raise ModelClientError("opaque Responses replay output changed")
            if record["replay_mode"] == "opaque_replay_accepted_normalization":
                canonical_arguments = canonical_arguments_by_call_id.get(call_id)
                if (
                    not isinstance(canonical_arguments, str)
                    or hashlib.sha256(canonical_arguments.encode("utf-8")).hexdigest()
                    != record["canonical_arguments_sha256"]
                ):
                    raise ModelClientError(
                        "normalized Responses replay canonical history changed"
                    )
                accepted_canonical_arguments[call_id] = canonical_arguments
            replay[call_id] = extract_responses_replay_items(
                {
                    "object": "response",
                    "status": "completed",
                    "output": record["output_items"],
                }
            )
        return replay, accepted_canonical_arguments

    def _post(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if request.get("model") != self.model:
            raise ModelClientError("provider request model identity changed")
        wire_request = {**request, "model": self.wire_model}
        if (
            self.provider_api == "chat_completions"
            and _gemini_requires_function_response_name(self.wire_model, self.base_url)
        ):
            wire_request = _with_function_response_names(wire_request)
        encoded = _canonical_json_bytes(wire_request)
        headers = {"Content-Type": "application/json"}
        if self.provider_api == "anthropic_messages":
            if self.requires_api_key:
                headers["x-api-key"] = self._read_api_key()
            headers["anthropic-version"] = ANTHROPIC_VERSION
            endpoint = anthropic_messages_url(self.base_url)
        else:
            if self.requires_api_key:
                headers["Authorization"] = f"Bearer {self._read_api_key()}"
            endpoint = (
                f"{self.base_url}/"
                + ("responses" if self.provider_api == "responses" else "chat/completions")
            )
        http_request = urllib.request.Request(
            endpoint,
            data=encoded,
            headers=headers,
            method="POST",
        )
        redaction_values = (
            headers.get("x-api-key", ""),
            headers.get("Authorization", "").removeprefix("Bearer "),
        )
        try:
            with self.opener.open(http_request, timeout=self.timeout_seconds) as response:
                value = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            body = exc.read(MAX_PROVIDER_ERROR_BODY_BYTES + 1)
            truncated = len(body) > MAX_PROVIDER_ERROR_BODY_BYTES
            body = body[:MAX_PROVIDER_ERROR_BODY_BYTES]
            status_code = int(exc.code)
            retryable = (
                status_code in {408, 425, 429}
                or 500 <= status_code <= 599
            )
            retry_after_seconds = parse_retry_after_seconds(
                exc.headers.get("Retry-After")
                if exc.headers is not None
                else None
            )
            raise ProviderRequestError(
                f"provider HTTP request failed with status {status_code}",
                failure_type=f"provider_http_{status_code}",
                retryable=retryable,
                status_code=status_code,
                response_body=body,
                response_body_truncated=truncated,
                retry_after_seconds=retry_after_seconds,
                redaction_values=redaction_values,
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderRequestError(
                f"provider transport request failed: {_redact_provider_error_text(str(exc.reason), redaction_values)}",
                failure_type="provider_transport_error",
                retryable=True,
                redaction_values=redaction_values,
            ) from exc
        except (ConnectionError, http.client.HTTPException) as exc:
            raise ProviderRequestError(
                f"provider transport request failed: {_redact_provider_error_text(str(exc), redaction_values)}",
                failure_type="provider_transport_error",
                retryable=True,
                redaction_values=redaction_values,
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ProviderRequestError(
                f"provider request timed out: {_redact_provider_error_text(str(exc), redaction_values)}",
                failure_type="provider_timeout",
                retryable=True,
                redaction_values=redaction_values,
            ) from exc
        if len(value) > MAX_PROVIDER_RESPONSE_BYTES:
            raise ProviderRequestError(
                "provider response exceeds size limit",
                failure_type="provider_response_oversized",
                retryable=False,
                redaction_values=redaction_values,
            )
        if self.provider_api == "anthropic_messages" and wire_request.get("stream") is True:
            stripped = value.lstrip()
            if stripped.startswith(b"{") and b"event:" not in value[:512]:
                raise ProviderRequestError(
                    "provider ignored stream=true and returned JSON",
                    failure_type="provider_stream_not_sse",
                    retryable=False,
                    response_body=value[:MAX_PROVIDER_ERROR_BODY_BYTES],
                    response_body_truncated=(len(value) > MAX_PROVIDER_ERROR_BODY_BYTES),
                    redaction_values=redaction_values,
                )
            try:
                events = parse_sse_json_events(value)
                decoded = assemble_anthropic_sse(events)
            except AnthropicMessagesAdapterError as exc:
                raise ProviderRequestError(
                    f"provider Anthropic stream is invalid: {_redact_provider_error_text(str(exc), redaction_values)}",
                    failure_type="provider_invalid_json",
                    retryable=True,
                    response_body=value[:MAX_PROVIDER_ERROR_BODY_BYTES],
                    response_body_truncated=(len(value) > MAX_PROVIDER_ERROR_BODY_BYTES),
                    redaction_values=redaction_values,
                ) from exc
            if not isinstance(decoded, Mapping):
                raise ProviderRequestError(
                    "provider Anthropic stream is not a JSON object",
                    failure_type="provider_non_object_json",
                    retryable=True,
                    redaction_values=redaction_values,
                )
            return decoded
        try:
            decoded = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderRequestError(
                "provider response is not strict UTF-8 JSON",
                failure_type="provider_invalid_json",
                retryable=True,
                response_body=value[:MAX_PROVIDER_ERROR_BODY_BYTES],
                response_body_truncated=(len(value) > MAX_PROVIDER_ERROR_BODY_BYTES),
                redaction_values=redaction_values,
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ProviderRequestError(
                "provider response is not a JSON object",
                failure_type="provider_non_object_json",
                retryable=True,
                response_body=value[:MAX_PROVIDER_ERROR_BODY_BYTES],
                response_body_truncated=(len(value) > MAX_PROVIDER_ERROR_BODY_BYTES),
                redaction_values=redaction_values,
            )
        return decoded

    def _read_api_key(self) -> str:
        if self.api_key_env is not None:
            key = os.environ.get(self.api_key_env, "").strip()
        elif self.api_key_file is not None and self.api_key_file.is_file():
            mode = stat.S_IMODE(self.api_key_file.stat().st_mode)
            if mode & 0o077 or self.api_key_file.stat().st_size > 4096:
                raise ModelClientError("API key file permissions or size are invalid")
            key = self.api_key_file.read_text(encoding="utf-8").strip()
        elif self.provider_api == "anthropic_messages":
            key = (
                os.environ.get("ANTHROPIC_API_KEY", "").strip()
                or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
            )
        else:
            key = (
                os.environ.get("OPENAI_API_KEY", "").strip()
                or os.environ.get("DASHSCOPE_API_KEY", "").strip()
                or os.environ.get("TOKEN_PLAN_API_KEY", "").strip()
            )
        if not key:
            raise ModelClientError("provider API key is unavailable")
        if "\r" in key or "\n" in key or any(ord(character) > 255 for character in key):
            raise ModelClientError("provider API key contains invalid HTTP header characters")
        return key


def _provider_generation_record(provider: Any) -> Mapping[str, Any]:
    """Return the effective generation contract, including legacy test providers."""

    provider_api = str(getattr(provider, "provider_api", "chat_completions"))
    if provider_api == "responses":
        generation = getattr(provider, "responses_generation", {})
        return {
            "provider_api": "responses",
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": None,
            "preserve_thinking": False,
            "reasoning_effort": generation.get("reasoning_effort"),
            "reasoning_context": generation.get("reasoning_context"),
            "reasoning_mode": generation.get("reasoning_mode"),
            "reasoning_summary": generation.get("reasoning_summary"),
            "text_verbosity": generation.get("text_verbosity"),
            "encrypted_reasoning_replay": True,
            "store": False,
        }
    if provider_api == "anthropic_messages":
        return {
            "provider_api": "anthropic_messages",
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": None,
            "preserve_thinking": True,
            "reasoning_effort": getattr(provider, "anthropic_effort", None),
            "reasoning_context": None,
            "reasoning_mode": None,
            "reasoning_summary": None,
            "text_verbosity": None,
            "thinking": None,
            "thinking_budget": None,
            "output_config": {"effort": getattr(provider, "anthropic_effort", None)},
            "stream": True,
            "adapter_version": ANTHROPIC_ADAPTER_VERSION,
        }
    reasoning_effort = getattr(provider, "chat_reasoning_effort", None)
    record = {
        "temperature": float(getattr(provider, "temperature", 0.1)),
        "top_p": float(getattr(provider, "top_p", 0.9)),
        "top_k": getattr(provider, "top_k", None),
        "repetition_penalty": getattr(provider, "repetition_penalty", None),
        "preserve_thinking": bool(getattr(provider, "preserve_thinking", False)),
        "reasoning_effort": reasoning_effort,
        "reasoning_summary": None,
        "thinking_budget": None,
    }
    if reasoning_effort is not None:
        record["thinking"] = {"type": "enabled", "clear_thinking": False}
    return record


def _validate_repeat_inflight(
    value: Mapping[str, Any] | None,
    *,
    next_sequence: int,
) -> Mapping[str, Any] | None:
    return validate_repeat_inflight(
        value,
        next_sequence=next_sequence,
        error_type=ModelClientError,
    )


def _write_resume_checkpoint(
    *,
    output_dir: Path,
    scene: SceneClient,
    experiment_id: str,
    scene_episode_id: str,
    prompt_sha256: str,
    conversation: EpisodeConversation,
    counters: Mapping[str, int],
    resume_session_index: int,
    state: str,
    pause_status: str | None = None,
    terminal_protocol_failure: str | None = None,
    terminal_provider_failure: str | None = None,
    repeat_inflight: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    return write_resume_checkpoint(
        output_dir=output_dir,
        scene=scene,
        experiment_id=experiment_id,
        scene_episode_id=scene_episode_id,
        prompt_sha256=prompt_sha256,
        conversation=conversation,
        counters=counters,
        resume_session_index=resume_session_index,
        state=state,
        pause_status=pause_status,
        terminal_protocol_failure=terminal_protocol_failure,
        terminal_provider_failure=terminal_provider_failure,
        repeat_inflight=repeat_inflight,
        write_json=_write_json,
        hash_file=_sha256_file,
        error_type=ModelClientError,
        write_views=_write_conversation_views,
    )


def _load_normalization_stats(
    output_dir: Path,
    *,
    resume: bool,
) -> dict[str, int]:
    stats = {name: 0 for name in _NORMALIZATION_STAT_FIELDS}
    path = Path(output_dir) / "normalization_stats.json"
    if not resume or not path.is_file():
        return stats
    value = _read_strict_json(path, max_bytes=4096)
    if (
        value.get("schema_version") != NORMALIZATION_STATS_VERSION
        or set(value) != {"schema_version", *_NORMALIZATION_STAT_FIELDS}
    ):
        raise ModelClientError("normalization stats contract changed")
    for name in _NORMALIZATION_STAT_FIELDS:
        count = value[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ModelClientError("normalization stats counter is invalid")
        stats[name] = count
    return stats


def _write_normalization_stats(
    output_dir: Path,
    stats: Mapping[str, int],
) -> None:
    _write_json(
        Path(output_dir) / "normalization_stats.json",
        {"schema_version": NORMALIZATION_STATS_VERSION, **stats},
    )


def _load_resume_checkpoint(
    path: Path,
    *,
    allow_active_for_persistent_migration: bool = False,
) -> Mapping[str, Any]:
    return load_resume_checkpoint(
        path,
        allow_active_for_persistent_migration=(
            allow_active_for_persistent_migration
        ),
        read_json=_read_strict_json,
        error_type=ModelClientError,
    )


def _write_model_rolling_checkpoint(
    *,
    output_dir: Path,
    resume_checkpoint: Mapping[str, Any],
    scene_checkpoint: Mapping[str, Any],
) -> Path:
    return write_model_rolling_checkpoint(
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
        scene_checkpoint=scene_checkpoint,
        write_json=_write_json,
        error_type=ModelClientError,
    )


def _load_latest_model_rolling_checkpoint(
    *,
    output_dir: Path,
    resident_action_counts: list[int],
) -> Mapping[str, Any]:
    return load_latest_model_rolling_checkpoint(
        output_dir=output_dir,
        resident_action_counts=resident_action_counts,
        read_json=_read_strict_json,
        error_type=ModelClientError,
    )


def _load_latest_persistent_model_checkpoint(
    *,
    output_dir: Path,
    maximum_action_count: int,
) -> Mapping[str, Any]:
    return load_latest_persistent_model_checkpoint(
        output_dir=output_dir,
        maximum_action_count=maximum_action_count,
        read_json=_read_strict_json,
        error_type=ModelClientError,
    )


def _write_scene_recovery_event(
    *,
    output_dir: Path,
    recovery_index: int,
    phase: str,
    error: SceneRequestError,
    outcome: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    if phase not in {"begin_episode", "step"}:
        raise ValueError("scene recovery phase changed")
    record = {
        "schema_version": "behavior-hl-model-scene-recovery/v1",
        "recovery_index": recovery_index,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "error_type": error.error_type,
        "error_id": error.error_id,
        "error_origin": error.error_origin,
        "public_recovery": error.recovery,
        "outcome": outcome,
        "details": {} if details is None else dict(details),
        "private_error_detail_received": False,
    }
    _write_json(
        Path(output_dir)
        / "scene_recovery_events"
        / f"recovery_{recovery_index:04d}.json",
        record,
    )


def _automatic_scene_recovery_signature(
    *,
    error: SceneRequestError,
    step_fields: Mapping[str, Any],
    checkpoint_action_count: int,
) -> str | None:
    recovery = error.recovery
    fingerprint = (
        recovery.get("error_fingerprint")
        if isinstance(recovery, Mapping)
        else None
    )
    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
    ):
        return None
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "error_fingerprint": fingerprint,
                "error_origin": error.error_origin,
                "step": dict(step_fields),
                "checkpoint_action_count": checkpoint_action_count,
            }
        )
    ).hexdigest()


def _request_with_public_repair_notice(
    request: Mapping[str, Any],
    notice: str,
) -> dict[str, Any]:
    """Append one public-only retry hint without mutating conversation state."""

    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ModelClientError("native request messages changed")
    value = dict(request)
    value["messages"] = [
        *messages,
        {
            "role": "user",
            "content": [{"type": "text", "text": notice}],
        },
    ]
    return value


def _bare_marker_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Change only Marker guidance for fresh common-client runs, never CLI runs."""
    path = BENCH_TRY_ROOT / "action20_bare_marker_schema.json"
    guidance = _read_strict_json(path, max_bytes=8192)["marker_selection_guidance"]
    if not isinstance(guidance, str) or not guidance.strip():
        raise ValueError("bare Marker guidance must be nonempty text")
    result = json.loads(json.dumps(schema))
    description = result["function"]["description"]
    anchor = "Marker selection:\n"
    if description.count(anchor) != 1:
        raise ValueError("Marker description anchor changed")
    result["function"]["description"] = description.replace(anchor, anchor + guidance + "\n")
    return result


def _is_marker_selection_protocol_error(error: ValueError) -> bool:
    return str(error) == "target is not in the current same-action Marker set"


def _public_error_allows_same_process_recovery(error: SceneRequestError) -> bool:
    recovery = error.recovery
    return bool(
        error.error_type
        in {
            "RECOVERABLE_SCENE_ERROR",
            "SCENE_PROTOCOL_ERROR",
            "INTEGRATION_ERROR",
        }
        and isinstance(recovery, Mapping)
        and recovery.get("scene_resident") is True
        and recovery.get("session_valid") is True
    )


def _typed_scene_client_failure(error: ModelClientError) -> SceneRequestError:
    if isinstance(error, SceneRequestError):
        return error
    return SceneRequestError(
        error_type="SCENE_TRANSPORT_ERROR",
        message="SCENE_TRANSPORT_ERROR",
        error_id=None,
        recovery=None,
        error_origin="transport",
    )


def _scene_recovery_from_status(status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scene_resident": status.get("scene_resident") is True,
        "session_valid": status.get("session_valid") is True,
        "episode_active": status.get("episode_active") is True,
        "episode_id": status.get("episode_id"),
        "next_sequence": status.get("next_sequence"),
        "rolling_checkpoint_action_counts": list(
            status.get("rolling_checkpoint_action_counts", ())
        ),
    }


def _probe_scene_after_transport_failure(
    *,
    scene: SceneClient,
    error: ModelClientError,
) -> SceneRequestError:
    """Resolve an ambiguous local-socket failure without replaying an action."""

    typed = _typed_scene_client_failure(error)
    try:
        status = scene.request("status")
    except ModelClientError:
        return typed
    if not isinstance(status, Mapping):
        return typed
    recovery = _scene_recovery_from_status(status)
    return SceneRequestError(
        error_type=(
            "RECOVERABLE_SCENE_ERROR"
            if recovery["session_valid"]
            else "SESSION_INVALID"
        ),
        message=(
            "RECOVERABLE_SCENE_ERROR"
            if recovery["session_valid"]
            else "SESSION_INVALID"
        ),
        error_id=typed.error_id,
        recovery=recovery,
        error_origin=typed.error_origin,
    )


def _finish_pre_episode_scene_failure(
    *,
    output_dir: Path,
    provider: Any,
    error: SceneRequestError,
    recovery_count: int,
    maximum_action_decisions: int,
    maximum_model_calls: int,
    maximum_consecutive_blocked: int,
    physical_history_max: int,
    prompt_sha256: str,
) -> Mapping[str, Any]:
    reload_required = not _public_error_allows_same_process_recovery(error)
    summary = {
        "status": (
            "FAIL_SCENE_RELOAD_REQUIRED"
            if reload_required
            else "FAIL_SCENE_RECOVERY_EXHAUSTED"
        ),
        "model": provider.model,
        "model_calls": 0,
        "actions_sent_to_scene": 0,
        "macro_actions_resolved": 0,
        "scene_step_attempts": 0,
        "maximum_action_decisions": maximum_action_decisions,
        "maximum_model_calls": maximum_model_calls,
        "maximum_consecutive_blocked": maximum_consecutive_blocked,
        "consecutive_blocked_at_end": 0,
        "maximum_consecutive_blocked_observed": 0,
        "terminated_by_consecutive_blocked": False,
        "terminal_stall_reason": None,
        "physical_history_max": physical_history_max,
        "model_observation_max": physical_history_max + 1,
        "maximum_output_tokens": provider.maximum_output_tokens,
        "generation": dict(_provider_generation_record(provider)),
        "prompt_sha256": prompt_sha256,
        "automatic_scene_recoveries": recovery_count,
        "marker_selection_retries": 0,
        "decision_only_recoveries": 0,
        "checkpoint_restores": 0,
        "same_scene_restarts": recovery_count,
        "maximum_automatic_decision_repairs_per_sequence": (
            MAXIMUM_AUTOMATIC_DECISION_REPAIRS_PER_SEQUENCE
        ),
        "maximum_automatic_checkpoint_restores": (
            MAXIMUM_AUTOMATIC_CHECKPOINT_RESTORES
        ),
        "maximum_automatic_same_scene_resets": (
            MAXIMUM_AUTOMATIC_SAME_SCENE_RESETS
        ),
        "rolled_back_completed_actions": 0,
        "scene_reload_required": reload_required,
        "terminal_client_error_type": None,
        "terminal_scene_error_type": error.error_type,
        "terminal_scene_error_id": error.error_id,
        "terminal_scene_error_origin": error.error_origin,
        "resume_available": False,
        "resident_episode_preserved": False,
        "private_score_received_from_scene": False,
        "provider_metrics": _aggregate_provider_metrics(Path(output_dir), 0),
    }
    _write_json(Path(output_dir) / "summary.json", summary)
    _write_json(
        Path(output_dir) / "conversation_unavailable.json",
        {
            "reason": "initial_scene_episode_not_established",
            "model_visible_observation_received": False,
            "private_error_detail_received": False,
        },
    )
    return summary


def _finish_pre_loop_client_failure(
    *,
    output_dir: Path,
    scene: Any,
    provider: Any,
    error: Exception,
    maximum_action_decisions: int,
    maximum_model_calls: int,
    maximum_consecutive_blocked: int,
    physical_history_max: int,
    prompt_sha256: str,
) -> Mapping[str, Any]:
    """Close a begun episode and always classify pre-loop client failures.

    This boundary covers first-observation materialization, image publication,
    conversation construction, and initial checkpoint persistence.  None of
    those failures imply that the resident simulator itself is corrupt.
    """

    cleanup_valid = False
    cleanup_error_type: str | None = None
    try:
        scene.request("end_episode", reason="client_error")
        status = scene.request("status")
        cleanup_valid = bool(
            isinstance(status, Mapping)
            and status.get("scene_resident") is True
            and status.get("session_valid") is True
            and status.get("episode_active") is False
        )
    except Exception as cleanup_error:
        cleanup_error_type = type(cleanup_error).__name__
    summary = {
        "status": (
            "FAIL_CLIENT_INITIALIZATION"
            if cleanup_valid
            else "FAIL_SCENE_RELOAD_REQUIRED"
        ),
        "model": provider.model,
        "model_calls": 0,
        "actions_sent_to_scene": 0,
        "macro_actions_resolved": 0,
        "scene_step_attempts": 0,
        "maximum_action_decisions": maximum_action_decisions,
        "maximum_model_calls": maximum_model_calls,
        "maximum_consecutive_blocked": maximum_consecutive_blocked,
        "consecutive_blocked_at_end": 0,
        "maximum_consecutive_blocked_observed": 0,
        "terminated_by_consecutive_blocked": False,
        "terminal_stall_reason": None,
        "physical_history_max": physical_history_max,
        "model_observation_max": physical_history_max + 1,
        "maximum_output_tokens": provider.maximum_output_tokens,
        "prompt_sha256": prompt_sha256,
        "automatic_scene_recoveries": 0,
        "marker_selection_retries": 0,
        "decision_only_recoveries": 0,
        "checkpoint_restores": 0,
        "same_scene_restarts": 0,
        "rolled_back_completed_actions": 0,
        "scene_reload_required": not cleanup_valid,
        "terminal_client_error_type": type(error).__name__,
        "terminal_scene_error_type": None,
        "terminal_scene_error_id": None,
        "terminal_scene_error_origin": None,
        "resume_available": False,
        "resident_episode_preserved": False,
        "private_score_received_from_scene": False,
        "provider_metrics": _aggregate_provider_metrics(Path(output_dir), 0),
    }
    _write_json(Path(output_dir) / "summary.json", summary)
    _write_json(
        Path(output_dir) / "conversation_unavailable.json",
        {
            "reason": "post_begin_client_initialization_failed",
            "model_visible_observation_received": False,
            "private_error_detail_received": False,
        },
    )
    _write_json(
        Path(output_dir) / "private" / "pre_loop_client_failure.json",
        {
            "error_type": type(error).__name__,
            "cleanup_valid": cleanup_valid,
            "cleanup_error_type": cleanup_error_type,
        },
    )
    return summary


def run_model_episode(
    *,
    scene: SceneClient,
    provider: OpenAICompatibleProvider,
    experiment_id: str,
    instruction: str,
    task_id: int = 25,
    task_name: str = "clearing_food_from_table_into_fridge",
    task_split: str = "train",
    task_instance_id: str = "0",
    task_seed: int = 19506,
    instruction_provenance: Mapping[str, Any] | None = None,
    maximum_model_calls: int,
    physical_history_max: int,
    output_dir: Path,
    maximum_action_decisions: int = DEFAULT_MAXIMUM_ACTION_DECISIONS,
    maximum_provider_attempts_per_protocol_attempt: int = (
        DEFAULT_MAXIMUM_PROVIDER_ATTEMPTS_PER_PROTOCOL_ATTEMPT
    ),
    provider_retry_policy: Mapping[str, Any] | None = None,
    maximum_consecutive_blocked: int = DEFAULT_MAXIMUM_CONSECUTIVE_BLOCKED,
    pause_on_client_failure: bool = False,
    resume_checkpoint: Mapping[str, Any] | None = None,
    persistent_resume: Mapping[str, Any] | None = None,
    campaign_binding: Mapping[str, Any] | None = None,
    marker_depth_protocol: str | None = None,
    overhead_view_mode: str = "off",
    automatic_interaction_approach_enabled: bool = False,
) -> Mapping[str, Any]:
    if persistent_resume is not None and resume_checkpoint is None:
        raise ValueError("persistent resume requires one paused model checkpoint")
    if marker_depth_protocol not in {None, RGBD_PROTOCOL_VERSION}:
        raise ValueError("marker depth protocol changed")
    if overhead_view_mode not in OVERHEAD_VIEW_MODES:
        raise ValueError("overhead view mode changed")
    if not isinstance(automatic_interaction_approach_enabled, bool):
        raise TypeError("automatic interaction approach flag must be boolean")
    episode_configuration = {
        **(
            {}
            if marker_depth_protocol is None
            else {"marker_depth_protocol": marker_depth_protocol}
        ),
        **(
            {}
            if overhead_view_mode == "off"
            else {"overhead_view_mode": overhead_view_mode}
        ),
        **(
            {"automatic_interaction_approach_enabled": True}
            if automatic_interaction_approach_enabled
            else {}
        ),
    }
    if physical_history_max not in PHYSICAL_HISTORY_OPTIONS:
        raise ValueError("physical history must be exactly 5 or 20")
    if (
        isinstance(maximum_action_decisions, bool)
        or not isinstance(maximum_action_decisions, int)
        or not 1 <= maximum_action_decisions <= MAXIMUM_ACTION_DECISIONS
    ):
        raise ValueError("maximum action decisions must be in [1, 150]")
    if (
        isinstance(maximum_model_calls, bool)
        or not isinstance(maximum_model_calls, int)
        or not 1 <= maximum_model_calls <= MAXIMUM_MODEL_CALLS
    ):
        raise ValueError("maximum model calls must be in [1, 30000]")
    if (
        isinstance(maximum_consecutive_blocked, bool)
        or not isinstance(maximum_consecutive_blocked, int)
        or maximum_consecutive_blocked <= 0
    ):
        raise ValueError("maximum consecutive blocked must be a positive integer")
    if (
        isinstance(maximum_provider_attempts_per_protocol_attempt, bool)
        or not isinstance(maximum_provider_attempts_per_protocol_attempt, int)
        or not 1 <= maximum_provider_attempts_per_protocol_attempt <= 10
    ):
        raise ValueError("maximum provider attempts must be in [1, 10]")
    try:
        retry_policy = ProviderRetryPolicy.from_mapping(provider_retry_policy)
    except (TypeError, ValueError) as exc:
        raise ValueError("provider retry policy is invalid") from exc
    thinking_tool_choice_auto = getattr(
        provider,
        "thinking_tool_choice_auto",
        False,
    )
    if not isinstance(thinking_tool_choice_auto, bool):
        raise TypeError("provider thinking tool-choice mode must be boolean")
    action_tag_compatibility = getattr(
        provider,
        "action_tag_compatibility",
        False,
    )
    if not isinstance(action_tag_compatibility, bool):
        raise TypeError("provider action-tag compatibility mode must be boolean")
    provider_generation = _provider_generation_record(provider)
    prompt = (
        load_common_system_prompt_revision(
            ACTION20_CORE_INSTRUCTIONS_PATH,
            physical_history_max=physical_history_max,
            maximum_action_decisions=maximum_action_decisions,
        )
        if resume_checkpoint is None
        else load_prompt_revision(
            Path(output_dir) / "system_prompt.txt",
            physical_history_max=physical_history_max,
            maximum_action_decisions=maximum_action_decisions,
        )
    )
    if resume_checkpoint is not None:
        prompt = SystemPromptRevision(
            text=prompt.text,
            source_path=ACTION20_CORE_INSTRUCTIONS_PATH,
        )
    prompt_sha256 = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
    source_tool_schema_path = (
        ACTION20_TOOL_SCHEMA_PATH
        if resume_checkpoint is None
        else Path(output_dir) / "action20_tool_schema.json"
    )
    tool_schema = load_action20_tool_schema(source_tool_schema_path)
    if resume_checkpoint is None:
        tool_schema = _bare_marker_schema(tool_schema)
    if resume_checkpoint is None and overhead_view_mode != "off":
        tool_schema = with_overhead_view_tool_schema(
            tool_schema,
            overhead_view_mode,
        )
    if resume_checkpoint is None and automatic_interaction_approach_enabled:
        tool_schema = with_automatic_interaction_approach_tool_schema(
            tool_schema
        )
    if resume_checkpoint is None:
        _write_json(Path(output_dir) / "action20_tool_schema.json", tool_schema)
    blocked_categories_enabled = (
        "blocked_category"
        in tool_schema["function"].get("description", "")
    )
    blocked_details_enabled = "blocked_detail" in tool_schema["function"].get("description", "")
    provider_api = getattr(provider, "provider_api", "chat_completions")
    client_manifest_record = {
        "prompt_provenance": prompt_provenance_record(
            prompt, tool_schema, instruction=instruction,
            maximum_action_decisions=maximum_action_decisions,
            harness_adapter="common_tool_calling_v1",
            overhead_view_mode=overhead_view_mode,
            automatic_interaction_approach_enabled=automatic_interaction_approach_enabled,
            physical_history_max=physical_history_max,
        ),
        "schema_version": (
            "behavior-hl-bench-try-model-client/"
            + (
                "v9-responses-macro-action-budget"
                if provider_api == "responses"
                else "v8-macro-action-budget"
            )
        ),
        "experiment_id": experiment_id,
        "task_id": task_id,
        "task_name": task_name,
        "task_split": task_split,
        "task_instance_id": task_instance_id,
        "task_seed": task_seed,
        "instruction_sha256": hashlib.sha256(
            instruction.encode("utf-8")
        ).hexdigest(),
        "instruction_provenance": dict(
            instruction_provenance
            if instruction_provenance is not None
            else {"kind": "task_config"}
        ),
        "model": provider.model,
        "scene_protocol_version": SCENE_PROTOCOL_VERSION,
        "marker_depth_protocol": marker_depth_protocol,
        "overhead_view_mode": overhead_view_mode,
        "automatic_interaction_approach_enabled": (
            automatic_interaction_approach_enabled
        ),
        "image_transport": dict(getattr(
            provider, "image_transport", {"enabled": False}
        )),
        "maximum_action_decisions": maximum_action_decisions,
        "maximum_model_calls": maximum_model_calls,
        "budget_semantics": {
            "public_counter": "macro_actions_resolved",
            "marker_micro_steps_count_separately": False,
            "provider_attempts_count_toward_public_budget": False,
            "protocol_attempts_count_toward_public_budget": False,
            "model_calls_are_internal_safety_limited": True,
        },
        "maximum_consecutive_blocked": maximum_consecutive_blocked,
        "physical_history_max": physical_history_max,
        "model_observation_max": physical_history_max + 1,
        "maximum_output_tokens": provider.maximum_output_tokens,
        "generation": dict(provider_generation),
        "thinking_tool_choice_auto": thinking_tool_choice_auto,
        "action_tag_compatibility": action_tag_compatibility,
        "pause_on_client_failure": pause_on_client_failure,
        "maximum_protocol_attempts_per_decision": (
            MAXIMUM_PROTOCOL_ATTEMPTS_PER_DECISION
        ),
        "maximum_provider_attempts_per_protocol_attempt": (
            maximum_provider_attempts_per_protocol_attempt
        ),
        "provider_retry_policy": retry_policy.to_record(),
        "maximum_automatic_provider_retry_windows": (
            MAXIMUM_AUTOMATIC_PROVIDER_RETRY_WINDOWS
        ),
        "automatic_provider_retry_window_delay_seconds": (
            AUTOMATIC_PROVIDER_RETRY_WINDOW_DELAY_SECONDS
        ),
        "maximum_automatic_decision_repairs_per_sequence": (
            MAXIMUM_AUTOMATIC_DECISION_REPAIRS_PER_SEQUENCE
        ),
        "maximum_automatic_checkpoint_restores": (
            MAXIMUM_AUTOMATIC_CHECKPOINT_RESTORES
        ),
        "maximum_automatic_same_scene_resets": (
            MAXIMUM_AUTOMATIC_SAME_SCENE_RESETS
        ),
        "protocol_attempt_seeds": list(PROTOCOL_ATTEMPT_SEEDS),
        "protocol_attempts_count_toward_model_call_budget": True,
        "provider_attempts_count_toward_model_call_budget": True,
        "source_sha256": {
            str(BENCH_TRY_ROOT / "action20_bare_marker_schema.json"): _sha256_file(
                BENCH_TRY_ROOT / "action20_bare_marker_schema.json"
            ),
            **{str(path): _sha256_file(path) for path in _AGENT_SOURCE_PATHS},
            str(BENCH_TRY_ROOT / "action20_protocol.py"): _sha256_file(BENCH_TRY_ROOT / "action20_protocol.py"),
            str(ACTION20_TOOL_SCHEMA_PATH): _sha256_file(
                ACTION20_TOOL_SCHEMA_PATH
            ),
            str(ACTION20_CORE_INSTRUCTIONS_PATH): (
                prompt.source_sha256 if resume_checkpoint is None
                else _sha256_file(ACTION20_CORE_INSTRUCTIONS_PATH)
            ),
            str(Path(__file__).resolve()): _sha256_file(Path(__file__).resolve()),
            str(Path(__file__).resolve().parent / "native_tool_preflight.py"): _sha256_file(Path(__file__).resolve().parent / "native_tool_preflight.py"),
        },
    }
    if provider_api == "responses":
        client_manifest_record["provider_api"] = "responses"
        client_manifest_record["source_sha256"][
            str(BENCH_TRY_ROOT / "responses_adapter.py")
        ] = _sha256_file(BENCH_TRY_ROOT / "responses_adapter.py")
        if client_manifest_record["image_transport"].get("enabled") is True:
            client_manifest_record["source_sha256"][
                str(BENCH_TRY_ROOT / "lab" / "responses_image_proxy.py")
            ] = _sha256_file(
                BENCH_TRY_ROOT / "lab" / "responses_image_proxy.py"
            )
    if campaign_binding is not None:
        client_manifest_record.update(dict(campaign_binding))
    client_manifest_path = Path(output_dir) / "client_manifest.json"
    if resume_checkpoint is None:
        _write_json(client_manifest_path, client_manifest_record)
        (output_dir / "system_prompt.txt").write_bytes(prompt.text.encode("utf-8"))
        try:
            live_status = scene.request("status")
        except ModelClientError as raw_exc:
            availability_error = (
                raw_exc
                if isinstance(raw_exc, SceneRequestError)
                else _probe_scene_after_transport_failure(
                    scene=scene,
                    error=raw_exc,
                )
            )
        else:
            recovery = _scene_recovery_from_status(live_status)
            availability_error = None
            if not (
                recovery["scene_resident"]
                and recovery["session_valid"]
                and not recovery["episode_active"]
            ):
                valid_resident = bool(
                    recovery["scene_resident"]
                    and recovery["session_valid"]
                )
                availability_error = SceneRequestError(
                    error_type=(
                        "SCENE_PROTOCOL_ERROR"
                        if valid_resident
                        else "SESSION_INVALID"
                    ),
                    message=(
                        "SCENE_PROTOCOL_ERROR"
                        if valid_resident
                        else "SESSION_INVALID"
                    ),
                    error_id=None,
                    recovery=recovery,
                    error_origin="resident",
                )
        if availability_error is not None:
            _write_scene_recovery_event(
                output_dir=output_dir,
                recovery_index=1,
                phase="begin_episode",
                error=availability_error,
                outcome="fresh_start_preflight_rejected",
            )
            return _finish_pre_episode_scene_failure(
                output_dir=output_dir,
                provider=provider,
                error=availability_error,
                recovery_count=0,
                maximum_action_decisions=maximum_action_decisions,
                maximum_model_calls=maximum_model_calls,
                maximum_consecutive_blocked=maximum_consecutive_blocked,
                physical_history_max=physical_history_max,
                prompt_sha256=prompt_sha256,
            )
        provider.capability_preflight(
            tool_schema=tool_schema,
            prompt_revision=prompt,
            physical_history_max=physical_history_max,
            artifact=output_dir / "provider_preflight.json",
            maximum_action_decisions=maximum_action_decisions,
            maximum_provider_attempts=(
                maximum_provider_attempts_per_protocol_attempt
            ),
            provider_retry_policy=retry_policy,
        )
    else:
        existing_manifest = _read_strict_json(
            client_manifest_path,
            max_bytes=MAX_RESUME_ARTIFACT_BYTES,
        )
        if "prompt_provenance" in existing_manifest:
            # The original source hash belongs to the frozen run. Resume reads
            # its rendered snapshot; verify rendered text, tools and conditions.
            client_manifest_record["prompt_provenance"]["source_sha256"] = (
                existing_manifest["prompt_provenance"]["source_sha256"]
            )
        else:
            # Legacy runs retain their existing checkpoint / Prompt checks.
            client_manifest_record.pop("prompt_provenance")
        if _resume_client_manifest_identity(
            existing_manifest
        ) != _resume_client_manifest_identity(client_manifest_record):
            raise ModelClientError("resume client manifest identity changed")
        preflight = _read_strict_json(
            output_dir / "provider_preflight.json",
            max_bytes=MAX_RESUME_ARTIFACT_BYTES,
        )
        if (
            preflight.get("status") != "PASS"
            or preflight.get("model") != provider.model
        ):
            raise ModelClientError("resume provider preflight identity changed")
    (
        automatic_provider_retry_windows,
        consecutive_provider_retry_windows,
    ) = _load_provider_retry_state(output_dir)
    contract = PrivateEpisodeContract(
        episode_id=experiment_id,
        task_id=str(task_id),
        split=task_split,
        instance_id=task_instance_id,
        seed=task_seed,
        model=provider.model,
        project_root=PROJECT_ROOT,
        instruction=instruction,
        physical_history_max=physical_history_max,
    )
    status = "FAIL_BUDGET_EXHAUSTED"
    marker_selection_retries = 0
    scene_recovery_count = 0
    decision_only_recoveries = 0
    decision_repairs_for_sequence = 0
    checkpoint_restores = 0
    same_scene_restarts = 0
    rolled_back_completed_actions = 0
    scene_step_attempts = 0
    recovery_branch_index = 0
    last_scene_recovery_signature: str | None = None
    consecutive_scene_recovery_failures = 0
    terminal_client_error_type: str | None = None
    terminal_scene_error_type: str | None = None
    terminal_scene_error_id: str | None = None
    terminal_scene_error_origin: str | None = None
    scene_reload_required = False
    repeat_inflight: Mapping[str, Any] | None = None
    normalization_stats = _load_normalization_stats(
        output_dir,
        resume=resume_checkpoint is not None,
    )
    _write_normalization_stats(output_dir, normalization_stats)
    if resume_checkpoint is None:
        calls = 0
        actions = 0
        macro_actions_resolved = 0
        invalid_model_responses = 0
        protocol_retries = 0
        decisions_recovered_after_retry = 0
        provider_failures = 0
        provider_retries = 0
        decisions_recovered_after_provider_retry = 0
        action_tag_compatibility_applied = 0
        resume_session_index = 0
        scene_episode_id = experiment_id
        while True:
            try:
                begin = scene.request(
                    "begin_episode",
                    episode_id=scene_episode_id,
                    **episode_configuration,
                )
                break
            except ModelClientError as raw_exc:
                exc = (
                    raw_exc
                    if isinstance(raw_exc, SceneRequestError)
                    else _probe_scene_after_transport_failure(
                        scene=scene,
                        error=raw_exc,
                    )
                )
                terminal_scene_error_type = exc.error_type
                terminal_scene_error_id = exc.error_id
                terminal_scene_error_origin = exc.error_origin
                if (
                    isinstance(exc.recovery, Mapping)
                    and exc.recovery.get("session_valid") is True
                    and exc.recovery.get("episode_active") is True
                    and exc.recovery.get("episode_id") == scene_episode_id
                    and exc.recovery.get("next_sequence") == 0
                ):
                    try:
                        scene.request("end_episode", reason="client_error")
                    except ModelClientError as close_error:
                        typed_close = _typed_scene_client_failure(close_error)
                        _write_scene_recovery_event(
                            output_dir=output_dir,
                            recovery_index=scene_recovery_count + 1,
                            phase="begin_episode",
                            error=typed_close,
                            outcome="ambiguous_begin_cleanup_failed",
                        )
                        return _finish_pre_episode_scene_failure(
                            output_dir=output_dir,
                            provider=provider,
                            error=typed_close,
                            recovery_count=scene_recovery_count,
                            maximum_action_decisions=maximum_action_decisions,
                            maximum_model_calls=maximum_model_calls,
                            maximum_consecutive_blocked=(
                                maximum_consecutive_blocked
                            ),
                            physical_history_max=physical_history_max,
                            prompt_sha256=prompt_sha256,
                        )
                    exc.recovery["episode_active"] = False
                    exc.recovery["episode_id"] = None
                    exc.recovery["next_sequence"] = 0
                if (
                    not _public_error_allows_same_process_recovery(exc)
                    or not isinstance(exc.recovery, Mapping)
                    or exc.recovery.get("episode_active") is True
                    or same_scene_restarts
                    >= MAXIMUM_AUTOMATIC_SAME_SCENE_RESETS
                ):
                    _write_scene_recovery_event(
                        output_dir=output_dir,
                        recovery_index=scene_recovery_count + 1,
                        phase="begin_episode",
                        error=exc,
                        outcome="reload_required_or_recovery_exhausted",
                    )
                    return _finish_pre_episode_scene_failure(
                        output_dir=output_dir,
                        provider=provider,
                        error=exc,
                        recovery_count=scene_recovery_count,
                        maximum_action_decisions=maximum_action_decisions,
                        maximum_model_calls=maximum_model_calls,
                        maximum_consecutive_blocked=maximum_consecutive_blocked,
                        physical_history_max=physical_history_max,
                        prompt_sha256=prompt_sha256,
                    )
                scene_recovery_count += 1
                same_scene_restarts += 1
                recovery_branch_index += 1
                next_episode_id = (
                    f"{experiment_id}_recovery_{scene_recovery_count:02d}"
                )
                _write_scene_recovery_event(
                    output_dir=output_dir,
                    recovery_index=scene_recovery_count,
                    phase="begin_episode",
                    error=exc,
                    outcome="same_scene_reset_retry",
                    details={"next_scene_episode_id": next_episode_id},
                )
                scene_episode_id = next_episode_id
        terminal_scene_error_type = None
        terminal_scene_error_id = None
        terminal_scene_error_origin = None
        active = True
        try:
            initial_observation = _materialize_observation(
                begin["observation"],
                output_dir=output_dir,
                scene_run_dir=scene.scene_run_dir,
            )
            initial_composite = _publish_action_after_view(
                initial_observation,
                output_dir=output_dir,
                step_index=0,
                action_name="initial",
                recovery_branch_index=recovery_branch_index,
            )
            if campaign_binding is not None:
                _write_campaign_initial_view_audit(
                    output_dir=output_dir,
                    composite_path=initial_composite,
                    observation=initial_observation,
                    campaign_binding=campaign_binding,
                )
            conversation = EpisodeConversation(
                contract,
                initial_observation,
                system_prompt_revision=prompt,
                tool_schema=tool_schema,
            )
        except Exception as exc:
            return _finish_pre_loop_client_failure(
                output_dir=output_dir,
                scene=scene,
                provider=provider,
                error=exc,
                maximum_action_decisions=maximum_action_decisions,
                maximum_model_calls=maximum_model_calls,
                maximum_consecutive_blocked=maximum_consecutive_blocked,
                physical_history_max=physical_history_max,
                prompt_sha256=prompt_sha256,
            )
    else:
        effective_resume_checkpoint = resume_checkpoint
        if persistent_resume is not None:
            paired_resume = persistent_resume.get("resume_checkpoint")
            scene_checkpoint = persistent_resume.get("scene_checkpoint")
            if (
                not isinstance(paired_resume, Mapping)
                or not isinstance(scene_checkpoint, Mapping)
                or paired_resume.get("state") != "ACTIVE"
                or not isinstance(
                    scene_checkpoint.get("persistent_checkpoint"),
                    Mapping,
                )
            ):
                raise ModelClientError(
                    "persistent model checkpoint contract changed"
                )
            effective_resume_checkpoint = paired_resume
        assert effective_resume_checkpoint is not None
        if (
            effective_resume_checkpoint.get("schema_version")
            not in {RESUME_CHECKPOINT_VERSION_V3, RESUME_CHECKPOINT_VERSION}
            or effective_resume_checkpoint.get("state")
            != ("ACTIVE" if persistent_resume is not None else "PAUSED")
            or effective_resume_checkpoint.get("experiment_id") != experiment_id
            or not isinstance(effective_resume_checkpoint.get("episode_id"), str)
            or not effective_resume_checkpoint.get("episode_id")
            or (
                persistent_resume is None
                and effective_resume_checkpoint.get("scene_run_dir")
                != str(Path(scene.scene_run_dir).resolve())
            )
            or not isinstance(
                effective_resume_checkpoint.get("counters"), Mapping
            )
        ):
            raise ModelClientError("resume checkpoint identity changed")
        counters = effective_resume_checkpoint["counters"]
        if set(counters) != set(_RESUME_COUNTER_FIELDS):
            raise ModelClientError("resume checkpoint counters changed")
        (
            calls,
            actions,
            macro_actions_resolved,
            invalid_model_responses,
            protocol_retries,
            decisions_recovered_after_retry,
            provider_failures,
            provider_retries,
            decisions_recovered_after_provider_retry,
            action_tag_compatibility_applied,
            marker_selection_retries,
            scene_step_attempts,
            scene_recovery_count,
            decision_only_recoveries,
            checkpoint_restores,
            same_scene_restarts,
            rolled_back_completed_actions,
        ) = tuple(counters[name] for name in _RESUME_COUNTER_FIELDS)
        scene_episode_id = str(effective_resume_checkpoint["episode_id"])
        scene_status = scene.request("status")
        if persistent_resume is None:
            if (
                scene_status.get("scene_resident") is not True
                or scene_status.get("session_valid") is not True
                or scene_status.get("episode_active") is not True
                or scene_status.get("episode_id") != scene_episode_id
                or scene_status.get("next_sequence") != actions
                or effective_resume_checkpoint.get("next_sequence") != actions
            ):
                raise ModelClientError(
                    "resident episode cannot be resumed exactly"
                )
        else:
            if (
                scene_status.get("scene_resident") is not True
                or scene_status.get("session_valid") is not True
                or scene_status.get("episode_active") is not False
                or scene_status.get("next_sequence") != 0
            ):
                raise ModelClientError(
                    "new resident is not clean for persistent resume"
                )
            persistent = persistent_resume["scene_checkpoint"][
                "persistent_checkpoint"
            ]
            new_session_index = resume_checkpoint["resume_session_index"] + 1
            scene_episode_id = _safe_component(
                f"{experiment_id}_migrated_{new_session_index:02d}"
            )
            restored = scene.request(
                "restore_persistent_checkpoint",
                manifest_path=persistent["manifest_path"],
                manifest_sha256=persistent["manifest_sha256"],
                episode_id=scene_episode_id,
            )
            if (
                restored.get("status") != "PASS"
                or restored.get("episode_id") != scene_episode_id
                or restored.get("next_sequence") != actions
                or not isinstance(restored.get("observation"), Mapping)
            ):
                raise ModelClientError(
                    "persistent scene restore response changed"
                )
            paused_actions = resume_checkpoint["counters"]["actions"]
            rolled_back_completed_actions += paused_actions - actions
            checkpoint_restores += 1
            _materialize_observation(
                restored["observation"],
                output_dir=output_dir,
                scene_run_dir=scene.scene_run_dir,
            )
            _write_json(
                Path(output_dir)
                / "scene_migrations"
                / f"migration_{new_session_index:04d}.json",
                {
                    "schema_version": (
                        "behavior-hl-model-scene-migration/v1"
                    ),
                    "source_scene_run_dir": effective_resume_checkpoint[
                        "scene_run_dir"
                    ],
                    "destination_scene_run_dir": str(
                        Path(scene.scene_run_dir).resolve()
                    ),
                    "source_episode_id": effective_resume_checkpoint[
                        "episode_id"
                    ],
                    "destination_episode_id": scene_episode_id,
                    "restored_action_count": actions,
                    "paused_action_count": paused_actions,
                    "rolled_back_action_count": paused_actions - actions,
                    "manifest_path": persistent["manifest_path"],
                    "manifest_sha256": persistent["manifest_sha256"],
                    "task_identity_revalidated_by_scene_server": True,
                    "official_verifier_revalidated_by_scene_server": True,
                },
            )
        conversation_record = effective_resume_checkpoint.get("conversation")
        if not isinstance(conversation_record, Mapping):
            raise ModelClientError("resume conversation is missing")
        conversation = EpisodeConversation.from_artifact_transcript(
            conversation_record,
            expected_contract=contract,
            system_prompt_revision=prompt,
            artifact_root=output_dir,
            tool_schema=tool_schema,
        )
        repeat_inflight = _validate_repeat_inflight(
            effective_resume_checkpoint.get("repeat_inflight"),
            next_sequence=actions,
        )
        if repeat_inflight is None:
            if conversation.state != "awaiting_assistant":
                raise ModelClientError("resume conversation state changed")
        else:
            pending = conversation.pending_navigation_repeat_dispatch.decision
            if (
                pending.action != repeat_inflight["action"]
                or pending.public_repeat
                != repeat_inflight["repeat_requested"]
            ):
                raise ModelClientError("resume navigation repeat changed")
        active = True
        resume_session_index = resume_checkpoint["resume_session_index"] + 1
    terminal_protocol_failure: str | None = None
    terminal_provider_failure: str | None = None
    terminated_by_official_success = False
    terminated_by_consecutive_blocked = False
    pending_recovery_notice: str | None = None
    def checkpoint_counters() -> dict[str, int]:
        return {
            "calls": calls,
            "actions": actions,
            "macro_actions_resolved": macro_actions_resolved,
            "invalid_model_responses": invalid_model_responses,
            "protocol_retries": protocol_retries,
            "decisions_recovered_after_retry": decisions_recovered_after_retry,
            "provider_failures": provider_failures,
            "provider_retries": provider_retries,
            "decisions_recovered_after_provider_retry": (
                decisions_recovered_after_provider_retry
            ),
            "action_tag_compatibility_applied": (
                action_tag_compatibility_applied
            ),
            "marker_selection_retries": marker_selection_retries,
            "scene_step_attempts": scene_step_attempts,
            "scene_recovery_count": scene_recovery_count,
            "decision_only_recoveries": decision_only_recoveries,
            "checkpoint_restores": checkpoint_restores,
            "same_scene_restarts": same_scene_restarts,
            "rolled_back_completed_actions": rolled_back_completed_actions,
        }

    try:
        _write_resume_checkpoint(
            output_dir=output_dir,
            scene=scene,
            experiment_id=experiment_id,
            scene_episode_id=scene_episode_id,
            prompt_sha256=prompt_sha256,
            conversation=conversation,
            counters=checkpoint_counters(),
            resume_session_index=resume_session_index,
            state="ACTIVE",
            repeat_inflight=repeat_inflight,
        )
    except Exception as exc:
        return _finish_pre_loop_client_failure(
            output_dir=output_dir,
            scene=scene,
            provider=provider,
            error=exc,
            maximum_action_decisions=maximum_action_decisions,
            maximum_model_calls=maximum_model_calls,
            maximum_consecutive_blocked=maximum_consecutive_blocked,
            physical_history_max=physical_history_max,
            prompt_sha256=prompt_sha256,
        )
    try:
        while (
            macro_actions_resolved < maximum_action_decisions
            and (repeat_inflight is not None or calls < maximum_model_calls)
        ):
            if repeat_inflight is not None:
                repeat_steps = repeat_inflight["steps"]
                last_repeat_step = repeat_steps[-1]
                repeat_complete = bool(
                    len(repeat_steps) == repeat_inflight["repeat_requested"]
                    or last_repeat_step["result"]["status"] == "blocked"
                    or last_repeat_step["episode_terminal"] is True
                )
                if repeat_complete:
                    branch_macro_actions_before = conversation.macro_action_count
                    conversation.append_tool_result(
                        _with_action_decisions_remaining(
                            PublicToolResult(
                                status=last_repeat_step["result"]["status"],
                                blocked_category=last_repeat_step["result"].get(
                                    "blocked_category"
                                ),
                                repeat_requested=repeat_inflight[
                                    "repeat_requested"
                                ],
                                repeat_executed=len(repeat_steps),
                                steps=tuple(
                                    {
                                        "step_index": step["step_index"],
                                        **step["result"],
                                    }
                                    for step in repeat_steps
                                ),
                            ),
                            maximum_action_decisions=maximum_action_decisions,
                            resolved_action_decisions=macro_actions_resolved,
                        )
                    )
                    for repeat_step in repeat_steps:
                        conversation.append_observation(
                            _observation_from_artifact(
                                repeat_step["observation"],
                                output_dir=output_dir,
                            ),
                            kind="physical",
                        )
                    branch_macro_actions_after = conversation.macro_action_count
                    if (
                        branch_macro_actions_after
                        != branch_macro_actions_before + 1
                    ):
                        raise ModelClientError(
                            "navigation repeat macro counter changed"
                        )
                    macro_actions_resolved += 1
                    episode_terminal = last_repeat_step["episode_terminal"]
                    repeat_inflight = None
                    if episode_terminal:
                        conversation.finish_after_official_success()
                        active = False
                        status = "DONE"
                        terminated_by_official_success = True
                        _write_resume_checkpoint(
                            output_dir=output_dir,
                            scene=scene,
                            experiment_id=experiment_id,
                            scene_episode_id=scene_episode_id,
                            prompt_sha256=prompt_sha256,
                            conversation=conversation,
                            counters=checkpoint_counters(),
                            resume_session_index=resume_session_index,
                            state="TERMINAL",
                        )
                        break
                    if (
                        conversation.consecutive_blocked_macro_actions
                        >= maximum_consecutive_blocked
                    ):
                        scene.request("end_episode", reason="budget_exhausted")
                        conversation.finish_after_consecutive_blocked_limit()
                        active = False
                        status = "FAIL_CONSECUTIVE_BLOCKED"
                        terminated_by_consecutive_blocked = True
                        _write_resume_checkpoint(
                            output_dir=output_dir,
                            scene=scene,
                            experiment_id=experiment_id,
                            scene_episode_id=scene_episode_id,
                            prompt_sha256=prompt_sha256,
                            conversation=conversation,
                            counters=checkpoint_counters(),
                            resume_session_index=resume_session_index,
                            state="TERMINAL",
                        )
                        break
                    _write_resume_checkpoint(
                        output_dir=output_dir,
                        scene=scene,
                        experiment_id=experiment_id,
                        scene_episode_id=scene_episode_id,
                        prompt_sha256=prompt_sha256,
                        conversation=conversation,
                        counters=checkpoint_counters(),
                        resume_session_index=resume_session_index,
                        state="ACTIVE",
                    )
                    continue
            # A failed response never enters conversation history. Provider
            # retries repeat the exact request and seed; only a protocol repair
            # advances to the next predeclared sampling seed.
            dispatch = (
                conversation.pending_navigation_repeat_dispatch
                if repeat_inflight is not None
                else None
            )
            decision_macro_actions_before = conversation.macro_action_count
            frozen_request: Mapping[str, Any] = {}
            if dispatch is None:
                frozen_request = conversation.build_native_request(
                    preserve_reasoning=provider_generation[
                        "preserve_thinking"
                    ],
                )
                if pending_recovery_notice is not None:
                    frozen_request = _request_with_public_repair_notice(
                        frozen_request,
                        pending_recovery_notice,
                    )
            attempts_for_decision = 0
            provider_failures_for_decision = 0
            marker_selection_repairs_for_decision = 0
            marker_selection_repair_notice: str | None = None
            length_repairs_for_decision = 0
            decision_seed_offset = (
                decision_repairs_for_sequence % len(PROTOCOL_ATTEMPT_SEEDS)
            )
            decision_seeds = (
                ()
                if dispatch is not None
                else (
                    PROTOCOL_ATTEMPT_SEEDS[decision_seed_offset:]
                    + PROTOCOL_ATTEMPT_SEEDS[:decision_seed_offset]
                )
            )
            for protocol_attempt_index, seed in enumerate(
                decision_seeds,
                start=1,
            ):
                response = None
                call_dir = None
                provider_retry_started_at = time.monotonic()
                provider_attempt_index = 0
                while (
                    provider_attempt_index
                    < maximum_provider_attempts_per_protocol_attempt
                ):
                    if calls >= maximum_model_calls:
                        break
                    provider_attempt_index += 1
                    call_dir = (
                        output_dir
                        / "provider_calls"
                        / f"call_{calls + 1:04d}"
                    )
                    calls += 1
                    attempts_for_decision += 1
                    request_for_protocol_attempt = frozen_request
                    if marker_selection_repair_notice is not None:
                        request_for_protocol_attempt = (
                            _request_with_public_repair_notice(
                                frozen_request,
                                marker_selection_repair_notice,
                            )
                        )
                    try:
                        response = provider.complete(
                            request_for_protocol_attempt,
                            call_dir=call_dir,
                            seed=seed,
                        )
                    except ProviderRequestError as exc:
                        provider_failures += 1
                        provider_failures_for_decision += 1
                        retry_candidate = (
                            exc.retryable
                            and provider_attempt_index
                            < maximum_provider_attempts_per_protocol_attempt
                            and calls < maximum_model_calls
                        )
                        retry_elapsed_seconds = max(
                            0.0,
                            time.monotonic() - provider_retry_started_at,
                        )
                        retry_delay_seconds = retry_policy.delay_seconds(
                            retry_index=provider_attempt_index,
                            jitter_key=(
                                f"{experiment_id}:{scene_episode_id}:"
                                f"{actions}:{protocol_attempt_index}"
                            ),
                            retry_after_seconds=(
                                exc.retry_after_seconds
                                if retry_policy.enabled
                                else None
                            ),
                        )
                        retry_time_budget_exhausted = (
                            retry_policy.enabled
                            and retry_elapsed_seconds + retry_delay_seconds
                            > retry_policy.maximum_elapsed_seconds
                        )
                        regular_retry_allowed = (
                            retry_candidate
                            and not retry_time_budget_exhausted
                        )
                        automatic_retry_allowed = (
                            exc.retryable
                            and not regular_retry_allowed
                            and pause_on_client_failure
                            and calls < maximum_model_calls
                            and consecutive_provider_retry_windows
                            < MAXIMUM_AUTOMATIC_PROVIDER_RETRY_WINDOWS
                        )
                        retry_allowed = (
                            regular_retry_allowed
                            or automatic_retry_allowed
                        )
                        effective_retry_delay_seconds = (
                            retry_delay_seconds
                            if regular_retry_allowed
                            else (
                                AUTOMATIC_PROVIDER_RETRY_WINDOW_DELAY_SECONDS
                                if automatic_retry_allowed
                                else 0.0
                            )
                        )
                        _write_json(
                            call_dir / "provider_error.json",
                            {
                                "status": "PROVIDER_ERROR",
                                "attempt_index": provider_attempt_index,
                                "maximum_attempts": (
                                    maximum_provider_attempts_per_protocol_attempt
                                ),
                                "provider_attempt_index": provider_attempt_index,
                                "maximum_provider_attempts_per_protocol_attempt": (
                                    maximum_provider_attempts_per_protocol_attempt
                                ),
                                "protocol_attempt_index": protocol_attempt_index,
                                "maximum_protocol_attempts_per_decision": (
                                    MAXIMUM_PROTOCOL_ATTEMPTS_PER_DECISION
                                ),
                                "seed": seed,
                                "retry_allowed": retry_allowed,
                                "retry_delay_seconds": (
                                    effective_retry_delay_seconds
                                    if retry_allowed
                                    else 0.0
                                ),
                                "retry_elapsed_seconds": round(
                                    retry_elapsed_seconds,
                                    6,
                                ),
                                "retry_time_budget_exhausted": (
                                    retry_time_budget_exhausted
                                ),
                                "provider_retry_policy": (
                                    retry_policy.to_record()
                                ),
                                "provider_retry_window_index": (
                                    consecutive_provider_retry_windows
                                ),
                                "automatic_retry_window_allowed": (
                                    automatic_retry_allowed
                                ),
                                "maximum_automatic_provider_retry_windows": (
                                    MAXIMUM_AUTOMATIC_PROVIDER_RETRY_WINDOWS
                                ),
                                **exc.artifact_fields(),
                            },
                        )
                        if regular_retry_allowed:
                            provider_retries += 1
                            if retry_delay_seconds > 0.0:
                                time.sleep(retry_delay_seconds)
                            continue
                        if automatic_retry_allowed:
                            provider_retries += 1
                            automatic_provider_retry_windows += 1
                            consecutive_provider_retry_windows += 1
                            _write_provider_retry_state(
                                output_dir,
                                total=automatic_provider_retry_windows,
                                consecutive=consecutive_provider_retry_windows,
                            )
                            time.sleep(
                                AUTOMATIC_PROVIDER_RETRY_WINDOW_DELAY_SECONDS
                            )
                            provider_attempt_index = 0
                            provider_retry_started_at = time.monotonic()
                            continue
                        terminal_provider_failure = exc.failure_type
                        status = "FAIL_PROVIDER"
                        break
                    except ModelClientError as exc:
                        provider_failures += 1
                        provider_failures_for_decision += 1
                        terminal_provider_failure = "provider_client_error"
                        status = "FAIL_PROVIDER"
                        _write_json(
                            call_dir / "provider_error.json",
                            {
                                "status": "PROVIDER_ERROR",
                                "failure_type": terminal_provider_failure,
                                "exception_type": type(exc).__name__,
                                "exception_message": _redact_provider_error_text(str(exc)),
                                "retryable": False,
                                "http_status": None,
                                "response_body_bytes": 0,
                                "response_body_sha256": hashlib.sha256(b"").hexdigest(),
                                "response_body_truncated": False,
                                "response_body_preview": "",
                                "attempt_index": provider_attempt_index,
                                "maximum_attempts": (
                                    maximum_provider_attempts_per_protocol_attempt
                                ),
                                "provider_attempt_index": provider_attempt_index,
                                "maximum_provider_attempts_per_protocol_attempt": (
                                    maximum_provider_attempts_per_protocol_attempt
                                ),
                                "protocol_attempt_index": protocol_attempt_index,
                                "maximum_protocol_attempts_per_decision": (
                                    MAXIMUM_PROTOCOL_ATTEMPTS_PER_DECISION
                                ),
                                "seed": seed,
                                "retry_allowed": False,
                            },
                        )
                        break
                    else:
                        if consecutive_provider_retry_windows:
                            consecutive_provider_retry_windows = 0
                            _write_provider_retry_state(
                                output_dir,
                                total=automatic_provider_retry_windows,
                                consecutive=0,
                            )
                        break
                if response is None:
                    break
                assert call_dir is not None
                compatibility_record: Mapping[str, Any] | None = None
                response_for_protocol = response
                try:
                    dispatch = conversation.append_assistant_completion(
                        response_for_protocol,
                        remaining_action_decisions=(
                            maximum_action_decisions
                            - macro_actions_resolved
                        ),
                    )
                except ValueError as native_exc:
                    compatibility_exc: ValueError | None = None
                    if action_tag_compatibility:
                        try:
                            (
                                response_for_protocol,
                                compatibility_record,
                            ) = _normalize_single_action_tag_response(response)
                        except ValueError:
                            pass
                        else:
                            try:
                                dispatch = conversation.append_assistant_completion(
                                    response_for_protocol,
                                    remaining_action_decisions=(
                                        maximum_action_decisions
                                        - macro_actions_resolved
                                    ),
                                    provider_response=response,
                                )
                            except ValueError as normalized_exc:
                                compatibility_exc = normalized_exc
                    if dispatch is None:
                        exc = compatibility_exc or native_exc
                    else:
                        exc = None
                else:
                    exc = None
                if dispatch is None:
                    assert exc is not None
                    invalid_model_responses += 1
                    marker_selection_error = (
                        _is_marker_selection_protocol_error(exc)
                    )
                    if marker_selection_error:
                        retry_allowed = (
                            marker_selection_repairs_for_decision
                            < MAXIMUM_AUTOMATIC_DECISION_REPAIRS_PER_SEQUENCE
                            and calls < maximum_model_calls
                        )
                    else:
                        retry_allowed = (
                            protocol_attempt_index
                            < MAXIMUM_PROTOCOL_ATTEMPTS_PER_DECISION
                            and calls < maximum_model_calls
                        )
                    failure_type = _native_protocol_failure_type(response)
                    if length_repairs_for_decision or (
                        _native_response_metadata(response)["finish_reason"] == "length"
                        and failure_type != "reasoning_budget_exhausted"
                    ):
                        retry_allowed = False
                    elif failure_type == "reasoning_budget_exhausted":
                        retry_allowed = (
                            protocol_attempt_index < MAXIMUM_PROTOCOL_ATTEMPTS_PER_DECISION
                            and calls < maximum_model_calls
                        )
                    _write_json(
                        call_dir / "protocol_attempt.json",
                        {
                            "status": "INVALID_MODEL_RESPONSE",
                            "failure_type": failure_type,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc)[:512],
                            "attempt_index": protocol_attempt_index,
                            "maximum_attempts": (
                                MAXIMUM_PROTOCOL_ATTEMPTS_PER_DECISION
                            ),
                            "protocol_attempt_index": protocol_attempt_index,
                            "maximum_protocol_attempts_per_decision": (
                                MAXIMUM_PROTOCOL_ATTEMPTS_PER_DECISION
                            ),
                            "seed": seed,
                            "retry_allowed": retry_allowed,
                            "marker_selection_error": marker_selection_error,
                            "length_repair_index": length_repairs_for_decision,
                            "maximum_length_repairs_per_decision": 1,
                            "marker_selection_repair_index": (
                                marker_selection_repairs_for_decision
                            ),
                            "maximum_marker_selection_repairs": (
                                MAXIMUM_AUTOMATIC_DECISION_REPAIRS_PER_SEQUENCE
                            ),
                            "action_tag_compatibility_enabled": (
                                action_tag_compatibility
                            ),
                            "action_tag_compatibility_applied": False,
                            **_native_response_metadata(response),
                        },
                    )
                    if retry_allowed:
                        protocol_retries += 1
                        if failure_type == "reasoning_budget_exhausted":
                            length_repairs_for_decision += 1
                            marker_selection_repair_notice = (
                                "The previous response reached its output limit without a tool call. "
                                "No robot action was dispatched; the scene and observation are unchanged. "
                                "Return exactly one legal robot_action tool call now, with no ordinary assistant text."
                            )
                            if conversation.pending_marker_action is not None:
                                marker_selection_repair_notice += (
                                    " Selection is pending for " + conversation.pending_marker_action + ". "
                                    "Choose one plausible task-relevant C-label from the current Marker image "
                                    "and submit it as target with the same action now. Do not match all labels "
                                    "across panels. If none is suitable, cancel using another action without target."
                                )
                        if marker_selection_error:
                            marker_selection_retries += 1
                            marker_selection_repairs_for_decision += 1
                            marker_selection_repair_notice = (
                                "The previous target selection was rejected "
                                "before execution. The scene and current Marker "
                                "observation are unchanged. Either continue the "
                                "current Marker action with one shown C-label, "
                                "or abandon it by choosing another action without "
                                "target. Do not attach a C-label to another action "
                                "or reuse a stale C-label."
                            )
                        continue
                    terminal_protocol_failure = (
                        "marker_selection_repair_exhausted"
                        if marker_selection_error
                        else failure_type
                    )
                    status = (
                        "PAUSED_MODEL_PROTOCOL"
                        if marker_selection_error
                        else "FAIL_MODEL_PROTOCOL"
                    )
                    break
                else:
                    accept_response = getattr(provider, "accept_response", None)
                    if callable(accept_response):
                        accept_response(
                            response_for_protocol,
                            output_dir=output_dir,
                            canonical_function_arguments=(
                                dispatch.decision.function_arguments()
                            ),
                        )
                    if compatibility_record is not None:
                        action_tag_compatibility_applied += 1
                        _write_json(
                            call_dir / "compatibility_normalization.json",
                            compatibility_record,
                        )
                    normalization = {
                        "ignored_target_fields": (
                            dispatch.decision.ignored_target_fields
                        ),
                        "ignored_navigation_fields": (
                            dispatch.decision.ignored_navigation_fields
                        ),
                        "eager_marker_targets_removed": int(
                            dispatch.decision.eager_marker_target_removed
                        ),
                    }
                    if any(normalization.values()):
                        normalization_stats["normalized_model_responses"] += 1
                        for name, count in normalization.items():
                            normalization_stats[name] += count
                        _write_json(
                            call_dir / "normalization.json",
                            {
                                "schema_version": NORMALIZATION_STATS_VERSION,
                                **normalization,
                            },
                        )
                        _write_normalization_stats(
                            output_dir,
                            normalization_stats,
                        )
                    _write_json(
                        call_dir / "protocol_attempt.json",
                        {
                            "status": "PASS",
                            "attempt_index": protocol_attempt_index,
                            "maximum_attempts": (
                                MAXIMUM_PROTOCOL_ATTEMPTS_PER_DECISION
                            ),
                            "protocol_attempt_index": protocol_attempt_index,
                            "maximum_protocol_attempts_per_decision": (
                                MAXIMUM_PROTOCOL_ATTEMPTS_PER_DECISION
                            ),
                            "seed": seed,
                            "retry_allowed": False,
                            "action_tag_compatibility_enabled": (
                                action_tag_compatibility
                            ),
                            "action_tag_compatibility_applied": (
                                compatibility_record is not None
                            ),
                            **_native_response_metadata(response),
                        },
                    )
                    if attempts_for_decision > 1:
                        decisions_recovered_after_retry += 1
                        if provider_failures_for_decision:
                            decisions_recovered_after_provider_retry += 1
                    break
            if dispatch is None:
                if (
                    terminal_protocol_failure is None
                    and terminal_provider_failure is None
                ):
                    terminal_protocol_failure = "internal_model_call_cap_exhausted"
                    status = "FAIL_INTERNAL_MODEL_CALL_CAP"
                if pause_on_client_failure and terminal_provider_failure is not None:
                    status = "PAUSED_PROVIDER"
                elif (
                    terminal_protocol_failure
                    == "marker_selection_repair_exhausted"
                    or (
                        pause_on_client_failure
                        and terminal_protocol_failure is not None
                        and terminal_protocol_failure
                        != "internal_model_call_cap_exhausted"
                    )
                ):
                    status = "PAUSED_MODEL_PROTOCOL"
                else:
                    scene.request(
                        "end_episode",
                        reason="client_error",
                    )
                    active = False
                if status in PAUSED_CLIENT_STATUSES:
                    _write_resume_checkpoint(
                        output_dir=output_dir,
                        scene=scene,
                        experiment_id=experiment_id,
                        scene_episode_id=scene_episode_id,
                        prompt_sha256=prompt_sha256,
                        conversation=conversation,
                        counters=checkpoint_counters(),
                        resume_session_index=resume_session_index,
                        state="PAUSED",
                        pause_status=status,
                        terminal_protocol_failure=terminal_protocol_failure,
                        terminal_provider_failure=terminal_provider_failure,
                    )
                break
            decision_macro_actions_after = conversation.macro_action_count
            if (
                decision_macro_actions_after
                == decision_macro_actions_before + 1
            ):
                macro_actions_resolved += 1
            elif decision_macro_actions_after != decision_macro_actions_before:
                raise ModelClientError(
                    "decision macro action counter changed unexpectedly"
                )
            if dispatch.decision.action == TERMINATION_ACTION:
                conversation.append_tool_result(
                    _with_action_decisions_remaining(
                        PublicToolResult("completed"),
                        maximum_action_decisions=maximum_action_decisions,
                        resolved_action_decisions=macro_actions_resolved,
                    )
                )
                conversation.finish_after_done()
                scene.request("end_episode", reason="done")
                active = False
                status = "DONE"
                break
            step_fields: dict[str, Any] = {
                "sequence": actions,
                "action": dispatch.decision.action,
            }
            if dispatch.decision.target is not None:
                step_fields["target"] = dispatch.decision.target
            if dispatch.decision.navigation_substeps > 1:
                step_fields["navigation_substeps"] = (
                    dispatch.decision.navigation_substeps
                )
            scene_step_attempts += 1
            try:
                step = scene.request("step", **step_fields)
            except ModelClientError as raw_exc:
                exc = (
                    raw_exc
                    if isinstance(raw_exc, SceneRequestError)
                    else _probe_scene_after_transport_failure(
                        scene=scene,
                        error=raw_exc,
                    )
                )
                terminal_scene_error_type = exc.error_type
                terminal_scene_error_id = exc.error_id
                terminal_scene_error_origin = exc.error_origin
                recovery = exc.recovery
                same_process_recovery = (
                    _public_error_allows_same_process_recovery(exc)
                    and isinstance(recovery, Mapping)
                )
                if not same_process_recovery:
                    scene_reload_required = not (
                        isinstance(recovery, Mapping)
                        and recovery.get("scene_resident") is True
                        and recovery.get("session_valid") is True
                    )
                    status = (
                        "FAIL_SCENE_RELOAD_REQUIRED"
                        if scene_reload_required
                        else "FAIL_SCENE_RECOVERY_EXHAUSTED"
                    )
                    active = bool(
                        isinstance(recovery, Mapping)
                        and recovery.get("episode_active") is True
                    )
                    _write_scene_recovery_event(
                        output_dir=output_dir,
                        recovery_index=scene_recovery_count + 1,
                        phase="step",
                        error=exc,
                        outcome=(
                            "scene_reload_required"
                            if scene_reload_required
                            else "automatic_recovery_budget_exhausted"
                        ),
                        details={"failed_sequence": actions},
                    )
                    break

                if (
                    recovery.get("episode_active") is True
                    and recovery.get("episode_id") == scene_episode_id
                    and recovery.get("next_sequence") == actions
                ):
                    current = _read_strict_json(
                        Path(output_dir) / "resume_checkpoint.json",
                        max_bytes=MAX_RESUME_ARTIFACT_BYTES,
                    )
                    transcript = current.get("conversation")
                    if (
                        current.get("state") != "ACTIVE"
                        or current.get("episode_id") != scene_episode_id
                        or current.get("next_sequence") != actions
                        or not isinstance(transcript, Mapping)
                    ):
                        raise ModelClientError(
                            "active model checkpoint cannot repair scene protocol error"
                        )
                    conversation = EpisodeConversation.from_artifact_transcript(
                        transcript,
                        expected_contract=contract,
                        system_prompt_revision=prompt,
                        artifact_root=output_dir,
                        tool_schema=tool_schema,
                    )
                    repeat_inflight = _validate_repeat_inflight(
                        current.get("repeat_inflight"),
                        next_sequence=actions,
                    )
                    if (
                        decision_repairs_for_sequence
                        >= MAXIMUM_AUTOMATIC_DECISION_REPAIRS_PER_SEQUENCE
                    ):
                        terminal_protocol_failure = (
                            "scene_protocol_decision_repair_exhausted"
                        )
                        status = "PAUSED_MODEL_PROTOCOL"
                        active = True
                        _write_scene_recovery_event(
                            output_dir=output_dir,
                            recovery_index=scene_recovery_count + 1,
                            phase="step",
                            error=exc,
                            outcome="decision_repair_budget_exhausted",
                            details={
                                "failed_sequence": actions,
                                "maximum_decision_repairs": (
                                    MAXIMUM_AUTOMATIC_DECISION_REPAIRS_PER_SEQUENCE
                                ),
                                "physical_scene_mutated": False,
                            },
                        )
                        _write_resume_checkpoint(
                            output_dir=output_dir,
                            scene=scene,
                            experiment_id=experiment_id,
                            scene_episode_id=scene_episode_id,
                            prompt_sha256=prompt_sha256,
                            conversation=conversation,
                            counters=checkpoint_counters(),
                            resume_session_index=resume_session_index,
                            state="PAUSED",
                            pause_status=status,
                            terminal_protocol_failure=terminal_protocol_failure,
                            repeat_inflight=repeat_inflight,
                        )
                        break
                    scene_recovery_count += 1
                    decision_only_recoveries += 1
                    decision_repairs_for_sequence += 1
                    recovery_branch_index += 1
                    pending_recovery_notice = (
                        "The previous robot_action request was rejected before "
                        "execution because it did not match the current public "
                        "action/Marker protocol. The scene and observation are "
                        "unchanged. Re-check the current observation and choose "
                        "exactly one legal action. If selecting a C-label, keep "
                        "the current Marker action and choose only a C-label "
                        "shown in the current observation."
                        if exc.error_type == "SCENE_PROTOCOL_ERROR"
                        else
                        "The previous robot action was not committed. The scene "
                        "and observation are unchanged. Re-evaluate the current "
                        "observation and choose one new legal action."
                    )
                    _write_scene_recovery_event(
                        output_dir=output_dir,
                        recovery_index=scene_recovery_count,
                        phase="step",
                        error=exc,
                        outcome="decision_rewound_without_scene_mutation",
                        details={"restored_action_count": actions},
                    )
                    terminal_scene_error_type = None
                    terminal_scene_error_id = None
                    terminal_scene_error_origin = None
                    continue

                retained = recovery.get("rolling_checkpoint_action_counts")
                if not isinstance(retained, list):
                    retained = []
                common_checkpoint: Mapping[str, Any] | None = None
                if retained:
                    try:
                        common_checkpoint = _load_latest_model_rolling_checkpoint(
                            output_dir=output_dir,
                            resident_action_counts=retained,
                        )
                    except ModelClientError:
                        common_checkpoint = None
                if common_checkpoint is not None:
                    checkpoint_resume = common_checkpoint["resume_checkpoint"]
                    checkpoint_actions = checkpoint_resume["next_sequence"]
                else:
                    checkpoint_actions = 0
                recovery_signature = _automatic_scene_recovery_signature(
                    error=exc,
                    step_fields=step_fields,
                    checkpoint_action_count=checkpoint_actions,
                )
                if recovery_signature is None:
                    last_scene_recovery_signature = None
                    consecutive_scene_recovery_failures = 0
                elif recovery_signature == last_scene_recovery_signature:
                    consecutive_scene_recovery_failures += 1
                else:
                    last_scene_recovery_signature = recovery_signature
                    consecutive_scene_recovery_failures = 1
                if consecutive_scene_recovery_failures >= 2:
                    status = "FAIL_SCENE_RELOAD_REQUIRED"
                    scene_reload_required = True
                    active = bool(recovery.get("episode_active") is True)
                    _write_scene_recovery_event(
                        output_dir=output_dir,
                        recovery_index=scene_recovery_count + 1,
                        phase="step",
                        error=exc,
                        outcome="duplicate_scene_error_recovery_suppressed",
                        details={
                            "failed_sequence": actions,
                            "checkpoint_action_count": checkpoint_actions,
                            "consecutive_occurrences": (
                                consecutive_scene_recovery_failures
                            ),
                            "recovery_signature": recovery_signature,
                        },
                    )
                    break

                if common_checkpoint is not None:
                    if (
                        checkpoint_restores
                        >= MAXIMUM_AUTOMATIC_CHECKPOINT_RESTORES
                    ):
                        status = "FAIL_SCENE_RECOVERY_EXHAUSTED"
                        active = bool(recovery.get("episode_active") is True)
                        _write_scene_recovery_event(
                            output_dir=output_dir,
                            recovery_index=scene_recovery_count + 1,
                            phase="step",
                            error=exc,
                            outcome="checkpoint_restore_budget_exhausted",
                            details={
                                "failed_sequence": actions,
                                "maximum_checkpoint_restores": (
                                    MAXIMUM_AUTOMATIC_CHECKPOINT_RESTORES
                                ),
                            },
                        )
                        break
                    scene_recovery_count += 1
                    checkpoint_restores += 1
                    recovery_branch_index += 1
                    decision_repairs_for_sequence = 0
                    pending_recovery_notice = (
                        "The previous robot action was not committed because the "
                        "resident simulator restored a verified checkpoint. "
                        "Re-evaluate the current observation and choose one new "
                        "legal action."
                    )
                    try:
                        restored = scene.request(
                            "restore_checkpoint",
                            captured_action_count=checkpoint_actions,
                        )
                    except ModelClientError as raw_restore_exc:
                        restore_exc = _typed_scene_client_failure(raw_restore_exc)
                        scene_reload_required = True
                        terminal_scene_error_type = restore_exc.error_type
                        terminal_scene_error_id = restore_exc.error_id
                        terminal_scene_error_origin = restore_exc.error_origin
                        status = "FAIL_SCENE_RELOAD_REQUIRED"
                        active = False
                        _write_scene_recovery_event(
                            output_dir=output_dir,
                            recovery_index=scene_recovery_count,
                            phase="step",
                            error=restore_exc,
                            outcome="checkpoint_restore_failed",
                            details={
                                "requested_action_count": checkpoint_actions,
                            },
                        )
                        break
                    if (
                        restored.get("status") != "PASS"
                        or restored.get("episode_id") != scene_episode_id
                        or restored.get("next_sequence") != checkpoint_actions
                        or not isinstance(restored.get("observation"), Mapping)
                    ):
                        raise ModelClientError(
                            "resident checkpoint restore response changed"
                        )
                    restored_observation = _materialize_observation(
                        restored["observation"],
                        output_dir=output_dir,
                        scene_run_dir=scene.scene_run_dir,
                    )
                    transcript = checkpoint_resume.get("conversation")
                    if not isinstance(transcript, Mapping):
                        raise ModelClientError(
                            "model rolling checkpoint conversation is missing"
                        )
                    conversation = EpisodeConversation.from_artifact_transcript(
                        transcript,
                        expected_contract=contract,
                        system_prompt_revision=prompt,
                        artifact_root=output_dir,
                        tool_schema=tool_schema,
                    )
                    rolled_back_completed_actions += actions - checkpoint_actions
                    actions = checkpoint_actions
                    repeat_inflight = _validate_repeat_inflight(
                        checkpoint_resume.get("repeat_inflight"),
                        next_sequence=actions,
                    )
                    active = True
                    _write_resume_checkpoint(
                        output_dir=output_dir,
                        scene=scene,
                        experiment_id=experiment_id,
                        scene_episode_id=scene_episode_id,
                        prompt_sha256=prompt_sha256,
                        conversation=conversation,
                        counters=checkpoint_counters(),
                        resume_session_index=resume_session_index,
                        state="ACTIVE",
                        repeat_inflight=repeat_inflight,
                    )
                    _write_scene_recovery_event(
                        output_dir=output_dir,
                        recovery_index=scene_recovery_count,
                        phase="step",
                        error=exc,
                        outcome="rolling_checkpoint_restored",
                        details={
                            "restored_action_count": actions,
                            "restored_observation_id": (
                                restored_observation.observation_id
                            ),
                            "conversation_rewound_to_same_action_count": True,
                        },
                    )
                    terminal_scene_error_type = None
                    terminal_scene_error_id = None
                    terminal_scene_error_origin = None
                    continue

                if same_scene_restarts >= MAXIMUM_AUTOMATIC_SAME_SCENE_RESETS:
                    status = "FAIL_SCENE_RECOVERY_EXHAUSTED"
                    active = bool(recovery.get("episode_active") is True)
                    _write_scene_recovery_event(
                        output_dir=output_dir,
                        recovery_index=scene_recovery_count + 1,
                        phase="step",
                        error=exc,
                        outcome="same_scene_reset_budget_exhausted",
                        details={
                            "failed_sequence": actions,
                            "maximum_same_scene_resets": (
                                MAXIMUM_AUTOMATIC_SAME_SCENE_RESETS
                            ),
                        },
                    )
                    break

                scene_recovery_count += 1
                recovery_branch_index += 1
                decision_repairs_for_sequence = 0
                if recovery.get("episode_active") is True:
                    try:
                        scene.request("end_episode", reason="client_error")
                    except ModelClientError as raw_close_exc:
                        close_exc = _typed_scene_client_failure(raw_close_exc)
                        scene_reload_required = True
                        terminal_scene_error_type = close_exc.error_type
                        terminal_scene_error_id = close_exc.error_id
                        terminal_scene_error_origin = close_exc.error_origin
                        status = "FAIL_SCENE_RELOAD_REQUIRED"
                        active = False
                        _write_scene_recovery_event(
                            output_dir=output_dir,
                            recovery_index=scene_recovery_count,
                            phase="step",
                            error=close_exc,
                            outcome="ambiguous_action_cleanup_failed",
                            details={"failed_sequence": actions},
                        )
                        break
                    active = False
                elif recovery.get("episode_active") is not False:
                    scene_reload_required = True
                    status = "FAIL_SCENE_RELOAD_REQUIRED"
                    active = False
                    _write_scene_recovery_event(
                        output_dir=output_dir,
                        recovery_index=scene_recovery_count,
                        phase="step",
                        error=exc,
                        outcome="ambiguous_scene_state_requires_reload",
                        details={"failed_sequence": actions},
                    )
                    break

                # Before the first retained checkpoint, the safest available
                # recovery is a normal in-memory task reset and a fresh model
                # conversation.  The scene remains loaded.
                same_scene_restarts += 1
                next_episode_id = (
                    f"{experiment_id}_recovery_{scene_recovery_count:02d}"
                )
                try:
                    begin = scene.request(
                        "begin_episode",
                        episode_id=next_episode_id,
                        **episode_configuration,
                    )
                except ModelClientError as raw_restart_exc:
                    restart_exc = _typed_scene_client_failure(raw_restart_exc)
                    scene_reload_required = not (
                        isinstance(restart_exc.recovery, Mapping)
                        and restart_exc.recovery.get("session_valid") is True
                    )
                    terminal_scene_error_type = restart_exc.error_type
                    terminal_scene_error_id = restart_exc.error_id
                    terminal_scene_error_origin = restart_exc.error_origin
                    status = (
                        "FAIL_SCENE_RELOAD_REQUIRED"
                        if scene_reload_required
                        else "FAIL_SCENE_RECOVERY_EXHAUSTED"
                    )
                    active = False
                    _write_scene_recovery_event(
                        output_dir=output_dir,
                        recovery_index=scene_recovery_count,
                        phase="step",
                        error=restart_exc,
                        outcome="same_scene_restart_failed",
                        details={"next_scene_episode_id": next_episode_id},
                    )
                    break
                rolled_back_completed_actions += actions
                actions = 0
                repeat_inflight = None
                scene_episode_id = next_episode_id
                active = True
                initial_observation = _materialize_observation(
                    begin["observation"],
                    output_dir=output_dir,
                    scene_run_dir=scene.scene_run_dir,
                )
                _publish_action_after_view(
                    initial_observation,
                    output_dir=output_dir,
                    step_index=0,
                    action_name="initial",
                    recovery_branch_index=recovery_branch_index,
                )
                conversation = EpisodeConversation(
                    contract,
                    initial_observation,
                    system_prompt_revision=prompt,
                    tool_schema=tool_schema,
                )
                pending_recovery_notice = (
                    "The previous episode was safely reset in the same loaded "
                    "scene before a checkpoint existed. Start the task again "
                    "from this new initial observation."
                )
                _write_resume_checkpoint(
                    output_dir=output_dir,
                    scene=scene,
                    experiment_id=experiment_id,
                    scene_episode_id=scene_episode_id,
                    prompt_sha256=prompt_sha256,
                    conversation=conversation,
                    counters=checkpoint_counters(),
                    resume_session_index=resume_session_index,
                    state="ACTIVE",
                )
                _write_scene_recovery_event(
                    output_dir=output_dir,
                    recovery_index=scene_recovery_count,
                    phase="step",
                    error=exc,
                    outcome="same_scene_reset_and_reasoning_restart",
                    details={"next_scene_episode_id": scene_episode_id},
                )
                terminal_scene_error_type = None
                terminal_scene_error_id = None
                terminal_scene_error_origin = None
                continue
            last_scene_recovery_signature = None
            consecutive_scene_recovery_failures = 0
            actions += 1
            decision_repairs_for_sequence = 0
            pending_recovery_notice = None
            result = step["action_result"]
            observation_record = step["observation"]
            public_result = _public_tool_result_from_scene_step(
                result,
                observation_record,
                expected_selection_stage=conversation.expected_selection_stage,
                allow_blocked_category=blocked_categories_enabled,
                allow_blocked_detail=blocked_details_enabled,
            )
            materialized_observation = _materialize_observation(
                observation_record,
                output_dir=output_dir,
                scene_run_dir=scene.scene_run_dir,
            )
            _publish_action_after_view(
                materialized_observation,
                output_dir=output_dir,
                step_index=actions,
                action_name=dispatch.decision.action,
                recovery_branch_index=recovery_branch_index,
            )
            episode_terminal = step.get("episode_terminal", False)
            if not isinstance(episode_terminal, bool):
                raise ModelClientError("scene terminal response changed")
            if episode_terminal:
                if step.get("session_valid") is not True:
                    raise ModelClientError(
                        "officially terminal scene did not preserve its session"
                    )
                if step.get("rolling_checkpoint") is not None:
                    raise ModelClientError(
                        "officially terminal scene returned a rolling checkpoint"
                    )
            requested_repeat = dispatch.decision.public_repeat
            if requested_repeat > 1:
                if (
                    observation_record.get("kind") != "physical"
                    or observation_record.get("marker_targets") != []
                    or public_result.status not in {"completed", "blocked"}
                    or public_result.body_odometry is None
                    or public_result.stage is not None
                ):
                    raise ModelClientError(
                        "navigation repeat physical-step result changed"
                    )
                if repeat_inflight is None:
                    repeat_steps: list[Mapping[str, Any]] = []
                else:
                    if (
                        repeat_inflight["action"] != dispatch.decision.action
                        or repeat_inflight["repeat_requested"]
                        != requested_repeat
                    ):
                        raise ModelClientError("navigation repeat dispatch changed")
                    repeat_steps = list(repeat_inflight["steps"])
                repeat_steps.append(
                    {
                        "step_index": len(repeat_steps) + 1,
                        "sequence": actions - 1,
                        "result": public_result.to_public_dict(),
                        "observation": materialized_observation.artifact_record(),
                        "episode_terminal": episode_terminal,
                    }
                )
                repeat_inflight = {
                    "action": dispatch.decision.action,
                    "repeat_requested": requested_repeat,
                    "steps": repeat_steps,
                }
                repeat_inflight = _validate_repeat_inflight(
                    repeat_inflight,
                    next_sequence=actions,
                )
                active_checkpoint = _write_resume_checkpoint(
                    output_dir=output_dir,
                    scene=scene,
                    experiment_id=experiment_id,
                    scene_episode_id=scene_episode_id,
                    prompt_sha256=prompt_sha256,
                    conversation=conversation,
                    counters=checkpoint_counters(),
                    resume_session_index=resume_session_index,
                    state="ACTIVE",
                    repeat_inflight=repeat_inflight,
                )
                scene_checkpoint = step.get("rolling_checkpoint")
                if scene_checkpoint is not None:
                    if not isinstance(scene_checkpoint, Mapping):
                        raise ModelClientError(
                            "scene rolling checkpoint response changed"
                        )
                    _write_model_rolling_checkpoint(
                        output_dir=output_dir,
                        resume_checkpoint=active_checkpoint,
                        scene_checkpoint=scene_checkpoint,
                    )
                continue
            branch_macro_actions_before = conversation.macro_action_count
            public_result = _with_action_decisions_remaining(
                public_result,
                maximum_action_decisions=maximum_action_decisions,
                resolved_action_decisions=macro_actions_resolved,
            )
            conversation.append_tool_result(public_result)
            conversation.append_observation(
                materialized_observation,
                kind=observation_record["kind"],
                marker_targets=observation_record.get("marker_targets", ()),
            )
            branch_macro_actions_after = conversation.macro_action_count
            if branch_macro_actions_after == branch_macro_actions_before + 1:
                macro_actions_resolved += 1
            elif branch_macro_actions_after != branch_macro_actions_before:
                raise ModelClientError("macro action counter changed unexpectedly")
            if episode_terminal:
                conversation.finish_after_official_success()
                active = False
                status = "DONE"
                terminated_by_official_success = True
                _write_resume_checkpoint(
                    output_dir=output_dir,
                    scene=scene,
                    experiment_id=experiment_id,
                    scene_episode_id=scene_episode_id,
                    prompt_sha256=prompt_sha256,
                    conversation=conversation,
                    counters=checkpoint_counters(),
                    resume_session_index=resume_session_index,
                    state="TERMINAL",
                )
                break
            if (
                conversation.consecutive_blocked_macro_actions
                >= maximum_consecutive_blocked
            ):
                # The scene has already run its official verifier for this
                # action.  Reuse the established budget-exhausted wire reason
                # so this client remains compatible with already-loaded
                # resident servers while recording the precise client policy
                # in its own terminal artifacts.
                scene.request("end_episode", reason="budget_exhausted")
                conversation.finish_after_consecutive_blocked_limit()
                active = False
                status = "FAIL_CONSECUTIVE_BLOCKED"
                terminated_by_consecutive_blocked = True
                _write_resume_checkpoint(
                    output_dir=output_dir,
                    scene=scene,
                    experiment_id=experiment_id,
                    scene_episode_id=scene_episode_id,
                    prompt_sha256=prompt_sha256,
                    conversation=conversation,
                    counters=checkpoint_counters(),
                    resume_session_index=resume_session_index,
                    state="TERMINAL",
                )
                break
            active_checkpoint = _write_resume_checkpoint(
                output_dir=output_dir,
                scene=scene,
                experiment_id=experiment_id,
                scene_episode_id=scene_episode_id,
                prompt_sha256=prompt_sha256,
                conversation=conversation,
                counters=checkpoint_counters(),
                resume_session_index=resume_session_index,
                state="ACTIVE",
            )
            scene_checkpoint = step.get("rolling_checkpoint")
            if scene_checkpoint is not None:
                if not isinstance(scene_checkpoint, Mapping):
                    raise ModelClientError(
                        "scene rolling checkpoint response changed"
                    )
                _write_model_rolling_checkpoint(
                    output_dir=output_dir,
                    resume_checkpoint=active_checkpoint,
                    scene_checkpoint=scene_checkpoint,
                )
        if (
            active
            and macro_actions_resolved < maximum_action_decisions
            and calls >= maximum_model_calls
            and status == "FAIL_BUDGET_EXHAUSTED"
        ):
            status = "FAIL_INTERNAL_MODEL_CALL_CAP"
            terminal_protocol_failure = "internal_model_call_cap_exhausted"
        if active and status not in PAUSED_CLIENT_STATUSES:
            end_reason = (
                "client_error"
                if status.startswith("FAIL_SCENE_")
                or status == "FAIL_INTERNAL_MODEL_CALL_CAP"
                else "budget_exhausted"
            )
            try:
                scene.request("end_episode", reason=end_reason)
            except ModelClientError:
                scene_reload_required = True
                if status.startswith("FAIL_SCENE_"):
                    status = "FAIL_SCENE_RELOAD_REQUIRED"
            active = False
    except KeyboardInterrupt:
        if (
            active
            and pause_on_client_failure
            and (
                conversation.state == "awaiting_assistant"
                or (
                    conversation.state == "awaiting_tool_result"
                    and repeat_inflight is not None
                )
            )
        ):
            status = "PAUSED_INTERRUPTED"
            _write_resume_checkpoint(
                output_dir=output_dir,
                scene=scene,
                experiment_id=experiment_id,
                scene_episode_id=scene_episode_id,
                prompt_sha256=prompt_sha256,
                conversation=conversation,
                counters=checkpoint_counters(),
                resume_session_index=resume_session_index,
                state="PAUSED",
                pause_status=status,
                repeat_inflight=repeat_inflight,
            )
        else:
            if active:
                try:
                    scene.request("end_episode", reason="client_error")
                except Exception:
                    pass
            raise
    except Exception as exc:
        terminal_client_error_type = type(exc).__name__
        cleanup_error: SceneRequestError | None = None
        if active:
            try:
                scene.request("end_episode", reason="client_error")
            except ModelClientError as raw_cleanup_error:
                cleanup_error = _typed_scene_client_failure(raw_cleanup_error)
        try:
            cleanup_status = scene.request("status")
        except ModelClientError as raw_cleanup_error:
            cleanup_error = _typed_scene_client_failure(raw_cleanup_error)
            cleanup_valid = False
        else:
            cleanup_valid = bool(
                cleanup_status.get("scene_resident") is True
                and cleanup_status.get("session_valid") is True
                and cleanup_status.get("episode_active") is False
            )
        if cleanup_error is not None:
            terminal_scene_error_type = cleanup_error.error_type
            terminal_scene_error_id = cleanup_error.error_id
            terminal_scene_error_origin = cleanup_error.error_origin
        scene_reload_required = not cleanup_valid
        status = (
            "FAIL_CLIENT_RUNTIME"
            if cleanup_valid
            else "FAIL_SCENE_RELOAD_REQUIRED"
        )
        active = False
    except BaseException:
        if active:
            try:
                scene.request("end_episode", reason="client_error")
            except Exception:
                pass
        raise
    finally:
        _write_conversation_views(
            output_dir=output_dir,
            conversation=conversation,
        )
        _write_json(
            output_dir
            / "private"
            / "selected_target_visual_evidence.json",
            conversation.artifact_selected_target_visual_evidence(),
        )
    summary = {
        "status": status,
        "model": provider.model,
        "model_calls": calls,
        "actions_sent_to_scene": actions,
        "macro_actions_resolved": macro_actions_resolved,
        "current_branch_macro_actions_resolved": conversation.macro_action_count,
        "scene_step_attempts": scene_step_attempts,
        "maximum_action_decisions": maximum_action_decisions,
        "maximum_model_calls": maximum_model_calls,
        "public_budget_exhausted": (
            macro_actions_resolved >= maximum_action_decisions
        ),
        "internal_model_call_cap_exhausted": (
            calls >= maximum_model_calls
            and macro_actions_resolved < maximum_action_decisions
        ),
        "maximum_consecutive_blocked": maximum_consecutive_blocked,
        "consecutive_blocked_at_end": (
            conversation.consecutive_blocked_macro_actions
        ),
        "maximum_consecutive_blocked_observed": (
            conversation.maximum_consecutive_blocked_macro_actions
        ),
        "terminated_by_consecutive_blocked": (
            terminated_by_consecutive_blocked
        ),
        "terminal_stall_reason": (
            "consecutive_blocked_limit"
            if terminated_by_consecutive_blocked
            else None
        ),
        "physical_history_max": physical_history_max,
        "model_observation_max": physical_history_max + 1,
        "maximum_output_tokens": provider.maximum_output_tokens,
        "thinking_tool_choice_auto": thinking_tool_choice_auto,
        "action_tag_compatibility": action_tag_compatibility,
        "action_tag_compatibility_applied": (
            action_tag_compatibility_applied
        ),
        "maximum_protocol_attempts_per_decision": (
            MAXIMUM_PROTOCOL_ATTEMPTS_PER_DECISION
        ),
        "maximum_provider_attempts_per_protocol_attempt": (
            maximum_provider_attempts_per_protocol_attempt
        ),
        "provider_retry_policy": retry_policy.to_record(),
        "protocol_attempt_seeds": list(PROTOCOL_ATTEMPT_SEEDS),
        "invalid_model_responses": invalid_model_responses,
        **normalization_stats,
        "protocol_retries": protocol_retries,
        "marker_selection_retries": marker_selection_retries,
        "decisions_recovered_after_retry": decisions_recovered_after_retry,
        "provider_failures": provider_failures,
        "provider_retries": provider_retries,
        "automatic_provider_retry_windows": (
            automatic_provider_retry_windows
        ),
        "consecutive_provider_retry_windows_at_end": (
            consecutive_provider_retry_windows
        ),
        "maximum_automatic_provider_retry_windows": (
            MAXIMUM_AUTOMATIC_PROVIDER_RETRY_WINDOWS
        ),
        "automatic_provider_retry_window_delay_seconds": (
            AUTOMATIC_PROVIDER_RETRY_WINDOW_DELAY_SECONDS
        ),
        "decisions_recovered_after_provider_retry": (
            decisions_recovered_after_provider_retry
        ),
        "terminal_protocol_failure": terminal_protocol_failure,
        "terminal_provider_failure": terminal_provider_failure,
        "terminal_client_error_type": terminal_client_error_type,
        "terminal_scene_error_type": terminal_scene_error_type,
        "terminal_scene_error_id": terminal_scene_error_id,
        "terminal_scene_error_origin": terminal_scene_error_origin,
        "prompt_sha256": prompt_sha256,
        "private_score_received_from_scene": False,
        "pause_on_client_failure": pause_on_client_failure,
        "resume_available": status in PAUSED_CLIENT_STATUSES,
        "resume_session_index": resume_session_index,
        "resident_episode_preserved": (
            active and status in PAUSED_CLIENT_STATUSES
        ),
        "terminated_by_official_success": terminated_by_official_success,
        "automatic_scene_recoveries": scene_recovery_count,
        "decision_only_recoveries": decision_only_recoveries,
        "checkpoint_restores": checkpoint_restores,
        "same_scene_restarts": same_scene_restarts,
        "maximum_automatic_decision_repairs_per_sequence": (
            MAXIMUM_AUTOMATIC_DECISION_REPAIRS_PER_SEQUENCE
        ),
        "maximum_automatic_checkpoint_restores": (
            MAXIMUM_AUTOMATIC_CHECKPOINT_RESTORES
        ),
        "maximum_automatic_same_scene_resets": (
            MAXIMUM_AUTOMATIC_SAME_SCENE_RESETS
        ),
        "rolled_back_completed_actions": rolled_back_completed_actions,
        "scene_reload_required": scene_reload_required,
        "current_scene_episode_id": scene_episode_id,
        "recovery_action_count_semantics": (
            "actions_sent_to_scene is the committed sequence in the current "
            "scene branch; scene_step_attempts counts all attempted branches"
        ),
        "provider_metrics": _aggregate_provider_metrics(output_dir, calls),
    }
    from lab.physical_model_results import collect_physical_result
    try:
        if getattr(scene, 'scene_run_dir', None) is not None:
            summary = collect_physical_result(summary, scene_run_dir=scene.scene_run_dir,
                episode_id=scene_episode_id, output_dir=output_dir)
    except (OSError, ValueError, TypeError) as exc:
        summary.update(status='FAIL_FINAL_SCORE', final_score_error=str(exc),
            official_success=False, official_task_success=False)
    _write_json(output_dir / "summary.json", summary)
    return summary


def _normalize_single_action_tag_response(
    response: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one exact targetless ``<action>`` into a native tool call.

    This is an explicit provider compatibility seam, not a general text
    parser.  It never extracts C-labels, descriptions, multiple actions, or
    surrounding prose.
    """

    if not isinstance(response, Mapping):
        raise ValueError("action-tag response must be a mapping")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("action-tag response must have one choice")
    choice = choices[0]
    if (
        not isinstance(choice, Mapping)
        or choice.get("finish_reason") != "stop"
        or not isinstance(choice.get("message"), Mapping)
    ):
        raise ValueError("action-tag response did not stop normally")
    message = choice["message"]
    if (
        message.get("role") != "assistant"
        or message.get("tool_calls") != []
        or message.get("refusal") is not None
        or message.get("annotations") is not None
        or message.get("audio") is not None
        or message.get("function_call") is not None
    ):
        raise ValueError("action-tag assistant fields changed")
    reasoning, reasoning_source, content = isolate_native_reasoning(message)
    if not isinstance(content, str):
        raise ValueError("action-tag assistant content is missing")
    match = _SINGLE_ACTION_TAG.fullmatch(content)
    targeted_match = (
        None
        if match is not None
        else _RYNNBRAIN_TARGETED_ACTION_BLOCK.fullmatch(content)
    )
    if match is None and targeted_match is None:
        raise ValueError("assistant content is not one supported action block")
    if match is not None:
        action = match.group(1)
        arguments: dict[str, Any] = {"action": action}
        source_syntax = "single_targetless_action_tag"
    else:
        assert targeted_match is not None
        action = targeted_match.group("action")
        if action not in MARKER_ACTIONS:
            raise ValueError("targeted action block selected a non-Marker action")
        target_description = targeted_match.group("target_description")
        if target_description != target_description.strip():
            raise ValueError("target description has surrounding whitespace")
        arguments = {
            "action": action,
            "target": int(targeted_match.group("target")),
            "target_description": target_description,
        }
        source_syntax = "rynnbrain_targeted_action_block_v1"
    if action not in PUBLIC_TOOL_ACTIONS:
        raise ValueError("action tag selected an unavailable action")

    raw_bytes = _canonical_json_bytes(response)
    normalized = json.loads(raw_bytes)
    normalized_choice = normalized["choices"][0]
    normalized_message = normalized_choice["message"]
    normalized_choice["finish_reason"] = "tool_calls"
    normalized_message["content"] = None
    if reasoning_source == "content_think":
        replay_native_reasoning(
            normalized_message,
            reasoning,
            reasoning_source,
        )
    normalized_message["tool_calls"] = [
        {
            "id": "compat-action-tag-" + hashlib.sha256(raw_bytes).hexdigest()[:24],
            "type": "function",
            "function": {
                "name": "robot_action",
                "arguments": json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        }
    ]
    record = {
        "schema_version": (
            "behavior-hl-provider-action-tag-compatibility/v2-reasoning-source"
        ),
        "source_response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_content_sha256": hashlib.sha256(
            message["content"].encode("utf-8")
        ).hexdigest(),
        "source_finish_reason": "stop",
        "source_syntax": source_syntax,
        "reasoning_source": reasoning_source,
        "normalized_action": action,
        "normalized_target": arguments.get("target"),
        "target_synthesized": False,
        "target_description_synthesized": False,
        "surrounding_text_accepted": False,
    }
    return normalized, record


def _native_response_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    finish_reason: str | None = None
    reasoning_present = False
    tool_call_count: int | None = None
    choices = response.get("choices") if isinstance(response, Mapping) else None
    if isinstance(choices, list) and len(choices) == 1:
        choice = choices[0]
        if isinstance(choice, Mapping):
            if isinstance(choice.get("finish_reason"), str):
                finish_reason = choice["finish_reason"]
            message = choice.get("message")
            if isinstance(message, Mapping):
                try:
                    reasoning, _, _ = isolate_native_reasoning(message)
                except ValueError:
                    reasoning = None
                reasoning_present = isinstance(reasoning, str) and bool(reasoning)
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    tool_call_count = len(tool_calls)
    return {
        "finish_reason": finish_reason,
        "native_reasoning_present": reasoning_present,
        "tool_call_count": tool_call_count,
    }


def _native_protocol_failure_type(response: Mapping[str, Any]) -> str:
    metadata = _native_response_metadata(response)
    if (
        metadata["finish_reason"] == "length"
        and metadata["tool_call_count"] in {None, 0}
        and isinstance(response.get("choices"), list)
        and len(response["choices"]) == 1
        and isinstance(response["choices"][0].get("message"), Mapping)
        and response["choices"][0]["message"].get("tool_calls") in (None, [])
    ):
        return "reasoning_budget_exhausted"
    if (
        metadata["native_reasoning_present"]
        and metadata["tool_call_count"] in {None, 0}
    ):
        return "missing_tool_call_after_reasoning"
    return "invalid_native_tool_response"


def _public_tool_result_from_scene_step(
    action_result: Mapping[str, Any],
    observation_record: Mapping[str, Any],
    *,
    expected_selection_stage: int | None,
    allow_blocked_category: bool = False,
    allow_blocked_detail: bool = False,
) -> PublicToolResult:
    """Normalize old and new resident Marker results at the public boundary."""

    if not isinstance(action_result, Mapping):
        raise ModelClientError("scene action result contract changed")
    kind = observation_record.get("kind")
    status = action_result.get("status")
    if kind == "marker":
        if status not in {"completed", "selection_required"}:
            raise ModelClientError(
                "Marker observation has no selection-required action result"
            )
        if expected_selection_stage not in {1, 2}:
            raise ModelClientError("Marker observation stage is unavailable")
        resident_stage = action_result.get("stage")
        if resident_stage is not None and resident_stage != expected_selection_stage:
            raise ModelClientError("resident Marker stage changed")
        return PublicToolResult(
            status="selection_required",
            stage=expected_selection_stage,
        )
    if status == "selection_required":
        raise ModelClientError(
            "selection_required result has no Marker observation"
        )
    return PublicToolResult(
        status=status,
        body_odometry=action_result.get("body_odometry"),
        blocked_category=(
            action_result.get("blocked_category")
            if allow_blocked_category
            else None
        ),
        blocked_detail=(
            action_result.get("blocked_detail")
            if allow_blocked_category and allow_blocked_detail else None
        ),
    )


def _with_action_decisions_remaining(
    result: PublicToolResult,
    *,
    maximum_action_decisions: int,
    resolved_action_decisions: int,
) -> PublicToolResult:
    remaining = maximum_action_decisions - resolved_action_decisions - 1
    if remaining < 0:
        raise ModelClientError("public action decision budget underflow")
    return replace(result, action_decisions_remaining=remaining)


def _materialize_observation(
    value: Mapping[str, Any],
    *,
    output_dir: Path,
    scene_run_dir: Path,
) -> PublicObservation:
    return materialize_observation(
        value,
        output_dir=output_dir,
        scene_run_dir=scene_run_dir,
        link_or_copy=_link_or_copy,
        hash_file=_sha256_file,
        error_type=ModelClientError,
    )


def _observation_from_artifact(
    value: Mapping[str, Any],
    *,
    output_dir: Path,
) -> PublicObservation:
    return observation_from_artifact(
        value,
        output_dir=output_dir,
        hash_file=_sha256_file,
        error_type=ModelClientError,
    )


def _publish_action_after_view(
    observation: PublicObservation,
    *,
    output_dir: Path,
    step_index: int,
    action_name: str,
    recovery_branch_index: int = 0,
) -> Path:
    return publish_action_after_view(
        observation,
        output_dir=output_dir,
        step_index=step_index,
        action_name=action_name,
        recovery_branch_index=recovery_branch_index,
        hash_file=_sha256_file,
        error_type=ModelClientError,
        publish_file=_link_or_copy,
    )


def _write_campaign_initial_view_audit(
    *,
    output_dir: Path,
    composite_path: Path,
    observation: PublicObservation,
    campaign_binding: Mapping[str, Any],
) -> None:
    """Record, but never hard-gate, the rendered official-reset first view."""

    source = Path(composite_path)
    _write_json(
        Path(output_dir) / "campaign_initial_view.json",
        {
            "schema_version": "behavior-hl-campaign-initial-view-audit/v1",
            "campaign_id": campaign_binding["campaign_id"],
            "campaign_manifest_sha256": campaign_binding[
                "campaign_manifest_sha256"
            ],
            "mode": campaign_binding["campaign_initial_view_mode"],
            "hard_gate": False,
            "comparison_policy": "human_review_or_coarse_pose_audit",
            "pixel_hash_mismatch_invalidates_run": False,
            "composite_path": str(source.resolve()),
            "composite_bytes": source.stat().st_size,
            "composite_sha256": _sha256_file(source),
            "camera_frames": [
                {
                    "camera": image.camera,
                    "sha256": _sha256_file(image.path),
                }
                for image in observation.images
            ],
        },
    )


def _redact_data_urls(value: Any) -> Any:
    if isinstance(value, Mapping):
        if isinstance(value.get("encrypted_content"), str):
            encrypted = value["encrypted_content"]
            return {
                str(key): (
                    {
                        "redacted": True,
                        "encoded_length": len(encrypted),
                        "encoded_sha256": hashlib.sha256(
                            encrypted.encode("utf-8")
                        ).hexdigest(),
                    }
                    if key == "encrypted_content"
                    else _redact_data_urls(item)
                )
                for key, item in value.items()
            }
        if isinstance(value.get("image_url"), str) and value[
            "image_url"
        ].startswith("data:image/png;base64,"):
            url = value["image_url"]
            return {
                str(key): (
                    {
                        "redacted": True,
                        "encoded_length": len(url),
                        "encoded_sha256": hashlib.sha256(
                            url.encode("ascii")
                        ).hexdigest(),
                    }
                    if key == "image_url"
                    else _redact_data_urls(item)
                )
                for key, item in value.items()
            }
        if set(value) == {"url"} and isinstance(value.get("url"), str):
            url = value["url"]
            if url.startswith("data:image/png;base64,"):
                return {
                    "url": "data:image/png;base64,<redacted>",
                    "encoded_length": len(url),
                    "encoded_sha256": hashlib.sha256(url.encode("ascii")).hexdigest(),
                }
            if "X-Amz-Signature=" in url or "Signature=" in url:
                return {
                    "url": "<signed-url-redacted>",
                    "encoded_length": len(url),
                    "encoded_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
                }
        if isinstance(value.get("data"), str) and value.get("type") == "base64":
            data = value["data"]
            return {
                str(key): (
                    {
                        "redacted": True,
                        "encoded_length": len(data),
                        "encoded_sha256": hashlib.sha256(data.encode("ascii")).hexdigest(),
                    }
                    if key == "data"
                    else _redact_data_urls(item)
                )
                for key, item in value.items()
            }
        if isinstance(value.get("signature"), str) and value.get("signature"):
            signature = value["signature"]
            return {
                str(key): (
                    {
                        "redacted": True,
                        "encoded_length": len(signature),
                        "encoded_sha256": hashlib.sha256(
                            signature.encode("utf-8")
                        ).hexdigest(),
                    }
                    if key == "signature"
                    else _redact_data_urls(item)
                )
                for key, item in value.items()
            }
        return {str(key): _redact_data_urls(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_data_urls(item) for item in value]
    return value


def _redact_provider_error_text(value: str, redaction_values: tuple[str, ...] = ()) -> str:
    """Bound and redact provider diagnostics before writing public artifacts."""

    text = str(value)
    credentials = tuple(value for value in redaction_values if value)
    if credentials:
        if "\ufffd" in text:
            return "<redacted undecodable provider diagnostic>"
        try:
            text = json.dumps(json.loads(text), ensure_ascii=True)
        except (ValueError, RecursionError):
            if text.lstrip().startswith(("{", "[", '\"')):
                return "<redacted invalid JSON provider diagnostic>"
    # Normalize JSON escapes and redact before a preview can cut through a key.
    for credential in credentials:
        text = text.replace(json.dumps(credential, ensure_ascii=True)[1:-1], "<redacted>")
        text = text.replace(credential, "<redacted>")
    text = re.sub(
        r"(?i)(\b(?:authorization|x-api-key|api[_ -]?key|access[_ -]?token|password)\b[\"']?\s*[:=]\s*)"
        r"(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|(?:Bearer\s+)?[^\s,;}\]]+)",
        r"\1<redacted>",
        text,
    )
    text = re.sub(r"(?i)\bBearer\s+[^\s,;\"'}\]]+", "Bearer <redacted>", text)
    text = re.sub(r"\bsk-[A-Za-z0-9._-]{8,}\b", "sk-<redacted>", text)
    return text[:MAX_PROVIDER_ERROR_PREVIEW_CHARS]


def _write_private_json(path: Path, value: Any) -> None:
    """Create one immutable 0600 JSON artifact without following a symlink."""

    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    encoded = _canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _read_private_json(path: Path, *, max_bytes: int) -> Mapping[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ModelClientError(f"private JSON artifact is unavailable: {path}")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) & 0o077 or metadata.st_size > max_bytes:
        raise ModelClientError("private JSON artifact permissions or size are invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelClientError("private JSON artifact is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ModelClientError("private JSON artifact is not an object")
    return value


def _responses_replay_path(output_dir: Path, call_id: str) -> Path:
    if not isinstance(call_id, str) or not call_id.strip():
        raise ModelClientError("Responses replay call ID is invalid")
    digest = hashlib.sha256(call_id.encode("utf-8")).hexdigest()
    return Path(output_dir) / "private" / "responses_replay" / f"{digest}.json"


def _historical_chat_call_arguments(
    request: Mapping[str, Any],
) -> dict[str, str]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ModelClientError("native request messages changed")
    result: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise ModelClientError("historical assistant tool call changed")
        call = calls[0]
        call_id = call.get("id") if isinstance(call, Mapping) else None
        if not isinstance(call_id, str) or not call_id.strip() or call_id in result:
            raise ModelClientError("historical assistant call ID changed")
        function = call.get("function")
        arguments = (
            function.get("arguments") if isinstance(function, Mapping) else None
        )
        if not isinstance(arguments, str) or not arguments.strip():
            raise ModelClientError("historical assistant arguments changed")
        result[call_id] = arguments
    return result


def _normalized_chat_call_id(response: Mapping[str, Any]) -> str:
    try:
        call_id = response["choices"][0]["message"]["tool_calls"][0]["id"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelClientError("normalized Responses call linkage changed") from exc
    if not isinstance(call_id, str) or not call_id.strip():
        raise ModelClientError("normalized Responses call ID is invalid")
    return call_id


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_provider_retry_state(output_dir: Path) -> tuple[int, int]:
    path = Path(output_dir) / "provider_retry_state.json"
    if not path.exists():
        return 0, 0
    value = _read_strict_json(path, max_bytes=16 * 1024)
    if (
        set(value)
        != {
            "schema_version",
            "automatic_retry_windows",
            "consecutive_retry_windows",
            "updated_at",
        }
        or value.get("schema_version") != PROVIDER_RETRY_STATE_VERSION
    ):
        raise ModelClientError("provider retry state contract changed")
    total = value.get("automatic_retry_windows")
    consecutive = value.get("consecutive_retry_windows")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total < 0
        or isinstance(consecutive, bool)
        or not isinstance(consecutive, int)
        or not 0 <= consecutive <= MAXIMUM_AUTOMATIC_PROVIDER_RETRY_WINDOWS
        or consecutive > total
    ):
        raise ModelClientError("provider retry state is invalid")
    return total, consecutive


def _write_provider_retry_state(
    output_dir: Path,
    *,
    total: int,
    consecutive: int,
) -> None:
    _write_json(
        Path(output_dir) / "provider_retry_state.json",
        {
            "schema_version": PROVIDER_RETRY_STATE_VERSION,
            "automatic_retry_windows": total,
            "consecutive_retry_windows": consecutive,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _write_pretty_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    temporary.write_bytes(encoded + b"\n")
    temporary.replace(path)


def _tcp_open(port: int) -> bool:
    stream = socket.socket()
    stream.settimeout(0.25)
    try:
        return stream.connect_ex(("127.0.0.1", port)) == 0
    finally:
        stream.close()


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not normalized or len(normalized) > 96:
        raise ValueError("artifact identity is invalid")
    return normalized


def _load_review_instruction(
    path: Path,
    *,
    task_id: int,
    task_name: str,
) -> tuple[str, Mapping[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size > MAX_INSTRUCTION_REVIEW_BYTES:
        raise ModelClientError("instruction review is missing or oversized")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelClientError("instruction review is not strict UTF-8 JSON") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version")
        != "behavior2026.certified53-public-instruction-rewrites/v1"
        or payload.get("status") not in {"DRAFT_FOR_HUMAN_REVIEW", "APPROVED"}
        or not isinstance(payload.get("deterministic_public_audit"), Mapping)
        or payload["deterministic_public_audit"].get("status") != "PASS"
        or not isinstance(payload.get("rewrites"), list)
    ):
        raise ModelClientError("instruction review contract is invalid")
    matches = [
        item
        for item in payload["rewrites"]
        if isinstance(item, Mapping)
        and item.get("task_id") == task_id
        and item.get("task_name") == task_name
    ]
    if len(matches) != 1:
        raise ModelClientError("instruction review has no unique matching task")
    instruction = matches[0].get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ModelClientError("reviewed instruction is invalid")
    normalized = instruction.strip()
    scope = payload.get("scope")
    scope_task_ids = (
        scope.get("task_ids") if isinstance(scope, Mapping) else None
    )
    return normalized, {
        "kind": "explicit_instruction_review",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "review_status": payload["status"],
        "reviewer": payload.get("reviewer"),
        "reviewed_at": payload.get("reviewed_at"),
        "scope_task_ids": (
            list(scope_task_ids) if isinstance(scope_task_ids, list) else None
        ),
        "rewrite_count": len(payload["rewrites"]),
        "deterministic_public_audit": "PASS",
        "selection_authorization": "explicit_cli_instruction_review",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id")
    parser.add_argument("--model")
    parser.add_argument("--wire-model")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--provider-api",
        choices=("chat_completions", "responses", "anthropic_messages"),
    )
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument(
        "--resume-scene-ready-path",
        type=Path,
        help=(
            "Migrate a paused run into a newly loaded matching resident by "
            "restoring its newest portable physical checkpoint."
        ),
    )
    parser.add_argument(
        "--pause-on-client-failure",
        action="store_true",
        help=(
            "Keep the resident episode active and checkpointed after bounded "
            "Provider or model-protocol exhaustion."
        ),
    )
    parser.add_argument("--task-id", type=int, default=25)
    parser.add_argument(
        "--task-name",
        default="clearing_food_from_table_into_fridge",
    )
    parser.add_argument("--task-split", default="train")
    parser.add_argument("--task-instance-id", default="0")
    parser.add_argument("--task-seed", type=int, default=19506)
    parser.add_argument("--instruction", default="Clear the food from the table into the fridge.")
    parser.add_argument("--instruction-review", type=Path)
    parser.add_argument(
        "--maximum-action-decisions",
        type=int,
        default=DEFAULT_MAXIMUM_ACTION_DECISIONS,
        help=(
            "Public episode budget. A complete Marker transaction counts once, "
            "regardless of whether it uses one, two, or three tool calls."
        ),
    )
    parser.add_argument(
        "--maximum-model-calls",
        type=int,
        default=DEFAULT_MAXIMUM_MODEL_CALLS,
        help=(
            "Independent internal safety cap for Provider calls and protocol "
            "retries; it does not consume the public action budget."
        ),
    )
    parser.add_argument("--maximum-provider-attempts", type=int)
    parser.add_argument("--provider-retry-initial-backoff-seconds", type=float)
    parser.add_argument("--provider-retry-maximum-backoff-seconds", type=float)
    parser.add_argument("--provider-retry-maximum-elapsed-seconds", type=float)
    parser.add_argument("--provider-retry-jitter-ratio", type=float)
    parser.add_argument(
        "--history-frames",
        type=int,
        choices=sorted(PHYSICAL_HISTORY_OPTIONS),
        default=PHYSICAL_HISTORY_MAX,
    )
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--maximum-output-tokens",
        type=lambda v: None if v.lower() == 'unlimited' else int(v),
        default=DEFAULT_MAXIMUM_OUTPUT_TOKENS,
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--preserve-thinking", action="store_true")
    parser.add_argument("--require-api-key", action="store_true")
    parser.add_argument(
        "--reasoning-effort",
        choices=sorted(RESPONSES_REASONING_EFFORTS),
    )
    parser.add_argument(
        "--reasoning-context",
        choices=sorted(RESPONSES_REASONING_CONTEXTS),
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=sorted(RESPONSES_REASONING_MODES),
    )
    parser.add_argument(
        "--reasoning-summary",
        choices=sorted(RESPONSES_REASONING_SUMMARIES),
    )
    parser.add_argument(
        "--text-verbosity",
        choices=sorted(RESPONSES_TEXT_VERBOSITIES),
    )
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--responses-image-proxy", action="store_true")
    parser.add_argument("--r2-upload-proxy")
    parser.add_argument(
        "--thinking-tool-choice-auto",
        action="store_true",
        help=(
            "Use provider-native thinking with tool_choice=auto; strict local "
            "validation still requires exactly one robot_action call."
        ),
    )
    parser.add_argument(
        "--action-tag-compatibility",
        action="store_true",
        help=(
            "Convert only an exact single targetless <action> response into "
            "robot_action; requires --thinking-tool-choice-auto."
        ),
    )
    parser.add_argument("--scene-ready-path", type=Path, default=DEFAULT_READY_PATH)
    parser.add_argument("--campaign-manifest", type=Path)
    parser.add_argument(
        "--marker-depth-protocol",
        choices=(RGBD_PROTOCOL_VERSION,),
        help="Expose allowlisted Marker approximate depth; default is RGB-only.",
    )
    parser.add_argument(
        "--overhead-view-mode",
        choices=sorted(OVERHEAD_VIEW_MODES),
        help="Private episode camera mode; default is off.",
    )
    parser.add_argument(
        "--automatic-interaction-approach-enabled",
        action="store_true",
        default=None,
        help="Navigate once to a safe local interaction station when needed.",
    )
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-local-port", type=int)
    parser.add_argument("--ssh-remote-port", type=int)
    return parser.parse_args(argv)


def _provider_retry_policy_from_args(
    args: argparse.Namespace,
) -> ProviderRetryPolicy | None:
    values = (
        args.provider_retry_initial_backoff_seconds,
        args.provider_retry_maximum_backoff_seconds,
        args.provider_retry_maximum_elapsed_seconds,
        args.provider_retry_jitter_ratio,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("provider retry policy requires all timing fields")
    return ProviderRetryPolicy(
        initial_backoff_seconds=values[0],
        maximum_backoff_seconds=values[1],
        maximum_elapsed_seconds=values[2],
        jitter_ratio=values[3],
    )


def _write_resume_invocation(path: Path, value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "experiment_id",
        "model",
        "wire_model",
        "base_url",
        "provider_api",
        "require_api_key",
        "task_id",
        "task_name",
        "task_split",
        "task_instance_id",
        "task_seed",
        "instruction",
        "instruction_provenance",
        "maximum_action_decisions",
        "maximum_model_calls",
        "maximum_provider_attempts",
        "provider_retry_policy",
        "history_frames",
        "timeout_seconds",
        "maximum_output_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "preserve_thinking",
        "reasoning_effort",
        "reasoning_context",
        "reasoning_mode",
        "reasoning_summary",
        "text_verbosity",
        "api_key_file",
        "thinking_tool_choice_auto",
        "action_tag_compatibility",
        "scene_ready_path",
        "ssh_host",
        "ssh_local_port",
        "ssh_remote_port",
        "pause_on_client_failure",
        "campaign_manifest_path",
        "campaign_manifest_sha256",
        "marker_depth_protocol",
        "overhead_view_mode",
        "automatic_interaction_approach_enabled",
        "responses_image_proxy",
        "r2_upload_proxy",
    }
    if set(value) != expected or value.get("schema_version") != RESUME_INVOCATION_VERSION:
        raise ValueError("resume invocation fields changed")
    _write_json(path, value)


def _load_resume_invocation(path: Path) -> Mapping[str, Any]:
    value = _read_strict_json(path, max_bytes=MAX_RESUME_ARTIFACT_BYTES)
    common = {
        "schema_version",
        "experiment_id",
        "model",
        "base_url",
        "task_id",
        "task_name",
        "task_split",
        "task_instance_id",
        "task_seed",
        "instruction",
        "instruction_provenance",
        "maximum_model_calls",
        "history_frames",
        "timeout_seconds",
        "maximum_output_tokens",
        "api_key_file",
        "thinking_tool_choice_auto",
        "action_tag_compatibility",
        "scene_ready_path",
        "ssh_host",
        "ssh_local_port",
        "ssh_remote_port",
        "pause_on_client_failure",
    }
    version = value.get("schema_version")
    if version == RESUME_INVOCATION_VERSION_V1:
        expected = common
    elif version == RESUME_INVOCATION_VERSION_V2:
        expected = common | {"campaign_manifest_path", "campaign_manifest_sha256"}
    elif version == RESUME_INVOCATION_VERSION_V3:
        expected = common | {
            "campaign_manifest_path",
            "campaign_manifest_sha256",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "preserve_thinking",
        }
    elif version == RESUME_INVOCATION_VERSION_V4:
        expected = common | {
            "campaign_manifest_path",
            "campaign_manifest_sha256",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "preserve_thinking",
            "provider_api",
            "require_api_key",
            "reasoning_effort",
            "reasoning_context",
            "reasoning_mode",
            "text_verbosity",
        }
    elif version == RESUME_INVOCATION_VERSION_V5:
        expected = common | {
            "campaign_manifest_path",
            "campaign_manifest_sha256",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "preserve_thinking",
            "provider_api",
            "require_api_key",
            "reasoning_effort",
            "reasoning_context",
            "reasoning_mode",
            "text_verbosity",
            "maximum_provider_attempts",
        }
    elif version == RESUME_INVOCATION_VERSION_V6:
        expected = common | {
            "campaign_manifest_path",
            "campaign_manifest_sha256",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "preserve_thinking",
            "provider_api",
            "require_api_key",
            "reasoning_effort",
            "reasoning_context",
            "reasoning_mode",
            "reasoning_summary",
            "text_verbosity",
            "maximum_provider_attempts",
        }
    elif version == RESUME_INVOCATION_VERSION_V7:
        expected = common | {
            "campaign_manifest_path",
            "campaign_manifest_sha256",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "preserve_thinking",
            "provider_api",
            "require_api_key",
            "reasoning_effort",
            "reasoning_context",
            "reasoning_mode",
            "reasoning_summary",
            "text_verbosity",
            "maximum_provider_attempts",
            "provider_retry_policy",
        }
    elif version == RESUME_INVOCATION_VERSION_V8:
        expected = common | {
            "campaign_manifest_path",
            "campaign_manifest_sha256",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "preserve_thinking",
            "provider_api",
            "require_api_key",
            "reasoning_effort",
            "reasoning_context",
            "reasoning_mode",
            "reasoning_summary",
            "text_verbosity",
            "maximum_provider_attempts",
            "provider_retry_policy",
            "maximum_action_decisions",
        }
    elif version == RESUME_INVOCATION_VERSION_V9:
        expected = common | {
            "campaign_manifest_path",
            "campaign_manifest_sha256",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "preserve_thinking",
            "provider_api",
            "require_api_key",
            "reasoning_effort",
            "reasoning_context",
            "reasoning_mode",
            "reasoning_summary",
            "text_verbosity",
            "maximum_provider_attempts",
            "provider_retry_policy",
            "maximum_action_decisions",
            "wire_model",
        }
    elif version == RESUME_INVOCATION_VERSION_V10:
        expected = common | {
            "campaign_manifest_path",
            "campaign_manifest_sha256",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "preserve_thinking",
            "provider_api",
            "require_api_key",
            "reasoning_effort",
            "reasoning_context",
            "reasoning_mode",
            "reasoning_summary",
            "text_verbosity",
            "maximum_provider_attempts",
            "provider_retry_policy",
            "maximum_action_decisions",
            "wire_model",
            "marker_depth_protocol",
        }
    elif version == RESUME_INVOCATION_VERSION_V11:
        expected = common | {
            "campaign_manifest_path",
            "campaign_manifest_sha256",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "preserve_thinking",
            "provider_api",
            "require_api_key",
            "reasoning_effort",
            "reasoning_context",
            "reasoning_mode",
            "reasoning_summary",
            "text_verbosity",
            "maximum_provider_attempts",
            "provider_retry_policy",
            "maximum_action_decisions",
            "wire_model",
            "marker_depth_protocol",
            "overhead_view_mode",
        }
    elif version == RESUME_INVOCATION_VERSION_V12:
        expected = common | {
            "campaign_manifest_path",
            "campaign_manifest_sha256",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "preserve_thinking",
            "provider_api",
            "require_api_key",
            "reasoning_effort",
            "reasoning_context",
            "reasoning_mode",
            "reasoning_summary",
            "text_verbosity",
            "maximum_provider_attempts",
            "provider_retry_policy",
            "maximum_action_decisions",
            "wire_model",
            "marker_depth_protocol",
            "overhead_view_mode",
            "automatic_interaction_approach_enabled",
        }
    else:
        expected = common | {
            "campaign_manifest_path",
            "campaign_manifest_sha256",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "preserve_thinking",
            "provider_api",
            "require_api_key",
            "reasoning_effort",
            "reasoning_context",
            "reasoning_mode",
            "reasoning_summary",
            "text_verbosity",
            "maximum_provider_attempts",
            "provider_retry_policy",
            "maximum_action_decisions",
            "wire_model",
            "marker_depth_protocol",
            "overhead_view_mode",
            "automatic_interaction_approach_enabled",
            "responses_image_proxy",
            "r2_upload_proxy",
        }
    if (
        set(value) != expected
        or version not in {
            RESUME_INVOCATION_VERSION_V1,
            RESUME_INVOCATION_VERSION_V2,
            RESUME_INVOCATION_VERSION_V3,
            RESUME_INVOCATION_VERSION_V4,
            RESUME_INVOCATION_VERSION_V5,
            RESUME_INVOCATION_VERSION_V6,
            RESUME_INVOCATION_VERSION_V7,
            RESUME_INVOCATION_VERSION_V8,
            RESUME_INVOCATION_VERSION_V9,
            RESUME_INVOCATION_VERSION_V10,
            RESUME_INVOCATION_VERSION_V11,
            RESUME_INVOCATION_VERSION_V12,
            RESUME_INVOCATION_VERSION,
        }
        or value.get("pause_on_client_failure") is not True
        or not isinstance(value.get("instruction_provenance"), Mapping)
    ):
        raise ModelClientError("resume invocation contract changed")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    provider_attempts_override = args.maximum_provider_attempts
    provider_retry_policy_override = _provider_retry_policy_from_args(args)
    resume_checkpoint: Mapping[str, Any] | None = None
    persistent_resume: Mapping[str, Any] | None = None
    if args.resume_run is not None:
        if any(
            value is not None
            for value in (
                args.experiment_id,
                args.model,
                args.wire_model,
                args.base_url,
                args.provider_api,
                args.instruction_review,
                args.ssh_host,
                args.ssh_local_port,
                args.ssh_remote_port,
                args.campaign_manifest,
                args.temperature,
                args.top_p,
                args.top_k,
                args.repetition_penalty,
                args.reasoning_effort,
                args.reasoning_context,
                args.reasoning_mode,
                args.reasoning_summary,
                args.text_verbosity,
                args.marker_depth_protocol,
                args.overhead_view_mode,
                args.automatic_interaction_approach_enabled,
                args.r2_upload_proxy,
            )
        ) or args.preserve_thinking or args.require_api_key or args.responses_image_proxy:
            raise ValueError("resume-run cannot be combined with fresh-run identity")
        output_dir = Path(args.resume_run).resolve()
        runs_root = DEFAULT_RUNS_ROOT.resolve()
        if (
            not output_dir.is_dir()
            or output_dir.is_symlink()
            or not output_dir.is_relative_to(runs_root)
            or output_dir.parent.name != "incomplete"
        ):
            raise ModelClientError("resume run is not a canonical incomplete run")
        requested_resume_scene = args.resume_scene_ready_path
        invocation = _load_resume_invocation(
            output_dir / "resume_invocation.json"
        )
        resume_checkpoint = _load_resume_checkpoint(
            output_dir / "resume_checkpoint.json",
            allow_active_for_persistent_migration=(
                requested_resume_scene is not None
            ),
        )
        args.experiment_id = invocation["experiment_id"]
        args.model = invocation["model"]
        args.wire_model = invocation.get("wire_model", invocation["model"])
        args.marker_depth_protocol = invocation.get("marker_depth_protocol")
        args.overhead_view_mode = invocation.get("overhead_view_mode", "off")
        args.automatic_interaction_approach_enabled = invocation.get(
            "automatic_interaction_approach_enabled", False
        )
        args.responses_image_proxy = invocation.get(
            "responses_image_proxy", False
        )
        args.r2_upload_proxy = invocation.get("r2_upload_proxy")
        args.base_url = invocation["base_url"]
        if invocation["schema_version"] in {
            RESUME_INVOCATION_VERSION_V4,
            RESUME_INVOCATION_VERSION_V5,
            RESUME_INVOCATION_VERSION_V6,
            RESUME_INVOCATION_VERSION_V7,
            RESUME_INVOCATION_VERSION_V8,
            RESUME_INVOCATION_VERSION_V9,
            RESUME_INVOCATION_VERSION_V10,
            RESUME_INVOCATION_VERSION_V11,
            RESUME_INVOCATION_VERSION_V12,
            RESUME_INVOCATION_VERSION,
        }:
            args.provider_api = invocation["provider_api"]
            args.require_api_key = invocation["require_api_key"]
            args.reasoning_effort = invocation["reasoning_effort"]
            args.reasoning_context = invocation["reasoning_context"]
            args.reasoning_mode = invocation["reasoning_mode"]
            args.reasoning_summary = (
                invocation["reasoning_summary"]
                if invocation["schema_version"] in {
                    RESUME_INVOCATION_VERSION_V6,
                    RESUME_INVOCATION_VERSION_V7,
                    RESUME_INVOCATION_VERSION_V8,
                    RESUME_INVOCATION_VERSION_V9,
                    RESUME_INVOCATION_VERSION_V10,
                    RESUME_INVOCATION_VERSION_V11,
                    RESUME_INVOCATION_VERSION_V12,
                    RESUME_INVOCATION_VERSION,
                }
                else None
            )
            args.text_verbosity = invocation["text_verbosity"]
        else:
            args.provider_api = "chat_completions"
            args.require_api_key = False
            args.reasoning_effort = None
            args.reasoning_context = None
            args.reasoning_mode = None
            args.reasoning_summary = None
            args.text_verbosity = None
        args.maximum_provider_attempts = (
            provider_attempts_override
            if provider_attempts_override is not None
            else (
                invocation["maximum_provider_attempts"]
                if invocation["schema_version"] in {
                    RESUME_INVOCATION_VERSION_V5,
                    RESUME_INVOCATION_VERSION_V6,
                    RESUME_INVOCATION_VERSION_V7,
                    RESUME_INVOCATION_VERSION_V8,
                    RESUME_INVOCATION_VERSION_V9,
                    RESUME_INVOCATION_VERSION_V10,
                    RESUME_INVOCATION_VERSION_V11,
                    RESUME_INVOCATION_VERSION_V12,
                    RESUME_INVOCATION_VERSION,
                }
                else DEFAULT_MAXIMUM_PROVIDER_ATTEMPTS_PER_PROTOCOL_ATTEMPT
            )
        )
        provider_retry_policy = (
            provider_retry_policy_override
            if provider_retry_policy_override is not None
            else ProviderRetryPolicy.from_mapping(
                invocation.get("provider_retry_policy")
            )
        )
        args.task_id = invocation["task_id"]
        args.task_name = invocation["task_name"]
        args.task_split = invocation["task_split"]
        args.task_instance_id = invocation["task_instance_id"]
        args.task_seed = invocation["task_seed"]
        args.instruction = invocation["instruction"]
        if invocation["schema_version"] in {
            RESUME_INVOCATION_VERSION_V8,
            RESUME_INVOCATION_VERSION_V9,
            RESUME_INVOCATION_VERSION_V10,
            RESUME_INVOCATION_VERSION_V11,
            RESUME_INVOCATION_VERSION_V12,
            RESUME_INVOCATION_VERSION,
        }:
            args.maximum_action_decisions = invocation[
                "maximum_action_decisions"
            ]
            args.maximum_model_calls = invocation["maximum_model_calls"]
        else:
            # Before v8 the only limit was named maximum_model_calls even
            # though users configured it as the episode's 300-action budget.
            # Preserve that public budget and derive a separate internal cap.
            args.maximum_action_decisions = invocation["maximum_model_calls"]
            args.maximum_model_calls = min(
                MAXIMUM_MODEL_CALLS,
                args.maximum_action_decisions * 10,
            )
        args.history_frames = invocation["history_frames"]
        args.timeout_seconds = invocation["timeout_seconds"]
        args.maximum_output_tokens = invocation["maximum_output_tokens"]
        if invocation["schema_version"] in {
            RESUME_INVOCATION_VERSION_V3,
            RESUME_INVOCATION_VERSION_V4,
            RESUME_INVOCATION_VERSION_V5,
            RESUME_INVOCATION_VERSION_V6,
            RESUME_INVOCATION_VERSION_V7,
            RESUME_INVOCATION_VERSION_V8,
            RESUME_INVOCATION_VERSION_V9,
            RESUME_INVOCATION_VERSION_V10,
            RESUME_INVOCATION_VERSION_V11,
            RESUME_INVOCATION_VERSION_V12,
            RESUME_INVOCATION_VERSION,
        }:
            args.temperature = invocation["temperature"]
            args.top_p = invocation["top_p"]
            args.top_k = invocation["top_k"]
            args.repetition_penalty = invocation["repetition_penalty"]
            args.preserve_thinking = invocation["preserve_thinking"]
        else:
            args.temperature = 0.1
            args.top_p = 0.9
            args.top_k = None
            args.repetition_penalty = None
            args.preserve_thinking = False
        args.api_key_file = (
            None
            if invocation["api_key_file"] is None
            else Path(invocation["api_key_file"])
        )
        args.thinking_tool_choice_auto = invocation["thinking_tool_choice_auto"]
        args.action_tag_compatibility = invocation["action_tag_compatibility"]
        args.scene_ready_path = (
            Path(invocation["scene_ready_path"])
            if requested_resume_scene is None
            else Path(requested_resume_scene)
        )
        if requested_resume_scene is not None:
            persistent_resume = _load_latest_persistent_model_checkpoint(
                output_dir=output_dir,
                maximum_action_count=resume_checkpoint["next_sequence"],
            )
        args.ssh_host = invocation["ssh_host"]
        args.ssh_local_port = invocation["ssh_local_port"]
        args.ssh_remote_port = invocation["ssh_remote_port"]
        args.pause_on_client_failure = True
        args.campaign_manifest = (
            None
            if invocation.get("campaign_manifest_path") is None
            else Path(str(invocation["campaign_manifest_path"]))
        )
        instruction = invocation["instruction"]
        instruction_provenance = dict(invocation["instruction_provenance"])
    else:
        if args.resume_scene_ready_path is not None:
            raise ValueError(
                "resume-scene-ready-path requires --resume-run"
            )
        if args.experiment_id is None or args.model is None or args.base_url is None:
            raise ValueError("fresh model run requires experiment-id, model, and base-url")
        args.wire_model = args.wire_model or args.model
        args.provider_api = args.provider_api or "chat_completions"
        responses_only_values = (
            args.reasoning_context,
            args.reasoning_mode,
            args.reasoning_summary,
            args.text_verbosity,
        )
        if args.provider_api == "responses" and (
            args.reasoning_effort is None
            or any(value is None for value in responses_only_values)
        ):
            raise ValueError(
                "Responses model requires all frozen reasoning and text parameters"
            )
        if args.provider_api == "anthropic_messages":
            if args.reasoning_effort not in ANTHROPIC_EFFORTS:
                raise ValueError("Anthropic Messages requires a frozen effort")
            if any(value is not None for value in responses_only_values):
                raise ValueError(
                    "GPT Responses parameters cannot be used with Anthropic Messages"
                )
        if args.provider_api == "chat_completions" and any(
            value is not None for value in responses_only_values
        ):
            raise ValueError(
                "Responses parameters require --provider-api responses"
            )
        if (
            args.provider_api == "chat_completions"
            and args.reasoning_effort is not None
            and args.reasoning_effort not in CHAT_COMPLETIONS_REASONING_EFFORTS
        ):
            raise ValueError("Chat Completions reasoning effort is invalid")
        instruction = args.instruction
        instruction_provenance: Mapping[str, Any] = {"kind": "task_config"}
        if args.instruction_review is not None:
            instruction, instruction_provenance = _load_review_instruction(
                args.instruction_review,
                task_id=args.task_id,
                task_name=_safe_component(args.task_name),
            )
        args.temperature = 0.1 if args.temperature is None else args.temperature
        args.top_p = 0.9 if args.top_p is None else args.top_p
        args.maximum_provider_attempts = (
            DEFAULT_MAXIMUM_PROVIDER_ATTEMPTS_PER_PROTOCOL_ATTEMPT
            if args.maximum_provider_attempts is None
            else args.maximum_provider_attempts
        )
        provider_retry_policy = (
            provider_retry_policy_override or ProviderRetryPolicy()
        )
        args.overhead_view_mode = args.overhead_view_mode or "off"
        args.automatic_interaction_approach_enabled = bool(
            args.automatic_interaction_approach_enabled
        )
    if args.responses_image_proxy and args.provider_api not in {"responses", "chat_completions"}:
        raise ValueError("Image proxy requires provider-api responses or chat_completions")
    if args.r2_upload_proxy is not None and not args.responses_image_proxy:
        raise ValueError("R2 upload proxy requires the Responses image proxy")
    if not 1 <= args.maximum_action_decisions <= MAXIMUM_ACTION_DECISIONS:
        raise ValueError("maximum-action-decisions must be in [1, 150]")
    if not 1 <= args.maximum_model_calls <= MAXIMUM_MODEL_CALLS:
        raise ValueError("maximum-model-calls must be in [1, 30000]")
    if not 1 <= args.maximum_provider_attempts <= 10:
        raise ValueError("maximum-provider-attempts must be in [1, 10]")
    experiment_id = _safe_component(args.experiment_id)
    model_component = _safe_component(args.model)
    task_name = _safe_component(args.task_name)
    if (
        isinstance(args.task_id, bool)
        or not 0 <= args.task_id <= 999
        or not isinstance(args.task_split, str)
        or not args.task_split.strip()
        or not isinstance(args.task_instance_id, str)
        or not args.task_instance_id.strip()
        or isinstance(args.task_seed, bool)
        or not 0 <= args.task_seed <= 0xFFFF_FFFF
    ):
        raise ValueError("model task identity is invalid")
    task_artifact_key = f"task{args.task_id}_{task_name}"
    campaign_binding = None
    if args.campaign_manifest is not None:
        try:
            campaign_binding = bind_campaign_run(
                args.campaign_manifest,
                experiment_id=experiment_id,
                provider_model=args.model,
                task_id=args.task_id,
                task_name=task_name,
                task_split=args.task_split,
                task_instance_id=args.task_instance_id,
                task_seed=args.task_seed,
                scene_ready_path=(
                    Path(invocation["scene_ready_path"])
                    if persistent_resume is not None
                    else args.scene_ready_path
                ),
            )
        except CampaignManifestError as exc:
            raise ModelClientError(str(exc)) from exc
        if (
            resume_checkpoint is not None
            and invocation.get("campaign_manifest_sha256")
            != campaign_binding.manifest_sha256
        ):
            raise ModelClientError("resume campaign manifest identity changed")
    if resume_checkpoint is None:
        output_dir = allocate_incomplete_run(
            DEFAULT_RUNS_ROOT,
            model=model_component,
            task=task_artifact_key,
            experiment_id=experiment_id,
        )
        _write_resume_invocation(
            output_dir / "resume_invocation.json",
            {
                "schema_version": RESUME_INVOCATION_VERSION,
                "experiment_id": experiment_id,
                "model": args.model,
                "wire_model": args.wire_model,
                "base_url": args.base_url,
                "provider_api": args.provider_api,
                "require_api_key": args.require_api_key,
                "task_id": args.task_id,
                "task_name": task_name,
                "task_split": args.task_split,
                "task_instance_id": args.task_instance_id,
                "task_seed": args.task_seed,
                "instruction": instruction,
                "instruction_provenance": dict(instruction_provenance),
                "maximum_action_decisions": args.maximum_action_decisions,
                "maximum_model_calls": args.maximum_model_calls,
                "maximum_provider_attempts": args.maximum_provider_attempts,
                "provider_retry_policy": provider_retry_policy.to_record(),
                "history_frames": args.history_frames,
                "timeout_seconds": args.timeout_seconds,
                "maximum_output_tokens": args.maximum_output_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "preserve_thinking": args.preserve_thinking,
                "reasoning_effort": args.reasoning_effort,
                "reasoning_context": args.reasoning_context,
                "reasoning_mode": args.reasoning_mode,
                "reasoning_summary": args.reasoning_summary,
                "text_verbosity": args.text_verbosity,
                "api_key_file": (
                    None
                    if args.api_key_file is None
                    else str(Path(args.api_key_file).resolve())
                ),
                "thinking_tool_choice_auto": args.thinking_tool_choice_auto,
                "action_tag_compatibility": args.action_tag_compatibility,
                "scene_ready_path": str(Path(args.scene_ready_path).resolve()),
                "ssh_host": args.ssh_host,
                "ssh_local_port": args.ssh_local_port,
                "ssh_remote_port": args.ssh_remote_port,
                "pause_on_client_failure": args.pause_on_client_failure,
                "campaign_manifest_path": (
                    None if campaign_binding is None else campaign_binding.manifest_path
                ),
                "campaign_manifest_sha256": (
                    None if campaign_binding is None else campaign_binding.manifest_sha256
                ),
                "marker_depth_protocol": args.marker_depth_protocol,
                "overhead_view_mode": args.overhead_view_mode,
                "automatic_interaction_approach_enabled": (
                    args.automatic_interaction_approach_enabled
                ),
                "responses_image_proxy": args.responses_image_proxy,
                "r2_upload_proxy": args.r2_upload_proxy,
            },
        )
    else:
        expected_output = (
            DEFAULT_RUNS_ROOT.resolve()
            / model_component
            / task_artifact_key
            / "incomplete"
            / experiment_id
        )
        if output_dir != expected_output:
            raise ModelClientError("resume run path does not match its identity")
    forward: SshLocalForward | None = None
    transport_dir = output_dir / "transport"
    if resume_checkpoint is not None:
        transport_dir = (
            transport_dir
            / f"resume_{resume_checkpoint['resume_session_index'] + 1:04d}"
        )
    image_transport_handle: Any | None = None
    image_transport_record: dict[str, Any] = {
        "enabled": args.responses_image_proxy,
        "mode": "r2_presigned_get_url" if args.responses_image_proxy else None,
        "started": False,
        "r2_upload_proxy_enabled": args.r2_upload_proxy is not None,
    }
    cleanup: Mapping[str, Any] | None = None
    try:
        if args.ssh_host is not None:
            if args.ssh_local_port is None or args.ssh_remote_port is None:
                raise ValueError("SSH model routing requires local and remote ports")
            expected = f"http://127.0.0.1:{args.ssh_local_port}/v1"
            if args.base_url != expected:
                raise ValueError("SSH model base URL does not match the owned loopback")
            forward = SshLocalForward(
                ssh_host=args.ssh_host,
                local_port=args.ssh_local_port,
                remote_port=args.ssh_remote_port,
                log_path=transport_dir / "ssh.log",
            )
            _write_json(transport_dir / "manifest.json", forward.start())
        effective_base_url = args.base_url
        if args.responses_image_proxy:
            if R2_IMAGE_TRANSPORT_ENV is None:
                raise ValueError("Private upload transport is unavailable in the public runner")
            from lab.responses_image_proxy import start_responses_image_proxy

            prefix = "action20/" + experiment_id + "/" + hashlib.sha256(
                f"{output_dir}:{time.time_ns()}".encode("utf-8")
            ).hexdigest()[:16]
            image_transport_handle = start_responses_image_proxy(
                upstream_url=args.base_url.rstrip("/") + (
                    "/responses" if args.provider_api == "responses" else "/chat/completions"
                ),
                run_dir=transport_dir / "image_proxy",
                r2_env=R2_IMAGE_TRANSPORT_ENV,
                r2_prefix=prefix,
                r2_upload_proxy=args.r2_upload_proxy,
                signed_url_seconds=R2_SIGNED_URL_SECONDS,
                upstream_read_timeout=args.timeout_seconds,
            )
            effective_base_url = image_transport_handle.base_url
            image_transport_record["started"] = True
        provider = OpenAICompatibleProvider(
            model=args.model,
            wire_model=args.wire_model,
            base_url=effective_base_url,
            timeout_seconds=args.timeout_seconds,
            api_key_file=args.api_key_file,
            maximum_output_tokens=args.maximum_output_tokens,
            thinking_tool_choice_auto=args.thinking_tool_choice_auto,
            action_tag_compatibility=args.action_tag_compatibility,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            preserve_thinking=args.preserve_thinking,
            provider_api=args.provider_api,
            require_api_key=args.require_api_key,
            reasoning_effort=args.reasoning_effort,
            reasoning_context=args.reasoning_context,
            reasoning_mode=args.reasoning_mode,
            reasoning_summary=args.reasoning_summary,
            text_verbosity=args.text_verbosity,
        )
        provider.image_transport = dict(image_transport_record)
        scene = SceneClient(
            ready_path=args.scene_ready_path,
            expected_task_id=args.task_id,
            expected_task_name=task_name,
            expected_split=args.task_split,
            expected_instance_id=args.task_instance_id,
        )
        summary = run_model_episode(
            scene=scene,
            provider=provider,
            experiment_id=experiment_id,
            instruction=instruction,
            task_id=args.task_id,
            task_name=task_name,
            task_split=args.task_split,
            task_instance_id=args.task_instance_id,
            task_seed=args.task_seed,
            instruction_provenance=instruction_provenance,
            maximum_action_decisions=args.maximum_action_decisions,
            maximum_model_calls=args.maximum_model_calls,
            maximum_provider_attempts_per_protocol_attempt=(
                args.maximum_provider_attempts
            ),
            provider_retry_policy=provider_retry_policy.to_record(),
            physical_history_max=args.history_frames,
            output_dir=output_dir,
            pause_on_client_failure=args.pause_on_client_failure,
            resume_checkpoint=resume_checkpoint,
            persistent_resume=persistent_resume,
            campaign_binding=(
                None if campaign_binding is None else campaign_binding.to_record()
            ),
            marker_depth_protocol=args.marker_depth_protocol,
            overhead_view_mode=args.overhead_view_mode,
            automatic_interaction_approach_enabled=(
                args.automatic_interaction_approach_enabled
            ),
        )
    finally:
        if image_transport_handle is not None:
            image_transport_handle.close()
            metrics_path = (
                transport_dir / "image_proxy" / "transport" / "summary.json"
            )
            try:
                image_transport_record["metrics"] = _read_strict_json(
                    metrics_path, max_bytes=256 * 1024
                )
            except Exception:
                image_transport_record["metrics"] = {
                    "summary_unavailable": True
                }
            _write_json(
                transport_dir / "image_transport.json",
                image_transport_record,
            )
        if forward is not None:
            cleanup = forward.stop()
            _write_json(transport_dir / "cleanup.json", cleanup)
    if summary["status"] not in PAUSED_CLIENT_STATUSES:
        output_dir = finalize_completed_run(output_dir, summary=summary)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary, "transport_cleanup": cleanup, "image_transport": image_transport_record}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
