"""Strict, side-effect-free loading of the public JSON configuration."""

import json
import os
from pathlib import Path
from typing import Any

from .tasks import profile_for

SCHEMA = "esi-benchmark/config-v1"
FIELDS = {
    "assets": {"root", "source_roots"},
    "run": {"task_id", "family", "output_root", "history_frames"},
    "model": {"endpoint", "model", "api_key_env", "api", "timeout", "maximum_output_tokens",
              "temperature", "top_p", "top_k", "repetition_penalty", "preserve_thinking",
              "reasoning_effort", "reasoning_context", "reasoning_mode", "reasoning_summary", "text_verbosity"},
    "runtime": {"workspace", "python", "behavior_root", "data_root", "toolchain_root",
                "overlay_root", "host_dlsym_shim", "gpu", "expected_gpu_uuid",
                "scene_load_timeout", "max_scene_loads", "server_port", "accept_eula", "vulkan_icd", "mdl_root"},
}
PATH_FIELDS = {"root", "output_root", "workspace", "python", "behavior_root", "data_root",
               "toolchain_root", "overlay_root", "host_dlsym_shim", "vulkan_icd", "mdl_root"}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate config field: {key}")
        result[key] = value
    return result


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Config must be an object")
    allowed = {"schema_version", "profile", "mode", *FIELDS}
    if set(config) - allowed:
        raise ValueError(f"Unknown config fields: {sorted(set(config) - allowed)}")
    if config.get("schema_version") != SCHEMA:
        raise ValueError(f"schema_version must be {SCHEMA}")
    profile = config.get("profile")
    if not isinstance(profile, str):
        raise ValueError("profile must name a frozen execution profile")
    frozen = profile_for(profile)
    if "bench_revision" not in frozen or "runtime_revision" not in frozen:
        raise ValueError(f"Profile has no executable source identity: {profile}")
    if config.get("mode") not in {"replay", "live"}:
        raise ValueError("mode must be replay or live")
    for section, fields in FIELDS.items():
        value = config.get(section, {})
        if not isinstance(value, dict) or set(value) - fields:
            raise ValueError(f"Invalid {section} fields")
        for key, item in value.items():
            if key in PATH_FIELDS and (not isinstance(item, str) or not item.strip()):
                raise ValueError(f"{section}.{key} must be a nonempty path string")
    run = config.get("run", {})
    for key in ("task_id", "family", "output_root"):
        if not isinstance(run.get(key), str) or not run[key].strip():
            raise ValueError(f"run.{key} is required")
    if type(run.get("history_frames")) is not int or run["history_frames"] not in {5, 20}:
        raise ValueError("run.history_frames must be 5 or 20")
    roots = config.get("assets", {}).get("source_roots", {})
    if not isinstance(roots, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in roots.items()):
        raise ValueError("assets.source_roots must map logical names to paths")
    if config["mode"] == "live":
        if not config.get("assets", {}).get("root"):
            raise ValueError("Live runs require assets.root")
        runtime = config.get("runtime", {})
        for key in ("workspace", "python", "behavior_root", "data_root", "toolchain_root", "overlay_root", "mdl_root", "expected_gpu_uuid"):
            if not isinstance(runtime.get(key), str) or not runtime[key]:
                raise ValueError(f"Live runs require runtime.{key}")
        if type(runtime.get("gpu")) is not int or runtime["gpu"] < 0:
            raise ValueError("runtime.gpu must be a nonnegative physical device index")
        if not runtime["expected_gpu_uuid"].startswith("GPU-"):
            raise ValueError("runtime.expected_gpu_uuid must be the explicitly owned GPU UUID")
        if type(runtime.get("server_port")) is not int or not 1024 <= runtime["server_port"] <= 65535:
            raise ValueError("runtime.server_port must be an explicitly selected unprivileged port")
        if runtime.get("accept_eula") is not True:
            raise ValueError("Read the external simulator terms, then set runtime.accept_eula=true to use Live")
        if runtime.get("max_scene_loads") != 1 or type(runtime.get("max_scene_loads")) is not int:
            raise ValueError("runtime.max_scene_loads must be 1; automatic scene retries are disabled")
        timeout = runtime.get("scene_load_timeout")
        if type(timeout) not in (int, float) or not 1 <= timeout <= 7200:
            raise ValueError("runtime.scene_load_timeout must be in [1, 7200] seconds")
        from .models import validate_model_config
        validate_model_config(config.get("model", {}), profile=profile)
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Config must be a JSON file no larger than 1 MiB")
    config = json.loads(path.read_text(), object_pairs_hook=_object,
                        parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {v}")))
    validate_config(config)
    for section in FIELDS:
        for key, value in config.get(section, {}).items():
            if key in PATH_FIELDS:
                target = Path(value).expanduser()
                joined = path.parent / target
                # Resolving the executable symlink bypasses a virtual environment.
                config[section][key] = os.path.abspath(joined) if section == "runtime" and key == "python" else str(joined.resolve())
        if section == "assets" and "source_roots" in config.get(section, {}):
            config[section]["source_roots"] = {k: str((path.parent / Path(v).expanduser()).resolve())
                                                  for k, v in config[section]["source_roots"].items()}
    return config
