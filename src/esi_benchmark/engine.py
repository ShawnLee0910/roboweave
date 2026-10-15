"""Fixed JSON boundary for one frozen evaluation operation per child process."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

from .profiles import DEFAULT_PROFILE, source_roots

SCHEMA = "esi-benchmark/engine-v1"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
SAFE_MODEL_INTERRUPT_PHASES = frozenset({
    "STARTING_PROVIDER_CALL", "WAITING_PROVIDER", "RETRYING_PROVIDER", "AGENT_CLI_RUNNING",
})
_L2_PROFILE = "l2-collection37"
_OUTER_WATCHDOG_SECONDS = 4 * 60 * 60
_FIELDS = {
    "inspect": set(),
    "validate_model": {"model_config"},
    "replay": {"config", "output_dir"},
    "score": {"record", "process_directory"},
    "episode": {"model_config", "task", "ready_path", "runtime_root", "output_dir",
                "experiment_id", "history_frames", "output_reserved"},
    "task_info": {"inputs"},
}


def validate_request(request):
    if (not isinstance(request, dict)
            or set(request) != {"schema_version", "operation", "profile", "payload"}
            or request["schema_version"] != SCHEMA):
        raise ValueError("Invalid engine request envelope")
    operation, profile, payload = (request[key] for key in ("operation", "profile", "payload"))
    if not isinstance(operation, str) or operation not in _FIELDS:
        raise ValueError("Unknown engine operation")
    if not isinstance(profile, str) or not profile:
        raise ValueError("Engine profile must be an explicit registered name")
    source_roots(profile)
    if not isinstance(payload, dict) or set(payload) != _FIELDS[operation]:
        raise ValueError("Invalid engine operation fields")
    for name in ("model_config", "task", "config", "record", "inputs"):
        if name in payload and not isinstance(payload[name], dict):
            raise ValueError(f"Engine {name} must be an object")
    for name in ("ready_path", "runtime_root", "output_dir", "process_directory"):
        value = payload.get(name)
        if name in payload and (value is not None or name != "process_directory"):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Engine {name} must be a nonempty path string")
    for name in ("config", "record", "task"):
        nested = payload.get(name, {})
        if "profile" in nested and nested["profile"] != profile:
            raise ValueError("Engine input profile differs from its source selection")
    return request


def _stop_owned(process):
    """Terminate only the session created for this operation, including children."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # A worker may already have exited while one of its children is still alive.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON")))


def _watchdog_json(path, maximum=2 * 1024 * 1024):
    if path.is_symlink():
        raise ValueError("Unsafe watchdog evidence path")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("Oversized watchdog evidence")
    value = _read_json(raw)
    if not isinstance(value, dict):
        raise ValueError("Watchdog evidence must be an object")
    return value


def _safe_model_phase(payload, run):
    """The frozen Common monitor's phase precedence, for this exact run only."""
    if run.is_symlink() or not run.is_dir() or not run.samefile(payload["output_dir"]):
        return None
    record = _watchdog_json(run / "run_record.json")
    manifest = _watchdog_json(run / "client_manifest.json")
    task = payload["task"]
    identity = {key: task[key] for key in ("task_id", "task_name", "task_split", "task_instance_id", "task_seed")}
    if (record.get("profile") != _L2_PROFILE or record.get("experiment_id") != payload["experiment_id"]
            or record.get("task_identity") != identity
            or manifest.get("experiment_id") != payload["experiment_id"]
            or any(manifest.get(key) != value for key, value in identity.items())
            or manifest.get("model") != payload["model_config"].get("model", "").strip()
            or manifest.get("pause_on_client_failure") is not True
            or manifest.get("agent_cli") is not None):
        return None
    _watchdog_json(run / "provider_preflight.json")
    folder = run / "provider_calls"
    if folder.is_symlink() or not folder.is_dir():
        return None
    maximum = task["maximum_model_calls"]
    if type(maximum) is not int or not 1 <= maximum <= 30000:
        return None
    latest = None
    # The historical monitor capped discovery at 512; L2 declares up to 3810 calls.
    with os.scandir(folder) as entries:
        for count, entry in enumerate(entries, 1):
            if count > maximum:
                return None
            match = re.fullmatch(r"call_([0-9]{4})", entry.name)
            if entry.is_dir(follow_symlinks=False) and match is not None:
                index = int(match[1])
                if latest is None or index > latest[0]:
                    latest = (index, Path(entry.path))
    if latest is None:
        return None
    call_dir = latest[1]
    if (call_dir / "provider_error.json").is_file():
        error = _watchdog_json(call_dir / "provider_error.json", 256 * 1024)
        return "RETRYING_PROVIDER" if error.get("retry_allowed") is True else None
    if (call_dir / "protocol_attempt.json").is_file():
        return None  # Original RETRYING_PROTOCOL/RUNNING phases are not safe points.
    if (call_dir / "response.json").is_file():
        return None  # The original PARSING_RESPONSE phase is not a safe point.
    return "WAITING_PROVIDER" if (call_dir / "request_manifest.json").is_file() else "STARTING_PROVIDER_CALL"


def _owned_sigint_ready(process):
    if process.poll() is not None or process.pid <= 1 or os.getpgid(process.pid) != process.pid:
        return False
    with (Path("/proc") / str(process.pid) / "status").open() as stream:
        status = stream.read(16 * 1024)
    ignored = next(line.split(":", 1)[1].strip() for line in status.splitlines() if line.startswith("SigIgn:"))
    return not int(ignored, 16) & (1 << (signal.SIGINT - 1))


def _process_elapsed_seconds(process):
    """Use the same Linux process-age clock as the historical model observer."""
    try:
        with (Path("/proc") / str(process.pid) / "stat").open() as stream:
            raw = stream.read(16 * 1024)
        fields = raw[raw.rfind(")") + 2:].split()
        start_ticks = int(fields[19])
        with Path("/proc/uptime").open() as stream:
            uptime = float(stream.read(1024).split()[0])
        elapsed = max(0.0, uptime - start_ticks / os.sysconf("SC_CLK_TCK"))
        return elapsed if math.isfinite(elapsed) else None
    except (IndexError, OSError, ValueError):
        return None


def _wall_timeout_evidence(process, payload, run, threshold):
    elapsed = _process_elapsed_seconds(process)
    if elapsed is None or elapsed < threshold:
        return None
    try:
        phase = _safe_model_phase(payload, run)
        if phase not in SAFE_MODEL_INTERRUPT_PHASES or not _owned_sigint_ready(process):
            return None
    except (OSError, ValueError, KeyError, TypeError, StopIteration, RecursionError):
        return None  # Never turn missing identity or progress evidence into a forced stop.
    return {"policy": "model_wall_timeout/v1", "reason": "model_wall_timeout",
            "threshold_elapsed_seconds": threshold, "observed_elapsed_seconds": elapsed,
            "observed_phase": phase, "pid": process.pid, "experiment_id": payload["experiment_id"],
            "run_dir": str(run), "requested_at": datetime.now(timezone.utc).isoformat(),
            "budget_scope": "task" if threshold != _OUTER_WATCHDOG_SECONDS else "outer",
            "outer_watchdog_seconds": _OUTER_WATCHDOG_SECONDS,
            "clock_origin": "owned_model_worker_proc_start_ticks", "scene_load_included": False}


def _write_wall_evidence(run, evidence):
    private = run / "private"
    if private.is_symlink():
        raise ValueError("Unsafe watchdog evidence directory")
    private.mkdir(mode=0o700, exist_ok=True)
    path = private / "model_wall_timeout.json"
    temporary = path.with_suffix(".json.tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(evidence, stream, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _communicate_l2_episode(process, raw, payload, threshold):
    run = Path(payload["output_dir"]).resolve()
    interrupted = False
    while True:
        try:
            return process.communicate(raw, timeout=1)
        except subprocess.TimeoutExpired:
            raw = None  # communicate resumes sending/reading the same request after a timeout.
            if interrupted:
                continue
            evidence = _wall_timeout_evidence(process, payload, run, threshold)
            if evidence is None:
                continue
            # The historical controller refreshed progress and ownership before SIGINT.
            evidence = _wall_timeout_evidence(process, payload, run, threshold)
            if evidence is None:
                continue
            try:
                _write_wall_evidence(run, evidence)
                if not _owned_sigint_ready(process):
                    continue
                os.killpg(process.pid, signal.SIGINT)
                interrupted = True
                evidence.update(signal="SIGINT", signal_sent_at=datetime.now(timezone.utc).isoformat())
                _write_wall_evidence(run, evidence)
            except (OSError, ValueError, StopIteration):
                # No escalation: unknown ownership/evidence must not kill a simulator or an action.
                continue


def _task_wall_seconds(request):
    if request["operation"] != "episode" or request["profile"] != _L2_PROFILE:
        return None
    budget = request["payload"]["task"].get("budget")
    wall = budget.get("wall_clock_seconds") if isinstance(budget, dict) else None
    if type(wall) is not int or not 60 <= wall <= _OUTER_WATCHDOG_SECONDS:
        raise ValueError("L2 Common requires its frozen task wall budget in [60, 14400] seconds")
    return wall


def call(operation: str, payload: Mapping[str, Any], *, profile: str = DEFAULT_PROFILE):
    """Run one fixed operation without importing frozen code in the caller.

    Credentials remain environment variables. L2 Common retains its historical
    safe-point watchdog; other profiles acquire no new wall-clock stop rule.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("Engine payload must be an object")
    request = validate_request({"schema_version": SCHEMA, "operation": operation,
                                "profile": profile, "payload": dict(payload)})
    try:
        raw = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Engine inputs must be finite JSON data") from exc
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ValueError("Engine request is too large")
    task_wall = _task_wall_seconds(request)
    package_root = Path(__file__).resolve().parents[1]
    script = ("import sys; sys.path.insert(0, " + repr(str(package_root)) + "); "
              "from esi_benchmark._native.worker import main; raise SystemExit(main())")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    process = subprocess.Popen([sys.executable, "-I", "-B", "-c", script],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=None, env=env, start_new_session=True)
    try:
        if task_wall is not None:
            stdout, _ = _communicate_l2_episode(process, raw, request["payload"], task_wall)
        else:
            stdout, _ = process.communicate(raw, timeout=None if operation == "episode" else 120)
    except subprocess.TimeoutExpired as exc:
        _stop_owned(process)
        raise TimeoutError("CPU engine operation exceeded its 120-second timeout") from exc
    except BaseException:
        _stop_owned(process)
        raise
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
    if len(stdout) > MAX_MESSAGE_BYTES:
        raise RuntimeError("Engine response is too large")
    try:
        response = _read_json(stdout)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise RuntimeError("Engine did not return a valid response; inspect its stderr") from exc
    if (not isinstance(response, dict) or set(response) != {"schema_version", "ok", "result", "error"}
            or response["schema_version"] != SCHEMA or type(response["ok"]) is not bool):
        raise RuntimeError("Invalid engine response envelope")
    if response["ok"]:
        if process.returncode != 0 or response["error"] is not None:
            raise RuntimeError("Engine exit status disagrees with its result")
        return response["result"]
    error = response["error"]
    if (response["result"] is not None or not isinstance(error, dict)
            or set(error) != {"type", "message"} or not all(isinstance(v, str) for v in error.values())):
        raise RuntimeError("Invalid engine failure response")
    kind = {"ValueError": ValueError, "FileExistsError": FileExistsError,
            "FileNotFoundError": FileNotFoundError, "TimeoutError": TimeoutError}.get(error["type"], RuntimeError)
    raise kind(error["message"] or error["type"])
