"""Bounded inspection of local runs, without exporting private trajectory data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
_MODES = {"live", "offline_replay", "unknown_legacy"}
_VARIANT_HISTORY = {"frozen_history_5": 5, "history_20_variant_not_baseline": 20}


def _path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Artifact reference must be a relative path")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Artifact reference escaped the run directory")
    return path


def _read(root: Path, relative: str) -> dict[str, Any]:
    path = _path(root, relative)
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"Missing or oversized run artifact: {relative}")
    try:
        value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON value")))
    except RecursionError as exc:
        raise ValueError(f"Run artifact exceeds JSON nesting limit: {relative}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Run artifact must be an object: {relative}")
    return value


def inspect_run(run_dir: str | Path) -> dict[str, Any]:
    """Inspect public images/provenance and preserve explicitly incomplete runs.

    Legacy runs may omit run_record.json; their execution mode remains unknown.
    Private transcripts and raw provider responses are never returned.
    """
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError("Run directory does not exist")
    summary = _read(root, "summary.json")
    if not isinstance(summary.get("status"), str) or not summary["status"]:
        raise ValueError("Run summary has no terminal status")
    record = _read(root, "run_record.json") if _path(root, "run_record.json").is_file() else {}
    mode = record.get("mode", "unknown_legacy")
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError("Unknown run evidence mode")
    variant = record.get("evaluation_variant", "unknown")
    if not isinstance(variant, str) or variant not in {*_VARIANT_HISTORY, "unknown"}:
        raise ValueError("Unknown evaluation variant")
    if variant in _VARIANT_HISTORY and summary.get("physical_history_max") not in (None, _VARIANT_HISTORY[variant]):
        raise ValueError("Evaluation variant disagrees with recorded physical history")
    for field in ("source_revisions", "task_identity", "model", "declared_budget"):
        if record.get(field) is not None and not isinstance(record[field], dict):
            raise ValueError(f"run_record.{field} must be an object or null")
    failure = None
    if _path(root, "candidate_error.json").is_file():
        error = _read(root, "candidate_error.json")
        for field in ("status", "stage", "error_type"):
            if not isinstance(error.get(field), str) or not error[field]:
                raise ValueError(f"candidate_error.{field} must be a nonempty string")
        failure = {field: error[field] for field in ("status", "stage", "error_type")}
    result = {
        "schema_version": "esi-benchmark/trajectory-inspection-v1", "run_dir": str(root),
        "mode": mode, "evaluation_variant": variant, "client_status": summary["status"], "official_success": None,
        "official_q": None, "score_available": False,
        "model_calls": summary.get("model_calls"), "actions_sent_to_scene": summary.get("actions_sent_to_scene"),
        "macro_actions_resolved": summary.get("macro_actions_resolved"),
        "source_revisions": record.get("source_revisions"), "profile": record.get("profile"),
        "official_benchmark_eligible": record.get("official_benchmark_eligible") is True,
        "strict_production_eligible": record.get("strict_production_eligible") is True,
        "private_artifacts_present": any((root / name).exists() for name in ("private", "conversation_private.json", "provider_calls")),
    }
    if failure is not None:
        result["candidate_error"] = failure
    if not _path(root, "conversation_chatml.json").is_file():
        unavailable = _path(root, "conversation_unavailable.json")
        if unavailable.is_file():
            reason = _read(root, "conversation_unavailable.json").get("reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError("Conversation-unavailable record lacks a reason")
        elif summary["status"] == "INITIALIZATION_FAILED":
            reason = "initialization_failed"
        elif failure is not None:
            reason = "candidate_failure"
        else:
            raise ValueError("Missing ChatML without an explicit initialization failure record")
        return {**result, "status": "INCOMPLETE", "messages": None, "images": None,
                "completeness": "candidate_failure" if failure is not None else (
                    "initialization_failed" if summary["status"] == "INITIALIZATION_FAILED" else "conversation_unavailable"),
                "reason": reason}
    chatml = _read(root, "conversation_chatml.json")
    manifest = _read(root, "client_manifest.json")
    if chatml.get("schema_version") != "behavior-hl-action20-chatml-v3" or not isinstance(chatml.get("messages"), list):
        raise ValueError("Unsupported or malformed ChatML trajectory")
    if not 1 <= len(chatml["messages"]) <= 100000:
        raise ValueError("Trajectory message count exceeds the inspection bounds")
    images = {}
    for message in chatml["messages"]:
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                or message["role"] not in {"system", "user", "assistant", "tool"}):
            raise ValueError("Malformed trajectory message")
        content = message.get("content")
        if not isinstance(content, list):
            if not isinstance(content, str):
                raise ValueError("Malformed trajectory message content")
            continue
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Malformed trajectory content block")
            if item.get("type") == "image":
                relative = item.get("image")
                image = _path(root, relative)
                if not image.is_file() or image.stat().st_size > MAX_IMAGE_BYTES:
                    raise ValueError("Missing or oversized trajectory image")
                images[relative] = hashlib.sha256(image.read_bytes()).hexdigest()
    prompt_path = _path(root, "system_prompt.txt")
    if not prompt_path.is_file() or prompt_path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Missing or oversized system prompt")
    prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    prompt_provenance = manifest.get("prompt_provenance", {})
    if not isinstance(prompt_provenance, dict):
        raise ValueError("client_manifest.prompt_provenance must be an object")
    if prompt_sha != summary.get("prompt_sha256") or prompt_sha != prompt_provenance.get("rendered_prompt_sha256"):
        raise ValueError("Rendered system prompt provenance mismatch")
    if not isinstance(manifest.get("source_sha256"), dict) or not manifest["source_sha256"]:
        raise ValueError("Run lacks source hash provenance")
    candidate = summary.get("candidate_result")
    if candidate is not None:
        if (not isinstance(candidate, dict) or not isinstance(candidate.get("status"), str)
                or candidate["status"] not in {"recorded", "unavailable"}):
            raise ValueError("Candidate result is malformed")
        evidence_records = candidate.get("evidence", {})
        if not isinstance(evidence_records, dict):
            raise ValueError("candidate_result.evidence must be an object")
        if any(not isinstance(evidence, dict) for evidence in evidence_records.values()):
            raise ValueError("candidate_result.evidence entries must be objects")
        if candidate["status"] == "unavailable":
            if candidate.get("official_success") is not None or candidate.get("official_q") is not None:
                raise ValueError("Unavailable result must not contain a score")
        else:
            from .models import recorded_score_result
            evidence = evidence_records.get("final_score.json", {})
            archived = _path(root, evidence.get("path"))
            if not archived.is_file() or archived.stat().st_size > 512 * 1024:
                raise ValueError("Archived final score is missing or oversized")
            if hashlib.sha256(archived.read_bytes()).hexdigest() != evidence.get("sha256"):
                raise ValueError("Archived final score hash mismatch")
            interpreted = recorded_score_result(_read(root, evidence["path"]), candidate.get("episode_id"))
            if any(type(candidate.get(key)) is not type(value) or candidate.get(key) != value
                   for key, value in interpreted.items()):
                raise ValueError("Candidate result differs from original final score")
    if failure is not None:
        return {**result, "status": "INCOMPLETE", "messages": len(chatml["messages"]), "images": len(images),
                "completeness": "candidate_failure", "reason": "Candidate execution failed; see candidate_error.json"}
    if mode == "live" and candidate is not None and candidate["status"] == "recorded":
        result.update(official_success=candidate["official_success"], official_q=candidate["official_q"], score_available=True)
    elif mode == "unknown_legacy":
        result.update(official_success=summary.get("official_task_success", summary.get("official_success")), official_q=summary.get("q_score"))
        result["score_available"] = type(result["official_success"]) is bool
    return {
        **result, "status": "PASS", "messages": len(chatml["messages"]), "images": len(images),
        "image_sha256": images, "source_hashes_recorded": len(manifest["source_sha256"]), "prompt_sha256": prompt_sha,
        "public_projection": "conversation_chatml.json and relative images; review model-authored text before sharing",
        "completeness": "bounded single-run ChatML and provenance inspection; private evaluator and source files are not re-executed",
    }


def summarize_runs(run_dirs: Iterable[str | Path]) -> dict[str, Any]:
    """Keep invalid/incomplete attempts visible and report both Live denominators."""
    runs = []
    for path in run_dirs:
        try:
            runs.append(inspect_run(path))
        except (ValueError, OSError) as exc:
            root = Path(path).resolve()
            mode = "unknown_legacy"
            variant = "unknown"
            try:
                record = _read(root, "run_record.json")
                recorded_mode = record.get("mode")
                recorded_variant = record.get("evaluation_variant")
                if isinstance(recorded_variant, str) and recorded_variant in _VARIANT_HISTORY:
                    variant = recorded_variant
                if isinstance(recorded_mode, str) and recorded_mode in _MODES:
                    mode = recorded_mode
            except (ValueError, OSError):
                pass
            runs.append({"run_dir": str(root), "mode": mode, "evaluation_variant": variant, "status": "INVALID", "error": str(exc),
                         "score_available": False, "official_success": None, "completeness": "invalid_artifacts"})
    if not runs:
        raise ValueError("Supply at least one run directory")
    counts = {mode: sum(run["mode"] == mode for run in runs) for mode in sorted(_MODES)}
    variant_counts = {variant: sum(run["evaluation_variant"] == variant for run in runs)
                      for variant in sorted({run["evaluation_variant"] for run in runs})}
    scored = [run for run in runs if run["mode"] == "live" and run["score_available"]]
    successes = sum(run["official_success"] is True for run in scored)
    live = counts["live"]
    return {
        "schema_version": "esi-benchmark/run-summary-v1", "runs": runs, "counts": counts,
        "evaluation_variant_counts": variant_counts, "mixed_evaluation_variants": len(variant_counts) > 1,
        "total_live_attempts": live, "scored_live_runs": len(scored), "successful_live_runs": successes,
        "success_rate_scored_live": successes / len(scored) if scored else None,
        "success_rate_all_live_attempts": successes / live if live else None,
        "live_score_coverage": len(scored) / live if live else None,
        "unscored_live_runs": live - len(scored),
        "invalid_runs": sum(run["status"] == "INVALID" for run in runs),
        "incomplete_runs": sum(run["status"] == "INCOMPLETE" for run in runs),
        "denominator_policy": "report all declared Live attempts and score-covered Live separately; offline/unknown excluded from Live rates",
    }


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Validate recorded outputs; incomplete attempts are reported as INCOMPLETE."""
    return inspect_run(run_dir)
