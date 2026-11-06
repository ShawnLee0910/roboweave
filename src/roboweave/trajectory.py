"""Bounded inspection of local runs, without exporting private trajectory data."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_SCORE_BYTES = 512 * 1024
_MODES = {"live", "offline_replay", "unknown_legacy"}
_VARIANT_HISTORY = {"frozen_history_5": 5, "history_20_variant_not_baseline": 20}
_CANDIDATE_FAILURE_STATUSES = {"INITIALIZATION_FAILED", "EPISODE_FAILED"}
_TASK_FIELDS = ("task_id", "task_name", "task_split", "task_instance_id", "task_seed")


def _path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Artifact reference must be a relative path")
    path = root / relative
    if (".." in Path(relative).parts or "\\" in relative
            or any(part.is_symlink() for part in (path, *path.parents) if part != root and root in part.parents)
            or not path.resolve().is_relative_to(root)):
        raise ValueError("Artifact reference escaped the run directory")
    return path


def _json(raw, name):
    def number(raw_number):
        value = float(raw_number)
        if not math.isfinite(value):
            raise ValueError("Non-finite JSON value")
        return value

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"Duplicate JSON field in run artifact: {name}")
            value[key] = item
        return value
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_float=number,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON value")))
    except RecursionError as exc:
        raise ValueError(f"Run artifact exceeds JSON nesting limit: {name}") from exc


def _read(root: Path, relative: str) -> dict[str, Any]:
    path = _path(root, relative)
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"Missing or oversized run artifact: {relative}")
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"Missing or oversized run artifact: {relative}")
    value = _json(raw, relative)
    if not isinstance(value, dict):
        raise ValueError(f"Run artifact must be an object: {relative}")
    return value


def _evidence(root, candidate):
    from .models import EVIDENCE_SCHEMAS
    schema = candidate.get("evidence_schema_version")
    if schema not in (None, *EVIDENCE_SCHEMAS):
        raise ValueError("Unsupported result evidence schema")
    records = candidate.get("evidence", {})
    if not isinstance(records, dict) or len(records) > 65:
        raise ValueError("candidate_result.evidence must be a bounded object")
    if schema in EVIDENCE_SCHEMAS and "task_binding.json" not in records:
        raise ValueError("Result evidence lacks its archived task binding")
    retained = {"final_score.json", "aborted_after_environment_error.json", "task_binding.json", "task_process_result.json",
                "v_collection_identity.json", "v_protocol_result.json", "v_protocol_abort.json", "l2_v/final.json",
                "l2_v/checkpoint_violations.json"}
    documents = {}
    for name, evidence in records.items():
        if not isinstance(name, str) or not isinstance(evidence, dict):
            raise ValueError("candidate_result.evidence entries must be named objects")
        if schema in EVIDENCE_SCHEMAS and (
                set(evidence) != {"source", "path", "sha256", "size_bytes"}
                or not isinstance(evidence["source"], str) or not evidence["source"]):
            raise ValueError("Scene evidence descriptor is malformed")
        suffix = Path(name).suffix
        if (suffix not in {".json", ".jsonl"} or Path(name).is_absolute()
                or ".." in Path(name).parts or "\\" in name):
            raise ValueError("Invalid scene evidence name")
        relative = evidence.get("path")
        path = _path(root, relative)
        expected = "private/" + ("scene_final_score.json" if name == "final_score.json" else
                                  "v_scene/" + name if name.startswith("v_") else name)
        if schema in EVIDENCE_SCHEMAS and relative != expected:
            raise ValueError("Scene evidence name differs from its archived path")
        limit = MAX_JSON_BYTES if name.startswith(("v_", "l2_v/")) or suffix == ".jsonl" else MAX_SCORE_BYTES
        if not path.is_file() or path.stat().st_size > limit:
            raise ValueError(f"Archived scene evidence is missing or oversized: {name}")
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError(f"Archived scene evidence is oversized: {name}")
        digest = evidence.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Archived scene evidence hash mismatch: {name}")
        size = evidence.get("size_bytes")
        if (schema in EVIDENCE_SCHEMAS or "size_bytes" in evidence) and (type(size) is not int or size != len(raw)):
            raise ValueError(f"Archived scene evidence size mismatch: {name}")
        if suffix == ".jsonl":
            count = 0
            for row in raw.splitlines():
                if row.strip():
                    if not isinstance(_json(row, name), dict):
                        raise ValueError(f"Scene JSONL evidence must contain objects: {name}")
                    count += 1
            documents[name] = count
        else:
            document = _json(raw, name)
            types = (dict, list) if name == "l2_v/checkpoint_violations.json" else (dict,)
            if not isinstance(document, types):
                raise ValueError(f"Invalid scene evidence structure: {name}")
            if name in retained:
                documents[name] = document
    return documents


def _v_identity(documents, score, candidate, record, summary):
    from .assets import asset_manifest
    from .models import EVIDENCE_SCHEMAS, _native_abort_score
    from .profiles import source_revisions
    from .tasks import get_task
    if candidate.get("evidence_schema_version") not in EVIDENCE_SCHEMAS:
        raise ValueError("V identity verification requires result-evidence-v2")
    identity = documents.get("v_collection_identity.json")
    protocol = documents.get("v_protocol_result.json")
    binding = documents.get("task_binding.json")
    aborted = _native_abort_score(score, "L1-V")
    if aborted:
        abort = documents.get("v_protocol_abort.json")
        recovery = documents.get("aborted_after_environment_error.json", {})
        if (not isinstance(abort, dict) or abort.get("aborted") is not True
                or abort.get("v_protocol_complete") is not False
                or abort.get("same_scene_recovery_allowed") is not False
                or abort.get("reason") != recovery.get("reason")):
            raise ValueError("V aborted score lacks its matching protocol abort evidence")
    if (not isinstance(identity, dict) or (not isinstance(protocol, dict) and not (aborted and protocol is None))
            or not isinstance(binding, dict)):
        raise ValueError("V result lacks its identity, protocol or task binding evidence")
    identity_fields = {"collection_config_key", "task_spec_id", "package_id", "branch", "runtime_revision",
                       "historical_oracle_runtime_revision", "source_oracle_success", "runtime_validation_only",
                       "v_package_sha256"}
    if (set(identity) != identity_fields
            or any(not isinstance(identity[field], str) or not identity[field] for field in
                   ("collection_config_key", "task_spec_id", "package_id", "branch", "runtime_revision", "v_package_sha256"))
            or not re.fullmatch(r"[0-9a-f]{64}", identity["v_package_sha256"])
            or type(identity["source_oracle_success"]) is not bool
            or type(identity["runtime_validation_only"]) is not bool
            or (identity["historical_oracle_runtime_revision"] is not None and
                (not isinstance(identity["historical_oracle_runtime_revision"], str)
                 or not re.fullmatch(r"[0-9a-f]{40}", identity["historical_oracle_runtime_revision"])))):
        raise ValueError("V scene identity lacks its original frozen identity fields")
    key = identity.get("collection_config_key")
    if not isinstance(key, str) or not key:
        raise ValueError("V collection identity is missing")
    task = get_task(key, "L1-V")
    profile = record["profile"]
    seed = asset_manifest(profile)["inputs_by_task"][key]["seed"]
    revisions = source_revisions(profile)
    if (task["profile"] != profile or identity.get("branch") != "D1"
            or identity.get("task_spec_id") != task["runtime_instruction_id"]
            or identity.get("runtime_revision") != revisions["runtime"]
            or record.get("source_revisions") != revisions
            or binding.get("collection_config_key") != key or binding.get("v_identity") != identity
            or (score.get("collection_config_key") != key and not (aborted and "collection_config_key" not in score))
            or score.get("v_protocol") != protocol
            or (isinstance(protocol, dict) and "branch" in protocol and protocol["branch"] != identity["branch"])):
        raise ValueError("V scene identity or protocol differs from its frozen task binding")
    expected = {"task_id": task["source"]["row_index"], "task_name": task["task_id"], "task_seed": seed}
    if any(type(binding.get(field)) is not type(value) or binding.get(field) != value for field, value in expected.items()):
        raise ValueError("V archived task differs from its frozen public identity")
    if (summary.get("v_protocol_complete") is not candidate.get("v_protocol_complete")
            or summary.get("official_task_success") is not candidate.get("official_success")):
        raise ValueError("V archived model result disagrees with its task/protocol evidence")


def _task_identity(documents, candidate, record, summary, manifest):
    """Bind a recorded Live score to its run record, client manifest, model and frozen catalog task."""
    from .assets import asset_manifest
    from .profiles import source_revisions
    from .tasks import validate_model_task
    binding = documents.get("task_binding.json")
    profile = record.get("profile")
    if not isinstance(binding, dict) or not isinstance(profile, str):
        raise ValueError("Recorded Live result lacks its archived task binding or run profile")
    if record.get("source_revisions") != source_revisions(profile):
        raise ValueError("Run source revisions differ from the frozen profile")
    task = validate_model_task(profile, binding)
    run_identity = record.get("task_identity")
    if (not isinstance(run_identity, dict) or type(binding.get("task_seed")) is not int
            or any(not isinstance(binding.get(field), str) or not binding[field] for field in ("task_split", "task_instance_id"))
            or any(type(run_identity.get(field)) is not type(binding[field]) or run_identity.get(field) != binding[field]
                   or type(manifest.get(field)) is not type(binding[field]) or manifest.get(field) != binding[field]
                   for field in _TASK_FIELDS)
            or manifest.get("instruction_sha256") != task["instruction_sha256"]):
        raise ValueError("Recorded model task or instruction differs from the archived task binding")
    frozen = asset_manifest(profile).get("inputs_by_task", {}).get(task["task_id"], {})
    identity = frozen.get("source_task_identity")
    if (("seed" in frozen and binding["task_seed"] != frozen["seed"])
            or (isinstance(identity, dict) and (binding["task_split"] != identity.get("split")
                                                or binding["task_instance_id"] != str(identity.get("instance_id"))))):
        raise ValueError("Recorded task seed or instance differs from its frozen inputs")
    budget = task["budget"]
    declared = record.get("declared_budget")
    if (not isinstance(declared, dict) or set(declared) != set(budget)
            or any(type(declared[key]) is not type(expected) or declared[key] != expected for key, expected in budget.items())):
        raise ValueError("Recorded task budget differs from the frozen task")
    for field in ("maximum_action_decisions", "maximum_model_calls"):
        if any(type(source.get(field)) is not int or source[field] != budget[field]
               for source in (record, manifest, summary)):
            raise ValueError("Recorded model budget differs from the frozen task")
    history = record.get("history_frames")
    if (type(history) is not int or history not in {5, 20}
            or _VARIANT_HISTORY.get(record.get("evaluation_variant")) != history
            or any(type(source.get("physical_history_max")) is not int or source["physical_history_max"] != history
                   for source in (manifest, summary))):
        raise ValueError("Recorded history differs from its evaluation variant")
    experiment = record.get("experiment_id")
    if (not isinstance(experiment, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", experiment)
            or manifest.get("experiment_id") != experiment):
        raise ValueError("Recorded experiment identity differs from its client manifest")
    model = record.get("model")
    if (not isinstance(model, dict) or not isinstance(model.get("model"), str) or not model["model"]
            or model["model"] != manifest.get("model") or model["model"] != summary.get("model")
            or summary.get("current_scene_episode_id") != candidate.get("episode_id")):
        raise ValueError("Recorded result disagrees with its run model or scene episode")


def _wall_timeout(root, record):
    """Read the owned L2 watchdog record without treating it as a score."""
    from .engine import SAFE_MODEL_INTERRUPT_PHASES
    if not _path(root, "private/model_wall_timeout.json").is_file():
        return None
    evidence = _read(root, "private/model_wall_timeout.json")
    threshold, elapsed = (evidence.get(field) for field in ("threshold_elapsed_seconds", "observed_elapsed_seconds"))
    if (evidence.get("policy") != "model_wall_timeout/v1" or evidence.get("reason") != "model_wall_timeout"
            or record.get("profile") != "l2-collection37"
            or not isinstance(evidence.get("experiment_id"), str) or not evidence["experiment_id"]
            or evidence.get("experiment_id") != record.get("experiment_id")
            or evidence.get("signal", "SIGINT") != "SIGINT"
            or not isinstance(evidence.get("observed_phase"), str) or evidence["observed_phase"] not in SAFE_MODEL_INTERRUPT_PHASES
            or type(threshold) not in (int, float) or not 0 < threshold <= 14400
            or type(elapsed) not in (int, float) or (type(elapsed) is float and not math.isfinite(elapsed)) or elapsed < threshold):
        raise ValueError("Model wall-timeout evidence differs from its owned L2 run")
    return {"signal_sent": "signal" in evidence, "observed_phase": evidence.get("observed_phase"),
            "observed_elapsed_seconds": evidence.get("observed_elapsed_seconds"),
            "threshold_elapsed_seconds": evidence.get("threshold_elapsed_seconds")}


def inspect_run(run_dir: str | Path) -> dict[str, Any]:
    """Inspect public images/provenance and preserve explicitly incomplete runs.

    Legacy runs may omit run_record.json; their execution mode remains unknown.
    Private transcripts and raw provider responses are never returned.
    """
    try:
        root = Path(run_dir).resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("Run directory cannot be resolved") from exc
    if not root.is_dir():
        raise ValueError("Run directory does not exist")
    summary = _read(root, "summary.json")
    if not isinstance(summary.get("status"), str) or not summary["status"]:
        raise ValueError("Run summary has no terminal status")
    if ((summary.get("outcome_class") is not None and not isinstance(summary["outcome_class"], str))
            or (summary.get("formal_score_eligible") is not None and type(summary["formal_score_eligible"]) is not bool)):
        raise ValueError("Native outcome classification is malformed")
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
    if "profile" in record and (not isinstance(record["profile"], str) or not record["profile"]):
        raise ValueError("run_record.profile must be a nonempty string")
    failure = None
    if _path(root, "candidate_error.json").is_file():
        error = _read(root, "candidate_error.json")
        for field in ("status", "stage", "error_type"):
            if not isinstance(error.get(field), str) or not error[field]:
                raise ValueError(f"candidate_error.{field} must be a nonempty string")
        failure = {field: error[field] for field in ("status", "stage", "error_type")}
    candidate = summary.get("candidate_result")
    documents = {}
    if candidate is not None:
        if (not isinstance(candidate, dict) or not isinstance(candidate.get("status"), str)
                or candidate["status"] not in {"recorded", "unavailable"}):
            raise ValueError("Candidate result is malformed")
        documents = _evidence(root, candidate)
        if candidate["status"] == "unavailable" and (
                candidate.get("official_success") is not None or candidate.get("official_q") is not None):
            raise ValueError("Unavailable result must not contain a score")
    result = {
        "schema_version": "roboweave/trajectory-inspection-v1", "run_dir": str(root),
        "mode": mode, "evaluation_variant": variant, "client_status": summary["status"], "official_success": None,
        "outcome_class": summary.get("outcome_class"), "formal_score_eligible": summary.get("formal_score_eligible"),
        "official_q": None, "score_available": False,
        "model_calls": summary.get("model_calls"), "actions_sent_to_scene": summary.get("actions_sent_to_scene"),
        "macro_actions_resolved": summary.get("macro_actions_resolved"),
        "source_revisions": record.get("source_revisions"), "profile": record.get("profile"),
        "official_benchmark_eligible": record.get("official_benchmark_eligible") is True,
        "strict_production_eligible": record.get("strict_production_eligible") is True,
        "private_artifacts_present": any((root / name).exists() for name in ("private", "conversation_private.json", "provider_calls")),
        "wall_timeout": _wall_timeout(root, record),
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
    # A candidate failure summary replaces a native summary that was never written.
    native_summary = failure is None or summary["status"] not in _CANDIDATE_FAILURE_STATUSES
    if ((native_summary and prompt_sha != summary.get("prompt_sha256"))
            or prompt_sha != prompt_provenance.get("rendered_prompt_sha256")):
        raise ValueError("Rendered system prompt provenance mismatch")
    if (not isinstance(manifest.get("source_sha256"), dict) or not manifest["source_sha256"]
            or any(not isinstance(name, str) or not name or not isinstance(digest, str)
                   or not re.fullmatch(r"[0-9a-f]{64}", digest) for name, digest in manifest["source_sha256"].items())):
        raise ValueError("Run lacks source hash provenance")
    if candidate is not None and candidate["status"] == "recorded":
        from .models import EVIDENCE_SCHEMAS, _native_abort_score, recorded_score_result
        from .tasks import profile_for
        score = documents.get("final_score.json")
        if score is None:
            raise ValueError("Archived final score is missing")
        profile = record.get("profile", "l0-english-v2")
        family = profile_for(profile)["family"]
        if score.get("collection_config_key") is not None and family != "L1-V":
            raise ValueError("V collection evidence requires its frozen V profile")
        interpreted = recorded_score_result(score, candidate.get("episode_id"), profile=profile)
        if any(type(candidate.get(key)) is not type(value) or candidate.get(key) != value
               for key, value in interpreted.items()):
            raise ValueError("Candidate result differs from original final score")
        strict = candidate.get("evidence_schema_version") in EVIDENCE_SCHEMAS
        aborted = _native_abort_score(score, family)
        if strict and aborted:
            abort = documents.get("aborted_after_environment_error.json")
            if (not isinstance(abort, dict) or not isinstance(abort.get("reason"), str) or not abort["reason"]
                    or type(abort.get("public_actions_executed")) is not int or abort["public_actions_executed"] < 0
                    or not isinstance(abort.get("symbolic_collision_filter_cleanup"), dict)
                    or abort.get("same_scene_reset_required_before_retry") is not True
                    or type(abort.get("recovery_admitted")) is not bool):
                raise ValueError("Aborted final score lacks its native recovery evidence")
        for name, field in (("task_process_result.json", "public_process_report"), ("l2_v/final.json", "v_protocol")):
            required = strict and field in score and (field != "v_protocol" or family == "L2")
            if required and name not in documents:
                raise ValueError(f"Archived final score lacks supporting evidence: {name}")
            if (name in documents and documents[name] != score.get(field)
                    and not (aborted and field == "public_process_report" and field not in score)):
                raise ValueError(f"Archived supporting evidence differs from final score: {name}")
        process = score.get("public_process_report")
        if aborted and process is None and "task_process_result.json" in documents:
            process = documents["task_process_result.json"]
            # Recovery writes its process receipt separately from its negative final score.
            recorded_score_result({**score, "public_process_report": process}, candidate.get("episode_id"), profile=profile)
        if strict and process is not None:
            for field, name in (("execution_events", "task_process_events.jsonl"),
                                ("existing_frame_references", "task_evidence_index.jsonl")):
                count = process.get(field)
                if type(count) is not int or count < 0 or documents.get(name, 0) != count:
                    raise ValueError(f"Archived process evidence count mismatch: {name}")
        if "l2_v/checkpoint_violations.json" in documents:
            protocol = score.get("v_protocol", {})
            if isinstance(protocol, dict) and "checkpoint_violations" in protocol and documents["l2_v/checkpoint_violations.json"] != protocol["checkpoint_violations"]:
                raise ValueError("Archived L2 checkpoint evidence differs from final score")
        if family == "L1-V":
            _v_identity(documents, score, candidate, record, summary)
        if strict and mode == "live":
            _task_identity(documents, candidate, record, summary, manifest)
    if failure is not None:
        return {**result, "status": "INCOMPLETE", "messages": len(chatml["messages"]), "images": len(images),
                "completeness": "candidate_failure", "reason": "Candidate execution failed; see candidate_error.json"}
    if mode in {"live", "unknown_legacy"} and candidate is not None and candidate["status"] == "recorded":
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
            try:
                root = Path(path).resolve()
            except (OSError, RuntimeError):
                root = Path(path).absolute()
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
            except (ValueError, OSError, RuntimeError):
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
    # The historical L2 scheduler recorded an unscored, signalled wall-timeout attempt as FAILED.
    timed_out = sum(run["mode"] == "live" and not run["score_available"]
                    and (run.get("wall_timeout") or {}).get("signal_sent") is True for run in runs)
    return {
        "schema_version": "roboweave/run-summary-v1", "runs": runs, "counts": counts,
        "evaluation_variant_counts": variant_counts, "mixed_evaluation_variants": len(variant_counts) > 1,
        "total_live_attempts": live, "scored_live_runs": len(scored), "successful_live_runs": successes,
        "success_rate_scored_live": successes / len(scored) if scored else None,
        "success_rate_all_live_attempts": successes / live if live else None,
        "live_score_coverage": len(scored) / live if live else None,
        "unscored_live_runs": live - len(scored),
        "wall_timeout_failed_live_runs": timed_out,
        "success_rate_scored_or_wall_timeout_live": (successes / (len(scored) + timed_out)
                                                     if scored or timed_out else None),
        "invalid_runs": sum(run["status"] == "INVALID" for run in runs),
        "incomplete_runs": sum(run["status"] == "INCOMPLETE" for run in runs),
        "denominator_policy": ("report all declared Live attempts and score-covered Live separately; signalled L2 "
                               "wall-timeout attempts without a score are also counted as failures in a third rate; "
                               "offline/unknown excluded from Live rates"),
    }


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Validate recorded outputs; incomplete attempts are reported as INCOMPLETE."""
    return inspect_run(run_dir)
