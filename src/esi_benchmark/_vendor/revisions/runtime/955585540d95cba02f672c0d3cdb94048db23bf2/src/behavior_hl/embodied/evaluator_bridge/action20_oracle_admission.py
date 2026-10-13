"""Immutable, CPU-only admission for Action20 Oracle Q=1 donors.

The admission file is a private control-plane sidecar.  It binds a small
Action20 campaign task set to exact preserved Oracle artifacts and re-attests
the current tree before any caller may consume one task.  Validation reads
only the exact files named by the sidecar; it never discovers or walks an
artifact directory.

``COUNTED`` means that a task contributes to this sidecar's admitted campaign
count.  ``SHADOW_ONLY`` is deliberately non-counting.  Neither mode mutates
the separately frozen 41-certified / 22-unresolved Oracle63 ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from behavior_hl.embodied.evaluator_bridge.action20_oracle_protocol_witness import (
    REMOVED_PUBLIC_ACTIONS,
    STRICT_ACTION20,
    compile_oracle63_witness,
    lower_profile,
)
from behavior_hl.embodied.evaluator_bridge.inside_goal_graph import (
    grounded_goal_option_identity,
)
from behavior_hl.embodied.evaluator_bridge.official_actual_option_mapping import (
    semantic_identity_sha256,
)
from behavior_hl.embodied.evaluator_bridge.prerender_dag_audit import audit_profile


ACTION20_ORACLE_ADMISSION_SCHEMA = (
    "behavior2026.action20-oracle-q1-admission/v187"
)
CURRENT_TREE_ATTESTATION_SCHEMA = (
    "behavior2026.action20-current-tree-static-reattestation/v1"
)
COUNTED = "COUNTED"
SHADOW_ONLY = "SHADOW_ONLY"
ADMISSION_MODES = frozenset({COUNTED, SHADOW_ONLY})

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ACTION20_ORACLE_ADMISSION_PATH = (
    PROJECT_ROOT
    / "configs"
    / "campaigns"
    / "action20_oracle_q1_admission_v187.json"
)

_ARTIFACT_FILE_KEYS = frozenset(
    {
        "manifest",
        "summary",
        "verifier",
        "journal",
        "visual",
        "source_freeze",
        "progress",
        "task_identity",
        "option_witness",
    }
)
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "immutable",
        "task_count",
        "counted_task_count",
        "shadow_task_count",
        "ledger",
        "current_tree_attestation",
        "tasks",
    }
)
_TASK_KEYS = frozenset(
    {
        "task_id",
        "task_name",
        "scene",
        "split",
        "instance_id",
        "seed",
        "admission_mode",
        "count_contribution",
        "baseline_membership",
        "artifact_root",
        "artifact_files",
        "source_freeze",
        "q1_attestation",
        "route_mode",
        "option_identity",
        "current_tree_profile",
    }
)


class Action20OracleAdmissionError(RuntimeError):
    """The admission sidecar, donor artifact, or current tree drifted."""


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Action20OracleAdmissionError(f"{label} is not a SHA-256 digest")
    return value


def _require_exact_keys(
    value: Any,
    *,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Action20OracleAdmissionError(f"{label} must be an object")
    actual = frozenset(str(key) for key in value)
    if actual != expected:
        raise Action20OracleAdmissionError(
            f"{label} keys drifted: expected {sorted(expected)}, got {sorted(actual)}"
        )
    return value


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Action20OracleAdmissionError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Action20OracleAdmissionError(f"{label} must be a non-empty string")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


@dataclass(frozen=True, slots=True)
class OracleTaskAdmission:
    task_id: int
    task_name: str
    admission_mode: str
    count_contribution: int
    record: Mapping[str, Any]

    @property
    def is_counted(self) -> bool:
        return self.admission_mode == COUNTED

    @property
    def is_shadow_only(self) -> bool:
        return self.admission_mode == SHADOW_ONLY


@dataclass(frozen=True, slots=True)
class Action20OracleAdmission:
    path: Path
    sidecar_sha256: str
    tasks: tuple[OracleTaskAdmission, ...]
    counted_task_count: int
    shadow_task_count: int
    current_tree_attested: bool
    artifacts_attested: bool
    ledger: Mapping[str, Any]

    @property
    def task_universe(self) -> frozenset[tuple[int, str]]:
        return frozenset((task.task_id, task.task_name) for task in self.tasks)


def _normalize_task_universe(
    values: Iterable[tuple[int, str]],
    *,
    label: str,
) -> frozenset[tuple[int, str]]:
    normalized: list[tuple[int, str]] = []
    for ordinal, item in enumerate(values):
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or isinstance(item[0], bool)
            or not isinstance(item[0], int)
            or item[0] < 0
            or not isinstance(item[1], str)
            or not item[1]
        ):
            raise Action20OracleAdmissionError(
                f"{label}[{ordinal}] is not one (task_id, task_name) pair"
            )
        normalized.append((item[0], item[1]))
    result = frozenset(normalized)
    if len(result) != len(normalized):
        raise Action20OracleAdmissionError(f"{label} contains duplicate identities")
    ids = [task_id for task_id, _ in normalized]
    names = [task_name for _, task_name in normalized]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise Action20OracleAdmissionError(
            f"{label} reuses a task ID or task name"
        )
    return result


def _safe_artifact_file(root: Path, record: Mapping[str, Any], *, label: str) -> Path:
    relative = _require_nonempty_string(record.get("path"), label=f"{label}.path")
    expected_hash = _require_sha256(record.get("sha256"), label=f"{label}.sha256")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise Action20OracleAdmissionError(
            f"{label} escapes its exact artifact root"
        ) from exc
    if not candidate.is_file():
        raise Action20OracleAdmissionError(f"{label} is missing: {candidate}")
    actual_hash = _sha256_file(candidate)
    if actual_hash != expected_hash:
        raise Action20OracleAdmissionError(
            f"{label} hash drifted: expected {expected_hash}, got {actual_hash}"
        )
    return candidate


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Action20OracleAdmissionError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise Action20OracleAdmissionError(f"{label} root must be an object")
    return value


def _last_jsonl_record(path: Path, *, label: str) -> Mapping[str, Any]:
    last: str | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = line
    if last is None:
        raise Action20OracleAdmissionError(f"{label} is empty")
    try:
        value = json.loads(last)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Action20OracleAdmissionError(
            f"{label} terminal record is invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise Action20OracleAdmissionError(
            f"{label} terminal record must be an object"
        )
    return value


def _require_q1(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != 1.0:
        raise Action20OracleAdmissionError(f"{label} must equal 1.0")


def _selected_leaf_identity_sha256(truth: Mapping[str, Any]) -> str:
    if not truth or any(value is not True for value in truth.values()):
        raise Action20OracleAdmissionError(
            "selected option does not have a non-empty all-true leaf vector"
        )
    return _sha256_json(sorted(str(key) for key in truth))


def _validate_option_witness(
    task: Mapping[str, Any],
    *,
    witness: Mapping[str, Any],
) -> None:
    option = task["option_identity"]
    if not isinstance(option, Mapping):
        raise Action20OracleAdmissionError("option_identity must be an object")
    canonical = option.get("canonical_index")
    live = option.get("live_index")
    if not isinstance(canonical, Mapping) or not isinstance(live, Mapping):
        raise Action20OracleAdmissionError("option index domains are missing")
    if canonical.get("domain") != "canonical_lazy_grounder_raw_index":
        raise Action20OracleAdmissionError("canonical option index domain drifted")
    if live.get("domain") != "live_official_ground_goal_actual_index":
        raise Action20OracleAdmissionError("live option index domain drifted")
    canonical_index = _require_int(
        canonical.get("value"), label="canonical option index"
    )
    live_index = _require_int(live.get("value"), label="live option index")
    if option.get("numeric_index_equality_required") is not False:
        raise Action20OracleAdmissionError(
            "raw and live option domains must not require numeric equality"
        )
    canonical_hash = _require_sha256(
        option.get("canonical_semantic_identity_sha256"),
        label="canonical semantic identity",
    )
    live_hash = _require_sha256(
        option.get("live_semantic_identity_sha256"),
        label="live semantic identity",
    )
    source_kind = option.get("identity_source_kind")
    if source_kind == "lazy_grounded_option_witness":
        if witness.get("option_index") != live_index:
            raise Action20OracleAdmissionError("live actual option index drifted")
        if witness.get("identity_sha256") != live_hash:
            raise Action20OracleAdmissionError("live option identity drifted")
        mapping = witness.get("official_actual_index_mapping")
        if not isinstance(mapping, Mapping):
            raise Action20OracleAdmissionError("official actual-index mapping is missing")
        if mapping.get("actual_option_index") != live_index:
            raise Action20OracleAdmissionError(
                "witness actual-index mapping drifted"
            )
        if mapping.get("semantic_identity_sha256") != live_hash:
            raise Action20OracleAdmissionError(
                "witness mapped semantic identity drifted"
            )
        leaves = witness.get("canonical_goal_leaves")
        if not isinstance(leaves, list):
            raise Action20OracleAdmissionError("witness semantic leaves are missing")
        if semantic_identity_sha256(tuple(str(leaf) for leaf in leaves)) != live_hash:
            raise Action20OracleAdmissionError(
                "witness semantic leaves do not hash to their identity"
            )
    elif source_kind == "grounded_option_geometry_ranking":
        if witness.get("selected_option_index") != live_index:
            raise Action20OracleAdmissionError("geometry-selected option index drifted")
        ranking = witness.get("ranking_private")
        if not isinstance(ranking, list):
            raise Action20OracleAdmissionError("geometry option ranking is missing")
        selected = [
            item
            for item in ranking
            if isinstance(item, Mapping) and item.get("option_index") == live_index
        ]
        if len(selected) != 1:
            raise Action20OracleAdmissionError(
                "geometry ranking lacks one selected option"
            )
        assignments = selected[0].get("assignment_identity_private")
        if not isinstance(assignments, list) or not assignments:
            raise Action20OracleAdmissionError(
                "selected geometry assignment identity is missing"
            )
        leaves: list[Any] = []
        for pair in assignments:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(value, str) or not value for value in pair)
            ):
                raise Action20OracleAdmissionError(
                    "selected geometry assignment identity is invalid"
                )
            leaves.append(["inside", pair[0], pair[1]])
        invariant_leaves = option.get("live_identity_invariant_leaves")
        if not isinstance(invariant_leaves, list):
            raise Action20OracleAdmissionError(
                "live identity invariant leaves are invalid"
            )
        leaves.extend(invariant_leaves)
        actual = semantic_identity_sha256(grounded_goal_option_identity(leaves))
        if actual != live_hash:
            raise Action20OracleAdmissionError(
                "selected geometry option semantic identity drifted"
            )
    elif source_kind == "authoritative_summary_truth_v187":
        # v187 oracle runs write no separate grounding witness; the entry's
        # option_witness artifact IS the authoritative summary.  The semantic
        # identity anchors to the frozen catalog's canonical goal leaves and
        # is enforced against the catalog by _validate_current_tree; here the
        # summary must agree on the selected option index and carry the
        # authoritative leaf truth.
        summary_truth = witness.get(
            "last_authoritative_selected_option_truth_private"
        )
        if not isinstance(summary_truth, Mapping) or not summary_truth:
            raise Action20OracleAdmissionError(
                "summary authoritative truth is missing"
            )
        if witness.get(
            "last_authoritative_selected_option_index_private"
        ) != live_index:
            raise Action20OracleAdmissionError(
                "summary authoritative option index drifted"
            )
        if not all(value is True for value in summary_truth.values()):
            raise Action20OracleAdmissionError(
                "summary authoritative truth is not all true"
            )
    else:
        raise Action20OracleAdmissionError("unknown option identity source kind")

    relation = option.get("semantic_identity_relation")
    if relation == "EQUAL":
        if canonical_hash != live_hash:
            raise Action20OracleAdmissionError(
                "equal semantic option domains have different identities"
            )
    elif relation == "DISTINCT_VALID_OPTIONS":
        if canonical_hash == live_hash:
            raise Action20OracleAdmissionError(
                "distinct semantic option identities unexpectedly match"
            )
    else:
        raise Action20OracleAdmissionError("semantic identity relation drifted")

    if int(task["task_id"]) == 98:
        if canonical_index != 64 or live_index != 16:
            raise Action20OracleAdmissionError(
                "Task98 raw/live option domain values drifted"
            )
        expected = (
            "0a05e69d4bf69f632013e44c0f9a1a334381627acc0931c0dafaff8641bab4e4"
        )
        if canonical_hash != expected or live_hash != expected:
            raise Action20OracleAdmissionError(
                "Task98 semantic option identity drifted"
            )


def _validate_artifact(task: Mapping[str, Any]) -> None:
    artifact_root_raw = _require_nonempty_string(
        task.get("artifact_root"), label="artifact_root"
    )
    artifact_root = Path(artifact_root_raw)
    if not artifact_root.is_absolute():
        raise Action20OracleAdmissionError("artifact_root must be absolute")
    artifact_root = artifact_root.resolve()
    files = task.get("artifact_files")
    if not isinstance(files, Mapping) or frozenset(files) != _ARTIFACT_FILE_KEYS:
        raise Action20OracleAdmissionError(
            "artifact_files must name the exact admission artifact set"
        )
    resolved = {
        key: _safe_artifact_file(
            artifact_root,
            files[key],
            label=f"Task{task['task_id']} artifact {key}",
        )
        for key in sorted(_ARTIFACT_FILE_KEYS)
    }
    # v187 oracle runs renamed the goal-leaf progress artifact and stopped
    # writing a separate grounding witness; the per-entry paths identify the
    # donor artifact generation.
    v187_donor_shape = str(files["progress"].get("path", "")).endswith(
        "/goal_leaf_progress_final.json"
    )
    manifest = _load_json(resolved["manifest"], label="manifest")
    summary = _load_json(resolved["summary"], label="summary")
    verifier = _load_json(resolved["verifier"], label="verifier")
    progress = _load_json(resolved["progress"], label="progress")
    visual = _load_json(resolved["visual"], label="visual frame index")
    source = _load_json(resolved["source_freeze"], label="source freeze")
    identity = _load_json(resolved["task_identity"], label="task identity")
    witness = _load_json(resolved["option_witness"], label="option witness")
    terminal = _last_jsonl_record(resolved["journal"], label="Oracle journal")

    expected_identity = (
        task["task_name"],
        task["scene"],
        task["split"],
        task["instance_id"],
        task["seed"],
    )
    observed_identity = (
        manifest.get("task"),
        identity.get("scene_name"),
        manifest.get("split"),
        manifest.get("instance_id"),
        manifest.get("seed"),
    )
    if observed_identity != expected_identity:
        raise Action20OracleAdmissionError(
            f"artifact task/scene/split/instance/seed drifted: "
            f"{observed_identity} != {expected_identity}"
        )
    if identity.get("activity_name") != task["task_name"]:
        raise Action20OracleAdmissionError("artifact task identity name drifted")
    if identity.get("activity_definition_id") != 0:
        raise Action20OracleAdmissionError("artifact definition ID drifted")
    profile = task["current_tree_profile"]
    if identity.get("sha256") != profile.get("task_state_sha256"):
        raise Action20OracleAdmissionError("artifact task-state identity drifted")
    if manifest.get("status") != "COMPLETE":
        raise Action20OracleAdmissionError("Oracle manifest is not COMPLETE")
    if manifest.get("source_freeze_revalidated") is not True:
        raise Action20OracleAdmissionError("Oracle source freeze was not revalidated")

    q1 = task["q1_attestation"]
    if not isinstance(q1, Mapping):
        raise Action20OracleAdmissionError("q1_attestation must be an object")
    q1_sources = [
        ("sidecar official_q", q1.get("official_q")),
        ("summary official_q", summary.get("official_q")),
    ]
    if not v187_donor_shape:
        q1_sources.append(
            ("progress official_q", progress.get("official_q_after_private"))
        )
    for label, value in q1_sources:
        _require_q1(value, label=label)
    if q1.get("official_success") is not True:
        raise Action20OracleAdmissionError("sidecar official success is false")
    if summary.get("official_success") is not True:
        raise Action20OracleAdmissionError("summary official success is false")
    if summary.get("status") != "PASS":
        raise Action20OracleAdmissionError("Oracle strategy summary is not PASS")
    if summary.get("verification_mode") != "official_bddl_predicate_verifier":
        raise Action20OracleAdmissionError("Oracle verifier adapter drifted")
    if summary.get("official_bddl_verifier_required") is not True:
        raise Action20OracleAdmissionError("official verifier was not required")
    if v187_donor_shape:
        if progress.get("complete") is not True:
            raise Action20OracleAdmissionError("selected-option progress is incomplete")
        vector = progress.get("current_truth_vector")
        if not isinstance(vector, list) or not vector or not all(
            item is True for item in vector
        ):
            raise Action20OracleAdmissionError(
                "progress selected leaves are not all true"
            )
        if progress.get("pending_recovery_leaf_ids"):
            raise Action20OracleAdmissionError(
                "progress terminated with pending recovery"
            )
    else:
        if progress.get("status") != "COMPLETE" or progress.get("complete") is not True:
            raise Action20OracleAdmissionError("selected-option progress is incomplete")
        if progress.get("official_success_after_private") is not True:
            raise Action20OracleAdmissionError("progress official success is false")
        if progress.get("selected_option_all_leaves_true") is not True:
            raise Action20OracleAdmissionError("progress selected leaves are not all true")

    final = verifier.get("final")
    if not isinstance(final, Mapping):
        raise Action20OracleAdmissionError("private verifier final snapshot is missing")
    _require_q1(final.get("official_q"), label="verifier official_q")
    if (
        final.get("ok") is not True
        or final.get("official_success") is not True
        or final.get("error") is not None
    ):
        raise Action20OracleAdmissionError("private verifier did not pass cleanly")

    if terminal.get("kind") != "terminal":
        raise Action20OracleAdmissionError("Oracle journal lacks a terminal record")
    terminal_state = terminal.get("authoritative_official_state_private")
    if not isinstance(terminal_state, Mapping):
        raise Action20OracleAdmissionError("terminal official state is missing")
    _require_q1(terminal_state.get("official_q"), label="journal official_q")
    if terminal_state.get("official_success") is not True:
        raise Action20OracleAdmissionError("journal official success is false")
    truth = terminal_state.get("selected_option_leaf_truth")
    if not isinstance(truth, Mapping):
        raise Action20OracleAdmissionError("journal selected leaf truth is missing")
    truth_hash = _selected_leaf_identity_sha256(truth)
    if len(truth) != q1.get("selected_leaf_count"):
        raise Action20OracleAdmissionError("selected leaf count drifted")
    if truth_hash != q1.get("selected_leaf_identity_sha256"):
        raise Action20OracleAdmissionError("selected leaf identity drifted")
    if q1.get("selected_option_all_leaves_true") is not True:
        raise Action20OracleAdmissionError("sidecar does not require all selected leaves")

    selected_index = task["option_identity"]["live_index"]["value"]
    progress_selected = (
        progress.get("option_index")
        if v187_donor_shape
        else progress.get("selected_plan_option_index")
    )
    for label, value in (
        ("summary selected option", summary.get("last_authoritative_selected_option_index_private")),
        ("progress selected option", progress_selected),
        ("journal selected option", terminal_state.get("selected_option_index")),
    ):
        if value != selected_index:
            raise Action20OracleAdmissionError(f"{label} drifted")

    if visual.get("complete") is not True:
        raise Action20OracleAdmissionError("visual frame index is incomplete")
    if visual.get("frame_count") != q1.get("visual_frame_count"):
        raise Action20OracleAdmissionError("visual frame count drifted")
    source_contract = task["source_freeze"]
    aggregate = _require_sha256(
        source_contract.get("aggregate_sha256"), label="source aggregate"
    )
    if source.get("aggregate_sha256") != aggregate:
        raise Action20OracleAdmissionError("source-freeze aggregate drifted")
    if manifest.get("source_freeze_aggregate_sha256") != aggregate:
        raise Action20OracleAdmissionError("manifest source aggregate drifted")
    if source.get("file_count") != source_contract.get("file_count"):
        raise Action20OracleAdmissionError("source-freeze file count drifted")

    route = task["route_mode"]
    strategy_summary = manifest.get("strategy_summary")
    if not isinstance(route, Mapping) or not isinstance(strategy_summary, Mapping):
        raise Action20OracleAdmissionError("Oracle route mode is missing")
    expected_route = {
        "method": summary.get("method"),
        "base_execution": summary.get("base_execution"),
        "semantic_execution_profile": summary.get("semantic_execution_profile"),
        "route_planner": strategy_summary.get("route_planner"),
        "public_action_route_loop": summary.get("public_action_route_loop"),
        "complete_route_precompiled_before_execution": summary.get(
            "complete_route_precompiled_before_execution"
        ),
    }
    if dict(route) != expected_route:
        raise Action20OracleAdmissionError("Oracle route mode drifted")

    _validate_option_witness(task, witness=witness)


def _resolve_project_file(relative: str) -> Path:
    candidate = (PROJECT_ROOT / relative).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise Action20OracleAdmissionError(
            f"current-tree source escaped project root: {relative}"
        ) from exc
    return candidate


def _find_profile(profile_contract: Mapping[str, Any]) -> Mapping[str, Any]:
    catalog_path = _resolve_project_file(
        _require_nonempty_string(
            profile_contract.get("catalog_path"), label="catalog_path"
        )
    )
    catalog = _load_json(catalog_path, label="current-tree profile catalog")
    if catalog.get("schema_version") != profile_contract.get("catalog_schema"):
        raise Action20OracleAdmissionError("profile catalog schema drifted")
    records = catalog.get("tasks")
    if not isinstance(records, list) or catalog.get("task_count") != len(records):
        raise Action20OracleAdmissionError("profile catalog task count drifted")
    selected = [
        item
        for item in records
        if isinstance(item, Mapping)
        and item.get("task_id") == profile_contract.get("task_id")
        and item.get("task_name") == profile_contract.get("task_name")
    ]
    if len(selected) != 1:
        raise Action20OracleAdmissionError(
            "current-tree profile identity is absent or duplicated"
        )
    return selected[0]


def _validate_oracle63_exact_report(attestation: Mapping[str, Any]) -> None:
    contract = attestation.get("oracle63_exact_report")
    if not isinstance(contract, Mapping):
        raise Action20OracleAdmissionError(
            "current-tree Oracle63 exact report is missing"
        )
    report_path = _resolve_project_file(
        _require_nonempty_string(contract.get("path"), label="Oracle63 report path")
    )
    expected_report_hash = _require_sha256(
        contract.get("sha256"), label="Oracle63 report hash"
    )
    if not report_path.is_file() or _sha256_file(report_path) != expected_report_hash:
        raise Action20OracleAdmissionError("current-tree Oracle63 report hash drifted")
    report = _load_json(report_path, label="current-tree Oracle63 report")
    if (
        report.get("schema_version") != "behavior2026.prerender-dag-audit/v1"
        or report.get("cpu_only") is not True
        or report.get("simulator_imported") is not False
        or report.get("profile_count") != 63
        or report.get("compile_failure_count") != 0
    ):
        raise Action20OracleAdmissionError(
            "current-tree Oracle63 report does not certify 63 CPU profiles"
        )
    profiles = report.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 63:
        raise Action20OracleAdmissionError("Oracle63 report profiles are incomplete")
    identities = _normalize_task_universe(
        (
            (profile.get("task_id"), profile.get("task_name"))
            for profile in profiles
            if isinstance(profile, Mapping)
        ),
        label="Oracle63 exact report profiles",
    )
    if len(identities) != len(profiles):
        raise Action20OracleAdmissionError(
            "Oracle63 exact report contains a non-object profile"
        )
    universe_hash = _sha256_json(
        sorted([list(identity) for identity in identities])
    )
    if universe_hash != _require_sha256(
        contract.get("task_universe_sha256"),
        label="Oracle63 task universe hash",
    ):
        raise Action20OracleAdmissionError("Oracle63 task universe hash drifted")
    if contract.get("task_count") != len(identities):
        raise Action20OracleAdmissionError("Oracle63 report task count drifted")
    required = tuple(contract.get("required_task_id_name", ()))
    excluded = tuple(contract.get("excluded_task_id_name", ()))
    if required != (66, "turning_out_all_lights_before_sleep"):
        raise Action20OracleAdmissionError("Oracle63 required Task66 identity drifted")
    if excluded != (52, "sorting_bottles_cans_and_paper"):
        raise Action20OracleAdmissionError("Oracle63 excluded Task52 identity drifted")
    if required not in identities or excluded in identities:
        raise Action20OracleAdmissionError(
            "Oracle63 corrected report must contain Task66 and exclude Task52"
        )
    task_set = report.get("oracle63_campaign_task_set")
    if (
        not isinstance(task_set, Mapping)
        or task_set.get("exact_task_id_name_set_equal") is not True
        or task_set.get("missing_task_id_name") != []
        or task_set.get("unexpected_task_id_name") != []
    ):
        raise Action20OracleAdmissionError(
            "Oracle63 report lacks exact campaign set equality"
        )
    witness = compile_oracle63_witness(report)
    witness_profiles = witness.get("profiles")
    if not isinstance(witness_profiles, list):
        raise Action20OracleAdmissionError(
            "corrected Oracle63 Action20 witness profiles are missing"
        )
    witness_identities = _normalize_task_universe(
        (
            (
                profile["private_certification"].get("task_id"),
                profile["private_certification"].get("task_name"),
            )
            for profile in witness_profiles
            if isinstance(profile, Mapping)
            and isinstance(profile.get("private_certification"), Mapping)
        ),
        label="corrected Oracle63 Action20 witness",
    )
    checks = witness.get("checks")
    if (
        witness_identities != identities
        or not isinstance(checks, Mapping)
        or checks.get("all_source_dags_topological") is not True
        or checks.get("all_dag_actions_lowered") is not True
        or checks.get("all_emitted_actions_in_strict_action20") is not True
        or checks.get("removed_action_occurrence_count") != 0
        or checks.get("private_identity_leak_count") != 0
    ):
        raise Action20OracleAdmissionError(
            "corrected Oracle63 Action20/privacy re-attestation failed"
        )

    campaign_path = _resolve_project_file(
        _require_nonempty_string(
            contract.get("campaign_path"), label="Oracle63 campaign path"
        )
    )
    expected_campaign_hash = _require_sha256(
        contract.get("campaign_sha256"), label="Oracle63 campaign hash"
    )
    if (
        not campaign_path.is_file()
        or _sha256_file(campaign_path) != expected_campaign_hash
    ):
        raise Action20OracleAdmissionError("current Oracle63 campaign hash drifted")
    campaign = _load_json(campaign_path, label="current Oracle63 campaign")
    campaign_tasks = campaign.get("tasks")
    if (
        campaign.get("schema_version")
        != "behavior2026.action23-oracle63-campaign/v1"
        or not isinstance(campaign_tasks, list)
        or campaign.get("task_count") != len(campaign_tasks)
    ):
        raise Action20OracleAdmissionError("current Oracle63 campaign is invalid")
    campaign_identities = _normalize_task_universe(
        (
            (task.get("task_id"), task.get("task_name"))
            for task in campaign_tasks
            if isinstance(task, Mapping)
        ),
        label="current Oracle63 campaign",
    )
    if (
        len(campaign_identities) != len(campaign_tasks)
        or campaign_identities != identities
    ):
        raise Action20OracleAdmissionError(
            "current Oracle63 campaign task ID/name set drifted"
        )

    superseded = contract.get("supersedes")
    if not isinstance(superseded, Mapping):
        raise Action20OracleAdmissionError(
            "Oracle63 historical supersession record is missing"
        )
    legacy_path = _resolve_project_file(
        _require_nonempty_string(
            superseded.get("path"), label="legacy Oracle63 report path"
        )
    )
    legacy_hash = _require_sha256(
        superseded.get("sha256"), label="legacy Oracle63 report hash"
    )
    if not legacy_path.is_file() or _sha256_file(legacy_path) != legacy_hash:
        raise Action20OracleAdmissionError(
            "superseded Oracle63 historical report was not preserved"
        )
    if not isinstance(superseded.get("reason"), str) or not superseded["reason"]:
        raise Action20OracleAdmissionError("Oracle63 supersession reason is missing")
    legacy = _load_json(legacy_path, label="legacy Oracle63 report")
    legacy_profiles = legacy.get("profiles")
    if not isinstance(legacy_profiles, list):
        raise Action20OracleAdmissionError("legacy Oracle63 profiles are missing")
    legacy_identities = {
        (profile.get("task_id"), profile.get("task_name"))
        for profile in legacy_profiles
        if isinstance(profile, Mapping)
    }
    if excluded not in legacy_identities or required in legacy_identities:
        raise Action20OracleAdmissionError(
            "legacy Oracle63 report no longer preserves the Task52/Task66 error"
        )


def _validate_current_tree(payload: Mapping[str, Any]) -> None:
    attestation = payload.get("current_tree_attestation")
    if not isinstance(attestation, Mapping):
        raise Action20OracleAdmissionError("current_tree_attestation is missing")
    if attestation.get("schema_version") != CURRENT_TREE_ATTESTATION_SCHEMA:
        raise Action20OracleAdmissionError("current-tree attestation schema drifted")
    _validate_oracle63_exact_report(attestation)
    source_files = attestation.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise Action20OracleAdmissionError("current-tree source hashes are missing")
    observed_paths: set[str] = set()
    required_gates = {
        "dag_compile",
        "verifier_adapter",
        "action20_protocol",
        "public_privacy",
    }
    observed_gates: set[str] = set()
    for ordinal, item in enumerate(source_files):
        if not isinstance(item, Mapping):
            raise Action20OracleAdmissionError(
                f"current-tree source {ordinal} is invalid"
            )
        relative = _require_nonempty_string(
            item.get("path"), label=f"current-tree source {ordinal}.path"
        )
        if relative in observed_paths:
            raise Action20OracleAdmissionError("current-tree source path is duplicated")
        observed_paths.add(relative)
        gate = _require_nonempty_string(
            item.get("gate"), label=f"current-tree source {ordinal}.gate"
        )
        observed_gates.add(gate)
        expected = _require_sha256(
            item.get("sha256"), label=f"current-tree source {relative}"
        )
        path = _resolve_project_file(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise Action20OracleAdmissionError(
                f"current-tree source hash drifted: {relative}"
            )
    if not required_gates <= observed_gates:
        raise Action20OracleAdmissionError(
            "current-tree sources do not cover every static gate"
        )

    action20 = attestation.get("action20_contract")
    if not isinstance(action20, Mapping):
        raise Action20OracleAdmissionError("Action20 static contract is missing")
    if tuple(action20.get("strict_actions", ())) != STRICT_ACTION20:
        raise Action20OracleAdmissionError("strict Action20 action set drifted")
    if frozenset(action20.get("removed_actions", ())) != REMOVED_PUBLIC_ACTIONS:
        raise Action20OracleAdmissionError("removed Action20 actions drifted")
    if set(STRICT_ACTION20) & REMOVED_PUBLIC_ACTIONS:
        raise Action20OracleAdmissionError("removed action re-entered Action20")
    action_yaml = _resolve_project_file(str(action20.get("profile_path", "")))
    profile_text = action_yaml.read_text(encoding="utf-8")
    for action in STRICT_ACTION20:
        if f"  - {action}\n" not in profile_text:
            raise Action20OracleAdmissionError(
                f"Action20 YAML lacks {action!r}"
            )
    for action in REMOVED_PUBLIC_ACTIONS:
        if f"  - {action}\n" in profile_text.split("forbidden:", 1)[0]:
            raise Action20OracleAdmissionError(
                f"removed action {action!r} re-entered the public profile"
            )
    if "task_truth_to_model" not in profile_text:
        raise Action20OracleAdmissionError("Action20 privacy prohibition drifted")

    for task in payload["tasks"]:
        profile_contract = task["current_tree_profile"]
        if not isinstance(profile_contract, Mapping):
            raise Action20OracleAdmissionError("current_tree_profile is invalid")
        record = _find_profile(profile_contract)
        if _sha256_json(record) != profile_contract.get("record_sha256"):
            raise Action20OracleAdmissionError("current-tree profile record drifted")
        expected_identity = (
            task["task_id"],
            task["task_name"],
            task["scene"],
            task["split"],
            task["instance_id"],
        )
        observed_identity = (
            record.get("task_id"),
            record.get("task_name"),
            record.get("scene_model"),
            record.get("split"),
            record.get("instance_id"),
        )
        if observed_identity != expected_identity:
            raise Action20OracleAdmissionError(
                "current-tree problem/task identity drifted"
            )
        canonical = task["option_identity"]["canonical_index"]
        if record.get("canonical_option_index") != canonical.get("value"):
            raise Action20OracleAdmissionError(
                "current-tree canonical raw option index drifted"
            )
        semantic = semantic_identity_sha256(
            grounded_goal_option_identity(record.get("canonical_goal_leaves", ()))
        )
        if semantic != task["option_identity"].get(
            "canonical_semantic_identity_sha256"
        ):
            raise Action20OracleAdmissionError(
                "current-tree canonical semantic identity drifted"
            )

        problem_path = Path(
            _require_nonempty_string(
                profile_contract.get("problem_path"), label="problem_path"
            )
        )
        task_state_path = Path(
            _require_nonempty_string(
                profile_contract.get("task_state_path"), label="task_state_path"
            )
        )
        if not problem_path.is_absolute() or not task_state_path.is_absolute():
            raise Action20OracleAdmissionError(
                "problem and task-state paths must be absolute"
            )
        for label, path, expected_hash in (
            (
                "problem definition",
                problem_path,
                profile_contract.get("problem_sha256"),
            ),
            (
                "task state",
                task_state_path,
                profile_contract.get("task_state_sha256"),
            ),
        ):
            expected_hash = _require_sha256(expected_hash, label=label)
            if not path.is_file() or _sha256_file(path) != expected_hash:
                raise Action20OracleAdmissionError(f"current-tree {label} drifted")

        audit = audit_profile(record, catalog=str(profile_contract["catalog_path"]))
        checks = audit.get("checks")
        if (
            audit.get("compile_error") is not None
            or not isinstance(checks, Mapping)
            or checks.get("unbound_leaf_ids") != []
            or checks.get("multiple_achiever_leaf_ids") != []
            or checks.get("unknown_dependency_node_ids") != []
            or checks.get("cycle_or_non_topological") is not False
        ):
            raise Action20OracleAdmissionError(
                "current-tree selected-option DAG did not compile cleanly"
            )
        lowered = lower_profile(audit, profile_ordinal=int(task["task_id"]))
        public = lowered.get("public_protocol_template")
        if not isinstance(public, Mapping) or public.get("artifact_only_not_model_prompt") is not True:
            raise Action20OracleAdmissionError(
                "Action20 privacy boundary was not re-attested"
            )


def _validate_payload_shape(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(payload, expected=_ROOT_KEYS, label="admission root")
    if payload.get("schema_version") != ACTION20_ORACLE_ADMISSION_SCHEMA:
        raise Action20OracleAdmissionError("admission schema drifted")
    if payload.get("immutable") is not True:
        raise Action20OracleAdmissionError("admission sidecar is not immutable")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise Action20OracleAdmissionError("admission tasks are missing")
    if payload.get("task_count") != len(tasks):
        raise Action20OracleAdmissionError("admission task count drifted")
    identities: list[tuple[int, str]] = []
    counted = 0
    shadow = 0
    for ordinal, task in enumerate(tasks):
        task = _require_exact_keys(
            task, expected=_TASK_KEYS, label=f"admission task {ordinal}"
        )
        task_id = _require_int(task.get("task_id"), label="task_id")
        task_name = _require_nonempty_string(task.get("task_name"), label="task_name")
        _require_nonempty_string(task.get("scene"), label="scene")
        _require_nonempty_string(task.get("split"), label="split")
        _require_int(task.get("instance_id"), label="instance_id")
        _require_int(task.get("seed"), label="seed")
        identities.append((task_id, task_name))
        mode = task.get("admission_mode")
        contribution = task.get("count_contribution")
        if mode not in ADMISSION_MODES:
            raise Action20OracleAdmissionError("unknown admission mode")
        if mode == COUNTED:
            if contribution != 1:
                raise Action20OracleAdmissionError(
                    "COUNTED admission must contribute exactly one"
                )
            counted += 1
        else:
            if contribution != 0:
                raise Action20OracleAdmissionError(
                    "SHADOW_ONLY admission must contribute zero"
                )
            shadow += 1
        baseline = task.get("baseline_membership")
        if not isinstance(baseline, Mapping):
            raise Action20OracleAdmissionError("baseline membership is missing")
        if baseline.get("ledger_counts_mutated") is not False:
            raise Action20OracleAdmissionError(
                "an admission record attempted to mutate the frozen ledger"
            )
        if task_id == 98 and (
            mode != SHADOW_ONLY
            or contribution != 0
            or baseline.get("status") != "POST_LEDGER_Q1_SHADOW"
        ):
            raise Action20OracleAdmissionError(
                "Task98 must remain explicit SHADOW_ONLY with zero contribution"
            )
    _normalize_task_universe(identities, label="admission tasks")
    if payload.get("counted_task_count") != counted:
        raise Action20OracleAdmissionError("counted task count drifted")
    if payload.get("shadow_task_count") != shadow:
        raise Action20OracleAdmissionError("shadow task count drifted")
    ledger = payload.get("ledger")
    if not isinstance(ledger, Mapping):
        raise Action20OracleAdmissionError("frozen Oracle63 ledger is missing")
    if (
        ledger.get("universe_count") != 63
        or ledger.get("certified_count") != 41
        or ledger.get("unresolved_count") != 22
        or ledger.get("mutable_by_this_sidecar") is not False
        or ledger.get("certified_count") + ledger.get("unresolved_count")
        != ledger.get("universe_count")
    ):
        raise Action20OracleAdmissionError(
            "frozen 41-certified / 22-unresolved ledger drifted"
        )


def load_action20_oracle_admission(
    path: Path = DEFAULT_ACTION20_ORACLE_ADMISSION_PATH,
    *,
    expected_task_universe: Iterable[tuple[int, str]] | None = None,
    verify_artifacts: bool = True,
    verify_current_tree: bool = True,
) -> Action20OracleAdmission:
    """Load and fully validate one immutable Action20 Oracle admission.

    The whole sidecar is validated before any individual task can be returned.
    ``expected_task_universe`` is exact set equality, not a count check.
    Disabling a verification class is intended only for focused pure tests;
    campaign callers should use both defaults.
    """

    source = Path(path)
    if not source.is_file():
        raise Action20OracleAdmissionError(
            f"Action20 Oracle admission is missing: {source}"
        )
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Action20OracleAdmissionError("admission sidecar is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise Action20OracleAdmissionError("admission sidecar root must be an object")
    _validate_payload_shape(payload)

    actual_universe = frozenset(
        (int(task["task_id"]), str(task["task_name"])) for task in payload["tasks"]
    )
    if expected_task_universe is not None:
        expected = _normalize_task_universe(
            expected_task_universe, label="expected_task_universe"
        )
        if actual_universe != expected:
            missing = sorted(expected - actual_universe)
            unexpected = sorted(actual_universe - expected)
            raise Action20OracleAdmissionError(
                "admission task universe drifted: "
                f"missing={missing}, unexpected={unexpected}"
            )

    if verify_artifacts:
        for task in payload["tasks"]:
            _validate_artifact(task)
    if verify_current_tree:
        _validate_current_tree(payload)

    tasks = tuple(
        OracleTaskAdmission(
            task_id=int(task["task_id"]),
            task_name=str(task["task_name"]),
            admission_mode=str(task["admission_mode"]),
            count_contribution=int(task["count_contribution"]),
            record=_freeze(task),
        )
        for task in sorted(
            payload["tasks"], key=lambda item: (item["task_id"], item["task_name"])
        )
    )
    return Action20OracleAdmission(
        path=source.resolve(),
        sidecar_sha256=hashlib.sha256(raw).hexdigest(),
        tasks=tasks,
        counted_task_count=int(payload["counted_task_count"]),
        shadow_task_count=int(payload["shadow_task_count"]),
        current_tree_attested=bool(verify_current_tree),
        artifacts_attested=bool(verify_artifacts),
        ledger=_freeze(payload["ledger"]),
    )


def require_task_admitted(
    admission: Action20OracleAdmission,
    *,
    task_id: int,
    task_name: str,
    require_count_contribution: bool = True,
) -> OracleTaskAdmission:
    """Return one already-validated task, rejecting non-counting shadow use.

    A caller may explicitly set ``require_count_contribution=False`` for a
    shadow/promotion lane.  This does not weaken validation: the caller must
    first possess an :class:`Action20OracleAdmission` returned after whole-file
    artifact and current-tree checks.
    """

    if not isinstance(admission, Action20OracleAdmission):
        raise TypeError("admission must be a validated Action20OracleAdmission")
    if not admission.artifacts_attested or not admission.current_tree_attested:
        raise Action20OracleAdmissionError(
            "task consumption requires full artifact and current-tree attestation"
        )
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
        raise Action20OracleAdmissionError("requested task_id is invalid")
    if not isinstance(task_name, str) or not task_name:
        raise Action20OracleAdmissionError("requested task_name is invalid")
    matches = [
        task
        for task in admission.tasks
        if task.task_id == task_id and task.task_name == task_name
    ]
    if len(matches) != 1:
        raise Action20OracleAdmissionError(
            f"task is not admitted: {(task_id, task_name)!r}"
        )
    task = matches[0]
    if require_count_contribution and task.count_contribution != 1:
        raise Action20OracleAdmissionError(
            f"task {(task_id, task_name)!r} is {task.admission_mode}, not COUNTED"
        )
    return task
