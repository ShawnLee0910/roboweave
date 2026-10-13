import copy
import json
from pathlib import Path

import pytest

from esi_benchmark.scoring import score_record


def official():
    return dict(schema_version="esi-benchmark/score-record-v1", route="official_full",
                initial_options=[[True, False, False]], snapshots=[{"options": [ [True, True, False]]}])


def test_official_new_credit_and_integrity():
    record = official()
    assert score_record(record)["final"]["q_score"] == pytest.approx(1/3)
    record["snapshots"][0]["options"] = [[True, True, True]]
    assert score_record(record)["final"]["success"]
    record["snapshots"][0]["integrity_ok"] = False
    assert score_record(record)["final"] == dict(operational=False, success=False, q_score=None,
                                                errors=["recorded execution integrity failure"])


def test_projected_stability_initial_credit_and_missing_evidence():
    path = Path(__file__).parents[1] / "examples/score-record.json"
    record = json.loads(path.read_text())
    result = score_record(record)
    assert not result["snapshots"][0]["success"]
    assert result["final"]["success"]
    record["initial_truth"][0]["value"] = True
    result = score_record(record)
    assert not result["final"]["success"]
    assert result["final"]["q_score"] == 0
    assert result["final"]["early_done"]
    record["snapshots"][-1]["truth"] = []
    assert score_record(record)["final"]["operational"] is False


def test_routes_and_shapes_cannot_be_substituted():
    record = official()
    record["snapshots"][0]["options"] = [[True]]
    with pytest.raises(ValueError, match="structure"):
        score_record(record)
    record = official()
    record["initial_options"] = [[1, False, False]]
    with pytest.raises(ValueError, match="boolean"):
        score_record(record)


def test_process_gate_preserves_canonical_q_and_rejects_missing_history():
    record = official()
    record["snapshots"] = [{"options": [[True, True, True]]}]
    record["requires_process"] = True
    result = score_record(record)
    assert result["canonical"]["success"] is True
    assert result["final"]["q_score"] == 1
    assert result["final"]["success"] is False
    assert result["final"]["failure_class"] == "VERIFIER_ERROR"


def test_original_process_audit_rejects_protected_object(tmp_path):
    record = official()
    record["snapshots"] = [{"options": [[True, True, True]]}]
    before = dict(held_roles=[], prerequisites=[], protected_relations=[True])
    after = dict(before, held_roles=["protected"])
    record["process"] = dict(roles=["protected"], requirements=dict(no_manipulation_roles=["protected"], before_placement=[]),
                             initial_truth=[], final_truth=[], events=[dict(action="pick", before=before, after=after,
                                                                           status="completed", action_index=0)])
    result = score_record(record, process_directory=tmp_path / "process")
    assert not result["final"]["success"]
    assert result["final"]["failure_class"] == "PUBLIC_PROCESS_REQUIREMENT_FAILED"
    assert result["process"]["violations"][0]["kind"] == "PROTECTED_OBJECT_MANIPULATED_OR_RELATION_LOST"
    assert (tmp_path / "process/private/task_process_events.jsonl").is_file()
