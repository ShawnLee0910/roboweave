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


def process_record():
    record = official()
    record["snapshots"] = [{"options": [[True, True, True]]}]
    snapshot = dict(held_roles=[], prerequisites=[], protected_relations=[True])
    record["process"] = dict(roles=["protected"],
                             requirements=dict(no_manipulation_roles=["protected"], before_placement=[]),
                             initial_truth=[], final_truth=[],
                             events=[dict(action="pick", before=copy.deepcopy(snapshot), after=snapshot,
                                          status="completed", action_index=0)])
    return record


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("snapshot", [None, {}, [], "unavailable", False,
    {"held_roles": [], "prerequisites": []},
    {"held_roles": "protected", "prerequisites": [], "protected_relations": [True]},
    {"held_roles": [], "prerequisites": [], "protected_relations": [1]},
    {"held_roles": [], "prerequisites": [], "protected_relations": {}},
    {"held_roles": [[]], "prerequisites": [], "protected_relations": [True]},
    {"held_roles": ["unknown"], "prerequisites": [], "protected_relations": [True]},
])
def test_process_replay_rejects_missing_or_malformed_snapshots(tmp_path, phase, snapshot):
    record = process_record()
    record["process"]["events"][0][phase] = snapshot
    directory = tmp_path / "process"
    with pytest.raises(ValueError):
        score_record(record, process_directory=directory)
    assert not directory.exists()


def test_process_replay_rejects_dropped_protection_evidence(tmp_path):
    record = process_record()
    record["process"]["events"][0]["after"]["protected_relations"] = []
    with pytest.raises(ValueError, match="readback counts changed"):
        score_record(record, process_directory=tmp_path / "process")


@pytest.mark.parametrize("status,selected,success,failure", [
    ("completed", [], True, None),
    ("completed", ["protected"], False, "PUBLIC_PROCESS_REQUIREMENT_FAILED"),
    ("blocked", ["protected"], False, "VERIFIER_ERROR"),
])
def test_replay_keeps_native_pass_violation_and_unavailable_classification(tmp_path, status, selected, success, failure):
    from behavior_hl.embodied.evaluator_bridge.l1_process_audit import ProcessAudit
    record = process_record()
    process = record["process"]
    event = process["events"][0]
    event["status"] = status
    event["before"]["selected_roles"] = selected
    result = score_record(record, process_directory=tmp_path / "replay")
    native = ProcessAudit(process["requirements"], role_object=lambda role: role, recipe_truth=lambda *args: True,
                          protected_relations=(), directory=tmp_path / "native")
    native.observe(**event)
    expected = native.report()
    for field in ("operational", "public_process_satisfied", "violations", "errors", "execution_events"):
        assert result["process"][field] == expected[field]
    assert result["canonical"]["success"] is True
    assert result["final"]["q_score"] == 1
    assert result["final"]["success"] is success
    assert result["final"].get("failure_class") == failure


@pytest.mark.parametrize("states", [None, [], {}, [1]])
def test_state_contract_requires_boolean_readbacks(tmp_path, states):
    record = process_record()
    record["process"]["requirements"]["state_protection_roles"] = ["protected"]
    for phase in ("before", "after"):
        record["process"]["events"][0][phase]["protected_states"] = states
    with pytest.raises(ValueError, match="protected_states"):
        score_record(record, process_directory=tmp_path / "process")


def test_state_change_remains_native_process_failure(tmp_path):
    record = process_record()
    record["process"]["requirements"]["state_protection_roles"] = ["protected"]
    record["process"]["events"][0]["before"]["protected_states"] = [True]
    record["process"]["events"][0]["after"]["protected_states"] = [False]
    result = score_record(record, process_directory=tmp_path / "process")
    assert result["final"]["failure_class"] == "PUBLIC_PROCESS_REQUIREMENT_FAILED"
    assert result["process"]["violations"][0]["kind"] == "PROTECTED_STATE_CHANGED"


@pytest.mark.parametrize("ready,expected", [(True, True), (False, False)])
def test_placement_contract_requires_prerequisites_and_goal_readbacks(tmp_path, ready, expected):
    record = process_record()
    record["process"]["requirements"] = dict(no_manipulation_roles=[], before_placement=[
        dict(subject_role="protected", required_relations=[dict(predicate="closed", subject_role="protected", desired=True)])])
    event = record["process"]["events"][0]
    event["action"] = "place_on_top"
    event["before"].update(held_roles=["protected"], prerequisites=[ready], placement_goals=[False])
    event["after"].update(prerequisites=[True], placement_goals=[True])
    result = score_record(record, process_directory=tmp_path / "valid")
    assert result["final"]["success"] is expected
    del event["after"]["placement_goals"]
    with pytest.raises(ValueError, match="placement_goals"):
        score_record(record, process_directory=tmp_path / "missing")


def test_current_reference_requires_complete_domain_readbacks(tmp_path):
    record = process_record()
    process = record["process"]
    process["roles"] += ["target", "support"]
    process["requirements"]["current_reference_contracts"] = [dict(binding_id="current",
        time="CURRENT_READBACK_BEFORE_TARGET_PLACEMENT", update="CURRENT", cardinality="EXACTLY_ONE",
        predicate="ontop", subject_role="protected", domain_roles=["support"],
        goal_predicate="ontop", goal_subject_role="target")]
    event = process["events"][0]
    event["action"] = "place_on_top"
    event["before"].update(held_roles=["target"], current_references=[dict(source_matches=["support"], target_matches=[])])
    event["after"]["current_references"] = [dict(source_matches=["support"], target_matches=["support"])]
    process["final_truth"] = [dict(atom=dict(predicate="ontop", subject_role="target", object_role="support", desired=True), value=True)]
    assert score_record(record, process_directory=tmp_path / "valid")["final"]["success"]
    for index, value in enumerate(([], [{}], [dict(source_matches=["support"], target_matches="support")])):
        event["after"]["current_references"] = value
        with pytest.raises(ValueError, match="current_references"):
            score_record(record, process_directory=tmp_path / str(index))


def test_milestone_contract_requires_declared_boolean_readbacks(tmp_path):
    record = process_record()
    process = record["process"]
    atom = dict(predicate="open", subject_role="protected", desired=True)
    process["requirements"] = dict(no_manipulation_roles=[], before_placement=[], process_program=dict(
        schema_version="l1-public-process-program/v1", milestones=[dict(milestone_id="opened", after=[], requirements=[atom])],
        final_requirements=[atom], final_placement_after=None,
        completion_rule="ALL_DECLARED_MILESTONES_AND_CURRENT_FINAL_REQUIREMENTS",
        condition_evaluation="CANONICAL_READBACK_AFTER_VERIFIED_EXECUTION_OR_INITIAL_READBACK",
        milestone_memory="LATCH_ONCE_AFTER_DECLARED_PREDECESSORS_SATISFIED",
        source_process={}, public_instruction_evidence={"instruction": "Open the protected device."}))
    process["initial_truth"] = [dict(atom=atom, value=False)]
    process["final_truth"] = [dict(atom=atom, value=True)]
    event = process["events"][0]
    event["action"] = "open"
    event["before"]["milestones"] = dict(opened=False)
    event["after"]["milestones"] = dict(opened=True)
    assert score_record(record, process_directory=tmp_path / "valid")["final"]["success"]
    for index, value in enumerate((None, {}, dict(opened=1))):
        event["after"]["milestones"] = value
        with pytest.raises(ValueError, match="milestones"):
            score_record(record, process_directory=tmp_path / str(index))
    process["requirements"]["process_program"]["global_unary_phases"] = True
    event["before"]["milestones"] = dict(opened=False, _unary=dict(protected=False))
    event["after"]["milestones"] = dict(opened=True, _unary=dict(protected=True))
    assert score_record(record, process_directory=tmp_path / "unary-valid")["final"]["success"]
    for index, value in enumerate((None, {}, dict(protected=1))):
        event["after"]["milestones"]["_unary"] = value
        with pytest.raises(ValueError, match="_unary"):
            score_record(record, process_directory=tmp_path / f"unary-{index}")


def test_history_replay_rejects_absent_per_event_truth(tmp_path):
    record = process_record()
    atom = dict(predicate="open", subject_role="protected", desired=True)
    record["process"]["requirements"]["history_program"] = dict(schema_version="l1-history-bindings/v1",
        bindings=[], clauses=[atom], expression=atom,
        process=dict(ordered_milestones=[dict(milestone_id="opened", condition=atom)]))
    with pytest.raises(ValueError, match="per-event predicate truth"):
        score_record(record, process_directory=tmp_path / "process")
    assert not (tmp_path / "process").exists()
