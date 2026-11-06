import json
from pathlib import Path
import pytest
from roboweave.offline import run_replay
from roboweave.trajectory import inspect_run, summarize_runs


def test_record_replay_uses_marker_protocol_and_preserves_evidence(tmp_path):
    output = tmp_path / 'replay'
    result = run_replay({'run': {'history_frames': 20}}, output)
    summary = result['summary']
    assert result['mode'] == 'offline_replay' and result['official_score'] is None
    assert summary['status'] == 'DONE'
    assert summary['model_calls'] == 4 and summary['actions_sent_to_scene'] == 3
    assert summary['macro_actions_resolved'] == 1
    transcript = json.loads((output/'conversation_private.json').read_text())
    assert [event['marker_stage'] for event in transcript['events'] if event['kind'] == 'observation'] == [None, 1, 2, None]
    inspection = inspect_run(output)
    assert inspection['status'] == 'PASS' and inspection['images'] == 4
    assert inspection['official_success'] is None and not inspection['score_available']
    report = summarize_runs([output])
    assert report['counts']['offline_replay'] == 1
    assert report['scored_live_runs'] == 0 and report['success_rate_scored_live'] is None
    with pytest.raises(FileExistsError):
        run_replay({}, output)


def test_image_reference_cannot_escape_replay(tmp_path):
    output = tmp_path / 'replay'
    run_replay({}, output)
    path = output / 'conversation_chatml.json'
    value = json.loads(path.read_text())
    value['messages'][1]['content'][1]['image'] = '../outside.png'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='escaped'):
        inspect_run(output)


def test_missing_image_and_modified_prompt_are_rejected(tmp_path):
    output = tmp_path / 'replay'
    run_replay({}, output)
    (output/'system_prompt.txt').write_text('changed')
    with pytest.raises(ValueError, match='provenance'):
        inspect_run(output)


def test_history_variants_are_recorded_and_not_combined_as_baseline(tmp_path):
    baseline = tmp_path/'history5'
    variant = tmp_path/'history20'
    first = run_replay({}, baseline)
    run_replay({'run': {'history_frames': 20}}, variant)
    assert first['summary']['physical_history_max'] == 5
    assert inspect_run(baseline)['evaluation_variant'] == 'frozen_history_5'
    assert inspect_run(variant)['evaluation_variant'] == 'history_20_variant_not_baseline'
    report = summarize_runs([baseline, variant])
    assert report['evaluation_variant_counts'] == {'frozen_history_5': 1, 'history_20_variant_not_baseline': 1}
    assert report['mixed_evaluation_variants'] is True
    assert summarize_runs([baseline])['mixed_evaluation_variants'] is False
    record = json.loads((variant/'run_record.json').read_text())
    record['evaluation_variant'] = 'frozen_history_5'
    (variant/'run_record.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='physical history'):
        inspect_run(variant)


@pytest.mark.parametrize('profile', ['l1-a30', 'l1-m30-public-v2', 'l1-v30',
                                    'l1-p-inside5', 'l1-p-ontop5', 'l1-p-additions20',
                                    'l2-collection37'])
def test_each_frozen_profile_replays_without_mixing_source_identity(tmp_path, profile):
    from roboweave.profiles import source_revisions
    output = tmp_path/profile
    result = run_replay({'profile': profile}, output)
    assert result['summary']['status'] == 'DONE'
    assert result['summary']['model_calls'] == 4
    record = json.loads((output/'run_record.json').read_text())
    assert record['profile'] == profile
    assert record['source_revisions'] == source_revisions(profile)
    assert record['source_verification']['profile'] == profile
    assert record['source_verification']['status'] == 'PASS'
    assert record['official_score'] is None
