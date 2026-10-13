import pytest
from test_public_oracle_goal_and_marker_recovery import _run


@pytest.mark.parametrize('missing,passed,publications,navigation', [(1, True, 2, 0), (2, True, 3, 1), (9, False, 3, 1)])
@pytest.mark.parametrize('mode', [None, 'empty'])
def test_missing_target_is_bounded_and_uses_fresh_publications(monkeypatch, tmp_path, missing, passed, publications, navigation, mode):
    run = _run(monkeypatch, tmp_path, missing_pick_publications=missing, pick_publication_mutation=mode)
    assert (run.result['status'] == 'PASS') == passed, run.result.get('error')
    picks = [r for r in run.calls if r[0] == 'pick']
    assert sum(not args for _,args,_ in picks) == publications
    assert sum(bool(args) for _,args,_ in picks) == int(passed)
    assert len(run.navigation) == navigation
    assert sum(r[0] == 'look_down' for r in run.calls) == 1
    if navigation:
        assert run.navigation[0]['failed_view_xy'] == ((0., 0.),)
    if not passed:
        assert run.result['failure_code'] == 'PICK_TARGET_MARKER_RECOVERY_EXHAUSTED'


@pytest.mark.parametrize('mutation', ['held', 'truth', 'physics', 'pose', 'stale'])
def test_unsafe_or_stale_publication_never_recovers(monkeypatch, tmp_path, mutation):
    run = _run(monkeypatch, tmp_path, missing_pick_publications=9, pick_publication_mutation=mutation)
    assert run.result['status'] == 'FAIL_EXCEPTION'
    assert not run.navigation
    assert not any(r[0] == 'look_down' for r in run.calls)
    assert sum(r[0] == 'pick' for r in run.calls) == 1


def test_blocked_view_still_has_only_one_station_retry(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, missing_pick_publications=9, pick_view_status='blocked')
    assert run.result['failure_code'] == 'PICK_TARGET_MARKER_RECOVERY_EXHAUSTED'
    assert len(run.navigation) == 1
    assert sum(r[0] == 'pick' for r in run.calls) == 3


def test_no_route_does_not_replenish_budget(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, missing_pick_publications=9, no_route=True)
    assert run.result['status'] == 'FAIL_EXCEPTION'
    assert len(run.navigation) == 1
    assert sum(r[0] == 'pick' for r in run.calls) == 2


def test_centered_target_still_gets_one_bounded_view_probe(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, missing_pick_publications=1, pick_view_centered=True)
    assert run.result['status'] == 'PASS', run.result.get('error')
    assert sum(r[0] == 'look_down' for r in run.calls) == 1
    assert not run.navigation
