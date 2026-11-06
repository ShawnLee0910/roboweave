"""Use exported start proposals in the loaded scene, without physical-action admission."""
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
from time import monotonic

from .resident_scene_batch import write_json_atomic


class CandidatePreflightError(RuntimeError):
    def __init__(self, reason, report, *, session_invalid=False):
        super().__init__(reason)
        self.private_report = report
        self.session_invalid = session_invalid


def select_shared_rank(*, row, family_members, family_state_path, source_hash,
                       try_candidate, restore, apply_selected, deadline):
    """A family's chosen rank cannot silently change between members or workers."""
    family_state_path = Path(family_state_path)
    family_state_path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version="oracle.symbolic-candidate-preflight/v1", task_spec_id=row['task_spec_id'],
                  status="FAIL", oracle_status="NOT_RUN", attempts=[], bounded=True,
                  cpu_search_metadata={key: value for key, value in row['oracle_preflight'].items() if key != 'candidates'},
                  complete_search=row.get("complete_search", False),
                  full_physical_action_path_check="NOT_RUN",
                  complete_visual_pair_audit="NOT_RUN", fair_fixed_start_admission="NOT_PROVEN",
                  policy="symbolic_exported_start_candidates_v1",
                  source_family_id=row["source_family_id"], source_candidates_sha256=source_hash)
    with family_state_path.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CandidatePreflightError("PREFLIGHT_FAMILY_BUSY", report) from exc
        state = json.loads(family_state_path.read_text()) if family_state_path.exists() else {
            "source_candidates_sha256": source_hash, "members": sorted(family_members),
            "rank": None, "validated_members": [], "failed_members": []}
        if state['source_candidates_sha256'] != source_hash or state['members'] != sorted(family_members):
            raise CandidatePreflightError("PREFLIGHT_FAMILY_IDENTITY_MISMATCH", report)
        candidates = row['oracle_preflight']['candidates']
        if state['rank'] is not None:
            candidates = [c for c in candidates if c['rank'] == state['rank']]
        def restore_checked():
            try:
                restore()
            except Exception as exc:
                raise CandidatePreflightError("PREFLIGHT_RESTORE_FAILED", report, session_invalid=True) from exc
        for candidate in candidates:
            if monotonic() >= deadline:
                report['budget_exhausted'] = True
                break
            restore_checked()
            try:
                prepared = try_candidate(candidate, deadline)
            except CandidatePreflightError as exc:
                report['attempts'].append(dict(rank=candidate['rank'], status='rejected', reason=str(exc), detail=exc.private_report))
                restore_checked()
                if exc.session_invalid:
                    raise
                continue
            except Exception:
                restore_checked()
                raise
            restore_checked()
            try:
                apply_selected(candidate)
            except CandidatePreflightError:
                restore_checked()
                raise
            except Exception as exc:
                restore_checked()
                raise CandidatePreflightError("PREFLIGHT_SELECTED_STATE_APPLY_FAILED", report) from exc
            state['rank'] = candidate['rank']
            state['validated_members'] = sorted(set(state['validated_members']) | {row['task_spec_id']})
            state['failed_members'] = [m for m in state['failed_members'] if m != row['task_spec_id']]
            state['family_common_rank_proven'] = state['validated_members'] == state['members'] and not state['failed_members']
            try:
                write_json_atomic(family_state_path, state)
            except Exception as exc:
                restore_checked()
                raise CandidatePreflightError("PREFLIGHT_EVIDENCE_WRITE_FAILED", report) from exc
            report.update(status='PASS', selected_rank=candidate['rank'], preparation=prepared,
                          family_state_path=str(family_state_path), family_common_rank_proven=state['family_common_rank_proven'],
                          selected_cpu_proposal=candidate, exact_route_chain_check='NOT_RUN')
            report['attempts'].append(dict(rank=candidate['rank'], status='accepted'))
            return report
        restore_checked()
        if state['rank'] is not None:
            state['failed_members'] = sorted(set(state['failed_members']) | {row['task_spec_id']})
            state['family_common_rank_proven'] = False
            write_json_atomic(family_state_path, state)
        raise CandidatePreflightError("PREFLIGHT_NO_VALID_CANDIDATE", report)


def prepare_exported_start(evaluator, *, row, family_members, state_dir, source_hash, initial_state_valid):
    import omnigibson as og
    import torch
    from .nearby_initial_pose import prepare_nearby_initial_pose, measured_base_admission
    from .private_traversability_route import check_official_local_segment

    robot, scene = evaluator.robot, evaluator.env.scene
    baseline = deepcopy(og.sim.dump_state(serialized=False))
    selected = {}
    def restore():
        og.sim.load_state(deepcopy(baseline), serialized=False)
        og.sim.render()
    def try_candidate(candidate, deadline):
        pose = candidate['robot_pose']
        reference, quaternion = robot.get_position_orientation()
        position = torch.as_tensor(pose['position'], dtype=reference.dtype, device=reference.device)
        orientation = torch.as_tensor(pose['orientation'], dtype=quaternion.dtype, device=quaternion.device)
        import math
        x, y, z, w = (float(v) for v in orientation)
        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        guard = check_official_local_segment(scene=scene, robot=robot,
            start_xy=pose['position'][:2], target_xy=pose['position'][:2], robot_z=pose['position'][2],
            start_yaw_rad=yaw, target_yaw_rad=yaw, allow_bounded_source_snap=False, allow_test_fallback=False)
        if not guard.traversable:
            raise CandidatePreflightError('PREFLIGHT_STATIC_BASE_REJECTED',
                dict(rank=candidate['rank'], local_guard_private=dict(guard.private_report)))
        robot.set_position_orientation(position, orientation)
        og.sim.render()
        if not initial_state_valid():
            raise CandidatePreflightError('PREFLIGHT_INITIAL_SEMANTICS_REJECTED', {})
        try:
            preparation = prepare_nearby_initial_pose(evaluator, initial_state_valid=initial_state_valid,
                enable_nearby_search=False, deadline_monotonic=deadline)
        except RuntimeError as exc:
            raise CandidatePreflightError('PREFLIGHT_PRESENTATION_REJECTED', getattr(exc, 'private_report', {}),
                                          session_invalid=getattr(exc, 'session_invalid', False)) from exc
        selected['state'] = deepcopy(og.sim.dump_state(serialized=False))
        return preparation
    def apply_selected(candidate):
        og.sim.load_state(selected['state'], serialized=False)
        og.sim.render()
        guard = measured_base_admission(scene, robot)
        if guard['traversable'] is not True:
            raise CandidatePreflightError('PREFLIGHT_SELECTED_BASE_REJECTED', guard)
        if not initial_state_valid():
            raise RuntimeError('selected initial semantics changed')
    key = hashlib.sha256(row['source_family_id'].encode()).hexdigest()
    return select_shared_rank(row=row, family_members=family_members,
        family_state_path=Path(state_dir)/f'{key}.json', source_hash=source_hash,
        try_candidate=try_candidate, restore=restore, apply_selected=apply_selected, deadline=monotonic()+600)
