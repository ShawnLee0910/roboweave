"""Opt-in one-shot external pose regression on an unchanged L0 task.

No task IDs, alternative goals, model-visible answers or scene resets live here.
The normal executor owns actions; this module owns only the external event.
"""
from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
import math

SCHEMA = 'behavior.v-pose-package/v1'


def _require(value, message):
    if not value:
        raise ValueError(message)


def _read_ref(ref, root):
    _require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'invalid V file reference')
    path = (root / ref['path']).resolve()
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == ref['sha256'], 'V source hash mismatch: ' + str(path))
    return path, data


def load_package(path, *, task_names, seed):
    """Validate the small V envelope before any simulator import."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config.get('schema_version') == 'behavior.v-r4-batch/v1':
        raise ValueError('legacy batch envelopes require their frozen Runtime')
    if config.get('schema_version') == 'behavior.v-r1-batch/v1':
        return load_r1_batch(path, config, task_names=task_names, seed=seed)
    if config.get('schema_version') == 'behavior.v-r4-package/v1':
        from .r4_package_gate import enabled
        if enabled(config):
            from .v_r4_intervention import load_r4_package
        else:
            from .v_r4_legacy import load_r4_package
        return load_r4_package(path, config, task_names=task_names, seed=seed)
    if config.get('schema_version') == 'behavior.v-r3-package/v1':
        from .v_r3_intervention import load_r3_package
        return load_r3_package(path, config, task_names=task_names, seed=seed)
    if config.get('schema_version') == 'behavior.v-r1-package/v1':
        from .v_r1_intervention import load_r1_package
        return load_r1_package(path, config, task_names=task_names, seed=seed)
    if config.get('schema_version') == 'behavior.v-flower-package/v1':
        raise ValueError('flower variants require their frozen Runtime')
    _require(set(config) == {'schema_version', 'package_id', 'base_package', 'task_spec_id',
             'seed', 'public_notice', 'target_scope', 'target_name', 'destination_scope',
             'source_support_scope', 'source_support_name', 'synchronization_physics_steps'},
             'unknown or missing V package fields')
    _require(config['schema_version'] == SCHEMA, 'unsupported V package schema')
    _require(tuple(task_names) == (config['task_spec_id'],), 'V requires one exact source task')
    _require(type(config['seed']) is int and config['seed'] == seed, 'V seed mismatch')
    _require(type(config['synchronization_physics_steps']) is int and
             config['synchronization_physics_steps'] == 0,
             'pose regression currently supports Symbolic render-only synchronization')
    for key in set(config) - {'seed', 'base_package', 'synchronization_physics_steps'}:
        _require(isinstance(config[key], str) and bool(config[key].strip()), 'invalid V field: ' + key)
    parent_path, parent_bytes = _read_ref(config['base_package'], path.parent)
    parent = json.loads(parent_bytes)
    _require(parent.get('schema_version') == 'action20-capability-benchmark.package/v1' and
             parent.get('stage') == 'COMPILED' and parent.get('runtime', {}).get('adapter') == 'fixed_l0_pool',
             'V pose package requires an existing compiled fixed L0 pool')
    _require(config['task_spec_id'] in parent['task_spec_ids'], 'V source not in parent package')
    spec_path, spec_bytes = _read_ref(parent['task_specs'], parent_path.parent)
    specs = [json.loads(line) for line in spec_bytes.splitlines() if line.strip()]
    selected = [s for s in specs if s['task_spec_id'] == config['task_spec_id']]
    _require(len(selected) == 1, 'missing or duplicate V source TaskSpec')
    spec = selected[0]
    _require(spec['private']['verifier']['route'] == 'official_projection', 'V source must use a projected goal')
    clauses = spec['private']['goal_spec']['clauses']
    _require(len(clauses) == 1 and clauses[0]['canonical_predicate'] == 'ontop' and
             clauses[0]['target_scopes'] == [config['target_scope']] and
             clauses[0]['destination_scopes'] == [config['destination_scope']] and
             clauses[0].get('destination_equivalence_members') == [config['destination_scope']],
             'this V contract requires one unchanged OnTop leaf')
    bindings = spec['private']['scope_bindings']
    _require(bindings[config['target_scope']]['actual_name'] == config['target_name'], 'V object binding mismatch')
    _read_ref(parent['runtime']['pool'], parent_path.parent)
    overlay_path, _ = _read_ref(parent['runtime']['reset_pose_overlay'], parent_path.parent)
    policy = parent['requested_execution_policy']
    _require(policy['action_mode'] == 'Symbolic' and policy['physics_steps_per_action'] == 0,
             'V pose contract cannot change the source action mode')
    config['max_action_decisions'] = spec['public']['max_action_decisions']
    return config, spec_path, overlay_path, policy


def load_r1_batch(path, index, *, task_names, seed):
    """Select same-scene members; each keeps its own event and prepared state."""
    _require(set(index) == {'schema_version', 'task_count', 'members', 'scene_loading_policy', 'oracle_status'},
             'invalid R1 batch fields')
    rows = index['members']
    _require(isinstance(rows, list) and type(index['task_count']) is int and len(rows) == index['task_count'],
             'R1 batch count mismatch')
    members = {row['task_spec_id']: row for row in rows}
    names = tuple(task_names)
    _require(len(members) == len(rows) and names and len(set(names)) == len(names)
             and set(names) <= set(members), 'R1 batch membership mismatch')
    configs, policies = {}, []
    for name in names:
        member_path, raw = _read_ref(members[name]['v_package'], path.parent)
        member = json.loads(raw)
        _require(member.get('schema_version') == 'behavior.v-r1-package/v1', 'nested R1 batch forbidden')
        config, _, overlay, policy = load_package(member_path, task_names=(name,), seed=seed)
        _require(config['_source_adapter'] == 'registered_l0' and overlay is None,
                 'R1 batch requires frozen registered members')
        configs[name] = config
        policies.append(policy)
    first = configs[names[0]]
    _require(all(c['_base_package_path'] == first['_base_package_path'] for c in configs.values())
             and all(p == policies[0] for p in policies), 'R1 batch source/policy mismatch')
    base_path = Path(first['_base_package_path'])
    base = json.loads(base_path.read_text())
    _, raw = _read_ref(base['runtime']['registration_catalog'], base_path.parent)
    catalog = {r['task_spec_id']: r for r in json.loads(raw)['tasks']}
    _require(len({catalog[n]['scene_model'] for n in names}) == 1, 'R1 resident batch must share a scene')
    config = dict(first)
    config['_member_configs'] = configs
    config['_prepared_states'] = {n: configs[n]['_prepared_states'][n] for n in names}
    config['package_id'] = 'r1-batch:' + ','.join(c['package_id'] for c in configs.values())
    return config, None, None, policies[0]


def get_session(evaluator):
    return getattr(evaluator, '_v_intervention_session', None)


def install_session(evaluator, config, branch, output_dir):
    _require(branch in {'D0', 'D1'}, 'unknown V branch')
    previous = get_session(evaluator)
    if previous is not None:
        previous.retire()
    from .v_protocol import protocol_name
    protocol = protocol_name(config)
    _require(protocol is not None, 'unsupported V intervention protocol')
    if config.get('schema_version') == 'behavior.v-r4-package/v1':
        from .r4_package_gate import enabled
        if enabled(config):
            from .v_r4_intervention import R4InterventionSession
        else:
            from .v_r4_legacy import R4InterventionSession
        session = R4InterventionSession(evaluator, config, branch, output_dir)
    elif config.get('schema_version') == 'behavior.v-r3-package/v1':
        from .v_r3_intervention import R3InterventionSession
        session = R3InterventionSession(evaluator, config, branch, output_dir)
    elif config.get('schema_version') == 'behavior.v-r1-package/v1':
        from .v_r1_intervention import R1InterventionSession
        session = R1InterventionSession(evaluator, config, branch, output_dir)
    elif config.get('schema_version') == 'behavior.v-flower-package/v1':
        raise ValueError('flower variants require their frozen Runtime')
    else:
        session = PoseInterventionSession(evaluator, config, branch, output_dir)
    evaluator._v_intervention_session = session
    session._save('protocol_identity.json', {'protocol': protocol, 'package_id': config.get('package_id'),
                  'branch': branch, 'implementation_module': type(session).__module__,
                  'historical_success_reused_as_current_validation': False})
    return session


def _vector(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    result = [float(x) for x in value]
    _require(all(math.isfinite(x) for x in result), 'non-finite V pose')
    return result


class PoseInterventionSession:
    def __init__(self, evaluator, config, branch, output_dir):
        self.evaluator, self.config, self.branch = evaluator, dict(config), branch
        self._cleanup_gate = None
        self.reset_episode(output_dir)

    def reset_episode(self, output_dir):
        self.retire()
        self.output_dir = Path(output_dir) / 'private' / 'v_intervention'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _require(not (self.output_dir / 'initial.json').exists(), 'V episode artifact identity reused')
        self.initialized = self.consumed = self.failed = self.delivered = False
        self.generation = 0
        self.action_count = 0
        self.last_goal = False
        self.event = None
        self._seen_actions = set()
        from .v_goal_termination import install_goal_termination_gate
        self._cleanup_gate = install_goal_termination_gate(self.evaluator.env.task)

    def retire(self):
        if self._cleanup_gate is not None:
            self._cleanup_gate()
            self._cleanup_gate = None

    def _save(self, name, payload):
        path = self.output_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')

    def _object(self, scope_name):
        obj = self.evaluator.env.task.object_scope[scope_name]
        # BDDL scope wrappers expose the live object via wrapped_obj.
        obj = getattr(obj, 'wrapped_obj', obj)
        _require(obj is not None, 'unbound V object: ' + scope_name)
        return obj

    def _refresh(self):
        for obj in (self.target, self.support, self.destination):
            obj.clear_states_cache()

    def _empty(self):
        from .construction_contents_readback import read_empty_contents
        evidence = read_empty_contents(self.evaluator.env.scene, ['target'], lambda role: self.target)[0]
        _require(evidence['passed'], 'V pose regression requires an empty container; contents are never removed')
        return evidence

    def _source_collision(self):
        import numpy as np
        import omnigibson as og
        from .spatial_symbolic_actions import SpatialPoseCandidate
        from .spatial_candidate_admission import query_spatial_candidate_collision_overlap
        return query_spatial_candidate_collision_overlap(
            omnigibson=og, held=self.target, closure_objects=(self.target,),
            candidate=SpatialPoseCandidate('v_prepared_pose', np.asarray(self.position), np.asarray(self.orientation)),
            contact_objects=(self.support,), xy_margin_m=0.0,
            candidate_aabb_world=tuple(np.asarray(v) for v in self.source_aabb))

    def _source_relation(self):
        from .r1pro_b1_component_builders import _read_marker_symbolic_canonical_state
        return bool(_read_marker_symbolic_canonical_state(self.target, 'OnTop', self.support,
                                                       symbolic_on_top=True))

    def before_action(self, runtime, read_snapshot, action_index=None):
        _require(callable(read_snapshot), 'V execution requires a canonical verifier')
        _require(not self.failed, 'V session invalid after intervention failure')
        # The normal model/Oracle owner enforces its existing macro/time budget.
        # Marker microsteps are recorded here, never charged as extra macros.
        if self.consumed:
            self.delivered = False
        _require(runtime.zero_physics and not runtime.placement_physical_consequences_enabled,
                 'V pose package requires the inherited Symbolic executor')
        snapshot = read_snapshot()
        _require(snapshot.ok, 'V verifier unavailable')
        if not self.initialized:
            _require(not snapshot.official_success, 'V prepared goal must initially be false')
            self.target = self._object(self.config['target_scope'])
            self.support = self._object(self.config['source_support_scope'])
            self.destination = self._object(self.config['destination_scope'])
            _require(self.target.name == self.config['target_name'] and
                     self.support.name == self.config['source_support_name'], 'V live binding changed')
            _require(not getattr(self.target, 'fixed_base', False), 'V target must be movable')
            self._refresh()
            _require(self._source_relation(), 'V prepared source support relation is false')
            contents = self._empty()
            position, orientation = self.target.get_position_orientation()
            self.position, self.orientation = _vector(position), _vector(orientation)
            from .spatial_symbolic_actions import object_aabb
            lower, upper = object_aabb(self.target)
            self.source_aabb = (_vector(lower), _vector(upper))
            _require(len(self.position) == 3 and len(self.orientation) == 4, 'invalid source pose shape')
            self._save('initial.json', {'config': self.config, 'branch': self.branch,
                       'prepared_source_pose': {'position': self.position, 'orientation_xyzw': self.orientation},
                       'prepared_source_aabb_world': {'lower': self.source_aabb[0], 'upper': self.source_aabb[1]},
                       'empty_contents': contents, 'initial_snapshot': snapshot.to_private_record(),
                       'prepared_pose_collision_diagnostic': self._source_collision().to_private_dict(),
                       'pose_source': 'live_prepared_state_before_first_action', 'physics_steps': 0})
            self.initialized = True
        self.last_goal = bool(snapshot.official_success)
        return snapshot

    def after_action(self, runtime, action, result, action_index, read_snapshot):
        # Navigation substeps share one public sequence. They are not new decisions.
        self._seen_actions.add(action_index)
        self.action_count = len(self._seen_actions)
        current = read_snapshot()
        _require(current.ok, 'V post-action verifier unavailable')
        if self.consumed or result.status != 'completed' or action.action != 'place_on_top' or self.last_goal or not current.official_success:
            return False
        owner = runtime.symbolic_primitives
        _require(owner._get_obj_in_hand() is None, 'V target not released')
        _require(getattr(owner, '_logical_inventory_snapshot', None) is None, 'V carry closure remains active')
        owner.require_marker_visual_projection_inactive()
        self._refresh()
        contents = self._empty()
        event = {'branch': self.branch, 'action_index': action_index,
                 'action_receipt_status': result.status, 'before': current.to_private_record(),
                 'empty_contents': contents, 'physics_steps': 0, 'status': 'STARTED'}
        self.consumed = True  # Consume before mutation: an exception must never replay it.
        self.delivered = False
        self._save('event.json', event)
        try:
            if self.branch == 'D1':
                import torch
                import omnigibson as og
                from .spatial_candidate_admission import query_symbolic_placement_final_collision
                # Restoring orientation requires prepared geometry, not a
                # translation of the last placement's differently oriented box.
                collision = self._source_collision()
                event['source_aabb_world_private'] = {'lower': self.source_aabb[0], 'upper': self.source_aabb[1]}
                event['target_pose_before_mutation_private'] = {
                    'position': _vector(self.target.get_position_orientation()[0]),
                    'orientation_xyzw': _vector(self.target.get_position_orientation()[1])}
                event['collision_before_private'] = collision.to_private_dict()
                _require(collision.accepted, 'V source pose is occupied or collision query failed')
                position, orientation = self.target.get_position_orientation()
                self.target.set_position_orientation(
                    position=torch.as_tensor(self.position, dtype=position.dtype, device=position.device),
                    orientation=torch.as_tensor(self.orientation, dtype=orientation.dtype, device=orientation.device))
                self.target.set_linear_velocity(torch.zeros_like(position))
                self.target.set_angular_velocity(torch.zeros_like(position))
                self._refresh()
                collision_after = query_symbolic_placement_final_collision(
                    omnigibson=og, moving_objects=(self.target,), contact_objects=(self.support,))
                event['collision_after_private'] = collision_after.to_private_dict()
                _require(collision_after.accepted and self._source_relation(), 'V restore lacks collision-free support')
            runtime.store.clear()
            self._refresh()
            after = read_snapshot()
            _require(after.ok and bool(after.official_success) == (self.branch == 'D0'),
                     'V intervention did not create the declared result')
            self.generation += 1
            self.event = event
            event.update(status='APPLIED', after=after.to_private_record(), generation=self.generation)
            self._save('event.json', event)
            return True
        except BaseException as exc:
            self.failed = True
            event.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            self._save('event.json', event)
            from behavior_hl.embodied.runtime.action_integrity import invalidate_execution
            invalidate_execution(self.evaluator, exc)
            raise

    def observation_delivered(self):
        if self.consumed and not self.failed:
            self.delivered = True

    def finish(self, reason, snapshot):
        valid = bool(self.initialized and self.consumed and self.delivered and not self.failed)
        report = {'branch': self.branch, 'trigger_reached': self.consumed,
                  'intervention_valid': self.event is not None and not self.failed,
                  'fresh_feedback_delivered': self.delivered, 'v_protocol_complete': valid,
                  'termination_reason': reason, 'canonical_goal_success': bool(snapshot.ok and snapshot.official_success),
                  'actions_seen': self.action_count, 'actions_seen_scope': 'distinct_executor_sequences_including_marker_microsteps',
                  'budget_owner': 'inherited_model_or_oracle_loop', 'physics_steps': 0,
                  'feedback_necessity': 'NOT_ESTABLISHED', 'visual_sufficiency': 'NOT_REVIEWED'}
        self._save('final.json', report)
        self.retire()
        return report
