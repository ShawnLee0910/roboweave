"""One committed Action20 effect, one external R1 event, unchanged goal verifier."""
from collections.abc import Mapping
from pathlib import Path
import json
import math
from types import SimpleNamespace

from .v_intervention import PoseInterventionSession, _read_ref, _require, _vector

SCHEMA = 'behavior.v-r1-package/v1'
ACTION_EFFECTS = {
    'place_inside': ('inside', True), 'place_on_top': ('ontop', True),
    'place_next_to': ('nextto', True), 'place_under': ('under', True),
    'place_on_floor': ('ontop', True), 'toggle_on': ('toggled_on', True),
    'toggle_off': ('toggled_on', False), 'open': ('open', True), 'close': ('open', False),
}
STATE_NAMES = {'inside': 'Inside', 'ontop': 'OnTop', 'nextto': 'NextTo', 'under': 'Under',
               'toggled_on': 'ToggledOn', 'open': 'Open'}


def load_r1_package(path, config, *, task_names, seed):
    if 'native_world' in config or 'history_attestation' in config.get('source', {}):
        from .v_r4_native_donor import load_r1_package as load_native
        return load_native(path, config, task_names=task_names, seed=seed)
    _require(set(config) == {'schema_version', 'package_id', 'task_spec_id', 'seed', 'public_notice',
        'task_specs', 'reset_pose_overlay', 'requested_execution_policy', 'trigger', 'landing',
        'source', 'prepared_state'}, 'unknown or missing R1 fields')
    _require(tuple(task_names) == (config['task_spec_id'],), 'R1 requires one exact source task')
    _require(type(config['seed']) is int and config['seed'] == seed, 'R1 seed mismatch')
    for key in ('package_id', 'task_spec_id', 'public_notice'):
        _require(isinstance(config[key], str) and config[key].strip(), 'invalid R1 ' + key)
    trigger = config['trigger']
    _require(set(trigger) == {'action', 'target_scopes', 'destination_scopes', 'predicate', 'desired'},
             'invalid R1 trigger fields')
    _require(trigger['action'] in ACTION_EFFECTS and type(trigger['desired']) is bool and
             (trigger['predicate'], trigger['desired']) == ACTION_EFFECTS[trigger['action']],
             'unsupported or mismatched R1 action effect')
    for key in ('target_scopes', 'destination_scopes'):
        values = trigger[key]
        _require(isinstance(values, list) and all(isinstance(s, str) and s for s in values)
                 and len(set(values)) == len(values), 'invalid R1 scope list')
    placement = trigger['action'].startswith('place_')
    _require(trigger['target_scopes'] and bool(trigger['destination_scopes']) == placement,
             'R1 state effects have no destination; placements need a destination')
    landing = config['landing']
    _require(set(landing) == {'max_candidates', 'max_distance_m', 'clearance_m'}, 'invalid R1 landing fields')
    _require(type(landing['max_candidates']) is int and 1 <= landing['max_candidates'] <= 25,
             'invalid R1 candidate budget')
    for key, upper, positive in [('max_distance_m', 1., True), ('clearance_m', .01, False)]:
        value = landing[key]
        _require(type(value) in (int, float) and math.isfinite(value) and
                 (value > 0 if positive else value >= 0) and value <= upper, 'invalid R1 ' + key)
    policy = config['requested_execution_policy']
    _require(policy == {'action_mode': 'Symbolic', 'physics_steps_per_action': 0,
                        'scene_load': 'Full', 'oracle_timeout_seconds_per_task': 1800}, 'unsupported R1 policy')
    source = config['source']
    _require(set(source) == {'base_package', 'base_task_spec_id', 'receipt'} and
             source['base_task_spec_id'] == config['task_spec_id'], 'R1 source identity mismatch')
    base_path, raw = _read_ref(source['base_package'], path.parent)
    base = json.loads(raw)
    _, receipt_bytes = _read_ref(source['receipt'], path.parent)
    receipt = json.loads(receipt_bytes)
    _require(receipt.get('status') == 'PASS' and receipt.get('official_q') == 1
             and receipt.get('official_success') is True and receipt.get('task_spec_id') == config['task_spec_id']
             and receipt.get('oracle_status') == 'COMPLETED'
             and receipt.get('oracle_execution_status') == 'COMPLETED'
             and receipt.get('verification_completed') is True, 'R1 source receipt is not complete PASS/Q1')
    adapter = base.get('runtime', {}).get('adapter')
    _require(adapter in {'fixed_l0_pool', 'registered_l0'}, 'unsupported R1 source adapter')
    _require(config['task_spec_id'] in base.get('task_spec_ids', []), 'R1 source not in frozen base')
    config['_base_package_path'], config['_source_adapter'] = str(base_path), adapter
    spec_path = None
    if config['task_specs'] is not None:
        spec_path, data = _read_ref(config['task_specs'], path.parent)
        selected = [json.loads(line) for line in data.splitlines() if line.strip()]
        selected = [s for s in selected if s['task_spec_id'] == config['task_spec_id']]
        _require(len(selected) == 1, 'missing or duplicate R1 TaskSpec')
        spec = selected[0]
        source_specs = base.get('task_specs')
        if source_specs is not None:
            _, original = _read_ref(source_specs, base_path.parent)
            originals = [json.loads(line) for line in original.splitlines() if line.strip()]
            originals = [s for s in originals if s['task_spec_id'] == config['task_spec_id']]
            _require(len(originals) == 1, 'missing source TaskSpec')
            _require(spec == originals[0], 'R1 must preserve complete source TaskSpec')
        scopes = spec.get('private', {}).get('scope_bindings')
        if scopes:
            _require(all(s in scopes for s in trigger['target_scopes'] + trigger['destination_scopes']),
                     'R1 trigger scope absent from source')
        _validate_trigger_goal(spec, trigger, base, base_path, adapter)
        if adapter != 'registered_l0':
            config['max_action_decisions'] = spec['public']['max_action_decisions']
        else:
            config['_task_specs_path'] = str(spec_path)
            spec_path = None
    else:
        raise ValueError('R1 requires exact source TaskSpecs for goal membership validation')
    overlay = None
    if config['reset_pose_overlay'] is not None:
        overlay, _ = _read_ref(config['reset_pose_overlay'], path.parent)
    if config['prepared_state'] is not None:
        _, data = _read_ref(config['prepared_state'], path.parent)
        bundle = json.loads(data)
        _require(isinstance(bundle, dict), 'invalid R1 prepared robot/camera bundle')
        config['_prepared_states'] = {config['task_spec_id']: bundle}
    return config, spec_path, overlay, policy


def _validate_trigger_goal(spec, trigger, base, base_path, adapter):
    """A trigger may regress only an explicitly inherited canonical goal atom."""
    def normalized(predicate, desired):
        aliases = {'closed': 'open', 'toggled_off': 'toggled_on'}
        return (aliases[predicate], not desired) if predicate in aliases else (predicate, desired)
    atoms = set()
    if adapter == 'registered_l0':
        catalog_path, raw = _read_ref(base['runtime']['registration_catalog'], base_path.parent)
        records = [r for r in json.loads(raw)['tasks'] if r['task_spec_id'] == spec['task_spec_id']]
        _require(len(records) == 1, 'R1 registration identity missing or duplicate')
        _, raw = _read_ref(records[0]['template_contract'], catalog_path.parent)
        roles = json.loads(raw)['role_to_scope']
        for clause in spec['goal_contract']['goal_clauses']:
            predicate, desired = normalized(clause['predicate'], clause.get('desired', True))
            subject = roles[clause['subject_role']]
            destinations = clause.get('destination_equivalence_roles') or [clause.get('object_role')]
            atoms.update((predicate, desired, subject, None if d is None else roles[d]) for d in destinations)
    else:
        for clause in spec['private']['goal_spec'].get('clauses', []):
            predicate, desired = normalized(clause['canonical_predicate'], clause.get('desired', True))
            destinations = clause.get('destination_equivalence_members') or clause.get('destination_scopes') or [None]
            atoms.update((predicate, desired, subject, destination)
                         for subject in clause['target_scopes'] for destination in destinations)
    requested = {(trigger['predicate'], trigger['desired'], subject, destination)
                 for subject in trigger['target_scopes'] for destination in (trigger['destination_scopes'] or [None])}
    _require(requested <= atoms, 'R1 trigger is not an inherited canonical goal atom')


class R1InterventionSession(PoseInterventionSession):
    def _refresh(self):
        for obj in self.objects.values():
            obj.clear_states_cache()

    def _effect(self, target_scope, destination_scope):
        from .spatial_symbolic_actions import read_action_relation
        self._refresh()
        trigger = self.config['trigger']
        value = read_action_relation(self.objects[target_scope], STATE_NAMES[trigger['predicate']],
            None if destination_scope is None else self.objects[destination_scope],
            predicate_mode='action20_symbolic')
        return bool(value) == trigger['desired']

    def before_action(self, runtime, read_snapshot, action_index=None):
        _require(not self.failed, 'R1 session invalid after intervention failure')
        _require(runtime.zero_physics and not runtime.placement_physical_consequences_enabled,
                 'R1 requires inherited Symbolic executor')
        snapshot = read_snapshot()
        _require(snapshot.ok, 'R1 verifier unavailable')
        trigger = self.config['trigger']
        if not self.initialized:
            from .spatial_symbolic_actions import object_aabb
            self.objects = {s: self._object(s) for s in trigger['target_scopes'] + trigger['destination_scopes']}
            _require(len({id(self.objects[s]) for s in trigger['target_scopes']}) == len(trigger['target_scopes']),
                     'R1 target aliases require an explicit contract')
            self.geometry = {}
            if trigger['action'].startswith('place_'):
                for scope in trigger['target_scopes']:
                    obj = self.objects[scope]
                    _require(not getattr(obj, 'fixed_base', False), 'R1 target is not movable')
                    pos, quat = obj.get_position_orientation()
                    lo, hi = object_aabb(obj)
                    self.geometry[scope] = {'position': _vector(pos), 'orientation': _vector(quat),
                                            'lower': _vector(lo), 'upper': _vector(hi)}
            self._save('initial.json', {'config': self.config, 'branch': self.branch,
                       'initial_snapshot': snapshot.to_private_record(), 'target_geometry': self.geometry})
            self.initialized = True
        if self.consumed:
            self.delivered = False
        self.pre_effects = {(s, d): self._effect(s, d) for s in trigger['target_scopes']
                            for d in (trigger['destination_scopes'] or [None])}
        held = runtime.symbolic_primitives._get_obj_in_hand()
        self.held_scope = next((s for s in trigger['target_scopes'] if self.objects[s] is held), None)
        marker_snapshot = getattr(runtime.store, 'active_snapshot', None)
        self.pre_bindings = {} if marker_snapshot is None else dict(marker_snapshot.bindings)
        return snapshot

    def _selected_pair(self, action):
        args = action.args
        marker = args.get('marker') if isinstance(args, Mapping) else getattr(args, 'marker', None)
        binding = self.pre_bindings.get(marker)
        if binding is None:
            return None
        name = getattr(binding, 'object_name', None)
        trigger = self.config['trigger']
        if action.action.startswith('place_'):
            if self.held_scope is None:
                return None
            destination = next((s for s in trigger['destination_scopes'] if self.objects[s].name == name), None)
            return None if destination is None else (self.held_scope, destination)
        target = next((s for s in trigger['target_scopes'] if self.objects[s].name == name), None)
        return None if target is None else (target, None)

    def _mutate_state(self, pair, event):
        from .spatial_symbolic_actions import state_by_name
        target = self.objects[pair[0]]
        trigger = self.config['trigger']
        state = state_by_name(target, STATE_NAMES[trigger['predicate']])
        _require(state is not None and callable(getattr(state, 'set_value', None)), 'R1 state setter unavailable')
        event['state_before'] = bool(state.get_value())
        if trigger['predicate'] == 'open':
            both_sides, _, _ = state.relevant_joints_info
            _require(not both_sides, 'R1 bidirectional articulated state requires a frozen direction contract')
            event['mutation_started'] = True
            accepted = state.set_value(not trigger['desired'], fully=True)
            event['state_setter_mode'] = 'fully_open_or_closed'
        else:
            event['mutation_started'] = True
            accepted = state.set_value(not trigger['desired'])
        _require(accepted is True, 'R1 state setter did not confirm mutation')
        self._refresh()
        event['state_after'] = bool(state.get_value())
        if trigger['predicate'] == 'open':
            import omnigibson as og
            from .v_legacy_final_collision import query_symbolic_placement_final_collision
            report = query_symbolic_placement_final_collision(omnigibson=og, moving_objects=(target,))
            event['articulated_final_collision'] = report.to_private_dict()
            _require(report.accepted, 'R1 articulated state changed into a collision')

    def _empty_target(self, target):
        from .construction_contents_readback import read_empty_contents
        from .spatial_symbolic_actions import state_by_name
        abilities = getattr(target, 'abilities', None)
        _require(isinstance(abilities, Mapping), 'R1 target abilities unavailable')
        if 'fillable' not in abilities and state_by_name(target, 'Contains') is None:
            return {'passed': True, 'basis': 'live_object_declares_no_fillable_ability'}
        return read_empty_contents(self.evaluator.env.scene, ['target'], lambda _: target)[0]

    def _mutate_pose(self, pair, event):
        """Reuse the finite surface/ray/collision kernels, never a saved world position."""
        import numpy as np
        import torch
        import omnigibson as og
        from omnigibson.utils.sampling_utils import raytest_batch
        from .spatial_symbolic_actions import (object_aabb, object_pose, symbolic_on_top_relation,
            _bounded_surface_candidates, candidate_aabb, build_downward_support_ray_requests,
            DownwardSupportRayEvidence, rebase_candidate_to_local_support)
        from .spatial_candidate_admission import (spatial_object_link_paths,
            query_spatial_candidate_collision_overlap)
        from .v_legacy_final_collision import query_symbolic_placement_final_collision
        target = self.objects[pair[0]]
        empty = self._empty_target(target)
        event['empty_contents'] = empty
        event['mutation_started'] = False
        self._save('event.json', event)
        _require(empty['passed'], 'R1 pose target carries contents; unsupported local mutation')
        saved = self.geometry[pair[0]]
        geometry = SimpleNamespace(aabb=(np.asarray(saved['lower']), np.asarray(saved['upper'])),
            get_position_orientation=lambda: (np.asarray(saved['position']), np.asarray(saved['orientation'])))
        origin, orientation = target.get_position_orientation()
        origin, orientation = origin.clone(), orientation.clone()
        anchor = np.asarray(_vector(origin))
        radius = self.config['landing']['max_distance_m']
        objects = self.evaluator.env.scene.objects
        scene_objects = tuple(objects.values() if isinstance(objects, Mapping) else objects)
        supports = []
        for support in scene_objects:
            if support is target or any(support is robot for robot in self.evaluator.env.robots):
                continue
            lo, hi = object_aabb(support)
            closest = np.maximum(lo[:2], np.minimum(anchor[:2], hi[:2]))
            if np.linalg.norm(closest - anchor[:2]) <= radius and abs(hi[2] - anchor[2]) <= radius:
                supports.append(support)
        supports.sort(key=lambda s: abs(object_aabb(s)[1][2] - anchor[2]))
        candidates = []
        for support in supports:
            lo, hi = object_aabb(support)
            candidates.extend((candidate, support) for candidate in _bounded_surface_candidates(geometry,
                surface_min_xy=np.maximum(lo[:2], anchor[:2] - radius),
                surface_max_xy=np.minimum(hi[:2], anchor[:2] + radius),
                support_top_z=float(hi[2]), name_prefix='r1_near_'))
        candidates.sort(key=lambda row: float(np.linalg.norm(row[0].position - anchor)))
        event['landing_candidates'] = []
        ignored = spatial_object_link_paths(target)
        for candidate, support in candidates[:self.config['landing']['max_candidates']]:
            record = {'candidate': candidate.name, 'support': support.name}
            event['landing_candidates'].append(record)
            if np.linalg.norm(candidate.position[:2] - anchor[:2]) > radius:
                record['rejected'] = 'outside_frozen_radius'
                continue
            requests = build_downward_support_ray_requests(geometry, candidate)
            hits = raytest_batch(torch.as_tensor(np.asarray([r.start_xyz for r in requests]), dtype=torch.float32),
                torch.as_tensor(np.asarray([r.end_xyz for r in requests]), dtype=torch.float32),
                only_closest=True, ignore_bodies=ignored, ignore_collisions=ignored)
            paths = spatial_object_link_paths(support)
            evidence = []
            for request, hit in zip(requests, hits, strict=True):
                path = str(hit.get('rigidBody', ''))
                matched = any(path == p or path.startswith(p + '/') for p in paths)
                evidence.append(DownwardSupportRayEvidence(sample_name=request.sample_name,
                    start_xyz=request.start_xyz, end_xyz=request.end_xyz,
                    hit_position_xyz=hit.get('position') if hit.get('hit') else None,
                    hit_normal_xyz=hit.get('normal') if hit.get('hit') else None,
                    hit_entity_key=path if hit.get('hit') else None,
                    entity_role='declared_support' if matched else 'structural_obstacle', error=None))
            rebase = rebase_candidate_to_local_support(geometry, candidate, evidence,
                clearance_m=self.config['landing']['clearance_m'])
            record['support_readback'] = rebase.to_private_dict()
            if not rebase.accepted:
                continue
            candidate = rebase.rebased_candidate
            collision = query_spatial_candidate_collision_overlap(omnigibson=og, held=target,
                closure_objects=(target,), candidate=candidate, contact_objects=(support,),
                candidate_aabb_world=candidate_aabb(geometry, candidate), xy_margin_m=0.)
            record['collision'] = collision.to_private_dict()
            if not collision.accepted:
                continue
            event['mutation_started'] = True
            target.set_position_orientation(position=torch.as_tensor(candidate.position, dtype=origin.dtype, device=origin.device),
                orientation=torch.as_tensor(candidate.orientation_xyzw, dtype=orientation.dtype, device=orientation.device))
            target.set_linear_velocity(torch.zeros_like(origin))
            target.set_angular_velocity(torch.zeros_like(origin))
            self._refresh()
            final = query_symbolic_placement_final_collision(omnigibson=og, moving_objects=(target,),
                                                            contact_objects=(support,))
            record['final_collision'] = final.to_private_dict()
            record['effect_remains_satisfied'] = self._effect(*pair)
            snapshot = self._event_read_snapshot()
            changed = ([] if not snapshot.ok else [key for key in self._event_before.leaf_truth
                       if self._event_before.leaf_truth[key] != snapshot.leaf_truth.get(key)])
            record['changed_goal_leaves'] = changed
            preserves_other_goals = (snapshot.ok and snapshot.leaf_truth.keys() == self._event_before.leaf_truth.keys()
                and len(changed) == 1 and self._event_before.leaf_truth[changed[0]] and not snapshot.leaf_truth[changed[0]])
            if (final.accepted and symbolic_on_top_relation(target, support) and
                    not record['effect_remains_satisfied'] and preserves_other_goals):
                event['selected_position'], event['selected_support'] = _vector(candidate.position), support.name
                return
            target.set_position_orientation(position=origin, orientation=orientation)
            self._refresh()
        raise ValueError('no supported R1 effect-breaking landing in frozen candidate budget')

    def after_action(self, runtime, action, result, action_index, read_snapshot):
        self._seen_actions.add(action_index)
        self.action_count = len(self._seen_actions)
        if self.consumed or result.status != 'completed' or action.action != self.config['trigger']['action']:
            return False
        pair = self._selected_pair(action)
        if pair is None or self.pre_effects[pair] or not self._effect(*pair):
            return False
        return self.apply_event(runtime, action, result, action_index, read_snapshot, pair)

    def apply_event(self, runtime, action, result, action_index, read_snapshot, pair, *,
                    effect_action=None, event_extra=None):
        effect_action = action.action if effect_action is None else effect_action
        owner = runtime.symbolic_primitives
        if action.action.startswith('place_'):
            _require(owner._get_obj_in_hand() is None and getattr(owner, '_logical_inventory_snapshot', None) is None,
                     'R1 placement has not fully released')
        owner.require_marker_visual_projection_inactive()
        before = read_snapshot()
        _require(before.ok, 'R1 pre-event verifier unavailable')
        event = {'branch': self.branch, 'status': 'STARTED', 'action_index': action_index,
                 'action_receipt_status': result.status, 'target_scope': pair[0], 'destination_scope': pair[1],
                 'before': before.to_private_record(), 'physics_steps': 0, 'mutation_started': False,
                 **(event_extra or {})}
        self.consumed, self.delivered = True, False
        self._event_before, self._event_read_snapshot = before, read_snapshot
        self._save('event.json', event)
        try:
            if self.branch == 'D1':
                if effect_action.startswith('place_'):
                    self._mutate_pose(pair, event)
                else:
                    self._mutate_state(pair, event)
            runtime.store.clear()
            self._refresh()
            after = read_snapshot()
            _require(after.ok and before.leaf_truth.keys() == after.leaf_truth.keys(), 'R1 goal readback changed shape')
            changed = [key for key in before.leaf_truth if before.leaf_truth[key] != after.leaf_truth[key]]
            if self.branch == 'D0':
                _require(not changed and self._effect(*pair), 'R1 no-op changed a goal')
            else:
                _require(not self._effect(*pair) and len(changed) == 1 and before.leaf_truth[changed[0]]
                         and not after.leaf_truth[changed[0]], 'R1 requires exactly one regressed goal leaf and no collateral change')
            self.generation += 1
            event.update(status='APPLIED', after=after.to_private_record(), changed_goal_leaves=changed,
                         generation=self.generation)
            self.event = event
            self._save('event.json', event)
            return True
        except BaseException as exc:
            self.failed = True
            event.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            self._save('event.json', event)
            from behavior_hl.embodied.runtime.action_integrity import invalidate_execution
            invalidate_execution(self.evaluator, exc)
            raise
