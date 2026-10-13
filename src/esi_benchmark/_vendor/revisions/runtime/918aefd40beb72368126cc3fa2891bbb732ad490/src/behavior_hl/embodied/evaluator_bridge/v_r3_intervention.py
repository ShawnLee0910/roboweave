"""R3 remembers a completed goal A and regresses it only after a different goal B."""
from copy import deepcopy
import json
from itertools import islice
from pathlib import Path

from .v_intervention import _require, _read_ref
from .v_r1_intervention import R1InterventionSession, load_r1_package, ACTION_EFFECTS

SCHEMA = 'behavior.v-r3-package/v1'


def load_r3_package(path, config, *, task_names, seed):
    _require(config.get('schema_version') == SCHEMA and 'later_trigger' in config, 'missing R3 later trigger')
    if 'historical_full_source' in config:
        return load_historical_full_package(path, config, task_names=task_names, seed=seed)
    base = deepcopy(config)
    later = base.pop('later_trigger')
    base['schema_version'] = 'behavior.v-r1-package/v1'
    primary, spec, overlay, policy = load_r1_package(path, base, task_names=task_names, seed=seed)
    other = deepcopy(config)
    other.pop('later_trigger')
    other['schema_version'], other['trigger'] = 'behavior.v-r1-package/v1', later
    load_r1_package(path, other, task_names=task_names, seed=seed)
    a = primary['trigger']
    def identities(trigger):
        return {(trigger['predicate'], trigger['desired'], s, d)
                for s in trigger['target_scopes'] for d in (trigger['destination_scopes'] or [None])}
    _require(any(first != second for first in identities(a) for second in identities(later)),
             'R3 requires distinct old and later goal effects')
    primary.update(schema_version=SCHEMA, later_trigger=later)
    return primary, spec, overlay, policy


def load_historical_full_package(path, config, *, task_names, seed):
    """Separate legacy full-task evidence; never weaken modern R1 receipt checks."""
    from .canonical_bddl_grounder import problem_goal, object_map_from_scope, iter_ground_options
    from .l0_taskspec_live_adapter import L0TaskSpecLiveAdapter
    from .action23_outcome_verifier import Action23OutcomeVerifier
    _require(set(config) == {'schema_version','package_id','task_spec_id','seed','public_notice','task_specs',
        'reset_pose_overlay','requested_execution_policy','trigger','later_trigger','landing',
        'historical_full_source','prepared_state'}, 'unknown historical R3 envelope fields')
    _require(tuple(task_names) == (config['task_spec_id'],) and type(seed) is int and type(config['seed']) is int
             and seed == config['seed'], 'historical R3 task/seed mismatch')
    _require(config['prepared_state'] is None and config['reset_pose_overlay'] is None,
             'historical full task uses standard native reset, not invented prepared state')
    policy=config['requested_execution_policy']
    _require(policy == {'action_mode':'Symbolic','physics_steps_per_action':0,'scene_load':'Full',
                        'oracle_timeout_seconds_per_task':1800}, 'historical R3 execution policy mismatch')
    landing=config['landing']
    _require(set(landing)=={'max_candidates','max_distance_m','clearance_m'} and
             type(landing['max_candidates']) is int and 1<=landing['max_candidates']<=25 and
             type(landing['max_distance_m']) in (int,float) and 0<landing['max_distance_m']<=1 and
             type(landing['clearance_m']) in (int,float) and 0<=landing['clearance_m']<=.01,
             'invalid historical R3 landing limits')
    _,raw=_read_ref(config['historical_full_source'],path.parent);source=json.loads(raw)
    _require(source['schema_version']=='behavior.r3-full-household-source/v1' and
             source['reset_mode']=='STANDARD_NATIVE_RESET', 'invalid full household source mode')
    _,raw=_read_ref(source['ledger'],path.parent);ledger=json.loads(raw)
    matched=[r for r in ledger['tasks'] if r['task_id']==source['task_id']]
    _require(len(matched)==1,'historical task identity absent or duplicate');historical=matched[0]
    _require(historical['task_name']==source['task_name'] and historical['seed']==seed,
             'historical full task identity changed')
    for key,original in [('summary',historical['artifacts']['summary']),('verifier',historical['artifacts']['private_verifier']),
                         ('trace',historical['journal']['exact_actions']['path']),('source_freeze_manifest',historical['artifacts']['source_freeze_manifest'])]:
        _require(source[key]['path']==original,'full source provenance path changed')
        _read_ref(source[key],path.parent)
    _,raw=_read_ref(source['summary'],path.parent);summary=json.loads(raw)
    _,raw=_read_ref(source['verifier'],path.parent);final=json.loads(raw)['final']
    _require(summary.get('status')=='PASS' and summary.get('official_q')==1 and summary.get('official_success') is True and
             final.get('ok') is True and final.get('official_q')==1 and final.get('official_success') is True,
             'historical full source is not complete PASS/Q1')
    _,raw=_read_ref(source['source_freeze_manifest'],path.parent)
    _require(json.loads(raw)['aggregate_sha256']==source['source_freeze_aggregate_sha256']==historical['source']['source_freeze_aggregate_sha256'],
             'historical source freeze identity mismatch')
    spec_path,raw=_read_ref(config['task_specs'],path.parent)
    specs=[json.loads(line) for line in raw.splitlines() if line.strip()]
    matched=[s for s in specs if s['task_spec_id']==config['task_spec_id']]
    _require(len(matched)==1,'historical full TaskSpec absent or duplicate');spec=matched[0]
    _require(source.get('instruction_provenance', {}).get('mode') == 'NEW_FAITHFUL_RENDERING_NOT_HISTORICAL_VERBATIM'
             and spec['public']['instruction'] == source.get('instruction_en')
             and spec['public'].get('instruction_zh') == source.get('instruction_zh'),
             'historical full goal requires an explicit faithful public instruction')
    expected={k:historical[k] for k in ('task_id','task_name','scene_model','split','instance_id')}
    _require(spec['private']['source_task_identity']==expected,'full TaskSpec source identity changed')
    _require(spec['private']['verifier']['route']=='official_full' and spec['private']['goal_spec']['kind']=='OFFICIAL_SOURCE_GOAL'
             and not spec['private']['goal_spec'].get('clauses'), 'historical goal cannot be projected or truncated')
    for key in ('bddl','template','tro_state'):
        _require(spec['private']['source_artifacts'][key]==source[key],'full source file identity changed')
        _read_ref(source[key],path.parent)
    _,bddl=_read_ref(source['bddl'],path.parent);goal=problem_goal(bddl.decode())
    _require(goal==source['full_goal_ast']==spec['private']['goal_spec']['full_goal_ast'],'complete original BDDL goal changed')
    _,raw=_read_ref(source['template'],path.parent);template=json.loads(raw)
    bindings=spec['private']['scope_bindings'];names=template['metadata']['task']['inst_to_name']
    _require(all(names.get(s)==binding['actual_name'] for s,binding in bindings.items()),'full native object binding changed')
    options=list(islice(iter_ground_options(goal,object_map_from_scope(bindings)),4097))
    _require(len(options)<=4096,'full trigger membership exceeds bounded CPU grounding')
    atoms=set()
    for option in options:
        for leaf in option:
            desired=True
            if leaf[0]=='not':leaf=leaf[1];desired=False
            atoms.add((leaf[0],desired,*leaf[1:]))
    pairs=[]
    for trigger in (config['trigger'],config['later_trigger']):
        _require(set(trigger)=={'action','predicate','desired','target_scopes','destination_scopes'} and
                 type(trigger['desired']) is bool and ACTION_EFFECTS.get(trigger['action'])==(trigger['predicate'],trigger['desired']),
                 'historical trigger action/effect mismatch')
        desired={(trigger['predicate'],trigger['desired'],s,*([d] if d else []))
                 for s in trigger['target_scopes'] for d in (trigger['destination_scopes'] or [None])}
        _require(desired and desired<=atoms,'R3 trigger not in complete original grounded goal');pairs.append(desired)
    _require(any(a!=b for a in pairs[0] for b in pairs[1]),'R3 requires distinct original effects')
    L0TaskSpecLiveAdapter(spec,stability_window=1,predicate_mode='action20_symbolic',
                         action_outcome_verifier=Action23OutcomeVerifier(lambda *a:False,lambda:None))
    config.update(_source_adapter='historical_full_household',_source_evidence_mode='HISTORICAL_FULL_RAW_SUMMARY_VERIFIER_PASS',
                  _native_reset_mode='STANDARD_NATIVE_RESET',_task_specs_path=str(spec_path),
                  max_action_decisions=spec['public']['max_action_decisions'])
    return config,spec_path,None,policy


class R3InterventionSession(R1InterventionSession):
    def reset_episode(self, output_dir):
        super().reset_episode(output_dir)
        self.old_goal = None

    def _later(self, method, *args):
        # Reuse exact R1 effect/binding code with B's declared operation.
        trigger, held = self.config['trigger'], self.held_scope
        self.config['trigger'], self.held_scope = self.config['later_trigger'], self.later_held_scope
        try:
            return method(*args)
        finally:
            self.config['trigger'], self.held_scope = trigger, held

    def before_action(self, runtime, read_snapshot, action_index=None):
        snapshot = super().before_action(runtime, read_snapshot, action_index)
        later = self.config['later_trigger']
        for scope in later['target_scopes'] + later['destination_scopes']:
            if scope not in self.objects:
                self.objects[scope] = self._object(scope)
        held = runtime.symbolic_primitives._get_obj_in_hand()
        self.later_held_scope = next((s for s in later['target_scopes'] if self.objects[s] is held), None)
        self.later_pre_effects = {(s, d): self._later(self._effect, s, d)
            for s in later['target_scopes'] for d in (later['destination_scopes'] or [None])}
        self.before_snapshot = snapshot
        return snapshot

    def after_action(self, runtime, action, result, action_index, read_snapshot):
        self._seen_actions.add(action_index)
        self.action_count = len(self._seen_actions)
        if self.consumed or result.status != 'completed':
            return False
        current = read_snapshot()
        _require(current.ok, 'R3 post-action verifier unavailable')
        if self.old_goal is None:
            if action.action != self.config['trigger']['action']:
                return False
            pair = self._selected_pair(action)
            if pair is None or self.pre_effects[pair] or not self._effect(*pair):
                return False
            previous = self.before_snapshot
            changed = [key for key in current.leaf_truth if previous.leaf_truth.get(key) != current.leaf_truth[key]]
            _require(previous.leaf_truth.keys() == current.leaf_truth.keys() and len(changed) == 1
                     and current.leaf_truth[changed[0]] and not previous.leaf_truth[changed[0]],
                     'R3 A must be one newly completed original goal leaf')
            if action.action.startswith('place_'):
                owner = runtime.symbolic_primitives
                _require(owner._get_obj_in_hand() is None and getattr(owner, '_logical_inventory_snapshot', None) is None,
                         'R3 old goal placement not released')
            self.old_goal = {'pair': pair, 'action_index': action_index, 'action': action.action,
                             'leaf_id': changed[0], 'snapshot': current.to_private_record()}
            self._save('old_goal_completed.json', self.old_goal)
            return False  # The action that completes A can never also trigger the event.
        old = self.old_goal
        if action_index <= old['action_index'] or action.action != self.config['later_trigger']['action']:
            return False
        pair = self._later(self._selected_pair, action)
        if pair is None or self.later_pre_effects[pair] or not self._later(self._effect, *pair):
            return False
        a, b = self.config['trigger'], self.config['later_trigger']
        if (a['predicate'], a['desired'], *old['pair']) == (b['predicate'], b['desired'], *pair):
            return False
        if not self.before_snapshot.leaf_truth.get(old['leaf_id']) or not current.leaf_truth.get(old['leaf_id']) or not self._effect(*old['pair']):
            return False  # Natural regression is not an authored R3 event.
        new_leaves = [key for key in current.leaf_truth if current.leaf_truth[key]
                      and self.before_snapshot.leaf_truth.get(key) is False and key != old['leaf_id']]
        if not new_leaves:
            return False
        if a['action'].startswith('place_'):
            owner = runtime.symbolic_primitives
            _require(owner._get_obj_in_hand() is None and getattr(owner, '_logical_inventory_snapshot', None) is None,
                     'R3 cannot move an old goal object while a carry closure is active')
        return self.apply_event(runtime, action, result, action_index, read_snapshot, old['pair'],
            effect_action=a['action'], event_extra={'recovery_class': 'R3', 'old_goal_completed': old,
                'later_action': action.action, 'later_target_scope': pair[0], 'later_destination_scope': pair[1],
                'later_new_goal_leaves': new_leaves, 'later_snapshot': current.to_private_record()})
