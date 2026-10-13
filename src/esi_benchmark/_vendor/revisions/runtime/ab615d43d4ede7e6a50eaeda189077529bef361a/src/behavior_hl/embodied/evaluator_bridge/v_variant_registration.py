"""Process-local native KB registration for an explicit logical task variant."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .inside_task_profiles import InsideTaskProfile
from .registered_l0_task_runtime import configure_registered_l0_tasks


def variant_artifact(spec, name):
    ref = spec['private']['source_artifacts'][name]
    path = Path(ref['path']).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('V variant artifact changed: ' + name)
    return path


def configure_variant_task(evaluator_module, spec, prepared, *, tro_state_path=None):
    """Keep native binding and projected goal on the same new BDDL identity."""
    identity = spec['private']['source_task_identity']
    name = identity['task_name']
    if name != spec['task_spec_id'] or identity['instance_id'] != 0:
        raise ValueError('V variant needs a unique activity equal to its TaskSpec ID and instance zero')
    bindings = spec['private']['scope_bindings']
    clauses = spec['private']['goal_spec']['clauses']
    targets = {s for c in clauses for s in c['target_scopes']}
    destinations = {s for c in clauses for s in c['destination_scopes']}
    synset = lambda s: s.rsplit('_', 1)[0]
    graspable = {synset(s) for s in targets}
    profile = InsideTaskProfile(**identity, task_language=spec['public']['instruction'],
        graspable_synsets=frozenset(graspable), container_synsets=frozenset(synset(s) for s in destinations),
        public_category_by_synset={synset(s): bindings[s]['category'] for s in targets},
        expected_graspable_count=len(targets), expected_container_count=len(destinations),
        expected_admitted_count=len(targets | destinations), expected_goal_leaf_count=len(clauses),
        canonical_goal_leaves=tuple(json.dumps([c['canonical_predicate'], c['target_scopes'][0],
                                               c['destination_scopes'][0]], separators=(',', ':')) for c in clauses))
    task = SimpleNamespace(**identity, candidate_id=spec['task_spec_id'], record=dict(spec),
        bddl_path=variant_artifact(spec, 'bddl'), template_path=variant_artifact(spec, 'template'),
        tro_state_path=Path(tro_state_path) if tro_state_path is not None else variant_artifact(spec, 'tro_state'), profile=lambda: profile,
        template_contract=lambda: {'scope_bindings': bindings},
        available_task=lambda: {0: {'scene_model': identity['scene_model'],
            'robot_start_position': list(prepared['robot_position']),
            'robot_start_orientation': list(prepared['robot_orientation']),
            'load_room_instances': None}})
    return configure_registered_l0_tasks(evaluator_module, {spec['task_spec_id']: task}, full_scene_load=True)
