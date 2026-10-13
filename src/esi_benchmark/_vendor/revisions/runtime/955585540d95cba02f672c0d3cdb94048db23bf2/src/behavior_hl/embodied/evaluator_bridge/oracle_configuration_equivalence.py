"""Conservative within-package Oracle reuse; never a model-evaluation filter."""
from copy import deepcopy
import hashlib
import json

SCHEMA = 'oracle-configuration-equivalence/v1'


def execution_signature(task):
    record = deepcopy(task.record)
    branch = record.get('task_contract')
    if (not isinstance(branch, dict)
            or branch.get('schema_version') not in ('l1-initial-branch/v1', 'l1-initial-branch/v2')
            or branch.get('pair_member') not in ('control', 'challenge')):
        return None
    # Only declared identity / natural-language differences are removed. Unknown
    # fields and all observation/process/branch conditions remain in the key.
    for key in ('candidate_id', 'activity_name', 'task_spec_id', 'task_id', 'public_instruction'):
        record.pop(key, None)
    for key in ('task_name', 'task_id', 'task_language'):
        record['profile'].pop(key, None)
    branch.pop('pair_member')
    contract = deepcopy(task.template_contract())
    contract.pop('candidate_id', None)
    contract.pop('template_contract_sha256', None)
    contract['task_contract'].pop('pair_member', None)
    record['template_contract'] = contract
    for field, path in (('template', task.template_path), ('tro_state', task.tro_state_path)):
        record[field] = hashlib.sha256(path.read_bytes()).hexdigest()
    bddl = task.bddl_path.read_text()
    header = '(problem ' + task.record['activity_name'] + '-0)'
    if bddl.count(header) != 1:
        return None  # No broad string replacement inside goals or object names.
    record['bddl'] = bddl.replace(header, '(problem oracle-equivalent-0)', 1)
    payload = {'schema': SCHEMA, 'record': record, 'reset_pose_record': task.reset_pose_record}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def group_equivalent_configurations(ids, registered):
    representatives = []
    aliases = {}
    seen = {}
    fingerprints = {}
    for name in ids:
        signature = execution_signature(registered[name])
        if signature is not None:
            fingerprints[name] = signature
        if signature is not None and signature in seen:
            aliases[name] = seen[signature]
        else:
            representatives.append(name)
            if signature is not None:
                seen[signature] = name
    return representatives, {'schema_version': SCHEMA, 'requested_task_names': list(ids),
        'aliases': aliases, 'fingerprints': fingerprints,
        'scope': 'same_package_seed_runtime_execution_policy',
        'evidence_scope': 'equivalent_authored_execution_inputs_not_independent_live_reproduction',
        'model_evaluation_deduplicated': False}
