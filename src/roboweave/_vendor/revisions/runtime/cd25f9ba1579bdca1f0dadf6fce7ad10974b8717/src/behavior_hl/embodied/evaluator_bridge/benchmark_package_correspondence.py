"""Read-only correspondence checks for compiled perception TaskSpecs.

File hashes lock bytes; these checks independently compare the task contract with
what the registered runner consumes. They do not certify geometry or RGB.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical_bddl_grounder import find_section, parse_bddl


TASK_SCHEMA = "action20-capability-benchmark.l0-perception-canary/v1"
L1_BRANCH_SCHEMA = 'action20-capability-benchmark.l1-initial-branch-task/v1'
L1_PROCESS_SCHEMA = 'action20-capability-benchmark.l1-initial-branch-task/v2'
_NEGATED = {"closed": "open", "toggled_off": "toggled_on"}
_CONSTRUCTION_FIELDS = (
    "initial_state_recipe", "role_orientations_xyzw", "empty_contents_roles",
    "material_rules", "visual_rules", "pair_base_task_spec_id",
    "final_initial_readback_required", "runtime_validation_pending",
    "camera_relative_orientations", "marker_label_counterbalance", "calibration_probe",
)


def _content(value: Any) -> Any:
    """A relocated reference retains its content identity, not its old path."""
    if isinstance(value, Mapping):
        return {k: _content(v) for k, v in value.items()
                if not (k == "path" and "sha256" in value)}
    if isinstance(value, list):
        return [_content(v) for v in value]
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    if _content(actual) != _content(expected):
        raise ValueError(f"task/runtime correspondence mismatch: {label}")


def _formula(value: Any) -> Any:
    """Normalize concrete literals and unordered and/or, retaining all branches."""
    if isinstance(value, str):
        return value.removeprefix("?")
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("invalid task/runtime logical expression")
    operator, *args = value
    if operator in {"and", "or"}:
        children = [_formula(x) for x in args]
        flat = []
        for child in children:
            flat.extend(child[1:] if isinstance(child, tuple) and child[0] == operator else [child])
        ordered = tuple(sorted(flat, key=repr))
        return ordered[0] if len(ordered) == 1 else (operator, *ordered)
    if operator == "not":
        if len(args) != 1:
            raise ValueError("invalid negated task/runtime expression")
        child = _formula(args[0])
        return child[1] if child[0] == "not" else ("not", child)
    if operator in {"forall", "exists", "forn", "forpairs", "fornpairs", "imply"}:
        raise ValueError("unsupported quantified TaskSpec correspondence contract")
    if not all(isinstance(x, str) for x in args):
        raise ValueError("invalid task/runtime atomic arguments")
    literal = (_NEGATED.get(operator, operator), *(_formula(x) for x in args))
    return ("not", literal) if operator in _NEGATED else literal


def _clause(clause: Mapping[str, Any], roles: Mapping[str, str], *, alternatives: bool) -> Any:
    subject = roles[clause["subject_role"]]
    destination = clause.get("object_role")
    destinations = clause.get("destination_equivalence_roles", [destination]) if alternatives else [destination]
    if not destinations:
        destinations = [None]
    leaves = []
    for role in destinations:
        leaf = [clause["predicate"], subject]
        if role is not None:
            leaf.append(roles[role])
        leaves.append(leaf if clause.get("desired", True) else ["not", leaf])
    return leaves[0] if len(leaves) == 1 else ["or", *leaves]


def _role_formula(value, roles):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('invalid complete role goal formula')
    operator, *args = value
    if operator in {'and', 'or', 'not'}:
        if not args or (operator == 'not' and len(args) != 1):
            raise ValueError('invalid complete goal connective')
        return [operator, *(_role_formula(child, roles) for child in args)]
    if not args or any(not isinstance(role, str) or role not in roles for role in args):
        raise ValueError('unbound role in complete goal formula')
    return [operator, *(roles[role] for role in args)]


def _read_ref(ref: Mapping[str, Any], owner: Path, cache: dict[Path, bytes]) -> tuple[Path, bytes]:
    path = Path(ref["path"])
    if not path.is_absolute():
        path = owner.parent / path
    path = path.resolve()
    if path not in cache:
        cache[path] = path.read_bytes()
    data = cache[path]
    if hashlib.sha256(data).hexdigest() != ref.get("sha256"):
        raise ValueError(f"task/runtime dependency hash mismatch: {path}")
    return path, data


def _validate_task(spec: Mapping[str, Any], record: Mapping[str, Any], contract: Mapping[str, Any],
                   template: Mapping[str, Any], tro: Mapping[str, Any], bddl: str) -> None:
    task_id = spec["task_spec_id"]
    def check(actual: Any, expected: Any, label: str) -> None:
        _equal(actual, expected, f"{task_id}: {label}")

    for key in ("task_spec_id", "candidate_id", "scene_model", "rooms", "public_instruction", "source_family_id"):
        check(record.get(key), spec.get(key), key)
    check(record.get("activity_name"), task_id, "activity_name")
    for key in ("candidate_id", "scene_model", "rooms"):
        check(contract.get(key), spec.get(key), "contract." + key)
    profile = record["profile"]
    for key in ("task_id", "scene_model", "split", "instance_id"):
        check(profile.get(key), record.get(key), "profile." + key)
    check(profile.get("task_name"), task_id, "profile.task_name")
    check(profile.get("task_language"), spec["public_instruction"], "profile.task_language")

    private = spec["private_runtime"]
    roles, bindings = contract["role_to_scope"], contract["scope_bindings"]
    check(roles, private["role_to_scope"], "role_to_scope")
    check(bindings, private["scope_bindings"], "scope_bindings")
    names = {scope: binding["actual_name"] for scope, binding in bindings.items()}
    check(contract["metadata_task"]["inst_to_name"], names, "contract.instance_names")
    check(template["metadata"]["task"]["inst_to_name"], names, "template.instance_names")
    check(template["init_info"]["args"]["scene_model"], spec["scene_model"], "template.scene")
    for role, intended in spec["role_bindings"].items():
        bound = bindings[roles[role]]
        if role not in bound["roles"]:
            raise ValueError(f"task/runtime correspondence mismatch: {task_id}: role {role}")
        for key in ("category", "model", "synset", "actual_name"):
            if key in intended:
                check(bound.get(key), intended[key], f"role.{role}.{key}")
        if len(spec['rooms']) > 1:
            intended_rooms = intended.get('in_rooms')
            if not isinstance(intended_rooms,list) or not intended_rooms or not set(intended_rooms)<=set(spec['rooms']):
                raise ValueError('multi-room TaskSpec role membership missing')
            if not set(intended_rooms)<=set(bound.get('in_rooms', [])):
                raise ValueError('multi-room scope lost role membership')
    for scope, binding in bindings.items():
        if binding["binding_kind"] == "RUNTIME_ROBOT":
            continue
        args = template["objects_info"]["init_info"][binding["actual_name"]]["args"]
        if len(spec['rooms']) > 1:
            if not set(binding['in_rooms'])<=set(args.get('in_rooms',[])):
                raise ValueError('template object room differs from task room binding')
        for key in ("category", "model"):
            check(args.get(key), binding[key], f"template.{scope}.{key}")
        if scope not in tro:
            raise ValueError(f"task/runtime correspondence mismatch: {task_id}: missing TRO scope {scope}")
        check(tro[scope], template["state"]["registry"]["object_registry"][binding["actual_name"]],
              f"tro.{scope}.template_state")

    construction = spec["construction_contract"]
    for key in _CONSTRUCTION_FIELDS:
        check(contract.get(key), construction.get(key), "construction." + key)
    check(record["construction"]["initial_relation_recipe"], construction["initial_state_recipe"], "catalog.recipe")
    for key in ("pair_base_task_spec_id", "runtime_validation_pending"):
        check(record["construction"].get(key), construction.get(key), "catalog." + key)
    check(contract["metadata_task"].get("robot_start_recipe"), construction.get("robot_start_recipe"), "robot_start_recipe")
    check(record["construction"].get("robot_start"), private.get("robot_start"), "robot_start")
    start = private["robot_start"]
    if start.get('preparation_room') is not None:
        anchors = [b for b in bindings.values() if b.get('actual_name') == start.get('anchor_name')]
        if (len(anchors) != 1 or start['preparation_room'] not in anchors[0].get('in_rooms', [])
                or start['preparation_room'] not in spec['rooms']
                or start.get('room_policy') != 'ANCHOR_ROOM_AND_CONNECTED_FREE_APPROACH_V1'):
            raise ValueError('robot preparation room/anchor contract mismatch')
    expected_start = {"R1Pro": [{"position": start["position"], "orientation": start["orientation"]}]}
    check(template["metadata"]["task"]["robot_poses"], expected_start, "template.robot_poses")
    check(tro["robot_poses"], expected_start, "tro.robot_poses")
    authored = [binding["actual_name"] for binding in bindings.values() if binding["binding_kind"] == "AUTHORED_OBJECT"]
    check(sorted(record["construction"]["added_runtime_names"]), sorted(authored), "added_runtime_names")
    additions = contract.get("added_object_construction", [])
    check(sorted(item["actual_name"] for item in additions), sorted(authored), "added_object_construction")
    for item in additions:
        scope = item["scope_name"]
        for key, value in bindings[scope].items():
            check(item.get(key), value, f"addition.{scope}.{key}")
        check(item.get("placement_method"), construction.get("placement_method"), f"addition.{scope}.placement_method")

    goal = spec["goal_contract"]
    if any('physical_template' in item for item in (spec,record,contract)):
        check(record.get('physical_template'),spec.get('physical_template'),'physical template record')
        check(contract.get('physical_template'),spec.get('physical_template'),'physical template contract')
        from .physical_template_runtime import validate_template
        validate_template(spec['physical_template'],roles)
    check(goal["verifier_mode"], "SHARED_CANONICAL_VERIFIER", "verifier mode")
    check(private["verifier"], goal["verifier_mode"], "private verifier")
    check(record["official_verifier"], "OfficialPredicateVerifier", "official verifier")
    check(record["oracle_route"], "GROUNDED_GOAL_GRAPH", "oracle route")
    expected = [_clause(c, roles, alternatives=True) for c in goal["goal_clauses"]]
    witness = [_clause({**c, "destination_equivalence_roles": c.get("destination_equivalence_roles", [c.get("object_role")])[:1]},
                       roles, alternatives=True) for c in goal["goal_clauses"]]
    complete_goal = spec.get('task_contract', {}).get('full_goal_acceptance', False)
    positive_projection = spec.get('task_contract', {}).get('positive_goal_projection', False)
    complete_formula = goal.get('goal_formula')
    if complete_goal:
        from .bounded_full_goal_acceptance import bounded_dnf, is_binary_negative
        check(contract.get('goal_formula'), complete_formula, 'complete goal formula contract')
        expected_formula = _formula(_role_formula(complete_formula, roles))
        branches = bounded_dnf(expected_formula)
        witness_branches = bounded_dnf(_formula(['and', *witness]))
        if positive_projection:
            full_witness = goal.get('full_goal_witness')
            check(contract.get('full_goal_witness'), full_witness, 'complete goal witness contract')
            full_branches = bounded_dnf(_formula(_role_formula(full_witness, roles)))
            if len(full_branches) != 1 or full_branches[0] not in branches:
                raise ValueError('complete witness is not a legal goal branch')
            projected = frozenset(atom for atom in full_branches[0] if not is_binary_negative(atom))
            if not projected or projected == full_branches[0] or witness_branches != [projected]:
                raise ValueError('planning projection must preserve every executable witness literal')
        elif len(witness_branches) != 1 or witness_branches[0] not in branches:
            raise ValueError('planning witness is not a complete legal goal branch')
    else:
        if complete_formula is not None or contract.get('goal_formula') is not None:
            raise ValueError('complete goal formula requires explicit acceptance contract')
        expected_formula = _formula(['and', *expected])
    if not positive_projection and (goal.get('full_goal_witness') is not None or contract.get('full_goal_witness') is not None):
        raise ValueError('complete witness field requires positive projection contract')
    check(_formula(["and", *profile["canonical_goal_leaves"]]), _formula(["and", *witness]), "profile goals")
    check(profile["expected_goal_leaf_count"], len(witness), "profile goal count")
    private_goals = []
    for atom in private["canonical_goal_atoms"]:
        leaves = [[atom["predicate"], atom["subject_scope"], *([] if d is None else [d])]
                  for d in (atom["destination_scopes"] or [None])]
        leaves = [leaf if atom["desired"] else ["not", leaf] for leaf in leaves]
        private_goals.append(leaves[0] if len(leaves) == 1 else ["or", *leaves])
    check(_formula(["and", *private_goals]), _formula(["and", *(witness if complete_goal else expected)]), "private goals")
    expressions = parse_bddl(bddl)
    if len(expressions) != 1 or expressions[0][0] != "define":
        raise ValueError(f"invalid BDDL contract: {task_id}")
    declarations = find_section(expressions[0], ":objects")[1:]
    declared, pending = {}, []
    iterator = iter(declarations)
    for token in iterator:
        if token == "-":
            synset = next(iterator)
            for scope in pending:
                if scope in declared:
                    raise ValueError(f"duplicate BDDL scope: {task_id}: {scope}")
                declared[scope] = synset
            pending = []
        else:
            pending.append(token)
    if pending:
        raise ValueError(f"untyped BDDL scope: {task_id}")
    check(declared, {scope: binding["synset"] for scope, binding in bindings.items()}, "BDDL objects")
    section = find_section(expressions[0], ":goal")
    if len(section) != 2:
        raise ValueError(f"invalid BDDL goal section: {task_id}")
    check(_formula(section[1]), expected_formula, "BDDL goal")
    initial = [_clause(c, roles, alternatives=False) for c in construction["initial_state_recipe"]]
    placed = {roles[c["subject_role"]] for c in construction["initial_state_recipe"]
              if c["predicate"] in {"inside", "ontop", "under"}}
    floor = roles["_floor_anchor"]
    for scope, binding in sorted(bindings.items()):
        if binding["binding_kind"] == "RUNTIME_ROBOT" or scope in placed:
            continue
        if len(spec['rooms']) > 1:
            member_rooms = binding.get('in_rooms')
            if not isinstance(member_rooms,list) or not member_rooms or not set(member_rooms)<=set(spec['rooms']):
                raise ValueError('unbound multi-room object membership')
            if 'sceneObject' in binding['canonical_properties']:
                initial.extend(['inroom',scope,r] for r in sorted({r.rsplit('_',1)[0] for r in member_rooms}))
            else:
                room_floors = [s for s,b in bindings.items() if b['category']=='floors' and member_rooms[0] in b['in_rooms']]
                if not room_floors: raise ValueError('missing declared room floor')
                initial.append(['ontop',scope,room_floors[0]])
        else:
            initial.append(["inroom", scope, spec["rooms"][0].rsplit("_", 1)[0]]
                           if "sceneObject" in binding["canonical_properties"] else ["ontop", scope, floor])
    robot = next(scope for scope, b in bindings.items() if b["binding_kind"] == "RUNTIME_ROBOT")
    initial.append(["ontop", robot, floor])
    check(_formula(["and", *find_section(expressions[0], ":init")[1:]]), _formula(["and", *initial]), "BDDL init")


def validate_package_correspondence(package_path: Path, task_spec_rows: Sequence[Mapping[str, Any]],
                                    catalog_path: Path, task_names: Sequence[str] = ()) -> None:
    """Check selected compiled contracts, including pure-visual Base dependencies.

    Legacy TaskSpec schemas retain their existing loader contract. A perception
    record may not opt out by removing its schema: presence of its construction
    and private-runtime fields also selects this validator.
    """
    catalog_path = Path(catalog_path).resolve()
    catalog = json.loads(catalog_path.read_text())
    records = catalog["tasks"]
    by_id = {row["task_spec_id"]: row for row in records}
    specs = {row["task_spec_id"]: row for row in task_spec_rows}
    if len(by_id) != len(records) or set(by_id) != set(specs):
        raise ValueError("task/runtime correspondence mismatch: catalog membership")
    selected = set(task_names) if task_names else set(specs)
    if not selected <= set(specs):
        raise ValueError("task/runtime correspondence mismatch: unknown selected member")
    for task_id in tuple(selected):
        base = specs[task_id].get("visual_variant", {}).get("base_task_spec_id")
        if base:
            if base not in specs:
                raise ValueError(f"task/runtime correspondence mismatch: absent Base {base}")
            selected.add(base)
    cache: dict[Path, bytes] = {}
    package_path = Path(package_path).resolve()
    package = json.loads(package_path.read_text())
    pair_ref = package.get("visual_pairs", {}).get("bindings")
    pairs = {}
    if pair_ref:
        _, data = _read_ref(pair_ref, package_path, cache)
        for line in data.decode().splitlines():
            if line.strip():
                pair = json.loads(line)
                pair_id = pair["pair_id"]
                if pair_id in pairs:
                    raise ValueError("duplicate package visual pair identity")
                pairs[pair_id] = pair
    for task_id in sorted(selected):
        spec, record = specs[task_id], by_id[task_id]
        if spec.get('benchmark_level') == 'L1' and spec.get('schema_version') not in {TASK_SCHEMA, L1_BRANCH_SCHEMA, L1_PROCESS_SCHEMA}:
            raise ValueError('unsupported L1 TaskSpec correspondence schema')
        if spec.get("schema_version") != TASK_SCHEMA and not ({"construction_contract", "private_runtime"} & set(spec)):
            continue
        try:
            documents = {}
            for key in ("template_contract", "template", "tro_state", "bddl"):
                _, data = _read_ref(record[key], catalog_path, cache)
                documents[key] = data.decode() if key == "bddl" else json.loads(data)
            _validate_task(spec, record, documents["template_contract"], documents["template"], documents["tro_state"], documents["bddl"])
            if spec.get('schema_version') in {L1_BRANCH_SCHEMA, L1_PROCESS_SCHEMA}:
                from .l1_initial_branch import validate_initial_branch, CAPABILITY, DECLARED_FACTS_CAPABILITY, uses_declared_initial_facts
                if spec.get('benchmark_level') != 'L1' or CAPABILITY not in catalog.get('required_capabilities', []):
                    raise ValueError('L1 branch capability/level missing')
                task_contract = spec['task_contract']
                validate_initial_branch(task_contract, documents['template_contract']['role_to_scope'])
                if any(r.get('placement_policy') for r in spec['construction_contract']['initial_state_recipe']):
                    from .initial_layout_search import validate_layout_contract
                    validate_layout_contract(documents['template_contract'],spec['goal_contract']['goal_clauses'])
                    if 'BOUNDED_INITIAL_LAYOUT_V1' not in catalog.get('required_capabilities',[]):
                        raise ValueError('initial layout capability missing')
                    from .initial_layout_search import validate_policy
                    for recipe in spec['construction_contract']['initial_state_recipe']:
                        validate_policy(recipe,documents['template_contract']['role_to_scope'])
                if task_contract.get('initial_contents_assertions') and 'L1_INITIAL_CONTENTS_ASSERTIONS_V1' not in catalog.get('required_capabilities', []):
                    raise ValueError('L1 initial contents assertions capability missing')
                if task_contract.get('process_requirements', {}).get('history_program'):
                    if 'L1_HISTORY_BINDINGS_V1' not in catalog.get('required_capabilities', []):
                        raise ValueError('L1 history bindings capability missing')
                    history = task_contract['process_requirements']['history_program']
                    from .l1_history_bindings import validate_history_program
                    validate_history_program(history, documents['template_contract']['role_to_scope'])
                    roles = documents['template_contract']['role_to_scope']
                    if task_contract.get('full_goal_acceptance'):
                        from .l1_dynamic_process_goals import validate_dynamic_goal, CAPABILITY as DYNAMIC_CAPABILITY
                        if DYNAMIC_CAPABILITY not in catalog.get('required_capabilities', []):raise ValueError('dynamic process capability missing')
                        validate_dynamic_goal(task_contract['process_requirements'], spec['goal_contract'].get('goal_formula'), roles)
                    else:
                        _equal(_formula(['and', *[_clause(g,roles,alternatives=False) for g in history['clauses']]]),
                               _formula(['and', *[_clause(g,roles,alternatives=False) for g in spec['goal_contract']['goal_clauses']]]),
                               'history static final goal correspondence')
                if task_contract.get('process_requirements', {}).get('process_program'):
                    from .l1_am_process_programs import CAPABILITY as MILESTONE_CAPABILITY, validate_program_final
                    if MILESTONE_CAPABILITY not in catalog.get('required_capabilities', []):
                        raise ValueError('L1 milestone capability missing')
                    if 'cycle_order_final' in task_contract['process_requirements']['process_program']:
                        from .l1_dynamic_process_goals import validate_dynamic_goal, CAPABILITY as DYNAMIC_CAPABILITY
                        if not task_contract.get('full_goal_acceptance') or DYNAMIC_CAPABILITY not in catalog.get('required_capabilities', []):raise ValueError('dynamic process full-goal capability missing')
                        validate_dynamic_goal(task_contract['process_requirements'], spec['goal_contract'].get('goal_formula'), documents['template_contract']['role_to_scope'])
                    else:
                        validate_program_final(task_contract['process_requirements']['process_program'], spec['goal_contract']['goal_clauses'])
                if task_contract.get('full_goal_acceptance') and 'L1_FULL_GOAL_ACCEPTANCE_V1' not in catalog.get('required_capabilities', []):
                    raise ValueError('L1 complete goal acceptance capability missing')
                if task_contract.get('positive_goal_projection') and 'L1_POSITIVE_GOAL_PROJECTION_V1' not in catalog.get('required_capabilities', []):
                    raise ValueError('L1 positive goal projection capability missing')
                if uses_declared_initial_facts(task_contract) and DECLARED_FACTS_CAPABILITY not in catalog.get('required_capabilities', []):
                    raise ValueError('L1 declared initial facts capability missing')
                _equal(record.get('task_contract'), task_contract, task_id+': catalog L1 contract')
                _equal(documents['template_contract'].get('task_contract'), task_contract, task_id+': template L1 contract')
                recipes = spec['construction_contract']['initial_state_recipe']
                if any(a.get('predicate') in {'under','nextto','next_to'} for a in recipes) and 'RELATIONAL_INITIAL_PLACEMENT_V1' not in catalog.get('required_capabilities', []):
                    raise ValueError('relational initial placement capability missing')
                if task_contract.get('process_requirements', {}).get('current_reference_contracts'):
                    if 'L1_CURRENT_REFERENCE_V1' not in catalog.get('required_capabilities', []):
                        raise ValueError('L1 current reference capability missing')
                    from .l1_process_audit import validate_current_reference_formula
                    validate_current_reference_formula(task_contract,spec['goal_contract'].get('goal_formula'))
                for assertion in task_contract['initial_assertions']:
                    if not any(all(recipe.get(k) == v for k,v in assertion.items()) for recipe in recipes):
                        raise ValueError('initial branch assertion absent from construction')
                if spec['schema_version'] == L1_PROCESS_SCHEMA:
                    if task_contract['schema_version'] != 'l1-initial-branch/v2' or 'L1_PUBLIC_PROCESS_V1' not in catalog.get('required_capabilities',[]):
                        raise ValueError('missing public process contract capability')
                    if task_contract['process_requirements'].get('state_protection_roles') and 'L1_STATE_PROTECTION_V1' not in catalog.get('required_capabilities', []):
                        raise ValueError('L1 state protection capability missing')
                    for rule in task_contract['initial_reference_goals']:
                        initial=task_contract['initial_assertions'][rule['initial_assertion_index']]
                        if not any(g.get('predicate') == rule['goal_predicate'] and g.get('subject_role') == rule['goal_subject_role']
                                   and g.get('object_role') == initial.get('object_role') and g.get('desired') is True
                                   for g in spec['goal_contract']['goal_clauses']):
                            raise ValueError('initial-reference final goal drift')
                elif task_contract['axis'] == 'M':
                    for assertion in task_contract['initial_assertions']:
                        if not any(all(goal.get(k) == v for k,v in assertion.items()) for goal in spec['goal_contract']['goal_clauses']):
                            raise ValueError('M initial state restoration goal drift')
                elif not uses_declared_initial_facts(task_contract):
                    region = task_contract['inspection_region']
                    if not any(recipe.get('predicate') == 'ontop' and recipe.get('subject_role') == region['subject_role']
                               and recipe.get('object_role') == region['support_role']
                               and recipe.get('placement_rank') == region['rank']
                               and recipe.get('placement_count') == region['count'] for recipe in recipes):
                        raise ValueError('A inspection region and construction preference drift')
            variant = spec.get("visual_variant")
            if variant:
                base_id = variant["base_task_spec_id"]
                _equal(spec["construction_contract"]["pair_base_task_spec_id"], base_id, task_id + ": variant Base")
                if pair_ref:
                    pair = pairs[variant["pair_id"]]
                    _equal(pair["variant_task_spec_id"], task_id, task_id + ": pair member")
                    _equal(pair["base_task_spec_id"], base_id, task_id + ": pair Base")
                    _equal(pair["construction_dependency"]["task_spec_id"], base_id, task_id + ": pair dependency")
                    for key, expected in (("inherit_canonical_goal", True), ("inherit_object_state", True),
                                          ("inherit_robot_pose", True), ("reconstruct_or_reselect_pose", False)):
                        _equal(pair["construction_dependency"].get(key), expected, task_id + ": " + key)
                base = specs[base_id]
                for key in ("scene_model", "rooms", "role_bindings", "goal_contract", "public_instruction"):
                    _equal(spec[key], base[key], task_id + ": pair." + key)
                for key in _CONSTRUCTION_FIELDS:
                    if key not in {"visual_rules", "runtime_validation_pending"}:
                        _equal(spec["construction_contract"].get(key), base["construction_contract"].get(key), task_id + ": pair." + key)
                _equal(spec["private_runtime"]["robot_start"], base["private_runtime"]["robot_start"], task_id + ": pair.robot_start")
        except (KeyError, TypeError, IndexError, StopIteration) as exc:
            raise ValueError(f"incomplete task/runtime correspondence contract: {task_id}: {exc}") from exc
