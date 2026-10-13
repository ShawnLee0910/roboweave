"""Bounded private reset facts for declared A/M initial branches.

This reads existing canonical predicates; it does not certify visual information
insufficiency, add history scoring, or publish private branch answers.
"""
from collections.abc import Mapping
import math

import numpy as np

CAPABILITY = "L1_INITIAL_BRANCH_V1"
DECLARED_FACTS_CAPABILITY = "L1_DECLARED_INITIAL_FACTS_V1"
CONTENTS_CAPABILITY = "L1_INITIAL_CONTENTS_ASSERTIONS_V1"
SCHEMA = "l1-initial-branch/v1"
PROCESS_SCHEMA = 'l1-initial-branch/v2'
# These are final-reference predicates, not additional construction operators.
INITIAL_REFERENCE_GOAL_PREDICATES = frozenset({'inside', 'ontop', 'under', 'next_to', 'nextto'})


def uses_declared_initial_facts(contract):
    return (contract.get('schema_version') == SCHEMA and contract.get('axis') == 'A'
            and contract.get('inspection_region') is None and contract.get('observation') is None)


class InitialObservationMutationError(RuntimeError):
    session_invalidated = True


def validate_initial_branch(contract, roles):
    if not isinstance(contract, Mapping) or contract.get("schema_version") not in {SCHEMA, PROCESS_SCHEMA}:
        raise ValueError("unsupported L1 initial branch contract")
    allowed = {"schema_version", "axis", "proposal_id", "world_branch_id", "pair_member",
               "initial_assertions", "inspection_region", "observation", "full_goal_acceptance", "positive_goal_projection",
               "initial_contents_assertions"}
    from .construction_contents_readback import validate_contents_assertions
    validate_contents_assertions(contract.get('initial_contents_assertions', []), roles)
    if type(contract.get('full_goal_acceptance', False)) is not bool:
        raise ValueError('full goal acceptance must be boolean')
    if type(contract.get('positive_goal_projection', False)) is not bool:
        raise ValueError('positive goal projection must be boolean')
    if contract.get('positive_goal_projection') and not contract.get('full_goal_acceptance'):
        raise ValueError('positive goal projection requires complete goal acceptance')
    if contract['schema_version'] == PROCESS_SCHEMA:
        allowed |= {'process_requirements','initial_reference_goals'}
        from .l1_process_audit import validate_process_requirements, protected_state_assertions
        validate_process_requirements(contract.get('process_requirements'),roles)
        program = contract['process_requirements'].get('process_program', {})
        if (len(program.get('milestones', [])) > 1 or any(k in program for k in ('last_cycle_final', 'global_unary_phases', 'exact_cycles_per_role', 'cycle_order_final'))) and contract.get('axis') != 'M':
            raise ValueError('M transition-ledger and last-cycle programs require M axis')
        history = contract['process_requirements'].get('history_program')
        if history and contract.get('axis') != 'M':
            raise ValueError('history programs require M axis')
        if (history or program) and contract.get('full_goal_acceptance'):
            from .l1_dynamic_process_goals import dynamic_goal_formula
            if contract.get('positive_goal_projection') or dynamic_goal_formula(history=history, process=program) is None:
                raise ValueError('only finite dynamic process goals support full-goal composition')
        if contract.get('full_goal_acceptance') and contract['process_requirements']['before_placement']:
            raise ValueError('complete goal acceptance does not support before-placement references')
        protected_state_assertions(contract)
        references=contract.get('initial_reference_goals')
        if not isinstance(references,list):
            raise ValueError('missing initial reference goals')
        for rule in references:
            if (not isinstance(rule, Mapping)
                or set(rule) != {'initial_assertion_index','goal_subject_role','goal_predicate'}
                or type(rule['initial_assertion_index']) is not int
                or not 0 <= rule['initial_assertion_index'] < len(contract.get('initial_assertions',[]))
                or rule['goal_subject_role'] not in roles
                or rule['goal_predicate'] not in INITIAL_REFERENCE_GOAL_PREDICATES):
                raise ValueError('invalid initial reference goal')
            initial = contract['initial_assertions'][rule['initial_assertion_index']]
            if (not isinstance(initial, Mapping) or initial.get('predicate') not in {'inside', 'ontop', 'under', 'next_to', 'nextto'}
                    or initial.get('object_role') not in roles):
                raise ValueError('initial reference goal requires a bound binary initial assertion')
        for requirement in contract['process_requirements']['before_placement']:
            matches = [rule for rule in references if rule['goal_subject_role'] == requirement['subject_role']]
            if len(matches) != 1:
                raise ValueError('process placement must resolve one declared final reference')
    if set(contract) - allowed:
        raise ValueError("unknown L1 initial branch contract fields")
    if contract.get("axis") not in {"A", "M"} or contract.get("pair_member") not in {"control", "challenge"}:
        raise ValueError("invalid L1 axis or pair member")
    if any(not isinstance(contract.get(k), str) or not contract[k] for k in ("proposal_id", "world_branch_id")):
        raise ValueError("missing L1 branch identity")
    assertions = contract.get("initial_assertions")
    if not isinstance(assertions, list) or not assertions:
        raise ValueError("L1 initial assertions are required")
    for row in assertions:
        if row.get('placement_policy'):
            from .initial_layout_search import validate_policy
            validate_policy(row,roles)
        if (not isinstance(row, Mapping) or set(row) - {"predicate", "subject_role", "object_role", "desired", "placement_policy"}
                or row.get("desired") is not True or row.get("subject_role") not in roles
                or row.get("predicate") not in {"ontop", "inside", "under", "next_to", "nextto", "open", "closed", "toggled_on", "toggled_off"}):
            raise ValueError("unsupported L1 initial assertion")
        if row["predicate"] in {"ontop", "inside", "under", "next_to", "nextto"} and row.get("object_role") not in roles:
            raise ValueError("unbound L1 assertion reference")
    region = contract.get("inspection_region")
    observation = contract.get("observation")
    if contract['schema_version'] == PROCESS_SCHEMA:
        if region is not None or observation is not None:
            raise ValueError('process branch uses content/initial-reference evidence, not camera overrides')
        return
    if contract["axis"] == "M":
        if region is not None or observation is not None:
            raise ValueError("M initial branch does not support observation overrides")
        return
    if uses_declared_initial_facts(contract):
        return
    if (not isinstance(region, Mapping) or set(region) != {"subject_role", "support_role", "rank", "count"}
            or region.get("subject_role") not in roles or region.get("support_role") not in roles
            or type(region.get("rank")) is not int or region["rank"] not in (0, 1)
            or type(region.get("count")) is not int or region["count"] != 2):
        raise ValueError("A requires a two-region inspection declaration")
    if (not isinstance(observation, Mapping) or set(observation) != {"yaw_offset_deg", "condition", "status"}
            or type(observation.get("yaw_offset_deg")) not in (int, float)
            or not math.isfinite(observation["yaw_offset_deg"])
            or observation.get("condition") not in {"TARGET_VISIBLE", "INITIAL_INFORMATION_INSUFFICIENT"}
            or observation.get("status") != "PENDING_REVIEW"):
        raise ValueError("invalid A observation declaration")


def materialize_initial_branch(contract, *, roles, role_object, recipe_truth,
                               robot, object_aabb, screen_axes, configuration_id):
    """Called once after construction hold restoration; returns private JSON facts."""
    validate_initial_branch(contract, roles)
    facts = [{"assertion": dict(row), "satisfied": bool(recipe_truth(row, role_object))}
             for row in contract["initial_assertions"]]
    if not all(row["satisfied"] for row in facts):
        raise ValueError("L1 declared initial branch does not match canonical readback")
    report = {"schema_version": SCHEMA, "configuration_id": configuration_id,
              "proposal_id": contract["proposal_id"], "world_branch_id": contract["world_branch_id"],
              "pair_member": contract["pair_member"], "initial_facts": facts,
              "facts_scope": "private construction readback; no dynamic history or axis certification",
              "axis_review_status": "PENDING_REVIEW"}
    if contract["axis"] == "A" and contract.get('inspection_region') is not None:
        region = contract["inspection_region"]
        position, quaternion = robot.get_position_orientation()
        def array(value):
            return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=float)
        pos, quat = array(position), array(quaternion)
        if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) == 0:
            raise ValueError("invalid robot pose for A observation")
        lo, hi = object_aabb(role_object(region["support_role"]))
        center = (array(lo)[:2] + array(hi)[:2]) / 2
        right, _ = screen_axes(observer_xy=pos[:2], target_center_xy=center)
        slo, shi = object_aabb(role_object(region["subject_role"]))
        coordinate = float(np.dot(right, (array(slo)[:2] + array(shi)[:2]) / 2 - center))
        if not math.isfinite(coordinate) or (coordinate >= 0 if region["rank"] == 0 else coordinate <= 0):
            raise ValueError("A subject is not in its declared inspection region")
        angle = math.radians(contract["observation"]["yaw_offset_deg"]) / 2
        s, c = math.sin(angle), math.cos(angle)
        x, y, z, w = quat
        rotated = np.array([c*x-s*y, s*x+c*y, c*z+s*w, c*w-s*z])
        if hasattr(quaternion, "new_tensor"):
            rotated = quaternion.new_tensor(rotated)
        try:
            robot.set_position_orientation(position=position, orientation=rotated)
        except Exception as exc:
            raise InitialObservationMutationError("A initial observation pose mutation failed") from exc
        report["inspection_region"] = {**dict(region), "screen_right_coordinate": coordinate}
        report["observation"] = {**dict(contract["observation"]), "applied_yaw_offset_deg": contract["observation"]["yaw_offset_deg"]}
    return report
