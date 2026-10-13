from copy import deepcopy

import pytest

from behavior_hl.embodied.evaluator_bridge.l1_dynamic_process_goals import dynamic_goal_formula, validate_dynamic_goal
from test_l1_history_subject_and_maintain import subject_program, ROLES
from test_m20_unary_process import ledger


@pytest.mark.parametrize('kind', ['history', 'second', 'last'])
def test_physical_alternatives_cover_all_answers_and_reject_fixed_answer(kind):
    roles = {r: r for r in ROLES} if kind == 'history' else {r: r for r in ('a', 'b', 'c')}
    requirements = ({'history_program': subject_program()} if kind == 'history' else
                    {'process_program': ledger(list(roles), ordinal=2 if kind == 'second' else 'last')})
    formula = dynamic_goal_formula(history=requirements.get('history_program'), process=requirements.get('process_program'))
    assert formula[0] == 'or' and len(formula) == 4
    validate_dynamic_goal(requirements, formula, roles)
    with pytest.raises(ValueError, match='alternatives differ'):
        validate_dynamic_goal(requirements, formula[1], roles)
    tampered = deepcopy(formula)
    tampered.pop()
    with pytest.raises(ValueError, match='alternatives differ'):
        validate_dynamic_goal(requirements, tampered, roles)
