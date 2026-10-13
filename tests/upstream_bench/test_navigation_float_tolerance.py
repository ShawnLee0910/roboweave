import copy
import json

import pytest

from agent_protocol import parse_native_chat_completion


def response(value, action='move_forward', field='move_distance_m'):
    return {'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
        'tool_calls':[{'id':'call_1','type':'function','function':{'name':'robot_action',
            'arguments':json.dumps({'action':action,field:value})}}]}}]}


@pytest.mark.parametrize('value,expected',[
    (0.14999999999999,0.15),(0.1499999999999,0.15),(0.1499999999999999,0.15),
    (0.59999999999999,0.6),(0.3000000005,0.3),(0.15,0.15),(0.3,0.3),(0.6,0.6),
    (0.1491,0.15),(0.3009,0.3),(0.5991,0.6),(0.6009,0.6),
    (0.149,0.15),(0.301,0.3),(0.601,0.6)])
def test_near_canonical_distance_is_normalized_without_mutating_response(value,expected):
    raw=response(value);saved=copy.deepcopy(raw)
    assert parse_native_chat_completion(raw).move_distance_m==expected
    assert raw==saved


@pytest.mark.parametrize('value',[True,False,'0.15',0,-0.15,0.2,1.0,
    0.148,0.152,0.298,0.302,0.598,0.602,
    float('nan'),float('inf'),-float('inf'),10**400])
def test_noncanonical_distances_still_fail(value):
    with pytest.raises(ValueError,match='invalid move distance'):
        parse_native_chat_completion(response(value))


def test_navigation_capability_gate_and_turn_rules_are_unchanged():
    with pytest.raises(ValueError,match='unavailable navigation step size'):
        parse_native_chat_completion(response(0.14999999999999),allow_navigation_step_sizes=False)
    with pytest.raises(ValueError,match='invalid turn angle'):
        parse_native_chat_completion(response(10.0000000001,'turn_left','turn_angle_deg'))
