import json
import pytest
from agent_protocol import parse_native_tool_message


def parse(angle):
    return parse_native_tool_message({'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'turn-test', 'type': 'function', 'function': {'name': 'robot_action',
         'arguments': json.dumps({'action': 'turn_right', 'turn_angle_deg': angle, 'repeat': 3})}}]})


@pytest.mark.parametrize('angle', ['30', '30.0', '3e1', ' 10 ', 30, 10])
def test_numeric_turn(angle):
    result = parse(angle)
    assert result.turn_angle_deg == float(angle)
    assert result.public_repeat == 3


@pytest.mark.parametrize('angle', ['31', '30deg', 'NaN', 'Infinity', '1e999', 'true', 'null', '[]', '"30"', '', True, 31])
def test_invalid_turn_remains_rejected(angle):
    with pytest.raises(ValueError, match='invalid turn angle'):
        parse(angle)
