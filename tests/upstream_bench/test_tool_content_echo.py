import copy
import json

import pytest

from agent_protocol import isolate_native_reasoning, parse_native_tool_message


def message(content):
    return {'role': 'assistant', 'content': content, 'reasoning_content': 'Choose a turn.',
            'tool_calls': [{'id': 'call_test', 'type': 'function', 'function': {
                'name': 'robot_action',
                'arguments': json.dumps({'action': 'turn_right', 'turn_angle_deg': 30})}}]}


ECHO = ('<function=robot_action>\n<parameter=action>move_forward</parameter>\n'
        '<parameter=move_distance_m>0.14999999999999999</parameter>\n'
        '</function>\n</tool_call>\n')


@pytest.mark.parametrize('content', [ECHO, ECHO * 17, '<tool_call>' + ECHO, '\n</think>\n',
    '</think>\n' + ECHO + '</think>\n' + ECHO + '</think>'])
def test_only_native_call_executes_and_raw_message_is_preserved(content):
    raw = message(content)
    before = copy.deepcopy(raw)
    decision = parse_native_tool_message(raw)
    assert decision.action == 'turn_right'
    assert decision.turn_angle_deg == 30
    assert isolate_native_reasoning(raw) == ('Choose a turn.', 'reasoning_content', None)
    assert raw == before


@pytest.mark.parametrize('content', ['do this ' + ECHO, ECHO + 'then done',
    ECHO.replace('robot_action', 'other_tool'), '<think>unfinished', '<function=robot_action>'])
def test_unknown_or_malformed_text_stays_rejected(content):
    with pytest.raises(ValueError):
        parse_native_tool_message(message(content))


@pytest.mark.parametrize('variant', ['none', 'multiple', 'bad_json', 'bad_angle', 'bad_action'])
def test_echo_never_bypasses_native_call_validation(variant):
    raw = message(ECHO)
    if variant == 'none':
        raw['tool_calls'] = []
    elif variant == 'multiple':
        raw['tool_calls'] *= 2
    else:
        raw['tool_calls'][0]['function']['arguments'] = {
            'bad_json': '{broken',
            'bad_angle': '{"action":"turn_right","turn_angle_deg":999}',
            'bad_action': '{"action":"teleport"}',
        }[variant]
    with pytest.raises(ValueError):
        parse_native_tool_message(raw)
