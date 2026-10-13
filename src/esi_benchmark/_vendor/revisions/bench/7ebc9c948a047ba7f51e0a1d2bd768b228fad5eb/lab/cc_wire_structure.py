"""Structural audit only: never persist prompts, images, headers or tool results."""
import hashlib
import json

EXPECTED_TOOLS = frozenset({
    'TaskCreate', 'TaskGet', 'TaskList', 'TaskUpdate',
    'mcp__action20__robot_action', 'mcp__action20__public_history_recall',
})
CORRECTION = ('Tool interface correction: that tool is not available. '
              'Available tools: ' + ', '.join(sorted(EXPECTED_TOOLS)) + '. '
              'Use mcp__action20__robot_action for robot actions. '
              'Do not use Bash, Read or ToolSearch. This is the only correction; '
              'the original task and budgets are unchanged.')


class ToolContractError(ValueError):
    pass


class CCToolGuard:
    """One correction turn per CC process; retain it on HTTP retries/history replay."""

    def __init__(self):
        self.corrected_ids = set()

    def apply(self, payload, *, count_tokens=False):
        tools = payload.get('tools', [])
        names = [t.get('name') for t in tools if isinstance(t, dict)]
        if (len(names) != len(EXPECTED_TOOLS) or set(names) != EXPECTED_TOOLS
                or any(t.get('defer_loading') is True
                       or not isinstance(t.get('input_schema'), dict)
                       or t['input_schema'].get('type') != 'object' for t in tools)):
            raise ToolContractError('cc_tool_inventory_mismatch')
        messages = payload.get('messages', [])
        calls = {}
        for message in messages:
            if message.get('role') == 'assistant' and isinstance(message.get('content'), list):
                for block in message['content']:
                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                        calls[block.get('id')] = block.get('name')
        # Only failed, unavailable calls in the newest user turn are eligible.
        # Never change a valid tool's error or a historical successful result.
        newest = messages[-1] if messages else {}
        blocks = newest.get('content', [])
        if not self.corrected_ids and not count_tokens and newest.get('role') == 'user' and isinstance(blocks, list):
            self.corrected_ids.update(
                b['tool_use_id'] for b in blocks if isinstance(b, dict)
                and b.get('type') == 'tool_result' and b.get('is_error') is True
                and isinstance(b.get('tool_use_id'), str)
                and b['tool_use_id'] in calls
                and calls[b['tool_use_id']] not in EXPECTED_TOOLS
            )
        changed = False
        for message in messages:
            if message.get('role') != 'user' or not isinstance(message.get('content'), list):
                continue
            for block in message['content']:
                if (not isinstance(block, dict) or block.get('type') != 'tool_result'
                        or block.get('is_error') is not True
                        or block.get('tool_use_id') not in self.corrected_ids):
                    continue
                content = block.get('content', '')
                if isinstance(content, str):
                    if CORRECTION not in content:
                        block['content'] = content + '\n' + CORRECTION
                        changed = True
                elif isinstance(content, list) and not any(
                    isinstance(b, dict) and b.get('text') == CORRECTION for b in content
                ):
                    content.append({'type': 'text', 'text': CORRECTION})
                    changed = True
        return changed


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def structure(payload):
    allowed = {'TaskCreate', 'TaskGet', 'TaskList', 'TaskUpdate', 'Bash', 'Read',
               'ToolSearch', 'mcp__action20__robot_action',
               'mcp__action20__public_history_recall'}
    def name(value):
        return value if isinstance(value, str) and value in allowed else 'other'
    return {
        'tools_sha256': digest(payload.get('tools')),
        'tools': [{'name': name(t.get('name')),
                   'schema_sha256': digest(t.get('input_schema')),
                   'defer_loading': t.get('defer_loading') is True}
                  for t in payload.get('tools', [])],
        'system_sha256': digest(payload.get('system')),
        'message_count': len(payload.get('messages', [])),
        'messages_sha256': digest(payload.get('messages')),
        'messages': [
            {'role': m.get('role') if m.get('role') in ('user', 'assistant') else 'other',
             'blocks': [{'type': b.get('type') if b.get('type') in
                         ('text', 'image', 'tool_use', 'tool_result', 'thinking',
                          'redacted_thinking', 'compaction') else 'other',
                         'tool': name(b.get('name')),
                         'is_error': b.get('is_error') is True,
                         'content_sha256': digest(b)}
                        for b in m.get('content', []) if isinstance(b, dict)]}
            for m in payload.get('messages', [])[-2:]],
    }


if __name__ == '__main__':
    sample = {'tools': [{'name': 'mcp__action20__robot_action',
                         'input_schema': {'type': 'object'}}],
              'messages': [{'role': 'user', 'content': [{'type': 'text',
                             'text': 'PRIVATE_PROMPT'}]}]}
    before = json.dumps(sample)
    result = structure(sample)
    assert 'PRIVATE_PROMPT' not in json.dumps(result)
    assert json.dumps(sample) == before
    assert result['tools'][0]['name'] == 'mcp__action20__robot_action'
    assert result['tools'][0]['defer_loading'] is False
    assert structure({'messages': [{'content': 'PRIVATE_PROMPT'}]})['messages'][0]['blocks'] == []
    print('PASS: metadata-only and no input mutation')
