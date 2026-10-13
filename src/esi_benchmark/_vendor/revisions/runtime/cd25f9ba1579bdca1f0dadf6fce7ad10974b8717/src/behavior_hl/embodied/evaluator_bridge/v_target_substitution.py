"""Frozen single-target substitution; all other scene state stays unchanged."""
from copy import deepcopy
import numpy as np
from .on_top_support_layers import quaternion_to_matrix_xyzw
from .canonical_bddl_grounder import find_section


def substitute_target(template, tro, expression, substitution, source_metadata, target_metadata):
    template, tro, expression = deepcopy(template), deepcopy(tro), deepcopy(expression)
    old, new = substitution['source_scope'], substitution['target_scope']
    mapping = template['metadata']['task']['inst_to_name']
    old_name = mapping.pop(old)
    name = substitution['actual_name']
    if new in mapping or name in template['objects_info']['init_info']:
        raise ValueError('replacement target already exists')
    if ['ontop', old, substitution['support_scope']] not in find_section(expression, ':init'):
        raise ValueError('replacement requires a declared source support')
    info = template['objects_info']['init_info'].pop(old_name)
    old_scale = np.asarray(info['args']['scale'])
    state = template['state']['registry']['object_registry'].pop(old_name)
    def bottom_offset(metadata, scale, rotation):
        box = metadata['link_bounding_boxes']['base_link']['visual']['axis_aligned']
        center = np.asarray(box['transform'])[:3, 3] * scale
        half = np.asarray(box['extent']) * scale / 2
        return float((rotation @ center)[2] - (np.abs(rotation) @ half)[2])

    # Preserve the source's measured lower face rather than reusing the old
    # root Z with a differently sized object. Live OnTop is still required.
    def replace_state(source):
        result = deepcopy(source)
        rotation = quaternion_to_matrix_xyzw(source['root_link']['ori'])
        result['root_link']['pos'][2] += (bottom_offset(source_metadata, old_scale, rotation)
                                        - bottom_offset(target_metadata, np.ones(3), rotation))
        return result
    state = replace_state(state)
    info['args'].update(name=name, category=substitution['category'], model=substitution['model'],
                        scale=[1., 1., 1.], bddl_object_scope=new)
    mapping[new] = name
    template['objects_info']['init_info'][name] = info
    template['state']['registry']['object_registry'][name] = state
    # TRO overrides the template during reset and can describe a different
    # layout. Never replace its target state with the template's old pose.
    tro[new] = replace_state(tro.pop(old))
    # The remaining source-category members keep their original type.
    objects = find_section(expression, ':objects')
    objects.remove(old)
    objects.extend([new, '-', new.rsplit('_', 1)[0]])
    def rename(value):
        return [rename(v) for v in value] if isinstance(value, list) else new if value == old else value
    init = find_section(expression, ':init')
    init[:] = rename(init)
    return template, tro, expression
