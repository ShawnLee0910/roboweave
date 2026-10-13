"""Read-only empty-container gate shared by construction and frozen reset."""
from collections.abc import Mapping

import numpy as np

from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    canonical_relation, object_aabb, state_by_name,
)


def read_empty_contents(scene, roles, role_object):
    if not isinstance(roles, list) or any(not isinstance(r, str) or not r for r in roles):
        raise ValueError('empty contents roles must be a list of role names')
    if len(set(roles)) != len(roles):
        raise ValueError('duplicate empty contents role')
    if not roles:
        return []
    objects = scene.objects.values() if isinstance(scene.objects, Mapping) else scene.objects
    objects = tuple(objects)
    systems = scene.active_systems
    if not isinstance(systems, Mapping):
        raise ValueError('active particle system inventory unavailable')
    reports = []
    for role in roles:
        container = role_object(role)
        if sum(obj is container for obj in objects) != 1:
            raise RuntimeError('empty container is not uniquely present in scene inventory')
        contains = state_by_name(container, 'Contains')
        if not callable(getattr(contains, 'get_value', None)):
            raise RuntimeError('canonical Contains unavailable for empty contents role: ' + role)
        lo, hi = object_aabb(container)
        if not np.isfinite([lo, hi]).all() or np.any(lo > hi):
            raise RuntimeError('invalid container AABB')
        inside, checked = [], 0
        for obj in objects:
            if obj is container:
                continue
            other_lo, other_hi = object_aabb(obj)
            if not np.isfinite([other_lo, other_hi]).all() or np.any(other_lo > other_hi):
                raise RuntimeError('invalid scene object AABB during contents readback')
            # Broad phase only; canonical Inside owns containment semantics.
            if np.any(other_hi < lo) or np.any(other_lo > hi):
                continue
            checked += 1
            if canonical_relation(obj, 'Inside', container):
                inside.append(obj.name)
        particles = []
        for name, system in sorted(systems.items()):
            value = contains.get_value(system)
            if value is None:
                raise RuntimeError('canonical Contains returned unknown for system: ' + name)
            if bool(value):
                particles.append(name)
        reports.append(dict(role=role, object_count_scanned=len(objects) - 1,
                            overlapping_objects_checked=checked, inside_objects=sorted(inside),
                            particle_systems_checked=sorted(systems), nonempty_particle_systems=particles,
                            passed=not inside and not particles,
                            mesh_embedded_contents_visual_check='NOT_RUN'))
    return reports


def validate_contents_assertions(assertions, declared_roles):
    if not isinstance(assertions, list):
        raise ValueError('contents assertions must be a list')
    for assertion in assertions:
        if (not isinstance(assertion, Mapping)
                or set(assertion) - {'op', 'object_role', 'domain', 'desired'}
                or assertion.get('op') != 'EMPTY_CONTENTS'
                or assertion.get('object_role') not in declared_roles
                or type(assertion.get('desired', True)) is not bool):
            raise ValueError('invalid contents assertion')
        role, domain = assertion['object_role'], assertion.get('domain')
        if domain == 'ALL_SCENE_OBJECTS_AND_SYSTEM_CONTENTS':
            continue
        if domain == 'ALL_DECLARED_RELEVANT_OBJECTS':
            members = sorted(set(declared_roles) - {role})
        elif isinstance(domain, Mapping) and set(domain) == {'kind', 'roles'} and domain['kind'] == 'EXPLICIT_ROLE_SET':
            members = domain['roles']
        else:
            raise ValueError('unsupported contents domain')
        if (not isinstance(members, list) or not members
                or any(not isinstance(r, str) or r not in declared_roles for r in members)
                or len(set(members)) != len(members) or role in members):
            raise ValueError('empty, duplicate or unbound contents domain')


def read_contents_assertions(scene, assertions, role_object, declared_roles):
    """Private initial truth, including occupied branches; never infer from cues.

    The caller freezes these results before the first action. Full-scene queries
    use the same object/particle reader as construction and frozen reset.
    """
    validate_contents_assertions(assertions, declared_roles)
    reports = []
    for assertion in assertions:
        role, domain = assertion['object_role'], assertion.get('domain')
        if domain == 'ALL_SCENE_OBJECTS_AND_SYSTEM_CONTENTS':
            evidence = read_empty_contents(scene, [role], role_object)[0]
            empty = evidence['passed']
        else:
            if domain == 'ALL_DECLARED_RELEVANT_OBJECTS':
                members = sorted(set(declared_roles) - {role})
            else:
                members = domain['roles']
            container = role_object(role)
            inside = []
            for member in members:
                obj = role_object(member)
                if obj is container:
                    raise ValueError('contents domain aliases its own container')
                truth = canonical_relation(obj, 'Inside', container)
                if truth is None:
                    raise RuntimeError('canonical Inside returned unknown')
                if bool(truth):
                    inside.append(member)
            empty = not inside
            evidence = dict(domain_roles=list(members), inside_roles=inside)
        reports.append(dict(assertion=dict(assertion), empty=empty,
                            satisfied=empty == assertion.get('desired', True), evidence=evidence))
    return reports
