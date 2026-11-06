"""Lamp state projection and native room-switch transitions, without physics."""
from .spatial_symbolic_actions import state_by_name


def sync_symbolic_lights(scene, owner):
    from omnigibson.eval.utils.light_utils import (
        LIGHT_CONTROL_CATEGORIES, ROOM_LIGHT_CATEGORIES, _set_light_prims_visible,
    )

    previous = dict(owner._symbolic_light_state or ())
    targets = {}
    for obj in scene.objects:
        category = getattr(obj, 'category', None)
        if category not in LIGHT_CONTROL_CATEGORIES:
            continue
        state = state_by_name(obj, 'ToggledOn')
        if state is None:
            continue
        value = bool(state.get_value())
        lights = (obj,) if category != 'electric_switch' else (
            light for room in (obj.in_rooms or ())
            for light in scene.object_registry('in_rooms', room, default_val=[])
            if getattr(light, 'category', None) in ROOM_LIGHT_CATEGORIES
        )
        for light in lights:
            entry = targets.setdefault(light.name, (light, category == 'electric_switch', {}))
            entry[2][obj.name] = value  # Deduplicate shared room memberships.
    updates, snapshot = [], []
    for name, (light, room_controlled, controls) in sorted(targets.items()):
        controls = tuple(sorted(controls.items()))
        if room_controlled:
            # Native room lights start on at reset; switch values are positions,
            # not absolute lamp states. Every changed switch flips its lights.
            old_controls, visible = previous.get(name, (controls, True))
            if tuple(k for k, _ in old_controls) != tuple(k for k, _ in controls):
                raise RuntimeError(f'room light binding changed within episode: {name}')
            flips = sum(old != current for (_, old), (_, current) in zip(old_controls, controls))
            visible = bool(visible) ^ bool(flips % 2)
        else:
            visible = controls[0][1]
        updates.append((light, visible))
        snapshot.append((name, (controls, visible)))
    for light, visible in updates:
        _set_light_prims_visible(light, visible)
    # Replace, never mutate: the existing runtime checkpoint holds this exact
    # immutable value, including the projected room-light visibility.
    owner._symbolic_light_state = tuple(snapshot)
    return {'version': 'symbolic-light-sync/v3', 'controlled_lights': len(updates),
            'physics_steps': 0, 'source': 'lamp_values_and_room_switch_transitions'}
