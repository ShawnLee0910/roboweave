"""Dispatch supported V protocols without enabling selected-R4-only strategies."""
from collections.abc import Mapping
from .r4_package_gate import enabled as selected_r4_enabled


def protocol_name(config):
    if not isinstance(config,Mapping):return None
    schema=config.get('schema_version')
    if schema=='behavior.v-r1-package/v1':return 'R1_LEGACY'
    if schema=='behavior.v-r3-package/v1':return 'R3_LEGACY'
    if schema=='behavior.v-r4-package/v1':
        if selected_r4_enabled(config):return 'R4_BEFORE_SELECTED_PLACE_INSIDE'
        mode=config.get('trigger_mode','AFTER_OPEN')
        if mode in {'AFTER_OPEN','AFTER_OPEN_THEN_TARGET_PICK'}:
            return 'R4_'+mode
    return None


def require_supported(config):
    if protocol_name(config) is None:
        raise ValueError('unsupported explicit V protocol; R2, unknown schemas and incomplete selected-R4 packages are not enabled')


def session_for(evaluator):
    session=getattr(evaluator,'_v_intervention_session',None)
    return session if protocol_name(getattr(session,'config',None)) is not None else None


def is_r4_session(session):
    name=protocol_name(getattr(session,'config',None))
    return bool(name and name.startswith('R4_'))
