"""Single opt-in contract for the new selected-reference R4 protocol."""
from collections.abc import Mapping


def enabled(config):
    return (isinstance(config, Mapping)
            and config.get('schema_version') == 'behavior.v-r4-package/v1'
            and config.get('trigger_mode') == 'BEFORE_SELECTED_PLACE_INSIDE'
            and isinstance(config.get('donor_v_package'), Mapping))


def session_for(evaluator):
    session = getattr(evaluator, '_v_intervention_session', None)
    return session if enabled(getattr(session, 'config', None)) else None
