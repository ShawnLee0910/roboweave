"""Resolve installed frozen sources without importing or activating them.

Only the private worker owns legacy import paths. These compatibility constants
refer to the default profile; new callers must pass an explicit profile.
"""
from .profiles import DEFAULT_PROFILE, source_revisions, source_roots

PROFILE = DEFAULT_PROFILE
REVISIONS = source_revisions(PROFILE)


def resolve(profile: str = PROFILE):
    return source_roots(profile)


def activate(profile: str = PROFILE):
    """Compatibility path lookup; this no longer changes Python's import path."""
    return resolve(profile)
