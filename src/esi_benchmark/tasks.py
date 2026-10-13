"""Public task catalog. Private preparation and verifier inputs live in assets."""

import json
from importlib.resources import files


def list_tasks(family=None):
    """Return independent public records in frozen source order."""
    tasks = json.loads(files("esi_benchmark").joinpath("data/tasks/catalog.json").read_text())["tasks"]
    if family is None:
        return tasks
    families = {task["family"] for task in tasks} | {"L2"}
    if family not in families:
        raise ValueError(f"Unknown task family: {family}; choose {', '.join(sorted(families))}")
    return [task for task in tasks if task["family"] == family]


def get_task(task_id, family=None):
    """Look up a public ID; require a family when an ID is ambiguous."""
    matches = [task for task in list_tasks(family) if task["task_id"] == task_id]
    if not matches:
        raise ValueError(f"Unknown task: {task_id}" + (f" in {family}" if family else ""))
    if len(matches) != 1:
        raise ValueError(f"Ambiguous task: {task_id}; provide family")
    return matches[0]


def profile_for(task):
    """Return the exact frozen profile, never a floating current revision."""
    profiles = json.loads(files("esi_benchmark").joinpath("data/tasks/profiles.json").read_text())
    key = task["profile"] if isinstance(task, dict) else task
    try:
        return profiles[key]
    except KeyError:
        raise ValueError(f"Unknown version profile: {key}") from None
