"""Public task catalog. Private preparation and verifier inputs live in assets."""

import json
from collections.abc import Mapping
from importlib.resources import files


def list_tasks(family=None):
    """Return independent public records in frozen source order."""
    tasks = json.loads(files("roboweave").joinpath("data/tasks/catalog.json").read_text())["tasks"]
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
    profiles = json.loads(files("roboweave").joinpath("data/tasks/profiles.json").read_text())
    key = task["profile"] if isinstance(task, dict) else task
    try:
        return profiles[key]
    except KeyError:
        raise ValueError(f"Unknown version profile: {key}") from None


def validate_model_task(profile, model_task):
    """Bind the model-facing task to the catalog before contacting a scene."""
    definition = profile_for(profile)
    if not isinstance(model_task, Mapping) or type(model_task.get("task_id")) is not int:
        raise ValueError("Frozen model task requires an integer public handle")
    matches = [task for task in list_tasks(definition["family"])
               if task["profile"] == profile and task["source"]["row_index"] == model_task["task_id"]]
    if len(matches) != 1:
        raise ValueError("Model public handle does not identify a task in its frozen profile")
    task = matches[0]
    name = task["task_id"] if task["family"] == "L1-V" else task["runtime_instruction_id"]
    if (model_task.get("profile", profile) != profile or model_task.get("task_name") != name
            or model_task.get("instruction") != task["instruction_en"]):
        raise ValueError("Model task identity or instruction differs from the frozen catalog")
    budget = model_task.get("budget")
    if (not isinstance(budget, Mapping) or set(budget) != set(task["budget"])
            or any(type(budget[key]) is not int or budget[key] != expected
                   or type(model_task.get(key)) is not int or model_task[key] != expected
                   for key, expected in task["budget"].items())):
        raise ValueError("Model task budget differs from the frozen catalog")
    provenance = model_task.get("instruction_provenance")
    if (not isinstance(provenance, Mapping) or provenance.get("profile") != profile
            or provenance.get("instruction_sha256") != task["instruction_sha256"]):
        raise ValueError("Model instruction provenance differs from the frozen catalog")
    return task
