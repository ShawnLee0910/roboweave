"""Task-family preparation; native imports occur only inside profile workers."""

from ..profiles import DEFAULT_PROFILE
from ..tasks import profile_for, validate_model_task


def _adapter(profile):
    family = profile_for(profile)["family"]
    if family in {"L1-A", "L1-M", "L1-V"}:
        from . import amv
        return amv
    if family in {"L1-P", "L2"}:
        from . import physical
        return physical
    raise ValueError(f"No family adapter for {profile}")


def resolve_inputs(config, task, manifest, rows, root):
    return _adapter(task["profile"]).resolve_inputs(config, task, manifest, rows, root)


def inspect_inputs(profile, inputs):
    if inputs.get("profile", profile) != profile:
        raise ValueError("Task inputs do not match the selected source profile")
    task = validate_model_task(profile, inputs.get("model_task"))
    if inputs.get("instruction_id") != task["runtime_instruction_id"]:
        raise ValueError("Native instruction differs from the frozen public task")
    if task["family"] == "L1-V" and inputs.get("public_config_id") != task["task_id"]:
        raise ValueError("V collection configuration differs from the frozen public task")
    if profile != DEFAULT_PROFILE:
        return _adapter(profile).inspect_inputs(profile, inputs)
    from pathlib import Path
    from instruction_handoff import compile_handoff, native_instruction_arguments
    matrix = compile_handoff(Path(inputs["handoff_json"]), {"model": 1})
    selected = matrix["instructions"][inputs["instruction_id"]]
    task = inputs["model_task"]
    identity = selected["source_task_identity"]
    if (task["task_split"] != identity["split"] or task["task_instance_id"] != str(identity["instance_id"])
            or task["task_seed"] != selected["seed"]):
        raise ValueError("Materialized task identity differs from its frozen model binding")
    return {**inputs, "entrypoint": "prepared", "native_args": native_instruction_arguments(
        Path(inputs["handoff_json"]), inputs["instruction_id"]), "cpu_admission": "PASS"}
