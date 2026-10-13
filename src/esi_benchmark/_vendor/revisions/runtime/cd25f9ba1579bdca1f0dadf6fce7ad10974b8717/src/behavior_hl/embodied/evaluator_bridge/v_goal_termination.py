"""Instance-local V gate for OmniGibson's predicate-success termination.

This only separates environment termination from goal truth. The V session owns
its explicit stop policy and must call cleanup when it releases the task. No
OmniGibson import, world mutation, success masking, or budget reset occurs here.
"""

from copy import deepcopy
from types import MethodType


def install_goal_termination_gate(task):
    """Defer only PredicateGoal success and return an idempotent cleanup.

    The original condition result is retained under the private
    ``info['done']['v_goal_termination']`` report. Predicate ``success`` and
    ``goal_status`` remain truthful, including after a later regression. The
    environment-facing predicate ``done`` is suppressed so Env._post_step's
    terminated/truncated accounting agrees with task.done. Every other
    termination condition, including simultaneous timeout, remains effective.

    Duplicate installation and unknown / inconsistent SDK contracts raise;
    normal task.reset() keeps the gate installed without retaining goal truth.
    """
    marker = "_v_goal_termination_gate"
    if not isinstance(getattr(task, "__dict__", None), dict):
        raise TypeError("V termination gate requires a mutable task instance")
    if hasattr(task, marker):
        raise RuntimeError("V termination gate already installed")
    conditions = getattr(task, "_termination_conditions", None)
    predicate = conditions.get("predicate") if isinstance(conditions, dict) else None
    original = getattr(task, "_step_termination", None)
    if (
        not callable(original)
        or type(predicate).__name__ != "PredicateGoal"
        or type(predicate).__module__ != "omnigibson.termination_conditions.predicate_goal"
        or getattr(predicate, "_terminate_is_success", None) is not True
        or not callable(getattr(predicate, "_check_goal_fn", None))
        or getattr(getattr(predicate, "step", None), "__module__", None)
        != "omnigibson.termination_conditions.termination_condition_base"
    ):
        raise TypeError("Unsupported OmniGibson predicate termination contract")

    missing = object()
    previous_override = task.__dict__.get("_step_termination", missing)
    cleaned = False

    def gated_step(self, env, action, info=None):
        current_conditions = getattr(self, "_termination_conditions", None)
        if not isinstance(current_conditions, dict) or current_conditions.get("predicate") is not predicate:
            raise RuntimeError("V predicate condition replaced while gate installed")
        done, raw = original(env=env, action=action, info=info)
        rows = raw.get("termination_conditions") if isinstance(raw, dict) else None
        if (
            type(done) is not bool
            or not isinstance(rows, dict)
            or set(rows) != set(current_conditions)
            or any(
                not isinstance(row, dict)
                or type(row.get("done")) is not bool
                or type(row.get("success")) is not bool
                or (row["success"] and not row["done"])
                for row in rows.values()
            )
            or done != any(row["done"] for row in rows.values())
            or type(raw.get("success")) is not bool
            or raw["success"] != any(row["success"] for row in rows.values())
        ):
            raise RuntimeError("Inconsistent OmniGibson termination result; V gate refused")
        result = dict(raw)
        result["termination_conditions"] = {name: dict(row) for name, row in rows.items()}
        deferred = rows["predicate"]["done"] and rows["predicate"]["success"]
        result["v_goal_termination"] = {
            "original_done": done,
            "original_termination_conditions": deepcopy(rows),
            "predicate_success_deferred": deferred,
        }
        if deferred:
            result["termination_conditions"]["predicate"]["done"] = False
        return any(row["done"] for row in result["termination_conditions"].values()), result

    wrapped = MethodType(gated_step, task)

    def cleanup():
        nonlocal cleaned
        if cleaned:
            return
        if task.__dict__.get("_step_termination") is not wrapped or task.__dict__.get(marker) is not cleanup:
            raise RuntimeError("V termination gate ownership changed; refusing to overwrite")
        if previous_override is missing:
            del task.__dict__["_step_termination"]
        else:
            task._step_termination = previous_override
        del task.__dict__[marker]
        cleaned = True

    task._step_termination = wrapped
    setattr(task, marker, cleanup)
    return cleanup
