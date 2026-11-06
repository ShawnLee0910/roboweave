"""Shared Physical lifecycle for a manually served L2 episode.

This deliberately reuses the Physical audit and verifier used by the Oracle
path.  It owns only Physical evidence; process and V lifecycle owners compose
their reports around this session.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .physical_template_review import goal_truth
from .physical_template_runtime import (
    PhysicalCompletionVerifier,
    PhysicalTemplateAudit,
    finalize_physical_template,
    sample_physical_final_window,
    validate_template,
    write_physical_template_failure,
)
from .spatial_symbolic_actions import object_pose


class PhysicalHumanSession:
    """Keep a fresh Physical episode on the same acceptance path as Oracle."""

    def __init__(self, *, task: Any, evaluator: Any, runtime: Any, directory: Path):
        from .official_predicate_verifier import OfficialPredicateVerifier
        from .registered_l0_task_runtime import _role_object_for_contract
        from .resident_scene_batch import write_json_atomic

        value = task.record.get("physical_template")
        if not isinstance(value, Mapping):
            raise ValueError("Physical human service requires a physical template")
        contract = task.template_contract()
        validate_template(value, contract["role_to_scope"])

        self.task = task
        self.evaluator = evaluator
        self.directory = Path(directory)
        self._template = value
        self._contract = contract
        self.audit = PhysicalTemplateAudit(
            value,
            lambda role: _role_object_for_contract(evaluator, contract, role),
            self.directory,
            evaluator,
        )
        self.audit.bind_runtime(runtime)
        self._verifier = PhysicalCompletionVerifier(
            OfficialPredicateVerifier(require_single_option=False, predicate_mode="official"),
            self.audit,
        )
        try:
            (
                self._initial,
                self._initial_facts,
                self._initial_goal,
                initial_readback,
            ) = self._capture_initial_readback()
        except BaseException as exc:
            self._write_initial_failure(exc)
            raise
        self._final = None
        self._result: dict[str, Any] | None = None

        write_json_atomic(self.directory / "private" / "physical_initial_binding.json", initial_readback)
        if initial_readback["status"] != "PASS":
            if initial_readback["initial_goal_false"] is not True:
                raise ValueError("Physical initial goal must be freshly measured false")
            raise ValueError("Physical initial readback did not certify a fresh task state")

    def _write_initial_failure(self, exc: BaseException) -> None:
        from .resident_scene_batch import write_json_atomic

        try:
            write_json_atomic(
                self.directory / "private" / "physical_initial_binding.json",
                {
                    "schema_version": "physical-human-initial-readback/v1",
                    "status": "FAIL",
                    "physical_execution": True,
                    "oracle_executed": False,
                    "admission_policy": "LIVE_PHYSICAL_BINDING_AND_FACTS_ONLY",
                    "not_symbolic_pose_equality": True,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        except Exception:
            pass

    def _capture_initial_readback(self):
        """Read the Physical state as it exists after this episode's reset.

        This intentionally does not compare poses against a Symbolic reset
        snapshot.  Physical admission is the declared object identity plus live
        Physical facts; authored pose equality is neither a Physical contract
        nor a stability claim.
        """
        role_to_scope = self._contract.get("role_to_scope")
        scope_bindings = self._contract.get("scope_bindings")
        if not isinstance(role_to_scope, Mapping) or not isinstance(scope_bindings, Mapping):
            raise ValueError("Physical initial readback requires exact scope bindings")

        bindings = []
        binding_failures = []
        for role in self.audit.task["roles"]:
            scope = role_to_scope.get(role)
            binding = scope_bindings.get(scope)
            if not isinstance(scope, str) or not isinstance(binding, Mapping):
                raise ValueError(f"Physical initial role is unbound: {role}")
            expected_name = binding.get("actual_name")
            if not isinstance(expected_name, str) or not expected_name:
                raise ValueError(f"Physical initial role has no object identity: {role}")
            obj = self.audit.role_object(role)
            actual_name = getattr(obj, "name", None)
            position, orientation = object_pose(obj)
            position = np.asarray(position, dtype=float)
            orientation = np.asarray(orientation, dtype=float)
            finite_pose = position.shape == (3,) and bool(np.isfinite(position).all())
            finite_orientation = orientation.shape == (4,) and bool(np.isfinite(orientation).all())
            unit_orientation = finite_orientation and bool(
                np.isclose(np.linalg.norm(orientation), 1.0)
            )
            binding_matches = actual_name == expected_name
            row = {
                "role": role,
                "scope": scope,
                "expected_object_name": expected_name,
                "actual_object_name": actual_name,
                "binding_matches": binding_matches,
                "finite_pose": finite_pose,
                "finite_orientation": finite_orientation,
                "unit_orientation": unit_orientation,
                "position": position.tolist() if finite_pose else None,
                "orientation_xyzw": orientation.tolist() if finite_orientation else None,
            }
            row["passed"] = bool(
                binding_matches and finite_pose and finite_orientation and unit_orientation
            )
            bindings.append(row)
            if not row["passed"]:
                binding_failures.append(role)

        initial = self._verifier.evaluate(None, self.evaluator)
        facts = self.audit.facts()
        initial_goal = goal_truth(self.audit.task["goal"], facts)
        initial_goal_false = None if initial_goal is None else not initial_goal
        native_q = getattr(initial, "official_q", None)
        native_q_valid = (
            type(native_q) in (int, float)
            and bool(np.isfinite(native_q))
            and 0.0 <= native_q <= 1.0
        )
        diagnostics = dict(getattr(initial, "research_diagnostics", {}) or {})
        verifier_goal = diagnostics.get("physical_goal_current")
        verifier_goal_consistent = (
            verifier_goal is None or verifier_goal is (initial_goal is True)
        )
        passed = bool(
            not binding_failures
            and initial_goal_false is True
            and native_q_valid
            and verifier_goal_consistent
        )
        instruction_id = str(
            self.task.record.get("candidate_id")
            or self.task.record.get("task_spec_id")
            or getattr(self.task, "candidate_id", "")
        )
        report = {
            "schema_version": "physical-human-initial-readback/v1",
            "status": "PASS" if passed else "FAIL",
            "instruction_id": instruction_id,
            "physical_execution": True,
            "oracle_executed": False,
            "admission_policy": "LIVE_PHYSICAL_BINDING_AND_FACTS_ONLY",
            "not_symbolic_pose_equality": True,
            "role_bindings": bindings,
            "binding_failures": binding_failures,
            "initial_goal_false": initial_goal_false,
            "initial_native_q": native_q,
            "initial_native_q_valid": native_q_valid,
            "verifier_physical_goal_current": verifier_goal,
            "verifier_goal_consistent": verifier_goal_consistent,
            "initial_facts": facts,
        }
        return initial, facts, initial_goal, report

    def initial_readback(self) -> dict[str, Any]:
        """Re-read and persist Physical admission facts without changing the world."""
        try:
            _, _, _, report = self._capture_initial_readback()
        except BaseException as exc:
            self._write_initial_failure(exc)
            raise
        from .resident_scene_batch import write_json_atomic

        write_json_atomic(
            self.directory / "private" / "physical_initial_binding.json", report
        )
        return report

    def observe(self):
        """Return the existing contact/audit context for one executed action."""
        return self.audit.observe()

    def complete_goal_now(self) -> bool:
        """Check the current full Physical formula; final stability remains in finish()."""
        snapshot = self._verifier.evaluate(None, self.evaluator)
        self._final = snapshot
        return bool(snapshot.ok and snapshot.official_success)

    def final_verification(self):
        if self._final is None:
            self._final = self._verifier.evaluate(None, self.evaluator)
        return self._final

    def finish(self) -> dict[str, Any]:
        """Run the mandatory final Physical window once and persist its review."""
        if self._result is not None:
            return dict(self._result)

        try:
            with self.observe():
                sample_physical_final_window(self.audit)
            final, review = finalize_physical_template(
                audit=self.audit,
                verifier=self._verifier,
                evaluator=self.evaluator,
                template=self._template,
                initial_facts=self._initial_facts,
                initial_complete_goal=self._initial_goal,
                initial_native_q=self._initial.official_q,
            )
            self._final = final
            passed = bool(final.ok and review["qualification_status"] == "PASS")
            diagnostics = dict(getattr(final, "research_diagnostics", {}) or {})
            self._result = {
                "status": "PASS" if passed else "FAIL_PHYSICAL_TEMPLATE",
                "task_success": passed,
                "official_q": final.official_q,
                "official_success": final.official_success,
                "native_proxy_success": diagnostics.get(
                    "native_proxy_success", final.official_success
                ),
                "physical_template_review": review,
                "failure_class": (
                    None if passed else "PHYSICAL_TEMPLATE_REQUIREMENT_NOT_CERTIFIED"
                ),
            }
            return dict(self._result)
        except BaseException as exc:
            write_physical_template_failure(self.audit, exc)
            raise
