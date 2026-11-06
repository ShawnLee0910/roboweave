"""Process-local registration for authored BDDL tasks without editing official data."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from behavior_hl.embodied.evaluator_bridge.oracle_progress_journal import json_safe
from behavior_hl.embodied.evaluator_bridge.construction_progress import checkpoint
from behavior_hl.embodied.evaluator_bridge.l0_reset_pose_overlay import load_l0_reset_pose_overlay
from behavior_hl.embodied.evaluator_bridge.private_visual_markers import MARKER_LABEL_COUNTERBALANCE
from behavior_hl.embodied.evaluator_bridge.construction_contents_readback import read_empty_contents, read_contents_assertions
from behavior_hl.embodied.evaluator_bridge.construction_camera_pose import (
    head_camera_position, orientation_toward_camera, camera_pose_readback,
)

from behavior_hl.embodied.evaluator_bridge.deterministic_inside_placement import (
    Aabb3, plan_deterministic_inside_candidates,
)
from behavior_hl.embodied.evaluator_bridge.on_top_support_layers import (
    read_base_aligned_box, matrix_to_quaternion_xyzw, quaternion_angle_rad,
    DEFAULT_PROTECTED_ROTATION_RAD,
)

from behavior_hl.embodied.evaluator_bridge import inside_task_profiles
from behavior_hl.embodied.evaluator_bridge.construction_material_adapter import (
    apply_vray_material_rules,
    is_vray_material,
    restore_material_changes,
    validate_vray_material_rules,
)
from behavior_hl.embodied.evaluator_bridge.inside_task_profiles import InsideTaskProfile
from behavior_hl.embodied.evaluator_bridge.mixed_goal_graph import (
    compile_grounded_mixed_option,
    relation_from_grounded_leaf,
)
from behavior_hl.embodied.evaluator_bridge.l0_taskspec_goal_plan import (
    audit_goal_plan_navigation_route,
)
from behavior_hl.embodied.evaluator_bridge.spatial_symbolic_actions import (
    SpatialPoseCandidate,
    candidate_aabb,
    candidate_blockers,
    canonical_relation,
    read_action_relation,
    symbolic_on_top_relation,
    object_aabb,
    object_pose,
    plan_default_on_top_candidates,
    state_by_name,
    _object_velocity, DEFAULT_STABILITY_WINDOW,
    DEFAULT_MAX_LINEAR_SPEED_M_S, DEFAULT_MAX_ANGULAR_SPEED_RAD_S,
    _screen_axes_xy, _candidate_center_xy,
)
from behavior_hl.embodied.runtime.plugins.restricted_kinematic_base import (
    RESTRICTED_KINEMATIC_PUBLIC_MOVE_M,
)


CATALOG_VERSION = "action20-capability-benchmark.runtime-registration/v1"
LIVE_CONSTRUCTION_VERSION = "behavior2026.registered-l0-live-construction/v1"
NAVIGATION_AUDIT_VERSION = "behavior2026.registered-l0-navigation-audit/v2"
LIVE_CONSTRUCTION_SETTLE_STEPS = 30
ROBOT_CONSTRUCTION_HOLD_POSITION_TOLERANCE_M = 1.0e-4
ROBOT_CONSTRUCTION_HOLD_ORIENTATION_TOLERANCE_RAD = 1.0e-4
_ACTIVE_REGISTERED_L0_TASKS: dict[str, "RegisteredL0Task"] = {}


class RegisteredL0ConstructionError(RuntimeError):
    def __init__(self, message: str, report: Mapping[str, Any]):
        super().__init__(message)
        self.report = dict(report)


def is_registered_l0_task(task_name: str) -> bool:
    return task_name in _ACTIVE_REGISTERED_L0_TASKS


def registered_preparation_room(task_name):
    task = _ACTIVE_REGISTERED_L0_TASKS.get(task_name)
    if task is None or task.reset_pose_record is not None:
        return None  # Legacy explicit overlays retain their own frozen contract.
    return task.record.get('construction', {}).get('robot_start', {}).get('preparation_room')


def registered_prepared_oracle_state(task_name):
    task = _ACTIVE_REGISTERED_L0_TASKS.get(task_name)
    return None if task is None else task.record.get('_prepared_oracle_state')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_ref(value: Any, field: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a file reference")
    path = Path(str(value.get("path", ""))).resolve()
    digest = value.get("sha256")
    if not path.is_file() or not isinstance(digest, str) or _sha256(path) != digest:
        raise ValueError(f"{field} file reference is unavailable or changed")
    return path


@dataclass(frozen=True, slots=True)
class RegisteredL0Task:
    record: Mapping[str, Any]
    bddl_path: Path
    template_path: Path
    tro_state_path: Path
    template_contract_path: Path
    reset_pose_record: Mapping[str, Any] | None = None

    @property
    def candidate_id(self) -> str:
        return str(self.record["candidate_id"])

    @property
    def task_name(self) -> str:
        return str(self.record["activity_name"])

    @property
    def task_id(self) -> int:
        return int(self.record["task_id"])

    @property
    def scene_model(self) -> str:
        return str(self.record["scene_model"])

    @property
    def split(self) -> str:
        return str(self.record["split"])

    @property
    def instance_id(self) -> int:
        return int(self.record["instance_id"])

    def profile(self) -> InsideTaskProfile:
        profile = dict(self.record["profile"])
        profile["canonical_goal_leaves"] = tuple(
            json.dumps(leaf, sort_keys=True, separators=(",", ":"))
            for leaf in profile["canonical_goal_leaves"]
        )
        return InsideTaskProfile(**profile)

    def available_task(self) -> dict[int, dict[str, Any]]:
        start = (self.record["construction"]["robot_start"] if self.reset_pose_record is None
                 else self.reset_pose_record["fixed_robot_pose"])
        return {
            0: {
                "scene_model": self.scene_model,
                "robot_start_position": list(start["position"]),
                "robot_start_orientation": list(start["orientation"]),
                "load_room_instances": list(self.record.get("rooms", ())) or None,
            }
        }

    def template_contract(self) -> Mapping[str, Any]:
        return json.loads(self.template_contract_path.read_text())


def load_registered_l0_tasks(path: Path, *, reset_pose_overlay: Path | None = None) -> Mapping[str, RegisteredL0Task]:
    catalog_path = Path(path).resolve()
    reset_poses = {} if reset_pose_overlay is None else load_l0_reset_pose_overlay(reset_pose_overlay)
    payload = json.loads(catalog_path.read_text())
    records = payload.get("tasks")
    if (
        payload.get("schema_version") not in {CATALOG_VERSION, CATALOG_VERSION.replace('/v1', '/v2')}
        or not isinstance(records, list)
        or len(records) != payload.get("task_count")
        or payload.get("loader_ready") is not True
    ):
        raise ValueError("registered L0 task catalog is malformed")
    capabilities = payload.get('required_capabilities', [])
    if (not isinstance(capabilities, list) or any(c not in {
            'EMPTY_CONTENTS_READBACK_V1', 'CAMERA_RELATIVE_POSE_V1', 'ROBOT_START_ROOM_V1', 'BOUNDED_INITIAL_LAYOUT_V1',
            'CALIBRATION_ONLY_V1', 'LIGHT_PROBE_V1', 'VISUAL_RULES_V1',
            'PAIRED_INITIAL_STATE_V1', 'L1_INITIAL_BRANCH_V1', 'L1_PUBLIC_PROCESS_V1',
            'L1_DECLARED_INITIAL_FACTS_V1',
            'L1_INITIAL_CONTENTS_ASSERTIONS_V1',
            'L1_STATE_PROTECTION_V1',
            'L1_FULL_GOAL_ACCEPTANCE_V1',
            'L1_POSITIVE_GOAL_PROJECTION_V1',
            'L1_CURRENT_REFERENCE_V1', 'RELATIONAL_INITIAL_PLACEMENT_V1',
            'L1_ORDERED_MILESTONES_V1',
            'L1_UNARY_TRANSITION_LEDGER_V1',
            'L1_LAST_CYCLE_FINAL_V1',
            'L1_HISTORY_BINDINGS_V1',
            'L1_GLOBAL_UNARY_PHASES_V1', 'L1_EXACT_CYCLE_LEDGER_V1', 'L1_CYCLE_ORDER_FINAL_V1',
            'L1_MAINTAIN_UNTIL_V1', 'L1_SUBJECT_HISTORY_BINDING_V1', 'L1_DYNAMIC_PROCESS_GOAL_V1',
            'FROZEN_PREPARED_ORACLE_STATE_V1',
            'PHYSICAL_RECEIVING_SUPPORT_V1', 'P12_PHYSICAL_TEMPLATE_V1',
        } for c in capabilities)
            or (payload['schema_version'].endswith('/v2') and 'EMPTY_CONTENTS_READBACK_V1' not in capabilities)):
        raise ValueError('unsupported registered construction capabilities')
    result: dict[str, RegisteredL0Task] = {}
    prepared_states = {}
    if 'FROZEN_PREPARED_ORACLE_STATE_V1' in capabilities:
        from .prepared_oracle_state import load_prepared_oracle_states
        prepared_states = load_prepared_oracle_states(payload.get('prepared_oracle_states'), records)
    elif payload.get('prepared_oracle_states') is not None:
        raise ValueError('prepared Oracle capability missing')
    task_ids: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("registered L0 task record must be a mapping")
        if prepared_states:
            record = dict(record, _prepared_oracle_state=prepared_states[record['task_spec_id']])
        task = RegisteredL0Task(
            record=dict(record),
            bddl_path=_file_ref(record.get("bddl"), "bddl"),
            template_path=_file_ref(record.get("template"), "template"),
            tro_state_path=_file_ref(record.get("tro_state"), "tro_state"),
            template_contract_path=_file_ref(
                record.get("template_contract"), "template_contract"
            ),
            reset_pose_record=reset_poses.get(str(record.get("candidate_id"))),
        )
        if task.candidate_id in result or task.task_id in task_ids:
            raise ValueError("registered L0 task names and IDs must be unique")
        if task.record.get("oracle_route") != "GROUNDED_GOAL_GRAPH":
            raise ValueError("registered L0 task does not use the grounded-goal Oracle")
        task.profile()
        from .l1_initial_branch import CAPABILITY, DECLARED_FACTS_CAPABILITY, uses_declared_initial_facts, validate_initial_branch
        template_contract = task.template_contract()
        if 'physical_template' in record or 'physical_template' in template_contract:
            from .physical_template_runtime import validate_template, CAPABILITY as P12_CAPABILITY
            if P12_CAPABILITY not in capabilities or record.get('physical_template') != template_contract.get('physical_template'):
                raise ValueError('physical template capability/correspondence missing')
            validate_template(record['physical_template'],template_contract['role_to_scope'])
        from .receiving_support_construction import validate_receiving_support, CAPABILITY as RECEIVER_CAPABILITY
        for recipe in template_contract.get('initial_state_recipe', []):
            validate_receiving_support(recipe, template_contract['role_to_scope'])
            if recipe.get('receiving_support') and RECEIVER_CAPABILITY not in capabilities:
                raise ValueError('Physical receiving support capability missing')
        layout_recipes=[r for r in template_contract.get('initial_state_recipe',[]) if r.get('placement_policy')]
        if layout_recipes:
            from .initial_layout_search import validate_layout_contract
            # Goal scopes are immutable; no alternate support may be a goal participant.
            validate_layout_contract(template_contract,[])
            alternatives={template_contract['role_to_scope'][role] for r in layout_recipes for role in r['placement_policy']['support_roles']}
            def mentions(value):
                return any(mentions(v) for v in value) if isinstance(value,list) else isinstance(value,str) and value in alternatives
            if mentions(record['profile']['canonical_goal_leaves']):raise ValueError('alternate support is a canonical goal participant')
            if 'BOUNDED_INITIAL_LAYOUT_V1' not in capabilities:raise ValueError('initial layout capability missing')
            from .initial_layout_search import validate_policy
            for recipe in layout_recipes:
                validate_policy(recipe,template_contract['role_to_scope'])
                for role in recipe['placement_policy']['support_roles']:
                    support=template_contract['scope_bindings'][template_contract['role_to_scope'][role]]
                    if support['binding_kind']!='SCENE_OBJECT':raise ValueError('alternate supports must be registered native objects')
        if any(a.get('predicate') in {'under','nextto','next_to'} for a in template_contract.get('initial_state_recipe', [])) and 'RELATIONAL_INITIAL_PLACEMENT_V1' not in capabilities:
            raise ValueError('relational initial placement capability missing')
        branch = template_contract.get('task_contract')
        if branch is not None or record.get('task_contract') is not None:
            if CAPABILITY not in capabilities or branch != record.get('task_contract'):
                raise ValueError('L1 initial branch capability or contract correspondence missing')
            validate_initial_branch(branch, template_contract.get('role_to_scope', {}))
            if branch.get('initial_contents_assertions') and 'L1_INITIAL_CONTENTS_ASSERTIONS_V1' not in capabilities:
                raise ValueError('L1 initial contents assertion capability missing')
            if branch.get('process_requirements', {}).get('history_program'):
                if 'L1_HISTORY_BINDINGS_V1' not in capabilities:
                    raise ValueError('L1 history bindings capability missing')
                from .l1_history_bindings import validate_history_program, required_history_capabilities
                from .benchmark_package_correspondence import _clause, _formula
                history = branch['process_requirements']['history_program']
                validate_history_program(history, template_contract['role_to_scope'])
                if not set(required_history_capabilities(history)) <= set(capabilities):raise ValueError('history extension capability missing')
                if branch.get('full_goal_acceptance'):
                    from .l1_dynamic_process_goals import validate_dynamic_goal, CAPABILITY as DYNAMIC_CAPABILITY
                    if DYNAMIC_CAPABILITY not in capabilities:raise ValueError('dynamic process capability missing')
                    validate_dynamic_goal(branch['process_requirements'], template_contract.get('goal_formula'), template_contract['role_to_scope'])
                expected = _formula(['and', *[_clause(g, template_contract['role_to_scope'], alternatives=False) for g in history['clauses']]])
                if not branch.get('full_goal_acceptance') and expected != _formula(['and', *record['profile']['canonical_goal_leaves']]):
                    raise ValueError('history static goals differ from registered canonical goals')
            if branch.get('process_requirements', {}).get('process_program'):
                from .l1_am_process_programs import CAPABILITY as MILESTONE_CAPABILITY
                from .l1_am_process_programs import EXTENSIONS
                if MILESTONE_CAPABILITY not in capabilities:
                    raise ValueError('L1 milestone capability missing')
                if any(c not in capabilities for field,c in EXTENSIONS.items() if field in branch['process_requirements']['process_program']):raise ValueError('milestone extension capability missing')
                from .benchmark_package_correspondence import _clause, _formula
                program_goals = branch['process_requirements']['process_program']['final_requirements']
                dynamic = 'cycle_order_final' in branch['process_requirements']['process_program']
                if dynamic:
                    from .l1_dynamic_process_goals import validate_dynamic_goal, CAPABILITY as DYNAMIC_CAPABILITY
                    if not branch.get('full_goal_acceptance') or DYNAMIC_CAPABILITY not in capabilities:raise ValueError('dynamic process full-goal capability missing')
                    validate_dynamic_goal(branch['process_requirements'], template_contract.get('goal_formula'), template_contract['role_to_scope'])
                if not dynamic and _formula(['and', *(_clause(a, template_contract['role_to_scope'], alternatives=False) for a in program_goals)]) != _formula(['and', *record['profile']['canonical_goal_leaves']]):
                    raise ValueError('milestone final requirements differ from profile goals')
            if uses_declared_initial_facts(branch) and DECLARED_FACTS_CAPABILITY not in capabilities:
                raise ValueError('L1 declared initial facts capability missing')
            if branch.get('schema_version') == 'l1-initial-branch/v2' and 'L1_PUBLIC_PROCESS_V1' not in capabilities:
                raise ValueError('L1 public process capability missing')
            if branch.get('process_requirements', {}).get('state_protection_roles') and 'L1_STATE_PROTECTION_V1' not in capabilities:
                raise ValueError('L1 state protection capability missing')
            if branch.get('full_goal_acceptance') and 'L1_FULL_GOAL_ACCEPTANCE_V1' not in capabilities:
                raise ValueError('L1 complete goal acceptance capability missing')
            if branch.get('full_goal_acceptance') and template_contract.get('goal_formula') is None:
                raise ValueError('L1 complete goal formula missing')
            if branch.get('process_requirements', {}).get('current_reference_contracts'):
                if 'L1_CURRENT_REFERENCE_V1' not in capabilities:
                    raise ValueError('L1 current reference capability missing')
                from .l1_process_audit import validate_current_reference_formula
                validate_current_reference_formula(branch,template_contract.get('goal_formula'))
            if branch.get('positive_goal_projection'):
                if 'L1_POSITIVE_GOAL_PROJECTION_V1' not in capabilities:
                    raise ValueError('L1 positive goal projection capability missing')
                if template_contract.get('full_goal_witness') is None:
                    raise ValueError('L1 complete goal witness missing')
        result[task.candidate_id] = task
        task_ids.add(task.task_id)
    return result


def _live_object(scene: Any, name: str) -> Any:
    obj = scene.object_registry("name", name)
    if obj is None:
        raise RuntimeError(f"registered live object is missing: {name}")
    return obj


def _matching_vector(value: Any, template: Any) -> Any:
    factory = getattr(template, "new_tensor", None)
    return factory(value) if callable(factory) else value


def _clone_runtime_value(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    return np.array(value, copy=True)


def _zero_runtime_value(value: Any) -> Any:
    try:
        return value * 0
    except Exception:
        return np.zeros_like(value)


def _numpy_runtime_value(value: Any) -> np.ndarray:
    item = value
    for method_name in ("detach", "cpu"):
        method = getattr(item, method_name, None)
        if callable(method):
            item = method()
    numpy_method = getattr(item, "numpy", None)
    if callable(numpy_method):
        item = numpy_method()
    return np.asarray(item, dtype=float)


class _RegisteredConstructionRobotHold:
    """Kinematic-only robot pose/joint hold around construction steps."""

    def __init__(self, *, evaluator: Any, task: RegisteredL0Task) -> None:
        robot = getattr(evaluator, "robot", None)
        if robot is None:
            raise RuntimeError("registered construction evaluator has no robot")
        required = (
            "reset",
            "set_position_orientation",
            "get_position_orientation",
            "get_joint_positions",
            "set_joint_positions",
            "set_joint_velocities",
            "keep_still",
        )
        missing = [name for name in required if not callable(getattr(robot, name, None))]
        if missing:
            raise RuntimeError(
                "registered construction robot hold is missing: "
                + ", ".join(missing)
            )
        start = task.record.get("construction", {}).get("robot_start")
        if not isinstance(start, Mapping):
            raise ValueError("registered construction robot_start is missing")
        position = np.asarray(start.get("position"), dtype=float)
        orientation = np.asarray(start.get("orientation"), dtype=float)
        if (
            position.shape != (3,)
            or orientation.shape != (4,)
            or not np.isfinite(position).all()
            or not np.isfinite(orientation).all()
            or not np.isclose(np.linalg.norm(orientation), 1.0)
        ):
            raise ValueError("registered construction robot_start is invalid")
        self._robot = robot
        if start.get('preparation_room') is not None:
            from .run_b1_marker_symbolic_rollout import _private_room_at_xy
            actual_room = _private_room_at_xy(evaluator.env.scene, tuple(position[:2]))
            if actual_room != start['preparation_room']:
                raise ValueError(f'ROBOT_START_ROOM_CONTRACT_MISMATCH: {actual_room} != {start["preparation_room"]}')
        self._catalog_position = position.tolist()
        self._catalog_orientation = orientation.tolist()
        self._position: Any = None
        self._orientation: Any = None
        self._joint_positions: Any = None
        self._step_count = 0
        self._step_attempts = 0
        self._restore_count = 0
        self._restore_errors: list[str] = []
        self._max_position_error_m = 0.0
        self._max_orientation_error_rad = 0.0
        robot.reset()
        current_position, current_orientation = robot.get_position_orientation(
            frame="world",
            clone=True,
        )
        robot.set_position_orientation(
            position=_matching_vector(self._catalog_position, current_position),
            orientation=_matching_vector(self._catalog_orientation, current_orientation),
            frame="world",
        )
        self._capture_pose_and_joints()
        self.restore()

    def _capture_pose_and_joints(self) -> None:
        position, orientation = self._robot.get_position_orientation(
            frame="world",
            clone=True,
        )
        self._position = _clone_runtime_value(position)
        self._orientation = _clone_runtime_value(orientation)
        self._joint_positions = _clone_runtime_value(
            self._robot.get_joint_positions()
        )

    def restore(self) -> None:
        self._restore_count += 1
        try:
            self._robot.set_position_orientation(
                position=_clone_runtime_value(self._position),
                orientation=_clone_runtime_value(self._orientation),
                frame="world",
            )
            self._robot.set_joint_positions(
                positions=_clone_runtime_value(self._joint_positions),
                drive=False,
            )
            self._robot.set_joint_velocities(
                velocities=_zero_runtime_value(self._joint_positions),
                drive=False,
            )
            self._robot.keep_still()
            position, orientation = self._robot.get_position_orientation(
                frame="world",
                clone=True,
            )
            position_array = _numpy_runtime_value(position)
            orientation_array = _numpy_runtime_value(orientation)
            if (
                position_array.shape != (3,)
                or orientation_array.shape != (4,)
                or not np.isfinite(position_array).all()
                or not np.isfinite(orientation_array).all()
            ):
                raise RuntimeError("robot construction hold pose readback is invalid")
            position_error = float(
                np.linalg.norm(
                    position_array - np.asarray(self._catalog_position, dtype=float)
                )
            )
            orientation_error = float(
                quaternion_angle_rad(
                    orientation_array,
                    np.asarray(self._catalog_orientation, dtype=float),
                )
            )
            self._max_position_error_m = max(
                self._max_position_error_m,
                position_error,
            )
            self._max_orientation_error_rad = max(
                self._max_orientation_error_rad,
                orientation_error,
            )
            if (
                position_error > ROBOT_CONSTRUCTION_HOLD_POSITION_TOLERANCE_M
                or orientation_error
                > ROBOT_CONSTRUCTION_HOLD_ORIENTATION_TOLERANCE_RAD
            ):
                raise RuntimeError(
                    "robot construction hold pose readback exceeded tolerance: "
                    f"position={position_error:.6g}m "
                    f"orientation={orientation_error:.6g}rad"
                )
        except Exception as exc:
            self._restore_errors.append(f"{type(exc).__name__}: {exc}")
            raise

    def step(self, bindings: Any, evaluator: Any, label: str) -> Any:
        self._step_attempts += 1
        checkpoint('ROBOT_HOLD_BEFORE_STEP', step_label=label)
        self.restore()
        try:
            result = bindings._raw_step(evaluator, label)
            self._step_count += 1
            return result
        finally:
            checkpoint('ROBOT_HOLD_AFTER_STEP', step_label=label)
            self.restore()

    def report(self) -> Mapping[str, Any]:
        return {
            "enabled": True,
            "mode": "KINEMATIC_CONSTRUCTION_ROBOT_POSE_JOINT_HOLD",
            "execution_fidelity": "kinematic_research",
            "official_robot_reset": True,
            "catalog_robot_start": {
                "position": list(self._catalog_position),
                "orientation": list(self._catalog_orientation),
            },
            "raw_step_count": self._step_count,
            "raw_step_attempts": self._step_attempts,
            "pose_restore_count": self._restore_count,
            "position_tolerance_m": ROBOT_CONSTRUCTION_HOLD_POSITION_TOLERANCE_M,
            "orientation_tolerance_rad": ROBOT_CONSTRUCTION_HOLD_ORIENTATION_TOLERANCE_RAD,
            "max_restore_position_error_m": self._max_position_error_m,
            "max_restore_orientation_error_rad": self._max_orientation_error_rad,
            "restored_after_construction": (
                not self._restore_errors
                and self._max_position_error_m
                <= ROBOT_CONSTRUCTION_HOLD_POSITION_TOLERANCE_M
                and self._max_orientation_error_rad
                <= ROBOT_CONSTRUCTION_HOLD_ORIENTATION_TOLERANCE_RAD
            ),
            "restore_errors": list(self._restore_errors),
            "natural_dynamics_stability_claim": False,
        }


class _RegisteredConstructionBindingsProxy:
    """Delegate bindings while wrapping only the construction raw-step hook."""

    def __init__(self, bindings: Any, hold: _RegisteredConstructionRobotHold) -> None:
        self._bindings = bindings
        self._hold = hold

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bindings, name)

    def _raw_step(self, evaluator: Any, label: str) -> Any:
        return self._hold.step(self._bindings, evaluator, label)


def _settle(evaluator: Any, control: Any, bindings: Any, phase: str) -> None:
    for index in range(LIVE_CONSTRUCTION_SETTLE_STEPS):
        control.hold()
        bindings._raw_step(
            evaluator,
            f"{phase}.settle.{index:03d}",
        )


def _initial_orientation_readback(orientations, role_object):
    """Read authored orientation drift; semantic admission is checked separately."""
    rows = []
    for role, requested in orientations.items():
        actual = np.asarray(object_pose(role_object(role))[1], dtype=float).copy()
        requested = np.asarray(requested, dtype=float).copy()
        for quaternion in (requested, actual):
            if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
                    or np.linalg.norm(quaternion) <= 1e-8):
                raise ValueError(f'invalid initial orientation readback: {role}')
        angle = quaternion_angle_rad(actual.copy(), requested.copy())
        rows.append(dict(role=role, angular_error_rad=angle,
                         requested_orientation_xyzw=requested.tolist(),
                         settled_orientation_xyzw=actual.tolist(),
                         requested_vs_settled_angular_error_rad=angle,
                         admission_policy='diagnostic_only_semantics_checked_separately'))
    return rows


def _rank_initial_candidates(candidates, support, evaluator, rank, count, center_xy):
    if rank is None:
        return candidates
    if isinstance(rank, bool) or isinstance(count, bool) or not isinstance(rank, int) or not isinstance(count, int) or not 0 <= rank < count:
        raise ValueError("invalid counterbalance placement rank")
    lo, hi = object_aabb(support)
    right, _ = _screen_axes_xy(observer_xy=object_pose(evaluator.env.robots[0])[0][:2],
                              target_center_xy=(lo[:2] + hi[:2]) / 2)
    projections = [np.dot(right, [x, y]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
    preferred = min(projections) + (max(projections) - min(projections)) * (rank + 1) / (count + 1)
    # Stable sort retains the existing planner's order for equal preferences.
    return tuple(sorted(candidates, key=lambda c: abs(np.dot(right, center_xy(c)) - preferred)))


def _construct_on_top(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    subject: Any,
    support: Any,
    phase: str,
    placement_rank: int | None = None,
    placement_count: int | None = None,
    predicate_mode: str = "official",
    candidate_override=None,
    receiving_support=None,
) -> dict[str, Any]:
    raw_scene_objects = evaluator.env.scene.objects
    scene_objects = (
        tuple(raw_scene_objects.values())
        if isinstance(raw_scene_objects, Mapping)
        else tuple(raw_scene_objects or ())
    )
    checkpoint('ONTOP_PLAN', subject=getattr(subject, 'name', None), support=getattr(support, 'name', None))
    plan = plan_default_on_top_candidates(
        subject,
        support,
        # Whole-object AABBs are diagnostics only. The live collision-shape
        # query below is the admission authority.
        scene_objects=(),
    )
    attempts = []
    candidates = _rank_initial_candidates(plan.candidates if candidate_override is None else candidate_override, support, evaluator,
        placement_rank, placement_count, lambda c: _candidate_center_xy(subject, c))
    center_report = None
    if receiving_support is not None:
        if predicate_mode != 'official' or candidate_override is not None:
            raise ValueError('receiving support requires direct Physical initial construction')
        from .receiving_support_construction import centered_receiver_candidate
        candidates, center_report = centered_receiver_candidate(subject, support, candidates)
    if not candidates:
        return {
            "status": "FAIL",
            "planning_error": plan.planning_error,
            "receiving_support_center": center_report,
            "admission_reports": [
                {
                    "candidate": item.candidate,
                    "accepted": item.accepted,
                    "reason": item.reason,
                    "blockers": list(item.blockers),
                }
                for item in plan.admission_reports
            ],
            "attempts": attempts,
        }

    from behavior_hl.embodied.evaluator_bridge.spatial_candidate_admission import (
        execute_admitted_spatial_candidate,
        query_spatial_candidate_collision_overlap,
    )
    import omnigibson as og

    # Candidates are authored from this pose. Cache their future bounds and
    # broad blockers before any failed canonical trial can move the subject.
    candidate_bounds = {}
    broad_blockers_by_candidate = {}
    for candidate in candidates:
        checkpoint('ONTOP_BROAD_COLLISION', candidate=candidate.name, candidate_count=len(candidates))
        candidate_bounds[candidate.name] = candidate_aabb(subject, candidate)
        broad_blockers_by_candidate[candidate.name] = candidate_blockers(
            held=subject,
            target=support,
            support=support,
            candidate=candidate,
            scene_objects=scene_objects,
            margin_m=0.0,
        )

    admission_reports = []
    for candidate in candidates:
        checkpoint('ONTOP_ADMISSION', candidate=candidate.name, candidate_count=len(candidates))
        broad_blockers = broad_blockers_by_candidate[candidate.name]
        if candidate_override is not None:
            lo,hi=candidate_bounds[candidate.name];slo,shi=object_aabb(support)
            from .spatial_symbolic_actions import DEFAULT_SUPPORT_INSET_M
            if np.any(lo[:2]<slo[:2]+DEFAULT_SUPPORT_INSET_M) or np.any(hi[:2]>shi[:2]-DEFAULT_SUPPORT_INSET_M):
                attempts.append(dict(candidate=candidate.name,accepted=False,reason='LIVE_SUPPORT_FOOTPRINT_REJECTED'))
                continue

        def apply_candidate(item):
            raw_position, raw_orientation = subject.get_position_orientation()
            subject.set_position_orientation(
                position=_matching_vector(item.position, raw_position),
                orientation=_matching_vector(
                    item.orientation_xyzw, raw_orientation
                ),
                frame="world",
            )
            keep_still = getattr(subject, "keep_still", None)
            if callable(keep_still):
                keep_still()
            _settle(evaluator, control, bindings, phase)

        execution = execute_admitted_spatial_candidate(
            candidate=candidate,
            broad_blockers=broad_blockers,
            collision_shape_query=lambda item: query_spatial_candidate_collision_overlap(
                omnigibson=og,
                held=subject,
                closure_objects=(subject,),
                # Match the Action OnTop kernel: the authored support contact
                # is geometry-certified; unrelated collision hits still block.
                geometry_certified_objects=(support,),
                candidate=item,
                candidate_aabb_world=candidate_bounds[item.name],
            ),
            apply_candidate=apply_candidate,
        )
        collision_report = execution.collision_shape_report
        admission = execution.decision
        admission_private = admission.to_private_dict()
        collision_private = collision_report.to_private_dict()
        admission_reports.append(
            {
                "candidate": candidate.name,
                "accepted": admission.accepted_for_execution,
                "reason": admission.reason,
                "blockers": list(admission.broad_blockers),
                "candidate_admission": admission_private,
                "collision_shape_admission": collision_private,
            }
        )
        if not execution.applied:
            attempts.append(
                {
                    "candidate": candidate.name,
                    "broad_blockers": list(broad_blockers),
                    "blocking_objects": list(broad_blockers),
                    "candidate_admission": admission_private,
                    "collision_shape_admission": collision_private,
                    "accepted": False,
                }
            )
            continue

        checkpoint('ONTOP_SETTLED_READBACK', candidate=candidate.name)
        settled_position, settled_orientation = object_pose(subject)
        subject_min, _ = object_aabb(subject)
        _, support_max = object_aabb(support)
        gap = float(subject_min[2] - support_max[2])
        settled_candidate = SpatialPoseCandidate(
            name=candidate.name,
            position=settled_position,
            orientation_xyzw=settled_orientation,
        )
        blockers = candidate_blockers(
            held=subject,
            target=support,
            support=support,
            candidate=settled_candidate,
            scene_objects=scene_objects,
            margin_m=0.0,
        )
        checkpoint('ONTOP_CANONICAL_READBACK', candidate=candidate.name)
        on_top = _registered_symbolic_relation_truth(
            subject, "on_top", support, predicate_mode=predicate_mode,
        )
        # The selected runtime relation is authoritative after settling. Broad live
        # AABBs for terrain layers may overlap even when OnTop is satisfied;
        # retain those blockers as diagnostics instead of a second verifier.
        accepted = bool(on_top)
        receiver_report = None
        if receiving_support is not None:
            from .receiving_support_construction import receiver_upright
            receiver_report = receiver_upright(subject)
            touching = _registered_symbolic_relation_truth(subject, 'touching', support, predicate_mode='official')
            receiver_report['touching'] = touching
            accepted = bool(accepted and touching and receiver_report['passed'])
        attempt = {
            "receiving_support_center": center_report,
            "receiving_support_upright": receiver_report,
            "candidate": candidate.name,
            "settled_position": settled_position.tolist(),
            "settled_orientation_xyzw": settled_orientation.tolist(),
            "bottom_to_support_gap_m": gap,
            "canonical_on_top": on_top,
            "predicate_mode": predicate_mode,
            "blocking_objects": list(blockers),
            "broad_blockers": list(broad_blockers),
            "candidate_admission": admission_private,
            "collision_shape_admission": collision_private,
            "accepted": accepted,
        }
        try:
            attempt["raw_official_on_top"] = read_action_relation(
                subject, "OnTop", support, predicate_mode="official",
            )
        except Exception as exc:
            attempt.update(raw_official_on_top=None, raw_official_error=f"{type(exc).__name__}: {exc}")
        if not accepted:
            try:
                adjacency = state_by_name(subject, 'VerticalAdjacency').get_value()
                attempt['canonical_components'] = {
                    'touching': canonical_relation(subject, 'Touching', support),
                    'support_below': support in adjacency.negative_neighbors,
                    'support_above': support in adjacency.positive_neighbors,
                    'subject_kinematic_only': bool(subject.kinematic_only),
                    'support_kinematic_only': bool(support.kinematic_only),
                    'linear_velocity': _object_velocity(subject, 'linear').tolist(),
                }
            except Exception as exc:
                attempt['canonical_components_error'] = f'{type(exc).__name__}: {exc}'
        attempts.append(attempt)
        if accepted:
            return {"status": "PASS", "attempts": attempts, **attempt}
    return {
        "status": "FAIL",
        "planning_error": plan.planning_error,
        "admission_reports": admission_reports,
        "attempts": attempts,
    }


def _construct_inside(*, evaluator, control, bindings, subject, support, phase,
                      placement_rank=None, placement_count=None):
    # Reuse the executor's read-only canonical meta-link adapter. Import here
    # avoids a module initialization cycle; no executor action is invoked.
    from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import _private_fillable_regions

    checkpoint('INSIDE_PLAN', subject=getattr(subject, 'name', None), support=getattr(support, 'name', None))
    box = read_base_aligned_box(subject)
    objects = evaluator.env.scene.objects
    objects = tuple(objects.values()) if isinstance(objects, Mapping) else tuple(objects)
    contents = {}
    for obj in objects:
        if obj is subject or obj is support or state_by_name(obj, "Inside") is None:
            continue
        if canonical_relation(obj, "Inside", support):
            lo, hi = object_aabb(obj)
            contents[obj.name] = Aabb3(tuple(lo), tuple(hi))
    raw_position, raw_orientation = subject.get_position_orientation()
    attempts, plans = [], []
    for link, volume in _private_fillable_regions(support):
        def in_volume(position):
            points = _matching_vector([position], raw_position)
            return bool(link.check_points_in_volume(points).item())

        plan = plan_deterministic_inside_candidates(
            held_extent_m=box.extent_local, fillable_aabb=volume,
            existing_content_aabbs=contents, point_in_fillable_volume=in_volume,
        )
        plans.append(dict(plan.private_record()))
        candidates = _rank_initial_candidates(plan.candidates, support, evaluator,
            placement_rank, placement_count, lambda c: np.asarray(c.position_m)[:2])
        for candidate in candidates:
            checkpoint('INSIDE_CANDIDATE', candidate_index=len(attempts), candidate_count=len(candidates))
            rotation = np.asarray(candidate.orientation_matrix)
            position = np.asarray(candidate.position_m) - rotation @ box.center_local
            orientation = matrix_to_quaternion_xyzw(rotation)
            subject.set_position_orientation(
                position=_matching_vector(position, raw_position),
                orientation=_matching_vector(orientation, raw_orientation), frame="world",
            )
            keep_still = getattr(subject, "keep_still", None)
            if callable(keep_still):
                keep_still()
            _settle(evaluator, control, bindings, phase)
            settled_pos, settled_ori = object_pose(subject)
            samples = []
            for sample_index in range(DEFAULT_STABILITY_WINDOW):
                control.hold()
                bindings._raw_step(evaluator, f"{phase}.inside_readback.{sample_index}")
                samples.append({
                    "inside": canonical_relation(subject, "Inside", support),
                    "linear_speed": float(np.linalg.norm(_object_velocity(subject, "linear"))),
                    "angular_speed": float(np.linalg.norm(_object_velocity(subject, "angular"))),
                })
            accepted = all(s["inside"] and s["linear_speed"] <= DEFAULT_MAX_LINEAR_SPEED_M_S
                           and s["angular_speed"] <= DEFAULT_MAX_ANGULAR_SPEED_RAD_S for s in samples)
            attempt = {"candidate": candidate.candidate_id, "fillable_link": link.name,
                       "canonical_inside": accepted, "accepted": accepted,
                       "stability_samples": samples,
                       "settled_position": settled_pos.tolist(),
                       "settled_orientation_xyzw": settled_ori.tolist()}
            attempts.append(attempt)
            if accepted:
                return {"status": "PASS", "plans": plans, "attempts": attempts, **attempt}
    return {"status": "FAIL", "plans": plans, "attempts": attempts,
            "planning_error": "no settled canonical Inside candidate"}


def _ordered_initial_recipes(recipes, role_to_scope):
    """Order initial placements by support dependency, not by goal truth."""
    unary = [r for r in recipes if r.get("object_role") is None]
    pending = [r for r in recipes if r.get("object_role") is not None]
    result = list(unary)
    while pending:
        subjects = {role_to_scope[r["subject_role"]] for r in pending}
        ready = [r for r in pending if role_to_scope[r["object_role"]] not in subjects]
        if not ready:
            raise ValueError("cyclic initial placement dependencies")
        result.extend(ready)
        pending = [r for r in pending if r not in ready]
    return result


def _initial_process_unfinished(contract, role_object, *, predicate_mode="action20_symbolic"):
    history = contract.get('task_contract', {}).get('process_requirements', {}).get('history_program')
    if history is not None:
        from .l1_history_bindings import validate_history_program
        validate_history_program(history, contract['role_to_scope'])
        # Event bindings and required process milestones start unset at reset,
        # including history programs with only maintained-state constraints.
        return True
    program = contract.get('task_contract', {}).get('process_requirements', {}).get('process_program')
    if program is None:
        return False
    from .l1_am_process_programs import MilestoneProgress, validate_runtime_program
    validate_runtime_program(program, contract['role_to_scope'])
    progress = MilestoneProgress(program, lambda atom: bool(_initial_recipe_truth(atom, role_object, predicate_mode=predicate_mode)))
    progress.observe(progress.read(), 0)
    return not progress.complete


def _initial_recipe_truth(recipe, role_object, *, predicate_mode="action20_symbolic"):
    if recipe.get('placement_policy'):
        from .initial_layout_search import validate_policy
        policy=recipe['placement_policy']
        validate_policy(recipe, {recipe['subject_role'],*policy['support_roles']})
        return any(_initial_recipe_truth({k:v for k,v in dict(recipe,object_role=role).items() if k!='placement_policy'},
            role_object,predicate_mode=predicate_mode) for role in policy['support_roles'])
    aliases = {"ontop": "on_top", "nextto": "next_to", "toggled_off": "not_toggled_on"}
    predicate = aliases.get(recipe["predicate"], recipe["predicate"])
    subject = role_object(recipe["subject_role"])
    reference = (role_object(recipe["object_role"])
                 if recipe.get("object_role") is not None else None)
    return _registered_symbolic_relation_truth(subject, predicate, reference, predicate_mode=predicate_mode)


def _position_counterbalance_readback(recipes, role_to_scope, role_object, evaluator):
    groups = {}
    for recipe in recipes:
        if "placement_rank" in recipe:
            groups.setdefault(role_to_scope[recipe["object_role"]], []).append(recipe)
    reports = []
    for group in groups.values():
        group = sorted(group, key=lambda recipe: recipe["placement_rank"])
        lo, hi = object_aabb(role_object(group[0]["object_role"]))
        right, _ = _screen_axes_xy(
            observer_xy=object_pose(evaluator.env.robots[0])[0][:2],
            target_center_xy=(lo[:2] + hi[:2]) / 2,
        )
        coordinates = [
            float(np.dot(right, sum(object_aabb(role_object(recipe["subject_role"])))[:2] / 2))
            for recipe in group
        ]
        passed = all(a < b for a, b in zip(coordinates, coordinates[1:]))
        reports.append({"screen_right_coordinates": coordinates, "passed": passed})
        if not passed:
            raise RuntimeError(
                "position counterbalance ordering did not survive restoration/construction"
            )
    return reports


def _task_spec_id(task: RegisteredL0Task) -> str:
    record = getattr(task, "record", {})
    if isinstance(record, Mapping):
        value = record.get("task_spec_id") or record.get("candidate_id")
        if value is not None:
            return str(value)
    return task.candidate_id


def _physical_contract_payload(task: RegisteredL0Task, contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the pair-invariant portion of a registered task contract."""

    physical_fields = (
        "role_to_scope",
        "scope_bindings",
        "initial_state_recipe",
        "material_rules",
        "role_orientations_xyzw",
        "camera_relative_orientations",
        "empty_contents_roles",
        "final_initial_readback_required",
    )
    physical_contract = {
        key: contract.get(key) for key in physical_fields if key in contract
    }
    record = getattr(task, "record", {})
    construction = record.get("construction", {}) if isinstance(record, Mapping) else {}
    profile = record.get("profile", {}) if isinstance(record, Mapping) else {}
    return {
        "scene_model": task.scene_model,
        "rooms": list(record.get("rooms", ())) if isinstance(record, Mapping) else [],
        "robot_start": construction.get("robot_start") if isinstance(construction, Mapping) else None,
        "initial_relation_recipe": (
            construction.get("initial_relation_recipe")
            if isinstance(construction, Mapping) else None
        ),
        "canonical_goal_leaves": profile.get("canonical_goal_leaves", ())
        if isinstance(profile, Mapping) else (),
        "contract": physical_contract,
    }


def _physical_contract_digest(task: RegisteredL0Task, contract: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _physical_contract_payload(task, contract),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bound_task_objects(evaluator: Any, contract: Mapping[str, Any]) -> tuple[Any, ...]:
    """Resolve every authored object binding used to protect render-only rules."""

    scopes = contract.get("scope_bindings")
    if not isinstance(scopes, Mapping):
        raise ValueError("registered template contract has no scope bindings")
    result = []
    seen: set[int] = set()
    for binding in scopes.values():
        if not isinstance(binding, Mapping):
            raise ValueError("registered template contract scope binding is malformed")
        # The robot is not an authored task object and is not a valid visual-rule target.
        if binding.get("binding_kind") == "RUNTIME_ROBOT":
            continue
        actual_name = binding.get("actual_name")
        if not isinstance(actual_name, str) or not actual_name:
            raise ValueError("registered template contract object name is missing")
        obj = _live_object(evaluator.env.scene, actual_name)
        if id(obj) not in seen:
            seen.add(id(obj))
            result.append(obj)
    return tuple(result)


def _apply_visual_rules(
    evaluator: Any,
    rules: Any,
    protected_objects: tuple[Any, ...],
) -> tuple[Any, ...]:
    if not rules:
        return ()
    from behavior_hl.embodied.evaluator_bridge.construction_visual_transform import (
        apply_visual_rules,
    )

    changes = apply_visual_rules(evaluator, rules, protected_objects)
    if changes is None:
        raise ValueError("visual rule adapter returned no restore changes")
    return tuple(changes)


def _apply_registered_render_overrides(
    *,
    evaluator: Any,
    contract: Mapping[str, Any],
    role_object: Callable[[Any], Any],
    apply_visual_transforms: bool,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Apply material and visual overrides through the one shared rollback stack."""

    material_changes = tuple(
        _apply_material_rules(evaluator.env.scene, contract.get("material_rules", []), role_object)
    )
    # Publish numeric/VRay changes before the visual adapter runs so an adapter
    # failure cannot strand the material half of the transaction.
    evaluator._registered_material_reset = material_changes
    visual_changes: tuple[Any, ...] = ()
    visual_rules = contract.get("visual_rules", [])
    if apply_visual_transforms and visual_rules:
        try:
            visual_changes = _apply_visual_rules(
                evaluator,
                visual_rules,
                _bound_task_objects(evaluator, contract),
            )
        except Exception as exc:
            try:
                restore_material_changes(material_changes)
            except Exception as restore_error:
                evaluator._registered_material_reset = ()
                raise RegisteredL0ConstructionError(
                    "visual rule rollback failed",
                    {
                        "status": "FAIL",
                        "construction_pass": False,
                        "failure_class": "RESTORE_ERROR",
                        "resident_session_invalidated": True,
                        "error": f"{type(restore_error).__name__}: {restore_error}",
                    },
                ) from exc
            evaluator._registered_material_reset = ()
            raise
        evaluator._registered_material_reset = material_changes + visual_changes
    return material_changes, visual_changes


def _restore_registered_render_overrides(evaluator: Any) -> Mapping[str, Any]:
    pending = tuple(getattr(evaluator, "_registered_material_reset", ()))
    if not pending:
        evaluator._registered_material_reset = ()
        return {
            "scope": "VRAY_ONLY",
            "change_count": 0,
            "numeric_change_count": 0,
            "vray_transaction_count": 0,
            "source_readback_passed": True,
            "input_readback_passed": True,
            "session_property_readback_passed": True,
        }
    try:
        result = restore_material_changes(pending)
    except Exception as exc:
        evaluator._registered_material_reset = ()
        report = {
            "status": "FAIL",
            "construction_pass": False,
            "failure_class": "RESTORE_ERROR",
            "resident_session_invalidated": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
        raise RegisteredL0ConstructionError(
            "registered render override restoration failed", report
        ) from exc
    evaluator._registered_material_reset = ()
    return result


def _scene_state_snapshot(scene: Any) -> Any:
    # Keep the native in-memory contract paired with load_state(serialized=False).
    # Scene.save() JSON-encodes tensors as lists; that is a file format, not a
    # native restore payload. Never round-trip the live snapshot through JSON.
    state = scene.dump_state(serialized=False)
    if not isinstance(state, Mapping):
        raise ValueError("Scene.dump_state() did not return native state")
    return deepcopy(state)


def capture_l1_prepared_facts(evaluator, task, *, predicate_mode="action20_symbolic"):
    """Private branch facts freshly checked after robot preparation; no mutation."""
    contract = task.template_contract()
    branch = contract.get('task_contract')
    if branch is None:
        return None
    from .l1_initial_branch import validate_initial_branch
    validate_initial_branch(branch, contract['role_to_scope'])
    role_object = lambda role: _role_object_for_contract(evaluator, contract, role)
    facts = [{'assertion': dict(a), 'satisfied': bool(_initial_recipe_truth(a, role_object, predicate_mode=predicate_mode))}
             for a in branch['initial_assertions']]
    contents = read_contents_assertions(evaluator.env.scene, branch.get('initial_contents_assertions', []),
                                       role_object, contract['role_to_scope']) if branch.get('initial_contents_assertions') else []
    if not all(f['satisfied'] for f in [*facts, *contents]) or not validate_registered_initial_semantics(evaluator=evaluator, task=task, predicate_mode=predicate_mode):
        raise ValueError('L1 prepared initial branch is not valid')
    return dict(configuration_id=task.candidate_id, world_branch_id=branch['world_branch_id'],
                pair_member=branch['pair_member'], initial_facts=facts, initial_contents_facts=contents,
                axis_review_status='PENDING_REVIEW', scope='PRIVATE_EPISODE_INITIAL_FACTS')


def _scene_state_digest(state: Any) -> str:
    # Audit conversion must not mutate or replace the typed restore snapshot.
    normalized = json.dumps(json_safe(state), sort_keys=True, allow_nan=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_scene_state(scene: Any, state: Any) -> None:
    scene.load_state(deepcopy(state), serialized=False)
    # No physics step: cached derived predicates must refer to restored poses.
    objects = getattr(scene, "objects", ())
    objects = objects.values() if isinstance(objects, Mapping) else objects
    for obj in objects:
        for object_state in getattr(obj, 'states', {}).values():
            object_state.clear_cache()


def _refresh_restored_scene(evaluator, control, bindings):
    # OG Simulator.load_state documents that contact-dependent predicates are
    # inaccurate until one simulator update. This is initial-state work, before
    # the Oracle deadline and its zero-physics public-action accounting.
    control.hold()
    bindings._raw_step(evaluator, 'registered_initial_state.paired_restore_contact_refresh')


def _apply_material_rules(scene, rules, role_object):
    """Set explicit render-only roughness, rejecting shared material side effects."""
    if not rules:
        return []
    objects = scene.objects.values() if isinstance(scene.objects, Mapping) else scene.objects
    objects = tuple(objects)
    changes, seen, vray_plans = [], set(), []
    allowed = {"reflection_roughness_constant", "reflection_roughness_texture_influence"}
    for rule in rules:
        subject = role_object(rule['subject_role'])
        inputs = rule['inputs']
        if not isinstance(inputs, Mapping) or not inputs or not set(inputs) <= allowed:
            raise ValueError('unsupported render-only material inputs')
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not np.isfinite(v) or not 0 <= v <= 1 for v in inputs.values()):
            raise ValueError('roughness values must be finite numbers in [0, 1]')
        materials = tuple(getattr(subject, 'materials', ()))
        if not materials:
            raise ValueError('material slots unavailable')
        for material in sorted(materials, key=lambda m: m.prim_path):
            if material.prim_path in seen:
                raise ValueError('duplicate material rule')
            seen.add(material.prim_path)
            if any(other is not subject and any(m.prim_path == material.prim_path
                   for m in getattr(other, 'materials', ())) for other in objects):
                raise ValueError('material is shared with another object; isolated binding required')
            if is_vray_material(material):
                validate_vray_material_rules(material, inputs)
                vray_plans.append((material, inputs))
                continue
            if not set(inputs) <= (material.shader_input_names | material.shader_default_input_names):
                supported = material.shader_input_names | material.shader_default_input_names
                related = sorted(k for k in supported if 'rough' in k or 'gloss' in k)
                raise ValueError(f'shader does not support requested roughness inputs: '
                                 f'{type(material).__name__} {material.prim_path}; '
                                 f'missing={sorted(set(inputs) - supported)}; available={related}')
            for key, value in sorted(inputs.items()):
                previous = material.get_input(key)
                if not isinstance(previous, (int, float)) or not np.isfinite(previous):
                    raise ValueError('material baseline is not restorable')
                changes.append((material, key, previous, float(value)))
    try:
        for material, key, previous, value in changes:
            material.set_input(key, value)
            if not np.isclose(material.get_input(key), value):
                raise RuntimeError('material readback mismatch')
        for material, inputs in vray_plans:
            changes.extend(apply_vray_material_rules(material, inputs))
    except Exception:
        restore_material_changes(changes)
        raise
    return changes


def construct_registered_l0_initial_state(
    *,
    evaluator: Any,
    control: Any,
    bindings: Any,
    task: RegisteredL0Task,
    apply_visual_transforms: bool = True,
    predicate_mode: str = "official",
) -> Mapping[str, Any]:
    """Materialize one registered initial-state recipe from live geometry."""

    if predicate_mode not in {"official", "action20_symbolic"}:
        raise ValueError("unknown initial relation mode")
    contract = task.template_contract()
    if getattr(task, 'record', {}).get('construction', {}).get('frozen_initial_state') is not None:
        try:
            return verify_registered_frozen_initial_state(
                evaluator=evaluator,
                task=task,
                apply_visual_transforms=apply_visual_transforms,
                predicate_mode=predicate_mode,
            )
        except RegisteredL0ConstructionError:
            raise
        except Exception as exc:
            failure_report = dict(status='FAIL', construction_pass=False, failure_class='RESTORE_ERROR',
                                   error=f'{type(exc).__name__}: {exc}')
            if getattr(exc, 'session_invalidated', False):
                failure_report['material_restore_error'] = failure_report['error']
                failure_report['resident_session_invalidated'] = True
            raise RegisteredL0ConstructionError(
                'frozen initial state restoration failed', failure_report
            ) from exc
    role_to_scope = contract.get("role_to_scope")
    scopes = contract.get("scope_bindings")
    recipes = contract.get("initial_state_recipe")
    if not isinstance(role_to_scope, Mapping) or not isinstance(scopes, Mapping):
        raise ValueError("registered template contract has no role bindings")
    if not isinstance(recipes, list) or not recipes:
        raise ValueError("registered template contract has no initial recipe")
    from .receiving_support_construction import validate_receiving_support
    for recipe in recipes:
        validate_receiving_support(recipe, role_to_scope)
        if recipe.get('receiving_support') and predicate_mode != 'official':
            raise ValueError('receiving support construction requires Physical official relations')

    def role_object(role: Any) -> Any:
        scope = role_to_scope.get(role)
        binding = scopes.get(scope)
        if not isinstance(scope, str) or not isinstance(binding, Mapping):
            raise ValueError(f"registered construction role is unbound: {role}")
        return _live_object(evaluator.env.scene, str(binding["actual_name"]))

    report: dict[str, Any] = {
        "schema_version": LIVE_CONSTRUCTION_VERSION,
        "candidate_id": task.candidate_id,
        "task_name": task.task_name,
        "status": "FAIL",
        "construction_pass": False,
        "predicate_mode": predicate_mode,
        "execution_fidelity": "kinematic_research",
        "official_challenge_eligible": False,
        "direct_state_modification": True,
        "settle_steps_per_candidate": LIVE_CONSTRUCTION_SETTLE_STEPS,
        "relations": [],
        "calibration_probe": contract.get('calibration_probe', {}),
    }
    try:
        MARKER_LABEL_COUNTERBALANCE.set(None)
        # USD material edits are not guaranteed to be part of a physics reset.
        # Restore the previous task's baseline before applying this task's rules.
        report['previous_material_restore'] = _restore_registered_render_overrides(evaluator)
        orientations = dict(contract.get('role_orientations_xyzw', {}))
        camera_rules = contract.get('camera_relative_orientations', {})
        if set(orientations) & set(camera_rules):
            raise ValueError('conflicting world and camera-relative orientations')
        for role, rule in camera_rules.items():
            placements = [r for r in recipes if r.get('subject_role') == role and r.get('predicate') == 'ontop']
            if len(placements) != 1:
                raise ValueError('camera-relative orientation requires one OnTop construction')
            lo, hi = object_aabb(role_object(placements[0]['object_role']))
            orientations[role] = orientation_toward_camera(rule['reference_orientation_xyzw'],
                                                          (lo + hi) / 2, head_camera_position(evaluator)).tolist()
        for role, orientation in orientations.items():
            if role not in role_to_scope or not any(r.get('subject_role') == role
                    and r.get('predicate') == 'ontop' for r in recipes):
                raise ValueError('orientation override requires an OnTop construction role')
            quaternion = np.asarray(orientation, dtype=float)
            if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion), 1):
                raise ValueError('orientation must be a finite unit xyzw quaternion')
            subject = role_object(role)
            position, current_orientation = subject.get_position_orientation()
            subject.set_position_orientation(position=position,
                orientation=_matching_vector(quaternion, current_orientation), frame='world')
        completed_recipes = []
        layout_done=set()
        ordered_recipes = _ordered_initial_recipes(recipes, role_to_scope)
        default_groups = {}
        if contract.get('task_contract', {}).get('axis') in {'A', 'M'} or contract.get('physical_template'):
            from .initial_layout_search import same_support_groups
            for group in same_support_groups(ordered_recipes):
                for item in group:
                    default_groups[id(item)] = group
        for index, recipe in enumerate(ordered_recipes):
            if recipe.get('placement_policy') or id(recipe) in default_groups:
                from .initial_layout_search import validate_policy, source_candidates, free_region_candidates, bounded_joint_search
                policy=recipe.get('placement_policy')
                if policy:
                    key=tuple(policy['support_roles'])
                    group=[r for r in recipes if r.get('placement_policy',{}).get('support_roles')==list(key)]
                    group_key=('explicit', key)
                else:
                    key=(recipe['object_role'],)
                    group=default_groups[id(recipe)]
                    group_key=('same_support', id(group[0]))
                if group_key in layout_done:continue
                for r in group:validate_policy(r,role_to_scope)
                group_by_role={r['subject_role']:r for r in group}
                if len(group_by_role)!=len(group):raise ValueError('duplicate joint layout subject')
                def restore_layout(state):
                    _load_scene_state(evaluator.env.scene,state)
                    _refresh_restored_scene(evaluator,control,bindings)
                    if not all(_initial_recipe_truth(r,role_object,predicate_mode=predicate_mode) for r in completed_recipes):
                        raise RuntimeError('layout restoration lost earlier initial relations')
                def layout_candidates(role,support_role):
                    obj=role_object(role);surface=role_object(support_role)
                    from .donor_layout_hints import donor_candidate
                    donor=donor_candidate(obj,surface,role,contract)
                    if donor is not None:
                        return (donor,)  # Frozen layout; no whole-table resampling.
                    objects=evaluator.env.scene.objects
                    objects=tuple(objects.values()) if isinstance(objects,Mapping) else tuple(objects)
                    item=group_by_role[role]
                    hints=source_candidates(obj,surface,support_role,item.get('placement_policy',{}).get('sources',[]))
                    defaults=plan_default_on_top_candidates(obj,surface,scene_objects=()).candidates
                    candidates=(*hints,*free_region_candidates(obj,surface,objects),*defaults)
                    return _rank_initial_candidates(candidates,surface,evaluator,
                        item.get('placement_rank'),item.get('placement_count'),
                        lambda candidate:_candidate_center_xy(obj,candidate))
                layout_report=bounded_joint_search(list(group_by_role),key,
                    candidates=layout_candidates,
                    attempt=lambda role,support_role,candidate:_construct_on_top(evaluator=evaluator,control=control,bindings=bindings,
                        subject=role_object(role),support=role_object(support_role),phase=f'registered_l0.{task.candidate_id}.joint',
                        candidate_override=(candidate,),predicate_mode=predicate_mode),
                    snapshot=lambda:_scene_state_snapshot(evaluator.env.scene),restore=restore_layout,
                    verify=lambda:all(_initial_recipe_truth(r,role_object,predicate_mode=predicate_mode) for r in [*completed_recipes,*group]),
                    max_attempts=min(r.get('placement_policy',{}).get('max_attempts',96) for r in group),
                    max_seconds=min(r.get('placement_policy',{}).get('max_seconds',180) for r in group))
                layout_report['policy_origin']='EXPLICIT_SUPPORT_ALTERNATIVES' if policy else 'DEFAULT_SAME_SUPPORT'
                report.setdefault('joint_layouts',[]).append(layout_report)
                if layout_report['status']!='PASS':raise RuntimeError('bounded joint initial layout failed')
                layout_done.add(group_key);completed_recipes.extend(group)
                for r in group:
                    report['relations'].append(dict(predicate='ontop',subject_name=role_object(r['subject_role']).name,
                        object_name=role_object(layout_report['support_role']).name,status='PASS',joint_layout=True))
                continue
            if not isinstance(recipe, Mapping) or recipe.get("desired") is not True:
                raise ValueError("only positive registered initial relations are supported")
            checkpoint('INITIAL_RELATION', relation_index=index, recipe=dict(recipe))
            predicate = str(recipe.get("predicate", "")).casefold()
            subject = role_object(recipe.get("subject_role"))
            relation_report: dict[str, Any] = {
                "index": index,
                "predicate": predicate,
                "subject_name": str(getattr(subject, "name", "")),
            }
            if predicate in {"ontop", "inside"}:
                support = role_object(recipe.get("object_role"))
                relation_report["object_name"] = str(
                    getattr(support, "name", "")
                )
                relation_report.update(
                    (_construct_on_top if predicate == "ontop" else _construct_inside)(
                        evaluator=evaluator,
                        control=control,
                        bindings=bindings,
                        subject=subject,
                        support=support,
                        phase=(
                            f"registered_l0.{task.candidate_id}.relation.{index:02d}"
                        ),
                        **({"predicate_mode": predicate_mode} if predicate == "ontop" else {}),
                        **({'receiving_support': recipe['receiving_support']} if recipe.get('receiving_support') else {}),
                        **({"placement_rank": recipe["placement_rank"],
                            "placement_count": recipe["placement_count"]}
                           if "placement_rank" in recipe else {}),
                    )
                )
            elif predicate in {'under', 'next_to', 'nextto'}:
                from .registered_initial_relation_materializer import materialize_initial_relation, InitialRelationRestoreError
                import omnigibson as og
                import torch
                reference = role_object(recipe.get('object_role'))
                def restore_trial(state):
                    _load_scene_state(evaluator.env.scene, state)
                    _refresh_restored_scene(evaluator, control, bindings)
                try:
                    relation_report.update(materialize_initial_relation(subject=subject, target=reference, predicate=predicate,
                        scene=evaluator.env.scene, robot=evaluator.env.robots[0], omnigibson=og, tensor_module=torch,
                        settle=lambda: _settle(evaluator, control, bindings, f'registered_l0.{predicate}'),
                        snapshot=lambda: _scene_state_snapshot(evaluator.env.scene), restore=restore_trial,
                        prior_truth=lambda: [_initial_recipe_truth(item, role_object) for item in completed_recipes]))
                except InitialRelationRestoreError as exc:
                    relation_report.update(exc.report)
                    report['relations'].append(relation_report)
                    raise
            elif predicate in {"toggled_on", "toggled_off"}:
                state = state_by_name(subject, "ToggledOn")
                setter = getattr(state, "set_value", None)
                getter = getattr(state, "get_value", None)
                if not callable(setter) or not callable(getter):
                    raise RuntimeError("canonical ToggledOn state is unavailable")
                desired = predicate == "toggled_on"
                setter(desired)
                actual = bool(getter())
                relation_report.update(
                    {
                        "status": "PASS" if actual == desired else "FAIL",
                        f"canonical_{predicate}": actual == desired,
                    }
                )
            elif predicate in {"open", "closed"}:
                state = state_by_name(subject, "Open")
                setter = getattr(state, "set_value", None)
                getter = getattr(state, "get_value", None)
                if not callable(setter) or not callable(getter):
                    raise RuntimeError("canonical Open state is unavailable")
                desired = predicate == "open"
                # Official Open defaults to a random near-threshold joint pose.
                # Use its full-endpoint initial-state API, then require settling.
                checkpoint('INITIAL_OPEN_SET', subject=getattr(subject, 'name', None), desired=desired)
                setter_result = setter(desired, fully=True)
                _settle(evaluator, control, bindings, f"registered_l0.{predicate}")
                relation_report.update(
                    {
                        "status": "PASS" if setter_result and bool(getter()) == desired else "FAIL",
                        f"canonical_{predicate}": bool(getter()) == desired,
                        "setter_success": bool(setter_result),
                        "fully": True,
                    }
                )
            else:
                raise ValueError(
                    f"unsupported registered initial predicate: {predicate}"
                )
            report["relations"].append(relation_report)
            checkpoint('INITIAL_RELATION_COMPLETE', relation_index=index,
                       predicate=predicate, status=relation_report['status'])
            if relation_report["status"] != "PASS":
                raise RuntimeError(
                    f"registered initial {predicate} construction failed"
                )
            completed_recipes.append(recipe)
        if contract.get('physical_template'):
            from .physical_template_construction import construct_fixtures
            report['physical_fixture_construction']=construct_fixtures(evaluator,control,bindings,task,role_object)
        if contract.get("final_initial_readback_required"):
            report["final_initial_readback"] = [
                {"recipe": recipe, "satisfied": _initial_recipe_truth(recipe, role_object, predicate_mode=predicate_mode)}
                for recipe in recipes
            ]
            if not all(r["satisfied"] for r in report["final_initial_readback"]):
                raise RuntimeError("initial relations changed during shared construction")
            report['position_counterbalance_readback'] = _position_counterbalance_readback(
                recipes, role_to_scope, role_object, evaluator
            )
        receiver_recipes = [r for r in recipes if r.get('receiving_support')]
        if receiver_recipes:
            from .receiving_support_construction import receiving_surface_readback
            _settle(evaluator, control, bindings, 'registered_initial_receiver.final')
            report['receiving_support_readback'] = [
                dict(role=r['subject_role'], **receiving_surface_readback(
                    evaluator, role_object(r['subject_role']), role_object(r['object_role']),
                    role_object(r['receiving_support']['placement_subject_role'])))
                for r in receiver_recipes
            ]
            if not all(row['passed'] for row in report['receiving_support_readback']):
                raise RuntimeError('initial receiving support is not usable by Physical OnTop')
            if not all(_initial_recipe_truth(r, role_object, predicate_mode=predicate_mode) for r in recipes):
                raise RuntimeError('initial relations changed during receiver readback')
        calibration_probe = contract.get('calibration_probe')
        task_record = getattr(task, 'record', {})
        construction_record = (
            task_record.get('construction', {})
            if isinstance(task_record, Mapping) else {}
        )
        report['calibration_only'] = bool(
            isinstance(construction_record, Mapping)
            and construction_record.get('calibration_only') is True
        )
        calibration_pose_probe = bool(
            isinstance(construction_record, Mapping)
            and construction_record.get('calibration_only') is True
            and isinstance(calibration_probe, Mapping)
            and str(calibration_probe.get('kind', '')).casefold() == 'pose'
        )
        report['orientation_readback'] = _initial_orientation_readback(orientations, role_object)
        settled_orientations = {
            row['role']: np.asarray(row['settled_orientation_xyzw'], dtype=float)
            for row in report['orientation_readback']
        }
        # Ordinary Oracle construction is admitted by its canonical and visual
        # semantics, not a generic angular tolerance or a calibration window.
        if calibration_probe and report['calibration_only']:
            pose_roles = tuple(orientations)
            stability_reference = (
                settled_orientations if calibration_pose_probe else orientations
            )
            pose_relations = {
                role: tuple(
                    recipe for recipe in recipes
                    if recipe.get('subject_role') == role
                    and str(recipe.get('predicate', '')).casefold() in {'ontop', 'inside'}
                )
                for role in pose_roles
            }
            pose_readback = {
                'status': 'PENDING',
                'passed': False,
                'required_samples': DEFAULT_STABILITY_WINDOW,
                'roles': list(pose_roles),
                'thresholds': {
                    'max_linear_speed_m_s': DEFAULT_MAX_LINEAR_SPEED_M_S,
                    'max_angular_speed_rad_s': DEFAULT_MAX_ANGULAR_SPEED_RAD_S,
                    'max_orientation_error_rad': DEFAULT_PROTECTED_ROTATION_RAD,
                },
                'orientation_reference': (
                    'NATURAL_SETTLED' if calibration_pose_probe else 'REQUESTED'
                ),
                'orientation_reference_xyzw': {
                    role: np.asarray(stability_reference[role], dtype=float).tolist()
                    for role in pose_roles
                },
                'samples': [],
                'failure_reasons': [],
            }
            report['pose_stability_readback'] = pose_readback
            if not pose_roles:
                pose_readback.update(status='SKIPPED_NO_POSE_ROLES', passed=True)
            else:
                for sample_index in range(DEFAULT_STABILITY_WINDOW):
                    control.hold()
                    bindings._raw_step(
                        evaluator,
                        f'registered_l0.{task.candidate_id}.calibration_probe.{sample_index:03d}',
                    )
                    sample = {'index': sample_index, 'roles': {}, 'failure_reasons': []}
                    for role in pose_roles:
                        subject = role_object(role)
                        position, actual_orientation = object_pose(subject)
                        position = np.asarray(position, dtype=float)
                        actual_orientation = np.asarray(actual_orientation, dtype=float)
                        linear_speed = float(np.linalg.norm(_object_velocity(subject, 'linear')))
                        angular_speed = float(np.linalg.norm(_object_velocity(subject, 'angular')))
                        try:
                            angular_error = float(
                                quaternion_angle_rad(
                                    actual_orientation, stability_reference[role]
                                )
                            )
                        except Exception:
                            angular_error = None
                        role_report = {
                            'position': position.tolist(),
                            'orientation_xyzw': actual_orientation.tolist(),
                            'angular_error_rad': angular_error,
                            'linear_speed_m_s': linear_speed,
                            'angular_speed_rad_s': angular_speed,
                            'canonical_initial_relations': [],
                            'failure_reasons': [],
                        }
                        if not np.isfinite(position).all() or not np.isfinite(actual_orientation).all():
                            role_report['failure_reasons'].append('non_finite_pose')
                        if (not np.isfinite(linear_speed)
                                or linear_speed > DEFAULT_MAX_LINEAR_SPEED_M_S):
                            role_report['failure_reasons'].append('linear_velocity_not_stable')
                        if (not np.isfinite(angular_speed)
                                or angular_speed > DEFAULT_MAX_ANGULAR_SPEED_RAD_S):
                            role_report['failure_reasons'].append('angular_velocity_not_stable')
                        if (angular_error is None or not np.isfinite(angular_error)
                                or angular_error > DEFAULT_PROTECTED_ROTATION_RAD):
                            role_report['failure_reasons'].append('orientation_not_stable')
                        for recipe in pose_relations[role]:
                            predicate = str(recipe['predicate']).casefold()
                            relation_name = {'ontop': 'OnTop', 'inside': 'Inside'}[predicate]
                            support = role_object(recipe['object_role'])
                            try:
                                satisfied = bool(canonical_relation(subject, relation_name, support))
                            except Exception as relation_error:
                                satisfied = False
                                role_report['canonical_initial_relations'].append({
                                    'predicate': predicate,
                                    'object_role': recipe['object_role'],
                                    'satisfied': False,
                                    'error': f'{type(relation_error).__name__}: {relation_error}',
                                })
                            else:
                                role_report['canonical_initial_relations'].append({
                                    'predicate': predicate,
                                    'object_role': recipe['object_role'],
                                    'satisfied': satisfied,
                                })
                            if not satisfied:
                                role_report['failure_reasons'].append(
                                    f'canonical_initial_relation_not_stable:{predicate}'
                                )
                        sample['roles'][role] = role_report
                        sample['failure_reasons'].extend(
                            f'{role}:{reason}' for reason in role_report['failure_reasons']
                        )
                    pose_readback['samples'].append(sample)
                    pose_readback['failure_reasons'].extend(sample['failure_reasons'])
                pose_readback['passed'] = not pose_readback['failure_reasons']
                pose_readback['status'] = 'PASS' if pose_readback['passed'] else 'FAIL'
                if not pose_readback['passed']:
                    raise RuntimeError(
                        'calibration probe pose stability failed: '
                        + '; '.join(pose_readback['failure_reasons'])
                    )
        report['camera_pose_readback'] = camera_pose_readback(camera_rules, head_camera_position(evaluator),
            lambda role: object_pose(role_object(role))) if camera_rules else []
        if not all(r['passed'] for r in report['camera_pose_readback']):
            raise RuntimeError('camera-relative semantic pose failed after placement')
        report['empty_contents_readback'] = read_empty_contents(
            evaluator.env.scene, contract.get('empty_contents_roles', []), role_object)
        if not all(r['passed'] for r in report['empty_contents_readback']):
            raise RuntimeError('requested empty container has live contents')
        report['initial_contents_facts'] = read_contents_assertions(evaluator.env.scene,
            contract.get('task_contract', {}).get('initial_contents_assertions', []), role_object, contract['role_to_scope'])
        if not all(r['satisfied'] for r in report['initial_contents_facts']):
            raise RuntimeError('initial contents branch differs from its declared truth')
        material_rules = contract.get('material_rules', [])
        material_probe = bool(calibration_probe and material_rules)
        material_state_before = None
        if material_probe:
            material_state_before = json.dumps(
                json.loads(evaluator.env.scene.save()), sort_keys=True, allow_nan=False
            )
        material_changes, visual_changes = _apply_registered_render_overrides(
            evaluator=evaluator,
            contract=contract,
            role_object=role_object,
            apply_visual_transforms=apply_visual_transforms,
        )
        if material_probe:
            material_state_after = json.dumps(
                json.loads(evaluator.env.scene.save()), sort_keys=True, allow_nan=False
            )
            parity = material_state_before == material_state_after
            report['material_physics_parity'] = {
                'status': 'PASS' if parity else 'FAIL',
                'passed': parity,
                'physics_parity': parity,
                'before_sha256': hashlib.sha256(material_state_before.encode('utf-8')).hexdigest(),
                'after_sha256': hashlib.sha256(material_state_after.encode('utf-8')).hexdigest(),
                'encoding': 'json.dumps(json.loads(scene.save()), sort_keys=True, allow_nan=False)',
                'scope': 'serialized scene state only; visual shader inputs only',
                'shader_inputs_only': True,
                'dynamics_counterfactual_claim': False,
            }
            if not parity:
                raise RuntimeError('material physics parity changed in serialized scene state')
        report['material_changes'] = [{'material': m.prim_path, 'input': key, 'previous': old, 'value': new}
                                      for m, key, old, new in material_changes]
        report['visual_rules'] = contract.get('visual_rules', [])
        report['visual_changes_count'] = len(visual_changes)
        policy = contract.get('marker_label_counterbalance')
        if policy is not None and (not isinstance(policy, Mapping) or set(policy) != {'seed', 'rotation'}
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in policy.values())):
            raise ValueError('invalid marker counterbalance policy')
        MARKER_LABEL_COUNTERBALANCE.set(policy)
        report['marker_label_counterbalance'] = policy
    except RegisteredL0ConstructionError:
        MARKER_LABEL_COUNTERBALANCE.set(None)
        raise
    except Exception as exc:
        MARKER_LABEL_COUNTERBALANCE.set(None)
        if getattr(exc,'layout_report',None) is not None:
            report.setdefault('joint_layouts',[]).append(exc.layout_report)
        try:
            _restore_registered_render_overrides(evaluator)
        except RegisteredL0ConstructionError as restore_error:
            raise restore_error from exc
        except Exception as restore_error:
            report['material_restore_error'] = f'{type(restore_error).__name__}: {restore_error}'
            report['resident_session_invalidated'] = True
        if getattr(exc, 'session_invalidated', False):
            report['material_restore_error'] = f'{type(exc).__name__}: {exc}'
            report['resident_session_invalidated'] = True
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise RegisteredL0ConstructionError(
            "registered L0 initial construction failed", report
        ) from exc
    report["status"] = "PASS"
    report["construction_pass"] = True
    return report


def capture_registered_initial_state(*, evaluator: Any, task: RegisteredL0Task, predicate_mode="action20_symbolic") -> Mapping[str, Any]:
    """Private construction evidence, captured before any Oracle action."""
    contract = task.template_contract()
    scope = evaluator.env.task.object_scope
    leaves = task.record['profile']['canonical_goal_leaves']
    truth = [_registered_goal_relation_truth(scope, relation_from_grounded_leaf(leaf), predicate_mode=predicate_mode) for leaf in leaves]
    if all(truth) and not _initial_process_unfinished(
            contract, lambda role: _role_object_for_contract(evaluator, contract, role), predicate_mode=predicate_mode):
        raise RuntimeError('cannot freeze an initially satisfied goal')
    poses = {}
    for name, binding in contract['scope_bindings'].items():
        if binding.get('binding_kind') == 'RUNTIME_ROBOT':
            continue
        obj = _live_object(evaluator.env.scene, binding['actual_name'])
        position, orientation = object_pose(obj)
        lo, hi = object_aabb(obj)
        poses[name] = dict(actual_name=binding['actual_name'], position=position.tolist(),
                           orientation=orientation.tolist(), aabb_min=lo.tolist(), aabb_max=hi.tolist())
    camera_capture = {}
    if contract.get('camera_relative_orientations'):
        position, orientation = object_pose(evaluator.env.robots[0])
        camera_capture = dict(camera_pose_capture=dict(head_position=head_camera_position(evaluator).tolist(),
            robot_position=position.tolist(), robot_orientation=orientation.tolist()))
    return json_safe(dict(schema_version='registered-initial-state-capture/v1', task_spec_id=task.candidate_id,
        source_template_sha256=task.record['template']['sha256'],
        source_contract_sha256=task.record['template_contract']['sha256'],
        goal_leaves=leaves, initial_goal_leaf_truth=truth, scope_geometry=poses,
        scene_document=evaluator.env.scene.save(as_dict=True), **camera_capture))


def verify_registered_frozen_initial_state(
    *, evaluator: Any, task: RegisteredL0Task, apply_visual_transforms: bool = True,
    predicate_mode: str = "action20_symbolic",
) -> Mapping[str, Any]:
    """Frozen consumers restore through the normal scene loader; never re-place."""
    MARKER_LABEL_COUNTERBALANCE.set(None)
    frozen = task.record['construction']['frozen_initial_state']
    if frozen.get('schema_version') != 'registered-frozen-initial-state/v1':
        raise ValueError('unsupported frozen initial state')
    contract = task.template_contract()
    expected = frozen['scope_geometry']
    required = {s for s, b in contract['scope_bindings'].items() if b.get('binding_kind') != 'RUNTIME_ROBOT'}
    if set(expected) != required:
        raise ValueError('frozen geometry membership mismatch')
    def role_object(role):
        binding = contract['scope_bindings'][contract['role_to_scope'][role]]
        return _live_object(evaluator.env.scene, binding['actual_name'])
    def drift_report(position, orientation, target, target_orientation):
        position, target = np.asarray(position), np.asarray(target)
        orientation, target_orientation = np.asarray(orientation), np.asarray(target_orientation)
        if any(v.shape != shape or not np.isfinite(v).all() for v, shape in (
                (position, (3,)), (target, (3,)), (orientation, (4,)), (target_orientation, (4,)))):
            raise RuntimeError('invalid frozen pose')
        if min(np.linalg.norm(orientation), np.linalg.norm(target_orientation)) <= 1e-8:
            raise RuntimeError('invalid frozen orientation')
        distance = float(np.linalg.norm(position - target))
        angle = float(quaternion_angle_rad(orientation, target_orientation))
        return dict(position_error_m=distance, orientation_error_rad=angle,
                    large_drift_warning=distance > .40 or angle > np.deg2rad(30))
    pose_drift = {}
    for scope, geometry in expected.items():
        if geometry['actual_name'] != contract['scope_bindings'][scope]['actual_name']:
            raise ValueError('frozen object identity mismatch')
        position, orientation = object_pose(_live_object(evaluator.env.scene, geometry['actual_name']))
        target = np.asarray(geometry['position'], dtype=float)
        target_orientation = np.asarray(geometry['orientation'], dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(position).all():
            raise RuntimeError('invalid frozen object pose')
        if (target_orientation.shape != (4,) or not np.isfinite(target_orientation).all()
                or not np.isclose(np.linalg.norm(target_orientation), 1.) or not np.isfinite(orientation).all()):
            raise RuntimeError('invalid frozen object orientation')
        pose_drift[scope] = drift_report(position, orientation, target, target_orientation)
    if not all(_initial_recipe_truth(r, role_object, predicate_mode=predicate_mode) for r in contract['initial_state_recipe']):
        raise RuntimeError('frozen initial relation readback failed')
    contents = read_empty_contents(evaluator.env.scene, contract.get('empty_contents_roles', []), role_object)
    if not all(r['passed'] for r in contents):
        raise RuntimeError('frozen empty container has live contents')
    initial_contents = read_contents_assertions(evaluator.env.scene,
        contract.get('task_contract', {}).get('initial_contents_assertions', []), role_object, contract['role_to_scope'])
    if not all(r['satisfied'] for r in initial_contents):
        raise RuntimeError('frozen initial contents branch differs from its declared truth')
    camera_rules = contract.get('camera_relative_orientations', {})
    camera_reports = camera_pose_readback(camera_rules, head_camera_position(evaluator),
        lambda role: object_pose(role_object(role))) if camera_rules else []
    if not all(r['passed'] for r in camera_reports):
        raise RuntimeError('frozen camera-relative semantic pose failed')
    scope = evaluator.env.task.object_scope
    initial_goal_true = all(_registered_goal_relation_truth(scope, relation_from_grounded_leaf(leaf), predicate_mode=predicate_mode)
                           for leaf in task.record['profile']['canonical_goal_leaves'])
    if initial_goal_true and not _initial_process_unfinished(contract, role_object, predicate_mode=predicate_mode):
        raise RuntimeError('frozen goal already satisfied')
    robot_position, robot_orientation = object_pose(evaluator.env.robots[0])
    start = task.record['construction']['robot_start']
    robot_drift = drift_report(
        robot_position, robot_orientation, start['position'], start['orientation']
    )
    position_readback = _position_counterbalance_readback(
        contract['initial_state_recipe'], contract['role_to_scope'], role_object, evaluator
    )
    orientation_readback = _initial_orientation_readback(
        contract.get('role_orientations_xyzw', {}), role_object
    )
    # Physical scene restore does not necessarily restore render-only overrides.
    previous_material_restore = _restore_registered_render_overrides(evaluator)
    try:
        material_changes, visual_changes = _apply_registered_render_overrides(
            evaluator=evaluator,
            contract=contract,
            role_object=role_object,
            apply_visual_transforms=apply_visual_transforms,
        )
    except RegisteredL0ConstructionError:
        raise
    except Exception as exc:
        try:
            _restore_registered_render_overrides(evaluator)
        except RegisteredL0ConstructionError as restore_error:
            raise restore_error from exc
        raise
    MARKER_LABEL_COUNTERBALANCE.set(contract.get('marker_label_counterbalance'))
    return dict(status='PASS', construction_pass=True, initialization_mode='FROZEN_RESTORE_READBACK',
                object_placement_executed=False, robot_pose_selection_executed=False,
                task_name=task.task_name, initial_goal_false=not initial_goal_true,
                previous_material_restore=previous_material_restore,
                material_changes=[
                    {'material': m.prim_path, 'input': key, 'previous': old, 'value': new}
                    for m, key, old, new in material_changes
                ],
                visual_rules=contract.get('visual_rules', []),
                visual_changes_count=len(visual_changes),
                empty_contents_readback=contents,
                initial_contents_facts=initial_contents,
                pose_admission_policy='finite_readback_semantics_required_drift_diagnostic_only',
                object_pose_drift=pose_drift,
                robot_pose_drift=robot_drift,
                orientation_readback=orientation_readback,
                position_counterbalance_readback=position_readback)


def validate_registered_initial_semantics(*, evaluator: Any, task: RegisteredL0Task, predicate_mode="official") -> bool:
    """Validate preserved initial relations after a base-only pose adjustment."""

    contract = task.template_contract()

    def role_object(role):
        binding = contract['scope_bindings'][contract['role_to_scope'][role]]
        return _live_object(evaluator.env.scene, binding['actual_name'])

    contents_assertions = contract.get('task_contract', {}).get('initial_contents_assertions', [])
    contents = read_contents_assertions(evaluator.env.scene, contents_assertions,
        role_object, contract['role_to_scope']) if contents_assertions else []
    if not all(row['satisfied'] for row in contents):
        return False
    if not all(
        _initial_recipe_truth(recipe, role_object, predicate_mode=predicate_mode)
        for recipe in contract['initial_state_recipe']
    ):
        return False
    scope = evaluator.env.task.object_scope
    if all(
        _registered_goal_relation_truth(scope, relation_from_grounded_leaf(leaf), predicate_mode=predicate_mode)
        for leaf in task.record['profile']['canonical_goal_leaves']
    ) and not _initial_process_unfinished(contract, role_object, predicate_mode=predicate_mode):
        return False
    try:
        _position_counterbalance_readback(
            contract['initial_state_recipe'], contract['role_to_scope'], role_object, evaluator
        )
    except RuntimeError:
        return False
    return True


def _pair_task_spec_id(task: RegisteredL0Task) -> str | None:
    value = task.template_contract().get("pair_base_task_spec_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("pair_base_task_spec_id must be a non-empty string")
    return value


def _cache_base_task_spec_id(task: RegisteredL0Task, base_task: RegisteredL0Task) -> str:
    pair_id = _pair_task_spec_id(task)
    if pair_id is not None:
        return pair_id
    return _task_spec_id(base_task)


def _role_object_for_contract(evaluator: Any, contract: Mapping[str, Any], role: Any) -> Any:
    role_to_scope = contract.get("role_to_scope")
    scopes = contract.get("scope_bindings")
    if not isinstance(role_to_scope, Mapping) or not isinstance(scopes, Mapping):
        raise ValueError("registered template contract has no role bindings")
    scope = role_to_scope.get(role)
    binding = scopes.get(scope)
    if not isinstance(scope, str) or not isinstance(binding, Mapping):
        raise ValueError(f"registered construction role is unbound: {role}")
    return _live_object(evaluator.env.scene, str(binding["actual_name"]))


def _cache_object_identity(
    evaluator: Any, contract: Mapping[str, Any],
) -> Mapping[str, tuple[str, Any, str | None]]:
    scopes = contract.get("scope_bindings")
    if not isinstance(scopes, Mapping):
        raise ValueError("registered template contract has no scope bindings")
    identities = {}
    for scope, binding in scopes.items():
        if not isinstance(binding, Mapping) or binding.get("binding_kind") == "RUNTIME_ROBOT":
            continue
        actual_name = binding.get("actual_name")
        if not isinstance(actual_name, str) or not actual_name:
            raise ValueError("registered template contract object name is missing")
        obj = _live_object(evaluator.env.scene, actual_name)
        identities[str(scope)] = (actual_name, obj, getattr(obj, "prim_path", None))
    return identities


def _pair_object_diagnostics(evaluator, contract):
    """Read-only pose evidence; missing diagnostics never change admission."""
    objects = {}
    for role in contract.get("role_to_scope", {}):
        row = {}
        objects[str(role)] = row
        try:
            obj = _role_object_for_contract(evaluator, contract, role)
            row.update(name=str(getattr(obj, "name", "")),
                       prim_path=str(getattr(obj, "prim_path", "")))
            for field, reader in (("pose", lambda: object_pose(obj)),
                                  ("aabb", lambda: object_aabb(obj))):
                try:
                    row[field] = json_safe(reader())
                except Exception as exc:
                    row[field + "_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            row["binding_error"] = f"{type(exc).__name__}: {exc}"
    return objects


def _validate_cached_pair_state(
    *,
    evaluator: Any,
    task: RegisteredL0Task,
    entry: Mapping[str, Any],
    diagnostic_report: dict | None = None,
    predicate_mode: str = "action20_symbolic",
) -> Mapping[str, Any]:
    if entry.get("scene") is not evaluator.env.scene:
        raise RuntimeError("paired initial-state scene identity drift")
    contract = task.template_contract()
    scopes = contract.get("scope_bindings")
    identities = entry.get("object_identities")
    if not isinstance(scopes, Mapping) or not isinstance(identities, Mapping):
        raise RuntimeError("paired initial-state identity metadata is malformed")
    for scope, identity in identities.items():
        if not isinstance(identity, tuple) or len(identity) != 3:
            raise RuntimeError("paired initial-state object identity metadata is malformed")
        actual_name, expected_object, expected_prim_path = identity
        binding = scopes.get(scope)
        if not isinstance(binding, Mapping) or binding.get("actual_name") != actual_name:
            raise RuntimeError("paired initial-state object binding drift")
        actual_object = _live_object(evaluator.env.scene, actual_name)
        if actual_object is not expected_object:
            raise RuntimeError(f"paired initial-state object identity drift: {actual_name}")
        if expected_prim_path is not None and getattr(actual_object, "prim_path", None) != expected_prim_path:
            raise RuntimeError(f"paired initial-state object path drift: {actual_name}")

    def role_object(role: Any) -> Any:
        return _role_object_for_contract(evaluator, contract, role)

    recipes = contract.get("initial_state_recipe")
    rows = []
    if diagnostic_report is not None:
        diagnostic_report["canonical_initial_relations"] = rows
    first_error = None
    for index, recipe in enumerate(recipes if isinstance(recipes, list) else ()):
        row = {"recipe_index": index, "recipe": dict(recipe), "satisfied": None}
        rows.append(row)
        try:
            row["satisfied"] = bool(_initial_recipe_truth(recipe, role_object, predicate_mode=predicate_mode))
            row["readback_source"] = (
                "action20_symbolic_geometry"
                if recipe["predicate"] in {"ontop", "on_top", "nextto", "next_to"}
                and predicate_mode == "action20_symbolic"
                else "shared_state_get_value"
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
    if not isinstance(recipes, list) or not all(row["satisfied"] for row in rows):
        failed = [row["recipe"] for row in rows if row["satisfied"] is not True]
        raise RuntimeError("paired initial-state canonical relation readback failed: "
                           + json.dumps(failed, sort_keys=True))
    scope = getattr(evaluator.env.task, "object_scope", None)
    profile = getattr(task, "record", {}).get("profile", {})
    leaves = profile.get("canonical_goal_leaves", ()) if isinstance(profile, Mapping) else ()
    if not isinstance(scope, Mapping) or not isinstance(leaves, (list, tuple)) or not leaves:
        raise RuntimeError("paired initial-state goal readback context is unavailable")
    goal_truth = [
        _registered_goal_relation_truth(scope, relation_from_grounded_leaf(leaf), predicate_mode=predicate_mode)
        for leaf in leaves
    ]
    if goal_truth and all(goal_truth) and not _initial_process_unfinished(contract, role_object, predicate_mode=predicate_mode):
        raise RuntimeError("paired initial-state goal is already satisfied")
    return {
        "canonical_initial_relations": rows,
        "initial_goal_leaf_truth": goal_truth,
    }


def _paired_report_from_cache(
    *,
    evaluator: Any,
    task: RegisteredL0Task,
    base_task: RegisteredL0Task,
    entry: Mapping[str, Any],
    load_state: bool = True,
    refresh_state: Any = None,
    predicate_mode: str = "action20_symbolic",
) -> Mapping[str, Any]:
    contract = task.template_contract()
    restore_report = _restore_registered_render_overrides(evaluator)
    diagnostics = {
        "schema_version": "paired-initial-readback-diagnostics/v1",
        "baseline_objects": entry.get("baseline_object_diagnostics"),
        "before_load_objects": _pair_object_diagnostics(evaluator, contract),
        "load_state_requested": load_state,
        "baseline_task_spec_id": entry.get("base_task_spec_id"),
        "scene_state_sha256": entry.get("scene_state_sha256"),
    }
    phase = "load_state"
    try:
        if load_state:
            _load_scene_state(evaluator.env.scene, entry["state"])
            if predicate_mode == "official":
                phase = 'contact_refresh'
                if not callable(refresh_state):
                    raise TypeError('paired restoration requires a contact refresh callback')
                refresh_state()
        diagnostics["after_load_objects"] = _pair_object_diagnostics(evaluator, contract)
        phase = "canonical_readback"
        readback = _validate_cached_pair_state(
            evaluator=evaluator, task=task, entry=entry, diagnostic_report=diagnostics,
            predicate_mode=predicate_mode,
        )
        role_object = lambda role: _role_object_for_contract(evaluator, contract, role)
        phase = "visual_rules"
        material_changes, visual_changes = _apply_registered_render_overrides(
            evaluator=evaluator,
            contract=contract,
            role_object=role_object,
            apply_visual_transforms=True,
        )
        policy = contract.get("marker_label_counterbalance")
        if policy is not None and (
            not isinstance(policy, Mapping)
            or set(policy) != {"seed", "rotation"}
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in policy.values())
        ):
            raise ValueError("invalid marker counterbalance policy")
        MARKER_LABEL_COUNTERBALANCE.set(policy)
    except RegisteredL0ConstructionError:
        raise
    except Exception as exc:
        try:
            _restore_registered_render_overrides(evaluator)
        except RegisteredL0ConstructionError as restore_error:
            raise restore_error from exc
        rollback_failed = bool(getattr(exc, "session_invalidated", False))
        if phase == "visual_rules" and rollback_failed:
            failure_class = "RESTORE_ERROR"
        elif phase == "visual_rules":
            failure_class = "VISUAL_TRANSFORM_ERROR"
        else:
            failure_class = "PAIRED_INITIAL_STATE_RESTORE_ERROR"
        report = {
            "status": "FAIL",
            "construction_pass": False,
            "failure_class": failure_class,
            "restore_phase": phase,
            "error": f"{type(exc).__name__}: {exc}",
            "paired_readback_diagnostics": diagnostics,
        }
        if phase != "visual_rules" or rollback_failed:
            report["resident_session_invalidated"] = True
        raise RegisteredL0ConstructionError(
            "paired initial state restoration failed", report
        ) from exc
    return {
        "schema_version": LIVE_CONSTRUCTION_VERSION,
        "candidate_id": task.candidate_id,
        "task_name": task.task_name,
        "status": "PASS",
        "construction_pass": True,
        "execution_fidelity": "kinematic_research",
        "official_challenge_eligible": False,
        "direct_state_modification": True,
        "initialization_mode": "PAIRED_INITIAL_STATE_RESTORE",
        "restore_contact_refresh_steps": int(load_state),
        "object_placement_executed": False,
        "robot_pose_selection_executed": False,
        "previous_material_restore": restore_report,
        "paired_readback_diagnostics": diagnostics,
        "relations": list(readback["canonical_initial_relations"]),
        "final_initial_readback": list(readback["canonical_initial_relations"]),
        "initial_goal_leaf_truth": list(readback["initial_goal_leaf_truth"]),
        "material_changes": [
            {"material": m.prim_path, "input": key, "previous": old, "value": new}
            for m, key, old, new in material_changes
        ],
        "visual_rules": contract.get("visual_rules", []),
        "visual_changes_count": len(visual_changes),
        "marker_label_counterbalance": contract.get("marker_label_counterbalance"),
        "calibration_only": bool(
            isinstance(getattr(task, "record", None), Mapping)
            and task.record.get("construction", {}).get("calibration_only") is True
        ),
        "calibration_probe": contract.get("calibration_probe", {}),
        "pose_stability_readback": deepcopy(entry.get("pose_stability_readback")),
        "paired_initial_state": {
            "private": True,
            "cache_hit": True,
            "physical_state_inherited": True,
            "baseline_task_spec_id": entry.get("base_task_spec_id", _task_spec_id(base_task)),
            "physical_contract_sha256": entry["physical_contract_sha256"],
            "scene_state_sha256": entry["scene_state_sha256"],
            "object_placement_executed": False,
            "visual_effect_claim": "NOT_CLAIMED",
        },
    }


def _validate_runtime_initial_relations(evaluator, task, result, predicate_mode):
    """Final admission uses the runtime reader, not a construction-only getter."""
    if predicate_mode == "official" or result.get("construction_pass") is not True:
        return result
    contract = task.template_contract()
    role_object = lambda role: _role_object_for_contract(evaluator, contract, role)
    rows = []
    for recipe in contract["initial_state_recipe"]:
        row = {"recipe": recipe}
        try:
            row["satisfied"] = _initial_recipe_truth(recipe, role_object, predicate_mode=predicate_mode)
            row["error"] = None
        except Exception as exc:
            row.update(satisfied=False, error=f"{type(exc).__name__}: {exc}")
        try:
            row["raw_official_satisfied"] = _initial_recipe_truth(recipe, role_object, predicate_mode="official")
        except Exception as exc:
            row.update(raw_official_satisfied=None, raw_official_error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    report = dict(result, predicate_mode=predicate_mode, runtime_initial_readback=rows,
                  construction_raw_readback_private=result.get("final_initial_readback"),
                  final_initial_readback=rows)
    if not all(row["satisfied"] for row in rows):
        report.update(status="FAIL", construction_pass=False, failure_class="INITIAL_RELATION_MODE_MISMATCH")
        raise RegisteredL0ConstructionError("initial relations do not satisfy the runtime contract", report)
    return report


def make_registered_l0_construction_callback(
    tasks: Mapping[str, RegisteredL0Task],
    *,
    allow_pending_initial_state: bool = False,
    allow_live_oracle_construction: bool = False,
    predicate_mode: str = "official",
) -> Callable[..., Mapping[str, Any]]:
    if predicate_mode not in {"official", "action20_symbolic"}:
        raise ValueError("unknown initial relation mode")
    if allow_pending_initial_state and allow_live_oracle_construction:
        raise ValueError(
            "allow_pending_initial_state and allow_live_oracle_construction are mutually exclusive"
        )
    task_by_name = {task.task_name: task for task in tasks.values()}
    if len(task_by_name) != len(tasks):
        raise ValueError("registered L0 construction task names must be unique")
    for task in task_by_name.values():
        construction = getattr(task, 'record', {}).get('construction', {})
        if construction.get('calibration_only') and (
            allow_live_oracle_construction or not allow_pending_initial_state
        ):
            raise ValueError('calibration specimen cannot enter Oracle')
        if (
            construction.get('initial_state_compilation_required')
            and not construction.get('frozen_initial_state')
            and not (allow_pending_initial_state or allow_live_oracle_construction)
        ):
            raise ValueError('initial state is not frozen; construction compilation required')
        contract = task.template_contract()
        recipes = contract.get("initial_state_recipe")
        roles = contract.get("role_to_scope")
        scopes = contract.get("scope_bindings")
        if not isinstance(recipes, list) or not recipes:
            raise ValueError("registered L0 construction recipe is missing")
        if not isinstance(roles, Mapping) or not isinstance(scopes, Mapping):
            raise ValueError("registered L0 construction role binding is missing")
        allowed_predicates = {"ontop", "inside", "open", "toggled_on", "toggled_off", "closed"}
        frozen = construction.get('frozen_initial_state')
        if isinstance(frozen, Mapping) and frozen.get('schema_version') == 'registered-frozen-initial-state/v1':
            # Frozen geometry uses shared relation readers; no placer is enabled.
            allowed_predicates.update({'under', 'nextto', 'next_to'})
        if any(
            not isinstance(item, Mapping)
            or item.get("desired") is not True
            or str(item.get("predicate", "")).casefold()
            not in allowed_predicates
            or item.get("subject_role") not in roles
            or roles.get(item.get("subject_role")) not in scopes
            or (
                str(item.get("predicate", "")).casefold() in {"ontop", "inside", "under", "nextto", "next_to"}
                and (
                    item.get("object_role") not in roles
                    or roles.get(item.get("object_role")) not in scopes
                )
            )
            for item in recipes
        ):
            raise ValueError("registered L0 construction recipe is unsupported")

    task_by_spec_id: dict[str, RegisteredL0Task] = {}
    for task in task_by_name.values():
        for identifier in (_task_spec_id(task), task.candidate_id, task.task_name):
            previous = task_by_spec_id.get(identifier)
            if previous is not None and previous is not task:
                raise ValueError(f"registered paired task identifier is ambiguous: {identifier}")
            task_by_spec_id[identifier] = task
    pair_base_for_task: dict[str, RegisteredL0Task] = {}
    unavailable_pair_sources: dict[str, str] = {}
    referenced_base_ids: set[str] = set()
    for task in task_by_name.values():
        pair_id = _pair_task_spec_id(task)
        if pair_id is None:
            continue
        self_identifiers = {_task_spec_id(task), task.candidate_id, task.task_name}
        if pair_id in self_identifiers:
            # The compiler marks a selected Base with a self-pair.  It is a
            # base marker, not a recursive dependency.
            continue
        referenced_base_ids.add(pair_id)
        base_task = task_by_spec_id.get(pair_id)
        if base_task is None:
            if allow_live_oracle_construction and not task.record.get('construction', {}).get('frozen_initial_state'):
                unavailable_pair_sources[task.task_name] = pair_id
            # A variant-only catalog is valid: its recipe is the base recipe
            # and its own visual rules are withheld for the baseline pass.
            pair_base_for_task[task.task_name] = task
            continue
        base_pair_id = _pair_task_spec_id(base_task)
        base_identifiers = {_task_spec_id(base_task), base_task.candidate_id, base_task.task_name}
        if base_task is task or (base_pair_id is not None and base_pair_id not in base_identifiers):
            raise ValueError("paired initial-state references must point to an unpaired base")
        if _physical_contract_digest(task, task.template_contract()) != _physical_contract_digest(
            base_task, base_task.template_contract()
        ):
            raise ValueError("paired tasks do not share the same physical contract")
        pair_base_for_task[task.task_name] = base_task
    for task in task_by_name.values():
        if task.task_name in pair_base_for_task:
            continue
        if any(task_by_spec_id.get(identifier) is task for identifier in referenced_base_ids):
            pair_base_for_task[task.task_name] = task

    paired_cache: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    resident_session_invalidated = False

    def callback(*, evaluator: Any, control: Any, bindings: Any, task: Any) -> Mapping[str, Any]:
        nonlocal resident_session_invalidated
        if resident_session_invalidated:
            raise RegisteredL0ConstructionError(
                "registered construction resident session is invalidated",
                {
                    "status": "FAIL",
                    "construction_pass": False,
                    "failure_class": "RESIDENT_SESSION_INVALIDATED",
                    "resident_session_invalidated": True,
                },
            )
        registered = task_by_name.get(task.task_name)
        if registered is None:
            raise KeyError(f"registered L0 task is unavailable: {task.task_name}")
        if registered.task_name in unavailable_pair_sources:
            raise RegisteredL0ConstructionError('paired initial state is unavailable in this worker', {
                'status': 'SKIPPED_BASE_BLOCKED', 'construction_pass': False,
                'initial_state_status': 'UNAVAILABLE', 'failure_class': 'INITIAL_STATE_UNAVAILABLE',
                'blocked_by_base': unavailable_pair_sources[registered.task_name],
                'oracle_status': 'NOT_RUN'})
        record = getattr(registered, "record", {})
        construction = record.get("construction", {}) if isinstance(record, Mapping) else {}
        hold = None
        construction_bindings = bindings
        construction_mode = bool(
            (allow_pending_initial_state or allow_live_oracle_construction)
            and construction.get("frozen_initial_state") is None
        )
        if construction_mode:
            hold = _RegisteredConstructionRobotHold(
                evaluator=evaluator,
                task=registered,
            )
            construction_bindings = _RegisteredConstructionBindingsProxy(
                bindings,
                hold,
            )
        try:
            base_task = pair_base_for_task.get(registered.task_name)
            if base_task is None or not construction_mode:
                result = construct_registered_l0_initial_state(
                    evaluator=evaluator,
                    control=control,
                    bindings=construction_bindings,
                    task=registered,
                    predicate_mode=predicate_mode,
                )
            else:
                contract = registered.template_contract()
                physical_digest = _physical_contract_digest(registered, contract)
                base_task_spec_id = _cache_base_task_spec_id(registered, base_task)
                cache_key = (id(evaluator.env.scene), base_task_spec_id, physical_digest)
                entry = paired_cache.get(cache_key)
                if entry is not None:
                    result = _paired_report_from_cache(
                        evaluator=evaluator,
                        task=registered,
                        base_task=base_task,
                        entry=entry,
                        refresh_state=lambda: _refresh_restored_scene(evaluator, control, construction_bindings),
                        predicate_mode=predicate_mode,
                    )
                else:
                    # A variant-first callback compiles the exact base recipe once,
                    # without its visual override, then shares that native state.
                    baseline = construct_registered_l0_initial_state(
                        evaluator=evaluator,
                        control=control,
                        bindings=construction_bindings,
                        task=base_task,
                        apply_visual_transforms=False,
                        predicate_mode=predicate_mode,
                    )
                    try:
                        state = _scene_state_snapshot(evaluator.env.scene)
                        entry = {
                            "scene": evaluator.env.scene,
                            "state": state,
                            "scene_state_sha256": _scene_state_digest(state),
                            "base_task_spec_id": base_task_spec_id,
                            "physical_contract_sha256": physical_digest,
                            "object_identities": _cache_object_identity(
                                evaluator, base_task.template_contract()
                            ),
                            "pose_stability_readback": baseline.get("pose_stability_readback"),
                            "baseline_object_diagnostics": _pair_object_diagnostics(
                                evaluator, base_task.template_contract()
                            ),
                        }
                        # Validate before publishing the cache so malformed baseline
                        # state can never be consumed by a later paired task.
                        _validate_cached_pair_state(
                            evaluator=evaluator, task=base_task, entry=entry,
                            predicate_mode=predicate_mode,
                        )
                    except RegisteredL0ConstructionError:
                        raise
                    except Exception as exc:
                        try:
                            _restore_registered_render_overrides(evaluator)
                        except RegisteredL0ConstructionError as restore_error:
                            raise restore_error from exc
                        raise RegisteredL0ConstructionError(
                            "paired baseline snapshot failed",
                            {
                                "status": "FAIL",
                                "construction_pass": False,
                                "failure_class": "PAIRED_INITIAL_STATE_SNAPSHOT_ERROR",
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        ) from exc
                    paired_cache[cache_key] = entry
                    result = _paired_report_from_cache(
                        evaluator=evaluator,
                        task=registered,
                        base_task=base_task,
                        entry=entry,
                        load_state=False,
                        predicate_mode=predicate_mode,
                    )
                    result = dict(result)
                    result["paired_initial_state"] = dict(
                        result["paired_initial_state"],
                        cache_hit=False,
                        physical_state_inherited=registered is not base_task,
                        baseline_construction_executed=True,
                    )
        except RegisteredL0ConstructionError as exc:
            if exc.report.get("resident_session_invalidated"):
                resident_session_invalidated = True
            if hold is None:
                raise
            report = dict(exc.report)
            try:
                hold.restore()
            except Exception as hold_error:
                resident_session_invalidated = True
                report.update(
                    {
                        "resident_session_invalidated": True,
                        "robot_construction_hold_restore_error": (
                            f"{type(hold_error).__name__}: {hold_error}"
                        ),
                    }
                )
            report["robot_construction_hold"] = dict(hold.report())
            raise RegisteredL0ConstructionError(str(exc), report) from exc
        except Exception as exc:
            if hold is not None:
                try:
                    hold.restore()
                except Exception as hold_error:
                    resident_session_invalidated = True
                    report = dict(hold.report())
                    report.update(
                        {
                            "status": "FAIL",
                            "construction_pass": False,
                            "failure_class": "ROBOT_CONSTRUCTION_HOLD_ERROR",
                            "resident_session_invalidated": True,
                            "error": f"{type(hold_error).__name__}: {hold_error}",
                            "construction_error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    raise RegisteredL0ConstructionError(
                        "registered construction robot hold failed",
                        report,
                    ) from hold_error
            raise
        def finalize_initial_branch(result):
            nonlocal resident_session_invalidated
            branch_contract = registered.template_contract()
            branch = branch_contract.get('task_contract')
            if branch is None:
                return result
            from .l1_initial_branch import materialize_initial_branch
            def branch_role_object(role):
                scope = branch_contract['role_to_scope'][role]
                return _live_object(evaluator.env.scene, branch_contract['scope_bindings'][scope]['actual_name'])
            try:
                facts = materialize_initial_branch(
                    branch, roles=branch_contract['role_to_scope'], role_object=branch_role_object,
                    recipe_truth=lambda recipe, resolve: _initial_recipe_truth(recipe, resolve, predicate_mode=predicate_mode), robot=evaluator.env.robots[0],
                    object_aabb=object_aabb, screen_axes=_screen_axes_xy,
                    configuration_id=_task_spec_id(registered),
                )
            except Exception as exc:
                invalidated = bool(getattr(exc, 'session_invalidated', False))
                resident_session_invalidated = resident_session_invalidated or invalidated
                raise RegisteredL0ConstructionError('L1 initial branch readback failed', {
                    **dict(result), 'status': 'FAIL', 'construction_pass': False,
                    'failure_class': 'L1_INITIAL_BRANCH_ERROR',
                    'resident_session_invalidated': invalidated,
                    'error': f'{type(exc).__name__}: {exc}',
                }) from exc
            return {**dict(result), 'l1_initial_branch': facts}
        if hold is None:
            return _validate_runtime_initial_relations(
                evaluator, registered, finalize_initial_branch(result), predicate_mode)
        try:
            hold.restore()
        except Exception as exc:
            report = dict(hold.report())
            report.update(
                {
                    "status": "FAIL",
                    "construction_pass": False,
                    "failure_class": "ROBOT_CONSTRUCTION_HOLD_ERROR",
                    "resident_session_invalidated": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            resident_session_invalidated = True
            raise RegisteredL0ConstructionError(
                "registered construction robot hold failed",
                report,
            ) from exc
        result = dict(result)
        result["robot_construction_hold"] = dict(hold.report())
        return _validate_runtime_initial_relations(
            evaluator, registered, finalize_initial_branch(result), predicate_mode)

    return callback


_CANONICAL_STATE_BY_RELATION = {
    "inside": ("Inside", True),
    "on_top": ("OnTop", True),
    "under": ("Under", True),
    "next_to": ("NextTo", True),
    "touching": ("Touching", True),
    "open": ("Open", True),
    "closed": ("Open", False),
    "toggled_on": ("ToggledOn", True),
    "not_toggled_on": ("ToggledOn", False),
}


def _registered_symbolic_relation_truth(subject: Any, kind: str, reference: Any = None, *, predicate_mode="action20_symbolic") -> bool:
    """Shared initial/reset/goal reader matching Action20 predicate semantics.

    OnTop and NextTo use the shared Symbolic geometry. Other states retain the same
    get_value API used by the Action and official predicate adapter. In
    particular, Touching is not replaced by an invented AABB approximation.
    """
    mapping = _CANONICAL_STATE_BY_RELATION.get(kind)
    if mapping is None:
        raise RuntimeError(f"registered relation is unsupported: {kind}")
    binary = kind in {"inside", "on_top", "under", "next_to", "touching"}
    if subject is None or binary != (reference is not None):
        raise RuntimeError(f"registered {kind} relation binding/arity mismatch")
    state_name, desired = mapping
    if predicate_mode == "official" and state_name in {"OnTop", "Touching"}:
        for obj in (subject, reference):
            clear = getattr(obj, "clear_states_cache", None)
            if callable(clear):
                clear()
    if reference is not None:
        observed = (symbolic_on_top_relation(subject, reference)
                    if state_name == "OnTop" and predicate_mode == "action20_symbolic"
                    else read_action_relation(subject, state_name, reference, predicate_mode=predicate_mode))
    else:
        state = state_by_name(subject, state_name)
        getter = getattr(state, "get_value", None)
        if not callable(getter):
            raise RuntimeError(f"canonical {state_name} state is unavailable")
        observed = bool(getter())
    return observed is desired


def _registered_goal_relation_truth(scope: Mapping[str, Any], relation: Any, *, predicate_mode="action20_symbolic") -> bool:
    return _registered_symbolic_relation_truth(
        scope.get(relation.subject), relation.kind,
        None if relation.reference is None else scope.get(relation.reference),
        predicate_mode=predicate_mode,
    )


def audit_registered_l0_navigation_route(
    *,
    evaluator: Any,
    task: RegisteredL0Task,
    route_planner: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Compute the selected Goal-DAG route without moving the robot."""

    if route_planner is None:
        from behavior_hl.embodied.evaluator_bridge.run_b1_marker_symbolic_rollout import (
            _private_official_oracle_route_plan as route_planner,
        )
    profile = task.profile()
    leaves = tuple(json.loads(leaf) for leaf in profile.canonical_goal_leaves)
    scope = getattr(getattr(evaluator, "env", None), "task", None)
    scope = getattr(scope, "object_scope", None)
    if not isinstance(scope, Mapping):
        raise RuntimeError("registered navigation audit object_scope is unavailable")
    relations = tuple(relation_from_grounded_leaf(leaf) for leaf in leaves)
    initially_true = tuple(
        relation.to_official_body()
        for relation in relations
        if _registered_goal_relation_truth(scope, relation)
    )
    contract = task.template_contract()
    if len(initially_true) == len(relations) and not _initial_process_unfinished(
            contract, lambda role: _role_object_for_contract(evaluator, contract, role)):
        raise RuntimeError("registered L0 goal is already satisfied initially")
    plan = compile_grounded_mixed_option(
        leaves,
        initially_true_relations=initially_true,
    )
    navigation = audit_goal_plan_navigation_route(
        evaluator=evaluator,
        plan=plan,
        route_planner=route_planner,
    )
    passed = navigation["route_exists"] is True
    return {
        "schema_version": NAVIGATION_AUDIT_VERSION,
        "candidate_id": task.candidate_id,
        "task_name": task.task_name,
        "status": "PASS" if passed else "FAIL",
        "failure_class": None if passed else "NAVIGATION_ERROR",
        "goal_actions": [node.action for node in plan.action_nodes],
        "goal_node_count": len(plan.action_nodes),
        "initially_true_goal_leaf_count": len(initially_true),
        **navigation,
        "static_collision_contract": (
            "official_xy_astar_relaxed_base_footprint_no_curobo_no_held_object"
        ),
        "public_move_m": RESTRICTED_KINEMATIC_PUBLIC_MOVE_M,
        "simulator_state_mutated": False,
        "official_challenge_eligible": False,
    }


def make_registered_l0_navigation_audit_callback(
    tasks: Mapping[str, RegisteredL0Task],
) -> Callable[..., Mapping[str, Any]]:
    task_by_name = {task.task_name: task for task in tasks.values()}

    def callback(*, evaluator: Any, task: Any) -> Mapping[str, Any]:
        registered = task_by_name.get(task.task_name)
        if registered is None:
            raise KeyError(
                f"registered L0 navigation task is unavailable: {task.task_name}"
            )
        return audit_registered_l0_navigation_route(
            evaluator=evaluator,
            task=registered,
        )

    return callback


def configure_registered_l0_tasks(
    evaluator_module: Any,
    tasks: Mapping[str, RegisteredL0Task],
    *,
    full_scene_load: bool = False,
    defer_process_goal_termination: bool = False,
) -> Callable[[], None]:
    """Install exact custom-task deltas for this process and return cleanup."""

    if not isinstance(full_scene_load, bool):
        raise TypeError("full_scene_load must be bool")
    if not isinstance(defer_process_goal_termination, bool):
        raise TypeError('defer_process_goal_termination must be bool')

    import omnigibson.eval.utils.eval_utils as eval_utils
    import omnigibson.eval.utils.score_utils as score_utils
    import omnigibson.utils.asset_utils as asset_utils
    import omnigibson.utils.bddl_utils as bddl_utils

    originals = {
        "load_available_tasks": evaluator_module.load_available_tasks,
        "evaluator_load_human_stats": evaluator_module.load_human_stats,
        "score_load_human_stats": score_utils.load_human_stats,
        "evaluator_get_task_instance_path": evaluator_module.get_task_instance_path,
        "asset_get_task_instance_path": asset_utils.get_task_instance_path,
    }
    ids_before = dict(eval_utils.TASK_NAMES_TO_INDICES)
    kb = bddl_utils.get_knowledge_base()
    task_by_name = {task.task_name: task for task in tasks.values()}
    if len(task_by_name) != len(tasks):
        raise ValueError("registered L0 activity names must be unique")
    if _ACTIVE_REGISTERED_L0_TASKS:
        raise RuntimeError("registered L0 task runtime is already configured")
    for task in tasks.values():
        if (
            kb.get_task(f"{task.task_name}-0") is not None
            or task.task_name in ids_before
            or task.task_id in ids_before.values()
            or task.task_name in inside_task_profiles._inside_profiles
        ):
            raise ValueError(
                f"registered L0 task collides with the official runtime: {task.task_name}"
            )
    added_task_names = []
    for task in tasks.values():
        kb_name = f"{task.task_name}-0"
        if kb.get_task(kb_name) is not None:
            raise ValueError(f"registered L0 task already exists in the BDDL knowledge base: {kb_name}")
        kb.add_task(name=kb_name, definition=task.bddl_path.read_text())
        added_task_names.append(kb_name)
        eval_utils.TASK_NAMES_TO_INDICES[task.task_name] = task.task_id
        inside_task_profiles._inside_profiles[task.task_name] = task.profile()
        _ACTIVE_REGISTERED_L0_TASKS[task.task_name] = task

    def load_available_tasks() -> dict[str, Any]:
        value = dict(originals["load_available_tasks"]())
        for task in tasks.values():
            available = task.available_task()
            if full_scene_load:
                available[0]["load_room_instances"] = None
            value[task.task_name] = available
        return value

    def load_human_stats(task_name: str, *args: Any, **kwargs: Any) -> dict[str, float]:
        if task_name in task_by_name:
            return {
                "length": 100_000.0,
                "distance_traveled": 1.0,
                "left_eef_displacement": 1.0,
                "right_eef_displacement": 1.0,
            }
        return originals["score_load_human_stats"](task_name, *args, **kwargs)

    def get_task_instance_path(scene_model: str, name: str, *args: Any, **kwargs: Any) -> Any:
        for task in tasks.values():
            expected = (
                f"{task.scene_model}_task_{task.task_name}_instances/"
                f"{task.scene_model}_task_{task.task_name}_0_0_template-tro_state"
            )
            if scene_model == task.scene_model and name == expected:
                # During construction-only compilation, the native Scene
                # completion context owns the only safe source for omitted
                # articulated joint fields.  It materializes a runner-owned
                # TRO copy for the existing evaluator path; outside that
                # context the frozen path remains unchanged.
                from behavior_hl.embodied.evaluator_bridge.resident_scene_state import (
                    materialize_tro_state_from_initial_state,
                )

                contract = task.template_contract()
                materialized = materialize_tro_state_from_initial_state(
                    tro_state_path=task.tro_state_path,
                    scope_bindings=contract.get("scope_bindings", {}),
                )
                if materialized is not None:
                    return str(materialized)
                return str(task.tro_state_path)
        return originals["asset_get_task_instance_path"](scene_model, name, *args, **kwargs)

    evaluator_module.load_available_tasks = load_available_tasks
    evaluator_module.load_human_stats = load_human_stats
    score_utils.load_human_stats = load_human_stats
    evaluator_module.get_task_instance_path = get_task_instance_path
    asset_utils.get_task_instance_path = get_task_instance_path
    marker_policy_token = MARKER_LABEL_COUNTERBALANCE.set(None)
    process_cleanup = None
    process_names = [task.task_name for task in (tasks.values() if defer_process_goal_termination else ())
        if task.record.get('task_contract', {}).get('process_requirements', {}).get('process_program')
        or task.record.get('task_contract', {}).get('process_requirements', {}).get('history_program')]
    if defer_process_goal_termination and process_names:
        from omnigibson.tasks.behavior_task import BehaviorTask
        from .process_goal_lifecycle import install_process_goal_lifecycle
        process_cleanup = install_process_goal_lifecycle(BehaviorTask, process_names)

    def cleanup() -> None:
        if process_cleanup is not None:
            process_cleanup()
        MARKER_LABEL_COUNTERBALANCE.reset(marker_policy_token)
        evaluator_module.load_available_tasks = originals["load_available_tasks"]
        evaluator_module.load_human_stats = originals["evaluator_load_human_stats"]
        score_utils.load_human_stats = originals["score_load_human_stats"]
        evaluator_module.get_task_instance_path = originals["evaluator_get_task_instance_path"]
        asset_utils.get_task_instance_path = originals["asset_get_task_instance_path"]
        eval_utils.TASK_NAMES_TO_INDICES.clear()
        eval_utils.TASK_NAMES_TO_INDICES.update(ids_before)
        for task_name in task_by_name:
            inside_task_profiles._inside_profiles.pop(task_name, None)
            _ACTIVE_REGISTERED_L0_TASKS.pop(task_name, None)
        for name in added_task_names:
            task = kb.tasks_by_name.pop(name)
            kb.tasks.remove(task)

    return cleanup
