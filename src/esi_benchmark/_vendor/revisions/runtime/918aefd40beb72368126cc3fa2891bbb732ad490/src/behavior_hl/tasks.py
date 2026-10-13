"""Five pilot tasks with leaf hints and complete canonical BDDL goals."""

from __future__ import annotations

from dataclasses import dataclass

from behavior_hl.actions import ActionName, EGO_NAVIGATION_ACTIONS


COMMON_ACTIONS = EGO_NAVIGATION_ACTIONS | {ActionName.NAVIGATE_TO, ActionName.STOP}


@dataclass(frozen=True, slots=True)
class TaskSpec:
    b100_index: int
    name: str
    scene_model: str
    goal_leaf_templates: tuple[str, ...]
    official_canonical_goal: str
    object_actions: frozenset[ActionName]
    physics_status: str
    physics_note: str

    @property
    def allowed_actions(self) -> frozenset[ActionName]:
        return frozenset(COMMON_ACTIONS | self.object_actions)

    @property
    def goal_fragments(self) -> tuple[str, ...]:
        """Compatibility alias for UI hints; never use this to verify BDDL."""

        return self.goal_leaf_templates


FIVE_TASKS: dict[str, TaskSpec] = {
    "turning_on_radio": TaskSpec(
        b100_index=0,
        name="turning_on_radio",
        scene_model="house_double_floor_lower",
        goal_leaf_templates=("(toggled_on ?radio_receiver.n.01_1)",),
        official_canonical_goal="( :goal ( and ( toggled_on ?radio_receiver.n.01_1 ) ) )",
        object_actions=frozenset({ActionName.TOGGLE_ON}),
        physics_status="blocked",
        physics_note="StarterSemanticActionPrimitives._toggle currently raises NotImplementedError.",
    ),
    "picking_up_trash": TaskSpec(
        b100_index=1,
        name="picking_up_trash",
        scene_model="house_double_floor_lower",
        goal_leaf_templates=("(inside ?can__of__soda.n.01 ?ashcan.n.01_1)",),
        official_canonical_goal=(
            "( :goal ( and ( forall ( ?can__of__soda.n.01 - can__of__soda.n.01 ) "
            "( inside ?can__of__soda.n.01 ?ashcan.n.01_1 ) ) ) )"
        ),
        object_actions=frozenset({ActionName.PICK_UP, ActionName.PLACE_INSIDE}),
        physics_status="candidate",
        physics_note="Official solve_behavior_task.py demonstrates GRASP + PLACE_INSIDE on this activity.",
    ),
    "set_up_a_coffee_station_in_your_kitchen": TaskSpec(
        b100_index=10,
        name="set_up_a_coffee_station_in_your_kitchen",
        scene_model="house_single_floor",
        goal_leaf_templates=(
            "(ontop ?coffee_maker.n.01_1 ?countertop.n.01_1)",
            "(nextto ?bottle__of__coffee.n.01_1 ?coffee_maker.n.01_1)",
            "(ontop ?paper_coffee_filter.n.01_1 ?coffee_maker.n.01_1)",
            "(nextto ?saucer.n.02_1 ?coffee_maker.n.01_1)",
            "(ontop ?coffee_cup.n.01_1 ?saucer.n.02_1)",
            "(nextto ?electric_kettle.n.01_1 ?coffee_maker.n.01_1)",
        ),
        official_canonical_goal=(
            "( :goal ( and ( ontop ?coffee_maker.n.01_1 ?countertop.n.01_1 ) "
            "( nextto ?bottle__of__coffee.n.01_1 ?coffee_maker.n.01_1 ) "
            "( ontop ?paper_coffee_filter.n.01_1 ?coffee_maker.n.01_1 ) "
            "( nextto ?saucer.n.02_1 ?coffee_maker.n.01_1 ) "
            "( ontop ?coffee_cup.n.01_1 ?saucer.n.02_1 ) "
            "( nextto ?electric_kettle.n.01_1 ?coffee_maker.n.01_1 ) ) )"
        ),
        object_actions=frozenset(
            {
                ActionName.PICK_UP,
                ActionName.PLACE_ON_TOP,
                ActionName.PLACE_NEXT_TO,
            }
        ),
        physics_status="blocked",
        physics_note="The official physical primitive set has no PLACE_NEXT_TO primitive; pilot extension is symbolic.",
    ),
    "putting_dishes_away_after_cleaning": TaskSpec(
        b100_index=11,
        name="putting_dishes_away_after_cleaning",
        scene_model="house_single_floor",
        goal_leaf_templates=(
            "(inside ?plate.n.04 ?cabinet.n.01)",
            "(not (open ?cabinet.n.01))",
        ),
        official_canonical_goal=(
            "( :goal ( and ( exists ( ?cabinet.n.01 - cabinet.n.01 ) "
            "( forall ( ?plate.n.04 - plate.n.04 ) "
            "( inside ?plate.n.04 ?cabinet.n.01 ) ) ) "
            "( forall ( ?cabinet.n.01 - cabinet.n.01 ) "
            "( not ( open ?cabinet.n.01 ) ) ) ) )"
        ),
        object_actions=frozenset(
            {
                ActionName.PICK_UP,
                ActionName.PLACE_INSIDE,
                ActionName.OPEN,
                ActionName.CLOSE,
            }
        ),
        physics_status="blocked",
        physics_note="StarterSemanticActionPrimitives._open_or_close currently raises NotImplementedError.",
    ),
    "bringing_in_wood": TaskSpec(
        b100_index=15,
        name="bringing_in_wood",
        scene_model="house_double_floor_lower",
        goal_leaf_templates=("(ontop ?plywood.n.01 ?floor.n.01_2)",),
        official_canonical_goal=(
            "( :goal ( and ( forall ( ?plywood.n.01 - plywood.n.01 ) "
            "( ontop ?plywood.n.01 ?floor.n.01_2 ) ) ) )"
        ),
        object_actions=frozenset({ActionName.PICK_UP, ActionName.PLACE_ON_TOP}),
        physics_status="candidate",
        physics_note="GRASP + PLACE_ON_TOP are available; all quantified plywood instances must be moved.",
    ),
}


def get_task_spec(name: str) -> TaskSpec:
    try:
        return FIVE_TASKS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported pilot task {name!r}; choose one of {sorted(FIVE_TASKS)}") from exc
