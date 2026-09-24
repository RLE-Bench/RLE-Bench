"""Stage credit: how much of a composite task a trial actually completed.

WHY THIS EXISTS. RoboCasa's composite tasks succeed only on a CONJUNCTION -- the fruit is
in the blender AND the blender is running. `Kitchen.reward()` is
`float(self._check_success())` and nothing else, so a run that gets three of four
conjuncts scores what a run that never moved scores. On tasks this hard that puts every
agent on the same number and ranks nothing.

So the evaluation task gets a function below re-implementing the conjuncts of its
`_check_success` INDIVIDUALLY. The stage arithmetic is covered by `tests/runtime/test_stage_credit.py`.
The intended properties are:

  EQUIVALENCE   all stages true at once  <=>  env._check_success() is True. A solved
                trial scores exactly 1.0 and the environment's predicate stays the sole
                authority on success. These functions grade PARTIAL progress; they never
                grant success.

  ZERO FLOOR    a trial in which the agent did nothing scores exactly 0.0. Free for the
                current split -- AfterwashSorting arrives with all three conjuncts false
                (`tasks/task02/dev/data/task02_stage_baseline.json`) -- but kept because it is not free in
                general: `gripper_obj_far` is true whenever the arm starts away from the
                object, and several RoboCasa scenes satisfy a conjunct at reset.
                Normalising against the reset state is the general fix.

GRANULARITY. One evaluation task means the reward IS this trial's stage fraction, so the
conjunct count is the resolution of the whole benchmark. AfterwashSorting has three, none
free at reset: 0 / 1/3 / 2/3 / 1. A two-conjunct held-out task would be nearly binary --
worth checking when adopting another activity group.

GROUND TRUTH, NOT AGENT REPORT. Everything here reads simulator state directly, inside the
root-only tree, and the agent never sees this module or any value it produces (CLAUDE.md
invariant #1).

MAINTENANCE. These functions mirror RoboCasa source; audit them after pin changes.
"""

from __future__ import annotations

from typing import Callable

# A stage vector: ordered (name, satisfied) pairs, one per conjunct.
Stages = list[tuple[str, bool]]


def _ou():
    """RoboCasa's object utilities, imported lazily so this module is importable -- and
    most of it testable -- without a simulator."""
    from robocasa.utils import object_utils as OU

    return OU


def _pos(env, name):
    """An object's world position, the way every `_check_success` reads it."""
    import numpy as np

    return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])


def _near(env, a: str, b: str, th: float) -> bool:
    import numpy as np

    return bool(np.linalg.norm(_pos(env, a) - _pos(env, b)) < th)


# --------------------------------------------------------------------------
# One function per evaluation task, mirroring its `_check_success` conjunct by conjunct
# in source order. Where the original folds checks behind control flow, each check becomes
# its own stage: finer credit, and it cannot break equivalence, since an early return out
# of a conjunction is still that conjunction.
# --------------------------------------------------------------------------

def afterwash_sorting(env) -> Stages:
    """AfterwashSorting: sort the washed produce into two bowls, then turn the water off.

    NOT A FLAT CONJUNCTION -- a disjunction inside one:

        (not water_on) and ((food1 & food2 in bowl1 and food3 in bowl2)
                         or (food1 & food2 in bowl2 and food3 in bowl1))

    Either bowl may hold the pair; only the PARTITION is graded, so the stages are phrased
    RELATIONALLY. Naming a bowl would break equivalence: an agent that used the other one
    would score zero on a "pair in bowl1" conjunct while `_check_success` called it solved.
    """
    OU = _ou()
    inb = {(f, b): OU.check_obj_in_receptacle(env, f, b)
           for f in ("food1", "food2", "food3") for b in ("bowl1", "bowl2")}

    pair_in_1 = inb[("food1", "bowl1")] and inb[("food2", "bowl1")]
    pair_in_2 = inb[("food1", "bowl2")] and inb[("food2", "bowl2")]
    # The odd one out belongs in whichever bowl the pair did NOT go in.
    separated = (pair_in_1 and inb[("food3", "bowl2")]) or \
                (pair_in_2 and inb[("food3", "bowl1")])

    return [
        ("water_off", not env.sink.get_handle_state(env=env)["water_on"]),
        ("pair_together", bool(pair_in_1 or pair_in_2)),
        ("odd_one_separated", bool(separated)),
    ]


def align_silverware(env) -> Stages:
    """AlignSilverware: fork left of the plate, spoon right of it, both on the table.

    A FLAT CONJUNCTION of five, so every conjunct becomes its own stage in source order.
    Two are compound in the original -- `fork_left` is a side test AND two distance
    tests -- and they stay compound here: splitting them would credit a fork that is
    merely on the correct side of the room.

    Positions are compared in the DINING COUNTER's local frame, because "left" and
    "right" are only meaningful relative to the seat. That is `-stool.rot + pi`, taken
    verbatim from `_check_success`; a global-frame comparison would grade a different
    task in every layout.

    `gripper_clear_of_plate` is true at reset, as it is in most RoboCasa scenes -- the
    arm starts away from everything. Reset-normalisation in the scorer is what stops
    that being free credit; see the module docstring.
    """
    import numpy as np

    OU = _ou()
    rot = -env.stool.rot + np.pi

    def _local(name):
        p = np.array(env.sim.data.body_xpos[env.obj_body_id[name]])
        return OU.transform_global_to_local(p[0], p[1], rot)

    fork_x, fork_y = _local("fork")
    spoon_x, spoon_y = _local("spoon")
    plate_x, plate_y = _local("plate")

    lateral_threshold, forward_threshold = 0.25, 0.10
    fork_left = (fork_x < plate_x
                 and abs(fork_x - plate_x) <= lateral_threshold
                 and abs(fork_y - plate_y) <= forward_threshold)
    spoon_right = (spoon_x > plate_x
                   and abs(spoon_x - plate_x) <= lateral_threshold
                   and abs(spoon_y - plate_y) <= forward_threshold)

    return [
        ("fork_left_of_plate", bool(fork_left)),
        ("spoon_right_of_plate", bool(spoon_right)),
        ("fork_on_table",
         bool(OU.check_obj_fixture_contact(env, "fork", env.dining_counter))),
        ("spoon_on_table",
         bool(OU.check_obj_fixture_contact(env, "spoon", env.dining_counter))),
        ("gripper_clear_of_plate", bool(OU.gripper_obj_far(env, obj_name="plate"))),
    ]


def collect_washing_supplies(env) -> Stages:
    """CollectWashingSupplies: both supplies on the counter and within 0.2 m of the sink."""
    OU = _ou()
    return [
        ("supplies_close_to_sink",
         bool(OU.obj_fixture_bbox_min_dist(env, "supply1", env.sink) < 0.20
              and OU.obj_fixture_bbox_min_dist(env, "supply2", env.sink) < 0.20)),
        ("supplies_on_counter",
         bool(OU.check_obj_fixture_contact(env, "supply1", env.counter)
              and OU.check_obj_fixture_contact(env, "supply2", env.counter))),
        ("gripper_clear",
         bool(OU.gripper_obj_far(env, obj_name="supply1")
              and OU.gripper_obj_far(env, obj_name="supply2"))),
    ]


def load_fridge_fifo(env) -> Stages:
    """LoadFridgeFifo: both meats on the rack, the new one behind the old, fridge shut.

    `fifo_order` compares depth in the FRIDGE's local frame -- `-fridge.rot`, verbatim
    from the original, because "behind" is meaningless in the global frame.
    """
    OU = _ou()
    rack = dict(compartment="fridge", rack_index=env.rack_index)
    old, new = _pos(env, "meat_old"), _pos(env, "meat_new")
    _, old_y = OU.transform_global_to_local(old[0], old[1], -env.fridge.rot)
    _, new_y = OU.transform_global_to_local(new[0], new[1], -env.fridge.rot)
    return [
        ("meat_old_on_rack", bool(env.fridge.check_rack_contact(env, "meat_old", **rack))),
        ("meat_new_on_rack", bool(env.fridge.check_rack_contact(env, "meat_new", **rack))),
        ("fifo_order", bool(new_y > old_y + 0.1)),
        ("fridge_closed", bool(env.fridge.is_closed(env))),
    ]


def maximize_freezer_space(env) -> Stages:
    """MaximizeFreezerSpace: both loose items moved onto the target rack, the two already
    there undisturbed, freezer shut."""
    target = -1 if env.move_direction == "up" else -2
    rack = dict(compartment="freezer", rack_index=target)
    return [
        ("item1_on_target_rack", bool(env.fridge.check_rack_contact(env, "item1", **rack))),
        ("item2_on_target_rack", bool(env.fridge.check_rack_contact(env, "item2", **rack))),
        ("item3_undisturbed", bool(env.fridge.check_rack_contact(env, "item3", **rack))),
        ("item4_undisturbed", bool(env.fridge.check_rack_contact(env, "item4", **rack))),
        ("freezer_closed", bool(env.fridge.is_closed(env=env, compartment="freezer"))),
    ]


def gather_tableware(env) -> Stages:
    """GatherTableware: the three glasses clustered well away from the bowl.

    The clustering test is purely RELATIVE (`max_glass * 1.5 < max_bowl`) and has no
    absolute threshold to split on, so it stays one conjunct.
    """
    OU = _ou()
    max_glass = max(env.dist_between_obj("glass1", "glass2"),
                    env.dist_between_obj("glass2", "glass3"),
                    env.dist_between_obj("glass3", "glass1"))
    max_bowl = max(env.dist_between_obj("bowl", "glass1"),
                   env.dist_between_obj("bowl", "glass2"),
                   env.dist_between_obj("bowl", "glass3"))
    return [
        ("gripper_clear_of_glass1", bool(OU.gripper_obj_far(env, obj_name="glass1"))),
        ("gripper_clear_of_glass2", bool(OU.gripper_obj_far(env, obj_name="glass2"))),
        ("gripper_clear_of_glass3", bool(OU.gripper_obj_far(env, obj_name="glass3"))),
        ("glasses_clustered_away_from_bowl", bool(max_glass * 1.5 < max_bowl)),
    ]


def drinkware_consolidation(env) -> Stages:
    """DrinkwareConsolidation: every drinkware item inside the cabinet.

    The count varies by scene, so "some" and "all" stand in for a per-item stage the
    fixed STAGE_NAMES schema cannot express. `some` is implied by `all`, so equivalence
    holds and the first item stowed earns credit.
    """
    OU = _ou()
    inside = [OU.obj_inside_of(env, f"obj_{i}", env.cab)
              for i in range(env.num_drinkware)]
    return [
        ("some_drinkware_in_cabinet", any(inside)),
        ("all_drinkware_in_cabinet", all(inside)),
        ("gripper_clear",
         all(OU.gripper_obj_far(env, f"obj_{i}") for i in range(env.num_drinkware))),
    ]


def frying_pan_adjustment(env) -> Stages:
    """FryingPanAdjustment: the pan moved to a different burner, and that burner lit."""
    loc = env._check_obj_location_on_stove("obj")
    lit = bool(loc is not None
               and loc in env.stove.get_knobs_state(env=env)
               and env.stove.is_burner_on(env=env, burner_loc=loc))
    return [
        ("burner_under_pan_on", lit),
        ("pan_moved_from_start", bool(loc != env.start_loc)),
    ]


def portion_fruit_bowl(env) -> Stages:
    """PortionFruitBowl: the four fruits split two and two between the bowls."""
    OU = _ou()
    fruits = ("fruit1", "fruit2", "fruit3", "fruit4")
    in1 = sum(OU.check_obj_in_receptacle(env, f, "bowl1") for f in fruits)
    in2 = sum(OU.check_obj_in_receptacle(env, f, "bowl2") for f in fruits)
    return [
        ("two_fruits_in_bowl1", in1 == 2),
        ("two_fruits_in_bowl2", in2 == 2),
        ("gripper_clear", all(OU.gripper_obj_far(env, f) for f in fruits)),
    ]


def butter_on_pan(env) -> Stages:
    """ButterOnPan: butter in the pan, pan on its assigned burner, burner lit."""
    OU = _ou()
    return [
        ("butter_in_pan", bool(OU.check_obj_in_receptacle(env, "butter", "pan"))),
        ("pan_on_burner",
         bool(env.stove.check_obj_location_on_stove(env=env, obj_name="pan",
                                                    threshold=0.15) == env.knob)),
        ("burner_on",
         bool(env.knob in env.stove.get_knobs_state(env=env)
              and env.stove.is_burner_on(env=env, burner_loc=env.knob))),
    ]


def organize_baking_ingredients(env) -> Stages:
    """OrganizeBakingIngredients: eggs and milk gathered within 0.2 m of the bowl."""
    OU = _ou()
    objs = ("bowl", "egg1", "egg2", "milk")
    return [
        ("egg1_near_bowl", _near(env, "bowl", "egg1", 0.20)),
        ("egg2_near_bowl", _near(env, "bowl", "egg2", 0.20)),
        ("milk_near_bowl", _near(env, "bowl", "milk", 0.20)),
        ("gripper_clear", all(OU.gripper_obj_far(env, o, th=0.15) for o in objs)),
    ]


def kettle_boiling(env) -> Stages:
    """KettleBoiling: kettle seated on a burner and that burner lit.

    The original scans the burners and returns on the first that satisfies both, so the
    stages are phrased EXISTENTIALLY -- naming a burner would fail on any other one.
    """
    import numpy as np

    OU = _ou()
    knobs = env.stove.get_knobs_state(env=env)
    kettle = _pos(env, env.objects["obj"].name)[0:2]
    seated, lit = False, False
    if OU.check_obj_fixture_contact(env, "obj", env.stove):
        for location, site in env.stove.burner_sites.items():
            if site is None:
                continue
            burner = np.array(env.sim.data.get_site_xpos(site.get("name")))[0:2]
            if np.linalg.norm(burner - kettle) < 0.15:
                seated = True
                if location in knobs and env.stove.is_burner_on(env=env,
                                                                burner_loc=location):
                    lit = True
    return [
        ("kettle_on_burner", bool(seated)),
        ("burner_under_kettle_on", bool(lit)),
        ("gripper_clear", bool(OU.gripper_obj_far(env))),
    ]


def move_to_counter(env) -> Stages:
    """MoveToCounter: the frosted food on the plate."""
    OU = _ou()
    return [
        ("food_on_plate",
         bool(OU.check_obj_in_receptacle(env, "frosted_food", "plate"))),
        ("gripper_clear", bool(OU.gripper_obj_far(env, "frosted_food"))),
    ]


def microwave_correct_meal(env) -> Stages:
    """MicrowaveCorrectMeal: the target bowl loaded with both its foods, in the microwave,
    door shut and running."""
    OU = _ou()
    bowl = f"bowl{env.target_bowl}"
    food0 = f"food0_bowl{env.target_bowl}"
    food1 = f"food1_bowl{env.target_bowl}"
    return [
        ("bowl_in_microwave", bool(OU.obj_inside_of(env, bowl, env.microwave))),
        ("food0_in_bowl", bool(OU.check_obj_in_receptacle(env, food0, bowl))),
        ("food1_in_bowl", bool(OU.check_obj_in_receptacle(env, food1, bowl))),
        ("door_closed", bool(env.microwave.is_closed(env))),
        ("microwave_on", bool(env.microwave.get_state()["turned_on"])),
        ("gripper_clear", bool(OU.gripper_obj_far(env, obj_name=bowl))),
    ]


def alcohol_serving_prep(env) -> Stages:
    """AlcoholServingPrep: alcohol and cup both set down on the dining table."""
    OU = _ou()
    return [
        ("gripper_clear_of_alcohol", bool(OU.gripper_obj_far(env, obj_name="alcohol"))),
        ("gripper_clear_of_cup", bool(OU.gripper_obj_far(env, obj_name="cup"))),
        ("alcohol_on_table",
         bool(OU.check_obj_fixture_contact(env, "alcohol", env.dining_table))),
        ("cup_on_table",
         bool(OU.check_obj_fixture_contact(env, "cup", env.dining_table))),
    ]


def arrange_buffet_dessert(env) -> Stages:
    """ArrangeBuffetDessert: both sweets on the tray, tray on the dining counter."""
    OU = _ou()
    return [
        ("sweet1_on_tray", bool(OU.check_obj_in_receptacle(env, "sweet1", "tray"))),
        ("sweet2_on_tray", bool(OU.check_obj_in_receptacle(env, "sweet2", "tray"))),
        ("tray_on_counter",
         bool(OU.check_obj_fixture_contact(env, "tray", env.dining_counter))),
        ("gripper_clear",
         all(OU.gripper_obj_far(env, obj_name=o, th=0.15)
             for o in ("sweet1", "sweet2", "tray"))),
    ]


def arrange_vegetables(env) -> Stages:
    """ArrangeVegetables: both vegetables on the cutting board."""
    OU = _ou()
    return [
        ("vegetable1_on_board",
         bool(OU.check_obj_in_receptacle(env, "vegetable1", "cutting_board"))),
        ("vegetable2_on_board",
         bool(OU.check_obj_in_receptacle(env, "vegetable2", "cutting_board"))),
        ("gripper_clear", bool(OU.gripper_obj_far(env, obj_name="cutting_board"))),
    ]


def add_ice_cubes(env) -> Stages:
    """AddIceCubes: every ice cube inside the blender.

    Split "some"/"all" for the same reason as `drinkware_consolidation`: the cube count
    is per-scene, so a per-cube stage cannot be named in advance.
    """
    OU = _ou()
    inside = [OU.obj_inside_of(env, f"ice_cube{i}", env.blender, th=0.01)
              for i in range(env.num_ice_cubes)]
    return [
        ("some_ice_in_blender", any(inside)),
        ("all_ice_in_blender", all(inside)),
        ("gripper_clear",
         all(OU.gripper_obj_far(env, f"ice_cube{i}")
             for i in range(env.num_ice_cubes))),
    ]


def heat_mug(env) -> Stages:
    """HeatMug: mug inside the microwave with the door shut."""
    OU = _ou()
    return [
        ("mug_in_microwave", bool(OU.obj_inside_of(env, "obj", env.microwave))),
        ("gripper_clear", bool(OU.gripper_obj_far(env))),
        ("door_closed", bool(env.microwave.is_closed(env=env))),
    ]


def pan_transfer(env) -> Stages:
    """PanTransfer: vegetable tipped onto the plate, pan back on a burner, never touched
    by the gripper."""
    OU = _ou()
    return [
        ("vegetable_on_plate",
         bool(OU.check_obj_in_receptacle(env, "vegetable", "plate"))),
        ("pan_on_stove",
         bool(env._check_obj_location_on_stove("vegetable_container") is not None)),
        ("gripper_clear", bool(OU.gripper_obj_far(env, "vegetable_container"))),
        ("food_never_touched", bool(not env._robot_touched_food)),
    ]


def make_fruit_bowl(env) -> Stages:
    """MakeFruitBowl: both fruits in the bowl and the cabinet shut behind you."""
    OU = _ou()
    return [
        ("fruit1_in_bowl", bool(OU.check_obj_in_receptacle(env, "fruit1", "bowl"))),
        ("fruit2_in_bowl", bool(OU.check_obj_in_receptacle(env, "fruit2", "bowl"))),
        ("cabinet_closed", bool(env.cab.is_closed(env=env))),
    ]


def prepare_storing_leftovers(env) -> Stages:
    """PrepareStoringLeftovers: tupperware and foil set on the dining counter within
    0.3 m of the plate."""
    OU = _ou()
    return [
        ("tupperware_on_counter",
         bool(OU.check_obj_fixture_contact(env, "tupperware", env.dining_counter))),
        ("foil_on_counter",
         bool(OU.check_obj_fixture_contact(env, "aluminum_foil", env.dining_counter))),
        ("tupperware_near_plate", _near(env, "tupperware", "plate", 0.3)),
        ("foil_near_plate", _near(env, "aluminum_foil", "plate", 0.3)),
        ("gripper_clear",
         all(OU.gripper_obj_far(env, o) for o in ("tupperware", "aluminum_foil"))),
    ]


def toast_bagel(env) -> Stages:
    """ToastBagel: bagel on a toaster rack, door shut, timer wound.

    Either rack level counts where the oven has two, so the stage is phrased over both.
    """
    OU = _ou()
    oven = env.toaster_oven
    if oven.has_multiple_rack_levels():
        in_toaster = (oven.check_rack_contact(env, "bagel", rack_level=0)
                      or oven.check_rack_contact(env, "bagel", rack_level=1))
    else:
        in_toaster = oven.check_rack_contact(env, "bagel")
    door_closed = bool(oven.is_closed(env))
    # The original reads the timer only behind `bagel_in_toaster and door_closed`, so a
    # timer wound before either is not credited -- mirrored rather than hoisted.
    timer_set = bool(in_toaster and door_closed
                     and oven.get_state(env)["time"] >= 0.1)
    return [
        ("bagel_in_toaster", bool(in_toaster)),
        ("door_closed", door_closed),
        ("timer_set", timer_set),
        ("gripper_clear", bool(OU.gripper_obj_far(env, "bagel"))),
    ]


# --------------------------------------------------------------------------
# The 2026-08 split. One mirror per newly adopted evaluation task, same rules as
# above: conjuncts in SOURCE ORDER, compound tests split only where the split is
# itself a conjunction (finer credit, equivalence preserved), and a check the
# original computes behind control flow mirrored with that control flow intact.
# --------------------------------------------------------------------------

def sorting_cleanup(env) -> Stages:
    """SortingCleanup: mug into the sink, bowl into the cabinet, cabinet shut."""
    OU = _ou()
    return [
        ("mug_in_sink", bool(OU.obj_inside_of(env, "mug", env.sink))),
        ("bowl_in_cabinet", bool(OU.obj_inside_of(env, "bowl", env.cab))),
        ("cabinet_closed", bool(env.cab.is_closed(env=env))),
        ("gripper_clear_of_mug", bool(OU.gripper_obj_far(env, "mug"))),
    ]


def setup_butter_plate(env) -> Stages:
    """SetupButterPlate: butter onto the plate, knife on a counter beside it.

    `knife_near_plate` is an unconditional xy distance in the ORIGINAL -- unlike
    SweetenCoffee below, it is not gated on the objects being on a counter -- so it can
    read true while the knife is still in the gripper. That is the predicate's own
    behaviour and is mirrored rather than tightened.
    """
    import numpy as np

    OU = _ou()
    knife, plate = _pos(env, "butter_knife"), _pos(env, "plate")
    return [
        ("butter_on_plate", bool(OU.check_obj_in_receptacle(env, "butter", "plate"))),
        ("knife_near_plate",
         bool(np.linalg.norm(knife[:2] - plate[:2]) <= 0.3)),
        ("gripper_clear_of_butter",
         bool(OU.gripper_obj_far(env, obj_name="butter"))),
        ("gripper_clear_of_knife",
         bool(OU.gripper_obj_far(env, obj_name="butter_knife"))),
        ("knife_on_counter",
         bool(OU.check_obj_any_counter_contact(env, "butter_knife"))),
    ]


def move_freezer_to_fridge(env) -> Stages:
    """MoveFreezerToFridge: frozen meat onto any fridge rack, both doors shut.

    The two distractor conjuncts are DO-NO-HARM guards, true at reset and lost only by
    knocking an item off its rack. They earn nothing under reset-normalisation and are
    kept because dropping them would break equivalence.
    """
    fridge = env.fridge
    return [
        ("meat_in_fridge",
         bool(fridge.check_rack_contact(env, "frozen_meat", compartment="fridge"))),
        ("freezer_closed", bool(fridge.is_closed(env, compartment="freezer"))),
        ("fridge_closed", bool(fridge.is_closed(env, compartment="fridge"))),
        ("freezer_distractor_undisturbed",
         bool(fridge.check_rack_contact(env, "freezer_distractor",
                                        compartment="freezer"))),
        ("fridge_distractor_undisturbed",
         bool(fridge.check_rack_contact(env, "fridge_distractor",
                                        compartment="fridge"))),
    ]


def separate_freezer_rack(env) -> Stages:
    """SeparateFreezerRack: meat and vegetables into their tupperwares, each onto a
    named freezer rack.

    Seven conjuncts, five of them earnable -- the highest resolution in the split. The
    rack indices are hard-coded -1/-2 in the original AND named in the instruction
    ("the highest rack", "the second highest rack"), so the target is perceivable.
    """
    OU = _ou()
    fridge = env.fridge
    return [
        ("meat_in_tupperware",
         bool(OU.check_obj_in_receptacle(env, "meat", "meat_tupperware"))),
        ("vegetable1_in_tupperware",
         bool(OU.check_obj_in_receptacle(env, "vegetable1", "veg_tupperware"))),
        ("vegetable2_in_tupperware",
         bool(OU.check_obj_in_receptacle(env, "vegetable2", "veg_tupperware"))),
        ("veg_tupperware_on_top_rack",
         bool(fridge.check_rack_contact(env, "veg_tupperware", compartment="freezer",
                                        rack_index=-1))),
        ("meat_tupperware_on_second_rack",
         bool(fridge.check_rack_contact(env, "meat_tupperware", compartment="freezer",
                                        rack_index=-2))),
        ("gripper_clear_of_meat_tupperware",
         bool(OU.gripper_obj_far(env, obj_name="meat_tupperware"))),
        ("gripper_clear_of_veg_tupperware",
         bool(OU.gripper_obj_far(env, obj_name="veg_tupperware"))),
    ]


def restock_pantry(env) -> Stages:
    """RestockPantry: both cans into the cabinet, each beside the cans already there.

    `cans_close` is one variable in the original but a conjunction of two independent
    per-object tests, so it becomes two stages. The proximity test itself calls the
    environment's own `_close_to_cab_cans` rather than re-deriving its 2:1 ratio --
    re-deriving it is exactly the drift the equivalence test exists to catch.
    """
    OU = _ou()
    return [
        ("obj1_in_cabinet", bool(OU.obj_inside_of(env, "obj1", env.cab))),
        ("obj2_in_cabinet", bool(OU.obj_inside_of(env, "obj2", env.cab))),
        ("obj1_beside_existing_cans", bool(env._close_to_cab_cans("obj1"))),
        ("obj2_beside_existing_cans", bool(env._close_to_cab_cans("obj2"))),
        ("gripper_clear", bool(OU.gripper_obj_far(env, "obj1")
                               and OU.gripper_obj_far(env, "obj2"))),
    ]


def candle_cleanup(env) -> Stages:
    """CandleCleanup: both decorations from the dining table into the cabinet, shut."""
    OU = _ou()
    return [
        ("cabinet_closed", bool(env.cab.is_closed(env=env))),
        ("obj1_in_cabinet", bool(OU.obj_inside_of(env, "obj1", env.cab))),
        ("obj2_in_cabinet", bool(OU.obj_inside_of(env, "obj2", env.cab))),
    ]


def meal_prep_staging(env) -> Stages:
    """MealPrepStaging: both pans onto different burners, then one food on each.

    `pans_on_stove` is a conjunction over the two pans and splits; `pans_diff` cannot,
    since it is a comparison. Note that `pan1_loc != pan2_loc` is TRUE when exactly one
    pan is off the stove (None != a burner), which is why the two location stages have
    to stand beside it -- all three together still mean what the original means.
    """
    OU = _ou()
    loc1 = env._check_obj_location_on_stove(obj_name="pan1")
    loc2 = env._check_obj_location_on_stove(obj_name="pan2")
    veg1 = OU.check_obj_in_receptacle(env, "vegetable", "pan1")
    veg2 = OU.check_obj_in_receptacle(env, "vegetable", "pan2")
    meat1 = OU.check_obj_in_receptacle(env, "meat", "pan1")
    meat2 = OU.check_obj_in_receptacle(env, "meat", "pan2")
    return [
        ("pan1_on_burner", loc1 is not None),
        ("pan2_on_burner", loc2 is not None),
        ("pans_on_different_burners", bool(loc1 != loc2)),
        ("foods_on_separate_pans", bool((veg1 and meat2) or (veg2 and meat1))),
    ]


def portion_hot_dogs(env) -> Stages:
    """PortionHotDogs: exactly one bun and one sausage on each of the two plates.

    Four independently graded placements and no fixture interaction at all -- the
    highest-coverage task in the split.
    """
    OU = _ou()
    buns = ("hotdog_bun1", "hotdog_bun2")
    sausages = ("sausage1", "sausage2")

    def exactly_one(names, plate):
        return sum(bool(OU.check_obj_in_receptacle(env, n, plate)) for n in names) == 1

    return [
        ("one_bun_on_plate1", exactly_one(buns, "plate1")),
        ("one_sausage_on_plate1", exactly_one(sausages, "plate1")),
        ("one_bun_on_plate2", exactly_one(buns, "plate2")),
        ("one_sausage_on_plate2", exactly_one(sausages, "plate2")),
        ("gripper_clear",
         all(bool(OU.gripper_obj_far(env, n)) for n in buns + sausages)),
    ]


def sweeten_coffee(env) -> Stages:
    """SweetenCoffee: sugar into the coffee, milk from the fridge beside it.

    `milk_close_to_coffee` is computed ONLY when both are on a counter in the original,
    so a milk carton held 10 cm from the mug does not read as placed. Mirrored with that
    gate intact.
    """
    import numpy as np

    OU = _ou()
    milk_on = bool(OU.check_obj_any_counter_contact(env, "milk"))
    coffee_on = bool(OU.check_obj_any_counter_contact(env, "coffee"))
    close = bool(milk_on and coffee_on and np.linalg.norm(
        _pos(env, "milk")[:2] - _pos(env, "coffee")[:2]) <= 0.25)
    return [
        ("milk_on_counter", milk_on),
        ("coffee_on_counter", coffee_on),
        ("milk_beside_coffee", close),
        ("sugar_in_coffee",
         bool(OU.check_obj_in_receptacle(env, "sugar_cube", "coffee"))),
        ("gripper_clear", bool(OU.gripper_obj_far(env, obj_name="milk")
                               and OU.gripper_obj_far(env, obj_name="sugar_cube"))),
    ]


def defrost_by_category(env) -> Stages:
    """DefrostByCategory: the two fruits into the sink, the two vegetables into the bowl.

    The original writes each pair as one variable; both are plain conjunctions over
    independent objects, so each splits into two stages -- 2 earnable conjuncts become 4.
    """
    OU = _ou()
    return [
        ("fruit1_in_sink", bool(OU.obj_inside_of(env, "obj0", env.sink))),
        ("fruit2_in_sink", bool(OU.obj_inside_of(env, "obj1", env.sink))),
        ("vegetable1_in_bowl",
         bool(OU.check_obj_in_receptacle(env, "obj2", "container"))),
        ("vegetable2_in_bowl",
         bool(OU.check_obj_in_receptacle(env, "obj3", "container"))),
        ("gripper_clear",
         all(bool(OU.gripper_obj_far(env, obj_name=f"obj{i}")) for i in range(4))),
    ]


def place_microwave_safe_item(env) -> Stages:
    """PlaceMicrowaveSafeItem: the microwavable item in, the other left out, door shut,
    started.

    The original returns early when the door is open and never reads `turned_on`, so
    `microwave_on` is gated the same way here. Ungating it would let a vector claim the
    appliance is running with the door open -- a state the predicate never grants.
    """
    OU = _ou()
    door_closed = bool(env.microwave.is_closed(env))
    return [
        ("safe_item_in_microwave",
         bool(OU.obj_inside_of(env, "microwave_safe_item", env.microwave))),
        ("unsafe_item_left_out",
         not bool(OU.obj_inside_of(env, "microwave_unsafe_item", env.microwave))),
        ("door_closed", door_closed),
        ("microwave_on",
         bool(door_closed and env.microwave.get_state()["turned_on"])),
    ]


def prepare_drink_station(env) -> Stages:
    """PrepareDrinkStation: cup and mug onto the tray, pitcher within 0.3 m of it."""
    import numpy as np

    OU = _ou()
    tray, pitcher = _pos(env, "tray"), _pos(env, "pitcher")
    return [
        ("cup_on_tray", bool(OU.check_obj_in_receptacle(env, "cup", "tray"))),
        ("mug_on_tray", bool(OU.check_obj_in_receptacle(env, "mug", "tray"))),
        ("pitcher_beside_tray",
         bool(np.linalg.norm(tray[:2] - pitcher[:2]) <= 0.3)),
        ("gripper_clear", bool(OU.gripper_obj_far(env, obj_name="cup")
                               and OU.gripper_obj_far(env, obj_name="mug")
                               and OU.gripper_obj_far(env, obj_name="pitcher"))),
    ]


def clear_cutting_board(env) -> Stages:
    """ClearCuttingBoard: both vegetables onto the board AND the non-vegetable off it.

    The only NEGATIVE conjunct in the split -- credit for removing something rather than
    placing it -- which is why this task is worth having.
    """
    OU = _ou()
    board = "non_vegetable_container"
    return [
        ("vegetable1_on_board",
         bool(OU.check_obj_in_receptacle(env, "vegetable1", board))),
        ("vegetable2_on_board",
         bool(OU.check_obj_in_receptacle(env, "vegetable2", board))),
        ("gripper_clear_of_board",
         bool(OU.gripper_obj_far(env, obj_name=board))),
        ("board_cleared_of_non_vegetable",
         not bool(OU.check_obj_in_receptacle(env, "non_vegetable", board))),
    ]


def prepare_smoothie(env) -> Stages:
    """PrepareSmoothie: both smoothie foods onto the plate, distractor left in place."""
    OU = _ou()
    return [
        ("food1_on_plate",
         bool(OU.check_obj_in_receptacle(env, "smoothie_food1", "plate"))),
        ("food2_on_plate",
         bool(OU.check_obj_in_receptacle(env, "smoothie_food2", "plate"))),
        ("gripper_clear", bool(OU.gripper_obj_far(env, "smoothie_food1")
                               and OU.gripper_obj_far(env, "smoothie_food2"))),
        ("distractor_left_in_fridge",
         bool(OU.obj_inside_of(env, "distr_item", env.fridge))),
    ]


def dessert_upgrade(env) -> Stages:
    """DessertUpgrade: both desserts from the plates onto the tray."""
    OU = _ou()
    return [
        ("dessert1_on_tray",
         bool(OU.check_obj_in_receptacle(env, "dessert1", "receptacle"))),
        ("dessert2_on_tray",
         bool(OU.check_obj_in_receptacle(env, "dessert2", "receptacle"))),
        ("gripper_clear_of_tray", bool(OU.gripper_obj_far(env, "receptacle"))),
    ]


def veggie_dip_prep(env) -> Stages:
    """VeggieDipPrep: both vegetables and the dip bowl onto the tray.

    The gripper test uses th=0.15 rather than the default -- taken verbatim, since a
    different threshold grades a different task.
    """
    OU = _ou()
    return [
        ("gripper_clear", bool(OU.gripper_obj_far(env, "bowl", th=0.15)
                               and OU.gripper_obj_far(env, "cucumber", th=0.15)
                               and OU.gripper_obj_far(env, "carrot", th=0.15))),
        ("cucumber_on_tray",
         bool(OU.check_obj_in_receptacle(env, "cucumber", "tray"))),
        ("carrot_on_tray", bool(OU.check_obj_in_receptacle(env, "carrot", "tray"))),
        ("bowl_on_tray", bool(OU.check_obj_in_receptacle(env, "bowl", "tray"))),
    ]


def store_leftovers_in_bowl(env) -> Stages:
    """StoreLeftoversInBowl: both leftovers into the bowl, bowl onto any fridge rack.

    `check_rack_contact` is called WITHOUT a rack index in the original, so any shelf
    counts -- which is what makes this the forgiving member of its group.
    """
    OU = _ou()
    return [
        ("chicken_in_bowl",
         bool(OU.check_obj_in_receptacle(env, "chicken_drumstick", "bowl"))),
        ("vegetable_in_bowl",
         bool(OU.check_obj_in_receptacle(env, "vegetable", "bowl"))),
        ("bowl_on_fridge_rack", bool(env.fridge.check_rack_contact(env, "bowl"))),
        ("gripper_clear_of_bowl", bool(OU.gripper_obj_far(env, "bowl"))),
    ]


def toast_on_correct_rack(env) -> Stages:
    """ToastOnCorrectRack: bread and meat each onto their assigned toaster-oven rack.

    The rack levels are drawn per episode but the instruction names them ("top",
    "bottom"), so the assignment is perceivable rather than hidden state.
    """
    OU = _ou()
    oven = env.toaster_oven
    return [
        ("bread_on_its_rack",
         bool(oven.check_rack_contact(env, "bread", rack_level=env.bread_rack_level))),
        ("meat_on_its_rack",
         bool(oven.check_rack_contact(env, "meat", rack_level=env.meat_rack_level))),
        ("gripper_clear", bool(OU.gripper_obj_far(env, "bread")
                               and OU.gripper_obj_far(env, "meat"))),
    ]


# --------------------------------------------------------------------------
# Held-out tasks adopted for their DIFFICULTY rather than by the coverage audit: each is
# the one member of its group that stays clear of hinged doors and enclosed cavities, and
# each was read out of RoboCasa source rather than trusted to `decompose.py`, which scores
# a task by the primitives its subtasks name and so misses a fixture the scene opens for
# you (or does not). See the difficulty gate in tasks/task02/build_groups.py.
# --------------------------------------------------------------------------

def dump_leftovers(env) -> Stages:
    """DumpLeftovers: empty the bowl onto the counter, then put the bowl in the sink.

    RoboCasa's shipped `_check_success` tests `leftover1` twice and never `leftover2`;
    `env._fix_dump_leftovers_predicate` corrects it at construction, and this mirror
    follows the corrected predicate: one conjunct per leftover.
    """
    OU = _ou()
    return [
        ("leftover1_off_bowl",
         not bool(OU.check_obj_in_receptacle(env, "leftover1", "bowl"))),
        ("leftover2_off_bowl",
         not bool(OU.check_obj_in_receptacle(env, "leftover2", "bowl"))),
        ("bowl_in_sink", bool(OU.obj_inside_of(env, "bowl", env.sink))),
        ("gripper_clear_of_bowl", bool(OU.gripper_obj_far(env, obj_name="bowl"))),
    ]


def pastry_display(env) -> Stages:
    """PastryDisplay: one pastry on each of the two plates.

    RELATIONAL, like `afterwash_sorting`: the predicate is a disjunction over which plate
    takes which pastry, so the graded fact is the PAIRING and not the assignment. Naming a
    plate would score zero on an agent that used the other one while `_check_success`
    called it solved.
    """
    OU = _ou()
    inr = {(p, r): bool(OU.check_obj_in_receptacle(env, p, r))
           for p in ("pastry1", "pastry2") for r in ("receptacle1", "receptacle2")}
    paired = ((inr[("pastry1", "receptacle1")] and inr[("pastry2", "receptacle2")])
              or (inr[("pastry1", "receptacle2")] and inr[("pastry2", "receptacle1")]))
    # `some` is IMPLIED by `paired`, so it cannot break equivalence -- and without it the
    # task is binary once reset-normalisation removes the two gripper stages, which are
    # both true the moment the episode starts. Same shape as `drinkware_consolidation`.
    return [
        ("some_pastry_on_a_plate", any(inr.values())),
        ("each_pastry_on_a_plate", paired),
        ("gripper_clear_of_pastry1", bool(OU.gripper_obj_far(env, obj_name="pastry1"))),
        ("gripper_clear_of_pastry2", bool(OU.gripper_obj_far(env, obj_name="pastry2"))),
    ]


def warm_croissant(env) -> Stages:
    """WarmCroissant: croissant into the pan already on the stove, then light its burner.

    The knob guard is `butter_on_pan`'s: `is_burner_on` is asked only about a knob the
    stove currently reports, so a half-built scene reads as "not on" rather than raising.
    """
    OU = _ou()
    return [
        ("croissant_in_pan",
         bool(OU.check_obj_in_receptacle(env, "croissant", "pan"))),
        ("burner_on",
         bool(env.knob in env.stove.get_knobs_state(env=env)
              and env.stove.is_burner_on(env=env, burner_loc=env.knob))),
        ("gripper_clear_of_croissant",
         bool(OU.gripper_obj_far(env, obj_name="croissant"))),
    ]


def simmering_sauce(env) -> Stages:
    """SimmeringSauce: the pan onto ONE NAMED burner, both foods in it, that burner lit.

    A FLAT CONJUNCTION of four, none free at reset -- the pan spawns on the counter and the
    stove starts off -- so this is the group's full-resolution task.

    `env.knob` is drawn per episode (`rng.choice(valid_knobs)`) and NAMED in the
    instruction, so `pan_on_named_burner` is a language-grounded target, not just a
    placement: the pan being on SOME burner does not satisfy it. The comparison is
    `_check_obj_location_on_stove(...) == env.knob` verbatim from `_check_success`,
    threshold and all; loosening it to "on the stove" would grade a different task.

    The knob guard is `warm_croissant`'s -- `is_burner_on` is asked only about a knob the
    stove currently reports, so a half-built scene reads as "not on" rather than raising.
    """
    OU = _ou()
    return [
        ("pan_on_named_burner",
         bool(env._check_obj_location_on_stove("pan") == env.knob)),
        ("tomato_in_pan", bool(OU.check_obj_in_receptacle(env, "tomato", "pan"))),
        ("onion_in_pan", bool(OU.check_obj_in_receptacle(env, "onion", "pan"))),
        ("burner_on",
         bool(env.knob in env.stove.get_knobs_state(env=env)
              and env.stove.is_burner_on(env=env, burner_loc=env.knob))),
    ]


def place_vegetables_evenly(env) -> Stages:
    """PlaceVegetablesEvenly: both vegetables in the pan, side by side rather than stacked.

    `_check_success` returns early unless both are in the pan, then tests the separation.
    Splitting the early return into its own stages cannot break equivalence -- an early
    return out of a conjunction is still that conjunction -- and it is what makes the task
    gradeable: getting one vegetable in is most of the work and scores nothing otherwise.

    The two separation stages are the source's thresholds verbatim: SAME height (stacking
    is what it rules out) and APART in the horizontal plane.
    """
    import numpy as np

    OU = _ou()
    vegs = ("veg1", "veg2")
    in_pan = {v: bool(OU.check_obj_in_receptacle(env, v, "pan")) for v in vegs}
    p1, p2 = (_pos(env, v) for v in vegs)
    return [
        ("veg1_in_pan", in_pan["veg1"]),
        ("veg2_in_pan", in_pan["veg2"]),
        ("not_stacked", bool(abs(p1[2] - p2[2]) < 0.02)),
        ("spread_apart", bool(np.linalg.norm(p1[:2] - p2[:2]) > 0.06)),
        ("gripper_clear", bool(all(OU.gripper_obj_far(env, v) for v in vegs))),
    ]


def match_cup_and_drink(env) -> Stages:
    """MatchCupAndDrink: each bottle carried to the vessel it belongs with.

    The two vessels are already on the dining counter and the two bottles start on the
    work counter, so this is two carries and a semantic pairing -- wine to the wine glass,
    juice to the glass cup -- with no fixture touched. `0.25` is the source's threshold,
    measured in the horizontal plane only.
    """
    import numpy as np

    OU = _ou()

    def near(a: str, b: str) -> bool:
        return bool(np.linalg.norm(_pos(env, a)[:2] - _pos(env, b)[:2]) <= 0.25)

    return [
        ("wine_bottle_by_wine_glass", near("wine_bottle", "wine_glass")),
        ("juice_bottle_by_glass_cup", near("juice_bottle", "glass_cup")),
        ("gripper_clear",
         bool(OU.gripper_obj_far(env, obj_name="wine_bottle")
              and OU.gripper_obj_far(env, obj_name="juice_bottle"))),
    ]


def place_beverages_together(env) -> Stages:
    """PlaceBeveragesTogether: the three drinks onto the dining counter, in a tight group.

    `all_on_counter` is split PER DRINK. The source folds three fixture-contact tests into
    one `all(...)`, and each is a whole carry across the kitchen -- collapsing them would
    make two thirds of the work score nothing. Splitting a conjunction cannot break
    equivalence.

    `drinks_clustered` stays whole because it is not a conjunction over drinks: it asks
    that EVERY drink have a nearest neighbour within 0.25 m, which is a property of the
    arrangement rather than of any one bottle. Note it is true at reset -- the three start
    together on the work counter -- so reset-normalisation removes it, and it is graded
    only to catch an agent that scatters them while moving them.
    """
    import numpy as np

    OU = _ou()
    drinks = ("alcohol", "juice", "bottled_water")
    pos = {d: _pos(env, d)[:2] for d in drinks}
    clustered = all(
        min(np.linalg.norm(pos[d] - pos[o]) for o in drinks if o != d) <= 0.25
        for d in drinks)
    return [
        ("alcohol_on_dining_counter",
         bool(OU.check_obj_fixture_contact(env, "alcohol", env.dining_counter))),
        ("juice_on_dining_counter",
         bool(OU.check_obj_fixture_contact(env, "juice", env.dining_counter))),
        ("water_on_dining_counter",
         bool(OU.check_obj_fixture_contact(env, "bottled_water", env.dining_counter))),
        ("drinks_clustered", bool(clustered)),
        ("gripper_clear",
         bool(all(OU.gripper_obj_far(env, obj_name=d, th=0.15) for d in drinks))),
    ]


# Keyed by RoboCasa task name, so this and `config.EVAL_TASKS` must agree -- asserted in
# tests, because a missing entry would silently reduce the trial to binary success.
STAGE_FNS: dict[str, Callable[[object], Stages]] = {
    "AfterwashSorting": afterwash_sorting,
    "CollectWashingSupplies": collect_washing_supplies,
    "AlignSilverware": align_silverware,
    "LoadFridgeFifo": load_fridge_fifo,
    "MaximizeFreezerSpace": maximize_freezer_space,
    "GatherTableware": gather_tableware,
    "DrinkwareConsolidation": drinkware_consolidation,
    "FryingPanAdjustment": frying_pan_adjustment,
    "PortionFruitBowl": portion_fruit_bowl,
    "ButterOnPan": butter_on_pan,
    "OrganizeBakingIngredients": organize_baking_ingredients,
    "KettleBoiling": kettle_boiling,
    "MoveToCounter": move_to_counter,
    "MicrowaveCorrectMeal": microwave_correct_meal,
    "AlcoholServingPrep": alcohol_serving_prep,
    "ArrangeBuffetDessert": arrange_buffet_dessert,
    "ArrangeVegetables": arrange_vegetables,
    "AddIceCubes": add_ice_cubes,
    "HeatMug": heat_mug,
    "PanTransfer": pan_transfer,
    "MakeFruitBowl": make_fruit_bowl,
    "PrepareStoringLeftovers": prepare_storing_leftovers,
    "ToastBagel": toast_bagel,
    "SortingCleanup": sorting_cleanup,
    "SetupButterPlate": setup_butter_plate,
    "MoveFreezerToFridge": move_freezer_to_fridge,
    "SeparateFreezerRack": separate_freezer_rack,
    "RestockPantry": restock_pantry,
    "CandleCleanup": candle_cleanup,
    "MealPrepStaging": meal_prep_staging,
    "PortionHotDogs": portion_hot_dogs,
    "SweetenCoffee": sweeten_coffee,
    "DefrostByCategory": defrost_by_category,
    "PlaceMicrowaveSafeItem": place_microwave_safe_item,
    "PrepareDrinkStation": prepare_drink_station,
    "ClearCuttingBoard": clear_cutting_board,
    "PrepareSmoothie": prepare_smoothie,
    "DessertUpgrade": dessert_upgrade,
    "VeggieDipPrep": veggie_dip_prep,
    "StoreLeftoversInBowl": store_leftovers_in_bowl,
    "ToastOnCorrectRack": toast_on_correct_rack,
    "DumpLeftovers": dump_leftovers,
    "PastryDisplay": pastry_display,
    "WarmCroissant": warm_croissant,
    "PlaceVegetablesEvenly": place_vegetables_evenly,
    "MatchCupAndDrink": match_cup_and_drink,
    "PlaceBeveragesTogether": place_beverages_together,
    "SimmeringSauce": simmering_sauce,
}


# The conjuncts each task is graded on, declared rather than discovered: this is the
# SCHEMA of the ledger's trial records, and two runs are only comparable if the names mean
# the same thing in both. `read_stages` checks what it got against this, so a branch that
# forgot a conjunct fails loudly instead of scoring against a shorter denominator.
STAGE_NAMES: dict[str, tuple[str, ...]] = {
    "AfterwashSorting": ("water_off", "pair_together", "odd_one_separated"),
    "CollectWashingSupplies": ("supplies_close_to_sink", "supplies_on_counter",
                               "gripper_clear"),
    "AlignSilverware": ("fork_left_of_plate", "spoon_right_of_plate", "fork_on_table",
                        "spoon_on_table", "gripper_clear_of_plate"),
    "LoadFridgeFifo": ("meat_old_on_rack", "meat_new_on_rack", "fifo_order",
                       "fridge_closed"),
    "MaximizeFreezerSpace": ("item1_on_target_rack", "item2_on_target_rack",
                             "item3_undisturbed", "item4_undisturbed",
                             "freezer_closed"),
    "GatherTableware": ("gripper_clear_of_glass1", "gripper_clear_of_glass2",
                        "gripper_clear_of_glass3", "glasses_clustered_away_from_bowl"),
    "DrinkwareConsolidation": ("some_drinkware_in_cabinet", "all_drinkware_in_cabinet",
                               "gripper_clear"),
    "FryingPanAdjustment": ("burner_under_pan_on", "pan_moved_from_start"),
    "PortionFruitBowl": ("two_fruits_in_bowl1", "two_fruits_in_bowl2", "gripper_clear"),
    "ButterOnPan": ("butter_in_pan", "pan_on_burner", "burner_on"),
    "OrganizeBakingIngredients": ("egg1_near_bowl", "egg2_near_bowl", "milk_near_bowl",
                                  "gripper_clear"),
    "KettleBoiling": ("kettle_on_burner", "burner_under_kettle_on", "gripper_clear"),
    "MoveToCounter": ("food_on_plate", "gripper_clear"),
    "MicrowaveCorrectMeal": ("bowl_in_microwave", "food0_in_bowl", "food1_in_bowl",
                             "door_closed", "microwave_on", "gripper_clear"),
    "AlcoholServingPrep": ("gripper_clear_of_alcohol", "gripper_clear_of_cup",
                           "alcohol_on_table", "cup_on_table"),
    "ArrangeBuffetDessert": ("sweet1_on_tray", "sweet2_on_tray", "tray_on_counter",
                             "gripper_clear"),
    "ArrangeVegetables": ("vegetable1_on_board", "vegetable2_on_board",
                          "gripper_clear"),
    "AddIceCubes": ("some_ice_in_blender", "all_ice_in_blender", "gripper_clear"),
    "HeatMug": ("mug_in_microwave", "gripper_clear", "door_closed"),
    "PanTransfer": ("vegetable_on_plate", "pan_on_stove", "gripper_clear",
                    "food_never_touched"),
    "MakeFruitBowl": ("fruit1_in_bowl", "fruit2_in_bowl", "cabinet_closed"),
    "PrepareStoringLeftovers": ("tupperware_on_counter", "foil_on_counter",
                                "tupperware_near_plate", "foil_near_plate",
                                "gripper_clear"),
    "ToastBagel": ("bagel_in_toaster", "door_closed", "timer_set", "gripper_clear"),
    "SortingCleanup": ("mug_in_sink", "bowl_in_cabinet", "cabinet_closed",
                       "gripper_clear_of_mug"),
    "SetupButterPlate": ("butter_on_plate", "knife_near_plate",
                         "gripper_clear_of_butter", "gripper_clear_of_knife",
                         "knife_on_counter"),
    "MoveFreezerToFridge": ("meat_in_fridge", "freezer_closed", "fridge_closed",
                            "freezer_distractor_undisturbed",
                            "fridge_distractor_undisturbed"),
    "SeparateFreezerRack": ("meat_in_tupperware", "vegetable1_in_tupperware",
                            "vegetable2_in_tupperware", "veg_tupperware_on_top_rack",
                            "meat_tupperware_on_second_rack",
                            "gripper_clear_of_meat_tupperware",
                            "gripper_clear_of_veg_tupperware"),
    "RestockPantry": ("obj1_in_cabinet", "obj2_in_cabinet", "obj1_beside_existing_cans",
                      "obj2_beside_existing_cans", "gripper_clear"),
    "CandleCleanup": ("cabinet_closed", "obj1_in_cabinet", "obj2_in_cabinet"),
    "MealPrepStaging": ("pan1_on_burner", "pan2_on_burner", "pans_on_different_burners",
                        "foods_on_separate_pans"),
    "PortionHotDogs": ("one_bun_on_plate1", "one_sausage_on_plate1",
                       "one_bun_on_plate2", "one_sausage_on_plate2", "gripper_clear"),
    "SweetenCoffee": ("milk_on_counter", "coffee_on_counter", "milk_beside_coffee",
                      "sugar_in_coffee", "gripper_clear"),
    "DefrostByCategory": ("fruit1_in_sink", "fruit2_in_sink", "vegetable1_in_bowl",
                          "vegetable2_in_bowl", "gripper_clear"),
    "PlaceMicrowaveSafeItem": ("safe_item_in_microwave", "unsafe_item_left_out",
                               "door_closed", "microwave_on"),
    "PrepareDrinkStation": ("cup_on_tray", "mug_on_tray", "pitcher_beside_tray",
                            "gripper_clear"),
    "ClearCuttingBoard": ("vegetable1_on_board", "vegetable2_on_board",
                          "gripper_clear_of_board", "board_cleared_of_non_vegetable"),
    "PrepareSmoothie": ("food1_on_plate", "food2_on_plate", "gripper_clear",
                        "distractor_left_in_fridge"),
    "DessertUpgrade": ("dessert1_on_tray", "dessert2_on_tray", "gripper_clear_of_tray"),
    "VeggieDipPrep": ("gripper_clear", "cucumber_on_tray", "carrot_on_tray",
                      "bowl_on_tray"),
    "StoreLeftoversInBowl": ("chicken_in_bowl", "vegetable_in_bowl",
                             "bowl_on_fridge_rack", "gripper_clear_of_bowl"),
    "ToastOnCorrectRack": ("bread_on_its_rack", "meat_on_its_rack", "gripper_clear"),
    "DumpLeftovers": ("leftover1_off_bowl", "leftover2_off_bowl", "bowl_in_sink",
                      "gripper_clear_of_bowl"),
    "PastryDisplay": ("some_pastry_on_a_plate", "each_pastry_on_a_plate",
                      "gripper_clear_of_pastry1", "gripper_clear_of_pastry2"),
    "WarmCroissant": ("croissant_in_pan", "burner_on",
                      "gripper_clear_of_croissant"),
    "PlaceVegetablesEvenly": ("veg1_in_pan", "veg2_in_pan", "not_stacked",
                              "spread_apart", "gripper_clear"),
    "MatchCupAndDrink": ("wine_bottle_by_wine_glass", "juice_bottle_by_glass_cup",
                         "gripper_clear"),
    "PlaceBeveragesTogether": ("alcohol_on_dining_counter", "juice_on_dining_counter",
                               "water_on_dining_counter", "drinks_clustered",
                               "gripper_clear"),
    "SimmeringSauce": ("pan_on_named_burner", "tomato_in_pan", "onion_in_pan",
                       "burner_on"),
}


# Tasks whose predicate has ONE real degree of freedom once the conjuncts that are true at
# reset are normalised out (`tasks/task02/dev/data/task02_stage_baseline.json`). Their reward is binary
# however it is scored, so read their success rate and not their reward.
#
# Not a to-do list: each was checked and cannot be split further without breaking
# equivalence. FryingPanAdjustment and MoveToCounter simply are two-conjunct tasks;
# GatherTableware turns on one purely RELATIVE comparison with no threshold to divide;
# PanTransfer starts with its pan already on the stove and its food untouched.
LOW_RESOLUTION = frozenset({
    "FryingPanAdjustment", "GatherTableware", "MoveToCounter", "PanTransfer",
})


def has_stages(task: str) -> bool:
    return task in STAGE_FNS


def read_stages(task: str, env) -> Stages | None:
    """The stage vector for `env` right now, or None if it cannot be read.

    Never raises. A predicate can legitimately fail on a half-built environment, and a
    trial that cannot be measured must be scored as unmeasured rather than crash the
    daemon measuring it. A vector whose names do not match STAGE_NAMES is discarded: a
    wrong denominator is worse than no measurement.
    """
    fn = STAGE_FNS.get(task)
    if fn is None or env is None:
        return None
    try:
        read = [(str(name), bool(value)) for name, value in fn(env)]
    except Exception:  # noqa: BLE001
        return None
    if tuple(name for name, _ in read) != STAGE_NAMES.get(task):
        return None
    return read


def satisfied(stages: Stages | None) -> int:
    return sum(1 for _, ok in stages or [] if ok)


def score_trial(at_reset: Stages | None, best: Stages | None,
                *, success: bool = False) -> float:
    """Credit for one trial, in [0, 1].

        score = (best_count/total - reset_count/total) / (1 - reset_count/total)

    the fraction of the AVAILABLE conjuncts the trial completed. Three properties:

      * a trial that changed nothing scores 0, since best_count == reset_count -- the
        whole reason for the reset term;
      * a solved trial scores 1, since success means every conjunct held at once;
      * regression is punished: knocking a conjunct back out lowers the count, and the
        clamp floors the result at 0 rather than letting it go negative.

    `best` is the best count reached ANYWHERE in the trial, matching how success is
    detected -- the segment loop tests `_check_success()` after every step and stops the
    moment it fires -- so grading the final frame alone would hold partial credit to a
    stricter standard than success. The COUNT is maximised, never the conjuncts
    individually: a per-conjunct maximum would let stages satisfied at different times add
    up to a "solved" trial that never was.

    `success=True` short-circuits to 1.0: the environment's predicate outranks this module
    by construction, and if they disagree the equivalence test has a bug to report.
    """
    if success:
        return 1.0
    if not best:
        return 0.0
    total = len(best)
    if total == 0:
        return 0.0
    reset_count = satisfied(at_reset) if at_reset else 0
    # A task already satisfied at reset has nothing to award. Unreachable for an audited
    # split, but scoring it as a free 1.0 would be the worst failure available here.
    if reset_count >= total:
        return 0.0
    gained = satisfied(best) - reset_count
    return max(0.0, min(1.0, gained / (total - reset_count)))


def merge_best(current: Stages | None, candidate: Stages | None) -> Stages | None:
    """Keep whichever sample satisfies more conjuncts.

    Whole vectors are compared, never merged field by field: a per-conjunct union would
    manufacture a state the trial never reached. Ties keep the earlier sample, which is
    arbitrary and does not affect the score.
    """
    if candidate is None:
        return current
    if current is None:
        return candidate
    return candidate if satisfied(candidate) > satisfied(current) else current
