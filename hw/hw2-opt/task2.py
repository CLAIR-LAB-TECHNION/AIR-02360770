"""Homework 2, Task 2: Integrated task and motion planning.

This module provides:
- The full table scene, and the `hasroom` fact: its domain and its state estimator.
- The length of a path in configuration space, and the cost of reaching each table.
- The domain with action costs, and the cost of a plan according to our cost model.
- Simulating actions and plans at the motion level, and executing the cheapest of
  several plans.
- The domain with lessons, what the robot learns from an infeasible action, and planning
  with simulation.
"""

import mujoco
import numpy as np

# NOTE: `scene.load_scene` replaces the model, the data and the blocks, so we always read
# them through the module, as `scene.model`, `scene.data` and `scene.BLOCKS`.
import scene
from scene import (
    HOME,
    TABLES,
    get_configuration,
    grasp,
    grasped_block,
    plan,
    release,
    reset,
    set_configuration,
)
from task1 import (
    PLACE_CLEARANCE,
    block_top,
    execute_with_monitoring,
    fact_string,
    make_problem,
    move_block,
    pick_configurations,
    place_configurations,
    sample_free_spot,
    settle,
    solve,
    table_point,
    true_state,
)


# --- a table with no room ---

def full_table_layout():
    """b1 alone on t3, and nine blocks spread over t2."""
    colors = ["yellow", "blue", "green", "red"]
    layout = {"b1": ("red", [-0.1, 0.55, 0.74])}
    spots = [(x, y) for x in [0.44, 0.60, 0.76] for y in [-0.18, 0.0, 0.18]]
    for i, (x, y) in enumerate(spots):
        layout[f"b{i + 2}"] = (colors[i % 4], [x, y, 0.74])
    return layout


DOMAIN_WITH_HASROOM = '''
(define (domain n-table-blocks-world)
  (:requirements :strips :typing)
  (:types block color table)
  (:predicates (on ?b1 ?b2 - block)
               (ontable ?b - block ?t - table)
               (clear ?b - block)
               (handempty)
               (holding ?b - block)
               (bcolor ?b - block ?c - color)
               (hasroom ?t - table))

  (:action pick-up
     :parameters (?b - block ?t - table)
     :precondition (and (clear ?b) (ontable ?b ?t) (handempty))
     :effect (and (not (ontable ?b ?t)) (not (clear ?b)) (not (handempty)) (holding ?b) (hasroom ?t)))

  (:action put-down
     :parameters (?b - block ?t - table)
     :precondition (and (holding ?b) (hasroom ?t))
     :effect (and (not (holding ?b)) (clear ?b) (handempty) (ontable ?b ?t)))

  (:action stack
     :parameters (?b1 ?b2 - block)
     :precondition (and (holding ?b1) (clear ?b2))
     :effect (and (not (holding ?b1)) (not (clear ?b2)) (clear ?b1) (handempty) (on ?b1 ?b2)))

  (:action unstack
     :parameters (?b1 ?b2 - block)
     :precondition (and (on ?b1 ?b2) (clear ?b1) (handempty))
     :effect (and (holding ?b1) (clear ?b2) (not (clear ?b1)) (not (handempty)) (not (on ?b1 ?b2))))
)
'''


def hasroom(table):
    """Whether a block can be put down on the table."""
    return sample_free_spot(table, grasped_block()) is not None


def estimate_state():
    """The abstract state of the world: the facts of Task 1, and which tables have room."""
    settle()
    return true_state() | {("hasroom", table) for table in TABLES if hasroom(table)}


# --- reasoning about the cost of moving ---

def path_length(waypoints):
    """The length of a path in configuration space (radians)."""
    #TODO sum the distances between consecutive configurations of the path
    length = ...
    return length


def access_cost(table, n_trials=5):
    """The mean length (radians) of planned paths from HOME to hovering above free spots of the table."""
    reset()
    lengths = []
    for _ in range(n_trials):
        q_high, q_low = place_configurations(table_point(table, sample_free_spot(table, None)))
        lengths.append(path_length(plan(HOME, q_high, verbose=False)))
    return np.mean(lengths)


#TODO add action costs to the domain below, which is the domain with `hasroom` (see task2.ipynb)
DOMAIN_WITH_COSTS = '''
(define (domain n-table-blocks-world)
  (:requirements :strips :typing)
  (:types block color table)
  (:predicates (on ?b1 ?b2 - block)
               (ontable ?b - block ?t - table)
               (clear ?b - block)
               (handempty)
               (holding ?b - block)
               (bcolor ?b - block ?c - color)
               (hasroom ?t - table))

  (:action pick-up
     :parameters (?b - block ?t - table)
     :precondition (and (clear ?b) (ontable ?b ?t) (handempty))
     :effect (and (not (ontable ?b ?t)) (not (clear ?b)) (not (handempty)) (holding ?b) (hasroom ?t)))

  (:action put-down
     :parameters (?b - block ?t - table)
     :precondition (and (holding ?b) (hasroom ?t))
     :effect (and (not (holding ?b)) (clear ?b) (handempty) (ontable ?b ?t)))

  (:action stack
     :parameters (?b1 ?b2 - block)
     :precondition (and (holding ?b1) (clear ?b2))
     :effect (and (not (holding ?b1)) (not (clear ?b2)) (clear ?b1) (handempty) (on ?b1 ?b2)))

  (:action unstack
     :parameters (?b1 ?b2 - block)
     :precondition (and (on ?b1 ?b2) (clear ?b1) (handempty))
     :effect (and (holding ?b1) (clear ?b2) (not (clear ?b1)) (not (handempty)) (not (on ?b1 ?b2))))
)
'''


def plan_cost(task_plan, numbers):
    """The cost of a plan according to our cost model (the numbers of the domain with action costs)."""
    total = 0
    for action in task_plan:
        name, args = action.action.name, [str(p) for p in action.actual_parameters]
        total += numbers[("access-cost", args[1])] if name in ("pick-up", "put-down") else numbers[("stack-cost",)]
    return total


# --- simulating a plan ---

def simulate_action(action, sampler=sample_free_spot):
    """Simulate an action: plan its motions, and teleport to its outcome. Returns the length of its motions (radians), or None if it is infeasible."""
    name, args = action.action.name, [str(p) for p in action.actual_parameters]
    if name in ("pick-up", "unstack"):
        q_high, q_low = pick_configurations(args[0])
    elif name == "stack":
        q_high, q_low = place_configurations(block_top(args[1]))
    else:  # put-down
        xy = sampler(args[1], args[0])
        if xy is None:
            return None
        q_high, q_low = place_configurations(table_point(args[1], xy))
    if q_high is None:
        return None  # no collision-free configurations: out of reach, or something is in the way
    waypoints = plan(get_configuration(), q_high, verbose=False)
    if waypoints is None:
        return None  # no path

    # the outcome: go down, grasp or release, come back up (the block the robot holds moves with it)
    set_configuration(q_low)
    if name in ("pick-up", "unstack"):
        grasp()
    else:
        block = grasped_block()
        release()
        move_block(block, scene.data.body(block).xpos - [0, 0, PLACE_CLEARANCE])  # the block drops onto the surface
    set_configuration(q_high)
    return path_length(waypoints) + 2 * np.linalg.norm(q_high - q_low)


def simulate_plan(task_plan, sampler=sample_free_spot):
    """The length of the motions of every action of a plan (None from the first infeasible action on). The world is left as it was."""
    snapshot = scene.data.qpos.copy(), scene.data.qvel.copy(), scene.data.eq_active.copy(), scene.model.eq_data.copy()
    costs = []
    for action in task_plan:
        costs.append(simulate_action(action, sampler) if None not in costs else None)
    scene.data.qpos[:], scene.data.qvel[:], scene.data.eq_active[:], scene.model.eq_data[:] = snapshot
    mujoco.mj_forward(scene.model, scene.data)
    return costs


def execute_cheapest(goals, domain, numbers, n_simulations=3):
    """Plan for each goal, simulate each plan, and execute the plan that is cheapest in simulation. Returns (success, frames)."""
    facts = estimate_state()
    simulated = {}  # goal -> mean simulated cost of its plan
    for goal in goals:
        task_plan = solve(make_problem(facts, goal, domain, numbers))
        if task_plan is None:
            print(f"{goal}: no plan")
            continue
        totals = []
        for _ in range(n_simulations):
            costs = simulate_plan(task_plan)
            totals.append(sum(costs) if None not in costs else np.inf)
        simulated[goal] = np.mean(totals)
        print(f"{goal}: {len(task_plan)} actions, simulated {simulated[goal]:.1f} rad")
    best = min(simulated, key=simulated.get, default=None)
    if best is None or simulated[best] == np.inf:
        print("no plan survives simulation")
        return False, []
    print("executing the plan for", best)
    return execute_with_monitoring(best, domain, numbers, estimator=estimate_state)


# --- learning from simulation ---

DOMAIN_WITH_LESSONS = '''
(define (domain n-table-blocks-world)
  (:requirements :strips :typing :action-costs :negative-preconditions :existential-preconditions)
  (:types block color table)
  (:predicates (on ?b1 ?b2 - block)
               (ontable ?b - block ?t - table)
               (clear ?b - block)
               (handempty)
               (holding ?b - block)
               (bcolor ?b - block ?c - color)
               (hasroom ?t - table)
               (cannot-pick ?b - block)
               (cannot-stack-on ?b - block))
  (:functions (total-cost) - number
              (access-cost ?t - table) - number
              (stack-cost) - number)

  (:action pick-up
     :parameters (?b - block ?t - table)
     :precondition (and (clear ?b) (ontable ?b ?t) (handempty) (not (cannot-pick ?b)))
     :effect (and (not (ontable ?b ?t)) (not (clear ?b)) (not (handempty)) (holding ?b) (hasroom ?t)
                  (not (cannot-stack-on ?b))
                  (increase (total-cost) (access-cost ?t))))

  (:action put-down
     :parameters (?b - block ?t - table)
     :precondition (and (holding ?b) (hasroom ?t))
     :effect (and (not (holding ?b)) (clear ?b) (handempty) (ontable ?b ?t)
                  (increase (total-cost) (access-cost ?t))))

  (:action stack
     :parameters (?b1 ?b2 - block)
     :precondition (and (holding ?b1) (clear ?b2) (not (cannot-stack-on ?b2)))
     :effect (and (not (holding ?b1)) (not (clear ?b2)) (clear ?b1) (handempty) (on ?b1 ?b2)
                  (increase (total-cost) (stack-cost))))

  (:action unstack
     :parameters (?b1 ?b2 - block)
     :precondition (and (on ?b1 ?b2) (clear ?b1) (handempty) (not (cannot-pick ?b1)))
     :effect (and (holding ?b1) (clear ?b2) (not (clear ?b1)) (not (handempty)) (not (on ?b1 ?b2))
                  (not (cannot-stack-on ?b1))
                  (increase (total-cost) (stack-cost))))
)
'''


def lesson(action):
    """The fact that the robot learns when the action turns out to be infeasible, or None."""
    name, args = action.action.name, [str(p) for p in action.actual_parameters]
    #TODO return the fact that the robot learns from an infeasible pick-up, unstack or stack (a tuple), and None from an
    # infeasible put-down
    ...


LESSONS = {}  # fact -> the position of its block when the robot learned it


def remembered_lessons():
    """The lessons about blocks that have not moved since the robot learned them."""
    return {fact for fact, position in LESSONS.items()
            if fact[1] in scene.BLOCKS and np.linalg.norm(scene.data.body(fact[1]).xpos - position) < 0.01}


def estimate_state_with_lessons():
    """The abstract state of the world: the facts of Task 1, which tables have room, and what the robot has learned."""
    return estimate_state() | remembered_lessons()


def add_fact(problem, fact):
    """Make a fact true in the initial state of a planning problem."""
    fluent = problem.fluent(fact[0])
    problem.set_initial_value(fluent(*[problem.object(name) for name in fact[1:]]), True)


def plan_with_simulation(problem, max_lessons=5):
    """Propose a plan, simulate it, and learn from the action that is infeasible, until a plan survives. Returns a plan or None."""
    for _ in range(max_lessons + 1):
        #TODO propose a plan for the problem (see `solve`). If there is none, return None
        task_plan = ...

        #TODO simulate the plan (see `simulate_plan`)
        costs = ...
        steps = []
        for action, cost in zip(task_plan, costs):
            steps.append(f"{action} " + ("INFEASIBLE" if cost is None else f"{cost:.1f}"))
            if cost is None:
                break
        print("  simulated: " + ", ".join(steps))

        #TODO if every action of the plan is feasible, the plan survived simulation: return it
        ...

        # the first infeasible action, and the blocks that the plan moved before it
        failed = costs.index(None)
        moved = {str(action.actual_parameters[0]) for action in task_plan[:failed]}

        #TODO find what the robot learns from the failed action (see `lesson`). If it learns nothing, or the lesson is
        # about a block that the plan moved before the failure, print "  nothing to learn" and return None
        fact = ...

        print("  learned:", fact_string(fact))
        #TODO remember the lesson in LESSONS, with the current position of its block, and tell the task planner: make
        # the fact true in the initial state of the problem (see `add_fact`)
        ...
    return None
