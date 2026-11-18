"""Homework 2, Task 1: Closing the loop.

This module provides:
- The task planner's model of the N-Table Blocks World, and conversions between our facts
  and planning problems.
- The state estimator (the true state, read from the simulator's contacts).
- Skills: from actions to motions.
- The sampler for `put-down`: free spots, reachable spots, and both together.
- Open-loop execution of plans, and the samplers and the adversary that make it fail.
- The facts that the task planner expects after each action of a plan.
- Execution monitoring and replanning.
"""

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, SequentialSimulator, get_environment

# NOTE: `scene.load_scene` replaces the model, the data and the blocks, so we always read
# them through the module, as `scene.model`, `scene.data` and `scene.BLOCKS`.
import scene
from scene import (
    BLOCK_HALF_HEIGHT,
    BLOCK_HALF_WIDTH,
    COLORS,
    FACING_DOWN,
    TABLES,
    analytical_ik,
    body_of_geom,
    get_block_positions,
    get_configuration,
    go_to,
    grasp,
    grasped_block,
    in_collision,
    move_to,
    nearest_equivalent,
    release,
    render,
)

get_environment().credits_stream = None  # keep the output clean (planners print their credits otherwise)


# --- the task planner's model ---

NTBW_DOMAIN = """
(define (domain n-table-blocks-world)
  (:requirements :strips :typing)
  (:types block color table)
  (:predicates (on ?b1 ?b2 - block)
               (ontable ?b - block ?t - table)
               (clear ?b - block)
               (handempty)
               (holding ?b - block)
               (bcolor ?b - block ?c - color))

  (:action pick-up
     :parameters (?b - block ?t - table)
     :precondition (and (clear ?b) (ontable ?b ?t) (handempty))
     :effect (and (not (ontable ?b ?t)) (not (clear ?b)) (not (handempty)) (holding ?b)))

  (:action put-down
     :parameters (?b - block ?t - table)
     :precondition (holding ?b)
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
"""

PROBLEM_TEMPLATE = """
(define (problem ntbw)
  (:domain n-table-blocks-world)
  (:objects {blocks} - block  {colors} - color  {tables} - table)
  (:init {init})
  (:goal {goal})
  {metric})
"""


# --- the state estimator: the true state, read from the simulator's contacts ---

def true_state():
    """The facts that actually hold in the simulation, from the contacts between the blocks and what they rest on."""
    held = grasped_block()
    support = {}  # block -> the block or table it rests on
    for i in range(scene.data.ncon):
        contact = scene.data.contact[i]
        body1, body2 = body_of_geom(contact.geom1), body_of_geom(contact.geom2)
        normal = contact.frame[:3]  # the contact normal, pointing from geom1 to geom2
        for upper, lower, direction in [(body2, body1, normal), (body1, body2, -normal)]:
            if upper in scene.BLOCKS and upper != held and lower in scene.BLOCKS + TABLES and direction[2] > 0.9:
                support[upper] = lower
    facts = {("holding", held)} if held is not None else {("handempty",)}
    for block, lower in support.items():
        facts.add(("ontable", block, lower) if lower in TABLES else ("on", block, lower))
    for block in scene.BLOCKS:
        if block != held and block not in support.values():
            facts.add(("clear", block))
    return facts


def settle(duration=0.2):
    """Let the simulation run (with the servos holding their targets) for the given time."""
    for _ in range(round(duration / scene.model.opt.timestep)):
        mujoco.mj_step(scene.model, scene.data)


def estimate_state():
    """The abstract state of the world, as our state estimator sees it."""
    settle()
    return true_state()


def fact_string(fact):
    return "(" + " ".join(fact) + ")"


def print_state(facts):
    print("  " + "  ".join(fact_string(fact) for fact in sorted(facts)))


# --- facts in and out of the planner ---

def facts_of(state, problem):
    """The facts that hold in a state of the planner, as tuples (without the colors, which never change)."""
    facts = set()
    for expression in problem.initial_values:  # lists every grounded fluent, which is a convenient way to enumerate them all
        if expression.type.is_bool_type() and expression.fluent().name != "bcolor" and state.get_value(expression).is_true():
            facts.add((expression.fluent().name,) + tuple(str(arg) for arg in expression.args))
    return facts


def goal_facts(problem):
    """The facts of the (conjunctive) goal of a problem, as tuples."""
    facts = set()
    for goal in problem.goals:
        for expression in (goal.args if goal.is_and() else [goal]):
            facts.add((expression.fluent().name,) + tuple(str(arg) for arg in expression.args))
    return facts


def goal_holds(problem):
    """Whether the goal of a problem holds in its initial state."""
    with SequentialSimulator(problem) as simulator:
        return simulator.is_goal(simulator.get_initial_state())


def make_problem(facts, goal, domain=NTBW_DOMAIN, numbers=None):
    """A planning problem whose initial state is the given set of facts and whose goal is the given PDDL expression.

    numbers gives the values of numeric functions, for domains with action costs: {("access-cost", "t1"): 5, ...}.
    """
    color_facts = {("bcolor", block, color) for block, color in scene.BLOCK_COLORS.items()}
    init = " ".join(fact_string(fact) for fact in sorted(facts | color_facts))
    metric = ""
    if numbers is not None:
        init += " (= (total-cost) 0) " + " ".join(f"(= {fact_string(function)} {value})" for function, value in numbers.items())
        metric = "(:metric minimize (total-cost))"
    text = PROBLEM_TEMPLATE.format(blocks=" ".join(scene.BLOCKS), colors=" ".join(COLORS), tables=" ".join(TABLES), init=init, goal=goal, metric=metric)
    return PDDLReader().parse_problem_string(domain, text)


def solve(problem):
    """An optimal plan for the problem (a list of action instances), or None if there is none."""
    with OneshotPlanner(name="fast-downward-opt") as planner:
        planner.skip_checks = True  # needed for action costs
        result = planner.solve(problem)
    return None if result.plan is None else list(result.plan.actions)


# --- skills: from actions to motions ---

APPROACH_HEIGHT = 0.1  # meters: the tool hovers this high above the target before descending straight down
GRASP_CLEARANCE = 0.005  # the suction cup grasps from 5 mm above the block
PLACE_CLEARANCE = 0.01  # and lets go from 1 cm above the surface


def block_top(block):
    """The center of the top face of a block."""
    return scene.data.body(block).xpos + [0, 0, BLOCK_HALF_HEIGHT]


def approach_configurations(target, low, high=APPROACH_HEIGHT):
    """Configurations for being `low` above the target (grasping or releasing) and for hovering `high` above it.

    Both are collision-free and in the same IK family, so that the straight-line motion between them is short (the
    descent and the lift). The low configuration is chosen first, as close as possible to where the robot is now, and
    the hovering one is the solution closest to it; if that is far (another family), the next low candidate is tried.
    Returns (q_high, q_low), or (None, None) if there is no such pair.
    """
    reference = get_configuration()
    lows = [nearest_equivalent(q, reference) for q in analytical_ik(target + [0, 0, low], FACING_DOWN) if not in_collision(q)]
    highs = [q for q in analytical_ik(target + [0, 0, high], FACING_DOWN) if not in_collision(q)]
    for q_low in sorted(lows, key=lambda q: np.linalg.norm(q - reference)):
        candidates = [nearest_equivalent(q, q_low) for q in highs]
        q_high = min(candidates, key=lambda q: np.linalg.norm(q - q_low), default=None)
        if q_high is not None and np.linalg.norm(q_high - q_low) < 1.0:
            return q_high, q_low
    return None, None


def pick_configurations(block):
    """The hovering and grasping configurations for picking up a block, or (None, None)."""
    return approach_configurations(block_top(block), GRASP_CLEARANCE)


def place_configurations(target_top):
    """The hovering and releasing configurations for putting the held block down with its bottom on target_top, or (None, None)."""
    block_height = 2 * BLOCK_HALF_HEIGHT
    return approach_configurations(target_top, block_height + PLACE_CLEARANCE, block_height + APPROACH_HEIGHT)


def pick_up(block):
    """The skill for pick-up and unstack: approach the block from above, descend, grasp, lift. Returns (success, frames)."""
    q_high, q_low = pick_configurations(block)
    if q_high is None:
        print(f"  cannot reach {block}")
        return False, []
    success, frames = go_to(q_high)  # planned motion, possibly around obstacles
    if not success:
        return False, frames
    success, new_frames = move_to(q_low)  # straight down
    frames += new_frames
    if not success:
        return False, frames
    grasped = grasp()
    success, new_frames = move_to(q_high)  # straight up
    return success and grasped == block, frames + new_frames


def place(target_top):
    """The skill for put-down and stack: put the held block down with its bottom on target_top (a point in the world)."""
    q_high, q_low = place_configurations(target_top)
    if q_high is None:
        print("  cannot reach the target")
        return False, []
    success, frames = go_to(q_high)
    if not success:
        return False, frames
    success, new_frames = move_to(q_low)
    frames += new_frames
    if not success:  # something is in the way: keep the block and back off
        _, new_frames = move_to(q_high)
        return False, frames + new_frames
    release()
    success, new_frames = move_to(q_high)
    return success, frames + new_frames


def stack(block, on_block):
    """The skill for stack: put the held block on top of on_block."""
    return place(block_top(on_block))


# --- the sampler: where to put a block down ---

MARGIN = 0.02  # a safety margin between blocks
MAX_REACH = 0.8  # meters from the robot's base: further out, the arm is stretched to its limit


def table_surface(table):
    """The height of the top face of a table, and the (x, y) center and half extents of the top."""
    top = scene.model.geom(f"{table}_top")
    return scene.data.geom(f"{table}_top").xpos[2] + top.size[2], scene.data.geom(f"{table}_top").xpos[:2].copy(), top.size[:2].copy()


def table_point(table, xy):
    """The point on the surface of the table at (x, y)."""
    return np.array([xy[0], xy[1], table_surface(table)[0]])


def sample_on_table(table):
    """A uniformly random (x, y) on the table, far enough from the edges for a block to sit fully on it."""
    _, center, half_size = table_surface(table)
    return center + np.random.uniform(-1, 1, size=2) * (half_size - BLOCK_HALF_WIDTH)


def is_free(xy, exclude=None):
    """Whether a block put down at (x, y) would overlap none of the blocks on the tables (with a safety margin)."""
    for block, position in get_block_positions().items():
        if block != exclude and np.all(np.abs(position[:2] - xy) < 2 * BLOCK_HALF_WIDTH + MARGIN):
            return False
    return True


def is_reachable(table, xy):
    """Whether the spot is well within the arm's reach and the robot has collision-free configurations for placing a block there."""
    #TODO a spot further than MAX_REACH from the robot's base (at the origin) is not reachable
    ...

    #TODO otherwise, the spot is reachable if there are configurations for placing a block on it (see
    # `place_configurations` and `table_point`)
    reachable = ...
    return reachable


def sample_free_spot(table, block, max_attempts=1000):
    """A free, reachable (x, y) on the table for the block, or None if none was found in max_attempts samples."""
    for _ in range(max_attempts):
        xy = sample_on_table(table)
        if is_free(xy, exclude=block) and is_reachable(table, xy):
            return xy
    return None


def put_down(block, table, xy):
    """The skill for put-down: put the held block on the table at (x, y)."""
    return place(table_point(table, xy))


def draw_map(spots=(), colors=(), title=None):
    """A top view of the workspace: tables, blocks, the robot's base and the wall, with the given spots marked."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for table in TABLES:
        _, center, half_size = table_surface(table)
        ax.add_patch(plt.Rectangle(center - half_size, 2 * half_size[0], 2 * half_size[1], fill=False, color="gray"))
        ax.text(*(center + [0, half_size[1] - 0.04]), table, ha="center", va="center", color="gray")
    for block in scene.BLOCKS:
        x, y, z = scene.data.body(block).xpos
        ax.add_patch(plt.Rectangle((x - BLOCK_HALF_WIDTH, y - BLOCK_HALF_WIDTH), 2 * BLOCK_HALF_WIDTH, 2 * BLOCK_HALF_WIDTH, color=scene.model.geom(block).rgba[:3], zorder=z))
        ax.text(x, y, block, ha="center", va="center", fontsize=7, zorder=z + 0.1)
    ax.add_patch(plt.Circle((0, 0), 0.12, color="k"))
    ax.add_patch(plt.Circle((0, 0), MAX_REACH, fill=False, color="k", linestyle=":"))  # the limit of the arm's reach
    ax.plot([-0.3, 0.3], [-0.34, -0.34], color="saddlebrown", linewidth=4)  # the wall with the window
    spots = np.asarray(spots)
    if len(spots):
        ax.scatter(spots[:, 0], spots[:, 1], c=colors, s=14, zorder=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    plt.show()


# --- executing plans ---

def execute_action(action, sampler=sample_free_spot):
    """Run the skill that implements one action instance of the task planner. Returns (success, frames)."""
    name, args = action.action.name, [str(p) for p in action.actual_parameters]
    if name in ("pick-up", "unstack"):
        return pick_up(args[0])
    if name == "stack":
        return stack(args[0], args[1])
    if name == "put-down":
        block, table = args[0], args[1]
        xy = sampler(table, block)
        if xy is None:
            print(f"  no free spot on {table}")
            return False, []
        return put_down(block, table, xy)
    raise ValueError(f"no skill for the action {name}")


def caption(frames, text, color=(255, 255, 255)):
    """Write a line of text along the top of every frame (in place). Returns the frames."""
    for frame in frames:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return frames


def freeze(text, seconds=2, color=(255, 200, 0)):
    """A still shot of the scene as it is now, with a caption, to hold in a video (at 30 frames per second)."""
    return caption([render()], text, color) * int(30 * seconds)


def execute_plan(task_plan, sampler=sample_free_spot, after_action=None):
    """Execute the actions of a plan in order, stopping at the first failure. Returns (success, frames).

    after_action(step) is called after every action, with the number of actions executed so far; it returns video
    frames (possibly none). The frames of every action are captioned with the action.
    """
    frames = []
    for i, action in enumerate(task_plan):
        success, new_frames = execute_action(action, sampler)
        frames += caption(new_frames, f"{i + 1}. {action}")
        print(f"  {i + 1}. {action}: {'ok' if success else 'FAILED'}")
        if not success:
            return False, frames
        if after_action is not None:
            frames += after_action(i + 1)
    return True, frames


# --- when things go wrong ---

proposals = []  # the spots that the naive sampler proposed


def naive_sampler(table, block):
    """Any spot on the table, without checking whether it is free or reachable."""
    xy = sample_on_table(table)
    proposals.append(xy)
    return xy


edge_spots = []  # the spots that the careless sampler proposed


def careless_sampler(table, block, max_attempts=100):
    """A free, reachable spot for the block, but the center may be up to half a block beyond the edges of the table."""
    _, center, half_size = table_surface(table)
    for _ in range(max_attempts):
        xy = center + np.random.uniform(-1, 1, size=2) * (half_size + BLOCK_HALF_WIDTH)  # the bug: + instead of -
        if is_free(xy, exclude=block) and is_reachable(table, xy):
            edge_spots.append(xy)
            return xy
    return None


def move_block(block, position):
    """The hand of our adversary: put a block at the given position (its center), upright. A privilege of simulation."""
    joint = scene.model.joint(f"{block}_joint")
    scene.data.qpos[joint.qposadr[0]:joint.qposadr[0] + 3] = position
    scene.data.qpos[joint.qposadr[0] + 3:joint.qposadr[0] + 7] = [1, 0, 0, 0]
    scene.data.qvel[joint.dofadr[0]:joint.dofadr[0] + 6] = 0
    mujoco.mj_forward(scene.model, scene.data)


def adversary_moves(block, position, text):
    """Our adversary moves a block. Returns video frames that stop to show the scene before and after."""
    print(f"  ({text})")
    before = freeze(text)
    move_block(block, position)
    return before + freeze(text)


def tower_adversary(step):
    """After the fourth action, our adversary moves b1 (wherever it is) to t3. Returns video frames."""
    if step != 4:
        return []
    height, _, _ = table_surface("t3")
    return adversary_moves("b1", [-0.15, 0.78, height + BLOCK_HALF_HEIGHT], "our adversary moves b1 to t3")


def adversary(step):
    """After the second action, our adversary puts b2 back on top of b1. Returns video frames."""
    if step != 2:
        return []
    return adversary_moves("b2", block_top("b1") + [0, 0, BLOCK_HALF_HEIGHT], "our adversary puts b2 back on b1")


# --- closing the loop: monitor and replan ---

def expected_states(problem, task_plan):
    """The facts that the task planner expects to hold after each action of the plan (a list of sets)."""
    states = []
    #TODO simulate the plan with a SequentialSimulator for the problem: start from its initial state, apply the actions
    # one after the other, and append the facts that hold after each action to `states` (see `facts_of`)
    ...
    return states


def execute_with_monitoring(goal, domain=NTBW_DOMAIN, numbers=None, max_replans=5, sampler=sample_free_spot,
                            after_action=None, planner=solve, estimator=estimate_state):
    """Plan, execute, check after every action, and replan when needed. Returns (success, frames).

    Args:
        goal: the goal, as a PDDL expression.
        domain, numbers: passed on to `make_problem`.
        max_replans: how many times to plan again before giving up.
        sampler: passed on to `execute_action`.
        after_action: if given, after_action(steps) is called after every action, with the number of actions executed
            so far (across all plans), and returns video frames to add to ours.
        planner: turns a problem into a plan (a list of action instances), or None if there is none.
        estimator: returns the facts that hold in the world.
    """
    frames = []
    steps = 0  # actions executed so far
    for attempt in range(max_replans + 1):
        facts = estimator()
        problem = make_problem(facts, goal, domain, numbers)

        #TODO if the goal holds in the estimated state (see `goal_holds`), print "goal achieved" and return success,
        # with the frames so far
        ...

        #TODO plan from the estimated state with `planner`. If there is no plan, print
        # "no plan achieves the goal from this state" and return failure, with the frames so far
        task_plan = ...

        print(f"plan {attempt + 1} ({len(task_plan)} actions): " + ", ".join(map(str, task_plan)))
        for action, expected in zip(task_plan, expected_states(problem, task_plan)):
            #TODO execute the action with `sampler` (see `execute_action`), and add its frames to ours. Count the action
            # in `steps`, and if `after_action` is given, call it with `steps` and add the frames it returns
            success = ...

            #TODO estimate the state again with `estimator`. If the skill succeeded and the state is the expected one,
            # print f"  {action}: ok" and continue to the next action
            facts = ...

            # the plan is no longer valid: report why, and plan again
            print(f"  {action}: {'the skill failed' if not success else 'unexpected state'}")
            for fact in sorted(expected - facts):
                print("    missing:   ", fact_string(fact))
            for fact in sorted(facts - expected):
                print("    unexpected:", fact_string(fact))
            break  # replan from the state we are in
    print("giving up")
    return False, frames
