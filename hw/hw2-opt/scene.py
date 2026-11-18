"""Homework 2 simulation scene. This file is provided, do not change it.

This module provides:
- The UR5e robot model, downloaded from the MuJoCo Menagerie on first use.
- The N-Table Blocks World: three tables, a wall with a window in front of table t1, and
  a UR5e with a suction cup. `load_scene(layout)` builds the world with any set of blocks.
- The tools from the previous tutorials: rendering, moving the robot (`move_to`,
  `follow_path`, `go_to`), inverse kinematics (`analytical_ik`), collision checking
  (`in_collision`), the suction gripper (`grasp`, `release`, `grasped_block`), and the
  OMPL-based motion planner (`plan`).

`load_scene` replaces `model`, `data`, `BLOCKS` and `BLOCK_COLORS`. Always read them
through the module (e.g. `scene.data`), never with `from scene import data`, which keeps
the old ones.
"""

from pathlib import Path
import random
import subprocess
import time

import mediapy as media
import mujoco
import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
from ur_analytic_ik import ur5e

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE_PATH = Path(__file__).parent / "mujoco_menagerie"
ROBOT_DIR = "universal_robots_ur5e"

# Download the UR5e robot MJCF files from DeepMind's MuJoCo Menagerie. The full
# repository takes up a few GB, so we only check out the model we use.
if not MENAGERIE_PATH.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         MENAGERIE_URL, str(MENAGERIE_PATH)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(MENAGERIE_PATH), "sparse-checkout", "set", ROBOT_DIR],
        check=True,
    )

UR5E_XML = str(MENAGERIE_PATH / ROBOT_DIR / "ur5e.xml")
SCENE_XML = """
<mujoco model="ntbw">
  <option integrator="implicitfast"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="table" rgba="0.75 0.75 0.75 1"/>
    <material name="pedestal" rgba="0.2 0.2 0.2 1"/>
  </asset>

  <!-- MuJoCo box sizes are HALF sizes: table tops are 60cm x 60cm x 2cm -->
  <default>
    <default class="table_top">
      <geom type="box" size="0.3 0.3 0.01" pos="0 0 0.71" material="table"/>
    </default>
    <default class="table_leg">
      <geom type="cylinder" size="0.02 0.35" material="table"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- the robot will be attached to this site, at the top of the pedestal (turned to face table t2) -->
    <body name="mount" pos="0 0 0.75" euler="0 0 180">
      <geom name="pedestal" type="cylinder" size="0.12 0.375" pos="0 0 -0.375" material="pedestal"/>
      <site name="mount_site"/>
    </body>

    <!-- three tables, named as in our PDDL problems -->
    <body name="t1" pos="0 -0.65 0">
      <geom name="t1_top" class="table_top"/>
      <geom class="table_leg" pos="0.26 -0.26 0.35"/>
      <geom class="table_leg" pos="0.26 0.26 0.35"/>
      <geom class="table_leg" pos="-0.26 -0.26 0.35"/>
      <geom class="table_leg" pos="-0.26 0.26 0.35"/>
    </body>
    <body name="t2" pos="0.65 0 0">
      <geom name="t2_top" class="table_top"/>
      <geom class="table_leg" pos="0.26 -0.26 0.35"/>
      <geom class="table_leg" pos="0.26 0.26 0.35"/>
      <geom class="table_leg" pos="-0.26 -0.26 0.35"/>
      <geom class="table_leg" pos="-0.26 0.26 0.35"/>
    </body>
    <body name="t3" pos="0 0.65 0">
      <geom name="t3_top" class="table_top"/>
      <geom class="table_leg" pos="0.26 -0.26 0.35"/>
      <geom class="table_leg" pos="0.26 0.26 0.35"/>
      <geom class="table_leg" pos="-0.26 -0.26 0.35"/>
      <geom class="table_leg" pos="-0.26 0.26 0.35"/>
    </body>

    <!-- cameras to watch the scene from -->
    <camera name="overview" mode="targetbody" target="mount" pos="1.9 -1.9 1.9"/>
    <camera name="front" mode="targetbody" target="mount" pos="2.4 0 1.5"/>
    <camera name="side" mode="targetbody" target="mount" pos="0 -2.4 1.5"/>
    <camera name="top" pos="0 0 3.0" xyaxes="0 1 0 -1 0 0"/>
    <!-- an RGB-D camera on the ceiling, looking straight down at the three tables -->
    <camera name="ceiling" pos="0.3 0 2.05" xyaxes="0 1 0 -1 0 0" fovy="58"/>
  </worldbody>
</mujoco>
"""

TABLES = ["t1", "t2", "t3"]
BLOCK_HALF_HEIGHT = 0.02
BLOCK_HALF_WIDTH = 0.04  # the blocks are 8cm x 8cm x 4cm
COLORS = {"yellow": [1, 0.9, 0, 1], "blue": [0.1, 0.3, 1, 1], "green": [0.1, 0.8, 0.2, 1], "red": [1, 0.1, 0.1, 1]}

# a layout says which blocks there are: name -> (color, position of the center). This is our default scene.
DEFAULT_LAYOUT = {
    "b1": ("yellow", [0.05, -0.6, 0.74]),
    "b2": ("blue", [0.05, -0.6, 0.78]),
    "b3": ("yellow", [0.6, 0.1, 0.74]),
    "b4": ("green", [-0.1, 0.55, 0.74]),
    "b5": ("red", [0.15, 0.7, 0.74]),
}

TOOL_XML = """
<mujoco model="suction_gripper">
  <worldbody>
    <body name="suction_cup">
      <geom name="suction_cup" type="cylinder" size="0.02 0.03" pos="0 0 0.03" rgba="0.3 0.3 0.3 1" mass="0.05"/>
      <site name="tcp" pos="0 0 0.06" size="0.005" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# a wall with a window (a serving hatch) between the robot and table t1: name, half sizes, position
WINDOW_GEOMS = [
    ("window_sill", [0.30, 0.01, 0.10], [0.00, -0.34, 0.82]),
    ("window_post_left", [0.02, 0.02, 0.30], [-0.28, -0.34, 1.02]),
    ("window_post_right", [0.02, 0.02, 0.30], [0.28, -0.34, 1.02]),
    ("window_lintel", [0.30, 0.02, 0.02], [0.00, -0.34, 1.30]),
]


def build_model(layout, window=True):
    scene = mujoco.MjSpec.from_string(SCENE_XML)
    robot = mujoco.MjSpec.from_file(UR5E_XML)
    tool = mujoco.MjSpec.from_string(TOOL_XML)

    scene.attach(robot, site="mount_site", prefix="robot_")
    scene.attach(tool, site="robot_attachment_site", prefix="tool_")

    for block, (color, position) in layout.items():
        body = scene.worldbody.add_body(name=block, pos=position)
        body.add_freejoint(name=f"{block}_joint")  # a freejoint lets a body move freely
        body.add_geom(name=block, type=mujoco.mjtGeom.mjGEOM_BOX, size=[BLOCK_HALF_WIDTH, BLOCK_HALF_WIDTH, BLOCK_HALF_HEIGHT],
                      mass=0.1, rgba=COLORS[color])
        # one (inactive) weld constraint per block, for our simplified suction gripper
        scene.add_equality(name=f"grasp_{block}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                           name1="tool_suction_cup", name2=block, active=False)

    if window:
        for name, size, pos in WINDOW_GEOMS:
            scene.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, rgba=[0.6, 0.4, 0.2, 1])

    return scene.compile()


ROBOT_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ROBOT_ACTUATORS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
HOME = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
renderer = None


def load_scene(layout=DEFAULT_LAYOUT, window=True):
    """Build the world with the given blocks, and reset it. Replaces model, data, BLOCKS and BLOCK_COLORS."""
    global BLOCKS, BLOCK_COLORS, model, data, renderer, qpos_idx, qvel_idx, ctrl_idx
    BLOCKS = list(layout)
    BLOCK_COLORS = {block: color for block, (color, _) in layout.items()}
    model = build_model(layout, window)  # MjModel: everything that does not change during simulation (geometry, masses, ...)
    data = mujoco.MjData(model)  # MjData: everything that does (positions, velocities, contacts, ...)
    if renderer is not None:
        renderer.close()
    renderer = mujoco.Renderer(model, height=480, width=640)
    qpos_idx = np.array([model.joint("robot_" + name).qposadr[0] for name in ROBOT_JOINTS])
    qvel_idx = np.array([model.joint("robot_" + name).dofadr[0] for name in ROBOT_JOINTS])
    ctrl_idx = np.array([model.actuator("robot_" + name).id for name in ROBOT_ACTUATORS])
    reset()


def render(camera="overview", scene_option=None):
    """An RGB image of the current state of the simulation, seen from the named camera."""
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    return renderer.render()


def show_video(frames, fps=30):
    if len(frames) == 0:
        print("(no frames to show)")
    else:
        media.show_video(frames, fps=fps)


def get_configuration():
    """The robot's joint angles (radians)."""
    return data.qpos[qpos_idx].copy()


def get_joint_velocities():
    return data.qvel[qvel_idx].copy()


def get_block_positions():
    """The position of the center of every block, in world coordinates."""
    return {block: data.body(block).xpos.copy() for block in BLOCKS}


def move_to(target, tolerance=0.03, max_steps=2000, render_every=50, camera="overview"):
    """Apply the target configuration until the robot reaches it (within tolerance, radians) and stops moving.

    Returns (success, frames): whether the target was reached, and video frames of the motion.
    """
    frames = []
    for i in range(max_steps):
        data.ctrl[ctrl_idx] = target
        mujoco.mj_step(model, data)
        if i % render_every == 0:
            frames.append(render(camera))
        if np.linalg.norm(get_configuration() - target) < tolerance and np.linalg.norm(get_joint_velocities()) < 0.05:
            return True, frames
    return False, frames


def follow_path(waypoints, tolerance=0.1, max_steps_per_waypoint=500, render_every=50, camera="overview"):
    """Move the robot along the path (switch to the next waypoint when close enough). Returns (success, frames)."""
    frames = []
    step = 0
    for i, target in enumerate(waypoints):
        if i == len(waypoints) - 1:  # stop precisely at the last waypoint
            success, new_frames = move_to(target, render_every=render_every, camera=camera)
            return success, frames + new_frames
        for _ in range(max_steps_per_waypoint):
            data.ctrl[ctrl_idx] = target
            mujoco.mj_step(model, data)
            if step % render_every == 0:
                frames.append(render(camera))
            step += 1
            if np.linalg.norm(get_configuration() - target) < tolerance:
                break  # close enough, on to the next waypoint
        else:
            return False, frames  # the robot got stuck
    return True, frames


def go_to(q_goal):
    """Plan a path from the current configuration to q_goal and follow it. Returns (success, frames)."""
    waypoints = plan(get_configuration(), q_goal, verbose=False)
    if waypoints is None:
        print("no path found")
        return False, []
    return follow_path(waypoints)


def reset():
    """Reset the simulation to the initial state, with the robot at the home configuration."""
    mujoco.mj_resetData(model, data)
    data.qpos[qpos_idx] = HOME
    data.ctrl[ctrl_idx] = HOME  # the home configuration as the servos' target
    data.eq_active[:] = 0  # no block is grasped
    mujoco.mj_forward(model, data)


def get_tcp_position():
    return data.site("tool_tcp").xpos.copy()


def get_tcp_pose():
    """Position (3,) and orientation (3x3 rotation matrix) of the tool center point in the world frame."""
    site = data.site("tool_tcp")
    return site.xpos.copy(), site.xmat.reshape(3, 3).copy()


def transformation(position, quaternion):
    """The 4x4 homogeneous transformation of a pose given as position and unit quaternion (w, x, y, z)."""
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    T = np.eye(4)
    T[:3, :3] = rotation.reshape(3, 3)
    T[:3, 3] = position
    return T


def wrap_angles(q):
    """Wrap angles to [-pi, pi]. Joint angles are periodic, and this keeps them inside the robot's joint limits."""
    return (q + np.pi) % (2 * np.pi) - np.pi


FACING_DOWN = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])



def world_to_base(position, rotation):
    """Transform a pose from world coordinates to the robot's base frame."""
    base = data.body("mount")
    base_position, base_rotation = base.xpos, base.xmat.reshape(3, 3)
    return base_rotation.T @ (position - base_position), base_rotation.T @ rotation


T_TOOL = np.eye(4)
T_TOOL[2, 3] = 0.06  # the tcp site relative to the flange: 6 cm out, at the tip of the suction cup


def analytical_ik(target_position, target_rotation):
    """All configurations that bring the tool center point to the target pose (world frame). Empty if unreachable."""
    position, rotation = world_to_base(target_position, target_rotation)
    T_flange = np.eye(4)
    T_flange[:3, :3] = rotation
    T_flange[:3, 3] = position
    return ur5e.inverse_kinematics_with_tcp(T_flange, T_TOOL)


def grasped_block():
    """The block currently held by the suction cup, or None."""
    for block in BLOCKS:
        if data.eq_active[model.equality(f"grasp_{block}").id]:
            return block
    return None


def body_of_geom(geom_id):
    """The name of the body a geom belongs to (the robot's collision geoms have no names of their own)."""
    return model.body(model.geom(geom_id).bodyid[0]).name


def nearest_equivalent(q, reference):
    """Shift each angle of q by a multiple of 2*pi to the equivalent angle nearest to reference, within [-pi, pi].

    (The UR5e joints turn +-2*pi, but we keep every configuration within the bounds of the motion planner, so that a
    planned motion can start wherever the previous one ended.)
    """
    shifted = reference + wrap_angles(q - reference)
    within_limits = (shifted >= JOINT_BOUNDS[:, 0]) & (shifted <= JOINT_BOUNDS[:, 1])
    return np.where(within_limits, shifted, wrap_angles(q))


def choose_solution(solutions, reference):
    """The collision-free solution closest to the reference configuration, or None if there is none."""
    free = [nearest_equivalent(q, reference) for q in solutions if not in_collision(q)]
    if not free:
        return None
    return min(free, key=lambda q: np.linalg.norm(q - reference))


def pose_above(block, height=0.1):
    """The tcp pose `height` meters above the top of the block, facing down."""
    top = data.body(block).xpos + [0, 0, BLOCK_HALF_HEIGHT]
    return top + [0, 0, height], FACING_DOWN


def grasp(max_distance=0.03):
    """Activate the suction: weld the block whose top is within max_distance of the tcp to the cup. Returns the block or None."""
    tcp = get_tcp_position()
    cup = data.body("tool_suction_cup")
    for block in BLOCKS:
        body = data.body(block)
        if np.linalg.norm(body.xpos + [0, 0, BLOCK_HALF_HEIGHT] - tcp) < max_distance:
            # the weld keeps the block at its current pose relative to the cup
            relative_position = cup.xmat.reshape(3, 3).T @ (body.xpos - cup.xpos)
            cup_quat_inverse, relative_quat = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(cup_quat_inverse, cup.xquat)
            mujoco.mju_mulQuat(relative_quat, cup_quat_inverse, body.xquat)
            equality = model.equality(f"grasp_{block}")
            # anchor, relative pose, torque scale, and (MuJoCo >= 3.17) a zero flag: we set the relative pose directly
            model.eq_data[equality.id] = np.concatenate([[0, 0, 0], relative_position, relative_quat, [1.0], np.zeros(mujoco.mjNEQDATA - 11)])
            data.eq_active[equality.id] = 1
            return block
    return None


def release():
    """Deactivate the suction."""
    data.eq_active[:] = 0


def interpolate(q_from, q_to, n_steps):
    """n_steps configurations evenly spaced on the straight line from q_from to q_to (inclusive)."""
    return [q_from + t * (q_to - q_from) for t in np.linspace(0, 1, n_steps)]


def path_in_collision(q_from, q_to, n_steps=50):
    """Whether the straight line in configuration space from q_from to q_to passes through a collision."""
    return any(in_collision(q) for q in interpolate(q_from, q_to, n_steps))

# --- planning while holding a block ---

def set_configuration(q):
    """Teleport the robot, and the block it holds (if any), to configuration q. No physics is simulated."""
    data.qpos[qpos_idx] = q
    block = grasped_block()
    if block is not None:
        mujoco.mj_kinematics(model, data)  # where is the cup now?
        cup = data.body("tool_suction_cup")
        weld_data = model.eq_data[model.equality(f"grasp_{block}").id]
        relative_position, relative_quat = weld_data[3:6], weld_data[6:10]  # the block's pose in the cup's frame

        position = cup.xpos + cup.xmat.reshape(3, 3) @ relative_position
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, cup.xquat, relative_quat)

        address = model.joint(f"{block}_joint").qposadr[0]  # where the block's free joint lives in qpos
        data.qpos[address:address + 3] = position
        data.qpos[address + 3:address + 7] = quat
    mujoco.mj_forward(model, data)


def robot_in_collision():
    """Whether the robot, its tool, or the block it holds currently touches anything (other than each other)."""
    held = grasped_block()
    for i in range(data.ncon):
        contact = data.contact[i]
        body1, body2 = body_of_geom(contact.geom1), body_of_geom(contact.geom2)
        if held is not None and {body1, body2} == {held, "tool_suction_cup"}:
            continue  # the held block touches the cup by design
        if body1.startswith(("robot_", "tool_")) or body2.startswith(("robot_", "tool_")) or held in (body1, body2):
            return True
    return False


def in_collision(q):
    """Whether the robot (with the block it holds, if any) is in contact with anything when placed at configuration q."""
    snapshot = data.qpos.copy()
    set_configuration(q)
    collision = robot_in_collision()
    data.qpos[:] = snapshot
    mujoco.mj_forward(model, data)
    return collision


# --- motion planning with OMPL ---


ou.setLogLevel(ou.LogLevel.LOG_WARN)  # OMPL is chatty; set LOG_INFO to see what the planners are up to


def make_state(space, values):
    state = space.allocState()
    for i, value in enumerate(values):
        state[i] = float(value)
    return state


def iterations(n):
    """A termination condition that stops the planner after n iterations."""
    count = [0]

    def done():
        count[0] += 1
        return count[0] > n

    return ob.PlannerTerminationCondition(ob.PlannerTerminationConditionFn(done))


def path_to_array(path, dimension):
    """The waypoints of an OMPL path as an array of shape (n_waypoints, dimension)."""
    return np.array([[path.getState(i)[j] for j in range(dimension)] for i in range(path.getStateCount())])




def set_seed(seed):
    """Seed every source of randomness we use: Python, NumPy and OMPL. Use it right before creating a planner.
    The seed must be positive: OMPL silently ignores a seed of 0."""
    if seed < 1:
        raise ValueError(f"the seed must be a positive integer (OMPL ignores 0), got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    log_level = ou.getLogLevel()
    ou.setLogLevel(ou.LogLevel.LOG_NONE)  # silence OMPL's complaint about being seeded again
    ou.RNG.setSeed(seed)
    ou.setLogLevel(log_level)


JOINT_BOUNDS = np.array([[-np.pi, np.pi]] * 6)  # the UR5e joints turn +-2*pi; we plan (and move) within +-pi


def make_arm_space():
    space = ob.RealVectorStateSpace(6)
    bounds = ob.RealVectorBounds(6)
    for i, (low, high) in enumerate(JOINT_BOUNDS):
        bounds.setLow(i, float(low))
        bounds.setHigh(i, float(high))
    space.setBounds(bounds)
    return space


def is_valid_configuration(state):
    set_configuration([state[i] for i in range(6)])
    return not robot_in_collision()


def plan(q_start, q_goal, planner_type=og.RRTConnect, max_iterations=20000, resolution=0.002, simplify=True, spacing=0.05, verbose=True):
    """Plan a collision-free path from q_start to q_goal. Returns a list of waypoints (start and goal included) or None."""
    snapshot = data.qpos.copy()  # the validity checks teleport the robot; we put everything back at the end

    # the configuration space and its free space
    space = make_arm_space()
    space_info = ob.SpaceInformation(space)
    space_info.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid_configuration))
    space_info.setStateValidityCheckingResolution(resolution)
    space_info.setup()

    # the problem: from the start to the goal. Angles are periodic, so we may wrap the goal into the bounds.
    problem = ob.ProblemDefinition(space_info)
    problem.setStartAndGoalStates(make_state(space, q_start), make_state(space, wrap_angles(q_goal)))

    # the planner
    planner = planner_type(space_info)
    planner.setProblemDefinition(problem)
    planner.setup()

    start_time = time.time()
    status = planner.solve(ob.plannerOrTerminationCondition(iterations(max_iterations), ob.exactSolnPlannerTerminationCondition(problem)))
    planning_time = time.time() - start_time

    waypoints = None
    if status.getStatus() == ob.PlannerStatus.EXACT_SOLUTION:
        path = problem.getSolutionPath()
        raw_length = path.length()
        if simplify:
            og.PathSimplifier(space_info).simplify(path, 0.0)
        if spacing is not None:
            path.interpolate(max(2, int(np.ceil(path.length() / spacing))))
        waypoints = list(path_to_array(path, 6))
        if verbose:
            print(f"{planner.getName()}: {status.asString()} in {planning_time:.2f}s, path length {raw_length:.2f} rad"
                  + (f" ({path.length():.2f} after simplification)" if simplify else "") + f", {len(waypoints)} waypoints")
    elif verbose:
        print(f"{planner.getName()}: {status.asString()} after {planning_time:.2f}s")

    data.qpos[:] = snapshot
    mujoco.mj_forward(model, data)
    return waypoints


load_scene()