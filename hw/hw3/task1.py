"""Homework 3, Task 1: MCTS.

This module provides:
- Random block placement and the stacking order.
- Utilities brought over from previous notebooks, for verifying the task and moving the
  robot.
- The action space: stacking a block at an offset from the base of the tower.
- The simulator API and the block stacking simulator.
- The components of UCT with double progressive widening, and the UCT search.
- The evaluation loop, and the UCT planning time.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import time
from typing import Any, Callable

import mediapy as media
import mujoco
import numpy as np
from tqdm.auto import tqdm
from ur_analytic_ik import ur5e

from scene import model


# table base height + half the table thickness
TABLE_SURFACE_HEIGHT = model.geom("table_top").pos[2] + model.geom("table_top").size[2]

# height of block is twice the defined size in z axis
BLOCK_HEIGHT = model.geom("red_box_geom").size[2] * 2


def sample_on_table(model):
    """
    sample (x, y, z) coordinates that are located on the table.
    """
    # get mujoco actionable entity for the table
    table_geom = model.geom("table_top")

    # retrieve table info
    x_size, y_size, _ = table_geom.size
    x, y, z = table_geom.pos

    # determine the range of allowed values in the x and y axes
    # leave some margin from table edges
    x_range = [x + 0.1, x + x_size * 2 - 0.25]
    y_range = [y - y_size + 0.15, y + y_size - 0.15]

    # TODO sample x,y coordinates uniformly within the allowed ranges.
    # z does not need to be sampled since the height of the table is consistent for all x,y values.
    # Remember that the MuJoCo creates the blocks centered at the given position, so we need to
    # add half the block height to the table height to get the correct z coordinate for placing.
    sample_z = ...
    sample_x, sample_y = ...

    # return coordinates as numpy array
    return np.array([sample_x, sample_y, sample_z])


def is_pos_colliding_with_block(model, block_name, sampled_pos, padding=0.05):
    """
    Determines whether placing a block in a sampled position will cause a collision with a specific block.

    Args:
        block_name (str): name of the block to check collision against
        sampled_pos (np.array): (x, y, z) coordinates of the sampled position
        padding (float): additional padding to consider around the block for collision checking
    """
    # get mujoco actionable entity for the block with which to check collision
    block_geom = model.geom(block_name)

    # get block info
    block_size = block_geom.size[:2] + padding
    block_pos = block_geom.pos[:2]

    # low and high shift based on full block size to account for the block being checked AND the block
    # being placed in the sampled position (assuming the blocks have the same shape and size).
    block_low = block_pos - block_size
    block_high = block_pos + block_size

    #TODO check if there is any overlap between the sampled position and the block bounding box
    # check only x-y axis
    is_colliding = ...

    return is_colliding


def set_blocks_qpos(model, data, pos_dict):
    for box, pos in pos_dict.items():
        #TODO find the box's qpos address via its freejoint element and update its position
        qpos_adr = ...
        data.qpos[qpos_adr : qpos_adr + 3] = pos


def reset_env(model, data):
    pos_dict = {}

    pos_dict["red"] = sample_on_table(model)
    model.geom("red_box_geom").pos = pos_dict["red"]

    pos_dict["green"] = sample_on_table(model)
    while is_pos_colliding_with_block(model, "red_box_geom", pos_dict["green"]):
        pos_dict["green"] = sample_on_table(model)
    model.geom("green_box_geom").pos = pos_dict["green"]

    pos_dict["blue"] = sample_on_table(model)
    while is_pos_colliding_with_block(model,
        "red_box_geom", pos_dict["blue"]
    ) or is_pos_colliding_with_block(model, "green_box_geom", pos_dict["blue"]):
        pos_dict["blue"] = sample_on_table(model)
    model.geom("blue_box_geom").pos = pos_dict["blue"]

    pos_dict["yellow"] = sample_on_table(model)
    while (
        is_pos_colliding_with_block(model, "red_box_geom", pos_dict["yellow"])
        or is_pos_colliding_with_block(model, "green_box_geom", pos_dict["yellow"])
        or is_pos_colliding_with_block(model, "blue_box_geom", pos_dict["yellow"])
    ):
        pos_dict["yellow"] = sample_on_table(model)
    model.geom("yellow_box_geom").pos = pos_dict["yellow"]

    mujoco.mj_resetDataKeyframe(model, data, model.key("robot_home").id)

    set_blocks_qpos(model, data, pos_dict)
    mujoco.mj_forward(model, data)


def get_stacking_order(model):
    xy_positions = {
        color: tuple(model.geom(f"{color}_box_geom").pos[:2])
        for color in ["red", "green", "blue", "yellow"]
    }

    return sorted(xy_positions.keys(), key=lambda color: xy_positions[color], reverse=True)


FACE_DOWN_ROTATION = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
HOME_CONFIG = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])


############################
# Task Verifying Utilities #
############################


def get_contacts(geom1, geom2, data: mujoco.MjData):
    """
    get the indices in the MuJoCo contacts database that match those of the given geoms, and the direction
    of contact (1 - geom1 to geom2; 2- geom2 to geom1)

    Args:
        geom1: first mujoco geom model element
        geom2: second mujoco geom model element

    Returns:
        contact_idx: list of indices in data.contact that correspond to contacts between geom1 and geom2
        contact_dir: list of directions for each contact (1 if geom1 to geom2, -1 if geom2 to geom1)
    """
    contact_idx = []
    contact_dir = []
    for i, (contact_id1, contact_id2) in enumerate(data.contact.geom):
        if contact_id1 == geom1.id and contact_id2 == geom2.id:
            contact_idx.append(i)
            contact_dir.append(1)
        elif contact_id2 == geom1.id and contact_id1 == geom2.id:
            contact_idx.append(i)
            contact_dir.append(-1)

    return contact_idx, contact_dir


def get_normal_force(geom1, geom2, data: mujoco.MjData):
    """
    get the normal force applied by geom1 on geom2

    Args:
        geom1: first mujoco geom model element
        geom2: second mujoco geom model element
        data: mujoco simulation data

    Returns:
        normal_force: numpy array of shape (3,) representing the mean normal force vector applied
                      by geom1 on geom2
    """
    geom_contacts, geom_contact_dirs = get_contacts(geom1, geom2, data)
    if not geom_contacts:
        return np.array([0, 0, 0])

    frame = data.contact.frame[geom_contacts]

    # normal force is index 0-2 of the force frame (Z-axis). direction is always geom1 to geom2
    # see https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h
    all_contact_normals = frame.T[
        0:3
    ]  # transpose to enable referencing (x,y,z) at the top level
    all_contact_normals = (
        all_contact_normals * geom_contact_dirs
    )  # set direction according to args order

    # average all contact normals for one definite normal force
    return all_contact_normals.mean(axis=1)


def entity_is_on_entity(entity1, entity2, data: mujoco.MjData) -> bool:
    """
    entity1 is considered "on" entity2 if the normal force from entity2 to entity1 is directly up

    Args:
        entity1: first mujoco geom model element
        entity2: second mujoco geom model element
        data: mujoco simulation data

    Returns:
        bool: True if entity1 is on entity2, False otherwise
    """
    # get normal force being applied by entity2 on entity1
    normal = get_normal_force(entity2, entity1, data)

    # calculate the distance between the normal force and the upward normal vector
    normal_dist_from_up_dir = np.linalg.norm(normal - np.array([0, 0, 1]))

    # determine whether the calculated distance is "close enough" to 0 using the `np.isclose` function
    # this function returns True if the distance we calculated is close to 0.
    # the default tolerance is 1e-08. this is too strict for us so we will change this to 0.2
    return np.isclose(normal_dist_from_up_dir, 0, atol=0.2)


def get_block_geom_by_color(block_color: str):
    """
    Returns the mujoco geom model element for the block of the specified color.

    Args:
        block_color (str): color of the block ("red", "green", "blue", "yellow")

    Returns:
        mujoco geom model element for the specified block
    """
    return model.geom(f"{block_color}_box_geom")


def holding(block_color: str, data: mujoco.MjData) -> bool:
    """
    Determines whether the gripper is holding the block of the specified color.
    This is calculated by checking whether all gripper pads are in contact with the block.

    Args:
        block_color (str): color of the block ("red", "green", "blue", "yellow")
        data (mujoco.MjData): mujoco simulation data

    Returns:
        bool: True if the gripper is holding the block, False otherwise
    """
    block_geom_id = get_block_geom_by_color(block_color).id

    gripper_pad_ids = [
        model.geom("gripper_left_pad1").id,
        model.geom("gripper_left_pad2").id,
        model.geom("gripper_right_pad1").id,
        model.geom("gripper_right_pad2").id,
    ]

    has_contact_with = [False for _ in gripper_pad_ids]
    for i in range(data.ncon):
        # get contact info
        contact = data.contact[i]

        geom_1_is_block = contact.geom1 == block_geom_id
        geom_2_is_block = contact.geom2 == block_geom_id

        for j, gripper_pad_id in enumerate(gripper_pad_ids):
            geom_1_is_gripper = contact.geom1 == gripper_pad_id
            geom_2_is_gripper = contact.geom2 == gripper_pad_id

            if (geom_1_is_block and geom_2_is_gripper) or (
                geom_2_is_block and geom_1_is_gripper
            ):
                has_contact_with[j] = True
                break

        if all(has_contact_with):
            break

    return all(has_contact_with)


def on(block1_id, block2_id, state):
    """
    Determines whether block1 is on top of block2 while not being held by the gripper.

    Args:
        block1_id: color of the first block ("red", "green", "blue", "yellow")
        block2_id: color of the second block ("red", "green", "blue", "yellow")
        state: mujoco simulation data

    Returns:
        bool: True if block1 is on block2, False otherwise
    """

    if holding(block1_id, state):
        return False

    block1 = get_block_geom_by_color(block1_id)
    block2 = get_block_geom_by_color(block2_id)

    return entity_is_on_entity(block1, block2, state)


##########################
# Robot Motion Utilities #
##########################


def transform_world_to_ur5e_base(
    data: mujoco.MjData,
    target_pos_world: np.ndarray,
    target_rot_mat_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms a target position and orientation from world frame to the UR5e robot base frame.

    Args:
        data (mujoco.MjData): mujoco MjData object
        target_pos_world (np.ndarray): target position in world frame (3,)
        target_rot_mat_world (np.ndarray): target orientation in world frame as rotation matrix (3,3)

    Returns:
        tuple[np.ndarray, np.ndarray]: target position (3,) and orientation (3,3) in UR5e base frame
    """
    # the robot base is directly on the "mount" body
    mount_body = data.body("mount")
    mount_pos, mount_rot_mat = mount_body.xpos, mount_body.xmat.reshape(3, 3)

    # Compute relative rotation
    target_rot_mat_ur5e = mount_rot_mat.T @ target_rot_mat_world

    # Compute relative position
    target_pos_ur5e = mount_rot_mat.T @ (target_pos_world - mount_pos)

    return target_pos_ur5e, target_rot_mat_ur5e


def compute_ik(
    data: mujoco.MjData,
    target_pos_world: np.ndarray,
    target_rot_mat_world: np.ndarray,
) -> list[np.ndarray]:
    """
    Computes the inverse kinematics solutions for the UR5e robot to reach a target
    end-effector pose specified in the world frame.

    Args:
        data (mujoco.MjData): mujoco MjData object
        target_pos_world (np.ndarray): target position in world frame (3,)
        target_rot_mat_world (np.ndarray): target orientation in world frame as rotation matrix (3,3)

    Returns:
        list[np.ndarray]: list of IK solutions, each solution is a numpy array of joint angles
    """

    # ur-analytic-ik expects the end-effector pose in the robot's base frame.
    # The target_pos_world and target_rot_mat_world are given in the world frame.
    # We need to account for the mount's translation and orientation if necessary.
    target_pos_ur5e, target_rot_mat_ur5e = transform_world_to_ur5e_base(
        data,
        target_pos_world,
        target_rot_mat_world,
    )

    # ur-analytic-ik expects the end-effector pose as a single 4x4 reference
    # transformation matrix.
    pose = np.identity(4)
    pose[:3, 3] = target_pos_ur5e
    pose[:3, :3] = target_rot_mat_ur5e

    # Compute IK solutions
    ik_solutions = ur5e.inverse_kinematics(pose)

    return ik_solutions


def get_ik_position(data, position: np.ndarray):
    """
    Get the closest collision-free IK solution for the given target position.
    The orientation is fixed to face down.

    Args:
        data (mujoco.MjData): mujoco MjData object
        position (np.ndarray): target position in world frame (3,)

    Returns:
        np.ndarray: closest collision-free IK solution as a numpy array of joint angles
    """
    # compute all IK solutions for the target position with face-down orientation
    grasp_q = compute_ik(
        data,
        target_pos_world=position,
        target_rot_mat_world=FACE_DOWN_ROTATION,
    )

    # if no IK solutions found, raise error
    if not grasp_q:
        raise ValueError(f"No collision-free IK solution found")

    # find the closest IK solution to the current robot position
    cur_robot_position = data.qpos[28:34]
    closest_q = min(
        grasp_q,
        key=lambda q: np.linalg.norm(q - cur_robot_position),
    )

    return closest_q


def wait(model, data, seconds, framerate=0):
    """
    Simulate for the given number of seconds while capturing frames at the specified framerate.

    Args:
        model: mujoco model
        data: mujoco data
        seconds: number of seconds to simulate
        framerate: framerate for capturing frames. Set to 0 to disable rendering.

    Returns:
        list of captured frames
    """
    frames = []
    start_time = data.time
    steps = int(seconds / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)

        # capture frame at the specified framerate
        if len(frames) < (data.time - start_time) * framerate:
            # render the scene
            with mujoco.Renderer(model) as renderer:
                renderer.update_scene(data, camera="frontview")
                pixels = renderer.render()
                frames.append(pixels)

    return frames


def go_to_config(
    target_q,
    model,
    data,
    goal_threshold=0.05,
    wait_after_move=1,
    max_steps=10_000,
    framerate=0,
):
    """
    Move the robot to the target joint configuration while capturing frames at the specified framerate.

    Args:
        target_q: target joint configuration as a numpy array
        model: mujoco model
        data: mujoco data
        goal_threshold: threshold for considering the goal reached
        wait_after_move: seconds to wait after reaching the goal
        max_steps: maximum number of simulation steps to perform
        framerate: framerate for capturing frames. Set to 0 to disable rendering.

    Returns:
        list of captured frames
    """
    data.ctrl[:6] = target_q

    frames = []
    start_time = data.time
    for _ in range(max_steps):
        mujoco.mj_step(model, data)

        # capture frame at the specified framerate
        if len(frames) < (data.time - start_time) * framerate:
            # render the scene
            with mujoco.Renderer(model) as renderer:
                renderer.update_scene(data, camera="frontview")
                pixels = renderer.render()
                frames.append(pixels)

        if np.linalg.norm(data.qpos[28:34] - target_q) < goal_threshold:
            break

    frames += wait(model, data, wait_after_move, framerate)

    return frames


# The length of the gripper from the wrist joint to the gripper tip
# used as an offset when calculating pick and place heights
GRIPPER_LENGTH = 0.16

# Pre-grasp and pick heights. choose exactly half a meter above the table surface.
# We choose this height to ensure that the gripper has enough clearance to avoid collisions.
GRIPPER_PREGRASP_HEIGHT = TABLE_SURFACE_HEIGHT + 0.5

# Pick height is the height at which the gripper will grasp the block
GRIPPER_PICK_HEIGHT = TABLE_SURFACE_HEIGHT + GRIPPER_LENGTH + BLOCK_HEIGHT / 2

# Additional drop offset to ensure the block is released slightly above the target position
# to avoid colliding with the tower.
DROP_OFFSET = 0.01


def stack_at_offset(
    color,
    offset,
    tower_base_color,
    cur_tower_size,
    model,
    data,
    goal_threshold=0.1,
    max_steps=10_000,
    framerate=30,
):
    """
    Stack the block of the given color at the specified offset on top of the tower.
    The order of operations is as follows:
    1. Move to pre-grasp position above the block to pick up
    2. Move to grasp position and close gripper
    3. Move back to pre-grasp position
    4. Move to pre-drop position above the place position
    5. Move to place position and open gripper
    6. Move back to pre-drop position


    Args:
        color: color of the block to stack ("red", "green", "blue", "yellow")
        offset: (x, y) offset from the tower base position as a numpy array
        tower_base_color: color of the base block of the tower, defining the tower position
        cur_tower_size: current size of the tower (number of blocks)
        model: mujoco model
        data: mujoco data
        goal_threshold: threshold for considering the goal positions are reached during each motion
        max_steps: maximum number of simulation steps to perform during each motion
        framerate: framerate for capturing frames. Set to 0 to disable rendering.

    Returns:
        list of captured frames
    """
    frames = []

    # Move to pre-grasp position above the block
    pick_pos = data.geom(f"{color}_box_geom").xpos.copy()
    pick_pos_pregrasp = pick_pos.copy()
    pick_pos_pregrasp[2] = GRIPPER_PREGRASP_HEIGHT
    pick_pregrasp_cfg = get_ik_position(data, pick_pos_pregrasp)

    frames = go_to_config(
        pick_pregrasp_cfg,
        model,
        data,
        goal_threshold,
        wait_after_move=1,
        max_steps=max_steps,
        framerate=framerate,
    )

    # Move to grasp position
    pick_pos[2] = GRIPPER_PICK_HEIGHT
    pick_cfg = get_ik_position(data, pick_pos)
    frames += go_to_config(
        pick_cfg,
        model,
        data,
        goal_threshold,
        wait_after_move=1,
        max_steps=max_steps,
        framerate=framerate,
    )

    # Close gripper
    data.ctrl[-1] = 255
    frames += wait(model, data, 0.5, framerate)

    # Move back to pre-grasp position
    frames += go_to_config(
        pick_pregrasp_cfg,
        model,
        data,
        goal_threshold,
        wait_after_move=0.1,
        max_steps=max_steps,
        framerate=framerate,
    )

    # Move to pre-drop position above the place position
    place_pos = data.geom(f"{tower_base_color}_box_geom").xpos.copy()
    place_pos_predrop = place_pos.copy()
    place_pos_predrop[2] = GRIPPER_PREGRASP_HEIGHT
    place_pos_predrop[:2] += offset
    place_predrop_cfg = get_ik_position(data, place_pos_predrop)
    frames += go_to_config(
        place_predrop_cfg,
        model,
        data,
        goal_threshold,
        wait_after_move=1,
        max_steps=max_steps,
        framerate=framerate,
    )

    # Move to place position
    place_pos[2] = (
        TABLE_SURFACE_HEIGHT
        + GRIPPER_LENGTH
        + BLOCK_HEIGHT * cur_tower_size
        + BLOCK_HEIGHT / 2
        + DROP_OFFSET
    )
    place_pos[:2] += offset
    place_cfg = get_ik_position(data, place_pos)
    frames += go_to_config(
        place_cfg,
        model,
        data,
        goal_threshold,
        wait_after_move=1,
        max_steps=max_steps,
        framerate=framerate,
    )

    # Open gripper
    data.ctrl[-1] = 0
    frames += wait(model, data, 0.5, framerate)

    # Move back to pre-grasp position
    frames += go_to_config(
        place_predrop_cfg,
        model,
        data,
        goal_threshold,
        wait_after_move=0.1,
        max_steps=max_steps,
        framerate=framerate,
    )

    return frames


class Simulator(ABC):
    """
    Abstract base class for a simulator environment.
    To be used as a simulator for MCTS.
    """

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool]:
        """
        Take a step in the environment with the given action.

        Args:
            action: The action to take.

        Returns:
            next_state: A *copy* of the next state after taking the action.
            reward: The reward received after taking the action.
            done: Whether the episode has ended.
            info: a dictionary containing additional information about the step.
        """

    @abstractmethod
    def reset(self) -> Any:
        """
        Reset the environment to an initial state.

        Returns:
            state: A *copy* of the initial state of the environment.
        """

    @abstractmethod
    def set_state(self, state: Any) -> None:
        """
        Set the environment to a *copy* of the given state.

        Args:
            state: The state to set the environment to.
        """

    @abstractmethod
    def sample_action(self, state: Any) -> Any:
        """
        Sample a random action valid for the given state.

        Args:
            state: The state from which to sample an action.

        Returns:
            action: A random action valid for the given state.
        """

    @abstractmethod
    def done(self) -> bool:
        """
        Check whether the current state is terminal.

        Returns:
            done: Whether the current state is terminal.
        """


# Define the range for sampling offsets when stacking blocks.
# Must be big enough to allow some variability to enable stabilizing, but small enough
# so that random action sampling is not too noisy.
OFFSET_RANGE = (-0.01, 0.01)

@dataclass
class BlockStackingState:
    """
    Represents the state of the block stacking environment.

    Attributes:
        stacked_blocks_offsets (list[np.ndarray]): List of (x, y) offsets for each stacked block
            relative to the base block position and according to the stacking order.
    """
    stacked_blocks_offsets: list[np.ndarray]

@dataclass
class BlockStackingAction:
    """
    Represents an action in the block stacking environment.

    Attributes:
        offset_x (float): x offset to stack the next block at.
        offset_y (float): y offset to stack the next block at.
    """
    offset_x: float
    offset_y: float

class BlockStackingEnv(Simulator):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        self.state = None

        self._stacking_order = None

    def reset(self):
        reset_env(self.model, self.data)
        self._stacking_order = get_stacking_order(self.model)

        self.state = BlockStackingState(
            # The offset of the bottom block is always zero (offset from itself)
            stacked_blocks_offsets=[np.array([0.0, 0.0])],
        )

        return deepcopy(self.state)

    def step(self, action: BlockStackingAction, framerate=0):
        offset = np.array([action.offset_x, action.offset_y])
        color_to_stack = self._stacking_order[len(self.state.stacked_blocks_offsets)]

        #TODO perform stacking action and capture the rendered frames.
        # use the `stack_at_offset` function defined earlier.
        # note the property `_tower_base_color` can be used to get the color of the base block of the tower.
        frames = ...

        # wait for simulation to settle
        frames += wait(self.model, self.data, seconds=2, framerate=framerate)

        stacked_block_position = self.data.geom(f"{color_to_stack}_box_geom").xpos[:2]
        stacked_block_offset = stacked_block_position - self._tower_base_position

        # update a copy of the state to avoid mutating the existing state in nodes
        self.state = deepcopy(self.state)
        self.state.stacked_blocks_offsets.append(stacked_block_offset)

        reward = float(self.success())

        return self.state, reward, self.done(), {'frames': frames}

    def set_state(self, state: BlockStackingState):
        self.reset()

        pos_dict = {}
        already_stacked_colors = self._stacking_order[:len(state.stacked_blocks_offsets)]
        for i, (color, offset) in enumerate(zip(already_stacked_colors, state.stacked_blocks_offsets)):
            block_z = TABLE_SURFACE_HEIGHT + BLOCK_HEIGHT / 2 + i * BLOCK_HEIGHT
            block_xy = self._tower_base_position + offset
            pos_dict[color] = np.array([*block_xy, block_z])
        set_blocks_qpos(self.model, self.data, pos_dict)

        self.state = deepcopy(state)

    def sample_action(self, state: BlockStackingState) -> BlockStackingAction:
        #TODO sample a random stacking action within the defined OFFSET_RANGE.
        # This is agnostic to the current state.
        offset_x, offset_y = ...
        return BlockStackingAction(offset_x=offset_x, offset_y=offset_y)

    def done(self):
        return len(self.state.stacked_blocks_offsets) >= len(self._stacking_order)

    def success(self):
        #TODO check whether the blocks are stacked correctly according to the stacking order.
        # use the `on` predicate function we imported from a previous notebook.
        success = ...

        return success

    def render(self) -> np.ndarray:
        with mujoco.Renderer(self.model) as renderer:
            renderer.update_scene(self.data, camera="frontview")
            pixels = renderer.render()
        return pixels

    def clone(self):
        cloned_env = BlockStackingEnv(deepcopy(self.model), deepcopy(self.data))
        cloned_env.set_state(deepcopy(self.state))
        return cloned_env

    @property
    def _tower_base_color(self):
        return self._stacking_order[0]

    @property
    def _tower_base_position(self):
        return self.data.geom(f"{self._tower_base_color}_box_geom").xpos.copy()[:2]


@dataclass
class MCTSNode:
    state: Any                                                      # The state represented by this node
    incoming_edge: "MCTSEdge" = None                                # The edge leading to this node
    outgoing_edges: list["MCTSEdge"] = field(default_factory=list)  # The edges leading to child nodes
    visits: int = 0                                                 # Number of times this node has been visited
    total_return: float = 0.0                                       # Total return accumulated through this node
    reward: float = 0.0                                             # Reward received upon reaching this node

    def top_ucb_edge(self, exploration_constant):
        """
        Select the outgoing edge with the highest UCB value.

        Args:
            exploration_constant: The exploration constant used in UCB calculation.

        Returns:
            best_edge: The outgoing edge with the highest UCB value.
        """

        best_ucb = -float('inf')
        best_edge = None

        for edge in self.outgoing_edges:
            #TODO compute UCB for each outgoing edge and maintain the edge with the highest UCB value.
            ucb = ...

        return best_edge

    def is_terminal(self, sim: Simulator):
        """
        Check whether the node represents a terminal state.

        Args:
            sim: The simulator used to check terminality.

        Returns:
            done: Whether the node represents a terminal state.
        """
        sim.set_state(self.state)
        return sim.done()


@dataclass
class MCTSEdge:
    parent: MCTSNode                 # The parent node from which this edge originates
    action: Any                      # The action taken to traverse this edge
    outcomes: list[MCTSNode] = None  # Possible outcome nodes resulting from this action
    visits: int = 0                  # Number of times this edge has been visited
    total_return: float = 0.0        # Total return accumulated through this edge

    def q(self):
        """
        Compute the Q value for this edge.

        Returns:
            q_value: The Q value of this edge.
        """
        if self.visits == 0:
            return 0.0

        # TODO compute the Q value as the average return of this edge.
        q_value = ...

        return q_value

    def ucb(self, exploration_constant):
        """
        Compute the UCB value for this edge.

        Args:
            exploration_constant: The exploration constant used in UCB calculation.

        Returns:
            ucb_score: The UCB value of this edge.
        """
        if self.visits == 0:
            return float('inf')

        #TODO compute the UCB value using the formula:
        ucb_score = ...

        return ucb_score


def progressive_widening_limit(visits: int, k, alpha) -> int:
    """
    Compute the progressive widening limit based on the number of visits.

    Args:
        visits: Number of visits to the node.
        k: Progressive widening scaling constant.
        alpha: Progressive widening throttle constant.
    """
    #TODO compute the progressive widening limit
    pw_limit = ...

    return pw_limit


def expand_outcome(edge: MCTSEdge, sim: Simulator) -> MCTSNode:
    """
    Expand a new outcome node for the given edge by simulating the action.

    Args:
        edge: The edge to expand an outcome for.
        sim: The simulator used to simulate the action.

    Returns:
        outcome_node: The newly created outcome node.
    """
    sim.set_state(edge.parent.state)

    #TODO sample the next state by simulating the edge action
    next_state, reward, _, _ = ...

    # create a new outcome node
    outcome_node = MCTSNode(state=next_state, incoming_edge=edge, reward=reward)

    # add the new outcome node to the edge's outcomes
    edge.outcomes.append(outcome_node)

    return outcome_node


def dpw_select_outcome(edge: MCTSEdge, sim: Simulator, pw_k, pw_alpha) -> MCTSNode:
    """
    Select an outcome node from the given edge using Progressive Widening.

    Args:
        edge: The edge to select an outcome from.
        sim: The simulator used for expansion.
        pw_k: Progressive widening scaling constant.
        pw_alpha: Progressive widening throttle constant.

    Returns:
        outcome_node: The selected outcome node.
    """
    #TODO find the PW limit for the edge
    pw_limit = ...

    if len(edge.outcomes) < pw_limit:
        # Add a new outcome
        return expand_outcome(edge, sim)

    # calculate the total visits across all outcomes
    total_visits = sum(outcome.visits for outcome in edge.outcomes)

    # handle edge case where all outcomes have 0 visits
    if total_visits == 0:
        return np.random.choice(edge.outcomes)

    #TODO sample an outcome based on visit counts
    visit_probs = ...
    outcome = np.random.choice(edge.outcomes, p=visit_probs)

    return outcome


def dpw_select_node(
    root: MCTSNode, sim: Simulator, exploration_constant, pw_k, pw_alpha
) -> MCTSNode:
    """
    Select a node from the tree using Double Progressive Widening.

    Args:
        root: The root node from which to start selection.
        sim: The simulator used for expansion.
        exploration_constant: The exploration constant used in UCB calculation.
        pw_k: Progressive widening scaling constant.
        pw_alpha: Progressive widening throttle constant.

    Returns:
        selected_node: The selected node.
    """
    current_node = root

    while not current_node.is_terminal(sim):
        #TODO find the PW limit for the current node
        # check progressive widening condition
        # this is like checking if the node is fully expanded in the standard MCTS
        pw_limit = ...

        # check if this node is considered "expanded enough"
        if len(current_node.outgoing_edges) < pw_limit:
            break  # Stop selection if we can add more edges

        #TODO Select the best edge using UCB
        best_edge = ...

        #TODO choose an outcome node using DPW
        current_node = ...

    return current_node


def dpw_expand_node(
    node: MCTSNode, sim: Simulator
) -> MCTSNode:
    """
    Expand a new action edge and outcome node from the given node.

    Args:
        node: The node to expand.
        sim: The simulator used for expansion.

    Returns:
        outcome_node: The newly created outcome node.
    """
    #TODO get a random action from the simulator
    action = ...

    # create new edge
    new_edge = MCTSEdge(parent=node, action=action, outcomes=[])
    node.outgoing_edges.append(new_edge)

    #TODO expand one outcome for the new edge
    outcome_node = ...

    return outcome_node


def simulate_rollout(
    node: MCTSNode, sim: Simulator, rollout_policy: Callable[[Any], Any]
) -> float:
    """
    Simulate a rollout from the given node using the provided rollout policy.
    Args:
        node: The node from which to start the rollout.
        sim: The simulator used for the rollout.
        rollout_policy: A function that takes a state and returns an action.

    Returns:
        total_reward: The total reward accumulated during the rollout.
    """
    state = node.state
    sim.set_state(state)

    total_reward = 0.0
    done = False
    while not done:
        action = rollout_policy(state)
        state, reward, done, _ = sim.step(action)
        total_reward += reward

    return total_reward


def zero_offset_rollout_policy(state: BlockStackingState) -> BlockStackingAction:
    """
    A simple rollout policy that always returns zero offsets.

    Args:
        state: The current state of the environment.

    Returns:
        action: An action with zero offsets.
    """
    return BlockStackingAction(offset_x=0.0, offset_y=0.0)


def backpropagate(
    node: MCTSNode, reward: float
):
    current_node = node
    while current_node is not None:
        #TODO update visit counts and total returns for the current node
        ...

        if current_node.incoming_edge is not None:
            edge = current_node.incoming_edge

            #TODO update visit counts and total returns for the incoming edge
            ...

            current_node = edge.parent
        else:
            current_node = None


class ContinuousActionStochasticOutcomeUCT:
    def __init__(
        self,
        sim: Simulator,
        rollout_policy: Callable[[Any], Any],
        pw_k: float,
        pw_alpha: float,
    ):
        self.sim = sim
        self.rollout_policy = rollout_policy
        self.pw_k = pw_k
        self.pw_alpha = pw_alpha

    def search(
        self,
        root_node: MCTSNode,
        time_limit: float,
        exploration_constant: float,
    ) -> Any:
        pbar = tqdm(total=time_limit, desc="MCTS Search", leave=False)
        start_time = time.time()
        while time.time() - start_time < time_limit:
            self.sim.set_state(root_node.state)

            try:
                # selection
                node_to_expand = dpw_select_node(
                    root_node,
                    self.sim,
                    exploration_constant,
                    self.pw_k,
                    self.pw_alpha,
                )

                # expansion
                if node_to_expand.is_terminal(self.sim):
                    new_child_node = node_to_expand
                else:
                    new_child_node = dpw_expand_node(node_to_expand, self.sim)

                # rollout
                if new_child_node.is_terminal(self.sim):
                    reward = new_child_node.reward
                else:
                    reward = simulate_rollout(
                        new_child_node,
                        self.sim,
                        self.rollout_policy,
                    )
            except ValueError as e:
                print(f"Error during selection: {e}")
                pbar.update(time.time() - start_time - pbar.n)
                continue

            # backpropagation
            backpropagate(new_child_node, reward)

            pbar.update(time.time() - start_time - pbar.n)
        pbar.close()

        # select the best action from the root node according to number of visits
        # This is called "robust child" in some literature. It is more robust than selecting
        # the action with the highest average reward with stochastic outcomes and imperfect rollouts.
        best_edge = max(
            root_node.outgoing_edges,
            key=lambda edge: edge.visits
        )

        return best_edge.action


def run_mcts_planning(
    env: BlockStackingEnv,
    mcts,
    root_node: MCTSNode,
    planning_time: float,
    exploration_constant: float,
    framerate: int = 15,
):
    nodes = [root_node]
    frames = []
    for step in range(3):
        print(f"=== MCTS Planning for Step {step + 1} ===")
        best_action = mcts.search(
            root_node=root_node,
            time_limit=planning_time,
            exploration_constant=exploration_constant,
        )
        print(f"mcts search root node stats: {root_node.visits} visits, {root_node.total_return} return, {len(root_node.outgoing_edges)} expansded edges")
        print(
            f"Best Action: offset_x={best_action.offset_x}, offset_y={best_action.offset_y}"
        )

        next_state, reward, done, info = env.step(best_action, framerate=framerate)
        print(f"Reward: {reward}, Done: {done}")
        media.show_image(env.render())

        frames += info['frames']

        # set up root node for next step
        root_node = MCTSNode(state=next_state)
        nodes.append(root_node)

    # If the final reward is non-zero, the stacking was successful
    if reward > 0:
        print("Successfully stacked the blocks!")
    else:
        print("Failed to stack the blocks correctly.")

    return nodes, frames


#TODO select a planning time
PLANNING_TIME = ...  # seconds
