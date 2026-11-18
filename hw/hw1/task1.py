"""Homework 1, Task 1: mapping between spaces.

This module provides:
- End-effector pose retrieval (forward kinematics).
- Reference frame transformations from the world frame to the UR5e base frame.
- Analytical inverse kinematics for the UR5e robot.
- A helper for finding the `qpos` indices of an entity.
- The pre-grasp pose above the red box.
"""

import mujoco
import numpy as np
from ur_analytic_ik import ur5e


def get_ee_pose(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the end-effector pose given the Mujoco simulation data.

    Args:
        data (mujoco.MjData): mujoco MjData object after simulating one time step

    Returns:
        tuple[np.ndarray, np.ndarray]: position (3,) and orientation (3,3) of the end-effector
    """

    #TODO return the position and orientation of the robot end effector
    return ...


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
    #TODO get the `mount` body position and orientation
    mount_pos, mount_rot_mat = ...

    #TODO Compute relative rotation
    target_rot_mat_ur5e = ...

    #TODO Compute relative position
    target_pos_ur5e = ...

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

    #TODO transform to UR5 base frame
    # ur-analytic-ik expects the end-effector pose in the robot's base frame.
    # The target_pos_world and target_rot_mat_world are given in the world frame.
    # We need to account for the mount's translation and orientation if necessary.
    target_pos_ur5e, target_rot_mat_ur5e = ...

    #TODO represent pose as transformation matrix
    # ur-analytic-ik expects the end-effector pose as a single 4x4 reference
    # transformation matrix.
    pose = ...

    # Compute IK solutions
    ik_solutions = ur5e.inverse_kinematics(pose)

    return ik_solutions


def get_entity_qpos_idx(model: mujoco.MjModel, prefix: str = "") -> list[int]:
    """
    Retrieves the indices of the `qpos` entries corresponding to joints whose
    names start with the given prefix. The main purpose of this function is to
    help identify which parts of the `qpos` array correspond to the robot.

    Args:
        model (mujoco.MjModel): mujoco MjModel object
        prefix (str): prefix string to match joint names

    Returns:
        list[int]: list of qpos indices corresponding to joints with the given prefix
    """

    # a container for qpos indices
    qpos_idx = []

    # iterate over all joints in the model
    for i in range(model.njnt):
        joint = model.joint(i)

        #TODO check if the joint name starts with the specified prefix and extend qpos_idx
        
    return qpos_idx


def get_above_block_pose(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the pre-grasp end-effector pose above the red box.

    Args:
        data (mujoco.MjData): mujoco MjData object

    Returns:
        tuple[np.ndarray, np.ndarray]: position (3,) and orientation (3,3) of the
            end-effector in the world frame
    """
    #TODO set the target position and orientation above the box
    # choose a high enough position to avoid collision.
    # set the rotation matrix to face downward
    above_block_pos = ...  # shape (3,)
    above_block_rot_mat = ...  # shape (3,3)

    return above_block_pos, above_block_rot_mat
