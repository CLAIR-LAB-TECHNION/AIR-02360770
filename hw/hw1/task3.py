"""Homework 1, Task 3: motion planning for manipulation.

This module provides:
- Parameterized collision checking and state setting that considers attached bodies.
- Pre-grasp and grasp configuration selection.
- Gripper control, and a policy that keeps the robot in place.
- The OMPL motion planning pipeline for the UR5e robot.
"""

import sys

import mujoco
import numpy as np
from ompl import base as ob
import ompl.geometric as og

from scene import model
from task1 import compute_ik, get_ee_pose, get_entity_qpos_idx
import task2
from task2 import PIDJointPositionControlPolicy, get_entity_ctrl_idx


def is_collision_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled_bodies_prefix: list[str],
    exclude_collision_prefix_pairs: list[tuple[str, str]] = [],
    verbose: bool = False
) -> bool:
    """
    Check if the current state in the Mujoco data has any collisions involving
    the specified controlled bodies.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.
        data (mujoco.MjData): The MuJoCo data object.
        controlled_bodies_prefix (list[str]): List of prefixes for body names
            that are considered controlled/moving bodies for which collisions
            should be detected.
        exclude_collision_prefix_pairs (list[tuple[str, str]], optional): List
            of pairs of body name prefixes for which collisions should be ignored.
            Defaults to [].
        verbose (bool, optional): If True, print details of detected collisions.
            Defaults to False.

    Returns:
        bool: True if a collision involving controlled bodies is detected, False otherwise.
    """
    # Check for collision between robot and environment
    # do not count collision between objects and themselves
    for i in range(data.ncon):
        # get contact info
        contact = data.contact[i]

        # get the names of the bodies involved in the contact
        geom1_body_id = model.geom(contact.geom1).bodyid[0]
        geom1_body_name = model.body(geom1_body_id).name
        geom2_body_id = model.geom(contact.geom2).bodyid[0]
        geom2_body_name = model.body(geom2_body_id).name

        #TODO check if this is an excluded collision pair
        # skip if the geoms are pairs to be excluded
        # check both that neither (A,B) and (B,A) are in the exclude list
        excluded_body_collision = ...
        if excluded_body_collision:
            continue  # excluded collision pair. skip contact

        #TODO check if the collision involves any moving body
        moving_body_collision = ...
        if moving_body_collision:
            if verbose:  # print detected collision info
                print(f"Collision detected between {geom1_body_name} and {geom2_body_name}")
            return True  # Collision detected

    return False  # No collision detected


def set_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_qpos: np.ndarray,
    robot_qpos_idx: list[int],
    attached_body_name: str = None,
):
    """
    Sets the robot joint positions while maintaining the pose of an attached body,
    if specified.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.
        data (mujoco.MjData): The MuJoCo data object.
        robot_qpos (np.ndarray): The desired joint positions for the robot.
        robot_qpos_idx (list[int]): The indices in the qpos array corresponding to the robot joints.
        attached_body_name (str, optional): The name of the body attached to the robot's end-effector.
            If provided, the function will maintain the pose of this body relative to the end-effector.
            Defaults to None.
    """
    if attached_body_name is not None:
        #TODO get end-effector pose before setting robot qpos
        prev_ee_pos, prev_ee_rot_mat = ...

    #TODO set robot qpos
    ...
    mujoco.mj_forward(model, data)

    if attached_body_name is None:
        return

    #TODO maintain the end-effector pose relative to the attached object
    new_ee_pos, new_ee_rot_mat = get_ee_pose(data)
    ee_pos_diff = new_ee_pos - prev_ee_pos
    ee_rot_mat_diff = new_ee_rot_mat @ prev_ee_rot_mat.T

    #TODO get attached body pose
    attached_body = data.body(attached_body_name)
    attached_body_pos = ...
    attached_body_rot_mat = ...

    #TODO update position
    attached_body_pos += ...

    #TODO update rotation
    attached_body_rot_mat = ...

    # convert to qpos
    attached_body_quat = np.zeros(4)
    mujoco.mju_mat2Quat(attached_body_quat, attached_body_rot_mat.flatten())
    attached_body_qpos = np.concatenate([attached_body_pos, attached_body_quat])

    #TODO update state
    ...
    mujoco.mj_forward(model, data)


def get_above_block_rot_mat2(above_block_rot_mat: np.ndarray) -> np.ndarray:
    """
    Get the alternative pre-grasp end-effector orientation above the red box.

    Args:
        above_block_rot_mat (np.ndarray): the first pre-grasp orientation as a
            rotation matrix (3,3)

    Returns:
        np.ndarray: the alternative orientation as a rotation matrix (3,3)
    """
    #TODO set an alternative orientation above the box (90 degrees rotated around z axis)
    above_block_rot_mat2 = ...

    return above_block_rot_mat2


def select_pose_configuration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grasp_poses: list[tuple[np.ndarray, np.ndarray]],
    controlled_bodies_prefix: list[str],
    exclude_collision_prefix_pairs: list[tuple[str, str]] = [],
    attached_body_name: str = None,
) -> np.ndarray | None:
    """
    Selects a collision-free robot configuration from the IK solutions
    for the given grasp poses. The function computes IK solutions for each grasp pose,
    and returns the collision-free configuration that is closest to the current robot
    configuration.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.
        data (mujoco.MjData): The MuJoCo data object.
        grasp_poses (list[tuple[np.ndarray, np.ndarray]]): List of grasp poses,
            each defined by a tuple of position (3,) and rotation matrix (3,3).
        controlled_bodies_prefix (list[str]): List of prefixes for body names
            that are considered controlled/moving bodies for which collisions
            should be detected.
        exclude_collision_prefix_pairs (list[tuple[str, str]], optional): List
            of pairs of body name prefixes for which collisions should be ignored.
            Defaults to [].
        attached_body_name (str, optional): The name of the body attached to the robot's
            end-effector. If provided, the function will maintain the pose of this body
            relative to the end-effector when testing configurations. Defaults to None.

    Returns:
        np.ndarray | None: A collision-free robot configuration if found, else None.
    """
    ik_solutions = []

    #TODO Get all ik solutions for each grasp pose
    ...

    #TODO sort solutions according to distance from the current configuration
    ...

    #TODO return the first collision-free solution
    for sol in ik_solutions:
        ...

    return None  # no collision-free solution found


def get_on_block_pos(above_block_pos: np.ndarray) -> np.ndarray:
    """
    Get the grasp position, given the pre-grasp position above the red box.

    Args:
        above_block_pos (np.ndarray): pre-grasp position in world frame (3,)

    Returns:
        np.ndarray: grasp position in world frame (3,)
    """
    #TODO define grasp poses to be slightly lower than pre-grasp but not touching the box
    on_block_pos = ...

    return on_block_pos


class GripperController:
    """
    A simple gripper controller to open and close the gripper.
    """
    def __init__(self, model: mujoco.MjModel):
        """
        Initialize the gripper controller by identifying the gripper actuator control indices.

        Args:
            model (mujoco.MjModel): The MuJoCo model object.
        """
        self.gripper_ctrl_ids = get_entity_ctrl_idx(model, prefix="gripper_")

    def open_gripper(self, data: mujoco.MjData):
        """
        Open the gripper by setting the appropriate control inputs.

        Args:
            data (mujoco.MjData): The MuJoCo data object.
        """
        #TODO set gripper control inputs to open position
        ...

    def close_gripper(self, data: mujoco.MjData):
        """
        Close the gripper by setting the appropriate control inputs.

        Args:
            data (mujoco.MjData): The MuJoCo data object.
        """
        #TODO set gripper control inputs to closed position
        ...


class StayInPlacePolicy(PIDJointPositionControlPolicy):
    """
    A policy that keeps the robot in place by maintaining its current joint positions.
    """
    def __init__(self, model: mujoco.MjModel):
        """
        Initialize the StayInPlacePolicy with the robot's current joint positions.

        Args:
            model (mujoco.MjModel): The MuJoCo model object.
        """
        # set dummy target to the initial robot configuration
        # this is required to initialize the parent class
        qpos0_robot = model.qpos0[get_entity_qpos_idx(model, "robot")]
        super().__init__(model, qpos0_robot, task2.KP, task2.KI, task2.KD)

        # remove target_qpos as a sign that we want to get it from the data
        self.target_qpos = None


    def set_control(self, data: mujoco.MjData):
        # the first time this is called, set the target to the current robot qpos
        if self.target_qpos is None:
            self.target_qpos = data.qpos[get_entity_qpos_idx(self.model, "robot")]

        super().set_control(data)

    def reset(self):
        super().reset()

        # forget the held configuration, so that it is taken from the data again
        self.target_qpos = None


def get_clipped_joint_bounds(model: mujoco.MjModel) -> list[tuple[float, float]]:
    """
    Get the joint bounds for the robot joints in the MuJoCo model,
    clipping them to the range [-pi, pi].

    Args:
        model (mujoco.MjModel): The MuJoCo model object.

    Returns:
        list[tuple[float, float]]: A list of tuples representing the clipped joint bounds
            for each robot joint in the model.
    """

    # a container for the clipped joint bounds
    bounds = []

    #TODO iterate over all joints in the model and get their (clipped) bounds
    ...

    return bounds


# define the number of joints for the UR5e robot
UR5E_NUM_JOINTS = len(get_clipped_joint_bounds(model))

def get_ur5e_state_space(model: mujoco.MjModel) -> ob.RealVectorStateSpace:
    """
    Create an OMPL state space for the UR5e robot with joint bounds obtained
    from the MuJoCo model.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.

    Returns:
        ob.RealVectorStateSpace: The OMPL state space for the UR5e robot.
    """
    # initialize a real vector state space with 6 dimensions (for 6 joints)
    space = ob.RealVectorStateSpace(UR5E_NUM_JOINTS)

    # initialize a bounds object
    bounds = ob.RealVectorBounds(UR5E_NUM_JOINTS)

    # iterate discovered joint bounds and set them in the bounds object
    joint_bounds = get_clipped_joint_bounds(model)
    for i in range(UR5E_NUM_JOINTS):
        low, high = joint_bounds[i]
        bounds.setLow(i, low)
        bounds.setHigh(i, high)

    # add the bounds to the state space
    space.setBounds(bounds)

    return space


class UR5eValidityChecker(ob.StateValidityChecker):
    """
    An OMPL state validity checker for the UR5e robot in MuJoCo.
    """

    def __init__(
        self,
        state_info: ob.SpaceInformation,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controlled_bodies_prefix: list[str],
        exclude_collision_prefix_pairs: list[tuple[str, str]] = [],
        attached_body_name: str = None,
    ):
        """
        Initialize the UR5eValidityChecker.

        Args:
            state_info (ob.SpaceInformation): The OMPL space information object.
            model (mujoco.MjModel): The MuJoCo model object.
            data (mujoco.MjData): The MuJoCo data object.
            controlled_bodies_prefix (list[str]): List of prefixes for body names
                that are considered controlled/moving bodies for which collisions
                should be detected.
            exclude_collision_prefix_pairs (list[tuple[str, str]], optional): List
                of pairs of body name prefixes for which collisions should be ignored.
                Defaults to [].
            attached_body_name (str, optional): The name of the body attached to the robot's
                end-effector. If provided, the function will maintain the pose of this body
                relative to the end-effector when testing configurations. Defaults to None.
        """
        super().__init__(state_info)
        self.model = model
        self.data = data
        self.controlled_bodies_prefix = controlled_bodies_prefix
        self.exclude_collision_prefix_pairs = exclude_collision_prefix_pairs
        self.attached_body_name = attached_body_name

        # get the robot joint position indices
        self.robot_qpos_idx = get_entity_qpos_idx(model, "robot")

    def isValid(self, state: ob.State) -> bool:
        """
        Check if the given OMPL state is valid (collision-free) in the MuJoCo environment.

        Args:
            state (ob.State): The OMPL state to be checked.

        Returns:
            bool: True if the state is valid (collision-free), False otherwise.
        """
        # extract the robot position OMPL state into an array
        robot_qpos = np.empty(UR5E_NUM_JOINTS)
        for i in range(UR5E_NUM_JOINTS):
            robot_qpos[i] = state[i]

        #TODO check if this is a valid state
        return ...


def get_state_space_info(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled_bodies_prefix: list[str],
    exclude_collision_prefix_pairs: list[tuple[str, str]] = [],
    attached_body_name: str = None,
    check_resolution: float = 0.01,
) -> ob.SpaceInformation:
    """
    Create an OMPL SpaceInformation object for the UR5e robot in MuJoCo.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.
        data (mujoco.MjData): The MuJoCo data object.
        controlled_bodies_prefix (list[str]): List of prefixes for body names
            that are considered controlled/moving bodies for which collisions
            should be detected.
        exclude_collision_prefix_pairs (list[tuple[str, str]], optional): List
            of pairs of body name prefixes for which collisions should be ignored.
            Defaults to [].
        attached_body_name (str, optional): The name of the body attached to the robot's
            end-effector. If provided, the function will maintain the pose of this body
            relative to the end-effector when testing configurations. Defaults to None.
        check_resolution (float, optional): The resolution for state validity checking.
            Defaults to 0.01.

    Returns:
        ob.SpaceInformation: The OMPL SpaceInformation object for the UR5e robot with MuJoCo
            validity checking.
    """

    # instantiate on our C-space
    state_info = ob.SpaceInformation(get_ur5e_state_space(model))

    #TODO create and set the validity checker
    validity_checker = ...
    state_info.setStateValidityChecker(validity_checker)

    # set the state validity checking resolution
    state_info.setStateValidityCheckingResolution(check_resolution)

    return state_info


def get_mp_problem(
    state_info: ob.SpaceInformation,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    goal_threshold: float = 0.01,
) -> ob.ProblemDefinition:
    """
    Create an OMPL motion planning problem definition for the UR5e robot in MuJoCo.

    Args:
        state_info (ob.SpaceInformation): The OMPL space information object.
        start_q (np.ndarray): The start joint configuration for the UR5e robot.
        goal_q (np.ndarray): The goal joint configuration for the UR5e robot.
        goal_threshold (float, optional): The threshold for considering the goal reached.
            Defaults to 0.01.

    Returns:
        ob.ProblemDefinition: The OMPL problem definition for motion planning of the UR5e robot
            in MuJoCo.
    """
    # create a problem definition
    pdef = ob.ProblemDefinition(state_info)

    # create start state
    start_state = ob.State(state_info.getStateSpace())
    for i in range(UR5E_NUM_JOINTS):
        start_state[i] = start_q[i]

    # create goal state
    goal_state = ob.State(state_info.getStateSpace())
    for i in range(UR5E_NUM_JOINTS):
        goal_state[i] = goal_q[i]

    # set start and goal states
    pdef.setStartAndGoalStates(start_state, goal_state, goal_threshold)

    return pdef


def get_planner_for_problem(
    start_q: np.ndarray,
    goal_q: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled_bodies_prefix: list[str],
    exclude_collision_prefix_pairs: list[tuple[str, str]] = [],
    attached_body_name: str = None,
    check_resolution: float = 0.01,
    goal_threshold: float = 0.01,
) -> list[np.ndarray] | None:
    """
    Create and configure an OMPL planner for the UR5e robot in MuJoCo.

    Args:
        start_q (np.ndarray): The start joint configuration for the UR5e robot.
        goal_q (np.ndarray): The goal joint configuration for the UR5e robot.
        model (mujoco.MjModel): The MuJoCo model object.
        data (mujoco.MjData): The MuJoCo data object.
        controlled_bodies_prefix (list[str]): List of prefixes for body names
            that are considered controlled/moving bodies for which collisions
            should be detected.
        exclude_collision_prefix_pairs (list[tuple[str, str]], optional): List
            of pairs of body name prefixes for which collisions should be ignored.
            Defaults to [].
        attached_body_name (str, optional): The name of the body attached to the robot's
            end-effector. If provided, the function will maintain the pose of this body
            relative to the end-effector when testing configurations. Defaults to None.
        check_resolution (float, optional): The resolution for state validity checking.
            Defaults to 0.01.
        goal_threshold (float, optional): The threshold for considering the goal reached.
            Defaults to 0.01.
    """

    # get state space info
    state_info = get_state_space_info(
        model,
        data,
        controlled_bodies_prefix,
        exclude_collision_prefix_pairs,
        attached_body_name,
        check_resolution,
    )

    # get motion planning problem definition
    pdef = get_mp_problem(state_info, start_q, goal_q, goal_threshold)

    #TODO initialize the planner
    planner = ...

    # configure the planner with the problem definition
    planner.setProblemDefinition(pdef)

    # finalize the planner setup
    planner.setup()

    return planner


def plan_motion(
    planner: ob.Planner, time_limit: float = 5.0, interpolate: int = 0
) -> tuple[list[np.ndarray] | None, ob.PlannerStatus]:
    """
    Plan a motion using the given OMPL planner within the specified time limit.

    Args:
        planner (ob.Planner): The OMPL planner object.
        time_limit (float, optional): The time limit for planning in seconds.
            Defaults to 5.0.
        interpolate (int, optional): The number of waypoints to interpolate
            between each pair of waypoints in the solution path for smoothing.
            Defaults to 0 (no interpolation).

    Returns:
        tuple[list[np.ndarray] | None, ob.PlannerStatus]: A tuple containing the list of
            waypoints in the planned path (or empty list if no solution found) and the
            planner status.
    """

    solved = planner.solve(time_limit)
    sys.stdout.flush()  # flush output to ensure logs are printed before proceeding

    # print the planning result status
    if solved.getStatus() == ob.PlannerStatus.EXACT_SOLUTION:
        print("Found exact solution.")
    elif solved.getStatus() == ob.PlannerStatus.APPROXIMATE_SOLUTION:
        print("Found approximate solution.")
    else:
        print("No solution found.")

    # a container for the waypoints
    waypoints = []

    # get the solution path
    path = planner.getProblemDefinition().getSolutionPath()

    # if no path found, return empty waypoints list
    if path is None:
        return waypoints, solved.getStatus()

    print("number of waypoints in solution:", path.getStateCount())

    # optional: interpolate the waypoints for smoother path
    if interpolate > 0:
        path.interpolate(interpolate)

    # extract the waypoints from the path
    for i in range(path.getStateCount()):
        state = path.getState(i)
        waypoint = [state[j] for j in range(6)]
        waypoints.append(waypoint)

    return waypoints, solved.getStatus()
