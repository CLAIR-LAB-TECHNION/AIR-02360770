"""Homework 1, Task 4: task execution policy.

This module provides:
- A sequential state machine policy and its nodes.
- Motion planning nodes for reaching target configurations and poses.
- Gripper nodes.
- The final pick-and-place policy, and the callback used to evaluate it.
"""

from abc import ABC, abstractmethod
from copy import deepcopy

import mujoco
import numpy as np
from ompl import base as ob
from tqdm.auto import tqdm

from scene import HOME_CONFIG, model
from task1 import get_entity_qpos_idx
# NOTE: read the PID gains as `task2.KP`, `task2.KI`, and `task2.KD`. This way, the
# values you set in task2.py are picked up as soon as you change them.
import task2
from task2 import PIDJointPositionControlPolicy, Policy
from task3 import (
    GripperController,
    StayInPlacePolicy,
    get_planner_for_problem,
    is_collision_state,
    plan_motion,
    select_pose_configuration,
)


class PolicySMNode(ABC):
    """
    An abstract base class for a state machine node that defines a policy
    that runs while this node is active.
    """

    @abstractmethod
    def get_policy(self, model: mujoco.MjModel, data: mujoco.MjData) -> Policy:
        """
        Get the policy to be executed while this state machine node is active.

        Args:
            model (mujoco.MjModel): The MuJoCo model object.
            data (mujoco.MjData): The MuJoCo data object.

        Returns:
            Policy: The policy to be executed.
        """
        pass

    @abstractmethod
    def is_complete(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """
        Check if the state machine node is complete.

        Args:
            model (mujoco.MjModel): The MuJoCo model object.
            data (mujoco.MjData): The MuJoCo data object.

        Returns:
            bool: True if the state machine node is done with its execution, False otherwise.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the state machine node to its initial state.
        """
        pass


class SMPolicy(Policy):
    """
    A policy that implements a state machine composed of multiple PolicySMNode objects. Nodes
    are executed in sequence, with each node providing its own policy and completion criteria.
    """

    def __init__(self, model: mujoco.MjModel, nodes: list[PolicySMNode]):
        """
        Initialize the state machine policy with the given model and nodes.

        Args:
            model (mujoco.MjModel): The MuJoCo model object.
            nodes (list[PolicySMNode]): The list of state machine nodes defining the state
                machine.
        """
        super().__init__(model)
        self.nodes = nodes
        self.cur_node_idx = 0
        self.cur_policy = None

        self.pbar = None

    def set_control(self, data: mujoco.MjData):
        if self.pbar is None:  # initialize progress bar
            self.pbar = tqdm(
                total=len(self.nodes), desc=f"{self.__class__.__name__} Progress"
            )

        if self.is_complete:
            data.ctrl[:] = 0.0  # no control if state machine is complete
            return

        # get the current node's policy if not already done
        if self.cur_policy is None:
            self.cur_policy = self.nodes[self.cur_node_idx].get_policy(self.model, data)

        # set control using the current node's policy
        self.cur_policy.set_control(data)

        # check if the current node is complete
        if self.nodes[self.cur_node_idx].is_complete(self.model, data):
            # advance to the next node
            self.cur_node_idx += 1
            self.cur_policy = None  # reset current policy for the new node

            # update the progress bar
            if self.pbar is not None:
                self.pbar.update(1)

            # close the progress bar if state machine is complete
            if self.is_complete and self.pbar is not None:
                self.pbar.close()
                self.pbar = None

    @property
    def is_complete(self) -> bool:
        """
        Check if the state machine has finished executing all nodes.
        """

        return self.cur_node_idx >= len(self.nodes)

    def reset(self):
        """
        Reset the state machine to the initial node.
        """

        # reset state machine state
        self.cur_node_idx = 0
        self.cur_policy = None

        # reset all nodes
        for node in self.nodes:
            node.reset()

        # close the progress bar if it exists
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class GoToConfigSMNode(PolicySMNode):
    """
    A state machine node that moves the robot to a target joint configuration.
    """

    def __init__(
        self,
        target_config: np.ndarray,
        goal_tolerance: float = 0.01,
    ):
        """
        Initialize the GoToConfigSMNode with the target configuration and goal tolerance.

        Args:
            target_config (np.ndarray): The target joint configuration for the robot.
            goal_tolerance (float, optional): The tolerance for considering the goal reached.
                Defaults to 0.01.
        """
        self.goal_tolerance = goal_tolerance
        self.target_config = target_config

        self.robot_qpos_idx = get_entity_qpos_idx(model, "robot")

    def get_policy(self, model: mujoco.MjModel, data: mujoco.MjData) -> Policy:

        #TODO create a PID controller policy to go to the target configuration
        return ...

    def is_complete(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:

        #TODO check if the robot is within goal tolerance of the target configuration
        return ...

    def reset(self):
        # nothing to reset for this node
        pass


class WaypointPolicy(SMPolicy):
    """
    A state machine policy that moves the robot through a series of waypoints.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        waypoints: list[np.ndarray],
        waypoint_tolerance: float = 0.01,
    ):
        """
        Initialize the WaypointPolicy with the given waypoints and tolerance.

        Args:
            model (mujoco.MjModel): The MuJoCo model.
            waypoints (list[np.ndarray]): The list of waypoints to follow.
            waypoint_tolerance (float, optional): The tolerance for reaching each waypoint.
                Defaults to 0.01.
        """
        #TODO create GoToConfigSMNode for each waypoint
        nodes = ...
        super().__init__(model, nodes)


# After the state machine policy has finished executing all its nodes, we
# can use the following callback function to end the simulation. If we
# don't do this, the simulation will continue running until it has fulfilled
# the input duration, which is not necessary once the policy is complete.

def end_on_sm_complete_callback(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    policy: SMPolicy,
) -> bool:
    """
    A callback function to end the simulation when the state machine policy has completed
    execution.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.
        data (mujoco.MjData): The MuJoCo data object.
        policy (SMPolicy): The state machine policy being executed.

    Returns:
        bool: True if the simulation should continue, False to end the simulation.
    """
    return not policy.is_complete


class MotionPlanToConfigSMNode(PolicySMNode):
    """
    A state machine node that plans and executes a motion to a target joint configuration.
    """

    def __init__(
        self,
        goal_q: np.ndarray,
        controlled_bodies_prefix: list[str],
        exclude_collision_prefix_pairs: list[tuple[str, str]] = [],
        attached_body_name: str = None,
        check_resolution: float = 0.01,
        goal_threshold: float = 0.01,
        goal_tolerance: float = 0.1,
        planning_time_limit: float = 5.0,
        expected_status: ob.PlannerStatus = ob.PlannerStatus.EXACT_SOLUTION,
        max_planning_retries: int = 5,
        reset_planning_on_retry: bool = True,
        plan_interpolate: int = 0,
    ):
        """
        Initialize the MotionPlanToConfigSMNode with the given parameters.

        Args:
            goal_q (np.ndarray): The target joint configuration for the robot.
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
            goal_tolerance (float, optional): The tolerance for reaching the goal configuration.
                Defaults to 0.1.
            planning_time_limit (float, optional): The time limit for motion planning in seconds.
                Defaults to 5.0.
            expected_status (ob.PlannerStatus, optional): The expected planner status for
                successful planning. Defaults to ob.PlannerStatus.EXACT_SOLUTION.
            max_planning_retries (int, optional): The maximum number of planning retries
                if planning fails. Defaults to 5.
            reset_planning_on_retry (bool, optional): Whether to reset the planner on each retry.
                Defaults to True.
            plan_interpolate (int, optional): The number of waypoints to interpolate
                between each pair of waypoints in the solution path for smoothing.
                Defaults to 0 (no interpolation).
        """
        self.goal_q = goal_q
        self.controlled_bodies_prefix = controlled_bodies_prefix
        self.exclude_collision_prefix_pairs = exclude_collision_prefix_pairs
        self.attached_body_name = attached_body_name
        self.check_resolution = check_resolution
        self.goal_threshold = goal_threshold
        self.goal_tolerance = goal_tolerance
        self.planning_time_limit = planning_time_limit
        self.expected_status = expected_status
        self.max_planning_retries = max_planning_retries
        self.reset_planning_on_retry = reset_planning_on_retry
        self.plan_interpolate = plan_interpolate

        self.wp_policy = None

        self.robot_qpos_idx = get_entity_qpos_idx(model, "robot")

    def get_policy(self, model: mujoco.MjModel, data: mujoco.MjData) -> Policy:
        # get the robot's start configuration from the current Mujoco data.
        # This will be used as the start state for motion planning.
        start_q = data.qpos[self.robot_qpos_idx]

        # make a copy of data to avoid modifying the original data during planning
        data_mp_copy = deepcopy(data)

        #TODO create the motion planner
        planner = ...

        # attempt motion planning with retries if necessary
        waypoints = [self.goal_q]  # default to goal only if planning fails
        for i in range(self.max_planning_retries):
            print(f'Planning attempt {i+1}/{self.max_planning_retries}...')

            # reset the planner if required
            if self.reset_planning_on_retry:
                planner.clear()

            #TODO plan the motion
            waypoints, status = ...
            print(f'Solution status: {status}')

            # check if the planning was successful
            if status == self.expected_status:
                break  # successful planning
            else:
                if i < self.max_planning_retries - 1:
                    print(f'Unexpected planner status. Retrying motion planning...')
                else:
                    print('Max planning retries reached. Proceeding with approximate solution.')
                
                # add goal as last waypoint to ensure we at least reach the goal
                waypoints.append(self.goal_q)

        #TODO create a waypoint policy to follow the planned waypoints
        self.wp_policy = ...

        return self.wp_policy

    def is_complete(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        return self.wp_policy is not None and self.wp_policy.is_complete

    def reset(self):
        # reset waypoint policy to close any existing state
        if self.wp_policy is not None:
            self.wp_policy.reset()

        # toss out the existing waypoint policy
        self.wp_policy = None


def get_home_policy(model: mujoco.MjModel) -> SMPolicy:
    """
    Get a state machine policy that moves the robot to the home configuration.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.

    Returns:
        SMPolicy: The state machine policy.
    """
    return SMPolicy(
        model,
        nodes=[
            #TODO create MotionPlanToConfigSMNode to go to HOME_CONFIG
            ...
        ]
    )


class MotionPlanToPoseSMNode(MotionPlanToConfigSMNode):
    """
    A state machine node that plans and executes a motion to reach one of the specified
    target end-effector poses.
    """

    def __init__(
        self,
        goal_poses: list[tuple[np.ndarray, np.ndarray]],
        controlled_bodies_prefix: list[str],
        exclude_collision_prefix_pairs: list[tuple[str, str]] = [],
        attached_body_name: str = None,
        check_resolution: float = 0.01,
        goal_threshold: float = 0.01,
        goal_tolerance: float = 0.1,
        planning_time_limit: float = 5.0,
        expected_status: ob.PlannerStatus = ob.PlannerStatus.EXACT_SOLUTION,
        max_planning_retries: int = 5,
        reset_planning_on_retry: bool = True,
        plan_interpolate: int = 0,
    ):
        """
        Initialize the MotionPlanToPoseSMNode with the given parameters.

        Args:
            goal_poses (list[tuple[np.ndarray, np.ndarray]]): The target poses for the robot.
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
            goal_tolerance (float, optional): The tolerance for reaching the goal configuration.
                Defaults to 0.1.
            planning_time_limit (float, optional): The time limit for motion planning in seconds.
                Defaults to 5.0.
            expected_status (ob.PlannerStatus, optional): The expected planner status for
                successful planning. Defaults to ob.PlannerStatus.EXACT_SOLUTION.
            max_planning_retries (int, optional): The maximum number of planning retries
                if planning fails. Defaults to 5.
            reset_planning_on_retry (bool, optional): Whether to reset the planner on each retry.
                Defaults to True.
            plan_interpolate (int, optional): The number of waypoints to interpolate
                between each pair of waypoints in the solution path for smoothing.
                Defaults to 0 (no interpolation).
        """

        super().__init__(
            goal_q=None,  # will be set later
            controlled_bodies_prefix=controlled_bodies_prefix,
            exclude_collision_prefix_pairs=exclude_collision_prefix_pairs,
            attached_body_name=attached_body_name,
            check_resolution=check_resolution,
            goal_threshold=goal_threshold,
            goal_tolerance=goal_tolerance,
            planning_time_limit=planning_time_limit,
            expected_status=expected_status,
            max_planning_retries=max_planning_retries,
            reset_planning_on_retry=reset_planning_on_retry,
            plan_interpolate=plan_interpolate,
        )
        self.goal_poses = goal_poses

    def get_policy(self, model: mujoco.MjModel, data: mujoco.MjData) -> Policy:

        # make a copy of data to avoid modifying the original data during pose selection
        data_pose_selector_copy = deepcopy(data)

        #TODO select a collision-free IK solution for one of the goal poses
        selected_config = ...

        # raise error if no collision-free IK solution found
        if selected_config is None:
            raise RuntimeError(
                "No collision-free IK solution found for the given goal poses."
            )
        
        # set the selected configuration as the goal for motion planning
        self.goal_q = selected_config
        return super().get_policy(model, data)

    def reset(self):
        # toss out the existing goal configuration
        self.goal_q = None

        # reset the base class state
        super().reset()


def get_to_box_policy(
    model: mujoco.MjModel,
    pre_grasp_poses: list[tuple[np.ndarray, np.ndarray]],
) -> SMPolicy:
    """
    Get a state machine policy that moves the robot to the pre-grasp pose above the
    red box.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.
        pre_grasp_poses (list[tuple[np.ndarray, np.ndarray]]): The pre-grasp poses above
            the red box.

    Returns:
        SMPolicy: The state machine policy.
    """
    return SMPolicy(
        model,
        nodes=[
            #TODO create motion planning nodes to go to the home configuration
            # and then to the box pre-grasp configuration
            ...
        ]
    )


class DurationConfigSMNode(PolicySMNode):
    """
    A state machine node that plans and executes a given policy for a fixed duration.
    """
    def __init__(
        self,
        policy: Policy,
        duration: float,
    ):
        """
        Initialize the DurationConfigSMNode with the given policy and duration.

        Args:
            policy (Policy): The policy to be executed.
            duration (float): The duration for which to execute the policy.
        """
        self.policy = policy
        self.duration = duration
        self.start_time = None

    def get_policy(self, model: mujoco.MjModel, data: mujoco.MjData) -> Policy:
        if self.start_time is None:
            self.start_time = data.time
        return self.policy

    def is_complete(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        #TODO check if the specified duration has elapsed
        # HINT: current time is available as `data.time`
        return ...

    def reset(self):
        # reset the start time and the policy
        self.start_time = None
        self.policy.reset()


class GripperControlPolicy(Policy):
    """
    A policy to control the gripper to open or close while keeping the robot
    in place.
    """
    def __init__(self, model: mujoco.MjModel, close: bool):
        """
        Initialize the GripperControlPolicy with the given model and gripper state.

        Args:
            model (mujoco.MjModel): The MuJoCo model object.
            close (bool): True to close the gripper, False to open it.
        """
        super().__init__(model)
        self.close = close

        #TODO initialize sub-policies
        self.stay_in_place_policy = ...
        self.gripper_controller = ...

    def set_control(self, data: mujoco.MjData):
        #TODO set controls for the gripper to open/close
        ...

        #TODO set controls for the robot to stay in place
        ...

    def reset(self):
        # the robot stays in place wherever the next run brings it
        self.stay_in_place_policy.reset()


def get_lift_box_policy(
    model: mujoco.MjModel,
    pre_grasp_poses: list[tuple[np.ndarray, np.ndarray]],
    grasp_poses: list[tuple[np.ndarray, np.ndarray]],
) -> SMPolicy:
    """
    Get a state machine policy that picks up the red box.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.
        pre_grasp_poses (list[tuple[np.ndarray, np.ndarray]]): The pre-grasp poses above
            the red box.
        grasp_poses (list[tuple[np.ndarray, np.ndarray]]): The grasp poses on the red box.

    Returns:
        SMPolicy: The state machine policy.
    """
    return SMPolicy(
        model,
        nodes=[
            #TODO create nodes to follow this behavior:
            # 1. Go to the home configuration
            # 2. Go to the box pre-grasp configuration
            # 3. go to the box grasp configuration
            # 4. Close the gripper to grasp the box
            # 5. Lift the box by moving back to the pre-grasp configuration
            ...
        ],
    )


def get_above_bin_pos(data: mujoco.MjData) -> np.ndarray:
    """
    Get the position above the green bin.

    Args:
        data (mujoco.MjData): The MuJoCo data object.

    Returns:
        np.ndarray: position above the green bin in world frame (3,)
    """
    #TODO find the position above the green bin
    above_bin_pos = ...

    return above_bin_pos


class PickAndPlacePolicy(SMPolicy):
    """
    A state machine policy that performs a pick-and-place task using motion planning.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        pre_grasp_poses: list[tuple[np.ndarray, np.ndarray]],
        grasp_poses: list[tuple[np.ndarray, np.ndarray]],
        place_poses: list[tuple[np.ndarray, np.ndarray]],
    ):
        """
        Initialize the PickAndPlacePolicy with the given model.

        Args:
            model (mujoco.MjModel): The MuJoCo model object.
            pre_grasp_poses (list[tuple[np.ndarray, np.ndarray]]): The pre-grasp poses above the red box.
            grasp_poses (list[tuple[np.ndarray, np.ndarray]]): The grasp poses on the red box.
            place_poses (list[tuple[np.ndarray, np.ndarray]]): The place poses above the green bin.
        """
        nodes = [
            # TODO create motion planning nodes to perform the full pick-and-place task:
            # STEP 1: Move to home configuration
            # STEP 2: Move to pre-grasp poses
            # STEP 3: Move to grasp poses
            # STEP 3.5: Close gripper to grasp the object
            # STEP 4: Move back to pre-grasp poses with the object
            # STEP 5: Move to above the bin poses
            # STEP 5.5: Open gripper to release the object
            # STEP 6: Move back to home configuration
            # Stay in place after execution
        ]
        super().__init__(model, nodes)


class EvaluatePolicyCallback:
    """
    A callback class to collect data used for evaluating the performance of the
    pick-and-place policy.
    """
    def __init__(self):
        self.has_collided = False
        self.task_success = False
        self.num_consecutive_timesteps_at_home = 0

        self.robot_qpos_idx = get_entity_qpos_idx(model, "robot")
        self._is_box_in_bin = False
        self._is_at_home_last_1s = False

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, policy: SMPolicy) -> bool:

        # check for collisions with the obstacles
        # we ignore collisions between the obstacle and the table since
        # the obstacles start rested on the table.
        if is_collision_state(
            model,
            data,
            controlled_bodies_prefix=["obstacle"],
            exclude_collision_prefix_pairs=[
                ("obstacle", "table"),  # obstacle can be on the table
            ],  # don't alert after knock down and on the floor
        ):
            if not self.has_collided:
                print("Collision with obstacle detected!")
            self.has_collided = True

        # check if the robot is at home configuration
        if np.linalg.norm(
            data.qpos[self.robot_qpos_idx] - HOME_CONFIG
        ) < 1e-1:
            self.num_consecutive_timesteps_at_home += 1
        else:
            self.num_consecutive_timesteps_at_home = 0

        # check if the robot has been at home for the last 1 second
        self._is_at_home_last_1s = self.at_home_last_1s()

        # check if the box is in the bin
        self._is_box_in_bin = self.box_in_bin(model, data)

        # task success is dependent on these two conditions holding
        self.task_success = self._is_box_in_bin and self._is_at_home_last_1s

        # test if the policy execution is done to potentially stop the simulation
        if policy.is_complete:
            if not self._is_box_in_bin:
                print("Policy ended without box in bin.")
            if not self._is_at_home_last_1s:
                print("Policy execution ended without being at home for last 1s.")
            return False  # stop execution
        return True  # continue execution

    def box_in_bin(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:

        # get box x,y position
        box_x, box_y, _ = data.body("red_box").xpos

        # get bin wall positions
        _, left_wall_y, _ = data.geom("green_bin_left_wall").xpos
        _, right_wall_y, _ = data.geom("green_bin_right_wall").xpos
        back_wall_x, _, _ = data.geom("green_bin_back_wall").xpos
        front_wall_x, _, _ = data.geom("green_bin_front_wall").xpos

        # check if box is within wall bounds
        within_wall_bounds = (left_wall_y <= box_y <= right_wall_y) and (
            back_wall_x <= box_x <= front_wall_x
        )

        # check if box is in contact with bin bottom
        in_contact_with_bin_bottom = False
        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            geom1_name = model.geom(contact.geom1).name
            geom2_name = model.geom(contact.geom2).name

            if (geom1_name == "red_box_geom" and geom2_name == "green_bin_bottom") or (
                geom2_name == "red_box_geom" and geom1_name == "green_bin_bottom"
            ):
                in_contact_with_bin_bottom = True
                break

        # return true if both conditions are met
        return within_wall_bounds and in_contact_with_bin_bottom

    def at_home_last_1s(self) -> bool:
        # check if the robot has been at home for the last 1 second
        return self.num_consecutive_timesteps_at_home * model.opt.timestep >= 1.0

    def reset(self):
        self.has_collided = False
        self.task_success = False
        self.num_consecutive_timesteps_at_home = 0

        self._is_box_in_bin = False
        self._is_at_home_last_1s = False
