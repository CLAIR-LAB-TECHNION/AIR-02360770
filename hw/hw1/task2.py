"""Homework 1, Task 2: PID controller in configuration space.

This module provides:
- The abstract policy class and a function for running and rendering policies.
- A helper for finding the `ctrl` and `qvel` indices of an entity.
- Proportional and PID joint position control policies.
- Gain tuning utilities, and the PID gains used in the rest of the assignment.
"""

from abc import ABC, abstractmethod

import mediapy as media
import mujoco
import numpy as np

from task1 import get_entity_qpos_idx


class Policy(ABC):
    """
    Abstract base class for control policies in MuJoCo.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model

    @abstractmethod
    def set_control(self, data: mujoco.MjData):
        """
        Set the control inputs for the actuators in the data object.
        Args:
            data (mujoco.MjData): The MuJoCo data object.
        """
        pass

    def reset(self):
        """
        Reset the policy to its initial state. By default there is nothing to reset.
        Policies that keep state between control steps override this method.
        """
        pass


def run_and_render_policy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    policy: Policy,
    duration: float,
    framerate: int,
    camera: str = "frontview",
    callback: callable = None,
):
    """
    Run a simulation with the given policy and render the result as a video.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.
        data (mujoco.MjData): The MuJoCo data object.
        policy (Policy): The control policy to use.
        duration (float): The duration of the simulation in seconds.
        framerate (int): The framerate for rendering the video in Hz.
        camera (str): The name of the camera to use for rendering. defaults to "frontview".
        callback (callable, optional): A callback function that takes (model, data, policy) as arguments.
            It is called at each simulation step. If it returns False, the simulation stops early.
    """
    mujoco.mj_resetData(model, data)  # Reset state and time.

    # Update the simulation state.
    # this is required to ensure that the data structure reflects the reset state
    # before calling the policy for the first time.
    mujoco.mj_forward(model, data)

    # container for rendered frames
    frames = []
    with mujoco.Renderer(model) as renderer:  # initialize renderer

        # simulate for the specified duration
        while data.time < duration:
            # simulate one time step
            policy.set_control(data)

            # advance the simulation by one time step
            mujoco.mj_step(model, data)

            # capture frame at the specified framerate
            if len(frames) < data.time * framerate:
                # update scene info according to current state for specific camera
                renderer.update_scene(data, camera=camera)

                # render the scene
                pixels = renderer.render()
                frames.append(pixels)

            # if a callback is provided, call it with the current model, data, and policy
            if callback is not None:
                continue_run = callback(model, data, policy)

                # if the callback returns False, stop the simulation
                if not continue_run:
                    break

    # show the captured video within the notebook if frames were recorded
    if framerate > 0:
        media.show_video(frames, fps=framerate)


def get_entity_ctrl_idx(model: mujoco.MjModel, prefix: str = "") -> list[int]:
    """
    Get the control indices for a specific entity in the MuJoCo model. The main
    purpose of this function is to help identify which parts of the `ctrl` array
    correspond to the robot's or gripper's actuators.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.
        prefix (str): The prefix to filter the entities by.

    Returns:
        list[int]: The control indices of the entities matching the prefix.
    """
    #TODO get the actuator IDs 
    return ...


class ProportionalJointPositionControlPolicy(Policy):
    def __init__(self, model: mujoco.MjModel, target_qpos: np.ndarray, kp: float):
        """
        A simple proportional controller in joint space.

        Args:
            model (mujoco.MjModel): MuJoCo model.
            target_qpos (np.ndarray): Target joint configuration (shape: [6] for UR5e).
            kp (float): Proportional gain.
        """
        super().__init__(model)
        self.target_qpos = target_qpos
        self.kp = kp

        # get the qpos and ctrl indices for the robot joints
        self.qpos_idx = get_entity_qpos_idx(model, prefix="robot_")
        self.ctrl_idx = get_entity_ctrl_idx(model, prefix="robot_")

    def set_control(self, data: mujoco.MjData):
        """
        Compute joint position commands to drive the robot to the target configuration.
        """
        # TODO Compute desired joint torque (proportional term)
        tau = ...

        # Set control inputs (assuming motor control)
        data.ctrl[self.ctrl_idx] = tau


class TargetDiffCallback:
    """
    A callback to record the norm of the difference between the current joint positions
    and the target joint positions at each simulation step.
    """

    def __init__(self):
        # a container to store the target differences
        self.target_diffs = []

    def __call__(self, model, data, policy):
        # Store the relevant information in the target_diffs list
        target_diff = np.linalg.norm(data.qpos[policy.qpos_idx] - policy.target_qpos)
        self.target_diffs.append(target_diff)

        return True


def get_entity_qvel_idx(model: mujoco.MjModel, prefix: str = "") -> list[int]:
    """
    Get the indices of the joint velocity (qvel) states for a specific entity.
    The main purpose of this function is to help identify which parts of the `qvel`
    array correspond to the robot's joints.

    Args:
        model (mujoco.MjModel): MuJoCo model.
        prefix (str, optional): Prefix to filter joint names. Defaults to "".

    Returns:
        list[int]: List of indices for the joint velocity states.
    """
    #TODO get the qvel indices
    return ...


class PIDJointPositionControlPolicy(Policy):
    def __init__(
        self,
        model: mujoco.MjModel,
        target_qpos: np.ndarray,
        kp: float,
        ki: float,
        kd: float,
        integral_decay: float = 0.95
    ):
        """
        A PID controller in joint space for configuration control.

        Args:
            model (mujoco.MjModel): MuJoCo model.
            target_qpos (np.ndarray): Target joint configuration (shape: [6] for UR5e).
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
        """
        super().__init__(model)
        self.target_qpos = target_qpos
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Initialize integral error accumulator
        self.integral_error = np.zeros_like(target_qpos)
        self.integral_decay = integral_decay

        # get the qpos, qvel, and ctrl indices for the robot joints
        self.qpos_idx = get_entity_qpos_idx(model, prefix="robot_")
        self.qvel_idx = get_entity_qvel_idx(model, prefix="robot_")
        self.ctrl_idx = get_entity_ctrl_idx(model, prefix="robot_")

    def set_control(self, data: mujoco.MjData):
        """
        Compute joint position commands to drive the robot to the target configuration.
        """
        dt = self.model.opt.timestep

        #TODO get the desired torque using PID control
        torque_desired = ...

        # Set control inputs (assuming velocity control)
        data.ctrl[self.ctrl_idx] = torque_desired

    def reset(self):
        # clear the accumulated error
        self.integral_error = np.zeros_like(self.integral_error)


#TODO tune monotonicity tolerance value EPSILON
# This value determines the maximum allowed increase in error to still consider the
# sequence monotonic.
# This corresponds to a small tolerance to account for overshooting and oscillations.
# Tuning it is up to you: a value that is too small rejects every controller as
# non-monotone, while a value that is too large accepts controllers that oscillate.
EPSILON = 1e-4


def get_best_params(
    diff_data: dict[tuple[float, float, float], list[float]],
) -> tuple[tuple[float, float, float], float, int, bool]:
    """
    Get the best PID controller parameters based on the minimum difference in target
    position over time, prioritizing monotonic decrease in error. An increase in error
    of up to `EPSILON` still counts as monotonic.

    Args:
        diff_data (dict[tuple[float, float, float], list[float]]):
            A dictionary mapping (kp, ki, kd) tuples to lists of target position differences
            over time.

    Returns:
        tuple[tuple[float, float, float], float, int, bool]: The best (kp, ki, kd) values,
            their minimum target position difference, the timestep at which it was
            reached, and whether these values are monotonic.
    """
    # the minimum value found so far (min_diff, time_index).
    # we use this tuple to compare both the minimum difference and the time it occurs
    # using lexicographic ordering.
    min_val = (float("inf"), float("inf"))

    # container for the best parameters found
    best_params = None

    # flag to indicate if we have found monotonic parameters
    found_monotone_params = False

    # iterate over all parameter combinations and their corresponding difference data
    for params, diffs in diff_data.items():
        min_diff = min_diff_time = prev_diff = float("inf")
        monotone = True  # assume monotonic until proven otherwise

        # iterate over the difference values to find mimimum and check monotonicity.
        # if not monotonic, keep the minimum value found up until monotonicity is violated.
        for i in range(len(diffs)):

            # first check for "approximate" monotonicity
            if diffs[i] > prev_diff + EPSILON:
                monotone = False
                break  # Stop checking if the sequence is no longer monotonic
            prev_diff = diffs[i]

            # then check for minimum difference
            if diffs[i] < min_diff:
                min_diff = diffs[i]
                min_diff_time = i

        # compare with the best found so far.
        # the first monotone parameters replace any non-monotone ones, even if those
        # reached a smaller difference before they stopped being monotonic.
        first_monotone = monotone and not found_monotone_params
        if first_monotone or (min_diff, min_diff_time) < min_val:

            # if we have already found monotone parameters, skip non-monotone ones.
            # otherwise, accept the new parameters if minimal.
            if found_monotone_params and not monotone:
                continue  # Skip non-monotone params if we already found monotone ones

            # set minimal parameters
            min_val = (min_diff, min_diff_time)
            best_params = params

            # update monotone flag
            # becomes true the first time we find monotone parameters.
            # once true, will not be turned false again.
            found_monotone_params = monotone

    min_diff, min_time = min_val
    return best_params, min_diff, min_time, found_monotone_params


def present_best_params(
    best_params, min_diff, min_time, found_monotone_params, label: str
):
    """
    A formatted printout of the best PID parameters found.
    """
    best_kp, best_ki, best_kd = best_params
    print(
        f"Best {label}{' non monotone' if not found_monotone_params else ''} parameters: Kp={best_kp}, Ki={best_ki}, Kd={best_kd} with min diff {min_diff} at time {min_time}"
    )


#TODO Set the desired PID gaiins for the pick-and-place task
KP = ...
KI = ...
KD = ...
