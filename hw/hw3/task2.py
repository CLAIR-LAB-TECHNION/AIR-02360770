"""Homework 3, Task 2: RL-guided MCTS.

This module provides:
- The Gymnasium version of the block stacking environment.
- The cold start callback, and utilities for training the RL agent.
- An edge type for PUCT, which takes the prior of its action into account.
- Guiding values from the RL agent, and the selection and expansion stages that use them.
- The PUCT search, and the PUCT planning time.
"""

from dataclasses import dataclass
import time
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import torch
from tqdm.auto import tqdm

from scene import model
# NOTE: read the offset range as `task1.OFFSET_RANGE`. This way, changes to it in
# task1.py are picked up as soon as you make them.
import task1
from task1 import (
    BlockStackingAction,
    BlockStackingEnv,
    BlockStackingState,
    MCTSEdge,
    MCTSNode,
    Simulator,
    backpropagate,
    dpw_select_outcome,
    expand_outcome,
    get_stacking_order,
    progressive_widening_limit,
)


class BlockStackingGymEnv(BlockStackingEnv, gym.Env):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        BlockStackingEnv.__init__(self, model, data)
        gym.Env.__init__(self)

    def reset(self, **kwargs):
        state = BlockStackingEnv.reset(self)
        obs = self.state_to_obs(state)
        return obs, {}

    def state_to_obs(self, state: BlockStackingState):
        num_offsets = len(self._stacking_order)
        obs = np.zeros((num_offsets, 2), dtype=np.float32)
        for i, offset in enumerate(state.stacked_blocks_offsets):
            obs[i] = offset

        return obs.flatten()

    def step(self, action: np.ndarray, framerate=0):
        block_action = BlockStackingAction(offset_x=action[0], offset_y=action[1])
        next_state, reward, done, info = BlockStackingEnv.step(self, block_action, framerate=framerate)
        obs = self.state_to_obs(next_state)
        return obs, reward, done, False, info

    @property
    def observation_space(self):
        num_offsets = len(self._stacking_order)
        low = np.array([task1.OFFSET_RANGE[0]] * num_offsets * 2, dtype=np.float32)
        high = np.array([task1.OFFSET_RANGE[1]] * num_offsets * 2, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=np.array([task1.OFFSET_RANGE[0], task1.OFFSET_RANGE[0]], dtype=np.float32),
            high=np.array([task1.OFFSET_RANGE[1], task1.OFFSET_RANGE[1]], dtype=np.float32),
            dtype=np.float32,
        )


class ZeroActionWarmupCallback(BaseCallback):
    def __init__(self, warmup_steps: int, verbose=0):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps

    def _on_step(self) -> bool:
        if self.num_timesteps < self.warmup_steps:
            # Override action to be zero during warmup
            self.locals["actions"] = np.zeros_like(self.locals["actions"])
        return True


def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

# this is a small network so training should be fast even on CPU.
# Let's still use GPU if available.
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.mps.is_available() else "cpu")
)


@dataclass
class MCTSEdgeWithPrior(MCTSEdge):
    prior: float = 0.0  # prior probability of selecting this edge

    def ucb(self, exploration_constant):
        """
        Compute the UCB value for this edge using its prior probability.

        Args:
            exploration_constant: The exploration constant used in UCB calculation.

        Returns:
            ucb_score: The UCB value of this edge using its prior.
        """
        if self.visits == 0:
            # return the prior value for unvisited edges
            return self.prior

        #TODO Compute the UCB value using the formula with prior
        pucb_score = ...

        return pucb_score


def state_to_obs(state: BlockStackingState) -> np.ndarray:
    """
    Convert a block stacking state to an observation for the RL agent. This is the same
    conversion as `BlockStackingGymEnv.state_to_obs`, but it does not need an environment
    that was reset. The functions below use it to query the RL agent for any state.

    Args:
        state: The block stacking state.

    Returns:
        obs: The observation of the state.
    """
    num_offsets = len(get_stacking_order(model))
    obs = np.zeros((num_offsets, 2), dtype=np.float32)
    for i, offset in enumerate(state.stacked_blocks_offsets):
        obs[i] = offset

    return obs.flatten()


def get_action_dist(rl_alg: SAC, state: BlockStackingState):
    """
    Get the action distribution parameters (mean and std) from the RL policy for the given state.

    Args:
        rl_alg: The trained RL algorithm.
        state: The block stacking state.

    Returns:
        mean: The mean of the action distribution.
        std: The standard deviation of the action distribution.
    """
    obs = state_to_obs(state)
    obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
    with torch.no_grad():
        mean, std, _ = rl_alg.policy.actor.get_action_dist_params(obs_tensor)
    return mean, std

def get_state_value(rl_alg: SAC, state: BlockStackingState):
    """
    Get the state value (Q-value) from the RL critic for the given state.

    Args:
        rl_alg: The trained RL algorithm.
        state: The block stacking state.

    Returns:
        state_value: The estimated state value.
    """
    obs = state_to_obs(state)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(rl_alg.device).unsqueeze(0)
    with torch.no_grad():
        # 1. extract latent features from the observation
        features = rl_alg.policy.actor.extract_features(obs_tensor, rl_alg.policy.actor.features_extractor)
        latent_pi = rl_alg.policy.actor.latent_pi(features)

        # 2. Get the deterministic action (mean of the distribution)
        mean_actions = rl_alg.policy.actor.mu(latent_pi) # Raw mean from the network

        # 3. Get Q-value for this action
        q1, q2 = rl_alg.policy.critic(obs_tensor, mean_actions)

        # 4. Take minimum to prevent overestimation (standard SAC practice)
        state_value = torch.min(q1, q2).cpu().detach().numpy().squeeze()

    return state_value


def set_node_guiding_values(
    node: MCTSNode, sim: Simulator, rl_alg: SAC
):
    """
    Set guiding values (action priors and state value) for the given node using the RL agent.

    Args:
        node: The MCTS node to set guiding values for.
        sim: The simulator used to check terminality.
        rl_alg: The trained RL algorithm.
    """
    if not node.is_terminal(sim):
        if getattr(node, "value", None) is None or node.value is None:
            # estimate the value of the node using the RL critic
            value = get_state_value(rl_alg, node.state)
            node.value = value
        if getattr(node, "actions_dist", None) is None or node.actions_dist is None:
            # get action distribution from the RL policy
            action_mean, action_std = get_action_dist(rl_alg, node.state)
            node.actions_dist = (action_mean, action_std)


def dpw_select_outcome_and_set_priors_and_values(
    edge: MCTSEdge, sim: Simulator, pw_k, pw_alpha, rl_alg: SAC
) -> MCTSNode:
    """
    Select an outcome node from the given edge using Progressive Widening,
    and set action priors and state values using the RL agent.

    Args:
        edge: The edge to select an outcome from.
        sim: The simulator used for expansion.
        pw_k: Progressive widening scaling constant.
        pw_alpha: Progressive widening throttle constant.
        rl_alg: The trained RL algorithm.

    Returns:
        outcome_node: The selected outcome node.
    """

    selected_outcome = dpw_select_outcome(edge, sim, pw_k, pw_alpha)

    #TODO set guiding values for the selected outcome node
    ...

    return selected_outcome


def dpw_select_node_with_action_priors(
    root: MCTSNode, sim: Simulator, exploration_constant, pw_k, pw_alpha, rl_alg: SAC
) -> MCTSNode:
    """
    Select a node from the tree using Double Progressive Widening,
    incorporating action priors from the RL agent.

    Args:
        root: The root node from which to start selection.
        sim: The simulator used for expansion.
        exploration_constant: The exploration constant used in UCB calculation.
        pw_k: Progressive widening scaling constant.
        pw_alpha: Progressive widening throttle constant.
        rl_alg: The trained RL algorithm.

    Returns:
        selected_node: The selected node.
    """
    current_node = root

    while not current_node.is_terminal(sim):
        # check progressive widening condition
        # this is like checking if the node is fully expanded in the standard MCTS
        pw_limit = progressive_widening_limit(current_node.visits, pw_k, pw_alpha)
        if len(current_node.outgoing_edges) < pw_limit:
            break  # Stop selection if we can add more edges

        # Select the best edge using UCB
        best_edge = current_node.top_ucb_edge(exploration_constant)

        #TODO choose an outcome node using DPW with priors
        current_node = ...

    #TODO set the guiding values fo the selected node
    ...

    return current_node


def dpw_expand_node_with_action_prior(
    node: MCTSNode, sim: Simulator, rl_alg: SAC
) -> MCTSNode:
    """
    Expand a new action edge and outcome node from the given node,
    using action priors from the RL agent.

    Args:
        node: The node to expand.
        sim: The simulator used for expansion.
        rl_alg: The trained RL algorithm.

    Returns:
        outcome_node: The newly created outcome node.
    """
    prior_mean, prior_std = node.actions_dist

    # randomly select an action to expand
    action, action_log_prob = rl_alg.policy.actor.action_dist.log_prob_from_params(prior_mean, prior_std)
    action = action.cpu().detach().numpy()
    action = rl_alg.policy.unscale_action(action).squeeze(0)
    action = BlockStackingAction(offset_x=action[0], offset_y=action[1])
    action_prob = torch.exp(action_log_prob).item()

    # create new edge
    new_edge = MCTSEdgeWithPrior(parent=node, action=action, outcomes=[], prior=action_prob)
    node.outgoing_edges.append(new_edge)

    #TODO expand one outcome for the new edge and set its guiding values
    outcome_node = ...
    set_node_guiding_values(outcome_node, sim, rl_alg)

    return outcome_node


class ContinuousActionStochasticOutcomePUCT:
    def __init__(
        self,
        sim: Simulator,
        rl_alg: SAC,
        pw_k: float,
        pw_alpha: float,
    ):
        self.sim = sim
        self.rl_alg = rl_alg
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

            # selection
            try:
                node_to_expand = dpw_select_node_with_action_priors(
                    root_node,
                    self.sim,
                    exploration_constant,
                    self.pw_k,
                    self.pw_alpha,
                    self.rl_alg,
                )

                # expansion
                if node_to_expand.is_terminal(self.sim):
                    new_child_node = node_to_expand
                else:
                    new_child_node = dpw_expand_node_with_action_prior(node_to_expand, self.sim, self.rl_alg)

                # rollout
                if new_child_node.is_terminal(self.sim):
                    reward = new_child_node.reward
                else:
                    reward = new_child_node.value

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


#TODO select a planning time
PLANNING_TIME = ...  # seconds
