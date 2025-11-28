from __future__ import annotations
import torch

from typing import Any, Callable, Dict, Tuple, cast

import numpy as np

# Import your environment (adjust import path if needed)
from rlo.envs import EndlessPlatformerEnv
from rlo.policies import Policy
from rlo.policies.param_nonlinear_jepa import ParamNonLinearPolicy_JEPA


def evaluate_candidate(
    params: np.ndarray,
    make_policy: Callable[[], Policy],
    make_features: Callable[[Dict[str, Any], Dict[str, Any], int], np.ndarray],
    horizon: int = 1800,
    seed: int = 0,
) -> Tuple[float, list[Dict[str, np.ndarray]]]:
    """Run one episode with the given policy parameters and return total reward."""

    # Create policy and set its parameters
    policy = make_policy()
    policy.set_params(params)
    policy.reset()

    # Create environment
    env = EndlessPlatformerEnv(
        # faster energy drain for training
        natural_drain_per_sec=0.1,
        move_drain_per_sec=0.15,
        jump_drain=0.2,
        seed=seed
    )
    obs, info = env.reset(seed=seed)
    prev_action = -1
    total_reward = 0.0

    # Initialize JEPA optimizer if policy has one
    jepa_optimizer = None
    if isinstance(policy, ParamNonLinearPolicy_JEPA):
        policy = cast(ParamNonLinearPolicy_JEPA, policy)
        # Simple SGD or Adam for online learning
        jepa_optimizer = torch.optim.Adam(policy.jepa.parameters(), lr=1e-3)

    policy_info = []

    for _ in range(horizon):
        features = make_features(obs, info, prev_action)
        action, act_info = policy.act(features, info)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        total_reward += float(reward)

        # JEPA Online Training Step
        if jepa_optimizer is not None:
            policy = cast(ParamNonLinearPolicy_JEPA, policy)
            
            # Prepare tensors
            device = policy.device
            obs_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
            
            # We need next features for the target
            next_features = make_features(next_obs, next_info, action)
            next_obs_t = torch.from_numpy(next_features).float().unsqueeze(0).to(device)
            
            action_vec = torch.zeros(1, policy.n_actions).to(device)
            action_vec[0, action] = 1.0
            
            # get energy from next_info to drive curiosity
            energy = next_info.get("energy", 0.0)
            metadata_t = torch.tensor([[energy]]).float().to(device)

            loss_dict = policy.jepa.compute_losses(obs_t, action_vec, next_obs_t, metadata_t)
            
            jepa_optimizer.zero_grad()
            loss_dict["loss"].backward()
            jepa_optimizer.step()
            
            policy.jepa.update_target_encoder()
            
            # Add JEPA stats to info for logging
            act_info.update(loss_dict)

        policy_info.append(act_info)

        prev_action = action
        obs = next_obs
        info = next_info

        if terminated or truncated:
            break

    env.close()

    return total_reward, policy_info
