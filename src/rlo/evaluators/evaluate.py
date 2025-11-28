from __future__ import annotations
import torch

from typing import Any, Callable, Dict, Tuple, cast

import numpy as np

# Import your environment (adjust import path if needed)
from rlo.envs import EndlessPlatformerEnv
from rlo.policies import Policy
from rlo.policies.jepa import JEPAModule


def evaluate_candidate(
    params: np.ndarray,
    make_policy: Callable[[], Policy],
    make_features: Callable[[Dict[str, Any], Dict[str, Any], int], np.ndarray],
    horizon: int = 1800,
    seed: int = 0,
    jepa: JEPAModule | None = None,
) -> Tuple[float, list[Dict[str, np.ndarray]], list[Dict[str, object]]]:
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
    
    # Inject Global JEPA if provided
    if hasattr(policy, "jepa") and jepa is not None:
        policy.jepa = jepa

    policy_info = []
    transitions = []

    for _ in range(horizon):
        features = make_features(obs, info, prev_action)
        action, act_info = policy.act(features, info)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        total_reward += float(reward)
        
        # Collect transition for Global JEPA training
        # We need next_features for the target encoder
        next_features = make_features(next_obs, next_info, action)
        
        # Metadata (Energy)
        energy = next_info.get("energy", 0.0)
        
        transitions.append({
            "obs": features,
            "action": action,
            "next_obs": next_features,
            "metadata": np.array([energy], dtype=np.float32)
        })

        policy_info.append(act_info)

        prev_action = action
        obs = next_obs
        info = next_info

        if terminated or truncated:
            break

    env.close()

    return total_reward, policy_info, transitions
