from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

# Import your environment (adjust import path if needed)
from rlo.envs.endless_platformer import EndlessPlatformerEnv
from rlo.policies.param_base import Policy


def evaluate_candidate(
    params: np.ndarray,
    make_policy: Callable[[], Policy],
    make_features: Callable[[Dict[str, Any], Dict[str, Any], int, bool], np.ndarray],
    horizon: int = 1800,
    seed: int = 0,
) -> float:
    """Run one episode with the given policy parameters and return total reward."""

    # Create policy and set its parameters
    policy = make_policy()
    policy.set_params(params)
    policy.reset()

    # Create environment with higher drain rates for faster evaluation
    env = EndlessPlatformerEnv(
        natural_drain_per_sec=0.1, move_drain_per_sec=0.1, jump_drain=0.25, seed=seed
    )
    obs, info = env.reset(seed=seed)
    prev_action, prev_on_ground = 0, bool(info.get("on_ground", True))
    total_reward = 0.0

    for _ in range(horizon):
        features = make_features(obs, info, prev_action, prev_on_ground)
        action = policy.act(features, info)

        # avoid repeated JUMPs when in air
        if not info.get("on_ground", True) and action == 2:
            action = 0  # NOOP

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        prev_action = action
        prev_on_ground = bool(info.get("on_ground", True))

        if terminated or truncated:
            break

    env.close()
    return total_reward
