from __future__ import annotations

from typing import Any, Dict

import numpy as np


def make_basic_features(
    obs: Dict[str, Any], info: Dict[str, Any], prev_action: int, prev_on_ground: bool
) -> np.ndarray:
    """Extracts a basic feature vector from the observation dictionary.

    Args:
        obs: A dictionary containing environment observations.
        info: A dictionary containing additional information from the environment.
        prev_action: The previous action taken by the agent.
        prev_on_ground: Whether the agent was on the ground in the previous step.
    Returns:
        8-dimensional feature vector as a numpy array.
    """

    energy, _, elapsed_s = obs["metrics"]
    t_min = elapsed_s / 60.0  # Scale time to minutes
    can_eat = 1.0 if info.get("can_eat", False) else 0.0

    # Provide policy with some state about previous action and on-ground status

    # One-hot encode previous action (3 actions)
    prev_action_one_hot = np.zeros(3, dtype=np.float32)
    if prev_action in (1, 2, 3):
        prev_action_one_hot[prev_action - 1] = 1.0

    on_ground = 1.0 if prev_on_ground else 0.0
    return np.array(
        [
            1.0,  # Bias term
            energy,
            t_min,
            can_eat,
            *prev_action_one_hot,
            on_ground,
        ],
        dtype=np.float32,
    )
