from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

FEATURE_NAMES: Tuple[str, ...] = (
    "bias_term",
    "energy_level",
    "elapsed_time_min",
    "x_position",
    "y_position",
    "on_ground",
    "fell",
    "can_eat",
    "near_food",
    "prev_action",
)


def make_basic_features(
    obs: Dict[str, Any], info: Dict[str, Any], prev_action: int
) -> np.ndarray:
    """Extracts a basic feature vector from the observation dictionary.

    Args:
        obs: A dictionary containing environment observations.
        info: A dictionary containing additional information from the environment.
        prev_action: The previous action taken by the agent.
    Returns:
        11-dimensional feature vector as a numpy array.
    """

    # Extract relevant values from obs and info
    energy, _, elapsed_s = obs["metrics"]
    t_min = elapsed_s / 60.0  # Scale time to minutes

    # bucket energy level to bad, okay, good
    if energy < 0.3:
        energy_cat = -1.0
    elif energy < 0.7:
        energy_cat = 0.0
    else:
        energy_cat = 1.0

    # One-hot encode previous action (4 actions) for temporal context
    prev_action_one_hot = np.zeros(4, dtype=np.float32)
    if (prev_action > -1):  # no prev action
        prev_action_one_hot[prev_action] = 1.0

    # one-hot encode previous energy level (3 levels)
    prev_energy_one_hot = np.zeros(3, dtype=np.float32)
    if energy_cat == -1:
        prev_energy_one_hot[0] = 1.0
    elif energy_cat == 0:
        prev_energy_one_hot[1] = 1.0
    else:
        prev_energy_one_hot[2] = 1.0

    return np.array(
        [
            #1.0,  # Bias term
            energy,
            #t_min,
            info["x_position"],
            #info["y_position"],
            #1.0 if info["on_ground"] else 0.0,
            # these features are not guaranteed to be present b/c they're calculated during step
            #1.0 if info.get("fell", False) else 0.0,
            1.0 if info.get("can_eat", False) else 0.0,
            1.0 if info.get("near_food", False) else 0.0,
            #*prev_action_one_hot,
            #*prev_energy_one_hot,
        ],
        dtype=np.float32,
    )
