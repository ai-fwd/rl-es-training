"""A base class for parameter-space optimization policies.
Provides an interface for policies that are represented by a single vector of parameters.
The optimizer (i.e. CMA-ES) will choose new parameter vectors to evaluate.
Feature extractions occurs outside of the policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class Policy(ABC):
    """Abstract policy:
    - Parameters are represented as a single vector that represent the agent. This is for consistency across parameter-space optimizers. (shape [d])
    - act() method takes in observations and produces actions.
    """

    @abstractmethod
    def num_params(self) -> int:
        """Returns the number of parameters in the policy."""
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Returns the current parameters of the policy as a 1D numpy array."""
        raise NotImplementedError

    @abstractmethod
    def set_params(self, params: np.ndarray) -> None:
        """Sets the parameters of the policy from a 1D numpy array."""
        raise NotImplementedError

    @abstractmethod
    def act(
        self, features: np.ndarray, info: Dict[str, Any]
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """Given the current features and info, returns the action to take + some diagnostics.

        Args:
            features: A 1D numpy array of features representing the current observation.
            info: A dictionary containing any additional information needed to select the action.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets any internal state of the policy between episodes, if necessary."""
        pass
