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

    def to_payload(self) -> dict:
        """Return JSON-serializable extra kwargs for this policy.

        By default no extra kwargs are exposed. Subclasses may override.
        """
        return {}

    @classmethod
    def from_payload(
        cls, flat_params: np.ndarray, n_actions: int, n_features: int, policy_kwargs: dict | None = None
    ) -> "Policy":
        """Construct a policy from saved payload. Tries `from_flat` if present.

        `policy_kwargs` is a dict of extra constructor args (may be empty).
        """
        # prefer an explicit from_flat if the class provides it (older code)
        if hasattr(cls, "from_flat"):
            return cls.from_flat(flat_params, n_actions, n_features)  # type: ignore[attr-defined]

        policy_kwargs = policy_kwargs or {}
        policy = cls(n_actions=n_actions, n_features=n_features, **policy_kwargs)
        policy.set_params(flat_params)
        return policy
