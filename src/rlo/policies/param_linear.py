from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import Policy


@dataclass
class ParamLinearPolicy(Policy):
    def __post_init__(self):
        # Initialize weights and biases
        self._W = np.zeros((self.n_actions, self.n_features), dtype=np.float32)
        self._b = np.zeros(self.n_actions, dtype=np.float32)

        # Validate shapes
        if self._W.ndim != 2:
            raise ValueError("weights must be a 2D array [actions, features].")
        if self._b.shape != (self._W.shape[0],):
            raise ValueError("bias must have shape (n_actions,).")

    def num_params(self) -> int:
        """Returns the number of parameters in the linear Wx + b policy."""
        return self.n_actions * self.n_features + self.n_actions

    def get_params(self) -> np.ndarray:
        """Returns the current parameters of the policy as a 1D numpy array."""
        return np.concatenate([self._W.ravel(), self._b])

    def set_params(self, params: np.ndarray) -> None:
        """Sets the parameters of the linear Wx + b policy from a 1D numpy array."""
        assert params.shape == (self.num_params(),)
        W_size = self.n_actions * self.n_features
        self._W = params[:W_size].reshape((self.n_actions, self.n_features))
        self._b = params[W_size:]

    def act(
        self, features: np.ndarray, info: Dict[str, Any]
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """Compute logits = Wx + b, pick argmax (deterministic & ES-friendly)."""

        # Compute feature contributions for explainability
        feature_contributions = self._W * features.reshape(1, -1)

        # Compute logits and select action
        logits = self._W @ features + self._b
        selected_action = int(np.argmax(logits))

        info = {
            "local_index": selected_action,
            "logits": logits,
            "probabilities": F.softmax(torch.tensor(logits), dim=-1).numpy(),
            "contributions": feature_contributions,
        }

        return selected_action, info

    @classmethod
    def from_flat(
        cls,
        flat_params: np.ndarray,
        n_actions: int,
        n_features: int,
    ) -> ParamLinearPolicy:
        """Creates a ParamLinearPolicy instance from 1-D flat parameters."""

        policy = cls(
            n_actions=n_actions,
            n_features=n_features,
        )
        policy.set_params(flat_params)
        return policy
