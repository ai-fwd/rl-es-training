from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .param_base import Policy


@dataclass
class ParamLinearPolicy(Policy):
    n_actions: int
    n_features: int

    def __post_init__(self):
        self._W = np.zeros((self.n_actions, self.n_features), dtype=np.float32)
        self._b = np.zeros(self.n_actions, dtype=np.float32)

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

    def act(self, features: np.ndarray, info: Dict[str, Any]) -> int:
        """Compute logits = Wx + b, pick argmax (deterministic & ES-friendly)."""
        logits = self._W @ features + self._b
        return int(np.argmax(logits))
