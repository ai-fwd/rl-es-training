from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Policy

# A simple MLP with one hidden layer that will be used in the ParamNonLinearPolicy
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # turn off autograd for all params
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


@dataclass
class ParamNonLinearPolicy(Policy):
    def __post_init__(self):
        # Instead of defining the weights and biases directly, we define an MLP to represent the policy
        self._model = MLP(input_dim=self.n_features, hidden_dim=self.n_features, output_dim=self.n_actions)

    def num_params(self) -> int:
        """Returns the number of parameters in the nonlinear policy."""
        # will give (input*hidden + hidden) + (hidden*out + out) = fc1+fc2 params
        return sum(p.numel() for p in self._model.parameters()) 

    def get_params(self) -> np.ndarray:
        """Returns the current parameters of the policy as a 1D numpy array."""
        return torch.cat([param.view(-1) for param in self._model.parameters()]).detach().cpu().numpy()

    def set_params(self, params: np.ndarray) -> None:
        """Sets the parameters of the nonlinear policy from a 1D numpy array."""
        idx = 0
        for p in self._model.parameters():
            s = p.numel()
            p_block = torch.from_numpy(params[idx:idx+s]).float()
            p.data.copy_(p_block.view(p.size()))
            idx += s

    def to_payload(self) -> dict:
        """Return a dict of extra policy kwargs suitable for JSON-serialization.

        The base nonlinear policy has no extra kwargs; subclasses may override
        to include things such as temperature, activation choices, etc.
        """
        return {}

    @classmethod
    def from_payload(
        cls,
        flat_params: np.ndarray,
        n_actions: int,
        n_features: int,
        policy_kwargs: dict | None = None,
    ) -> "ParamNonLinearPolicy":
        """Instantiate a policy from serialized payload.

        Subclasses that accept extra constructor args should accept them
        as keyword args so they are applied here via `policy_kwargs`.
        """
        policy_kwargs = policy_kwargs or {}
        policy = cls(n_actions=n_actions, n_features=n_features, **policy_kwargs)
        policy.set_params(flat_params)
        return policy

    @abstractmethod
    def act(
        self, features: np.ndarray, info: Dict[str, Any]
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """Determine action"""
    
        raise NotImplementedError

    @classmethod
    def from_flat(
        cls,
        flat_params: np.ndarray,
        n_actions: int,
        n_features: int,
    ) -> ParamNonLinearPolicy:
        """Creates a ParamNonLinearPolicy instance from 1-D flat parameters."""

        policy = cls(
            n_actions=n_actions,
            n_features=n_features,
        )
        policy.set_params(flat_params)
        return policy
