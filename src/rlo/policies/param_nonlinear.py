from __future__ import annotations

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
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@dataclass
class ParamNonLinearPolicy(Policy):
    n_actions: int
    n_features: int

    def __post_init__(self):
        # Instead of defining the weights and biases directly, we define an MLP to represent the policy
        self._model = MLP(input_dim=self.n_features, hidden_dim=self.n_features*2, output_dim=self.n_actions)

    def num_params(self) -> int:
        """Returns the number of parameters in the nonlinear policy."""
        # will give (input*hidden + hidden) + (hidden*out + out) = fc1+fc2 params
        return sum(p.numel() for p in self._model.parameters()) 

    def get_params(self) -> np.ndarray:
        """Returns the current parameters of the policy as a 1D numpy array."""
        return torch.cat([param.view(-1) for param in self._model.parameters()]).detach().numpy()

    def set_params(self, params: np.ndarray) -> None:
        """Sets the parameters of the nonlinear policy from a 1D numpy array."""
        idx = 0
        for p in self._model.parameters():
            s = p.numel()
            p_block = torch.from_numpy(params[idx:idx+s]).float()
            p.data.copy_(p_block.view(p.size()))
            idx += s

    def act(
        self, features: np.ndarray, info: Dict[str, Any]
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """Compute logits = Wx + b, pick argmax (deterministic & ES-friendly)."""

        # Compute feature contributions for explainability
        layer1_feature_contributions = self._model.fc1.weight * features.reshape(1, -1)

        # Compute logits and select action from MLP
        x = torch.tensor(features).float()
        logits = self._model(x)
        selected_action = int(np.argmax(logits))

        info = {
            "local_index": selected_action,
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1).numpy(),
            "contributions": layer1_feature_contributions.numpy(),
        }

        return selected_action, info

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
