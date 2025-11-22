from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Policy
from .param_nonlinear_base import ParamNonLinearPolicy
from rlo.policies import register_policy


@dataclass
@register_policy
class ParamNonLinearPolicy_ArgMax(ParamNonLinearPolicy):
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
        probabilities = F.softmax(logits, dim=-1).detach().numpy()
        
        info = {
            "selected_action": selected_action,
            "logits": logits,
            "probabilities": probabilities,
            "contributions": layer1_feature_contributions.numpy(),
        }

        return selected_action, info
    
@dataclass
@register_policy
class ParamNonLinearPolicy_Stochastic(ParamNonLinearPolicy):
    """Stochastic MLP policy with configurable temperature.

    The `temperature` field is stored in the serialized payload so validation
    and training runs can use different temperatures without code changes.
    """
    temperature: float = 1.0

    def __post_init__(self):
        # initialize base model
        super().__post_init__()

    def to_payload(self) -> dict:
        return {"temperature": float(self.temperature)}

    @classmethod
    def from_payload(
        cls, flat_params: np.ndarray, n_actions: int, n_features: int, policy_kwargs: dict | None = None
    ) -> ParamNonLinearPolicy_Stochastic:
        policy_kwargs = policy_kwargs or {}
        return super().from_payload(flat_params, n_actions, n_features, policy_kwargs)

    def act(
        self, features: np.ndarray, info: Dict[str, Any]
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        # Compute feature contributions for explainability
        layer1_feature_contributions = self._model.fc1.weight * features.reshape(1, -1)

        # Compute logits and select action from MLP
        x = torch.tensor(features).float()
        logits = self._model(x)

        # temperature may be set on the instance (default 1.0)
        T = getattr(self, "temperature", 1.0)
        probabilities = F.softmax(logits / float(T), dim=-1)

        # sample an action from the probability distribution
        dist = torch.distributions.Categorical(probs=probabilities)
        selected_action = int(dist.sample())

        info = {
            "selected_action": selected_action,
            "logits": logits,
            "probabilities": probabilities.detach().numpy(),
            "contributions": layer1_feature_contributions.numpy(),
        }

        return selected_action, info