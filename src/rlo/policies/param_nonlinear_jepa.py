from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .param_nonlinear_base import ParamNonLinearPolicy, MLP
from .jepa import JEPAModule
from rlo.policies import register_policy

@dataclass
@register_policy
class ParamNonLinearPolicy_JEPA(ParamNonLinearPolicy):
    """Policy that uses JEPA-based curiosity scores as input features."""
    
    # JEPA config
    jepa_hidden_dim: int = 64
    jepa_latent_dim: int = 32
    device: str = "cpu"
    
    def __post_init__(self):
        # Initialize JEPA
        self.jepa = JEPAModule(
            input_dim=self.n_features,
            hidden_dim=self.jepa_hidden_dim,
            latent_dim=self.jepa_latent_dim,
            action_dim=self.n_actions,
        )
        
        # Initialize Policy MLP
        # Input to MLP is features + curiosity scores per action == n_features + n_actions
        self._model = MLP(
            input_dim=self.n_features + self.n_actions,
            hidden_dim=self.n_features, # Keep hidden dim same as features for now
            output_dim=self.n_actions
        )

        self.device = torch.device(self.device)
        self.jepa.to(self.device)
        self._model.to(self.device)

    def to_payload(self) -> dict:
        """Serialize JEPA weights along with policy config."""
        # Save JEPA state dict to a bytes buffer, then base64 encode
        buffer = io.BytesIO()
        torch.save(self.jepa.state_dict(), buffer)
        jepa_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "jepa_hidden_dim": self.jepa_hidden_dim,
            "jepa_latent_dim": self.jepa_latent_dim,
            "jepa_state": jepa_str,
            "device": str(self.device)
        }

    @classmethod
    def from_payload(
        cls,
        flat_params: np.ndarray,
        n_actions: int,
        n_features: int,
        policy_kwargs: dict | None = None,
    ) -> "ParamNonLinearPolicy_JEPA":
        policy_kwargs = policy_kwargs or {}
        jepa_state = policy_kwargs.pop("jepa_state", None)
        
        policy = cls(n_actions=n_actions, n_features=n_features, **policy_kwargs)
        policy.set_params(flat_params)
        
        if jepa_state:
            buffer = io.BytesIO(base64.b64decode(jepa_state))
            policy.jepa.load_state_dict(torch.load(buffer))
            
        return policy

    def act(
        self, features: np.ndarray, info: Dict[str, Any]
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        # 1. Prepare inputs
        obs_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device) # (1, F)
        
        # 2. Compute Curiosity Scores
        with torch.no_grad():
            # Current State Embeddings
            z_t = self.jepa.encode(obs_tensor) # (1, L)
            
            # Baseline for Probe: P(Target(s))
            # "project z_t so the probe consistently uses target embeddings"
            # We use the target encoder on the current state to get the baseline z'
            z_prime_t = self.jepa.encode_target(obs_tensor) # (1, L)
            v_now = self.jepa.probe(z_prime_t) # (1, 1)
            
            scores = []
            for a in range(self.n_actions):
                # One-hot action
                action_vec = torch.zeros(1, self.n_actions).to(self.device)
                action_vec[0, a] = 1.0
                
                # Imagine future
                z_hat_next = self.jepa.imagine(z_t, action_vec) # (1, L)
                
                # Get a validity percentage for this imagined future
                validity = self.jepa.get_validity(z_t, z_hat_next) # (1,)
                
                # Survival Impact: P(z_hat_next) - P(z_prime_t)
                # Both inputs to probe are in Target Space (Predicted vs Actual)
                v_next = self.jepa.probe(z_hat_next) # (1, 1)
                delta_f = v_next - v_now
                
                # Score
                score = validity * torch.tanh(delta_f)
                scores.append(score.item())
                
            scores_arr = np.array(scores, dtype=np.float32)
            
        # 3. Policy Forward
        # Concatenate features and scores
        input_features = np.concatenate([features, scores_arr])
        x = torch.from_numpy(input_features).float().to(self.device)
        
        logits = self._model(x)
        
        # Deterministic argmax for now
        selected_action = int(np.argmax(logits.detach().cpu().numpy()))
        
        info = {
            "selected_action": selected_action,
            "logits": logits,
            "curiosity_scores": scores_arr,
            "validity": validity.item(),
        }
        
        return selected_action, info
