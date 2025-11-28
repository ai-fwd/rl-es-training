import torch
import numpy as np
import pytest
from typing import Dict, Any
from unittest.mock import MagicMock

from rlo.evaluators.evaluate import evaluate_candidate
from rlo.policies.param_nonlinear_jepa import ParamNonLinearPolicy_JEPA

def make_features(obs, info, prev_action):
    # Dummy feature extractor
    return np.random.randn(8).astype(np.float32)

def test_evaluate_candidate_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")
    
    n_features = 8
    n_actions = 4
    
    def make_policy():
        return ParamNonLinearPolicy_JEPA(n_actions=n_actions, n_features=n_features)
    
    # Create dummy params
    policy = make_policy()
    params = policy.get_params()
    
    # Run evaluation
    # This should trigger the JEPA optimizer step which was failing
    reward, info = evaluate_candidate(
        params=params,
        make_policy=make_policy,
        make_features=make_features,
        horizon=10, # Short horizon for testing
        seed=42
    )
    
    assert isinstance(reward, float)
    assert len(info) > 0
    # Check if loss was computed (implies optimizer step ran)
    assert "loss" in info[0]

if __name__ == "__main__":
    test_evaluate_candidate_gpu()
    print("Evaluate GPU test passed!")
