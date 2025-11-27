import torch
import numpy as np
import pytest
from rlo.policies.jepa import JEPAModule
from rlo.policies.param_nonlinear_jepa import ParamNonLinearPolicy_JEPA

def test_jepa_module_shapes():
    input_dim = 10
    hidden_dim = 20
    latent_dim = 5
    action_dim = 3
    
    jepa = JEPAModule(input_dim, hidden_dim, latent_dim, action_dim)
    
    batch_size = 4
    obs = torch.randn(batch_size, input_dim)
    action = torch.randn(batch_size, action_dim)
    next_obs = torch.randn(batch_size, input_dim)
    metadata = torch.randn(batch_size, 1)
    
    # Test encode
    z = jepa.encode(obs)
    assert z.shape == (batch_size, latent_dim)
    
    # Test imagine
    z_hat = jepa.imagine(z, action)
    assert z_hat.shape == (batch_size, latent_dim)
    
    # Test losses
    losses = jepa.compute_losses(obs, action, next_obs, metadata)
    assert "loss" in losses
    assert "loss_trans" in losses
    assert "loss_vicreg" in losses
    assert "loss_probe" in losses
    
    # Test validity
    validity = jepa.get_validity(z, z_hat)
    assert validity.shape == (batch_size,)
    assert torch.all(validity >= 0) and torch.all(validity <= 1)

def test_policy_integration():
    n_features = 8
    n_actions = 4
    
    policy = ParamNonLinearPolicy_JEPA(n_actions=n_actions, n_features=n_features)
    
    # Test act
    features = np.random.randn(n_features).astype(np.float32)
    info = {}
    
    action, act_info = policy.act(features, info)
    
    assert isinstance(action, int)
    assert 0 <= action < n_actions
    assert "curiosity_scores" in act_info
    assert len(act_info["curiosity_scores"]) == n_actions
    
    # Test serialization
    payload = policy.to_payload()
    assert "jepa_state" in payload
    
    # Test deserialization
    params = policy.get_params()
    new_policy = ParamNonLinearPolicy_JEPA.from_payload(
        params, n_actions, n_features, payload
    )
    
    # Check if weights match
    for p1, p2 in zip(policy.jepa.parameters(), new_policy.jepa.parameters()):
        assert torch.allclose(p1, p2)

if __name__ == "__main__":
    test_jepa_module_shapes()
    test_policy_integration()
    print("All tests passed!")
