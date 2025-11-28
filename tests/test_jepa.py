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
    
    policy = ParamNonLinearPolicy_JEPA(n_actions=n_actions, n_features=n_features, temperature=1.0)
    
    # Inject a JEPA (simulating Global JEPA)
    jepa = JEPAModule(n_features, 64, 32, n_actions)
    policy.jepa = jepa
    
    # Test act
    features = np.random.randn(n_features).astype(np.float32)
    info = {}
    
    # Run multiple times to check for stochasticity (not guaranteed but likely different)
    actions = []
    for _ in range(10):
        action, act_info = policy.act(features, info)
        actions.append(action)
        assert isinstance(action, int)
        assert 0 <= action < n_actions
        assert "curiosity_scores" in act_info
        assert "probabilities" in act_info
    
    # Check if we got at least some variation (might fail if logits are extreme, but unlikely with random init)
    # With random init, logits should be close to 0, so uniform probs.
    assert len(set(actions)) > 1, "Policy should be stochastic with temp=1.0"
    
    # Test serialization
    payload = policy.to_payload()
    assert "jepa_state" in payload
    assert "temperature" in payload
    
    # Test deserialization
    params = policy.get_params()
    new_policy = ParamNonLinearPolicy_JEPA.from_payload(
        params, n_actions, n_features, payload
    )
    
    # Check if weights match
    for p1, p2 in zip(policy.jepa.parameters(), new_policy.jepa.parameters()):
        assert torch.allclose(p1, p2)
        
def test_global_jepa_training_loop():
    # Mock training loop components
    n_features = 8
    n_actions = 4
    jepa = JEPAModule(n_features, 64, 32, n_actions)
    optimizer = torch.optim.Adam(jepa.parameters(), lr=1e-3)
    
    # Create dummy transitions
    batch_size = 10
    obs = torch.randn(batch_size, n_features)
    next_obs = torch.randn(batch_size, n_features)
    actions = torch.randint(0, n_actions, (batch_size,))
    action_vec = torch.zeros(batch_size, n_actions)
    action_vec[range(batch_size), actions] = 1.0
    metadata = torch.randn(batch_size, 1)
    
    # Train step
    loss_dict = jepa.compute_losses(obs, action_vec, next_obs, metadata)
    optimizer.zero_grad()
    loss_dict["loss"].backward()
    optimizer.step()
    jepa.update_target_encoder()
    
    assert loss_dict["loss"].item() > 0

if __name__ == "__main__":
    test_jepa_module_shapes()
    test_policy_integration()
    test_global_jepa_training_loop()
    print("All tests passed!")
