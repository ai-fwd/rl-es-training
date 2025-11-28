import torch
import numpy as np
import pytest
from rlo.policies.param_nonlinear_jepa import ParamNonLinearPolicy_JEPA

def test_gpu_availability_and_usage():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")
    
    n_features = 8
    n_actions = 4
    
    policy = ParamNonLinearPolicy_JEPA(n_actions=n_actions, n_features=n_features, device="cuda")
    
    # Check if models are on GPU
    assert next(policy.jepa.parameters()).is_cuda
    assert next(policy._model.parameters()).is_cuda
    
    # Test act
    features = np.random.randn(n_features).astype(np.float32)
    info = {}
    
    action, act_info = policy.act(features, info)
    
    assert isinstance(action, int)
    assert 0 <= action < n_actions
    
    # Check if returned info is on CPU (numpy)
    assert isinstance(act_info["curiosity_scores"], np.ndarray)
    assert isinstance(act_info["logits"], torch.Tensor)
    assert act_info["logits"].device.type == "cuda" # Logits might still be on CUDA in info dict, but that's okay as long as we handled it for argmax
    
    # Test get_params (should return numpy array)
    params = policy.get_params()
    assert isinstance(params, np.ndarray)
    
    # Test set_params
    new_params = params + 0.01
    policy.set_params(new_params)
    
    # Verify params changed on GPU
    current_params = policy.get_params()
    assert np.allclose(current_params, new_params)
    assert next(policy._model.parameters()).is_cuda

if __name__ == "__main__":
    test_gpu_availability_and_usage()
    print("GPU verification test passed!")
