
import numpy as np
import tempfile
from pathlib import Path
from rlo.policies.param_nonlinear import ParamNonLinearPolicy_Stochastic
from rlo.utils.serialization import PolicyBundle

def test_policy_override():
    # 1. Create a policy with default temperature
    n_actions = 3
    n_features = 5
    policy = ParamNonLinearPolicy_Stochastic(n_actions=n_actions, n_features=n_features, temperature=1.0)
    
    # Initialize random weights
    flat_params = np.random.randn(policy.num_params())
    policy.set_params(flat_params)
    
    # 2. Save it to a bundle
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_policy.npz"
        bundle = PolicyBundle(policy=policy)
        bundle.save(save_path)
        
        # 3. Load it back with NO override
        loaded_bundle_default = PolicyBundle.load(save_path)
        loaded_policy_default = loaded_bundle_default.policy
        print(f"Default loaded temperature: {loaded_policy_default.temperature}")
        assert loaded_policy_default.temperature == 1.0
        
        # 4. Load it back WITH override
        loaded_bundle_override = PolicyBundle.load(save_path, policy_overrides={"temperature": 0.5})
        loaded_policy_override = loaded_bundle_override.policy
        print(f"Overridden loaded temperature: {loaded_policy_override.temperature}")
        assert loaded_policy_override.temperature == 0.5
        
        print("Verification SUCCESS!")

if __name__ == "__main__":
    test_policy_override()
