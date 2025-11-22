import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np

from rlo.features import FEATURE_NAMES
from rlo.policies import Policy, get_policy_class


@dataclass
class PolicyBundle:
    """A convenience wrapper bundling a policy with metadata."""

    policy: Policy
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, object]:
        """Convert bundle into JSON/NumPy friendly payload."""
        return {
            "n_actions": (
                self.policy.n_actions
            ),
            "n_features": (
                self.policy.n_features
            ),
            "params": self.policy.get_params(),
            # fully-qualified class name so we can rebuild the same class
            "policy_class": f"{self.policy.__class__.__module__}.{self.policy.__class__.__name__}",
            # extra per-policy kwargs (JSON-serializable)
            "policy_kwargs": json.dumps(self.policy.to_payload()),
            "feature_names": np.array(FEATURE_NAMES, dtype=object),
            "metadata": json.dumps(self.metadata),
        }

    def save(self, path: str | Path) -> Path:
        """Persist the bundle to ``.npz`` format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_payload()
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PolicyBundle":
        """Load a previously saved policy bundle."""
        data = np.load(Path(path), allow_pickle=True)
        n_actions = data["n_actions"].item()
        n_features = data["n_features"].item()
        params = data["params"]
        metadata = json.loads(data["metadata"].item())

        # determine policy class
        policy_class_name = None
        if "policy_class" in data.files:
            policy_class_name = data["policy_class"].item()

        policy_kwargs = {}
        if "policy_kwargs" in data.files:
            policy_kwargs = json.loads(data["policy_kwargs"].item())

        if policy_class_name is None:
            raise ValueError("Saved policy bundle missing 'policy_class' field")

        policy_cls = get_policy_class(policy_class_name)
        if policy_cls is None:
            raise ValueError(f"Unknown policy class '{policy_class_name}'. Make sure it's imported so it can register.")

        policy = policy_cls.from_payload(
            flat_params=params, n_actions=n_actions, n_features=n_features, policy_kwargs=policy_kwargs
        )

        return cls(policy=policy, metadata=metadata)
