import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np

from rlo.features import FEATURE_NAMES
from rlo.policies import ParamLinearPolicy, Policy


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
                if isinstance(self.policy, ParamLinearPolicy)
                else None
            ),
            "n_features": (
                self.policy.n_features
                if isinstance(self.policy, ParamLinearPolicy)
                else None
            ),
            "params": self.policy.get_params(),
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
        policy = ParamLinearPolicy(
            n_actions=n_actions,
            n_features=n_features,
        )
        policy.set_params(params)
        return cls(policy=policy, metadata=metadata)
