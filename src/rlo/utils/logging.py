from dataclasses import dataclass
from typing import Dict, Any

from rlo.envs.endless_platformer import EndlessPlatformerEnv


@dataclass
class GenerationStats:
    """Book-keeping structure stored in ``history.json``."""

    generation: int
    wall_time_s: float
    best_reward: float
    # mean_reward: float
    # median_reward: float
    # reward_std: float
    sigma: float
    policy_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        def _format_action(x):
            action = int(x['selected_action'])
            logits = x['logits']
            label = EndlessPlatformerEnv.ACTION_LABELS[action]
            return f"{action}: {label} (logits={logits.numpy()})"

        return {
            "generation": int(self.generation),
            "wall_time_s": float(self.wall_time_s),
            "best_reward": float(self.best_reward),
            # "mean_reward": float(self.mean_reward),
            # "median_reward": float(self.median_reward),
            # "reward_std": float(self.reward_std),
            "sigma": float(self.sigma),
            "selected_actions": [_format_action(x) for x in self.policy_info],
        }