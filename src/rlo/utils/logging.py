from dataclasses import dataclass
from typing import Dict


@dataclass
class GenerationStats:
    """Book-keeping structure stored in ``history.json``."""

    generation: int
    wall_time_s: float
    best_reward: float
    mean_reward: float
    median_reward: float
    reward_std: float
    sigma: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "generation": int(self.generation),
            "wall_time_s": float(self.wall_time_s),
            "best_reward": float(self.best_reward),
            "mean_reward": float(self.mean_reward),
            "median_reward": float(self.median_reward),
            "reward_std": float(self.reward_std),
            "sigma": float(self.sigma),
        }
