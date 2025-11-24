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
    population_traces: list[Dict[str, Any]] | None = None

    def to_dict(self) -> Dict[str, Any]:
        def _serialize_logits(logits: Any) -> list[float]:
            # tensors/arrays → list for JSON
            if hasattr(logits, "detach"):
                logits = logits.detach()
            if hasattr(logits, "cpu"):
                logits = logits.cpu()
            if hasattr(logits, "numpy"):
                logits = logits.numpy()
            return [float(x) for x in logits]

        def _serialize_action(idx: int, x: Dict[str, Any]) -> Dict[str, Any]:
            action = int(x["selected_action"])
            label = EndlessPlatformerEnv.ACTION_LABELS[action]
            return {
                "step": idx,
                "action_index": action,
                "action_label": label,
                "logits": _serialize_logits(x["logits"]),
            }

        serialized_best_trace = [_serialize_action(i, x) for i, x in enumerate(self.policy_info)]

        return {
            "generation": int(self.generation),
            "wall_time_s": float(self.wall_time_s),
            "best_reward": float(self.best_reward),
            # "mean_reward": float(self.mean_reward),
            # "median_reward": float(self.median_reward),
            # "reward_std": float(self.reward_std),
            "sigma": float(self.sigma),
            "best_policy": {
                "trace": serialized_best_trace,
            },
            "population": self.population_traces or [],
        }
