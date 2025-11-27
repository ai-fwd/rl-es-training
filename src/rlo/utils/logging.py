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
        def _serialize_float_list(values: Any) -> list[float]:
            """Convert tensors/arrays/sequences to a list of floats for JSON."""
            if hasattr(values, "detach"):
                values = values.detach()
            if hasattr(values, "cpu"):
                values = values.cpu()
            if hasattr(values, "numpy"):
                values = values.numpy()
            if hasattr(values, "tolist"):
                values = values.tolist()
            if isinstance(values, (list, tuple)):
                return [float(x) for x in values]
            return [float(values)]

        def _serialize_action(idx: int, x: Dict[str, Any]) -> Dict[str, Any]:
            action = int(x["selected_action"])
            label = EndlessPlatformerEnv.ACTION_LABELS[action]
            serialized = {
                "step": idx,
                "action_index": action,
                "action_label": label,
                "logits": _serialize_float_list(x["logits"]),
            }
            if "curiosity_scores" in x:
                serialized["curiosity_scores"] = _serialize_float_list(x["curiosity_scores"])
            if "validity" in x:
                try:
                    serialized["validity"] = float(x["validity"])
                except Exception:
                    serialized["validity"] = _serialize_float_list(x["validity"])[0]
            return serialized

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
