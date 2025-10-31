from __future__ import annotations

from pathlib import Path

import numpy as np

from rlo.features.param_basic import make_basic_features
from rlo.policies.param_linear import ParamLinearPolicy
from rlo.trainers.cma_es_trainer import train_cma_es


def make_policy() -> ParamLinearPolicy:
    return ParamLinearPolicy(n_actions=4, n_features=8)


if __name__ == "__main__":
    result = train_cma_es(
        make_policy=make_policy,
        make_features=make_basic_features,
        generations=60,
        init_sigma=0.5,
        pop_multiplier=1.0,
        base_seed=0,
        horizon=1800,  # 30s seconds at 60 FPS
    )

    best_params = result["best_params"]
    print("\n=== Training complete ===")
    print("Best return:", result["best_return"])
    print("Dim:", result["dim"], " Popsize:", result["popsize"])

    # Save the best parameters to a .npz for later use
    if best_params is not None:
        Path("runs/basic").mkdir(parents=True, exist_ok=True)
        np.savez("runs/basic/best_linear_policy.npz", theta=best_params)
        print("Saved: runs/basic/best_linear_policy.npz")
