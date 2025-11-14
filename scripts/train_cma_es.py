from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from rlo.features import make_basic_features
from rlo.policies import ParamLinearPolicy, ParamNonLinearPolicy
from rlo.trainers import train_cma_es


def make_linear_policy() -> ParamLinearPolicy:
    return ParamLinearPolicy(n_actions=3, n_features=7)

def make_nonlinear_policy() -> ParamNonLinearPolicy:
    return ParamNonLinearPolicy(n_actions=3, n_features=7)

def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    bundle, history = train_cma_es(
        make_policy=make_nonlinear_policy,
        make_features=make_basic_features,
        generations=60,
        init_sigma=0.6,
        pop_multiplier=1.0,
        base_seed=0,
        horizon=1800,  # 30s seconds at 60 FPS
    )

    print("\n=== Training complete ===")

    # Save the training results
    save_dir = Path("runs/nonlinear")
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "history.json"
    write_json(
        history_path,
        {
            "history": [s.to_dict() for s in history],
        },
    )
    print(f"Wrote history to {history_path}")

    policy_path = save_dir / f"policy.npz"
    bundle.save(policy_path)
    print(f"Saved best policy bundle to {policy_path}")
    print("Best policy metadata:", json.dumps(bundle.metadata, indent=2))
