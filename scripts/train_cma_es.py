from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from rlo.features import make_basic_features
from rlo.policies import ParamLinearPolicy, ParamNonLinearPolicy, ParamNonLinearPolicy_ArgMax, ParamNonLinearPolicy_Stochastic
from rlo.trainers import train_cma_es


def make_linear_policy() -> ParamLinearPolicy:
    return ParamLinearPolicy(n_actions=4, n_features=11)

def make_nonlinear_policy() -> ParamNonLinearPolicy:
    return ParamNonLinearPolicy_Stochastic(n_actions=4, n_features=4)

def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise the Endless Platformer environment and policies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic layouts.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    bundle, history = train_cma_es(
        make_policy=make_nonlinear_policy,
        make_features=make_basic_features,
        generations=20,
        init_sigma=1,
        pop_multiplier=1.0,
        base_seed=args.seed if args.seed is not None else 0,
        horizon=7200,  # steps not seconds
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
            "best_gen": bundle.metadata.get("best_gen"),
        },
    )
    print(f"Wrote history to {history_path}")

    policy_path = save_dir / f"policy.npz"
    bundle.save(policy_path)
    print(f"Saved best policy bundle to {policy_path}")
    print("Best policy metadata:", json.dumps(bundle.metadata, indent=2))

 