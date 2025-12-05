# This will train JEPA off-policy using a random walk policy so JEPA learns a useful representation

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from rlo.features import make_basic_features
from rlo.params import ParamReader
from rlo.policies import ParamLinearPolicy_RandomWalk
from rlo.policies.param_base import Policy
from typing import Callable
from enum import Enum, auto

from rlo.trainers.jepa_trainer import train_jepa

class PolicyType(Enum):
    LINEAR_RANDOM_WALK = auto()

def make_policy(n_features: int, n_actions: int, device: str = "cpu", policy_overrides: Dict[str, object] = None) -> Callable[[], Policy]:
    return lambda: ParamLinearPolicy_RandomWalk(n_actions=n_actions, n_features=n_features)

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
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for training (cpu or cuda).",
    )

    parser.add_argument(
        "--test-run",
        action="store_true",
        default=False,
        help="Run a quick test (set to True if flag is present).",
    )

    parser.add_argument(
        "--policy-args",
        nargs="*",
        help="Optional key=value overrides for policy parameters (e.g. temperature=0.5).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Initialize ParamReader and set overrides
    reader = ParamReader.get_instance()
    
    # Load default params if available
    default_params = Path("params.yaml")
    if default_params.exists():
        reader.load(str(default_params))
    
    overrides = {}
    if args.policy_args:
        for item in args.policy_args:
            if "=" not in item:
                print(f"Warning: ignoring malformed policy arg '{item}' (expected key=value)")
                continue
            key, val_str = item.split("=", 1)
            # Simple type inference
            if val_str.lower() == "true":
                val = True
            elif val_str.lower() == "false":
                val = False
            else:
                try:
                    val = int(val_str)
                except ValueError:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
            overrides[key] = val

    if overrides:
        print(f"Policy overrides: {overrides}")
        reader.set_overrides(**overrides)

    policy = PolicyType.LINEAR_RANDOM_WALK

    save_dir = Path(f"runs/{policy.name.lower()}")


    history = train_jepa(
        make_policy=make_policy(n_features=4, n_actions=4),
        make_features=make_basic_features,
        base_seed=args.seed if args.seed is not None else 42,
        horizon=1800,  # steps not seconds
        checkpoint_dir=save_dir
    )

    print("\n=== Training complete ===")

    # Save the training results
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "history.json"
    write_json(
        history_path,
        {
            "history": [s.to_dict() for s in history],
            "best_gen": history.index(max(history, key=lambda s: s.best_reward)),
        },
    )
    print(f"Wrote history to {history_path}")

    # policy_path = save_dir / f"policy.npz"
    # bundle.save(policy_path)
    # print(f"Saved best policy bundle to {policy_path}")
    # print("Best policy metadata:", json.dumps(bundle.metadata, indent=2))

 
