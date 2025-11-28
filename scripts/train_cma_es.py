from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from rlo.features import make_basic_features
from rlo.policies import ParamLinearPolicy, ParamNonLinearPolicy_ArgMax, ParamNonLinearPolicy_Stochastic
from rlo.policies.param_base import Policy
from rlo.policies.param_nonlinear_jepa import ParamNonLinearPolicy_JEPA
from rlo.trainers import train_cma_es
from enum import Enum, auto
from typing import Callable


class PolicyType(Enum):
    LINEAR = auto()
    NONLINEAR_ARGMAX = auto()
    NONLINEAR_STOCHASTIC = auto()
    NONLINEAR_JEPA = auto()

def make_policy_factory(policy_type: PolicyType, n_features: int, n_actions: int, device: str = "cpu", policy_overrides: Dict[str, object] = None) -> Callable[[], Policy]:
    """Return a zero-arg callable that constructs the requested policy."""
    match policy_type:
        case PolicyType.LINEAR:
            return lambda: ParamLinearPolicy(n_actions=n_actions, n_features=n_features)
        case PolicyType.NONLINEAR_ARGMAX:
            return lambda: ParamNonLinearPolicy_ArgMax(n_actions=n_actions, n_features=n_features)
        case PolicyType.NONLINEAR_STOCHASTIC:
            return lambda: ParamNonLinearPolicy_Stochastic(n_actions=n_actions, n_features=n_features)
        case PolicyType.NONLINEAR_JEPA:
            return lambda: ParamNonLinearPolicy_JEPA(n_actions=n_actions, n_features=n_features)
        case _:
            raise ValueError(f"Unknown policy type: {policy_type}")

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
    #print(f"Policy overrides: {overrides}")

    policy = PolicyType.NONLINEAR_JEPA
    
    bundle, history = train_cma_es(
        make_policy=make_policy_factory(policy, n_features=4, n_actions=4, device=args.device, policy_overrides=overrides),
        make_features=make_basic_features,
        generations=1 if args.test_run else 10,
        init_sigma=1,
        pop_multiplier=1.0,
        base_seed=args.seed if args.seed is not None else 0,
        horizon=7200,  # steps not seconds
    )

    print("\n=== Training complete ===")

    # Save the training results
    save_dir = Path(f"runs/{policy.name.lower()}")
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

 